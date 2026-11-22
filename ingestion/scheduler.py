"""
Configurable Web Refresh Scheduler.

Users define per-source refresh schedules (value + unit: minute/hour/day/week).
APScheduler manages the cron jobs; a new schedule is added/updated/removed
dynamically when the user changes settings via the Admin UI.

This module exposes:
  - RefreshScheduler  : lifecycle manager (start / stop / add / remove)
  - schedule_web_source()  : convenience helper called from the API
"""

from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from config.settings import RefreshUnit

UNIT_TO_SECONDS: dict[RefreshUnit, int] = {
    RefreshUnit.MINUTE: 60,
    RefreshUnit.HOUR: 3600,
    RefreshUnit.DAY: 86400,
    RefreshUnit.WEEK: 604800,
}


class RefreshScheduler:
    """
    Wraps APScheduler. Each registered web source gets its own job keyed
    by `source_id`. On restart, jobs are restored from the database.
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self):
        self._scheduler.start()
        logger.info("RefreshScheduler started")

    def stop(self):
        self._scheduler.shutdown(wait=False)
        logger.info("RefreshScheduler stopped")

    def add_or_update(
        self,
        source_id: str,
        tenant_id: str,
        refresh_value: int,
        refresh_unit: RefreshUnit,
    ) -> datetime:
        """
        Register or update a refresh job for a web source.
        Returns the next scheduled run datetime.
        """
        seconds = UNIT_TO_SECONDS[refresh_unit] * refresh_value
        job_id = f"refresh_{source_id}"

        trigger = IntervalTrigger(seconds=seconds)

        if self._scheduler.get_job(job_id):
            self._scheduler.reschedule_job(job_id, trigger=trigger)
            logger.info(f"Rescheduled refresh for source {source_id}: every {refresh_value} {refresh_unit.value}(s)")
        else:
            self._scheduler.add_job(
                func=_run_refresh,
                trigger=trigger,
                id=job_id,
                kwargs={"source_id": source_id, "tenant_id": tenant_id},
                replace_existing=True,
                misfire_grace_time=300,
            )
            logger.info(f"Scheduled refresh for source {source_id}: every {refresh_value} {refresh_unit.value}(s)")

        job = self._scheduler.get_job(job_id)
        return job.next_run_time

    def remove(self, source_id: str) -> None:
        job_id = f"refresh_{source_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info(f"Removed refresh schedule for source {source_id}")

    def list_jobs(self) -> list[dict]:
        return [
            {
                "job_id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in self._scheduler.get_jobs()
        ]


# ── Singleton ────────────────────────────────────────────────────────────────
refresh_scheduler = RefreshScheduler()


# ── Refresh execution function (called by APScheduler) ───────────────────────
async def _run_refresh(source_id: str, tenant_id: str) -> None:
    """
    Re-ingest a web source:
      1. Delete existing chunks from vector store
      2. Re-crawl the URL
      3. Re-chunk and re-index
      4. Update last_ingested_at and next_refresh_at in DB
    """
    logger.info(f"Running scheduled refresh for source_id={source_id}")
    try:
        # Import here to avoid circular imports
        from api.database import AsyncSessionLocal
        from ingestion.pipeline import IngestionPipeline

        async with AsyncSessionLocal() as session:
            pipeline = IngestionPipeline(session=session)
            await pipeline.reingest_source(source_id=source_id, tenant_id=tenant_id)

        logger.info(f"Refresh complete for source_id={source_id}")
    except Exception as e:
        logger.error(f"Refresh failed for source_id={source_id}: {e}")
