"""
Celery task definitions for background ingestion jobs.
"""

from celery import Celery
from loguru import logger

from config.settings import settings

celery_app = Celery("responsive_agent", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"


@celery_app.task(name="run_ingestion_task", bind=True, max_retries=3)
def run_ingestion_task(self, source_id: str, tenant_id: str):
    """
    Background task: parse → chunk → index a document source.
    Runs in a separate worker process.
    """
    import asyncio

    async def _ingest():
        from api.database import AsyncSessionLocal
        from ingestion.pipeline import IngestionPipeline
        async with AsyncSessionLocal() as session:
            pipeline = IngestionPipeline(session=session)
            await pipeline.ingest_source(source_id=source_id, tenant_id=tenant_id)

    try:
        asyncio.run(_ingest())
    except Exception as exc:
        logger.error(f"Ingestion task failed for source {source_id}: {exc}")
        raise self.retry(exc=exc, countdown=60)
