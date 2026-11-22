"""Health check route."""
from fastapi import APIRouter

from ingestion.scheduler import refresh_scheduler

router = APIRouter()


@router.get("/health")
@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "scheduled_refresh_jobs": len(refresh_scheduler.list_jobs()),
    }
