"""
Ingestion routes — upload PDFs or register web URLs with optional refresh schedule.
Includes dual execution engine (Celery background worker with automatic fallback
to FastAPI BackgroundTasks for standalone demo runs).
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AsyncSessionLocal, get_db
from api.models import DocumentSource, IngestionStatus, SourceType
from config.settings import RefreshUnit, settings
from ingestion.indexer import HybridIndexer
from ingestion.parsers.router import detect_source_type, get_parser, get_supported_extensions
from ingestion.pipeline import IngestionPipeline
from ingestion.scheduler import refresh_scheduler

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Dual Execution Dispatcher ────────────────────────────────────────────────
async def _execute_in_process_ingestion(source_id: str, tenant_id: str) -> None:
    """Fallback execution directly inside the FastAPI process."""
    logger.info(f"Running in-process ingestion for source_id={source_id}")
    try:
        async with AsyncSessionLocal() as session:
            pipeline = IngestionPipeline(session=session)
            await pipeline.ingest_source(source_id=source_id, tenant_id=tenant_id)
    except Exception as e:
        logger.error(f"In-process ingestion failed for source_id={source_id}: {e}")


def dispatch_ingestion_task(
    source_id: str,
    tenant_id: str,
    background_tasks: BackgroundTasks,
) -> str:
    """
    Attempts to dispatch via Celery+Redis; if Redis is down or Celery fails,
    gracefully falls back to FastAPI's in-process BackgroundTasks.
    """
    try:
        from api.tasks import run_ingestion_task
        run_ingestion_task.delay(source_id=source_id, tenant_id=tenant_id)
        logger.info(f"Dispatched ingestion task {source_id} via Celery worker.")
        return "celery"
    except Exception as e:
        logger.warning(
            f"Celery/Redis unavailable ({e}). Falling back to in-process BackgroundTasks."
        )
        background_tasks.add_task(_execute_in_process_ingestion, source_id, tenant_id)
        return "background_tasks"


# ── 1. Unified Polymorphic File Upload (PDF, DOCX, TXT, MD, etc.) ─────────────
@router.post("/file")
@router.post("/pdf")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Form(default=settings.default_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Unified, polymorphic file ingestion endpoint.
    Dynamically resolves parser strategy at runtime based on file type.
    Supports PDF, DOCX, TXT, MD, CSV, JSON and any registered format.
    """
    filename = file.filename or "uploaded_file"
    dummy_path = Path(filename)

    # 1. Validate that a parser exists for this file format
    try:
        get_parser(mime_type=file.content_type or "", file_path=dummy_path)
    except ValueError as e:
        supported = ", ".join(get_supported_extensions())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{dummy_path.suffix}'. Supported extensions: [{supported}]",
        )

    # 2. Automatically resolve source_type
    detected_type = detect_source_type(file_path=dummy_path, mime_type=file.content_type or "")

    source_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{source_id}_{filename}"

    contents = await file.read()
    file_path.write_bytes(contents)

    source = DocumentSource(
        id=source_id,
        tenant_id=tenant_id,
        source_type=detected_type,
        name=filename,
        object_storage_path=str(file_path),
        file_size_bytes=len(contents),
        mime_type=file.content_type or "application/octet-stream",
        ingestion_status=IngestionStatus.PENDING,
        refresh_enabled=False,
    )
    db.add(source)
    await db.commit()

    dispatcher = dispatch_ingestion_task(source_id, tenant_id, background_tasks)

    return {
        "source_id": source_id,
        "status": "queued",
        "filename": filename,
        "detected_type": detected_type,
        "execution_mode": dispatcher,
    }


# ── 2. Unified Polymorphic Batch File Upload ─────────────────────────────────
@router.post("/files")
@router.post("/batch")
@router.post("/batch-pdfs")
async def ingest_batch_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    tenant_id: str = Form(default=settings.default_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Unified batch upload endpoint for N files of mixed types (PDF, DOCX, TXT, MD...).
    """
    results = []

    for file in files:
        filename = file.filename or "uploaded_file"
        dummy_path = Path(filename)

        try:
            get_parser(mime_type=file.content_type or "", file_path=dummy_path)
            detected_type = detect_source_type(file_path=dummy_path, mime_type=file.content_type or "")
        except ValueError:
            results.append({
                "filename": filename,
                "status": "rejected",
                "error": f"Unsupported extension '{dummy_path.suffix}'",
            })
            continue

        source_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{source_id}_{filename}"

        contents = await file.read()
        file_path.write_bytes(contents)

        source = DocumentSource(
            id=source_id,
            tenant_id=tenant_id,
            source_type=detected_type,
            name=filename,
            object_storage_path=str(file_path),
            file_size_bytes=len(contents),
            mime_type=file.content_type or "application/octet-stream",
            ingestion_status=IngestionStatus.PENDING,
            refresh_enabled=False,
        )
        db.add(source)
        await db.commit()

        dispatcher = dispatch_ingestion_task(source_id, tenant_id, background_tasks)

        results.append({
            "source_id": source_id,
            "filename": filename,
            "detected_type": detected_type,
            "status": "queued",
            "execution_mode": dispatcher,
        })

    return {
        "message": f"Processed {len(files)} file(s).",
        "queued_count": len([r for r in results if r.get("status") == "queued"]),
        "files": results,
    }


# ── 3. Web URL Registration (With Configurable Refresh) ─────────────────────
class WebURLRequest(BaseModel):
    url: str
    name: Optional[str] = None
    tenant_id: str = settings.default_tenant_id
    # Refresh schedule
    refresh_enabled: bool = False
    refresh_value: int = settings.refresh_default_value
    refresh_interval: Optional[int] = None
    refresh_unit: RefreshUnit = settings.refresh_default_unit

    @field_validator("refresh_unit", mode="before")
    @classmethod
    def normalize_unit(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean.endswith("s"):
                v_clean = v_clean[:-1]
            try:
                return RefreshUnit(v_clean)
            except ValueError:
                return RefreshUnit.HOUR
        return v

    def resolve_refresh_value(self) -> int:
        return self.refresh_interval if self.refresh_interval is not None else self.refresh_value


@router.post("/url")
async def ingest_url(
    request: WebURLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Register a web URL for ingestion, with optional configurable refresh schedule."""
    source_id = str(uuid.uuid4())
    resolved_refresh_value = request.resolve_refresh_value()

    # If refresh_interval or refresh_value was explicitly set by client, enable refresh
    is_refresh_enabled = request.refresh_enabled or (request.refresh_interval is not None)

    source = DocumentSource(
        id=source_id,
        tenant_id=request.tenant_id,
        source_type=SourceType.WEB,
        name=request.name or request.url,
        url=request.url,
        ingestion_status=IngestionStatus.PENDING,
        refresh_enabled=is_refresh_enabled,
        refresh_value=resolved_refresh_value if is_refresh_enabled else None,
        refresh_unit=request.refresh_unit.value if is_refresh_enabled else None,
    )
    db.add(source)
    await db.commit()

    # Schedule refresh if enabled
    if is_refresh_enabled:
        next_run = refresh_scheduler.add_or_update(
            source_id=source_id,
            tenant_id=request.tenant_id,
            refresh_value=resolved_refresh_value,
            refresh_unit=request.refresh_unit,
        )
        if next_run:
            source.next_refresh_at = next_run.replace(tzinfo=None) if next_run.tzinfo else next_run
        await db.commit()

    dispatcher = dispatch_ingestion_task(source_id, request.tenant_id, background_tasks)

    return {
        "source_id": source_id,
        "status": "queued",
        "url": request.url,
        "refresh_enabled": is_refresh_enabled,
        "refresh_schedule": (
            f"every {resolved_refresh_value} {request.refresh_unit.value}(s)"
            if is_refresh_enabled
            else None
        ),
        "execution_mode": dispatcher,
    }


# ── 4. Update Refresh Schedule ──────────────────────────────────────────────
class RefreshUpdateRequest(BaseModel):
    refresh_enabled: bool
    refresh_value: int = settings.refresh_default_value
    refresh_unit: RefreshUnit = settings.refresh_default_unit

    @field_validator("refresh_unit", mode="before")
    @classmethod
    def normalize_unit(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean.endswith("s"):
                v_clean = v_clean[:-1]
            try:
                return RefreshUnit(v_clean)
            except ValueError:
                return RefreshUnit.HOUR
        return v


@router.patch("/url/{source_id}/refresh")
async def update_refresh_schedule(
    source_id: str,
    request: RefreshUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update or toggle the refresh schedule for a web source."""
    result = await db.execute(select(DocumentSource).where(DocumentSource.id == source_id))
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if source.source_type != SourceType.WEB:
        raise HTTPException(status_code=400, detail="Refresh scheduling is only for web sources")

    source.refresh_enabled = request.refresh_enabled
    if request.refresh_enabled:
        source.refresh_value = request.refresh_value
        source.refresh_unit = request.refresh_unit.value
        next_run = refresh_scheduler.add_or_update(
            source_id=source_id,
            tenant_id=source.tenant_id,
            refresh_value=request.refresh_value,
            refresh_unit=request.refresh_unit,
        )
        if next_run:
            source.next_refresh_at = next_run.replace(tzinfo=None) if next_run.tzinfo else next_run
        message = f"Refresh scheduled every {request.refresh_value} {request.refresh_unit.value}(s)"
    else:
        refresh_scheduler.remove(source_id)
        source.next_refresh_at = None
        message = "Refresh schedule disabled"

    await db.commit()
    return {"source_id": source_id, "message": message}


# ── 5. List Sources ─────────────────────────────────────────────────────────
@router.get("/sources")
async def list_sources(
    tenant_id: str = settings.default_tenant_id,
    db: AsyncSession = Depends(get_db),
):
    """List all registered document and web sources."""
    result = await db.execute(
        select(DocumentSource).where(DocumentSource.tenant_id == tenant_id)
    )
    sources = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "source_type": s.source_type,
            "url": s.url,
            "status": s.ingestion_status,
            "ingestion_status": s.ingestion_status,
            "chunk_count": s.chunk_count or 0,
            "error": s.ingestion_error,
            "ingestion_error": s.ingestion_error,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_ingested_at": s.last_ingested_at.isoformat() if s.last_ingested_at else None,
            "refresh_enabled": s.refresh_enabled,
            "refresh_interval": s.refresh_value,
            "refresh_value": s.refresh_value,
            "refresh_unit": s.refresh_unit,
            "refresh_schedule": (
                f"every {s.refresh_value} {s.refresh_unit}(s)" if s.refresh_enabled else "Manual"
            ),
            "next_refresh_at": s.next_refresh_at.isoformat() if s.next_refresh_at else None,
        }
        for s in sources
    ]


# ── 6. Get Source Ingestion Status & Metrics ────────────────────────────────
@router.get("/sources/{source_id}/status")
async def get_source_status(
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll ingestion state, chunk count, and any error messages for a source."""
    result = await db.execute(select(DocumentSource).where(DocumentSource.id == source_id))
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    return {
        "id": source.id,
        "name": source.name,
        "source_type": source.source_type,
        "status": source.ingestion_status,
        "chunk_count": source.chunk_count,
        "error": source.ingestion_error,
        "last_ingested_at": (
            source.last_ingested_at.isoformat() if source.last_ingested_at else None
        ),
        "next_refresh_at": (
            source.next_refresh_at.isoformat() if source.next_refresh_at else None
        ),
    }


# ── 7. Delete Source (Purge DB record + Qdrant vectors + Scheduler) ─────────
@router.delete("/sources/{source_id}")
async def delete_source(
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Purge a source from PostgreSQL, delete its vectors from Qdrant, and cancel refresh."""
    result = await db.execute(select(DocumentSource).where(DocumentSource.id == source_id))
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    # 1. Remove from APScheduler if it had a refresh schedule
    if source.refresh_enabled:
        refresh_scheduler.remove(source_id)

    # 2. Delete vectors from Qdrant
    try:
        indexer = HybridIndexer()
        await indexer.delete_source(source_id=source_id, tenant_id=source.tenant_id)
        logger.info(f"Deleted vectors from Qdrant for source_id={source_id}")
    except Exception as e:
        logger.warning(f"Could not purge vectors from Qdrant for source_id={source_id}: {e}")

    # 3. Delete uploaded file if stored locally
    if source.object_storage_path:
        try:
            local_file = Path(source.object_storage_path)
            if local_file.exists():
                local_file.unlink()
        except Exception:
            pass

    # 4. Delete record from PostgreSQL
    await db.delete(source)
    await db.commit()

    return {"message": "Source and associated vectors deleted successfully", "id": source_id}


# ── 8. Manual Re-Ingest Trigger ─────────────────────────────────────────────
@router.post("/sources/{source_id}/reingest")
@router.post("/reingest/{source_id}")
async def trigger_reingest(
    source_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Manually re-trigger full ingestion of an existing source."""
    result = await db.execute(select(DocumentSource).where(DocumentSource.id == source_id))
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    dispatcher = dispatch_ingestion_task(source_id, source.tenant_id, background_tasks)

    return {
        "message": f"Re-ingestion triggered for '{source.name}'",
        "source_id": source_id,
        "execution_mode": dispatcher,
    }
