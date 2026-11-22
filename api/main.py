"""
FastAPI application entry point.
Provides:
  - /api/chat       — streaming chat with SSE
  - /api/ingest     — file / URL ingestion
  - /api/admin      — MCP server registry + agent config management
  - /health         — health check
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.middleware import TenantMiddleware
from api.routes import admin, chat, health, ingest
from ingestion.scheduler import refresh_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Responsive Agent API...")
    try:
        from api.database import init_db
        await init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not auto-initialize DB tables on startup (PostgreSQL may be offline): {e}")

    refresh_scheduler.start()

    # Restore scheduled web source jobs from DB on startup
    await _restore_refresh_schedules()

    yield

    # Shutdown
    refresh_scheduler.stop()
    logger.info("Responsive Agent API shut down.")


async def _restore_refresh_schedules():
    """Re-register refresh jobs for all web sources with refresh_enabled=True."""
    try:
        from sqlalchemy import select

        from api.database import AsyncSessionLocal
        from api.models import DocumentSource
        from config.settings import RefreshUnit

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DocumentSource).where(DocumentSource.refresh_enabled == True)
            )
            sources = result.scalars().all()

        for source in sources:
            if source.refresh_value and source.refresh_unit:
                refresh_scheduler.add_or_update(
                    source_id=source.id,
                    tenant_id=source.tenant_id,
                    refresh_value=source.refresh_value,
                    refresh_unit=RefreshUnit(source.refresh_unit),
                )
        logger.info(f"Restored {len(sources)} web refresh schedules")
    except Exception as e:
        logger.warning(f"Could not restore refresh schedules: {e}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Responsive Agent Platform",
        description="Configurable multi-agent RAG + MCP platform built on Google ADK",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TenantMiddleware)

    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(ingest.router, prefix="/api/ingest", tags=["Ingestion"])
    app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
    # Direct alias so /mcp/servers is reachable directly alongside /api/admin/mcp/servers
    app.include_router(admin.router, tags=["Admin (Direct)"], include_in_schema=False)

    # Serve interactive Web UI (Chat, Ingestion, MCP, Config, Traces)
    ui_path = Path(__file__).parent.parent / "ui"
    if ui_path.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/ui/")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
