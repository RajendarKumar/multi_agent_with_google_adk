"""
API route integration tests for Ingestion endpoints using FastAPI TestClient.
Database sessions are mocked so unit tests pass even when PostgreSQL is offline.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_db
from api.main import app


@pytest.fixture
def client():
    # Mock database session
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result
    mock_db.commit.return_value = None

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "scheduled_refresh_jobs" in data

    # Verify /api/health alias also works
    api_resp = client.get("/api/health")
    assert api_resp.status_code == 200
    assert api_resp.json()["status"] == "ok"


def test_list_sources_endpoint(client):
    response = client.get("/api/ingest/sources")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_web_url_request_validation(client):
    # Invalid request body (missing required 'url' field)
    response = client.post("/api/ingest/url", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_pipeline_flags_failure_on_empty_content():
    """Verify that IngestionPipeline sets status to FAILED and records error when content is empty."""
    from ingestion.pipeline import IngestionPipeline
    from api.models import DocumentSource, IngestionStatus

    mock_source = DocumentSource(
        id="src-fail-01",
        tenant_id="default",
        source_type="web",
        name="Empty Web Page",
        url="https://broken-url.com",
        ingestion_status=IngestionStatus.PENDING,
    )

    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = mock_source
    mock_db.execute.return_value = mock_res
    mock_db.commit = AsyncMock()

    pipeline = IngestionPipeline(mock_db)

    # Mock parser to return empty document
    mock_parser = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.markdown_content = ""
    mock_parser.parse.return_value = mock_doc

    with patch("ingestion.pipeline.get_parser", return_value=mock_parser):
        with pytest.raises(ValueError, match="empty content"):
            await pipeline.ingest_source("src-fail-01", "default")

    assert mock_source.ingestion_status == IngestionStatus.FAILED
    assert "empty content" in mock_source.ingestion_error.lower()


def test_reingest_endpoints_routing(client):
    """Verify both /api/ingest/sources/{id}/reingest and /api/ingest/reingest/{id} are routed to reingest handler."""
    with patch("api.routes.ingest.dispatch_ingestion_task", return_value="in_process"):
        # Both endpoints should route to trigger_reingest and return 200
        resp1 = client.post("/api/ingest/sources/test-src-123/reingest")
        assert resp1.status_code == 200
        assert resp1.json()["source_id"] == "test-src-123"

        resp2 = client.post("/api/ingest/reingest/test-src-123")
        assert resp2.status_code == 200
        assert resp2.json()["source_id"] == "test-src-123"


