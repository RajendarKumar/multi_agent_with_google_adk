"""
Tests for DOCX parser, Text parser, and the unified polymorphic file ingestion API.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.database import get_db
from api.main import app
from ingestion.parsers.docx_parser import DocxParser
from ingestion.parsers.router import detect_source_type, get_supported_extensions
from ingestion.parsers.text_parser import TextParser


@pytest.fixture
def client():
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


@pytest.mark.asyncio
async def test_docx_parser_real_document():
    """Verify DocxParser can parse docs/responsive agent.docx directly."""
    docx_path = Path("docs/responsive_agent.docx")
    if not docx_path.exists():
        docx_path = Path("docs/responsive agent.docx")
    assert docx_path.exists(), "Test docx file not found"

    parser = DocxParser()
    doc = await parser.parse(
        source_id="test-docx-1",
        tenant_id="default",
        file_path=docx_path,
    )

    assert doc.source_id == "test-docx-1"
    assert doc.source_type == "docx"
    assert len(doc.markdown_content) > 100
    assert "Requirements" in doc.markdown_content or "Agent" in doc.markdown_content


@pytest.mark.asyncio
async def test_text_parser(tmp_path):
    txt_file = tmp_path / "notes.md"
    txt_file.write_text("# Release Notes\n\nVersion 1.0.0 released with multi-agent RAG.")

    parser = TextParser()
    doc = await parser.parse(
        source_id="test-txt-1",
        tenant_id="default",
        file_path=txt_file,
    )

    assert doc.source_type == "markdown"
    assert "Release Notes" in doc.markdown_content


def test_supported_extensions_discovery():
    exts = get_supported_extensions()
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".txt" in exts
    assert ".md" in exts


def test_detect_source_type():
    assert detect_source_type(Path("doc.pdf")) == "pdf"
    assert detect_source_type(Path("doc.docx")) == "docx"
    assert detect_source_type(Path("doc.md")) == "markdown"
    assert detect_source_type(Path("doc.txt")) == "text"
    assert detect_source_type(url="https://google.com") == "web"


def test_polymorphic_file_upload_api(client):
    """Test unified /api/ingest/file endpoint with a text/markdown file."""
    files = {"file": ("notes.md", b"# Architecture Notes\nTesting upload.", "text/markdown")}
    response = client.post("/api/ingest/file", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["detected_type"] == "markdown"
    assert data["filename"] == "notes.md"


def test_polymorphic_file_upload_rejects_unsupported(client):
    """Test that an unsupported extension (.xyz) is cleanly rejected with 400."""
    files = {"file": ("bad.xyz", b"binary content", "application/octet-stream")}
    response = client.post("/api/ingest/file", files=files)

    assert response.status_code == 400
    assert "Unsupported format '.xyz'" in response.json()["detail"]
