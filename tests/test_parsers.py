"""
Unit tests for document parsers (PDF, Web, and Router).
"""

from pathlib import Path

import pytest

from ingestion.parsers.base import SourceType
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.router import get_parser
from ingestion.parsers.web_parser import WebParser


def test_pdf_parser_can_handle():
    parser = PDFParser()
    assert parser.can_handle(extension=".pdf") is True
    assert parser.can_handle(mime_type="application/pdf") is True
    assert parser.can_handle(extension=".docx") is False


def test_web_parser_can_handle():
    parser = WebParser()
    assert parser.can_handle(extension=".html") is True
    assert parser.can_handle(mime_type="text/html") is True
    assert parser.can_handle(extension=".pdf") is False


def test_router_resolution():
    pdf_parser = get_parser(file_path=Path("sample.pdf"))
    assert isinstance(pdf_parser, PDFParser)

    web_parser = get_parser(url="https://example.com")
    assert isinstance(web_parser, WebParser)

    with pytest.raises(ValueError, match="Unsupported file format"):
        get_parser(file_path=Path("unsupported.xyz"))


@pytest.mark.asyncio
async def test_web_parser_fallback(monkeypatch):
    """Test web parser extraction using BeautifulSoup fallback."""
    parser = WebParser()

    mock_html = """
    <html>
        <head><title>Test Documentation Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a paragraph about our AI system.</p>
        </body>
    </html>
    """

    class MockResponse:
        text = mock_html
        def raise_for_status(self):
            pass

    async def mock_get(self, url, headers):
        return MockResponse()

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    doc = await parser.parse(
        source_id="src-123",
        tenant_id="default",
        url="https://example.com/docs",
    )

    assert doc.source_id == "src-123"
    assert doc.source_type == SourceType.WEB
    assert "Test Documentation Page" in doc.source_name
    assert "This is a paragraph about our AI system." in doc.markdown_content


@pytest.mark.asyncio
async def test_web_parser_heading_preservation(monkeypatch):
    """Test that WebParser converts HTML headings to Markdown headings and preserves section headings in chunks."""
    from ingestion.chunker import SemanticChunker

    parser = WebParser()

    mock_html = """
    <html>
        <head><title>Cricket Match Live Score</title></head>
        <body>
            <h1>1st Test: India vs New Zealand</h1>
            <p>Welcome to Basin Reserve, Wellington.</p>
            <h2>1st Innings - New Zealand</h2>
            <p>Henry Nicholls scored 175 runs.</p>
            <h3>Bowling Analysis</h3>
            <p>Bumrah took 3 wickets.</p>
        </body>
    </html>
    """

    class MockResponse:
        text = mock_html
        def raise_for_status(self):
            pass

    async def mock_get(self, url, headers):
        return MockResponse()

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    doc = await parser.parse(
        source_id="src-cric-1",
        tenant_id="default",
        url="https://espncricinfo.com/match/123",
    )

    assert doc.source_id == "src-cric-1"
    assert doc.source_type == SourceType.WEB
    assert "# 1st Test: India vs New Zealand" in doc.markdown_content
    assert "## 1st Innings - New Zealand" in doc.markdown_content
    assert "### Bowling Analysis" in doc.markdown_content

    from ingestion.chunker import ChunkerConfig, SemanticChunker

    chunker = SemanticChunker(ChunkerConfig(min_chunk_size=10))
    chunks = chunker.chunk(doc)
    child_chunks = [c for c in chunks if not c.extra_metadata.get("is_parent")]
    headings = [c.section_heading for c in child_chunks]

    assert any("1st Test: India vs New Zealand" in h for h in headings)
    assert any("1st Innings - New Zealand" in h for h in headings)
    assert all(h != "" for h in headings)
