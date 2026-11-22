"""
Unit tests for the structure-aware SemanticChunker.
Tests heading boundary splits, table preservation, and parent-child linking.
"""

from datetime import datetime

import pytest

from ingestion.chunker import ChunkerConfig, SemanticChunker
from ingestion.parsers.base import CanonicalDocument, SourceType


@pytest.fixture
def sample_document():
    markdown = """# Architecture Overview

This section introduces the responsive agent platform and its modular architecture.

## Ingestion Pipeline

The ingestion pipeline handles documents and web links seamlessly.
It preserves markdown tables and extracts clean structure.

| Format | Parser | Status |
|---|---|---|
| PDF | Docling | Active |
| Web | Crawl4AI | Active |

## Query Engine

Queries are processed via Google ADK orchestration and hybrid search.
"""
    return CanonicalDocument(
        source_id="doc-test-1",
        tenant_id="default",
        source_type=SourceType.PDF,
        source_name="Architecture_Doc.pdf",
        source_uri="/path/to/Architecture_Doc.pdf",
        markdown_content=markdown,
        parsed_at=datetime.utcnow(),
    )


def test_chunker_preserves_sections(sample_document):
    chunker = SemanticChunker(ChunkerConfig(chunk_size=200, chunk_overlap=20))
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 0

    # Ensure all chunks inherit correct document metadata
    for c in chunks:
        assert c.source_id == "doc-test-1"
        assert c.tenant_id == "default"
        assert c.source_name == "Architecture_Doc.pdf"
        assert len(c.text) > 0


def test_table_preservation(sample_document):
    """Ensure the markdown table is not broken mid-row."""
    chunker = SemanticChunker(ChunkerConfig(chunk_size=150, chunk_overlap=10))
    chunks = chunker.chunk(sample_document)

    table_chunks = [c for c in chunks if "| Format | Parser |" in c.text]
    assert len(table_chunks) >= 1

    # Check that both rows are present in the table chunk
    table_text = table_chunks[0].text
    assert "| PDF | Docling | Active |" in table_text
    assert "| Web | Crawl4AI | Active |" in table_text


def test_parent_child_chunk_linking(sample_document):
    """Ensure child chunks contain a parent_chunk_id when parent chunk is generated."""
    chunker = SemanticChunker(ChunkerConfig(chunk_size=100, parent_chunk_size=1000))
    chunks = chunker.chunk(sample_document)

    # Filter out parent chunks
    child_chunks = [c for c in chunks if not c.extra_metadata.get("is_parent")]
    parent_chunks = [c for c in chunks if c.extra_metadata.get("is_parent")]

    if len(child_chunks) > 1 and parent_chunks:
        parent_id = parent_chunks[0].chunk_id
        matching_children = [c for c in child_chunks if c.parent_chunk_id == parent_id]
        assert len(matching_children) > 0


def test_unheaded_document_chunk_has_fallback_heading():
    """Ensure documents without markdown headings fallback to source_name for section_heading."""
    doc = CanonicalDocument(
        source_id="doc-web-flat",
        tenant_id="default",
        source_type=SourceType.WEB,
        source_name="Basin Reserve Live Updates",
        source_uri="https://example.com/live",
        markdown_content="This is plain text with no headings at all across all paragraphs.",
        parsed_at=datetime.utcnow(),
    )
    chunker = SemanticChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    for c in chunks:
        assert c.section_heading == "Basin Reserve Live Updates"
