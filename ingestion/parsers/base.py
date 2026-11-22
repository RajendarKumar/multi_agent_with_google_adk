"""
Pluggable document parser interface.
Every parser emits a canonical CanonicalDocument — the single contract
between ingestion and indexing layers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SourceType(str, Enum):
    PDF = "pdf"
    WEB = "web"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class DocumentChunk:
    """A single indexable unit with full provenance metadata."""
    chunk_id: str
    source_id: str                  # FK to DocumentSource.id
    tenant_id: str
    text: str                       # clean Markdown text
    source_type: SourceType
    source_name: str                # filename or page title
    source_uri: str                 # file path or URL
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None   # for parent-child indexing
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CanonicalDocument:
    """Full parsed document before chunking."""
    source_id: str
    tenant_id: str
    source_type: SourceType
    source_name: str
    source_uri: str
    markdown_content: str           # full document as Markdown
    page_texts: list[str] = field(default_factory=list)   # per-page text
    total_pages: Optional[int] = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    parsed_at: datetime = field(default_factory=datetime.utcnow)


class BaseDocumentParser(ABC):
    """
    Strategy interface. Add a new parser:
      1. Subclass BaseDocumentParser
      2. Declare supported_mime_types / supported_extensions
      3. Register in ParserRouter (parsers/router.py)
    No other code changes required.
    """

    supported_mime_types: list[str] = []
    supported_extensions: list[str] = []

    @abstractmethod
    async def parse(
        self,
        source_id: str,
        tenant_id: str,
        file_path: Optional[Path] = None,
        url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> CanonicalDocument:
        """Parse source into a CanonicalDocument."""
        ...

    def can_handle(self, mime_type: str = "", extension: str = "") -> bool:
        return (
            mime_type in self.supported_mime_types
            or extension.lower() in self.supported_extensions
        )
