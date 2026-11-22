"""
Plain text, Markdown, and tabular text parser.
Handles .txt, .md, .markdown, .json, .csv files.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.parsers.base import BaseDocumentParser, CanonicalDocument, SourceType


class TextParser(BaseDocumentParser):
    supported_mime_types = [
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
    ]
    supported_extensions = [".txt", ".md", ".markdown", ".json", ".csv"]

    async def parse(
        self,
        source_id: str,
        tenant_id: str,
        file_path: Optional[Path] = None,
        url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> CanonicalDocument:
        if file_path is None:
            raise ValueError("TextParser requires a file_path")

        logger.info(f"Parsing Text/Markdown document: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1
            content = file_path.read_text(encoding="latin-1")

        ext = file_path.suffix.lower()
        if ext in (".md", ".markdown"):
            source_type = SourceType.MARKDOWN
        elif ext == ".csv":
            source_type = SourceType.CSV
        else:
            source_type = SourceType.MARKDOWN  # treated as markdown for structure chunking

        return CanonicalDocument(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=source_type,
            source_name=file_path.name,
            source_uri=str(file_path),
            markdown_content=content,
            page_texts=[content],
            total_pages=1,
            extra_metadata=extra_metadata or {},
        )
