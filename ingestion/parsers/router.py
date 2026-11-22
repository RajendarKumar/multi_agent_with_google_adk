"""
Parser Router — maps MIME type / file extension to the right parser.
To add a new format: instantiate your parser below and add it to REGISTRY.
Downstream pipeline and REST API automatically discover and support it.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.parsers.base import BaseDocumentParser
from ingestion.parsers.docx_parser import DocxParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.text_parser import TextParser
from ingestion.parsers.web_parser import WebParser

# ── Dynamic Registry ────────────────────────────────────────────────────────
REGISTRY: list[BaseDocumentParser] = [
    PDFParser(),
    DocxParser(),
    TextParser(),
    WebParser(),
]


def get_supported_extensions() -> list[str]:
    """Returns a flat list of all file extensions currently handled by registered parsers."""
    exts = set()
    for parser in REGISTRY:
        exts.update(parser.supported_extensions)
    return sorted(list(exts))


def detect_source_type(file_path: Optional[Path] = None, mime_type: str = "", url: Optional[str] = None) -> str:
    """Infers the source_type identifier (e.g. 'pdf', 'docx', 'web', 'markdown') at runtime."""
    if url:
        return "web"
    ext = file_path.suffix.lower() if file_path else ""
    if ext == ".pdf" or "pdf" in mime_type:
        return "pdf"
    elif ext in (".docx", ".doc") or "word" in mime_type:
        return "docx"
    elif ext in (".md", ".markdown"):
        return "markdown"
    elif ext == ".csv":
        return "csv"
    elif ext in (".txt", ".json"):
        return "text"
    return ext.replace(".", "") or "unknown"


def get_parser(
    mime_type: str = "",
    file_path: Optional[Path] = None,
    url: Optional[str] = None,
) -> BaseDocumentParser:
    """
    Return the first registered parser that can handle the given source.
    Falls back to WebParser for URLs and raises for unknown file types.
    """
    extension = file_path.suffix if file_path else ""

    for parser in REGISTRY:
        if parser.can_handle(mime_type=mime_type, extension=extension):
            return parser

    if url:
        logger.warning(f"No specific parser for mime={mime_type}, falling back to WebParser for {url}")
        return WebParser()

    supported = ", ".join(get_supported_extensions())
    raise ValueError(
        f"Unsupported file format: extension='{extension}', mime='{mime_type}'. "
        f"Currently supported formats: [{supported}]"
    )
