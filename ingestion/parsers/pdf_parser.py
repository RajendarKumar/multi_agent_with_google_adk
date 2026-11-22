"""
PDF parser using Docling — layout-aware, table-preserving, page-numbered.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.parsers.base import BaseDocumentParser, CanonicalDocument, SourceType


class PDFParser(BaseDocumentParser):
    supported_mime_types = ["application/pdf"]
    supported_extensions = [".pdf"]

    async def parse(
        self,
        source_id: str,
        tenant_id: str,
        file_path: Optional[Path] = None,
        url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> CanonicalDocument:
        if file_path is None:
            raise ValueError("PDFParser requires a file_path")

        logger.info(f"Parsing PDF: {file_path}")

        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            markdown_content = result.document.export_to_markdown()

            # Extract per-page text for page-level citation
            page_texts: list[str] = []
            if hasattr(result.document, "pages"):
                for page in result.document.pages.values():
                    page_texts.append(page.export_to_markdown() if hasattr(page, "export_to_markdown") else "")

        except ImportError:
            logger.warning("Docling not installed — falling back to basic text extraction")
            markdown_content = self._fallback_pdf_parse(file_path)
            page_texts = []

        return CanonicalDocument(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=SourceType.PDF,
            source_name=file_path.name,
            source_uri=str(file_path),
            markdown_content=markdown_content,
            page_texts=page_texts,
            total_pages=len(page_texts) or None,
            extra_metadata=extra_metadata or {},
        )

    def _fallback_pdf_parse(self, file_path: Path) -> str:
        """Minimal fallback using PyPDF2 if docling is unavailable."""
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text_parts.append(f"\n\n## Page {i + 1}\n\n{page.extract_text()}")
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Fallback PDF parse failed: {e}")
            return ""
