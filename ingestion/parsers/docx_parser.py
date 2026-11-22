"""
Layout-aware Word Document (.docx) parser.
Extracts paragraphs and tables, converting them to clean Markdown.
Supports both python-docx (if installed) and pure Python zipfile/XML parsing.
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.parsers.base import BaseDocumentParser, CanonicalDocument, SourceType


class DocxParser(BaseDocumentParser):
    supported_mime_types = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]
    supported_extensions = [".docx", ".doc"]

    async def parse(
        self,
        source_id: str,
        tenant_id: str,
        file_path: Optional[Path] = None,
        url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> CanonicalDocument:
        if file_path is None:
            raise ValueError("DocxParser requires a file_path")

        logger.info(f"Parsing DOCX document: {file_path}")

        try:
            markdown_content = self._parse_with_python_docx(file_path)
        except Exception as e:
            logger.warning(f"python-docx unavailable or failed ({e}), falling back to zipfile/XML parser")
            markdown_content = self._parse_with_xml(file_path)

        return CanonicalDocument(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=SourceType.DOCX,
            source_name=file_path.name,
            source_uri=str(file_path),
            markdown_content=markdown_content,
            page_texts=[markdown_content],
            total_pages=1,
            extra_metadata=extra_metadata or {},
        )

    def _parse_with_python_docx(self, file_path: Path) -> str:
        """Parse using python-docx if available."""
        import docx

        doc = docx.Document(str(file_path))
        lines = []

        # Process paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Basic heading detection by style name
            style_name = para.style.name.lower() if para.style else ""
            if "heading 1" in style_name:
                lines.append(f"# {text}")
            elif "heading 2" in style_name:
                lines.append(f"## {text}")
            elif "heading 3" in style_name:
                lines.append(f"### {text}")
            else:
                lines.append(text)

        # Process tables
        for table in doc.tables:
            table_lines = []
            for i, row in enumerate(table.rows):
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                row_str = "| " + " | ".join(row_cells) + " |"
                table_lines.append(row_str)
                if i == 0:
                    # Markdown table header separator
                    sep = "| " + " | ".join(["---"] * len(row_cells)) + " |"
                    table_lines.append(sep)
            if table_lines:
                lines.append("\n" + "\n".join(table_lines) + "\n")

        return "\n\n".join(lines)

    def _parse_with_xml(self, file_path: Path) -> str:
        """Built-in fallback: extracts document.xml from docx zip archive."""
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        lines = []

        with zipfile.ZipFile(str(file_path), "r") as z:
            if "word/document.xml" not in z.namelist():
                raise ValueError("Invalid .docx file: word/document.xml missing")
            xml_content = z.read("word/document.xml").decode("utf-8")

        root = ET.fromstring(xml_content)

        for elem in root.iter():
            # Paragraph
            if elem.tag == f"{{{ns['w']}}}p":
                para_texts = [t.text for t in elem.findall(".//w:t", ns) if t.text]
                if para_texts:
                    lines.append("".join(para_texts))
            # Table
            elif elem.tag == f"{{{ns['w']}}}tbl":
                tbl_rows = []
                for tr in elem.findall(".//w:tr", ns):
                    cells = []
                    for tc in tr.findall(".//w:tc", ns):
                        cell_texts = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                        cells.append("".join(cell_texts).strip().replace("\n", " "))
                    if cells:
                        tbl_rows.append("| " + " | ".join(cells) + " |")
                if tbl_rows:
                    sep = "| " + " | ".join(["---"] * (tbl_rows[0].count("|") - 1)) + " |"
                    tbl_rows.insert(1, sep)
                    lines.append("\n" + "\n".join(tbl_rows) + "\n")

        return "\n\n".join(lines)
