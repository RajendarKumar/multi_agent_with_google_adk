"""
Semantic + structure-aware chunker.
Respects Markdown headings, keeps tables intact, and builds parent-child pairs
for high-precision retrieval (fetch small child chunks, pass larger parent to LLM).
"""

import re
import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ingestion.parsers.base import CanonicalDocument, DocumentChunk


@dataclass
class ChunkerConfig:
    chunk_size: int = 512          # target characters per child chunk
    chunk_overlap: int = 64        # character overlap between chunks
    parent_chunk_size: int = 2048  # parent chunk size (passed to LLM)
    min_chunk_size: int = 50       # discard chunks shorter than this


class SemanticChunker:
    """
    Splits a CanonicalDocument into DocumentChunks respecting structure.

    Strategy:
      1. Split on Markdown headings (##, ###) first to preserve sections.
      2. Within each section, split on paragraph boundaries.
      3. For long sections, apply sliding window with overlap.
      4. Tables are kept as single chunks (never split mid-table).
      5. Build parent chunks (2048 chars) wrapping child chunks (512 chars).
    """

    # Regex to detect Markdown tables — keep them intact
    TABLE_PATTERN = re.compile(r"(\|.+\|[\r\n]+)+", re.MULTILINE)

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.cfg = config or ChunkerConfig()

    def chunk(self, document: CanonicalDocument) -> list[DocumentChunk]:
        sections = self._split_by_headings(document.markdown_content)
        all_chunks: list[DocumentChunk] = []

        default_heading = document.source_name or ""
        for heading, section_text in sections:
            effective_heading = heading.strip() if heading and heading.strip() else default_heading
            section_chunks = self._split_section(
                section_text=section_text,
                heading=effective_heading,
                document=document,
            )
            all_chunks.extend(section_chunks)

        logger.debug(f"Chunked '{document.source_name}' into {len(all_chunks)} chunks")
        return all_chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split Markdown text at ## / ### headings."""
        heading_pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
        parts = heading_pattern.split(text)

        sections: list[tuple[str, str]] = []
        current_heading = ""
        for part in parts:
            if heading_pattern.match(part):
                current_heading = part.strip()
            else:
                if part.strip():
                    sections.append((current_heading, part.strip()))

        if not sections:
            sections = [("", text)]
        return sections

    def _split_section(
        self,
        section_text: str,
        heading: str,
        document: CanonicalDocument,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        parent_id = str(uuid.uuid4())
        parent_text_parts: list[str] = []

        # Protect tables — extract them before splitting
        tables: dict[str, str] = {}
        def replace_table(m: re.Match) -> str:
            key = f"__TABLE_{uuid.uuid4().hex}__"
            tables[key] = m.group(0)
            return key

        processed = self.TABLE_PATTERN.sub(replace_table, section_text)

        # Split on double newlines first
        raw_paras = [p.strip() for p in re.split(r"\n{2,}", processed) if p.strip()]

        # Ensure no paragraph exceeds chunk_size (sub-split by single newlines or sentence boundaries)
        paragraphs: list[str] = []
        for p in raw_paras:
            if len(p) <= self.cfg.chunk_size:
                paragraphs.append(p)
            else:
                lines = [line.strip() for line in p.split("\n") if line.strip()]
                sub_buf = ""
                for line in lines:
                    if len(line) > self.cfg.chunk_size:
                        # Split oversized line by sentences
                        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
                        for s in sentences:
                            if len(sub_buf) + len(s) + 1 > self.cfg.chunk_size and sub_buf:
                                paragraphs.append(sub_buf.strip())
                                sub_buf = s
                            else:
                                sub_buf += (" " if sub_buf else "") + s
                    elif len(sub_buf) + len(line) + 1 > self.cfg.chunk_size and sub_buf:
                        paragraphs.append(sub_buf.strip())
                        sub_buf = line
                    else:
                        sub_buf += ("\n" if sub_buf else "") + line

                if sub_buf.strip():
                    paragraphs.append(sub_buf.strip())

        buffer = ""
        char_start = 0

        for para in paragraphs:
            # Restore table placeholders
            para = tables.get(para, para)
            for key, table in tables.items():
                para = para.replace(key, table)

            if len(buffer) + len(para) > self.cfg.chunk_size and buffer:
                chunk = self._make_chunk(
                    text=buffer.strip(),
                    heading=heading,
                    document=document,
                    parent_id=parent_id,
                    char_start=char_start,
                )
                if chunk:
                    chunks.append(chunk)
                    parent_text_parts.append(buffer.strip())
                # Overlap: carry last N chars into next buffer
                overlap_start = max(0, len(buffer) - self.cfg.chunk_overlap)
                buffer = buffer[overlap_start:] + "\n\n" + para
            else:
                if not buffer:
                    char_start = section_text.find(para[:30])
                buffer += ("\n\n" if buffer else "") + para

        # Flush remaining buffer
        if buffer.strip():
            chunk = self._make_chunk(
                text=buffer.strip(),
                heading=heading,
                document=document,
                parent_id=parent_id,
                char_start=char_start,
            )
            if chunk:
                chunks.append(chunk)
                parent_text_parts.append(buffer.strip())

        # Build parent chunk
        parent_text = "\n\n".join(parent_text_parts)[: self.cfg.parent_chunk_size]
        if parent_text.strip() and len(chunks) > 1:
            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                source_id=document.source_id,
                tenant_id=document.tenant_id,
                text=parent_text,
                source_type=document.source_type,
                source_name=document.source_name,
                source_uri=document.source_uri,
                section_heading=heading,
                extra_metadata={"is_parent": True},
            )
            chunks.insert(0, parent_chunk)

        return chunks

    def _make_chunk(
        self,
        text: str,
        heading: str,
        document: CanonicalDocument,
        parent_id: str,
        char_start: int,
    ) -> Optional[DocumentChunk]:
        if len(text) < self.cfg.min_chunk_size:
            return None
        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            source_id=document.source_id,
            tenant_id=document.tenant_id,
            text=text,
            source_type=document.source_type,
            source_name=document.source_name,
            source_uri=document.source_uri,
            section_heading=heading,
            parent_chunk_id=parent_id,
            char_start=char_start,
            char_end=char_start + len(text),
        )
