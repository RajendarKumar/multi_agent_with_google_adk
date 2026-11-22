"""
End-to-end ingestion pipeline — orchestrates parse → chunk → index.
"""

from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import DocumentSource, IngestionStatus
from ingestion.chunker import SemanticChunker
from ingestion.indexer import HybridIndexer
from ingestion.parsers.router import get_parser


class IngestionPipeline:
    def __init__(self, session: AsyncSession):
        self.db = session
        self.chunker = SemanticChunker()
        self.indexer = HybridIndexer()

    async def ingest_source(self, source_id: str, tenant_id: str) -> None:
        result = await self.db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise ValueError(f"Source {source_id} not found")

        source.ingestion_status = IngestionStatus.PROCESSING
        await self.db.commit()

        try:
            parser = get_parser(
                mime_type=source.mime_type or "",
                file_path=Path(source.object_storage_path) if source.object_storage_path else None,
                url=source.url,
            )

            document = await parser.parse(
                source_id=source_id,
                tenant_id=tenant_id,
                file_path=Path(source.object_storage_path) if source.object_storage_path else None,
                url=source.url,
            )

            if not document or not document.markdown_content or not document.markdown_content.strip():
                raise ValueError(f"Document extraction produced empty content for source '{source.name}'")

            chunks = self.chunker.chunk(document)
            if not chunks:
                raise ValueError(f"Chunking produced 0 chunks for source '{source.name}'")

            indexed_count = await self.indexer.index_chunks(chunks)
            if indexed_count == 0:
                raise ValueError(f"Vector indexing failed: 0 chunks were indexed for source '{source.name}'")

            source.ingestion_status = IngestionStatus.COMPLETED
            source.chunk_count = indexed_count
            source.last_ingested_at = datetime.utcnow()
            source.ingestion_error = None
            logger.info(f"Ingested source {source_id}: {indexed_count} chunks")

        except Exception as e:
            source.ingestion_status = IngestionStatus.FAILED
            source.ingestion_error = str(e)
            logger.error(f"Ingestion failed for source {source_id}: {e}")
            raise

        finally:
            await self.db.commit()

    async def reingest_source(self, source_id: str, tenant_id: str) -> None:
        """
        Transactional / Shadow Re-ingestion:
        1. Parse and chunk fresh content FIRST without deleting existing vectors.
        2. If fresh ingestion fails, retain the existing knowledge base so chat flows never break.
        3. Only when fresh chunks are validated and indexed, purge old vectors and update metadata.
        """
        result = await self.db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise ValueError(f"Source {source_id} not found")

        source.ingestion_status = IngestionStatus.PROCESSING
        await self.db.commit()

        try:
            parser = get_parser(
                mime_type=source.mime_type or "",
                file_path=Path(source.object_storage_path) if source.object_storage_path else None,
                url=source.url,
            )

            document = await parser.parse(
                source_id=source_id,
                tenant_id=tenant_id,
                file_path=Path(source.object_storage_path) if source.object_storage_path else None,
                url=source.url,
            )

            if not document or not document.markdown_content or not document.markdown_content.strip():
                raise ValueError(f"Fresh fetch produced empty content for source '{source.name}'")

            chunks = self.chunker.chunk(document)
            if not chunks:
                raise ValueError(f"Chunking produced 0 chunks for source '{source.name}'")

            # Verified new chunks exist: now safely swap by purging previous vectors and indexing new batch
            await self.indexer.delete_source(source_id=source_id, tenant_id=tenant_id)
            indexed_count = await self.indexer.index_chunks(chunks)

            source.ingestion_status = IngestionStatus.COMPLETED
            source.chunk_count = indexed_count
            source.last_ingested_at = datetime.utcnow()
            source.ingestion_error = None
            logger.info(f"Successfully re-ingested source {source_id}: {indexed_count} chunks")

        except Exception as e:
            source.ingestion_status = IngestionStatus.FAILED
            source.ingestion_error = f"Re-ingestion failed: {e}. Existing index data preserved."
            logger.error(f"Re-ingestion failed for source {source_id}: {e}")
            raise

        finally:
            await self.db.commit()
