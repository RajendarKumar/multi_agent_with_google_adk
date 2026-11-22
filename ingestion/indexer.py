"""
Hybrid indexer — stores chunks in Qdrant with:
  • Dense vectors (from configured embedding provider)
  • BM25 sparse vectors (via Qdrant's sparse vector support)
  • Full provenance metadata for filtering and citation

Reciprocal Rank Fusion (RRF) merges dense and sparse rankings at query time.
"""


from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    models as qmodels,
)

from config.settings import settings
from ingestion.chunker import DocumentChunk
from ingestion.embedder import get_embedder


class HybridIndexer:
    """
    Indexes DocumentChunks into Qdrant with dense + BM25 sparse vectors.

    Collection schema:
      vectors:
        "dense"  — float embeddings from configured provider
        "sparse" — BM25 sparse embeddings (Qdrant native)
      payload: all DocumentChunk metadata fields
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.embedder = get_embedder()
        self.collection = settings.qdrant_collection_name

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=self.embedder.dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(),
                },
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    async def index_chunks(self, chunks: list[DocumentChunk], batch_size: int = 32) -> int:
        """Embed and index a list of chunks. Returns count indexed."""
        await self.ensure_collection()

        indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            # Dense embeddings
            dense_vecs = await self.embedder.embed(texts)

            # BM25 sparse (simple TF implementation; replace with Qdrant fastembed for production)
            sparse_vecs = [self._bm25_sparse(text) for text in texts]

            points = []
            for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "tenant_id": chunk.tenant_id,
                    "text": chunk.text,
                    "source_type": chunk.source_type.value,
                    "source_name": chunk.source_name,
                    "source_uri": chunk.source_uri,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "is_parent": chunk.extra_metadata.get("is_parent", False),
                    "ingested_at": chunk.ingested_at.isoformat(),
                    **{k: v for k, v in chunk.extra_metadata.items() if k != "is_parent"},
                }
                points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            self.DENSE_VECTOR_NAME: dense,
                            self.SPARSE_VECTOR_NAME: SparseVector(
                                indices=sparse["indices"],
                                values=sparse["values"],
                            ),
                        },
                        payload=payload,
                    )
                )

            await self.client.upsert(collection_name=self.collection, points=points)
            indexed += len(points)
            logger.debug(f"Indexed batch {i // batch_size + 1} ({len(points)} chunks)")

        return indexed

    async def delete_source(self, source_id: str, tenant_id: str) -> None:
        """Remove all chunks belonging to a source (used on refresh)."""
        await self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    ]
                )
            ),
        )

    # ── BM25 sparse vector helper ────────────────────────────────────────────
    # In production, replace with qdrant-fastembed's sparse encoder for
    # proper IDF weighting across the full corpus.

    def _bm25_sparse(self, text: str, max_terms: int = 128) -> dict:
        """
        Approximate BM25 sparse vector using TF (term frequency).
        Replace with fastembed.SparseTextEmbedding for production quality.
        """
        import re
        from collections import Counter

        tokens = re.findall(r"\b\w+\b", text.lower())
        counts = Counter(tokens)
        total = sum(counts.values()) or 1

        # vocabulary hash as term index (mod 2^20 for sparse dimension)
        indices, values = [], []
        for token, cnt in sorted(counts.items(), key=lambda x: -x[1])[:max_terms]:
            tf = cnt / total
            idx = abs(hash(token)) % (2**20)
            indices.append(idx)
            values.append(float(tf))

        return {"indices": indices, "values": values}
