"""
Reranker factory — pluggable cross-encoder reranking.
Supported providers: Cohere, Google (Vertex AI Ranking API), local (BGE-Reranker).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config.settings import RerankerProvider, settings


@dataclass
class EvidenceItem:
    """Normalised evidence from RAG or MCP — common schema for reranking."""
    text: str
    source_name: str
    source_uri: str
    origin_type: str              # "rag" | "mcp"
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_id: Optional[str] = None
    mcp_server_name: Optional[str] = None
    raw_score: float = 0.0        # score from retriever (before reranking)
    rerank_score: float = 0.0     # score from cross-encoder


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: list[EvidenceItem],
        top_n: Optional[int] = None,
    ) -> list[EvidenceItem]:
        """Return items sorted by relevance, filtered to top_n."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Cohere Rerank 3.5
# ─────────────────────────────────────────────────────────────────────────────
class CohereReranker(BaseReranker):
    def __init__(
        self,
        model: str = settings.reranker_model,
        min_score: float = settings.reranker_min_score,
    ):
        self.model = model
        self.min_score = min_score

    async def rerank(
        self,
        query: str,
        items: list[EvidenceItem],
        top_n: Optional[int] = None,
    ) -> list[EvidenceItem]:
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        documents = [item.text for item in items]
        k = top_n or settings.reranker_top_n

        response = await client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=min(k, len(documents)),
        )

        reranked = []
        for r in response.results:
            if r.relevance_score >= self.min_score:
                item = items[r.index]
                item.rerank_score = r.relevance_score
                reranked.append(item)

        if not reranked and response.results and items:
            top_r = response.results[0]
            fallback_item = items[top_r.index]
            fallback_item.rerank_score = top_r.relevance_score
            reranked.append(fallback_item)

        return sorted(reranked, key=lambda x: x.rerank_score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Google Vertex AI Ranking API
# ─────────────────────────────────────────────────────────────────────────────
class GoogleReranker(BaseReranker):
    def __init__(
        self,
        model: Optional[str] = None,
        min_score: float = settings.reranker_min_score,
    ):
        self.model = model or settings.google_reranker_model
        self.min_score = min_score

    async def rerank(
        self,
        query: str,
        items: list[EvidenceItem],
        top_n: Optional[int] = None,
    ) -> list[EvidenceItem]:
        # Google Discovery Engine Ranking API

        from google.cloud import discoveryengine_v1alpha as discoveryengine

        client = discoveryengine.RankServiceAsyncClient()
        k = top_n or settings.reranker_top_n

        ranking_records = [
            discoveryengine.RankingRecord(id=str(i), content=item.text)
            for i, item in enumerate(items)
        ]
        request = discoveryengine.RankRequest(
            ranking_config="default_ranking_config",
            model=self.model,
            top_n=min(k, len(items)),
            query=query,
            records=ranking_records,
        )
        response = await client.rank(request=request)

        id_to_score = {r.id: r.score for r in response.records}
        reranked = []
        for i, item in enumerate(items):
            score = id_to_score.get(str(i), 0.0)
            if score >= self.min_score:
                item.rerank_score = score
                reranked.append(item)

        return sorted(reranked, key=lambda x: x.rerank_score, reverse=True)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Local (BGE-Reranker-v2-m3 — self-hosted, no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
class LocalReranker(BaseReranker):
    def __init__(
        self,
        model: Optional[str] = None,
        min_score: float = settings.reranker_min_score,
    ):
        self.model_name = model or settings.local_reranker_model
        self.min_score = min_score
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    async def rerank(
        self,
        query: str,
        items: list[EvidenceItem],
        top_n: Optional[int] = None,
    ) -> list[EvidenceItem]:
        import asyncio
        self._load()
        k = top_n or settings.reranker_top_n
        pairs = [(query, item.text) for item in items]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, lambda: self._model.predict(pairs))

        reranked = []
        for item, score in zip(items, scores):
            if float(score) >= self.min_score:
                item.rerank_score = float(score)
                reranked.append(item)

        return sorted(reranked, key=lambda x: x.rerank_score, reverse=True)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_reranker(provider: Optional[RerankerProvider] = None) -> BaseReranker:
    p = provider or settings.reranker_provider
    if p == RerankerProvider.COHERE:
        return CohereReranker()
    elif p == RerankerProvider.GOOGLE:
        return GoogleReranker()
    elif p == RerankerProvider.LOCAL:
        return LocalReranker()
    else:
        raise ValueError(f"Unknown reranker provider: {p}")
