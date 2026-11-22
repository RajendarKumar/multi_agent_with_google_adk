"""
Embedding factory — returns the right embedding function based on EMBEDDING_PROVIDER.
Supports: Google (Vertex AI), OpenAI, Cohere, local (BGE-M3 via sentence-transformers).
All model names and vector dimensions are loaded dynamically from settings / environment variables.
"""

from abc import ABC, abstractmethod
from typing import Optional

from config.settings import EmbeddingProvider, settings


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Google (Vertex AI / Gemini Embedding API)
# ─────────────────────────────────────────────────────────────────────────────
class GoogleEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model = model or settings.embedding_model
        self._dim = dimension or settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"] if isinstance(result["embedding"][0], float) else result["embedding"]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────
class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model = model or settings.openai_embedding_model
        self._dim = dimension or settings.openai_embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────────────────────
# Cohere
# ─────────────────────────────────────────────────────────────────────────────
class CohereEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model = model or settings.cohere_embedding_model
        self._dim = dimension or settings.cohere_embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        response = await client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
            embedding_types=["float"],
        )
        return [e for e in response.embeddings.float_]


# ─────────────────────────────────────────────────────────────────────────────
# Local (BGE-M3 via sentence-transformers — dense + sparse natively)
# ─────────────────────────────────────────────────────────────────────────────
class LocalEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model_name = model or settings.local_embedding_model
        self._model = None
        self._dim = dimension or settings.local_embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        self._load()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True)
        )
        return embeddings.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_embedder(provider: Optional[EmbeddingProvider] = None) -> BaseEmbedder:
    p = provider or settings.embedding_provider
    if p == EmbeddingProvider.GOOGLE:
        return GoogleEmbedder()
    elif p == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder()
    elif p == EmbeddingProvider.COHERE:
        return CohereEmbedder()
    elif p == EmbeddingProvider.LOCAL:
        return LocalEmbedder()
    else:
        raise ValueError(f"Unknown embedding provider: {p}")
