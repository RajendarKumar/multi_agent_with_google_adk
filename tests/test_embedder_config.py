"""
Unit tests verifying that all Embedder and Reranker classes load their model
names and dimensions from settings / environment variables without hardcoded values.
"""

from agents.reranker import (
    CohereReranker,
    GoogleReranker,
    LocalReranker,
)
from config.settings import settings
from ingestion.embedder import (
    CohereEmbedder,
    GoogleEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)


def test_embedders_pick_from_settings():
    # 1. Google
    google_emb = GoogleEmbedder()
    assert google_emb.model == settings.embedding_model
    assert google_emb.dimension == settings.embedding_dimension

    # 2. OpenAI
    openai_emb = OpenAIEmbedder()
    assert openai_emb.model == settings.openai_embedding_model
    assert openai_emb.dimension == settings.openai_embedding_dimension

    # 3. Cohere
    cohere_emb = CohereEmbedder()
    assert cohere_emb.model == settings.cohere_embedding_model
    assert cohere_emb.dimension == settings.cohere_embedding_dimension

    # 4. Local
    local_emb = LocalEmbedder()
    assert local_emb.model_name == settings.local_embedding_model
    assert local_emb.dimension == settings.local_embedding_dimension


def test_embedders_allow_custom_overrides():
    custom_openai = OpenAIEmbedder(model="text-embedding-3-small", dimension=1536)
    assert custom_openai.model == "text-embedding-3-small"
    assert custom_openai.dimension == 1536

    custom_cohere = CohereEmbedder(model="custom-cohere-v4", dimension=512)
    assert custom_cohere.model == "custom-cohere-v4"
    assert custom_cohere.dimension == 512

    custom_local = LocalEmbedder(model="custom/model-path", dimension=768)
    assert custom_local.model_name == "custom/model-path"
    assert custom_local.dimension == 768


def test_rerankers_pick_from_settings():
    cohere_rerank = CohereReranker()
    assert cohere_rerank.model == settings.reranker_model

    google_rerank = GoogleReranker()
    assert google_rerank.model == settings.google_reranker_model

    local_rerank = LocalReranker()
    assert local_rerank.model_name == settings.local_reranker_model
