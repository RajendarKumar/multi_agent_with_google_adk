"""
Central application settings loaded from environment variables / .env file.
All tuneable knobs are here — no magic strings scattered across the codebase.
"""

import os
from enum import Enum
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TenancyMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


class RerankerProvider(str, Enum):
    COHERE = "cohere"
    GOOGLE = "google"
    LOCAL = "local"


class RefreshUnit(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "change-me"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ Tenancy
    tenancy_mode: TenancyMode = TenancyMode.SINGLE
    default_tenant_id: str = "default"

    # ------------------------------------------------------------------ Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/responsive_agent"
    sync_database_url: str = "postgresql://user:password@localhost:5432/responsive_agent"

    # ------------------------------------------------------------------ Redis
    redis_url: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------ Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "responsive_agent_docs"

    # ------------------------------------------------------------------ Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.GOOGLE
    google_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""
    
    # Models & Dimensions per provider
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 3072

    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimension: int = 3072

    cohere_embedding_model: str = "embed-english-v3.0"
    cohere_embedding_dimension: int = 1024

    local_embedding_model: str = "BAAI/bge-m3"
    local_embedding_dimension: int = 1024

    # ------------------------------------------------------------------ Reranking
    reranker_provider: RerankerProvider = RerankerProvider.COHERE
    reranker_model: str = "rerank-english-v3.0"
    google_reranker_model: str = "semantic-ranker-512@latest"
    local_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 15
    reranker_min_score: float = 0.0001

    # ------------------------------------------------------------------ LLM Models
    orchestrator_model: str = "gemini-3.5-flash-lite"
    fast_model: str = "gemini-3.5-flash-lite"
    summarizer_model: str = "gemini-3.5-flash-lite"

    # ------------------------------------------------------------------ Web Refresh
    refresh_default_value: int = 24
    refresh_default_unit: RefreshUnit = RefreshUnit.HOUR

    # ------------------------------------------------------------------ MCP
    mcp_credential_encryption_key: str = "change-me-32-byte-key-here!!!!!!"
    mcp_default_read_only: bool = True
    mcp_max_servers_per_query: int = 3

    # ------------------------------------------------------------------ Auth
    jwt_secret: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ------------------------------------------------------------------ Observability
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ------------------------------------------------------------------ UI / Chat Display Flags
    show_evidence_sources: bool = False

    # ------------------------------------------------------------------ Weather MCP
    weatherapi_key: str = ""

    @property
    def is_langfuse_active(self) -> bool:
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key.strip()
            and self.langfuse_secret_key.strip()
        )

    # ------------------------------------------------------------------ RAG tuning
    rag_top_k_retrieve: int = 20       # candidates before reranking
    rag_top_k_final: int = 10          # after reranking
    rag_dense_weight: float = 0.7      # weight in RRF fusion
    rag_bm25_weight: float = 0.3

    @field_validator("mcp_credential_encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("MCP_CREDENTIAL_ENCRYPTION_KEY must be at least 32 characters")
        return v


# Singleton — import this everywhere
settings = Settings()

# Propagate credentials to os.environ so external SDKs (Google GenAI, Cohere, OpenAI) find them
if settings.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["GEMINI_API_KEY"] = settings.google_api_key

if settings.cohere_api_key and not os.environ.get("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = settings.cohere_api_key

if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
