"""
Database models — all tables carry an optional tenant_id column.
In SINGLE tenancy mode tenant_id is always DEFAULT_TENANT_ID.
In MULTI tenancy mode tenant_id isolates data per tenant.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Tenants  (only populated in MULTI tenancy mode)
# ─────────────────────────────────────────────────────────────────────────────
class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# Document Sources (PDFs + Web URLs)
# ─────────────────────────────────────────────────────────────────────────────
class SourceType(str):
    PDF = "pdf"
    WEB = "web"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class IngestionStatus(str):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentSource(Base):
    """
    Represents a registered ingestion source (PDF file or web URL).
    Web sources carry refresh scheduling config.
    """
    __tablename__ = "document_sources"
    __table_args__ = (
        Index("ix_doc_sources_tenant", "tenant_id"),
        Index("ix_doc_sources_status", "ingestion_status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)

    # For web sources
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # For file sources
    object_storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Ingestion state
    ingestion_status: Mapped[str] = mapped_column(String(32), default=IngestionStatus.PENDING)
    ingestion_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    last_ingested_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Refresh scheduling (web sources only)
    # refresh_enabled=False means one-time snapshot (like PDFs)
    refresh_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    refresh_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # e.g. 24
    refresh_unit: Mapped[Optional[str]] = mapped_column(String(16), nullable=True) # e.g. "hour"
    next_refresh_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    created_by: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Registry
# ─────────────────────────────────────────────────────────────────────────────
class MCPTransport(str):
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"
    STDIO = "stdio"


class MCPAuthType(str):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2 = "oauth2"


class MCPActionScope(str):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"   # future


class MCPServer(Base):
    """
    Registered external MCP server. Credentials are stored encrypted.
    action_scope controls whether write tools are exposed (future extensibility).
    """
    __tablename__ = "mcp_servers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_mcp_tenant_name"),
        Index("ix_mcp_servers_tenant_enabled", "tenant_id", "enabled"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Transport
    transport_type: Mapped[str] = mapped_column(String(32), nullable=False, default=MCPTransport.STREAMABLE_HTTP)
    endpoint_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stdio_command: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # for stdio transport

    # Auth (credential stored encrypted in auth_secret_encrypted)
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False, default=MCPAuthType.NONE)
    auth_secret_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scope & Access Control
    # READ_ONLY by default; set to READ_WRITE to allow write tools (future)
    action_scope: Mapped[str] = mapped_column(String(32), nullable=False, default=MCPActionScope.READ_ONLY)
    allowed_tools: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # None = all tools
    blocked_tools: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Cached tool schemas (populated on registration / health-check)
    cached_tool_schemas: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tools_last_refreshed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Query / Execution Trace (Observability)
# ─────────────────────────────────────────────────────────────────────────────
class QueryTrace(Base):
    __tablename__ = "query_traces"
    __table_args__ = (
        Index("ix_traces_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    routing_decision: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # search | summarize | both

    # Evidence
    rag_evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    mcp_evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reranked_evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Response
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    grounding_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Metrics
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reranker_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
