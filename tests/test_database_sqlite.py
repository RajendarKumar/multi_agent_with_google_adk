"""
Database integration tests using in-memory SQLite (aiosqlite).
Validates schema creation, SQLAlchemy ORM mappings, constraints, and transactions.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from api.models import Base, Tenant, DocumentSource, MCPServer, QueryTrace, IngestionStatus, MCPTransport, MCPActionScope


@pytest_asyncio.fixture
async def sqlite_engine():
    """Create a temporary in-memory SQLite database engine with async support."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(sqlite_engine):
    """Provide an isolated database session per test with automatic rollback/close."""
    session_factory = async_sessionmaker(
        bind=sqlite_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest.mark.asyncio
async def test_sqlite_tenant_crud(db_session: AsyncSession):
    """Verify Tenant model creation, query, and fields on SQLite."""
    tenant = Tenant(name="Tenant Alpha", is_active=True, metadata_={"plan": "enterprise"})
    db_session.add(tenant)
    await db_session.commit()

    result = await db_session.execute(select(Tenant).where(Tenant.name == "Tenant Alpha"))
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.id is not None
    assert fetched.name == "Tenant Alpha"
    assert fetched.metadata_ == {"plan": "enterprise"}
    assert fetched.is_active is True


@pytest.mark.asyncio
async def test_sqlite_document_source_crud(db_session: AsyncSession):
    """Verify DocumentSource model creation, updates, and JSON metadata on SQLite."""
    source = DocumentSource(
        tenant_id="default",
        source_type="pdf",
        name="Architecture_Guide.pdf",
        object_storage_path="/uploads/arch.pdf",
        file_size_bytes=1048576,
        mime_type="application/pdf",
        ingestion_status=IngestionStatus.PENDING,
        chunk_count=0,
        extra_metadata={"author": "Engineering", "department": "Platform"},
    )
    db_session.add(source)
    await db_session.commit()

    # Query source
    result = await db_session.execute(
        select(DocumentSource).where(DocumentSource.name == "Architecture_Guide.pdf")
    )
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.source_type == "pdf"
    assert fetched.extra_metadata["author"] == "Engineering"

    # Update status to completed
    fetched.ingestion_status = IngestionStatus.COMPLETED
    fetched.chunk_count = 42
    await db_session.commit()

    # Re-query
    result2 = await db_session.execute(
        select(DocumentSource).where(DocumentSource.id == fetched.id)
    )
    updated = result2.scalar_one_or_none()
    assert updated.ingestion_status == IngestionStatus.COMPLETED
    assert updated.chunk_count == 42


@pytest.mark.asyncio
async def test_sqlite_mcp_server_crud_and_constraints(db_session: AsyncSession):
    """Verify MCPServer model CRUD and unique constraint behavior on SQLite."""
    server1 = MCPServer(
        tenant_id="tenant_1",
        name="jira_server",
        description="Jira integration",
        transport_type=MCPTransport.STREAMABLE_HTTP,
        endpoint_url="https://jira.example.com/mcp",
        action_scope=MCPActionScope.READ_ONLY,
        allowed_tools=["jira_search"],
        blocked_tools=["jira_delete"],
        cached_tool_schemas=[{"name": "jira_search", "description": "Search tickets"}],
        enabled=True,
    )
    db_session.add(server1)
    await db_session.commit()

    result = await db_session.execute(
        select(MCPServer).where(MCPServer.tenant_id == "tenant_1", MCPServer.name == "jira_server")
    )
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.name == "jira_server"
    assert fetched.action_scope == MCPActionScope.READ_ONLY
    assert len(fetched.cached_tool_schemas) == 1

    # Verify duplicate insertion raises IntegrityError
    server_dup = MCPServer(
        tenant_id="tenant_1",
        name="jira_server",
        description="Duplicate",
    )
    db_session.add(server_dup)
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_sqlite_query_trace_persistence(db_session: AsyncSession):
    """Verify QueryTrace model persists execution latencies, token counts, and evidence snippets."""
    trace = QueryTrace(
        tenant_id="default",
        session_id="sess_sqlite_1",
        user_query="What is the data retention policy?",
        rewritten_query="Acme data retention policy timeline",
        routing_decision="search",
        rag_evidence=[{"text": "Data is retained for 7 years.", "source": "compliance.pdf"}],
        mcp_evidence=None,
        reranked_evidence=[{"text": "Data is retained for 7 years.", "score": 0.94}],
        final_answer="According to compliance policies, data is retained for 7 years.",
        grounding_passed=True,
        total_latency_ms=850,
        input_tokens=150,
        output_tokens=45,
        reranker_scores={"bge_cross_encoder": [0.94]},
    )
    db_session.add(trace)
    await db_session.commit()

    result = await db_session.execute(
        select(QueryTrace).where(QueryTrace.session_id == "sess_sqlite_1")
    )
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.user_query == "What is the data retention policy?"
    assert fetched.grounding_passed is True
    assert fetched.total_latency_ms == 850
    assert fetched.reranker_scores["bge_cross_encoder"] == [0.94]
