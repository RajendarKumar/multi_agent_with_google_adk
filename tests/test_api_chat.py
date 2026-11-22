"""
Integration tests for the Chat API routes, SSE streaming, and Query Tracing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from api.main import app
from api.database import get_db
from api.models import QueryTrace
from api.tracing import record_query_trace


@pytest.fixture
def mock_db_session():
    """Mock async database session for isolated API testing."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_record_query_trace(mock_db_session):
    """Verify record_query_trace persists metrics to QueryTrace table."""
    trace_id = await record_query_trace(
        tenant_id="default",
        session_id="sess_trace_01",
        user_query="What is the refund policy?",
        final_answer="Refunds are processed in 30 days.",
        rewritten_query="Acme refund policy timeline",
        total_latency_ms=1250,
        grounding_passed=True,
        rag_evidence=[{"source_name": "terms.pdf", "text": "30 days refund"}],
        db=mock_db_session,
    )

    assert trace_id is not None
    assert mock_db_session.add.called
    added_obj = mock_db_session.add.call_args[0][0]
    assert isinstance(added_obj, QueryTrace)
    assert added_obj.session_id == "sess_trace_01"
    assert added_obj.total_latency_ms == 1250
    assert added_obj.grounding_passed is True


@pytest.mark.asyncio
async def test_chat_sync_endpoint(mock_db_session):
    """Test POST /api/chat/ returns structured answer and sources."""
    app.dependency_overrides[get_db] = lambda: mock_db_session

    # Mock _run_agent response
    mock_result = {
        "session_id": "sess_test_123",
        "answer": "Acme refunds take up to 30 business days [terms.pdf, p.5].",
        "sources": [{"source_name": "terms.pdf", "page_number": 5}],
        "grounding_passed": True,
        "latency_ms": 1150,
        "trace_id": "tr_12345",
    }

    with patch("api.routes.chat._run_agent", new=AsyncMock(return_value=mock_result)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/chat/",
                json={"message": "How do I get a refund?", "session_id": "sess_test_123"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "sess_test_123"
            assert "terms.pdf" in data["answer"]
            assert data["grounding_passed"] is True

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_sse_stream_endpoint(mock_db_session):
    """Test POST /api/chat/stream yields SSE events in proper W3C format."""
    app.dependency_overrides[get_db] = lambda: mock_db_session

    async def mock_stream_gen(request, db):
        yield "event: status\ndata: {\"message\": \"🔍 Analysing your query...\"}\n\n"
        yield "event: status\ndata: {\"message\": \"📄 Searching documents...\"}\n\n"
        yield "event: status\ndata: {\"message\": \"✅ Verifying citations...\"}\n\n"
        yield "event: answer\ndata: {\"session_id\": \"sess_stream\", \"answer\": \"Answer text\", \"sources\": [], \"grounding_passed\": true}\n\n"
        yield "event: done\ndata: {}\n\n"

    with patch("api.routes.chat._stream_agent", side_effect=mock_stream_gen):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/chat/stream",
                json={"message": "Explain the architecture", "session_id": "sess_stream"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text

            assert "event: status" in body
            assert "Analysing your query..." in body
            assert "Searching documents..." in body
            assert "event: answer" in body
            assert "Answer text" in body
            assert "event: done" in body

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_admin_list_and_get_traces(mock_db_session):
    """Test GET /api/admin/traces endpoints."""
    app.dependency_overrides[get_db] = lambda: mock_db_session

    mock_trace = MagicMock()
    mock_trace.id = "trace_001"
    mock_trace.session_id = "sess_01"
    mock_trace.user_id = "user_01"
    mock_trace.user_query = "Test query"
    mock_trace.rewritten_query = "Rewritten test query"
    mock_trace.final_answer = "Final answer text"
    mock_trace.grounding_passed = True
    mock_trace.total_latency_ms = 950
    mock_trace.input_tokens = 20
    mock_trace.output_tokens = 50
    mock_trace.created_at = None
    mock_trace.routing_decision = "search"
    mock_trace.rag_evidence = None
    mock_trace.mcp_evidence = None
    mock_trace.reranked_evidence = None
    mock_trace.reranker_scores = None

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_trace]
    mock_result.scalar_one_or_none.return_value = mock_trace
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. List traces
        list_resp = await client.get("/api/admin/traces")
        assert list_resp.status_code == 200
        traces = list_resp.json()
        assert len(traces) == 1
        assert traces[0]["id"] == "trace_001"

        # 2. Get single trace
        get_resp = await client.get("/api/admin/traces/trace_001")
        assert get_resp.status_code == 200
        trace_data = get_resp.json()
        assert trace_data["id"] == "trace_001"
        assert trace_data["user_query"] == "Test query"

    app.dependency_overrides.clear()


def test_sanitize_ai_error():
    """Verify that 3rd-party model errors (503, 429, raw JSON) are sanitized into user-friendly copy."""
    from api.routes.chat import sanitize_ai_error

    # 1. 503 UNAVAILABLE error
    err_503 = Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}")
    msg_503 = sanitize_ai_error(err_503)
    assert "503" not in msg_503
    assert "{" not in msg_503
    assert "high demand" in msg_503 or "heavy traffic" in msg_503

    # 2. 429 RESOURCE_EXHAUSTED error
    err_429 = Exception("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Quota exceeded.'}}")
    msg_429 = sanitize_ai_error(err_429)
    assert "429" not in msg_429
    assert "{" not in msg_429
    assert "high volume of requests" in msg_429

    # 3. Generic internal error
    err_generic = Exception("Unexpected connection reset by peer")
    msg_generic = sanitize_ai_error(err_generic)
    assert "connection" in msg_generic.lower() or "temporary" in msg_generic.lower()


def test_langfuse_configurable_toggle():
    """Verify that Langfuse tracing can be enabled or disabled cleanly via settings."""
    from config.settings import settings
    from api.tracing import get_langfuse_client, _publish_to_langfuse

    # 1. When disabled via setting
    with patch.object(settings, "langfuse_enabled", False):
        assert settings.is_langfuse_active is False
        assert get_langfuse_client() is None
        # Calling publish does nothing and never raises
        _publish_to_langfuse(
            tenant_id="default",
            session_id="s1",
            user_id="u1",
            user_query="hi",
            rewritten_query="hello",
            final_answer="answer",
            routing_decision="search",
            rag_evidence=None,
            mcp_evidence=None,
            reranked_evidence=None,
            grounding_passed=True,
            total_latency_ms=100,
            input_tokens=10,
            output_tokens=10,
            reranker_scores=None,
        )

    # 2. When keys are empty
    with patch.object(settings, "langfuse_enabled", True):
        with patch.object(settings, "langfuse_public_key", ""):
            assert settings.is_langfuse_active is False
            assert get_langfuse_client() is None

