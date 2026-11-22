"""
Unit tests for the Memory Layer (STM & LTM).
Validates session creation, state mutation, in-memory fallback, and session persistence.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory.memory import STMService, LTMService


@pytest.mark.asyncio
async def test_stm_service_in_memory_lifecycle():
    """Verify STMService session creation, retrieval, and state update with fallback."""
    stm = STMService(use_in_memory=True)

    session_id = "test_sess_001"
    user_id = "user_42"
    tenant_id = "tenant_acme"

    # 1. Create or get session
    session = await stm.get_or_create_session(session_id, user_id, tenant_id)
    assert session is not None
    assert session.id == session_id
    assert session.state.get("tenant_id") == tenant_id

    # 2. Update state dictionary
    updates = {
        "rewritten_query": "What is our vacation policy?",
        "rag_evidence": [{"text": "20 days PTO per year.", "source_name": "handbook.pdf"}],
        "draft_answer": "You receive 20 days PTO.",
    }
    await stm.update_state(session_id, user_id, tenant_id, updates)

    # 3. Retrieve session again and assert state updated
    refreshed = await stm.get_or_create_session(session_id, user_id, tenant_id)
    assert refreshed.state.get("rewritten_query") == "What is our vacation policy?"
    assert len(refreshed.state.get("rag_evidence", [])) == 1
    assert refreshed.state.get("draft_answer") == "You receive 20 days PTO."


@pytest.mark.asyncio
async def test_ltm_service_persist_session():
    """Verify LTMService persist_session gracefully extracts durable facts without raising."""
    ltm = LTMService()
    assert ltm.get_service() is not None

    mock_session = MagicMock()
    mock_session.id = "sess_ltm_001"

    mock_runner = MagicMock()
    mock_runner.memory_service = MagicMock()
    mock_runner.memory_service.add_session_to_memory = AsyncMock(return_value=None)

    # Should succeed without raising exceptions
    await ltm.persist_session(mock_session, mock_runner)
    mock_runner.memory_service.add_session_to_memory.assert_awaited_once_with(mock_session)
