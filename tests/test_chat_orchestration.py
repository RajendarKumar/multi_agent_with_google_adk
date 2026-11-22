"""
Unit tests for the Google ADK Chat Orchestrator and Agent Graph.
Validates multi-agent construction, RerankerAgent, hybrid search tool, and state keys.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agents.orchestrator import (
    build_root_orchestrator,
    build_search_pipeline,
    build_local_rag_agent,
    build_mcp_agent,
    query_rewriter_agent,
    content_summarizer_agent,
    grounded_synthesizer,
    verifier_agent,
    synthesis_loop,
    RerankerAgent,
    hybrid_search_tool,
)
from agents.reranker import EvidenceItem


def test_agent_graph_construction():
    """Verify that the multi-agent graph builds with correct sub-agents and tools."""
    tenant_id = "tenant_test"
    mcp_toolsets = []

    rag_agent = build_local_rag_agent(tenant_id)
    assert rag_agent.name == "local_rag_agent"
    assert rag_agent.output_key == "rag_evidence"

    mcp_agent = build_mcp_agent(mcp_toolsets)
    assert mcp_agent.name == "dynamic_mcp_agent"
    assert mcp_agent.output_key == "mcp_evidence"

    pipeline = build_search_pipeline(tenant_id, mcp_toolsets)
    assert pipeline.name == "unified_search_pipeline"
    sub_agent_names = [a.name for a in pipeline.sub_agents]
    assert "query_rewriter_agent" in sub_agent_names
    assert "parallel_evidence_gatherer" in sub_agent_names
    assert "reranker_agent" in sub_agent_names
    assert "synthesis_loop" in sub_agent_names

    orchestrator = build_root_orchestrator(tenant_id, mcp_toolsets)
    assert orchestrator.name == "platform_orchestrator"
    assert len(orchestrator.tools) >= 2


def test_agent_configurations_loaded_from_yaml():
    """Verify that agent properties (instructions, models) are loaded from config/agents.yaml."""
    assert query_rewriter_agent.name == "query_rewriter_agent"
    assert query_rewriter_agent.output_key == "rewritten_query"
    assert content_summarizer_agent.name == "content_summarizer_agent"
    assert content_summarizer_agent.output_key == "summary"
    assert grounded_synthesizer.name == "grounded_synthesizer"
    assert grounded_synthesizer.output_key == "draft_answer"
    assert verifier_agent.name == "verifier_agent"
    assert synthesis_loop.name == "synthesis_loop"
    assert synthesis_loop.max_iterations == 2


@pytest.mark.asyncio
async def test_reranker_agent_deterministic_scoring():
    """Verify RerankerAgent as a deterministic BaseAgent sorting mixed evidence."""
    reranker_agent = RerankerAgent()

    # Mock context session state
    mock_ctx = MagicMock()
    mock_ctx.session.state = {
        "user_query": "return policy",
        "rewritten_query": "Acme Corp return and refund policy",
        "rag_evidence": [
            {"text": "Acme returns are accepted within 30 days.", "source_name": "terms.pdf", "source_uri": "uploads/terms.pdf", "chunk_id": "c1"},
            {"text": "Irrelevant document about office parking.", "source_name": "parking.txt", "source_uri": "uploads/parking.txt", "chunk_id": "c2"},
        ],
        "mcp_evidence": [
            {"text": "Jira ticket: Update return policy button on website.", "source_name": "jira", "origin_type": "mcp", "mcp_server_name": "jira"},
        ],
    }

    # Mock the underlying reranker
    mock_reranked = [
        EvidenceItem(text="Acme returns are accepted within 30 days.", source_name="terms.pdf", source_uri="uploads/terms.pdf", origin_type="rag", rerank_score=0.95),
        EvidenceItem(text="Jira ticket: Update return policy button on website.", source_name="jira", source_uri="mcp://jira", origin_type="mcp", rerank_score=0.82),
        EvidenceItem(text="Irrelevant document about office parking.", source_name="parking.txt", source_uri="uploads/parking.txt", origin_type="rag", rerank_score=0.10),
    ]

    with patch.object(reranker_agent._reranker, "rerank", new=AsyncMock(return_value=mock_reranked)):
        events = []
        async for evt in reranker_agent._run_async_impl(mock_ctx):
            events.append(evt)

        assert len(events) == 1
        state_delta = events[0].actions.state_delta
        assert "reranked_evidence" in state_delta
        reranked = state_delta["reranked_evidence"]
        assert len(reranked) == 3
        # Should be ordered highest score first
        assert reranked[0]["source_name"] == "terms.pdf"
        assert reranked[0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_hybrid_search_tool_rrf_blending():
    """Verify hybrid_search_tool correctly blends dense and BM25 search via RRF."""
    mock_point_1 = MagicMock(id="p1", payload={"text": "First chunk text", "source_name": "doc1.pdf", "page_number": 1})
    mock_point_2 = MagicMock(id="p2", payload={"text": "Second chunk text", "source_name": "doc2.pdf", "page_number": 2})

    mock_dense_resp = MagicMock(points=[mock_point_1, mock_point_2])
    mock_sparse_resp = MagicMock(points=[mock_point_2, mock_point_1])

    with patch("ingestion.indexer.HybridIndexer") as mock_indexer_cls, \
         patch("ingestion.embedder.get_embedder") as mock_get_embedder:

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
        mock_get_embedder.return_value = mock_embedder

        mock_indexer_instance = MagicMock()
        mock_indexer_instance.collection = "test_collection"
        mock_indexer_instance.client.query_points = AsyncMock(side_effect=[mock_dense_resp, mock_sparse_resp])
        mock_indexer_instance._bm25_sparse.return_value = {"indices": [1, 2], "values": [1.0, 1.0]}
        mock_indexer_cls.return_value = mock_indexer_instance

        results = await hybrid_search_tool("sample query", tenant_id="default", top_k=2)
        assert len(results) == 2
        chunk_ids = [r["chunk_id"] for r in results]
        assert "p1" in chunk_ids
        assert "p2" in chunk_ids
