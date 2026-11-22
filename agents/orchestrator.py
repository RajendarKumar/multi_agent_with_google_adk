"""
Google ADK Agent Definitions.

Agents are configured from config/agents.yaml — no code changes needed
when adjusting models, temperatures, or instructions.

Orchestration pattern: Hybrid Coordinator + Workflow Agents
  Root Orchestrator (LlmAgent)
    ├── content_summarizer_agent  (via AgentTool)
    └── unified_search_pipeline   (SequentialAgent, via AgentTool)
          ├── query_rewriter_agent     (Step 1 — Smart deterministic/fast LLM)
          ├── parallel_evidence_gatherer (Step 2 — ParallelAgent / LocalRagAgent)
          │     ├── local_rag_agent    (Hybrid Qdrant dense+BM25 + RRF)
          │     └── dynamic_mcp_agent
          ├── reranker_agent           (Step 3 — deterministic BaseAgent)
          └── synthesis_loop           (Step 4 — LlmAgent + VerifierAgent)
"""

import json
from pathlib import Path
from typing import AsyncIterator

import yaml
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.tools import AgentTool, load_memory
from loguru import logger

from agents.reranker import EvidenceItem, get_reranker
from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Load agent configs from YAML
# ─────────────────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "agents.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["agents"]


_CFG = _load_config()


def _agent_kwargs(name: str) -> dict:
    cfg = _CFG.get(name, {})
    return {
        "model": cfg.get("model", settings.fast_model),
        "instruction": cfg.get("instruction", ""),
        "generate_content_config": {
            "temperature": cfg.get("temperature", 0.1),
            "max_output_tokens": cfg.get("max_output_tokens", 4096),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Query Rewriter (BaseAgent with intelligent fast pass-through)
# ─────────────────────────────────────────────────────────────────────────────
class QueryRewriterAgent(BaseAgent):
    """
    Rewrites the user's question into a standalone search query.
    If it is the first turn or query is self-contained (no ambiguous pronouns),
    passes it directly to preserve LLM quota, prevent topic bleeding, and minimize latency.
    """

    name: str = "query_rewriter_agent"
    output_key: str = "rewritten_query"

    async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
        import re

        # Extract the latest user query from the newest user event
        query = ""
        user_events = []
        for event in (ctx.session.events or []):
            if getattr(event, "author", "") == "user":
                user_events.append(event)
                if event.content and event.content.parts and event.content.parts[0].text:
                    query = event.content.parts[0].text

        if not query:
            query = ctx.session.state.get("user_query", "")

        # Check if the query has ambiguous pronouns that refer back to previous conversation turns
        has_pronouns = bool(re.search(r"\b(he|she|it|they|him|her|them|his|their|its|this|that|these|those)\b", query, re.IGNORECASE))

        # Single-turn or clear self-contained question: pass through directly
        if len(user_events) <= 1 or not has_pronouns or not query:
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "user_query": query,
                    "rewritten_query": query,
                }),
            )
            return

        # Multi-turn conversation with ambiguous pronouns: rewrite using fast model
        rewritten = query
        try:
            from google import genai
            client = genai.Client(api_key=settings.google_api_key)
            history_lines = []
            for e in user_events[:-1]:
                txt = e.content.parts[0].text if e.content and e.content.parts and e.content.parts[0].text else ""
                if txt:
                    history_lines.append(f"User: {txt}")

            history_txt = "\n".join(history_lines[-4:])
            prompt = (
                f"Rewrite the last user query into a clear, standalone search query by resolving pronouns from the history:\n\n"
                f"{history_txt}\n\nLast Query: {query}\n\n"
                "Return only the standalone query string without quotes or preamble:"
            )
            resp = client.models.generate_content(
                model=_CFG["query_rewriter_agent"]["model"],
                contents=prompt,
            )
            cleaned = resp.text.strip().strip('"').strip("'").strip()
            if cleaned:
                rewritten = cleaned
        except Exception as e:
            logger.warning("QueryRewriterAgent fallback to original query: {}", e)
            rewritten = query

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "user_query": query,
                "rewritten_query": rewritten,
            }),
        )


query_rewriter_agent = QueryRewriterAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Local RAG Agent (Hybrid Search in Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
class LocalRagAgent(BaseAgent):
    """
    Performs hybrid dense+BM25 search in Qdrant with RRF fusion.
    Directly writes candidate chunks to session state rag_evidence.
    Consumes zero LLM tokens for maximum reliability and speed.
    """

    name: str = "local_rag_agent"
    tenant_id: str = "default"
    output_key: str = "rag_evidence"

    async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
        state = ctx.session.state
        query = state.get("rewritten_query") or state.get("user_query", "")
        if not query:
            for event in reversed(ctx.session.events or []):
                if event.content and event.content.parts and event.author == "user":
                    query = event.content.parts[0].text
                    break

        try:
            results = await hybrid_search_tool(
                query=query,
                tenant_id=self.tenant_id,
                top_k=settings.rag_top_k_retrieve,
            )
        except Exception as e:
            logger.warning(f"LocalRagAgent search error: {e}")
            results = []
        logger.info(
            f"LocalRagAgent: retrieved {len(results)} chunks for query: '{query[:60]}...'"
        )
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "rag_evidence": results,
                "user_query": query,
            }),
        )


local_rag_agent = LocalRagAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Reranker Agent (Deterministic Cross-Encoder — no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────
class RerankerAgent(BaseAgent):
    """
    Reads rag_evidence + mcp_evidence from session state,
    runs Cross-Encoder reranking (deterministic, no LLM),
    writes top-K results to reranked_evidence and formatted_evidence.
    """

    name: str = "reranker_agent"

    def __init__(self, **kwargs):
        super().__init__(name="reranker_agent", **kwargs)
        self._reranker = get_reranker()

    async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
        state = ctx.session.state
        query = state.get("rewritten_query") or state.get("user_query", "")

        raw_rag = state.get("rag_evidence", [])
        raw_mcp = state.get("mcp_evidence", [])

        # Normalize raw_rag (handles list, dict, or JSON/plain string)
        if isinstance(raw_rag, str):
            try:
                parsed = json.loads(raw_rag)
                raw_rag = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                raw_rag = [{"text": raw_rag, "source_name": "Knowledge Base", "source_uri": "local://rag"}]
        elif isinstance(raw_rag, dict):
            raw_rag = [raw_rag]
        elif not isinstance(raw_rag, list):
            raw_rag = []

        # Normalize raw_mcp
        if isinstance(raw_mcp, str):
            try:
                parsed = json.loads(raw_mcp)
                raw_mcp = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                raw_mcp = [{"text": raw_mcp, "source_name": "MCP Tool", "source_uri": "mcp://tools"}]
        elif isinstance(raw_mcp, dict):
            raw_mcp = [raw_mcp]
        elif not isinstance(raw_mcp, list):
            raw_mcp = []

        items: list[EvidenceItem] = []
        for r in raw_rag:
            if isinstance(r, dict):
                items.append(EvidenceItem(
                    text=r.get("text", "") or str(r),
                    source_name=r.get("source_name", "Knowledge Base"),
                    source_uri=r.get("source_uri", "local://rag"),
                    origin_type="rag",
                    page_number=r.get("page_number"),
                    section_heading=r.get("section_heading"),
                    chunk_id=r.get("chunk_id"),
                ))
            elif isinstance(r, str) and r.strip():
                items.append(EvidenceItem(
                    text=r,
                    source_name="Knowledge Base",
                    source_uri="local://rag",
                    origin_type="rag",
                ))

        for m in raw_mcp:
            if isinstance(m, dict):
                items.append(EvidenceItem(
                    text=m.get("text", "") or str(m),
                    source_name=m.get("source_name", "MCP Tool"),
                    source_uri=m.get("source_uri", "mcp://tools"),
                    origin_type="mcp",
                    mcp_server_name=m.get("mcp_server_name"),
                ))
            elif isinstance(m, str) and m.strip():
                items.append(EvidenceItem(
                    text=m,
                    source_name="MCP Tool",
                    source_uri="mcp://tools",
                    origin_type="mcp",
                ))

        if not items:
            logger.warning("RerankerAgent: no evidence to rerank")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "reranked_evidence": [],
                    "formatted_evidence": "No matching evidence found in knowledge base.",
                }),
            )
            return

        try:
            reranked = await self._reranker.rerank(query=query, items=items)
        except Exception as e:
            logger.warning(f"RerankerAgent error: {e}, falling back to top candidates")
            reranked = items[:settings.rag_top_k_final]

        reranked_dicts = [
            {
                "text": item.text,
                "source_name": item.source_name,
                "source_uri": item.source_uri,
                "origin_type": item.origin_type,
                "page_number": item.page_number,
                "section_heading": item.section_heading,
                "chunk_id": item.chunk_id,
                "mcp_server_name": item.mcp_server_name,
                "rerank_score": item.rerank_score,
            }
            for item in reranked
        ]

        formatted_items = []
        for i, item in enumerate(reranked):
            formatted_items.append(
                f"[{i+1}] Source: {item.source_name} ({item.source_uri})\nExcerpt: {item.text}"
            )
        formatted_str = "\n\n".join(formatted_items) if formatted_items else "No matching evidence found."

        logger.info(f"RerankerAgent: {len(items)} items → {len(reranked_dicts)} after reranking")
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "reranked_evidence": reranked_dicts,
                "formatted_evidence": formatted_str,
            }),
        )


reranker_agent = RerankerAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4a: Grounded Synthesizer
# ─────────────────────────────────────────────────────────────────────────────
grounded_synthesizer = LlmAgent(
    name="grounded_synthesizer",
    output_key="draft_answer",
    **_agent_kwargs("grounded_synthesizer"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b: Verifier Agent (LoopAgent validator)
# ─────────────────────────────────────────────────────────────────────────────
class VerifierAgent(BaseAgent):
    """
    Checks if draft_answer claims are supported by reranked_evidence.
    Sets session.state["grounding_passed"] = True/False.
    Guarantees session.state["final_answer"] is ALWAYS populated with draft_answer.
    """

    name: str = "verifier_agent"

    async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
        state = ctx.session.state
        draft = state.get("draft_answer", "")
        evidence = state.get("formatted_evidence", "")

        passed = True
        if draft and evidence and evidence != "No matching evidence found in knowledge base.":
            try:
                from google import genai
                client = genai.Client(api_key=settings.google_api_key)
                prompt = (
                    f"Evidence:\n{evidence[:25000]}\n\n"
                    f"Draft Answer:\n{draft}\n\n"
                    "Check if the factual claims in the draft answer are directly supported by the evidence.\n"
                    "If the answer accurately reflects facts, scores, numbers, or names found in the evidence, return pass as true.\n"
                    "Return JSON only: {\"pass\": true} or {\"pass\": false}"
                )
                resp = client.models.generate_content(
                    model=_CFG["verifier_agent"]["model"],
                    contents=prompt,
                )
                clean = resp.text.strip().strip("`").removeprefix("json").removeprefix("JSON").strip()
                result = json.loads(clean)
                passed = result.get("pass", True)
            except Exception as e:
                logger.warning("VerifierAgent skipped check due to: {}, defaulting to True", e)
                passed = True

        updates: dict = {
            "grounding_passed": passed,
            "final_answer": draft,
        }

        yield Event(
            author=self.name,
            actions=EventActions(
                escalate=passed,
                state_delta=updates,
            ),
        )


verifier_agent = VerifierAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Synthesis Loop (max 2 iterations)
# ─────────────────────────────────────────────────────────────────────────────
synthesis_loop = LoopAgent(
    name="synthesis_loop",
    sub_agents=[grounded_synthesizer, verifier_agent],
    max_iterations=2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Content Summarizer Agent
# ─────────────────────────────────────────────────────────────────────────────
content_summarizer_agent = LlmAgent(
    name="content_summarizer_agent",
    output_key="summary",
    **_agent_kwargs("content_summarizer_agent"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search Tool (called by LocalRagAgent)
# ─────────────────────────────────────────────────────────────────────────────
async def hybrid_search_tool(query: str, tenant_id: str, top_k: int = 40) -> list[dict]:
    """
    Performs hybrid dense+BM25 search in Qdrant with RRF fusion.
    Returns a list of evidence dicts for the reranker.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

    from ingestion.embedder import get_embedder
    from ingestion.indexer import HybridIndexer

    indexer = HybridIndexer()
    embedder = get_embedder()

    dense_vec = (await embedder.embed([query]))[0]

    # Dense search via modern Qdrant query_points
    dense_resp = await indexer.client.query_points(
        collection_name=indexer.collection,
        query=dense_vec,
        using="dense",
        limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]),
        with_payload=True,
    )
    dense_results = dense_resp.points or []

    # Sparse BM25 search
    sparse = indexer._bm25_sparse(query)
    sparse_resp = await indexer.client.query_points(
        collection_name=indexer.collection,
        query=SparseVector(indices=sparse["indices"], values=sparse["values"]),
        using="sparse",
        limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]),
        with_payload=True,
    )
    sparse_results = sparse_resp.points or []

    # RRF fusion
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    def rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    for rank, r in enumerate(dense_results):
        pid = str(r.id)
        scores[pid] = scores.get(pid, 0) + settings.rag_dense_weight * rrf(rank)
        payloads[pid] = r.payload

    for rank, r in enumerate(sparse_results):
        pid = str(r.id)
        scores[pid] = scores.get(pid, 0) + settings.rag_bm25_weight * rrf(rank)
        if pid not in payloads:
            payloads[pid] = r.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "chunk_id": pid,
            "text": payloads[pid].get("text", ""),
            "source_name": payloads[pid].get("source_name", ""),
            "source_uri": payloads[pid].get("source_uri", ""),
            "page_number": payloads[pid].get("page_number"),
            "section_heading": payloads[pid].get("section_heading"),
            "raw_score": score,
        }
        for pid, score in ranked
        if payloads.get(pid)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Agent Builder Functions (called at request time with live MCP toolsets)
# ─────────────────────────────────────────────────────────────────────────────
def build_local_rag_agent(tenant_id: str) -> LocalRagAgent:
    return LocalRagAgent(name="local_rag_agent", tenant_id=tenant_id)


def build_mcp_agent(mcp_toolsets: list) -> LlmAgent:
    return LlmAgent(
        name="dynamic_mcp_agent",
        output_key="mcp_evidence",
        tools=mcp_toolsets,
        **_agent_kwargs("dynamic_mcp_agent"),
    )


def build_query_rewriter_agent() -> QueryRewriterAgent:
    return QueryRewriterAgent()


def build_reranker_agent() -> RerankerAgent:
    return RerankerAgent()


def build_synthesis_loop() -> LoopAgent:
    synthesizer = LlmAgent(
        name="grounded_synthesizer",
        output_key="draft_answer",
        **_agent_kwargs("grounded_synthesizer"),
    )
    return LoopAgent(
        name="synthesis_loop",
        sub_agents=[synthesizer, VerifierAgent()],
        max_iterations=2,
    )


def build_search_pipeline(tenant_id: str, mcp_toolsets: list) -> SequentialAgent:
    gather_sub_agents = [build_local_rag_agent(tenant_id)]
    if mcp_toolsets:
        gather_sub_agents.append(build_mcp_agent(mcp_toolsets))

    evidence_gatherer = ParallelAgent(
        name="parallel_evidence_gatherer",
        sub_agents=gather_sub_agents,
    )

    return SequentialAgent(
        name="unified_search_pipeline",
        sub_agents=[
            build_query_rewriter_agent(),
            evidence_gatherer,
            build_reranker_agent(),
            build_synthesis_loop(),
        ],
    )


def build_root_orchestrator(tenant_id: str, mcp_toolsets: list) -> LlmAgent:
    search_pipeline = build_search_pipeline(tenant_id, mcp_toolsets)
    return LlmAgent(
        name="platform_orchestrator",
        tools=[
            load_memory,
            AgentTool(agent=search_pipeline),
            AgentTool(agent=content_summarizer_agent),
        ],
        **_agent_kwargs("platform_orchestrator"),
    )
