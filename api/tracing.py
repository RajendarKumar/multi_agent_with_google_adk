"""
Query tracing and observability service.
Persists execution metrics, intermediate agent state, and evidence to PostgreSQL,
and optionally publishes distributed traces to Langfuse when configured and enabled.
"""

import time
from typing import Optional, Any
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AsyncSessionLocal
from api.models import QueryTrace, new_uuid
from config.settings import settings

_langfuse_client: Optional[Any] = None


def get_langfuse_client() -> Optional[Any]:
    """
    Lazily initialize and cache the Langfuse client if enabled and keys are configured.
    Returns None if Langfuse is disabled or dependencies/credentials are missing.
    """
    global _langfuse_client
    if not settings.is_langfuse_active:
        return None

    if _langfuse_client is None:
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse observability client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}")
            return None
    return _langfuse_client


def _publish_to_langfuse(
    tenant_id: str,
    session_id: str,
    user_id: Optional[str],
    user_query: str,
    rewritten_query: Optional[str],
    final_answer: str,
    routing_decision: Optional[str],
    rag_evidence: Optional[list[dict]],
    mcp_evidence: Optional[list[dict]],
    reranked_evidence: Optional[list[dict]],
    grounding_passed: Optional[bool],
    total_latency_ms: Optional[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    reranker_scores: Optional[dict[str, float]],
) -> None:
    """
    Dispatches a structured multi-span trace to Langfuse.
    Runs defensively without ever interrupting the main application flow.
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        from langfuse import propagate_attributes

        with client.start_as_current_observation(name="agentic_chat_pipeline", as_type="chain") as root_span:
            with propagate_attributes(
                user_id=user_id or "anonymous",
                session_id=session_id,
                trace_name="agentic_chat_pipeline",
                tags=[f"tenant:{tenant_id}", f"routing:{routing_decision or 'search'}"],
            ):
                root_span.update(
                    input={"user_query": user_query, "rewritten_query": rewritten_query},
                    output={"final_answer": final_answer},
                    metadata={
                        "tenant_id": tenant_id,
                        "latency_ms": total_latency_ms,
                        "grounding_passed": grounding_passed,
                    },
                )

                # 1. Query Rewriter Span
                if rewritten_query:
                    with client.start_as_current_observation(name="query_rewriter", as_type="span") as s_rewriter:
                        s_rewriter.update(
                            input={"user_query": user_query},
                            output={"rewritten_query": rewritten_query},
                        )

                # 2. Parallel Evidence Gathering Span
                with client.start_as_current_observation(name="evidence_gathering", as_type="retriever") as s_ret:
                    s_ret.update(
                        input=rewritten_query or user_query,
                        metadata={
                            "rag_count": len(rag_evidence) if rag_evidence else 0,
                            "mcp_count": len(mcp_evidence) if mcp_evidence else 0,
                        },
                    )

                # 3. Deterministic Reranker Span
                if reranked_evidence:
                    with client.start_as_current_observation(name="deterministic_reranker", as_type="span") as s_rerank:
                        s_rerank.update(
                            output={"reranked_count": len(reranked_evidence)},
                            metadata={"scores": reranker_scores or {}},
                        )

                # 4. Grounded Synthesizer Generation
                with client.start_as_current_observation(
                    name="grounded_synthesizer",
                    as_type="generation",
                    model="gemini-2.5-pro",
                    usage_details={
                        "input": input_tokens or 0,
                        "output": output_tokens or 0,
                        "total": (input_tokens or 0) + (output_tokens or 0),
                    },
                ) as s_gen:
                    s_gen.update(
                        input=rewritten_query or user_query,
                        output=final_answer,
                    )

                # 5. Verifier Agent Evaluator Span
                with client.start_as_current_observation(name="verifier_agent", as_type="evaluator") as s_ver:
                    s_ver.update(
                        output={"grounding_passed": grounding_passed}
                    )

        client.flush()
        logger.debug(f"Langfuse trace published for session {session_id}")

    except Exception as e:
        logger.warning(f"Failed to publish trace to Langfuse (non-fatal): {e}")


async def record_query_trace(
    tenant_id: str,
    session_id: str,
    user_query: str,
    final_answer: str,
    rewritten_query: Optional[str] = None,
    routing_decision: Optional[str] = None,
    rag_evidence: Optional[list[dict]] = None,
    mcp_evidence: Optional[list[dict]] = None,
    reranked_evidence: Optional[list[dict]] = None,
    grounding_passed: Optional[bool] = True,
    total_latency_ms: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    reranker_scores: Optional[dict[str, float]] = None,
    user_id: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> Optional[str]:
    """
    Persist an execution trace to the query_traces table, and optionally
    publish to Langfuse if configured and enabled.
    Can use an existing AsyncSession or open a dedicated session.
    Fails safely without raising exceptions to protect the chat flow.
    """
    trace_id = None
    try:
        trace_id = new_uuid()
        trace = QueryTrace(
            id=trace_id,
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id or "anonymous",
            user_query=user_query,
            rewritten_query=rewritten_query,
            routing_decision=routing_decision or "search",
            rag_evidence={"items": rag_evidence} if rag_evidence else None,
            mcp_evidence={"items": mcp_evidence} if mcp_evidence else None,
            reranked_evidence={"items": reranked_evidence} if reranked_evidence else None,
            final_answer=final_answer,
            grounding_passed=grounding_passed,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens or (len(user_query.split()) * 2),
            output_tokens=output_tokens or (len(final_answer.split()) * 2 if final_answer else 0),
            reranker_scores=reranker_scores,
        )

        trace_id = trace.id

        if db is not None:
            db.add(trace)
            await db.commit()
        else:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    session.add(trace)
                await session.commit()

        logger.debug(f"QueryTrace {trace_id} saved for session {session_id} ({total_latency_ms}ms)")

        # Publish trace to Langfuse if enabled
        _publish_to_langfuse(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            rewritten_query=rewritten_query,
            final_answer=final_answer,
            routing_decision=routing_decision,
            rag_evidence=rag_evidence,
            mcp_evidence=mcp_evidence,
            reranked_evidence=reranked_evidence,
            grounding_passed=grounding_passed,
            total_latency_ms=total_latency_ms,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            reranker_scores=reranker_scores,
        )

        return trace_id

    except Exception as e:
        logger.warning(f"Failed to record query trace (non-fatal): {e}")
        return None
