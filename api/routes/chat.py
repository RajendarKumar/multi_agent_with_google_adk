"""
Chat route — SSE streaming chat with the platform orchestrator.
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import MCPServer
from api.tracing import record_query_trace
from config.settings import settings

router = APIRouter()


def sanitize_ai_error(e: Exception) -> str:
    """
    Translates low-level third-party model exceptions and raw status dumps
    (e.g., 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, JSON payloads) into clear,
    professional application-level user messages.
    """
    raw = str(e)
    # Check for 503 / High Demand / Service Unavailable
    if "503" in raw or "UNAVAILABLE" in raw or "high demand" in raw.lower():
        return (
            "The AI service is currently experiencing heavy traffic and high demand. "
            "Please try asking your question again in a few moments."
        )
    # Check for 429 / Rate limit / Quota exceeded
    if "429" in raw or "RESOURCE_EXHAUSTED" in raw or "quota" in raw.lower():
        return (
            "The system is currently receiving a high volume of requests. "
            "Please wait a moment before sending another message."
        )
    # Check for authentication / API key issues
    if "401" in raw or "403" in raw or "api_key" in raw.lower() or "permission_denied" in raw.lower():
        return (
            "The service could not authenticate with the AI provider. "
            "Please contact your administrator or check the system API configuration."
        )
    # Check for connection / timeout errors
    if "timeout" in raw.lower() or "connect" in raw.lower() or "connection" in raw.lower():
        return (
            "A temporary network timeout occurred while connecting to the AI service. "
            "Please try again."
        )
    # Generic fallback that never exposes raw third-party json or code tracebacks
    return (
        "We encountered a temporary service issue while processing your request. "
        "Please try again in a moment."
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    user_id: str = "anonymous"
    tenant_id: str = settings.default_tenant_id


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = []
    grounding_passed: bool = True
    latency_ms: Optional[int] = None
    trace_id: Optional[str] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Non-streaming chat endpoint."""
    result = await _run_agent(request, db=db)
    return result


@router.post("/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """SSE streaming chat — sends intermediate status events + final answer."""
    return StreamingResponse(
        _stream_agent(request, db=db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_agent(request: ChatRequest, db: AsyncSession) -> dict:
    from google.adk.runners import Runner
    from google.genai.types import Content, Part

    from agents.orchestrator import build_root_orchestrator
    from mcp_registry.factory import MCPToolsetFactory
    from memory.memory import LTMService, STMService

    start_time = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    tenant_id = request.tenant_id

    # ── 1. Load active MCP servers for this tenant ──────────────────────────
    res = await db.execute(
        select(MCPServer).where(
            MCPServer.tenant_id == tenant_id,
            MCPServer.enabled == True,
        )
    )
    mcp_servers = res.scalars().all()

    # ── 2. Build MCP toolsets ────────────────────────────────────────────────
    factory = MCPToolsetFactory()
    mcp_toolsets = await factory.build_toolsets(mcp_servers, query=request.message)

    # ── 3. Build orchestrator ────────────────────────────────────────────────
    orchestrator = build_root_orchestrator(tenant_id=tenant_id, mcp_toolsets=mcp_toolsets)

    # ── 4. Memory services ───────────────────────────────────────────────────
    stm = STMService()
    ltm = LTMService()

    runner = Runner(
        agent=orchestrator,
        app_name="responsive_agent",
        session_service=stm.get_service(),
        memory_service=ltm.get_service(),
    )

    # ── 5. Get or create ADK session ─────────────────────────────────────────
    await stm.get_or_create_session(
        session_id=session_id,
        user_id=request.user_id,
        tenant_id=tenant_id,
    )

    # Reset turn-level scratchpad to prevent previous conversation turns from polluting evidence
    await stm.update_state(
        session_id=session_id,
        user_id=request.user_id,
        tenant_id=tenant_id,
        updates={
            "user_query": request.message,
            "rewritten_query": request.message,
            "rag_evidence": [],
            "mcp_evidence": [],
            "reranked_evidence": [],
            "formatted_evidence": "",
            "draft_answer": "",
            "final_answer": "",
            "grounding_passed": True,
        },
    )

    # ── 6. Run agent ─────────────────────────────────────────────────────────
    user_message = Content(role="user", parts=[Part(text=request.message)])

    final_answer = ""
    grounding_passed = True
    agent_error = None

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            async for event in runner.run_async(
                user_id=f"{tenant_id}:{request.user_id}",
                session_id=session_id,
                new_message=user_message,
            ):
                if event.is_final_response():
                    final_answer = event.content.parts[0].text if event.content else ""
            agent_error = None
            break
        except Exception as e:
            err_text = str(e)
            if ("503" in err_text or "UNAVAILABLE" in err_text) and attempt < max_retries:
                logger.warning(f"Sync runner hit transient 503 on attempt {attempt+1}, retrying in 2s...")
                await asyncio.sleep(2.0)
                continue
            logger.error("Chat sync execution error: {}", e)
            agent_error = e
            break

    if agent_error is not None:
        final_answer = sanitize_ai_error(agent_error)
        grounding_passed = False

    # ── 7. Read state after run ───────────────────────────────────────────────
    refreshed = await stm.get_or_create_session(
        session_id=session_id,
        user_id=request.user_id,
        tenant_id=tenant_id,
    )
    state = refreshed.state if refreshed else {}
    sources = state.get("reranked_evidence", [])
    grounding_passed = state.get("grounding_passed", True)
    final_answer = final_answer or state.get("final_answer") or state.get("draft_answer") or ""
    if not final_answer:
        if sources:
            final_answer = f"Retrieved {len(sources)} relevant evidence sources from the knowledge base, but could not synthesize a text answer."
        else:
            final_answer = "I don't have enough information in the ingested knowledge base to answer that confidently."
    total_latency_ms = int((time.perf_counter() - start_time) * 1000)

    # ── 8. Persist to LTM ────────────────────────────────────────────────────
    try:
        await ltm.persist_session(refreshed, runner)
    except Exception:
        pass  # LTM persistence is best-effort

    # ── 9. Record Query Trace ────────────────────────────────────────────────
    trace_id = await record_query_trace(
        tenant_id=tenant_id,
        session_id=session_id,
        user_id=request.user_id,
        user_query=request.message,
        rewritten_query=state.get("rewritten_query"),
        routing_decision=state.get("routing_decision", "search"),
        rag_evidence=state.get("rag_evidence"),
        mcp_evidence=state.get("mcp_evidence"),
        reranked_evidence=sources,
        final_answer=final_answer,
        grounding_passed=grounding_passed,
        total_latency_ms=total_latency_ms,
        db=db,
    )

    return {
        "session_id": session_id,
        "answer": final_answer,
        "sources": sources,
        "grounding_passed": grounding_passed,
        "latency_ms": total_latency_ms,
        "trace_id": trace_id,
    }


async def _stream_agent(
    request: ChatRequest, db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    Yield SSE events:
      - 'status'  — intermediate progress messages
      - 'answer'  — final answer with sources
      - 'error'   — on failure
      - 'done'    — always last
    """

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    start_time = time.perf_counter()
    yield sse("status", {"message": "🔍 Analysing your query..."})

    try:
        from google.adk.runners import Runner
        from google.genai.types import Content, Part

        from agents.orchestrator import build_root_orchestrator
        from mcp_registry.factory import MCPToolsetFactory
        from memory.memory import LTMService, STMService

        session_id = request.session_id or str(uuid.uuid4())
        tenant_id = request.tenant_id

        # Load MCP servers
        res = await db.execute(
            select(MCPServer).where(
                MCPServer.tenant_id == tenant_id,
                MCPServer.enabled == True,
            )
        )
        mcp_servers = res.scalars().all()

        yield sse("status", {"message": "🔌 Loading MCP toolsets..."})
        factory = MCPToolsetFactory()
        mcp_toolsets = await factory.build_toolsets(mcp_servers, query=request.message)

        orchestrator = build_root_orchestrator(tenant_id=tenant_id, mcp_toolsets=mcp_toolsets)
        stm = STMService()
        ltm = LTMService()

        runner = Runner(
            agent=orchestrator,
            app_name="responsive_agent",
            session_service=stm.get_service(),
            memory_service=ltm.get_service(),
        )

        await stm.get_or_create_session(
            session_id=session_id,
            user_id=request.user_id,
            tenant_id=tenant_id,
        )

        # Reset turn-level scratchpad to prevent previous conversation turns from polluting evidence
        await stm.update_state(
            session_id=session_id,
            user_id=request.user_id,
            tenant_id=tenant_id,
            updates={
                "user_query": request.message,
                "rewritten_query": request.message,
                "rag_evidence": [],
                "mcp_evidence": [],
                "reranked_evidence": [],
                "formatted_evidence": "",
                "draft_answer": "",
                "final_answer": "",
                "grounding_passed": True,
            },
        )

        user_message = Content(role="user", parts=[Part(text=request.message)])
        final_answer = ""

        # Stream events from the runner with automatic retry on transient 503
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async for event in runner.run_async(
                    user_id=f"{tenant_id}:{request.user_id}",
                    session_id=session_id,
                    new_message=user_message,
                ):
                    # Emit intermediate status based on which agent is active
                    author = getattr(event, "author", "")
                    if author == "query_rewriter_agent":
                        yield sse("status", {"message": "✍️ Rewriting query..."})
                    elif author == "local_rag_agent":
                        yield sse("status", {"message": "📄 Searching documents..."})
                    elif author == "dynamic_mcp_agent":
                        yield sse("status", {"message": "🔌 Querying MCP servers..."})
                    elif author == "reranker_agent":
                        yield sse("status", {"message": "🏆 Reranking evidence..."})
                    elif author == "grounded_synthesizer":
                        yield sse("status", {"message": "✍️ Composing answer..."})
                    elif author == "verifier_agent":
                        yield sse("status", {"message": "✅ Verifying citations..."})

                    if event.content and event.content.parts and event.content.parts[0].text:
                        final_answer = event.content.parts[0].text
                break
            except Exception as stream_err:
                err_text = str(stream_err)
                if ("503" in err_text or "UNAVAILABLE" in err_text) and attempt < max_retries:
                    logger.warning(f"Streaming runner hit 503 on attempt {attempt+1}, retrying in 2s...")
                    yield sse("status", {"message": "⏳ AI service is temporarily busy, retrying automatically..."})
                    await asyncio.sleep(2.0)
                    continue
                raise stream_err

        # Read final state
        refreshed = await stm.get_or_create_session(
            session_id=session_id,
            user_id=request.user_id,
            tenant_id=tenant_id,
        )
        state = refreshed.state if refreshed else {}
        sources = state.get("reranked_evidence", [])
        grounding_passed = state.get("grounding_passed", True)
        
        # Robust resolution of final answer text
        final_answer = final_answer or state.get("final_answer") or state.get("draft_answer") or ""
        if not final_answer:
            if sources:
                final_answer = f"Retrieved {len(sources)} relevant evidence sources from the knowledge base, but could not synthesize a text answer."
            else:
                final_answer = "I don't have enough information in the ingested knowledge base to answer that confidently."
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        try:
            await ltm.persist_session(refreshed, runner)
        except Exception:
            pass

        # Record Query Trace
        trace_id = await record_query_trace(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=request.user_id,
            user_query=request.message,
            rewritten_query=state.get("rewritten_query"),
            routing_decision=state.get("routing_decision", "search"),
            rag_evidence=state.get("rag_evidence"),
            mcp_evidence=state.get("mcp_evidence"),
            reranked_evidence=sources,
            final_answer=final_answer,
            grounding_passed=grounding_passed,
            total_latency_ms=total_latency_ms,
            db=db,
        )

        yield sse("answer", {
            "session_id": session_id,
            "answer": final_answer,
            "chunk": final_answer,
            "text": final_answer,
            "sources": sources,
            "grounding_passed": grounding_passed,
            "latency_ms": total_latency_ms,
            "trace_id": trace_id,
        })

        yield sse("done", {
            "session_id": session_id,
            "final_answer": final_answer,
            "sources": sources,
            "grounding_passed": grounding_passed,
            "total_latency_ms": total_latency_ms,
            "trace_id": trace_id,
        })

    except Exception as e:
        logger.error("Chat streaming error: {}", e)
        user_friendly_error = sanitize_ai_error(e)
        yield sse("error", {"message": user_friendly_error, "detail": user_friendly_error})
        yield sse("done", {"error": user_friendly_error, "grounding_passed": False})
