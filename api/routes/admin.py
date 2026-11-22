"""
Admin route — MCP server registration & agent config management.
"""

from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import MCPActionScope, MCPAuthType, MCPServer, MCPTransport, QueryTrace
from config.settings import settings
from mcp_registry.factory import encrypt_secret

router = APIRouter()


# ── MCP Server Models ────────────────────────────────────────────────────────
class MCPServerCreate(BaseModel):
    name: str
    description: str = ""
    transport_type: str = MCPTransport.STREAMABLE_HTTP
    endpoint_url: Optional[str] = None
    stdio_command: Optional[str] = None
    auth_type: str = MCPAuthType.NONE
    auth_secret: Optional[str] = None      # plaintext — encrypted at rest
    action_scope: str = MCPActionScope.READ_ONLY   # read_only | read_write (future)
    allowed_tools: Optional[list[str]] = None
    blocked_tools: Optional[list[str]] = None
    tenant_id: str = settings.default_tenant_id


class MCPServerUpdate(BaseModel):
    description: Optional[str] = None
    enabled: Optional[bool] = None
    auth_secret: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    blocked_tools: Optional[list[str]] = None
    action_scope: Optional[str] = None


# ── Register MCP Server ──────────────────────────────────────────────────────
@router.post("/mcp/servers")
async def register_mcp_server(
    payload: MCPServerCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new external MCP server. Validates connectivity before saving."""

    # Validate connectivity — list available tools
    cached_schemas = await _probe_mcp_server(payload)

    encrypted_secret = encrypt_secret(payload.auth_secret) if payload.auth_secret else None

    # Check for existing registration under same tenant and name (idempotent upsert)
    stmt = select(MCPServer).where(
        MCPServer.tenant_id == payload.tenant_id,
        MCPServer.name == payload.name,
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.description = payload.description or existing.description
        existing.transport_type = payload.transport_type
        existing.endpoint_url = payload.endpoint_url
        existing.stdio_command = payload.stdio_command
        existing.auth_type = payload.auth_type
        if payload.auth_secret:
            existing.auth_secret_encrypted = encrypted_secret
        existing.action_scope = payload.action_scope
        existing.allowed_tools = payload.allowed_tools
        existing.blocked_tools = payload.blocked_tools
        existing.cached_tool_schemas = cached_schemas
        existing.enabled = True
        server = existing
    else:
        server = MCPServer(
            tenant_id=payload.tenant_id,
            name=payload.name,
            description=payload.description,
            transport_type=payload.transport_type,
            endpoint_url=payload.endpoint_url,
            stdio_command=payload.stdio_command,
            auth_type=payload.auth_type,
            auth_secret_encrypted=encrypted_secret,
            action_scope=payload.action_scope,
            allowed_tools=payload.allowed_tools,
            blocked_tools=payload.blocked_tools,
            cached_tool_schemas=cached_schemas,
            enabled=True,
        )
        db.add(server)

    await db.commit()
    await db.refresh(server)

    return {
        "id": server.id,
        "name": server.name,
        "tools_discovered": len(cached_schemas),
        "tool_names": [t["name"] for t in cached_schemas],
        "action_scope": server.action_scope,
        "message": "MCP server registered successfully" if not existing else "MCP server updated successfully",
    }


@router.get("/mcp/servers")
async def list_mcp_servers(
    tenant_id: str = settings.default_tenant_id,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MCPServer).where(MCPServer.tenant_id == tenant_id)
    )
    servers = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "transport_type": s.transport_type,
            "endpoint_url": s.endpoint_url,
            "auth_type": s.auth_type,
            "action_scope": s.action_scope,
            "enabled": s.enabled,
            "tool_count": len(s.cached_tool_schemas or []),
            "tools": [t["name"] for t in (s.cached_tool_schemas or [])],
        }
        for s in servers
    ]


@router.patch("/mcp/servers/{server_id}")
async def update_mcp_server(
    server_id: str,
    payload: MCPServerUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MCPServer).where(MCPServer.id == server_id))
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    if payload.description is not None:
        server.description = payload.description
    if payload.enabled is not None:
        server.enabled = payload.enabled
    if payload.auth_secret is not None:
        server.auth_secret_encrypted = encrypt_secret(payload.auth_secret)
    if payload.allowed_tools is not None:
        server.allowed_tools = payload.allowed_tools
    if payload.blocked_tools is not None:
        server.blocked_tools = payload.blocked_tools
    if payload.action_scope is not None:
        server.action_scope = payload.action_scope

    await db.commit()
    return {"message": "Updated", "id": server_id}


@router.delete("/mcp/servers/{server_id}")
async def delete_mcp_server(server_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MCPServer).where(MCPServer.id == server_id))
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    await db.delete(server)
    await db.commit()
    return {"message": "Deleted", "id": server_id}


@router.post("/mcp/servers/{server_id}/test")
async def test_mcp_server(server_id: str, db: AsyncSession = Depends(get_db)):
    """Re-probe an existing MCP server and refresh cached tool schemas."""
    result = await db.execute(select(MCPServer).where(MCPServer.id == server_id))
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    class _Payload:
        transport_type = server.transport_type
        endpoint_url = server.endpoint_url
        stdio_command = server.stdio_command
        auth_secret = None  # don't re-encrypt

    schemas = await _probe_mcp_server(_Payload())
    server.cached_tool_schemas = schemas
    from datetime import datetime
    server.tools_last_refreshed_at = datetime.utcnow()
    await db.commit()

    return {
        "tools_discovered": len(schemas),
        "tool_names": [t["name"] for t in schemas],
    }


# ── Helper: probe MCP server ─────────────────────────────────────────────────
async def _probe_mcp_server(payload) -> list[dict]:
    """Connect to the MCP server and retrieve its tool list."""
    try:
        from google.adk.tools.mcp_tool import (
            McpToolset,
            SseConnectionParams,
            StdioConnectionParams,
        )

        if payload.transport_type in ("streamable_http", "sse"):
            conn = SseConnectionParams(url=payload.endpoint_url or "")
        elif payload.transport_type == "stdio":
            parts = (payload.stdio_command or "").split()
            if not parts:
                return []
            try:
                from mcp.client.stdio import StdioServerParameters
                server_params = StdioServerParameters(command=parts[0], args=parts[1:])
                conn = StdioConnectionParams(server_params=server_params)
            except Exception:
                conn = StdioConnectionParams(command=parts[0], args=parts[1:])
        else:
            return []

        toolset = McpToolset(connection_params=conn)
        tools = await toolset.get_tools() if hasattr(toolset, "get_tools") else await toolset.load_tools()
        return [{"name": t.name, "description": t.description or ""} for t in tools]
    except Exception as e:
        logger.warning(f"MCP server probe failed: {e}")
        return []


# ── Query Traces & Observability Endpoints ────────────────────────────────────
@router.get("/traces")
async def list_query_traces(
    tenant_id: str = settings.default_tenant_id,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List execution query traces for audit, latency monitoring, and evaluation."""
    stmt = (
        select(QueryTrace)
        .where(QueryTrace.tenant_id == tenant_id)
        .order_by(QueryTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    traces = result.scalars().all()
    return [
        {
            "id": t.id,
            "session_id": t.session_id,
            "user_id": t.user_id,
            "user_query": t.user_query,
            "rewritten_query": t.rewritten_query,
            "final_answer": (t.final_answer[:120] + "...") if t.final_answer and len(t.final_answer) > 120 else t.final_answer,
            "grounding_passed": t.grounding_passed,
            "total_latency_ms": t.total_latency_ms,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in traces
    ]


@router.get("/traces/{trace_id}")
async def get_query_trace(trace_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve full audit trace record including raw retrieved chunks and rerank scores."""
    result = await db.execute(select(QueryTrace).where(QueryTrace.id == trace_id))
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Query trace not found")

    return {
        "id": trace.id,
        "tenant_id": trace.tenant_id,
        "session_id": trace.session_id,
        "user_id": trace.user_id,
        "user_query": trace.user_query,
        "rewritten_query": trace.rewritten_query,
        "routing_decision": trace.routing_decision,
        "rag_evidence": trace.rag_evidence,
        "mcp_evidence": trace.mcp_evidence,
        "reranked_evidence": trace.reranked_evidence,
        "final_answer": trace.final_answer,
        "grounding_passed": trace.grounding_passed,
        "total_latency_ms": trace.total_latency_ms,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "reranker_scores": trace.reranker_scores,
        "created_at": trace.created_at.isoformat() if trace.created_at else None,
    }


# ── Agent Configuration Endpoints ─────────────────────────────────────────────
@router.get("/config/agents")
async def get_agent_configurations():
    """Retrieve all configurable agent instructions, models, and hyperparameters."""
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Agent configuration file not found")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("agents", {})


@router.get("/config/ui")
async def get_ui_configuration():
    """Retrieve UI client configuration flags and default settings."""
    return {
        "show_evidence_sources": settings.show_evidence_sources,
        "default_tenant_id": settings.default_tenant_id,
    }


@router.patch("/config/agents")
async def update_agent_configurations(payload: dict):
    """
    Update agent models, temperatures, max tokens, or system instructions
    without code redeployment.
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Agent configuration file not found")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    agents = data.setdefault("agents", {})
    for agent_name, updates in payload.items():
        if agent_name in agents and isinstance(updates, dict):
            agents[agent_name].update(updates)
        elif isinstance(updates, dict):
            agents[agent_name] = updates

    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    return {"message": "Agent configurations updated successfully", "agents": agents}
