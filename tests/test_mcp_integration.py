"""
Comprehensive MCP Integration Tests.
Covers:
  - AES credential encryption/decryption roundtrip
  - MCPToolsetFactory toolset construction across transports (stdio, sse, streamable_http)
  - Auth header generation (Bearer, ApiKey, Basic)
  - Semantic MCP Tool Router (top-N selection)
  - Full Admin API CRUD and live health probe endpoints with in-memory SQLite
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from api.main import app
from api.database import get_db
from api.models import Base, MCPServer, MCPTransport, MCPAuthType, MCPActionScope
from mcp_registry.factory import (
    MCPToolsetFactory,
    encrypt_secret,
    decrypt_secret,
)


@pytest_asyncio.fixture
async def sqlite_engine():
    """Create in-memory SQLite database engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(sqlite_engine):
    """Provide isolated session per test."""
    session_factory = async_sessionmaker(bind=sqlite_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


def test_mcp_credential_encryption_roundtrip():
    """Verify AES-256 Fernet encrypt/decrypt roundtrip."""
    secret = "ghp_SuperSecretGithubPersonalAccessToken12345!"
    encrypted = encrypt_secret(secret)
    assert encrypted != secret
    decrypted = decrypt_secret(encrypted)
    assert decrypted == secret


def test_mcp_auth_headers_generation():
    """Verify auth headers for Bearer, ApiKey, and Basic."""
    factory = MCPToolsetFactory()

    # Bearer
    s_bearer = MagicMock()
    s_bearer.auth_type = MCPAuthType.BEARER
    s_bearer.auth_secret_encrypted = encrypt_secret("my_token")
    headers = factory._build_auth_headers(s_bearer)
    assert headers == {"Authorization": "Bearer my_token"}

    # API Key
    s_key = MagicMock()
    s_key.auth_type = MCPAuthType.API_KEY
    s_key.auth_secret_encrypted = encrypt_secret("api_key_xyz")
    headers_key = factory._build_auth_headers(s_key)
    assert headers_key == {"X-API-Key": "api_key_xyz"}

    # Basic Auth
    s_basic = MagicMock()
    s_basic.auth_type = "basic"
    s_basic.auth_secret_encrypted = encrypt_secret("user:pass123")
    headers_basic = factory._build_auth_headers(s_basic)
    assert "Authorization" in headers_basic
    assert headers_basic["Authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_mcp_toolset_construction_transports():
    """Verify MCPToolset is built with proper connection parameters for stdio and sse."""
    factory = MCPToolsetFactory()

    # Stdio transport
    s_stdio = MagicMock()
    s_stdio.transport_type = MCPTransport.STDIO
    s_stdio.stdio_command = "npx -y @modelcontextprotocol/server-jira"
    s_stdio.action_scope = "read_only"
    s_stdio.allowed_tools = None
    s_stdio.blocked_tools = []
    s_stdio.cached_tool_schemas = [{"name": "jira_search", "description": "search"}]
    s_stdio.auth_type = MCPAuthType.NONE
    s_stdio.auth_secret_encrypted = None

    toolset_stdio = await factory._build_single_toolset(s_stdio)
    assert toolset_stdio is not None
    conn_stdio = toolset_stdio.connection_params
    cmd = conn_stdio.server_params.command if hasattr(conn_stdio, "server_params") else conn_stdio.command
    args = conn_stdio.server_params.args if hasattr(conn_stdio, "server_params") else conn_stdio.args
    assert cmd == "npx"
    assert args == ["-y", "@modelcontextprotocol/server-jira"]

    # SSE / Streamable HTTP transport
    s_sse = MagicMock()
    s_sse.transport_type = MCPTransport.STREAMABLE_HTTP
    s_sse.endpoint_url = "https://mcp.internal.acme.com/sse"
    s_sse.action_scope = "read_only"
    s_sse.allowed_tools = None
    s_sse.blocked_tools = []
    s_sse.cached_tool_schemas = [{"name": "fetch_doc", "description": "fetch"}]
    s_sse.auth_type = MCPAuthType.BEARER
    s_sse.auth_secret_encrypted = encrypt_secret("bearer_token_123")

    toolset_sse = await factory._build_single_toolset(s_sse)
    assert toolset_sse is not None
    conn_sse = toolset_sse.connection_params
    assert conn_sse.url == "https://mcp.internal.acme.com/sse"
    assert conn_sse.headers == {"Authorization": "Bearer bearer_token_123"}


@pytest.mark.asyncio
async def test_mcp_semantic_tool_router():
    """Verify _route_servers selects the top-N most semantically relevant servers."""
    factory = MCPToolsetFactory()

    s1 = MagicMock(name="jira", description="Jira issue tracker and agile project management")
    s2 = MagicMock(name="github", description="GitHub source code repository and pull requests")
    s3 = MagicMock(name="postgres", description="PostgreSQL relational database queries and schemas")
    s4 = MagicMock(name="slack", description="Slack team messaging and notifications")

    # Mock embedder to return deterministic vectors
    mock_embedder = MagicMock()
    # Let query match Jira best
    mock_embedder.embed_text = AsyncMock(side_effect=[
        [1.0, 0.0, 0.0],  # query: "issue tickets"
        [0.9, 0.1, 0.0],  # s1 jira
        [0.1, 0.9, 0.0],  # s2 github
        [0.0, 0.1, 0.9],  # s3 postgres
        [0.0, 0.0, 0.1],  # s4 slack
    ])

    with patch("ingestion.embedder.get_embedder", return_value=mock_embedder):
        with patch("config.settings.settings.mcp_max_servers_per_query", 2):
            routed = await factory._route_servers("search issue tickets", [s1, s2, s3, s4])
            assert len(routed) == 2
            assert routed[0] == s1


@pytest.mark.asyncio
async def test_mcp_admin_api_endpoints_full_lifecycle(sqlite_engine, db_session):
    """Verify complete CRUD and probe lifecycle for MCP servers via Admin API on SQLite."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    mock_probed_tools = [
        {"name": "confluence_search", "description": "Search pages"},
        {"name": "confluence_get_page", "description": "Get page content"},
    ]

    with patch("api.routes.admin._probe_mcp_server", return_value=mock_probed_tools):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # 1. Register new MCP server
            reg_resp = await client.post(
                "/api/admin/mcp/servers",
                json={
                    "name": "confluence_wiki",
                    "description": "Internal wiki documentation",
                    "transport_type": "streamable_http",
                    "endpoint_url": "https://wiki.acme.com/mcp",
                    "auth_type": "bearer",
                    "auth_secret": "secret_wiki_token",
                    "action_scope": "read_only",
                },
            )
            assert reg_resp.status_code == 200
            reg_data = reg_resp.json()
            assert reg_data["name"] == "confluence_wiki"
            assert reg_data["tools_discovered"] == 2
            server_id = reg_data["id"]

            # 2. List MCP servers
            list_resp = await client.get("/api/admin/mcp/servers")
            assert list_resp.status_code == 200
            servers = list_resp.json()
            assert len(servers) == 1
            assert servers[0]["name"] == "confluence_wiki"
            assert servers[0]["tool_count"] == 2

            # 3. Test & re-probe connectivity
            test_resp = await client.post(f"/api/admin/mcp/servers/{server_id}/test")
            assert test_resp.status_code == 200
            assert test_resp.json()["tools_discovered"] == 2

            # 4. Update action scope and disabled
            patch_resp = await client.patch(
                f"/api/admin/mcp/servers/{server_id}",
                json={"action_scope": "read,create", "enabled": False},
            )
            assert patch_resp.status_code == 200

            # Verify updated values in list
            list_resp2 = await client.get("/api/admin/mcp/servers")
            assert list_resp2.json()[0]["action_scope"] == "read,create"
            assert list_resp2.json()[0]["enabled"] is False

            # 5. Delete MCP server
            del_resp = await client.delete(f"/api/admin/mcp/servers/{server_id}")
            assert del_resp.status_code == 200

            # Verify deletion
            list_resp3 = await client.get("/api/admin/mcp/servers")
            assert len(list_resp3.json()) == 0

    app.dependency_overrides.clear()
