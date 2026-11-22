"""
MCP Server Registry — dynamic plug-and-play MCP connectivity.

Design:
  - MCP server configs stored in PostgreSQL (MCPServer table)
  - Credentials AES-256 encrypted at rest
  - Action scope: READ_ONLY by default; extend to READ_WRITE by flipping action_scope
  - MCPToolsetFactory builds google.adk McpToolset instances at runtime
  - Tool Router selects the top-N most relevant servers per query to avoid
    LLM tool-selection degradation with too many tools
"""

import base64
import json
from typing import Optional

from cryptography.fernet import Fernet
from loguru import logger

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Credential encryption (Fernet / AES-128-CBC)
# ─────────────────────────────────────────────────────────────────────────────
def _get_cipher() -> Fernet:
    key = settings.mcp_credential_encryption_key.encode()[:32]
    # Pad/truncate to 32 bytes, then base64-encode to Fernet key format
    padded = key.ljust(32, b"\x00")[:32]
    b64_key = base64.urlsafe_b64encode(padded)
    return Fernet(b64_key)


def encrypt_secret(plaintext: str) -> str:
    return _get_cipher().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    return _get_cipher().decrypt(ciphertext.encode()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# MCPToolsetFactory
# ─────────────────────────────────────────────────────────────────────────────
class MCPToolsetFactory:
    """
    Builds google.adk McpToolset instances from database MCPServer records.

    Extending for write actions:
      1. Change server.action_scope to MCPActionScope.READ_WRITE in the DB.
      2. The factory will expose all tools (not just read-only filtered ones).
      3. Add an MCPActionPolicy callback in _apply_action_policy() for HITL approval.

    No code changes needed in the orchestrator or search pipeline.
    """

    async def build_toolsets(
        self,
        server_records: list,  # list of MCPServer ORM objects
        query: Optional[str] = None,
    ) -> list:
        """
        Build McpToolset instances for active servers.
        If query is provided, the Tool Router pre-filters to the most relevant servers.
        """

        active_servers = [s for s in server_records if s.enabled]

        # Route: pick top-N relevant servers based on query + server descriptions
        if query and len(active_servers) > settings.mcp_max_servers_per_query:
            active_servers = await self._route_servers(query, active_servers)

        toolsets = []
        for server in active_servers:
            try:
                toolset = await self._build_single_toolset(server)
                if toolset:
                    toolsets.append(toolset)
            except Exception as e:
                logger.error(f"Failed to build toolset for MCP server '{server.name}': {e}")

        return toolsets

    async def _build_single_toolset(self, server) -> Optional[object]:
        try:
            from google.adk.tools.mcp_tool import (
                McpToolset,
                SseConnectionParams,
                StdioConnectionParams,
                StreamableHTTPConnectionParams,
            )
            SseParams = SseConnectionParams
            StdioParams = StdioConnectionParams
            ToolsetClass = McpToolset
        except (ImportError, AttributeError):
            # Fallback when official 'mcp' dependency is not installed in the environment
            class ToolsetClass:  # type: ignore
                def __init__(self, connection_params, tool_filter=None):
                    self.connection_params = connection_params
                    self.tool_filter = tool_filter

                async def get_tools(self):
                    return []

            class SseParams:  # type: ignore
                def __init__(self, url, headers=None):
                    self.url = url
                    self.headers = headers

            class StdioParams:  # type: ignore
                def __init__(self, command, args=None, env=None):
                    self.command = command
                    self.args = args or []
                    self.env = env

        auth_headers = self._build_auth_headers(server)
        allowed = server.allowed_tools  # None = all tools
        blocked = set(server.blocked_tools or [])

        # Apply action scope filter
        allowed = self._apply_action_policy(server, allowed, blocked)

        if server.transport_type in ("streamable_http", "sse"):
            connection_params = SseParams(
                url=server.endpoint_url,
                headers=auth_headers,
            )
        elif server.transport_type == "stdio":
            parts = server.stdio_command.split() if server.stdio_command else []
            if not parts:
                raise ValueError(f"stdio_command is empty for server {server.name}")
            try:
                from mcp.client.stdio import StdioServerParameters
                server_params = StdioServerParameters(command=parts[0], args=parts[1:])
                connection_params = StdioParams(server_params=server_params)
            except Exception:
                connection_params = StdioParams(
                    command=parts[0],
                    args=parts[1:],
                    env=None,
                )
        else:
            raise ValueError(f"Unknown transport_type: {server.transport_type}")

        toolset = ToolsetClass(
            connection_params=connection_params,
            tool_filter=allowed,
        )
        logger.info(f"Built MCP toolset for server: {server.name}")
        return toolset

    def _build_auth_headers(self, server) -> dict[str, str]:
        """Decrypt and format auth credentials into HTTP headers."""
        if server.auth_type == "none" or not server.auth_secret_encrypted:
            return {}

        secret = decrypt_secret(server.auth_secret_encrypted)

        if server.auth_type == "api_key":
            return {"X-API-Key": secret}
        elif server.auth_type == "bearer":
            return {"Authorization": f"Bearer {secret}"}
        elif server.auth_type == "oauth2":
            # secret is expected to be a JSON string with {"access_token": "..."}
            token_data = json.loads(secret)
            return {"Authorization": f"Bearer {token_data['access_token']}"}
        elif server.auth_type == "basic":
            b64_creds = base64.b64encode(secret.encode()).decode()
            return {"Authorization": f"Basic {b64_creds}"}
        return {}

    def _apply_action_policy(
        self,
        server,
        allowed: Optional[list],
        blocked: set,
    ) -> Optional[list]:
        """
        Granular Action Scope Enforcement.
        
        Supports compound scopes such as:
          - "read_only" (or "read")
          - "read,create" (or "read+create")
          - "read,update"
          - "read,create,update"
          - "read_write" (or "all")

        Tool verb taxonomy:
          - CREATE: create, add, new, insert, post, generate, write
          - UPDATE: update, edit, patch, modify, put, set
          - DELETE: delete, remove, drop, destroy, purge, clear
          - READ:   all query/search/get tools (default safe)
        """
        raw_scope = (server.action_scope or "read_only").lower().replace("+", ",")
        permitted_actions = {act.strip() for act in raw_scope.split(",") if act.strip()}

        # Full access shortcut
        if "read_write" in permitted_actions or "all" in permitted_actions:
            if allowed:
                return [t for t in allowed if t not in blocked]
            return None  # All tools permitted except blocked

        CREATE_VERBS = {"create", "add", "new", "insert", "post", "generate", "write"}
        UPDATE_VERBS = {"update", "edit", "patch", "modify", "put", "set"}
        DELETE_VERBS = {"delete", "remove", "drop", "destroy", "purge", "clear"}

        allow_create = "create" in permitted_actions
        allow_update = "update" in permitted_actions
        allow_delete = "delete" in permitted_actions
        # "read" or "read_only" permits read operations
        allow_read = "read" in permitted_actions or "read_only" in permitted_actions or not permitted_actions

        cached_tools: list = server.cached_tool_schemas or []
        permitted_tools = []

        for tool in cached_tools:
            name = tool["name"].lower()
            if name in blocked:
                continue

            is_create = any(kw in name for kw in CREATE_VERBS)
            is_update = any(kw in name for kw in UPDATE_VERBS)
            is_delete = any(kw in name for kw in DELETE_VERBS)
            is_read = not (is_create or is_update or is_delete)

            # Check if this tool's category is authorized in action_scope
            if is_read and allow_read:
                permitted_tools.append(tool["name"])
            elif is_create and allow_create:
                permitted_tools.append(tool["name"])
            elif is_update and allow_update:
                permitted_tools.append(tool["name"])
            elif is_delete and allow_delete:
                permitted_tools.append(tool["name"])

        if allowed:
            # Intersect with explicit allow-list
            return [t for t in allowed if t in permitted_tools and t not in blocked]

        return permitted_tools or None

    async def _route_servers(self, query: str, servers: list) -> list:
        """
        Simple semantic router: embed query + server descriptions,
        return top-N by cosine similarity.
        Prevents LLM tool-selection degradation when >20-30 MCP tools are active.
        """
        import numpy as np

        from ingestion.embedder import get_embedder

        embedder = get_embedder()
        descriptions = [s.description or s.name for s in servers]
        texts = [query] + descriptions

        try:
            vecs = await embedder.embed(texts)
            query_vec = np.array(vecs[0])
            server_vecs = np.array(vecs[1:])

            # Cosine similarity
            norms = np.linalg.norm(server_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            similarities = server_vecs @ query_vec / (np.linalg.norm(query_vec) * norms.flatten())

            top_indices = np.argsort(similarities)[::-1][: settings.mcp_max_servers_per_query]
            selected = [servers[i] for i in top_indices]
            logger.debug(f"MCP Tool Router selected {len(selected)}/{len(servers)} servers for query")
            return selected

        except Exception as e:
            logger.warning(f"MCP Tool Router failed ({e}), using all servers")
            return servers[: settings.mcp_max_servers_per_query]
