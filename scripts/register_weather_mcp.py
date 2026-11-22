#!/usr/bin/env python3
"""
Seed / Register WeatherService MCP Server via REST API endpoint.

Sends an HTTP POST request to '/api/admin/mcp/servers' (or '/mcp/servers') to test
and execute the full end-to-end MCP registration lifecycle:
  1. Payload validation (Pydantic MCPServerCreate)
  2. Subprocess probe & tool discovery via Google ADK McpToolset
  3. Schema extraction & caching
  4. Database persistence / upsert in PostgreSQL
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx
from api.main import app


async def register_weather_server_via_api(base_url: str = "http://localhost:8000", endpoint: str = "/api/admin/mcp/servers"):
    server_script = str(project_root / "mcp_servers" / "weather" / "server.py")
    stdio_cmd = f"{sys.executable} {server_script} --stdio"

    payload = {
        "name": "WeatherService",
        "description": (
            "Real-time meteorological intelligence server providing live conditions, "
            "multi-day forecasts, wind, and air quality indices via WeatherAPI.com."
        ),
        "transport_type": "stdio",
        "stdio_command": stdio_cmd,
        "auth_type": "none",
        "action_scope": "read_only",
        "tenant_id": "default",
    }

    target_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    print(f"📡 Registering WeatherService MCP Server via API endpoint: {target_url}")
    print(f"   Command: {stdio_cmd}")

    client_connected = False
    data = None
    status_code = None

    # Step 1: Try connecting to live running server over HTTP network socket
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(target_url, json=payload)
            if resp.status_code == 404 and endpoint != "/mcp/servers":
                # Fallback to direct /mcp/servers alias
                fallback_url = f"{base_url.rstrip('/')}/mcp/servers"
                resp = await client.post(fallback_url, json=payload)
                target_url = fallback_url

            status_code = resp.status_code
            if resp.is_success:
                data = resp.json()
                client_connected = True
            else:
                print(f"⚠️  Live server returned HTTP {resp.status_code}: {resp.text}")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Step 2: If live HTTP server is offline, execute through FastAPI ASGI transport end-to-end
    if not client_connected:
        print("ℹ️  Live HTTP server offline on port 8000. Invoking FastAPI route handler via ASGI transport...")
        try:
            from api.database import init_db
            await init_db()
        except Exception:
            pass
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000", timeout=30.0) as asgi_client:
            # Try /mcp/servers and /api/admin/mcp/servers
            resp = await asgi_client.post(endpoint, json=payload)
            if resp.status_code == 404:
                resp = await asgi_client.post("/mcp/servers", json=payload)
                target_url = "http://localhost:8000/mcp/servers"
            status_code = resp.status_code
            if resp.is_success:
                data = resp.json()
            else:
                print(f"❌ API Error ({resp.status_code}): {resp.text}")
                sys.exit(1)

    print(f"\n✅ API Response received (HTTP {status_code}):")
    print(f"   • Server ID:         {data.get('id')}")
    print(f"   • Server Name:       {data.get('name')}")
    print(f"   • Action Scope:      {data.get('action_scope')}")
    print(f"   • Tools Discovered:  {data.get('tools_discovered')}")
    print(f"   • Discovered Tools:  {data.get('tool_names')}")
    print(f"   • Message:           {data.get('message')}")
    print("\n🎉 WeatherService successfully registered and validated end-to-end through the API!")


def main():
    parser = argparse.ArgumentParser(description="Register Weather MCP Server via REST API")
    parser.add_argument(
        "--url",
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MCP_ENDPOINT", "/mcp/servers"),
        help="MCP server registration endpoint path (default: /mcp/servers)",
    )
    args = parser.parse_args()

    asyncio.run(register_weather_server_via_api(base_url=args.url, endpoint=args.endpoint))


if __name__ == "__main__":
    main()
