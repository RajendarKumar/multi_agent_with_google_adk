"""
Integration tests for UI endpoints, static asset serving, and Agent Config API.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.mark.asyncio
async def test_root_redirect_to_ui():
    """Verify GET / redirects to /ui/."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False) as client:
        resp = await client.get("/")
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/ui/"


@pytest.mark.asyncio
async def test_ui_index_html_serving():
    """Verify /ui/ serves index.html with all required Phase 3 UI modules."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text

        # Verify 3.19 Admin UI elements
        assert "MCP Toolset Registry" in html
        assert "Agent Configuration" in html
        assert "Observability & Traces" in html

        # Verify 3.20 Chat UI elements
        assert "Multi-Agent Research Pipeline" in html
        assert "liveBreadcrumbs" in html
        assert "citationInspector" in html
        assert "toggleShowSources" in html

        # Verify 3.21 File upload & Ingestion UI elements
        assert "Polymorphic Document Upload" in html
        assert "fileDropzone" in html
        assert "Web URL Registration & Refresh Scheduler" in html


@pytest.mark.asyncio
async def test_ui_static_assets_serving():
    """Verify /ui/style.css and /ui/app.js are served cleanly."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        css_resp = await client.get("/ui/style.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]

        js_resp = await client.get("/ui/app.js")
        assert js_resp.status_code == 200
        assert "javascript" in js_resp.headers["content-type"] or "text/plain" in js_resp.headers["content-type"]


@pytest.mark.asyncio
async def test_agent_configuration_api_read_and_update():
    """Verify GET and PATCH /api/admin/config/agents."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Read existing config
        get_resp = await client.get("/api/admin/config/agents")
        assert get_resp.status_code == 200
        configs = get_resp.json()
        assert "platform_orchestrator" in configs
        assert "query_rewriter_agent" in configs

        orig_temp = configs["query_rewriter_agent"].get("temperature", 0.0)

        # 2. Update a config field
        patch_resp = await client.patch(
            "/api/admin/config/agents",
            json={"query_rewriter_agent": {"temperature": 0.05}},
        )
        assert patch_resp.status_code == 200
        updated = patch_resp.json()
        assert updated["agents"]["query_rewriter_agent"]["temperature"] == 0.05

        # 3. Revert to original temperature to maintain pristine config
        revert_resp = await client.patch(
            "/api/admin/config/agents",
            json={"query_rewriter_agent": {"temperature": orig_temp}},
        )
        assert revert_resp.status_code == 200


@pytest.mark.asyncio
async def test_ui_configuration_endpoint():
    """Verify GET /api/admin/config/ui returns display flag defaults."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/admin/config/ui")
        assert resp.status_code == 200
        data = resp.json()
        assert "show_evidence_sources" in data
        assert data["show_evidence_sources"] is False
        assert "default_tenant_id" in data
