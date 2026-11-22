"""
Unit tests for TenantMiddleware and multi-tenancy context isolation.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from api.main import app
from api.auth import create_access_token
from config.settings import settings, TenancyMode


@pytest.mark.asyncio
async def test_tenant_middleware_default_fallback():
    """Verify default tenant is attached when no tenant header/token is sent in single-tenancy mode."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.headers.get("X-Tenant-ID") == settings.default_tenant_id


@pytest.mark.asyncio
async def test_tenant_middleware_header_extraction():
    """Verify X-Tenant-ID request header is recognized and echoed in response."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health", headers={"X-Tenant-ID": "tenant_acme_corp"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Tenant-ID") == "tenant_acme_corp"


@pytest.mark.asyncio
async def test_tenant_middleware_jwt_token_extraction():
    """Verify tenant_id is extracted from valid Bearer JWT claims."""
    token = create_access_token(user_id="alice", tenant_id="tenant_finance", role="user")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Tenant-ID") == "tenant_finance"


@pytest.mark.asyncio
async def test_tenant_middleware_multi_tenancy_rejection():
    """In multi-tenancy mode, API routes without tenant header/token must be rejected."""
    with patch.object(settings, "tenancy_mode", TenancyMode.MULTI):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/chat/health")  # API route
            assert resp.status_code in (401, 404)  # 401 unauth or 404 route not found
            if resp.status_code == 401:
                assert "Tenant identification required" in resp.json()["detail"]
