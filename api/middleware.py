"""
Multi-tenancy and Request Context Middleware.

Extracts tenant_id from:
1. `X-Tenant-ID` request header
2. `Authorization: Bearer <jwt>` claims (tenant_id claim)
3. Default configured tenant (`settings.default_tenant_id`)

Attaches `request.state.tenant_id` and adds `X-Tenant-ID` header to HTTP responses.
"""

from typing import Optional
from fastapi import Request, Response
from jose import jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from config.settings import settings, TenancyMode


def extract_tenant_from_token(token: str) -> Optional[str]:
    """Safely decode tenant_id claim from JWT without raising exceptions."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
        return payload.get("tenant_id")
    except JWTError:
        return None


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware that resolves the current tenant for every incoming request
    and sets `request.state.tenant_id`.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        tenant_id = None

        # 1. Check X-Tenant-ID header
        header_tenant = request.headers.get("X-Tenant-ID")
        if header_tenant and header_tenant.strip():
            tenant_id = header_tenant.strip()

        # 2. Check Authorization Bearer token
        if not tenant_id:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
                tenant_id = extract_tenant_from_token(token)

        # 3. Fallback to default tenant
        if not tenant_id:
            tenant_id = settings.default_tenant_id

        # 4. In MULTI tenancy mode, enforce valid tenant on API routes
        path = request.url.path
        is_api_route = path.startswith("/api/")
        is_public_path = path in ("/health", "/api/health", "/docs", "/openapi.json", "/redoc")

        if (
            settings.tenancy_mode == TenancyMode.MULTI
            and is_api_route
            and not is_public_path
            and tenant_id == "default"
            and not header_tenant
        ):
            # In multi-tenant mode, default unauthenticated tenant is disallowed on API routes
            return JSONResponse(
                status_code=401,
                content={"detail": "Tenant identification required in multi-tenant mode (via X-Tenant-ID header or JWT)."},
            )

        # Set tenant in request state
        request.state.tenant_id = tenant_id

        # Process downstream request
        response = await call_next(request)

        # Attach tenant header to response for audit & traceability
        response.headers["X-Tenant-ID"] = tenant_id
        return response


def get_current_tenant(request: Request) -> str:
    """Dependency helper to get the active tenant_id from request state."""
    return getattr(request.state, "tenant_id", settings.default_tenant_id)
