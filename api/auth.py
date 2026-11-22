"""
JWT Authentication & RBAC middleware.

Usage:
  - Protect any route with:   current_user: User = Depends(get_current_user)
  - Admin-only routes:        current_user: User = Depends(require_admin)
  - Tenant injection:         tenant_id extracted automatically from JWT
"""


from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config.settings import settings

# ── Password hashing ─────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ── Token / User models ───────────────────────────────────────────────────────
class TokenData(BaseModel):
    user_id: str
    tenant_id: str
    role: str = "user"  # "user" | "admin"


class User(BaseModel):
    user_id: str
    tenant_id: str
    role: str = "user"


# ── Token creation ────────────────────────────────────────────────────────────
def create_access_token(user_id: str, tenant_id: str, role: str = "user") -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Token verification ────────────────────────────────────────────────────────
def _decode_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub", "")
        tenant_id: str = payload.get("tenant_id", settings.default_tenant_id)
        role: str = payload.get("role", "user")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing subject")
        return TokenData(user_id=user_id, tenant_id=tenant_id, role=role)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ──────────────────────────────────────────────────────
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> User:
    """
    Extracts and validates the Bearer JWT.
    In SINGLE tenancy mode, falls back to anonymous user if no token provided.
    In MULTI tenancy mode, token is required.
    """
    if credentials is None:
        if settings.tenancy_mode.value == "single":
            # Allow unauthenticated access in single-tenant dev mode
            return User(
                user_id="anonymous",
                tenant_id=settings.default_tenant_id,
                role="admin",  # full access in single-tenant dev
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = _decode_token(credentials.credentials)
    return User(
        user_id=token_data.user_id,
        tenant_id=token_data.tenant_id,
        role=token_data.role,
    )


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Restrict route to admin role only."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


# ── Login endpoint helper ─────────────────────────────────────────────────────
# Wire this into a /api/auth/login route when user management is added.
def authenticate_user(username: str, password: str, hashed_password: str) -> Optional[str]:
    """Returns a JWT token if credentials are valid, else None."""
    if not verify_password(password, hashed_password):
        return None
    return create_access_token(user_id=username, tenant_id=settings.default_tenant_id)
