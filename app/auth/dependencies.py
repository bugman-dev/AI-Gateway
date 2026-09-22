from __future__ import annotations

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.api_keys import KeyStore
from app.errors.exceptions import AuthenticationError
from app.observability.request_context import set_identity_id
from app.schemas.internal import Identity

_bearer = HTTPBearer(auto_error=False)


async def get_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Identity:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise AuthenticationError("Missing API key")
    store: KeyStore = request.app.state.key_store
    identity = store.lookup(credentials.credentials)
    if identity is None:
        raise AuthenticationError("Invalid API key")
    set_identity_id(identity.id)
    return identity
