from __future__ import annotations

from app.errors.exceptions import AuthorizationError
from app.schemas.internal import Identity

WILDCARD = "*"


class ModelAccessPolicy:
    def authorize(self, identity: Identity, logical_model: str) -> None:
        if WILDCARD in identity.allow_models:
            return
        if logical_model not in identity.allow_models:
            raise AuthorizationError("Not authorized to use this model")
