from __future__ import annotations

import pytest

from app.authorization.policies import ModelAccessPolicy
from app.errors.exceptions import AuthorizationError
from app.schemas.internal import Identity


def test_allows_listed_model() -> None:
    policy = ModelAccessPolicy()
    identity = Identity(id="dev", allow_models=("fast", "general"))
    policy.authorize(identity, "fast")


def test_rejects_unlisted_model() -> None:
    policy = ModelAccessPolicy()
    identity = Identity(id="dev", allow_models=("fast",))
    with pytest.raises(AuthorizationError):
        policy.authorize(identity, "general")


def test_wildcard_allows_any_model() -> None:
    policy = ModelAccessPolicy()
    identity = Identity(id="admin", allow_models=("*",))
    policy.authorize(identity, "general")
    policy.authorize(identity, "unknown")
