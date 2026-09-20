from __future__ import annotations

from app.auth.api_keys import KeyStore
from app.config import ApiKeyConfig
from app.errors.exceptions import AuthenticationError
from tests.conftest import make_settings


def test_lookup_returns_identity_without_raw_key() -> None:
    store = KeyStore(make_settings().api_keys)
    identity = store.lookup("sk-local-dev")
    assert identity is not None
    assert identity.id == "local-dev"
    assert "fast" in identity.allow_models
    assert "sk-local-dev" not in identity.__dict__.values()


def test_lookup_rejects_unknown_key() -> None:
    store = KeyStore(make_settings().api_keys)
    assert store.lookup("sk-unknown") is None


def test_lookup_rejects_empty_key() -> None:
    store = KeyStore(make_settings().api_keys)
    assert store.lookup("") is None


def test_different_length_keys_do_not_match() -> None:
    store = KeyStore([ApiKeyConfig(id="a", key="abcd", allow_models=["fast"])])
    assert store.lookup("abc") is None


def test_authentication_error_defaults() -> None:
    error = AuthenticationError()
    assert error.code == "invalid_api_key"
    assert error.error_type == "authentication_error"
