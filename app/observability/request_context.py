from __future__ import annotations

from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_timestamp: ContextVar[str | None] = ContextVar("request_timestamp", default=None)
_identity_id: ContextVar[str | None] = ContextVar("identity_id", default=None)
_logical_model: ContextVar[str | None] = ContextVar("logical_model", default=None)
_backend: ContextVar[str | None] = ContextVar("backend", default=None)
_backend_model: ContextVar[str | None] = ContextVar("backend_model", default=None)
_error_class: ContextVar[str | None] = ContextVar("error_class", default=None)


def set_request_id(request_id: str) -> Token:
    return _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_timestamp(value: str) -> None:
    _request_timestamp.set(value)


def get_request_timestamp() -> str | None:
    return _request_timestamp.get()


def set_identity_id(identity_id: str) -> None:
    _identity_id.set(identity_id)


def get_identity_id() -> str | None:
    return _identity_id.get()


def set_route_context(logical_model: str, backend: str, backend_model: str) -> None:
    _logical_model.set(logical_model)
    _backend.set(backend)
    _backend_model.set(backend_model)


def set_error_class(name: str) -> None:
    _error_class.set(name)


def snapshot() -> dict[str, str | None]:
    return {
        "request_id": _request_id.get(),
        "timestamp": _request_timestamp.get(),
        "identity_id": _identity_id.get(),
        "logical_model": _logical_model.get(),
        "backend": _backend.get(),
        "backend_model": _backend_model.get(),
        "error_class": _error_class.get(),
    }


def reset_context(
    request_token: Token | None = None,
) -> None:
    if request_token is not None:
        _request_id.reset(request_token)
    else:
        _request_id.set(None)
    _request_timestamp.set(None)
    _identity_id.set(None)
    _logical_model.set(None)
    _backend.set(None)
    _backend_model.set(None)
    _error_class.set(None)
