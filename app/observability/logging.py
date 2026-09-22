from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.request_context import (
    reset_context,
    set_error_class,
    set_request_id,
    set_request_timestamp,
    snapshot,
)

_LOGGER_NAME = "ai_gateway"


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_request(*, http_status: int, latency_ms: float, error_class: str | None = None) -> None:
    context = snapshot()
    payload = {
        "timestamp": context["timestamp"] or datetime.now(UTC).isoformat(),
        "request_id": context["request_id"],
        "identity_id": context["identity_id"],
        "logical_model": context["logical_model"],
        "backend": context["backend"],
        "backend_model": context["backend_model"],
        "http_status": http_status,
        "latency_ms": round(latency_ms, 2),
        "error_class": error_class or context["error_class"],
    }
    logging.getLogger(_LOGGER_NAME).info(json.dumps(payload, default=str))


def normalize_request_id(value: str | None) -> str:
    if (
        value
        and value.strip()
        and len(value) <= 128
        and value.isascii()
        and all(char.isalnum() or char in "-_" for char in value.strip())
    ):
        return value.strip()
    return str(uuid4())


class RequestContextMiddleware:
    """Pure ASGI middleware so request-context ContextVars survive the call."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        request_id = normalize_request_id(headers.get("x-request-id"))
        request_timestamp = datetime.now(UTC).isoformat()
        token = set_request_id(request_id)
        set_request_timestamp(request_timestamp)
        started = time.perf_counter()
        status = 500
        error_class: str | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                mutable = MutableHeaders(raw=message.setdefault("headers", []))
                mutable["X-Request-ID"] = request_id
                mutable["X-Request-Timestamp"] = request_timestamp
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            set_error_class(type(exc).__name__)
            error_class = type(exc).__name__
            raise
        finally:
            context_error = snapshot().get("error_class")
            log_request(
                http_status=status,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_class=error_class or context_error,
            )
            reset_context(token)
