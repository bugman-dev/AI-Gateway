from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BackendError,
    BackendUnavailableError,
    GatewayError,
    InvalidRequestError,
    UnknownModelError,
)
from app.observability.request_context import get_request_id, set_error_class


def _error_body(message: str, error_type: str, code: str | None) -> dict[str, object]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def _response(status_code: int, message: str, error_type: str, code: str | None) -> JSONResponse:
    headers = {}
    request_id = get_request_id()
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=_error_body(message, error_type, code),
        headers=headers,
    )


def _status_for(exc: GatewayError) -> int:
    if isinstance(exc, AuthenticationError):
        return 401
    if isinstance(exc, AuthorizationError):
        return 403
    if isinstance(exc, UnknownModelError):
        return 404
    if isinstance(exc, InvalidRequestError):
        return 400
    if isinstance(exc, BackendUnavailableError):
        return 503
    if isinstance(exc, BackendError):
        return 502
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_request: Request, exc: GatewayError) -> JSONResponse:
        set_error_class(type(exc).__name__)
        return _response(_status_for(exc), exc.message, exc.error_type, exc.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        set_error_class("RequestValidationError")
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            detail = error.get("msg", "invalid request")
            messages.append(f"{location}: {detail}" if location else detail)
        message = "; ".join(messages) or "Invalid request"
        return _response(400, message, "invalid_request_error", "invalid_request")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        set_error_class("HTTPException")
        if exc.status_code == 401:
            return _response(401, str(exc.detail), "authentication_error", "invalid_api_key")
        if exc.status_code == 403:
            return _response(403, str(exc.detail), "authorization_error", "model_not_authorized")
        if exc.status_code == 404:
            return _response(404, str(exc.detail), "invalid_request_error", "not_found")
        return _response(exc.status_code, str(exc.detail), "api_error", "http_error")

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        set_error_class(type(exc).__name__)
        logging.getLogger("ai_gateway").exception("unhandled gateway error")
        return _response(500, "Internal gateway error", "internal_error", "internal_error")
