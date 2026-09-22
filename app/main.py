from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.v1.router import api_router
from app.auth.api_keys import KeyStore
from app.authorization.policies import ModelAccessPolicy
from app.backends.base import ModelBackend
from app.backends.ollama import OllamaBackend
from app.config import Settings, load_settings
from app.errors.handlers import register_exception_handlers
from app.observability.logging import RequestContextMiddleware, configure_logging
from app.routing.model_router import ModelRouter
from app.services.chat_service import ChatService


def create_app(
    settings: Settings | None = None,
    backends: Mapping[str, ModelBackend] | None = None,
) -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="AI Gateway",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.provided_settings = settings
    app.state.provided_backends = backends
    app.include_router(api_router)
    register_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.provided_settings or load_settings()
    app.state.settings = settings
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))
    app.state.http_client = client
    backends = app.state.provided_backends
    if backends is None:
        backends = {"ollama": OllamaBackend(client, settings.ollama_base_url)}
    app.state.backends = backends
    app.state.model_router = ModelRouter(settings.models)
    app.state.chat_service = ChatService(
        router=app.state.model_router,
        backends=backends,
        authorizer=ModelAccessPolicy(),
    )
    app.state.key_store = KeyStore(settings.api_keys)
    try:
        yield
    finally:
        await client.aclose()


app = create_app()
