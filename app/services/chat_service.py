from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

from app.authorization.policies import ModelAccessPolicy
from app.backends.base import ModelBackend
from app.errors.exceptions import BackendUnavailableError
from app.observability.request_context import set_route_context
from app.routing.model_router import ModelRouter
from app.schemas.internal import ChatChunk, ChatRequest, ChatResult, Identity, ResolvedRoute


class ChatService:
    def __init__(
        self,
        *,
        router: ModelRouter,
        backends: Mapping[str, ModelBackend],
        authorizer: ModelAccessPolicy | None = None,
    ) -> None:
        self._router = router
        self._backends = backends
        self._authorizer = authorizer or ModelAccessPolicy()

    def prepare(self, identity: Identity, logical_model: str) -> ResolvedRoute:
        return self._prepare(identity, logical_model)

    async def complete_chat(self, identity: Identity, logical_model: str, request: ChatRequest) -> ChatResult:
        route = self._prepare(identity, logical_model)
        backend = self._backend_for(route)
        return await backend.chat(request, route.backend_model)

    async def stream_chat(
        self,
        identity: Identity,
        logical_model: str,
        request: ChatRequest,
    ) -> AsyncIterator[ChatChunk]:
        route = self._prepare(identity, logical_model)
        backend = self._backend_for(route)
        async for chunk in backend.stream_chat(request, route.backend_model):
            yield chunk

    def _prepare(self, identity: Identity, logical_model: str) -> ResolvedRoute:
        self._authorizer.authorize(identity, logical_model)
        route = self._router.resolve(logical_model)
        set_route_context(route.logical_model, route.backend, route.backend_model)
        return route

    def _backend_for(self, route: ResolvedRoute) -> ModelBackend:
        backend = self._backends.get(route.backend)
        if backend is None:
            raise BackendUnavailableError("Inference backend is unavailable")
        return backend
