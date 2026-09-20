from __future__ import annotations

import pytest

from app.authorization.policies import ModelAccessPolicy
from app.errors.exceptions import AuthorizationError, BackendUnavailableError, UnknownModelError
from app.routing.model_router import ModelRouter
from app.schemas.internal import ChatMessage, ChatRequest, Identity
from app.services.chat_service import ChatService
from tests.conftest import FakeBackend, make_settings


def _service(backend: FakeBackend | None = None) -> tuple[ChatService, FakeBackend]:
    fake = backend or FakeBackend()
    service = ChatService(
        router=ModelRouter(make_settings().models),
        backends={"ollama": fake},
        authorizer=ModelAccessPolicy(),
    )
    return service, fake


def _request() -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content="hi"),))


@pytest.mark.asyncio
async def test_complete_chat_routes_to_backend_model() -> None:
    service, fake = _service()
    identity = Identity(id="dev", allow_models=("fast", "general"))
    result = await service.complete_chat(identity, "fast", _request())
    assert fake.chat_models == ["qwen3:4b"]
    assert result.content.startswith("echo:qwen3:4b:")


@pytest.mark.asyncio
async def test_complete_chat_rejects_unauthorized_model() -> None:
    service, _ = _service()
    identity = Identity(id="fast-only", allow_models=("fast",))
    with pytest.raises(AuthorizationError):
        await service.complete_chat(identity, "general", _request())


@pytest.mark.asyncio
async def test_complete_chat_unknown_model() -> None:
    service, _ = _service()
    identity = Identity(id="admin", allow_models=("*",))
    with pytest.raises(UnknownModelError):
        await service.complete_chat(identity, "missing", _request())


@pytest.mark.asyncio
async def test_missing_backend_is_unavailable() -> None:
    service = ChatService(
        router=ModelRouter(make_settings().models),
        backends={},
        authorizer=ModelAccessPolicy(),
    )
    identity = Identity(id="dev", allow_models=("fast",))
    with pytest.raises(BackendUnavailableError):
        await service.complete_chat(identity, "fast", _request())


@pytest.mark.asyncio
async def test_stream_chat_uses_backend_model() -> None:
    service, fake = _service()
    identity = Identity(id="dev", allow_models=("fast", "general"))
    chunks = [chunk async for chunk in service.stream_chat(identity, "general", _request())]
    assert fake.stream_models == ["qwen3:8b"]
    assert any(chunk.content for chunk in chunks)
