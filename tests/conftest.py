from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import ApiKeyConfig, ModelRouteConfig, Settings
from app.schemas.internal import ChatChunk, ChatRequest, ChatResult


def make_settings() -> Settings:
    return Settings(
        server_host="0.0.0.0",
        server_port=8080,
        ollama_base_url="http://127.0.0.1:11434",
        models={
            "fast": ModelRouteConfig(backend="ollama", model="qwen3:4b"),
            "general": ModelRouteConfig(backend="ollama", model="qwen3:8b"),
        },
        api_keys=[
            ApiKeyConfig(id="local-dev", key="sk-local-dev", allow_models=["fast", "general"]),
            ApiKeyConfig(id="fast-only", key="sk-fast-only", allow_models=["fast"]),
            ApiKeyConfig(id="admin", key="sk-star", allow_models=["*"]),
        ],
    )


class FakeBackend:
    def __init__(self) -> None:
        self.chat_models: list[str] = []
        self.stream_models: list[str] = []

    async def chat(self, request: ChatRequest, model: str) -> ChatResult:
        self.chat_models.append(model)
        last = request.messages[-1].content if request.messages else ""
        return ChatResult(
            content=f"echo:{model}:{last}",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
        )

    async def stream_chat(self, request: ChatRequest, model: str) -> AsyncIterator[ChatChunk]:
        self.stream_models.append(model)
        yield ChatChunk(role="assistant", content="hello")
        yield ChatChunk(content=" world")
        yield ChatChunk(finish_reason="stop", prompt_tokens=1, completion_tokens=2)
