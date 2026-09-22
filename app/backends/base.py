from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.schemas.internal import ChatChunk, ChatRequest, ChatResult


class ModelBackend(Protocol):
    async def chat(self, request: ChatRequest, model: str) -> ChatResult: ...

    async def stream_chat(self, request: ChatRequest, model: str) -> AsyncIterator[ChatChunk]: ...
