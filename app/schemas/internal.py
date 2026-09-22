from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    id: str
    allow_models: tuple[str, ...]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | tuple[str, ...] | None = None


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ChatChunk:
    content: str | None = None
    role: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ResolvedRoute:
    logical_model: str
    backend: str
    backend_model: str
