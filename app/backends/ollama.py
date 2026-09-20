from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.errors.exceptions import BackendError, BackendUnavailableError
from app.schemas.internal import ChatChunk, ChatRequest, ChatResult

_CHAT_PATH = "/api/chat"


class OllamaBackend:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def chat(self, request: ChatRequest, model: str) -> ChatResult:
        payload = _to_ollama_payload(request, model, stream=False)
        try:
            response = await self._client.post(self._url(), json=payload)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Inference backend timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError("Inference backend is unavailable") from exc
        _raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendError("Inference backend failed") from exc
        return _to_chat_result(data)

    async def stream_chat(self, request: ChatRequest, model: str) -> AsyncIterator[ChatChunk]:
        payload = _to_ollama_payload(request, model, stream=True)
        try:
            async with self._client.stream("POST", self._url(), json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise BackendError("Inference backend failed") from exc
                    chunk = _to_chat_chunk(data)
                    if chunk is not None:
                        yield chunk
        except BackendError:
            raise
        except BackendUnavailableError:
            raise
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Inference backend timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError("Inference backend is unavailable") from exc

    def _url(self) -> str:
        return f"{self._base_url}{_CHAT_PATH}"


def _to_ollama_payload(request: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop is not None:
        options["stop"] = list(request.stop) if isinstance(request.stop, tuple) else request.stop
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": message.role, "content": message.content} for message in request.messages],
        "stream": stream,
    }
    if options:
        payload["options"] = options
    return payload


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code >= 500:
        raise BackendUnavailableError("Inference backend is unavailable")
    if response.status_code >= 400:
        raise BackendError("Inference backend failed")


def _finish_reason(data: dict[str, Any]) -> str:
    reason = data.get("done_reason") or "stop"
    if reason == "length":
        return "length"
    return "stop"


def _to_chat_result(data: dict[str, Any]) -> ChatResult:
    message = data.get("message") or {}
    content = message.get("content") or ""
    return ChatResult(
        content=content,
        finish_reason=_finish_reason(data),
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
    )


def _to_chat_chunk(data: dict[str, Any]) -> ChatChunk | None:
    message = data.get("message") or {}
    content = message.get("content")
    role = message.get("role")
    done = bool(data.get("done"))
    if done:
        return ChatChunk(
            content=content or None,
            role=role,
            finish_reason=_finish_reason(data),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
    if content is None and role is None:
        return None
    return ChatChunk(content=content or None, role=role)
