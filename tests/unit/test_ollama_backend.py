from __future__ import annotations

import json

import httpx
import pytest
from app.backends.ollama import OllamaBackend
from app.errors.exceptions import BackendError, BackendUnavailableError
from app.schemas.internal import ChatMessage, ChatRequest


def _request() -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content="hi"),))


@pytest.mark.asyncio
async def test_chat_maps_ollama_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/chat")
        body = json.loads(request.content)
        assert body["model"] == "qwen3:4b"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client, "http://127.0.0.1:11434")
    result = await backend.chat(_request(), "qwen3:4b")
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 4
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_maps_ndjson() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = (
            json.dumps({"message": {"role": "assistant", "content": "Hel"}, "done": False})
            + "\n"
            + json.dumps(
                {
                    "message": {"role": "assistant", "content": "lo"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 2,
                }
            )
            + "\n"
        )
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client, "http://ollama.internal:11434")
    chunks = [chunk async for chunk in backend.stream_chat(_request(), "qwen3:8b")]
    await client.aclose()
    assert "".join(chunk.content or "" for chunk in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_backend_http_error_does_not_leak_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="ollama panic in /root/.ollama")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client, "http://127.0.0.1:11434")
    with pytest.raises(BackendUnavailableError, match="unavailable") as exc_info:
        await backend.chat(_request(), "qwen3:4b")
    await client.aclose()
    assert "ollama panic" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_backend_client_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found on disk"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client, "http://127.0.0.1:11434")
    with pytest.raises(BackendError, match="failed") as exc_info:
        await backend.chat(_request(), "qwen3:4b")
    await client.aclose()
    assert "model not found on disk" not in str(exc_info.value)
