from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
pytestmark = pytest.mark.integration


def _ollama_ready() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=2.0)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    names = [model.get("name", "") for model in response.json().get("models", [])]
    return any(name.startswith("qwen3:4b") for name in names) and any(
        name.startswith("qwen3:8b") for name in names
    )


skip_without_ollama = pytest.mark.skipif(
    not _ollama_ready(),
    reason="Ollama is not running with qwen3:4b and qwen3:8b",
)


@skip_without_ollama
def test_non_streaming_fast_and_general(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GATEWAY_API_KEYS",
        '[{"id":"it","key":"sk-it","allow_models":["fast","general"]}]',
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL)
    with TestClient(create_app()) as client:
        for model in ("fast", "general"):
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-it"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
                timeout=120.0,
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model"] == model
            assert body["choices"][0]["message"]["content"]
            assert "qwen" not in body["model"]


@skip_without_ollama
def test_streaming_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GATEWAY_API_KEYS",
        '[{"id":"it","key":"sk-it","allow_models":["*"]}]',
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL)
    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-it"},
            json={
                "model": "fast",
                "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                "stream": True,
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=120.0,
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()
    assert "data: [DONE]" in body
    assert "chat.completion.chunk" in body
