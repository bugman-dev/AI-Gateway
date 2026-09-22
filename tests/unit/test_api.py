from __future__ import annotations

import json

from app.errors.exceptions import BackendUnavailableError
from app.main import create_app
from fastapi.testclient import TestClient
from tests.conftest import FakeBackend, make_settings

AUTH = {"Authorization": "Bearer sk-local-dev"}


def _client(backend: FakeBackend | None = None):
    fake = backend or FakeBackend()
    return TestClient(create_app(settings=make_settings(), backends={"ollama": fake})), fake


def test_health() -> None:
    client, _ = _client()
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_models_requires_auth() -> None:
    client, _ = _client()
    with client:
        response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_list_models() -> None:
    client, _ = _client()
    with client:
        response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert ids == {"fast", "general"}
    assert response.headers["x-request-id"]


def test_invalid_api_key() -> None:
    client, _ = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-wrong"},
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 401


def test_unauthorized_model() -> None:
    client, _ = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-fast-only"},
            json={"model": "general", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_not_authorized"


def test_unknown_model() -> None:
    client, _ = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-star"},
            json={"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 404


def test_chat_completion_uses_logical_model_in_response() -> None:
    client, fake = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fast"
    assert body["choices"][0]["message"]["content"].startswith("echo:qwen3:4b:")
    assert fake.chat_models == ["qwen3:4b"]
    assert "sk-local-dev" not in response.text


def test_general_routes_to_8b() -> None:
    client, fake = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "general", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert fake.chat_models == ["qwen3:8b"]
    assert response.json()["model"] == "general"


def test_streaming_chat_completion() -> None:
    client, fake = _client()
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = response.read().decode()
    assert fake.stream_models == ["qwen3:4b"]
    assert "data: [DONE]" in body
    events = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(event["choices"][0]["delta"].get("content") or "" for event in events)
    assert content == "hello world"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"


def test_honors_request_id_header() -> None:
    client, _ = _client()
    with client:
        response = client.get("/v1/models", headers={**AUTH, "X-Request-ID": "req-123"})
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-request-timestamp"]


def test_rejects_tools_over_http() -> None:
    client, _ = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "fast",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
            },
        )
    assert response.status_code == 400


def test_rejects_unknown_fields_over_http() -> None:
    client, _ = _client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "fast",
                "messages": [{"role": "user", "content": "hi"}],
                "made_up_option": True,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_structured_log_includes_identity_and_route() -> None:
    import logging

    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Handler()
    logger = logging.getLogger("ai_gateway")
    client, _ = _client()
    logger.addHandler(handler)
    try:
        with client:
            response = client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 200
    payload = json.loads(records[-1])
    assert payload["identity_id"] == "local-dev"
    assert payload["logical_model"] == "fast"
    assert payload["backend"] == "ollama"
    assert payload["backend_model"] == "qwen3:4b"
    assert payload["http_status"] == 200
    assert payload["timestamp"]
    assert "T" in payload["timestamp"]
    assert "sk-local-dev" not in records[-1]


def test_streaming_backend_down_returns_503() -> None:
    class DownBackend:
        async def chat(self, request, model):
            raise BackendUnavailableError()

        async def stream_chat(self, request, model):
            raise BackendUnavailableError()
            yield  # pragma: no cover

    with TestClient(create_app(settings=make_settings(), backends={"ollama": DownBackend()})) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_unavailable"
