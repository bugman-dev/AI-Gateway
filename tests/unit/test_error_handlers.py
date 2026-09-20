from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.errors.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BackendError,
    BackendUnavailableError,
    UnknownModelError,
)
from app.errors.handlers import _status_for
from app.main import create_app
from tests.conftest import FakeBackend, make_settings


def test_status_mapping() -> None:
    assert _status_for(AuthenticationError()) == 401
    assert _status_for(AuthorizationError()) == 403
    assert _status_for(UnknownModelError("x")) == 404
    assert _status_for(BackendError()) == 502
    assert _status_for(BackendUnavailableError()) == 503


def test_http_error_shape_hides_backend_details() -> None:
    with TestClient(create_app(settings=make_settings(), backends={"ollama": FakeBackend()})) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-local-dev"},
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fast"
    assert "ollama" not in json.dumps({k: body[k] for k in body if k != "choices"}).lower()


def test_validation_error_is_openai_shaped() -> None:
    with TestClient(create_app(settings=make_settings(), backends={"ollama": FakeBackend()})) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-local-dev"},
            json={"model": "fast"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
