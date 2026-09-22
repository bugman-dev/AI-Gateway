from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.config import load_settings


def test_load_settings_from_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
server:
  host: 0.0.0.0
  port: 8080
backends:
  ollama:
    base_url: http://127.0.0.1:11434
models:
  fast:
    backend: ollama
    model: qwen3:4b
  general:
    backend: ollama
    model: qwen3:8b
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "GATEWAY_API_KEYS",
        json.dumps([{"id": "dev", "key": "sk-test", "allow_models": ["*"]}]),
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    settings = load_settings(config)
    assert settings.models["fast"].model == "qwen3:4b"
    assert settings.models["general"].model == "qwen3:8b"
    assert settings.ollama_base_url == "http://host.docker.internal:11434"
    assert settings.api_keys[0].id == "dev"
