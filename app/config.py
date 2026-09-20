from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_CONFIG = _REPO_ROOT / "config" / "models.yaml"


class ApiKeyConfig(BaseModel):
    id: str
    key: str
    allow_models: list[str] = Field(min_length=1)

    @field_validator("id", "key")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ModelRouteConfig(BaseModel):
    backend: str
    model: str


class Settings(BaseModel):
    server_host: str
    server_port: int
    ollama_base_url: str
    models: dict[str, ModelRouteConfig]
    api_keys: list[ApiKeyConfig]


class EnvSettings(BaseSettings):
    gateway_api_keys: str
    ollama_base_url: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Models config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("models.yaml must contain a mapping")
    return data


def _parse_api_keys(raw: str) -> list[ApiKeyConfig]:
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GATEWAY_API_KEYS must be valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("GATEWAY_API_KEYS must be a non-empty JSON array")
    return [ApiKeyConfig.model_validate(item) for item in payload]


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_MODELS_CONFIG
    raw = _load_yaml(path)
    env = EnvSettings()  # type: ignore[call-arg]

    server = raw.get("server") or {}
    backends = raw.get("backends") or {}
    ollama = backends.get("ollama") or {}
    models_raw = raw.get("models") or {}
    if not models_raw:
        raise ValueError("models.yaml must define at least one logical model")

    ollama_base_url = env.ollama_base_url or ollama.get("base_url")
    if not ollama_base_url:
        raise ValueError("Ollama base_url is missing from config and OLLAMA_BASE_URL")

    return Settings(
        server_host=str(server.get("host", "0.0.0.0")),
        server_port=int(server.get("port", 8080)),
        ollama_base_url=str(ollama_base_url).rstrip("/"),
        models={name: ModelRouteConfig.model_validate(cfg) for name, cfg in models_raw.items()},
        api_keys=_parse_api_keys(env.gateway_api_keys),
    )
