# AI Gateway

OpenAI-compatible API gateway for a personal self-hosted AI server. Agents call this gateway; Ollama stays on localhost behind it.

Logical models: `fast` → `qwen3:4b`, `general` → `qwen3:8b`.

## Requirements

- Python 3.12+
- Ollama running on the host (`http://127.0.0.1:11434`) with `qwen3:4b` and `qwen3:8b` pulled
- Docker (optional, for container deployment)

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b
```

## Configuration

```bash
cp .env.example .env
```

Set `GATEWAY_API_KEYS` in `.env`. Each key is an identity:

```json
[{"id":"local-dev","key":"sk-local-dev","allow_models":["fast","general"]}]
```

Use `["*"]` to allow every configured logical model. Raw keys are never logged.

Model routing lives in `config/models.yaml`. Override Ollama’s URL with `OLLAMA_BASE_URL` (defaults to `http://127.0.0.1:11434`).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Run with Docker

Ollama remains a host service bound to `127.0.0.1:11434`. The gateway is the only LAN-facing process (`0.0.0.0:8080`).

Compose uses `network_mode: host` so the container can reach that loopback address. Do not bind Ollama to `0.0.0.0` — that would expose it on the LAN.

```bash
cp .env.example .env
docker compose up --build
```

On Docker Desktop (Windows/Mac), enable **Settings > Resources > Network > Enable host networking** so host mode can use the host loopback interface. On Linux this is native.

Ollama is not published by this compose file.

## API

Authenticate every `/v1` request:

```
Authorization: Bearer <api-key>
```

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Liveness, no auth |
| `GET` | `/v1/models` | Lists `fast` and `general` |
| `POST` | `/v1/chat/completions` | JSON or SSE when `"stream": true` |

```bash
curl http://127.0.0.1:8080/v1/models ^
  -H "Authorization: Bearer sk-local-dev"

curl http://127.0.0.1:8080/v1/chat/completions ^
  -H "Authorization: Bearer sk-local-dev" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"fast\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

OpenAI Python client from another LAN workstation:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<gateway-host>:8080/v1", api_key="sk-local-dev")
print(
    client.chat.completions.create(
        model="fast",
        messages=[{"role": "user", "content": "Hello"}],
    )
)
```

`X-Request-ID` and `X-Request-Timestamp` are returned on every response. Structured JSON logs include that same request timestamp, request ID, identity id, logical model, backend, backend model, status, and latency — not prompts, responses, or API keys.

## Tests

```bash
python -m pytest
```

Unit tests do not need Ollama. Integration tests run against a local Ollama instance and skip if it (or the required models) is unavailable.

CI runs Ruff and unit tests on every pull request (`python -m pytest -m "not integration"`).
