from __future__ import annotations

import pytest

from app.errors.exceptions import UnknownModelError
from app.routing.model_router import ModelRouter
from tests.conftest import make_settings


def test_resolves_fast_to_qwen3_4b() -> None:
    router = ModelRouter(make_settings().models)
    route = router.resolve("fast")
    assert route.backend == "ollama"
    assert route.backend_model == "qwen3:4b"


def test_resolves_general_to_qwen3_8b() -> None:
    router = ModelRouter(make_settings().models)
    route = router.resolve("general")
    assert route.backend == "ollama"
    assert route.backend_model == "qwen3:8b"


def test_unknown_logical_model() -> None:
    router = ModelRouter(make_settings().models)
    with pytest.raises(UnknownModelError):
        router.resolve("gpt-4")
