from __future__ import annotations

from app.config import ModelRouteConfig
from app.errors.exceptions import UnknownModelError
from app.schemas.internal import ResolvedRoute


class ModelRouter:
    def __init__(self, models: dict[str, ModelRouteConfig]) -> None:
        self._models = models

    def resolve(self, logical_model: str) -> ResolvedRoute:
        config = self._models.get(logical_model)
        if config is None:
            raise UnknownModelError(logical_model)
        return ResolvedRoute(
            logical_model=logical_model,
            backend=config.backend,
            backend_model=config.model,
        )

    def list_logical_models(self) -> list[str]:
        return list(self._models.keys())
