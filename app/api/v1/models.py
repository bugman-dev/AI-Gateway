from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.auth.dependencies import get_identity
from app.config import Settings
from app.schemas.internal import Identity
from app.schemas.openai import ModelListResponse, ModelObject

router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    _identity: Identity = Depends(get_identity),
    settings: Settings = Depends(get_settings),
) -> ModelListResponse:
    created = int(time.time())
    return ModelListResponse(
        data=[
            ModelObject(id=name, created=created, owned_by="ai-gateway")
            for name in settings.models
        ]
    )
