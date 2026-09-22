from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat_completions
from app.api.v1 import models as models_api

api_router = APIRouter(prefix="/v1")
api_router.include_router(models_api.router)
api_router.include_router(chat_completions.router)
