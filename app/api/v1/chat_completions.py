from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.auth.dependencies import get_identity
from app.errors.exceptions import GatewayError
from app.observability.request_context import set_error_class
from app.schemas.internal import ChatChunk, ChatMessage, ChatRequest, ChatResult, Identity
from app.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from app.services.chat_service import ChatService

router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    identity: Identity = Depends(get_identity),
    service: ChatService = Depends(get_chat_service),
):
    internal = _to_internal_request(payload)
    # Authorize and resolve before headers are sent so 403/404 are not buried in a 200 stream.
    service.prepare(identity, payload.model)
    if payload.stream:
        backend_stream = service.stream_chat(identity, payload.model, internal)
        try:
            first_chunk = await anext(backend_stream)
        except StopAsyncIteration:
            first_chunk = None
        return StreamingResponse(
            _sse_stream(backend_stream, first_chunk, payload.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    result = await service.complete_chat(identity, payload.model, internal)
    return _to_completion_response(
        result,
        completion_id=_completion_id(),
        created=int(time.time()),
        model=payload.model,
    )


def _to_internal_request(payload: ChatCompletionRequest) -> ChatRequest:
    stop: str | tuple[str, ...] | None
    if isinstance(payload.stop, list):
        stop = tuple(payload.stop)
    else:
        stop = payload.stop
    return ChatRequest(
        messages=tuple(ChatMessage(role=message.role, content=message.content) for message in payload.messages),
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stop=stop,
    )


def _to_completion_response(
    result: ChatResult,
    *,
    completion_id: str,
    created: int,
    model: str,
) -> ChatCompletionResponse:
    usage = None
    if result.prompt_tokens is not None and result.completion_tokens is not None:
        usage = ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=usage,
    )


async def _sse_stream(
    backend_stream: AsyncIterator[ChatChunk],
    first_chunk: ChatChunk | None,
    logical_model: str,
) -> AsyncIterator[str]:
    completion_id = _completion_id()
    created = int(time.time())
    sent_role = False

    def emit(chunk: ChatChunk) -> str:
        nonlocal sent_role
        delta = ChatCompletionChunkDelta()
        if chunk.role and not sent_role:
            delta.role = "assistant"
            sent_role = True
        if chunk.content:
            delta.content = chunk.content
        event = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=logical_model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=chunk.finish_reason,
                )
            ],
        )
        return f"data: {event.model_dump_json()}\n\n"

    try:
        if first_chunk is not None:
            yield emit(first_chunk)
        async for chunk in backend_stream:
            yield emit(chunk)
        yield "data: [DONE]\n\n"
    except GatewayError as exc:
        set_error_class(type(exc).__name__)
        return


def _completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"
