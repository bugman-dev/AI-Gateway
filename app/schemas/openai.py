from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def content_must_be_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("multimodal content is not supported")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    n: int | None = None
    tools: Any = None
    functions: Any = None
    tool_choice: Any = None
    response_format: Any = None
    logit_bias: Any = None
    logprobs: Any = None
    modalities: Any = None

    @model_validator(mode="after")
    def reject_unsupported_behavior(self) -> ChatCompletionRequest:
        if self.n is not None and self.n != 1:
            raise ValueError("n greater than 1 is not supported")
        if self.tools:
            raise ValueError("tools are not supported")
        if self.functions:
            raise ValueError("functions are not supported")
        if self.tool_choice is not None:
            raise ValueError("tool_choice is not supported")
        if self.response_format is not None:
            raise ValueError("response_format is not supported")
        if self.logit_bias is not None:
            raise ValueError("logit_bias is not supported")
        if self.logprobs:
            raise ValueError("logprobs are not supported")
        if self.modalities:
            raise ValueError("modalities are not supported")
        if self.role_is_tool():
            raise ValueError("tool role messages are not supported")
        return self

    def role_is_tool(self) -> bool:
        return any(message.role == "tool" for message in self.messages)


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "ai-gateway"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
