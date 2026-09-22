from __future__ import annotations

import pytest
from app.schemas.openai import ChatCompletionRequest
from pydantic import ValidationError


def test_valid_request() -> None:
    payload = ChatCompletionRequest.model_validate(
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "max_tokens": 16,
        }
    )
    assert payload.model == "fast"
    assert payload.stream is False


def test_rejects_tools() -> None:
    with pytest.raises(ValidationError, match="tools"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            }
        )


def test_rejects_n_greater_than_one() -> None:
    with pytest.raises(ValidationError, match="n greater than 1"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            }
        )


def test_rejects_multimodal_content() -> None:
    with pytest.raises(ValidationError, match="multimodal"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            }
        )


def test_rejects_functions() -> None:
    with pytest.raises(ValidationError, match="functions"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [{"role": "user", "content": "hello"}],
                "functions": [{"name": "do"}],
            }
        )


def test_rejects_unknown_request_field() -> None:
    with pytest.raises(ValidationError, match="not_a_real_parameter"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [{"role": "user", "content": "hello"}],
                "not_a_real_parameter": True,
            }
        )


def test_rejects_unknown_message_field() -> None:
    with pytest.raises(ValidationError, match="mystery"):
        ChatCompletionRequest.model_validate(
            {
                "model": "fast",
                "messages": [{"role": "user", "content": "hello", "mystery": "x"}],
            }
        )
