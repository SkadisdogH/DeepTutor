"""Protocol fallback decisions + adapter error surfacing.

Regression coverage for the misleading-error bug: ``OpenAICompatProvider``
must only degrade from the Responses API to chat/completions when the
Responses endpoint itself is missing/unsupported — NOT on transient
provider-side failures (5xx, 429, ``model_not_found``), which would surface
a misleading ``protocol_not_supported`` error for Responses-only models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deeptutor.core.agentic.client import _ProviderOpenAIStream
from deeptutor.services.llm.provider_core.base import LLMResponse
from deeptutor.services.llm.provider_core.openai_compat_provider import (
    OpenAICompatProvider,
)


class _FakeAPIError(RuntimeError):
    """Fake an ``openai.APIStatusError``-shaped exception (raisable)."""

    def __init__(self, status_code: int, body_text: str) -> None:
        super().__init__(f"Error code: {status_code} - {body_text}")
        self.status_code = status_code
        self.body = body_text
        self.response = SimpleNamespace(status_code=status_code, text=body_text)
        self.message = str(self)


def _sdk_error(status_code: int, body_text: str) -> Exception:
    return _FakeAPIError(status_code, body_text)


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        # Endpoint/protocol missing → fallback to chat/completions is OK.
        (404, '{"error": {"message": "not found"}}', True),
        (405, '{"error": {"message": "method not allowed"}}', True),
        (400, '{"error": {"message": "The Responses API is not supported"}}', True),
        (422, '{"error": {"message": "unknown parameter"}}', True),
        (501, '{"error": {"message": "not implemented"}}', True),
        # Transient/provider-side failures → must NOT fall back, or the real
        # cause (relay outage / no available channel) gets hidden behind a
        # misleading secondary error.
        (503, '{"error": {"code": "model_not_found", "message": "无可用渠道"}}', False),
        (500, '{"error": {"message": "internal server error"}}', False),
        (429, '{"error": {"message": "rate limit"}}', False),
        (408, '{"error": {"message": "timeout"}}', False),
        (401, '{"error": {"message": "invalid key"}}', False),
    ],
)
def test_should_fallback_from_responses_error(status_code: int, body: str, expected: bool) -> None:
    exc = _sdk_error(status_code, body)
    assert OpenAICompatProvider._should_fallback_from_responses_error(exc) is expected


class _FakeProvider:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response

    async def chat_stream(self, **kwargs: object) -> LLMResponse:
        return self._response


async def _drain(stream: _ProviderOpenAIStream) -> list[object]:
    chunks: list[object] = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_stream_raises_on_error_response_instead_of_emitting_text() -> None:
    provider = _FakeProvider(
        LLMResponse(
            content="Error: {'code': 'protocol_not_supported', 'message': 'x', 'type': 'y'}",
            finish_reason="error",
        )
    )
    stream = _ProviderOpenAIStream(
        provider=provider,
        messages=[],
        tools=None,
        model="gpt-5.6-terra",
        max_tokens=100,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
        extra_kwargs={},
    )
    with pytest.raises(RuntimeError, match="protocol_not_supported"):
        await _drain(stream)


@pytest.mark.asyncio
async def test_stream_emits_normal_content() -> None:
    provider = _FakeProvider(LLMResponse(content="hello", finish_reason="stop"))
    stream = _ProviderOpenAIStream(
        provider=provider,
        messages=[],
        tools=None,
        model="gpt-5.6-terra",
        max_tokens=100,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
        extra_kwargs={},
    )
    chunks = await _drain(stream)
    assert chunks
    # First chunk carries the content delta.
    first = chunks[0]
    assert first.choices[0].delta.content == "hello"


class _FakeClient:
    """Stub of the AsyncOpenAI surface used by OpenAICompatProvider."""

    def __init__(self, responses_error: Exception) -> None:
        self._responses_error = responses_error
        self.responses_calls = 0
        self.chat_completions_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.responses = SimpleNamespace(create=self._responses_create)

    async def _responses_create(self, **kwargs: object) -> object:
        self.responses_calls += 1
        raise self._responses_error

    async def _chat_create(self, **kwargs: object) -> object:
        self.chat_completions_calls += 1
        raise _sdk_error(
            400,
            "{'error': {'code': 'protocol_not_supported', 'message': "
            "'模型 gpt-5.6-terra 不支持 chat completions 协议', 'type': 'packy_api_error'}}",
        )


async def _provider_with_client(responses_error: Exception) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(
        api_key="sk-test",
        api_base="https://relay.test/v1",
        default_model="gpt-5.6-terra",
        spec=None,
        provider_name="openai",
    )
    provider._client = _FakeClient(responses_error)  # type: ignore[attr-defined]
    return provider


@pytest.mark.asyncio
async def test_chat_surfaces_real_error_without_degrading_to_chat_completions() -> None:
    # The exact production scenario: the relay is having an outage (503,
    # ``model_not_found`` / no available channel). The provider must surface
    # THAT error instead of degrading into a chat-completions call that only
    # returns a misleading ``protocol_not_supported`` for a Responses-only
    # model.
    provider = await _provider_with_client(
        _sdk_error(
            503,
            "{'error': {'code': 'model_not_found', 'message': "
            "'分组 codex 下模型 gpt-5.6-terra 无可用渠道', 'type': 'packy_api_error'}}",
        )
    )
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.6-terra",
    )
    assert result.finish_reason == "error"
    assert "model_not_found" in result.content
    assert "protocol_not_supported" not in result.content
    client = provider._client  # type: ignore[attr-defined]
    assert client.responses_calls == 1
    assert client.chat_completions_calls == 0


@pytest.mark.asyncio
async def test_chat_stream_surfaces_real_error_without_degrading() -> None:
    provider = await _provider_with_client(
        _sdk_error(
            503,
            "{'error': {'code': 'model_not_found', 'message': '无可用渠道', "
            "'type': 'packy_api_error'}}",
        )
    )
    result = await provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.6-terra",
    )
    assert result.finish_reason == "error"
    assert "model_not_found" in result.content
    client = provider._client  # type: ignore[attr-defined]
    assert client.responses_calls == 1
    assert client.chat_completions_calls == 0


@pytest.mark.asyncio
async def test_chat_falls_back_when_responses_endpoint_missing() -> None:
    # A 404 from the Responses route is a genuine protocol-absence signal:
    # the fallback to chat/completions stays enabled (e.g. models that only
    # speak chat completions keep working).
    provider = await _provider_with_client(
        _sdk_error(404, '{"error": {"message": "not found"}}')
    )
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3.8-max",
    )
    assert result.finish_reason == "error"
    client = provider._client  # type: ignore[attr-defined]
    assert client.responses_calls == 1
    assert client.chat_completions_calls == 1
