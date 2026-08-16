"""User-facing error formatting: raw provider errors → readable messages.

Regression coverage for the misleading-error bug: a transient relay failure
(e.g. 503 ``model_not_found``) that degraded into a chat-completions call and
surfaced ``protocol_not_supported`` as if the model configuration were wrong.
The formatter must name the *real* cause and never dump the raw dict body.
"""

from __future__ import annotations

from deeptutor.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    ProviderContextWindowError,
)
from deeptutor.utils.error_utils import format_user_facing_error


def test_sdk_wrapped_protocol_error_zh() -> None:
    exc = RuntimeError(
        "Error code: 400 - {'error': {'code': 'protocol_not_supported', "
        "'message': '模型 gpt-5.6-terra 不支持 chat completions 协议 "
        "(request id: 01M0X)', 'type': 'packy_api_error'}}"
    )
    msg = format_user_facing_error(exc, language="zh")
    # Names the real cause instead of dumping the raw body.
    assert "不支持当前调用协议" in msg
    assert "chat completions 协议" in msg
    assert "Error code" not in msg
    assert "packy_api_error" not in msg


def test_sdk_wrapped_protocol_error_en() -> None:
    exc = RuntimeError(
        "Error code: 400 - {'error': {'code': 'protocol_not_supported', "
        "'message': 'model gpt-5.6-terra does not support chat completions', "
        "'type': 'packy_api_error'}}"
    )
    msg = format_user_facing_error(exc, language="en")
    assert "does not support the requested protocol" in msg
    assert "chat completions" in msg


def test_model_not_found_503_zh() -> None:
    exc = RuntimeError(
        "Error code: 503 - {'error': {'code': 'model_not_found', 'message': "
        "'分组 codex 下模型 qwen3.8-max 无可用渠道（distributor），请尝试切换其他分组', "
        "'type': 'packy_api_error'}}"
    )
    msg = format_user_facing_error(exc, language="zh")
    # The real cause (channel/relay issue) is what must surface, not a
    # secondary protocol error.
    assert "无可用渠道" in msg or "模型当前不可用" in msg
    assert "qwen3.8-max" in msg


def test_unwrapped_relay_error_dict_zh() -> None:
    # Some gateways return the error body unwrapped (no ``error`` key).
    exc = RuntimeError(
        "{'code': 'model_not_found', 'message': '模型 gpt-5.6-terra 无可用渠道', "
        "'type': 'packy_api_error'}"
    )
    msg = format_user_facing_error(exc, language="zh")
    assert "模型当前不可用" in msg
    assert "gpt-5.6-terra" in msg


def test_status_based_rate_limit() -> None:
    exc = LLMRateLimitError(
        "Error code: 429 - {'error': {'message': 'Rate limit reached'}}",
        provider="openai",
    )
    msg = format_user_facing_error(exc, language="zh")
    assert "限流" in msg or "过于频繁" in msg


def test_status_based_auth() -> None:
    exc = LLMAuthenticationError("Invalid API key provided", provider="openai")
    msg = format_user_facing_error(exc, language="zh")
    assert "API 密钥" in msg


def test_context_window_error() -> None:
    exc = ProviderContextWindowError(
        "This model's maximum context length is 128000 tokens",
        provider="openai",
    )
    msg = format_user_facing_error(exc, language="zh")
    assert "上下文窗口" in msg


def test_plain_message_no_json() -> None:
    exc = RuntimeError("connection reset by peer")
    msg = format_user_facing_error(exc, language="zh")
    assert msg.startswith("模型调用失败")
    assert "connection reset by peer" in msg


def test_long_message_capped() -> None:
    exc = RuntimeError("boom " + "x" * 500)
    msg = format_user_facing_error(exc, language="zh")
    assert len(msg) < 400
