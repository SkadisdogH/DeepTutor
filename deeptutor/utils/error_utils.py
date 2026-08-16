#!/usr/bin/env python
"""
Error Utilities - Error formatting and handling utilities
"""

import ast
import json
from typing import Optional


def _find_json_block(message: str) -> Optional[str]:
    """Extract potential JSON block from message by matching braces."""
    start_idx = message.find("{")
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for char_idx in range(start_idx, len(message)):
        char = message[char_idx]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return message[start_idx : char_idx + 1]

    return None


def format_exception_message(exc: Exception) -> str:
    """
    Format exception message for better readability

    Args:
        exc: The exception to format

    Returns:
        Formatted error message
    """
    message = str(exc)

    # Try to parse JSON error messages (common in API errors)
    potential_json = _find_json_block(message)
    if potential_json:
        try:
            error_data = json.loads(potential_json)

            # Standard extraction logic
            if isinstance(error_data, dict) and "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    parts = []
                    if "message" in error_info:
                        parts.append(f"Message: {error_info['message']}")
                    if "type" in error_info:
                        parts.append(f"Type: {error_info['type']}")
                    if "code" in error_info:
                        parts.append(f"Code: {error_info['code']}")
                    if parts:
                        return " | ".join(parts)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Return original message if parsing fails
    return message


# ---------------------------------------------------------------------------
# User-facing error formatting
# ---------------------------------------------------------------------------
#
# LLM/provider failures currently travel to the UI as ``str(exc)`` — often a
# raw provider error body like ``Error code: 400 - {'error': {'code':
# 'protocol_not_supported', 'message': ..., 'type': ...}}``. Showing that
# verbatim misleads: a transient relay outage that surfaces as one error can
# be masked by the *next* error the fallback path triggers (e.g. a 503
# ``model_not_found`` degrades into a 400 ``protocol_not_supported``). These
# helpers extract the provider's code/message and turn them into a readable
# line that names the real cause.

_PROVIDER_STATUS_HINTS_ZH: dict[int, str] = {
    400: "模型调用失败（请求被拒绝）",
    401: "API 密钥无效或未配置，请检查模型设置",
    403: "API 密钥无权访问该模型",
    404: "模型或接口不存在，请检查模型配置",
    408: "模型响应超时，请稍后重试",
    429: "请求过于频繁（已限流），请稍后重试",
    500: "模型服务内部错误，请稍后重试",
    502: "模型服务暂时不可用，请稍后重试",
    503: "模型服务暂时不可用（服务商/渠道异常），请稍后重试或切换模型",
}

_PROVIDER_STATUS_HINTS_EN: dict[int, str] = {
    400: "Model request rejected",
    401: "API key is invalid or missing — check the model settings",
    403: "API key is not authorized for this model",
    404: "Model or endpoint not found — check the model configuration",
    408: "Model request timed out — please retry later",
    429: "Too many requests (rate limited) — please retry later",
    500: "Model service internal error — please retry later",
    502: "Model service temporarily unavailable — please retry later",
    503: "Model service temporarily unavailable (provider/channel issue) — retry later or switch models",
}

_PROVIDER_CODE_HINTS_ZH: dict[str, str] = {
    "protocol_not_supported": "模型不支持当前调用协议（可能是模型与接口协议不匹配）",
    "model_not_found": "模型当前不可用（无可用渠道/分组，或模型已被移除）",
    "model_not_configured": "模型未配置，请在设置中选择可用模型",
    "invalid_api_key": "API 密钥无效，请检查模型设置",
    "insufficient_quota": "账户额度不足，请检查充值或切换模型",
    "context_length_exceeded": "内容超出模型上下文窗口",
}

_PROVIDER_CODE_HINTS_EN: dict[str, str] = {
    "protocol_not_supported": "Model does not support the requested protocol (model/endpoint mismatch)",
    "model_not_found": "Model currently unavailable (no available channel/group, or model removed)",
    "model_not_configured": "Model is not configured — pick an available model in settings",
    "invalid_api_key": "Invalid API key — check the model settings",
    "insufficient_quota": "Account quota exhausted — top up or switch models",
    "context_length_exceeded": "Content exceeds the model context window",
}

_CONTEXT_WINDOW_MARKERS = ("context length", "maximum context", "context_length_exceeded")
_TIMEOUT_MARKERS = ("timeout", "timed out", "timed_out", "deadline exceeded")


def _extract_provider_error(exc: Exception) -> tuple[str, str, int | None]:
    """Pull ``(code, message, status_code)`` out of an exception.

    Understands the shapes providers actually emit: the OpenAI SDK's
    ``Error code: 400 - {'error': {...}}``, the unwrapped
    ``{'code': ..., 'message': ..., 'type': ...}`` some gateways return, and
    DeepTutor's own ``LLMAPIError`` subclasses (which carry ``status_code``).
    """
    status_code = getattr(exc, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    message = str(getattr(exc, "message", None) or exc)
    code = ""
    block = _find_json_block(message)
    if block:
        data = None
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            # Provider bodies often arrive as ``str(dict)`` — a Python repr
            # with single quotes (``{'error': {'code': ...}}``) that is not
            # valid JSON. ``ast.literal_eval`` parses exactly those shapes.
            try:
                data = ast.literal_eval(block)
            except (ValueError, SyntaxError):
                data = None
        if isinstance(data, dict):
            if isinstance(data.get("error"), dict):
                data = data["error"]
            if isinstance(data.get("error"), str):
                code = code or str(data["error"])
            code = code or str(data.get("code") or "").strip()
            extracted_message = str(data.get("message") or "").strip()
            if extracted_message:
                message = extracted_message
    return code, message, status_code


def format_user_facing_error(exc: Exception, language: str = "zh") -> str:
    """Render an LLM/provider exception as a short, actionable user message.

    The returned string names the *real* cause (channel outage, auth, rate
    limit, protocol mismatch, …) instead of dumping the raw error body. The
    full exception is still available in backend logs for debugging.
    """
    zh = language.lower().startswith("zh")
    raw = str(exc)
    code, message, status_code = _extract_provider_error(exc)

    hint: str | None = None
    if status_code is not None:
        hints = _PROVIDER_STATUS_HINTS_ZH if zh else _PROVIDER_STATUS_HINTS_EN
        hint = hints.get(status_code)
    if hint is None:
        hints = _PROVIDER_CODE_HINTS_ZH if zh else _PROVIDER_CODE_HINTS_EN
        hint = hints.get(code)
    if hint is None:
        lower = raw.lower()
        if any(marker in lower for marker in _CONTEXT_WINDOW_MARKERS):
            hint = (
                "内容超出模型上下文窗口，请精简问题或新建会话"
                if zh
                else "Content exceeds the model context window — shorten the question or start a new session"
            )
        elif any(marker in lower for marker in _TIMEOUT_MARKERS):
            hint = "模型响应超时，请稍后重试" if zh else "Model request timed out — please retry later"
        elif "无可用渠道" in message or "no available channel" in lower:
            hint = (
                "模型当前不可用（服务商/渠道异常），请稍后重试或切换模型"
                if zh
                else "Model currently unavailable (provider/channel issue) — retry later or switch models"
            )
    if hint is None:
        hint = "模型调用失败" if zh else "Model request failed"

    # The provider message often repeats the code in English (e.g. the code
    # itself) — keep the human line and cap its length.
    message = (message or "").strip()
    if message and message.lower() in {code, code.lower()}:
        message = ""
    if len(message) > 300:
        message = message[:300] + "…"

    separator = "：" if zh else ": "
    return f"{hint}{separator}{message}" if message else hint
