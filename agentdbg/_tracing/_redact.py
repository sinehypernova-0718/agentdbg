"""
Pure redaction and truncation utilities.
Only depends on agentdbg.constants and agentdbg.config (for AgentDbgConfig type).
"""

import json
import re
import traceback
from typing import Any

from agentdbg.config import AgentDbgConfig
from agentdbg.constants import DEPTH_LIMIT, REDACTED_MARKER, TRUNCATED_MARKER

# TODO: Remove the _RECURSION_LIMIT and use DEPTH_LIMIT instead
_RECURSION_LIMIT = DEPTH_LIMIT


def _key_matches_redact(key: str, redact_keys: list[str]) -> bool:
    """True if key matches any redact key (case-insensitive substring)."""
    k = key.lower()
    return any(rk.lower() in k for rk in redact_keys)


# Matches --option=value or -o=value (option name can have letters, digits, hyphens, underscores).
_ARGV_OPTION_VALUE = re.compile(r"^(-{1,2})([a-zA-Z0-9_-]+)=(.*)$")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=-]{8,})")


def _redact_argv(argv: list[str], config: AgentDbgConfig) -> list[str]:
    """
    Redact only sensitive option values in argv. E.g. --api-key=sk-secret -> --api-key=__REDACTED__.
    Option name is matched against config.redact_keys (with hyphens normalized to underscores).
    Returns a new list; does not mutate input.
    """
    if not argv or not config.redact:
        return list(argv)
    out: list[str] = []
    for item in argv:
        match = _ARGV_OPTION_VALUE.match(item)
        if match:
            prefix, key, _value = match.groups()
            key_normalized = key.replace("-", "_")
            if _key_matches_redact(key_normalized, config.redact_keys):
                out.append(f"{prefix}{key}={REDACTED_MARKER}")
                continue
        out.append(item)
    return out


def _truncate_string(s: str, max_bytes: int) -> str:
    """Truncate string so result (including TRUNCATED_MARKER) fits in max_bytes. O(n), single encode/decode."""
    if max_bytes <= 0:
        return s
    enc = "utf-8"
    b = s.encode(enc)
    if len(b) <= max_bytes:
        return s
    marker_bytes = len(TRUNCATED_MARKER.encode(enc))
    limit = max(0, max_bytes - marker_bytes)
    b_trunc = b[:limit]
    return b_trunc.decode(enc, errors="ignore") + TRUNCATED_MARKER


def _truncate_only(
    obj: Any,
    config: AgentDbgConfig,
    depth: int = 0,
) -> Any:
    """
    Recursively truncate large values without performing key-based redaction.

    Used on the producer side to keep queued payloads bounded in size.
    """
    if depth > _RECURSION_LIMIT:
        return TRUNCATED_MARKER
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _truncate_string(obj, config.max_field_bytes)
    if isinstance(obj, dict):
        return {str(k): _truncate_only(v, config, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_truncate_only(item, config, depth + 1) for item in obj]
    s = str(obj)
    return (
        _truncate_string(s, config.max_field_bytes)
        if len(s.encode("utf-8")) > config.max_field_bytes
        else s
    )


def _redact_and_truncate(
    obj: Any,
    config: AgentDbgConfig,
    depth: int = 0,
) -> Any:
    """
    Recursively redact keys matching config.redact_keys and truncate large strings.
    Limit recursion to _RECURSION_LIMIT. Returns a new structure; does not mutate input.
    """
    if depth > _RECURSION_LIMIT:
        return TRUNCATED_MARKER
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _truncate_string(obj, config.max_field_bytes)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key_str = str(k)
            if config.redact and _key_matches_redact(key_str, config.redact_keys):
                out[key_str] = REDACTED_MARKER
            else:
                out[key_str] = _redact_and_truncate(v, config, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact_and_truncate(item, config, depth + 1) for item in obj]
    s = str(obj)
    return (
        _truncate_string(s, config.max_field_bytes)
        if len(s.encode("utf-8")) > config.max_field_bytes
        else s
    )


def _normalize_usage(usage: Any) -> dict[str, int | None] | None:
    """Normalize LLM usage to shape: prompt_tokens, completion_tokens, total_tokens (null if unknown)."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return None

    def _token_val(key: str) -> int | None:
        v = usage.get(key)
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            try:
                return int(v)
            except (OverflowError, ValueError):
                return None
        return None

    return {
        "prompt_tokens": _token_val("prompt_tokens"),
        "completion_tokens": _token_val("completion_tokens"),
        "total_tokens": _token_val("total_tokens"),
    }


def _apply_redaction_truncation(
    payload: Any, meta: Any, config: AgentDbgConfig
) -> tuple[Any, Any]:
    """Apply producer-side truncation to payload and meta; returns (payload, meta)."""
    return (
        _truncate_only(payload, config),
        _truncate_only(meta, config) if meta is not None else {},
    )


def _build_error_payload(
    exc_or_message: BaseException | str | dict[str, Any] | None,
    config: AgentDbgConfig,
    include_stack: bool = True,
) -> dict[str, Any] | None:
    """
    Build a consistent error object for TOOL_CALL/LLM_CALL payloads.
    Returns None if exc_or_message is None; otherwise dict with error_type, message, optional details, optional stack.
    Uses error_type (same as ERROR event payload per SPEC) for consumer consistency.
    Result is truncated per config before being handed to the background writer.
    """
    if exc_or_message is None:
        return None
    if isinstance(exc_or_message, BaseException):
        err = {
            "error_type": type(exc_or_message).__name__,
            "message": str(exc_or_message),
            "details": None,
            "stack": traceback.format_exc() if include_stack else None,
        }
    elif isinstance(exc_or_message, str):
        err = {
            "error_type": "Error",
            "message": exc_or_message,
            "details": None,
            "stack": None,
        }
    elif isinstance(exc_or_message, dict):
        # Accept both error_type and type for backward compatibility when building from dict
        err = {
            "error_type": exc_or_message.get("error_type")
            or exc_or_message.get("type", "Error"),
            "message": exc_or_message.get("message", ""),
            "details": exc_or_message.get("details"),
            "stack": exc_or_message.get("stack") if include_stack else None,
        }
    else:
        err = {
            "error_type": "Error",
            "message": str(exc_or_message),
            "details": None,
            "stack": None,
        }
    return _truncate_only(err, config)


def _scrub_serialized_json_text(text: str) -> str:
    """Redact known secret token shapes in serialized JSON text."""
    text = _OPENAI_KEY_RE.sub(REDACTED_MARKER, text)
    text = _GITHUB_TOKEN_RE.sub(REDACTED_MARKER, text)
    return _BEARER_RE.sub(r"\1" + REDACTED_MARKER, text)


def _serialize_event_for_storage(event: dict[str, Any], config: AgentDbgConfig) -> str:
    """
    Apply worker-side redaction and emit a compact JSON line safe for disk.

    Known token regexes are scrubbed even when key-based redaction is disabled.
    """
    safe_event = (
        _redact_and_truncate(event, config)
        if config.redact
        else _truncate_only(event, config)
    )
    line = json.dumps(safe_event, ensure_ascii=False, separators=(",", ":"))
    return _scrub_serialized_json_text(line)
