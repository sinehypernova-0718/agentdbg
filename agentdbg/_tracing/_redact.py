"""Helpers for truncating and redacting trace data before it hits disk."""

import json
import re
import traceback
from typing import Any

from agentdbg.config import AgentDbgConfig
from agentdbg.constants import DEPTH_LIMIT, REDACTED_MARKER, TRUNCATED_MARKER

# TODO: fold this back into DEPTH_LIMIT once the old name is no longer used.
_RECURSION_LIMIT = DEPTH_LIMIT


def _key_matches_redact(key: str, redact_keys: list[str]) -> bool:
    """Return True when the key looks sensitive enough to redact."""
    k = key.lower()
    return any(rk.lower() in k for rk in redact_keys)


# Matches `--option=value` or `-o=value`.
_ARGV_OPTION_VALUE = re.compile(r"^(-{1,2})([a-zA-Z0-9_-]+)=(.*)$")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,})\b")
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([a-z0-9._~+/=-]{8,})")


def _redact_argv(argv: list[str], config: AgentDbgConfig) -> list[str]:
    """
    Redact sensitive CLI flag values like `--api-key=...`.

    We only touch `key=value` style arguments here and leave positional args
    alone. The original list is not mutated.
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
    """Trim a string so the final UTF-8 payload still fits in `max_bytes`."""
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
    Recursively trim values without doing any key-based redaction.

    This runs on the producer side so queued payloads do not grow without bound.
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
    Recursively redact sensitive keys and trim oversized string values.

    We keep the traversal shallow enough to avoid runaway nesting and always
    build a fresh structure instead of mutating the input.
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
    """Normalize usage data into the token fields we expose downstream."""
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
    """Apply the producer-side redaction/truncation pass to payload and meta."""
    return (
        _redact_and_truncate(payload, config)
        if config.redact
        else _truncate_only(payload, config),
        (
            _redact_and_truncate(meta, config)
            if config.redact
            else _truncate_only(meta, config)
        )
        if meta is not None
        else {},
    )


def _build_error_payload(
    exc_or_message: BaseException | str | dict[str, Any] | None,
    config: AgentDbgConfig,
    include_stack: bool = True,
) -> dict[str, Any] | None:
    """
    Build a consistent error payload for tool and LLM events.

    The shape matches the ERROR event payload closely so downstream consumers do
    not have to special-case it.
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
        # Older callers may still pass `type` instead of `error_type`.
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
    """Catch token shapes that slip through structured redaction."""
    text = _OPENAI_KEY_RE.sub(REDACTED_MARKER, text)
    text = _GITHUB_TOKEN_RE.sub(REDACTED_MARKER, text)
    return _BEARER_RE.sub(r"\1" + REDACTED_MARKER, text)


def _serialize_event_for_storage(event: dict[str, Any], config: AgentDbgConfig) -> str:
    """
    Prepare one event for `events.jsonl`.

    The worker does the deeper redaction pass right before writing so cached
    queue items can stay lightweight, and then we do one last regex scrub on the
    serialized JSON for token-looking strings.
    """
    safe_event = (
        _redact_and_truncate(event, config)
        if config.redact
        else _truncate_only(event, config)
    )
    line = json.dumps(safe_event, ensure_ascii=False, separators=(",", ":"))
    return _scrub_serialized_json_text(line)
