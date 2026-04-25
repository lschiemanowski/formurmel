from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from formurmel.message import ToolSpec
from formurmel.tools.base import Tool, tool_error, tool_ok


_MAX_LIMIT = 50
_MAX_CHARS = 20000


@dataclass(frozen=True)
class _TranscriptMessage:
    index: int
    role: str
    msg_type: str | None
    tool_name: str | None
    text: str
    raw: Mapping[str, Any]


def _as_int(value: Any, name: str, *, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    if value is None:
        result = default
    elif isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    else:
        result = value
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _as_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _as_bool(value: Any, name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 20:
        return text[:max_chars], True
    return f"{text[: max_chars - 14]}\n...[truncated]", True


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_tool_name(raw: Mapping[str, Any]) -> str | None:
    content = raw.get("content")
    if isinstance(content, Mapping):
        name = content.get("name")
        if isinstance(name, str):
            return name
    return None


def _normalize_messages(payload: Mapping[str, Any]) -> list[_TranscriptMessage]:
    raw_messages = payload.get("conversation")
    if raw_messages is None:
        raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("transcript must contain a conversation or messages list")

    messages: list[_TranscriptMessage] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            continue
        role = str(raw.get("role", "unknown"))
        msg_type_value = raw.get("msg_type")
        msg_type = str(msg_type_value) if msg_type_value is not None else None
        tool_name = _extract_tool_name(raw)
        text = _json_text(raw)
        messages.append(
            _TranscriptMessage(
                index=index,
                role=role,
                msg_type=msg_type,
                tool_name=tool_name,
                text=text,
                raw=dict(raw),
            )
        )
    return messages


def _content_id(message: _TranscriptMessage) -> str | None:
    content = message.raw.get("content")
    if not isinstance(content, Mapping):
        return None
    value = content.get("id")
    return value if isinstance(value, str) else None


def _nearest_preceding_tool_call_index(message: _TranscriptMessage, messages: list[_TranscriptMessage]) -> int | None:
    if message.msg_type != "tool_response" and message.role != "tool":
        return None

    response_id = _content_id(message)
    nearest_index: int | None = None
    for candidate in reversed(messages):
        if candidate.index >= message.index or candidate.msg_type != "tool_call":
            continue
        if nearest_index is None:
            nearest_index = candidate.index
        if response_id is not None and _content_id(candidate) == response_id:
            return candidate.index
    return nearest_index


def _tool_call_index_fields(message: _TranscriptMessage, messages: list[_TranscriptMessage]) -> dict[str, int | None]:
    return {
        "previous_tool_call_index": _nearest_preceding_tool_call_index(message, messages),
        "self_tool_call_index": message.index if message.msg_type == "tool_call" else None,
    }


def _message_record(message: _TranscriptMessage, *, max_chars: int, messages: list[_TranscriptMessage]) -> dict[str, Any]:
    snippet, truncated = _truncate(message.text, max_chars)
    return {
        "index": message.index,
        "role": message.role,
        "msg_type": message.msg_type,
        "tool_name": message.tool_name,
        **_tool_call_index_fields(message, messages),
        "text": snippet,
        "truncated": truncated,
    }


def _tool_payload(raw: Mapping[str, Any]) -> Any:
    content = raw.get("content")
    if not isinstance(content, Mapping):
        return None
    return content.get("content")


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _looks_like_error(message: _TranscriptMessage) -> bool:
    raw = message.raw
    content = raw.get("content")
    if isinstance(content, Mapping) and content.get("is_error") is True:
        return True
    payload = _tool_payload(raw)
    if payload is None:
        return False
    for value in _walk(payload):
        if not isinstance(value, Mapping):
            continue
        if value.get("ok") is False or value.get("success") is False:
            return True
        exit_code = value.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        error = value.get("error")
        if isinstance(error, str) and error.strip():
            return True
    text = _json_text(payload)
    return "error:" in text or "error(" in text or "failed" in text.lower()


def _payload_error_summary(payload: Any) -> str | None:
    if payload is None:
        return None
    best: str | None = None
    for value in _walk(payload):
        if isinstance(value, Mapping):
            error = value.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            stdout = value.get("stdout")
            if isinstance(stdout, str) and ("error:" in stdout or "error(" in stdout):
                best = stdout.strip()
            stderr = value.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                best = stderr.strip()
    if best:
        return best
    text = _json_text(payload)
    if "failed" in text.lower():
        return text
    return None


class TranscriptInspectTool(Tool):
    """Read-only inspection tool for one saved formurmel/cleaner transcript."""

    def __init__(self, transcript_path: str | Path) -> None:
        self.transcript_path = Path(transcript_path).expanduser().resolve()
        self._payload: Mapping[str, Any] | None = None
        self._messages: list[_TranscriptMessage] | None = None

    @property
    def name(self) -> str:
        return "transcript_inspect"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Inspect the failed agent transcript for diagnosis. Use summary first, then errors, search, "
                "messages, or message to ground claims in exact transcript indices. Use action=message with "
                "index for one exact message; action=messages returns a window and accepts start, with index "
                "as a start alias for compatibility. Message-like results include previous_tool_call_index "
                "for tool responses and self_tool_call_index for tool calls. This tool is read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "errors", "search", "messages", "message"],
                    },
                    "query": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "start": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT},
                    "index": {"type": "integer", "minimum": 0},
                    "role": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 200, "maximum": _MAX_CHARS},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

    def _load(self) -> tuple[Mapping[str, Any], list[_TranscriptMessage]]:
        if self._payload is not None and self._messages is not None:
            return self._payload, self._messages
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"transcript not found: {self.transcript_path}")
        with self.transcript_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("transcript root must be a JSON object")
        messages = _normalize_messages(payload)
        self._payload = dict(payload)
        self._messages = messages
        return self._payload, self._messages

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = _as_optional_str(payload.get("action"), "action")
        if action is None:
            return tool_error("missing action")
        try:
            if action == "summary":
                return tool_ok(self._summary(payload))
            if action == "errors":
                return tool_ok(self._errors(payload))
            if action == "search":
                return tool_ok(self._search(payload))
            if action == "messages":
                return tool_ok(self._messages_action(payload))
            if action == "message":
                return tool_ok(self._message(payload))
        except (ValueError, re.error) as exc:
            return tool_error(str(exc))
        return tool_error(f"unsupported action: {action}")

    def _summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        max_chars = _as_int(payload.get("max_chars"), "max_chars", default=3000, minimum=200, maximum=_MAX_CHARS)
        transcript, messages = self._load()
        role_counts: dict[str, int] = {}
        tool_calls: dict[str, int] = {}
        tool_responses: dict[str, int] = {}
        error_indices: list[int] = []
        for message in messages:
            role_counts[message.role] = role_counts.get(message.role, 0) + 1
            if message.msg_type == "tool_call":
                name = message.tool_name or "unknown"
                tool_calls[name] = tool_calls.get(name, 0) + 1
            if message.msg_type == "tool_response" or message.role == "tool":
                name = message.tool_name or "unknown"
                tool_responses[name] = tool_responses.get(name, 0) + 1
                if _looks_like_error(message):
                    error_indices.append(message.index)

        final_message = transcript.get("final_message")
        final_snippet = None
        if final_message is not None:
            final_snippet, _ = _truncate(_json_text(final_message), max_chars)
        return {
            "transcript_path": str(self.transcript_path),
            "status": transcript.get("status"),
            "steps": transcript.get("steps"),
            "error": transcript.get("error"),
            "message_count": len(messages),
            "role_counts": role_counts,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "error_message_indices": error_indices[:_MAX_LIMIT],
            "error_message_count": len(error_indices),
            "final_message": final_snippet,
        }

    def _iter_filtered(self, payload: Mapping[str, Any]) -> list[_TranscriptMessage]:
        _, messages = self._load()
        role = _as_optional_str(payload.get("role"), "role")
        tool_name = _as_optional_str(payload.get("tool_name"), "tool_name")
        filtered = messages
        if role is not None:
            filtered = [message for message in filtered if message.role == role]
        if tool_name is not None:
            filtered = [message for message in filtered if message.tool_name == tool_name]
        return filtered

    def _messages_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        used_index_alias = "start" not in payload and "index" in payload
        start_value = payload.get("index") if used_index_alias else payload.get("start")
        start_name = "index" if used_index_alias else "start"
        start = _as_int(start_value, start_name, default=0, minimum=0)
        limit = _as_int(payload.get("limit"), "limit", default=20, minimum=1, maximum=_MAX_LIMIT)
        max_chars = _as_int(payload.get("max_chars"), "max_chars", default=1200, minimum=200, maximum=_MAX_CHARS)
        _, messages = self._load()
        filtered = self._iter_filtered(payload)
        selected = filtered[start : start + limit]
        result: dict[str, Any] = {
            "start": start,
            "limit": limit,
            "total_matching": len(filtered),
            "messages": [_message_record(message, max_chars=max_chars, messages=messages) for message in selected],
        }
        if used_index_alias:
            result["note"] = "messages.index was interpreted as messages.start; use action=message for one exact index"
        return result

    def _message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _, messages = self._load()
        index = _as_int(payload.get("index"), "index", default=-1, minimum=0)
        max_chars = _as_int(payload.get("max_chars"), "max_chars", default=8000, minimum=200, maximum=_MAX_CHARS)
        if index >= len(messages):
            raise ValueError(f"index out of range: {index}")
        return _message_record(messages[index], max_chars=max_chars, messages=messages)

    def _search(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        query = _as_optional_str(payload.get("query"), "query")
        if query is None:
            raise ValueError("search requires query")
        regex = _as_bool(payload.get("regex"), "regex", default=False)
        case_sensitive = _as_bool(payload.get("case_sensitive"), "case_sensitive", default=False)
        limit = _as_int(payload.get("limit"), "limit", default=20, minimum=1, maximum=_MAX_LIMIT)
        max_chars = _as_int(payload.get("max_chars"), "max_chars", default=1200, minimum=200, maximum=_MAX_CHARS)
        _, messages = self._load()
        filtered = self._iter_filtered(payload)
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query if regex else re.escape(query), flags=flags)
        results = []
        total_matches = 0
        context = max(50, max_chars // 2)
        for message in filtered:
            match = pattern.search(message.text)
            if match is None:
                continue
            total_matches += 1
            if len(results) >= limit:
                continue
            start = max(0, match.start() - context)
            end = min(len(message.text), match.end() + context)
            snippet, truncated = _truncate(message.text[start:end], max_chars)
            results.append(
                {
                    "index": message.index,
                    "role": message.role,
                    "msg_type": message.msg_type,
                    "tool_name": message.tool_name,
                    **_tool_call_index_fields(message, messages),
                    "snippet": snippet,
                    "truncated": truncated or start > 0 or end < len(message.text),
                }
            )
        return {"query": query, "total_matches": total_matches, "matches": results}

    def _errors(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        limit = _as_int(payload.get("limit"), "limit", default=30, minimum=1, maximum=_MAX_LIMIT)
        max_chars = _as_int(payload.get("max_chars"), "max_chars", default=2500, minimum=200, maximum=_MAX_CHARS)
        _, messages = self._load()
        errors = []
        total = 0
        for message in messages:
            if not _looks_like_error(message):
                continue
            total += 1
            if len(errors) >= limit:
                continue
            summary = _payload_error_summary(_tool_payload(message.raw)) or message.text
            snippet, truncated = _truncate(summary, max_chars)
            errors.append(
                {
                    "index": message.index,
                    "role": message.role,
                    "msg_type": message.msg_type,
                    "tool_name": message.tool_name,
                    **_tool_call_index_fields(message, messages),
                    "summary": snippet,
                    "truncated": truncated,
                }
            )
        return {"total_errors": total, "errors": errors}
