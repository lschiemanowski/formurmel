from __future__ import annotations

import json
from typing import Any, Mapping

from formurmel.message import Conversation, Message, MessageType, Role, ToolCall, ToolResponse, ToolSpec


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def tool_to_openai(tool_spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_spec.name,
            "description": tool_spec.description,
            "parameters": dict(tool_spec.parameters),
        },
    }


def parse_tool_call_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return {}
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            return {"_raw": trimmed}
        if isinstance(parsed, Mapping):
            return dict(parsed)
        return {"_raw": str(parsed)}
    return {"_raw": as_text(raw)}


def conversation_to_chat_messages(
    conversation: Conversation,
    *,
    reasoning_field: str | None = None,
    require_reasoning_for_tool_calls: bool = False,
    provider_state_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    provider_fields: dict[str, Any] = {}

    def flush_assistant_run() -> None:
        nonlocal reasoning_parts, content_parts, tool_calls, provider_fields
        if not reasoning_parts and not content_parts and not tool_calls:
            provider_fields = {}
            return
        content = "\n\n".join(part for part in content_parts if part).strip()
        reasoning = "\n\n".join(part for part in reasoning_parts if part).strip()
        payload: dict[str, Any] = {"role": "assistant"}
        for field_name, field_value in provider_fields.items():
            payload[field_name] = field_value
        if reasoning_field is not None and (reasoning or (tool_calls and require_reasoning_for_tool_calls)):
            payload[reasoning_field] = reasoning if reasoning else ""
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id or f"call_{index + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": as_text(tool_call.arguments),
                    },
                }
                for index, tool_call in enumerate(tool_calls)
            ]
            payload["content"] = content if content else None
            messages.append(payload)
        elif content or (reasoning_field is not None and reasoning):
            payload["content"] = content
            messages.append(payload)
        reasoning_parts = []
        content_parts = []
        tool_calls = []
        provider_fields = {}

    for message in conversation.messages:
        if message.role == Role.ASSISTANT:
            if message.provider_state is not None:
                for field_name in provider_state_fields:
                    if field_name not in provider_fields and field_name in message.provider_state:
                        provider_fields[field_name] = message.provider_state[field_name]
            if message.msg_type == MessageType.REASONING:
                if isinstance(message.content, str) and message.content.strip():
                    reasoning_parts.append(message.content.strip())
                continue
            if message.msg_type == MessageType.TOOL_CALL:
                if isinstance(message.content, ToolCall):
                    tool_calls.append(message.content)
                continue
            if isinstance(message.content, str):
                if message.content.strip():
                    content_parts.append(message.content.strip())
                continue

        flush_assistant_run()

        if message.role == Role.SYSTEM:
            text = as_text(message.content)
            if text:
                messages.append({"role": "system", "content": text})
            continue
        if message.role == Role.USER:
            messages.append({"role": "user", "content": as_text(message.content)})
            continue
        if message.role == Role.TOOL:
            if not isinstance(message.content, ToolResponse):
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.content.id or "call_unknown",
                    "content": as_text(message.content.content),
                }
            )
            continue
        messages.append({"role": message.role.value, "content": as_text(message.content)})

    flush_assistant_run()
    return messages


def assistant_messages_from_chat_payload(
    assistant_payload: Mapping[str, Any],
    *,
    reasoning_fields: tuple[str, ...] = ("reasoning_content", "reasoning"),
) -> list[Message]:
    messages: list[Message] = []
    provider_state = _provider_state_from_chat_payload(assistant_payload)
    provider_state_used = False

    def consume_provider_state() -> Mapping[str, Any] | None:
        nonlocal provider_state_used
        if provider_state is None or provider_state_used:
            return None
        provider_state_used = True
        return provider_state

    reasoning_text = ""
    for field_name in reasoning_fields:
        reasoning = assistant_payload.get(field_name)
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning_text = reasoning.strip()
            break
    if not reasoning_text:
        reasoning_text = reasoning_text_from_details(assistant_payload.get("reasoning_details"))
    if reasoning_text:
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=reasoning_text,
                msg_type=MessageType.REASONING,
                provider_state=consume_provider_state(),
            )
        )

    content = assistant_payload.get("content")
    content_text = as_text(content).strip() if content is not None else ""
    raw_tool_calls = assistant_payload.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise TypeError("assistant tool_calls must be a list")

    parsed_calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls, start=1):
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = raw_call.get("name")
            arguments = raw_call.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        parsed_calls.append(ToolCall(id=call_id, name=name, arguments=parse_tool_call_arguments(arguments)))

    if content_text:
        messages.append(Message(role=Role.ASSISTANT, content=content_text, provider_state=consume_provider_state()))
    for tool_call in parsed_calls:
        messages.append(Message(role=Role.ASSISTANT, content=tool_call, provider_state=consume_provider_state()))
    if not messages:
        raise ValueError("assistant payload did not contain reasoning, text, or tool calls")
    return messages


def reasoning_text_from_details(raw_details: Any) -> str:
    if not isinstance(raw_details, list):
        return ""
    parts: list[str] = []
    for detail in raw_details:
        if not isinstance(detail, Mapping):
            continue
        for field_name in ("text", "summary"):
            value = detail.get(field_name)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    return "\n\n".join(parts)


def _provider_state_from_chat_payload(assistant_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw_details = assistant_payload.get("reasoning_details")
    if not isinstance(raw_details, list):
        return None
    return {"reasoning_details": list(raw_details)}
