from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(StrEnum):
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    REASONING = "reasoning"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: Any) -> "ToolCall":
        if not isinstance(data, Mapping):
            raise TypeError("tool call payload must be a mapping")
        args = data.get("arguments", {})
        if not isinstance(args, Mapping):
            raise TypeError("tool call arguments must be a mapping")
        return cls(id=str(data["id"]), name=str(data["name"]), arguments=dict(args))


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ToolSpec":
        if not isinstance(data, Mapping):
            raise TypeError("tool spec payload must be a mapping")
        params = data.get("parameters", {})
        if not isinstance(params, Mapping):
            raise TypeError("tool spec parameters must be a mapping")
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            parameters=dict(params),
        )


@dataclass
class ToolResponse:
    id: str
    name: str
    content: str | Mapping[str, Any]
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": dict(self.content) if isinstance(self.content, Mapping) else self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ToolResponse":
        if not isinstance(data, Mapping):
            raise TypeError("tool response payload must be a mapping")
        raw_content = data.get("content", "")
        if not isinstance(raw_content, (str, Mapping)):
            raise TypeError("tool response content must be a string or mapping")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            content=str(raw_content) if isinstance(raw_content, str) else dict(raw_content),
            is_error=bool(data.get("is_error", False)),
        )


MessageContent = str | ToolCall | ToolResponse


@dataclass
class Message:
    role: Role
    content: MessageContent
    msg_type: Optional[MessageType] = None
    provider_state: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.msg_type is None:
            if isinstance(self.content, ToolCall):
                self.msg_type = MessageType.TOOL_CALL
            elif isinstance(self.content, ToolResponse):
                self.msg_type = MessageType.TOOL_RESPONSE

        if isinstance(self.content, ToolCall):
            if self.role != Role.ASSISTANT:
                raise ValueError("tool call messages must use assistant role")
            if self.msg_type != MessageType.TOOL_CALL:
                raise ValueError("tool call messages must use tool_call msg_type")
            return

        if isinstance(self.content, ToolResponse):
            if self.role != Role.TOOL:
                raise ValueError("tool response messages must use tool role")
            if self.msg_type != MessageType.TOOL_RESPONSE:
                raise ValueError("tool response messages must use tool_response msg_type")
            return

        if not isinstance(self.content, str):
            raise TypeError("message content must be a string, ToolCall, or ToolResponse")
        if self.msg_type in {MessageType.TOOL_CALL, MessageType.TOOL_RESPONSE}:
            raise ValueError("tool messages must use ToolCall or ToolResponse content")

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, (ToolCall, ToolResponse)):
            serialized_content: str | dict[str, Any] = self.content.to_dict()
        else:
            serialized_content = self.content

        payload: dict[str, Any] = {
            "role": self.role.value,
            "content": serialized_content,
        }
        if self.msg_type is not None:
            payload["msg_type"] = self.msg_type.value
        if self.provider_state is not None:
            payload["provider_state"] = dict(self.provider_state)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "Message":
        if not isinstance(data, Mapping):
            raise TypeError("message payload must be a mapping")

        role = Role(str(data["role"]))
        msg_type = data.get("msg_type")
        parsed_msg_type = MessageType(str(msg_type)) if msg_type is not None else None
        provider_state_raw = data.get("provider_state")
        if provider_state_raw is not None and not isinstance(provider_state_raw, Mapping):
            raise TypeError("message provider_state must be a mapping when present")

        raw_content = data["content"]
        if isinstance(raw_content, str):
            content: MessageContent = raw_content
        elif isinstance(raw_content, Mapping):
            if (
                parsed_msg_type == MessageType.TOOL_CALL
                or (role == Role.ASSISTANT and {"id", "name", "arguments"} <= set(raw_content.keys()))
            ):
                content = ToolCall.from_dict(raw_content)
            elif (
                parsed_msg_type == MessageType.TOOL_RESPONSE
                or role == Role.TOOL
                or {"id", "name", "content"} <= set(raw_content.keys())
            ):
                content = ToolResponse.from_dict(raw_content)
            else:
                raise TypeError("structured content must be a tool call or tool response payload")
        else:
            raise TypeError("message content must be a string or mapping")

        return cls(
            role=role,
            content=content,
            msg_type=parsed_msg_type,
            provider_state=dict(provider_state_raw) if isinstance(provider_state_raw, Mapping) else None,
        )

    def pretty_str(self) -> str:
        header = self.role.value.upper()
        if self.msg_type is not None:
            header = f"{header} [{self.msg_type.value}]"

        if isinstance(self.content, str):
            body = self.content.strip()
            return f"{header}: {body}" if body else header

        if isinstance(self.content, ToolCall):
            args_str = json.dumps(self.content.arguments, indent=2, sort_keys=True)
            return f"{header}: {self.content.name} (id={self.content.id})\narguments:\n{args_str}"

        result = self.content.content
        label = f"{header}: {self.content.name} (id={self.content.id})"
        if self.content.is_error:
            label = f"{label} [error]"
        result_str = result.strip() if isinstance(result, str) else json.dumps(result, indent=2, sort_keys=True)
        return f"{label}\n{result_str}" if result_str else label


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        self.messages.extend(messages)

    def to_dict(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    @classmethod
    def from_dict(cls, data: Any) -> "Conversation":
        if not isinstance(data, list):
            raise TypeError("conversation payload must be a list")
        return cls(messages=[Message.from_dict(item) for item in data])

