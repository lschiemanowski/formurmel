from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from formurmel.llm.base import LLMBackend, LLMBackendError
from formurmel.logging import Logger, NullLogger
from formurmel.message import Conversation, Message, MessageType, Role, ToolCall, ToolResponse
from formurmel.tools.base import ToolError, ToolRegistry, tool_error


class AgentStatus(StrEnum):
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentRunResult:
    status: AgentStatus
    conversation: Conversation
    final_message: Message | None = None
    error: str | None = None
    steps: int = 0


def build_initial_conversation(system_prompt: str, user_prompt: str) -> Conversation:
    messages: list[Message] = []
    if system_prompt.strip():
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))
    messages.append(Message(role=Role.USER, content=user_prompt))
    return Conversation(messages=messages)


def save_transcript(transcript_path: Path, result: AgentRunResult, logger: Logger) -> None:
    payload = {
        "status": result.status.value,
        "steps": result.steps,
        "error": result.error,
        "final_message": result.final_message.to_dict() if result.final_message is not None else None,
        "conversation": result.conversation.to_dict(),
    }
    try:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.log(f"failed to save transcript to {transcript_path}: {exc}", verbosity_level=0)


def _make_tool_response_message(tool_call: ToolCall, payload: Any) -> Message:
    if isinstance(payload, Mapping):
        content: str | Mapping[str, Any] = dict(payload)
        ok_value = payload.get("ok")
        is_error = isinstance(ok_value, bool) and not ok_value
    elif isinstance(payload, str):
        content = payload
        is_error = False
    else:
        content = {"raw_result": payload}
        is_error = False

    return Message(
        role=Role.TOOL,
        content=ToolResponse(
            id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=is_error,
        ),
    )


def _execute_tool_call(tool_registry: ToolRegistry, tool_call: ToolCall) -> Message:
    tool = tool_registry.get(tool_call.name)
    if tool is None:
        return _make_tool_response_message(
            tool_call,
            tool_error(f"unknown tool '{tool_call.name}'"),
        )

    try:
        payload = tool.execute(dict(tool_call.arguments))
    except (ToolError, ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        payload = tool_error(f"tool error: {exc}")
    except Exception as exc:  # noqa: BLE001
        payload = tool_error(f"tool crashed: {exc}")

    return _make_tool_response_message(tool_call, payload)


def _kb_completion_state(tool_registry: ToolRegistry) -> Mapping[str, Any] | None:
    kb_tool = tool_registry.get("kb")
    if kb_tool is None:
        return None
    completion_fn = getattr(kb_tool, "completion_state", None)
    if not callable(completion_fn):
        return None
    try:
        state = completion_fn()
    except Exception:  # noqa: BLE001
        return None
    return state if isinstance(state, Mapping) else None


def run(
    initial_conversation: Conversation,
    tool_registry: ToolRegistry,
    backend: LLMBackend,
    *,
    transcript_path: Path | None = None,
    logger: Logger | None = None,
    max_steps: int = 300,
    require_kb_completion: bool = True,
) -> AgentRunResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    active_logger = logger if logger is not None else NullLogger()
    conversation = Conversation(messages=list(initial_conversation.messages))
    tool_specs = tool_registry.specs()
    steps_taken = 0
    result: AgentRunResult | None = None

    for step in range(1, max_steps + 1):
        steps_taken = step
        try:
            assistant_messages = backend.query(conversation, tool_specs=tool_specs)
        except LLMBackendError as exc:
            result = AgentRunResult(
                status=AgentStatus.ERROR,
                conversation=conversation,
                error=f"backend query failed: {exc}",
                steps=steps_taken,
            )
            break

        active_logger.log(f"[step={step}] replies={len(assistant_messages)}", verbosity_level=0)
        if not assistant_messages:
            result = AgentRunResult(
                status=AgentStatus.ERROR,
                conversation=conversation,
                error="backend returned no messages",
                steps=steps_taken,
            )
            break

        saw_tool_call = False
        saw_user_facing_reply = False
        saw_reasoning = False
        final_message: Message | None = None
        tool_messages: list[Message] = []

        for message in assistant_messages:
            conversation.append(message)
            active_logger.log(message.pretty_str(), verbosity_level=1)

            if message.msg_type == MessageType.REASONING:
                saw_reasoning = True
                continue

            if message.msg_type == MessageType.TOOL_CALL:
                if not isinstance(message.content, ToolCall):
                    result = AgentRunResult(
                        status=AgentStatus.ERROR,
                        conversation=conversation,
                        error="invalid TOOL_CALL message: content is not a ToolCall",
                        steps=steps_taken,
                    )
                    break

                saw_tool_call = True
                tool_messages.append(_execute_tool_call(tool_registry, message.content))
                continue

            if message.role == Role.ASSISTANT and message.msg_type is None and isinstance(message.content, str):
                if message.content.strip():
                    final_message = message
                    saw_user_facing_reply = True

        if result is not None:
            break

        for tool_message in tool_messages:
            conversation.append(tool_message)
            active_logger.log(tool_message.pretty_str(), verbosity_level=1)

        if saw_tool_call:
            continue

        if saw_user_facing_reply:
            kb_state = _kb_completion_state(tool_registry) if require_kb_completion else None
            if kb_state is not None:
                pending = kb_state.get("pending_statement_ids")
                if isinstance(pending, list) and pending:
                    result = AgentRunResult(
                        status=AgentStatus.ERROR,
                        conversation=conversation,
                        final_message=final_message,
                        error=(
                            "assistant ended the run before the KB task was completed; "
                            f"pending statement(s): {', '.join(str(item) for item in pending)}"
                        ),
                        steps=steps_taken,
                    )
                    break
            result = AgentRunResult(
                status=AgentStatus.DONE,
                conversation=conversation,
                final_message=final_message,
                steps=steps_taken,
            )
            break

        if saw_reasoning:
            active_logger.log(
                f"[step={step}] assistant returned reasoning only; continuing",
                verbosity_level=0,
            )
            continue

        result = AgentRunResult(
            status=AgentStatus.ERROR,
            conversation=conversation,
            error="assistant returned no tool call and no user-facing response",
            steps=steps_taken,
        )
        break

    if result is None:
        result = AgentRunResult(
            status=AgentStatus.ERROR,
            conversation=conversation,
            error=f"agent exceeded max_steps={max_steps}",
            steps=steps_taken,
        )

    if transcript_path is not None:
        save_transcript(transcript_path, result, active_logger)
    return result
