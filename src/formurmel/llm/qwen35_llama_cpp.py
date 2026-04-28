from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping

import requests

from formurmel.llm.base import LLMBackend, LLMBackendError
from formurmel.llm.shared import assistant_messages_from_chat_payload, conversation_to_chat_messages, tool_to_openai
from formurmel.message import Conversation, Message, ToolCall, ToolSpec


_NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^\n>]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(r"<parameter=([^\n>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _prompt_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tool_argument_to_prompt(value: Any) -> str:
    if isinstance(value, Mapping):
        return _json_dumps(dict(value))
    if isinstance(value, (list, tuple)):
        return _json_dumps(list(value))
    if value is None or isinstance(value, (bool, int, float)):
        return _json_dumps(value)
    return str(value)


def _parse_tool_argument_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if (
        text[:1] in {"{", "["}
        or text in {"true", "false", "null"}
        or _NUMBER_RE.match(text)
        or (text.startswith('"') and text.endswith('"'))
    ):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                nested = parsed.strip()
                if nested and (
                    nested[:1] in {"{", "["}
                    or nested in {"true", "false", "null"}
                    or _NUMBER_RE.match(nested)
                    or (nested.startswith('"') and nested.endswith('"'))
                ):
                    try:
                        return json.loads(nested)
                    except json.JSONDecodeError:
                        return parsed
            return parsed
        except json.JSONDecodeError:
            pass
    return text


class Qwen35LlamaCppCompletionBackend(LLMBackend):
    """Qwen 3.5 prompt renderer for llama.cpp /completion."""

    def __init__(
        self,
        *,
        base_url: str,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300.0,
        retries: int = 3,
        retry_cooldown_seconds: float = 1.0,
        parse_retries: int = 1,
        enable_thinking: bool = True,
        debug: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_cooldown_seconds = max(0.0, float(retry_cooldown_seconds))
        self.parse_retries = max(0, int(parse_retries))
        self.enable_thinking = enable_thinking
        self.debug = debug

    def query(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> list[Message]:
        payload = self._build_prompt(conversation, tool_specs)
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                raw = requests.post(f"{self.base_url}/completion", json=payload, timeout=self.timeout)
                raw.raise_for_status()
                response = raw.json()
                completion = self._extract_completion(response)
                if completion is None:
                    raise LLMBackendError(f"backend returned no completion: {response!r}")
                return self._parse_completion(completion)
            except (requests.RequestException, ValueError, LLMBackendError) as exc:
                last_error = exc
                if self.debug:
                    print(f"Qwen 3.5 llama.cpp backend error: {exc}")
                if attempt < self.retries - 1:
                    time.sleep(self.retry_cooldown_seconds)

        raise LLMBackendError(f"qwen35 llama.cpp backend failed after {self.retries} attempts: {last_error}")

    def _build_prompt(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> dict[str, object]:
        prompt = self._render_prompt(
            conversation_to_chat_messages(conversation, reasoning_field="reasoning_content"),
            [tool_to_openai(spec) for spec in tool_specs],
        )
        payload: dict[str, object] = {
            "prompt": prompt,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.presence_penalty is not None:
            payload["presence_penalty"] = float(self.presence_penalty)
        if self.max_tokens is not None:
            payload["n_predict"] = int(self.max_tokens)
        return payload

    def _render_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        if not messages:
            raise LLMBackendError("No messages provided")

        first_message = messages[0]
        parts: list[str] = []
        if tools:
            parts.append("<|im_start|>system\n")
            parts.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
            for tool in tools:
                parts.append("\n")
                parts.append(_json_dumps(tool))
            parts.append("\n</tools>")
            parts.append(_TOOL_INSTRUCTIONS)
            if first_message.get("role") == "system":
                system_content = _prompt_text(first_message.get("content", ""))
                if system_content:
                    parts.append("\n\n")
                    parts.append(system_content)
            parts.append("<|im_end|>\n")
        elif first_message.get("role") == "system":
            system_content = _prompt_text(first_message.get("content", ""))
            parts.append(f"<|im_start|>system\n{system_content}<|im_end|>\n")

        last_user_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        if last_user_index < 0:
            raise LLMBackendError("No user query found in messages")

        index = 0
        while index < len(messages):
            message = messages[index]
            role = message.get("role")
            content = _prompt_text(message.get("content", ""))

            if role == "system":
                if index != 0:
                    raise LLMBackendError("System message must be at the beginning")
                index += 1
                continue
            if role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
                index += 1
                continue
            if role == "assistant":
                reasoning = _prompt_text(message.get("reasoning_content", ""))
                body = content
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    for tool_index, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, Mapping):
                            continue
                        function = tool_call.get("function")
                        if isinstance(function, Mapping):
                            tool_name = function.get("name")
                            raw_arguments = function.get("arguments", {})
                        else:
                            tool_name = tool_call.get("name")
                            raw_arguments = tool_call.get("arguments", {})
                        if not isinstance(tool_name, str) or not tool_name:
                            continue
                        block_parts = ["<tool_call>", f"<function={tool_name}>"]
                        if isinstance(raw_arguments, Mapping):
                            argument_items = raw_arguments.items()
                        else:
                            try:
                                parsed_arguments = json.loads(str(raw_arguments))
                            except json.JSONDecodeError:
                                parsed_arguments = {}
                            argument_items = parsed_arguments.items() if isinstance(parsed_arguments, Mapping) else []
                        for arg_name, arg_value in argument_items:
                            block_parts.extend(
                                [
                                    f"<parameter={arg_name}>",
                                    _tool_argument_to_prompt(arg_value),
                                    "</parameter>",
                                ]
                            )
                        block_parts.extend(["</function>", "</tool_call>"])
                        block = "\n".join(block_parts)
                        body = f"{body}\n\n{block}" if body and tool_index == 0 else f"{body}\n{block}" if body else block
                parts.append("<|im_start|>assistant\n")
                if index > last_user_index:
                    parts.append("<think>\n")
                    if reasoning:
                        parts.append(reasoning)
                        parts.append("\n")
                    parts.append("</think>\n\n")
                parts.append(body)
                parts.append("<|im_end|>\n")
                index += 1
                continue
            if role == "tool":
                parts.append("<|im_start|>user")
                while index < len(messages) and messages[index].get("role") == "tool":
                    tool_content = _prompt_text(messages[index].get("content", ""))
                    parts.append(f"\n<tool_response>\n{tool_content}\n</tool_response>")
                    index += 1
                parts.append("<|im_end|>\n")
                continue
            raise LLMBackendError(f"Unexpected message role: {role}")

        parts.append("<|im_start|>assistant\n")
        if self.enable_thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    @staticmethod
    def _extract_completion(response: Mapping[str, Any]) -> str | None:
        if isinstance(response, str):
            return response
        if not isinstance(response, Mapping):
            return None
        content = response.get("content")
        if isinstance(content, str):
            return content
        completion = response.get("completion")
        if isinstance(completion, str):
            return completion
        return None

    def _parse_completion(self, completion: str) -> list[Message]:
        adjusted = completion.strip()
        if not adjusted:
            raise LLMBackendError("empty completion payload")
        attempts = 0
        while True:
            try:
                return self._parse_completion_once(adjusted)
            except Exception as exc:
                if attempts >= self.parse_retries:
                    return [Message.from_dict({"role": "assistant", "content": adjusted})]
                attempts += 1
                if self.debug:
                    print(f"failed to parse qwen35 completion ({attempts}/{self.parse_retries}): {exc}")
                adjusted = adjusted.replace("<|im_end|>", "").strip()

    def _parse_completion_once(self, completion: str) -> list[Message]:
        adjusted = completion.strip()
        if adjusted.startswith("<|im_start|>assistant"):
            adjusted = adjusted[len("<|im_start|>assistant") :].lstrip()
        if adjusted.endswith("<|im_end|>"):
            adjusted = adjusted[: -len("<|im_end|>")].rstrip()

        reasoning = ""
        if adjusted.startswith("<think>"):
            adjusted = adjusted[len("<think>") :].lstrip()
        if "</think>" in adjusted:
            reasoning, adjusted = adjusted.split("</think>", 1)
            reasoning = reasoning.strip()
            adjusted = adjusted.lstrip()

        tool_calls: list[ToolCall] = []
        residual_parts: list[str] = []
        cursor = 0
        for call_index, match in enumerate(_TOOL_CALL_RE.finditer(adjusted), start=1):
            prefix = adjusted[cursor : match.start()].strip()
            if prefix:
                residual_parts.append(prefix)
            function_name = match.group(1).strip()
            function_body = match.group(2)
            arguments: dict[str, Any] = {}
            for param_match in _PARAMETER_RE.finditer(function_body):
                arguments[param_match.group(1).strip()] = _parse_tool_argument_value(param_match.group(2))
            if function_name:
                tool_calls.append(ToolCall(id=f"call_{call_index}", name=function_name, arguments=arguments))
            cursor = match.end()

        suffix = adjusted[cursor:].strip()
        if suffix:
            residual_parts.append(suffix)

        assistant_payload: dict[str, Any] = {}
        if reasoning:
            assistant_payload["reasoning_content"] = reasoning
        residual_text = "\n\n".join(part for part in residual_parts if part).strip()
        if residual_text:
            assistant_payload["content"] = residual_text
        if tool_calls:
            assistant_payload["tool_calls"] = [
                {"id": tool_call.id, "name": tool_call.name, "arguments": dict(tool_call.arguments)}
                for tool_call in tool_calls
            ]

        try:
            return assistant_messages_from_chat_payload(assistant_payload)
        except (TypeError, ValueError) as exc:
            raise LLMBackendError(f"completion did not contain final text or tool calls: {exc}") from exc
