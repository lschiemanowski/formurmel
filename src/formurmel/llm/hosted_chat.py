from __future__ import annotations

import os
import time
from typing import Any, Mapping

import requests

from formurmel.llm.base import LLMBackend, LLMBackendError
from formurmel.llm.shared import assistant_messages_from_chat_payload, conversation_to_chat_messages, tool_to_openai
from formurmel.message import Conversation, Message, ToolSpec


class _ChatCompletionsBackend(LLMBackend):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300.0,
        retries: int = 3,
        retry_cooldown_seconds: float = 1.0,
        debug: bool = False,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if not api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty string")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_cooldown_seconds = max(0.0, float(retry_cooldown_seconds))
        self.debug = debug

    @property
    def _debug_name(self) -> str:
        return self.__class__.__name__

    def query(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> list[Message]:
        payload = self._build_payload(conversation, tool_specs)
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                raw = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                raw.raise_for_status()
                response = raw.json()
                message = self._extract_message(response)
                return self._parse_assistant_message(message)
            except (requests.RequestException, ValueError, LLMBackendError, TypeError) as exc:
                last_error = exc
                if self.debug:
                    print(f"{self._debug_name} backend error: {exc}")
                if attempt < self.retries - 1:
                    time.sleep(self.retry_cooldown_seconds)

        raise LLMBackendError(f"{self._debug_name} failed after {self.retries} attempts: {last_error}")

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMBackendError(f"missing API key environment variable: {self.api_key_env}")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._conversation_messages(conversation),
            "stream": False,
        }
        if tool_specs:
            payload["tools"] = [tool_to_openai(spec) for spec in tool_specs]
        if self._include_sampling_parameters:
            if self.temperature is not None:
                payload["temperature"] = float(self.temperature)
            if self.top_p is not None:
                payload["top_p"] = float(self.top_p)
            if self.presence_penalty is not None:
                payload["presence_penalty"] = float(self.presence_penalty)
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        self._add_backend_payload_fields(payload)
        return payload

    @property
    def _include_sampling_parameters(self) -> bool:
        return True

    def _conversation_messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        return conversation_to_chat_messages(conversation)

    def _add_backend_payload_fields(self, payload: dict[str, Any]) -> None:
        return None

    def _parse_assistant_message(self, assistant_payload: Mapping[str, Any]) -> list[Message]:
        return assistant_messages_from_chat_payload(assistant_payload)

    @staticmethod
    def _extract_message(response: Any) -> Mapping[str, Any]:
        if not isinstance(response, Mapping):
            raise LLMBackendError(f"backend returned non-object response: {response!r}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMBackendError(f"backend response did not contain choices: {response!r}")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMBackendError(f"backend choice was not an object: {first_choice!r}")
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMBackendError(f"backend choice did not contain a message: {first_choice!r}")
        return message


class DeepSeekChatCompletionBackend(_ChatCompletionsBackend):
    """DeepSeek OpenAI-compatible Chat Completions backend."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-pro",
        api_key_env: str = "DEEPSEEK_API_KEY",
        reasoning_enabled: bool = True,
        reasoning_effort: str | None = "high",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key_env=api_key_env, **kwargs)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort

    @property
    def _include_sampling_parameters(self) -> bool:
        return not self.reasoning_enabled

    def _conversation_messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        return conversation_to_chat_messages(
            conversation,
            reasoning_field="reasoning_content",
            require_reasoning_for_tool_calls=True,
        )

    def _add_backend_payload_fields(self, payload: dict[str, Any]) -> None:
        payload["thinking"] = {"type": "enabled" if self.reasoning_enabled else "disabled"}
        if self.reasoning_enabled and self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

    def _parse_assistant_message(self, assistant_payload: Mapping[str, Any]) -> list[Message]:
        return assistant_messages_from_chat_payload(
            assistant_payload,
            reasoning_fields=("reasoning_content", "reasoning"),
        )


class OpenRouterChatCompletionBackend(_ChatCompletionsBackend):
    """OpenRouter OpenAI-compatible Chat Completions backend."""

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning_max_tokens: int | None = None,
        reasoning_exclude: bool = False,
        site_url: str | None = None,
        app_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key_env=api_key_env, **kwargs)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_exclude = reasoning_exclude
        self.site_url = site_url
        self.app_name = app_name

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _conversation_messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        return conversation_to_chat_messages(
            conversation,
            reasoning_field="reasoning",
            provider_state_fields=("reasoning_details",),
        )

    def _add_backend_payload_fields(self, payload: dict[str, Any]) -> None:
        reasoning: dict[str, Any] = {}
        if self.reasoning_enabled is not None:
            reasoning["enabled"] = self.reasoning_enabled
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_max_tokens is not None:
            reasoning["max_tokens"] = int(self.reasoning_max_tokens)
        if self.reasoning_exclude:
            reasoning["exclude"] = True
        elif reasoning:
            reasoning["exclude"] = False
        if reasoning:
            payload["reasoning"] = reasoning

    def _parse_assistant_message(self, assistant_payload: Mapping[str, Any]) -> list[Message]:
        return assistant_messages_from_chat_payload(
            assistant_payload,
            reasoning_fields=("reasoning", "reasoning_content"),
        )
