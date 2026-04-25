from __future__ import annotations

from abc import ABC, abstractmethod

from formurmel.message import Conversation, Message, ToolSpec


class LLMBackendError(RuntimeError):
    """Raised when backend calls or response decoding fails."""


class LLMBackend(ABC):
    @abstractmethod
    def query(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> list[Message]:
        """Query the LLM backend and return one or more assistant messages."""
        raise NotImplementedError

