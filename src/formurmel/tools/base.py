from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from formurmel.message import ToolSpec


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    payload: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        value = dict(self.payload)
        value.setdefault("ok", self.ok)
        return value


class ToolError(RuntimeError):
    """Raised when a tool invocation fails for expected reasons."""


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tool_description(self) -> ToolSpec:
        raise NotImplementedError

    @abstractmethod
    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def tool_ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def tool_error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def clone(self) -> "ToolRegistry":
        registry = ToolRegistry()
        registry._tools = dict(self._tools)
        return registry

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def require(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered") from exc

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def items(self):
        return self._tools.items()

    def specs(self) -> list[ToolSpec]:
        return [tool.tool_description() for tool in self._tools.values()]
