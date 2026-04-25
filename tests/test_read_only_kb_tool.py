from __future__ import annotations

from typing import Any, Mapping

from formurmel.message import ToolSpec
from formurmel.tools.base import Tool
from formurmel.tools.read_only_kb_tool import ReadOnlyKBTool


class FakeKBTool(Tool):
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    @property
    def name(self) -> str:
        return "kb"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name="kb",
            description="fake kb",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get_node", "update_node", "compile_node", "compile_candidate"],
                    }
                },
            },
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {"ok": True, "result": {"payload": dict(payload)}}

    def completion_state(self) -> dict[str, Any]:
        return {"pending_statement_ids": ["target"]}


def test_read_only_kb_tool_delegates_allowed_actions() -> None:
    wrapped = FakeKBTool()
    tool = ReadOnlyKBTool(wrapped)

    result = tool.execute({"action": "get_node", "id": "target"})

    assert result["ok"] is True
    assert wrapped.calls == [{"action": "get_node", "id": "target"}]


def test_read_only_kb_tool_blocks_mutating_actions() -> None:
    wrapped = FakeKBTool()
    tool = ReadOnlyKBTool(wrapped)

    result = tool.execute({"action": "update_node", "id": "target", "lean_proof": "by trivial"})

    assert result["ok"] is False
    assert "read-only during diagnosis" in result["error"]
    assert wrapped.calls == []


def test_read_only_kb_tool_removes_mutating_actions_from_schema() -> None:
    tool = ReadOnlyKBTool(FakeKBTool())

    actions = tool.tool_description().parameters["properties"]["action"]["enum"]

    assert "get_node" in actions
    assert "compile_candidate" in actions
    assert "update_node" not in actions
    assert "compile_node" not in actions


def test_read_only_kb_tool_delegates_completion_state() -> None:
    tool = ReadOnlyKBTool(FakeKBTool())

    assert tool.completion_state() == {"pending_statement_ids": ["target"]}
