from __future__ import annotations

from formurmel.agent import AgentStatus, build_initial_conversation, run
from formurmel.llm.base import LLMBackend
from formurmel.message import Conversation, Message, Role, ToolSpec
from formurmel.tools.base import Tool, ToolRegistry


class FakeBackend(LLMBackend):
    def query(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> list[Message]:
        return [Message(role=Role.ASSISTANT, content="diagnosis")]


class FakeKBTool(Tool):
    @property
    def name(self) -> str:
        return "kb"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(name="kb", description="fake", parameters={"type": "object"})

    def execute(self, payload):
        return {"ok": True}

    def completion_state(self):
        return {"pending_statement_ids": ["target"]}


def test_run_can_disable_kb_completion_check_for_diagnosis() -> None:
    registry = ToolRegistry()
    registry.register(FakeKBTool())
    result = run(
        build_initial_conversation("", "diagnose"),
        registry,
        FakeBackend(),
        require_kb_completion=False,
    )
    assert result.status == AgentStatus.DONE
    assert result.error is None


def test_run_keeps_kb_completion_check_enabled_by_default() -> None:
    registry = ToolRegistry()
    registry.register(FakeKBTool())
    result = run(build_initial_conversation("", "formalize"), registry, FakeBackend())
    assert result.status == AgentStatus.ERROR
    assert "pending statement" in (result.error or "")
