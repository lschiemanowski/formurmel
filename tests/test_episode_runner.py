from __future__ import annotations

from pathlib import Path

import pytest

from formurmel.agent import AgentStatus
from formurmel.config import AgentConfig, AppConfig, BackendConfig, ToolsConfig
from formurmel.episode import AgentEpisodeRunner
from formurmel.llm.base import LLMBackend
from formurmel.message import Conversation, Message, Role, ToolSpec
from formurmel.tools.base import Tool, ToolRegistry, tool_ok
from formurmel.tools.kb_tool import KBTool


class FinalAnswerBackend(LLMBackend):
    def __init__(self) -> None:
        self.calls = 0
        self.tool_names_by_call: list[list[str]] = []

    def query(self, conversation: Conversation, tool_specs: list[ToolSpec]) -> list[Message]:
        self.calls += 1
        self.tool_names_by_call.append([spec.name for spec in tool_specs])
        return [Message(role=Role.ASSISTANT, content=f"final {self.calls}")]


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="A dummy reusable tool.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, payload):
        return tool_ok({"dummy": True})


def _template_config() -> AppConfig:
    return AppConfig(
        backend=BackendConfig(),
        tools=ToolsConfig(kb_autosave=True),
        agent=AgentConfig(system_prompt="system", max_steps=3),
    )


def test_episode_runner_reuses_backend_and_adds_episode_kb_tool(tmp_path: Path) -> None:
    backend = FinalAnswerBackend()
    base_registry = ToolRegistry()
    base_registry.register(DummyTool())
    runner = AgentEpisodeRunner(
        _template_config(),
        backend=backend,
        base_tool_registry=base_registry,
    )

    first = runner.run_episode(
        user_prompt="prove this",
        kb_path=tmp_path / "one.kb.json",
        transcript_path=tmp_path / "one.transcript.json",
        require_kb_completion=False,
    )
    second = runner.run_episode(
        user_prompt="prove that",
        kb_path=tmp_path / "two.kb.json",
        transcript_path=tmp_path / "two.transcript.json",
        require_kb_completion=False,
    )

    assert backend.calls == 2
    assert backend.tool_names_by_call == [["dummy", "kb"], ["dummy", "kb"]]
    assert first.status == AgentStatus.DONE
    assert second.final_message_text == "final 2"
    assert first.kb_path == (tmp_path / "one.kb.json").resolve()
    assert second.kb_path == (tmp_path / "two.kb.json").resolve()
    assert (tmp_path / "one.transcript.json").exists()
    assert (tmp_path / "two.transcript.json").exists()


def test_episode_runner_can_run_without_kb_tool(tmp_path: Path) -> None:
    backend = FinalAnswerBackend()
    runner = AgentEpisodeRunner(
        _template_config(),
        backend=backend,
        base_tool_registry=ToolRegistry(),
    )

    result = runner.run_episode(
        user_prompt="diagnose this",
        transcript_path=tmp_path / "transcript.json",
        require_kb_completion=False,
    )

    assert result.status == AgentStatus.DONE
    assert backend.tool_names_by_call == [[]]
    assert result.kb_path is None


def test_episode_runner_rejects_base_registry_with_kb_tool(tmp_path: Path) -> None:
    base_registry = ToolRegistry()
    base_registry.register(KBTool(kb_path=tmp_path / "base.kb.json"))

    with pytest.raises(ValueError, match="must not include a kb tool"):
        AgentEpisodeRunner(
            _template_config(),
            backend=FinalAnswerBackend(),
            base_tool_registry=base_registry,
        )
