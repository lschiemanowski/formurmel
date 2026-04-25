from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from formurmel.agent import AgentRunResult, AgentStatus, build_initial_conversation, run
from formurmel.config import AppConfig
from formurmel.llm.base import LLMBackend
from formurmel.logging import Logger
from formurmel.runtime import build_backend, build_logger, build_tool_registry
from formurmel.tools.base import ToolRegistry
from formurmel.tools.kb_tool import KBTool


@dataclass(frozen=True)
class AgentEpisodeResult:
    agent_result: AgentRunResult
    config: AppConfig
    kb_path: Path | None
    transcript_path: Path | None

    @property
    def status(self) -> AgentStatus:
        return self.agent_result.status

    @property
    def error(self) -> str | None:
        return self.agent_result.error

    @property
    def steps(self) -> int:
        return self.agent_result.steps

    @property
    def final_message_text(self) -> str | None:
        final_message = self.agent_result.final_message
        if final_message is None or not isinstance(final_message.content, str):
            return None
        text = final_message.content.strip()
        return text or None


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


class AgentEpisodeRunner:
    """Reusable agent invocation boundary for rollout/training code.

    The runner keeps the backend and non-KB tools alive across episodes. Each
    episode gets a shallow-cloned registry and, when configured, a fresh KB tool
    pointed at that episode's KB path. Use one runner per rollout worker/thread.
    """

    def __init__(
        self,
        template_config: AppConfig,
        *,
        backend: LLMBackend | None = None,
        base_tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.template_config = template_config
        self.backend = backend if backend is not None else build_backend(template_config.backend)
        if base_tool_registry is None:
            base_tools = replace(template_config.tools, kb_path=None)
            base_tool_registry = build_tool_registry(base_tools)
        elif base_tool_registry.get("kb") is not None:
            raise ValueError("base_tool_registry must not include a kb tool; KB tools are per-episode")
        self._base_tool_registry = base_tool_registry

    def run_episode(
        self,
        *,
        user_prompt: str,
        kb_path: str | Path | None = None,
        transcript_path: str | Path | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        logger: Logger | None = None,
        require_kb_completion: bool = True,
    ) -> AgentEpisodeResult:
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        resolved_kb_path = _resolve_path(kb_path) if kb_path is not None else self.template_config.tools.kb_path
        resolved_transcript_path = (
            _resolve_path(transcript_path)
            if transcript_path is not None
            else self.template_config.agent.transcript_path
        )
        episode_tools = replace(self.template_config.tools, kb_path=resolved_kb_path)
        episode_agent = replace(
            self.template_config.agent,
            transcript_path=resolved_transcript_path,
            max_steps=max_steps if max_steps is not None else self.template_config.agent.max_steps,
            system_prompt=system_prompt if system_prompt is not None else self.template_config.agent.system_prompt,
            system_prompt_path=None if system_prompt is not None else self.template_config.agent.system_prompt_path,
        )
        episode_config = AppConfig(
            backend=self.template_config.backend,
            tools=episode_tools,
            agent=episode_agent,
        )

        tool_registry = self._build_episode_tool_registry(episode_config)
        active_logger = logger if logger is not None else build_logger(episode_config.agent)
        conversation = build_initial_conversation(episode_config.agent.system_prompt, user_prompt)
        agent_result = run(
            initial_conversation=conversation,
            tool_registry=tool_registry,
            backend=self.backend,
            transcript_path=episode_config.agent.transcript_path,
            logger=active_logger,
            max_steps=episode_config.agent.max_steps,
            require_kb_completion=require_kb_completion,
        )
        return AgentEpisodeResult(
            agent_result=agent_result,
            config=episode_config,
            kb_path=episode_config.tools.kb_path,
            transcript_path=episode_config.agent.transcript_path,
        )

    def _build_episode_tool_registry(self, episode_config: AppConfig) -> ToolRegistry:
        registry = self._base_tool_registry.clone()
        kb_path = episode_config.tools.kb_path
        if kb_path is not None:
            registry.register(
                KBTool(
                    kb_path=kb_path,
                    lake_project=episode_config.tools.kb_lake_project,
                    autosave=episode_config.tools.kb_autosave,
                )
            )
        return registry

    def close(self) -> None:
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()


def run_episode_from_config(
    config: AppConfig,
    *,
    user_prompt: str,
    kb_path: str | Path | None = None,
    transcript_path: str | Path | None = None,
    max_steps: int | None = None,
    system_prompt: str | None = None,
    logger: Logger | None = None,
    require_kb_completion: bool = True,
) -> AgentEpisodeResult:
    runner = AgentEpisodeRunner(config)
    try:
        return runner.run_episode(
            user_prompt=user_prompt,
            kb_path=kb_path,
            transcript_path=transcript_path,
            max_steps=max_steps,
            system_prompt=system_prompt,
            logger=logger,
            require_kb_completion=require_kb_completion,
        )
    finally:
        runner.close()
