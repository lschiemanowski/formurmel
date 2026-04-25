from __future__ import annotations

from formurmel.agent import AgentRunResult, build_initial_conversation, run
from formurmel.config import AgentConfig, AppConfig, BackendConfig, ToolsConfig
from formurmel.llm.base import LLMBackend
from formurmel.llm.qwen35_llama_cpp import Qwen35LlamaCppCompletionBackend
from formurmel.logging import ConsoleLogger, Logger, NullLogger
from formurmel.tools.base import ToolRegistry
from formurmel.tools.registry import create_tool_registry


def build_backend(config: BackendConfig) -> LLMBackend:
    if config.type == "qwen35_llama_cpp":
        return Qwen35LlamaCppCompletionBackend(
            base_url=config.llama_base_url,
            temperature=config.temperature,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            retries=config.retries,
            retry_cooldown_seconds=config.retry_cooldown_seconds,
            parse_retries=config.parse_retries,
            enable_thinking=config.qwen35_enable_thinking,
            debug=config.debug,
        )
    raise ValueError(f"unsupported backend type: {config.type}")


def build_tool_registry(config: ToolsConfig) -> ToolRegistry:
    return create_tool_registry(
        kb_path=config.kb_path,
        kb_lake_project=config.kb_lake_project,
        kb_autosave=config.kb_autosave,
        murmel_cache_dir=config.murmel_cache_dir,
        murmel_config_path=config.murmel_config_path,
        murmel_mathlib_rev=config.murmel_mathlib_rev,
        murmel_semantic_device=config.murmel_semantic_device,
        murmel_semantic_score_chunk_size=config.murmel_semantic_score_chunk_size,
        lean_lake_project=config.lean_lake_project,
        lean_default_timeout_seconds=config.lean_default_timeout_seconds,
    )


def build_logger(config: AgentConfig) -> Logger:
    if config.verbose:
        return ConsoleLogger(verbosity=1)
    return NullLogger()


def build_runtime(config: AppConfig) -> tuple[LLMBackend, ToolRegistry]:
    return build_backend(config.backend), build_tool_registry(config.tools)


def run_from_config(
    config: AppConfig,
    user_prompt: str,
    *,
    logger: Logger | None = None,
    require_kb_completion: bool = True,
) -> AgentRunResult:
    backend, tool_registry = build_runtime(config)
    active_logger = logger if logger is not None else build_logger(config.agent)
    conversation = build_initial_conversation(config.agent.system_prompt, user_prompt)
    return run(
        initial_conversation=conversation,
        tool_registry=tool_registry,
        backend=backend,
        transcript_path=config.agent.transcript_path,
        logger=active_logger,
        max_steps=config.agent.max_steps,
        require_kb_completion=require_kb_completion,
    )
