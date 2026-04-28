from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


_SUPPORTED_BACKENDS = {"qwen35_llama_cpp"}


@dataclass(frozen=True)
class BackendConfig:
    type: str = "qwen35_llama_cpp"
    llama_base_url: str = "http://localhost:8080"
    qwen35_enable_thinking: bool = True
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    timeout: float = 300.0
    retries: int = 3
    retry_cooldown_seconds: float = 1.0
    parse_retries: int = 1
    debug: bool = False


@dataclass(frozen=True)
class ToolsConfig:
    kb_path: Path | None = None
    kb_lake_project: Path | None = None
    kb_autosave: bool = True
    murmel_cache_dir: Path | None = None
    murmel_config_path: Path | None = None
    murmel_mathlib_rev: str | None = None
    murmel_semantic_device: str | None = "cpu"
    murmel_semantic_score_chunk_size: int = 16384
    lean_lake_project: Path | None = None
    lean_default_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str = ""
    system_prompt_path: Path | None = None
    transcript_path: Path | None = None
    max_steps: int = 300
    verbose: bool = False


@dataclass(frozen=True)
class AppConfig:
    backend: BackendConfig
    tools: ToolsConfig
    agent: AgentConfig


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"path must be a non-empty string: {value!r}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _as_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _as_bool(value: Any, name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _as_float(value: Any, name: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _as_int(value: Any, name: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _load_prompt_text(
    *,
    prompt_value: Any,
    prompt_path_value: Any,
    base: Path,
) -> tuple[str, Path | None]:
    prompt_path = _resolve_path(prompt_path_value, base=base)
    if prompt_value is None and prompt_path is None:
        return "", None
    if prompt_value is not None:
        if not isinstance(prompt_value, str):
            raise ValueError("agent.system_prompt must be a string")
        return prompt_value, prompt_path
    if prompt_path is None:
        raise ValueError("agent.system_prompt_path must be a valid path")
    return prompt_path.read_text(encoding="utf-8"), prompt_path


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError("config must be a TOML table")

    base_dir = config_path.parent

    backend_raw = data.get("backend", {}) or {}
    if not isinstance(backend_raw, Mapping):
        raise ValueError("backend config must be a table")
    backend_type = _as_optional_str(backend_raw.get("type"), "backend.type") or "qwen35_llama_cpp"
    if backend_type not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise ValueError(f"backend.type must be one of: {allowed}")
    backend = BackendConfig(
        type=backend_type,
        llama_base_url=_as_optional_str(backend_raw.get("llama_base_url"), "backend.llama_base_url")
        or "http://localhost:8080",
        qwen35_enable_thinking=_as_bool(
            backend_raw.get("qwen35_enable_thinking"),
            "backend.qwen35_enable_thinking",
            default=True,
        ),
        temperature=_as_optional_float(backend_raw.get("temperature"), "backend.temperature"),
        top_p=_as_optional_float(backend_raw.get("top_p"), "backend.top_p"),
        presence_penalty=_as_optional_float(
            backend_raw.get("presence_penalty"),
            "backend.presence_penalty",
        ),
        max_tokens=_as_optional_int(backend_raw.get("max_tokens"), "backend.max_tokens"),
        timeout=_as_float(backend_raw.get("timeout"), "backend.timeout", default=300.0),
        retries=_as_int(backend_raw.get("retries"), "backend.retries", default=3),
        retry_cooldown_seconds=_as_float(
            backend_raw.get("retry_cooldown_seconds"),
            "backend.retry_cooldown_seconds",
            default=1.0,
        ),
        parse_retries=_as_int(backend_raw.get("parse_retries"), "backend.parse_retries", default=1),
        debug=_as_bool(backend_raw.get("debug"), "backend.debug", default=False),
    )

    tools_raw = data.get("tools", {}) or {}
    if not isinstance(tools_raw, Mapping):
        raise ValueError("tools config must be a table")
    tools = ToolsConfig(
        kb_path=_resolve_path(tools_raw.get("kb_path"), base=base_dir),
        kb_lake_project=_resolve_path(tools_raw.get("kb_lake_project"), base=base_dir),
        kb_autosave=_as_bool(tools_raw.get("kb_autosave"), "tools.kb_autosave", default=True),
        murmel_cache_dir=_resolve_path(tools_raw.get("murmel_cache_dir"), base=base_dir),
        murmel_config_path=_resolve_path(tools_raw.get("murmel_config_path"), base=base_dir),
        murmel_mathlib_rev=_as_optional_str(
            tools_raw.get("murmel_mathlib_rev"),
            "tools.murmel_mathlib_rev",
        ),
        murmel_semantic_device=_as_optional_str(
            tools_raw.get("murmel_semantic_device"),
            "tools.murmel_semantic_device",
        )
        or "cpu",
        murmel_semantic_score_chunk_size=_as_int(
            tools_raw.get("murmel_semantic_score_chunk_size"),
            "tools.murmel_semantic_score_chunk_size",
            default=16384,
        ),
        lean_lake_project=_resolve_path(tools_raw.get("lean_lake_project"), base=base_dir),
        lean_default_timeout_seconds=_as_float(
            tools_raw.get("lean_default_timeout_seconds"),
            "tools.lean_default_timeout_seconds",
            default=60.0,
        ),
    )

    agent_raw = data.get("agent", {}) or {}
    if not isinstance(agent_raw, Mapping):
        raise ValueError("agent config must be a table")
    system_prompt, system_prompt_path = _load_prompt_text(
        prompt_value=agent_raw.get("system_prompt"),
        prompt_path_value=agent_raw.get("system_prompt_path"),
        base=base_dir,
    )
    agent = AgentConfig(
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        transcript_path=_resolve_path(agent_raw.get("transcript_path"), base=base_dir),
        max_steps=_as_int(agent_raw.get("max_steps"), "agent.max_steps", default=300),
        verbose=_as_bool(agent_raw.get("verbose"), "agent.verbose", default=False),
    )

    if backend.timeout <= 0:
        raise ValueError("backend.timeout must be positive")
    if backend.retries <= 0:
        raise ValueError("backend.retries must be positive")
    if backend.retry_cooldown_seconds < 0:
        raise ValueError("backend.retry_cooldown_seconds must be non-negative")
    if backend.parse_retries < 0:
        raise ValueError("backend.parse_retries must be non-negative")
    if tools.murmel_semantic_device is not None and not tools.murmel_semantic_device.strip():
        raise ValueError("tools.murmel_semantic_device must be a non-empty string")
    if tools.murmel_semantic_score_chunk_size <= 0:
        raise ValueError("tools.murmel_semantic_score_chunk_size must be positive")
    if tools.lean_default_timeout_seconds <= 0:
        raise ValueError("tools.lean_default_timeout_seconds must be positive")
    if agent.max_steps <= 0:
        raise ValueError("agent.max_steps must be positive")

    return AppConfig(backend=backend, tools=tools, agent=agent)
