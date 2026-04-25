from __future__ import annotations

from pathlib import Path

from formurmel.tools.base import ToolRegistry
from formurmel.tools.kb_tool import KBTool
from formurmel.tools.lean_tool import LeanTool
from formurmel.tools.murmel_tool import MurmelTool


def create_tool_registry(
    *,
    kb_path: Path | None,
    kb_lake_project: Path | None,
    kb_autosave: bool,
    murmel_cache_dir: Path | None,
    murmel_config_path: Path | None,
    murmel_mathlib_rev: str | None,
    murmel_semantic_device: str | None,
    murmel_semantic_score_chunk_size: int,
    lean_lake_project: Path | None,
    lean_default_timeout_seconds: float = 60.0,
) -> ToolRegistry:
    registry = ToolRegistry()
    if kb_path is not None:
        registry.register(
            KBTool(
                kb_path=kb_path,
                lake_project=kb_lake_project,
                autosave=kb_autosave,
            )
        )
    registry.register(
        MurmelTool(
            cache_dir=murmel_cache_dir,
            config_path=murmel_config_path,
            mathlib_rev=murmel_mathlib_rev,
            semantic_device=murmel_semantic_device,
            semantic_score_chunk_size=murmel_semantic_score_chunk_size,
        )
    )
    registry.register(
        LeanTool(
            lake_project=lean_lake_project if lean_lake_project is not None else kb_lake_project,
            default_timeout_seconds=lean_default_timeout_seconds,
        )
    )
    return registry
