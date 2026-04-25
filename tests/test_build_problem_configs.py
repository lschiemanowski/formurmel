from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from formurmel.config import load_config
from formurmel.runtime import build_tool_registry


def test_build_problem_configs_generates_formurmel_configs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "demo" / "lean_dataset_json" / "train"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "problem_demo.json").write_text(
        json.dumps(
            {
                "statement": "Zero is a left identity for addition.",
                "lean_statement": "theorem demo_zero_add (n : Nat) : 0 + n = n",
                "proof": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "demo_run"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_problem_configs.py",
            "--datasets-root",
            str(tmp_path / "datasets"),
            "--dataset",
            "demo",
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-name",
            "demo_run",
            "--lean-project",
            str(tmp_path / "lake"),
            "--murmel-cache-dir",
            str(tmp_path / "murmel-cache"),
            "--max-steps",
            "5",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    config_path = run_dir / "configs" / "demo" / "train" / "problem_demo.agent.toml"
    kb_path = run_dir / "configs" / "demo" / "train" / "problem_demo.kb.json"
    summary_path = run_dir / "generation_summary.json"

    config = load_config(config_path)
    registry = build_tool_registry(config.tools)
    tool_names = [spec.name for spec in registry.specs()]
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert tool_names == ["kb", "murmel", "lean"]
    assert config.tools.murmel_cache_dir == tmp_path / "murmel-cache"
    assert config.tools.lean_lake_project == tmp_path / "lake"
    assert config.agent.max_steps == 5
    assert kb["nodes"]["problem_demo"]["lean_name"] == "demo_zero_add"
    assert kb["nodes"]["problem_demo"]["natural_language_proof"] is None
    assert summary["generated_problems"] == 1
