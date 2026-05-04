#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formurmel.kb import KB  # noqa: E402


DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "system_prompt.md"
DEFAULT_LEAN_PROJECT = (
    REPO_ROOT.parent / "cleaner" / "lean_env"
    if (REPO_ROOT.parent / "cleaner" / "lean_env").exists()
    else REPO_ROOT / "lean_env"
)
THEOREM_NAME_RE = re.compile(
    r"^\s*(?:theorem|lemma|def|abbrev|example)\s+([^\s:(\[{]+)",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-problem KB/config artifacts for formurmel datasets.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Directory containing dataset folders. Default: ./datasets.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset folder name under --datasets-root. May be repeated. "
            "Defaults to every child that contains lean_dataset_json."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        type=Path,
        help=(
            "Explicit dataset root. Accepts either a dataset folder or its lean_dataset_json "
            "directory. May be repeated."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Optional split filter, for example: train eval validation. Defaults to all split directories.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root directory under which the generated run directory is created.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional explicit run directory name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--backend",
        default="qwen35_llama_cpp",
        choices=("deepseek", "openrouter", "qwen35_llama_cpp"),
        help="Backend type written into generated configs. Default: qwen35_llama_cpp.",
    )
    parser.add_argument(
        "--llama-base-url",
        default="http://localhost:8080",
        help="llama.cpp /completion server base URL written into configs.",
    )
    parser.add_argument("--model", help="Hosted backend model name written into configs.")
    parser.add_argument("--api-base-url", help="Hosted backend API base URL written into configs.")
    parser.add_argument("--api-key-env", help="Hosted backend API key environment variable.")
    parser.add_argument(
        "--enable-reasoning",
        dest="reasoning_enabled",
        action="store_true",
        default=None,
        help="Write reasoning_enabled = true for hosted backends.",
    )
    parser.add_argument(
        "--disable-reasoning",
        dest="reasoning_enabled",
        action="store_false",
        help="Write reasoning_enabled = false for hosted backends.",
    )
    parser.add_argument("--reasoning-effort", help="Hosted backend reasoning effort.")
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        help="OpenRouter reasoning token budget written into configs.",
    )
    parser.add_argument(
        "--reasoning-exclude",
        action="store_true",
        help="Write reasoning_exclude = true for OpenRouter configs.",
    )
    parser.add_argument("--openrouter-site-url", help="Optional OpenRouter HTTP-Referer header value.")
    parser.add_argument("--openrouter-app-name", help="Optional OpenRouter X-Title header value.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Write qwen35_enable_thinking = false.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature written into generated configs. Default: 1.0.",
    )
    parser.add_argument("--top-p", type=float, help="Optional top_p written into configs.")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Optional presence_penalty written into configs.",
    )
    parser.add_argument("--max-tokens", type=int, help="Optional max_tokens written into configs.")
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=300.0,
        help="Backend request timeout in seconds. Default: 300.0.",
    )
    parser.add_argument(
        "--backend-retries",
        type=int,
        default=3,
        help="Backend retry count. Default: 3.",
    )
    parser.add_argument(
        "--lean-project",
        type=Path,
        default=DEFAULT_LEAN_PROJECT,
        help="Lake project path written for kb/lean tools. Default: ../cleaner/lean_env if present.",
    )
    parser.add_argument(
        "--lean-timeout",
        type=float,
        default=60.0,
        help="Lean tool timeout in seconds. Default: 60.0.",
    )
    parser.add_argument(
        "--murmel-cache-dir",
        type=Path,
        help="Optional murmel cache directory written into configs.",
    )
    parser.add_argument(
        "--murmel-config-path",
        type=Path,
        help="Optional murmel config file written into configs.",
    )
    parser.add_argument(
        "--murmel-mathlib-rev",
        help="Optional mathlib revision selector written into configs.",
    )
    parser.add_argument(
        "--murmel-semantic-device",
        help="Optional semantic search device written into configs.",
    )
    parser.add_argument(
        "--murmel-semantic-score-chunk-size",
        type=int,
        default=16384,
        help="Semantic scoring chunk size written into configs. Default: 16384.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="System prompt file referenced by generated configs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Agent max_steps written into each config. Default: 300.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write agent.verbose = true.",
    )
    parser.add_argument(
        "--no-kb-autosave",
        action="store_true",
        help="Write kb_autosave = false.",
    )
    return parser.parse_args()


def quote_toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def relpath(target: Path, start: Path) -> str:
    return Path(os.path.relpath(str(target.resolve()), str(start.resolve()))).as_posix()


def resolve_dataset_root(path: Path) -> tuple[str, Path]:
    resolved = path.expanduser().resolve()
    if (resolved / "lean_dataset_json").is_dir():
        return resolved.name, resolved / "lean_dataset_json"
    if resolved.name == "lean_dataset_json" and resolved.is_dir():
        return resolved.parent.name, resolved
    raise SystemExit(f"dataset root must be a dataset folder or lean_dataset_json dir: {resolved}")


def discover_dataset_roots(args: argparse.Namespace) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    if args.dataset_root:
        roots.extend(resolve_dataset_root(path) for path in args.dataset_root)

    datasets_root = args.datasets_root.expanduser().resolve()
    if args.dataset:
        for dataset_name in args.dataset:
            if not dataset_name.strip():
                raise SystemExit("--dataset entries must be non-empty")
            roots.append(resolve_dataset_root(datasets_root / dataset_name))
    elif not args.dataset_root:
        if not datasets_root.is_dir():
            raise SystemExit(f"datasets root not found: {datasets_root}")
        for child in sorted(datasets_root.iterdir()):
            if child.is_dir() and (child / "lean_dataset_json").is_dir():
                roots.append((child.name, child / "lean_dataset_json"))

    if not roots:
        raise SystemExit("no datasets selected")

    unique: dict[Path, tuple[str, Path]] = {}
    for name, root in roots:
        unique[root.resolve()] = (name, root.resolve())
    return list(unique.values())


def split_problem_files(dataset_root: Path, splits: list[str] | None) -> dict[str, list[Path]]:
    if splits is None:
        split_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    else:
        split_dirs = []
        for split in splits:
            split_dir = dataset_root / split
            if not split_dir.exists():
                raise SystemExit(f"dataset split not found: {split_dir}")
            if not split_dir.is_dir():
                raise SystemExit(f"dataset split is not a directory: {split_dir}")
            split_dirs.append(split_dir)

    by_split: dict[str, list[Path]] = {}
    for split_dir in split_dirs:
        files = sorted(split_dir.glob("*.json"))
        if files:
            by_split[split_dir.name] = files
    return by_split


def parse_theorem_name(lean_statement: str, fallback: str) -> str:
    match = THEOREM_NAME_RE.search(lean_statement)
    return fallback if match is None else match.group(1).strip()


def load_problem(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"problem file is not a JSON object: {path}")

    statement = raw.get("statement")
    lean_statement = raw.get("lean_statement")
    proof = raw.get("proof", "")
    if proof is None:
        proof = ""
    if not isinstance(statement, str) or not statement.strip():
        raise SystemExit(f"problem file missing non-empty 'statement': {path}")
    if not isinstance(lean_statement, str) or not lean_statement.strip():
        raise SystemExit(f"problem file missing non-empty 'lean_statement': {path}")
    if not isinstance(proof, str):
        raise SystemExit(f"problem file field 'proof' must be a string when present: {path}")

    return {
        "statement": statement.strip(),
        "lean_statement": lean_statement.rstrip(),
        "proof": proof.strip(),
    }


def write_kb(problem_path: Path, kb_path: Path, lean_project: Path) -> None:
    problem = load_problem(problem_path)
    node_id = problem_path.stem
    kb = KB.create(lake_project=lean_project)
    kb.add_node(
        id=node_id,
        type="statement",
        natural_language=problem["statement"],
        lean_name=parse_theorem_name(problem["lean_statement"], fallback=node_id),
        lean=problem["lean_statement"],
        natural_language_proof=problem["proof"] if problem["proof"] else None,
    )
    kb.save(kb_path)


def build_backend_lines(args: argparse.Namespace) -> list[str]:
    lines = [
        "[backend]",
        f"type = {quote_toml_string(args.backend)}",
        f"timeout = {args.backend_timeout}",
        f"retries = {args.backend_retries}",
    ]
    if args.backend == "qwen35_llama_cpp":
        lines.extend(
            [
                f"llama_base_url = {quote_toml_string(args.llama_base_url)}",
                f"qwen35_enable_thinking = {'false' if args.disable_thinking else 'true'}",
                f"temperature = {args.temperature}",
            ]
        )
    else:
        if args.model is not None:
            lines.append(f"model = {quote_toml_string(args.model)}")
        if args.api_base_url is not None:
            lines.append(f"api_base_url = {quote_toml_string(args.api_base_url)}")
        if args.api_key_env is not None:
            lines.append(f"api_key_env = {quote_toml_string(args.api_key_env)}")
        if args.reasoning_enabled is not None:
            lines.append(f"reasoning_enabled = {'true' if args.reasoning_enabled else 'false'}")
        if args.reasoning_effort is not None:
            lines.append(f"reasoning_effort = {quote_toml_string(args.reasoning_effort)}")
        if args.reasoning_max_tokens is not None:
            lines.append(f"reasoning_max_tokens = {args.reasoning_max_tokens}")
        if args.reasoning_exclude:
            lines.append("reasoning_exclude = true")
        if args.openrouter_site_url is not None:
            lines.append(f"openrouter_site_url = {quote_toml_string(args.openrouter_site_url)}")
        if args.openrouter_app_name is not None:
            lines.append(f"openrouter_app_name = {quote_toml_string(args.openrouter_app_name)}")
        if args.temperature is not None:
            lines.append(f"temperature = {args.temperature}")
    if args.top_p is not None:
        lines.append(f"top_p = {args.top_p}")
    if args.presence_penalty is not None:
        lines.append(f"presence_penalty = {args.presence_penalty}")
    if args.max_tokens is not None:
        lines.append(f"max_tokens = {args.max_tokens}")
    return lines


def build_tools_lines(
    *,
    args: argparse.Namespace,
    config_path: Path,
    kb_path: Path,
    lean_project: Path,
) -> list[str]:
    config_dir = config_path.parent
    lines = [
        "[tools]",
        f"kb_path = {quote_toml_string(relpath(kb_path, config_dir))}",
        f"kb_lake_project = {quote_toml_string(relpath(lean_project, config_dir))}",
        f"kb_autosave = {'false' if args.no_kb_autosave else 'true'}",
        f"lean_lake_project = {quote_toml_string(relpath(lean_project, config_dir))}",
        f"lean_default_timeout_seconds = {args.lean_timeout}",
        f"murmel_semantic_score_chunk_size = {args.murmel_semantic_score_chunk_size}",
    ]
    if args.murmel_cache_dir is not None:
        lines.append(
            f"murmel_cache_dir = {quote_toml_string(relpath(args.murmel_cache_dir, config_dir))}"
        )
    if args.murmel_config_path is not None:
        lines.append(
            f"murmel_config_path = {quote_toml_string(relpath(args.murmel_config_path, config_dir))}"
        )
    if args.murmel_mathlib_rev:
        lines.append(f"murmel_mathlib_rev = {quote_toml_string(args.murmel_mathlib_rev)}")
    if args.murmel_semantic_device:
        lines.append(f"murmel_semantic_device = {quote_toml_string(args.murmel_semantic_device)}")
    return lines


def build_agent_lines(
    *,
    args: argparse.Namespace,
    config_path: Path,
    transcript_path: Path,
) -> list[str]:
    config_dir = config_path.parent
    return [
        "[agent]",
        f"system_prompt_path = {quote_toml_string(relpath(args.system_prompt_path, config_dir))}",
        f"transcript_path = {quote_toml_string(relpath(transcript_path, config_dir))}",
        f"max_steps = {args.max_steps}",
        f"verbose = {'true' if args.verbose else 'false'}",
    ]


def write_config(
    *,
    args: argparse.Namespace,
    config_path: Path,
    kb_path: Path,
    transcript_path: Path,
    lean_project: Path,
) -> None:
    lines: list[str] = []
    lines.extend(build_backend_lines(args))
    lines.append("")
    lines.extend(
        build_tools_lines(
            args=args,
            config_path=config_path,
            kb_path=kb_path,
            lean_project=lean_project,
        )
    )
    lines.append("")
    lines.extend(
        build_agent_lines(
            args=args,
            config_path=config_path,
            transcript_path=transcript_path,
        )
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.lean_timeout <= 0:
        raise SystemExit("--lean-timeout must be positive")
    if args.backend_timeout <= 0:
        raise SystemExit("--backend-timeout must be positive")
    if args.backend_retries <= 0:
        raise SystemExit("--backend-retries must be positive")
    if args.backend == "openrouter" and (args.model is None or not args.model.strip()):
        raise SystemExit("--model is required for --backend openrouter")
    if args.api_base_url is not None and not args.api_base_url.strip():
        raise SystemExit("--api-base-url must be non-empty")
    if args.api_key_env is not None and not args.api_key_env.strip():
        raise SystemExit("--api-key-env must be non-empty")
    if args.model is not None and not args.model.strip():
        raise SystemExit("--model must be non-empty")
    if args.reasoning_effort is not None and not args.reasoning_effort.strip():
        raise SystemExit("--reasoning-effort must be non-empty")
    if args.reasoning_max_tokens is not None and args.reasoning_max_tokens <= 0:
        raise SystemExit("--reasoning-max-tokens must be positive")
    if args.backend == "openrouter" and args.reasoning_effort is not None and args.reasoning_max_tokens is not None:
        raise SystemExit("--reasoning-effort and --reasoning-max-tokens are mutually exclusive for OpenRouter")
    if args.openrouter_site_url is not None and not args.openrouter_site_url.strip():
        raise SystemExit("--openrouter-site-url must be non-empty")
    if args.openrouter_app_name is not None and not args.openrouter_app_name.strip():
        raise SystemExit("--openrouter-app-name must be non-empty")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if args.top_p is not None and not (0 < args.top_p <= 1):
        raise SystemExit("--top-p must be in the interval (0, 1]")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.murmel_semantic_score_chunk_size <= 0:
        raise SystemExit("--murmel-semantic-score-chunk-size must be positive")
    if args.murmel_semantic_device is not None and not args.murmel_semantic_device.strip():
        raise SystemExit("--murmel-semantic-device must be non-empty")
    if not args.system_prompt_path.expanduser().exists():
        raise SystemExit(f"system prompt not found: {args.system_prompt_path.expanduser()}")


def main() -> int:
    args = parse_args()
    validate_args(args)

    datasets = discover_dataset_roots(args)
    selected_splits = args.splits if args.splits is not None else None
    run_name = args.run_name or f"problem_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.runs_root.expanduser().resolve() / run_name
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")

    configs_root = run_dir / "configs"
    transcripts_root = run_dir / "transcripts"
    configs_root.mkdir(parents=True, exist_ok=False)
    transcripts_root.mkdir(parents=True, exist_ok=False)

    lean_project = args.lean_project.expanduser().resolve()
    counts: dict[str, dict[str, int]] = {}
    generated = 0

    for dataset_name, dataset_root in datasets:
        split_files = split_problem_files(dataset_root, selected_splits)
        if not split_files:
            continue
        counts[dataset_name] = {}
        for split, problem_paths in split_files.items():
            counts[dataset_name][split] = len(problem_paths)
            for problem_path in problem_paths:
                base_name = problem_path.stem
                config_dir = configs_root / dataset_name / split
                transcript_dir = transcripts_root / dataset_name / split
                kb_path = config_dir / f"{base_name}.kb.json"
                config_path = config_dir / f"{base_name}.agent.toml"
                transcript_path = transcript_dir / f"{base_name}.transcript.json"

                write_kb(problem_path, kb_path, lean_project)
                write_config(
                    args=args,
                    config_path=config_path,
                    kb_path=kb_path,
                    transcript_path=transcript_path,
                    lean_project=lean_project,
                )
                generated += 1

    if generated == 0:
        raise SystemExit("no problem files found for selected datasets/splits")

    summary = {
        "run_dir": str(run_dir),
        "generated_problems": generated,
        "datasets": counts,
        "backend": args.backend,
        "lean_project": str(lean_project),
        "configs_root": str(configs_root),
        "transcripts_root": str(transcripts_root),
    }
    (run_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"run directory: {run_dir}")
    print(f"generated problems: {generated}")
    print(f"configs and KBs: {configs_root}")
    print(f"transcripts: {transcripts_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
