#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formurmel.config import AppConfig, load_config  # noqa: E402
from formurmel.episode import AgentEpisodeRunner  # noqa: E402
from formurmel.logging import ConsoleLogger, NullLogger  # noqa: E402


DEFAULT_PROMPT = (
    "Formalize the target problem stored in the knowledge base. "
    "There is exactly one target statement. Use the available tools. "
    "Only stop after the stored proof has been verified with kb.compile_node. "
    "If the mathematical statement itself is wrong, report that with kb.report_statement_issue."
)


@dataclass(frozen=True)
class RolloutTask:
    config_path: Path
    dataset: str
    split: str
    problem_stem: str
    sample_index: int
    episode_dir: Path


class ProblemRolloutProgress:
    def __init__(self, *, tasks: list[RolloutTask], samples_per_problem: int) -> None:
        self.samples_per_problem = int(samples_per_problem)
        self._lock = threading.Lock()
        self._problem_order: list[str] = []
        self._states: dict[str, dict[int, str]] = {}
        for task in tasks:
            if task.problem_stem not in self._states:
                self._problem_order.append(task.problem_stem)
            samples = self._states.setdefault(
                task.problem_stem,
                {sample_index: "queued" for sample_index in range(self.samples_per_problem)},
            )
            samples.setdefault(task.sample_index, "queued")
        self._rendered = False

    def start_display(self) -> None:
        with self._lock:
            if self._rendered:
                return
            for problem_stem in self._problem_order:
                print(self._format_problem_line(problem_stem), flush=False)
            sys.stdout.flush()
            self._rendered = True

    def start(self, task: RolloutTask) -> None:
        self._set(task, "running")

    def step(self, task: RolloutTask, step: int) -> None:
        self._set(task, f"step {int(step)}")

    def finish(self, task: RolloutTask, *, status: str, steps: int | None = None) -> None:
        if status == "skipped":
            value = "skipped"
        elif steps is None:
            value = status
        else:
            value = f"{status}({steps})"
        self._set(task, value)

    def _set(self, task: RolloutTask, value: str) -> None:
        with self._lock:
            self._states.setdefault(task.problem_stem, {})[task.sample_index] = value
            self._render_locked()

    def _format_problem_line(self, problem_stem: str) -> str:
        samples = self._states.get(problem_stem, {})
        parts = []
        for sample_index in range(self.samples_per_problem):
            state = samples.get(sample_index, "queued")
            parts.append(f"rollout {sample_index + 1}: {state}")
        return f"{problem_stem}: " + ", ".join(parts)

    def _render_locked(self) -> None:
        if not self._problem_order:
            return
        if not self._rendered:
            for problem_stem in self._problem_order:
                print(self._format_problem_line(problem_stem), flush=False)
            sys.stdout.flush()
            self._rendered = True
            return

        # Cursor is kept immediately after the status block. Move back to the
        # first status line, rewrite the whole block, and leave the cursor below it.
        sys.stdout.write(f"\x1b[{len(self._problem_order)}F")
        for problem_stem in self._problem_order:
            sys.stdout.write("\x1b[2K" + self._format_problem_line(problem_stem) + "\n")
        sys.stdout.flush()


class RolloutProgressLogger:
    _STEP_RE = re.compile(r"^\[step=(\d+)\]")

    def __init__(self, *, display: ProblemRolloutProgress, task: RolloutTask) -> None:
        self.display = display
        self.task = task

    def log(self, message: str, verbosity_level: int = 0) -> None:
        if verbosity_level != 0:
            return
        match = self._STEP_RE.match(message.strip())
        if match is None:
            return
        self.display.step(self.task, int(match.group(1)))


_WORKER_STATE = threading.local()
_SUMMARY_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated formurmel rollouts for generated problem configs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory created by scripts/build_problem_configs.py.",
    )
    parser.add_argument("--dataset", default="basic_problems", help="Dataset name under <run-dir>/configs.")
    parser.add_argument("--split", default="train", help="Split name under <run-dir>/configs/<dataset>.")
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=2,
        help="Number of rollouts to generate per problem. Default: 2.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of rollout worker threads. Default: 1.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt for every rollout.")
    parser.add_argument(
        "--max-turns",
        type=int,
        help=(
            "Maximum agent turns per rollout. One turn is one model-query iteration "
            "in the agent loop. Alias for --max-steps."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Deprecated alias for --max-turns.",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        help="Optional explicit configs root. Defaults to <run-dir>/configs.",
    )
    parser.add_argument(
        "--rollouts-root",
        type=Path,
        help="Optional explicit rollout artifact root. Defaults to <run-dir>/rollouts.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional JSONL summary path. Defaults to <run-dir>/rollout_summary.jsonl.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip samples whose transcript already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing sample directories before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected rollout count without running the agent.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-problem rollout step status lines.")
    parser.add_argument("--verbose-rollouts", action="store_true", help="Print full agent/tool traffic.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_problem <= 0:
        raise SystemExit("--samples-per-problem must be positive")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.max_turns is not None and args.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.max_turns is not None and args.max_steps is not None and args.max_turns != args.max_steps:
        raise SystemExit("use at most one of --max-turns and --max-steps")
    if args.resume and args.overwrite:
        raise SystemExit("use at most one of --resume and --overwrite")
    if not args.prompt.strip():
        raise SystemExit("--prompt must be non-empty")


def effective_max_turns(args: argparse.Namespace) -> int | None:
    return args.max_turns if args.max_turns is not None else args.max_steps


def discover_tasks(args: argparse.Namespace) -> list[RolloutTask]:
    run_dir = args.run_dir.expanduser().resolve()
    configs_root = args.configs_root.expanduser().resolve() if args.configs_root else run_dir / "configs"
    split_dir = configs_root / args.dataset / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"config split directory not found: {split_dir}")

    rollouts_root = args.rollouts_root.expanduser().resolve() if args.rollouts_root else run_dir / "rollouts"
    config_paths = sorted(split_dir.glob("*.agent.toml"))
    if not config_paths:
        raise SystemExit(f"no *.agent.toml configs found under {split_dir}")

    tasks: list[RolloutTask] = []
    for config_path in config_paths:
        problem_stem = config_path.name.removesuffix(".agent.toml")
        for sample_index in range(args.samples_per_problem):
            episode_dir = rollouts_root / args.dataset / args.split / problem_stem / f"sample_{sample_index:02d}"
            tasks.append(
                RolloutTask(
                    config_path=config_path,
                    dataset=args.dataset,
                    split=args.split,
                    problem_stem=problem_stem,
                    sample_index=sample_index,
                    episode_dir=episode_dir,
                )
            )
    return tasks


def _runner_key(config: AppConfig) -> tuple[Any, ...]:
    return (
        config.backend,
        replace(config.tools, kb_path=None),
        config.agent.system_prompt,
        config.agent.max_steps,
        config.agent.verbose,
    )


def _get_runner(config: AppConfig) -> AgentEpisodeRunner:
    key = _runner_key(config)
    runners = getattr(_WORKER_STATE, "runners", None)
    if runners is None:
        runners = {}
        _WORKER_STATE.runners = runners
    runner = runners.get(key)
    if runner is None:
        template = replace(
            config,
            tools=replace(config.tools, kb_path=None),
            agent=replace(config.agent, transcript_path=None),
        )
        runner = AgentEpisodeRunner(template)
        runners[key] = runner
    return runner


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _SUMMARY_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _extract_final_message_text(result: Any) -> str | None:
    text = getattr(result, "final_message_text", None)
    if isinstance(text, str) and text.strip():
        return text
    return None


def run_task(
    task: RolloutTask,
    *,
    prompt: str,
    max_steps: int | None,
    summary_path: Path,
    resume: bool,
    overwrite: bool,
    progress_display: ProblemRolloutProgress | None,
    verbose_rollouts: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    transcript_path = task.episode_dir / "transcript.json"
    kb_path = task.episode_dir / "kb.json"
    if resume and transcript_path.exists():
        record = {
            "schema_version": "formurmel_rollout.v1",
            "event": "rollout",
            "skipped": True,
            "skip_reason": "transcript_exists",
            "dataset": task.dataset,
            "split": task.split,
            "problem_stem": task.problem_stem,
            "sample_index": task.sample_index,
            "config_path": str(task.config_path),
            "episode_dir": str(task.episode_dir),
            "kb_path": str(kb_path),
            "transcript_path": str(transcript_path),
            "max_turns": max_steps,
        }
        _append_jsonl(summary_path, record)
        if progress_display is not None:
            progress_display.finish(task, status="skipped")
        return record
    if overwrite and task.episode_dir.exists():
        shutil.rmtree(task.episode_dir)
    elif transcript_path.exists() or kb_path.exists():
        raise FileExistsError(f"rollout artifacts already exist for {task.episode_dir}; use --resume or --overwrite")

    config = load_config(task.config_path)
    if config.tools.kb_path is None:
        raise ValueError(f"config has no tools.kb_path: {task.config_path}")
    source_kb_path = config.tools.kb_path
    if not source_kb_path.exists():
        raise FileNotFoundError(f"source KB not found: {source_kb_path}")

    task.episode_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_kb_path, kb_path)

    runner = _get_runner(config)
    if progress_display is not None:
        progress_display.start(task)
    logger = (
        ConsoleLogger(verbosity=1)
        if verbose_rollouts
        else RolloutProgressLogger(display=progress_display, task=task)
        if progress_display is not None
        else NullLogger()
    )
    try:
        episode = runner.run_episode(
            user_prompt=prompt,
            kb_path=kb_path,
            transcript_path=transcript_path,
            max_steps=max_steps if max_steps is not None else config.agent.max_steps,
            system_prompt=config.agent.system_prompt,
            logger=logger,
        )
    except Exception:
        if progress_display is not None:
            progress_display.finish(task, status="exception")
        raise
    if progress_display is not None:
        progress_display.finish(task, status=episode.status.value, steps=episode.steps)
    duration = time.perf_counter() - started
    record = {
        "schema_version": "formurmel_rollout.v1",
        "event": "rollout",
        "skipped": False,
        "dataset": task.dataset,
        "split": task.split,
        "problem_stem": task.problem_stem,
        "sample_index": task.sample_index,
        "config_path": str(task.config_path),
        "episode_dir": str(task.episode_dir),
        "kb_path": str(kb_path),
        "transcript_path": str(transcript_path),
        "agent_status": episode.status.value,
        "agent_error": episode.error,
        "steps": episode.steps,
        "max_turns": max_steps,
        "final_message_text": _extract_final_message_text(episode),
        "duration_seconds": duration,
    }
    _append_jsonl(summary_path, record)
    return record


def main() -> int:
    args = parse_args()
    _validate_args(args)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")
    summary_path = args.summary_path.expanduser().resolve() if args.summary_path else run_dir / "rollout_summary.jsonl"
    tasks = discover_tasks(args)
    max_turns = effective_max_turns(args)
    progress_display = (
        ProblemRolloutProgress(tasks=tasks, samples_per_problem=args.samples_per_problem)
        if not args.no_progress and not args.verbose_rollouts and not args.dry_run
        else None
    )

    print(
        f"running {len(tasks)} rollout(s): dataset={args.dataset} split={args.split} "
        f"problems={len(tasks) // args.samples_per_problem} samples_per_problem={args.samples_per_problem} "
        f"jobs={args.jobs} max_turns={max_turns if max_turns is not None else 'config'}",
        flush=True,
    )
    print(f"summary: {summary_path}", flush=True)
    if args.dry_run:
        if tasks:
            print(f"first: {tasks[0].config_path} sample={tasks[0].sample_index}", flush=True)
            print(f"last: {tasks[-1].config_path} sample={tasks[-1].sample_index}", flush=True)
        return 0
    if progress_display is not None:
        progress_display.start_display()

    ok = 0
    errors = 0
    skipped = 0

    def _run(task: RolloutTask) -> dict[str, Any]:
        return run_task(
            task,
            prompt=args.prompt,
            max_steps=max_turns,
            summary_path=summary_path,
            resume=args.resume,
            overwrite=args.overwrite,
            progress_display=progress_display,
            verbose_rollouts=args.verbose_rollouts,
        )

    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            if progress_display is None:
                print(f"[{index}/{len(tasks)}] {task.problem_stem} sample={task.sample_index}", flush=True)
            record = _run(task)
            if record.get("skipped"):
                skipped += 1
            elif record.get("agent_status") == "done":
                ok += 1
            else:
                errors += 1
            if progress_display is None:
                print(
                    f"[{index}/{len(tasks)}] status={record.get('agent_status', 'skipped')} "
                    f"steps={record.get('steps', '-')}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_task = {executor.submit(_run, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                record = future.result()
                if record.get("skipped"):
                    skipped += 1
                elif record.get("agent_status") == "done":
                    ok += 1
                else:
                    errors += 1
                if progress_display is None:
                    print(
                        f"[{index}/{len(tasks)}] {task.problem_stem} sample={task.sample_index} "
                        f"status={record.get('agent_status', 'skipped')} steps={record.get('steps', '-')}",
                        flush=True,
                    )

    print(f"complete: done={ok} error={errors} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
