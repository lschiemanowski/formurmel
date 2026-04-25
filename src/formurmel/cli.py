from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from formurmel.agent import AgentStatus
from formurmel.config import load_config
from formurmel.diagnosis import DEFAULT_DIAGNOSIS_PROMPT, run_diagnosis_from_config
from formurmel.logging import ConsoleLogger
from formurmel.runtime import run_from_config


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("use either --prompt or --prompt-file, not both")
    if args.prompt_file is not None:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return sys.stdin.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formurmel Lean formalization agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("formurmel.toml"),
        help="TOML config file. Defaults to ./formurmel.toml.",
    )
    parser.add_argument("--prompt", help="User prompt to send to the agent.")
    parser.add_argument("--prompt-file", type=Path, help="Read the user prompt from this file.")
    parser.add_argument("--verbose", action="store_true", help="Log full agent/tool traffic.")
    return parser


def build_diagnose_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose a failed formurmel/cleaner-style transcript.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("formurmel.toml"),
        help="TOML config file. Backend and normal tools are reused from this config.",
    )
    parser.add_argument(
        "--failed-transcript",
        type=Path,
        required=True,
        help="Saved failed transcript JSON to inspect.",
    )
    parser.add_argument(
        "--diagnosis-transcript",
        type=Path,
        help="Where to save the diagnosis agent transcript. Defaults to a sibling *.diagnosis.transcript.json file.",
    )
    parser.add_argument("--output", type=Path, help="Write structured diagnosis artifact JSON here.")
    parser.add_argument("--prompt", help="Override the default diagnosis prompt.")
    parser.add_argument("--prompt-file", type=Path, help="Read a custom diagnosis prompt from this file.")
    parser.add_argument("--verbose", action="store_true", help="Log full diagnosis agent/tool traffic.")
    return parser


def _default_diagnosis_transcript_path(failed_transcript: Path) -> Path:
    name = failed_transcript.name
    if name.endswith(".transcript.json"):
        return failed_transcript.with_name(f"{name[: -len('.transcript.json')]}.diagnosis.transcript.json")
    if name.endswith(".json"):
        return failed_transcript.with_name(f"{name[: -len('.json')]}.diagnosis.transcript.json")
    return failed_transcript.with_name(f"{name}.diagnosis.transcript.json")


def _run_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        prompt = _read_prompt(args)
        if not prompt.strip():
            parser.error("provide a non-empty prompt with --prompt, --prompt-file, or stdin")
        config = load_config(args.config)
        if args.verbose:
            config = replace(config, agent=replace(config.agent, verbose=True))
        result = run_from_config(config, prompt, logger=ConsoleLogger(verbosity=1 if args.verbose else 0))
    except Exception as exc:  # noqa: BLE001
        print(f"formurmel: {exc}", file=sys.stderr)
        return 2

    if result.final_message is not None and isinstance(result.final_message.content, str):
        print(result.final_message.content)

    if result.status == AgentStatus.DONE:
        return 0

    if result.error:
        print(result.error, file=sys.stderr)
    return 1


def _diagnose_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        prompt = _read_prompt(args) if args.prompt is not None or args.prompt_file is not None else None
        config = load_config(args.config)
        if args.verbose:
            config = replace(config, agent=replace(config.agent, verbose=True))
        diagnosis_transcript = args.diagnosis_transcript or _default_diagnosis_transcript_path(args.failed_transcript)
        result = run_diagnosis_from_config(
            config,
            args.failed_transcript,
            diagnosis_transcript_path=diagnosis_transcript,
            output_path=args.output,
            prompt=prompt if prompt is not None else DEFAULT_DIAGNOSIS_PROMPT,
            logger=ConsoleLogger(verbosity=1 if args.verbose else 0),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"formurmel diagnose: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.artifact, ensure_ascii=False, indent=2))
    if result.agent_result.status != AgentStatus.DONE:
        return 1
    if result.diagnosis is None:
        parse_error = result.artifact.get("diagnosis_run", {}).get("parse_error")
        if parse_error:
            print(parse_error, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "diagnose":
        parser = build_diagnose_parser()
        args = parser.parse_args(raw_args[1:])
        return _diagnose_main(args, parser)
    if raw_args and raw_args[0] == "run":
        raw_args = raw_args[1:]
    parser = build_parser()
    args = parser.parse_args(raw_args)
    return _run_main(args, parser)
