from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from formurmel.agent import AgentRunResult, build_initial_conversation, run
from formurmel.config import AppConfig
from formurmel.logging import Logger
from formurmel.runtime import build_backend, build_logger, build_tool_registry
from formurmel.tools.read_only_kb_tool import ReadOnlyKBTool
from formurmel.tools.transcript_inspect_tool import TranscriptInspectTool


_BARE_TACTIC_PROOF_RE = re.compile(
    r"^(?:by_contra|intro|intros|exact|apply|rw|simp|simpa|have|let|classical|cases'?|rcases|obtain|"
    r"constructor|left|right|use|refine|push_neg|push|linarith|nlinarith|norm_num|ring|omega|tauto|"
    r"contradiction|assumption)(?=\s|;|,|\{|$)"
)
_TRANSCRIPT_INDEX_FIELDS = {"message_index", "preceding_tool_call_index", "checked_preceding_tool_call"}
_BARE_TACTIC_TOOL_BLAME_RE = re.compile(
    r"\b(?:kb\s+)?(?:parser|parsing)\s+(?:bug|limitation|limitations)\b"
    r"|\b(?:block syntax|unsupported syntax)\b"
    r"|\bkb\s+parsing\s+does\s+not\s+support\b"
    r"|\bcompile_node\s+(?:bug|limitation|limitations|issue|problem)\b"
    r"|\btool/environment\s+limitation\b",
    flags=re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(?:not|no|without|isn't|is\s+not|wasn't|was\s+not|rather\s+than|instead\s+of)\b")


DIAGNOSIS_SYSTEM_PROMPT = """You are a Lean theorem-proving failure-analysis agent.

Your job is to diagnose a failed agent run, not to solve the theorem from scratch. Use the `transcript_inspect` tool to inspect the failed transcript. You may use the other available tools, such as `lean`, `murmel`, and `kb` when available, to verify whether a claimed tool/environment issue is real or whether a Lean error was caused by bad proof code.

Do not trust the failed agent's own reasoning, final answer, or self-diagnosis as facts. Treat them as claims to check against higher-authority evidence: run status, run error, tool outputs, stored KB fields such as `compiles` and `verified`, and direct verification snippets. Do not conclude that a tool or environment is broken merely because the failed agent said so or because the same malformed proof failed repeatedly.

For every cited Lean or KB error, inspect both the failing tool response and the immediately preceding tool call that submitted the code. Interpret the error in light of the submitted code before assigning blame. For KB statement proofs, remember the KB stores `lean` as the theorem statement and compiles a stored `lean_proof` by appending it after `:=`; therefore `lean_proof` must be a Lean proof term. Tactic-mode proofs must include their own proof-term `by`, for example `by exact ...` or `by\n  ...`. Bare tactic text such as `exact ...` or `by_contra h; ...` is not valid as a stored `lean_proof` unless it is wrapped as a proof term. `by_contra` is a tactic name, not the proof-term `by`.

Use correct proof-format examples: `by by_contra h; ...` for one-line tactic proofs, or `by\n  by_contra h\n  ...` for multiline tactic proofs. Do not recommend `by (by_contra h); ... end`.

Do not update the knowledge base unless explicitly asked. Prefer concrete causal diagnoses grounded in submitted-code excerpts, observed-error excerpts, and transferable correction rules over transcript bookkeeping."""


DEFAULT_DIAGNOSIS_PROMPT = """Analyze the failed Lean formalization run in the configured transcript.

You must use `transcript_inspect` before answering. Your goal is to answer:
- What failed?
- What was the first critical mistake or failure mode?
- Was it a real tool/environment problem, a proof-strategy problem, a Lean syntax/type error, bad recovery behavior, or incorrect self-assessment?
- What should the agent do differently next time to avoid this failure mode?

When diagnosing, apply these standards:
- Prefer the earliest recoverable mistake over later downstream symptoms.
- Start from the run metadata: status, step count, final run error, final message, and any KB verification fields.
- Treat assistant reasoning and final summaries in the failed transcript as unreliable model claims, not ground truth.
- If the model claimed success, compare that claim to the stored run status, tool outputs, and verification state.
- If the model claimed a tool or environment limitation, look for independent evidence. A real tool/environment issue should be supported by a minimal reproduction, a direct tool output unrelated to bad proof code, or consistent failure on otherwise valid code.
- If a tool failed, verify whether it was a real tool/environment failure or a misuse/recovery failure. For Lean and KB failures, inspect the exact submitted code or stored proof format before blaming the tool.
- For every Lean or KB error you cite, inspect the failing tool response and the immediately preceding tool call. The previous tool call is the submitted code; the tool response alone is not enough to diagnose root cause. In the final JSON, cite submitted-code excerpts and observed-error excerpts, not transcript indices.
- For KB `compile_node`/`compile_candidate` failures on statements, apply the KB proof-format contract: `lean_proof` is inserted after `:=`, so it must be a proof term. If the stored proof is tactic-mode text, it must start with the proof-term `by`; bare tactics such as `exact ...`, `by_contra h; ...`, or `intro h; ...` indicate a proof-format error, not a KB parser bug. `by_contra h` starts with the tactic `by_contra`, not with the proof-term `by`.
- Use correct proof-format examples: one-line `by by_contra h; ...` or multiline `by\n  by_contra h\n  ...`. Do not recommend invalid or non-idiomatic examples like `by (by_contra h); ... end`.
- If the model repeated similar failing edits, describe the repeated pattern and the missing recovery action.
- Distinguish final symptoms from root causes. A late compile error may be downstream of an earlier bad proof strategy, bad proof format, wrong lemma, or unverified model assumption.
- Ground every major claim in concrete evidence. When citing a model claim, say it is a model claim; when citing a tool output, include the relevant submitted-code excerpt and observed-error excerpt.
- Keep the advice operational: future rules should be actions an agent can follow during a run.

Return exactly one JSON object and nothing else. The JSON object must have these fields:
- `failure_type`: short snake_case category naming the root cause
- `first_critical_mistake`: object with fields `summary`, `submitted_code_excerpt`, `observed_error_excerpt`, and `why_it_failed`
- `root_cause`: string
- `evidence`: list of objects with fields `evidence_type`, `summary`, `submitted_code_excerpt`, `observed_error_excerpt`, and `interpretation`
- `kb_error_analysis`: object with fields `inspected_kb_node`, `inspected_kb_node_id`, `stored_lean_proof_excerpt`, `stored_lean_proof_is_valid_proof_term`, `proof_format_summary`, and `supports_tool_limitation`
- `false_model_claims`: list of strings
- `tool_or_environment_issues`: list of strings
- `future_strategy`: list of strings
- `minimal_correction`: string
- `confidence`: number between 0 and 1

For every item in `evidence`, at least one of `submitted_code_excerpt` or `observed_error_excerpt` must be a non-null string. Do not put the only concrete excerpt inside `interpretation`. For `evidence_type: "stored_kb_node"`, put the stored `lean_proof` or other inspected field directly in `submitted_code_excerpt`, for example `lean_proof = "by_contra h; ..."`. For `evidence_type: "tool_response"`, put the relevant error text directly in `observed_error_excerpt`.

Do not include structured index fields such as `message_index`, `preceding_tool_call_index`, or `checked_preceding_tool_call` in the final JSON. The diagnosis is intended as self-distillation training data, so prefer transferable evidence: exact snippets of bad submitted code, exact snippets of observed errors, and concise causal explanations. Free-text transcript coordinates are allowed when useful for debugging, but they are less important than the transferable failure pattern. For non-tool evidence such as a failed model's final claim, set `submitted_code_excerpt` to null and put the quoted or paraphrased claim in `observed_error_excerpt` or `summary`.

For `kb_error_analysis`, if any KB compile error is relevant, inspect `kb.get_node` or the transcript's stored KB node, quote the stored `lean_proof` in `stored_lean_proof_excerpt`, and decide whether that stored text is a valid proof term appended after `:=`. Set `inspected_kb_node` to the boolean `true` or `false` only. Put the inspected node id, such as `"problem_54"`, in `inspected_kb_node_id`; if no KB node was inspected, set `inspected_kb_node_id` to null. Set `supports_tool_limitation` to true only if the stored proof format is valid and independent checks support a real KB/tool limitation. If the stored `lean_proof` starts with a bare tactic such as `by_contra`, `intro`, `exact`, `apply`, or `cases`, mark it as invalid proof-term text and do not blame block syntax, parser limitations, unsupported syntax, or KB parsing. If that KB compile error is central, identify the missing proof-term `by` as the proof-format error; if an earlier proof-strategy mistake is the root cause, diagnose that root cause while still recording the stored proof-format issue. If no KB compile error is relevant, set `inspected_kb_node` to false, `inspected_kb_node_id`, `stored_lean_proof_excerpt`, and `stored_lean_proof_is_valid_proof_term` to null, and explain why in `proof_format_summary`.
"""


@dataclass(frozen=True)
class DiagnosisResult:
    agent_result: AgentRunResult
    diagnosis: dict[str, Any] | None
    artifact: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```json\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    else:
        start = stripped.find("{")
        if start > 0:
            stripped = stripped[start:]
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(stripped)
    if stripped[end:].strip():
        raise ValueError("diagnosis output contained trailing non-JSON content")
    if not isinstance(obj, dict):
        raise ValueError("diagnosis output must be a JSON object")
    return obj


def validate_diagnosis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "failure_type",
        "first_critical_mistake",
        "root_cause",
        "evidence",
        "kb_error_analysis",
        "false_model_claims",
        "tool_or_environment_issues",
        "future_strategy",
        "minimal_correction",
        "confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"diagnosis JSON missing required fields: {', '.join(missing)}")
    if not isinstance(payload["failure_type"], str) or not payload["failure_type"].strip():
        raise ValueError("diagnosis.failure_type must be a non-empty string")
    first = payload["first_critical_mistake"]
    if not isinstance(first, Mapping):
        raise ValueError("diagnosis.first_critical_mistake must be an object")
    _validate_first_critical_mistake(first)
    if not isinstance(payload["root_cause"], str) or not payload["root_cause"].strip():
        raise ValueError("diagnosis.root_cause must be a non-empty string")
    for key in ("evidence", "false_model_claims", "tool_or_environment_issues", "future_strategy"):
        if not isinstance(payload[key], list):
            raise ValueError(f"diagnosis.{key} must be a list")
    for index, item in enumerate(payload["evidence"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"diagnosis.evidence[{index}] must be an object")
        _validate_evidence_item(item, f"diagnosis.evidence[{index}]")
    for key in ("false_model_claims", "tool_or_environment_issues", "future_strategy"):
        for index, item in enumerate(payload[key]):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"diagnosis.{key}[{index}] must be a non-empty string")
    _validate_kb_error_analysis(payload["kb_error_analysis"])
    if not isinstance(payload["minimal_correction"], str) or not payload["minimal_correction"].strip():
        raise ValueError("diagnosis.minimal_correction must be a non-empty string")
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("diagnosis.confidence must be a number in [0, 1]")
    _validate_bare_tactic_proof_diagnosis(payload)
    return payload


def _validate_bare_tactic_proof_diagnosis(payload: Mapping[str, Any]) -> None:
    kb_analysis = payload["kb_error_analysis"]
    if not isinstance(kb_analysis, Mapping):
        return
    excerpt = kb_analysis.get("stored_lean_proof_excerpt")
    if not isinstance(excerpt, str) or not _looks_like_bare_tactic_proof(excerpt):
        return

    if _contains_bare_tactic_tool_blame(payload):
        raise ValueError(
            "diagnosis for a stored bare tactic lean_proof must not blame block syntax, "
            "parser limitations, unsupported syntax, KB parsing, or compile_node limitations"
        )


def _normalize_proof_excerpt(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            text = decoded.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _looks_like_bare_tactic_proof(value: str) -> bool:
    text = _normalize_proof_excerpt(value)
    return bool(_BARE_TACTIC_PROOF_RE.match(text))


def _contains_bare_tactic_tool_blame(payload: Mapping[str, Any]) -> bool:
    kb_analysis = payload.get("kb_error_analysis")
    first = payload.get("first_critical_mistake")
    texts: list[str] = []

    for key in ("root_cause", "minimal_correction"):
        value = payload.get(key)
        if isinstance(value, str):
            texts.append(value)
    if isinstance(first, Mapping):
        for key in ("summary", "why_it_failed"):
            value = first.get(key)
            if isinstance(value, str):
                texts.append(value)
    if isinstance(kb_analysis, Mapping):
        value = kb_analysis.get("proof_format_summary")
        if isinstance(value, str):
            texts.append(value)
    for key in ("tool_or_environment_issues", "future_strategy"):
        value = payload.get(key)
        if isinstance(value, list):
            texts.extend(item for item in value if isinstance(item, str))
    value = payload.get("evidence")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                interpretation = item.get("interpretation")
                if isinstance(interpretation, str):
                    texts.append(interpretation)

    return any(_contains_unnegated_tool_blame(text) for text in texts)


def _contains_unnegated_tool_blame(text: str) -> bool:
    for match in _BARE_TACTIC_TOOL_BLAME_RE.finditer(text):
        prefix = text[max(0, match.start() - 40) : match.start()]
        if _NEGATION_RE.search(prefix):
            continue
        return True
    return False


def _validate_optional_excerpt(value: Any, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string or null")


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_first_critical_mistake(item: Mapping[str, Any]) -> None:
    forbidden = sorted(_TRANSCRIPT_INDEX_FIELDS & set(item))
    if forbidden:
        raise ValueError(
            "diagnosis.first_critical_mistake must not include transcript index field(s): "
            + ", ".join(forbidden)
        )
    required = {"summary", "submitted_code_excerpt", "observed_error_excerpt", "why_it_failed"}
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"diagnosis.first_critical_mistake missing required fields: {', '.join(missing)}")
    _require_non_empty_string(item["summary"], "diagnosis.first_critical_mistake.summary")
    _validate_optional_excerpt(
        item["submitted_code_excerpt"], "diagnosis.first_critical_mistake.submitted_code_excerpt"
    )
    _validate_optional_excerpt(
        item["observed_error_excerpt"], "diagnosis.first_critical_mistake.observed_error_excerpt"
    )
    if item["submitted_code_excerpt"] is None and item["observed_error_excerpt"] is None:
        raise ValueError(
            "diagnosis.first_critical_mistake requires at least one of submitted_code_excerpt "
            "or observed_error_excerpt"
        )
    _require_non_empty_string(item["why_it_failed"], "diagnosis.first_critical_mistake.why_it_failed")


def _validate_evidence_item(item: Mapping[str, Any], name: str) -> None:
    forbidden = sorted(_TRANSCRIPT_INDEX_FIELDS & set(item))
    if forbidden:
        raise ValueError(f"{name} must not include transcript index field(s): {', '.join(forbidden)}")
    required = {"evidence_type", "summary", "submitted_code_excerpt", "observed_error_excerpt", "interpretation"}
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(missing)}")
    _require_non_empty_string(item["evidence_type"], f"{name}.evidence_type")
    _require_non_empty_string(item["summary"], f"{name}.summary")
    _validate_optional_excerpt(item["submitted_code_excerpt"], f"{name}.submitted_code_excerpt")
    _validate_optional_excerpt(item["observed_error_excerpt"], f"{name}.observed_error_excerpt")
    if item["submitted_code_excerpt"] is None and item["observed_error_excerpt"] is None:
        raise ValueError(f"{name} requires at least one of submitted_code_excerpt or observed_error_excerpt")
    _require_non_empty_string(item["interpretation"], f"{name}.interpretation")


def _validate_kb_error_analysis(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("diagnosis.kb_error_analysis must be an object")
    required = {
        "inspected_kb_node",
        "inspected_kb_node_id",
        "stored_lean_proof_excerpt",
        "stored_lean_proof_is_valid_proof_term",
        "proof_format_summary",
        "supports_tool_limitation",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"diagnosis.kb_error_analysis missing required fields: {', '.join(missing)}")
    if not isinstance(value["inspected_kb_node"], bool):
        raise ValueError("diagnosis.kb_error_analysis.inspected_kb_node must be a boolean")
    inspected = value["inspected_kb_node"]
    inspected_id = value["inspected_kb_node_id"]
    if inspected:
        if not isinstance(inspected_id, str) or not inspected_id.strip():
            raise ValueError(
                "diagnosis.kb_error_analysis.inspected_kb_node_id must be a non-empty string "
                "when inspected_kb_node is true"
            )
    elif inspected_id is not None:
        raise ValueError("diagnosis.kb_error_analysis.inspected_kb_node_id must be null when inspected_kb_node is false")
    excerpt = value["stored_lean_proof_excerpt"]
    if excerpt is not None and not isinstance(excerpt, str):
        raise ValueError("diagnosis.kb_error_analysis.stored_lean_proof_excerpt must be a string or null")
    proof_term = value["stored_lean_proof_is_valid_proof_term"]
    if proof_term is not None and not isinstance(proof_term, bool):
        raise ValueError("diagnosis.kb_error_analysis.stored_lean_proof_is_valid_proof_term must be a boolean or null")
    if not isinstance(value["proof_format_summary"], str) or not value["proof_format_summary"].strip():
        raise ValueError("diagnosis.kb_error_analysis.proof_format_summary must be a non-empty string")
    if not isinstance(value["supports_tool_limitation"], bool):
        raise ValueError("diagnosis.kb_error_analysis.supports_tool_limitation must be a boolean")
    supports_tool_limitation = value["supports_tool_limitation"]
    if supports_tool_limitation and proof_term is not True:
        raise ValueError(
            "diagnosis.kb_error_analysis.supports_tool_limitation requires "
            "stored_lean_proof_is_valid_proof_term to be true"
        )
    if isinstance(excerpt, str) and _looks_like_bare_tactic_proof(excerpt):
        if proof_term is True:
            raise ValueError(
                "diagnosis.kb_error_analysis.stored_lean_proof_is_valid_proof_term "
                "cannot be true for bare tactic text"
            )
        if supports_tool_limitation:
            raise ValueError(
                "diagnosis.kb_error_analysis.supports_tool_limitation cannot be true "
                "for bare tactic text"
            )


def run_diagnosis_from_config(
    config: AppConfig,
    failed_transcript_path: str | Path,
    *,
    diagnosis_transcript_path: str | Path | None = None,
    output_path: str | Path | None = None,
    prompt: str = DEFAULT_DIAGNOSIS_PROMPT,
    logger: Logger | None = None,
) -> DiagnosisResult:
    failed_transcript = Path(failed_transcript_path).expanduser().resolve()
    if not failed_transcript.exists():
        raise FileNotFoundError(f"failed transcript not found: {failed_transcript}")

    diagnosis_transcript = Path(diagnosis_transcript_path).expanduser().resolve() if diagnosis_transcript_path else None
    diagnosis_config = replace(
        config,
        agent=replace(
            config.agent,
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            system_prompt_path=None,
            transcript_path=diagnosis_transcript,
        ),
    )
    backend = build_backend(diagnosis_config.backend)
    tool_registry = build_tool_registry(diagnosis_config.tools)
    kb_tool = tool_registry.get("kb")
    if kb_tool is not None:
        tool_registry.register(ReadOnlyKBTool(kb_tool))
    tool_registry.register(TranscriptInspectTool(failed_transcript))

    active_logger = logger if logger is not None else build_logger(diagnosis_config.agent)
    conversation = build_initial_conversation(diagnosis_config.agent.system_prompt, prompt)
    agent_result = run(
        initial_conversation=conversation,
        tool_registry=tool_registry,
        backend=backend,
        transcript_path=diagnosis_config.agent.transcript_path,
        logger=active_logger,
        max_steps=diagnosis_config.agent.max_steps,
        require_kb_completion=False,
    )

    diagnosis: dict[str, Any] | None = None
    parse_error: str | None = None
    if agent_result.final_message is not None and isinstance(agent_result.final_message.content, str):
        try:
            diagnosis = validate_diagnosis_payload(extract_json_object(agent_result.final_message.content))
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)
    elif agent_result.status.value == "done":
        parse_error = "diagnosis run completed without a final text message"

    artifact = {
        "schema_version": "single_run_diagnosis.v1",
        "created_at": utc_now_iso(),
        "failed_transcript_path": str(failed_transcript),
        "diagnosis_run": {
            "agent_status": agent_result.status.value,
            "agent_error": agent_result.error,
            "steps": agent_result.steps,
            "transcript_path": str(diagnosis_transcript) if diagnosis_transcript is not None else None,
            "parse_error": parse_error,
        },
        "diagnosis": diagnosis,
    }
    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return DiagnosisResult(agent_result=agent_result, diagnosis=diagnosis, artifact=artifact)
