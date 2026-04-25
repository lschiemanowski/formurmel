from __future__ import annotations

from formurmel.diagnosis import DEFAULT_DIAGNOSIS_PROMPT, DIAGNOSIS_SYSTEM_PROMPT


def test_diagnosis_prompt_requires_checking_failed_agent_claims() -> None:
    system = DIAGNOSIS_SYSTEM_PROMPT.lower()
    prompt = DEFAULT_DIAGNOSIS_PROMPT.lower()

    assert "do not trust the failed agent's own reasoning" in system
    assert "treat assistant reasoning and final summaries" in prompt
    assert "unreliable model claims" in prompt


def test_diagnosis_prompt_requires_independent_tool_limitation_evidence() -> None:
    system = DIAGNOSIS_SYSTEM_PROMPT.lower()
    prompt = DEFAULT_DIAGNOSIS_PROMPT.lower()

    assert "do not conclude that a tool or environment is broken" in system
    assert "independent evidence" in prompt
    assert "inspect the exact submitted code or stored proof format" in prompt


def test_diagnosis_prompt_requires_previous_tool_call_for_errors() -> None:
    system = DIAGNOSIS_SYSTEM_PROMPT.lower()
    prompt = DEFAULT_DIAGNOSIS_PROMPT.lower()

    assert "immediately preceding tool call" in system
    assert "immediately preceding tool call" in prompt
    assert "tool response alone is not enough" in prompt
    assert "submitted-code excerpts" in system
    assert "observed-error excerpts" in system
    assert "submitted-code excerpts" in prompt
    assert "observed-error excerpts" in prompt


def test_diagnosis_prompt_explains_kb_lean_proof_contract() -> None:
    system = DIAGNOSIS_SYSTEM_PROMPT.lower()
    prompt = DEFAULT_DIAGNOSIS_PROMPT.lower()

    assert "lean_proof" in system
    assert "appending it after `:=`" in system
    assert "inserted after `:=`" in prompt
    assert "must start with the proof-term `by`" in prompt
    assert "`by_contra` is a tactic name" in system
    assert "not with the proof-term `by`" in prompt
    assert "by by_contra h" in system
    assert "by by_contra h" in prompt
    assert "do not recommend" in prompt
    assert "`by (by_contra h); ... end`" in prompt


def test_diagnosis_prompt_requires_structured_evidence_and_kb_analysis() -> None:
    prompt = DEFAULT_DIAGNOSIS_PROMPT
    prompt_lower = prompt.lower()

    assert "`submitted_code_excerpt`" in prompt
    assert "`observed_error_excerpt`" in prompt
    assert "`why_it_failed`" in prompt
    assert "`interpretation`" in prompt
    assert "at least one of `submitted_code_excerpt` or `observed_error_excerpt` must be a non-null string" in prompt
    assert "do not put the only concrete excerpt inside `interpretation`" in prompt_lower
    assert 'For `evidence_type: "stored_kb_node"`' in prompt
    assert 'lean_proof = "by_contra h; ..."' in prompt
    assert "`kb_error_analysis`" in prompt
    assert "`inspected_kb_node_id`" in prompt
    assert "`stored_lean_proof_excerpt`" in prompt
    assert "`supports_tool_limitation`" in prompt
    assert "`false_model_claims`" in prompt
    assert "set `inspected_kb_node` to the boolean `true` or `false` only" in prompt_lower
    assert "do not include structured index fields" in prompt_lower
    assert "free-text transcript coordinates are allowed" in prompt_lower
    assert "missing proof-term `by`" in prompt_lower
    assert "do not blame block syntax" in prompt_lower
