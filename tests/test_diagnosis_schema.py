from __future__ import annotations

import pytest

from formurmel.diagnosis import validate_diagnosis_payload


def valid_payload() -> dict:
    return {
        "failure_type": "proof_format_error",
        "first_critical_mistake": {
            "summary": "The agent stored bare tactic text as a KB proof.",
            "submitted_code_excerpt": 'lean_proof = "by_contra h; intro x hx"',
            "observed_error_excerpt": "Unknown identifier `h`",
            "why_it_failed": "KB appends lean_proof after `:=`, so tactic text must be wrapped in `by`.",
        },
        "root_cause": "The stored proof was not a valid proof term.",
        "evidence": [
            {
                "evidence_type": "kb_compile_error",
                "summary": "compile_node rejected a stored proof that started with a bare tactic.",
                "submitted_code_excerpt": 'lean_proof = "by_contra h; intro x hx"',
                "observed_error_excerpt": "Unknown identifier `h`",
                "interpretation": "The error is caused by invalid proof-term format, not a KB tool bug.",
            }
        ],
        "kb_error_analysis": {
            "inspected_kb_node": True,
            "inspected_kb_node_id": "problem_54",
            "stored_lean_proof_excerpt": "by_contra h; intro x hx",
            "stored_lean_proof_is_valid_proof_term": False,
            "proof_format_summary": "The stored proof is bare tactic text appended after `:=`.",
            "supports_tool_limitation": False,
        },
        "false_model_claims": ["The failed run blamed the KB environment."],
        "tool_or_environment_issues": [],
        "future_strategy": ["Store tactic proofs as proof terms starting with `by`."],
        "minimal_correction": "Store `by\n  by_contra h\n  ...` rather than bare tactic text.",
        "confidence": 0.9,
    }


def test_validate_diagnosis_payload_accepts_strict_schema() -> None:
    payload = valid_payload()

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_rejects_transcript_index_fields_in_first_mistake() -> None:
    payload = valid_payload()
    payload["first_critical_mistake"]["message_index"] = 341

    with pytest.raises(ValueError, match="must not include transcript index"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_rejects_transcript_index_fields_in_evidence() -> None:
    payload = valid_payload()
    payload["evidence"][0]["preceding_tool_call_index"] = 340

    with pytest.raises(ValueError, match="must not include transcript index"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_allows_free_text_index_coordinates() -> None:
    payload = valid_payload()
    payload["first_critical_mistake"]["summary"] = "At index 221, the agent stored bare tactic text."

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_allows_free_text_indices_coordinates() -> None:
    payload = valid_payload()
    payload["evidence"][0]["summary"] = "The same pattern appeared at indices 221, 321, and 328."

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_allows_free_text_message_coordinates() -> None:
    payload = valid_payload()
    payload["false_model_claims"][0] = "Message 411 blamed the KB."

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_allows_non_coordinate_index_language() -> None:
    payload = valid_payload()
    payload["future_strategy"][0] = "For indexed families, still inspect the submitted proof term."

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_requires_transferable_first_mistake_fields() -> None:
    payload = valid_payload()
    del payload["first_critical_mistake"]["why_it_failed"]

    with pytest.raises(ValueError, match="why_it_failed"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_evidence_excerpts() -> None:
    payload = valid_payload()
    payload["evidence"][0]["submitted_code_excerpt"] = None
    payload["evidence"][0]["observed_error_excerpt"] = None

    with pytest.raises(ValueError, match="requires at least one"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_kb_error_analysis() -> None:
    payload = valid_payload()
    del payload["kb_error_analysis"]

    with pytest.raises(ValueError, match="kb_error_analysis"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_kb_tool_limitation_flag() -> None:
    payload = valid_payload()
    del payload["kb_error_analysis"]["supports_tool_limitation"]

    with pytest.raises(ValueError, match="supports_tool_limitation"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_kb_node_id_field() -> None:
    payload = valid_payload()
    del payload["kb_error_analysis"]["inspected_kb_node_id"]

    with pytest.raises(ValueError, match="inspected_kb_node_id"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_boolean_kb_node_inspection_flag() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["inspected_kb_node"] = "problem_54"

    with pytest.raises(ValueError, match="inspected_kb_node must be a boolean"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_requires_id_when_kb_node_was_inspected() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["inspected_kb_node_id"] = None

    with pytest.raises(ValueError, match="inspected_kb_node_id must be a non-empty string"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_accepts_no_inspected_kb_node() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["inspected_kb_node"] = False
    payload["kb_error_analysis"]["inspected_kb_node_id"] = None
    payload["kb_error_analysis"]["stored_lean_proof_excerpt"] = None
    payload["kb_error_analysis"]["stored_lean_proof_is_valid_proof_term"] = None
    payload["kb_error_analysis"]["proof_format_summary"] = "No KB compile error was relevant."

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_rejects_bare_tactic_as_valid_proof_term() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["stored_lean_proof_excerpt"] = '"by_contra h; intro x hx"'
    payload["kb_error_analysis"]["stored_lean_proof_is_valid_proof_term"] = True

    with pytest.raises(ValueError, match="bare tactic text"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_rejects_tool_limitation_for_invalid_proof_term() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["supports_tool_limitation"] = True

    with pytest.raises(ValueError, match="requires stored_lean_proof_is_valid_proof_term"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_accepts_tactic_wrapped_in_proof_term() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["stored_lean_proof_excerpt"] = "by\n  by_contra h\n  exact absurd h (by trivial)"
    payload["kb_error_analysis"]["stored_lean_proof_is_valid_proof_term"] = True
    payload["kb_error_analysis"]["supports_tool_limitation"] = True

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_rejects_parser_blame_for_bare_tactic_proof() -> None:
    payload = valid_payload()
    payload["kb_error_analysis"]["proof_format_summary"] = (
        "The stored proof starts with by_contra h but KB parsing does not support block syntax."
    )

    with pytest.raises(ValueError, match="must not blame"):
        validate_diagnosis_payload(payload)


def test_validate_diagnosis_payload_allows_proof_strategy_diagnosis_for_bare_tactic_proof() -> None:
    payload = valid_payload()
    payload["failure_type"] = "proof_strategy_error"
    payload["root_cause"] = "The agent introduced hypotheses in the wrong order after starting contradiction."
    payload["minimal_correction"] = "After starting contradiction, derive the negated property before using it."
    payload["future_strategy"] = ["Check that every referenced hypothesis has already been introduced."]
    payload["kb_error_analysis"]["proof_format_summary"] = (
        "The stored proof is invalid as a proof term and also has a tactic-order error."
    )

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_allows_negated_kb_parsing_bug_for_bare_tactic_proof() -> None:
    payload = valid_payload()
    payload["evidence"][0]["interpretation"] = (
        "This is not a KB parsing bug; the stored proof is malformed tactic text."
    )

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_allows_code_caused_parsing_failure_for_bare_tactic_proof() -> None:
    payload = valid_payload()
    payload["first_critical_mistake"]["why_it_failed"] = (
        "Without `by`, Lean treats `by_contra h` as term-level syntax, causing parsing failure."
    )

    assert validate_diagnosis_payload(payload) is payload


def test_validate_diagnosis_payload_ignores_false_claims_when_checking_tool_blame() -> None:
    payload = valid_payload()
    payload["false_model_claims"] = ["The KB parser limitation prevented a valid proof from compiling."]

    assert validate_diagnosis_payload(payload) is payload
