from __future__ import annotations

from pathlib import Path


def test_formalizer_system_prompt_explains_kb_proof_format() -> None:
    prompt = Path("prompts/system_prompt.md").read_text(encoding="utf-8")

    assert "KB proof format:" in prompt
    assert "lean_proof` as the proof term appended after `:=`" in prompt
    assert "If you want tactic mode, include the `by` yourself" in prompt
    assert "Do not store bare tactic text" in prompt
