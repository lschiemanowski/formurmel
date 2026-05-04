from __future__ import annotations

from pathlib import Path

import pytest

from formurmel.kb import KB, KBError
from formurmel.tools.kb_tool import KBTool


def test_kb_tool_cannot_remove_statement_with_protected_text(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    kb = KB.create()
    kb.add_node(
        "target",
        "statement",
        "A target.",
        lean="theorem target : True",
        statement_write_protected=True,
    )
    kb.save(kb_path)
    tool = KBTool(kb_path=kb_path)

    result = tool.execute({"action": "remove_node", "id": "target"})

    assert result["ok"] is False
    assert "statement fields are write-protected" in result["error"]
    assert KB.load(kb_path).get_node("target").id == "target"


def test_kb_tool_can_remove_unprotected_definition(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    kb = KB.create()
    kb.add_node("helper", "definition", "A helper definition.")
    kb.save(kb_path)
    tool = KBTool(kb_path=kb_path)

    result = tool.execute({"action": "remove_node", "id": "helper"})

    assert result["ok"] is True
    assert KB.load(kb_path).list_nodes() == []


def test_kb_tool_rejects_direct_verified_update(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    kb = KB.create()
    kb.add_node("target", "statement", "A target.", lean="theorem target : True")
    kb.save(kb_path)
    tool = KBTool(kb_path=kb_path)

    result = tool.execute({"action": "update_node", "id": "target", "verified": True})

    assert result["ok"] is False
    assert "unknown field(s): verified" in result["error"]
    assert KB.load(kb_path).get_node("target").verified is None


def test_kb_tool_does_not_advertise_derived_status_fields(tmp_path: Path) -> None:
    tool = KBTool(kb_path=tmp_path / "kb.json")

    properties = tool.tool_description().parameters["properties"]

    assert "compiles" not in properties
    assert "error" not in properties
    assert "verified" not in properties


def test_kb_core_rejects_direct_status_updates() -> None:
    kb = KB.create()
    kb.add_node("target", "statement", "A target.", lean="theorem target : True")

    with pytest.raises(KBError, match="Cannot update field 'verified'"):
        kb.update_node("target", verified=True)


def test_completion_state_does_not_solve_proofless_verified_legacy_node() -> None:
    kb = KB.create()
    kb.add_node(
        "target",
        "statement",
        "A target.",
        lean="theorem target : True",
        verified=True,
    )

    state = kb.completion_state()

    assert state["solved_statement_ids"] == []
    assert state["pending_statement_ids"] == ["target"]
