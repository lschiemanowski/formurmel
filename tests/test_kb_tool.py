from __future__ import annotations

from pathlib import Path

from formurmel.kb import KB
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
