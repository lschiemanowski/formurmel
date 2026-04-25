from __future__ import annotations

from dataclasses import dataclass

from formurmel.tools.murmel_tool import MurmelTool


@dataclass
class FakeResult:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


class FakeMurmel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def lexical_search(self, *args, **kwargs):
        self.calls.append(("lexical_search", args, kwargs))
        return [
            FakeResult(
                {
                    "name": "Nat.add_comm",
                    "kind": "theorem",
                    "module": "Mathlib.Init.Data.Nat.Basic",
                    "declaration_head": "theorem Nat.add_comm (n m : Nat) : n + m = m + n",
                    "snippet": [{"line": 12, "text": "theorem Nat.add_comm ..."}],
                }
            )
        ]

    def semantic_search(self, *args, **kwargs):
        self.calls.append(("semantic_search", args, kwargs))
        return [
            FakeResult(
                {
                    "name": "Set.image_subset_iff",
                    "kind": "theorem",
                    "module": "Mathlib.Data.Set.Image",
                    "semantic_score": 0.8,
                    "declaration_head": "theorem Set.image_subset_iff ...",
                }
            )
        ]

    def show(self, *args, **kwargs):
        self.calls.append(("show", args, kwargs))
        return FakeResult(
            {
                "name": "Nat.zero_add",
                "kind": "theorem",
                "module": "Mathlib.Init.Data.Nat.Basic",
                "declaration_head": "theorem Nat.zero_add (n : Nat) : 0 + n = n",
            }
        )

    def describe(self, *args, **kwargs):
        self.calls.append(("describe", args, kwargs))
        return FakeResult(
            {
                "name": "Nat.zero_add",
                "kind": "theorem",
                "module": "Mathlib.Init.Data.Nat.Basic",
                "lean_declaration": "theorem Nat.zero_add (n : Nat) : 0 + n = n",
                "natural_language": "Zero is a left identity for natural-number addition.",
            }
        )


def test_murmel_tool_exposes_single_tool_name() -> None:
    tool = MurmelTool(app=FakeMurmel())
    spec = tool.tool_description()

    assert spec.name == "murmel"
    assert "semantic" in spec.description


def test_murmel_lexical_search_serializes_results() -> None:
    app = FakeMurmel()
    tool = MurmelTool(app=app, mathlib_rev="abc123")

    result = tool.execute(
        {
            "action": "search",
            "mode": "lexical",
            "query": "add_comm",
            "limit": 3,
            "regex": False,
            "with_snippet": True,
            "max_lines": 5,
        }
    )

    assert result["ok"] is True
    assert app.calls[0] == (
        "lexical_search",
        ("add_comm",),
        {
            "mathlib_rev": "abc123",
            "regex": False,
            "kind": None,
            "module": None,
            "path_fragment": None,
            "limit": 3,
            "with_snippet": True,
            "max_lines": 5,
        },
    )
    match = result["result"]["matches"][0]
    assert match["lean_declaration"].startswith("theorem Nat.add_comm")
    assert match["snippet_text"] == "12: theorem Nat.add_comm ..."


def test_murmel_semantic_search_uses_top_k_and_device() -> None:
    app = FakeMurmel()
    tool = MurmelTool(app=app, semantic_device="cpu", semantic_score_chunk_size=7)

    result = tool.execute(
        {
            "action": "search",
            "mode": "semantic",
            "query": "image subset iff",
            "limit": 4,
            "kind": "theorem",
        }
    )

    assert result["ok"] is True
    assert app.calls[0] == (
        "semantic_search",
        ("image subset iff",),
        {
            "mathlib_rev": None,
            "top_k": 4,
            "kind": "theorem",
            "module": None,
            "with_snippet": False,
            "max_lines": 80,
            "score_chunk_size": 7,
            "device": "cpu",
        },
    )


def test_murmel_show_and_describe() -> None:
    app = FakeMurmel()
    tool = MurmelTool(app=app)

    show = tool.execute({"action": "show", "name": "Nat.zero_add", "exact": True, "max_lines": 2})
    describe = tool.execute({"action": "describe", "name": "Nat.zero_add"})

    assert show["ok"] is True
    assert show["result"]["declaration"]["lean_declaration"].startswith("theorem Nat.zero_add")
    assert describe["ok"] is True
    assert "left identity" in describe["result"]["declaration"]["natural_language"]
    assert app.calls[0] == ("show", ("Nat.zero_add",), {"mathlib_rev": None, "max_lines": 2, "exact": True})
    assert app.calls[1] == ("describe", ("Nat.zero_add",), {"mathlib_rev": None, "exact": True})


def test_murmel_semantic_search_rejects_lexical_only_fields() -> None:
    result = MurmelTool(app=FakeMurmel()).execute(
        {"action": "search", "mode": "semantic", "query": "addition", "regex": True}
    )

    assert result["ok"] is False
    assert "does not support field 'regex'" in result["error"]
