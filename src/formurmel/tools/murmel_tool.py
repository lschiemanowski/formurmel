from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from formurmel.message import ToolSpec
from formurmel.tools.base import Tool, tool_error, tool_ok


_ACTION_FIELDS: dict[str, set[str]] = {
    "search": {
        "action",
        "mode",
        "query",
        "limit",
        "regex",
        "kind",
        "module",
        "path_fragment",
        "with_snippet",
        "max_lines",
    },
    "show": {"action", "name", "max_lines", "exact"},
    "describe": {"action", "name", "exact"},
}


def _load_murmel_api() -> tuple[type[Any], type[BaseException]]:
    try:
        from murmel import Murmel
        from murmel.errors import MurmelError

        return Murmel, MurmelError
    except ModuleNotFoundError as exc:
        if exc.name != "murmel":
            raise

    sibling_src = Path(__file__).resolve().parents[4] / "murmel" / "src"
    if sibling_src.exists() and str(sibling_src) not in sys.path:
        sys.path.insert(0, str(sibling_src))

    from murmel import Murmel
    from murmel.errors import MurmelError

    return Murmel, MurmelError


def _require_str(payload: Mapping[str, Any], field: str, action: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"murmel {action} requires a non-empty string field '{field}'")
    return value.strip()


def _optional_str(payload: Mapping[str, Any], field: str, action: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"murmel {action} field '{field}' must be a string")
    value = value.strip()
    return value or None


def _optional_bool(payload: Mapping[str, Any], field: str, action: str, *, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"murmel {action} field '{field}' must be a boolean")
    return value


def _optional_int(
    payload: Mapping[str, Any],
    field: str,
    action: str,
    *,
    default: int,
    minimum: int = 1,
) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"murmel {action} field '{field}' must be an integer")
    if value < minimum:
        raise ValueError(f"murmel {action} field '{field}' must be >= {minimum}")
    return value


def _object_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        data = value.to_dict()
    elif isinstance(value, Mapping):
        data = dict(value)
    elif is_dataclass(value):
        data = asdict(value)
    else:
        raise TypeError(f"cannot serialize murmel result of type {type(value).__name__}")
    if not isinstance(data, Mapping):
        raise TypeError("murmel result serializer did not return an object")
    return dict(data)


def _attach_model_friendly_fields(data: dict[str, Any]) -> dict[str, Any]:
    declaration_head = data.get("declaration_head")
    if isinstance(declaration_head, str) and declaration_head.strip():
        data.setdefault("lean_declaration", declaration_head.strip())

    snippet = data.get("snippet")
    if isinstance(snippet, list):
        lines: list[str] = []
        for item in snippet:
            if not isinstance(item, Mapping):
                continue
            line_number = item.get("line")
            text = item.get("text")
            if isinstance(line_number, int) and isinstance(text, str):
                lines.append(f"{line_number}: {text}")
        if lines:
            data["snippet_text"] = "\n".join(lines)
    return data


def _serialize_result(value: Any) -> dict[str, Any]:
    return _attach_model_friendly_fields(_object_to_dict(value))


class MurmelTool(Tool):
    """Search and inspect mathlib declarations through murmel."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        config_path: Path | None = None,
        mathlib_rev: str | None = None,
        semantic_device: str | None = None,
        semantic_score_chunk_size: int = 16384,
        app: Any | None = None,
    ) -> None:
        if semantic_score_chunk_size <= 0:
            raise ValueError("semantic_score_chunk_size must be positive")
        self._cache_dir = cache_dir.expanduser().resolve() if cache_dir is not None else None
        self._config_path = config_path.expanduser().resolve() if config_path is not None else None
        self._mathlib_rev = mathlib_rev
        self._semantic_device = semantic_device
        self._semantic_score_chunk_size = int(semantic_score_chunk_size)
        self._app = app

    @property
    def name(self) -> str:
        return "murmel"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Search and inspect mathlib through murmel. Use `search` with mode `semantic` "
                "for meaning-based retrieval, `search` with mode `lexical` for names/text, "
                "`show` for the Lean source of one declaration, and `describe` for murmel's "
                "natural-language declaration description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": sorted(_ACTION_FIELDS)},
                    "mode": {
                        "type": "string",
                        "enum": ["lexical", "semantic"],
                        "description": "Required for action=search.",
                    },
                    "query": {"type": "string", "description": "Search query."},
                    "name": {"type": "string", "description": "Declaration name for show/describe."},
                    "limit": {"type": "integer", "description": "Maximum search results."},
                    "regex": {"type": "boolean", "description": "Lexical search treats query as a regex."},
                    "kind": {"type": "string", "description": "Optional declaration kind filter."},
                    "module": {"type": "string", "description": "Optional module filter."},
                    "path_fragment": {"type": "string", "description": "Optional lexical file-path filter."},
                    "with_snippet": {"type": "boolean", "description": "Include source snippets in search results."},
                    "max_lines": {"type": "integer", "description": "Maximum snippet/source lines."},
                    "exact": {"type": "boolean", "description": "Require exact declaration name for show/describe."},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return tool_error("murmel payload must be an object")
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return tool_error("murmel field 'action' must be a non-empty string")
        action = action.strip()

        allowed_fields = _ACTION_FIELDS.get(action)
        if allowed_fields is None:
            return tool_error(f"unknown murmel action '{action}'")
        unknown_fields = sorted(str(key) for key in payload.keys() if key not in allowed_fields)
        if unknown_fields:
            return tool_error(f"murmel {action} payload contains unknown field(s): {', '.join(unknown_fields)}")

        try:
            return self._execute_checked(action, payload)
        except Exception as exc:  # noqa: BLE001
            return tool_error(str(exc))

    def _execute_checked(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if action == "search":
            return self._search(payload)
        if action == "show":
            name = _require_str(payload, "name", action)
            result = self._app_instance().show(
                name,
                mathlib_rev=self._mathlib_rev,
                max_lines=_optional_int(payload, "max_lines", action, default=80),
                exact=_optional_bool(payload, "exact", action, default=False),
            )
            return tool_ok({"action": action, "declaration": _serialize_result(result)})
        if action == "describe":
            name = _require_str(payload, "name", action)
            result = self._app_instance().describe(
                name,
                mathlib_rev=self._mathlib_rev,
                exact=_optional_bool(payload, "exact", action, default=True),
            )
            return tool_ok({"action": action, "declaration": _serialize_result(result)})
        raise ValueError(f"unknown murmel action '{action}'")

    def _search(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = "search"
        query = _require_str(payload, "query", action)
        mode = _require_str(payload, "mode", action)
        if mode not in {"lexical", "semantic"}:
            raise ValueError("murmel search field 'mode' must be lexical or semantic")
        limit = _optional_int(payload, "limit", action, default=10 if mode == "semantic" else 20)
        with_snippet = _optional_bool(payload, "with_snippet", action, default=False)
        max_lines = _optional_int(payload, "max_lines", action, default=80)
        kind = _optional_str(payload, "kind", action)
        module = _optional_str(payload, "module", action)

        if mode == "lexical":
            results = self._app_instance().lexical_search(
                query,
                mathlib_rev=self._mathlib_rev,
                regex=_optional_bool(payload, "regex", action, default=False),
                kind=kind,
                module=module,
                path_fragment=_optional_str(payload, "path_fragment", action),
                limit=limit,
                with_snippet=with_snippet,
                max_lines=max_lines,
            )
        else:
            if "regex" in payload:
                raise ValueError("murmel semantic search does not support field 'regex'")
            if "path_fragment" in payload:
                raise ValueError("murmel semantic search does not support field 'path_fragment'")
            results = self._app_instance().semantic_search(
                query,
                mathlib_rev=self._mathlib_rev,
                top_k=limit,
                kind=kind,
                module=module,
                with_snippet=with_snippet,
                max_lines=max_lines,
                score_chunk_size=self._semantic_score_chunk_size,
                device=self._semantic_device,
            )

        return tool_ok(
            {
                "action": action,
                "mode": mode,
                "matches": [_serialize_result(result) for result in results],
            }
        )

    def _app_instance(self) -> Any:
        if self._app is None:
            Murmel, _ = _load_murmel_api()
            self._app = Murmel(cache_dir=self._cache_dir, config_path=self._config_path)
        return self._app
