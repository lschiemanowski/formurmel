from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formurmel.kb import KB, KBError
from formurmel.message import ToolSpec
from formurmel.tools.base import Tool, tool_error, tool_ok


_NODE_TYPES = {"definition", "statement"}
_SEARCH_FIELDS = {
    "natural_language",
    "lean",
    "lean_name",
    "natural_language_proof",
    "lean_proof",
}
_STATEMENT_ONLY_FIELDS = {
    "statement_write_protected",
    "proof_write_protected",
    "natural_language_proof",
    "lean_proof",
    "verified",
    "issue_kind",
    "issue_explanation",
    "issue_confidence",
    "issue_proposed_natural_language",
    "issue_proposed_lean",
}
_ADD_NODE_OPTIONAL_FIELDS = {
    "lean_name",
    "lean",
    "write_protected",
    "statement_write_protected",
    "proof_write_protected",
    "compiles",
    "error",
    "natural_language_proof",
    "lean_proof",
    "verified",
    "issue_kind",
    "issue_explanation",
    "issue_confidence",
    "issue_proposed_natural_language",
    "issue_proposed_lean",
}
_UPDATE_NODE_MUTABLE_FIELDS = {
    "natural_language",
    "lean_name",
    "lean",
    "compiles",
    "error",
    "natural_language_proof",
    "lean_proof",
    "verified",
    "issue_kind",
    "issue_explanation",
    "issue_confidence",
    "issue_proposed_natural_language",
    "issue_proposed_lean",
}
_ACTION_FIELDS: dict[str, set[str]] = {
    "list_nodes": {"action", "type"},
    "get_node": {"action", "id"},
    "search_nodes": {"action", "query", "field"},
    "get_dependencies": {"action", "id"},
    "get_dependents": {"action", "id"},
    "get_neighborhood": {"action", "id", "depth"},
    "get_context": {"action", "id", "radius"},
    "unproven_statements": {"action"},
    "ready_to_prove": {"action", "id"},
    "report_statement_issue": {
        "action",
        "id",
        "kind",
        "explanation",
        "confidence",
        "proposed_natural_language",
        "proposed_lean",
    },
    "compile_node": {"action", "id", "include_source"},
    "compile_candidate": {"action", "id", "lean_name", "lean", "lean_proof", "include_source"},
    "add_node": {"action", "id", "type", "natural_language", *_ADD_NODE_OPTIONAL_FIELDS},
    "update_node": {"action", "id", *_UPDATE_NODE_MUTABLE_FIELDS},
    "remove_node": {"action", "id"},
    "add_edge": {"action", "source", "target"},
    "remove_edge": {"action", "source", "target"},
}
_NODE_ID_ALIAS_ACTIONS = {
    "get_node",
    "get_dependencies",
    "get_dependents",
    "get_neighborhood",
    "get_context",
    "ready_to_prove",
    "report_statement_issue",
    "compile_node",
    "compile_candidate",
    "add_node",
    "update_node",
    "remove_node",
}
_NODE_ID_ALIASES = ("node", "node_id", "node_name")
_REPORT_ISSUE_FIELD_ALIASES = {
    "issue_kind": "kind",
    "issue_explanation": "explanation",
    "issue_confidence": "confidence",
    "issue_proposed_natural_language": "proposed_natural_language",
    "issue_proposed_lean": "proposed_lean",
}


def _flatten_mapping_fields(
    normalized: dict[str, Any],
    wrapper_keys: tuple[str, ...],
    *,
    field_aliases: Mapping[str, str] | None = None,
) -> None:
    aliases = field_aliases or {}
    for wrapper_key in wrapper_keys:
        nested = normalized.get(wrapper_key)
        if not isinstance(nested, Mapping):
            continue
        for key, value in nested.items():
            normalized_key = aliases.get(str(key), str(key))
            if normalized_key not in normalized:
                normalized[normalized_key] = value
        normalized.pop(wrapper_key, None)


def _normalize_action_payload(payload: Mapping[str, Any], action: str) -> dict[str, Any]:
    normalized = dict(payload)
    if action == "update_node":
        _flatten_mapping_fields(normalized, ("node", "update", "update_fields", "updates"))
        if "statement" in normalized and "natural_language" not in normalized:
            normalized["natural_language"] = normalized["statement"]
        normalized.pop("statement", None)
    elif action == "compile_candidate":
        _flatten_mapping_fields(normalized, ("candidate",))
    elif action == "report_statement_issue":
        for alias, canonical in _REPORT_ISSUE_FIELD_ALIASES.items():
            if canonical not in normalized and alias in normalized:
                normalized[canonical] = normalized[alias]
            normalized.pop(alias, None)
        _flatten_mapping_fields(
            normalized,
            ("issue", "report"),
            field_aliases=_REPORT_ISSUE_FIELD_ALIASES,
        )
    if action in _NODE_ID_ALIAS_ACTIONS:
        if "id" not in normalized:
            for alias in _NODE_ID_ALIASES:
                value = normalized.get(alias)
                if isinstance(value, str) and value.strip():
                    normalized["id"] = value
                    break
        for alias in _NODE_ID_ALIASES:
            normalized.pop(alias, None)
    return normalized


def _require_str(payload: Mapping[str, Any], field: str, action: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KBError(f"kb {action} requires a non-empty string field '{field}'")
    return value.strip()


def _optional_str(payload: Mapping[str, Any], field: str, action: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise KBError(f"kb {action} field '{field}' must be a string")
    return value


def _optional_bool(payload: Mapping[str, Any], field: str, action: str, *, default: bool = False) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise KBError(f"kb {action} field '{field}' must be a boolean")
    return value


def _optional_int(payload: Mapping[str, Any], field: str, action: str, *, default: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KBError(f"kb {action} field '{field}' must be an integer")
    return value


def _optional_number(payload: Mapping[str, Any], field: str, action: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KBError(f"kb {action} field '{field}' must be a number")
    return float(value)


class KBTool(Tool):
    """Read and mutate the working mathematical knowledge base."""

    def __init__(
        self,
        *,
        kb_path: Path | None = None,
        lake_project: Path | None = None,
        autosave: bool = True,
    ) -> None:
        if not isinstance(autosave, bool):
            raise ValueError("autosave must be a boolean")
        self._kb_path = kb_path.expanduser().resolve() if kb_path is not None else None
        resolved_lake_project = lake_project.expanduser().resolve() if lake_project is not None else None
        self._autosave = autosave
        if self._kb_path is not None and self._kb_path.exists():
            self._kb = KB.load(self._kb_path, lake_project=resolved_lake_project)
        else:
            self._kb = KB.create(lake_project=resolved_lake_project)

    @property
    def name(self) -> str:
        return "kb"

    def completion_state(self) -> dict[str, Any]:
        return self._kb.completion_state()

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read and update the working knowledge base. Use `add_node`/`update_node` "
                "to record formalization work, `compile_node` or `compile_candidate` to check "
                "KB-local Lean code, and dependency actions to inspect prerequisites. For statement "
                "nodes, `lean_proof` is appended after `:=`, so it must be a Lean proof term; "
                "tactic-mode proofs must include their own `by`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(_ACTION_FIELDS),
                        "description": "KB operation to perform.",
                    },
                    "id": {"type": "string", "description": "Node id."},
                    "type": {"type": "string", "enum": sorted(_NODE_TYPES), "description": "Node type."},
                    "natural_language": {"type": "string", "description": "Natural-language statement or definition."},
                    "query": {"type": "string", "description": "Text query for search_nodes."},
                    "field": {"type": "string", "enum": sorted(_SEARCH_FIELDS), "description": "Optional search field."},
                    "source": {"type": "string", "description": "Source node id for an edge."},
                    "target": {"type": "string", "description": "Target dependency node id for an edge."},
                    "lean_name": {"type": "string"},
                    "lean": {"type": "string"},
                    "lean_proof": {
                        "type": "string",
                        "description": (
                            "Proof term for a statement node, appended after `:=`. Include `by` for tactic "
                            "proofs; do not provide bare tactic text."
                        ),
                    },
                    "natural_language_proof": {"type": "string"},
                    "include_source": {"type": "boolean"},
                    "depth": {"type": "integer"},
                    "radius": {"type": "integer"},
                    "kind": {"type": "string"},
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number"},
                    "proposed_natural_language": {"type": "string"},
                    "proposed_lean": {"type": "string"},
                    "write_protected": {"type": "boolean"},
                    "statement_write_protected": {"type": "boolean"},
                    "proof_write_protected": {"type": "boolean"},
                    "compiles": {"type": ["boolean", "null"]},
                    "verified": {"type": ["boolean", "null"]},
                    "error": {"type": ["string", "null"]},
                    "issue_kind": {"type": ["string", "null"]},
                    "issue_explanation": {"type": ["string", "null"]},
                    "issue_confidence": {"type": ["number", "null"]},
                    "issue_proposed_natural_language": {"type": ["string", "null"]},
                    "issue_proposed_lean": {"type": ["string", "null"]},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return tool_error("kb payload must be an object")

        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return tool_error("kb field 'action' must be a non-empty string")
        action = action.strip()
        payload = _normalize_action_payload(payload, action)

        allowed_fields = _ACTION_FIELDS.get(action)
        if allowed_fields is None:
            return tool_error(f"unknown kb action '{action}'")
        unknown_fields = sorted(str(key) for key in payload.keys() if key not in allowed_fields)
        if unknown_fields:
            return tool_error(f"kb {action} payload contains unknown field(s): {', '.join(unknown_fields)}")

        try:
            return self._execute_checked(action, payload)
        except (KBError, OSError, ValueError) as exc:
            return tool_error(str(exc))

    def _execute_checked(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if action == "list_nodes":
            node_type = payload.get("type")
            if node_type is not None and (not isinstance(node_type, str) or node_type not in _NODE_TYPES):
                raise KBError("kb list_nodes field 'type' must be definition or statement")
            return tool_ok({"action": action, "nodes": self._kb.list_nodes(type=node_type)})

        if action == "get_node":
            node_id = _require_str(payload, "id", action)
            return tool_ok({"action": action, "node": self._kb.get_node(node_id).to_dict()})

        if action == "search_nodes":
            query = _require_str(payload, "query", action)
            field = payload.get("field")
            if field is not None and (not isinstance(field, str) or field not in _SEARCH_FIELDS):
                raise KBError(
                    "kb search_nodes field 'field' must be one of "
                    "natural_language, lean, lean_name, natural_language_proof, lean_proof"
                )
            return tool_ok({"action": action, "matches": self._kb.search_nodes(query=query, field=field)})

        if action == "get_dependencies":
            node_id = _require_str(payload, "id", action)
            return tool_ok({"action": action, "id": node_id, "dependencies": self._kb.get_dependencies(node_id)})

        if action == "get_dependents":
            node_id = _require_str(payload, "id", action)
            return tool_ok({"action": action, "id": node_id, "dependents": self._kb.get_dependents(node_id)})

        if action == "get_neighborhood":
            node_id = _require_str(payload, "id", action)
            depth = _optional_int(payload, "depth", action, default=1)
            return tool_ok({"action": action, "neighborhood": self._kb.get_neighborhood(node_id, depth=depth)})

        if action == "get_context":
            node_id = _require_str(payload, "id", action)
            radius = _optional_int(payload, "radius", action, default=2)
            return tool_ok({"action": action, "id": node_id, "context": self._kb.get_context(node_id, radius=radius)})

        if action == "unproven_statements":
            return tool_ok({"action": action, "statements": self._kb.unproven_statements()})

        if action == "ready_to_prove":
            node_id = _require_str(payload, "id", action)
            return tool_ok({"action": action, "id": node_id, "ready": self._kb.ready_to_prove(node_id)})

        if action == "report_statement_issue":
            node_id = _require_str(payload, "id", action)
            kind = _require_str(payload, "kind", action)
            explanation = _require_str(payload, "explanation", action)
            issue = self._kb.report_statement_issue(
                node_id,
                kind=kind,
                explanation=explanation,
                confidence=_optional_number(payload, "confidence", action),
                proposed_natural_language=_optional_str(payload, "proposed_natural_language", action),
                proposed_lean=_optional_str(payload, "proposed_lean", action),
            )
            self._persist()
            return tool_ok({"action": action, "issue": issue})

        if action == "compile_node":
            node_id = _require_str(payload, "id", action)
            compilation = self._kb.compile_node(
                node_id,
                include_source=_optional_bool(payload, "include_source", action, default=False),
            )
            self._persist()
            return tool_ok({"action": action, "id": node_id, "compilation": compilation})

        if action == "compile_candidate":
            node_id = _require_str(payload, "id", action)
            compilation = self._kb.compile_candidate(
                node_id,
                lean_name=_optional_str(payload, "lean_name", action),
                lean=_optional_str(payload, "lean", action),
                lean_proof=_optional_str(payload, "lean_proof", action),
                include_source=_optional_bool(payload, "include_source", action, default=False),
            )
            return tool_ok({"action": action, "id": node_id, "compilation": compilation})

        if action == "add_node":
            result = self._add_node(payload, action)
            self._persist()
            return result

        if action == "update_node":
            result = self._update_node(payload, action)
            self._persist()
            return result

        if action == "remove_node":
            node_id = _require_str(payload, "id", action)
            self._kb.remove_node(node_id)
            self._persist()
            return tool_ok({"action": action, "id": node_id, "removed": True})

        if action == "add_edge":
            source = _require_str(payload, "source", action)
            target = _require_str(payload, "target", action)
            edge = self._kb.add_edge(source, target)
            self._persist()
            return tool_ok({"action": action, "edge": edge.to_dict()})

        source = _require_str(payload, "source", action)
        target = _require_str(payload, "target", action)
        self._kb.remove_edge(source, target)
        self._persist()
        return tool_ok({"action": action, "source": source, "target": target, "removed": True})

    def _add_node(self, payload: Mapping[str, Any], action: str) -> dict[str, Any]:
        node_id = _require_str(payload, "id", action)
        node_type = payload.get("type")
        if not isinstance(node_type, str) or node_type not in _NODE_TYPES:
            raise KBError("kb add_node field 'type' must be definition or statement")
        natural_language = _require_str(payload, "natural_language", action)
        if node_type == "definition":
            invalid_statement_fields = sorted(field for field in _STATEMENT_ONLY_FIELDS if field in payload)
            if invalid_statement_fields:
                raise KBError(
                    "kb add_node for definitions does not accept statement-only "
                    f"field(s): {', '.join(invalid_statement_fields)}"
                )

        kwargs: dict[str, Any] = {}
        for field in _ADD_NODE_OPTIONAL_FIELDS:
            if field not in payload:
                continue
            value = payload[field]
            if field in {"write_protected", "statement_write_protected", "proof_write_protected"}:
                if not isinstance(value, bool):
                    raise KBError(f"kb add_node field '{field}' must be a boolean")
            elif field in {"compiles", "verified"}:
                if value is not None and not isinstance(value, bool):
                    raise KBError(f"kb add_node field '{field}' must be a boolean or null")
            elif field == "issue_confidence":
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise KBError("kb add_node field 'issue_confidence' must be a number or null")
                    value = float(value)
            else:
                if value is not None and not isinstance(value, str):
                    raise KBError(f"kb add_node field '{field}' must be a string or null")
            kwargs[field] = value

        node = self._kb.add_node(id=node_id, type=node_type, natural_language=natural_language, **kwargs)
        return tool_ok({"action": action, "node": node.to_dict()})

    def _update_node(self, payload: Mapping[str, Any], action: str) -> dict[str, Any]:
        node_id = _require_str(payload, "id", action)
        node = self._kb.get_node(node_id)
        fields = {key: value for key, value in payload.items() if key not in {"action", "id"}}
        if not fields:
            raise KBError("kb update_node requires at least one mutable field")

        if node.type.value == "definition":
            invalid_statement_fields = sorted(field for field in _STATEMENT_ONLY_FIELDS if field in fields)
            if invalid_statement_fields:
                raise KBError(
                    "kb update_node for definitions does not accept statement-only "
                    f"field(s): {', '.join(invalid_statement_fields)}"
                )

        for field, value in list(fields.items()):
            if field not in _UPDATE_NODE_MUTABLE_FIELDS:
                raise KBError(f"kb update_node field '{field}' is not mutable")
            if field == "natural_language":
                if not isinstance(value, str) or not value.strip():
                    raise KBError("kb update_node field 'natural_language' must be a non-empty string")
                fields[field] = value.strip()
            elif field in {"compiles", "verified"}:
                if value is not None and not isinstance(value, bool):
                    raise KBError(f"kb update_node field '{field}' must be a boolean or null")
            elif field == "issue_confidence":
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise KBError("kb update_node field 'issue_confidence' must be a number or null")
                    fields[field] = float(value)
            else:
                if value is not None and not isinstance(value, str):
                    raise KBError(f"kb update_node field '{field}' must be a string or null")

        updated = self._kb.update_node(node_id, **fields)
        return tool_ok({"action": action, "node": updated.to_dict()})

    def _persist(self) -> None:
        if self._autosave and self._kb_path is not None:
            self._kb.save(self._kb_path)
