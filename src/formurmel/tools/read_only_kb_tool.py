from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from formurmel.message import ToolSpec
from formurmel.tools.base import Tool, tool_error


READ_ONLY_KB_ACTIONS = {
    "list_nodes",
    "get_node",
    "search_nodes",
    "get_dependencies",
    "get_dependents",
    "get_neighborhood",
    "get_context",
    "unproven_statements",
    "ready_to_prove",
    "compile_candidate",
}


class ReadOnlyKBTool(Tool):
    """Expose the KB under the normal name while blocking persistent mutations."""

    def __init__(self, wrapped: Tool) -> None:
        if wrapped.name != "kb":
            raise ValueError("ReadOnlyKBTool can only wrap the kb tool")
        self._wrapped = wrapped

    @property
    def name(self) -> str:
        return "kb"

    def tool_description(self) -> ToolSpec:
        spec = self._wrapped.tool_description()
        parameters = deepcopy(dict(spec.parameters))
        action = parameters.get("properties", {}).get("action")
        if isinstance(action, dict):
            action["enum"] = sorted(READ_ONLY_KB_ACTIONS)
            action["description"] = "Read-only KB operation to perform during diagnosis."
        return ToolSpec(
            name=spec.name,
            description=(
                "Read-only diagnosis view of the knowledge base. Mutating actions and compile_node are disabled. "
                "Use get_node/search actions to inspect stored data and compile_candidate for non-persistent checks."
            ),
            parameters=parameters,
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return tool_error("kb payload must be an object")
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return tool_error("kb field 'action' must be a non-empty string")
        action = action.strip()
        if action not in READ_ONLY_KB_ACTIONS:
            return tool_error(
                f"kb is read-only during diagnosis; action '{action}' is disabled. "
                "Use get_node/search actions, compile_candidate, or transcript_inspect instead."
            )
        return self._wrapped.execute(payload)

    def completion_state(self) -> dict[str, Any]:
        completion_state = getattr(self._wrapped, "completion_state", None)
        if callable(completion_state):
            return completion_state()
        return {}
