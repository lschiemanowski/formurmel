from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DEFINITION = "definition"
    STATEMENT = "statement"


@dataclass
class Node:
    id: str
    type: NodeType
    natural_language: str
    lean_name: str | None = None
    lean: str | None = None
    write_protected: bool = False
    statement_write_protected: bool = False
    proof_write_protected: bool = False
    compiles: bool | None = None
    error: str | None = None
    natural_language_proof: str | None = None
    lean_proof: str | None = None
    verified: bool | None = None
    issue_kind: str | None = None
    issue_explanation: str | None = None
    issue_confidence: float | None = None
    issue_proposed_natural_language: str | None = None
    issue_proposed_lean: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        if self.type == NodeType.DEFINITION:
            for key in (
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
            ):
                del payload[key]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Node":
        data = dict(payload)
        data["type"] = NodeType(data["type"])
        return cls(**data)


@dataclass
class Edge:
    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "Edge":
        return cls(**payload)


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    text_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "text_order": list(self.text_order),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Graph":
        return cls(
            nodes={node_id: Node.from_dict(node) for node_id, node in payload.get("nodes", {}).items()},
            edges=[Edge.from_dict(edge) for edge in payload.get("edges", [])],
            text_order=list(payload.get("text_order", [])),
        )

