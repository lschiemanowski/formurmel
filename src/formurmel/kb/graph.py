from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from formurmel.kb.model import Edge, Graph, Node, NodeType


class KBError(Exception):
    pass


_PLACEHOLDER_TOKEN_RE = re.compile(r"\b(?:sorry|admit)\b", re.IGNORECASE)


def _strip_lean_comments_and_strings(text: str) -> str:
    out: list[str] = []
    index = 0
    length = len(text)
    block_depth = 0
    in_line_comment = False
    in_string = False
    while index < length:
        ch = text[index]
        nxt = text[index + 1] if index + 1 < length else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            index += 1
            continue
        if block_depth > 0:
            if ch == "/" and nxt == "-":
                block_depth += 1
                index += 2
                continue
            if ch == "-" and nxt == "/":
                block_depth -= 1
                index += 2
                continue
            if ch == "\n":
                out.append(ch)
            index += 1
            continue
        if in_string:
            if ch == "\\" and index + 1 < length:
                index += 2
                continue
            if ch == "\"":
                in_string = False
            index += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            index += 2
            continue
        if ch == "/" and nxt == "-":
            block_depth = 1
            index += 2
            continue
        if ch == "\"":
            in_string = True
            index += 1
            continue
        out.append(ch)
        index += 1
    return "".join(out)


def _contains_placeholder(*texts: str) -> bool:
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        if "uses 'sorry'" in lowered or "uses sorry" in lowered:
            return True
        if _PLACEHOLDER_TOKEN_RE.search(_strip_lean_comments_and_strings(text)):
            return True
    return False


class KB:
    def __init__(self, graph: Graph | None = None, lake_project: str | Path | None = None) -> None:
        self._graph = graph or Graph()
        self._lake_project = Path(lake_project).expanduser().resolve() if lake_project else None

    @classmethod
    def create(cls, lake_project: str | Path | None = None) -> "KB":
        return cls(lake_project=lake_project)

    @classmethod
    def load(cls, path: str | Path, lake_project: str | Path | None = None) -> "KB":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(Graph.from_dict(data), lake_project=lake_project)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._graph.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def add_node(self, id: str, type: NodeType | str, natural_language: str, **kwargs: Any) -> Node:
        if id in self._graph.nodes:
            raise KBError(f"Node '{id}' already exists")
        if isinstance(type, str):
            try:
                type = NodeType(type)
            except ValueError as exc:
                raise KBError(f"Unknown node type '{type}'") from exc
        lean_name = kwargs.get("lean_name")
        if lean_name is not None:
            self._ensure_unique_lean_name(lean_name)
        node = Node(id=id, type=type, natural_language=natural_language, **kwargs)
        self._graph.nodes[id] = node
        self._graph.text_order.append(id)
        return node

    def get_node(self, id: str) -> Node:
        try:
            return self._graph.nodes[id]
        except KeyError as exc:
            raise KBError(f"Node '{id}' not found") from exc

    def update_node(self, node_id: str, **fields: Any) -> Node:
        node = self.get_node(node_id)
        if node.write_protected:
            raise KBError(f"Node '{node_id}' is write-protected")
        disallowed = {
            "id",
            "type",
            "write_protected",
            "statement_write_protected",
            "proof_write_protected",
            "compiles",
            "error",
            "verified",
        }
        statement_fields = {"natural_language", "lean_name", "lean"}
        proof_fields = {"natural_language_proof", "lean_proof"}
        updated_fields: set[str] = set()
        if "lean_name" in fields and fields["lean_name"] is not None:
            self._ensure_unique_lean_name(fields["lean_name"], exclude_id=node_id)
        for key, value in fields.items():
            if key in disallowed:
                raise KBError(f"Cannot update field '{key}'")
            if not hasattr(node, key):
                raise KBError(f"Unknown field '{key}'")
            if node.type == NodeType.STATEMENT and node.statement_write_protected and key in statement_fields:
                raise KBError(f"Node '{node_id}' statement fields are write-protected")
            if node.type == NodeType.STATEMENT and node.proof_write_protected and key in proof_fields:
                raise KBError(f"Node '{node_id}' proof fields are write-protected")
            setattr(node, key, value)
            updated_fields.add(key)
        if "lean" in updated_fields or "lean_proof" in updated_fields:
            if "compiles" not in updated_fields:
                node.compiles = None
            if "error" not in updated_fields:
                node.error = None
            if node.type == NodeType.STATEMENT and "verified" not in updated_fields:
                node.verified = None
        return node

    def remove_node(self, id: str) -> None:
        node = self.get_node(id)
        if node.write_protected:
            raise KBError(f"Node '{id}' is write-protected")
        if node.type == NodeType.STATEMENT and node.statement_write_protected:
            raise KBError(f"Node '{id}' statement fields are write-protected")
        del self._graph.nodes[id]
        self._graph.edges = [edge for edge in self._graph.edges if edge.source != id and edge.target != id]
        self._graph.text_order = [node_id for node_id in self._graph.text_order if node_id != id]

    def add_edge(self, source: str, target: str) -> Edge:
        if source not in self._graph.nodes:
            raise KBError(f"Node '{source}' not found")
        if target not in self._graph.nodes:
            raise KBError(f"Node '{target}' not found")
        for edge in self._graph.edges:
            if edge.source == source and edge.target == target:
                raise KBError(f"Edge '{source}' -> '{target}' already exists")
        if self._would_create_cycle(source, target):
            raise KBError(f"Edge '{source}' -> '{target}' would create a cycle")
        edge = Edge(source=source, target=target)
        self._graph.edges.append(edge)
        return edge

    def remove_edge(self, source: str, target: str) -> None:
        for index, edge in enumerate(self._graph.edges):
            if edge.source == source and edge.target == target:
                self._graph.edges.pop(index)
                return
        raise KBError(f"Edge '{source}' -> '{target}' not found")

    def get_dependencies(self, id: str) -> list[str]:
        self.get_node(id)
        return [edge.target for edge in self._graph.edges if edge.source == id]

    def get_dependents(self, id: str) -> list[str]:
        self.get_node(id)
        return [edge.source for edge in self._graph.edges if edge.target == id]

    def unproven_statements(self) -> list[str]:
        return [
            node_id
            for node_id in self._graph.text_order
            if node_id in self._graph.nodes
            and self._graph.nodes[node_id].type == NodeType.STATEMENT
            and self._graph.nodes[node_id].lean_proof is None
        ]

    def ready_to_prove(self, id: str) -> bool:
        node = self.get_node(id)
        if node.type != NodeType.STATEMENT:
            raise KBError(f"Node '{id}' is not a statement")
        for dependency_id in self.get_dependencies(id):
            dependency = self._graph.nodes[dependency_id]
            if dependency.type == NodeType.DEFINITION and not dependency.compiles:
                return False
            if dependency.type == NodeType.STATEMENT and not dependency.verified:
                return False
        return True

    def report_statement_issue(
        self,
        id: str,
        *,
        kind: str,
        explanation: str,
        confidence: float | None = None,
        proposed_natural_language: str | None = None,
        proposed_lean: str | None = None,
    ) -> dict[str, Any]:
        node = self.get_node(id)
        if node.type != NodeType.STATEMENT:
            raise KBError(f"Node '{id}' is not a statement")
        normalized_kind = kind.strip().lower()
        if normalized_kind != "statement_wrong":
            raise KBError("Unsupported issue kind. Allowed kinds: statement_wrong")
        normalized_explanation = explanation.strip()
        if not normalized_explanation:
            raise KBError("Issue explanation must be a non-empty string")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise KBError("Issue confidence must be in [0, 1]")
        node.issue_kind = normalized_kind
        node.issue_explanation = normalized_explanation
        node.issue_confidence = confidence
        node.issue_proposed_natural_language = proposed_natural_language.strip() if proposed_natural_language else None
        node.issue_proposed_lean = proposed_lean.strip() if proposed_lean else None
        return {
            "id": node.id,
            "issue_kind": node.issue_kind,
            "issue_explanation": node.issue_explanation,
            "issue_confidence": node.issue_confidence,
            "issue_proposed_natural_language": node.issue_proposed_natural_language,
            "issue_proposed_lean": node.issue_proposed_lean,
        }

    def search_nodes(self, query: str, field: str | None = None) -> list[str]:
        normalized_query = query.strip().lower()
        if not normalized_query:
            raise KBError("Query must be a non-empty string")
        searchable_fields = ("natural_language", "lean", "lean_name", "natural_language_proof", "lean_proof")
        if field is not None and field not in searchable_fields:
            raise KBError(f"Unknown search field '{field}'")
        fields_to_search = (field,) if field is not None else searchable_fields
        results: list[str] = []
        for node_id in self._graph.text_order:
            node = self._graph.nodes[node_id]
            for field_name in fields_to_search:
                value = getattr(node, field_name, None)
                if value is not None and normalized_query in value.lower():
                    results.append(node_id)
                    break
        return results

    def list_nodes(self, type: NodeType | str | None = None) -> list[dict[str, Any]]:
        if isinstance(type, str):
            try:
                type = NodeType(type)
            except ValueError as exc:
                raise KBError(f"Unknown node type '{type}'") from exc
        summaries: list[dict[str, Any]] = []
        for node_id in self._graph.text_order:
            node = self._graph.nodes[node_id]
            if type is not None and node.type != type:
                continue
            summary: dict[str, Any] = {
                "id": node.id,
                "type": node.type.value,
                "lean_name": node.lean_name,
                "natural_language": node.natural_language[:120],
                "has_lean": node.lean is not None,
                "compiles": node.compiles,
            }
            if node.type == NodeType.STATEMENT:
                summary["has_proof"] = node.lean_proof is not None
                summary["verified"] = node.verified
            summaries.append(summary)
        return summaries

    def get_neighborhood(self, id: str, depth: int = 1) -> dict[str, Any]:
        self.get_node(id)
        if depth < 0:
            raise KBError("Depth must be non-negative")

        def collect(start: str, direction: str, max_depth: int) -> list[str]:
            visited: set[str] = set()
            frontier = [start]
            for _ in range(max_depth):
                next_frontier: list[str] = []
                for node_id in frontier:
                    neighbors = self.get_dependencies(node_id) if direction == "dependencies" else self.get_dependents(node_id)
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor != start:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier
            return sorted(visited)

        return {
            "node": self.get_node(id).to_dict(),
            "dependencies": collect(id, "dependencies", depth),
            "dependents": collect(id, "dependents", depth),
        }

    def get_context(self, id: str, radius: int = 2) -> list[str]:
        self.get_node(id)
        if radius < 0:
            raise KBError("Radius must be non-negative")
        try:
            index = self._graph.text_order.index(id)
        except ValueError:
            return [id]
        return self._graph.text_order[max(0, index - radius) : min(len(self._graph.text_order), index + radius + 1)]

    def build_lean_source(self, id: str) -> str:
        node = self.get_node(id)
        return self._build_lean_source_for_node(id, node)

    def build_candidate_lean_source(
        self,
        id: str,
        *,
        lean_name: str | None = None,
        lean: str | None = None,
        lean_proof: str | None = None,
    ) -> str:
        candidate = self._build_candidate_node(id, lean_name=lean_name, lean=lean, lean_proof=lean_proof)
        return self._build_lean_source_for_node(id, candidate)

    def compile_node(self, id: str, *, include_source: bool = False) -> dict[str, Any]:
        node = self.get_node(id)
        lean_source = self.build_lean_source(id)
        compilation = self._compile_lean_source(lean_source, include_source=include_source)
        node.compiles = compilation["success"]
        node.error = None if compilation["success"] else compilation["error"]
        if node.type == NodeType.STATEMENT:
            node.verified = compilation["success"] if node.lean_proof is not None else None
        return compilation

    def compile_candidate(self, id: str, *, lean_name: str | None = None, lean: str | None = None, lean_proof: str | None = None, include_source: bool = False) -> dict[str, Any]:
        lean_source = self.build_candidate_lean_source(id, lean_name=lean_name, lean=lean, lean_proof=lean_proof)
        return self._compile_lean_source(lean_source, include_source=include_source)

    def completion_state(self) -> dict[str, Any]:
        statements: list[dict[str, Any]] = []
        pending_statement_ids: list[str] = []
        solved_statement_ids: list[str] = []
        issue_reported_ids: list[str] = []
        for node_id in self._graph.text_order:
            node = self._graph.nodes[node_id]
            if node.type != NodeType.STATEMENT:
                continue
            entry = {
                "id": node.id,
                "proof_recorded": node.lean_proof is not None,
                "compiles": node.compiles,
                "verified": node.verified,
                "issue_kind": node.issue_kind,
            }
            statements.append(entry)
            if node.verified and node.lean_proof is not None:
                solved_statement_ids.append(node.id)
            elif node.issue_kind is not None:
                issue_reported_ids.append(node.id)
            else:
                pending_statement_ids.append(node.id)
        return {
            "statement_count": len(statements),
            "statements": statements,
            "pending_statement_ids": pending_statement_ids,
            "solved_statement_ids": solved_statement_ids,
            "issue_reported_ids": issue_reported_ids,
        }

    def _build_lean_source_for_node(self, id: str, node: Node) -> str:
        if node.lean is None:
            raise KBError(f"Node '{id}' has no Lean code to compile")
        lines = ["import Mathlib", "", "noncomputable section", ""]
        for dependency_id in self._transitive_deps(id):
            dependency = self._graph.nodes[dependency_id]
            if dependency.lean is not None:
                lines.append(self._node_to_lean(dependency))
                lines.append("")
        lines.append(self._node_to_lean(node))
        lines.append("")
        return "\n".join(lines)

    def _compile_lean_source(self, lean_source: str, *, include_source: bool = False) -> dict[str, Any]:
        if self._lake_project is None:
            raise KBError("KB compilation requires a configured lake_project")
        if not self._lake_project.exists():
            raise KBError(f"configured lake_project does not exist: {self._lake_project}")
        if not self._lake_project.is_dir():
            raise KBError(f"configured lake_project is not a directory: {self._lake_project}")
        with tempfile.TemporaryDirectory(prefix="kb-compile-") as workdir:
            source_path = Path(workdir) / "Main.lean"
            source_path.write_text(lean_source, encoding="utf-8")
            try:
                process = subprocess.run(
                    ["lake", "env", "lean", str(source_path)],
                    cwd=self._lake_project,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise KBError(f"Lean or Lake executable not available: {exc}") from exc
            except Exception as exc:
                raise KBError(f"failed to run Lean: {exc}") from exc
        stdout_text = (process.stdout or "").strip()
        stderr_text = (process.stderr or "").strip()
        combined_output = "\n".join(part for part in (stdout_text, stderr_text) if part)
        uses_sorry = _contains_placeholder(lean_source, combined_output)
        success = process.returncode == 0 and not uses_sorry
        primary_error = None if success else "Lean compilation uses sorry" if uses_sorry else stderr_text or stdout_text or "Lean compilation failed"
        result: dict[str, Any] = {
            "success": success,
            "error": primary_error,
            "exit_code": process.returncode,
            "stderr": stderr_text,
            "stdout": stdout_text,
            "uses_sorry": uses_sorry,
        }
        if include_source:
            result["lean_source"] = lean_source
        return result

    def _build_candidate_node(self, id: str, *, lean_name: str | None = None, lean: str | None = None, lean_proof: str | None = None) -> Node:
        node = self.get_node(id)
        candidate = replace(node)
        if lean_name is not None:
            candidate.lean_name = lean_name
        if lean is not None:
            candidate.lean = lean
        if lean_proof is not None:
            if candidate.type != NodeType.STATEMENT:
                raise KBError("Only statement nodes support lean_proof candidates")
            candidate.lean_proof = lean_proof
        return candidate

    def _ensure_unique_lean_name(self, lean_name: str, exclude_id: str | None = None) -> None:
        for node_id, node in self._graph.nodes.items():
            if exclude_id is not None and node_id == exclude_id:
                continue
            if node.lean_name == lean_name:
                raise KBError(f"lean_name '{lean_name}' is already used by node '{node_id}'")

    def _would_create_cycle(self, source: str, target: str) -> bool:
        stack = [target]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == source:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.get_dependencies(current))
        return False

    def _transitive_deps(self, id: str) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for dependency_id in self.get_dependencies(node_id):
                visit(dependency_id)
            order.append(node_id)

        visit(id)
        order.pop()
        return order

    def _proofless_statement_to_axiom(self, lean_decl: str) -> str:
        lines = lean_decl.splitlines()
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("theorem "):
                lines[index] = f"{indent}axiom {stripped[len('theorem '):]}"
                return "\n".join(lines)
            if stripped.startswith("lemma "):
                lines[index] = f"{indent}axiom {stripped[len('lemma '):]}"
                return "\n".join(lines)
        return lean_decl

    def _node_to_lean(self, node: Node) -> str:
        if node.lean is None:
            raise KBError(f"Node '{node.id}' has no Lean code")
        if node.type == NodeType.STATEMENT and node.lean_proof is not None:
            body = f"{node.lean} := {node.lean_proof}"
        elif node.type == NodeType.STATEMENT:
            body = self._proofless_statement_to_axiom(node.lean)
        else:
            body = node.lean
        if node.lean_name is None:
            return body
        namespace_parts = [part for part in node.lean_name.split(".")[:-1] if part]
        if not namespace_parts or body.lstrip().startswith("namespace "):
            return body
        lines: list[str] = []
        for namespace in namespace_parts:
            lines.append(f"namespace {namespace}")
        lines.append(body)
        for namespace in reversed(namespace_parts):
            lines.append(f"end {namespace}")
        return "\n".join(lines)
