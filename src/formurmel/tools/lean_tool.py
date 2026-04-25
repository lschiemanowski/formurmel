from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from formurmel.message import ToolSpec
from formurmel.tools.base import Tool, tool_error, tool_ok


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


class LeanTool(Tool):
    """Run self-contained Lean snippets inside a configured Lake project."""

    def __init__(
        self,
        *,
        lake_project: Path | None = None,
        default_timeout_seconds: float = 60.0,
    ) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self._lake_project = lake_project.expanduser().resolve() if lake_project is not None else None
        self._default_timeout = float(default_timeout_seconds)

    @property
    def name(self) -> str:
        return "lean"

    def tool_description(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Run a self-contained Lean snippet inside the configured Lake project. "
                "Pass a flat object with required field `snippet`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "snippet": {
                        "type": "string",
                        "description": "Self-contained Lean code, including required imports.",
                    },
                },
                "required": ["snippet"],
                "additionalProperties": False,
            },
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return tool_error("lean payload must be an object")
        unknown_fields = sorted(str(key) for key in payload.keys() if key != "snippet")
        if unknown_fields:
            return tool_error(f"lean payload contains unknown field(s): {', '.join(unknown_fields)}")

        snippet = payload.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            return tool_error("lean requires a non-empty string field 'snippet'")

        lake_project = self._lake_project
        if lake_project is None:
            return tool_error("lean tool needs a configured lake_project")
        if not lake_project.exists():
            return tool_error(f"configured lake_project does not exist: {lake_project}")
        if not lake_project.is_dir():
            return tool_error(f"configured lake_project is not a directory: {lake_project}")

        with tempfile.TemporaryDirectory(prefix="lean-agent-") as workdir:
            source_path = Path(workdir) / "snippet.lean"
            source_path.write_text(snippet, encoding="utf-8")
            return self._run_lean(source_path=source_path, cwd=lake_project, timeout=self._default_timeout)

    def _run_lean(self, *, source_path: Path, cwd: Path, timeout: float) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["lake", "env", "lean", str(source_path)],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            return tool_error(f"Lean or Lake executable not available: {exc}")
        except subprocess.TimeoutExpired:
            return tool_error(f"Lean timed out after {timeout} seconds")
        except Exception as exc:  # noqa: BLE001
            return tool_error(f"failed to run Lean: {exc}")

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined_output = "\n".join(part for part in (stdout, stderr) if part)
        source_text = source_path.read_text(encoding="utf-8")
        uses_sorry = _contains_placeholder(source_text, combined_output)
        success = result.returncode == 0 and not uses_sorry
        return tool_ok(
            {
                "success": success,
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "uses_sorry": uses_sorry,
            }
        )
