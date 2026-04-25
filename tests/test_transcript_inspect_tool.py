from __future__ import annotations

import json
from pathlib import Path

from formurmel.tools.transcript_inspect_tool import TranscriptInspectTool


def write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "error",
                "steps": 3,
                "error": "assistant ended early",
                "final_message": {
                    "role": "assistant",
                    "content": "The environment is broken.",
                },
                "conversation": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "formalize target"},
                    {
                        "role": "assistant",
                        "msg_type": "tool_call",
                        "content": {
                            "id": "call_1",
                            "name": "lean",
                            "arguments": {"snippet": "import Bad"},
                        },
                    },
                    {
                        "role": "tool",
                        "msg_type": "tool_response",
                        "content": {
                            "id": "call_1",
                            "name": "lean",
                            "content": {
                                "ok": True,
                                "result": {
                                    "success": False,
                                    "exit_code": 1,
                                    "stdout": "snippet.lean:1:0: error: object file missing",
                                },
                            },
                            "is_error": False,
                        },
                    },
                    {"role": "assistant", "content": "The Lean environment is not working."},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_transcript_inspect_summary_search_and_errors(tmp_path: Path) -> None:
    transcript_path = tmp_path / "failed.transcript.json"
    write_transcript(transcript_path)
    tool = TranscriptInspectTool(transcript_path)

    summary = tool.execute({"action": "summary"})["result"]
    assert summary["status"] == "error"
    assert summary["message_count"] == 5
    assert summary["tool_calls"] == {"lean": 1}
    assert summary["error_message_indices"] == [3]

    errors = tool.execute({"action": "errors"})["result"]
    assert errors["total_errors"] == 1
    assert errors["errors"][0]["index"] == 3
    assert errors["errors"][0]["previous_tool_call_index"] == 2
    assert "object file missing" in errors["errors"][0]["summary"]

    search = tool.execute({"action": "search", "query": "environment"})["result"]
    assert search["total_matches"] >= 1
    assert any(match["index"] == 4 for match in search["matches"])

    error_search = tool.execute({"action": "search", "query": "object file"})["result"]
    assert error_search["matches"][0]["index"] == 3
    assert error_search["matches"][0]["previous_tool_call_index"] == 2

    message = tool.execute({"action": "message", "index": 4})["result"]
    assert "Lean environment" in message["text"]
    assert message["previous_tool_call_index"] is None
    assert message["self_tool_call_index"] is None

    tool_call = tool.execute({"action": "message", "index": 2})["result"]
    assert tool_call["self_tool_call_index"] == 2
    assert tool_call["previous_tool_call_index"] is None

    tool_response = tool.execute({"action": "message", "index": 3})["result"]
    assert tool_response["previous_tool_call_index"] == 2
    assert tool_response["self_tool_call_index"] is None


def test_transcript_inspect_messages_accepts_index_as_start_alias(tmp_path: Path) -> None:
    transcript_path = tmp_path / "failed.transcript.json"
    write_transcript(transcript_path)
    tool = TranscriptInspectTool(transcript_path)

    result = tool.execute({"action": "messages", "index": 3, "limit": 1})["result"]

    assert result["start"] == 3
    assert result["messages"][0]["index"] == 3
    assert result["messages"][0]["previous_tool_call_index"] == 2
    assert "interpreted as messages.start" in result["note"]


def test_transcript_inspect_messages_prefers_explicit_start_over_index(tmp_path: Path) -> None:
    transcript_path = tmp_path / "failed.transcript.json"
    write_transcript(transcript_path)
    tool = TranscriptInspectTool(transcript_path)

    result = tool.execute({"action": "messages", "start": 2, "index": 4, "limit": 1})["result"]

    assert result["start"] == 2
    assert result["messages"][0]["index"] == 2
    assert result["messages"][0]["self_tool_call_index"] == 2
    assert "note" not in result
