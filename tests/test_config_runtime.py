from __future__ import annotations

from pathlib import Path

from formurmel.config import load_config
from formurmel.llm.qwen35_llama_cpp import Qwen35LlamaCppCompletionBackend
from formurmel.message import MessageType
from formurmel.runtime import build_tool_registry


def test_config_loads_murmel_and_qwen_runtime_surface(tmp_path: Path) -> None:
    prompt_path = tmp_path / "system.md"
    prompt_path.write_text("system prompt\n", encoding="utf-8")
    config_path = tmp_path / "formurmel.toml"
    config_path.write_text(
        """
[backend]
type = "qwen35_llama_cpp"
llama_base_url = "http://localhost:9999"

[tools]
kb_path = "kb.json"
murmel_cache_dir = "murmel-cache"
murmel_mathlib_rev = "rev1"
murmel_semantic_device = "cpu"
lean_lake_project = "lake"

[agent]
system_prompt_path = "system.md"
max_steps = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    registry = build_tool_registry(config.tools)
    tool_names = [spec.name for spec in registry.specs()]

    assert config.backend.llama_base_url == "http://localhost:9999"
    assert config.tools.murmel_cache_dir == tmp_path / "murmel-cache"
    assert config.tools.murmel_mathlib_rev == "rev1"
    assert config.agent.system_prompt == "system prompt\n"
    assert tool_names == ["kb", "murmel", "lean"]
    assert "mlsearch" not in tool_names
    assert "nlmlsearch" not in tool_names

    kb_spec = next(spec for spec in registry.specs() if spec.name == "kb")
    assert "lean_proof" in kb_spec.description
    assert "appended after `:=`" in kb_spec.description
    assert "Include `by`" in kb_spec.parameters["properties"]["lean_proof"]["description"]


def test_qwen_parser_accepts_murmel_tool_call() -> None:
    backend = Qwen35LlamaCppCompletionBackend(base_url="http://localhost:8080")
    messages = backend._parse_completion(
        """
<think>
Need a search.
</think>

<tool_call>
<function=murmel>
<parameter=action>
search
</parameter>
<parameter=mode>
semantic
</parameter>
<parameter=query>
commutativity of addition on naturals
</parameter>
</function>
</tool_call>
""".strip()
    )

    assert messages[0].msg_type == MessageType.REASONING
    assert messages[1].msg_type == MessageType.TOOL_CALL
    assert messages[1].content.name == "murmel"
    assert messages[1].content.arguments["action"] == "search"
    assert messages[1].content.arguments["mode"] == "semantic"
