from __future__ import annotations

from pathlib import Path

from formurmel.config import load_config
from formurmel.llm.hosted_chat import DeepSeekChatCompletionBackend, OpenRouterChatCompletionBackend
from formurmel.llm.qwen35_llama_cpp import Qwen35LlamaCppCompletionBackend
from formurmel.message import Conversation, Message, MessageType, Role, ToolCall, ToolResponse
from formurmel.runtime import build_backend, build_tool_registry


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


def test_config_defaults_murmel_semantic_device_to_cpu(tmp_path: Path) -> None:
    prompt_path = tmp_path / "system.md"
    prompt_path.write_text("system prompt\n", encoding="utf-8")
    config_path = tmp_path / "formurmel.toml"
    config_path.write_text(
        """
[backend]
type = "qwen35_llama_cpp"

[tools]
murmel_cache_dir = "murmel-cache"

[agent]
system_prompt_path = "system.md"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.tools.murmel_semantic_device == "cpu"


def test_config_loads_openrouter_backend_surface(tmp_path: Path) -> None:
    config_path = tmp_path / "formurmel.toml"
    config_path.write_text(
        """
[backend]
type = "openrouter"
model = "deepseek/deepseek-r1"
api_key_env = "TEST_OPENROUTER_KEY"
reasoning_enabled = true
reasoning_effort = "high"
openrouter_site_url = "https://example.invalid"
openrouter_app_name = "formurmel-test"

[agent]
system_prompt = "system"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    backend = build_backend(config.backend)

    assert isinstance(backend, OpenRouterChatCompletionBackend)
    assert backend.model == "deepseek/deepseek-r1"
    assert backend.api_key_env == "TEST_OPENROUTER_KEY"
    assert backend.reasoning_enabled is True
    assert backend.reasoning_effort == "high"


def test_openrouter_config_requires_model(tmp_path: Path) -> None:
    config_path = tmp_path / "formurmel.toml"
    config_path.write_text(
        """
[backend]
type = "openrouter"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "backend.model must be set for openrouter" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected openrouter config without model to fail")


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


def test_deepseek_payload_preserves_reasoning_for_tool_call_turns() -> None:
    backend = DeepSeekChatCompletionBackend(
        api_key_env="TEST_DEEPSEEK_KEY",
        temperature=0.7,
        max_tokens=1024,
    )
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="question"),
            Message(role=Role.ASSISTANT, content="need the tool", msg_type=MessageType.REASONING),
            Message(
                role=Role.ASSISTANT,
                content=ToolCall(id="call_1", name="murmel", arguments={"query": "Nat.add_comm"}),
            ),
            Message(
                role=Role.TOOL,
                content=ToolResponse(id="call_1", name="murmel", content="search result"),
            ),
        ]
    )

    payload = backend._build_payload(conversation, [])
    assistant_payload = payload["messages"][1]

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload
    assert assistant_payload["reasoning_content"] == "need the tool"
    assert assistant_payload["tool_calls"][0]["function"]["name"] == "murmel"


def test_openrouter_reasoning_details_become_reasoning_and_are_preserved() -> None:
    backend = OpenRouterChatCompletionBackend(
        model="deepseek/deepseek-r1",
        api_key_env="TEST_OPENROUTER_KEY",
        reasoning_enabled=True,
    )
    assistant_messages = backend._parse_assistant_message(
        {
            "role": "assistant",
            "content": "final",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "structured thought",
                    "id": "reasoning-text-1",
                    "index": 0,
                }
            ],
        }
    )
    conversation = Conversation(messages=[Message(role=Role.USER, content="question"), *assistant_messages])

    rendered = backend._conversation_messages(conversation)

    assert assistant_messages[0].msg_type == MessageType.REASONING
    assert assistant_messages[0].content == "structured thought"
    assert assistant_messages[0].provider_state == {
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "structured thought",
                "id": "reasoning-text-1",
                "index": 0,
            }
        ]
    }
    assert rendered[1]["reasoning"] == "structured thought"
    assert rendered[1]["reasoning_details"][0]["text"] == "structured thought"


def test_openrouter_payload_uses_unified_reasoning_object() -> None:
    backend = OpenRouterChatCompletionBackend(
        model="anthropic/claude-sonnet-4.5",
        api_key_env="TEST_OPENROUTER_KEY",
        reasoning_max_tokens=2048,
        app_name="formurmel-test",
    )
    payload = backend._build_payload(Conversation(messages=[Message(role=Role.USER, content="question")]), [])

    assert payload["model"] == "anthropic/claude-sonnet-4.5"
    assert payload["reasoning"] == {"max_tokens": 2048, "exclude": False}


def test_qwen_prompt_renderer_does_not_render_none_message_content() -> None:
    backend = Qwen35LlamaCppCompletionBackend(base_url="http://localhost:8080")

    prompt = backend._render_prompt(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "murmel",
                            "arguments": {
                                "action": "search",
                                "mode": "lexical",
                                "query": "x",
                                "regex": False,
                                "module": None,
                                "limit": 3,
                            },
                        },
                    }
                ],
            },
            {"role": "tool", "content": None},
        ],
        [],
    )

    assert "None" not in prompt
    assert "False" not in prompt
    assert "\nnull\n" in prompt
    assert "\nfalse\n" in prompt
    assert "<function=murmel>" in prompt
    assert "<tool_response>\n\n</tool_response>" in prompt
