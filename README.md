# formurmel

`formurmel` is a Lean 4 formalization agent runtime. It runs a tool-using LLM
against mathlib, with `murmel` as the mathlib search/inspection interface, a
Lean checker for candidate snippets, and an optional mutable knowledge base for
problem state.

The package is intentionally an agent runtime, not a training framework. It
provides:

- a CLI for running one formalization task;
- a diagnosis CLI for inspecting a failed transcript;
- a reusable Python episode API for rollout/training code;
- dataset helper scripts that generate per-problem configs and run repeated
  rollouts.

## Agent Semantics

A normal run starts from a system prompt and one user prompt. The runtime builds
the configured backend and tool registry, sends the conversation plus tool specs
to the model, executes any tool calls the model returns, appends tool responses,
and repeats until one of these conditions occurs:

- the assistant returns a non-empty user-facing message with no tool call;
- the backend query fails;
- the assistant returns an invalid tool-call message;
- the assistant returns neither a tool call nor a final message;
- `agent.max_steps` is reached.

Run status is `done` only when the assistant has produced a final message and
the configured completion checks pass. If a KB tool is configured, the runtime
also checks the KB completion state before accepting a final message; a run that
stops while statement nodes are still pending is reported as `error`.

The transcript writer records the final status, step count, error text if any,
final assistant message, and the full conversation including reasoning messages,
tool calls, and tool responses.

## Tools

The normal tool registry contains these tools:

- `murmel`: searches and inspects mathlib declarations. It supports semantic
  search, lexical search, source display, and natural-language declaration
  descriptions. Semantic search defaults to CPU unless configured otherwise.
- `lean`: runs a self-contained Lean snippet through `lake env lean` in the
  configured Lake project. The tool reports failure when the snippet or Lean
  output indicates `sorry` or `admit`.
- `kb`: optional. When `tools.kb_path` is set, this tool loads or creates a JSON
  knowledge base, lets the model inspect and update definition/statement nodes,
  checks KB-local candidates, and stores compile/verification state.

Diagnosis mode reuses the configured backend and normal tools, adds a read-only
`transcript_inspect` tool for the failed transcript, and asks the model to emit
a structured failure diagnosis.

## Backends

Supported backend types are:

- `qwen35_llama_cpp`: talks to a local llama.cpp `/completion` server using the
  Qwen 3.5 formatting expected by this repo.
- `deepseek`: talks to the DeepSeek chat-completions API. By default it uses
  `DEEPSEEK_API_KEY`, `deepseek-v4-pro`, and high-effort reasoning.
- `openrouter`: talks to the OpenRouter chat-completions API. `backend.model`
  must be set, and the API key defaults to `OPENROUTER_API_KEY`.

Hosted reasoning content is preserved in transcripts. For OpenRouter, structured
`reasoning_details` blocks are also passed back on later turns when present.

## Configuration

Configuration is TOML with three top-level tables:

- `[backend]` selects the LLM backend and request parameters such as model,
  endpoint, temperature, token budget, retry settings, and reasoning options.
- `[tools]` selects paths for the KB, Lake projects, murmel cache/config, mathlib
  revision, semantic-search device, and Lean timeout.
- `[agent]` selects the system prompt, transcript path, verbosity, and maximum
  number of model-query steps.

Relative paths in a config file are resolved relative to that config file, not
relative to the current shell directory.

## Python Episode API

Training and rollout code should use `AgentEpisodeRunner` instead of shelling
out to the CLI. A runner keeps the backend and non-KB tools alive across
episodes. Each episode gets a shallow-cloned tool registry and, when a KB path
is provided, a fresh KB tool for that episode. Use one runner per rollout
worker/thread.

## Batch Helpers

`scripts/build_problem_configs.py` reads dataset files under
`datasets/<dataset>/lean_dataset_json/<split>/` and writes one `.agent.toml` and
one `.kb.json` per problem under `runs/<run-name>/configs/...`.

`scripts/run_problem_rollouts.py` runs repeated episodes for generated configs,
writes per-sample artifacts under `runs/<run-name>/rollouts/...`, and appends
one JSONL record per rollout to `runs/<run-name>/rollout_summary.jsonl`.

For judging a rollout, prefer the KB artifact over the high-level agent status:
the useful success signal is that the target statement was compiled and marked
verified in the sample's `kb.json`.

## Example: Run One Formalization Task

The most direct way to run a dataset problem is to generate its KB/config pair
and then run that generated config. This example uses `problem_1` from the
`basic_problems` `validation_plus` split.

First generate configs:

```bash
python scripts/build_problem_configs.py \
  --dataset basic_problems \
  --splits validation_plus \
  --run-name example \
  --murmel-cache-dir ../murmel/.murmel-cache \
  --lean-project ../cleaner/lean_env \
  --temperature 0.2 \
  --max-tokens 4096 \
  --max-steps 120
```

Start a compatible llama.cpp server separately, then run one generated problem:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel \
  --config runs/example/configs/basic_problems/validation_plus/problem_1.agent.toml \
  --prompt "Formalize the target problem stored in the knowledge base."
```

The command prints the final assistant message, writes the full transcript to
`runs/example/transcripts/basic_problems/validation_plus/problem_1.transcript.json`,
and autosaves KB changes to
`runs/example/configs/basic_problems/validation_plus/problem_1.kb.json`.

To diagnose a failed transcript:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel diagnose \
  --config runs/example/configs/basic_problems/validation_plus/problem_1.agent.toml \
  --failed-transcript runs/example/transcripts/basic_problems/validation_plus/problem_1.transcript.json \
  --output runs/example/diagnoses/problem_1.diagnosis.json
```

To reuse the runtime from Python:

```python
from formurmel import AgentEpisodeRunner, load_config

config = load_config("runs/example/configs/basic_problems/validation_plus/problem_1.agent.toml")
runner = AgentEpisodeRunner(config)

episode = runner.run_episode(
    user_prompt="Formalize the target problem stored in the knowledge base.",
    kb_path="runs/example/configs/basic_problems/validation_plus/problem_1.kb.json",
    transcript_path="runs/example/transcripts/basic_problems/validation_plus/problem_1.transcript.json",
    max_steps=120,
)

print(episode.status, episode.steps, episode.error)
runner.close()
```
