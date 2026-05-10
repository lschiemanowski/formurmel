# formurmel

`formurmel` is a Lean 4 formalization agent. It gives an LLM a theorem together
with a natural-language proof sketch, lets it search mathlib, test Lean
snippets, and update a small JSON knowledge base, then records the whole
tool-using conversation as a transcript. The task is proof formalization, not
open-ended proof finding from the statement alone.

For example, `problem_6` in the bundled `basic_problems` validation split asks
for the finite-sum identity for cubes:

```lean
theorem sum_range_cubes (n : ℕ) :
    (∑ k ∈ Finset.range (n + 1), k ^ 3) = ((n * (n + 1)) / 2) ^ 2
```

The generated knowledge base starts with one pending statement node for that
problem. That node contains the formal Lean statement and the supplied
natural-language proof. A successful run is not just a chat answer saying
"done": the agent must translate the proof into Lean, compile it through the
configured Lake project, and leave the KB node verified.

## What the Agent Does

A normal run starts with a system prompt, a user prompt, one backend, and a tool
registry. On each step, the model can either call tools or produce a final
assistant message. Tool calls are executed, their responses are appended to the
conversation, and the model is queried again.

The run stops when the model produces a non-empty final answer with no tool
call, when the backend or tool-call protocol fails, or when `agent.max_steps` is
reached. If a KB is configured, a final answer only counts as `done` after the
KB completion check says no target statements remain pending.

The normal tools are:

- `kb`: reads and mutates the working formalization state. It stores
  definition and statement nodes, dependencies, candidate Lean code, compile
  results, and verification status.
- `murmel`: searches and inspects mathlib declarations. It supports semantic
  search, lexical search, source display, and natural-language declaration
  descriptions.
- `lean`: runs self-contained Lean snippets through `lake env lean` in the
  configured Lake project and rejects snippets that compile only by using
  `sorry` or `admit`.

Supported backends are `qwen35_llama_cpp`, `deepseek`, and `openrouter`.
Hosted-provider reasoning content is preserved in transcripts when the provider
returns it.

Training and rollout code should use `AgentEpisodeRunner` rather than shelling
out to the CLI. The runner keeps the backend and non-KB tools alive across
episodes, while each episode gets its own KB path and transcript path.

## Running Problem 6

The usual workflow is to generate the KB/config pair from the dataset and then
run the generated config. This keeps the example close to the files used in
batch rollouts.

```bash
python scripts/build_problem_configs.py \
  --dataset basic_problems \
  --splits validation \
  --run-name example_problem6 \
  --murmel-cache-dir ../murmel/.murmel-cache \
  --lean-project ../cleaner/lean_env \
  --temperature 0.2 \
  --max-tokens 4096 \
  --max-steps 120
```

That command generates configs for the validation split, including:

- `runs/example_problem6/configs/basic_problems/validation/problem_6.agent.toml`
- `runs/example_problem6/configs/basic_problems/validation/problem_6.kb.json`

Start a compatible backend separately, for example a llama.cpp `/completion`
server for a Qwen 3.5 model, then run the agent on the generated problem:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel \
  --config runs/example_problem6/configs/basic_problems/validation/problem_6.agent.toml \
  --prompt "Formalize the target problem stored in the knowledge base."
```

The CLI prints the final assistant message. The durable artifacts are more
important:

- the transcript at
  `runs/example_problem6/transcripts/basic_problems/validation/problem_6.transcript.json`;
- the updated KB at
  `runs/example_problem6/configs/basic_problems/validation/problem_6.kb.json`.

Judge success from the KB, not from the high-level chat status alone. The useful
signal is that the target statement node has compiled Lean code and
`verified: true`.

To inspect a failed run, use diagnosis mode on the saved transcript:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel diagnose \
  --config runs/example_problem6/configs/basic_problems/validation/problem_6.agent.toml \
  --failed-transcript runs/example_problem6/transcripts/basic_problems/validation/problem_6.transcript.json \
  --output runs/example_problem6/diagnoses/problem_6.diagnosis.json
```

Diagnosis mode reuses the configured backend and tools, adds a read-only
transcript inspection tool, and asks the model to produce structured failure
analysis grounded in the transcript and KB state.

## Configuration Files

Configs are TOML files with three tables:

- `[backend]`: model endpoint, sampling parameters, token limits, retries, and
  backend-specific reasoning options.
- `[tools]`: KB path, Lake project paths, murmel cache/config paths, mathlib
  revision, semantic-search device, and Lean timeout.
- `[agent]`: system prompt, transcript path, verbosity, and maximum step count.

Relative paths in config files are resolved relative to the config file itself,
not relative to the shell's current directory.

## Batch Rollouts

`scripts/build_problem_configs.py` writes one `.agent.toml` and one `.kb.json`
per selected dataset problem.

`scripts/run_problem_rollouts.py` runs repeated episodes for generated configs,
writes per-sample artifacts under `runs/<run-name>/rollouts/...`, and appends
one JSONL record per rollout to `runs/<run-name>/rollout_summary.jsonl`.
