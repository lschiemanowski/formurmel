# formurmel

A Lean formalization agent runtime that reuses the cleaner-style agent loop while replacing the old mathlib search tools with one `murmel` tool.

This first pass intentionally keeps the surface small:

- one backend: local llama.cpp `/completion` with Qwen 3.5 formatting
- tools: `murmel`, `lean`, and optional `kb`
- no training code and no port of the older `mlsearch`/`nlmlsearch` tools

Example `formurmel.toml`:

```toml
[backend]
type = "qwen35_llama_cpp"
llama_base_url = "http://localhost:8080"
temperature = 0.2
max_tokens = 4096

[tools]
murmel_cache_dir = "../murmel/.murmel-cache"
lean_lake_project = "../mathlib4"

[agent]
system_prompt_path = "prompts/system_prompt.md"
transcript_path = "runs/latest_transcript.json"
max_steps = 120
```

Hosted backend examples:

```toml
[backend]
type = "deepseek"
model = "deepseek-v4-pro"
api_key_env = "DEEPSEEK_API_KEY"
reasoning_enabled = true
reasoning_effort = "high"
max_tokens = 4096
```

```toml
[backend]
type = "openrouter"
model = "deepseek/deepseek-r1"
api_key_env = "OPENROUTER_API_KEY"
reasoning_enabled = true
reasoning_effort = "high"
max_tokens = 4096
```

DeepSeek reasoning is read from `reasoning_content`. OpenRouter reasoning is
read from `reasoning` when present, and structured `reasoning_details` blocks
are preserved in transcripts and passed back on later turns.

Run:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel --config formurmel.toml --prompt "Formalize ..."
```

Diagnose a failed run:

```bash
PYTHONPATH=src:../murmel/src python -m formurmel diagnose \
  --config runs/basic_validation_qwen35/configs/basic_problems/validation/problem_54.agent.toml \
  --failed-transcript runs/basic_validation_qwen35/transcripts/basic_problems/validation/problem_54.transcript.json \
  --output runs/basic_validation_qwen35/diagnoses/problem_54.diagnosis.json
```

Diagnosis mode reuses the configured backend plus normal tools, adds a read-only
`transcript_inspect` tool for the failed transcript, and asks the model for a
structured JSON diagnosis of what failed and how to avoid the failure mode. The
diagnosis agent transcript is written next to the failed transcript by default
as `*.diagnosis.transcript.json`.

Generate per-problem configs from copied datasets:

```bash
python scripts/build_problem_configs.py \
  --dataset basic_problems \
  --splits validation \
  --murmel-cache-dir ../murmel/.murmel-cache \
  --lean-project ../cleaner/lean_env \
  --run-name basic_validation_qwen35
```

This writes one `.agent.toml` and one `.kb.json` per problem under `runs/<run-name>/configs/...`.

Run repeated rollouts for generated configs:

```bash
PYTHONPATH=src:../murmel/src python scripts/run_problem_rollouts.py \
  --run-dir runs/basic_train_eval_qwen35 \
  --dataset basic_problems \
  --split train \
  --samples-per-problem 2 \
  --max-turns 80 \
  --jobs 1
```

This writes per-sample artifacts under `runs/<run-name>/rollouts/...` and appends
one record per rollout to `runs/<run-name>/rollout_summary.jsonl`. The command
assumes the configured llama.cpp server is already running. By default it keeps
one live status line per problem, for example
`problem_10: rollout 1: step 4, rollout 2: queued`; pass `--no-progress` to
disable the in-place status display.

## Python episode API

Training code should call the agent through `AgentEpisodeRunner` instead of
shelling out to the CLI or depending on low-level agent-loop details:

```python
from formurmel import AgentEpisodeRunner, load_config

template_config = load_config("runs/template.agent.toml")
runner = AgentEpisodeRunner(template_config)

episode = runner.run_episode(
    user_prompt="Formalize the target problem stored in the knowledge base.",
    kb_path="outputs/run/rollouts/update_0001/problem_1/sample_00/kb.json",
    transcript_path="outputs/run/rollouts/update_0001/problem_1/sample_00/transcript.json",
    max_steps=120,
)

print(episode.status, episode.steps, episode.error)
```

Use one runner per rollout worker/thread. The runner keeps the backend and
non-KB tools such as `murmel` alive across episodes, while each episode gets a
fresh `kb` tool pointed at that episode's KB path.
