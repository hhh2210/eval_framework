# eval_framework

A lightweight LLM evaluation harness built for **Rubrics RL** research — fast,
repeatable scoring of the many checkpoints an RL run produces. It wraps a single
vLLM-compatible inference path with task runners for IF-EVAL, IFBench,
WritingBench, HealthBench, Arena-Hard, and AlpacaEval.

**Why it's fast.** Evaluation is driven through vLLM's continuous batching: the
runner fires `--num-threads` concurrent requests at one server so the GPU stays
saturated end-to-end (a single H100 holds ~700 W throughout the run). On one
H100 an 8B checkpoint clears every task in roughly 15 minutes;
[`examples/shard_parallel_eval.sh`](examples/shard_parallel_eval.sh) shards a
single model across N GPUs to go faster still.

## Installation

```bash
git clone https://github.com/hhh2210/eval_framework.git
cd eval_framework
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install vllm --torch-backend=auto
```

Or install the released package from PyPI (no clone needed):

```bash
pip install llm-eval-framework
```

After installation the `eval-framework` command is available in the venv.

Most benchmark data ships with the package — Arena-Hard v2.0 questions/baselines
and the AlpacaEval GPT-4 baseline are bundled under `tasks/arena_hard/data/` and
`tasks/alpaca_eval/data/`. **IFBench is the one exception:** its verifier source
is not redistributed here, so clone it only if you plan to run that task:

```bash
git clone https://github.com/allenai/IFBench .external/IFBench   # only for --tasks ifbench
```

(or point `--ifbench-dir` at an existing checkout). Every other task runs without
this step.

## Quick Start

The runner needs a model to call. There are two ways to provide one.

### Option A — server mode (recommended, saturates the GPU)

Start a vLLM server in one terminal:

```bash
vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --gpu-memory-utilization 0.95
```

Then point the runner at it. Begin with a 4-example smoke test to confirm the
setup is wired up correctly (~30 s) before launching a full run:

```bash
eval-framework \
  --tasks ifeval \
  --model Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --max-examples 4 \
  --output-dir outputs/smoke
```

Drop `--max-examples` for the full task. The `--model` value must match the
server's `--served-model-name`.

### Option B — `--local` (zero setup, single process)

`--local` loads vLLM in-process, so no separate server is needed — handy for a
quick one-off:

```bash
eval-framework \
  --tasks ifeval \
  --model Qwen/Qwen3-4B \
  --local \
  --max-examples 4 \
  --output-dir outputs/smoke
```

Use `--local` for convenience; use **server mode for throughput**. In server
mode the runner drives vLLM with `--num-threads` concurrent requests, which
vLLM batches continuously to keep the GPU fully utilised — this is what gets a
single card to ~700 W and an 8B checkpoint through all tasks in ~15 minutes.

To sweep a whole RL run across many checkpoints and GPUs, use the ready-made
scripts in `examples/` — see [Multi-GPU Batch Evaluation](#multi-gpu-batch-evaluation) below.

## Tasks

| Task | Judge needed? | Key flags |
|------|:---:|---|
| `ifeval` | No (rule-based) | `--ifeval-input` |
| `ifbench` | No (rule-based) | `--ifbench-dir`, `--ifbench-input` |
| `writingbench` | Yes | `--writingbench-query`, `--writingbench-write-excel` |
| `healthbench` | Yes | `--healthbench-data` |
| `arena-hard` | Yes | `--arena-hard-dir`, `--arena-hard-benchmark` |
| `alpaca-eval` | Yes | `--alpaca-eval-reference`, `--alpaca-eval-hf-dataset` |

### Modes

- **`--inference-only`** — generate responses, skip judging. Judge later with `--judge-only`.
- **`--judge-only`** — score existing responses. Only supports writingbench / healthbench / arena-hard / alpaca-eval (ifeval and ifbench are rule-based and score during inference).

## Multi-GPU Batch Evaluation

For RL experiments you typically need to evaluate many checkpoints across all benchmarks. We provide ready-to-use scripts in `examples/`:

| Script | Use case |
|--------|----------|
| [`examples/shard_parallel_eval.sh`](examples/shard_parallel_eval.sh) | Evaluate ONE model on all benchmarks — shards data across N GPUs for max throughput |
| [`examples/batch_eval.sh`](examples/batch_eval.sh) | Evaluate one training run — auto-detects checkpoints, schedules across N GPUs in rounds, judges, plots |

**Usage:**

```bash
# 1. Copy and edit the CONFIG section at the top of the script
cp examples/batch_eval.sh my_eval.sh
vim my_eval.sh   # edit CKPT_DIR, OUT_DIR, STEPS, etc.

# 2. Run
bash my_eval.sh
```

**What the scripts handle automatically:**

- **Multi-round scheduling** — if you have more checkpoints than GPUs, the script runs them in rounds and cleans up vLLM between rounds
- **vLLM lifecycle** — starts servers, waits for health checks, kills process groups after eval
- **Judge batching** — runs judge jobs in small batches to respect API rate limits (configurable `JUDGE_BATCH_SIZE`)
- **Phase control** — set `RUN_INFERENCE=0` / `RUN_JUDGE=0` / `RUN_PLOT=0` to skip phases (e.g. re-run judge only after fixing an issue)
- **Logging** — all vLLM and eval logs go to `LOG_DIR` for debugging; judge stderr (tqdm) is tee'd to terminal

### vLLM Tips

- **Do NOT set `--max-model-len`** unless you know exactly what you're doing. Let the model use its native context length (e.g. 32768 for Qwen3-4B). Setting it too low causes `VLLMValidationError` on long prompts.
- **`--gpu-memory-utilization 0.95`** is safe for H100s and maximizes KV cache.
- Increase `--num-threads` when GPU utilization is low and the serving backend
  has available capacity.
- **Kill process groups, not just PIDs** — `kill -- -${pid}` ensures all vLLM child processes are cleaned up. Follow with `pkill -f "vllm serve"` between rounds.

## Output structure

```
outputs/
├── step_120/
│   ├── run_0/                     # one subdir per sample (mean@N evaluation)
│   │   ├── ifeval/       # summary.json, responses.jsonl
│   │   ├── ifbench/      # summary.json, responses.jsonl, eval_results_*.jsonl
│   │   ├── writingbench/ # responses.jsonl, scores.jsonl, summary.json
│   │   ├── healthbench/  # responses.jsonl, scores.jsonl, summary.json
│   │   ├── arena-hard/   # model_answer/, model_judgment/, summary.json
│   │   └── alpaca-eval/  # model_answer/, model_judgment/, summary.json
│   ├── run_1/ ...                 # up to run_{N-1}
│   ├── ifeval/summary_agg.json    # aggregated mean / std / sem / per_run
│   ├── healthbench/summary_agg.json
│   └── ...
├── step_240/
│   └── ...
└── plots/
    ├── ifeval.png
    ├── ifbench.png
    ├── healthbench.png
    ├── writingbench.png
    ├── arena-hard.png
    ├── alpaca-eval.png
    └── all_tasks.png
```

`run_k/` holds the k-th sample's raw artifacts; `summary_agg.json` at the step
root is what plotting consumes. With `N=1` everything still works but error bars
collapse to zero width.

## Sampling variance (mean@N + error bars)

`batch_eval.sh` runs each checkpoint N times per task and then aggregates.
Because the same live vLLM server handles all N samples, prefix caching
amortises prefill — wall time is roughly `decode(N)×`, not `N×` cold starts.

**Per-task defaults (override with env vars):**

| Task | Default N | Why |
|---|---:|---|
| `ifeval` / `ifbench` | 8 | Rule-based scoring, cost is only GPU decode |
| `healthbench` | 8 | Rubric-based, judge cost 8× but gives honest error bars |
| `writingbench` | 4 | Large rubric per prompt; 4 samples is usually enough |
| `arena-hard` / `alpaca-eval` | 1 | These already report internal bootstrap CI; extra sampling rarely helps |

Override any of them:

```bash
N_SAMPLES_HEALTHBENCH=4 N_SAMPLES_WRITINGBENCH=1 bash examples/batch_eval.sh
```

Set them all to 1 to reproduce the original single-run behavior.

## Plotting

After inference + judge + aggregate, combine eval results from any set of
checkpoints into training curves. Steps that carry a `summary_agg.json` get
error bars automatically; those without fall back to a plain line.

```bash
python tools/plot_training_curves.py \
  --runs "run_a=outputs/run_a" \
  --runs "run_b=outputs/run_b" \
  --name-pattern "run_a=step_{step}" \
  --name-pattern "run_b=step_{step}" \
  --steps "120,240,360,480,600" \
  --tasks "ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval" \
  --plot-dir outputs/plots \
  --show-errorbar ci95          # ci95 (1.96·SEM) | sem | std | none
```

`batch_eval.sh` runs `aggregate_runs.py` during its plotting phase. To aggregate
manually:

```bash
python tools/aggregate_runs.py \
  --out-dir outputs/run_a \
  --steps   120,240,360,480,600 \
  --tasks   ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval \
  --n-samples ifeval=8,ifbench=8,healthbench=8,writingbench=4,arena-hard=1,alpaca-eval=1
```

## Judge comparison

Compare scores from different judge models:

```bash
python tools/judge_compare.py \
  --judges flash=outputs/qwen3-4B \
  --judges plus=outputs/qwen3-4B-judge-qwen-plus \
  --out outputs/judge_compare.json
```

## Global request throttle

When running many judge jobs in parallel (e.g. 5 background `eval-framework` processes), all
remote API requests share a **file-lock-based global throttle** to prevent 429 rate-limit errors.

| Env var | Default | Description |
|---------|---------|-------------|
| `MIN_INTERVAL_S` | `0.005` (≈200 QPS) | Minimum interval between consecutive API requests across all threads/processes |
| `EVAL_THROTTLE_STATE_PATH` | `/tmp/eval_framework_global_throttle.state` | Shared state file path; processes using the same path share one throttle |

```bash
export MIN_INTERVAL_S=0.01          # ~100 QPS global cap
export EVAL_THROTTLE_STATE_PATH=/tmp/eval_framework_global_throttle.state
```

Set `MIN_INTERVAL_S=0` to disable throttling entirely.

## Notes

- `--output-dir` controls where responses/scores/summaries go. With `--tasks`, output is written to `<output-dir>/<task>/`.
- If you set `--served-model-name` in `vllm serve`, pass that same name via `--model`.
- IFBench test data is bundled at `tasks/ifbench/data/IFBench_test.jsonl`. The AllenAI verifier source resolves from `.external/IFBench` unless you pass `--ifbench-dir`.
- Arena-Hard questions and baselines (`o3-mini-2025-01-31`, `gemini-2.0-flash-001` for v2.0) are bundled at `tasks/arena_hard/data/`. Falls back to `.external/arena-hard-auto` if present. Override with `--arena-hard-dir` to use a custom repo (e.g. a newer bench version).
- AlpacaEval reference outputs auto-download from HuggingFace. Override with `--alpaca-eval-reference`.
- IFBench also needs `emoji` + `syllapy` installed (included in `pyproject.toml` deps).
- `setuptools<81` is pinned because `syllapy` depends on `pkg_resources` which was removed in setuptools 82.

## License And Third-Party Assets

The framework code is released under Apache-2.0. Bundled benchmark assets remain
under their original upstream licenses and citation requirements. Before
redistributing modified benchmark data, check the upstream projects for the
current license and attribution terms.
