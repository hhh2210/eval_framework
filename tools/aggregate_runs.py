#!/usr/bin/env python3
"""Aggregate mean / std / SEM across N runs of the same checkpoint.

For each (step, task), reads the scalar metric from:

    {out_dir}/step_{step}/run_{k}/{task}/summary.json   for k in 0..N-1

and writes a merged summary to:

    {out_dir}/step_{step}/{task}/summary_agg.json

`plot_training_curves.py` picks this file up to render mean + error bars.

Usage:
    python tools/aggregate_runs.py \
        --out-dir outputs/my_exp \
        --steps   120,240,360,480,600 \
        --tasks   ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval \
        --n-samples ifeval=8,ifbench=8,healthbench=8,writingbench=4,arena-hard=1,alpaca-eval=1

`--n-samples` also accepts a single int ("8") to apply the same N to every task.
Runs that are missing on disk are counted as "not yet finished" and skipped
rather than treated as zeros, so this script is safe to run mid-experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-task scalar extractors. These mirror plot_training_curves.load_* so the
# same metric that ends up on the training curve is also the metric we
# aggregate across runs. If you change one, change the other.
# ---------------------------------------------------------------------------

def _find_file(*candidates: str) -> Optional[str]:
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_ifeval(d: str) -> Optional[float]:
    p = _find_file(os.path.join(d, "ifeval", "summary.json"))
    if not p:
        return None
    return json.loads(Path(p).read_text(encoding="utf-8")).get("strict", {}).get("prompt_accuracy")


def _load_ifbench(d: str) -> Optional[float]:
    p = _find_file(os.path.join(d, "ifbench", "summary.json"))
    if not p:
        return None
    return json.loads(Path(p).read_text(encoding="utf-8")).get("loose", {}).get("prompt_accuracy")


def _load_healthbench(d: str) -> Optional[float]:
    p = _find_file(
        os.path.join(d, "healthbench", "scores.jsonl"),
        os.path.join(d, "healthbench", "scores", "scores.jsonl"),
    )
    if not p:
        return None
    scores = [r["score"] for r in _iter_jsonl(p) if isinstance(r.get("score"), (int, float))]
    return float(np.mean(scores)) if scores else None


def _load_writingbench(d: str) -> Optional[float]:
    p = _find_file(os.path.join(d, "writingbench", "scores", "scores.jsonl"))
    if not p:
        return None
    per_prompt = []
    for rec in _iter_jsonl(p):
        vals = []
        for _dim, items in rec.get("scores", {}).items():
            if isinstance(items, list):
                for item in items:
                    v = item.get("score")
                    if isinstance(v, (int, float)):
                        vals.append(v)
        if vals:
            per_prompt.append(np.mean(vals))
    return float(np.mean(per_prompt)) if per_prompt else None


def _load_arena_hard(d: str) -> Optional[float]:
    p = _find_file(os.path.join(d, "arena-hard", "summary.json"))
    if not p:
        return None
    return json.loads(Path(p).read_text(encoding="utf-8")).get("metrics", {}).get("overall", {}).get("winrate")


def _load_alpaca_eval(d: str) -> Optional[float]:
    p = _find_file(os.path.join(d, "alpaca-eval", "summary.json"))
    if not p:
        return None
    return json.loads(Path(p).read_text(encoding="utf-8")).get("metrics", {}).get("overall", {}).get("winrate")


LOADERS: Dict[str, Callable[[str], Optional[float]]] = {
    "ifeval": _load_ifeval,
    "ifbench": _load_ifbench,
    "healthbench": _load_healthbench,
    "writingbench": _load_writingbench,
    "arena-hard": _load_arena_hard,
    "alpaca-eval": _load_alpaca_eval,
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _parse_n_samples(spec: str) -> Dict[str, int]:
    spec = spec.strip()
    if not spec:
        return {"_default": 1}
    if "=" not in spec:
        return {"_default": int(spec)}
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        task, n = part.split("=", 1)
        out[task.strip()] = int(n)
    return out


def _summarize(scores: List[float]) -> dict:
    mean = float(np.mean(scores))
    if len(scores) > 1:
        std = float(np.std(scores, ddof=1))
        sem = std / math.sqrt(len(scores))
    else:
        std = 0.0
        sem = 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "n_runs": len(scores),
        "per_run": scores,
    }


def aggregate(
    out_dir: str,
    steps: List[str],
    tasks: List[str],
    n_per_task: Dict[str, int],
) -> None:
    default_n = n_per_task.get("_default", 1)
    for step in steps:
        step_dir = os.path.join(out_dir, f"step_{step}")
        if not os.path.isdir(step_dir):
            print(f"[step_{step}] missing dir, skip ({step_dir})")
            continue
        print(f"\n[step_{step}]")
        for task in tasks:
            loader = LOADERS.get(task)
            if loader is None:
                print(f"  [{task:<12}] unknown task, skip")
                continue
            n = n_per_task.get(task, default_n)
            scores: List[float] = []
            for k in range(n):
                run_dir = os.path.join(step_dir, f"run_{k}")
                v = loader(run_dir)
                if v is not None:
                    scores.append(float(v))
            if not scores:
                print(f"  [{task:<12}] no data (expected {n} run(s))")
                continue
            summary = _summarize(scores)
            agg_dir = os.path.join(step_dir, task)
            os.makedirs(agg_dir, exist_ok=True)
            out_path = os.path.join(agg_dir, "summary_agg.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            partial = "" if len(scores) == n else f" (partial: {len(scores)}/{n})"
            print(
                f"  [{task:<12}] mean={summary['mean']:.4f} "
                f"std={summary['std']:.4f} sem={summary['sem']:.4f} "
                f"n={summary['n_runs']}{partial}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True, help="eval output dir (contains step_* subdirs)")
    p.add_argument("--steps", required=True, help="comma-separated step numbers, e.g. 120,240,360")
    p.add_argument(
        "--tasks",
        default="ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval",
    )
    p.add_argument(
        "--n-samples",
        required=True,
        help="either an int (same N for all) or 'task=N,task=N' per-task spec",
    )
    args = p.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    n_per_task = _parse_n_samples(args.n_samples)
    aggregate(args.out_dir, steps, tasks, n_per_task)


if __name__ == "__main__":
    main()
