#!/usr/bin/env python3
"""Plot evaluation scores across training steps for multiple model sets.

Reads eval_outputs/{model_name}/{task}/... and produces per-task line plots
comparing different training runs (e.g. biased-judge vs normal).

Usage:
    python tools/plot_training_curves.py \
        --runs "Qwen3-4B_healthbench=outputs/eval_outputs" \
        --runs "root=outputs/eval_outputs" \
        --steps 50,100,150,200,250,300,350 \
        --plot-dir outputs/judge_plots

Each --runs entry is "label=base_dir".  Model directories are resolved as
base_dir/{prefix}_step{N}/  where prefix is derived from the label
(healthbench for "Qwen3-4B_healthbench", root for "root", etc.).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Score loaders
#
# Each load_* first checks for an aggregated summary_agg.json (written by
# tools/aggregate_runs.py when mean@N evaluation is on). If present, it
# returns the mean from there. Otherwise it falls back to the original
# single-run summary.json so legacy outputs still plot fine.
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


def _try_load_agg(model_dir: str, task: str) -> Optional[Dict[str, Any]]:
    """Return aggregated dict {mean, std, sem, n_runs, per_run} or None."""
    path = os.path.join(model_dir, task, "summary_agg.json")
    if not os.path.exists(path):
        return None
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if obj.get("mean") is None:
        return None
    return obj


def _try_load_healthbench_summary_json(model_dir: str) -> Optional[Dict[str, Any]]:
    """Load healthbench from step-level summary.json written by eval_framework --num-runs N."""
    path = os.path.join(model_dir, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    avg = obj.get("avg_score")
    if not isinstance(avg, (int, float)):
        return None
    run_dirs = obj.get("run_dirs", [])
    if len(run_dirs) > 1:
        per_run: List[float] = []
        for rdir in run_dirs:
            scores_path = os.path.join(rdir, "scores.jsonl")
            if os.path.exists(scores_path):
                scores = [r["score"] for r in _iter_jsonl(scores_path)
                          if isinstance(r.get("score"), (int, float))]
                if scores:
                    per_run.append(float(np.mean(scores)))
        if len(per_run) > 1:
            mean = float(np.mean(per_run))
            std = float(np.std(per_run, ddof=1))
            sem = std / np.sqrt(len(per_run))
            return {"mean": mean, "std": std, "sem": sem, "n_runs": len(per_run)}
        if per_run:
            return {"mean": per_run[0], "std": 0.0, "sem": 0.0, "n_runs": 1}
    return {"mean": float(avg), "std": 0.0, "sem": 0.0, "n_runs": 1}


def load_ifeval(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "ifeval")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "ifeval", "summary.json"),
    )
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("strict", {}).get("prompt_accuracy")


def load_ifbench(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "ifbench")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "ifbench", "summary.json"),
    )
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("loose", {}).get("prompt_accuracy")


def load_healthbench(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "healthbench")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "healthbench", "scores.jsonl"),
        os.path.join(model_dir, "healthbench", "scores", "scores.jsonl"),
        os.path.join(model_dir, "scores.jsonl"),  # single-task output (no task subdir)
    )
    if not path:
        agg = _try_load_healthbench_summary_json(model_dir)
        return agg.get("mean") if agg is not None else None
    scores = []
    for record in _iter_jsonl(path):
        s = record.get("score")
        if isinstance(s, (int, float)):
            scores.append(s)
    return float(np.mean(scores)) if scores else None


def load_writingbench(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "writingbench")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "writingbench", "scores", "scores.jsonl"),
    )
    if not path:
        return None
    all_scores = []
    for record in _iter_jsonl(path):
        dim_scores = record.get("scores", {})
        vals = []
        for _dim, items in dim_scores.items():
            if isinstance(items, list):
                for item in items:
                    v = item.get("score")
                    if isinstance(v, (int, float)):
                        vals.append(v)
        if vals:
            all_scores.append(np.mean(vals))
    return float(np.mean(all_scores)) if all_scores else None


def load_arena_hard(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "arena-hard")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "arena-hard", "summary.json"),
    )
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("metrics", {}).get("overall", {}).get("winrate")


def load_alpaca_eval(model_dir: str) -> Optional[float]:
    agg = _try_load_agg(model_dir, "alpaca-eval")
    if agg is not None:
        return agg.get("mean")
    path = _find_file(
        os.path.join(model_dir, "alpaca-eval", "summary.json"),
    )
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("metrics", {}).get("overall", {}).get("winrate")


TASK_LOADERS = {
    "ifeval": ("IFEval (strict prompt acc)", load_ifeval),
    "ifbench": ("IFBench (loose prompt acc)", load_ifbench),
    "healthbench": ("HealthBench (avg score)", load_healthbench),
    "writingbench": ("WritingBench (avg score)", load_writingbench),
    "arena-hard": ("Arena-Hard (winrate)", load_arena_hard),
    "alpaca-eval": ("AlpacaEval (winrate)", load_alpaca_eval),
}


# ScorePoint = (mean, std, sem, n). n == 0 ⇒ missing, n == 1 ⇒ no error bar.
ScorePoint = Tuple[Optional[float], Optional[float], Optional[float], int]


def load_score_with_error(model_dir: str, task: str) -> ScorePoint:
    """Return (mean, std, sem, n) for a single (model_dir, task) data point.

    Prefers summary_agg.json when present; otherwise falls back to the plain
    scalar loader (n=1, no error bar)."""
    agg = _try_load_agg(model_dir, task)
    if agg is not None:
        return (
            agg.get("mean"),
            float(agg.get("std", 0.0) or 0.0),
            float(agg.get("sem", 0.0) or 0.0),
            int(agg.get("n_runs", 1) or 1),
        )
    if task == "healthbench":
        agg = _try_load_healthbench_summary_json(model_dir)
        if agg is not None:
            return (
                agg.get("mean"),
                float(agg.get("std", 0.0) or 0.0),
                float(agg.get("sem", 0.0) or 0.0),
                int(agg.get("n_runs", 1) or 1),
            )
    pair = TASK_LOADERS.get(task)
    if pair is None:
        return (None, None, None, 0)
    _, loader = pair
    val = loader(model_dir)
    if val is None:
        return (None, None, None, 0)
    return (float(val), 0.0, 0.0, 1)


def _yerr(point: ScorePoint, kind: str) -> float:
    _, std, sem, n = point
    if n is None or n <= 1 or kind == "none":
        return 0.0
    if kind == "std":
        return float(std or 0.0)
    if kind == "sem":
        return float(sem or 0.0)
    # ci95: 1.96 * SEM is a fine approximation for N >= 6. For N=2-5 it's a
    # mild underestimate (true t-critical is larger) but still more honest
    # than plotting no error bar at all.
    return 1.96 * float(sem or 0.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_series(ax, steps: List[int], label: str, points: List[ScorePoint], errorbar: str):
    xs: List[int] = []
    ys: List[float] = []
    yerrs: List[float] = []
    for s, pt in zip(steps, points):
        mean = pt[0]
        if mean is None:
            continue
        xs.append(s)
        ys.append(mean)
        yerrs.append(_yerr(pt, errorbar))
    if not xs:
        return
    has_err = any(e > 0 for e in yerrs)
    if has_err:
        ax.errorbar(
            xs, ys, yerr=yerrs,
            marker="o", linewidth=2, capsize=4,
            label=label,
        )
    else:
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)


def plot_task(
    plot_dir: str,
    task_key: str,
    title: str,
    steps: List[int],
    series: Dict[str, List[ScorePoint]],
    errorbar: str = "ci95",
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, points in series.items():
        _draw_series(ax, steps, label, points, errorbar)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("step", fontsize=12)
    ax.set_ylabel("score", fontsize=12)
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, f"{task_key}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_all_tasks_combined(
    plot_dir: str,
    steps: List[int],
    all_data: Dict[str, Dict[str, List[ScorePoint]]],
    errorbar: str = "ci95",
):
    """Single figure with subplots for all tasks."""
    tasks_with_data = {
        k: v for k, v in all_data.items()
        if any(any(pt[0] is not None for pt in pts) for pts in v.values())
    }
    n = len(tasks_with_data)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task_key, series) in enumerate(tasks_with_data.items()):
        ax = axes[idx]
        title = TASK_LOADERS.get(task_key, (task_key, None))[0]
        for label, points in series.items():
            _draw_series(ax, steps, label, points, errorbar)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("step", fontsize=10)
        ax.set_ylabel("score", fontsize=10)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for idx in range(len(tasks_with_data), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "all_tasks.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs", action="append", required=True,
        help="Repeatable: label=base_dir (model dirs: base_dir/{prefix}_step{N}/)",
    )
    parser.add_argument(
        "--steps", default="50,100,150,200,250,300,350",
        help="Comma-separated training steps",
    )
    parser.add_argument(
        "--tasks", default="ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval",
        help="Comma-separated task names",
    )
    parser.add_argument(
        "--plot-dir", default="outputs/judge_plots",
        help="Output directory for PNG plots",
    )
    parser.add_argument(
        "--name-pattern", action="append", default=None,
        help="Repeatable: label=pattern (e.g. 'biased=healthbench_step{step}'). "
             "Default: derive from label.",
    )
    parser.add_argument(
        "--show-errorbar",
        choices=["ci95", "sem", "std", "none"],
        default="ci95",
        help="Error bar kind when a summary_agg.json with N>1 is available. "
             "'ci95' uses 1.96*SEM, 'sem' uses SEM, 'std' uses raw std. "
             "'none' disables error bars even when aggregated data exists.",
    )
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    runs: Dict[str, str] = {}
    for r in args.runs:
        label, base_dir = r.split("=", 1)
        runs[label] = base_dir

    patterns: Dict[str, str] = {}
    if args.name_pattern:
        for p in args.name_pattern:
            label, pat = p.split("=", 1)
            patterns[label] = pat

    all_data: Dict[str, Dict[str, List[ScorePoint]]] = {}

    for task_key in tasks:
        if task_key not in TASK_LOADERS:
            print(f"Unknown task: {task_key}, skipping")
            continue
        title, _ = TASK_LOADERS[task_key]
        series: Dict[str, List[ScorePoint]] = {}

        for label, base_dir in runs.items():
            points: List[ScorePoint] = []
            pat = patterns.get(label, f"{label}_step{{step}}")
            for step in steps:
                model_name = pat.format(step=step)
                model_dir = os.path.join(base_dir, model_name)
                points.append(load_score_with_error(model_dir, task_key))
            series[label] = points
        all_data[task_key] = series
        plot_task(plot_dir, task_key, title, steps, series, errorbar=args.show_errorbar)

    plot_all_tasks_combined(plot_dir, steps, all_data, errorbar=args.show_errorbar)

    # Print summary table. When aggregated data is present, show mean ± yerr.
    print("\n=== Summary Table ===")
    header = f"{'Task':<20} {'Step':>6}"
    for label in runs:
        header += f" {label:>24}"
    print(header)
    print("-" * len(header))
    for task_key in tasks:
        if task_key not in all_data:
            continue
        series = all_data[task_key]
        for i, step in enumerate(steps):
            row = f"{task_key if i == 0 else '':<20} {step:>6}"
            for label in runs:
                pts = series.get(label, [(None, None, None, 0)] * len(steps))
                mean, _, _, n = pts[i]
                if mean is None:
                    row += f" {'N/A':>24}"
                else:
                    err = _yerr(pts[i], args.show_errorbar)
                    if n and n > 1 and err > 0:
                        row += f" {mean:>10.4f} ± {err:<8.4f} (n={n})".rjust(24)
                    else:
                        row += f" {mean:>24.4f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
