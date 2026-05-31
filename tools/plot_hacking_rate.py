#!/usr/bin/env python3
"""Plot reward-hacking report rates emitted by log_monitor_agent.py."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_group_step(dirname: str, group_patterns: list[str]) -> tuple[str, int] | None:
    for item in group_patterns:
        if "=" not in item:
            raise ValueError(f"group pattern must be label=prefix, got: {item}")
        label, prefix = item.split("=", 1)
        if dirname.startswith(prefix):
            suffix = dirname.removeprefix(prefix)
            try:
                return label, int(suffix)
            except ValueError:
                return None
    return None


def load_report_rates(base_dir: Path, group_patterns: list[str]) -> pd.DataFrame:
    data = []

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        parsed = _parse_group_step(model_dir.name, group_patterns)
        if parsed is None:
            continue
        group, step = parsed

        for report_path in model_dir.glob("hacking_report_*.md"):
            task = report_path.name.replace("hacking_report_", "").replace(".md", "")
            text = report_path.read_text(encoding="utf-8")

            scanned_match = re.search(r"\*\*Cases Scanned\*\*: (\d+)", text)
            detected_match = re.search(r"\*\*Hacking Cases Detected\*\*: (\d+)", text)
            if not scanned_match or not detected_match:
                continue

            scanned = int(scanned_match.group(1))
            detected = int(detected_match.group(1))
            data.append(
                {
                    "Group": group,
                    "Step": step,
                    "Task": task,
                    "Hacking Rate": detected / scanned if scanned > 0 else 0,
                }
            )

    return pd.DataFrame(data)


def plot_rates(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        raise ValueError("no hacking_report_*.md data found for the requested patterns")

    plt.style.use("ggplot")
    tasks = sorted(df["Task"].unique())
    cols = 2
    rows = max(1, (len(tasks) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.8 * rows))
    axes = list(getattr(axes, "flatten", lambda: [axes])())

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = df[df["Task"] == task]

        for group in task_df["Group"].unique():
            group_df = task_df[task_df["Group"] == group].sort_values(by="Step")
            ax.plot(
                group_df["Step"],
                group_df["Hacking Rate"],
                marker="o",
                markersize=7,
                linewidth=2,
                label=group,
            )

        ax.set_title(f"Reward Hacking Rate: {task}", fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_ylabel("Meta-Judge Hacking Detection Rate", fontsize=11)
        ax.set_xlabel("Training Step", fontsize=11)
        if i == 0:
            ax.legend(title="Training Group", fontsize=10, title_fontsize=11)

    for ax in axes[len(tasks) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="outputs/eval_outputs")
    parser.add_argument("--output", default="outputs/judge_plots/hacking_rate_comparison.png")
    parser.add_argument(
        "--group-pattern",
        action="append",
        default=["biased=healthbench_step", "normal=root_step"],
        help="Repeatable label=directory_prefix mapping, e.g. biased=healthbench_step",
    )
    args = parser.parse_args()

    df = load_report_rates(Path(args.base_dir), args.group_pattern)
    plot_rates(df, Path(args.output))
    print(f"Plot successfully saved to: {os.fspath(args.output)}")


if __name__ == "__main__":
    main()
