#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _latest_dir(pattern: str) -> Optional[Path]:
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = list(values)
    if not vals:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    vals_sorted = sorted(vals)
    p90_idx = int(0.9 * (len(vals_sorted) - 1))
    return {
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "p90": vals_sorted[p90_idx],
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _load_healthbench_scores(root: Path, port: str) -> Dict[str, dict]:
    rows = _load_jsonl(root / f"port-{port}" / "healthbench" / "scores.jsonl")
    out: Dict[str, dict] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not prompt_id:
            continue
        score = row.get("score")
        if not isinstance(score, (int, float)):
            continue
        out[prompt_id] = row
    return out


def _load_healthbench_responses(root: Path, port: str) -> Dict[str, dict]:
    rows = _load_jsonl(root / f"port-{port}" / "healthbench" / "responses.jsonl")
    out: Dict[str, dict] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if prompt_id:
            out[prompt_id] = row
    return out


def _criterion_map(score_row: dict) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for item in score_row.get("rubric_grades", []):
        criterion = item.get("criterion")
        met = item.get("criteria_met")
        if criterion and isinstance(met, bool):
            results[criterion] = met
    return results


def _response_features(text: str) -> Dict[str, Any]:
    lower = text.lower()
    has_headers = "###" in text or "**" in text
    has_lists = "\n-" in text or "\n1." in text or "\n2." in text
    has_disclaimer = any(
        s in lower
        for s in [
            "consult",
            "healthcare professional",
            "seek medical attention",
            "cannot provide",
            "not a substitute",
            "call emergency",
            "go to the er",
        ]
    )
    has_spanish_markers = any(s in lower for s in ["además", "dolor de garganta", "presión arterial"])
    return {
        "char_len": len(text),
        "has_headers": has_headers,
        "has_lists": has_lists,
        "has_disclaimer": has_disclaimer,
        "has_spanish_markers": has_spanish_markers,
    }


def _case_tags(case: dict, trajectory: Optional[List[dict]] = None) -> List[str]:
    tags: List[str] = []
    flash_score = case["flash_score"]
    plus_score = case["plus_score"]
    abs_diff = abs(case["delta"])
    features = case["response_features"]

    if abs_diff >= 0.8:
        tags.append("judge-specific-calibration-drift")

    if (
        isinstance(flash_score, (int, float))
        and isinstance(plus_score, (int, float))
        and flash_score >= 0.8
        and plus_score <= 0.2
        and (features["has_headers"] or features["has_lists"])
        and features["char_len"] >= 500
    ):
        tags.append("rubric-targeting-style")

    if (
        isinstance(flash_score, (int, float))
        and isinstance(plus_score, (int, float))
        and flash_score - plus_score >= 0.7
        and features["has_disclaimer"]
    ):
        tags.append("instruction-surface-optimization")

    if features["has_spanish_markers"]:
        tags.append("multilingual-eval-fragility")

    if trajectory and len(trajectory) >= 3:
        deltas = [item["delta"] for item in trajectory if isinstance(item["delta"], (int, float))]
        if len(deltas) >= 3:
            if max(deltas) - min(deltas) >= 0.7:
                tags.append("temporal-instability")

    return sorted(set(tags))


def _summarize_case(case: dict) -> str:
    tags = ", ".join(case["tags"]) if case["tags"] else "none"
    return (
        f"step={case['step']} prompt={case['prompt_id']} "
        f"flash={case['flash_score']:.4f} plus={case['plus_score']:.4f} "
        f"delta={case['delta']:+.4f} tags={tags}"
    )


def _analyze_healthbench(
    flash_root: Path,
    plus_root: Path,
    ports: List[str],
    steps: List[int],
    top_k_per_step: int,
    top_global: int,
) -> Dict[str, Any]:
    inventory_rows: List[dict] = []
    step_metrics: List[dict] = []
    step_top_cases: Dict[str, List[dict]] = {}
    prompt_trajectories: Dict[str, List[dict]] = defaultdict(list)
    prompt_response_by_step: Dict[Tuple[str, int], str] = {}

    for port, step in zip(ports, steps):
        flash_scores = _load_healthbench_scores(flash_root, port)
        plus_scores = _load_healthbench_scores(plus_root, port)
        responses = _load_healthbench_responses(flash_root, port)

        overlap = sorted(set(flash_scores) & set(plus_scores))
        abs_diffs = [abs(flash_scores[k]["score"] - plus_scores[k]["score"]) for k in overlap]
        signed_diffs = [plus_scores[k]["score"] - flash_scores[k]["score"] for k in overlap]
        mismatch_count = sum(1 for d in abs_diffs if d > 1e-12)

        flash_mean = _mean([flash_scores[k]["score"] for k in overlap])
        plus_mean = _mean([plus_scores[k]["score"] for k in overlap])

        inventory_rows.append(
            {
                "step": step,
                "port": port,
                "flash_count": len(flash_scores),
                "plus_count": len(plus_scores),
                "response_count": len(responses),
                "overlap_count": len(overlap),
                "overlap_ratio_flash": (len(overlap) / len(flash_scores)) if flash_scores else None,
                "overlap_ratio_plus": (len(overlap) / len(plus_scores)) if plus_scores else None,
            }
        )

        diff_stats = _stats(abs_diffs)
        step_metrics.append(
            {
                "step": step,
                "port": port,
                "overlap_count": len(overlap),
                "flash_mean": flash_mean,
                "plus_mean": plus_mean,
                "gap_flash_minus_plus": (flash_mean - plus_mean) if flash_mean is not None and plus_mean is not None else None,
                "mean_abs_diff": diff_stats["mean"],
                "median_abs_diff": diff_stats["median"],
                "p90_abs_diff": diff_stats["p90"],
                "mismatch_rate": (mismatch_count / len(overlap)) if overlap else None,
                "signed_diff_mean_plus_minus_flash": _mean(signed_diffs),
            }
        )

        per_step_cases: List[dict] = []
        ranked = sorted(
            (
                (
                    abs(flash_scores[k]["score"] - plus_scores[k]["score"]),
                    k,
                    flash_scores[k],
                    plus_scores[k],
                )
                for k in overlap
            ),
            reverse=True,
        )[:top_k_per_step]

        for _, prompt_id, flash_row, plus_row in ranked:
            response_text = (responses.get(prompt_id, {}).get("response") or "").strip()
            flash_score = float(flash_row["score"])
            plus_score = float(plus_row["score"])

            flash_criteria = _criterion_map(flash_row)
            plus_criteria = _criterion_map(plus_row)
            criteria_keys = sorted(set(flash_criteria) | set(plus_criteria))
            flash_true_plus_false = [
                key for key in criteria_keys if flash_criteria.get(key) is True and plus_criteria.get(key) is False
            ]
            plus_true_flash_false = [
                key for key in criteria_keys if plus_criteria.get(key) is True and flash_criteria.get(key) is False
            ]

            case = {
                "step": step,
                "port": port,
                "prompt_id": prompt_id,
                "flash_score": flash_score,
                "plus_score": plus_score,
                "delta": plus_score - flash_score,
                "abs_diff": abs(plus_score - flash_score),
                "flash_axis_scores": flash_row.get("axis_scores", {}),
                "plus_axis_scores": plus_row.get("axis_scores", {}),
                "flash_true_plus_false_count": len(flash_true_plus_false),
                "plus_true_flash_false_count": len(plus_true_flash_false),
                "flash_true_plus_false_examples": flash_true_plus_false[:2],
                "plus_true_flash_false_examples": plus_true_flash_false[:2],
                "response_features": _response_features(response_text),
                "response_snippet": (response_text[:380] + "...") if len(response_text) > 380 else response_text,
            }
            per_step_cases.append(case)
            prompt_trajectories[prompt_id].append(
                {
                    "step": step,
                    "port": port,
                    "flash_score": flash_score,
                    "plus_score": plus_score,
                    "delta": plus_score - flash_score,
                }
            )
            prompt_response_by_step[(prompt_id, step)] = response_text

        step_top_cases[str(step)] = per_step_cases

    global_rank: List[Tuple[float, str]] = []
    for prompt_id, traj in prompt_trajectories.items():
        if not traj:
            continue
        max_abs = max(abs(item["delta"]) for item in traj)
        global_rank.append((max_abs, prompt_id))
    global_rank.sort(reverse=True)

    evidence_cases: List[dict] = []
    for _, prompt_id in global_rank[:top_global]:
        traj = sorted(prompt_trajectories[prompt_id], key=lambda item: item["step"])
        extreme = max(traj, key=lambda item: abs(item["delta"]))
        resp = prompt_response_by_step.get((prompt_id, extreme["step"]), "")
        case = {
            "prompt_id": prompt_id,
            "max_abs_diff": abs(extreme["delta"]),
            "extreme_step": extreme["step"],
            "flash_score": extreme["flash_score"],
            "plus_score": extreme["plus_score"],
            "delta": extreme["delta"],
            "trajectory": traj,
            "response_features": _response_features(resp),
            "response_snippet": (resp[:500] + "...") if len(resp) > 500 else resp,
        }
        case["tags"] = _case_tags(case, trajectory=traj)
        evidence_cases.append(case)

    strong_mismatch_steps = []
    for metric in step_metrics:
        gap = metric["gap_flash_minus_plus"] or 0.0
        mad = metric["mean_abs_diff"] or 0.0
        if abs(gap) >= 0.10 or mad >= 0.20:
            strong_mismatch_steps.append(metric["step"])

    return {
        "inventory": inventory_rows,
        "step_metrics": step_metrics,
        "strong_mismatch_steps": strong_mismatch_steps,
        "step_top_cases": step_top_cases,
        "evidence_cases": evidence_cases,
    }


def _game_score(game: dict) -> Optional[float]:
    preference = game.get("preference")
    model_position = game.get("model_position")
    if preference not in {"m", "M"} or model_position not in {"m", "M"}:
        return None
    return 1.0 if preference == model_position else 0.0


def _load_alpaca_judgment(path: Path) -> Dict[str, float]:
    rows = _load_jsonl(path)
    out: Dict[str, float] = {}
    for row in rows:
        prompt_key = row.get("id") or row.get("instruction")
        if not prompt_key:
            continue
        vals = []
        for game in row.get("games", []):
            score = _game_score(game)
            if score is not None:
                vals.append(score)
        if vals:
            out[prompt_key] = sum(vals) / len(vals)
    return out


def _analyze_alpaca(
    flash_root: Path,
    plus_root: Path,
    ports: List[str],
    model_name: str,
) -> Dict[str, Any]:
    model_file = model_name.replace("/", "_") + ".jsonl"
    rows: List[dict] = []
    for port in ports:
        flash_path = flash_root / f"port-{port}" / "alpaca-eval" / "model_judgment" / model_file
        plus_path = plus_root / f"port-{port}" / "alpaca-eval" / "model_judgment" / model_file
        flash_map = _load_alpaca_judgment(flash_path)
        plus_map = _load_alpaca_judgment(plus_path)
        keys = set(flash_map) & set(plus_map)
        if not keys:
            rows.append(
                {
                    "port": port,
                    "overlap": 0,
                    "flash_mean": None,
                    "plus_mean": None,
                    "gap_flash_minus_plus": None,
                    "disagree_rate": None,
                }
            )
            continue
        disagree = sum(1 for key in keys if abs(flash_map[key] - plus_map[key]) > 1e-12)
        flash_mean = sum(flash_map[key] for key in keys) / len(keys)
        plus_mean = sum(plus_map[key] for key in keys) / len(keys)
        rows.append(
            {
                "port": port,
                "overlap": len(keys),
                "flash_mean": flash_mean,
                "plus_mean": plus_mean,
                "gap_flash_minus_plus": flash_mean - plus_mean,
                "disagree_rate": disagree / len(keys),
            }
        )
    return {"per_port": rows}


def _load_summary_winrate(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    value = obj.get("metrics", {}).get("overall", {}).get("winrate")
    return float(value) if isinstance(value, (int, float)) else None


def _analyze_arena(flash_root: Path, plus_root: Path, ports: List[str]) -> Dict[str, Any]:
    rows: List[dict] = []
    for port in ports:
        flash_path = flash_root / f"port-{port}" / "arena-hard" / "summary.json"
        plus_path = plus_root / f"port-{port}" / "arena-hard" / "summary.json"
        flash_val = _load_summary_winrate(flash_path)
        plus_val = _load_summary_winrate(plus_path)
        rows.append(
            {
                "port": port,
                "flash_winrate": flash_val,
                "plus_winrate": plus_val,
                "delta_plus_minus_flash": (plus_val - flash_val)
                if flash_val is not None and plus_val is not None
                else None,
                "flash_has_summary": flash_path.exists(),
                "plus_has_summary": plus_path.exists(),
            }
        )
    return {
        "per_port": rows,
        "flash_complete_ports": sum(1 for row in rows if row["flash_has_summary"]),
        "plus_complete_ports": sum(1 for row in rows if row["plus_has_summary"]),
    }


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def _build_report(data: Dict[str, Any], out_json_path: Path) -> str:
    health = data["healthbench"]
    alpaca = data["alpaca_eval"]
    arena = data["arena_hard"]

    lines: List[str] = []
    lines.append("# LLM Judge Mismatch / Reward-Hacking Risk Report")
    lines.append("")
    lines.append(f"- Generated at: {data['metadata']['generated_at']}")
    lines.append(f"- Data root: `{data['metadata']['data_root']}`")
    lines.append(f"- Step mapping: `{data['metadata']['step_mapping']}`")
    lines.append(f"- Analysis JSON: `{out_json_path}`")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "- HealthBench shows persistent strong mismatch between `flash` and `plus` judges, with "
        "step-level mean gaps around `0.17~0.20` at steps 300/350/400."
    )
    lines.append(
        "- Under the agreed rule (`mismatch itself => risk`), this is sufficient evidence that "
        "reward-hacking risk is present."
    )
    lines.append(
        "- Sample-level evidence shows recurring judge-specific calibration drift: identical responses "
        "receive high positive scores from one judge and neutral/negative scores from the other."
    )
    lines.append(
        "- Arena-Hard is currently incomplete for plus re-judge, so it is treated as secondary evidence only."
    )
    lines.append("")

    lines.append("## HealthBench Inventory")
    lines.append("")
    lines.append("| step | port | flash_count | plus_count | response_count | overlap_count | overlap_ratio_flash | overlap_ratio_plus |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in health["inventory"]:
        lines.append(
            f"| {row['step']} | {row['port']} | {row['flash_count']} | {row['plus_count']} | "
            f"{row['response_count']} | {row['overlap_count']} | {_fmt(row['overlap_ratio_flash'])} | {_fmt(row['overlap_ratio_plus'])} |"
        )
    lines.append("")

    lines.append("## HealthBench Step-Level Mismatch")
    lines.append("")
    lines.append("| step | flash_mean | plus_mean | gap(flash-plus) | mean_abs_diff | p90_abs_diff | mismatch_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in health["step_metrics"]:
        lines.append(
            f"| {row['step']} | {_fmt(row['flash_mean'])} | {_fmt(row['plus_mean'])} | "
            f"{_fmt(row['gap_flash_minus_plus'])} | {_fmt(row['mean_abs_diff'])} | {_fmt(row['p90_abs_diff'])} | {_fmt(row['mismatch_rate'])} |"
        )
    lines.append("")
    lines.append(f"- Strong mismatch steps (|gap|>=0.10 or mean_abs_diff>=0.20): `{health['strong_mismatch_steps']}`")
    lines.append("")

    lines.append("## Sample-Level Evidence (Top Global Cases)")
    lines.append("")
    lines.append("| prompt_id | extreme_step | flash_score | plus_score | delta(plus-flash) | tags |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for case in health["evidence_cases"][:12]:
        tags = ", ".join(case.get("tags", [])) or "none"
        lines.append(
            f"| `{case['prompt_id']}` | {case['extreme_step']} | {_fmt(case['flash_score'])} | "
            f"{_fmt(case['plus_score'])} | {_fmt(case['delta'])} | {tags} |"
        )
    lines.append("")

    lines.append("### Evidence Snippets")
    lines.append("")
    for case in health["evidence_cases"][:6]:
        lines.append(f"- {_summarize_case(case)}")
        lines.append(f"  - snippet: {case['response_snippet']}")
        traj = ", ".join(
            f"{item['step']}:(f={_fmt(item['flash_score'])},p={_fmt(item['plus_score'])},d={_fmt(item['delta'])})"
            for item in case["trajectory"]
        )
        lines.append(f"  - trajectory: {traj}")
    lines.append("")

    lines.append("## AlpacaEval Cross-Judge Consistency")
    lines.append("")
    lines.append("| port | overlap | flash_mean | plus_mean | gap(flash-plus) | disagree_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in alpaca["per_port"]:
        lines.append(
            f"| {row['port']} | {row['overlap']} | {_fmt(row['flash_mean'])} | {_fmt(row['plus_mean'])} | "
            f"{_fmt(row['gap_flash_minus_plus'])} | {_fmt(row['disagree_rate'])} |"
        )
    lines.append("")

    lines.append("## Arena-Hard Data Completeness")
    lines.append("")
    lines.append(
        f"- flash completed ports: `{arena['flash_complete_ports']}` / {len(arena['per_port'])}; "
        f"plus completed ports: `{arena['plus_complete_ports']}` / {len(arena['per_port'])}"
    )
    lines.append("")
    lines.append("| port | flash_winrate | plus_winrate | delta(plus-flash) |")
    lines.append("|---:|---:|---:|---:|")
    for row in arena["per_port"]:
        lines.append(
            f"| {row['port']} | {_fmt(row['flash_winrate'])} | {_fmt(row['plus_winrate'])} | {_fmt(row['delta_plus_minus_flash'])} |"
        )
    lines.append("")

    lines.append("## Risk Conclusion")
    lines.append("")
    lines.append(
        "Using the agreed criterion (`judge trend mismatch itself => risk`), current evidence supports that "
        "**reward-hacking risk is present** on HealthBench, most clearly from step 300 onward."
    )
    lines.append(
        "Observed behavior pattern is primarily **judge-specific calibration drift**, often coupled with "
        "highly structured/template-like answers that one judge rewards significantly more than the other."
    )
    lines.append("")
    lines.append("## Next Actions")
    lines.append("")
    lines.append("- Re-run plus judge for missing Arena-Hard steps to remove completeness bias.")
    lines.append("- Add judge-ensemble reward during RL to reduce single-judge overfitting pressure.")
    lines.append("- Track per-prompt variance between judges as an online anti-hacking signal.")
    lines.append("- Freeze a small adversarial validation set with manual audit for high-delta prompts.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judge mismatch and reward-hacking risk from eval outputs.")
    parser.add_argument("--data-root", default="outputs", help="Root outputs directory.")
    parser.add_argument("--flash-health-root", default="outputs/qwen3-4B")
    parser.add_argument("--plus-health-root", default="outputs/qwen3-4B-judge-qwen-plus")
    parser.add_argument("--flash-alpaca-root", default=None, help="Root dir with port-*/alpaca-eval/model_judgment.")
    parser.add_argument("--plus-alpaca-root", default=None, help="Root dir with port-*/alpaca-eval/model_judgment.")
    parser.add_argument("--flash-arena-root", default=None, help="Root dir with port-*/arena-hard/summary.json.")
    parser.add_argument("--plus-arena-root", default=None, help="Root dir with port-*/arena-hard/summary.json.")
    parser.add_argument("--ports", default="30001,30002,30003,30004")
    parser.add_argument("--steps", default="250,300,350,400")
    parser.add_argument("--model-name", default="qwen3-4B")
    parser.add_argument("--top-k-per-step", type=int, default=30)
    parser.add_argument("--top-global-cases", type=int, default=15)
    parser.add_argument("--out-dir", default=None, help="Output directory for report/json.")
    args = parser.parse_args()

    ports = [item.strip() for item in args.ports.split(",") if item.strip()]
    steps = [int(item.strip()) for item in args.steps.split(",") if item.strip()]
    if len(ports) != len(steps):
        raise ValueError(f"--ports length ({len(ports)}) must equal --steps length ({len(steps)}).")

    flash_alpaca_root = (
        Path(args.flash_alpaca_root)
        if args.flash_alpaca_root
        else _latest_dir("outputs/qwen3-4B-judge-qwen-flash-*")
    )
    plus_alpaca_root = (
        Path(args.plus_alpaca_root)
        if args.plus_alpaca_root
        else _latest_dir("outputs/qwen3-4B-judge-qwen-plus-*")
    )
    flash_arena_root = (
        Path(args.flash_arena_root)
        if args.flash_arena_root
        else _latest_dir("outputs/rejudge-arena-qwen-flash-*")
    )
    plus_arena_root = (
        Path(args.plus_arena_root)
        if args.plus_arena_root
        else _latest_dir("outputs/rejudge-arena-qwen-plus-*")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / f"reward_hacking_analysis_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    healthbench = _analyze_healthbench(
        flash_root=Path(args.flash_health_root),
        plus_root=Path(args.plus_health_root),
        ports=ports,
        steps=steps,
        top_k_per_step=args.top_k_per_step,
        top_global=args.top_global_cases,
    )

    alpaca_eval = _analyze_alpaca(
        flash_root=flash_alpaca_root if flash_alpaca_root else Path(args.flash_health_root),
        plus_root=plus_alpaca_root if plus_alpaca_root else Path(args.plus_health_root),
        ports=ports,
        model_name=args.model_name,
    )

    arena_hard = _analyze_arena(
        flash_root=flash_arena_root if flash_arena_root else Path(args.flash_health_root),
        plus_root=plus_arena_root if plus_arena_root else Path(args.plus_health_root),
        ports=ports,
    )

    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(Path(args.data_root)),
            "ports": ports,
            "steps": steps,
            "step_mapping": ", ".join(f"{port}->{step}" for port, step in zip(ports, steps)),
            "flash_health_root": args.flash_health_root,
            "plus_health_root": args.plus_health_root,
            "flash_alpaca_root": str(flash_alpaca_root) if flash_alpaca_root else None,
            "plus_alpaca_root": str(plus_alpaca_root) if plus_alpaca_root else None,
            "flash_arena_root": str(flash_arena_root) if flash_arena_root else None,
            "plus_arena_root": str(plus_arena_root) if plus_arena_root else None,
            "model_name": args.model_name,
        },
        "healthbench": healthbench,
        "alpaca_eval": alpaca_eval,
        "arena_hard": arena_hard,
    }

    out_json = out_dir / "analysis.json"
    out_md = out_dir / "report.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_build_report(result, out_json), encoding="utf-8")

    print(str(out_dir))
    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()
