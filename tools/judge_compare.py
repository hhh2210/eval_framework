#!/usr/bin/env python3
import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path


def _resolve_port_dir(root, port):
    expected = f"port-{port}"
    if root.name == expected:
        return root
    if root.name.startswith("port-"):
        sibling = root.parent / expected
        if sibling.exists():
            return sibling
    candidate = root / expected
    if candidate.exists():
        return candidate
    return candidate


def _linreg(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    meanx = sum(xs) / n
    meany = sum(ys) / n
    ssx = sum((x - meanx) ** 2 for x in xs)
    if ssx == 0:
        return None
    slope = sum((x - meanx) * (y - meany) for x, y in zip(xs, ys)) / ssx
    intercept = meany - slope * meanx
    ss_tot = sum((y - meany) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def _diff_stats(a, b):
    keys = set(a) & set(b)
    if not keys:
        return {
            "overlap": 0,
            "mean_abs_diff": None,
            "median_abs_diff": None,
            "p90_abs_diff": None,
            "mean_signed_diff": None,
        }
    diffs = [abs(a[k] - b[k]) for k in keys]
    diffs_sorted = sorted(diffs)
    mean_abs = sum(diffs) / len(diffs)
    median_abs = statistics.median(diffs_sorted)
    p90_idx = int(0.9 * (len(diffs_sorted) - 1))
    p90_abs = diffs_sorted[p90_idx]
    mean_signed = sum((b[k] - a[k]) for k in keys) / len(keys)
    return {
        "overlap": len(keys),
        "mean_abs_diff": mean_abs,
        "median_abs_diff": median_abs,
        "p90_abs_diff": p90_abs,
        "mean_signed_diff": mean_signed,
    }


def _load_health(root, port):
    path = _resolve_port_dir(root, port) / "healthbench/scores.jsonl"
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        pid = obj.get("prompt_id")
        score = obj.get("score")
        if pid and isinstance(score, (int, float)):
            data[pid] = score
    return data


def _load_writing(root, port):
    path = _resolve_port_dir(root, port) / "writingbench/scores/scores.jsonl"
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        idx = obj.get("index")
        scores = obj.get("scores", {})
        vals = []
        for _, lst in scores.items():
            if isinstance(lst, list):
                for item in lst:
                    val = item.get("score")
                    if isinstance(val, (int, float)):
                        vals.append(val)
        if idx is not None and vals:
            data[idx] = sum(vals) / len(vals)
    return data


def _load_ifeval(root, port):
    path = _resolve_port_dir(root, port) / "ifeval/summary.json"
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    strict = obj.get("strict", {}).get("prompt_accuracy")
    loose = obj.get("loose", {}).get("prompt_accuracy")
    return {"strict_prompt_accuracy": strict, "loose_prompt_accuracy": loose}


def _load_pairwise_winrate(root, port, task_name):
    path = _resolve_port_dir(root, port) / task_name / "summary.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    winrate = obj.get("metrics", {}).get("overall", {}).get("winrate")
    if isinstance(winrate, (int, float)):
        return {"overall": winrate}
    return {}


def _load_alpaca_eval(root, port):
    return _load_pairwise_winrate(root, port, "alpaca-eval")


def _load_arena_hard(root, port):
    return _load_pairwise_winrate(root, port, "arena-hard")


def _per_port_ifeval_metric(ifeval_per_port, judges, ports, metric):
    per_port = {jname: {} for jname in judges}
    for port in ports:
        for jname in judges:
            entry = ifeval_per_port.get(port, {}).get(jname)
            val = None
            if entry and isinstance(entry.get(metric), (int, float)):
                val = entry.get(metric)
            per_port[jname][port] = val
    return per_port


def _per_port_avg(load_fn, judges, ports):
    per_port = {jname: {} for jname in judges}
    for port in ports:
        for jname, jroot in judges.items():
            data = load_fn(jroot, port)
            if data:
                per_port[jname][port] = sum(data.values()) / len(data)
            else:
                per_port[jname][port] = None
    return per_port


def _per_port_pair_stats(load_fn, judges, ports):
    stats = {}
    for j1, j2 in combinations(judges.keys(), 2):
        key = f"{j1}_vs_{j2}"
        stats[key] = {}
        for port in ports:
            a = load_fn(judges[j1], port)
            b = load_fn(judges[j2], port)
            stats[key][port] = _diff_stats(a, b)
    return stats


def _regressions(per_port_avg_map, ports):
    reg = {}
    for jname, port_map in per_port_avg_map.items():
        xs, ys = [], []
        for i, port in enumerate(ports, 1):
            val = port_map.get(port)
            if isinstance(val, (int, float)):
                xs.append(i)
                ys.append(val)
        reg[jname] = _linreg(xs, ys)
    return reg


def _x_axis(ports, step_start=None, step_size=50):
    if step_start is None:
        x_vals = []
        for i, port in enumerate(ports, 1):
            try:
                x_vals.append(int(port))
            except ValueError:
                x_vals.append(i)
        return x_vals, [str(p) for p in ports], "port"
    x_vals = [step_start + i * step_size for i in range(len(ports))]
    labels = [str(x) for x in x_vals]
    return x_vals, labels, "step"


def _plot_lines(out_path, title, x_vals, x_labels, x_label, series, y_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (xs, ys) in series.items():
        if xs and ys:
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3)
    if any(xs and ys for xs, ys in series.values()):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_task(
    out_dir,
    task_name,
    per_port_avg_map,
    ports,
    x_vals,
    x_labels,
    x_label,
    y_label,
    title=None,
):
    series = {}
    for jname, port_map in per_port_avg_map.items():
        xs, ys = [], []
        for i, port in enumerate(ports):
            val = port_map.get(port)
            if isinstance(val, (int, float)):
                xs.append(x_vals[i])
                ys.append(val)
        series[jname] = (xs, ys)
    if out_dir:
        out_path = Path(out_dir) / f"{task_name}.png"
        _plot_lines(
            out_path, title or task_name, x_vals, x_labels, x_label, series, y_label
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare multi-judge outputs across ports and benchmarks."
    )
    parser.add_argument(
        "--ports",
        default="30001,30002,30003,30004",
        help="Comma-separated ports, e.g. 30001,30002,30003,30004",
    )
    parser.add_argument(
        "--judges",
        action="append",
        required=True,
        help="Repeatable: name=path (path should contain port-*/...)",
    )
    parser.add_argument(
        "--out",
        default="outputs/judge_compare.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional output directory for line plots (PNG).",
    )
    parser.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="Optional step number for the first port (x-axis).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Step size between ports when --step-start is set.",
    )
    parser.add_argument(
        "--ifeval-metric",
        default="strict_prompt_accuracy",
        choices=["strict_prompt_accuracy", "loose_prompt_accuracy"],
        help="IFEval metric to plot.",
    )
    args = parser.parse_args()

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    judges = {}
    for j in args.judges:
        name, path = j.split("=", 1)
        judges[name] = Path(path)

    plot_dir = args.plot_dir
    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    x_vals, x_labels, x_label = _x_axis(
        ports, step_start=args.step_start, step_size=args.step_size
    )

    report = {
        "metadata": {
            "ports": ports,
            "judges": list(judges.keys()),
            "benchmarks": [
                "ifeval",
                "writingbench",
                "healthbench",
                "alpaca-eval",
                "arena-hard",
            ],
        }
    }

    hb_avg = _per_port_avg(_load_health, judges, ports)
    report["healthbench"] = {
        "per_port_avg": hb_avg,
        "pairwise_diff_stats": _per_port_pair_stats(_load_health, judges, ports),
        "regression": _regressions(hb_avg, ports),
    }
    _plot_task(
        plot_dir,
        "healthbench",
        hb_avg,
        ports,
        x_vals,
        x_labels,
        x_label,
        "avg score",
    )

    wb_avg = _per_port_avg(_load_writing, judges, ports)
    report["writingbench"] = {
        "per_port_avg": wb_avg,
        "pairwise_diff_stats": _per_port_pair_stats(_load_writing, judges, ports),
        "regression": _regressions(wb_avg, ports),
    }
    _plot_task(
        plot_dir,
        "writingbench",
        wb_avg,
        ports,
        x_vals,
        x_labels,
        x_label,
        "avg score",
    )

    ae_avg = _per_port_avg(_load_alpaca_eval, judges, ports)
    report["alpaca-eval"] = {
        "per_port_avg": ae_avg,
        "pairwise_diff_stats": _per_port_pair_stats(_load_alpaca_eval, judges, ports),
        "regression": _regressions(ae_avg, ports),
    }
    _plot_task(
        plot_dir,
        "alpaca-eval",
        ae_avg,
        ports,
        x_vals,
        x_labels,
        x_label,
        "winrate",
    )

    ah_avg = _per_port_avg(_load_arena_hard, judges, ports)
    report["arena-hard"] = {
        "per_port_avg": ah_avg,
        "pairwise_diff_stats": _per_port_pair_stats(_load_arena_hard, judges, ports),
        "regression": _regressions(ah_avg, ports),
    }
    _plot_task(
        plot_dir,
        "arena-hard",
        ah_avg,
        ports,
        x_vals,
        x_labels,
        x_label,
        "winrate",
    )

    ifeval = {}
    for port in ports:
        ifeval[port] = {j: _load_ifeval(judges[j], port) for j in judges}
    report["ifeval"] = {"per_port": ifeval}
    if plot_dir:
        ifeval_metric = _per_port_ifeval_metric(
            ifeval, judges, ports, args.ifeval_metric
        )
        _plot_task(
            plot_dir,
            "ifeval",
            ifeval_metric,
            ports,
            x_vals,
            x_labels,
            x_label,
            "prompt accuracy",
            title=f"ifeval ({args.ifeval_metric})",
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
