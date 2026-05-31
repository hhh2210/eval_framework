import argparse
import json
import os
from datetime import datetime
from typing import Any, Callable, Optional

from .samplers import VLLMChatSampler
from .tasks import (
    AlpacaEvalTask,
    ArenaHardTask,
    HealthBenchTask,
    IFBenchTask,
    IFEvalTask,
    WritingBenchTask,
)

ALLOWED_TASKS = ["ifeval", "ifbench", "writingbench", "healthbench", "arena-hard", "alpaca-eval"]


def _default_output_dir(task: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", "eval", task, ts)


def build_sampler(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    max_tokens: int,
    timeout: int,
    local: bool,
    tp_size: int,
    max_model_len: Optional[int],
    gpu_mem_util: float,
    trust_remote_code: bool,
) -> VLLMChatSampler:
    return VLLMChatSampler(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        timeout=timeout,
        local=local,
        tp_size=tp_size,
        max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
        trust_remote_code=trust_remote_code,
    )


def _resolve_output_dir(base_dir: Optional[str], task: str, multiple_tasks: bool) -> str:
    if base_dir is None:
        return _default_output_dir(task)
    if multiple_tasks:
        return os.path.join(base_dir, task)
    return base_dir


def _parse_tasks(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    if args.tasks:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("One of --task or --tasks is required.")
    invalid = [task for task in tasks if task not in ALLOWED_TASKS]
    if invalid:
        parser.error(f"Unknown task(s): {', '.join(invalid)}. Choose from: {', '.join(ALLOWED_TASKS)}")
    return tasks


def _average_summaries(summaries: list[dict]) -> dict:
    """Recursively average numeric values across run summaries; take first for non-numerics."""
    if not summaries:
        return {}
    result: dict[str, Any] = {}
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(s.keys())
    for key in all_keys:
        values = [s[key] for s in summaries if key in s]
        non_null = [v for v in values if v is not None]
        if not non_null:
            result[key] = None
            continue
        sample = non_null[0]
        if isinstance(sample, (int, float)):
            numeric = [v for v in non_null if isinstance(v, (int, float))]
            result[key] = sum(numeric) / len(numeric)
        elif isinstance(sample, dict):
            dicts = [v for v in non_null if isinstance(v, dict)]
            result[key] = _average_summaries(dicts)
        else:
            result[key] = sample
    return result


def _dispatch_task(
    task_name: str,
    output_dir: str,
    args: argparse.Namespace,
    sampler: VLLMChatSampler,
    lazy_judge_sampler: Callable[[], VLLMChatSampler],
) -> dict:
    if task_name == "ifeval":
        task = IFEvalTask(num_threads=args.num_threads)
        return task.run(
            sampler=sampler,
            output_dir=output_dir,
            input_path=args.ifeval_input,
            max_examples=args.max_examples,
            skip_nltk_download=args.ifeval_skip_nltk_download,
        )
    elif task_name == "ifbench":
        task = IFBenchTask(
            ifbench_dir=args.ifbench_dir,
            num_threads=args.num_threads,
        )
        return task.run(
            sampler=sampler,
            output_dir=output_dir,
            input_path=args.ifbench_input,
            max_examples=args.max_examples,
            skip_nltk_download=args.ifbench_skip_nltk_download,
            responses_path=args.ifbench_responses,
        )
    elif task_name == "writingbench":
        js = None if args.inference_only else lazy_judge_sampler()
        task = WritingBenchTask(num_threads=args.num_threads)
        return task.run(
            sampler=sampler,
            judge_sampler=js,
            output_dir=output_dir,
            query_file=args.writingbench_query,
            max_examples=args.max_examples,
            responses_path=args.writingbench_responses,
            scores_path=args.writingbench_scores,
            write_excel=args.writingbench_write_excel,
            judge_only=args.judge_only,
            inference_only=args.inference_only,
        )
    elif task_name == "healthbench":
        js = None if args.inference_only else lazy_judge_sampler()
        task = HealthBenchTask(num_threads=args.num_threads)
        return task.run(
            sampler=sampler,
            judge_sampler=js,
            output_dir=output_dir,
            data_path=args.healthbench_data,
            max_examples=args.max_examples,
            responses_path=args.healthbench_responses,
            scores_path=args.healthbench_scores,
            judge_only=args.judge_only,
            inference_only=args.inference_only,
        )
    elif task_name == "arena-hard":
        js = None if args.inference_only else lazy_judge_sampler()
        judge_name = args.arena_hard_judge_name or args.judge_model or args.model
        task = ArenaHardTask(
            arena_hard_dir=args.arena_hard_dir,
            bench_name=args.arena_hard_benchmark,
            judge_name=judge_name,
            baseline_model=args.arena_hard_baseline_model,
            answers_dir=args.arena_hard_answers_dir,
            judgments_dir=args.arena_hard_judgments_dir,
            num_threads=args.num_threads,
        )
        return task.run(
            sampler=sampler,
            judge_sampler=js,
            output_dir=output_dir,
            model_name=args.model,
            max_examples=args.max_examples,
            judge_only=args.judge_only,
            inference_only=args.inference_only,
        )
    else:  # alpaca-eval
        js = None if args.inference_only else lazy_judge_sampler()
        task = AlpacaEvalTask(
            reference_outputs=args.alpaca_eval_reference,
            data_path=args.alpaca_eval_data,
            answers_dir=args.alpaca_eval_answers_dir,
            judgments_dir=args.alpaca_eval_judgments_dir,
            baseline_name=args.alpaca_eval_baseline_name,
            hf_dataset=args.alpaca_eval_hf_dataset,
            num_threads=args.num_threads,
        )
        return task.run(
            sampler=sampler,
            judge_sampler=js,
            output_dir=output_dir,
            model_name=args.model,
            max_examples=args.max_examples,
            judge_only=args.judge_only,
            inference_only=args.inference_only,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight eval framework")
    parser.add_argument(
        "--task",
        choices=ALLOWED_TASKS,
        help="Run a single task (use --tasks for multiple)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated list of tasks to run in order",
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key for base URL")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--local", action="store_true", help="Use local vLLM instead of server")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Only judge/score using existing responses; do not generate responses",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only generate responses; do not judge/score",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="Run evaluation N times and report averaged metrics (default: 1). "
             "Each run is stored in run_0/, run_1/, ... subdirectories for checkpoint resume support.",
    )

    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--judge-top-k", type=int, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)

    parser.add_argument("--ifeval-input", default=None, help="IF-EVAL input_data.jsonl path")
    parser.add_argument("--ifeval-skip-nltk-download", action="store_true")
    parser.add_argument("--ifbench-dir", default=None, help="Path to IFBench repo root")
    parser.add_argument("--ifbench-input", default=None, help="IFBench input jsonl path")
    parser.add_argument("--ifbench-responses", default=None, help="IFBench responses jsonl path")
    parser.add_argument("--ifbench-skip-nltk-download", action="store_true")

    parser.add_argument("--writingbench-query", default=None, help="WritingBench query jsonl path")
    parser.add_argument(
        "--writingbench-responses",
        default=None,
        help="WritingBench responses.jsonl path",
    )
    parser.add_argument(
        "--writingbench-scores",
        default=None,
        help="WritingBench scores.jsonl path",
    )
    parser.add_argument("--writingbench-write-excel", action="store_true")

    parser.add_argument("--healthbench-data", default=None, help="HealthBench eval jsonl path")
    parser.add_argument(
        "--healthbench-responses",
        default=None,
        help="HealthBench responses.jsonl path",
    )
    parser.add_argument(
        "--healthbench-scores",
        default=None,
        help="HealthBench scores.jsonl path",
    )

    parser.add_argument("--arena-hard-dir", default=None, help="Path to arena-hard-auto repo root")
    parser.add_argument("--arena-hard-benchmark", default="arena-hard-v2.0")
    parser.add_argument("--arena-hard-judge-name", default=None)
    parser.add_argument("--arena-hard-baseline-model", default=None)
    parser.add_argument("--arena-hard-answers-dir", default=None)
    parser.add_argument("--arena-hard-judgments-dir", default=None)

    parser.add_argument("--alpaca-eval-reference", default=None, help="Reference outputs jsonl path")
    parser.add_argument("--alpaca-eval-data", default=None, help="Optional instruction jsonl path")
    parser.add_argument("--alpaca-eval-baseline-name", default="text-davinci-003")
    parser.add_argument("--alpaca-eval-answers-dir", default=None)
    parser.add_argument("--alpaca-eval-judgments-dir", default=None)
    parser.add_argument("--alpaca-eval-hf-dataset", default=None)

    parser.add_argument("--num-threads", type=int, default=64, help="Number of parallel threads for pairwise tasks")

    args = parser.parse_args()
    tasks = _parse_tasks(parser, args)
    multiple_tasks = len(tasks) > 1
    if args.judge_only and args.inference_only:
        parser.error("--judge-only and --inference-only are mutually exclusive")
    if args.judge_only:
        unsupported = [
            t for t in tasks if t not in {"writingbench", "healthbench", "arena-hard", "alpaca-eval"}
        ]
        if unsupported:
            parser.error(
                f"--judge-only only supports writingbench/healthbench/arena-hard/alpaca-eval; "
                f"unsupported task(s): {', '.join(unsupported)}"
            )
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")

    sampler = build_sampler(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        local=args.local,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )

    judge_sampler = None

    def _lazy_judge_sampler():
        nonlocal judge_sampler
        if judge_sampler is None:
            judge_sampler = build_sampler(
                model=args.judge_model or args.model,
                base_url=args.judge_base_url or args.base_url,
                api_key=args.judge_api_key or args.api_key,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                top_k=args.judge_top_k,
                max_tokens=args.judge_max_tokens,
                timeout=args.timeout,
                local=args.local,
                tp_size=args.tp_size,
                max_model_len=args.max_model_len,
                gpu_mem_util=args.gpu_mem_util,
                trust_remote_code=args.trust_remote_code,
            )
        return judge_sampler

    failed_tasks: list[tuple[str, str]] = []
    for task_name in tasks:
        output_dir = _resolve_output_dir(args.output_dir, task_name, multiple_tasks)
        try:
            if args.num_runs == 1:
                summary = _dispatch_task(task_name, output_dir, args, sampler, _lazy_judge_sampler)
            else:
                run_summaries: list[dict] = []
                for run_idx in range(args.num_runs):
                    run_dir = os.path.join(output_dir, f"run_{run_idx}")
                    print(f"\n[run {run_idx + 1}/{args.num_runs}] output: {run_dir}")
                    run_summary = _dispatch_task(
                        task_name, run_dir, args, sampler, _lazy_judge_sampler
                    )
                    run_summaries.append(run_summary)
                    print(f"[run {run_idx + 1}/{args.num_runs}] done: {run_summary}")

                summary = _average_summaries(run_summaries)
                summary["num_runs"] = args.num_runs
                summary["run_dirs"] = [
                    os.path.join(output_dir, f"run_{i}") for i in range(args.num_runs)
                ]
                os.makedirs(output_dir, exist_ok=True)
                summary_path = os.path.join(output_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

        except Exception as e:
            import traceback
            # Only tolerate per-task failures when running multiple tasks so a
            # single-task CLI call still surfaces the error via a non-zero exit.
            if not multiple_tasks:
                raise
            failed_tasks.append((task_name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] Task '{task_name}' failed; continuing with remaining tasks.")
            traceback.print_exc()
            continue

        if multiple_tasks:
            print(f"Task '{task_name}' summary:")
        else:
            print("Summary:")
        print(summary)
        print(f"Outputs saved to: {output_dir}")

    if failed_tasks:
        print("")
        print(f"[WARN] {len(failed_tasks)} task(s) failed:")
        for name, err in failed_tasks:
            print(f"  - {name}: {err}")
        raise SystemExit(2)
