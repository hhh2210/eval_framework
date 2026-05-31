import os
from typing import Any, Dict, Optional

from ..samplers import SamplerBase, SkipExampleError
from ..utils import (
    ThreadSafeJsonlWriter,
    ensure_dir,
    iter_jsonl,
    map_with_progress,
    safe_json_loads,
    strip_think_tags,
)
from .writingbench import calculate_scores as scores_module

# Import bundled modules
from .writingbench import prompt as prompt_module


class WritingBenchTask:
    def __init__(
        self,
        writingbench_dir: Optional[str] = None,
        num_threads: int = 64,
    ) -> None:
        # Default to bundled data directory
        if writingbench_dir is None:
            self.writingbench_dir = os.path.join(
                os.path.dirname(__file__), "writingbench"
            )
        else:
            self.writingbench_dir = writingbench_dir
        self.writingbench_dir = os.path.abspath(self.writingbench_dir)
        self.num_threads = num_threads

    def _parse_score_response(self, text: str) -> Optional[Dict[str, Any]]:
        data = safe_json_loads(text)
        if not data:
            return None
        score = data.get("score")
        reason = data.get("reason")
        if not isinstance(score, int):
            return None
        if not isinstance(reason, str):
            return None
        return {"score": score, "reason": reason}

    def run(
        self,
        sampler: SamplerBase,
        judge_sampler: SamplerBase,
        output_dir: str,
        query_file: Optional[str] = None,
        max_examples: Optional[int] = None,
        responses_path: Optional[str] = None,
        scores_path: Optional[str] = None,
        max_retries: int = 3,
        write_excel: bool = True,
        judge_only: bool = False,
        inference_only: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(output_dir)

        # Use bundled prompt module
        evaluate_system = prompt_module.evaluate_system
        evaluate_prompt = prompt_module.evaluate_prompt

        query_file = query_file or os.path.join(
            self.writingbench_dir, "benchmark_query", "benchmark_all.jsonl"
        )

        if not os.path.exists(query_file):
            raise FileNotFoundError(
                f"WritingBench query file not found at: {query_file}\n"
                "The bundled data should be at: "
                f"{os.path.join(os.path.dirname(__file__), 'writingbench', 'benchmark_query', 'benchmark_all.jsonl')}"
            )

        queries = list(iter_jsonl(query_file))
        if max_examples is not None:
            queries = queries[:max_examples]

        responses_dir = os.path.join(output_dir, "responses")
        scores_dir = os.path.join(output_dir, "scores")
        ensure_dir(responses_dir)
        ensure_dir(scores_dir)
        responses_path = responses_path or os.path.join(
            responses_dir, "responses.jsonl"
        )
        scores_path = scores_path or os.path.join(scores_dir, "scores.jsonl")

        # Phase 1: Generate responses
        existing_responses: Dict[int, str] = {}
        if os.path.exists(responses_path):
            for record in iter_jsonl(responses_path):
                existing_responses[record["index"]] = record["response"]

        pending_queries = [q for q in queries if q["index"] not in existing_responses]

        if judge_only and pending_queries:
            raise RuntimeError(
                "judge-only is set, but responses are missing. "
                f"missing_count={len(pending_queries)}; "
                f"responses_path={responses_path}"
            )

        if pending_queries:
            writer = ThreadSafeJsonlWriter(responses_path)

            def generate_one(record: dict) -> dict:
                response_text = sampler(
                    [{"role": "user", "content": record["query"]}]
                ).response_text
                response_text = strip_think_tags(response_text)
                return {"index": record["index"], "response": response_text}

            results = map_with_progress(
                generate_one,
                pending_queries,
                num_threads=self.num_threads,
                desc="Generating responses",
                writer=writer,
            )
            # Update in-memory dict with new results
            for result in results:
                existing_responses[result["index"]] = result["response"]

        if inference_only:
            print(f"[inference-only] {len(existing_responses)} responses saved, skipping scoring.")
            return {"responses_count": len(existing_responses)}

        # Phase 2: Score responses
        existing_scores: Dict[int, dict] = {}
        if os.path.exists(scores_path):
            for record in iter_jsonl(scores_path):
                existing_scores[record["index"]] = record

        # Only score queries that have responses and don't have scores yet
        pending_score = [
            q
            for q in queries
            if q["index"] not in existing_scores and q["index"] in existing_responses
        ]
        skipped = len(queries) - len(existing_scores) - len(pending_score)
        if skipped > 0:
            print(f"Warning: Skipping {skipped} queries with missing responses")

        if pending_score:
            writer = ThreadSafeJsonlWriter(scores_path)

            def score_one(record: dict) -> dict:
                idx = record["index"]
                response_text = existing_responses[idx]
                scores: Dict[str, list] = {}
                for criterion in record.get("checklist", []):
                    name = criterion["name"]
                    prompt = evaluate_prompt.format(
                        query=record["query"],
                        response=response_text,
                        criteria=criterion,
                    )
                    attempt = 0
                    parsed = None
                    try:
                        while attempt < max_retries and parsed is None:
                            messages = [
                                {"role": "system", "content": evaluate_system},
                                {"role": "user", "content": prompt},
                            ]
                            judge_response = judge_sampler(messages).response_text
                            parsed = self._parse_score_response(judge_response)
                            attempt += 1
                        if parsed is None:
                            raise ValueError(
                                f"Failed to parse score for index {idx} criterion {name}"
                            )
                        scores.setdefault(name, []).append(parsed)
                    except SkipExampleError:
                        # Skip this criterion if judge API reports content filter issues
                        print(
                            f"Skipping criterion '{name}' for index {idx} due to content filter violation"
                        )
                        continue
                # If all criteria were skipped, return None to skip this record entirely
                if not scores:
                    return None
                return {"index": idx, "scores": scores}

            map_with_progress(
                score_one,
                pending_score,
                num_threads=self.num_threads,
                desc="Scoring responses",
                writer=writer,
            )

        # Use bundled scores module for calculation
        scores_data, overall_avg, query_count, scores_data_details = (
            scores_module.read_scores_file(scores_path)
        )
        domain_data = scores_module.read_domain_file(query_file)
        domain1_avg_scores, domain2_avg_scores = scores_module.calculate_domain_scores(
            scores_data, domain_data
        )
        requirement_R, requirement_C = scores_module.read_requirement_file(
            os.path.join(self.writingbench_dir, "benchmark_query", "requirement")
        )
        requirement_R_score, requirement_C_score = (
            scores_module.calculate_requirement_scores(
                scores_data, scores_data_details, requirement_R, requirement_C
            )
        )

        summary = {
            # 模型信息，便于后续聚合和对比
            "model": getattr(sampler, "model", None),
            "judge_model": getattr(judge_sampler, "model", None),
            "overall_avg": overall_avg,
            "n_queries": query_count,
            "domain1": domain1_avg_scores,
            "domain2": domain2_avg_scores,
            "requirement_R": requirement_R_score,
            "requirement_C": requirement_C_score,
            "responses_path": responses_path,
            "scores_path": scores_path,
        }

        if write_excel:
            excel_path = os.path.join(output_dir, "scores.xlsx")
            scores_module.aggregate_scores(
                input_directory=os.path.dirname(scores_path),
                domain_file=query_file,
                output_excel_file=excel_path,
                requirement_dir=os.path.join(
                    self.writingbench_dir, "benchmark_query", "requirement"
                ),
            )
            summary["excel_path"] = excel_path

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            import json

            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
