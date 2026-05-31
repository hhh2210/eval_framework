import os
from typing import Any, Dict, List, Optional

from ..samplers import SamplerBase, SkipExampleError
from ..utils import (
    ThreadSafeJsonlWriter,
    ensure_dir,
    iter_jsonl,
    map_with_progress,
    safe_json_loads,
    strip_think_tags,
)

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class HealthBenchTask:
    def __init__(
        self,
        healthbench_path: Optional[str] = None,
        num_threads: int = 64,
    ) -> None:
        # Default to bundled data file
        if healthbench_path is None:
            self.healthbench_path = os.path.join(
                os.path.dirname(__file__),
                "healthbench",
                "data",
                "healthbench_eval.jsonl",
            )
        else:
            self.healthbench_path = healthbench_path
        self.healthbench_path = os.path.abspath(self.healthbench_path)
        self.num_threads = num_threads

    @staticmethod
    def _format_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append("[non-text content]")
            return " ".join(parts)
        return str(content)

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = message.get("role", "unknown")
            content = self._format_content(message.get("content", ""))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_judge_response(self, text: str) -> Optional[Dict[str, Any]]:
        data = safe_json_loads(text)
        if not data:
            return None
        criteria_met = data.get("criteria_met")
        explanation = data.get("explanation")
        if isinstance(criteria_met, str):
            if criteria_met.lower() in {"true", "yes"}:
                criteria_met = True
            elif criteria_met.lower() in {"false", "no"}:
                criteria_met = False
        if not isinstance(criteria_met, bool):
            return None
        if not isinstance(explanation, str):
            return None
        return {"criteria_met": criteria_met, "explanation": explanation}

    def _score_rubrics(
        self, rubrics: List[Dict[str, Any]], grading: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_positive = sum(item["points"] for item in rubrics if item["points"] > 0)
        achieved = 0.0
        for item, grade in zip(rubrics, grading, strict=True):
            if grade["criteria_met"]:
                achieved += item["points"]
        score = achieved / total_positive if total_positive > 0 else None

        axis_scores: Dict[str, Optional[float]] = {}
        axis_items: Dict[str, List[tuple[float, bool]]] = {}
        for item, grade in zip(rubrics, grading, strict=True):
            for tag in item.get("tags", []):
                if isinstance(tag, str) and tag.startswith("axis:"):
                    axis = tag.split(":", 1)[1]
                    axis_items.setdefault(axis, []).append(
                        (item["points"], grade["criteria_met"])
                    )
        for axis, items in axis_items.items():
            total_pos = sum(p for p, _ in items if p > 0)
            achieved_axis = sum(p for p, met in items if met)
            axis_scores[axis] = achieved_axis / total_pos if total_pos > 0 else None

        return {"score": score, "axis_scores": axis_scores}

    def run(
        self,
        sampler: SamplerBase,
        judge_sampler: SamplerBase,
        output_dir: str,
        data_path: Optional[str] = None,
        max_examples: Optional[int] = None,
        responses_path: Optional[str] = None,
        scores_path: Optional[str] = None,
        max_retries: int = 3,
        judge_only: bool = False,
        inference_only: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(output_dir)
        data_path = data_path or self.healthbench_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"HealthBench data file not found at: {data_path}\n"
                "The bundled data should be at: "
                f"{os.path.join(os.path.dirname(__file__), 'healthbench', 'data', 'healthbench_eval.jsonl')}"
            )
        responses_path = responses_path or os.path.join(output_dir, "responses.jsonl")
        scores_path = scores_path or os.path.join(output_dir, "scores.jsonl")

        records = list(iter_jsonl(data_path))
        if max_examples is not None:
            records = records[:max_examples]

        # Phase 1: Generate responses
        existing_responses: Dict[str, str] = {}
        if os.path.exists(responses_path):
            for record in iter_jsonl(responses_path):
                existing_responses[record["prompt_id"]] = record["response"]

        pending_records = [
            r for r in records if r["prompt_id"] not in existing_responses
        ]

        if judge_only and pending_records:
            raise RuntimeError(
                "judge-only is set, but responses are missing. "
                f"missing_count={len(pending_records)}; "
                f"responses_path={responses_path}"
            )

        if pending_records:
            writer = ThreadSafeJsonlWriter(responses_path)

            def generate_one(record: dict) -> dict:
                prompt_messages = record["prompt"]
                response_text = sampler(prompt_messages).response_text
                response_text = strip_think_tags(response_text)
                return {"prompt_id": record["prompt_id"], "response": response_text}

            results = map_with_progress(
                generate_one,
                pending_records,
                num_threads=self.num_threads,
                desc="Generating responses",
                writer=writer,
            )
            for result in results:
                existing_responses[result["prompt_id"]] = result["response"]

        if inference_only:
            print(f"[inference-only] {len(existing_responses)} responses saved, skipping scoring.")
            return {"responses_count": len(existing_responses)}

        # Phase 2: Judge responses
        existing_scores: Dict[str, dict] = {}
        if os.path.exists(scores_path):
            for record in iter_jsonl(scores_path):
                existing_scores[record["prompt_id"]] = record

        # Only judge records that have responses and don't have scores yet
        pending_judge = [
            r
            for r in records
            if r["prompt_id"] not in existing_scores
            and r["prompt_id"] in existing_responses
        ]
        skipped = len(records) - len(existing_scores) - len(pending_judge)
        if skipped > 0:
            print(f"Warning: Skipping {skipped} records with missing responses")

        if pending_judge:
            writer = ThreadSafeJsonlWriter(scores_path)

            def judge_one(record: dict) -> dict | None:
                prompt_id = record["prompt_id"]
                rubrics = record.get("rubrics", [])
                conversation = list(record["prompt"])
                conversation.append(
                    {"role": "assistant", "content": existing_responses[prompt_id]}
                )
                conversation_text = self._format_conversation(conversation)

                grading_results = []
                for rubric in rubrics:
                    rubric_text = rubric.get("criterion", "")
                    prompt = GRADER_TEMPLATE.replace(
                        "<<conversation>>", conversation_text
                    ).replace("<<rubric_item>>", rubric_text)
                    parsed = None
                    attempt = 0
                    try:  # 可能存在解析错误的情况，需要额外重试
                        while attempt < max_retries and parsed is None:
                            judge_response = judge_sampler(
                                [{"role": "user", "content": prompt}]
                            ).response_text
                            parsed = self._parse_judge_response(judge_response)
                            attempt += 1
                        if parsed is None:
                            raise ValueError(
                                f"Failed to parse judge response for {prompt_id}"
                            )
                        grading_results.append(parsed)
                    except SkipExampleError:
                        # 跳过被内容审核拦截的样本
                        print(
                            f"Skipping prompt {prompt_id} due to content filter violation from judge API"
                        )
                        return None

                score_info = self._score_rubrics(rubrics, grading_results)
                return {
                    "prompt_id": prompt_id,
                    "score": score_info["score"],
                    "axis_scores": score_info["axis_scores"],
                    "rubric_grades": [
                        {
                            "criterion": rubric.get("criterion"),
                            "points": rubric.get("points"),
                            "criteria_met": grade["criteria_met"],
                            "explanation": grade["explanation"],
                        }
                        for rubric, grade in zip(rubrics, grading_results, strict=True)
                    ],
                }

            results = map_with_progress(
                judge_one,
                pending_judge,
                num_threads=self.num_threads,
                desc="Judging responses",
                writer=writer,
            )
            for result in results:
                existing_scores[result["prompt_id"]] = result

        # Aggregate metrics
        all_scores: List[float] = []
        axis_sum: Dict[str, float] = {}
        axis_count: Dict[str, int] = {}
        for record in records:
            prompt_id = record["prompt_id"]
            if prompt_id in existing_scores:
                score_val = existing_scores[prompt_id].get("score")
                if isinstance(score_val, (int, float)):
                    all_scores.append(score_val)
                for axis, value in (
                    existing_scores[prompt_id].get("axis_scores", {}).items()
                ):
                    if isinstance(value, (int, float)):
                        axis_sum[axis] = axis_sum.get(axis, 0.0) + value
                        axis_count[axis] = axis_count.get(axis, 0) + 1

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        axis_avg = {
            axis: axis_sum[axis] / axis_count[axis]
            for axis in axis_sum
            if axis_count[axis] > 0
        }
        summary = {
            # 模型信息，便于后续聚合和对比
            "model": getattr(sampler, "model", None),
            "judge_model": getattr(judge_sampler, "model", None),
            "avg_score": avg_score,
            "n_examples": len(all_scores),
            "axis_avg": axis_avg,
            "responses_path": responses_path,
            "scores_path": scores_path,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            import json

            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
