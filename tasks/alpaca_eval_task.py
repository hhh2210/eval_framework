import os
import re
import time
import uuid
from functools import partial
from typing import Any, Dict, List, Optional

from ..samplers import SkipExampleError
from ..utils import (
    ThreadSafeJsonlWriter,
    ensure_dir,
    iter_jsonl,
    map_with_progress,
    strip_think_tags,
)
from .pairwise_base import PairwiseComparisonTask

# Official AlpacaEval system prompt (from alpaca_eval_clf_gpt4_turbo)
ALPACA_EVAL_SYSTEM_PROMPT = (
    "You are a highly efficient assistant, who evaluates and selects the best large language model (LLM) "
    "based on the quality of their responses to a given instruction. This process will be used to create "
    "a leaderboard reflecting the most accurate and human-preferred answers."
)

# Official AlpacaEval user prompt template (from alpaca_eval_clf_gpt4_turbo)
ALPACA_EVAL_USER_TEMPLATE = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": \"\"\"{instruction}\"\"\",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": \"\"\"{output_1}\"\"\"
    }},
    {{
        "model_identifier": "M",
        "output": \"\"\"{output_2}\"\"\"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier"""


class AlpacaEvalTask(PairwiseComparisonTask):
    def __init__(
        self,
        reference_outputs: Optional[str] = None,
        data_path: Optional[str] = None,
        answers_dir: Optional[str] = None,
        judgments_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        baseline_name: str = "text-davinci-003",
        hf_dataset: Optional[str] = None,
        num_threads: int = 64,
    ) -> None:
        self.reference_outputs = reference_outputs
        self.data_path = data_path
        self.answers_dir = answers_dir
        self.judgments_dir = judgments_dir
        self.system_prompt = system_prompt or ALPACA_EVAL_SYSTEM_PROMPT
        self.user_template = user_template or ALPACA_EVAL_USER_TEMPLATE
        self.baseline_name = baseline_name
        self.hf_dataset = hf_dataset or "alpaca_eval_gpt4_baseline"
        self.num_threads = num_threads

    def load_prompts(self, max_examples: Optional[int] = None) -> List[dict]:
        records = self._load_prompt_records()
        if max_examples is not None:
            records = records[:max_examples]
        return records

    def generate_outputs(
        self,
        sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
        judge_only: bool = False,
    ) -> str:
        answers_dir = self._answers_dir(output_dir)
        ensure_dir(answers_dir)
        answers_path = os.path.join(answers_dir, f"{self._model_filename(model_name)}.jsonl")
        answers_path = os.path.join(
            answers_dir, f"{self._model_filename(model_name)}.jsonl"
        )
        existing = self._load_key_map(answers_path)

        # Filter records that need processing
        all_records = self.load_prompts(max_examples)
        pending_records = [r for r in all_records if self._get_key(r) not in existing]

        if not pending_records:
            return answers_path
        if judge_only:
            raise RuntimeError(
                "judge-only is set, but answers are missing. "
                f"missing_count={len(pending_records)}; "
                f"answers_path={answers_path}"
            )

        writer = ThreadSafeJsonlWriter(answers_path)

        def process_one(record: dict) -> dict | None:
            prompt = self._format_instruction(record)
            try:
                response = sampler([{"role": "user", "content": prompt}]).response_text
            except SkipExampleError:
                # 跳过被内容审核拦截的样本
                print(
                    "Skipping example due to content filter violation from base model API"
                )
                return None
            response = strip_think_tags(response)
            return {
                "id": record.get("id"),
                "instruction": record.get("instruction"),
                "input": record.get("input", ""),
                "output": response,
                "dataset": record.get("dataset"),
                "model": model_name,
                "tstamp": time.time(),
                "ans_id": uuid.uuid4().hex,
            }

        map_with_progress(
            process_one,
            pending_records,
            num_threads=self.num_threads,
            desc="Generating outputs",
            writer=writer,
        )

        return answers_path

    def load_baseline_outputs(
        self, max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        records = self._load_reference_records()
        if max_examples is not None:
            records = records[:max_examples]
        baseline = {}
        for record in records:
            key = self._get_key(record)
            if key is None:
                continue
            baseline[key] = record
        return baseline

    def judge_pairs(
        self,
        judge_sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
    ) -> str:
        judgments_dir = self._judgments_dir(output_dir)
        ensure_dir(judgments_dir)
        judgments_path = os.path.join(
            judgments_dir, f"{self._model_filename(model_name)}.jsonl"
        )
        existing = self._load_key_map(judgments_path)

        answers_path = os.path.join(
            self._answers_dir(output_dir), f"{self._model_filename(model_name)}.jsonl"
        )
        if not os.path.exists(answers_path):
            raise FileNotFoundError(f"Model answers not found: {answers_path}")
        model_outputs = self._load_key_map(answers_path)

        baseline_outputs = self.load_baseline_outputs(max_examples)
        prompts = self.load_prompts(max_examples)

        # Build list of tasks to process
        pending_tasks = []
        for record in prompts:
            key = self._get_key(record)
            if key is None or key in existing:
                continue
            baseline = baseline_outputs.get(key)
            model = model_outputs.get(key)
            if baseline is None or model is None:
                continue
            pending_tasks.append(
                {
                    "record": record,
                    "baseline_text": baseline.get("output") or "",
                    "model_text": model.get("output") or "",
                }
            )
        if not pending_tasks:
            return judgments_path

        writer = ThreadSafeJsonlWriter(judgments_path)

        def process_one(task: dict) -> dict | None:
            record = task["record"]
            baseline_text = task["baseline_text"]
            model_text = task["model_text"]

            games = []
            try:
                # Round 1: baseline=m, model=M
                games.append(
                    self._judge_one(
                        judge_sampler,
                        instruction=record.get("instruction") or "",
                        output_1=baseline_text,
                        output_2=model_text,
                        model_position="M",
                    )
                )
                # Round 2: model=m, baseline=M (swapped)
                games.append(
                    self._judge_one(
                        judge_sampler,
                        instruction=record.get("instruction") or "",
                        output_1=model_text,
                        output_2=baseline_text,
                        model_position="m",
                    )
                )
            except SkipExampleError:
                # 跳过被内容审核拦截的样本
                return None

            return {
                "id": record.get("id"),
                "instruction": record.get("instruction"),
                "dataset": record.get("dataset"),
                "model": model_name,
                "baseline": self.baseline_name,
                "games": games,
            }

        map_with_progress(
            process_one,
            pending_tasks,
            num_threads=self.num_threads,
            desc="Judging pairs",
            writer=writer,
        )

        return judgments_path

    def aggregate_winrate(self, judgments_path: str) -> Dict[str, Any]:
        scores: List[float] = []
        scores_by_dataset: Dict[str, List[float]] = {}
        for record in iter_jsonl(judgments_path):
            games = record.get("games") or []
            if not games:
                continue
            for game in games:
                score = self._game_score(game)
                if score is None:
                    continue
                scores.append(score)
                dataset = record.get("dataset") or "default"
                scores_by_dataset.setdefault(dataset, []).append(score)

        mean_val, lower, upper = self._bootstrap_mean_ci(scores)
        metrics = {
            "overall": {
                "winrate": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(scores),
            },
            "by_dataset": {},
        }
        for dataset, values in scores_by_dataset.items():
            mean_val, lower, upper = self._bootstrap_mean_ci(values)
            metrics["by_dataset"][dataset] = {
                "winrate": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        return metrics

    def _judge_one(
        self,
        judge_sampler,
        instruction: str,
        output_1: str,
        output_2: str,
        model_position: str,
    ) -> dict:
        user_prompt = self.user_template.format(
            instruction=instruction,
            output_1=output_1,
            output_2=output_2,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = judge_sampler(messages).response_text
        response = strip_think_tags(response)
        preference = self._parse_preference(response)
        return {
            "preference": preference,
            "model_position": model_position,
            "judgment": {"answer": response},
        }

    @staticmethod
    def _parse_preference(text: str) -> Optional[str]:
        """Parse judge response to get preference: 'm', 'M', or None."""
        text = text.strip()
        # Official format expects just "m" or "M"
        if text in ("m", "M"):
            return text
        # Fallback: find m or M in the response
        # Look for standalone m or M (case-sensitive, as per official format)
        match = re.search(r"\b([mM])\b", text)
        if match:
            return match.group(1)
        # Check for tie indicators
        if re.search(r"\b(tie|equal|both|same)\b", text, flags=re.IGNORECASE):
            return "tie"
        return None

    def _game_score(self, game: dict) -> Optional[float]:
        """Convert preference to score where 1.0 = model wins, 0.0 = baseline wins."""
        preference = game.get("preference")
        if preference is None:
            return None
        if preference == "tie":
            return 0.5
        model_position = game.get("model_position")
        # model_position tells us where the model output was placed ("m" or "M")
        if preference == model_position:
            return 1.0  # Model wins
        return 0.0  # Baseline wins

    def _load_prompt_records(self) -> List[dict]:
        if self.data_path and os.path.exists(self.data_path):
            return list(iter_jsonl(self.data_path))
        if self.reference_outputs and os.path.exists(self.reference_outputs):
            return list(iter_jsonl(self.reference_outputs))
        return self._load_reference_records()

    def _load_reference_records(self) -> List[dict]:
        if self.reference_outputs and os.path.exists(self.reference_outputs):
            return list(iter_jsonl(self.reference_outputs))
        return self._load_bundled_or_hf_records()

    def _load_bundled_or_hf_records(self) -> List[dict]:
        """Load reference outputs from bundled data, falling back to HF."""
        import json

        bundled_path = os.path.join(
            os.path.dirname(__file__), "alpaca_eval", "data", f"{self.hf_dataset}.json"
        )
        if os.path.exists(bundled_path):
            with open(bundled_path, encoding="utf-8") as f:
                return json.load(f)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ValueError(
                f"Bundled data not found at {bundled_path} and huggingface_hub is not installed. "
                "Pass --alpaca-eval-reference or install huggingface_hub."
            ) from exc

        filename = f"{self.hf_dataset}.json"
        path = hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            filename=filename,
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _answers_dir(self, output_dir: str) -> str:
        return self.answers_dir or os.path.join(output_dir, "model_answer")

    def _judgments_dir(self, output_dir: str) -> str:
        return self.judgments_dir or os.path.join(output_dir, "model_judgment")

    @staticmethod
    def _format_instruction(record: dict) -> str:
        instruction = record.get("instruction") or ""
        input_text = record.get("input") or ""
        if input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction

    @staticmethod
    def _get_key(record: dict) -> Optional[str]:
        return record.get("id") or record.get("instruction")

    @staticmethod
    def _model_filename(model_name: str) -> str:
        return model_name.replace("/", "_")

    @staticmethod
    def _load_key_map(path: str) -> Dict[str, dict]:
        if not os.path.exists(path):
            return {}
        data = {}
        for record in iter_jsonl(path):
            key = record.get("id") or record.get("instruction")
            if key is not None:
                data[key] = record
        return data
