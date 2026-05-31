import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..utils import append_jsonl, ensure_dir, iter_jsonl


class PairwiseComparisonTask(ABC):
    @abstractmethod
    def load_prompts(self, max_examples: Optional[int] = None) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def generate_outputs(
        self,
        sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
        judge_only: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_baseline_outputs(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def judge_pairs(
        self,
        judge_sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def aggregate_winrate(self, judgments_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def run(
        self,
        sampler,
        judge_sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
        judge_only: bool = False,
        inference_only: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(output_dir)
        answers_path = self.generate_outputs(
            sampler=sampler,
            output_dir=output_dir,
            model_name=model_name,
            max_examples=max_examples,
            judge_only=judge_only,
        )
        if inference_only:
            print(f"[inference-only] Answers saved to {answers_path}, skipping judging.")
            return {"answers_path": answers_path}
        judgments_path = self.judge_pairs(
            judge_sampler=judge_sampler,
            output_dir=output_dir,
            model_name=model_name,
            max_examples=max_examples,
        )
        metrics = self.aggregate_winrate(judgments_path)
        summary = {
            "model_name": model_name,
            "judge_model": getattr(judge_sampler, "model", None),
            "answers_path": answers_path,
            "judgments_path": judgments_path,
            "metrics": metrics,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    @staticmethod
    def _load_jsonl_map(path: str, key_field: str) -> Dict[Any, dict]:
        data = {}
        if not os.path.exists(path):
            return data
        for record in iter_jsonl(path):
            if key_field in record:
                data[record[key_field]] = record
        return data

    @staticmethod
    def _bootstrap_mean_ci(
        values: Iterable[float],
        num_rounds: int = 200,
        alpha: float = 0.05,
        seed: int = 123,
    ) -> Tuple[float, float, float]:
        values = list(values)
        if not values:
            return 0.0, 0.0, 0.0
        rng = random.Random(seed)
        n = len(values)
        means = []
        for _ in range(num_rounds):
            sample = [values[rng.randrange(0, n)] for _ in range(n)]
            means.append(float(np.mean(sample)))
        means.sort()
        lower_idx = int((alpha / 2) * len(means))
        upper_idx = int((1 - alpha / 2) * len(means)) - 1
        mean_val = float(np.mean(values))
        return mean_val, float(means[lower_idx]), float(means[upper_idx])
