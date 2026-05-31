import json
import os
from typing import Any, Dict, List, Optional

from ..samplers import SamplerBase, SkipExampleError
from ..utils import (
    ThreadSafeJsonlWriter,
    ensure_dir,
    map_with_progress,
    maybe_download_nltk,
    strip_think_tags,
)
from .ifeval import (
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_loose,
    test_instruction_following_strict,
    write_outputs,
)


class IFEvalTask:
    def __init__(
        self,
        num_threads: int = 64,
    ) -> None:
        self.num_threads = num_threads

    def _compute_metrics(self, outputs: List[Any]) -> Dict[str, float]:
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0
        for example in outputs:
            prompt_total += 1
            follow_list = example.follow_instruction_list
            if all(follow_list):
                prompt_correct += 1
            instruction_total += len(follow_list)
            instruction_correct += sum(follow_list)
        return {
            "prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
            "instruction_accuracy": instruction_correct / instruction_total
            if instruction_total
            else 0.0,
            "n_prompts": prompt_total,
        }

    def run(
        self,
        sampler: SamplerBase,
        output_dir: str,
        input_path: Optional[str] = None,
        max_examples: Optional[int] = None,
        skip_nltk_download: bool = False,
        responses_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        ensure_dir(output_dir)
        maybe_download_nltk(skip_nltk_download)

        # Default input path: download from HuggingFace or use bundled data
        if input_path is None:
            input_path = self._get_default_input_path()

        inputs = read_prompt_list(input_path)
        if max_examples is not None:
            inputs = inputs[:max_examples]

        responses_path = responses_path or os.path.join(output_dir, "responses.jsonl")

        existing = {}
        if os.path.exists(responses_path):
            existing = read_prompt_to_response_dict(responses_path)

        if not existing:
            if os.path.exists(responses_path):
                os.remove(responses_path)

        # Filter pending inputs
        pending_inputs = [inp for inp in inputs if inp.prompt not in existing]

        if pending_inputs:
            writer = ThreadSafeJsonlWriter(responses_path)

            def process_one(inp) -> dict | None:
                try:
                    response = sampler(
                        [{"role": "user", "content": inp.prompt}]
                    ).response_text
                    response = strip_think_tags(response)
                    return {"prompt": inp.prompt, "response": response}
                except SkipExampleError:
                    # Skip examples blocked by content filter
                    print(
                        "Skipping prompt due to content filter violation from base model API"
                    )
                    return None

            map_with_progress(
                process_one,
                pending_inputs,
                num_threads=self.num_threads,
                desc="Generating responses",
                writer=writer,
            )

        prompt_to_response = read_prompt_to_response_dict(responses_path)
        outputs_strict = [
            test_instruction_following_strict(inp, prompt_to_response) for inp in inputs
        ]
        outputs_loose = [
            test_instruction_following_loose(inp, prompt_to_response) for inp in inputs
        ]

        strict_path = os.path.join(output_dir, "eval_results_strict.jsonl")
        loose_path = os.path.join(output_dir, "eval_results_loose.jsonl")
        write_outputs(strict_path, outputs_strict)
        write_outputs(loose_path, outputs_loose)

        summary = {
            # Model info for aggregation and comparison
            "model": getattr(sampler, "model", None),
            "strict": self._compute_metrics(outputs_strict),
            "loose": self._compute_metrics(outputs_loose),
            "responses_path": responses_path,
            "eval_results_strict": strict_path,
            "eval_results_loose": loose_path,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def _get_default_input_path(self) -> str:
        """Get default input path from bundled data."""
        # Use bundled data file
        bundled_path = os.path.join(
            os.path.dirname(__file__), "ifeval", "data", "input_data.jsonl"
        )
        if os.path.exists(bundled_path):
            return bundled_path

        raise RuntimeError(
            f"IF-EVAL data file not found at {bundled_path}\n"
            "Please provide --ifeval-input path manually."
        )
