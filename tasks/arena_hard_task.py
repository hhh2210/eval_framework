import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .pairwise_base import PairwiseComparisonTask
from ..samplers import SkipExampleError
from ..utils import (
    ensure_dir,
    iter_jsonl,
    map_with_progress,
    strip_think_tags,
    ThreadSafeJsonlWriter,
)


DEFAULT_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)

DEFAULT_REGEX_PATTERNS = [
    r"\[\[([AB<>=]+)\]\]",
    r"\[([AB<>=]+)\]",
]

OG_ARENA_HARD_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the "
    "user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to "
    "evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging "
    "any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and "
    "correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
    "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one "
    "interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user "
    "than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are "
    "appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing "
    "important information in the assistants' answers that would be beneficial to include when responding to the user "
    "prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict with a "
    "label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    "Example output: \"My final verdict is tie: [[A=B]]\"."
)

JUDGE_SETTINGS = {
    "hard_prompt": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
            "to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your "
            "job is to evaluate which assistant's answer is better.\n\n"
            "When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct "
            "any mistakes or inaccurate information.\n\n"
            "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
            "correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or "
            "more than one interpretation, it is more helpful and appropriate to ask for clarifications or more "
            "information from the user than providing an answer based on assumptions. Relevant means all parts of the "
            "response closely connect or are appropriate to what is being asked. Concise means the response is clear "
            "and not verbose or excessive.\n\n"
            "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any "
            "missing important information in the assistants' answers that would be beneficial to include when "
            "responding to the user prompt.\n\n"
            "After providing your explanation, you must output only one of the following choices as your final verdict "
            "with a label:\n\n"
            "1. Assistant A is significantly better: [[A>>B]]\n"
            "2. Assistant A is slightly better: [[A>B]]\n"
            "3. Tie, relatively the same: [[A=B]]\n"
            "4. Assistant B is slightly better: [[B>A]]\n"
            "5. Assistant B is significantly better: [[B>>A]]\n\n"
            "Example output: \"My final verdict is tie: [[A=B]]\"."
        ),
    },
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}


class ArenaHardTask(PairwiseComparisonTask):
    def __init__(
        self,
        arena_hard_dir: Optional[str] = None,
        bench_name: str = "arena-hard-v2.0",
        judge_name: str = "gpt-4.1",
        baseline_model: Optional[str] = None,
        answers_dir: Optional[str] = None,
        judgments_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        regex_patterns: Optional[List[str]] = None,
        weight: int = 3,
        num_threads: int = 64,
    ) -> None:
        if arena_hard_dir is None:
            bundled = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "arena_hard")
            )
            repo_external = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", ".external", "arena-hard-auto")
            )
            legacy_external = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", ".external", "arena-hard-auto")
            )
            if os.path.exists(os.path.join(bundled, "data", bench_name, "question.jsonl")):
                base_dir = bundled
            elif os.path.exists(repo_external):
                base_dir = repo_external
            else:
                base_dir = legacy_external
        else:
            base_dir = os.path.abspath(arena_hard_dir)
        data_dir = os.path.join(base_dir, "data")

        self.bench_name = bench_name
        self.judge_name = judge_name
        self.baseline_model = baseline_model
        self.weight = weight
        self.num_threads = num_threads
        self.questions_path = os.path.join(data_dir, bench_name, "question.jsonl")
        self._explicit_answers_dir = answers_dir
        self._repo_answers_dir = os.path.join(data_dir, bench_name, "model_answer")
        self._explicit_judgments_dir = judgments_dir
        judge_dir_name = self._model_filename(judge_name)
        self._repo_judgments_dir = os.path.join(data_dir, bench_name, "model_judgment", judge_dir_name)
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.regex_patterns = regex_patterns or DEFAULT_REGEX_PATTERNS

    def load_prompts(self, max_examples: Optional[int] = None) -> List[dict]:
        prompts = list(iter_jsonl(self.questions_path))
        if max_examples is not None:
            prompts = prompts[:max_examples]
        return prompts

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
        existing = self._load_jsonl_map(answers_path, "uid")

        # Filter questions that need processing
        all_questions = self.load_prompts(max_examples)
        pending_questions = [q for q in all_questions if q["uid"] not in existing]
        n_skip = len(all_questions) - len(pending_questions)
        if n_skip > 0:
            print(f"[arena-hard] generate_outputs: skip {n_skip} questions (already in {answers_path})")
        if not pending_questions:
            print(f"[arena-hard] generate_outputs: all {len(all_questions)} questions already have answers, skip generation")
            return answers_path
        if judge_only:
            raise RuntimeError(
                "judge-only is set, but answers are missing. "
                f"missing_count={len(pending_questions)}; "
                f"answers_path={answers_path}"
            )

        writer = ThreadSafeJsonlWriter(answers_path)

        def process_one(question: dict) -> dict:
            response = sampler([{"role": "user", "content": question["prompt"]}]).response_text
            response = strip_think_tags(response)
            return {
                "uid": question["uid"],
                "ans_id": uuid.uuid4().hex,
                "model": model_name,
                "messages": [
                    {"role": "user", "content": question["prompt"]},
                    {"role": "assistant", "content": {"answer": response}},
                ],
                "tstamp": time.time(),
                "metadata": {},
            }

        map_with_progress(
            process_one,
            pending_questions,
            num_threads=self.num_threads,
            desc="Generating outputs",
            writer=writer,
        )

        return answers_path

    def load_baseline_outputs(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        prompts = self.load_prompts(max_examples)
        baseline_models = {self._get_baseline_model(p.get("category")) for p in prompts}
        baseline_maps: Dict[str, Dict[str, Any]] = {}
        for model in baseline_models:
            path = os.path.join(self._repo_answers_dir, f"{self._model_filename(model)}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Baseline answers not found: {path}")
            baseline_maps[model] = self._load_jsonl_map(path, "uid")

        baseline_by_uid: Dict[str, Any] = {}
        for question in prompts:
            uid = question["uid"]
            model = self._get_baseline_model(question.get("category"))
            record = baseline_maps.get(model, {}).get(uid)
            if record is not None:
                baseline_by_uid[uid] = record
        return baseline_by_uid

    def judge_pairs(
        self,
        judge_sampler,
        output_dir: str,
        model_name: str,
        max_examples: Optional[int] = None,
    ) -> str:
        judgments_dir = self._judgments_dir(output_dir)
        ensure_dir(judgments_dir)
        judgments_path = os.path.join(judgments_dir, f"{self._model_filename(model_name)}.jsonl")
        existing = self._load_jsonl_map(judgments_path, "uid")
        answers_path = os.path.join(self._answers_dir(output_dir), f"{self._model_filename(model_name)}.jsonl")
        if not os.path.exists(answers_path):
            raise FileNotFoundError(f"Model answers not found: {answers_path}")

        model_answers = self._load_jsonl_map(answers_path, "uid")
        baseline_answers = self.load_baseline_outputs(max_examples)
        prompts = self.load_prompts(max_examples)

        # Build list of tasks to process
        pending_tasks = []
        n_skip_existing = 0
        n_skip_no_record = 0
        n_skip_no_answer = 0
        for question in prompts:
            uid = question["uid"]
            if uid in existing:
                n_skip_existing += 1
                continue
            model_record = model_answers.get(uid)
            baseline_record = baseline_answers.get(uid)
            if model_record is None or baseline_record is None:
                n_skip_no_record += 1
                continue
            model_answer = self._extract_answer(model_record)
            baseline_answer = self._extract_answer(baseline_record)
            if model_answer is None or baseline_answer is None:
                n_skip_no_answer += 1
                continue
            pending_tasks.append({
                "question": question,
                "model_answer": model_answer,
                "baseline_answer": baseline_answer,
            })
        if n_skip_existing > 0:
            print(f"[arena-hard] judge_pairs: skip {n_skip_existing} questions (judgment already in {judgments_path})")
        if n_skip_no_record > 0:
            print(f"[arena-hard] judge_pairs: skip {n_skip_no_record} questions (missing model or baseline record)")
        if n_skip_no_answer > 0:
            print(f"[arena-hard] judge_pairs: skip {n_skip_no_answer} questions (empty model or baseline answer)")
        if not pending_tasks:
            print(f"[arena-hard] judge_pairs: all {len(prompts)} questions already judged or skipped, skip judging")
            return judgments_path

        writer = ThreadSafeJsonlWriter(judgments_path)

        def process_one(task: dict) -> dict | None:
            question = task["question"]
            model_answer = task["model_answer"]
            baseline_answer = task["baseline_answer"]

            games: List[dict] = []
            try:
                # Round 1: baseline=A, model=B
                games.append(
                    self._judge_one(
                        judge_sampler,
                        question=question,
                        answer_a=baseline_answer,
                        answer_b=model_answer,
                    )
                )
                # Round 2: model=A, baseline=B (swapped)
                games.append(
                    self._judge_one(
                        judge_sampler,
                        question=question,
                        answer_a=model_answer,
                        answer_b=baseline_answer,
                    )
                )
            except SkipExampleError:
                # If the judge API reports content filter issues (e.g., data_inspection_failed),
                # skip this example entirely and do not write any judgment record.
                print(f"Skipping question {question['uid']} due to content filter violation from judge API")
                return None

            return {
                "uid": question["uid"],
                "category": question.get("category"),
                "judge": self.judge_name,
                "model": model_name,
                "baseline": self._get_baseline_model(question.get("category")),
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
        label_to_score = {
            "A>B": [1],
            "A>>B": [1] * self.weight,
            "A=B": [0.5],
            "A<<B": [0] * self.weight,
            "A<B": [0],
            "B>A": [0],
            "B>>A": [0] * self.weight,
            "B=A": [0.5],
            "B<<A": [1] * self.weight,
            "B<A": [1],
        }

        scores: List[float] = []
        scores_by_category: Dict[str, List[float]] = {}
        for record in iter_jsonl(judgments_path):
            games = record.get("games") or []
            if len(games) < 2:
                continue
            game_a = games[0]
            game_b = games[1]
            if not game_a or not game_b:
                continue
            score_a = (game_a.get("score") or "").upper()
            score_b = (game_b.get("score") or "").upper()
            if score_a not in label_to_score or score_b not in label_to_score:
                continue
            values = label_to_score[score_b] + [1 - s for s in label_to_score[score_a]]
            scores.extend(values)
            category = record.get("category") or "unknown"
            scores_by_category.setdefault(category, []).extend(values)

        mean_val, lower, upper = self._bootstrap_mean_ci(scores)
        metrics = {
            "overall": {
                "winrate": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(scores),
            },
            "by_category": {},
            "weight": self.weight,
        }

        for category, values in scores_by_category.items():
            mean_val, lower, upper = self._bootstrap_mean_ci(values)
            metrics["by_category"][category] = {
                "winrate": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        return metrics

    def _judge_one(self, judge_sampler, question: dict, answer_a: str, answer_b: str) -> dict:
        prompt_args = {
            "QUESTION": question["prompt"],
            "ANSWER_A": answer_a,
            "ANSWER_B": answer_b,
        }
        user_prompt = self.prompt_template.format(**prompt_args)
        system_prompt = self._get_system_prompt(question.get("category"))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = judge_sampler(messages).response_text
        response = strip_think_tags(response)
        score = self._extract_score(response)
        return {
            "score": score,
            "judgment": {"answer": response},
        }

    def _extract_score(self, text: str) -> Optional[str]:
        for pattern in self.regex_patterns:
            matches = re.findall(pattern, text.upper())
            matches = [m for m in matches if m]
            if matches:
                return matches[-1].strip("\n")
        return None

    def _get_baseline_model(self, category: Optional[str]) -> str:
        if self.baseline_model:
            return self.baseline_model
        if category in JUDGE_SETTINGS:
            return JUDGE_SETTINGS[category]["baseline"]
        raise ValueError(f"Unknown category for baseline: {category}")

    def _get_system_prompt(self, category: Optional[str]) -> str:
        if category in JUDGE_SETTINGS:
            return JUDGE_SETTINGS[category]["system_prompt"]
        return OG_ARENA_HARD_PROMPT

    def _answers_dir(self, output_dir: str) -> str:
        return self._explicit_answers_dir or os.path.join(output_dir, "model_answer")

    def _judgments_dir(self, output_dir: str) -> str:
        return self._explicit_judgments_dir or os.path.join(output_dir, "model_judgment", self._model_filename(self.judge_name))

    @staticmethod
    def _model_filename(model_name: str) -> str:
        return model_name.replace("/", "_")

    @staticmethod
    def _extract_answer(record: dict) -> Optional[str]:
        messages = record.get("messages") or []
        if not messages:
            return None
        content = messages[-1].get("content")
        if isinstance(content, dict):
            return content.get("answer") or content.get("content")
        if isinstance(content, str):
            return content
        return None
