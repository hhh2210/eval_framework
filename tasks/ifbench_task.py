import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..samplers import SamplerBase, SkipExampleError
from ..utils import (
    ThreadSafeJsonlWriter,
    ensure_dir,
    map_with_progress,
    maybe_download_nltk_resources,
    strip_think_tags,
)

IFBENCH_MODULES = (
    "instructions_util",
    "instructions",
    "instructions_registry",
    "evaluation_lib",
)

IFBENCH_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

IFBENCH_PYTHON_DEPS = ("emoji", "syllapy")


def _load_registered_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class IFBenchTask:
    def __init__(
        self,
        ifbench_dir: Optional[str] = None,
        num_threads: int = 64,
    ) -> None:
        if ifbench_dir is None:
            repo_external = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", ".external", "IFBench")
            )
            legacy_external = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", ".external", "IFBench")
            )
            self.ifbench_dir = repo_external if os.path.exists(repo_external) else legacy_external
        else:
            self.ifbench_dir = os.path.abspath(ifbench_dir)

        self.num_threads = num_threads
        self._loaded_modules: Optional[Dict[str, Any]] = None
        self._evaluation_lib = None

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

    def _ensure_python_deps(self) -> None:
        missing = [
            package
            for package in IFBENCH_PYTHON_DEPS
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise RuntimeError(
                "IFBench requires additional Python packages that are not installed: "
                f"{missing_text}. Install them with `uv pip install {missing_text}` "
                "inside evaluation/eval_framework/.venv."
            )

    def _load_ifbench_evaluation_lib(self):
        if self._evaluation_lib is not None:
            return self._evaluation_lib

        missing_files = [
            os.path.join(self.ifbench_dir, f"{module_name}.py")
            for module_name in IFBENCH_MODULES
            if not os.path.exists(os.path.join(self.ifbench_dir, f"{module_name}.py"))
        ]
        if missing_files:
            raise FileNotFoundError(
                "IFBench source files are missing. Clone `https://github.com/allenai/IFBench` "
                f"into `{self.ifbench_dir}` or pass `--ifbench-dir`."
            )

        self._ensure_python_deps()
        saved_modules = {name: sys.modules.get(name) for name in IFBENCH_MODULES}
        loaded_modules: Dict[str, Any] = {}
        try:
            for name in IFBENCH_MODULES:
                sys.modules.pop(name, None)
            for name in IFBENCH_MODULES:
                loaded_modules[name] = _load_registered_module(
                    name, os.path.join(self.ifbench_dir, f"{name}.py")
                )
        finally:
            for name, original in saved_modules.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self._loaded_modules = loaded_modules
        self._evaluation_lib = loaded_modules["evaluation_lib"]
        return self._evaluation_lib

    def _get_default_input_path(self) -> str:
        bundled_path = os.path.join(
            os.path.dirname(__file__),
            "ifbench",
            "data",
            "IFBench_test.jsonl",
        )
        if os.path.exists(bundled_path):
            return bundled_path
        raise RuntimeError(
            f"IFBench data file not found at {bundled_path}\n"
            "Please provide `--ifbench-input` manually."
        )

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

        missing_nltk = maybe_download_nltk_resources(
            skip_nltk_download,
            IFBENCH_NLTK_RESOURCES,
        )
        if missing_nltk:
            missing_text = ", ".join(sorted(set(missing_nltk)))
            raise RuntimeError(
                "IFBench requires NLTK resources that are not available: "
                f"{missing_text}. Download them first or rerun without "
                "`--ifbench-skip-nltk-download`."
            )

        evaluation_lib = self._load_ifbench_evaluation_lib()
        input_path = input_path or self._get_default_input_path()
        inputs = evaluation_lib.read_prompt_list(input_path)
        if max_examples is not None:
            inputs = inputs[:max_examples]

        responses_path = responses_path or os.path.join(output_dir, "responses.jsonl")
        existing = {}
        if os.path.exists(responses_path):
            existing = evaluation_lib.read_prompt_to_response_dict(responses_path)
        if not existing and os.path.exists(responses_path):
            os.remove(responses_path)

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
                    print(
                        "Skipping prompt due to content filter violation from base model API"
                    )
                    return None

            map_with_progress(
                process_one,
                pending_inputs,
                num_threads=self.num_threads,
                desc="Generating IFBench responses",
                writer=writer,
            )

        prompt_to_response = evaluation_lib.read_prompt_to_response_dict(responses_path)

        # Upstream IFBench checkers occasionally raise (e.g. WordsPositionChecker
        # has a known off-by-one on short responses introduced by commit 86ee248).
        # We collapse any per-prompt checker crash into "did not follow", which is
        # the semantically correct outcome, and log it for diagnosis. This lets
        # one broken response avoid poisoning the whole eval run, and removes the
        # need to patch vendored code in .external/IFBench. See upstream issue:
        # https://github.com/allenai/IFBench/issues/22
        def _safe_eval(fn, inp):
            try:
                return fn(inp, prompt_to_response)
            except Exception as e:  # noqa: BLE001 — deliberately broad
                n = len(inp.instruction_id_list)
                print(
                    f"[ifbench] WARN: {fn.__name__} crashed on prompt "
                    f"(ids={inp.instruction_id_list}): {type(e).__name__}: {e}. "
                    f"Marking all {n} instruction(s) as not followed."
                )
                OutputExample = evaluation_lib.OutputExample
                response = prompt_to_response.get(inp.prompt, "") or ""
                return OutputExample(
                    instruction_id_list=inp.instruction_id_list,
                    prompt=inp.prompt,
                    response=response,
                    follow_all_instructions=False,
                    follow_instruction_list=[False] * n,
                )

        outputs_strict = [
            _safe_eval(evaluation_lib.test_instruction_following_strict, inp)
            for inp in inputs
        ]
        outputs_loose = [
            _safe_eval(evaluation_lib.test_instruction_following_loose, inp)
            for inp in inputs
        ]

        strict_path = os.path.join(output_dir, "eval_results_strict.jsonl")
        loose_path = os.path.join(output_dir, "eval_results_loose.jsonl")
        evaluation_lib.write_outputs(strict_path, outputs_strict)
        evaluation_lib.write_outputs(loose_path, outputs_loose)

        summary = {
            "model": getattr(sampler, "model", None),
            "strict": self._compute_metrics(outputs_strict),
            "loose": self._compute_metrics(outputs_loose),
            "responses_path": responses_path,
            "eval_results_strict": strict_path,
            "eval_results_loose": loose_path,
            "ifbench_dir": self.ifbench_dir,
            "input_path": input_path,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
