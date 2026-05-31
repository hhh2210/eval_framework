"""Smoke tests for the bundled Arena-Hard data path.

Verifies that after vendoring arena-hard-v2.0 questions + baselines under
`tasks/arena_hard/data/`, ArenaHardTask works fully offline without the
`.external/arena-hard-auto` repo.

Run directly:
    python tests/test_arena_hard_bundled.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_importable() -> None:
    # Make `eval_framework` importable whether the venv has it installed or not.
    sys.path.insert(0, str(_repo_root().parent))


_ensure_importable()

from eval_framework.tasks.arena_hard_task import ArenaHardTask  # noqa: E402


BUNDLED_DATA = _repo_root() / "tasks" / "arena_hard" / "data" / "arena-hard-v2.0"


def test_bundled_files_exist() -> None:
    assert (BUNDLED_DATA / "question.jsonl").is_file(), "missing bundled question.jsonl"
    assert (BUNDLED_DATA / "model_answer" / "o3-mini-2025-01-31.jsonl").is_file()
    assert (BUNDLED_DATA / "model_answer" / "gemini-2.0-flash-001.jsonl").is_file()
    print("[ok] bundled files exist")


def test_default_construction_resolves_to_bundled() -> None:
    task = ArenaHardTask()
    expected = str(BUNDLED_DATA / "question.jsonl")
    assert os.path.abspath(task.questions_path) == expected, (
        f"default questions_path {task.questions_path!r} != bundled {expected!r}"
    )
    assert os.path.isdir(task._repo_answers_dir), (
        f"baseline dir not found: {task._repo_answers_dir}"
    )
    print(f"[ok] default construction resolves to bundled: {task.questions_path}")


def test_load_prompts_and_baselines() -> None:
    task = ArenaHardTask()
    prompts = task.load_prompts()
    assert len(prompts) == 750, f"expected 750 prompts, got {len(prompts)}"
    categories = {p.get("category") for p in prompts}
    assert categories == {"hard_prompt", "creative_writing"}, (
        f"unexpected categories: {categories}"
    )
    for p in prompts[:3]:
        assert "uid" in p and "prompt" in p and "category" in p
    baselines = task.load_baseline_outputs()
    assert len(baselines) == 750, (
        f"expected 750 baseline entries, got {len(baselines)}"
    )
    missing = [p["uid"] for p in prompts if p["uid"] not in baselines]
    assert not missing, f"baseline missing for {len(missing)} prompts (e.g. {missing[:3]})"
    print(f"[ok] loaded {len(prompts)} prompts with full baseline coverage")


def test_max_examples_truncation() -> None:
    task = ArenaHardTask()
    prompts = task.load_prompts(max_examples=5)
    assert len(prompts) == 5
    baselines = task.load_baseline_outputs(max_examples=5)
    assert len(baselines) == 5
    print("[ok] max_examples truncation works")


def test_score_extraction() -> None:
    task = ArenaHardTask()
    cases = [
        ("My final verdict is: [[A>>B]]", "A>>B"),
        ("The better answer is [[B>A]].", "B>A"),
        ("I say [A=B] overall.", "A=B"),
        ("[[garbage]] then [[A<<B]]", "A<<B"),
        ("no verdict here", None),
    ]
    for text, expected in cases:
        got = task._extract_score(text)
        assert got == expected, f"_extract_score({text!r}) => {got!r}, expected {expected!r}"
    print("[ok] _extract_score handles all patterns")


def test_extract_answer() -> None:
    rec_dict = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": {"answer": "hello"}}]}
    rec_str = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "hello"}]}
    assert ArenaHardTask._extract_answer(rec_dict) == "hello"
    assert ArenaHardTask._extract_answer(rec_str) == "hello"
    assert ArenaHardTask._extract_answer({"messages": []}) is None
    print("[ok] _extract_answer handles dict and str content")


def test_baseline_routing_by_category() -> None:
    task = ArenaHardTask()
    assert task._get_baseline_model("hard_prompt") == "o3-mini-2025-01-31"
    assert task._get_baseline_model("coding") == "o3-mini-2025-01-31"
    assert task._get_baseline_model("math") == "o3-mini-2025-01-31"
    assert task._get_baseline_model("creative_writing") == "gemini-2.0-flash-001"
    assert task._get_baseline_model("arena-hard-v0.1") == "gpt-4-0314"
    print("[ok] baseline routing matches JUDGE_SETTINGS")


def test_aggregate_winrate_on_synthetic() -> None:
    task = ArenaHardTask()
    prompts = task.load_prompts(max_examples=4)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        path = f.name
        for i, p in enumerate(prompts):
            score_a = "A>>B" if i % 2 == 0 else "A<<B"
            score_b = "B>>A" if i % 2 == 0 else "B<<A"
            record = {
                "uid": p["uid"],
                "category": p.get("category", "unknown"),
                "judge": "synthetic",
                "model": "m",
                "baseline": "b",
                "games": [
                    {"score": score_a, "judgment": {"answer": ""}},
                    {"score": score_b, "judgment": {"answer": ""}},
                ],
            }
            f.write(json.dumps(record) + "\n")
    try:
        metrics = task.aggregate_winrate(path)
    finally:
        os.unlink(path)
    assert "overall" in metrics and "winrate" in metrics["overall"]
    assert 0.0 <= metrics["overall"]["winrate"] <= 1.0
    assert metrics["overall"]["n"] > 0
    print(
        f"[ok] aggregate_winrate on synthetic → "
        f"winrate={metrics['overall']['winrate']:.3f}, n={metrics['overall']['n']}"
    )


def test_override_arena_hard_dir(tmp_dir: Path | None = None) -> None:
    """Passing --arena-hard-dir pointing at a custom location still works."""
    with tempfile.TemporaryDirectory() as tmp:
        custom = Path(tmp)
        bench = "arena-hard-v2.0"
        (custom / "data" / bench / "model_answer").mkdir(parents=True)
        src_q = BUNDLED_DATA / "question.jsonl"
        (custom / "data" / bench / "question.jsonl").write_text(
            src_q.read_text(encoding="utf-8"), encoding="utf-8"
        )
        for name in ("o3-mini-2025-01-31.jsonl", "gemini-2.0-flash-001.jsonl"):
            src = BUNDLED_DATA / "model_answer" / name
            (custom / "data" / bench / "model_answer" / name).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )
        task = ArenaHardTask(arena_hard_dir=str(custom))
        assert str(custom) in task.questions_path
        prompts = task.load_prompts(max_examples=3)
        assert len(prompts) == 3
        baselines = task.load_baseline_outputs(max_examples=3)
        assert len(baselines) == 3
    print("[ok] --arena-hard-dir override still works")


def test_cli_exposes_arena_hard_dir() -> None:
    cli_src = (_repo_root() / "cli.py").read_text(encoding="utf-8")
    assert '"--arena-hard-dir"' in cli_src, "cli.py no longer exposes --arena-hard-dir"
    assert "arena_hard_dir=args.arena_hard_dir" in cli_src, (
        "cli.py no longer plumbs arena_hard_dir into ArenaHardTask"
    )
    print("[ok] cli.py still exposes --arena-hard-dir and wires it into ArenaHardTask")


def _run_all() -> int:
    tests = [
        test_bundled_files_exist,
        test_default_construction_resolves_to_bundled,
        test_load_prompts_and_baselines,
        test_max_examples_truncation,
        test_score_extraction,
        test_extract_answer,
        test_baseline_routing_by_category,
        test_aggregate_winrate_on_synthetic,
        test_override_arena_hard_dir,
        test_cli_exposes_arena_hard_dir,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"[FAIL] {t.__name__}: {exc}")
    print()
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} failed:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1
    print(f"✅ all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
