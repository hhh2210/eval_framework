# Release Checklist

This repository has development history that should not be pushed as the public
paper artifact. A previous development commit contained a hardcoded API key, so
the public repository must be created from a cleaned file tree rather than by
merging or fast-forwarding the full `dev` history.

## Public Git Release

Recommended flow:

1. Keep `dev` private/internal.
2. Build the public branch from the cleaned tree on `public-clean`.
3. Publish it as a single clean commit, for example with an orphan branch or a
   squash merge into a fresh public `main`.
4. Do not merge `origin/dev` directly into public `main`.
5. Rotate any credential that ever appeared in development history.

Example local workflow:

```bash
git switch public-clean
uv run python -m pytest -q
uv build
uvx twine check dist/*

git switch --orphan public-main
git add .
git commit -m "release: publish cleaned academic eval framework"
```

Before pushing, inspect `git log --oneline` on `public-main`; it should contain
only the clean public commit(s), not the historical `dev` commits.

## PyPI Release

PyPI normalizes project names by lowercasing and replacing runs of `.`, `_`, and
`-` with `-`. The name `eval_framework` is therefore equivalent to
`eval-framework`, which is already taken on PyPI. The configured distribution
name is `llm-eval-framework`; the import package remains `eval_framework`, and
the CLI remains `eval-framework`.

Build and validate:

```bash
rm -rf dist
uv build
uvx twine check dist/*
uvx --from dist/llm_eval_framework-0.1.0-py3-none-any.whl eval-framework --help
```

Upload with a PyPI API token:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD='pypi-...' uvx twine upload dist/*
```

The first upload creates the PyPI project automatically. Do not use the legacy
`setup.py register` or `setup.py upload` workflow.

## Pre-Publish Checks

Run these from the repository root:

```bash
bash -n examples/batch_eval.sh examples/compare_two_runs.sh examples/shard_parallel_eval.sh tools/rejudge_alpaca_arena_ports.sh
uv run python -m pytest -q
uv build
uvx twine check dist/*
uvx --from dist/llm_eval_framework-0.1.0-py3-none-any.whl eval-framework --help
rg -n "(sk-[A-Za-z0-9]{20,}|sk_[A-Za-z0-9_-]{20,}|/Users/larry|/home/|/data/(haozy|wangxk)|hackingRubricsRL|PPIO|pa/claude|DASHSCOPE_API_KEY)" README.md examples tools pyproject.toml CITATION.cff LICENSE tasks/*.py tasks/*/__init__.py
rg -n "(^|[^A-Za-z0-9_])(sk-[A-Za-z0-9]{20,}|sk_[A-Za-z0-9_-]{20,}|pypi-[A-Za-z0-9_-]{20,})" --glob '!dist/**' --glob '!build/**' --glob '!*.egg-info/**' --glob '!__pycache__/**' .
```

Both `rg` commands should return no matches. The first command intentionally
scans code and documentation for private development paths and internal service
names. The second command scans the full working tree for real-looking tokens;
benchmark data may contain generic example paths such as `/home/user`, so do not
use broad path patterns over bundled benchmark JSONL assets.
