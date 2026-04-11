# Development Guide

This guide covers local setup, quality checks, and the release path used in this repository.

## Quickstart

```bash
git clone https://github.com/momonga-ml/gower-express.git
cd gower-express
uv sync --all-extras --dev
pre-commit install
```

If you prefer an isolated shell, create a virtual environment first with `uv venv`.

## Repository Layout

- `gower_exp/`: package code, including the core distance implementation and optional sklearn helpers.
- `tests/`: pytest suite for matrix logic, top-N search, sklearn compatibility, and regressions.
- `benchmarks/`: ad hoc performance scripts for CPU, vectorized, numba, and memory profiling.
- `docs/`: contributor and user documentation.
- `examples/`: notebooks and example material.

## Quality Checks

Run these before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run bandit -r gower_exp/ -c pyproject.toml
uv run pytest tests/ --cov=gower_exp --cov-report=term-missing
```

Useful shortcuts are defined in `justfile`:

```bash
just lint
just format-check
just test
just check
```

## Testing and Benchmarks

Use targeted tests while iterating, then finish with the full suite.

```bash
uv run pytest tests/test_gower_matrix.py
uv run pytest tests/test_sklearn_compat.py
uv run python benchmarks/clean_benchmark.py
uv run python benchmarks/ultimate_benchmark.py
```

Performance-sensitive changes should include benchmark notes in the pull request summary.

## Pull Requests

Use short imperative commit messages such as `fix sklearn edge case` or `bump test coverage`. Pull requests should describe the behavior change, list the checks you ran, and call out any benchmark impact for algorithm or accelerator changes.

## Releases

PyPI publishing is triggered by publishing a GitHub Release, not by pushing a tag alone.

1. Update `version` in `pyproject.toml`.
2. Run `just check` and `uv build`.
3. Commit the version bump.
4. Create and push tag `vX.Y.Z`.
5. Publish a GitHub Release from that tag to trigger `.github/workflows/publish.yml`.

The `just release <version>` recipe automates the local version bump, checks, commit, and tag push, then leaves the GitHub Release publication step to the maintainer.
