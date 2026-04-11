# Repository Guidelines

## Project Structure & Module Organization
`gower_exp/` contains the package code. Core distance logic lives in `gower_dist.py`, with supporting modules such as `accelerators.py`, `vectorized.py`, `parallel.py`, `topn.py`, and `sklearn_compat.py`. Tests are in `tests/` and follow `test_*.py` naming. Benchmarks live in `benchmarks/`, user-facing guides in `docs/`, and runnable examples in `examples/`.

## Build, Test, and Development Commands
Use `uv` for local development.

- `uv sync --all-extras --dev`: install runtime and development dependencies.
- `uv run pytest tests/ --cov=gower_exp --cov-report=term-missing`: run the full test suite with coverage.
- `uv run ruff check .`: run linting and import-order checks.
- `uv run ruff format .`: format Python source.
- `uv run bandit -r gower_exp/`: run the security scan used in pre-commit.
- `uv build`: build the package.

If you use `just`, equivalent shortcuts are defined in `justfile` for `dev`, `test`, `lint`, `format`, `check`, and `build`.

## Coding Style & Naming Conventions
Target Python 3.11+ features, but keep public APIs stable. Use 4-space indentation, double quotes, and Ruff’s default formatting with an 88-character line length. Follow existing module and function naming: `snake_case` for functions and files, `CapWords` for classes. Public functions should include type hints; scientific variable names that match the current codebase are acceptable.

## Testing Guidelines
Pytest is the test framework; coverage is tracked for `gower_exp/`. Add or update tests with every behavior change, especially for edge cases, mixed-type inputs, sklearn compatibility, and performance-sensitive paths. Prefer focused files like `tests/test_gower_matrix.py` or `tests/test_sklearn_compat.py`. Run targeted tests during development, then finish with the full coverage command.

## Commit & Pull Request Guidelines
Recent history uses short, imperative summaries such as `bump test coverage and fix edge cases`. Keep commits small and descriptive. Pull requests should explain the behavior change, list test coverage, and note benchmark impact for algorithmic or accelerator changes. Link related issues when applicable, and include examples or screenshots only when docs or notebook output changes.

## Security & Configuration Tips
Optional extras include `gpu` and `sklearn`; keep fallback behavior working when those extras are absent. Do not commit secrets or local environment files. Run `pre-commit run --all-files` before opening a PR.
