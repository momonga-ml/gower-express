# Gower Express

**Fast Gower distance for mixed data in Python**

[![PyPI version](https://img.shields.io/pypi/v/gower_exp.svg)](https://pypi.org/project/gower_exp/)
[![Python Version](https://img.shields.io/pypi/pyversions/gower_exp.svg)](https://pypi.org/project/gower_exp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/momonga-ml/gower-express/actions/workflows/pr.yml/badge.svg)](https://github.com/momonga-ml/gower-express/actions/workflows/pr.yml)

Gower Express provides Gower distance utilities for mixed numerical and categorical data, with optional sklearn integration and accelerator paths for larger workloads.

## Install

```bash
pip install gower_exp
```

Optional extras:

```bash
pip install gower_exp[gpu]
pip install gower_exp[sklearn]
pip install gower_exp[gpu,sklearn]
```

## Quick Start

```python
import pandas as pd
import gower_exp as gower

data = pd.DataFrame(
    {
        "age": [25, 30, 35, 40],
        "category": ["A", "B", "A", "C"],
        "salary": [50_000, 60_000, 55_000, 65_000],
        "city": ["NYC", "LA", "NYC", "Chicago"],
    }
)

distances = gower.gower_matrix(data)
similar = gower.gower_topn(data.iloc[0:1], data, n=3)
```

## Features

- Pairwise Gower distance for mixed-type datasets
- Top-N similarity search helpers
- Optional sklearn-compatible interfaces
- Optional accelerator paths using numba, vectorized code, and GPU support when available

## Project Layout

- `gower_exp/`: package source
- `tests/`: pytest suite
- `benchmarks/`: performance scripts
- `docs/`: user and contributor documentation
- `examples/`: notebooks and examples

## Development

Set up a contributor environment with `uv`:

```bash
uv sync --all-extras --dev
pre-commit install
```

Run the standard checks before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run bandit -r gower_exp/ -c pyproject.toml
uv run pytest tests/ --cov=gower_exp --cov-report=term-missing
```

Short imperative commit messages work best, for example `fix sklearn edge case` or `bump test coverage`. If a change affects performance-sensitive code, include benchmark notes in the pull request summary.

See [docs/development.md](docs/development.md) for the full contributor workflow.

## Releases

PyPI publishing is triggered by publishing a GitHub Release from a `vX.Y.Z` tag. Pushing a tag alone does not publish a package.

The expected maintainer flow is:

1. Update `version` in `pyproject.toml`.
2. Run `just check` and `uv build`.
3. Commit the version bump.
4. Create and push tag `vX.Y.Z`.
5. Publish a GitHub Release from that tag to trigger `.github/workflows/publish.yml`.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/development.md](docs/development.md)
- [docs/sklearn-integration.md](docs/sklearn-integration.md)
- [docs/benchmarks.md](docs/benchmarks.md)

## License

MIT. See [LICENSE](LICENSE).
