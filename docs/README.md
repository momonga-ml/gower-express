# Documentation

This directory contains the main project guides for users and contributors.

## Guides

- [README](../README.md): package overview, installation, and quick start
- [development.md](development.md): contributor workflow, quality checks, and release steps
- [sklearn-integration.md](sklearn-integration.md): sklearn compatibility notes
- [benchmarks.md](benchmarks.md): benchmark results and performance discussion

## Where To Start

- New user: start with [README](../README.md)
- Contributor: start with [development.md](development.md)
- sklearn use case: start with [sklearn-integration.md](sklearn-integration.md)
- performance review: start with [benchmarks.md](benchmarks.md)

## Documentation Maintenance

Keep docs aligned with the repository state:

- prefer `uv sync --all-extras --dev` for setup examples
- use `benchmarks/`, not `benchmark/`
- describe PyPI publishing as GitHub Release-driven
- avoid references to files or channels that do not exist in the repository

If a code change affects contributor workflow, release steps, or benchmark expectations, update the relevant document in the same pull request.
