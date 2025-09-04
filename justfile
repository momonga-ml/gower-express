# Justfile for gower_exp development
# Run 'just --list' to see available commands

# Default recipe - show available commands
default:
    @just --list

# Install development dependencies
dev:
    uv sync --all-extras --dev

# Run tests with coverage
test:
    uv run pytest tests/ --cov=gower_exp --cov-report=term-missing

# Run specific test file
test-file file:
    uv run pytest {{file}} --cov=gower_exp --cov-report=term-missing

# Run linting checks
lint:
    uv run ruff check .

# Fix linting issues automatically
lint-fix:
    uv run ruff check --fix .

# Check code formatting
format-check:
    uv run ruff format --check .

# Format code
format:
    uv run ruff format .

# Run all quality checks (lint + format + tests)
check: lint format-check test

# Build the package
build:
    uv build

# Clean build artifacts
clean:
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -delete
    find . -type f -name "*.pyc" -delete

# Publish to Test PyPI (requires PYPI_TEST_TOKEN env var)
publish-test:
    #!/usr/bin/env bash
    if [ -z "$PYPI_TEST_TOKEN" ]; then
        echo "Error: PYPI_TEST_TOKEN environment variable is required"
        exit 1
    fi
    uv build
    uv publish --publish-url https://test.pypi.org/legacy/ --token $PYPI_TEST_TOKEN

# Publish to PyPI (requires PYPI_TOKEN env var)
publish:
    #!/usr/bin/env bash
    if [ -z "$PYPI_TOKEN" ]; then
        echo "Error: PYPI_TOKEN environment variable is required"
        exit 1
    fi
    uv build
    uv publish --token $PYPI_TOKEN

# Run benchmarks
benchmark:
    uv run python benchmark/clean_benchmark.py

# Run comprehensive benchmarks
benchmark-full:
    uv run python benchmark/ultimate_benchmark.py

# Start Jupyter for examples
jupyter:
    uv run jupyter lab examples/

# Show package info
info:
    uv run python -c "import gower_exp; print(f'Version: {gower_exp.__version__}')"

# Update version in pyproject.toml (usage: just version 0.1.5)
version new_version:
    sed -i '' 's/version = "[^"]*"/version = "{{new_version}}"/' pyproject.toml
    @echo "Updated version to {{new_version}}"

# Create a new release (builds, tags, and pushes)
release version:
    #!/usr/bin/env bash
    set -e
    echo "Creating release {{version}}..."

    # Update version
    just version {{version}}

    # Run all checks
    just check

    # Build package
    just build

    # Commit version change
    git add pyproject.toml
    git commit -m "Bump version to {{version}}"

    # Create and push tag
    git tag v{{version}}
    git push origin HEAD
    git push origin v{{version}}

    echo "Release {{version}} created! Check GitHub Actions for publishing status."
