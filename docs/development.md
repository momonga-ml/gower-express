# Development Guide

This guide covers everything you need to contribute to Gower Express, from setting up your development environment to submitting pull requests.

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/momonga-ml/gower-express.git
cd gower-express

# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create development environment
uv venv
uv pip install -e ".[dev,gpu,sklearn]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Environment Setup

#### Using uv (Recommended)
```bash
# Create virtual environment
uv venv

# Install all dependencies
uv pip install -e ".[dev,gpu,sklearn]"

# Activate environment (if needed)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

#### Using pip
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,gpu,sklearn]"
```

### 2. Code Quality Tools

We use several tools to maintain code quality:

```bash
# Check code style and linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check and format in one command
uv run ruff check --fix . && uv run ruff format .

# Security scanning
uv run bandit -r gower_exp/

# Pre-commit hooks (runs automatically on commit)
pre-commit run --all-files
```

### 3. Testing

#### Run All Tests
```bash
# Run complete test suite
uv run pytest tests/

# With coverage
uv run pytest tests/ --cov=gower_exp --cov-report=term-missing

# Parallel execution
uv run pytest tests/ -n auto
```

#### Run Specific Test Categories
```bash
# Core functionality tests
uv run pytest tests/test_gower_matrix.py tests/test_gower_topn.py

# sklearn compatibility tests
uv run pytest tests/test_sklearn_compat.py

# Performance tests
uv run pytest tests/test_performance.py

# Edge cases and coverage
uv run pytest tests/test_edge_cases.py tests/test_coverage_boost.py

# GPU tests (requires CUDA)
uv run pytest tests/ -k "gpu"
```

#### Test Coverage Requirements
- Minimum 95% code coverage
- All new features must include tests
- Edge cases and error conditions must be tested

### 4. Benchmarking

```bash
# Quick performance test
uv run python benchmark/clean_benchmark.py

# Comprehensive benchmarks
uv run python benchmark/ultimate_benchmark.py

# Specific optimization tests
uv run python benchmark/benchmark_numba.py
uv run python benchmark/benchmark_vectorized.py

# Memory benchmarks
uv run python benchmark/memory_benchmark.py
```

## Project Structure

```
gower-express/
├── gower_exp/                 # Main package
│   ├── __init__.py           # Public API exports
│   ├── gower_dist.py         # Core distance calculations
│   └── sklearn_compat.py     # scikit-learn integration
├── tests/                     # Test suite
│   ├── test_gower_matrix.py  # Matrix computation tests
│   ├── test_gower_topn.py    # Top-N search tests
│   ├── test_sklearn_compat.py # sklearn integration tests
│   ├── test_performance.py   # Performance benchmarks
│   └── test_*.py             # Additional test modules
├── benchmark/                 # Performance benchmarking
├── examples/                  # Usage examples and tutorials
├── docs/                      # Documentation
└── .github/                   # CI/CD workflows
```

## Code Architecture

### Core Components

#### `gower_dist.py` - Core Implementation
- `gower_matrix()`: Pairwise distance computation
- `gower_topn()`: Optimized top-N similarity search
- `gower_get()`: Internal optimized distance calculation
- Multiple implementation strategies (numba, GPU, vectorized)

#### `sklearn_compat.py` - scikit-learn Integration
- `GowerDistance`: Metric-compatible class for sklearn
- `make_gower_knn_*()`: Convenience classifier/regressor factories
- `precomputed_gower_matrix()`: Optimized precomputed distances
- `gower_distance()`: Single distance calculation

### Design Principles

1. **Performance First**: All optimizations must be benchmarked
2. **Backward Compatibility**: Public API should remain stable
3. **Optional Dependencies**: Core functionality works without sklearn/cupy
4. **Type Safety**: Full type hints for all public APIs
5. **Memory Efficiency**: Minimize memory allocations and copies

## Adding New Features

### 1. Feature Development Process

1. **Open an Issue**: Describe the feature and get feedback
2. **Write Tests First**: Test-driven development preferred
3. **Implement Feature**: Follow existing code patterns
4. **Add Documentation**: Update docstrings and guides
5. **Benchmark Performance**: Ensure no regressions
6. **Submit PR**: Include tests, docs, and benchmarks

### 2. Code Style Guidelines

```python
# Use type hints for all public functions
def gower_matrix(
    data_x: ArrayLike,
    data_y: Optional[ArrayLike] = None,
    weight: Optional[ArrayLike] = None,
    cat_features: Optional[Union[List[int], List[bool], str]] = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute Gower distance matrix.

    Parameters
    ----------
    data_x : array-like
        Input data matrix
    data_y : array-like, optional
        Second data matrix for pairwise distances
    weight : array-like, optional
        Feature weights
    cat_features : list or str, optional
        Categorical feature specification
    use_gpu : bool, default=False
        Enable GPU acceleration

    Returns
    -------
    ndarray
        Distance matrix
    """
```

### 3. Testing New Features

```python
# tests/test_new_feature.py
import pytest
import numpy as np
import pandas as pd
from gower_exp import new_feature

class TestNewFeature:
    """Test suite for new feature functionality."""

    def test_basic_functionality(self):
        """Test basic feature operation."""
        data = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        result = new_feature(data)
        assert result is not None

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty data
        with pytest.raises(ValueError):
            new_feature(pd.DataFrame())

    def test_performance(self):
        """Test performance requirements."""
        large_data = generate_test_data(size=10000)
        start_time = time.time()
        result = new_feature(large_data)
        duration = time.time() - start_time
        assert duration < 5.0  # Must complete in under 5 seconds
```

## Performance Optimization

### 1. Optimization Strategy
- **Profile First**: Use `cProfile` and `line_profiler` to identify bottlenecks
- **Benchmark Everything**: Measure before and after optimizations
- **Multiple Approaches**: Test different algorithms (numba, vectorized, GPU)
- **Memory Analysis**: Use `memory_profiler` for memory optimization

### 2. Common Optimization Techniques

#### Numba JIT Compilation
```python
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def optimized_distance_kernel(X, Y, cat_mask):
    """JIT-compiled distance calculation kernel."""
    # Implementation optimized for numba
```

#### GPU Acceleration
```python
try:
    import cupy as cp

    def gpu_gower_matrix(X_gpu, Y_gpu=None):
        """GPU-accelerated distance matrix computation."""
        # CuPy implementation
except ImportError:
    # Fallback to CPU implementation
    pass
```

#### Vectorized Operations
```python
def vectorized_distance(X, Y):
    """Vectorized distance computation using NumPy."""
    # Avoid loops, use broadcasting and vectorized operations
    categorical_diff = (X[:, None, :] != Y[None, :, :]).astype(float)
    numerical_diff = np.abs(X[:, None, :] - Y[None, :, :])
    return np.mean(np.concatenate([categorical_diff, numerical_diff], axis=2), axis=2)
```

### 3. Benchmarking Standards

All performance improvements must be benchmarked with:
- Multiple data sizes (1K, 10K, 100K records)
- Different feature ratios (categorical vs numerical)
- Various hardware configurations
- Memory usage profiling

## CI/CD and Automation

### GitHub Actions Workflows

#### Pull Request Checks (`.github/workflows/pr.yml`)
- Code formatting (ruff)
- Linting and type checking
- Complete test suite
- Security scanning (bandit)
- Performance regression tests

#### Publishing (`.github/workflows/publish.yml`)
- Automated PyPI publishing on version tags
- Build verification across Python versions
- Documentation deployment

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- Code formatting with ruff
- Import sorting
- Security scanning
- Large file detection
- JSON/YAML validation

## Documentation

### 1. Code Documentation
- **Docstrings**: NumPy-style docstrings for all public functions
- **Type Hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings

### 2. User Documentation
- **README**: Marketing-focused overview
- **docs/**: Detailed guides and tutorials
- **examples/**: Jupyter notebooks with real-world use cases

### 3. API Reference
Auto-generated from docstrings using Sphinx or similar tools.

## Release Process

### 1. Version Management
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md with release notes
# Commit changes
git commit -m "Prepare release v0.1.5"

# Create and push tag
git tag v0.1.5
git push origin v0.1.5
```

### 2. Automated Publishing
- GitHub Actions automatically publishes to PyPI on version tags
- Build artifacts are tested before publishing
- Rollback process available for failed releases

### 3. Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Security scan clean
- [ ] Version number incremented
- [ ] CHANGELOG.md updated
- [ ] Tag created and pushed

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord/Slack**: Real-time chat (link in README)
- **Stack Overflow**: Use tag `gower-distance`

## Contributing Guidelines

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Implement the feature
5. Run the full test suite
6. Submit a pull request

### Code Review Standards
- All PRs require review from maintainers
- CI checks must pass
- Performance regressions require justification
- Breaking changes need special approval

### Recognition
Contributors are acknowledged in:
- CONTRIBUTORS.md file
- GitHub contributors graph
- Release notes for significant contributions

Thank you for contributing to Gower Express! 🚀
