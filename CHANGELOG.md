# Changelog

All notable changes to this project will be documented in this file.

This project is a fork of [Michael Yan's original gower package](https://github.com/wwwjk366/gower).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 🚀 **GPU Acceleration**: CuPy support for massive performance gains on CUDA-enabled GPUs
- ⚡ **Performance Optimizations**: Numba JIT compilation for faster CPU computations
- 🔧 **Modern Development Tooling**: 
  - uv for fast dependency management and virtual environments
  - ruff for lightning-fast linting and code formatting
- 🧪 **Enhanced Testing**: Improved test coverage and performance benchmarks
- 📦 **Optional Dependencies**: Separate `gpu` and `dev` dependency groups for cleaner installs
- 📋 **Development Documentation**: Comprehensive CLAUDE.md for development guidance

### Fixed
- 🐛 **Negative Values**: Resolved issues with negative distance calculations when data contains negative values
- 🔢 **NaN Handling**: Improved handling of missing values (NaN) in distance calculations
- ⚖️ **Weight Normalization**: Correct weight summation when NaNs are present in the data

### Changed
- 📦 **Dependency Updates**: Updated to modern versions of NumPy, SciPy, pandas, and added Numba/Joblib
- 🏗️ **Build System**: Migrated to modern `pyproject.toml` configuration
- 🧹 **Code Quality**: Applied consistent formatting and linting with ruff
- 📚 **Documentation**: Updated README with fork acknowledgment and improved installation instructions

### Technical Improvements
- Performance optimizations through Numba JIT compilation
- Parallel processing improvements with joblib
- Optional GPU acceleration via CuPy for CUDA-enabled systems
- Modern Python packaging standards
- Comprehensive test suite with performance benchmarks

## Original Package History

This fork is based on Michael Yan's gower package (v0.1.x), which included:
- Core Gower distance implementation by Marcelo Beckmann
- Support for mixed categorical/numerical data
- Distance matrix and top-N similarity calculations
- Basic NaN handling

For the complete history of the original package, see: https://github.com/wwwjk366/gower

---

## Credits

- **Original Author**: [Michael Yan](https://github.com/wwwjk366) - Created the original gower package
- **Core Algorithm**: [Marcelo Beckmann](https://sourceforge.net/projects/gower-distance-4python/files/) - Wrote the core distance functions  
- **Contributors**: Dominic Dall'Osto and other community contributors who improved the original
- **This Fork**: Enhanced by Momonga ML with GPU acceleration, performance optimizations, and modern tooling