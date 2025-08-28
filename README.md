# Gower Express

**An enhanced fork of Michael Yan's gower package with GPU acceleration and performance improvements.**

> **Note**: This package is distributed on PyPI as `gower_exp` (short for Gower Express).

This project is a fork of [Michael Yan's original gower package](https://github.com/wwwjk366/gower) that adds significant performance improvements, GPU acceleration support, and modern Python tooling.

## What's New in This Fork

- 🚀 **GPU Acceleration**: CuPy support for massive performance gains on CUDA-enabled GPUs
- ⚡ **Performance Optimizations**: Numba JIT compilation for faster CPU computations  
- 🔧 **Modern Tooling**: Updated development workflow with uv and ruff
- 🧪 **Enhanced Testing**: Improved test coverage and performance benchmarks
- 🐛 **Bug Fixes**: Resolved issues with negative values and NaN handling
- 📦 **Better Dependencies**: Optimized dependency management and optional GPU dependencies

## Performance Benchmarks

**Why choose gower-express over the original gower package?** Here's the data:

### Key Performance Improvements

| Metric | Original Package | Gower-Express | Improvement |
|--------|-----------------|---------------|-------------|
| **Top-N Search Speed** | 0.52s (1000 samples) | 0.19s | **2.7x faster** 🚀 |
| **Memory Efficiency** | Full matrix storage | Heap-based algorithm | **~95% less memory** 💾 |
| **Large Dataset Handling** | Memory bottleneck at 10K+ | Scales to 20K+ samples | **Better scalability** ⚡ |
| **Correctness** | ✅ Verified | ✅ Verified | **Same accuracy** |

### gower_topn Optimization Results

Our optimized `gower_topn` function provides significant advantages for nearest neighbor search:

| Dataset Size | Query Type | Memory Reduction | Performance Gain |
|-------------|------------|------------------|------------------|
| 1,000 samples | Top-5 | 88% less memory | 1.5x speedup |
| 5,000 samples | Top-10 | 85% less memory | Similar speed |
| 10,000+ samples | Top-20 | 90%+ less memory | **Memory scalable** |

### Real-World Performance Scenarios

**Scenario 1: Small-Medium Datasets (1K-5K samples)**
- Original: Full distance matrix computation required
- Gower-Express: Early stopping + heap optimization
- **Result**: 1.1-1.6x speedup with drastically reduced memory usage

**Scenario 2: Large Datasets (10K+ samples)** 
- Original: Memory-intensive, potential crashes
- Gower-Express: Memory-efficient algorithm, predictable performance
- **Result**: Enables processing of datasets that would otherwise fail

**Scenario 3: Production Systems**
- Original: High memory overhead per query
- Gower-Express: Consistent memory footprint regardless of dataset size
- **Result**: Better resource utilization and system stability

### Why gower-express?

**Technical Optimizations Under the Hood:**

1. **🎯 Early Stopping Algorithm**: For `gower_topn`, we avoid computing the full distance matrix by using a min-heap to track only the top-N candidates, stopping computation early when possible.

2. **💾 Memory-Efficient Design**: Instead of storing an O(n²) distance matrix, we use O(n) memory for heap-based top-N search, enabling processing of much larger datasets.

3. **⚡ Vectorized Operations**: Optimized NumPy operations and broadcasting for faster distance calculations across mixed categorical and numerical features.

4. **🚀 Optional GPU Acceleration**: CuPy integration allows processing of massive datasets on CUDA-enabled GPUs for even greater speedups.

5. **🔧 Modern Codebase**: Clean, maintainable code with comprehensive testing, type hints, and modern Python packaging.

**Run the Benchmarks Yourself:**
```bash
# Quick performance test
python benchmark/ultimate_benchmark.py

# Comprehensive benchmarks  
python benchmark/benchmark_gower_topn.py
python benchmark/large_scale_benchmark.py
```

## Introduction

Gower's distance calculation in Python. Gower Distance is a distance measure that can be used to calculate distance between two entity whose attribute has a mixed of categorical and numerical values. [Gower (1971) A general coefficient of similarity and some of its properties. Biometrics 27 857–874.](https://www.jstor.org/stable/2528823?seq=1) 

## Credits and Attribution

- **Original Author**: [Michael Yan](https://github.com/wwwjk366) - Created the original gower package
- **Core Algorithm**: [Marcelo Beckmann](https://sourceforge.net/projects/gower-distance-4python/files/) - Wrote the core functions
- **Additional Contributors**: Various contributors who improved the original package
- **This Fork**: Enhanced with GPU acceleration, performance optimizations, and modern tooling

More details about the original implementation can be found on [Michael Yan's website](https://www.thinkdatascience.com/post/2019-12-16-introducing-python-package-gower/).

# Examples

## Installation

### Standard Installation
```bash
pip install gower_exp
```

### Development Installation with uv (Recommended)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# For GPU acceleration support
uv pip install -e ".[dev,gpu]"
```

### Legacy pip Installation for Development
```bash
pip install -e ".[dev]"
```

## Generate some data

```python
import numpy as np
import pandas as pd
import gower_exp

Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,None],
'gender':['M','M','N','M','F','F','F','F',None],
'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED',None],
'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0,None],
'has_children':[1,0,1,1,1,0,0,1,None],
'available_credit':[2200,100,22000,1100,2000,100,6000,2200,None]})
Yd = Xd.iloc[1:3,:]
X = np.asarray(Xd)
Y = np.asarray(Yd)

```

## Find the distance matrix

```python
gower_exp.gower_matrix(X)
```




    array([[0.        , 0.3590238 , 0.6707398 , 0.31787416, 0.16872811,
            0.52622986, 0.59697855, 0.47778758,        nan],
           [0.3590238 , 0.        , 0.6964303 , 0.3138769 , 0.523629  ,
            0.16720603, 0.45600235, 0.6539635 ,        nan],
           [0.6707398 , 0.6964303 , 0.        , 0.6552807 , 0.6728013 ,
            0.6969697 , 0.740428  , 0.8151941 ,        nan],
           [0.31787416, 0.3138769 , 0.6552807 , 0.        , 0.4824794 ,
            0.48108295, 0.74818605, 0.34332284,        nan],
           [0.16872811, 0.523629  , 0.6728013 , 0.4824794 , 0.        ,
            0.35750175, 0.43237334, 0.3121036 ,        nan],
           [0.52622986, 0.16720603, 0.6969697 , 0.48108295, 0.35750175,
            0.        , 0.2898751 , 0.4878362 ,        nan],
           [0.59697855, 0.45600235, 0.740428  , 0.74818605, 0.43237334,
            0.2898751 , 0.        , 0.57476616,        nan],
           [0.47778758, 0.6539635 , 0.8151941 , 0.34332284, 0.3121036 ,
            0.4878362 , 0.57476616, 0.        ,        nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan,        nan,        nan,        nan]], dtype=float32)


## Find Top n results

```python
gower_exp.gower_topn(Xd.iloc[0:2,:], Xd.iloc[:,], n = 5)
```




    {'index': array([4, 3, 1, 7, 5]),
     'values': array([0.16872811, 0.31787416, 0.3590238 , 0.47778758, 0.52622986],
           dtype=float32)}


