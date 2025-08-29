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

## Performance and Features

**Why choose gower-express over the original gower package?** Here are the key improvements:

### Core Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| **GPU Acceleration** | ✅ Available | CuPy integration for CUDA-enabled GPUs |
| **Numba JIT Compilation** | ✅ Working | Faster CPU computations with JIT acceleration |
| **Improved Testing** | ✅ Enhanced | Better test coverage and benchmarking suite |
| **Bug Fixes** | ✅ Resolved | Fixed issues with negative values and NaN handling |
| **Modern Tooling** | ✅ Updated | Modern development workflow with uv and ruff |

### Performance Notes

**Current Performance Status:**
- **Matrix Computation**: Enhanced with specialized Numba kernels (15-25% faster)
- **GPU Support**: Available with CuPy for large-scale computations
- **Top-N Search**: ✅ **Optimized** - Vectorized implementation delivers 1.1-1.2x consistent speedup
- **Memory Usage**: 25-40% reduction through optimized allocations and in-place operations

**Benchmark Results:**
Significant performance improvements over the original package:
- **Top-N queries**: 1.1-1.2x faster with vectorized algorithm
- **Matrix computation**: 15-25% faster with enhanced Numba optimizations
- **Memory efficiency**: 25-40% reduction in memory usage
- **All improvements maintain full backward compatibility**

See [Benchmarks](docs/Benchmark.MD) for detailed performance analysis across different dataset sizes and configurations.

### Why Choose gower-express?

**Key Advantages:**

1. **🚀 GPU Acceleration**: Optional CuPy integration for CUDA-enabled GPUs allows processing of massive datasets with significant speedups when available.

2. **⚡ Enhanced Performance**:
   - **15-25% faster** matrix computation with specialized Numba kernels
   - **1.1-1.2x faster** top-N search with vectorized algorithms
   - **25-40% memory reduction** through optimized allocations

3. **🧪 Enhanced Testing**: Comprehensive test suite with edge case handling, performance benchmarks, and correctness verification.

4. **🐛 Bug Fixes**: Resolved issues with negative values, NaN handling, and edge cases that existed in the original package.

5. **🔧 Modern Development**: Updated packaging, dependency management, and development tools (uv, ruff) for better maintainability.

6. **📦 Flexible Dependencies**: Optional GPU dependencies and sklearn compatibility - install only what you need.

**Run the Benchmarks Yourself:**
```bash
# Quick performance test
python benchmark/clean_benchmark.py

# Run all benchmarks systematically
python run_all_benchmarks.py

# Individual benchmark tests
python benchmark/benchmark_gower_topn.py  # Test top-N improvements
python benchmark/memory_benchmark.py      # Test memory optimizations
python benchmark/benchmark_numba.py       # Test Numba enhancements
```

## Introduction

Gower's distance calculation in Python. Gower Distance is a distance measure that can be used to calculate distance between two entity whose attribute has a mixed of categorical and numerical values. [Gower (1971) A general coefficient of similarity and some of its properties. Biometrics 27 857–874.](https://www.jstor.org/stable/2528823?seq=1)

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


# Scikit-learn Integration

Gower-express provides lightweight integration with scikit-learn through optional compatibility functions. This allows you to use Gower distance seamlessly with sklearn's machine learning algorithms.

## Installation with sklearn support

```bash
# Install with sklearn compatibility
pip install gower_exp[sklearn]

# Or for development with all optional dependencies
uv pip install -e ".[dev,sklearn]"
```

## Using as a Custom Distance Metric

You can use Gower distance with any sklearn algorithm that accepts custom distance metrics:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from gower_exp import GowerDistance

# Configure Gower distance with your feature types
gower_metric = GowerDistance(cat_features=[True, False, True, False, True, False])

# Use with k-NN classifier
knn = KNeighborsClassifier(metric=gower_metric, algorithm='brute', n_neighbors=3)
knn.fit(X, y)
predictions = knn.predict(X_test)

# Use with DBSCAN clustering
clustering = DBSCAN(metric=gower_metric, eps=0.3)
cluster_labels = clustering.fit_predict(X)
```

## Convenience Functions

For common use cases, gower-express provides ready-to-use convenience functions:

```python
from gower_exp import make_gower_knn_classifier, make_gower_knn_regressor

# Create a k-NN classifier with Gower distance
classifier = make_gower_knn_classifier(
    n_neighbors=5,
    cat_features=[True, False, True, False, True, False],  # Specify categorical features
    weights='distance'  # Use distance weighting
)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Create a k-NN regressor
regressor = make_gower_knn_regressor(
    n_neighbors=3,
    cat_features='auto',  # Auto-detect categorical features (pandas DataFrames)
    feature_weights=[2.0, 1.0, 1.0, 0.5, 1.0, 1.0]  # Custom feature weights
)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
```

## Precomputed Distance Matrices (Recommended for Large Datasets)

For better performance with repeated operations on the same dataset, use precomputed distance matrices:

```python
from sklearn.neighbors import KNeighborsClassifier
from gower_exp import precomputed_gower_matrix

# Compute distance matrices once
distances = precomputed_gower_matrix(X_train, X_test, cat_features=[True, False, True])

# Train classifier with precomputed distances
knn = KNeighborsClassifier(metric='precomputed', n_neighbors=5)
knn.fit(distances['train'], y_train)

# Predict using test-to-train distances
predictions = knn.predict(distances['test'])
```

## Integration with sklearn Pipelines

Gower distance works seamlessly with sklearn's pipeline and model selection tools:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from gower_exp import make_gower_knn_classifier

# Create classifier
clf = make_gower_knn_classifier(cat_features=[True, False, True])

# Use with cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Use with grid search (note: limited to hyperparameters that don't affect the metric)
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(
    make_gower_knn_classifier(cat_features=[True, False, True]),
    param_grid,
    cv=3,
    scoring='accuracy'
)
grid_search.fit(X, y)
```

## Feature Type Detection

The sklearn integration supports automatic categorical feature detection when using pandas DataFrames:

```python
import pandas as pd
from gower_exp import make_gower_knn_classifier

# Create DataFrame with mixed types
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'category': ['A', 'B', 'A', 'C'],  # Automatically detected as categorical
    'salary': [50000, 60000, 55000, 65000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago']  # Automatically detected as categorical
})

# Auto-detect categorical features (recommended for DataFrames)
classifier = make_gower_knn_classifier(cat_features='auto')
classifier.fit(df, y)
```

## Performance Tips

1. **Use precomputed matrices** for repeated operations on the same dataset
2. **Specify categorical features explicitly** when possible to avoid auto-detection overhead
3. **Consider feature weights** based on domain knowledge for better results
4. **Use algorithm='brute'** is required for custom metrics in sklearn (automatically set by convenience functions)

## Supported sklearn Algorithms

Gower distance works with any sklearn algorithm that accepts:
- **Custom distance metrics**: KNeighborsClassifier, KNeighborsRegressor, DBSCAN, etc.
- **Precomputed distances**: Most clustering algorithms, some dimensionality reduction techniques

Popular combinations:
- **Classification**: `KNeighborsClassifier` with Gower distance
- **Regression**: `KNeighborsRegressor` with Gower distance
- **Clustering**: `DBSCAN`, `AgglomerativeClustering` with precomputed Gower distances
- **Outlier Detection**: `LocalOutlierFactor` with Gower distance
