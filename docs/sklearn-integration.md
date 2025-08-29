# scikit-learn Integration Guide

Gower Express provides seamless integration with scikit-learn through optional compatibility functions. This allows you to use Gower distance with sklearn's machine learning algorithms while maintaining performance and ease of use.

## Installation

```bash
# Install with sklearn support
pip install gower_exp[sklearn]

# Or for development with all optional dependencies
pip install gower_exp[dev,sklearn]
```

## Quick Start

```python
from gower_exp import make_gower_knn_classifier
import pandas as pd

# Your mixed data
data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'category': ['A', 'B', 'A', 'C'],
    'salary': [50000, 60000, 55000, 65000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago']
})

# Create classifier with auto-detected categorical features
clf = make_gower_knn_classifier(n_neighbors=3, cat_features='auto')
clf.fit(data, labels)
predictions = clf.predict(new_data)
```

## Available Functions

### GowerDistance Class
Transform-style interface for sklearn compatibility:

```python
from sklearn.neighbors import KNeighborsClassifier
from gower_exp import GowerDistance

# Configure Gower distance
gower_metric = GowerDistance(cat_features=[True, False, True, False])

# Use with any sklearn algorithm that accepts custom metrics
knn = KNeighborsClassifier(metric=gower_metric, algorithm='brute')
knn.fit(X, y)
```

### Convenience Functions

#### make_gower_knn_classifier / make_gower_knn_regressor
Pre-configured k-NN estimators:

```python
from gower_exp import make_gower_knn_classifier, make_gower_knn_regressor

# Classification
classifier = make_gower_knn_classifier(
    n_neighbors=5,
    cat_features='auto',  # Auto-detect from pandas DataFrame
    weights='distance'    # Distance-weighted predictions
)

# Regression
regressor = make_gower_knn_regressor(
    n_neighbors=3,
    cat_features=[True, False, True],  # Explicit specification
    feature_weights=[2.0, 1.0, 1.5]   # Custom feature weights
)
```

#### precomputed_gower_matrix
Optimized for repeated operations:

```python
from gower_exp import precomputed_gower_matrix
from sklearn.neighbors import KNeighborsClassifier

# Compute distance matrices once
distances = precomputed_gower_matrix(X_train, X_test, cat_features='auto')

# Use precomputed distances for faster training/prediction
knn = KNeighborsClassifier(metric='precomputed')
knn.fit(distances['train'], y_train)
predictions = knn.predict(distances['test'])
```

## Feature Type Detection

### Automatic Detection (Recommended for DataFrames)
```python
# Automatically detect categorical features from pandas DataFrame
clf = make_gower_knn_classifier(cat_features='auto')
```

### Manual Specification
```python
# Boolean array specifying which features are categorical
clf = make_gower_knn_classifier(cat_features=[True, False, True, False])

# Integer indices of categorical features
clf = make_gower_knn_classifier(cat_features=[0, 2])
```

## Advanced Usage

### Custom Feature Weights
```python
# Give more importance to certain features
clf = make_gower_knn_classifier(
    cat_features='auto',
    feature_weights=[2.0, 1.0, 0.5, 1.0]  # First feature has 2x weight
)
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create pipeline (note: StandardScaler won't affect categorical features)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', make_gower_knn_classifier(cat_features='auto'))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
```

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Note: cat_features must remain constant during grid search
base_classifier = make_gower_knn_classifier(cat_features='auto')

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(
    base_classifier,
    param_grid,
    cv=3,
    scoring='accuracy'
)
grid_search.fit(X, y)
```

## Supported Algorithms

### Direct Metric Support
These algorithms work directly with the `GowerDistance` metric:
- `KNeighborsClassifier`
- `KNeighborsRegressor`
- `RadiusNeighborsClassifier`
- `RadiusNeighborsRegressor`
- `NearestCentroid`
- `DBSCAN`
- `LocalOutlierFactor`

### Precomputed Distance Support
These algorithms work with precomputed Gower distance matrices:
- `AgglomerativeClustering`
- `DBSCAN`
- `SpectralClustering`
- `MDS` (Multidimensional Scaling)
- `TSNE` (with metric='precomputed')

## Performance Tips

1. **Use precomputed matrices** for repeated operations on the same dataset
2. **Specify categorical features explicitly** when possible to avoid detection overhead
3. **Set algorithm='brute'** (done automatically by convenience functions)
4. **Consider feature scaling** for numerical features if they have very different ranges
5. **Use feature weights** to emphasize important attributes

## Common Patterns

### Clustering Mixed Data
```python
from sklearn.cluster import AgglomerativeClustering
from gower_exp import precomputed_gower_matrix

# Compute distances once
distances = precomputed_gower_matrix(data, cat_features='auto')

# Hierarchical clustering
clustering = AgglomerativeClustering(
    n_clusters=5,
    affinity='precomputed',
    linkage='average'
)
labels = clustering.fit_predict(distances['train'])
```

### Anomaly Detection
```python
from sklearn.neighbors import LocalOutlierFactor
from gower_exp import GowerDistance

# Detect outliers using Gower distance
lof = LocalOutlierFactor(
    metric=GowerDistance(cat_features='auto'),
    algorithm='brute'
)
outlier_scores = lof.fit_predict(data)
```

### Dimensionality Reduction
```python
from sklearn.manifold import MDS
from gower_exp import gower_matrix

# Reduce dimensionality while preserving Gower distances
distances = gower_matrix(data)
mds = MDS(dissimilarity='precomputed', random_state=42)
reduced_data = mds.fit_transform(distances)
```

## Troubleshooting

### Common Issues

**Error: "algorithm must be 'brute' for custom metrics"**
- Solution: Use `algorithm='brute'` or use convenience functions which set this automatically

**Memory errors with large datasets**
- Solution: Use `precomputed_gower_matrix` to compute distances in chunks
- Consider using GPU acceleration with `use_gpu=True`

**Poor performance with DataFrames**
- Solution: Explicitly specify `cat_features` instead of using `'auto'`

**Inconsistent results across runs**
- Solution: Set `random_state` parameter in sklearn algorithms

### Performance Comparison

| Method | Small Data (<1K) | Medium Data (1-10K) | Large Data (>10K) |
|--------|------------------|---------------------|-------------------|
| Direct metric | ✅ Recommended | ⚠️ Slower | ❌ Too slow |
| Precomputed | ⚠️ Overhead | ✅ Recommended | ✅ With GPU |
| GPU acceleration | ❌ Overhead | ✅ Good | ✅ Excellent |

## API Reference

See the complete API documentation for detailed parameter descriptions and examples.
