### I Was Wrong: Start Simple, Then Move to More Complex

About four years ago, we released the package [DenseClus](https://github.com/awslabs/amazon-denseclus) for clustering tabular mixed data.

It seemed like a good idea at the time: take categorical and numeric features, then map them into a lower dimensional space for clustering.

However, it's quite intensive. You need to transform and clean the features, then run [UMAP](https://umap-learn.readthedocs.io/) at least two times to get the weighted representation of all features. The complexity is roughly O(N log N) for each UMAP run, plus the clustering algorithm on top. More importantly, it's stochastic (different results each run), requires hyperparameter tuning, and adds multiple preprocessing steps before you even start clustering.

I forgot that the Zen of Python applies to problem solving too:

```python
import this
```

`Simple is better than complex: Favor straightforward solutions over overly complicated ones.`

The embedding approach has its place, but it shouldn't be your first choice for every mixed-data clustering problem.


## How I Would Solve It Now

I would start simple and not reinvent the wheel. There's a great distance metric from the 1970s called [Gower](https://en.wikipedia.org/wiki/Gower%27s_distance) which was made for mixed-type data.

In essence, Gower is a weighted average of the dice distance of categorical variables and the Manhattan distance of numeric variables (with possible rank step for discrete). There's a lot written on it, so I encourage you to look at the linked Wikipedia page or run a search, as covering it is out of scope for this post.

That said, Gower requires O(N² × F) time to calculate the full distance matrix (where F is the number of features), which needs to be stored in memory. While this scales quadratically, it's deterministic, interpretable, and eliminates the complexity of embedding hyperparameter tuning.

## Why Try Gower First?

Before jumping to embedding-based approaches, Gower offers several compelling advantages:

**Deterministic Results**: Same input always produces the same output. No random seeds, no variability between runs. Your clustering results are reproducible.

**Zero Hyperparameters**: Works out of the box. No need to tune `n_neighbors`, `min_dist`, `n_components`, or any other parameters that can dramatically affect your results.

**Interpretable Distances**: You can examine the actual distance calculation. When two samples are similar or different, you know exactly why.

**Faster Iteration**: Skip the embedding step entirely. Compute distances and cluster immediately, making experimentation much faster.

**Predictable Memory Usage**: You know exactly what you're storing (an N×N distance matrix), making memory planning straightforward.

## Making Something Even Better

In Python, there's a good package [gower](https://pypi.org/project/gower/) for it already. However, it looks like it's not recently updated.
It's 2025.
I have a weekend, Claude Code, and some knowledge of Machine Learning – can I make this better with a fork?

The answer appears to be: yes, yes we can.

## Gower Express

Proud to open source and introduce [Gower Express](https://github.com/momonga-ml/gower-express), an optimized version of Gower that can perform about 20% faster with 40% less memory usage.

```python
uv pip install gower_exp[gpu,sklearn]
```

### Features

    - Numba JIT compilation
    - Scikit-learn compatibility
    - Runs on GPUs
    - Automatic feature type detection
    - Missing value handling

### Examples

Easy to use with clustering like the following for full clustering:
```python
import gower_exp as gower
from sklearn.cluster import AgglomerativeClustering
distances = gower.gower_matrix(customer_data)
clusters = AgglomerativeClustering(affinity='precomputed', linkage='average').fit(distances)
```

If you just need the most similar than use `topn` (heap optimization that won't run in quadratic time):

```python
import gower_exp as gower
product_distances = gower.gower_matrix(product_catalog)
recommendations = gower.gower_topn(target_product, product_catalog, n=10)
```

CuPy integration is setup so if you need to run on massive dataset. Just set `use_gpu=True`.

```python
# Find similar patients for treatment recommendations
patient_similarity = gower.gower_matrix(patient_records, use_gpu=True)
```

## Results: Performance That Scales

| Dataset Size | CPU Time | GPU Time | Memory Usage |
|--------------|----------|----------|--------------|
| 1K records   | 0.08s    | 0.05s    | 12MB         |
| 10K records  | 2.1s     | 0.8s     | 180MB        |
| 100K records | 45s      | 12s      | 1.2GB        |
| 1M records   | 18min    | 3.8min   | 8GB          |


## Conclusion

If you've gotten this far, thanks for reading. It's okay to be wrong about things because it's an opportunity to learn.

**Start with Gower for its simplicity and interpretability.** [Gower](https://github.com/momonga-ml/gower-express) is a mature, well-understood distance metric that works immediately without hyperparameter tuning. You'll get reproducible results and can focus on understanding your data rather than debugging embedding parameters.

Only move to embedding-based approaches when you hit specific limitations. Most clustering tasks on mixed data can be solved more directly with Gower distances.
