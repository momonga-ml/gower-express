"""Top-N nearest neighbor search using Gower distance.

Provides heap-based optimized algorithms for finding the N closest matches.
This module implements efficient algorithms for top-N search that avoid computing
the full distance matrix when only the nearest neighbors are needed.
"""

import heapq
import logging

__all__ = [
    "smallest_indices",
    "gower_topn_optimized",
    "_gower_topn_heap",
    "_compute_single_distance",
]

import numpy as np

logger = logging.getLogger(__name__)

from .accelerators import NUMBA_AVAILABLE, smallest_indices_numba  # noqa: E402


def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten().astype(np.float32)

    # Try numba version first
    if NUMBA_AVAILABLE:
        try:
            flat_copy = flat.copy()
            indices, values = smallest_indices_numba(flat_copy, n)
            return {"index": indices, "values": values}
        except Exception as e:
            # Fall back to numpy version
            logger.debug(
                "Numba optimization failed for topn, using numpy fallback: %s", str(e)
            )

    # Original numpy implementation
    flat = np.nan_to_num(flat, nan=999)
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {"index": indices, "values": values}


def gower_topn_optimized(data_x, data_y, weight=None, cat_features=None, n=5):
    """
    Optimized top-N implementation using incremental distance computation.
    Only computes necessary distances instead of full matrix.
    """

    # Input validation
    X = data_x
    Y = data_y

    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    # Setup feature types
    x_n_cols = X.shape[1]
    y_n_rows = Y.shape[0]

    if cat_features is None:
        if not isinstance(X, np.ndarray):
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    # Convert to numpy arrays
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    # Prepare data
    Z = np.concatenate((X, Y))

    # Split numerical and categorical features
    Z_num = Z[:, np.logical_not(cat_features)]
    Z_cat = Z[:, cat_features]

    # Calculate ranges for numerical features
    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32)
        max_val = np.nanmax(col_array)
        min_val = np.nanmin(col_array)

        if np.isnan(max_val):
            max_val = 0.0
        if np.isnan(min_val):
            min_val = 0.0
        num_max[col] = max_val
        num_ranges[col] = np.abs(1 - min_val / max_val) if (max_val != 0) else 0.0

    # Normalize numerical features
    Z_num = np.divide(Z_num, num_max, out=np.zeros_like(Z_num), where=num_max != 0)

    # Setup weights
    if weight is None:
        weight = np.ones(Z.shape[1])

    weight_cat = weight[cat_features]
    weight_num = weight[np.logical_not(cat_features)]
    weight_sum = weight.sum()

    # Get query data
    query_cat = Z_cat[0, :]
    query_num = Z_num[0, :]

    # Get dataset data
    data_cat = Z_cat[1:, :]
    data_num = Z_num[1:, :]

    # Use heap-based algorithm for top-N
    return _gower_topn_heap(
        query_cat,
        query_num,
        data_cat,
        data_num,
        weight_cat,
        weight_num,
        weight_sum,
        num_ranges,
        n,
        y_n_rows,
    )


def _gower_topn_heap(
    query_cat,
    query_num,
    data_cat,
    data_num,
    weight_cat,
    weight_num,
    weight_sum,
    num_ranges,
    n,
    total_rows,
):
    """
    Heap-based incremental top-N computation.
    Uses max-heap to maintain top-N candidates with early stopping.
    """

    # Initialize heap with first n distances
    heap = []
    n_actual = min(n, total_rows)

    for i in range(n_actual):
        # Compute distance for row i
        dist = _compute_single_distance(
            query_cat,
            query_num,
            data_cat[i, :] if data_cat.ndim > 1 else data_cat,
            data_num[i, :] if data_num.ndim > 1 else data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )
        # Use negative distance for max-heap behavior
        heapq.heappush(heap, (-dist, i))

    # Early stopping threshold
    if heap:
        max_dist = -heap[0][0]

    # Process remaining rows with early stopping
    for i in range(n_actual, total_rows):
        # Compute distance
        dist = _compute_single_distance(
            query_cat,
            query_num,
            data_cat[i, :] if data_cat.ndim > 1 else data_cat,
            data_num[i, :] if data_num.ndim > 1 else data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        # Only update heap if distance is better
        if dist < max_dist:
            heapq.heapreplace(heap, (-dist, i))
            max_dist = -heap[0][0]

    # Extract results from heap
    results = sorted(heap, key=lambda x: -x[0])  # Sort by distance (ascending)
    indices = np.array([idx for _, idx in results], dtype=np.int32)
    distances = np.array([-dist for dist, _ in results], dtype=np.float32)

    return {"index": indices, "values": distances}


def _compute_single_distance(
    query_cat,
    query_num,
    row_cat,
    row_num,
    weight_cat,
    weight_num,
    weight_sum,
    num_ranges,
):
    """
    Compute Gower distance between query and a single row.
    """

    # Categorical distance
    cat_dist = 0.0
    if len(query_cat) > 0:
        cat_diff = (query_cat != row_cat).astype(np.float32)
        cat_dist = np.dot(cat_diff, weight_cat)

    # Numerical distance
    num_dist = 0.0
    if len(query_num) > 0:
        abs_delta = np.abs(query_num - row_num)
        normalized_delta = np.divide(
            abs_delta, num_ranges, out=np.zeros_like(abs_delta), where=num_ranges != 0
        )
        num_dist = np.dot(normalized_delta, weight_num)

    # Combined distance
    return (cat_dist + num_dist) / weight_sum
