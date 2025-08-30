"""Top-N nearest neighbor search using Gower distance.

Provides heap-based optimized algorithms for finding the N closest matches.
This module implements efficient algorithms for top-N search that avoid computing
the full distance matrix when only the nearest neighbors are needed.
"""

import logging

__all__ = [
    "smallest_indices",
    "gower_topn_optimized",
    "_gower_topn_heap",
    "_compute_batch_distances_vectorized",
]

import numpy as np

logger = logging.getLogger(__name__)

from .accelerators import (  # noqa: E402
    NUMBA_AVAILABLE,
    smallest_indices_numba,
    smallest_indices_numba_heap,
)


def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten().astype(np.float32)
    flat = np.nan_to_num(flat, nan=999)

    # Handle edge case where n >= array length
    if n >= len(flat):
        indices = np.argsort(flat)
        values = flat[indices]
        return {"index": indices, "values": values}

    # Try optimized numba version first
    if NUMBA_AVAILABLE:
        try:
            flat_copy = flat.copy().astype(np.float32)
            # Use heap-based algorithm for better performance on larger arrays
            if len(flat_copy) > 100:
                indices, values = smallest_indices_numba_heap(flat_copy, n)
            else:
                indices, values = smallest_indices_numba(
                    flat_copy.astype(np.float64), n
                )
                values = values.astype(np.float32)
            return {"index": indices, "values": values}
        except Exception as e:
            # Fall back to numpy version
            logger.debug(
                "Numba optimization failed for topn, using numpy fallback: %s", str(e)
            )

    # Original numpy implementation
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
    Vectorized top-N computation using batch processing and argpartition.
    Much faster than heap-based row-by-row approach.
    """

    n_actual = min(n, total_rows)

    # Determine optimal batch size based on dataset size and memory constraints
    # For most cases, larger batches are better to amortize overhead
    if total_rows <= 5000:
        batch_size = total_rows  # Process all at once for moderate datasets
    elif total_rows <= 15000:
        batch_size = 5000  # Larger batches for efficiency
    else:
        batch_size = 10000  # Very large batches for very large datasets

    all_distances = []
    all_indices = []

    # Process in batches using vectorized operations
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)

        # Vectorized distance computation for entire batch
        batch_distances = _compute_batch_distances_vectorized(
            query_cat,
            query_num,
            data_cat[start_idx:end_idx] if data_cat.ndim > 1 else data_cat,
            data_num[start_idx:end_idx] if data_num.ndim > 1 else data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        all_distances.extend(batch_distances)
        all_indices.extend(range(start_idx, end_idx))

    # Convert to numpy arrays for efficient processing
    distances = np.array(all_distances, dtype=np.float32)
    indices = np.array(all_indices, dtype=np.int32)

    # Use argpartition for efficient top-N selection (O(n) instead of O(n log n))
    if n_actual >= len(distances):
        # If n is larger than data, return everything sorted
        sorted_idx = np.argsort(distances)
    else:
        # Use argpartition to find n smallest distances efficiently
        partitioned_idx = np.argpartition(distances, n_actual)[:n_actual]
        # Sort only the top-n results
        sorted_idx = partitioned_idx[np.argsort(distances[partitioned_idx])]

    result_indices = indices[sorted_idx]
    result_distances = distances[sorted_idx]

    return {"index": result_indices, "values": result_distances}


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


def _compute_single_distance_cached(
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
    Optimized distance computation with minimal overhead.
    Uses vectorized operations where possible.
    """

    # Fast path for common cases
    total_dist = 0.0

    # Categorical distance - vectorized comparison
    if len(query_cat) > 0 and len(weight_cat) > 0:
        cat_matches = query_cat == row_cat
        cat_dist = np.sum(weight_cat[~cat_matches])
        total_dist += cat_dist

    # Numerical distance - vectorized operations
    if len(query_num) > 0 and len(weight_num) > 0:
        # Only compute for non-zero ranges to avoid division
        valid_ranges = num_ranges != 0
        if np.any(valid_ranges):
            abs_delta = np.abs(query_num[valid_ranges] - row_num[valid_ranges])
            normalized_delta = abs_delta / num_ranges[valid_ranges]
            num_dist = np.dot(normalized_delta, weight_num[valid_ranges])
            total_dist += num_dist

    return total_dist / weight_sum


def _compute_batch_distances_vectorized(
    query_cat,
    query_num,
    batch_data_cat,
    batch_data_num,
    weight_cat,
    weight_num,
    weight_sum,
    num_ranges,
):
    """
    Fully vectorized computation of distances for a batch of data.
    Uses broadcast operations for maximum efficiency.
    """

    # Handle single row case and ensure proper array structure
    if batch_data_cat.ndim == 1:
        batch_data_cat = batch_data_cat.reshape(1, -1)
    if batch_data_num.ndim == 1:
        batch_data_num = batch_data_num.reshape(1, -1)

    batch_size = batch_data_cat.shape[0]
    total_distances = np.zeros(batch_size, dtype=np.float32)

    # Categorical distance computation (vectorized)
    if len(query_cat) > 0 and len(weight_cat) > 0 and batch_data_cat.shape[1] > 0:
        try:
            # Ensure categorical data is properly handled
            query_cat_broadcast = np.asarray(query_cat)
            batch_data_cat_array = np.asarray(batch_data_cat)

            # Broadcast comparison: (batch_size, n_cat_features) != (n_cat_features,)
            cat_mismatches = batch_data_cat_array != query_cat_broadcast[np.newaxis, :]
            # Sum weighted mismatches for each row
            cat_distances = np.dot(
                cat_mismatches.astype(np.float32), weight_cat.astype(np.float32)
            )
            total_distances += cat_distances.astype(np.float32)
        except Exception:
            # Fall back to row-by-row for categorical if vectorized fails
            for i in range(batch_size):
                cat_diff = (batch_data_cat[i] != query_cat).astype(np.float32)
                cat_dist = np.dot(cat_diff, weight_cat.astype(np.float32))
                total_distances[i] += cat_dist

    # Numerical distance computation (vectorized)
    if len(query_num) > 0 and len(weight_num) > 0 and batch_data_num.shape[1] > 0:
        # Only process features with non-zero ranges to avoid division by zero
        valid_ranges_mask = num_ranges != 0

        if np.any(valid_ranges_mask):
            try:
                # Extract valid features and ensure float types
                valid_query_num = query_num[valid_ranges_mask].astype(np.float32)
                valid_batch_num = batch_data_num[:, valid_ranges_mask].astype(
                    np.float32
                )
                valid_weight_num = weight_num[valid_ranges_mask].astype(np.float32)
                valid_num_ranges = num_ranges[valid_ranges_mask].astype(np.float32)

                # Broadcast absolute difference: (batch_size, n_num_features) - (n_num_features,)
                abs_deltas = np.abs(valid_batch_num - valid_query_num[np.newaxis, :])

                # Normalize by ranges: (batch_size, n_num_features) / (n_num_features,)
                normalized_deltas = abs_deltas / valid_num_ranges[np.newaxis, :]

                # Weight and sum for each row: (batch_size,)
                num_distances = np.dot(normalized_deltas, valid_weight_num)
                total_distances += num_distances.astype(np.float32)
            except Exception:
                # Fall back to row-by-row for numerical if vectorized fails
                for i in range(batch_size):
                    valid_query = query_num[valid_ranges_mask].astype(np.float32)
                    valid_row = batch_data_num[i, valid_ranges_mask].astype(np.float32)
                    valid_weight = weight_num[valid_ranges_mask].astype(np.float32)
                    valid_ranges = num_ranges[valid_ranges_mask].astype(np.float32)

                    abs_delta = np.abs(valid_row - valid_query)
                    normalized_delta = abs_delta / valid_ranges
                    num_dist = np.dot(normalized_delta, valid_weight)
                    total_distances[i] += num_dist

    # Normalize by total weight
    return (total_distances / weight_sum).astype(np.float32)
