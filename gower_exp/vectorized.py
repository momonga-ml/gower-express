"""Vectorized implementations of Gower distance computation.

Provides optimized CPU and GPU vectorized operations using NumPy/CuPy broadcasting.
This module implements fully vectorized algorithms that eliminate Python loops
and use efficient broadcasting operations for computing all pairwise distances
simultaneously.
"""

import numpy as np

__all__ = [
    "gower_matrix_vectorized_gpu",
    "gower_matrix_vectorized",
]


def gower_matrix_vectorized_gpu(
    X_cat,
    X_num,
    Y_cat,
    Y_num,
    weight_cat,
    weight_num,
    weight_sum,
    num_ranges,
    is_symmetric,
    xp,
):
    """
    GPU-accelerated vectorized implementation of Gower distance matrix computation.
    Uses CuPy for GPU operations with same logic as CPU version.

    Parameters:
    -----------
    X_cat, Y_cat : ndarray
        Categorical features for datasets X and Y
    X_num, Y_num : ndarray
        Numerical features for datasets X and Y
    weight_cat, weight_num : ndarray
        Feature weights for categorical and numerical features
    weight_sum : float
        Total sum of all feature weights
    num_ranges : ndarray
        Range values for numerical feature normalization
    is_symmetric : bool
        Whether the distance matrix should be symmetric (X == Y)
    xp : module
        Array module (numpy or cupy)

    Returns:
    --------
    ndarray : Gower distance matrix of shape (n_samples_X, n_samples_Y)
    """
    x_n_rows = X_cat.shape[0]
    y_n_rows = Y_cat.shape[0]

    # Pre-allocate output with GPU memory pool management
    out = xp.zeros((x_n_rows, y_n_rows), dtype=xp.float32)

    # Handle categorical features with memory-efficient GPU processing
    if X_cat.shape[1] > 0:
        # Process in chunks to manage GPU memory more efficiently
        n_cat_features = X_cat.shape[1]
        # Adjust chunk size based on available GPU memory, prevent division by zero
        if x_n_rows * y_n_rows == 0:
            chunk_size = n_cat_features  # Process all features at once for empty arrays
        else:
            chunk_size = min(n_cat_features, max(1, 50000000 // (x_n_rows * y_n_rows)))

        for start_feat in range(0, n_cat_features, chunk_size):
            end_feat = min(start_feat + chunk_size, n_cat_features)

            X_chunk = X_cat[:, start_feat:end_feat]
            Y_chunk = Y_cat[:, start_feat:end_feat]
            weight_chunk = weight_cat[start_feat:end_feat]

            # Use memory views for broadcasting
            X_reshaped = X_chunk[:, xp.newaxis, :]
            Y_reshaped = Y_chunk[xp.newaxis, :, :]

            # Immediate computation and accumulation to reduce intermediate storage
            cat_diff = (X_reshaped != Y_reshaped).astype(xp.float32)
            weighted_diff = cat_diff * weight_chunk[xp.newaxis, xp.newaxis, :]
            out += xp.sum(weighted_diff, axis=2)

            # Force garbage collection on GPU
            del cat_diff, weighted_diff, X_reshaped, Y_reshaped

    # Handle numerical features with memory-efficient GPU processing
    if X_num.shape[1] > 0:
        # Process in chunks to manage GPU memory
        n_num_features = X_num.shape[1]
        # Prevent division by zero when y_n_rows is 0
        if x_n_rows * y_n_rows == 0:
            chunk_size = n_num_features  # Process all features at once for empty arrays
        else:
            chunk_size = min(n_num_features, max(1, 50000000 // (x_n_rows * y_n_rows)))

        for start_feat in range(0, n_num_features, chunk_size):
            end_feat = min(start_feat + chunk_size, n_num_features)

            X_chunk = X_num[:, start_feat:end_feat]
            Y_chunk = Y_num[:, start_feat:end_feat]
            weight_chunk = weight_num[start_feat:end_feat]
            ranges_chunk = num_ranges[start_feat:end_feat]

            # Memory views for broadcasting
            X_reshaped = X_chunk[:, xp.newaxis, :]
            Y_reshaped = Y_chunk[xp.newaxis, :, :]

            # Compute distances with immediate normalization
            abs_delta = xp.abs(X_reshaped - Y_reshaped)

            # In-place normalization where possible
            ranges_broadcast = ranges_chunk[xp.newaxis, xp.newaxis, :]
            mask = ranges_broadcast != 0
            xp.divide(abs_delta, ranges_broadcast, out=abs_delta, where=mask)

            # Apply weights and accumulate to output
            weighted_diff = abs_delta * weight_chunk[xp.newaxis, xp.newaxis, :]
            out += xp.sum(weighted_diff, axis=2)

            # Cleanup GPU memory
            del abs_delta, weighted_diff, X_reshaped, Y_reshaped

    # Normalize by total weight (in-place)
    out /= weight_sum

    # Ensure diagonal is zero for symmetric matrices
    if is_symmetric and x_n_rows == y_n_rows:
        xp.fill_diagonal(out, 0.0)

    return out


def gower_matrix_vectorized(
    X_cat,
    X_num,
    Y_cat,
    Y_num,
    weight_cat,
    weight_num,
    weight_sum,
    num_ranges,
    is_symmetric,
):
    """
    Fully vectorized implementation of Gower distance matrix computation.
    Uses NumPy broadcasting to compute all pairwise distances at once.

    This eliminates the row-by-row Python loop from the original implementation
    and computes all pairwise distances using efficient NumPy broadcasting operations.

    Key optimizations:
    - Broadcasting instead of nested loops
    - Vectorized categorical comparisons
    - Vectorized numerical distance calculations
    - Efficient handling of weights and normalization

    Parameters:
    -----------
    X_cat, Y_cat : ndarray
        Categorical features for datasets X and Y
    X_num, Y_num : ndarray
        Numerical features for datasets X and Y
    weight_cat, weight_num : ndarray
        Feature weights for categorical and numerical features
    weight_sum : float
        Total sum of all feature weights
    num_ranges : ndarray
        Range values for numerical feature normalization
    is_symmetric : bool
        Whether the distance matrix should be symmetric (X == Y)

    Returns:
    --------
    ndarray : Gower distance matrix of shape (n_samples_X, n_samples_Y)
    """
    x_n_rows = X_cat.shape[0]
    y_n_rows = Y_cat.shape[0]

    # Pre-allocate final output matrix with optimal memory layout
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32, order="C")

    # Handle categorical features using memory-efficient broadcasting
    if X_cat.shape[1] > 0:
        # Process categorical features in chunks to reduce peak memory usage
        n_cat_features = X_cat.shape[1]
        # Prevent division by zero when y_n_rows is 0
        if x_n_rows * y_n_rows == 0:
            chunk_size = n_cat_features  # Process all features at once for empty arrays
        else:
            chunk_size = min(
                n_cat_features, max(1, 100000000 // (x_n_rows * y_n_rows))
            )  # Adaptive chunking

        for start_feat in range(0, n_cat_features, chunk_size):
            end_feat = min(start_feat + chunk_size, n_cat_features)

            # Process chunk of categorical features
            X_chunk = X_cat[:, start_feat:end_feat]
            Y_chunk = Y_cat[:, start_feat:end_feat]
            weight_chunk = weight_cat[start_feat:end_feat]

            # Use view-based broadcasting instead of creating expanded copies
            X_reshaped = X_chunk[:, np.newaxis, :]
            Y_reshaped = Y_chunk[np.newaxis, :, :]

            # Vectorized comparison with immediate weighting to reduce memory
            cat_diff = (X_reshaped != Y_reshaped).astype(np.float32)

            # Apply weights and accumulate directly to output (in-place)
            weighted_diff = cat_diff * weight_chunk[np.newaxis, np.newaxis, :]
            out += np.sum(weighted_diff, axis=2)

            # Cleanup intermediate arrays to free memory
            del cat_diff, weighted_diff, X_reshaped, Y_reshaped

    # Handle numerical features using memory-efficient broadcasting
    if X_num.shape[1] > 0:
        # Process numerical features in chunks to reduce peak memory usage
        n_num_features = X_num.shape[1]
        # Prevent division by zero when y_n_rows is 0
        if x_n_rows * y_n_rows == 0:
            chunk_size = n_num_features  # Process all features at once for empty arrays
        else:
            chunk_size = min(
                n_num_features, max(1, 100000000 // (x_n_rows * y_n_rows))
            )  # Adaptive chunking

        for start_feat in range(0, n_num_features, chunk_size):
            end_feat = min(start_feat + chunk_size, n_num_features)

            # Process chunk of numerical features
            X_chunk = X_num[:, start_feat:end_feat]
            Y_chunk = Y_num[:, start_feat:end_feat]
            weight_chunk = weight_num[start_feat:end_feat]
            ranges_chunk = num_ranges[start_feat:end_feat]

            # Use view-based broadcasting
            X_reshaped = X_chunk[:, np.newaxis, :]
            Y_reshaped = Y_chunk[np.newaxis, :, :]

            # Vectorized numerical distance computation with immediate normalization
            abs_delta = np.abs(X_reshaped - Y_reshaped)

            # Normalize by ranges in-place where possible
            ranges_broadcast = ranges_chunk[np.newaxis, np.newaxis, :]
            mask = ranges_broadcast != 0
            np.divide(abs_delta, ranges_broadcast, out=abs_delta, where=mask)

            # Apply weights and accumulate directly to output (in-place)
            weighted_diff = abs_delta * weight_chunk[np.newaxis, np.newaxis, :]
            out += np.sum(weighted_diff, axis=2)

            # Cleanup intermediate arrays
            del abs_delta, weighted_diff, X_reshaped, Y_reshaped

    # Normalize by total weight (in-place)
    out /= weight_sum

    # For symmetric matrices, ensure diagonal is exactly 0 (unless all weights are zero)
    if is_symmetric and x_n_rows == y_n_rows and weight_sum > 0:
        np.fill_diagonal(out, 0.0)

    return out
