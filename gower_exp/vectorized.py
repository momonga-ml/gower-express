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

    # Handle categorical features using broadcasting
    if X_cat.shape[1] > 0:
        # Reshape for broadcasting
        X_cat_expanded = X_cat[:, xp.newaxis, :]
        Y_cat_expanded = Y_cat[xp.newaxis, :, :]

        # Vectorized categorical comparison
        cat_diff = (X_cat_expanded != Y_cat_expanded).astype(xp.float32)

        # Apply weights and sum across features
        weighted_cat_diff = cat_diff * weight_cat[xp.newaxis, xp.newaxis, :]
        cat_distances = xp.sum(weighted_cat_diff, axis=2)
    else:
        cat_distances = xp.zeros((x_n_rows, y_n_rows), dtype=xp.float32)

    # Handle numerical features using broadcasting
    if X_num.shape[1] > 0:
        # Reshape for broadcasting
        X_num_expanded = X_num[:, xp.newaxis, :]
        Y_num_expanded = Y_num[xp.newaxis, :, :]

        # Vectorized numerical distance computation
        abs_delta = xp.abs(X_num_expanded - Y_num_expanded)

        # Normalize by ranges
        normalized_delta = xp.divide(
            abs_delta,
            num_ranges[xp.newaxis, xp.newaxis, :],
            out=xp.zeros_like(abs_delta),
            where=num_ranges[xp.newaxis, xp.newaxis, :] != 0,
        )

        # Apply weights and sum across features
        weighted_num_diff = normalized_delta * weight_num[xp.newaxis, xp.newaxis, :]
        num_distances = xp.sum(weighted_num_diff, axis=2)
    else:
        num_distances = xp.zeros((x_n_rows, y_n_rows), dtype=xp.float32)

    # Combine distances and normalize
    total_distances = cat_distances + num_distances
    out = total_distances / weight_sum

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

    # Handle categorical features using broadcasting
    if X_cat.shape[1] > 0:
        # Reshape for broadcasting: X_cat (x_n_rows, 1, n_cat_features), Y_cat (1, y_n_rows, n_cat_features)
        X_cat_expanded = X_cat[:, np.newaxis, :]  # Shape: (x_n_rows, 1, n_cat_features)
        Y_cat_expanded = Y_cat[np.newaxis, :, :]  # Shape: (1, y_n_rows, n_cat_features)

        # Vectorized categorical comparison - 1 if different, 0 if same
        cat_diff = (X_cat_expanded != Y_cat_expanded).astype(np.float32)

        # Apply weights and sum across features
        weighted_cat_diff = cat_diff * weight_cat[np.newaxis, np.newaxis, :]
        cat_distances = np.sum(weighted_cat_diff, axis=2)
    else:
        cat_distances = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    # Handle numerical features using broadcasting
    if X_num.shape[1] > 0:
        # Reshape for broadcasting: X_num (x_n_rows, 1, n_num_features), Y_num (1, y_n_rows, n_num_features)
        X_num_expanded = X_num[:, np.newaxis, :]  # Shape: (x_n_rows, 1, n_num_features)
        Y_num_expanded = Y_num[np.newaxis, :, :]  # Shape: (1, y_n_rows, n_num_features)

        # Vectorized numerical distance computation
        abs_delta = np.abs(X_num_expanded - Y_num_expanded)

        # Normalize by ranges, handling division by zero
        normalized_delta = np.divide(
            abs_delta,
            num_ranges[np.newaxis, np.newaxis, :],
            out=np.zeros_like(abs_delta),
            where=num_ranges[np.newaxis, np.newaxis, :] != 0,
        )

        # Apply weights and sum across features
        weighted_num_diff = normalized_delta * weight_num[np.newaxis, np.newaxis, :]
        num_distances = np.sum(weighted_num_diff, axis=2)
    else:
        num_distances = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    # Combine categorical and numerical distances
    total_distances = cat_distances + num_distances

    # Normalize by total weight
    out = total_distances / weight_sum

    # For symmetric matrices, ensure diagonal is exactly 0 (unless all weights are zero)
    if is_symmetric and x_n_rows == y_n_rows and weight_sum > 0:
        np.fill_diagonal(out, 0.0)

    return out
