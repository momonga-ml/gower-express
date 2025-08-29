"""Main Gower distance matrix computation.

Provides the primary user-facing API for computing Gower distance matrices.
This module serves as the main entry point for Gower distance calculations,
orchestrating various optimization strategies including GPU acceleration,
parallel processing, and vectorized operations.
"""

import time

import numpy as np
from scipy.sparse import issparse

# Import from our new modules
from .accelerators import (
    GPU_AVAILABLE,
    NUMBA_AVAILABLE,
    compute_ranges_numba,
    cp,
    get_array_module,
)
from .core import gower_get
from .parallel import _compute_gower_matrix_parallel
from .topn import (
    gower_topn_optimized,
    smallest_indices,
)
from .vectorized import gower_matrix_vectorized, gower_matrix_vectorized_gpu

# Re-export numpy for backward compatibility


def gower_matrix(
    data_x,
    data_y=None,
    weight=None,
    cat_features=None,
    n_jobs=1,
    use_gpu=False,
    verbose=False,
):
    """
    Compute the Gower distance matrix between data_x and data_y.

    Parameters:
    -----------
    data_x : array-like, shape (n_samples, n_features)
        First dataset
    data_y : array-like, shape (m_samples, n_features), optional
        Second dataset. If None, computes distance matrix within data_x
    weight : array-like, shape (n_features,), optional
        Feature weights. If None, uses equal weights
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features
    n_jobs : int, default=1
        Number of parallel jobs to use for computation.
        - 1: Use sequential processing (default)
        - -1: Use all available CPU cores
        - n > 1: Use n cores
        For small datasets (<100 samples), sequential processing is used regardless.
    use_gpu : bool, default=False
        Use GPU acceleration if available (requires CuPy)
    verbose : bool, default=False
        Verbose logging - will show run times

    Returns:
    --------
    ndarray, shape (n_samples, m_samples)
        Gower distance matrix

    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import gower_exp
    >>>
    >>> # Create mixed-type dataset
    >>> data = pd.DataFrame({
    ...     'age': [25, 32, 28, 35, 29],
    ...     'gender': ['M', 'F', 'M', 'F', 'M'],
    ...     'salary': [50000, 75000, 45000, 85000, 60000],
    ...     'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    ...     'experience': [2.5, 8.0, 3.5, 10.0, 5.5]
    ... })
    >>>
    >>> # Compute Gower distance matrix
    >>> distances = gower_exp.gower_matrix(data)
    >>> print(distances.shape)
    (5, 5)
    >>> print(f"Distance between samples 0 and 1: {distances[0, 1]:.3f}")
    Distance between samples 0 and 1: 0.567
    """
    # Validate Inputs
    if data_x is None:
        raise ValueError("data_x cannot be None")

    # Handle empty arrays
    if hasattr(data_x, "shape") and data_x.shape[0] == 0:
        if data_y is None:
            return np.array([]).reshape(0, 0).astype(np.float32)
        else:
            if hasattr(data_y, "shape"):
                return np.array([]).reshape(0, data_y.shape[0]).astype(np.float32)
            else:
                return np.array([]).reshape(0, 0).astype(np.float32)

    if weight is not None:
        weight = np.asarray(weight)
        if weight.shape[0] != data_x.shape[1]:
            raise ValueError(
                f"Weight dimension {weight.shape[0]} doesn't match feature dimension {data_x.shape[1]}"
            )
        if np.any(weight < 0):
            raise ValueError("Weights must be non-negative")

    if verbose:
        start_time = time.time()

    xp = get_array_module(use_gpu)

    # function checks
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y
    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if cat_features is None:
        if not isinstance(X, np.ndarray):

            def is_number_safe(dtype):
                try:
                    return not np.issubdtype(dtype, np.number)
                except TypeError:
                    # Handle pandas dtypes (like CategoricalDtype) that np.issubdtype can't process
                    return True  # Treat as categorical if we can't determine

            is_number = np.vectorize(is_number_safe)
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    # print(cat_features)

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)]

    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    # Use numba-optimized range calculation if available
    if NUMBA_AVAILABLE:
        try:
            compute_ranges_numba(Z_num, num_ranges, num_max)
        except Exception:
            # Fall back to numpy version
            for col in range(num_cols):
                col_array = Z_num[:, col].astype(np.float32)
                max_val = np.nanmax(col_array)
                min_val = np.nanmin(col_array)

                if np.isnan(max_val):
                    max_val = 0.0
                if np.isnan(min_val):
                    min_val = 0.0
                num_max[col] = max_val
                num_ranges[col] = (
                    np.abs(1 - min_val / max_val) if (max_val != 0) else 0.0
                )
    else:
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

    # This is to normalize the numeric values between 0 and 1.
    # Ensure Z_num is float to handle division properly
    Z_num = Z_num.astype(np.float32)
    Z_num = np.divide(Z_num, num_max, out=np.zeros_like(Z_num), where=num_max != 0)
    Z_cat = Z[:, cat_features]

    if weight is None:
        weight = np.ones(Z.shape[1])

    # print(weight)

    weight_cat = weight[cat_features]
    weight_num = weight[np.logical_not(cat_features)]

    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    weight_sum = weight.sum()

    X_cat = Z_cat[x_index,]
    X_num = Z_num[x_index,]
    Y_cat = Z_cat[y_index,]
    Y_num = Z_num[y_index,]

    # print(X_cat,X_num,Y_cat,Y_num)

    # Determine computation strategy
    if use_gpu and GPU_AVAILABLE:
        # Use GPU-accelerated vectorized computation
        try:
            # Transfer data to GPU
            X_cat_gpu = xp.asarray(X_cat)
            X_num_gpu = xp.asarray(X_num)
            Y_cat_gpu = xp.asarray(Y_cat)
            Y_num_gpu = xp.asarray(Y_num)
            weight_cat_gpu = xp.asarray(weight_cat)
            weight_num_gpu = xp.asarray(weight_num)
            num_ranges_gpu = xp.asarray(num_ranges)

            try:
                # Compute on GPU
                out_gpu = gower_matrix_vectorized_gpu(
                    X_cat_gpu,
                    X_num_gpu,
                    Y_cat_gpu,
                    Y_num_gpu,
                    weight_cat_gpu,
                    weight_num_gpu,
                    weight_sum,
                    num_ranges_gpu,
                    x_n_rows == y_n_rows,
                    xp,
                )

                # Transfer result back to CPU
                out = cp.asnumpy(out_gpu) if use_gpu else out_gpu
                return out
            finally:
                # Clean up GPU memory
                if use_gpu and cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            # Fall back to CPU if GPU computation fails
            use_gpu = False
            xp = np

    # CPU computation path
    if x_n_rows * y_n_rows < 1000000:
        # Use vectorized implementation for medium-sized datasets
        out = gower_matrix_vectorized(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            x_n_rows == y_n_rows,
        )
    elif (
        n_jobs == 1 or x_n_rows < 100
    ):  # Use sequential for small datasets or single job
        # Sequential implementation with symmetric optimization
        for i in range(x_n_rows):
            if x_n_rows == y_n_rows:
                # Symmetric case: only compute upper triangle (including diagonal)
                j_start = i
                res = gower_get(
                    X_cat[i, :],
                    X_num[i, :],
                    Y_cat[j_start:y_n_rows, :],
                    Y_num[j_start:y_n_rows, :],
                    weight_cat,
                    weight_num,
                    weight_sum,
                    cat_features,
                    num_ranges,
                    num_max,
                )
                # Fill upper triangle
                out[i, j_start:] = res
                # Fill lower triangle symmetrically (skip diagonal to avoid overwriting)
                if len(res) > 1:  # Only if there are off-diagonal elements
                    out[j_start + 1 :, i] = res[1:]
            else:
                # Non-symmetric case: compute full row
                res = gower_get(
                    X_cat[i, :],
                    X_num[i, :],
                    Y_cat,
                    Y_num,
                    weight_cat,
                    weight_num,
                    weight_sum,
                    cat_features,
                    num_ranges,
                    num_max,
                )
                out[i, :] = res
    else:
        # Use parallel processing for large datasets
        out = _compute_gower_matrix_parallel(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            cat_features,
            num_ranges,
            num_max,
            x_n_rows,
            y_n_rows,
            n_jobs,
        )

    if verbose:
        print(f"Gower matrix computed in {time.time() - start_time:.2f}s")  # type:ignore
        if use_gpu and GPU_AVAILABLE:
            method = "GPU-Vectorized"
        elif x_n_rows * y_n_rows < 1000000:
            method = "CPU-Vectorized"
        elif n_jobs > 1 and x_n_rows >= 100:
            method = "CPU-Parallel"
        else:
            method = "CPU-Sequential"
        print(f"Method used: {method}")
        if NUMBA_AVAILABLE:
            print("Numba acceleration: Enabled")
        else:
            print("Numba acceleration: Not available")

    return out


def gower_topn(
    data_x, data_y=None, weight=None, cat_features=None, n=5, use_optimized=True
):
    """
    Find the top-n closest matches using Gower distance.

    Parameters:
    -----------
    data_x : array-like, shape (1, n_features)
        Query point (must be single row)
    data_y : array-like, shape (m_samples, n_features)
        Dataset to search
    weight : array-like, optional
        Feature weights
    cat_features : array-like, optional
        Boolean mask for categorical features
    n : int, default=5
        Number of nearest neighbors to return
    use_optimized : bool, default=True
        Use incremental computation for better performance

    Returns:
    --------
    dict with 'index' and 'values' keys containing nearest neighbor indices and distances

    Examples:
    ---------
    >>> import pandas as pd
    >>> import gower_exp
    >>>
    >>> # Create dataset to search
    >>> data = pd.DataFrame({
    ...     'age': [25, 32, 28, 35, 29, 31],
    ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    ...     'salary': [50000, 75000, 45000, 85000, 60000, 70000],
    ...     'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC']
    ... })
    >>>
    >>> # Find 3 most similar to first row
    >>> query = data.iloc[[0]]  # Single row query
    >>> result = gower_exp.gower_topn(query, data, n=3)
    >>> print("Most similar indices:", result['index'])
    Most similar indices: [0 2 4]
    >>> print("Distances:", result['values'].round(3))
    Distances: [0.000 0.167 0.233]
    """

    if data_x.shape[0] >= 2:
        raise TypeError("Only support `data_x` of 1 row.")

    # NOTE: The heap-based optimization is currently disabled due to performance regression
    # See benchmark results showing 10x slower performance on large datasets
    # TODO: Implement proper vectorized top-N algorithm or use different approach

    # Disable optimized version until performance issues are resolved
    use_optimized = False  # Force disable until fixed

    if use_optimized and data_y is not None:
        n_samples = data_y.shape[0]
        # Use optimization only for very large datasets with small n
        # where computing full matrix would be expensive
        if n_samples >= 10000 and n <= 20:
            return gower_topn_optimized(data_x, data_y, weight, cat_features, n)

    # Original implementation for backward compatibility
    dm = gower_matrix(data_x, data_y, weight, cat_features)
    return smallest_indices(np.nan_to_num(dm[0], nan=1), n)
