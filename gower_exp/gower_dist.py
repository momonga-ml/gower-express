import heapq
import os
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse

# Try to import numba for JIT compilation
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorators when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(x):
        return range(x)


# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    GPU_AVAILABLE = (
        cp.cuda.is_available() if hasattr(cp.cuda, "is_available") else False
    )
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback alias


def get_array_module(use_gpu=False):
    """Returns cupy or numpy based on availability and request"""
    if use_gpu and GPU_AVAILABLE:
        return cp
    return np


@jit(nopython=True, parallel=True)
def gower_get_numba(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    ranges_of_numeric,
):
    """
    Numba-optimized version of gower_get function.
    """
    n_rows = xj_cat.shape[0]
    result = np.zeros(n_rows, dtype=np.float32)

    for i in prange(n_rows):
        sum_cat = 0.0
        sum_num = 0.0

        # Categorical distance calculation
        for j in range(len(xi_cat)):
            if xi_cat[j] != xj_cat[i, j]:
                sum_cat += feature_weight_cat[j]

        # Numerical distance calculation
        for j in range(len(xi_num)):
            if ranges_of_numeric[j] != 0.0:
                abs_delta = abs(xi_num[j] - xj_num[i, j])
                sij_num = abs_delta / ranges_of_numeric[j]
                sum_num += feature_weight_num[j] * sij_num

        result[i] = (sum_cat + sum_num) / feature_weight_sum

    return result


@jit(nopython=True)
def compute_ranges_numba(Z_num, num_ranges, num_max):
    """
    Numba-optimized computation of ranges for numerical features.
    """
    num_cols = Z_num.shape[1]
    for col in range(num_cols):
        # Initialize min/max with first non-NaN value
        max_val = -np.inf
        min_val = np.inf

        # Find actual min/max values
        for row in range(Z_num.shape[0]):
            val = Z_num[row, col]
            if not np.isnan(val):
                if val > max_val:
                    max_val = val
                if val < min_val:
                    min_val = val

        # Handle case where all values are NaN
        if max_val == -np.inf or min_val == np.inf:
            max_val = 0.0
            min_val = 0.0

        num_max[col] = max_val
        if max_val != 0:
            num_ranges[col] = abs(1 - min_val / max_val)
        else:
            num_ranges[col] = 0.0


@jit(nopython=True)
def smallest_indices_numba(ary_flat, n):
    """
    Numba-optimized version of smallest_indices.
    """
    # Handle NaN values by replacing with large number
    for i in range(len(ary_flat)):
        if np.isnan(ary_flat[i]):
            ary_flat[i] = 999.0

    # Simple selection sort for the n smallest values
    indices = np.arange(len(ary_flat), dtype=np.int32)

    for i in range(min(n, len(ary_flat))):
        min_idx = i
        for j in range(i + 1, len(ary_flat)):
            if ary_flat[j] < ary_flat[min_idx]:
                min_idx = j
        # Swap values and indices
        ary_flat[i], ary_flat[min_idx] = ary_flat[min_idx], ary_flat[i]
        indices[i], indices[min_idx] = indices[min_idx], indices[i]

    return indices[:n], ary_flat[:n]


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

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 'A'], [2, 'B'], [3, 'A']])
    >>> cat_features = [False, True]
    >>> dist_matrix = gower_matrix(X, cat_features=cat_features)
    >>> print(dist_matrix.shape)
    (3, 3)
    """
    # Validate Inputs
    if data_x is None or (hasattr(data_x, "__len__") and len(data_x) == 0):
        raise ValueError("data_x cannot be None or empty")

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
        except Exception:
            # Fall back to CPU if GPU computation fails
            use_gpu = False
            xp = np

    # CPU computation path
    if True:
        if (
            n_jobs == 1 or x_n_rows < 100
        ):  # Use sequential for small datasets or single job
            # Original sequential implementation
            for i in range(x_n_rows):
                j_start = i
                if x_n_rows != y_n_rows:
                    j_start = 0
                # call the main function
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
                # print(res)
                out[i, j_start:] = res
                if x_n_rows == y_n_rows:
                    out[i:, j_start] = res
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
        print(
            f"Method used: {'GPU' if use_gpu and GPU_AVAILABLE else 'Numba' if NUMBA_AVAILABLE else 'NumPy'}"
        )

    return out


def _compute_gower_matrix_parallel(
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
):
    """
    Compute Gower distance matrix using parallel processing.

    This function splits the computation into chunks and processes them in parallel
    using joblib.Parallel. Each chunk computes a subset of rows in the distance matrix.
    """
    # Determine the actual number of jobs to use
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs < -1:
        n_jobs = max(1, os.cpu_count() + 1 + n_jobs)

    # Create chunks of row indices to process
    chunk_size = max(1, x_n_rows // n_jobs)
    row_chunks = []

    for i in range(0, x_n_rows, chunk_size):
        end_idx = min(i + chunk_size, x_n_rows)
        row_chunks.append((i, end_idx))

    # Process chunks in parallel using loky backend to avoid Numba conflicts
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_chunk)(
            start_idx,
            end_idx,
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
        )
        for start_idx, end_idx in row_chunks
    )

    # Aggregate results into the final output matrix
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    for (start_idx, end_idx), chunk_result in zip(row_chunks, results):
        out[start_idx:end_idx, :] = chunk_result

    # Handle symmetric matrix case - fill lower triangle
    if x_n_rows == y_n_rows:
        for i in range(x_n_rows):
            for j in range(i):
                out[i, j] = out[j, i]

    # For symmetric matrices, ensure diagonal is exactly 0
    if x_n_rows == y_n_rows:
        np.fill_diagonal(out, 0.0)

    return out


def _compute_chunk(
    start_idx,
    end_idx,
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
):
    """
    Compute a chunk of the Gower distance matrix.

    This function processes rows from start_idx to end_idx and returns
    the corresponding chunk of the distance matrix.
    """
    chunk_size = end_idx - start_idx
    chunk_out = np.zeros((chunk_size, y_n_rows), dtype=np.float32)

    for i in range(chunk_size):
        row_idx = start_idx + i
        j_start = row_idx
        if x_n_rows != y_n_rows:
            j_start = 0

        # call the main function
        res = gower_get(
            X_cat[row_idx, :],
            X_num[row_idx, :],
            Y_cat[j_start:y_n_rows, :],
            Y_num[j_start:y_n_rows, :],
            weight_cat,
            weight_num,
            weight_sum,
            cat_features,
            num_ranges,
            num_max,
        )

        chunk_out[i, j_start:] = res

        # Handle symmetric matrix case
        if x_n_rows == y_n_rows:
            # For symmetric matrices, we need to handle the upper/lower triangle properly
            # This implementation focuses on correctness rather than optimal symmetric handling
            pass

    return chunk_out


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

    # For symmetric matrices, ensure diagonal is exactly 0
    if is_symmetric and x_n_rows == y_n_rows:
        np.fill_diagonal(out, 0.0)

    return out


def gower_get(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    categorical_features,
    ranges_of_numeric,
    max_of_numeric,
):
    # Use numba-optimized version if available and arrays are compatible
    # Don't use numba for empty categorical arrays as it doesn't handle them properly
    if NUMBA_AVAILABLE and xi_cat.ndim == 1 and xj_cat.ndim == 2 and len(xi_cat) > 0:
        try:
            return gower_get_numba(
                xi_cat,
                xi_num,
                xj_cat,
                xj_num,
                feature_weight_cat,
                feature_weight_num,
                feature_weight_sum,
                ranges_of_numeric,
            )
        except Exception:
            # Fall back to numpy version if numba fails
            pass

    # Original numpy implementation as fallback
    # categorical columns
    if len(xi_cat) > 0:
        # Handle categorical comparison including NaN values
        # For string/object arrays, NaN comparisons work differently
        equal_mask = xi_cat == xj_cat

        # Handle cases where both are NaN (np.nan == np.nan is False, but both being NaN should be equal)
        # Use a try-catch approach since xi_cat might contain mixed types
        try:
            both_nan_mask = np.isnan(xi_cat.astype(float)) & np.isnan(
                xj_cat.astype(float)
            )
        except (ValueError, TypeError):
            # If can't convert to float, assume no NaN values in categorical data
            both_nan_mask = np.zeros_like(equal_mask, dtype=bool)
        final_equal_mask = equal_mask | both_nan_mask

        # Ensure sij_cat is numeric by using explicit float arrays
        sij_cat = np.where(
            final_equal_mask,
            np.zeros(xi_cat.shape, dtype=np.float32),
            np.ones(xi_cat.shape, dtype=np.float32),
        )
        sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)
    else:
        # Handle empty categorical arrays - return zeros with correct shape
        # When xi_cat is empty, the output should match the number of samples in xj
        # Use xj_num to determine the number of samples since it's the main data
        output_shape = xj_num.shape[0] if xj_num.ndim > 1 else 1
        sum_cat = np.zeros(output_shape, dtype=np.float32)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)

    # Handle NaN values properly: when both values are NaN, distance should be 0
    both_nan = np.isnan(xi_num) & np.isnan(xj_num)
    abs_delta = np.where(both_nan, 0.0, abs_delta)

    sij_num = np.divide(
        abs_delta,
        ranges_of_numeric,
        out=np.zeros_like(abs_delta),
        where=ranges_of_numeric != 0,
    )

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij


def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten().astype(np.float32)

    # Try numba version first
    if NUMBA_AVAILABLE:
        try:
            flat_copy = flat.copy()
            indices, values = smallest_indices_numba(flat_copy, n)
            return {"index": indices, "values": values}
        except Exception:
            # Fall back to numpy version
            pass

    # Original numpy implementation
    flat = np.nan_to_num(flat, nan=999)
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {"index": indices, "values": values}


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
    """

    if data_x.shape[0] >= 2:
        raise TypeError("Only support `data_x` of 1 row.")

    # Use optimized version for larger datasets where it's beneficial
    # The incremental algorithm has overhead, so only use for large datasets with small n
    if use_optimized and data_y is not None and data_y.shape[0] > 5000 and n < 50:
        return gower_topn_optimized(data_x, data_y, weight, cat_features, n)

    # Original implementation for backward compatibility
    dm = gower_matrix(data_x, data_y, weight, cat_features)
    return smallest_indices(np.nan_to_num(dm[0], nan=1), n)


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

    cat_dist = 0.0
    if len(query_cat) > 0:
        cat_diff = (query_cat != row_cat).astype(np.float32)
        cat_dist = np.dot(cat_diff, weight_cat)

    num_dist = 0.0
    if len(query_num) > 0:
        abs_delta = np.abs(query_num - row_num)
        normalized_delta = np.divide(
            abs_delta, num_ranges, out=np.zeros_like(abs_delta), where=num_ranges != 0
        )
        num_dist = np.dot(normalized_delta, weight_num)

    return (cat_dist + num_dist) / weight_sum
