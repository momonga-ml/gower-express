"""Hardware acceleration utilities for Gower distance computation.

Provides GPU support via CuPy and JIT compilation via Numba with graceful fallbacks.
This module handles all hardware-specific optimizations including:
- Numba JIT compilation for CPU acceleration
- CuPy GPU support for large-scale computations
- Graceful fallback mechanisms when acceleration is unavailable
"""

import numpy as np

__all__ = [
    "GPU_AVAILABLE",
    "NUMBA_AVAILABLE",
    "cp",
    "jit",
    "prange",
    "get_array_module",
    "gower_get_numba",
    "gower_get_numba_numerical_only",
    "gower_get_numba_categorical_only",
    "gower_get_numba_mixed_optimized",
    "compute_ranges_numba",
    "compute_ranges_numba_parallel",
    "smallest_indices_numba",
    "smallest_indices_numba_heap",
    "gower_matrix_numba_parallel",
]

# Try to import numba for JIT compilation
try:
    import numba.core.types as nb_types  # noqa:F401
    from numba import jit, prange, types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy types namespace when numba is not available
    class types:  # noqa:N801
        @staticmethod
        def Tuple(args):  # noqa:N802
            def decorator(func):
                return func

            return decorator

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


@jit(
    "float32[:](float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64, float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
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
        has_nan = False

        # Categorical distance calculation
        for j in range(len(xi_cat)):
            # Handle NaN values: if both are NaN, they are considered equal
            xi_val = xi_cat[j]
            xj_val = xj_cat[i, j]

            # Check if both values are NaN
            both_nan = np.isnan(xi_val) and np.isnan(xj_val)

            # If both NaN, distance is 0 (equal), so skip
            if both_nan:
                continue

            # If one is NaN or values are different, add to categorical distance
            if np.isnan(xi_val) or np.isnan(xj_val) or xi_val != xj_val:
                sum_cat += feature_weight_cat[j]

        # Numerical distance calculation
        for j in range(len(xi_num)):
            if ranges_of_numeric[j] != 0.0:
                xi_val = xi_num[j]
                xj_val = xj_num[i, j]

                # Handle NaN values: when both values are NaN, distance should be 0
                both_nan = np.isnan(xi_val) and np.isnan(xj_val)

                if both_nan:
                    abs_delta = 0.0
                else:
                    abs_delta = abs(xi_val - xj_val)
                    # If abs_delta is NaN (one value is NaN), mark this row as having NaN
                    if np.isnan(abs_delta):
                        has_nan = True
                        break

                sij_num = abs_delta / ranges_of_numeric[j]
                sum_num += feature_weight_num[j] * sij_num

        if has_nan:
            result[i] = np.nan
        else:
            result[i] = (sum_cat + sum_num) / feature_weight_sum

    return result


@jit(
    "void(float64[:,:], float64[:], float64[:])",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
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


@jit(
    "types.Tuple([int32[:], float64[:]])(float64[:], int32)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def smallest_indices_numba(ary_flat, n):
    """
    Numba-optimized version of smallest_indices.
    """
    length = len(ary_flat)
    # Handle edge cases
    if n >= length:
        indices = np.argsort(ary_flat).astype(np.int32)
        sorted_values = ary_flat[indices].astype(np.float64)
        return indices, sorted_values
    if n <= 0:
        empty_indices = np.empty(0, dtype=np.int32)
        empty_values = np.empty(0, dtype=np.float64)
        return empty_indices, empty_values

    # Create a copy to preserve original NaN positions
    original_values = ary_flat.copy()

    # Handle NaN values by replacing with large number for sorting
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

    # Restore original values (including NaNs) for the result
    result_values = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result_values[i] = original_values[indices[i]]

    return indices[:n], result_values


# ============================================================================
# SPECIALIZED OPTIMIZED KERNELS
# ============================================================================


@jit(
    "float32[:](float64[:], float64[:,:], float64[:], float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def gower_get_numba_numerical_only(
    xi_num,
    xj_num,
    feature_weight_num,
    ranges_of_numeric,
):
    """
    Specialized Numba kernel for pure numerical data.
    Optimized for datasets with only numerical features.
    """
    n_rows = xj_num.shape[0]
    result = np.zeros(n_rows, dtype=np.float32)
    weight_sum = feature_weight_num.sum()

    for i in prange(n_rows):
        sum_num = 0.0
        has_nan = False

        # Numerical distance calculation - optimized loop
        for j in range(len(xi_num)):
            if ranges_of_numeric[j] != 0.0:
                xi_val = xi_num[j]
                xj_val = xj_num[i, j]

                # Handle NaN values efficiently
                both_nan = np.isnan(xi_val) and np.isnan(xj_val)
                if both_nan:
                    # Both NaN, distance is 0, continue (no contribution to sum)
                    continue
                elif np.isnan(xi_val) or np.isnan(xj_val):
                    # One is NaN, mark row as having NaN and break
                    has_nan = True
                    break

                abs_delta = abs(xi_val - xj_val)
                sij_num = abs_delta / ranges_of_numeric[j]
                sum_num += feature_weight_num[j] * sij_num

        if has_nan:
            result[i] = np.nan
        else:
            result[i] = sum_num / weight_sum

    return result


@jit(
    "float32[:](float64[:], float64[:,:], float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def gower_get_numba_categorical_only(
    xi_cat,
    xj_cat,
    feature_weight_cat,
):
    """
    Specialized Numba kernel for pure categorical data.
    Optimized for datasets with only categorical features.
    """
    n_rows = xj_cat.shape[0]
    result = np.zeros(n_rows, dtype=np.float32)
    weight_sum = feature_weight_cat.sum()

    for i in prange(n_rows):
        sum_cat = 0.0

        # Categorical distance calculation - optimized loop
        for j in range(len(xi_cat)):
            xi_val = xi_cat[j]
            xj_val = xj_cat[i, j]

            # Handle NaN values: if both are NaN, they are considered equal
            both_nan = np.isnan(xi_val) and np.isnan(xj_val)

            # If both NaN, distance is 0 (equal), so skip
            if both_nan:
                continue

            # If one is NaN or values are different, add to categorical distance
            if np.isnan(xi_val) or np.isnan(xj_val) or xi_val != xj_val:
                sum_cat += feature_weight_cat[j]

        result[i] = sum_cat / weight_sum

    return result


@jit(
    "float32[:](float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64, float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def gower_get_numba_mixed_optimized(
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
    Optimized version of mixed-type Gower distance computation.
    Improved memory access patterns and vectorization hints.
    """
    n_rows = xj_cat.shape[0]
    result = np.zeros(n_rows, dtype=np.float32)

    for i in prange(n_rows):
        sum_cat = 0.0
        sum_num = 0.0
        has_nan = False

        # Process categorical features first (better cache locality)
        for j in range(len(xi_cat)):
            xi_val = xi_cat[j]
            xj_val = xj_cat[i, j]

            # Optimized NaN handling
            both_nan = np.isnan(xi_val) and np.isnan(xj_val)

            # If both NaN, distance is 0 (equal), so skip
            if both_nan:
                continue

            # If one is NaN or values are different, add to categorical distance
            if np.isnan(xi_val) or np.isnan(xj_val) or xi_val != xj_val:
                sum_cat += feature_weight_cat[j]

        # Process numerical features with early termination
        for j in range(len(xi_num)):
            range_val = ranges_of_numeric[j]
            if range_val != 0.0:
                xi_val = xi_num[j]
                xj_val = xj_num[i, j]

                # Optimized NaN handling with early exit
                if np.isnan(xi_val) and np.isnan(xj_val):
                    continue  # Both NaN, distance is 0
                elif np.isnan(xi_val) or np.isnan(xj_val):
                    has_nan = True
                    break  # Early exit on first NaN mismatch

                abs_delta = abs(xi_val - xj_val)
                sij_num = abs_delta / range_val
                sum_num += feature_weight_num[j] * sij_num

        if has_nan:
            result[i] = np.nan
        else:
            result[i] = (sum_cat + sum_num) / feature_weight_sum

    return result


# ============================================================================
# ENHANCED UTILITY FUNCTIONS
# ============================================================================


@jit(
    "void(float64[:,:], float64[:], float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def compute_ranges_numba_parallel(Z_num, num_ranges, num_max):
    """
    Enhanced parallel version of range computation with better memory access patterns.
    """
    num_cols = Z_num.shape[1]

    # Process columns in parallel for better cache utilization
    for col in prange(num_cols):
        max_val = -np.inf
        min_val = np.inf

        # Sequential scan of rows within each column for better memory locality
        for row in range(Z_num.shape[0]):
            val = Z_num[row, col]
            if not np.isnan(val):
                # Use min/max without branching for better performance
                max_val = max(max_val, val)
                min_val = min(min_val, val)

        # Handle all-NaN case
        if max_val == -np.inf:
            max_val = 0.0
            min_val = 0.0

        num_max[col] = max_val
        if max_val != 0.0:
            num_ranges[col] = abs(1.0 - min_val / max_val)
        else:
            num_ranges[col] = 0.0


@jit(
    "void(float32[:], int32[:], int32, int32)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _heap_sift_down(heap_values, heap_indices, start, end):
    """Helper function for heap operations"""
    root = start
    while True:
        child = root * 2 + 1
        if child > end:
            break

        if child + 1 <= end and heap_values[child] < heap_values[child + 1]:
            child += 1

        if heap_values[root] < heap_values[child]:
            heap_values[root], heap_values[child] = (
                heap_values[child],
                heap_values[root],
            )
            heap_indices[root], heap_indices[child] = (
                heap_indices[child],
                heap_indices[root],
            )
            root = child
        else:
            break


@jit(
    "types.Tuple([int32[:], float32[:]])(float32[:], int32)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def smallest_indices_numba_heap(ary_flat, n):
    """
    Optimized heap-based top-N selection using Numba.
    Significantly faster than selection sort for larger arrays.
    """
    length = len(ary_flat)

    # Handle edge cases
    if n >= length:
        indices = np.argsort(ary_flat).astype(np.int32)
        sorted_values = ary_flat[indices].astype(np.float32)
        return indices, sorted_values

    if n <= 0:
        empty_indices = np.empty(0, dtype=np.int32)
        empty_values = np.empty(0, dtype=np.float32)
        return empty_indices, empty_values

    # Replace NaN with large value for sorting
    working_array = ary_flat.copy()
    for i in range(length):
        if np.isnan(working_array[i]):
            working_array[i] = 999.0

    # Use partial sort (argpartition equivalent) for O(n) average case
    # Build min-heap with first n elements
    heap_indices = np.arange(n, dtype=np.int32)
    heap_values = working_array[:n].copy()

    # Heapify
    for i in range(n // 2 - 1, -1, -1):
        _heap_sift_down(heap_values, heap_indices, i, n - 1)

    # Process remaining elements
    for i in range(n, length):
        if working_array[i] < heap_values[0]:
            heap_values[0] = working_array[i]
            heap_indices[0] = i
            _heap_sift_down(heap_values, heap_indices, 0, n - 1)

    # Sort the final n elements
    for i in range(n - 1, 0, -1):
        # Swap first and i-th elements
        heap_values[0], heap_values[i] = heap_values[i], heap_values[0]
        heap_indices[0], heap_indices[i] = heap_indices[i], heap_indices[0]
        _heap_sift_down(heap_values, heap_indices, 0, i - 1)

    return heap_indices, heap_values


# ============================================================================
# PARALLEL MATRIX COMPUTATION
# ============================================================================


@jit(
    "float32[:,:](float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], float64, float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def gower_matrix_numba_parallel(
    X_cat,
    X_num,
    Y_cat,
    Y_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    ranges_of_numeric,
):
    """
    Parallel Numba kernel for computing full Gower distance matrix.
    Optimized for medium to large-scale datasets.
    """
    x_rows = X_cat.shape[0]
    y_rows = Y_cat.shape[0]
    result = np.zeros((x_rows, y_rows), dtype=np.float32)

    # Parallel over X rows
    for i in prange(x_rows):
        for j in range(y_rows):
            sum_cat = 0.0
            sum_num = 0.0
            has_nan = False

            # Categorical features
            for k in range(X_cat.shape[1]):
                xi_val = X_cat[i, k]
                yj_val = Y_cat[j, k]

                both_nan = np.isnan(xi_val) and np.isnan(yj_val)

                # If both NaN, distance is 0 (equal), so skip
                if both_nan:
                    continue

                # If one is NaN or values are different, add to categorical distance
                if np.isnan(xi_val) or np.isnan(yj_val) or xi_val != yj_val:
                    sum_cat += feature_weight_cat[k]

            # Numerical features
            for k in range(X_num.shape[1]):
                if ranges_of_numeric[k] != 0.0:
                    xi_val = X_num[i, k]
                    yj_val = Y_num[j, k]

                    if np.isnan(xi_val) and np.isnan(yj_val):
                        continue
                    elif np.isnan(xi_val) or np.isnan(yj_val):
                        has_nan = True
                        break

                    abs_delta = abs(xi_val - yj_val)
                    sij_num = abs_delta / ranges_of_numeric[k]
                    sum_num += feature_weight_num[k] * sij_num

            if has_nan:
                result[i, j] = np.nan
            else:
                result[i, j] = (sum_cat + sum_num) / feature_weight_sum

    return result
