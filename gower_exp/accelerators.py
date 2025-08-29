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
    "compute_ranges_numba",
    "smallest_indices_numba",
]

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


@jit(
    "float32[:](float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64, float64[:])",
    nopython=True,
    parallel=True,
    cache=True,
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

            # If not both NaN and values are different, add to categorical distance
            if not both_nan and xi_val != xj_val:
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
)
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
