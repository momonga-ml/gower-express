"""Core Gower distance computation functions.

Provides the main internal computation logic for mixed-type distance calculations.
This module contains the fundamental algorithms for computing Gower distances
between individual records, with optimizations and fallback mechanisms.
"""

import logging

import numpy as np

__all__ = [
    "gower_get",
]

logger = logging.getLogger(__name__)

from .accelerators import NUMBA_AVAILABLE, gower_get_numba  # noqa: E402


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
    """
    Core Gower distance computation function.

    Computes distances between a single query point and multiple target points.
    Uses numba-optimized version when available and compatible, otherwise falls
    back to numpy implementation.

    Parameters:
    -----------
    xi_cat : array-like
        Categorical features of query point
    xi_num : array-like
        Numerical features of query point
    xj_cat : array-like
        Categorical features of target points
    xj_num : array-like
        Numerical features of target points
    feature_weight_cat : array-like
        Weights for categorical features
    feature_weight_num : array-like
        Weights for numerical features
    feature_weight_sum : float
        Sum of all feature weights
    categorical_features : array-like
        Boolean mask indicating categorical features
    ranges_of_numeric : array-like
        Range values for numerical feature normalization
    max_of_numeric : array-like
        Maximum values for numerical features

    Returns:
    --------
    ndarray : Array of distances from query point to each target point
    """
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
        except Exception as e:
            # Fall back to numpy version if numba fails
            logger.debug("Numba optimization failed, using numpy fallback: %s", str(e))

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
    if len(xi_num) > 0:
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
    else:
        # Handle empty numerical arrays - return zeros with correct shape
        # Use sum_cat shape to match the output
        sum_num = np.zeros_like(sum_cat, dtype=np.float32)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij
