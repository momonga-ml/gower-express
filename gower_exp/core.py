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

from .accelerators import (  # noqa: E402
    NUMBA_AVAILABLE,
    gower_get_numba_categorical_only,
    gower_get_numba_mixed_optimized,
    gower_get_numba_numerical_only,
)


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
    # Use optimized numba-based kernel selection when available
    if NUMBA_AVAILABLE and xi_cat.ndim == 1 and xj_cat.ndim == 2:
        try:
            # Auto-detect optimal kernel based on data characteristics
            has_categorical = len(xi_cat) > 0
            has_numerical = len(xi_num) > 0

            if has_categorical and has_numerical:
                # Mixed data - use optimized mixed kernel
                return gower_get_numba_mixed_optimized(
                    xi_cat,
                    xi_num,
                    xj_cat,
                    xj_num,
                    feature_weight_cat,
                    feature_weight_num,
                    feature_weight_sum,
                    ranges_of_numeric,
                )
            elif has_numerical and not has_categorical:
                # Pure numerical data - use specialized numerical kernel
                return gower_get_numba_numerical_only(
                    xi_num,
                    xj_num,
                    feature_weight_num,
                    ranges_of_numeric,
                )
            elif has_categorical and not has_numerical:
                # Pure categorical data - use specialized categorical kernel
                return gower_get_numba_categorical_only(
                    xi_cat,
                    xj_cat,
                    feature_weight_cat,
                )
            # If no features, fall through to numpy version

        except Exception as e:
            # Fall back to numpy version if numba fails
            logger.debug("Numba optimization failed, using numpy fallback: %s", str(e))

    # Original numpy implementation as fallback
    # categorical columns - optimized to reduce temporary arrays
    if len(xi_cat) > 0:
        # Pre-allocate output shape
        output_shape = xj_cat.shape[0] if xj_cat.ndim > 1 else 1
        sum_cat = np.zeros(output_shape, dtype=np.float32)

        # Process each feature to avoid large intermediate arrays
        for feat_idx in range(len(xi_cat)):
            xi_val = xi_cat[feat_idx]
            xj_vals = (
                xj_cat[:, feat_idx]
                if xj_cat.ndim > 1
                else xj_cat[feat_idx : feat_idx + 1]
            )

            # Handle categorical comparison including NaN values
            equal_mask = xi_val == xj_vals

            # Handle cases where both are NaN
            try:
                both_nan_mask = np.isnan(float(xi_val)) & np.isnan(
                    xj_vals.astype(float)
                )
                final_equal_mask = equal_mask | both_nan_mask
            except (ValueError, TypeError):
                # If can't convert to float, use direct equality
                final_equal_mask = equal_mask

            # Add weighted contribution directly to sum_cat (in-place)
            weight = feature_weight_cat[feat_idx]
            sum_cat += np.where(final_equal_mask, 0.0, weight)
    else:
        # Handle empty categorical arrays - return zeros with correct shape
        output_shape = xj_num.shape[0] if xj_num.ndim > 1 else 1
        sum_cat = np.zeros(output_shape, dtype=np.float32)

    # numerical columns - optimized to reduce temporary arrays
    if len(xi_num) > 0:
        output_shape = sum_cat.shape[0]
        sum_num = np.zeros(output_shape, dtype=np.float32)

        # Process numerical features with minimal temporary array creation
        for feat_idx in range(len(xi_num)):
            xi_val = xi_num[feat_idx]
            xj_vals = (
                xj_num[:, feat_idx]
                if xj_num.ndim > 1
                else xj_num[feat_idx : feat_idx + 1]
            )
            range_val = ranges_of_numeric[feat_idx]
            weight = feature_weight_num[feat_idx]

            if range_val != 0:
                # Compute absolute difference
                abs_delta = np.abs(xi_val - xj_vals)

                # Handle NaN values: when both values are NaN, distance should be 0
                both_nan_mask = np.isnan(xi_val) & np.isnan(xj_vals)
                abs_delta = np.where(both_nan_mask, 0.0, abs_delta)

                # Normalize and add weighted contribution directly to sum_num
                normalized_delta = abs_delta / range_val
                sum_num += weight * normalized_delta
    else:
        # Handle empty numerical arrays - return zeros with correct shape
        sum_num = np.zeros_like(sum_cat, dtype=np.float32)
    # Final computation - in-place addition and division to minimize allocations
    sum_cat += sum_num  # In-place addition
    sum_cat /= feature_weight_sum  # In-place division

    return sum_cat
