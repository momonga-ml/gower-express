"""Parallel processing utilities for Gower distance computation.

Implements chunked parallel processing using joblib for large-scale computations.
This module provides efficient parallel algorithms for computing Gower distance
matrices when dealing with large datasets that benefit from multi-core processing.
"""

import os

__all__ = [
    "_compute_gower_matrix_parallel",
    "_compute_chunk",
]

import numpy as np
from joblib import Parallel, delayed


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
    For symmetric matrices, uses optimization to compute only the upper triangle.
    """
    # Import here to avoid circular imports

    # Determine the actual number of jobs to use
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs < -1:
        n_jobs = max(1, os.cpu_count() + 1 + n_jobs)

    is_symmetric = x_n_rows == y_n_rows

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
            is_symmetric,
        )
        for start_idx, end_idx in row_chunks
    )

    # Pre-allocate output matrix with optimal memory layout
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32, order="C")

    # Aggregate results with memory-efficient copying
    for (start_idx, end_idx), chunk_result in zip(row_chunks, results):
        # Use direct assignment for memory efficiency
        chunk_size = end_idx - start_idx
        out[start_idx:end_idx, :] = chunk_result
        # Clear reference to chunk result to free memory earlier
        del chunk_result

    # Handle symmetric matrix case - fill lower triangle
    if is_symmetric:
        for i in range(x_n_rows):
            for j in range(i):
                out[i, j] = out[j, i]

    # For symmetric matrices, ensure diagonal is exactly 0 (unless all weights are zero)
    if is_symmetric and weight_sum > 0:
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
    is_symmetric,
):
    """
    Compute a chunk of the Gower distance matrix.

    This function processes rows from start_idx to end_idx and returns
    the corresponding chunk of the distance matrix.
    For symmetric matrices, only computes upper triangle elements.
    """
    # Import here to avoid circular imports
    from .core import gower_get

    chunk_size = end_idx - start_idx

    # Pre-allocate chunk output with optimal memory layout
    chunk_out = np.zeros((chunk_size, y_n_rows), dtype=np.float32, order="C")

    # Process rows with memory-conscious approach
    for i in range(chunk_size):
        row_idx = start_idx + i

        if is_symmetric:
            # Symmetric case: only compute upper triangle (including diagonal)
            j_start = row_idx
            if j_start < y_n_rows:
                # Create views instead of copies where possible
                Y_cat_slice = Y_cat[j_start:y_n_rows, :]
                Y_num_slice = Y_num[j_start:y_n_rows, :]

                res = gower_get(
                    X_cat[row_idx, :],
                    X_num[row_idx, :],
                    Y_cat_slice,
                    Y_num_slice,
                    weight_cat,
                    weight_num,
                    weight_sum,
                    cat_features,
                    num_ranges,
                    num_max,
                )
                chunk_out[i, j_start:] = res
                # Clear references to slices
                del Y_cat_slice, Y_num_slice, res
        else:
            # Non-symmetric case: compute full row
            res = gower_get(
                X_cat[row_idx, :],
                X_num[row_idx, :],
                Y_cat,
                Y_num,
                weight_cat,
                weight_num,
                weight_sum,
                cat_features,
                num_ranges,
                num_max,
            )
            chunk_out[i, :] = res
            # Clear reference to result
            del res

    return chunk_out
