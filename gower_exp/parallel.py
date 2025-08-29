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
    """
    # Import here to avoid circular imports

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
    # Import here to avoid circular imports
    from .core import gower_get

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
