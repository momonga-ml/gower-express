"""Additional tests to boost coverage of topn.py module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gower_exp.topn import (
    _compute_batch_distances_vectorized,
    _compute_single_distance,
    _gower_topn_heap,
    gower_topn_optimized,
    smallest_indices,
)


class TestTopnEdgeCases:
    """Test edge cases in topn module"""

    def test_gower_topn_optimized_dataframe_column_mismatch(self):
        """Test error when DataFrames have different columns"""
        df_x = pd.DataFrame({"a": [1], "b": [2]})
        df_y = pd.DataFrame({"a": [3, 4], "c": [5, 6]})  # Different columns

        with pytest.raises(TypeError, match="X and Y must have same columns"):
            gower_topn_optimized(df_x, df_y)

    def test_gower_topn_optimized_array_shape_mismatch(self):
        """Test error when arrays have different shapes"""
        x = np.array([[1, 2, 3]])
        y = np.array([[4, 5], [6, 7]])  # Different number of features

        with pytest.raises(TypeError, match="X and Y must have same y-dim"):
            gower_topn_optimized(x, y)

    def test_gower_topn_optimized_categorical_array(self):
        """Test when cat_features is provided as array"""
        x = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        cat_features = np.array([True, False, False])

        result = gower_topn_optimized(x, y, cat_features=cat_features, n=2)
        assert "index" in result
        assert "values" in result

    def test_smallest_indices_heap_path(self):
        """Test smallest_indices using heap algorithm for larger arrays"""
        # Create array large enough to trigger heap algorithm (>100 elements)
        np.random.seed(42)
        large_array = np.random.randn(150).reshape(-1)
        n = 10

        with patch("gower_exp.topn.NUMBA_AVAILABLE", True):
            with patch("gower_exp.topn.smallest_indices_numba_heap") as mock_heap:
                mock_heap.return_value = (np.arange(n), np.sort(large_array)[:n])

                result = smallest_indices(large_array, n)

                # Verify heap version was called for large array
                assert mock_heap.called
                assert len(result["index"]) == n
                assert len(result["values"]) == n

    def test_smallest_indices_non_heap_path(self):
        """Test smallest_indices using non-heap algorithm for small arrays"""
        small_array = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        n = 3

        with patch("gower_exp.topn.NUMBA_AVAILABLE", True):
            with patch("gower_exp.topn.smallest_indices_numba") as mock_numba:
                mock_indices = np.array([1, 3, 0])
                mock_values = np.array([1.0, 1.0, 3.0])
                mock_numba.return_value = (mock_indices, mock_values)

                result = smallest_indices(small_array, n)

                # Verify non-heap version was called for small array
                assert mock_numba.called
                assert len(result["index"]) == n

    def test_smallest_indices_numba_exception_fallback(self):
        """Test fallback when Numba fails"""
        array = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        n = 3

        with patch("gower_exp.topn.NUMBA_AVAILABLE", True):
            with patch("gower_exp.topn.smallest_indices_numba_heap") as mock_heap:
                mock_heap.side_effect = Exception("Numba compilation error")

                result = smallest_indices(array, n)

                # Should fall back to numpy implementation
                assert len(result["index"]) == n
                assert len(result["values"]) == n
                # Verify correct values despite exception
                assert result["values"][0] == 1.0
                assert result["values"][1] == 1.0

    def test_compute_single_distance(self):
        """Test _compute_single_distance function"""
        query_cat = np.array([1.0, 2.0])
        query_num = np.array([0.5, 1.5])
        row_cat = np.array([1.0, 3.0])  # One mismatch
        row_num = np.array([1.0, 2.0])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])

        distance = _compute_single_distance(
            query_cat,
            query_num,
            row_cat,
            row_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert isinstance(distance, float)
        assert 0 <= distance <= 1

    def test_compute_single_distance_empty_categorical(self):
        """Test _compute_single_distance with no categorical features"""
        query_cat = np.array([])
        query_num = np.array([0.5, 1.5])
        row_cat = np.array([])
        row_num = np.array([1.0, 2.0])
        weight_cat = np.array([])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 2.0
        num_ranges = np.array([2.0, 3.0])

        distance = _compute_single_distance(
            query_cat,
            query_num,
            row_cat,
            row_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert isinstance(distance, float)
        assert 0 <= distance <= 1

    def test_compute_single_distance_empty_numerical(self):
        """Test _compute_single_distance with no numerical features"""
        query_cat = np.array([1.0, 2.0])
        query_num = np.array([])
        row_cat = np.array([1.0, 3.0])
        row_num = np.array([])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([])
        weight_sum = 2.0
        num_ranges = np.array([])

        distance = _compute_single_distance(
            query_cat,
            query_num,
            row_cat,
            row_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert isinstance(distance, float)
        assert 0 <= distance <= 1

    def test_compute_single_distance_zero_ranges(self):
        """Test _compute_single_distance with zero ranges"""
        query_cat = np.array([1.0])
        query_num = np.array([0.5, 1.5])
        row_cat = np.array([1.0])
        row_num = np.array([1.0, 2.0])
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 3.0
        num_ranges = np.array([0.0, 3.0])  # First range is zero

        distance = _compute_single_distance(
            query_cat,
            query_num,
            row_cat,
            row_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert isinstance(distance, float)
        assert not np.isnan(distance)
        assert not np.isinf(distance)

    def test_compute_batch_distances_vectorized(self):
        """Test _compute_batch_distances_vectorized function"""
        query_cat = np.array([1.0, 2.0])
        query_num = np.array([0.5, 1.5])
        batch_data_cat = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 2.0]])
        batch_data_num = np.array([[0.5, 1.5], [1.0, 2.0], [0.0, 1.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])

        distances = _compute_batch_distances_vectorized(
            query_cat,
            query_num,
            batch_data_cat,
            batch_data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert len(distances) == 3
        assert distances[0] == 0.0  # Exact match
        assert 0 <= distances[1] <= 1
        assert 0 <= distances[2] <= 1

    def test_compute_batch_distances_vectorized_single_row(self):
        """Test _compute_batch_distances_vectorized with single row (1D arrays)"""
        query_cat = np.array([1.0, 2.0])
        query_num = np.array([0.5, 1.5])
        # Single row as 1D array
        batch_data_cat = np.array([1.0, 3.0])
        batch_data_num = np.array([1.0, 2.0])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])

        distances = _compute_batch_distances_vectorized(
            query_cat,
            query_num,
            batch_data_cat,
            batch_data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert len(distances) == 1
        assert 0 <= distances[0] <= 1

    def test_compute_batch_distances_vectorized_exception_fallback(self):
        """Test fallback when vectorized operations fail"""
        query_cat = np.array(["cat1", "cat2"])  # String categorical
        query_num = np.array([0.5, 1.5])
        batch_data_cat = np.array([["cat1", "cat3"], ["cat2", "cat2"]])
        batch_data_num = np.array([[0.5, 1.5], [1.0, 2.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])

        # This should trigger the exception handler for categorical comparison
        distances = _compute_batch_distances_vectorized(
            query_cat,
            query_num,
            batch_data_cat,
            batch_data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert len(distances) == 2
        # Should still compute valid distances via fallback

    def test_compute_batch_distances_numerical_exception_fallback(self):
        """Test fallback when numerical vectorized operations fail"""
        query_cat = np.array([1.0, 2.0])
        query_num = np.array([0.5, 1.5])
        batch_data_cat = np.array([[1.0, 2.0], [1.0, 3.0]])
        batch_data_num = np.array([[0.5, 1.5], [1.0, 2.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])

        # Mock abs to trigger numerical fallback
        original_abs = np.abs
        call_count = [0]

        def mock_abs(x):
            call_count[0] += 1
            if call_count[0] == 1 and isinstance(x, np.ndarray) and x.ndim == 2:
                raise Exception("Vectorization failed")
            return original_abs(x)

        with patch("numpy.abs", side_effect=mock_abs):
            distances = _compute_batch_distances_vectorized(
                query_cat,
                query_num,
                batch_data_cat,
                batch_data_num,
                weight_cat,
                weight_num,
                weight_sum,
                num_ranges,
            )

            assert len(distances) == 2

    def test_gower_topn_heap(self):
        """Test _gower_topn_heap function"""
        query_cat = np.array([1.0, 2.0])  # 1D array for single query
        query_num = np.array([0.5, 1.5])  # 1D array for single query
        data_cat = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.0]])
        data_num = np.array([[0.5, 1.5], [1.0, 2.0], [0.0, 1.0], [2.0, 3.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])
        n = 2
        total_rows = 4

        result = _gower_topn_heap(
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
        )

        # _gower_topn_heap returns a dict with index and values
        assert isinstance(result, dict)
        assert "index" in result
        assert "values" in result
        assert len(result["index"]) == n  # Should return n results
        assert len(result["values"]) == n

    def test_gower_topn_heap_batch_processing(self):
        """Test _gower_topn_heap with batch size larger than Y"""
        query_cat = np.array([1.0, 2.0])  # 1D array for single query
        query_num = np.array([0.5, 1.5])  # 1D array for single query
        # Small Y to ensure batch_size > n_samples_y
        data_cat = np.array([[1.0, 2.0], [1.0, 3.0]])
        data_num = np.array([[0.5, 1.5], [1.0, 2.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])
        n = 2
        total_rows = 2

        result = _gower_topn_heap(
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
        )

        # _gower_topn_heap returns a dict with index and values
        assert isinstance(result, dict)
        assert "index" in result
        assert "values" in result
        assert len(result["index"]) == n  # Should return n results
        assert len(result["values"]) == n
