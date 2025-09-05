"""Additional tests to boost coverage of gower_dist.py module."""

import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gower_exp import gower_matrix, gower_topn


class TestGowerMatrixEdgeCases:
    """Test edge cases and missing coverage in gower_matrix"""

    def test_data_x_none_raises_error(self):
        """Test that ValueError is raised when data_x is None"""
        with pytest.raises(ValueError, match="data_x cannot be None"):
            gower_matrix(None)

    def test_empty_array_with_data_y_none(self):
        """Test empty array handling when data_y is None"""
        empty_array = np.array([]).reshape(0, 3)
        result = gower_matrix(empty_array)
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_empty_array_with_data_y(self):
        """Test empty array with data_y having shape"""
        empty_x = np.array([]).reshape(0, 3)
        data_y = np.array([[1, 2, 3], [4, 5, 6]])
        result = gower_matrix(empty_x, data_y)
        assert result.shape == (0, 2)
        assert result.dtype == np.float32

    def test_empty_array_with_data_y_no_shape(self):
        """Test empty array with data_y without shape attribute"""
        empty_x = np.array([]).reshape(0, 3)
        data_y = [1, 2, 3]  # List doesn't have shape attribute
        result = gower_matrix(empty_x, data_y)
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_verbose_mode(self):
        """Test verbose mode prints timing information"""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = gower_matrix(data, verbose=True)
            output = captured_output.getvalue()

            # Check that verbose output is printed
            assert "Gower matrix computed" in output
            assert "seconds" in output
            assert result.shape == (3, 3)
        finally:
            sys.stdout = sys.__stdout__

    def test_verbose_mode_with_data_y(self):
        """Test verbose mode with data_y"""
        data_x = np.array([[1, 2, 3], [4, 5, 6]])
        data_y = np.array([[7, 8, 9], [10, 11, 12]])

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = gower_matrix(data_x, data_y, verbose=True)
            output = captured_output.getvalue()

            assert "Gower matrix computed" in output
            assert result.shape == (2, 2)
        finally:
            sys.stdout = sys.__stdout__

    def test_gpu_fallback_on_exception(self):
        """Test GPU fallback when exception occurs"""
        # Setup data
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with patch("gower_exp.gower_dist.GPU_AVAILABLE", True):
            with patch(
                "gower_exp.vectorized.gower_matrix_vectorized_gpu"
            ) as mock_gpu_func:
                # Mock GPU vectorized function to raise exception
                mock_gpu_func.side_effect = Exception("GPU computation failed")

                # Should fall back to CPU computation
                result = gower_matrix(data, use_gpu=True)

                assert result.shape == (3, 3)
                assert np.all(result >= 0)  # Valid distances
                # Verify it completed despite GPU failure

    def test_sequential_computation_small_dataset(self):
        """Test sequential computation path for small datasets"""
        # Create small dataset that triggers sequential path
        data_x = np.array([[1, 2], [3, 4]])
        data_y = np.array([[5, 6], [7, 8]])

        # Force sequential by setting n_jobs=1 and small data size
        # The function uses vectorized for small datasets, so we need to patch that
        with patch("gower_exp.gower_dist.gower_matrix_vectorized") as mock_vec:
            mock_vec.return_value = np.array([[0.5, 0.7], [0.3, 0.4]], dtype=np.float32)
            result = gower_matrix(data_x, data_y, n_jobs=1)

            # Verify result shape
            assert result.shape == (2, 2)

    def test_sequential_symmetric_computation(self):
        """Test sequential computation with symmetric case (X==Y)"""
        data = np.array([[1, 2], [3, 4], [5, 6]])

        # Create a small enough dataset and force sequential
        with patch("gower_exp.gower_dist.gower_get") as mock_gower_get:
            # Return different sizes for upper triangle computation
            mock_gower_get.side_effect = [
                np.array([0.0, 0.3, 0.5]),  # Row 0: distances to rows 0,1,2
                np.array([0.0, 0.4]),  # Row 1: distances to rows 1,2
                np.array([0.0]),  # Row 2: distance to row 2
            ]

            result = gower_matrix(data, n_jobs=1)

            assert result.shape == (3, 3)
            # Check diagonal is 0
            np.testing.assert_array_almost_equal(np.diag(result), [0.0, 0.0, 0.0])
            # Check symmetry
            assert result[0, 1] == result[1, 0]
            assert result[0, 2] == result[2, 0]
            assert result[1, 2] == result[2, 1]

    def test_weight_dimension_mismatch(self):
        """Test error when weight dimension doesn't match features"""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weight = np.array([0.5, 0.5])  # Wrong size

        with pytest.raises(
            ValueError, match="Weight dimension .* doesn't match feature dimension"
        ):
            gower_matrix(data, weight=weight)

    def test_negative_weights_error(self):
        """Test error when weights contain negative values"""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weight = np.array([0.5, -0.5, 0.5])  # Negative weight

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            gower_matrix(data, weight=weight)

    @patch("gower_exp.gower_dist.GPU_AVAILABLE", True)
    @patch("gower_exp.gower_dist.cp")
    def test_gpu_memory_cleanup(self, mock_cp):
        """Test that GPU memory is cleaned up after computation"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Setup mock GPU
        mock_memory_pool = MagicMock()
        mock_cp.get_default_memory_pool.return_value = mock_memory_pool
        mock_cp.asarray.return_value = data
        mock_cp.zeros.return_value = np.zeros((2, 2), dtype=np.float32)
        mock_cp.asnumpy.return_value = np.zeros((2, 2), dtype=np.float32)

        # Mock GPU vectorized computation
        with patch("gower_exp.vectorized.gower_matrix_vectorized_gpu") as mock_gpu_func:
            mock_gpu_func.return_value = np.zeros((2, 2), dtype=np.float32)

            result = gower_matrix(data, use_gpu=True)

            # Verify memory cleanup was called
            mock_memory_pool.free_all_blocks.assert_called()
            assert result.shape == (2, 2)


class TestGowerTopnEdgeCases:
    """Test edge cases in gower_topn function"""

    def test_single_row_requirement(self):
        """Test error when X has more than one row"""
        df_x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # 2 rows
        df_y = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        with pytest.raises(TypeError, match="Only support `data_x` of 1 row"):
            gower_topn(df_x, df_y)

    def test_shape_mismatch_error_arrays(self):
        """Test error when X and Y have different shapes (array case)"""
        x = np.array([[1, 2, 3]])
        y = np.array([[4, 5], [6, 7]])  # Different number of columns

        with pytest.raises(TypeError, match="X and Y must have same y-dim"):
            gower_topn(x, y)

    def test_categorical_features_as_array(self):
        """Test passing cat_features as an array"""
        data_x = np.array([[1, 2, 3]])
        data_y = np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
        cat_features = np.array([True, False, False])

        result = gower_topn(data_x, data_y, cat_features=cat_features, n=2)

        assert "index" in result
        assert "values" in result
        assert len(result["index"]) <= 2


class TestGowerDistHelpers:
    """Test helper functions and internal logic"""

    def test_verbose_timing_messages(self):
        """Test all verbose timing messages are printed"""
        data = np.random.randn(50, 10)

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Use enough data to trigger different paths
            _ = gower_matrix(data, verbose=True, n_jobs=2)
            output = captured_output.getvalue()

            # Check for timing messages
            assert "Gower matrix computed" in output
            assert "seconds" in output

        finally:
            sys.stdout = sys.__stdout__

    def test_is_number_safe_helper(self):
        """Test the is_number_safe helper function for DataFrames"""
        # Create a DataFrame with mixed types
        df = pd.DataFrame(
            {
                "num1": [1, 2, 3],
                "cat1": ["a", "b", "c"],
                "num2": [1.0, 2.0, 3.0],
                "cat2": pd.Categorical(["x", "y", "z"]),
            }
        )

        result = gower_matrix(df)
        assert result.shape == (3, 3)

    @patch("gower_exp.core.NUMBA_AVAILABLE", False)
    def test_fallback_when_numba_not_available(self):
        """Test computation falls back correctly when Numba is not available"""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Should still work without Numba
        result = gower_matrix(data)
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
