from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import gower


class TestGowerMatrix:
    def test_basic_gower_matrix_numpy(self):
        """Test basic Gower distance calculation with numpy arrays"""
        X = np.array([[1.0, "A", 10], [2.0, "B", 20], [3.0, "A", 30]], dtype=object)

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert np.all(np.diag(result) == 0)  # Diagonal should be 0
        assert result[0, 1] > 0  # Different values should have distance > 0

    def test_gower_matrix_pandas(self):
        """Test Gower distance with pandas DataFrame"""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "categorical": ["A", "B", "A"],
                "mixed": [10, 20, 30],
            }
        )

        result = gower.gower_matrix(df)
        assert result.shape == (3, 3)
        assert np.all(np.diag(result) == 0)

    def test_gower_matrix_with_separate_datasets(self):
        """Test Gower distance between two different datasets"""
        X = np.array([[1.0, "A"], [2.0, "B"]], dtype=object)
        Y = np.array([[3.0, "C"], [4.0, "D"], [5.0, "E"]], dtype=object)

        result = gower.gower_matrix(X, Y)
        assert result.shape == (2, 3)  # X rows x Y rows

    def test_gower_matrix_with_weights(self):
        """Test Gower distance with custom feature weights"""
        X = np.array([[1.0, "A", 10], [2.0, "B", 20], [3.0, "A", 30]], dtype=object)

        weights = np.array([2.0, 1.0, 0.5])
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (3, 3)

    def test_gower_matrix_with_categorical_features(self):
        """Test with explicit categorical features specification"""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        cat_features = [False, True, False]  # Middle column is categorical
        result = gower.gower_matrix(X, cat_features=cat_features)
        assert result.shape == (3, 3)

    def test_gower_matrix_all_categorical(self):
        """Test with all categorical features"""
        X = np.array([["A", "X", "P"], ["B", "Y", "Q"], ["A", "X", "R"]], dtype=object)

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert (
            result[0, 2] < result[0, 1]
        )  # More similar values should have smaller distance

    def test_gower_matrix_all_numerical(self):
        """Test with all numerical features"""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)

    def test_gower_matrix_with_missing_values(self):
        """Test handling of missing values (NaN)"""
        X = np.array(
            [[1.0, "A", 10], [np.nan, "B", 20], [3.0, None, np.nan]], dtype=object
        )

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.any(np.isnan(np.diag(result)))  # Diagonal shouldn't have NaN

    def test_gower_matrix_parallel_processing(self):
        """Test parallel processing with n_jobs parameter"""
        X = np.random.rand(150, 10)  # Large enough to trigger parallel

        # Test with different n_jobs values
        result_sequential = gower.gower_matrix(X, n_jobs=1)
        result_parallel = gower.gower_matrix(X, n_jobs=2)
        result_all_cores = gower.gower_matrix(X, n_jobs=-1)

        # Results should be similar regardless of parallelization
        np.testing.assert_allclose(result_sequential, result_parallel, rtol=1e-5)
        np.testing.assert_allclose(result_sequential, result_all_cores, rtol=1e-5)

    def test_gower_matrix_small_dataset_no_parallel(self):
        """Test that small datasets don't use parallel processing"""
        X = np.random.rand(50, 5)  # Small dataset

        result = gower.gower_matrix(X, n_jobs=-1)
        assert result.shape == (50, 50)

    def test_gower_matrix_dimension_mismatch(self):
        """Test error handling for dimension mismatch"""
        X = np.array([[1, 2, 3]])
        Y = np.array([[4, 5]])  # Different number of columns

        with pytest.raises(TypeError, match="must have same"):
            gower.gower_matrix(X, Y)

    def test_gower_matrix_pandas_column_mismatch(self):
        """Test error handling for pandas column mismatch"""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"c": [3], "d": [4]})

        with pytest.raises(TypeError, match="must have same columns"):
            gower.gower_matrix(df1, df2)

    def test_gower_matrix_sparse_not_supported(self):
        """Test that sparse matrices raise an error"""
        from scipy.sparse import csr_matrix

        X = csr_matrix([[1, 2], [3, 4]])

        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            gower.gower_matrix(X)

    @patch("gower.gower_dist.GPU_AVAILABLE", True)
    @patch("gower.gower_dist.cp")
    def test_gower_matrix_gpu_acceleration(self, mock_cp):
        """Test GPU acceleration path"""
        mock_cp.cuda.is_available.return_value = True
        mock_cp.asarray = lambda x: x
        mock_cp.asnumpy = lambda x: x

        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Mock the GPU function to return a simple result
        with patch("gower.gower_dist.gower_matrix_vectorized_gpu") as mock_gpu_func:
            mock_gpu_func.return_value = np.zeros((2, 2))
            gower.gower_matrix(X, use_gpu=True)
            assert mock_gpu_func.called

    @patch("gower.gower_dist.GPU_AVAILABLE", False)
    def test_gower_matrix_gpu_fallback(self):
        """Test fallback when GPU is not available"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = gower.gower_matrix(X, use_gpu=True)
        assert result.shape == (2, 2)  # Should still work with CPU

    def test_gower_matrix_negative_values(self):
        """Test handling of negative values in data"""
        X = np.array([[-1.0, 2.0], [3.0, -4.0], [-5.0, -6.0]])

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert np.all(result >= 0)  # Distances should be non-negative

    def test_gower_matrix_zero_range(self):
        """Test handling when all values in a column are the same"""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.any(np.isnan(result))

    def test_gower_matrix_single_row(self):
        """Test with single row input"""
        X = np.array([[1.0, 2.0, 3.0]])

        result = gower.gower_matrix(X)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_gower_matrix_empty_weights(self):
        """Test with None weights (should use equal weights)"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = gower.gower_matrix(X, weight=None)
        assert result.shape == (2, 2)
