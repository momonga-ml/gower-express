"""Additional tests to increase coverage and fix edge cases"""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, "../gower_exp")
import gower_exp
from gower_exp.parallel import _compute_chunk
from gower_exp.topn import _compute_single_distance, _gower_topn_heap, smallest_indices
from gower_exp.vectorized import gower_matrix_vectorized_gpu


class TestAdditionalCoverage:
    def test_gower_matrix_dataframe_with_dtypes(self):
        """Test with DataFrame to trigger dtype checking path"""
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0],
                "cat1": pd.Categorical(["A", "B", "A"]),
                "num2": [10, 20, 30],
            }
        )

        result = gower_exp.gower_matrix(df)
        assert result.shape == (3, 3)

    def test_vectorized_gpu_all_features(self):
        """Test GPU vectorized with both categorical and numerical"""
        X_cat = np.array([["A", "B"], ["C", "D"]])
        X_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 2.0])

        # Test with numpy as xp (GPU not available)
        result = gower_matrix_vectorized_gpu(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            is_symmetric=True,
            xp=np,
        )

        assert result.shape == (2, 2)
        assert np.all(np.diag(result) == 0)

    def test_vectorized_gpu_no_categorical(self):
        """Test GPU vectorized with no categorical features"""
        X_cat = np.zeros((2, 0))  # Empty but proper shape
        X_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 2.0
        num_ranges = np.array([2.0, 2.0])

        result = gower_matrix_vectorized_gpu(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            is_symmetric=True,
            xp=np,
        )

        assert result.shape == (2, 2)

    def test_vectorized_gpu_no_numerical(self):
        """Test GPU vectorized with no numerical features"""
        X_cat = np.array([["A", "B"], ["C", "D"]])
        X_num = np.zeros((2, 0))  # Empty but proper shape
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([])
        weight_sum = 2.0
        num_ranges = np.array([])

        result = gower_matrix_vectorized_gpu(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            is_symmetric=True,
            xp=np,
        )

        assert result.shape == (2, 2)

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    @patch("gower_exp.gower_dist.compute_ranges_numba")
    def test_compute_ranges_numba_used(self, mock_compute):
        """Test that compute_ranges_numba is called when available"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        gower_exp.gower_matrix(X)
        assert mock_compute.called

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    @patch("gower_exp.gower_dist.compute_ranges_numba", side_effect=Exception("Error"))
    def test_compute_ranges_numba_fallback(self, mock_compute):
        """Test fallback when compute_ranges_numba fails"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gower_exp.gower_matrix(X)
        assert result.shape == (2, 2)

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", False)
    def test_compute_ranges_no_numba(self):
        """Test range computation without Numba"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gower_exp.gower_matrix(X)
        assert result.shape == (2, 2)

    def test_compute_ranges_all_nan(self):
        """Test range computation with all NaN values"""
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        result = gower_exp.gower_matrix(X)
        assert result.shape == (2, 2)

    @patch("gower_exp.accelerators.NUMBA_AVAILABLE", True)
    @patch("gower_exp.topn.smallest_indices_numba")
    def test_smallest_indices_numba_used(self, mock_smallest):
        """Test that smallest_indices_numba is called when available"""
        mock_smallest.return_value = (np.array([0, 1]), np.array([0.1, 0.2]))

        arr = np.array([[0.5, 0.1, 0.8]])
        smallest_indices(arr, 2)
        assert mock_smallest.called

    @patch("gower_exp.accelerators.NUMBA_AVAILABLE", True)
    @patch("gower_exp.topn.smallest_indices_numba", side_effect=Exception("Error"))
    def test_smallest_indices_numba_fallback(self, mock_smallest):
        """Test fallback when smallest_indices_numba fails"""
        arr = np.array([[0.5, 0.1, 0.8]])
        result = smallest_indices(arr, 2)
        assert len(result["index"]) == 2

    @patch("gower_exp.accelerators.NUMBA_AVAILABLE", False)
    def test_smallest_indices_no_numba(self):
        """Test smallest_indices without Numba"""
        arr = np.array([[0.5, 0.1, 0.8]])
        result = smallest_indices(arr, 2)
        assert len(result["index"]) == 2

    def test_gower_topn_optimized_small_n(self):
        """Test optimized top-N with small n value"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(5500, 2)  # Large dataset

        result = gower_exp.gower_topn(X, Y, n=5, use_optimized=True)
        assert len(result["index"]) == 5

    def test_gower_topn_optimized_large_n(self):
        """Test optimized top-N with large n value (shouldn't optimize)"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(5500, 2)  # Large dataset

        result = gower_exp.gower_topn(X, Y, n=60, use_optimized=True)  # n > 50
        assert len(result["index"]) == 60

    def test_parallel_negative_n_jobs(self):
        """Test parallel with n_jobs < -1"""
        X = np.random.rand(150, 10)

        with patch("gower_exp.parallel.os.cpu_count", return_value=8):
            result = gower_exp.gower_matrix(X, n_jobs=-3)  # Should use 6 cores
            assert result.shape == (150, 150)

    def test_gower_topn_heap_early_stopping(self):
        """Test heap algorithm with early stopping"""
        query_cat = np.array([])
        query_num = np.array([1.0, 1.0])

        # Create data where first items are closest
        data_cat = np.zeros((100, 0))
        data_num = np.vstack(
            [
                np.array([[1.0, 1.0], [1.01, 1.01]]),  # Very close
                np.random.rand(98, 2) * 100,  # Far away
            ]
        )

        weight_cat = np.array([])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 2.0
        num_ranges = np.array([100.0, 100.0])

        result = _gower_topn_heap(
            query_cat,
            query_num,
            data_cat,
            data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            n=2,
            total_rows=100,
        )

        assert len(result["index"]) == 2
        assert all(idx < 2 for idx in result["index"])

    def test_gpu_exception_handling(self):
        """Test GPU exception handling during computation"""
        X = np.array([[1.0, "A"], [2.0, "B"]], dtype=object)

        with patch("gower_exp.accelerators.GPU_AVAILABLE", True):
            with patch("gower_exp.accelerators.cp") as mock_cp:
                mock_cp.cuda.is_available.return_value = True
                mock_cp.asarray.side_effect = Exception("GPU Memory Error")

                # Should fall back to CPU
                result = gower_exp.gower_matrix(X, use_gpu=True)
                assert result.shape == (2, 2)

    def test_parallel_symmetric_handling(self):
        """Test symmetric matrix handling in parallel processing"""
        X = np.random.rand(100, 10)

        result = gower_exp.gower_matrix(X, n_jobs=2)

        # Check symmetry
        np.testing.assert_allclose(result, result.T, rtol=1e-5)

    def test_chunk_symmetric_case(self):
        """Test chunk computation for symmetric case"""
        X_cat = np.array([["A"], ["B"]])
        X_num = np.array([[1.0], [2.0]])
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0])
        weight_sum = 2.0
        cat_features = np.array([True, False])
        num_ranges = np.array([1.0])
        num_max = np.array([2.0])

        # Test symmetric case (X == Y)
        result = _compute_chunk(
            0,
            1,
            X_cat,
            X_num,
            X_cat,
            X_num,
            weight_cat,
            weight_num,
            weight_sum,
            cat_features,
            num_ranges,
            num_max,
            2,
            2,
        )

        assert result.shape == (1, 2)

    def test_compute_single_distance_zero_ranges(self):
        """Test single distance computation with zero ranges"""
        query_cat = np.array([])
        query_num = np.array([1.0, 2.0])
        row_cat = np.array([])
        row_num = np.array([1.5, 2.5])
        weight_cat = np.array([])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 2.0
        num_ranges = np.array([0.0, 1.0])  # First range is zero

        result = _compute_single_distance(
            query_cat,
            query_num,
            row_cat,
            row_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )

        assert not np.isnan(result)
        assert result >= 0
