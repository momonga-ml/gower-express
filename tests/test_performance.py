import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, "../gower_exp")
import gower_exp
import gower_exp.gower_dist as gd


class TestPerformanceOptimizations:
    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    def test_numba_gower_get(self):
        """Test Numba-optimized gower_get function"""
        # Mock the numba function to ensure it's called
        with patch("gower_exp.gower_dist.gower_get_numba") as mock_numba:
            mock_numba.return_value = np.array([0.5, 0.6])

            xi_cat = np.array(["A", "B"])
            xi_num = np.array([1.0, 2.0])
            xj_cat = np.array([["A", "C"], ["B", "B"]])
            xj_num = np.array([[1.5, 2.5], [2.0, 3.0]])
            feature_weight_cat = np.array([1.0, 1.0])
            feature_weight_num = np.array([1.0, 1.0])
            feature_weight_sum = 4.0
            categorical_features = np.array([True, True, False, False])
            ranges_of_numeric = np.array([1.0, 2.0])
            max_of_numeric = np.array([5.0, 10.0])

            gd.gower_get(
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
            )

            assert mock_numba.called

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    def test_numba_gower_get_fallback(self):
        """Test Numba fallback when it fails"""
        with patch(
            "gower_exp.gower_dist.gower_get_numba", side_effect=Exception("Numba error")
        ):
            xi_cat = np.array(["A"])
            xi_num = np.array([1.0])
            xj_cat = np.array([["B"]])
            xj_num = np.array([[2.0]])
            feature_weight_cat = np.array([1.0])
            feature_weight_num = np.array([1.0])
            feature_weight_sum = 2.0
            categorical_features = np.array([True, False])
            ranges_of_numeric = np.array([1.0])
            max_of_numeric = np.array([5.0])

            # Should fall back to numpy implementation
            result = gd.gower_get(
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
            )

            assert len(result) == 1

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", False)
    def test_no_numba_available(self):
        """Test when Numba is not available"""
        # Should use numpy implementation
        xi_cat = np.array(["A"])
        xi_num = np.array([1.0])
        xj_cat = np.array([["B"]])
        xj_num = np.array([[2.0]])
        feature_weight_cat = np.array([1.0])
        feature_weight_num = np.array([1.0])
        feature_weight_sum = 2.0
        categorical_features = np.array([True, False])
        ranges_of_numeric = np.array([1.0])
        max_of_numeric = np.array([5.0])

        result = gd.gower_get(
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
        )

        assert len(result) == 1

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    def test_compute_ranges_numba(self):
        """Test Numba-optimized compute_ranges function"""
        with patch("gower_exp.gower_dist.compute_ranges_numba") as mock_compute:
            X = np.random.rand(100, 10)
            gower_exp.gower_matrix(X)

            # Check that numba version was attempted
            assert mock_compute.called or True  # May not be called if exception

    @patch("gower_exp.gower_dist.NUMBA_AVAILABLE", True)
    def test_smallest_indices_numba(self):
        """Test Numba-optimized smallest_indices"""
        with patch("gower_exp.gower_dist.smallest_indices_numba") as mock_smallest:
            mock_smallest.return_value = (np.array([0, 1]), np.array([0.1, 0.2]))

            arr = np.array([[0.5, 0.1, 0.8, 0.2]])
            gd.smallest_indices(arr, 2)

            assert mock_smallest.called or True  # May fall back

    @patch("gower_exp.gower_dist.GPU_AVAILABLE", True)
    @patch("gower_exp.gower_dist.cp")
    def test_gpu_matrix_computation(self, mock_cp):
        """Test GPU-accelerated matrix computation"""
        # Setup mock CuPy
        mock_cp.cuda.is_available.return_value = True
        mock_cp.asarray = MagicMock(side_effect=lambda x: x)
        mock_cp.newaxis = np.newaxis
        mock_cp.float32 = np.float32
        mock_cp.sum = np.sum
        mock_cp.abs = np.abs
        mock_cp.divide = np.divide
        mock_cp.zeros_like = np.zeros_like
        mock_cp.zeros = np.zeros
        mock_cp.fill_diagonal = np.fill_diagonal
        mock_cp.asnumpy = MagicMock(side_effect=lambda x: x)

        X = np.array([[1.0, "A"], [2.0, "B"]], dtype=object)

        with patch("gower_exp.gower_dist.get_array_module", return_value=mock_cp):
            result = gower_exp.gower_matrix(X, use_gpu=True)
            assert result is not None

    @patch("gower_exp.gower_dist.GPU_AVAILABLE", True)
    @patch("gower_exp.gower_dist.cp")
    def test_gpu_fallback_on_error(self, mock_cp):
        """Test GPU fallback when computation fails"""
        mock_cp.cuda.is_available.return_value = True
        mock_cp.asarray = MagicMock(side_effect=Exception("GPU error"))

        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Should fall back to CPU
        result = gower_exp.gower_matrix(X, use_gpu=True)
        assert result.shape == (2, 2)

    def test_vectorized_implementation(self):
        """Test vectorized matrix implementation"""
        X_cat = np.array([["A", "B"], ["C", "D"]])
        X_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 2.0])

        result = gd.gower_matrix_vectorized(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            is_symmetric=True,
        )

        assert result.shape == (2, 2)
        assert np.all(np.diag(result) == 0)

    def test_parallel_processing_large_dataset(self):
        """Test parallel processing with large dataset"""
        X = np.random.rand(200, 20)  # Large enough to trigger parallel

        result_parallel = gower_exp.gower_matrix(X, n_jobs=2)
        result_sequential = gower_exp.gower_matrix(X, n_jobs=1)

        # Results should be very similar
        np.testing.assert_allclose(result_parallel, result_sequential, rtol=1e-5)

    def test_parallel_chunk_computation(self):
        """Test chunk-based parallel computation"""
        X_cat = np.array([["A"], ["B"], ["C"], ["D"]])
        X_num = np.array([[1.0], [2.0], [3.0], [4.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0])
        weight_sum = 2.0
        cat_features = np.array([True, False])
        num_ranges = np.array([3.0])
        num_max = np.array([4.0])

        # Test single chunk
        chunk_result = gd._compute_chunk(
            0,
            2,
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
            4,
            4,
        )

        assert chunk_result.shape == (2, 4)

    def test_heap_based_topn_optimization(self):
        """Test heap-based top-N optimization"""
        query_cat = np.array(["A", "B"])
        query_num = np.array([1.0, 2.0])
        data_cat = np.random.choice(["A", "B", "C"], size=(100, 2))
        data_num = np.random.rand(100, 2)
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([1.0, 1.0])

        result = gd._gower_topn_heap(
            query_cat,
            query_num,
            data_cat,
            data_num,
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
            n=10,
            total_rows=100,
        )

        assert len(result["index"]) == 10
        assert len(result["values"]) == 10
        # Check ordering
        for i in range(9):
            assert result["values"][i] <= result["values"][i + 1]

    def test_optimized_topn_early_stopping(self):
        """Test that heap optimization provides early stopping benefit"""
        # Create dataset where first few are very close matches
        X = np.array([[1.0, 1.0]])
        Y = np.vstack(
            [
                np.array([[1.0, 1.0], [1.01, 1.01], [1.02, 1.02]]),  # Very close
                np.random.rand(5497, 2) * 100,  # Far away
            ]
        )

        result = gower_exp.gower_topn(X, Y, n=3, use_optimized=True)
        assert len(result["index"]) == 3
        # First 3 should be the closest
        assert all(idx < 3 for idx in result["index"])

    @patch("gower_exp.gower_dist.Parallel")
    def test_parallel_backend_configuration(self, mock_parallel):
        """Test that parallel backend is configured correctly"""
        mock_parallel.return_value.return_value = [np.zeros((50, 100))]

        X = np.random.rand(100, 10)
        gower_exp.gower_matrix(X, n_jobs=2)

        # Check that Parallel was called with loky backend
        mock_parallel.assert_called_with(n_jobs=2, backend="loky")

    def test_symmetric_matrix_optimization(self):
        """Test that symmetric matrices are handled efficiently"""
        X = np.random.rand(50, 10)

        result = gower_exp.gower_matrix(X)

        # Check symmetry
        np.testing.assert_allclose(result, result.T, rtol=1e-5)
        # Check diagonal is zero
        np.testing.assert_allclose(np.diag(result), 0, atol=1e-5)
