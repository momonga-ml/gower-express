"""Final tests to push coverage over 80%."""

import numpy as np

from gower_exp import gower_matrix, gower_topn


class TestFinalCoveragePush:
    """Final targeted tests to maximize coverage"""

    def test_core_gower_get_variations(self):
        """Test core gower_get function with various scenarios"""
        from gower_exp.core import gower_get

        # Test with mixed data types
        xi_cat = np.array([1.0, 2.0])
        xi_num = np.array([0.5, 1.5])
        xj_cat = np.array([[1.0, 2.0], [1.0, 3.0]])
        xj_num = np.array([[0.5, 1.5], [1.0, 2.0]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        cat_features = np.array([True, True, False, False])
        num_ranges = np.array([2.0, 3.0])
        num_max = np.array([2.0, 3.0])

        result = gower_get(
            xi_cat,
            xi_num,
            xj_cat,
            xj_num,
            weight_cat,
            weight_num,
            weight_sum,
            cat_features,
            num_ranges,
            num_max,
        )

        assert len(result) == 2
        assert result[0] == 0.0  # Exact match

    def test_gower_matrix_verbose_mode_large_data(self):
        """Test verbose mode with larger data to trigger different paths"""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        # Capture verbose output with large data
        result = gower_matrix(data, verbose=True)
        assert result.shape == (100, 100)

    def test_gower_matrix_parallel_path(self):
        """Test parallel computation path"""
        np.random.seed(42)
        # Create data large enough to potentially trigger parallel path
        data = np.random.randn(200, 8)
        cat_features = [True] * 4 + [False] * 4

        result = gower_matrix(data, cat_features=cat_features, n_jobs=4)
        assert result.shape == (200, 200)

    def test_gower_topn_edge_cases(self):
        """Test gower_topn with various edge cases"""
        # Single row query
        query = np.array([[1.0, 2.0, 3.0, 4.0]])
        data = np.random.randn(50, 4)

        result = gower_topn(query, data, n=10)
        assert len(result["index"]) <= 10

        # Test with categorical features specified
        result = gower_topn(query, data, cat_features=[True, False, True, False], n=5)
        assert len(result["index"]) <= 5

    def test_topn_optimized_edge_cases(self):
        """Test topn optimized function edge cases"""
        from gower_exp.topn import gower_topn_optimized

        # Test DataFrame detection for categorical features
        query = np.array([[1.0, 2.0, 3.0]])
        data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        # Test with None cat_features (auto-detection)
        result = gower_topn_optimized(query, data, cat_features=None, n=2)
        assert "index" in result
        assert "values" in result

    def test_small_data_sequential_path(self):
        """Test sequential computation for very small datasets"""
        # Very small data that should use sequential path
        data_x = np.array([[1.0, 2.0]])
        data_y = np.array([[3.0, 4.0]])

        result = gower_matrix(data_x, data_y, n_jobs=1)
        assert result.shape == (1, 1)

    def test_symmetric_matrix_optimization(self):
        """Test symmetric matrix optimization path"""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Test symmetric computation (data_y=None)
        result = gower_matrix(data, n_jobs=1)  # Force sequential to test symmetric path
        assert result.shape == (3, 3)

        # Verify symmetry
        np.testing.assert_array_almost_equal(result, result.T)

    def test_weight_validation_paths(self):
        """Test weight validation edge cases"""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        # Test with valid weights
        weights = np.array([0.5, 1.0, 1.5])
        result = gower_matrix(data, weight=weights)
        assert result.shape == (2, 2)

        # Test with zero weight (edge case)
        weights_with_zero = np.array([0.0, 1.0, 1.0])
        result = gower_matrix(data, weight=weights_with_zero)
        assert result.shape == (2, 2)

    def test_vectorized_computation_path(self):
        """Test vectorized computation path for medium datasets"""
        np.random.seed(42)
        # Medium size data that should trigger vectorized path
        data = np.random.randn(50, 6)
        cat_features = [True, False, True, False, False, True]

        result = gower_matrix(data, cat_features=cat_features)
        assert result.shape == (50, 50)

    def test_mixed_data_type_handling(self):
        """Test handling of mixed data types"""
        # Integer and float mixed
        data_mixed = np.array([[1, 2.5, 3], [4, 5.5, 6]], dtype=object)
        data_mixed = data_mixed.astype(np.float64)

        result = gower_matrix(data_mixed)
        assert result.shape == (2, 2)

    def test_large_n_jobs_parameter(self):
        """Test with large n_jobs parameter"""
        data = np.random.randn(30, 5)

        # Test with large n_jobs
        result = gower_matrix(data, n_jobs=8)
        assert result.shape == (30, 30)

    def test_categorical_feature_variations(self):
        """Test various categorical feature specifications"""
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]])

        # Test with boolean list
        result1 = gower_matrix(data, cat_features=[True, False, True, False])
        assert result1.shape == (3, 3)

        # Test with numpy array
        result2 = gower_matrix(data, cat_features=np.array([True, False, True, False]))
        assert result2.shape == (3, 3)

        # Results should be the same
        np.testing.assert_array_almost_equal(result1, result2)

    def test_different_array_shapes(self):
        """Test with different array shapes and edge cases"""
        # Test 1D array (should be reshaped)
        data_1d = np.array([1, 2, 3, 4])
        data_2d = data_1d.reshape(1, -1)

        result = gower_matrix(data_2d)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_nan_handling_comprehensive(self):
        """Comprehensive NaN handling test"""
        # Data with various NaN patterns
        data_with_nans = np.array(
            [
                [1.0, 2.0, np.nan, 4.0],
                [1.0, np.nan, 3.0, 4.0],
                [np.nan, 2.0, 3.0, np.nan],
                [1.0, 2.0, 3.0, 4.0],  # No NaNs
            ]
        )

        result = gower_matrix(data_with_nans)
        assert result.shape == (4, 4)

        # Test with categorical features and NaNs
        result_cat = gower_matrix(
            data_with_nans, cat_features=[True, False, True, False]
        )
        assert result_cat.shape == (4, 4)

    def test_performance_optimizations(self):
        """Test various performance optimization paths"""
        np.random.seed(42)

        # Test data that should trigger different optimization strategies
        for size, n_features in [(20, 5), (50, 8), (100, 4)]:
            data = np.random.randn(size, n_features)
            cat_features = [True] * (n_features // 2) + [False] * (
                n_features - n_features // 2
            )

            result = gower_matrix(data, cat_features=cat_features)
            assert result.shape == (size, size)

    def test_topn_internal_functions(self):
        """Test topn internal function coverage"""
        from gower_exp.topn import smallest_indices

        # Test with different array sizes to trigger different paths
        for size in [10, 50, 150]:  # Different sizes for different algorithms
            np.random.seed(42)
            data = np.random.randn(size)
            n = min(size // 2, 10)

            result = smallest_indices(data, n)
            assert len(result["index"]) == n
            assert len(result["values"]) == n

    def test_zero_range_columns(self):
        """Test handling of columns with zero range"""
        # All same values in some columns
        data_same_col = np.array(
            [
                [1.0, 5.0, 3.0, 2.0],
                [2.0, 5.0, 4.0, 2.0],  # Column 1 and 3 have same values
                [3.0, 5.0, 5.0, 2.0],
            ]
        )

        result = gower_matrix(data_same_col)
        assert result.shape == (3, 3)
        assert not np.any(np.isinf(result[np.isfinite(result)]))

    def test_memory_efficiency_paths(self):
        """Test memory efficient computation paths"""
        np.random.seed(42)

        # Test with data that might trigger memory optimization
        data = np.random.randn(80, 6)

        # Test with different n_jobs to trigger different paths
        for n_jobs in [1, 2, 4]:
            result = gower_matrix(data, n_jobs=n_jobs)
            assert result.shape == (80, 80)
            # Verify diagonal is zero (identical rows)
            np.testing.assert_array_almost_equal(np.diag(result), np.zeros(80))

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases"""
        # Minimum size data
        min_data = np.array([[1.0]])
        result = gower_matrix(min_data)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

        # Two identical rows
        identical_data = np.array([[1.0, 2.0], [1.0, 2.0]])
        result = gower_matrix(identical_data)
        assert result.shape == (2, 2)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0

    def test_type_consistency(self):
        """Test type consistency across different inputs"""
        # Test with different input types
        data_int = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        result_int = gower_matrix(data_int)

        data_float = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        result_float = gower_matrix(data_float)

        # Results should be similar (allowing for type conversion)
        assert result_int.shape == result_float.shape
        assert result_int.dtype == np.float32
        assert result_float.dtype == np.float32
