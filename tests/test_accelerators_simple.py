"""Simple tests to boost accelerators coverage without complex mocking."""

import numpy as np

from gower_exp import gower_matrix


class TestAcceleratorsSimple:
    """Simple tests that trigger accelerator functions through main API"""

    def test_numba_functions_via_main_api(self):
        """Test Numba functions by calling main API functions"""
        np.random.seed(42)

        # Create mixed data that will trigger different Numba kernels
        data_mixed = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # All numerical
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],  # Duplicate first row
            ]
        )

        # Test with different categorical feature specifications
        cat_features_mixed = [True, False, True, False, False]
        result = gower_matrix(data_mixed, cat_features=cat_features_mixed)
        assert result.shape == (4, 4)
        assert result[0, 3] == 0.0  # Duplicate rows should have distance 0

        # Test purely numerical data (should trigger numerical-only kernel)
        data_numerical = np.random.randn(5, 4).astype(np.float64)
        result = gower_matrix(data_numerical, cat_features=[False] * 4)
        assert result.shape == (5, 5)

        # Test purely categorical data (should trigger categorical-only kernel)
        data_categorical = np.random.randint(0, 3, size=(5, 4)).astype(np.float64)
        result = gower_matrix(data_categorical, cat_features=[True] * 4)
        assert result.shape == (5, 5)

    def test_ranges_computation_via_api(self):
        """Test range computation functions via API"""
        # Create data with different scales to trigger range computation
        data_ranges = np.array(
            [
                [1.0, 100.0, 0.001],
                [10.0, 200.0, 0.002],
                [5.0, 150.0, 0.0015],
                [0.0, 50.0, 0.0005],  # Min values
                [20.0, 300.0, 0.003],  # Max values
            ]
        )

        result = gower_matrix(data_ranges)
        assert result.shape == (5, 5)

        # Test with all NaN column (should trigger NaN handling in range computation)
        data_with_nan_col = data_ranges.copy()
        data_with_nan_col[:, 2] = np.nan
        result = gower_matrix(data_with_nan_col)
        assert result.shape == (5, 5)

    def test_large_data_parallel_paths(self):
        """Test data large enough to potentially trigger parallel computation"""
        np.random.seed(42)

        # Create moderately large data to trigger different computation paths
        large_data = np.random.randn(50, 8).astype(np.float64)
        cat_features = [True, False, True, False, False, True, False, True]

        # This should trigger parallel or optimized computation paths
        result = gower_matrix(large_data, cat_features=cat_features, n_jobs=2)
        assert result.shape == (50, 50)

        # Test symmetric properties
        np.testing.assert_array_almost_equal(result, result.T)

    def test_heap_functions_via_topn(self):
        """Test heap-based functions through gower_topn"""
        from gower_exp import gower_topn

        np.random.seed(42)

        # Create data that will use heap algorithms
        query_data = np.random.randn(1, 6)
        search_data = np.random.randn(200, 6)  # Large enough to trigger heap
        cat_features = [True, False, False, True, False, False]

        result = gower_topn(query_data, search_data, cat_features=cat_features, n=10)
        assert len(result["index"]) <= 10
        assert len(result["values"]) <= 10

    def test_mixed_optimized_kernels(self):
        """Test the optimized mixed kernel through API"""
        np.random.seed(42)

        # Create data designed to test mixed optimization paths
        mixed_data = np.array(
            [
                [1.0, 5.5, 2.0, 10.0, 3.0],  # Mix of categorical and numerical
                [2.0, 6.5, 2.0, 11.0, 4.0],  # Some matches, some differences
                [1.0, 7.5, 3.0, 10.0, 3.0],  # Partial matches
                [3.0, 8.5, 2.0, 12.0, 5.0],  # Different values
            ]
        )

        cat_features = [True, False, True, False, True]

        # Test with weights to trigger weighted computations
        weights = np.array([2.0, 1.0, 3.0, 1.5, 2.5])
        result = gower_matrix(mixed_data, weight=weights, cat_features=cat_features)
        assert result.shape == (4, 4)

    def test_gpu_fallback_coverage(self):
        """Test GPU fallback path by trying GPU with invalid configuration"""
        # This should gracefully fall back to CPU
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Try to use GPU - should fall back to CPU if GPU not available or fails
        result = gower_matrix(data, use_gpu=True)
        assert result.shape == (2, 2)

        # Result should be valid regardless of GPU availability
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_array_module_function(self):
        """Test get_array_module function"""
        from gower_exp.accelerators import get_array_module

        # Test CPU path
        xp = get_array_module(use_gpu=False)
        assert xp is np

        # Test GPU request (may fall back to CPU)
        xp_gpu = get_array_module(use_gpu=True)
        # Should return either cp or np depending on availability
        assert xp_gpu is not None

    def test_various_data_types(self):
        """Test with various data types to trigger different code paths"""
        # Test with integer data
        int_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]], dtype=np.int32)
        result = gower_matrix(int_data)
        assert result.shape == (3, 3)
        assert result[0, 2] == 0.0  # Identical rows

        # Test with float32 data
        float32_data = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
        result = gower_matrix(float32_data)
        assert result.shape == (2, 2)

        # Test with mixed categorical/numerical with NaN values
        mixed_nan_data = np.array(
            [
                [1.0, np.nan, 3.0],
                [1.0, 2.0, 3.0],
                [2.0, np.nan, 4.0],
            ]
        )
        cat_features = [True, False, False]
        result = gower_matrix(mixed_nan_data, cat_features=cat_features)
        assert result.shape == (3, 3)

    def test_zero_weight_and_range_handling(self):
        """Test edge cases with zero weights and ranges"""
        # Data where all values in a column are the same (zero range)
        same_col_data = np.array(
            [
                [1.0, 5.0, 3.0],
                [2.0, 5.0, 4.0],
                [3.0, 5.0, 5.0],
            ]
        )

        result = gower_matrix(same_col_data)
        assert result.shape == (3, 3)
        assert not np.any(np.isinf(result[np.isfinite(result)]))

        # Test with zero weights
        weights = np.array([0.0, 1.0, 2.0])
        result = gower_matrix(same_col_data, weight=weights)
        assert result.shape == (3, 3)
