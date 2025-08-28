import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# Import the module to test internal functions
sys.path.insert(0, '../gower')
import gower.gower_dist as gd


class TestHelperFunctions:
    
    def test_get_array_module_cpu(self):
        """Test get_array_module returns numpy for CPU"""
        module = gd.get_array_module(use_gpu=False)
        assert module is np
    
    @patch('gower.gower_dist.GPU_AVAILABLE', True)
    @patch('gower.gower_dist.cp')
    def test_get_array_module_gpu(self, mock_cp):
        """Test get_array_module returns cupy for GPU when available"""
        module = gd.get_array_module(use_gpu=True)
        assert module is mock_cp
    
    @patch('gower.gower_dist.GPU_AVAILABLE', False)
    def test_get_array_module_gpu_not_available(self):
        """Test get_array_module falls back to numpy when GPU not available"""
        module = gd.get_array_module(use_gpu=True)
        assert module is np
    
    def test_gower_get_basic(self):
        """Test basic gower_get function"""
        xi_cat = np.array(['A', 'B'])
        xi_num = np.array([1.0, 2.0])
        xj_cat = np.array([['A', 'C'], ['B', 'B']])
        xj_num = np.array([[1.5, 2.5], [2.0, 3.0]])
        feature_weight_cat = np.array([1.0, 1.0])
        feature_weight_num = np.array([1.0, 1.0])
        feature_weight_sum = 4.0
        categorical_features = np.array([True, True, False, False])
        ranges_of_numeric = np.array([1.0, 2.0])
        max_of_numeric = np.array([5.0, 10.0])
        
        result = gd.gower_get(
            xi_cat, xi_num, xj_cat, xj_num,
            feature_weight_cat, feature_weight_num,
            feature_weight_sum, categorical_features,
            ranges_of_numeric, max_of_numeric
        )
        
        assert len(result) == 2
        assert all(r >= 0 for r in result)
    
    def test_gower_get_all_categorical(self):
        """Test gower_get with only categorical features"""
        xi_cat = np.array(['A', 'B', 'C'])
        xi_num = np.array([])
        xj_cat = np.array([['A', 'B', 'C'], ['D', 'B', 'F']])
        xj_num = np.array([[], []]).T  # Empty array with correct shape
        feature_weight_cat = np.array([1.0, 1.0, 1.0])
        feature_weight_num = np.array([])
        feature_weight_sum = 3.0
        categorical_features = np.array([True, True, True])
        ranges_of_numeric = np.array([])
        max_of_numeric = np.array([])
        
        result = gd.gower_get(
            xi_cat, xi_num, xj_cat, xj_num,
            feature_weight_cat, feature_weight_num,
            feature_weight_sum, categorical_features,
            ranges_of_numeric, max_of_numeric
        )
        
        assert len(result) == 2
    
    def test_gower_get_all_numerical(self):
        """Test gower_get with only numerical features"""
        xi_cat = np.array([])
        xi_num = np.array([1.0, 2.0, 3.0])
        xj_cat = np.array([[], []]).T  # Empty array with correct shape
        xj_num = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
        feature_weight_cat = np.array([])
        feature_weight_num = np.array([1.0, 1.0, 1.0])
        feature_weight_sum = 3.0
        categorical_features = np.array([False, False, False])
        ranges_of_numeric = np.array([2.0, 3.0, 4.0])
        max_of_numeric = np.array([10.0, 10.0, 10.0])
        
        result = gd.gower_get(
            xi_cat, xi_num, xj_cat, xj_num,
            feature_weight_cat, feature_weight_num,
            feature_weight_sum, categorical_features,
            ranges_of_numeric, max_of_numeric
        )
        
        assert len(result) == 2
    
    def test_gower_get_zero_range(self):
        """Test gower_get with zero range (division by zero handling)"""
        xi_cat = np.array([])
        xi_num = np.array([1.0, 2.0])
        xj_cat = np.array([[], []]).T
        xj_num = np.array([[1.5, 2.5], [2.0, 3.0]])
        feature_weight_cat = np.array([])
        feature_weight_num = np.array([1.0, 1.0])
        feature_weight_sum = 2.0
        categorical_features = np.array([False, False])
        ranges_of_numeric = np.array([0.0, 1.0])  # First range is zero
        max_of_numeric = np.array([5.0, 5.0])
        
        result = gd.gower_get(
            xi_cat, xi_num, xj_cat, xj_num,
            feature_weight_cat, feature_weight_num,
            feature_weight_sum, categorical_features,
            ranges_of_numeric, max_of_numeric
        )
        
        assert len(result) == 2
        assert not np.any(np.isnan(result))
    
    def test_smallest_indices_basic(self):
        """Test smallest_indices function"""
        arr = np.array([[5, 2, 8, 1, 9, 3]])
        result = gd.smallest_indices(arr, 3)
        
        assert 'index' in result
        assert 'values' in result
        assert len(result['index']) == 3
        assert result['values'][0] <= result['values'][1]
        assert result['values'][1] <= result['values'][2]
    
    def test_smallest_indices_with_nan(self):
        """Test smallest_indices with NaN values"""
        arr = np.array([[5, np.nan, 8, 1, 9, 3]])
        result = gd.smallest_indices(arr, 3)
        
        assert len(result['index']) == 3
        assert not np.any(np.isnan(result['values']))
    
    def test_smallest_indices_all_same(self):
        """Test smallest_indices when all values are the same"""
        arr = np.array([[5, 5, 5, 5]])
        result = gd.smallest_indices(arr, 2)
        
        assert len(result['index']) == 2
        assert result['values'][0] == result['values'][1]
    
    def test_compute_single_distance(self):
        """Test _compute_single_distance helper function"""
        query_cat = np.array(['A', 'B'])
        query_num = np.array([1.0, 2.0])
        row_cat = np.array(['A', 'C'])
        row_num = np.array([1.5, 2.5])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        num_ranges = np.array([2.0, 3.0])
        
        result = gd._compute_single_distance(
            query_cat, query_num, row_cat, row_num,
            weight_cat, weight_num, weight_sum, num_ranges
        )
        
        assert isinstance(result, (float, np.floating))
        assert result >= 0
    
    def test_compute_single_distance_empty_categorical(self):
        """Test _compute_single_distance with no categorical features"""
        query_cat = np.array([])
        query_num = np.array([1.0, 2.0])
        row_cat = np.array([])
        row_num = np.array([1.5, 2.5])
        weight_cat = np.array([])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 2.0
        num_ranges = np.array([2.0, 3.0])
        
        result = gd._compute_single_distance(
            query_cat, query_num, row_cat, row_num,
            weight_cat, weight_num, weight_sum, num_ranges
        )
        
        assert result >= 0
    
    def test_compute_single_distance_empty_numerical(self):
        """Test _compute_single_distance with no numerical features"""
        query_cat = np.array(['A', 'B'])
        query_num = np.array([])
        row_cat = np.array(['A', 'C'])
        row_num = np.array([])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([])
        weight_sum = 2.0
        num_ranges = np.array([])
        
        result = gd._compute_single_distance(
            query_cat, query_num, row_cat, row_num,
            weight_cat, weight_num, weight_sum, num_ranges
        )
        
        assert result >= 0
    
    def test_gower_topn_heap(self):
        """Test _gower_topn_heap function"""
        query_cat = np.array(['A'])
        query_num = np.array([1.0])
        data_cat = np.array([['A'], ['B'], ['A']])
        data_num = np.array([[1.5], [2.0], [1.1]])
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0])
        weight_sum = 2.0
        num_ranges = np.array([2.0])
        n = 2
        total_rows = 3
        
        result = gd._gower_topn_heap(
            query_cat, query_num, data_cat, data_num,
            weight_cat, weight_num, weight_sum,
            num_ranges, n, total_rows
        )
        
        assert 'index' in result
        assert 'values' in result
        assert len(result['index']) == 2
        assert result['values'][0] <= result['values'][1]
    
    def test_compute_chunk(self):
        """Test _compute_chunk function for parallel processing"""
        X_cat = np.array([['A', 'B'], ['C', 'D']])
        X_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y_cat = np.array([['A', 'B'], ['E', 'F'], ['G', 'H']])
        Y_num = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        weight_cat = np.array([1.0, 1.0])
        weight_num = np.array([1.0, 1.0])
        weight_sum = 4.0
        cat_features = np.array([True, True, False, False])
        num_ranges = np.array([2.0, 3.0])
        num_max = np.array([10.0, 10.0])
        x_n_rows = 2
        y_n_rows = 3
        
        result = gd._compute_chunk(
            0, 1, X_cat, X_num, Y_cat, Y_num,
            weight_cat, weight_num, weight_sum, cat_features,
            num_ranges, num_max, x_n_rows, y_n_rows
        )
        
        assert result.shape == (1, 3)
    
    def test_compute_gower_matrix_parallel(self):
        """Test _compute_gower_matrix_parallel function"""
        X_cat = np.array([['A'], ['B'], ['C'], ['D']])
        X_num = np.array([[1.0], [2.0], [3.0], [4.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0])
        weight_sum = 2.0
        cat_features = np.array([True, False])
        num_ranges = np.array([3.0])
        num_max = np.array([4.0])
        x_n_rows = 4
        y_n_rows = 4
        
        result = gd._compute_gower_matrix_parallel(
            X_cat, X_num, Y_cat, Y_num,
            weight_cat, weight_num, weight_sum,
            cat_features, num_ranges, num_max,
            x_n_rows, y_n_rows, n_jobs=2
        )
        
        assert result.shape == (4, 4)
        assert np.all(np.diag(result) == 0) or np.allclose(np.diag(result), 0)
    
    @patch('gower.gower_dist.os.cpu_count', return_value=4)
    def test_compute_gower_matrix_parallel_all_cores(self, mock_cpu_count):
        """Test parallel processing with n_jobs=-1"""
        X_cat = np.array([['A'], ['B']])
        X_num = np.array([[1.0], [2.0]])
        Y_cat = X_cat
        Y_num = X_num
        weight_cat = np.array([1.0])
        weight_num = np.array([1.0])
        weight_sum = 2.0
        cat_features = np.array([True, False])
        num_ranges = np.array([1.0])
        num_max = np.array([2.0])
        
        result = gd._compute_gower_matrix_parallel(
            X_cat, X_num, Y_cat, Y_num,
            weight_cat, weight_num, weight_sum,
            cat_features, num_ranges, num_max,
            2, 2, n_jobs=-1
        )
        
        assert result.shape == (2, 2)