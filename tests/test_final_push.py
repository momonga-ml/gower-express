"""Final push to reach 90% coverage - focus on uncovered lines"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, '../gower')
import gower
import gower.gower_dist as gd


class TestFinalPush:
    """Target specific uncovered lines to reach 90%"""
    
    def test_import_paths_without_optional_deps(self):
        """Test import paths when optional dependencies are missing"""
        # Test the module constants
        assert hasattr(gd, 'NUMBA_AVAILABLE')
        assert hasattr(gd, 'GPU_AVAILABLE')
    
    def test_dummy_decorators(self):
        """Test dummy decorators when numba is not available"""
        # When NUMBA is not available, we have dummy decorators
        if not gd.NUMBA_AVAILABLE:
            # The dummy jit decorator
            @gd.jit(nopython=True)
            def test_func(x):
                return x * 2
            assert test_func(5) == 10
            
            # The dummy prange
            assert list(gd.prange(3)) == [0, 1, 2]
    
    def test_gpu_not_available_path(self):
        """Test when GPU/CuPy is not available"""
        with patch('gower.gower_dist.GPU_AVAILABLE', False):
            module = gd.get_array_module(use_gpu=True)
            assert module is np
    
    @patch('gower.gower_dist.GPU_AVAILABLE', True)
    @patch('gower.gower_dist.cp', create=True)
    def test_gpu_available_path(self, mock_cp):
        """Test when GPU is available"""
        module = gd.get_array_module(use_gpu=True)
        assert module is mock_cp
    
    def test_gower_matrix_with_non_numeric_dtypes(self):
        """Test automatic categorical detection with object dtypes"""
        X = np.array([
            [1, 'text', 3.0],
            [2, 'more', 4.0],
            [3, 'data', 5.0]
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_nan_range_computation(self):
        """Test range computation with NaN values"""
        X = np.array([
            [np.nan, 1.0],
            [2.0, np.nan],
            [np.nan, np.nan]
        ])
        
        # This should handle NaN without errors
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    @patch('gower.gower_dist.NUMBA_AVAILABLE', True)
    @patch('gower.gower_dist.compute_ranges_numba')
    def test_compute_ranges_numba_called(self, mock_compute):
        """Test that numba range computation is called"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        gower.gower_matrix(X)
        assert mock_compute.called
    
    @patch('gower.gower_dist.NUMBA_AVAILABLE', True)
    @patch('gower.gower_dist.compute_ranges_numba', side_effect=Exception)
    def test_compute_ranges_numba_exception(self, mock_compute):
        """Test fallback when numba range computation fails"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gower.gower_matrix(X)
        assert result.shape == (2, 2)
    
    def test_compute_ranges_nan_handling(self):
        """Test range computation handles NaN max/min"""
        with patch('gower.gower_dist.NUMBA_AVAILABLE', False):
            X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            result = gower.gower_matrix(X)
            assert result.shape == (2, 2)
    
    def test_parallel_n_jobs_negative(self):
        """Test parallel with negative n_jobs values"""
        X = np.random.rand(150, 10)
        
        # n_jobs = -2
        with patch('gower.gower_dist.os.cpu_count', return_value=4):
            result = gower.gower_matrix(X, n_jobs=-2)
            assert result.shape == (150, 150)
    
    def test_parallel_chunk_aggregation(self):
        """Test chunk result aggregation in parallel processing"""
        X = np.random.rand(120, 10)
        
        # Force parallel processing
        result = gower.gower_matrix(X, n_jobs=2)
        assert result.shape == (120, 120)
        
        # Check symmetry for square matrix
        np.testing.assert_allclose(result, result.T, rtol=1e-5)
    
    def test_gower_topn_small_dataset(self):
        """Test topn with dataset smaller than n"""
        X = np.array([[1.0, 2.0]])
        Y = np.array([[3.0, 4.0], [5.0, 6.0]])  # Only 2 rows
        
        result = gower.gower_topn(X, Y, n=5)  # Request 5 but only 2 available
        assert len(result['index']) == 2
    
    def test_gower_topn_optimized_conditions(self):
        """Test conditions for using optimized topn"""
        X = np.array([[1.0, 2.0]])
        
        # Small dataset - shouldn't optimize
        Y = np.random.rand(100, 2)
        result = gower.gower_topn(X, Y, n=5, use_optimized=True)
        assert len(result['index']) == 5
        
        # Large dataset, large n - shouldn't optimize
        Y = np.random.rand(6000, 2)
        result = gower.gower_topn(X, Y, n=60, use_optimized=True)
        assert len(result['index']) == 60
    
    def test_gower_topn_optimized_categorical_detection(self):
        """Test categorical detection in optimized topn"""
        X = pd.DataFrame({'num': [1.0], 'cat': ['A']})
        Y = pd.DataFrame({'num': [1.1, 1.2], 'cat': ['B', 'C']})
        
        result = gd.gower_topn_optimized(X, Y, n=2)
        assert len(result['index']) == 2
    
    def test_vectorized_implementations(self):
        """Test vectorized implementations directly"""
        # Test with mixed features
        X_cat = np.array([['A', 'B']])
        X_num = np.array([[1.0, 2.0]])
        Y_cat = np.array([['A', 'C'], ['D', 'B']])
        Y_num = np.array([[1.5, 2.5], [2.0, 3.0]])
        
        result = gd.gower_matrix_vectorized(
            X_cat, X_num, Y_cat, Y_num,
            np.array([1.0, 1.0]), np.array([1.0, 1.0]),
            4.0, np.array([1.0, 2.0]), False
        )
        assert result.shape == (1, 2)
    
    @patch('gower.gower_dist.GPU_AVAILABLE', True)
    @patch('gower.gower_dist.cp')
    def test_gpu_vectorized_implementation(self, mock_cp):
        """Test GPU vectorized implementation"""
        # Setup minimal mock
        mock_cp.newaxis = np.newaxis
        mock_cp.float32 = np.float32
        mock_cp.zeros = np.zeros
        mock_cp.sum = np.sum
        mock_cp.abs = np.abs
        mock_cp.divide = np.divide
        mock_cp.zeros_like = np.zeros_like
        mock_cp.fill_diagonal = np.fill_diagonal
        
        X_cat = np.array([['A', 'B']])
        X_num = np.array([[1.0, 2.0]])
        
        result = gd.gower_matrix_vectorized_gpu(
            X_cat, X_num, X_cat, X_num,
            np.array([1.0, 1.0]), np.array([1.0, 1.0]),
            4.0, np.array([1.0, 2.0]), True, mock_cp
        )
        assert result.shape == (1, 1)
    
    def test_heap_algorithm_initialization(self):
        """Test heap algorithm initialization phase"""
        # Test with small n
        query_cat = np.array(['A'])
        query_num = np.array([1.0])
        data_cat = np.array([['B'], ['C'], ['D']])
        data_num = np.array([[2.0], [3.0], [4.0]])
        
        result = gd._gower_topn_heap(
            query_cat, query_num, data_cat, data_num,
            np.array([1.0]), np.array([1.0]), 2.0,
            np.array([3.0]), n=2, total_rows=3
        )
        assert len(result['index']) == 2
    
    def test_smallest_indices_edge_cases(self):
        """Test smallest_indices with edge cases"""
        # Test with flat array
        arr = np.array([0.5, 0.1, 0.8, 0.2])
        flat_arr = arr.flatten()
        
        result = gd.smallest_indices(flat_arr.reshape(1, -1), 2)
        assert len(result['index']) == 2
    
    def test_compute_single_distance_comprehensive(self):
        """Test single distance computation comprehensively"""
        # Mixed features
        result = gd._compute_single_distance(
            np.array(['A']), np.array([1.0]),
            np.array(['B']), np.array([2.0]),
            np.array([1.0]), np.array([1.0]),
            2.0, np.array([1.0])
        )
        assert 0 <= result <= 1
        
        # Zero range handling
        result = gd._compute_single_distance(
            np.array([]), np.array([1.0]),
            np.array([]), np.array([2.0]),
            np.array([]), np.array([1.0]),
            1.0, np.array([0.0])  # Zero range
        )
        assert result >= 0