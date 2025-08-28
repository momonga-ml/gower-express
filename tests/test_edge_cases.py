import pytest
import numpy as np
import pandas as pd
import gower
from scipy.sparse import csr_matrix


class TestEdgeCases:
    
    def test_empty_array(self):
        """Test with empty arrays"""
        X = np.array([]).reshape(0, 0)
        with pytest.raises((IndexError, ValueError)):
            gower.gower_matrix(X)
    
    def test_single_element_array(self):
        """Test with single element"""
        X = np.array([[5.0]])
        result = gower.gower_matrix(X)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0
    
    def test_all_nan_values(self):
        """Test with all NaN values"""
        X = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        result = gower.gower_matrix(X)
        assert result.shape == (2, 2)
    
    def test_mixed_nan_values(self):
        """Test with mixed NaN and valid values"""
        X = np.array([
            [1.0, np.nan, 'A'],
            [np.nan, 2.0, 'B'],
            [3.0, 4.0, None]
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_infinite_values(self):
        """Test with infinite values"""
        X = np.array([
            [1.0, 2.0],
            [np.inf, 3.0],
            [-np.inf, 4.0]
        ])
        
        # Should handle infinities without crashing
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_very_large_values(self):
        """Test with very large values"""
        X = np.array([
            [1e100, 1e-100],
            [1e99, 1e-99],
            [1e98, 1e-98]
        ])
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.any(np.isnan(result))
    
    def test_zero_variance_columns(self):
        """Test with columns that have zero variance"""
        X = np.array([
            [1.0, 5.0, 'A'],
            [2.0, 5.0, 'A'],
            [3.0, 5.0, 'A']
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.any(np.isnan(result))
    
    def test_single_unique_categorical(self):
        """Test with categorical column having single unique value"""
        X = np.array([
            [1.0, 'Same'],
            [2.0, 'Same'],
            [3.0, 'Same']
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_extreme_weights(self):
        """Test with extreme weight values"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Very large weights
        weights = np.array([1e10, 1e-10])
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (2, 2)
        
        # Zero weights
        weights = np.array([0.0, 1.0])
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (2, 2)
    
    def test_mismatched_data_types(self):
        """Test with mismatched data types between X and Y"""
        X = pd.DataFrame({'a': [1, 2], 'b': ['A', 'B']})
        Y = np.array([[3, 'C'], [4, 'D']], dtype=object)
        
        result = gower.gower_matrix(X, Y)
        assert result.shape == (2, 2)
    
    def test_unicode_categorical(self):
        """Test with Unicode categorical values"""
        X = np.array([
            [1.0, '中文'],
            [2.0, 'émoji'],
            [3.0, '🎉']
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_boolean_categorical(self):
        """Test with boolean values as categorical"""
        X = np.array([
            [1.0, True],
            [2.0, False],
            [3.0, True]
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_mixed_numeric_types(self):
        """Test with mixed integer and float types"""
        X = np.array([
            [1, 2.5],
            [3, 4.7],
            [5, 6.9]
        ])
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_dataframe_different_dtypes(self):
        """Test DataFrame with various dtypes"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        
        result = gower.gower_matrix(df)
        assert result.shape == (3, 3)
    
    def test_sparse_matrix_error(self):
        """Test that sparse matrices raise appropriate error"""
        X = csr_matrix([[1, 2], [3, 4]])
        
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            gower.gower_matrix(X)
    
    def test_y_sparse_matrix_error(self):
        """Test that sparse Y matrix raises error"""
        X = np.array([[1, 2]])
        Y = csr_matrix([[3, 4]])
        
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            gower.gower_matrix(X, Y)
    
    def test_dimension_mismatch_numpy(self):
        """Test dimension mismatch with numpy arrays"""
        X = np.array([[1, 2, 3]])
        Y = np.array([[4, 5]])
        
        with pytest.raises(TypeError, match="same y-dim"):
            gower.gower_matrix(X, Y)
    
    def test_dimension_mismatch_pandas(self):
        """Test column mismatch with pandas DataFrames"""
        X = pd.DataFrame({'a': [1], 'b': [2]})
        Y = pd.DataFrame({'c': [3], 'd': [4]})
        
        with pytest.raises(TypeError, match="same columns"):
            gower.gower_matrix(X, Y)
    
    def test_topn_multiple_rows_error(self):
        """Test topn with multiple rows raises error"""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6]])
        
        with pytest.raises(TypeError, match="Only support.*1 row"):
            gower.gower_topn(X, Y)
    
    def test_topn_empty_y(self):
        """Test topn with empty Y dataset"""
        X = np.array([[1, 2]])
        Y = np.array([]).reshape(0, 2)
        
        result = gower.gower_topn(X, Y, n=5)
        assert len(result['index']) == 0
    
    def test_negative_n_jobs(self):
        """Test with negative n_jobs values"""
        X = np.random.rand(10, 5)
        
        # n_jobs = -2 should use all cores minus 1
        result = gower.gower_matrix(X, n_jobs=-2)
        assert result.shape == (10, 10)
    
    def test_categorical_detection_edge_cases(self):
        """Test automatic categorical detection edge cases"""
        # Mixed object types
        X = np.array([
            [1, '2', 3.0],
            [4, '5', 6.0],
            [7, '8', 9.0]
        ], dtype=object)
        
        result = gower.gower_matrix(X)
        assert result.shape == (3, 3)
    
    def test_weight_length_mismatch(self):
        """Test behavior with mismatched weight length"""
        X = np.array([[1, 2], [3, 4]])
        
        # This should work - extra weights are ignored or missing weights default to 1
        weights = np.array([0.5])  # Only one weight for two features
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (2, 2)
    
    def test_cat_features_length_mismatch(self):
        """Test with mismatched cat_features length"""
        X = np.array([[1, 2], [3, 4]])
        
        # Too few categorical features specified
        cat_features = [True]  # Only one for two features
        with pytest.raises((IndexError, ValueError)):
            gower.gower_matrix(X, cat_features=cat_features)
    
    def test_all_zero_weights(self):
        """Test with all zero weights"""
        X = np.array([[1, 2], [3, 4]])
        weights = np.array([0.0, 0.0])
        
        # Should handle division by zero
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (2, 2)
    
    def test_negative_weights(self):
        """Test with negative weights"""
        X = np.array([[1, 2], [3, 4]])
        weights = np.array([-1.0, 1.0])
        
        # Should still work, though results may be unexpected
        result = gower.gower_matrix(X, weight=weights)
        assert result.shape == (2, 2)
    
    def test_topn_n_zero(self):
        """Test topn with n=0"""
        X = np.array([[1, 2]])
        Y = np.array([[3, 4], [5, 6]])
        
        result = gower.gower_topn(X, Y, n=0)
        assert len(result['index']) == 0
    
    def test_topn_negative_n(self):
        """Test topn with negative n"""
        X = np.array([[1, 2]])
        Y = np.array([[3, 4], [5, 6]])
        
        # Should either raise error or return empty
        result = gower.gower_topn(X, Y, n=-1)
        assert len(result['index']) == 0 or len(result['index']) == 2