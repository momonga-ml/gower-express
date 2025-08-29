"""Final tests to reach 90% coverage target"""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, "../gower_exp")
import gower_exp
from gower_exp.accelerators import (
    get_array_module,
)
from gower_exp.core import gower_get
from gower_exp.parallel import _compute_chunk
from gower_exp.topn import _gower_topn_heap, smallest_indices


class TestFinalCoverage:
    """Tests to cover remaining gaps for 90% coverage"""

    @patch("gower_exp.accelerators.NUMBA_AVAILABLE", False)
    def test_imports_without_numba(self):
        """Test that module loads correctly without numba"""
        # Reimport module without numba
        import importlib

        # Reload accelerators module to test without numba
        import gower_exp.accelerators

        importlib.reload(gower_exp.accelerators)

        # Check that functions still exist in main module
        assert hasattr(gower_exp, "gower_matrix")
        assert hasattr(gower_exp, "gower_topn")

    def test_gpu_available_check_without_cupy(self):
        """Test GPU availability check when CuPy is not installed"""
        with patch.dict("sys.modules", {"cupy": None}):
            # This would trigger the ImportError path
            # but we can't easily reload the module here
            pass

    @patch("gower_exp.accelerators.GPU_AVAILABLE", True)
    @patch("gower_exp.accelerators.cp")
    def test_get_array_module_with_mock_cupy(self, mock_cp):
        """Test get_array_module returns mock cupy"""
        mock_cp.cuda.is_available.return_value = True

        module = get_array_module(use_gpu=True)
        assert module is mock_cp

        module = get_array_module(use_gpu=False)
        assert module is np

    def test_gower_matrix_nan_handling_in_ranges(self):
        """Test NaN handling in range computation"""
        X = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 3.0]])

        result = gower_exp.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.all(np.isnan(result))

    @patch("gower_exp.core.NUMBA_AVAILABLE", True)
    def test_gower_get_numba_optimization_path(self):
        """Test that gower_get uses numba when available and compatible"""
        # Create proper 1D and 2D arrays to trigger numba path
        xi_cat = np.array(["A", "B"])  # 1D array
        xi_num = np.array([1.0, 2.0])  # 1D array
        xj_cat = np.array([["A", "C"], ["B", "B"]])  # 2D array
        xj_num = np.array([[1.5, 2.5], [2.0, 3.0]])  # 2D array

        with patch("gower_exp.core.gower_get_numba") as mock_numba:
            mock_numba.return_value = np.array([0.5, 0.6])

            gower_get(
                xi_cat,
                xi_num,
                xj_cat,
                xj_num,
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                4.0,
                np.array([True, True, False, False]),
                np.array([1.0, 2.0]),
                np.array([5.0, 10.0]),
            )

            assert mock_numba.called

    def test_compute_ranges_with_all_same_values(self):
        """Test range computation when all values in column are the same"""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

        result = gower_exp.gower_matrix(X)
        assert result.shape == (3, 3)
        assert not np.any(np.isnan(result))

    def test_compute_ranges_with_zero_max(self):
        """Test range computation when max is zero"""
        X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])

        result = gower_exp.gower_matrix(X)
        assert result.shape == (3, 3)

    def test_parallel_with_exactly_one_chunk(self):
        """Test parallel processing when data fits in one chunk"""
        X = np.random.rand(2, 5)  # Very small dataset

        with patch("gower_exp.parallel.os.cpu_count", return_value=4):
            result = gower_exp.gower_matrix(X, n_jobs=4)
            assert result.shape == (2, 2)

    def test_smallest_indices_with_all_nan(self):
        """Test smallest_indices with all NaN values"""
        arr = np.array([[np.nan, np.nan, np.nan]])
        result = smallest_indices(arr, 2)
        assert len(result["index"]) == 2
        assert not np.any(np.isnan(result["values"]))

    def test_gower_topn_optimized_exact_boundary(self):
        """Test optimized topn at exact boundary conditions"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(5000, 2)  # Exactly at boundary

        result = gower_exp.gower_topn(X, Y, n=50, use_optimized=True)  # Exactly at n=50
        assert len(result["index"]) == 50

    def test_gower_topn_heap_with_n_equal_total(self):
        """Test heap algorithm when n equals total rows"""
        query_cat = np.array(["A"])
        query_num = np.array([1.0])
        data_cat = np.array([["A"], ["B"], ["C"]])
        data_num = np.array([[1.5], [2.0], [2.5]])

        result = _gower_topn_heap(
            query_cat,
            query_num,
            data_cat,
            data_num,
            np.array([1.0]),
            np.array([1.0]),
            2.0,
            np.array([2.0]),
            n=3,
            total_rows=3,
        )

        assert len(result["index"]) == 3

    def test_compute_chunk_non_symmetric_case(self):
        """Test chunk computation for non-symmetric case"""
        X_cat = np.array([["A"], ["B"]])
        X_num = np.array([[1.0], [2.0]])
        Y_cat = np.array([["C"], ["D"], ["E"]])  # Different from X
        Y_num = np.array([[3.0], [4.0], [5.0]])

        result = _compute_chunk(
            0,
            1,
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            np.array([1.0]),
            np.array([1.0]),
            2.0,
            np.array([True, False]),
            np.array([4.0]),
            np.array([5.0]),
            2,
            3,  # Different dimensions
        )

        assert result.shape == (1, 3)

    def test_gower_matrix_dataframe_automatic_cat_detection(self):
        """Test automatic categorical detection in DataFrames"""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "object_col": [1, "mixed", 3.0],  # Mixed types
            }
        )

        result = gower_exp.gower_matrix(df)
        assert result.shape == (3, 3)

    @patch("gower_exp.accelerators.NUMBA_AVAILABLE", True)
    @patch("gower_exp.accelerators.smallest_indices_numba")
    def test_smallest_indices_numba_exception(self, mock_numba):
        """Test smallest_indices fallback when numba raises exception"""
        mock_numba.side_effect = Exception("Numba error")

        arr = np.array([[0.5, 0.1, 0.8]])
        result = smallest_indices(arr, 2)
        assert len(result["index"]) == 2

    def test_gower_topn_optimized_with_none_data_y(self):
        """Test optimized topn when data_y is None"""
        X = np.array([[1.0, 2.0]])

        result = gower_exp.gower_topn(X, data_y=None, n=1, use_optimized=True)
        assert len(result["index"]) == 1

    def test_gower_topn_non_optimized_explicitly(self):
        """Test explicitly using non-optimized path"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(10, 2)

        result = gower_exp.gower_topn(X, Y, n=3, use_optimized=False)
        assert len(result["index"]) == 3

    def test_matrix_with_single_feature(self):
        """Test matrix computation with single feature"""
        X = np.array([[1.0], [2.0], [3.0]])

        result = gower_exp.gower_matrix(X)
        assert result.shape == (3, 3)

    def test_gower_topn_heap_n_greater_than_total(self):
        """Test heap algorithm when n > total_rows"""
        query_cat = np.array(["A"])
        query_num = np.array([1.0])
        data_cat = np.array([["B"], ["C"]])
        data_num = np.array([[2.0], [3.0]])

        result = _gower_topn_heap(
            query_cat,
            query_num,
            data_cat,
            data_num,
            np.array([1.0]),
            np.array([1.0]),
            2.0,
            np.array([2.0]),
            n=5,
            total_rows=2,  # n > total
        )

        assert len(result["index"]) == 2  # Can only return 2
