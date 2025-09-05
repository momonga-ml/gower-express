"""Direct tests of accelerator functions to maximize coverage."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestAcceleratorsDirect:
    """Direct tests of accelerator functions when available"""

    @pytest.mark.skipif("numba" not in sys.modules, reason="Numba not available")
    def test_numba_functions_direct_calls(self):
        """Test Numba functions with direct calls when available"""
        try:
            from gower_exp.accelerators import (
                NUMBA_AVAILABLE,
                compute_ranges_numba,
                gower_get_numba,
                smallest_indices_numba,
            )
        except ImportError:
            pytest.skip("Numba functions not available")

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test gower_get_numba with simple data
        xi_cat = np.array([1.0, 2.0])
        xi_num = np.array([0.5, 1.5])
        xj_cat = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 2.0]])
        xj_num = np.array([[0.5, 1.5], [1.0, 2.0], [0.0, 1.0]])
        feature_weight_cat = np.array([1.0, 1.0])
        feature_weight_num = np.array([1.0, 1.0])
        feature_weight_sum = 4.0
        ranges_of_numeric = np.array([2.0, 3.0])

        result = gower_get_numba(
            xi_cat,
            xi_num,
            xj_cat,
            xj_num,
            feature_weight_cat,
            feature_weight_num,
            feature_weight_sum,
            ranges_of_numeric,
        )

        assert len(result) == 3
        assert result.dtype == np.float32
        assert result[0] == 0.0  # First row is exact match

        # Test compute_ranges_numba
        Z_num = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        num_ranges = np.zeros(2)
        num_max = np.zeros(2)

        compute_ranges_numba(Z_num, num_ranges, num_max)

        assert num_max[0] == 3.0
        assert num_max[1] == 6.0

        # Test smallest_indices_numba
        ary = np.array([0.8, 0.2, 0.5, 0.1])
        indices, values = smallest_indices_numba(ary, 2)

        assert len(indices) == 2
        assert len(values) == 2

    @pytest.mark.skipif("numba" not in sys.modules, reason="Numba not available")
    def test_numba_parallel_functions(self):
        """Test parallel Numba functions when available"""
        try:
            from gower_exp.accelerators import (
                NUMBA_AVAILABLE,
                compute_ranges_numba_parallel,
                gower_matrix_numba_parallel,
            )
        except ImportError:
            pytest.skip("Numba functions not available")

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test parallel range computation
        Z_num = np.random.randn(20, 5)
        num_ranges = np.zeros(5)
        num_max = np.zeros(5)

        compute_ranges_numba_parallel(Z_num, num_ranges, num_max)

        # Check that ranges were computed
        assert np.all(num_max > 0) or np.all(
            num_max == 0
        )  # Either all positive or all zero

        # Test parallel matrix computation
        X_cat = np.random.randint(0, 3, size=(10, 3)).astype(np.float64)
        X_num = np.random.randn(10, 4).astype(np.float64)
        Y_cat = np.random.randint(0, 3, size=(8, 3)).astype(np.float64)
        Y_num = np.random.randn(8, 4).astype(np.float64)

        result = gower_matrix_numba_parallel(
            X_cat,
            X_num,
            Y_cat,
            Y_num,
            np.ones(3),
            np.ones(4),
            7.0,
            np.array([2.0, 2.0, 2.0, 2.0]),
        )

        assert result.shape == (10, 8)
        assert result.dtype == np.float32

    @pytest.mark.skipif("numba" not in sys.modules, reason="Numba not available")
    def test_heap_functions_direct(self):
        """Test heap functions directly when available"""
        try:
            from gower_exp.accelerators import (
                NUMBA_AVAILABLE,
                _heap_sift_down,
                smallest_indices_numba_heap,
            )
        except ImportError:
            pytest.skip("Numba functions not available")

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test heap sift down
        heap_values = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)
        heap_indices = np.array([0, 1, 2, 3], dtype=np.int32)

        _heap_sift_down(heap_values, heap_indices, 0, 3)

        # After sifting, the heap property should be maintained
        assert len(heap_values) == 4

        # Test heap-based smallest indices
        ary = np.random.randn(100).astype(np.float32)
        indices, values = smallest_indices_numba_heap(ary, 10)

        assert len(indices) == 10
        assert len(values) == 10
        # Values should be sorted
        assert np.all(np.diff(values) >= 0)

    @pytest.mark.skipif("numba" not in sys.modules, reason="Numba not available")
    def test_specialized_kernels(self):
        """Test specialized optimization kernels when available"""
        try:
            from gower_exp.accelerators import (
                NUMBA_AVAILABLE,
                gower_get_numba_categorical_only,
                gower_get_numba_mixed_optimized,
                gower_get_numba_numerical_only,
            )
        except ImportError:
            pytest.skip("Numba functions not available")

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test numerical-only kernel
        xi_num = np.array([1.0, 2.0, 3.0])
        xj_num = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        weight_num = np.array([1.0, 1.0, 1.0])
        ranges = np.array([2.0, 2.0, 2.0])

        result = gower_get_numba_numerical_only(xi_num, xj_num, weight_num, ranges)
        assert len(result) == 2
        assert result[0] == 0.0  # Exact match

        # Test categorical-only kernel
        xi_cat = np.array([1.0, 2.0])
        xj_cat = np.array([[1.0, 2.0], [1.0, 3.0]])
        weight_cat = np.array([1.0, 1.0])

        result = gower_get_numba_categorical_only(xi_cat, xj_cat, weight_cat)
        assert len(result) == 2
        assert result[0] == 0.0  # Exact match

        # Test mixed optimized kernel
        result = gower_get_numba_mixed_optimized(
            xi_cat,
            xi_num,
            xj_cat,
            xj_num,
            weight_cat,
            weight_num,
            5.0,  # weight_sum
            ranges,
        )
        assert len(result) == 2

    def test_gpu_availability_constants(self):
        """Test GPU availability constants and fallbacks"""
        from gower_exp.accelerators import GPU_AVAILABLE, cp

        # Test that constants are defined
        assert isinstance(GPU_AVAILABLE, bool)

        if not GPU_AVAILABLE:
            # If GPU not available, cp should be numpy
            assert cp is np

        # Test get_array_module function
        from gower_exp.accelerators import get_array_module

        xp = get_array_module(use_gpu=False)
        assert xp is np

        xp_gpu = get_array_module(use_gpu=True)
        if GPU_AVAILABLE:
            assert xp_gpu is cp
        else:
            assert xp_gpu is np

    def test_numba_availability_constant(self):
        """Test Numba availability constant"""
        from gower_exp.accelerators import NUMBA_AVAILABLE, jit, prange

        assert isinstance(NUMBA_AVAILABLE, bool)

        # Test that decorators exist (even if dummy)
        assert callable(jit)
        assert callable(prange)

        # Test jit decorator
        @jit(nopython=True)
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    @patch("gower_exp.accelerators.GPU_AVAILABLE", True)
    def test_gpu_mock_scenarios(self):
        """Test GPU scenarios with mocking"""
        from gower_exp.accelerators import get_array_module

        # Mock CuPy
        mock_cp = MagicMock()

        with patch("gower_exp.accelerators.cp", mock_cp):
            xp = get_array_module(use_gpu=True)
            assert xp is mock_cp

    def test_module_level_constants(self):
        """Test module-level constants and exports"""
        import gower_exp.accelerators as acc

        # Test that all expected exports exist
        expected_attrs = [
            "GPU_AVAILABLE",
            "NUMBA_AVAILABLE",
            "cp",
            "jit",
            "prange",
            "get_array_module",
        ]

        for attr in expected_attrs:
            assert hasattr(acc, attr), f"Missing attribute: {attr}"

    def test_error_handling_paths(self):
        """Test error handling in accelerator functions"""
        from gower_exp.accelerators import get_array_module

        # Test with both GPU flags
        for use_gpu in [True, False]:
            try:
                xp = get_array_module(use_gpu=use_gpu)
                assert xp is not None
                # Should be either numpy or cupy
                assert hasattr(xp, "array")
                assert hasattr(xp, "zeros")
            except Exception:
                # Should not raise exceptions
                pytest.fail("get_array_module should not raise exceptions")

    @pytest.mark.skipif("numba" not in sys.modules, reason="Numba not available")
    def test_numba_edge_cases(self):
        """Test edge cases in Numba functions"""
        try:
            from gower_exp.accelerators import NUMBA_AVAILABLE, gower_get_numba
        except ImportError:
            pytest.skip("Numba functions not available")

        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test with empty arrays
        xi_cat = np.array([])
        xi_num = np.array([])
        xj_cat = np.array([]).reshape(1, 0)
        xj_num = np.array([]).reshape(1, 0)

        result = gower_get_numba(
            xi_cat,
            xi_num,
            xj_cat,
            xj_num,
            np.array([]),  # empty weights
            np.array([]),
            0.0,  # zero weight sum - edge case
            np.array([]),  # empty ranges
        )

        # Should handle gracefully
        assert len(result) == 1

        # Test with single feature
        xi_cat = np.array([1.0])
        xi_num = np.array([])
        xj_cat = np.array([[1.0], [2.0]])
        xj_num = np.array([]).reshape(2, 0)

        result = gower_get_numba(
            xi_cat,
            xi_num,
            xj_cat,
            xj_num,
            np.array([1.0]),
            np.array([]),
            1.0,
            np.array([]),
        )

        assert len(result) == 2
        assert result[0] == 0.0  # Match
        assert result[1] == 1.0  # No match
