"""Test fallback behaviors for accelerators module.

This file tests the fallback paths when dependencies are not available.
"""

import sys
from unittest.mock import MagicMock

import numpy as np


def test_cupy_fallback_paths():
    """Test CuPy fallback and GPU detection"""
    # Save original modules
    original_cupy = sys.modules.get("cupy")
    original_accelerators = sys.modules.get("gower_exp.accelerators")

    try:
        # Test 1: CuPy available but CUDA not available (line 63-65)
        mock_cupy = MagicMock()
        mock_cupy.cuda.is_available.return_value = False

        sys.modules["cupy"] = mock_cupy
        if "gower_exp.accelerators" in sys.modules:
            del sys.modules["gower_exp.accelerators"]

        import gower_exp.accelerators as acc1

        # This should trigger line 63-65 and set GPU_AVAILABLE = False
        assert not acc1.GPU_AVAILABLE
        assert acc1.cp is not acc1.np  # cp should be the mock, not np

        # Test 2: CuPy not available (line 67-68)
        if "cupy" in sys.modules:
            del sys.modules["cupy"]
        if "gower_exp.accelerators" in sys.modules:
            del sys.modules["gower_exp.accelerators"]

        import gower_exp.accelerators as acc2

        assert not acc2.GPU_AVAILABLE
        assert acc2.cp is acc2.np  # Should fallback to numpy

    finally:
        # Restore modules
        if original_cupy is not None:
            sys.modules["cupy"] = original_cupy
        elif "cupy" in sys.modules:
            del sys.modules["cupy"]

        if original_accelerators is not None:
            sys.modules["gower_exp.accelerators"] = original_accelerators
        elif "gower_exp.accelerators" in sys.modules:
            del sys.modules["gower_exp.accelerators"]


def test_get_array_module_coverage():
    """Test get_array_module function to cover remaining lines"""
    import gower_exp.accelerators as acc

    # These should cover the remaining lines in get_array_module
    result1 = acc.get_array_module(use_gpu=False)
    assert result1 is np

    result2 = acc.get_array_module()  # Default parameter
    assert result2 is np
