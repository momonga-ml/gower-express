#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time

def create_simple_test():
    """Create simple test data"""
    data = np.array([
        [1.0, 'A', 10.0],
        [2.0, 'B', 20.0],
        [3.0, 'A', 30.0]
    ], dtype=object)
    cat_features = [False, True, False]
    return data, cat_features

def test_simple_case():
    """Test with simple case to verify functionality"""
    from gower import gower_matrix

    data, cat_features = create_simple_test()
    print("Testing with simple data:")
    print(data)
    print("Cat features:", cat_features)

    result = gower_matrix(data, cat_features=cat_features)
    print("Result:")
    print(result)
    print("Diagonal (should be 0):", np.diag(result))
    return result is not None

if __name__ == "__main__":
    test_simple_case()
