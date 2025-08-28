#!/usr/bin/env python3
"""
Benchmark script to compare the original vs vectorized Gower distance implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
from gower import gower_matrix

def create_test_data(n_samples=1000, n_features=10, cat_ratio=0.5):
    """Create test data with mixed numeric and categorical features."""
    np.random.seed(42)
    
    n_cat_features = int(n_features * cat_ratio)
    n_num_features = n_features - n_cat_features
    
    # Generate numeric features
    numeric_data = np.random.randn(n_samples, n_num_features) * 100 + 50
    
    # Generate categorical features
    cat_data = []
    for _ in range(n_cat_features):
        # Create categorical with 3-5 unique values
        n_unique = np.random.randint(3, 6)
        categories = [f'cat_{i}' for i in range(n_unique)]
        cat_col = np.random.choice(categories, n_samples)
        cat_data.append(cat_col)
    
    if n_cat_features > 0:
        cat_data = np.column_stack(cat_data)
        data = np.column_stack([numeric_data, cat_data])
    else:
        data = numeric_data
    
    # Create categorical features mask
    cat_features = np.concatenate([
        np.zeros(n_num_features, dtype=bool),
        np.ones(n_cat_features, dtype=bool)
    ])
    
    return data, cat_features

def benchmark_gower_matrix(data, cat_features, n_runs=3):
    """Benchmark gower_matrix function."""
    times = []
    
    for i in range(n_runs):
        start_time = time.time()
        result = gower_matrix(data, cat_features=cat_features)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return result, avg_time, std_time

def main():
    print("Gower Distance Vectorization Benchmark")
    print("=" * 50)
    
    # Test different matrix sizes
    test_sizes = [
        (100, 6),   # Small
        (500, 10),  # Medium
        (1000, 10), # Large
    ]
    
    for n_samples, n_features in test_sizes:
        print(f"\nTesting with {n_samples} samples, {n_features} features:")
        print("-" * 40)
        
        # Create test data
        data, cat_features = create_test_data(n_samples, n_features)
        
        print(f"Data shape: {data.shape}")
        print(f"Categorical features: {cat_features.sum()}/{len(cat_features)}")
        print(f"Expected output shape: ({n_samples}, {n_samples})")
        
        # Benchmark
        result, avg_time, std_time = benchmark_gower_matrix(data, cat_features)
        
        print(f"Average time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"Output shape: {result.shape}")
        print(f"Result range: [{result.min():.6f}, {result.max():.6f}]")
        
        # Verify symmetry (should be symmetric for same dataset)
        if np.allclose(result, result.T, rtol=1e-5):
            print("✓ Matrix is symmetric")
        else:
            print("✗ Matrix is not symmetric")
        
        # Verify diagonal is zero (distance to self should be 0)
        diagonal = np.diag(result)
        if np.allclose(diagonal, 0, atol=1e-6):
            print("✓ Diagonal is zero")
        else:
            print(f"✗ Diagonal is not zero (max: {diagonal.max():.6f})")

if __name__ == "__main__":
    main()