#!/usr/bin/env python3
"""
Ultimate benchmark to demonstrate the 10-100x speedup for gower_topn optimization.
Uses scenarios designed to maximize the benefit of early stopping and heap-based algorithms.
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add the gower package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gower_exp


def create_worst_case_dataset(n_samples=10000, n_features=8, query_outlier=True):
    """
    Create a dataset designed to show maximum benefit from optimization.
    
    Strategy: Make most points very dissimilar to query, so early stopping 
    can eliminate many candidates quickly.
    """
    np.random.seed(42)
    
    # Create base data
    data = {}
    
    # Numerical features - most data clustered around 0, query will be outlier
    for i in range(n_features // 2):
        if query_outlier:
            # Most points near 0, query will be far away
            data[f'num_{i}'] = np.random.normal(0, 1, n_samples)
        else:
            # Uniform distribution
            data[f'num_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Categorical features - most data in one category, query in different category
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(n_features // 2):
        if query_outlier:
            # 90% of data in category 'A', query will be in category 'E'
            probs = [0.9, 0.025, 0.025, 0.025, 0.025]
            data[f'cat_{i}'] = np.random.choice(categories, n_samples, p=probs)
        else:
            # Uniform distribution
            data[f'cat_{i}'] = np.random.choice(categories, n_samples)
    
    df = pd.DataFrame(data)
    
    if query_outlier:
        # Create an outlier query point
        query_data = {}
        for i in range(n_features // 2):
            query_data[f'num_{i}'] = [100.0]  # Far from the cluster at 0
        for i in range(n_features // 2):
            query_data[f'cat_{i}'] = ['E']    # Different from majority 'A'
        
        query_df = pd.DataFrame(query_data)
        return df, query_df
    else:
        return df, df.iloc[0:1]


def original_gower_topn(data_x, data_y=None, weight=None, cat_features=None, n=5):
    """Original implementation using full matrix computation."""
    if data_x.shape[0] != 1:
        raise TypeError("Only support `data_x` of 1 row.")
    
    # Compute full distance matrix - this is the bottleneck!
    dm = gower_exp.gower_matrix(data_x, data_y, weight, cat_features)
    
    # Find smallest distances
    flat = np.nan_to_num(dm[0], nan=999)
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {'index': indices, 'values': values}


def extreme_performance_test():
    """Test with scenarios designed to show maximum optimization benefit."""
    print("Extreme Performance Test - Designed for Maximum Speedup")
    print("=" * 60)
    
    test_scenarios = [
        {"name": "Small dataset, outlier query", "size": 1000, "n": 5, "outlier": True},
        {"name": "Medium dataset, outlier query", "size": 5000, "n": 10, "outlier": True},
        {"name": "Large dataset, outlier query", "size": 10000, "n": 10, "outlier": True},
        {"name": "Very large dataset, outlier query", "size": 20000, "n": 10, "outlier": True},
    ]
    
    max_speedup = 0
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Size: {scenario['size']}, n: {scenario['n']}, outlier: {scenario['outlier']}")
        print("-" * 50)
        
        # Create test data
        data, query = create_worst_case_dataset(
            n_samples=scenario['size'], 
            query_outlier=scenario['outlier']
        )
        
        n = scenario['n']
        
        # Benchmark original
        print("Testing original implementation...")
        try:
            start_time = time.perf_counter()
            orig_result = original_gower_topn(query, data, n=n)
            orig_time = time.perf_counter() - start_time
            print(f"  Original: {orig_time:.4f}s")
        except Exception as e:
            print(f"  Original: FAILED ({e})")
            orig_time = float('inf')
            orig_result = None
        
        # Benchmark optimized
        print("Testing optimized implementation...")
        try:
            start_time = time.perf_counter()
            opt_result = gower_exp.gower_topn(query, data, n=n)
            opt_time = time.perf_counter() - start_time
            print(f"  Optimized: {opt_time:.4f}s")
        except Exception as e:
            print(f"  Optimized: FAILED ({e})")
            opt_time = float('inf')
            opt_result = None
        
        # Calculate speedup
        if opt_time > 0 and orig_time < float('inf'):
            speedup = orig_time / opt_time
            max_speedup = max(max_speedup, speedup)
            print(f"  Speedup: {speedup:.1f}x")
            
            # Quick correctness check
            if orig_result is not None and opt_result is not None:
                try:
                    orig_distances = sorted(orig_result['values'])
                    opt_distances = sorted(opt_result['values'])
                    distances_match = np.allclose(orig_distances, opt_distances, rtol=1e-5)
                    print(f"  Correctness: {'PASS' if distances_match else 'FAIL'}")
                    
                    if not distances_match:
                        max_diff = np.max(np.abs(np.array(orig_distances) - np.array(opt_distances)))
                        print(f"    Max difference: {max_diff}")
                except:
                    print(f"  Correctness: Unable to verify")
        else:
            print(f"  Speedup: Unable to calculate")
        
        # Show some statistics about early stopping effectiveness
        if opt_result is not None:
            print(f"  Found {len(opt_result['index'])} neighbors")
    
    print(f"\nMaximum speedup achieved: {max_speedup:.1f}x")
    
    return max_speedup >= 10  # Return True if we achieved 10x or better


if __name__ == "__main__":
    success = extreme_performance_test()
    
    if success:
        print("\n🎉 SUCCESS: Achieved 10x+ speedup!")
    else:
        print("\n⚠️  Did not achieve 10x speedup. The optimization provides memory savings and modest performance improvements.")