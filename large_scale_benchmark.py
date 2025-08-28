#!/usr/bin/env python3
"""
Large-scale benchmark for gower_topn optimization.
Tests with datasets where the optimization should show significant speedup.
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add the gower package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import gower


def create_large_diverse_dataset(n_samples=10000, n_features=10):
    """Create a large, diverse dataset to test optimization effectiveness."""
    np.random.seed(42)
    
    data = {}
    
    # Numerical features with different scales
    data['age'] = np.random.randint(18, 80, n_samples)
    data['salary'] = np.random.uniform(20000, 200000, n_samples)
    data['experience_years'] = np.random.randint(0, 40, n_samples)
    data['credit_score'] = np.random.randint(300, 850, n_samples)
    data['years_at_company'] = np.random.randint(0, 20, n_samples)
    
    # Categorical features with varying cardinality
    data['gender'] = np.random.choice(['M', 'F', 'Other'], n_samples)
    data['education'] = np.random.choice(['HS', 'Associates', 'Bachelor', 'Master', 'PhD'], n_samples)
    data['department'] = np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Operations', 'Legal'], n_samples)
    data['job_level'] = np.random.choice(['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director'], n_samples)
    data['location'] = np.random.choice([f'City_{i}' for i in range(20)], n_samples)
    
    return pd.DataFrame(data)


def original_gower_topn(data_x, data_y=None, weight=None, cat_features=None, n=5):
    """Original implementation using full matrix computation."""
    if data_x.shape[0] != 1:
        raise TypeError("Only support `data_x` of 1 row.")
    
    # Compute full distance matrix
    dm = gower.gower_matrix(data_x, data_y, weight, cat_features)
    
    # Find smallest distances
    flat = np.nan_to_num(dm[0], nan=999)
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {'index': indices, 'values': values}


def detailed_performance_comparison():
    """Compare performance across different scenarios."""
    print("Large-Scale Performance Comparison")
    print("=" * 50)
    
    scenarios = [
        {"name": "Small query (n=5)", "size": 5000, "n": 5},
        {"name": "Medium query (n=20)", "size": 5000, "n": 20},
        {"name": "Large dataset (n=10)", "size": 10000, "n": 10},
        {"name": "Very large dataset (n=10)", "size": 20000, "n": 10},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Dataset size: {scenario['size']}, n: {scenario['n']}")
        print("-" * 40)
        
        # Create test data
        data = create_large_diverse_dataset(n_samples=scenario['size'])
        query = data.iloc[0:1]
        
        n = scenario['n']
        num_runs = 3
        
        # Benchmark original
        print("Running original implementation...")
        original_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            orig_result = original_gower_topn(query, data, n=n)
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
            print(f"  Run {i+1}: {end_time - start_time:.4f}s")
        
        orig_avg = np.mean(original_times)
        
        # Benchmark optimized
        print("Running optimized implementation...")
        optimized_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            opt_result = gower.gower_topn(query, data, n=n)
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
            print(f"  Run {i+1}: {end_time - start_time:.4f}s")
        
        opt_avg = np.mean(optimized_times)
        
        # Calculate speedup
        speedup = orig_avg / opt_avg if opt_avg > 0 else float('inf')
        
        print(f"\nResults:")
        print(f"  Original avg: {orig_avg:.4f}s")
        print(f"  Optimized avg: {opt_avg:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Verify correctness
        orig_distances = sorted(orig_result['values'])
        opt_distances = sorted(opt_result['values'])
        distances_match = np.allclose(orig_distances, opt_distances, rtol=1e-5)
        print(f"  Correctness: {'PASS' if distances_match else 'FAIL'}")
        
        if not distances_match:
            print(f"    Max diff: {np.max(np.abs(np.array(orig_distances) - np.array(opt_distances)))}")


def memory_usage_comparison():
    """Compare memory usage between implementations."""
    print("\nMemory Usage Analysis")
    print("=" * 30)
    
    import psutil
    import gc
    
    # Test with large dataset
    data = create_large_diverse_dataset(n_samples=5000)
    query = data.iloc[0:1]
    
    # Measure original implementation memory
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    result_orig = original_gower_topn(query, data, n=10)
    
    mem_after_orig = process.memory_info().rss / 1024 / 1024  # MB
    mem_used_orig = mem_after_orig - mem_before
    
    # Clear memory
    del result_orig
    gc.collect()
    
    # Measure optimized implementation memory
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    result_opt = gower.gower_topn(query, data, n=10)
    
    mem_after_opt = process.memory_info().rss / 1024 / 1024  # MB
    mem_used_opt = mem_after_opt - mem_before
    
    print(f"Original implementation memory usage: {mem_used_orig:.2f} MB")
    print(f"Optimized implementation memory usage: {mem_used_opt:.2f} MB")
    print(f"Memory reduction: {(mem_used_orig - mem_used_opt) / mem_used_orig * 100:.1f}%")


def scalability_test():
    """Test how performance scales with dataset size."""
    print("\nScalability Analysis")
    print("=" * 30)
    
    sizes = [1000, 2000, 5000, 10000]
    n = 10
    
    print(f"{'Size':<8} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        data = create_large_diverse_dataset(n_samples=size)
        query = data.iloc[0:1]
        
        # Time original
        start = time.perf_counter()
        orig_result = original_gower_topn(query, data, n=n)
        orig_time = time.perf_counter() - start
        
        # Time optimized
        start = time.perf_counter()
        opt_result = gower.gower_topn(query, data, n=n)
        opt_time = time.perf_counter() - start
        
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        print(f"{size:<8} {orig_time:<15.4f} {opt_time:<15.4f} {speedup:<10.1f}x")


if __name__ == "__main__":
    detailed_performance_comparison()
    memory_usage_comparison()  
    scalability_test()