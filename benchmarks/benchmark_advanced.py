#!/usr/bin/env python3
"""
Advanced benchmark to test new optimizations:
1. True incremental top-N algorithm
2. GPU acceleration with CuPy
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add the gower package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import gower_exp


def create_test_dataset(n_samples=10000, n_features=10):
    """Create a mixed dataset for testing."""
    np.random.seed(42)

    data = {}

    # Numerical features (50%)
    for i in range(n_features // 2):
        data[f'num_{i}'] = np.random.normal(0, 1, n_samples)

    # Categorical features (50%)
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(n_features // 2):
        data[f'cat_{i}'] = np.random.choice(categories, n_samples)

    return pd.DataFrame(data)


def benchmark_topn_incremental():
    """Benchmark the incremental top-N optimization."""
    print("\n" + "="*60)
    print("BENCHMARK: Incremental Top-N Algorithm")
    print("="*60)

    test_sizes = [100, 500, 1000, 5000, 10000]
    n_values = [5, 10, 20, 50]

    for size in test_sizes:
        print(f"\nDataset size: {size} samples")
        print("-" * 40)

        # Create test data
        data = create_test_dataset(n_samples=size, n_features=8)
        query = data.iloc[0:1]

        for n in n_values:
            if n > size:
                continue

            # Benchmark original (full matrix computation)
            start_time = time.perf_counter()
            result_orig = gower_exp.gower_topn(query, data, n=n, use_optimized=False)
            time_orig = time.perf_counter() - start_time

            # Benchmark optimized (incremental computation)
            start_time = time.perf_counter()
            result_opt = gower_exp.gower_topn(query, data, n=n, use_optimized=True)
            time_opt = time.perf_counter() - start_time

            # Calculate speedup
            speedup = time_orig / time_opt if time_opt > 0 else 0

            # Verify correctness (distances should match)
            orig_values = sorted(result_orig['values'])
            opt_values = sorted(result_opt['values'])
            correct = np.allclose(orig_values, opt_values, rtol=1e-5)

            print(f"  Top-{n:2d}: Original: {time_orig:6.4f}s, "
                  f"Optimized: {time_opt:6.4f}s, "
                  f"Speedup: {speedup:5.1f}x, "
                  f"Correct: {correct}")


def benchmark_gpu_acceleration():
    """Benchmark GPU acceleration if available."""
    print("\n" + "="*60)
    print("BENCHMARK: GPU Acceleration")
    print("="*60)

    # Check if GPU is available
    from gower_exp.gower_dist import GPU_AVAILABLE

    if not GPU_AVAILABLE:
        print("GPU not available - skipping GPU benchmarks")
        print("To enable GPU, install CuPy: pip install cupy-cuda11x")
        return

    test_sizes = [100, 500, 1000, 2000, 5000]

    for size in test_sizes:
        print(f"\nMatrix size: {size}x{size}")
        print("-" * 40)

        # Create test data
        data = create_test_dataset(n_samples=size, n_features=8)

        # Benchmark CPU
        start_time = time.perf_counter()
        dm_cpu = gower_exp.gower_matrix(data, use_gpu=False)
        time_cpu = time.perf_counter() - start_time

        # Benchmark GPU
        start_time = time.perf_counter()
        dm_gpu = gower_exp.gower_matrix(data, use_gpu=True)
        time_gpu = time.perf_counter() - start_time

        # Calculate speedup
        speedup = time_cpu / time_gpu if time_gpu > 0 else 0

        # Verify correctness
        correct = np.allclose(dm_cpu, dm_gpu, rtol=1e-5)

        print(f"  CPU: {time_cpu:6.4f}s, GPU: {time_gpu:6.4f}s, "
              f"Speedup: {speedup:5.1f}x, Correct: {correct}")


def benchmark_combined():
    """Benchmark combined optimizations."""
    print("\n" + "="*60)
    print("BENCHMARK: Combined Optimizations")
    print("="*60)

    # Large dataset test
    print("\nTesting with large dataset (10,000 samples)...")
    data = create_test_dataset(n_samples=10000, n_features=10)
    query = data.iloc[0:1]

    # Test finding top-10 from 10,000 samples
    print("\nFinding top-10 nearest neighbors from 10,000 samples:")

    # Original: Full matrix computation
    start_time = time.perf_counter()
    result_orig = gower_exp.gower_topn(query, data, n=10, use_optimized=False)
    time_orig = time.perf_counter() - start_time
    print(f"  Original (full matrix):     {time_orig:6.4f}s")

    # Optimized: Incremental computation
    start_time = time.perf_counter()
    result_opt = gower_exp.gower_topn(query, data, n=10, use_optimized=True)
    time_opt = time.perf_counter() - start_time
    print(f"  Optimized (incremental):    {time_opt:6.4f}s")

    speedup = time_orig / time_opt if time_opt > 0 else 0
    print(f"  Total speedup:              {speedup:5.1f}x")

    # Verify correctness
    orig_values = sorted(result_orig['values'][:5])  # Check first 5
    opt_values = sorted(result_opt['values'][:5])
    if np.allclose(orig_values, opt_values, rtol=1e-5):
        print(f"  Correctness check:          PASS")
    else:
        print(f"  Correctness check:          FAIL")
        print(f"    Original values: {orig_values}")
        print(f"    Optimized values: {opt_values}")


def main():
    """Run all benchmarks."""
    print("\nGower Distance - Advanced Optimizations Benchmark")
    print("Testing: Incremental Top-N and GPU Acceleration")

    # Run benchmarks
    benchmark_topn_incremental()
    benchmark_gpu_acceleration()
    benchmark_combined()

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
