#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized gower_topn implementations.
Tests performance improvements for top-N nearest neighbor search.
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add the gower package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import gower


def create_test_data(n_samples=1000, n_features=6, mixed_types=True):
    """Create test dataset with mixed categorical and numerical features."""
    np.random.seed(42)
    
    data = {}
    
    # Numerical features
    data['age'] = np.random.randint(18, 80, n_samples)
    data['salary'] = np.random.uniform(20000, 150000, n_samples)
    data['experience'] = np.random.randint(0, 40, n_samples)
    
    if mixed_types:
        # Categorical features
        data['gender'] = np.random.choice(['M', 'F', 'Other'], n_samples)
        data['education'] = np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n_samples)
        data['department'] = np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_samples)
    else:
        # All numerical for comparison
        data['score1'] = np.random.uniform(0, 100, n_samples)
        data['score2'] = np.random.uniform(0, 100, n_samples)  
        data['score3'] = np.random.uniform(0, 100, n_samples)
    
    return pd.DataFrame(data)


def smallest_indices(ary, n):
    """Returns the n smallest indices from a numpy array."""
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {'index': indices, 'values': values}


def original_gower_topn(data_x, data_y=None, weight=None, cat_features=None, n=5):
    """Original implementation using full matrix computation."""
    if data_x.shape[0] != 1:
        raise TypeError("Only support `data_x` of 1 row.")
    
    # Compute full distance matrix
    dm = gower.gower_matrix(data_x, data_y, weight, cat_features)
    
    # Use the smallest_indices function
    return smallest_indices(np.nan_to_num(dm[0], nan=1), n)


def benchmark_implementation(func_name, func, query, data, n_values, num_runs=5):
    """Benchmark a specific implementation."""
    results = {}
    
    for n in n_values:
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = func(query, data, n=n)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"Error in {func_name} with n={n}: {e}")
                times.append(float('inf'))
                break
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[n] = {'avg': avg_time, 'std': std_time, 'times': times}
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing implementations."""
    print("Gower Distance Top-N Optimization Benchmark")
    print("=" * 50)
    
    # Test configurations
    dataset_sizes = [100, 500, 1000, 2000]
    n_values = [5, 10, 20, 50]
    
    for size in dataset_sizes:
        print(f"\nDataset Size: {size} samples")
        print("-" * 30)
        
        # Create test data
        data = create_test_data(n_samples=size)
        query = data.iloc[0:1]  # Single query point
        
        print(f"Data shape: {data.shape}")
        print(f"Query shape: {query.shape}")
        
        # Benchmark original implementation
        print("\nBenchmarking Original Implementation...")
        original_results = benchmark_implementation(
            "Original", original_gower_topn, query, data, n_values, num_runs=3
        )
        
        # Benchmark optimized implementation  
        print("Benchmarking Optimized Implementation...")
        optimized_results = benchmark_implementation(
            "Optimized", gower.gower_topn, query, data, n_values, num_runs=3
        )
        
        # Compare results
        print(f"\n{'N':<5} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10} {'Accuracy':<10}")
        print("-" * 70)
        
        for n in n_values:
            if n in original_results and n in optimized_results:
                orig_time = original_results[n]['avg']
                opt_time = optimized_results[n]['avg']
                
                if opt_time > 0 and orig_time < float('inf'):
                    speedup = orig_time / opt_time
                    speedup_str = f"{speedup:.1f}x"
                else:
                    speedup_str = "N/A"
                
                # Quick accuracy check (compare first result)
                try:
                    orig_result = original_gower_topn(query, data, n=n)
                    opt_result = gower.gower_topn(query, data, n=n)
                    
                    # Check if top result is the same (allowing for minor floating point differences)
                    accuracy = "OK" if len(orig_result['index']) > 0 and len(opt_result['index']) > 0 and \
                              abs(orig_result['values'][0] - opt_result['values'][0]) < 1e-6 else "DIFF"
                except:
                    accuracy = "ERR"
                
                print(f"{n:<5} {orig_time:<15.6f} {opt_time:<15.6f} {speedup_str:<10} {accuracy:<10}")
        
        print()


def detailed_correctness_test():
    """Test that optimized implementation produces same results as original."""
    print("Correctness Verification")
    print("=" * 30)
    
    # Create test data
    data = create_test_data(n_samples=100)
    query = data.iloc[0:1]
    
    test_cases = [
        {"n": 5, "description": "Top-5 neighbors"},
        {"n": 10, "description": "Top-10 neighbors"},
        {"n": 20, "description": "Top-20 neighbors"},
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        n = test_case["n"]
        desc = test_case["description"]
        
        try:
            # Get results from both implementations
            original_result = original_gower_topn(query, data, n=n)
            optimized_result = gower.gower_topn(query, data, n=n)
            
            # Compare indices (may be in different order for ties)
            orig_indices = set(original_result['index'])
            opt_indices = set(optimized_result['index'])
            
            # Compare distances (should be very close)
            orig_distances = sorted(original_result['values'])
            opt_distances = sorted(optimized_result['values'])
            
            indices_match = orig_indices == opt_indices
            distances_close = np.allclose(orig_distances, opt_distances, rtol=1e-5, atol=1e-8)
            
            status = "PASS" if indices_match and distances_close else "FAIL"
            if status == "FAIL":
                all_passed = False
            
            print(f"{desc}: {status}")
            if not indices_match:
                print(f"  Index mismatch: orig={orig_indices}, opt={opt_indices}")
            if not distances_close:
                print(f"  Distance mismatch: max_diff={np.max(np.abs(np.array(orig_distances) - np.array(opt_distances)))}")
                
        except Exception as e:
            print(f"{desc}: ERROR - {e}")
            all_passed = False
    
    print(f"\nOverall correctness: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    # Run correctness tests first
    if detailed_correctness_test():
        print("\nProceeding with performance benchmarks...\n")
        run_comprehensive_benchmark()
    else:
        print("\nCorrectness tests failed! Please fix implementation before benchmarking.")