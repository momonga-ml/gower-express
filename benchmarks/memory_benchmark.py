#!/usr/bin/env python3
"""
Memory usage benchmark for Gower distance optimizations.

This benchmark measures memory usage and performance improvements
from memory optimization changes to the gower_exp package.
"""

import gc
import os

# Add the parent directory to path and import the optimized gower_exp package
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gower_exp


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Fallback to tracemalloc if psutil not available
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024  # Convert to MB


def create_test_dataset(
    n_samples: int, n_features: int, categorical_ratio: float = 0.3
) -> pd.DataFrame:
    """Create a mixed-type test dataset with specified characteristics"""
    np.random.seed(42)  # For reproducibility

    n_categorical = int(n_features * categorical_ratio)
    n_numerical = n_features - n_categorical

    data = {}

    # Create categorical features
    for i in range(n_categorical):
        categories = [f"cat_{i}_{j}" for j in range(10)]  # 10 categories per feature
        data[f"cat_feature_{i}"] = np.random.choice(categories, size=n_samples)

    # Create numerical features
    for i in range(n_numerical):
        data[f"num_feature_{i}"] = np.random.normal(0, 1, size=n_samples)

    return pd.DataFrame(data)


def benchmark_memory_and_time(
    dataset_sizes: List[Tuple[int, int]], test_functions: List[str]
) -> Dict[str, List[Dict]]:
    """
    Benchmark memory usage and execution time for different dataset sizes.

    Args:
        dataset_sizes: List of (n_samples, n_features) tuples
        test_functions: List of test function names to run

    Returns:
        Dictionary with benchmark results
    """
    results = {func_name: [] for func_name in test_functions}

    for n_samples, n_features in dataset_sizes:
        print(f"\nTesting dataset size: {n_samples} samples x {n_features} features")

        # Create test dataset
        data = create_test_dataset(n_samples, n_features)
        print(f"Dataset created. Memory usage: {get_memory_usage():.1f} MB")

        for func_name in test_functions:
            print(f"  Running {func_name}...")

            # Clear memory before test
            gc.collect()

            # Start memory and time tracking
            tracemalloc.start()
            memory_before = get_memory_usage()
            start_time = time.time()

            try:
                # Run the test
                if func_name == "gower_matrix_small":
                    result = gower_exp.gower_matrix(data.head(min(100, n_samples)))
                elif func_name == "gower_matrix_medium":
                    result = gower_exp.gower_matrix(data.head(min(500, n_samples)))
                elif func_name == "gower_matrix_full":
                    result = gower_exp.gower_matrix(data)
                elif func_name == "gower_matrix_vectorized":
                    # Force vectorized implementation
                    result = gower_exp.gower_matrix(data, n_jobs=1)
                elif func_name == "gower_matrix_parallel":
                    # Force parallel implementation
                    result = gower_exp.gower_matrix(data, n_jobs=2)
                elif func_name == "gower_topn":
                    query = data.head(1)
                    result = gower_exp.gower_topn(query, data, n=10)
                else:
                    continue

                # Record timing
                end_time = time.time()
                execution_time = end_time - start_time

                # Record memory usage
                memory_after = get_memory_usage()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                peak_memory_mb = peak / 1024 / 1024
                memory_delta = memory_after - memory_before

                # Store results
                result_dict = {
                    "dataset_size": (n_samples, n_features),
                    "execution_time": execution_time,
                    "memory_before_mb": memory_before,
                    "memory_after_mb": memory_after,
                    "memory_delta_mb": memory_delta,
                    "peak_memory_mb": peak_memory_mb,
                    "output_shape": getattr(result, "shape", None)
                    if hasattr(result, "shape")
                    else len(result)
                    if isinstance(result, dict)
                    else None,
                }
                results[func_name].append(result_dict)

                print(
                    f"    Time: {execution_time:.3f}s, Peak Memory: {peak_memory_mb:.1f}MB, Delta: {memory_delta:+.1f}MB"
                )

                # Cleanup
                del result

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                tracemalloc.stop()
                results[func_name].append(
                    {"dataset_size": (n_samples, n_features), "error": str(e)}
                )

            # Force garbage collection between tests
            gc.collect()

    return results


def analyze_memory_efficiency(results: Dict[str, List[Dict]]) -> None:
    """Analyze and print memory efficiency metrics"""
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY ANALYSIS")
    print("=" * 60)

    for func_name, func_results in results.items():
        if not func_results or "error" in func_results[0]:
            continue

        print(f"\n{func_name.upper()}:")
        print("-" * 40)

        for result in func_results:
            n_samples, n_features = result["dataset_size"]
            output_shape = result["output_shape"]

            # Calculate theoretical minimum memory for output
            if output_shape and isinstance(output_shape, tuple):
                theoretical_output_mb = (
                    (output_shape[0] * output_shape[1] * 4) / 1024 / 1024
                )  # 4 bytes per float32
            else:
                theoretical_output_mb = 0

            # Calculate memory efficiency ratio
            peak_memory = result["peak_memory_mb"]
            efficiency_ratio = (
                theoretical_output_mb / peak_memory if peak_memory > 0 else 0
            )

            print(
                f"  Dataset: {n_samples:4d}x{n_features:2d} | "
                f"Time: {result['execution_time']:6.3f}s | "
                f"Peak: {peak_memory:6.1f}MB | "
                f"Efficiency: {efficiency_ratio:.3f}"
            )


def run_memory_stress_test() -> None:
    """Run a stress test to identify memory bottlenecks"""
    print("\n" + "=" * 60)
    print("MEMORY STRESS TEST")
    print("=" * 60)

    # Test with progressively larger datasets to find memory limits
    stress_sizes = [
        (100, 20),
        (500, 50),
        (1000, 100),
        (2000, 50),
        (5000, 20),
    ]

    for n_samples, n_features in stress_sizes:
        print(f"\nStress test: {n_samples} samples x {n_features} features")

        try:
            # Create dataset
            data = create_test_dataset(n_samples, n_features)
            print(
                f"  Dataset memory: ~{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            )

            # Test memory usage with different approaches
            gc.collect()
            _ = get_memory_usage()

            tracemalloc.start()

            # Run computation
            result = gower_exp.gower_matrix(data)

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            _ = get_memory_usage()
            peak_mb = peak / 1024 / 1024

            # Calculate memory amplification factor
            input_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            output_size_mb = (result.size * 4) / 1024 / 1024  # 4 bytes per float32
            amplification = peak_mb / input_size_mb

            print(
                f"  Input: {input_size_mb:.1f}MB, Output: {output_size_mb:.1f}MB, Peak: {peak_mb:.1f}MB"
            )
            print(f"  Memory amplification factor: {amplification:.1f}x")

            del data, result
            gc.collect()

        except MemoryError:
            print("  MEMORY ERROR: Dataset too large")
            break
        except Exception as e:
            print(f"  ERROR: {str(e)}")


def main():
    """Main benchmark function"""
    print("=" * 60)
    print("GOWER DISTANCE MEMORY OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")

    # Define test datasets of increasing size
    dataset_sizes = [
        (50, 10),  # Small dataset
        (100, 20),  # Medium dataset
        (200, 30),  # Larger dataset
        (500, 50),  # Large dataset
    ]

    # Define test functions
    test_functions = [
        "gower_matrix_small",
        "gower_matrix_medium",
        "gower_matrix_vectorized",
        "gower_matrix_parallel",
        "gower_topn",
    ]

    # Run benchmarks
    results = benchmark_memory_and_time(dataset_sizes, test_functions)

    # Analyze results
    analyze_memory_efficiency(results)

    # Run stress test
    run_memory_stress_test()

    print(f"\nFinal memory usage: {get_memory_usage():.1f} MB")
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
