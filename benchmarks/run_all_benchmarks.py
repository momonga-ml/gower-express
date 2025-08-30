#!/usr/bin/env python3
"""
Master Benchmark Runner

Runs all benchmark scripts systematically and generates a consolidated report
showing the performance improvements from recent optimizations.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_benchmark(script_path, timeout=300):
    """Run a single benchmark script and capture output."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_path}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run the benchmark script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Collect results
        benchmark_result = {
            "script": script_path,
            "duration": duration,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s)")
            print(result.stdout)
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        return benchmark_result

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return {
            "script": script_path,
            "duration": timeout,
            "success": False,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "return_code": -1,
        }
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return {
            "script": script_path,
            "duration": 0,
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }


def main():
    """Run all benchmark scripts in order."""
    print("=" * 80)
    print("GOWER EXPRESS - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing performance improvements from recent optimizations:")
    print("• Fixed top-N algorithm (2-3x speedup expected)")
    print("• Memory optimization (25-40% reduction expected)")
    print("• Enhanced Numba optimizations (15-25% speedup expected)")

    # Define benchmark scripts in order of execution
    benchmark_scripts = [
        "benchmark/clean_benchmark.py",
        "benchmark/benchmark_numba.py",
        "benchmark/benchmark_vectorized.py",
        "benchmark/benchmark_gower_topn.py",
        "benchmark/benchmark_advanced.py",
        "benchmark/large_scale_benchmark.py",
        "benchmark/ultimate_benchmark.py",
        "benchmark/memory_benchmark.py",
        "benchmark/performance_comparison.py",
    ]

    # Check that all scripts exist
    missing_scripts = []
    for script in benchmark_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)

    if missing_scripts:
        print(f"\n❌ Missing benchmark scripts: {missing_scripts}")
        return 1

    # Run all benchmarks
    results = []
    start_time = time.time()

    for script in benchmark_scripts:
        result = run_benchmark(script)
        results.append(result)

    total_time = time.time() - start_time

    # Generate summary report
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total benchmarks: {len(results)}")
    print(f"Successful: {len(successful)} ✅")
    print(f"Failed: {len(failed)} ❌")
    print(f"Total time: {total_time:.1f}s")

    if failed:
        print("\nFAILED BENCHMARKS:")
        for result in failed:
            print(f"  ❌ {result['script']}: {result['stderr']}")

    print("\nSUCCESSFUL BENCHMARKS:")
    for result in successful:
        print(f"  ✅ {result['script']} ({result['duration']:.1f}s)")

    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"

    with open(results_file, "w") as f:
        f.write("GOWER EXPRESS - BENCHMARK RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Successful: {len(successful)}/{len(results)}\n\n")

        for result in results:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"SCRIPT: {result['script']}\n")
            f.write(f"STATUS: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"DURATION: {result['duration']:.1f}s\n")
            f.write(f"RETURN CODE: {result['return_code']}\n")
            f.write(f"{'=' * 60}\n")

            if result["stdout"]:
                f.write("STDOUT:\n")
                f.write(result["stdout"])
                f.write("\n")

            if result["stderr"]:
                f.write("STDERR:\n")
                f.write(result["stderr"])
                f.write("\n")

    print(f"\n📄 Detailed results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review benchmark results above")
    print("2. Update Benchmark.MD with new performance data")
    print("3. Update README.md with performance highlights")
    print(
        "4. Document improvements: top-N fix, memory optimization, Numba enhancements"
    )

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
