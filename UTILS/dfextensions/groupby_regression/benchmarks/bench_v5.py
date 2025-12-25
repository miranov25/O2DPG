"""
Benchmark Suite for make_parallel_fit_v5

Phase 12.10.BF: Standardized v5 benchmark with memory profiling.

This module provides:
- v5_batch_fit: Main benchmark for make_parallel_fit_v5
- get_benchmarks(): Discovery function for runner

Usage:
    # Direct execution
    python bench_v5.py --scenario S2 --n-jobs 4
    
    # Via runner
    python -m dfextensions.benchmarks.runner --subproject groupby_regression

Benchmark Design:
- Warmup: 2 runs on minimal data (trigger JIT per n_jobs)
- Timing: 3 runs, report mean ± std
- Memory: Peak RSS (process-wide)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Callable

# =============================================================================
# PATH SETUP FOR JOBLIB WORKERS
# =============================================================================
# Add groupby_regression directory to sys.path so joblib workers can find
# groupby_regression_optimized module (same strategy as working tests)

_gr_dir = Path(__file__).parent.parent.resolve()
_gr_dir_str = str(_gr_dir)

if _gr_dir_str not in sys.path:
    sys.path.insert(0, _gr_dir_str)

# Also set PYTHONPATH for spawned workers (macOS uses spawn)
_current_pythonpath = os.environ.get("PYTHONPATH", "")
if _gr_dir_str not in _current_pythonpath:
    if _current_pythonpath:
        os.environ["PYTHONPATH"] = f"{_gr_dir_str}:{_current_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = _gr_dir_str

# =============================================================================

from .scenarios import (
    Scenario,
    SCENARIOS,
    get_scenarios,
    get_scenario,
    create_test_data,
    create_minimal_warmup_data,
)


# =============================================================================
# LAZY IMPORT FOR make_parallel_fit_v5
# =============================================================================

_v5_func = None

def get_v5_function():
    """
    Lazy import of make_parallel_fit_v5.
    
    Uses direct import (not package import) to match test strategy.
    This ensures __module__ is 'groupby_regression_optimized' which
    joblib workers can resolve.
    """
    global _v5_func
    
    if _v5_func is None:
        # Direct import - same as working tests
        # __module__ will be 'groupby_regression_optimized'
        from groupby_regression_optimized import make_parallel_fit_v5
        _v5_func = make_parallel_fit_v5
    
    return _v5_func


# =============================================================================
# WARMUP
# =============================================================================

_warmed_up = set()  # Track (n_jobs, parallel_backend) combinations

def warmup_v5(n_jobs: int = 1, parallel_backend: str = "auto"):
    """
    Warmup JIT for specific n_jobs configuration.
    
    Called before timing to ensure Numba kernels are compiled.
    Warmup is tracked per (n_jobs, parallel_backend) to avoid redundant work.
    """
    global _warmed_up
    
    key = (n_jobs, parallel_backend)
    if key in _warmed_up:
        return
    
    make_parallel_fit_v5 = get_v5_function()
    df = create_minimal_warmup_data()
    
    # Run once to trigger compilation
    _ = make_parallel_fit_v5(
        df=df,
        gb_columns=["group"],
        fit_columns=["y1"],
        suffixes=["_warmup"],
        linear_columns=[["x1", "x2"]],
        weights=["w"],
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    
    _warmed_up.add(key)


def reset_warmup():
    """Reset warmup tracking (for testing)."""
    global _warmed_up
    _warmed_up.clear()


# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================

def bench_v5_batch_fit(
    scenario: str = "S1",
    n_jobs: int = 1,
    n_chunks: Optional[int] = None,
    parallel_backend: str = "auto",
    seed: int = 42,
    **kwargs,
) -> dict:
    """
    Benchmark make_parallel_fit_v5 batch fitting.
    
    Parameters:
        scenario: Scenario name (S1, S2, S3, S4, S5)
        n_jobs: Number of parallel jobs
        n_chunks: Number of chunks (default: n_jobs)
        parallel_backend: "auto", "numba", or "sequential"
        seed: Random seed for data generation
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Dict with benchmark metadata:
            - n_groups_output: Number of groups in result
            - n_columns_output: Number of columns in result
            - n_rows_input: Number of input rows
    """
    # Get scenario
    scen = get_scenario(scenario)
    
    # Default n_chunks
    if n_chunks is None:
        n_chunks = max(1, n_jobs)
    
    # Create test data
    df = create_test_data(scen, seed=seed)
    
    # Build fit specification (n_fits targets)
    fit_columns = [f"y{i+1}" for i in range(scen.n_fits)]
    suffixes = [f"_F{i}" for i in range(scen.n_fits)]
    linear_columns = [["x1", "x2"]] * scen.n_fits
    weights = ["w"] * scen.n_fits
    
    # Get v5 function
    make_parallel_fit_v5 = get_v5_function()
    
    # Run benchmark
    result = make_parallel_fit_v5(
        df=df,
        gb_columns=["group"],
        fit_columns=fit_columns,
        suffixes=suffixes,
        linear_columns=linear_columns,
        weights=weights,
        n_jobs=n_jobs,
        n_chunks=n_chunks,
        parallel_backend=parallel_backend,
    )
    
    return {
        "n_groups_output": len(result),
        "n_columns_output": len(result.columns),
        "n_rows_input": len(df),
    }


# Benchmark metadata
bench_v5_batch_fit.name = "v5_batch_fit"
bench_v5_batch_fit.version = 1
bench_v5_batch_fit.description = "Batch fitting with make_parallel_fit_v5"


# =============================================================================
# DISCOVERY FUNCTION
# =============================================================================

def get_benchmarks(suite: str = "quick") -> list[dict]:
    """
    Get benchmark specifications for the runner.
    
    Parameters:
        suite: "quick" or "release"
    
    Returns:
        List of benchmark specs with:
            - name: Benchmark name
            - func: Benchmark function
            - scenarios: List of scenario names
            - params: Default parameters
    """
    scenarios = get_scenarios(suite)
    scenario_names = [s.name for s in scenarios]
    
    return [
        {
            "name": "v5_batch_fit",
            "func": bench_v5_batch_fit,
            "scenarios": scenario_names,
            "params": {
                "parallel_backend": "auto",
            },
        },
    ]


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Standalone benchmark execution for quick testing."""
    parser = argparse.ArgumentParser(description="v5_batch_fit benchmark")
    parser.add_argument("--scenario", default="S2", help="Scenario name")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of jobs")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()
    
    print(f"Benchmark: v5_batch_fit")
    print(f"Scenario:  {args.scenario}")
    print(f"n_jobs:    {args.n_jobs}")
    print(f"n_runs:    {args.n_runs}")
    print(f"warmup:    {args.warmup}")
    print()
    
    # Import profiling utilities
    try:
        from dfextensions.benchmarks.profiler import run_benchmark_with_memory, get_peak_rss_mb
    except ImportError:
        # Fallback for standalone execution
        import time
        
        def run_benchmark_with_memory(func, *args, n_runs=3, warmup_runs=2, **kwargs):
            """Simplified timing without memory tracking."""
            # Warmup
            for _ in range(warmup_runs):
                func(*args, **kwargs)
            
            # Time
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - start)
            
            class Stats:
                peak_rss_mb = 0.0
                peak_tracemalloc_mb = None
                top_allocations = None
            
            return times, Stats(), result
    
    # Warmup
    print(f"Warming up (n_jobs={args.n_jobs})...")
    warmup_v5(n_jobs=args.n_jobs)
    
    # Run benchmark
    print(f"Running {args.n_runs} timed runs...")
    
    times, mem_stats, result = run_benchmark_with_memory(
        bench_v5_batch_fit,
        scenario=args.scenario,
        n_jobs=args.n_jobs,
        n_runs=args.n_runs,
        warmup_runs=0,  # Already warmed up
        profile=args.profile,
    )
    
    # Report
    import numpy as np
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    scen = get_scenario(args.scenario)
    throughput = scen.n_rows / mean_time
    
    print()
    print(f"Results:")
    print(f"  Time:       {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"  Peak RSS:   {mem_stats.peak_rss_mb:.0f} MB")
    print(f"  Throughput: {throughput/1_000_000:.2f}M rows/s")
    print(f"  Output:     {result['n_groups_output']} groups, {result['n_columns_output']} columns")


if __name__ == "__main__":
    main()
