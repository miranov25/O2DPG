#!/usr/bin/env python3
"""
Phase 12.12b.0: Scientific V5 Benchmark
Single source of truth. No calibration factors.

Usage:
    PYTHONPATH=. python benchmarks/bench_v5_scientific.py --output benchmarks/results/v5_scientific.json

Measurements:
    1. V4 baseline (n_jobs=1)
    2. V5 cold (Run 1) — captures T_jit
    3. V5 warm (Runs 2-3) — discard
    4. V5 steady (Runs 4-13) — mean ± std
    5. V5 scaling (n_jobs=1,2,4,8,12)
    6. Memory delta (RSS + tracemalloc)

Output: Single JSON with complete provenance.
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

import numpy as np
import pandas as pd

# Try to import psutil for RSS measurement
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# =============================================================================
# Scenario Definition
# =============================================================================

SCENARIOS = {
    "S4": {"n_rows": 500_000, "n_groups": 5_000, "n_fits": 6, "n_params": 3},
    "S5": {"n_rows": 1_000_000, "n_groups": 10_000, "n_fits": 6, "n_params": 3},
}


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_test_dataframe(
    n_rows: int,
    n_groups: int,
    n_params: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate test DataFrame matching V5 expected format.
    
    Returns DataFrame with:
    - group_id: Group identifier (randomized order)
    - y1-y6: Target variables (6 fits)
    - x1, x2: Predictor variables
    - w: Weights
    """
    np.random.seed(seed)
    
    # Group column (randomized order like real data)
    rows_per_group = n_rows // n_groups
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    # Handle remainder
    remainder = n_rows - len(group_ids)
    if remainder > 0:
        group_ids = np.concatenate([group_ids, np.arange(remainder)])
    np.random.shuffle(group_ids)
    
    # Build DataFrame
    data = {
        "group_id": group_ids,
        "y1": np.random.randn(n_rows).astype(np.float64),
        "y2": np.random.randn(n_rows).astype(np.float64),
        "y3": np.random.randn(n_rows).astype(np.float64),
        "y4": np.random.randn(n_rows).astype(np.float64),
        "y5": np.random.randn(n_rows).astype(np.float64),
        "y6": np.random.randn(n_rows).astype(np.float64),
        "x1": np.random.randn(n_rows).astype(np.float64),
        "x2": np.random.randn(n_rows).astype(np.float64),
        "w": (np.abs(np.random.randn(n_rows)) + 0.1).astype(np.float64),
    }
    
    return pd.DataFrame(data)


# =============================================================================
# V4 and V5 Measurement Functions
# =============================================================================

def measure_v4(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols: List[str],
    weights: str,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """
    Measure V4 baseline performance.
    
    V4 can accept multiple fit_columns, so we call it once with all fits.
    """
    try:
        from groupby_regression.groupby_regression_optimized import make_parallel_fit_v4
    except ImportError:
        return {"error": "Cannot import make_parallel_fit_v4"}
    
    times_ms = []
    
    for run_idx in range(n_runs):
        gc.collect()
        
        t0 = time.perf_counter()
        
        # V4 accepts fit_columns (plural) - call once with all fits
        result = make_parallel_fit_v4(
            df=df,
            gb_columns=gb_cols,
            fit_columns=fit_cols,  # All fits at once
            suffix="_v4",
            linear_columns=linear_cols,
            weights=weights,
        )
        
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    
    return {
        "times_ms": times_ms,
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "n_runs": n_runs,
        "n_fits": len(fit_cols),
    }


def measure_v5_single(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols_list: List[List[str]],
    weights_list: List[str],
    suffixes: List[str],
    n_jobs: int = 1,
) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Single V5 measurement. Returns (time_ms, result_df).
    """
    try:
        from groupby_regression.groupby_regression_optimized import make_parallel_fit_v5
    except ImportError:
        return -1.0, None
    
    gc.collect()
    
    t0 = time.perf_counter()
    
    result = make_parallel_fit_v5(
        df=df,
        gb_columns=gb_cols,
        fit_columns=fit_cols,
        suffixes=suffixes,
        linear_columns=linear_cols_list,
        weights=weights_list,
        n_jobs=n_jobs,
    )
    
    t1 = time.perf_counter()
    
    return (t1 - t0) * 1000, result


def measure_v5_full(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols_list: List[List[str]],
    weights_list: List[str],
    suffixes: List[str],
    n_jobs: int = 1,
    n_warmup: int = 2,
    n_steady: int = 10,
) -> Dict[str, Any]:
    """
    Measure V5 with cold/warm/steady separation.
    
    Execution order:
    1. Run 1: Cold (includes JIT)
    2. Runs 2-3: Warmup (discard)
    3. Runs 4-13: Steady state (record)
    """
    # Run 1: Cold
    t_cold, _ = measure_v5_single(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs
    )
    
    if t_cold < 0:
        return {"error": "Cannot import make_parallel_fit_v5"}
    
    # Runs 2-3: Warmup (discard)
    for _ in range(n_warmup):
        measure_v5_single(
            df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs
        )
    
    # Runs 4+: Steady state
    steady_times = []
    for _ in range(n_steady):
        t, _ = measure_v5_single(
            df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs
        )
        steady_times.append(t)
    
    # Compute warm (first steady run after warmup)
    t_warm = steady_times[0]
    
    return {
        "cold_ms": round(t_cold, 2),
        "warm_ms": round(t_warm, 2),
        "steady_ms": round(statistics.mean(steady_times), 2),
        "steady_std_ms": round(statistics.stdev(steady_times), 2) if len(steady_times) > 1 else 0.0,
        "steady_times_ms": [round(t, 2) for t in steady_times],
        "T_jit_ms": round(t_cold - t_warm, 2),
        "n_warmup": n_warmup,
        "n_steady": n_steady,
        "n_jobs": n_jobs,
    }


def measure_v5_scaling(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols_list: List[List[str]],
    weights_list: List[str],
    suffixes: List[str],
    thread_counts: List[int] = [1, 2, 4, 8, 12],
    n_warmup: int = 2,
    n_steady: int = 5,
) -> Dict[str, Any]:
    """
    Measure V5 parallel scaling across thread counts.
    """
    results = {}
    baseline_time = None
    
    for n_jobs in thread_counts:
        v5_result = measure_v5_full(
            df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes,
            n_jobs=n_jobs, n_warmup=n_warmup, n_steady=n_steady
        )
        
        if "error" in v5_result:
            results[str(n_jobs)] = v5_result
            continue
        
        if baseline_time is None:
            baseline_time = v5_result["steady_ms"]
        
        speedup = baseline_time / v5_result["steady_ms"] if v5_result["steady_ms"] > 0 else 0
        efficiency = speedup / n_jobs if n_jobs > 0 else 0
        
        results[str(n_jobs)] = {
            "steady_ms": v5_result["steady_ms"],
            "std_ms": v5_result["steady_std_ms"],
            "speedup": round(speedup, 2),
            "efficiency": round(efficiency, 2),
        }
    
    return {
        "thread_counts": thread_counts,
        "results": results,
        "baseline_ms": baseline_time,
    }


# =============================================================================
# Memory Measurement
# =============================================================================

def measure_memory(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols_list: List[List[str]],
    weights_list: List[str],
    suffixes: List[str],
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Measure memory usage during V5 execution.
    
    Reports:
    - RSS peak (OS allocation)
    - tracemalloc peak (Python allocations)
    """
    try:
        from groupby_regression.groupby_regression_optimized import make_parallel_fit_v5
    except ImportError:
        return {"error": "Cannot import make_parallel_fit_v5"}
    
    gc.collect()
    
    # Get baseline RSS
    rss_before = 0
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        rss_before = process.memory_info().rss
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Run V5
    result = make_parallel_fit_v5(
        df=df,
        gb_columns=gb_cols,
        fit_columns=fit_cols,
        suffixes=suffixes,
        linear_columns=linear_cols_list,
        weights=weights_list,
        n_jobs=n_jobs,
    )
    
    # Get tracemalloc peak
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Get RSS after
    rss_after = 0
    if PSUTIL_AVAILABLE:
        rss_after = process.memory_info().rss
    
    return {
        "rss_before_mb": round(rss_before / 1024 / 1024, 1),
        "rss_after_mb": round(rss_after / 1024 / 1024, 1),
        "rss_delta_mb": round((rss_after - rss_before) / 1024 / 1024, 1),
        "tracemalloc_current_mb": round(current / 1024 / 1024, 1),
        "tracemalloc_peak_mb": round(peak / 1024 / 1024, 1),
        "result_rows": len(result),
        "psutil_available": PSUTIL_AVAILABLE,
    }


# =============================================================================
# Environment and Metadata
# =============================================================================

def get_git_info() -> Dict[str, str]:
    """Get current git revision and status."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
        
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        dirty = "dirty" if status else "clean"
        
        return {"revision": rev, "status": dirty}
    except Exception:
        return {"revision": "unknown", "status": "unknown"}


def get_tool_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    versions = {"python": platform.python_version()}
    
    for pkg in ["numpy", "pandas", "numba"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = mod.__version__
        except ImportError:
            versions[pkg] = "unavailable"
    
    return versions


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information."""
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_scientific_benchmark(
    scenario: str = "S4",
    primitives_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete scientific benchmark for V5.
    
    Returns single JSON structure with all measurements and provenance.
    """
    params = SCENARIOS.get(scenario, SCENARIOS["S4"])
    n_rows = params["n_rows"]
    n_groups = params["n_groups"]
    n_fits = params["n_fits"]
    n_params = params["n_params"]
    
    if verbose:
        print("=" * 70)
        print("Phase 12.12b.0: Scientific V5 Benchmark")
        print("=" * 70)
        print(f"Scenario: {scenario}")
        print(f"  Rows: {n_rows:,}, Groups: {n_groups:,}")
        print(f"  Fits: {n_fits}, Params: {n_params}")
        print()
    
    # Generate test data
    if verbose:
        print("Generating test data...")
    df = generate_test_dataframe(n_rows, n_groups, n_params)
    
    # Setup parameters
    gb_cols = ["group_id"]
    fit_cols = ["y1", "y2", "y3", "y4", "y5", "y6"][:n_fits]
    linear_cols = ["x1", "x2"][:n_params - 1]  # Intercept is implicit
    weights = "w"
    
    suffixes = [f"_{i+1}" for i in range(n_fits)]
    linear_cols_list = [linear_cols for _ in range(n_fits)]
    weights_list = [weights for _ in range(n_fits)]
    
    # Initialize results structure
    results = {
        "meta": {
            "phase": "12.12b.0",
            "type": "scientific_benchmark",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "scenario": scenario,
            "parameters": params,
            "git": get_git_info(),
            "tool_versions": get_tool_versions(),
            "hardware": get_hardware_info(),
        },
        "inputs": {
            "primitives_json": primitives_path,
        },
    }
    
    # ==========================================================================
    # 1. V4 Baseline
    # ==========================================================================
    if verbose:
        print("\n[1/5] Measuring V4 baseline (n_jobs=1)...")
    
    v4_result = measure_v4(df, gb_cols, fit_cols, linear_cols, weights, n_runs=5)
    results["v4_baseline"] = v4_result
    
    if verbose:
        if "error" in v4_result:
            print(f"  ERROR: {v4_result['error']}")
        else:
            print(f"  V4: {v4_result['mean_ms']:.1f} ± {v4_result['std_ms']:.1f} ms")
    
    # ==========================================================================
    # 2. V5 Sequential (Cold/Warm/Steady)
    # ==========================================================================
    if verbose:
        print("\n[2/5] Measuring V5 sequential (cold/warm/steady)...")
    
    v5_seq = measure_v5_full(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes,
        n_jobs=1, n_warmup=2, n_steady=10
    )
    results["v5_sequential"] = v5_seq
    
    if verbose:
        if "error" in v5_seq:
            print(f"  ERROR: {v5_seq['error']}")
        else:
            print(f"  V5 cold:   {v5_seq['cold_ms']:.1f} ms")
            print(f"  V5 warm:   {v5_seq['warm_ms']:.1f} ms")
            print(f"  V5 steady: {v5_seq['steady_ms']:.1f} ± {v5_seq['steady_std_ms']:.1f} ms")
            print(f"  T_jit:     {v5_seq['T_jit_ms']:.1f} ms")
    
    # ==========================================================================
    # 3. V5 vs V4 Comparison
    # ==========================================================================
    if verbose:
        print("\n[3/5] Computing V5 vs V4 speedup...")
    
    if "error" not in v4_result and "error" not in v5_seq:
        v5_vs_v4_speedup = v4_result["mean_ms"] / v5_seq["steady_ms"] if v5_seq["steady_ms"] > 0 else 0
        results["comparison"] = {
            "v5_vs_v4_speedup": round(v5_vs_v4_speedup, 2),
            "v4_ms": v4_result["mean_ms"],
            "v5_steady_ms": v5_seq["steady_ms"],
        }
        
        if verbose:
            print(f"  V4:  {v4_result['mean_ms']:.1f} ms")
            print(f"  V5:  {v5_seq['steady_ms']:.1f} ms")
            print(f"  Speedup: {v5_vs_v4_speedup:.2f}×")
    
    # ==========================================================================
    # 4. V5 Parallel Scaling
    # ==========================================================================
    if verbose:
        print("\n[4/5] Measuring V5 parallel scaling...")
    
    v5_scaling = measure_v5_scaling(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes,
        thread_counts=[1, 2, 4, 8, 12],
        n_warmup=2, n_steady=5
    )
    results["v5_scaling"] = v5_scaling
    
    if verbose:
        print(f"  {'n_jobs':>6} {'time_ms':>10} {'speedup':>8} {'efficiency':>10}")
        print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
        for n_jobs, data in v5_scaling["results"].items():
            if "error" not in data:
                print(f"  {n_jobs:>6} {data['steady_ms']:>10.1f} {data['speedup']:>8.2f} {data['efficiency']:>10.2f}")
    
    # ==========================================================================
    # 5. Memory Measurement
    # ==========================================================================
    if verbose:
        print("\n[5/5] Measuring memory usage...")
    
    memory = measure_memory(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs=1
    )
    results["memory"] = memory
    
    if verbose:
        if "error" in memory:
            print(f"  ERROR: {memory['error']}")
        else:
            print(f"  RSS before:        {memory['rss_before_mb']:.1f} MB")
            print(f"  RSS after:         {memory['rss_after_mb']:.1f} MB")
            print(f"  RSS delta:         {memory['rss_delta_mb']:.1f} MB")
            print(f"  tracemalloc peak:  {memory['tracemalloc_peak_mb']:.1f} MB")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if "comparison" in results:
            c = results["comparison"]
            print(f"  V5 vs V4 speedup:    {c['v5_vs_v4_speedup']:.2f}×")
        
        if "v5_scaling" in results and "12" in results["v5_scaling"]["results"]:
            s12 = results["v5_scaling"]["results"]["12"]
            if "error" not in s12:
                print(f"  V5 parallel (12 threads): {s12['speedup']:.2f}× speedup, {s12['efficiency']:.0%} efficiency")
        
        if "memory" in results and "error" not in results["memory"]:
            print(f"  Memory (RSS delta):  {results['memory']['rss_delta_mb']:.1f} MB")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 12.12b.0: Scientific V5 Benchmark",
    )
    parser.add_argument(
        "--scenario", "-s",
        default="S4",
        choices=list(SCENARIOS.keys()),
        help="Scenario to benchmark (default: S4)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--primitives", "-p",
        default=None,
        help="Path to primitives JSON file (for provenance)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    results = run_scientific_benchmark(
        scenario=args.scenario,
        primitives_path=args.primitives,
        verbose=not args.quiet,
    )
    
    if args.output:
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults written to: {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
