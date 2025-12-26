#!/usr/bin/env python3
"""
Phase 12.12: Scientific Performance Framework — Complete Primitive Benchmarks

Entry point for running ALL primitive microbenchmarks.

Usage:
    python bench_primitives_v2.py [--output results.json] [--scenario S4]
    
Output:
    BF-compatible JSON with threading info, all primitive results, and diagnosis.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

import numpy as np

# Import primitives
from primitives.overhead import (
    bench_python_loop,
    bench_numba_loop,
    bench_numba_dispatch,
    bench_parallel_overhead,
    get_threading_info,
    NUMBA_AVAILABLE,
)
from primitives.compute import (
    compare_median,
    compare_mad,
    compare_dot_XtWX,
    compare_solve,
    compare_cholesky,
    compare_cond,
    bench_matrix_vector,
    bench_residual,
)
from primitives.memory import (
    bench_stream_read,
    bench_gather,
    bench_scatter_reduce,
    bench_alloc_copy,
    bench_inner_loop_alloc,
    bench_strided_gather,
)
from primitives.sort import (
    bench_argsort,
    bench_group_boundaries,
    bench_unique_count,
    bench_boundary_to_slices,
    compare_boundaries,
)
from theory.roofline import (
    parallel_efficiency,
    is_overhead_dominated,
    diagnose_parallel_failure,
)


# =============================================================================
# Scenario Definitions (match bench_groupby_fits.py)
# =============================================================================

SCENARIOS = {
    "S1": {"n_rows": 10_000, "n_groups": 100, "n_fits": 6, "n_params": 3},
    "S2": {"n_rows": 50_000, "n_groups": 500, "n_fits": 6, "n_params": 3},
    "S3": {"n_rows": 100_000, "n_groups": 1_000, "n_fits": 6, "n_params": 3},
    "S4": {"n_rows": 500_000, "n_groups": 5_000, "n_fits": 6, "n_params": 3},
    "S5": {"n_rows": 1_000_000, "n_groups": 10_000, "n_fits": 6, "n_params": 3},
    "S6": {"n_rows": 5_000_000, "n_groups": 50_000, "n_fits": 6, "n_params": 3},
}


# =============================================================================
# Environment and Metadata
# =============================================================================

def get_tool_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    versions = {
        "python": platform.python_version(),
    }
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except:
        versions["numpy"] = "unavailable"
    
    try:
        import numba
        versions["numba"] = numba.__version__
    except:
        versions["numba"] = "unavailable"
    
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except:
        versions["pandas"] = "unavailable"
    
    return versions


def get_env_id() -> str:
    """Generate unique environment ID hash."""
    versions = get_tool_versions()
    env_str = f"{platform.system()}_{platform.machine()}_{versions}"
    return hashlib.md5(env_str.encode()).hexdigest()[:8]


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

def run_primitive_benchmarks(
    scenario: str = "S4",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ALL primitive benchmarks.
    
    Returns BF-compatible JSON structure.
    """
    params = SCENARIOS.get(scenario, SCENARIOS["S4"])
    n_rows = params["n_rows"]
    n_groups = params["n_groups"]
    n_fits = params["n_fits"]
    n_params = params["n_params"]
    rows_per_group = n_rows // n_groups
    
    # Total calls for operations
    n_per_group_fit_calls = n_groups * n_fits  # 30,000 for S4
    n_median_calls = n_per_group_fit_calls * 2  # 60,000 (2 medians per MAD)
    n_mad_calls = n_per_group_fit_calls  # 30,000
    
    if verbose:
        print("=" * 70)
        print("Phase 12.12: Complete Primitive Benchmarks")
        print("=" * 70)
        print(f"Scenario: {scenario}")
        print(f"  Rows: {n_rows:,}")
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows/group: {rows_per_group}")
        print(f"  Fits: {n_fits}")
        print(f"  Params: {n_params}")
        print(f"  Per-group-fit calls: {n_per_group_fit_calls:,}")
        print()
    
    # Get threading info FIRST (parallel validity gate)
    threading_info = get_threading_info()
    
    if verbose:
        print("Threading Info:")
        print(f"  Layer: {threading_info['layer']}")
        print(f"  Threads: {threading_info['num_threads']}")
        print(f"  Parallel active: {threading_info['parallel_active']}")
        print(f"  Parallel valid: {threading_info['parallel_valid']}")
        print()
    
    benchmarks = []
    diagnosis = []
    ratios = {}  # Track NumPy vs Numba ratios
    
    # -------------------------------------------------------------------------
    # O1-O4: Overhead Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("OVERHEAD PRIMITIVES (O1-O4)")
        print("=" * 70)
    
    # O1: Python loop
    o1 = bench_python_loop(n_iterations=n_groups)
    benchmarks.append(o1.to_dict())
    if verbose:
        print(f"  O1 python_loop: {o1.time_per_call_ns:.1f} ns/iter")
    
    # O2: Numba loop (serial)
    o2_serial = bench_numba_loop(n_iterations=n_groups, parallel=False)
    benchmarks.append(o2_serial.to_dict())
    if verbose:
        print(f"  O2 numba_loop (serial): {o2_serial.time_per_call_ns:.1f} ns/iter")
    
    # O2: Numba loop (parallel)
    o2_parallel = bench_numba_loop(n_iterations=n_groups, parallel=True)
    benchmarks.append(o2_parallel.to_dict())
    if verbose:
        print(f"  O2 numba_loop (parallel): {o2_parallel.time_per_call_ns:.1f} ns/iter")
    
    # O3: Numba dispatch
    o3 = bench_numba_dispatch(n_calls=n_groups)
    benchmarks.append(o3.to_dict())
    if verbose:
        print(f"  O3 numba_dispatch: {o3.time_per_call_ns:.1f} ns/call")
    
    # O4: Parallel overhead (PRIORITY)
    o4 = bench_parallel_overhead(n_groups=n_groups)
    benchmarks.append(o4.to_dict())
    if verbose:
        print(f"  O4 parallel_overhead: {o4.time_per_call_ns:.1f} ns/group")
    print()
    
    # -------------------------------------------------------------------------
    # S1-S4: Sort/Index Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("SORT/INDEX PRIMITIVES (S1-S4)")
        print("=" * 70)
    
    # S1: Argsort
    s1 = bench_argsort(n_rows=n_rows)
    benchmarks.append(s1.to_dict())
    if verbose:
        print(f"  S1 argsort: {s1.time_per_row_ns:.2f} ns/row ({s1.wall_time_s*1000:.1f} ms total)")
    
    # S2: Group boundaries (NumPy vs Numba)
    s2_numpy, s2_numba, s2_ratio = compare_boundaries(n_rows=n_rows, n_groups=n_groups)
    benchmarks.append(s2_numpy.to_dict())
    benchmarks.append(s2_numba.to_dict())
    ratios["S2"] = s2_ratio
    if verbose:
        print(f"  S2 group_boundaries (numpy): {s2_numpy.time_per_row_ns:.2f} ns/row")
        print(f"  S2 group_boundaries (numba): {s2_numba.time_per_row_ns:.2f} ns/row")
        print(f"     Ratio: {s2_ratio:.2f}×")
    
    # S3: Unique count
    s3 = bench_unique_count(n_rows=n_rows, n_groups=n_groups)
    benchmarks.append(s3.to_dict())
    if verbose:
        print(f"  S3 unique_count: {s3.time_per_row_ns:.2f} ns/row ({s3.wall_time_s*1000:.1f} ms total)")
    
    # S4: Boundary to slices
    s4 = bench_boundary_to_slices(n_groups=n_groups, rows_per_group=rows_per_group)
    benchmarks.append(s4.to_dict())
    if verbose:
        print(f"  S4 boundary_to_slices: {s4.time_per_row_ns:.1f} ns/group ({s4.wall_time_s*1000:.1f} ms total)")
    print()
    
    # -------------------------------------------------------------------------
    # M1-M5: Memory Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("MEMORY PRIMITIVES (M1-M5)")
        print("=" * 70)
    
    # M1: Stream read
    m1 = bench_stream_read(n_rows=n_rows)
    benchmarks.append(m1.to_dict())
    if verbose:
        print(f"  M1 stream_read: {m1.bandwidth_gbs:.1f} GB/s")
    
    # M2: Gather (random access)
    m2 = bench_gather(n_rows=n_rows)
    benchmarks.append(m2.to_dict())
    if verbose:
        print(f"  M2 gather: {m2.bandwidth_gbs:.1f} GB/s")
    
    # M3: Scatter reduce
    m3 = bench_scatter_reduce(n_rows=n_rows, n_groups=n_groups)
    benchmarks.append(m3.to_dict())
    if verbose:
        print(f"  M3 scatter_reduce: {m3.bandwidth_gbs:.1f} GB/s")
    
    # M4: Allocation + copy
    m4 = bench_alloc_copy(n_rows=n_rows)
    benchmarks.append(m4.to_dict())
    if verbose:
        print(f"  M4 alloc_copy: {m4.bandwidth_gbs:.1f} GB/s")
    
    # M4a: Inner loop allocation
    m4a = bench_inner_loop_alloc(n_groups=n_groups, rows_per_group=rows_per_group)
    benchmarks.append(m4a.to_dict())
    if verbose:
        print(f"  M4a inner_loop_alloc: {m4a.bandwidth_gbs:.1f} GB/s ({m4a.wall_time_s*1000:.1f} ms)")
    
    # M5: Strided gather
    m5 = bench_strided_gather(n_rows=n_rows)
    benchmarks.append(m5.to_dict())
    if verbose:
        print(f"  M5 strided_gather: {m5.bandwidth_gbs:.1f} GB/s")
    print()
    
    # -------------------------------------------------------------------------
    # C1-C8: Compute Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("COMPUTE PRIMITIVES (C1-C8)")
        print("=" * 70)
    
    # C1: dot_XtWX
    c1_numpy, c1_numba, c1_ratio = compare_dot_XtWX(
        n_rows=rows_per_group, n_cols=n_params, n_calls=n_per_group_fit_calls
    )
    benchmarks.append(c1_numpy.to_dict())
    benchmarks.append(c1_numba.to_dict())
    ratios["C1"] = c1_ratio
    if verbose:
        print(f"  C1 dot_XtWX (numpy): {c1_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C1 dot_XtWX (numba): {c1_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c1_ratio:.2f}×")
    
    # C2: solve
    c2_numpy, c2_numba, c2_ratio = compare_solve(n_params=n_params, n_calls=n_per_group_fit_calls)
    benchmarks.append(c2_numpy.to_dict())
    benchmarks.append(c2_numba.to_dict())
    ratios["C2"] = c2_ratio
    if verbose:
        print(f"  C2 solve (numpy): {c2_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C2 solve (numba): {c2_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c2_ratio:.2f}×")
    
    # C3: cholesky
    c3_numpy, c3_numba, c3_ratio = compare_cholesky(n_params=n_params, n_calls=n_per_group_fit_calls)
    benchmarks.append(c3_numpy.to_dict())
    benchmarks.append(c3_numba.to_dict())
    ratios["C3"] = c3_ratio
    if verbose:
        print(f"  C3 cholesky (numpy): {c3_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C3 cholesky (numba): {c3_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c3_ratio:.2f}×")
    
    # C4: condition number
    c4_numpy, c4_numba, c4_ratio = compare_cond(n_params=n_params, n_calls=n_per_group_fit_calls)
    benchmarks.append(c4_numpy.to_dict())
    benchmarks.append(c4_numba.to_dict())
    ratios["C4"] = c4_ratio
    if verbose:
        print(f"  C4 cond (numpy): {c4_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C4 cond (numba proxy): {c4_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c4_ratio:.2f}×")
    
    # C5: Median comparison
    c5_numpy, c5_numba, c5_ratio = compare_median(
        array_size=rows_per_group, n_calls=n_median_calls
    )
    benchmarks.append(c5_numpy.to_dict())
    benchmarks.append(c5_numba.to_dict())
    ratios["C5"] = c5_ratio
    if verbose:
        print(f"  C5 median (numpy): {c5_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C5 median (numba): {c5_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c5_ratio:.2f}×")
    
    # C6: MAD comparison
    c6_numpy, c6_numba, c6_ratio = compare_mad(
        array_size=rows_per_group, n_calls=n_mad_calls
    )
    benchmarks.append(c6_numpy.to_dict())
    benchmarks.append(c6_numba.to_dict())
    ratios["C6"] = c6_ratio
    if verbose:
        print(f"  C6 mad (numpy): {c6_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C6 mad (numba): {c6_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c6_ratio:.2f}×")
    
    # C7: matrix_vector
    c7_numpy = bench_matrix_vector(n_rows=rows_per_group, n_cols=n_params, 
                                    n_calls=n_per_group_fit_calls, implementation="numpy")
    c7_numba = bench_matrix_vector(n_rows=rows_per_group, n_cols=n_params,
                                    n_calls=n_per_group_fit_calls, implementation="numba")
    benchmarks.append(c7_numpy.to_dict())
    benchmarks.append(c7_numba.to_dict())
    c7_ratio = c7_numba.time_per_call_ns / c7_numpy.time_per_call_ns if c7_numpy.time_per_call_ns > 0 else float('inf')
    ratios["C7"] = c7_ratio
    if verbose:
        print(f"  C7 matrix_vector (numpy): {c7_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C7 matrix_vector (numba): {c7_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c7_ratio:.2f}×")
    
    # C8: residual
    c8_numpy = bench_residual(n_rows=rows_per_group, n_cols=n_params,
                              n_calls=n_per_group_fit_calls, implementation="numpy")
    c8_numba = bench_residual(n_rows=rows_per_group, n_cols=n_params,
                              n_calls=n_per_group_fit_calls, implementation="numba")
    benchmarks.append(c8_numpy.to_dict())
    benchmarks.append(c8_numba.to_dict())
    c8_ratio = c8_numba.time_per_call_ns / c8_numpy.time_per_call_ns if c8_numpy.time_per_call_ns > 0 else float('inf')
    ratios["C8"] = c8_ratio
    if verbose:
        print(f"  C8 residual (numpy): {c8_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C8 residual (numba): {c8_numba.time_per_call_ns:.1f} ns/call")
        print(f"     Ratio: {c8_ratio:.2f}×")
    print()
    
    # -------------------------------------------------------------------------
    # Diagnosis
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)
    
    # Check parallel validity
    if not threading_info["parallel_valid"]:
        diagnosis.append("❌ PARALLEL INVALID: Threading layer not active or diagnostics empty")
        diagnosis.append("   → All n_jobs>1 results should be marked INVALID")
    
    # Check Numba ratios
    slow_primitives = []
    fast_primitives = []
    for name, ratio in ratios.items():
        if ratio != float('inf') and ratio > 0:
            if ratio > 2.0:
                slow_primitives.append((name, ratio))
            elif ratio < 0.5:
                fast_primitives.append((name, ratio))
    
    if slow_primitives:
        for name, ratio in slow_primitives:
            diagnosis.append(f"⚠️ {name}: Numba is {ratio:.1f}× slower than NumPy")
    
    if fast_primitives:
        for name, ratio in fast_primitives:
            if ratio > 0:
                diagnosis.append(f"✅ {name}: Numba is {1/ratio:.1f}× faster than NumPy")
    
    # Memory bandwidth analysis
    bandwidth_ratio = m1.bandwidth_gbs / m2.bandwidth_gbs if m2.bandwidth_gbs > 0 else 0
    if bandwidth_ratio > 2:
        diagnosis.append(f"⚠️ Random access ({m2.bandwidth_gbs:.1f} GB/s) is {bandwidth_ratio:.1f}× slower than contiguous ({m1.bandwidth_gbs:.1f} GB/s)")
    
    # Check if overhead dominated
    t_work_per_group_us = (c2_numba.time_per_call_ns + c6_numba.time_per_call_ns) / 1000  # solve + MAD
    t_overhead_us = o4.time_per_call_ns / 1000
    
    if t_overhead_us > 0 and t_work_per_group_us > 0:
        if is_overhead_dominated(t_work_per_group_us, t_overhead_us):
            diagnosis.append(f"⚠️ Thread overhead ({t_overhead_us:.2f} μs) may dominate work per group ({t_work_per_group_us:.1f} μs)")
        else:
            diagnosis.append(f"✅ Work per group ({t_work_per_group_us:.1f} μs) >> thread overhead ({t_overhead_us:.2f} μs)")
    
    if verbose:
        for d in diagnosis:
            print(d)
        if not diagnosis:
            print("✓ No issues detected")
        print()
    
    # -------------------------------------------------------------------------
    # Build output
    # -------------------------------------------------------------------------
    n_passed = sum(1 for b in benchmarks if b.get("status") == "OK")
    n_slow = sum(1 for b in benchmarks if b.get("status") == "SLOW")
    
    # Handle infinity for JSON
    def safe_ratio(r):
        if r == float('inf') or r != r:  # inf or nan
            return None
        return round(r, 2)
    
    output = {
        "meta": {
            "suite": "primitives",
            "version": "12.12.2",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "env_id": get_env_id(),
            "tool_versions": get_tool_versions(),
            "hardware": get_hardware_info(),
            "scenario": scenario,
            "parameters": params,
            "threading": threading_info,
        },
        "summary": {
            "n_primitives": len(benchmarks),
            "n_passed": n_passed,
            "n_slow": n_slow,
            "ratios": {k: safe_ratio(v) for k, v in ratios.items()},
            "parallel_overhead_ns": round(o4.time_per_call_ns, 1),
            "stream_bandwidth_gbs": round(m1.bandwidth_gbs, 1),
            "gather_bandwidth_gbs": round(m2.bandwidth_gbs, 1),
        },
        "benchmarks": benchmarks,
        "diagnosis": diagnosis,
    }
    
    return output


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 12.12: Complete Primitive Microbenchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", "-s",
        default="S4",
        choices=list(SCENARIOS.keys()),
        help="Scenario to use for sizing (default: S4)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_primitive_benchmarks(
        scenario=args.scenario,
        verbose=not args.quiet,
    )
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {args.output}")
    else:
        if args.quiet:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
