#!/usr/bin/env python3
"""
Phase 12.12: Scientific Performance Framework — Primitive Benchmarks

Entry point for running primitive microbenchmarks.

Usage:
    python bench_primitives_v1.py [--output results.json] [--scenario S4]
    
Output:
    BF-compatible JSON with threading info, primitive results, and diagnosis.
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
)
from primitives.memory import (
    bench_stream_read,
    bench_gather,
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
    Run all implemented primitive benchmarks.
    
    Returns BF-compatible JSON structure.
    """
    params = SCENARIOS.get(scenario, SCENARIOS["S4"])
    n_rows = params["n_rows"]
    n_groups = params["n_groups"]
    n_fits = params["n_fits"]
    rows_per_group = n_rows // n_groups
    
    # Total calls for C5/C6 (n_groups × n_fits × 2 for MAD)
    n_median_calls = n_groups * n_fits * 2
    n_mad_calls = n_groups * n_fits
    
    if verbose:
        print("=" * 60)
        print("Phase 12.12: Primitive Benchmarks")
        print("=" * 60)
        print(f"Scenario: {scenario}")
        print(f"  Rows: {n_rows:,}")
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows/group: {rows_per_group}")
        print(f"  Fits: {n_fits}")
        print(f"  Median calls: {n_median_calls:,}")
        print(f"  MAD calls: {n_mad_calls:,}")
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
    
    # -------------------------------------------------------------------------
    # O1-O4: Overhead Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("Running overhead primitives (O1-O4)...")
    
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
    # C5-C6: Compute Primitives (median/mad)
    # -------------------------------------------------------------------------
    if verbose:
        print("Running compute primitives (C5-C6)...")
    
    # C5: Median comparison
    c5_numpy, c5_numba, c5_ratio = compare_median(
        array_size=rows_per_group,
        n_calls=n_median_calls,
    )
    benchmarks.append(c5_numpy.to_dict())
    benchmarks.append(c5_numba.to_dict())
    
    if verbose:
        print(f"  C5 median (numpy): {c5_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C5 median (numba): {c5_numba.time_per_call_ns:.1f} ns/call")
        print(f"  C5 ratio: {c5_ratio:.2f}× {'⚠️ SLOW' if c5_ratio > 3 else '✓'}")
    
    # C6: MAD comparison
    c6_numpy, c6_numba, c6_ratio = compare_mad(
        array_size=rows_per_group,
        n_calls=n_mad_calls,
    )
    benchmarks.append(c6_numpy.to_dict())
    benchmarks.append(c6_numba.to_dict())
    
    if verbose:
        print(f"  C6 mad (numpy): {c6_numpy.time_per_call_ns:.1f} ns/call")
        print(f"  C6 mad (numba): {c6_numba.time_per_call_ns:.1f} ns/call")
        print(f"  C6 ratio: {c6_ratio:.2f}× {'⚠️ SLOW' if c6_ratio > 3 else '✓'}")
    print()
    
    # -------------------------------------------------------------------------
    # M1-M2: Memory Primitives
    # -------------------------------------------------------------------------
    if verbose:
        print("Running memory primitives (M1-M2)...")
    
    # M1: Stream read
    m1 = bench_stream_read(n_rows=n_rows)
    benchmarks.append(m1.to_dict())
    if verbose:
        print(f"  M1 stream_read: {m1.bandwidth_gbs:.1f} GB/s")
    
    # M2: Gather
    m2 = bench_gather(n_rows=n_rows)
    benchmarks.append(m2.to_dict())
    if verbose:
        print(f"  M2 gather: {m2.bandwidth_gbs:.1f} GB/s")
    print()
    
    # -------------------------------------------------------------------------
    # Diagnosis
    # -------------------------------------------------------------------------
    if verbose:
        print("Generating diagnosis...")
    
    # Check if Numba median is slow
    if c5_ratio > 3.0:
        diagnosis.append(f"❌ C5 (median): Numba is {c5_ratio:.1f}× slower than NumPy")
        diagnosis.append("   → V5 MAD computation likely bottleneck")
    
    if c6_ratio > 3.0:
        diagnosis.append(f"❌ C6 (mad): Numba is {c6_ratio:.1f}× slower than NumPy")
    
    # Check if overhead dominated
    # Work per group estimate: solve (~1μs) + mad (~10μs based on C6)
    t_work_per_group_us = c6_numpy.time_per_call_ns / 1000 * 2  # 2 medians per MAD
    t_overhead_us = o4.time_per_call_ns / 1000
    
    if is_overhead_dominated(t_work_per_group_us, t_overhead_us):
        diagnosis.append(f"⚠️ Thread overhead ({t_overhead_us:.1f} μs) may exceed work per group ({t_work_per_group_us:.1f} μs)")
    
    # Check parallel validity
    if not threading_info["parallel_valid"]:
        diagnosis.append("❌ PARALLEL INVALID: Threading layer not active or diagnostics empty")
        diagnosis.append("   → All n_jobs>1 results should be marked INVALID")
    
    # Memory bandwidth comparison
    bandwidth_ratio = m1.bandwidth_gbs / m2.bandwidth_gbs if m2.bandwidth_gbs > 0 else 0
    if bandwidth_ratio > 2:
        diagnosis.append(f"⚠️ Random access ({m2.bandwidth_gbs:.1f} GB/s) is {bandwidth_ratio:.1f}× slower than contiguous ({m1.bandwidth_gbs:.1f} GB/s)")
    
    if verbose:
        print()
        print("=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)
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
            "version": "12.12.1",
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
            "median_ratio": safe_ratio(c5_ratio),
            "mad_ratio": safe_ratio(c6_ratio),
            "parallel_overhead_ns": round(o4.time_per_call_ns, 1),
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
        description="Phase 12.12: Primitive Microbenchmarks",
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
