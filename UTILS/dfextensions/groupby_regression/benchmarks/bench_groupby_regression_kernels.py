#!/usr/bin/env python
"""
Phase 12.14.GB: Kernel Performance Benchmark Suite

Standalone benchmark for groupby regression kernels.
Run this to verify performance on your platform.

Usage:
    python bench_groupby_regression_kernels.py
    python bench_groupby_regression_kernels.py --quick     # Fast mode
    python bench_groupby_regression_kernels.py --full      # Extended benchmarks

Author: Team 3 Coder
Date: 2025-12-31
Phase: 12.14.GB
"""

import numpy as np
import time
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

# Import kernels - handle various directory structures
_import_success = False

# Try 1: Direct import (when module is in same directory or PYTHONPATH)
if not _import_success:
    try:
        from groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            fit_groups_dispatch,
            INVALID_ASSUME_CLEAN, INVALID_DETECT, INVALID_FILTER,
            _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

# Try 2: Relative import from benchmarks/ subdirectory
if not _import_success:
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            fit_groups_dispatch,
            INVALID_ASSUME_CLEAN, INVALID_DETECT, INVALID_FILTER,
            _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

# Try 3: Package import
if not _import_success:
    try:
        from ..groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            fit_groups_dispatch,
            INVALID_ASSUME_CLEAN, INVALID_DETECT, INVALID_FILTER,
            _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

if not _import_success:
    print("ERROR: Could not import groupby_regression_kernels")
    print("Make sure the module is in the parent directory or PYTHONPATH")
    sys.exit(1)


def generate_data(n_groups, rows_per_group, n_feat, n_targets=1, seed=42):
    """Generate test data for benchmarking."""
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group
    
    X_all = np.random.randn(n_rows, n_feat)
    
    if n_targets == 1:
        Y_all = np.random.randn(n_rows)
    else:
        Y_all = np.random.randn(n_rows, n_targets)
    
    W_all = np.empty(0, dtype=np.float64)  # Unweighted
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)
    
    return X_all, Y_all, W_all, offsets


def benchmark_single_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                         n_runs=5, warmup=2):
    """Benchmark single-fit kernel."""
    # Allocate outputs
    out_beta = np.empty((n_groups, n_params), dtype=np.float64)
    out_errors = np.empty((n_groups, n_params), dtype=np.float64)
    out_rms = np.empty(n_groups, dtype=np.float64)
    out_mad = np.empty(n_groups, dtype=np.float64)
    out_status = np.empty(n_groups, dtype=np.uint8)
    out_n_valid = np.empty(n_groups, dtype=np.int64)
    out_n_filtered = np.empty(n_groups, dtype=np.int64)
    out_cond = np.empty(n_groups, dtype=np.float64)
    
    # Warmup
    for _ in range(warmup):
        fit_groups_single_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit_groups_single_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
        times.append(time.perf_counter() - t0)
    
    return np.median(times), np.std(times)


def benchmark_multi_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
                        n_runs=5, warmup=2):
    """Benchmark multi-fit kernel."""
    # Allocate outputs
    out_beta = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
    out_errors = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
    out_rms = np.empty((n_groups, n_targets), dtype=np.float64)
    out_mad = np.empty((n_groups, n_targets), dtype=np.float64)
    out_status = np.empty((n_groups, n_targets), dtype=np.uint8)
    out_n_valid = np.empty(n_groups, dtype=np.int64)
    out_n_filtered = np.empty(n_groups, dtype=np.int64)
    out_cond = np.empty(n_groups, dtype=np.float64)
    
    # Warmup
    for _ in range(warmup):
        fit_groups_multifit_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_targets, n_params,
            True, 5, False,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit_groups_multifit_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_targets, n_params,
            True, 5, False,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
        times.append(time.perf_counter() - t0)
    
    return np.median(times), np.std(times)


def benchmark_numpy_baseline(X_all, Y_all, offsets, n_groups, n_runs=3):
    """Benchmark NumPy lstsq baseline."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for gi in range(n_groups):
            i0, i1 = offsets[gi], offsets[gi + 1]
            X_slice = X_all[i0:i1]
            Y_slice = Y_all[i0:i1] if Y_all.ndim == 1 else Y_all[i0:i1, 0]
            X_design = np.column_stack([np.ones(len(X_slice)), X_slice])
            np.linalg.lstsq(X_design, Y_slice, rcond=None)
        times.append(time.perf_counter() - t0)
    
    return np.median(times), np.std(times)


def run_benchmark_suite(scenarios, n_runs=5):
    """Run complete benchmark suite."""
    results = []
    
    for scenario in scenarios:
        name = scenario['name']
        n_groups = scenario['n_groups']
        rows_per_group = scenario['rows_per_group']
        n_feat = scenario['n_feat']
        n_targets = scenario.get('n_targets', 1)
        n_params = n_feat + 1
        
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"{'='*60}")
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows/group: {rows_per_group}")
        print(f"  Features: {n_feat}")
        print(f"  Parameters: {n_params}")
        print(f"  Targets: {n_targets}")
        
        # Generate data
        X_all, Y_all, W_all, offsets = generate_data(
            n_groups, rows_per_group, n_feat, n_targets
        )
        
        result = {
            'name': name,
            'n_groups': n_groups,
            'rows_per_group': rows_per_group,
            'n_feat': n_feat,
            'n_targets': n_targets,
            'n_params': n_params,
        }
        
        # NumPy baseline
        numpy_time, numpy_std = benchmark_numpy_baseline(
            X_all, Y_all if n_targets == 1 else Y_all[:, 0:1], offsets, n_groups
        )
        result['numpy_time_ms'] = numpy_time * 1000
        result['numpy_groups_per_sec'] = n_groups / numpy_time
        print(f"  NumPy baseline: {numpy_time*1000:.2f} ms ({n_groups/numpy_time:,.0f} groups/sec)")
        
        # Single-fit kernel
        if n_targets == 1:
            Y_1d = Y_all
        else:
            Y_1d = Y_all[:, 0]
        
        single_time, single_std = benchmark_single_fit(
            X_all, Y_1d, W_all, offsets, n_groups, n_feat, n_params, n_runs
        )
        result['single_time_ms'] = single_time * 1000
        result['single_groups_per_sec'] = n_groups / single_time
        result['numba_vs_numpy_speedup'] = numpy_time / single_time
        print(f"  Single-fit: {single_time*1000:.2f} ms ({n_groups/single_time:,.0f} groups/sec)")
        print(f"  → Numba vs NumPy: {numpy_time/single_time:.1f}×")
        
        # Multi-fit kernel (if multiple targets)
        if n_targets > 1:
            multi_time, multi_std = benchmark_multi_fit(
                X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params, n_runs
            )
            
            # Compare to single-fit called n_targets times
            single_total_time = single_time * n_targets
            
            result['multi_time_ms'] = multi_time * 1000
            result['multi_groups_per_sec'] = n_groups / multi_time
            result['multi_vs_single_speedup'] = single_total_time / multi_time
            
            print(f"  Multi-fit: {multi_time*1000:.2f} ms ({n_groups/multi_time:,.0f} groups/sec)")
            print(f"  Single×{n_targets}: {single_total_time*1000:.2f} ms")
            print(f"  → Multi vs Single×{n_targets}: {single_total_time/multi_time:.2f}×")
        
        results.append(result)
    
    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Scenario':<30} {'Groups/sec':>15} {'Numba/NumPy':>12} {'Multi/Single':>12}")
    print("-"*80)
    
    for r in results:
        groups_per_sec = r.get('single_groups_per_sec', 0)
        numba_speedup = r.get('numba_vs_numpy_speedup', 0)
        multi_speedup = r.get('multi_vs_single_speedup', '-')
        
        if isinstance(multi_speedup, float):
            multi_str = f"{multi_speedup:.2f}×"
        else:
            multi_str = multi_speedup
        
        print(f"{r['name']:<30} {groups_per_sec:>15,.0f} {numba_speedup:>11.1f}× {multi_str:>12}")
    
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark groupby regression kernels')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (fewer scenarios)')
    parser.add_argument('--full', action='store_true', help='Full benchmark (more scenarios)')
    parser.add_argument('--json', type=str, help='Output results to JSON file')
    args = parser.parse_args()
    
    if not _NUMBA_AVAILABLE:
        print("ERROR: Numba not available. Install with: pip install numba")
        sys.exit(1)
    
    print("="*80)
    print("PHASE 12.14.GB: KERNEL PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Numba available: {_NUMBA_AVAILABLE}")
    
    # Define scenarios
    if args.quick:
        scenarios = [
            {'name': 'Quick: 500 groups', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Quick: 500 groups, 6 targets', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2, 'n_targets': 6},
        ]
        n_runs = 3
    elif args.full:
        scenarios = [
            # Vary group count
            {'name': '100 groups, 20 rows', 'n_groups': 100, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '1K groups, 20 rows', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '10K groups, 20 rows', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            
            # Vary rows per group
            {'name': '1K groups, 10 rows', 'n_groups': 1000, 'rows_per_group': 10, 'n_feat': 2},
            {'name': '1K groups, 50 rows', 'n_groups': 1000, 'rows_per_group': 50, 'n_feat': 2},
            {'name': '1K groups, 100 rows', 'n_groups': 1000, 'rows_per_group': 100, 'n_feat': 2},
            
            # Vary features
            {'name': '1K groups, 2 feat', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': '1K groups, 4 feat', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 4},
            {'name': '1K groups, 8 feat', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 8},
            
            # Multi-target
            {'name': '1K groups, 2 targets', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 2},
            {'name': '1K groups, 4 targets', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 4},
            {'name': '1K groups, 6 targets', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 6},
            
            # Multi-target with more features (where multi-fit shines)
            {'name': '1K groups, 4 feat, 6 tgt', 'n_groups': 1000, 'rows_per_group': 50, 'n_feat': 4, 'n_targets': 6},
        ]
        n_runs = 7
    else:
        # Default scenarios
        scenarios = [
            {'name': 'Small: 1K groups, 20 rows', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium: 5K groups, 30 rows', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': 'Large: 10K groups, 20 rows', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Multi: 1K, 6 targets', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 6},
            {'name': 'Multi: 1K, 4 feat, 6 tgt', 'n_groups': 1000, 'rows_per_group': 50, 'n_feat': 4, 'n_targets': 6},
        ]
        n_runs = 5
    
    # Run benchmarks
    results = run_benchmark_suite(scenarios, n_runs)
    
    # Print summary
    print_summary(results)
    
    # Verify gates
    print("\n" + "="*80)
    print("PERFORMANCE GATES")
    print("="*80)
    
    # Gate 1: Numba ≥5× faster than NumPy
    numba_speedups = [r['numba_vs_numpy_speedup'] for r in results]
    min_numba_speedup = min(numba_speedups)
    gate1_pass = min_numba_speedup >= 5.0
    print(f"\n1. Numba vs NumPy: {min_numba_speedup:.1f}× (min) {'✓ PASS' if gate1_pass else '✗ FAIL'} (required: ≥5×)")
    
    # Gate 2: Multi-fit not drastically slower
    multi_speedups = [r.get('multi_vs_single_speedup') for r in results if 'multi_vs_single_speedup' in r]
    if multi_speedups:
        min_multi_speedup = min(multi_speedups)
        gate2_pass = min_multi_speedup >= 0.5
        print(f"2. Multi vs Single: {min_multi_speedup:.2f}× (min) {'✓ PASS' if gate2_pass else '✗ FAIL'} (required: ≥0.5×)")
    
    # Save to JSON if requested
    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'numba_available': _NUMBA_AVAILABLE,
            'results': results,
            'gates': {
                'numba_vs_numpy_min': min_numba_speedup,
                'numba_vs_numpy_pass': gate1_pass,
            }
        }
        if multi_speedups:
            output['gates']['multi_vs_single_min'] = min_multi_speedup
            output['gates']['multi_vs_single_pass'] = gate2_pass
        
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
