#!/usr/bin/env python
"""
Phase 12.14a.GB: Memory Tracking Benchmark Suite

Measures memory consumption, RSS drift, and fragmentation for groupby regression kernels.

Usage:
    python bench_groupby_regression_memory.py
    python bench_groupby_regression_memory.py --iterations 100
    python bench_groupby_regression_memory.py --json results.json

Author: Team 3 Coder
Date: 2025-12-31
Phase: 12.14a.GB
"""

import numpy as np
import time
import argparse
import sys
import json
import gc
import os
from datetime import datetime
from pathlib import Path

# Try to import memory tracking tools
try:
    import tracemalloc
    _TRACEMALLOC_AVAILABLE = True
except ImportError:
    _TRACEMALLOC_AVAILABLE = False

try:
    import resource
    _RESOURCE_AVAILABLE = True
except ImportError:
    _RESOURCE_AVAILABLE = False

# Import kernels
_import_success = False

if not _import_success:
    try:
        from groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            INVALID_DETECT,
            _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

if not _import_success:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            INVALID_DETECT,
            _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

if not _import_success:
    print("ERROR: Could not import groupby_regression_kernels")
    sys.exit(1)


def get_rss_mb():
    """Get current RSS (Resident Set Size) in MB."""
    if _RESOURCE_AVAILABLE:
        # Unix/Linux/Mac
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Mac, ru_maxrss is in bytes; on Linux, it's in KB
        if sys.platform == 'darwin':
            return usage.ru_maxrss / (1024 * 1024)
        else:
            return usage.ru_maxrss / 1024
    else:
        # Fallback: try /proc/self/status on Linux
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024  # KB to MB
        except:
            pass
        return 0.0


def get_current_rss_mb():
    """Get current RSS using /proc on Linux or ps on Mac."""
    if sys.platform == 'darwin':
        # Mac: use ps
        import subprocess
        try:
            pid = os.getpid()
            result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], 
                                    capture_output=True, text=True)
            return int(result.stdout.strip()) / 1024  # KB to MB
        except:
            return 0.0
    else:
        # Linux: use /proc
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024  # KB to MB
        except:
            pass
        return 0.0


def generate_data(n_groups, rows_per_group, n_feat, n_targets=1, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group
    
    X_all = np.random.randn(n_rows, n_feat).astype(np.float64)
    
    if n_targets == 1:
        Y_all = np.random.randn(n_rows).astype(np.float64)
    else:
        Y_all = np.random.randn(n_rows, n_targets).astype(np.float64)
    
    W_all = np.empty(0, dtype=np.float64)
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)
    
    return X_all, Y_all, W_all, offsets


def run_kernel_iteration(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                         out_beta, out_errors, out_rms, out_mad,
                         out_status, out_n_valid, out_n_filtered, out_cond):
    """Run a single kernel iteration."""
    fit_groups_single_numba(
        X_all, Y_all, W_all, offsets,
        n_groups, n_feat, n_params,
        True, 5, False, INVALID_DETECT,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
    )


def benchmark_memory_single_run(n_groups, rows_per_group, n_feat, n_iterations,
                                 measure_interval=10):
    """
    Run single-fit kernel multiple times and track memory.
    
    Parameters
    ----------
    n_groups : int
    rows_per_group : int
    n_feat : int
    n_iterations : int
        Number of iterations to run
    measure_interval : int
        Measure RSS every N iterations
    
    Returns
    -------
    dict with memory metrics
    """
    n_params = n_feat + 1
    
    # Generate data
    X_all, Y_all, W_all, offsets = generate_data(n_groups, rows_per_group, n_feat)
    
    # Pre-allocate outputs (reused across iterations)
    out_beta = np.empty((n_groups, n_params), dtype=np.float64)
    out_errors = np.empty((n_groups, n_params), dtype=np.float64)
    out_rms = np.empty(n_groups, dtype=np.float64)
    out_mad = np.empty(n_groups, dtype=np.float64)
    out_status = np.empty(n_groups, dtype=np.uint8)
    out_n_valid = np.empty(n_groups, dtype=np.int64)
    out_n_filtered = np.empty(n_groups, dtype=np.int64)
    out_cond = np.empty(n_groups, dtype=np.float64)
    
    # Force GC before measurement
    gc.collect()
    
    # Baseline RSS
    rss_baseline = get_current_rss_mb()
    
    # Track RSS over iterations
    rss_samples = [rss_baseline]
    iteration_samples = [0]
    
    # Warmup
    for _ in range(3):
        run_kernel_iteration(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond
        )
    
    gc.collect()
    rss_after_warmup = get_current_rss_mb()
    
    # Main benchmark loop
    t0 = time.perf_counter()
    for i in range(n_iterations):
        run_kernel_iteration(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond
        )
        
        if (i + 1) % measure_interval == 0:
            rss_samples.append(get_current_rss_mb())
            iteration_samples.append(i + 1)
    
    elapsed = time.perf_counter() - t0
    
    # Final RSS
    rss_final = get_current_rss_mb()
    rss_samples.append(rss_final)
    iteration_samples.append(n_iterations)
    
    # Compute metrics
    rss_peak = max(rss_samples)
    rss_drift = rss_final - rss_after_warmup
    rss_drift_pct = (rss_drift / rss_after_warmup * 100) if rss_after_warmup > 0 else 0
    
    # Check for sawtooth pattern (high variance)
    rss_array = np.array(rss_samples[1:])  # Skip baseline
    rss_std = np.std(rss_array) if len(rss_array) > 1 else 0
    rss_mean = np.mean(rss_array)
    rss_cv = rss_std / rss_mean if rss_mean > 0 else 0  # Coefficient of variation
    
    return {
        'n_groups': n_groups,
        'rows_per_group': rows_per_group,
        'n_feat': n_feat,
        'n_iterations': n_iterations,
        'elapsed_sec': elapsed,
        'iterations_per_sec': n_iterations / elapsed,
        'rss_baseline_mb': rss_baseline,
        'rss_after_warmup_mb': rss_after_warmup,
        'rss_peak_mb': rss_peak,
        'rss_final_mb': rss_final,
        'rss_drift_mb': rss_drift,
        'rss_drift_pct': rss_drift_pct,
        'rss_std_mb': rss_std,
        'rss_cv': rss_cv,
        'rss_samples': rss_samples,
        'iteration_samples': iteration_samples,
    }


def benchmark_memory_allocation_tracking(n_groups, rows_per_group, n_feat, n_iterations):
    """
    Track Python-level allocations using tracemalloc.
    
    Returns dict with allocation metrics.
    """
    if not _TRACEMALLOC_AVAILABLE:
        return {'error': 'tracemalloc not available'}
    
    n_params = n_feat + 1
    
    # Generate data
    X_all, Y_all, W_all, offsets = generate_data(n_groups, rows_per_group, n_feat)
    
    # Pre-allocate outputs
    out_beta = np.empty((n_groups, n_params), dtype=np.float64)
    out_errors = np.empty((n_groups, n_params), dtype=np.float64)
    out_rms = np.empty(n_groups, dtype=np.float64)
    out_mad = np.empty(n_groups, dtype=np.float64)
    out_status = np.empty(n_groups, dtype=np.uint8)
    out_n_valid = np.empty(n_groups, dtype=np.int64)
    out_n_filtered = np.empty(n_groups, dtype=np.int64)
    out_cond = np.empty(n_groups, dtype=np.float64)
    
    # Warmup
    for _ in range(3):
        run_kernel_iteration(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond
        )
    
    gc.collect()
    
    # Start tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    # Run iterations
    for _ in range(n_iterations):
        run_kernel_iteration(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond
        )
    
    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Compare snapshots
    top_diffs = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    # Sum allocations
    total_diff_bytes = sum(stat.size_diff for stat in top_diffs)
    
    return {
        'n_iterations': n_iterations,
        'current_traced_mb': current / (1024 * 1024),
        'peak_traced_mb': peak / (1024 * 1024),
        'total_diff_bytes': total_diff_bytes,
        'total_diff_mb': total_diff_bytes / (1024 * 1024),
        'bytes_per_iteration': total_diff_bytes / n_iterations if n_iterations > 0 else 0,
    }


def run_memory_benchmark_suite(scenarios, n_iterations=50):
    """Run memory benchmark suite."""
    results = []
    
    print("\n" + "="*80)
    print("MEMORY BENCHMARK SUITE")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Iterations per scenario: {n_iterations}")
    print(f"tracemalloc available: {_TRACEMALLOC_AVAILABLE}")
    print(f"resource module available: {_RESOURCE_AVAILABLE}")
    
    for scenario in scenarios:
        name = scenario['name']
        n_groups = scenario['n_groups']
        rows_per_group = scenario['rows_per_group']
        n_feat = scenario['n_feat']
        
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"{'='*60}")
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows/group: {rows_per_group}")
        print(f"  Features: {n_feat}")
        print(f"  Total rows: {n_groups * rows_per_group:,}")
        
        # RSS tracking
        gc.collect()
        rss_result = benchmark_memory_single_run(
            n_groups, rows_per_group, n_feat, n_iterations
        )
        
        print(f"\n  RSS Metrics:")
        print(f"    Baseline: {rss_result['rss_baseline_mb']:.1f} MB")
        print(f"    After warmup: {rss_result['rss_after_warmup_mb']:.1f} MB")
        print(f"    Peak: {rss_result['rss_peak_mb']:.1f} MB")
        print(f"    Final: {rss_result['rss_final_mb']:.1f} MB")
        print(f"    Drift: {rss_result['rss_drift_mb']:+.2f} MB ({rss_result['rss_drift_pct']:+.1f}%)")
        print(f"    Std Dev: {rss_result['rss_std_mb']:.2f} MB (CV: {rss_result['rss_cv']:.3f})")
        print(f"  Performance: {rss_result['iterations_per_sec']:.1f} iter/sec")
        
        # tracemalloc tracking (optional)
        if _TRACEMALLOC_AVAILABLE:
            gc.collect()
            alloc_result = benchmark_memory_allocation_tracking(
                n_groups, rows_per_group, n_feat, min(n_iterations, 20)
            )
            if 'error' not in alloc_result:
                print(f"\n  Allocation Tracking:")
                print(f"    Peak traced: {alloc_result['peak_traced_mb']:.2f} MB")
                print(f"    Bytes/iteration: {alloc_result['bytes_per_iteration']:.0f}")
                rss_result['allocation_tracking'] = alloc_result
        
        # Evaluate gates
        gates = {
            'rss_drift_ok': abs(rss_result['rss_drift_pct']) < 5.0,  # < 5% drift
            'rss_cv_ok': rss_result['rss_cv'] < 0.1,  # Low variance (no sawtooth)
        }
        rss_result['gates'] = gates
        
        gate_status = "✓ PASS" if all(gates.values()) else "✗ FAIL"
        print(f"\n  Memory Gates: {gate_status}")
        for gate_name, gate_pass in gates.items():
            status = "✓" if gate_pass else "✗"
            print(f"    {status} {gate_name}")
        
        results.append({
            'name': name,
            **rss_result
        })
    
    return results


def print_memory_summary(results):
    """Print memory benchmark summary."""
    print("\n" + "="*80)
    print("MEMORY BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Scenario':<30} {'Peak RSS':>12} {'Drift':>12} {'CV':>10} {'Gates':>10}")
    print("-"*80)
    
    for r in results:
        peak_rss = f"{r['rss_peak_mb']:.1f} MB"
        drift = f"{r['rss_drift_pct']:+.1f}%"
        cv = f"{r['rss_cv']:.3f}"
        gates_pass = all(r.get('gates', {}).values())
        gates_str = "✓ PASS" if gates_pass else "✗ FAIL"
        
        print(f"{r['name']:<30} {peak_rss:>12} {drift:>12} {cv:>10} {gates_str:>10}")
    
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Memory benchmark for groupby regression kernels')
    parser.add_argument('--iterations', type=int, default=50, help='Iterations per scenario')
    parser.add_argument('--json', type=str, help='Output results to JSON file')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark')
    args = parser.parse_args()
    
    if not _NUMBA_AVAILABLE:
        print("ERROR: Numba not available")
        sys.exit(1)
    
    # Define scenarios
    if args.quick:
        scenarios = [
            {'name': 'Small', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium', 'n_groups': 2000, 'rows_per_group': 30, 'n_feat': 2},
        ]
        n_iterations = 20
    else:
        scenarios = [
            {'name': 'Small: 1K groups', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium: 5K groups', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': 'Large: 10K groups', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Wide: 1K, 8 feat', 'n_groups': 1000, 'rows_per_group': 50, 'n_feat': 8},
        ]
        n_iterations = args.iterations
    
    # Run benchmarks
    results = run_memory_benchmark_suite(scenarios, n_iterations)
    
    # Print summary
    print_memory_summary(results)
    
    # Check gates
    print("\n" + "="*80)
    print("MEMORY GATES")
    print("="*80)
    
    all_pass = True
    
    # Gate 1: RSS drift < 5%
    max_drift = max(abs(r['rss_drift_pct']) for r in results)
    drift_pass = max_drift < 5.0
    print(f"\n1. RSS Drift: {max_drift:.1f}% (max) {'✓ PASS' if drift_pass else '✗ FAIL'} (required: <5%)")
    if not drift_pass:
        all_pass = False
    
    # Gate 2: No sawtooth (CV < 0.1)
    max_cv = max(r['rss_cv'] for r in results)
    cv_pass = max_cv < 0.1
    print(f"2. RSS Stability: CV={max_cv:.3f} (max) {'✓ PASS' if cv_pass else '✗ FAIL'} (required: CV<0.1)")
    if not cv_pass:
        all_pass = False
    
    # Save to JSON
    if args.json:
        # Recursively convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'n_iterations': n_iterations,
            'results': [convert_for_json(r) for r in results],
            'gates': {
                'rss_drift_max_pct': float(max_drift),
                'rss_drift_pass': bool(drift_pass),
                'rss_cv_max': float(max_cv),
                'rss_cv_pass': bool(cv_pass),
                'all_pass': bool(all_pass),
            }
        }
        
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")
    
    print("\n" + "="*80)
    if all_pass:
        print("ALL MEMORY GATES PASSED ✓")
    else:
        print("SOME MEMORY GATES FAILED ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()
