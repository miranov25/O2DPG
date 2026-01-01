#!/usr/bin/env python
"""
Phase 12.14a.GB + 12.14b.GB: Memory Tracking Benchmark Suite

Measures memory consumption, RSS drift, and fragmentation.
Includes Benchmark Framework integration.

Usage:
    python bench_groupby_regression_memory.py
    python bench_groupby_regression_memory.py --quick
    python bench_groupby_regression_memory.py --json results.json

Author: Team 3 Coder
Date: 2025-12-31
Phase: 12.14a.GB (original), 12.14b.GB (BF integration)
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
            fit_groups_single_numba, INVALID_DETECT, _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

if not _import_success:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from groupby_regression_kernels import (
            fit_groups_single_numba, INVALID_DETECT, _NUMBA_AVAILABLE,
        )
        _import_success = True
    except ImportError:
        pass

if not _import_success:
    print("ERROR: Could not import groupby_regression_kernels")
    sys.exit(1)


def get_current_rss_mb():
    """Get current RSS in MB."""
    if sys.platform == 'darwin':
        import subprocess
        try:
            pid = os.getpid()
            result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], 
                                    capture_output=True, text=True)
            return int(result.stdout.strip()) / 1024
        except:
            return 0.0
    else:
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except:
            pass
        return 0.0


def generate_data(n_groups, rows_per_group, n_feat, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group
    X_all = np.random.randn(n_rows, n_feat).astype(np.float64)
    Y_all = np.random.randn(n_rows).astype(np.float64)
    W_all = np.empty(0, dtype=np.float64)
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)
    return X_all, Y_all, W_all, offsets


def _allocate_single_outputs(n_groups, n_params):
    """Pre-allocate output arrays."""
    return (
        np.empty((n_groups, n_params), dtype=np.float64),
        np.empty((n_groups, n_params), dtype=np.float64),
        np.empty(n_groups, dtype=np.float64),
        np.empty(n_groups, dtype=np.float64),
        np.empty(n_groups, dtype=np.uint8),
        np.empty(n_groups, dtype=np.int64),
        np.empty(n_groups, dtype=np.int64),
        np.empty(n_groups, dtype=np.float64),
    )


def benchmark_memory_single_run(n_groups, rows_per_group, n_feat, n_iterations, measure_interval=10):
    """Run kernel multiple times and track memory."""
    n_params = n_feat + 1
    X_all, Y_all, W_all, offsets = generate_data(n_groups, rows_per_group, n_feat)
    out = _allocate_single_outputs(n_groups, n_params)
    
    gc.collect()
    rss_baseline = get_current_rss_mb()
    rss_samples = [rss_baseline]
    
    # Warmup
    for _ in range(3):
        fit_groups_single_numba(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                                True, 5, False, INVALID_DETECT, *out)
    
    gc.collect()
    rss_after_warmup = get_current_rss_mb()
    
    t0 = time.perf_counter()
    for i in range(n_iterations):
        fit_groups_single_numba(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                                True, 5, False, INVALID_DETECT, *out)
        if (i + 1) % measure_interval == 0:
            rss_samples.append(get_current_rss_mb())
    elapsed = time.perf_counter() - t0
    
    rss_final = get_current_rss_mb()
    rss_samples.append(rss_final)
    rss_peak = max(rss_samples)
    rss_drift = rss_final - rss_after_warmup
    rss_drift_pct = (rss_drift / rss_after_warmup * 100) if rss_after_warmup > 0 else 0
    rss_array = np.array(rss_samples[1:])
    rss_std = np.std(rss_array) if len(rss_array) > 1 else 0
    rss_mean = np.mean(rss_array)
    rss_cv = rss_std / rss_mean if rss_mean > 0 else 0
    
    return {
        'n_groups': n_groups, 'rows_per_group': rows_per_group, 'n_feat': n_feat,
        'n_iterations': n_iterations, 'elapsed_sec': elapsed,
        'iterations_per_sec': n_iterations / elapsed,
        'rss_baseline_mb': rss_baseline, 'rss_after_warmup_mb': rss_after_warmup,
        'rss_peak_mb': rss_peak, 'rss_final_mb': rss_final,
        'rss_drift_mb': rss_drift, 'rss_drift_pct': rss_drift_pct,
        'rss_std_mb': rss_std, 'rss_cv': rss_cv,
    }


def run_memory_benchmark_suite(scenarios, n_iterations=50):
    """Run memory benchmark suite."""
    results = []
    print("\n" + "="*80)
    print("MEMORY BENCHMARK SUITE")
    print("="*80)
    
    for scenario in scenarios:
        name = scenario['name']
        gc.collect()
        result = benchmark_memory_single_run(
            scenario['n_groups'], scenario['rows_per_group'], 
            scenario['n_feat'], n_iterations
        )
        result['gates'] = {
            'rss_drift_ok': abs(result['rss_drift_pct']) < 5.0,
            'rss_cv_ok': result['rss_cv'] < 0.1,
        }
        print(f"\n{name}: drift={result['rss_drift_pct']:+.1f}%, CV={result['rss_cv']:.3f}")
        results.append({'name': name, **result})
    return results


# =============================================================================
# BF INTEGRATION (Phase 12.14b.GB)
# =============================================================================

MEMORY_SCENARIOS = {
    "M1": {"n_groups": 500,  "rows_per_group": 20, "n_feat": 2, "description": "Small"},
    "M2": {"n_groups": 2000, "rows_per_group": 30, "n_feat": 2, "description": "Medium"},
    "M3": {"n_groups": 5000, "rows_per_group": 20, "n_feat": 2, "description": "Large"},
    "M4": {"n_groups": 1000, "rows_per_group": 50, "n_feat": 8, "description": "Wide"},
}

QUICK_MEMORY_SCENARIOS = ["M1", "M2"]
RELEASE_MEMORY_SCENARIOS = ["M1", "M2", "M3", "M4"]


def bench_memory_rss(scenario: str = "M1", n_iterations: int = 50, seed: int = 42, **kwargs) -> dict:
    """BF-compatible memory benchmark."""
    if scenario not in MEMORY_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    scen = MEMORY_SCENARIOS[scenario]
    
    result = benchmark_memory_single_run(
        scen["n_groups"], scen["rows_per_group"], scen["n_feat"], n_iterations
    )
    
    drift_ok = abs(result['rss_drift_pct']) < 5.0
    cv_ok = result['rss_cv'] < 0.1
    status = "OK" if (drift_ok and cv_ok) else "FAILED"
    
    return {
        "time_s": result['elapsed_sec'], "time_std_s": 0.0, "n_runs": 1, "status": status,
        "n_groups": scen["n_groups"], "rows_per_group": scen["rows_per_group"],
        "n_feat": scen["n_feat"], "n_iterations": n_iterations,
        "rss_baseline_mb": result['rss_baseline_mb'],
        "rss_peak_mb": result['rss_peak_mb'],
        "rss_drift_pct": result['rss_drift_pct'],
        "rss_cv": result['rss_cv'],
        "iterations_per_sec": result['iterations_per_sec'],
        "drift_gate_pass": drift_ok, "cv_gate_pass": cv_ok,
    }


def get_benchmarks(suite: str = "quick") -> list:
    """
    BF discovery function for memory benchmarks.
    
    Phase 12.14b.GB-addendum: Added uses_n_jobs: False for ID hygiene.
    """
    scenarios = QUICK_MEMORY_SCENARIOS if suite == "quick" else RELEASE_MEMORY_SCENARIOS
    n_iter = 20 if suite == "quick" else 50
    return [{
        "name": "memory_rss_tracking",
        "func": bench_memory_rss,
        "scenarios": scenarios,
        "uses_n_jobs": False,  # Phase 12.14b.GB-addendum: ID hygiene
        "params": {"n_iterations": n_iter},
    }]


# =============================================================================
# STANDALONE CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Memory benchmark')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--json', type=str)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if not _NUMBA_AVAILABLE:
        print("ERROR: Numba not available")
        sys.exit(1)
    
    if args.quick:
        scenarios = [
            {'name': 'Small', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium', 'n_groups': 2000, 'rows_per_group': 30, 'n_feat': 2},
        ]
        n_iter = 20
    else:
        scenarios = [
            {'name': 'Small', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': 'Large', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
        ]
        n_iter = args.iterations
    
    results = run_memory_benchmark_suite(scenarios, n_iter)
    
    all_pass = all(all(r.get('gates', {}).values()) for r in results)
    print("\n" + "="*80)
    print("ALL MEMORY GATES PASSED ✓" if all_pass else "SOME GATES FAILED ✗")
    
    if args.json:
        def conv(o):
            if isinstance(o, (np.bool_, bool)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list): return [conv(x) for x in o]
            return o
        with open(args.json, 'w') as f:
            json.dump({'results': [conv(r) for r in results], 'all_pass': bool(all_pass)}, f, indent=2)
        print(f"Saved: {args.json}")
    
    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
