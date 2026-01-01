#!/usr/bin/env python
"""
Phase 12.14.GB + 12.14b.GB: Kernel Performance Benchmark Suite

Standalone benchmark for groupby regression kernels with Benchmark Framework integration.

Usage:
    # Standalone execution
    python bench_groupby_regression_kernels.py
    python bench_groupby_regression_kernels.py --quick     # Fast mode
    python bench_groupby_regression_kernels.py --full      # Extended benchmarks

    # Via BF runner
    python -m dfextensions.benchmarks.runner --subproject groupby_regression

Author: Team 3 Coder
Date: 2025-12-31
Phase: 12.14.GB (original), 12.14b.GB (BF integration)
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


def generate_data(n_groups, rows_per_group, n_feat, n_targets=1, seed=42, 
                  return_true_coeffs=False, noise_std=0.1):
    """
    Generate test data for benchmarking with known true coefficients.
    
    Y = intercept + X @ slopes + noise
    
    True coefficients: intercept=1.0, slopes=[2.0, 3.0, 4.0, ...]
    """
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group
    
    X_all = np.random.randn(n_rows, n_feat)
    
    true_intercept = 1.0
    true_slopes = np.arange(2.0, 2.0 + n_feat)
    true_coeffs = np.concatenate([[true_intercept], true_slopes])
    
    Y_base = true_intercept + X_all @ true_slopes
    
    if n_targets == 1:
        Y_all = Y_base + np.random.randn(n_rows) * noise_std
    else:
        Y_all = np.column_stack([
            Y_base * (1.0 + 0.1 * t) + np.random.randn(n_rows) * noise_std
            for t in range(n_targets)
        ])
        true_coeffs = np.stack([
            true_coeffs * (1.0 + 0.1 * t) for t in range(n_targets)
        ])
    
    W_all = np.empty(0, dtype=np.float64)
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)
    
    if return_true_coeffs:
        return X_all, Y_all, W_all, offsets, true_coeffs
    return X_all, Y_all, W_all, offsets


def validate_correctness(out_beta, true_coeffs, out_status, rows_per_group=30, 
                         noise_std=0.1):
    """Validate fitted coefficients against true values."""
    from groupby_regression_kernels import STATUS_OK
    
    safety_factor = 5.0
    expected_std = noise_std / np.sqrt(rows_per_group)
    rtol = max(expected_std * safety_factor, 0.05)
    atol = rtol * 0.1
    
    if out_beta.ndim == 2:
        ok_mask = out_status == STATUS_OK
        beta_ok = out_beta[ok_mask]
        true_broadcast = true_coeffs
    else:
        ok_mask = np.all(out_status == STATUS_OK, axis=1)
        beta_ok = out_beta[ok_mask]
        true_broadcast = true_coeffs
    
    if len(beta_ok) == 0:
        return {'passed': False, 'max_error': np.nan, 'mean_error': np.nan, 
                'n_checked': 0, 'message': 'No OK fits to validate'}
    
    errors = np.abs(beta_ok - true_broadcast)
    rel_errors = errors / (np.abs(true_broadcast) + 1e-10)
    
    max_error = np.max(rel_errors)
    mean_error = np.mean(rel_errors)
    passed = np.allclose(beta_ok, true_broadcast, rtol=rtol, atol=atol)
    
    return {
        'passed': passed,
        'max_error': max_error,
        'mean_error': mean_error,
        'n_checked': len(beta_ok),
        'rtol_used': rtol,
        'message': 'OK' if passed else f'Max rel error {max_error:.4f} > rtol {rtol:.3f}'
    }


def _allocate_single_outputs(n_groups, n_params):
    """Pre-allocate output arrays for single-fit kernel."""
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


def _allocate_multi_outputs(n_groups, n_targets, n_params):
    """Pre-allocate output arrays for multi-fit kernel."""
    return (
        np.empty((n_groups, n_targets, n_params), dtype=np.float64),
        np.empty((n_groups, n_targets, n_params), dtype=np.float64),
        np.empty((n_groups, n_targets), dtype=np.float64),
        np.empty((n_groups, n_targets), dtype=np.float64),
        np.empty((n_groups, n_targets), dtype=np.uint8),
        np.empty(n_groups, dtype=np.int64),
        np.empty(n_groups, dtype=np.int64),
        np.empty(n_groups, dtype=np.float64),
    )


# =============================================================================
# TIMING HELPERS (P0-A: No wrapper nesting)
# =============================================================================

def _time_single_kernel_core(
    X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
    out_beta, out_errors, out_rms, out_mad,
    out_status, out_n_valid, out_n_filtered, out_cond,
    n_runs: int = 5,
) -> tuple:
    """
    Time single-fit kernel ONLY (no data gen, no validation).
    
    TIMING CONTRACT (P0-B):
    - INCLUDED: fit_groups_single_numba() call
    - EXCLUDED: Data generation, output allocation, validation, warmup
    """
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
    
    return np.mean(times), np.std(times)


def _time_multi_kernel_core(
    X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
    out_beta, out_errors, out_rms, out_mad,
    out_status, out_n_valid, out_n_filtered, out_cond,
    n_runs: int = 5,
) -> tuple:
    """
    Time multi-fit kernel ONLY (no data gen, no validation).
    """
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
    
    return np.mean(times), np.std(times)


def _time_numpy_baseline(X_all, Y_all, offsets, n_groups, n_runs: int = 3) -> tuple:
    """Time NumPy lstsq baseline."""
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
    
    return np.mean(times), np.std(times)


# =============================================================================
# ORIGINAL BENCHMARK FUNCTIONS (preserved for backward compatibility)
# =============================================================================

def benchmark_single_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                         n_runs=5, warmup=2, return_outputs=False):
    """Benchmark single-fit kernel."""
    out_arrays = _allocate_single_outputs(n_groups, n_params)
    out_beta, out_errors, out_rms, out_mad, out_status, out_n_valid, out_n_filtered, out_cond = out_arrays
    
    for _ in range(warmup):
        fit_groups_single_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    median_time, std_time = _time_single_kernel_core(
        X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
        n_runs=n_runs,
    )
    
    if return_outputs:
        return median_time, std_time, out_beta, out_status
    return median_time, std_time


def benchmark_multi_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
                        n_runs=5, warmup=2, return_outputs=False):
    """Benchmark multi-fit kernel."""
    out_arrays = _allocate_multi_outputs(n_groups, n_targets, n_params)
    out_beta, out_errors, out_rms, out_mad, out_status, out_n_valid, out_n_filtered, out_cond = out_arrays
    
    for _ in range(warmup):
        fit_groups_multifit_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_targets, n_params,
            True, 5, False,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    median_time, std_time = _time_multi_kernel_core(
        X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
        n_runs=n_runs,
    )
    
    if return_outputs:
        return median_time, std_time, out_beta, out_status
    return median_time, std_time


def benchmark_numpy_baseline(X_all, Y_all, offsets, n_groups, n_runs=3):
    """Benchmark NumPy lstsq baseline."""
    return _time_numpy_baseline(X_all, Y_all, offsets, n_groups, n_runs)


def run_benchmark_suite(scenarios, n_runs=5, validate=True):
    """Run complete benchmark suite with correctness validation."""
    results = []
    correctness_failures = []
    
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
        print(f"  Groups: {n_groups:,}, Rows/group: {rows_per_group}, Features: {n_feat}, Targets: {n_targets}")
        
        data = generate_data(
            n_groups, rows_per_group, n_feat, n_targets,
            return_true_coeffs=validate, noise_std=0.1
        )
        if validate:
            X_all, Y_all, W_all, offsets, true_coeffs = data
        else:
            X_all, Y_all, W_all, offsets = data
            true_coeffs = None
        
        result = {
            'name': name, 'n_groups': n_groups, 'rows_per_group': rows_per_group,
            'n_feat': n_feat, 'n_targets': n_targets, 'n_params': n_params,
        }
        
        # NumPy baseline
        numpy_time, numpy_std = benchmark_numpy_baseline(
            X_all, Y_all if n_targets == 1 else Y_all[:, 0:1], offsets, n_groups
        )
        result['numpy_time_ms'] = numpy_time * 1000
        result['numpy_groups_per_sec'] = n_groups / numpy_time
        print(f"  NumPy: {numpy_time*1000:.2f} ms ({n_groups/numpy_time:,.0f} groups/sec)")
        
        # Single-fit kernel
        Y_1d = Y_all if n_targets == 1 else Y_all[:, 0]
        true_single = true_coeffs if n_targets == 1 else (true_coeffs[0] if validate else None)
        
        single_result = benchmark_single_fit(
            X_all, Y_1d, W_all, offsets, n_groups, n_feat, n_params, n_runs,
            return_outputs=validate
        )
        
        if validate:
            single_time, single_std, out_beta, out_status = single_result
            validation = validate_correctness(out_beta, true_single, out_status, rows_per_group)
            result['single_correctness'] = validation['passed']
            result['single_max_error'] = validation['max_error']
            if not validation['passed']:
                correctness_failures.append(f"{name} (single): {validation['message']}")
                print(f"  ⚠ Correctness: FAIL")
            else:
                print(f"  ✓ Correctness: PASS (max error: {validation['max_error']:.2e})")
        else:
            single_time, single_std = single_result
        
        result['single_time_ms'] = single_time * 1000
        result['single_groups_per_sec'] = n_groups / single_time
        result['numba_vs_numpy_speedup'] = numpy_time / single_time
        print(f"  Numba: {single_time*1000:.2f} ms → {numpy_time/single_time:.1f}× vs NumPy")
        
        # Multi-fit kernel (if multiple targets)
        if n_targets > 1:
            multi_result = benchmark_multi_fit(
                X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params, n_runs,
                return_outputs=validate
            )
            
            if validate:
                multi_time, multi_std, out_beta_m, out_status_m = multi_result
                validation = validate_correctness(out_beta_m, true_coeffs, out_status_m, rows_per_group)
                result['multi_correctness'] = validation['passed']
                if not validation['passed']:
                    correctness_failures.append(f"{name} (multi): {validation['message']}")
            else:
                multi_time, multi_std = multi_result
            
            single_total = single_time * n_targets
            result['multi_time_ms'] = multi_time * 1000
            result['multi_groups_per_sec'] = n_groups / multi_time
            result['multi_vs_single_speedup'] = single_total / multi_time
            print(f"  Multi: {multi_time*1000:.2f} ms → {single_total/multi_time:.2f}× vs Single×{n_targets}")
        
        results.append(result)
    
    return results, correctness_failures


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Scenario':<30} {'Groups/sec':>15} {'Numba/NumPy':>12} {'Multi/Single':>12}")
    print("-"*80)
    
    for r in results:
        groups_per_sec = r.get('single_groups_per_sec', 0)
        numba_speedup = r.get('numba_vs_numpy_speedup', 0)
        multi_speedup = r.get('multi_vs_single_speedup', '-')
        multi_str = f"{multi_speedup:.2f}×" if isinstance(multi_speedup, float) else multi_speedup
        print(f"{r['name']:<30} {groups_per_sec:>15,.0f} {numba_speedup:>11.1f}× {multi_str:>12}")
    
    print("-"*80)


# =============================================================================
# BF INTEGRATION (Phase 12.14b.GB)
# =============================================================================

KERNEL_SCENARIOS = {
    "K1": {"n_groups": 500,    "rows_per_group": 20, "n_feat": 2, "description": "Quick"},
    "K2": {"n_groups": 1000,   "rows_per_group": 20, "n_feat": 2, "description": "Small"},
    "K3": {"n_groups": 5000,   "rows_per_group": 30, "n_feat": 2, "description": "Medium"},
    "K4": {"n_groups": 10000,  "rows_per_group": 20, "n_feat": 2, "description": "Large"},
    "K5": {"n_groups": 5000,   "rows_per_group": 50, "n_feat": 4, "description": "Wide"},
    "K6": {"n_groups": 100000, "rows_per_group": 20, "n_feat": 2, "description": "Stress"},
}

QUICK_KERNEL_SCENARIOS = ["K1", "K2"]
RELEASE_KERNEL_SCENARIOS = ["K1", "K2", "K3", "K4", "K5", "K6"]

SUITE_PARAMS = {
    "quick":   {"warmup": 1, "n_runs": 3, "validate": True},
    "release": {"warmup": 2, "n_runs": 7, "validate": True},
}


def _get_kernel_scenario(scenario: str) -> dict:
    """Get scenario parameters by name."""
    if scenario not in KERNEL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return KERNEL_SCENARIOS[scenario].copy()


def bench_kernel_single_fit(
    scenario: str = "K1",
    n_runs: int = 5,
    warmup: int = 2,
    warmup_runs: int = None,  # BF alias for warmup
    validate: bool = True,
    seed: int = 42,
    n_jobs: int = None,  # Ignored - kernel benchmarks don't use parallelism
    **kwargs,
) -> dict:
    """
    BF-compatible benchmark for single-fit Numba kernel.
    
    TIMING CONTRACT (P0-B):
    - time_s measures KERNEL EXECUTION ONLY
    """
    # Handle BF's warmup_runs vs our warmup
    if warmup_runs is not None:
        warmup = warmup_runs
    
    scen = _get_kernel_scenario(scenario)
    n_groups = scen["n_groups"]
    rows_per_group = scen["rows_per_group"]
    n_feat = scen["n_feat"]
    n_params = n_feat + 1
    
    # Setup (NOT timed)
    data = generate_data(n_groups, rows_per_group, n_feat, n_targets=1,
                         return_true_coeffs=validate, noise_std=0.1, seed=seed)
    if validate:
        X_all, Y_all, W_all, offsets, true_coeffs = data
    else:
        X_all, Y_all, W_all, offsets = data
        true_coeffs = None
    
    out_arrays = _allocate_single_outputs(n_groups, n_params)
    out_beta, out_errors, out_rms, out_mad, out_status, out_n_valid, out_n_filtered, out_cond = out_arrays
    
    # Warmup (NOT timed)
    for _ in range(warmup):
        fit_groups_single_numba(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    # Timed runs (ONLY this is time_s)
    mean_time, std_time = _time_single_kernel_core(
        X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
        n_runs=n_runs,
    )
    
    # NumPy baseline (separate timing)
    numpy_mean, numpy_std = _time_numpy_baseline(X_all, Y_all, offsets, n_groups, n_runs=3)
    
    # Validation (NOT timed)
    correctness_ok = True
    if validate:
        validation = validate_correctness(out_beta, true_coeffs, out_status, rows_per_group)
        correctness_ok = validation['passed']
    
    speedup = numpy_mean / mean_time if mean_time > 0 else 0
    throughput = n_groups / mean_time if mean_time > 0 else 0
    speedup_gate_pass = speedup >= 3.0
    status = "OK" if (correctness_ok and speedup_gate_pass) else "FAILED"
    
    return {
        "time_s": mean_time,
        "time_std_s": std_time,
        "n_runs": n_runs,
        "status": status,
        "n_groups": n_groups,
        "rows_per_group": rows_per_group,
        "n_feat": n_feat,
        "n_params": n_params,
        "numba_time_s": mean_time,
        "numpy_time_s": numpy_mean,
        "speedup_vs_numpy": speedup,
        "throughput_groups_per_sec": throughput,
        "correctness_validated": validate,
        "correctness_passed": correctness_ok,
        "speedup_gate_threshold": 3.0,
        "speedup_gate_pass": speedup_gate_pass,
    }


def bench_kernel_multi_fit(
    scenario: str = "K1",
    n_targets: int = 6,
    n_runs: int = 5,
    warmup: int = 2,
    warmup_runs: int = None,  # BF alias for warmup
    validate: bool = True,
    seed: int = 42,
    n_jobs: int = None,  # Ignored - kernel benchmarks don't use parallelism
    **kwargs,
) -> dict:
    """
    BF-compatible benchmark for multi-fit Numba kernel.
    
    P0-A: Uses timing helpers, NOT wrapper calls for speedup comparison.
    """
    # Handle BF's warmup_runs vs our warmup
    if warmup_runs is not None:
        warmup = warmup_runs
    
    scen = _get_kernel_scenario(scenario)
    n_groups = scen["n_groups"]
    rows_per_group = scen["rows_per_group"]
    n_feat = scen["n_feat"]
    n_params = n_feat + 1
    
    # Setup
    data = generate_data(n_groups, rows_per_group, n_feat, n_targets=n_targets,
                         return_true_coeffs=validate, noise_std=0.1, seed=seed)
    if validate:
        X_all, Y_all, W_all, offsets, true_coeffs = data
    else:
        X_all, Y_all, W_all, offsets = data
        true_coeffs = None
    
    multi_out = _allocate_multi_outputs(n_groups, n_targets, n_params)
    out_beta, out_errors, out_rms, out_mad, out_status, out_n_valid, out_n_filtered, out_cond = multi_out
    
    single_out = _allocate_single_outputs(n_groups, n_params)
    
    # Warmup multi-fit
    for _ in range(warmup):
        fit_groups_multifit_numba(
            X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
            True, 5, False,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
    
    # Timed runs for multi-fit
    mean_time, std_time = _time_multi_kernel_core(
        X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
        n_runs=n_runs,
    )
    
    # Single-fit comparison (P0-A: use timing helper, NOT wrapper)
    Y_single = Y_all[:, 0]
    s_beta, s_errors, s_rms, s_mad, s_status, s_n_valid, s_n_filtered, s_cond = single_out
    for _ in range(warmup):
        fit_groups_single_numba(
            X_all, Y_single, W_all, offsets, n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT,
            s_beta, s_errors, s_rms, s_mad,
            s_status, s_n_valid, s_n_filtered, s_cond,
        )
    
    single_mean, single_std = _time_single_kernel_core(
        X_all, Y_single, W_all, offsets, n_groups, n_feat, n_params,
        s_beta, s_errors, s_rms, s_mad,
        s_status, s_n_valid, s_n_filtered, s_cond,
        n_runs=n_runs,
    )
    
    # Validation
    correctness_ok = True
    if validate:
        validation = validate_correctness(out_beta, true_coeffs, out_status, rows_per_group)
        correctness_ok = validation['passed']
    
    single_total = single_mean * n_targets
    speedup_vs_single = single_total / mean_time if mean_time > 0 else 0
    throughput = n_groups / mean_time if mean_time > 0 else 0
    status = "OK" if correctness_ok else "FAILED"
    
    return {
        "time_s": mean_time,
        "time_std_s": std_time,
        "n_runs": n_runs,
        "status": status,
        "n_groups": n_groups,
        "rows_per_group": rows_per_group,
        "n_feat": n_feat,
        "n_params": n_params,
        "n_targets": n_targets,
        "multi_time_s": mean_time,
        "single_time_s": single_mean,
        "single_total_s": single_total,
        "speedup_vs_single_x_n": speedup_vs_single,
        "throughput_groups_per_sec": throughput,
        "correctness_validated": validate,
        "correctness_passed": correctness_ok,
    }


def get_benchmarks(suite: str = "quick") -> list:
    """
    BF discovery function for kernel benchmarks.
    
    P0-C: This function is PURE and CHEAP.
    - No Numba JIT compilation
    - No large array allocations
    """
    scenarios = QUICK_KERNEL_SCENARIOS if suite == "quick" else RELEASE_KERNEL_SCENARIOS
    params = SUITE_PARAMS.get(suite, SUITE_PARAMS["quick"])
    
    return [
        {
            "name": "kernel_single_fit",
            "func": bench_kernel_single_fit,
            "scenarios": scenarios,
            "params": {
                "validate": params["validate"],
            },
        },
        {
            "name": "kernel_multi_fit",
            "func": bench_kernel_multi_fit,
            "scenarios": scenarios,
            "params": {
                "validate": params["validate"],
                "n_targets": 6,
            },
        },
    ]


# =============================================================================
# STANDALONE CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark groupby regression kernels')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark')
    parser.add_argument('--full', action='store_true', help='Full benchmark')
    parser.add_argument('--json', type=str, help='Output results to JSON file')
    args = parser.parse_args()
    
    if not _NUMBA_AVAILABLE:
        print("ERROR: Numba not available")
        sys.exit(1)
    
    print("="*80)
    print("PHASE 12.14.GB: KERNEL PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    
    if args.quick:
        scenarios = [
            {'name': 'Quick: 500 groups', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Quick: 500 groups, 6 targets', 'n_groups': 500, 'rows_per_group': 20, 'n_feat': 2, 'n_targets': 6},
        ]
        n_runs = 3
    elif args.full:
        scenarios = [
            {'name': '1K groups', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '10K groups', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '50K groups', 'n_groups': 50000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '5K groups, 6 targets', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 6},
            {'name': '5K groups, 4 feat, 6 tgt', 'n_groups': 5000, 'rows_per_group': 50, 'n_feat': 4, 'n_targets': 6},
        ]
        n_runs = 7
    else:
        scenarios = [
            {'name': 'Small: 1K groups', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Medium: 5K groups', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': 'Large: 10K groups', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': 'Multi: 1K, 6 targets', 'n_groups': 1000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 6},
        ]
        n_runs = 5
    
    results, failures = run_benchmark_suite(scenarios, n_runs, validate=True)
    print_summary(results)
    
    # Gates
    print("\n" + "="*80)
    print("GATES")
    print("="*80)
    
    all_pass = len(failures) == 0
    print(f"Correctness: {'✓ PASS' if not failures else '✗ FAIL'}")
    
    typical = [r for r in results if r['rows_per_group'] <= 50]
    if typical:
        min_speedup = min(r['numba_vs_numpy_speedup'] for r in typical)
        gate1 = min_speedup >= 5.0
        print(f"Numba vs NumPy: {min_speedup:.1f}× {'✓ PASS' if gate1 else '✗ FAIL'} (≥5×)")
        all_pass = all_pass and gate1
    
    if args.json:
        def convert(obj):
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, (np.integer, int)): return int(obj)
            if isinstance(obj, (np.floating, float)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(x) for x in obj]
            return obj
        
        with open(args.json, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 
                       'results': [convert(r) for r in results],
                       'gates': {'all_pass': bool(all_pass)}}, f, indent=2)
        print(f"\nSaved: {args.json}")
    
    print("\n" + "="*80)
    print("ALL GATES PASSED ✓" if all_pass else "SOME GATES FAILED ✗")
    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
