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


def generate_data(n_groups, rows_per_group, n_feat, n_targets=1, seed=42, 
                  return_true_coeffs=False, noise_std=0.1):
    """
    Generate test data for benchmarking with known true coefficients.
    
    Y = intercept + X @ slopes + noise
    
    True coefficients: intercept=1.0, slopes=[2.0, 3.0, 4.0, ...]
    
    Parameters
    ----------
    return_true_coeffs : bool
        If True, return true coefficients for correctness validation
    noise_std : float
        Standard deviation of noise (0 for exact fit)
    
    Returns
    -------
    X_all, Y_all, W_all, offsets, [true_coeffs]
    """
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group
    
    X_all = np.random.randn(n_rows, n_feat)
    
    # True coefficients: intercept=1, slopes=[2, 3, 4, ...]
    true_intercept = 1.0
    true_slopes = np.arange(2.0, 2.0 + n_feat)
    true_coeffs = np.concatenate([[true_intercept], true_slopes])
    
    # Generate Y with known relationship
    Y_base = true_intercept + X_all @ true_slopes
    
    if n_targets == 1:
        Y_all = Y_base + np.random.randn(n_rows) * noise_std
    else:
        # Multiple targets with scaling factors
        Y_all = np.column_stack([
            Y_base * (1.0 + 0.1 * t) + np.random.randn(n_rows) * noise_std
            for t in range(n_targets)
        ])
        # True coeffs per target
        true_coeffs = np.stack([
            true_coeffs * (1.0 + 0.1 * t) for t in range(n_targets)
        ])
    
    W_all = np.empty(0, dtype=np.float64)  # Unweighted
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)
    
    if return_true_coeffs:
        return X_all, Y_all, W_all, offsets, true_coeffs
    return X_all, Y_all, W_all, offsets


def validate_correctness(out_beta, true_coeffs, out_status, rows_per_group=30, 
                         noise_std=0.1):
    """
    Validate fitted coefficients against true values.
    
    Uses adaptive tolerance based on sample size and noise level.
    Statistical tolerance: roughly noise_std / sqrt(rows_per_group) * safety_factor
    
    Parameters
    ----------
    out_beta : ndarray
        Fitted coefficients (n_groups, n_params) or (n_groups, n_targets, n_params)
    true_coeffs : ndarray
        True coefficients (n_params,) or (n_targets, n_params)
    out_status : ndarray
        Status array to filter OK fits
    rows_per_group : int
        Rows per group (affects expected variance)
    noise_std : float
        Noise standard deviation used in data generation
    
    Returns
    -------
    dict with 'passed', 'max_error', 'mean_error', 'n_checked'
    """
    from groupby_regression_kernels import STATUS_OK
    
    # Adaptive tolerance: higher for fewer rows, lower noise
    # Formula: noise_std / sqrt(rows) * safety_factor
    # safety_factor accounts for finite sample effects
    safety_factor = 5.0
    expected_std = noise_std / np.sqrt(rows_per_group)
    rtol = max(expected_std * safety_factor, 0.05)  # At least 5%
    atol = rtol * 0.1
    
    # Filter to OK fits only
    if out_beta.ndim == 2:
        # Single target: (n_groups, n_params)
        ok_mask = out_status == STATUS_OK
        beta_ok = out_beta[ok_mask]
        true_broadcast = true_coeffs  # (n_params,)
    else:
        # Multi target: (n_groups, n_targets, n_params)
        ok_mask = np.all(out_status == STATUS_OK, axis=1)
        beta_ok = out_beta[ok_mask]  # (n_ok, n_targets, n_params)
        true_broadcast = true_coeffs  # (n_targets, n_params)
    
    if len(beta_ok) == 0:
        return {'passed': False, 'max_error': np.nan, 'mean_error': np.nan, 
                'n_checked': 0, 'message': 'No OK fits to validate'}
    
    # Compute errors
    errors = np.abs(beta_ok - true_broadcast)
    rel_errors = errors / (np.abs(true_broadcast) + 1e-10)
    
    max_error = np.max(rel_errors)
    mean_error = np.mean(rel_errors)
    
    # Check tolerance
    passed = np.allclose(beta_ok, true_broadcast, rtol=rtol, atol=atol)
    
    return {
        'passed': passed,
        'max_error': max_error,
        'mean_error': mean_error,
        'n_checked': len(beta_ok),
        'rtol_used': rtol,
        'message': 'OK' if passed else f'Max rel error {max_error:.4f} > rtol {rtol:.3f}'
    }


def benchmark_single_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_params,
                         n_runs=5, warmup=2, return_outputs=False):
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
    
    median_time = np.median(times)
    std_time = np.std(times)
    
    if return_outputs:
        return median_time, std_time, out_beta, out_status
    return median_time, std_time


def benchmark_multi_fit(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params,
                        n_runs=5, warmup=2, return_outputs=False):
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
    
    median_time = np.median(times)
    std_time = np.std(times)
    
    if return_outputs:
        return median_time, std_time, out_beta, out_status
    return median_time, std_time


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


def run_benchmark_suite(scenarios, n_runs=5, validate=True):
    """
    Run complete benchmark suite with correctness validation.
    
    Parameters
    ----------
    scenarios : list of dict
        Benchmark scenarios
    n_runs : int
        Number of timed runs per scenario
    validate : bool
        If True, validate correctness against known true coefficients
    """
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
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows/group: {rows_per_group}")
        print(f"  Features: {n_feat}")
        print(f"  Parameters: {n_params}")
        print(f"  Targets: {n_targets}")
        
        # Generate data with known true coefficients
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
        
        # Single-fit kernel with correctness check
        if n_targets == 1:
            Y_1d = Y_all
            true_coeffs_single = true_coeffs
        else:
            Y_1d = Y_all[:, 0]
            true_coeffs_single = true_coeffs[0] if validate else None
        
        single_result = benchmark_single_fit(
            X_all, Y_1d, W_all, offsets, n_groups, n_feat, n_params, n_runs,
            return_outputs=validate
        )
        
        if validate:
            single_time, single_std, out_beta, out_status = single_result
            # Validate correctness with adaptive tolerance
            validation = validate_correctness(
                out_beta, true_coeffs_single, out_status, 
                rows_per_group=rows_per_group
            )
            result['single_correctness'] = validation['passed']
            result['single_max_error'] = validation['max_error']
            result['single_rtol_used'] = validation.get('rtol_used', 0.1)
            if not validation['passed']:
                correctness_failures.append(f"{name} (single): {validation['message']}")
                print(f"  ⚠ Correctness: FAIL - {validation['message']}")
            else:
                print(f"  ✓ Correctness: PASS (max rel error: {validation['max_error']:.2e}, rtol: {validation['rtol_used']:.2f})")
        else:
            single_time, single_std = single_result
        
        result['single_time_ms'] = single_time * 1000
        result['single_groups_per_sec'] = n_groups / single_time
        result['numba_vs_numpy_speedup'] = numpy_time / single_time
        print(f"  Single-fit: {single_time*1000:.2f} ms ({n_groups/single_time:,.0f} groups/sec)")
        print(f"  → Numba vs NumPy: {numpy_time/single_time:.1f}×")
        
        # Multi-fit kernel (if multiple targets)
        if n_targets > 1:
            multi_result = benchmark_multi_fit(
                X_all, Y_all, W_all, offsets, n_groups, n_feat, n_targets, n_params, n_runs,
                return_outputs=validate
            )
            
            if validate:
                multi_time, multi_std, out_beta_m, out_status_m = multi_result
                # Validate correctness with adaptive tolerance
                validation = validate_correctness(
                    out_beta_m, true_coeffs, out_status_m,
                    rows_per_group=rows_per_group
                )
                result['multi_correctness'] = validation['passed']
                result['multi_max_error'] = validation['max_error']
                if not validation['passed']:
                    correctness_failures.append(f"{name} (multi): {validation['message']}")
                    print(f"  ⚠ Multi correctness: FAIL - {validation['message']}")
                else:
                    print(f"  ✓ Multi correctness: PASS (max rel error: {validation['max_error']:.2e})")
            else:
                multi_time, multi_std = multi_result
            
            # Compare to single-fit called n_targets times
            single_total_time = single_time * n_targets
            
            result['multi_time_ms'] = multi_time * 1000
            result['multi_groups_per_sec'] = n_groups / multi_time
            result['multi_vs_single_speedup'] = single_total_time / multi_time
            
            print(f"  Multi-fit: {multi_time*1000:.2f} ms ({n_groups/multi_time:,.0f} groups/sec)")
            print(f"  Single×{n_targets}: {single_total_time*1000:.2f} ms")
            print(f"  → Multi vs Single×{n_targets}: {single_total_time/multi_time:.2f}×")
        
        results.append(result)
    
    return results, correctness_failures


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
            # Vary group count (production-relevant sizes)
            {'name': '100 groups, 20 rows', 'n_groups': 100, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '1K groups, 20 rows', 'n_groups': 1000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '10K groups, 20 rows', 'n_groups': 10000, 'rows_per_group': 20, 'n_feat': 2},
            {'name': '50K groups, 20 rows', 'n_groups': 50000, 'rows_per_group': 20, 'n_feat': 2},
            
            # Vary rows per group (typical range: 10-50)
            {'name': '5K groups, 10 rows', 'n_groups': 5000, 'rows_per_group': 10, 'n_feat': 2},
            {'name': '5K groups, 30 rows', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': '5K groups, 50 rows', 'n_groups': 5000, 'rows_per_group': 50, 'n_feat': 2},
            
            # Vary features
            {'name': '5K groups, 2 feat', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2},
            {'name': '5K groups, 4 feat', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 4},
            {'name': '5K groups, 8 feat', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 8},
            
            # Multi-target
            {'name': '5K groups, 2 targets', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 2},
            {'name': '5K groups, 4 targets', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 4},
            {'name': '5K groups, 6 targets', 'n_groups': 5000, 'rows_per_group': 30, 'n_feat': 2, 'n_targets': 6},
            
            # Multi-target with more features (where multi-fit shines)
            {'name': '5K groups, 4 feat, 6 tgt', 'n_groups': 5000, 'rows_per_group': 50, 'n_feat': 4, 'n_targets': 6},
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
    
    # Run benchmarks with correctness validation
    results, correctness_failures = run_benchmark_suite(scenarios, n_runs, validate=True)
    
    # Print summary
    print_summary(results)
    
    # Verify gates
    print("\n" + "="*80)
    print("GATES SUMMARY")
    print("="*80)
    
    all_gates_pass = True
    
    # Gate 0: Correctness
    if correctness_failures:
        print(f"\n0. Correctness: ✗ FAIL ({len(correctness_failures)} failures)")
        for failure in correctness_failures:
            print(f"   - {failure}")
        all_gates_pass = False
    else:
        n_validated = sum(1 for r in results if r.get('single_correctness', False))
        n_validated += sum(1 for r in results if r.get('multi_correctness', False))
        print(f"\n0. Correctness: ✓ PASS ({n_validated} scenarios validated)")
    
    # Gate 1: Numba ≥5× faster than NumPy (for typical scenarios)
    # Note: For very large groups (100+ rows), speedup naturally decreases as linear algebra dominates
    typical_results = [r for r in results if r['rows_per_group'] <= 50]
    if typical_results:
        numba_speedups = [r['numba_vs_numpy_speedup'] for r in typical_results]
        min_numba_speedup = min(numba_speedups)
        min_scenario = min(typical_results, key=lambda r: r['numba_vs_numpy_speedup'])['name']
        gate1_pass = min_numba_speedup >= 5.0
        print(f"1. Numba vs NumPy: {min_numba_speedup:.1f}× (min: {min_scenario}) {'✓ PASS' if gate1_pass else '✗ FAIL'} (required: ≥5×)")
        if not gate1_pass:
            # Show all failing scenarios
            failing = [r for r in typical_results if r['numba_vs_numpy_speedup'] < 5.0]
            for f in failing:
                print(f"   - {f['name']}: {f['numba_vs_numpy_speedup']:.1f}×")
            all_gates_pass = False
    else:
        min_numba_speedup = 0
        gate1_pass = False
        print(f"1. Numba vs NumPy: NO TYPICAL SCENARIOS ✗ FAIL")
    
    # Gate 2: Multi-fit not drastically slower
    multi_speedups = [r.get('multi_vs_single_speedup') for r in results if 'multi_vs_single_speedup' in r]
    if multi_speedups:
        min_multi_speedup = min(multi_speedups)
        gate2_pass = min_multi_speedup >= 0.5
        print(f"2. Multi vs Single: {min_multi_speedup:.2f}× (min) {'✓ PASS' if gate2_pass else '✗ FAIL'} (required: ≥0.5×)")
        if not gate2_pass:
            all_gates_pass = False
    
    # Save to JSON if requested
    if args.json:
        # Convert numpy types to native Python for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            return obj
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'numba_available': bool(_NUMBA_AVAILABLE),
            'results': [convert_for_json(r) for r in results],
            'correctness_failures': correctness_failures,
            'gates': {
                'correctness_pass': len(correctness_failures) == 0,
                'numba_vs_numpy_min': float(min_numba_speedup),
                'numba_vs_numpy_pass': bool(gate1_pass),
            }
        }
        if multi_speedups:
            output['gates']['multi_vs_single_min'] = float(min_multi_speedup)
            output['gates']['multi_vs_single_pass'] = bool(gate2_pass)
        output['gates']['all_pass'] = bool(all_gates_pass)
        
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")
    
    # Exit with error code if gates fail
    print("\n" + "="*80)
    if all_gates_pass:
        print("ALL GATES PASSED ✓")
    else:
        print("SOME GATES FAILED ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()
