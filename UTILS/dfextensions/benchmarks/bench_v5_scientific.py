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
# Scenario Definition (Phase 12.12b.1: Uniform 2× steps, R=100 constant)
# =============================================================================

SCENARIOS = {
    "S4": {"n_rows": 500_000, "n_groups": 5_000, "n_fits": 6, "n_params": 3},
    "S5": {"n_rows": 1_000_000, "n_groups": 10_000, "n_fits": 6, "n_params": 3},
    "S6": {"n_rows": 2_000_000, "n_groups": 20_000, "n_fits": 6, "n_params": 3},
    "S7": {"n_rows": 4_000_000, "n_groups": 40_000, "n_fits": 6, "n_params": 3},
}

# =============================================================================
# Ground Truth for Correctness Validation
# =============================================================================

TRUE_BETA = np.array([2.0, 3.0, -1.5])  # [intercept, slope_x1, slope_x2]
NOISE_SCALE = 0.01  # Small noise for tight validation


# =============================================================================
# Test Data Generation — Parametric with Known Ground Truth
# =============================================================================

def generate_parametric_test_data(
    n_rows: int,
    n_groups: int,
    n_fits: int = 6,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate test DataFrame with KNOWN ground truth for correctness validation.
    
    Model: y = β₀ + β₁x₁ + β₂x₂ + ε
    Where: β = TRUE_BETA = [2.0, 3.0, -1.5]
           ε ~ N(0, NOISE_SCALE²)
    
    Returns
    -------
    df : pd.DataFrame
        Test data with columns: group_id, x1, x2, w, y1-y6
    true_beta : np.ndarray
        Ground truth coefficients [intercept, slope_x1, slope_x2]
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
    
    # Generate predictors
    x1 = np.random.randn(n_rows).astype(np.float64)
    x2 = np.random.randn(n_rows).astype(np.float64)
    
    # Generate y = β₀ + β₁x₁ + β₂x₂ (true signal)
    y_true = TRUE_BETA[0] + TRUE_BETA[1] * x1 + TRUE_BETA[2] * x2
    
    # Build DataFrame
    data = {
        "group_id": group_ids,
        "x1": x1,
        "x2": x2,
        "w": np.ones(n_rows, dtype=np.float64),  # Unit weights
    }
    
    # All fits use same relationship with INDEPENDENT noise per fit
    for i in range(n_fits):
        noise = NOISE_SCALE * np.random.randn(n_rows)
        data[f"y{i+1}"] = (y_true + noise).astype(np.float64)
    
    return pd.DataFrame(data), TRUE_BETA


def generate_test_dataframe(
    n_rows: int,
    n_groups: int,
    n_params: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Legacy wrapper for compatibility. Returns only DataFrame.
    Use generate_parametric_test_data() for correctness validation.
    """
    df, _ = generate_parametric_test_data(n_rows, n_groups, n_fits=6, seed=seed)
    return df


# =============================================================================
# Correctness Validation (Phase 12.12b.1)
# =============================================================================

def compute_validation_tolerance(
    n_rows: int,
    n_groups: int,
    noise_scale: float = NOISE_SCALE,
    n_sigma: float = 5.0,
    sigma_round: float = 0.0001,
) -> Dict[str, Any]:
    """
    Compute n-sigma tolerances for validation.
    
    Two levels:
    1. Within-group: σ_stat = σ_noise / √R
    2. Across-groups: σ_stat = σ_group / √G
    
    Parameters
    ----------
    sigma_round : float
        Numerical precision floor. Default 0.0001 for float64.
    """
    rows_per_group = n_rows // n_groups
    
    # Within-group: σ_stat = σ_noise / √R
    sigma_stat_group = float(noise_scale / np.sqrt(rows_per_group))
    sigma_total_group = float(np.sqrt(sigma_stat_group**2 + sigma_round**2))
    tolerance_group = float(n_sigma * sigma_total_group)
    
    # Across-groups: σ_stat = σ_group / √G (NOT σ_noise / √N)
    sigma_stat_mean = float(sigma_stat_group / np.sqrt(n_groups))
    sigma_total_mean = float(np.sqrt(sigma_stat_mean**2 + sigma_round**2))
    tolerance_mean = float(n_sigma * sigma_total_mean)
    
    return {
        "config": {
            "n_sigma": float(n_sigma),
            "noise_scale": float(noise_scale),
            "sigma_round": float(sigma_round),
            "sigma_round_source": "parameter",
        },
        "within_group": {
            "R": int(rows_per_group),
            "sigma_stat": sigma_stat_group,
            "sigma_total": sigma_total_group,
            "tolerance": tolerance_group,
            "criterion": "quantile_99.9",
        },
        "across_groups": {
            "G": int(n_groups),
            "sigma_stat": sigma_stat_mean,
            "sigma_total": sigma_total_mean,
            "tolerance": tolerance_mean,
            "criterion": "max_of_means",
        },
    }


def validate_correctness(
    result_df: pd.DataFrame,
    true_beta: np.ndarray,
    tolerance_config: Dict[str, Any],
    fit_suffix: str = "_v5",
    target_col: str = "y1",
    linear_cols: List[str] = ["x1", "x2"],
) -> Dict[str, Any]:
    """
    Two-level validation with proper statistical criteria.
    
    Level 1 (within-group): 99.9% quantile of group errors < tolerance
    Level 2 (across-groups): max mean error < tolerance
    """
    n_groups = len(result_df)
    n_sigma = tolerance_config["config"]["n_sigma"]
    
    # Build column names based on V4/V5 naming convention
    # Pattern: {target}_intercept{suffix}, {target}_slope_{col}{suffix}
    col_intercept = f"{target_col}_intercept{fit_suffix}"
    col_x1 = f"{target_col}_slope_{linear_cols[0]}{fit_suffix}"
    col_x2 = f"{target_col}_slope_{linear_cols[1]}{fit_suffix}"
    
    # Check columns exist
    missing = []
    for col in [col_intercept, col_x1, col_x2]:
        if col not in result_df.columns:
            missing.append(col)
    
    if missing:
        return {
            "passed": False,
            "status": "❌ FAIL - Missing columns",
            "missing_columns": missing,
            "available_columns": list(result_df.columns)[:20],
        }
    
    # Extract fitted coefficients
    fitted_intercept = result_df[col_intercept].values
    fitted_x1 = result_df[col_x1].values
    fitted_x2 = result_df[col_x2].values
    
    # === LEVEL 1: Within-group (quantile criterion) ===
    tol_group = tolerance_config["within_group"]["tolerance"]
    sigma_group = tolerance_config["within_group"]["sigma_total"]
    
    group_errors = np.column_stack([
        np.abs(fitted_intercept - true_beta[0]),
        np.abs(fitted_x1 - true_beta[1]),
        np.abs(fitted_x2 - true_beta[2]),
    ])
    
    # Use 99.9% quantile instead of max (handles multiple comparisons)
    max_error_per_group = group_errors.max(axis=1)
    quantile_99_9 = np.percentile(max_error_per_group, 99.9)
    fraction_failing = np.mean(max_error_per_group > tol_group)
    
    group_passed = quantile_99_9 < tol_group
    
    # === LEVEL 2: Across-groups (mean) ===
    tol_mean = tolerance_config["across_groups"]["tolerance"]
    sigma_mean = tolerance_config["across_groups"]["sigma_total"]
    
    mean_errors = {
        "intercept": {
            "true": float(true_beta[0]),
            "fitted_mean": float(fitted_intercept.mean()),
            "fitted_std": float(fitted_intercept.std()),
            "error": float(abs(fitted_intercept.mean() - true_beta[0])),
        },
        "x1": {
            "true": float(true_beta[1]),
            "fitted_mean": float(fitted_x1.mean()),
            "fitted_std": float(fitted_x1.std()),
            "error": float(abs(fitted_x1.mean() - true_beta[1])),
        },
        "x2": {
            "true": float(true_beta[2]),
            "fitted_mean": float(fitted_x2.mean()),
            "fitted_std": float(fitted_x2.std()),
            "error": float(abs(fitted_x2.mean() - true_beta[2])),
        },
    }
    
    max_mean_error = max(e["error"] for e in mean_errors.values())
    mean_passed = max_mean_error < tol_mean
    
    # === Combined verdict ===
    passed = bool(group_passed and mean_passed)
    
    return {
        "passed": passed,
        "status": "✅ PASS" if passed else "❌ FAIL",
        "config": tolerance_config["config"],
        
        "within_group": {
            "passed": bool(group_passed),
            "criterion": "quantile_99.9",
            "tolerance": float(tol_group),
            "quantile_99_9": float(quantile_99_9),
            "quantile_99_9_sigma": float(quantile_99_9 / sigma_group),
            "fraction_failing": float(fraction_failing),
            "status": "✅ PASS" if group_passed else "❌ FAIL",
        },
        
        "across_groups": {
            "passed": bool(mean_passed),
            "criterion": "max_of_means",
            "tolerance": float(tol_mean),
            "max_error": float(max_mean_error),
            "max_error_sigma": float(max_mean_error / sigma_mean),
            "errors": mean_errors,
            "status": "✅ PASS" if mean_passed else "❌ FAIL",
        },
    }


# =============================================================================
# V4 and V5 Measurement Functions
# =============================================================================

def measure_v4(
    df: pd.DataFrame,
    gb_cols: List[str],
    fit_cols: List[str],
    linear_cols: List[str],
    weights: str,
    n_warmup: int = 2,
    n_steady: int = 5,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Measure V4 baseline performance with warmup equalization.
    
    Same protocol as V5: warmup runs before timing.
    Returns (timing_result, result_df) for correctness validation.
    
    V4 returns tuple: (input_df, coefficients_df)
    We need coefficients_df (index 1) for validation.
    """
    try:
        from groupby_regression.groupby_regression_optimized import make_parallel_fit_v4
    except ImportError:
        return {"error": "Cannot import make_parallel_fit_v4"}, None
    
    result_df = None
    
    def extract_coefficients(result):
        """Extract coefficients DataFrame from V4 result.
        
        V4 returns (input_df, coefficients_df) — we need index [1].
        """
        if isinstance(result, tuple):
            return result[1]  # coefficients_df
        return result
    
    # Warmup runs (discard, same as V5)
    for _ in range(n_warmup):
        gc.collect()
        result = make_parallel_fit_v4(
            df=df,
            gb_columns=gb_cols,
            fit_columns=fit_cols,
            suffix="_v4",
            linear_columns=linear_cols,
            weights=weights,
        )
        result_df = extract_coefficients(result)
    
    # Steady state timing
    times_ms = []
    for run_idx in range(n_steady):
        gc.collect()
        
        t0 = time.perf_counter()
        
        result = make_parallel_fit_v4(
            df=df,
            gb_columns=gb_cols,
            fit_columns=fit_cols,
            suffix="_v4",
            linear_columns=linear_cols,
            weights=weights,
        )
        
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
        result_df = extract_coefficients(result)
    
    return {
        "times_ms": times_ms,
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "n_warmup": n_warmup,
        "n_steady": n_steady,
        "n_fits": len(fit_cols),
    }, result_df


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
    Single V5 measurement. Returns (time_ms, coefficients_df).
    
    V5 returns tuple: (input_df, coefficients_df)
    We need coefficients_df (index 1) for validation.
    """
    try:
        from groupby_regression.groupby_regression_optimized import make_parallel_fit_v5
    except ImportError:
        return -1.0, None
    
    def extract_coefficients(result):
        """Extract coefficients DataFrame from V5 result.
        
        V5 returns (input_df, coefficients_df) — we need index [1].
        """
        if isinstance(result, tuple):
            return result[1]  # coefficients_df
        return result
    
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
    
    return (t1 - t0) * 1000, extract_coefficients(result)


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
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Measure V5 with cold/warm/steady separation.
    
    Execution order:
    1. Run 1: Cold (includes JIT)
    2. Runs 2-3: Warmup (discard)
    3. Runs 4-13: Steady state (record)
    
    Returns (timing_result, result_df) for correctness validation.
    """
    # Run 1: Cold
    t_cold, _ = measure_v5_single(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs
    )
    
    if t_cold < 0:
        return {"error": "Cannot import make_parallel_fit_v5"}, None
    
    # Runs 2-3: Warmup (discard)
    for _ in range(n_warmup):
        measure_v5_single(
            df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes, n_jobs
        )
    
    # Runs 4+: Steady state (keep last result for correctness)
    steady_times = []
    result_df = None
    for _ in range(n_steady):
        t, result_df = measure_v5_single(
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
    }, result_df


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
        v5_result, _ = measure_v5_full(
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
    
    # Extract coefficients DataFrame (index 1)
    if isinstance(result, tuple):
        result = result[1]  # coefficients_df
    
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
    correctness_only: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete scientific benchmark for V5.
    
    Phase 12.12b.1: Includes correctness validation as BLOCKING GATE.
    
    Parameters
    ----------
    correctness_only : bool
        If True, stop after correctness validation (skip performance tests).
    
    Returns single JSON structure with all measurements and provenance.
    """
    params = SCENARIOS.get(scenario, SCENARIOS["S4"])
    n_rows = params["n_rows"]
    n_groups = params["n_groups"]
    n_fits = params["n_fits"]
    n_params = params["n_params"]
    
    if verbose:
        print("=" * 70)
        print("Phase 12.12b.1: Scientific V5 Benchmark")
        print("=" * 70)
        print(f"Scenario: {scenario}")
        print(f"  Rows: {n_rows:,}, Groups: {n_groups:,}, R={n_rows//n_groups}")
        print(f"  Fits: {n_fits}, Params: {n_params}")
        print(f"  Ground truth β = {list(TRUE_BETA)}")
        print(f"  Noise scale σ = {NOISE_SCALE}")
        print()
    
    # Generate PARAMETRIC test data with known ground truth
    if verbose:
        print("Generating parametric test data...")
    df, true_beta = generate_parametric_test_data(n_rows, n_groups, n_fits)
    
    # Setup parameters
    gb_cols = ["group_id"]
    fit_cols = ["y1", "y2", "y3", "y4", "y5", "y6"][:n_fits]
    linear_cols = ["x1", "x2"][:n_params - 1]  # Intercept is implicit
    weights = "w"
    
    suffixes = [f"_{i+1}" for i in range(n_fits)]
    linear_cols_list = [linear_cols for _ in range(n_fits)]
    weights_list = [weights for _ in range(n_fits)]
    
    # Compute validation tolerances
    tolerance_config = compute_validation_tolerance(n_rows, n_groups)
    
    # Initialize results structure
    results = {
        "meta": {
            "phase": "12.12b.1",
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
            "true_beta": list(true_beta),
            "noise_scale": NOISE_SCALE,
        },
        "tolerance_config": tolerance_config,
    }
    
    # ==========================================================================
    # STEP 0: CORRECTNESS VALIDATION (BLOCKING GATE)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 0: CORRECTNESS VALIDATION (BLOCKING GATE)")
        print("=" * 70)
    
    # --- V4 Correctness ---
    if verbose:
        print("\n[0a] Validating V4 correctness...")
    
    v4_timing, v4_result_df = measure_v4(
        df, gb_cols, fit_cols, linear_cols, weights, n_warmup=2, n_steady=3
    )
    
    if "error" in v4_timing or v4_result_df is None:
        results["correctness"] = {
            "v4": {"passed": False, "status": "❌ FAIL - Import error", "error": v4_timing.get("error")},
            "v5": {"passed": False, "status": "⏭️ SKIPPED"},
            "gate_passed": False,
        }
        if verbose:
            print(f"  V4 ERROR: {v4_timing.get('error')}")
            print("\n🚫 CORRECTNESS GATE FAILED — Cannot proceed")
        return results
    
    v4_correctness = validate_correctness(
        v4_result_df, true_beta, tolerance_config,
        fit_suffix="_v4", target_col="y1", linear_cols=linear_cols
    )
    
    # Check for missing columns error
    if "missing_columns" in v4_correctness:
        if verbose:
            print(f"  V4 ERROR: Missing columns: {v4_correctness['missing_columns']}")
            print(f"  Available columns (first 20): {v4_correctness.get('available_columns', [])}")
        results["correctness"] = {
            "v4": v4_correctness,
            "v5": {"passed": False, "status": "⏭️ SKIPPED"},
            "gate_passed": False,
        }
        return results
    
    if verbose:
        print(f"  V4 within-group (99.9%): {v4_correctness['within_group']['quantile_99_9']:.6f} (tol: {v4_correctness['within_group']['tolerance']:.6f}) {v4_correctness['within_group']['status']}")
        print(f"  V4 across-groups (max):  {v4_correctness['across_groups']['max_error']:.6f} (tol: {v4_correctness['across_groups']['tolerance']:.6f}) {v4_correctness['across_groups']['status']}")
        print(f"  V4 Overall: {v4_correctness['status']}")
    
    # --- V5 Correctness ---
    if verbose:
        print("\n[0b] Validating V5 correctness...")
    
    v5_timing, v5_result_df = measure_v5_full(
        df, gb_cols, fit_cols, linear_cols_list, weights_list, suffixes,
        n_jobs=1, n_warmup=2, n_steady=3
    )
    
    if "error" in v5_timing or v5_result_df is None:
        results["correctness"] = {
            "v4": v4_correctness,
            "v5": {"passed": False, "status": "❌ FAIL - Import error", "error": v5_timing.get("error")},
            "gate_passed": False,
        }
        if verbose:
            print(f"  V5 ERROR: {v5_timing.get('error')}")
            print("\n🚫 CORRECTNESS GATE FAILED — Cannot proceed")
        return results
    
    # V5 uses suffix like "_1" for first fit
    v5_correctness = validate_correctness(
        v5_result_df, true_beta, tolerance_config,
        fit_suffix="_1", target_col="y1", linear_cols=linear_cols
    )
    
    # Check for missing columns error
    if "missing_columns" in v5_correctness:
        if verbose:
            print(f"  V5 ERROR: Missing columns: {v5_correctness['missing_columns']}")
            print(f"  Available columns (first 20): {v5_correctness.get('available_columns', [])}")
        results["correctness"] = {
            "v4": v4_correctness,
            "v5": v5_correctness,
            "gate_passed": False,
        }
        return results
    
    if verbose:
        print(f"  V5 within-group (99.9%): {v5_correctness['within_group']['quantile_99_9']:.6f} (tol: {v5_correctness['within_group']['tolerance']:.6f}) {v5_correctness['within_group']['status']}")
        print(f"  V5 across-groups (max):  {v5_correctness['across_groups']['max_error']:.6f} (tol: {v5_correctness['across_groups']['tolerance']:.6f}) {v5_correctness['across_groups']['status']}")
        print(f"  V5 Overall: {v5_correctness['status']}")
    
    # --- Gate Decision ---
    gate_passed = bool(v4_correctness["passed"] and v5_correctness["passed"])
    
    results["correctness"] = {
        "v4": v4_correctness,
        "v5": v5_correctness,
        "gate_passed": gate_passed,
    }
    
    if verbose:
        print("\n" + "-" * 70)
        if gate_passed:
            print("✅ CORRECTNESS GATE PASSED — Proceeding to performance tests")
        else:
            print("🚫 CORRECTNESS GATE FAILED — Cannot proceed")
            print("   Fix the algorithm before trusting performance numbers!")
    
    if not gate_passed or correctness_only:
        return results
    
    # ==========================================================================
    # STEP 1: V4 Baseline (with warmup equalization)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("PERFORMANCE TESTS")
        print("=" * 70)
        print("\n[1/5] Measuring V4 baseline (with warmup)...")
    
    v4_result, _ = measure_v4(df, gb_cols, fit_cols, linear_cols, weights, n_warmup=2, n_steady=5)
    results["v4_baseline"] = v4_result
    
    if verbose:
        if "error" in v4_result:
            print(f"  ERROR: {v4_result['error']}")
        else:
            print(f"  V4: {v4_result['mean_ms']:.1f} ± {v4_result['std_ms']:.1f} ms (n_warmup={v4_result['n_warmup']}, n_steady={v4_result['n_steady']})")
    
    # ==========================================================================
    # STEP 2: V5 Sequential (Cold/Warm/Steady)
    # ==========================================================================
    if verbose:
        print("\n[2/5] Measuring V5 sequential (cold/warm/steady)...")
    
    v5_seq, _ = measure_v5_full(
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
    # STEP 3: V5 vs V4 Comparison
    # ==========================================================================
    if verbose:
        print("\n[3/5] Computing V5 vs V4 speedup...")
    
    if "error" not in v4_result and "error" not in v5_seq:
        v5_vs_v4_speedup = v4_result["mean_ms"] / v5_seq["steady_ms"] if v5_seq["steady_ms"] > 0 else 0
        results["comparison"] = {
            "v5_vs_v4_speedup": round(v5_vs_v4_speedup, 2),
            "v4_ms": v4_result["mean_ms"],
            "v5_steady_ms": v5_seq["steady_ms"],
            "note": "Fair comparison: both use 2-warmup + N-steady protocol",
        }
        
        if verbose:
            print(f"  V4:  {v4_result['mean_ms']:.1f} ms")
            print(f"  V5:  {v5_seq['steady_ms']:.1f} ms")
            print(f"  Speedup: {v5_vs_v4_speedup:.2f}×")
    
    # ==========================================================================
    # STEP 4: V5 Parallel Scaling
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
    # STEP 5: Memory Measurement
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
        
        print(f"\n  Correctness: {'✅ PASS' if gate_passed else '❌ FAIL'}")
        
        if "comparison" in results:
            c = results["comparison"]
            print(f"  V5 vs V4 speedup: {c['v5_vs_v4_speedup']:.2f}×")
        
        if "v5_scaling" in results and "12" in results["v5_scaling"]["results"]:
            s12 = results["v5_scaling"]["results"]["12"]
            if "error" not in s12:
                print(f"  V5 parallel (12 threads): {s12['speedup']:.2f}× speedup, {s12['efficiency']:.0%} efficiency")
        
        if "memory" in results and "error" not in results["memory"]:
            print(f"  Memory (RSS delta): {results['memory']['rss_delta_mb']:.1f} MB")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 12.12b.1: Scientific V5 Benchmark with Correctness Validation",
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
        "--correctness-only", "-c",
        action="store_true",
        help="Run only correctness validation (skip performance tests)",
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
        correctness_only=args.correctness_only,
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
