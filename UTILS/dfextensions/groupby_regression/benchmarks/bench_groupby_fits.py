#!/usr/bin/env python3
"""
bench_groupby_fits.py — BF-Integrated Benchmark for GroupBy Regression

Phase 12.11: Profiling-enabled benchmark for v4/v5 performance recovery.

This module integrates with the Benchmark Framework (dfextensions/benchmarks/).
It can be used via BF runner.py or as a convenience standalone script.

Usage via BF:
    python -m dfextensions.benchmarks.runner \\
        --subproject groupby_regression \\
        --benchmark groupby_fits \\
        --suite quick

Direct usage (convenience wrapper):
    python bench_groupby_fits.py --scenarios S3,S4

Engines:
    - v4: Numba JIT kernel (production baseline)
    - v5: Batch fits with streaming (target of optimization)
    - v2, v3: Legacy (deferred to v1.1)

Key Features:
    - CPU profiling for n_jobs=1 (cProfile)
    - Wall-time always measured (consistent metric)
    - Memory tracking (RSS + tracemalloc)
    - Backend selection logging (verifies Numba fix)
    - BF schema-compliant output

Author: Team 3 Coder (Claude)
Date: 2025-12-25
"""

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import numpy as np
import pandas as pd

# These imports will work when run from dfextensions package
# For standalone testing, we use local imports
try:
    from dfextensions.benchmarks.profiler import (
        profile_function_full,
        CombinedProfileResult,
        get_peak_rss_mb,
    )
    from dfextensions.benchmarks.schema import (
        ProfileInfo,
        BackendInfo,
        BenchmarkResult,
        RunMeta,
        RunSummary,
        BenchmarkRun,
    )
    from dfextensions.groupby_regression.groupby_regression_optimized import (
        make_parallel_fit_v4,
        make_parallel_fit_v5,
        _select_parallel_backend,
        _NUMBA_AVAILABLE,
    )
except ImportError:
    # Standalone mode - imports handled at runtime
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_NAME = "groupby_fits"

# Engine registry
ENGINES = {
    'v4': 'make_parallel_fit_v4',
    'v5': 'make_parallel_fit_v5',
    # v2, v3 deferred to v1.1
}

DEFAULT_ENGINES = ['v4', 'v5']

# Scenarios (matching BF conventions)
SCENARIOS = {
    'S1': {'n_rows': 10_000, 'n_groups': 100, 'description': 'Quick test'},
    'S2': {'n_rows': 50_000, 'n_groups': 500, 'description': 'Small'},
    'S3': {'n_rows': 100_000, 'n_groups': 1_000, 'description': 'Development'},
    'S4': {'n_rows': 500_000, 'n_groups': 5_000, 'description': 'Integration'},
    'S5': {'n_rows': 1_000_000, 'n_groups': 10_000, 'description': 'Performance'},
    'S6': {'n_rows': 5_000_000, 'n_groups': 50_000, 'description': 'Stress'},
}

DEFAULT_SCENARIOS = ['S3', 'S4']


# =============================================================================
# DATA GENERATION (Excluded from profiling)
# =============================================================================

def create_test_data(
    n_rows: int,
    n_groups: int,
    n_fit_cols: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create deterministic test DataFrame.
    
    Called BEFORE profiling to exclude data generation from measurements.
    Uses fixed seed for reproducibility across runs.
    """
    np.random.seed(seed)
    rows_per_group = n_rows // n_groups
    
    group = np.repeat(np.arange(n_groups), rows_per_group)
    if len(group) < n_rows:
        group = np.concatenate([group, np.zeros(n_rows - len(group), dtype=int)])
    group = group[:n_rows]
    
    x1 = np.random.randn(n_rows)
    x2 = np.random.randn(n_rows)
    
    fit_cols = {}
    for i in range(n_fit_cols):
        noise = np.random.randn(n_rows) * 0.1
        fit_cols[f"y{i+1}"] = 1.0 + 0.5 * x1 + 0.3 * x2 + noise
    
    w = np.abs(np.random.randn(n_rows)) + 0.1
    
    return pd.DataFrame({
        "group": group,
        "x1": x1,
        "x2": x2,
        **fit_cols,
        "w": w,
    })


# =============================================================================
# BENCHMARK ID
# =============================================================================

def make_benchmark_id(engine: str, scenario: str, n_jobs: int) -> str:
    """
    Create stable benchmark ID.
    
    Format: groupby_fits:{engine}:{scenario}:n_jobs={n_jobs}
    Example: groupby_fits:v5:S3:n_jobs=1
    """
    return f"{BENCHMARK_NAME}:{engine}:{scenario}:n_jobs={n_jobs}"


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    engine: str
    scenario: str
    n_jobs: int
    n_fits: int
    profiles_dir: Path
    profile_top_n: int
    warmup_runs: int
    timed_runs: int


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def _run_v4_all_fits(
    make_parallel_fit_v4_func: Callable,
    df: pd.DataFrame,
    fit_columns: List[str],
) -> List:
    """
    Run v4 for all fit columns (v4 only supports single fit_columns).
    
    This wrapper makes v4 comparable to v5 which processes all fits at once.
    """
    results = []
    for fit_col in fit_columns:
        result = make_parallel_fit_v4_func(
            df=df,
            gb_columns=["group"],
            fit_columns=fit_col,
            linear_columns=["x1", "x2"],
            weights="w",
        )
        results.append(result)
    return results


def run_v4_benchmark(
    df: pd.DataFrame,
    config: BenchmarkConfig,
    make_parallel_fit_v4_func: Callable,
    numba_available: bool,
    profile_function_full_func: Callable,
    get_peak_rss_mb_func: Callable,
) -> Dict:
    """
    Run v4 benchmark with profiling.
    
    Note: v4 doesn't have n_jobs parameter — always single-threaded.
    v4 only supports single fit_columns, so we loop over all fits for fair comparison.
    """
    # v4 only runs with n_jobs=1
    if config.n_jobs != 1:
        return {
            "id": make_benchmark_id("v4", config.scenario, config.n_jobs),
            "status": "SKIPPED",
            "error_message": f"v4 does not support n_jobs={config.n_jobs}",
        }
    
    benchmark_id = make_benchmark_id("v4", config.scenario, 1)
    profile_name = f"v4_{config.scenario}_n1"
    
    # Build fit columns list (same as v5 for fair comparison)
    fit_columns = [f"y{i+1}" for i in range(config.n_fits)]
    
    # Warmup runs (for Numba JIT) - run all fits
    for _ in range(config.warmup_runs):
        _ = _run_v4_all_fits(make_parallel_fit_v4_func, df, fit_columns)
    
    gc.collect()
    
    # Timed runs with profiling
    times = []
    profile_result = None
    
    for i in range(config.timed_runs):
        # Profile only first run (CPU profile)
        do_cpu_profile = (i == 0)
        
        # Profile the wrapper that runs all fits
        result, profile = profile_function_full_func(
            _run_v4_all_fits,
            make_parallel_fit_v4_func,
            df,
            fit_columns,
            output_dir=str(config.profiles_dir),
            name=profile_name,
            top_n=config.profile_top_n,
            cpu_enabled=do_cpu_profile,
            memory_enabled=(i == 0),
        )
        
        # Always use wall_time (consistent metric!)
        times.append(profile.cpu.wall_time_s)
        
        if i == 0:
            profile_result = profile
    
    # Build result dict
    return {
        "id": benchmark_id,
        "name": BENCHMARK_NAME,
        "scenario": config.scenario,
        "status": "OK",
        "time_s": round(float(np.mean(times)), 4),
        "time_std_s": round(float(np.std(times)), 4) if len(times) > 1 else 0.0,
        "n_runs": config.timed_runs,
        "peak_rss_mb": profile_result.peak_rss_mb if profile_result else get_peak_rss_mb_func(),
        "params": {
            "engine": "v4",
            "scenario": config.scenario,
            "n_jobs": 1,
            "n_fits": config.n_fits,  # Now tracking n_fits for v4 too
            "n_rows": len(df),
            "n_groups": int(df["group"].nunique()),
        },
        "profile": {
            "cpu_enabled": True,
            "wall_time_s": profile_result.cpu.wall_time_s if profile_result else None,
            "cpu_total_time_s": profile_result.cpu.cpu_total_time_s if profile_result else None,
            "cpu_prof_path": profile_result.cpu.prof_path if profile_result else None,
            "cpu_txt_path": profile_result.cpu.txt_path if profile_result else None,
            "cpu_json_path": profile_result.cpu.json_path if profile_result else None,
            "cpu_total_calls": profile_result.cpu.total_calls if profile_result else None,
            "cpu_top_functions": profile_result.cpu.top_functions if profile_result else None,
            "cpu_sort_key": "cumulative",
            "tracemalloc_peak_mb": profile_result.tracemalloc_peak_mb if profile_result else None,
        },
        "backend": {
            "selected_backend": "numba" if numba_available else "numpy",
            "n_jobs": 1,
            "numba_available": numba_available,
        },
    }


def run_v5_benchmark(
    df: pd.DataFrame,
    config: BenchmarkConfig,
    make_parallel_fit_v5_func: Callable,
    select_parallel_backend_func: Callable,
    numba_available: bool,
    profile_function_full_func: Callable,
    get_peak_rss_mb_func: Callable,
) -> Dict:
    """
    Run v5 benchmark with profiling.
    
    CPU profiling only for n_jobs=1 (cProfile misleading for parallel).
    Wall-time always measured for consistent metrics.
    """
    benchmark_id = make_benchmark_id("v5", config.scenario, config.n_jobs)
    profile_name = f"v5_{config.scenario}_n{config.n_jobs}"
    
    fit_columns = [f"y{i+1}" for i in range(config.n_fits)]
    suffixes = [f"_f{i+1}" for i in range(config.n_fits)]
    
    # Capture ACTUAL backend selection (not inferred)
    actual_backend = select_parallel_backend_func("auto", config.n_jobs)
    
    # CPU profiling condition: n_jobs=1 only
    cpu_profile_enabled = (config.n_jobs == 1)
    
    # Warmup runs
    for _ in range(config.warmup_runs):
        _ = make_parallel_fit_v5_func(
            df=df,
            gb_columns=["group"],
            fit_columns=fit_columns,
            linear_columns=["x1", "x2"],
            suffixes=suffixes,
            weights="w",
            n_jobs=config.n_jobs,
        )
    
    gc.collect()
    
    # Timed runs
    times = []
    profile_result = None
    
    for i in range(config.timed_runs):
        do_cpu_profile = cpu_profile_enabled and (i == 0)
        
        result, profile = profile_function_full_func(
            make_parallel_fit_v5_func,
            df=df,
            gb_columns=["group"],
            fit_columns=fit_columns,
            linear_columns=["x1", "x2"],
            suffixes=suffixes,
            weights="w",
            n_jobs=config.n_jobs,
            output_dir=str(config.profiles_dir),
            name=profile_name,
            top_n=config.profile_top_n,
            cpu_enabled=do_cpu_profile,
            memory_enabled=(i == 0),
        )
        
        # Always use wall_time (consistent metric!)
        times.append(profile.cpu.wall_time_s)
        
        if i == 0:
            profile_result = profile
    
    # Build profile dict (only if CPU profiling was enabled)
    profile_dict = {
        "cpu_enabled": cpu_profile_enabled,
        "wall_time_s": profile_result.cpu.wall_time_s if profile_result else None,
        "tracemalloc_peak_mb": profile_result.tracemalloc_peak_mb if profile_result else None,
    }
    
    if cpu_profile_enabled and profile_result and profile_result.cpu.enabled:
        profile_dict.update({
            "cpu_total_time_s": profile_result.cpu.cpu_total_time_s,
            "cpu_prof_path": profile_result.cpu.prof_path,
            "cpu_txt_path": profile_result.cpu.txt_path,
            "cpu_json_path": profile_result.cpu.json_path,
            "cpu_total_calls": profile_result.cpu.total_calls,
            "cpu_top_functions": profile_result.cpu.top_functions,
            "cpu_sort_key": "cumulative",
        })
    
    return {
        "id": benchmark_id,
        "name": BENCHMARK_NAME,
        "scenario": config.scenario,
        "status": "OK",
        "time_s": round(float(np.mean(times)), 4),
        "time_std_s": round(float(np.std(times)), 4) if len(times) > 1 else 0.0,
        "n_runs": config.timed_runs,
        "peak_rss_mb": profile_result.peak_rss_mb if profile_result else get_peak_rss_mb_func(),
        "params": {
            "engine": "v5",
            "scenario": config.scenario,
            "n_jobs": config.n_jobs,
            "n_fits": config.n_fits,
            "n_rows": len(df),
            "n_groups": int(df["group"].nunique()),
        },
        "profile": profile_dict,
        "backend": {
            "selected_backend": actual_backend,
            "n_jobs": config.n_jobs,
            "numba_available": numba_available,
        },
    }


# =============================================================================
# BENCHMARK SUITE (BF Integration)
# =============================================================================

def get_benchmark_suite(
    engines: List[str] = None,
    scenarios: List[str] = None,
    n_jobs_list: List[int] = None,
) -> List[Dict]:
    """
    Return benchmark configurations for BF runner.
    
    This is the interface used by runner.py to discover benchmarks.
    """
    engines = engines or DEFAULT_ENGINES
    scenarios = scenarios or DEFAULT_SCENARIOS
    n_jobs_list = n_jobs_list or [1]
    
    suite = []
    
    for scenario in scenarios:
        for engine in engines:
            for n_jobs in n_jobs_list:
                # Skip v4 with n_jobs > 1
                if engine == 'v4' and n_jobs != 1:
                    continue
                
                suite.append({
                    'id': make_benchmark_id(engine, scenario, n_jobs),
                    'engine': engine,
                    'scenario': scenario,
                    'n_jobs': n_jobs,
                    'scenario_config': SCENARIOS[scenario],
                })
    
    return suite


# =============================================================================
# CONVENIENCE CLI (Standalone mode)
# =============================================================================

def main():
    """Convenience CLI for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 12.11: GroupBy Fits Benchmark with CPU Profiling",
        epilog="For full options, use: python -m dfextensions.benchmarks.runner --help"
    )
    
    parser.add_argument(
        "--engines",
        default=",".join(DEFAULT_ENGINES),
        help=f"Comma-separated engines (default: {','.join(DEFAULT_ENGINES)})",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=f"Comma-separated scenarios (default: {','.join(DEFAULT_SCENARIOS)})",
    )
    parser.add_argument(
        "--n-jobs",
        default="1",
        help="Comma-separated n_jobs values (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs (default: 3)",
    )
    parser.add_argument(
        "--profile-top-n",
        type=int,
        default=10,
        help="Top N functions in profile (default: 10)",
    )
    parser.add_argument(
        "--n-fits",
        type=int,
        default=6,
        help="Number of fit columns for v5 (default: 6)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory (default: results/)",
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    engines = [e.strip() for e in args.engines.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    n_jobs_list = [int(n.strip()) for n in args.n_jobs.split(",")]
    
    # Import required modules (standalone mode)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from dfextensions.benchmarks.profiler import (
            profile_function_full,
            get_peak_rss_mb,
        )
        from dfextensions.groupby_regression.groupby_regression_optimized import (
            make_parallel_fit_v4,
            make_parallel_fit_v5,
            _select_parallel_backend,
            _NUMBA_AVAILABLE,
        )
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure dfextensions is in your PYTHONPATH")
        sys.exit(1)
    
    # Create timestamped run directory (matching BF convention)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = run_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Phase 12.11: GroupBy Fits Benchmark")
    print(f"# Engines: {engines}")
    print(f"# Scenarios: {scenarios}")
    print(f"# n_jobs: {n_jobs_list}")
    print(f"# Warmup: {args.warmup}, Runs: {args.runs}")
    print(f"# Profile top N: {args.profile_top_n}")
    print(f"# Numba available: {_NUMBA_AVAILABLE}")
    print(f"# Output: {run_dir}")
    print(f"{'#'*60}")
    
    # Get benchmark suite
    suite = get_benchmark_suite(engines, scenarios, n_jobs_list)
    
    # Run benchmarks
    results = []
    
    for i, bench_config in enumerate(suite, 1):
        engine = bench_config['engine']
        scenario = bench_config['scenario']
        n_jobs = bench_config['n_jobs']
        scenario_config = bench_config['scenario_config']
        
        print(f"\n[{i}/{len(suite)}] Running {bench_config['id']}...")
        
        # Generate test data (excluded from profiling)
        print(f"    Generating test data ({scenario_config['n_rows']:,} rows)...")
        df = create_test_data(
            n_rows=scenario_config['n_rows'],
            n_groups=scenario_config['n_groups'],
            n_fit_cols=args.n_fits,
        )
        print(f"    Data ready: {len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
        
        config = BenchmarkConfig(
            engine=engine,
            scenario=scenario,
            n_jobs=n_jobs,
            n_fits=args.n_fits,
            profiles_dir=profiles_dir,
            profile_top_n=args.profile_top_n,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
        
        try:
            if engine == 'v4':
                result = run_v4_benchmark(
                    df, config,
                    make_parallel_fit_v4_func=make_parallel_fit_v4,
                    numba_available=_NUMBA_AVAILABLE,
                    profile_function_full_func=profile_function_full,
                    get_peak_rss_mb_func=get_peak_rss_mb,
                )
            elif engine == 'v5':
                result = run_v5_benchmark(
                    df, config,
                    make_parallel_fit_v5_func=make_parallel_fit_v5,
                    select_parallel_backend_func=_select_parallel_backend,
                    numba_available=_NUMBA_AVAILABLE,
                    profile_function_full_func=profile_function_full,
                    get_peak_rss_mb_func=get_peak_rss_mb,
                )
            else:
                print(f"    Skipping {engine} (not implemented)")
                continue
            
            results.append(result)
            
            if result['status'] == 'OK':
                profile_status = "profiled" if result.get('profile', {}).get('cpu_enabled') else "wall-time"
                print(f"    Time: {result['time_s']:.3f}s ± {result['time_std_s']:.3f}s")
                print(f"    RSS: {result['peak_rss_mb']:.0f} MB | {profile_status}")
                print(f"    Backend: {result.get('backend', {}).get('selected_backend', 'unknown')}")
            else:
                print(f"    Status: {result['status']}")
                if result.get('error_message'):
                    print(f"    Error: {result['error_message']}")
                    
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "id": bench_config['id'],
                "name": BENCHMARK_NAME,
                "scenario": scenario,
                "status": "FAILED",
                "time_s": 0,
                "time_std_s": 0,
                "n_runs": 0,
                "peak_rss_mb": 0,
                "error_message": str(e),
                "params": {
                    "engine": engine,
                    "scenario": scenario,
                    "n_jobs": n_jobs,
                },
            })
    
    # Save results (BF schema)
    run_data = {
        "meta": {
            "schema_version": 1,
            "runner_version": "1.0.0",
            "timestamp": timestamp,
            "suite": "groupby_fits",
            "subproject": "groupby_regression",
            "warmup_runs": args.warmup,
            "n_runs": args.runs,
            "profile_top_n": args.profile_top_n,
            "tool_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "numba": "available" if _NUMBA_AVAILABLE else "not installed",
            },
        },
        "summary": {
            "n_benchmarks": len(results),
            "n_passed": sum(1 for r in results if r.get('status') == "OK"),
            "n_failed": sum(1 for r in results if r.get('status') == "FAILED"),
            "n_skipped": sum(1 for r in results if r.get('status') == "SKIPPED"),
        },
        "benchmarks": results,
        "alarms": [],
    }
    
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(run_data, indent=2, default=str))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        status_icon = "✓" if r.get('status') == "OK" else ("⊘" if r.get('status') == "SKIPPED" else "✗")
        time_str = f"{r.get('time_s', 0):.3f}s" if r.get('status') == "OK" else "N/A"
        print(f"  {status_icon} {r['id']}: {time_str}, {r.get('peak_rss_mb', 0):.0f} MB")
    
    print(f"\nResults: {results_path}")
    print(f"Profiles: {profiles_dir}")
    
    # Return exit code
    n_failed = sum(1 for r in results if r.get('status') == "FAILED")
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
