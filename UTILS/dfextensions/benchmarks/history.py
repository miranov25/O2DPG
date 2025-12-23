"""
Benchmark Framework v1.0 — History Loading

Load and filter benchmark history for regression detection.

Phase 12.10.BF-Analysis: Historical data management.

Usage:
    from dfextensions.benchmarks.history import load_history, filter_by_env
    
    # Load all history for a subproject
    df = load_history("groupby_regression")
    
    # Filter to same environment
    df_filtered = filter_by_env(df, current_env_id)
    
    # Get baseline for specific benchmark
    baseline = get_baseline(df_filtered, "v5_batch_fit:S2:n_jobs=4")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd

from .schema import (
    get_benchmark_prefix,
    BenchmarkRun,
    SCHEMA_VERSION,
)


# =============================================================================
# HISTORY LOADING
# =============================================================================

def discover_runs(
    subproject: str,
    prefix: Optional[Path] = None,
) -> List[Path]:
    """
    Discover all benchmark runs for a subproject.
    
    Parameters:
        subproject: Subproject name (e.g., "groupby_regression")
        prefix: Override benchmark prefix (default: $BENCHMARK_PREFIX)
    
    Returns:
        List of paths to results.json files, sorted by timestamp (newest first)
    """
    if prefix is None:
        prefix = get_benchmark_prefix()
    
    prefix = Path(prefix)
    if not prefix.exists():
        return []
    
    results = []
    
    # Walk through timestamp directories
    for ts_dir in prefix.iterdir():
        if not ts_dir.is_dir():
            continue
        
        # Check for subproject directory
        subproject_dir = ts_dir / subproject
        if not subproject_dir.exists():
            continue
        
        # Check for results.json
        results_file = subproject_dir / "results.json"
        if results_file.exists():
            results.append(results_file)
    
    # Sort by timestamp (directory name), newest first
    results.sort(key=lambda p: p.parent.parent.name, reverse=True)
    
    return results


def load_run(path: Union[Path, str]) -> Optional[BenchmarkRun]:
    """
    Load a single benchmark run from JSON.
    
    Returns None if file is invalid or incompatible schema version.
    """
    try:
        run = BenchmarkRun.load(path)
        
        # Check schema version
        if run.meta.schema_version != SCHEMA_VERSION:
            return None
        
        return run
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Invalid JSON or missing required fields
        return None


def load_history(
    subproject: str,
    prefix: Optional[Path] = None,
    max_runs: int = 100,
    exclude_profile: bool = True,
) -> pd.DataFrame:
    """
    Load benchmark history as a DataFrame.
    
    Parameters:
        subproject: Subproject name
        prefix: Override benchmark prefix
        max_runs: Maximum number of runs to load (newest first)
        exclude_profile: Exclude run_mode="profile" (tracemalloc overhead)
    
    Returns:
        DataFrame with columns:
            - run_id, timestamp, commit, env_id, run_mode, suite
            - benchmark_id, name, scenario, n_jobs
            - time_s, time_std_s, peak_rss_mb, throughput_rows_per_sec
            - status
    """
    runs = discover_runs(subproject, prefix)[:max_runs]
    
    rows = []
    for path in runs:
        run = load_run(path)
        if run is None:
            continue
        
        # Skip profile runs if requested
        if exclude_profile and run.meta.run_mode == "profile":
            continue
        
        # Extract run-level metadata
        meta = {
            "run_id": run.meta.run_id,
            "timestamp": run.meta.timestamp,
            "commit": run.meta.commit,
            "branch": run.meta.branch,
            "env_id": run.meta.env_id,
            "run_mode": run.meta.run_mode,
            "suite": run.meta.suite,
            "hostname": run.meta.hostname,
        }
        
        # Add each benchmark result as a row
        for bench in run.benchmarks:
            row = {
                **meta,
                "benchmark_id": bench.id,
                "name": bench.name,
                "scenario": bench.scenario,
                "n_jobs": bench.params.get("n_jobs", 1),
                "time_s": bench.time_s,
                "time_std_s": bench.time_std_s,
                "peak_rss_mb": bench.peak_rss_mb,
                "throughput_rows_per_sec": bench.throughput_rows_per_sec,
                "status": bench.status,
            }
            rows.append(row)
    
    if not rows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "run_id", "timestamp", "commit", "branch", "env_id",
            "run_mode", "suite", "hostname",
            "benchmark_id", "name", "scenario", "n_jobs",
            "time_s", "time_std_s", "peak_rss_mb", "throughput_rows_per_sec",
            "status",
        ])
    
    df = pd.DataFrame(rows)
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by timestamp descending
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    
    return df


# =============================================================================
# FILTERING
# =============================================================================

def filter_by_env(
    df: pd.DataFrame,
    env_id: str,
) -> pd.DataFrame:
    """
    Filter history to same environment.
    
    Parameters:
        df: History DataFrame
        env_id: Current environment ID
    
    Returns:
        Filtered DataFrame
    """
    return df[df["env_id"] == env_id].copy()


def filter_by_benchmark(
    df: pd.DataFrame,
    benchmark_id: str,
) -> pd.DataFrame:
    """
    Filter history to specific benchmark.
    
    Parameters:
        df: History DataFrame
        benchmark_id: Benchmark ID (e.g., "v5_batch_fit:S2:n_jobs=4")
    
    Returns:
        Filtered DataFrame
    """
    return df[df["benchmark_id"] == benchmark_id].copy()


def filter_passed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only passed benchmarks."""
    return df[df["status"] == "OK"].copy()


# =============================================================================
# BASELINE CALCULATION
# =============================================================================

def get_baseline(
    df: pd.DataFrame,
    benchmark_id: str,
    n_runs: int = 20,
    method: str = "median",
) -> Optional[dict]:
    """
    Calculate baseline statistics for a benchmark.
    
    Parameters:
        df: History DataFrame (should be pre-filtered by env_id)
        benchmark_id: Benchmark ID
        n_runs: Number of most recent runs to use
        method: "median" (default) or "mean"
    
    Returns:
        Dict with baseline statistics, or None if insufficient history:
            - time_s: Baseline time
            - time_std_s: Standard deviation of times
            - peak_rss_mb: Baseline memory
            - rss_std_mb: Standard deviation of memory
            - n_samples: Number of samples used
            - method: Method used ("median" or "mean")
    """
    # Filter to specific benchmark and passed runs
    bench_df = filter_by_benchmark(df, benchmark_id)
    bench_df = filter_passed_only(bench_df)
    
    if len(bench_df) == 0:
        return None
    
    # Take most recent n_runs
    bench_df = bench_df.head(n_runs)
    
    n_samples = len(bench_df)
    
    if method == "median":
        time_baseline = bench_df["time_s"].median()
        rss_baseline = bench_df["peak_rss_mb"].median()
    else:
        time_baseline = bench_df["time_s"].mean()
        rss_baseline = bench_df["peak_rss_mb"].mean()
    
    return {
        "time_s": float(time_baseline),
        "time_std_s": float(bench_df["time_s"].std()) if n_samples > 1 else 0.0,
        "peak_rss_mb": float(rss_baseline),
        "rss_std_mb": float(bench_df["peak_rss_mb"].std()) if n_samples > 1 else 0.0,
        "n_samples": n_samples,
        "method": method,
    }


def get_all_baselines(
    df: pd.DataFrame,
    n_runs: int = 20,
    method: str = "median",
) -> dict:
    """
    Calculate baselines for all benchmarks in history.
    
    Parameters:
        df: History DataFrame (should be pre-filtered by env_id)
        n_runs: Number of most recent runs per benchmark
        method: "median" or "mean"
    
    Returns:
        Dict mapping benchmark_id → baseline dict
    """
    baselines = {}
    
    for benchmark_id in df["benchmark_id"].unique():
        baseline = get_baseline(df, benchmark_id, n_runs=n_runs, method=method)
        if baseline is not None:
            baselines[benchmark_id] = baseline
    
    return baselines


# =============================================================================
# HISTORY SUMMARY
# =============================================================================

def summarize_history(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for history DataFrame.
    
    Returns:
        Dict with summary info:
            - n_runs: Number of unique runs
            - n_benchmarks: Number of unique benchmark IDs
            - date_range: (earliest, latest) timestamps
            - environments: List of unique env_ids
            - pass_rate: Fraction of passed benchmarks
    """
    if len(df) == 0:
        return {
            "n_runs": 0,
            "n_benchmarks": 0,
            "date_range": (None, None),
            "environments": [],
            "pass_rate": 0.0,
        }
    
    return {
        "n_runs": df["run_id"].nunique(),
        "n_benchmarks": df["benchmark_id"].nunique(),
        "date_range": (
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ),
        "environments": df["env_id"].unique().tolist(),
        "pass_rate": (df["status"] == "OK").mean(),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Discovery
    "discover_runs",
    "load_run",
    "load_history",
    # Filtering
    "filter_by_env",
    "filter_by_benchmark",
    "filter_passed_only",
    # Baselines
    "get_baseline",
    "get_all_baselines",
    # Summary
    "summarize_history",
]
