"""
Benchmark Framework v1.0 — AliasDataFrame Integration

Phase 12.14c.GB D1: Load benchmark results into AliasDataFrame with subframes.

Public API:
    load_benchmark_adf(subproject, max_runs, ...) -> AliasDataFrame
    compute_benchmark_statistics(subproject, baseline) -> pd.DataFrame

Private helpers:
    _load_history_with_dirs(...)
    _extract_cpu_profiles(...)
    _extract_memory_stats(...)

Key constraints (from review):
    P0-2: Use fcn_list for TopCPU sorting (cumulative time order)
    P0-3: _run_dir from history metadata, no guessing
    Graceful degradation: try/except in extraction loops
    Performance bounded: profile_limit, max_runs parameters

Usage:
    from dfextensions.benchmarks.benchmark_adf import (
        load_benchmark_adf,
        compute_benchmark_statistics,
    )
    
    # Load benchmark history as AliasDataFrame
    adf = load_benchmark_adf("groupby_regression", max_runs=20)
    
    # Access main frame
    adf.df.head()
    
    # Access subframes
    adf.subframes['TopCPU'].df   # CPU profile data
    adf.subframes['TopMemory'].df  # Memory statistics
    
    # Compute noise statistics
    stats = compute_benchmark_statistics("groupby_regression", baseline="7d")
"""

import logging
import pstats
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from .schema import get_benchmark_prefix
from .history import discover_runs, load_run

if TYPE_CHECKING:
    from dfextensions.AliasDataFrame import AliasDataFrame

logger = logging.getLogger(__name__)


# =============================================================================
# LAZY IMPORT (avoid circular imports)
# =============================================================================

_AliasDataFrame = None


def _get_alias_dataframe():
    """Lazy import AliasDataFrame to avoid circular imports."""
    global _AliasDataFrame
    if _AliasDataFrame is None:
        from dfextensions.AliasDataFrame import AliasDataFrame
        _AliasDataFrame = AliasDataFrame
    return _AliasDataFrame


# =============================================================================
# STEP 1: _load_history_with_dirs (P0-3 compliant)
# =============================================================================

def _load_history_with_dirs(
    subproject: str,
    prefix: Optional[Path] = None,
    max_runs: Optional[int] = None,
    exclude_profile: bool = True,
) -> pd.DataFrame:
    """
    Load benchmark history with _run_dir column for profile path resolution.
    
    P0-3 Requirement: _run_dir derived from history, not guessed.
    
    Parameters
    ----------
    subproject : str
        Subproject name (e.g., "groupby_regression")
    prefix : Path, optional
        Override benchmark prefix
    max_runs : int, optional
        Maximum runs to load (None = all)
    exclude_profile : bool, default True
        Exclude run_mode="profile" runs
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns including _run_dir and profile_path
    """
    run_paths = discover_runs(subproject, prefix)
    if max_runs:
        run_paths = run_paths[:max_runs]
    
    rows = []
    for path in run_paths:
        try:
            run = load_run(path)
            if run is None:
                continue
            
            # Skip profile runs if requested
            if exclude_profile and run.meta.run_mode == "profile":
                continue
            
            # P0-3: _run_dir is the directory containing results.json
            run_dir = str(path.parent)
            
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
                "_run_dir": run_dir,  # P0-3: For profile path resolution
            }
            
            # Add each benchmark result as a row
            for bench in run.benchmarks:
                row = {
                    **meta,
                    "benchmark_id": bench.id,
                    "name": bench.name,
                    "scenario": bench.scenario,
                    "time_s": bench.time_s,
                    "wall_time_s": bench.wall_time_s,
                    "time_std_s": bench.time_std_s,
                    "n_runs": bench.n_runs,
                    "peak_rss_mb": bench.peak_rss_mb,
                    "throughput_rows_per_sec": bench.throughput_rows_per_sec,
                    "status": bench.status,
                    "profile_path": bench.profile_path,
                    "profile_note": getattr(bench, 'profile_note', None),
                    "params": bench.params,  # Keep as dict for later parsing
                }
                rows.append(row)
        
        except Exception as e:
            # Graceful degradation: skip corrupt files
            logger.debug(f"Skipped {path}: {e}")
            continue
    
    if not rows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "run_id", "timestamp", "commit", "branch", "env_id",
            "run_mode", "suite", "hostname", "_run_dir",
            "benchmark_id", "name", "scenario",
            "time_s", "wall_time_s", "time_std_s", "n_runs",
            "peak_rss_mb", "throughput_rows_per_sec", "status",
            "profile_path", "profile_note", "params",
        ])
    
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    
    return df


# =============================================================================
# STEP 2: load_benchmark_adf (Main loader - PUBLIC API)
# =============================================================================

def load_benchmark_adf(
    subproject: str,
    max_runs: Optional[int] = None,
    history_range: Optional[str] = None,
    include_profiles: bool = True,
    profile_limit: int = 10,
) -> "AliasDataFrame":
    """
    Load benchmark results into AliasDataFrame with subframes.
    
    Parameters
    ----------
    subproject : str
        Subproject name (e.g., "groupby_regression")
    max_runs : int, optional
        Maximum number of runs to load
    history_range : str, optional
        Time range filter (e.g., "7d", "30d")
    include_profiles : bool, default True
        Include TopCPU subframe from .prof files
    profile_limit : int, default 10
        Max functions to extract per profile
    
    Returns
    -------
    AliasDataFrame
        AliasDataFrame with:
        - Main frame: benchmark results
        - TopCPU subframe: CPU profile data (if profiles exist)
        - TopMemory subframe: Memory statistics
    
    Example
    -------
    >>> adf = load_benchmark_adf("groupby_regression", max_runs=20)
    >>> adf.df.head()  # Main benchmark results
    >>> adf.subframes['TopCPU'].df  # CPU profile data
    """
    AliasDataFrame = _get_alias_dataframe()
    
    # Step 1: Load history with _run_dir
    df = _load_history_with_dirs(subproject, max_runs=max_runs)
    
    if df.empty:
        logger.warning(f"No benchmark history found for {subproject}")
        return AliasDataFrame(df)
    
    # Apply history_range filter
    if history_range:
        df = _filter_by_range(df, history_range)
    
    # Parse params column into individual columns
    df = _flatten_params(df)
    
    # Create AliasDataFrame
    adf = AliasDataFrame(df)
    
    # Step 3: Extract memory stats (TopMemory subframe)
    memory_df = _extract_memory_stats(df)
    if not memory_df.empty:
        adf.register_subframe("TopMemory", AliasDataFrame(memory_df), index_columns=["benchmark_id"])
    
    # Step 5: Extract CPU profiles (TopCPU subframe)
    if include_profiles:
        cpu_df = _extract_cpu_profiles(df, profile_limit=profile_limit)
        if not cpu_df.empty:
            adf.register_subframe("TopCPU", AliasDataFrame(cpu_df), index_columns=["benchmark_id"])
    
    return adf


def _filter_by_range(df: pd.DataFrame, range_str: str) -> pd.DataFrame:
    """
    Filter DataFrame to time range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    range_str : str
        Range string like "7d" (days) or "24h" (hours)
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not range_str:
        return df
    
    # Parse range string
    try:
        if range_str.endswith('d'):
            days = int(range_str[:-1])
            cutoff = datetime.now() - timedelta(days=days)
        elif range_str.endswith('h'):
            hours = int(range_str[:-1])
            cutoff = datetime.now() - timedelta(hours=hours)
        else:
            logger.warning(f"Unknown range format: {range_str}, expected '7d' or '24h'")
            return df
        
        # Make cutoff timezone-aware if timestamps are
        if df["timestamp"].dt.tz is not None:
            cutoff = cutoff.replace(tzinfo=df["timestamp"].dt.tz)
        
        return df[df["timestamp"] >= cutoff].copy()
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse range '{range_str}': {e}")
        return df


def _flatten_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract common params into individual columns.
    
    Handles missing params gracefully (robust params handling from P1).
    """
    if "params" not in df.columns:
        return df
    
    df = df.copy()
    
    # Extract common params with safe access
    for key in ["n_jobs", "n_rows", "n_groups", "parallel_backend"]:
        df[key] = df["params"].apply(
            lambda p: _safe_param_get(p, key)
        )
    
    return df


def _safe_param_get(params, key, default=None):
    """Safely extract param from dict/JSON string/None."""
    if params is None:
        return default
    
    if isinstance(params, str):
        try:
            import json
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return default
    
    if isinstance(params, dict):
        return params.get(key, default)
    
    return default


# =============================================================================
# STEP 3: _extract_memory_stats (TopMemory subframe)
# =============================================================================

def _extract_memory_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract memory statistics for TopMemory subframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame with memory columns
    
    Returns
    -------
    pd.DataFrame
        Memory statistics with columns:
        benchmark_id, run_id, peak_rss_mb, timestamp, name, scenario, n_jobs
    """
    if df.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    memory_cols = [
        "benchmark_id", "run_id", "peak_rss_mb", "timestamp",
        "name", "scenario"
    ]
    available_cols = [c for c in memory_cols if c in df.columns]
    
    if "peak_rss_mb" not in df.columns:
        logger.debug("peak_rss_mb not in DataFrame; TopMemory skipped")
        return pd.DataFrame()
    
    memory_df = df[available_cols].copy()
    
    # Add n_jobs if available
    if "n_jobs" in df.columns:
        memory_df["n_jobs"] = df["n_jobs"]
    
    return memory_df


# =============================================================================
# STEP 4: compute_benchmark_statistics (Noise analysis - PUBLIC API)
# =============================================================================

def compute_benchmark_statistics(
    subproject: str,
    baseline: str = "7d",
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Compute noise statistics for benchmarks.
    
    Parameters
    ----------
    subproject : str
        Subproject name
    baseline : str, default "7d"
        Time range for statistics (e.g., "7d", "30d")
    min_samples : int, default 3
        Minimum samples required for statistics
    
    Returns
    -------
    pd.DataFrame
        Statistics with columns:
        benchmark_id, mean_time_s, std_time_s, cv_pct, n_samples,
        threshold_warn, threshold_alarm, high_noise
    
    Example
    -------
    >>> stats = compute_benchmark_statistics("groupby_regression", "7d")
    >>> stats[stats["high_noise"]]  # Benchmarks with high variance
    """
    df = _load_history_with_dirs(subproject)
    
    if df.empty:
        return pd.DataFrame(columns=[
            "benchmark_id", "mean_time_s", "std_time_s", "cv_pct",
            "n_samples", "threshold_warn", "threshold_alarm", "high_noise"
        ])
    
    # Filter by range
    df = _filter_by_range(df, baseline)
    
    # Filter to passed only
    df = df[df["status"] == "OK"]
    
    if df.empty:
        return pd.DataFrame(columns=[
            "benchmark_id", "mean_time_s", "std_time_s", "cv_pct",
            "n_samples", "threshold_warn", "threshold_alarm", "high_noise"
        ])
    
    # Group by benchmark_id and compute statistics
    stats = []
    for bid, group in df.groupby("benchmark_id"):
        n_samples = len(group)
        if n_samples < min_samples:
            continue
        
        mean_time = group["time_s"].mean()
        std_time = group["time_s"].std()
        cv_pct = (std_time / mean_time * 100) if mean_time > 0 else 0.0
        
        # Threshold calculation (3-sigma for warn, 5-sigma for alarm)
        threshold_warn = mean_time + 3 * std_time
        threshold_alarm = mean_time + 5 * std_time
        
        # High noise flag (CV > 10%)
        high_noise = cv_pct > 10.0
        
        stats.append({
            "benchmark_id": bid,
            "mean_time_s": float(mean_time),
            "std_time_s": float(std_time),
            "cv_pct": float(cv_pct),
            "n_samples": int(n_samples),
            "threshold_warn": float(threshold_warn),
            "threshold_alarm": float(threshold_alarm),
            "high_noise": bool(high_noise),
        })
    
    return pd.DataFrame(stats)


# =============================================================================
# STEP 5: _extract_cpu_profiles (TopCPU subframe, P0-2 compliant)
# =============================================================================

def _extract_cpu_profiles(
    df: pd.DataFrame,
    profile_limit: int = 10,
) -> pd.DataFrame:
    """
    Extract CPU profile data from .prof files.
    
    P0-2 Requirement: Use fcn_list for proper cumulative-time sorting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with _run_dir and profile_path columns
    profile_limit : int, default 10
        Max functions to extract per profile
    
    Returns
    -------
    pd.DataFrame
        Profile data with columns:
        benchmark_id, run_id, function, filename, lineno,
        ncalls, tottime_s, cumtime_s, rank
    """
    if df.empty:
        return pd.DataFrame()
    
    if "_run_dir" not in df.columns:
        logger.warning("_run_dir not in DataFrame; profile extraction disabled")
        return pd.DataFrame()
    
    rows = []
    
    for _, row in df.iterrows():
        profile_path = row.get("profile_path")
        if not profile_path:
            continue
        
        run_dir = row.get("_run_dir")
        if not run_dir:
            continue
        
        # Resolve full path
        full_path = Path(run_dir) / profile_path
        if not full_path.exists():
            logger.debug(f"Profile not found: {full_path}")
            continue
        
        try:
            # Load profile
            stats = pstats.Stats(str(full_path))
            stats.sort_stats('cumulative')
            
            # P0-2: Use fcn_list for correctly sorted order
            # fcn_list is populated after sort_stats() with functions in sorted order
            for rank, func_key in enumerate(stats.fcn_list[:profile_limit], 1):
                # func_key = (filename, lineno, funcname)
                filename, lineno, funcname = func_key
                
                # Get stats for this function
                # stats.stats[func_key] = (ncalls, totcalls, tottime, cumtime, callers)
                func_stats = stats.stats.get(func_key)
                if not func_stats:
                    continue
                
                ncalls, totcalls, tottime, cumtime, callers = func_stats
                
                rows.append({
                    "benchmark_id": row.get("benchmark_id"),
                    "run_id": row.get("run_id"),
                    "function": funcname,
                    "filename": filename,
                    "lineno": lineno,
                    "ncalls": int(ncalls),
                    "tottime_s": float(tottime),
                    "cumtime_s": float(cumtime),
                    "rank": rank,
                })
        
        except Exception as e:
            # Graceful degradation: log and continue
            logger.debug(f"Profile extraction failed for {full_path}: {e}")
            continue
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "load_benchmark_adf",
    "compute_benchmark_statistics",
]
