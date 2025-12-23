"""
Benchmark Framework v1.0 — Regression Detection

Detect performance regressions against historical baselines.

Phase 12.10.BF-Analysis: Automated regression detection.

Algorithm:
    1. Load history for same env_id (exclude profile runs)
    2. Calculate baseline using median of last N runs
    3. Require minimum 3 samples before triggering alarms (MIN_BASELINE_SAMPLES)
    4. Compare current result against baseline
    5. Trigger alarm if threshold exceeded

Thresholds:
    - Time: 10% (DEFAULT_TIME_THRESHOLD)
    - Memory: 15% (DEFAULT_MEMORY_THRESHOLD) — higher due to RSS variance
    - Minimum samples: 3 (MIN_BASELINE_SAMPLES) — prevents false positives

Usage:
    from dfextensions.benchmarks.regression import detect_regressions
    
    alarms = detect_regressions(
        current_run=run,
        history_df=df,
        time_threshold=0.10,
        memory_threshold=0.15,
    )
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .schema import (
    BenchmarkRun,
    BenchmarkResult,
    Alarm,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
)
from .history import (
    filter_by_env,
    filter_passed_only,
    get_baseline,
)

# Minimum number of baseline samples required before triggering regression alarms.
# This prevents false positives during the first few runs of a new benchmark.
MIN_BASELINE_SAMPLES = 3


# =============================================================================
# REGRESSION RESULT
# =============================================================================

@dataclass
class RegressionResult:
    """Result of regression check for a single benchmark."""
    benchmark_id: str
    
    # Current values
    current_time_s: float
    current_rss_mb: float
    
    # Baseline values (None if no baseline)
    baseline_time_s: Optional[float] = None
    baseline_rss_mb: Optional[float] = None
    baseline_samples: int = 0
    
    # Regression status
    time_regression: bool = False
    time_change_pct: float = 0.0
    
    memory_regression: bool = False
    memory_change_pct: float = 0.0
    
    # Overall
    has_regression: bool = False
    has_baseline: bool = False
    
    def to_alarm(self, name: str, scenario: str, n_jobs: int) -> Optional[Alarm]:
        """Convert to Alarm if regression detected."""
        if not self.has_regression:
            return None
        
        # Report the worse regression (time or memory)
        if self.time_regression and (not self.memory_regression or 
                                      self.time_change_pct >= self.memory_change_pct):
            return Alarm(
                benchmark=name,
                scenario=scenario,
                param_n_jobs=n_jobs,
                metric="time_s",
                current=self.current_time_s,
                baseline=self.baseline_time_s,
                baseline_type="vs_median",
                change_pct=self.time_change_pct,
            )
        else:
            return Alarm(
                benchmark=name,
                scenario=scenario,
                param_n_jobs=n_jobs,
                metric="peak_rss_mb",
                current=self.current_rss_mb,
                baseline=self.baseline_rss_mb,
                baseline_type="vs_median",
                change_pct=self.memory_change_pct,
            )


# =============================================================================
# SINGLE BENCHMARK REGRESSION CHECK
# =============================================================================

def check_regression(
    current: BenchmarkResult,
    history_df: pd.DataFrame,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    memory_threshold: float = DEFAULT_MEMORY_THRESHOLD,
    n_baseline_runs: int = 20,
) -> RegressionResult:
    """
    Check for regression in a single benchmark.
    
    Parameters:
        current: Current benchmark result
        history_df: Historical data (pre-filtered by env_id)
        time_threshold: Threshold for time regression (0.10 = 10%)
        memory_threshold: Threshold for memory regression (0.15 = 15%)
        n_baseline_runs: Number of runs to use for baseline
    
    Returns:
        RegressionResult with comparison details
    """
    result = RegressionResult(
        benchmark_id=current.id,
        current_time_s=current.time_s,
        current_rss_mb=current.peak_rss_mb,
    )
    
    # Get baseline
    baseline = get_baseline(
        history_df,
        current.id,
        n_runs=n_baseline_runs,
        method="median",
    )
    
    if baseline is None or baseline["n_samples"] == 0:
        # No baseline available
        result.has_baseline = False
        return result
    
    # Require minimum samples before triggering regression alarms
    # This prevents false positives during the first few runs
    if baseline["n_samples"] < MIN_BASELINE_SAMPLES:
        result.has_baseline = False  # Treat as insufficient history
        result.baseline_samples = baseline["n_samples"]
        return result
    
    result.has_baseline = True
    result.baseline_time_s = baseline["time_s"]
    result.baseline_rss_mb = baseline["peak_rss_mb"]
    result.baseline_samples = baseline["n_samples"]
    
    # Check time regression
    if result.baseline_time_s > 0:
        time_change = (result.current_time_s - result.baseline_time_s) / result.baseline_time_s
        result.time_change_pct = time_change * 100
        result.time_regression = time_change > time_threshold
    
    # Check memory regression
    if result.baseline_rss_mb > 0:
        memory_change = (result.current_rss_mb - result.baseline_rss_mb) / result.baseline_rss_mb
        result.memory_change_pct = memory_change * 100
        result.memory_regression = memory_change > memory_threshold
    
    result.has_regression = result.time_regression or result.memory_regression
    
    return result


# =============================================================================
# FULL RUN REGRESSION CHECK
# =============================================================================

def detect_regressions(
    current_run: BenchmarkRun,
    history_df: pd.DataFrame,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    memory_threshold: float = DEFAULT_MEMORY_THRESHOLD,
    n_baseline_runs: int = 20,
    filter_env: bool = True,
) -> Tuple[List[Alarm], List[RegressionResult]]:
    """
    Detect regressions for all benchmarks in a run.
    
    Parameters:
        current_run: Current benchmark run
        history_df: Historical data
        time_threshold: Threshold for time regression
        memory_threshold: Threshold for memory regression
        n_baseline_runs: Number of runs to use for baseline
        filter_env: Filter history to same env_id (default: True)
    
    Returns:
        (alarms, results):
            - alarms: List of Alarm objects for detected regressions
            - results: List of RegressionResult for all benchmarks
    """
    # Filter history by environment
    if filter_env:
        history_df = filter_by_env(history_df, current_run.meta.env_id)
    
    # Filter to passed runs only
    history_df = filter_passed_only(history_df)
    
    alarms = []
    results = []
    
    for bench in current_run.benchmarks:
        # Skip failed benchmarks
        if bench.status != "OK":
            continue
        
        # Check for regression
        result = check_regression(
            current=bench,
            history_df=history_df,
            time_threshold=time_threshold,
            memory_threshold=memory_threshold,
            n_baseline_runs=n_baseline_runs,
        )
        results.append(result)
        
        # Create alarm if regression detected
        if result.has_regression:
            n_jobs = bench.params.get("n_jobs", 1)
            alarm = result.to_alarm(bench.name, bench.scenario, n_jobs)
            if alarm is not None:
                alarms.append(alarm)
    
    return alarms, results


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_regression_summary(
    results: List[RegressionResult],
    alarms: List[Alarm],
):
    """Print regression summary to console."""
    n_checked = len(results)
    n_with_baseline = sum(1 for r in results if r.has_baseline)
    n_regressions = len(alarms)
    
    print()
    print("─" * 68)
    print("REGRESSION CHECK")
    print("─" * 68)
    print(f"  Benchmarks checked:    {n_checked}")
    print(f"  With baseline:         {n_with_baseline}")
    print(f"  Regressions detected:  {n_regressions}")
    print()
    
    if n_regressions == 0:
        print("  ✓ No regressions detected")
    else:
        print("  ⚠ REGRESSIONS DETECTED:")
        print()
        for alarm in alarms:
            print(f"    • {alarm.benchmark}:{alarm.scenario}:n_jobs={alarm.param_n_jobs}")
            print(f"      {alarm.metric}: {alarm.current:.3f} vs {alarm.baseline:.3f} (+{alarm.change_pct:.1f}%)")
            print()
    
    print("─" * 68)


def format_regression_report(
    results: List[RegressionResult],
    alarms: List[Alarm],
) -> str:
    """Format regression report as string."""
    lines = []
    
    n_checked = len(results)
    n_with_baseline = sum(1 for r in results if r.has_baseline)
    n_regressions = len(alarms)
    
    lines.append("=" * 68)
    lines.append("REGRESSION REPORT")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"Benchmarks checked:    {n_checked}")
    lines.append(f"With baseline:         {n_with_baseline}")
    lines.append(f"Regressions detected:  {n_regressions}")
    lines.append("")
    
    if n_regressions == 0:
        lines.append("✓ No regressions detected")
    else:
        lines.append("⚠ REGRESSIONS:")
        lines.append("")
        
        for alarm in alarms:
            lines.append(f"  {alarm.benchmark}:{alarm.scenario}:n_jobs={alarm.param_n_jobs}")
            lines.append(f"    {alarm.metric}: {alarm.current:.3f} → {alarm.baseline:.3f} (+{alarm.change_pct:.1f}%)")
            lines.append("")
    
    lines.append("")
    lines.append("DETAILED RESULTS:")
    lines.append("-" * 68)
    
    for r in results:
        status = "⚠ REGRESS" if r.has_regression else "✓ OK" if r.has_baseline else "○ NO_BASELINE"
        lines.append(f"{status:12} {r.benchmark_id}")
        
        if r.has_baseline:
            lines.append(f"             time:   {r.current_time_s:.3f}s vs {r.baseline_time_s:.3f}s ({r.time_change_pct:+.1f}%)")
            lines.append(f"             memory: {r.current_rss_mb:.0f}MB vs {r.baseline_rss_mb:.0f}MB ({r.memory_change_pct:+.1f}%)")
        else:
            lines.append(f"             time:   {r.current_time_s:.3f}s (no baseline)")
            lines.append(f"             memory: {r.current_rss_mb:.0f}MB (no baseline)")
        lines.append("")
    
    lines.append("=" * 68)
    
    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "MIN_BASELINE_SAMPLES",
    # Results
    "RegressionResult",
    # Detection
    "check_regression",
    "detect_regressions",
    # Output
    "print_regression_summary",
    "format_regression_report",
]
