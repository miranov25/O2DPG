"""
Benchmark Framework v1.0 — Runner

CLI entry point for running benchmarks.

Phase 12.10.BF: Standardized benchmark execution.
Phase 12.14b.GB: Added kernel and memory benchmark discovery.

Usage:
    # Quick suite (default)
    python -m dfextensions.benchmarks.runner --subproject groupby_regression
    
    # Release suite
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --suite release
    
    # With profiling (tracemalloc + top allocations)
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --profile
    
    # Check regressions only (no new run)
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --check-only
    
Exit Codes:
    0 - All benchmarks passed, no regressions
    1 - Regression detected
    2 - Benchmark execution error
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable


# =============================================================================
# PATH SETUP FOR PARALLEL WORKERS
# =============================================================================

def setup_pythonpath():
    """
    Ensure PYTHONPATH includes the UTILS directory for joblib workers.
    
    When joblib spawns worker processes, they need to be able to import
    modules like groupby_regression. This sets PYTHONPATH so workers
    can find all required modules.
    """
    # Find UTILS directory (parent of dfextensions)
    this_file = Path(__file__).resolve()
    utils_dir = this_file.parent.parent.parent  # benchmarks -> dfextensions -> UTILS
    
    if utils_dir.exists():
        utils_str = str(utils_dir)
        
        # Add to sys.path if not already there
        if utils_str not in sys.path:
            sys.path.insert(0, utils_str)
        
        # Set PYTHONPATH for child processes (joblib workers)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if utils_str not in current_pythonpath:
            if current_pythonpath:
                os.environ["PYTHONPATH"] = f"{utils_str}:{current_pythonpath}"
            else:
                os.environ["PYTHONPATH"] = utils_str


# Run path setup on import
setup_pythonpath()


from .schema import (
    SCHEMA_VERSION,
    RUNNER_VERSION,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_N_RUNS,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
    BenchmarkResult,
    RunMeta,
    RunSummary,
    BenchmarkRun,
    get_run_output_dir,
    get_results_path,
    get_alarms_path,
    validate_run,
)
from .profiler import (
    check_platform_support,
    get_peak_rss_mb,
    run_benchmark_with_memory,
    MemoryStats,
)
from .history import (
    load_history,
    summarize_history,
)
from .regression import (
    detect_regressions,
    print_regression_summary,
)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_header(subproject: str, meta: RunMeta):
    """Print benchmark run header."""
    print("═" * 68)
    print(f"BENCHMARK RESULTS: {subproject}")
    print("═" * 68)
    print()
    print(f"  Commit:    {meta.commit} ({meta.branch}){' [dirty]' if meta.dirty else ''}")
    print(f"  Host:      {meta.hostname} ({meta.cpu_count} cores)")
    print(f"  Env:       {meta.env_id}")
    print(f"  Suite:     {meta.suite}")
    print(f"  Warmup:    {meta.warmup_runs} runs")
    print(f"  Timing:    {meta.n_runs} runs")
    print()


def print_benchmark_result(result: BenchmarkResult, index: int, total: int):
    """Print single benchmark result."""
    status_icon = {
        "OK": "✓",
        "FAILED": "✗",
        "SKIPPED": "○",
    }.get(result.status, "?")
    
    time_str = f"{result.time_s:.3f}s ± {result.time_std_s:.3f}s"
    mem_str = f"{result.peak_rss_mb:.0f}MB"
    
    print(f"  [{index}/{total}] {status_icon} {result.id}")
    print(f"         Time: {time_str}  |  RSS: {mem_str}")
    
    if result.throughput_rows_per_sec:
        throughput = result.throughput_rows_per_sec
        if throughput >= 1_000_000:
            print(f"         Throughput: {throughput/1_000_000:.2f}M rows/s")
        else:
            print(f"         Throughput: {throughput/1_000:.1f}K rows/s")
    
    if result.status == "FAILED" and result.error_message:
        print(f"         Error: {result.error_message}")
    print()


def print_summary(run: BenchmarkRun, elapsed_total: float):
    """Print run summary."""
    summary = run.summary
    
    print("─" * 68)
    print(f"  Duration:   {elapsed_total:.1f}s")
    print(f"  Benchmarks: {summary.n_passed} passed", end="")
    if summary.n_failed > 0:
        print(f", {summary.n_failed} failed", end="")
    if summary.n_skipped > 0:
        print(f", {summary.n_skipped} skipped", end="")
    print()
    print(f"  Peak RSS:   {summary.peak_rss_mb:.0f} MB")
    print("─" * 68)


def print_regression(alarm: dict):
    """Print regression alarm."""
    print()
    print("─" * 68)
    print("⚠️  REGRESSION DETECTED")
    print("─" * 68)
    print()
    print(f"  Benchmark:  {alarm['benchmark']} / {alarm['scenario']} / n_jobs={alarm['param_n_jobs']}")
    print(f"  Metric:     {alarm['metric']}")
    print(f"  Current:    {alarm['current']:.3f}")
    print(f"  Baseline:   {alarm['baseline']:.3f} ({alarm['baseline_type']})")
    print(f"  Change:     +{alarm['change_pct']:.1f}%")
    print()


def print_exit_code(code: int):
    """Print exit code message."""
    print("─" * 68)
    messages = {
        0: "Exit code: 0 (all passed, no regressions)",
        1: "Exit code: 1 (regression detected)",
        2: "Exit code: 2 (benchmark execution error)",
    }
    print(messages.get(code, f"Exit code: {code}"))
    print()


# =============================================================================
# N_JOBS AUTO-DETECTION
# =============================================================================

def get_n_jobs_list(cpu_count: Optional[int] = None) -> list[int]:
    """
    Get list of n_jobs values based on CPU count.
    
    Parameters:
        cpu_count: Override CPU count (default: auto-detect)
    
    Returns:
        List of n_jobs values to benchmark
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    
    if cpu_count >= 64:
        return [1, 4, 16, 64]
    elif cpu_count >= 32:
        return [1, 4, 16, 32]
    else:
        return [1, 4, 8]


# =============================================================================
# BENCHMARK DISCOVERY
# =============================================================================

def discover_benchmarks(subproject: str, suite: str = "quick") -> list[dict]:
    """
    Discover benchmarks for a subproject.
    
    Phase 12.14b.GB: Now includes kernel and memory benchmarks.
    
    Returns list of benchmark specs:
        [{"name": ..., "func": ..., "scenarios": [...], "params": {...}}, ...]
    """
    if subproject == "groupby_regression":
        # Setup path for groupby_regression directory so workers can find modules
        gr_dir = Path(__file__).parent.parent / "groupby_regression"
        gr_dir_str = str(gr_dir.resolve())
        
        if gr_dir_str not in sys.path:
            sys.path.insert(0, gr_dir_str)
        
        # Set PYTHONPATH for spawned workers (macOS uses spawn)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if gr_dir_str not in current_pythonpath:
            if current_pythonpath:
                os.environ["PYTHONPATH"] = f"{gr_dir_str}:{current_pythonpath}"
            else:
                os.environ["PYTHONPATH"] = gr_dir_str
        
        benchmarks = []
        
        # V5 benchmarks (high-level API)
        try:
            from dfextensions.groupby_regression.benchmarks.bench_v5 import (
                get_benchmarks as get_v5_benchmarks
            )
            benchmarks.extend(get_v5_benchmarks(suite=suite))
        except ImportError as e:
            print(f"Warning: Could not import v5 benchmarks: {e}")
        
        # Kernel benchmarks (low-level Numba) - Phase 12.14b.GB
        try:
            from dfextensions.groupby_regression.benchmarks.bench_groupby_regression_kernels import (
                get_benchmarks as get_kernel_benchmarks
            )
            benchmarks.extend(get_kernel_benchmarks(suite=suite))
        except ImportError as e:
            print(f"Warning: Could not import kernel benchmarks: {e}")
        
        # Memory benchmarks - Phase 12.14b.GB
        try:
            from dfextensions.groupby_regression.benchmarks.bench_groupby_regression_memory import (
                get_benchmarks as get_memory_benchmarks
            )
            benchmarks.extend(get_memory_benchmarks(suite=suite))
        except ImportError as e:
            print(f"Warning: Could not import memory benchmarks: {e}")
        
        if not benchmarks:
            print(f"Warning: No benchmarks discovered for {subproject}")
        
        return benchmarks
    
    else:
        print(f"Warning: Unknown subproject: {subproject}")
        return []


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

def run_single_benchmark(
    name: str,
    func: Callable,
    scenario: str,
    params: dict,
    n_runs: int = DEFAULT_N_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    profile: bool = False,
) -> BenchmarkResult:
    """
    Run a single benchmark with timing and memory tracking.
    
    Parameters:
        name: Benchmark name
        func: Benchmark function
        scenario: Scenario name (S1, S2, etc.)
        params: Benchmark parameters
        n_runs: Number of timed runs
        warmup_runs: Number of warmup runs
        profile: Enable tracemalloc profiling
    
    Returns:
        BenchmarkResult
    """
    try:
        times, mem_stats, result = run_benchmark_with_memory(
            func,
            scenario=scenario,
            n_runs=n_runs,
            warmup_runs=warmup_runs,
            profile=profile,
            **params,
        )
        
        # Extract n_rows from benchmark result if available
        benchmark_params = {"scenario": scenario, **params}
        if isinstance(result, dict) and "n_rows_input" in result:
            benchmark_params["n_rows"] = result["n_rows_input"]
        
        # Phase 12.14b.GB: Merge adapter metrics into params
        if isinstance(result, dict):
            # Copy relevant metrics from adapter return dict
            for key in ["speedup_vs_numpy", "throughput_groups_per_sec", 
                        "rss_drift_pct", "rss_cv", "correctness_passed",
                        "speedup_gate_pass", "drift_gate_pass", "cv_gate_pass"]:
                if key in result:
                    benchmark_params[key] = result[key]
        
        bench_result = BenchmarkResult.from_timing(
            name=name,
            scenario=scenario,
            params=benchmark_params,
            times=times,
            peak_rss_mb=mem_stats.peak_rss_mb,
            peak_tracemalloc_mb=mem_stats.peak_tracemalloc_mb,
            memory_top_allocations=mem_stats.top_allocations,
        )
        
        return bench_result
    
    except Exception as e:
        # Create failed result
        n_jobs = params.get("n_jobs", 1)
        bench_id = f"{name}:{scenario}:n_jobs={n_jobs}"
        
        return BenchmarkResult(
            id=bench_id,
            name=name,
            scenario=scenario,
            params={"scenario": scenario, **params},
            time_s=0.0,
            time_std_s=0.0,
            n_runs=0,
            peak_rss_mb=get_peak_rss_mb(),
            status="FAILED",
            error_message=str(e),
        )


def run_benchmarks(
    subproject: str,
    suite: str = "quick",
    n_runs: int = DEFAULT_N_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    profile: bool = False,
    verbose: bool = True,
) -> BenchmarkRun:
    """
    Run all benchmarks for a subproject.
    
    Parameters:
        subproject: Subproject name
        suite: Benchmark suite ("quick" or "release")
        n_runs: Number of timed runs
        warmup_runs: Number of warmup runs
        profile: Enable tracemalloc profiling
        verbose: Print progress
    
    Returns:
        BenchmarkRun with results
    """
    start_time = time.time()
    
    # Create metadata
    meta = RunMeta.create(subproject=subproject)
    meta.suite = suite
    meta.n_runs = n_runs
    meta.warmup_runs = warmup_runs
    meta.run_mode = "profile" if profile else "gate"
    
    if verbose:
        print_header(subproject, meta)
    
    # Discover benchmarks
    benchmark_specs = discover_benchmarks(subproject, suite)
    
    if not benchmark_specs:
        print(f"No benchmarks found for {subproject}")
        return BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=[],
            alarms=[],
        )
    
    # Build list of all benchmark configurations
    all_configs = []
    n_jobs_list = get_n_jobs_list()
    
    for spec in benchmark_specs:
        # Phase 12.14b.GB: Check if benchmark uses n_jobs
        # Kernel and memory benchmarks don't use n_jobs parallelization
        benchmark_name = spec["name"]
        uses_n_jobs = benchmark_name not in [
            "kernel_single_fit", "kernel_multi_fit", "memory_rss_tracking"
        ]
        
        for scenario in spec["scenarios"]:
            if uses_n_jobs:
                # V5-style benchmarks: iterate over n_jobs
                for n_jobs in n_jobs_list:
                    all_configs.append({
                        "name": spec["name"],
                        "func": spec["func"],
                        "scenario": scenario,
                        "params": {**spec.get("params", {}), "n_jobs": n_jobs},
                    })
            else:
                # Kernel/memory benchmarks: single config per scenario
                all_configs.append({
                    "name": spec["name"],
                    "func": spec["func"],
                    "scenario": scenario,
                    "params": spec.get("params", {}),
                })
    
    # Run all benchmarks
    results = []
    total = len(all_configs)
    
    for i, config in enumerate(all_configs, 1):
        result = run_single_benchmark(
            name=config["name"],
            func=config["func"],
            scenario=config["scenario"],
            params=config["params"],
            n_runs=n_runs,
            warmup_runs=warmup_runs,
            profile=profile,
        )
        results.append(result)
        
        if verbose:
            print_benchmark_result(result, i, total)
    
    # Create summary
    summary = RunSummary(
        n_benchmarks=len(results),
        n_passed=sum(1 for r in results if r.status == "OK"),
        n_failed=sum(1 for r in results if r.status == "FAILED"),
        n_skipped=sum(1 for r in results if r.status == "SKIPPED"),
        total_time_s=sum(r.time_s for r in results),
        peak_rss_mb=max((r.peak_rss_mb for r in results), default=0),
    )
    
    # Create run
    run = BenchmarkRun(
        meta=meta,
        summary=summary,
        benchmarks=results,
        alarms=[],
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print_summary(run, elapsed)
    
    return run


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_run(run: BenchmarkRun) -> Path:
    """
    Save benchmark run to JSON file.
    
    Returns path to saved file.
    """
    output_path = get_results_path(
        subproject=run.meta.subproject,
        timestamp=run.meta.timestamp,
    )
    
    run.save(output_path)
    
    return output_path


def save_alarms(run: BenchmarkRun) -> Optional[Path]:
    """
    Save alarms to separate JSON file.
    
    Returns path to saved file, or None if no alarms.
    """
    if not run.alarms:
        return None
    
    output_path = get_alarms_path(
        subproject=run.meta.subproject,
        timestamp=run.meta.timestamp,
    )
    
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    alarm_data = {
        "run_id": run.meta.run_id,
        "timestamp": run.meta.timestamp,
        "commit": run.meta.commit,
        "n_alarms": len(run.alarms),
        "alarms": [a.to_dict() for a in run.alarms],
    }
    
    output_path.write_text(json.dumps(alarm_data, indent=2))
    
    return output_path


# =============================================================================
# CLI
# =============================================================================

def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Framework v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick suite
    python -m dfextensions.benchmarks.runner --subproject groupby_regression
    
    # Run release suite
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --suite release
    
    # With profiling
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --profile
    
    # Check regressions only
    python -m dfextensions.benchmarks.runner --subproject groupby_regression --check-only
""",
    )
    
    # Required
    parser.add_argument(
        "--subproject",
        required=True,
        help="Subproject to benchmark (e.g., groupby_regression)",
    )
    
    # Suite
    parser.add_argument(
        "--suite",
        choices=["quick", "release"],
        default="quick",
        help="Benchmark suite to run (default: quick)",
    )
    
    # Profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable tracemalloc profiling",
    )
    
    # Runs
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of timed runs (default: {DEFAULT_N_RUNS})",
    )
    
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP_RUNS})",
    )
    
    # Thresholds
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=DEFAULT_TIME_THRESHOLD,
        help=f"Time regression threshold (default: {DEFAULT_TIME_THRESHOLD})",
    )
    
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=DEFAULT_MEMORY_THRESHOLD,
        help=f"Memory regression threshold (default: {DEFAULT_MEMORY_THRESHOLD})",
    )
    
    # Modes
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check regressions only, don't run benchmarks",
    )
    
    parser.add_argument(
        "--cross-env",
        action="store_true",
        help="Allow cross-environment baseline comparison",
    )
    
    # Output
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report after run",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without saving",
    )
    
    return parser.parse_args(args)


def main(args=None) -> int:
    """
    Main entry point.
    
    Returns exit code:
        0 - All passed, no regressions
        1 - Regression detected
        2 - Execution error
    """
    parsed = parse_args(args)
    
    # Check platform
    supported, msg = check_platform_support()
    if not supported:
        print(f"Warning: {msg}")
    
    # Check-only mode
    if parsed.check_only:
        # Load most recent run and check for regressions
        from .history import discover_runs, load_run
        
        runs = discover_runs(parsed.subproject)
        if not runs:
            print(f"No runs found for {parsed.subproject}")
            return 2
        
        latest_run = load_run(runs[0])
        if latest_run is None:
            print(f"Could not load latest run: {runs[0]}")
            return 2
        
        print(f"Checking regressions for: {latest_run.meta.run_id}")
        
        history_df = load_history(
            subproject=parsed.subproject,
            exclude_profile=True,
        )
        
        if len(history_df) == 0:
            print("No history found for regression detection.")
            return 0
        
        alarms, regression_results = detect_regressions(
            current_run=latest_run,
            history_df=history_df,
            time_threshold=parsed.time_threshold,
            memory_threshold=parsed.memory_threshold,
            filter_env=not parsed.cross_env,
        )
        
        print_regression_summary(regression_results, alarms)
        
        if alarms:
            print_exit_code(1)
            return 1
        else:
            print_exit_code(0)
            return 0
    
    try:
        # Run benchmarks
        run = run_benchmarks(
            subproject=parsed.subproject,
            suite=parsed.suite,
            n_runs=parsed.n_runs,
            warmup_runs=parsed.warmup_runs,
            profile=parsed.profile,
            verbose=not parsed.quiet,
        )
        
        # Validate
        errors = validate_run(run)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 2
        
        # Save results
        if not parsed.dry_run:
            output_path = save_run(run)
            print(f"\nResults saved to: {output_path}")
        
        # Detect regressions
        if not parsed.profile:  # Skip for profile runs (different overhead)
            history_df = load_history(
                subproject=parsed.subproject,
                exclude_profile=True,
            )
            
            if len(history_df) > 0:
                alarms, regression_results = detect_regressions(
                    current_run=run,
                    history_df=history_df,
                    time_threshold=parsed.time_threshold,
                    memory_threshold=parsed.memory_threshold,
                    filter_env=not parsed.cross_env,
                )
                
                # Update run with alarms
                run.alarms = alarms
                run.summary.n_regressions = len(alarms)
                
                # Print summary
                if not parsed.quiet:
                    print_regression_summary(regression_results, alarms)
                
                # Save alarms
                if not parsed.dry_run and alarms:
                    alarms_path = save_alarms(run)
                    print(f"Alarms saved to: {alarms_path}")
            else:
                if not parsed.quiet:
                    print("\nNo history found for regression detection.")
        
        # TODO: Generate report
        
        # Determine exit code
        if run.summary.n_failed > 0:
            print_exit_code(2)
            return 2
        elif run.summary.n_regressions > 0:
            print_exit_code(1)
            return 1
        else:
            print_exit_code(0)
            return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
