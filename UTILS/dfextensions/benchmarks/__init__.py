"""
Benchmark Framework v1.0

Standardized benchmarking for dfextensions subprojects.

Phase 12.10.BF: Regression detection with memory profiling.

Usage:
    from dfextensions.benchmarks import BenchmarkRun, run_benchmarks
    
    run = run_benchmarks(subproject="groupby_regression", suite="quick")
    run.save("results.json")

CLI:
    python -m dfextensions.benchmarks.runner --subproject groupby_regression
"""

from .schema import (
    # Constants
    SCHEMA_VERSION,
    RUNNER_VERSION,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_N_RUNS,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
    # Dataclasses
    BenchmarkParams,
    BenchmarkResult,
    RunMeta,
    RunSummary,
    Alarm,
    BenchmarkRun,
    # Utilities
    get_git_info,
    get_tool_versions,
    get_benchmark_prefix,
    validate_run,
)

from .profiler import (
    check_platform_support,
    get_peak_rss_mb,
    get_current_rss_mb,
    MemoryStats,
    track_peak_rss,
    track_memory_detailed,
    MemoryGuard,
    run_benchmark_with_memory,
)

from .runner import (
    run_benchmarks,
    run_single_benchmark,
    get_n_jobs_list,
    save_run,
)

from .history import (
    discover_runs,
    load_run,
    load_history,
    filter_by_env,
    filter_by_benchmark,
    get_baseline,
    get_all_baselines,
    summarize_history,
)

from .regression import (
    MIN_BASELINE_SAMPLES,
    RegressionResult,
    check_regression,
    detect_regressions,
    print_regression_summary,
    format_regression_report,
)

from .report import (
    HAS_MATPLOTLIB,
    BenchmarkTrend,
    BenchmarkReport,
    extract_trend,
    extract_all_trends,
    plot_benchmark_trend,
    plot_benchmark_trends,
    generate_report,
    format_report_text,
    format_report_html,
    save_report,
)

__version__ = RUNNER_VERSION

__all__ = [
    # Version
    "__version__",
    # Constants
    "SCHEMA_VERSION",
    "RUNNER_VERSION",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_N_RUNS",
    "DEFAULT_TIME_THRESHOLD",
    "DEFAULT_MEMORY_THRESHOLD",
    "MIN_BASELINE_SAMPLES",
    # Dataclasses
    "BenchmarkParams",
    "BenchmarkResult",
    "RunMeta",
    "RunSummary",
    "Alarm",
    "BenchmarkRun",
    # Schema utilities
    "get_git_info",
    "get_tool_versions",
    "get_benchmark_prefix",
    "validate_run",
    # Profiler
    "check_platform_support",
    "get_peak_rss_mb",
    "get_current_rss_mb",
    "MemoryStats",
    "track_peak_rss",
    "track_memory_detailed",
    "MemoryGuard",
    "run_benchmark_with_memory",
    # Runner
    "run_benchmarks",
    "run_single_benchmark",
    "get_n_jobs_list",
    "save_run",
    # History
    "discover_runs",
    "load_run",
    "load_history",
    "filter_by_env",
    "filter_by_benchmark",
    "get_baseline",
    "get_all_baselines",
    "summarize_history",
    # Regression
    "RegressionResult",
    "check_regression",
    "detect_regressions",
    "print_regression_summary",
    "format_regression_report",
    # Report
    "HAS_MATPLOTLIB",
    "BenchmarkTrend",
    "BenchmarkReport",
    "extract_trend",
    "extract_all_trends",
    "plot_benchmark_trend",
    "plot_benchmark_trends",
    "generate_report",
    "format_report_text",
    "format_report_html",
    "save_report",
]
