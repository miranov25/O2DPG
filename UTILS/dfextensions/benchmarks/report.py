"""
Benchmark Framework v1.0 — Reporting and Visualization

Generate reports and trend visualizations from benchmark history.

Phase 12.10.BF-Polish: Reporting integration.

Features:
    - Historical trend plots (time_s, peak_rss_mb over time)
    - Regression summary reports
    - Integration with AliasDataFrame.draw_batch() style
    - Export to PNG/PDF

Usage:
    from dfextensions.benchmarks.report import (
        plot_benchmark_trends,
        generate_report,
        save_report,
    )
    
    # Load history and plot trends
    df = load_history("groupby_regression")
    fig = plot_benchmark_trends(df, benchmark_id="v5_batch_fit:S2:n_jobs=4")
    fig.savefig("trends.png")
    
    # Generate full report
    report = generate_report(df, current_run=run)
    save_report(report, "benchmark_report.html")
"""

from dataclasses import dataclass, field
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Optional matplotlib import (graceful degradation)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from .schema import (
    BenchmarkRun,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
)
from .history import (
    load_history,
    filter_by_env,
    filter_by_benchmark,
    get_baseline,
    summarize_history,
)
from .regression import (
    MIN_BASELINE_SAMPLES,
    RegressionResult,
    detect_regressions,
)


# =============================================================================
# REPORT DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkTrend:
    """Trend data for a single benchmark."""
    benchmark_id: str
    name: str
    scenario: str
    n_jobs: int
    
    # Historical data
    timestamps: List[datetime] = field(default_factory=list)
    time_values: List[float] = field(default_factory=list)
    rss_values: List[float] = field(default_factory=list)
    commits: List[str] = field(default_factory=list)
    
    # Baseline
    baseline_time: Optional[float] = None
    baseline_rss: Optional[float] = None
    
    # Current (if available)
    current_time: Optional[float] = None
    current_rss: Optional[float] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.timestamps)
    
    @property
    def has_regression(self) -> bool:
        """Check if either time or memory exceeds regression threshold."""
        if self.baseline_time is None or self.current_time is None:
            return False
        
        # Check time regression
        time_change = (self.current_time - self.baseline_time) / self.baseline_time
        if time_change > DEFAULT_TIME_THRESHOLD:
            return True
        
        # Check memory regression
        if self.baseline_rss is not None and self.current_rss is not None:
            rss_change = (self.current_rss - self.baseline_rss) / self.baseline_rss
            if rss_change > DEFAULT_MEMORY_THRESHOLD:
                return True
        
        return False


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    title: str
    generated_at: str
    subproject: str
    
    # Summary
    n_benchmarks: int = 0
    n_with_history: int = 0
    n_regressions: int = 0
    
    # Environment info
    env_id: Optional[str] = None
    commit: Optional[str] = None
    branch: Optional[str] = None
    
    # Trends per benchmark
    trends: Dict[str, BenchmarkTrend] = field(default_factory=dict)
    
    # Alarms
    alarms: List[dict] = field(default_factory=list)
    
    # History summary
    history_summary: Optional[dict] = None


# =============================================================================
# TREND EXTRACTION
# =============================================================================

def extract_trend(
    history_df: pd.DataFrame,
    benchmark_id: str,
    current_result: Optional[dict] = None,
) -> BenchmarkTrend:
    """
    Extract trend data for a single benchmark.
    
    Parameters:
        history_df: Historical data (pre-filtered by env_id if desired)
        benchmark_id: Benchmark ID to extract
        current_result: Optional current benchmark result dict
    
    Returns:
        BenchmarkTrend with historical and baseline data
    """
    # Filter to specific benchmark
    bench_df = filter_by_benchmark(history_df, benchmark_id)
    bench_df = bench_df[bench_df["status"] == "OK"]
    
    # Sort by timestamp ascending for plotting
    bench_df = bench_df.sort_values("timestamp", ascending=True)
    
    # Parse benchmark_id for name, scenario, n_jobs
    parts = benchmark_id.split(":")
    name = parts[0] if len(parts) > 0 else benchmark_id
    scenario = parts[1] if len(parts) > 1 else "default"
    n_jobs = 1
    if len(parts) > 2 and "n_jobs=" in parts[2]:
        try:
            n_jobs = int(parts[2].split("=")[1])
        except (ValueError, IndexError):
            pass
    
    trend = BenchmarkTrend(
        benchmark_id=benchmark_id,
        name=name,
        scenario=scenario,
        n_jobs=n_jobs,
    )
    
    if len(bench_df) > 0:
        trend.timestamps = bench_df["timestamp"].tolist()
        trend.time_values = bench_df["time_s"].tolist()
        trend.rss_values = bench_df["peak_rss_mb"].tolist()
        trend.commits = bench_df["commit"].tolist()
        
        # Calculate baseline (median of last N)
        baseline = get_baseline(history_df, benchmark_id)
        if baseline is not None:
            trend.baseline_time = baseline["time_s"]
            trend.baseline_rss = baseline["peak_rss_mb"]
    
    # Add current result if provided
    if current_result is not None:
        trend.current_time = current_result.get("time_s")
        trend.current_rss = current_result.get("peak_rss_mb")
    
    return trend


def extract_all_trends(
    history_df: pd.DataFrame,
    current_run: Optional[BenchmarkRun] = None,
) -> Dict[str, BenchmarkTrend]:
    """
    Extract trends for all benchmarks in history.
    
    Parameters:
        history_df: Historical data
        current_run: Optional current run for comparison
    
    Returns:
        Dict mapping benchmark_id → BenchmarkTrend
    """
    trends = {}
    
    # Get all unique benchmark IDs
    benchmark_ids = history_df["benchmark_id"].unique()
    
    # Build current results lookup
    current_results = {}
    if current_run is not None:
        for bench in current_run.benchmarks:
            current_results[bench.id] = {
                "time_s": bench.time_s,
                "peak_rss_mb": bench.peak_rss_mb,
            }
    
    for benchmark_id in benchmark_ids:
        current_result = current_results.get(benchmark_id)
        trends[benchmark_id] = extract_trend(
            history_df, benchmark_id, current_result
        )
    
    return trends


# =============================================================================
# PLOTTING
# =============================================================================

def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_benchmark_trend(
    trend: BenchmarkTrend,
    figsize: Tuple[float, float] = (10, 6),
    show_baseline: bool = True,
    show_threshold: bool = True,
    title: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot time trend for a single benchmark.
    
    Parameters:
        trend: BenchmarkTrend data
        figsize: Figure size
        show_baseline: Show baseline line
        show_threshold: Show regression threshold band
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    check_matplotlib()
    
    # Handle empty trend data
    if len(trend.timestamps) == 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f"{trend.name} : {trend.scenario} : n_jobs={trend.n_jobs}",
                    fontsize=12, fontweight="bold")
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=12, color="gray")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        ax2.axis("off")
        return fig
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Title
    if title is None:
        title = f"{trend.name} : {trend.scenario} : n_jobs={trend.n_jobs}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    
    # Convert timestamps to matplotlib dates
    dates = mdates.date2num(trend.timestamps)
    
    # --- Time plot ---
    ax1.plot(dates, trend.time_values, "b-o", markersize=4, label="time_s")
    
    if show_baseline and trend.baseline_time is not None:
        ax1.axhline(
            trend.baseline_time, color="green", linestyle="--",
            label=f"baseline ({trend.baseline_time:.3f}s)"
        )
        
        if show_threshold:
            threshold_value = trend.baseline_time * (1 + DEFAULT_TIME_THRESHOLD)
            ax1.axhline(
                threshold_value, color="red", linestyle=":",
                label=f"+{DEFAULT_TIME_THRESHOLD*100:.0f}% threshold"
            )
            ax1.fill_between(
                [dates[0], dates[-1]] if len(dates) > 0 else [0, 1],
                trend.baseline_time,
                threshold_value,
                alpha=0.1, color="orange"
            )
    
    # Mark current if present
    if trend.current_time is not None and len(dates) > 0:
        ax1.axhline(
            trend.current_time, color="purple", linestyle="-",
            linewidth=2, label=f"current ({trend.current_time:.3f}s)"
        )
    
    ax1.set_ylabel("Time (seconds)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # --- Memory plot ---
    ax2.plot(dates, trend.rss_values, "r-o", markersize=4, label="peak_rss_mb")
    
    if show_baseline and trend.baseline_rss is not None:
        ax2.axhline(
            trend.baseline_rss, color="green", linestyle="--",
            label=f"baseline ({trend.baseline_rss:.0f}MB)"
        )
        
        if show_threshold:
            threshold_value = trend.baseline_rss * (1 + DEFAULT_MEMORY_THRESHOLD)
            ax2.axhline(
                threshold_value, color="red", linestyle=":",
                label=f"+{DEFAULT_MEMORY_THRESHOLD*100:.0f}% threshold"
            )
    
    ax2.set_ylabel("Peak RSS (MB)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_benchmark_trends(
    history_df: pd.DataFrame,
    benchmark_id: Optional[str] = None,
    env_id: Optional[str] = None,
    current_run: Optional[BenchmarkRun] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> "plt.Figure":
    """
    Plot trends for one or more benchmarks.
    
    Parameters:
        history_df: Historical data
        benchmark_id: Specific benchmark to plot (or None for all)
        env_id: Filter to specific environment
        current_run: Current run for comparison
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    check_matplotlib()
    
    # Filter by environment if specified
    if env_id is not None:
        history_df = filter_by_env(history_df, env_id)
    
    if benchmark_id is not None:
        # Single benchmark
        current_result = None
        if current_run is not None:
            for bench in current_run.benchmarks:
                if bench.id == benchmark_id:
                    current_result = {
                        "time_s": bench.time_s,
                        "peak_rss_mb": bench.peak_rss_mb,
                    }
                    break
        
        trend = extract_trend(history_df, benchmark_id, current_result)
        return plot_benchmark_trend(trend, figsize=figsize)
    else:
        # Multiple benchmarks - create grid
        trends = extract_all_trends(history_df, current_run)
        n_benchmarks = len(trends)
        
        if n_benchmarks == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No benchmark data available",
                   ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig
        
        # Create subplot grid (2 columns)
        n_cols = 2
        n_rows = (n_benchmarks + 1) // 2
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize[0] * 1.5, figsize[1] * n_rows / 2)
        )
        axes = axes.flatten() if n_benchmarks > 1 else [axes]
        
        for idx, (benchmark_id, trend) in enumerate(trends.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            if len(trend.timestamps) > 0:
                dates = mdates.date2num(trend.timestamps)
                ax.plot(dates, trend.time_values, "b-o", markersize=3)
                
                if trend.baseline_time is not None:
                    ax.axhline(trend.baseline_time, color="green",
                              linestyle="--", alpha=0.7)
                
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            
            ax.set_title(f"{trend.name}:{trend.scenario}", fontsize=9)
            ax.set_ylabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_benchmarks, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    history_df: pd.DataFrame,
    subproject: str,
    current_run: Optional[BenchmarkRun] = None,
    title: Optional[str] = None,
) -> BenchmarkReport:
    """
    Generate a complete benchmark report.
    
    Parameters:
        history_df: Historical data
        subproject: Subproject name
        current_run: Optional current run for comparison
        title: Optional custom title
    
    Returns:
        BenchmarkReport with all analysis
    """
    if title is None:
        title = f"Benchmark Report: {subproject}"
    
    report = BenchmarkReport(
        title=title,
        generated_at=datetime.now().isoformat(),
        subproject=subproject,
    )
    
    # Extract environment info from current run
    if current_run is not None:
        report.env_id = current_run.meta.env_id
        report.commit = current_run.meta.commit
        report.branch = current_run.meta.branch
        
        # Filter history to same environment
        history_df = filter_by_env(history_df, current_run.meta.env_id)
    
    # History summary
    report.history_summary = summarize_history(history_df)
    
    # Extract trends
    report.trends = extract_all_trends(history_df, current_run)
    report.n_benchmarks = len(report.trends)
    report.n_with_history = sum(
        1 for t in report.trends.values() if t.n_samples >= MIN_BASELINE_SAMPLES
    )
    
    # Detect regressions if current run provided
    if current_run is not None:
        # Note: history_df is already filtered by env_id above, but we use
        # filter_env=True as a safety measure to ensure consistent behavior
        alarms, _ = detect_regressions(current_run, history_df, filter_env=True)
        report.n_regressions = len(alarms)
        report.alarms = [a.to_dict() for a in alarms]
    
    return report


def format_report_text(report: BenchmarkReport) -> str:
    """
    Format report as plain text.
    
    Parameters:
        report: BenchmarkReport
    
    Returns:
        Formatted text report
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append(report.title)
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated: {report.generated_at}")
    lines.append(f"Subproject: {report.subproject}")
    
    if report.env_id:
        lines.append(f"Environment: {report.env_id}")
    if report.commit:
        lines.append(f"Commit: {report.commit}")
    if report.branch:
        lines.append(f"Branch: {report.branch}")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total benchmarks:    {report.n_benchmarks}")
    lines.append(f"  With baseline (≥{MIN_BASELINE_SAMPLES}): {report.n_with_history}")
    lines.append(f"  Regressions:         {report.n_regressions}")
    
    if report.history_summary:
        lines.append(f"  History runs:        {report.history_summary.get('n_runs', 0)}")
        lines.append(f"  Pass rate:           {report.history_summary.get('pass_rate', 0)*100:.1f}%")
    
    lines.append("")
    
    if report.n_regressions > 0:
        lines.append("-" * 70)
        lines.append("⚠ REGRESSIONS")
        lines.append("-" * 70)
        for alarm in report.alarms:
            lines.append(f"  {alarm['benchmark']}:{alarm['scenario']}:n_jobs={alarm['param_n_jobs']}")
            lines.append(f"    {alarm['metric']}: {alarm['current']:.3f} → {alarm['baseline']:.3f} (+{alarm['change_pct']:.1f}%)")
        lines.append("")
    
    lines.append("-" * 70)
    lines.append("BENCHMARK DETAILS")
    lines.append("-" * 70)
    
    for benchmark_id, trend in report.trends.items():
        status = "✓" if not trend.has_regression else "⚠"
        lines.append(f"  {status} {benchmark_id}")
        lines.append(f"      samples: {trend.n_samples}")
        if trend.baseline_time is not None:
            lines.append(f"      baseline: {trend.baseline_time:.3f}s / {trend.baseline_rss:.0f}MB")
        if trend.current_time is not None:
            lines.append(f"      current:  {trend.current_time:.3f}s / {trend.current_rss:.0f}MB")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def format_report_html(report: BenchmarkReport) -> str:
    """
    Format report as HTML.
    
    Parameters:
        report: BenchmarkReport
    
    Returns:
        HTML report string
    """
    html = []
    
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append(f"<title>{html_escape(report.title)}</title>")
    html.append("<style>")
    html.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .summary dt { font-weight: bold; display: inline-block; width: 200px; }
        .summary dd { display: inline-block; margin: 0; }
        .alarm { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alarm-title { font-weight: bold; color: #856404; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f0f0f0; }
        .status-ok { color: green; }
        .status-warn { color: orange; }
        .meta { color: #666; font-size: 0.9em; }
    """)
    html.append("</style></head><body>")
    
    html.append(f"<h1>{html_escape(report.title)}</h1>")
    html.append(f"<p class='meta'>Generated: {html_escape(report.generated_at)}</p>")
    
    if report.env_id:
        html.append(f"<p class='meta'>Environment: <code>{html_escape(report.env_id)}</code></p>")
    if report.commit:
        html.append(f"<p class='meta'>Commit: <code>{html_escape(report.commit)}</code> ({html_escape(report.branch or 'unknown')})</p>")
    
    html.append("<h2>Summary</h2>")
    html.append("<div class='summary'><dl>")
    html.append(f"<dt>Total benchmarks:</dt><dd>{report.n_benchmarks}</dd><br>")
    html.append(f"<dt>With baseline (≥{MIN_BASELINE_SAMPLES}):</dt><dd>{report.n_with_history}</dd><br>")
    html.append(f"<dt>Regressions:</dt><dd>{report.n_regressions}</dd><br>")
    if report.history_summary:
        html.append(f"<dt>History runs:</dt><dd>{report.history_summary.get('n_runs', 0)}</dd><br>")
        html.append(f"<dt>Pass rate:</dt><dd>{report.history_summary.get('pass_rate', 0)*100:.1f}%</dd>")
    html.append("</dl></div>")
    
    if report.n_regressions > 0:
        html.append("<h2>⚠ Regressions Detected</h2>")
        for alarm in report.alarms:
            html.append("<div class='alarm'>")
            alarm_title = f"{alarm['benchmark']}:{alarm['scenario']}:n_jobs={alarm['param_n_jobs']}"
            html.append(f"<div class='alarm-title'>{html_escape(alarm_title)}</div>")
            html.append(f"<div>{html_escape(alarm['metric'])}: {alarm['current']:.3f} → {alarm['baseline']:.3f} (+{alarm['change_pct']:.1f}%)</div>")
            html.append("</div>")
    
    html.append("<h2>Benchmark Details</h2>")
    html.append("<table>")
    html.append("<tr><th>Benchmark</th><th>Samples</th><th>Baseline</th><th>Current</th><th>Status</th></tr>")
    
    for benchmark_id, trend in report.trends.items():
        status_class = "status-ok" if not trend.has_regression else "status-warn"
        status_text = "✓ OK" if not trend.has_regression else "⚠ Regression"
        
        baseline_str = f"{trend.baseline_time:.3f}s / {trend.baseline_rss:.0f}MB" if trend.baseline_time else "—"
        current_str = f"{trend.current_time:.3f}s / {trend.current_rss:.0f}MB" if trend.current_time else "—"
        
        html.append(f"<tr>")
        html.append(f"<td>{html_escape(benchmark_id)}</td>")
        html.append(f"<td>{trend.n_samples}</td>")
        html.append(f"<td>{baseline_str}</td>")
        html.append(f"<td>{current_str}</td>")
        html.append(f"<td class='{status_class}'>{status_text}</td>")
        html.append(f"</tr>")
    
    html.append("</table>")
    html.append("</body></html>")
    
    return "\n".join(html)


def save_report(
    report: BenchmarkReport,
    path: Union[Path, str],
    format: str = "auto",
) -> Path:
    """
    Save report to file.
    
    Parameters:
        report: BenchmarkReport
        path: Output path
        format: "text", "html", or "auto" (detect from extension)
    
    Returns:
        Path to saved file
    """
    path = Path(path)
    
    if format == "auto":
        if path.suffix in [".html", ".htm"]:
            format = "html"
        else:
            format = "text"
    
    if format == "html":
        content = format_report_html(report)
    else:
        content = format_report_text(report)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    
    return path


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "BenchmarkTrend",
    "BenchmarkReport",
    # Trend extraction
    "extract_trend",
    "extract_all_trends",
    # Plotting
    "HAS_MATPLOTLIB",
    "check_matplotlib",
    "plot_benchmark_trend",
    "plot_benchmark_trends",
    # Report generation
    "generate_report",
    "format_report_text",
    "format_report_html",
    "save_report",
]
