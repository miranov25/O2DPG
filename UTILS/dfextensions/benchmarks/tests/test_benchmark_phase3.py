"""
Phase 12.10.BF-Polish: Report and Visualization Tests

Tests for:
- report.py: Trend extraction, plotting, report generation
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from dfextensions.benchmarks.schema import (
    BenchmarkResult,
    RunMeta,
    RunSummary,
    BenchmarkRun,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
)

from dfextensions.benchmarks.report import (
    HAS_MATPLOTLIB,
    BenchmarkTrend,
    BenchmarkReport,
    extract_trend,
    extract_all_trends,
    generate_report,
    format_report_text,
    format_report_html,
    save_report,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_history_df():
    """Create sample history DataFrame with multiple benchmarks."""
    rows = []
    
    # Create 10 runs with 2 benchmarks each
    for i in range(10):
        base_date = f"2024-12-{10+i:02d}"
        
        # Benchmark 1: v5_batch_fit:S1:n_jobs=1
        rows.append({
            "run_id": f"run_{i}_b1",
            "timestamp": pd.Timestamp(f"{base_date}T10:00:00Z"),
            "commit": f"abc{i:03d}",
            "branch": "main",
            "env_id": "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0",
            "run_mode": "gate",
            "suite": "quick",
            "hostname": "test",
            "benchmark_id": "v5_batch_fit:S1:n_jobs=1",
            "name": "v5_batch_fit",
            "scenario": "S1",
            "n_jobs": 1,
            "time_s": 0.5 + np.random.randn() * 0.02,
            "time_std_s": 0.01,
            "peak_rss_mb": 500 + np.random.randn() * 10,
            "throughput_rows_per_sec": 100000,
            "status": "OK",
        })
        
        # Benchmark 2: v5_batch_fit:S2:n_jobs=4
        rows.append({
            "run_id": f"run_{i}_b2",
            "timestamp": pd.Timestamp(f"{base_date}T10:00:00Z"),
            "commit": f"abc{i:03d}",
            "branch": "main",
            "env_id": "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0",
            "run_mode": "gate",
            "suite": "quick",
            "hostname": "test",
            "benchmark_id": "v5_batch_fit:S2:n_jobs=4",
            "name": "v5_batch_fit",
            "scenario": "S2",
            "n_jobs": 4,
            "time_s": 0.3 + np.random.randn() * 0.01,
            "time_std_s": 0.005,
            "peak_rss_mb": 600 + np.random.randn() * 15,
            "throughput_rows_per_sec": 333333,
            "status": "OK",
        })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_run():
    """Create sample benchmark run."""
    meta = RunMeta.create(subproject="groupby_regression")
    meta.env_id = "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0"
    meta.commit = "abc999"
    meta.branch = "main"
    
    benchmarks = [
        BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.52,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=510,
            status="OK",
        ),
        BenchmarkResult(
            id="v5_batch_fit:S2:n_jobs=4",
            name="v5_batch_fit",
            scenario="S2",
            params={"n_jobs": 4},
            time_s=0.31,
            time_std_s=0.005,
            n_runs=3,
            peak_rss_mb=605,
            status="OK",
        ),
    ]
    
    return BenchmarkRun(
        meta=meta,
        summary=RunSummary(n_benchmarks=len(benchmarks), n_passed=len(benchmarks)),
        benchmarks=benchmarks,
        alarms=[],
    )


# =============================================================================
# BENCHMARK TREND TESTS
# =============================================================================

class TestBenchmarkTrend:
    """Tests for BenchmarkTrend dataclass."""
    
    def test_trend_creation(self):
        """BenchmarkTrend can be created."""
        trend = BenchmarkTrend(
            benchmark_id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            n_jobs=1,
        )
        
        assert trend.benchmark_id == "test:S1:n_jobs=1"
        assert trend.n_samples == 0
        assert not trend.has_regression
    
    def test_trend_n_samples(self):
        """n_samples returns correct count."""
        trend = BenchmarkTrend(
            benchmark_id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            n_jobs=1,
            timestamps=[datetime.now(), datetime.now()],
            time_values=[0.5, 0.6],
            rss_values=[500, 510],
        )
        
        assert trend.n_samples == 2
    
    def test_trend_has_regression(self):
        """has_regression detects when time threshold exceeded."""
        trend = BenchmarkTrend(
            benchmark_id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            n_jobs=1,
            baseline_time=0.5,
            current_time=0.6,  # 20% regression
        )
        
        assert trend.has_regression
    
    def test_trend_has_memory_regression(self):
        """has_regression detects when memory threshold exceeded."""
        trend = BenchmarkTrend(
            benchmark_id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            n_jobs=1,
            baseline_time=0.5,
            current_time=0.5,  # No time regression
            baseline_rss=500.0,
            current_rss=600.0,  # 20% memory regression (> 15% threshold)
        )
        
        assert trend.has_regression
    
    def test_trend_no_regression_within_thresholds(self):
        """has_regression returns False when within both thresholds."""
        trend = BenchmarkTrend(
            benchmark_id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            n_jobs=1,
            baseline_time=0.5,
            current_time=0.52,  # 4% - within 10% threshold
            baseline_rss=500.0,
            current_rss=520.0,  # 4% - within 15% threshold
        )
        
        assert not trend.has_regression


# =============================================================================
# EXTRACT TREND TESTS
# =============================================================================

class TestExtractTrend:
    """Tests for extract_trend()."""
    
    def test_extract_trend_returns_trend(self, sample_history_df):
        """extract_trend returns BenchmarkTrend."""
        trend = extract_trend(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        
        assert isinstance(trend, BenchmarkTrend)
        assert trend.benchmark_id == "v5_batch_fit:S1:n_jobs=1"
        assert trend.name == "v5_batch_fit"
        assert trend.scenario == "S1"
        assert trend.n_jobs == 1
    
    def test_extract_trend_has_data(self, sample_history_df):
        """extract_trend populates data correctly."""
        trend = extract_trend(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        
        assert trend.n_samples == 10
        assert len(trend.time_values) == 10
        assert len(trend.rss_values) == 10
        assert len(trend.commits) == 10
    
    def test_extract_trend_calculates_baseline(self, sample_history_df):
        """extract_trend calculates baseline."""
        trend = extract_trend(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        
        assert trend.baseline_time is not None
        assert trend.baseline_rss is not None
        assert 0.4 < trend.baseline_time < 0.6  # Around 0.5
        assert 480 < trend.baseline_rss < 520  # Around 500
    
    def test_extract_trend_with_current(self, sample_history_df):
        """extract_trend includes current result."""
        current = {"time_s": 0.55, "peak_rss_mb": 520}
        trend = extract_trend(
            sample_history_df,
            "v5_batch_fit:S1:n_jobs=1",
            current_result=current,
        )
        
        assert trend.current_time == 0.55
        assert trend.current_rss == 520
    
    def test_extract_trend_unknown_benchmark(self, sample_history_df):
        """extract_trend handles unknown benchmark."""
        trend = extract_trend(sample_history_df, "unknown:S1:n_jobs=1")
        
        assert trend.n_samples == 0
        assert trend.baseline_time is None


class TestExtractAllTrends:
    """Tests for extract_all_trends()."""
    
    def test_extract_all_trends(self, sample_history_df):
        """extract_all_trends returns dict of trends."""
        trends = extract_all_trends(sample_history_df)
        
        assert isinstance(trends, dict)
        assert len(trends) == 2
        assert "v5_batch_fit:S1:n_jobs=1" in trends
        assert "v5_batch_fit:S2:n_jobs=4" in trends
    
    def test_extract_all_trends_with_run(self, sample_history_df, sample_run):
        """extract_all_trends includes current run data."""
        trends = extract_all_trends(sample_history_df, current_run=sample_run)
        
        trend1 = trends["v5_batch_fit:S1:n_jobs=1"]
        assert trend1.current_time == 0.52
        
        trend2 = trends["v5_batch_fit:S2:n_jobs=4"]
        assert trend2.current_time == 0.31


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

class TestGenerateReport:
    """Tests for generate_report()."""
    
    def test_generate_report_structure(self, sample_history_df):
        """generate_report returns BenchmarkReport."""
        report = generate_report(
            sample_history_df,
            subproject="groupby_regression",
        )
        
        assert isinstance(report, BenchmarkReport)
        assert report.subproject == "groupby_regression"
        assert report.n_benchmarks == 2
    
    def test_generate_report_with_run(self, sample_history_df, sample_run):
        """generate_report includes current run info."""
        report = generate_report(
            sample_history_df,
            subproject="groupby_regression",
            current_run=sample_run,
        )
        
        assert report.env_id == sample_run.meta.env_id
        assert report.commit == sample_run.meta.commit
        assert report.branch == sample_run.meta.branch
    
    def test_generate_report_has_trends(self, sample_history_df):
        """generate_report populates trends."""
        report = generate_report(
            sample_history_df,
            subproject="groupby_regression",
        )
        
        assert len(report.trends) == 2
        assert "v5_batch_fit:S1:n_jobs=1" in report.trends
    
    def test_generate_report_history_summary(self, sample_history_df):
        """generate_report includes history summary."""
        report = generate_report(
            sample_history_df,
            subproject="groupby_regression",
        )
        
        assert report.history_summary is not None
        assert report.history_summary["n_runs"] > 0


# =============================================================================
# REPORT FORMATTING TESTS
# =============================================================================

class TestFormatReport:
    """Tests for report formatting."""
    
    def test_format_report_text(self, sample_history_df):
        """format_report_text produces string."""
        report = generate_report(sample_history_df, subproject="test")
        text = format_report_text(report)
        
        assert isinstance(text, str)
        assert "test" in text
        assert "SUMMARY" in text
        assert "BENCHMARK DETAILS" in text
    
    def test_format_report_html(self, sample_history_df):
        """format_report_html produces HTML."""
        report = generate_report(sample_history_df, subproject="test")
        html = format_report_html(report)
        
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "test" in html
    
    def test_format_report_text_with_regression(self, sample_history_df, sample_run):
        """format_report_text shows regressions."""
        # Modify run to have regression
        sample_run.benchmarks[0].time_s = 0.7  # 40% regression
        
        report = generate_report(
            sample_history_df,
            subproject="test",
            current_run=sample_run,
        )
        
        text = format_report_text(report)
        
        if report.n_regressions > 0:
            assert "REGRESSIONS" in text


# =============================================================================
# SAVE REPORT TESTS
# =============================================================================

class TestSaveReport:
    """Tests for save_report()."""
    
    def test_save_report_text(self, sample_history_df, tmp_path):
        """save_report saves text file."""
        report = generate_report(sample_history_df, subproject="test")
        output_path = tmp_path / "report.txt"
        
        saved_path = save_report(report, output_path)
        
        assert saved_path.exists()
        content = saved_path.read_text()
        assert "test" in content
    
    def test_save_report_html(self, sample_history_df, tmp_path):
        """save_report saves HTML file."""
        report = generate_report(sample_history_df, subproject="test")
        output_path = tmp_path / "report.html"
        
        saved_path = save_report(report, output_path)
        
        assert saved_path.exists()
        content = saved_path.read_text()
        assert "<!DOCTYPE html>" in content
    
    def test_save_report_auto_format(self, sample_history_df, tmp_path):
        """save_report auto-detects format from extension."""
        report = generate_report(sample_history_df, subproject="test")
        
        # HTML by extension
        html_path = save_report(report, tmp_path / "r.html")
        assert "<!DOCTYPE html>" in html_path.read_text()
        
        # Text by extension
        txt_path = save_report(report, tmp_path / "r.txt")
        assert "<!DOCTYPE html>" not in txt_path.read_text()


# =============================================================================
# PLOTTING TESTS (Skip if no matplotlib)
# =============================================================================

class TestPlotting:
    """Tests for plotting functions."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_benchmark_trend(self, sample_history_df):
        """plot_benchmark_trend creates figure."""
        from dfextensions.benchmarks.report import plot_benchmark_trend
        
        trend = extract_trend(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        fig = plot_benchmark_trend(trend)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_benchmark_trends(self, sample_history_df):
        """plot_benchmark_trends creates figure."""
        from dfextensions.benchmarks.report import plot_benchmark_trends
        
        fig = plot_benchmark_trends(
            sample_history_df,
            benchmark_id="v5_batch_fit:S1:n_jobs=1",
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_all_benchmarks(self, sample_history_df):
        """plot_benchmark_trends handles all benchmarks."""
        from dfextensions.benchmarks.report import plot_benchmark_trends
        
        fig = plot_benchmark_trends(sample_history_df)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_has_matplotlib_flag(self):
        """HAS_MATPLOTLIB flag is boolean."""
        assert isinstance(HAS_MATPLOTLIB, bool)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_empty_trend(self):
        """plot_benchmark_trend handles empty trend gracefully."""
        from dfextensions.benchmarks.report import plot_benchmark_trend
        
        # Create trend with no data
        empty_trend = BenchmarkTrend(
            benchmark_id="empty:S1:n_jobs=1",
            name="empty",
            scenario="S1",
            n_jobs=1,
            timestamps=[],
            time_values=[],
            rss_values=[],
        )
        
        # Should not raise - returns figure with "No data available"
        fig = plot_benchmark_trend(empty_trend)
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# HTML ESCAPING TESTS
# =============================================================================

class TestHTMLEscaping:
    """Tests for HTML escaping in reports."""
    
    def test_html_escapes_special_characters(self, sample_history_df):
        """HTML report escapes special characters."""
        # Create report with potentially dangerous characters
        report = generate_report(sample_history_df, subproject="test")
        report.title = "Test <script>alert('xss')</script>"
        report.env_id = "Linux_CPU<>&_3.9.6"
        report.commit = "abc<123>"
        report.branch = "feature/test&branch"
        
        html = format_report_html(report)
        
        # Verify special characters are escaped
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;&gt;&amp;" in html or "&lt;" in html


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
