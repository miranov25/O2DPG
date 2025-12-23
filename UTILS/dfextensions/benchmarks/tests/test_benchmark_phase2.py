"""
Phase 12.10.BF-Analysis: History and Regression Tests

Tests for:
- history.py: Loading and filtering benchmark history
- regression.py: Regression detection against baselines

Directory Structure (run pytest from O2DPG/UTILS):
    O2DPG/UTILS/
    ├── groupby_regression/           ← Existing package
    │   └── benchmarks/
    │       ├── scenarios.py
    │       └── bench_v5.py
    └── dfextensions/
        └── benchmarks/               ← Framework
            ├── history.py            ← NEW
            ├── regression.py         ← NEW
            └── tests/
                └── test_benchmark_phase2.py  ← This file
"""

import json
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
    Alarm,
    DEFAULT_TIME_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD,
)

from dfextensions.benchmarks.history import (
    discover_runs,
    load_run,
    load_history,
    filter_by_env,
    filter_by_benchmark,
    filter_passed_only,
    get_baseline,
    get_all_baselines,
    summarize_history,
)

from dfextensions.benchmarks.regression import (
    RegressionResult,
    check_regression,
    detect_regressions,
    print_regression_summary,
    format_regression_report,
    MIN_BASELINE_SAMPLES,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_benchmark_dir(tmp_path):
    """Create temporary benchmark directory with sample runs."""
    prefix = tmp_path / "benchmarks"
    
    # Create 3 runs with different timestamps
    dates = ["2024-12-20", "2024-12-21", "2024-12-22"]
    
    for i, date in enumerate(dates):
        # Directory name format: YYYY-MM-DDTHH-MM-SS
        ts_dir = f"{date}T10-00-00"
        run_dir = prefix / ts_dir / "groupby_regression"
        run_dir.mkdir(parents=True)
        
        # ISO timestamp format: YYYY-MM-DDTHH:MM:SS.sssZ
        iso_ts = f"{date}T10:00:00.000Z"
        
        # Create run with consistent benchmark times
        meta = RunMeta(
            run_id=f"run_{i}",
            timestamp=iso_ts,
            commit=f"abc{i}",
            branch="main",
            env_id="Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0",
            subproject="groupby_regression",
            run_mode="gate",
            suite="quick",
        )
        
        benchmarks = [
            BenchmarkResult(
                id="v5_batch_fit:S1:n_jobs=1",
                name="v5_batch_fit",
                scenario="S1",
                params={"n_jobs": 1, "n_rows": 50000},
                time_s=0.5 + i * 0.01,  # Slight increase each run
                time_std_s=0.01,
                n_runs=3,
                peak_rss_mb=500 + i * 5,
                status="OK",
            ),
            BenchmarkResult(
                id="v5_batch_fit:S2:n_jobs=4",
                name="v5_batch_fit",
                scenario="S2",
                params={"n_jobs": 4, "n_rows": 100000},
                time_s=0.3 + i * 0.005,
                time_std_s=0.005,
                n_runs=3,
                peak_rss_mb=600 + i * 3,
                status="OK",
            ),
        ]
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(n_benchmarks=len(benchmarks), n_passed=len(benchmarks)),
            benchmarks=benchmarks,
            alarms=[],
        )
        
        results_path = run_dir / "results.json"
        run.save(results_path)
    
    return prefix


@pytest.fixture
def sample_history_df():
    """Create sample history DataFrame."""
    rows = []
    for i in range(10):
        rows.append({
            "run_id": f"run_{i}",
            "timestamp": pd.Timestamp(f"2024-12-{10+i}T10:00:00Z"),
            "commit": f"abc{i}",
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
    
    return pd.DataFrame(rows)


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestDiscoverRuns:
    """Tests for discover_runs()"""
    
    def test_discover_runs_finds_all(self, temp_benchmark_dir):
        """discover_runs finds all runs."""
        runs = discover_runs("groupby_regression", prefix=temp_benchmark_dir)
        assert len(runs) == 3
    
    def test_discover_runs_sorted_newest_first(self, temp_benchmark_dir):
        """discover_runs returns newest first."""
        runs = discover_runs("groupby_regression", prefix=temp_benchmark_dir)
        # Newest timestamp should be first
        assert "2024-12-22" in str(runs[0])
    
    def test_discover_runs_empty_subproject(self, temp_benchmark_dir):
        """discover_runs returns empty for unknown subproject."""
        runs = discover_runs("unknown_project", prefix=temp_benchmark_dir)
        assert len(runs) == 0
    
    def test_discover_runs_empty_prefix(self, tmp_path):
        """discover_runs handles missing prefix."""
        runs = discover_runs("groupby_regression", prefix=tmp_path / "nonexistent")
        assert len(runs) == 0


class TestLoadRun:
    """Tests for load_run()"""
    
    def test_load_run_valid(self, temp_benchmark_dir):
        """load_run loads valid JSON."""
        runs = discover_runs("groupby_regression", prefix=temp_benchmark_dir)
        run = load_run(runs[0])
        
        assert run is not None
        assert run.meta.subproject == "groupby_regression"
        assert len(run.benchmarks) == 2
    
    def test_load_run_invalid_json(self, tmp_path):
        """load_run returns None for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")
        
        run = load_run(bad_file)
        assert run is None


class TestLoadHistory:
    """Tests for load_history()"""
    
    def test_load_history_returns_dataframe(self, temp_benchmark_dir):
        """load_history returns DataFrame."""
        df = load_history("groupby_regression", prefix=temp_benchmark_dir)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 runs × 2 benchmarks
    
    def test_load_history_has_expected_columns(self, temp_benchmark_dir):
        """load_history DataFrame has expected columns."""
        df = load_history("groupby_regression", prefix=temp_benchmark_dir)
        
        expected_cols = [
            "run_id", "timestamp", "env_id", "benchmark_id",
            "time_s", "peak_rss_mb", "status"
        ]
        for col in expected_cols:
            assert col in df.columns
    
    def test_load_history_max_runs(self, temp_benchmark_dir):
        """load_history respects max_runs."""
        df = load_history("groupby_regression", prefix=temp_benchmark_dir, max_runs=2)
        
        # Should only have 2 runs × 2 benchmarks = 4 rows
        assert len(df) == 4
    
    def test_load_history_empty(self, tmp_path):
        """load_history returns empty DataFrame for no history."""
        df = load_history("groupby_regression", prefix=tmp_path / "empty")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestFiltering:
    """Tests for filter functions."""
    
    def test_filter_by_env(self, sample_history_df):
        """filter_by_env filters correctly."""
        df = sample_history_df.copy()
        # Add row with different env
        df.loc[len(df)] = df.iloc[0].copy()
        df.loc[len(df)-1, "env_id"] = "Different_Env"
        
        filtered = filter_by_env(df, "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0")
        assert len(filtered) == 10  # Original 10 rows
    
    def test_filter_by_benchmark(self, sample_history_df):
        """filter_by_benchmark filters correctly."""
        filtered = filter_by_benchmark(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        assert len(filtered) == 10
    
    def test_filter_passed_only(self, sample_history_df):
        """filter_passed_only removes failed."""
        df = sample_history_df.copy()
        df.loc[0, "status"] = "FAILED"
        
        filtered = filter_passed_only(df)
        assert len(filtered) == 9


class TestGetBaseline:
    """Tests for get_baseline()"""
    
    def test_get_baseline_returns_dict(self, sample_history_df):
        """get_baseline returns dict with expected keys."""
        baseline = get_baseline(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        
        assert baseline is not None
        assert "time_s" in baseline
        assert "peak_rss_mb" in baseline
        assert "n_samples" in baseline
        assert "method" in baseline
    
    def test_get_baseline_uses_median(self, sample_history_df):
        """get_baseline uses median by default."""
        baseline = get_baseline(sample_history_df, "v5_batch_fit:S1:n_jobs=1")
        
        assert baseline["method"] == "median"
    
    def test_get_baseline_limits_samples(self, sample_history_df):
        """get_baseline respects n_runs limit."""
        baseline = get_baseline(sample_history_df, "v5_batch_fit:S1:n_jobs=1", n_runs=5)
        
        assert baseline["n_samples"] == 5
    
    def test_get_baseline_no_history(self, sample_history_df):
        """get_baseline returns None for unknown benchmark."""
        baseline = get_baseline(sample_history_df, "unknown:S1:n_jobs=1")
        
        assert baseline is None


class TestSummarizeHistory:
    """Tests for summarize_history()"""
    
    def test_summarize_history(self, sample_history_df):
        """summarize_history returns expected structure."""
        summary = summarize_history(sample_history_df)
        
        assert summary["n_runs"] == 10
        assert summary["n_benchmarks"] == 1
        assert summary["pass_rate"] == 1.0
        assert len(summary["environments"]) == 1


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestCheckRegression:
    """Tests for check_regression()"""
    
    def test_no_regression_within_threshold(self, sample_history_df):
        """No regression when within threshold."""
        # Create current result similar to history
        current = BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.52,  # Within 10% of 0.5
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=510,  # Within 15% of 500
            status="OK",
        )
        
        result = check_regression(current, sample_history_df)
        
        assert result.has_baseline
        assert not result.has_regression
        assert not result.time_regression
        assert not result.memory_regression
    
    def test_time_regression_detected(self, sample_history_df):
        """Time regression detected when above threshold."""
        current = BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.6,  # 20% above 0.5
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=500,
            status="OK",
        )
        
        result = check_regression(current, sample_history_df)
        
        assert result.has_regression
        assert result.time_regression
        assert result.time_change_pct > 10.0
    
    def test_memory_regression_detected(self, sample_history_df):
        """Memory regression detected when above threshold."""
        current = BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.5,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=600,  # 20% above 500
            status="OK",
        )
        
        result = check_regression(current, sample_history_df)
        
        assert result.has_regression
        assert result.memory_regression
        assert result.memory_change_pct > 15.0
    
    def test_no_baseline_no_regression(self):
        """No regression when no baseline available."""
        current = BenchmarkResult(
            id="unknown:S1:n_jobs=1",
            name="unknown",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=1.0,
            time_std_s=0.1,
            n_runs=3,
            peak_rss_mb=1000,
            status="OK",
        )
        
        empty_df = pd.DataFrame(columns=[
            "benchmark_id", "time_s", "peak_rss_mb", "status"
        ])
        
        result = check_regression(current, empty_df)
        
        assert not result.has_baseline
        assert not result.has_regression
    
    def test_insufficient_samples_no_regression(self):
        """No regression when baseline has fewer than MIN_BASELINE_SAMPLES."""
        # Create history with only 2 samples (less than MIN_BASELINE_SAMPLES=3)
        rows = []
        for i in range(2):
            rows.append({
                "run_id": f"run_{i}",
                "timestamp": pd.Timestamp(f"2024-12-{10+i}T10:00:00Z"),
                "commit": f"abc{i}",
                "branch": "main",
                "env_id": "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0",
                "run_mode": "gate",
                "suite": "quick",
                "hostname": "test",
                "benchmark_id": "v5_batch_fit:S1:n_jobs=1",
                "name": "v5_batch_fit",
                "scenario": "S1",
                "n_jobs": 1,
                "time_s": 0.5,
                "time_std_s": 0.01,
                "peak_rss_mb": 500,
                "throughput_rows_per_sec": 100000,
                "status": "OK",
            })
        
        insufficient_df = pd.DataFrame(rows)
        
        # Current result with 50% regression (would trigger if enough samples)
        current = BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.75,  # 50% slower
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=500,
            status="OK",
        )
        
        result = check_regression(current, insufficient_df)
        
        # Should NOT trigger regression due to insufficient history
        assert not result.has_baseline  # Treated as no baseline
        assert not result.has_regression
        assert result.baseline_samples == 2  # But samples count is recorded
        assert MIN_BASELINE_SAMPLES == 3  # Verify constant value


class TestDetectRegressions:
    """Tests for detect_regressions()"""
    
    def test_detect_no_regressions(self, sample_history_df):
        """detect_regressions returns empty alarms when no regression."""
        meta = RunMeta.create(subproject="test")
        meta.env_id = "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0"
        
        benchmarks = [
            BenchmarkResult(
                id="v5_batch_fit:S1:n_jobs=1",
                name="v5_batch_fit",
                scenario="S1",
                params={"n_jobs": 1},
                time_s=0.51,  # Within threshold
                time_std_s=0.01,
                n_runs=3,
                peak_rss_mb=505,
                status="OK",
            ),
        ]
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=benchmarks,
            alarms=[],
        )
        
        alarms, results = detect_regressions(run, sample_history_df)
        
        assert len(alarms) == 0
        assert len(results) == 1
    
    def test_detect_regression_creates_alarm(self, sample_history_df):
        """detect_regressions creates alarm for regression."""
        meta = RunMeta.create(subproject="test")
        meta.env_id = "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0"
        
        benchmarks = [
            BenchmarkResult(
                id="v5_batch_fit:S1:n_jobs=1",
                name="v5_batch_fit",
                scenario="S1",
                params={"n_jobs": 1},
                time_s=0.7,  # 40% regression
                time_std_s=0.01,
                n_runs=3,
                peak_rss_mb=500,
                status="OK",
            ),
        ]
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=benchmarks,
            alarms=[],
        )
        
        alarms, results = detect_regressions(run, sample_history_df)
        
        assert len(alarms) == 1
        assert alarms[0].benchmark == "v5_batch_fit"
        assert alarms[0].metric == "time_s"
        assert alarms[0].change_pct > 30.0
    
    def test_skips_failed_benchmarks(self, sample_history_df):
        """detect_regressions skips failed benchmarks."""
        meta = RunMeta.create(subproject="test")
        meta.env_id = "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0"
        
        benchmarks = [
            BenchmarkResult(
                id="v5_batch_fit:S1:n_jobs=1",
                name="v5_batch_fit",
                scenario="S1",
                params={"n_jobs": 1},
                time_s=1.0,
                time_std_s=0.01,
                n_runs=3,
                peak_rss_mb=1000,
                status="FAILED",  # Should be skipped
            ),
        ]
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=benchmarks,
            alarms=[],
        )
        
        alarms, results = detect_regressions(run, sample_history_df)
        
        assert len(alarms) == 0
        assert len(results) == 0  # Failed benchmarks not checked


class TestRegressionOutput:
    """Tests for regression output formatting."""
    
    def test_format_regression_report(self):
        """format_regression_report produces string."""
        results = [
            RegressionResult(
                benchmark_id="v5_batch_fit:S1:n_jobs=1",
                current_time_s=0.5,
                current_rss_mb=500,
                has_baseline=True,
                baseline_time_s=0.5,
                baseline_rss_mb=500,
            ),
        ]
        alarms = []
        
        report = format_regression_report(results, alarms)
        
        assert isinstance(report, str)
        assert "REGRESSION REPORT" in report
        assert "v5_batch_fit" in report
    
    def test_regression_result_to_alarm(self):
        """RegressionResult.to_alarm creates Alarm."""
        result = RegressionResult(
            benchmark_id="v5_batch_fit:S1:n_jobs=1",
            current_time_s=0.6,
            current_rss_mb=500,
            has_baseline=True,
            baseline_time_s=0.5,
            baseline_rss_mb=500,
            time_regression=True,
            time_change_pct=20.0,
            has_regression=True,
        )
        
        alarm = result.to_alarm("v5_batch_fit", "S1", 1)
        
        assert alarm is not None
        assert alarm.benchmark == "v5_batch_fit"
        assert alarm.scenario == "S1"
        assert alarm.metric == "time_s"
        assert alarm.change_pct == 20.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for history + regression pipeline."""
    
    def test_full_pipeline(self, temp_benchmark_dir):
        """Test full history → regression pipeline."""
        # Load history
        history_df = load_history("groupby_regression", prefix=temp_benchmark_dir)
        assert len(history_df) > 0
        
        # Create a new run with regression
        meta = RunMeta.create(subproject="groupby_regression")
        meta.env_id = "Linux_TestCPU_3.9.6_np1.24.0_nb0.58.0"
        
        benchmarks = [
            BenchmarkResult(
                id="v5_batch_fit:S1:n_jobs=1",
                name="v5_batch_fit",
                scenario="S1",
                params={"n_jobs": 1},
                time_s=0.8,  # ~60% regression from ~0.5
                time_std_s=0.01,
                n_runs=3,
                peak_rss_mb=500,
                status="OK",
            ),
        ]
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=benchmarks,
            alarms=[],
        )
        
        # Detect regressions
        alarms, results = detect_regressions(run, history_df)
        
        # Should detect time regression
        assert len(alarms) == 1
        assert alarms[0].metric == "time_s"


# =============================================================================
# RSS SEMANTIC TESTS
# =============================================================================

class TestRSSSemantic:
    """Tests documenting RSS memory measurement semantics."""
    
    def test_rss_is_process_wide_peak(self):
        """
        Document that peak_rss_mb is process-lifetime maximum RSS.
        
        This is NOT a per-benchmark isolated measurement. The value represents
        the highest RSS observed during the benchmark execution window, which
        includes any prior allocations in the same process.
        
        Implications:
        - Cross-run comparisons are valid (same benchmark, different runs)
        - Within-run comparisons may be affected by benchmark ordering
        - Memory threshold (15%) is higher than time (10%) due to this variance
        """
        # This test documents the semantic, not functionality
        # The actual RSS measurement is tested in test_benchmark_phase1.py
        
        # Verify our memory threshold accounts for RSS variance
        assert DEFAULT_MEMORY_THRESHOLD == 0.15  # 15% > 10% time threshold
        assert DEFAULT_TIME_THRESHOLD == 0.10
        
        # Document that we're comparing absolute values, not deltas
        # This is the expected behavior for v1.0
    
    def test_regression_comparison_uses_absolute_rss(self, sample_history_df):
        """Regression detection compares absolute peak_rss_mb values."""
        current = BenchmarkResult(
            id="v5_batch_fit:S1:n_jobs=1",
            name="v5_batch_fit",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.5,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=500,  # Absolute value, not delta
            status="OK",
        )
        
        result = check_regression(current, sample_history_df)
        
        # Baseline is also absolute peak_rss_mb from history
        if result.has_baseline:
            assert result.baseline_rss_mb > 0  # Absolute value
            # memory_change_pct compares absolute values
            assert isinstance(result.memory_change_pct, float)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
