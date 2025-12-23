"""
Phase 12.10.BF-Core: Benchmark Framework Tests

Tests for Phase 1 deliverables:
- schema.py: Validation, env_id format
- profiler.py: RSS measurement
- runner.py: Smoke test
- scenarios.py: Data generation
- bench_v5.py: Benchmark execution

Directory Structure (run pytest from O2DPG/UTILS):
    O2DPG/UTILS/
    └── dfextensions/
        ├── benchmarks/           ← Shared framework
        │   ├── schema.py
        │   ├── profiler.py
        │   ├── runner.py
        │   └── tests/
        │       └── test_benchmark_phase1.py  ← This file
        └── groupby_regression/
            └── benchmarks/
                ├── scenarios.py
                └── bench_v5.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

# =============================================================================
# IMPORTS - Use full paths from O2DPG/UTILS
# =============================================================================

# Shared framework imports (from dfextensions/benchmarks/)
from dfextensions.benchmarks.schema import (
    SCHEMA_VERSION,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_MEMORY_THRESHOLD,
    get_git_info,
    get_tool_versions,
    build_env_id,
    BenchmarkResult,
    RunMeta,
    RunSummary,
    BenchmarkRun,
    Alarm,
    validate_run,
    get_benchmark_prefix,
    get_results_path,
)

from dfextensions.benchmarks.profiler import (
    check_platform_support,
    get_peak_rss_mb,
    get_current_rss_mb,
    track_peak_rss,
    track_memory_detailed,
    MemoryGuard,
    run_benchmark_with_memory,
)

from dfextensions.benchmarks.runner import run_single_benchmark

# Subproject imports (from dfextensions/groupby_regression/benchmarks/)
from dfextensions.groupby_regression.benchmarks.scenarios import (
    SCENARIOS,
    QUICK_SCENARIOS,
    get_scenario,
    get_scenarios,
    create_test_data,
    create_minimal_warmup_data,
)

from dfextensions.groupby_regression.benchmarks.bench_v5 import (
    bench_v5_batch_fit,
    warmup_v5,
    reset_warmup,
    get_benchmarks,
)


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestSchema:
    """Tests for schema.py"""
    
    def test_schema_version(self):
        """Schema version is set."""
        assert SCHEMA_VERSION == 1
    
    def test_default_warmup_runs(self):
        """Default warmup is 2 (per review requirement)."""
        assert DEFAULT_WARMUP_RUNS == 2
    
    def test_default_memory_threshold(self):
        """Default memory threshold is 15% (per review requirement)."""
        assert DEFAULT_MEMORY_THRESHOLD == 0.15
    
    def test_git_info_structure(self):
        """get_git_info returns expected structure."""
        git = get_git_info()
        assert "commit" in git
        assert "branch" in git
        assert "dirty" in git
        assert isinstance(git["dirty"], bool)
    
    def test_tool_versions_structure(self):
        """get_tool_versions returns expected packages."""
        versions = get_tool_versions()
        assert "numpy" in versions
        assert "pandas" in versions
        assert "numba" in versions
    
    def test_env_id_format(self):
        """env_id includes numpy and numba versions (per review requirement)."""
        env_id = build_env_id(
            plat="Linux",
            cpu_model="TestCPU",
            python_version="3.9.6",
            numpy_version="1.24.0",
            numba_version="0.58.0",
        )
        assert "np1.24.0" in env_id
        assert "nb0.58.0" in env_id
        assert "Linux" in env_id
        assert "3.9.6" in env_id
    
    def test_run_meta_creation(self):
        """RunMeta.create() populates all required fields."""
        meta = RunMeta.create(subproject="test_project")
        
        assert meta.schema_version == SCHEMA_VERSION
        assert meta.subproject == "test_project"
        assert meta.run_id  # Not empty
        assert meta.env_id  # Not empty
        assert meta.timestamp  # Not empty
        assert "np" in meta.env_id  # numpy version
        assert "nb" in meta.env_id  # numba version
        assert meta.warmup_runs == DEFAULT_WARMUP_RUNS
    
    def test_run_id_includes_pid(self):
        """run_id includes PID (per review requirement)."""
        meta = RunMeta.create(subproject="test")
        pid = str(os.getpid())
        assert pid in meta.run_id
    
    def test_benchmark_result_from_timing(self):
        """BenchmarkResult.from_timing creates valid result."""
        result = BenchmarkResult.from_timing(
            name="test_bench",
            scenario="S1",
            params={"n_jobs": 1, "n_rows": 1000},
            times=[0.1, 0.11, 0.09],
            peak_rss_mb=100.0,
        )
        
        assert result.id == "test_bench:S1:n_jobs=1"
        assert result.status == "OK"
        assert 0.09 <= result.time_s <= 0.11
        assert result.time_std_s > 0
        assert result.peak_rss_mb == 100.0
        assert result.throughput_rows_per_sec == 1000 / result.time_s
    
    def test_benchmark_run_to_dict(self):
        """BenchmarkRun serializes to valid dict."""
        meta = RunMeta.create(subproject="test")
        result = BenchmarkResult(
            id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.1,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=100.0,
        )
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=[result],
            alarms=[],
        )
        run.update_summary()
        
        d = run.to_dict()
        
        assert d["meta"]["schema_version"] == 1
        assert d["summary"]["n_benchmarks"] == 1
        assert d["summary"]["n_passed"] == 1
        assert len(d["benchmarks"]) == 1
        assert d["benchmarks"][0]["id"] == "test:S1:n_jobs=1"
    
    def test_validate_run_valid(self):
        """validate_run accepts valid run."""
        meta = RunMeta.create(subproject="test")
        result = BenchmarkResult(
            id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            params={},
            time_s=0.1,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=100.0,
        )
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=[result],
            alarms=[],
        )
        
        errors = validate_run(run)
        assert errors == []
    
    def test_validate_run_invalid(self):
        """validate_run rejects invalid run."""
        meta = RunMeta()  # Empty meta
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(),
            benchmarks=[],
            alarms=[],
        )
        
        errors = validate_run(run)
        assert len(errors) > 0  # Should have validation errors
    
    def test_save_load_roundtrip(self):
        """BenchmarkRun saves and loads correctly."""
        meta = RunMeta.create(subproject="test")
        result = BenchmarkResult(
            id="test:S1:n_jobs=1",
            name="test",
            scenario="S1",
            params={"n_jobs": 1},
            time_s=0.123,
            time_std_s=0.01,
            n_runs=3,
            peak_rss_mb=100.5,
        )
        
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(total_time_s=0.123),
            benchmarks=[result],
            alarms=[],
        )
        run.update_summary()
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            run.save(f.name)
            loaded = BenchmarkRun.load(f.name)
        
        assert loaded.meta.run_id == meta.run_id
        assert loaded.benchmarks[0].time_s == 0.123
        assert loaded.benchmarks[0].peak_rss_mb == 100.5
        
        os.unlink(f.name)


# =============================================================================
# PROFILER TESTS
# =============================================================================

class TestProfiler:
    """Tests for profiler.py"""
    
    def test_platform_support(self):
        """Platform check returns tuple."""
        supported, msg = check_platform_support()
        assert isinstance(supported, bool)
        assert isinstance(msg, str)
    
    def test_peak_rss_positive(self):
        """get_peak_rss_mb returns positive value."""
        rss = get_peak_rss_mb()
        assert rss > 0
    
    def test_peak_rss_sanity(self):
        """get_peak_rss_mb returns reasonable value."""
        rss = get_peak_rss_mb()
        assert rss < 100_000  # Less than 100 GB
    
    def test_track_peak_rss(self):
        """track_peak_rss context manager works."""
        with track_peak_rss() as stats:
            data = np.zeros(1_000_000)  # Allocate ~8 MB
        
        assert stats.peak_rss_mb > 0
    
    def test_track_memory_detailed(self):
        """track_memory_detailed captures allocations."""
        with track_memory_detailed(top_n=5) as stats:
            data = np.zeros(100_000)
        
        assert stats.peak_rss_mb > 0
        assert stats.peak_tracemalloc_mb is not None
        assert stats.top_allocations is not None
    
    def test_memory_guard(self):
        """MemoryGuard checks limit."""
        guard = MemoryGuard(limit_mb=100_000)  # 100 GB - should pass
        assert guard.check() == True
        
        guard_small = MemoryGuard(limit_mb=1)  # 1 MB - should fail
        assert guard_small.check() == False
    
    def test_memory_guard_raises(self):
        """MemoryGuard.assert_within_limit raises on exceed."""
        guard = MemoryGuard(limit_mb=1)  # Impossibly small
        
        with pytest.raises(MemoryError):
            guard.assert_within_limit()
    
    def test_run_benchmark_with_memory(self):
        """run_benchmark_with_memory times function."""
        def dummy():
            return np.sum(np.arange(1000))
        
        times, mem_stats, result = run_benchmark_with_memory(
            dummy,
            n_runs=3,
            warmup_runs=1,
        )
        
        assert len(times) == 3
        assert all(t > 0 for t in times)
        assert mem_stats.peak_rss_mb > 0
        assert result == np.sum(np.arange(1000))


# =============================================================================
# SCENARIOS TESTS
# =============================================================================

class TestScenarios:
    """Tests for scenarios.py"""
    
    def test_scenarios_defined(self):
        """All scenarios S1-S5 are defined."""
        for name in ["S1", "S2", "S3", "S4", "S5"]:
            assert name in SCENARIOS
    
    def test_quick_scenarios(self):
        """Quick suite has S1-S4."""
        assert QUICK_SCENARIOS == ["S1", "S2", "S3", "S4"]
    
    def test_scenario_scaling(self):
        """Scenarios scale by 2x."""
        s1 = get_scenario("S1")
        s2 = get_scenario("S2")
        s3 = get_scenario("S3")
        s4 = get_scenario("S4")
        
        assert s2.n_rows == s1.n_rows * 2
        assert s3.n_rows == s2.n_rows * 2
        assert s4.n_rows == s3.n_rows * 2
    
    def test_n_fits_fixed(self):
        """All scenarios have n_fits=6."""
        for name in SCENARIOS:
            assert SCENARIOS[name].n_fits == 6
    
    def test_create_test_data(self):
        """create_test_data generates correct shape."""
        scen = get_scenario("S1")
        df = create_test_data(scen)
        
        assert len(df) == scen.n_rows
        assert "group" in df.columns
        assert "x1" in df.columns
        assert "y1" in df.columns
        assert "w" in df.columns
    
    def test_create_test_data_reproducible(self):
        """create_test_data is reproducible with seed."""
        scen = get_scenario("S1")
        df1 = create_test_data(scen, seed=42)
        df2 = create_test_data(scen, seed=42)
        
        assert df1.equals(df2)
    
    def test_create_minimal_warmup_data(self):
        """Warmup data is small."""
        df = create_minimal_warmup_data()
        assert len(df) == 200  # 10 groups × 20 rows


# =============================================================================
# BENCH_V5 TESTS
# =============================================================================

class TestBenchV5:
    """Tests for bench_v5.py"""
    
    def test_get_benchmarks(self):
        """get_benchmarks returns benchmark specs."""
        benchmarks = get_benchmarks(suite="quick")
        
        assert len(benchmarks) >= 1
        assert benchmarks[0]["name"] == "v5_batch_fit"
        assert "S1" in benchmarks[0]["scenarios"]
    
    def test_warmup(self):
        """warmup_v5 completes without error."""
        reset_warmup()
        warmup_v5(n_jobs=1)  # Should not raise
    
    def test_bench_v5_batch_fit_executes(self):
        """bench_v5_batch_fit runs and returns result."""
        warmup_v5(n_jobs=1)
        
        result = bench_v5_batch_fit(scenario="S1", n_jobs=1)
        
        assert "n_groups_output" in result
        assert "n_rows_input" in result
        assert result["n_groups_output"] == 500  # S1 has 500 groups
        assert result["n_rows_input"] == 50_000  # S1 has 50K rows


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_runner_smoke(self):
        """Runner executes minimal benchmark."""
        warmup_v5(n_jobs=1)
        
        result = run_single_benchmark(
            name="v5_batch_fit",
            func=bench_v5_batch_fit,
            scenario="S1",
            params={"n_jobs": 1},
            n_runs=1,
            warmup_runs=0,
        )
        
        assert result.status == "OK"
        assert result.time_s > 0
        assert result.peak_rss_mb > 0
    
    def test_full_pipeline(self):
        """Full pipeline: create meta, run benchmark, validate, save."""
        # Create metadata
        meta = RunMeta.create(subproject="groupby_regression", suite="quick")
        
        # Run benchmark
        warmup_v5(n_jobs=1)
        result = run_single_benchmark(
            name="v5_batch_fit",
            func=bench_v5_batch_fit,
            scenario="S1",
            params={"n_jobs": 1},
            n_runs=1,
            warmup_runs=0,
        )
        
        # Create run
        run = BenchmarkRun(
            meta=meta,
            summary=RunSummary(total_time_s=result.time_s),
            benchmarks=[result],
            alarms=[],
        )
        run.update_summary()
        
        # Validate
        errors = validate_run(run)
        assert errors == [], f"Validation errors: {errors}"
        
        # Save and verify JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            run.save(f.name)
            
            with open(f.name) as rf:
                data = json.load(rf)
        
        # Verify JSON structure
        assert data["meta"]["schema_version"] == 1
        assert "np" in data["meta"]["env_id"]
        assert "nb" in data["meta"]["env_id"]
        assert data["meta"]["warmup_runs"] == 2
        assert data["summary"]["n_passed"] == 1
        assert data["benchmarks"][0]["peak_rss_mb"] > 0
        
        os.unlink(f.name)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
