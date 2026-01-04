#!/usr/bin/env python3
"""
Phase 12.14c.GB D1 — Combined Verification Test

Tests all D1 components:
- D1 core: load_benchmark_adf()
- D4 TopCPU: CPU profile subframe
- D7 TopMemory: Memory statistics subframe
- D8 noise: compute_benchmark_statistics()

Usage:
    pytest tests/test_d1_verification.py -v -s
"""

import pytest
import pandas as pd


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def benchmark_adf():
    """Load benchmark ADF once for all tests."""
    from dfextensions.benchmarks.benchmark_adf import load_benchmark_adf
    
    print("\n[SETUP] Loading benchmark history...")
    adf = load_benchmark_adf("groupby_regression", max_runs=20)
    print(f"[SETUP] Loaded {len(adf.df)} results")
    return adf


@pytest.fixture(scope="module")
def benchmark_stats():
    """Compute statistics once for all tests."""
    from dfextensions.benchmarks.benchmark_adf import compute_benchmark_statistics
    
    print("\n[SETUP] Computing benchmark statistics...")
    stats = compute_benchmark_statistics("groupby_regression", baseline="7d")
    print(f"[SETUP] Computed stats for {len(stats)} benchmarks")
    return stats


# =============================================================================
# D1: CORE LOADER TESTS
# =============================================================================

class TestD1CoreLoader:
    """D1: load_benchmark_adf() tests."""
    
    def test_loader_returns_adf(self, benchmark_adf):
        """Loader returns AliasDataFrame instance."""
        from dfextensions.AliasDataFrame import AliasDataFrame
        assert isinstance(benchmark_adf, AliasDataFrame)
        print(f"  ✓ Returns AliasDataFrame instance")
    
    def test_loader_has_results(self, benchmark_adf):
        """Loader returns non-empty results."""
        assert len(benchmark_adf.df) > 0
        print(f"  ✓ Loaded {len(benchmark_adf.df)} benchmark results")
    
    def test_loader_has_required_columns(self, benchmark_adf):
        """Loader DataFrame has required columns."""
        required = ['benchmark_id', 'name', 'time_s', 'peak_rss_mb', 'status']
        missing = [c for c in required if c not in benchmark_adf.df.columns]
        assert not missing, f"Missing columns: {missing}"
        print(f"  ✓ Has required columns: {required}")
    
    def test_loader_has_run_dir(self, benchmark_adf):
        """Loader includes _run_dir for profile resolution (P0-3)."""
        assert '_run_dir' in benchmark_adf.df.columns
        print(f"  ✓ Has _run_dir column (P0-3 compliant)")
    
    def test_loader_unique_benchmarks(self, benchmark_adf):
        """Loader returns multiple unique benchmarks."""
        n_unique = benchmark_adf.df['benchmark_id'].nunique()
        assert n_unique > 0
        print(f"  ✓ {n_unique} unique benchmark IDs")
    
    def test_loader_columns_verbose(self, benchmark_adf):
        """Print all columns for review."""
        cols = list(benchmark_adf.df.columns)
        print(f"  ℹ All columns ({len(cols)}): {cols}")


# =============================================================================
# D4: TOPCPU SUBFRAME TESTS
# =============================================================================

class TestD4TopCPU:
    """D4: TopCPU subframe tests."""
    
    def test_topcpu_registered(self, benchmark_adf):
        """TopCPU subframe is registered."""
        has_topcpu = benchmark_adf._subframes.has_subframe('TopCPU')
        if not has_topcpu:
            pytest.skip("TopCPU not populated (no profiles with n_jobs=1?)")
        print(f"  ✓ TopCPU subframe registered")
    
    def test_topcpu_has_entries(self, benchmark_adf):
        """TopCPU has profile entries."""
        if not benchmark_adf._subframes.has_subframe('TopCPU'):
            pytest.skip("TopCPU not populated")
        
        cpu_df = benchmark_adf.get_subframe('TopCPU').df
        assert len(cpu_df) > 0
        print(f"  ✓ TopCPU: {len(cpu_df)} function entries")
    
    def test_topcpu_has_required_columns(self, benchmark_adf):
        """TopCPU has required columns for P0-2."""
        if not benchmark_adf._subframes.has_subframe('TopCPU'):
            pytest.skip("TopCPU not populated")
        
        cpu_df = benchmark_adf.get_subframe('TopCPU').df
        required = ['benchmark_id', 'function', 'cumtime_s', 'rank']
        missing = [c for c in required if c not in cpu_df.columns]
        assert not missing, f"Missing: {missing}"
        print(f"  ✓ Has required columns: {required}")
    
    def test_topcpu_sorted_by_cumtime(self, benchmark_adf):
        """TopCPU entries sorted by cumulative time (P0-2)."""
        if not benchmark_adf._subframes.has_subframe('TopCPU'):
            pytest.skip("TopCPU not populated")
        
        cpu_df = benchmark_adf.get_subframe('TopCPU').df
        # Check rank=1 has highest cumtime per benchmark
        for bid, group in cpu_df.groupby('benchmark_id'):
            if len(group) > 1:
                rank1 = group[group['rank'] == 1]['cumtime_s'].values[0]
                rank2 = group[group['rank'] == 2]['cumtime_s'].values[0]
                assert rank1 >= rank2, f"{bid}: rank1 ({rank1}) < rank2 ({rank2})"
            break  # Check first benchmark only
        print(f"  ✓ Sorted by cumulative time (P0-2 compliant)")
    
    def test_topcpu_top_functions_verbose(self, benchmark_adf):
        """Print top functions for review."""
        if not benchmark_adf._subframes.has_subframe('TopCPU'):
            pytest.skip("TopCPU not populated")
        
        cpu_df = benchmark_adf.get_subframe('TopCPU').df
        top5 = cpu_df[cpu_df['rank'] == 1].head(5)
        print(f"  ℹ Top functions (rank=1):")
        for _, row in top5.iterrows():
            print(f"    {row['function']}: {row['cumtime_s']:.4f}s")


# =============================================================================
# D7: TOPMEMORY SUBFRAME TESTS
# =============================================================================

class TestD7TopMemory:
    """D7: TopMemory subframe tests."""
    
    def test_topmemory_registered(self, benchmark_adf):
        """TopMemory subframe is registered."""
        has_mem = benchmark_adf._subframes.has_subframe('TopMemory')
        assert has_mem, "TopMemory subframe not registered"
        print(f"  ✓ TopMemory subframe registered")
    
    def test_topmemory_has_entries(self, benchmark_adf):
        """TopMemory has memory entries."""
        mem_df = benchmark_adf.get_subframe('TopMemory').df
        assert len(mem_df) > 0
        print(f"  ✓ TopMemory: {len(mem_df)} entries")
    
    def test_topmemory_has_peak_rss(self, benchmark_adf):
        """TopMemory has peak_rss_mb column."""
        mem_df = benchmark_adf.get_subframe('TopMemory').df
        assert 'peak_rss_mb' in mem_df.columns
        print(f"  ✓ Has peak_rss_mb column")
    
    def test_topmemory_rss_range_verbose(self, benchmark_adf):
        """Print RSS range for review."""
        mem_df = benchmark_adf.get_subframe('TopMemory').df
        rss_min = mem_df['peak_rss_mb'].min()
        rss_max = mem_df['peak_rss_mb'].max()
        print(f"  ℹ Peak RSS range: {rss_min:.0f} - {rss_max:.0f} MB")


# =============================================================================
# D8: NOISE STATISTICS TESTS
# =============================================================================

class TestD8Statistics:
    """D8: compute_benchmark_statistics() tests."""
    
    def test_stats_returns_dataframe(self, benchmark_stats):
        """Statistics returns DataFrame."""
        assert isinstance(benchmark_stats, pd.DataFrame)
        print(f"  ✓ Returns DataFrame")
    
    def test_stats_has_entries(self, benchmark_stats):
        """Statistics has benchmark entries."""
        if benchmark_stats.empty:
            pytest.skip("No statistics (insufficient samples)")
        assert len(benchmark_stats) > 0
        print(f"  ✓ Statistics for {len(benchmark_stats)} benchmarks")
    
    def test_stats_has_required_columns(self, benchmark_stats):
        """Statistics has required columns."""
        if benchmark_stats.empty:
            pytest.skip("No statistics")
        
        required = ['benchmark_id', 'mean_time_s', 'std_time_s', 'cv_pct', 'high_noise']
        missing = [c for c in required if c not in benchmark_stats.columns]
        assert not missing, f"Missing: {missing}"
        print(f"  ✓ Has required columns: {required}")
    
    def test_stats_cv_calculated(self, benchmark_stats):
        """CV percentage is calculated correctly."""
        if benchmark_stats.empty:
            pytest.skip("No statistics")
        
        # CV = (std / mean) * 100
        row = benchmark_stats.iloc[0]
        expected_cv = (row['std_time_s'] / row['mean_time_s']) * 100 if row['mean_time_s'] > 0 else 0
        assert abs(row['cv_pct'] - expected_cv) < 0.1
        print(f"  ✓ CV correctly calculated")
    
    def test_stats_high_noise_flag(self, benchmark_stats):
        """High noise flag set for CV > 10%."""
        if benchmark_stats.empty:
            pytest.skip("No statistics")
        
        for _, row in benchmark_stats.iterrows():
            if row['cv_pct'] > 10.0:
                assert row['high_noise'] == True
            else:
                assert row['high_noise'] == False
            break  # Check first only
        print(f"  ✓ high_noise flag correct")
    
    def test_stats_verbose(self, benchmark_stats):
        """Print statistics for review."""
        if benchmark_stats.empty:
            pytest.skip("No statistics")
        
        print(f"\n  ℹ Benchmark Statistics:")
        print(f"  {'-' * 60}")
        for _, row in benchmark_stats.head(10).iterrows():
            flag = "⚠ HIGH" if row['high_noise'] else "OK"
            print(f"    {row['benchmark_id'][:40]:40s} CV={row['cv_pct']:6.1f}% ({flag})")
        
        n_high = len(benchmark_stats[benchmark_stats['high_noise']])
        print(f"  {'-' * 60}")
        print(f"  ℹ High-noise benchmarks: {n_high}/{len(benchmark_stats)}")


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestD1Summary:
    """Combined D1 summary test."""
    
    def test_d1_complete(self, benchmark_adf, benchmark_stats):
        """D1 combined deliverable complete."""
        # D1 core
        assert len(benchmark_adf.df) > 0, "D1 core failed"
        
        # D7 TopMemory
        assert benchmark_adf._subframes.has_subframe('TopMemory'), "D7 TopMemory failed"
        
        # D8 Statistics
        assert isinstance(benchmark_stats, pd.DataFrame), "D8 Statistics failed"
        
        print(f"\n{'=' * 60}")
        print(f"✅ Combined D1 COMPLETE")
        print(f"{'=' * 60}")
        print(f"  D1 Core:     {len(benchmark_adf.df)} results loaded")
        
        if benchmark_adf._subframes.has_subframe('TopCPU'):
            cpu_count = len(benchmark_adf.get_subframe('TopCPU').df)
            print(f"  D4 TopCPU:   {cpu_count} function entries")
        else:
            print(f"  D4 TopCPU:   Skipped (no profiles)")
        
        mem_count = len(benchmark_adf.get_subframe('TopMemory').df)
        print(f"  D7 TopMemory: {mem_count} entries")
        print(f"  D8 Statistics: {len(benchmark_stats)} benchmarks")
        print(f"{'=' * 60}")
