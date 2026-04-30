"""
Phase 13.18.DF: Robust Statistics Extension — Tests

Tests for always-on robust stats (median, MAD, mad_sigma) and
optional stat_fields groups (quantiles, shape).

Covers: hist, hist2d, profile, vector passthrough, edge cases.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def df_gaussian():
    """Large Gaussian sample for precise robust stats verification."""
    np.random.seed(18)
    n = 10000
    return pd.DataFrame({
        'x': np.random.normal(0, 1, n),
        'y': np.random.normal(5, 2, n),
        'z': np.random.uniform(-1, 1, n),
    })


@pytest.fixture
def df_small():
    """Small dataset for edge-case testing."""
    return pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def df_constant():
    """Constant data for MAD=0 edge case."""
    n = 100
    return pd.DataFrame({
        'x': np.full(n, 3.14),
        'y': np.full(n, 2.72),
    })


@pytest.fixture
def df_vector():
    """Multi-column for vector expression tests."""
    np.random.seed(19)
    n = 500
    return pd.DataFrame({
        'y1': np.random.normal(0, 1, n),
        'y2': np.random.normal(1, 2, n),
    })


# =============================================================================
# §1 Base + Robust stats (always present)
# =============================================================================

class TestRobustStatsAlwaysOn:
    """Phase 13.18.DF: median, mad, mad_sigma always in stats dict."""

    def test_base_stats_always_present(self, df_gaussian):
        """Base group keys unchanged from pre-13.18.DF."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        for key in ('n', 'mean', 'std', 'min', 'max'):
            assert key in stats, f"Base key '{key}' missing from stats dict"
        plt.close('all')

    def test_robust_stats_always_present(self, df_gaussian):
        """Robust group keys present WITHOUT stat_fields parameter."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        for key in ('median', 'mad', 'mad_sigma'):
            assert key in stats, (
                f"[Phase 13.18.DF] Robust key '{key}' missing from hist stats dict. "
                f"Keys present: {sorted(stats.keys())}"
            )
        plt.close('all')

    def test_median_correctness(self, df_gaussian):
        """median matches np.nanmedian to full precision."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        expected = float(np.nanmedian(df_gaussian['x'].values))
        assert abs(stats['median'] - expected) < 1e-10, (
            f"[Phase 13.18.DF invariance] median={stats['median']}, "
            f"expected={expected}"
        )
        plt.close('all')

    def test_mad_correctness(self, df_gaussian):
        """MAD matches np.nanmedian(|data - median|) to full precision."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        data = df_gaussian['x'].values
        median = np.nanmedian(data)
        expected_mad = float(np.nanmedian(np.abs(data - median)))
        assert abs(stats['mad'] - expected_mad) < 1e-10, (
            f"[Phase 13.18.DF invariance] mad={stats['mad']}, "
            f"expected={expected_mad}"
        )
        plt.close('all')

    def test_mad_sigma_gaussian(self, df_gaussian):
        """On N(0,1) with n=10000, mad_sigma ≈ 1.0 ± 0.05."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        assert abs(stats['mad_sigma'] - 1.0) < 0.05, (
            f"[Phase 13.18.DF] mad_sigma={stats['mad_sigma']:.4f} on N(0,1) "
            f"with n=10000; expected ≈ 1.0 ± 0.05"
        )
        plt.close('all')


# =============================================================================
# §2 Optional stat_fields groups
# =============================================================================

class TestStatFieldsGroups:
    """Phase 13.18.DF: stat_fields parameter controls optional groups."""

    def test_quantiles_not_present_by_default(self, df_gaussian):
        """Quantile keys absent when stat_fields not set."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50)
        assert 'q16' not in stats, (
            f"[Phase 13.18.DF contract] q16 should NOT be in stats when "
            f"stat_fields is not set. Keys: {sorted(stats.keys())}"
        )
        plt.close('all')

    def test_quantiles_present_on_request(self, df_gaussian):
        """stat_fields='quantiles' adds all quantile keys."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50, stat_fields='quantiles')
        for key in ('q05', 'q16', 'q50', 'q84', 'q95', 'iqr'):
            assert key in stats, (
                f"[Phase 13.18.DF] Quantile key '{key}' missing with "
                f"stat_fields='quantiles'. Keys: {sorted(stats.keys())}"
            )
        # q50 should match median
        assert abs(stats['q50'] - stats['median']) < 1e-10
        plt.close('all')

    def test_shape_present_on_request(self, df_gaussian):
        """stat_fields='shape' adds skewness, kurtosis, robust_to_std."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50, stat_fields='shape')
        for key in ('skewness', 'kurtosis', 'robust_to_std'):
            assert key in stats, (
                f"[Phase 13.18.DF] Shape key '{key}' missing with "
                f"stat_fields='shape'. Keys: {sorted(stats.keys())}"
            )
        # For N(0,1): skewness ≈ 0 ± 0.1, kurtosis ≈ 0 ± 0.15
        assert abs(stats['skewness']) < 0.1, (
            f"skewness={stats['skewness']:.4f} on N(0,1), expected ≈ 0"
        )
        assert abs(stats['kurtosis']) < 0.15, (
            f"kurtosis={stats['kurtosis']:.4f} on N(0,1), expected ≈ 0"
        )
        # robust_to_std ≈ 1.0 for Gaussian
        assert abs(stats['robust_to_std'] - 1.0) < 0.1, (
            f"robust_to_std={stats['robust_to_std']:.4f}, expected ≈ 1.0 for Gaussian"
        )
        plt.close('all')

    def test_stat_fields_all(self, df_gaussian):
        """stat_fields='all' includes every group."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50, stat_fields='all')
        all_keys = {'n', 'mean', 'std', 'min', 'max',
                     'median', 'mad', 'mad_sigma',
                     'q05', 'q16', 'q50', 'q84', 'q95', 'iqr',
                     'skewness', 'kurtosis', 'robust_to_std'}
        missing = all_keys - set(stats.keys())
        assert not missing, (
            f"[Phase 13.18.DF] stat_fields='all' missing keys: {missing}"
        )
        plt.close('all')

    def test_stat_fields_list_combo(self, df_gaussian):
        """stat_fields=['quantiles', 'shape'] combines both groups."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist("x", bins=50, stat_fields=['quantiles', 'shape'])
        for key in ('q16', 'q84', 'skewness', 'kurtosis'):
            assert key in stats, f"Key '{key}' missing with combined stat_fields"
        plt.close('all')

    def test_stat_fields_invalid_raises(self, df_gaussian):
        """Unrecognized stat_fields value raises ValueError."""
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="unrecognized group"):
            drawer.hist("x", bins=50, stat_fields='quantile')  # missing 's'
        plt.close('all')


# =============================================================================
# §3 hist2d and profile robust stats
# =============================================================================

class TestHist2dProfileRobustStats:
    """Phase 13.18.DF: 2D methods get per-axis robust stats."""

    def test_hist2d_robust_stats_per_axis(self, df_gaussian):
        """hist2d returns median_x, mad_x, mad_sigma_x, median_y, etc."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist2d("y:x", bins=50)
        for suffix in ('_x', '_y'):
            for key in ('median', 'mad', 'mad_sigma'):
                full_key = f"{key}{suffix}"
                assert full_key in stats, (
                    f"[Phase 13.18.DF] hist2d missing '{full_key}'. "
                    f"Keys: {sorted(stats.keys())}"
                )
        # corr must still be present (R3)
        assert 'corr' in stats, "hist2d missing 'corr' key (R3 regression)"
        # median_x should match np.nanmedian of x data
        expected = float(np.nanmedian(df_gaussian['x'].values))
        assert abs(stats['median_x'] - expected) < 1e-10
        plt.close('all')

    def test_profile_robust_summary_stats(self, df_gaussian):
        """profile() summary dict has per-axis robust stats."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=50)
        for suffix in ('_x', '_y'):
            for key in ('median', 'mad', 'mad_sigma'):
                full_key = f"{key}{suffix}"
                assert full_key in stats, (
                    f"[Phase 13.18.DF] profile missing '{full_key}'. "
                    f"Keys: {sorted(stats.keys())}"
                )
        plt.close('all')

    def test_hist2d_stat_fields_quantiles(self, df_gaussian):
        """hist2d stat_fields='quantiles' adds per-axis quantile keys."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.hist2d("y:x", bins=50, stat_fields='quantiles')
        for suffix in ('_x', '_y'):
            for key in ('q05', 'q16', 'q84', 'q95'):
                full_key = f"{key}{suffix}"
                assert full_key in stats, (
                    f"hist2d missing '{full_key}' with stat_fields='quantiles'"
                )
        plt.close('all')


# =============================================================================
# §4 Vector passthrough + edge cases
# =============================================================================

class TestVectorAndEdgeCases:
    """Phase 13.18.DF: vector dispatch + edge cases."""

    def test_vector_stat_fields_passthrough(self, df_vector):
        """hist("[y1,y2]", stat_fields='all') → both stats dicts have all fields."""
        drawer = DFDraw(df_vector)
        _, _, stats_list = drawer.hist("[y1,y2]", bins=50, stat_fields='all')
        assert len(stats_list) == 2, f"Expected 2 stats, got {len(stats_list)}"
        for i, s in enumerate(stats_list):
            for key in ('median', 'mad', 'mad_sigma', 'q16', 'q84', 'skewness'):
                assert key in s, (
                    f"[Phase 13.18.DF vector] stats[{i}] missing '{key}' "
                    f"with stat_fields='all'"
                )
        plt.close('all')

    def test_robust_stats_edge_cases(self, df_constant, df_small):
        """Edge cases: constant data, small n."""
        # Constant data → mad=0, mad_sigma=0
        drawer_c = DFDraw(df_constant)
        _, _, stats_c = drawer_c.hist("x", bins=10, stat_fields='shape')
        assert stats_c['mad'] == 0.0, f"Constant data: mad should be 0, got {stats_c['mad']}"
        assert stats_c['mad_sigma'] == 0.0, f"Constant data: mad_sigma should be 0"
        # robust_to_std should be NaN (division by zero guarded)
        assert np.isnan(stats_c['robust_to_std']), (
            f"Constant data: robust_to_std should be NaN, got {stats_c['robust_to_std']}"
        )
        plt.close('all')

        # Small n (n=5 < 10) → skewness/kurtosis = NaN
        drawer_s = DFDraw(df_small)
        _, _, stats_s = drawer_s.hist("x", bins=5, stat_fields='shape')
        assert np.isnan(stats_s['skewness']), (
            f"n=5: skewness should be NaN, got {stats_s['skewness']}"
        )
        assert np.isnan(stats_s['kurtosis']), (
            f"n=5: kurtosis should be NaN, got {stats_s['kurtosis']}"
        )
        plt.close('all')
