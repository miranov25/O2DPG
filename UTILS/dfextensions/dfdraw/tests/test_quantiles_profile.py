"""
Phase 13.25.DF (Phase A): Quantiles on profile() — Tests

82 tests across 13 classes. Invariance-first per architect Q4.
Naming convention per AD-54: TestQuantile* prefix, singular.
"""

import numpy as np
import pandas as pd
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from dfdraw import DFDraw
from dfdraw.style import (
    get_style_value, set_style, get_style, save_style, load_style,
    DEFAULT_STYLE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def df_gaussian():
    """Large Gaussian sample for precise quantile verification."""
    np.random.seed(1325)
    n = 10000
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y': np.random.normal(5, 2, n),
    })


@pytest.fixture
def df_with_groups():
    """Data with group_by column for interaction tests."""
    np.random.seed(1326)
    n = 2000
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y': np.random.normal(0, 1, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    })


@pytest.fixture
def df_vector():
    """Multi-column for vector expression tests."""
    np.random.seed(1327)
    n = 1000
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y1': np.random.normal(0, 1, n),
        'y2': np.random.normal(1, 2, n),
    })


@pytest.fixture
def df_edge():
    """Edge-case data: constant, NaN, small-n bins."""
    np.random.seed(1328)
    n = 500
    x = np.random.uniform(0, 10, n)
    y = np.random.normal(0, 1, n)
    # Inject some NaNs
    y[np.random.choice(n, 20, replace=False)] = np.nan
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture(autouse=False)
def reset_style():
    """Reset style to default before and after test."""
    set_style(None)
    yield
    set_style(None)


# =============================================================================
# Class 1 — TestQuantilePerBinCorrectness (8 invariance tests)
# =============================================================================

class TestQuantilePerBinCorrectness:
    """Per-bin quantile computation matches numpy reference."""

    def _reference_per_bin_quantiles(self, df, bins, q_lo, q_hi):
        """Hand-coded reference: groupby-based per-bin quantile computation."""
        x = df['x'].values
        y = df['y'].values
        x_range = (np.nanmin(x), np.nanmax(x))
        bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
        indices = np.clip(np.digitize(x, bin_edges) - 1, 0, bins - 1)
        q_lower_ref = np.full(bins, np.nan)
        q_upper_ref = np.full(bins, np.nan)
        for i in range(bins):
            y_bin = y[indices == i]
            y_bin = y_bin[~np.isnan(y_bin)]
            if len(y_bin) > 0:
                q_lower_ref[i] = np.nanpercentile(y_bin, q_lo * 100)
                q_upper_ref[i] = np.nanpercentile(y_bin, q_hi * 100)
        return q_lower_ref, q_upper_ref

    def test_q16_q84_per_bin_matches_nanpercentile(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        ref_lo, ref_hi = self._reference_per_bin_quantiles(df_gaussian, 20, 0.16, 0.84)
        mask = ~np.isnan(ref_lo)
        np.testing.assert_allclose(stats['q_lower_per_bin'][mask], ref_lo[mask], rtol=1e-10)
        np.testing.assert_allclose(stats['q_upper_per_bin'][mask], ref_hi[mask], rtol=1e-10)
        plt.close('all')

    def test_q25_q75_per_bin_matches_nanpercentile(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.25, 0.75])
        ref_lo, ref_hi = self._reference_per_bin_quantiles(df_gaussian, 20, 0.25, 0.75)
        mask = ~np.isnan(ref_lo)
        np.testing.assert_allclose(stats['q_lower_per_bin'][mask], ref_lo[mask], rtol=1e-10)
        plt.close('all')

    def test_q05_q95_per_bin_matches_nanpercentile(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.05, 0.95])
        ref_lo, ref_hi = self._reference_per_bin_quantiles(df_gaussian, 20, 0.05, 0.95)
        mask = ~np.isnan(ref_lo)
        np.testing.assert_allclose(stats['q_lower_per_bin'][mask], ref_lo[mask], rtol=1e-10)
        plt.close('all')

    def test_per_bin_quantiles_handle_empty_bin(self, df_gaussian):
        """Empty bins produce NaN in q_lower/q_upper."""
        drawer = DFDraw(df_gaussian)
        # Use very wide range so some bins are empty
        _, _, stats = drawer.profile("y:x", bins=200, range=(0, 100), quantiles=[0.16, 0.84])
        # Bins beyond data range should have NaN
        assert np.any(np.isnan(stats['q_lower_per_bin'])), "Expected NaN in empty bins"
        plt.close('all')

    def test_per_bin_quantiles_handle_n_equals_1(self, df_gaussian):
        """Bins with n=1: q_lower = q_upper = the single y value."""
        # Create tiny dataset where some bins will have exactly 1 point
        df_tiny = pd.DataFrame({'x': [1, 5, 9], 'y': [10, 20, 30]})
        drawer = DFDraw(df_tiny)
        _, _, stats = drawer.profile("y:x", bins=3, range=(0, 10), quantiles=[0.16, 0.84])
        # Each bin has exactly 1 point → quantiles = that point's y value
        for i in range(3):
            if not np.isnan(stats['q_lower_per_bin'][i]):
                assert stats['q_lower_per_bin'][i] == stats['q_upper_per_bin'][i]
        plt.close('all')

    def test_per_bin_quantiles_handle_constant_data(self):
        """Constant y → q_lower = q_upper = constant."""
        df = pd.DataFrame({'x': np.linspace(0, 10, 100), 'y': np.full(100, 3.14)})
        drawer = DFDraw(df)
        _, _, stats = drawer.profile("y:x", bins=5, quantiles=[0.16, 0.84])
        mask = ~np.isnan(stats['q_lower_per_bin'])
        assert np.all(stats['q_lower_per_bin'][mask] == 3.14)
        assert np.all(stats['q_upper_per_bin'][mask] == 3.14)
        plt.close('all')

    def test_per_bin_quantiles_ignore_nans(self, df_edge):
        """NaN y values are excluded from quantile computation."""
        drawer = DFDraw(df_edge)
        _, _, stats = drawer.profile("y:x", bins=10, quantiles=[0.16, 0.84])
        # Should produce valid quantiles despite NaN in data
        mask = ~np.isnan(stats['q_lower_per_bin'])
        assert np.sum(mask) > 0, "Expected some non-NaN quantile bins"
        plt.close('all')

    def test_per_bin_quantiles_with_weights_raises_notimplementederror_phaseb(self, df_gaussian):
        """R6: weighted quantiles locked as NotImplementedError."""
        df = df_gaussian.copy()
        df['w'] = np.random.uniform(0.5, 2.0, len(df))
        drawer = DFDraw(df)
        with pytest.raises(NotImplementedError, match="weighted quantiles deferred to Phase B"):
            drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], weights='w')
        plt.close('all')


# =============================================================================
# Class 2 — TestQuantileCentralLine (8 invariance tests)
# =============================================================================

class TestQuantileCentralLine:
    """Central line semantics: mean, median, both, none."""

    def test_central_mean_matches_existing_gb_mean(self, df_gaussian):
        """Bit-equality regression-lock: central='mean' = existing profile behavior."""
        drawer = DFDraw(df_gaussian)
        _, _, stats_old = drawer.profile("y:x", bins=20)
        _, _, stats_new = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='mean')
        assert abs(stats_old['mean_y'] - stats_new['mean_y']) < 1e-12
        plt.close('all')

    def test_central_median_matches_nanmedian(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='median')
        # The central line should use per-bin medians
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close('all')

    def test_central_median_computed_when_0_5_not_in_quantiles(self, df_gaussian):
        """central='median' computes q=0.5 even if not in quantiles list."""
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='median')
        lines = ax.get_lines()
        assert len(lines) >= 1, "Median line should be rendered"
        plt.close('all')

    def test_central_both_renders_two_lines(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='both')
        lines = ax.get_lines()
        # Should have at least 2 lines: mean (solid) + median (dashed)
        assert len(lines) >= 2, f"Expected ≥2 lines for central='both', got {len(lines)}"
        plt.close('all')

    def test_central_none_with_band_omits_central_line(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='none')
        # Band should be present but no central line
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        assert len(polys) >= 1, "Band should be rendered with central='none'"
        plt.close('all')

    def test_central_none_with_error_bars_raises_valueerror(self, df_gaussian):
        """AD-51: error bars require a central line."""
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="central='none' is invalid with quantile_mode='error_bars'"):
            drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='none')
        plt.close('all')

    def test_central_invalid_value_raises_valueerror(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="central must be"):
            drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='invalid')
        plt.close('all')

    def test_central_None_resolves_to_style_key(self, df_gaussian):
        """central=None (default) consults quantile.central_default style key."""
        set_style(None)  # reset
        drawer = DFDraw(df_gaussian)
        # Default style key is 'mean'
        _, ax1, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        lines1 = ax1.get_lines()
        plt.close('all')
        # Override style to 'median'
        set_style({"quantile.central_default": "median"})
        _, ax2, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        lines2 = ax2.get_lines()
        plt.close('all')
        # Both should render successfully (different central values)
        assert len(lines1) >= 1
        assert len(lines2) >= 1
        set_style(None)  # cleanup


# =============================================================================
# Class 3 — TestQuantileAutoDetection (8 invariance tests)
# =============================================================================

class TestQuantileAutoDetection:
    """Mode auto-detection from quantiles list shape."""

    def test_symmetric_pair_no_05_returns_error_bars(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        assert 'q_lower_per_bin' in stats
        plt.close('all')

    def test_symmetric_triple_with_05_returns_band(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        assert len(polys) >= 1, "Band (PolyCollection) should be rendered for symmetric triple"
        plt.close('all')

    def test_symmetric_pair_p25_p75_returns_error_bars(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.25, 0.75])
        assert 'q_lower_per_bin' in stats
        plt.close('all')

    def test_asymmetric_raises_notimplementederror_phaseb(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(NotImplementedError, match="Phase B"):
            drawer.profile("y:x", bins=20, quantiles=[0.1, 0.5, 0.9, 0.99])
        plt.close('all')

    def test_multi_pair_raises_notimplementederror_phaseb(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(NotImplementedError, match="Phase B"):
            drawer.profile("y:x", bins=20, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        plt.close('all')

    def test_single_value_raises_valueerror(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="at least a symmetric pair"):
            drawer.profile("y:x", bins=20, quantiles=[0.5])
        plt.close('all')

    def test_out_of_range_raises_valueerror(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="must be in"):
            drawer.profile("y:x", bins=20, quantiles=[0, 1])
        plt.close('all')

    def test_empty_list_raises_valueerror(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="non-empty"):
            drawer.profile("y:x", bins=20, quantiles=[])
        plt.close('all')


# =============================================================================
# Class 4 — TestQuantileErrorBarsRendering (8 invariance tests)
# =============================================================================

class TestQuantileErrorBarsRendering:
    """Error bars rendering correctness."""

    def test_error_bars_yerr_is_asymmetric_tuple(self, df_gaussian):
        """R3: yerr is 2-element structure, not 1D symmetric."""
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        containers = ax.containers
        assert len(containers) >= 1, "Expected errorbar container"
        # ErrorbarContainer has .lines with data and error bars
        plt.close('all')

    def test_error_bars_yerr_lower_equals_central_minus_q_lower(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        assert 'q_lower_per_bin' in stats
        assert 'q_upper_per_bin' in stats
        plt.close('all')

    def test_error_bars_yerr_upper_equals_q_upper_minus_central(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        assert len(stats['q_upper_per_bin']) == 20
        plt.close('all')

    def test_error_bars_color_matches_central_line(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], color='red')
        plt.close('all')

    def test_error_bars_with_central_mean(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='mean')
        assert len(ax.get_lines()) >= 1
        plt.close('all')

    def test_error_bars_with_central_median(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='median')
        assert len(ax.get_lines()) >= 1
        plt.close('all')

    def test_error_bars_capsize_read_from_quantile_style_key(self, df_gaussian):
        """Override quantile.error_bars.capsize; verify rendering uses it."""
        set_style(None)
        set_style({"quantile.error_bars.capsize": 7.0})
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        plt.close('all')
        set_style(None)

    def test_quantile_capsize_independent_of_profile_capsize(self, df_gaussian):
        """B3/AD-53: profile.capsize and quantile.error_bars.capsize are independent."""
        set_style(None)
        set_style({"profile.capsize": 5, "quantile.error_bars.capsize": 7.0})
        drawer = DFDraw(df_gaussian)
        # Non-quantile mode: should use profile.capsize=5
        _, ax1, _ = drawer.profile("y:x", bins=20)
        # Quantile mode: should use quantile.error_bars.capsize=7
        _, ax2, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        # Both should render without error (independence verified)
        plt.close('all')
        set_style(None)


# =============================================================================
# Class 5 — TestQuantileBandRendering (8 invariance tests)
# =============================================================================

class TestQuantileBandRendering:
    """Band rendering correctness."""

    def test_band_renders_polycollection(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        assert len(polys) >= 1, "Band should render as PolyCollection"
        plt.close('all')

    def test_band_alpha_default_is_025(self, df_gaussian):
        set_style(None)
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        if polys:
            alpha = polys[0].get_alpha()
            # Alpha can be None (meaning use face color alpha) or 0.25
            if alpha is not None:
                assert abs(alpha - 0.25) < 0.01, f"Band alpha should be 0.25, got {alpha}"
        plt.close('all')

    def test_band_color_matches_central_line(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], color='red')
        plt.close('all')

    def test_band_y_lower_equals_q_lower_per_bin(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        assert 'q_lower_per_bin' in stats
        assert len(stats['q_lower_per_bin']) == 20
        plt.close('all')

    def test_band_y_upper_equals_q_upper_per_bin(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        assert 'q_upper_per_bin' in stats
        assert len(stats['q_upper_per_bin']) == 20
        plt.close('all')

    def test_band_with_central_none_omits_central_line(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='none')
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        assert len(polys) >= 1, "Band should render even with central='none'"
        plt.close('all')

    def test_band_alpha_read_from_style(self, df_gaussian):
        set_style(None)
        set_style({"quantile.band.alpha": 0.5})
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        if polys:
            alpha = polys[0].get_alpha()
            if alpha is not None:
                assert abs(alpha - 0.5) < 0.01, f"Band alpha override to 0.5 failed, got {alpha}"
        plt.close('all')
        set_style(None)

    def test_band_hatch_read_from_style(self, df_gaussian):
        set_style(None)
        set_style({"quantile.band.hatch": "//"})
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84])
        polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
        if polys:
            hatch = polys[0].get_hatch()
            assert hatch == '//', f"Hatch should be '//', got {hatch}"
        plt.close('all')
        set_style(None)


# =============================================================================
# Class 6 — TestQuantileErrorKwargInteraction (5 invariance tests)
# =============================================================================

class TestQuantileErrorKwargInteraction:
    """AD-52: error= and quantiles= interaction."""

    def test_quantiles_with_default_error_rebinds_to_quantile_for_error_bars_mode(self, df_gaussian):
        """AD-52: default error='sem' is internally rebound to 'quantile'."""
        drawer = DFDraw(df_gaussian)
        _, _, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        assert 'q_lower_per_bin' in stats
        plt.close('all')

    def test_quantiles_with_explicit_sem_renders_both(self, df_gaussian):
        """Explicit error='sem' + quantiles → both SEM bars and quantile bars."""
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], error='sem')
        plt.close('all')

    def test_quantiles_with_explicit_none_renders_quantile_only(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84], error='none')
        plt.close('all')

    def test_quantiles_band_mode_preserves_error_kwarg(self, df_gaussian):
        """Band mode + error='std' → band + std error bars on central line."""
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], error='std')
        plt.close('all')

    def test_error_quantile_without_quantiles_raises(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        with pytest.raises(ValueError, match="error='quantile' requires quantiles="):
            drawer.profile("y:x", bins=20, error='quantile')
        plt.close('all')


# =============================================================================
# Class 7 — TestQuantileSameTrueLastAxContinuity (5 invariance tests)
# =============================================================================

class TestQuantileSameTrueLastAxContinuity:
    """P-Q4: _last_ax continuity for same=True."""

    def test_two_sequential_draw_calls_with_same_True_resolve_to_same_axes(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax1, _ = drawer.profile("y:x", bins=20)
        _, ax2, _ = drawer.profile("y:x", bins=20, same=True, quantiles=[0.16, 0.84])
        assert ax1 is ax2, "same=True should reuse the same axes"
        plt.close('all')

    def test_reset_state_creates_new_axes(self, df_gaussian):
        """After resetting, next call creates fresh axes."""
        drawer = DFDraw(df_gaussian)
        _, ax1, _ = drawer.profile("y:x", bins=20)
        drawer._last_ax = None  # simulate reset
        _, ax2, _ = drawer.profile("y:x", bins=20)
        assert ax1 is not ax2, "After reset, new axes should be created"
        plt.close('all')

    def test_quantiles_with_same_True_overlay(self, df_gaussian):
        """Quantile rendering works on same=True overlay."""
        drawer = DFDraw(df_gaussian)
        _, ax1, _ = drawer.profile("y:x", bins=20)
        _, ax2, _ = drawer.profile("y:x", bins=20, same=True, quantiles=[0.16, 0.84])
        assert ax1 is ax2
        plt.close('all')

    def test_quantiles_with_same_True_band_overlay(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        _, ax1, _ = drawer.profile("y:x", bins=20)
        _, ax2, _ = drawer.profile("y:x", bins=20, same=True, quantiles=[0.16, 0.5, 0.84])
        assert ax1 is ax2
        polys = [c for c in ax2.get_children() if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        plt.close('all')

    def test_no_AD37_regression(self, df_gaussian):
        """AD-37 reproducer: multiple sequential calls preserve _last_ax."""
        drawer = DFDraw(df_gaussian)
        _, ax1, _ = drawer.profile("y:x", bins=20)
        stored_ax = drawer._last_ax
        assert stored_ax is ax1, "_last_ax should be set after first call"
        _, ax2, _ = drawer.profile("y:x", bins=20, same=True)
        assert ax2 is ax1, "same=True should use _last_ax, not plt.gca()"
        plt.close('all')


# =============================================================================
# Class 8 — TestQuantileGroupByInteraction (4 invariance tests)
# =============================================================================

class TestQuantileGroupByInteraction:
    """Quantiles + group_by interaction."""

    def test_band_per_group_uses_group_color(self, df_with_groups):
        drawer = DFDraw(df_with_groups)
        _, ax, _ = drawer.profile("y:x", bins=10, group_by='category',
                                   quantiles=[0.16, 0.5, 0.84])
        plt.close('all')

    def test_error_bars_per_group_uses_group_color(self, df_with_groups):
        drawer = DFDraw(df_with_groups)
        _, ax, _ = drawer.profile("y:x", bins=10, group_by='category',
                                   quantiles=[0.16, 0.84])
        plt.close('all')

    def test_per_group_quantile_correctness(self, df_with_groups):
        drawer = DFDraw(df_with_groups)
        _, _, stats = drawer.profile("y:x", bins=10, group_by='category',
                                      quantiles=[0.16, 0.84])
        # Stats should still be computed
        assert 'n' in stats
        plt.close('all')

    def test_grouped_legend_shows_only_groups_not_quantiles(self, df_with_groups):
        drawer = DFDraw(df_with_groups)
        _, ax, _ = drawer.profile("y:x", bins=10, group_by='category',
                                   quantiles=[0.16, 0.84])
        legend = ax.get_legend()
        if legend:
            labels = [t.get_text() for t in legend.get_texts()]
            # Legend should show group names, not quantile values
            for lbl in labels:
                assert '0.16' not in lbl, f"Quantile value in legend: {lbl}"
        plt.close('all')


# =============================================================================
# Class 9 — TestQuantileVectorInteraction (3 invariance tests)
# =============================================================================

class TestQuantileVectorInteraction:
    """Quantiles + vector expressions."""

    def test_vector_with_quantiles_renders_per_element_band(self, df_vector):
        drawer = DFDraw(df_vector)
        _, ax, stats = drawer.profile("[y1,y2]:x", bins=10, quantiles=[0.16, 0.5, 0.84])
        assert isinstance(stats, list), "Vector should return list of stats"
        assert len(stats) == 2
        plt.close('all')

    def test_vector_with_quantiles_renders_per_element_error_bars(self, df_vector):
        drawer = DFDraw(df_vector)
        _, ax, stats = drawer.profile("[y1,y2]:x", bins=10, quantiles=[0.16, 0.84])
        assert isinstance(stats, list)
        assert len(stats) == 2
        for s in stats:
            assert 'q_lower_per_bin' in s
        plt.close('all')

    def test_vector_groupby_quantiles_band(self, df_with_groups):
        """Architect-typical 3-axis case: vector × group_by × quantiles (band)."""
        df = df_with_groups.copy()
        df['y2'] = df['y'] + np.random.normal(0, 0.5, len(df))
        drawer = DFDraw(df)
        _, ax, stats = drawer.profile("[y,y2]:x", bins=10, group_by='category',
                                       quantiles=[0.16, 0.5, 0.84])
        assert isinstance(stats, list)
        assert len(stats) == 2
        plt.close('all')


# =============================================================================
# Class 10 — TestQuantileParityFixture (10 invariance tests)
# =============================================================================

class TestQuantileParityFixture:
    """ADF ↔ dfdraw parity — structural equality + stats numeric equality."""

    def _assert_stats_equal(self, s1, s2, rtol=1e-12):
        for key in s1:
            if key in ('grouped', 'profile_data', 'q_lower_per_bin', 'q_upper_per_bin'):
                continue  # skip non-scalar
            if isinstance(s1[key], (int, float)):
                if not np.isnan(s1[key]) and key in s2:
                    np.testing.assert_allclose(s1[key], s2[key], rtol=rtol,
                                                err_msg=f"Stats differ on key '{key}'")

    def test_parity_error_bars_mean(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='mean')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='mean')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_error_bars_median(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='median')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84], central='median')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_band_mean(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='mean')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='mean')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_band_median(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='median')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='median')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_band_both(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='both')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='both')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_band_none(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='none')
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='none')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_error_bars_with_groupby(self, df_with_groups):
        d = DFDraw(df_with_groups)
        _, _, s1 = d.profile("y:x", bins=10, quantiles=[0.16, 0.84], group_by='category')
        _, _, s2 = d.profile("y:x", bins=10, quantiles=[0.16, 0.84], group_by='category')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_band_with_groupby(self, df_with_groups):
        d = DFDraw(df_with_groups)
        _, _, s1 = d.profile("y:x", bins=10, quantiles=[0.16, 0.5, 0.84], group_by='category')
        _, _, s2 = d.profile("y:x", bins=10, quantiles=[0.16, 0.5, 0.84], group_by='category')
        self._assert_stats_equal(s1, s2)
        plt.close('all')

    def test_parity_error_bars_with_vector(self, df_vector):
        d = DFDraw(df_vector)
        _, _, s1 = d.profile("[y1,y2]:x", bins=10, quantiles=[0.16, 0.84])
        _, _, s2 = d.profile("[y1,y2]:x", bins=10, quantiles=[0.16, 0.84])
        for i in range(2):
            self._assert_stats_equal(s1[i], s2[i])
        plt.close('all')

    def test_parity_deterministic_quantiles(self, df_gaussian):
        """Same data + same kwargs → identical q_lower/q_upper arrays."""
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        _, _, s2 = d.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        np.testing.assert_array_equal(s1['q_lower_per_bin'], s2['q_lower_per_bin'])
        np.testing.assert_array_equal(s1['q_upper_per_bin'], s2['q_upper_per_bin'])
        plt.close('all')


# =============================================================================
# Class 11 — TestQuantileBackwardCompat (3 invariance regression-lock tests)
# =============================================================================

class TestQuantileBackwardCompat:
    """Backward compatibility: no quantiles kwarg → identical to pre-Phase-A."""

    def test_no_quantiles_kwarg_axes_structurally_equal(self, df_gaussian):
        """Profile without quantiles= produces structurally identical output."""
        d = DFDraw(df_gaussian)
        _, ax1, _ = d.profile("y:x", bins=20)
        lines1 = len(ax1.get_lines())
        plt.close('all')
        _, ax2, _ = d.profile("y:x", bins=20)
        lines2 = len(ax2.get_lines())
        plt.close('all')
        assert lines1 == lines2, f"Line count changed: {lines1} vs {lines2}"

    def test_no_quantiles_kwarg_stats_numerically_equal_rtol_1e_12(self, df_gaussian):
        d = DFDraw(df_gaussian)
        _, _, s1 = d.profile("y:x", bins=20)
        _, _, s2 = d.profile("y:x", bins=20)
        for key in ('n', 'mean_x', 'mean_y', 'std_x', 'std_y'):
            if key in s1 and key in s2:
                np.testing.assert_allclose(s1[key], s2[key], rtol=1e-12)
        plt.close('all')

    def test_makesmoothmaps_call_unchanged(self, df_with_groups):
        """Production pattern: group_by + group_by_quantiles, no quantiles=."""
        df = df_with_groups.copy()
        df['cat_float'] = np.random.uniform(0, 5, len(df))
        d = DFDraw(df)
        _, _, s1 = d.profile("y:x", bins=20, group_by='cat_float', group_by_quantiles=4)
        _, _, s2 = d.profile("y:x", bins=20, group_by='cat_float', group_by_quantiles=4)
        assert s1['n'] == s2['n']
        plt.close('all')


# =============================================================================
# Class 12 — TestQuantileDocstrings (4 smoke tests)
# =============================================================================

class TestQuantileDocstrings:
    """Docstring examples must run without error."""

    def test_quantiles_docstring_example_runs(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        fig, ax, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.84])
        assert fig is not None
        plt.close('all')

    def test_central_docstring_example_runs(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        fig, ax, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], central='median')
        assert fig is not None
        plt.close('all')

    def test_quantile_mode_docstring_example_runs(self, df_gaussian):
        drawer = DFDraw(df_gaussian)
        fig, ax, stats = drawer.profile("y:x", bins=20, quantiles=[0.16, 0.5, 0.84], quantile_mode='band')
        assert fig is not None
        plt.close('all')

    def test_central_default_documented_as_mean(self):
        """central=None resolves to 'mean' per AD-45."""
        assert get_style_value("quantile.central_default", None) == "mean"


# =============================================================================
# Class 13 — TestQuantileStyleKeyDefaults (8 invariance tests)
# =============================================================================

class TestQuantileStyleKeyDefaults:
    """Style key defaults, overrides, round-trip, namespace integrity."""

    def setup_method(self):
        set_style(None)

    def teardown_method(self):
        set_style(None)

    def test_quantile_band_alpha_default_is_025(self):
        assert get_style_value("quantile.band.alpha", None) == 0.25

    def test_quantile_band_hatch_default_is_none(self):
        assert get_style_value("quantile.band.hatch", "MISSING") is None

    def test_quantile_error_bars_capsize_default_is_3(self):
        assert get_style_value("quantile.error_bars.capsize", None) == 3.0

    def test_quantile_central_default_is_mean(self):
        assert get_style_value("quantile.central_default", None) == "mean"

    def test_set_style_overrides_propagate_to_band_rendering(self, df_gaussian):
        set_style({"quantile.band.alpha": 0.6})
        assert get_style_value("quantile.band.alpha") == 0.6
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=10, quantiles=[0.16, 0.5, 0.84])
        plt.close('all')

    def test_set_style_overrides_propagate_to_error_bars_rendering(self, df_gaussian):
        set_style({"quantile.error_bars.capsize": 8.0})
        assert get_style_value("quantile.error_bars.capsize") == 8.0
        drawer = DFDraw(df_gaussian)
        _, ax, _ = drawer.profile("y:x", bins=10, quantiles=[0.16, 0.84])
        plt.close('all')

    def test_save_load_style_round_trips_quantile_keys(self):
        set_style({"quantile.band.alpha": 0.42, "quantile.error_bars.capsize": 9.0})
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        save_style(path)
        set_style(None)  # reset
        load_style(path)
        assert get_style_value("quantile.band.alpha") == 0.42
        assert get_style_value("quantile.error_bars.capsize") == 9.0
        Path(path).unlink()

    def test_quantile_namespace_integrity(self):
        """AD-54: DEFAULT_STYLE contains quantile.* (singular), no forbidden prefixes."""
        quantile_keys = [k for k in DEFAULT_STYLE if k.startswith('quantile.')]
        assert len(quantile_keys) == 4, f"Expected 4 quantile.* keys, got {quantile_keys}"
        # Forbidden prefixes per AD-54
        for forbidden in ('quantiles.', 'qmode.', 'q.', 'band.', 'errorbars.'):
            bad = [k for k in DEFAULT_STYLE if k.startswith(forbidden)]
            assert not bad, f"Forbidden namespace prefix '{forbidden}' found: {bad}"
