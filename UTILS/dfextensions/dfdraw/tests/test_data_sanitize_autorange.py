"""
Tests for Phase 13.28.DF — Robust Data Handling.

Part A (NaN/inf filter): 12 invariance tests on sanitize_for_plot().
Part B (Hybrid autorange): 9 invariance tests on compute_autorange() and
                            hybrid_autorange().

References
----------
- Proposal: PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §8
- AD-69, AD-70, AD-71 (Part A); AD-72, AD-73, AD-77 (Part B)
"""
import warnings
import pytest
import numpy as np

from dfdraw.plots._data_sanitize import sanitize_for_plot
from dfdraw.plots._autorange import (
    compute_autorange,
    hybrid_autorange,
    VALID_STRATEGIES,
)


# =============================================================================
# Class 1 — TestNaNInfFilter (5 tests, Part A core)
# =============================================================================

class TestNaNInfFilter:
    """Default-policy NaN/inf filtering with counter reporting."""

    def test_inf_filtered_silently_with_default_policy(self):
        """data=[1,2,np.inf,4]; default nan_policy='filter';
        assert n_input==4 AND len(clean)==3 AND n_inf_x==1 AND no warning."""
        x = np.array([1.0, 2.0, np.inf, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x_clean, y_clean, stats = sanitize_for_plot(x)
        assert stats["n_input"] == 4
        assert len(x_clean) == 3
        assert stats["n_inf_x"] == 1
        assert stats["n_nan_x"] == 0
        assert stats["n_filtered"] == 1
        assert y_clean is None

    def test_nan_filtered_silently_with_default_policy(self):
        """data=[1,2,np.nan,4]; default nan_policy='filter';
        assert n_input==4 AND len(clean)==3 AND n_nan_x==1."""
        x = np.array([1.0, 2.0, np.nan, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x_clean, _, stats = sanitize_for_plot(x)
        assert stats["n_input"] == 4
        assert len(x_clean) == 3
        assert stats["n_nan_x"] == 1
        assert stats["n_inf_x"] == 0
        assert stats["n_filtered"] == 1

    def test_inf_y_only_counted_correctly(self):
        """2D with inf only in y; assert n_inf_x==0 AND n_inf_y>0 (asymmetric)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0.5, np.inf, 1.5, 2.0])
        x_clean, y_clean, stats = sanitize_for_plot(x, y)
        assert stats["n_inf_x"] == 0
        assert stats["n_nan_x"] == 0
        assert stats["n_inf_y"] == 1
        assert stats["n_nan_y"] == 0
        assert len(x_clean) == 3
        assert len(y_clean) == 3
        assert stats["n_filtered"] == 1

    def test_no_finite_data_warns_with_filter_policy(self):
        """All NaN; assert UserWarning emitted AND empty arrays returned (not raised)."""
        x = np.array([np.nan, np.nan, np.nan])
        with pytest.warns(UserWarning, match="all 3 rows dropped"):
            x_clean, _, stats = sanitize_for_plot(x, nan_policy="filter")
        assert stats["n_input"] == 3
        assert len(x_clean) == 0
        assert stats["n_filtered"] == 3
        assert stats["n_nan_x"] == 3

    def test_clean_data_no_counters_change(self):
        """Clean data; assert all counter keys present and zero AND no warnings
        AND n_input == len(x_clean) (regression lock — no behavior change)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x_clean, y_clean, stats = sanitize_for_plot(x, y)
        for k in ("n_inf_x", "n_nan_x", "n_inf_y", "n_nan_y", "n_filtered"):
            assert stats[k] == 0, f"{k} should be 0 for clean data"
        assert stats["n_input"] == 5
        assert len(x_clean) == 5
        assert len(y_clean) == 5
        np.testing.assert_array_equal(x_clean, x)
        np.testing.assert_array_equal(y_clean, y)


# =============================================================================
# Class 2 — TestNanPolicy (5 tests, Part A optional behavior)
# =============================================================================

class TestNanPolicy:
    """nan_policy parameter behavior across 'filter', 'warn', 'raise' modes."""

    def test_nan_policy_raise(self):
        """data has inf; nan_policy='raise';
        assert ValueError; message names column 'x' AND inf count."""
        x = np.array([1.0, 2.0, np.inf])
        with pytest.raises(ValueError) as exc_info:
            sanitize_for_plot(x, nan_policy="raise", column_names=("x", "y"))
        msg = str(exc_info.value)
        assert "'x'" in msg
        assert "1 inf" in msg

    def test_nan_policy_warn_emits_warning_then_filters(self):
        """data has inf; nan_policy='warn';
        assert UserWarning AND data filtered (not raised)."""
        x = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.warns(UserWarning, match="filtered 1"):
            x_clean, _, stats = sanitize_for_plot(x, nan_policy="warn")
        assert len(x_clean) == 3
        assert stats["n_filtered"] == 1

    def test_nan_policy_filter_preserves_pre_phase_behavior(self):
        """default 'filter' on clean data is a no-op; counters all zero
        AND clean data preserved bit-identical (regression lock)."""
        x = np.array([1.5, 2.5, 3.5])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x_clean, _, stats = sanitize_for_plot(x, nan_policy="filter")
        np.testing.assert_array_equal(x_clean, x)
        assert stats["n_filtered"] == 0
        assert all(stats[k] == 0 for k in
                   ("n_inf_x", "n_nan_x", "n_inf_y", "n_nan_y"))

    def test_nan_policy_invalid_value_raises(self):
        """nan_policy='banana' raises ValueError naming valid options."""
        with pytest.raises(ValueError) as exc_info:
            sanitize_for_plot(np.array([1.0]), nan_policy="banana")
        msg = str(exc_info.value)
        assert "nan_policy" in msg
        assert "filter" in msg and "raise" in msg and "warn" in msg

    def test_nan_policy_raise_only_fires_when_invalid_present(self):
        """Clean data with nan_policy='raise' does NOT raise (only fires on invalid)."""
        x = np.array([1.0, 2.0, 3.0])
        x_clean, _, stats = sanitize_for_plot(x, nan_policy="raise")
        assert len(x_clean) == 3
        assert stats["n_filtered"] == 0


# =============================================================================
# Class 3 — TestHybridAutorange (5 tests, Part B core)
# Verifies §4.1 spec across 4 worked examples + constant-data degenerate.
# =============================================================================

class TestHybridAutorange:
    """Hybrid autorange algorithm — §4.1 spec verification."""

    def test_clean_gaussian_uses_minmax(self):
        """Clean N(0,1), N=1000, no outliers;
        assert range_used == (data.min(), data.max()) within 1e-9."""
        rng = np.random.default_rng(seed=42)
        data = rng.normal(0.0, 1.0, size=1000)
        # Sanity: clean Gaussian — extremes within ~3.5 sigma, no outliers
        lo, hi = hybrid_autorange(data, k_robust=4.0, k_outlier=1.5)
        assert abs(lo - data.min()) < 1e-9, \
            f"Clean Gaussian: expected lo==data.min()={data.min():.4f}, got {lo:.4f}"
        assert abs(hi - data.max()) < 1e-9, \
            f"Clean Gaussian: expected hi==data.max()={data.max():.4f}, got {hi:.4f}"

    def test_outlier_high_clips_to_robust(self):
        """N(0,1), N=999 + 1 point at +100;
        assert range_used[1] is robust bound (~ 4·sigma_MAD ≈ 4) AND
        range_used[0] == data.min() (no outlier on low side)."""
        rng = np.random.default_rng(seed=42)
        data = np.concatenate([rng.normal(0.0, 1.0, size=999), [100.0]])
        lo, hi = hybrid_autorange(data, k_robust=4.0, k_outlier=1.5)
        # High side clipped to robust window
        assert hi < 10.0, f"Expected hi clipped (<10), got {hi:.4f}"
        assert hi > 3.0, f"Expected hi >3 (~4 sigma), got {hi:.4f}"
        # Low side preserved (no low outlier)
        finite_data = data
        assert abs(lo - finite_data.min()) < 1e-9, \
            f"Low side should be data.min()={finite_data.min():.4f}, got {lo:.4f}"

    def test_outlier_low_clips_to_robust(self):
        """N(0,1), N=999 + 1 point at -100;
        assert range_used[0] is robust bound AND range_used[1] == data.max()."""
        rng = np.random.default_rng(seed=42)
        data = np.concatenate([rng.normal(0.0, 1.0, size=999), [-100.0]])
        lo, hi = hybrid_autorange(data, k_robust=4.0, k_outlier=1.5)
        assert lo > -10.0, f"Expected lo clipped (>-10), got {lo:.4f}"
        assert lo < -3.0, f"Expected lo <-3 (~-4 sigma), got {lo:.4f}"
        assert abs(hi - data.max()) < 1e-9, \
            f"High side should be data.max()={data.max():.4f}, got {hi:.4f}"

    def test_asymmetric_distribution_one_sided_clip(self):
        """Long right tail, otherwise tight body.
        Construct: 1000 points around median 50 with MAD ~10, max=200.
        assert lo == data.min() AND hi < data.max() (right side clipped)."""
        rng = np.random.default_rng(seed=42)
        # Bulk at median 50, MAD ~10
        bulk = rng.normal(50.0, 14.826, size=999)  # sigma=14.826 → MAD~10
        # Single right-tail outlier at 200
        data = np.concatenate([bulk, [200.0]])
        lo, hi = hybrid_autorange(data, k_robust=4.0, k_outlier=1.5)
        assert abs(lo - data.min()) < 1e-9, \
            f"Low side should be preserved at data.min()={data.min():.4f}"
        assert hi < data.max(), \
            f"High side should be clipped (data.max()={data.max():.4f}), got {hi:.4f}"

    def test_constant_data_returns_unit_range(self):
        """data = [7]*1000; sigma_MAD == 0 → unit window centered on median.
        assert range_used == (6.5, 7.5)."""
        data = np.full(1000, 7.0)
        lo, hi = hybrid_autorange(data)
        assert abs(lo - 6.5) < 1e-9, f"Expected lo=6.5, got {lo}"
        assert abs(hi - 7.5) < 1e-9, f"Expected hi=7.5, got {hi}"


# =============================================================================
# Class 4 — TestAutorangeStrategies (3 tests, Part B presets)
# =============================================================================

class TestAutorangeStrategies:
    """Strategy preset behavior — minmax, percentile, invalid strategy."""

    def test_minmax_strategy_equals_data_min_max(self):
        """strategy='minmax' on any data; range == (data.min(), data.max())."""
        rng = np.random.default_rng(seed=42)
        # Test on multiple shapes/distributions
        for data in (
            rng.normal(0.0, 1.0, size=100),
            rng.uniform(-5.0, 5.0, size=500),
            np.arange(50, dtype=float),
        ):
            lo, hi = compute_autorange(data, strategy="minmax")
            assert abs(lo - data.min()) < 1e-9
            assert abs(hi - data.max()) < 1e-9

    def test_percentile_99_clips_to_quantiles(self):
        """strategy='percentile_99'; range == (np.percentile(data, 1), np.percentile(data, 99))."""
        rng = np.random.default_rng(seed=42)
        data = rng.normal(0.0, 1.0, size=10000)
        lo, hi = compute_autorange(data, strategy="percentile_99")
        expected_lo = float(np.percentile(data, 1))
        expected_hi = float(np.percentile(data, 99))
        assert abs(lo - expected_lo) < 1e-9, \
            f"Expected lo={expected_lo:.6f}, got {lo:.6f}"
        assert abs(hi - expected_hi) < 1e-9, \
            f"Expected hi={expected_hi:.6f}, got {hi:.6f}"

    def test_invalid_strategy_raises(self):
        """strategy='banana' raises ValueError with valid options listed."""
        with pytest.raises(ValueError) as exc_info:
            compute_autorange(np.array([1.0, 2.0, 3.0]), strategy="banana")
        msg = str(exc_info.value)
        assert "strategy" in msg
        # Must list at least the canonical options
        for s in ("minmax", "hybrid", "percentile_99"):
            assert s in msg, f"Error message should list {s!r}"


# =============================================================================
# Class 5 — TestStatsDictAdditive (3 tests — sanitize + autorange diagnostic)
# =============================================================================

class TestStatsDictAdditive:
    """Stats dict additions are backward-compatible and well-formed."""

    def test_sanitize_stats_keys_present_and_well_typed(self):
        """sanitize_for_plot returns dict with all 6 expected keys, all int."""
        x = np.array([1.0, 2.0, np.inf, np.nan])
        y = np.array([10.0, np.nan, 30.0, 40.0])
        _, _, stats = sanitize_for_plot(x, y)
        expected = {"n_input", "n_filtered", "n_inf_x", "n_nan_x", "n_inf_y", "n_nan_y"}
        assert set(stats.keys()) == expected
        for k in expected:
            assert isinstance(stats[k], int), f"{k} should be int, got {type(stats[k])}"
            assert stats[k] >= 0, f"{k} should be non-negative"

    def test_sanitize_counters_arithmetic_consistent(self):
        """n_filtered == n_input - len(clean) AND counters sum correctly."""
        x = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
        x_clean, _, stats = sanitize_for_plot(x)
        assert stats["n_filtered"] == stats["n_input"] - len(x_clean)
        assert stats["n_inf_x"] + stats["n_nan_x"] == stats["n_filtered"]

    def test_compute_autorange_returns_finite_tuple(self):
        """compute_autorange always returns a finite (lo, hi) tuple of floats
        for any non-empty input AND for empty data returns (0.0, 1.0) sentinel."""
        rng = np.random.default_rng(seed=42)
        data = rng.normal(0.0, 1.0, size=500)
        for strategy in VALID_STRATEGIES:
            lo, hi = compute_autorange(data, strategy=strategy)
            assert isinstance(lo, float) and isinstance(hi, float), \
                f"{strategy}: expected floats, got {type(lo)}, {type(hi)}"
            assert np.isfinite(lo) and np.isfinite(hi), \
                f"{strategy}: expected finite, got ({lo}, {hi})"
            assert lo <= hi, f"{strategy}: lo > hi ({lo} > {hi})"

        # Empty input — sentinel (0.0, 1.0)
        empty = np.array([], dtype=float)
        for strategy in VALID_STRATEGIES:
            lo, hi = compute_autorange(empty, strategy=strategy)
            assert (lo, hi) == (0.0, 1.0), \
                f"{strategy} on empty: expected (0.0, 1.0), got ({lo}, {hi})"


# =============================================================================
# Class 6 — TestPlotIntegration (5 tests, integration of Part A + Part B
# into draw_hist / draw_hist2d / draw_hexbin / draw_scatter / draw_profile)
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # headless for CI
import pandas as pd
from dfdraw import DFDraw


class TestPlotIntegration:
    """End-to-end integration: real plot calls with inf/NaN data + autorange."""

    def test_hist2d_with_inf_does_not_crash_and_reports_counters(self):
        """Architect's exact bug case (y/x with x=0 produces inf):
        previously crashed matplotlib; must now succeed and report n_inf_y."""
        df = pd.DataFrame({
            "x":       [1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0],
            "y":       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "detType": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        d = DFDraw(df)

        fig, ax, stats = d.hist2d("y/x:x", selection="detType==0")

        # No crash — call returned successfully
        assert "n" in stats
        # Counters present and report the inf rows
        assert stats["n_inf_y"] == 1, \
            f"Expected n_inf_y==1 (the x=0 row produced inf in y/x), got {stats['n_inf_y']}"
        assert stats["n"] == 3, \
            f"After filter, expected 3 rows, got {stats['n']}"
        assert stats["n_input"] == 4, \
            f"Selected 4 rows before sanitize, got {stats['n_input']}"
        # Autorange diagnostics present
        assert "autorange_used" in stats
        assert "autorange_strategy" in stats

    def test_hist1d_clean_data_autorange_diagnostics(self):
        """Plain 1D hist on clean data: stats include autorange_used and strategy."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        d = DFDraw(df)

        fig, ax, stats = d.hist("y")

        assert "autorange_used" in stats
        lo, hi = stats["autorange_used"]
        # Hybrid on clean data == minmax
        assert lo == 1.0 and hi == 8.0, \
            f"Expected (1.0, 8.0), got ({lo}, {hi})"
        assert stats["autorange_strategy"] == "hybrid"
        # Counter keys also populated
        for k in ("n_input", "n_filtered", "n_inf_x", "n_nan_x"):
            assert k in stats

    def test_hist1d_explicit_minmax_strategy(self):
        """range='minmax' explicit string forces minmax behavior; strategy reported."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        d = DFDraw(df)

        fig, ax, stats = d.hist("y", range="minmax")

        assert stats["autorange_strategy"] == "minmax"
        assert stats["autorange_used"] == (1.0, 8.0)

    def test_hist1d_explicit_numeric_range_strategy_explicit(self):
        """Explicit numeric range tuple records strategy='explicit' AND
        n stats reflect post-sanitize count, NOT range-clipped count (per AD-77 / §6.4)."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        d = DFDraw(df)

        fig, ax, stats = d.hist("y", range=(2.0, 6.0))

        assert stats["autorange_strategy"] == "explicit"
        assert stats["autorange_used"] == (2.0, 6.0)
        # All 8 rows are post-sanitize finite — not clipped to range
        assert stats["n"] == 8, \
            f"stats['n'] should be 8 (post-sanitize), not range-clipped. Got {stats['n']}"

    def test_nan_policy_raise_propagates_through_hist(self):
        """nan_policy='raise' kwarg flows through DFDraw.hist into draw_hist
        and raises ValueError on inf data."""
        df = pd.DataFrame({"y": [1.0, 2.0, np.inf, 4.0]})
        d = DFDraw(df)

        with pytest.raises(ValueError, match="invalid values"):
            d.hist("y", nan_policy="raise")
