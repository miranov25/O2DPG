"""
Tests for Phase 13.28.DF — Robust Data Handling.

Part A (NaN/inf filter): 13 real invariance tests on sanitize_for_plot().
Part B (Hybrid autorange): 8 skip stubs — code not yet implemented.

When Part B implementation lands, the 8 stubs become real bodies in the
same file. No extra restructuring needed.

References
----------
- Proposal: PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §8
- AD-69, AD-70, AD-71 (Part A); AD-72, AD-73, AD-77 (Part B)
"""
import warnings
import pytest
import numpy as np

# Phase 13.28.DF — Part A is functional; tested directly.
from dfdraw.plots._data_sanitize import sanitize_for_plot


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
            warnings.simplefilter("error")  # any warning -> test fails
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
        # All counters present and zero
        for k in ("n_inf_x", "n_nan_x", "n_inf_y", "n_nan_y", "n_filtered"):
            assert stats[k] == 0, f"{k} should be 0 for clean data"
        assert stats["n_input"] == 5
        assert len(x_clean) == 5
        assert len(y_clean) == 5
        # Clean data passes through unchanged
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
        # Should not raise:
        x_clean, _, stats = sanitize_for_plot(x, nan_policy="raise")
        assert len(x_clean) == 3
        assert stats["n_filtered"] == 0


# =============================================================================
# Class 3 — TestHybridAutorange (5 stubs, Part B)
# Real bodies land alongside compute_autorange / hybrid_autorange implementation.
# =============================================================================

class TestHybridAutorange:
    """Hybrid autorange algorithm — verifies §4.1 spec across 4 worked examples."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_clean_gaussian_uses_minmax(self):
        """Clean Gaussian: range_used == (data.min(), data.max())."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_outlier_high_clips_to_robust(self):
        """N(0,1)+1 outlier at +100: range_used[1] ≈ 4·1.4826."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_outlier_low_clips_to_robust(self):
        """Asymmetric outlier at -100: range_used[0] ≈ -4·1.4826."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_asymmetric_distribution_one_sided_clip(self):
        """Long right tail: low side preserved, high side clipped."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_constant_data_returns_unit_range(self):
        """Constant data: range_used == (median - 0.5, median + 0.5)."""


# =============================================================================
# Class 4 — TestAutorangeStrategies (3 stubs, Part B)
# =============================================================================

class TestAutorangeStrategies:
    """Strategy preset behavior — minmax, percentile, style-key default."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_minmax_strategy_equals_data_min_max(self):
        """range='minmax': range_used == (data.min(), data.max())."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_percentile_99_clips_to_quantiles(self):
        """range='percentile_99': range_used == np.percentile(data, [1, 99])."""

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange not yet implemented")
    def test_strategy_style_key_default(self):
        """set_style({'autorange.strategy':'minmax'}) propagates as default."""


# =============================================================================
# Class 5 — TestStatsDictAdditive (3 tests — sanitize half real, autorange stub)
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
        # In this case, x had 1 inf and 1 nan, so 2 dropped:
        assert stats["n_inf_x"] + stats["n_nan_x"] == stats["n_filtered"]

    @pytest.mark.skip(reason="PHASE_13_28_DF Part B — autorange diagnostic keys not yet implemented")
    def test_explicit_numeric_range_records_strategy_explicit(self):
        """Per AD-77: stats['autorange_strategy']=='explicit' when user passes
        numeric range; stats['n'] not clipped to range (visual only)."""
