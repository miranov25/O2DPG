"""
Tests for Phase 13.28.DF FIX1 — Restore autorange.* style keys in DEFAULT_STYLE.

Bug context
-----------
Phase 13.28.DF introduced the autorange system with four style keys:
    autorange.strategy
    autorange.k_robust
    autorange.k_outlier
    autorange.percentile

These keys are referenced by `get_style_value("autorange.X", <fallback>)` calls
in plots/profile.py and plots/histogram.py, so plots render correctly using the
hard-coded fallbacks. However, the keys were never registered in `DEFAULT_STYLE`.
Result: `set_style({"autorange.k_robust": 8.0})` raises
`ValueError: Unknown style keys: {'autorange.k_robust'}` because set_style()
validates incoming keys against DEFAULT_STYLE.keys().

Discovered: Sonet50 documentation audit, 2026-05-14.
Tag: BUG_dfdraw_20260512_phase1328_style_keys_removed
Filed under: Phase 13.28.DF FIX1.

Each load-bearing assertion below is marked `# §9.<class>.<id>` per Coder QRC v1.30.
"""
import pytest

from dfdraw.style import (
    DEFAULT_STYLE, set_style, get_style_value
)


class TestAutorangeStyleKeysRegistered:
    """Each autorange.* key referenced by code must exist in DEFAULT_STYLE."""

    def test_autorange_strategy_in_default_style(self):
        """§9.AutorangeKeys.1 — autorange.strategy registered."""
        assert "autorange.strategy" in DEFAULT_STYLE, \
            "autorange.strategy must be in DEFAULT_STYLE for set_style() to accept it"
        # §9.AutorangeKeys.1 — default value matches the fallback used in code
        assert DEFAULT_STYLE["autorange.strategy"] == "hybrid", \
            f"Default must match the fallback used in plots/*.py get_style_value() calls; " \
            f"got {DEFAULT_STYLE['autorange.strategy']!r}"

    def test_autorange_k_robust_in_default_style(self):
        """§9.AutorangeKeys.2 — autorange.k_robust registered with default 4.0."""
        assert "autorange.k_robust" in DEFAULT_STYLE
        assert DEFAULT_STYLE["autorange.k_robust"] == 4.0

    def test_autorange_k_outlier_in_default_style(self):
        """§9.AutorangeKeys.3 — autorange.k_outlier registered with default 1.5."""
        assert "autorange.k_outlier" in DEFAULT_STYLE
        assert DEFAULT_STYLE["autorange.k_outlier"] == 1.5

    def test_autorange_percentile_in_default_style(self):
        """§9.AutorangeKeys.4 — autorange.percentile registered with default (1.0, 99.0)."""
        assert "autorange.percentile" in DEFAULT_STYLE
        assert DEFAULT_STYLE["autorange.percentile"] == (1.0, 99.0)


class TestAutorangeKeysAcceptedBySetStyle:
    """set_style({autorange.* : ...}) must not raise — the original user-visible bug."""

    def test_set_style_accepts_k_robust_override(self):
        """§9.SetStyle.1 — the production reproducer Sonet50 cited: set k_robust=8.0."""
        # §9.SetStyle.1 — does NOT raise
        try:
            set_style({"autorange.k_robust": 8.0})
        except ValueError as e:
            if "Unknown style keys" in str(e) and "autorange.k_robust" in str(e):
                pytest.fail(
                    f"BUG_dfdraw_20260512_phase1328_style_keys_removed still present: {e}"
                )
            raise
        finally:
            set_style(None)  # reset to defaults so other tests are unaffected

        # §9.SetStyle.1 — override actually applied
        # (set again here since reset above; verify get_style_value reflects the override
        # while it is active)
        set_style({"autorange.k_robust": 8.0})
        try:
            assert get_style_value("autorange.k_robust") == 8.0
        finally:
            set_style(None)

    def test_set_style_accepts_strategy_override(self):
        """§9.SetStyle.2 — set autorange.strategy to a non-default value."""
        set_style({"autorange.strategy": "percentile_99"})
        try:
            assert get_style_value("autorange.strategy") == "percentile_99"
        finally:
            set_style(None)

    def test_set_style_accepts_combined_autorange_overrides(self):
        """§9.SetStyle.3 — set multiple autorange.* keys in one call."""
        set_style({
            "autorange.strategy": "robust_3mad",
            "autorange.k_robust": 3.0,
            "autorange.k_outlier": 2.0,
            "autorange.percentile": (5.0, 95.0),
        })
        try:
            assert get_style_value("autorange.strategy") == "robust_3mad"
            assert get_style_value("autorange.k_robust") == 3.0
            assert get_style_value("autorange.k_outlier") == 2.0
            assert get_style_value("autorange.percentile") == (5.0, 95.0)
        finally:
            set_style(None)


class TestNoRegressionInExistingStyleKeys:
    """Verify the patch did not accidentally drop any pre-Phase-13.28-FIX1 key."""

    def test_quantile_band_alpha_still_present(self):
        """§9.NoRegression.1 — pre-existing quantile.band.alpha untouched."""
        assert "quantile.band.alpha" in DEFAULT_STYLE
        assert DEFAULT_STYLE["quantile.band.alpha"] == 0.25

    def test_profile_marker_still_present(self):
        """§9.NoRegression.2 — pre-existing profile.marker untouched."""
        assert "profile.marker" in DEFAULT_STYLE
        assert DEFAULT_STYLE["profile.marker"] == "o"
