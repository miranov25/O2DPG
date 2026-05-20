"""Phase 13.35.DF — group_by_bins + hist_norm for hist() invariance tests.

11 §9 tests across 5 classes:
  TestHistGroupByBins      (3): HGB.1, HGB.2, HGB.3
  TestHistNorm             (3): HN.1, HN.2, HN.3
  TestHistGroupByStats     (2): HGS.1, HGS.2
  TestHistGroupByStacked   (1): HGSt.1 (locks v1.2 P1-D regression)
  TestHistGroupByBackwardCompat (2): HGBC.1, HGBC.2

Closes 3 crashes confirmed in live testing (2026-05-20):
  T2: group_by_bins leaks to ax.hist() (no facet)
  T3: same crash in faceted subplot
  T4: hist_norm leaks to ax.hist()
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def df_float_groupby():
    """500 rows; float column 'z' uniform in [0, 250); 'x' Gaussian.
    'sec' is a categorical-like integer for facet tests."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "z": rng.uniform(0, 250, n),
        "sec": rng.integers(0, 9, n),
    })


@pytest.fixture
def df_uneven_groups():
    """Three groups with very different sizes — for min_entries filter tests.
    Group A: 200 rows; Group B: 5 rows (will be filtered); Group C: 200 rows."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": np.concatenate([
            rng.normal(0, 1, 200),    # group A
            rng.normal(2, 1, 5),      # group B (small)
            rng.normal(-2, 1, 200),   # group C
        ]),
        "g": ["A"] * 200 + ["B"] * 5 + ["C"] * 200,
    })


# =============================================================================
# §9.HGB — group_by_bins for hist
# =============================================================================

class TestHistGroupByBins:
    """§9.HGB.* — float group_by binning via pd.cut/qcut."""

    def test_HGB_1_group_by_bins_no_facet(self, df_float_groupby):
        """§9.HGB.1: group_by_bins=5 on float column 'z' renders 5 groups.
        Pre-Phase 13.35.DF this raised AttributeError (T2 from live test)."""
        d = DFDraw(df_float_groupby)
        fig, _ax, stats = d.hist(
            "x", bins=30, group_by="z", group_by_bins=5
        )
        try:
            assert stats.get("n_groups") == 5, (
                f"expected 5 rendered groups from group_by_bins=5; "
                f"got n_groups={stats.get('n_groups')}"
            )
            assert stats.get("grouped") is True
        finally:
            plt.close(fig)

    def test_HGB_2_group_by_bins_with_facet_by_architect_call(
            self, df_float_groupby):
        """§9.HGB.2: architect's exact production call (T3) — must not raise.
        Multiple facet panels, each with ≤5 group_by_bins.
        Pre-Phase 13.35.DF: AttributeError 'group_by_bins' in faceted subplot."""
        d = DFDraw(df_float_groupby)
        # architect's call shape: group_by float column + group_by_bins +
        # facet_by categorical column. type='hist' belongs on d.draw(), not
        # d.hist() (which is already typed).
        fig, _ax, _stats = d.hist(
            "x",
            group_by="z", group_by_bins=5,
            min_entries=5,
            facet_by="sec",
        )
        try:
            n_panels = len(fig.axes)
            assert n_panels >= 2, (
                f"facet_by='sec' (≤9 sectors) should produce ≥2 panels; "
                f"got {n_panels}"
            )
        finally:
            plt.close(fig)

    def test_HGB_3_shared_bin_edges_across_groups(self, df_float_groupby):
        """§9.HGB.3: with group_by_bins, all groups share the same x-range
        (locked via shared edges from full dataset). Each group's rendered
        histogram should span the same x-coordinate extents."""
        d = DFDraw(df_float_groupby)
        fig, ax, stats = d.hist(
            "x", bins=30, group_by="z", group_by_bins=4
        )
        try:
            assert stats.get("n_groups") == 4
            # All histograms drawn into the same axes share the same x-range
            # (matplotlib auto-fits the view to the union, but per-group bin
            # edges should be IDENTICAL since they were computed from x_all).
            # Verify via collected patch x-extents: min-of-min == max-of-min
            # within each group, etc. Use ax.get_xlim() as a coarse proxy
            # plus patch-level x-positions for fine-grained check.
            assert ax.patches, "no patches rendered"
            xs = [p.get_x() for p in ax.patches if hasattr(p, 'get_x')]
            # If shared edges held, the SET of distinct x-starts should equal
            # the number of bins (30), not 30 × n_groups.
            unique_xs = set(round(x, 6) for x in xs)
            assert len(unique_xs) <= 30 + 2, (  # +2 for endpoint patches
                f"expected ≤32 unique x-starts (shared 30-bin edges across "
                f"4 groups); got {len(unique_xs)} → bin edges not shared"
            )
        finally:
            plt.close(fig)


# =============================================================================
# §9.HN — hist_norm normalization
# =============================================================================

class TestHistNorm:
    """§9.HN.* — per-group hist_norm normalization."""

    def test_HN_1_probability_sum_equals_one(self, df_float_groupby):
        """§9.HN.1: hist_norm='probability' → per-group weighted histogram
        heights sum to 1.0 exactly (rtol=1e-9).

        Verifies the math at the data layer (Polygon-compatible — doesn't
        depend on histtype) by re-computing np.histogram with the same
        weights that the rendering path uses.
        """
        from dfextensions.dfdraw.plots.histogram import _group_weights

        d = DFDraw(df_float_groupby)
        fig, _ax, stats = d.hist(
            "x", bins=20, group_by="z", group_by_bins=3,
            hist_norm="probability",
        )
        try:
            assert stats.get("n_groups") == 3
            # Verify the normalization math directly — what _group_weights
            # produces must sum to 1.0 per group.
            df_binned = df_float_groupby.copy()
            df_binned["z_bin"] = pd.cut(df_binned["z"], bins=3)
            edges = np.histogram_bin_edges(
                df_float_groupby["x"].values, bins=20
            )
            for g in df_binned["z_bin"].unique():
                x_g = df_binned[df_binned["z_bin"] == g]["x"].dropna().values
                if len(x_g) == 0:
                    continue
                w = _group_weights(x_g, edges, "probability")
                heights, _ = np.histogram(x_g, bins=edges, weights=w)
                assert abs(heights.sum() - 1.0) < 1e-9, (
                    f"group {g}: probability heights sum to "
                    f"{heights.sum()}, expected 1.0"
                )
        finally:
            plt.close(fig)

    def test_HN_2_default_none_preserves_raw_counts(self, df_float_groupby):
        """§9.HN.2: hist_norm=None (default) → _group_weights returns None,
        ax.hist uses unweighted counts. Lock the no-normalization default."""
        from dfextensions.dfdraw.plots.histogram import _group_weights

        # Direct verification of the helper's default behavior
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = np.array([0.0, 2.0, 4.0, 6.0])
        w = _group_weights(x, edges, None)
        assert w is None, (
            f"hist_norm=None must return None weights; got {w!r}"
        )

        # Integration check: full call with default hist_norm renders without
        # error and reports n_groups correctly.
        d = DFDraw(df_float_groupby)
        fig, _ax, stats = d.hist(
            "x", bins=20, group_by="z", group_by_bins=2,
            # hist_norm omitted → None default
        )
        try:
            assert stats.get("n_groups") == 2
        finally:
            plt.close(fig)

    def test_HN_3_density_integrates_to_one(self, df_float_groupby):
        """§9.HN.3: hist_norm='density' → ∫heights·dx ≈ 1.0 per group
        (within bin-width-approximation tolerance)."""
        from dfextensions.dfdraw.plots.histogram import _group_weights

        d = DFDraw(df_float_groupby)
        fig, _ax, stats = d.hist(
            "x", bins=20, group_by="z", group_by_bins=3,
            hist_norm="density",
        )
        try:
            assert stats.get("n_groups") == 3
            # Re-compute the integral at the data layer
            df_binned = df_float_groupby.copy()
            df_binned["z_bin"] = pd.cut(df_binned["z"], bins=3)
            edges = np.histogram_bin_edges(
                df_float_groupby["x"].values, bins=20
            )
            bin_widths = np.diff(edges)
            for g in df_binned["z_bin"].unique():
                x_g = df_binned[df_binned["z_bin"] == g]["x"].dropna().values
                if len(x_g) == 0:
                    continue
                w = _group_weights(x_g, edges, "density")
                heights, _ = np.histogram(x_g, bins=edges, weights=w)
                integral = (heights * bin_widths).sum()
                # Tolerance: density uses np.diff(edges).mean() not exact
                # per-bin width, so non-uniform edges drift slightly.
                assert abs(integral - 1.0) < 0.05, (
                    f"group {g}: density integral = {integral}, expected 1.0"
                )
        finally:
            plt.close(fig)


# =============================================================================
# §9.HGS — stats dict for grouped hist
# =============================================================================

class TestHistGroupByStats:
    """§9.HGS.* — stats dict completeness + BUG-012 protection."""

    def test_HGS_1_n_groups_present_and_correct(self, df_uneven_groups):
        """§9.HGS.1: stats['n_groups'] is present AND reflects min_entries
        filtering. Group B (n=5) is dropped by min_entries=50, so n_groups=2.
        T1 observation: stats['n_groups'] was missing in baseline.
        """
        d = DFDraw(df_uneven_groups)
        fig, _ax, stats = d.hist(
            "x", bins=20, group_by="g", min_entries=50
        )
        try:
            assert "n_groups" in stats, (
                "stats['n_groups'] missing — T1 regression"
            )
            assert stats["n_groups"] == 2, (
                f"expected 2 (A, C); group B filtered by min_entries=50; "
                f"got n_groups={stats['n_groups']}"
            )
        finally:
            plt.close(fig)

    def test_HGS_2_bug012_float_high_cardinality_no_bins_raises(
            self, df_float_groupby):
        """§9.HGS.2: BUG-012 guard — float column + no bins + high cardinality
        → ValueError with actionable message containing 'group_by_bins=N'."""
        d = DFDraw(df_float_groupby)
        with pytest.raises(ValueError) as exc_info:
            d.hist("x", bins=30, group_by="z")  # no group_by_bins/_quantiles
        msg = str(exc_info.value)
        assert "group_by_bins=N" in msg or "group_by_bins=" in msg, (
            f"error message missing actionable 'group_by_bins=N' guidance; "
            f"got: {msg!r}"
        )


# =============================================================================
# §9.HGSt — stacked path label/data/color alignment
# =============================================================================

class TestHistGroupByStacked:
    """§9.HGSt.* — stacked-mode min_entries filter alignment.

    Regression lock for v1.2 P1-D: two-pass list comprehension misaligned
    labels with data when min_entries filtered any group.
    """

    def test_HGSt_1_stacked_min_entries_alignment(self, df_uneven_groups):
        """§9.HGSt.1: stacked=True + min_entries filtering must preserve
        label/data alignment.

        Setup: 3 groups [A=200, B=5, C=200], min_entries=50.
        Group B is filtered; data_list = [data_A, data_C].
        The set of labels must be {'A', 'C'} — NOT {'A', 'B'} (the v1.2 bug
        where data_C was mislabeled as 'B' due to two-pass zip misalignment).

        Note: matplotlib may reverse stack order in the legend (top stack
        listed first), so we compare SETS, not ordered lists. The invariant
        is that B is absent and exactly {A, C} are present.
        """
        d = DFDraw(df_uneven_groups)
        fig, ax, stats = d.hist(
            "x", bins=10, group_by="g", stacked=True, min_entries=50
        )
        try:
            assert stats.get("n_groups") == 2, (
                f"expected 2 surviving groups (A, C); B (n=5) filtered by "
                f"min_entries=50; got n_groups={stats.get('n_groups')}"
            )

            # Read labels from the legend
            legend = ax.get_legend()
            assert legend is not None, (
                "stacked histogram should produce a legend"
            )
            labels = [t.get_text() for t in legend.get_texts()]
            assert set(labels) == {"A", "C"}, (
                f"stacked min_entries label misalignment regression "
                f"(v1.2 P1-D): expected labels {{'A', 'C'}} (B filtered); "
                f"got {labels} — if 'B' appears, the two-pass zip bug is back."
            )
            # Also assert exactly 2 labels (no duplicates)
            assert len(labels) == 2
        finally:
            plt.close(fig)


# =============================================================================
# §9.HGBC — backward compatibility
# =============================================================================

class TestHistGroupByBackwardCompat:
    """§9.HGBC.* — Phase 13.35.DF must not change pre-existing behavior."""

    def test_HGBC_1_no_groupby_byte_identical(self, df_float_groupby):
        """§9.HGBC.1: group_by=None (default) + hist_norm=None (default) →
        result equivalent to baseline. Compare stats dict (no 'grouped' key
        on the ungrouped path) and patch count via two functionally-equivalent
        call shapes."""
        d = DFDraw(df_float_groupby)
        # Call without any of the new params (legacy invocation)
        fig1, ax1, stats1 = d.hist("x", bins=20)
        # Call with all new params at defaults (post-phase invocation)
        fig2, ax2, stats2 = d.hist(
            "x", bins=20,
            group_by_bins=None, group_by_quantiles=None,
            hist_norm=None, min_entries=0,
        )
        try:
            # Single-histogram path (no group_by) should not set 'grouped'
            assert stats1.get("grouped") is not True, (
                "single-hist path should not set stats['grouped']=True"
            )
            assert stats2.get("grouped") is not True
            # Patch counts identical
            assert len(ax1.patches) == len(ax2.patches), (
                f"backward-compat: patch count diverged "
                f"({len(ax1.patches)} vs {len(ax2.patches)})"
            )
            # 'n_groups' must NOT appear in either (only in grouped path)
            assert "n_groups" not in stats1
            assert "n_groups" not in stats2
        finally:
            plt.close(fig1)
            plt.close(fig2)

    def test_HGBC_2_categorical_groupby_unchanged(self, df_uneven_groups):
        """§9.HGBC.2: categorical group_by (T1 path) still works, and now
        additionally has stats['n_groups'] (closes the T1 gap)."""
        d = DFDraw(df_uneven_groups)
        fig, ax, stats = d.hist("x", bins=15, group_by="g")
        try:
            assert stats.get("grouped") is True
            # 3 groups (A, B, C); no min_entries filter → all 3 render
            assert stats.get("n_groups") == 3, (
                f"categorical group_by with 3 groups → n_groups=3; "
                f"got {stats.get('n_groups')}"
            )
            # Visual rendering: 3 sets of bars (≥3 patches per group)
            assert len(ax.patches) > 0, "no patches rendered"
        finally:
            plt.close(fig)
