"""
Tests for Phase 13.32.DF v1.2 — group_by × quantiles × facet_by binning integration.

Implements the §9 test plan from PHASE_13_32_DF_v1_2_Proposal §4.

Test classes:
- TestSubfix1                 (3) — facet_by='group_by' honors group_by_bins/_quantiles at dispatch
- TestSubfix2                 (5) — _draw_profile_grouped renders quantiles per group
- TestSubfix3Profile          (5) — facet_by_bins / facet_by_quantiles on profile()
- TestSubfix3AllPlots         (3) — facet_by_bins works for hist / hist2d / scatter
- TestRepro                   (2) — architect's production reproducers In[126] and In[129]

Each load-bearing assertion is marked with `# §9.<class>.<id>` per Coder QRC v1.30 Rule 14.

References
----------
- Proposal: PHASE_13_32_DF_v1_2_Proposal_GroupByQuantilesFacetBinning.md
- Predecessors: PHASE_13_31_DF_FacetByColumn_v1_0_END (f3ca432a),
                PHASE_13_28_DF_FIX1_END (57576ebf)
- AD-78 (facet_by tagged union), AD-79 (symmetric facet_by_bins/_quantiles)
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def df_60_groups():
    """DataFrame with 60 unique values in 'driftM_bin25' — simulates architect's
    production case where raw column cardinality exceeds the 16-cap but
    group_by_bins=5 should collapse to 5 facets."""
    rng = np.random.default_rng(seed=42)
    n_per_group = 20
    rows = []
    for g in range(60):
        for _ in range(n_per_group):
            rows.append({
                "row": rng.uniform(0, 150),
                "dyp_I0345_intercept": g * 0.01 + rng.normal(0, 0.3),
                "driftM_bin25": g,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_with_side_column():
    """DataFrame with a string-typed 'side' column for facet_by column-mode tests."""
    rng = np.random.default_rng(seed=43)
    n_per_side = 200
    rows = []
    for side in ("A", "B"):
        for _ in range(n_per_side):
            rows.append({
                "row": rng.uniform(0, 150),
                "y": (0.3 if side == "A" else -0.2) + rng.normal(0, 0.5),
                "x": rng.uniform(0, 10),
                "side": side,
                "quartile_val": rng.uniform(0, 100),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_simple_grouped():
    """3-group dataset for sub-fix 2 quantile rendering tests."""
    rng = np.random.default_rng(seed=44)
    n_per_group = 200
    rows = []
    for g in range(3):
        for _ in range(n_per_group):
            rows.append({
                "row": rng.uniform(0, 150),
                "y": g * 0.5 + rng.normal(0, 0.4),
                "sector": g,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Class 1 — Sub-fix 1: facet_by='group_by' honors group_by_bins / _quantiles
# =============================================================================

class TestSubfix1:
    """Phase 13.32 Sub-fix 1 — binning hoisted into _dispatch_faceted_render."""

    def test_facet_groupby_with_bins_honors_n(self, df_60_groups):
        """§9.S1.1 — facet_by='group_by' + group_by_bins=5 → 5 subplots,
        each with exactly one profile line (NOT 5 spurious sub-groups from
        re-binning inside per-subplot recursion)."""
        d = DFDraw(df_60_groups)
        fig, axes, stats = d.profile(
            "dyp_I0345_intercept:row",
            group_by="driftM_bin25", group_by_bins=5,
            facet_by="group_by",
        )

        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S1.1 — exactly 5 subplots (the binned cardinality, not the raw 60)
        assert len(visible_axes) == 5, \
            f"facet_by='group_by' + group_by_bins=5 should produce 5 subplots; " \
            f"got {len(visible_axes)}. If you see 60, the binning is not " \
            f"applied at dispatch level (Sub-fix 1 broken)."

        # §9.S1.1 — each subplot has exactly ONE profile (one ErrorbarContainer).
        # NB: errorbar() creates 3 Line2D objects per call (main + 2 cap lines),
        # so counting ax.lines is wrong. ax.containers has exactly one
        # ErrorbarContainer per profile — that's the invariant we lock.
        # Locks FIX 1 invariant from Sonet51/Claude48/GPT1 v1.0 review:
        # multiple containers in a subplot would mean group_by/group_by_bins
        # kwargs leaked into per-subplot recursion.
        for i, ax in enumerate(visible_axes):
            n_profiles = len(ax.containers)
            assert n_profiles == 1, (
                f"Subplot {i}: expected exactly 1 ErrorbarContainer (1 profile), "
                f"got {n_profiles}. Multiple profiles indicate group_by/"
                f"group_by_bins kwargs leaked into per-subplot recursion "
                f"(FIX 1 P1 from v1.0 panel review)."
            )
        plt.close(fig)

    def test_facet_groupby_with_quantiles_honors_n(self, df_60_groups):
        """§9.S1.2 — facet_by='group_by' + group_by_quantiles=5 → 5 subplots."""
        d = DFDraw(df_60_groups)
        fig, axes, stats = d.profile(
            "dyp_I0345_intercept:row",
            group_by="driftM_bin25", group_by_quantiles=5,
            facet_by="group_by",
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S1.2 — exactly 5 subplots (qcut quantile binning)
        assert len(visible_axes) == 5
        # §9.S1.2 — each subplot has exactly 1 profile (1 ErrorbarContainer);
        # see §9.S1.1 comment for why ax.containers (not ax.lines) is correct.
        for ax in visible_axes:
            assert len(ax.containers) == 1
        plt.close(fig)

    def test_facet_groupby_cap_still_fires_without_binning(self, df_60_groups):
        """§9.S1.3 — Regression: without group_by_bins, raw 60-unique column
        must still trigger the facet_max=16 cap."""
        d = DFDraw(df_60_groups)
        # §9.S1.3 — raise ValueError matching 'facet_max' / cardinality message
        with pytest.raises(ValueError, match=r"facet_max|exceeding"):
            d.profile(
                "dyp_I0345_intercept:row",
                group_by="driftM_bin25",  # no group_by_bins → raw 60 values
                facet_by="group_by",
            )


# =============================================================================
# Class 2 — Sub-fix 2: _draw_profile_grouped renders quantiles per group
# =============================================================================

class TestSubfix2:
    """Phase 13.32 Sub-fix 2 — quantile rendering in grouped path."""

    def test_quantile_band_per_group_color(self, df_simple_grouped):
        """§9.S2.1 — group_by + quantiles=(lo,hi) + mode='band' produces N
        band patches (PolyCollection from fill_between)."""
        from matplotlib.collections import PolyCollection
        d = DFDraw(df_simple_grouped)
        fig, ax, stats = d.profile(
            "y:row",
            group_by="sector",
            quantiles=[0.16, 0.5, 0.84],
            quantile_mode="band",
        )
        # §9.S2.1 — one PolyCollection (from fill_between) per group.
        # NB: errorbar() also registers a LineCollection per group on
        # ax.collections, so we must filter by type — counting raw
        # ax.collections would double the value (3 PolyCollection + 3
        # LineCollection = 6).
        n_bands = sum(1 for c in ax.collections if isinstance(c, PolyCollection))
        assert n_bands == 3, \
            f"Expected 3 band patches (one PolyCollection per group), got " \
            f"{n_bands}. This locks Sub-fix 2: _draw_profile_grouped must " \
            f"call _render_quantile_band per group."
        plt.close(fig)

    def test_quantile_discrete_per_group(self, df_simple_grouped):
        """§9.S2.2 — discrete mode: 3 groups × 3 quantiles (excluding median).
        Per v1.2 §3.2 inline simplified block, each group renders its
        per-quantile lines in group_color + linestyle='--'."""
        d = DFDraw(df_simple_grouped)
        fig, ax, stats = d.profile(
            "y:row",
            group_by="sector",
            quantiles=[0.25, 0.5, 0.75],
            quantile_mode="discrete",
            central="median",
        )
        # §9.S2.2 — total Line2D objects: per group, 1 central + 3 quantile = 4
        # Across 3 groups: 12 lines minimum (errorbar may add caps; >=12)
        n_lines = len(ax.lines)
        assert n_lines >= 12, \
            f"Expected >=12 lines (3 groups × (1 central + 3 quantile)), " \
            f"got {n_lines}"
        plt.close(fig)

    def test_stats_dict_quantile_keys_present(self, df_simple_grouped):
        """§9.S2.3 — stats_dict for grouped + quantile contains the
        documented schema: 'quantile_mode', 'quantile_pair', 'per_group'."""
        d = DFDraw(df_simple_grouped)
        fig, ax, stats = d.profile(
            "y:row",
            group_by="sector",
            quantiles=[0.16, 0.5, 0.84],
            quantile_mode="band",
        )
        # §9.S2.3 — required top-level keys per §3.2 schema
        assert stats.get("grouped") is True
        assert "quantile_mode" in stats, \
            "stats_dict missing 'quantile_mode' key (per v1.2 §3.2 schema)"
        assert stats["quantile_mode"] == "band"
        assert "per_group" in stats, \
            "stats_dict missing 'per_group' key (per v1.2 §3.2 schema)"
        # §9.S2.3 — per-group entries contain q_lower_per_bin / q_upper_per_bin
        for group_label, group_stats in stats["per_group"].items():
            assert "q_lower_per_bin" in group_stats, \
                f"Per-group stats for {group_label} missing q_lower_per_bin"
            assert "q_upper_per_bin" in group_stats, \
                f"Per-group stats for {group_label} missing q_upper_per_bin"
        plt.close(fig)

    def test_weights_compose_with_groupby_quantiles(self, df_simple_grouped):
        """§9.S2.4 — Phase 13.12 F4 weights still composes with grouped path.
        NB: weighted-quantile computation is deferred to Phase 13.25 Phase B
        (`NotImplementedError: weighted quantiles deferred to Phase B`), so
        this test locks the *unweighted-quantile path with weighted central
        line* — the regression we actually need to protect for Phase 13.12 F4.
        Once weighted quantiles land, this test should be tightened to add
        `quantiles=...` alongside `weights=`."""
        from matplotlib.collections import PolyCollection
        df = df_simple_grouped.copy()
        df["w"] = 1.0  # uniform weights → same result as unweighted
        d = DFDraw(df)
        fig, ax, stats = d.profile(
            "y:row",
            group_by="sector",
            weights="w",
        )
        # §9.S2.4 — grouped plot still rendered (3 ErrorbarContainer, one per
        # group; ax.containers is the invariant — see §9.S1.1 comment).
        assert len(ax.containers) == 3
        assert stats.get("grouped") is True
        plt.close(fig)

    def test_nested_band_with_groupby_raises(self, df_simple_grouped):
        """§9.S2.5 — nested_band mode in grouped path raises NotImplementedError
        explicitly rather than silently dropping quantiles (this is the exact
        silent-drop bug class Phase 13.32 exists to fix)."""
        d = DFDraw(df_simple_grouped)
        # §9.S2.5 — auto-detected nested_band (4+ symmetric quantiles) with
        # group_by → explicit NotImplementedError
        with pytest.raises(NotImplementedError, match=r"nested_band"):
            d.profile(
                "y:row",
                group_by="sector",
                quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                # quantile_mode defaults to 'auto' → resolves to 'nested_band'
            )

    def test_discrete_grouped_uses_group_color_dashed(self, df_simple_grouped):
        """§9.S2.6 — Per v1.2 §3.2 (option B inline): discrete-mode quantile
        lines in grouped path use group_color + linestyle='--' (no FIX2
        channel cycling). Locks the inline simplified design."""
        d = DFDraw(df_simple_grouped)
        fig, ax, stats = d.profile(
            "y:row",
            group_by="sector",
            quantiles=[0.25, 0.5, 0.75],
            quantile_mode="discrete",
            central="median",
        )
        # §9.S2.6 — find the quantile lines (linestyle='--' identifies them).
        # There should be 3 groups × 3 quantiles = 9 dashed lines.
        dashed_lines = [ln for ln in ax.lines if ln.get_linestyle() == '--']
        assert len(dashed_lines) >= 9, \
            f"Expected >=9 dashed lines (3 groups × 3 quantiles); got " \
            f"{len(dashed_lines)}. Locks Sub-fix 2 inline simplified discrete " \
            f"rendering (v1.2 §3.2 option B, no FIX2 channel cycling)."
        plt.close(fig)


# =============================================================================
# Class 3 — Sub-fix 3 on profile(): facet_by_bins / facet_by_quantiles
# =============================================================================

class TestSubfix3Profile:
    """Phase 13.32 Sub-fix 3 — facet_by_bins / facet_by_quantiles on profile()."""

    def test_facet_by_bins_on_column_facet(self, df_with_side_column):
        """§9.S3.1 — profile() with facet_by='quartile_val' + facet_by_bins=5
        → 5 subplots from the raw float column."""
        d = DFDraw(df_with_side_column)
        fig, axes, stats = d.profile(
            "y:row",
            facet_by="quartile_val", facet_by_bins=5,
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S3.1 — exactly 5 subplots from pd.cut binning
        assert len(visible_axes) == 5, \
            f"Expected 5 subplots; got {len(visible_axes)}"
        plt.close(fig)

    def test_facet_by_quantiles_on_column_facet(self, df_with_side_column):
        """§9.S3.2 — same with facet_by_quantiles=5 (pd.qcut)."""
        d = DFDraw(df_with_side_column)
        fig, axes, stats = d.profile(
            "y:row",
            facet_by="quartile_val", facet_by_quantiles=5,
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S3.2 — exactly 5 subplots from pd.qcut binning
        assert len(visible_axes) == 5
        plt.close(fig)

    def test_facet_by_bins_on_channel_facet_raises(self, df_simple_grouped):
        """§9.S3.3 — facet_by='group_by' + facet_by_bins=5 raises ValueError;
        binning a channel-enum value is invalid (channel names aren't columns)."""
        d = DFDraw(df_simple_grouped)
        # §9.S3.3 — error message names that facet_by must be a column
        with pytest.raises(ValueError, match=r"column"):
            d.profile(
                "y:row",
                group_by="sector",
                facet_by="group_by", facet_by_bins=5,
            )

    def test_facet_by_bins_quantiles_mutual_exclusion(self, df_with_side_column):
        """§9.S3.4 — Cannot specify both facet_by_bins AND facet_by_quantiles.
        Mirror of Phase 13.12 Feature 3 mutex rule for group_by_bins/quantiles."""
        d = DFDraw(df_with_side_column)
        # §9.S3.4 — ValueError matching 'both' / 'Cannot specify'
        with pytest.raises(ValueError, match=r"both|Cannot specify"):
            d.profile(
                "y:row",
                facet_by="quartile_val",
                facet_by_bins=5,
                facet_by_quantiles=5,
            )

    def test_facet_by_bins_without_facet_by_raises(self, df_with_side_column):
        """§9.S3.5 — facet_by_bins= without facet_by= must raise (defensive
        validation per P2 from Claude48)."""
        d = DFDraw(df_with_side_column)
        # §9.S3.5 — ValueError mentions facet_by= must be set
        with pytest.raises(ValueError, match=r"facet_by"):
            d.profile(
                "y:row",
                facet_by_bins=5,  # no facet_by= → must raise
            )


# =============================================================================
# Class 4 — Sub-fix 3 plot-type-agnostic: hist, hist2d, scatter
# =============================================================================

class TestSubfix3AllPlots:
    """Phase 13.32 Sub-fix 3 per AD-79 — facet_by_bins/_quantiles plot-type-agnostic."""

    def test_facet_by_bins_for_hist(self, df_with_side_column):
        """§9.S3.6 — hist() with facet_by='col', facet_by_bins=5 → 5 subplots."""
        d = DFDraw(df_with_side_column)
        fig, axes, stats = d.hist(
            "y",
            facet_by="quartile_val", facet_by_bins=5,
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S3.6 — exactly 5 subplots (locks plot-type-agnostic dispatch)
        assert len(visible_axes) == 5
        plt.close(fig)

    def test_facet_by_bins_for_hist2d(self, df_with_side_column):
        """§9.S3.7 — hist2d() with facet_by_bins."""
        d = DFDraw(df_with_side_column)
        # hist2d expects "y:x" expression form
        fig, axes, stats = d.hist2d(
            "y:row",
            facet_by="quartile_val", facet_by_bins=5,
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S3.7 — facet_max governs subplot count; with binning to 5,
        # we expect 5 subplots (plus possibly colorbars, which are not
        # primary axes — count only those with title or labels matching
        # the y-x expression)
        # Use len(stats['per_group']) for robustness if available
        # otherwise fall back to axes count >= 5
        assert len(visible_axes) >= 5, \
            f"Expected at least 5 hist2d subplots; got {len(visible_axes)}"
        plt.close(fig)

    def test_facet_by_bins_for_scatter(self, df_with_side_column):
        """§9.S3.8 — scatter() with facet_by_bins."""
        d = DFDraw(df_with_side_column)
        fig, axes, stats = d.scatter(
            "y:row",
            facet_by="quartile_val", facet_by_bins=5,
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.S3.8 — exactly 5 subplots for scatter
        assert len(visible_axes) == 5
        plt.close(fig)


# =============================================================================
# Class 5 — Reproducers for architect's production calls In[126] and In[129]
# =============================================================================

class TestRepro:
    """The two production failures Phase 13.32 exists to fix."""

    def test_in_126_overlay_with_quantiles(self, df_60_groups):
        """§9.Repro.1 — architect's In[126] call: overlay form with
        group_by + quantiles must render bands per group (NOT silently drop
        quantiles as the v1.0 bug did)."""
        d = DFDraw(df_60_groups)
        fig, ax, stats = d.profile(
            "dyp_I0345_intercept:row",
            group_by="driftM_bin25", group_by_bins=5,
            quantiles=[0.1, 0.2, 0.5, 0.8, 0.9],
            quantile_mode="discrete",  # explicit to avoid auto→nested_band
            central="median",
        )
        # §9.Repro.1 — 5 groups rendered (binned from 60)
        assert stats.get("grouped") is True, \
            "stats['grouped'] must be True (the only thing v1.0 already did " \
            "correctly)"
        # §9.Repro.1 — quantile keys now present in stats (v1.0 silently dropped)
        assert "quantile_mode" in stats, \
            "Production In[126] reproducer: quantile_mode missing from stats. " \
            "Quantiles silently dropped — Sub-fix 2 broken."
        assert "per_group" in stats
        # §9.Repro.1 — discrete mode renders per-quantile lines per group
        # 5 groups × 5 quantiles = 25 quantile lines + 5 central lines = 30 min
        # (errorbar caps may add more; we just verify ample line presence)
        assert len(ax.lines) >= 25, \
            f"Production In[126] reproducer: only {len(ax.lines)} lines; " \
            f"expected at least 25 (5 groups × 5 quantiles)"
        plt.close(fig)

    def test_in_129_facet_with_groupby_bins(self, df_60_groups):
        """§9.Repro.2 — architect's In[129] call: facet form with
        facet_by='group_by' + group_by_bins=5. Must produce 5 subplots
        (NOT raise 'exceeds facet_max=16')."""
        d = DFDraw(df_60_groups)
        fig, axes, stats = d.profile(
            "dyp_I0345_intercept:row",
            group_by="driftM_bin25", group_by_bins=5,
            facet_by="group_by",
            quantiles=[0.1, 0.2, 0.5, 0.8, 0.9],
            quantile_mode="discrete",  # explicit
            central="median",
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # §9.Repro.2 — exactly 5 subplots (binned cardinality, not raw 60)
        assert len(visible_axes) == 5, \
            f"Production In[129] reproducer: expected 5 subplots from " \
            f"group_by_bins=5; got {len(visible_axes)}. Sub-fix 1 broken."
        plt.close(fig)
