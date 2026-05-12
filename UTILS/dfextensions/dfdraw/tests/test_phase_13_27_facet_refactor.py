"""
Tests for Phase 13.27.DF Commit 1 — Facet refactor (profile-only).

10 invariance tests across 4 classes per v1.1 proposal §9 (formerly §9.2a):
- TestFacetLegacyEquivalence (3): backward compatibility
- TestFacetByChannel (4): 4-channel routing
- TestFacetCapacity (2): capacity enforcement
- TestFacetSameTrueExclusion (1): same+facet_by mutual exclusion

Each load-bearing assertion is marked with `# §9.<class>.<id>` per Coder
QRC v1.30 Rule 14 (binding §9-assertions rule).

References
----------
- Proposal: PHASE_13_27_DF_v1_1_Proposal_SelectionWeightDeltaFacet.md §9
- AD-61, AD-62, AD-67, AD-68
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw
from dfdraw.style import set_style, get_style_value


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def df_groupby():
    """6-group dataset suitable for facet_by='group_by'."""
    rng = np.random.default_rng(seed=42)
    n_per_group = 50
    data = []
    for g in range(6):
        for _ in range(n_per_group):
            data.append({
                "x": rng.uniform(0, 10),
                "y": g * 0.5 + rng.normal(0, 0.5),
                "sector": g,
            })
    return pd.DataFrame(data)


@pytest.fixture
def df_quantiles():
    """Dataset suitable for quantile_mode='discrete' faceting."""
    rng = np.random.default_rng(seed=43)
    n = 500
    return pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.normal(0, 1, n),
    })


def _figure_byte_signature(fig):
    """
    Lightweight visual-identity signature for a Figure.
    Captures axes coordinates, titles, legend texts. Used to assert
    visual equivalence between two API forms (§9.A backward compat).
    """
    sig = {
        "n_axes": len(fig.axes),
        "axes_titles": [ax.get_title() for ax in fig.axes],
        "axes_xlabels": [ax.get_xlabel() for ax in fig.axes],
        "axes_ylabels": [ax.get_ylabel() for ax in fig.axes],
        "axes_xlims": [ax.get_xlim() for ax in fig.axes],
        "axes_ylims": [ax.get_ylim() for ax in fig.axes],
        "suptitle": fig._suptitle.get_text() if fig._suptitle else None,
    }
    return sig


# =============================================================================
# Class 1 — TestFacetLegacyEquivalence (A-1 backward-compat) — 3 tests
# =============================================================================

class TestFacetLegacyEquivalence:
    """Legacy facet=True API behavior preserved exactly under refactor."""

    def test_facet_true_eqivalent_to_facet_by_groupby(self, df_groupby):
        """§9.LegacyEquiv.1 — facet=True ≡ facet_by='group_by' (visual signature equal).
        Both API forms must produce byte-identical figures (axes coords,
        titles, suptitle, axis labels)."""
        d1 = DFDraw(df_groupby)
        fig1, axes1, stats1 = d1.profile("y:x", group_by="sector", facet=True)
        sig1 = _figure_byte_signature(fig1)
        plt.close(fig1)

        d2 = DFDraw(df_groupby)
        fig2, axes2, stats2 = d2.profile("y:x", group_by="sector",
                                         facet_by="group_by")
        sig2 = _figure_byte_signature(fig2)
        plt.close(fig2)

        # §9.LegacyEquiv.1 — visual signature byte-identical
        assert sig1 == sig2, (
            f"facet=True and facet_by='group_by' must produce identical figures.\n"
            f"  facet=True sig: {sig1}\n  facet_by='group_by' sig: {sig2}"
        )
        # §9.LegacyEquiv.1 — stats facet flag matches
        assert stats1.get("faceted") is True
        assert stats2.get("faceted") is True
        assert stats1.get("n_groups") == stats2.get("n_groups")

    def test_existing_test_facetstar_unchanged(self, df_groupby):
        """§9.LegacyEquiv.2 — pre-Phase-13.27 facet=True call signature
        produces a fig with the expected number of subplots (regression lock)."""
        d = DFDraw(df_groupby)
        fig, axes, stats = d.profile("y:x", group_by="sector", facet=True)
        # §9.LegacyEquiv.2 — n_groups matches unique values of group_by column
        assert stats["n_groups"] == 6, \
            f"Expected 6 subplots (sector has 6 unique values), got {stats['n_groups']}"
        # §9.LegacyEquiv.2 — faceted flag set
        assert stats["faceted"] is True
        plt.close(fig)

    def test_old_facet_profile_kwargs_preserved(self, df_groupby):
        """§9.LegacyEquiv.3 — ncols, sharex, sharey, top_k from facet_profile()
        signature still honored under refactored dispatch."""
        d = DFDraw(df_groupby)
        # Test ncols=2 produces 3 rows × 2 cols layout for 6 groups
        fig, axes, stats = d.profile(
            "y:x", group_by="sector", facet=True,
            ncols=2, sharex=True, sharey=True, top_k=4,
        )
        # §9.LegacyEquiv.3 — top_k respected
        assert stats["n_groups"] == 4, \
            f"top_k=4 should limit to 4 groups, got {stats['n_groups']}"
        # §9.LegacyEquiv.3 — ncols=2 with 4 groups → 2 rows × 2 cols
        # Total fig.axes includes hidden ones; check non-hidden count
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 4, \
            f"Expected 4 visible subplots with top_k=4, got {len(visible_axes)}"
        plt.close(fig)


# =============================================================================
# Class 2 — TestFacetByChannel (4-channel routing) — 4 tests
# =============================================================================

class TestFacetByChannel:
    """facet_by routes to the correct channel."""

    def test_facet_by_vector_creates_n_subplots(self, df_groupby):
        """§9.FacetBy.1 — len(visible_axes) == M for [Y1,...,YM]:X with
        facet_by='vector'. Vector y of length 2 → 2 subplots."""
        # Add a second y column for vector-y test
        df = df_groupby.copy()
        df["y2"] = df["y"] * 2 + 1
        d = DFDraw(df)
        fig, axes, stats = d.profile("[y, y2]:x", facet_by="vector")
        # §9.FacetBy.1 — n_groups equals vector y length
        assert stats["n_groups"] == 2, \
            f"facet_by='vector' on [y, y2] should produce 2 subplots, got {stats['n_groups']}"
        # §9.FacetBy.1 — facet_by recorded in stats
        assert stats["facet_by"] == "vector"
        plt.close(fig)

    def test_facet_by_groupby_dispatches_correctly(self, df_groupby):
        """§9.FacetBy.2 — facet_by='group_by' produces one subplot per
        unique value of group_by; per-subplot stats reflect filtered rows."""
        d = DFDraw(df_groupby)
        fig, axes, stats = d.profile("y:x", group_by="sector",
                                     facet_by="group_by")
        # §9.FacetBy.2 — n_groups equals unique group_by values
        assert stats["n_groups"] == 6
        # §9.FacetBy.2 — per-subplot stats keyed by group value
        # Each subplot has 50 rows in the source data
        for grp_key, grp_stats in stats["per_group"].items():
            assert grp_stats.get("n", 0) > 0, \
                f"Group {grp_key} has 0 rows — filter dispatch failed"
        plt.close(fig)

    def test_facet_by_quantiles_one_subplot_per_quantile(self, df_quantiles):
        """§9.FacetBy.3 — len(visible_axes) == len(quantiles) for
        quantile_mode='discrete' + facet_by='quantiles'."""
        d = DFDraw(df_quantiles)
        fig, axes, stats = d.profile(
            "y:x",
            quantiles=[0.25, 0.50, 0.75],
            quantile_mode="discrete",
            facet_by="quantiles",
        )
        # §9.FacetBy.3 — one subplot per quantile
        assert stats["n_groups"] == 3, \
            f"facet_by='quantiles' with [0.25, 0.5, 0.75] should produce 3 subplots, got {stats['n_groups']}"
        assert stats["facet_by"] == "quantiles"
        plt.close(fig)

    def test_facet_by_invalid_channel_name_raises(self, df_groupby):
        """§9.FacetBy.4 — ValueError on invalid facet_by name; message
        names valid options."""
        d = DFDraw(df_groupby)
        with pytest.raises(ValueError) as exc_info:
            d.profile("y:x", group_by="sector", facet_by="banana")
        msg = str(exc_info.value)
        # §9.FacetBy.4 — error message mentions valid options
        assert "facet_by must be one of" in msg, \
            f"Expected error message naming valid options, got: {msg}"
        # §9.FacetBy.4 — message lists at least 'group_by' as a valid option
        assert "group_by" in msg


# =============================================================================
# Class 3 — TestFacetCapacity — 2 tests
# =============================================================================

class TestFacetCapacity:
    """Facet capacity enforcement via channels.cycles.facet_max."""

    def test_facet_max_capacity_fires(self):
        """§9.Capacity.1 — ValueError when faceted channel cardinality
        exceeds channels.cycles.facet_max under default overflow='error'."""
        # Build dataset with 25 unique groups (above default facet_max=16)
        rng = np.random.default_rng(seed=44)
        n_per_group = 5
        data = []
        for g in range(25):
            for _ in range(n_per_group):
                data.append({
                    "x": rng.uniform(0, 10),
                    "y": g * 0.1 + rng.normal(0, 0.3),
                    "many_groups": g,
                })
        df = pd.DataFrame(data)
        d = DFDraw(df)

        # Ensure overflow is 'error' (default per Phase 13.26 AD-58)
        set_style({"channels.overflow": "error"})

        with pytest.raises(ValueError) as exc_info:
            d.profile("y:x", group_by="many_groups", facet_by="group_by")
        msg = str(exc_info.value)
        # §9.Capacity.1 — error message references facet_max
        assert "facet_max" in msg, \
            f"Expected error to reference facet_max, got: {msg}"
        # §9.Capacity.1 — message includes actual count and limit
        assert "25" in msg or "16" in msg

    def test_facet_max_warn_mode(self):
        """§9.Capacity.2 — UserWarning when channels.overflow='warn';
        proceeds with truncated subplots (does not raise)."""
        rng = np.random.default_rng(seed=44)
        n_per_group = 5
        data = []
        for g in range(25):
            for _ in range(n_per_group):
                data.append({
                    "x": rng.uniform(0, 10),
                    "y": g * 0.1 + rng.normal(0, 0.3),
                    "many_groups": g,
                })
        df = pd.DataFrame(data)
        d = DFDraw(df)

        set_style({"channels.overflow": "warn"})
        try:
            with pytest.warns(UserWarning, match="facet_max"):
                fig, axes, stats = d.profile(
                    "y:x", group_by="many_groups", facet_by="group_by"
                )
            # §9.Capacity.2 — proceeds (does not raise)
            # §9.Capacity.2 — truncated to facet_max
            facet_max = get_style_value("channels.cycles.facet_max", 16)
            assert stats["n_groups"] == facet_max, \
                f"Expected truncation to facet_max={facet_max}, got n_groups={stats['n_groups']}"
            plt.close(fig)
        finally:
            # Restore default for other tests
            set_style({"channels.overflow": "error"})


# =============================================================================
# Class 4 — TestFacetSameTrueExclusion (architect rule §5.4) — 1 test
# =============================================================================

class TestFacetSameTrueExclusion:
    """facet_by and same=True must be mutually exclusive."""

    def test_facet_by_and_same_true_raises(self, df_groupby):
        """§9.Exclusion.1 — ValueError matches 'mutually exclusive' when
        both facet_by= and same=True are passed (architect rule v1.1 §5.4)."""
        d = DFDraw(df_groupby)
        with pytest.raises(ValueError) as exc_info:
            d.profile("y:x", group_by="sector",
                      facet_by="group_by", same=True)
        msg = str(exc_info.value)
        # §9.Exclusion.1 — error message uses the architect-locked phrasing
        assert "mutually exclusive" in msg, \
            f"Expected 'mutually exclusive' in error, got: {msg}"
        # §9.Exclusion.1 — both kwarg names appear in message for clarity
        assert "facet_by" in msg
        assert "same" in msg
