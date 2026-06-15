"""Phase 13.55.ADF — ADF dispatch audit + Phase 13.51/13.52 type-gap fix tests.

Test scaffold per `PHASE_13_55_ADF_DrawFiguresAudit_Proposal_v1_2.md` §4.

25 tests across 5 groups:
- Group 1 (T1-T7, T7.5): A-1/A-2/A-3 type coverage + A-10 scatter3d-in-spec
  clean-error limitation lock
- Group 2 (T8-T12, T-auto): A-6 kwarg regression locks via draw_figures +
  A-8 'auto' sentinel regression (literal and omitted, both surfaces)
- Group 3 (T13-T15): A-5 on_error behavior (new 'raise' default)
- Group 4 (T16-T21): routing-change regression (existing contracts preserved)
- Group 5 (T22-T23): A-7 adf.draw_batch on_error default (§11.4 Option A,
  architect-ratified)

The Option B fix routes adf.draw() and adf.draw_figures() through
DFDraw.draw() with ADF-side 'auto' pre-resolution via _resolve_plot_type
(A-8, architect decision "OK. B", 2026-06-10).

Architect decisions (v1.2 §0):
- §11 items 1-4 ratified 2026-06-10
- §11 item 5: draw_fit_summary keeps on_error='skip' (documented exception)
- §11 item 6: docs/ARCHITECT_DECISIONS.md created (AD-1/13.55.ADF)

References:
- PHASE_13_55_ADF_DrawFiguresAudit_Proposal_v1_2.md
- Sonnet1 main-reviewer summary v1.2 (8/8, [OK])
- Sister phase: PHASE_13_55_DF (landed; drawer.py:7235/7369/7580)
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

def _make_basic_df(n=600, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "t": np.linspace(0, 7200, n),          # epoch-style seconds for time axis
        "sec": rng.integers(0, 4, n).astype(np.int64),
    })


def _make_outlier_df(n=600, outlier_pct=0.10, outlier_scale=100.0, seed=42):
    """Outlier fixture per Phase 13.51 T9b: 10% rows at 100x so median != mean."""
    df = _make_basic_df(n=n, seed=seed)
    k = int(n * outlier_pct)
    df.loc[df.index[:k], "y"] = df.loc[df.index[:k], "y"] * outlier_scale + outlier_scale
    return df


@pytest.fixture
def adf():
    return AliasDataFrame(_make_basic_df())


@pytest.fixture
def adf_outlier():
    return AliasDataFrame(_make_outlier_df())


@pytest.fixture
def adf_subframe():
    """Parent ADF with registered subframe for T19."""
    rng = np.random.default_rng(7)
    parent = pd.DataFrame({
        "event_id": np.arange(50, dtype=np.int64),
        "x": rng.standard_normal(50),
    })
    child = pd.DataFrame({
        "event_id": np.arange(50, dtype=np.int64),
        "val": rng.standard_normal(50),
    })
    a = AliasDataFrame(parent)
    a.register_subframe("S", AliasDataFrame(child), index_columns="event_id")
    return a


def _figspec(plots, name="f"):
    return [{"name": name, "plots": plots}]


def _no_error_placeholder(ax):
    """C-3 strengthened assertion: no '[ERROR]' title AND no 'Error:' text."""
    assert not ax.get_title().startswith("[ERROR]"), (
        f"axis title is an error placeholder: {ax.get_title()!r}")
    for txt in ax.texts:
        assert "Error:" not in txt.get_text(), (
            f"axis contains error text: {txt.get_text()[:80]!r}")


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# =============================================================================
# Group 1 — A-1/A-2/A-3 type coverage (T1-T7, T7.5)
# =============================================================================

@pytest.mark.invariance
class TestGroup1TypeCoverage:

    def test_T1_draw_overlay_string(self, adf):
        """T1 (A-1): adf.draw with overlay 'hist2d+profile' returns valid tuple."""
        fig, ax, stats = adf.draw("y:x", type="hist2d+profile", bins=20)
        assert fig is not None and ax is not None

    def test_T2_draw_histo_alias(self, adf):
        """T2 (A-3): adf.draw with ROOT-convention alias 'histo' works."""
        fig, ax, stats = adf.draw("x", type="histo", bins=30)
        assert fig is not None
        assert stats is not None and "mean" in stats

    def test_T3_draw_profile2d_no_regression(self, adf):
        """T3 (A-4 baseline): profile2d still works via new route."""
        fig, ax, stats = adf.draw("z:y:x", type="profile2d", bins=8)
        assert fig is not None

    def test_T4_draw_scatter3d_no_regression(self, adf):
        """T4 (A-4 baseline): scatter3d single-spec creates its own 3D figure."""
        fig, ax, stats = adf.draw("z:y:x", type="scatter3d")
        assert fig is not None
        assert getattr(ax, "name", "") == "3d"

    def test_T5_figures_overlay_in_spec(self, adf):
        """T5 (A-2): overlay string in draw_figures spec renders a real plot."""
        res = adf.draw_figures(
            _figspec([{"expr": "y:x", "type": "hist2d+profile", "bins": 20}]),
            verbose=False)
        ax0 = res["f"]["axes"][0]
        _no_error_placeholder(ax0)
        assert res["f"]["stats"][0] is not None

    def test_T6_figures_histo_alias_in_spec(self, adf):
        """T6 (A-3 via draw_figures): 'histo' alias in spec renders a real plot."""
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "histo", "bins": 30}]),
            verbose=False)
        ax0 = res["f"]["axes"][0]
        _no_error_placeholder(ax0)
        assert res["f"]["stats"][0] is not None

    def test_T7_figures_profile2d_in_spec(self, adf):
        """T7 (AMENDED in PHASE_13_56_ADF): profile2d inside a draw_figures
        spec is now guarded. The original assertion locked the E-3
        silent-defect state — audit AUDIT_ADF_GRAPHICS_2026_06 proved the
        dashboard panel rendered EMPTY while stats looked valid (the
        _no_error_placeholder + stats-not-None pass was the silent wrong
        output itself). The supported surfaces are adf.draw and
        adf.draw_batch (locked by T3/T-G6b); the figures-side contract is
        now the clean guard error (T-G1a/b own the detailed assertions)."""
        with pytest.raises(ValueError, match="profile2d.*draw_figures"):
            adf.draw_figures(
                _figspec([{"expr": "z:y:x", "type": "profile2d", "bins": 8}]),
                verbose=False)

    def test_T7_5_figures_scatter3d_clean_error(self, adf):
        """T7.5 (A-10): scatter3d in draw_figures spec -> clean actionable error.

        Limitation lock, not a fix: draw_figures builds a 2D grid; scatter3d
        needs a 3D-projection axis. Both on_error modes give a clean message.
        """
        spec = _figspec([{"expr": "z:y:x", "type": "scatter3d"}])
        # raise mode (default): clean ValueError naming both surfaces
        with pytest.raises(ValueError, match="scatter3d.*draw_figures"):
            adf.draw_figures(spec, verbose=False)
        # skip mode: clean message in the placeholder
        res = adf.draw_figures(spec, on_error="skip", verbose=False)
        ax0 = res["f"]["axes"][0]
        assert ax0.get_title().startswith("[ERROR]")
        joined = " ".join(t.get_text() for t in ax0.texts)
        assert "scatter3d" in joined and "draw_figures" in joined


# =============================================================================
# Group 2 — A-6 kwarg regression locks via draw_figures + A-8 T-auto
# =============================================================================

@pytest.mark.invariance
class TestGroup2KwargLocks:

    def test_T8_central_median_via_figures(self, adf_outlier):
        """T8 (13.51 S-2): central='median' != mean on outlier fixture (A≡B)."""
        res_mean = adf_outlier.draw_figures(
            _figspec([{"expr": "y:x", "type": "profile", "bins": 10}]),
            verbose=False)
        res_med = adf_outlier.draw_figures(
            _figspec([{"expr": "y:x", "type": "profile", "bins": 10,
                       "central": "median"}]),
            verbose=False)
        s_mean, s_med = res_mean["f"]["stats"][0], res_med["f"]["stats"][0]
        assert s_mean is not None and s_med is not None
        # Assert via rendered profile line (sister-phase T7 pattern — per-bin
        # central values are not in the stats dict).
        ym = res_mean["f"]["axes"][0].lines[0].get_ydata()
        yq = res_med["f"]["axes"][0].lines[0].get_ydata()
        ym, yq = np.asarray(ym, float), np.asarray(yq, float)
        ok = np.isfinite(ym) & np.isfinite(yq)
        assert ok.any()
        assert not np.allclose(ym[ok], yq[ok]), (
            "median central equals mean central on outlier fixture")

    def test_T9_time_format_via_figures(self, adf):
        """T9 (13.51 S-8): time_format='%H:%M' produces HH:MM tick labels."""
        res = adf.draw_figures(
            _figspec([{"expr": "y:t", "type": "hist2d", "bins": 12,
                       "time_format": "%H:%M"}]),
            verbose=False)
        ax0 = res["f"]["axes"][0]
        _no_error_placeholder(ax0)
        ax0.figure.canvas.draw()
        labels = [t.get_text() for t in ax0.get_xticklabels() if t.get_text()]
        assert any(":" in l and len(l) == 5 for l in labels), (
            f"no HH:MM labels found: {labels}")

    def test_T10_auto_title_profile_via_figures(self, adf):
        """T10 (13.54, C-2: profile not scatter): auto_title yields non-empty title."""
        res = adf.draw_figures(
            _figspec([{"expr": "y:x", "type": "profile", "bins": 10,
                       "auto_title": True}]),
            verbose=False)
        ax0 = res["f"]["axes"][0]
        _no_error_placeholder(ax0)
        assert ax0.get_title().strip() != ""

    def test_T11_fit_gauss_via_figures(self, adf):
        """T11 (13.42): fit='gauss' params present in stats."""
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "hist", "bins": 40,
                       "fit": "gauss"}]),
            verbose=False)
        s = res["f"]["stats"][0]
        assert s is not None and "fit" in s and s["fit"] is not None

    def test_T12_selection_vector_via_figures(self, adf):
        """T12 (13.27): selection_vector renders two curves."""
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "hist", "bins": 30,
                       "selection_vector": ["x>0", "x<0"]}]),
            verbose=False)
        ax0 = res["f"]["axes"][0]
        _no_error_placeholder(ax0)
        n_artists = len(ax0.lines) + len(ax0.patches) + len(ax0.containers)
        assert n_artists >= 2

    def test_Tauto_sentinel_regression(self, adf):
        """T-auto (C-5, strengthened per A-8): 'auto' literal AND omitted type
        dispatch correctly on both surfaces — no 'Unknown plot type auto'."""
        # adf.draw — omitted (defaults to 'auto') and literal
        fig, ax, stats = adf.draw("x", bins=20)
        assert "mean" in stats                       # hist stats
        fig, ax, stats = adf.draw("x", type="auto", bins=20)
        assert "mean" in stats
        fig, ax, stats = adf.draw("y:x", type="auto")
        assert stats is not None                     # scatter
        # draw_figures — omitted and literal in spec
        res = adf.draw_figures(
            _figspec([{"expr": "x", "bins": 20},
                      {"expr": "y:x"},
                      {"expr": "x", "type": "auto", "bins": 20},
                      {"expr": "y:x", "type": "auto"}]),
            verbose=False)
        for i, ax_i in enumerate(res["f"]["axes"]):
            _no_error_placeholder(ax_i)
            assert res["f"]["stats"][i] is not None


# =============================================================================
# Group 3 — A-5 on_error behavior (T13-T15)
# =============================================================================

@pytest.mark.invariance
class TestGroup3OnError:

    def test_T13_invalid_type_default_raises(self, adf):
        """T13 (A-5): invalid type raises by default (new on_error='raise')."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            adf.draw_figures(
                _figspec([{"expr": "x", "type": "not_a_plot_type"}]),
                verbose=False)

    def test_T14_invalid_type_explicit_skip(self, adf):
        """T14: explicit on_error='skip' renders error placeholder, no raise."""
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "not_a_plot_type"}]),
            on_error="skip", verbose=False)
        ax0 = res["f"]["axes"][0]
        assert ax0.get_title().startswith("[ERROR]")
        assert res["f"]["stats"][0] is None

    def test_T15_valid_type_raise_regression(self, adf):
        """T15: valid spec under on_error='raise' succeeds and returns stats."""
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "hist", "bins": 30}]),
            on_error="raise", verbose=False)
        assert res["f"]["stats"][0] is not None


# =============================================================================
# Group 4 — Routing-change regression (T16-T21)
# =============================================================================

@pytest.mark.invariance
class TestGroup4RoutingRegression:

    def test_T16_draw_return_shape(self, adf):
        """T16: adf.draw returns (fig, ax, stats) 3-tuple, same as pre-fix."""
        result = adf.draw("x", bins=20)
        assert isinstance(result, tuple) and len(result) == 3
        fig, ax, stats = result
        assert hasattr(fig, "savefig") and isinstance(stats, dict)

    def test_T17_figures_return_shape(self, adf):
        """T17 (C-1): per-figure dict has 'stats' key (NOT 'stats_list')."""
        res = adf.draw_figures(_figspec([{"expr": "x", "bins": 20}]),
                               verbose=False)
        entry = res["f"]
        assert "stats" in entry and "stats_list" not in entry
        assert {"fig", "axes", "stats"} <= set(entry.keys())
        assert isinstance(entry["stats"], list)

    def test_T18_facet_by_lazy_alias_regression_lock(self, adf):
        """T18 (regression lock for BUG_20260609, NOT new fix — C-6):
        list-valued facet_by with an unmaterialized alias survives routing."""
        adf.add_alias("side_type", "(sec >= 2)*1")
        adf.add_alias("lazy_q", "(x > 0)*1")
        fig, ax, stats = adf.draw("y:x", type="profile", bins=8,
                                  facet_by=["side_type", "lazy_q"])
        assert fig is not None

    def test_T19_subframe_rewrite_regression_lock(self, adf_subframe):
        """T19 (regression lock, NOT new fix — C-6): Subframe.col resolution."""
        fig, ax, stats = adf_subframe.draw("S.val:x", type="profile", bins=5)
        assert fig is not None and stats is not None

    def test_T20_data_source_preserved(self, adf):
        """T20: _data_source survives the DFDraw.draw() route — schema axis
        titles still resolve."""
        adf.set_axis_title("x", "X title (a.u.)")
        fig, ax, stats = adf.draw("x", bins=20)
        assert ax.get_xlabel() == "X title (a.u.)"

    def test_T21_cleanup_after_dispatch(self, adf):
        """T21: dematerialize cleanup runs after routed dispatch; aliases we
        added for the plot are dropped again."""
        adf.add_alias("tmp_alias", "x * 2")
        before = set(adf.df.columns)
        fig, ax, stats = adf.draw("tmp_alias", bins=20, lazy=True,
                                  keep_materialized=False)
        after = set(adf.df.columns)
        assert "tmp_alias" not in after
        assert before == after


# =============================================================================
# Group 5 — A-7 adf.draw_batch on_error (T22-T23) — §11.4 Option A ratified
# =============================================================================

@pytest.mark.invariance
class TestGroup5DrawBatchOnError:

    def test_T22_batch_invalid_default_raises(self, adf):
        """T22 (A-7): invalid spec raises under new draw_batch default."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            adf.draw_batch({"bad": {"expr": "x", "type": "not_a_plot_type"}},
                           verbose=False)

    def test_T23_batch_explicit_skip(self, adf):
        """T23: explicit on_error='skip' records error, does not raise."""
        results = adf.draw_batch(
            {"bad": {"expr": "x", "type": "not_a_plot_type"},
             "good": {"expr": "x", "type": "hist", "bins": 20}},
            on_error="skip", verbose=False)
        assert "_errors" in results and "bad" in results["_errors"]
        assert "good" in results and results["good"]["stats"] is not None

# =============================================================================
# Group 6 — Post-gallery regression fix (fig08 class): 3-var profile promotion
# =============================================================================

@pytest.mark.invariance
class TestGroup6ProfilePromotion:
    """PHASE_13_55_ADF post-gallery fix. Gallery fig08 regressed because
    DFDraw.profile() promotes 3-variable expressions to profile2d but
    DFDraw.draw(type='profile') does not. ADF promotes before routing.
    FM#12: T3b reproduces the exact fig08 public call form (selection +
    bins + auto_title through adf.draw with type='profile')."""

    def test_T3b_draw_3var_profile_promotion(self, adf):
        """T3b (AMENDED in PHASE_13_57_DF gate — third 13.55 amendment):
        adf.draw 3-var expr with type='profile' → profile2d (fig08 form).

        History: the original form passed only because dfdraw's profile2d
        early-dispatch silently DROPPED selection= pre-13.57 (K-4 root
        cause; FX-3 — gallery fig08/fig35 rendered unfiltered). The test
        was unknowingly locking that F-B silent-drop state (same class as
        T7/T7b). Amended per the F-A contract (plain alias selection needs
        lazy=True) with a REAL cut and an EFFECT assertion so a dropped
        selection can never pass again."""
        adf.add_alias("good", "(x > 0)*1")          # real cut: ~half (x is standard normal)
        n_full = len(adf.df)
        fig, ax, stats = adf.draw("z:y:x", selection="good>0", type="profile",
                                  bins=8, auto_title=True, lazy=True)
        assert fig is not None and stats is not None
        assert stats["n"] < n_full, (
            f"selection had no effect (n={stats['n']} == full {n_full}) — "
            f"silent-drop regression (FX-3 class)")

    def test_T3c_draw_vector_expr_profile_not_promoted(self, adf):
        """T3c: vector expr '[a,b]:x' (1 top-level colon) stays 2-var profile."""
        fig, ax, stats = adf.draw("[y, z]:x", type="profile", bins=8)
        assert fig is not None

    def test_T7b_figures_3var_profile_promotion(self, adf):
        """T7b (AMENDED in PHASE_13_56_ADF): the 3-var promotion inside a
        draw_figures spec now lands on the profile2d guard (binding order
        §3.4) instead of the E-3 empty panel the original assertion was
        unknowingly locking. Promotion itself stays locked on the working
        surfaces by T3b (adf.draw) and T-G6b (adf.draw_batch)."""
        with pytest.raises(ValueError, match="profile2d.*draw_figures"):
            adf.draw_figures(
                _figspec([{"expr": "z:y:x", "type": "profile", "bins": 8}]),
                verbose=False)

    def test_T3d_colon_counter_units(self, adf):
        """T3d: bracket-aware counter unit checks."""
        c = adf._top_level_colon_count
        assert c("x") == 0
        assert c("y:x") == 1
        assert c("z:y:x") == 2
        assert c("[a, b]:x") == 1
        assert c("f(a, b):g(c, d)") == 1
        assert c("[a:b]:x") == 1  # colon inside brackets not counted
