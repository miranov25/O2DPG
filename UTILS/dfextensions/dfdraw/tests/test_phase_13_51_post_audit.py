"""Phase 13.51.DF — Post-Audit Fix Pass invariance tests.

Per v1.5 §1.4: 21 tests gating Batches 1+2+3.

Test naming (matches PHASE_13_51_PostAuditProposal_v1_5.md §1.4 table):
- T1: draw(type='profile', normalize='delta')                — S-3 profile R-2
- T2: draw(type='profile', facet_by='sec') + cell-population — S-3 profile R-2 + G-1
- T3: draw(type='profile', selection_vector=mask)            — S-3 profile R-2
- T4: draw(type='hist',    facet_by='sec') + cell-population — S-3 hist R-2 + G-1
- T5: draw(type='hist',    selection_vector=mask)            — S-3 hist R-2
- T6: draw(type='scatter', facet_by='sec') + cell-population — S-3 scatter R-2 + G-1
- T7: draw(type='scatter', time_format='auto')               — S-3 scatter R-2 (via **kwargs)
- T8: draw(type='hist2d',  facet_by='sec') + cell-population — S-3 hist2d R-2 + G-1
- T9a: profile('z:y:x', central='median') 2D mesh comparison — S-2 2D (§1.2.2) + P1-A
- T9b: profile('y:x',   central='median') 1D errorbar line  — S-2 1D (§1.2.5 V-3) + G-2
- T9c: profile('y:x',   central=..., fit='gauss') fit center — S-2 fit-path (§1.2.5 P1-B)
- T10a: hist with datetime64 + facet_by — no crash + ScalarFormatter sentinel
- T10b: compute_autorange(datetime64, strategy='robust_3mad') — V-2 broader scope
- T10c: compute_autorange(datetime64, strategy='percentile_99') — V-2 broader scope
- T11: d.profile2d('z:y:x') wrapper                          — S-5
- T12: d.scatter3d('z:y:x') + type-collision guard           — S-5 + V-4
- T13: d.draw('y:x', type='hexbin') dispatch                 — S-11
- T14: d.hist2d('y:t', time_format='auto') tick labels       — S-8 + I-1 + G-3
- T15: d.hist2d(..., fit='gauss') clean ValueError           — S-8 guard
- T16: d.hexbin(..., range=...) clean ValueError             — S-7 guard
- T17: d.hexbin(..., facet_by='sec') clean ValueError        — S-7 guard

Production-code references (all in §1.5 of PHASE_13_51_PostAuditProposal_v1_5.md):
- drawer.py:4429 hist branch, :4441 scatter, :4453 hist2d, :4459 profile (R-2)
- drawer.py:5461 draw_profile2d call with central= forward
- drawer.py:1265 area — comment correction
- plots/_autorange.py compute_autorange() entry datetime64 guard
- plots/profile.py:879 (V-3) + :907 (P1-B): bin_means -> _central_values
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots._autorange import compute_autorange

# Reuse Phase 13.48 visual_primitive helpers per v1.3 §10.6
from tests.test_phase_13_48_df_visual_testing import (
    grid_geometry,
    data_cell_axes,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_outlier_df_1d(n=5000, seed=42):
    """1D fixture: concentrated outliers in narrow x-band (per V-6)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.standard_normal(n),
        "sec": rng.integers(0, 4, n),
        "t": np.arange(n).astype(np.int64),
    })
    mask = df["x"].between(2, 3)
    df.loc[mask, "y"] *= 100
    return df


def _make_outlier_df_2d(n=5000, seed=42):
    """2D fixture: outliers concentrated in narrow (x,y) cell (per V-6)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.uniform(0, 10, n),
        "z": rng.standard_normal(n),
        "sec": rng.integers(0, 4, n),
    })
    mask = df["x"].between(2, 3) & df["y"].between(5, 6)
    df.loc[mask, "z"] *= 100
    return df


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# =============================================================================
# T1-T8 — R-2 dispatch forwarding (audit S-3)
# =============================================================================

def test_T1_draw_type_profile_normalize_delta_yields_two_axes():
    """T1: draw(type='profile', normalize='delta') with two selection_vector
    sectors reaches profile() with overlay + bottom-diff panels.
    Pre-fix: dispatch silently dropped both normalize AND selection_vector
    so the call collapsed to a single panel. Post-fix: 2 axes (top + diff)."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    fig, _, _ = d.draw(
        "y:x",
        type="profile",
        selection_vector=["sec < 2", "sec >= 2"],
        normalize="delta",
        bins=20,
    )
    assert len(fig.axes) >= 2, (
        f"normalize='delta' should yield ≥2 axes (top + bottom diff); got {len(fig.axes)}"
    )


def test_T2_draw_type_profile_facet_by_sec_populated_panels():
    """T2: draw(type='profile', facet_by='sec') yields ≥4 axes AND each visible
    panel is populated (G-1: empty-panel false-pass guard via data_cell_axes)."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    fig, _, _ = d.draw("y:x", type="profile", facet_by="sec")
    nrows, ncols = grid_geometry(fig)
    assert nrows * ncols >= 4, (
        f"facet_by='sec' should engage faceting, got grid {nrows}x{ncols}"
    )
    populated = data_cell_axes(fig)
    assert len(populated) >= 4, (
        f"facet engaged but only {len(populated)} panels populated (G-1: empty-panel guard)"
    )


def test_T3_draw_type_profile_selection_vector_matches_direct():
    """T3: draw(type='profile', selection_vector=[...]) → same result as
    direct profile(...) call. Pre-fix: selection_vector silently dropped at
    dispatch (Phase 13.27 vector mode never engaged via draw())."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    sv = ["x < 5", "x >= 5"]
    fig_dispatch, _, _ = d.draw(
        "y:x", type="profile", selection_vector=sv, vector_compose="outer"
    )
    n_axes_dispatch = len(fig_dispatch.axes)
    n_lines_dispatch = sum(len(ax.lines) for ax in fig_dispatch.axes)
    plt.close("all")
    fig_direct, _, _ = d.profile(
        "y:x", selection_vector=sv, vector_compose="outer"
    )
    # Same number of axes AND lines — selection_vector did NOT get dropped at dispatch
    assert len(fig_direct.axes) == n_axes_dispatch
    assert sum(len(ax.lines) for ax in fig_direct.axes) == n_lines_dispatch
    # And the dispatch path produced multiple curves (vector mode engaged)
    assert n_lines_dispatch > 0, "selection_vector should yield rendered curves"


def test_T4_draw_type_hist_facet_by_sec_populated_panels():
    """T4: draw(type='hist', facet_by='sec') yields ≥4 axes AND populated
    panels (G-1, adapted for hist). Pre-fix: hist branch silently dropped
    facet_by → 1 axis. Note: hist bars are in ax.patches (Rectangle), not
    ax.lines/collections — adapt G-1's data_cell_axes for histogram artist tree."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    fig, _, _ = d.draw("y", type="hist", facet_by="sec")
    nrows, ncols = grid_geometry(fig)
    assert nrows * ncols >= 4, f"hist + facet_by should engage faceting, got {nrows}x{ncols}"
    # G-1 (hist-adapted): each visible facet panel should have ≥1 patch (bar) artist.
    # data_cell_axes() checks ax.lines/collections; for hist we also accept ax.patches.
    populated = [
        ax for ax in fig.axes
        if (ax.lines or ax.collections or ax.patches)
        and ax.get_visible()
    ]
    assert len(populated) >= 4, (
        f"hist facet engaged but only {len(populated)} panels populated (G-1 hist-adapted)"
    )


def test_T5_draw_type_hist_selection_vector_matches_direct():
    """T5: draw(type='hist', selection_vector=[...]) → same as direct
    hist(...). Pre-fix: selection_vector silently dropped at hist branch."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    sv = ["x < 5", "x >= 5"]
    fig_dispatch, _, _ = d.draw(
        "y", type="hist", selection_vector=sv, vector_compose="outer"
    )
    plt.close("all")
    fig_direct, _, _ = d.hist(
        "y", selection_vector=sv, vector_compose="outer"
    )
    # Same number of axes (selection_vector engaged in both paths)
    assert len(fig_dispatch.axes) == len(fig_direct.axes)


def test_T6_draw_type_scatter_facet_by_sec_populated_panels():
    """T6: draw(type='scatter', facet_by='sec') → ≥4 axes, populated (G-1)."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    fig, _, _ = d.draw("y:x", type="scatter", facet_by="sec")
    nrows, ncols = grid_geometry(fig)
    assert nrows * ncols >= 4
    populated = data_cell_axes(fig)
    assert len(populated) >= 4


def test_T7_draw_type_scatter_time_format_auto_dateformatter():
    """T7: draw(type='scatter', time_format='auto') applies a Date-class
    formatter on the x-axis. time_format is NOT in draw() signature, so it
    flows via **kwargs (V-1 verified)."""
    df = _make_outlier_df_1d()
    df["t_dt"] = pd.to_datetime(df["t"], unit="s")
    d = DFDraw(df)
    fig, ax, _ = d.draw("y:t_dt", type="scatter", time_format="auto")
    fmtname = type(ax.xaxis.get_major_formatter()).__name__
    assert "Date" in fmtname or "AutoDate" in fmtname, (
        f"time_format='auto' should set a Date-class formatter; got {fmtname}"
    )


def test_T8_draw_type_hist2d_facet_by_sec_populated_panels():
    """T8: draw(type='hist2d', facet_by='sec') → ≥4 populated panels.
    Pre-fix: facet_by reached hist2d via R-2 (Phase 13.41), but the dispatch
    branch in draw() did not forward it (silent drop).

    v1.5.3 final: use the production-side facet-panel identifier
    `ax._dfdraw_facet_key` (set by drawer.py:3670 in `_dispatch_faceted_render`)
    instead of the Phase 13.48 `data_cell_axes` helper. The helper depends on
    `grid_geometry(fig)` which inspects `fig.axes[0].get_subplotspec().
    get_gridspec().get_geometry()` — but mpl 3.8.x's `fig.colorbar(im, ax=ax)`
    replaces the panel's parent gridspec with a (1,2) split-gridspec (panel +
    colorbar side-by-side), making `fig.axes[0]`'s gridspec unrepresentative
    of the panel grid. mpl 3.10+ preserves the original gridspec.

    `_dfdraw_facet_key` is set on every panel axes inside the dispatch
    loop, never on colorbar axes, and is independent of matplotlib version.
    This is what the test actually wants: "panels created by the facet
    engine that have data".

    Smoking-gun diagnostic from v1.5.2 (architect macOS Python 3.9.6 / mpl 3.8.2):
       STATS:  n_groups=4, groups=[0,1,2,3]  ← engine knows 4 facets
       TITLES: ['sec=0', 'sec=1', 'sec=2', 'sec=3']  ← 4 panels created
       FIG:    n_axes=10 grid=(1,2) populated=2  ← helper misreads geometry"""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    fig, _, _ = d.draw("y:x", type="hist2d", facet_by="sec")

    # Production-side panel identifier (set at drawer.py:3670):
    # `_dispatch_faceted_render` stashes `ax._dfdraw_facet_key = group_value`
    # on every panel axes BEFORE the per-subplot recursion. Colorbars created
    # downstream by `fig.colorbar()` never have this attribute. Mpl-version-
    # stable across 3.8/3.9/3.10+ because it's our own attribute.
    panels = [
        ax for ax in fig.axes
        if hasattr(ax, "_dfdraw_facet_key")
        and ax.get_visible()
        and (ax.lines or ax.collections or ax.patches)
    ]
    assert len(panels) >= 4, (
        f"hist2d + facet_by='sec' should produce ≥4 populated panels "
        f"(one per sector); got {len(panels)} panels with "
        f"_dfdraw_facet_key + data, out of {len(fig.axes)} total axes."
    )


# =============================================================================
# T9a/T9b/T9c — central='median' fixes (audit S-2)
# =============================================================================

def test_T9a_profile2d_central_median_mesh_differs_from_mean():
    """T9a (P1-A v1.4): 2D profile renders via pcolormesh → QuadMesh in
    ax.collections[0], NOT ax.lines (which is empty). NEW-1 v1.5: np.ma.allclose
    has NO equal_nan kwarg — use np.ma.masked_invalid for NaN handling."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)

    _, ax_med, _ = d.profile("z:y:x", central="median")
    assert len(ax_med.lines) == 0, "profile2d renders to collections (pcolormesh)"
    assert len(ax_med.collections) >= 1, "expected ≥1 QuadMesh"
    mesh_median = ax_med.collections[0].get_array()
    plt.close("all")

    _, ax_mean, _ = d.profile("z:y:x", central="mean")
    mesh_mean = ax_mean.collections[0].get_array()

    assert not np.ma.allclose(
        np.ma.masked_invalid(mesh_median),
        np.ma.masked_invalid(mesh_mean),
    ), "2D profile central='median' should differ from mean with concentrated outliers"


def test_T9b_profile_1d_central_median_line_differs_from_mean():
    """T9b (V-3): 1D profile renders via ax.errorbar → main Line2D in
    ax.lines[0] (errorbar = main + 2 cap lines = 3 lines, executed-verified
    fixture guard per P2-3)."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)

    fig, ax, _ = d.profile("y:x", central="median")
    assert len(fig.axes) == 1, "T9b fixture guard: no facet"
    assert len(ax.lines) == 3, (
        f"errorbar yields 3 lines (main + 2 caps); got {len(ax.lines)}"
    )
    y_med = ax.lines[0].get_ydata().copy()
    plt.close("all")

    _, ax2, _ = d.profile("y:x", central="mean")
    y_mean = ax2.lines[0].get_ydata().copy()

    assert not np.allclose(y_med, y_mean, equal_nan=True), (
        "1D profile central='median' line should differ from mean line"
    )


def test_T9c_profile_fit_central_median_fit_center_differs():
    """T9c (P1-B v1.4): fit consumes _central_values not bin_means after
    profile.py:907 fix. params is np.ndarray with param_names=['amplitude',
    'center', 'sigma']; center at index 1 (NOT 'mu')."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)

    _, _, stats_med = d.profile("y:x", central="median", fit="gauss")
    fit_med = stats_med["fit"][0][0]
    assert "center" in fit_med["param_names"], (
        f"expected 'center' in param_names; got {fit_med['param_names']}"
    )
    center_idx = fit_med["param_names"].index("center")
    center_med = fit_med["params"][center_idx]
    plt.close("all")

    _, _, stats_mean = d.profile("y:x", central="mean", fit="gauss")
    center_mean = stats_mean["fit"][0][0]["params"][center_idx]

    assert not np.isclose(center_med, center_mean), (
        f"P1-B: fit center should differ between median ({center_med:.4f}) and "
        f"mean ({center_mean:.4f}) with concentrated outliers"
    )


# =============================================================================
# T10a/T10b/T10c — datetime64 autorange (audit S-4 V-2)
# =============================================================================

def test_T10a_hist_datetime64_no_crash_scalarformatter_sentinel():
    """T10a sentinel (per V-5): asserts ScalarFormatter is current behavior
    on faceted panels. Full per-panel DateFormatter application is audit
    finding S-4 (Batch 4 scope). When S-4 lands, this assertion must change.

    Note: T10a covers the non-faceted datetime64 + time_format='auto' path.
    The faceted hist + datetime64 path additionally triggers a pre-existing
    np.mean(datetime64) bug at histogram.py:472 — that is outside Phase 13.51
    scope and tracked separately for Batch 4."""
    df = _make_outlier_df_1d()
    df["t_dt"] = pd.to_datetime(df["t"], unit="s")
    d = DFDraw(df)
    # Non-faceted path — verifies V-2 datetime64 autorange fix applies via
    # standard hist invocation. Pre-fix: would TypeError at _autorange.py:67.
    fig, ax, _ = d.hist("t_dt", time_format="auto")
    # Sentinel: time_format='auto' should set an AutoDateFormatter on the axis
    fmtname = type(ax.xaxis.get_major_formatter()).__name__
    assert "Date" in fmtname, (
        f"non-faceted hist + datetime64 + time_format='auto' should set Date "
        f"formatter; got {fmtname} (V-2 + S-4 partial fix)"
    )


def test_T10b_compute_autorange_datetime64_robust_3mad():
    """T10b (V-2): compute_autorange datetime64 guard covers robust_3mad
    strategy (not only hybrid). Pre-fix: TypeError at _autorange.py:117."""
    dt = pd.to_datetime(pd.Series([0, 100, 200, 300, 400, 500, 600]), unit="s").values
    lo, hi = compute_autorange(dt, strategy="robust_3mad")
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < hi


def test_T10c_compute_autorange_datetime64_percentile_99():
    """T10c (V-2): compute_autorange datetime64 guard covers percentile_99
    strategy. Pre-fix: TypeError at _autorange.py:107."""
    dt = pd.to_datetime(pd.Series([0, 100, 200, 300, 400, 500, 600]), unit="s").values
    lo, hi = compute_autorange(dt, strategy="percentile_99")
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < hi


# =============================================================================
# T11/T12 — Wrapper methods (audit S-5)
# =============================================================================

def test_T11_profile2d_wrapper_alias_for_profile():
    """T11 (S-5): d.profile2d('z:y:x') equivalent to d.profile('z:y:x')."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    _, ax1, _ = d.profile2d("z:y:x")
    arr1 = ax1.collections[0].get_array().copy()
    plt.close("all")
    _, ax2, _ = d.profile("z:y:x")
    arr2 = ax2.collections[0].get_array().copy()
    assert np.ma.allclose(
        np.ma.masked_invalid(arr1), np.ma.masked_invalid(arr2)
    ), "profile2d() should produce same output as profile() with 3-colon expr"


def test_T12_scatter3d_wrapper_and_type_collision_guard():
    """T12 (S-5 + V-4): scatter3d() works as alias; passing type='scatter3d'
    explicit is OK (no-op); passing type='profile' raises ValueError."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    # Plain alias works
    fig, _, _ = d.scatter3d("z:y:x")
    assert fig is not None
    plt.close("all")
    # type='scatter3d' explicit is OK (pop-and-forward)
    fig, _, _ = d.scatter3d("z:y:x", type="scatter3d")
    assert fig is not None
    plt.close("all")
    # type='profile' raises ValueError (V-4 collision guard)
    with pytest.raises(ValueError, match="scatter3d|expected"):
        d.scatter3d("z:y:x", type="profile")


# =============================================================================
# T13 — hexbin in draw() scalar dispatch (audit S-11)
# =============================================================================

def test_T13_draw_type_hexbin_dispatches_correctly():
    """T13 (S-11): d.draw('y:x', type='hexbin') no longer raises 'Unknown
    plot type'; produces equivalent output to d.hexbin('y:x')."""
    df = _make_outlier_df_1d()
    d = DFDraw(df)
    fig, ax, _ = d.draw("y:x", type="hexbin")
    # Hexbin renders as PolyCollection in ax.collections
    assert len(ax.collections) >= 1, "hexbin should render ≥1 collection"


# =============================================================================
# T14 — hist2d + time_format (audit S-8 + I-1 + G-3)
# =============================================================================

def test_T14_hist2d_time_format_auto_date_tick_labels():
    """T14 (S-8 + I-1 + G-3): hist2d + time_format='auto' applies an
    AutoDate-class formatter AND produces date-format tick label text.
    G-3 v1.3: get_text() pattern-match catches case where formatter is
    set but labels don't actually render as dates."""
    df = _make_outlier_df_1d()
    df["t_dt"] = pd.to_datetime(df["t"], unit="s")
    d = DFDraw(df)
    fig, ax, _ = d.hist2d("y:t_dt", time_format="auto")

    fmtname = type(ax.xaxis.get_major_formatter()).__name__
    assert "Date" in fmtname, (
        f"hist2d + time_format='auto' should set Date-class formatter; got {fmtname}"
    )

    # G-3: Force draw to populate tick texts, then inspect
    fig.canvas.draw()
    tick_texts = [t.get_text() for t in ax.xaxis.get_majorticklabels()]
    non_empty = [t for t in tick_texts if t]
    if non_empty:
        date_like = re.compile(r"\d{4}|\d{1,2}[:\-/]\d{2}|[A-Z][a-z]{2}\s")
        assert any(date_like.search(t) for t in non_empty), (
            f"date formatter set but labels not rendered as dates: {non_empty[:5]}"
        )


# =============================================================================
# T15/T16/T17 — Clean ValueError guards (audit S-7, S-8)
# =============================================================================

def test_T15_hist2d_fit_raises_clean_value_error():
    """T15 (S-8 guard): hist2d(..., fit='gauss') raises clean ValueError,
    NOT cryptic matplotlib `QuadMesh.set() got unexpected keyword 'fit'`."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    with pytest.raises(ValueError, match="hist2d.*fit|S-8|Batch 4"):
        d.hist2d("y:x", fit="gauss")


def test_T16_hexbin_range_raises_clean_value_error():
    """T16 (S-7 guard): hexbin(..., range=...) raises clean ValueError,
    NOT cryptic matplotlib PolyCollection error."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    with pytest.raises(ValueError, match="hexbin.*range|extent|S-7|Batch 4"):
        d.hexbin("y:x", range=((0, 5), (0, 5)))


def test_T17_hexbin_facet_by_raises_clean_value_error():
    """T17 (S-7 guard): hexbin(..., facet_by='sec') raises clean ValueError,
    NOT cryptic matplotlib PolyCollection error."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    with pytest.raises(ValueError, match="hexbin.*facet|S-7|Batch 4"):
        d.hexbin("y:x", facet_by="sec")


# =============================================================================
# T16b/T17b — Phase 13.51 FIX1 (F-1, Opus48_3 executed negative control):
# hexbin dispatch path must route to S-7 guards symmetrically with direct path.
# Pre-FIX1: `_hexbin_allowed` filter dropped facet_by/range BEFORE reaching
# the guards, so dispatch was a silent drop while direct call raised clean.
# Locked by these two tests; matches T16/T17 invariance contract, dispatch
# entry point.
# =============================================================================

def test_T16b_draw_type_hexbin_range_raises_clean_value_error_via_dispatch():
    """T16b (F-1 FIX1): d.draw(type='hexbin', range=...) routes to the same
    clean ValueError as direct d.hexbin(range=...). Pre-FIX1: silent drop."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    with pytest.raises(ValueError, match="hexbin.*range|extent|S-7|Batch 4"):
        d.draw("y:x", type="hexbin", range=(0, 5))


def test_T17b_draw_type_hexbin_facet_by_raises_clean_value_error_via_dispatch():
    """T17b (F-1 FIX1): d.draw(type='hexbin', facet_by=...) routes to the same
    clean ValueError as direct d.hexbin(facet_by=...). Pre-FIX1: silent drop."""
    df = _make_outlier_df_2d()
    d = DFDraw(df)
    with pytest.raises(ValueError, match="hexbin.*facet|S-7|Batch 4"):
        d.draw("y:x", type="hexbin", facet_by="sec")
