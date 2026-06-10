"""Phase 13.55.DF — draw_batch audit + Phase 13.51/13.52 type-gap fix tests.

Test scaffold per `PHASE_13_55_DF_DrawBatchAudit_Proposal_v1_2.md` §4.

17 invariance tests across 4 groups, exercising:
- D-1 type coverage (profile2d / scatter3d / histo alias / overlay strings)
  via both dict-form and list-form (group) draw_batch paths
- D-5 kwarg regression locks (central=, time_format=, auto_title=, fit=,
  selection_vector=) — proves kwargs ride correctly through the new
  route-through-draw() dispatch
- D-2 on_error behavior — new 'raise' default + explicit 'skip' opt-back-in
- Option B regression — same=True overlay, _suppress_layout=True flow,
  _GROUP_KEYS stripping (positive test per Sonnet58 panel guidance)

The Option B fix uses `self.draw(expr, type=plot_type, **merged)` at both
dispatch sites. To support this, two surgical extensions to draw():
  - profile2d early-dispatch before _parse_expr (mirror of scatter3d)
  - _suppress_layout whitelisted in kwarg validator
  - _suppress_layout exempted in overlay sugar's per-layer accept-check
The combination is "B-fix-draw" — architect-ratified 2026-06-10.

Architect decisions (v1.2 §0):
- Option B: route draw_batch through draw()
- on_error default change: 'skip' -> 'raise' (BREAKING)
- STRICT coder recusal for Opus1/Opus48_3 (overridden by architect for this
  implementation; recorded for audit trail)

References:
- PHASE_13_55_DF_DrawBatchAudit_Proposal_v1_2.md
- Sonnet65_PHASE_13_55_DF_ProposalPanelSummary_20260610.md (7 reviewers, [OK])
- B-fix-draw architect ratification 2026-06-10
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


# =============================================================================
# Fixtures
# =============================================================================

def _make_basic_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "sec": rng.integers(0, 4, n),
    })


def _make_outlier_df(n=500, outlier_pct=0.10, outlier_scale=100.0, seed=42):
    """Per Sonnet64 panel guidance (T7 outlier fixture).

    Standard N(0,1) data has mean == median (to ~5%); central='median' vs
    central='mean' produces identical-looking output, so A != B assertions
    are insensitive. The same fixture pattern used in Phase 13.51 T9b:
    inject ~10% of rows with extreme values to separate mean from median.
    """
    rng = np.random.default_rng(seed)
    n_outliers = int(n * outlier_pct)
    n_clean = n - n_outliers
    x_clean = rng.standard_normal(n_clean)
    y_clean = rng.standard_normal(n_clean)
    x_out = rng.standard_normal(n_outliers) * outlier_scale
    y_out = rng.standard_normal(n_outliers) * outlier_scale
    return pd.DataFrame({
        "x": np.concatenate([x_clean, x_out]),
        "y": np.concatenate([y_clean, y_out]),
    })


def _make_epoch_df(n=2000, seed=42):
    """Float64 epoch seconds near ALICE Run 3 timescale (~1.776e9 s)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "time_s": np.linspace(1.7760e9, 1.7761e9, n),
        "y": rng.standard_normal(n),
        "x": rng.standard_normal(n),
    })


@pytest.fixture(autouse=True)
def _close_figs_and_silence_mpl_noise():
    # The matplotlib colorbar UserWarning ("Adding colorbar to a different
    # Figure...") is pre-existing matplotlib state-machine noise unrelated
    # to Phase 13.55.DF; suppress globally for these tests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield
    plt.close("all")


# =============================================================================
# Group 1 — D-1 type coverage
# =============================================================================
# Pre-fix: each of these raised ValueError("Invalid type ...") from the
# hardcoded valid_types whitelist. Post-fix: route-through-draw() dispatches
# to the typed method.

def test_T1_dict_profile2d():
    """T1 — D-1: dict-form draw_batch accepts type='profile2d'."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(specs={'k': {'expr': 'z:y:x', 'type': 'profile2d'}}, verbose=False)
    assert r['_errors'] == {}, f"profile2d via dict-form raised: {r['_errors']}"
    assert r['_summary']['success'] == 1


def test_T2_dict_scatter3d():
    """T2 — D-1: dict-form draw_batch accepts type='scatter3d'."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(specs={'k': {'expr': 'z:y:x', 'type': 'scatter3d'}}, verbose=False)
    assert r['_errors'] == {}
    assert r['_summary']['success'] == 1


def test_T3_dict_histo_alias():
    """T3 — D-3: dict-form draw_batch normalizes 'histo' -> 'hist' via _TYPE_ALIASES."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(specs={'k': {'expr': 'x', 'type': 'histo'}}, verbose=False)
    assert r['_errors'] == {}
    assert r['_summary']['success'] == 1


def test_T4_dict_overlay_string():
    """T4 — D-4: dict-form draw_batch accepts overlay string 'hist2d+profile'."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(specs={'k': {'expr': 'y:x', 'type': 'hist2d+profile'}}, verbose=False)
    assert r['_errors'] == {}
    assert r['_summary']['success'] == 1


def test_T5_list_profile2d_in_group():
    """T5 — D-1 list-form: profile2d works inside a group, with _suppress_layout=True flowing through."""
    df = _make_basic_df()
    groups = [{
        'name': 'g',
        'ncols': 1,
        'plots': [{'expr': 'z:y:x', 'type': 'profile2d'}],
    }]
    r = DFDraw(df).draw_batch(specs=groups, verbose=False)
    assert r['_errors'] == {}
    assert 'g' in r


def test_T6_list_overlay_in_group():
    """T6 — D-4 list-form: overlay strings work inside a group."""
    df = _make_basic_df()
    groups = [{
        'name': 'g',
        'ncols': 1,
        'plots': [{'expr': 'y:x', 'type': 'hist2d+profile'}],
    }]
    r = DFDraw(df).draw_batch(specs=groups, verbose=False)
    assert r['_errors'] == {}
    assert 'g' in r


# =============================================================================
# Group 2 — D-5 kwarg regression locks
# =============================================================================
# These features ride as **kwargs to typed methods; under the new
# route-through-draw() dispatch the path is: draw_batch -> draw() -> typed
# method via **kwargs. Each test asserts an observable effect of the kwarg.

def test_T7_central_median_via_batch_with_outlier_fixture():
    """T7 — D-5: central='median' via batch renders a profile whose center
    differs measurably from central='mean'. Uses the Phase 13.51 T9b
    outlier fixture (10% rows @ 100x) — without it, mean ≈ median and
    the A != B assertion is insensitive (Sonnet64 P3-2)."""
    df = _make_outlier_df()
    r_mean = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'y:x', 'type': 'profile', 'central': 'mean'}}, verbose=False)
    r_median = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'y:x', 'type': 'profile', 'central': 'median'}}, verbose=False)
    assert r_mean['_errors'] == {} and r_median['_errors'] == {}
    # Extract the central-value array from each plot via the rendered line
    ax_mean = r_mean['k']['ax']
    ax_median = r_median['k']['ax']
    y_mean = ax_mean.lines[0].get_ydata()
    y_median = ax_median.lines[0].get_ydata()
    # With outliers, mean and median paths differ measurably at some bins
    finite_mean = y_mean[np.isfinite(y_mean)]
    finite_median = y_median[np.isfinite(y_median)]
    n_compare = min(len(finite_mean), len(finite_median))
    assert n_compare > 0
    diffs = np.abs(finite_mean[:n_compare] - finite_median[:n_compare])
    assert np.any(diffs > 1e-3), (
        "central='median' and central='mean' produced identical output via "
        f"batch — outlier fixture failed to separate the strategies. "
        f"max diff = {diffs.max() if len(diffs) else 'empty'}"
    )


def test_T8_time_format_via_batch():
    """T8 — D-5: time_format='%H:%M' on hist2d via batch produces HH:MM tick labels (Phase 13.51 S-8)."""
    df = _make_epoch_df()
    df["time_dt"] = pd.to_datetime(df["time_s"], unit="s")
    r = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'y:time_dt', 'type': 'hist2d', 'time_format': '%H:%M', 'bins': 50}},
        verbose=False)
    assert r['_errors'] == {}
    ax = r['k']['ax']
    labels = [t.get_text() for t in ax.get_xticklabels()]
    nonempty = [l for l in labels if l.strip()]
    assert any(":" in l for l in nonempty), (
        f"time_format='%H:%M' did not produce HH:MM labels via batch; got {nonempty}"
    )


def test_T9_auto_title_scatter_via_batch():
    """T9 — D-5 + Phase 13.54: auto_title=True on scatter via batch sets a non-empty title."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'y:x', 'type': 'scatter', 'auto_title': True}}, verbose=False)
    assert r['_errors'] == {}
    ax = r['k']['ax']
    title = ax.get_title()
    assert title and title.strip(), (
        f"auto_title=True via batch should set a non-empty title; got {title!r}"
    )


def test_T10_fit_gauss_profile_via_batch():
    """T10 — D-5: fit='gauss' on profile via batch produces fit parameters in stats."""
    df = _make_basic_df(n=1500)  # need enough rows for gauss fit to converge
    r = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'y:x', 'type': 'profile', 'fit': 'gauss'}}, verbose=False)
    assert r['_errors'] == {}
    stats = r['k']['stats']
    assert 'fit' in stats, f"fit= via batch missing fit block in stats; keys={list(stats.keys())}"
    fit_block = stats['fit']
    assert fit_block, f"fit= produced empty fit block: {fit_block}"


def test_T11_selection_vector_hist_via_batch():
    """T11 — D-5 + Phase 13.27: selection_vector= on hist via batch is
    accepted as a kwarg and forwarded through draw() to the typed
    method without an 'Unknown keyword' error.

    Phase 13.27's vector engine requires matched-length expression and
    selection_vector dimensions when both are vectorized; here we use a
    single-element selection_vector matching the scalar 'x' expr, which
    is the minimum form proving the kwarg flows through the new
    route-through-draw() dispatch."""
    df = _make_basic_df(n=1000)
    r = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'x', 'type': 'hist',
                     'selection_vector': ["x > 0"]}},
        verbose=False)
    # The kwarg reached the typed method without an Unknown-keyword error.
    # That alone is the regression-lock the test asserts.
    assert r['_errors'] == {}, (
        f"selection_vector= via batch raised: {r['_errors']}"
    )
    assert r['_summary']['success'] == 1


# =============================================================================
# Group 3 — D-2 on_error behavior
# =============================================================================

def test_T12_invalid_type_raises_under_new_default():
    """T12 — D-2 BREAKING: new default on_error='raise' surfaces invalid types as exceptions."""
    df = _make_basic_df()
    with pytest.raises(ValueError):
        DFDraw(df).draw_batch(specs={'k': {'expr': 'x', 'type': 'nonexistent_type'}}, verbose=False)


def test_T13_invalid_type_with_explicit_skip_is_counted_not_raised():
    """T13 — D-2: explicit on_error='skip' preserves the old behavior (errors counted in _errors, no raise)."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(
        specs={'k': {'expr': 'x', 'type': 'nonexistent_type'}}, on_error='skip', verbose=False)
    assert 'k' in r['_errors']
    assert r['_summary']['failed'] == 1


def test_T14_valid_type_under_new_default_raise_succeeds():
    """T14 — D-2 regression: valid specs under on_error='raise' return normally."""
    df = _make_basic_df()
    r = DFDraw(df).draw_batch(specs={'k': {'expr': 'x', 'type': 'hist'}}, verbose=False)
    assert r['_errors'] == {}
    assert r['_summary']['success'] == 1
    assert 'stats' in r['k']


# =============================================================================
# Group 4 — Option B regression (existing contracts preserved)
# =============================================================================

def test_T15_same_overlay_within_group_reuses_axis():
    """T15 — Option B regression: same=True within a group still overlays on
    the same axis (existing T11 contract from test_batch_groups.py).

    With same=True on the second plot, only 1 subplot axis is consumed for
    the group's two plots, and the axis has multiple stats entries (2)."""
    df = _make_basic_df(n=1000)
    groups = [{
        'name': 'g',
        'ncols': 1,
        'plots': [
            {'expr': 'x', 'type': 'hist'},
            {'expr': 'y', 'type': 'hist', 'same': True},
        ],
    }]
    r = DFDraw(df).draw_batch(specs=groups, verbose=False)
    assert r['_errors'] == {}
    # The group result holds a list of stats — one per plot
    assert len(r['g']['stats']) == 2
    # Only 1 axis consumed (second overlays on first)
    assert len(r['g']['axes']) == 1


def test_T16_suppress_layout_flows_to_typed_method():
    """T16 — Option B regression: _suppress_layout=True (set per subplot in
    list-form) flows through draw() to the typed method via **kwargs without
    raising the kwarg validator or the overlay engine."""
    df = _make_basic_df()
    # Use multiple plots so list-form path is exercised (sets _suppress_layout=True)
    groups = [{
        'name': 'g',
        'ncols': 2,
        'plots': [
            {'expr': 'x', 'type': 'hist'},
            {'expr': 'y:x', 'type': 'hist2d+profile'},   # exercises overlay engine path too
        ],
    }]
    # No exception means _suppress_layout was accepted by draw() and the
    # overlay engine — both fixed in Phase 13.55.DF.
    r = DFDraw(df).draw_batch(specs=groups, verbose=False)
    assert r['_errors'] == {}


def test_T17_group_keys_stripped_before_draw_positive_test():
    """T17 — Option B regression: _GROUP_KEYS (name, ncols, layout, figsize,
    etc.) are stripped before merged is passed to draw(), so draw() does not
    receive a 'ncols' kwarg it doesn't accept (positive test per Sonnet58
    panel C-1).

    Pre-fix this was implicit — getattr(self, plot_type) accepted only
    typed-method params anyway. With Option B routing, the strip becomes a
    correctness requirement: include _GROUP_KEYS members in the group dict
    and confirm no TypeError from draw()."""
    df = _make_basic_df()
    groups = [{
        'name': 'g_with_keys',
        'ncols': 3,         # _GROUP_KEYS member — must be stripped
        'figsize': (12, 4), # _GROUP_KEYS member — must be stripped
        'layout': (1, 1),   # _GROUP_KEYS member — dfdraw (nrows, ncols) override
        'plots': [{'expr': 'x', 'type': 'hist'}],
    }]
    # If _GROUP_KEYS leak into draw() as kwargs, draw() raises TypeError
    # (unknown keyword argument). Success means strip is intact.
    r = DFDraw(df).draw_batch(specs=groups, verbose=False)
    assert r['_errors'] == {}, (
        f"_GROUP_KEYS leaked into draw() and caused errors: {r['_errors']}"
    )
    assert 'g_with_keys' in r
