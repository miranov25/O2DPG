"""PHASE_13_84_DF — 1D histogram ratio normalization (two branches).

This file is intentionally small.  It proves the exact deadline seam before
any product edit:

    1D hist + two weights_vector branches + vector_compose="outer"
    + hist_errors=True + normalize="ratio" + overlay+diff

Positive control:
    the same two weighted histogram branches WITHOUT normalize already work.

This file began as the pre-fix RED characterization (1 passed, 1 failed on the
pre-fix product). It is now the permanent PHASE_13_84 test: every test must pass.

The failing test must show that the two upper histogram branches exist but the
requested normalization panel is absent.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw


N_BINS = 4
RANGE = (0.0, 4.0)
EDGES = np.linspace(RANGE[0], RANGE[1], N_BINS + 1)


def _fixture():
    """Deterministic weighted fixture; every bin is populated in both branches."""
    return pd.DataFrame(
        {
            "x": [
                0.15, 0.35,
                1.10, 1.35, 1.70,
                2.20, 2.55,
                3.10, 3.35, 3.70,
            ],
            "w0": [
                1.0, 2.0,
                1.5, 0.5, 2.5,
                3.0, 1.0,
                1.0, 2.0, 4.0,
            ],
            "w1": [
                2.0, 1.0,
                1.0, 2.0, 1.0,
                1.5, 2.5,
                3.0, 1.0, 2.0,
            ],
        }
    )


def _numpy_truth(frame):
    x = frame["x"].to_numpy()
    w0 = frame["w0"].to_numpy()
    w1 = frame["w1"].to_numpy()

    h0, _ = np.histogram(x, bins=EDGES, weights=w0)
    h1, _ = np.histogram(x, bins=EDGES, weights=w1)

    e0 = np.sqrt(np.histogram(x, bins=EDGES, weights=w0 * w0)[0])
    e1 = np.sqrt(np.histogram(x, bins=EDGES, weights=w1 * w1)[0])

    # PHASE_13_84 deliberately reuses existing profile-normalization semantics.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = h0 / h1
        ratio_error = np.sqrt((e0 / h1) ** 2 + (h0 * e1 / h1**2) ** 2)
    return h0, h1, e0, e1, ratio, ratio_error


def _bar_series(ax):
    """Return one bin-height array per rendered histogram branch."""
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    return [np.asarray([p.get_height() for p in c.patches], dtype=float)
            for c in bars]


def _errorbar_count(ax):
    return sum(isinstance(c, ErrorbarContainer) for c in ax.containers)


def _draw_kwargs(**over):
    kw = dict(
        type="hist",
        bins=N_BINS,
        range=RANGE,
        weights_vector=["w0", "w1"],
        vector_compose="outer",
        hist_errors=True,
        histtype="bar",
        auto_title=False,
    )
    kw.update(over)
    return kw


def _lower_panel(fig, ax_top):
    """The lower (ratio) axes: the other axes of the figure, sharing x with ax_top."""
    others = [a for a in fig.axes if a is not ax_top]
    assert len(others) == 1, f"expected exactly one lower panel, found {len(others)}"
    return others[0]


def _rendered_ratio(ax_diff):
    """Read the ratio points back from the rendered errorbar artist (not the payload)."""
    ebs = [c for c in ax_diff.containers if isinstance(c, ErrorbarContainer)]
    assert ebs, "lower panel has no rendered errorbar artist"
    line = ebs[0].lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Positive control (kept from the red-first characterization)
# ---------------------------------------------------------------------------

def test_positive_control_two_weighted_histograms_without_normalize():
    """Existing two-branch weighted histogram behavior must stay exactly as it was."""
    frame = _fixture()
    h0, h1, _, _, _, _ = _numpy_truth(frame)
    d = DFDraw(frame)
    try:
        fig, ax, stats = d.draw("x", **_draw_kwargs())
        rendered = _bar_series(ax)
        assert len(rendered) == 2
        np.testing.assert_allclose(rendered[0], h0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(rendered[1], h1, rtol=0.0, atol=1e-12)
        assert _errorbar_count(ax) == 2
        assert len(fig.axes) == 1, "normalize=None must not create a lower panel"
        assert isinstance(stats, list) and len(stats) == 2, \
            "normalize=None keeps the per-branch stats list unchanged"
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# The phase deliverable: ratio panel, upper histograms untouched
# ---------------------------------------------------------------------------

def test_hist_ratio_upper_panel_histograms_unchanged():
    """With normalize='ratio' the two upper histograms are still the exact NumPy truth."""
    frame = _fixture()
    h0, h1, _, _, _, _ = _numpy_truth(frame)
    try:
        fig, ax, stats = DFDraw(frame).draw("x", normalize="ratio", **_draw_kwargs())
        rendered = _bar_series(ax)
        assert len(rendered) == 2, "both histogram branches must survive, in declared order"
        np.testing.assert_allclose(rendered[0], h0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(rendered[1], h1, rtol=0.0, atol=1e-12)
        assert _errorbar_count(ax) == 2
    finally:
        plt.close("all")


def test_hist_ratio_lower_panel_rendered_values_match_independent_truth():
    """The rendered lower-panel artist carries branch0/branch1 at the bin centres."""
    frame = _fixture()
    _, _, _, _, ratio, ratio_err = _numpy_truth(frame)
    try:
        fig, ax, stats = DFDraw(frame).draw("x", normalize="ratio", **_draw_kwargs())
        ax_diff = _lower_panel(fig, ax)
        x, y = _rendered_ratio(ax_diff)
        centres = 0.5 * (EDGES[:-1] + EDGES[1:])
        np.testing.assert_allclose(x, centres, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(y, ratio, rtol=0.0, atol=1e-12)
        # the reference line at 1 is drawn (profile convention for ratio)
        assert any(np.allclose(l.get_ydata(), 1.0) for l in ax_diff.get_lines()
                   if len(l.get_ydata()) and np.all(np.isfinite(l.get_ydata()))), \
            "ratio panel must carry the reference line at 1"
    finally:
        plt.close("all")


def test_hist_ratio_payload_values_and_errors_match_profile_convention():
    """normalize_data has the profile schema and the propagated ratio error."""
    frame = _fixture()
    h0, h1, e0, e1, ratio, ratio_err = _numpy_truth(frame)
    try:
        fig, ax, stats = DFDraw(frame).draw("x", normalize="ratio", **_draw_kwargs())
        assert isinstance(stats, dict), "normalize returns the single normalize stats dict (profile precedent)"
        assert stats["normalize_mode"] == "ratio"
        nd = stats["normalize_data"]
        for col in ("x_center", "value", "error", "mask_undefined",
                    "signal_central", "signal_sigma", "reference_central", "reference_sigma"):
            assert col in nd.columns, f"normalize_data missing column {col}"
        np.testing.assert_allclose(nd["signal_central"], h0, atol=1e-12)
        np.testing.assert_allclose(nd["reference_central"], h1, atol=1e-12)
        np.testing.assert_allclose(nd["signal_sigma"], e0, atol=1e-12)
        np.testing.assert_allclose(nd["reference_sigma"], e1, atol=1e-12)
        np.testing.assert_allclose(nd["value"], ratio, atol=1e-12)
        np.testing.assert_allclose(nd["error"], ratio_err, atol=1e-12)
        assert not nd["mask_undefined"].any(), "fully populated fixture: no undefined bins"
        assert stats["n_masked_bins"] == 0 and stats["n_total_bins"] == N_BINS
        # the per-branch histogram tables carry the sufficient accumulators
        b0, b1 = stats["hist_branch_data"]
        np.testing.assert_allclose(b0["sum_w"], h0, atol=1e-12)
        np.testing.assert_allclose(b1["sum_w2"], e1 ** 2, atol=1e-12)
    finally:
        plt.close("all")


def test_hist_ratio_step_histtype_acceptance_form():
    """The motivating ADF form (histtype='step', linewidth=2) yields the same ratio."""
    frame = _fixture()
    _, _, _, _, ratio, ratio_err = _numpy_truth(frame)
    try:
        fig, ax, stats = DFDraw(frame).draw(
            "x", normalize="ratio", normalize_layout="overlay+diff",
            **_draw_kwargs(histtype="step", linewidth=2),
        )
        assert len(fig.axes) == 2
        np.testing.assert_allclose(stats["normalize_data"]["value"], ratio, atol=1e-12)
        np.testing.assert_allclose(stats["normalize_data"]["error"], ratio_err, atol=1e-12)
        x, y = _rendered_ratio(_lower_panel(fig, ax))
        np.testing.assert_allclose(y, ratio, atol=1e-12)
    finally:
        plt.close("all")


def test_hist_ratio_draw_door_equals_typed_hist_door():
    """draw(type='hist', ...) and hist(...) produce the same normalize_data."""
    frame = _fixture()
    kw = _draw_kwargs()
    kw.pop("type")
    try:
        _, _, s_draw = DFDraw(frame).draw("x", type="hist", normalize="ratio", **kw)
        _, _, s_hist = DFDraw(frame).hist("x", normalize="ratio", **kw)
        pd.testing.assert_frame_equal(s_draw["normalize_data"], s_hist["normalize_data"])
    finally:
        plt.close("all")


def test_hist_ratio_selection_vector_two_branches():
    """Two selection_vector branches are an equally valid two-branch source."""
    frame = _fixture()
    x = frame["x"].to_numpy()
    lo, hi = x < 2.0, x >= 0.0     # 'hi' is the full frame → denominator never 0
    h_lo, _ = np.histogram(x[lo], bins=EDGES)
    h_hi, _ = np.histogram(x[hi], bins=EDGES)
    try:
        fig, ax, stats = DFDraw(frame).hist(
            "x", bins=N_BINS, range=RANGE, hist_errors=True,
            selection_vector=["x < 2.0", "x >= 0.0"], vector_compose="outer",
            normalize="ratio", auto_title=False,
        )
        nd = stats["normalize_data"]
        expected = np.where(h_lo > 0, h_lo / h_hi, np.nan)
        got = nd["value"].to_numpy()
        # profile convention: bins with a zero numerator are masked as undefined
        assert np.array_equal(nd["mask_undefined"].to_numpy(), h_lo == 0)
        np.testing.assert_allclose(got[h_lo > 0], expected[h_lo > 0], atol=1e-12)
    finally:
        plt.close("all")


def test_hist_ratio_zero_denominator_is_masked_like_profile():
    """A bin empty in the reference is undefined (masked), exactly as the profile path does."""
    frame = _fixture().copy()
    frame.loc[frame["x"] > 3.0, "w1"] = 0.0      # empty the last reference bin
    h0, h1, *_ = _numpy_truth(frame)
    assert h1[-1] == 0.0
    try:
        fig, ax, stats = DFDraw(frame).draw("x", normalize="ratio", **_draw_kwargs())
        nd = stats["normalize_data"]
        assert bool(nd["mask_undefined"].iloc[-1]) is True
        assert stats["n_masked_bins"] == 1
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (h0 / h1)[:-1]
        np.testing.assert_allclose(nd["value"].to_numpy()[:-1], expected, atol=1e-12)
    finally:
        plt.close("all")


def test_hist_ratio_diff_only_layout():
    """normalize_layout='diff_only' renders only the ratio panel."""
    frame = _fixture()
    _, _, _, _, ratio, _ = _numpy_truth(frame)
    try:
        fig, ax, stats = DFDraw(frame).draw(
            "x", normalize="ratio", normalize_layout="diff_only", **_draw_kwargs())
        assert len(fig.axes) == 1
        x, y = _rendered_ratio(ax)
        np.testing.assert_allclose(y, ratio, atol=1e-12)
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Loud refusals outside the ratified domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(weights_vector=["w0"]),                          # one branch
    dict(weights_vector=["w0", "w1", "w0"]),              # three branches
])
def test_hist_ratio_refuses_wrong_branch_count(bad):
    frame = _fixture()
    with pytest.raises(ValueError, match="exactly 2 resolved"):
        DFDraw(frame).draw("x", normalize="ratio", **_draw_kwargs(**bad))
    plt.close("all")


@pytest.mark.parametrize("bad, exc", [
    (dict(normalize="delta"), NotImplementedError),
    (dict(normalize="pull"), NotImplementedError),
    (dict(normalize="ratio", facet_by="x"), NotImplementedError),
    (dict(normalize="ratio", group_by="x"), NotImplementedError),
    (dict(normalize="ratio", cumulative=True), NotImplementedError),
    (dict(normalize="ratio", normalize_layout="sideways"), ValueError),
])
def test_hist_ratio_refuses_outside_domain(bad, exc):
    frame = _fixture()
    kw = _draw_kwargs()
    kw.update(bad)
    with pytest.raises(exc):
        DFDraw(frame).draw("x", **kw)
    plt.close("all")


def test_hist_ratio_creates_no_stray_figures():
    """Exactly one figure per call, also for diff_only (off-screen branch drawing is closed)."""
    frame = _fixture()
    plt.close("all")
    try:
        DFDraw(frame).draw("x", normalize="ratio", normalize_layout="diff_only", **_draw_kwargs())
        assert len(plt.get_fignums()) == 1
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Readability: the two upper histograms must be distinguishable
# (caught by the real-data acceptance figure, not by the numerical tests)
# ---------------------------------------------------------------------------

def _distinct_branch_colors(ax, histtype):
    """Return the set of colours that identify the branches in the upper panel."""
    from matplotlib.colors import to_hex
    if histtype == "step":
        cols = {to_hex(p.get_edgecolor()) for p in ax.patches}
    else:
        cols = {to_hex(p.get_facecolor()) for p in ax.patches}
    return cols


@pytest.mark.parametrize("histtype", ["step", "bar", "stepfilled"])
def test_hist_ratio_upper_panel_branches_have_distinct_colors(histtype):
    """Each branch has its own colour; error bars follow their branch (step: outline colour)."""
    from matplotlib.colors import to_hex
    frame = _fixture()
    try:
        fig, ax, stats = DFDraw(frame).draw(
            "x", normalize="ratio", **_draw_kwargs(histtype=histtype))
        cols = _distinct_branch_colors(ax, histtype)
        assert len(cols) == 2, f"{histtype}: expected two distinguishable branches, got colours {cols}"
        ebs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        assert len(ebs) == 2
        eb_cols = {to_hex(c.lines[2][0].get_color()) if len(c.lines[2]) else None for c in ebs}
        assert eb_cols <= cols, f"{histtype}: error-bar colours {eb_cols} must match branch colours {cols}"
    finally:
        plt.close("all")


def test_hist_ratio_user_color_applies_to_both_branches():
    """An explicit color= is a uniform user override (same rule as elsewhere in dfdraw).

    Tested through the typed hist() door: draw(type="hist", color=...) does not
    forward color= to hist() at all — a pre-existing door divergence unrelated to
    normalize (verified on the plain histogram path), left for PHASE_13_82's
    door-equivalence classification and recorded in the CRR.
    """
    from matplotlib.colors import to_hex
    frame = _fixture()
    kw = _draw_kwargs(histtype="step")
    kw.pop("type")
    try:
        fig, ax, stats = DFDraw(frame).hist(
            "x", normalize="ratio", color="green", **kw)
        assert _distinct_branch_colors(ax, "step") == {to_hex("green")}
    finally:
        plt.close("all")


def test_hist_ratio_auto_title_is_applied_like_plain_hist():
    """auto_title=True titles the normalized figure exactly like a plain histogram would."""
    frame = _fixture()
    try:
        _, ax_plain, _ = DFDraw(frame).hist("x", bins=N_BINS, range=RANGE,
                                            selection="x > 0", auto_title=True)
        plain_title = ax_plain.get_title()
        assert plain_title, "positive control: plain hist auto_title yields a title"
        kw = _draw_kwargs(); kw.pop("type"); kw.pop("auto_title")
        fig, ax, stats = DFDraw(frame).hist("x", normalize="ratio", selection="x > 0",
                                            auto_title=True, **kw)
        assert ax.get_title() == plain_title, (
            f"normalized histogram title {ax.get_title()!r} must equal the plain "
            f"histogram title {plain_title!r}")
        fig2, ax2, _ = DFDraw(frame).hist("x", normalize="ratio", title="fixed", **kw)
        assert ax2.get_title() == "fixed", "explicit title= wins"
    finally:
        plt.close("all")
