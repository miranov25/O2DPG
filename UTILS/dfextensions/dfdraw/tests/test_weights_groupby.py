"""weights= combined with group_by= — raw weighted counts (dfdraw bug fix).

Behavioral / baseline-free: each group's rendered weighted histogram must equal
the manual np.histogram(x_group, weights=w_group) on the per-group JOINT (x,w)
finite mask. Fixture deliberately injects NaNs into BOTH x and w (different rows)
so a per-group weight misalignment cannot pass. These assert the RESULT, not the
implementation, so they carry into PHASE 13.64 (StatResult unification) as the
acceptance baseline for the same use case.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.pyplot as plt

from dfdraw import DFDraw

EDGES = np.linspace(0, 10, 11)


@pytest.fixture
def df_nan():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.uniform(0, 10, n)
    w = rng.uniform(0.5, 2.0, n)
    x[rng.choice(n, 40, replace=False)] = np.nan      # NaN in x
    w[rng.choice(n, 40, replace=False)] = np.nan      # NaN in w (different rows)
    return pd.DataFrame({"x": x, "g": rng.integers(0, 2, n).astype(str), "w": w})


def _capture_hist_calls(fn):
    """Run fn() while capturing every (data, weights) passed to ax.hist."""
    calls = []
    orig = matplotlib.axes.Axes.hist

    def spy(self, xx, *a, **k):
        wt = k.get("weights")
        calls.append((np.asarray(xx).copy(),
                      None if wt is None else np.asarray(wt).copy()))
        return orig(self, xx, *a, **k)

    matplotlib.axes.Axes.hist = spy
    try:
        fn()
    finally:
        matplotlib.axes.Axes.hist = orig
    return calls


def _manual(df, group):
    m = ((df["g"] == group).to_numpy()
         & np.isfinite(df["x"].to_numpy())
         & np.isfinite(df["w"].to_numpy()))
    return np.histogram(df["x"].to_numpy()[m], bins=EDGES,
                        weights=df["w"].to_numpy()[m])[0]


def test_weights_groupby_raw_matches_manual(df_nan):
    """Each group's weighted histogram == manual per-group weighted np.histogram,
    with the per-row weights joint-masked and aligned to the group's rows."""
    calls = _capture_hist_calls(
        lambda: DFDraw(df_nan).hist("x", weights="w", group_by="g", bins=EDGES))
    groups = sorted(df_nan["g"].unique())
    assert len(calls) == len(groups), "one ax.hist call per group expected"
    for grp, (gd, gw) in zip(groups, calls):
        assert gw is not None and len(gw) == len(gd), \
            "per-row weights must be joint-masked and aligned to group_data"
        rendered = np.histogram(gd, bins=EDGES, weights=gw)[0]
        assert np.allclose(rendered, _manual(df_nan, grp)), \
            f"group {grp}: weighted heights diverge from manual reference"


def test_weights_groupby_no_weights_regression(df_nan):
    """group_by WITHOUT weights is unchanged: ax.hist gets weights=None and the
    counts equal the unweighted per-group np.histogram (x-only finite)."""
    calls = _capture_hist_calls(
        lambda: DFDraw(df_nan).hist("x", group_by="g", bins=EDGES))
    groups = sorted(df_nan["g"].unique())
    for grp, (gd, gw) in zip(groups, calls):
        assert gw is None
        m = (df_nan["g"] == grp).to_numpy() & np.isfinite(df_nan["x"].to_numpy())
        ref = np.histogram(df_nan["x"].to_numpy()[m], bins=EDGES)[0]
        assert np.allclose(np.histogram(gd, bins=EDGES)[0], ref)


def test_weights_groupby_normalized_raises(df_nan):
    """Normalized weighted grouping is deferred (per-row vs per-group precedence,
    PHASE 13.64) — must raise loudly, not silently mis-normalize."""
    with pytest.raises(NotImplementedError):
        DFDraw(df_nan).hist("x", weights="w", group_by="g",
                            bins=EDGES, hist_norm="probability")


def test_weights_groupby_stacked_raises(df_nan):
    """Stacked weighted grouping is deferred — must raise loudly."""
    with pytest.raises(NotImplementedError):
        DFDraw(df_nan).hist("x", weights="w", group_by="g",
                            bins=EDGES, stacked=True)
