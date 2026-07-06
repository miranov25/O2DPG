"""Profile binning invariance suite (A ≡ B), NOT smoke.

Each test asserts a law a binning bug would break — specifically the overflow-fold
bug where np.clip folded out-of-range rows into edge bins, making a bin's value
depend on range=. Invariant: a bin's statistic is a property of the data IN that
bin, independent of the plot's x-window. Covers mean, std, median, quantile band,
and the grouped path (each group's per-bin call).

Baseline-free (no saved reference) → acceptance lock for PHASE 13.64.
Classified "invariance" in test_layer_classification.py → feature is Verified.
"""
import numpy as np
import pytest

from dfdraw.plots import profile as PF

MINBINS = 10


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    n = 300_000
    x = rng.uniform(-3, 3, n)
    y = 0.3 + 0.2 * np.abs(x) + rng.normal(0, 0.05, n)
    return x, y


def _centers(a, b, nb):
    e = np.linspace(a, b, nb + 1)
    return np.round((e[:-1] + e[1:]) / 2, 6)


def _prof_df(x, y, bins, rng):
    return PF._compute_profile(x, y, bins, rng, "sem", True, None)[4]


def _assert_range_invariant(wide, narrow, label):
    shared = sorted(set(wide) & set(narrow))
    assert len(shared) >= MINBINS, f"{label}: too few shared bins ({len(shared)})"
    for c in shared:
        assert np.isclose(wide[c], narrow[c], atol=1e-9, equal_nan=True), \
            f"{label}: value at x={c} changed with range= ({wide[c]} vs {narrow[c]})"


def test_profile_mean_range_invariance():
    x, y = _fixture()
    W = _prof_df(x, y, 30, (-1.5, 1.5)); N = _prof_df(x, y, 20, (-1.0, 1.0))
    _assert_range_invariant(dict(zip(np.round(W["x_center"], 6), W["y_mean"])),
                            dict(zip(np.round(N["x_center"], 6), N["y_mean"])), "mean")


def test_profile_std_range_invariance():
    x, y = _fixture()
    W = _prof_df(x, y, 30, (-1.5, 1.5)); N = _prof_df(x, y, 20, (-1.0, 1.0))
    _assert_range_invariant(dict(zip(np.round(W["x_center"], 6), W["y_std"])),
                            dict(zip(np.round(N["x_center"], 6), N["y_std"])), "std")


def test_profile_median_range_invariance():
    x, y = _fixture()
    dW = dict(zip(_centers(-1.5, 1.5, 30), PF._compute_per_bin_median(x, y, 30, (-1.5, 1.5))))
    dN = dict(zip(_centers(-1.0, 1.0, 20), PF._compute_per_bin_median(x, y, 20, (-1.0, 1.0))))
    _assert_range_invariant(dW, dN, "median")


def test_profile_quantile_range_invariance():
    x, y = _fixture()
    qW = PF._compute_per_bin_quantiles(x, y, 30, (-1.5, 1.5), (0.25, 0.75), None)
    qN = PF._compute_per_bin_quantiles(x, y, 20, (-1.0, 1.0), (0.25, 0.75), None)
    loW = qW["q_lower"] if hasattr(qW, "keys") else qW[0]
    loN = qN["q_lower"] if hasattr(qN, "keys") else qN[0]
    _assert_range_invariant(dict(zip(_centers(-1.5, 1.5, 30), loW)),
                            dict(zip(_centers(-1.0, 1.0, 20), loN)), "quantile-lo")


def test_profile_grouped_range_invariance():
    x, y = _fixture()
    g = np.random.default_rng(1).integers(0, 3, len(x))
    for gi in (0, 1, 2):
        m = g == gi
        W = _prof_df(x[m], y[m], 30, (-1.5, 1.5)); N = _prof_df(x[m], y[m], 20, (-1.0, 1.0))
        _assert_range_invariant(dict(zip(np.round(W["x_center"], 6), W["y_mean"])),
                                dict(zip(np.round(N["x_center"], 6), N["y_mean"])),
                                f"grouped mean (group {gi})")


def test_profile_mean_no_edge_pileup():
    x, y = _fixture()
    N = _prof_df(x, y, 20, (-1.0, 1.0))
    edge = N["y_mean"].to_numpy()[-1]
    m = (x >= 0.9) & (x < 1.0)
    assert np.isclose(edge, y[m].mean(), atol=1e-6), "edge bin inflated by overflow pileup"
