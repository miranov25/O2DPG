"""PHASE_13_96_DF Stage A: independently prove median shares profile bins.

The existing median owner must use exactly the resolved x-range employed by
mean/std/SEM.  These tests intentionally use distant x outliers so the old
plain-minmax median path cannot accidentally match the profile autorange.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfextensions.dfdraw.drawer import DFDraw


def _data():
    rng = np.random.default_rng(1396)
    x = np.r_[np.linspace(0, 10, 2000), np.repeat(100.0, 5)]
    y = np.sin(x / 2.0) + 0.35 * rng.normal(size=x.size)
    return pd.DataFrame({'x': x, 'y': y})


def _median_on_exported_edges(df, profile):
    """Independent NumPy truth using actual profile bin boundaries."""
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    lo = profile['x_low'].to_numpy()
    hi = profile['x_high'].to_numpy()
    values = []
    for k, (left, right) in enumerate(zip(lo, hi)):
        belongs = (x >= left) & ((x <= right) if k == len(lo)-1 else (x < right))
        values.append(float(np.median(y[belongs])) if belongs.any() else np.nan)
    return np.asarray(values)


def test_M1_both_median_curve_agrees_with_independent_exact_bin_truth():
    df = _data()
    fig, ax, stats = DFDraw(df).profile('y:x', bins=10, central='both',
                                      min_entries=2, return_data=True)
    try:
        table = stats['profile_data']
        assert stats['autorange_used'][1] < 100.0  # fixture must falsify minmax
        expected = _median_on_exported_edges(df, table)
        median_lines = [artist for artist in ax.lines if '(median)' in artist.get_label()]
        assert len(median_lines) == 1
        visible = table['count'].to_numpy() >= 2
        np.testing.assert_allclose(median_lines[0].get_ydata(), expected[visible],
                                   atol=1e-12, rtol=1e-12, equal_nan=True)
    finally:
        plt.close(fig)


def test_M2_median_exported_y_central_agrees_with_same_bin_truth():
    df = _data()
    fig, ax, stats = DFDraw(df).profile('y:x', bins=10, central='median',
                                      min_entries=2, return_data=True)
    try:
        table = stats['profile_data']
        expected = _median_on_exported_edges(df, table)
        np.testing.assert_allclose(table['y_central'].to_numpy(), expected,
                                   atol=1e-12, rtol=1e-12, equal_nan=True)
        assert not np.allclose(table['y_mean'].to_numpy(), expected, equal_nan=True)
    finally:
        plt.close(fig)


def test_M3_explicit_xrange_median_remains_exact():
    df = _data()
    fig, ax, stats = DFDraw(df).profile('y:x', bins=10, range=(0.0, 12.0),
                                      central='median', min_entries=2,
                                      return_data=True)
    try:
        table = stats['profile_data']
        expected = _median_on_exported_edges(df, table)
        np.testing.assert_allclose(table['y_central'].to_numpy(), expected,
                                   atol=1e-12, rtol=1e-12, equal_nan=True)
        assert tuple(stats['autorange_used']) == (0.0, 12.0)
    finally:
        plt.close(fig)
