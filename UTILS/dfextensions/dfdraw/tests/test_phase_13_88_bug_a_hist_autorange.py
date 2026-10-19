"""PHASE_13_88_DF Bug A: canonical shared histogram autorange."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots._autorange import resolve_range_1d
from dfextensions.dfdraw.style import get_style_value


def _canonical_range(range_arg, values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return resolve_range_1d(
        range_arg, arr,
        style_strategy=get_style_value('autorange.strategy', 'hybrid'),
        style_k_robust=get_style_value('autorange.k_robust', 4.0),
        style_k_outlier=get_style_value('autorange.k_outlier', 1.5),
        style_percentile=get_style_value('autorange.percentile', (1.0, 99.0)),
    )[0]


def _fixture():
    rng = np.random.default_rng(1388)
    n = 1500
    x0 = np.r_[rng.normal(-0.4, 0.65, n), -35.0]
    x1 = np.r_[rng.normal(1.1, 1.05, n), 80.0]
    return pd.DataFrame({
        'x': np.r_[x0, x1],
        'side': np.r_[np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)],
    })


def _assert_common(stats, expected):
    branches = stats['hist_branch_data']
    assert len(branches) == 2
    e0 = branches[0][['x_low', 'x_high']].to_numpy(dtype=float)
    e1 = branches[1][['x_low', 'x_high']].to_numpy(dtype=float)
    np.testing.assert_array_equal(e0, e1)
    assert e0[0, 0] == pytest.approx(expected[0])
    assert e0[-1, 1] == pytest.approx(expected[1])
    assert 'normalize_data' in stats


@pytest.mark.parametrize('range_arg', [None, 'hybrid', 'robust_4mad'])
def test_A1_hist_selection_vector_uses_one_canonical_pooled_range(range_arg):
    df = _fixture()
    expected = _canonical_range(range_arg, df['x'].to_numpy())
    raw = (float(df['x'].min()), float(df['x'].max()))
    if range_arg in (None, 'hybrid'):
        assert not np.allclose(expected, raw)
    fig, _, stats = DFDraw(df).hist(
        'x', selection_vector=['side==0', 'side==1'], vector_compose='outer',
        bins=24, range=range_arg, normalize='ratio')
    try:
        _assert_common(stats, expected)
    finally:
        plt.close(fig)


def test_A2_explicit_numeric_range_remains_exact_green_control():
    df = _fixture()
    explicit = (-4.0, 5.0)
    fig, _, stats = DFDraw(df).hist(
        'x', selection_vector=['side==0', 'side==1'], vector_compose='outer',
        bins=24, range=explicit, normalize='ratio')
    try:
        _assert_common(stats, explicit)
    finally:
        plt.close(fig)


def test_A3_weights_vector_same_rows_remains_green_control():
    df = _fixture()
    df['w0'] = 1.0
    df['w1'] = np.linspace(0.5, 1.5, len(df))
    expected = _canonical_range(None, df['x'].to_numpy())
    fig, _, stats = DFDraw(df).hist(
        'x', weights_vector=['w0', 'w1'], vector_compose='outer',
        bins=24, normalize='ratio')
    try:
        _assert_common(stats, expected)
    finally:
        plt.close(fig)


def test_A4_two_expression_vector_uses_union_domain():
    rng = np.random.default_rng(1882)
    df = pd.DataFrame({
        'a': np.r_[rng.normal(-1.5, 0.45, 1800), -25.0],
        'b': np.r_[rng.normal(2.0, 0.80, 1800), 40.0],
    })
    expected = _canonical_range(None, np.concatenate([df['a'], df['b']]))
    fig, _, stats = DFDraw(df).hist(
        '[a,b]', bins=30, normalize='ratio', vector_compose='outer')
    try:
        _assert_common(stats, expected)
    finally:
        plt.close(fig)
