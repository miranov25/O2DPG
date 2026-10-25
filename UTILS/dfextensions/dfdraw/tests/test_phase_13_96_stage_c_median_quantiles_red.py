"""PHASE_13_96 Stage C: independent median and quantile rendering oracles.

These tests should fail on the installed Stage B candidate and pass on C.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def _data():
    rng = np.random.default_rng(1396)
    x = np.repeat(np.arange(6, dtype=float) + 0.5, 41)
    y = 0.55*x + rng.gamma(shape=1.5, scale=0.8, size=len(x))
    return pd.DataFrame({'x': x, 'y': y, 'w': 0.5 + x/6, 'g': np.where(x < 3, 'A', 'B')})


def _expected(d, q):
    return np.array([np.quantile(d.loc[(d.x >= i) & (d.x < i+1), 'y'], q)
                     for i in range(6)])


def _call(d=None, **kwargs):
    return DFDraw(_data() if d is None else d).profile(
        'y:x', bins=6, range=(0,6), return_data=True, **kwargs)


def test_C01_median_matches_numpy_at_same_bin_centers_and_has_no_errorbars():
    fig, ax, stats = _call(stat_list=['median'])
    try:
        np.testing.assert_allclose(stats['statistic_curves']['median'], _expected(_data(), .5), atol=1e-12)
        median = [line for line in ax.lines if line.get_label() == 'median']
        assert len(median) == 1
        np.testing.assert_allclose(median[0].get_xdata(), stats['profile_data'].x_center.to_numpy())
        np.testing.assert_allclose(median[0].get_ydata(), _expected(_data(), .5))
        assert not ax.collections  # median has no invented error bars
    finally: plt.close(fig)


def test_C02_mean_median_std_sem_four_distinct_curves_and_one_profile_computation(monkeypatch):
    import dfextensions.dfdraw.plots.profile as owner
    original = owner._compute_profile
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, '_compute_profile', counted)
    fig, ax, stats = _call(stat_list=['mean','median','std','sem'])
    try:
        assert len(calls) == 1
        assert list(stats['statistic_curves']) == ['mean','median','std','sem']
        assert len([l for l in ax.lines if l.get_label() == 'median']) == 1
        assert len([l for l in ax.lines if l.get_label() == 'std']) == 1
        assert len([l for l in ax.lines if l.get_label() == 'sem']) == 1
    finally: plt.close(fig)


@pytest.mark.parametrize('mode,qs', [
    ('error_bars', [.16,.84]), ('band', [.16,.5,.84]),
    ('discrete', [.2,.65]), ('nested_band',[.05,.16,.5,.84,.95])])
def test_C03_quantiles_only_reuses_legacy_mode_and_reports_values(mode, qs):
    obj = DFDraw(_data())
    fig0,ax0,old = obj.profile('y:x', bins=6, range=(0,6),
                                  quantiles=qs, quantile_mode=mode, return_data=True)
    fig1,ax1,new = obj.profile('y:x', bins=6, range=(0,6),
                                  quantiles=qs, quantile_mode=mode,
                                  stat_list=['quantiles'], return_data=True)
    try:
        assert len(ax0.lines) == len(ax1.lines)
        assert len(ax0.collections) == len(ax1.collections)
        for a,b in zip(ax0.lines, ax1.lines):
            np.testing.assert_allclose(a.get_ydata(), b.get_ydata(), equal_nan=True)
        q = new['statistic_curves']['quantiles']
        assert q['mode'] == mode
        assert tuple(q['fractions']) == tuple(sorted(qs))
        for fraction in qs:
            np.testing.assert_allclose(q['values'][fraction], _expected(_data(), fraction), atol=1e-12)
    finally: plt.close(fig0);plt.close(fig1)


def test_C04_mean_and_band_quantile_have_one_central_mean_curve():
    fig,ax,stats = _call(stat_list=['mean','quantiles'], quantiles=[.16,.5,.84], quantile_mode='band')
    try:
        assert list(stats['statistic_curves']) == ['mean','quantiles']
        assert len([c for c in ax.containers if c.get_label() == 'mean']) == 1
        assert len(ax.containers) == 1
        assert len(ax.collections) == 2  # mean error bars + fill_between quantile band
        assert len([l for l in ax.lines if l.get_label() == 'mean']) == 0
    finally: plt.close(fig)


def test_C05_mean_quantile_asym_errorbars_overlay_without_duplicate_mean_curve():
    fig,ax,s = _call(stat_list=['mean','quantiles'], quantiles=[.16,.84])
    try:
        assert len([c for c in ax.containers if c.get_label() == 'mean']) == 1
        assert len(ax.containers) == 2  # central mean + marker-free quantile geometry
        assert len([l for l in ax.lines if l.get_label() == 'mean']) == 0
        assert len([l for l in ax.lines if l.get_label() == 'quantiles']) == 0
    finally: plt.close(fig)


def test_C06_median_discrete_quantile_half_is_not_duplicated():
    fig,ax,s = _call(stat_list=['median','quantiles'], quantiles=[.2,.5,.8], quantile_mode='discrete')
    try:
        assert len([l for l in ax.lines if l.get_label() == 'median']) == 1
        assert len([l for l in ax.lines if l.get_label() == 'q=20%']) == 1
        assert len([l for l in ax.lines if l.get_label() == 'q=80%']) == 1
    finally: plt.close(fig)


@pytest.mark.parametrize('extra,expect', [
    ({'stat_list':['quantiles']}, 'quantiles='),
    ({'stat_list':['median'], 'weights':'w'}, 'median'),
    ({'stat_list':['median'], 'group_by':'g'}, 'group'),
    ({'stat_list':['mean','quantiles'], 'weights':'w','quantiles':[.16,.84]}, 'weighted'),
    ({'stat_list':['median'], 'central':'mean'}, 'central'),
    ({'stat_list':['mean','quantiles'], 'quantiles':[.1,.9], 'quantile_mode':'discrete', 'weights':'w'}, 'weighted'),
])
def test_C07_refuse_unsupported_before_new_figure(extra,expect):
    original = set(plt.get_fignums())
    with pytest.raises((ValueError, NotImplementedError), match=expect):
        _call(**extra)
    assert set(plt.get_fignums()) == original


def test_C08_draw_door_median_and_quantile_parity():
    obj = DFDraw(_data())
    opts = dict(bins=6, range=(0,6), stat_list=['median','quantiles'],
                quantiles=[.16,.5,.84], return_data=True)
    a,b,s1 = obj.profile('y:x', **opts)
    c,d,s2 = obj.draw('y:x', type='profile', **opts)
    try:
        for name in ('median',):
            np.testing.assert_array_equal(s1['statistic_curves'][name], s2['statistic_curves'][name])
        for q in [.16,.5,.84]:
            np.testing.assert_array_equal(s1['statistic_curves']['quantiles']['values'][q],
                                          s2['statistic_curves']['quantiles']['values'][q])
    finally:plt.close(a);plt.close(c)


def test_C09_quantile_values_match_raw_numpy_on_actual_resolved_autorange_edges():
    # An extreme x outlier makes the profile's resolved range differ from the
    # raw min/max. The independent oracle must use the profile export's edges.
    rng = np.random.default_rng(3916)
    x = np.r_[rng.uniform(0, 10, size=2200), [100.0] * 12]
    y = np.r_[np.sin(x[:2200] / 2) + rng.normal(0, .5, 2200), [90.0] * 12]
    frame = pd.DataFrame({'x': x, 'y': y})
    fig,ax,s = DFDraw(frame).profile('y:x', bins=10, min_entries=5,
                           stat_list=['quantiles'], quantiles=[.16,.5,.84],
                           return_data=True)
    try:
        exported = s['profile_data']
        qvalues = s['statistic_curves']['quantiles']['values']
        assert exported.x_high.iloc[-1] < 100  # outlier excluded by autorange
        for i, (low,high) in enumerate(zip(exported.x_low,exported.x_high)):
            cell = frame.y[(frame.x >= low) & ((frame.x <= high) if i == len(exported)-1
                                                    else (frame.x < high))]
            for q in [.16,.5,.84]:
                truth = np.quantile(cell,q) if len(cell) else np.nan
                np.testing.assert_allclose(qvalues[q][i], truth, atol=1e-12,
                                           equal_nan=True)
    finally: plt.close(fig)


def test_C10_returned_statistic_identity_exists_without_return_data():
    fig,ax,s = DFDraw(_data()).profile(
        'y:x', bins=6, range=(0,6),
        stat_list=['median','quantiles'], quantiles=[.16,.5,.84],
        return_data=False)
    try:
        assert 'profile_data' not in s
        assert list(s['statistic_curves']) == ['median','quantiles']
        assert len(s['statistic_curves']['median']) == 6
        assert list(s['statistic_curves']['quantiles']['values']) == [.16,.5,.84]
    finally: plt.close(fig)
