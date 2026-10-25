"""PHASE_13_96 Stage B: statistic curves reuse one existing profile computation.

Stage B is deliberately scalar-profile only. Branch/facet/group and median/quantile
composition belong to the remaining approved stage units, not this change.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def _df():
    rng = np.random.default_rng(1396)
    x = np.repeat(np.arange(6, dtype=float) + 0.5, 30)
    y = x * 3.0 + rng.normal(0, .5, len(x))
    return pd.DataFrame({'x': x, 'y': y, 'z': 2*y, 'g': np.where(x < 3, 'A', 'B'),
                         'w': np.linspace(.5, 2, len(x))})


def _truth(frame, stat):
    values = []
    for i in range(6):
        y = frame.loc[(frame.x >= i) & (frame.x < i+1), 'y'].to_numpy()
        if stat == 'mean': values.append(np.mean(y))
        elif stat == 'std': values.append(np.std(y, ddof=1))
        elif stat == 'sem': values.append(np.std(y, ddof=1)/np.sqrt(len(y)))
    return np.asarray(values)


@pytest.mark.parametrize('stat', ['mean', 'std', 'sem'])
def test_B01_stat_curves_match_independent_numpy(stat):
    frame = _df()
    fig, ax, data = DFDraw(frame).profile('y:x', bins=6, range=(0,6),
                                          stat_list=[stat], return_data=True)
    try:
        assert list(data['statistic_curves']) == [stat]
        np.testing.assert_allclose(data['statistic_curves'][stat], _truth(frame, stat), atol=1e-12)
        if stat == 'mean':
            containers = [c for c in ax.containers if c.get_label() == 'mean']
            assert len(containers) == 1
            line = containers[0].lines[0]
        else:
            lines = [l for l in ax.lines if l.get_label() == stat]
            assert len(lines) == 1
            line = lines[0]
        np.testing.assert_allclose(line.get_ydata(), _truth(frame, stat), atol=1e-12)
    finally:
        plt.close(fig)


def test_B02_two_stat_curves_in_order_and_without_mean_uncertainties():
    fig, ax, data = DFDraw(_df()).profile('y:x', bins=6, range=(0,6),
                stat_list=['mean','std','sem'], return_data=True)
    try:
        assert list(data['statistic_curves']) == ['mean','std','sem']
        for name in ('std', 'sem'):
            lines = [l for l in ax.lines if l.get_label() == name]
            assert len(lines) == 1
            assert lines[0].get_marker() in ('^','v')
        # mean retains existing error bars, whereas std/sem are independent values
        assert len(ax.collections) == 1
        assert len(data['profile_data']) == 6
    finally: plt.close(fig)


def test_B03_draw_and_typed_profile_equivalent():
    obj = DFDraw(_df())
    fig1, ax1, s1 = obj.profile('y:x', bins=6, range=(0,6), stat_list=['mean','sem'])
    fig2, ax2, s2 = obj.draw('y:x', type='profile', bins=6, range=(0,6), stat_list=['mean','sem'])
    try:
        assert list(s1['statistic_curves']) == list(s2['statistic_curves'])
        for key in s1['statistic_curves']:
            np.testing.assert_array_equal(s1['statistic_curves'][key], s2['statistic_curves'][key])
    finally: plt.close(fig1); plt.close(fig2)


def test_B04_historical_default_none_is_unchanged():
    obj = DFDraw(_df())
    fig1, ax1, s1 = obj.profile('y:x', bins=6, range=(0,6), return_data=True)
    fig2, ax2, s2 = obj.profile('y:x', bins=6, range=(0,6), return_data=True, stat_list=None)
    try:
        assert list(s1) == list(s2)
        pd.testing.assert_frame_equal(s1['profile_data'], s2['profile_data'])
        assert len(ax1.lines) == len(ax2.lines)
    finally:plt.close(fig1);plt.close(fig2)


def test_B05_compute_profile_called_once(monkeypatch):
    import dfextensions.dfdraw.plots.profile as owner
    old = owner._compute_profile
    count = []
    def counted(*args, **kwargs):
        count.append(True)
        return old(*args, **kwargs)
    monkeypatch.setattr(owner, '_compute_profile', counted)
    fig,ax,s = DFDraw(_df()).profile('y:x', bins=6, range=(0,6), stat_list=['mean','std','sem'])
    try: assert len(count) == 1
    finally:plt.close(fig)


def test_B06_weighted_statistics_equal_existing_profile_data():
    obj = DFDraw(_df())
    fig,ax,s = obj.profile('y:x',bins=6, range=(0,6),stat_list=['mean','std','sem'], weights='w',return_data=True)
    try:
        for name in ('mean','std','sem'):
            np.testing.assert_array_equal(s['statistic_curves'][name], s['profile_data']['y_'+name].to_numpy())
    finally:plt.close(fig)


@pytest.mark.parametrize('bad,words', [({'stat_list':['median'], 'weights':'w'}, 'median'),
                                     ({'stat_list':['quantiles']}, 'quantiles'),
                                     ({'stat_list':['median'], 'group_by':'g'}, 'group'),
                                     ({'stat_list':['median'], 'facet_by':'g'}, 'facet'),
                                     ({'stat_list':['mean'],'central':'both'}, 'central')])
def test_B07_future_or_conflicting_combinations_fail_loudly(bad, words):
    with pytest.raises((NotImplementedError, ValueError), match=words):
        DFDraw(_df()).profile('y:x', bins=6, range=(0,6), **bad)


def test_B08_weighted_statistics_agree_with_direct_weighted_numpy():
    frame = _df()
    fig, ax, result = DFDraw(frame).profile(
        'y:x', bins=6, range=(0,6), weights='w', return_data=True,
        stat_list=['mean', 'std', 'sem'])
    try:
        for i in range(6):
            subset = frame.loc[(frame.x >= i) & (frame.x < i+1)]
            y, w = subset.y.to_numpy(), subset.w.to_numpy()
            mean = np.sum(w*y)/np.sum(w)
            std = np.sqrt(np.sum(w*(y-mean)**2)/np.sum(w))
            n_eff = np.sum(w)**2/np.sum(w**2)
            expected = {'mean': mean, 'std': std, 'sem': std/np.sqrt(n_eff)}
            for stat, value in expected.items():
                np.testing.assert_allclose(result['statistic_curves'][stat][i], value,
                                           atol=1e-12, rtol=1e-12)
    finally:
        plt.close(fig)


def test_B09_refusal_never_allocates_a_new_figure(monkeypatch):
    called = []
    real = plt.subplots
    def monitored(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)
    monkeypatch.setattr(plt, 'subplots', monitored)
    with pytest.raises(NotImplementedError, match='group'):
        DFDraw(_df()).profile('y:x', group_by='g', stat_list=['median'])
    assert called == []
