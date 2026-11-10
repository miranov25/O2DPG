"""PHASE_13_108_DF S4 - numeric bins: raw-row membership oracle.

Plain-language background
-------------------------
PHASE_13_108 v0.2 (commit f2484556) gave every numeric group/facet bin a
distinct name.  The name is the grouping key, so before v0.2 two narrow bins
that printed the same ('0.000-0.000') were merged into one group or facet:
their rows and statistics were combined.  After v0.2 each bin is its own
group again, so in exactly those collision cases the number of groups, their
membership and their statistics change (by design; Architect decision
AD-5/13.108.DF).

The S3 tests only check that the names differ.  The independent v0.2 review
(GPT29, 2026-11-10) asked for an oracle that checks the *rows*: each group
or facet must contain exactly the raw rows that ``pd.cut`` / ``pd.qcut``
put in its original numeric interval, and its statistics must equal the
statistics of those rows.  The oracle does not use dfdraw's names to select
rows: it selects rows by interval and only parses a name to find which
interval it denotes (each printed edge must agree with the true edge to its
last printed decimal).

Routes (the category inventory of review section 5): group_by_bins and
group_by_quantiles without facets, with facet-local grouping, with global
grouping; facet_by_bins, facet_by_quantiles and facet_by='group_by'.  Each
route is checked on narrow data (names collided before v0.2: RED on the
v0.1 bytes) and on normal data (names never collided: control).

Seat: Fable5_1:draw-coder, 2026-11-10.
"""
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

N_BINS = 8
QUANTILES = [0.2, 0.5, 0.85]          # discrete: per-group quantile curves
X_BINS = 10
X_RANGE = (0.0, 1.0)
_NAME = re.compile(r'^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$')


def _frame(narrow, n=8000, seed=10841):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.uniform(0.0, 1.0, n),
        'f': rng.integers(0, 3, n),
        'm': rng.uniform(0.0, 0.0004 if narrow else 4.0, n),
    })
    df['y'] = rng.normal(0.0, 1.0, n) + 0.3 * df.f + 50.0 * df.m
    return df


def _classify(values, how):
    return (pd.cut(values, bins=N_BINS) if how == 'bins'
            else pd.qcut(values, q=N_BINS, duplicates='drop'))


def _name_to_interval(name, intervals):
    """The unique interval whose edges agree with the printed name."""
    match = _NAME.match(str(name))
    assert match, name
    found = []
    for interval in intervals:
        ok = True
        for text, edge in zip(match.groups(), (interval.left, interval.right)):
            decimals = len(text.split('.')[1]) if '.' in text else 0
            if abs(float(text) - edge) > 0.5 * 10.0 ** -decimals + 1e-12:
                ok = False
        if ok:
            found.append(interval)
    assert len(found) == 1, (name, found)
    return found[0]


def _reference_quantiles(rows):
    """Quantile curves of these rows alone, with the same x binning."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, _, stats = DFDraw(rows).draw('y:x', type='profile',
                                         quantiles=QUANTILES, bins=X_BINS,
                                         range=X_RANGE)
    plt.close(fig)
    return stats['quantiles_per_bin']


def _assert_groups_match(per_group, rows, how):
    """``per_group`` (name -> stats) holds exactly the rows of each interval."""
    classified = _classify(rows['m'], how)
    intervals = list(classified.cat.categories)
    assert len(per_group) == len(intervals), (sorted(per_group), intervals)
    seen = set()
    for name, stats in per_group.items():
        interval = _name_to_interval(name, intervals)
        assert interval not in seen, name
        seen.add(interval)
        members = rows[classified == interval]
        assert stats['n'] == len(members), name
        if 'quantiles_per_bin' in stats:
            expected = _reference_quantiles(members)
            for q in QUANTILES:
                np.testing.assert_allclose(
                    stats['quantiles_per_bin'][q], expected[q],
                    rtol=0, atol=1e-12, equal_nan=True)
        else:
            assert stats['mean_y'] == pytest.approx(members['y'].mean(),
                                                    rel=0, abs=1e-12)
            assert stats['std_y'] == pytest.approx(members['y'].std(ddof=0),
                                                   rel=1e-9, abs=1e-12)


def _draw(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DFDraw(df).draw('y:x', type='profile', bins=X_BINS,
                               range=X_RANGE, **kw)


@pytest.mark.parametrize('narrow', [True, False], ids=['narrow', 'normal'])
@pytest.mark.parametrize('how', ['bins', 'quantiles'])
def test_S4_01_group_by_unfaceted(narrow, how):
    df = _frame(narrow)
    fig, _, stats = _draw(df, group_by='m', quantiles=QUANTILES,
                          **{f'group_by_{how}': N_BINS})
    try:
        _assert_groups_match(stats['per_group'], df, how)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('narrow', [True, False], ids=['narrow', 'normal'])
def test_S4_02_group_by_facet_local(narrow):
    """Default scope: each facet cell classifies its own rows."""
    df = _frame(narrow)
    fig, _, stats = _draw(df, group_by='m', group_by_bins=N_BINS,
                          facet_by='f', quantiles=QUANTILES)
    try:
        for cell, cell_stats in stats['per_group'].items():
            rows = df[df.f.astype(str) == str(cell)]
            _assert_groups_match(cell_stats['per_group'], rows, 'bins')
    finally:
        plt.close(fig)


@pytest.mark.parametrize('narrow', [True, False], ids=['narrow', 'normal'])
def test_S4_03_group_by_global(narrow):
    """Global scope: intervals from all rows, then each cell's rows."""
    df = _frame(narrow)
    fig, _, stats = _draw(df, group_by='m', group_by_quantiles=N_BINS,
                          facet_by='f', group_by_scope='global',
                          quantiles=QUANTILES)
    try:
        classified = _classify(df['m'], 'quantiles')
        intervals = list(classified.cat.categories)
        for cell, cell_stats in stats['per_group'].items():
            in_cell = (df.f.astype(str) == str(cell)).to_numpy()
            per_group = cell_stats['per_group']
            for name, group_stats in per_group.items():
                interval = _name_to_interval(name, intervals)
                members = df[in_cell & (classified == interval).to_numpy()]
                assert group_stats['n'] == len(members), (cell, name)
                expected = _reference_quantiles(members)
                for q in QUANTILES:
                    np.testing.assert_allclose(
                        group_stats['quantiles_per_bin'][q], expected[q],
                        rtol=0, atol=1e-12, equal_nan=True)
            assert sum(g['n'] for g in per_group.values()) == in_cell.sum()
    finally:
        plt.close(fig)


@pytest.mark.parametrize('narrow', [True, False], ids=['narrow', 'normal'])
@pytest.mark.parametrize('how', ['bins', 'quantiles'])
def test_S4_04_facet_by_numeric_bins(narrow, how):
    df = _frame(narrow)
    fig, _, stats = _draw(df, facet_by='m', **{f'facet_by_{how}': N_BINS})
    try:
        _assert_groups_match(stats['per_group'], df, how)
        assert stats['n_groups'] == N_BINS
    finally:
        plt.close(fig)


@pytest.mark.parametrize('narrow', [True, False], ids=['narrow', 'normal'])
def test_S4_05_facet_by_group_by_bins(narrow):
    df = _frame(narrow)
    fig, _, stats = _draw(df, facet_by='group_by', group_by='m',
                          group_by_bins=N_BINS)
    try:
        _assert_groups_match(stats['per_group'], df, 'bins')
    finally:
        plt.close(fig)
