"""PHASE_13_107_DF S2e - fixes of the S2d code review (GPT27 summary,
2026-11-11, five reviewers).

Plain-language background
-------------------------
S2d (commit 6a150638) was committed by Architect decision with four
confirmed defects, fixed here:

F1  weighted ``inverted_cdf`` (W1) used a ``1 - 1e-12`` margin and could
    return the neighbouring value when q lay on or next to a step of the
    cumulative weight.  Now: NumPy's exact definition (first x whose
    cumulative weight share is >= q).  The reference here is exact rational
    arithmetic, independent of the production code (the S2d reference
    repeated the margin and could not see the defect).
F2  ``stacked=True`` with ``norm=`` / ``hist_norm=`` drew raw counts under
    a Density/Probability title.  Architect 2026-11-11: "We need stacked,
    normalized histograms", Matplotlib conventions, no crash:
      norm='density'      whole stack has unit area (Matplotlib);
      norm='probability'  whole stack sums to 1 (each row weighs 1/N);
      hist_norm=...       each group normalized on its own, then stacked.
F3  66 or more quantile levels raised StopIteration after the Figure
    existed (a fixed list of 64 line styles).  Now the styles repeat.
F4  the negative-weight check ran on all selected rows before sample=, so
    rows that are not drawn still refused the call.  Now only the drawn
    rows count (selected, sampled, finite X, inside a selection branch).
W   weighted ``norm='probability'``: each bin's share of the total weight
    (bars sum to 1); before, the bars summed to the mean weight (Architect
    2026-11-11: common-sense value).

``test_RED_*`` fail on S2d (6a150638); ``test_CTL_*`` pass on both.

Seat: Fable5_1:draw-coder, 2026-11-11.
"""
from fractions import Fraction

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw


def exact_w1(x, w, q):
    """First x (sorted) whose cumulative weight share is >= q, exactly."""
    pairs = sorted((float(a), float(b)) for a, b in zip(x, w) if b > 0)
    total = sum(Fraction(b) for _, b in pairs)
    target = Fraction(float(q)) * total
    cum = Fraction(0)
    for value, weight in pairs:
        cum += Fraction(weight)
        if cum >= target:
            return value
    return pairs[-1][0]


def _close():
    plt.close('all')


# ------------------------------------------------------------- F1: exact W1

# (x, w, q, expected): the S2d margin returned the previous value in these
W1_RED = {
    'just-above-half': ([1.0, 2.0], [1.0, 1.0], float(np.nextafter(0.5, 1.0)), 2.0),
    'tiny-tail-q1': ([1.0, 2.0], [1.0, 1e-13], 1.0, 2.0),
    'just-above-decimal-step': (list(range(10)), [1.0] * 10,
                                float(np.nextafter(0.3, 1.0)), 3.0),
}
# controls: right before and after
W1_CTL = {
    'exactly-half': ([1.0, 2.0], [1.0, 1.0], 0.5, 1.0),
    'decimal-on-step': (list(range(10)), [1.0] * 10, 0.3, 2.0),
    'zero-weight-first': ([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 0.0, 2.0),
}


def _check_w1(x, w, q, expected):
    from dfextensions.dfdraw.plots._hist_quantiles import curve_quantiles
    assert exact_w1(x, w, q) == expected          # the reference itself
    record = curve_quantiles(np.array(x), np.array(w), [q], 'inverted_cdf')
    assert record['x'] == [expected]
    # the same through the public histogram call
    df = pd.DataFrame({'x': x, 'w': w})
    _, _, stats = DFDraw(df).draw('x', type='hist', weights='w', quantiles=[q],
                                  bins=4)
    try:
        assert stats['hist_quantiles']['x'] == [expected]
    finally:
        _close()


@pytest.mark.parametrize('case', sorted(W1_RED))
def test_RED_E01_w1_exact_first_crossing(case):
    _check_w1(*W1_RED[case])


@pytest.mark.parametrize('case', sorted(W1_CTL))
def test_CTL_E01a_w1_unchanged_cases(case):
    _check_w1(*W1_CTL[case])


def test_RED_E01b_w1_equals_exact_reference_on_and_next_to_steps():
    """Random integer weights whose total is a power of two, so every step
    of the cumulative share is exact in binary: q on every step and on the
    float neighbours of each step.  (W1 follows NumPy's float64 share; with
    such weights it equals the exact rational first crossing.)"""
    from dfextensions.dfdraw.plots._hist_quantiles import curve_quantiles
    rng = np.random.default_rng(20261111)
    for trial in range(30):
        n = int(rng.integers(2, 12))
        x = rng.normal(size=n).round(3)
        w = rng.integers(1, 5, n).astype(float)
        w = np.append(w, 64.0 - w.sum())           # total 64: exact shares
        x = np.append(x, 9.0)
        share = np.cumsum(w) / w.sum()
        qs = sorted({float(v) for s in share[:-1] for v in
                     (s, np.nextafter(s, 0.0), np.nextafter(s, 1.0))})
        got = curve_quantiles(x, w, qs, 'inverted_cdf')['x']
        assert got == [exact_w1(x, w, q) for q in qs], trial


@pytest.mark.skipif(tuple(int(v) for v in np.__version__.split('.')[:2]) < (2, 0),
                    reason='np.quantile(weights=) needs NumPy >= 2')
def test_CTL_E01c_w1_equals_numpy_weighted_inverted_cdf():
    from dfextensions.dfdraw.plots._hist_quantiles import curve_quantiles
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = rng.normal(size=50)
        w = rng.uniform(0.1, 3.0, 50)
        qs = [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]
        assert curve_quantiles(x, w, qs, 'inverted_cdf')['x'] == [
            float(np.quantile(x, q, weights=w, method='inverted_cdf'))
            for q in qs]


# ----------------------------------------------------- F2: stacked + norm

def _stack_frame():
    rng = np.random.default_rng(11)
    n = 900
    return pd.DataFrame({'x': np.concatenate([rng.normal(-1, 1, 300),
                                              rng.normal(0.5, 0.7, 400),
                                              rng.normal(1.5, 1.5, 200)]),
                         'g': np.repeat([0, 1, 2], [300, 400, 200])})


EDGES = np.array([-4.0, -1.0, 0.0, 0.5, 4.0])     # unequal widths, clipping


def _expected_layers(df, how):
    widths = np.diff(EDGES)
    groups = [df[df.g == g].x.to_numpy() for g in (0, 1, 2)]
    counts = [np.histogram(d, bins=EDGES)[0].astype(float) for d in groups]
    n_all = sum(len(d) for d in groups)
    n_in_all = sum(c.sum() for c in counts)
    if how == 'norm-density':
        return [c / (n_in_all * widths) for c in counts], 'Density'
    if how == 'norm-probability':
        return [c / n_all for c in counts], 'Probability'
    if how == 'hist_norm-density':
        return [c / (c.sum() * widths) for c in counts], 'Density'
    if how == 'hist_norm-probability':
        return [c / len(d) for c, d in zip(counts, groups)], 'Probability'
    return counts, 'Count'


STACK_CASES = {
    'norm-probability': dict(norm='probability'),
    'hist_norm-density': dict(hist_norm='density'),
    'hist_norm-probability': dict(hist_norm='probability'),
    'hist_norm-wins': dict(norm='probability', hist_norm='density'),
}


@pytest.mark.parametrize('case', sorted(STACK_CASES))
def test_RED_E02_stacked_normalized_layers(case):
    df = _stack_frame()
    how = 'hist_norm-density' if case == 'hist_norm-wins' else case
    expected, title = _expected_layers(df, how)
    _, ax, _ = DFDraw(df).draw('x', type='hist', group_by='g', stacked=True,
                               bins=EDGES, histtype='bar',
                               **STACK_CASES[case])
    try:
        drawn = [np.array([p.get_height() for p in c]) for c in ax.containers]
        assert len(drawn) == 3
        for got, want in zip(drawn, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
        assert ax.get_ylabel() == title
    finally:
        _close()


@pytest.mark.parametrize('case', ['norm-density', 'count'])
def test_CTL_E02b_stacked_density_and_counts_unchanged(case):
    df = _stack_frame()
    kw = dict(norm='density') if case == 'norm-density' else {}
    expected, title = _expected_layers(df, case)
    _, ax, _ = DFDraw(df).draw('x', type='hist', group_by='g', stacked=True,
                               bins=EDGES, histtype='bar', **kw)
    try:
        drawn = [np.array([p.get_height() for p in c]) for c in ax.containers]
        for got, want in zip(drawn, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
        assert ax.get_ylabel() == title
    finally:
        _close()


# ------------------------------------------------------ F3: many levels

def _many_levels(n_levels):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({'x': rng.normal(size=2000)})
    levels = list(np.linspace(0.02, 0.98, n_levels))
    if 0.5 not in levels:
        levels[len(levels) // 2] = 0.5
    _, ax, stats = DFDraw(df).draw('x', type='hist', quantiles=levels)
    try:
        record = stats['hist_quantiles']
        assert len(record['x']) == n_levels
        lines = [l for l in ax.lines if getattr(l, '_dfdraw_quantile', None)
                 is not None]
        assert len(lines) == sum(record['marker_visible'])
        np.testing.assert_allclose(sorted(l.get_xdata()[0] for l in lines),
                                   sorted(np.quantile(df.x, levels)))
    finally:
        _close()


@pytest.mark.parametrize('n_levels', [66, 100])
def test_RED_E03_any_number_of_levels(n_levels):
    _many_levels(n_levels)


def test_CTL_E03a_64_levels_unchanged():
    _many_levels(64)


# ------------------------------------- F4: only the drawn rows are checked

def _weight_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame({'x': rng.normal(size=100), 'g': np.arange(100) % 3,
                         'w': rng.uniform(0.5, 2.0, 100)})


def test_RED_E04_negative_weight_in_a_row_not_sampled():
    df = _weight_frame()
    sampled = DFDraw(df)._apply_sampling(df, 10).index
    outside = [i for i in df.index if i not in set(sampled)][0]
    df.loc[outside, 'w'] = -1.0
    _, _, stats = DFDraw(df).draw('x', type='hist', weights='w', sample=10,
                                  quantiles=[0.5])
    try:
        rows = df.loc[sampled]
        assert stats['hist_quantiles']['n'] == 10
        assert stats['hist_quantiles']['x'] == [exact_w1(rows.x, rows.w, 0.5)]
    finally:
        _close()


@pytest.mark.parametrize('case', ['outside-every-branch', 'missing-x'])
def test_RED_E04b_negative_weight_in_a_row_not_drawn(case):
    df = _weight_frame()
    kw = dict(quantiles=[0.5], weights='w')
    if case == 'outside-every-branch':
        df.loc[df.index[df.g == 2][0], 'w'] = -1.0
        kw.update(selection_vector=['g==0', 'g==1'], vector_compose='outer')
    else:
        df.loc[3, 'x'] = np.nan
        df.loc[3, 'w'] = -1.0
    try:
        DFDraw(df).draw('x', type='hist', **kw)        # no refusal
    finally:
        _close()


def test_CTL_E04c_negative_weight_in_a_drawn_row_is_refused_before_figure():
    df = _weight_frame()
    sampled = DFDraw(df)._apply_sampling(df, 10).index
    df.loc[sampled[0], 'w'] = -1.0
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match='negative'):
        DFDraw(df).draw('x', type='hist', weights='w', sample=10,
                        quantiles=[0.5])
    assert set(plt.get_fignums()) == before


# --------------------------------- W: weighted probability = weight share

@pytest.mark.parametrize('range_', [None, (-1.0, 1.0)], ids=['full', 'range'])
def test_RED_E05_weighted_probability_is_the_share_of_the_total_weight(range_):
    df = _weight_frame()
    edges = np.linspace(-4, 4, 9) if range_ is None else np.linspace(-1, 1, 5)
    _, ax, _ = DFDraw(df).draw('x', type='hist', weights='w',
                               norm='probability', bins=edges,
                               histtype='bar', hist_errors=True)
    try:
        heights = np.array([p.get_height() for p in ax.containers[0]])
        sum_w, _ = np.histogram(df.x, bins=edges, weights=df.w)
        sum_w2, _ = np.histogram(df.x, bins=edges, weights=df.w ** 2)
        np.testing.assert_allclose(heights, sum_w / df.w.sum(), rtol=1e-12)
        if range_ is None:
            assert heights.sum() == pytest.approx(1.0, rel=1e-12)
        segments = [seg for c in ax.collections
                    for seg in getattr(c, 'get_segments', lambda: [])()]
        half = sorted(round(abs(s[1][1] - s[0][1]) / 2, 12) for s in segments)
        want = sorted(round(v, 12) for v, h in zip(np.sqrt(sum_w2) / df.w.sum(),
                                                   sum_w) if h > 0)
        assert half == want
        assert ax.get_ylabel() == 'Probability'
    finally:
        _close()
