"""PHASE_13_107_DF S2 - histogram X-quantile markers and one consistent
histogram normalization.

Plain-language background
-------------------------
PHASE_13_107 adds ``quantiles=[...]`` to ``hist()`` / ``draw(type='hist')``:
one vertical line per requested level for every drawn histogram curve, and
the X positions in ``stats['hist_quantiles']``.  The quantiles of a curve
use **all its selected finite values**, also those outside ``range=`` or the
bin edges (Architect decision D107-3).  Without physical weights the
estimator is NumPy ``linear``; with ``weights=`` it is W1 ``inverted_cdf``
(default, NumPy's weighted method) or W2 ``linear`` (Julia StatsBase
definition, equal to NumPy ``linear`` for equal weights).  Decisions D-1..D-9
of the S0 report (2026-11-10) apply.

Category Q (markers): the S0 inventory found 26 histogram routes; all of
them refused ``quantiles=`` before.  Every route is a test case here, each
checked against an independent oracle computed from the raw rows of each
drawn curve (NumPy, or this module's own reference kernels for weights).

Category N (normalization, S1): the bars and the error bars / returned bin
data were normalized in two places with different rules:
N1 density with a range: error bars at the in-range fraction of the bars;
N2 grouped hist_norm='density': area < 1 (all rows, mean bin width);
N3 grouped norm='density': error bars at raw counts;
N4 grouped norm='probability': bars were raw counts;
N5 weighted norm='probability': error bars divided by sum(w), bars by n.
One owner now computes both; every route must give error bars on the bars
and unit area for density.

Also D-8: a datetime column without ``time_format`` failed; it now uses the
automatic date axis; and grouped time histograms drew their bars in raw
nanoseconds / epoch seconds while the bins were date numbers (bars far
outside the axis); groups now use the same conversion owner.  And F: found while testing the 3-D route, one
infinite value made the shared range of every 3-D facet Figure infinite
(hist, scatter and profile failed); the shared range now uses finite values.

Real-data review of the S2 candidate (categories L, Y, T): on faceted
histograms with markers every cell repeated the same quantile legend and
covered the data.  The faceted histogram legend now has the facet
coordinator as its one owner, as profiles do: an explicit
``legend='shared'`` is composed from all cells and placed above the grid
(before, it was added afterwards over the bottom row, and 3-D facets
failed), and D-10 makes that shared legend the default for faceted
histograms with markers.  Grouped ``hist_norm=`` / the top panel of
``normalize=`` drew density bars under the Y title "Count".  New option
``quantile_text=`` ('level', 'value', 'both') writes text on each line.

``test_RED_*`` fail on the PHASE_13_108 END code (abb7e8e6); ``test_CTL_*``
pass before and after; ``test_UNIT_*`` test the new kernel module.

Seat: Fable5_1:draw-coder, 2026-11-10.
"""
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Polygon

from dfextensions.dfdraw.drawer import DFDraw

LEVELS = [0.05, 0.101, 0.5, 0.9]


def _frame(n=6000, seed=10741):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.normal(0.0, 1.0, n),
        'y': rng.normal(1.0, 2.0, n),
        'f': rng.integers(0, 3, n),
        'h': rng.integers(0, 2, n),
        'g': rng.integers(0, 2, n),
        'w': rng.uniform(0.2, 3.0, n),
    })
    df.loc[df.index[:7], 'x'] = np.nan
    df.loc[df.index[7:9], 'x'] = np.inf
    df['t'] = (pd.Timestamp('2026-01-01')
               + pd.to_timedelta(rng.uniform(0, 86400, n), unit='s'))
    df['e'] = (df.t - pd.Timestamp('1970-01-01')).dt.total_seconds()
    return df


# ------------------------------------------------- reference estimators


def ref_w1(x, w, levels):
    x, w = np.asarray(x, float), np.asarray(w, float)
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.lexsort((w, x))
    x, w = x[order], w[order]
    cum = np.cumsum(w) / w.sum()
    return [float(x[0]) if q == 0 else
            float(x[min(int(np.argmax(cum >= q - 1e-12)), len(x) - 1)])
            for q in levels]


def ref_w2(x, w, levels):
    x, w = np.asarray(x, float), np.asarray(w, float)
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.lexsort((w, x))
    x, w = x[order], w[order]
    if len(x) == 1:
        return [float(x[0])] * len(levels)
    cum = np.cumsum(w)
    return [float(np.interp(q * (cum[-1] - w[0]) + w[0], cum, x))
            for q in levels]


def oracle(x, w=None, method=None, levels=LEVELS):
    x = np.asarray(x, float)
    if w is None:
        x = x[np.isfinite(x)]
        return list(np.quantile(x, levels, method='linear'))
    w = np.asarray(w, float)
    keep = np.isfinite(x) & np.isfinite(w)
    kernel = ref_w2 if method == 'linear' else ref_w1
    return kernel(x[keep], w[keep], levels)


# ----------------------------------------------------------- helpers


def records(stats):
    """Every per-curve quantile record found in a stats structure."""
    out = []

    def visit(node):
        if isinstance(node, dict):
            hq = node.get('hist_quantiles')
            if isinstance(hq, dict) and 'population' in hq:
                out.append(hq)
            elif isinstance(hq, dict):
                out.extend(r for r in hq.values() if isinstance(r, dict))
            for key, value in node.items():
                if key != 'hist_quantiles':
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
    visit(stats)
    return out


def figures(fig):
    return list(fig) if isinstance(fig, (list, tuple)) else [fig]


def tagged_lines(fig):
    return [line for f in figures(fig) for ax in f.axes for line in ax.lines
            if getattr(line, '_dfdraw_quantile', None) is not None]


def _key(values):
    return tuple(round(float(v), 9) for v in values)


def _draw(df, expr, **kw):
    return DFDraw(df).draw(expr, type='hist', **kw)


# ---------------------------------------------------------- route table
# Each route: (expression, options, function(df) -> expected curves), where a
# curve is (x values, weights or None, method).

def _split(df, cols, value='x', w=None):
    return [(sub[value], None if w is None else sub[w], None)
            for _, sub in df.groupby(cols)]


ROUTES = {
    'single': ('x', {}, lambda d: [(d.x, None, None)]),
    'facet-1d': ('x', dict(facet_by='f'), lambda d: _split(d, 'f')),
    'facet-2d': ('x', dict(facet_by=['f', 'h']), lambda d: _split(d, ['f', 'h'])),
    'facet-3d': ('x', dict(facet_by=['f', 'h', 'g']),
                 lambda d: _split(d, ['f', 'h', 'g'])),
    'facet-binned': ('x', dict(facet_by='y', facet_by_bins=3),
                     lambda d: [(s.x, None, None) for _, s in
                                d.groupby(pd.cut(d.y, 3), observed=True)]),
    'group': ('x', dict(group_by='g'), lambda d: _split(d, 'g')),
    'group-binned': ('x', dict(group_by='y', group_by_bins=3),
                     lambda d: [(s.x, None, None) for _, s in
                                d.groupby(pd.cut(d.y, 3), observed=True)]),
    'group-x-facet': ('x', dict(group_by='g', facet_by='f'),
                      lambda d: _split(d, ['f', 'g'])),
    'legacy-facet': ('x', dict(group_by='g', facet=True), lambda d: _split(d, 'g')),
    'stacked': ('x', dict(group_by='g', stacked=True), lambda d: _split(d, 'g')),
    'vector': ('[x,y]', {}, lambda d: [(d.x, None, None), (d.y, None, None)]),
    'vector-x-facet': ('[x,y]', dict(facet_by='f'),
                       lambda d: _split(d, 'f') + _split(d, 'f', value='y')),
    'facet-by-vector': ('[x,y]', dict(facet_by='vector'),
                        lambda d: [(d.x, None, None), (d.y, None, None)]),
    'selection-vector': ('x', dict(selection_vector=['f==0', 'f==1'],
                                   vector_compose='outer'),
                         lambda d: [(d[d.f == 0].x, None, None),
                                    (d[d.f == 1].x, None, None)]),
    'selection-vector-x-facet': ('x', dict(selection_vector=['g==0', 'g==1'],
                                           vector_compose='outer', facet_by='f'),
                                 lambda d: _split(d, ['f', 'g'])),
    'weights-w1': ('x', dict(weights='w'), lambda d: [(d.x, d.w, None)]),
    'weights-w2': ('x', dict(weights='w', weighted_quantile_method='linear'),
                   lambda d: [(d.x, d.w, 'linear')]),
    'weights-vector': ('x', dict(weights_vector=['w', '1+0*w'],
                                 vector_compose='outer'),
                       lambda d: [(d.x, d.w, None), (d.x, 1 + 0 * d.w, None)]),
    'group-weights': ('x', dict(group_by='g', weights='w'),
                      lambda d: _split(d, 'g', w='w')),
    'facet-weights': ('x', dict(facet_by='f', weights='w'),
                      lambda d: _split(d, 'f', w='w')),
    'same': ('x', dict(same=True), lambda d: [(d.x, None, None)]),
    'cumulative': ('x', dict(cumulative=True), lambda d: [(d.x, None, None)]),
    'hist-errors': ('x', dict(hist_errors=True), lambda d: [(d.x, None, None)]),
    'fit': ('x', dict(fit='gaus'), lambda d: [(d.x, None, None)]),
    'normalize-ratio': ('x', dict(selection_vector=['g==0', 'g==1'],
                                  vector_compose='outer', normalize='ratio'),
                        lambda d: [(d[d.g == 0].x, None, None),
                                   (d[d.g == 1].x, None, None)]),
    'time-format': ('t', dict(time_format='%H'),
                    lambda d: [(mdates.date2num(d.t.to_numpy()), None, None)]),
    'time-auto': ('t', {}, lambda d: [(mdates.date2num(d.t.to_numpy()), None, None)]),
    'time-group': ('t', dict(group_by='g'),
                   lambda d: [(mdates.date2num(s.t.to_numpy()), None, None)
                              for _, s in d.groupby('g')]),
    'time-epoch-group': ('e', dict(group_by='g', time_format='%H'),
                         lambda d: [(mdates.date2num(s.t.to_numpy()), None, None)
                                    for _, s in d.groupby('g')]),
}


def _expected(route, df):
    out = []
    for x, w, method in ROUTES[route][2](df):
        out.append(_key(oracle(x, w, method)))
    return sorted(out)


@pytest.mark.parametrize('route', sorted(ROUTES))
def test_RED_Q01_every_route_draws_correct_markers_per_curve(route):
    df = _frame()
    expr, kw, _ = ROUTES[route]
    fig, axes, stats = _draw(df, expr, quantiles=LEVELS, **kw)
    try:
        recs = records(stats)
        got = sorted(_key(r['x']) for r in recs)
        assert got == _expected(route, df)
        # one drawn line per visible level of every curve, at the record x
        lines = tagged_lines(fig)
        assert len(lines) == sum(sum(r['marker_visible']) for r in recs)
        xs = {round(float(line.get_xdata()[0]), 9) for line in lines}
        visible = {round(float(v), 9) for r in recs
                   for v, shown in zip(r['x'], r['marker_visible']) if shown}
        assert xs == visible
    finally:
        plt.close('all')


# ------------------------------------------ full population and visibility


@pytest.mark.parametrize('display', [
    dict(range=(-0.5, 0.5)),
    dict(bins=[-0.4, -0.1, 0.0, 0.3, 0.6]),
    dict(range=(-0.5, 0.5), norm='density'),
    dict(bins=7, norm='probability'),
], ids=['range', 'edges', 'range-density', 'bins-probability'])
def test_RED_Q02_quantiles_use_all_values_not_the_shown_range(display):
    df = _frame()
    fig, _, stats = _draw(df, 'x', quantiles=LEVELS, **display)
    try:
        rec = stats['hist_quantiles']
        assert rec['x'] == pytest.approx(oracle(df.x), abs=1e-12)
        assert rec['n'] == int(np.isfinite(df.x).sum())
        lo, hi = ((display['bins'][0], display['bins'][-1])
                  if isinstance(display.get('bins'), list)
                  else stats['autorange_used'])
        assert rec['marker_visible'] == [lo <= v <= hi for v in rec['x']]
    finally:
        plt.close('all')


def test_RED_Q03_markers_never_change_the_axis_limits():
    df = _frame()
    plain, ax0, _ = _draw(df, 'x', range=(-0.5, 0.5))
    marked, ax1, _ = _draw(df, 'x', range=(-0.5, 0.5), quantiles=LEVELS)
    try:
        assert ax1.get_xlim() == pytest.approx(ax0.get_xlim())
    finally:
        plt.close('all')


# --------------------------------------------------- identity of the lines


def test_RED_Q04_names_identity_and_median():
    df = _frame()
    levels = [0.1, 0.100000001, 0.5, 0.5000000001, 0.9]
    fig, ax, stats = _draw(df, 'x', quantiles=levels)
    try:
        lines = tagged_lines(fig)
        assert [line._dfdraw_quantile for line in lines] == levels
        names = [line.get_label() for line in lines]
        assert names == ['q=10%', 'q=10.0000001%', 'median',
                         'q=50.00000001%', 'q=90%']
        assert lines[2].get_linestyle() == '-'
        patterns = {str(line._unscaled_dash_pattern) for line in lines}
        assert len(patterns) == len(lines)
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(names) <= set(legend)
    finally:
        plt.close('all')


def test_RED_Q05_one_legend_entry_per_level_for_several_curves():
    df = _frame()
    fig, ax, stats = _draw(df, 'x', group_by='g', quantiles=[0.2, 0.8])
    try:
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend.count('q=20%') == 1 and legend.count('q=80%') == 1
        assert len(tagged_lines(fig)) == 4
    finally:
        plt.close('all')


def test_RED_Q06_repeated_level_drawn_once():
    df = _frame()
    fig, _, stats = _draw(df, 'x', quantiles=[0.5, 0.2, 0.5])
    try:
        assert stats['hist_quantiles']['levels'] == [0.5, 0.2]
        assert len(tagged_lines(fig)) == 2
    finally:
        plt.close('all')


# ----------------------------------------------------- weights and refusals


@pytest.mark.parametrize('norm', [None, 'probability', 'density'])
def test_RED_Q07_display_normalization_never_weights_the_quantiles(norm):
    df = _frame()
    fig, _, stats = _draw(df, 'x', quantiles=LEVELS, norm=norm)
    try:
        assert stats['hist_quantiles']['weighted'] is False
        assert stats['hist_quantiles']['x'] == pytest.approx(oracle(df.x),
                                                             abs=1e-12)
    finally:
        plt.close('all')


def test_RED_Q08_zero_mass_curve_has_no_quantiles():
    df = _frame()
    df.loc[df.f == 2, 'w'] = 0.0
    fig, _, stats = _draw(df, 'x', facet_by='f', weights='w', quantiles=LEVELS)
    try:
        cell = stats['per_group']['2']['hist_quantiles']
        assert all(np.isnan(cell['x'])) and cell['n_positive_weight'] == 0
        assert not any(cell['marker_visible'])
        other = stats['per_group']['0']['hist_quantiles']
        assert other['method'] == 'inverted_cdf' and other['weighted']
    finally:
        plt.close('all')


@pytest.mark.parametrize('kw,match', [
    (dict(quantiles=[0.5, True]), 'real numbers'),
    (dict(quantiles=[0.5, float('nan')]), r'\[0, 1\]'),
    (dict(quantiles=[1.5]), r'\[0, 1\]'),
    (dict(quantiles=[]), 'empty'),
    (dict(quantiles='0.5'), 'list'),
    (dict(quantiles=[0.5], weighted_quantile_method='linear'), 'weights'),
    (dict(quantiles=[0.5], weights='w', weighted_quantile_method='hazen'),
     'inverted_cdf'),
    (dict(weights='w', weighted_quantile_method='linear'), 'quantiles'),
    (dict(quantiles=[0.5], weights='w - 2.0'), 'negative'),
    (dict(quantiles=[0.5], quantile_text='percent'), "'level', 'value'"),
    (dict(quantile_text='level'), 'quantiles'),
], ids=['bool', 'nan', 'above-1', 'empty', 'string', 'method-no-weights',
        'unknown-method', 'method-no-quantiles', 'negative-weights',
        'unknown-text', 'text-no-quantiles'])
def test_RED_Q09_refusals_before_any_figure(kw, match):
    df = _frame()
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match=match):
        _draw(df, 'x', **kw)
    assert set(plt.get_fignums()) == before


def test_CTL_Q10_hist_without_quantiles_unchanged():
    df = _frame()
    fig, ax, stats = _draw(df, 'x')
    try:
        assert 'hist_quantiles' not in stats
        assert tagged_lines(fig) == []
        assert ax.get_legend() is None
    finally:
        plt.close('all')


# ------------------------------------------------------- kernel (UNIT)

TRUTH = [
    ([0, 10], [1, 1], [0, 0, 0, 10, 10], [0, 2.5, 5, 7.5, 10]),
    ([0, 10], [1, 3], [0, 0, 10, 10, 10], [0, 2.5, 5, 7.5, 10]),
    ([0, 10, 20, 30], [1, 1, 1, 1], [0, 0, 10, 20, 30], [0, 7.5, 15, 22.5, 30]),
    ([0, 10, 20], [1, 1, 5], [0, 10, 20, 20, 20], [0, 11, 14, 17, 20]),
    ([0, 10, 20], [5, 1, 1], [0, 0, 0, 10, 20], [0, 5, 10, 15, 20]),
    ([0, 0, 10, 10], [.5, .5, .5, .5], [0, 0, 0, 10, 10], [0, 0, 5, 10, 10]),
    ([0, 0, 10], [1, 3, 1], [0, 0, 0, 0, 10], [0, 0, 0, 0, 10]),
    ([5], [2], [5] * 5, [5] * 5),
    ([0, 10, 20], [1, 0, 1], [0, 0, 0, 20, 20], [0, 5, 10, 15, 20]),
]


@pytest.mark.parametrize('x,w,w1,w2', TRUTH, ids=[f'case{i}' for i in range(len(TRUTH))])
def test_UNIT_Q11_truth_table(x, w, w1, w2):
    from dfextensions.dfdraw.plots._hist_quantiles import curve_quantiles
    q = [0, 0.25, 0.5, 0.75, 1]
    assert curve_quantiles(x, w, q, 'inverted_cdf')['x'] == pytest.approx(w1)
    assert curve_quantiles(x, w, q, 'linear')['x'] == pytest.approx(w2)


def test_UNIT_Q12_invariants():
    from dfextensions.dfdraw.plots._hist_quantiles import curve_quantiles
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    w = rng.uniform(0.1, 3.0, 300)
    x[5] = x[6]
    q = [0.0, 0.1, 0.37, 0.5, 0.9, 1.0]
    perm = rng.permutation(300)
    for method in ('inverted_cdf', 'linear'):
        base = curve_quantiles(x, w, q, method)['x']
        assert curve_quantiles(x[perm], 7.3 * w[perm], q, method)['x'] == \
            pytest.approx(base, abs=1e-12)
    assert curve_quantiles(x, np.ones(300), q, 'linear')['x'] == \
        pytest.approx(np.quantile(x, q), abs=1e-12)
    assert curve_quantiles(x, w, q, 'inverted_cdf')['x'] == \
        pytest.approx(ref_w1(x, w, q), abs=0)


# --------------------------------------------- normalization (category N)


def _outlines(ax):
    out = []
    for p in ax.patches:
        if isinstance(p, Polygon):
            v = p.get_xy()
            out.append([(min(a[0], b[0]), max(a[0], b[0]), a[1])
                        for a, b in zip(v[:-1], v[1:])
                        if a[1] == b[1] and a[0] != b[0]])
    return out


def _errorbar_centres(ax):
    out = []
    for c in ax.containers:
        if type(c).__name__ != 'ErrorbarContainer':
            continue
        if c.lines[0] is not None:
            out.extend(zip(*c.lines[0].get_data()))
        else:
            out.extend((s[0][0], 0.5 * (s[0][1] + s[1][1]))
                       for s in c.lines[2][0].get_segments())
    return out


def _measure(fig):
    """(areas, sums, error-bar/bar ratios) of every drawn histogram."""
    areas, sums, ratios = [], [], []
    for ax in [a for f in figures(fig) for a in f.axes if a.patches]:
        polys = _outlines(ax)
        areas += [sum((x1 - x0) * h for x0, x1, h in p) for p in polys]
        sums += [sum(h for _, _, h in p) for p in polys]
        for xc, yc in _errorbar_centres(ax):
            hs = [h for p in polys for x0, x1, h in p if x0 <= xc <= x1 and h > 0]
            if hs:
                ratios.append(yc / min(hs, key=lambda h: abs(h - yc)))
    return areas, sums, ratios


EDGES = [-3, -1, -0.5, 0, 0.5, 1, 3]
DENSITY_ROUTES = {
    'N1-range': dict(norm='density', range=(-1, 1)),
    'N1-edges': dict(norm='density', bins=EDGES),
    'N1-weights-range': dict(norm='density', weights='w', range=(-1, 1)),
    'N1-facet-range': dict(norm='density', facet_by='f', range=(-1, 1)),
    'N2-group-hist-norm-range': dict(hist_norm='density', group_by='g', range=(-1, 1)),
    'N2-group-hist-norm-edges': dict(hist_norm='density', group_by='g', bins=EDGES),
    'N3-group-norm': dict(norm='density', group_by='g'),
    'N3-group-norm-range': dict(norm='density', group_by='g', range=(-1, 1)),
}


@pytest.mark.parametrize('route', sorted(DENSITY_ROUTES))
def test_RED_N01_density_unit_area_and_error_bars_on_bars(route):
    df = _frame()
    fig, _, _ = _draw(df, 'x', hist_errors=True, **DENSITY_ROUTES[route])
    try:
        areas, _, ratios = _measure(fig)
        assert areas and all(a == pytest.approx(1.0, abs=1e-9) for a in areas)
        assert ratios and all(r == pytest.approx(1.0, abs=1e-9) for r in ratios)
    finally:
        plt.close('all')


PROBABILITY_ROUTES = {
    'N4-group-norm': (dict(norm='probability', group_by='g'), 1.0),
    'N4-group-norm-range': (dict(norm='probability', group_by='g', range=(-1, 1)), None),
    'N5-weights': (dict(norm='probability', weights='w'), None),
    'N5-weights-range': (dict(norm='probability', weights='w', range=(-1, 1)), None),
}


@pytest.mark.parametrize('route', sorted(PROBABILITY_ROUTES))
def test_RED_N02_probability_error_bars_on_bars(route):
    df = _frame()
    kw, total = PROBABILITY_ROUTES[route]
    fig, _, _ = _draw(df, 'x', hist_errors=True, **kw)
    try:
        _, sums, ratios = _measure(fig)
        assert ratios and all(r == pytest.approx(1.0, abs=1e-9) for r in ratios)
        if total is not None:
            assert all(s == pytest.approx(total, abs=1e-9) for s in sums)
    finally:
        plt.close('all')


def test_RED_N03_density_heights_equal_numpy_in_range():
    """Independent oracle: drawn heights = numpy density of the in-range
    rows, and the returned bin data (normalize dispatcher) equal them."""
    df = _frame()
    fig, ax, stats = _draw(df, 'x', norm='density', range=(-1, 1), bins=8,
                           hist_errors=True)
    try:
        x = df.x.to_numpy()
        x = x[np.isfinite(x)]
        expected, edges = np.histogram(x, bins=8, range=(-1, 1), density=True)
        heights = [h for x0, x1, h in _outlines(ax)[0] if h > 0]
        assert heights == pytest.approx(list(expected), rel=1e-12)
        centres = sorted(_errorbar_centres(ax))
        assert [c[1] for c in centres] == pytest.approx(
            [h for h in expected if h > 0], rel=1e-12)
    finally:
        plt.close('all')


@pytest.mark.parametrize('kw', [dict(norm='probability'), dict(),
                                dict(hist_norm='probability', group_by='g'),
                                dict(norm='density')],
                         ids=['probability', 'count', 'group-hist-norm-prob',
                              'density-full'])
def test_CTL_N04_already_consistent_routes_unchanged(kw):
    df = _frame()
    fig, _, _ = _draw(df, 'x', hist_errors=True, **kw)
    try:
        _, _, ratios = _measure(fig)
        assert ratios and all(r == pytest.approx(1.0, abs=1e-9) for r in ratios)
    finally:
        plt.close('all')


def test_RED_D08_datetime_without_time_format_draws():
    df = _frame()
    fig, ax, stats = _draw(df, 't')
    try:
        assert stats['n'] == len(df)
        assert isinstance(ax.xaxis.get_major_formatter(),
                          mdates.AutoDateFormatter)
    finally:
        plt.close('all')


@pytest.mark.parametrize('kind,expr', [('hist', 'x'), ('scatter', 'y:x'),
                                       ('profile', 'y:x')])
def test_RED_F01_3d_facets_ignore_infinite_values_in_the_shared_range(kind, expr):
    df = _frame()
    fig, axes, stats = DFDraw(df).draw(expr, type=kind, facet_by=['f', 'h', 'g'])
    try:
        x = df.x.to_numpy()
        x = x[np.isfinite(x)]
        for ax in [a for f in figures(fig) for a in f.axes if a.get_visible()
                   and (a.patches or a.collections or a.lines)]:
            lo, hi = ax.get_xlim()
            assert np.isfinite(lo) and np.isfinite(hi)
            assert lo <= x.min() + 1e-9 and hi >= x.max() - 1e-9
    finally:
        plt.close('all')


@pytest.mark.parametrize('expr,kw', [('t', dict(group_by='g')),
                                     ('e', dict(group_by='g', time_format='%H')),
                                     ('t', dict(group_by='g', stacked=True))],
                         ids=['datetime', 'epoch-time-format', 'stacked'])
def test_RED_D09_grouped_time_bars_inside_the_date_axis(expr, kw):
    df = _frame()
    fig, ax, _ = _draw(df, expr, **kw)
    try:
        lo, hi = mdates.date2num(df.t.min()), mdates.date2num(df.t.max())
        xs = [v for p in ax.patches for v in np.asarray(p.get_xy())[:, 0]] \
            if all(isinstance(p, Polygon) for p in ax.patches) else \
            [p.get_x() for p in ax.patches]
        assert xs and min(xs) >= lo - 1 and max(xs) <= hi + 1
        # every group actually draws its rows inside the bins (before, the
        # raw values fell outside the date-number bins: empty outlines)
        if all(isinstance(p, Polygon) for p in ax.patches):
            assert all(sum(h for _, _, h in poly) > 0 for poly in _outlines(ax))
        else:
            assert sum(p.get_height() for p in ax.patches) > 0
    finally:
        plt.close('all')


# ------------------------------------------- category L: default legend (D-10)
# Real-data review of the S2 candidate: on faceted histograms with markers
# every cell repeated the same quantile legend and covered the data.  D-10:
# the call that owns the Figure moves identical cell legends into ONE Figure
# legend by default (as AD-2/13.108.DF for profiles).  Explicit legend= /
# show_legend= win; differing cell legends stay in their cells; a cell or
# overlay call (ax=, same=True) never decides for the Figure.

QNAMES = ['q=10%', 'median', 'q=90%']
LEGEND_ROUTES = {
    'facet-1d': ('x', dict(facet_by='f'), QNAMES),
    'facet-2d': ('x', dict(facet_by=['f', 'h']), QNAMES),
    'facet-3d': ('x', dict(facet_by=['f', 'h', 'g']), QNAMES),
    'facet-binned': ('x', dict(facet_by='y', facet_by_bins=3), QNAMES),
    'facet-vector': ('[x,y]', dict(facet_by='vector'), QNAMES),
    'legacy-facet': ('x', dict(group_by='g', facet=True), QNAMES),
    'facet-weights': ('x', dict(facet_by='f', weights='w'), QNAMES),
    'group-x-facet': ('x', dict(facet_by='f', group_by='g'),
                      ['0', '1'] + QNAMES),
    'group-x-facet-2d': ('x', dict(facet_by=['f', 'h'], group_by='g'),
                         ['0', '1'] + QNAMES),
    'vector-x-facet': ('[x,y]', dict(facet_by='f'), ['x', 'y'] + QNAMES),
}


def _cell_legends(fig):
    return [[t.get_text() for t in ax.get_legend().get_texts()]
            for f in figures(fig) for ax in f.axes
            if ax.get_legend() is not None]


@pytest.mark.parametrize('route', sorted(LEGEND_ROUTES))
def test_RED_L01_faceted_markers_share_one_figure_legend(route):
    expr, kw, names = LEGEND_ROUTES[route]
    fig, _, _ = _draw(_frame(), expr, quantiles=[0.1, 0.5, 0.9], **kw)
    try:
        assert _cell_legends(fig) == []
        for one in figures(fig):
            assert len(one.legends) == 1
            assert [t.get_text() for t in one.legends[0].get_texts()] == names
    finally:
        plt.close('all')


def test_RED_L02_grouped_legend_lists_groups_before_levels():
    fig, ax, _ = _draw(_frame(), 'x', group_by='g', quantiles=[0.1, 0.5, 0.9])
    try:
        assert _cell_legends(fig) == [['0', '1'] + QNAMES]
    finally:
        plt.close('all')


def _cell_call(df):
    fig, axes = plt.subplots(1, 2)
    for ax, (_, sub) in zip(axes, df.groupby('g')):
        DFDraw(sub).draw('x', type='hist', ax=ax, quantiles=[0.1, 0.5, 0.9])
    return fig


@pytest.mark.parametrize('case', ['legend-all', 'show-legend-false',
                                  'differing-cells', 'single',
                                  'cell-calls-with-ax'])
def test_RED_L03_where_the_default_does_not_apply(case):
    """Markers are drawn, but D-10 must not move the legends: explicit
    choice, differing cells, one axis, or calls that do not own the Figure.
    (RED only because quantiles= was refused on the old code.)"""
    df = _frame()
    q = dict(quantiles=[0.1, 0.5, 0.9])
    try:
        if case == 'legend-all':
            fig, _, _ = _draw(df, 'x', facet_by='f', legend='all', **q)
            assert len(_cell_legends(fig)) == 3 and not fig.legends
        elif case == 'show-legend-false':
            fig, _, _ = _draw(df, 'x', facet_by='f', show_legend=False, **q)
            assert _cell_legends(fig) == [] and not fig.legends
        elif case == 'differing-cells':
            # facet-local group intervals: the cell legends differ
            fig, _, _ = _draw(df, 'x', facet_by='f', group_by='y',
                              group_by_bins=3, **q)
            cells = _cell_legends(fig)
            assert len(cells) == 3 and len({tuple(c) for c in cells}) > 1
            assert not fig.legends
        elif case == 'single':
            fig, _, _ = _draw(df, 'x', **q)
            assert _cell_legends(fig) == [QNAMES] and not fig.legends
        else:
            fig = _cell_call(df)
            assert _cell_legends(fig) == [QNAMES, QNAMES]
            assert not fig.legends
    finally:
        plt.close('all')


@pytest.mark.parametrize('kw', [dict(facet_by='f', group_by='g'),
                                dict(facet_by=['f', 'h'], group_by='g')],
                         ids=['1d', '2d'])
def test_CTL_L04_without_quantiles_legends_unchanged(kw):
    fig, _, _ = _draw(_frame(), 'x', **kw)
    try:
        assert len(_cell_legends(fig)) == len(fig.axes) - len(
            [a for a in fig.axes if not a.get_visible()])
        assert not fig.legends
    finally:
        plt.close('all')


# ---------------------------------------- category Y: Y title = drawn norm
# Same review: grouped hist_norm='density' and the top panel of
# normalize='ratio' with norm='density' drew density bars under the Y title
# "Count".  The Y title names the normalization the bars use (D-9 category).

@pytest.mark.parametrize('kw,title', [
    (dict(group_by='g', hist_norm='density'), 'Density'),
    (dict(group_by='g', hist_norm='probability'), 'Probability'),
    (dict(group_by='g', norm='probability', hist_norm='density'), 'Density'),
    (dict(selection_vector=['g==0', 'g==1'], vector_compose='outer',
          normalize='ratio', norm='density'), 'Density'),
    (dict(selection_vector=['g==0', 'g==1'], vector_compose='outer',
          normalize='ratio', norm='probability'), 'Probability'),
], ids=['grouped-hist-norm-density', 'grouped-hist-norm-probability',
        'hist-norm-wins', 'normalize-top-density', 'normalize-top-probability'])
def test_RED_Y01_y_title_names_the_drawn_normalization(kw, title):
    fig, _, _ = _draw(_frame(), 'x', **kw)
    try:
        assert figures(fig)[0].axes[0].get_ylabel() == title
    finally:
        plt.close('all')


@pytest.mark.parametrize('kw,title', [
    (dict(), 'Count'), (dict(norm='density'), 'Density'),
    (dict(norm='probability'), 'Probability'),
    (dict(group_by='g'), 'Count'), (dict(group_by='g', norm='density'), 'Density'),
    (dict(selection_vector=['g==0', 'g==1'], vector_compose='outer',
          normalize='ratio'), 'Count'),
], ids=['count', 'density', 'probability', 'grouped-count', 'grouped-norm-density',
        'normalize-top-count'])
def test_CTL_Y02_y_titles_already_right(kw, title):
    fig, _, _ = _draw(_frame(), 'x', **kw)
    try:
        assert figures(fig)[0].axes[0].get_ylabel() == title
    finally:
        plt.close('all')


# ------------------------- category L: one owner, placement, 3-D legend modes

def _extent(artist, renderer):
    return artist.get_window_extent(renderer)


def _legend_collisions(fig):
    """Cell elements (frame, title, axis titles, tick labels) that a Figure
    legend covers."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    for legend in fig.legends:
        box = _extent(legend, r)
        for i, ax in enumerate(fig.axes):
            if not ax.get_visible():
                continue
            parts = [('frame', ax)]
            parts += [(n, t) for n, t in (('title', ax.title),
                                          ('xlabel', ax.xaxis.label),
                                          ('ylabel', ax.yaxis.label))
                      if t.get_visible() and t.get_text()]
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                parts += [('tick', tick.label1) for tick in axis.get_major_ticks()
                          if tick.label1.get_visible() and tick.label1.get_text()
                          and lo <= tick.get_loc() <= hi]
            hits += [f'ax{i}.{n}' for n, a in parts if box.overlaps(_extent(a, r))]
    return hits


@pytest.mark.parametrize('kw', [
    dict(facet_by='f', quantiles=[0.1, 0.5, 0.9]),
    dict(facet_by=['f', 'h'], quantiles=[0.1, 0.5, 0.9]),
    dict(facet_by=['f', 'h', 'g'], quantiles=[0.1, 0.5, 0.9]),
    dict(facet_by='f', group_by='g', quantiles=[0.1, 0.5, 0.9]),
    dict(facet_by='f', group_by='g', legend='shared'),
    dict(facet_by=['f', 'h'], group_by='g', legend='shared'),
], ids=['default-1d', 'default-2d', 'default-3d', 'default-group-x-facet',
        'explicit-shared-1d', 'explicit-shared-2d'])
def test_RED_L04_figure_legend_covers_no_cell(kw):
    """The shared legend is placed by the facet coordinator above the grid.
    Before, an explicit legend='shared' was added afterwards over the
    bottom row (frame, tick labels, axis title)."""
    fig, _, _ = _draw(_frame(), 'x', auto_title=True, **kw)
    try:
        for one in figures(fig):
            assert len(one.legends) == 1
            assert _legend_collisions(one) == []
    finally:
        plt.close('all')


def test_RED_L05_explicit_shared_names_every_group_of_every_cell():
    """The composer collects the entries of all cells; before, only the
    first cell with a legend was read."""
    df = _frame()
    df = df[~((df.f == 0) & (df.g == 1))]      # group 1 absent in cell f=0
    fig, _, _ = _draw(df, 'x', facet_by='f', group_by='g', legend='shared')
    try:
        assert [t.get_text() for t in fig.legends[0].get_texts()] == ['0', '1']
    finally:
        plt.close('all')


@pytest.mark.parametrize('mode,figure_legends,cell_legends', [
    ('shared', 1, 0), ('none', 0, 0), ('first', 0, 1)])
def test_RED_L06_3d_facets_honour_every_legend_mode(mode, figure_legends,
                                                     cell_legends):
    """3-D facets return one Figure per page; the legend mode applies to
    each page (before, legend='shared'/'none'/'first' failed on the list
    of Figures).  Groups give every cell a legend without quantiles."""
    df = _frame()
    df = df[np.isfinite(df.x)].copy()    # separate from defect F01
    df['k'] = np.arange(len(df)) % 2
    fig, _, _ = _draw(df, 'x', facet_by=['f', 'h', 'g'], group_by='k',
                      legend=mode)
    try:
        assert len(fig) == 2
        for one in fig:
            assert len(one.legends) == figure_legends
            assert sum(ax.get_legend() is not None
                       for ax in one.axes) == cell_legends
    finally:
        plt.close('all')


# ------------------------------------------- category T: quantile_text=

def _texts(fig):
    return [t for f in figures(fig) for ax in f.axes for t in ax.texts
            if getattr(t, '_dfdraw_quantile_text', None) is not None]


def _expected_text(q, x, mode, time_axis=False):
    from dfextensions.dfdraw.plots._numeric_labels import quantile_percent_labels
    level = 'median' if q == 0.5 else quantile_percent_labels(LEVELS)[q]
    value = (mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S') if time_axis
             else f'{x:.4g}')
    return {'level': level, 'value': value, 'both': f'{level}: {value}'}[mode]


@pytest.mark.parametrize('mode', ['level', 'value', 'both'])
@pytest.mark.parametrize('route', ['single', 'group', 'facet-1d',
                                   'facet-by-vector', 'weights-w2'])
def test_RED_T01_one_text_per_drawn_line(route, mode):
    expr, kw, _ = ROUTES[route]
    fig, _, _ = _draw(_frame(), expr, quantiles=LEVELS, quantile_text=mode,
                      **kw)
    try:
        for one in figures(fig):
            for ax in one.axes:
                lines = [l for l in ax.lines
                         if getattr(l, '_dfdraw_quantile', None) is not None]
                texts = [t for t in ax.texts
                         if getattr(t, '_dfdraw_quantile_text', None) is not None]
                assert len(texts) == len(lines)
                for line, text in zip(lines, texts):
                    q = line._dfdraw_quantile
                    assert text._dfdraw_quantile_text == q
                    assert text.get_text() == _expected_text(
                        q, line.get_xdata()[0], mode)
                    assert text.get_color() == line.get_color()
        assert _texts(fig)
    finally:
        plt.close('all')


def test_RED_T02_value_text_on_a_time_axis_is_a_date():
    fig, ax, stats = _draw(_frame(), 't', quantiles=[0.1, 0.5, 0.9],
                           quantile_text='value')
    try:
        record = stats['hist_quantiles']
        assert [t.get_text() for t in _texts(fig)] == [
            mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S') for x in record['x']]
    finally:
        plt.close('all')


def test_RED_T03_no_text_for_a_level_outside_the_shown_range():
    fig, ax, stats = _draw(_frame(), 'x', quantiles=LEVELS, range=(-1, 1),
                           quantile_text='both')
    try:
        shown = [q for q, v in zip(LEVELS, stats['hist_quantiles']['marker_visible'])
                 if v]
        assert 0 < len(shown) < len(LEVELS)
        assert [t._dfdraw_quantile_text for t in _texts(fig)] == shown
    finally:
        plt.close('all')


def test_RED_T04_texts_of_two_curves_on_one_axis_take_opposite_sides():
    fig, ax, _ = _draw(_frame(), 'x', group_by='g', quantiles=[0.5],
                       quantile_text='level')
    try:
        texts = _texts(fig)
        assert len(texts) == 2
        assert {t.get_horizontalalignment() for t in texts} == {'left', 'right'}
    finally:
        plt.close('all')


def test_RED_T05_default_draws_no_text():
    fig, _, _ = _draw(_frame(), 'x', facet_by='f', quantiles=LEVELS)
    try:
        assert _texts(fig) == []
    finally:
        plt.close('all')


@pytest.mark.parametrize('mode,figure_legends,cell_legends', [
    ('shared', 1, 0), ('none', 0, 0), ('first', 0, 1)])
@pytest.mark.parametrize('kind', ['profile', 'scatter'])
def test_RED_L07_3d_legend_modes_profile_and_scatter(kind, mode,
                                                     figure_legends,
                                                     cell_legends):
    """Same defect as L06 for the other kinds that draw cell legends on
    3-D pages: one owner of legend modes now applies them per page."""
    df = _frame()
    df = df[np.isfinite(df.x)].copy()
    df['k'] = np.arange(len(df)) % 2
    fig, _, _ = DFDraw(df).draw('y:x', type=kind, facet_by=['f', 'h', 'g'],
                                group_by='k', legend=mode)
    try:
        assert len(fig) == 2
        for one in fig:
            assert len(one.legends) == figure_legends
            assert sum(ax.get_legend() is not None
                       for ax in one.axes) == cell_legends
    finally:
        plt.close('all')


# ------------- category L: shared presentation of histogram facets (S2b review)
# Real-data review of S2b: with the shared legend the cells are laid out by
# the coordinator's shared presentation (narrower cells, legend band), and
# the long expression repeated as X title in every cell overlapped the
# neighbouring cells.  Whenever a histogram Figure legend is shared, the
# repeated axis titles become one Figure title (D108-A rule, as profiles).

LONG = '(qTotExpR0-qTotExpR0_TaylorV3FitFull)/qTotExpR0_TaylorV3MADFitFull'


def _long_frame():
    df = _frame()
    df = df[np.isfinite(df.x)].copy()
    df['qTotExpR0'] = df.x
    df['qTotExpR0_TaylorV3FitFull'] = 0.1
    df['qTotExpR0_TaylorV3MADFitFull'] = 1.0
    df['F'] = df.f * 2 + df.h
    return df


def _layout_collisions(fig):
    """Pairs of elements of different cells (frame, titles, tick labels)
    or Figure elements (legends, Figure axis titles) that overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        items.append((f'ax{i}', 'frame', ax.get_window_extent(r)))
        for name, t in (('title', ax.title), ('xlabel', ax.xaxis.label),
                        ('ylabel', ax.yaxis.label)):
            if t.get_visible() and t.get_text():
                items.append((f'ax{i}', name, t.get_window_extent(r)))
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            items += [(f'ax{i}', 'tick', tick.label1.get_window_extent(r))
                      for tick in axis.get_major_ticks()
                      if tick.label1.get_visible() and tick.label1.get_text()
                      and lo <= tick.get_loc() <= hi]
    items += [('fig', f'legend{i}', L.get_window_extent(r))
              for i, L in enumerate(fig.legends)]
    for name in ('_supxlabel', '_supylabel'):
        t = getattr(fig, name, None)
        if t is not None and t.get_visible() and t.get_text():
            items.append(('fig', name, t.get_window_extent(r)))
    return sorted({(f'{o1}.{n1}', f'{o2}.{n2}')
                   for (o1, n1, b1), (o2, n2, b2) in itertools.combinations(items, 2)
                   if o1 != o2 and b1.overlaps(b2)})


@pytest.mark.parametrize('kw', [
    dict(facet_by='F', ncols=3, quantiles=[0.05, 0.5, 0.95]),
    dict(facet_by=['f', 'y'], facet_by_bins=[None, 3], quantiles=[0.05, 0.5, 0.95]),
    dict(facet_by='y', facet_by_bins=3, group_by='f', quantiles=[0.1, 0.5, 0.9],
         quantile_text='level'),
    dict(facet_by='F', ncols=3, group_by='g', legend='shared'),
], ids=['1d-default', '2d-default', 'group-x-facet-text', '1d-explicit-shared'])
def test_RED_L08_shared_histogram_presentation_has_one_x_title(kw):
    fig, _, _ = _draw(_long_frame(), LONG, bins=60, norm='density',
                      auto_title=True, **kw)
    try:
        assert fig._supxlabel is not None and fig._supxlabel.get_text() == LONG
        assert all(ax.get_xlabel() == '' for ax in fig.axes if ax.get_visible())
        assert _layout_collisions(fig) == []
    finally:
        plt.close('all')


def test_RED_L09_histogram_facets_without_legend_share_titles_too():
    """AD-6: the shared titles do not depend on the legend."""
    fig, _, _ = _draw(_long_frame(), LONG, bins=60, facet_by='F', ncols=3)
    try:
        assert fig._supxlabel.get_text() == LONG
        assert all(ax.get_xlabel() == '' for ax in fig.axes if ax.get_visible())
        assert not fig.legends
        assert _layout_collisions(fig) == []
    finally:
        plt.close('all')


# ---------------- category A: shared axis titles for every faceted kind (AD-6)
# Architect 2026-11-11: "In general for facet plots shared_xlabel=True,
# shared_ylabel=True should be default."  The D108-A rule (PHASE_13_108):
# a Figure X/Y title replaces the cell titles when every cell renders the
# same title; an explicit shared_xlabel=/shared_ylabel= wins.  Before, only
# profiles and the hist2d+profile overlay had it.

@pytest.mark.parametrize('facet', ['f', ['f', 'h'], ['f', 'h', 'g']],
                         ids=['1d', '2d', '3d'])
@pytest.mark.parametrize('kind,expr,ytitle', [
    ('hist', 'x', 'Count'), ('hist2d', 'y:x', 'y'), ('scatter', 'y:x', 'y')])
def test_RED_A01_every_faceted_kind_shares_identical_axis_titles(
        kind, expr, ytitle, facet):
    df = _frame()
    df = df[np.isfinite(df.x)].copy()
    fig, _, _ = DFDraw(df).draw(expr, type=kind, facet_by=facet)
    try:
        for one in figures(fig):
            assert one._supxlabel.get_text() == 'x'
            assert one._supylabel.get_text() == ytitle
            cells = [ax for ax in one.axes if ax.get_visible()
                     and ax.get_label() != '<colorbar>']
            assert all(ax.get_xlabel() == '' for ax in cells)
            assert not one.legends        # no default legend for these kinds
            assert _layout_collisions(one) == []
    finally:
        plt.close('all')


@pytest.mark.parametrize('kind,expr', [('hist', 'x'), ('scatter', 'y:x')])
@pytest.mark.parametrize('facet', ['f', ['f', 'h']], ids=['1d', '2d'])
def test_CTL_A02_explicit_false_keeps_cell_titles(kind, expr, facet):
    df = _frame()
    df = df[np.isfinite(df.x)].copy()
    fig, _, _ = DFDraw(df).draw(expr, type=kind, facet_by=facet,
                                shared_xlabel=False, shared_ylabel=False)
    try:
        sx = getattr(fig, '_supxlabel', None)
        assert sx is None or not sx.get_text()
        assert any(ax.get_xlabel() == 'x' for ax in fig.axes)
    finally:
        plt.close('all')
