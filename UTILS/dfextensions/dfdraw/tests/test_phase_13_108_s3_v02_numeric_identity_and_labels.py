"""PHASE_13_108_DF v0.2 (S3) - distinct numbers keep distinct names; shared
titles keep every line.

Plain-language background
-------------------------
The PHASE_13_108 v0.1 PRECOMMIT review (GPT29 consolidated, 2026-11-10,
4/4 REQUEST CHANGES) found two defect categories.  Both are fixed here for
the whole category, not only for the reviewers' examples (working style v4
section 10.5).

Category Q - "a number written as text must not lose its identity"
    (review finding P1-108-01).  Mechanism: dfdraw rounds a number to a
    fixed short text and then treats equal text as equal identity.  Members:

    * quantile labels on every profile route (single, vector, grouped,
      faceted 1-D/2-D, hist2d+profile overlay) and the shared Figure legend,
      which de-duplicated by text, so q=0.100000001 and q=0.100000002 became
      one "q=10%" entry;
    * the "median" name, given to any level within 1e-9 of 0.5, and the
      band-mode detection that used the same tolerance;
    * the quantile list in the automatic title ('q [0.1, 0.1]');
    * numeric bin names of group_by_bins/quantiles and facet_by_bins/
      quantiles ('0.000-0.000'): the name is the grouping key, so bins with
      one name were merged into one group or facet (data merged), and the
      PHASE_13_95 global route refused them instead.

    Fix: one formatter module (plots/_numeric_labels.py) that keeps today's
    short text where it is unambiguous and adds digits only where two
    different numbers would print the same; the legend composer matches
    quantile lines by their numeric level; "median" means exactly 0.5.

Category M - "a shared Figure title must keep the whole axis title"
    (review finding P1-108-02).  Mechanism: the 2-D facet route put a
    'row=value' line in front of the left-column Y labels and the shared
    title then kept only the last line, so 'Residual\\n(units)' became
    '(units)'.  Fix: the cell remembers its original label and the shared
    title uses it; no line is ever dropped by guessing.  Members: scalar
    and vector 2-D facets, explicit and AliasDataFrame multi-line titles,
    X and Y, explicit shared-title precedence, outer headings.

Category B - "the shared Figure legend keeps every identity of the cell
    legends it replaces" (found while testing v0.2; same owner as P1-108-01).
    Mechanism: the composer built the Figure legend from expressions and
    quantile levels only.  Members: grouped profiles with discrete quantiles
    (v0.1 default and explicit legend='shared' dropped the group names, 1-D
    and 2-D facets), and the PHASE_13_95 1-D global group legend, which never
    named the quantile levels its cells draw.  Fix: the composer also keeps
    the cell-legend names; an existing global group legend is completed with
    the drawn quantile levels exactly as an explicit legend='shared' does.

``test_RED_*`` fail on the committed v0.1 code (dcdded01); ``test_CTL_*``
pass before and after (today's short labels and names are unchanged);
``test_UNIT_*`` test the new formatter module directly.

Seat: Fable5_1:draw-coder, 2026-11-10.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

Q_CLOSE = [0.100000001, 0.100000002, 0.9]
Q_NEAR_MEDIAN = [0.2, 0.5000000001, 0.8]


def _frame(n=8000, seed=10831, tiny=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.uniform(0.0, 1.0, n),
        'f': rng.integers(0, 3, n),
        'g': rng.integers(0, 2, n),
        'h': rng.integers(0, 2, n),
        'm': rng.uniform(0.0, 0.0004 if tiny else 4.0, n),
    })
    df['y0'] = rng.normal(0.0, 1.0, n) + 0.2 * df.f
    df['y1'] = 1.0 + rng.normal(0.0, 1.0, n)
    df['yi'] = rng.integers(0, 3, n).astype(float)   # tied quantile curves
    return df


def _close(fig):
    plt.close(fig)


def _legend_texts(fig, axes):
    texts = []
    for legend in fig.legends:
        texts.extend(t.get_text() for t in legend.get_texts())
    for ax in np.asarray(axes).ravel():
        if ax.get_legend() is not None:
            texts.extend(t.get_text() for t in ax.get_legend().get_texts())
    return texts


def _quantile_texts(texts):
    return [t for t in texts if t.startswith('q=') or t == 'median']


# --------------------------------------------------------------------------
# Category Q: quantile levels
# --------------------------------------------------------------------------

ROUTES = {
    'single': dict(expr='y0:x'),
    'facet-1d': dict(expr='y0:x', facet_by='f'),
    'facet-1d-vector': dict(expr='[y0,y1]:x', facet_by='f'),
    'facet-2d': dict(expr='y0:x', facet_by=['g', 'h']),
    'facet-1d-grouped-global': dict(expr='y0:x', facet_by='f', group_by='g',
                                    group_by_scope='global'),
    'overlay-facet-1d': dict(expr='y0:x', facet_by='f', type='hist2d+profile'),
}


def _draw(route, quantiles, df=None, **extra):
    kw = dict(ROUTES[route])
    expr = kw.pop('expr')
    kw.setdefault('type', 'profile')
    kw.update(extra)
    return DFDraw(_frame() if df is None else df).draw(
        expr, quantiles=quantiles, **kw)


@pytest.mark.parametrize('route', sorted(ROUTES))
def test_RED_Q01_close_levels_keep_separate_legend_entries(route):
    fig, axes, _ = _draw(route, Q_CLOSE)
    try:
        texts = _legend_texts(fig, axes)
        if route == 'facet-1d-grouped-global':
            # group colour legend + factored quantile entries
            q_texts = [t for t in fig.legends[0].get_texts()
                       if t.get_text().startswith('q=')]
            names = [t.get_text() for t in q_texts]
        else:
            names = list(dict.fromkeys(_quantile_texts(texts)))
        assert len(names) == 3, names
        assert len(set(names)) == 3, names
    finally:
        _close(fig)


@pytest.mark.parametrize('route', ['single', 'facet-1d', 'facet-2d',
                                   'overlay-facet-1d'])
def test_RED_Q02_near_median_level_is_not_called_median(route):
    fig, axes, _ = _draw(route, Q_NEAR_MEDIAN, quantile_mode='discrete')
    try:
        names = list(dict.fromkeys(_quantile_texts(_legend_texts(fig, axes))))
        assert names == ['q=20%', 'q=50.00000001%', 'q=80%'], names
    finally:
        _close(fig)


def test_RED_Q03_near_median_triple_is_not_a_band():
    """[0.2, 0.5000000001, 0.8] is three different levels; the automatic
    mode must not draw it as a symmetric band around 'the median'."""
    from dfextensions.dfdraw.plots.profile import _detect_quantile_mode
    assert _detect_quantile_mode(Q_NEAR_MEDIAN) == 'discrete'
    assert _detect_quantile_mode([0.2, 0.5, 0.8]) == 'band'


def test_RED_Q04_auto_title_keeps_close_levels_apart():
    from dfextensions.dfdraw.plots._auto_title import (
        format_profile_quantile_semantics)
    text = format_profile_quantile_semantics(Q_CLOSE)
    assert text == 'q [0.100000001, 0.100000002, 0.9]'


def test_CTL_Q05_tied_curves_still_two_entries():
    """Integer y: q=0.2 and q=0.3 are often the same curve; both levels
    were requested and both are named."""
    fig, axes, _ = _draw('facet-1d', [0.2, 0.3, 0.9],
                         df=_frame().assign(y0=_frame().yi))
    try:
        names = list(dict.fromkeys(_quantile_texts(_legend_texts(fig, axes))))
        assert names == ['q=20%', 'q=30%', 'q=90%'], names
    finally:
        _close(fig)


def test_RED_Q06_on_line_tags_match_legend_names():
    fig, axes, _ = _draw('single', Q_CLOSE)
    try:
        tags = sorted(t.get_text() for t in np.asarray(axes).ravel()[0].texts
                      if t.get_text().endswith('%'))
        assert len(set(tags)) == len(tags) == 3, tags
    finally:
        _close(fig)


@pytest.mark.parametrize('levels,expected', [
    ([0.05, 0.101, 0.5, 0.9, 0.95],
     ['5%', '10.1%', '50%', '90%', '95%']),
    ([0.1, 0.100000001], ['10%', '10.0000001%']),
    ([0.123456789], ['12.3457%']),
    ([0.0001, 0.9999], ['0.01%', '99.99%']),
], ids=['round', 'close-pair', 'six-digits', 'extremes'])
def test_UNIT_Q07_formatter_short_where_unambiguous(levels, expected):
    from dfextensions.dfdraw.plots._numeric_labels import (
        quantile_percent_labels)
    labels = quantile_percent_labels(levels)
    assert [labels[float(q)] for q in levels] == expected


def test_CTL_Q08_round_levels_unchanged_and_exact_median():
    fig, axes, _ = _draw('facet-1d', [0.05, 0.101, 0.5, 0.9, 0.95])
    try:
        names = list(dict.fromkeys(_quantile_texts(_legend_texts(fig, axes))))
        assert names == ['q=5%', 'q=10.1%', 'median', 'q=90%', 'q=95%']
    finally:
        _close(fig)


def test_CTL_Q09_same_level_in_every_cell_is_one_entry():
    fig, axes, _ = _draw('facet-1d', [0.2, 0.7])
    try:
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert texts.count('q=20%') == 1 and texts.count('q=70%') == 1
    finally:
        _close(fig)


# --------------------------------------------------------------------------
# Category Q: numeric bin names (the name is the grouping key)
# --------------------------------------------------------------------------

BIN_ROUTES = {
    'group-bins': dict(expr='y0:x', group_by='m', group_by_bins=8),
    'group-quantiles': dict(expr='y0:x', group_by='m', group_by_quantiles=8),
    'group-bins-facet': dict(expr='y0:x', group_by='m', group_by_bins=8,
                             facet_by='f'),
    'group-quantiles-global': dict(expr='y0:x', group_by='m',
                                   group_by_quantiles=8, facet_by='f',
                                   group_by_scope='global'),
    'facet-bins': dict(expr='y0:x', facet_by='m', facet_by_bins=8),
    'facet-quantiles': dict(expr='y0:x', facet_by='m', facet_by_quantiles=8),
    'facet-group-bins': dict(expr='y0:x', facet_by='group_by', group_by='m',
                             group_by_bins=8),
}


def _interval_like(text):
    return any(c.isdigit() for c in text) and '-' in text[1:]


def _group_names(fig, axes, stats, route):
    """Names of the groups (legend entries) or facets (stats['groups'])
    that were actually drawn."""
    if route.startswith('facet-'):
        return [str(g) for g in stats['groups']]
    return sorted({t for t in _legend_texts(fig, axes) if _interval_like(t)})


@pytest.mark.parametrize('route', sorted(BIN_ROUTES))
def test_RED_I01_narrow_bins_are_never_merged(route):
    kw = dict(BIN_ROUTES[route])
    expr = kw.pop('expr')
    fig, axes, stats = DFDraw(_frame(tiny=True)).draw(expr, type='profile',
                                                       **kw)
    try:
        names = _group_names(fig, axes, stats, route)
        assert len(names) == 8, names
        assert len(set(names)) == 8, names
        from dfextensions.dfdraw.plots._numeric_labels import _edge_texts
        for name in names:
            left, right = _edge_texts(name)
            assert left != right, name
    finally:
        _close(fig)


def test_UNIT_I02_unambiguous_bin_names_unchanged():
    from dfextensions.dfdraw.plots._numeric_labels import interval_labels
    from dfextensions.dfdraw.plots.profile import _format_interval_label
    intervals = pd.IntervalIndex.from_breaks(
        [0.0, 0.40, 0.68, 0.91, 1.029, 1.093])
    labels = interval_labels(intervals)
    assert [labels[iv] for iv in intervals] == [
        _format_interval_label(iv) for iv in intervals]
    assert [labels[iv] for iv in intervals] == [
        '0.00-0.40', '0.40-0.68', '0.68-0.91', '0.91-1.03', '1.029-1.093']


def test_CTL_I03_group_names_unchanged_on_normal_data():
    fig, axes, stats = DFDraw(_frame()).draw(
        'y0:x', type='profile', group_by='m', group_by_bins=4)
    try:
        assert _group_names(fig, axes, stats, 'group-bins') == [
            '-0.0-1.0', '1.00-2.00', '2.0-3.0', '3.00-4.00']
    finally:
        _close(fig)


# --------------------------------------------------------------------------
# Category M: shared titles keep every line
# --------------------------------------------------------------------------

MULTI_Y = 'Residual\n(units)'
MULTI_X = 'Occupancy\n(norm. mean)'


class _TitleSource:
    def get_axis_title(self, name):
        return {'y0': r'$dE/dx$' + '\n' + '(MIP units)',
                'x': 'occupancy\n(norm)'}.get(name)


def _draw2d(expr='y0:x', titles=False, **kw):
    d = DFDraw(_frame())
    if titles:
        d._data_source = _TitleSource()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return d.draw(expr, type='profile', facet_by=['g', 'h'], **kw)


def _sup(fig, axis):
    artist = getattr(fig, '_supylabel' if axis == 'y' else '_supxlabel', None)
    return None if artist is None else artist.get_text()


def test_RED_M01_multiline_ylabel_kept_whole():
    fig, axes, _ = _draw2d(ylabel=MULTI_Y, xlabel=MULTI_X)
    try:
        assert _sup(fig, 'y') == MULTI_Y
        assert _sup(fig, 'x') == MULTI_X
    finally:
        _close(fig)


def test_RED_M02_multiline_alias_titles_kept_whole():
    fig, axes, _ = _draw2d(titles=True)
    try:
        assert _sup(fig, 'y') == r'$dE/dx$' + '\n' + '(MIP units)'
        assert _sup(fig, 'x') == 'occupancy\n(norm)'
    finally:
        _close(fig)


def test_RED_M03_shared_true_keeps_multiline():
    fig, axes, _ = _draw2d(ylabel=MULTI_Y, shared_ylabel=True)
    try:
        assert _sup(fig, 'y') == MULTI_Y
    finally:
        _close(fig)


def test_CTL_M04_single_line_and_row_identity():
    fig, axes, _ = _draw2d(ylabel='Residual')
    try:
        assert _sup(fig, 'y') == 'Residual'
        headings = [t.get_text() for t in fig.texts]
        assert any(t.startswith('g=') for t in headings), headings
    finally:
        _close(fig)


def test_CTL_M05_explicit_shared_title_wins():
    fig, axes, _ = _draw2d(ylabel=MULTI_Y, shared_ylabel='custom')
    try:
        assert _sup(fig, 'y') == 'custom'
    finally:
        _close(fig)


def test_CTL_M06_shared_false_keeps_cell_labels():
    fig, axes, _ = _draw2d(ylabel=MULTI_Y, shared_ylabel=False)
    try:
        assert _sup(fig, 'y') is None
        left = [ax.get_ylabel() for ax in np.asarray(axes)[:, 0]]
        assert all(label.endswith(MULTI_Y) for label in left), left
    finally:
        _close(fig)


def test_RED_M07_outer_headings_multiline():
    fig, axes, _ = _draw2d(ylabel=MULTI_Y, facet_labels='outer')
    try:
        assert _sup(fig, 'y') == MULTI_Y
    finally:
        _close(fig)


def test_CTL_M08_vector_shared_title_is_the_cell_label():
    """Vector 2-D facets: the cells render the expression list; the shared
    Y title is that whole label (nothing dropped)."""
    fig, axes, _ = _draw2d('[y0,y1]:x')
    try:
        assert _sup(fig, 'y') == '[y0, y1]'
    finally:
        _close(fig)


# --------------------------------------------------------------------------
# Category B (found while testing v0.2): the Figure legend keeps every
# identity of the cell legends it replaces
# --------------------------------------------------------------------------

@pytest.mark.parametrize('facet_by', ['f', ['f', 'h']], ids=['1d', '2d'])
@pytest.mark.parametrize('legend', [None, 'shared'], ids=['default', 'shared'])
def test_RED_B01_grouped_discrete_quantiles_keep_group_names(facet_by, legend):
    """Grouped profile with discrete quantiles: the cells name the groups
    ('0', '1'); the shared Figure legend must name them too, next to the
    quantile levels.  v0.1 kept only the quantile entries."""
    df = _frame()
    df['y0'] = df.y0 + df.g
    kw = {} if legend is None else {'legend': legend}
    fig, axes, _ = DFDraw(df).draw('y0:x', type='profile', facet_by=facet_by,
                                   group_by='g', quantiles=[0.1, 0.101, 0.9],
                                   **kw)
    try:
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert '0' in texts and '1' in texts, texts
        assert {'q=10%', 'q=10.1%', 'q=90%'} <= set(texts), texts
    finally:
        _close(fig)


@pytest.mark.parametrize('kw', [
    dict(group_by='g'),
    dict(group_by='m', group_by_quantiles=3),
], ids=['discrete-groups', 'global-bins'])
def test_RED_B02_global_group_legend_names_quantile_levels(kw):
    """1-D facets with group_by_scope='global': the Figure group legend
    also names the quantile levels the cells draw (as legend='shared' and
    the 2-D global route already do)."""
    fig, axes, _ = DFDraw(_frame()).draw(
        'y0:x', type='profile', facet_by='f', group_by_scope='global',
        quantiles=[0.1, 0.101, 0.9], **kw)
    try:
        assert len(fig.legends) == 1
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert {'q=10%', 'q=10.1%', 'q=90%'} <= set(texts), texts
        assert len([t for t in texts if not t.startswith('q=')]) in (2, 3)
    finally:
        _close(fig)


def test_CTL_B03_global_group_legend_without_quantiles_unchanged():
    fig, axes, _ = DFDraw(_frame()).draw(
        'y0:x', type='profile', facet_by='f', group_by='g',
        group_by_scope='global')
    try:
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert texts == ['0', '1']
    finally:
        _close(fig)
