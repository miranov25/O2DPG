"""PHASE_13_113_DF - legend truth follow-up of PHASE_13_112: tests.

Plain-language background
-------------------------
The POSTCOMMIT review of PHASE_13_112 (consolidated summary GPG36,
2026-11-21) found two legends that tell the reader something false.  This
module pins both on the PHASE_13_112 end state (tag PHASE_13_112_DF_END =
7ce5469a) and checks the fix.  Layout findings of the same review are left
to a later phase on layout and styling.

* F112-01  A faceted scatter plot with a categorical ``group_by``: each cell
           coloured its groups in the order they first appear in that cell,
           so one group had different colours in different cells, and the
           shared legend showed only the first cell's colours.
           Fix: the cells get the ordered group list of the whole call (as
           profiles and histograms already do, PHASE_13_111 CR-1); a group's
           colour and marker follow its rank in that list.  The shared
           legend takes its entries from every cell, not only the first, in
           that order.
* F112-02  The group legend title named the number of bins asked for
           (``group_by_quantiles=3`` -> "3 bins") even when duplicate quantile
           edges collapse them to fewer.  Fix: the title names the number of
           bins the binning produced; "1 bin" in the singular.
* S1b      Review of S1 (GPG36 consolidated, 2026-11-21, REVISE, P1): when
           the facet cells produce different numbers of bins (one cell's
           quantile edges collapse, the other's do not), every cell legend
           still named the requested count.  Fix: each axes legend names the
           count of its own axes; a Figure legend names the common count, or
           the range ('1-3 bins') when the cells differ.
* F112-04  Review question: can an explicit ``group_by_scope='global'``
           skip the refusal before drawing?  Not a defect: every unsupported
           variant is refused before any Figure is created.  These tests lock
           that behaviour.

``test_RED_*`` fail on the PHASE_13_112 end state and pass with the fix.
``test_CTL_*`` and ``test_LOCK_*`` pass on both.

Seat: Fable5.1_5:draw-coder, 2026-11-21.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from dfextensions.dfdraw.drawer import DFDraw

_TAB10 = [to_hex(plt.colormaps.get_cmap('tab10')(i)) for i in range(10)]


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close('all')
    yield
    plt.close('all')


# ----------------------------------------------------------------- helpers

def _scatter_frame(n=3000, seed=21, missing_in_first_cell=True):
    """Three categorical groups 0, 1, 2; facet columns f and h.  Group 1 is
    removed below f = 0.4, so the first of three f cells (f < 1/3) has no
    group 1 and draws group 2 second."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'x': rng.uniform(0, 1, n), 'y': rng.normal(0, 1, n),
                       'f': rng.uniform(0, 1, n), 'h': rng.uniform(0, 1, n),
                       'gi': rng.integers(0, 3, n)})
    if missing_in_first_cell:
        df.loc[(df.f < 0.4) & (df.gi == 1), 'gi'] = 2
    return df


def _first_cell_labels(fig):
    return sorted(c.get_label() for c in fig.axes[0].collections
                  if c.get_label() and not c.get_label().startswith('_'))


def _group_artists(fig):
    """{group name: [(colour, marker path vertices), ...]} over all cells."""
    out = {}
    for ax in fig.axes:
        for coll in ax.collections:
            label = coll.get_label()
            if not label or label.startswith('_'):
                continue
            path = coll.get_paths()[0].vertices.round(6).tobytes()
            out.setdefault(label, []).append((to_hex(coll.get_facecolor()[0]), path))
    return out


def _legend_entries(legend):
    entries = []
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        colour = (to_hex(handle.get_facecolor()[0])
                  if hasattr(handle, 'get_facecolor') else None)
        entries.append((text.get_text(), colour))
    return entries


def _titles(fig):
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                   if ax.get_legend() is not None]
    return sorted({(lg.get_title().get_text(), len(lg.get_texts()))
                   for lg in legends})


# ---------------------------------------------------------------- F112-01

_SCATTER_CASES = {
    '1d-shared': dict(facet_by='f', facet_by_bins=3, legend='shared'),
    '2d-shared': dict(facet_by=['f', 'h'], facet_by_bins=[3, 2], legend='shared'),
    '1d-cell-legends': dict(facet_by='f', facet_by_bins=3, legend='all'),
    '2d-cell-legends': dict(facet_by=['f', 'h'], facet_by_bins=[3, 2], legend='all'),
}


@pytest.mark.parametrize('case', list(_SCATTER_CASES))
def test_RED_F112_01_scatter_group_has_one_colour_and_marker_in_every_cell(case):
    df = _scatter_frame()
    DFDraw(df).draw('y:x', type='scatter', group_by='gi', **_SCATTER_CASES[case])
    groups = _group_artists(plt.gcf())
    assert sorted(groups) == ['0', '1', '2']
    for rank, name in enumerate(['0', '1', '2']):
        colours = {c for c, _ in groups[name]}
        markers = {m for _, m in groups[name]}
        assert colours == {_TAB10[rank]}, (name, colours)
        assert len(markers) == 1, (name, len(markers))


@pytest.mark.parametrize('case', ['1d-shared', '2d-shared'])
def test_RED_F112_01_shared_legend_is_true_in_every_cell(case):
    df = _scatter_frame()
    DFDraw(df).draw('y:x', type='scatter', group_by='gi', **_SCATTER_CASES[case])
    fig = plt.gcf()
    assert _first_cell_labels(fig) == ['0', '2']     # the test's premise
    assert len(fig.legends) == 1
    entries = _legend_entries(fig.legends[0])
    assert entries == [('0', _TAB10[0]), ('1', _TAB10[1]), ('2', _TAB10[2])]
    groups = _group_artists(fig)
    for name, colour in entries:
        assert {c for c, _ in groups[name]} == {colour}


def test_RED_F112_01_reviewers_call():
    """The exact call of the review: random categorical groups, three cells."""
    rng = np.random.default_rng(5)
    n = 900
    df = pd.DataFrame({'x': rng.uniform(0, 1, n), 'y': rng.normal(0, 1, n),
                       'f': rng.uniform(0, 1, n), 'gi': rng.integers(0, 3, n)})
    DFDraw(df).draw('y:x', type='scatter', group_by='gi',
                    facet_by='f', facet_by_bins=3, legend='shared')
    groups = _group_artists(plt.gcf())
    for rank, name in enumerate(['0', '1', '2']):
        assert {c for c, _ in groups[name]} == {_TAB10[rank]}


def test_CTL_F112_01_unfaceted_scatter_keeps_first_appearance_order():
    """A scatter plot with one axes is unchanged: the first group drawn takes
    the first colour (no order list is given outside the facet route)."""
    df = pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       'gi': [2, 2, 0, 0, 1, 1]})
    DFDraw(df).draw('y:x', type='scatter', group_by='gi')
    groups = _group_artists(plt.gcf())
    assert {c for c, _ in groups['2']} == {_TAB10[0]}
    assert {c for c, _ in groups['0']} == {_TAB10[1]}
    assert {c for c, _ in groups['1']} == {_TAB10[2]}


def test_CTL_F112_01_complete_cells_keep_their_colours():
    """Cells that hold every group in sorted order look as before."""
    df = _scatter_frame(missing_in_first_cell=False)
    df = df.sort_values('gi', kind='stable').reset_index(drop=True)
    DFDraw(df).draw('y:x', type='scatter', group_by='gi',
                    facet_by='f', facet_by_bins=3, legend='shared')
    groups = _group_artists(plt.gcf())
    for rank, name in enumerate(['0', '1', '2']):
        assert {c for c, _ in groups[name]} == {_TAB10[rank]}


# ---------------------------------------------------------------- F112-02

def _binary_frame(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'x': rng.uniform(0, 1, n), 'y': rng.normal(0, 1, n),
                         'a': rng.uniform(0, 1, n), 'b': rng.uniform(0, 1, n),
                         'p': np.r_[np.zeros(n // 2), np.ones(n - n // 2)],
                         'q': rng.uniform(0, 1, n)})


_TITLE_CASES = {
    # name: (expression, keywords, expected [(title, number of entries)])
    'global-1d-collapsed': ('y:x', dict(type='profile', bins=7, group_by='p',
                            group_by_quantiles=3, facet_by='a', facet_by_bins=2,
                            legend='shared'),
                            [('Colors: p (1 global bin)', 1)]),
    'global-2d-collapsed': ('y:x', dict(type='profile', bins=7, group_by='p',
                            group_by_quantiles=3, facet_by=['a', 'b'],
                            facet_by_bins=[2, 2], legend='shared'),
                            [('Colors: p (1 global bin)', 1)]),
    'unfaceted-profile-collapsed': ('y:x', dict(type='profile', bins=7,
                                    group_by='p', group_by_quantiles=3),
                                    [('Colors: p (1 bin)', 1)]),
    'per-cell-collapsed': ('y:x', dict(type='profile', bins=7, group_by='p',
                           group_by_quantiles=3, facet_by='a', facet_by_bins=2,
                           legend='all'),
                           [('Colors: p (1 bin)', 1)]),
    'unfaceted-hist-collapsed': ('y', dict(type='hist', bins=7, group_by='p',
                                 group_by_quantiles=3),
                                 [('Colors: p (1 bin)', 1)]),
}


@pytest.mark.parametrize('case', list(_TITLE_CASES))
def test_RED_F112_02_title_names_the_bins_that_exist(case):
    expr, kw, expected = _TITLE_CASES[case]
    DFDraw(_binary_frame()).draw(expr, **kw)
    assert _titles(plt.gcf()) == expected


_TITLE_CONTROLS = {
    'global-1d-normal': ('y:x', dict(type='profile', bins=7, group_by='q',
                         group_by_quantiles=3, facet_by='a', facet_by_bins=2,
                         legend='shared'),
                         [('Colors: q (3 global bins)', 3)]),
    'unfaceted-bins': ('y:x', dict(type='profile', bins=7, group_by='q',
                       group_by_bins=3),
                       [('Colors: q (3 bins)', 3)]),
    'explicit-edges-global': ('y:x', dict(type='profile', bins=7, group_by='q',
                              group_by_bins=[0, 0.5, 1], facet_by='a',
                              facet_by_bins=2, legend='shared'),
                              [('Colors: q (2 bins)', 2)]),
}


@pytest.mark.parametrize('case', list(_TITLE_CONTROLS))
def test_CTL_F112_02_title_unchanged_when_no_bin_collapses(case):
    expr, kw, expected = _TITLE_CONTROLS[case]
    DFDraw(_binary_frame()).draw(expr, **kw)
    assert _titles(plt.gcf()) == expected


# ---------------------------------------------------------------- F112-04

_GLOBAL_REFUSALS = {
    'hist-bins': ('y', dict(type='hist', bins=8, group_by_bins=3,
                            facet_by='a', facet_by_bins=2)),
    'hist-quantiles': ('y', dict(type='hist', bins=8, group_by_quantiles=3,
                                 facet_by='a', facet_by_bins=2)),
    'hist2d-bins': ('y:x', dict(type='hist2d', bins=8, group_by_bins=3,
                                facet_by='a', facet_by_bins=2)),
    'scatter-bins': ('y:x', dict(type='scatter', group_by_bins=3,
                                 facet_by='a', facet_by_bins=2)),
    'profile-3-levels': ('y:x', dict(type='profile', bins=5, group_by_bins=3,
                                     facet_by=['a', 'b', 'c'],
                                     facet_by_bins=[2, 2, 2])),
    'profile-top-k': ('y:x', dict(type='profile', bins=5, group_by_bins=3,
                                  facet_by='a', facet_by_bins=2, top_k=2)),
}


@pytest.mark.parametrize('case', list(_GLOBAL_REFUSALS))
def test_LOCK_F112_04_explicit_global_refused_before_any_figure(case):
    rng = np.random.default_rng(3)
    n = 2000
    df = pd.DataFrame({k: rng.uniform(0, 1, n) for k in 'xyabcp'})
    expr, kw = _GLOBAL_REFUSALS[case]
    created = []
    original = mfig.Figure.__init__

    def counting(self, *a, **k):
        created.append(1)
        original(self, *a, **k)
    mfig.Figure.__init__ = counting
    try:
        with pytest.raises((NotImplementedError, ValueError)):
            DFDraw(df).draw(expr, group_by='p', group_by_scope='global',
                            legend='shared', **kw)
    finally:
        mfig.Figure.__init__ = original
    assert not created, f'{len(created)} Figure(s) created before the refusal'
    assert plt.get_fignums() == []


# ---------------------------------------------------------------- S1b

def _two_cell_frame(n=60):
    """The reviewers' case: cell f=0 has a two-valued p (quantile edges
    collapse to one interval), cell f=1 a continuous p (three intervals)."""
    return pd.DataFrame({
        'x': np.r_[np.linspace(.1, .9, n), np.linspace(.1, .9, n)],
        'y': np.r_[np.linspace(0, 1, n), np.linspace(0, 1, n)],
        'f': np.r_[np.zeros(n), np.ones(n)],
        'p': np.r_[np.repeat([0., 1.], n // 2), np.linspace(0, 1, n)],
    })


def _axes_titles(fig):
    return [(ax.get_legend().get_title().get_text(), len(ax.get_legend().get_texts()))
            for ax in fig.axes if ax.get_legend() is not None]


_S1B_CASES = {
    'profile-all': ('y:x', 'profile', 'all',
                    [('Colors: p (1 bin)', 1), ('Colors: p (3 bins)', 3)]),
    'profile-first': ('y:x', 'profile', 'first', [('Colors: p (1 bin)', 1)]),
    'hist-all': ('y', 'hist', 'all',
                 [('Colors: p (1 bin)', 1), ('Colors: p (3 bins)', 3)]),
    'hist-first': ('y', 'hist', 'first', [('Colors: p (1 bin)', 1)]),
}


@pytest.mark.parametrize('case', list(_S1B_CASES))
def test_RED_S1b_each_cell_legend_names_its_own_bin_count(case):
    expr, kind, mode, expected = _S1B_CASES[case]
    DFDraw(_two_cell_frame()).draw(expr, type=kind, bins=7, group_by='p',
                                   group_by_quantiles=3, facet_by='f',
                                   facet_by_bins=2, legend=mode)
    assert _axes_titles(plt.gcf()) == expected


@pytest.mark.parametrize('kind', ['profile', 'hist'])
def test_CTL_S1b_fixed_bins_name_the_defined_count_in_every_cell(kind):
    """group_by_bins=3 defines three intervals in every cell, also where one
    of them is empty (two entries shown): the title stays '3 bins'."""
    expr = 'y:x' if kind == 'profile' else 'y'
    DFDraw(_two_cell_frame()).draw(expr, type=kind, bins=7, group_by='p',
                                   group_by_bins=3, facet_by='f',
                                   facet_by_bins=2, legend='all')
    assert _axes_titles(plt.gcf()) == [('Colors: p (3 bins)', 2),
                                       ('Colors: p (3 bins)', 3)]


def test_S1b_range_title_text():
    from dfextensions.dfdraw.drawer import _group_legend_title
    parts = dict(name='p', binned=True, scope_global=False)
    assert _group_legend_title(n_defined=(1, 3), **parts) == 'Colors: p (1\u20133 bins)'
    assert _group_legend_title(n_defined=(2, 2), **parts) == 'Colors: p (2 bins)'
    assert _group_legend_title(n_defined=(1, 1), **parts) == 'Colors: p (1 bin)'


def test_S1b_figure_legend_over_cells_with_different_counts_names_the_range():
    """A Figure legend titled with the requested count before layout gets the
    range when its cells produced different numbers of bins."""
    from dfextensions.dfdraw.drawer import _title_group_legends
    fig, (a1, a2) = plt.subplots(1, 2)
    a1._dfdraw_group_bin_edges = [np.array([0., 1.])]
    a2._dfdraw_group_bin_edges = [np.array([0., .3, .6, 1.])]
    a2.plot([0, 1], [0, 1], label='(0.0, 0.3]')
    fig.legend(*a2.get_legend_handles_labels(), title='Colors: p (3 bins)')
    _title_group_legends(fig, 'Colors: p (3 bins)', binned=True, group_names=None,
                         figure_title='Colors: p (1\u20133 bins)',
                         replace={'Colors: p (3 bins)'})
    assert fig.legends[0].get_title().get_text() == 'Colors: p (1\u20133 bins)'


def test_CTL_S1b_shared_legend_on_caller_axes_lists_every_label_once():
    """Contract of legend='shared' (documented in _apply_legend_mode): every
    label of every axes, once, in the order first seen."""
    from dfextensions.dfdraw.plots._legend import (_apply_legend_mode,
                                                   _normalize_legend_spec)
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.plot([0, 1], [0, 1], label='a')
    a1.plot([0, 1], [1, 0], label='b')
    a1.legend()
    a2.plot([0, 1], [0, 1], label='b')
    a2.plot([0, 1], [1, 0], label='c')
    a2.legend()
    _apply_legend_mode(fig, _normalize_legend_spec('shared'))
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ['a', 'b', 'c']
    assert all(ax.get_legend() is None for ax in fig.axes)
