"""PHASE_13_112_DF S0 - legend, label and placement defects: failing (RED) tests and controls.

Plain-language background
-------------------------
PHASE_13_112 (proposal v0.2, approved 2026-11-19, with the S0 decision addendum)
fixes presentation defects found after PHASE_13_111.  None changes a computed
number.  This module pins each defect on the start code (tag
PHASE_13_112_DF_BEGIN = fe34f080) before any product code is changed.

* ``test_RED_*``: fail on the start code.  Each is marked
  ``xfail(strict=True)`` with the stage that will fix it, so the canonical run
  stays at 0 failed, and a RED that passes early is reported as a failure.
  The fixing stage removes its marker.  S1a (2026-11-19) removed the markers
  of F-1, F-6, F-8, F-4, O-5 and F-2 (17 tests); S1b (2026-11-20) fixed F-7
  for group legends; the facet-title test became a control (option b).
  S2 (2026-11-20) removed the markers of F-5, P-5, P-6 and F-3 (6 tests)
  and added the S2 controls and mutants at the end of this module.  S3
  (2026-11-20) removed the last two markers (D-5, D-6) and added the S3
  tests after the S2 ones.  No expected-failure marker is left.
* ``test_CTL_*``: neighbouring behaviour that must not change; pass now.

Items (proposal v0.2 section 2; addendum section 1):

* F-1  ``same=True`` chains through a new drawer per call (as AliasDataFrame.draw
       does): every curve after the first gets the second colour, and the first
       name is lost.                                           -> S1
* F-6  a group legend does not name the grouping variable.      -> S1
* F-8  (new in S0) a user legend title in ``legend={'mode': 'all', 'title': ...}``
       is ignored.                                             -> S1
* F-4  group legend order follows the first facet cell.         -> S1
* O-5  ``ylabel=`` is ignored for a vector profile (13.108 O-5). -> S1
* F-2  the Y title shows "(+-1 more)" / "(+0 more)".            -> S1
* F-7  group legend interval names do not show the bin closure, "(a, b]".
       -> S1b (Architect option b, 2026-11-20: legends only; facet titles
       keep the PHASE_13_97 default)
* F-5  ``legend="shared"`` with intervals computed per cell: a merged or a
       bottom legend instead of one top legend or a refusal.   -> S2
* P-5  (13.107 item 5) faceted scatter shared legend at the bottom. -> S2
* P-6  (13.107 item 6) empty band above the 1-D shared legend.  -> S2
* F-3  the legend covers the fit-result box.                    -> S2
* D-5  a fit line over filled histogram bars has the bar colour. -> S3
* D-6  in-panel quantile level labels repeated for every curve although the
       legend shows the levels.                                -> S3

Seat: Fable5.1_5:draw-coder (PHASE_13_112 coder), 2026-11-19.
"""
import colorsys
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgb

from dfextensions.dfdraw.drawer import DFDraw

from tests.test_phase_13_111_s0_identity_colour_red import _palette   # noqa: E402

PCT = re.compile(r'^\d+(\.\d+)?%$')
CLOSED = re.compile(r'^[\(\[]-?\d+(\.\d+)?, -?\d+(\.\d+)?[\)\]]$')


def RED(item, stage):
    return pytest.mark.xfail(strict=True, reason=f'PHASE_13_112 {item}: RED until {stage}')


@pytest.fixture(autouse=True)
def _close_all():
    plt.close('all')
    yield
    plt.close('all')


# --------------------------------------------------------------------------
# data and helpers
# --------------------------------------------------------------------------
def _frame(n=30000, seed=11201):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'x': rng.uniform(0.0, 4.0, n), 'a': rng.uniform(0.0, 1.0, n),
                       'b': rng.uniform(0.0, 1.0, n), 'p': rng.uniform(-2.0, 2.0, n),
                       'g': rng.integers(0, 4, n)})
    for k in (1, 2, 3, 4):
        df[f'y{k}'] = 10.0 * k + rng.normal(0.0, 0.3, n)
    df['r'] = rng.normal(0.0, 0.1 + 0.03 * df['x'])
    return df


def _cells_differ(df):
    """The grouping variable spans half the range in the cells a < 0.5."""
    d = df.copy()
    d.loc[d['a'] < 0.5, 'p'] *= 0.5
    return d


def _cells_identical(df):
    """Every facet cell (a, b at 0.5) holds p = -2 and p = +2 exactly, so the
    intervals computed per cell are identical, edges included."""
    rows = []
    for a in (0.25, 0.75):
        for b in (0.25, 0.75):
            for p in (-2.0, 2.0):
                rows.append({'x': 1.0, 'a': a, 'b': b, 'p': p, 'g': 0,
                             'y1': 10.0, 'y2': 20.0, 'y3': 30.0, 'y4': 40.0, 'r': 0.0})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def _legends(fig):
    return list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]


def _texts(legend):
    return [t.get_text() for t in legend.get_texts()]


def _curve_colours(ax):
    """Colour of every central line with data, in drawing order."""
    return [to_hex(ln.get_color()) for ln in ax.lines
            if len(ln.get_xdata()) > 1 and not getattr(ln, '_dfdraw_quantile', None)
            and ln.get_linestyle() in ('-', 'solid')]


def _fig_box(fig, artist):
    fig.canvas.draw()
    return artist.get_window_extent().transformed(fig.transFigure.inverted())


def _grid_top(fig):
    return max(ax.get_position().y1 for ax in fig.axes if ax.get_visible())


def _refused_and_closed(call, match):
    """The call is refused with ValueError and leaves no Figure of its own
    open; the caller's current Figure stays current.  (S2b, review S2-C02:
    renamed from _refused_before_figure.  The refusal comes after the cells
    were drawn, because the real bin edges are known only then; the Figure
    is closed before the error is raised.  Architect waiver, option a.)"""
    before = set(plt.get_fignums())
    current = plt.gcf() if before else None
    with pytest.raises(ValueError, match=match):
        call()
    assert set(plt.get_fignums()) == before, 'a Figure was left open'
    if current is not None:
        assert plt.gcf() is current, 'the current Figure changed'


# --------------------------------------------------------------------------
# F-1: same=True chains through a new drawer per call
# --------------------------------------------------------------------------
def _fresh_chain(df, n, **kw):
    for k in range(1, n + 1):
        DFDraw(df).draw(f'y{k}:x', type='profile', bins=8, same=(k > 1), **kw)
    return plt.gcf().axes[0]


@pytest.mark.parametrize('n', [2, 3, 4])
def test_RED_F1_fresh_drawer_chain_colours_and_names(n):
    ax = _fresh_chain(_frame(), n)
    assert _curve_colours(ax) == [_palette(i) for i in range(n)]
    assert _texts(ax.get_legend())[:n] == [f'y{k} vs x' for k in range(1, n + 1)]


def test_RED_F1_adf_draw_chain():
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    adf = adf_module.AliasDataFrame(_frame())
    for k in (1, 2, 3):
        adf.draw(f'y{k}:x', type='profile', bins=8, same=(k > 1))
    ax = plt.gcf().axes[0]
    assert _curve_colours(ax) == [_palette(i) for i in range(3)]
    assert _texts(ax.get_legend())[:3] == ['y1 vs x', 'y2 vs x', 'y3 vs x']


def test_RED_F1_user_line_then_fresh_chain():
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [0.0, 0.0], label='user')
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, ax=ax)
    DFDraw(_frame()).draw('y2:x', type='profile', bins=8, ax=ax, same=True)
    assert _curve_colours(ax) == [_palette(0), _palette(1), _palette(2)]
    names = _texts(ax.get_legend())
    assert names[:3] == ['user', 'y1 vs x', 'y2 vs x'], names


def test_CTL_F1_one_drawer_chain():
    d = DFDraw(_frame())
    for k in (1, 2, 3, 4):
        d.draw(f'y{k}:x', type='profile', bins=8, same=(k > 1))
    ax = plt.gcf().axes[0]
    assert _curve_colours(ax) == [_palette(i) for i in range(4)]
    assert _texts(ax.get_legend())[:4] == [f'y{k} vs x' for k in (1, 2, 3, 4)]


def test_CTL_F1_cleared_axes_restart_at_first_colour():
    fig, ax = plt.subplots()
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, ax=ax)
    ax.cla()
    DFDraw(_frame()).draw('y2:x', type='profile', bins=8, ax=ax)
    assert _curve_colours(ax) == [_palette(0)]


def test_CTL_F1_explicit_colour_wins_in_chain():
    """Through the profile() method.  draw(type='profile', color=...) ignores
    color= on the start code (S0 inventory finding F-9, not in this test)."""
    d = DFDraw(_frame())
    d.profile('y1:x', bins=8)
    d.profile('y2:x', bins=8, same=True, color='k')
    assert _curve_colours(plt.gcf().axes[0]) == [_palette(0), '#000000']


# --------------------------------------------------------------------------
# F-6 / F-8: the group legend names the grouping variable; a user title wins
# --------------------------------------------------------------------------
ROUTES_WITHOUT_TITLE = {
    'reported-call': ('y1:x', dict(type='profile', bins=8, group_by='p', group_by_bins=7,
                                   group_by_scope='global', facet_by='a', facet_by_quantiles=3,
                                   legend='shared', ncols=3), 'p'),
    '1d-global-default': ('y1:x', dict(type='profile', bins=8, group_by='p', group_by_bins=5,
                                       group_by_scope='global', facet_by='a', facet_by_bins=3), 'p'),
    'unfaceted-binned': ('y1:x', dict(type='profile', bins=8, group_by='p', group_by_bins=5), 'p'),
    'unfaceted-categorical': ('y1:x', dict(type='profile', bins=8, group_by='g'), 'g'),
    '1d-categorical': ('y1:x', dict(type='profile', bins=8, group_by='g', facet_by='a',
                                    facet_by_bins=3), 'g'),
    'hist-binned': ('y1', dict(type='hist', bins=40, group_by='p', group_by_bins=3), 'p'),
    'hist-1d-categorical': ('y1', dict(type='hist', bins=40, group_by='g', facet_by='a',
                                       facet_by_bins=2), 'g'),
}


@pytest.mark.parametrize('route', sorted(ROUTES_WITHOUT_TITLE))
def test_RED_F6_group_legend_title_names_the_variable(route):
    expr, kw, var = ROUTES_WITHOUT_TITLE[route]
    DFDraw(_frame()).draw(expr, **kw)
    titles = [lg.get_title().get_text() for lg in _legends(plt.gcf())]
    assert titles and all(t.startswith('Colors: ') and var in t for t in titles), titles


def test_CTL_F6_2d_global_route_already_has_the_title():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='p', group_by_bins=5,
                          group_by_scope='global', facet_by=['a', 'b'], facet_by_bins=[2, 2],
                          legend='shared')
    titles = [lg.get_title().get_text() for lg in _legends(plt.gcf())]
    assert titles == ['Colors: p (5 global bins)'], titles


def test_CTL_F6_no_group_no_colour_title():
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8)
    assert not any(lg.get_title().get_text().startswith('Colors:') for lg in _legends(plt.gcf()))


def test_RED_F8_user_legend_title_kept():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g',
                          legend={'mode': 'all', 'title': 'Custom'})
    assert [lg.get_title().get_text() for lg in _legends(plt.gcf())] == ['Custom']


# --------------------------------------------------------------------------
# F-4: Figure legend in the global group order when the first cell lacks a group
# --------------------------------------------------------------------------
def test_RED_F4_global_group_order_when_first_cell_lacks_a_group():
    df = _frame()
    df = df[~((df['g'] == 0) & (df['a'] < 0.5))]
    DFDraw(df).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a', facet_by_bins=2)
    names = [t for lg in _legends(plt.gcf()) for t in _texts(lg)]
    groups = [t for t in names if t in {'0', '1', '2', '3'}]
    assert groups == ['0', '1', '2', '3'], names


# --------------------------------------------------------------------------
# O-5 / F-2: vector Y titles
# --------------------------------------------------------------------------
def test_RED_O5_user_ylabel_wins_for_vector_profile():
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, ylabel='MY TITLE')
    assert plt.gcf().axes[0].get_ylabel() == 'MY TITLE'


def test_CTL_O5_user_ylabel_scalar_profile():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, ylabel='MY TITLE')
    assert plt.gcf().axes[0].get_ylabel() == 'MY TITLE'


LONG = {'y1': 'a' * 32, 'y2': 'b' * 28, 'y3': 'c' * 20, 'y4': 'd' * 20}


def _long_ylabel(k):
    df = _frame().rename(columns=LONG)
    names = [LONG[f'y{i}'] for i in range(1, k + 1)]
    DFDraw(df).draw('[' + ','.join(names) + ']:x', type='profile', bins=8)
    return plt.gcf().axes[0].get_ylabel(), names


@pytest.mark.parametrize('k', [2, 3])
def test_RED_F2_no_zero_or_negative_more_count(k):
    label, names = _long_ylabel(k)
    assert 'more' not in label and all(n in label for n in names), label


def test_CTL_F2_four_long_names_say_one_more():
    label, _ = _long_ylabel(4)
    assert '(+1 more)' in label, label


# --------------------------------------------------------------------------
# F-7: interval display shows the bin closure, "(a, b]"
# --------------------------------------------------------------------------
def test_RED_F7_profile_group_legend_shows_closure():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='p', group_by_bins=3)
    names = _texts(_legends(plt.gcf())[0])
    assert names and all(CLOSED.match(t) for t in names), names


def test_RED_F7_global_group_legend_shows_closure():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='p', group_by_bins=3,
                          group_by_scope='global', facet_by='a', facet_by_bins=2)
    names = [t for lg in _legends(plt.gcf()) for t in _texts(lg)]
    assert len(names) == 3 and all(CLOSED.match(t) for t in names), names


def test_CTL_F7_group_keys_in_stats_unchanged():
    """Display only: the stored group names stay 'a-b' (D-3)."""
    _, _, stats = DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='p',
                                         group_by_bins=3, return_data=True)
    groups = set(map(str, stats['profile_data']['group'])) if 'profile_data' in stats else set()
    assert groups and all(not g.startswith('(') for g in groups), groups


def test_CTL_F7_facet_titles_keep_the_13_97_default():
    """Architect option (b), 2026-11-20: facet titles keep the PHASE_13_97
    default text; facet_label_format='auto' gives '(a, b]' there."""
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, facet_by='p', facet_by_bins=3)
    titles = [ax.get_title() for ax in plt.gcf().axes if ax.get_title()]
    assert titles == ['p=-2.0--0.7', 'p=-0.7-0.7', 'p=0.7-2.0'], titles


def test_CTL_F7_histogram_group_legend_already_closed_form():
    DFDraw(_frame()).draw('y1', type='hist', bins=40, group_by='p', group_by_bins=3)
    names = _texts(_legends(plt.gcf())[0])
    assert names and all(CLOSED.match(t) for t in names), names


# --------------------------------------------------------------------------
# F-5 / P-5 / P-6: shared legend placement and identity
# --------------------------------------------------------------------------
def _per_cell_shared(df, facets, **extra):
    kw = (dict(facet_by='a', facet_by_bins=2) if facets == 1
          else dict(facet_by=['a', 'b'], facet_by_bins=[2, 2]))
    return DFDraw(df).draw('y1:x', type='profile', bins=8, group_by='p', group_by_bins=3,
                           legend='shared', **kw, **extra)


@pytest.mark.parametrize('facets', [1, 2], ids=['form-ii-merged-1d', 'form-i-bottom-2d'])
def test_RED_F5_cells_with_different_intervals_one_global_legend(facets):
    """Change rule, S4-A (Architect decision 1, 2026-11-20): with
    legend='shared' and bins given as a count, a profile's bins are computed
    once for the whole Figure (global scope) instead of being refused.  Was
    test_RED_F5_cells_with_different_intervals_refused (S2)."""
    fig, _, _ = _per_cell_shared(_cells_differ(_frame()), facets)
    assert len(fig.legends) == 1 and not any(ax.get_legend() for ax in fig.axes)
    assert fig.legends[0].get_title().get_text() == 'Colors: p (3 global bins)'
    assert [t for t in _texts(fig.legends[0]) if CLOSED.match(t)] == \
        ['(-2.0, -0.7]', '(-0.7, 0.7]', '(0.7, 2.0]']


def _no_figure_created(call, match):
    """The call is refused before any Figure is constructed (decision 1)."""
    import matplotlib.figure as mfig
    created = []
    original = mfig.Figure.__init__

    def counting(self, *a, **k):
        created.append(1)
        original(self, *a, **k)
    mfig.Figure.__init__ = counting
    try:
        with pytest.raises(ValueError, match=match):
            call()
    finally:
        mfig.Figure.__init__ = original
    assert not created, f'{len(created)} Figure(s) were created before the refusal'


@pytest.mark.parametrize('facets', [1, 2], ids=['1d', '2d'])
def test_S4A_D1_explicit_facet_scope_with_shared_refused_before_drawing(facets):
    _no_figure_created(
        lambda: _per_cell_shared(_cells_differ(_frame()), facets, group_by_scope='facet'),
        'group_by_scope')


def test_RED_F5_identical_cells_give_one_top_legend():
    _per_cell_shared(_cells_identical(_frame()), 2)
    fig = plt.gcf()
    assert len(fig.legends) == 1 and not any(ax.get_legend() for ax in fig.axes)
    assert _fig_box(fig, fig.legends[0]).y0 >= _grid_top(fig) - 1e-3


def test_CTL_F5_global_scope_shared_legend_on_top():
    DFDraw(_cells_differ(_frame())).draw('y1:x', type='profile', bins=8, group_by='p',
                                         group_by_bins=3, group_by_scope='global',
                                         facet_by=['a', 'b'], facet_by_bins=[2, 2],
                                         legend='shared')
    fig = plt.gcf()
    assert len(fig.legends) == 1
    assert _fig_box(fig, fig.legends[0]).y0 >= _grid_top(fig) - 1e-3


def test_RED_P5_scatter_shared_legend_on_top():
    DFDraw(_frame()).draw('y1:x', type='scatter', sample=3000, group_by='g', facet_by='a',
                          facet_by_bins=3, legend='shared')
    fig = plt.gcf()
    assert len(fig.legends) == 1
    assert _fig_box(fig, fig.legends[0]).y0 >= _grid_top(fig) - 1e-3


def test_RED_P6_no_empty_band_above_1d_shared_legend():
    """Provisional bound (S2 review): at most 6 % of the Figure height between
    the top of the shared legend and the bottom of the Figure title."""
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a',
                          facet_by_bins=3, legend='shared', auto_title=True)
    fig = plt.gcf()
    gap = _fig_box(fig, fig._suptitle).y0 - _fig_box(fig, fig.legends[0]).y1
    assert gap <= 0.06, f'empty band of {gap:.3f} of the Figure height'


def test_RED_F3_legend_does_not_cover_the_fit_box():
    DFDraw(_frame()).draw('r:x', type='profile', bins=20, quantiles=[0.11, 0.5, 0.9], fit='pol1')
    fig = plt.gcf()
    ax = fig.axes[0]
    fig.canvas.draw()
    legend = ax.get_legend().get_window_extent()
    boxes = [t for t in ax.texts if 'pol1' in t.get_text()]
    assert boxes and not any(t.get_window_extent().overlaps(legend) for t in boxes)


# --------------------------------------------------------------------------
# D-5 / D-6 (S3, after the Architect decisions recorded in the addendum)
# --------------------------------------------------------------------------
def _lightness(colour):
    return colorsys.rgb_to_hls(*to_rgb(colour))[1]


def _outline(line):
    """(colour, extra width) of the outline drawn around ``line``, or None."""
    effects = line.get_path_effects() or []
    strokes = [e for e in effects if type(e).__name__ == 'Stroke']
    if not strokes:
        return None
    gc = strokes[0]._gc
    return to_hex(gc.get('foreground')), gc.get('linewidth') - line.get_linewidth()


def test_RED_D5_fit_over_filled_bars_is_visible():
    """Change rule, S4-A (Architect decision 3b, 2026-11-20): the fit keeps
    the bar colour and gets a white outline (was: a darker shade, S3).  Was
    test_RED_D5_fit_over_filled_bars_is_a_darker_shade."""
    DFDraw(_frame()).draw('y1', type='hist', bins=60, fit='gauss')
    ax = plt.gcf().axes[0]
    bar, fit = ax.patches[0].get_facecolor(), ax.lines[0]
    assert to_hex(fit.get_color()) == to_hex(bar, keep_alpha=False)
    assert _outline(fit) == ('#ffffff', 2.0)


def test_RED_D6_level_labels_omitted_when_the_legend_shows_the_levels():
    DFDraw(_frame()).draw('[y1,y2,y3]:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9])
    ax = plt.gcf().axes[0]
    assert ax.get_legend() is not None
    assert not [t.get_text() for t in ax.texts if PCT.match(t.get_text())]


def test_CTL_D6_single_curve_keeps_its_level_labels():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9])
    labels = sorted(t.get_text() for t in plt.gcf().axes[0].texts if PCT.match(t.get_text()))
    assert labels == ['11%', '50%', '90%']


# --------------------------------------------------------------------------
# S2 controls and mutants (added 2026-11-20 with the S2 fixes)
# --------------------------------------------------------------------------
def _mutant_same_names_different_edges(df):
    """One cell holds one more point, a hundred-thousandth below -2: its bin
    edges differ from the other cells, but every printed name is the same."""
    extra = {'x': 1.0, 'a': 0.25, 'b': 0.25, 'p': -2.00001, 'g': 0,
             'y1': 10.0, 'y2': 20.0, 'y3': 30.0, 'y4': 40.0, 'r': 0.0}
    return pd.concat([_cells_identical(df), pd.DataFrame([extra])], ignore_index=True)


def test_S2_F5_mutant_equal_names_different_edges_refused():
    df = _mutant_same_names_different_edges(_frame())
    _, axes, _ = DFDraw(df).draw('y1:x', type='profile', bins=8, group_by='p',
                                 group_by_bins=3, facet_by=['a', 'b'],
                                 facet_by_bins=[2, 2], legend='all')
    names = {tuple(_texts(ax.get_legend())) for ax in np.ravel(axes)}
    assert len(names) == 1, 'the mutant must print the same names in every cell'
    plt.close('all')
    # S4-A (decision 1): with the default scope the bins are now global, so
    # the rounded-name mutant cannot produce a false legend any more; with an
    # explicit per-cell scope the call is refused before drawing.
    fig, _, _ = _per_cell_shared(df, 2)
    assert fig.legends[0].get_title().get_text() == 'Colors: p (3 global bins)'
    plt.close('all')
    _no_figure_created(lambda: _per_cell_shared(df, 2, group_by_scope='facet'),
                       'group_by_scope')


def test_S2_F5_hist_cells_with_different_intervals_refused():
    df = _cells_differ(_frame())
    _no_figure_created(
        lambda: DFDraw(df).draw('y1', type='hist', bins=30, group_by='p',
                                group_by_bins=3, facet_by='a', facet_by_bins=2,
                                legend='shared'),
        'group_by_scope')


def test_S2_F5_3d_refusal_leaves_no_page_open():
    df = _cells_differ(_frame())
    df['c'] = np.where(df['b'] < 0.5, 0, 1)
    _no_figure_created(
        lambda: DFDraw(df).draw('y1:x', type='profile', bins=8, group_by='p',
                                group_by_bins=3, facet_by=['a', 'b', 'c'],
                                facet_by_bins=[2, 2, None], legend='shared'),
        'group_by_scope')


def test_CTL_S2_F5_per_cell_intervals_without_shared_unchanged():
    """Per-cell intervals keep their per-cell legends when no shared legend
    is asked for (the D108-B default); nothing is refused."""
    fig, axes, _ = DFDraw(_cells_differ(_frame())).draw(
        'y1:x', type='profile', bins=8, group_by='p', group_by_bins=3,
        facet_by=['a', 'b'], facet_by_bins=[2, 2])
    assert not fig.legends
    assert all(ax.get_legend() is not None for ax in np.ravel(axes))


def test_CTL_S2_F5_global_domain_cell_without_a_group_one_legend():
    """V2-04 control: one global set of intervals, one cell lacks a group:
    still one Figure legend, with every group."""
    df = _frame()
    df = df[~((df['a'] < 0.5) & (df['p'] > 1.0))]
    fig, _, _ = DFDraw(df).draw('y1:x', type='profile', bins=8, group_by='p',
                                group_by_bins=3, group_by_scope='global',
                                facet_by=['a', 'b'], facet_by_bins=[2, 2],
                                legend='shared')
    assert len(fig.legends) == 1
    assert len([t for t in _texts(fig.legends[0]) if CLOSED.match(t)]) == 3


def test_CTL_S2_P5_caller_loc_keeps_the_old_generic_defaults():
    DFDraw(_frame()).draw('y1:x', type='scatter', sample=3000, group_by='g',
                          facet_by='a', facet_by_bins=3,
                          legend={'mode': 'shared', 'loc': 'lower center'})
    fig = plt.gcf()
    assert len(fig.legends) == 1
    assert fig.legends[0]._loc == 8 and fig.legends[0]._ncols == 1


def test_CTL_S2_P6_explicit_margins_unchanged():
    margins = (0.1, 0.95, 0.6, 0.15)
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a',
                          facet_by_bins=3, legend='shared', auto_title=True,
                          figure_margins=margins)
    assert abs(_grid_top(plt.gcf()) - 0.6) < 1e-9


def test_CTL_S2_F3_caller_legend_location_not_moved():
    fig, ax, _ = DFDraw(_frame()).draw('r:x', type='profile', bins=20,
                                       quantiles=[0.11, 0.5, 0.9], fit='pol1')
    ax.legend(loc='upper left')
    from dfextensions.dfdraw.drawer import _move_best_legends_off_text_boxes
    _move_best_legends_off_text_boxes(fig)
    assert ax.get_legend()._loc == 2


def test_CTL_S2_global_binned_group_no_unknown_keyword_warning():
    """S1b passed a private flag that the profile typo guard reported as an
    unknown keyword argument; S2 consumes it before the guard."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='p',
                              group_by_bins=3, group_by_scope='global',
                              facet_by=['a', 'b'], facet_by_bins=[2, 2])
    assert not [w for w in caught if 'Unknown keyword argument' in str(w.message)]


class _AliasTitles:
    """Stands for AliasDataFrame: dfdraw asks the data source for axis titles."""
    def get_axis_title(self, name):
        return {'p': 'p_T (GeV)'}.get(name)


@pytest.mark.parametrize('facets', [1, 2], ids=['1d-global', '2d-global'])
def test_S2_F12_global_group_legend_title_uses_the_alias_title(facets):
    """F-12: the 2-D global-scope legend title used the column name 'p'
    while every other group legend title uses the alias title (D-2)."""
    kw = (dict(facet_by='a', facet_by_bins=2) if facets == 1
          else dict(facet_by=['a', 'b'], facet_by_bins=[2, 2]))
    drawer = DFDraw(_frame())
    drawer._data_source = _AliasTitles()
    fig, _, _ = drawer.draw('y1:x', type='profile', bins=8, group_by='p', group_by_bins=3,
                            group_by_scope='global', **kw)
    assert [lg.get_title().get_text() for lg in fig.legends] == ['Colors: p_T (GeV) (3 global bins)']


# --------------------------------------------------------------------------
# S3 (2026-11-20): D-5 fit shade over filled bars, D-6 level labels
# --------------------------------------------------------------------------


def _fits(ax):
    """(peak x, colour) of every fit line (200 points) of a histogram axes."""
    return [(float(ln.get_xdata()[int(np.argmax(ln.get_ydata()))]), to_hex(ln.get_color()))
            for ln in ax.lines if len(ln.get_xdata()) == 200]


def _bars(ax):
    return {p.get_label(): to_hex(p.get_facecolor(), keep_alpha=False) for p in ax.patches}


def _fit_lines(ax):
    return [ln for ln in ax.lines if len(ln.get_xdata()) == 200]


@pytest.mark.parametrize('rank', range(10))
def test_S4A_D5_outline_for_every_palette_colour(rank):
    """Decision 3b: over filled bars the fit has the bar colour and a white
    outline (style fit.outline_color / fit.outline_width, 1 pt each side)."""
    fig, ax, _ = DFDraw(_frame()).hist('y1', bins=60, fit='gauss', color=_palette(rank))
    (fit,) = _fit_lines(ax)
    assert to_hex(fit.get_color()) == _palette(rank)
    assert _outline(fit) == ('#ffffff', 2.0)


def test_S4A_D5_black_bars_fit_visible():
    fig, ax, _ = DFDraw(_frame()).hist('y1', bins=60, fit='gauss', color='black')
    (fit,) = _fit_lines(ax)
    assert to_hex(fit.get_color()) == '#000000' and _outline(fit)[0] == '#ffffff'


def test_S4A_D5_outline_from_style_and_switch_off():
    from dfextensions.dfdraw.style import set_style
    try:
        set_style({'fit.outline_color': 'black', 'fit.outline_width': 0.5})
        fig, ax, _ = DFDraw(_frame()).hist('y1', bins=60, fit='gauss')
        assert _outline(_fit_lines(ax)[0]) == ('#000000', 1.0)
        plt.close('all')
        set_style({'fit.outline_width': 0})
        fig, ax, _ = DFDraw(_frame()).hist('y1', bins=60, fit='gauss')
        assert _outline(_fit_lines(ax)[0]) is None
    finally:
        set_style(None)


def test_CTL_S3_D5_step_histogram_fit_keeps_the_curve_colour():
    DFDraw(_frame()).draw('y1', type='hist', bins=60, fit='gauss', histtype='step')
    (fit,) = _fit_lines(plt.gcf().axes[0])
    assert to_hex(fit.get_color()) == _palette(0) and _outline(fit) is None


def test_S3_D5_group_fit_in_its_own_group_colour():
    """Groups with different centres: the fit of group g peaks at 10 + 3 g
    and must have group g's bar colour.  Before S3 the fits took the axes
    cycle in data order, so group 1's fit was drawn in group 0's colour."""
    df = _frame()
    df['z'] = df['y1'] + 3.0 * df['g']
    DFDraw(df).draw('z', type='hist', bins=80, fit='gauss', group_by='g')
    ax = plt.gcf().axes[0]
    bars = _bars(ax)
    for peak, colour in _fits(ax):
        g = str(int(round((peak - 10.0) / 3.0)))
        assert colour == bars[g], (peak, colour, bars)
    assert all(_outline(ln) == ('#ffffff', 2.0) for ln in _fit_lines(ax))


def test_S3_D5_facet_cell_fit_in_the_cell_bar_colour():
    """Before S3 the fit of a faceted histogram took the next cycle colour
    (orange over blue bars)."""
    DFDraw(_frame()).draw('y1', type='hist', bins=60, fit='gauss', facet_by='a', facet_by_bins=2)
    for ax in plt.gcf().axes:
        bar = to_hex(ax.patches[0].get_facecolor(), keep_alpha=False)
        assert [c for _, c in _fits(ax)] == [bar]


def test_CTL_S3_D5_profile_fit_keeps_the_curve_colour():
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, fit='pol1')
    ax = plt.gcf().axes[0]
    assert len({to_hex(ln.get_color()) for ln in ax.lines}) == 1
    assert all(_outline(ln) is None for ln in ax.lines)


def _level_labels(fig):
    return [[t for t in ax.texts if getattr(t, '_dfdraw_level_label', None) is not None]
            for ax in fig.axes]


def test_S3_D6_fresh_drawer_chain_labels_omitted():
    """AliasDataFrame-like chain: the rule is evaluated on the whole panel."""
    for k in (1, 2, 3):
        DFDraw(_frame()).draw(f'y{k}:x', type='profile', bins=8,
                              quantiles=[0.11, 0.5, 0.9], same=(k > 1))
    assert _level_labels(plt.gcf()) == [[]]


def test_S3_D6_faceted_figure_legend_labels_omitted():
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9],
                          facet_by='a', facet_by_bins=2)
    fig = plt.gcf()
    assert len(fig.legends) == 1 and all(not cell for cell in _level_labels(fig))


def test_S3_D6_without_legend_labels_kept_and_moved_apart():
    """(b): no legend names the levels, so every curve keeps its labels, at
    different places on its own line."""
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9],
                          facet_by='a', facet_by_bins=2, legend='none')
    for cell in _level_labels(plt.gcf()):
        assert sorted(t.get_text() for t in cell) == ['11%', '11%', '50%', '50%', '90%', '90%']
        xs = {}
        for t in cell:
            xs.setdefault(to_hex(t.get_color()), set()).add(round(float(t.xy[0]), 6))
        assert len(xs) == 2 and all(len(v) == 1 for v in xs.values())
        assert len({next(iter(v)) for v in xs.values()}) == 2


def test_S3_D6_labels_put_back_when_a_later_call_removes_the_legend():
    """The evaluation is repeated on the whole panel: labels removed while
    the legend named the levels come back when it no longer does."""
    from dfextensions.dfdraw.drawer import _resolve_level_labels
    DFDraw(_frame()).draw('[y1,y2,y3]:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9])
    fig = plt.gcf()
    assert _level_labels(fig) == [[]]
    fig.axes[0].get_legend().remove()
    _resolve_level_labels(fig)
    assert len(_level_labels(fig)[0]) == 9


def test_CTL_S3_D6_quantile_style_color_has_no_labels():
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, quantiles=[0.11, 0.5, 0.9],
                          quantile_style='color')
    assert _level_labels(plt.gcf()) == [[]]


# --------------------------------------------------------------------------
# S2b (2026-11-20): corrections after the S2 review (GPG35 summary)
# --------------------------------------------------------------------------
# S2-C01: per-cell intervals with identical edges, one interval missing in one
# cell.  Every curve's value tells its interval: y = 10 + 10 * rank (rank 0,
# 1, 2 for the lower, middle and upper third of p in [-2, 2]).
EDGES3 = [-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]


def _sparse_frame(missing, facets=1, n=6000, seed=7, endpoints=False):
    """Cell 0 lacks the interval of rank ``missing`` (None: no gap).

    ``endpoints=False``: the bins are given as explicit edges (EDGES3), the
    same in every cell by definition.  ``endpoints=True`` (the reviewers'
    case, middle gap only): every cell holds p = -2 and p = +2, so the bins
    computed from the data of each cell have identical edges."""
    rng = np.random.default_rng(seed)
    rows = []
    cells = [(0.25, 0.25), (0.75, 0.25)] if facets == 1 else \
        [(a, b) for a in (0.25, 0.75) for b in (0.25, 0.75)]
    for c, (a, b) in enumerate(cells):
        for rank in (0, 1, 2):
            if c == 0 and rank == missing:
                continue
            p = rng.uniform(EDGES3[rank] + 0.05, EDGES3[rank + 1] - 0.05, n)
            rows.append(pd.DataFrame({'x': rng.uniform(0, 4, n), 'a': a, 'b': b, 'p': p,
                                      'y1': 10.0 + 10.0 * rank + rng.normal(0, 0.1, n)}))
        if endpoints:
            rows.append(pd.DataFrame({'x': [1.0, 1.0], 'a': a, 'b': b, 'p': [-2.0, 2.0],
                                      'y1': [10.0, 30.0]}))
    return pd.concat(rows, ignore_index=True)


def _rank_colours(fig):
    """{rank: set of colours} over all cells, from each curve's median value."""
    out = {}
    for ax in fig.axes:
        for ln in ax.lines:
            y = np.asarray(ln.get_ydata(), float)
            if y.size < 3 or getattr(ln, '_dfdraw_quantile', None) is not None:
                continue
            rank = int(round((np.nanmedian(y) - 10.0) / 10.0))
            out.setdefault(rank, set()).add(to_hex(ln.get_color()))
    return out


def _sparse_draw(missing, facets, endpoints=False, **kw):
    fb = dict(facet_by='a', facet_by_bins=2) if facets == 1 else \
        dict(facet_by=['a', 'b'], facet_by_bins=[2, 2])
    return DFDraw(_sparse_frame(missing, facets, endpoints=endpoints)).draw(
        'y1:x', type='profile', bins=4, group_by='p',
        group_by_bins=(3 if endpoints else EDGES3), **fb, **kw)


@pytest.mark.parametrize('facets', [1, 2], ids=['1d', '2d'])
@pytest.mark.parametrize('legend', [None, 'shared'], ids=['cell-legends', 'shared'])
def test_S2b_C01_reviewers_case_data_edges_middle_missing(facets, legend):
    """GPG37's counterexample: edges computed from each cell's data are equal
    (p = -2 and +2 in every cell), the middle interval is missing in cell 0."""
    _sparse_draw(1, facets, endpoints=True, **({'legend': legend} if legend else {}))
    colours = _rank_colours(plt.gcf())
    assert colours == {r: {_palette(r)} for r in (0, 1, 2)}, colours


@pytest.mark.parametrize('facets', [1, 2], ids=['1d', '2d'])
@pytest.mark.parametrize('missing', [0, 1, 2], ids=['first', 'middle', 'last'])
@pytest.mark.parametrize('legend', [None, 'shared'], ids=['cell-legends', 'shared'])
def test_S2b_C01_interval_keeps_its_colour_when_another_is_missing(missing, facets, legend):
    """Each interval has one colour in every cell: palette(rank in the full
    ordered interval list, empty intervals included)."""
    _sparse_draw(missing, facets, **({'legend': legend} if legend else {}))
    colours = _rank_colours(plt.gcf())
    assert colours == {r: {_palette(r)} for r in (0, 1, 2)}, colours


@pytest.mark.parametrize('missing', [0, 1, 2], ids=['first', 'middle', 'last'])
def test_S2b_C01_shared_legend_one_entry_per_interval_in_order(missing):
    fig, _, _ = _sparse_draw(missing, 2, legend='shared')
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    entries = [(t.get_text(), to_hex(h.get_color()))
               for t, h in zip(legend.get_texts(), legend.legend_handles)]
    assert [c for _, c in entries] == [_palette(r) for r in (0, 1, 2)], entries
    assert [t for t, _ in entries] == ['(-2.0, -0.7]', '(-0.7, 0.7]', '(0.7, 2.0]'], entries


def test_S2b_C01_histogram_interval_keeps_its_colour():
    df = _sparse_frame(1)
    fig, _, _ = DFDraw(df).draw('y1', type='hist', bins=60, range=(5, 35), group_by='p',
                                group_by_bins=EDGES3, facet_by='a', facet_by_bins=2)
    by_label = {}
    for ax in fig.axes:
        for p in ax.patches:
            if p.get_label() and not p.get_label().startswith('_'):
                by_label.setdefault(p.get_label(), set()).add(
                    to_hex(p.get_facecolor(), keep_alpha=False))
    assert all(len(v) == 1 for v in by_label.values()), by_label
    ranked = sorted(by_label, key=lambda s: float(s.strip('([').split(',')[0]))
    assert [next(iter(by_label[k])) for k in ranked] == [_palette(r) for r in range(3)]


def test_CTL_S2b_C01_unfaceted_complete_groups_unchanged():
    """With every interval present the colours are the palette order, as before."""
    fig, ax, _ = DFDraw(_sparse_frame(None)).draw('y1:x', type='profile', bins=4,
                                                  group_by='p', group_by_bins=EDGES3)
    assert _rank_colours(fig) == {r: {_palette(r)} for r in (0, 1, 2)}


def test_S2b_C01_group_marker_kept_when_a_group_is_missing():
    """The marker follows the group's rank too (it followed the position in
    the cell): a categorical group missing from one cell, and a binned one."""
    df = _frame()
    df['yg'] = 10.0 * (df['g'] + 1) + 0.1 * df['y1']
    df = df[~((df['a'] >= 0.5) & (df['g'] == 1))]       # group 1 missing in cell 1
    DFDraw(df).draw('yg:x', type='profile', bins=6, facet_by='a', facet_by_bins=2, group_by='g')
    styles = {}
    for ax in plt.gcf().axes:
        for ln in ax.lines:
            if len(ln.get_xdata()) > 2 and str(ln.get_marker()) not in ('_', 'None', ''):
                g = int(round((np.nanmedian(ln.get_ydata()) - 1.0) / 10.0)) - 1
                styles.setdefault(g, set()).add((to_hex(ln.get_color()), str(ln.get_marker())))
    assert all(len(v) == 1 for v in styles.values()), styles
    plt.close('all')
    _sparse_draw(1, 2)
    markers = {}
    for ax in plt.gcf().axes:
        for ln in ax.lines:
            y = np.asarray(ln.get_ydata(), float)
            if y.size > 2 and str(ln.get_marker()) not in ('_', 'None', ''):
                markers.setdefault(int(round((np.nanmedian(y) - 10.0) / 10.0)), set()).add(
                    str(ln.get_marker()))
    assert all(len(v) == 1 for v in markers.values()), markers


# S2-C03: a non-empty cell whose intervals cannot be verified is not skipped.
def test_S2b_C03_cell_without_interval_record_refused():
    from dfextensions.dfdraw.drawer import _refuse_shared_over_cell_intervals
    fig, axes = plt.subplots(1, 3)
    edges = np.array([-2.004, -0.6667, 0.6667, 2.0])
    axes[0]._dfdraw_group_bin_edges = [edges]
    axes[1]._dfdraw_group_bin_edges = [edges.copy()]
    axes[2].plot([0, 1], [0, 1])            # data, but no interval record
    with pytest.raises(ValueError, match='group_by_scope'):
        _refuse_shared_over_cell_intervals(fig, axes)


def test_CTL_S2b_C03_empty_cell_without_record_accepted():
    from dfextensions.dfdraw.drawer import _refuse_shared_over_cell_intervals
    fig, axes = plt.subplots(1, 3)
    edges = np.array([-2.004, -0.6667, 0.6667, 2.0])
    axes[0]._dfdraw_group_bin_edges = [edges]
    axes[1]._dfdraw_group_bin_edges = [edges.copy()]
    _refuse_shared_over_cell_intervals(fig, axes)     # axes[2] has no data
    assert plt.fignum_exists(fig.number)


def test_S2b_C03_two_groupings_in_one_cell_both_compared():
    from dfextensions.dfdraw.drawer import _refuse_shared_over_cell_intervals
    fig, axes = plt.subplots(1, 2)
    e1 = np.array([-2.004, -0.6667, 0.6667, 2.0])
    axes[0]._dfdraw_group_bin_edges = [e1, e1.copy()]
    axes[1]._dfdraw_group_bin_edges = [e1.copy(), e1 * 0.5]
    with pytest.raises(ValueError, match='group_by_scope'):
        _refuse_shared_over_cell_intervals(fig, axes)


# S2-C02 (Architect waiver, option a): the refusal comes after the cells were
# drawn; every Figure the call made is closed, nothing of the caller changes.
def test_S2b_C02_refusal_keeps_the_callers_figure_current_and_unchanged():
    own = plt.figure()
    own_ax = own.add_subplot()
    own_ax.plot([0, 1], [0, 1])
    before = (plt.get_fignums(), len(own_ax.lines), plt.gcf())
    with pytest.raises(ValueError, match='group_by_scope'):
        _per_cell_shared(_cells_differ(_frame()), 2, group_by_scope='facet')
    assert (plt.get_fignums(), len(own_ax.lines), plt.gcf()) == before


# S2-C04 / S2-C07: F-3 counts every drawn artist and reports when no free place exists.
def _box_and_points_axes():
    """Axes with a framed text in the upper left, scatter points everywhere
    except the upper left and the lower right, and a one-entry legend."""
    fig, ax = plt.subplots()
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 1, (6000, 2))
    free = (((xy[:, 0] < 0.35) & (xy[:, 1] > 0.65))
            | ((xy[:, 0] > 0.65) & (xy[:, 1] < 0.35)))
    ax.scatter(xy[~free, 0], xy[~free, 1], s=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([], [], label='curve')
    ax.text(0.02, 0.98, 'fit\nbox\ntext', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round'))
    legend = ax.legend(loc='best')
    fig.canvas.draw()
    return fig, ax, legend, [ax.texts[0]]


def test_S2b_C04_best_legend_off_box_counts_scatter_points():
    """S4-A (review C-S2b-04): the location choice is tested directly, so the
    test does not depend on where a given Matplotlib version puts 'best'.
    Counting only line points (S2), the upper right looked free; with the
    scatter points counted, the lower right (no box, no points) is chosen."""
    from dfextensions.dfdraw.drawer import _choose_legend_location
    fig, ax, legend, boxes = _box_and_points_axes()
    assert _choose_legend_location(ax, legend, boxes, fig.canvas.get_renderer()) == 4


def test_S4A_C03_long_line_segment_counted():
    """Review C-S2b-03: a straight segment whose two vertices lie outside the
    legend area still crosses it; the occupancy count samples along it."""
    from dfextensions.dfdraw.drawer import _drawn_data_points, _set_legend_loc
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([], [], label='segment')
    legend = ax.legend()
    _set_legend_loc(legend, 4)
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    # a horizontal segment through the middle of the legend, with both
    # vertices outside the axes
    yc = ax.transData.inverted().transform((box.x0, 0.5 * (box.y0 + box.y1)))[1]
    ax.lines[0].set_data([0.2, 1.2], [yc, yc])
    pts = _drawn_data_points(ax)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    vertices = ax.lines[0].get_transform().transform(ax.lines[0].get_xydata())
    assert not any(box.contains(*v) for v in vertices)     # vertex-only count: 0
    assert inside.sum() > 0


def test_S2b_C07_no_free_place_warns_and_keeps_best():
    import warnings
    from dfextensions.dfdraw.drawer import _move_best_legends_off_text_boxes
    fig, ax = plt.subplots(figsize=(2.5, 2))
    for i in range(6):
        ax.plot([0, 1], [i, i], label=f'curve {i}')
    for x, y, ha, va in ((0.02, 0.98, 'left', 'top'), (0.98, 0.98, 'right', 'top'),
                         (0.02, 0.02, 'left', 'bottom'), (0.98, 0.02, 'right', 'bottom'),
                         (0.5, 0.5, 'center', 'center')):
        ax.text(x, y, 'box\nbox\nbox', transform=ax.transAxes, ha=ha, va=va,
                bbox=dict(boxstyle='round'), fontsize=14)
    ax.legend(loc='best')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _move_best_legends_off_text_boxes(fig)
    assert ax.get_legend()._loc == 0
    assert any('F-3' in str(w.message) for w in caught)


# S2-C05: constrained layout keeps the legend off the panels.
def test_S2b_C05_constrained_layout_shared_legend_does_not_overlap():
    fig = plt.figure(figsize=(6, 4), layout='constrained')
    ax = fig.add_subplot()
    DFDraw(_frame()).draw('y1:x', type='scatter', sample=2000, group_by='g', ax=ax,
                          legend='shared')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    assert ax.get_tightbbox(renderer).y1 <= legend.y0 + 0.5


# S2-C06 (Architect option a): at most 4 % of the Figure height between the
# legend and the title, at three Figure sizes, and no overlap.
@pytest.mark.parametrize('figsize', [(12, 4), (16, 6), (10, 8)], ids=['12x4', '16x6', '10x8'])
def test_S2b_C06_band_bound_at_three_sizes(figsize):
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a',
                          facet_by_bins=3, legend='shared', auto_title=True, figsize=figsize)
    fig = plt.gcf()
    title, legend = _fig_box(fig, fig._suptitle), _fig_box(fig, fig.legends[0])
    assert 0.0 <= title.y0 - legend.y1 <= 0.04, (title.y0, legend.y1)
    assert legend.y0 >= _grid_top(fig) - 1e-3


# --------------------------------------------------------------------------
# S4-A (2026-11-20): corrections after the S3 + S2b review (GPG35 summary)
# and the Architect decisions 1, 2b, 3b of 2026-11-20
# --------------------------------------------------------------------------
@pytest.mark.parametrize('kind', ['profile', 'hist'])
def test_S4A_D1_explicit_edges_shared_allowed(kind):
    """Bins given as explicit edges are the same in every cell by definition:
    one shared legend, no refusal, also for histograms."""
    df = _cells_differ(_frame())
    expr = 'y1:x' if kind == 'profile' else 'y1'
    fig, _, _ = DFDraw(df).draw(expr, type=kind, bins=8, group_by='p',
                                group_by_bins=[-2.0, -1.0, 0.0, 1.0, 2.0],
                                facet_by='a', facet_by_bins=2, legend='shared')
    assert len(fig.legends) == 1 and not any(ax.get_legend() for ax in fig.axes)


def test_S4A_D1_cells_without_shared_keep_per_cell_bins():
    """Without legend='shared' nothing changes: bins per cell, one legend per cell."""
    fig, axes, _ = DFDraw(_cells_differ(_frame())).draw(
        'y1:x', type='profile', bins=8, group_by='p', group_by_bins=3,
        facet_by='a', facet_by_bins=2)
    names = [tuple(_texts(ax.get_legend())) for ax in np.ravel(axes)]
    assert len(set(names)) == 2, names


def test_S4A_C02_two_records_against_one_refused():
    """Review C-S2b-02: a cell with two groupings and a cell with one."""
    from dfextensions.dfdraw.drawer import _refuse_shared_over_cell_intervals
    fig, axes = plt.subplots(1, 2)
    e = np.array([-2.004, -0.6667, 0.6667, 2.0])
    axes[0]._dfdraw_group_bin_edges = [e, e.copy()]
    axes[1]._dfdraw_group_bin_edges = [e.copy()]
    with pytest.raises(ValueError, match='group_by_scope'):
        _refuse_shared_over_cell_intervals(fig, axes)


def test_CTL_S4A_C02_no_binned_grouping_nothing_to_verify():
    """Categorical groups leave no interval record: nothing to verify."""
    from dfextensions.dfdraw.drawer import _refuse_shared_over_cell_intervals
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.plot([0, 1], [0, 1])
    _refuse_shared_over_cell_intervals(fig, axes)
    assert plt.fignum_exists(fig.number)


def test_S4A_C_S3_01_cleared_axes_do_not_get_old_labels_back():
    """Review C-S3-01: labels left out on an axes must not come back after the
    axes is cleared and reused for another plot."""
    fig, ax, _ = DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8,
                                       quantiles=[0.11, 0.5, 0.9])
    ax = fig.axes[0]
    assert not _level_labels(fig)[0]
    ax.clear()
    DFDraw(_frame()).draw('y3:x', type='profile', bins=8, ax=ax)
    assert not [t for t in ax.texts if PCT.match(t.get_text())]


def test_S4A_C_S3_02_same_colour_curves_are_two_curves():
    """Review C-S3-02: two curves in the same user colour draw the same
    levels; the legend names the levels, so the labels are left out."""
    d = DFDraw(_frame())
    d.profile('y1:x', bins=8, quantiles=[0.11, 0.5, 0.9], color='red')
    d.profile('y2:x', bins=8, quantiles=[0.11, 0.5, 0.9], color='red', same=True)
    assert _level_labels(plt.gcf()) == [[]]


def test_CTL_S4A_C_S3_02_same_colour_without_legend_labels_kept_apart():
    from dfextensions.dfdraw.drawer import _resolve_level_labels
    d = DFDraw(_frame())
    d.profile('y1:x', bins=8, quantiles=[0.11, 0.5, 0.9], color='red')
    d.profile('y2:x', bins=8, quantiles=[0.11, 0.5, 0.9], color='red', same=True)
    fig = plt.gcf()
    fig.axes[0].get_legend().remove()
    _resolve_level_labels(fig)
    labels = _level_labels(fig)[0]
    assert len(labels) == 6
    assert len({round(float(t.xy[0]), 6) for t in labels}) == 2


@pytest.mark.parametrize('figsize', [(12, 4), (16, 6), (10, 8)], ids=['12x4', '16x6', '10x8'])
def test_S4A_D2_gap_is_the_style_value_in_points(figsize):
    """Decision 2b: the gap between the legend and the title is the style key
    legend.figure_title_gap_pt (default 6 points), at every Figure size; at
    most one text line."""
    DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a',
                          facet_by_bins=3, legend='shared', auto_title=True, figsize=figsize)
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    gap_pt = (fig._suptitle.get_window_extent(r).y0
              - fig.legends[0].get_window_extent(r).y1) * 72.0 / fig.dpi
    assert 5.0 <= gap_pt <= 7.0, gap_pt
    assert gap_pt <= fig.legends[0].prop.get_size_in_points() * 1.5


@pytest.mark.parametrize('route', ['composer', 'generic'])
def test_S4A_D2_gap_follows_the_style_key(route):
    from dfextensions.dfdraw.style import set_style
    try:
        set_style({'legend.figure_title_gap_pt': 20.0})
        if route == 'composer':
            DFDraw(_frame()).draw('y1:x', type='profile', bins=8, group_by='g', facet_by='a',
                                  facet_by_bins=3, legend='shared', auto_title=True)
        else:
            DFDraw(_frame()).draw('y1:x', type='scatter', sample=2000, group_by='g',
                                  facet_by='a', facet_by_bins=3, legend='shared', title='T')
        fig = plt.gcf()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        gap_pt = (fig._suptitle.get_window_extent(r).y0
                  - fig.legends[0].get_window_extent(r).y1) * 72.0 / fig.dpi
        assert 19.0 <= gap_pt <= 21.0, gap_pt
    finally:
        set_style(None)


def _overlaps(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(r)
    bad = []
    title = fig._suptitle
    if title is not None and title.get_text() and legend.overlaps(title.get_window_extent(r)):
        bad.append('title')
    for ax in fig.axes:
        box = ax.get_tightbbox(r) if ax.get_visible() else None
        if box is not None and legend.overlaps(box):
            bad.append('axes')
    if legend.y1 > fig.bbox.y1 + 0.5:
        bad.append('above the Figure')
    return bad


@pytest.mark.parametrize('layout', [None, 'tight', 'constrained'])
@pytest.mark.parametrize('figsize', [(2.5, 2), (4, 3)], ids=['2.5x2', '4x3'])
@pytest.mark.parametrize('kind', ['scatter', 'profile'])
def test_S4A_C01_caller_axes_shared_legend_no_overlap(layout, figsize, kind):
    """Review C-S2b-01: the shared Figure legend on a caller's Figure never
    overlaps the title or the panels, for every layout engine and small sizes."""
    fig = plt.figure(figsize=figsize, layout=layout)
    fig.suptitle('Title')
    kw = dict(sample=2000) if kind == 'scatter' else dict(bins=8)
    DFDraw(_frame()).draw('y1:x', type=kind, group_by='g', ax=fig.add_subplot(),
                          legend='shared', **kw)
    assert _overlaps(fig) == []


@pytest.mark.parametrize('figsize', [(3, 2), (4, 3), (6, 3)], ids=['3x2', '4x3', '6x3'])
@pytest.mark.parametrize('kind', ['scatter', 'profile'])
def test_S4A_C01_faceted_shared_legend_no_overlap(figsize, kind):
    kw = dict(sample=2000) if kind == 'scatter' else dict(bins=8)
    DFDraw(_frame()).draw('y1:x', type=kind, group_by='g', facet_by='a', facet_by_bins=3,
                          legend='shared', figsize=figsize, title='Title', **kw)
    assert _overlaps(plt.gcf()) == []


def test_S4A_C_R1_warning_names_remedies_that_work():
    import warnings
    from dfextensions.dfdraw.drawer import _move_best_legends_off_text_boxes
    fig, ax = plt.subplots(figsize=(2.5, 2))
    for i in range(6):
        ax.plot([0, 1], [i, i], label=f'curve {i}')
    for x, y, ha, va in ((0.02, 0.98, 'left', 'top'), (0.98, 0.98, 'right', 'top'),
                         (0.02, 0.02, 'left', 'bottom'), (0.98, 0.02, 'right', 'bottom'),
                         (0.5, 0.5, 'center', 'center')):
        ax.text(x, y, 'box\nbox\nbox', transform=ax.transAxes, ha=ha, va=va,
                bbox=dict(boxstyle='round'), fontsize=14)
    ax.legend(loc='best')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _move_best_legends_off_text_boxes(fig)
    text = ' '.join(str(w.message) for w in caught)
    assert 'figsize' in text and "legend='shared'" in text and 'fit.position' in text
    assert "'mode': 'all'" not in text
