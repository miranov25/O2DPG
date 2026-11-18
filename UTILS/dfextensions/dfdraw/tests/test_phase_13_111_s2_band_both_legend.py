"""PHASE_13_111_DF S2 - explicit band check, central='both' check, grey quantile legend samples.

Plain-language background
-------------------------
Three decisions of the Architect (2026-11-18, proposal v0.2):

* D-4 (a): ``quantile_mode='band'`` draws one filled band between a symmetric
  pair of levels around the median and accepts exactly three levels
  ``[q, 0.5, 1-q]``.  Any other list used to crash, leave a Figure open, draw
  the band between q[0] and q[2] (5 % to 50 % for five levels), or silently
  draw error bars instead (2-level facets).  It is now refused before any
  Figure, on every route (profile, faceted profile, ``draw_profile``,
  overlay, faceted overlay), with a message that names the alternatives.
* D-7 (a): ``central='both'`` draws its median line dashed, the line style of
  the first quantile level.  With discrete quantile lines the two cannot be
  told apart and a 0.5 level is drawn twice, so this combination is refused
  before any Figure.  With a band or error bars it stays allowed.
* D-1 (b): when two or more curves draw the same quantile level, the legend
  shows that level once, in its line style, in neutral grey (as the factored
  legend of AD-59 already does).  A single curve keeps its own colour; curve
  and group entries stay real artists.

``test_RED_*`` fail on the start code (tag 78281b26); ``test_CTL_*`` pass on
both.  Tests named ``RED_S1_D1`` / ``RED_S9_D7`` check an S2 rule that also needs
the S1 colours, so they fail on the start code for the S1 reason.  Curve k lies
at y = 10*k, as in the S0 module.

Seat: Fable5.1_5:draw-coder (PHASE_13_111 coder), 2026-11-18.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from dfextensions.dfdraw.drawer import DFDraw
from dfextensions.dfdraw.plots.profile import draw_profile

from tests.test_phase_13_111_s0_identity_colour_red import (   # noqa: E402  shared helpers
    Q3, _frame, _check, _data_axes, _profile_pairs, _hist_pairs, _legend_entries,
)

GREY = to_hex('gray')


@pytest.fixture(autouse=True)
def _close_all():
    plt.close('all')
    yield
    plt.close('all')


def _refused(call, match):
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match=match):
        call()
    assert set(plt.get_fignums()) == before, 'a Figure was left open'


# --------------------------------------------------------------------------
# D-4: explicit band
# --------------------------------------------------------------------------
BAD_BANDS = [[0.16, 0.84], [0.1, 0.3, 0.9], [0.05, 0.25, 0.5, 0.75, 0.95], [0.5]]


@pytest.mark.parametrize('q', BAD_BANDS, ids=['pair', 'asym-triple', 'five', 'one'])
def test_RED_D4_profile_method_refuses_non_triple_band(q):
    _refused(lambda: DFDraw(_frame()).profile('y1:x', bins=4, quantiles=q,
                                              quantile_mode='band'), 'band')


@pytest.mark.parametrize('q', BAD_BANDS[:3], ids=['pair', 'asym-triple', 'five'])
def test_RED_D4_low_level_draw_profile_refuses_non_triple_band(q):
    _refused(lambda: draw_profile(_frame(), 'x', 'y1', bins=4, quantiles=q,
                                  quantile_mode='band'), 'band')


@pytest.mark.parametrize('q', BAD_BANDS[:3], ids=['pair', 'asym-triple', 'five'])
def test_RED_D4_overlay_layers_refuse_non_triple_band(q):
    layers = [{'type': 'hist2d', 'bins': 10},
              {'type': 'profile', 'bins': 10, 'quantiles': q, 'quantile_mode': 'band'}]
    _refused(lambda: DFDraw(_frame()).overlay('y1:x', layers=layers), 'band')


def test_RED_D4_message_names_the_alternatives():
    with pytest.raises(ValueError) as e:
        DFDraw(_frame()).profile('y1:x', bins=4, quantiles=[0.16, 0.84], quantile_mode='band')
    for word in ('error_bars', 'nested_band', 'discrete', '[q, 0.5, 1-q]'):
        assert word in str(e.value)


@pytest.mark.parametrize('route', ['profile', 'facet1', 'facet2', 'overlay', 'overlay-facet'])
def test_CTL_D4_symmetric_triple_band_drawn_on_every_route(route):
    kind = 'hist2d+profile' if route.startswith('overlay') else 'profile'
    kw = {}
    if route in ('facet1', 'overlay-facet'):
        kw = dict(facet_by='a', facet_by_bins=2)
    elif route == 'facet2':
        kw = dict(facet_by=['a', 'b'], facet_by_bins=[2, 2])
    DFDraw(_frame()).draw('y1:x', type=kind, quantiles=[0.16, 0.5, 0.84],
                          quantile_mode='band', **kw)
    fig = plt.gcf()
    bands = [c for ax in _data_axes(fig) for c in ax.collections
             if 'PolyCollection' in type(c).__name__ and 'Quad' not in type(c).__name__]
    assert bands, f'{route}: no band drawn'


def test_CTL_D4_symmetric_triple_band_values():
    df = _frame()
    DFDraw(df).draw('y1:x', type='profile', bins=1, range=(0.0, 4.0),
                    quantiles=[0.16, 0.5, 0.84], quantile_mode='band')
    band = [c for c in plt.gcf().axes[0].collections if 'PolyCollection' in type(c).__name__][0]
    v = band.get_paths()[0].vertices[:, 1]
    lo, hi = np.quantile(df['y1'], [0.16, 0.84])
    assert abs(v.min() - lo) < 1e-9 and abs(v.max() - hi) < 1e-9


# --------------------------------------------------------------------------
# D-7: central='both' with discrete quantile lines
# --------------------------------------------------------------------------
@pytest.mark.parametrize('expr,q,mode', [
    ('y1:x', [0.11, 0.5, 0.9], 'auto'),
    ('y1:x', [0.11, 0.9], 'auto'),
    ('[y1,y2]:x', [0.11, 0.5, 0.9], 'auto'),
    ('y1:x', [0.16, 0.5, 0.84], 'discrete'),
], ids=['single-3', 'single-2', 'vector-3', 'explicit-discrete'])
def test_RED_D7_central_both_with_discrete_quantiles_refused(expr, q, mode):
    _refused(lambda: DFDraw(_frame()).draw(expr, type='profile', bins=4, quantiles=q,
                                           quantile_mode=mode, central='both'),
             "central='both'")


def test_RED_D7_faceted_central_both_with_discrete_quantiles_refused():
    _refused(lambda: DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=4, quantiles=Q3,
                                           central='both', facet_by='a', facet_by_bins=2),
             "central='both'")


@pytest.mark.parametrize('q', [[0.16, 0.84], [0.16, 0.5, 0.84]], ids=['error-bars', 'band'])
def test_RED_S9_D7_central_both_with_error_bars_or_band_in_curve_colours(q):
    # Allowed by D-7; fails on the start code only because of the S-9 colour.
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=4, quantiles=q, central='both')
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2})


# --------------------------------------------------------------------------
# D-1: grey legend samples for quantile levels shared by curves
# --------------------------------------------------------------------------
def _level_entries(fig):
    return [(t, c) for t, c in _legend_entries(fig) if t.startswith('q=') or t == 'median']


def _assert_grey_levels(fig, n_levels=3):
    levels = _level_entries(fig)
    assert len(levels) == n_levels, levels                      # each level once
    assert all(c == GREY for _, c in levels), levels


@pytest.mark.parametrize('route', ['unfaceted', 'facet1', 'facet2', 'selection', 'same-chain'])
def test_RED_D1_shared_levels_shown_once_in_grey(route):
    d = DFDraw(_frame())
    if route == 'same-chain':
        d.draw('y1:x', type='profile', bins=6, quantiles=Q3)
        d.draw('y2:x', type='profile', bins=6, quantiles=Q3, same=True)
    elif route == 'selection':
        d.draw('ys:x', type='profile', bins=6, quantiles=Q3,
               selection_vector=['t<50', 't>=50'], vector_compose='outer')
    else:
        kw = {'facet1': dict(facet_by='a', facet_by_bins=2),
              'facet2': dict(facet_by=['a', 'b'], facet_by_bins=[2, 2], legend='shared')}.get(route, {})
        d.draw('[y1,y2,y3]:x', type='profile', bins=6, quantiles=Q3, **kw)
    _assert_grey_levels(plt.gcf())


def test_RED_D1_histogram_vector_quantile_markers_grey():
    DFDraw(_frame()).draw('[y1,y2]', type='hist', bins=70, range=(0.0, 35.0),
                          quantiles=[0.16, 0.5, 0.84])
    fig = plt.gcf()
    _assert_grey_levels(fig)
    curve = _check(_hist_pairs(fig.axes[0]), {1, 2})
    names = dict(_legend_entries(fig))
    assert names.get('y1') == curve[1] and names.get('y2') == curve[2], names


def test_RED_S1_D1_single_curve_levels_keep_the_curve_colour():
    # One curve: no grey (D-1); fails on the start code because of S-1.
    DFDraw(_frame()).draw('y1:x', type='profile', bins=6, quantiles=Q3)
    fig = plt.gcf()
    curve = _check(_profile_pairs(fig.axes[0]), {1})
    assert {c for _, c in _level_entries(fig)} == {curve[1]}


def test_RED_S1_D1_curve_entries_stay_real_colours():
    # Fails on the start code because of S-1 (curve 1's children).
    DFDraw(_frame()).draw('[y1,y2,y3]:x', type='profile', bins=6, quantiles=Q3,
                          facet_by=['a', 'b'], facet_by_bins=[2, 2], legend='shared')
    fig = plt.gcf()
    curve = _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})
    names = dict(_legend_entries(fig))
    assert [names.get(f'y{k}') for k in (1, 2, 3)] == [curve[1], curve[2], curve[3]], names
