"""PHASE_13_111_DF S1 - identity colour: inventory extensions and negative controls.

Plain-language background
-------------------------
S0 (``test_phase_13_111_s0_identity_colour_red.py``) holds the RED tests for the
reported family: the first curve's quantile lines, labels, fit and median line
borrowed other curves' colours; its legend name could land on another curve;
1-level facets started at the second palette colour; a group missing from a
facet cell shifted the other groups' colours.

This module adds the inventory members that S0 listed for S1, each checked with
the same rule (curve k lies at value 10*k; every drawn artist is assigned to a
curve by its position; one colour per curve, none shared, palette order on fresh
axes):

* ``central='median'`` with discrete quantiles;
* three-level facets;
* a custom ``colors.palette`` style;
* ``draw_batch``;
* histogram groups missing from a facet cell (1- and 2-level), the histogram
  member of S-8;
* group intervals computed per cell (13.108 R2) keep their per-cell colours;
* the public ``same=True`` chain for a profile keeps each name on its own curve;
* negative controls: with the S1 colour resolution switched off, the S0 rule
  fails again (the tests would catch a regression).

Tests named ``test_RED_*`` fail on the start code (tag 78281b26) and pass after S1.
``test_CTL_*`` pass on both.  ``test_NEG_*`` prove that the checks detect the
defect when the fix is removed.

Seat: Fable5.1_5:draw-coder (PHASE_13_111 coder), 2026-11-18.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dfextensions.dfdraw.drawer import DFDraw
from dfextensions.dfdraw.style import get_style_value, set_style

from tests.test_phase_13_111_s0_identity_colour_red import (   # noqa: E402  shared helpers
    Q3, _frame, _check, _data_axes, _profile_pairs, _hist_pairs, _legend_entries,
    _missing_group_frame, _palette,
)


@pytest.fixture(autouse=True)
def _close_all():
    plt.close('all')
    yield
    plt.close('all')


# --------------------------------------------------------------------------
# RED on the start code, GREEN after S1
# --------------------------------------------------------------------------
def test_RED_central_median_vector_discrete_quantiles():
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, quantiles=Q3, central='median')
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2})


def test_RED_three_level_facets_single_curve_discrete_quantiles():
    """Expression vectors with three facet levels are refused (PHASE_13_94);
    the three-level route is checked with one curve and with groups."""
    DFDraw(_frame()).draw('y1:x', type='profile', bins=6, quantiles=Q3,
                          facet_by=['a', 'b', 't'], facet_by_bins=[2, 2, 2])
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1})


def test_RED_three_level_facets_group_missing_from_cells():
    DFDraw(_missing_group_frame()).draw('yg:x', type='profile', bins=6, group_by='g',
                                        facet_by=['a', 'b', 't'], facet_by_bins=[2, 2, 2])
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})


@pytest.fixture
def _custom_palette():
    set_style({'colors.palette': 'Set1'})
    yield
    set_style(None)


def test_RED_custom_palette_order_and_children(_custom_palette):
    assert get_style_value('colors.palette') == 'Set1'
    DFDraw(_frame()).draw('[y1,y2,y3]:x', type='profile', bins=8, quantiles=Q3)
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2, 3})        # palette(k) from Set1


def test_RED_draw_batch_vector_discrete_quantiles():
    res = DFDraw(_frame()).draw_batch(
        {'p': {'expr': '[y1,y2]:x', 'type': 'profile', 'bins': 8, 'quantiles': Q3}},
        verbose=False, close_figures=False)
    assert not res.get('_errors'), res.get('_errors')
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2})


@pytest.mark.parametrize('facet', [1, 2])
def test_RED_hist_group_missing_from_cell_keeps_colours(facet):
    kw = (dict(facet_by='a', facet_by_bins=2) if facet == 1
          else dict(facet_by=['a', 'b'], facet_by_bins=[2, 2]))
    DFDraw(_missing_group_frame()).draw('yg', type='hist', bins=70, range=(0.0, 35.0),
                                        group_by='g', **kw)
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _hist_pairs(ax)], {1, 2, 3})


def test_RED_same_true_chain_profile_names_on_own_curves():
    d = DFDraw(_frame())
    d.draw('y1:x', type='profile', bins=8, quantiles=Q3)
    d.draw('y2:x', type='profile', bins=8, quantiles=Q3, same=True)
    fig = plt.gcf()
    curve = _check(_profile_pairs(fig.axes[0]), {1, 2})
    names = {t: c for t, c in _legend_entries(fig) if ' vs ' in t}
    assert names.get('y1 vs x') == curve[1] and names.get('y2 vs x') == curve[2], names


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def test_CTL_per_cell_group_intervals_keep_per_cell_colours():
    """13.108 R2: intervals computed per cell are not the same identity across
    cells, so their colours follow the cell's own interval order."""
    df = _frame()
    DFDraw(df).draw('yg:x', type='profile', bins=6, group_by='t', group_by_bins=2,
                    facet_by='a', facet_by_bins=2)
    fig = plt.gcf()
    for ax in _data_axes(fig):
        cols = sorted({c for _, c, w in _profile_pairs(ax) if w == 'line'})
        assert set(cols) <= {_palette(0), _palette(1)}, cols


def test_CTL_explicit_colour_draw_vector_profile_method():
    DFDraw(_frame()).profile('[y1,y2]:x', bins=8, quantiles=Q3, color='k')
    cols = {c for _, c, _ in _profile_pairs(plt.gcf().axes[0])}
    assert cols == {'#000000'}, cols


# --------------------------------------------------------------------------
# Negative controls: remove the S1 resolution, the rule must fail again
# --------------------------------------------------------------------------
def _without_s1(monkeypatch):
    real = getattr(DFDraw, '_resolve_identity_color', None)

    def no_first(self, same, ax):
        if not same:
            return None                    # the old behaviour: first curve anonymous
        return real(self, same, ax)
    monkeypatch.setattr(DFDraw, '_resolve_identity_color', no_first, raising=False)


def test_NEG_first_curve_without_resolution_fails_the_rule(monkeypatch):
    _without_s1(monkeypatch)
    DFDraw(_frame()).draw('[y1,y2,y3]:x', type='profile', bins=8, quantiles=Q3)
    with pytest.raises(AssertionError, match='curve 1: colours'):
        _check(_profile_pairs(plt.gcf().axes[0]), {1, 2, 3})


def test_NEG_cell_order_without_identity_list_fails_the_rule(monkeypatch):
    import dfextensions.dfdraw.plots.histogram as H
    real = H._draw_hist_grouped

    def drop_order(*a, **kw):
        kw.pop('_group_order', None)
        return real(*a, **kw)
    monkeypatch.setattr(H, '_draw_hist_grouped', drop_order)
    DFDraw(_missing_group_frame()).draw('yg', type='hist', bins=70, range=(0.0, 35.0),
                                        group_by='g', facet_by='a', facet_by_bins=2)
    fig = plt.gcf()
    with pytest.raises(AssertionError):
        _check([p for ax in _data_axes(fig) for p in _hist_pairs(ax)], {1, 2, 3})
