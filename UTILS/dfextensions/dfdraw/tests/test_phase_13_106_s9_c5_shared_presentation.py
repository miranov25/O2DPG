"""PHASE_13_106_DF S9 - C5 shared presentation: measured geometry.

Plain-language background
-------------------------
C5 is the 2x2 shared presentation of a faceted expression-vector profile:
one shared legend, one shared X and one shared Y label, panel headings on
the outer panels, explicit figure margins and subplot spacing, and the
automatic title.  It needed no code change in PHASE_13_106 (the options
exist since PHASE_13_93/13_99/13_103), so the change review relied on a
real-data PDF.  The PRECOMMIT review (consolidated summary GPT29,
2026-11-09, item P1-4) asked for measured geometry instead of appearance.

This module pins Marian's real C5 call shape on synthetic data with the same
column names, and measures the drawn boxes with the Matplotlib renderer:

* exactly one legend (the four components), one shared X label, one shared
  Y label, four panel headings, a title naming the facet column and the
  quantiles;
* no two of these boxes overlap, and no panel box (axes plus tick labels)
  overlaps another element;
* every box lies inside the Figure;
* no PHASE_13_93 "insufficient room" warning.

Two sensitivity controls show that the measurement detects problems: margins
that are too small push elements outside the Figure (and dfdraw warns), and
zero subplot spacing makes panels overlap.

Seat: Fable5_1:draw-coder, 2026-11-09.
"""
import itertools
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

COMPONENTS = ['ratioT50', 'ratioT60', 'ratioT70', 'ratioT100']
EXPR = '[' + ','.join(COMPONENTS) + ']:invDNdx20'
C5 = dict(type='profile', quantiles=[0.2, 0.5, 0.8], bins=10, range=[0.05, 1],
          facet_by='nPoints', facet_by_bins=4, ncols=2, legend='shared',
          shared_xlabel=r'$20/(dN/dx)$', shared_ylabel=r'$dEdx_{trunc}/dNdx$',
          facet_labels='outer', figure_margins=[0.10, 0.98, 0.86, 0.10],
          subplot_spacing=[0.18, 0.22], auto_title=True)
ROOM_WARNING = 'PHASE_13_93: Figure margins leave insufficient room'


def _frame():
    rng = np.random.default_rng(10609)
    n = 20000
    x = rng.uniform(0.05, 1.0, n)
    df = pd.DataFrame({'invDNdx20': x, 'nPoints': rng.uniform(40, 160, n)})
    for i, name in enumerate(COMPONENTS):
        df[name] = 1.0 + 0.1 * i + 0.3 * x + rng.normal(0.0, 0.05, n)
    return df


def figure_boxes(fig, axes):
    """Renderer boxes (display pixels) of every Figure-level element."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {}
    if fig._suptitle is not None and fig._suptitle.get_text():
        boxes['title'] = fig._suptitle.get_window_extent(renderer)
    for i, leg in enumerate(fig.legends):
        boxes[f'legend{i}'] = leg.get_window_extent(renderer)
    for name in ('_supxlabel', '_supylabel'):
        text = getattr(fig, name, None)
        if text is not None and text.get_text():
            boxes[name.strip('_')] = text.get_window_extent(renderer)
    special = {fig._suptitle, getattr(fig, '_supxlabel', None),
               getattr(fig, '_supylabel', None)}
    for text in fig.texts:
        if text not in special and text.get_visible() and text.get_text():
            boxes[f'heading[{text.get_text()}]'] = text.get_window_extent(renderer)
    for i, ax in enumerate(np.asarray(axes).ravel()):
        if ax.get_visible():
            boxes[f'panel{i}'] = ax.get_tightbbox(renderer)
    return boxes


def geometry_problems(fig, axes):
    boxes = figure_boxes(fig, axes)
    width, height = fig.bbox.width, fig.bbox.height
    overlaps = [(a, b) for a, b in itertools.combinations(boxes, 2)
                if boxes[a].overlaps(boxes[b])]
    outside = [k for k, b in boxes.items()
               if b.x0 < -0.5 or b.y0 < -0.5
               or b.x1 > width + 0.5 or b.y1 > height + 0.5]
    return boxes, overlaps, outside


def _draw(**over):
    params = dict(C5)
    params.update(over)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, axes, stats = DFDraw(_frame()).draw(EXPR, **params)
    room = [w for w in caught if ROOM_WARNING in str(w.message)]
    return fig, axes, stats, room


def test_S9_01_c5_elements_present_once():
    fig, axes, _, _ = _draw()
    try:
        assert np.asarray(axes).size == 4
        assert len(fig.legends) == 1
        assert [t.get_text() for t in fig.legends[0].get_texts()] == COMPONENTS
        assert fig._supxlabel.get_text() == C5['shared_xlabel']
        assert fig._supylabel.get_text() == C5['shared_ylabel']
        assert all(ax.get_xlabel() == '' and ax.get_ylabel() == ''
                   for ax in np.asarray(axes).ravel())
        boxes = figure_boxes(fig, axes)
        assert len([k for k in boxes if k.startswith('heading[')]) == 4
        title = fig._suptitle.get_text()
        assert 'faceted by nPoints' in title
        assert '0.2' in title and '0.8' in title
    finally:
        plt.close(fig)


def test_S9_02_c5_no_overlap_nothing_outside_no_warning():
    fig, axes, _, room = _draw()
    try:
        boxes, overlaps, outside = geometry_problems(fig, axes)
        assert overlaps == []
        assert outside == []
        assert room == []
        assert len(boxes) == 1 + 1 + 2 + 4 + 4   # title, legend, labels, headings, panels
    finally:
        plt.close(fig)


@pytest.mark.parametrize('margins', [
    [0.10, 0.98, 0.97, 0.10],   # no room above the panels
    [0.02, 0.98, 0.86, 0.10],   # no room for the shared Y label
    [0.10, 0.98, 0.86, 0.01],   # no room for the shared X label
], ids=['top', 'left', 'bottom'])
def test_S9_03_control_too_small_margins_detected(margins):
    """Sensitivity: the measurement finds elements outside the Figure, and
    dfdraw's own PHASE_13_93 check warns."""
    fig, axes, _, room = _draw(figure_margins=margins)
    try:
        _, _, outside = geometry_problems(fig, axes)
        assert outside != []
        assert room != []
    finally:
        plt.close(fig)


def test_S9_04_control_zero_spacing_overlap_detected():
    """Sensitivity: with zero subplot spacing the panels overlap and the
    measurement reports it.  (dfdraw does not warn here: explicit spacing is
    the caller's choice and is applied unchanged.)"""
    fig, axes, _, _ = _draw(subplot_spacing=[0.0, 0.0])
    try:
        _, overlaps, _ = geometry_problems(fig, axes)
        assert ('panel0', 'panel1') in overlaps
    finally:
        plt.close(fig)


def test_S9_05_adf_public_path_same_geometry():
    """Native ADF integration gate: skips only where AliasDataFrame is not
    installed; the Alma2 canonical run must execute it."""
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    df = _frame()
    direct = DFDraw(df).draw(EXPR, **C5)
    result = adf_module.AliasDataFrame(df).draw(EXPR, **C5)
    try:
        b_d, ov_d, out_d = geometry_problems(direct[0], direct[1])
        b_a, ov_a, out_a = geometry_problems(result[0], result[1])
        assert ov_a == [] and out_a == []
        assert sorted(b_a) == sorted(b_d)
        for key in b_d:
            np.testing.assert_allclose(b_a[key].extents, b_d[key].extents,
                                       rtol=0, atol=0.5, err_msg=key)
    finally:
        plt.close(direct[0])
        plt.close(result[0])
