"""PHASE_13_108_DF S2 - every default facet layout fits inside its Figure.

Plain-language background
-------------------------
PHASE_13_108 made one shared X/Y title and one Figure legend the default for
faceted profiles and the faceted hist2d+profile overlay.  On real data
(Alma2, Matplotlib 3.8.2) two defects of the shared-presentation code then
showed up on ordinary calls:

* with one row of panels, the shared Y title (a long expression written
  vertically) was taller than the Figure and stuck out of it;
* the top tick labels of the overlay colorbars touched the Figure legend,
  because the legend placement made room above the panels only.

Both are the same category: a Figure-level text that dfdraw places itself
must fit inside the Figure and must not touch any drawn element.  The fix
covers the whole category, not only the two reported cases:

* automatic shared X/Y titles and the automatic suptitle are fitted to their
  span (panel height, panel width, Figure width): wrapped at an operator
  boundary first (every character kept), then reduced in size to a minimum
  of 7 pt; explicit caller strings are never changed, the existing
  PHASE_13_93 diagnostic warns instead;
* colorbars and other non-panel axes count as the inner boundary for the
  legend, headings and titles.

The geometry is measured element by element (frames, titles, axis labels,
drawn tick labels, legends, Figure texts), because the combined
``Axes.get_tightbbox`` boxes miss long labels on Matplotlib 3.8.

Seat: Fable5_1:draw-coder, 2026-11-10.
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

Q = [0.05, 0.101, 0.5, 0.9, 0.95]
NUM = 'qTotExpR0'
FIT = 'qTotExpR0_TaylorV3FitFull'
MAD = 'qTotExpR0_TaylorV3MADFitFull'
EXPR = f'({NUM}-{FIT})/{MAD}:fFt0OccN'
ROOM_WARNING = 'PHASE_13_93: Figure margins leave insufficient room'


def _frame():
    rng = np.random.default_rng(10821)
    n = 30000
    df = pd.DataFrame({
        'fFt0OccN': rng.uniform(0.0, 4.0, n),
        'fPidIndex': rng.choice([0, 2, 3, 4, 5, 6], n),
        'rate': rng.uniform(0.0, 10.0, n),
    })
    df[NUM] = rng.normal(1.0, 0.1, n)
    df[FIT] = 1.0
    df[MAD] = 0.1
    return df


def element_problems(fig):
    """Overlaps between elements of different owners; elements outside."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    elements = []
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        owner = f'axes{i}'
        elements.append((owner, 'frame', ax.get_window_extent(r)))
        for name, text in (('title', ax.title), ('xlabel', ax.xaxis.label),
                           ('ylabel', ax.yaxis.label)):
            if text.get_visible() and text.get_text():
                elements.append((owner, name, text.get_window_extent(r)))
        for name, axis in (('xtick', ax.xaxis), ('ytick', ax.yaxis)):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * (hi - lo)
            for tick in axis.get_major_ticks():
                label = tick.label1
                if (tick.get_visible() and label.get_visible()
                        and label.get_text()
                        and lo - eps <= tick.get_loc() <= hi + eps):
                    elements.append((owner, f'{name}:{label.get_text()}',
                                     label.get_window_extent(r)))
        if ax.get_legend() is not None:
            elements.append((owner, 'legend',
                             ax.get_legend().get_window_extent(r)))
    for i, leg in enumerate(fig.legends):
        elements.append(('figure', f'legend{i}', leg.get_window_extent(r)))
    for i, text in enumerate(fig.texts):
        if text.get_visible() and text.get_text():
            elements.append(('figure', f'text{i}', text.get_window_extent(r)))
    width, height = fig.bbox.width, fig.bbox.height
    overlaps = sorted({(f'{o1}.{n1}', f'{o2}.{n2}')
                       for (o1, n1, b1), (o2, n2, b2)
                       in itertools.combinations(elements, 2)
                       if o1 != o2 and b1.overlaps(b2)})
    outside = sorted({f'{o}.{n}' for o, n, b in elements
                      if b.x0 < -0.5 or b.y0 < -0.5
                      or b.x1 > width + 0.5 or b.y1 > height + 0.5})
    return overlaps, outside


LAYOUTS = {
    'overlay-1x2': ('hist2d+profile', [0, 2], dict(facet_by='fPidIndex', ncols=2)),
    'overlay-1x3': ('hist2d+profile', [0, 2, 3], dict(facet_by='fPidIndex', ncols=3)),
    'overlay-2x2': ('hist2d+profile', [0, 2, 3, 4], dict(facet_by='fPidIndex', ncols=2)),
    'overlay-2x3': ('hist2d+profile', None, dict(facet_by='fPidIndex', ncols=3)),
    'profile-1x3': ('profile', [0, 2, 3], dict(facet_by='fPidIndex', ncols=3)),
    'profile-2x3': ('profile', None, dict(facet_by='fPidIndex', ncols=3)),
    'profile-2d-3x3': ('profile', [0, 2, 3],
                       dict(facet_by=['fPidIndex', 'rate'], facet_by_bins=[None, 3])),
    'profile-2d-6x2': ('profile', None,
                       dict(facet_by=['fPidIndex', 'rate'], facet_by_bins=[None, 2])),
}


def _draw(layout, **extra):
    kind, pids, kw = LAYOUTS[layout]
    df = _frame()
    if pids is not None:
        df = df[df.fPidIndex.isin(pids)]
    kw = dict(kw, **extra)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, axes, stats = DFDraw(df).draw(EXPR, type=kind, quantiles=Q,
                                           auto_title=True, **kw)
    room = [w for w in caught if ROOM_WARNING in str(w.message)]
    return fig, axes, stats, room


@pytest.mark.parametrize('layout', sorted(LAYOUTS))
def test_S2_01_default_layout_fits_and_nothing_overlaps(layout):
    fig, axes, _, room = _draw(layout)
    try:
        overlaps, outside = element_problems(fig)
        assert overlaps == []
        assert outside == []
        assert room == []
    finally:
        plt.close(fig)


@pytest.mark.parametrize('layout', ['overlay-1x2', 'overlay-1x3',
                                    'overlay-2x2', 'overlay-2x3'])
def test_S2_02_figure_legend_above_every_colorbar(layout):
    """The legend clears the colorbars' tick labels, not only the panels."""
    fig, axes, _, _ = _draw(layout)
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        panels = set(np.asarray(axes).ravel())
        colorbars = [a for a in fig.axes if a.get_visible() and a not in panels]
        assert len(colorbars) == len([a for a in panels if a.get_visible()])
        legend_bottom = fig.legends[0].get_window_extent(r).y0
        assert legend_bottom >= max(c.get_tightbbox(r).y1 for c in colorbars)
    finally:
        plt.close(fig)


def test_S2_03_automatic_y_title_keeps_every_character():
    """One row: the automatic Y title is wrapped/shrunk, not abbreviated."""
    fig, axes, _, _ = _draw('overlay-1x3')
    try:
        text = fig._supylabel.get_text()
        assert text.replace('\n', '') == f'<({NUM}-{FIT})/{MAD}>'
    finally:
        plt.close(fig)


def test_S2_04_explicit_title_is_never_changed_but_warned():
    long_label = 'explicit ' + 'very long Y title ' * 12
    fig, axes, _, room = _draw('overlay-1x3', shared_ylabel=long_label)
    try:
        reference = plt.figure()
        default_size = reference.supylabel('x').get_fontsize()
        plt.close(reference)
        assert fig._supylabel.get_text() == long_label
        assert fig._supylabel.get_fontsize() == default_size
        assert room != []
    finally:
        plt.close(fig)


@pytest.mark.parametrize('line,parts', [
    (f'<({NUM}-{FIT})/{MAD}>', 2),
    (f'<({NUM}-{FIT})/{MAD}>', 3),
    ('a vs b  [faceted by x × y]', 2),
], ids=['expression-2', 'expression-3', 'title-2'])
def test_S2_05_wrap_keeps_text_and_breaks_at_operators(line, parts):
    from dfextensions.dfdraw.drawer import _wrap_text_line
    wrapped = _wrap_text_line(line, parts)
    assert wrapped is not None
    pieces = wrapped.split('\n')
    assert len(pieces) == parts
    # every character is kept (only the spaces at a break may move)
    assert ''.join(pieces).replace(' ', '') == line.replace(' ', '')
    # every break follows an operator or a space of the original text
    position = 0
    for piece in pieces[:-1]:
        position = line.index(piece.strip(), position) + len(piece.strip())
        assert line[position - 1] in '/*+-,' or line[position] == ' '


def test_S2_06_mathtext_is_never_wrapped():
    from dfextensions.dfdraw.drawer import _fit_text_to_extent
    fig = plt.figure(figsize=(3, 2))
    try:
        text = fig.supylabel(r'$(q_{\mathrm{Tot}}-q_{\mathrm{fit}})/'
                             r'\mathrm{MAD}_{\mathrm{fit}}$' * 3)
        fig.canvas.draw()
        original = text.get_text()
        _fit_text_to_extent(text, fig.canvas.get_renderer(), 20.0, 'y')
        assert text.get_text() == original
        assert text.get_fontsize() == 7.0
    finally:
        plt.close(fig)
