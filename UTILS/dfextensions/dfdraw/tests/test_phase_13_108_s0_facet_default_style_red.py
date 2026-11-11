"""PHASE_13_108_DF S0 - facet default style: RED tests and controls.

Plain-language background
-------------------------
PHASE_13_108 makes faceted profile figures readable by default:

* D108-A: one Figure X title and one Figure Y title instead of a title in
  every cell, decided separately per axis: an axis is shared only when every
  cell renders the same label on it (review R1).  The shared titles are the
  rendered cell labels, so AliasDataFrame axis titles are kept (R4).
* D108-B: one Figure legend instead of identical per-cell legends, built by
  the existing composer from the real drawn lines.  It only replaces legends
  that exist today (band mode draws none and still draws none, R3), and it
  never merges group legends whose intervals differ between cells (R2).
* D108-C: the same for the faceted ``hist2d+profile`` overlay, where
  ``legend="shared"`` and ``legend={"mode": "shared", "ncol": 2}`` are now
  accepted; colorbars stay one per cell.
* D108-D: quantile labels keep their precision: 0.101 is "q=10.1%", not
  "q=10%", so 0.1 and 0.101 are never merged.  Round levels keep their text.

Tests named ``test_RED_*`` describe the new behaviour and fail on the
PHASE_13_106 v0.2 code (commit 70c5f297).  Tests named ``test_CTL_*`` are
controls: they pass now and must still pass after the change (explicit
options, one-cell figures, the PHASE_13_94 route, non-profile facets,
facet-local groups, band mode, and unchanged statistics).

The AliasDataFrame title lookup is exercised through its duck-typed hook
(``_data_source.get_axis_title``), so no AliasDataFrame install is needed.

Seat: Fable5_1:draw-coder, 2026-11-10.
"""
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

Q_DISCRETE = [0.05, 0.101, 0.5, 0.9, 0.95]      # asymmetric -> one line per level
Q_BAND = [0.1, 0.5, 0.9]                        # symmetric -> band, no legend
LABELS_DISCRETE = ['q=5%', 'q=10.1%', 'median', 'q=90%', 'q=95%']
TITLES = {'y0': r'$dE/dx_{70}$', 'y1': r'$dE/dx_{50}$', 'x': r'$20/(dN/dx)$'}


class _TitleSource:
    """Duck-typed stand-in for AliasDataFrame axis titles."""

    def get_axis_title(self, name):
        return TITLES.get(name)


def _frame(n=6000, seed=10801):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.uniform(0.0, 1.0, n),
        'f': rng.integers(0, 3, n),
        'g': rng.integers(0, 2, n),
        'h': rng.integers(0, 2, n),
        'w': rng.uniform(0.0, 10.0, n),
    })
    df['y0'] = rng.normal(0.0, 1.0, n) + 0.2 * df.f
    df['y1'] = 1.0 + rng.normal(0.0, 1.0, n)
    return df


def _draw(expr, df=None, titles=True, **kw):
    d = DFDraw(_frame() if df is None else df)
    if titles:
        d._data_source = _TitleSource()
    return d.draw(expr, **kw)


def _state(fig, axes):
    cells = [a for a in np.asarray(axes).ravel() if a.get_visible()]
    sx = getattr(fig, '_supxlabel', None)
    sy = getattr(fig, '_supylabel', None)
    return {
        'n': len(cells),
        'cell_x': sorted({a.get_xlabel() for a in cells}),
        'cell_y': sorted({a.get_ylabel() for a in cells}),
        'sup_x': sx.get_text() if sx is not None else None,
        'sup_y': sy.get_text() if sy is not None else None,
        'cell_legends': [[t.get_text() for t in a.get_legend().get_texts()]
                         for a in cells if a.get_legend() is not None],
        'fig_legends': [[t.get_text() for t in L.get_texts()]
                        for L in fig.legends],
        'fig_legend_ncols': [getattr(L, '_ncols', None) for L in fig.legends],
    }


def _close(*figs):
    for f in figs:
        plt.close(f)


# ---------------------------------------------------------------------------
# D108-A: shared titles by default, per axis, from the rendered cell labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('q', [Q_DISCRETE, Q_BAND, None],
                         ids=['discrete', 'band', 'no-quantiles'])
def test_RED_A01_scalar_1d_shared_alias_titles(q):
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f', quantiles=q)
    try:
        s = _state(fig, axes)
        assert s['n'] == 3
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert s['sup_x'] == TITLES['x']
        assert s['sup_y'] == TITLES['y0']
    finally:
        _close(fig)


def test_RED_A02_scalar_2d_shared_alias_titles():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by=['f', 'g'],
                         facet_by_bins=[None, None], quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['n'] == 6
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert (s['sup_x'], s['sup_y']) == (TITLES['x'], TITLES['y0'])
    finally:
        _close(fig)


def test_RED_A03_vector_1d_shared_titles_are_the_rendered_labels():
    """The vector cell renders the Y label '[y0, y1]'; the shared title is
    that rendered text, not a Python list representation."""
    fig, axes, _ = _draw('[y0,y1]:x', type='profile', facet_by='f',
                         quantiles=[0.2, 0.5, 0.8])
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert s['sup_x'] == TITLES['x']
        assert s['sup_y'] == '[y0, y1]'
    finally:
        _close(fig)


def test_RED_A04_facet_by_vector_shares_x_only():
    """Review R1: each cell has a different Y expression, so the Y titles
    stay in the cells; the common X title is shared."""
    fig, axes, _ = _draw('[y0,y1]:x', type='profile', facet_by='vector',
                         quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == ['']
        assert s['sup_x'] == TITLES['x']
        assert s['cell_y'] == sorted([TITLES['y0'], TITLES['y1']])
        assert not s['sup_y']
    finally:
        _close(fig)


def test_RED_A05_overlay_shared_alias_titles():
    """Review F3/R4: the faceted overlay keeps the alias Y title (today an
    explicit shared_ylabel=True shows the raw expression 'y0')."""
    fig, axes, _ = _draw('y0:x', type='hist2d+profile', facet_by='f',
                         quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert (s['sup_x'], s['sup_y']) == (TITLES['x'], TITLES['y0'])
    finally:
        _close(fig)


def test_RED_A06_overlay_explicit_true_keeps_alias_y_title():
    fig, axes, _ = _draw('y0:x', type='hist2d+profile', facet_by='f',
                         quantiles=Q_DISCRETE, shared_xlabel=True,
                         shared_ylabel=True)
    try:
        assert _state(fig, axes)['sup_y'] == TITLES['y0']
    finally:
        _close(fig)


def test_RED_A07_grouped_profile_shares_titles():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f', group_by='h')
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert (s['sup_x'], s['sup_y']) == (TITLES['x'], TITLES['y0'])
    finally:
        _close(fig)


# ---------------------------------------------------------------------------
# D108-B / D108-C: one Figure legend replaces identical per-cell legends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['profile', 'hist2d+profile'])
def test_RED_B01_discrete_quantiles_one_figure_legend(kind):
    fig, axes, _ = _draw('y0:x', type=kind, facet_by='f', quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert len(s['fig_legends']) == 1
        assert s['fig_legends'][0] == LABELS_DISCRETE
    finally:
        _close(fig)


def test_RED_B02_scalar_2d_one_figure_legend():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by=['f', 'g'],
                         facet_by_bins=[None, None], quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert s['fig_legends'] == [LABELS_DISCRETE]
    finally:
        _close(fig)


def test_RED_B03_vector_1d_one_figure_legend_of_expressions():
    fig, axes, _ = _draw('[y0,y1]:x', type='profile', facet_by='f',
                         quantiles=[0.2, 0.5, 0.8])
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert len(s['fig_legends']) == 1
        assert s['fig_legends'][0][:2] == ['y0', 'y1']
    finally:
        _close(fig)


def test_RED_B04_facet_by_vector_one_quantile_legend():
    fig, axes, _ = _draw('[y0,y1]:x', type='profile', facet_by='vector',
                         quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert s['fig_legends'] == [LABELS_DISCRETE]
    finally:
        _close(fig)


def test_RED_B05_categorical_groups_one_figure_legend():
    """Categorical groups mean the same in every cell (global identities)."""
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f', group_by='h')
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert s['fig_legends'] == [['0', '1']]
    finally:
        _close(fig)


@pytest.mark.parametrize('legend,ncols', [
    ('shared', None),
    ({'mode': 'shared', 'ncol': 2}, 2),
], ids=['shared', 'dict-ncol2'])
def test_RED_C01_overlay_explicit_shared_legend_accepted(legend, ncols):
    fig, axes, _ = _draw('y0:x', type='hist2d+profile', facet_by='f',
                         quantiles=Q_DISCRETE, legend=legend)
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == []
        assert s['fig_legends'] == [LABELS_DISCRETE]
        if ncols is not None:
            assert s['fig_legend_ncols'] == [ncols]
        # one colorbar per cell stays (D108-C)
        assert len(fig.axes) == 2 * s['n']
    finally:
        _close(fig)


@pytest.mark.parametrize('legend,expected', [
    ('all', 3), ('none', 0), (False, 0),
], ids=['all', 'none', 'False'])
def test_RED_C02_overlay_explicit_per_cell_modes(legend, expected):
    fig, axes, _ = _draw('y0:x', type='hist2d+profile', facet_by='f',
                         quantiles=Q_DISCRETE, legend=legend)
    try:
        s = _state(fig, axes)
        assert len(s['cell_legends']) == expected
        assert s['fig_legends'] == []
    finally:
        _close(fig)


# ---------------------------------------------------------------------------
# D108-D: quantile label precision
# ---------------------------------------------------------------------------

def test_RED_D01_unfaceted_labels_keep_precision():
    fig, ax, _ = _draw('y0:x', type='profile', quantiles=[0.1, 0.101, 0.5])
    try:
        labels = ax.get_legend_handles_labels()[1]
        assert labels == ['q=10%', 'q=10.1%', 'median']
    finally:
        _close(fig)


def test_RED_D02_shared_legend_never_merges_close_levels():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f',
                         quantiles=[0.1, 0.101, 0.5], legend='shared')
    try:
        assert _state(fig, axes)['fig_legends'] == [['q=10%', 'q=10.1%',
                                                     'median']]
    finally:
        _close(fig)


def test_RED_D03_overlay_in_panel_tags_keep_precision():
    fig, ax, _ = _draw('y0:x', type='hist2d+profile', quantiles=Q_DISCRETE)
    try:
        assert [t.get_text() for t in ax.texts] == ['5%', '10.1%', '50%',
                                                   '90%', '95%']
    finally:
        _close(fig)


# ---------------------------------------------------------------------------
# Geometry: the default figure with a long expression is readable
# ---------------------------------------------------------------------------

LONG_Y = 'qTotExpR0_ResidualOverTaylorV3MADFitFullNormalised'
LONG_X = 'fFt0OccN'


def _long_frame():
    rng = np.random.default_rng(10802)
    n = 30000
    return pd.DataFrame({
        LONG_X: rng.uniform(0.0, 3.0, n),
        'fPidIndex': rng.choice([0, 2, 3, 4, 5, 6], n),
        LONG_Y: rng.normal(0.0, 1.0, n),
    })


def _element_problems(fig):
    """Overlaps between drawn elements of different owners, and elements
    outside the Figure.

    Each element is measured on its own (frame, title, axis labels, every
    drawn tick label, cell legend; Figure legends and texts).  Combined
    ``Axes.get_tightbbox`` boxes are not used: with Matplotlib 3.8 they do
    not count long axis labels, so they missed real overlaps (S1 review on
    Alma2).  Tick labels outside the axis view interval are not drawn and
    are skipped.
    """
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


@pytest.mark.parametrize('kind', ['profile', 'hist2d+profile'])
def test_RED_G01_long_expression_3x2_default_no_overlap(kind):
    fig, axes, _ = DFDraw(_long_frame()).draw(
        f'{LONG_Y}:{LONG_X}', type=kind, quantiles=Q_DISCRETE,
        facet_by='fPidIndex', ncols=3, auto_title=True)
    try:
        overlaps, outside = _element_problems(fig)
        assert overlaps == []
        assert outside == []
    finally:
        _close(fig)


# ---------------------------------------------------------------------------
# Controls: must pass before and after the change
# ---------------------------------------------------------------------------

def test_CTL_01_single_cell_keeps_cell_titles():
    df = _frame()
    fig, axes, _ = _draw('y0:x', df=df[df.f == 0], type='profile',
                         facet_by='f', quantiles=Q_DISCRETE)
    try:
        s = _state(fig, axes)
        assert s['n'] == 1
        assert s['cell_y'] == [TITLES['y0']]
        assert not s['sup_y']
    finally:
        _close(fig)


def test_CTL_02_explicit_false_keeps_cell_titles():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f',
                         quantiles=Q_DISCRETE, shared_xlabel=False,
                         shared_ylabel=False)
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == [TITLES['x']] and s['cell_y'] == [TITLES['y0']]
        assert not s['sup_x'] and not s['sup_y']
    finally:
        _close(fig)


def test_CTL_03_explicit_strings_win():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f',
                         quantiles=Q_DISCRETE, shared_xlabel='X text',
                         shared_ylabel='Y text')
    try:
        s = _state(fig, axes)
        assert (s['sup_x'], s['sup_y']) == ('X text', 'Y text')
    finally:
        _close(fig)


def test_CTL_04_explicit_all_keeps_per_cell_legends():
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f',
                         quantiles=Q_DISCRETE, legend='all')
    try:
        s = _state(fig, axes)
        assert len(s['cell_legends']) == 3
        assert s['fig_legends'] == []
    finally:
        _close(fig)


def test_CTL_05_band_mode_adds_no_legend():
    """Review R3: band mode draws no legend today and gets none."""
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f', quantiles=Q_BAND)
    try:
        s = _state(fig, axes)
        assert s['cell_legends'] == [] and s['fig_legends'] == []
    finally:
        _close(fig)


@pytest.mark.parametrize('kw', [
    dict(group_by='w', group_by_bins=3),
    dict(group_by='w', group_by_quantiles=3),
], ids=['bins', 'quantiles'])
def test_CTL_06_facet_local_group_intervals_keep_per_cell_legends(kw):
    """Review R2: group intervals resolved in each facet differ between
    cells; one Figure legend would be untruthful."""
    fig, axes, _ = _draw('y0:x', type='profile', facet_by='f', **kw)
    try:
        s = _state(fig, axes)
        assert len(s['cell_legends']) == 3
        assert s['fig_legends'] == []
    finally:
        _close(fig)


def test_CTL_07_phase_13_94_vector_2d_route_keeps_its_defaults():
    fig, axes, _ = _draw('[y0,y1]:x', type='profile', facet_by=['f', 'g'],
                         facet_by_bins=[None, None], quantiles=[0.2, 0.5, 0.8])
    try:
        s = _state(fig, axes)
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert s['sup_x'] == TITLES['x']
        assert len(s['fig_legends']) == 1 and s['cell_legends'] == []
    finally:
        _close(fig)


@pytest.mark.parametrize('kind,expr', [
    ('hist', 'y0'), ('hist2d', 'y0:x'), ('scatter', 'y0:x'),
])
def test_CTL_08_non_profile_facets_share_titles_without_default_legend(kind, expr):
    """Changed by PHASE_13_107_DF AD-6 (Architect 2026-11-11): the D108-A
    shared X/Y titles are the default for every faceted plot kind.  The
    default Figure legend (D108-B) is still not added for these kinds
    (before: they kept per-cell titles)."""
    fig, axes, _ = _draw(expr, type=kind, facet_by='f')
    try:
        s = _state(fig, axes)
        assert s['sup_x'] and s['sup_y']
        assert s['cell_x'] == [''] and s['cell_y'] == ['']
        assert s['fig_legends'] == []
    finally:
        _close(fig)


def _stats_equal(a, b, where=''):
    if isinstance(a, dict):
        assert sorted(a) == sorted(b), where
        for k in a:
            _stats_equal(a[k], b[k], f'{where}/{k}')
    elif isinstance(a, (list, tuple)) and a and isinstance(a[0], dict):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _stats_equal(x, y, f'{where}[{i}]')
    elif isinstance(a, (np.ndarray, list, tuple)):
        np.testing.assert_array_equal(np.asarray(a, dtype=object),
                                      np.asarray(b, dtype=object),
                                      err_msg=where)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), where
    else:
        assert a == b, where


@pytest.mark.parametrize('kind', ['profile', 'hist2d+profile'])
def test_CTL_09_presentation_never_changes_statistics(kind):
    """Default presentation versus explicit per-cell presentation: every
    per-cell statistic is identical (presentation only)."""
    explicit = dict(shared_xlabel=False, shared_ylabel=False)
    if kind == 'profile':
        explicit['legend'] = 'all'
    f1, _, s1 = _draw('y0:x', type=kind, facet_by='f', quantiles=Q_DISCRETE)
    f2, _, s2 = _draw('y0:x', type=kind, facet_by='f', quantiles=Q_DISCRETE,
                      **explicit)
    try:
        _stats_equal(s1, s2)
    finally:
        _close(f1, f2)
