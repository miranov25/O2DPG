"""PHASE_13_93_DF Stage A handover: neighboring and public-door invariants.

These supplement the original 14 RED/GREEN tests without altering those tests.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dfextensions.dfdraw import DFDraw


def _table(fig_levels=('Figure-B', 'Figure-A')):
    data = []
    for fidx, fig in enumerate(fig_levels):
        for row_index, row in enumerate(('Rate', 'Base')):
            for col_index, col in enumerate(('Tgl', 'Constant')):
                for k in range(3):
                    data.append((fig, row, col, float(k),
                                 0.1 + .02 * row_index + .01 * col_index + .001 * k,
                                 .002 + .0001 * k))
    df = pd.DataFrame(data, columns=['fig', 'row', 'col', 'region', 'coefficient', 'error'])
    for key, cats in [('fig', fig_levels), ('row', ('Rate', 'Base')),
                      ('col', ('Tgl', 'Constant'))]:
        df[key] = pd.Categorical(df[key], categories=cats, ordered=True)
    return df


def test_A_H1_public_draw_retains_unscaled_scatter_errorbar_values():
    df = _table().query("fig == 'Figure-B'")
    result = DFDraw(df).draw(
        'coefficient:region', type='scatter', yerr='error',
        facet_by=['row', 'col'], facet_labels='outer',
        shared_xlabel='Region', shared_ylabel='Coefficient',
        figure_margins=[.12, .91, .88, .14],
        subplot_spacing=[.35, .31], share_y='none')
    fig, axes, stats = result
    try:
        assert axes.shape == (2, 2)
        assert all(key in stats for key in [('Rate', 'Tgl'), ('Rate', 'Constant'),
                                           ('Base', 'Tgl'), ('Base', 'Constant')])
        # The point coordinates must be the original fitted quantities,
        # without the presentation-only 100* factor previously proposed.
        for row_i, row in enumerate(('Rate', 'Base')):
            for col_i, col in enumerate(('Tgl', 'Constant')):
                expected = df[(df['row'] == row) & (df['col'] == col)]
                # Legacy fixed-color errorbar mode uses Line2D markers;
                # PHASE_13_92's mapped-color case uses PathCollection points.
                # This fixture intentionally exercises the legacy owner.
                ax = axes[row_i, col_i]
                np.testing.assert_allclose(ax.lines[0].get_xdata(), expected.region)
                np.testing.assert_allclose(ax.lines[0].get_ydata(), expected.coefficient)
                segments = ax.collections[0].get_segments()
                assert len(segments) == len(expected)
                for segment, x, y, error in zip(segments, expected.region,
                                                  expected.coefficient, expected.error):
                    np.testing.assert_allclose(segment[:, 0], [x, x])
                    np.testing.assert_allclose(segment[:, 1], [y-error, y+error])
        assert fig.subplotpars.left == .12
        assert fig.subplotpars.wspace == .35
    finally:
        plt.close(fig)


@pytest.mark.parametrize('plot_kind,expression,extra', [
    ('scatter', 'coefficient:region', {}),
    ('hist', 'coefficient', {'bins': 5}),
    ('profile', 'coefficient:region', {'bins': 3}),
    ('hist2d', 'coefficient:region', {'bins': 5}),
])
def test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner(monkeypatch, plot_kind,
                                                                 expression, extra):
    df = _table().query("fig == 'Figure-B'")
    called = []
    original = Figure.tight_layout
    def count(self, *args, **kwargs):
        called.append(id(self))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Figure, 'tight_layout', count)
    fig, axes, stats = DFDraw(df).draw(
        expression, type=plot_kind, facet_by=['row', 'col'],
        share_y='none', **extra)
    try:
        assert len(called) == 1
        assert axes.shape == (2, 2)
    finally:
        plt.close(fig)


def test_A_H3_3D_categorical_figure_order_and_outer_headings():
    df = _table(fig_levels=('Figure-B', 'Figure-A'))
    figures, axes_list, stats = DFDraw(df).scatter(
        'coefficient:region', facet_by=['row', 'col', 'fig'],
        facet_labels='outer', shared_ylabel='Coefficient', share_y='none')
    try:
        assert [entry['figid_value'] for entry in stats] == ['Figure-B', 'Figure-A']
        assert len(figures) == 2
        for fig, axes in zip(figures, axes_list):
            assert axes.shape == (2, 2)
            texts = [t.get_text() for t in fig.texts]
            assert texts.count('Rate') == 1
            assert texts.count('Base') == 1
            assert texts.count('Tgl') == 1
            assert texts.count('Constant') == 1
    finally:
        for fig in figures:plt.close(fig)


@pytest.mark.parametrize('bad', [
    {'row': ['Rate', 'Rate']},
    {'row': ['Rate', 'unknown']},
    {'row': ['Rate']},
    {'not_a_facet': ['Rate', 'Base']},
])
def test_A_H4_invalid_facet_order_never_degrades_silently(bad):
    with pytest.raises(ValueError, match='facet_order'):
        DFDraw(_table()).scatter('coefficient:region',
                                 facet_by=['row', 'col'], facet_order=bad)


def test_A_H5_scalar_facet_rejects_ND_only_presentation_options():
    with pytest.raises(ValueError, match='require'):
        DFDraw(_table()).scatter('coefficient:region', facet_by='row',
                                 facet_labels='outer')


def test_A_H6_shared_ylabel_without_outer_mode_keeps_row_identity():
    df = _table().query("fig == 'Figure-B'")
    fig, axes, stats = DFDraw(df).scatter('coefficient:region',
        facet_by=['row', 'col'], shared_ylabel=True, share_y='none')
    try:
        texts = [t.get_text() for t in fig.texts]
        assert 'row=Rate' in texts and 'row=Base' in texts
        assert 'coefficient' in texts
        assert all(ax.get_ylabel() == '' for ax in axes.flat)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('keyword,value', [
    ('figure_margins', [.6, .4, .9, .1]),
    ('figure_margins', [.1, .9, .2, .3]),
    ('subplot_spacing', [-.2, .2]),
])
def test_A_H7_invalid_layout_parameters_raise_before_figure_allocation(keyword,value):
    options = {keyword:value}
    with pytest.raises(ValueError, match=keyword):
        DFDraw(_table()).scatter('coefficient:region',
                                 facet_by=['row', 'col'], **options)
