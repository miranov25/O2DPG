"""PHASE_13_99_DF RED-first: 1-D/2-D facet semantic presentation parity.

Only Figure presentation is tested here.  Numerical invariance and gallery
acceptance remain separate commissioning gates.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw


def _frame():
    rng = np.random.default_rng(1399)
    n = 1800
    f = pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'f': np.repeat(np.arange(6), n // 6).astype(float),
        'g': np.tile(np.repeat([0, 1], n // 12), 6).astype(float),
        'y0': rng.normal(size=n),
        'y1': 2 + rng.normal(size=n),
    })
    return f


def _draw(**kwargs):
    params = dict(type='profile', bins=5, quantiles=[.15, .5, .9],
                  facet_by='f', facet_by_bins=6, ncols=3,
                  legend='shared', auto_title=True)
    params.update(kwargs)
    return DFDraw(_frame()).draw('[y0,y1]:x', **params)


def _legend_labels(fig):
    return [t.get_text() for leg in fig.legends for t in leg.get_texts()]


def test_FP01_FP02_FP03_six_real_panels_and_factored_shared_legend():
    fig, axes, stats = _draw()
    try:
        assert len(np.asarray(axes).flat) == 6
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in axes)
        labels = _legend_labels(fig)
        assert labels.count('q=15%') == 1
        assert labels.count('median') == 1
        assert labels.count('q=90%') == 1
        assert any('y0' in text for text in labels)
        assert any('y1' in text for text in labels)
        assert len(stats) == 2 and all(s['n_groups'] == 6 for s in stats)
    finally:
        plt.close(fig)


def test_FP04_FP05_scalar_one_dimensional_accepts_common_labels_outer_heads():
    fig, axes, stats = _draw(shared_xlabel='FT0 occupancy',
                             shared_ylabel='Residual', facet_labels='outer',
                             figure_margins=[.14,.97,.80,.12],
                             subplot_spacing=[.22,.28])
    try:
        assert axes.shape == (6,)
        assert fig._supxlabel.get_text() == 'FT0 occupancy'
        assert fig._supylabel.get_text() == 'Residual'
        assert all(not ax.get_xlabel() and not ax.get_ylabel() for ax in axes)
        assert all(not ax.get_title() for ax in axes)
        assert len(stats) == 2
        assert len(fig.legends) == 1
    finally:
        plt.close(fig)


def test_FP06_shared_legend_explicit_lower_position_still_works():
    fig, axes, _ = _draw(legend={'mode':'shared','loc':'lower center','ncol':2})
    try:
        assert len(fig.legends) == 1
        assert fig.legends[0]._loc == 8  # Matplotlib lower center
        assert fig.legends[0]._ncols == 2
    finally:
        plt.close(fig)


def test_FP08_FP15_two_dimensional_legend_contains_quantile_semantics():
    fig, axes, _ = _draw(facet_by=['f','g'], facet_by_bins=[3,2])
    try:
        labels = _legend_labels(fig)
        assert len(fig.legends) == 1
        for semantic in ('y0','y1','q=15%','median','q=90%'):
            assert labels.count(semantic) == 1
    finally:
        plt.close(fig)


def test_FP14_automatic_title_mentions_both_vector_components():
    fig, _, _ = _draw()
    try:
        assert 'y0' in fig._suptitle.get_text()
        assert 'y1' in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_FP12_1d_facet_label_format_auto_still_refuses():
    with pytest.raises(ValueError, match='two facet dimensions'):
        _draw(facet_label_format='auto')


def test_FP13_1d_legend_does_not_intersect_axes_or_panel_headings():
    fig, axes, _ = _draw(
        shared_xlabel='FT0 occupancy', shared_ylabel='Residual',
        facet_labels='outer', figure_margins=[.16,.96,.80,.12],
        subplot_spacing=[.22,.28])
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = fig.legends[0].get_window_extent(renderer)
        assert all(not legend_bbox.overlaps(ax.get_tightbbox(renderer))
                   for ax in axes if ax.get_visible())
        panel_titles = [text for text in fig.texts if text is not fig._suptitle
                        and text is not fig._supxlabel and text is not fig._supylabel]
        assert len(panel_titles) == 6
        assert all(not legend_bbox.overlaps(t.get_window_extent(renderer))
                   for t in panel_titles)
        assert 'faceted by f' in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_FP06_legacy_legend_modes_respected():
    for legend_mode in ('none','first','all'):
        fig, axes, _ = _draw(legend=legend_mode, auto_title=False)
        try:
            if legend_mode == 'none':
                assert not fig.legends
                assert all(ax.get_legend() is None for ax in axes)
            elif legend_mode == 'first':
                assert sum(ax.get_legend() is not None for ax in axes) <= 1
            elif legend_mode == 'all':
                assert not fig.legends
        finally:
            plt.close(fig)


def test_FP07_one_element_list_vector_facet_refusal_not_widened():
    # PHASE_13_94 explicitly refuses list-form singleton expression-vector
    # facets before entering the Figure coordinator. Preserve that boundary.
    with pytest.raises(NotImplementedError, match='exactly two facet columns'):
        _draw(facet_by=['f'], facet_by_bins=[6],
              shared_xlabel='x', facet_labels='outer')


def test_FP16_legacy_shared_group_legend_with_sparse_last_facet_row():
    """A hidden unused grid cell has no tight bbox; layout must skip it.

    Preserve the PHASE_13_50 shared-legend contract when a 4-bin facet
    occupies a 2x3 grid (the last row has two hidden Axes).
    """
    rng = np.random.default_rng(139916)
    n = 400
    df = pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'y': rng.normal(size=n),
        'sec': np.repeat(np.arange(4), n // 4),
        'g': np.tile([0, 1], n // 2),
    })
    fig, axes, _ = DFDraw(df).profile(
        'y:x', bins=8, group_by='g', facet_by='sec',
        ncols=3, legend='shared',
    )
    try:
        # The public 1-D API returns only the four physical facet axes;
        # Figure.axes includes the two hidden GridSpec placeholder cells.
        assert len(axes) == 4
        assert len(fig.axes) == 6
        assert sum(ax.get_visible() for ax in fig.axes) == 4
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert all(ax.get_tightbbox(renderer) is None for ax in fig.axes
                   if not ax.get_visible())
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in fig.axes)
    finally:
        plt.close(fig)
