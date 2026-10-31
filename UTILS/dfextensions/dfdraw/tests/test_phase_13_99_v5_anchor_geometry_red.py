"""PHASE_13_99 v5: user placement authority and outer-heading page bounds.

These tests are deliberately RED on the independently reviewed v4 bytes.
They do not inspect or modify numerical estimators.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def _data():
    rng = np.random.default_rng(139905)
    n = 1800
    return pd.DataFrame({
        'x': rng.uniform(.88, 1.02, n),
        'fTgl': np.linspace(-1.178, 1.175, n),
        'fFt0Occ': rng.uniform(-233., 231581., n),
        'group': np.tile(np.repeat(['A', 'B', 'C'], 100), 6),
        'y0': .92 + .07 * rng.normal(size=n),
        'y1': 1.10 + .09 * rng.normal(size=n),
    })


def _draw(two_dimensional, legend):
    opts = dict(
        type='profile', bins=6,
        facet_by=['fTgl', 'fFt0Occ'] if two_dimensional else 'fTgl',
        facet_by_bins=[3, 2] if two_dimensional else 6,
        group_by='group', group_by_scope='global',
        quantiles=[.15, .5, .9], quantile_mode='discrete',
        ncols=3, legend=legend, auto_title=False,
        shared_xlabel='x', shared_ylabel='response', facet_labels='outer',
        figure_margins=[.16, .96, .72, .13],
        subplot_spacing=[.22, .3],
    )
    return DFDraw(_data()).draw('[y0,y1]:x', **opts)


@pytest.mark.parametrize('two_dimensional', [False, True], ids=['one_dim','two_dim'])
@pytest.mark.parametrize('with_loc', [False, True], ids=['anchor_only','loc_and_anchor'])
def test_FP06_user_bbox_anchor_preserved_after_figure_positioning(two_dimensional, with_loc):
    spec = {'mode': 'shared', 'bbox_to_anchor': (.30, .20),
            'ncol': 3, 'title': 'Explicit position'}
    if with_loc:
        spec['loc'] = 'lower center'
    fig, axes, stats = _draw(two_dimensional, spec)
    try:
        fig.canvas.draw()
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        np.testing.assert_allclose(anchor.bounds[:2], (.30, .20), atol=1e-8, rtol=0)
        assert legend.get_title().get_text() == 'Explicit position'
        assert legend._ncols == 3
        if with_loc:
            assert legend._loc == 8
    finally:
        plt.close(fig)


def test_FP13_2d_real_range_row_headings_not_clipped_by_figure_edge():
    fig, axes, _ = _draw(True, 'shared')
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Row headings are distinct from the supertitle and shared labels.
        headings = [t for t in fig.texts if t is not fig._suptitle and
                    t is not fig._supxlabel and t is not fig._supylabel and
                    ',' in t.get_text() and t.get_position()[0] < .2]
        assert len(headings) == 3, [(t.get_text(), t.get_position()) for t in fig.texts]
        left_edges = [t.get_window_extent(renderer).x0 for t in headings]
        assert min(left_edges) >= 0, left_edges
        assert all('(' in t.get_text() and ']' in t.get_text() for t in headings)
    finally:
        plt.close(fig)

@pytest.mark.parametrize('two_dimensional', [False, True], ids=['one_dim', 'two_dim'])
def test_FP10_requested_figsize_is_effective_before_facet_artist_render(two_dimensional):
    sizes = [(11., 7.), (15., 10.)]
    returned = []
    for requested in sizes:
        opts = dict(type='profile', bins=6,
                    facet_by=['fTgl', 'fFt0Occ'] if two_dimensional else 'fTgl',
                    facet_by_bins=[3, 2] if two_dimensional else 6,
                    ncols=3, legend='shared', quantiles=[.15,.5,.9],
                    quantile_mode='discrete', auto_title=False,
                    shared_xlabel='x', shared_ylabel='response',
                    facet_labels='outer', figsize=requested)
        fig, axes, _ = DFDraw(_data()).draw('[y0,y1]:x', **opts)
        try:
            np.testing.assert_allclose(fig.get_size_inches(), requested, atol=1e-12)
            assert len(fig.axes) == 6
            returned.append(tuple(fig.get_size_inches()))
        finally:
            plt.close(fig)
    assert returned[0] != returned[1]


def test_FP10_invalid_requested_figsize_rejected_before_allocation():
    with pytest.raises(ValueError, match='figsize'):
        DFDraw(_data()).draw('[y0,y1]:x', type='profile', facet_by='fTgl',
                             facet_by_bins=6, figsize=(0, 4))
