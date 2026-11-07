"""PHASE_13_103_DF B3: bounded renderer-measured shared-legend acceptance.

The RED baseline is B2b commit d9f6d157: its factored legend has correct
semantic content but does not wrap long/large identity sets to Figure width.
The B3 fix must preserve scientific values and caller-owned geometry.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def _frame(groups):
    rng = np.random.default_rng(13103)
    rows = []
    for f in range(6):
        for g in range(groups):
            for j in range(20):
                x = (j + 0.5) / 20
                rows.append((f, f // 3, f % 3, f'group_{g:02d}_ABCDE', x,
                             0.2 * g + x + rng.normal(0, 0.1),
                             0.1 * g - x + rng.normal(0, 0.2)))
    return pd.DataFrame(rows, columns=['f', 'row', 'column', 'g', 'x', 'y0', 'y1'])


def _draw(frame, *, facet='2d', figsize=(8, 6), legend='shared', **overrides):
    opts = dict(type='profile', bins=5, range=(0, 1), min_entries=3,
                group_by='g', group_by_scope='global', legend=legend,
                quantiles=[.15, .5, .9], quantile_mode='discrete',
                auto_title=False, figsize=figsize)
    if facet == '1d':
        opts.update(facet_by='f', facet_by_bins=6, ncols=3)
    else:
        opts.update(facet_by=['row', 'column'], facet_by_bins=[2, 3])
    opts.update(overrides)
    return DFDraw(frame).draw('[y0,y1]:x', **opts)


def _bbox_check(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    bbox = legend.get_window_extent(renderer)
    pad = 1.0
    assert bbox.x0 >= fig.bbox.x0 - pad, ('left clipped', bbox.bounds)
    assert bbox.x1 <= fig.bbox.x1 + pad, ('right clipped', bbox.bounds)
    assert bbox.y0 >= fig.bbox.y0 - pad, ('bottom clipped', bbox.bounds)
    assert bbox.y1 <= fig.bbox.y1 + pad, ('top clipped', bbox.bounds)
    for ax in np.asarray(axes).flat:
        if ax.get_visible():
            tight = ax.get_tightbbox(renderer)
            assert not bbox.overlaps(tight), ('legend overlaps axes', bbox.bounds, tight.bounds)
    if fig._suptitle is not None:
        assert not bbox.overlaps(fig._suptitle.get_window_extent(renderer))
    for artist in (fig._supxlabel, fig._supylabel):
        if artist is not None:
            assert not bbox.overlaps(artist.get_window_extent(renderer))
    return legend


@pytest.mark.parametrize('groups,facet,figsize', [
    (2, '1d', (8, 6)),
    (5, '1d', (8, 6)),
    (7, '1d', (6, 4)),
    (10, '1d', (6, 4)),
    (2, '2d', (8, 6)),
    (5, '2d', (8, 6)),
    (7, '2d', (6, 4)),
    (10, '2d', (6, 4)),
])
def test_B3_01_02_04_05_06_auto_wrapping_and_semantic_completeness(groups, facet, figsize):
    fig, axes, _ = _draw(_frame(groups), facet=facet, figsize=figsize)
    try:
        legend = _bbox_check(fig, axes)
        labels = [t.get_text() for t in legend.get_texts()]
        expected = (['y0', 'y1'] +
                    [f'group_{g:02d}_ABCDE' for g in range(groups)] +
                    ['q=15%', 'median', 'q=90%'])
        assert sorted(labels) == sorted(expected)
        assert len(labels) == len(set(labels)) == 2 + groups + 3
        if groups >= 5:
            assert legend._ncols < min(len(expected), 8)
        assert all(ax.get_legend() is None for ax in np.asarray(axes).flat)
    finally:
        plt.close(fig)


def test_B3_03_two_dimensional_outer_headings_title_and_shared_labels():
    fig, axes, _ = _draw(_frame(5), figsize=(8, 6),
                         facet_labels='outer',
                         shared_xlabel='x occupancy',
                         shared_ylabel='Residual',
                         title='Global group profile comparison')
    try:
        _bbox_check(fig, axes)
    finally:
        plt.close(fig)


def test_B3_07_explicit_columns_location_anchor_and_title_remain_authoritative():
    opts = {'mode': 'shared', 'ncol': 8, 'loc': 'lower center',
            'bbox_to_anchor': (.5, .02), 'fontsize': 7,
            'title': 'Analyst groups', 'frameon': True}
    fig, axes, _ = _draw(_frame(5), facet='1d', legend=opts)
    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend._ncols == 8
        assert legend._loc == 8
        assert legend.get_title().get_text() == opts['title']
        assert legend.get_texts()[0].get_fontsize() == opts['fontsize']
        assert legend.get_frame_on()
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        np.testing.assert_allclose((anchor.x0, anchor.y0), opts['bbox_to_anchor'])
    finally:
        plt.close(fig)


def test_B3_08_numerical_lines_unchanged_by_auto_wrapping():
    frame = _frame(5)
    fig1, axes1, _ = _draw(frame, facet='1d', figsize=(6, 4), legend='shared')
    fig2, axes2, _ = _draw(frame, facet='1d', figsize=(6, 4), legend='none')
    try:
        for left, right in zip(np.asarray(axes1).flat, np.asarray(axes2).flat):
            assert len(left.lines) == len(right.lines)
            for l, r in zip(left.lines, right.lines):
                np.testing.assert_allclose(l.get_xdata(), r.get_xdata(), equal_nan=True)
                np.testing.assert_allclose(l.get_ydata(), r.get_ydata(), equal_nan=True)
    finally:
        plt.close(fig1)
        plt.close(fig2)
