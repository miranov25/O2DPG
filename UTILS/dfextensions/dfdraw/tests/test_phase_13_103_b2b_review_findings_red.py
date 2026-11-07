"""PHASE_13_103_DF B2b — independent reviewer findings, RED on c84ac3dd.

Covers default-size facet fit-box overflow, explicit caller formatting via
DFDraw.draw(), and retention of prior fit annotations across a new DFDraw
instance with same=True. This file is a focused RED gate, not a gallery change.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


@pytest.fixture(autouse=True)
def _isolate_pyplot():
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def data():
    rows = []
    for b in (0, 1):
        for f in (0, 1):
            for i in range(8):
                for k in range(14):
                    x = (i + 0.5) / 8
                    delta = (k - 6.5) / 100
                    values = [0.3 * j + (0.7 + 0.3 * j) * x
                              + (j + 1) * delta + 0.1 * b - 0.08 * f
                              for j in range(4)]
                    rows.append((x, b + 0.2, f + 0.2, *values))
    return pd.DataFrame(rows, columns=('x', 'B', 'F', 'y0', 'y1', 'y2', 'y3'))


EXPR = '[y0,y1,y2,y3]:x'


def _fit_boxes(ax):
    return [t for t in ax.texts if getattr(t, '_dfdraw_fit_textbox', False)]


@pytest.mark.parametrize('figsize', [None, (7, 5)], ids=('default', 'small'))
def test_B2B_01_four_fit_labels_stay_within_each_2d_facet(data, figsize):
    kwargs = {} if figsize is None else {'figsize': figsize}
    fig, axes, stats = DFDraw(data).draw(
        EXPR, type='profile', fit='pol1', bins=8, range=(0, 1),
        min_entries=3, facet_by=['B', 'F'], facet_by_bins=[2, 2],
        auto_title=False, **kwargs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    populated = 0
    for ax in np.asarray(axes, dtype=object).flat:
        fits = _fit_boxes(ax)
        if not fits:
            continue
        populated += 1
        assert len(fits) == 1
        text = fits[0]
        for label in ('[y0]', '[y1]', '[y2]', '[y3]'):
            assert label in text.get_text(), f'missing branch {label}'
        patch = text.get_bbox_patch()
        assert patch is not None, 'fit annotation bbox must remain rendered'
        bbox = patch.get_window_extent(renderer)
        owner = ax.get_window_extent(renderer)
        tolerance = 1.5  # rasterization/rounding only, not a layout margin
        assert bbox.x0 >= owner.x0 - tolerance, f'left overflow: {bbox} vs {owner}'
        assert bbox.x1 <= owner.x1 + tolerance, f'right overflow: {bbox} vs {owner}'
        assert bbox.y0 >= owner.y0 - tolerance, f'bottom overflow: {bbox} vs {owner}'
        assert bbox.y1 <= owner.y1 + tolerance, f'top overflow: {bbox} vs {owner}'
        assert bbox.x0 >= figure_bbox.x0 - tolerance
        assert bbox.x1 <= figure_bbox.x1 + tolerance
        assert bbox.y0 >= figure_bbox.y0 - tolerance
        assert bbox.y1 <= figure_bbox.y1 + tolerance
        if ax.get_legend() is not None:
            assert not bbox.overlaps(ax.get_legend().get_window_extent(renderer)), (
                'fit annotation crosses its own axes legend')
        for figure_legend in fig.legends:
            assert not bbox.overlaps(figure_legend.get_window_extent(renderer)), (
                'fit annotation crosses a figure-level legend')
    assert populated == 4


def test_B2B_02_generic_draw_forwards_fit_textbox_options(data):
    _, ax, _ = DFDraw(data).draw(
        EXPR, type='profile', fit='pol1', bins=8,
        range=(0, 1), min_entries=3, auto_title=False,
        fit_textbox_kwargs={'format': 'multiline', 'fontsize': 6})
    fits = _fit_boxes(ax)
    assert len(fits) == 1
    assert fits[0].get_fontsize() == 6
    assert '\n' in fits[0].get_text()
    assert '  ' in fits[0].get_text(), 'expected multiline fit parameter indentation'


def test_B2B_03_generic_draw_rejects_unknown_textbox_options(data):
    with pytest.raises(ValueError, match='unknown sub-key'):
        DFDraw(data).draw(
            EXPR, type='profile', fit='pol1', bins=8,
            range=(0, 1), min_entries=3, auto_title=False,
            fit_textbox_kwargs={'bogus': True})


def test_B2B_04_same_true_new_object_preserves_earlier_fit_box(data):
    fig, ax = plt.subplots()
    plt.sca(ax)
    _, first_ax, _ = DFDraw(data).profile(
        'y0:x', fit='pol1', bins=8, range=(0, 1),
        min_entries=3, auto_title=False, ax=ax)
    assert first_ax is ax
    first = _fit_boxes(ax)
    assert len(first) == 1
    previous_text = first[0]
    previous_content = previous_text.get_text()
    plt.sca(ax)
    _, reused_ax, stats = DFDraw(data).profile(
        EXPR, fit='pol1', bins=8, range=(0, 1),
        min_entries=3, auto_title=False, same=True)
    assert reused_ax is ax
    assert len(stats) == 4
    after = _fit_boxes(ax)
    assert len(after) == 2, 'new vector annotation must not remove prior scalar fit box'
    assert previous_text in after and previous_text.get_text() == previous_content
    assert any(all(f'[{label}]' in t.get_text() for label in ('y0', 'y1', 'y2', 'y3'))
               for t in after if t is not previous_text)
