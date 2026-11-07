"""PHASE_13_103_DF B2 — vector-profile fit textbox consolidation.

RED on B1 committed baseline: independent scalar fit textboxes occupy the
same axes position.  GREEN must preserve fit payloads and overlay semantics
while providing exactly one branch-labelled parameter annotation per axes.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw

BRANCHES = ('y0', 'y1', 'y2', 'y3')
VECTOR_EXPR = '[y0,y1,y2,y3]:x'


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def data():
    # Repeated x-values guarantee well-populated bins in each facet.
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
    return pd.DataFrame(rows, columns=('x', 'B', 'F', *BRANCHES))


def _axes(axes):
    return list(np.asarray(axes, dtype=object).ravel())


def _fit_boxes(ax):
    # Fit renderer annotates fit names; do not count auto titles/labels.
    return [artist for artist in ax.texts if 'pol1' in artist.get_text().lower()]


def _assert_consolidated(axes):
    populated = 0
    for ax in _axes(axes):
        texts = _fit_boxes(ax)
        if not texts:
            continue
        populated += 1
        assert len(texts) == 1, (
            'B2: vector fits must use one consolidated textbox per axes, '
            f'found {len(texts)} at {[t.get_position() for t in texts]}'
        )
        combined = texts[0].get_text()
        for branch in BRANCHES:
            assert branch in combined, (
                f'B2: branch {branch!r} missing from fit annotation: {combined!r}'
            )
    assert populated >= 1, 'B2: no fit textboxes rendered'


@pytest.mark.parametrize('entry', ('draw', 'profile'), ids=('generic-draw', 'typed-profile'))
def test_B2_01_four_branch_fit_annotation_is_consolidated(data, entry):
    drawer = DFDraw(data)
    opts = dict(bins=8, range=(0, 1), min_entries=3, fit='pol1',
                auto_title=False)
    if entry == 'draw':
        fig, axes, stats = drawer.draw(VECTOR_EXPR, type='profile', **opts)
    else:
        fig, axes, stats = drawer.profile(VECTOR_EXPR, **opts)
    assert isinstance(stats, list) and len(stats) == len(BRANCHES)
    assert all('fit' in s for s in stats)
    _assert_consolidated(axes)


@pytest.mark.parametrize('facets,counts', [(['B'], [2]), (['B','F'], [2,2])],
                         ids=('ordinary-1d', 'ordinary-2d'))
def test_B2_02_consolidated_annotations_in_each_facet(data, facets, counts):
    fig, axes, stats = DFDraw(data).draw(
        VECTOR_EXPR, type='profile', fit='pol1', bins=8,
        range=(0, 1), min_entries=3,
        facet_by=(facets[0] if len(facets) == 1 else facets),
        facet_by_bins=(counts[0] if len(counts) == 1 else counts),
        auto_title=False)
    populated = 0
    for ax in _axes(axes):
        if _fit_boxes(ax):
            populated += 1
            _assert_consolidated(ax)
    assert populated == (2 if len(facets) == 1 else 4)


def test_B2_03_paired_fit_specs_keep_distinct_branch_text(data):
    fig, ax, stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        fit=['pol1'] * 4, auto_title=False)
    assert len(stats) == 4
    _assert_consolidated(ax)


def test_B2_04_consolidation_respects_textbox_format_and_fontsize(data):
    fig, ax, stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        fit='pol1', auto_title=False,
        fit_textbox_kwargs={'format': 'compact', 'fontsize': 6})
    _assert_consolidated(ax)
    assert _fit_boxes(ax)[0].get_fontsize() == 6


def test_B2_05_scalar_profile_fit_textbox_remains_unchanged(data):
    fig, ax, stats = DFDraw(data).profile(
        'y0:x', bins=8, range=(0, 1), min_entries=3,
        fit='pol1', auto_title=False)
    assert 'fit' in stats
    assert len(_fit_boxes(ax)) == 1


def test_B2_06_vector_profile_without_fit_adds_no_fit_annotation(data):
    fig, ax, stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        auto_title=False)
    assert len(stats) == 4
    assert len(_fit_boxes(ax)) == 0


def test_B2_07_four_numerical_fit_payloads_equal_scalar_fits(data):
    fig, ax, vector_stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        fit='pol1', auto_title=False)
    assert len(vector_stats) == 4
    for branch, stat in zip(BRANCHES, vector_stats):
        sf, sa, scalar_stat = DFDraw(data).profile(
            f'{branch}:x', bins=8, range=(0, 1), min_entries=3,
            fit='pol1', auto_title=False)
        vector_fit = stat['fit'][0][0]
        scalar_fit = scalar_stat['fit'][0][0]
        assert vector_fit['fit_status'] == scalar_fit['fit_status'] == 'ok'
        np.testing.assert_allclose(vector_fit['params'], scalar_fit['params'],
                                   rtol=1e-10, atol=1e-10)


def test_B2_08_fit_overlays_preserve_branch_colors_and_title(data):
    fig, ax, stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        fit='pol1', auto_title=True)
    # Fit overlays use a 200-point dense grid; profile series use bin centers.
    overlays = [line for line in ax.lines if len(line.get_xdata()) == 200]
    profiles = [line for line in ax.lines if len(line.get_xdata()) == 8]
    # ErrorbarContainer contributes extra short-cap Line2D artists, so
    # compare the four semantic colors rather than line-object counts.
    assert len(overlays) == len(BRANCHES)
    from matplotlib.colors import to_rgba
    # Normalize Matplotlib color strings, cycles and RGBA tuples to one
    # representation, without relaxing the actual display-color equality.
    rgba = lambda line: tuple(to_rgba(line.get_color()))
    assert {rgba(line) for line in overlays} == {rgba(line) for line in profiles}
    assert ax.get_title(), 'B2 must not drop existing automatic scientific title'
    _assert_consolidated(ax)


@pytest.mark.parametrize('facets,counts', [(['B'], [2]), (['B', 'F'], [2, 2])],
                         ids=('ordinary-1d', 'ordinary-2d'))
def test_B2_09_rendered_fit_textboxes_fit_within_axes(data, facets, counts):
    fig, axes, stats = DFDraw(data).profile(
        VECTOR_EXPR, bins=8, range=(0, 1), min_entries=3,
        fit='pol1', auto_title=False, figsize=(16, 10),
        facet_by=(facets[0] if len(facets) == 1 else facets),
        facet_by_bins=(counts[0] if len(counts) == 1 else counts))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    populated = 0
    for ax in _axes(axes):
        boxes = _fit_boxes(ax)
        if not boxes:
            continue
        populated += 1
        assert len(boxes) == 1
        bb = boxes[0].get_window_extent(renderer)
        aa = ax.get_window_extent(renderer)
        assert aa.x0 <= bb.x0 and bb.x1 <= aa.x1
        assert aa.y0 <= bb.y0 and bb.y1 <= aa.y1
    assert populated == (2 if len(facets) == 1 else 4)


@pytest.mark.parametrize('facets,counts', [(['B'], [2]), (['B', 'F'], [2, 2])],
                         ids=('ordinary-1d', 'ordinary-2d'))
def test_B2_10_adf_public_path_fit_annotations_match_direct(data, facets, counts):
    # Must execute on Alma2.  The isolated reviewer runtime lacks ADF.
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    opts = dict(type='profile', bins=8, range=(0, 1), min_entries=3,
                fit='pol1', auto_title=False,
                facet_by=(facets[0] if len(facets) == 1 else facets),
                facet_by_bins=(counts[0] if len(counts) == 1 else counts))
    direct = DFDraw(data).draw(VECTOR_EXPR, **opts)
    result = adf_module.AliasDataFrame(data).draw(VECTOR_EXPR, **opts)
    assert np.asarray(direct[1]).shape == np.asarray(result[1]).shape
    for ax_direct, ax_adf in zip(_axes(direct[1]), _axes(result[1])):
        if _fit_boxes(ax_direct):
            assert [t.get_text() for t in _fit_boxes(ax_direct)] == [
                t.get_text() for t in _fit_boxes(ax_adf)]
    assert _fit_boxes(_axes(result[1])[0]), 'ADF did not expose fitted vector'
