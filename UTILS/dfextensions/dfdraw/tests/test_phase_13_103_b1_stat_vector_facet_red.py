"""PHASE_13_103_DF B1: singleton std + expression-vector ordinary facets.

Scientific oracle: each faceted expression's per-bin sample standard
 deviation must match both the existing scalar profile owner and an
 independent NumPy calculation (ddof=1). No new estimator is introduced.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots.profile import _format_interval_label


Y0 = 'predict_mean_Mean-function_value'
Y1 = 'predict_mean_Linear-function_value'
EXPR = f'[{Y0},{Y1}]:C'


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def data():
    # Exactly populated reference bins, deliberately nonidentical fluctuations.
    rows = []
    noise = np.linspace(-1.0, 1.0, 18)
    for bi in range(2):
        for fi in range(3):
            for xi in range(8):
                for j, delta in enumerate(noise):
                    x = (xi + 0.5 + .0001 * delta) / 8
                    rows.append((
                        x, bi + .2, fi + .2,
                        .1 * bi + .05 * xi + (1 + .2 * fi) * delta,
                        -.05 * fi + .03 * xi + (1.6 + .1 * bi) * delta,
                        j != 0,
                    ))
    return pd.DataFrame(rows,
        columns=['C', 'B', 'actual_fraction', Y0, Y1, 'keep'])


def _opts(facet, facet_bins):
    return dict(type='profile', stat_list=['std'], facet_by=facet,
                facet_by_bins=facet_bins, bins=8, range=(0.0, 1.0),
                min_entries=3, legend='shared', auto_title=False,
                figsize=(16, 10))


def _std_from_rows(rows, y_name):
    edges = np.linspace(0, 1, 9)
    values = []
    for i in range(8):
        keep = ((rows.C >= edges[i]) &
                ((rows.C < edges[i + 1]) if i < 7 else
                 (rows.C <= edges[i + 1])))
        ys = rows.loc[keep, y_name].to_numpy()
        values.append(np.std(ys, ddof=1) if len(ys) >= 3 else np.nan)
    return np.asarray(values)


def _compare_scalar_and_numpy(data, facets, facet_bins, vector_stats):
    if isinstance(facets, str):
        bins = pd.cut(data[facets], bins=facet_bins)
        labels = [_format_interval_label(v) for v in bins.cat.categories]
        assert isinstance(vector_stats, list) and len(vector_stats) == 2
        for i, y_name in enumerate((Y0, Y1)):
            assert vector_stats[i]['groups'] == labels
            for key, interval in zip(labels, bins.cat.categories):
                array = vector_stats[i]['per_group'][key]['statistic_curves']['std']
                np.testing.assert_allclose(array, _std_from_rows(data[bins == interval], y_name),
                                           rtol=1e-12, atol=1e-12, equal_nan=True)
            sf, _, scalar = DFDraw(data).draw(f'{y_name}:C',
                **_opts(facets, facet_bins))
            try:
                for key in labels:
                    np.testing.assert_allclose(
                        vector_stats[i]['per_group'][key]['statistic_curves']['std'],
                        scalar['per_group'][key]['statistic_curves']['std'],
                        rtol=0, atol=0, equal_nan=True)
            finally:
                plt.close(sf)
    else:
        cuts = [pd.cut(data[col], bins=nb) for col, nb in zip(facets, facet_bins)]
        cell_keys = [key for key in vector_stats if isinstance(key, tuple)]
        assert len(cell_keys) == facet_bins[0] * facet_bins[1]
        for key in cell_keys:
            branch_stats = vector_stats[key]
            if not isinstance(branch_stats, list):
                assert branch_stats.get('empty')
                continue
            sub = data[(cuts[0] == key[0]) & (cuts[1] == key[1])]
            assert len(branch_stats) == 2
            for i, y_name in enumerate((Y0, Y1)):
                assert branch_stats[i]['statistic_branch']['expression_index'] == i
                vals = branch_stats[i]['statistic_curves']['std']
                np.testing.assert_allclose(vals, _std_from_rows(sub, y_name),
                                           rtol=1e-12, atol=1e-12, equal_nan=True)
        for i, y_name in enumerate((Y0, Y1)):
            sf, _, scalar = DFDraw(data).draw(f'{y_name}:C',
                **_opts(facets, facet_bins))
            try:
                for key in cell_keys:
                    if isinstance(vector_stats[key], list):
                        np.testing.assert_allclose(
                            vector_stats[key][i]['statistic_curves']['std'],
                            scalar[key]['statistic_curves']['std'],
                            rtol=0, atol=0, equal_nan=True)
            finally:
                plt.close(sf)


def test_B1_01_actual_banked_reproducer_no_group_2d(data):
    fig, axes, stats = DFDraw(data).draw(EXPR,
        **_opts(['B', 'actual_fraction'], [2, 3]),
        shared_xlabel=True, shared_ylabel=True,
        figure_margins=[.18, .97, .78, .27],
        subplot_spacing=[.25, .35])
    assert axes.shape == (2, 3)
    np.testing.assert_array_equal(fig.get_size_inches(), [16.0, 10.0])
    assert all(isinstance(stats[key], list) and len(stats[key]) == 2
               for key in stats if isinstance(key, tuple))


@pytest.mark.parametrize('facet,bins', [
    (['B', 'actual_fraction'], [2, 3]),
    (['B', 'actual_fraction'], [3, 5]),
    (['B', 'actual_fraction'], [4, 2]),
    ('B', 2),
], ids=['2d-2x3', '2d-3x5', '2d-4x2', '1d-2'])
def test_B1_02_noncommissioned_facet_counts_are_legal(data, facet, bins):
    fig, axes, stats = DFDraw(data).draw(EXPR, **_opts(facet, bins))
    expected_shape = (bins,) if isinstance(facet, str) else tuple(bins)
    assert axes.shape == expected_shape
    assert len(fig.axes) == int(np.prod(expected_shape))
    assert stats is not None


@pytest.mark.parametrize('facet,bins', [
    ('B', 2), (['B', 'actual_fraction'], [2, 3]),
], ids=['1d', '2d'])
def test_B1_03_scalar_decomposition_and_independent_numpy_std(data, facet, bins):
    fig, axes, stats = DFDraw(data).draw(EXPR, **_opts(facet, bins))
    _compare_scalar_and_numpy(data, facet, bins, stats)


@pytest.mark.parametrize('facet,bins', [
    ('B', 2), (['B', 'actual_fraction'], [2, 3]),
], ids=['1d', '2d'])
def test_B1_04_shared_legend_has_actual_expression_identity(data, facet, bins):
    fig, axes, stats = DFDraw(data).draw(EXPR, **_opts(facet, bins))
    assert len(fig.legends) == 1
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [Y0, Y1]
    for ax in axes.flat:
        if ax.lines:
            assert len(ax.lines) == 2
            assert ax.lines[0].get_color() != ax.lines[1].get_color()
            assert all(line._dfdraw_facet_semantic['kind'] == 'expression'
                       for line in ax.lines)


@pytest.mark.parametrize('facet,bins', [
    ('B', 2), (['B', 'actual_fraction'], [2, 3]),
], ids=['1d', '2d'])
def test_B1_05_typed_and_generic_public_surfaces_are_equivalent(data, facet, bins):
    kw = _opts(facet, bins)
    a = DFDraw(data).draw(EXPR, **kw)
    kw.pop('type')
    b = DFDraw(data).profile(EXPR, **kw)
    assert a[1].shape == b[1].shape
    assert [[t.get_text() for t in legend.get_texts()] for legend in a[0].legends] == [
           [t.get_text() for t in legend.get_texts()] for legend in b[0].legends]
    for aa, bb in zip(a[1].flat, b[1].flat):
        assert len(aa.lines) == len(bb.lines)
        for line_a, line_b in zip(aa.lines, bb.lines):
            np.testing.assert_array_equal(line_a.get_ydata(), line_b.get_ydata())


@pytest.mark.parametrize('case', [
    'median', 'multiple-statistics', 'multi-selection', 'multi-weights',
    'normalize', 'fit', 'grouped-uncommissioned', 'third-facet',
    'explicit-color', 'forced-marker',
])
def test_B1_06_unsupported_neighbors_refuse_before_figure(data, case):
    kw = _opts(['B', 'actual_fraction'], [2, 3])
    if case == 'median': kw['stat_list'] = ['median']
    elif case == 'multiple-statistics': kw['stat_list'] = ['mean', 'std']
    elif case == 'multi-selection': kw['selection_vector'] = ['C<0.5', 'C>=0.5']
    elif case == 'multi-weights': kw['weights_vector'] = ['C+1', 'C+2']
    elif case == 'normalize': kw['normalize'] = 'ratio'
    elif case == 'fit': kw['fit'] = 'pol1'
    elif case == 'grouped-uncommissioned': kw['group_by'] = 'keep'
    elif case == 'third-facet':
        kw['facet_by'] = ['B', 'actual_fraction', 'keep']
        kw['facet_by_bins'] = [2, 3, None]
    elif case == 'explicit-color': kw['color'] = 'red'
    elif case == 'forced-marker': kw['vector_style'] = 'marker'
    before = set(plt.get_fignums())
    with pytest.raises((NotImplementedError, ValueError)):
        DFDraw(data).draw(EXPR, **kw)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('facet,bins', [
    ('B', 2), (['B', 'actual_fraction'], [2, 3]),
], ids=['1d', '2d'])
def test_B1_07_legacy_vector_without_stat_list_still_renders(data, facet, bins):
    kw = _opts(facet, bins); kw.pop('stat_list')
    fig, axes, stats = DFDraw(data).draw(EXPR, **kw)
    assert len(fig.axes) == len(axes.flat)
    assert stats is not None


@pytest.mark.parametrize('facet,bins', [
    ('B', 2), (['B', 'actual_fraction'], [2, 3]),
], ids=['1d', '2d'])
def test_B1_08_adf_public_path_matches_direct(data, facet, bins):
    # Native ADF integration gate. Locally skips only when ADF is not installed;
    # Alma2 canonical commissioning must execute both parametrizations.
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    # Keep the cross-project dispatch check independent of ADF's alias parser
    # rules for hyphenated source names; the exact banked spelling is exercised
    # by B1-01 on the native dfdraw surface.
    plain = data.rename(columns={Y0: 'y0', Y1: 'y1'})
    adf = adf_module.AliasDataFrame(plain)
    kw = _opts(facet, bins)
    direct = DFDraw(plain).draw('[y0,y1]:C', **kw)
    result = adf.draw('[y0,y1]:C', **kw)
    assert direct[1].shape == result[1].shape
    for ax_a, ax_b in zip(direct[1].flat, result[1].flat):
        assert len(ax_a.lines) == len(ax_b.lines)
        for la, lb in zip(ax_a.lines, ax_b.lines):
            np.testing.assert_allclose(la.get_ydata(), lb.get_ydata(),
                                       rtol=0, atol=1e-12, equal_nan=True)
