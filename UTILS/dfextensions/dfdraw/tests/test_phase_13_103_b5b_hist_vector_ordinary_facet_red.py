"""PHASE_13_103_DF B5-B: expression-vector hist × one ordinary facet.

Locked D8 bin policy: one common facet-local range from the EXISTING
resolve_range_1d autorange on the concatenated, finite expression branches.
Caller-provided bins/range override auto. sharex changes display, never edges.
The list-shaped public stats remain vector-major, with facet-keyed entries.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw
from dfextensions.dfdraw.plots._autorange import resolve_range_1d
from dfextensions.dfdraw.style import get_style_value


def _frame():
    rng = np.random.default_rng(13505)
    n = 120
    b = np.repeat(['A', 'B', 'C'], n)
    y0 = np.r_[rng.normal(-3, .4, n), rng.normal(2, .5, n), np.full(n, np.nan)]
    y1 = np.r_[rng.normal(7, .8, n), rng.normal(11, .3, n), np.full(n, np.nan)]
    y0[7] = np.nan
    y1[15] = np.inf
    weights = np.tile(np.linspace(.5, 1.8, n), 3)
    weights[29] = np.nan
    return pd.DataFrame(dict(B=b, y0=y0, y1=y1, w=weights,
                             s=np.arange(len(b)) % 3, x=np.arange(len(b)) / 100.))


def _common_autorange(df, names):
    values = np.concatenate([df[name].to_numpy(dtype=float) for name in names])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (0., 1.)
    lohi, _ = resolve_range_1d(
        None, values, style_strategy=get_style_value('autorange.strategy', 'hybrid'),
        style_k_robust=get_style_value('autorange.k_robust', 4.0),
        style_k_outlier=get_style_value('autorange.k_outlier', 1.5),
        style_percentile=get_style_value('autorange.percentile', (1., 99.)))
    return lohi


def _assert_hist_data(stats, axes, frame, edges, *, weights=None):
    assert isinstance(stats, list) and len(stats) == 2
    for branch, name in enumerate(['y0', 'y1']):
        assert stats[branch]['expression'] == name
        assert stats[branch]['expression_index'] == branch
        assert stats[branch]['facet_by'] == 'B'
        for cell_index, facet in enumerate(['A', 'B', 'C']):
            cell = frame[frame.B == facet]
            values = cell[name].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if weights:
                ww = cell[weights].to_numpy(dtype=float)
                valid &= np.isfinite(ww)
                ww = ww[valid]
            else:
                ww = None
            raw, _ = np.histogram(values[valid], bins=edges[facet], weights=ww)
            patches = axes[cell_index].patches
            nbins = len(edges[facet]) - 1
            assert len(patches) == 2 * nbins
            branch_patches = patches[branch*nbins:(branch+1)*nbins]
            np.testing.assert_allclose([p.get_height() for p in branch_patches], raw,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose([p.get_x() for p in branch_patches], edges[facet][:-1])
            np.testing.assert_allclose([p.get_width() for p in branch_patches], np.diff(edges[facet]))
            assert stats[branch]['per_group'][facet]['n'] == sum(valid)


def test_B5_01_02_facet_local_autorange_and_numpy_counts():
    df = _frame()
    f, axes, stats = DFDraw(df).hist('[y0,y1]', facet_by='B', bins=9, histtype='bar')
    try:
        assert len(axes) == 3
        edges = {b: np.histogram_bin_edges(np.array([0.]), bins=9, range=_common_autorange(df[df.B == b], ['y0','y1'])) for b in ['A', 'B', 'C']}
        assert edges['A'][0] != edges['B'][0]  # facet-local, not figure-wide
        _assert_hist_data(stats, axes, df, edges)
    finally:
        plt.close(f)


@pytest.mark.parametrize('explicit_bins,range_arg', [
    (np.linspace(-8, 16, 13), None),
    (8, (-8., 16.)),
])
def test_B5_02_explicit_bins_and_range_authoritative(explicit_bins, range_arg):
    df = _frame()
    f, axes, stats = DFDraw(df).hist('[y0,y1]', facet_by='B', bins=explicit_bins,
                                   range=range_arg, histtype='bar')
    try:
        edges = {b: (np.asarray(explicit_bins) if not isinstance(explicit_bins, int)
                     else np.linspace(*range_arg, explicit_bins+1)) for b in ['A','B','C']}
        _assert_hist_data(stats, axes, df, edges)
    finally:
        plt.close(f)


def test_B5_03_scalar_weights_and_selection_are_numerically_correct():
    df = _frame()
    opts = dict(facet_by='B', bins=8, range=(-8, 16), weights='w', selection='s != 0', histtype='bar')
    f, axes, stats = DFDraw(df).hist('[y0,y1]', **opts)
    try:
        subset = df[df.s != 0]
        edges = {b: np.linspace(-8, 16, 9) for b in ['A','B','C']}
        _assert_hist_data(stats, axes, subset, edges, weights='w')
        for b in ['A', 'B', 'C']:
            for i, name in enumerate(['y0','y1']):
                sf, sa, scalar = DFDraw(subset[subset.B == b]).hist(name, bins=8,
                    range=(-8,16), weights='w', histtype='bar')
                try:
                    np.testing.assert_allclose([p.get_height() for p in sa.patches],
                                               [p.get_height() for p in axes[['A','B','C'].index(b)].patches[i*8:(i+1)*8]])
                finally:
                    plt.close(sf)
    finally:
        plt.close(f)


def test_B5_04_generic_draw_and_histo_alias_parity():
    df = _frame()
    returned = []
    for t in ('hist', 'histo'):
        fig, axes, stats = DFDraw(df).draw('[y0,y1]', type=t, facet_by='B',
                          bins=7, range=(-8,16), histtype='bar')
        returned.append((fig, axes, stats))
    try:
        assert len(returned[0][1]) == 3
        for i in range(2):
            for b in ['A','B','C']:
                cell_idx = ['A','B','C'].index(b)
                p0 = returned[0][1][cell_idx].patches[i*7:(i+1)*7]
                p1 = returned[1][1][cell_idx].patches[i*7:(i+1)*7]
                np.testing.assert_array_equal([p.get_height() for p in p0], [p.get_height() for p in p1])
    finally:
        for fig, _, _ in returned:
            plt.close(fig)


def test_B5_05_06_old_nonfaceted_and_vector_facet_hist_preserved():
    df = _frame().iloc[:240]
    for kwargs in ({}, {'facet_by':'vector'}):
        fig, ax, stats = DFDraw(df).hist('[y0,y1]', bins=8, range=(-8,16), **kwargs)
        try:
            assert len(plt.gcf().axes) >= 1
            assert stats
        finally:
            plt.close(fig)


def test_B5_07_08_scalar_hist_facets_and_profile_scatter_unmodified():
    df = _frame().iloc[:240]
    cases = [('hist', 'y0'), ('profile', 'y0:x'), ('scatter', 'y0:x')]
    for kind, expr in cases:
        fig, axes, stats = DFDraw(df).draw(expr, type=kind, facet_by='B', bins=7)
        try:
            assert isinstance(stats, dict)
            assert list(stats['per_group']) == ['A','B']
        finally:
            plt.close(fig)


def test_B5_09_empty_nan_and_range_filtered_cells():
    df = _frame()
    fig, axes, stats = DFDraw(df).hist('[y0,y1]', facet_by='B', bins=8,
                  range=(-8, 8), histtype='bar')
    try:
        for i in range(2):
            assert stats[i]['per_group']['C']['n'] == 0
            assert sum(p.get_height() for p in axes[2].patches[i*8:(i+1)*8]) == 0
        assert sum(p.get_height() for p in axes[1].patches[8:16]) == 0
    finally:
        plt.close(fig)


def test_B5_10_quantiles_drawn_per_branch_and_cell():
    """PHASE_13_107_DF (change rule): histogram quantiles were refused on
    this route because their semantics were undefined; PHASE_13_107 defines
    them (all selected finite values of each drawn curve).  Every branch of
    every cell now carries its own record, equal to NumPy on its rows."""
    df = _frame()
    fig, axes, stats = DFDraw(df).draw('[y0,y1]', type='hist', facet_by='B',
                                       quantiles=[.2, .8])
    try:
        for branch, expr in zip(stats, ('y0', 'y1')):
            for cell, cell_stats in branch['per_group'].items():
                rows = df[df.B.astype(str) == str(cell)][expr].to_numpy(float)
                rows = rows[np.isfinite(rows)]
                got = cell_stats['hist_quantiles']['x']
                if rows.size == 0:      # an all-NaN cell has no quantiles
                    assert all(np.isnan(got))
                    continue
                assert got == pytest.approx(np.quantile(rows, [.2, .8]),
                                            abs=1e-12)
    finally:
        plt.close(fig)


def test_B5_duplicate_expression_identity_refused_preallocation():
    df = _frame()
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match='duplicate|unique'):
        DFDraw(df).hist('[y0,y0]', facet_by='B')
    assert set(plt.get_fignums()) == before


def test_B5_unsupported_vector_cross_products_refuse_preallocation():
    df = _frame()
    before = set(plt.get_fignums())
    with pytest.raises(NotImplementedError, match='selection_vector'):
        DFDraw(df).hist('[y0,y1]', facet_by='B', selection_vector=['s==1','s==2'])
    assert set(plt.get_fignums()) == before


def test_B5_11_adf_public_path_matches_scalar_numpy_reference():
    adf_mod = pytest.importorskip('dfextensions.AliasDataFrame')
    df = _frame()
    adf = adf_mod.AliasDataFrame(df)
    opts = dict(type='hist', facet_by='B', bins=10,
                range=(-8,16), selection='s != 0', weights='w', histtype='bar')
    fig1, axes1, stats1 = DFDraw(df).draw('[y0,y1]', **opts)
    fig2, axes2, stats2 = adf.draw('[y0,y1]', **opts)
    try:
        assert len(stats1) == len(stats2) == 2
        assert len(axes1) == len(axes2) == 3
        for i in range(2):
            for facet in ['A','B','C']:
                assert stats1[i]['per_group'][facet]['n'] == stats2[i]['per_group'][facet]['n']
        for a, b in zip(axes1, axes2):
            np.testing.assert_allclose([p.get_height() for p in a.patches],
                                       [p.get_height() for p in b.patches])
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_B5_computed_expression_vectors_and_facet_binning():
    df = _frame().iloc[:240].copy()
    df['F'] = (df.x * 4).astype(float)
    fig, axes, stats = DFDraw(df).draw('[y0+1,y1-y0]', type='histo',
           facet_by='F', facet_by_bins=4, bins=12, range=(-8,16), histtype='bar')
    try:
        assert len(stats) == 2
        assert len(axes) == 4
        assert set(stats[0]['per_group']) == set(stats[1]['per_group'])
        assert stats[0]['expression'] == 'y0+1'
        assert stats[1]['expression'] == 'y1-y0'
    finally:
        plt.close(fig)
