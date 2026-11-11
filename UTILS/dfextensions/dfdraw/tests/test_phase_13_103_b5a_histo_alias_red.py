"""PHASE_13_103_DF B5-A: ROOT-style histo alias before expression-vector dispatch.

This slice originally normalized a spelling only. B5-B now adds ordinary
column-facet support; alias parity and unsupported neighbor refusals remain.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def data():
    x = np.linspace(-2.75, 3.25, 240)
    return pd.DataFrame({
        'a': x,
        'b': 1.7 * x + 0.25,
        'B': np.repeat(np.arange(6), 40),
        'C': np.tile([0, 1], 120),
        'D': np.tile([0, 1, 2], 80),
    })


def _args(mode):
    opts = {'bins': 13, 'range': (-6.0, 8.0), 'auto_title': False}
    if mode == 'ratio':
        opts.update(normalize='ratio', vector_compose='outer')
    if mode == 'vector-facet':
        opts.update(facet_by='vector', ncols=2)
    return opts


def _compare_draw_results(left, right, mode):
    fig_a, axes_a, stat_a = left
    fig_b, axes_b, stat_b = right
    assert len(fig_a.axes) == len(fig_b.axes)
    assert len(np.asarray(axes_a, dtype=object).ravel()) == len(
        np.asarray(axes_b, dtype=object).ravel())
    for ax_a, ax_b in zip(fig_a.axes, fig_b.axes):
        assert len(ax_a.patches) == len(ax_b.patches)
        assert len(ax_a.lines) == len(ax_b.lines)
        np.testing.assert_allclose(ax_a.get_xlim(), ax_b.get_xlim(), rtol=0, atol=0)
        for pa, pb in zip(ax_a.patches, ax_b.patches):
            np.testing.assert_allclose(pa.get_path().vertices, pb.get_path().vertices,
                                       rtol=0, atol=0)
    if mode == 'ratio':
        pd.testing.assert_frame_equal(stat_a['normalize_data'],
                                      stat_b['normalize_data'], check_exact=True)
        for a, b in zip(stat_a['hist_branch_data'], stat_b['hist_branch_data']):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif mode == 'vector-facet':
        assert stat_a['n_groups'] == stat_b['n_groups'] == 2
        assert stat_a['n_total'] == stat_b['n_total']
    else:
        assert len(stat_a) == len(stat_b) == 2
        for a, b in zip(stat_a, stat_b):
            assert a['n'] == b['n']
            assert a['mean'] == b['mean']


@pytest.mark.parametrize('mode', ['overlay', 'ratio', 'vector-facet'])
def test_B5A_01_vector_alias_matches_hist_dispatch_and_numerics(data, mode):
    opts = _args(mode)
    expected = DFDraw(data).draw('[a,b]', type='hist', **opts)
    actual = DFDraw(data).draw('[a,b]', type='histo', **opts)
    _compare_draw_results(expected, actual, mode)


@pytest.mark.parametrize('faceted', [False, True], ids=['scalar', 'ordinary-facet'])
def test_B5A_02_scalar_alias_parity_remains_unchanged(data, faceted):
    opts = {'bins': 13, 'range': (-6.0, 8.0)}
    if faceted:
        opts.update(facet_by='B', facet_by_bins=6)
    a = DFDraw(data).draw('a', type='hist', **opts)
    b = DFDraw(data).draw('a', type='histo', **opts)
    assert len(a[0].axes) == len(b[0].axes)
    assert a[2]['n_total' if faceted else 'n'] == b[2]['n_total' if faceted else 'n']


@pytest.mark.parametrize('case', ['ordinary-facet', 'quantiles', 'normalize'])
def test_B5A_03_hist_vector_ordinary_facet_alias_and_refusals(data, case):
    opts = {'facet_by': 'B', 'facet_by_bins': 6, 'bins': 13}
    if case == 'quantiles':
        opts['quantiles'] = [0.10, 0.50, 0.90]
    if case == 'normalize':
        opts['normalize'] = 'ratio'
    if case == 'quantiles':
        # PHASE_13_107_DF (change rule): histogram quantiles are now defined
        # (per drawn curve); both spellings give identical records.
        first = DFDraw(data).draw('[a,b]', type='hist', **opts)
        second = DFDraw(data).draw('[a,b]', type='histo', **opts)
        for left, right in zip(first[2], second[2]):
            for cell in left['per_group']:
                assert (left['per_group'][cell]['hist_quantiles']
                        == right['per_group'][cell]['hist_quantiles'])
        plt.close(first[0])
        plt.close(second[0])
        return
    if case == 'ordinary-facet':
        # B5-B changes this one previously refused neighbor to supported.
        first = DFDraw(data).draw('[a,b]', type='hist', **opts)
        second = DFDraw(data).draw('[a,b]', type='histo', **opts)
        assert len(first[1]) == len(second[1]) == 6
        assert len(first[2]) == len(second[2]) == 2
        for left, right in zip(first[2], second[2]):
            for cell in left['per_group']:
                assert left['per_group'][cell]['n'] == right['per_group'][cell]['n']
        return
    messages = []
    before = set(plt.get_fignums())
    for spelling in ('hist', 'histo'):
        with pytest.raises((NotImplementedError, ValueError)) as exc:
            DFDraw(data).draw('[a,b]', type=spelling, **opts)
        messages.append(str(exc.value))
        assert set(plt.get_fignums()) == before
    assert messages[0] == messages[1]
    assert ('facet_by' in messages[0] or 'facet' in messages[0]
            or 'quantiles' in messages[0])


@pytest.mark.parametrize('expr', ['a', '[a,b]'], ids=['scalar', 'vector'])
def test_B5A_04_draw_batch_histo_alias_matches_hist(data, expr):
    hist = DFDraw(data).draw_batch({'p': {'expr': expr, 'type': 'hist',
                                          'bins': 13}}, verbose=False)
    alias = DFDraw(data).draw_batch({'p': {'expr': expr, 'type': 'histo',
                                           'bins': 13}}, verbose=False)
    assert hist['_summary']['success'] == alias['_summary']['success'] == 1
    assert hist['_errors'] == alias['_errors'] == {}
    if expr == 'a':
        assert hist['p']['stats']['n'] == alias['p']['stats']['n']
    else:
        assert [s['n'] for s in hist['p']['stats']] == [
            s['n'] for s in alias['p']['stats']]


@pytest.mark.parametrize('mode', ['overlay', 'ratio', 'vector-facet'])
def test_B5A_05_canonical_hist_vector_routes_stay_supported(data, mode):
    fig, ax, stats = DFDraw(data).draw('[a,b]', type='hist', **_args(mode))
    assert fig is not None
    if mode == 'overlay':
        assert isinstance(stats, list) and len(stats) == 2
    elif mode == 'ratio':
        assert len(stats['hist_branch_data']) == 2
    else:
        assert stats['n_groups'] == 2


@pytest.mark.parametrize('kind', ['profile', 'scatter'])
def test_B5A_06_other_vector_plot_types_remain_unchanged(data, kind):
    opts = {'bins': 13} if kind == 'profile' else {}
    fig, ax, stats = DFDraw(data).draw('[a,b]:B', type=kind, **opts)
    assert fig is not None
    assert stats is not None


def test_B5A_07_B4_three_dimensional_facet_figsize_refusal_before_figure(data):
    """B4 review C2: lock the typed 3-D refusal reached after B4 forwarding."""
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match='figsize for three-dimensional faceting'):
        DFDraw(data).draw('a:B', type='profile', bins=13,
                          facet_by=['B', 'C', 'D'],
                          facet_by_bins=[None, None, None], figsize=(16, 10))
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('case', ['overlay', 'ordinary-facet'])
def test_B5A_08_adf_histo_alias_matches_dfdraw(data, case):
    adf_mod = pytest.importorskip('dfextensions.AliasDataFrame')
    adf = adf_mod.AliasDataFrame(data)
    opts = _args('overlay')
    if case == 'ordinary-facet':
        opts.update(facet_by='B', facet_by_bins=6)
        left = DFDraw(data).draw('[a,b]', type='hist', **opts)
        right = adf.draw('[a,b]', type='histo', **opts)
        assert len(left[1]) == len(right[1]) == 6
        assert len(left[2]) == len(right[2]) == 2
        for branch in range(2):
            for cell in left[2][branch]['per_group']:
                assert left[2][branch]['per_group'][cell]['n'] == right[2][branch]['per_group'][cell]['n']
    else:
        left = DFDraw(data).draw('[a,b]', type='hist', **opts)
        right = adf.draw('[a,b]', type='histo', **opts)
        _compare_draw_results(left, right, 'overlay')
