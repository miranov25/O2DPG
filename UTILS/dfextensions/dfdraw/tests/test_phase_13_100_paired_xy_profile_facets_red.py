"""PHASE_13_100_DF — paired X/Y vector profile × ordinary facets.

RED-first acceptance probes. Source of truth: scalar decomposition over
individually resolved X ranges and branch-local finite (X,Y) row domains.
The source change must not alter the established nonfaceted or shared-X paths.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def make_df():
    rows = []
    for facet in ('A', 'B', 'C'):
        for x in range(4):
            for rep in range(5):
                shift = {'A': 0, 'B': 5, 'C': -3}[facet]
                rows.append(dict(
                    f=facet, keep=(rep != 4),
                    x0=float(x) + rep * .08,
                    x1=float(50 + 10*x) + rep * .7,
                    y0=float(shift + x * 2 + rep),
                    y1=float(100 + shift + 3*x + 2*rep)))
    return pd.DataFrame(rows)


def check_cells(df, stats, branch_i, y, x, lohi, **options):
    assert len(stats) == 2
    branch = stats[branch_i]
    assert branch['faceted'] is True
    assert branch['groups'] == ['A', 'B', 'C']
    assert branch['facet_by'] == 'f'
    for facet in branch['groups']:
        cell = branch['per_group'][facet]
        sub = df[df.f == facet]
        _, _, scalar = DFDraw(sub).profile(
            f'{y}:{x}', bins=4, range=lohi, return_data=True,
            min_entries=1, **options)
        t = cell['profile_data'].reset_index(drop=True)
        expected = scalar['profile_data'].reset_index(drop=True)
        for col in ['x_low', 'x_high', 'x_center', 'count', 'y_mean', 'y_std', 'y_sem']:
            np.testing.assert_allclose(t[col], expected[col], rtol=1e-12, atol=1e-12,
                                       equal_nan=True, err_msg=f'{facet} {branch_i} {col}')


def test_PXY01_old_shared_x_remains_green():
    df = make_df()
    fig, _, stats = DFDraw(df).profile('[y0,y1]:x0', bins=4,
           range=(-.5, 4), facet_by='f', return_data=True, min_entries=1)
    assert len(stats) == 2
    assert all(stats[i]['n_groups'] == 3 for i in range(2))


def test_PXY02_no_facet_pair_is_already_supported():
    fig, _, stats = DFDraw(make_df()).profile('[y0,y1]:[x0,x1]', bins=4,
                range=(-.5, 90), return_data=True, min_entries=1)
    assert isinstance(stats, list) and len(stats) == 2


@pytest.mark.parametrize('door', ['typed', 'draw'])
def test_PXY03_pair_with_ordinary_facet_is_scalar_equivalent(door):
    df = make_df()
    drawer = DFDraw(df)
    method = drawer.profile if door == 'typed' else drawer.draw
    opts = {'type':'profile'} if door == 'draw' else {}
    fig, axes, stats = method('[y0,y1]:[x0,x1]', facet_by='f',
        bins=4, range=(0., 90.), return_data=True, min_entries=1, **opts)
    assert len(np.asarray(axes).ravel()) == 3
    check_cells(df, stats, 0, 'y0', 'x0', (0., 90.))
    check_cells(df, stats, 1, 'y1', 'x1', (0., 90.))


def test_PXY04_asymmetric_finiteness_and_selection_are_branch_local():
    df = make_df()
    df.loc[(df.f == 'B') & (df.x0 < 1), 'y0'] = np.nan
    df.loc[(df.f == 'B') & (df.x1 > 68), 'y1'] = np.nan
    fig, _, stats = DFDraw(df).profile('[y0,y1]:[x0,x1]', facet_by='f',
        selection='keep', bins=4, range=(0., 90.), return_data=True, min_entries=1)
    selected = df[df.keep]
    check_cells(selected, stats, 0, 'y0', 'x0', (0., 90.))
    check_cells(selected, stats, 1, 'y1', 'x1', (0., 90.))
    assert stats[0]['per_group']['B']['n'] != stats[1]['per_group']['B']['n']


def test_PXY05_independent_auto_ranges_reused_in_every_facet():
    df = make_df()
    fig, _, stats = DFDraw(df).profile('[y0,y1]:[x0,x1]', facet_by='f',
             bins=4, return_data=True, min_entries=1, range='minmax')
    for i in (0,1):
        edges = [branch['profile_data'][['x_low','x_high']].to_numpy()
                 for branch in stats[i]['per_group'].values()]
        for e in edges[1:]:
            np.testing.assert_allclose(e, edges[0], rtol=0, atol=0)
    assert all(s['autorange_strategy'] == 'minmax' for branch in stats
               for s in branch['per_group'].values())
    e0 = stats[0]['per_group']['A']['profile_data'].x_low.iloc[0]
    e1 = stats[1]['per_group']['A']['profile_data'].x_low.iloc[0]
    assert e0 < 2 and e1 > 45


def test_PXY06_discrete_quantiles_and_legend_keep_pair_identity():
    df = make_df()
    fig, _, stats = DFDraw(df).profile('[y0,y1]:[x0,x1]', facet_by='f',
             bins=4, range=(0., 90.), quantiles=[.15,.5,.9],
             legend='shared', return_data=True, min_entries=1)
    assert len(stats) == 2
    labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    assert any('y0' in label and 'x0' in label for label in labels), labels
    assert any('y1' in label and 'x1' in label for label in labels), labels


def test_PXY07_vector_length_mismatch_refuses_before_figure():
    df = make_df()
    with pytest.raises(ValueError, match='broadcast|length|mismatch'):
        DFDraw(df).profile('[y0,y1,y0]:[x0,x1]', facet_by='f')


def test_PXY08_computed_x_components_materialize_correctly():
    df = make_df()
    fig, _, stats = DFDraw(df).profile('[y0,y1]:[x0+1,x1*2]', facet_by='f',
             bins=4, range=(0,180), return_data=True, min_entries=1)
    assert len(stats) == 2
    assert stats[0]['n_groups'] == 3 and stats[1]['n_groups'] == 3


def test_PXY09_synthetic_ALICE_calibration_signature_six_facets():
    """The exact public-argument pattern, on injected synthetic columns."""
    df = make_df().rename(columns={
        'x0': 'qTotRA0_Corr0', 'x1': 'qTotRA0_FitFull',
        'y0': 'qTotRA0_Corr0_v0', 'y1': 'qTotRA0_FitFull_v0',
    })
    df['isOKR0'] = True
    df['isOK0'] = True
    df['fPidIndex'] = 2
    df['fTgl'] = np.arange(len(df), dtype=float) % 6
    fig, axes, stats = DFDraw(df).draw(
        '[qTotRA0_Corr0_v0,qTotRA0_FitFull_v0]:'
        '[qTotRA0_Corr0,qTotRA0_FitFull]',
        type='profile', selection='isOKR0&isOK0&(fPidIndex==2)',
        quantiles=[.15,.5,.9], facet_by='fTgl', facet_by_bins=6,
        ncols=3, sharex=True, sharey=True, legend='shared', auto_title=True,
        bins=4, return_data=True, min_entries=1)
    assert len(np.asarray(axes).ravel()) == 6
    assert len(stats) == 2
    for branch in stats:
        assert branch['n_groups'] == 6
    assert fig._suptitle is not None
    labels = [tx.get_text() for leg in fig.legends for tx in leg.get_texts()]
    assert any('Corr0_v0' in v and 'Corr0' in v for v in labels)
    assert any('FitFull_v0' in v and 'FitFull' in v for v in labels)
