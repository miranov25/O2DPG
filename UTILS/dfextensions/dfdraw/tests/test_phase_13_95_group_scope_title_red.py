"""PHASE_13_95_DF Stage A: RED-first grouping-scope and title contracts.

Independent physical-membership oracles use pandas.cut/qcut on the selected
full Figure domain. No expected failures are marked xfail: this file must be
RED against the committed PHASE_13_94_DF_END production baseline.
"""
from __future__ import annotations

import inspect

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


def data():
    """Two different group ranges, with a shared middle region and empty groups."""
    rows = []
    for facet_idx, facet in enumerate(('east', 'west')):
        for sub_idx, sub in enumerate(('low', 'high')):
            for component in range(6):
                for x in range(4):
                    for replicate in range(5):
                        group = facet_idx * 12. + component * 2. + .03 * replicate
                        rows.append(dict(facet=facet, sub=sub, x=float(x), g=group,
                                         y=facet_idx * 10 + sub_idx * 3 + component + x + replicate * .1,
                                         y2=20 + 3 * facet_idx - component + x * .5,
                                         keep=(replicate != 4)))
    return pd.DataFrame(rows)


def render(df=None, *, door='typed', expr='y:x', **kwargs):
    df = data() if df is None else df
    kw = dict(bins=4, range=(-.5, 3.5), min_entries=1, return_data=True)
    kw.update(kwargs)
    obj = DFDraw(df)
    if door == 'draw':
        return obj.draw(expr, type='profile', **kw)
    return obj.profile(expr, **kw)


def profile_tables(stats):
    """Read returned per-facet payloads without assuming a plotting color."""
    if not isinstance(stats, dict):
        return []
    result = []
    if isinstance(stats.get('per_group'), dict):
        for cell in stats['per_group'].values():
            if isinstance(cell, dict) and 'profile_data' in cell:
                result.append(cell['profile_data'])
    for cell in stats.values():
        if isinstance(cell, dict) and 'profile_data' in cell:
            result.append(cell['profile_data'])
        elif isinstance(cell, dict) and isinstance(cell.get('per_group'), dict):
            for inner in cell['per_group'].values():
                if isinstance(inner, dict) and 'profile_data' in inner:
                    result.append(inner['profile_data'])
    return result


def all_legend_labels(fig):
    return [[text.get_text() for text in leg.get_texts()] for leg in fig.legends]


def test_R1_public_signature_and_invalid_values():
    signature = inspect.signature(DFDraw.profile)
    assert signature.parameters['group_by_scope'].default == 'auto'
    assert inspect.signature(DFDraw.draw).parameters['group_by_scope'].default == 'auto'
    with pytest.raises(ValueError, match='group_by_scope'):
        render(group_by='g', group_by_bins=3, facet_by='facet', group_by_scope='WRONG')


def test_R2_auto_preserves_historical_1d_faceted_numbers():
    df = data()
    _, _, before = render(df, facet_by='facet', group_by='g', group_by_bins=3)
    _, _, after = render(df, facet_by='facet', group_by='g', group_by_bins=3,
                         group_by_scope='auto')
    assert repr(before) == repr(after)


def test_R3_explicit_global_1d_reuses_exact_physical_groups():
    df = data()
    fig, axes, stats = render(df, group_by='g', group_by_bins=3,
                               facet_by='facet', group_by_scope='global')
    assert axes is not None
    global_categories = pd.cut(df.g, bins=3).cat.categories
    assert len(global_categories) == 3
    titles = all_legend_labels(fig)
    assert len(titles) == 1
    assert len(titles[0]) == 3
    assert len(profile_tables(stats)) >= 1


def test_R4_explicit_local_1d_matches_auto():
    df = data()
    _, _, before = render(df, facet_by='facet', group_by='g', group_by_bins=3)
    _, _, after = render(df, facet_by='facet', group_by='g', group_by_bins=3,
                         group_by_scope='facet')
    assert repr(before) == repr(after)


def test_R5_global_quantiles_in_1d():
    df = data()
    fig, axes, stats = render(df, facet_by='facet', group_by='g',
                               group_by_quantiles=3, group_by_scope='global')
    assert pd.qcut(df.g, q=3, duplicates='drop').cat.categories.size == 3
    assert len(fig.legends) == 1
    assert profile_tables(stats)


def test_R6_global_missing_group_2d_has_stable_identity():
    df = data()
    fig, axes, stats = render(df, facet_by=['facet', 'sub'], group_by='g',
                               group_by_bins=4, group_by_scope='global')
    assert axes.shape == (2, 2)
    assert len(fig.legends) == 1
    assert all(not ax.get_legend() for ax in axes.flat)
    assert len([k for k in stats if isinstance(k, tuple)]) == 4


def test_R7_global_figure_legend_obeys_explicit_hide():
    fig, axes, stats = render(facet_by='facet', group_by='g', group_by_bins=3,
                               group_by_scope='global', legend=False)
    assert not fig.legends
    assert not any(ax.get_legend() for ax in np.asarray(axes).flat)


def test_R8_facet_local_has_no_false_global_numeric_legend():
    fig, axes, stats = render(facet_by=['facet', 'sub'], group_by='g',
                               group_by_bins=3, group_by_scope='facet')
    assert not fig.legends


def test_R9_13_94_auto_matches_omitted_policy():
    df = data()
    expr = '[y,y2]:x'
    _, _, old = render(df, expr=expr, facet_by=['facet', 'sub'],
                       group_by='g', group_by_bins=3)
    _, _, auto = render(df, expr=expr, facet_by=['facet', 'sub'],
                        group_by='g', group_by_bins=3, group_by_scope='auto')
    for coord in (('east', 'low'), ('east', 'high'), ('west', 'low'), ('west', 'high')):
        for i in (0, 1):
            pd.testing.assert_frame_equal(old[coord][i]['profile_data'],
                                          auto[coord][i]['profile_data'])


def test_R10_13_94_explicit_facet_policy_changes_group_boundaries():
    df = data()
    _, _, global_stats = render(df, expr='[y,y2]:x', facet_by=['facet', 'sub'],
                                 group_by='g', group_by_bins=3)
    _, _, local_stats = render(df, expr='[y,y2]:x', facet_by=['facet', 'sub'],
                                group_by='g', group_by_bins=3,
                                group_by_scope='facet')
    g = global_stats['east', 'low'][0]['profile_data']['group'].dropna().astype(str).unique()
    l = local_stats['east', 'low'][0]['profile_data']['group'].dropna().astype(str).unique()
    assert set(g) != set(l)
    assert not plt.gcf().legends, 'local bins must not have a false global interval legend'


def test_R11_1d_quantiles_appear_once_in_auto_title():
    fig, axes, stats = render(facet_by='facet', quantiles=[.45, .55], auto_title=True)
    assert fig._suptitle is not None
    text = fig._suptitle.get_text()
    assert '.45' in text and '.55' in text
    assert sum('.45' in t.get_text() for ax in np.asarray(axes).flat
               for t in ax.texts) == 0


def test_R12_title_false_and_explicit_title_precedence():
    fig, _, _ = render(facet_by='facet', quantiles=[.45, .55], auto_title=False)
    assert fig._suptitle is None
    fig2, _, _ = render(facet_by='facet', quantiles=[.45, .55],
                         auto_title=True, title='User title')
    assert fig2._suptitle.get_text() == 'User title'


@pytest.mark.parametrize('door', ['typed', 'draw'])
def test_R13_typed_draw_parity_for_explicit_global(door):
    fig, axes, stats = render(door=door, facet_by=['facet', 'sub'],
                               group_by='g', group_by_quantiles=3,
                               group_by_scope='global')
    assert axes.shape == (2, 2)
    assert len(fig.legends) == 1


def test_R14_categorical_no_binning_no_numeric_changes():
    df = data().assign(cat=lambda d: np.where(d.g < 12, 'first', 'second'))
    _, _, old = render(df, facet_by='facet', group_by='cat')
    _, _, new = render(df, facet_by='facet', group_by='cat', group_by_scope='auto')
    assert repr(old) == repr(new)


def test_R15_single_global_cut_event(monkeypatch):
    orig = pd.cut
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)
    monkeypatch.setattr(pd, 'cut', counted)
    render(facet_by=['facet', 'sub'], group_by='g', group_by_bins=3,
           group_by_scope='global')
    # pd.cut also runs for facet bucketing only when explicitly configured;
    # this case uses categorical facets.
    assert len(calls) == 1, calls


def test_R16_selection_domain_global_bin_membership_independent_pandas():
    df = data()
    selected = df[df.keep]
    classified = pd.cut(selected.g, bins=3)
    fig, axes, stats = render(df, facet_by='facet', group_by='g',
                               group_by_bins=3, group_by_scope='global',
                               selection='keep')
    legend_groups = all_legend_labels(fig)[0]
    assert len(legend_groups) == len(classified.cat.categories)
    for facet in ('east', 'west'):
        table = stats['per_group'][facet]['profile_data']
        actual = table.groupby('group')['count'].sum()
        for category, display in zip(classified.cat.categories, legend_groups):
            n = int(((selected.facet == facet) & (classified == category)).sum())
            assert int(actual.get(display, 0)) == n, (facet, category, display)


def test_R17_single_dimension_list_is_equivalent_to_scalar_facet():
    df = data()
    _, _, a = render(df, facet_by='facet', group_by='g',
                      group_by_bins=3, group_by_scope='global')
    _, _, b = render(df, facet_by=['facet'], group_by='g',
                      group_by_bins=3, group_by_scope='global')
    for coord in ('east', 'west'):
        pd.testing.assert_frame_equal(
            a['per_group'][coord]['profile_data'],
            b['per_group'][coord]['profile_data'])


def test_R18_global_quantile_boundaries_derived_from_selection_not_raw_input():
    df = data()
    selected = df[df.keep & (df.g > 10)]
    fig, axes, stats = render(df, facet_by='facet', group_by='g',
                               group_by_quantiles=3,
                               group_by_scope='global',
                               selection='keep & (g > 10)')
    classified = pd.qcut(selected.g, q=3, duplicates='drop')
    labels = all_legend_labels(fig)[0]
    assert len(labels) == len(classified.cat.categories)
    for facet in ('east', 'west'):
        table = stats['per_group'][facet]['profile_data']
        actual = table.groupby('group')['count'].sum()
        for category, label in zip(classified.cat.categories, labels):
            assert int(actual.get(label, 0)) == int(
                ((selected.facet == facet) & (classified == category)).sum())


def test_R19_more_than_two_quantiles_are_listed_not_misrepresented_as_range():
    fig, _, _ = render(facet_by='facet', quantiles=[.2, .5, .8], auto_title=True)
    text = fig._suptitle.get_text()
    assert '0.2' in text and '0.5' in text and '0.8' in text
    assert '–' not in text


def _assert_refusal_creates_no_figure(monkeypatch, operation, match):
    """Reject before Figure allocation without assuming pyplot starts empty.

    Full pytest-xdist workers can retain Figures created by other tests.
    Observe creation attempts and existing Figure identities independently.
    """
    before = tuple(plt.get_fignums())
    original_figure = plt.figure
    created = []

    def record_figure(*args, **kwargs):
        created.append(True)
        return original_figure(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(plt, 'figure', record_figure)
        with pytest.raises(NotImplementedError, match=match):
            operation()
    assert not created, 'unsupported scope attempted to allocate a Figure'
    assert tuple(plt.get_fignums()) == before, 'refusal changed existing Figures'


def test_R20_no_scope_widening_to_other_plot_families(monkeypatch):
    df = data()
    _assert_refusal_creates_no_figure(
        monkeypatch,
        lambda: DFDraw(df).draw('y:x', type='scatter', group_by_scope='global'),
        'group_by_scope',
    )


def test_R21_global_top_k_does_not_claim_unsupported_global_ranking(monkeypatch):
    # Deliberately start with a live Figure to prevent the earlier xdist
    # false failure (assert not plt.get_fignums()).
    plt.figure()
    _assert_refusal_creates_no_figure(
        monkeypatch,
        lambda: render(facet_by='facet', group_by='g', group_by_bins=3,
                       top_k=2, group_by_scope='global'),
        'top_k',
    )


def test_R22_2d_scalar_global_matches_selected_pandas_counts():
    df = data()
    selected = df[df.keep]
    groups = pd.cut(selected.g, bins=3)
    fig, axes, stats = render(df, facet_by=['facet', 'sub'], group_by='g',
                               group_by_bins=3, group_by_scope='global',
                               selection='keep')
    labels = all_legend_labels(fig)[0]
    for (facet, sub), payload in ((key, val) for key,val in stats.items()
                                  if isinstance(key, tuple)):
        table = payload['profile_data']
        counts = table.groupby('group')['count'].sum()
        for interval, label in zip(groups.cat.categories, labels):
            n = int(((selected.facet == facet) & (selected['sub'] == sub) &
                     (groups == interval)).sum())
            assert int(counts.get(label, 0)) == n


def test_R23_auto_vector_13_94_figure_legend_obeys_explicit_hide():
    fig, axes, stats = render(expr='[y,y2]:x', facet_by=['facet', 'sub'],
                               group_by='g', group_by_bins=3,
                               legend=False)
    assert len(fig.legends) == 0
    assert not any(ax.get_legend() for ax in axes.flat)


def test_R24_explicit_all_overrides_default_shared_global_legend():
    fig, axes, stats = render(facet_by='facet', group_by='g',
                               group_by_bins=3, group_by_scope='global',
                               legend='all')
    assert not fig.legends
    assert all(ax.get_legend() is not None for ax in np.asarray(axes).flat)


@pytest.mark.parametrize('door', ['draw', 'typed'])
def test_R25_vector_1d_global_bins_match_full_selected_pandas_and_scalar(door, monkeypatch):
    """Real dEdxAI reproducer: [y,y2]:x × 1-D facet × global groups.

    The old draw() special bracket-vector delegation dropped group_by_scope,
    and pd.cut ran twice PER FACET rather than once on the Figure domain.
    Compare independent selected row membership and branch-wise scalar truth.
    """
    df = data()
    selected = df[df.keep]
    truth = pd.cut(selected.g, bins=3)
    from dfextensions.dfdraw.plots.profile import _format_interval_label
    labels = [str(_format_interval_label(interval)) for interval in truth.cat.categories]
    original_cut = pd.cut
    sizes = []

    def observed_cut(*args, **kwargs):
        sizes.append(len(args[0]))
        return original_cut(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd, 'cut', observed_cut)
        fig, axes, branches = render(
            df, door=door, expr='[y,y2]:x', facet_by='facet',
            group_by='g', group_by_bins=3, group_by_scope='global',
            selection='keep')
    assert sizes == [len(selected)], 'classification must run exactly once before faceting'
    assert len(branches) == 2
    assert all_legend_labels(fig) == [labels]
    assert all(ax.get_legend() is None for ax in np.asarray(axes).flat)
    for facet in ('east', 'west'):
        for branch_idx, column in enumerate(('y', 'y2')):
            payload = branches[branch_idx]['per_group'][facet]['profile_data']
            actual_counts = payload.groupby('group')['count'].sum()
            for category, display in zip(truth.cat.categories, labels):
                n = int(((selected.facet == facet) & (truth == category)).sum())
                assert int(actual_counts.get(display, 0)) == n, (facet, column, display)
            # A separate scalar render on exactly the same Figure-domain
            # selection provides an independent branch-value oracle.
            scalar_fig, _, scalar = render(
                df, door='typed', expr=f'{column}:x', facet_by='facet',
                group_by='g', group_by_bins=3, group_by_scope='global',
                selection='keep')
            pd.testing.assert_frame_equal(
                payload.reset_index(drop=True),
                scalar['per_group'][facet]['profile_data'].reset_index(drop=True),
                check_exact=True)
            plt.close(scalar_fig)


@pytest.mark.parametrize('door', ['draw', 'typed'])
def test_R26_vector_1d_global_quantile_scope_reuses_selected_boundaries(door, monkeypatch):
    df = data()
    selected = df[df.keep & (df.g > 5)]
    truth = pd.qcut(selected.g, q=3, duplicates='drop')
    from dfextensions.dfdraw.plots.profile import _format_interval_label
    labels = [str(_format_interval_label(interval)) for interval in truth.cat.categories]
    original_qcut = pd.qcut
    sizes = []

    def observed_qcut(*args, **kwargs):
        sizes.append(len(args[0]))
        return original_qcut(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd, 'qcut', observed_qcut)
        fig, axes, branches = render(
            df, door=door, expr='[y,y2]:x', facet_by='facet',
            group_by='g', group_by_quantiles=3,
            group_by_scope='global', selection='keep & (g > 5)')
    assert sizes == [len(selected)]
    assert all_legend_labels(fig) == [labels]
    for facet in ('east', 'west'):
        for branch in branches:
            counts = branch['per_group'][facet]['profile_data'].groupby('group')['count'].sum()
            for interval, label in zip(truth.cat.categories, labels):
                n = int(((selected.facet == facet) & (truth == interval)).sum())
                assert int(counts.get(label, 0)) == n


def test_R27_vector_1d_auto_and_facet_retain_legacy_local_semantics():
    df = data()
    _, _, auto = render(df, door='draw', expr='[y,y2]:x',
                        facet_by='facet', group_by='g', group_by_bins=3,
                        group_by_scope='auto')
    _, _, explicit = render(df, door='draw', expr='[y,y2]:x',
                            facet_by='facet', group_by='g', group_by_bins=3,
                            group_by_scope='facet')
    for branch_auto, branch_facet in zip(auto, explicit):
        for facet in ('east', 'west'):
            pd.testing.assert_frame_equal(
                branch_auto['per_group'][facet]['profile_data'],
                branch_facet['per_group'][facet]['profile_data'],
                check_exact=True)
