"""PHASE_13_96 Stage F RED/GREEN oracles.

Bounded contract: exactly two profile expressions x five Figure-global groups x
3x5 facets x stat_list=['std']; group=color, expression=marker, singleton
statistic consumes no visual channel.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.colors import to_rgba

from dfextensions.dfdraw.drawer import DFDraw
from dfextensions.dfdraw.plots.profile import _format_interval_label


def _fixture():
    rows = []
    # 3 row facets x 5 col facets x 5 groups x 4 x bins x 55 rows.
    # One sparse cell intentionally lacks the middle group; one x bin is below
    # min_entries after selection to exercise the non-plotted-bin contract.
    noise = np.linspace(-1.0, 1.0, 55)
    for fr in range(3):
        for fc in range(5):
            for gi in range(5):
                if fr == 1 and fc == 2 and gi == 2:
                    continue
                for xb in range(4):
                    for k, eps in enumerate(noise):
                        keep = True
                        if fr == 2 and fc == 4 and gi == 4 and xb == 3 and k >= 49:
                            keep = False
                        rows.append({
                            'x': xb + 0.5 + 1e-4 * eps,
                            'g': -2.0 + gi + 0.05 * eps,
                            'fr': fr + 0.2,
                            'fc': fc + 0.2,
                            'y0': 0.25 * xb + 0.05 * gi + 0.03 * fr + (1.0 + 0.05 * fc + 0.08 * gi) * eps,
                            'y1': -0.10 * xb + 0.08 * gi - 0.02 * fc + (1.4 + 0.03 * fr + 0.07 * gi) * eps,
                            'keep': keep,
                        })
    return pd.DataFrame(rows)


def _kwargs():
    return dict(
        selection='keep',
        group_by='g', group_by_bins=5, group_by_scope='global',
        facet_by=['fr', 'fc'], facet_by_bins=[3, 5],
        bins=4, range=(0.0, 4.0), min_entries=50,
        stat_list=['std'], legend='shared',
        group_style='color', vector_style='marker', linestyle='none',
        figure_margins=[0.12, 0.96, 0.89, 0.08],
        subplot_spacing=[0.2, 0.25], facet_labels='outer',
        auto_title=True, shared_xlabel='x', shared_ylabel='std residual',
        markersize=6,
    )


def _call(owner, typed=False, **overrides):
    kw = _kwargs(); kw.update(overrides)
    if typed:
        return owner.profile('[y0,y1]:x', **kw)
    return owner.draw('[y0,y1]:x', type='profile', **kw)


def _global_truth(df):
    selected = df.loc[df.keep].copy()
    cuts = pd.cut(selected['g'], bins=5)
    selected['global_group'] = cuts.map(_format_interval_label)
    fr_cut = pd.cut(selected['fr'], bins=3)
    fc_cut = pd.cut(selected['fc'], bins=5)
    selected['fr_key'] = fr_cut
    selected['fc_key'] = fc_cut
    return selected, list(cuts.cat.categories), list(fr_cut.cat.categories), list(fc_cut.cat.categories)


def _expected_std(values, xbin):
    ys = values.loc[(values.x >= xbin) & (values.x < xbin + 1), 'value'].to_numpy()
    return (np.std(ys, ddof=1) if len(ys) > 1 else np.nan), len(ys)


def test_F01_exact_stagef_query_renders_through_both_public_doors():
    d = _fixture()
    for typed in (False, True):
        fig, axes, stats = _call(DFDraw(d), typed=typed)
        try:
            assert axes.shape == (3, 5)
            assert len([k for k in stats if isinstance(k, tuple)]) == 15
        finally:
            plt.close(fig)


def test_F02_every_occupied_expression_group_facet_xbin_matches_pandas_numpy_truth():
    d = _fixture(); truth, _, _, _ = _global_truth(d)
    fig, axes, stats = _call(DFDraw(d), typed=False)
    try:
        for key, cell in stats.items():
            if not isinstance(key, tuple) or len(key) != 2 or not isinstance(cell, list):
                continue
            fr_key, fc_key = key
            source = truth[(truth.fr_key == fr_key) & (truth.fc_key == fc_key)]
            assert len(cell) == 2
            for expr_i, branch in enumerate(cell):
                assert branch['statistic_branch']['expression_index'] == expr_i
                assert list(branch['statistic_curves'])
                yname = ('y0', 'y1')[expr_i]
                for group_name, stat_bundle in branch['statistic_curves'].items():
                    gsrc = source[source.global_group == group_name].copy()
                    gsrc['value'] = gsrc[yname]
                    for xb in range(4):
                        expected, _ = _expected_std(gsrc, xb)
                        np.testing.assert_allclose(
                            stat_bundle['std'][xb], expected,
                            rtol=1e-12, atol=1e-12, equal_nan=True)
    finally:
        plt.close(fig)


def test_F03_sparse_groups_and_min_entries_do_not_invent_rendered_curves_or_points():
    d = _fixture(); fig, axes, stats = _call(DFDraw(d))
    try:
        # Explicit sparse cell: middle row facet × middle column facet lacks
        # the physical middle global group by construction.
        _, _, frcats, fccats = _global_truth(d)
        target_key = (frcats[1], fccats[2])
        cell = stats[target_key]
        assert isinstance(cell, list) and len(cell) == 2
        assert len(cell[0]['statistic_curves']) == 4
        assert len(cell[1]['statistic_curves']) == 4
        # One deliberately under-populated x-bin must be absent from the actual line.
        lengths = [len(line.get_xdata()) for ax in axes.flat for line in ax.lines]
        assert any(n == 3 for n in lengths)
        assert all(n <= 4 for n in lengths)
    finally:
        plt.close(fig)


def test_F04_global_group_colors_stay_physical_through_sparse_middle_group():
    d = _fixture(); fig, axes, stats = _call(DFDraw(d))
    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
        # first five entries are the global physical group colors
        group_labels = labels[:5]
        color_map = {name: to_rgba(handles[i].get_color()) for i, name in enumerate(group_labels)}
        # Check the deliberately sparse middle/middle facet explicitly:
        # physical group 2 is absent, but groups 3/4 must retain their global
        # color indexes rather than shifting left.
        _, _, frcats, fccats = _global_truth(d)
        cell = stats[(frcats[1], fccats[2])]
        present = list(cell[0]['statistic_curves'])
        assert len(present) == 4
        assert group_labels[2] not in [str(x) for x in present]
        ax = axes[1, 2]
        assert len(ax.lines) == 2 * len(present)
        for block in (ax.lines[:len(present)], ax.lines[len(present):2*len(present)]):
            for line, group_name in zip(block, present):
                np.testing.assert_allclose(to_rgba(line.get_color()), color_map[str(group_name)])
    finally:
        plt.close(fig)


def test_F05_two_expression_markers_own_identity_and_all_curves_are_marker_only_std():
    d = _fixture(); fig, axes, stats = _call(DFDraw(d))
    try:
        markers_by_branch = [set(), set()]
        for ax in axes.flat:
            # full cells: 5 group lines per expression; sparse cell may have 4
            n = len(ax.lines) // 2
            if n == 0:
                continue
            for line in ax.lines[:n]:
                markers_by_branch[0].add(line.get_marker())
                assert line.get_linestyle() in ('None', 'none', '')
            for line in ax.lines[n:2*n]:
                markers_by_branch[1].add(line.get_marker())
                assert line.get_linestyle() in ('None', 'none', '')
        assert len(markers_by_branch[0]) == 1
        assert len(markers_by_branch[1]) == 1
        assert markers_by_branch[0] != markers_by_branch[1]
        for cell in stats.values():
            if isinstance(cell, list):
                for branch in cell:
                    assert list(branch['statistic_curves'].values())[0].keys() == {'std'}
    finally:
        plt.close(fig)


def test_F06_one_factored_figure_legend_truthful_and_nonoverlapping():
    d = _fixture(); fig, axes, stats = _call(DFDraw(d))
    try:
        assert len(fig.legends) == 1
        assert all(not any(isinstance(a, Legend) for a in ax.get_children()) for ax in axes.flat)
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        assert len(labels) == 7
        assert labels[-2:] == ['y0', 'y1']
        handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
        assert handles[-2].get_marker() != handles[-1].get_marker()
        assert handles[-2].get_linestyle() in ('None', 'none', '')
        assert handles[-1].get_linestyle() in ('None', 'none', '')
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        lb = legend.get_window_extent(renderer)
        if fig._suptitle is not None:
            assert not lb.overlaps(fig._suptitle.get_window_extent(renderer))
        assert all(not lb.overlaps(ax.get_tightbbox(renderer)) for ax in axes.flat)
        # Outer facet headings are Figure-owned text artists in dense-layout mode.
        # The shared x/y labels are checked too; there must be no decoration overlap.
        assert all(
            not lb.overlaps(text.get_window_extent(renderer))
            for text in fig.texts
            if text.get_visible() and text.get_text() and text is not fig._suptitle
        )
    finally:
        plt.close(fig)


def test_F07_public_result_shape_remains_facet_keyed_with_ordered_expression_branches():
    d = _fixture(); fig, axes, stats = _call(DFDraw(d))
    try:
        cell_items = [(k, v) for k, v in stats.items() if isinstance(k, tuple)]
        assert len(cell_items) == 15
        for key, cell in cell_items:
            assert isinstance(cell, list)
            assert len(cell) == 2
            assert [b['statistic_branch']['expression_index'] for b in cell] == [0, 1]
            assert [list(b['statistic_curves'].keys()) for b in cell]
            for b in cell:
                assert all(list(bundle) == ['std'] for bundle in b['statistic_curves'].values())
    finally:
        plt.close(fig)


def test_F08_stat_list_none_preserves_existing_nd_vector_group_facet_route():
    d = _fixture(); kw = _kwargs(); kw.pop('stat_list')
    f0, a0, s0 = DFDraw(d).draw('[y0,y1]:x', type='profile', **kw)
    f1, a1, s1 = DFDraw(d).profile('[y0,y1]:x', **kw)
    try:
        assert a0.shape == a1.shape == (3, 5)
        assert list(s0.keys()) == list(s1.keys())
        for key in s0:
            assert type(s0[key]) is type(s1[key])
    finally:
        plt.close(f0); plt.close(f1)


@pytest.mark.parametrize('override', [
    {'marker': 'o'},
    {'color': 'red'},
    {'stat_list': ['mean', 'std']},
    {'selection_vector': ['y0>0', 'y0<=0']},
    {'weights_vector': ['x+1', 'x+2']},
    {'weights': 'x+1'},
    {'normalize': 'ratio'},
])
def test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation(override):
    d = _fixture(); owner = DFDraw(d)
    before = set(plt.get_fignums())
    with pytest.raises((NotImplementedError, ValueError)):
        _call(owner, **override)
    assert set(plt.get_fignums()) == before


def test_F09b_private_stagef_coordinator_token_is_not_a_public_escape_hatch():
    d = _fixture(); owner = DFDraw(d)
    before = set(plt.get_fignums())
    with pytest.raises(TypeError, match='internal dfdraw parameter'):
        _call(owner, typed=True, _stagef_singleton_std_nd=True)
    assert set(plt.get_fignums()) == before


def test_F10_global_group_resolution_and_figure_layout_have_single_owners(monkeypatch):
    import dfextensions.dfdraw.drawer as owner_mod
    from matplotlib.figure import Figure
    d = _fixture()
    group_calls = []
    layout_calls = []
    orig_group = owner_mod._scope_group_domain
    orig_layout = Figure.tight_layout
    def tracked_group(*args, **kwargs):
        group_calls.append(1)
        return orig_group(*args, **kwargs)
    def tracked_layout(self, *args, **kwargs):
        layout_calls.append(1)
        return orig_layout(self, *args, **kwargs)
    monkeypatch.setattr(owner_mod, '_scope_group_domain', tracked_group)
    monkeypatch.setattr(Figure, 'tight_layout', tracked_layout)
    fig, axes, stats = _call(DFDraw(d))
    try:
        assert len(group_calls) == 1
        assert len(layout_calls) == 1
        assert axes.shape == (3, 5)
    finally:
        plt.close(fig)
