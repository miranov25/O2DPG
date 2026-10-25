"""PHASE_13_96_DF Stage E: group/facet semantic and numerical truth."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dfextensions.dfdraw.drawer import DFDraw


def fixture():
    rng = np.random.default_rng(1396)
    n = 3600
    x = rng.uniform(0, 12, n)
    g = np.where(np.arange(n) % 3 == 0, 'A', np.where(np.arange(n) % 3 == 1, 'B', 'C'))
    f = (np.arange(n) // 3) % 3
    w = rng.uniform(.5, 2, n)
    y = .4 * x + np.where(g == 'A', 0, np.where(g == 'B', 2, 4)) + .1 * f + rng.normal(0, 0.7, n)
    return pd.DataFrame({'x': x, 'y': y, 'g': g, 'f': f, 'f2': np.arange(n) % 2, 'w': w})


def independent(data, labels, xlim=(0, 12), bins=6):
    edges = np.linspace(*xlim, bins + 1)
    result = {}
    for group in labels:
        group_data = data[data.g == group]
        values = {'mean': [], 'std': [], 'sem': []}
        for j in range(bins):
            mask = ((group_data.x >= edges[j]) &
                    ((group_data.x < edges[j+1]) if j < bins-1 else (group_data.x <= edges[j+1])))
            ys = group_data.loc[mask, 'y'].to_numpy()
            values['mean'].append(np.mean(ys) if len(ys) else np.nan)
            values['std'].append(np.std(ys, ddof=1) if len(ys)>1 else np.nan)
            values['sem'].append(np.std(ys, ddof=1) / np.sqrt(len(ys)) if len(ys)>1 else np.nan)
        result[group] = {k: np.array(v) for k, v in values.items()}
    return result


def test_E1_grouped_identity_and_direct_numpy_truth():
    d = fixture()
    fig, ax, stats = DFDraw(d).profile('y:x', group_by='g', bins=6,
               range=(0, 12), stat_list=['mean', 'std', 'sem'], error='sem')
    try:
        expected = independent(d, ['A','B','C'])
        assert list(stats['statistic_curves']) == ['A','B','C']
        for g in expected:
            for name in ['mean','std','sem']:
                np.testing.assert_allclose(stats['statistic_curves'][g][name], expected[g][name], rtol=1e-11, atol=1e-11)
        assert len(ax.lines) >= 9
        assert ax.get_legend() is not None
        assert len(ax.get_legend().texts) == 6 # groups + statistics, not 9
    finally: plt.close(fig)


def test_E2_grouped_draw_typed_equivalence():
    d = fixture(); kw = dict(group_by='g', bins=6, range=(0,12), stat_list=['mean','std'])
    f1, a1, s1 = DFDraw(d).profile('y:x', **kw)
    f2, a2, s2 = DFDraw(d).draw('y:x', type='profile', **kw)
    try:
        for group in s1['statistic_curves']:
            for stat in ('mean','std'):
                np.testing.assert_array_equal(s1['statistic_curves'][group][stat], s2['statistic_curves'][group][stat])
    finally: plt.close(f1);plt.close(f2)


def test_E3_one_dimensional_facet_stats_truth():
    d = fixture(); fig, axes, stats = DFDraw(d).profile('y:x', facet_by='f', bins=6,
        range=(0,12), stat_list=['mean','std','sem'], min_entries=2)
    try:
        assert len(axes) >= 3
        for facet in (0,1,2):
            cell = stats['per_group'][str(facet)] if 'per_group' in stats else None
            assert cell is not None, (list(stats), stats)
            data = d[d.f == facet]
            truth = independent(data.assign(g='A'), ['A'])['A']
            for k in ('mean','std','sem'):
                np.testing.assert_allclose(cell['statistic_curves'][k], truth[k], rtol=1e-10, atol=1e-10)
    finally: plt.close(fig)


def test_E4_refusals_before_figure_allocation():
    d = fixture(); owner=DFDraw(d)
    for kw in ({'group_by':'g','stat_list':['median']}, {'facet_by':'f','stat_list':['median']},
               {'group_by':'g','stat_list':['quantiles'],'quantiles':[.2,.8]},
               {'group_by':'g','stat_list':['mean'],'selection_vector':['f==0','f==1']}):
        n_before=len(plt.get_fignums())
        with pytest.raises(NotImplementedError): owner.profile('y:x', **kw)
        assert len(plt.get_fignums())==n_before


def test_E5_grouped_weighted_numpy_truth():
    d=fixture(); fig,ax,s=DFDraw(d).profile('y:x', group_by='g', weights='w',
           bins=6,range=(0,12),stat_list=['mean','std','sem'])
    try:
        for g in ('A','B','C'):
            sub=d[d.g==g]
            for j in range(6):
                cur=sub[(sub.x>=j*2)&(sub.x<(j+1)*2)]
                y,w=cur.y.to_numpy(),cur.w.to_numpy()
                mean=np.sum(w*y)/sum(w)
                std=np.sqrt(np.sum(w*(y-mean)**2)/sum(w))
                neff=sum(w)**2/np.sum(w**2)
                for k,value in [('mean',mean),('std',std),('sem',std/np.sqrt(neff))]:
                    np.testing.assert_allclose(s['statistic_curves'][g][k][j], value, rtol=1e-12, atol=1e-12)
    finally:plt.close(fig)


def test_E6_global_group_bins_1d_facets_semantic_legend():
    d=fixture(); fig,axs,s=DFDraw(d).draw('y:x',type='profile',
           group_by='w',group_by_bins=4,group_by_scope='global', facet_by='f',
           bins=6,range=(0,12), stat_list=['mean','std','sem'])
    try:
        assert len(fig.legends)==1
        assert all(ax.get_legend() is None for ax in axs[:3])
        names=[t.get_text() for t in fig.legends[0].get_texts()]
        assert names[-3:]==['mean','std','sem']
        assert len(names)==7
        allkeys=[]
        for k in ('0','1','2'):
            allkeys.append(set(s['per_group'][k]['statistic_curves']))
        assert all(x.issubset(names[:4]) for x in allkeys)
        assert len(set(names[:4]))==4
    finally:plt.close(fig)


def test_E7_global_group_quantiles_2d_facets_one_figure_legend():
    d=fixture(); fig,axs,s=DFDraw(d).draw('y:x',type='profile',
        group_by='w',group_by_quantiles=3,group_by_scope='global',
        facet_by=['f','f2'],bins=6,range=(0,12),
        stat_list=['mean','std','sem'],facet_labels='outer')
    try:
        assert axs.shape==(3,2)
        assert len(fig.legends)==1
        names=[t.get_text() for t in fig.legends[0].get_texts()]
        assert len(names)==6 and names[-3:]==['mean','std','sem']
        assert all(ax.get_legend() is None for ax in axs.flat)
        assert all('statistic_curves' in val for val in s.values() if isinstance(val,dict) and val.get('n',0)>0)
    finally:plt.close(fig)


def test_E8_ungrouped_facets_single_figure_stat_legend():
    d=fixture()
    for facet in ('f',['f','f2']):
        fig,axs,s=DFDraw(d).profile('y:x',facet_by=facet,bins=6,
            range=(0,12),stat_list=['mean','sem'])
        try:
            assert len(fig.legends)==1
            assert [z.get_text() for z in fig.legends[0].get_texts()]==['mean','sem']
            assert all(a.get_legend() is None for a in np.ravel(axs))
        finally:plt.close(fig)


def test_E9_legacy_grouped_and_faceted_payload_when_none():
    d=fixture()
    for kw in (dict(group_by='g'),dict(facet_by='f'),dict(facet_by=['f','f2'])):
        f0,a0,s0=DFDraw(d).profile('y:x',bins=6,range=(0,12),**kw)
        f1,a1,s1=DFDraw(d).profile('y:x',bins=6,range=(0,12),stat_list=None,**kw)
        try:
            assert set(s0)==set(s1)
            assert 'statistic_curves' not in s0
        finally:plt.close(f0);plt.close(f1)


def test_E10_grouped_count_compute_once(monkeypatch):
    import dfextensions.dfdraw.plots.profile as owner
    d=fixture()
    counts=[]; orig=owner._compute_profile
    def tracked(*args,**kwargs):
        counts.append(1)
        return orig(*args,**kwargs)
    monkeypatch.setattr(owner,'_compute_profile',tracked)
    fig,ax,s=DFDraw(d).profile('y:x',group_by='g',bins=6,range=(0,12),
                            stat_list=['mean','std','sem'])
    try:
        assert len(counts)==3 # 3 groups, NOT 9 curves
    finally:plt.close(fig)


def test_E11_global_group_sparse_cell_colors_stable():
    d=fixture()
    d=d[~((d.f==0)&(d.w>.95))].copy() # first facet lacks highest global groups
    fig,axes,stats=DFDraw(d).profile('y:x',group_by='w',group_by_bins=3,
         group_by_scope='global',facet_by='f',bins=6,range=(0,12),
         stat_list=['mean','std'])
    try:
        assert len(fig.legends)==1
        all_labels=[x.get_text() for x in fig.legends[0].get_texts()]
        assert all_labels[-2:]==['mean','std']
        # Verify actual mean curves use their Figure-global legend colors
        proxies=fig.legends[0].legend_handles if hasattr(fig.legends[0],'legend_handles') else fig.legends[0].legendHandles
        colmap={key:proxies[i].get_color() for i,key in enumerate(all_labels[:-2])}
        from matplotlib.colors import to_rgba
        for key,ax in zip(stats['groups'],axes):
            cell=stats['per_group'][str(key)]['statistic_curves']
            meanlines=[line for line in ax.lines if line.get_marker()=='o']
            assert len(meanlines)==len(cell)
            for idx,g in enumerate(cell):
                np.testing.assert_allclose(to_rgba(meanlines[idx].get_color()),to_rgba(colmap[str(g)]))
    finally:plt.close(fig)


def test_E12_grouped_explicit_legend_false():
    d=fixture();f,a,s=DFDraw(d).profile('y:x',group_by='w',group_by_bins=3,
        group_by_scope='global',facet_by='f',stat_list=['mean','sem'],legend=False,
        bins=6,range=(0,12))
    try:
        assert not f.legends
        assert all(ax.get_legend() is None for ax in a)
    finally:plt.close(f)


def test_E13_two_dimensional_global_group_statistics_match_raw_pandas():
    from dfextensions.dfdraw.plots.profile import _format_interval_label
    d=fixture()
    # The independent truth partitions the complete Figure domain BEFORE
    # the two facet coordinates; this catches accidental per-cell re-binning.
    cut=pd.cut(d['w'],bins=3)
    data=d.assign(global_key=cut.map(_format_interval_label))
    fig,axs,s=DFDraw(d).profile('y:x',group_by='w',group_by_bins=3,
        group_by_scope='global',facet_by=['f','f2'],bins=6,range=(0,12),
        stat_list=['mean','std','sem'],min_entries=2,return_data=True)
    try:
        for (fa,fb),stats in s.items():
            source=data[(data.f==fa)&(data.f2==fb)]
            rendered=stats['statistic_curves']
            for group_name,curves in rendered.items():
                values=source[source.global_key==group_name]
                for i in range(6):
                    ys=values.loc[(values.x >= i*2)&(values.x<(i+1)*2),'y'].to_numpy()
                    assert len(ys)>2
                    expected={'mean':np.mean(ys), 'std':np.std(ys,ddof=1),
                              'sem':np.std(ys,ddof=1)/np.sqrt(len(ys))}
                    for name,actual in curves.items():
                        np.testing.assert_allclose(actual[i],expected[name],rtol=1e-12,atol=1e-12)
    finally:plt.close(fig)


def test_E14_unsupported_3d_facet_and_group_color_refuse():
    d=fixture();owner=DFDraw(d)
    for bad in (dict(facet_by=['f','f2','g']),dict(group_by='g',facet_by='f',color='red')):
        before=set(plt.get_fignums())
        with pytest.raises(NotImplementedError): owner.profile('y:x',stat_list=['mean','std'],**bad)
        assert set(plt.get_fignums())==before


# PHASE_13_96 Stage E Fix1: public-real-use-case regression.
# With no facet, explicit Figure-global grouping has only one domain and must
# agree with historical auto grouping. The old PHASE_13_95 guard rejected this
# before the already-valid grouped statistical owner was reached.
def test_E15_draw_nonfaceted_explicit_global_bins_against_pandas():
    from dfextensions.dfdraw.plots.profile import _format_interval_label
    d = fixture()
    kw = dict(type='profile', selection='f != 0', group_by='w',
              group_by_bins=4, group_by_scope='global', bins=6,
              range=(0, 12), stat_list=['mean', 'std', 'sem'],
              min_entries=2)
    f, ax, result = DFDraw(d).draw('y:x', **kw)
    try:
        f_auto, a_auto, result_auto = DFDraw(d).draw(
            'y:x', **{**kw, 'group_by_scope': 'auto'})
        try:
            assert list(result['statistic_curves']) == list(result_auto['statistic_curves'])
            selected = d.loc[d.f != 0].copy()
            cuts = pd.cut(selected['w'], bins=4)
            for category in cuts.cat.categories:
                group_name = str(_format_interval_label(category))
                assert group_name in result['statistic_curves']
                sub = selected.loc[cuts == category]
                truth = independent(sub.assign(g='A'), ['A'])['A']
                for name in ('mean', 'std', 'sem'):
                    np.testing.assert_allclose(result['statistic_curves'][group_name][name],
                                               truth[name], rtol=1e-11, atol=1e-11)
                    np.testing.assert_array_equal(result['statistic_curves'][group_name][name],
                                                  result_auto['statistic_curves'][group_name][name])
            legend = ax.get_legend()
            assert legend is not None
            assert len(legend.texts) == 4 + 3
            assert not f.legends  # historical single-Axes factored legend
        finally:
            plt.close(f_auto)
    finally:
        plt.close(f)


def test_E16_typed_nonfaceted_explicit_global_quantiles_equal_auto():
    d = fixture()
    kw = dict(group_by='w', group_by_quantiles=4,
              bins=6, range=(0,12), stat_list=['mean', 'std', 'sem'])
    f, ax, got = DFDraw(d).profile('y:x', group_by_scope='global', **kw)
    f0, ax0, reference = DFDraw(d).profile('y:x', group_by_scope='auto', **kw)
    try:
        assert list(got['statistic_curves']) == list(reference['statistic_curves'])
        for group, curves in got['statistic_curves'].items():
            for name, values in curves.items():
                np.testing.assert_array_equal(values, reference['statistic_curves'][group][name])
        assert len(ax.get_legend().texts) == 7
    finally:
        plt.close(f); plt.close(f0)


def test_E17_nonfaceted_global_without_stat_list_preserves_legacy():
    d=fixture()
    kw=dict(group_by='w', group_by_bins=4, bins=6, range=(0,12))
    f, ax, result=DFDraw(d).draw('y:x',type='profile',group_by_scope='global',**kw)
    f0, ax0, ref=DFDraw(d).draw('y:x',type='profile',group_by_scope='auto',**kw)
    try:
        assert result.keys()==ref.keys()
        assert not f.legends
        assert len(ax.lines)==len(ax0.lines)
        for curve,baseline in zip(ax.lines,ax0.lines):
            np.testing.assert_array_equal(curve.get_xdata(),baseline.get_xdata())
            np.testing.assert_array_equal(curve.get_ydata(),baseline.get_ydata())
    finally:
        plt.close(f);plt.close(f0)


def test_E18_nonfaceted_scope_refusal_boundaries_before_figure_allocation():
    d=fixture();owner=DFDraw(d)
    bad=(
        dict(group_by_scope='global',stat_list=['mean']),  # no group_by
        dict(group_by='w',group_by_bins=4,group_by_scope='facet',stat_list=['mean']),
        dict(group_by='w',group_by_bins=4,group_by_scope='global',top_k=2,stat_list=['mean']),
        dict(group_by='w',group_by_bins=4,group_by_scope='global',facet=True,stat_list=['mean']),
    )
    for kwargs in bad:
        before=set(plt.get_fignums())
        with pytest.raises((ValueError,NotImplementedError)):
            owner.draw('y:x', type='profile', **kwargs)
        assert set(plt.get_fignums())==before
