"""PHASE_13_96_DF Stage D: B × S over already-resolved vector branches.

Independent scalar-reference numerical oracle; deterministic identity and legend
without relying on the old implementation as the sole truth.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def frame():
    rng=np.random.default_rng(1396)
    x=np.tile(np.arange(6,dtype=float)+.5,72)
    return pd.DataFrame(dict(x=x, y0=2*x+rng.normal(0,1,len(x)),
        y1=-x+rng.normal(0,1.3,len(x)),
        s=np.tile(np.arange(72),6), w=np.where(np.arange(len(x))%2==0,1.,2.),
        v=np.linspace(.5,2.,len(x))))


def _close(fig):
    plt.close(fig)


def test_D01_expression_vector_stat_outer_product_draw_typed_and_numpy():
    data=frame()
    for door in ('draw','profile'):
        obj=DFDraw(data)
        call=getattr(obj,door)
        k={'type':'profile'} if door=='draw' else {}
        fig,ax,returned=call('[y0,y1]:x',bins=6,range=(0,6),
            stat_list=['mean','std','sem'],**k)
        try:
            assert len(returned)==2
            assert [s['statistic_branch']['expression'] for s in returned]==['y0:x','y1:x']
            assert [list(s['statistic_curves']) for s in returned] == [
                ['mean','std','sem'],['mean','std','sem']]
            for i,ycol in enumerate(('y0','y1')):
                for name in ('mean','std','sem'):
                    truth=[]
                    for j in range(6):
                        y=data.loc[data.x==j+.5,ycol].to_numpy()
                        truth.append({'mean':np.mean(y),'std':np.std(y,ddof=1),
                            'sem':np.std(y,ddof=1)/np.sqrt(len(y))}[name])
                    np.testing.assert_allclose(returned[i]['statistic_curves'][name], truth,
                        atol=1e-12,rtol=1e-12)
            labels=ax.get_legend_handles_labels()[1]
            assert 'mean' in labels and 'std' in labels and 'sem' in labels
            assert not any('y0 / mean' in label or 'y1 / std' in label for label in labels)
        finally: _close(fig)


def test_D02_selection_vector_stat_outer_preserves_selections():
    d=frame()
    obj=DFDraw(d)
    fig,ax,s=obj.profile('y0:x',selection_vector=['s<36','s>=36'],
        vector_compose='outer',bins=6,range=(0,6),stat_list=['mean','std'])
    try:
        assert len(s)==2
        for k,cut in enumerate(('s<36','s>=36')):
            part=d.query(cut)
            for name in ('mean','std'):
                expected=np.array([getattr(part.loc[part.x==j+.5,'y0'],name)(ddof=1)
                    if name=='std' else part.loc[part.x==j+.5,'y0'].mean()
                    for j in range(6)])
                np.testing.assert_allclose(s[k]['statistic_curves'][name],expected,
                    rtol=1e-12,atol=1e-12)
            assert s[k]['statistic_branch']['selection_index']==k
    finally: _close(fig)


def test_D03_weight_vector_stat_outer_matches_scalar_reference():
    d=frame()
    fig,ax,s=DFDraw(d).profile('y0:x',weights_vector=['w','v'],
            vector_compose='outer',bins=6,range=(0,6),stat_list=['mean','sem'])
    try:
        assert len(s)==2
        for i,col in enumerate(('w','v')):
            ref,_,truth=DFDraw(d).profile('y0:x',weights=col,bins=6,
                    range=(0,6),stat_list=['mean','sem'])
            _close(ref)
            for name in ('mean','sem'):
                np.testing.assert_allclose(s[i]['statistic_curves'][name],
                    truth['statistic_curves'][name],rtol=1e-12,atol=1e-12)
            assert s[i]['statistic_branch']['weights_index']==i
    finally: _close(fig)


def test_D04_mean_colors_follow_branch_not_statistic():
    fig,ax,s=DFDraw(frame()).profile('[y0,y1]:x',bins=6,range=(0,6),
        stat_list=['mean','std','sem'])
    try:
        # Per-branch color must be common for all three statistics.
        import matplotlib.colors as mcolors
        means=[c.lines[0] for c in ax.containers if c.get_label()=='mean']
        stds=[l for l in ax.lines if l.get_label()=='std']
        sems=[l for l in ax.lines if l.get_label()=='sem']
        assert len(means)==len(stds)==len(sems)==2
        for i in range(2):
            assert mcolors.to_rgba(means[i].get_color()) == mcolors.to_rgba(stds[i].get_color())
            assert mcolors.to_rgba(means[i].get_color()) == mcolors.to_rgba(sems[i].get_color())
        assert means[0].get_color()!=means[1].get_color()
    finally:_close(fig)


def test_D05_channel_collision_refused_before_creating_figure():
    d=frame()
    # Two independent branch dimensions can reserve marker; statistic needs it.
    with pytest.raises((NotImplementedError, ValueError), match='statistic|channel|marker'):
        DFDraw(d).profile('[y0,y1]:x',selection_vector=['s<36','s>=36'],
            weights_vector=['w','v'],vector_compose='outer',
            bins=6,range=(0,6),stat_list=['mean','std'])


def test_D06_legacy_vector_payload_unchanged_when_stat_list_none():
    d=frame()
    a,b=DFDraw(d).profile('[y0,y1]:x',bins=6,range=(0,6)),DFDraw(d).profile(
        '[y0,y1]:x',bins=6,range=(0,6),stat_list=None)
    try:
        assert len(a[2])==len(b[2])==2
        assert [list(x) for x in a[2]]==[list(x) for x in b[2]]
    finally:_close(a[0]);_close(b[0])


def test_D07_explicit_show_legend_false_and_marker_identity():
    fig,ax,s=DFDraw(frame()).draw('[y0,y1]:x',type='profile',bins=6,
        range=(0,6),stat_list=['mean','std'],show_legend=False)
    try: assert ax.get_legend() is None
    finally:_close(fig)


def test_D08_expression_and_selection_branch_cross_product():
    d=frame()
    fig,ax,s=DFDraw(d).profile('[y0,y1]:x', selection_vector=['s<36','s>=36'],
        selection_labels=['Low', 'High'],vector_compose='outer',
        bins=6,range=(0,6),stat_list=['mean','sem'])
    try:
        assert len(s)==4
        assert [(q['statistic_branch']['expression_index'],
                 q['statistic_branch']['selection_index']) for q in s] == [
                    (0,0),(0,1),(1,0),(1,1)]
        for branch in s:
            ident=branch['statistic_branch']
            sub=d.query('s<36' if ident['selection_index']==0 else 's>=36')
            arr=np.array([sub.loc[sub.x==j+.5,['y0','y1'][ident['expression_index']]].mean()
                          for j in range(6)])
            np.testing.assert_allclose(branch['statistic_curves']['mean'],arr,atol=1e-12)
        leg=ax.get_legend()
        assert leg is not None
        legend_labels=[t.get_text() for t in leg.get_texts()]
        assert len(legend_labels)==len(s)+2
        assert any('Low' in t for t in legend_labels)
        assert any('High' in t for t in legend_labels)
        assert len([t for t in legend_labels if t == 'mean'])==1
    finally:_close(fig)


def test_D09_quantile_band_composes_with_expression_vector_without_extra_mean():
    d=frame()
    fig,ax,s=DFDraw(d).profile('[y0,y1]:x',bins=6,range=(0,6),
        stat_list=['mean','quantiles'],quantiles=[.16,.5,.84],
        quantile_mode='band')
    try:
        assert len(s)==2
        assert len(ax.collections)>=2
        for i,expr in enumerate(('y0','y1')):
            q=s[i]['statistic_curves']['quantiles']
            assert list(s[i]['statistic_curves'])==['mean','quantiles']
            expected=np.array([np.quantile(d.loc[d.x==j+.5,expr],.16)
                               for j in range(6)])
            np.testing.assert_allclose(q['values'][.16],expected,atol=1e-12)
    finally:_close(fig)


def test_D10_discrete_quantile_channel_refuses_and_no_plot():
    d=frame()
    old=plt.get_fignums()
    with pytest.raises((NotImplementedError,ValueError),match='channel|cycle capacity'):
        DFDraw(d).profile('[y0,y1]:x',bins=6,range=(0,6),
            stat_list=['quantiles'],quantiles=[.1,.3,.5,.7,.9],
            quantile_mode='discrete')
    assert plt.get_fignums()==old


@pytest.mark.parametrize('stat', ['median','quantiles'])
def test_D11_weight_vector_unsupported_statistics_refuse_before_artists(stat):
    d=frame()
    old=plt.get_fignums()
    kwargs={'quantiles':[.16,.5,.84]} if stat=='quantiles' else {}
    with pytest.raises(NotImplementedError,match='weights'):
        DFDraw(d).profile('[y0,y1]:x', weights_vector=['w','v'],
            vector_compose='outer',bins=6,range=(0,6),
            stat_list=['mean',stat],**kwargs)
    assert old==plt.get_fignums()
