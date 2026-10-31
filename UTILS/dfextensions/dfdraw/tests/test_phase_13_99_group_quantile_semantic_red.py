"""PHASE_13_99 v4 RED: grouped facet semantic legend contract."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw.drawer import DFDraw


def _frame():
    rng = np.random.default_rng(1399004)
    n = 1800
    return pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'f': np.repeat(np.arange(6), n // 6),
        'h': np.tile(np.repeat([0, 1], n // 12), 6),
        'g': np.tile(np.repeat(['A', 'B', 'C'], 100), 6),
        'y0': rng.normal(size=n),
        'y1': 1.5 + rng.normal(size=n),
    })


@pytest.mark.parametrize('facet_by,facet_bins', [
    ('f', 6), (['f', 'h'], [6, 2]),
])
def test_FP03_group_vector_quantile_legend_complete(facet_by, facet_bins):
    fig, axes, _ = DFDraw(_frame()).draw(
        '[y0,y1]:x', type='profile', bins=6,
        facet_by=facet_by, facet_by_bins=facet_bins, ncols=3,
        group_by='g', group_by_scope='global',
        quantiles=[.15, .5, .9], quantile_mode='discrete',
        legend='shared', auto_title=False,
    )
    try:
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in np.asarray(axes).flat)
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        for required in ('y0', 'y1', 'q=15%', 'median', 'q=90%', 'A', 'B', 'C'):
            assert labels.count(required) == 1, (facet_by, required, labels)
    finally:
        plt.close(fig)


def test_FP03_group_quantile_legend_sparse_groups():
    frame = _frame()
    frame = frame[~((frame['f'] == 0) & (frame['g'] == 'B'))]
    fig, axes, _ = DFDraw(frame).draw(
        '[y0,y1]:x', type='profile', bins=6,
        facet_by='f', facet_by_bins=6, ncols=3,
        group_by='g', group_by_scope='global',
        quantiles=[.15, .5, .9], quantile_mode='discrete',
        legend='shared', auto_title=False,
    )
    try:
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        for required in ('y0','y1','q=15%','median','q=90%','A','B','C'):
            assert labels.count(required) == 1, (required,labels)
    finally:
        plt.close(fig)

@pytest.mark.parametrize('facet_by,facet_bins', [
    ('f', 6), (['f', 'h'], [6, 2]),
])
def test_FP03_style_keys_are_rendered_artist_styles(facet_by, facet_bins):
    fig, axes, _ = DFDraw(_frame()).draw(
        '[y0,y1]:x', type='profile', bins=6, range=(0., 1.),
        facet_by=facet_by, facet_by_bins=facet_bins, ncols=3,
        group_by='g', group_by_scope='global',
        quantiles=[.15, .5, .9], quantile_mode='discrete',
        legend='shared', auto_title=False,
    )
    try:
        semantic = [(ln, ln._dfdraw_facet_semantic)
                    for ax in np.asarray(axes).flat for ln in ax.lines
                    if hasattr(ln, '_dfdraw_facet_semantic')]
        for exp in ('y0', 'y1'):
            assert any(meta['kind']=='expression' and meta['expression']==exp
                       for ln,meta in semantic)
        sample = {q: next(ln for ln, meta in semantic
                          if meta['kind']=='quantile'
                          and meta['expression']=='y0'
                          and meta['group']=='A'
                          and meta['quantile']==q)
                  for q in (.15,.5,.9)}
        # Each q has a distinct rendered linestyle; a legend with labels but
        # three identical dashed source lines is not a truthful q mapping.
        assert len({line.get_linestyle() for line in sample.values()}) == 3
        leg = fig.legends[0]
        handles = getattr(leg, 'legend_handles', None)
        assert handles is not None
        labelled = dict(zip([t.get_text() for t in leg.get_texts()], handles))
        for q, name in ((.15,'q=15%'), (.5,'median'), (.9,'q=90%')):
            assert labelled[name].get_linestyle() == sample[q].get_linestyle()
        for exp in ('y0','y1'):
            original = next(ln for ln, meta in semantic
                            if meta['kind']=='expression' and meta['expression']==exp)
            assert labelled[exp].get_marker() == original.get_marker()
    finally:
        plt.close(fig)


def test_FP07_grouped_quantiles_agree_with_independent_raw_percentiles():
    frame = _frame()
    fig, axes, _ = DFDraw(frame).draw(
        '[y0,y1]:x', type='profile', bins=6, range=(0.,1.),
        facet_by='f', facet_by_bins=6, ncols=3,
        group_by='g', group_by_scope='global',
        quantiles=[.15,.5,.9], quantile_mode='discrete',
        legend='shared', auto_title=False)
    try:
        panel = np.asarray(axes).flat[0]
        subset = frame[(frame['f']==0)&(frame['g']=='A')]
        edges = np.linspace(0.,1.,7)
        centers = (edges[:-1]+edges[1:])/2
        for expr in ('y0','y1'):
            for q in (.15,.5,.9):
                line = next(ln for ln in panel.lines
                            if getattr(ln,'_dfdraw_facet_semantic',{}).get('kind')=='quantile'
                            and ln._dfdraw_facet_semantic['expression']==expr
                            and ln._dfdraw_facet_semantic['group']=='A'
                            and ln._dfdraw_facet_semantic['quantile']==q)
                expected = []
                for i in range(6):
                    mask = ((subset.x>=edges[i]) &
                            ((subset.x<edges[i+1]) if i<5 else (subset.x<=edges[i+1])))
                    values = subset.loc[mask, expr].to_numpy()
                    expected.append(np.nanpercentile(values, q*100) if len(values) else np.nan)
                np.testing.assert_allclose(line.get_xdata(), centers, rtol=0, atol=1e-12)
                np.testing.assert_allclose(line.get_ydata(), expected, rtol=0, atol=1e-12)
    finally:
        plt.close(fig)


def test_FP06_grouped_custom_legend_placement_remains_authoritative():
    fig, axes, _ = DFDraw(_frame()).draw(
        '[y0,y1]:x', type='profile', bins=6,
        facet_by='f', facet_by_bins=6, ncols=3,
        group_by='g', group_by_scope='global',
        quantiles=[.15,.5,.9], quantile_mode='discrete',
        legend={'mode':'shared','loc':'lower center','ncol':2,
                'title':'Analyst legend'}, auto_title=False)
    try:
        assert len(fig.legends) == 1
        assert fig.legends[0]._loc == 8
        assert fig.legends[0]._ncols == 2
        assert fig.legends[0].get_title().get_text() == 'Analyst legend'
    finally:
        plt.close(fig)


def test_FP03_nonfaceted_group_quantile_linestyles_not_changed():
    fig, axes, _ = DFDraw(_frame()).draw(
        '[y0,y1]:x', type='profile', bins=6,
        group_by='g', quantiles=[.15,.5,.9],
        quantile_mode='discrete', legend='none', auto_title=False)
    try:
        q_lines = [ln for ax in fig.axes for ln in ax.lines
                   if getattr(ln,'_dfdraw_facet_semantic',{}).get('kind')=='quantile']
        assert q_lines
        assert {line.get_linestyle() for line in q_lines} == {'--'}
    finally:
        plt.close(fig)
