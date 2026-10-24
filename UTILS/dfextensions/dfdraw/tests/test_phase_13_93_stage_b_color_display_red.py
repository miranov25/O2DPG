"""PHASE_13_93_DF Stage B: figure-owned categorical scatter presentation.

The machine oracle inspects actual Matplotlib artists, not only labels.
Numeric Side is intentional: the PHASE_13_92 categorical string + errorbar
refusal remains separate and must not be weakened here.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest
from dfextensions.dfdraw import DFDraw


def _frame(*, singleton=False):
    data=[]
    for i, r in enumerate(['Rate','Base']):
        for j, c in enumerate(['B','A']):
            for side in (0, 1):
                if singleton and r == 'Rate' and c == 'B' and side == 0:
                    continue
                for region in range(2):
                    data.append((r,c,side,region, 0.25 + i*.04+j*.03+side*.10+region*.02, .01))
    df=pd.DataFrame(data,columns=['row','col','side','region','coefficient','error'])
    df['row']=pd.Categorical(df['row'],categories=['Base','Rate'],ordered=True)
    df['col']=pd.Categorical(df['col'],categories=['A','B'],ordered=True)
    return df


def _draw(df, **kw):
    opts=dict(type='scatter', color='side', marker='side > 0.5',yerr='error',
              facet_by=['row','col'],facet_labels='outer',share_y='none',
              shared_xlabel='Region',shared_ylabel='Coefficient',auto_title=True)
    opts.update(kw)
    return DFDraw(df).draw('coefficient:region',**opts)


def _point_colors(ax):
    colors = []
    for collection in ax.collections:
        if getattr(collection, 'get_array', lambda:None)() is not None:
            arr = np.asarray(collection.get_array(),dtype=float)
            collection.update_scalarmappable()
            rgba = np.asarray(collection.get_facecolors())
            colors += list(zip(arr,rgba))
    return colors


def test_b1_omitted_color_display_keeps_original_colorbars():
    fig, axes, _=_draw(_frame())
    try:
        assert len(fig.legends)==0
        assert len(fig.axes)>axes.size  # normal per-cell colorbars remain
    finally:plt.close(fig)


@pytest.mark.parametrize('mode',['auto','legend'])
def test_b2_b3_binary_numeric_side_one_figure_legend(mode):
    fig, axes,_=_draw(_frame(),color_display=mode)
    try:
        assert axes.shape==(2,2)
        assert len(fig.axes)==4  # no per-panel colorbar axes
        assert len(fig.legends)==1
        assert [t.get_text() for t in fig.legends[0].get_texts()]==['side=0','side=1']
        assert len(fig.legends[0].legend_handles)==2
    finally:plt.close(fig)


def test_b4_explicit_colorbar_is_not_silently_changed_to_legend():
    fig,axes,_=_draw(_frame(),color_display='colorbar')
    try:
        assert len(fig.legends)==0
        assert len(fig.axes)>axes.size
    finally:plt.close(fig)


def test_b5_same_side_same_rgba_in_all_facets_even_singleton():
    fig,axes,_=_draw(_frame(singleton=True),color_display='legend')
    try:
        colors_by_side={0:[],1:[]}
        for ax in axes.flat:
            for side, rgba in _point_colors(ax):
                colors_by_side[int(side)].append(rgba)
        for vals in colors_by_side.values():
            assert vals
            assert np.allclose(vals,np.repeat([vals[0]],len(vals),axis=0))
        assert not np.allclose(colors_by_side[0][0], colors_by_side[1][0])
    finally:plt.close(fig)


def test_b6_no_repeated_axes_legend():
    fig,axes,_=_draw(_frame(),color_display='legend')
    try:
        assert len(fig.legends)==1
        assert all(ax.get_legend() is None for ax in axes.flat)
    finally:plt.close(fig)


def test_b8_matching_marker_partition_produces_combined_legend():
    fig,axes,_=_draw(_frame(),color_display='legend')
    try:
        legend=fig.legends[0]
        marker_codes=[handle.get_marker() for handle in legend.legend_handles]
        assert marker_codes==['o','s']
        assert len(legend.legend_handles)==2
    finally:plt.close(fig)


def test_b8b_non_equivalent_marker_partition_refuses():
    df=_frame()
    with pytest.raises(ValueError,match='marker.*partition|partition.*marker'):
        _draw(df,marker='region > 0.5',color_display='legend')


def test_b9_uncertainties_are_retained_with_two_marker_classes():
    fig,axes,_=_draw(_frame(),color_display='legend')
    try:
        for ax in axes.flat:
            assert len(ax.containers)==1  # errorbars remain one owner
            assert len(ax.collections)>=3 # two point collections + bar collection
    finally:plt.close(fig)


def test_b10_auto_threshold_controls_discrete_heuristic():
    fig,axes,_=_draw(_frame(),color_display='auto',max_legend_categories=1)
    try:
        assert len(fig.legends)==0
        assert len(fig.axes)>axes.size
    finally:plt.close(fig)


def test_b10b_explicit_legend_limit_refuses():
    with pytest.raises(ValueError,match='max_legend_categories'):
        _draw(_frame(),color_display='legend',max_legend_categories=1)


def test_b11_bad_mode_rejected_before_plotting():
    with pytest.raises(ValueError,match='color_display'):
        _draw(_frame(),color_display='surprise')


def test_b12_invalid_threshold_rejected():
    with pytest.raises(ValueError,match='max_legend_categories'):
        _draw(_frame(),color_display='auto',max_legend_categories=0)


def test_b13_real_grid_40_cells_one_legend():
    df=_frame()
    df['row']=pd.Categorical(df['row'],categories=['Base','Rate'],ordered=True)
    # Extend fixture without changing the source point/error contract.
    pieces=[]
    for i, row in enumerate(['Base','Rate','TPC','Unfm','Ft0']):
        for j, col in enumerate(['Constant','Tgl','Qpt','TglQpt','MdEdx','TglMdEdx','MdEdxQpt','TglMdEdxQpt']):
            part=df.iloc[:4].copy()
            part['row']=row
            part['col']=col
            part['coefficient'] += .001*i + .0001*j
            pieces.append(part)
    full=pd.concat(pieces,ignore_index=True)
    full['row']=pd.Categorical(full['row'],categories=['Base','Rate','TPC','Unfm','Ft0'],ordered=True)
    full['col']=pd.Categorical(full['col'],categories=['Constant','Tgl','Qpt','TglQpt','MdEdx','TglMdEdx','MdEdxQpt','TglMdEdxQpt'],ordered=True)
    fig,axes,_=_draw(full,color_display='legend',figure_margins=[.08,.96,.89,.08],subplot_spacing=[.25,.3])
    try:
        assert axes.shape==(5,8)
        assert len(fig.axes)==40
        assert len(fig.legends)==1
        assert fig.subplotpars.top==pytest.approx(.89)
    finally:plt.close(fig)


def test_b14_layout_still_one_owner(monkeypatch):
    from matplotlib.figure import Figure
    hits = []
    original = Figure.tight_layout
    def spy(self,*args,**kwargs):
        hits.append(id(self))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Figure,'tight_layout',spy)
    fig,axes,_=_draw(_frame(),color_display='legend')
    try:
        assert hits == [id(fig)]
    finally:plt.close(fig)


def test_b15_max_legend_alone_does_not_silently_do_nothing():
    with pytest.raises(ValueError,match='max_legend_categories requires color_display'):
        _draw(_frame(),max_legend_categories=8)


def test_b16_string_color_with_errorbars_preserves_explicit_refusal():
    df=_frame()
    df['color_name']=np.where(df['side']==0,'A','B')
    with pytest.raises(ValueError,match='numeric or boolean|categorical'):
        _draw(df,color='color_name',color_display='legend')


def test_b17_three_dimensional_facets_keep_one_color_domain_across_figures():
    df=_frame(singleton=True)
    df['page']=np.where(df['side']==0,'X','Y')
    fig_list,axes_list,stats_list=DFDraw(df).draw(
        'coefficient:region',type='scatter',color='side',
        marker='side > 0.5',yerr='error',
        facet_by=['row','col','page'],share_y='none',
        color_display='legend',facet_labels='outer')
    try:
        assert len(fig_list)==2
        assert all(len(fig.legends)==1 for fig in fig_list)
        assert all([t.get_text() for t in fig.legends[0].get_texts()]==['side=0','side=1']
                   for fig in fig_list)
        colors=[]
        for fig, axs in zip(fig_list,axes_list):
            for ax in axs.flat:
                for side,rgba in _point_colors(ax):
                    colors.append((int(side),rgba))
        for side in (0,1):
            vals=[v for k,v in colors if k==side]
            assert vals and all(np.allclose(v,vals[0]) for v in vals)
    finally:
        for fig in fig_list: plt.close(fig)


def test_b18_numeric_color_expression_three_categories_with_one_legend():
    df=_frame()
    df['third']=np.arange(len(df))%3
    fig,axes,_=_draw(df,color='third',marker='o',color_display='legend')
    try:
        assert [t.get_text() for t in fig.legends[0].get_texts()]==[
            'third=0','third=1','third=2']
        assert len(fig.axes)==4
    finally:plt.close(fig)


def test_b19_grouped_color_owner_does_not_silently_fabricate_side_legend():
    with pytest.raises(ValueError,match='color_display with group_by'):
        _draw(_frame(),group_by='region',color_display='legend')


def test_b20_nonfaceted_explicit_none_is_legacy_compatible():
    fig,ax,_=DFDraw(_frame()).scatter('coefficient:region',color='side',
                                       color_display=None)
    try:
        assert len(fig.legends)==0
    finally:plt.close(fig)


def test_b21_nonfaceted_legend_refuses_before_matplotlib():
    with pytest.raises(ValueError,match='color_display.*facet_by'):
        DFDraw(_frame()).scatter('coefficient:region',color='side',
                                color_display='legend')


@pytest.mark.parametrize('top_margin', [.89, .92])
def test_b_R11_legend_disjoint_from_column_headings_and_title(top_margin):
    """Renderer bbox oracle at both approved and previously colliding top gutters."""
    df = _frame()
    pieces = []
    rows = ['Base', 'Rate', 'TPC', 'Unfm', 'Ft0']
    cols = ['Constant', 'Tgl', 'Qpt', 'TglQpt',
            'MdEdx', 'TglMdEdx', 'MdEdxQpt', 'TglMdEdxQpt']
    for row in rows:
        for col in cols:
            part = df.iloc[:4].copy()
            part['row'] = row
            part['col'] = col
            pieces.append(part)
    full = pd.concat(pieces, ignore_index=True)
    full['row'] = pd.Categorical(full['row'], categories=rows, ordered=True)
    full['col'] = pd.Categorical(full['col'], categories=cols, ordered=True)
    margins = [.08, .96, top_margin, .08]
    fig, axes, _ = _draw(
        full, color_display='legend', figure_margins=margins,
        subplot_spacing=[.25, .30],
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pad = 1.5 * fig.dpi / 72.0
        legend_bbox = fig.legends[0].get_window_extent(renderer)
        title_bbox = fig._suptitle.get_window_extent(renderer)
        head_bboxes = [next(t for t in fig.texts if t.get_text() == col)
                      .get_window_extent(renderer) for col in cols]
        assert fig.subplotpars.top == pytest.approx(top_margin)
        assert all(not legend_bbox.overlaps(head) for head in head_bboxes)
        assert all(head.y1 + pad < legend_bbox.y0 for head in head_bboxes)
        assert legend_bbox.y1 + pad < title_bbox.y0
        assert title_bbox.y1 < fig.bbox.height
    finally:
        plt.close(fig)


def test_b_R12_after_return_manual_annotation_edit_is_not_reversed():
    """No draw_event/savefig callback may undo scientists' manual edits."""
    fig, axes, _ = _draw(_frame(), color_display='legend',
                         figure_margins=[.16,.94,.85,.12])
    try:
        fig.legends[0].set_bbox_to_anchor((.30, .75), transform=fig.transFigure)
        fig._supylabel.set_x(.025)
        expected_anchor = fig.legends[0].get_bbox_to_anchor().bounds
        fig.canvas.draw()
        assert fig.legends[0].get_bbox_to_anchor().bounds == pytest.approx(expected_anchor)
        assert fig._supylabel.get_position()[0] == pytest.approx(.025)
    finally:
        plt.close(fig)
