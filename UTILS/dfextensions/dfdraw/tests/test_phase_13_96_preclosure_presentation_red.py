"""PHASE_13_96 pre-closure presentation RED/GREEN oracles.

Two bounded corrections only:
1) 2-D list-facet auto-title must retain/fits the effective string selection.
2) Stage F must not change the historical no-stat grouped N-D vector legend.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dfextensions.dfdraw.drawer import DFDraw


def _fixture():
    rows = []
    for fr in ('R0', 'R1'):
        for fc in ('C0', 'C1'):
            for gi, g in enumerate(('G0', 'G1', 'G2')):
                for i in range(24):
                    rows.append({
                        'x': i / 3.0,
                        'y0': 0.25 * i + gi,
                        'y1': -0.15 * i + 0.5 * gi,
                        'g': g,
                        'fr': fr,
                        'fc': fc,
                        'keep': (i % 5) != 0,
                    })
    return pd.DataFrame(rows)


def _stagef_fixture():
    rows = []
    noise = np.linspace(-1.0, 1.0, 6)
    for fr in range(3):
        for fc in range(5):
            for gi in range(5):
                for xb in range(2):
                    for eps in noise:
                        rows.append({
                            'x': xb + 0.5 + 1e-4 * eps,
                            'g': gi + 0.03 * eps,
                            'fr': fr + 0.2,
                            'fc': fc + 0.2,
                            'y0': 0.3 * xb + 0.1 * gi + (1.0 + 0.05 * fc) * eps,
                            'y1': -0.2 * xb + 0.08 * gi + (1.3 + 0.04 * fr) * eps,
                            'keep': True,
                        })
    return pd.DataFrame(rows)


def _legend_handles(legend):
    if hasattr(legend, 'legend_handles'):
        return legend.legend_handles
    return legend.legendHandles


def _legacy_kwargs(**extra):
    kw = dict(
        selection='keep',
        group_by='g', group_by_scope='global',
        facet_by=['fr', 'fc'],
        bins=5,
        legend='shared', facet_labels='outer',
        auto_title=True,
    )
    kw.update(extra)
    return kw


def _stagef_kwargs(**extra):
    kw = dict(
        selection='keep',
        group_by='g', group_by_bins=5, group_by_scope='global',
        facet_by=['fr', 'fc'], facet_by_bins=[3, 5],
        bins=2, range=(0.0, 2.0), min_entries=2,
        stat_list=['std'], legend='shared',
        group_style='color', vector_style='marker', linestyle='none',
        facet_labels='outer', auto_title=True,
    )
    kw.update(extra)
    return kw


def test_PC01_2d_facet_auto_title_includes_effective_selection():
    fig, axes, stats = DFDraw(_fixture()).profile(
        'y0:x', selection='keep', facet_by=['fr', 'fc'], bins=5,
        facet_labels='outer', auto_title=True)
    try:
        assert fig._suptitle is not None
        assert 'keep' in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_PC02_2d_facet_auto_title_reuses_width_fit_helper(monkeypatch):
    import dfextensions.dfdraw.plots._auto_title as auto_mod
    calls = []
    original = auto_mod.fit_auto_title_to_figure

    def tracked(fig, title_dict, sub_fontsize=None):
        calls.append(dict(title_dict))
        return original(fig, title_dict, sub_fontsize=sub_fontsize)

    monkeypatch.setattr(auto_mod, 'fit_auto_title_to_figure', tracked)
    fig, axes, stats = DFDraw(_fixture()).profile(
        'y0:x', selection='keep', facet_by=['fr', 'fc'], bins=5,
        facet_labels='outer', auto_title=True)
    try:
        assert len(calls) == 1
        assert calls[0]['sub'] == 'keep'
    finally:
        plt.close(fig)


def test_PC03_auto_title_false_does_not_inject_selection_title():
    fig, axes, stats = DFDraw(_fixture()).profile(
        'y0:x', selection='keep', facet_by=['fr', 'fc'], bins=5,
        facet_labels='outer', auto_title=False)
    try:
        assert fig._suptitle is None
    finally:
        plt.close(fig)


def test_PC04_explicit_title_remains_owner():
    fig, axes, stats = DFDraw(_fixture()).profile(
        'y0:x', selection='keep', facet_by=['fr', 'fc'], bins=5,
        facet_labels='outer', auto_title=True, title='USER TITLE')
    try:
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == 'USER TITLE'
    finally:
        plt.close(fig)


def test_PC05_selection_title_preserves_shared_stagef_legend_and_shape():
    fig, axes, stats = DFDraw(_stagef_fixture()).profile('[y0,y1]:x', **_stagef_kwargs())
    try:
        assert axes.shape == (3, 5)
        assert len(fig.legends) == 1
        assert fig._suptitle is not None and 'keep' in fig._suptitle.get_text()
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert len(labels) == 7 and labels[-2:] == ['y0', 'y1']
    finally:
        plt.close(fig)


def test_PC06_legacy_no_stat_legend_order_matches_pre_stagef():
    fig, axes, stats = DFDraw(_fixture()).profile('[y0,y1]:x', **_legacy_kwargs())
    try:
        assert len(fig.legends) == 1
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert labels == ['y0', 'y1', 'G0', 'G1', 'G2']
    finally:
        plt.close(fig)


def test_PC07_legacy_marker_request_keeps_historical_expression_proxy_identity():
    fig, axes, stats = DFDraw(_fixture()).profile(
        '[y0,y1]:x', **_legacy_kwargs(vector_style='marker'))
    try:
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        handles = _legend_handles(legend)
        assert labels == ['y0', 'y1', 'G0', 'G1', 'G2']
        assert handles[0].get_marker() in (None, 'None', '')
        assert handles[1].get_marker() in (None, 'None', '')
        assert handles[0].get_linestyle() == '-'
        assert handles[1].get_linestyle() == '--'
    finally:
        plt.close(fig)


def test_PC08_stagef_legend_order_and_marker_proxies_remain_new_contract():
    fig, axes, stats = DFDraw(_stagef_fixture()).profile('[y0,y1]:x', **_stagef_kwargs())
    try:
        legend = fig.legends[0]
        labels = [t.get_text() for t in legend.get_texts()]
        handles = _legend_handles(legend)
        assert len(labels) == 7 and labels[-2:] == ['y0', 'y1']
        assert handles[-2].get_marker() == 'o'
        assert handles[-1].get_marker() == 's'
        assert handles[-2].get_linestyle() in ('None', 'none', '')
        assert handles[-1].get_linestyle() in ('None', 'none', '')
    finally:
        plt.close(fig)


def test_PC09_presentation_paths_preserve_public_stats_shape():
    d = _stagef_fixture()
    legacy_kw = _stagef_kwargs()
    legacy_kw.pop('stat_list')
    f0, a0, s0 = DFDraw(d).profile('[y0,y1]:x', **legacy_kw)
    f1, a1, s1 = DFDraw(d).profile('[y0,y1]:x', **_stagef_kwargs())
    try:
        assert a0.shape == a1.shape == (3, 5)
        assert [k for k in s0 if isinstance(k, tuple)] == [k for k in s1 if isinstance(k, tuple)]
        for k in [k for k in s0 if isinstance(k, tuple)]:
            assert isinstance(s0[k], list) and len(s0[k]) == 2
            assert isinstance(s1[k], list) and len(s1[k]) == 2
    finally:
        plt.close(f0); plt.close(f1)
