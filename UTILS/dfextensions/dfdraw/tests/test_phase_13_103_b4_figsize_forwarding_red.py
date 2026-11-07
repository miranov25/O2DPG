"""PHASE_13_103_DF B4: generic scalar facet figsize forwarding.

RED on the October 31 source: DFDraw.draw binds figsize as a named
argument, then omits it from scalar hist/scatter/profile facet dispatch.
The typed methods and vector-profile facet owner already support it.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


@pytest.fixture
def data():
    rng = np.random.default_rng(1310304)
    n = 720
    return pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'y': rng.normal(size=n),
        'y2': rng.normal(size=n),
        'f': np.repeat(np.arange(6), n // 6).astype(float),
        'g': np.tile(np.repeat([0, 1], n // 12), 6),
        'h': np.tile(np.repeat(np.arange(3), n // 18), 6).astype(float),
    })


def _facet(dim):
    if dim == '1d':
        return {'facet_by': 'f', 'facet_by_bins': 6, 'ncols': 3}
    return {'facet_by': ['f', 'h'], 'facet_by_bins': [2, 3]}


def _type_args(kind):
    return ('x' if kind == 'hist' else 'y:x',
            {'bins': 8} if kind in ('hist', 'profile') else {})


def _size(fig):
    return tuple(float(x) for x in fig.get_size_inches())


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_01_explicit_figsize_through_generic_scalar_dispatch(data, kind, dim):
    expr, opts = _type_args(kind)
    fig, axes, stats = DFDraw(data).draw(expr, type=kind, figsize=(16, 10),
                                         **_facet(dim), **opts)
    try:
        assert _size(fig) == (16.0, 10.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_02_omitted_figsize_matches_typed_existing_defaults(data, kind, dim):
    expr, opts = _type_args(kind)
    plotter = DFDraw(data)
    fig_draw, _, _ = plotter.draw(expr, type=kind, **_facet(dim), **opts)
    fig_typed, _, _ = getattr(plotter, kind)(expr, **_facet(dim), **opts)
    try:
        assert _size(fig_draw) == _size(fig_typed)
    finally:
        plt.close(fig_draw)
        plt.close(fig_typed)


@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_03_grouped_profile_explicit_figsize(data, dim):
    fig, _, _ = DFDraw(data).draw('y:x', type='profile', bins=8,
                                  group_by='g', figsize=(16, 10),
                                  **_facet(dim))
    try:
        assert _size(fig) == (16.0, 10.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_04_typed_facet_owner_remains_correct(data, kind, dim):
    expr, opts = _type_args(kind)
    fig, _, _ = getattr(DFDraw(data), kind)(expr, figsize=(16, 10),
                                            **_facet(dim), **opts)
    try:
        assert _size(fig) == (16.0, 10.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_05_vector_profile_facets_remain_correct(data, dim):
    fig, _, _ = DFDraw(data).draw('[y,y2]:x', type='profile', bins=8,
                                  figsize=(16, 10), **_facet(dim))
    try:
        assert _size(fig) == (16.0, 10.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_06_margins_and_spacing_survive_generic_forwarding(data, dim):
    fig, _, _ = DFDraw(data).draw(
        'y:x', type='profile', bins=8, figsize=(16, 10),
        figure_margins=[.18, .97, .78, .27],
        subplot_spacing=[.25, .35], **_facet(dim),
    )
    try:
        assert _size(fig) == (16.0, 10.0)
        p = fig.subplotpars
        assert np.allclose([p.left, p.right, p.top, p.bottom, p.wspace, p.hspace],
                           [.18, .97, .78, .27, .25, .35], atol=1e-12)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('dim', ['1d', '2d'])
def test_B4_07_adf_public_path_matches_direct_dfdraw(data, dim):
    # Alma2 has ADF installed; local source reconstruction may not.
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    adf = adf_module.AliasDataFrame(data)
    opts = dict(type='profile', bins=8, figsize=(16, 10), **_facet(dim))
    fig_adf, _, _ = adf.draw('y:x', **opts)
    fig_direct, _, _ = DFDraw(data).draw('y:x', **opts)
    try:
        assert _size(fig_adf) == _size(fig_direct) == (16.0, 10.0)
    finally:
        plt.close(fig_adf)
        plt.close(fig_direct)


# PHASE_13_103_DF B4 legacy-facet closeout (Opus48_3, 2026-11-06):
# facet=True with group_by= has a typed Figure owner just like facet_by=.
# Assert the generic route preserves both the caller's size and all group
# statistics. These are real rendered Figure-size checks, not mock calls.


def _legacy_draw_args(kind):
    expr, opts = _type_args(kind)
    return expr, dict(opts, group_by='g', facet=True)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
@pytest.mark.parametrize('requested', [(16, 10), (7, 5)])
def test_B4_08_legacy_explicit_figsize_matches_typed_and_statistics(
        data, kind, requested):
    expr, opts = _legacy_draw_args(kind)
    fig_typed = fig_draw = None
    try:
        fig_typed, _, stats_typed = getattr(DFDraw(data), kind)(
            expr, figsize=requested, **opts,
        )
        fig_draw, _, stats_draw = DFDraw(data).draw(
            expr, type=kind, figsize=requested, **opts,
        )
        assert np.allclose(_size(fig_typed), requested, atol=1e-12)
        assert np.allclose(_size(fig_draw), requested, atol=1e-12)
        assert len(fig_draw.axes) == len(fig_typed.axes)
        # Actual group and statistical payload invariance: no scalarized
        # values, selection counts or autorange may change for a size fix.
        assert stats_draw['n_groups'] == stats_typed['n_groups'] == 2
        assert stats_draw['groups'] == stats_typed['groups']
        assert stats_draw['n_total'] == stats_typed['n_total'] == len(data)
        assert stats_draw['per_group'] == stats_typed['per_group']
    finally:
        if fig_typed is not None:
            plt.close(fig_typed)
        if fig_draw is not None:
            plt.close(fig_draw)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
def test_B4_09_legacy_omitted_figsize_keeps_typed_default(data, kind):
    expr, opts = _legacy_draw_args(kind)
    fig_typed = fig_draw = None
    try:
        fig_typed, _, stats_typed = getattr(DFDraw(data), kind)(expr, **opts)
        fig_draw, _, stats_draw = DFDraw(data).draw(expr, type=kind, **opts)
        assert _size(fig_draw) == _size(fig_typed)
        assert stats_draw == stats_typed
    finally:
        if fig_typed is not None:
            plt.close(fig_typed)
        if fig_draw is not None:
            plt.close(fig_draw)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
def test_B4_10_legacy_figsize_preserves_explicit_margins(data, kind):
    expr, opts = _legacy_draw_args(kind)
    fig = None
    try:
        fig, _, _ = DFDraw(data).draw(
            expr, type=kind, figsize=(12, 7),
            figure_margins=[.18, .97, .78, .27],
            subplot_spacing=[.25, .35], **opts,
        )
        assert np.allclose(_size(fig), (12, 7), atol=1e-12)
        p = fig.subplotpars
        assert np.allclose(
            [p.left, p.right, p.top, p.bottom, p.wspace, p.hspace],
            [.18, .97, .78, .27, .25, .35], atol=1e-12,
        )
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize('kind', ['profile', 'hist', 'scatter'])
def test_B4_11_unfaceted_figsize_remains_unchanged(data, kind):
    expr, opts = _type_args(kind)
    fig = None
    try:
        fig, _, _ = DFDraw(data).draw(
            expr, type=kind, figsize=(9, 5), **opts,
        )
        assert np.allclose(_size(fig), (9, 5), atol=1e-12)
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize('kind', ['profile', 'hist'])
def test_B4_12_adf_legacy_facet_matches_direct_public_path(data, kind):
    # Requires installed AliasDataFrame on Alma2; exact same public contract.
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    adf = adf_module.AliasDataFrame(data)
    expr, opts = _legacy_draw_args(kind)
    fig_adf = fig_direct = None
    try:
        fig_adf, _, stats_adf = adf.draw(
            expr, type=kind, figsize=(12, 7), **opts,
        )
        fig_direct, _, stats_direct = DFDraw(data).draw(
            expr, type=kind, figsize=(12, 7), **opts,
        )
        assert _size(fig_adf) == _size(fig_direct) == (12.0, 7.0)
        assert stats_adf['n_groups'] == stats_direct['n_groups']
        assert stats_adf['n_total'] == stats_direct['n_total']
        assert stats_adf['per_group'] == stats_direct['per_group']
    finally:
        if fig_adf is not None:
            plt.close(fig_adf)
        if fig_direct is not None:
            plt.close(fig_direct)
