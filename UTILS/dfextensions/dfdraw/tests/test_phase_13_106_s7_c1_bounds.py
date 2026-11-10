"""PHASE_13_106_DF S7 - bounds of the faceted ``hist2d+profile`` overlay (C1).

Plain-language background
-------------------------
C1 (committed in ``1aaa7518``) added ``draw(expr, type='hist2d+profile',
facet_by=<column>)``: one cell per facet value, each cell the existing
unfaceted overlay of that facet's rows.  The PRECOMMIT review of that
candidate (consolidated summary GPT29, 2026-11-09) found that the new route
accepted more than the Architect approved:

* P1-1: other type pairs (``hexbin+profile``, ``hist2d+scatter``,
  ``profile2d+profile``) were faceted too, although choice D106-3 approved
  only ``hist2d+profile``;
* P1-2: ``weights=`` without quantiles weighted only the profile while the
  density map stayed unweighted, and several named ``draw()`` options
  (``same``, ``normalize``, the selection/weight label options, ...) were
  dropped silently.

This module pins the corrected bounds:

* only ``type='hist2d+profile'`` is faceted; other pairs keep the
  ``ValueError`` they raised before PHASE_13_106;
* a closed list of accepted keywords; everything else is refused, naming the
  keyword, before any Figure is created;
* the accepted density keywords reach every cell's density map, and
  ``bins=`` changes the density bins only (the profile keeps its default
  50 X bins: the existing overlay routing, documented, not changed);
* controls: the same pairs and keywords without ``facet_by`` behave as
  before (nothing unfaceted is narrowed).

Seat: Fable5_1:draw-coder, 2026-11-09.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh

from dfextensions.dfdraw.drawer import DFDraw

OTHER_PAIRS = ['hexbin+profile', 'hist2d+scatter', 'profile2d+profile',
               'hexbin+scatter']


def _frame():
    rng = np.random.default_rng(10607)
    n = 3000
    return pd.DataFrame({
        'x': rng.uniform(0.0, 1.0, n),
        'y': rng.normal(size=n),
        'pid': rng.integers(0, 3, n),
        'w': rng.uniform(0.5, 2.0, n),
        'y2': rng.normal(size=n),
    })


def _no_new_figure(fn):
    before = set(plt.get_fignums())
    try:
        fn()
    finally:
        assert set(plt.get_fignums()) == before, 'a Figure was left behind'


# -- P1-1: only hist2d+profile is faceted ------------------------------------

@pytest.mark.parametrize('pair', OTHER_PAIRS)
def test_S7_01_other_type_pairs_refused_with_facet(pair):
    def call():
        with pytest.raises(ValueError,
                           match=re.escape("only for type='hist2d+profile'")):
            DFDraw(_frame()).draw('y:x', type=pair, facet_by='pid')
    _no_new_figure(call)


@pytest.mark.parametrize('pair', OTHER_PAIRS)
def test_S7_02_other_type_pairs_unfaceted_unchanged(pair):
    """Control: without facet_by these pairs still draw as before."""
    fig, ax, stats = DFDraw(_frame()).draw('y:x', type=pair)
    try:
        assert len(stats['layers']) == 2
    finally:
        plt.close(fig)


# -- P1-2: weights and every non-accepted keyword refused --------------------

def test_S7_03_weights_without_quantiles_refused():
    def call():
        with pytest.raises(NotImplementedError,
                           match='density map would stay unweighted'):
            DFDraw(_frame()).draw('y:x', type='hist2d+profile',
                                  facet_by='pid', weights='w')
    _no_new_figure(call)


REFUSED = [
    # replicated or channel options that would change the population of one
    # layer, or are dropped by the string form
    ('selection_vector', ['x > 0.5', 'x <= 0.5']),
    ('weights_vector', ['w', 'w']),
    ('selection_labels', ['a', 'b']),
    ('weights_labels', ['a', 'b']),
    ('selection_categorical', True),
    ('weights_categorical', True),
    ('vector_compose', 'outer'),
    ('group_by', 'pid'),
    ('same', True),
    ('normalize', 'ratio'),
    ('normalize_layout', 'overlay'),
    ('delta_facet', 'pid'),
    ('fit', 'pol1'),
    ('fit_textbox_kwargs', {}),
    ('summary_fit', 'pol1'),
    # density keyword that would limit only one layer's population
    ('range', ((0.0, 0.5), (-1.0, 1.0))),
    ('nan_policy', 'warn'),
    # accepted by the unfaceted overlay, not reviewed for facets
    ('norm', 'log'),
    ('stats', True),
    ('color', 'red'),
    ('min_entries', 5),
    ('xlabel', 'x'),
]


@pytest.mark.parametrize('key,value', REFUSED, ids=[k for k, _ in REFUSED])
def test_S7_04_keyword_refused_before_any_figure(key, value):
    def call():
        with pytest.raises(NotImplementedError,
                           match=re.escape(f'does not support {key}=')) as e:
            DFDraw(_frame()).draw('y:x', type='hist2d+profile',
                                  facet_by='pid', **{key: value})
        assert 'accepted keywords: ' in str(e.value)
    _no_new_figure(call)


@pytest.mark.parametrize('key,value,match', [
    ('stat_list', ['std'], "stat_list is only supported for type='profile'"),
    ('group_by_scope', 'facet', 'group_by_scope is only supported'),
    ('facet_label_format', '.1f', 'facet_label_format Stage A supports'),
], ids=['stat_list', 'group_by_scope', 'facet_label_format'])
def test_S7_04b_keyword_refused_by_existing_draw_guard(key, value, match):
    """These three are refused for every overlay type by older draw()
    guards, before the faceted route is reached."""
    def call():
        with pytest.raises(NotImplementedError, match=re.escape(match)):
            DFDraw(_frame()).draw('y:x', type='hist2d+profile',
                                  facet_by='pid', **{key: value})
    _no_new_figure(call)


@pytest.mark.parametrize('key,value', [
    ('weights', 'w'),
    ('range', ((0.0, 0.5), (-1.0, 1.0))),
    ('nan_policy', 'warn'),
    ('stats', True),
])
def test_S7_05_same_keyword_unfaceted_unchanged(key, value):
    """Control: the refusals apply to the faceted route only."""
    fig, ax, stats = DFDraw(_frame()).draw('y:x', type='hist2d+profile',
                                           **{key: value})
    try:
        assert len(stats['layers']) == 2
    finally:
        plt.close(fig)


# -- accepted keywords reach the intended layer in every cell -----------------

def _densities(axes):
    out = []
    for ax in np.asarray(axes).flat:
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        out.append(meshes[0])
    return out


def test_S7_06_bins_set_the_density_only_profile_keeps_default():
    """Review P2-2: plain bins= is routed to the density base layer (the
    existing overlay rule for a keyword both layers accept).  The profile
    keeps its default 50 X bins.  Pinned so the behaviour is documented,
    not changed."""
    fig, axes, stats = DFDraw(_frame()).draw(
        'y:x', type='hist2d+profile', facet_by='pid', bins=10,
        quantiles=[0.5])
    try:
        for mesh in _densities(axes):
            assert mesh.get_array().shape == (10, 10)
        for label in ('0', '1', '2'):
            qpb = stats['per_group'][label]['layers'][1]['quantiles_per_bin']
            assert len(qpb[0.5]) == 50
    finally:
        plt.close(fig)


def test_S7_07_density_style_keywords_reach_every_cell():
    fig, axes, _ = DFDraw(_frame()).draw(
        'y:x', type='hist2d+profile', facet_by='pid', cmap='magma',
        vmin=1, vmax=7, clabel='entries')
    try:
        for mesh in _densities(axes):
            assert mesh.get_cmap().name == 'magma'
            assert mesh.get_clim() == (1.0, 7.0)
        cbar_axes = [a for a in fig.axes
                     if a not in list(np.asarray(axes).flat)]
        assert len(cbar_axes) == 3
        assert all(a.get_ylabel() == 'entries' for a in cbar_axes)
    finally:
        plt.close(fig)


def test_S7_08_colorbar_false_removes_every_cell_colorbar():
    fig, axes, _ = DFDraw(_frame()).draw(
        'y:x', type='hist2d+profile', facet_by='pid', colorbar=False)
    try:
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


# -- public AliasDataFrame door ------------------------------------------------

def test_S7_09_adf_public_path_same_cells_and_same_refusal():
    """Native ADF integration gate: skips only where AliasDataFrame is not
    installed; the Alma2 canonical run must execute it."""
    adf_module = pytest.importorskip('dfextensions.AliasDataFrame')
    df = _frame()
    adf = adf_module.AliasDataFrame(df)
    kw = dict(type='hist2d+profile', facet_by='pid', quantiles=[0.1, 0.5],
              bins=20, selection='x > 0.2')
    direct = DFDraw(df).draw('y:x', **kw)
    result = adf.draw('y:x', **kw)
    try:
        s_d, s_a = direct[2], result[2]
        assert s_d['groups'] == s_a['groups']
        assert s_d['n_total'] == s_a['n_total']
        for label in s_d['per_group']:
            q_d = s_d['per_group'][label]['layers'][1]['quantiles_per_bin']
            q_a = s_a['per_group'][label]['layers'][1]['quantiles_per_bin']
            for q in (0.1, 0.5):
                np.testing.assert_allclose(q_a[q], q_d[q], rtol=0, atol=1e-12,
                                           equal_nan=True)
    finally:
        plt.close(direct[0])
        plt.close(result[0])

    def call():
        with pytest.raises(ValueError,
                           match=re.escape("only for type='hist2d+profile'")):
            adf.draw('y:x', type='hexbin+profile', facet_by='pid')
        with pytest.raises(NotImplementedError,
                           match='density map would stay unweighted'):
            adf.draw('y:x', type='hist2d+profile', facet_by='pid', weights='w')
    _no_new_figure(call)
