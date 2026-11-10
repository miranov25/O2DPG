"""PHASE_13_106_DF S2 - faceted ``type='hist2d+profile'`` (C1): contract details.

Plain-language background
-------------------------
C1 adds the faceted overlay that was planned as Phase 13.53 and never
delivered: ``draw(expr, type='hist2d+profile', facet_by=<column>)``.  The
main RED tests are in ``test_phase_13_106_s0_red.py`` (one cell per facet;
cell = the unfaceted overlay of that facet's rows; raw numpy truth; selection
per cell; one colorbar per cell; weighted quantiles refused).  This module
pins the remaining parts of the S0 contract:

* independent axes by default, so no cell is clipped to another cell's range;
* ``n_total`` and per-cell ``n`` count the selected rows (this fixture has
  only finite values; in general they count the selected rows that are
  finite in x and y, see ``test_phase_13_106_s8_c1_numeric_oracle.py``);
* selection and sampling are applied once, before the frame is split;
* binned facets reuse the existing facet machinery (same groups as hist2d);
* the automatic title names the requested quantiles;
* refusals, raised before any Figure is created: the explicit
  ``layers=[...]`` form with ``facet_by`` (Architect choice D106-3: only the
  string form is supported), two facet columns, channel facets, ``share_x``,
  expression vectors.  (``legend=`` was refused here until PHASE_13_108,
  which supports it; see the note at S2_06.)

Seat: Fable5_1:draw-coder, 2026-11-09.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

QUANTILES = [0.05, 0.101, 0.5, 0.9, 0.95]


def _frame():
    rng = np.random.default_rng(10602)
    n = 6000
    pid = rng.integers(0, 3, n)
    return pd.DataFrame({
        'x': rng.uniform(0.5, 3.0, n),
        'y': rng.normal(size=n) * (0.3 + 0.4 * pid),   # wider spread per PID
        'pid': pid,
        'w': rng.uniform(0.5, 1.5, n),
    })


def _draw(df=None, **kw):
    params = dict(type='hist2d+profile', quantiles=QUANTILES, facet_by='pid',
                  auto_title=True)
    params.update(kw)
    return DFDraw(_frame() if df is None else df).draw('y:x', **params)


def test_S2_01_cells_have_independent_axes_equal_to_unfaceted_overlay():
    df = _frame()
    fig, axes, _ = _draw(df)
    try:
        for k, ax in zip((0, 1, 2), np.asarray(axes).flat):
            fig_k, ax_k, _ = DFDraw(df[df.pid == k].reset_index(drop=True)).draw(
                'y:x', type='hist2d+profile', quantiles=QUANTILES)
            try:
                assert ax.get_ylim() == ax_k.get_ylim()
                assert ax.get_xlim() == ax_k.get_xlim()
            finally:
                plt.close(fig_k)
    finally:
        plt.close(fig)


def test_S2_02_entry_counts_follow_the_selected_rows():
    df = _frame()
    fig, axes, stats = _draw(df, selection='x > 1')
    try:
        assert stats['n_total'] == int((df.x > 1).sum())
        for k in (0, 1, 2):
            assert stats['per_group'][str(k)]['n'] == int(
                ((df.x > 1) & (df.pid == k)).sum())
    finally:
        plt.close(fig)


def test_S2_03_sampling_happens_once_before_the_split():
    fig, axes, stats = _draw(sample=900)
    try:
        assert stats['n_total'] == 900
    finally:
        plt.close(fig)


def test_S2_04_binned_facet_reuses_existing_facet_groups():
    df = _frame()
    fig, axes, stats = _draw(df, facet_by='x', facet_by_bins=4, ncols=2)
    fig_h, axes_h, stats_h = DFDraw(df).draw('y:x', type='hist2d',
                                             facet_by='x', facet_by_bins=4,
                                             ncols=2)
    try:
        assert stats['groups'] == stats_h['groups']
        assert len(np.asarray(axes).flat) == 4
    finally:
        plt.close(fig)
        plt.close(fig_h)


def test_S2_05_auto_title_names_the_quantiles():
    fig, axes, _ = _draw()
    try:
        assert '0.101' in fig._suptitle.get_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize('kw,err,match', [
    (dict(facet_by=['pid', 'x'], facet_by_bins=[None, 2]),
     NotImplementedError, 'exactly one ordinary facet column'),
    (dict(facet_by='vector'), NotImplementedError, 'ordinary data-column'),
    (dict(share_x='row'), NotImplementedError, 'share_x'),
], ids=['two-columns', 'channel-facet', 'share_x'])
# PHASE_13_108 (Architect D108-C, 2026-11-10): the former 'legend' case
# (legend='shared' refused) is removed; legend= is now supported through the
# existing legend system and tested positively in
# test_phase_13_108_s0_facet_default_style_red.py (C01, C02).
def test_S2_06_unsupported_forms_refused_before_any_figure(kw, err, match):
    before = set(plt.get_fignums())
    with pytest.raises(err, match=match):
        _draw(**kw)
    assert set(plt.get_fignums()) == before


def test_S2_07_explicit_layers_form_with_facet_refused():
    """Architect choice D106-3: only the string form is supported."""
    with pytest.raises(ValueError, match='does not facet'):
        DFDraw(_frame()).overlay(
            'y:x', layers=[{'type': 'hist2d'}, {'type': 'profile'}],
            facet_by='pid')


def test_S2_08_expression_vector_refused():
    df = _frame().assign(y2=lambda f: 2 * f.y)
    with pytest.raises((NotImplementedError, ValueError)):
        DFDraw(df).draw('[y,y2]:x', type='hist2d+profile', facet_by='pid')


def test_S2_09_weighted_quantiles_refused_before_any_figure():
    """Refused up front, not only by the profile layer inside the first cell
    (which would leave a half-drawn Figure behind)."""
    before = set(plt.get_fignums())
    with pytest.raises(NotImplementedError, match='weighted quantiles'):
        _draw(weights='w')
    assert set(plt.get_fignums()) == before
