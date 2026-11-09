"""PHASE_13_106_DF S0 - RED characterization for the approved bug-fix cases.

Plain-language background
-------------------------
PHASE_13_106 fixes user-reported plotting failures before PHASE_13_82 resumes.
This module is written **before** any product change.  It records:

* C4 - a one-element facet list.  ``facet_by=['f']`` with an expression
  vector (``'[y0,y1]:x'``) is refused today, because every list goes to the
  two-column PHASE_13_94 path.  ``facet_by='f'`` works.  The approved fix makes
  the one-element list mean exactly the same as the string.
* C1 - a faceted overlay.  ``type='hist2d+profile'`` with ``facet_by`` is
  refused today ("needs faceted overlay (Phase 13.53)"); that phase never
  landed.  Approved contract (S0): **each facet cell is the unfaceted overlay
  applied to that facet's rows** - the same rule the existing faceted profile
  already follows (cell = profile of the subset, facet-local automatic edges).
  One colorbar per cell, as faceted hist2d does today.  Weighted quantiles stay
  refused, as in the existing profile owner.
* Controls that must stay as they are: two-column facet lists, refusal of
  three-column lists, scalar one-element lists (already working), the
  unfaceted overlay, and the Architect-deferred C2/C3 refusals.

Expected now (snapshot 93de6798): C4_01 (2 cases) and C1_01..C1_06 FAIL
(8 failed); the 8 controls pass.  Two existing tests pin the old refusals and will be inverted, with
disclosure, in the slices that change them: 13.99 FP07 (C4, slice S1) and
13.52 T-S5c (C1, slice S2).

Seat: Fable5_1:draw-coder, 2026-11-09.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

QUANTILES = [0.05, 0.101, 0.5, 0.9, 0.95]   # the reported user call


def _vector_frame():
    rng = np.random.default_rng(13106)
    n = 2400
    return pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'f': rng.uniform(0, 4, n),
        'g': np.tile([0.0, 1.0], n // 2),
        'y0': rng.normal(size=n),
        'y1': 2 + rng.normal(size=n),
    })


def _overlay_frame():
    rng = np.random.default_rng(1061)
    n = 3000
    pid = rng.integers(0, 3, n)
    return pd.DataFrame({
        'x': rng.uniform(0, 1, n) * (1 + pid),     # facet-dependent x range
        'y': rng.normal(size=n) + 0.5 * pid,
        'pid': pid,
        'w': rng.uniform(0.5, 1.5, n),
    })


def _close(fig):
    plt.close(fig)


def _quantiles_equal(a, b):
    assert sorted(a) == sorted(b)
    for q in a:
        assert len(a[q]) == len(b[q])
        np.testing.assert_array_equal(np.asarray(a[q]), np.asarray(b[q]))


def _cell_stats_equal(a, b, where=''):
    """Every statistic of one facet cell is identical (scalars, arrays, dicts).

    The key set is compared too, so a statistic present on one side only
    (for example ``q_lower_per_bin`` versus ``quantiles_per_bin``) fails."""
    assert sorted(a) == sorted(b), where
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _cell_stats_equal(va, vb, f'{where}/{key}')
        elif isinstance(va, (np.ndarray, list)):
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb),
                                          err_msg=f'{where}/{key}')
        else:
            assert va == vb or (va != va and vb != vb), f'{where}/{key}'


# --------------------------------------------------------------------------
# C4 - one-element facet list
# --------------------------------------------------------------------------

def _texts(fig):
    """All visible figure text: suptitle, panel titles, axis labels, legends."""
    out = [fig._suptitle.get_text() if fig._suptitle is not None else '']
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    for leg in fig.legends:
        out += [t.get_text() for t in leg.get_texts()]
    out += [t.get_text() for t in fig.texts]
    return out


def _vector_profile(**kw):
    params = dict(type='profile', bins=5, quantiles=[0.2, 0.5, 0.8],
                  auto_title=True)
    params.update(kw)
    return DFDraw(_vector_frame()).draw('[y0,y1]:x', **params)


@pytest.mark.parametrize('bins_list', [4, [4]], ids=['bins-int', 'bins-list'])
def test_C4_01_singleton_list_equals_string_facet(bins_list):
    """RED: facet_by=['f'] must give the same panels and every per-cell
    statistic of 'f' (all keys, scalars and per-bin arrays)."""
    fig_s, axes_s, stats_s = _vector_profile(facet_by='f', facet_by_bins=4)
    fig_l, axes_l, stats_l = _vector_profile(facet_by=['f'],
                                             facet_by_bins=bins_list)
    try:
        assert np.asarray(axes_l).shape == np.asarray(axes_s).shape
        assert len(fig_l.axes) == len(fig_s.axes)
        assert _texts(fig_l) == _texts(fig_s)
        assert len(stats_l) == len(stats_s)
        for s_l, s_s in zip(stats_l, stats_s):
            assert s_l['n_groups'] == s_s['n_groups']
            assert list(s_l['groups']) == list(s_s['groups'])
            assert sorted(s_l['per_group']) == sorted(s_s['per_group'])
            for key in s_s['per_group']:
                _cell_stats_equal(s_l['per_group'][key],
                                  s_s['per_group'][key], where=key)
    finally:
        _close(fig_s)
        _close(fig_l)


def test_C4_02_two_column_list_keeps_two_dimensional_path():
    """Control: a genuine two-column list still uses the 2-D facet path."""
    fig, axes, stats = _vector_profile(facet_by=['f', 'g'],
                                       facet_by_bins=[2, 2])
    try:
        assert len(np.asarray(axes).flat) == 4
    finally:
        _close(fig)


def test_C4_03_three_column_list_still_refused():
    """Control: more than two facet columns stays refused."""
    with pytest.raises(NotImplementedError):
        _vector_profile(facet_by=['f', 'g', 'x'], facet_by_bins=[2, 2, 2])


@pytest.mark.parametrize('kind,expr', [('profile', 'y0:x'), ('hist', 'y0')])
def test_C4_04_scalar_singleton_list_already_equals_string(kind, expr):
    """Control: scalar expressions already accept a one-element list."""
    df = _vector_frame()
    fig_s, axes_s, st_s = DFDraw(df).draw(expr, type=kind, bins=5,
                                          facet_by='f', facet_by_bins=4)
    fig_l, axes_l, st_l = DFDraw(df).draw(expr, type=kind, bins=5,
                                          facet_by=['f'], facet_by_bins=4)
    try:
        assert np.asarray(axes_l).shape == np.asarray(axes_s).shape
        assert st_l['groups'] == st_s['groups']
    finally:
        _close(fig_s)
        _close(fig_l)


# --------------------------------------------------------------------------
# C1 - faceted hist2d+profile (string form, the reported call shape)
# --------------------------------------------------------------------------

def _faceted_overlay(df, **kw):
    params = dict(type='hist2d+profile', quantiles=QUANTILES,
                  facet_by='pid', auto_title=True)
    params.update(kw)
    return DFDraw(df).draw('y:x', **params)


def test_C1_01_string_form_returns_one_cell_per_facet():
    """RED: three PID values give three data axes and a faceted envelope."""
    fig, axes, stats = _faceted_overlay(_overlay_frame())
    try:
        assert len(np.asarray(axes).flat) == 3
        assert stats['faceted'] is True
        assert list(stats['groups']) == [0, 1, 2]
        for key in ('0', '1', '2'):
            assert len(stats['per_group'][key]['layers']) == 2
    finally:
        _close(fig)


def test_C1_02_each_cell_equals_unfaceted_overlay_on_its_rows():
    """RED: cell k = the existing unfaceted overlay on the rows with pid == k
    (all five quantiles, including the discrete 0.101 level)."""
    df = _overlay_frame()
    fig, axes, stats = _faceted_overlay(df)
    try:
        for k in (0, 1, 2):
            sub = df[df.pid == k].reset_index(drop=True)
            fig_k, _, ref = DFDraw(sub).draw('y:x', type='hist2d+profile',
                                             quantiles=QUANTILES)
            try:
                cell = stats['per_group'][str(k)]['layers']
                for lay, lay_ref in zip(cell, ref['layers']):
                    for name in ('n', 'n_input', 'mean_x', 'mean_y',
                                 'median_y', 'std_y'):
                        assert lay[name] == lay_ref[name], (k, name)
                _quantiles_equal(cell[1]['quantiles_per_bin'],
                                 ref['layers'][1]['quantiles_per_bin'])
            finally:
                _close(fig_k)
    finally:
        _close(fig)


def test_C1_03_cell_counts_and_moments_match_raw_rows():
    """RED: independent numpy truth per PID, both layers on the same rows."""
    df = _overlay_frame()
    fig, axes, stats = _faceted_overlay(df)
    try:
        for k in (0, 1, 2):
            sub = df[df.pid == k]
            for lay in stats['per_group'][str(k)]['layers']:
                assert lay['n'] == len(sub)
                assert lay['mean_y'] == pytest.approx(sub.y.mean(), rel=1e-12)
                assert lay['median_y'] == pytest.approx(sub.y.median(),
                                                        rel=1e-12)
    finally:
        _close(fig)


def test_C1_04_selection_is_applied_inside_every_cell():
    """RED: a global selection reaches both layers of every cell."""
    df = _overlay_frame()
    fig, axes, stats = _faceted_overlay(df, selection='y > 0')
    try:
        for k in (0, 1, 2):
            want = int(((df.pid == k) & (df.y > 0)).sum())
            for lay in stats['per_group'][str(k)]['layers']:
                assert lay['n'] == want
    finally:
        _close(fig)


def test_C1_05_one_colorbar_per_cell_like_faceted_hist2d():
    """RED: figure holds 3 data axes + 3 colorbars, as faceted hist2d does."""
    df = _overlay_frame()
    fig, axes, stats = _faceted_overlay(df)
    fig_h, axes_h, _ = DFDraw(df).draw('y:x', type='hist2d', facet_by='pid')
    try:
        assert len(fig.axes) == len(fig_h.axes) == 6
    finally:
        _close(fig)
        _close(fig_h)


def test_C1_06_weighted_quantiles_refused_early_as_in_profile():
    """RED: weights + quantiles stay refused by the profile owner's rule,
    not by the old facet guard."""
    with pytest.raises(NotImplementedError, match='weighted quantiles'):
        _faceted_overlay(_overlay_frame(), weights='w')


def test_C1_07_unfaceted_overlay_unchanged():
    """Control: the unfaceted overlay still returns one axes with two layers."""
    fig, ax, stats = DFDraw(_overlay_frame()).draw(
        'y:x', type='hist2d+profile', quantiles=QUANTILES)
    try:
        assert len(stats['layers']) == 2
        assert len(fig.axes) == 2
    finally:
        _close(fig)


# --------------------------------------------------------------------------
# C2 / C3 - Architect decision 2026-11-09: both deferred; refusals stay
# --------------------------------------------------------------------------

def test_C2_deferred_hist_normalize_with_facet_or_group_stays_refused():
    df = pd.DataFrame({'a': np.linspace(17, 23, 600),
                       'b': np.linspace(17, 23, 600)[::-1],
                       'f': np.tile([0.0, 1.0], 300)})
    for extra in ({'facet_by': 'f'}, {'group_by': 'f'}):
        with pytest.raises(NotImplementedError, match='hist normalize='):
            DFDraw(df).draw('[a,b]', type='hist', bins=10, range=(17.0, 23.0),
                            normalize='ratio', vector_compose='outer', **extra)


def test_C3_deferred_scatter_vector_facet_stays_refused():
    with pytest.raises(NotImplementedError, match='expression-vector × facet_by'):
        DFDraw(_vector_frame()).draw('[y0,y1]:x', type='scatter', facet_by='f')


def test_C3_histogram_vector_facet_still_supported():
    """Control: PHASE_13_103 B5-B histogram vector x one ordinary facet."""
    fig, axes, stats = DFDraw(_vector_frame()).draw(
        '[y0,y1]', type='hist', bins=8, facet_by='g')
    try:
        assert len(np.asarray(axes).flat) == 2
    finally:
        _close(fig)
