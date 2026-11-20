"""PHASE_13_111_DF S0 - one identity (colour and name) per curve: RED tests and controls.

Plain-language background
-------------------------
When dfdraw draws several curves, every curve except the first gets its colour
and legend name before it is drawn.  The first curve gets neither, so its
quantile lines, quantile labels, fit line and median line take the colours of
other curves, and its legend name is attached afterwards to "the first unnamed
line", which may belong to another curve.  Groups in a facet cell take their
colour from the drawing order, so a missing group shifts the others.  An
explicit ``quantile_mode='band'`` with the wrong number of levels crashes or
silently draws the wrong range.  Proposal: PHASE_13_111_DF_BugFixQuantile v0.2.

How the tests identify a curve
------------------------------
Every curve is made recognisable by its values: curve k lies at y = 10*k
(profiles) or x = 10*k (histograms), with a spread of 0.3.  Any drawn artist is
assigned to a curve by rounding its data position / 10, so the tests check the
rule, not the implementation:

* every artist of one curve has one colour (children inherit, CR-3);
* two curves never share a colour (no borrowed colour);
* on fresh axes curve k has palette colour k-1, in every facet mode and cell (CR-1).

Tests named ``test_RED_*`` fail on the start code (dfdraw = tag 78281b26) for the
stated reason and must pass after S1/S2.  Tests named ``test_CTL_*`` pass now and
must still pass.  Tests marked ``D2`` or ``D4`` encode the proposed decisions D-2
(fit line in the curve colour) and D-4 (explicit band: symmetric triple only);
the D4 band tests accept either decision option where the option matters.

Seat: Fable5.1_5:draw-coder (PHASE_13_111 coder), 2026-11-18.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from dfextensions.dfdraw.drawer import DFDraw
from dfextensions.dfdraw.style import get_style_value

Q3 = [0.11, 0.5, 0.9]          # asymmetric -> discrete lines (the reported call)
PCT = re.compile(r'^\d+(\.\d+)?%$')


# --------------------------------------------------------------------------
# data and helpers
# --------------------------------------------------------------------------
def _frame(n=24000, seed=11101):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.uniform(0.0, 4.0, n),
        'a': rng.uniform(0.0, 1.0, n),
        'b': rng.uniform(0.0, 1.0, n),
        't': rng.uniform(0.0, 100.0, n),
        'g': rng.integers(1, 4, n),
    })
    for k in (1, 2, 3, 4):
        df[f'y{k}'] = 10.0 * k + rng.normal(0.0, 0.3, n)
    df['yg'] = 10.0 * df['g'] + rng.normal(0.0, 0.3, n)
    df['ys'] = np.where(df['t'] < 50, 10.0, 20.0) + rng.normal(0.0, 0.3, n)
    return df


@pytest.fixture(autouse=True)
def _close_all():
    plt.close('all')
    yield
    plt.close('all')


def _palette(i):
    cmap = plt.colormaps.get_cmap(get_style_value('colors.palette', 'tab10'))
    return to_hex(cmap(i), keep_alpha=False)


def _hex(c):
    a = np.asarray(c, dtype=object)
    if a.ndim == 2:
        c = a[0]
    return to_hex(c, keep_alpha=False)


def _ident(v):
    return int(round(float(v) / 10.0))


def _data_axes(fig):
    return [ax for ax in fig.axes if ax.lines or ax.patches or ax.collections]


def _profile_pairs(ax):
    """(identity, colour, what) for every data artist of a profile axes."""
    out = []
    for ln in ax.lines:
        y = np.asarray(ln.get_ydata(), float)
        if y.size >= 2 and np.isfinite(y).any():
            out.append((_ident(np.nanmean(y)), _hex(ln.get_color()), 'line'))
    for c in ax.collections:
        name = type(c).__name__
        if name == 'LineCollection':
            segs = c.get_segments()
            if segs:
                ys = np.concatenate([np.asarray(s)[:, 1] for s in segs])
                if np.isfinite(ys).any():
                    out.append((_ident(np.nanmean(ys)), _hex(c.get_colors()), 'errorbar'))
        elif 'PolyCollection' in name:
            v = c.get_paths()[0].vertices[:, 1] if c.get_paths() else np.array([])
            if v.size and np.isfinite(v).any():
                out.append((_ident(np.nanmean(v)), _hex(c.get_facecolor()), 'band'))
        elif name == 'PathCollection':
            off = np.asarray(c.get_offsets(), float)
            if off.size:
                out.append((_ident(np.nanmean(off[:, 1])), _hex(c.get_facecolor()), 'points'))
    for t in ax.texts:
        if PCT.match(t.get_text()) and t.get_transform() is ax.transData:
            out.append((_ident(t.get_position()[1]), _hex(t.get_color()), 'label'))
    return out


def _hist_pairs(ax):
    """(identity, colour, what) for every data artist of a histogram axes."""
    out = []
    for ln in ax.lines:
        x = np.asarray(ln.get_xdata(), float)
        y = np.asarray(ln.get_ydata(), float)
        if x.size < 2:
            continue
        if np.allclose(x, x[0]):
            out.append((_ident(x[0]), _hex(ln.get_color()), 'marker'))
        elif np.isfinite(y).any():
            out.append((_ident(x[int(np.nanargmax(y))]), _hex(ln.get_color()), 'fit'))
    for p in ax.patches:
        v = p.get_path().vertices
        v = p.get_patch_transform().transform(v)
        if v.size and np.nanmax(v[:, 1]) > 0:
            out.append((_ident(v[int(np.nanargmax(v[:, 1])), 0]), _hex(p.get_facecolor()), 'bars'))
    return out


def _check(pairs, ids, order=True, where=''):
    """One colour per identity, no shared colour, optional palette order."""
    colours = {}
    for i, c, what in pairs:
        if i in ids:
            colours.setdefault(i, set()).add((c, what))
    problems = []
    for i in ids:
        cs = {c for c, _ in colours.get(i, set())}
        if len(cs) != 1:
            problems.append(f'curve {i}: colours {sorted(colours.get(i, set()))}')
    first = {i: sorted({c for c, _ in colours.get(i, set())})[0]
             for i in ids if colours.get(i)}
    if len(set(first.values())) != len(first):
        problems.append(f'shared colour between curves: {first}')
    if order:
        for rank, i in enumerate(sorted(ids)):
            if i in first and len({c for c, _ in colours[i]}) == 1 and first[i] != _palette(rank):
                problems.append(f'curve {i}: {first[i]} instead of palette({rank}) {_palette(rank)}')
    assert not problems, where + '; '.join(problems)
    return first


def _legend_entries(fig):
    """(text, handle colour) of every axes and Figure legend."""
    legs = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    out = []
    for lg in legs:
        for h, t in zip(lg.legend_handles, lg.get_texts()):
            col = h.get_facecolor() if type(h).__name__ in ('Rectangle', 'Polygon') else h.get_color()
            out.append((t.get_text(), _hex(col)))
    return out


def _profile(expr, **kw):
    kw.setdefault('type', 'profile')
    kw.setdefault('bins', 8)
    DFDraw(_frame()).draw(expr, **kw)
    return plt.gcf()


def _hist(expr, **kw):
    kw.setdefault('type', 'hist')
    kw.setdefault('bins', 70)
    kw.setdefault('range', (0.0, 35.0))
    DFDraw(_frame()).draw(expr, **kw)
    return plt.gcf()


# --------------------------------------------------------------------------
# RED: first curve takes other curves' colours (S-1, S-2, S-9)
# --------------------------------------------------------------------------
@pytest.mark.parametrize('n', [3, 4])
def test_RED_S1_vector_discrete_quantiles_unfaceted(n):
    expr = '[' + ','.join(f'y{k}' for k in range(1, n + 1)) + ']:x'
    fig = _profile(expr, quantiles=Q3)
    _check(_profile_pairs(fig.axes[0]), set(range(1, n + 1)), where=f'{n} curves: ')


def test_RED_S1_vector_discrete_quantiles_2level_facets_shared_legend():
    fig = _profile('[y1,y2,y3]:x', quantiles=Q3, facet_by=['a', 'b'],
                   facet_by_bins=[2, 2], legend='shared')
    pairs = [p for ax in _data_axes(fig) for p in _profile_pairs(ax)]
    curve = _check(pairs, {1, 2, 3})
    # quantile-level legend samples must not wear another curve's colour
    # (true for both D-1 options: the first curve's colour, or neutral grey)
    others = {curve[2], curve[3]}
    bad = [(t, c) for t, c in _legend_entries(fig)
           if (t.startswith('q=') or t == 'median') and c in others]
    assert not bad, f'quantile legend samples in other curves colours: {bad}'


def test_RED_S1_selection_branches_discrete_quantiles():
    fig = _profile('ys:x', quantiles=Q3, selection_vector=['t<50', 't>=50'],
                   vector_compose='outer')
    _check(_profile_pairs(fig.axes[0]), {1, 2})


def test_RED_S1_single_curve_discrete_quantiles():
    fig = _profile('y1:x', quantiles=Q3)
    _check(_profile_pairs(fig.axes[0]), {1})


def test_RED_S1_same_true_chain_profile_quantiles():
    d = DFDraw(_frame())
    d.draw('y1:x', type='profile', bins=8, quantiles=Q3)
    d.draw('y2:x', type='profile', bins=8, quantiles=Q3, same=True)
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2})


def test_RED_S1_through_aliasdataframe_draw():
    adf_mod = pytest.importorskip('dfextensions.AliasDataFrame')
    adf = adf_mod.AliasDataFrame(_frame())
    adf.draw('[y1,y2,y3]:x', type='profile', bins=8, quantiles=Q3)
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2, 3})


def test_RED_S2_central_none():
    fig = _profile('[y1,y2]:x', quantiles=Q3, central='none')
    _check(_profile_pairs(fig.axes[0]), {1, 2})


def test_RED_S9_central_both():
    fig = _profile('[y1,y2]:x', central='both')
    _check(_profile_pairs(fig.axes[0]), {1, 2})


# --------------------------------------------------------------------------
# RED: fit lines (S-3, S-4)
# --------------------------------------------------------------------------
def test_RED_D2_S3_profile_single_curve_fit():
    fig = _profile('y1:x', fit='pol1')
    _check(_profile_pairs(fig.axes[0]), {1})


def test_RED_D2_S4_hist_single_curve_fit():
    fig = _hist('y1', fit='gauss')
    _check(_hist_pairs(fig.axes[0]), {1})


def test_RED_S4_hist_vector_fit():
    fig = _hist('[y1,y2]', fit='gauss')
    _check(_hist_pairs(fig.axes[0]), {1, 2})


# --------------------------------------------------------------------------
# RED: legend name on another curve's artist (S-5)
# --------------------------------------------------------------------------
def _assert_names(fig, names, curve):
    entries = dict(_legend_entries(fig))
    missing = [n for n in names if n not in entries]
    wrong = {n: entries[n] for n in names if n in entries and entries[n] != curve[int(n[1:])]}
    assert not missing and not wrong, f'missing {missing}; wrong colour {wrong}; legend {_legend_entries(fig)}'


def test_RED_S5_hist_vector_quantiles_legend_names():
    fig = _hist('[y1,y2]', quantiles=[0.5])
    curve = _check(_hist_pairs(fig.axes[0]), {1, 2})
    _assert_names(fig, ['y1', 'y2'], curve)


def test_RED_S5_same_true_chain_hist_quantiles_legend_names():
    d = DFDraw(_frame())
    d.draw('y1', type='hist', bins=70, range=(0.0, 35.0), quantiles=[0.5])
    d.draw('y2', type='hist', bins=70, range=(0.0, 35.0), quantiles=[0.5], same=True)
    fig = plt.gcf()
    curve = _check(_hist_pairs(fig.axes[0]), {1, 2})
    _assert_names(fig, ['y1', 'y2'], curve)


# --------------------------------------------------------------------------
# RED: colour order across facet modes and cells (S-6, S-8)
# --------------------------------------------------------------------------
def test_RED_S6_profile_one_level_facet_order():
    fig = _profile('[y1,y2]:x', facet_by='a', facet_by_bins=2)
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2})


def test_RED_S6_hist_one_level_facet_order():
    fig = _hist('[y1,y2]', facet_by='a', facet_by_bins=2)
    _check([p for ax in _data_axes(fig) for p in _hist_pairs(ax)], {1, 2})


def _missing_group_frame():
    df = _frame()
    return df[~((df['a'] >= 0.5) & (df['g'] == 1))].copy()


def test_RED_S8_missing_group_one_level_facet_legend():
    DFDraw(_missing_group_frame()).draw('yg:x', type='profile', bins=6,
                                        group_by='g', facet_by='a', facet_by_bins=2)
    fig = plt.gcf()
    curve = _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})
    wrong = [(t, c) for t, c in _legend_entries(fig)
             if t in ('1', '2', '3') and c != curve[int(t)]]
    assert not wrong, f'legend names a colour that belongs to another group: {wrong}'


def test_RED_S8_missing_group_two_level_facet_stable_colours():
    DFDraw(_missing_group_frame()).draw('yg:x', type='profile', bins=6,
                                        group_by='g', facet_by=['a', 'b'], facet_by_bins=[2, 2])
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})


# --------------------------------------------------------------------------
# RED: caller axes with an existing user line (S-10)
# --------------------------------------------------------------------------
def test_RED_S10_caller_axes_with_user_line():
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [0.0, 0.0], label='user')          # identity 0
    DFDraw(_frame()).draw('[y1,y2]:x', type='profile', bins=8, ax=ax)
    _check(_profile_pairs(ax), {0, 1, 2}, order=False)


# --------------------------------------------------------------------------
# RED: explicit quantile_mode='band' (S-7).  Valid outcomes: a ValueError before
# any Figure, or (D-4 option b, two levels only) a band at the outer pair.
# --------------------------------------------------------------------------
def _band_call(kind, q, **kw):
    df = _frame()
    before = set(plt.get_fignums())
    try:
        rng = {'range': (0.0, 4.0)} if kind == 'profile' else {}
        DFDraw(df).draw('y1:x', type=kind, bins=1, quantiles=q, quantile_mode='band', **rng, **kw)
    except ValueError as e:
        new = set(plt.get_fignums()) - before
        assert not new, f'refused, but {len(new)} Figure(s) left open'
        assert 'band' in str(e)
        return None
    except Exception as e:                                   # pragma: no cover - the RED reason
        new = set(plt.get_fignums()) - before
        pytest.fail(f'{type(e).__name__}: {str(e)[:80]} ({len(new)} Figure(s) left open)')
    return plt.gcf(), np.quantile(df['y1'], [min(q), max(q)])


def _band_extent(fig):
    ext = []
    for ax in _data_axes(fig):
        for c in ax.collections:
            if 'PolyCollection' in type(c).__name__ and c.get_paths():
                v = c.get_paths()[0].vertices[:, 1]
                ext.append((np.nanmin(v), np.nanmax(v)))
    return ext


@pytest.mark.parametrize('kind', ['profile', 'hist2d+profile'])
def test_RED_S7_five_levels_band_is_refused(kind):
    out = _band_call(kind, [0.05, 0.25, 0.5, 0.75, 0.95])
    assert out is None, f'{kind}: five levels drawn as band {_band_extent(out[0])}'


@pytest.mark.parametrize('kind,facet', [('profile', None), ('profile', 1), ('profile', 2),
                                        ('hist2d+profile', None), ('hist2d+profile', 1)])
def test_RED_D4_S7_two_levels_band_refused_or_outer_pair(kind, facet):
    kw = {}
    if facet == 1:
        kw = dict(facet_by='a', facet_by_bins=2)
    elif facet == 2:
        kw = dict(facet_by=['a', 'b'], facet_by_bins=[2, 2])
    out = _band_call(kind, [0.16, 0.84], **kw)
    if out is None:
        return                                               # D-4 (a): refused cleanly
    fig, (lo, hi) = out
    ext = _band_extent(fig)
    assert ext, 'two-level band request drew no band (silently replaced)'
    assert all(abs(e[0] - lo) < 0.15 and abs(e[1] - hi) < 0.15 for e in ext), \
        f'band {ext} is not the outer pair {lo:.2f}..{hi:.2f}'


# --------------------------------------------------------------------------
# Controls: correct today, must stay correct
# --------------------------------------------------------------------------
def test_CTL_auto_band_vector():
    fig = _profile('[y1,y2,y3]:x', quantiles=[0.1, 0.5, 0.9])
    _check(_profile_pairs(fig.axes[0]), {1, 2, 3})


def test_CTL_nested_band_vector():
    fig = _profile('[y1,y2]:x', quantiles=[0.05, 0.25, 0.75, 0.95], quantile_mode='nested_band')
    _check(_profile_pairs(fig.axes[0]), {1, 2})


def test_CTL_error_bars_vector():
    fig = _profile('[y1,y2]:x', quantiles=[0.16, 0.84])
    _check(_profile_pairs(fig.axes[0]), {1, 2})


def test_CTL_group_by_complete_cells_quantiles():
    DFDraw(_frame()).draw('yg:x', type='profile', bins=6, group_by='g', quantiles=Q3,
                          facet_by='a', facet_by_bins=2)
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})


def test_CTL_group_scope_global_missing_group():
    DFDraw(_missing_group_frame()).draw('yg:x', type='profile', bins=6, group_by='g',
                                        facet_by='a', facet_by_bins=2, group_by_scope='global')
    fig = plt.gcf()
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2, 3})


def test_CTL_hist_vector_quantile_marker_colours():
    fig = _hist('[y1,y2]', quantiles=[0.16, 0.5, 0.84])
    _check(_hist_pairs(fig.axes[0]), {1, 2})


def test_CTL_scatter_vector():
    DFDraw(_frame()).draw('[y1,y2]:x', type='scatter', sample=2000)
    _check(_profile_pairs(plt.gcf().axes[0]), {1, 2})


def test_CTL_profile_vector_fit():
    fig = _profile('[y1,y2]:x', fit='pol1')
    _check(_profile_pairs(fig.axes[0]), {1, 2})


def test_CTL_two_level_facet_vector_order():
    fig = _profile('[y1,y2]:x', facet_by=['a', 'b'], facet_by_bins=[2, 2])
    _check([p for ax in _data_axes(fig) for p in _profile_pairs(ax)], {1, 2})


def test_CTL_explicit_colour_profile_method_wins():
    DFDraw(_frame()).profile('y1:x', bins=8, color='k', quantiles=Q3)
    cols = {c for _, c, _ in _profile_pairs(plt.gcf().axes[0])}
    assert cols == {'#000000'}, cols


def test_CTL_valid_symmetric_triple_band_values():
    df = _frame()
    DFDraw(df).draw('y1:x', type='profile', bins=1, range=(0.0, 4.0),
                    quantiles=[0.1, 0.5, 0.9], quantile_mode='band')
    lo, hi = np.quantile(df['y1'], [0.1, 0.9])
    ext = _band_extent(plt.gcf())
    assert ext and abs(ext[0][0] - lo) < 1e-9 and abs(ext[0][1] - hi) < 1e-9, (ext, lo, hi)


def test_CTL_numbers_vector_quantile_lines_equal_numpy():
    """Drawn quantile values are unchanged by a colour fix: compare with NumPy."""
    df = _frame()
    DFDraw(df).draw('[y1,y2]:x', type='profile', bins=4, range=(0.0, 4.0), quantiles=Q3)
    ax = plt.gcf().axes[0]
    idx = np.clip(np.floor(df['x'].to_numpy()).astype(int), 0, 3)
    for k in (1, 2):
        for q, ls in zip(Q3, ['--', '-.', ':']):
            truth = [np.quantile(df[f'y{k}'].to_numpy()[idx == b], q) for b in range(4)]
            drawn = [ln for ln in ax.lines if ln.get_linestyle() == ls
                     and _ident(np.nanmean(ln.get_ydata())) == k]
            assert len(drawn) == 1, (k, q, len(drawn))
            assert np.allclose(drawn[0].get_ydata(), truth, rtol=1e-12, atol=1e-12), (k, q)
