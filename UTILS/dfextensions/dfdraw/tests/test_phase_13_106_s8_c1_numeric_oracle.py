"""PHASE_13_106_DF S8 - independent numbers for the faceted overlay (C1) and
the C4 companion forms.

Plain-language background
-------------------------
C1 is ``draw(expr, type='hist2d+profile', facet_by=<column>)``: one cell per
facet value, each with a density map and a profile with quantile lines.  The
earlier tests showed that every cell equals the existing unfaceted overlay of
that facet's rows.  The PRECOMMIT review (consolidated summary GPT29,
2026-11-09, item P1-3) asked for more: recompute the plotted numbers with raw
NumPy, independently of dfdraw's own binning code.

This module does that on one fixed data set:

* three facet values with very different populations (3000, 800, 150 rows);
* a global selection ``x > 0.1``;
* a few NaN and inf rows that pass the selection (they are not plotted).

For every cell it checks, against NumPy only:

* the counts: ``n`` is the number of selected rows that are finite in x and
  y, ``n_filtered`` the number of non-finite ones (review P2-2);
* the density map: its bin edges are the evenly spaced edges of the range the
  density layer reports, and every 2-D bin count equals ``numpy.histogram2d``
  on the finite rows;
* the profile: X-bin membership is rebuilt with explicit comparisons
  (lower edge inclusive, upper edge exclusive, the last bin also takes a row
  exactly on the upper edge, rows outside the range are dropped).  From that,
  the per-bin mean (the drawn central line) and all five quantiles
  ``0.05, 0.101, 0.5, 0.9, 0.95`` with ``numpy.percentile`` (linear) are
  compared: the stored per-bin arrays (empty bin -> NaN) and the drawn lines
  (only bins with at least ``min_entries=3`` rows are drawn).

The density and profile edges are checked separately; nothing assumes they
are equal.  The fixture is built so that the upper-edge rule and empty bins
are actually exercised; the test asserts that.

C4 companions (review P2-1): an expression-vector profile with
``facet_by=['f']`` and quantile facet binning equals ``facet_by='f'``, and
``facet_by=[]`` gives the same clear error as for scalar expressions.

Seat: Fable5_1:draw-coder, 2026-11-09.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh

from dfextensions.dfdraw.drawer import DFDraw

QUANTILES = [0.05, 0.101, 0.5, 0.9, 0.95]
QUANTILE_LABELS = ['q=5%', 'q=10%', 'median', 'q=90%', 'q=95%']
SELECTION = 'x > 0.1'
PROFILE_BINS = 50          # profile default X bins (see S7_06)
MIN_ENTRIES = 3            # profile default for drawn bins


def _frame():
    rng = np.random.default_rng(1068)
    parts = []
    for pid, size in ((0, 3000), (1, 800), (2, 150)):
        parts.append(pd.DataFrame({
            'x': rng.uniform(0.0, 2.0, size),
            'y': rng.normal(pid, 1.0 + pid, size),
            'pid': pid,
        }))
    df = pd.concat(parts, ignore_index=True)
    # non-finite rows that pass the selection (x > 0.1 holds for inf)
    df.loc[[11, 12, 3011], 'y'] = np.nan
    df.loc[[13, 3012], 'x'] = np.inf
    df.loc[3000 + 800 + 7, 'y'] = -np.inf
    return df


@pytest.fixture(scope='module')
def drawn():
    df = _frame()
    fig, axes, stats = DFDraw(df).draw(
        'y:x', type='hist2d+profile', facet_by='pid', quantiles=QUANTILES,
        selection=SELECTION)
    yield df, fig, np.asarray(axes).ravel(), stats
    plt.close(fig)


def _cell_rows(df, pid):
    """Raw selected rows of one facet, split into finite and non-finite."""
    rows = df[(df.x > 0.1) & (df.pid == pid)]
    finite = np.isfinite(rows.x.to_numpy()) & np.isfinite(rows.y.to_numpy())
    return rows, finite


def _profile_members(x, edges):
    """Explicit X-bin membership: [lo, hi) per bin, the last bin also takes
    x == upper edge, rows outside [first edge, last edge] belong to no bin."""
    nb = len(edges) - 1
    masks = []
    for i in range(nb):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if i == nb - 1:
            m |= x == edges[-1]
        masks.append(m)
    return masks


def _cells(drawn):
    df, fig, axes, stats = drawn
    for pid, ax in zip((0, 1, 2), axes):
        label = str(pid)
        assert label in stats['per_group']
        yield pid, ax, stats['per_group'][label]


def test_S8_01_counts_are_the_finite_selected_rows(drawn):
    df, fig, axes, stats = drawn
    total = 0
    for pid, ax, cell in _cells(drawn):
        rows, finite = _cell_rows(df, pid)
        density, profile = cell['layers']
        assert cell['n'] == int(finite.sum())
        assert density['n'] == profile['n'] == int(finite.sum())
        assert density['n_filtered'] == int((~finite).sum())
        total += int(finite.sum())
    assert stats['n_total'] == total
    # the non-finite control is really present in every cell
    assert all(int((~_cell_rows(df, p)[1]).sum()) > 0 for p in (0, 1, 2))


def test_S8_02_density_counts_equal_numpy_histogram2d(drawn):
    df, fig, axes, stats = drawn
    for pid, ax, cell in _cells(drawn):
        rows, finite = _cell_rows(df, pid)
        x = rows.x.to_numpy()[finite]
        y = rows.y.to_numpy()[finite]
        (xlo, xhi), (ylo, yhi) = cell['layers'][0]['autorange_used']
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        coords = meshes[0].get_coordinates()
        xedges, yedges = coords[0, :, 0], coords[:, 0, 1]
        nbx, nby = len(xedges) - 1, len(yedges) - 1
        np.testing.assert_allclose(xedges, np.linspace(xlo, xhi, nbx + 1),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(yedges, np.linspace(ylo, yhi, nby + 1),
                                   rtol=0, atol=1e-12)
        truth, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        drawn_counts = np.ma.filled(meshes[0].get_array(), 0.0)
        np.testing.assert_array_equal(
            np.asarray(drawn_counts).reshape(nby, nbx), truth.T,
            err_msg=f'density counts, facet {pid}')


def _profile_truth(df, pid, cell):
    rows, finite = _cell_rows(df, pid)
    x = rows.x.to_numpy()[finite]
    y = rows.y.to_numpy()[finite]
    lo, hi = cell['layers'][1]['autorange_used']
    edges = np.linspace(lo, hi, PROFILE_BINS + 1)
    members = _profile_members(x, edges)
    counts = np.array([m.sum() for m in members])
    means = np.array([y[m].mean() if m.any() else np.nan for m in members])
    quant = {q: np.array([np.percentile(y[m], 100 * q) if m.any() else np.nan
                          for m in members]) for q in QUANTILES}
    centers = 0.5 * (edges[:-1] + edges[1:])
    on_upper_edge = int((x == hi).sum())
    return counts, means, quant, centers, on_upper_edge


def test_S8_03_profile_quantiles_per_bin_equal_numpy(drawn):
    df, fig, axes, stats = drawn
    for pid, ax, cell in _cells(drawn):
        counts, means, quant, centers, _ = _profile_truth(df, pid, cell)
        stored = cell['layers'][1]['quantiles_per_bin']
        assert sorted(stored) == sorted(QUANTILES)
        for q in QUANTILES:
            np.testing.assert_allclose(
                stored[q], quant[q], rtol=0, atol=1e-12, equal_nan=True,
                err_msg=f'quantile {q}, facet {pid}')
            assert np.all(np.isnan(stored[q][counts == 0]))


def test_S8_04_drawn_mean_and_quantile_lines_equal_numpy(drawn):
    df, fig, axes, stats = drawn
    for pid, ax, cell in _cells(drawn):
        counts, means, quant, centers, _ = _profile_truth(df, pid, cell)
        shown = counts >= MIN_ENTRIES
        mean_line = ax.containers[0].lines[0]
        np.testing.assert_allclose(mean_line.get_xdata(), centers[shown],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(mean_line.get_ydata(), means[shown],
                                   rtol=0, atol=1e-12,
                                   err_msg=f'mean line, facet {pid}')
        qlines = [ln for ln in ax.lines if ln.get_label() in QUANTILE_LABELS]
        assert [ln.get_label() for ln in qlines] == QUANTILE_LABELS
        for q, ln in zip(QUANTILES, qlines):
            np.testing.assert_allclose(ln.get_xdata(), centers[shown],
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(ln.get_ydata(), quant[q][shown],
                                       rtol=0, atol=1e-12,
                                       err_msg=f'line {q}, facet {pid}')


def test_S8_05_fixture_exercises_edge_and_empty_bins(drawn):
    """Guard on the fixture itself: the upper-edge rule, empty bins and
    hidden low-count bins really occur, so S8_03/S8_04 test them."""
    df, fig, axes, stats = drawn
    upper, empty, hidden = 0, 0, 0
    for pid, ax, cell in _cells(drawn):
        counts, _, _, _, on_edge = _profile_truth(df, pid, cell)
        upper += on_edge
        empty += int((counts == 0).sum())
        hidden += int(((counts > 0) & (counts < MIN_ENTRIES)).sum())
    assert upper > 0
    assert empty > 0
    assert hidden > 0


# -- C4 companion forms (review P2-1) -----------------------------------------

def _vector_frame():
    rng = np.random.default_rng(13108)
    n = 2400
    return pd.DataFrame({
        'x': rng.uniform(0, 1, n),
        'f': rng.uniform(0, 4, n),
        'y0': rng.normal(size=n),
        'y1': 2 + rng.normal(size=n),
    })


def _vector_profile(**kw):
    return DFDraw(_vector_frame()).draw(
        '[y0,y1]:x', type='profile', bins=5, quantiles=[0.2, 0.5, 0.8],
        auto_title=True, **kw)


def _assert_same(a, b, where=''):
    if isinstance(a, dict):
        assert sorted(a) == sorted(b), where
        for k in a:
            _assert_same(a[k], b[k], f'{where}/{k}')
    elif isinstance(a, (list, tuple)) and a and isinstance(a[0], dict):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f'{where}[{i}]')
    elif isinstance(a, (np.ndarray, list, tuple)):
        np.testing.assert_array_equal(np.asarray(a, dtype=object),
                                      np.asarray(b, dtype=object),
                                      err_msg=where)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), where
    else:
        assert a == b, where


def _texts(fig):
    out = [fig._suptitle.get_text() if fig._suptitle is not None else '']
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    for leg in fig.legends:
        out += [t.get_text() for t in leg.get_texts()]
    return out


@pytest.mark.parametrize('fq', [4, [0.0, 0.25, 0.5, 0.75, 1.0]],
                         ids=['count', 'cut-points'])
def test_S8_06_c4_singleton_list_with_quantile_facet_equals_string(fq):
    fig_s, axes_s, stats_s = _vector_profile(facet_by='f',
                                             facet_by_quantiles=fq)
    fig_l, axes_l, stats_l = _vector_profile(facet_by=['f'],
                                             facet_by_quantiles=fq)
    try:
        assert np.asarray(axes_s).shape == np.asarray(axes_l).shape
        assert np.asarray(axes_s).size == 4
        _assert_same(stats_s, stats_l)
        assert _texts(fig_s) == _texts(fig_l)
    finally:
        plt.close(fig_s)
        plt.close(fig_l)


@pytest.mark.parametrize('expr,kind', [
    ('[y0,y1]:x', 'profile'),            # was the two-column message
    ('y0:x', 'profile'),                 # control: already this message
    ('y0:x', 'hist2d+profile'),          # faceted overlay
])
def test_S8_07_empty_facet_list_clear_error_before_any_figure(expr, kind):
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match='facet_by list cannot be empty'):
        DFDraw(_vector_frame()).draw(expr, type=kind, facet_by=[])
    assert set(plt.get_fignums()) == before
