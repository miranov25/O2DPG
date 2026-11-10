"""PHASE_13_108_DF S1 - explicit quantile drawing switch on the faceted overlay.

Plain-language background
-------------------------
A profile draws its quantiles either as a filled area (band) or as one line
per level.  With the default ``quantile_mode='auto'`` the choice follows the
levels: symmetric levels (e.g. 0.05/0.1/0.5/0.9/0.95) give a band, and a
single asymmetric level (e.g. 0.101) gives lines.  The explicit switch
``quantile_mode=`` ('discrete' = lines, 'band' / 'nested_band' = area,
'error_bars', 'auto') already exists on ``profile``.  The Architect asked
(2026-11-10) that lines be selectable explicitly instead of by making the
levels slightly asymmetric.  The faceted ``hist2d+profile`` overlay refused
the keyword (PHASE_13_106 closed keyword list); PHASE_13_108 accepts it as a
presentation switch of the profile layer.

* 'discrete' with symmetric levels: one line per level in every cell and one
  Figure legend (PHASE_13_108 default) with exact labels;
* 'band' with an asymmetric level: no quantile lines and no legend;
* every cell's numbers equal the unfaceted overlay with the same switch;
* an unknown value is refused before any Figure is created.

Seat: Fable5_1:draw-coder, 2026-11-10.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw.drawer import DFDraw

Q_SYMMETRIC = [0.05, 0.1, 0.5, 0.9, 0.95]
Q_ASYMMETRIC = [0.05, 0.101, 0.5, 0.9, 0.95]


def _frame():
    rng = np.random.default_rng(10811)
    n = 4500
    return pd.DataFrame({'x': rng.uniform(0.0, 1.0, n),
                         'y': rng.normal(0.0, 1.0, n),
                         'f': rng.integers(0, 3, n)})


def _quantile_lines(ax):
    return [ln for ln in ax.lines
            if ln.get_label().startswith('q=') or ln.get_label() == 'median']


def test_S1_01_discrete_draws_lines_for_symmetric_levels():
    fig, axes, _ = DFDraw(_frame()).draw(
        'y:x', type='hist2d+profile', facet_by='f', quantiles=Q_SYMMETRIC,
        quantile_mode='discrete')
    try:
        for ax in np.asarray(axes).ravel():
            assert len(_quantile_lines(ax)) == 5
        assert [[t.get_text() for t in L.get_texts()] for L in fig.legends] \
            == [['q=5%', 'q=10%', 'median', 'q=90%', 'q=95%']]
    finally:
        plt.close(fig)


def test_S1_02_band_draws_no_lines_for_asymmetric_levels():
    fig, axes, _ = DFDraw(_frame()).draw(
        'y:x', type='hist2d+profile', facet_by='f', quantiles=Q_ASYMMETRIC,
        quantile_mode='band')
    try:
        for ax in np.asarray(axes).ravel():
            assert _quantile_lines(ax) == []
            assert ax.get_legend() is None
        assert fig.legends == []
    finally:
        plt.close(fig)


@pytest.mark.parametrize('mode', ['discrete', 'band'])
def test_S1_03_cells_equal_unfaceted_overlay_with_same_switch(mode):
    df = _frame()
    fig, axes, stats = DFDraw(df).draw(
        'y:x', type='hist2d+profile', facet_by='f', quantiles=Q_SYMMETRIC,
        quantile_mode=mode)
    try:
        for k in (0, 1, 2):
            fk, _, sk = DFDraw(df[df.f == k].reset_index(drop=True)).draw(
                'y:x', type='hist2d+profile', quantiles=Q_SYMMETRIC,
                quantile_mode=mode)
            try:
                cell = stats['per_group'][str(k)]['layers'][1]
                ref = sk['layers'][1]
                assert sorted(cell) == sorted(ref)
                for key in ('n', 'q_lower_per_bin', 'q_upper_per_bin',
                            'quantiles_per_bin'):
                    if key in ref:
                        a, b = cell[key], ref[key]
                        if isinstance(b, dict):
                            for q in b:
                                np.testing.assert_array_equal(a[q], b[q])
                        else:
                            np.testing.assert_array_equal(a, b)
            finally:
                plt.close(fk)
    finally:
        plt.close(fig)


def test_S1_04_unknown_value_refused_before_any_figure():
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="quantile_mode must be 'auto'"):
        DFDraw(_frame()).draw('y:x', type='hist2d+profile', facet_by='f',
                              quantiles=Q_SYMMETRIC, quantile_mode='lines')
    assert set(plt.get_fignums()) == before
