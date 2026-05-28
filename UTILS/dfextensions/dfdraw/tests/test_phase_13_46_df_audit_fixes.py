"""
Phase 13.46.DF — Audit Bucket ① fix invariance tests (F.64–F.70).

Covers the five convergent audit findings:
  C-1  fit="gaus"  ROOT TF1 alias            — F.64
  C-2  type="histo" ROOT type alias          — F.65
  C-7  kwarg-typo guard (difflib did-you-mean)— F.66
  C-9  range= on scatter via shared resolver  — F.67 (non-faceted),
                                                F.68 (faceted, shared-axis),
                                                F.69a (strategy parity)
  C-9  original profile/hist unpack bug       — F.69b
  C-4  _get_suptitle public-API helper        — F.70

All tests go through the public DFDraw API (the path production code hits).
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots.fits import available_fits
from dfextensions.dfdraw.plots._autorange import compute_autorange
from dfextensions.dfdraw.drawer import _get_suptitle, _TYPE_ALIASES


# ============================================================================
# Fixtures
# ============================================================================

def _scatter_df(n=400, seed=0):
    rs = np.random.RandomState(seed)
    return pd.DataFrame({
        'x': rs.normal(5, 2, n),
        'y': rs.normal(10, 3, n),
        'g': rs.randint(0, 3, n),
    })


def _two_cluster_df(n_per=150, seed=1):
    """Two well-separated clusters by group, so a per-cell vs shared-axis
    range difference is large and unambiguous."""
    rs = np.random.RandomState(seed)
    return pd.DataFrame({
        'x': np.concatenate([rs.normal(0, 1, n_per), rs.normal(50, 1, n_per)]),
        'y': np.concatenate([rs.normal(0, 1, n_per), rs.normal(50, 1, n_per)]),
        'g': [0] * n_per + [1] * n_per,
    })


# ============================================================================
# TestPhase1346AuditFixes — F.64 .. F.70
# ============================================================================

class TestPhase1346AuditFixes:
    """Phase 13.46.DF audit bucket ① invariance regressions."""

    # -- F.64 — C-1: ROOT 'gaus' alias -----------------------------------

    def test_f64_gaus_root_alias(self):
        """fit='gaus' (ROOT TF1 name) resolves to the gaussian fit, with no
        ValueError, and produces a fit result identical in shape to
        fit='gauss'."""
        assert 'gaus' in available_fits()
        df = _scatter_df()
        fig_a, ax_a, st_a = DFDraw(df).hist('x', bins=40, fit='gaus')
        fig_b, ax_b, st_b = DFDraw(df).hist('x', bins=40, fit='gauss')
        assert 'fit' in st_a and st_a['fit'], "fit='gaus' must produce a fit result"
        assert 'fit' in st_b and st_b['fit']
        # Same underlying fit family: identical parameter-name structure.
        fa, fb = st_a['fit'][0][0], st_b['fit'][0][0]
        assert fa.get('param_names') == fb.get('param_names'), \
            "fit='gaus' must resolve to the same gaussian as fit='gauss'"
        plt.close(fig_a); plt.close(fig_b)

    # -- F.65 — C-2: ROOT 'histo' type alias -----------------------------

    def test_f65_histo_type_alias(self):
        """type='histo' (ROOT name) dispatches to the same plot kind as
        type='hist' with no ValueError."""
        assert _TYPE_ALIASES.get('histo') == 'hist'
        df = _scatter_df()
        fig, ax, st = DFDraw(df).draw('x', type='histo', bins=30)
        assert ax is not None
        plt.close(fig)

    # -- F.66 — C-7: kwarg-typo guard (difflib did-you-mean) -------------

    def test_f66_kwarg_typo_guard(self):
        """A near-miss kwarg (facet_by_bin) raises ValueError naming the
        intended kwarg (facet_by_bins); a genuinely-unknown far kwarg only
        warns (matplotlib passthrough preserved)."""
        df = _scatter_df()
        # near-miss → raise with suggestion
        with pytest.raises(ValueError) as exc:
            DFDraw(df).draw('x', type='hist', bins=20,
                            facet_by='g', facet_by_bin=3)
        assert 'facet_by_bins' in str(exc.value), \
            "typo guard must suggest the correct kwarg name"
        # far-unknown → warn, not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, st = DFDraw(df).draw('x', type='hist', bins=20, zorder=5)
            assert any("Unknown keyword" in str(wi.message) for wi in w), \
                "far-unknown kwarg should warn"
            plt.close(fig)
        # valid type-specific kwarg (cumulative) → NO spurious warning (N-1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, st = DFDraw(df).draw('x', type='hist', bins=20,
                                          cumulative=True, group_by='g')
            assert not any("Unknown keyword" in str(wi.message) for wi in w), \
                "valid type-specific kwarg must not warn"
            plt.close(fig)

    # -- F.67 — C-9: scatter range="minmax" (non-faceted) ----------------

    def test_f67_scatter_range_minmax_nonfaceted(self):
        """Non-faceted scatter range='minmax' sets axis limits exactly to the
        per-axis data min/max (via the shared resolver), no AttributeError,
        and records the applied strategy honestly."""
        df = _scatter_df()
        fig, ax, st = DFDraw(df).scatter('y:x', range="minmax")
        exp_x = compute_autorange(df['x'].values, strategy="minmax")
        exp_y = compute_autorange(df['y'].values, strategy="minmax")
        assert np.allclose(ax.get_xlim(), exp_x), "xlim must equal data minmax"
        assert np.allclose(ax.get_ylim(), exp_y), "ylim must equal data minmax"
        assert st.get("autorange_strategy") == "minmax"
        plt.close(fig)

    # -- F.68 — C-9: scatter range="minmax" (faceted, shared-axis) -------

    def test_f68_scatter_range_minmax_faceted(self):
        """Faceted scatter range='minmax' must not crash and must NOT exhibit
        last-cell-wins corruption. Because facet grids use SHARED axes
        (sharex/sharey), the consistent behavior — matching faceted hist — is
        that the shared axes cover the GLOBAL data extent across all cells,
        not one cell's range applied to all.

        Per-cell strategy tightening under faceting is a Phase 13.46.DF FIX1
        item (see CRR §2); this test locks the no-crash + global-coverage
        invariant so the silent last-cell-wins regression cannot return."""
        df = _two_cluster_df()
        res = DFDraw(df).scatter('y:x', facet_by='g', facet_by_bins=2,
                                 range="minmax")
        fig = res[0]
        axes = [a for a in fig.get_axes() if a.collections]
        assert len(axes) >= 2
        gx_lo, gx_hi = float(df['x'].min()), float(df['x'].max())
        # Every cell's shared x-limits must cover the global data extent
        # (NOT be pinned to a single cluster's [47,53] window).
        for a in axes:
            lo, hi = a.get_xlim()
            assert lo <= gx_lo + 1.0 and hi >= gx_hi - 1.0, \
                f"faceted scatter range must cover global extent, got {(lo, hi)}"
        plt.close(fig)

    # -- F.69a — C-9: strategy parity (hybrid, not minmax-only) ----------

    def test_f69a_scatter_range_strategy_parity(self):
        """Scatter range='hybrid' resolves to the hybrid window (proving ALL
        resolver strategies work on scatter, not just minmax), and an explicit
        ((xlo,xhi),(ylo,yhi)) tuple passes through verbatim."""
        df = _scatter_df()
        fig, ax, st = DFDraw(df).scatter('y:x', range="hybrid")
        exp_x = compute_autorange(df['x'].values, strategy="hybrid")
        assert np.allclose(ax.get_xlim(), exp_x), "hybrid window must be applied"
        assert st.get("autorange_strategy") == "hybrid"
        plt.close(fig)
        # explicit tuple
        fig2, ax2, st2 = DFDraw(df).scatter('y:x', range=((0, 10), (0, 20)))
        assert ax2.get_xlim() == (0, 10) and ax2.get_ylim() == (0, 20)
        assert st2.get("autorange_strategy") == "explicit"
        plt.close(fig2)

    # -- F.69b — C-9: original profile/hist unpack bug -------------------

    def test_f69b_profile_hist_range_minmax_no_unpack_error(self):
        """The original audit bug: range='minmax' on faceted profile/hist
        raised 'too many values to unpack' (string where a (lo,hi) tuple was
        expected). Must now render cleanly (minmax is the default strategy
        there)."""
        df = _two_cluster_df()
        # profile
        fig, ax, st = DFDraw(df).profile('y:x', bins=10, range="minmax")
        assert ax is not None
        plt.close('all')
        # hist
        fig, ax, st = DFDraw(df).hist('x', bins=10, range="minmax")
        assert ax is not None
        plt.close('all')

    # -- F.70 — C-4: _get_suptitle public-API helper ---------------------

    def test_f70_get_suptitle_live_path(self):
        """_get_suptitle returns the public-API suptitle text for a figure
        that actually has a suptitle (the live path, not just the empty
        fallback), and '' when absent."""
        fig = plt.figure()
        fig.suptitle("Calibration QA")
        assert _get_suptitle(fig) == "Calibration QA"
        plt.close(fig)
        fig2 = plt.figure()
        assert _get_suptitle(fig2) == ""
        plt.close(fig2)

    # -- F.71 — C-9 FIX1: scatter range REMOVES out-of-range points ------

    def test_f71_scatter_range_removes_out_of_range_points(self):
        """Phase 13.46.DF FIX1: scatter range= must DROP out-of-range points
        (point filter), not merely clip the view. Consistent with hist/profile
        range= excluding points from binning.

        With explicit outliers, range='percentile_99' (and an explicit tuple)
        must reduce the plotted point count; range='minmax' (window == full
        data) must remove nothing."""
        rs = np.random.RandomState(0)
        x = np.concatenate([rs.normal(0, 1, 990),
                            np.array([100., -100, 200, -200, 300,
                                      150, -150, 180, -180, 120])])
        df = pd.DataFrame({'x': x, 'y': rs.normal(0, 1, 1000),
                           'c': rs.rand(1000)})
        n_in = len(df)

        # percentile_99 drops outliers
        fig, ax, st = DFDraw(df).scatter('y:x', range="percentile_99")
        n_pct = len(ax.collections[0].get_offsets())
        assert n_pct < n_in, "percentile_99 must remove out-of-range points"
        plt.close(fig)

        # explicit tuple drops points outside the window
        fig, ax, st = DFDraw(df).scatter('y:x', range=((-3, 3), (-3, 3)))
        n_tup = len(ax.collections[0].get_offsets())
        assert n_tup < n_in, "explicit-tuple range must remove out-of-range points"
        plt.close(fig)

        # minmax window == full data → nothing removed
        df_in = df.iloc[:990]  # inliers only, no extreme outliers
        fig, ax, st = DFDraw(df_in).scatter('y:x', range="minmax")
        assert len(ax.collections[0].get_offsets()) == len(df_in), \
            "minmax must not remove any points"
        plt.close(fig)

        # parallel color array stays aligned through filtering (no length crash)
        fig, ax, st = DFDraw(df).scatter('y:x', color='c', range="percentile_99")
        assert len(ax.collections[0].get_offsets()) < n_in
        plt.close(fig)
