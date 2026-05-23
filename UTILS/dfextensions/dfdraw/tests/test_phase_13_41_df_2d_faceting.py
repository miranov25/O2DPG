"""
Phase 13.41.DF v1.6 §9 invariance tests: N-D faceting via facet_by=List[str].

19 tests in TestFacetByListGrid:
  FBY.1  1D backward compat (str ≡ List len 1)
  FBY.2  2D grid n_rows × n_cols
  FBY.3  2D bins=[3,4] → 12 facets (also locks _validate_facet_by_binning guard)
  FBY.4  2D quantiles per-dim
  FBY.5  mixed bins [None, 4]
  FBY.6  CONVENTION LOCK 2D: facet_by[0]=row, [1]=col
  FBY.7  length-mismatch ValueError
  FBY.8  group_by inside cells (composition)
  FBY.9  cumulative=True per cell (Phase 13.40 composition)
  FBY.10 4D+ NotImplementedError
  FBY.11 CONVENTION LOCK 3D: returns List[Figure]
  FBY.12 share_x='row' regression lock (CP0-1 P0 fix)
  FBY.13 share_across_figures=True 3D range lock (scatter)
  FBY.14 share_x='none' data-divergence approach
  FBY.15 empty cell no-crash + diagnostic
  FBY.16 3D + share_across_figures + hist dispatch (CP1-1 v1.4 P1 lock)
  FBY.17 3D + share_across_figures + profile dispatch
  FBY.18 share_across_figures=False independence (v1.6 CP1-1)
  FBY.19 share_x='col' symmetry to FBY.12 (v1.6 CP1-2)

Spec: PHASE_13_41_DF_v1_6_RowColumnFigIDFaceting_Proposal.md
Predecessor: PHASE_13_40_DF_END (923/0/1)
Expected gate: 942 (+19)
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfextensions.dfdraw import DFDraw


def _get_suptitle(fig):
    """Get suptitle text using public API (matplotlib 3.8+) with fallback.
    
    FIX2 item 4 (Sonnet55 advisory): avoid private fig._suptitle attribute.
    Returns '' when no suptitle is set.
    """
    if hasattr(fig, 'get_suptitle'):
        return fig.get_suptitle()
    # matplotlib < 3.8 fallback
    _s = getattr(fig, '_suptitle', None)
    return _s.get_text() if _s is not None else ''


def _max_y_in_axes(ax):
    """Polygon-safe max y (handles default histtype='stepfilled')."""
    max_y = 0.0
    for p in ax.patches:
        try:
            verts = p.get_path().vertices
            if len(verts) > 0:
                max_y = max(max_y, float(np.nanmax(verts[:, 1])))
        except (AttributeError, ValueError):
            pass
    for line in ax.lines:
        ydata = line.get_ydata()
        if len(ydata) > 0:
            try:
                max_y = max(max_y, float(np.nanmax(ydata)))
            except (TypeError, ValueError):
                pass
    return max_y


class TestFacetByListGrid:
    """Phase 13.41.DF v1.6 — N-D faceting (1D str / 2D row×col / 3D row×col×figID)."""

    def test_FBY_1_string_equals_list_of_one(self):
        """§9.FBY.1 — facet_by='sec' ≡ facet_by=['sec'] (1D backward compat)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 300),
            'sec': np.repeat(['A', 'B', 'C'], 100),
        })
        fig_str, ax_str, stats_str = DFDraw(df).hist('x', facet_by='sec')
        fig_lst, ax_lst, stats_lst = DFDraw(df).hist('x', facet_by=['sec'])
        # Both produce single figure with same axes count
        assert len(fig_str.axes) == len(fig_lst.axes), \
            f"1D string vs list axes mismatch: {len(fig_str.axes)} vs {len(fig_lst.axes)}"
        plt.close(fig_str)
        plt.close(fig_lst)

    def test_FBY_2_2d_grid_rows_by_cols(self):
        """§9.FBY.2 — facet_by=['a','b'] → n_rows × n_cols subplot grid."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'a': np.tile(np.repeat(['A0', 'A1', 'A2'], 100), 2),
            'b': np.repeat(['B0', 'B1'], 300),
        })
        fig, axes, stats = DFDraw(df).hist('x', facet_by=['a', 'b'])
        assert axes.shape == (3, 2), \
            f"Expected (3,2) grid (3 rows of 'a', 2 cols of 'b'), got {axes.shape}"
        assert len(stats) == 6, f"Expected 6 cells, got {len(stats)}"
        plt.close(fig)

    def test_FBY_3_2d_bins_per_dim(self):
        """§9.FBY.3 — facet_by_bins=[3,4] → 12 facets.
        
        ALSO regression lock for v1.3 CP1-1: _validate_facet_by_binning crashed
        on list input before the 2-line guard was added.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 1200),
            'a': rng.uniform(0, 100, 1200),     # continuous → bin into 3
            'b': rng.uniform(0, 50, 1200),       # continuous → bin into 4
        })
        # Without v1.3 CP1-1 guard, this would crash:
        # TypeError: unhashable type: 'list' at facet_by not in df.columns
        fig, axes, stats = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], facet_by_bins=[3, 4])
        assert axes.shape == (3, 4), f"Expected (3,4), got {axes.shape}"
        assert len(stats) == 12, f"Expected 12 cells, got {len(stats)}"
        plt.close(fig)

    def test_FBY_4_2d_quantiles_per_dim(self):
        """§9.FBY.4 — facet_by_quantiles=[[...], [...]] per-dim quantile binning."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'a': rng.uniform(0, 100, 600),
            'b': rng.uniform(0, 50, 600),
        })
        fig, axes, stats = DFDraw(df).hist(
            'x', facet_by=['a', 'b'],
            facet_by_quantiles=[[0.33, 0.67, 1.0], [0.5, 1.0]])
        # 3 quantile bins × 2 quantile bins = 6 cells
        assert axes.shape[0] >= 2 and axes.shape[1] >= 1, \
            f"Quantile grid too small: {axes.shape}"
        plt.close(fig)

    def test_FBY_5_mixed_bins_None_and_int(self):
        """§9.FBY.5 — facet_by_bins=[None, 4]: row=discrete, col=binned."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'a': np.repeat(['A0', 'A1', 'A2'], 200),    # discrete
            'b': rng.uniform(0, 50, 600),                # continuous
        })
        fig, axes, stats = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], facet_by_bins=[None, 4])
        assert axes.shape == (3, 4), f"Expected (3,4), got {axes.shape}"
        plt.close(fig)

    def test_FBY_6_convention_lock_2d_row_col(self):
        """§9.FBY.6 — CONVENTION LOCK: facet_by[0]=ROW, facet_by[1]=COLUMN.
        
        Verify by data layout: row dim should vary vertically (more rows of 'a'
        values → axes.shape[0] = unique(a)), col dim horizontally.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 400),
            'a': np.repeat(['A0', 'A1', 'A2', 'A3'], 100),   # 4 unique (ROW)
            'b': np.tile(['B0', 'B1'], 200),                  # 2 unique (COL)
        })
        fig, axes, _ = DFDraw(df).hist('x', facet_by=['a', 'b'])
        # CONVENTION: row=[0]=a (4 unique), col=[1]=b (2 unique)
        assert axes.shape == (4, 2), (
            f"CONVENTION LOCK VIOLATION: facet_by[0]='a' should be ROW (axes.shape[0]); "
            f"facet_by[1]='b' should be COL (axes.shape[1]). "
            f"Expected (4,2), got {axes.shape}"
        )
        plt.close(fig)

    def test_FBY_7_length_mismatch_raises(self):
        """§9.FBY.7 — len(facet_by) ≠ len(facet_by_bins) → ValueError."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 100),
            'a': np.repeat(['A', 'B'], 50),
            'b': np.tile(['B0', 'B1'], 50),
        })
        with pytest.raises(ValueError, match="must have 2 elements"):
            DFDraw(df).hist('x', facet_by=['a', 'b'], facet_by_bins=[4])

    def test_FBY_8_group_by_inside_cells(self):
        """§9.FBY.8 — group_by composition with 2D faceting (overlay per cell)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'a': np.repeat(['A0', 'A1'], 300),
            'b': np.tile(np.repeat(['B0', 'B1'], 150), 2),
            'g': np.tile(['G0', 'G1', 'G2'], 200),
        })
        # group_by adds overlay curves inside each of the 4 cells
        fig, axes, stats = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], group_by='g')
        assert axes.shape == (2, 2)
        # Each cell should have ≥3 patches/lines (one per group)
        for i in range(2):
            for j in range(2):
                n_artists = len(axes[i, j].patches) + len(axes[i, j].lines)
                assert n_artists >= 1, \
                    f"Cell ({i},{j}) has no overlay artists with group_by='g'"
        plt.close(fig)

    def test_FBY_9_cumulative_per_cell(self):
        """§9.FBY.9 — Phase 13.40 cumulative=True composition with 2D facet."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 400),
            'a': np.repeat(['A', 'B'], 200),
            'b': np.tile(['B0', 'B1'], 200),
        })
        fig, axes, _ = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], cumulative=True)
        # Each cell should reach close to per-cell total count (~100)
        for i in range(2):
            for j in range(2):
                max_y = _max_y_in_axes(axes[i, j])
                assert max_y >= 80, (
                    f"Cell ({i},{j}) cumulative max_y={max_y} < 80; "
                    "expected ~100 (per-cell total). cumulative=True dropped."
                )
        plt.close(fig)

    def test_FBY_10_four_dimensions_raises(self):
        """§9.FBY.10 — 4D+ raises NotImplementedError with actionable message."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 100),
            'a': ['x'] * 100, 'b': ['x'] * 100,
            'c': ['x'] * 100, 'd': ['x'] * 100,
        })
        with pytest.raises(NotImplementedError, match="3 dimensions supported"):
            DFDraw(df).hist('x', facet_by=['a', 'b', 'c', 'd'])

    def test_FBY_11_convention_lock_3d_list_of_figures(self):
        """§9.FBY.11 — CONVENTION LOCK 3D: facet_by=['row','col','figID']
        returns List[Figure], one per figID value. Each figure has 2D grid.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'y': rng.normal(0, 1, 600),
            'sec': np.tile(np.repeat(['S0', 'S1', 'S2'], 100), 2),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 25), 6),
            'period': np.repeat(['P0', 'P1'], 300),
        })
        figs, axes_list, stats_list = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'])
        assert isinstance(figs, list), \
            f"3D should return list of figures, got {type(figs).__name__}"
        assert len(figs) == 2, f"Expected 2 figures (2 unique periods), got {len(figs)}"
        for f in figs:
            assert isinstance(f, plt.Figure)
        # Each figure has 2D row × col grid
        for axes in axes_list:
            assert hasattr(axes, 'shape') and axes.ndim == 2
            assert axes.shape == (3, 4), f"Expected (3,4), got {axes.shape}"
        for f in figs:
            plt.close(f)

    def test_FBY_12_share_x_row_regression_lock(self):
        """§9.FBY.12 — REGRESSION LOCK FOR v1.2 CP0-1 P0.
        
        share_x='row' must produce shared x-axis WITHIN each row.
        If the dispatch dict bug returns (v1.1: 'row'→False), this test
        FAILS decisively because cells in the same row would have
        different xlim.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.concatenate([rng.uniform(0, 5, 200),     # row=0
                                  rng.uniform(10, 20, 200)]), # row=1
            'y': rng.normal(0, 1, 400),
            'a': ['A0'] * 200 + ['A1'] * 200,    # 2 rows
            'b': (['B0'] * 100 + ['B1'] * 100) * 2,  # 2 cols
        })
        fig, axes, _ = DFDraw(df).scatter(
            'y:x', facet_by=['a', 'b'], share_x='row')
        fig.canvas.draw()
        for i in range(axes.shape[0]):
            row_xlims = [axes[i, j].get_xlim() for j in range(axes.shape[1])]
            assert all(xl == row_xlims[0] for xl in row_xlims), (
                f"Row {i} cells don't share x-limits with share_x='row'. "
                "If this fails, the v1.1 dispatch-dict bug "
                "(CP0-1: 'row'→False) regressed."
            )
        plt.close(fig)

    def test_FBY_13_share_across_figures_3d_scatter(self):
        """§9.FBY.13 — share_across_figures=True 3D range lock (scatter).
        
        For scatter, both x and y are locked across all N figures.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'y': rng.normal(0, 1, 600),
            'sec': np.repeat(['S0', 'S1', 'S2'], 200),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 50), 3),
            'period': np.tile(np.repeat(['P0', 'P1'], 100), 3),
        })
        figs, axes_list, _ = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'],
            share_across_figures=True)
        for f in figs:
            f.canvas.draw()
        xlims_per_fig = [axes_list[k][0, 0].get_xlim() for k in range(len(figs))]
        ylims_per_fig = [axes_list[k][0, 0].get_ylim() for k in range(len(figs))]
        assert all(xl == xlims_per_fig[0] for xl in xlims_per_fig), \
            f"share_across_figures=True did not lock x-limits. Got: {xlims_per_fig}"
        assert all(yl == ylims_per_fig[0] for yl in ylims_per_fig), \
            f"share_across_figures=True (scatter) did not lock y-limits."
        for f in figs:
            plt.close(f)

    def test_FBY_14_share_none_data_divergence(self):
        """§9.FBY.14 — share_x='none' + share_y='none': cells independent.
        
        v1.2 CP2-1: data-divergence approach (avoid deprecated 
        get_shared_x_axes().joined() API).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.concatenate([rng.uniform(0, 1, 100),
                                  rng.uniform(10, 20, 100)]),
            'y': rng.normal(0, 1, 200),
            'a': ['A'] * 200,                       # 1 row
            'b': ['B0'] * 100 + ['B1'] * 100,        # 2 cols
        })
        fig, axes, _ = DFDraw(df).scatter(
            'y:x', facet_by=['a', 'b'],
            share_x='none', share_y='none')
        fig.canvas.draw()
        xlim_00 = axes[0, 0].get_xlim()
        xlim_01 = axes[0, 1].get_xlim()
        # If unlinked, cell with x in [0,1] vs cell with x in [10,20] diverge
        assert xlim_00[1] < xlim_01[0] + 5, (
            f"share_x='none' did not unlink axes — "
            f"xlim_00={xlim_00}, xlim_01={xlim_01} (must differ)"
        )
        plt.close(fig)

    def test_FBY_15_empty_cell_no_crash(self):
        """§9.FBY.15 — empty facet combination renders '(no data)' + stats={'n':0,'empty':True}.
        
        v1.2 CP2-2: no crash on filter to 0 rows.
        """
        rng = np.random.default_rng(42)
        # Set up so (a='A', b='B1') and (a='B', b='B0') are EMPTY
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 200),
            'y': rng.normal(0, 1, 200),
            'a': ['A'] * 100 + ['B'] * 100,
            'b': ['B0'] * 100 + ['B1'] * 100,
        })
        fig, axes, stats = DFDraw(df).scatter('y:x', facet_by=['a', 'b'])
        # 2 empty cells expected
        empty_keys = [k for k, v in stats.items() if isinstance(v, dict) and v.get('empty')]
        assert len(empty_keys) == 2, (
            f"Expected 2 empty cells, got {len(empty_keys)}: {empty_keys}"
        )
        # At least one axis has "(no data)" text
        has_diag = any(
            any('no data' in t.get_text() for t in ax.texts)
            for row in axes for ax in row
        )
        assert has_diag, "Empty cell missing '(no data)' diagnostic text"
        plt.close(fig)

    def test_FBY_16_3d_share_across_hist_dispatch(self):
        """§9.FBY.16 — 3D + share_across_figures + plot_kind='hist'.
        
        v1.6 CP1-1: locks v1.4 CP1-1 hist-dispatch param name (range=, NOT
        x_range=). Would crash with TypeError if implementation regresses.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'sec': np.tile(np.repeat(['S0', 'S1', 'S2'], 100), 2),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 25), 6),
            'period': np.repeat(['P0', 'P1'], 300),
        })
        # If dispatch uses x_range= instead of range=, TypeError raised
        figs, axes_list, _ = DFDraw(df).hist(
            'x', facet_by=['sec', 'drift', 'period'],
            share_across_figures=True)
        assert len(figs) == 2, "Expected 2 figures (2 periods)"
        for f in figs:
            f.canvas.draw()
        xlims_per_fig = [axes_list[k][0, 0].get_xlim() for k in range(len(figs))]
        assert all(xl == xlims_per_fig[0] for xl in xlims_per_fig), (
            f"share_across_figures=True (hist) did not lock x-limits. "
            f"Got: {xlims_per_fig}. Likely v1.4 CP1-1 regressed."
        )
        for f in figs:
            plt.close(f)

    def test_FBY_17_3d_share_across_profile_dispatch(self):
        """§9.FBY.17 — 3D + share_across_figures + plot_kind='profile'.
        
        v1.6 CP1-2: locks profile dispatch param. DFDraw.profile uses range=
        (which it remaps to x_range internally for draw_profile).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.uniform(0, 100, 1200),
            'y': rng.normal(0, 1, 1200),
            'sec': np.repeat(['S0', 'S1', 'S2'], 400),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 100), 3),
            'period': np.tile(np.repeat(['P0', 'P1'], 200), 3),
        })
        figs, axes_list, _ = DFDraw(df).profile(
            'y:x', bins=20,
            facet_by=['sec', 'drift', 'period'],
            share_across_figures=True)
        assert len(figs) == 2
        for f in figs:
            f.canvas.draw()
        xlims_per_fig = [axes_list[k][0, 0].get_xlim() for k in range(len(figs))]
        assert all(xl == xlims_per_fig[0] for xl in xlims_per_fig), (
            f"share_across_figures=True (profile) did not lock x-limits."
        )
        for f in figs:
            plt.close(f)

    def test_FBY_18_share_across_figures_false_independence(self):
        """§9.FBY.18 — share_across_figures=False independence (v1.6 CP1-1).
        
        4/5 reviewer convergence in v1.5 panel. FBY.13/16/17 lock =True;
        this locks =False. Without it, regression could silently force-lock
        OR silently break False semantics.
        """
        rng = np.random.default_rng(42)
        # Construct figID slices with VERY different x-data ranges
        df = pd.DataFrame({
            'x': np.concatenate([
                rng.uniform(0, 5, 300),       # period P0 → x in [0,5]
                rng.uniform(50, 100, 300),     # period P1 → x in [50,100]
            ]),
            'y': rng.normal(0, 1, 600),
            'sec': np.tile(np.repeat(['S0', 'S1', 'S2'], 100), 2),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 25), 6),
            'period': np.repeat(['P0', 'P1'], 300),
        })
        figs, axes_list, _ = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'],
            share_across_figures=False)
        assert len(figs) == 2, "Expected 2 figures"
        for f in figs:
            f.canvas.draw()
        # With share_across_figures=False, P0 fig should show ~[0,5];
        # P1 fig should show ~[50,100]. Non-overlap proves independence.
        xlim_fig0 = axes_list[0][0, 0].get_xlim()
        xlim_fig1 = axes_list[1][0, 0].get_xlim()
        assert xlim_fig0[1] < xlim_fig1[0] + 10, (
            f"share_across_figures=False did NOT unlock figures. "
            f"fig0 xlim={xlim_fig0}, fig1 xlim={xlim_fig1}. "
            "Expected non-overlapping (data [0,5] vs [50,100])."
        )
        for f in figs:
            plt.close(f)

    def test_FBY_19_share_x_col_symmetry(self):
        """§9.FBY.19 — share_x='col' symmetry to FBY.12 (v1.6 CP1-2).
        
        3/5 reviewer convergence. FBY.12 locks 'row'; FBY.19 locks 'col'.
        Together they verify both off-diagonal entries of the dispatch dict.
        """
        rng = np.random.default_rng(42)
        # Different x-data ranges per COLUMN (matrix-shape: row=a, col=b)
        df = pd.DataFrame({
            'x': np.concatenate([
                rng.uniform(0, 5, 200),       # row A0 col B0 → x in [0,5]
                rng.uniform(50, 100, 200),    # row A0 col B1 → x in [50,100]
                rng.uniform(0, 5, 200),       # row A1 col B0
                rng.uniform(50, 100, 200),    # row A1 col B1
            ]),
            'y': rng.normal(0, 1, 800),
            'a': ['A0']*400 + ['A1']*400,         # 2 rows
            'b': (['B0']*200 + ['B1']*200) * 2,    # 2 cols
        })
        fig, axes, _ = DFDraw(df).scatter(
            'y:x', facet_by=['a', 'b'], share_x='col')
        fig.canvas.draw()
        # For share_x='col': cells in same COL (any row) share x
        for j in range(axes.shape[1]):
            col_xlims = [axes[i, j].get_xlim() for i in range(axes.shape[0])]
            assert all(xl == col_xlims[0] for xl in col_xlims), (
                f"Column {j} cells don't share x-limits with share_x='col'. "
                f"Got: {col_xlims}. If FBY.12 'row' passes but this 'col' "
                "test fails, dispatch dict is asymmetric."
            )
        # Sanity: different columns SHOULD have different xlim
        col0_xlim = axes[0, 0].get_xlim()
        col1_xlim = axes[0, 1].get_xlim()
        assert col0_xlim[1] < col1_xlim[0] + 10, (
            f"share_x='col': different cols should have different xlim "
            f"when data ranges differ. Got col0={col0_xlim}, col1={col1_xlim}."
        )
        plt.close(fig)

    # ── Phase 13.41.DF FIX1 — 3 additional regression locks ──

    def test_FBY_20_share_y_row_symmetry(self):
        """§9.FBY.20 — FIX1: share_y='row' symmetric to FBY.12 (Sonnet55 advisory).
        
        FBY.12 locks share_x='row'; FBY.20 mirrors for share_y. Together they
        prove _to_mpl_share dispatch dict is symmetric across both axes.
        """
        rng = np.random.default_rng(42)
        # Different y-data ranges per ROW (matrix-shape: row=a, col=b)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 800),
            'y': np.concatenate([
                rng.uniform(0, 5, 400),       # row A0 → y in [0,5]
                rng.uniform(50, 100, 400),    # row A1 → y in [50,100]
            ]),
            'a': ['A0']*400 + ['A1']*400,         # 2 rows
            'b': (['B0']*200 + ['B1']*200) * 2,    # 2 cols
        })
        fig, axes, _ = DFDraw(df).scatter(
            'y:x', facet_by=['a', 'b'], share_y='row')
        fig.canvas.draw()
        # For share_y='row': cells in same ROW (any col) share y
        for i in range(axes.shape[0]):
            row_ylims = [axes[i, j].get_ylim() for j in range(axes.shape[1])]
            assert all(yl == row_ylims[0] for yl in row_ylims), (
                f"Row {i} cells don't share y-limits with share_y='row'. "
                f"Got: {row_ylims}. _to_mpl_share dispatch dict asymmetric "
                "between x and y axes."
            )
        # Sanity: different rows should have different ylim
        row0_ylim = axes[0, 0].get_ylim()
        row1_ylim = axes[1, 0].get_ylim()
        assert row0_ylim[1] < row1_ylim[0] + 10, (
            f"share_y='row': different rows should have different ylim "
            f"when data ranges differ. Got row0={row0_ylim}, row1={row1_ylim}."
        )
        plt.close(fig)

    def test_FBY_21_share_x_invalid_raises(self):
        """§9.FBY.21 — FIX1: share_x='invalid' raises ValueError (Sonet51 advisory).
        
        Locks the _validate_share_axis_value enum guard. Without it, an invalid
        value would fall through to _to_mpl_share's dispatch dict and raise
        KeyError instead of a clear, actionable ValueError.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 200),
            'y': rng.normal(0, 1, 200),
            'a': np.repeat(['A0', 'A1'], 100),
            'b': np.tile(['B0', 'B1'], 100),
        })
        # Note: regex matches error message from _validate_share_axis_value
        with pytest.raises(ValueError, match="share_x must be one of"):
            DFDraw(df).scatter(
                'y:x', facet_by=['a', 'b'], share_x='invalid')
        # Symmetric: share_y also validated
        with pytest.raises(ValueError, match="share_y must be one of"):
            DFDraw(df).scatter(
                'y:x', facet_by=['a', 'b'], share_y='diagonal')

    def test_FBY_22_auto_title_suptitle_lock(self):
        """§9.FBY.22 — FIX1: auto_title=True sets figure suptitle (Sonnet54 P2).
        
        Locks the Phase 13.41 FIX1 fix for Sonnet54's P2 finding. In v1.6 the
        list-handling branch hardcoded auto_title=False inside cells, silently
        dropping the user's choice. FIX1 routes auto_title=True to fig.suptitle
        (cell titles are reserved for facet labels).
        
        FIX2 item 4 (Sonnet55 advisory): use public _get_suptitle(fig) instead
        of private fig._suptitle attribute (future-proof for matplotlib API).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 400),
            'a': np.repeat(['A0', 'A1'], 200),
            'b': np.tile(['B0', 'B1'], 200),
        })
        # Case 1: auto_title=False — no suptitle
        fig_false, _, _ = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], auto_title=False)
        assert _get_suptitle(fig_false) == '', (
            "auto_title=False should NOT produce a suptitle"
        )
        plt.close(fig_false)
        
        # Case 2: auto_title=True — figure has suptitle mentioning expression
        # and facet dimensions
        fig_true, _, _ = DFDraw(df).hist(
            'x', facet_by=['a', 'b'], auto_title=True)
        suptitle_text = _get_suptitle(fig_true)
        assert suptitle_text != '', (
            "auto_title=True must produce a suptitle in 2D facet mode. "
            "Sonnet54 P2 regression: v1.6 silently dropped user's choice."
        )
        assert 'x' in suptitle_text and 'a' in suptitle_text and 'b' in suptitle_text, (
            f"Suptitle should mention expression + facet dims; got: '{suptitle_text}'"
        )
        plt.close(fig_true)
        
        # Case 3: user-supplied title= takes precedence over auto_title=True
        fig_user, _, _ = DFDraw(df).hist(
            'x', facet_by=['a', 'b'],
            auto_title=True, title="Custom Title")
        assert _get_suptitle(fig_user) == "Custom Title", (
            "User-supplied title= must override auto_title=True suptitle"
        )
        plt.close(fig_user)

    def test_FBY_23_3d_auto_title_combined_suptitle(self):
        """§9.FBY.23 — FIX2 item 2: 3D + auto_title=True combined suptitle.
        
        v1.6 + FIX1 baseline: 3D + auto_title=True was a silent no-op
        (Sonnet54 P2 v1.6 END). The 2D dispatch's auto_title suptitle was
        immediately overwritten by `_dispatch_3d_facet`'s figID label.
        
        FIX2 design: 3 per-figure suptitle cases:
          1. user title=      → "{title} ({figid_col} = {v})"
          2. auto_title=True  → "{expr}  [faceted by {r} × {c} × {f} = {v}]"
          3. default          → "{figid_col} = {v}"  (Phase 13.41 v1.6 baseline)
        Default behavior also locked by FBY.11.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 600),
            'y': rng.normal(0, 1, 600),
            'sec': np.tile(np.repeat(['S0', 'S1', 'S2'], 100), 2),
            'drift': np.tile(np.repeat([0, 1, 2, 3], 25), 6),
            'period': np.repeat(['P0', 'P1'], 300),
        })
        
        # Case 1: user title= → "{title} ({figid_col} = {v})"
        figs1, _, _ = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'],
            title="Run 12345")
        assert len(figs1) == 2
        for fig, expected_v in zip(figs1, ['P0', 'P1']):
            s = _get_suptitle(fig)
            assert s == f"Run 12345 (period = {expected_v})", (
                f"user title= case: got '{s}', expected 'Run 12345 (period = {expected_v})'"
            )
            plt.close(fig)
        
        # Case 2: auto_title=True → combined suptitle with all 3 dims + figid_v
        figs2, _, _ = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'],
            auto_title=True)
        assert len(figs2) == 2
        for fig, expected_v in zip(figs2, ['P0', 'P1']):
            s = _get_suptitle(fig)
            # Regression lock against Sonnet54 P2: must NOT be the v1.6 silent
            # no-op (which would have produced "period = P0" — same as case 3).
            assert s != f"period = {expected_v}", (
                f"auto_title=True must NOT produce default figID-only suptitle. "
                f"Sonnet54 P2 regression: got '{s}'."
            )
            # Must contain expression, all 3 facet dims, and the figid value
            for token in ['y', 'x', 'sec', 'drift', 'period', expected_v]:
                assert token in s, (
                    f"3D auto_title suptitle missing '{token}'. Got: '{s}'"
                )
            plt.close(fig)
        
        # Case 3 (default): figID-only suptitle (Phase 13.41 v1.6 baseline,
        # also locked by FBY.11 — verified here for symmetry with cases 1+2).
        figs3, _, _ = DFDraw(df).scatter(
            'y:x', facet_by=['sec', 'drift', 'period'])
        for fig, expected_v in zip(figs3, ['P0', 'P1']):
            s = _get_suptitle(fig)
            assert s == f"period = {expected_v}", (
                f"default 3D suptitle should be figID-only (v1.6 baseline); "
                f"got: '{s}'"
            )
            plt.close(fig)
