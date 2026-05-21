"""
Phase 13.39.DF §9 invariance tests: 2D profile + time axis + scatter3D.

24 tests across 3 classes:
  TestProfile2D       (9)  — z:y:x → pcolormesh; backward-compat + group_by lock
  TestTimeAxis        (7)  — time_format= pre-conversion; datetime64[ns] lock (CP1-4)
  TestScatter3D       (8)  — type='scatter3d' + z:y:x; reuse Phase 13.38 helpers

Sonet50 panel CP fixes (v1.1 → v1.2):
  CP1-1: TA.5 realistic timestamps (epoch-0 made both paths < 100000)
  CP1-2: scipy range pseudocode 3-level nested → 1-level
  CP1-3: SC3D.6 lock all 3 means to 1e-9 (not just mean_x)
  CP1-4: time_format datetime64[ns] auto-detect (avoid pd.to_datetime unit=s crash)
  CP1-5: DFDraw.profile dispatch insertion point specified
  CP1-6: scipy required (drop fallback)
  CP2-1: group_by + profile2d/scatter3d raises (§9.P2D.10, §9.SC3D.7)
  CP2-2: same=True onto non-3D axes raises (§9.SC3D.8)

Spec: PHASE_13_39_DF_v1_2_Profile2D_TimeAxis_Scatter3D_Proposal.md
Predecessor: PHASE_13_38_DF_END (889/0/1)
Expected gate: 913 (+24)
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from dfextensions.dfdraw import DFDraw


# ====================================================================== #
# TestProfile2D (9 tests) — z:y:x → pcolormesh                            #
# ====================================================================== #

class TestProfile2D:
    """Phase 13.39.DF Item 1 — 2D profile via z:y:x expression."""

    def _make_df(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'x': rng.uniform(0, 10, n),
            'y': rng.uniform(0, 5, n),
            'z': rng.normal(0, 1, n),
            'z_raw': rng.normal(0, 1, n),
        })

    def test_P2D_1_quadmesh_rendered(self):
        """§9.P2D.1 — z:y:x → QuadMesh present on ax."""
        df = self._make_df()
        fig, ax, _ = DFDraw(df).profile("z:y:x", bins=[20, 10])
        qm = [c for c in ax.collections if isinstance(c, mc.QuadMesh)]
        assert len(qm) >= 1, f"Expected QuadMesh in ax.collections, got {[type(c).__name__ for c in ax.collections]}"
        plt.close(fig)

    def test_P2D_2_per_cell_mean_correctness(self):
        """§9.P2D.2 — per-cell mean ≡ binned_statistic_2d ground truth (1e-9).

        CP1-2 fix: spec previously used pd.cut for ground truth (edge mismatch).
        v1.2 uses binned_statistic_2d directly so render + ground-truth use
        the same edges.
        """
        from scipy.stats import binned_statistic_2d
        df = self._make_df(n=2000)
        fig, ax, stats = DFDraw(df).profile("z:y:x", bins=[20, 10])

        # Ground truth from same scipy call
        z_expected, _, _, _ = binned_statistic_2d(
            df['x'].values, df['y'].values, df['z'].values,
            statistic='mean', bins=[20, 10],
        )
        # Rendered values
        qm = next(c for c in ax.collections if isinstance(c, mc.QuadMesh))
        z_rendered = qm.get_array()

        # pcolormesh array can be a 1D masked array (flattened) — handle both
        if z_rendered.ndim == 1:
            # Compare flatten-to-flatten, skipping NaN cells (no data)
            expected_flat = z_expected.T.flatten()  # T because we render z_mean.T
        else:
            expected_flat = z_expected.T.flatten()
            z_rendered = z_rendered.flatten()

        # Compare only finite cells (NaN cells: no data, expected to be NaN both sides)
        mask = np.isfinite(expected_flat) & np.isfinite(np.asarray(z_rendered))
        # Cast both to plain float arrays for comparison
        np.testing.assert_allclose(
            np.asarray(z_rendered)[mask], expected_flat[mask],
            atol=1e-9, err_msg="Rendered cell values differ from binned_statistic_2d"
        )
        plt.close(fig)

    def test_P2D_3_min_entries_masks_low_count_cells(self):
        """§9.P2D.3 — min_entries=N masks low-count cells (set to NaN)."""
        df = self._make_df(n=500)
        # Use fine binning so many cells have few points
        fig, ax, stats = DFDraw(df).profile(
            "z:y:x", bins=[50, 50], min_entries_2d=10,
        )
        # stats reports masked count
        assert 'n_masked_cells' in stats
        # With 500 points / 2500 cells / mean 0.2 pts/cell, most are masked
        assert stats['n_masked_cells'] >= 0
        # All non-NaN rendered cells had >= 10 points
        z_count = stats['z_count']
        qm = next(c for c in ax.collections if isinstance(c, mc.QuadMesh))
        z_rendered = np.asarray(qm.get_array())
        # Flatten both: rendered uses .T orientation; count uses original
        count_flat = z_count.T.flatten()
        z_flat = z_rendered.flatten() if z_rendered.ndim > 0 else z_rendered
        finite_idx = np.where(np.isfinite(z_flat))[0]
        if len(finite_idx) > 0:
            assert (count_flat[finite_idx] >= 10).all(), \
                "Rendered (non-NaN) cells have count < min_entries"
        plt.close(fig)

    def test_P2D_4_dfeval_expression_for_z(self):
        """§9.P2D.4 — z=df.eval() expression differs from z=column."""
        df = self._make_df()
        fig_col, ax_col, _ = DFDraw(df).profile("z_raw:y:x", bins=[10, 5])
        fig_eval, ax_eval, _ = DFDraw(df).profile("abs(z_raw):y:x", bins=[10, 5])

        qm_col = next(c for c in ax_col.collections if isinstance(c, mc.QuadMesh))
        qm_eval = next(c for c in ax_eval.collections if isinstance(c, mc.QuadMesh))
        z_col = np.asarray(qm_col.get_array())
        z_eval = np.asarray(qm_eval.get_array())
        # abs(z) ≠ z in general
        assert not np.allclose(z_col, z_eval, equal_nan=True), \
            "Expression z=abs(z_raw) produced identical render to z=z_raw"
        plt.close(fig_col)
        plt.close(fig_eval)

    def test_P2D_5_bins_list_vs_bins2_shape_invariance(self):
        """§9.P2D.5 — bins=[nx,ny] ≡ bins=nx + bins2=ny (shape invariance)."""
        df = self._make_df()
        fig_a, ax_a, _ = DFDraw(df).profile("z:y:x", bins=[20, 10])
        fig_b, ax_b, _ = DFDraw(df).profile("z:y:x", bins=20, bins2=10)

        qa = next(c for c in ax_a.collections if isinstance(c, mc.QuadMesh))
        qb = next(c for c in ax_b.collections if isinstance(c, mc.QuadMesh))
        # Compare array shape (or length for flat arrays)
        arr_a = np.asarray(qa.get_array())
        arr_b = np.asarray(qb.get_array())
        assert arr_a.shape == arr_b.shape, \
            f"Shape mismatch: bins=[20,10] → {arr_a.shape}, bins=20+bins2=10 → {arr_b.shape}"
        plt.close(fig_a)
        plt.close(fig_b)

    def test_P2D_6_colorbar_labeled_single_key(self):
        """§9.P2D.6 — colorbar labeled (CP1-1 v1.1 fix: single-key value check)."""
        df = self._make_df()
        fig, ax, stats = DFDraw(df).profile("z:y:x", bins=10, clabel="<z> (cm)")
        # colorbar axis is added to fig
        assert len(fig.axes) >= 2, "Colorbar axis not present"
        # stats dict has the clabel value
        assert stats.get('clabel') == "<z> (cm)", \
            f"stats['clabel'] != '<z> (cm)', got {stats.get('clabel')!r}"
        plt.close(fig)

    def test_P2D_7_selection_applied_before_binning(self):
        """§9.P2D.7 — selection= reduces row count before binning."""
        df = self._make_df()
        df['z'] = df['z'] * 3  # widen z range so selection removes some
        _, _, stats_all = DFDraw(df).profile("z:y:x", bins=10)
        _, _, stats_sel = DFDraw(df).profile("z:y:x", bins=10, selection="abs(z)<2")
        assert stats_sel['n'] < stats_all['n'], \
            f"selection didn't reduce row count: all={stats_all['n']}, sel={stats_sel['n']}"

    def test_P2D_8_backward_compat_1d_profile(self):
        """§9.P2D.8 — "y:x" still routes to 1D profile (Line2D, not QuadMesh).

        Regression-lock: colon_count==2 dispatch must NOT break colon_count==1 path.
        """
        df = self._make_df()
        fig, ax, _ = DFDraw(df).profile("y:x", bins=20)
        # No QuadMesh — 1D profile uses Line2D + errorbar
        qm = [c for c in ax.collections if isinstance(c, mc.QuadMesh)]
        assert len(qm) == 0, \
            f"1D profile incorrectly produced QuadMesh: {len(qm)}"
        # Line2D present
        assert len(ax.get_lines()) >= 1, "1D profile missing Line2D"
        plt.close(fig)

    def test_P2D_10_group_by_raises_value_error(self):
        """§9.P2D.10 — CP2-1: group_by + colon-2 expression → ValueError.

        Scope §7 boundary lock: 2D profile + group_by is deferred.
        """
        df = self._make_df()
        df['g'] = np.repeat(['A', 'B'], 250)
        with pytest.raises(ValueError, match="(group_by|2D profile)"):
            DFDraw(df).profile("z:y:x", bins=[10, 5], group_by="g")


# ====================================================================== #
# TestTimeAxis (7 tests) — time_format= pre-conversion                    #
# ====================================================================== #

class TestTimeAxis:
    """Phase 13.39.DF Item 2 — time_format= kwarg with dtype auto-detect."""

    def _make_unix_df(self, n=100):
        """Realistic Unix timestamps (~1.7e9) — CP1-1 fix."""
        base = int(pd.Timestamp('2024-01-01').timestamp())  # ~1.704e9
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            't': base + np.arange(0, n * 36, 36, dtype=float)[:n],
            'y': rng.normal(0, 1, n),
        })

    def test_TA_1_profile_time_format_pct_HM(self):
        """§9.TA.1 — profile + time_format="%H:%M" → DateFormatter."""
        df = self._make_unix_df()
        fig, ax, _ = DFDraw(df).profile("y:t", bins=20, time_format="%H:%M")
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter)
        plt.close(fig)

    def test_TA_2_profile_time_format_auto(self):
        """§9.TA.2 — time_format="auto" → AutoDateFormatter."""
        df = self._make_unix_df()
        fig, ax, _ = DFDraw(df).profile("y:t", bins=20, time_format="auto")
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.AutoDateFormatter)
        plt.close(fig)

    def test_TA_3_default_no_time_format_backward_compat(self):
        """§9.TA.3 — time_format=None (default) → formatter unchanged."""
        df = self._make_unix_df()
        fig, ax, _ = DFDraw(df).profile("y:t", bins=20)
        # Neither DateFormatter nor AutoDateFormatter
        fmt = ax.xaxis.get_major_formatter()
        assert not isinstance(fmt, mdates.DateFormatter)
        assert not isinstance(fmt, mdates.AutoDateFormatter)
        plt.close(fig)

    def test_TA_4_scatter_time_format(self):
        """§9.TA.4 — scatter + time_format → DateFormatter."""
        df = self._make_unix_df()
        fig, ax, _ = DFDraw(df).scatter("y:t", time_format="%H:%M")
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter)
        plt.close(fig)

    def test_TA_5_hist_time_format_realistic_timestamps(self):
        """§9.TA.5 — CP1-1: hist + time_format with REALISTIC timestamps.

        Locks pre-conversion (N3): ticks must be matplotlib date numbers
        (~19700), NOT raw Unix seconds (~1.7e9). With epoch-0 data
        (np.arange(0, 3600, 36)), BOTH pre-converted (0.0) and raw (0) paths
        would pass `< 100000`, defeating the lock. Realistic 2024 timestamps
        (~1.7e9) make the lock work as intended.
        """
        df = self._make_unix_df()  # realistic timestamps ~1.7e9
        fig, ax, _ = DFDraw(df).hist("t", bins=20, time_format="%H:%M")
        fig.canvas.draw()
        ticks = ax.xaxis.get_majorticklocs()
        # Pre-converted: ~19700 (matplotlib date number for 2024-01-01)
        # Raw Unix: ~1.704e9 (would fail this assertion)
        assert len(ticks) > 0, "No ticks rendered"
        assert ticks[0] < 100000, (
            f"hist time_format pre-conversion failed: ticks[0]={ticks[0]:.1f} "
            f"(expected ~19700 matplotlib date number, NOT ~1.7e9 raw Unix). "
            "Pre-conversion (N3) was not applied before ax.hist()."
        )
        plt.close(fig)

    def test_TA_6_profile2d_x_axis_date_formatter(self):
        """§9.TA.6 — profile2d + time_format: x-axis DateFormatter, y-axis unchanged."""
        df = self._make_unix_df(n=500)
        df['y2'] = np.random.normal(0, 1, 500)
        df['z'] = np.random.normal(0, 1, 500)
        fig, ax, _ = DFDraw(df).profile(
            "z:y2:t", bins=[20, 5], time_format="%H:%M",
        )
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter), \
            "x-axis not DateFormatter for profile2d"
        # y-axis should NOT be DateFormatter (out of scope per §7)
        assert not isinstance(ax.yaxis.get_major_formatter(), mdates.DateFormatter), \
            "y-axis unexpectedly got DateFormatter"
        plt.close(fig)

    def test_TA_7_datetime64_column_no_crash(self):
        """§9.TA.7 — CP1-4: datetime64[ns] column with time_format does NOT crash.

        Spec v1.1 used `pd.to_datetime(x_arr, unit='s')` which crashes on
        datetime64[ns] columns (ValueError: year 116049 out of range — pd
        treats int64-nanosecond representation AS seconds). v1.2 auto-detects
        the original column dtype BEFORE astype(float) and routes via
        mdates.date2num() directly.

        Locks the most common production case (pandas datetime columns).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'time': pd.date_range('2026-01-01', periods=100, freq='1min'),
            'y': rng.normal(0, 1, 100),
        })
        # Verify dtype is datetime64[ns] or similar
        assert np.issubdtype(df['time'].dtype, np.datetime64), \
            f"Test setup error: df['time'].dtype = {df['time'].dtype}"

        # Should NOT raise
        fig, ax, _ = DFDraw(df).profile("y:time", bins=20, time_format="%H:%M")
        fig.canvas.draw()
        ticks = ax.xaxis.get_majorticklocs()
        # Lock that ticks are matplotlib date numbers ~20000 (Jan 2026)
        assert 0 < ticks[0] < 100000, \
            f"datetime64 column produced unexpected ticks: {ticks[0]:.1f}"
        plt.close(fig)


# ====================================================================== #
# TestScatter3D (8 tests) — type='scatter3d' + z:y:x                      #
# ====================================================================== #

class TestScatter3D:
    """Phase 13.39.DF Item 3 — scatter3D via mpl_toolkits.mplot3d."""

    def _make_df(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'x': rng.uniform(0, 10, n),
            'y': rng.uniform(0, 5, n),
            'z': rng.normal(0, 1, n),
            'tgl': rng.normal(0, 1, n),
            'ncl': rng.integers(50, 160, n).astype(float),
        })

    def test_SC3D_1_basic_render(self):
        """§9.SC3D.1 — Axes3D + Path3DCollection rendered; stats n correct."""
        df = self._make_df()
        fig, ax, stats = DFDraw(df).draw("z:y:x", type="scatter3d")
        assert isinstance(ax, Axes3D), f"ax is {type(ax).__name__}, expected Axes3D"
        path_colls = [c for c in ax.collections if isinstance(c, Path3DCollection)]
        assert len(path_colls) >= 1, "No Path3DCollection on scatter3d axes"
        assert stats['n'] == len(df), f"stats['n']={stats['n']}, expected {len(df)}"
        plt.close(fig)

    def test_SC3D_2_color_expression(self):
        """§9.SC3D.2 — color="abs(tgl)" expression (Phase 13.38 _process_color reuse)."""
        df = self._make_df()
        fig, ax, _ = DFDraw(df).draw(
            "z:y:x", type="scatter3d", color="abs(tgl)", cmap="viridis",
        )
        path_coll = next(c for c in ax.collections if isinstance(c, Path3DCollection))
        arr = path_coll.get_array()
        assert arr is not None, "Colormap array not set for expression color"
        # abs(tgl) values are non-negative
        assert arr.min() >= 0, f"abs(tgl) array min={arr.min()}, expected >=0"
        plt.close(fig)

    def test_SC3D_3_size_column(self):
        """§9.SC3D.3 — size="ncl" column (Phase 13.38 _process_size reuse)."""
        df = self._make_df()
        fig, ax, _ = DFDraw(df).draw("z:y:x", type="scatter3d", size="ncl")
        path_coll = next(c for c in ax.collections if isinstance(c, Path3DCollection))
        sizes = path_coll.get_sizes()
        assert len(np.unique(sizes)) > 1, \
            f"Per-point sizes not applied: only {len(np.unique(sizes))} unique"
        plt.close(fig)

    def test_SC3D_4_selection_reduces_count(self):
        """§9.SC3D.4 — selection= reduces point count."""
        df = self._make_df()
        _, _, stats_all = DFDraw(df).draw("z:y:x", type="scatter3d")
        _, _, stats_sel = DFDraw(df).draw(
            "z:y:x", type="scatter3d", selection="abs(z)<1",
        )
        assert stats_sel['n'] < stats_all['n'], \
            f"selection didn't reduce: all={stats_all['n']}, sel={stats_sel['n']}"

    def test_SC3D_5_two_variable_expr_raises(self):
        """§9.SC3D.5 — "y:x" (colon_count!=2) with type='scatter3d' → ValueError."""
        df = self._make_df()
        with pytest.raises(ValueError, match="3-variable"):
            DFDraw(df).draw("y:x", type="scatter3d")

    def test_SC3D_6_stats_dict_locks_all_three_means(self):
        """§9.SC3D.6 — CP1-3: stats dict locks mean_x AND mean_y AND mean_z to 1e-9.

        Spec v1.1 only value-locked mean_x. A coder swapping y/z in the stats
        dict assignment would pass the v1.1 test silently. v1.2 locks all three.
        """
        df = self._make_df()
        fig, ax, stats = DFDraw(df).draw("z:y:x", type="scatter3d")
        assert 'mean_x' in stats and 'mean_y' in stats and 'mean_z' in stats
        np.testing.assert_allclose(stats['mean_x'], df['x'].mean(), atol=1e-9)
        np.testing.assert_allclose(stats['mean_y'], df['y'].mean(), atol=1e-9)
        np.testing.assert_allclose(stats['mean_z'], df['z'].mean(), atol=1e-9)
        plt.close(fig)

    def test_SC3D_7_group_by_raises_value_error(self):
        """§9.SC3D.7 — CP2-1: group_by + type='scatter3d' → ValueError.

        Scope §7 boundary lock.
        """
        df = self._make_df()
        df['g'] = np.repeat(['A', 'B', 'C'], 100)
        with pytest.raises(ValueError, match="(group_by|scatter3d)"):
            DFDraw(df).draw("z:y:x", type="scatter3d", group_by="g")

    def test_SC3D_8_same_true_non_3d_axes_raises(self):
        """§9.SC3D.8 — CP2-2: same=True onto non-3D axes → ValueError.

        Cannot overlay 3D on 2D — different projection types.
        """
        df = self._make_df()
        fig2d, ax2d = plt.subplots()  # 2D axes
        with pytest.raises(ValueError, match="(3D projection|Axes3D|not 3D)"):
            DFDraw(df).draw("z:y:x", type="scatter3d", ax=ax2d, same=True)
        plt.close(fig2d)
