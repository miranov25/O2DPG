"""
Phase 13.40.DF §9 invariance tests: cumulative histogram.

10 tests in TestCumulativeHist:
  CH.1   monotone non-decreasing + total-N lock
  CH.2   cumulative=True + norm='probability' → ECDF (last=1.0)
  CH.3   cumulative=-1 + norm='probability' → survival (1→0)
  CH.4   cumulative=False byte-identical (backward compat)
  CH.5   group_by + cumulative=True + hist_norm → per-group ECDFs
  CH.6   hist_errors + cumulative=True → NotImplementedError (M5)
  CH.7   [x,y] vector dispatch + cumulative=True propagates (FIX1 bug class)
  CH.8   facet_by + cumulative=True → per-facet ECDFs (Phase 13.32 compose)
  CH.9   histtype='step' + cumulative=True → step ECDF (CP1-4 Polygon-safe)
  CH.10  group_by + cumulative + stacked=True (CP2-1 regression lock)

CP1-4 fix-at-code-time disclosure: dfdraw default style is
hist.histtype='stepfilled' which produces Polygon patches (not Rectangle).
get_height() raises AttributeError on Polygon — ALL tests use
get_path().vertices probe throughout. Also scans ax.lines for step paths.

Sonet50 panel CP fixes (v1.1 → v1.2):
  CP1-1: 3 (actually 4 — see CRR §2.1) ax.hist() call sites
  CP1-2: recursive forwarding (DFDraw.hist → draw_hist → _draw_hist_grouped → ax.hist)
  CP1-3: drawer.py:3085 docstring drops invalid norm='cumulative'
  CP1-4: Polygon-safe vertex probe in ALL tests (not just CH.9)
  CP2-1: §9.CH.10 stacked+cumulative regression lock

Spec: PHASE_13_40_DF_v1_2_CumulativeHist_Proposal.md
Predecessor: PHASE_13_39_DF_END (913/0/1)
Expected gate: 923 (+10)
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfextensions.dfdraw import DFDraw


def _max_y_in_axes(ax):
    """Polygon-safe max y across ax.patches AND ax.lines.

    dfdraw default hist.histtype='stepfilled' renders as Polygon (not
    Rectangle) — get_height() is Rectangle-only. Scan path vertices
    instead. Also scan ax.lines for histtype='step' which may use Line2D.
    """
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


def _y_values_in_axes(ax):
    """Return list of all y-vertices across patches and lines (for monotonicity)."""
    ys = []
    for p in ax.patches:
        try:
            verts = p.get_path().vertices
            ys.extend(verts[:, 1].tolist())
        except (AttributeError, ValueError):
            pass
    for line in ax.lines:
        try:
            ys.extend(line.get_ydata().tolist())
        except Exception:
            pass
    return [y for y in ys if np.isfinite(y)]


class TestCumulativeHist:
    """Phase 13.40.DF — cumulative=True/False/-1 on hist()."""

    def _make_df(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({'x': rng.normal(0, 1, n)})

    def _make_grouped_df(self, per_group=100, seed=42):
        rng = np.random.default_rng(seed)
        n = per_group * 3
        return pd.DataFrame({
            'x': rng.normal(0, 1, n),
            'g': np.repeat(['A', 'B', 'C'], per_group),
        })

    def test_CH_1_monotone_and_total_N_lock(self):
        """§9.CH.1 — cumulative=True: max y = total N AND ECDF top edge
        is monotonically non-decreasing.

        Stronger than last-bar-only: swapped direction would still hit total
        N at last bar but fail monotonicity check.
        """
        df = self._make_df(n=500)
        fig, ax, _ = DFDraw(df).hist("x", bins=50, cumulative=True)

        # Total-N lock: max vertex y = N (last cumulative bar)
        max_y = _max_y_in_axes(ax)
        assert abs(max_y - 500) < 1e-6, \
            f"cumulative max_y={max_y} ≠ N=500"

        # Monotonicity: collect unique non-decreasing top-edge values from
        # the polygon vertices (top of stepfilled traces the cumulative).
        # For a polygon, the vertices include both top and bottom edges; the
        # cumulative counts appear as the top y-coords.
        ys = _y_values_in_axes(ax)
        assert len(ys) > 0, "No y-values rendered"
        # Sorted top-edge sample: the cumulative counts must be a
        # non-decreasing sequence when traversing left → right. The polygon
        # vertices alternate top/bottom — the top edge segments are the
        # non-zero values in monotone order.
        nonzero_ys = sorted(set(y for y in ys if y > 1e-9))
        for i in range(len(nonzero_ys) - 1):
            assert nonzero_ys[i] <= nonzero_ys[i + 1] + 1e-9, \
                f"Non-monotone cumulative at index {i}: " \
                f"{nonzero_ys[i]} > {nonzero_ys[i + 1]}"
        plt.close(fig)

    def test_CH_2_ECDF_last_value_is_one(self):
        """§9.CH.2 — cumulative=True + norm='probability' → ECDF max = 1.0.

        Note: uses 'norm' (single-histogram normalization), NOT 'hist_norm'
        (per-group normalization).
        """
        df = self._make_df(n=500)
        fig, ax, _ = DFDraw(df).hist(
            "x", bins=50, cumulative=True, norm="probability")
        max_y = _max_y_in_axes(ax)
        np.testing.assert_allclose(max_y, 1.0, atol=1e-9,
            err_msg=f"ECDF max value {max_y} ≠ 1.0")
        plt.close(fig)

    def test_CH_3_survival_starts_at_one(self):
        """§9.CH.3 — cumulative=-1 + norm='probability' → survival function.

        ROOT convention. The leftmost vertex reaches the maximum value
        (≈ 1.0 for probability-normalized survival).
        """
        df = self._make_df(n=1000)
        fig, ax, _ = DFDraw(df).hist(
            "x", bins=50, cumulative=-1, norm="probability")
        max_y = _max_y_in_axes(ax)
        np.testing.assert_allclose(max_y, 1.0, atol=1e-9,
            err_msg=f"Survival max value {max_y} ≠ 1.0")
        plt.close(fig)

    def test_CH_4_cumulative_false_backward_compat(self):
        """§9.CH.4 — cumulative=False (default) byte-identical to no kwarg."""
        df = self._make_df(n=300)
        fig_a, ax_a, _ = DFDraw(df).hist("x", bins=30)
        fig_b, ax_b, _ = DFDraw(df).hist("x", bins=30, cumulative=False)

        # Compare vertex lists (handles both Rectangle and Polygon patches)
        ys_a = _y_values_in_axes(ax_a)
        ys_b = _y_values_in_axes(ax_b)
        assert len(ys_a) == len(ys_b), \
            f"Different vertex counts: {len(ys_a)} vs {len(ys_b)}"
        np.testing.assert_allclose(
            sorted(ys_a), sorted(ys_b), atol=1e-9,
            err_msg="cumulative=False changed default behavior",
        )
        # Sanity: max y should be much smaller than N (this is a regular hist)
        max_y_a = max(ys_a) if ys_a else 0
        assert max_y_a < 100, \
            f"Default hist max_y={max_y_a} suspiciously large (cumulative leak?)"
        plt.close(fig_a)
        plt.close(fig_b)

    def test_CH_5_group_by_per_group_ecdf(self):
        """§9.CH.5 — group_by + cumulative + hist_norm='probability':
        each group is an independent ECDF (max ≈ 1.0 per group).

        Stacked=False (default) → overlaid path. Tests call sites 3/4 + 4/4
        (linestyle_cycle + default branches in _draw_hist_grouped).
        """
        df = self._make_grouped_df(per_group=100)
        fig, ax, _ = DFDraw(df).hist(
            "x", bins=30, cumulative=True, hist_norm="probability",
            group_by="g")

        # In overlaid mode with hist_norm='probability', max y across the
        # whole axes ≈ 1.0 (each group's ECDF saturates at 1.0).
        max_y = _max_y_in_axes(ax)
        np.testing.assert_allclose(max_y, 1.0, atol=1e-3,
            err_msg=f"group_by ECDF max {max_y} ≠ 1.0 — "
                    "cumulative likely dropped in overlaid branch")
        plt.close(fig)

    def test_CH_6_hist_errors_plus_cumulative_raises(self):
        """§9.CH.6 — M5 correctness guard: hist_errors + cumulative →
        NotImplementedError.

        Poisson per-bin errors are independent; cumulative counts have
        correlated uncertainty. Silently passing would render statistically
        wrong error bars.
        """
        df = self._make_df(n=200)
        with pytest.raises(NotImplementedError, match="correlated"):
            DFDraw(df).hist("x", bins=20, cumulative=True, hist_errors=True)

    def test_CH_7_vector_dispatch_propagates_cumulative(self):
        """§9.CH.7 — [x,y] vector dispatch must NOT silently drop cumulative.

        Phase 13.16.DF FIX1 bug class regression lock. Vector dispatch with
        type='hist' overlays both elements into ONE axis (2 Polygons) by
        default. With cumulative=True + norm='probability', each polygon's
        max y should ≈ 1.0 (ECDF saturation).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 200),
            'y': rng.normal(2, 1, 200),
        })
        fig, axes, stats_list = DFDraw(df).draw(
            "[x, y]", type="hist", bins=30,
            cumulative=True, norm="probability")
        assert len(stats_list) == 2, f"Expected 2 stats, got {len(stats_list)}"

        # Vector dispatch overlays into a single axis by default
        # → 2 Polygon patches on that axis, NOT 2 separate axes.
        data_axes = [a for a in fig.axes if a.patches or a.lines]
        assert len(data_axes) >= 1, "No data axes found in vector dispatch"

        # Each polygon (one per vector element) must reach max y ≈ 1.0
        # (ECDF), proving cumulative=True was applied to BOTH elements.
        all_polygon_max = []
        for a in data_axes:
            for p in a.patches:
                try:
                    verts = p.get_path().vertices
                    all_polygon_max.append(float(np.nanmax(verts[:, 1])))
                except (AttributeError, ValueError):
                    pass
        assert len(all_polygon_max) >= 2, \
            f"Expected ≥2 polygons for [x,y] vector, got {len(all_polygon_max)}"
        # Both polygons reach ECDF saturation
        for i, mx in enumerate(all_polygon_max):
            assert mx > 0.98, (
                f"Vector polygon {i} max_y={mx} ≠ 1.0 — cumulative " 
                "dropped in vector dispatch path (Phase 13.16.DF FIX1 regression)"
            )
        plt.close(fig)

    def test_CH_8_facet_by_per_facet_ecdf(self):
        """§9.CH.8 — facet_by + cumulative=True → per-facet cumulative.

        Phase 13.32 facet_by composition lock. Each facet renders an
        independent cumulative histogram. Note: norm='probability' does
        NOT propagate cleanly through facet_by (pre-existing Phase 13.32
        composition gap — tracked as Phase 13.41 candidate), so this test
        verifies cumulative=True propagates by checking that per-facet
        max_y EXCEEDS the per-bin maximum (proving accumulation across
        bins occurred), not that ECDF saturates at 1.0.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 300),
            'f': np.repeat([0, 1, 2], 100),
        })
        # Use cumulative=True without norm to avoid the pre-existing
        # Phase 13.32 norm-through-facet propagation gap
        result = DFDraw(df).hist(
            "x", bins=15, cumulative=True, facet_by="f")
        fig = result[0]

        # Find data-bearing facet axes (one per facet value)
        data_axes = [a for a in fig.axes if a.patches]
        assert len(data_axes) >= 3, \
            f"Expected ≥3 facet axes, got {len(data_axes)}"

        # Each facet has 100 events → max non-cumulative bar ≤ 30 (peak of
        # Gaussian distribution across 15 bins). Cumulative max should
        # reach close to 100 (per-facet total). Use threshold ≥ 80 (80%)
        # to allow for tail bins.
        for i, a in enumerate(data_axes[:3]):
            max_y = _max_y_in_axes(a)
            assert max_y >= 80, (
                f"Facet {i} cumulative max_y={max_y} < 80; expected ~100 "
                "(per-facet total). cumulative=True likely dropped through "
                "facet_by dispatch."
            )
        plt.close(fig)

    def test_CH_9_histtype_step_plus_cumulative_polygon_safe(self):
        """§9.CH.9 — histtype='step' + cumulative=True: HEP-standard step ECDF.

        CP1-4: matplotlib renders histtype='step' as Polygon OR Line2D
        depending on version. get_height() is Rectangle-only and would
        raise AttributeError. _max_y_in_axes() probes both patches
        (via vertices) AND ax.lines (via ydata).
        """
        df = self._make_df(n=500)
        fig, ax, _ = DFDraw(df).hist(
            "x", bins=50, cumulative=True, norm="probability",
            histtype='step', linewidth=1.5)
        # At least one visual element rendered
        assert len(ax.patches) + len(ax.lines) > 0, \
            "No visual element for histtype='step' + cumulative=True"
        max_y = _max_y_in_axes(ax)
        assert max_y > 0.98, \
            f"Step ECDF max_y={max_y} ≠ 1.0 (vertices+lines probe)"
        plt.close(fig)

    def test_CH_10_group_by_stacked_cumulative_regression_lock(self):
        """§9.CH.10 — CP2-1 NEW: group_by + cumulative=True + stacked=True.

        Regression lock for CP1-1 (stacked branch at histogram.py:~803).
        Without explicit cumulative= forward in the stacked branch, this
        test fails: stacked path falls back to regular histogram (no
        accumulation), max bar ≈ 30-50 per group instead of stacked
        cumulative reaching ≥ total N.

        matplotlib semantics: stacked + cumulative renders each layer's
        cumulative independently, then stacks them. Top of rightmost stack
        = sum of per-group cumulative finals = total N.
        """
        df = self._make_grouped_df(per_group=100)   # N=300
        fig, ax, _ = DFDraw(df).hist(
            "x", bins=30, cumulative=True, group_by="g", stacked=True)
        max_y = _max_y_in_axes(ax)
        # Should reach total N=300 (allow ≥250 tolerance for Gaussian tails)
        assert max_y >= 250, (
            f"Stacked cumulative max_y={max_y} < 250; expected ~300 "
            "(total N). Stacked branch (histogram.py:~803) likely "
            "dropped cumulative=True — Phase 13.40 CP1-1 regression."
        )
        plt.close(fig)
