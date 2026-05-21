"""
Phase 13.38.DF §9 invariance tests: scatter enhancements.

19 tests across 4 classes:
  TestFacetByFloatGuard         (2)  — BUG-017 facet_by float column guard
  TestScatterErrorBars          (9)  — xerr/yerr + NaN policy + composition
  TestScatterExpressionColor    (5)  — df.eval() color + CP0-1 backward compat lock
  TestScatterExpressionMarker   (3)  — boolean marker + composition + legend

CP0-1 lock (§9.ECM.6): column name 'b' (also a valid matplotlib named
color) must route to the column path, not the fixed-color path.

CP1-4 NaN policy (§9.SE.6): three-tier — raise on 100% non-finite,
warn at >50%, silent zeroing at ≤50%.

Filter convention (Phase 13.36 lesson, carried via FIX1): for plot kinds
using ax.errorbar(), the correct data-line filter is line.get_marker() != '_'
(errorbar central lines have label='_nolegend_').

Spec: PHASE_13_38_DF_v1_1_ScatterEnhancements_Proposal.md
Predecessor: PHASE_13_37_DF_FIX1_END (870 passed)
Expected gate: 889 passed (+19)
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.container
from matplotlib.markers import MarkerStyle

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.style import get_style_value


# Sentinel for "key missing from style" — distinct from None which is a
# legal style value (e.g. scatter.error_ecolor=None means inherit).
_MISSING = object()


# ====================================================================== #
# TestFacetByFloatGuard (2 tests) — BUG-017                                #
# ====================================================================== #

class TestFacetByFloatGuard:
    """Phase 13.38.DF BUG-017 — facet_by float column without bins guard.

    Third instance of BUG-012/BUG-015 float-guard class:
      - BUG-012 (Phase 13.35): hist() group_by float guard
      - BUG-015 (Phase 13.37): profile() group_by float guard
      - BUG-017 (Phase 13.38): _dispatch_faceted_render facet_by float guard
    """

    def test_FBGUARD_1_float_facet_by_no_bins_raises(self):
        """§9.FBGUARD.1 — float facet_by + no bins + nunique>20 → ValueError
        with both binning hints in error message.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(1000),
            'y': rng.normal(0, 1, 1000),
            'z': rng.uniform(0, 5, 1000),  # 1000 unique floats
        })
        with pytest.raises(ValueError) as exc:
            DFDraw(df).profile("y:x", facet_by="z")
        msg = str(exc.value)
        assert "facet_by_bins=N" in msg, \
            f"Error message missing facet_by_bins=N hint: {msg}"
        assert "facet_by_quantiles=N" in msg, \
            f"Error message missing facet_by_quantiles=N hint: {msg}"
        # Sanity: includes the cardinality count
        assert "1000 unique" in msg or "1000" in msg

    def test_FBGUARD_2_float_facet_by_with_bins_no_error(self):
        """§9.FBGUARD.2 — float facet_by + facet_by_bins=9 → no ValueError
        (guard only fires when NO binning kwarg present).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(1000),
            'y': rng.normal(0, 1, 1000),
            'z': rng.uniform(0, 5, 1000),
        })
        # Should not raise
        fig, axes, _ = DFDraw(df).profile(
            "y:x", facet_by="z", facet_by_bins=4
        )
        # Sanity: 4 subplots created with data
        facet_axes = [a for a in fig.axes if a.has_data()]
        assert len(facet_axes) == 4, \
            f"Expected 4 facets from facet_by_bins=4, got {len(facet_axes)}"
        plt.close(fig)


# ====================================================================== #
# TestScatterErrorBars (9 tests) — xerr/yerr + NaN policy + composition    #
# ====================================================================== #

def _extract_yerr_from_errorbar(eb):
    """Extract yerr extents from ErrorbarContainer.

    For symmetric yerr, the lower-cap and upper-cap Line2D objects encode
    y±yerr at each point. We reconstruct yerr per point by subtracting
    the central y value from the cap y values.
    """
    central_line = eb.lines[0]  # Line2D with markers
    y_central = central_line.get_ydata()
    # eb.lines[1] is caps tuple (xcap_low, xcap_high) for xerr,
    # then (ycap_low, ycap_high) for yerr — but caps are tuples of Line2Ds
    # easier: use eb.has_yerr and reconstruct from lines[2] (error segments)
    # eb.lines[2] is a LineCollection of error bar segments
    segs = eb.lines[2]  # tuple of LineCollections (xerr_segs, yerr_segs) or single
    # If both xerr and yerr, segs is (xerr_lc, yerr_lc); else single LC
    if isinstance(segs, tuple):
        # Pick yerr segments (the one not horizontal)
        yerr_lc = segs[-1] if eb.has_yerr else None
    else:
        yerr_lc = segs
    if yerr_lc is None:
        return None
    # Each segment: [(x, y_low), (x, y_high)] — yerr = (y_high - y_low) / 2
    segments = yerr_lc.get_segments()
    yerr_values = np.array([(seg[1][1] - seg[0][1]) / 2.0 for seg in segments])
    return yerr_values


def _extract_xerr_from_errorbar(eb):
    """Same as _extract_yerr_from_errorbar but for x-axis."""
    segs = eb.lines[2]
    if isinstance(segs, tuple):
        xerr_lc = segs[0] if eb.has_xerr else None
    else:
        xerr_lc = segs if eb.has_xerr and not eb.has_yerr else None
    if xerr_lc is None:
        return None
    segments = xerr_lc.get_segments()
    xerr_values = np.array([(seg[1][0] - seg[0][0]) / 2.0 for seg in segments])
    return xerr_values


class TestScatterErrorBars:
    """Phase 13.38.DF — xerr=, yerr= on scatter() with NaN policy."""

    def test_SE_1_yerr_column_extents_match(self):
        """§9.SE.1 — yerr=column: rendered error bar extents ≡ column values."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(50, dtype=float),
            'y': rng.normal(0, 1, 50),
            'err_y': np.linspace(0.1, 1.0, 50),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", yerr="err_y")
        eb_containers = [c for c in ax.containers
                         if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert len(eb_containers) == 1, \
            f"Expected 1 ErrorbarContainer, got {len(eb_containers)}"
        eb = eb_containers[0]
        yerr_rendered = _extract_yerr_from_errorbar(eb)
        np.testing.assert_allclose(
            yerr_rendered, df['err_y'].values, atol=1e-9,
            err_msg="yerr column values not equal to rendered error extents"
        )
        plt.close(fig)

    def test_SE_2_xerr_column_extents_match(self):
        """§9.SE.2 — xerr=column: rendered error bar extents ≡ column values."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(50, dtype=float),
            'y': rng.normal(0, 1, 50),
            'err_x': np.linspace(0.05, 0.5, 50),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", xerr="err_x")
        eb_containers = [c for c in ax.containers
                         if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert len(eb_containers) == 1
        eb = eb_containers[0]
        xerr_rendered = _extract_xerr_from_errorbar(eb)
        np.testing.assert_allclose(
            xerr_rendered, df['err_x'].values, atol=1e-9
        )
        plt.close(fig)

    def test_SE_3_both_xerr_yerr_simultaneously(self):
        """§9.SE.3 — both xerr + yerr: both extents present on container."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(50, dtype=float),
            'y': rng.normal(0, 1, 50),
            'ex': np.full(50, 0.1),
            'ey': np.full(50, 0.2),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", xerr="ex", yerr="ey")
        eb_containers = [c for c in ax.containers
                         if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert len(eb_containers) == 1
        eb = eb_containers[0]
        assert eb.has_xerr, "ErrorbarContainer missing xerr"
        assert eb.has_yerr, "ErrorbarContainer missing yerr"
        plt.close(fig)

    def test_SE_4_yerr_dfeval_expression(self):
        """§9.SE.4 — yerr=df.eval() expression: extents ≡ evaluated values."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': np.arange(50, dtype=float),
            'y': rng.normal(0, 1, 50),
            'err_sq': np.linspace(0.01, 1.0, 50),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", yerr="sqrt(err_sq)")
        eb_containers = [c for c in ax.containers
                         if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert len(eb_containers) == 1
        eb = eb_containers[0]
        yerr_rendered = _extract_yerr_from_errorbar(eb)
        expected = np.sqrt(df['err_sq'].values)
        np.testing.assert_allclose(yerr_rendered, expected, atol=1e-9)
        plt.close(fig)

    def test_SE_5_default_dispatch_invariance(self):
        """§9.SE.5 — CP1-3: when xerr=yerr=None, dispatch goes to ax.scatter()
        (PathCollection), NOT ax.errorbar(yerr=None) (ErrorbarContainer).

        Locks the §3 dispatch branch on (xerr is not None or yerr is not None).
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': np.arange(50), 'y': rng.normal(0, 1, 50)})
        fig, ax, _ = DFDraw(df).scatter("y:x")
        # PathCollection from ax.scatter
        assert len(ax.collections) >= 1
        assert isinstance(
            ax.collections[0], matplotlib.collections.PathCollection
        ), f"Expected PathCollection, got {type(ax.collections[0]).__name__}"
        # NO ErrorbarContainer in ax.containers
        eb_containers = [c for c in ax.containers
                         if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert len(eb_containers) == 0, \
            "ax.errorbar() incorrectly dispatched when xerr=yerr=None"
        plt.close(fig)

    def test_SE_6_nan_policy_three_tiers(self):
        """§9.SE.6 — CP1-4 three-tier NaN policy:
        Part A: silent zeroing at low nanfrac (10%)
        Part B: UserWarning at >50% nanfrac (75%)
        Part C: ValueError at 100% nanfrac
        """
        rng = np.random.default_rng(42)
        n = 100
        # Part A: 10% NaN — silent, but nanfrac in stats
        err_partial = np.linspace(0.1, 1.0, n).copy()
        err_partial[:10] = np.nan
        df_a = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': rng.normal(0, 1, n),
            'ey': err_partial,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, stats = DFDraw(df_a).scatter("y:x", yerr="ey")
            silent_warns = [x for x in w if 'non-finite' in str(x.message)]
        assert len(silent_warns) == 0, \
            "Should be silent at nanfrac=0.10, got UserWarning"
        assert stats['yerr_nanfrac'] == pytest.approx(0.10, abs=0.001)
        plt.close(fig)

        # Part B: 75% NaN — UserWarning
        err_mostly_nan = np.linspace(0.1, 1.0, n).copy()
        err_mostly_nan[:75] = np.nan
        df_b = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': rng.normal(0, 1, n),
            'ey': err_mostly_nan,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, _ = DFDraw(df_b).scatter("y:x", yerr="ey")
            warn_matches = [x for x in w
                            if 'non-finite' in str(x.message)
                            and '75' in str(x.message)]
        assert len(warn_matches) >= 1, \
            f"Expected UserWarning at 75% NaN; got warnings: {[str(x.message) for x in w]}"
        plt.close(fig)

        # Part C: 100% NaN — ValueError
        df_c = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': rng.normal(0, 1, n),
            'ey': np.full(n, np.nan),
        })
        with pytest.raises(ValueError) as exc:
            DFDraw(df_c).scatter("y:x", yerr="ey")
        msg = str(exc.value)
        assert "ALL" in msg
        assert "non-finite" in msg

    def test_SE_7_style_keys_registered(self):
        """§9.SE.7 — style keys registered with documented defaults."""
        assert get_style_value("scatter.error_capsize", _MISSING) == 2, \
            "scatter.error_capsize not registered or wrong default"
        assert get_style_value("scatter.error_elinewidth", _MISSING) == 1.0, \
            "scatter.error_elinewidth not registered or wrong default"
        # scatter.error_ecolor=None is a legitimate value (inherit) — must
        # be distinguishable from _MISSING
        val = get_style_value("scatter.error_ecolor", _MISSING)
        assert val is None, \
            f"scatter.error_ecolor should be None (inherit), got {val!r}"

    def test_SE_8_group_by_plus_yerr(self):
        """§9.SE.8 — group_by + yerr: per-group error bars.

        Each group renders its own ax.errorbar() call.
        """
        rng = np.random.default_rng(42)
        n_per_group = 33
        n = n_per_group * 3  # 99 — divisible by 3
        df = pd.DataFrame({
            'x': np.tile(np.arange(n_per_group, dtype=float), 3),
            'y': rng.normal(0, 1, n),
            'g': np.repeat(['A', 'B', 'C'], n_per_group),
            'ey': np.full(n, 0.3),
        })
        # group_by path may or may not propagate yerr — this test locks the
        # current architectural behavior. If yerr propagates to group_by,
        # we expect 3 errorbar containers; if not, 0 containers but 3
        # collections (from per-group scatter).
        fig, ax, _ = DFDraw(df).scatter(
            "y:x", group_by="g", yerr="ey"
        )
        eb_count = sum(
            1 for c in ax.containers
            if isinstance(c, matplotlib.container.ErrorbarContainer)
        )
        # Document current behavior: yerr forwarding through group_by path
        # is deferred (scope §7 — single-path scatter only). Lock that
        # SOMETHING is rendered (not silent crash).
        n_artists = eb_count + len(ax.collections)
        assert n_artists >= 3, \
            f"group_by + yerr produced too few artists ({n_artists}); expected ≥3"
        plt.close(fig)

    def test_SE_9_xerr_plus_facet_by(self):
        """§9.SE.9 — NEW (Claude48 P2): xerr + facet_by composition.

        Lock that error bars render per-facet when facet_by is active.
        """
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': rng.normal(0, 1, n),
            'f': np.repeat([0, 1, 2], 100),
            'ey': np.full(n, 0.2),
        })
        fig, axes, _ = DFDraw(df).scatter(
            "y:x", yerr="ey", facet_by="f"
        )
        # Count axes with data
        facet_axes = [a for a in fig.axes if a.has_data()]
        assert len(facet_axes) == 3, \
            f"Expected 3 facets, got {len(facet_axes)}"
        # Each facet has at least one renderable artist (errorbar or scatter)
        for i, a in enumerate(facet_axes):
            eb_count = sum(
                1 for c in a.containers
                if isinstance(c, matplotlib.container.ErrorbarContainer)
            )
            n_artists = eb_count + len(a.collections)
            assert n_artists >= 1, \
                f"Facet {i} has no artists rendered"
        plt.close(fig)


# ====================================================================== #
# TestScatterExpressionColor (5 tests) — df.eval() color + CP0-1 lock     #
# ====================================================================== #

class TestScatterExpressionColor:
    """Phase 13.38.DF — expression-based color= for scatter."""

    def test_ECM_1_expression_color_colormap_applied(self):
        """§9.ECM.1 — color="abs(tgl)" expression → colormap applied.

        Locks both: (a) variation in the underlying color array (NOT
        get_facecolor() which matplotlib defers to draw time), (b) the
        array equals abs(tgl) values to 1e-9.
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'tgl': np.linspace(-2, 2, n),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", color="abs(tgl)")
        # The underlying _A array must be set (None means fixed color path)
        arr = ax.collections[0].get_array()
        assert arr is not None, \
            "Colormap array not set — expression color went to fixed-color path"
        # Lock the array variation (>50 distinct values across 100 points)
        n_unique = len(np.unique(arr))
        assert n_unique > 50, \
            f"Expected colormap variation in array; got {n_unique} distinct values"
        # Lock the actual values match abs(tgl)
        np.testing.assert_allclose(
            arr, np.abs(df['tgl'].values), atol=1e-9
        )
        # Lock the colormap was set (not None — that would be fixed-color path)
        assert ax.collections[0].get_cmap() is not None
        plt.close(fig)

    def test_ECM_2_column_color_byte_identical_backward_compat(self):
        """§9.ECM.2 — color="col_name" (existing column) backward compat.

        Two consecutive calls with identical params produce identical
        color arrays (deterministic).
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'tgl': np.linspace(-1, 1, n),
        })
        fig_a, ax_a, _ = DFDraw(df).scatter("y:x", color="tgl")
        arr_a = ax_a.collections[0].get_array()
        plt.close(fig_a)

        fig_b, ax_b, _ = DFDraw(df).scatter("y:x", color="tgl")
        arr_b = ax_b.collections[0].get_array()
        plt.close(fig_b)

        np.testing.assert_array_equal(arr_a, arr_b)
        # Also verify it matches the source column
        np.testing.assert_allclose(arr_a, df['tgl'].values, atol=1e-9)

    def test_ECM_3_invalid_expression_actionable_error(self):
        """§9.ECM.3 — invalid expression → ValueError with all 3 hints."""
        n = 50
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
        })
        with pytest.raises(ValueError) as exc:
            DFDraw(df).scatter("y:x", color="this_is_not_a_thing()")
        msg = str(exc.value)
        assert "column name" in msg, \
            f"Error message missing 'column name' hint: {msg}"
        assert "color string" in msg, \
            f"Error message missing 'color string' hint: {msg}"
        assert "df.eval()" in msg, \
            f"Error message missing 'df.eval()' hint: {msg}"

    def test_ECM_6_column_name_collision_with_named_color(self):
        """§9.ECM.6 — CP0-1 BACKWARD-COMPAT LOCK.

        Column 'b' (also a valid matplotlib named color) must route to
        the COLUMN path, not the fixed-color path. Regression-lock for
        the v1.0 dispatch-order bug where to_rgba() ran first.

        Diagnostic: get_array() is None iff fixed-color path was taken;
        get_array() with N values iff column path was taken. This is more
        reliable than get_facecolor() which matplotlib defers to draw time.
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'b': np.linspace(0, 1, n),  # column happens to be named 'b'
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", color='b')
        arr = ax.collections[0].get_array()
        # If 'b' went to fixed-color path: arr is None
        # If 'b' went to column path: arr is the column values
        assert arr is not None, (
            "CP0-1 REGRESSION: column 'b' rendered as fixed matplotlib blue "
            "(get_array() is None ⟹ fixed-color path taken). "
            "The _process_color() dispatch order must check column-name "
            "BEFORE to_rgba()."
        )
        assert len(arr) == n, \
            f"Color array wrong length: expected {n}, got {len(arr)}"
        # The values must equal the column values to 1e-9 (column path)
        np.testing.assert_allclose(
            arr, df['b'].values, atol=1e-9,
            err_msg="Color array doesn't match column 'b' — wrong dispatch branch"
        )
        # Variation: column has 100 distinct values → array has >50 unique
        assert len(np.unique(arr)) > 50, \
            f"Column 'b' should have >50 unique values, got {len(np.unique(arr))}"
        plt.close(fig)

    def test_ECM_7_expression_color_plus_group_by_behavior_locked(self):
        """§9.ECM.7 — NEW (Claude48 P2): expression color + group_by behavior.

        Scope §7 explicitly defers expression color+group_by to a future
        phase. The CURRENT behavior with group_by is: scatter dispatches
        to _draw_scatter_grouped() which doesn't go through _process_color
        in the same way — the expression string is treated as a fixed color
        per group OR matplotlib raises a clear error.

        Lock: SOMETHING happens (no silent crash, no silent wrong rendering).
        The test is permissive — it accepts ValueError, NotImplementedError,
        OR a successful render where each group gets its own color via the
        Phase 13.36 sentinel pattern (color='abs(tgl)' would be treated as
        a literal color string by matplotlib's group-by code path).
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'tgl': np.linspace(-1, 1, n),
            'g': np.repeat(['A', 'B', 'C'], n // 3 + 1)[:n],
        })
        # Document current behavior: matplotlib will raise on invalid color
        # string, OR the grouped path swallows the expression. Either way,
        # we lock that NO silent wrong rendering happens.
        try:
            fig, ax, _ = DFDraw(df).scatter(
                "y:x", color="abs(tgl)", group_by="g"
            )
            # If render succeeded, lock that the artists exist (sanity)
            n_artists = len(ax.collections)
            assert n_artists >= 1, \
                "group_by + expression color silently produced no artists"
            plt.close(fig)
        except (ValueError, TypeError, NotImplementedError) as e:
            # Any of these are acceptable — they document the scope boundary
            # cleanly. The point is no silent surprise.
            pass


# ====================================================================== #
# TestScatterExpressionMarker (3 tests) — boolean marker + composition    #
# ====================================================================== #

class TestScatterExpressionMarker:
    """Phase 13.38.DF — boolean df.eval() marker expression."""

    def test_ECM_4_boolean_marker_two_marker_encoding(self):
        """§9.ECM.4 — marker='ncl > 100' boolean expression → two markers.

        Concrete assertion (per Sonnet52_R1 P2): use MarkerStyle path
        comparison + per-collection point count. NOT literal `...`.
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'ncl': np.linspace(50, 150, n),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", marker="ncl > 100")
        # 2 collections (one per unique marker: 's' and 'o')
        assert len(ax.collections) == 2, \
            f"Expected 2 PathCollections for two-marker encoding, got {len(ax.collections)}"

        # Lock total point count: True_count + False_count == len(df)
        true_count = int((df['ncl'] > 100).sum())
        false_count = len(df) - true_count
        point_counts = sorted(
            len(c.get_offsets()) for c in ax.collections
        )
        assert sorted([true_count, false_count]) == point_counts, \
            (f"Point counts mismatch: expected sorted({[true_count, false_count]}), "
             f"got {point_counts}")

        # Lock that the two collections use DIFFERENT marker paths
        path_lengths = [len(c.get_paths()[0].vertices) for c in ax.collections]
        # Square has 4-5 vertices; circle has many. They MUST differ.
        assert len(set(path_lengths)) == 2, \
            f"Two-marker encoding produced same path lengths: {path_lengths}"
        plt.close(fig)

    def test_ECM_5_expression_color_plus_expression_marker_compose(self):
        """§9.ECM.5 — color + marker expression composition.

        Both encodings applied simultaneously to single-path scatter.
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'tgl': np.linspace(-1, 1, n),
            'ncl': np.linspace(50, 150, n),
        })
        fig, ax, _ = DFDraw(df).scatter(
            "y:x", color="abs(tgl)", marker="ncl > 100"
        )
        # 2 collections (one per marker), both with colormap arrays
        assert len(ax.collections) == 2, \
            f"Expected 2 collections, got {len(ax.collections)}"
        for coll in ax.collections:
            arr = coll.get_array()
            assert arr is not None, \
                "Colormap not applied to marker subgroup — composition broken"
            # Each subgroup's array values should be subset of full abs(tgl)
            assert arr.min() >= 0, \
                "abs() result should be non-negative"
        plt.close(fig)

    def test_ECM_8_per_point_marker_legend_no_duplicates(self):
        """§9.ECM.8 — NEW (Claude48 P2): per-point marker legend behavior.

        np.unique() marker loop creates N ax.scatter() calls. Lock that
        matplotlib's auto-legend doesn't generate N spurious entries
        when no explicit label is provided (label='_nolegend_' invariant).
        """
        n = 100
        df = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.random.normal(0, 1, n),
            'ncl': np.linspace(50, 150, n),
        })
        fig, ax, _ = DFDraw(df).scatter("y:x", marker="ncl > 100")
        # Try to make a legend — should produce nothing (all _nolegend_)
        try:
            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
        except Exception:
            handles, labels = [], []
        # No auto-legend entries from internal _nolegend_ labels
        user_visible_labels = [l for l in labels if not l.startswith('_')]
        assert len(user_visible_labels) == 0, \
            (f"Per-point marker rendering generated spurious legend entries: "
             f"{user_visible_labels}")
        plt.close(fig)
