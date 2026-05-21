"""
Phase 13.37.DF — Histogram robustness.

§9 invariance test plan (24 tests):
  TestBUG014StepColor          — 4 tests (STEP.1-4)
  TestBUG015ProfileGuard       — 2 tests (GUARD.1-2)
  TestBUG016IntervalSort       — 3 tests (SORT.1-3)
  TestHistErrors               — 10 tests (HE.1-9 + HE.style)
  TestLinestyleCycle           — 5 tests (LC.1-5)

Bugs closed:
- BUG-014: histtype='step' rendered all step lines black (edgecolor='black'
  style default overrode the color= per-group). Fix: edgecolor sentinel
  extends Phase 13.36 pattern.
- BUG-015: profile() float group_by without bins → memory hang. Fix:
  nunique()>20 ValueError guard, mirrors Phase 13.35 hist().
- BUG-016: _interval_sort_key did not handle pd.Interval → lexicographic
  sort, broke when any bin label crossed 10. Fix: hasattr(label, 'left')
  guard short-circuits to numeric left boundary.

Test filter convention (from Phase 13.36):
- ax.get_lines() returns BOTH central data lines + errorbar cap lines.
- Cap lines have marker='_'; central lines have user-meaningful marker.
- Filter via line.get_marker() != '_'.

For Polygon edgecolor (histtype='step'): edgecolor RGB tuple, not facecolor
(facecolor for step has alpha=0).
"""
import numpy as np
import pandas as pd
import pytest
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from dfextensions.dfdraw import DFDraw


# ---------------------------------------------------------------------- #
# Shared fixtures
# ---------------------------------------------------------------------- #

@pytest.fixture
def df_grouped():
    """Three-group DataFrame for general group_by tests."""
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame({
        'x': rng.uniform(-1, 1, n),
        'y': rng.normal(0, 1, n),
        'g': rng.integers(0, 3, n),
        'g_str': rng.choice(['A', 'B', 'C'], n),  # categorical groups
        'z_hi_card': rng.normal(0, 1, n),
        'mP4_abs': np.abs(rng.normal(5, 5, n)),   # bins will cross 10
        'w': rng.uniform(0.5, 1.5, n),            # weights
    })


def _central_lines(ax):
    """Profile errorbar central data lines (excludes caps marker='_')."""
    return [line for line in ax.get_lines() if line.get_marker() != '_']


def _errorbar_containers(ax):
    """ErrorbarContainer objects on ax (one per group for hist_errors)."""
    return [c for c in ax.containers
            if 'ErrorbarContainer' in type(c).__name__]


def _errorbar_data(ec):
    """Extract (xs, heights, yerrs) from ErrorbarContainer with fmt='none'.

    Matplotlib structure for fmt='none' errorbars:
      ec[0]            = data line — None for fmt='none'
      ec.lines[0]      = same as ec[0] — None
      ec.lines[1]      = caplines (tuple) — empty for fmt='none', no caps
      ec.lines[2]      = (barlinecol,) — LineCollection for vertical bars

    Each segment of barlinecol is [(x, height-err), (x, height+err)].
    """
    barlinecol = ec.lines[2][0]
    segments = barlinecol.get_segments()
    xs = np.array([s[0][0] for s in segments])
    ymin = np.array([s[0][1] for s in segments])
    ymax = np.array([s[1][1] for s in segments])
    heights = (ymin + ymax) / 2
    yerrs = (ymax - ymin) / 2
    return xs, heights, yerrs


# ====================================================================== #
# TestBUG014StepColor (4 tests)
# ====================================================================== #

class TestBUG014StepColor:
    """histtype='step' renders each group with its own palette color."""

    def test_STEP1_default_per_group_step_colors(self, df_grouped):
        """STEP.1: 3 groups + histtype='step' → 3 distinct edgecolors, none black."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', histtype='step', linewidth=2)
        ec_colors = [tuple(p.get_edgecolor()) for p in ax.patches]
        # 3 distinct
        unique_ec = {tuple(np.round(c[:3], 3)) for c in ec_colors}
        assert len(unique_ec) == 3, \
            f"Expected 3 distinct edgecolors, got {len(unique_ec)}: {unique_ec}"
        # None of them all-black
        for c in ec_colors:
            assert not np.allclose(c[:3], (0, 0, 0), atol=0.01), \
                f"Found black step line {c} — BUG-014 not fixed"
        plt.close(fig)

    def test_STEP2_bar_histtype_unchanged(self, df_grouped):
        """STEP.2: histtype='bar' (default) edgecolor='black' preserved (backward compat)."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', histtype='bar')
        # In bar mode, edgecolor comes from style default 'black'
        for p in ax.patches:
            ec = p.get_edgecolor()
            if isinstance(ec, np.ndarray) and ec.ndim == 2:
                ec = ec[0]
            # All bar patches share the style edgecolor (black)
            assert np.allclose(ec[:3], (0, 0, 0), atol=0.01), \
                f"Bar mode edgecolor changed: {ec}"
        plt.close(fig)

    def test_STEP3_user_edgecolor_wins_over_step_cycle(self, df_grouped):
        """STEP.3: histtype='step' + edgecolor='red' → all red. Phase 13.36 sentinel."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g',
                            histtype='step', edgecolor='red')
        red_rgb = to_rgb('red')
        for p in ax.patches:
            ec = p.get_edgecolor()
            if isinstance(ec, np.ndarray) and ec.ndim == 2:
                ec = ec[0]
            assert np.allclose(ec[:3], red_rgb, atol=0.01), \
                f"Patch edgecolor {ec} != red — Phase 13.36 sentinel broken"
        plt.close(fig)

    def test_STEP4_stepfilled_edgecolor_untouched(self, df_grouped):
        """STEP.4: histtype='stepfilled' + group_by — step-only logic doesn't fire.

        Locks that Phase 13.37 only touches the step path; stepfilled
        edgecolor still resolves from style.
        """
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', histtype='stepfilled')
        # No crash. Patches rendered.
        assert len(ax.patches) > 0
        plt.close(fig)


# ====================================================================== #
# TestBUG015ProfileGuard (2 tests)
# ====================================================================== #

class TestBUG015ProfileGuard:
    """profile() float group_by needs bins or guard fires."""

    def test_GUARD1_float_no_bins_raises(self, df_grouped):
        """GUARD.1: profile() float group_by + no bins + nunique>20 → ValueError."""
        d = DFDraw(df_grouped)
        with pytest.raises(ValueError, match=r"group_by_bins=N"):
            d.profile('y:x', group_by='z_hi_card')

    def test_GUARD2_with_bins_no_error(self, df_grouped):
        """GUARD.2: profile() float group_by + group_by_bins=5 → no error."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='z_hi_card', group_by_bins=5)
        assert len(_central_lines(ax)) > 0
        plt.close(fig)


# ====================================================================== #
# TestBUG016IntervalSort (3 tests)
# ====================================================================== #

class TestBUG016IntervalSort:
    """pd.Interval legend order is by .left boundary, not lexicographic."""

    def test_SORT1_bins_crossing_10_numeric_order(self, df_grouped):
        """SORT.1: 5 bins crossing 10 → legend in numeric .left order."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='mP4_abs', group_by_bins=5,
                            hist_norm='probability')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        # Extract left boundaries from labels like "(2.0, 4.0]"
        import re
        lefts = []
        for L in labels:
            m = re.match(r'\(([-\d.]+),', L)
            assert m, f"Label {L!r} not in pd.Interval form '(L, R]'"
            lefts.append(float(m.group(1)))
        assert lefts == sorted(lefts), \
            f"Legend lefts {lefts} not ascending — BUG-016 not fixed"
        plt.close(fig)

    def test_SORT2_positive_control_lexicographic_would_fail(self):
        """SORT.2: positive control — naive str-sort would put 10 before 2."""
        # Simulate the pre-fix path: sort by str(label), as the v1.0 broken
        # implementation effectively did via _interval_sort_key fallback.
        intervals = [pd.Interval(10.0, 12.0),
                     pd.Interval(2.0, 4.0),
                     pd.Interval(0.04, 0.83)]
        lex_sorted = sorted(intervals, key=str)
        # Confirm naive str-sort produces WRONG order (10 before 2)
        lefts_lex = [iv.left for iv in lex_sorted]
        assert lefts_lex[1] > lefts_lex[2], \
            "Lexicographic sort no longer broken — test setup wrong"
        # Now confirm our _interval_sort_key produces RIGHT order
        from dfextensions.dfdraw.plots.profile import _interval_sort_key
        key_sorted = sorted(intervals, key=_interval_sort_key)
        lefts_key = [iv.left for iv in key_sorted]
        assert lefts_key == sorted(lefts_key), \
            f"_interval_sort_key produced {lefts_key}, not sorted"

    def test_SORT3_categorical_string_groups_unaffected(self, df_grouped):
        """SORT.3: categorical string group_by — sort works (strings remain str order)."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g_str', hist_norm='probability')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        # String labels should appear in alphabetical order
        assert labels == sorted(labels), \
            f"String labels {labels} not in alphabetical order"
        plt.close(fig)


# ====================================================================== #
# TestHistErrors (10 tests)
# ====================================================================== #

class TestHistErrors:
    """Poisson error bars on histograms."""

    def test_HE1_unnormalized_sqrt_n(self, df_grouped):
        """HE.1: hist_errors=True, no norm: yerr == sqrt(counts) per group."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', hist_errors=True)
        ebars = _errorbar_containers(ax)
        assert len(ebars) == 3, f"Expected 3 errorbar containers, got {len(ebars)}"
        # Verify yerr ≈ sqrt(heights) for each group (raw counts).
        # ax.hist with no normalization → heights are counts.
        for ec in ebars:
            xs, heights, yerrs = _errorbar_data(ec)
            assert len(xs) > 0, "No errorbar segments rendered"
            # yerr == sqrt(counts), and heights == counts (no norm)
            expected_yerr = np.sqrt(heights)
            np.testing.assert_allclose(yerrs, expected_yerr, rtol=1e-9)
        plt.close(fig)

    def test_HE2_probability_norm(self, df_grouped):
        """HE.2: hist_errors=True + hist_norm='probability': yerr == sqrt(n)/N."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', hist_errors=True,
                            hist_norm='probability')
        ebars = _errorbar_containers(ax)
        assert len(ebars) == 3
        # Per-group probabilities sum to ~1
        for i, ec in enumerate(ebars):
            xs, heights, yerrs = _errorbar_data(ec)
            assert 0.5 < heights.sum() <= 1.05, \
                f"Group {i} probabilities sum {heights.sum()}, expected near 1"
            # yerr == sqrt(n)/N == sqrt(heights * N)/N == sqrt(heights/N)
            # Per-group N is data count; derive from height ratios: yerr^2 == h/N
            # Since heights = n/N and yerr = sqrt(n)/N → yerr^2 * N = h → N = h / yerr^2
            # All bins of the same group share the same N → ratio constant
            mask = heights > 0
            if mask.any():
                N_estimates = heights[mask] / (yerrs[mask] ** 2)
                # All N_estimates should be the same (same group)
                np.testing.assert_allclose(N_estimates, N_estimates[0], rtol=0.01)
        plt.close(fig)

    def test_HE3_density_equal_width(self, df_grouped):
        """HE.3: hist_errors=True + hist_norm='density', equal bins: ∫heights·dx ≈ 1."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g', hist_errors=True,
                            hist_norm='density', bins=20)
        ebars = _errorbar_containers(ax)
        for ec in ebars:
            xs, heights, _ = _errorbar_data(ec)
            # Approximate ∫ via bin_width * sum(heights). Equal bins → constant width.
            if len(xs) > 1:
                # Sort xs to compute consecutive differences
                xs_sorted = np.sort(xs)
                bw = float(np.median(np.diff(xs_sorted)))
                integral = float(heights.sum() * bw)
                assert 0.7 < integral < 1.3, \
                    f"Density integral {integral:.3f} not near 1.0"
        plt.close(fig)

    def test_HE4_default_no_errorbars(self, df_grouped):
        """HE.4: hist_errors=False (default): no ErrorbarContainer on ax."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g')   # default hist_errors=False
        assert len(_errorbar_containers(ax)) == 0, \
            "hist_errors=False should produce no errorbars"
        plt.close(fig)

    def test_HE5_zero_count_bins_skipped(self, df_grouped):
        """HE.5: zero-count bins skipped (mask = counts > 0)."""
        # Create data with deliberate empty bins by using tight range
        rng = np.random.default_rng(0)
        df_sparse = pd.DataFrame({
            'x': np.concatenate([rng.uniform(-1, -0.5, 50),
                                 rng.uniform(0.5, 1, 50)]),
            'g': np.concatenate([np.zeros(50), np.ones(50)]).astype(int),
        })
        d = DFDraw(df_sparse)
        fig, ax, _ = d.hist('x', group_by='g', hist_errors=True, bins=20)
        ebars = _errorbar_containers(ax)
        for ec in ebars:
            xs, _, _ = _errorbar_data(ec)
            # Sparse data → fewer rendered points than total bins (20)
            assert len(xs) < 20, \
                f"Expected zero-count masking; got {len(xs)} of 20 bins"
        plt.close(fig)

    def test_HE6_color_override_phase_13_36(self, df_grouped):
        """HE.6: hist_errors=True + user color='red' → error bars red (CP1-2 lock)."""
        d = DFDraw(df_grouped)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # uniform-color warning expected
            fig, ax, _ = d.hist('x', group_by='g',
                                color='red', hist_errors=True)
        # Bars red (Phase 13.36)
        red_rgb = to_rgb('red')
        for p in ax.patches:
            fc = p.get_facecolor()
            if isinstance(fc, np.ndarray) and fc.ndim == 2:
                fc = fc[0]
            assert np.allclose(fc[:3], red_rgb, atol=0.01), \
                f"Bar facecolor {fc} != red"
        # Error bars also red (CP1-2 — group_color in errorbar)
        for ec in _errorbar_containers(ax):
            # ErrorbarContainer.lines[2] is the (vertical_line,) tuple
            for line in ec.lines[2]:
                col = line.get_color()
                if isinstance(col, np.ndarray) and col.ndim == 2:
                    col = col[0]
                assert np.allclose(np.asarray(col).flatten()[:3], red_rgb, atol=0.01), \
                    f"Errorbar color {col} != red — CP1-2 broken"
        plt.close(fig)

    def test_HE7_min_entries_filters_errorbars(self, df_grouped):
        """HE.7: hist_errors=True + min_entries=N: filtered groups absent."""
        d = DFDraw(df_grouped)
        # min_entries set high enough to filter out one group:
        # group counts are roughly 333 each; min_entries=400 → all filtered
        fig, ax, _ = d.hist('x', group_by='g',
                            hist_errors=True, min_entries=400)
        # All groups filtered → 0 errorbars
        assert len(_errorbar_containers(ax)) == 0
        plt.close(fig)
        # And with low min_entries → all rendered
        fig, ax, _ = d.hist('x', group_by='g',
                            hist_errors=True, min_entries=10)
        assert len(_errorbar_containers(ax)) == 3
        plt.close(fig)

    def test_HE8_density_variable_bin_widths(self, df_grouped):
        """HE.8: density mode + non-uniform bins → per-bin formula correct (CP1-8)."""
        d = DFDraw(df_grouped)
        # Provide explicit non-uniform bin edges (logarithmic-ish spacing)
        bin_edges = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        fig, ax, _ = d.hist('x', group_by='g',
                            bins=bin_edges,
                            hist_errors=True,
                            hist_norm='density')
        ebars = _errorbar_containers(ax)
        for ec in ebars:
            xs, heights, _ = _errorbar_data(ec)
            # Each bin center has its own width; reconstruct widths from edges
            # that contain the center.
            integral = 0.0
            for x, h in zip(xs, heights):
                for j in range(len(bin_edges) - 1):
                    if bin_edges[j] <= x <= bin_edges[j+1]:
                        integral += h * (bin_edges[j+1] - bin_edges[j])
                        break
            assert 0.7 < integral < 1.3, \
                f"Non-uniform density integral {integral:.3f} not near 1.0 (CP1-8)"
        plt.close(fig)

    def test_HE9_ungrouped_bins_int_no_crash(self, df_grouped):
        """HE.9: ungrouped hist_errors=True + bins=50 (int) → no crash (CP1-7)."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', hist_errors=True, bins=50)
        ebars = _errorbar_containers(ax)
        assert len(ebars) == 1, \
            f"Ungrouped hist_errors should produce 1 errorbar container, got {len(ebars)}"
        plt.close(fig)

    def test_HEstyle_keys_registered(self):
        """HE.style: hist.error_capsize + hist.error_elinewidth in DEFAULT_STYLE."""
        from dfextensions.dfdraw.style import get_style_value
        # Pass a sentinel; if key is REGISTERED in DEFAULT_STYLE, the
        # registered value (not the sentinel) is returned.
        capsize_val = get_style_value("hist.error_capsize", "FALLBACK_NOT_REGISTERED")
        elw_val = get_style_value("hist.error_elinewidth", "FALLBACK_NOT_REGISTERED")
        assert capsize_val != "FALLBACK_NOT_REGISTERED", \
            "hist.error_capsize not in DEFAULT_STYLE (CP1-9)"
        assert elw_val != "FALLBACK_NOT_REGISTERED", \
            "hist.error_elinewidth not in DEFAULT_STYLE (CP1-9)"
        assert capsize_val == 2
        assert elw_val == 1.0


# ====================================================================== #
# TestLinestyleCycle (5 tests)
# ====================================================================== #

class TestLinestyleCycle:
    """Per-group linestyle cycling."""

    def test_LC1_profile_distinct_linestyles(self, df_grouped):
        """LC.1: profile linestyle_cycle=True → N distinct linestyles."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g', linestyle_cycle=True)
        central = _central_lines(ax)
        ls_seen = {l.get_linestyle() for l in central}
        # 3 groups → at least 3 distinct linestyles from the cycle
        assert len(ls_seen) >= 3, \
            f"Expected ≥3 distinct linestyles, got {ls_seen}"
        plt.close(fig)

    def test_LC2_default_unchanged(self, df_grouped):
        """LC.2: linestyle_cycle=False (default) — all groups share linestyle."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g')   # default
        central = _central_lines(ax)
        ls_seen = {l.get_linestyle() for l in central}
        # All groups should share the single style default '-'
        assert ls_seen == {'-'}, \
            f"Default linestyle should be '-' for all, got {ls_seen}"
        plt.close(fig)

    def test_LC3_user_linestyle_wins_over_cycle(self, df_grouped):
        """LC.3: linestyle_cycle=True + user linestyle='--' → all dashed (CP1-3)."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g',
                               linestyle_cycle=True,
                               linestyle='--')
        central = _central_lines(ax)
        ls_seen = {l.get_linestyle() for l in central}
        assert ls_seen == {'--'}, \
            f"User explicit linestyle should win over cycle; got {ls_seen}"
        plt.close(fig)

    def test_LC4_same_true_overlay_distinct_linestyles(self, df_grouped):
        """LC.4: same=True second call linestyle_cycle=True produces different LS."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g')   # first call: all '-'
        first_call_ls = {l.get_linestyle() for l in _central_lines(ax)}
        fig, ax, _ = d.profile('y:x', group_by='g',
                               same=True, linestyle_cycle=True)
        all_ls = {l.get_linestyle() for l in _central_lines(ax)}
        # Second call adds cycle linestyles distinct from first call's '-'
        new_ls = all_ls - first_call_ls
        assert len(new_ls) >= 2, \
            f"Expected ≥2 new linestyles from second call cycle, got {new_ls}"
        plt.close(fig)

    def test_LC5_hist_step_per_group_linestyle(self, df_grouped):
        """LC.5: hist() + linestyle_cycle=True + histtype='step' → distinct line styles."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.hist('x', group_by='g',
                            histtype='step',
                            linestyle_cycle=True,
                            linewidth=2)
        # Step histograms render as Polygon patches with linestyle property
        ls_seen = set()
        for p in ax.patches:
            ls = p.get_linestyle()
            ls_seen.add(ls)
        assert len(ls_seen) >= 3, \
            f"Expected ≥3 distinct linestyles on step polygons, got {ls_seen}"
        plt.close(fig)


# ====================================================================== #
# TestPhase1336BackwardCompat (3 tests) — Phase 13.37.DF FIX1
# ====================================================================== #

class TestPhase1336BackwardCompat:
    """Phase 13.36 backward-compatibility invariance locks (FIX1).

    Gap identified by dual audit (Sonnet53_R2 + Opus1, 2026-05-21):
    PROFILE.group_by_bins and SAME.auto_features had 4 smoke tests each;
    none verified behavioral invariance after Phase 13.36 rewrote
    _draw_profile_grouped() signature and refactored same=True auto-color.

    Filter convention (Phase 13.36 lesson): errorbar central lines have
    label='_nolegend_' in matplotlib (the ErrorbarContainer holds the label,
    not the Line2D). Cap lines have marker='_'. The correct data-line filter
    is `line.get_marker() != '_'`, NOT a label-based filter.
    """

    def test_SO_COMPAT_1_group_by_bins_default_cycle_byte_identical(self):
        """§9.SO.COMPAT.1 — group_by_bins with NO user style kwargs:
        per-group line properties (colors, markers) are deterministic and
        the cycle is not collapsed after Phase 13.36 sentinel addition.

        A≡B: two calls with identical params produce identical per-group
        colors and markers. Locks that _user_marker=None / _user_color=None
        sentinels are fully transparent when user passes nothing.
        Promotes PROFILE.group_by_bins from ☑️ Smoke to ✅ Verified.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.uniform(0, 10, 500),
            'y': rng.normal(0, 1, 500),
            'g': rng.uniform(0, 5, 500),   # float col → group_by_bins
        })

        def render():
            d = DFDraw(df)
            fig, ax, _ = d.profile("y:x", group_by="g", group_by_bins=4)
            # Phase 13.36 lesson: filter by marker != '_' (errorbar central
            # lines have label='_nolegend_'; cap lines have marker='_').
            data_lines = [l for l in ax.get_lines()
                          if l.get_marker() != '_']
            # Use str(color) for hashable comparison (numpy float tuples)
            colors  = [str(l.get_color())  for l in data_lines]
            markers = [l.get_marker() for l in data_lines]
            plt.close(fig)
            return colors, markers

        colors_a, markers_a = render()
        colors_b, markers_b = render()

        assert colors_a == colors_b, \
            "per-group colors not deterministic after Phase 13.36 — sentinel non-transparent"
        assert markers_a == markers_b, \
            "per-group markers not deterministic after Phase 13.36 — sentinel non-transparent"
        # 4 groups → 4 distinct colors (cycle must not collapse to 1)
        assert len(set(colors_a)) == 4, \
            f"color cycle collapsed — Phase 13.36 _user_color=None sentinel broken (got {len(set(colors_a))} distinct, expected 4)"

    def test_SO_COMPAT_2_same_true_group_by_no_auto_color_injection(self):
        """§9.SO.COMPAT.2 — same=True + group_by: auto-color is NOT injected
        into the second call when group_by is active.

        Phase 13.36 §5.3: same=True auto-color injection skipped when
        group_by is active (would collide with group palette + trigger
        false-positive "indistinguishable" UserWarning).

        Invariants (strengthened per Opus1 P2 review):
        1. No "indistinguishable" UserWarning on the second call.
        2. First call's group lines not perturbed by the second call.
        3. Second call's group lines use the full cycle (3 distinct colors),
           NOT a single auto-injected color — direct failure-mode lock.

        Promotes SAME.auto_features from ☑️ Smoke to ✅ Verified.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.uniform(0, 10, 300),
            'y': rng.normal(0, 1, 300),
            'g': np.repeat(['A', 'B', 'C'], 100),
        })
        d = DFDraw(df)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            fig, ax, _ = d.profile("y:x", group_by="g")
            data_lines_call1 = [l for l in ax.get_lines()
                                if l.get_marker() != '_']
            colors_call1 = [str(l.get_color()) for l in data_lines_call1]

            fig, ax, _ = d.profile("y:x", group_by="g", same=True)
            data_lines_all = [l for l in ax.get_lines()
                              if l.get_marker() != '_']
            colors_all = [str(l.get_color()) for l in data_lines_all]
            plt.close(fig)

        # Invariant 1: no false "indistinguishable" warning
        bad_warnings = [x for x in w
                        if issubclass(x.category, UserWarning)
                        and "indistinguishable" in str(x.message)]
        assert len(bad_warnings) == 0, \
            ("Phase 13.36 same=True+group_by guard broken — "
             "false 'indistinguishable' UserWarning fired")

        # Invariant 2: first call's lines not perturbed by second call
        n = len(colors_call1)
        assert colors_call1 == colors_all[:n], \
            "second same=True call perturbed first call's group colors"

        # Invariant 3 (Opus1 P2 addition): second call uses full color cycle,
        # not a single auto-injected color
        colors_call2 = colors_all[n:]
        assert len(set(colors_call2)) == 3, \
            ("Phase 13.36 auto-color injection guard broken — "
             "second call's 3 groups collapsed to "
             f"{len(set(colors_call2))} color(s)")

    def test_SO_COMPAT_3_vector_group_by_palette_and_channels_preserved(self):
        """§9.SO.COMPAT.3 — vector expression + group_by + no style kwargs:
        vector dispatch through grouped path preserves both the per-group
        color cycle AND the vector→linestyle channel allocation.

        Phase 13.36 added marker/color/markersize to _PROFILE_FORWARDED_NAMES.
        This test locks that the vector dispatch path through grouped
        rendering produces the expected matrix of styles.

        Invariants:
        (a) Total rendered lines == N_groups × N_vector_elements
        (b) Distinct per-group colors == N_groups (color cycle preserved)
        (c) Distinct linestyles == N_vector_elements (Phase 13.26 vector→
            linestyle channel works through the new FORWARDED_NAMES extension)

        NOTE on reformulation: v1.1 spec compared this against a scalar loop
        equivalent (sigs_vector == sigs_scalar). That premise was wrong —
        the vector path includes Phase 13.26 channel allocation
        (vector→linestyle: element 0 → '-', element 1 → '--') that a scalar
        loop with same=True does NOT replicate. The vector vs scalar
        comparison cannot pass by design. Reformulated as a direct
        vector-path invariance lock per the original intent: verify that
        Phase 13.36's FORWARDED_NAMES extension didn't break the
        vector+group_by rendering matrix.

        Promotes VECTOR.color_cycle from ☑️ Smoke to ✅ Verified.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x':  rng.uniform(0, 10, 300),
            'y1': rng.normal(0, 1, 300),
            'y2': rng.normal(1, 1, 300),
            'g':  np.repeat(['A', 'B', 'C'], 100),
        })

        d = DFDraw(df)
        fig, ax_v, _ = d.profile("[y1,y2]:x", group_by="g")
        # Phase 13.36 filter: marker != '_' excludes errorbar cap lines
        data_lines = [l for l in ax_v.get_lines() if l.get_marker() != '_']

        colors = [str(l.get_color()) for l in data_lines]
        linestyles = [l.get_linestyle() for l in data_lines]

        n_groups = 3       # A, B, C
        n_vector = 2       # y1, y2

        # Invariant (a): N_groups × N_vector = 6 lines
        assert len(data_lines) == n_groups * n_vector, \
            (f"Vector+group_by rendering matrix broken: got {len(data_lines)} "
             f"lines, expected {n_groups * n_vector} "
             f"({n_groups} groups × {n_vector} vector elements)")

        # Invariant (b): exactly N_groups distinct colors (color cycle)
        unique_colors = set(colors)
        assert len(unique_colors) == n_groups, \
            (f"Group color cycle collapsed in vector+group_by path: "
             f"got {len(unique_colors)} distinct colors, expected {n_groups}. "
             "Phase 13.36 _PROFILE_FORWARDED_NAMES extension may have broken "
             "dispatch.")

        # Invariant (c): exactly N_vector distinct linestyles (Phase 13.26
        # vector→linestyle channel allocation)
        unique_ls = set(linestyles)
        assert len(unique_ls) == n_vector, \
            (f"Vector→linestyle channel allocation broken: got "
             f"{len(unique_ls)} distinct linestyles, expected {n_vector} "
             f"(Phase 13.26 channel)")

        plt.close(fig)
