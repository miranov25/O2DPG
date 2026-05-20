"""
Phase 13.36.DF — User style kwargs override auto-cycle in group_by path.

§9 invariance test plan (10 tests, locks BUG-013 fix and v1.2 architecture).

Bug closed: marker='s', color='red', markersize=10 silently ignored when
group_by was active. Per-group cycle ran unconditionally, discarding user
style kwargs.

Architect's primary use case (TPC/ITS calibration overlay):
    adf.draw("y:row", group_by="drift", group_by_bins=5)
    adf.draw("y:row", group_by="drift", group_by_bins=5,
             same=True, marker='s')

Test filter convention (Line2D from ax.errorbar):
  - ax.get_lines() returns both central data lines AND errorbar cap lines.
  - Cap lines have marker='_' (matplotlib convention for horizontal caps).
  - Central data lines have the user-meaningful marker.
  - All Line2D objects have label='_nolegend_' because errorbar's label is
    on the ErrorbarContainer, not the Line2D children.
  → Filter by `line.get_marker() != '_'` to get only central lines.
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


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def df_grouped():
    """Three-group DataFrame for group_by tests."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        'x': rng.uniform(-1, 1, n),
        'y': rng.normal(0, 1, n),
        'g': rng.integers(0, 3, n),
        'y1': rng.normal(0, 1, n),  # for vector path test
        'y2': rng.normal(0, 1, n),
    })


def _central_lines(ax):
    """Errorbar central data lines (excludes cap lines with marker='_')."""
    return [line for line in ax.get_lines() if line.get_marker() != '_']


def _line_color_rgb(line):
    """Normalize Line2D color to (R, G, B) tuple."""
    c = line.get_color()
    if isinstance(c, str):
        return to_rgb(c)
    return tuple(c[:3])


# ----------------------------------------------------------------------
# §9.SO — TestUserStyleOverride (5 tests)
# ----------------------------------------------------------------------

class TestUserStyleOverride:
    """User-passed marker/color/markersize overrides per-group cycle."""

    def test_SO1_marker_override_uniform(self, df_grouped):
        """SO.1: marker='*' applies to ALL groups (cycle ignored)."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g', marker='*')
        markers = {line.get_marker() for line in _central_lines(ax)}
        assert markers == {'*'}, f"Expected all *, got {markers}"
        plt.close(fig)

    def test_SO2_color_override_with_warning(self, df_grouped):
        """SO.2: color='blue' applies uniformly + triggers UserWarning."""
        d = DFDraw(df_grouped)
        with pytest.warns(UserWarning, match=r"indistinguishable"):
            fig, ax, _ = d.profile('y:x', group_by='g', color='blue')
        blue_rgb = to_rgb('blue')
        for line in _central_lines(ax):
            assert np.allclose(_line_color_rgb(line), blue_rgb, atol=0.01), \
                f"Line color {line.get_color()} != blue"
        plt.close(fig)

    def test_SO3_markersize_override(self, df_grouped):
        """SO.3: markersize=12 applies uniformly to all groups."""
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g', markersize=12)
        for line in _central_lines(ax):
            assert abs(line.get_markersize() - 12) < 0.1, \
                f"Line markersize {line.get_markersize()} != 12"
        plt.close(fig)

    def test_SO4_default_cycle_preserved(self, df_grouped):
        """SO.4: with NO user override, default cycle (tab10) is preserved.

        Locks backward compatibility: Phase 13.36 doesn't change behavior
        for users who don't pass marker/color/markersize.
        """
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g')
        colors_seen = [_line_color_rgb(line) for line in _central_lines(ax)]
        # 3 groups → 3 distinct tab10 colors expected
        unique_colors = {tuple(np.round(c, 3)) for c in colors_seen}
        assert len(unique_colors) == 3, \
            f"Expected 3 distinct tab10 colors, got {len(unique_colors)}"
        # Confirm they match tab10[0..2]
        tab10 = plt.colormaps.get_cmap('tab10')
        expected = [tuple(np.round(tab10(i)[:3], 3)) for i in range(3)]
        for c in colors_seen:
            assert tuple(np.round(c, 3)) in expected, \
                f"Color {c} not in tab10[0..2]"
        plt.close(fig)

    def test_SO5_bug013_same_true_marker_overlay(self, df_grouped):
        """SO.5: BUG-013 architect primary use case.

        First call: cycle markers (o, s, ^) — per-group differentiation.
        Second call same=True + marker='s': all groups override to 's'.
        """
        d = DFDraw(df_grouped)
        # No UserWarning expected — user didn't pass color
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig, ax, _ = d.profile('y:x', group_by='g')
            fig, ax, _ = d.profile('y:x', group_by='g', same=True, marker='s')
            # No 'indistinguishable' warnings (user didn't pass color)
            indist = [wi for wi in w if 'indistinguishable' in str(wi.message)]
            assert len(indist) == 0, \
                f"Unexpected indistinguishable warning(s): {[str(wi.message) for wi in indist]}"
        markers = [line.get_marker() for line in _central_lines(ax)]
        # First call: o, s, ^ (default cycle for 3 groups)
        # Second call: all 's' (user override)
        assert markers[:3] == ['o', 's', '^'], \
            f"First call markers {markers[:3]} != default cycle"
        assert set(markers[3:]) == {'s'}, \
            f"Second call markers {markers[3:]} not all 's'"
        plt.close(fig)


# ----------------------------------------------------------------------
# §9.SOH — TestHistStyleOverride (2 tests)
# ----------------------------------------------------------------------

class TestHistStyleOverride:
    """Hist user style overrides (color works; marker triggers warning)."""

    def test_SOH1_marker_warns_no_effect(self, df_grouped):
        """SOH.1: marker= triggers UserWarning (ax.hist accepts no marker)."""
        d = DFDraw(df_grouped)
        with pytest.warns(UserWarning, match=r"no effect"):
            fig, ax, _ = d.hist('x', group_by='g', marker='s')
        plt.close(fig)

    def test_SOH2_color_uniform_step_histtype(self, df_grouped):
        """SOH.2: color='blue' applies to all group histograms (step mode).

        For histtype='step', matplotlib stores the color in patch.facecolor
        (with alpha=0 for transparency); edgecolor is a stylesheet default.
        Verify the RGB component of facecolor matches the user's color.
        """
        d = DFDraw(df_grouped)
        with pytest.warns(UserWarning, match=r"indistinguishable"):
            fig, ax, _ = d.hist('x', group_by='g',
                                color='blue', histtype='step')
        blue_rgb = to_rgb('blue')
        # Step histograms produce Polygon patches; facecolor RGB carries the
        # user color even though alpha=0 makes it transparent.
        assert len(ax.patches) > 0, "No patches rendered"
        for patch in ax.patches:
            fc = patch.get_facecolor()
            if isinstance(fc, np.ndarray) and fc.ndim == 2:
                fc = fc[0]
            assert np.allclose(fc[:3], blue_rgb, atol=0.01), \
                f"Patch facecolor RGB {fc[:3]} != blue"
        plt.close(fig)


# ----------------------------------------------------------------------
# §9.LS — TestLinestyleDocumentationLock (1 test)
# ----------------------------------------------------------------------

class TestLinestyleDocumentationLock:
    """linestyle was OUT OF SCOPE for Phase 13.36 — works today, lock it."""

    def test_LS1_linestyle_None_group_by(self, df_grouped):
        """LS.1: linestyle='None' in group_by path — regression lock.

        Brainstorm panel + v1.0 panel live-tested 2026-05-20: all 9 lines
        render with linestyle='None' as requested. Locks that Phase 13.36
        did NOT regress this existing behavior.
        """
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('y:x', group_by='g', linestyle='None')
        for line in _central_lines(ax):
            ls = line.get_linestyle()
            assert ls in ('None', 'none', '', ' '), \
                f"Line linestyle {ls!r} != 'None'"
        plt.close(fig)


# ----------------------------------------------------------------------
# §9.SC — TestScatterUntouched (1 test)
# ----------------------------------------------------------------------

class TestScatterUntouched:
    """Phase 13.36 must NOT touch the scatter path."""

    def test_SC1_scatter_color_marker_still_work(self, df_grouped):
        """SC.1: scatter(color, marker) works (already in
        _SCATTER_FORWARDED_NAMES).
        """
        d = DFDraw(df_grouped)
        # No exception — scatter still accepts color/marker via existing path
        fig, ax, _ = d.scatter('x:y', color='red', marker='*')
        assert len(ax.collections) > 0, "No scatter collection rendered"
        coll = ax.collections[0]
        fc = coll.get_facecolor()
        # facecolor can be (N, 4) or (1, 4)
        if fc.ndim == 2:
            fc = fc[0]
        assert np.allclose(fc[:3], to_rgb('red'), atol=0.01), \
            f"Scatter facecolor {fc} != red"
        plt.close(fig)


# ----------------------------------------------------------------------
# §9.VF — TestVectorPathForwarding (1 test)
# ----------------------------------------------------------------------

class TestVectorPathForwarding:
    """marker/color/markersize reach grouped path via vector dispatch."""

    def test_VF1_marker_via_vector_expression(self, df_grouped):
        """VF.1: vector expression "[y1,y2]:x" + marker='s' applies uniformly.

        Engages the vector dispatch path which uses _PROFILE_FORWARDED_NAMES.
        Phase 13.36 added marker/color/markersize to that tuple so they're
        not stripped before reaching draw_profile.
        """
        d = DFDraw(df_grouped)
        fig, ax, _ = d.profile('[y1,y2]:x', marker='s')
        markers = {line.get_marker() for line in _central_lines(ax)}
        assert 's' in markers, \
            f"Vector path markers {markers} missing 's'"
        # All vector elements should have marker='s' (no per-element cycle)
        assert markers == {'s'}, \
            f"Vector path markers {markers} not uniform 's'"
        plt.close(fig)
