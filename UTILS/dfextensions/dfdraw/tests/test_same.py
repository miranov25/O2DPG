"""
Tests for Phase 13.13.DF v1.0: same=True superposition.

AD-15: self._last_ax with plt.gca() fallback
AD-16: Auto-increment colors from palette
AD-17: Auto-generate labels from expression
AD-18: Append to title when same=True + auto_title=True
AD-35: No hard limit on title lines
AD-36: Truncation: (+N more)

Test classes:
- TestSameBasic (5 tests): same=True reuses axes, color cycling, label generation
- TestSameWithAutoTitle (4 tests): title appending, multiple overlays, subtitle merging
- TestSameOverride (3 tests): ax= precedence, explicit color=/label= override
- TestSameFallback (2 tests): plt.gca() fallback when no _last_ax
- TestSameAcrossMethods (4 tests): profile on hist2d, mixed plot types
- TestColorCycleReset (2 tests): new figure resets cycle
- TestSameLegend (2 tests): legend auto-shown, legend behavior
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dfdraw import DFDraw
from dfdraw.style import set_style


@pytest.fixture(autouse=True)
def cleanup():
    """Close all figures after each test."""
    yield
    plt.close('all')
    set_style(None)  # Reset style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    x = np.random.uniform(0, 10, n)
    y1 = 2 * x + np.random.normal(0, 1, n)
    y2 = -x + 5 + np.random.normal(0, 1, n)
    y3 = x ** 0.5 + np.random.normal(0, 0.5, n)
    return pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3})


@pytest.fixture
def plotter(sample_df):
    """Create DFDraw instance."""
    return DFDraw(sample_df)


# =========================================================================
# TestSameBasic: 5 tests
# =========================================================================

class TestSameBasic:
    """AD-15: same=True reuses axes."""

    def test_same_reuses_axes_profile(self, plotter):
        """same=True returns same axes as previous draw."""
        fig1, ax1, _ = plotter.profile("y1:x")
        fig2, ax2, _ = plotter.profile("y2:x", same=True)
        assert ax1 is ax2, "same=True should reuse axes"
        assert fig1 is fig2, "same=True should reuse figure"

    def test_same_reuses_axes_hist(self, plotter):
        """same=True works for histograms."""
        fig1, ax1, _ = plotter.hist("y1")
        fig2, ax2, _ = plotter.hist("y2", same=True)
        assert ax1 is ax2, "same=True should reuse axes for hist"

    def test_same_false_creates_new(self, plotter):
        """same=False (default) creates new figure."""
        fig1, ax1, _ = plotter.profile("y1:x")
        fig2, ax2, _ = plotter.profile("y2:x")  # same=False by default
        assert ax1 is not ax2, "Default should create new axes"

    def test_same_stores_last_ax(self, plotter):
        """_last_ax is updated after each draw."""
        assert plotter._last_ax is None
        fig, ax, _ = plotter.profile("y1:x")
        assert plotter._last_ax is ax

    def test_same_three_overlays(self, plotter):
        """Three consecutive same=True calls all use same axes."""
        fig1, ax1, _ = plotter.profile("y1:x")
        fig2, ax2, _ = plotter.profile("y2:x", same=True)
        fig3, ax3, _ = plotter.profile("y3:x", same=True)
        assert ax1 is ax2 is ax3, "All three should share axes"


# =========================================================================
# TestSameWithAutoTitle: 4 tests
# =========================================================================

class TestSameWithAutoTitle:
    """AD-18: Title append when same=True + auto_title=True."""

    def test_title_append(self, plotter):
        """same=True appends to existing auto-title."""
        plotter.profile("y1:x", auto_title=True)
        plotter.profile("y2:x", same=True, auto_title=True)
        ax = plotter._last_ax
        title = ax.get_title()
        assert "y1 vs x" in title
        assert "y2 vs x" in title

    def test_title_multiline(self, plotter):
        """Multiple overlays produce multi-line title (AD-35: no limit)."""
        plotter.profile("y1:x", auto_title=True)
        plotter.profile("y2:x", same=True, auto_title=True)
        plotter.profile("y3:x", same=True, auto_title=True)
        ax = plotter._last_ax
        title = ax.get_title()
        lines = title.strip().split('\n')
        assert len(lines) == 3, f"Expected 3 title lines, got {len(lines)}"

    def test_explicit_title_overrides(self, plotter):
        """Explicit title= replaces all (§5.4 Precedence Rule 1)."""
        plotter.profile("y1:x", title="Custom Title")
        ax = plotter._last_ax
        assert ax.get_title() == "Custom Title"

    def test_subtitle_merge(self, plotter):
        """Selections merge in subtitle with semicolons."""
        plotter.profile("y1:x", auto_title=True, selection="x<5")
        plotter.profile("y2:x", same=True, auto_title=True, selection="x>=5")
        ax = plotter._last_ax
        # Check subtitle texts
        subs = [t for t in ax.texts if getattr(t, '_is_auto_subtitle', False)]
        assert len(subs) > 0, "Should have subtitle artist"
        sub_text = subs[0].get_text()
        assert "x<5" in sub_text
        assert "x>=5" in sub_text


# =========================================================================
# TestSameOverride: 3 tests
# =========================================================================

class TestSameOverride:
    """Explicit parameters take precedence over same=True auto-features."""

    def test_ax_precedence(self, plotter):
        """ax= takes precedence over same=True (§5.4 Rule 2)."""
        fig1, ax1, _ = plotter.profile("y1:x")
        # Create separate axes
        fig_ext, ax_ext = plt.subplots()
        fig2, ax2, _ = plotter.profile("y2:x", same=True, ax=ax_ext)
        assert ax2 is ax_ext, "Explicit ax= should win over same=True"
        assert ax2 is not ax1

    def test_explicit_color(self, plotter):
        """Explicit color= overrides auto-color (AD-16)."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True, color='red')
        ax = plotter._last_ax
        # The last errorbar container should use red
        containers = ax.containers
        if len(containers) >= 2:
            # errorbar creates Line2D children
            last_lines = containers[-1].get_children()
            if last_lines:
                line_color = last_lines[0].get_color()
                # 'red' maps to (1.0, 0.0, 0.0, 1.0) or 'red'
                assert line_color == 'red' or (
                    hasattr(line_color, '__len__') and line_color[0] > 0.9
                )

    def test_explicit_label(self, plotter):
        """Explicit label= overrides auto-label (AD-17)."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True, label="Custom Label")
        ax = plotter._last_ax
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Custom Label" in labels


# =========================================================================
# TestSameFallback: 2 tests
# =========================================================================

class TestSameFallback:
    """AD-15: plt.gca() fallback when _last_ax is None."""

    def test_fallback_to_gca(self, plotter):
        """same=True uses plt.gca() when _last_ax is None."""
        # Create a figure with data manually
        fig, ax_manual = plt.subplots()
        ax_manual.plot([1, 2, 3], [1, 2, 3])  # add data so has_data() is True
        # plotter has no _last_ax yet, should fall back to gca
        fig2, ax2, _ = plotter.profile("y1:x", same=True)
        assert ax2 is ax_manual, "Should fall back to plt.gca()"

    def test_fallback_no_axes_creates_new(self, plotter):
        """same=True creates new figure when no valid axes exist."""
        plt.close('all')  # ensure no figures
        # _last_ax is None, gca() creates empty default axes
        fig, ax, _ = plotter.profile("y1:x", same=True)
        assert ax is not None, "Should create new axes when nothing exists"


# =========================================================================
# TestSameAcrossMethods: 4 tests
# =========================================================================

class TestSameAcrossMethods:
    """Mixed plot types with same=True."""

    def test_profile_on_hist2d(self, plotter):
        """Profile overlay on hist2d — key use case §6.5."""
        fig1, ax1, _ = plotter.hist2d("y1:x")
        fig2, ax2, _ = plotter.profile("y1:x", same=True)
        assert ax1 is ax2, "Profile should overlay on hist2d axes"

    def test_profile_on_hexbin(self, plotter):
        """Profile overlay on hexbin."""
        fig1, ax1, _ = plotter.hexbin("y1:x")
        fig2, ax2, _ = plotter.profile("y1:x", same=True)
        assert ax1 is ax2, "Profile should overlay on hexbin axes"

    def test_hist_overlay(self, plotter):
        """Two histograms via same=True."""
        fig1, ax1, _ = plotter.hist("y1")
        fig2, ax2, _ = plotter.hist("y2", same=True)
        assert ax1 is ax2

    def test_draw_dispatch_same(self, plotter):
        """same=True works through draw() dispatcher."""
        fig1, ax1, _ = plotter.draw("y1:x", type='profile')
        fig2, ax2, _ = plotter.draw("y2:x", type='profile', same=True)
        assert ax1 is ax2, "same=True should work through draw() dispatch"


# =========================================================================
# TestColorCycleReset: 2 tests
# =========================================================================

class TestColorCycleReset:
    """Color cycle management."""

    def test_new_figure_resets_cycle(self, plotter):
        """Creating new figure resets color cycle index."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True)
        assert plotter._color_cycle_index == 2  # started at 1, incremented once
        # New figure should reset
        plotter.profile("y3:x")  # same=False (default)
        assert plotter._color_cycle_index == 1  # reset to 1

    def test_colors_differ(self, plotter):
        """Overlaid plots get different colors (AD-16)."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True)
        plotter.profile("y3:x", same=True)
        ax = plotter._last_ax
        # Collect colors from errorbar containers
        containers = ax.containers
        colors = []
        for c in containers:
            children = c.get_children()
            if children:
                colors.append(children[0].get_color())
        # Should have at least 2 distinct colors
        if len(colors) >= 2:
            # Check that not all colors are identical
            assert not all(
                np.array_equal(colors[0], c) for c in colors[1:]
            ), "Overlaid plots should have different colors"


# =========================================================================
# TestSameLegend: 2 tests
# =========================================================================

class TestSameLegend:
    """Legend behavior with same=True."""

    def test_legend_auto_shown(self, plotter):
        """Legend is automatically shown when same=True (AD-17)."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True)
        ax = plotter._last_ax
        legend = ax.get_legend()
        assert legend is not None, "Legend should be auto-shown with same=True"

    def test_auto_label_content(self, plotter):
        """Auto-generated label matches expression (AD-17)."""
        plotter.profile("y1:x")
        plotter.profile("y2:x", same=True)
        ax = plotter._last_ax
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "y2 vs x" in labels, f"Expected 'y2 vs x' in {labels}"
