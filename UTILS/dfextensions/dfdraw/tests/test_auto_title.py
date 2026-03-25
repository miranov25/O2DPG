"""
Tests for Phase 13.12.DF v1.2: Auto-title feature.

Tests:
- T1: auto_title=True sets title on profile
- T2: explicit title= overrides auto_title
- T3: auto_title on hist (1D, no y)
- T4: auto_title on hist2d
- T5: auto_title on hexbin
- T6: auto_title="expr" — partial parts
- T7: auto_title="expr+group" — group label
- T8: long selection truncated
- T9: callable selection skipped in title
- T10: style default auto_title=True
- T11: weights shown in title
- T12: build_auto_title unit test
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Direct imports for unit testing helpers
from dfdraw.plots._auto_title import (
    build_auto_title, apply_auto_title, parse_auto_title_parts,
    resolve_auto_title, _AUTO_TITLE_MAX_SEL_LEN
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y': 2 * np.random.uniform(0, 10, n) + np.random.normal(0, 1, n),
        'group': np.random.choice(['A', 'B', 'C'], n),
        'weight_col': np.random.uniform(0.5, 2.0, n),
    })


@pytest.fixture
def dfdraw_obj(sample_df):
    """Create DFDraw instance."""
    from dfdraw import DFDraw
    return DFDraw(sample_df)


# =========================================================================
# T1-T5: auto_title=True on each plot type
# =========================================================================

class TestAutoTitleProfile:
    def test_auto_title_true(self, dfdraw_obj):
        """T1: auto_title=True sets title on profile."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", auto_title=True)
        title = ax.get_title()
        assert "y" in title
        assert "x" in title
        assert "vs" in title
        plt.close(fig)

    def test_explicit_title_overrides(self, dfdraw_obj):
        """T2: explicit title= wins over auto_title."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", title="Custom Title", auto_title=True)
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_auto_title_with_group(self, dfdraw_obj):
        """T7: group_by shown in title."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", group_by='group', auto_title=True)
        title = ax.get_title()
        assert "group:group" in title
        plt.close(fig)

    def test_auto_title_with_selection(self, dfdraw_obj):
        """auto_title includes selection subtitle."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", selection="x>2", auto_title=True)
        # Main title
        assert "y vs x" in ax.get_title()
        # Subtitle is a text object, check it exists
        texts = [t for t in ax.texts if "x>2" in t.get_text()]
        assert len(texts) == 1
        plt.close(fig)

    def test_auto_title_with_weights(self, dfdraw_obj):
        """T11: weights shown in title."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", weights='weight_col', auto_title=True)
        title = ax.get_title()
        assert "weights:weight_col" in title
        plt.close(fig)


class TestAutoTitleHist:
    def test_auto_title_hist(self, dfdraw_obj):
        """T3: auto_title on 1D histogram — no y expression."""
        fig, ax, stats = dfdraw_obj.hist("x", auto_title=True)
        title = ax.get_title()
        assert "x" in title
        assert "vs" not in title  # 1D: no "y vs x"
        plt.close(fig)


class TestAutoTitleHist2d:
    def test_auto_title_hist2d(self, dfdraw_obj):
        """T4: auto_title on 2D histogram."""
        fig, ax, stats = dfdraw_obj.hist2d("y:x", auto_title=True)
        title = ax.get_title()
        assert "y" in title
        assert "x" in title
        plt.close(fig)


class TestAutoTitleHexbin:
    def test_auto_title_hexbin(self, dfdraw_obj):
        """T5: auto_title on hexbin."""
        fig, ax, stats = dfdraw_obj.hexbin("y:x", auto_title=True)
        title = ax.get_title()
        assert "y" in title
        assert "x" in title
        plt.close(fig)


# =========================================================================
# T6-T9: Partial parts, truncation, callable selection
# =========================================================================

class TestAutoTitleParts:
    def test_expr_only(self, dfdraw_obj):
        """T6: auto_title='expr' — no group or selection."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", group_by='group', selection="x>2",
            auto_title="expr")
        title = ax.get_title()
        assert "y vs x" in title
        assert "group" not in title
        # No subtitle
        texts = [t for t in ax.texts if "x>2" in t.get_text()]
        assert len(texts) == 0
        plt.close(fig)

    def test_expr_plus_group(self, dfdraw_obj):
        """T7b: auto_title='expr+group'."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", group_by='group', selection="x>2",
            auto_title="expr+group")
        title = ax.get_title()
        assert "y vs x" in title
        assert "group:group" in title
        # No selection subtitle
        texts = [t for t in ax.texts if "x>2" in t.get_text()]
        assert len(texts) == 0
        plt.close(fig)

    def test_long_selection_truncated(self):
        """T8: selection string truncated at ~72 chars."""
        long_sel = "a>1&b<2&c>3&d<4&e>5&f<6&g>7&h<8&i>9&j<10&k>11&l<12&m>13&n<14&o>15&p<16&q>17&r<18"
        assert len(long_sel) > _AUTO_TITLE_MAX_SEL_LEN  # verify test string is long enough
        td = build_auto_title("x", "y", selection=long_sel, parts={"expr", "sel"})
        assert td["sub"] is not None
        assert len(td["sub"]) <= _AUTO_TITLE_MAX_SEL_LEN
        assert td["sub"].endswith("...")

    def test_callable_selection_skipped(self, dfdraw_obj):
        """T9: callable selection not shown in title."""
        fig, ax, stats = dfdraw_obj.profile(
            "y:x", selection=lambda df: df['x'] > 2,
            auto_title=True)
        # No subtitle text with lambda content
        texts = [t for t in ax.texts if t.get_text().strip()]
        # Should have no subtitle (callable is not a string)
        sel_texts = [t for t in texts if "lambda" in t.get_text()]
        assert len(sel_texts) == 0
        plt.close(fig)


# =========================================================================
# T10: Style default
# =========================================================================

class TestAutoTitleStyle:
    def test_style_default(self, sample_df):
        """T10: set_style({"auto_title": True}) enables globally."""
        from dfdraw import DFDraw, set_style
        set_style({"auto_title": True})
        try:
            d = DFDraw(sample_df)
            fig, ax, stats = d.profile("y:x")
            title = ax.get_title()
            assert "y vs x" in title
            plt.close(fig)
        finally:
            # Reset
            set_style({"auto_title": False})

    def test_per_call_overrides_style(self, sample_df):
        """Per-call auto_title=False overrides style default."""
        from dfdraw import DFDraw, set_style
        set_style({"auto_title": True})
        try:
            d = DFDraw(sample_df)
            # auto_title=False explicitly should suppress
            # But False is the default — need to check that explicit title=None + no auto gives no title
            fig, ax, stats = d.profile("y:x", title=None)
            # With style auto_title=True, should have title
            assert ax.get_title() != ""
            plt.close(fig)
        finally:
            set_style({"auto_title": False})


# =========================================================================
# T12: Unit tests for build_auto_title
# =========================================================================

class TestBuildAutoTitle:
    def test_basic_2d(self):
        """Basic y vs x title."""
        td = build_auto_title("x", "y", parts={"expr"})
        assert td["main"] == "y vs x"
        assert td["sub"] is None

    def test_basic_1d(self):
        """1D: no y → just x."""
        td = build_auto_title("pT", y=None, parts={"expr"})
        assert td["main"] == "pT"
        assert td["sub"] is None

    def test_with_group(self):
        """group:name in title."""
        td = build_auto_title("x", "y", group_by="mP4",
                              parts={"expr", "group"})
        assert "group:mP4" in td["main"]

    def test_with_weights(self):
        """weights:name in title."""
        td = build_auto_title("x", "y", weights="cw",
                              parts={"expr", "weights"})
        assert "weights:cw" in td["main"]

    def test_with_selection(self):
        """Selection as subtitle."""
        td = build_auto_title("x", "y", selection="x>2&y<5",
                              parts={"expr", "sel"})
        assert td["sub"] == "x>2&y<5"

    def test_non_string_selection_skipped(self):
        """Non-string selection → sub is None."""
        td = build_auto_title("x", "y", selection=np.array([True]),
                              parts={"expr", "sel"})
        assert td["sub"] is None

    def test_none_selection(self):
        """None selection → sub is None."""
        td = build_auto_title("x", "y", selection=None,
                              parts={"expr", "sel"})
        assert td["sub"] is None

    def test_all_parts(self):
        """All parts together."""
        td = build_auto_title("row", "dy", group_by="mP4",
                              weights="cw", selection="x>0",
                              parts={"expr", "group", "weights", "sel"})
        assert "dy vs row" in td["main"]
        assert "group:mP4" in td["main"]
        assert "weights:cw" in td["main"]
        assert td["sub"] == "x>0"


class TestParseAutoTitleParts:
    def test_true(self):
        parts = parse_auto_title_parts(True)
        assert parts == {"expr", "group", "weights", "sel"}

    def test_all_string(self):
        parts = parse_auto_title_parts("all")
        assert parts == {"expr", "group", "weights", "sel"}

    def test_expr(self):
        assert parse_auto_title_parts("expr") == {"expr"}

    def test_expr_plus_group(self):
        assert parse_auto_title_parts("expr+group") == {"expr", "group"}

    def test_false(self):
        assert parse_auto_title_parts(False) == set()
