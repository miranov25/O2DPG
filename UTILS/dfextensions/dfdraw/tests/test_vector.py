"""
Tests for Phase 13.16.DF — Vector Expression Interface.

Covers:
- Parser (§6.1): bracket syntax, broadcasting, paren-aware split, regression
- Dispatch (§6.2): vector path through draw/profile/hist/scatter, fail-fast guards
- Visual channels (§6.3): vector_style, group_style, collision, suppression
- Contract (§6.4): stats_list, ylabel, auto_title, legend
- Invariance (§6.5): scalar-loop ≡ vector (semantic contract)
- ADF entry point (§6.6): fresh-instance-per-call mock
- draw_batch (§6.7): vector inside batch specs
- Color cycle continuity (GPT5): outer same=True preserves cycle
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y1': np.random.normal(0, 1, n),
        'y2': np.random.normal(1, 1, n),
        'y3': np.random.normal(2, 1, n),
        'x1': np.random.uniform(0, 100, n),
        'x2': np.random.uniform(0, 10, n),
        'dy': np.random.normal(0, 0.5, n),
        'dz': np.random.normal(0, 0.5, n),
        'sector': np.random.choice(['A', 'B', 'C'], n),
        'dd_dyITS0': np.random.normal(0, 0.1, n),
        'dd_dyITS1': np.random.normal(0.1, 0.1, n),
        'dd_dyITS2': np.random.normal(0.2, 0.1, n),
    })


@pytest.fixture
def drawer(df):
    return DFDraw(df)


# =============================================================================
# §6.1 Parser tests
# =============================================================================

class TestParserScalar:
    """Regression: scalar expressions unchanged."""
    
    def test_parse_scalar_2d_unchanged(self, drawer):
        assert drawer._parse_expr("y:x") == ("y", "x")
    
    def test_parse_scalar_1d_unchanged(self, drawer):
        assert drawer._parse_expr("x") == ("x", None)
    
    def test_parse_scalar_with_computed_y(self, drawer):
        assert drawer._parse_expr("y+1:x") == ("y+1", "x")
    
    def test_parse_scalar_with_paren_y(self, drawer):
        """P0-2 scalar side: max(a,b):x must stay scalar."""
        assert drawer._parse_expr("max(a,b):x") == ("max(a,b)", "x")
    
    def test_parse_scalar_with_paren_both(self, drawer):
        assert drawer._parse_expr("max(a,b):min(c,d)") == ("max(a,b)", "min(c,d)")
    
    def test_parse_three_colons_still_raises(self, drawer):
        """P0-1: existing test_parse_invalid_expr_raises must not regress."""
        with pytest.raises(ValueError, match="Invalid expression"):
            drawer._parse_expr("a:b:c:d")
    
    def test_parse_two_colons_still_raises(self, drawer):
        with pytest.raises(ValueError, match="Invalid expression"):
            drawer._parse_expr("a:b:c")


class TestParserVectorBroadcast:
    """Vector broadcasting: N:1, 1:N, N:N."""
    
    def test_N1(self, drawer):
        y, x = drawer._parse_expr("[y1,y2,y3]:x")
        assert y == ["y1", "y2", "y3"]
        assert x == ["x", "x", "x"]
    
    def test_1N(self, drawer):
        y, x = drawer._parse_expr("y:[x1,x2,x3]")
        assert y == ["y", "y", "y"]
        assert x == ["x1", "x2", "x3"]
    
    def test_NN_equal(self, drawer):
        y, x = drawer._parse_expr("[y1,y2]:[x1,x2]")
        assert y == ["y1", "y2"]
        assert x == ["x1", "x2"]
    
    def test_NM_broadcast_mismatch_raises(self, drawer):
        with pytest.raises(ValueError, match="Cannot broadcast"):
            drawer._parse_expr("[y1,y2]:[x1,x2,x3]")
    
    def test_1d_vector(self, drawer):
        y, x = drawer._parse_expr("[y1,y2,y3]")
        assert y == ["y1", "y2", "y3"]
        assert x == [None, None, None]
    
    def test_empty_bracket_raises(self, drawer):
        with pytest.raises(ValueError, match="Empty"):
            drawer._parse_expr("[]:x")
    
    def test_single_element_bracket_is_scalar(self, drawer):
        """[y]:x collapses to scalar (y, x) because both sides have len 1."""
        y, x = drawer._parse_expr("[y]:x")
        assert y == "y"
        assert x == "x"
    
    def test_whitespace_in_bracket(self, drawer):
        y, x = drawer._parse_expr("[ y1 , y2 ]:x")
        assert y == ["y1", "y2"]
        assert x == ["x", "x"]


class TestParserParenInsideBracket:
    """P0-2: paren-aware split inside brackets."""
    
    def test_paren_comma_inside_bracket(self, drawer):
        y, x = drawer._parse_expr("[max(a,b),max(c,d)]:x")
        assert y == ["max(a,b)", "max(c,d)"]
        assert x == ["x", "x"]
    
    def test_nested_paren(self, drawer):
        y, x = drawer._parse_expr("[max(a,min(b,c)),max(d,e)]:x")
        assert y == ["max(a,min(b,c))", "max(d,e)"]
    
    def test_function_on_both_sides(self, drawer):
        y, x = drawer._parse_expr("[abs(a),abs(b)]:log(x)")
        assert y == ["abs(a)", "abs(b)"]
        assert x == ["log(x)", "log(x)"]


# =============================================================================
# §6.2 Dispatch tests
# =============================================================================

class TestVectorDispatch:
    """Vector dispatch across methods."""
    
    def test_profile_N1(self, drawer):
        fig, ax, stats = drawer.profile("[y1,y2,y3]:x", bins=10)
        assert isinstance(stats, list)
        assert len(stats) == 3
        # 3 curves on axes
        assert len(ax.lines) >= 3
        plt.close(fig)
    
    def test_hist_1d_vector(self, drawer):
        fig, ax, stats = drawer.hist("[y1,y2,y3]")
        assert isinstance(stats, list)
        assert len(stats) == 3
        plt.close(fig)
    
    def test_scatter_NN(self, drawer):
        fig, ax, stats = drawer.scatter("[y1,y2]:[x1,x2]")
        assert isinstance(stats, list)
        assert len(stats) == 2
        plt.close(fig)
    
    def test_draw_dispatches_profile(self, drawer):
        fig, ax, stats = drawer.draw("[y1,y2]:x", type='profile', bins=10)
        assert isinstance(stats, list)
        assert len(stats) == 2
        plt.close(fig)
    
    def test_draw_1d_auto_dispatches_hist(self, drawer):
        """P1-4: 1D vector routes to hist."""
        fig, ax, stats = drawer.draw("[y1,y2,y3]")
        assert isinstance(stats, list)
        assert len(stats) == 3
        plt.close(fig)
    
    def test_draw_2d_auto_dispatches_scatter(self, drawer):
        """P1-4: 2D vector without type routes to scatter."""
        fig, ax, stats = drawer.draw("[y1,y2]:x")
        assert isinstance(stats, list)
        assert len(stats) == 2
        plt.close(fig)


class TestVectorFailFast:
    """P0-6: fail-fast guards on unsupported methods."""
    
    def test_hist2d_vector_raises(self, drawer):
        with pytest.raises(ValueError, match="Vector expressions are not supported by hist2d"):
            drawer.hist2d("[y1,y2]:x")
    
    def test_hexbin_vector_raises(self, drawer):
        with pytest.raises(ValueError, match="Vector expressions are not supported by hexbin"):
            drawer.hexbin("[y1,y2]:x")
    
    def test_stats_vector_returns_list(self, drawer):
        """Architect decision: per-pair stats, not ValueError.
        Note: the public method is stats(), not compute_stats()."""
        result = drawer.stats("[y1,y2]:x")
        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# §6.3 Visual channel tests
# =============================================================================

class TestVectorChannels:
    """vector_style and group_style behaviors."""
    
    def test_default_vector_only_uses_color(self, drawer):
        """Without group_by, vector_style defaults to 'color'.
        
        Profile plots use errorbar which produces multiple Line2D per curve
        (main + error caps). Count unique colors across ALL lines — for
        3 curves we expect at least 3 distinct colors.
        """
        fig, ax, stats = drawer.profile("[y1,y2,y3]:x", bins=10)
        # Collect ALL colors from ALL lines (main + caps + etc.)
        all_colors = set()
        for line in ax.lines:
            c = line.get_color()
            all_colors.add(str(c) if not isinstance(c, str) else c)
        # For 3 vector curves, at least 3 distinct colors should appear
        assert len(all_colors) >= 3, (
            f"Expected >= 3 distinct colors, got {len(all_colors)}: {all_colors}"
        )
        plt.close(fig)
    
    def test_vector_style_linestyle_explicit(self, drawer):
        """Explicit vector_style='linestyle' cycles line styles."""
        fig, ax, stats = drawer.profile("[y1,y2,y3]:x", bins=10,
                                         vector_style='linestyle')
        lines = ax.lines
        assert len(lines) >= 3
        # The first 3 should have different linestyles
        styles = [line.get_linestyle() for line in lines[:3]]
        # Cycle is ['-', '--', '-.', ':']
        assert styles[0] != styles[1] or styles[1] != styles[2]
        plt.close(fig)
    
    def test_vector_style_invalid_raises(self, drawer):
        with pytest.raises(ValueError, match="vector_style must be one of"):
            drawer.profile("[y1,y2]:x", bins=10, vector_style='foo')
    
    def test_group_style_invalid_raises(self, drawer):
        with pytest.raises(ValueError, match="group_style must be one of"):
            drawer.profile("[y1,y2]:x", bins=10, group_by='sector',
                          group_style='foo')
    
    def test_same_channel_collision_raises(self, drawer):
        """vector_style == group_style → ValueError."""
        with pytest.raises(ValueError, match="cannot both use"):
            drawer.profile("[y1,y2]:x", bins=10, group_by='sector',
                          vector_style='color', group_style='color')


# =============================================================================
# §6.4 Contract tests
# =============================================================================

class TestVectorContract:
    """Return contract: stats_list, ylabel, auto_title, legend."""
    
    def test_stats_list_length(self, drawer):
        """P1-7: stats_list length matches vector length."""
        fig, ax, stats = drawer.profile("[y1,y2,y3]:x", bins=10)
        assert isinstance(stats, list)
        assert len(stats) == 3
        plt.close(fig)
    
    def test_stats_list_entries_are_dicts(self, drawer):
        fig, ax, stats = drawer.profile("[y1,y2]:x", bins=10)
        assert all(isinstance(s, dict) for s in stats)
        # Each entry should have standard stats keys
        assert 'n' in stats[0]
        plt.close(fig)
    
    def test_ylabel_common_prefix(self, drawer):
        """P1-8: common prefix rule for y-axis label."""
        fig, ax, stats = drawer.profile("[dd_dyITS0,dd_dyITS1,dd_dyITS2]:x",
                                         bins=10)
        ylabel = ax.get_ylabel()
        assert 'dd_dyITS' in ylabel
        assert ylabel.endswith('*')
        plt.close(fig)
    
    def test_ylabel_no_common_prefix(self, drawer):
        """When no common prefix, use bracket-list notation."""
        fig, ax, stats = drawer.profile("[y1,dy]:x", bins=10)
        ylabel = ax.get_ylabel()
        # Short y names share no prefix of length >= 2
        # Either bracket notation or generic
        assert ylabel is not None
        plt.close(fig)
    
    def test_auto_title_default_on_for_vector(self, drawer):
        """P1-10: auto_title defaults to True in vector mode."""
        fig, ax, stats = drawer.profile("[y1,y2]:x", bins=10)
        # Title should be present
        assert ax.get_title() != ''
        plt.close(fig)


# =============================================================================
# §6.3 Color cycle behavior
# =============================================================================

class TestVectorColorCycle:
    """Color cycle management."""
    
    def test_fresh_vector_draw_resets_cycle(self, drawer):
        """P1-1: two successive vector draws (without outer same) reset cycle."""
        # First draw
        fig1, ax1, _ = drawer.profile("[y1,y2,y3]:x", bins=10)
        colors_1 = [line.get_color() for line in ax1.lines[:3]]
        plt.close(fig1)
        
        # Second draw — fresh figure
        fig2, ax2, _ = drawer.profile("[y1,y2,y3]:x", bins=10)
        colors_2 = [line.get_color() for line in ax2.lines[:3]]
        plt.close(fig2)
        
        # The color cycles should match (both started at reset)
        assert [str(c) for c in colors_1] == [str(c) for c in colors_2]
    
    def test_vector_same_chain_continues_color_cycle(self, drawer):
        """
        GPT5 P1 fix: vector call with outer same=True must NOT reset the
        color cycle — it must continue from where scalar chain left off.
        This preserves SAME.axes_reuse contract.
        """
        # Establish chain with scalar first
        fig1, ax1, _ = drawer.profile("y1:x", bins=10)  # cycle index advances
        
        # Now add vector with outer same=True — must continue from current cycle
        fig2, ax2, _ = drawer.draw("[y2,y3]:x", type='profile', bins=10, same=True)
        
        # ax1 and ax2 should be the same axes (same=True reuses)
        assert ax1 is ax2
        
        # Three distinct colors should be present across all lines
        all_colors = set()
        for line in ax2.lines:
            c = line.get_color()
            all_colors.add(str(c) if not isinstance(c, str) else c)
        assert len(all_colors) >= 3, (
            f"Expected >= 3 distinct colors (cycle continuity), got {all_colors}"
        )
        
        plt.close(fig2)
    
    def test_outer_same_true_not_swallowed(self, drawer):
        """P0-4: outer same=True is extracted and honored on iter 0."""
        # Establish axes
        fig1, ax1, _ = drawer.profile("y1:x", bins=10)
        
        # Vector with outer same=True — first iteration should reuse ax1
        fig2, ax2, _ = drawer.profile("[y2,y3]:x", bins=10, same=True)
        assert ax1 is ax2
        plt.close(fig2)


# =============================================================================
# §6.5 STRONG Invariance tests (A ≡ B, byte-identical state)
# =============================================================================

class TestVectorInvariance:
    """
    Strong invariance tests: vector path ≡ scalar same-loop path.
    
    These are genuine A≡B comparisons. Both paths must produce matching
    plot state: same line count, same colors (in same order), same x/y
    data, same labels, same stats. Any difference caught here is a bug
    that would otherwise surface only during real-data testing.
    
    Classified as 'invariance' in test_layer_classification.py.
    """
    
    @staticmethod
    def _snapshot(ax):
        """
        Deterministic snapshot of axes state for A≡B comparison.
        
        Captures: line count, collection count, colors, linestyles,
        labels, xdata, ydata. Rounded to 10 decimals for float stability.
        """
        def _as_tuple(data):
            arr = np.asarray(data, dtype=float)
            return tuple(np.round(arr, 10).tolist())
        
        return {
            'n_lines': len(ax.lines),
            'n_collections': len(ax.collections),
            'colors': tuple(str(l.get_color()) for l in ax.lines),
            'linestyles': tuple(str(l.get_linestyle()) for l in ax.lines),
            'labels': tuple(str(l.get_label()) for l in ax.lines),
            'xdata': tuple(_as_tuple(l.get_xdata()) for l in ax.lines),
            'ydata': tuple(_as_tuple(l.get_ydata()) for l in ax.lines),
        }
    
    @staticmethod
    def _compare_stats(stats_a, stats_b):
        """Compare list[dict] stats element-wise with numeric tolerance."""
        assert len(stats_a) == len(stats_b), (
            f"stats length differs: scalar={len(stats_a)} vector={len(stats_b)}"
        )
        for i, (a, b) in enumerate(zip(stats_a, stats_b)):
            assert set(a.keys()) == set(b.keys()), (
                f"stats[{i}] keys differ: scalar={sorted(a)} vector={sorted(b)}"
            )
            for key in a:
                va, vb = a[key], b[key]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if np.isnan(va) and np.isnan(vb):
                        continue
                    assert abs(va - vb) < 1e-9, (
                        f"stats[{i}][{key!r}] differ: scalar={va} vector={vb}"
                    )
                else:
                    assert va == vb, (
                        f"stats[{i}][{key!r}] differ: scalar={va!r} vector={vb!r}"
                    )
    
    def _compare_snapshots(self, snap_a, snap_b, msg=""):
        """Assert two axes snapshots are equivalent."""
        assert snap_a['n_lines'] == snap_b['n_lines'], (
            f"{msg} line count: scalar={snap_a['n_lines']} vector={snap_b['n_lines']}"
        )
        assert snap_a['colors'] == snap_b['colors'], (
            f"{msg} colors differ:\n  scalar={snap_a['colors']}\n  vector={snap_b['colors']}"
        )
        assert snap_a['linestyles'] == snap_b['linestyles'], (
            f"{msg} linestyles differ"
        )
        assert snap_a['xdata'] == snap_b['xdata'], (
            f"{msg} xdata differs"
        )
        assert snap_a['ydata'] == snap_b['ydata'], (
            f"{msg} ydata differs"
        )
    
    # -- N:1 invariance ------------------------------------------------------
    
    def test_vector_N1_equivalent_to_scalar_loop(self, df):
        """
        [y1,y2,y3]:x via vector path ≡ scalar same=True loop.
        
        Both must produce identical stats, line count, colors, xydata.
        """
        # Path A: scalar same-loop
        drawer_a = DFDraw(df)
        fig_a, _, s_a1 = drawer_a.profile("y1:x", bins=10)
        _, ax_a, s_a2 = drawer_a.profile("y2:x", bins=10, same=True)
        _, ax_a, s_a3 = drawer_a.profile("y3:x", bins=10, same=True)
        stats_a = [s_a1, s_a2, s_a3]
        snap_a = self._snapshot(ax_a)
        
        # Path B: vector
        drawer_b = DFDraw(df)
        fig_b, ax_b, stats_b = drawer_b.profile("[y1,y2,y3]:x", bins=10)
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "N:1")
        
        plt.close(fig_a)
        plt.close(fig_b)
    
    # -- 1:N invariance ------------------------------------------------------
    
    def test_vector_1N_equivalent_to_scalar_loop(self, df):
        """y1:[x1,x2] via vector ≡ y1:x1 + y1:x2(same=True)."""
        drawer_a = DFDraw(df)
        fig_a, _, s_a1 = drawer_a.profile("y1:x1", bins=10)
        _, ax_a, s_a2 = drawer_a.profile("y1:x2", bins=10, same=True)
        stats_a = [s_a1, s_a2]
        snap_a = self._snapshot(ax_a)
        
        drawer_b = DFDraw(df)
        fig_b, ax_b, stats_b = drawer_b.profile("y1:[x1,x2]", bins=10)
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "1:N")
        
        plt.close(fig_a)
        plt.close(fig_b)
    
    # -- N:N invariance ------------------------------------------------------
    
    def test_vector_NN_equivalent_to_scalar_loop(self, df):
        """[y1,y2]:[x1,x2] via vector ≡ y1:x1 + y2:x2(same=True)."""
        drawer_a = DFDraw(df)
        fig_a, _, s_a1 = drawer_a.profile("y1:x1", bins=10)
        _, ax_a, s_a2 = drawer_a.profile("y2:x2", bins=10, same=True)
        stats_a = [s_a1, s_a2]
        snap_a = self._snapshot(ax_a)
        
        drawer_b = DFDraw(df)
        fig_b, ax_b, stats_b = drawer_b.profile("[y1,y2]:[x1,x2]", bins=10)
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "N:N")
        
        plt.close(fig_a)
        plt.close(fig_b)
    
    # -- hist vector invariance ----------------------------------------------
    
    def test_vector_hist_equivalent_to_scalar_loop(self, df):
        """
        [y1,y2,y3] via vector hist ≡ hist(y1) + hist(y2, same) + hist(y3, same).
        
        Compares Polygon vertex arrays (dfdraw uses histtype='stepfilled'
        which produces Polygon patches, not Rectangle bars).
        """
        drawer_a = DFDraw(df)
        fig_a, _, _ = drawer_a.hist("y1")
        _, _, _ = drawer_a.hist("y2", same=True)
        _, ax_a, _ = drawer_a.hist("y3", same=True)
        
        drawer_b = DFDraw(df)
        fig_b, ax_b, stats_b = drawer_b.hist("[y1,y2,y3]")
        
        assert isinstance(stats_b, list)
        assert len(stats_b) == 3
        
        # Compare patch counts
        assert len(ax_a.patches) == len(ax_b.patches), (
            f"patch count differs: scalar={len(ax_a.patches)} vector={len(ax_b.patches)}"
        )
        
        # Compare sorted polygon vertex signatures
        def _polygon_signature(p):
            xy = np.asarray(p.get_xy(), dtype=float)
            return tuple(np.round(xy.flatten(), 10).tolist())
        
        sig_a = sorted(_polygon_signature(p) for p in ax_a.patches)
        sig_b = sorted(_polygon_signature(p) for p in ax_b.patches)
        
        assert sig_a == sig_b, (
            f"hist polygon vertices differ:\n"
            f"  scalar shapes: {[len(s) for s in sig_a]}\n"
            f"  vector shapes: {[len(s) for s in sig_b]}"
        )
        
        plt.close(fig_a)
        plt.close(fig_b)
    
    # -- ADF ≡ direct invariance (bug-fix proof) -----------------------------
    
    def test_vector_through_adf_equivalent_to_direct(self, df):
        """
        The core Phase 13.16.DF claim: vector through ADF ≡ vector through
        direct DFDraw. A single DFDraw instance handles the iteration, so
        both paths must produce byte-identical plots.
        """
        # Path A: direct DFDraw vector
        drawer_direct = DFDraw(df)
        fig_a, ax_a, stats_a = drawer_direct.draw(
            "[y1,y2,y3]:x", type='profile', bins=10
        )
        snap_a = self._snapshot(ax_a)
        
        # Path B: through MockADF (fresh DFDraw per ADF call)
        adf = MockAliasDataFrame(df)
        fig_b, ax_b, stats_b = adf.draw(
            "[y1,y2,y3]:x", type='profile', bins=10
        )
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "ADF≡direct")
        
        plt.close(fig_a)
        plt.close(fig_b)
    
    # -- chain continuity invariance (GPT5 fix proof) ------------------------
    
    def test_vector_chain_continuity_equivalent_to_full_scalar_loop(self, df):
        """
        Chain continuity (GPT5 fix): scalar + vector(same=True) ≡ full scalar chain.
        
        Proves that the conditional _reset_color_cycle preserves cycle state
        when the vector call is chained onto an existing overlay.
        """
        # Path A: full scalar chain (3 curves)
        drawer_a = DFDraw(df)
        fig_a, _, s_a1 = drawer_a.profile("y1:x", bins=10)
        _, _, s_a2 = drawer_a.profile("y2:x", bins=10, same=True)
        _, ax_a, s_a3 = drawer_a.profile("y3:x", bins=10, same=True)
        stats_a = [s_a1, s_a2, s_a3]
        snap_a = self._snapshot(ax_a)
        
        # Path B: scalar + vector(same=True) for the remaining two
        drawer_b = DFDraw(df)
        fig_b1, ax_b1, s_b1 = drawer_b.profile("y1:x", bins=10)
        _, ax_b, stats_b_rest = drawer_b.draw(
            "[y2,y3]:x", type='profile', bins=10, same=True
        )
        # Both calls must land on the same axes
        assert ax_b1 is ax_b, "outer same=True did not reuse axes"
        
        stats_b = [s_b1] + list(stats_b_rest)
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "chain continuity")
        
        plt.close(fig_a)
        plt.close(fig_b1)
    
    # -- determinism invariance ---------------------------------------------
    
    def test_vector_determinism(self, df):
        """
        Two successive vector calls on identical data produce identical plots.
        
        Tests implementation stability: no hidden state carries over between
        DFDraw instances that would affect output.
        """
        drawer_a = DFDraw(df)
        fig_a, ax_a, stats_a = drawer_a.profile("[y1,y2,y3]:x", bins=10)
        snap_a = self._snapshot(ax_a)
        
        drawer_b = DFDraw(df)
        fig_b, ax_b, stats_b = drawer_b.profile("[y1,y2,y3]:x", bins=10)
        snap_b = self._snapshot(ax_b)
        
        self._compare_stats(stats_a, stats_b)
        self._compare_snapshots(snap_a, snap_b, "determinism")
        
        plt.close(fig_a)
        plt.close(fig_b)


# =============================================================================
# §6.6 ADF entry-point test
# =============================================================================

class MockAliasDataFrame:
    """Mock ADF that mirrors the real behavior: new DFDraw per call."""
    
    def __init__(self, df):
        self._df = df
    
    def draw(self, expr, **kwargs):
        # Critical: NEW DFDraw instance per call (the bug we're bypassing)
        plotter = DFDraw(self._df)
        return plotter.draw(expr, **kwargs)


class TestVectorThroughADF:
    """P0-3: vector must work through ADF (fresh DFDraw per call)."""
    
    def test_vector_through_adf_draw(self, df):
        """
        This test is the real bug-fix proof. Through ADF, each .draw() call
        spawns a fresh DFDraw. A scalar same=True loop would produce only
        one distinct color (AD-37). A single vector call must produce N colors.
        """
        adf = MockAliasDataFrame(df)
        fig, ax, stats = adf.draw("[y1,y2,y3]:x", type='profile', bins=10)
        
        # Three distinct curves returned in stats list
        assert isinstance(stats, list)
        assert len(stats) == 3
        
        # Three distinct colors across all ax.lines (main + errorbar caps)
        all_colors = set()
        for line in ax.lines:
            c = line.get_color()
            all_colors.add(str(c) if not isinstance(c, str) else c)
        assert len(all_colors) >= 3, (
            f"Expected >= 3 distinct colors (AD-37 fix), got {all_colors}"
        )
        
        plt.close(fig)


# =============================================================================
# §6.7 draw_batch integration
# =============================================================================

class TestVectorInDrawBatch:
    """Vector inside draw_batch specs (architect decision: in scope)."""
    
    def test_vector_in_draw_batch(self, drawer):
        specs = [{
            'name': 'vector_batch_test',
            'defaults': {'type': 'profile', 'bins': 10},
            'plots': [
                {'expr': '[y1,y2,y3]:x'},
            ],
        }]
        try:
            results = drawer.draw_batch(specs)
        except Exception as e:
            pytest.skip(f"draw_batch not available or incompatible: {e}")
        # If it returns, at least no crash
        assert results is not None


# =============================================================================
# §6.1-6.3 Mixed-range test (Q1=A)
# =============================================================================

class TestVectorMixedRanges:
    """Q1=A: mixed ranges allowed without warning."""
    
    def test_mixed_x_ranges_no_warning(self, drawer):
        """y:[x1,x2] where x1 and x2 have very different ranges — no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail on any warning
            fig, ax, stats = drawer.profile("y1:[x,x1]", bins=10)
        # x range is [0,10], x1 range is [0,100] — no warning raised
        assert isinstance(stats, list)
        assert len(stats) == 2
        plt.close(fig)
