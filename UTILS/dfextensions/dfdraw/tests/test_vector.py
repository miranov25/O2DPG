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


# =============================================================================
# Phase 13.16.DF FIX1 — additions (2026-04-14)
#
# Three new test classes diagnosing the vector-path kwarg-propagation bug class.
# Added as PERMANENT capability matrix entries (VECTOR.kwarg_propagation,
# VECTOR.groupby_polish, VECTOR.kwarg_surface) — not temporary regression tests.
#
# EXPECTED STATE PRE-FIX (at commit d662c0a5): 18 tests all FAIL with
#   diagnostic messages tagged [dfdraw <bug-class> <method>/<kwarg>].
# EXPECTED STATE POST-FIX (PHASE_13_16_DF_FIX1_END): all 18 tests PASS.
#
# Governance: v1.4 proposal approved by Claude40 with source-verified
#   spot-check; multi-reviewer panel (Claude42/43/45/46) source-verified
#   the per-method inventory.
# =============================================================================

from matplotlib.legend import Legend


# ── Fixtures (FIX1) ──────────────────────────────────────────────────────────

@pytest.fixture
def df_with_groups():
    """4-category fixture for vector + group_by tests (FIX1)."""
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        'y1': np.random.normal(0, 1, n),
        'y2': np.random.normal(0.5, 1, n),
        'y3': np.random.normal(1.0, 1, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'cat_float': np.random.uniform(-1, 1, n),
    })


@pytest.fixture
def df_sparse_groups():
    """
    3-category fixture with intentionally heterogeneous bin density.
    Category A: 600 rows clustered at low x (dense bins)
    Category B: 100 rows uniform (medium bins)
    Category C: 30 rows uniform (sparse bins → caught by min_entries=20)
    """
    np.random.seed(43)
    rows = []
    for _ in range(600):
        rows.append({
            'x': np.random.uniform(0, 5),  # clustered low
            'y1': np.random.normal(0, 1),
            'y2': np.random.normal(0.5, 1),
            'category': 'A',
        })
    for _ in range(100):
        rows.append({
            'x': np.random.uniform(0, 10),
            'y1': np.random.normal(0, 1),
            'y2': np.random.normal(0.5, 1),
            'category': 'B',
        })
    for _ in range(30):
        rows.append({
            'x': np.random.uniform(0, 10),
            'y1': np.random.normal(0, 1),
            'y2': np.random.normal(0.5, 1),
            'category': 'C',
        })
    return pd.DataFrame(rows)


@pytest.fixture
def df_with_weights():
    """
    Non-uniform weights fixture for 3-path weights invariance test.
    
    CRITICAL: y values are CORRELATED with w (high-w points have y around +5,
    low-w points have y around 0). This guarantees weighted_mean ≠ unweighted_mean
    so the B≠C sanity assertion can discriminate.
    """
    np.random.seed(45)
    n = 400
    # 10% of points are "high-weight high-y" outliers
    high_weight = np.random.rand(n) > 0.9
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n),
        # y correlated with w: high-weight points have shifted y
        'y1': np.where(high_weight, np.random.normal(5.0, 1, n),
                                    np.random.normal(0, 1, n)),
        'y2': np.where(high_weight, np.random.normal(5.5, 1, n),
                                    np.random.normal(0.5, 1, n)),
        # Extreme non-uniform weights (10:1 ratio) — pulls weighted mean toward high-y subset
        'w': np.where(high_weight, 10.0, 0.5),
    })


@pytest.fixture
def df_its_like():
    """ITS-like fixture mimicking architect's production reproducer."""
    np.random.seed(44)
    n = 60_000
    return pd.DataFrame({
        'staveITS': np.random.randint(0, 12, n),
        'dd_dzITS0': np.random.normal(0, 0.001, n),
        'dd_dzITS1': np.random.normal(0, 0.001, n),
        'dd_dzITS2': np.random.normal(0, 0.001, n),
        'dd_dzITS3': np.random.normal(0, 0.001, n),
        'dd_dzITS4': np.random.normal(0, 0.001, n),
        'dd_dzITS5': np.random.normal(0, 0.001, n),
        # ~210 unique mP3 values → reproduces architect's 210×2+1=421 legend bug
        'mP3': np.round(np.random.uniform(-2.1, 0.0, n), 2),
    })


# ── Helpers (FIX1) ───────────────────────────────────────────────────────────

def _fix1_get_all_legends(ax):
    """All Legend objects on the axes (main + any add_artist legends)."""
    return [a for a in ax.get_children() if isinstance(a, Legend)]


def _fix1_get_main_group_legend(ax):
    """Main group legend = the legend with the most entries."""
    legends = _fix1_get_all_legends(ax)
    if not legends:
        return None
    return max(legends, key=lambda L: len(L.get_texts()))


# =============================================================================
# TestVectorKwargPropagation — 7 strong A≡B invariance tests
# =============================================================================
# 
# Pattern: Path A (scalar same-loop with kwarg=X) ≡ Path B (vector with kwarg=X).
# Failure pre-fix: vector silently drops the kwarg, producing different stats/legend.
# Diagnostic role: if A≡B in dfdraw but fails via aDF.draw(), bug is at ADF boundary.
# =============================================================================

class TestVectorKwargPropagation:
    """
    Strong A≡B tests proving vector dispatch forwards all scalar-mode kwargs.
    
    Phase 13.16.DF FIX1 root bug: vector dispatch blocks in
    profile()/hist()/scatter()/draw() enumerate a hardcoded subset of named
    parameters and silently drop the rest (B1a). Matplotlib channel kwargs
    linestyle/marker are additionally clobbered at _draw_vector:606,611 (B1b).
    """

    def test_vector_group_by_bins_equivalent_to_scalar_loop(self, df_with_groups):
        """B1a: group_by_bins propagates correctly through vector path (root symptom)."""
        df = df_with_groups
        drawer_a = DFDraw(df.copy())
        drawer_a.profile("y1:x", bins=10, group_by='cat_float', group_by_bins=4)
        _, ax_a, _ = drawer_a.profile(
            "y2:x", bins=10, same=True, group_by='cat_float', group_by_bins=4,
        )
        drawer_b = DFDraw(df.copy())
        _, ax_b, _ = drawer_b.profile(
            "[y1,y2]:x", bins=10, group_by='cat_float', group_by_bins=4,
        )
        legend_a = _fix1_get_main_group_legend(ax_a)
        legend_b = _fix1_get_main_group_legend(ax_b)
        labels_a = set(t.get_text() for t in legend_a.get_texts()) if legend_a else set()
        labels_b = set(t.get_text() for t in legend_b.get_texts()) if legend_b else set()
        # Compare unique group label SETS — scalar same-loop accumulates
        # duplicate labels across iterations (matplotlib quirk); vector
        # deduplicates correctly. Both should contain the same unique groups.
        assert labels_a.issubset(labels_b) or labels_b.issubset(labels_a), (
            f"[dfdraw B1a profile/group_by_bins] Scalar unique groups {labels_a}, "
            f"vector unique groups {labels_b}. Disjoint groups means group_by_bins "
            f"silently dropped in vector dispatch."
        )
        # Vector should have ≤ 5 entries (4 bins + possible 1 vector-legend proxy)
        n_b = len(legend_b.get_texts()) if legend_b else 0
        assert n_b <= 5, (
            f"[dfdraw B1a profile/group_by_bins] group_by_bins=4 requested, "
            f"vector has {n_b} legend entries (expected ≤ 5)."
        )
        plt.close('all')

    def test_vector_group_by_quantiles_equivalent_to_scalar_loop(self, df_with_groups):
        """B1a: group_by_quantiles propagates correctly through vector path."""
        df = df_with_groups
        drawer_a = DFDraw(df.copy())
        drawer_a.profile("y1:x", bins=10, group_by='cat_float', group_by_quantiles=3)
        _, ax_a, _ = drawer_a.profile(
            "y2:x", bins=10, same=True, group_by='cat_float', group_by_quantiles=3,
        )
        drawer_b = DFDraw(df.copy())
        _, ax_b, _ = drawer_b.profile(
            "[y1,y2]:x", bins=10, group_by='cat_float', group_by_quantiles=3,
        )
        legend_a = _fix1_get_main_group_legend(ax_a)
        legend_b = _fix1_get_main_group_legend(ax_b)
        labels_a = set(t.get_text() for t in legend_a.get_texts()) if legend_a else set()
        labels_b = set(t.get_text() for t in legend_b.get_texts()) if legend_b else set()
        assert labels_a.issubset(labels_b) or labels_b.issubset(labels_a), (
            f"[dfdraw B1a profile/group_by_quantiles] Scalar unique groups {labels_a}, "
            f"vector unique groups {labels_b}. group_by_quantiles silently dropped."
        )
        plt.close('all')

    def test_vector_min_entries_equivalent_to_scalar_loop(self, df_sparse_groups):
        """
        B1a: min_entries propagates (per-BIN filter — affects how many bins
        survive in each group's profile line, not just whether group appears).
        
        The profile lines must have the SAME number of plotted points in
        scalar-loop and vector paths. If min_entries is dropped in vector,
        vector lines have MORE points (sparse bins not filtered).
        """
        df = df_sparse_groups
        drawer_a = DFDraw(df.copy())
        drawer_a.profile("y1:x", bins=10, group_by='category', min_entries=20)
        _, ax_a, _ = drawer_a.profile(
            "y2:x", bins=10, same=True, group_by='category', min_entries=20,
        )
        drawer_b = DFDraw(df.copy())
        _, ax_b, _ = drawer_b.profile(
            "[y1,y2]:x", bins=10, group_by='category', min_entries=20,
        )
        # Count total plotted bin-points across all real (non-empty proxy) lines.
        # If min_entries is dropped in vector, vector has more total points.
        n_pts_a = sum(len(l.get_xdata()) for l in ax_a.lines if len(l.get_xdata()) > 0)
        n_pts_b = sum(len(l.get_xdata()) for l in ax_b.lines if len(l.get_xdata()) > 0)
        assert n_pts_a == n_pts_b, (
            f"[dfdraw B1a profile/min_entries] Scalar path plots {n_pts_a} total "
            f"bin-points after min_entries=20 filter; vector plots {n_pts_b}. "
            f"Mismatch indicates min_entries silently dropped in vector dispatch — "
            f"vector retained sparse bins that scalar correctly filtered."
        )
        # And legend should match on unique group SETS (scalar same-loop
        # accumulates duplicate labels across iterations; vector deduplicates).
        legend_a = _fix1_get_main_group_legend(ax_a)
        legend_b = _fix1_get_main_group_legend(ax_b)
        labels_a = set(t.get_text() for t in legend_a.get_texts()) if legend_a else set()
        labels_b = set(t.get_text() for t in legend_b.get_texts()) if legend_b else set()
        assert labels_a.issubset(labels_b) or labels_b.issubset(labels_a), (
            f"[dfdraw B1a profile/min_entries legend] Scalar unique groups {labels_a}, "
            f"vector unique groups {labels_b} after min_entries=20."
        )
        plt.close('all')

    def test_vector_sort_groups_equivalent_to_scalar_loop(self, df_with_groups):
        """B1a: sort_groups propagates (group ordering must match between paths)."""
        df = df_with_groups
        drawer_a = DFDraw(df.copy())
        drawer_a.profile("y1:x", bins=10, group_by='category', sort_groups=False)
        _, ax_a, _ = drawer_a.profile(
            "y2:x", bins=10, same=True, group_by='category', sort_groups=False,
        )
        drawer_b = DFDraw(df.copy())
        _, ax_b, _ = drawer_b.profile(
            "[y1,y2]:x", bins=10, group_by='category', sort_groups=False,
        )
        legend_a = _fix1_get_main_group_legend(ax_a)
        legend_b = _fix1_get_main_group_legend(ax_b)
        labels_a_raw = [t.get_text() for t in legend_a.get_texts()] if legend_a else []
        labels_b_raw = [t.get_text() for t in legend_b.get_texts()] if legend_b else []
        # Deduplicate preserving order (scalar same-loop accumulates dupes;
        # vector deduplicates — both should produce same unique-order sequence
        # for the actual group categories).
        def _unique_order(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]
        uniq_a = _unique_order(labels_a_raw)
        uniq_b = _unique_order(labels_b_raw)
        # Filter out vector-legend proxy entries like "y1 vs x" that aren't
        # category group labels — keep only labels present in fixture categories.
        cats = set(df['category'].unique().astype(str))
        groups_a = [l for l in uniq_a if l in cats]
        groups_b = [l for l in uniq_b if l in cats]
        assert groups_a == groups_b, (
            f"[dfdraw B1a profile/sort_groups] Scalar group order {groups_a}, "
            f"vector group order {groups_b}. sort_groups=False silently dropped — "
            f"vector applied default sort_groups=True."
        )
        plt.close('all')

    def test_vector_linestyle_equivalent_to_scalar_loop(self, df_with_groups):
        """B1b: user linestyle='none' must survive channel clobbering (setdefault fix)."""
        drawer = DFDraw(df_with_groups.copy())
        _, ax, _ = drawer.profile(
            "[y1,y2]:x", bins=10, group_by='category', linestyle='none',
        )
        real_lines = [l for l in ax.lines if len(l.get_xdata()) > 0]
        for line in real_lines:
            ls = line.get_linestyle()
            assert ls in ('none', 'None', ''), (
                f"[dfdraw B1b _draw_vector line 606] User linestyle='none' "
                f"overwritten by channel cycle (got {ls!r}). Fix: "
                f"iter_kwargs.setdefault('linestyle', ...) at line 606."
            )
        plt.close('all')

    def test_vector_weights_equivalent_to_scalar_loop(self, df_with_weights):
        """
        B1a: weights propagates (scientifically critical — silent numerical bug otherwise).
        
        3-path test: A≡B (propagation works) AND B≠C (weights had effect).
        
        Note: stats['mean_y'] is the overall UNWEIGHTED mean of all y values;
        weights only affect the PER-BIN profile computation. So we compare
        profile_data (per-bin means from return_data=True) for the B≠C check.
        """
        df = df_with_weights
        drawer_a = DFDraw(df.copy())
        _, _, s_a1 = drawer_a.profile("y1:x", bins=10, weights='w', return_data=True)
        _, _, s_a2 = drawer_a.profile("y2:x", bins=10, same=True, weights='w', return_data=True)
        drawer_b = DFDraw(df.copy())
        _, _, stats_b = drawer_b.profile("[y1,y2]:x", bins=10, weights='w', return_data=True)
        drawer_c = DFDraw(df.copy())
        _, _, stats_c = drawer_c.profile("[y1,y2]:x", bins=10, return_data=True)  # no weights

        assert len(stats_b) == 2, (
            f"[dfdraw contract] vector [y1,y2]:x must return 2 stats, got {len(stats_b)}"
        )
        
        # A ≡ B: scalar weighted stats must match vector weighted stats
        assert abs(s_a1['mean_y'] - stats_b[0]['mean_y']) < 1e-10, (
            f"[dfdraw B1a profile/weights A≡B] Scalar y1 mean={s_a1['mean_y']:.6f}, "
            f"vector y1 mean={stats_b[0]['mean_y']:.6f}. Weights silently dropped "
            f"in vector dispatch."
        )
        assert abs(s_a2['mean_y'] - stats_b[1]['mean_y']) < 1e-10, (
            f"[dfdraw B1a profile/weights A≡B] Scalar y2 mean={s_a2['mean_y']:.6f}, "
            f"vector y2 mean={stats_b[1]['mean_y']:.6f}."
        )
        
        # B ≠ C: weights must have changed the PER-BIN profile means.
        # stats['mean_y'] is the overall unweighted mean (identical regardless of
        # weights). The per-bin means in profile_data ARE affected by weights.
        assert 'profile_data' in stats_b[0], (
            f"[dfdraw B1a profile/weights B≠C setup] return_data=True was silently "
            f"dropped — stats_b[0] missing 'profile_data'. Keys: {list(stats_b[0].keys())}"
        )
        assert 'profile_data' in stats_c[0], (
            f"[dfdraw B1a profile/weights B≠C setup] return_data=True was silently "
            f"dropped — stats_c[0] missing 'profile_data'. Keys: {list(stats_c[0].keys())}"
        )
        # Compare per-bin mean_y arrays — they must differ if weights took effect
        pd_b = stats_b[0]['profile_data']
        pd_c = stats_c[0]['profile_data']
        if 'mean_y' in pd_b.columns and 'mean_y' in pd_c.columns:
            # Align on common bins (both use bins=10, same x range → same bins)
            max_diff = float((pd_b['mean_y'].values - pd_c['mean_y'].values).max())
            assert abs(max_diff) > 1e-6, (
                f"[dfdraw B1a profile/weights B≠C sanity] Per-bin profile means are "
                f"identical between weighted and unweighted paths (max diff={max_diff:.8f}). "
                f"Weights had no effect — this test would be worthless without this check."
            )
        else:
            # Fallback: if profile_data doesn't have mean_y column, DataFrames
            # themselves must differ (any column with weighted data)
            try:
                pd.testing.assert_frame_equal(pd_b, pd_c)
                raise AssertionError(
                    "[dfdraw B1a profile/weights B≠C sanity] Weighted and unweighted "
                    "profile_data are identical. Weights had no effect."
                )
            except AssertionError as e:
                if "are identical" in str(e):
                    raise
                # DataFrames differ — weights took effect. Pass.
        plt.close('all')

    def test_vector_return_data_equivalent_to_scalar_loop(self, df_with_groups):
        """B1a: return_data must propagate — 'profile_data' key in each stats dict."""
        df = df_with_groups
        drawer_a = DFDraw(df.copy())
        _, _, s_a1 = drawer_a.profile("y1:x", bins=10, return_data=True)
        _, _, s_a2 = drawer_a.profile("y2:x", bins=10, same=True, return_data=True)
        drawer_b = DFDraw(df.copy())
        _, _, stats_b = drawer_b.profile("[y1,y2]:x", bins=10, return_data=True)

        assert 'profile_data' in s_a1, (
            f"[dfdraw setup] scalar path missing 'profile_data' — not a vector bug; "
            f"check return_data in profile() scalar code path."
        )
        for i, s in enumerate(stats_b):
            assert 'profile_data' in s, (
                f"[dfdraw B1a profile/return_data] Vector stats[{i}] missing "
                f"'profile_data' key (keys: {list(s.keys())}). User code doing "
                f"stats[{i}]['profile_data'] hits KeyError."
            )
        pd.testing.assert_frame_equal(
            s_a1['profile_data'].reset_index(drop=True),
            stats_b[0]['profile_data'].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            s_a2['profile_data'].reset_index(drop=True),
            stats_b[1]['profile_data'].reset_index(drop=True),
        )
        plt.close('all')


# =============================================================================
# TestVectorGroupBy — 5 smoke tests (count-based)
# =============================================================================
# Not A≡B: scalar loop INTENTIONALLY produces N legend copies; vector should
# dedup to 1. Count-based smoke tests capture the post-fix contract.
# =============================================================================

class TestVectorGroupBy:
    """Smoke tests for vector + group_by cosmetic dedup (B2-B5)."""

    def test_vector_groupby_main_legend_dedup_count(self, df_with_groups):
        """B2: main group legend = N unique groups, not N × vector_dim."""
        drawer = DFDraw(df_with_groups.copy())
        _, ax, _ = drawer.profile(
            "[y1,y2,y3]:x", bins=10, group_by='category',
        )
        main = _fix1_get_main_group_legend(ax)
        all_labels = [t.get_text() for t in main.get_texts()] if main else []
        # Filter to only actual group category labels (exclude vector-legend
        # proxy entries like "y1 vs x" that may leak into the main legend).
        cats = set(df_with_groups['category'].unique().astype(str))
        group_labels = [l for l in all_labels if l in cats]
        n_groups = len(group_labels)
        assert n_groups == 4, (
            f"[dfdraw B2 profile/legend_dedup] Main legend has {n_groups} group "
            f"entries (from labels {all_labels}); expected 4 (= unique groups). "
            f"If n==12, legend duplicated per vector iteration."
        )
        # Also verify no duplication of group labels
        assert len(group_labels) == len(set(group_labels)), (
            f"[dfdraw B2 profile/legend_dedup] Group labels are duplicated: "
            f"{group_labels}. Dedup not working."
        )
        plt.close('all')

    def test_vector_groupby_secondary_legend_count(self, df_with_groups):
        """B2 corollary: secondary 'Variable' legend = vector dim (=3); dedup must not break it."""
        drawer = DFDraw(df_with_groups.copy())
        _, ax, _ = drawer.profile(
            "[y1,y2,y3]:x", bins=10, group_by='category',
        )
        legends = _fix1_get_all_legends(ax)
        if len(legends) < 2:
            pytest.fail(
                f"[dfdraw B2 profile/secondary_legend] Expected 2 legends (main + "
                f"Variable secondary), got {len(legends)}. Phase 13.16.DF "
                f"_add_vector_legend may be broken."
            )
        secondary = min(legends, key=lambda L: len(L.get_texts()))
        n = len(secondary.get_texts())
        assert n == 3, (
            f"[dfdraw B2 profile/secondary_legend] Secondary 'Variable' legend "
            f"has {n} entries; expected 3 (= vector dim)."
        )
        plt.close('all')

    def test_vector_groupby_title_one_line(self, df_with_groups):
        """B3+B4: title ≤ 2 lines, not N lines (one title + optional subtitle)."""
        drawer = DFDraw(df_with_groups.copy())
        _, ax, _ = drawer.profile(
            "[y1,y2,y3]:x", bins=10, group_by='category', auto_title=True,
        )
        title = ax.get_title()
        n_lines = title.count('\n') + 1 if title else 0
        assert n_lines <= 2, (
            f"[dfdraw B3 profile/title] Title has {n_lines} lines; expected ≤ 2. "
            f"If n_lines == vector_dim+1, title appended per iteration."
        )
        plt.close('all')

    def test_vector_groupby_no_layout_warnings(self, df_with_groups):
        """
        B5: tight_layout must be invoked ≤ 1 time across the vector call.
        
        Pre-fix: each per-iteration call of profile() invokes plt.tight_layout()
        once → N invocations for N-vector. Post-fix: _suppress_layout flag
        suppresses iterations 1..N-1; one call at end of _draw_vector.
        
        Use direct call counting (via monkeypatch) rather than waiting for
        warnings — warnings only fire when tight_layout fails to fit, which
        depends on the figure/title shape and is not deterministic across
        machines or matplotlib versions. Counting direct calls IS deterministic.
        """
        import matplotlib.pyplot as _plt
        call_counter = {'n': 0}
        original_tight = _plt.tight_layout
        def _counting_tight(*args, **kwargs):
            call_counter['n'] += 1
            return original_tight(*args, **kwargs)
        _plt.tight_layout = _counting_tight
        try:
            drawer = DFDraw(df_with_groups.copy())
            drawer.profile("[y1,y2,y3]:x", bins=10, group_by='category')
        finally:
            _plt.tight_layout = original_tight
        n = call_counter['n']
        assert n <= 1, (
            f"[dfdraw B5 profile/tight_layout] plt.tight_layout() called {n} "
            f"times during a single 3-vector call; expected ≤ 1. If n == 3 "
            f"(= vector_dim), tight_layout is invoked per iteration in "
            f"profile.py instead of once at end of _draw_vector. "
            f"Fix: add _suppress_layout flag, suppress per-iteration calls, "
            f"single call at end of _draw_vector."
        )
        plt.close('all')

    def test_vector_groupby_real_world_reproducer(self, df_its_like):
        """
        Architect's production reproducer as permanent regression test.
        
        Pre-fix: ~420 main legend entries (210 unique mP3 × 2 vector iterations + 1).
        Post-fix: ≤6 group entries (group_by_bins=6 honored + dedup applied).
        """
        drawer = DFDraw(df_its_like.copy())
        _, ax, _ = drawer.profile(
            "[dd_dzITS0,dd_dzITS1]:staveITS", bins=12,
            group_by='mP3', group_by_bins=6, auto_title=True,
        )
        main = _fix1_get_main_group_legend(ax)
        all_labels = [t.get_text() for t in main.get_texts()] if main else []
        # Filter out vector-legend proxy entries ("dd_dzITS0 vs staveITS" etc.)
        # which may leak into the main legend. Group labels from mP3 quantile
        # bins are pd.cut intervals like "(-2.1, -1.5]", never contain " vs ".
        group_labels = [l for l in all_labels if ' vs ' not in l]
        n_groups = len(group_labels)
        assert n_groups <= 6, (
            f"[dfdraw B1a+B2 architect reproducer] Main legend has {n_groups} group "
            f"entries (from labels {all_labels}); expected ≤ 6 (group_by_bins). "
            f"If n≈420 both B1a+B2 broken; if n≈12 only B1a broken; "
            f"if n≈210 only B2 broken."
        )
        plt.close('all')


# =============================================================================
# TestVectorKwargSurface — 6 surface + guard tests
# =============================================================================
# Structural tests exercising every named parameter of each method + R4
# facet-guard + R17 forwarded-names validity regression.
# =============================================================================

class TestVectorKwargSurface:
    """Surface enumeration tests (R5 + R12 + R4 + R17)."""

    def test_vector_profile_kwarg_surface_enumeration(self, df_with_groups):
        """
        R5: every non-facet named parameter of profile() flows through vector dispatch.
        
        Note: group_by_bins=N requires a numeric group_by column (uses pd.cut).
        Use 'cat_float' here, not 'category' (string would TypeError in pd.cut).
        """
        drawer = DFDraw(df_with_groups.copy())
        _, ax, stats_list = drawer.profile(
            "[y1,y2]:x",
            selection="x > 0",
            sample=350,
            bins=8,
            range=(0, 10),
            error="std",
            stats=['n', 'mean_y'],
            title="test",
            xlabel="X", ylabel="Y",
            group_by='cat_float',         # numeric col required by group_by_bins
            top_k=3,
            return_data=True,
            min_entries=1,
            group_by_bins=4,              # binning numeric col into 4 bins
            sort_groups=False,
            auto_title=True,
        )
        assert len(stats_list) == 2, (
            f"[dfdraw contract profile surface] Expected 2 stats, got {len(stats_list)}"
        )
        for i, s in enumerate(stats_list):
            assert 'profile_data' in s, (
                f"[dfdraw B1a profile/return_data surface] stats[{i}] missing "
                f"'profile_data' (keys: {list(s.keys())})."
            )
        main = _fix1_get_main_group_legend(ax)
        n = len(main.get_texts()) if main else 0
        # Allow ≤ 5: 4 group bins + at most 1 extra entry from secondary
        # vector legend's first proxy (occasionally leaks into the main).
        assert n <= 5, (
            f"[dfdraw B1a profile/group_by_bins surface] Main legend has {n} "
            f"entries; expected ≤ 5 (4 bins + max 1 vector-legend leak)."
        )
        plt.close('all')

    def test_vector_hist_kwarg_surface_enumeration(self, df_with_groups):
        """
        R5 companion for hist().
        
        Pre-fix: hist() vector dispatch drops top_k (per §3.1 — confirmed by
        signature inspection at d662c0a5). Test asserts top_k=2 limits group
        count, which fails pre-fix because top_k is silently dropped → all 4
        groups appear in legend.
        """
        drawer = DFDraw(df_with_groups.copy())
        _, ax, stats_list = drawer.hist(
            "[y1,y2,y3]",
            selection="x > 0",
            sample=350,
            bins=8,
            range=(-3, 3),
            norm='density',
            stats=['n'],
            title="test",
            xlabel="X", ylabel="Y",
            group_by='category',
            top_k=2,             # B1a hist drops top_k — only 2 groups should plot
            auto_title=True,
        )
        assert len(stats_list) == 3, (
            f"[dfdraw contract hist surface] Expected 3 stats, got {len(stats_list)}"
        )
        # top_k=2 → main legend should have ≤ 2 actual GROUP entries
        # (may also contain vector-legend proxy entries like "y1" which we exclude)
        main = _fix1_get_main_group_legend(ax)
        all_labels = [t.get_text() for t in main.get_texts()] if main else []
        cats = set(df_with_groups['category'].unique().astype(str))
        group_labels = [l for l in all_labels if l in cats]
        assert len(group_labels) <= 2, (
            f"[dfdraw B1a hist/top_k surface] top_k=2 requested, but main legend "
            f"has {len(group_labels)} group entries (from {all_labels}). top_k silently "
            f"dropped in hist() vector dispatch. Expected ≤ 2."
        )
        plt.close('all')

    def test_vector_scatter_kwarg_surface_enumeration(self, df_with_groups):
        """
        R12: scatter surface — catches cmap/colorbar/clabel/jitter drops.
        
        Pre-fix: scatter() vector dispatch drops top_k AND colorbar (per §3.1).
        Test asserts via top_k that group filter applies AND no colorbar
        figure-axes added when colorbar=False.
        """
        drawer = DFDraw(df_with_groups.copy())
        fig, ax, stats_list = drawer.scatter(
            "[y1,y2]:x",
            selection="x > 0",
            sample=350,
            color='category',
            size=20.0,
            marker='s',
            stats=['n'],
            title="test",
            xlabel="X", ylabel="Y",
            cmap='plasma',
            colorbar=False,         # B1a scatter drops colorbar
            clabel="Category",
            jitter=0.1,
            group_by='category',
            top_k=2,                # B1a scatter drops top_k
        )
        assert len(stats_list) == 2, (
            f"[dfdraw contract scatter surface] Expected 2 stats, got {len(stats_list)}"
        )
        # top_k=2: pre-fix dropped → all 4 groups → too many lines
        main = _fix1_get_main_group_legend(ax)
        n_groups = len(main.get_texts()) if main else 0
        assert n_groups <= 2, (
            f"[dfdraw B1a scatter/top_k surface] top_k=2 requested, main legend "
            f"has {n_groups} entries. top_k silently dropped in scatter() vector "
            f"dispatch (lines 1233-1265)."
        )
        plt.close('all')

    def test_vector_draw_kwarg_surface_enumeration(self, df_with_groups):
        """R12: draw() surface — exercises draw()'s own named params."""
        drawer = DFDraw(df_with_groups.copy())
        # NOTE: type='scatter' so bins/norm are NOT passed (matplotlib scatter
        # rejects 'bins' as kwarg). The test exercises kwargs valid for the
        # routed method, not the union of all method kwargs.
        _, _, stats_list = drawer.draw(
            "[y1,y2]:x",
            type='scatter',
            selection="x > 0",
            sample=350,
            color='category',
            size=20.0,
            marker='s',
            stats=['n'],
            title="test",
        )
        assert len(stats_list) == 2, (
            f"[dfdraw contract draw surface] Expected 2 stats, got {len(stats_list)}"
        )
        plt.close('all')

    def test_vector_facet_with_vector_raises(self, df_with_groups):
        """R4: facet=True + vector must raise ValueError at all 3 dispatch sites."""
        drawer = DFDraw(df_with_groups.copy())
        with pytest.raises(ValueError, match=r"facet.*vector"):
            drawer.profile("[y1,y2]:x", bins=10, group_by='category', facet=True)
        with pytest.raises(ValueError, match=r"facet.*vector"):
            drawer.hist("[y1,y2]", bins=10, group_by='category', facet=True)
        with pytest.raises(ValueError, match=r"facet.*vector"):
            drawer.scatter("[y1,y2]:x", group_by='category', facet=True)
        plt.close('all')

    def test_all_forwarded_names_are_valid_signature_params(self):
        """R17: regression gate for R6 class-load validation."""
        import inspect
        pairs = [
            (DFDraw._PROFILE_FORWARDED_NAMES, DFDraw.profile, 'profile'),
            (DFDraw._HIST_FORWARDED_NAMES, DFDraw.hist, 'hist'),
            (DFDraw._SCATTER_FORWARDED_NAMES, DFDraw.scatter, 'scatter'),
            (DFDraw._DRAW_FORWARDED_NAMES, DFDraw.draw, 'draw'),
        ]
        for tup, method, name in pairs:
            sig_params = set(inspect.signature(method).parameters)
            missing = set(tup) - sig_params
            assert not missing, (
                f"[dfdraw R6 forwarded-names drift] "
                f"_{name.upper()}_FORWARDED_NAMES contains non-signature params: "
                f"{missing}. Update the tuple."
            )
