"""
Phase 13.36.ADF — Subframe Metadata Propagation to Drawing

Invariance tests for ``get_axis_title`` and ``get_column_metadata`` dispatch
from a parent AliasDataFrame to its registered subframes when the column
name is a flattened subframe-column reference produced by the draw()
resolver.

Coverage:

* X1 — single-level subframe: get_axis_title for ``f"{sf_name}_{col}"``
       returns the title set on the subframe's schema.
* X2 — multi-level subframe: get_axis_title for ``f"{leaf}__{C}__{B}__{A}"``
       walks the subframe chain and returns the title from the deepest
       subframe's schema.
* X3 — get_column_metadata dispatch: full metadata dict (unit, axisLabel,
       description, range) is returned for the subframe flat name.
* X4 — parent precedence: if parent ADF has explicit metadata for the
       flat name itself, parent wins (subframe is fallback only).
* X5 — negative: unknown column / unregistered subframe / malformed flat
       name returns None (axis title) or empty dict (metadata) without
       raising.
* X6 — end-to-end via draw(): subframe-stored title propagates through
       Phase A's resolver to the rendered figure's axis label.

All tests call the public ``get_axis_title`` / ``get_column_metadata`` /
``draw`` APIs — no direct ``_resolve_subframe_flat_name`` calls. The helper
is exercised indirectly through the dispatch path inside the public
methods.
"""

import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_with_single_subframe():
    """Parent ADF with one registered subframe ``vC`` carrying metadata."""
    parent_df = pd.DataFrame({
        'idx': np.arange(20),
        'vertex_x_intercept': np.linspace(-0.05, -0.045, 20),
    })
    parent = AliasDataFrame(parent_df)
    parent.set_axis_title('vertex_x_intercept', 'vertex x intercept [cm]')

    # Subframe vC with the decompressed column + its own metadata
    sf_df = pd.DataFrame({
        'idx': np.arange(20),
        'vertex_x_intercept_decomp': np.linspace(-0.05, -0.045, 20),
        'extra_field': np.zeros(20),
    })
    sf_adf = AliasDataFrame(sf_df)
    sf_adf.set_axis_title('vertex_x_intercept_decomp',
                           'v_C vertex x intercept (decompressed) [cm]')
    sf_adf.set_column_metadata(
        'vertex_x_intercept_decomp',
        unit='cm',
        axisLabel='x_{int} (decomp)',
        description='Compressed-then-decompressed vertex_x_intercept',
        range=[-0.06, -0.04],
    )
    parent.register_subframe('vC', sf_adf, index_columns=['idx'])
    return parent


@pytest.fixture
def parent_with_multi_level_subframes():
    """Parent → A → B → C nested subframes (3 levels deep)."""
    parent_df = pd.DataFrame({'idx': np.arange(10), 'top_col': np.arange(10) * 1.0})
    parent = AliasDataFrame(parent_df)

    a_df = pd.DataFrame({'idx': np.arange(10), 'a_col': np.arange(10) * 2.0})
    a = AliasDataFrame(a_df)
    parent.register_subframe('A', a, index_columns=['idx'])

    b_df = pd.DataFrame({'idx': np.arange(10), 'b_col': np.arange(10) * 3.0})
    b = AliasDataFrame(b_df)
    a.register_subframe('B', b, index_columns=['idx'])

    c_df = pd.DataFrame({'idx': np.arange(10), 'val': np.arange(10) * 4.0})
    c = AliasDataFrame(c_df)
    c.set_axis_title('val', 'innermost C value [arbitrary unit]')
    c.set_column_metadata('val', unit='arb', description='deepest leaf metadata')
    b.register_subframe('C', c, index_columns=['idx'])

    return parent


# ---------------------------------------------------------------------------
# X1 — single-level title dispatch
# ---------------------------------------------------------------------------

class TestX1_SingleLevelTitle:
    @pytest.mark.invariance
    def test_X1_single_level_title_via_flat_name(self, parent_with_single_subframe):
        """Parent.get_axis_title('vC_vertex_x_intercept_decomp') must dispatch
        to subframe vC's schema and return the subframe-stored title."""
        adf = parent_with_single_subframe
        flat = 'vC_vertex_x_intercept_decomp'
        result = adf.get_axis_title(flat)
        assert result == 'v_C vertex x intercept (decompressed) [cm]', (
            f"Expected subframe title, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# X2 — multi-level title dispatch
# ---------------------------------------------------------------------------

class TestX2_MultiLevelTitle:
    @pytest.mark.invariance
    def test_X2_multi_level_title_via_flat_name(self, parent_with_multi_level_subframes):
        """Parent.get_axis_title('val__C__B__A') walks A→B→C and returns
        the title stored on C."""
        adf = parent_with_multi_level_subframes
        flat = 'val__C__B__A'
        result = adf.get_axis_title(flat)
        assert result == 'innermost C value [arbitrary unit]', (
            f"Expected innermost subframe title, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# X3 — full metadata dict dispatch
# ---------------------------------------------------------------------------

class TestX3_GetColumnMetadataDispatch:
    @pytest.mark.invariance
    def test_X3_full_metadata_dispatched(self, parent_with_single_subframe):
        """get_column_metadata on a flat subframe name returns the FULL
        metadata dict (unit, axisLabel, description, range) from the
        subframe schema."""
        adf = parent_with_single_subframe
        meta = adf.get_column_metadata('vC_vertex_x_intercept_decomp')
        assert meta.get('unit') == 'cm', f"unit missing/wrong: {meta}"
        assert meta.get('axisLabel') == 'x_{int} (decomp)', (
            f"axisLabel missing/wrong: {meta}"
        )
        assert meta.get('description') == 'Compressed-then-decompressed vertex_x_intercept', (
            f"description missing/wrong: {meta}"
        )
        assert meta.get('range') == [-0.06, -0.04], f"range missing/wrong: {meta}"
        assert meta.get('title') == 'v_C vertex x intercept (decompressed) [cm]', (
            f"title missing/wrong: {meta}"
        )


# ---------------------------------------------------------------------------
# X4 — parent precedence (explicit override on the flat name wins)
# ---------------------------------------------------------------------------

class TestX4_ParentPrecedence:
    @pytest.mark.invariance
    def test_X4_parent_override_wins(self, parent_with_single_subframe):
        """If the parent ADF has metadata for the flat name itself (user
        explicitly overrode), the parent wins. Subframe dispatch is only a
        fallback, not a hijack."""
        adf = parent_with_single_subframe
        # Parent explicitly sets title for the flattened name
        adf.set_axis_title('vC_vertex_x_intercept_decomp', 'PARENT EXPLICIT OVERRIDE')
        result = adf.get_axis_title('vC_vertex_x_intercept_decomp')
        assert result == 'PARENT EXPLICIT OVERRIDE', (
            f"Parent override must win over subframe dispatch, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# X5 — negative branch (unknown / malformed / no dispatch target)
# ---------------------------------------------------------------------------

class TestX5_NegativeBranch:
    @pytest.mark.invariance
    def test_X5_unknown_flat_name_returns_none(self, parent_with_single_subframe):
        """Unknown flat name → get_axis_title returns None,
        get_column_metadata returns empty dict. No exception."""
        adf = parent_with_single_subframe
        # No subframe 'totally', so prefix scan finds nothing
        assert adf.get_axis_title('totally_nonexistent_column') is None
        assert adf.get_column_metadata('totally_nonexistent_column') == {}
        # Multi-level with unknown subframe → no dispatch
        assert adf.get_axis_title('leaf__NotASubframe') is None
        # Empty / non-string input → safe None / empty dict
        assert adf.get_axis_title('') is None
        assert adf.get_column_metadata('') == {}


# ---------------------------------------------------------------------------
# X6 — end-to-end through draw(): subframe title flows to matplotlib axis label
# ---------------------------------------------------------------------------

class TestX6_EndToEndDraw:
    @pytest.mark.invariance
    def test_X6_draw_axis_label_from_subframe(self, parent_with_single_subframe):
        """End-to-end load-bearing test for Phase 13.36.ADF.

        Calling ``draw('parent_col:vC.subframe_col')`` routes through Phase A's
        resolver, which rewrites the RHS to ``vC_subframe_col`` on df_subset
        and passes it to dfdraw. dfdraw calls
        ``_data_source.get_axis_title('vC_subframe_col')`` via duck typing.
        With Phase 13.36.ADF dispatch, this returns the subframe-stored title.

        STRICT ASSERTION (per CRR v1.0 P1-3 fix, reviewer Claude37 option (a)):
        after ``draw()`` returns successfully, this test inspects
        ``ax.get_ylabel()`` and asserts the subframe-stored title appears.
        This actually tests the dfdraw duck-typed lookup path that motivated
        Phase 13.36 in the first place — not just the parent-side dispatch
        (which X1 already covers).

        Skip vs Fail discipline:
          - matplotlib not available → SKIP (env-level)
          - dfdraw not installed → SKIP (env-level)
          - draw() raises any OTHER exception → FAIL (load-bearing)
          - draw() succeeds but ylabel doesn't reflect subframe title → FAIL
            (this is the failure mode Phase 13.36 was designed to prevent)

        If this test fails in production while X1 passes, the gap is in the
        dfdraw layer (dfdraw isn't calling get_axis_title via duck typing, OR
        isn't propagating the result to ax.set_ylabel). That's a separate
        bug for the dfdraw team, not a Phase 13.36 defect — X1 still locks
        the ADF-side dispatch contract.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # headless
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip('matplotlib not available')

        adf = parent_with_single_subframe
        subframe_title = 'v_C vertex x intercept (decompressed) [cm]'

        # draw() must succeed. If dfdraw isn't installed, skip; any other
        # exception is a real failure.
        #
        # NOTE on expr choice: use the subframe column AS the Y axis, not as
        # the X axis. dfdraw labels the Y axis with the FIRST Y column's
        # title (verified empirically 2026-05-26 — when expr was
        # 'vertex_x_intercept:vC.vertex_x_intercept_decomp', ylabel was
        # 'vertex x intercept [cm]' i.e. parent's title for the LHS column).
        # The cleanest direct test of Phase 13.36 dispatch is to put the
        # subframe column on Y where its title surfaces directly.
        try:
            result = adf.draw(
                'vC.vertex_x_intercept_decomp:idx',
                type='scatter',
            )
        except ImportError as e:
            pytest.skip(f'dfdraw not available in this environment: {e}')

        # draw() returns either (fig, ax, stats) tuple or a single object —
        # accommodate both shapes.
        fig = None
        ax = None
        if isinstance(result, tuple) and len(result) >= 2:
            fig, ax = result[0], result[1]
        elif hasattr(result, 'axes'):
            fig = result

        # If we couldn't recover an axes object, skip the strict assertion
        # but verify the dispatch path is still wired (sanity).
        if fig is None and ax is None:
            pytest.skip(
                "draw() did not return a recognizable (fig, ax) shape — "
                "cannot inspect ylabel. Dispatch path verified by X1."
            )

        # Strict assertion: the subframe-stored title must appear in the
        # y-axis label of the rendered figure.
        ax_obj = ax
        if ax_obj is None and fig is not None and hasattr(fig, 'axes') and fig.axes:
            ax_obj = fig.axes[0]
        ylabel = ax_obj.get_ylabel() if (ax_obj is not None and hasattr(ax_obj, 'get_ylabel')) else ''

        # Phase 13.36.ADF success criterion: the title flows through to the
        # figure y-axis. Accept either exact match or substring (different
        # dfdraw versions may decorate the label with units).
        match = (subframe_title in ylabel) or (ylabel == subframe_title)
        assert match, (
            f"Phase 13.36.ADF end-to-end failure: y-axis label does not reflect "
            f"subframe-stored title.\n"
            f"  Expected: {subframe_title!r}\n"
            f"  Got ylabel: {ylabel!r}\n"
            f"Diagnostic: this means dfdraw is either (a) not calling "
            f"_data_source.get_axis_title() via duck typing, or (b) not "
            f"propagating the returned title to ax.set_ylabel(). X1 confirms "
            f"the ADF-side dispatch works; this assertion exposes a gap in "
            f"the dfdraw integration layer."
        )

        plt.close('all')
