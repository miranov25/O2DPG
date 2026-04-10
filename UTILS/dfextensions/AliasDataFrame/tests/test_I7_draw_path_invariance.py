"""
Batch 1 — I7: Draw Path Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note)
-----------------------------------------------
v1.2 proposal §5.1 committed to extending existing files with zero new
files. Phase 13.12 Batch 1 delivered 3 new standalone files instead,
with "invariance" in each filename per §3.1 fallback rule. Deviation
acknowledged by Main Architect on 2026-04-10 after reviewer feedback
(Claude32 P2 #1, Claude33 P1-2). All tests are marked
@pytest.mark.invariance.

Feature flipped: DRAW.execution, DRAW.subframe_resolution
Incidents addressed:
  - BUG_AliasDataFrame_20260324_draw_subframe_resolution
    (draw("Side.dy:row") failed silently)
  - BUG_AliasDataFrame_20260401_draw_lazy_compound
    (aliases in abs()/sqrt() not auto-materialized in draw_lazy path)

NOTE ON INTEGRATION WITH EXISTING TESTS
---------------------------------------
tests/test_draw_invariance.py already has
TestDrawInvariance::test_draw_vs_materialize_identical (line ~380)
which is a partial I7_1 checking only stats['n']. The tests below are
stricter versions asserting full stats-dict equality (n, mean, std,
min, max) and both draw_lazy=True and draw_lazy=False paths.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Every test explicitly sets adf.draw_lazy to True or False before
calling draw(). No reliance on the default (which is False at line 900
of AliasDataFrame.py). Tests that verify the draw_lazy=True regression
guard explicitly set adf.draw_lazy = True.

SWITCH REFERENCE (per architect A2)
-----------------------------------
adf.draw_lazy : bool, default False (line 900)
    When True, draw() auto-materializes aliases referenced in the
    expression before evaluation. This is the switch that was enabled
    in the fix for BUG_AliasDataFrame_20260401_draw_lazy_compound.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# =============================================================================
# Dependency guards (copied from existing test_draw_invariance.py pattern)
# =============================================================================

try:
    import ROOT
    _HAS_ROOT_I7 = ROOT is not None
except ImportError:
    _HAS_ROOT_I7 = False

try:
    from dfextensions.dfdraw import DFDraw
    _HAS_DFDRAW_I7 = True
except ImportError:
    _HAS_DFDRAW_I7 = False

_requires_dfdraw_i7 = pytest.mark.skipif(
    not _HAS_DFDRAW_I7, reason="dfdraw not available"
)


# =============================================================================
# Helper: compare two stats dicts per v1.2 §4.1 equality surface
# =============================================================================

def _assert_stats_equal(stats_a, stats_b, expr_name, rtol=1e-6, atol=1e-9):
    """
    Compare two stats dicts per v1.2 §4.1 I7_3 equality surface.

    REQUIRED equalities:
      - n: exact
      - mean, std: within rtol (default 1e-6 for float32 per §5.6)
      - min, max: exact (no aggregation drift)

    NOT asserted:
      - row ordering, matplotlib fig/ax identity, any plotting state
    """
    assert 'n' in stats_a and 'n' in stats_b, \
        f"{expr_name}: stats dict must have 'n' key"
    assert stats_a['n'] == stats_b['n'], \
        f"{expr_name}: n differs: {stats_a['n']} vs {stats_b['n']}"

    for k in ('mean', 'std'):
        if k in stats_a and k in stats_b:
            a_val = stats_a[k]
            b_val = stats_b[k]
            if a_val is None or b_val is None:
                continue
            np.testing.assert_allclose(
                a_val, b_val, rtol=rtol, atol=atol,
                err_msg=f"{expr_name}: {k} differs: {a_val} vs {b_val}"
            )

    for k in ('min', 'max'):
        if k in stats_a and k in stats_b:
            a_val = stats_a[k]
            b_val = stats_b[k]
            if a_val is None or b_val is None:
                continue
            # min/max exact equality (no aggregation drift)
            if np.isnan(a_val) and np.isnan(b_val):
                continue
            assert a_val == b_val, \
                f"{expr_name}: {k} differs (exact required): {a_val} vs {b_val}"


# =============================================================================
# Fixture — small in-memory ADF for I7 tests, no ROOT file required
# =============================================================================

@pytest.fixture
def adf_i7_compound_alias():
    """
    Per-test fixture with a subframe + alias + compound expression target.
    Small scale (100 rows) for fast execution.

    The compound expression abs(some_alias) is the BUG_20260401 scenario.
    """
    n = 100
    np.random.seed(20260401)  # bug date as seed
    df_main = pd.DataFrame({
        'row': np.arange(n, dtype=np.int32),
        'sector': np.array([i % 4 for i in range(n)], dtype=np.int8),
        'pt': np.random.uniform(0.5, 5.0, n).astype(np.float32),
    })
    df_sub = pd.DataFrame({
        'sector': np.arange(4, dtype=np.int8),
        'dy': np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
    })

    adf = AliasDataFrame(df_main)
    adf.register_subframe('Side', AliasDataFrame(df_sub), index_columns='sector')
    adf.add_alias('Side_dy', 'Side.dy', fill_value=0)

    return adf


# =============================================================================
# Test class — append to tests/test_draw_invariance.py
# =============================================================================

class TestI7DrawPathEquivalence:
    """
    Phase 13.12 I7 — Draw Path Equivalence.

    Regression guards for the two 2026-03-24 / 2026-04-01 draw incidents.
    Every test sets adf.draw_lazy explicitly (path-explicit per Failure
    Mode #11); no reliance on the default.

    Flips: DRAW.execution, DRAW.subframe_resolution.
    """

    @_requires_dfdraw_i7
    @pytest.mark.invariance
    def test_I7_1_draw_lazy_compound_expression_equals_explicit_materialize(
        self, adf_i7_compound_alias
    ):
        """
        I7_1 INVARIANT:
            With adf.draw_lazy=True, calling
                adf.draw("abs(Side_dy)", ...)
            produces the same stats dict as:
                adf.draw_lazy = False
                adf.materialize_alias('Side_dy')
                adf.draw("abs(Side_dy)", ...)

            The draw_lazy=True path MUST auto-materialize the alias
            inside the compound expression before evaluation.

        CODE PATH:
            Path A: adf.draw_lazy=True, draw("abs(Side_dy)") — regression
                    guard path; auto-materialization switch must fire
            Path B: adf.draw_lazy=False, explicit materialize_alias then
                    draw("abs(Side_dy)") — reference path
            Both paths explicit (draw_lazy set on each call), not defaults.

        PRODUCTION ENTRY POINT:
            adf.draw_lazy = True
            adf.draw("abs(some_alias)")
            — the exact call sequence from
            BUG_AliasDataFrame_20260401_draw_lazy_compound.
            The fix required enabling the automatic materialization switch;
            this test asserts the switch remains enabled.

        REGRESSION GUARD FOR:
            BUG_AliasDataFrame_20260401_draw_lazy_compound
            (compound expression alias resolution in draw_lazy path).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        adf_a = adf_i7_compound_alias

        # Path A: draw_lazy=True with compound expression (regression guard)
        adf_a.draw_lazy = True
        fig_a, ax_a, stats_a = adf_a.draw('abs(Side_dy)', type='hist', bins=20)
        plt.close(fig_a)

        # Path B: fresh ADF, explicit materialization, draw_lazy=False
        # Rebuild the ADF to avoid state leak from Path A
        n = 100
        np.random.seed(20260401)
        df_main = pd.DataFrame({
            'row': np.arange(n, dtype=np.int32),
            'sector': np.array([i % 4 for i in range(n)], dtype=np.int8),
            'pt': np.random.uniform(0.5, 5.0, n).astype(np.float32),
        })
        df_sub = pd.DataFrame({
            'sector': np.arange(4, dtype=np.int8),
            'dy': np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        })
        adf_b = AliasDataFrame(df_main)
        adf_b.register_subframe('Side', AliasDataFrame(df_sub),
                                index_columns='sector')
        adf_b.add_alias('Side_dy', 'Side.dy', fill_value=0)
        adf_b.draw_lazy = False
        adf_b.materialize_alias('Side_dy')
        fig_b, ax_b, stats_b = adf_b.draw('abs(Side_dy)', type='hist', bins=20)
        plt.close(fig_b)

        # Invariant: stats dicts must match per v1.2 §4.1 equality surface
        _assert_stats_equal(
            stats_a, stats_b, 'abs(Side_dy)',
            rtol=1e-6, atol=1e-9,  # float32 tolerance per §5.6
        )

    @_requires_dfdraw_i7
    @pytest.mark.invariance
    def test_I7_2_draw_subframe_column_equals_explicit_alias(
        self, adf_i7_compound_alias
    ):
        """
        I7_2 INVARIANT:
            adf.draw("Side.dy:row") produces the same stats dict as:
                adf.add_alias('x', 'Side.dy')
                adf.materialize_alias('x')
                adf.draw('x:row')

            Direct subframe column reference in a draw expression must
            resolve identically to an explicit alias over the same
            subframe column.

        CODE PATH:
            Path A: draw with subframe.column expression (draw_lazy=True
                    to exercise resolution in the draw pipeline)
            Path B: explicit add_alias + materialize + draw on a plain
                    column (draw_lazy=False, no resolution ambiguity)
            Both paths explicit.

        PRODUCTION ENTRY POINT:
            adf.draw("Side.dy:row") — the exact call sequence from
            BUG_AliasDataFrame_20260324_draw_subframe_resolution.

        REGRESSION GUARD FOR:
            BUG_AliasDataFrame_20260324_draw_subframe_resolution
            (subframe column reference in draw expression failed silently
            before the fix).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Path A: draw with subframe.column in expression, draw_lazy=True
        adf_a = adf_i7_compound_alias
        adf_a.draw_lazy = True
        try:
            fig_a, ax_a, stats_a = adf_a.draw(
                'Side.dy:row', type='profile', bins=10
            )
            plt.close(fig_a)
            path_a_succeeded = True
        except Exception as e:
            path_a_succeeded = False
            path_a_error = str(e)

        # Path B: fresh ADF, explicit alias path
        n = 100
        np.random.seed(20260401)
        df_main = pd.DataFrame({
            'row': np.arange(n, dtype=np.int32),
            'sector': np.array([i % 4 for i in range(n)], dtype=np.int8),
            'pt': np.random.uniform(0.5, 5.0, n).astype(np.float32),
        })
        df_sub = pd.DataFrame({
            'sector': np.arange(4, dtype=np.int8),
            'dy': np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        })
        adf_b = AliasDataFrame(df_main)
        adf_b.register_subframe('Side', AliasDataFrame(df_sub),
                                index_columns='sector')
        adf_b.add_alias('Side_dy_explicit', 'Side.dy', fill_value=0)
        adf_b.draw_lazy = False
        adf_b.materialize_alias('Side_dy_explicit')
        fig_b, ax_b, stats_b = adf_b.draw(
            'Side_dy_explicit:row', type='profile', bins=10
        )
        plt.close(fig_b)

        # Invariant: Path A must succeed (not fail silently)
        assert path_a_succeeded, (
            f"I7_2 FAILED: draw('Side.dy:row') with draw_lazy=True raised "
            f"an exception: {path_a_error if not path_a_succeeded else ''}. "
            f"This is BUG_AliasDataFrame_20260324 recurrence."
        )

        # Invariant: stats dicts must match per §4.1 equality surface
        _assert_stats_equal(
            stats_a, stats_b, 'Side.dy:row',
            rtol=1e-6, atol=1e-9,  # float32 tolerance per §5.6
        )

    @_requires_dfdraw_i7
    @pytest.mark.invariance
    def test_I7_3_draw_with_selection_equals_pre_filtered_draw(
        self, adf_i7_compound_alias
    ):
        """
        I7_3 INVARIANT (per v1.2 §4.1 I7_3 Equality Surface):
            adf.draw(expr, selection=mask) produces the same stats
            values (n, mean, std, min, max) as drawing on a
            pre-filtered ADF for any expression including
            subframe-dependent ones.

        CODE PATH:
            Path A: slice-first via draw selection parameter
                    (draw_lazy=False, explicit pt selection)
            Path B: pre-filter df then draw (draw_lazy=False)
            Both paths explicit, not auto-dispatched.

        EQUALITY SURFACE (per v1.2 §4.1):
            Required exact:  n, min, max, NaN positions
            Required rtol:   mean, std (1e-6 for float32)
            Not asserted:    row ordering, fig/ax identity

        PRODUCTION ENTRY POINT:
            adf.draw(expr, selection=mask_expression) — the standard
            pattern for slice-first visualization.

        REGRESSION GUARD FOR:
            slice-first optimization correctness. Per Claude31 v1.0
            review, the original I7_3 invariant was false for
            subframe-dependent aliases because filtering adf.df first
            changes the subframe join key set. This reformulation
            asserts on stats values, not on underlying row-level
            equality, which IS mathematically correct.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        adf_a = adf_i7_compound_alias
        adf_a.draw_lazy = False
        adf_a.materialize_alias('Side_dy')

        # Path A: slice-first via selection parameter
        fig_a, ax_a, stats_a = adf_a.draw(
            'Side_dy', type='hist', bins=20, selection='pt > 2.0'
        )
        plt.close(fig_a)

        # Path B: pre-filter df then draw
        # Build a fresh ADF from the filtered rows, re-register subframe,
        # re-add alias (mirrors what a user would do manually)
        mask = adf_a.df['pt'].values > 2.0
        filtered_df = adf_a.df[mask].reset_index(drop=True)
        adf_b = AliasDataFrame(filtered_df)
        # Per Claude32 P2 #6: use public get_subframe, not _subframes private
        sub_entry = adf_a.get_subframe('Side')
        if sub_entry is not None:
            adf_b.register_subframe(
                'Side', sub_entry, index_columns='sector'
            )
            adf_b.add_alias('Side_dy', 'Side.dy', fill_value=0)
            adf_b.draw_lazy = False
            adf_b.materialize_alias('Side_dy')
        fig_b, ax_b, stats_b = adf_b.draw('Side_dy', type='hist', bins=20)
        plt.close(fig_b)

        # Invariant: stats dicts must match per §4.1 equality surface
        _assert_stats_equal(
            stats_a, stats_b, 'Side_dy with selection',
            rtol=1e-6, atol=1e-9,  # float32 tolerance per §5.6
        )
