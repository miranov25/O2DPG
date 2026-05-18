"""
Tests S10–S19 for BUG_AliasDataFrame_20260518_draw_subframe_alias_not_materialized.

Phase A fix coverage:
  - Site 1: AliasDataFrame.draw() guard at line 11036
  - Site 2: AliasDataFrame.draw_batch() guard at line 12060
  - Site 3: AliasDataFrame.draw_figures() guard at line 12360
  - Site 4: AliasDataFrame._scatter_subframe_column guard at line 3108
  - Cleanups 1-4: except: pass -> warnings.warn in draw_batch + draw_figures

All tests follow these disciplines:
  - Failure Mode #12 entry-point rule: every test calls the exact public API
    the user hit (adf.draw, adf.draw_batch, adf.draw_figures).
  - Cold-draw discipline: NO test pre-accesses adf.SubFrame.alias_col before
    the first adf.draw(...) call. Pre-access trivially "fixes" the bug as a
    side effect.
  - Fixture verification guards: each fixture asserts the precondition
    (alias defined AND alias NOT yet materialized) before yielding the ADF.
    Without these guards, a fixture-bug masquerades as a passing test.
  - Bug-fix-test entry-point rule: every test exercises the public API the
    original reproducer hit, and would fail without the fix.

Drafter: Claude36 (Coder)
Routing: non-Claude36 reviewer required per session COI declaration.
"""
import warnings

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures with mandatory precondition guards
# =============================================================================

@pytest.fixture
def adf_with_alias_subframe():
    """ADF with a subframe whose target column is an ALIAS (not raw column).

    Mirrors the compressADF decompression-alias production pattern.
    """
    rng = np.random.default_rng(42)
    n = 100
    main_df = pd.DataFrame({
        "group_id": np.repeat(np.arange(5), n // 5),
        "y_val":    rng.normal(0, 1, n).astype(np.float32),
    })
    sub_raw = pd.DataFrame({
        "group_id": np.arange(5),
        "coeff_a":  np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    })
    sub_adf = AliasDataFrame(sub_raw)
    sub_adf.add_alias("y_baseline", "coeff_a * 2.0")  # ALIAS, not raw column

    # ━━ Mandatory fixture verification guards ━━
    assert "y_baseline" not in sub_adf.df.columns, (
        "Fixture broken: y_baseline must NOT be in sub_adf.df.columns"
    )
    assert "y_baseline" in sub_adf.aliases, (
        "Fixture broken: y_baseline must be in sub_adf.aliases"
    )

    adf = AliasDataFrame(main_df)
    adf.draw_lazy = True
    adf.register_subframe("Calib", sub_adf, index_columns="group_id")

    # Post-register guard: registration must not have triggered materialization
    assert "y_baseline" not in adf.get_subframe("Calib").df.columns, (
        "Fixture broken: register_subframe should not eagerly materialize the alias"
    )
    return adf


@pytest.fixture
def adf_with_multilevel_alias_subframe():
    """ADF with NESTED subframe (A.B.alias_col pattern) — production case from compressADF.

    Outer subframe 'Outer' itself contains a subframe 'Inner', and the target
    column 'decoded' is an alias on 'Inner'. Exercises _scatter_subframe_column
    (Site 4) which the single-level resolver patches do not cover.
    """
    rng = np.random.default_rng(42)
    main_df = pd.DataFrame({
        "group_id": np.repeat(np.arange(5), 20),
        "y_val":    rng.normal(0, 1, 100).astype(np.float32),
    })
    inner_raw = pd.DataFrame({
        "group_id_inner": np.arange(5),
        "raw_c":          np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    })
    inner_adf = AliasDataFrame(inner_raw)
    inner_adf.add_alias("decoded", "raw_c * 0.5")  # ALIAS on inner

    # Inner-fixture precondition guards
    assert "decoded" not in inner_adf.df.columns
    assert "decoded" in inner_adf.aliases

    outer_raw = pd.DataFrame({
        "group_id":       np.arange(5),
        "group_id_inner": np.arange(5),
    })
    outer_adf = AliasDataFrame(outer_raw)
    outer_adf.register_subframe("Inner", inner_adf, index_columns="group_id_inner")

    adf = AliasDataFrame(main_df)
    adf.draw_lazy = True
    adf.register_subframe("Outer", outer_adf, index_columns="group_id")

    # Post-register: still must not be materialized on inner
    assert "decoded" not in adf.get_subframe("Outer").get_subframe("Inner").df.columns, (
        "Fixture broken: nested registration eagerly materialized 'decoded'"
    )
    return adf


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.invariance
class TestS10_DrawSubframeAliasNotMaterialized:
    """Phase A bug-fix tests for BUG_AliasDataFrame_20260518.

    The bug: draw() and friends silently skipped subframe references when the
    target column was an alias (in sf.aliases) rather than a physical column
    (in sf.df.columns). Symptom: UndefinedVariableError on the dotted ref.
    Fix: materialize the alias on the subframe before the existing guard.
    """

    # -------------------------------------------------------------------------
    # Site 1 — draw() single-level resolver
    # -------------------------------------------------------------------------

    def test_S10_alias_in_xy_expression(self, adf_with_alias_subframe):
        """draw('y:Sub.alias') resolves alias in X position (scatter)."""
        adf = adf_with_alias_subframe
        # Cold draw — NO adf.Calib.y_baseline pre-access.
        result = adf.draw("y_val:Calib.y_baseline")
        # Tuple unpacking — DFDraw returns (fig, ax, stats)
        assert result is not None, "draw() returned None"
        fig, ax, stats = result
        assert stats is not None, "stats dict is None"
        assert stats.get("n", 0) > 0, f"expected n>0, got stats={stats}"
        plt.close("all")

    def test_S11_alias_as_y_in_expression(self, adf_with_alias_subframe):
        """draw('Sub.alias:y') resolves alias in Y position."""
        adf = adf_with_alias_subframe
        result = adf.draw("Calib.y_baseline:y_val")
        assert result is not None
        fig, ax, stats = result
        assert stats.get("n", 0) > 0
        plt.close("all")

    def test_S12_alias_as_sole_expression_histogram(self, adf_with_alias_subframe):
        """draw('Sub.alias') as sole expression resolves (histogram branch)."""
        adf = adf_with_alias_subframe
        result = adf.draw("Calib.y_baseline")
        assert result is not None
        fig, ax, stats = result
        assert stats.get("n", 0) > 0
        plt.close("all")

    def test_S13_alias_in_arithmetic_expression(self, adf_with_alias_subframe):
        """draw() resolves alias inside an arithmetic expression."""
        adf = adf_with_alias_subframe
        result = adf.draw("y_val - Calib.y_baseline")
        assert result is not None
        fig, ax, stats = result
        assert stats.get("n", 0) > 0
        plt.close("all")

    def test_S14_alias_in_selection(self, adf_with_alias_subframe):
        """draw() with subframe alias inside selection string.

        selection is folded into all_text at line 10991 before regex tokenization,
        so this exercises the same Site 1 patch.
        """
        adf = adf_with_alias_subframe
        result = adf.draw("y_val", selection="Calib.y_baseline > 0")
        assert result is not None
        fig, ax, stats = result
        # The selection should filter — n_filtered depends on data, just verify
        # the call completed without UndefinedVariableError.
        assert stats is not None
        plt.close("all")

    def test_S15_raw_column_still_works_regression_guard(self, adf_with_alias_subframe):
        """Regression guard: raw column in subframe (not alias) still resolves.

        Locks against the Phase A fix accidentally breaking the existing S6-S9
        code path. coeff_a is a raw column in sub_adf.df.columns, not an alias.
        """
        adf = adf_with_alias_subframe
        result = adf.draw("y_val:Calib.coeff_a")
        assert result is not None
        fig, ax, stats = result
        assert stats.get("n", 0) > 0
        plt.close("all")

    # -------------------------------------------------------------------------
    # Site 2 — draw_batch()
    # -------------------------------------------------------------------------

    def test_S16_draw_batch_with_alias_subframe(self, adf_with_alias_subframe):
        """draw_batch() resolves subframe alias in batch spec."""
        adf = adf_with_alias_subframe
        # draw_batch takes a dict of {name: spec} where spec has 'expr'
        specs = {
            "p1": {"expr": "y_val:Calib.y_baseline"},
        }
        # Should not raise UndefinedVariableError
        try:
            adf.draw_batch(specs)
        except (NameError, ValueError, KeyError) as e:
            if "Calib" in str(e) and "not defined" in str(e):
                pytest.fail(f"BUG_20260518 reproduces in draw_batch: {e}")
            raise
        plt.close("all")

    # -------------------------------------------------------------------------
    # Site 3 — draw_figures()
    # -------------------------------------------------------------------------

    def test_S17_draw_figures_with_alias_subframe(self, adf_with_alias_subframe):
        """draw_figures() resolves subframe alias in figures spec."""
        adf = adf_with_alias_subframe
        # draw_figures takes a list of figure-specs, each with 'plots' key
        # containing a list of plot dicts with 'expr' key
        specs = [
            {"plots": [{"expr": "y_val:Calib.y_baseline"}]},
        ]
        try:
            adf.draw_figures(specs)
        except (NameError, ValueError, KeyError) as e:
            if "Calib" in str(e) and "not defined" in str(e):
                pytest.fail(f"BUG_20260518 reproduces in draw_figures: {e}")
            raise
        plt.close("all")

    # -------------------------------------------------------------------------
    # Site 4 — _scatter_subframe_column (multi-level path)
    # -------------------------------------------------------------------------

    def test_S18_multilevel_alias_on_inner_subframe(self, adf_with_multilevel_alias_subframe):
        """draw('y:Outer.Inner.decoded') where 'decoded' is alias on Inner.

        Exercises Site 4 (_scatter_subframe_column) via the multi-level chain
        walked by _prepare_subframe_joins. The single-level fix at Site 1 is
        not enough — without the Site 4 fix this still fails.
        """
        adf = adf_with_multilevel_alias_subframe
        result = adf.draw("y_val:Outer.Inner.decoded")
        assert result is not None
        fig, ax, stats = result
        assert stats.get("n", 0) > 0
        plt.close("all")

    # -------------------------------------------------------------------------
    # Cold-draw discipline lock
    # -------------------------------------------------------------------------

    def test_S19_cold_draw_no_workaround_used(self, adf_with_alias_subframe):
        """Explicit lock: alias must NOT be materialized before draw is called.

        Verifies the fixture preserves the cold-draw state and that the public
        API path through draw() is what does the materialization. Without this
        test, future fixture changes could silently introduce a workaround
        (pre-access) and S10-S15 would pass trivially.
        """
        adf = adf_with_alias_subframe

        # Precondition: cold state
        sf = adf.get_subframe("Calib")
        assert "y_baseline" not in sf.df.columns, (
            "Cold-draw precondition broken: y_baseline materialized before draw"
        )
        assert "y_baseline" in sf.aliases

        # Now draw — the fix should materialize as a side effect on sf.df
        result = adf.draw("y_val:Calib.y_baseline")
        assert result is not None

        # Post-condition: the materialization happened via the draw path
        sf_after = adf.get_subframe("Calib")
        assert "y_baseline" in sf_after.df.columns, (
            "After draw, y_baseline should be materialized on the subframe "
            "(this is the side effect of the Phase A fix; Phase B may change "
            "this to non-mutating evaluation)"
        )
        plt.close("all")
