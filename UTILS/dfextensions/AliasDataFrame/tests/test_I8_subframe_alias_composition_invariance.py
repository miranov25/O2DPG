"""
Batch 2 — I8: Subframe + Alias Composition Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established in Batch 1)
----------------------------------------------------------------------
Per architect waiver of §5.1 (2026-04-10), Phase 13.12 uses standalone
files with "invariance" in the filename per §3.1 fallback rule.

All tests are marked @pytest.mark.invariance.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Each test names the production entry point. Composite-key tests use
pd.merge as an independent reference path (lesson from Batch 1 I6_1
P2 #3 — never use direct fixture-array slicing for reference values).
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_for_composition():
    """Small ADF with one subframe registered, no aliases yet."""
    df_main = pd.DataFrame({
        'sector': np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 5, dtype=np.int8),
        'pt': np.linspace(0.5, 5.0, 50).astype(np.float32),
        'col': np.linspace(1.0, 2.0, 50).astype(np.float32),
    })
    df_sub = pd.DataFrame({
        'sector': np.arange(5, dtype=np.int8),
        'gain': np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('T', AliasDataFrame(df_sub), index_columns='sector')
    return adf, df_main, df_sub


@pytest.fixture
def adf_composite_key():
    """ADF with a composite-key subframe (2 keys)."""
    df_main = pd.DataFrame({
        'k1': np.array([0, 1, 0, 1, 0, 1] * 5, dtype=np.int32),
        'k2': np.array([0, 0, 1, 1, 2, 2] * 5, dtype=np.int32),
        'pt': np.linspace(1.0, 2.0, 30).astype(np.float32),
    })
    keys = [(a, b) for a in [0, 1] for b in [0, 1, 2]]
    df_sub = pd.DataFrame({
        'k1': np.array([k[0] for k in keys], dtype=np.int32),
        'k2': np.array([k[1] for k in keys], dtype=np.int32),
        'val': np.linspace(10.0, 16.0, 6).astype(np.float32),
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('K', AliasDataFrame(df_sub),
                          index_columns=['k1', 'k2'])
    return adf, df_main, df_sub


class TestI8SubframeAliasCompositionInvariance:
    """Phase 13.12 I8 — Subframe + Alias Composition."""

    @pytest.mark.invariance
    def test_I8_1_inline_subframe_access_equals_alias_composition(
        self, adf_for_composition
    ):
        """
        I8_1 INVARIANT:
            Inline materialization via add_alias('x', 'T.gain * pt') and
            two-step (add_alias('g','T.gain'); add_alias('x','g*pt'))
            produce identical values.
        CODE PATH: add_alias + materialize_aliases (explicit names list).
        PRODUCTION ENTRY POINT: adf.materialize_aliases(names=['x'])
        REGRESSION GUARD: subframe access in compound expressions.
        """
        adf, _, _ = adf_for_composition

        # Path A: inline compound
        adf.add_alias('x_inline', 'T.gain * pt')
        adf.materialize_aliases(names=['x_inline'])
        a = adf.df['x_inline'].values.copy()

        # Path B: two-step
        adf.add_alias('g', 'T.gain')
        adf.add_alias('x_step', 'g * pt')
        adf.materialize_aliases(names=['x_step'])
        b = adf.df['x_step'].values.copy()

        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I8_1: inline vs two-step composition differ"
        )

    @pytest.mark.invariance
    def test_I8_2_composite_key_join_equals_pandas_merge(
        self, adf_composite_key
    ):
        """
        I8_2 INVARIANT:
            register_subframe(index_columns=['k1','k2']) followed by
            an alias K.val produces values equal to pd.merge on the
            same composite key.
        CODE PATH: composite-key registration, materialize_aliases.
        PRODUCTION ENTRY POINT: adf.materialize_aliases(names=['v'])
        REGRESSION GUARD: composite-key linearization correctness.
        """
        adf, df_main, df_sub = adf_composite_key

        adf.add_alias('v', 'K.val')
        adf.materialize_aliases(names=['v'])
        actual = adf.df['v'].values

        # Independent reference via pd.merge (NOT direct slicing)
        reference = df_main[['k1', 'k2']].merge(
            df_sub[['k1', 'k2', 'val']], on=['k1', 'k2'], how='left'
        )['val'].values

        np.testing.assert_allclose(
            actual, reference, rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I8_2: composite-key join differs from pd.merge reference"
        )

    @pytest.mark.invariance
    def test_I8_3_auto_aliased_subframe_column_equals_manual_alias(
        self, adf_for_composition
    ):
        """
        I8_3 INVARIANT:
            auto_alias_subframe('T') generates an alias for T.gain that
            produces the same values as a manually-added alias for T.gain.
        CODE PATH: auto_alias_subframe + materialize_aliases.
        PRODUCTION ENTRY POINT: adf.auto_alias_subframe('T')
        REGRESSION GUARD: auto-alias correctness.
        """
        adf, _, _ = adf_for_composition

        # Path A: manual alias
        adf.add_alias('manual_gain', 'T.gain')
        adf.materialize_aliases(names=['manual_gain'])
        manual = adf.df['manual_gain'].values.copy()

        # Path B: auto_alias_subframe
        # Per Claude33 P1 review + source line 9305: auto_alias_subframe
        # generates aliases with the plain column name (e.g. 'gain'), NOT
        # prefixed (no 'T_gain'). The return dict contains 'created' list
        # per source line 9194. Assert against the return dict directly
        # to distinguish "name mismatch" from "skipped due to conflict".
        result = adf.auto_alias_subframe('T', verbose=False)
        assert 'gain' in result.get('created', []), (
            f"I8_3: auto_alias_subframe('T') did not create alias 'gain'. "
            f"Return dict: {result}. "
            f"Available aliases: {list(adf.aliases.keys())}"
        )
        auto_name = 'gain'
        adf.materialize_aliases(names=[auto_name])
        auto = adf.df[auto_name].values

        np.testing.assert_allclose(
            manual, auto, rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I8_3: manual alias vs auto-aliased subframe column differ"
        )
