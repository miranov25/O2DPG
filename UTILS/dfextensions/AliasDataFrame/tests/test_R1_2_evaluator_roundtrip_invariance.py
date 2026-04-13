"""
Phase 13.18.ADF — R1_2: Evaluator Roundtrip Bit-Exact Invariance

STANDALONE NEW TEST FILE. Own file per v1.1 P1-B (Capability Matrix
unambiguity — this test verifies the BRIDGE, not metadata, and must
map to FUNC.regression_persistence, not FUNC.regression_metadata).

Marked @pytest.mark.invariance.

Per v1.1 §4.3 / §5.1 R1_2:
    After export_tree -> read_tree, calling
    register_evaluator_from_metadata on the restored metadata MUST
    produce a BIT-EXACT equivalent evaluator — same query positions
    return BYTE-IDENTICAL per-target values.

Uses assert_array_equal (not allclose) per P1-D: same code path,
same data, deterministic. Rtol reserved for Phase 13.19
cross-language Python<->C++ tests.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestR1_2EvaluatorRoundtripInvariance:

    @pytest.mark.invariance
    def test_R1_2_evaluator_equivalence_after_roundtrip(self, tmp_path):
        """
        R1_2 INVARIANT:
            Build evaluator, record values at query positions;
            export_tree + read_tree; re-register evaluator from
            restored metadata; query at SAME positions; values must
            be BIT-EXACT equal (assert_array_equal, not allclose).
        PRODUCTION ENTRY POINT:
            adf.register_regression_metadata(...)
            adf.register_evaluator_from_metadata('corr', 'TPC_model')
            eval_before = adf.df[...]  # via alias
            adf.export_tree(path)
            adf2 = AliasDataFrame.read_tree(path)
            adf2.register_evaluator_from_metadata('corr', 'TPC_model')
            eval_after = adf2.df[...]
            assert_array_equal(eval_before, eval_after)
        REGRESSION GUARD:
            Catches any divergence in from_dfGB, in bin_centers
            inference, in valid_mask construction, in register_evaluator
            wrapping, or in the schema persistence path.
        """
        n_bins = 5
        df_main = pd.DataFrame({
            'sector': np.arange(50, dtype=np.int32) % n_bins,
            'meanIDC': np.linspace(0.1, 2.0, 50).astype(np.float32),
        })
        # Deterministic coefficient values.
        df_sub = pd.DataFrame({
            'sector': np.arange(n_bins, dtype=np.int32),
            'dX_intercept_sw': np.array(
                [10.0, 20.0, 30.0, 40.0, 50.0]
            ),
            'dX_slope_meanIDC_sw': np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ),
        })

        adf = AliasDataFrame(df_main)
        adf.register_subframe('TPC_corr', AliasDataFrame(df_sub),
                              index_columns='sector')
        adf.register_regression_metadata(
            'TPC_model',
            subframe_name='TPC_corr',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
        )
        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')

        # Query before export. Build an alias that uses the evaluator.
        adf.add_alias('dX_pred_before', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_pred_before'])
        values_before = adf.df['dX_pred_before'].values.copy()

        path = str(tmp_path / 'r1_2.root')
        adf.export_tree(path, treename='tree')

        adf2 = AliasDataFrame.read_tree(path, treename='tree')
        adf2.register_evaluator_from_metadata('corr_dX', 'TPC_model',
                                              overwrite=True)
        adf2.add_alias('dX_pred_after', 'corr_dX(sector, meanIDC)')
        adf2.materialize_aliases(names=['dX_pred_after'])
        values_after = adf2.df['dX_pred_after'].values

        # Bit-exact per P1-D.
        np.testing.assert_array_equal(
            values_before, values_after,
            err_msg=(
                "R1_2: evaluator values differ after schema roundtrip "
                "(not bit-exact). Either from_dfGB is nondeterministic, "
                "or bin_centers inference drifted, or a remap path "
                "changed between build and rebuild."
            )
        )
