"""
Phase 13.18.ADF — R5: Natural-Label Remap Correctness Invariance

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

Covers:
    R5_1: Subframe with natural-label NON-CONTIGUOUS index values
          (e.g., sector in {2, 5, 9}) produces evaluator results
          equal to an independent pd.merge-based reference. Uses
          LARGE deterministic coefficient values to prevent silent-pass
          if the remap silently drops to contiguous indices.

Independent reference path via pd.merge per Phase 13.12 Batch 1
I6_1 P2 #3 lesson (direct fixture-array slicing is fragile).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestR5RemapCorrectnessInvariance:

    @pytest.mark.invariance
    def test_R5_1_noncontiguous_natural_labels_match_pd_merge(self):
        """
        R5_1 INVARIANT:
            Subframe uses natural labels {2, 5, 9} (non-contiguous).
            Main DataFrame queries all of those labels.
            Evaluator result == pd.merge(main, sub, on='sector',
            how='left')['dX_intercept_sw'].values.
        REGRESSION GUARD:
            If from_dfGB silently reduces non-contiguous labels to
            contiguous integers via argsort, the query result would
            drift from the pd.merge reference. This test catches that.
        """
        natural_labels = np.array([2, 5, 9], dtype=np.int32)
        # LARGE distinct values per P1-C style.
        coef_intercept = np.array([200.0, 500.0, 900.0], dtype=np.float64)

        df_sub = pd.DataFrame({
            'sector': natural_labels,
            'dX_intercept_sw': coef_intercept,
            'dX_slope_meanIDC_sw': np.zeros(3, dtype=np.float64),
        })

        # Main DataFrame: queries natural labels, repeated.
        df_main = pd.DataFrame({
            'sector': np.tile(natural_labels, 5).astype(np.int32),
            'meanIDC': np.zeros(15, dtype=np.float64),
        })

        adf = AliasDataFrame(df_main)
        adf.register_subframe('TPC_corr',
                              AliasDataFrame(df_sub),
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
        adf.add_alias('dX_pred', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_pred'])
        evaluator_result = adf.df['dX_pred'].values

        # Independent pd.merge reference. Because meanIDC=0, the
        # evaluator result equals the intercept per sector.
        reference = df_main[['sector']].merge(
            df_sub[['sector', 'dX_intercept_sw']],
            on='sector', how='left'
        )['dX_intercept_sw'].values

        np.testing.assert_allclose(
            evaluator_result, reference,
            rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg=(
                "R5_1: evaluator result diverges from pd.merge "
                "reference on non-contiguous natural-label indices. "
                "Either from_dfGB collapses natural labels to "
                "contiguous indices silently, or the remap path is "
                "wrong, or the alias evaluation path passes raw "
                "natural labels without remapping."
            )
        )
