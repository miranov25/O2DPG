"""
Phase 13.18.ADF — R5: Natural-Label Remap Correctness Invariance

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

Currently **xfail(strict=True)**. See
BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md.

Same root cause as R3 family: method='lookup' uses raw grid indices
rather than remapping through bin_centers. For non-contiguous natural
labels {2,5,9}, the raw-index semantic collapses them to 0-based grid
positions [0,1,2], so:
    - query sector=2 returns coefs[2]=900 (expected coefs[0]=200)
    - query sector=5 is grid-index 5 >= grid_shape=3 → NaN (expected
      coefs[1]=500)
    - query sector=9 is grid-index 9 >= grid_shape=3 → NaN (expected
      coefs[2]=900)
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
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics: "
            "method='lookup' does not remap natural labels via bin_centers."
        ),
    )
    def test_R5_1_noncontiguous_natural_labels_match_pd_merge(self):
        """
        XFAIL: Non-contiguous natural labels produce wrong results.

        Status: 🧨 Broken (correctness violation on non-contiguous labels)
        Limitation ID: GB_EVALUATOR_SAFETY
        Bug Report: BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md
        Resolution: Phase 13.19.ADF-GB (or later).

        Evidence (2026-04-13):
            natural_labels=[2,5,9], coefs=[200,500,900]
            Query [2,5,9,...] returns [900, NaN, NaN, 900, NaN, NaN, ...]
            Expected:             [200, 500, 900, 200, 500, 900, ...]
            Root cause: method='lookup' indexes grid[sector_value]:
              - sector=2 → grid[2]=coefs[2]=900
              - sector=5 → grid[5] out-of-range → NaN
              - sector=9 → grid[9] out-of-range → NaN

        Workaround (application-level):
            Pre-remap natural labels to contiguous 0..N-1:
                remap = {2:0, 5:1, 9:2}
                df['sector_idx'] = df['sector'].map(remap)
                # Then query with 'sector_idx'
            Alternatively, use pd.merge directly for lookup-table use.

        When to remove xfail:
            After GB adds 'strict_lookup' method with bin_centers
            remapping, OR after bridge does pre-remap itself, OR after
            proposal scope change.

        R5_1 INVARIANT (target, currently violated):
            Evaluator result == pd.merge(main, sub, on='sector',
            how='left')['dX_intercept_sw'].values.
        """
        natural_labels = np.array([2, 5, 9], dtype=np.int32)
        coef_intercept = np.array([200.0, 500.0, 900.0], dtype=np.float64)

        df_sub = pd.DataFrame({
            'sector': natural_labels,
            'dX_intercept_sw': coef_intercept,
            'dX_slope_meanIDC_sw': np.zeros(3, dtype=np.float64),
        })
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

        reference = df_main[['sector']].merge(
            df_sub[['sector', 'dX_intercept_sw']],
            on='sector', how='left'
        )['dX_intercept_sw'].values

        np.testing.assert_allclose(
            evaluator_result, reference,
            rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg=(
                "R5_1: evaluator result diverges from pd.merge "
                "reference on non-contiguous natural-label indices."
            )
        )
