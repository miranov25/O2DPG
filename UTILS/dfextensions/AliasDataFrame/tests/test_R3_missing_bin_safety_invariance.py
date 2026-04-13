"""
Phase 13.18.ADF — R3: Missing-Bin Safety Invariance

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

**Hard Constraint: Safety.** Per v0.3 §4.4 and v1.1 §3.4.
Missing bins and out-of-range positions MUST return NaN — NEVER zero,
NEVER neighbor values, NEVER edge-clamped values (under bounds='nan').

Covers:
    R3_1: Sparse subframe, query at unpopulated INTERIOR bin returns NaN.
    R3_2: Under default_bounds='clamp', out-of-range clamps but
          interior unpopulated bins still return NaN.
    R3_3: Under default_bounds='nan', out-of-range returns NaN
          (A-3 closure — derived from v0.3 §4.4 Safety Hard Constraint).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_sparse_adf(populated_sectors, coef_values):
    """
    Subframe with ONLY the specified sectors populated.
    populated_sectors: list of int
    coef_values: list of float, same length
    """
    df_sub = pd.DataFrame({
        'sector': np.array(populated_sectors, dtype=np.int32),
        'dX_intercept_sw': np.array(coef_values, dtype=np.float64),
        'dX_slope_meanIDC_sw': np.zeros(len(populated_sectors),
                                         dtype=np.float64),
    })
    # Main queries all sectors 0..9
    df_main = pd.DataFrame({
        'sector': np.arange(10, dtype=np.int32),
        'meanIDC': np.zeros(10, dtype=np.float64),
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('TPC_corr',
                          AliasDataFrame(df_sub),
                          index_columns='sector')
    return adf


class TestR3MissingBinSafetyInvariance:

    @pytest.mark.invariance
    def test_R3_1_unpopulated_bin_returns_nan_not_silent_value(self):
        """
        R3_1 SAFETY HARD CONSTRAINT:
            Subframe populated for sectors [0, 2, 4, 6, 8]. Query at
            sectors [1, 3, 5, 7, 9] must return NaN.
            Must NOT return: zero (silent), neighbor value, clamped
            edge value.
        REGRESSION GUARD:
            BUG_20260331-analogue — silent propagation of wrong value
            through dependent alias expressions is the same class of
            failure as the fill_value-dependency incident.
        """
        populated = [0, 2, 4, 6, 8]
        coefs = [10.0, 20.0, 30.0, 40.0, 50.0]
        adf = _build_sparse_adf(populated, coefs)
        adf.register_regression_metadata(
            'TPC_model',
            subframe_name='TPC_corr',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
            default_bounds='nan',
        )
        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')
        adf.add_alias('dX_pred', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_pred'])
        result = adf.df['dX_pred'].values

        # Populated sectors [0,2,4,6,8] must match the coefs.
        for sec, expected in zip(populated, coefs):
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert result[row_idx] == expected, (
                f"R3_1: populated sector {sec} returned "
                f"{result[row_idx]} instead of {expected}"
            )

        # Unpopulated sectors [1,3,5,7,9] must be NaN.
        unpopulated = [1, 3, 5, 7, 9]
        for sec in unpopulated:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_1 SAFETY VIOLATION: unpopulated sector {sec} "
                f"returned {result[row_idx]!r} instead of NaN. "
                f"This is exactly the failure mode Phase 13.12 I6 "
                f"and BUG_20260331 were built to catch."
            )

    @pytest.mark.invariance
    def test_R3_2_bounds_clamp_still_nans_interior_missing_bins(self):
        """
        R3_2 INVARIANT:
            Under default_bounds='clamp', out-of-range query positions
            clamp to the edge, but interior unpopulated bins still
            return NaN. Distinguishes 'out of range' from 'missing in
            range'.
        """
        populated = [0, 2, 4, 6, 8]
        coefs = [10.0, 20.0, 30.0, 40.0, 50.0]
        adf = _build_sparse_adf(populated, coefs)
        adf.register_regression_metadata(
            'TPC_model',
            subframe_name='TPC_corr',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
            default_bounds='clamp',
        )
        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')
        adf.add_alias('dX_pred', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_pred'])
        result = adf.df['dX_pred'].values

        # Interior unpopulated sectors [1, 3, 5, 7] — all still NaN
        # (interior means within range of populated values 0..8, but
        # not present in the populated list).
        for sec in [1, 3, 5, 7]:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_2 SAFETY VIOLATION: under bounds='clamp', "
                f"interior sector {sec} (between populated 0 and 8) "
                f"returned {result[row_idx]!r} instead of NaN. "
                f"Clamp must only affect OUT-OF-RANGE positions, "
                f"not interior unpopulated bins."
            )

    @pytest.mark.invariance
    def test_R3_3_bounds_nan_out_of_range_returns_nan(self):
        """
        R3_3 INVARIANT (A-3 closure):
            Under default_bounds='nan' (the ADF bridge default), query
            positions outside the grid range must return NaN. This is
            the out-of-range variant of §4.4 Safety — distinct from
            R3_1's interior-missing case.
        PROOF OF SAFETY HARD CONSTRAINT CLOSURE:
            v0.3 §4.4 says missing bins return NaN. v1.1 §3.4 promoted
            A-3 to derive the same result for out-of-range. R3_3 is
            the test that verifies the derivation holds.
        """
        populated = [2, 3, 4, 5]  # range: 2..5
        coefs = [20.0, 30.0, 40.0, 50.0]
        adf = _build_sparse_adf(populated, coefs)
        # Main tree has sectors 0..9, so 0,1 are below-range and 6..9
        # are above-range.
        adf.register_regression_metadata(
            'TPC_model',
            subframe_name='TPC_corr',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
            default_bounds='nan',
        )
        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')
        adf.add_alias('dX_pred', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_pred'])
        result = adf.df['dX_pred'].values

        # Below-range [0, 1] and above-range [6,7,8,9] all NaN.
        out_of_range_sectors = [0, 1, 6, 7, 8, 9]
        for sec in out_of_range_sectors:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_3 SAFETY VIOLATION: out-of-range sector {sec} "
                f"(grid is [2..5]) returned {result[row_idx]!r} "
                f"under bounds='nan'. Must be NaN."
            )
