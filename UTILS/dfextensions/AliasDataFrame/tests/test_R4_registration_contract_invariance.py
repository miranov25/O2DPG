"""
Phase 13.18.ADF — R4: Registration Contract Invariance (Fail-Fast)

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

Per v1.1 §3.4: contracts enforced at registration time, raising
ValueError for structural violations. Derived from Safety Hard
Constraint (no silent wrong results).

Covers:
    R4_1: Missing REQUIRED coefficient column (e.g., 'dX_intercept_sw')
          raises ValueError at register_evaluator_from_metadata.
          Missing OPTIONAL column (e.g., 'dX_rmse_sw') does not raise.
    R4_2: group_columns length/order mismatch raises ValueError.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_minimal_adf():
    df_main = pd.DataFrame({
        'sector': np.arange(9, dtype=np.int32) % 3,
        'meanIDC': np.zeros(9, dtype=np.float64),
    })
    adf = AliasDataFrame(df_main)
    return adf


class TestR4RegistrationContractInvariance:

    @pytest.mark.invariance
    def test_R4_1_missing_required_coefficient_column_raises(self):
        """
        R4_1 FAIL-FAST INVARIANT:
            A subframe missing the required 'dX_intercept_sw' column
            (while fit_intercept=True) must trigger ValueError at
            register_evaluator_from_metadata time.
            A subframe missing an OPTIONAL column such as
            'dX_rmse_sw' must NOT raise.
        """
        adf = _build_minimal_adf()

        # Subframe WITHOUT intercept column — required, must fail.
        df_sub_bad = pd.DataFrame({
            'sector': np.arange(3, dtype=np.int32),
            'dX_slope_meanIDC_sw': np.zeros(3),
        })
        adf.register_subframe('bad_sub',
                              AliasDataFrame(df_sub_bad),
                              index_columns='sector')
        adf.register_regression_metadata(
            'model_bad',
            subframe_name='bad_sub',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
        )
        with pytest.raises(ValueError, match='dX_intercept_sw'):
            adf.register_evaluator_from_metadata('corr_bad', 'model_bad')

        # Subframe WITH required, WITHOUT optional — must succeed.
        df_sub_ok = pd.DataFrame({
            'sector': np.arange(3, dtype=np.int32),
            'dX_intercept_sw': np.array([1.0, 2.0, 3.0]),
            'dX_slope_meanIDC_sw': np.zeros(3),
            # No 'dX_rmse_sw' — optional, not required.
        })
        adf.register_subframe('ok_sub',
                              AliasDataFrame(df_sub_ok),
                              index_columns='sector')
        adf.register_regression_metadata(
            'model_ok',
            subframe_name='ok_sub',
            group_columns=['sector'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
        )
        # Must not raise.
        adf.register_evaluator_from_metadata('corr_ok', 'model_ok')

    @pytest.mark.invariance
    def test_R4_2_group_columns_mismatch_raises(self):
        """
        R4_2 FAIL-FAST INVARIANT:
            group_columns declared in metadata must match columns
            actually present in the subframe. A metadata entry naming
            a nonexistent index column must raise at registration.
        """
        adf = _build_minimal_adf()
        df_sub = pd.DataFrame({
            'sector': np.arange(3, dtype=np.int32),
            'dX_intercept_sw': np.array([1.0, 2.0, 3.0]),
            'dX_slope_meanIDC_sw': np.zeros(3),
        })
        adf.register_subframe('sub_has_sector',
                              AliasDataFrame(df_sub),
                              index_columns='sector')
        # Metadata claims 'wrong_col' which is not in the subframe.
        adf.register_regression_metadata(
            'model_mismatch',
            subframe_name='sub_has_sector',
            group_columns=['wrong_col'],
            predictor_columns=['meanIDC'],
            targets=['dX'],
            suffix='_sw',
            fit_intercept=True,
        )
        with pytest.raises(ValueError, match='wrong_col'):
            adf.register_evaluator_from_metadata(
                'corr_mismatch', 'model_mismatch'
            )
