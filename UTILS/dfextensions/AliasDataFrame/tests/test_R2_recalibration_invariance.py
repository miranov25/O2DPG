"""
Phase 13.18.ADF — R2: Recalibration Invariance

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

Covers:
    R2_1: Subframe swap via update_regression_metadata produces
          evaluator returning v3 values, not v2 values. Fixtures use
          LARGE deterministic offset (v2=1.0, v3=1000.0) per P1-C
          to catch silent overwrite-ignored failures (Phase 13.12
          Batch 2 I8_3 silent-pass lesson).
    R2_2: Revert flow: swap back to v2 returns v2 values bit-exact.
    R2_3: Lazy subframe reference per A-1 option (b):
          update_regression_metadata(subframe_name='not_yet_registered')
          succeeds; register_evaluator_from_metadata then raises
          KeyError; register the missing subframe; retry succeeds.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


N_BINS = 3


def _make_subframe(coef_value):
    """Sparse-tolerant subframe with a single predictor * single target."""
    return pd.DataFrame({
        'sector': np.arange(N_BINS, dtype=np.int32),
        'dX_intercept_sw': np.full(N_BINS, coef_value, dtype=np.float64),
        'dX_slope_meanIDC_sw': np.zeros(N_BINS, dtype=np.float64),
    })


def _build_adf(coef_value_v2):
    df_main = pd.DataFrame({
        'sector': np.arange(30, dtype=np.int32) % N_BINS,
        'meanIDC': np.zeros(30, dtype=np.float64),  # zero -> no slope term
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('TPC_corr_v2',
                          AliasDataFrame(_make_subframe(coef_value_v2)),
                          index_columns='sector')
    adf.register_regression_metadata(
        'TPC_model',
        subframe_name='TPC_corr_v2',
        group_columns=['sector'],
        predictor_columns=['meanIDC'],
        targets=['dX'],
        suffix='_sw',
        fit_intercept=True,
    )
    return adf


class TestR2RecalibrationInvariance:

    @pytest.mark.invariance
    def test_R2_1_subframe_swap_produces_new_calibration_values(self):
        """
        R2_1 INVARIANT:
            Register metadata -> build evaluator -> get values (v2=1.0).
            update_regression_metadata(subframe_name='v3') + rebuild
            with overwrite=True -> get values (v3=1000.0).
            LARGE offset between v2 and v3 per P1-C prevents silent-
            pass if overwrite is ignored.
        """
        adf = _build_adf(coef_value_v2=1.0)
        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')
        adf.add_alias('dX_before', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_before'])
        before = adf.df['dX_before'].values.copy()

        # v2 values should be all 1.0 (meanIDC=0, so intercept only).
        np.testing.assert_array_equal(
            before, np.ones_like(before),
            err_msg="R2_1 precondition: v2 evaluator did not return 1.0"
        )

        # Register v3 subframe and swap the metadata reference.
        adf.register_subframe(
            'TPC_corr_v3',
            AliasDataFrame(_make_subframe(1000.0)),
            index_columns='sector',
        )
        adf.update_regression_metadata(
            'TPC_model', subframe_name='TPC_corr_v3'
        )
        adf.register_evaluator_from_metadata(
            'corr_dX', 'TPC_model', overwrite=True
        )
        adf.add_alias('dX_after', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_after'])
        after = adf.df['dX_after'].values

        np.testing.assert_array_equal(
            after, np.full_like(after, 1000.0),
            err_msg=(
                "R2_1: v3 swap did not take effect. Got "
                f"values {after[:3]} instead of 1000.0. Either "
                "update_regression_metadata silently ignored the swap, "
                "or register_evaluator_from_metadata(overwrite=True) "
                "rebuilt from v2 instead of v3."
            )
        )

    @pytest.mark.invariance
    def test_R2_2_revert_to_previous_subframe_bit_exact(self):
        """
        R2_2 INVARIANT:
            After a v2->v3 swap, reverting to v2 yields values
            BIT-EXACT equal to the original v2 values.
        """
        adf = _build_adf(coef_value_v2=1.0)
        adf.register_subframe(
            'TPC_corr_v3',
            AliasDataFrame(_make_subframe(1000.0)),
            index_columns='sector',
        )

        adf.register_evaluator_from_metadata('corr_dX', 'TPC_model')
        adf.add_alias('dX_orig', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_orig'])
        original = adf.df['dX_orig'].values.copy()

        # Swap forward.
        adf.update_regression_metadata(
            'TPC_model', subframe_name='TPC_corr_v3'
        )
        adf.register_evaluator_from_metadata(
            'corr_dX', 'TPC_model', overwrite=True
        )

        # Revert.
        adf.update_regression_metadata(
            'TPC_model', subframe_name='TPC_corr_v2'
        )
        adf.register_evaluator_from_metadata(
            'corr_dX', 'TPC_model', overwrite=True
        )
        adf.add_alias('dX_reverted', 'corr_dX(sector, meanIDC)')
        adf.materialize_aliases(names=['dX_reverted'])
        reverted = adf.df['dX_reverted'].values

        np.testing.assert_array_equal(
            reverted, original,
            err_msg=(
                "R2_2: revert to v2 did not yield bit-exact original "
                "values. Either from_dfGB is nondeterministic or the "
                "remap path differs between first build and rebuild."
            )
        )

    @pytest.mark.invariance
    def test_R2_3_lazy_subframe_reference_validated_at_register(self):
        """
        R2_3 INVARIANT (A-1 option b):
            update_regression_metadata(subframe_name='not_registered')
            succeeds; subsequent register_evaluator_from_metadata
            raises KeyError; after registering the subframe,
            register_evaluator_from_metadata succeeds.
        """
        adf = _build_adf(coef_value_v2=1.0)

        # Step 1: swap to a subframe that does NOT exist. Must succeed.
        result = adf.update_regression_metadata(
            'TPC_model', subframe_name='TPC_corr_future'
        )
        assert result['subframe_name'] == 'TPC_corr_future', (
            "R2_3: update_regression_metadata did not record pending "
            "subframe reference"
        )

        # describe_regression should show pending status.
        desc = adf.describe_regression(as_dict=True)
        assert desc['TPC_model']['subframe_status'] == 'pending', (
            "R2_3: describe_regression should show subframe_status='pending' "
            f"for unregistered subframe. Got {desc['TPC_model']['subframe_status']}"
        )

        # Step 2: register_evaluator_from_metadata must raise.
        with pytest.raises(KeyError):
            adf.register_evaluator_from_metadata(
                'corr_dX', 'TPC_model', overwrite=True
            )

        # Step 3: register the subframe, retry, must succeed.
        adf.register_subframe(
            'TPC_corr_future',
            AliasDataFrame(_make_subframe(7.0)),
            index_columns='sector',
        )
        summary = adf.register_evaluator_from_metadata(
            'corr_dX', 'TPC_model', overwrite=True
        )
        assert summary['subframe_name'] == 'TPC_corr_future'

        # describe_regression now shows registered.
        desc2 = adf.describe_regression(as_dict=True)
        assert desc2['TPC_model']['subframe_status'] == 'registered', (
            "R2_3: describe_regression should show subframe_status="
            "'registered' after late subframe registration"
        )
