"""
Phase 13.18.ADF — R3: Missing-Bin Safety Invariance

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

**Hard Constraint: Safety.** Per v0.3 §4.4 and v1.1 §3.4.
Missing bins and out-of-range positions MUST return NaN — NEVER zero,
NEVER neighbor values, NEVER edge-clamped values (under bounds='nan').

All three R3 tests are currently **xfail(strict=True)**. See
BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md.

ROOT CAUSE (summary):
    v1.1 §3.1 default_method='lookup' was interpreted to mean
    "natural-label table lookup" but GB's method='lookup' has a
    different semantic: positions are used as RAW 0-based grid
    indices, not remapped through bin_centers. None of GB's current
    methods satisfy the Safety Hard Constraint for interior missing
    bins:
        - method='lookup' — raw indexing, no safety semantics
        - method='nearest_fast'/'nearest' — snaps to nearest populated
          bin, returns value (not NaN)
        - method='multilinear' + invalid_strategy='nan' — would return
          NaN for missing interior, but interpolates between bins,
          which is not the intended lookup-table semantic

RESOLUTION OPTIONS (pending architect decision):
    Option 1: GB adds 'strict_lookup' method (natural-label match via
              bin_centers; NaN on any no-match). Cleanest.
    Option 2: ADF bridge does pre-remap itself. Duplicates GB logic.
    Option 3: Relax v1.1 §3.4 Safety — "missing bins MAY return
              nearest-valid value; caller pre-filters".
              Architect decision required.

strict=True so accidental pass (silent semantic drift) fails the test
and forces investigation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_sparse_adf(populated_sectors, coef_values):
    df_sub = pd.DataFrame({
        'sector': np.array(populated_sectors, dtype=np.int32),
        'dX_intercept_sw': np.array(coef_values, dtype=np.float64),
        'dX_slope_meanIDC_sw': np.zeros(len(populated_sectors),
                                         dtype=np.float64),
    })
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
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics: "
            "method='lookup' uses raw grid indices, no bin_centers remap. "
            "Pending Option 1/2/3 resolution."
        ),
    )
    def test_R3_1_unpopulated_bin_returns_nan_not_silent_value(self):
        """
        XFAIL: Interior unpopulated bin returns wrong value instead of NaN.

        Status: 🧨 Broken (Safety Hard Constraint violation)
        Limitation ID: GB_EVALUATOR_SAFETY
        Bug Report: BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md
        Resolution: Phase 13.19.ADF-GB (or later) after architect decision.

        Evidence (2026-04-13 test run):
            populated=[0,2,4,6,8], coefs=[10,20,30,40,50]
            Query sector=2 returns 30.0 (expected: 20.0)
            Root cause: method='lookup' uses 2 as raw grid index,
            retrieves coefs[2]=30. bin_centers=[0,2,4,6,8] is built
            but never consulted in _eval_lookup (line 1186 of
            groupby_regression_evaluator.py).

        Workaround (application-level):
            mask = df['sector'].isin(populated_sectors)
            result = np.full(len(df), np.nan)
            result[mask] = adf.df['dX_pred'].values[mask]

        When to remove xfail:
            After GB adds 'strict_lookup' method (Option 1), OR after
            bridge pre-remap (Option 2), OR after architect approves
            scope change (Option 3).

        R3_1 SAFETY HARD CONSTRAINT (target, currently violated):
            Subframe populated for sectors [0,2,4,6,8]. Query at
            sectors [1,3,5,7,9] must return NaN. No silent values.
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

        for sec, expected in zip(populated, coefs):
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert result[row_idx] == expected, (
                f"R3_1: populated sector {sec} returned "
                f"{result[row_idx]} instead of {expected}"
            )
        for sec in [1, 3, 5, 7, 9]:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_1 SAFETY VIOLATION: unpopulated sector {sec} "
                f"returned {result[row_idx]!r} instead of NaN."
            )

    @pytest.mark.invariance
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics: "
            "interior-missing under bounds='clamp' — same root cause as R3_1."
        ),
    )
    def test_R3_2_bounds_clamp_still_nans_interior_missing_bins(self):
        """
        XFAIL: Interior missing under bounds='clamp' returns neighbor
               value instead of NaN.

        Status: 🧨 Broken (Safety Hard Constraint violation)
        Limitation ID: GB_EVALUATOR_SAFETY
        Bug Report: BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md
        Resolution: Phase 13.19.ADF-GB (or later).

        Evidence (2026-04-13):
            populated=[0,2,4,6,8], coefs=[10,20,30,40,50], bounds='clamp'
            Query sector=1 returns 20.0 (expected: NaN)
            Root cause: same as R3_1 — lookup uses raw index;
            bounds='clamp' only affects out-of-range not interior.

        Workaround: same as R3_1.
        When to remove xfail: same as R3_1.

        R3_2 INVARIANT (target, currently violated):
            Under bounds='clamp', out-of-range clamps to edge,
            but interior unpopulated bins still return NaN.
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

        for sec in [1, 3, 5, 7]:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_2 SAFETY VIOLATION: interior sector {sec} "
                f"returned {result[row_idx]!r} instead of NaN."
            )

    @pytest.mark.invariance
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics: "
            "out-of-range under bounds='nan' with method='lookup' — "
            "raw-index semantic sidesteps bin_centers bounds check."
        ),
    )
    def test_R3_3_bounds_nan_out_of_range_returns_nan(self):
        """
        XFAIL: Out-of-range under bounds='nan' returns clamped-edge
               value instead of NaN.

        Status: 🧨 Broken (Safety Hard Constraint violation)
        Limitation ID: GB_EVALUATOR_SAFETY
        Bug Report: BUG_AliasDataFrame_20260413_gb_evaluator_safety_semantics.md
        Resolution: Phase 13.19.ADF-GB (or later).

        Evidence (2026-04-13):
            populated=[2,3,4,5], coefs=[20,30,40,50], bounds='nan'
            Query sector=0 returns 20.0 (expected: NaN)
            Root cause: with method='lookup' and grid_shape=(4,), raw
            index 0 is inside [0, grid_shape-1]=[0,3], so _eval_lookup's
            bounds='nan' check (raw<0 or raw>=grid_shape) does NOT
            trigger. The raw-index semantic sidesteps bin_centers-based
            bounds detection entirely.

        Workaround: same as R3_1.
        When to remove xfail: same as R3_1.

        R3_3 INVARIANT (target, currently violated):
            Under bounds='nan', positions outside the grid range must
            return NaN.
        """
        populated = [2, 3, 4, 5]
        coefs = [20.0, 30.0, 40.0, 50.0]
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

        for sec in [0, 1, 6, 7, 8, 9]:
            row_idx = np.where(adf.df['sector'].values == sec)[0][0]
            assert np.isnan(result[row_idx]), (
                f"R3_3 SAFETY VIOLATION: out-of-range sector {sec} "
                f"returned {result[row_idx]!r}."
            )
