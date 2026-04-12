"""
Batch 4 — I16: Expression Evaluator Dtype Preservation Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE. All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.4)
---------------------
Verify that the expression evaluator respects explicit dtype via
add_alias(dtype=...) and that numeric operations on float32 inputs
do not silently produce float64 outputs when dtype is pinned.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestI16DtypePreservationInvariance:

    @pytest.mark.invariance
    def test_I16_1_explicit_dtype_float32_pinned(self):
        """
        I16_1: add_alias(..., dtype=np.float32) results in a float32
        column after materialize, even when numpy default promotion
        would produce float64.
        """
        df = pd.DataFrame({
            'x': np.linspace(0.0, 1.0, 50).astype(np.float32),
        })
        adf = AliasDataFrame(df)
        # 2.0 is a Python float (float64) — without explicit dtype,
        # x*2.0 would promote to float64. With dtype pinned, result
        # must be float32.
        adf.add_alias('y', 'x * 2.0', dtype=np.float32)
        adf.materialize_aliases(names=['y'])
        assert adf.df['y'].dtype == np.float32, (
            f"I16_1: dtype not pinned. Got {adf.df['y'].dtype}, "
            f"expected float32"
        )
        np.testing.assert_allclose(
            adf.df['y'].values, (df['x'].values * 2.0).astype(np.float32),
            rtol=1e-6, atol=1e-9,
            err_msg="I16_1: dtype-pinned values differ from float32 reference"
        )

    @pytest.mark.invariance
    def test_I16_2_default_dtype_is_not_reduced_below_input(self):
        """
        I16_2: Without explicit dtype, expression on float32 inputs
        does not produce a dtype NARROWER than float32 (i.e., no
        silent downcast to float16).
        """
        df = pd.DataFrame({
            'x': np.linspace(0.0, 1.0, 50).astype(np.float32),
            'y': np.linspace(1.0, 2.0, 50).astype(np.float32),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('z', 'x + y')  # no explicit dtype
        adf.materialize_aliases(names=['z'])
        result_dtype = adf.df['z'].dtype
        # Result must be at least as wide as float32 (float32 or float64)
        assert result_dtype in (np.float32, np.float64), (
            f"I16_2: unexpected output dtype {result_dtype} "
            f"(should be float32 or float64, never narrower)"
        )
        # Values must match numpy reference regardless of chosen width
        np.testing.assert_allclose(
            adf.df['z'].values.astype(np.float64),
            (df['x'].values + df['y'].values).astype(np.float64),
            rtol=1e-6, atol=1e-9,
            err_msg="I16_2: values differ from numpy reference"
        )
