"""
Batch 4 — I15: Materialization Order Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE. All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.4)
---------------------
Verify that the order in which multiple independent aliases are
materialized does not change their final values.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_multi_alias():
    df = pd.DataFrame({
        'a': np.linspace(0.0, 1.0, 100).astype(np.float32),
        'b': np.linspace(1.0, 2.0, 100).astype(np.float32),
        'c': np.linspace(2.0, 3.0, 100).astype(np.float32),
    })
    return AliasDataFrame(df)


class TestI15MaterializationOrderInvariance:

    @pytest.mark.invariance
    def test_I15_1_batch_equals_sequential_independent_aliases(
        self, adf_multi_alias
    ):
        """
        I15_1: Batch materialize_aliases(['X','Y','Z']) == sequential
        materialize_alias('X'); materialize_alias('Y'); ... for
        independent aliases (no shared dependencies).
        """
        adf_batch = adf_multi_alias
        adf_batch.add_alias('X', 'a * 2.0')
        adf_batch.add_alias('Y', 'b + 1.0')
        adf_batch.add_alias('Z', 'c - 0.5')
        adf_batch.materialize_aliases(names=['X', 'Y', 'Z'])

        # Sequential ADF
        df_seq = adf_batch.df[['a', 'b', 'c']].copy()
        adf_seq = AliasDataFrame(df_seq)
        adf_seq.add_alias('X', 'a * 2.0')
        adf_seq.add_alias('Y', 'b + 1.0')
        adf_seq.add_alias('Z', 'c - 0.5')
        adf_seq.materialize_alias('X')
        adf_seq.materialize_alias('Y')
        adf_seq.materialize_alias('Z')

        for col in ['X', 'Y', 'Z']:
            np.testing.assert_allclose(
                adf_batch.df[col].values, adf_seq.df[col].values,
                rtol=1e-6, atol=1e-9, equal_nan=True,
                err_msg=f"I15_1: batch vs sequential differ for {col}"
            )

    @pytest.mark.invariance
    def test_I15_2_reversed_order_equals_forward_order(
        self, adf_multi_alias
    ):
        """
        I15_2: materialize_aliases(['X','Y','Z']) ==
               materialize_aliases(['Z','Y','X']) for independent aliases.
        """
        adf_a = adf_multi_alias
        adf_a.add_alias('X', 'a + b')
        adf_a.add_alias('Y', 'b + c')
        adf_a.add_alias('Z', 'a + c')
        adf_a.materialize_aliases(names=['X', 'Y', 'Z'])

        df_b = adf_a.df[['a', 'b', 'c']].copy()
        adf_b = AliasDataFrame(df_b)
        adf_b.add_alias('X', 'a + b')
        adf_b.add_alias('Y', 'b + c')
        adf_b.add_alias('Z', 'a + c')
        adf_b.materialize_aliases(names=['Z', 'Y', 'X'])

        for col in ['X', 'Y', 'Z']:
            np.testing.assert_allclose(
                adf_a.df[col].values, adf_b.df[col].values,
                rtol=1e-6, atol=1e-9, equal_nan=True,
                err_msg=f"I15_2: forward vs reversed order differs for {col}"
            )
