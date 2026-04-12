"""
Batch 4 — I14: Join Index Linearization Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE. All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.4)
---------------------
Verify that subframe join produces correct row-to-row mapping:
  - Single-key: equivalent to pd.merge on the key
  - Composite-key: order of keys in index_columns does not change
    the join result (i.e. register(['k1','k2']) == register(['k2','k1'])
    after pd.merge normalization)

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Uses pd.merge as independent reference, never direct fixture slicing
(Batch 1 I6_1 P2 #3 lesson).
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestI14JoinIndexInvariance:

    @pytest.mark.invariance
    def test_I14_1_single_key_join_equals_pandas_merge(self):
        """I14_1: Single-key subframe join == pd.merge reference."""
        rng = np.random.RandomState(14141)
        df_main = pd.DataFrame({
            'k': rng.randint(0, 10, 200).astype(np.int32),
            'w': rng.uniform(0.0, 1.0, 200).astype(np.float32),
        })
        df_sub = pd.DataFrame({
            'k': np.arange(10, dtype=np.int32),
            'v': np.linspace(100.0, 109.0, 10).astype(np.float32),
        })
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub),
                              index_columns='k')
        adf.add_alias('sv', 'S.v')
        adf.materialize_aliases(names=['sv'])

        ref = df_main[['k']].merge(df_sub, on='k', how='left')['v'].values
        np.testing.assert_allclose(
            adf.df['sv'].values, ref,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I14_1: single-key join diverges from pd.merge"
        )

    @pytest.mark.invariance
    def test_I14_2_composite_key_order_does_not_affect_values(self):
        """
        I14_2: Registering a subframe with index_columns=['k1','k2']
        vs ['k2','k1'] yields the same per-row values after join
        (modulo pd.merge normalization).
        """
        rng = np.random.RandomState(14242)
        df_main = pd.DataFrame({
            'k1': rng.randint(0, 3, 60).astype(np.int32),
            'k2': rng.randint(0, 4, 60).astype(np.int32),
        })
        keys = [(a, b) for a in range(3) for b in range(4)]
        df_sub = pd.DataFrame({
            'k1': np.array([k[0] for k in keys], dtype=np.int32),
            'k2': np.array([k[1] for k in keys], dtype=np.int32),
            'v': np.linspace(1.0, 12.0, 12).astype(np.float32),
        })

        adf_a = AliasDataFrame(df_main.copy())
        adf_a.register_subframe('S', AliasDataFrame(df_sub.copy()),
                                index_columns=['k1', 'k2'])
        adf_a.add_alias('v', 'S.v')
        adf_a.materialize_aliases(names=['v'])

        adf_b = AliasDataFrame(df_main.copy())
        adf_b.register_subframe('S', AliasDataFrame(df_sub.copy()),
                                index_columns=['k2', 'k1'])
        adf_b.add_alias('v', 'S.v')
        adf_b.materialize_aliases(names=['v'])

        np.testing.assert_allclose(
            adf_a.df['v'].values, adf_b.df['v'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I14_2: composite-key order changes per-row values"
        )
