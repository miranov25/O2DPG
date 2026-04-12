"""
Batch 4 — I13: Backend Equivalence Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established Batch 1).
All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.4)
---------------------
Verify that numba and numpy backends produce identical results for
the same alias expression on the same data. use_numba is a public
constructor parameter (AliasDataFrame.py:781) that controls backend
dispatch; this enables side-by-side comparison.

KNOWN-BROKEN TEST FOR CONTEXT
-----------------------------
test_invariance_backend.py::test_I2_6_chained_subframe_expressions_numba_vs_numpy
is 🧨 BACK.invariance (xfail). I13 tests below cover DIFFERENT paths
(simple arithmetic and join-scatter) to avoid the known-broken chained
subframe expression path.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Each test constructs two ADFs from the same DataFrame with different
backend toggles (use_numba=True/False), applies the same alias, and
compares results. Entry point: AliasDataFrame(df, use_numba=...).
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame, NUMBA_AVAILABLE

_requires_numba = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not available"
)


@pytest.fixture
def two_adfs_numba_numpy():
    """Two ADFs sharing the same DataFrame, one with numba, one without."""
    n = 500
    rng = np.random.RandomState(13131)
    df = pd.DataFrame({
        'x': rng.uniform(-5.0, 5.0, n).astype(np.float32),
        'y': rng.uniform(0.1, 10.0, n).astype(np.float32),
    })
    adf_numba = AliasDataFrame(df.copy(), use_numba=True)
    adf_numpy = AliasDataFrame(df.copy(), use_numba=False)
    return adf_numba, adf_numpy


class TestI13BackendEquivalenceInvariance:
    """Phase 13.12 I13 — Backend equivalence (numba == numpy)."""

    @_requires_numba
    @pytest.mark.invariance
    def test_I13_1_simple_arithmetic_numba_equals_numpy(
        self, two_adfs_numba_numpy
    ):
        """I13_1: 'x * y + 2.0' identical under numba vs numpy backends."""
        adf_n, adf_p = two_adfs_numba_numpy
        adf_n.add_alias('r', 'x * y + 2.0')
        adf_p.add_alias('r', 'x * y + 2.0')
        adf_n.materialize_aliases(names=['r'])
        adf_p.materialize_aliases(names=['r'])
        np.testing.assert_allclose(
            adf_n.df['r'].values, adf_p.df['r'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I13_1: numba vs numpy diverge on simple arithmetic"
        )

    @_requires_numba
    @pytest.mark.invariance
    def test_I13_2_compound_expression_numba_equals_numpy(
        self, two_adfs_numba_numpy
    ):
        """I13_2: 'sqrt(x*x + y*y)' identical under numba vs numpy."""
        adf_n, adf_p = two_adfs_numba_numpy
        adf_n.add_alias('mag', 'sqrt(x*x + y*y)')
        adf_p.add_alias('mag', 'sqrt(x*x + y*y)')
        adf_n.materialize_aliases(names=['mag'])
        adf_p.materialize_aliases(names=['mag'])
        np.testing.assert_allclose(
            adf_n.df['mag'].values, adf_p.df['mag'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I13_2: numba vs numpy diverge on sqrt(x^2+y^2)"
        )

    @_requires_numba
    @pytest.mark.invariance
    def test_I13_3_subframe_scatter_numba_equals_numpy(self):
        """
        I13_3: Simple subframe scatter 'S.gain' identical under
        numba vs numpy backends. Avoids the known-broken chained-
        subframe path (test_I2_6 🧨 BACK.invariance).
        """
        n = 500
        rng = np.random.RandomState(13133)
        df_main = pd.DataFrame({
            'sector': rng.randint(0, 36, n).astype(np.int8),
            'pt': rng.uniform(0.5, 10.0, n).astype(np.float32),
        })
        df_sub = pd.DataFrame({
            'sector': np.arange(36, dtype=np.int8),
            'gain': np.linspace(0.9, 1.1, 36).astype(np.float32),
        })

        adf_n = AliasDataFrame(df_main.copy(), use_numba=True)
        adf_n.register_subframe('S', AliasDataFrame(df_sub.copy()),
                                index_columns='sector')
        adf_n.add_alias('g', 'S.gain')
        adf_n.materialize_aliases(names=['g'])

        adf_p = AliasDataFrame(df_main.copy(), use_numba=False)
        adf_p.register_subframe('S', AliasDataFrame(df_sub.copy()),
                                index_columns='sector')
        adf_p.add_alias('g', 'S.gain')
        adf_p.materialize_aliases(names=['g'])

        np.testing.assert_allclose(
            adf_n.df['g'].values, adf_p.df['g'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I13_3: numba vs numpy diverge on subframe scatter"
        )
