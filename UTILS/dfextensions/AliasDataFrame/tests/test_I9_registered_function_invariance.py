"""
Batch 2 — I9: Registered Function Identity Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established in Batch 1).
All tests are marked @pytest.mark.invariance.

SCOPE DEVIATION FROM v1.2 PROPOSAL §4.2 (per Claude30/Claude32 P1 review)
------------------------------------------------------------------------
The v1.2 proposal §4.2 specified I9_2 as the polynomial identity test
using register_polynomial_from_subframe. During implementation the Coder
(Claude31) observed that polynomial persistence is already extensively
covered by existing tests:
  - tests/test_polynomial_persistence.py (the Phase 13.9.Fix1 suite)
  - tests/test_polynomial_spec.py::TestInvariancePolynomial

Adding a third polynomial test would duplicate existing coverage. Instead,
the delivered I9 suite covers two genuinely untested paths:

  I9_1  = register_function with a one-arg callable    (unchanged from §4.2)
  I9_2  = register_function with a two-arg callable    (SCOPE CHANGE from §4.2)

The polynomial-identity invariant originally at §4.2 I9_2 is considered
already covered by the above existing tests per Claude11 v1.1 P2-3
overlap check. This deviation was:
  - Flagged by Claude30 (P1, both rounds) and Claude32 (P1)
  - Resolved by Main Architect 2026-04-11 per Claude30 consolidation:
    "Accept substitution + rename; polynomial coverage existing."

Test method names below make the actual scope explicit.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
register_function (line 10218) is the public API. The test calls it
directly, then verifies that the registered function produces values
identical to direct numpy computation on the same arrays.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_for_func():
    """Small ADF with a numeric column for function tests."""
    df = pd.DataFrame({
        'x': np.linspace(-2.0, 2.0, 50).astype(np.float32),
        'y': np.linspace(0.5, 5.0, 50).astype(np.float32),
    })
    return AliasDataFrame(df)


class TestI9RegisteredFunctionInvariance:
    """Phase 13.12 I9 — Registered Function Identity.

    Scope deviation from v1.2 §4.2: see module docstring.
    """

    @pytest.mark.invariance
    def test_I9_1_register_function_one_arg_equals_direct_computation(
        self, adf_for_func
    ):
        """
        I9_1 INVARIANT (matches v1.2 §4.2):
            register_function('myDouble', lambda a: a*2) followed by
            add_alias('y', 'myDouble(x)') and materialize produces
            values identical to direct '2*x' computation on the same
            array.
        CODE PATH: register_function (line 10218), add_alias,
                   materialize_aliases. NOT auto-dispatch.
        PRODUCTION ENTRY POINT:
            adf.register_function('myDouble', func)
            adf.add_alias('y', 'myDouble(x)')
            adf.materialize_aliases(names=['y'])
            adf.df['y']
        REGRESSION GUARD: registered function identity through alias
                          materialization pipeline (one-arg callable).
        """
        adf = adf_for_func
        x_orig = adf.df['x'].values.copy()

        def my_double(a):
            return a * 2.0

        adf.register_function('myDouble', my_double)
        adf.add_alias('y_func', 'myDouble(x)')
        adf.materialize_aliases(names=['y_func'])

        actual = adf.df['y_func'].values
        expected = 2.0 * x_orig

        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg=(
                "I9_1: register_function one-arg path differs from "
                "direct numpy computation"
            )
        )

    @pytest.mark.invariance
    def test_I9_2_register_function_two_arg_equals_direct_computation(
        self, adf_for_func
    ):
        """
        I9_2 INVARIANT (SCOPE DEVIATION from v1.2 §4.2 — see module docstring):
            A two-argument registered function (weighted_diff) applied
            via alias produces the same values as the direct numpy
            expression on the same columns.

            This test covers two-arg register_function dispatch — a path
            not otherwise tested in the suite. The polynomial-identity
            invariant originally at v1.2 §4.2 I9_2 is already covered by
            test_polynomial_persistence.py and test_polynomial_spec.py.

        CODE PATH: register_function (line 10218) with multi-arg callable.
        PRODUCTION ENTRY POINT:
            adf.register_function('wdiff', func)
            adf.add_alias('w', 'wdiff(x, y)')
            adf.materialize_aliases(names=['w'])
        REGRESSION GUARD: multi-arg registered function dispatch.
        """
        adf = adf_for_func
        x_orig = adf.df['x'].values.copy()
        y_orig = adf.df['y'].values.copy()

        def weighted_diff(a, b):
            return a * 0.3 - b * 0.7

        adf.register_function('wdiff', weighted_diff)
        adf.add_alias('w', 'wdiff(x, y)')
        adf.materialize_aliases(names=['w'])

        actual = adf.df['w'].values
        expected = x_orig * 0.3 - y_orig * 0.7

        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg=(
                "I9_2: two-arg register_function differs from direct "
                "numpy computation"
            )
        )
