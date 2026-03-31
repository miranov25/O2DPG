"""
Test for BUG_AliasDataFrame_20260331_fill_value_dependency_resolution

fill_value must be applied when alias is materialized as a dependency,
not only when materialized directly.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_with_incomplete_subframe():
    """ADF where subframe has fewer keys than main frame."""
    df_main = pd.DataFrame({
        'key': [0, 1, 2, 3, 4, 5],
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    })
    # Subframe only has keys 0,1,2 — keys 3,4,5 will produce NaN on join
    df_sub = pd.DataFrame({
        'key': [0, 1, 2],
        'coeff': [0.1, 0.2, 0.3],
        'offset': [1.0, 2.0, 3.0],
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('S', AliasDataFrame(df_sub), index_columns=['key'])
    return adf


class TestFillValueDependencyResolution:
    """BUG_AliasDataFrame_20260331: fill_value skipped during dependency resolution."""

    def test_direct_materialize_applies_fill_value(self, adf_with_incomplete_subframe):
        """Direct materialize_alias applies fill_value correctly."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.materialize_alias('A')
        assert adf.df['A'].isna().sum() == 0
        # Keys 3,4,5 should be 0 (fill_value), not NaN
        assert adf.df['A'].iloc[3] == 0
        assert adf.df['A'].iloc[4] == 0
        assert adf.df['A'].iloc[5] == 0

    def test_dependency_materialize_applies_fill_value(self, adf_with_incomplete_subframe):
        """Materializing B that depends on A with fill_value — A's fill_value must apply."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.add_alias('B', 'x + A')
        adf.materialize_aliases(names=['B'])
        # B should have NO NaN — A's fill_value=0 should be applied
        assert adf.df['B'].isna().sum() == 0
        # Verify values: for key=3, A=0 (fill), x=4.0 → B=4.0
        np.testing.assert_allclose(adf.df['B'].iloc[3], 4.0, atol=1e-6)
        np.testing.assert_allclose(adf.df['B'].iloc[4], 5.0, atol=1e-6)

    def test_dependency_chain_applies_fill_value(self, adf_with_incomplete_subframe):
        """Chain: C depends on B depends on A (with fill_value)."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.add_alias('B', 'x * A')
        adf.add_alias('C', 'y - B')
        adf.materialize_aliases(names=['C'])
        # C should have NO NaN
        assert adf.df['C'].isna().sum() == 0
        # For key=4: A=0, B=5*0=0, C=50-0=50
        np.testing.assert_allclose(adf.df['C'].iloc[4], 50.0, atol=1e-6)

    def test_multiple_fill_value_dependencies(self, adf_with_incomplete_subframe):
        """Two dependencies both have fill_value."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.add_alias('B', 'S.offset', fill_value=0)
        adf.add_alias('C', 'A + B')
        adf.materialize_aliases(names=['C'])
        assert adf.df['C'].isna().sum() == 0
        # For key=0: A=0.1, B=1.0 → C=1.1
        np.testing.assert_allclose(adf.df['C'].iloc[0], 1.1, atol=1e-6)
        # For key=4: A=0, B=0 → C=0
        np.testing.assert_allclose(adf.df['C'].iloc[4], 0.0, atol=1e-6)

    def test_fill_value_with_multiplication_guard(self, adf_with_incomplete_subframe):
        """Production pattern: (row<152) * SubframeCoeff — NaN * 0 = NaN, not 0."""
        adf = adf_with_incomplete_subframe
        # Simulate: (key<3) * S.coeff — keys 3,4,5 have key>=3 so multiplier is 0,
        # but S.coeff is NaN for those keys. 0 * NaN = NaN.
        adf.add_alias('correction', '(key < 3) * S.coeff', fill_value=0)
        adf.add_alias('residual', 'y - correction')
        adf.materialize_aliases(names=['residual'])
        assert adf.df['residual'].isna().sum() == 0

    def test_batch_materialize_preserves_direct_fill_value(self, adf_with_incomplete_subframe):
        """Batch materialize of A directly still applies fill_value."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.materialize_aliases(names=['A'])
        assert adf.df['A'].isna().sum() == 0

    @pytest.mark.invariance
    def test_invariance_direct_vs_dependency(self, adf_with_incomplete_subframe):
        """Invariance: direct materialize == dependency materialize for fill_value alias."""
        adf = adf_with_incomplete_subframe
        adf.add_alias('A', 'S.coeff', fill_value=0)
        adf.add_alias('B', 'x + A')

        # Path 1: materialize A directly, then B
        adf.materialize_alias('A')
        adf.materialize_alias('B')
        b_direct = adf.df['B'].values.copy()

        # Reset
        adf.df.drop(columns=['A', 'B'], inplace=True)

        # Path 2: materialize only B (A as dependency)
        adf.materialize_aliases(names=['B'])
        b_dependency = adf.df['B'].values.copy()

        # Must be identical
        np.testing.assert_allclose(b_direct, b_dependency, atol=1e-6,
                                    err_msg="Direct vs dependency materialization differ")
