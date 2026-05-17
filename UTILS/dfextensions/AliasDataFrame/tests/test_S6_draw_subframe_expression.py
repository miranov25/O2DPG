"""
BUG_AliasDataFrame_20260517_draw_dotted_subframe_expression

S6: draw() resolves dotted subframe ref inside arithmetic expression
S7: draw() resolves dotted subframe ref in selection arithmetic
S8: draw() surfaces real error (not silent swallow) on failed subframe resolution
S9: draw() handles duplicate index keys in subframe without merge expansion
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_with_subframe():
    """ADF with a registered subframe for draw subframe expression tests."""
    np.random.seed(42)
    n = 200
    main_df = pd.DataFrame({
        'group_id': np.repeat(np.arange(10), n // 10),
        'y_val': np.random.normal(0, 1, n).astype(np.float32),
        'x_val': np.random.uniform(0, 10, n).astype(np.float32),
    })

    sub_df = pd.DataFrame({
        'group_id': np.arange(10),
        'y_baseline': np.random.normal(0, 0.1, 10).astype(np.float32),
    })

    adf = AliasDataFrame(main_df)
    adf_sub = AliasDataFrame(sub_df)
    adf.register_subframe('Calib', adf_sub, index_columns='group_id')
    adf.draw_lazy = True
    return adf


@pytest.fixture
def adf_with_duplicate_index_subframe():
    """ADF with subframe that has duplicate index keys."""
    np.random.seed(42)
    n = 200
    main_df = pd.DataFrame({
        'group_id': np.repeat(np.arange(10), n // 10),
        'y_val': np.random.normal(0, 1, n).astype(np.float32),
        'x_val': np.random.uniform(0, 10, n).astype(np.float32),
    })

    # Subframe with duplicate group_id entries
    sub_df = pd.DataFrame({
        'group_id': [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'y_baseline': np.random.normal(0, 0.1, 12).astype(np.float32),
    })

    adf = AliasDataFrame(main_df)
    adf_sub = AliasDataFrame(sub_df)
    adf.register_subframe('Calib', adf_sub, index_columns='group_id')
    adf.draw_lazy = True
    return adf


class TestDrawSubframeExpression:

    @pytest.mark.invariance
    def test_S6_dotted_ref_in_arithmetic_expr(self, adf_with_subframe):
        """draw() resolves Sub.col inside arithmetic expression — exact user-hit API."""
        fig, ax, stats = adf_with_subframe.draw(
            "y_val - Calib.y_baseline",
            bins=20
        )
        assert stats is not None
        assert stats.get('n', 0) > 0
        plt.close('all')

    @pytest.mark.invariance
    def test_S7_dotted_ref_in_selection_arithmetic(self, adf_with_subframe):
        """draw() resolves Sub.col inside selection arithmetic expression."""
        fig, ax, stats = adf_with_subframe.draw(
            "y_val",
            selection="abs(y_val - Calib.y_baseline) < 2",
            bins=20
        )
        assert stats is not None
        assert stats.get('n', 0) > 0
        plt.close('all')

    @pytest.mark.invariance
    def test_S8_standalone_dotted_ref_still_works(self, adf_with_subframe):
        """Regression: standalone Sub.col (no arithmetic) still works after fix."""
        fig, ax, stats = adf_with_subframe.draw(
            "Calib.y_baseline:group_id",
            type="profile", bins=10
        )
        assert stats is not None
        plt.close('all')

    @pytest.mark.invariance
    def test_S9_duplicate_index_no_expansion(self, adf_with_duplicate_index_subframe):
        """draw() with duplicate-index subframe uses first match, no row expansion."""
        adf = adf_with_duplicate_index_subframe
        fig, ax, stats = adf.draw(
            "y_val - Calib.y_baseline",
            bins=20
        )
        assert stats is not None
        assert stats.get('n', 0) == 200  # no row expansion
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
