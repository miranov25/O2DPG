"""
Tests for BUG_AliasDataFrame_20260324_draw_subframe_resolution

Verifies that draw() correctly resolves Subframe.column references
when the subframe has columns with names that conflict with main frame columns.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_with_subframe():
    """Create ADF with subframe that has conflicting column names."""
    df_main = pd.DataFrame({
        'group': [0, 0, 0, 1, 1, 1],
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        'dy': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    })
    df_sub = pd.DataFrame({
        'group': [0, 1],
        'dy': [0.2, 0.5],     # conflicts with main frame
        'dz': [0.01, 0.02],   # no conflict
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('Sub', AliasDataFrame(df_sub), index_columns=['group'])
    return adf


class TestDrawSubframeResolution:
    """draw() should resolve Subframe.column references automatically."""

    def test_draw_subframe_column_no_conflict(self, adf_with_subframe):
        """draw('Sub.dz:x') works — dz has no name conflict."""
        import matplotlib
        matplotlib.use('Agg')
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dz:x", type='profile', bins=3)
        assert fig is not None
        assert ax is not None

    def test_draw_subframe_column_with_conflict(self, adf_with_subframe):
        """draw('Sub.dy:x') works — dy exists in both main and subframe."""
        import matplotlib
        matplotlib.use('Agg')
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=3)
        assert fig is not None
        assert ax is not None

    def test_draw_subframe_column_values_correct(self, adf_with_subframe):
        """Sub.dy values match subframe join (not main frame dy)."""
        import matplotlib
        matplotlib.use('Agg')
        adf = adf_with_subframe

        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=2,
                                   return_data=True)
        profile_data = stats.get('profile_data')
        assert profile_data is not None

        # Group 0 rows (x=1,2,3) should have Sub.dy = 0.2 (subframe value)
        # Group 1 rows (x=4,5,6) should have Sub.dy = 0.5 (subframe value)
        # Profile mean in first bin (x~2) should be ~0.2
        # Profile mean in second bin (x~5) should be ~0.5
        means = profile_data['y_mean'].values
        assert abs(means[0] - 0.2) < 0.01, f"Expected ~0.2, got {means[0]}"
        assert abs(means[1] - 0.5) < 0.01, f"Expected ~0.5, got {means[1]}"

    def test_draw_subframe_in_selection(self, adf_with_subframe):
        """Subframe.column works in selection parameter."""
        import matplotlib
        matplotlib.use('Agg')
        adf = adf_with_subframe
        # Use Sub.dz in selection (no conflict column)
        fig, ax, stats = adf.draw("dy:x", type='profile', bins=3,
                                   selection="Sub.dz<0.015")
        assert fig is not None
        # Should only have group 0 data (Sub.dz=0.01 < 0.015)

    def test_auto_alias_subframe_conflict_skips(self, adf_with_subframe):
        """auto_alias_subframe skips columns that exist in main frame."""
        adf = adf_with_subframe
        result = adf.auto_alias_subframe('Sub')
        # 'dy' should be skipped (exists in main frame)
        assert 'dy' in result['skipped_column']
        # 'dz' should be created (no conflict)
        assert any('dz' in alias for alias in result['created'])
