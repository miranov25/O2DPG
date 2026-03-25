"""
Tests for BUG_AliasDataFrame_20260324_draw_subframe_resolution

Verifies that draw(), draw_batch() correctly resolve Subframe.column
references in all combinations of:
  - expr, selection, group_by
  - with/without entry_end (sliced DataFrames)
  - conflicting vs non-conflicting column names
  - single vs multiple subframe references
  - single-key vs multi-key joins
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

import matplotlib
matplotlib.use('Agg')


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


@pytest.fixture
def adf_multikey_subframe():
    """ADF with multi-key subframe (closer to production use case)."""
    df_main = pd.DataFrame({
        'sector': [0, 0, 0, 0, 1, 1, 1, 1],
        'bin': [0, 0, 1, 1, 0, 0, 1, 1],
        'x': np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
        'dy': np.array([.1, .2, .3, .4, .5, .6, .7, .8]),
    })
    df_sub = pd.DataFrame({
        'sector': [0, 0, 1, 1],
        'bin': [0, 1, 0, 1],
        'dyS': [0.15, 0.35, 0.55, 0.75],
        'dzS': [0.01, 0.02, 0.03, 0.04],
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('Side', AliasDataFrame(df_sub),
                          index_columns=['sector', 'bin'])
    return adf


# =========================================================================
# draw() tests
# =========================================================================

class TestDrawSubframeResolution:
    """draw() should resolve Subframe.column references automatically."""

    def test_draw_subframe_column_no_conflict(self, adf_with_subframe):
        """draw('Sub.dz:x') works — dz has no name conflict."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dz:x", type='profile', bins=3)
        assert fig is not None
        assert ax is not None

    def test_draw_subframe_column_with_conflict(self, adf_with_subframe):
        """draw('Sub.dy:x') works — dy exists in both main and subframe."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=3)
        assert fig is not None

    def test_draw_subframe_column_values_correct(self, adf_with_subframe):
        """Sub.dy values match subframe join (not main frame dy)."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=2,
                                   return_data=True)
        profile_data = stats.get('profile_data')
        assert profile_data is not None
        # Group 0 rows (x=1,2,3) → Sub.dy = 0.2, Group 1 (x=4,5,6) → Sub.dy = 0.5
        means = profile_data['y_mean'].values
        assert abs(means[0] - 0.2) < 0.01, f"Expected ~0.2, got {means[0]}"
        assert abs(means[1] - 0.5) < 0.01, f"Expected ~0.5, got {means[1]}"

    def test_draw_subframe_in_selection(self, adf_with_subframe):
        """Subframe.column works in selection parameter."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("dy:x", type='profile', bins=3,
                                   selection="Sub.dz<0.015")
        assert fig is not None

    def test_draw_subframe_with_entry_end(self, adf_with_subframe):
        """Subframe.column works with entry_end (sliced DataFrame)."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dz:x", type='profile', bins=2,
                                   entry_end=4)
        assert fig is not None

    def test_draw_subframe_with_entry_end_values(self, adf_with_subframe):
        """Sliced DataFrame gets correct subframe values."""
        adf = adf_with_subframe
        # entry_end=3 gives only group 0 rows (x=1,2,3)
        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=1,
                                   entry_end=3, return_data=True)
        profile_data = stats.get('profile_data')
        assert profile_data is not None
        # All rows are group 0, Sub.dy should be 0.2
        assert abs(profile_data['y_mean'].values[0] - 0.2) < 0.01

    def test_draw_multiple_subframe_refs(self, adf_with_subframe):
        """Multiple Subframe.column refs in same draw call."""
        adf = adf_with_subframe
        fig, ax, stats = adf.draw("Sub.dy:x", type='profile', bins=2,
                                   selection="Sub.dz<0.025")
        assert fig is not None

    def test_draw_subframe_multikey(self, adf_multikey_subframe):
        """Subframe with multi-key join works in draw."""
        adf = adf_multikey_subframe
        fig, ax, stats = adf.draw("Side.dyS:x", type='profile', bins=4)
        assert fig is not None

    def test_draw_subframe_multikey_with_entry_end(self, adf_multikey_subframe):
        """Multi-key subframe works with entry_end."""
        adf = adf_multikey_subframe
        fig, ax, stats = adf.draw("Side.dyS:x", type='profile', bins=2,
                                   entry_end=4)
        assert fig is not None

    def test_auto_alias_subframe_conflict_skips(self, adf_with_subframe):
        """auto_alias_subframe skips columns that exist in main frame."""
        adf = adf_with_subframe
        result = adf.auto_alias_subframe('Sub')
        assert 'dy' in result['skipped_column']
        assert any('dz' in alias for alias in result['created'])


# =========================================================================
# draw_batch() tests
# =========================================================================

class TestDrawBatchSubframeResolution:
    """draw_batch() should resolve Subframe.column references."""

    def test_draw_batch_subframe_basic(self, adf_with_subframe):
        """draw_batch with Subframe.column in expr."""
        adf = adf_with_subframe
        plots = {
            "test_plot": {
                "expr": "Sub.dz:x", "type": "profile", "bins": 3,
            }
        }
        results = adf.draw_batch(plots, close_figures=True)
        assert 'test_plot' in results
        assert len(results.get('_errors', {})) == 0

    def test_draw_batch_subframe_in_selection(self, adf_with_subframe):
        """draw_batch with Subframe.column in selection."""
        adf = adf_with_subframe
        plots = {
            "test_plot": {
                "expr": "dy:x", "type": "profile", "bins": 3,
                "selection": "Sub.dz<0.015",
            }
        }
        results = adf.draw_batch(plots, close_figures=True)
        assert 'test_plot' in results
        assert len(results.get('_errors', {})) == 0

    def test_draw_batch_multiple_plots(self, adf_with_subframe):
        """draw_batch with multiple plots using subframe refs."""
        adf = adf_with_subframe
        plots = {
            "plot1": {"expr": "Sub.dy:x", "type": "profile", "bins": 2},
            "plot2": {"expr": "Sub.dz:x", "type": "profile", "bins": 2},
        }
        results = adf.draw_batch(plots, close_figures=True)
        assert 'plot1' in results
        assert 'plot2' in results
        assert len(results.get('_errors', {})) == 0

    def test_draw_batch_subframe_conflict(self, adf_with_subframe):
        """draw_batch works when subframe column conflicts with main."""
        adf = adf_with_subframe
        plots = {
            "conflict": {
                "expr": "Sub.dy:x", "type": "profile", "bins": 2,
            }
        }
        results = adf.draw_batch(plots, close_figures=True)
        assert 'conflict' in results
        assert len(results.get('_errors', {})) == 0

    def test_draw_batch_multikey(self, adf_multikey_subframe):
        """draw_batch with multi-key subframe."""
        adf = adf_multikey_subframe
        plots = {
            "multikey": {
                "expr": "Side.dyS:x", "type": "profile", "bins": 4,
            }
        }
        results = adf.draw_batch(plots, close_figures=True)
        assert 'multikey' in results
        assert len(results.get('_errors', {})) == 0


# =========================================================================
# Edge cases
# =========================================================================

class TestDrawSubframeEdgeCases:
    """Edge cases for subframe resolution in draw."""

    def test_no_subframe_registered(self):
        """draw works normally when no subframes exist."""
        df = pd.DataFrame({'x': [1., 2., 3.], 'y': [4., 5., 6.]})
        adf = AliasDataFrame(df)
        fig, ax, stats = adf.draw("y:x", type='profile', bins=2)
        assert fig is not None

    def test_subframe_column_not_found(self, adf_with_subframe):
        """Non-existent subframe column is handled gracefully."""
        adf = adf_with_subframe
        # Sub.nonexistent — should raise because column doesn't exist
        with pytest.raises((ValueError, KeyError)):
            adf.draw("Sub.nonexistent:x", type='profile', bins=2)

    def test_dot_in_non_subframe_name(self, adf_with_subframe):
        """Dot expressions that aren't subframe refs don't break."""
        adf = adf_with_subframe
        adf.add_alias('y_scaled', 'x * 3.14')
        adf.materialize_alias('y_scaled')
        fig, ax, stats = adf.draw("y_scaled:x", type='profile', bins=2)
        assert fig is not None

    def test_draw_preserves_original_df(self, adf_with_subframe):
        """draw() doesn't modify self.df with temporary subframe columns."""
        adf = adf_with_subframe
        cols_before = set(adf.df.columns)
        fig, ax, stats = adf.draw("Sub.dz:x", type='profile', bins=2)
        cols_after = set(adf.df.columns)
        assert cols_before == cols_after, f"Columns leaked: {cols_after - cols_before}"


# =========================================================================
# draw_figures() tests
# =========================================================================

class TestDrawFiguresSubframeResolution:
    """draw_figures() should resolve Subframe.column references."""

    def test_draw_figures_subframe_basic(self, adf_with_subframe):
        """draw_figures with Subframe.column in plot expr."""
        adf = adf_with_subframe
        specs = [{
            'name': 'test_fig',
            'plots': [
                {'expr': 'Sub.dz:x', 'type': 'profile', 'bins': 3},
            ]
        }]
        results = adf.draw_figures(specs)
        assert 'test_fig' in results
        assert results['test_fig'].get('error') is None

    def test_draw_figures_subframe_conflict(self, adf_with_subframe):
        """draw_figures with conflicting column name."""
        adf = adf_with_subframe
        specs = [{
            'name': 'conflict_fig',
            'plots': [
                {'expr': 'Sub.dy:x', 'type': 'profile', 'bins': 2},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['conflict_fig'].get('error') is None

    def test_draw_figures_mixed_subframe_and_local(self, adf_with_subframe):
        """draw_figures with both subframe and local columns in same figure."""
        adf = adf_with_subframe
        specs = [{
            'name': 'mixed_fig',
            'ncols': 2,
            'plots': [
                {'expr': 'dy:x', 'type': 'profile', 'bins': 2},
                {'expr': 'Sub.dz:x', 'type': 'profile', 'bins': 2},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['mixed_fig'].get('error') is None

    def test_draw_figures_multikey(self, adf_multikey_subframe):
        """draw_figures with multi-key subframe."""
        adf = adf_multikey_subframe
        specs = [{
            'name': 'multikey_fig',
            'plots': [
                {'expr': 'Side.dyS:x', 'type': 'profile', 'bins': 4},
                {'expr': 'Side.dzS:x', 'type': 'profile', 'bins': 4},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['multikey_fig'].get('error') is None

    def test_draw_figures_defaults_cascade(self, adf_with_subframe):
        """Per-figure defaults propagate to plots (linestyle, type, bins)."""
        adf = adf_with_subframe
        specs = [{
            'name': 'defaults_fig',
            'defaults': {'type': 'profile', 'bins': 3, 'linestyle': 'none'},
            'plots': [
                {'expr': 'Sub.dz:x'},
                {'expr': 'dy:x'},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['defaults_fig'].get('error') is None

    def test_draw_figures_layout_tuple(self, adf_with_subframe):
        """layout=(nrows, ncols) tuple works."""
        adf = adf_with_subframe
        specs = [{
            'name': 'layout_fig',
            'layout': (1, 2),
            'plots': [
                {'expr': 'dy:x', 'type': 'profile', 'bins': 2},
                {'expr': 'Sub.dz:x', 'type': 'profile', 'bins': 2},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['layout_fig'].get('error') is None

    def test_draw_figures_subframe_in_selection(self, adf_with_subframe):
        """Subframe.column works in selection within draw_figures."""
        adf = adf_with_subframe
        specs = [{
            'name': 'sel_fig',
            'plots': [
                {'expr': 'dy:x', 'type': 'profile', 'bins': 2,
                 'selection': 'Sub.dz<0.015'},
            ]
        }]
        results = adf.draw_figures(specs)
        assert results['sel_fig'].get('error') is None

    def test_draw_figures_multiple_figures(self, adf_with_subframe):
        """Multiple figures in one draw_figures call."""
        adf = adf_with_subframe
        specs = [
            {
                'name': 'fig1',
                'plots': [{'expr': 'Sub.dz:x', 'type': 'profile', 'bins': 2}]
            },
            {
                'name': 'fig2',
                'plots': [{'expr': 'Sub.dy:x', 'type': 'profile', 'bins': 2}]
            },
        ]
        results = adf.draw_figures(specs)
        assert 'fig1' in results
        assert 'fig2' in results
        assert results['fig1'].get('error') is None
        assert results['fig2'].get('error') is None

    def test_draw_figures_preserves_df(self, adf_with_subframe):
        """draw_figures doesn't modify self.df."""
        adf = adf_with_subframe
        cols_before = set(adf.df.columns)
        specs = [{
            'name': 'preserve_fig',
            'plots': [{'expr': 'Sub.dz:x', 'type': 'profile', 'bins': 2}]
        }]
        adf.draw_figures(specs)
        cols_after = set(adf.df.columns)
        assert cols_before == cols_after
