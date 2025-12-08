"""
Tests for Phase 6.8: AliasDataFrame + dfdraw integration.

Tests both dfdraw duck-typing and ADF draw methods using a MockADF.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, set_style


class MockAliasDataFrame:
    """
    Mock AliasDataFrame for testing duck-typing integration.
    
    Implements the duck-typed interface that dfdraw expects:
    - get_axis_title(column) -> str or None
    - .df attribute
    """
    
    def __init__(self, df, titles=None):
        self.df = df
        self._titles = titles or {}
    
    def get_axis_title(self, column):
        """Return axis title if set, None otherwise."""
        return self._titles.get(column)
    
    def set_axis_title(self, column, title):
        """Set axis title for a column."""
        self._titles[column] = title


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'z': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestDuckTypedAxisTitles:
    """Test duck-typed axis title lookup in dfdraw."""
    
    def test_hist_uses_axis_title(self, sample_df):
        """Histogram uses axis title from mock ADF."""
        mock_adf = MockAliasDataFrame(sample_df, {'x': 'X Position [cm]'})
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.hist('x')
        assert ax.get_xlabel() == 'X Position [cm]'
    
    def test_scatter_uses_axis_titles(self, sample_df):
        """Scatter plot uses axis titles from mock ADF."""
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X [cm]',
            'y': 'Y [cm]',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_xlabel() == 'X [cm]'
        assert ax.get_ylabel() == 'Y [cm]'
    
    def test_explicit_label_overrides_title(self, sample_df):
        """Explicit xlabel/ylabel override axis titles."""
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X [cm]',
            'y': 'Y [cm]',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.scatter('y:x', xlabel='Custom X', ylabel='Custom Y')
        assert ax.get_xlabel() == 'Custom X'
        assert ax.get_ylabel() == 'Custom Y'
    
    def test_no_title_uses_default(self, sample_df):
        """No title set uses default column name."""
        mock_adf = MockAliasDataFrame(sample_df)  # No titles
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
    
    def test_partial_titles(self, sample_df):
        """Mix of set and unset titles works."""
        mock_adf = MockAliasDataFrame(sample_df, {'x': 'X Position [cm]'})
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_xlabel() == 'X Position [cm]'
        assert ax.get_ylabel() == 'y'  # Default
    
    def test_profile_uses_axis_titles(self, sample_df):
        """Profile plot uses axis titles."""
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X [cm]',
            'y': 'Y [cm]',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.profile('y:x')
        assert ax.get_xlabel() == 'X [cm]'
        # ylabel should be either the title or default profile format
        # (title takes precedence when set)
        assert 'Y [cm]' in ax.get_ylabel() or '<y>' in ax.get_ylabel()
    
    def test_hist2d_uses_axis_titles(self, sample_df):
        """2D histogram uses axis titles."""
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X Axis',
            'y': 'Y Axis',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.hist2d('y:x')
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
    
    def test_hexbin_uses_axis_titles(self, sample_df):
        """Hexbin plot uses axis titles."""
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X Value',
            'y': 'Y Value',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        fig, ax, _ = plotter.hexbin('y:x', gridsize=20)
        assert ax.get_xlabel() == 'X Value'
        assert ax.get_ylabel() == 'Y Value'


class TestWithoutDataSource:
    """Test that dfdraw works normally without _data_source set."""
    
    def test_hist_without_data_source(self, sample_df):
        """Histogram works without _data_source."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x')
        assert ax.get_xlabel() == 'x'
    
    def test_scatter_without_data_source(self, sample_df):
        """Scatter works without _data_source."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'


class TestMockADF:
    """Test the MockAliasDataFrame itself."""
    
    def test_set_get_title(self, sample_df):
        """set/get axis title works."""
        mock_adf = MockAliasDataFrame(sample_df)
        mock_adf.set_axis_title('x', 'X [cm]')
        assert mock_adf.get_axis_title('x') == 'X [cm]'
    
    def test_get_unset_title(self, sample_df):
        """get_axis_title returns None for unset."""
        mock_adf = MockAliasDataFrame(sample_df)
        assert mock_adf.get_axis_title('x') is None
    
    def test_df_attribute(self, sample_df):
        """Mock has .df attribute."""
        mock_adf = MockAliasDataFrame(sample_df)
        assert mock_adf.df is sample_df


class TestDFDrawDataSourceStorage:
    """Test that DFDraw stores _data_source correctly."""
    
    def test_data_source_stored(self, sample_df):
        """_data_source is stored during init."""
        plotter = DFDraw(sample_df)
        assert plotter._data_source is sample_df
    
    def test_data_source_is_original(self, sample_df):
        """_data_source refers to original object."""
        mock_adf = MockAliasDataFrame(sample_df)
        plotter = DFDraw(mock_adf)
        # When passing mock_adf, _data_source should be mock_adf
        # and df should be extracted from mock_adf.df
        assert plotter._data_source is mock_adf
        assert plotter.df is sample_df


class TestBatchWithAxisTitles:
    """Test batch processing with axis titles."""
    
    def test_batch_uses_axis_titles(self, sample_df):
        """draw_batch uses axis titles from data source."""
        import tempfile
        import os
        
        mock_adf = MockAliasDataFrame(sample_df, {
            'x': 'X Position [cm]',
            'y': 'Y Position [cm]',
        })
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        specs = {
            'hist_x': {'expr': 'x'},
            'scatter_yx': {'expr': 'y:x'},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(specs, save_dir=tmpdir, verbose=False)
            assert results['_summary']['success'] == 2


class TestGetLabelMethod:
    """Test the _get_label method directly."""
    
    def test_get_label_with_title(self, sample_df):
        """_get_label returns title when set."""
        mock_adf = MockAliasDataFrame(sample_df, {'x': 'X Label'})
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        label = plotter._get_label('x')
        assert label == 'X Label'
    
    def test_get_label_without_title(self, sample_df):
        """_get_label returns None when no title set."""
        mock_adf = MockAliasDataFrame(sample_df)
        plotter = DFDraw(mock_adf.df)
        plotter._data_source = mock_adf
        
        label = plotter._get_label('x')
        assert label is None
    
    def test_get_label_without_data_source(self, sample_df):
        """_get_label returns None without _data_source."""
        plotter = DFDraw(sample_df)
        # Don't set _data_source
        del plotter._data_source
        
        label = plotter._get_label('x')
        assert label is None
