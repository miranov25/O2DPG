"""
Tests for hexbin plotting in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, hexbin, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame with correlated data."""
    np.random.seed(42)
    n = 5000
    x = np.random.randn(n)
    y = x + np.random.randn(n) * 0.5  # Correlated with x
    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestBasicHexbin:
    """Test basic hexbin functionality."""
    
    def test_hexbin_returns_tuple(self, sample_df):
        """hexbin() returns (fig, ax, stats_dict)."""
        plotter = DFDraw(sample_df)
        result = plotter.hexbin('y:x')
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, stats = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert isinstance(stats, dict)
    
    def test_hexbin_stats_keys(self, sample_df):
        """Stats dict contains expected 2D keys."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hexbin('y:x')
        assert 'n' in stats
        assert 'mean_x' in stats
        assert 'mean_y' in stats
        assert 'corr' in stats
    
    def test_hexbin_stats_values(self, sample_df):
        """Stats values are reasonable."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hexbin('y:x')
        assert stats['n'] == 5000
        # y = x + noise, so correlation should be high
        assert stats['corr'] > 0.8
    
    def test_hexbin_requires_2d_expr(self, sample_df):
        """hexbin requires 'y:x' format."""
        plotter = DFDraw(sample_df)
        with pytest.raises(ValueError, match="requires 'y:x' format"):
            plotter.hexbin('x')


class TestHexbinGridsize:
    """Test hexbin gridsize parameter."""
    
    def test_hexbin_gridsize_default(self, sample_df):
        """Default gridsize works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        assert isinstance(fig, plt.Figure)
    
    def test_hexbin_gridsize_small(self, sample_df):
        """Small gridsize works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', gridsize=10)
        assert isinstance(fig, plt.Figure)
    
    def test_hexbin_gridsize_large(self, sample_df):
        """Large gridsize works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', gridsize=100)
        assert isinstance(fig, plt.Figure)


class TestHexbinNormalization:
    """Test hexbin normalization modes."""
    
    def test_hexbin_norm_default(self, sample_df):
        """Default normalization (count) works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        assert isinstance(fig, plt.Figure)
    
    def test_hexbin_norm_log(self, sample_df):
        """Log normalization works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', norm='log')
        assert isinstance(fig, plt.Figure)


class TestHexbinColormap:
    """Test hexbin colormap options."""
    
    def test_hexbin_default_cmap(self, sample_df):
        """Default colormap works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        assert isinstance(fig, plt.Figure)
    
    def test_hexbin_custom_cmap(self, sample_df):
        """Custom colormap works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', cmap='plasma')
        assert isinstance(fig, plt.Figure)
    
    def test_hexbin_colorbar_true(self, sample_df):
        """Colorbar shown by default."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        # Should have main axes + colorbar
        assert len(fig.axes) >= 1
    
    def test_hexbin_colorbar_false(self, sample_df):
        """colorbar=False hides colorbar."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', colorbar=False)
        # Should only have main axes
        assert len(fig.axes) == 1
    
    def test_hexbin_clabel(self, sample_df):
        """Colorbar label works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', clabel='Events')
        assert isinstance(fig, plt.Figure)


class TestHexbinLabels:
    """Test hexbin labels and titles."""
    
    def test_hexbin_xlabel_default(self, sample_df):
        """Default xlabel is x column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        assert ax.get_xlabel() == 'x'
    
    def test_hexbin_ylabel_default(self, sample_df):
        """Default ylabel is y column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x')
        assert ax.get_ylabel() == 'y'
    
    def test_hexbin_custom_labels(self, sample_df):
        """Custom labels work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', xlabel='X Axis', ylabel='Y Axis')
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
    
    def test_hexbin_title(self, sample_df):
        """Title parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', title='Test Hexbin')
        assert ax.get_title() == 'Test Hexbin'


class TestHexbinSelection:
    """Test hexbin with selection."""
    
    def test_hexbin_string_selection(self, sample_df):
        """String query selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hexbin('y:x', selection='x > 0')
        assert stats['n'] < 5000
        assert stats['mean_x'] > 0
    
    def test_hexbin_callable_selection(self, sample_df):
        """Callable selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hexbin('y:x', selection=lambda df: df.x > 0)
        assert stats['n'] < 5000


class TestHexbinStatsBox:
    """Test hexbin stats box."""
    
    def test_hexbin_stats_box_show(self, sample_df):
        """stats=True shows stats box."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', stats=True)
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0


class TestHexbinFunctionalAPI:
    """Test functional API for hexbin."""
    
    def test_functional_hexbin(self, sample_df):
        """Functional hexbin() works."""
        fig, ax, stats = hexbin(sample_df, 'y:x')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 5000
    
    def test_functional_hexbin_kwargs(self, sample_df):
        """Functional hexbin() accepts kwargs."""
        fig, ax, _ = hexbin(sample_df, 'y:x', gridsize=30, title='Test')
        assert ax.get_title() == 'Test'


class TestHexbinComputedExpression:
    """Test hexbin with computed expressions."""
    
    def test_hexbin_computed_y(self, sample_df):
        """Computed y expression works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hexbin('y + z : x')
        assert stats['n'] == 5000
    
    def test_hexbin_computed_both(self, sample_df):
        """Computed x and y expressions work."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hexbin('y * 2 : x + z')
        assert stats['n'] == 5000


class TestHexbinExistingAxes:
    """Test hexbin on existing axes."""
    
    def test_hexbin_on_existing_axes(self, sample_df):
        """Plotting on existing axes works."""
        fig, ax = plt.subplots()
        plotter = DFDraw(sample_df)
        returned_fig, returned_ax, _ = plotter.hexbin('y:x', ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig


class TestHexbinSampling:
    """Test hexbin with sampling."""
    
    def test_hexbin_sampling(self, sample_df):
        """Sampling limits points."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hexbin('y:x', sample=100)
        assert stats['n'] <= 100


class TestHexbinMincnt:
    """Test hexbin mincnt parameter."""
    
    def test_hexbin_mincnt(self, sample_df):
        """mincnt parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', mincnt=5)
        assert isinstance(fig, plt.Figure)


class TestHexbinColorScale:
    """Test hexbin color scale options."""
    
    def test_hexbin_vmin_vmax(self, sample_df):
        """vmin/vmax parameters work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hexbin('y:x', vmin=0, vmax=50)
        assert isinstance(fig, plt.Figure)


class TestHexbinFacet:
    """Test hexbin with facet=True."""
    
    def test_hexbin_facet_true(self, sample_df):
        """hexbin(facet=True) creates faceted plot."""
        plotter = DFDraw(sample_df)
        fig, axes, stats = plotter.hexbin('y:x', group_by='category', facet=True)
        assert stats['faceted'] is True
        assert stats['n_groups'] == 3
    
    def test_hexbin_facet_top_k(self, sample_df):
        """hexbin(facet=True, top_k=2) limits facets."""
        plotter = DFDraw(sample_df)
        fig, axes, stats = plotter.hexbin('y:x', group_by='category', facet=True, top_k=2)
        assert stats['n_groups'] == 2
    
    def test_hexbin_facet_with_gridsize(self, sample_df):
        """hexbin(facet=True, gridsize=...) works."""
        plotter = DFDraw(sample_df)
        fig, axes, stats = plotter.hexbin(
            'y:x', group_by='category', facet=True, gridsize=20
        )
        assert stats['faceted'] is True
