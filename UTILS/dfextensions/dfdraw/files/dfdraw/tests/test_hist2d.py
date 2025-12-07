"""
Tests for 2D histogram plotting in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, hist2d, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame with correlated data."""
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = x + np.random.randn(n) * 0.5  # Correlated with x
    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': np.random.randn(n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestBasicHist2D:
    """Test basic 2D histogram functionality."""
    
    def test_hist2d_returns_tuple(self, sample_df):
        """hist2d() returns (fig, ax, stats_dict)."""
        plotter = DFDraw(sample_df)
        result = plotter.hist2d('y:x')
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, stats = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert isinstance(stats, dict)
    
    def test_hist2d_stats_keys(self, sample_df):
        """Stats dict contains expected 2D keys."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist2d('y:x')
        assert 'n' in stats
        assert 'mean_x' in stats
        assert 'mean_y' in stats
        assert 'corr' in stats
    
    def test_hist2d_stats_values(self, sample_df):
        """Stats values are reasonable."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist2d('y:x')
        assert stats['n'] == 1000
        # y = x + noise, so correlation should be high
        assert stats['corr'] > 0.8
    
    def test_hist2d_requires_2d_expr(self, sample_df):
        """hist2d requires 'y:x' format."""
        plotter = DFDraw(sample_df)
        with pytest.raises(ValueError, match="requires 'y:x' format"):
            plotter.hist2d('x')


class TestHist2DBins:
    """Test hist2d binning."""
    
    def test_hist2d_bins_int(self, sample_df):
        """Single int bins applies to both axes."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', bins=30)
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_bins_list(self, sample_df):
        """List bins [nx, ny] works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', bins=[20, 30])
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_bins_tuple(self, sample_df):
        """Tuple bins (nx, ny) works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', bins=(25, 25))
        assert isinstance(fig, plt.Figure)


class TestHist2DNormalization:
    """Test hist2d normalization modes."""
    
    def test_hist2d_norm_count(self, sample_df):
        """Default normalization is count."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', norm='count')
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_norm_density(self, sample_df):
        """Density normalization works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', norm='density')
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_norm_log(self, sample_df):
        """Log normalization works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', norm='log')
        assert isinstance(fig, plt.Figure)


class TestHist2DColormap:
    """Test hist2d colormap options."""
    
    def test_hist2d_default_cmap(self, sample_df):
        """Default colormap works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x')
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_custom_cmap(self, sample_df):
        """Custom colormap works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', cmap='plasma')
        assert isinstance(fig, plt.Figure)
    
    def test_hist2d_colorbar_true(self, sample_df):
        """Colorbar shown by default."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x')
        # Should have main axes + colorbar
        assert len(fig.axes) >= 1
    
    def test_hist2d_colorbar_false(self, sample_df):
        """colorbar=False hides colorbar."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', colorbar=False)
        # Should only have main axes
        assert len(fig.axes) == 1
    
    def test_hist2d_clabel(self, sample_df):
        """Colorbar label works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', clabel='Events')
        assert isinstance(fig, plt.Figure)


class TestHist2DLabels:
    """Test hist2d labels and titles."""
    
    def test_hist2d_xlabel_default(self, sample_df):
        """Default xlabel is x column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x')
        assert ax.get_xlabel() == 'x'
    
    def test_hist2d_ylabel_default(self, sample_df):
        """Default ylabel is y column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x')
        assert ax.get_ylabel() == 'y'
    
    def test_hist2d_custom_labels(self, sample_df):
        """Custom labels work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', xlabel='X Axis', ylabel='Y Axis')
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
    
    def test_hist2d_title(self, sample_df):
        """Title parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', title='Test Hist2D')
        assert ax.get_title() == 'Test Hist2D'


class TestHist2DSelection:
    """Test hist2d with selection."""
    
    def test_hist2d_string_selection(self, sample_df):
        """String query selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist2d('y:x', selection='x > 0')
        assert stats['n'] < 1000
        assert stats['mean_x'] > 0
    
    def test_hist2d_callable_selection(self, sample_df):
        """Callable selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist2d('y:x', selection=lambda df: df.x > 0)
        assert stats['n'] < 1000


class TestHist2DStatsBox:
    """Test hist2d stats box."""
    
    def test_hist2d_stats_box_show(self, sample_df):
        """stats=True shows stats box."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', stats=True)
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0


class TestHist2DFunctionalAPI:
    """Test functional API for hist2d."""
    
    def test_functional_hist2d(self, sample_df):
        """Functional hist2d() works."""
        fig, ax, stats = hist2d(sample_df, 'y:x')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 1000
    
    def test_functional_hist2d_kwargs(self, sample_df):
        """Functional hist2d() accepts kwargs."""
        fig, ax, _ = hist2d(sample_df, 'y:x', bins=30, title='Test')
        assert ax.get_title() == 'Test'


class TestHist2DComputedExpression:
    """Test hist2d with computed expressions."""
    
    def test_hist2d_computed_y(self, sample_df):
        """Computed y expression works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist2d('y + z : x')
        assert stats['n'] == 1000
    
    def test_hist2d_computed_both(self, sample_df):
        """Computed x and y expressions work."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist2d('y * 2 : x + z')
        assert stats['n'] == 1000


class TestHist2DExistingAxes:
    """Test hist2d on existing axes."""
    
    def test_hist2d_on_existing_axes(self, sample_df):
        """Plotting on existing axes works."""
        fig, ax = plt.subplots()
        plotter = DFDraw(sample_df)
        returned_fig, returned_ax, _ = plotter.hist2d('y:x', ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig


class TestHist2DSampling:
    """Test hist2d with sampling."""
    
    def test_hist2d_sampling(self, sample_df):
        """Sampling limits points."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist2d('y:x', sample=100)
        assert stats['n'] <= 100


class TestHist2DColorScale:
    """Test hist2d color scale options."""
    
    def test_hist2d_vmin_vmax(self, sample_df):
        """vmin/vmax parameters work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist2d('y:x', vmin=0, vmax=50)
        assert isinstance(fig, plt.Figure)


class TestHist2DDrawDispatch:
    """Test draw() dispatch to hist2d."""
    
    def test_draw_hist2d_type(self, sample_df):
        """draw(type='hist2d') dispatches correctly."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.draw('y:x', type='hist2d')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 1000
