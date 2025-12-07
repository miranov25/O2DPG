"""
Tests for scatter plotting in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, scatter, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    return pd.DataFrame({
        'x': x,
        'y': x + np.random.randn(n) * 0.5,  # Correlated with x
        'z': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'size_col': np.random.uniform(10, 100, n),
        'color_col': np.random.uniform(0, 1, n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestBasicScatter:
    """Test basic scatter plot functionality."""
    
    def test_scatter_returns_tuple(self, sample_df):
        """scatter() returns (fig, ax, stats_dict)."""
        plotter = DFDraw(sample_df)
        result = plotter.scatter('y:x')
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, stats = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert isinstance(stats, dict)
    
    def test_scatter_stats_keys(self, sample_df):
        """Stats dict contains expected 2D keys."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.scatter('y:x')
        assert 'n' in stats
        assert 'mean_x' in stats
        assert 'mean_y' in stats
        assert 'std_x' in stats
        assert 'std_y' in stats
        assert 'corr' in stats
    
    def test_scatter_stats_values(self, sample_df):
        """Stats values are reasonable."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.scatter('y:x')
        assert stats['n'] == 500
        # x and y are correlated, so correlation should be positive
        assert stats['corr'] > 0.5
    
    def test_scatter_requires_2d_expr(self, sample_df):
        """Scatter requires 'y:x' format."""
        plotter = DFDraw(sample_df)
        with pytest.raises(ValueError, match="requires 'y:x' format"):
            plotter.scatter('x')  # Only 1D


class TestScatterLabels:
    """Test scatter plot labels and titles."""
    
    def test_scatter_xlabel_default(self, sample_df):
        """Default xlabel is x column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_xlabel() == 'x'
    
    def test_scatter_ylabel_default(self, sample_df):
        """Default ylabel is y column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x')
        assert ax.get_ylabel() == 'y'
    
    def test_scatter_custom_labels(self, sample_df):
        """Custom labels work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', xlabel='X Axis', ylabel='Y Axis')
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
    
    def test_scatter_title(self, sample_df):
        """Title parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', title='Test Scatter')
        assert ax.get_title() == 'Test Scatter'


class TestScatterSelection:
    """Test scatter with selection/cuts."""
    
    def test_scatter_string_selection(self, sample_df):
        """String query selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.scatter('y:x', selection='x > 0')
        assert stats['n'] < 500
        assert stats['mean_x'] > 0
    
    def test_scatter_callable_selection(self, sample_df):
        """Callable selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.scatter('y:x', selection=lambda df: df.x > 0)
        assert stats['n'] < 500


class TestScatterColorMapping:
    """Test scatter color mapping."""
    
    def test_scatter_fixed_color(self, sample_df):
        """Fixed color string works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', color='red')
        # Just verify it runs without error
        assert isinstance(fig, plt.Figure)
    
    def test_scatter_color_column(self, sample_df):
        """Color mapped to column works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', color='color_col')
        # Should have colorbar by default
        assert len(fig.axes) >= 1  # Main axes + colorbar
    
    def test_scatter_colorbar_false(self, sample_df):
        """colorbar=False hides colorbar."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', color='color_col', colorbar=False)
        # Should only have main axes
        assert len(fig.axes) == 1


class TestScatterSizeMapping:
    """Test scatter size mapping."""
    
    def test_scatter_fixed_size(self, sample_df):
        """Fixed size works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', size=100)
        assert isinstance(fig, plt.Figure)
    
    def test_scatter_size_column(self, sample_df):
        """Size mapped to column works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', size='size_col')
        assert isinstance(fig, plt.Figure)


class TestScatterGroupBy:
    """Test scatter with group_by."""
    
    def test_scatter_grouped(self, sample_df):
        """Group-by creates multiple scatter series."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.scatter('y:x', group_by='category')
        assert stats.get('grouped', False)
        legend = ax.get_legend()
        assert legend is not None
    
    def test_scatter_grouped_top_k(self, sample_df):
        """top_k limits groups."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', group_by='category', top_k=2)
        legend = ax.get_legend()
        assert len(legend.get_texts()) <= 2


class TestScatterJitter:
    """Test scatter jitter."""
    
    def test_scatter_jitter_bool(self, sample_df):
        """jitter=True adds jitter."""
        plotter = DFDraw(sample_df)
        # First get original data
        fig1, ax1, _ = plotter.scatter('y:x')
        # Then with jitter
        fig2, ax2, _ = plotter.scatter('y:x', jitter=True)
        # Both should work
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
    
    def test_scatter_jitter_float(self, sample_df):
        """jitter=float works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', jitter=0.1)
        assert isinstance(fig, plt.Figure)
    
    def test_scatter_jitter_tuple(self, sample_df):
        """jitter=(x, y) tuple works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', jitter=(0.05, 0.1))
        assert isinstance(fig, plt.Figure)


class TestScatterStatsBox:
    """Test scatter stats box."""
    
    def test_scatter_stats_box_show(self, sample_df):
        """stats=True shows stats box."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', stats=True)
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0
    
    def test_scatter_stats_box_custom(self, sample_df):
        """Custom stats fields work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', stats=['n', 'corr'])
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0


class TestScatterFunctionalAPI:
    """Test functional API for scatter."""
    
    def test_functional_scatter(self, sample_df):
        """Functional scatter() works."""
        fig, ax, stats = scatter(sample_df, 'y:x')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 500
    
    def test_functional_scatter_kwargs(self, sample_df):
        """Functional scatter() accepts kwargs."""
        fig, ax, stats = scatter(sample_df, 'y:x', title='Test', color='blue')
        assert ax.get_title() == 'Test'


class TestScatterComputedExpression:
    """Test scatter with computed expressions."""
    
    def test_scatter_computed_y(self, sample_df):
        """Computed y expression works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.scatter('x + y : x')
        assert stats['n'] == 500
    
    def test_scatter_computed_both(self, sample_df):
        """Computed x and y expressions work."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.scatter('x + y : x * 2')
        assert stats['n'] == 500


class TestScatterExistingAxes:
    """Test scatter on existing axes."""
    
    def test_scatter_on_existing_axes(self, sample_df):
        """Plotting on existing axes works."""
        fig, ax = plt.subplots()
        plotter = DFDraw(sample_df)
        returned_fig, returned_ax, _ = plotter.scatter('y:x', ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig


class TestScatterSampling:
    """Test scatter with sampling."""
    
    def test_scatter_sampling(self, sample_df):
        """Sampling limits points."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.scatter('y:x', sample=100)
        assert stats['n'] <= 100


class TestScatterMarkers:
    """Test scatter marker options."""
    
    def test_scatter_custom_marker(self, sample_df):
        """Custom marker works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', marker='s')
        assert isinstance(fig, plt.Figure)
    
    def test_scatter_alpha(self, sample_df):
        """Alpha parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.scatter('y:x', alpha=0.3)
        assert isinstance(fig, plt.Figure)
