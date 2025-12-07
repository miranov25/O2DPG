"""
Tests for histogram plotting in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, hist, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'value': np.random.uniform(0, 100, 1000),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestBasicHistogram:
    """Test basic histogram functionality."""
    
    def test_hist_returns_tuple(self, sample_df):
        """hist() returns (fig, ax, stats_dict)."""
        plotter = DFDraw(sample_df)
        result = plotter.hist('x')
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, stats = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert isinstance(stats, dict)
    
    def test_hist_stats_keys(self, sample_df):
        """Stats dict contains expected keys."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist('x')
        assert 'n' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_hist_stats_values(self, sample_df):
        """Stats values are reasonable."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist('x')
        assert stats['n'] == 1000
        assert -0.5 < stats['mean'] < 0.5  # Should be near 0
        assert 0.8 < stats['std'] < 1.2  # Should be near 1
    
    def test_hist_with_bins(self, sample_df):
        """Histogram respects bins parameter."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', bins=20)
        # Check that histogram was created
        assert len(ax.patches) > 0


class TestHistogramNormalization:
    """Test histogram normalization modes."""
    
    def test_hist_norm_count(self, sample_df):
        """Default normalization is count."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', norm='count')
        # Y-axis should be count
        assert ax.get_ylabel() == 'Count'
    
    def test_hist_norm_density(self, sample_df):
        """Density normalization works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', norm='density')
        assert ax.get_ylabel() == 'Density'
    
    def test_hist_norm_probability(self, sample_df):
        """Probability normalization works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', norm='probability')
        assert ax.get_ylabel() == 'Probability'


class TestHistogramLabels:
    """Test histogram labels and titles."""
    
    def test_hist_xlabel_default(self, sample_df):
        """Default xlabel is column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x')
        assert ax.get_xlabel() == 'x'
    
    def test_hist_xlabel_custom(self, sample_df):
        """Custom xlabel works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', xlabel='My X Label')
        assert ax.get_xlabel() == 'My X Label'
    
    def test_hist_title(self, sample_df):
        """Title parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', title='Test Histogram')
        assert ax.get_title() == 'Test Histogram'


class TestHistogramSelection:
    """Test histogram with selection/cuts."""
    
    def test_hist_with_string_selection(self, sample_df):
        """String query selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist('x', selection='x > 0')
        assert stats['n'] < 1000  # Should have fewer points
        assert stats['mean'] > 0  # Mean should be positive
    
    def test_hist_with_callable_selection(self, sample_df):
        """Callable selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist('x', selection=lambda df: df.x > 0)
        assert stats['n'] < 1000
        assert stats['mean'] > 0


class TestHistogramGroupBy:
    """Test histogram with group_by overlay."""
    
    def test_hist_grouped(self, sample_df):
        """Group-by creates multiple histograms."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist('x', group_by='category')
        assert stats.get('grouped', False)
        # Should have legend
        legend = ax.get_legend()
        assert legend is not None
    
    def test_hist_grouped_top_k(self, sample_df):
        """top_k limits number of groups."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist('x', group_by='category', top_k=2)
        legend = ax.get_legend()
        assert legend is not None
        # Should have at most 2 groups in legend
        assert len(legend.get_texts()) <= 2


class TestHistogramStatsBox:
    """Test histogram stats box."""
    
    def test_hist_stats_box_show(self, sample_df):
        """stats=True shows stats box."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', stats=True)
        # Check for text annotation
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0
    
    def test_hist_stats_box_custom_fields(self, sample_df):
        """Custom stats fields work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', stats=['n', 'mean'])
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0


class TestHistogramFunctionalAPI:
    """Test functional API for histogram."""
    
    def test_functional_hist(self, sample_df):
        """Functional hist() works."""
        fig, ax, stats = hist(sample_df, 'x')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 1000
    
    def test_functional_hist_with_kwargs(self, sample_df):
        """Functional hist() accepts kwargs."""
        fig, ax, stats = hist(sample_df, 'x', bins=20, title='Test')
        assert ax.get_title() == 'Test'


class TestHistogramComputedExpression:
    """Test histogram with computed expressions."""
    
    def test_hist_computed_expression(self, sample_df):
        """Computed expression like 'x + y' works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist('x + y')
        assert stats['n'] == 1000
    
    def test_hist_complex_expression(self, sample_df):
        """Complex expression works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.hist('x * 2 + y')
        assert stats['n'] == 1000


class TestHistogramExistingAxes:
    """Test histogram on existing axes."""
    
    def test_hist_on_existing_axes(self, sample_df):
        """Plotting on existing axes works."""
        fig, ax = plt.subplots()
        plotter = DFDraw(sample_df)
        returned_fig, returned_ax, _ = plotter.hist('x', ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig


class TestHistogramSampling:
    """Test histogram with sampling."""
    
    def test_hist_sampling(self, sample_df):
        """Sampling limits points."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.hist('x', sample=100)
        assert stats['n'] <= 100


class TestHistogramRange:
    """Test histogram with range parameter."""
    
    def test_hist_range(self, sample_df):
        """Range parameter limits x-axis."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', range=(-1, 1))
        xlim = ax.get_xlim()
        # xlim should be approximately (-1, 1)
        assert xlim[0] <= -1
        assert xlim[1] >= 1
