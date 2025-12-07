"""
Tests for profile plotting in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, profile, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame with correlated data."""
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = x * 2 + np.random.randn(n) * 0.5  # Correlated with x
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


class TestBasicProfile:
    """Test basic profile functionality."""
    
    def test_profile_returns_tuple(self, sample_df):
        """profile() returns (fig, ax, stats_dict)."""
        plotter = DFDraw(sample_df)
        result = plotter.profile('y:x')
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, stats = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert isinstance(stats, dict)
    
    def test_profile_stats_keys(self, sample_df):
        """Stats dict contains expected 2D keys."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.profile('y:x')
        assert 'n' in stats
        assert 'mean_x' in stats
        assert 'mean_y' in stats
        assert 'corr' in stats
    
    def test_profile_stats_values(self, sample_df):
        """Stats values are reasonable."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.profile('y:x')
        assert stats['n'] == 1000
        # y = 2*x + noise, so correlation should be high
        assert stats['corr'] > 0.9
    
    def test_profile_requires_2d_expr(self, sample_df):
        """Profile requires 'y:x' format."""
        plotter = DFDraw(sample_df)
        with pytest.raises(ValueError, match="requires 'y:x' format"):
            plotter.profile('x')


class TestProfileBins:
    """Test profile binning."""
    
    def test_profile_bins_parameter(self, sample_df):
        """Bins parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', bins=20)
        assert isinstance(fig, plt.Figure)
    
    def test_profile_range_parameter(self, sample_df):
        """Range parameter limits x binning."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', range=(-1, 1))
        xlim = ax.get_xlim()
        assert xlim[0] <= -1
        assert xlim[1] >= 1


class TestProfileErrorTypes:
    """Test profile error bar types."""
    
    def test_profile_error_sem(self, sample_df):
        """SEM error (default) works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', error='sem')
        assert isinstance(fig, plt.Figure)
    
    def test_profile_error_std(self, sample_df):
        """STD error works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', error='std')
        assert isinstance(fig, plt.Figure)
    
    def test_profile_error_none(self, sample_df):
        """No error bars works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', error='none')
        assert isinstance(fig, plt.Figure)


class TestProfileLabels:
    """Test profile labels and titles."""
    
    def test_profile_xlabel_default(self, sample_df):
        """Default xlabel is x column name."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x')
        assert ax.get_xlabel() == 'x'
    
    def test_profile_ylabel_default(self, sample_df):
        """Default ylabel is <y>."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x')
        assert '<y>' in ax.get_ylabel()
    
    def test_profile_custom_labels(self, sample_df):
        """Custom labels work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', xlabel='X Axis', ylabel='Y Axis')
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
    
    def test_profile_title(self, sample_df):
        """Title parameter works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', title='Test Profile')
        assert ax.get_title() == 'Test Profile'


class TestProfileSelection:
    """Test profile with selection."""
    
    def test_profile_string_selection(self, sample_df):
        """String query selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.profile('y:x', selection='x > 0')
        assert stats['n'] < 1000
        assert stats['mean_x'] > 0
    
    def test_profile_callable_selection(self, sample_df):
        """Callable selection works."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.profile('y:x', selection=lambda df: df.x > 0)
        assert stats['n'] < 1000


class TestProfileGroupBy:
    """Test profile with group_by."""
    
    def test_profile_grouped(self, sample_df):
        """Group-by creates multiple profiles."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile('y:x', group_by='category')
        assert stats.get('grouped', False)
        legend = ax.get_legend()
        assert legend is not None
    
    def test_profile_grouped_top_k(self, sample_df):
        """top_k limits groups."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', group_by='category', top_k=2)
        legend = ax.get_legend()
        assert len(legend.get_texts()) <= 2


class TestProfileStatsBox:
    """Test profile stats box."""
    
    def test_profile_stats_box_show(self, sample_df):
        """stats=True shows stats box."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.profile('y:x', stats=True)
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text) 
                 and 'N =' in child.get_text()]
        assert len(texts) > 0


class TestProfileFunctionalAPI:
    """Test functional API for profile."""
    
    def test_functional_profile(self, sample_df):
        """Functional profile() works."""
        fig, ax, stats = profile(sample_df, 'y:x')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 1000
    
    def test_functional_profile_kwargs(self, sample_df):
        """Functional profile() accepts kwargs."""
        fig, ax, _ = profile(sample_df, 'y:x', bins=20, title='Test')
        assert ax.get_title() == 'Test'


class TestProfileComputedExpression:
    """Test profile with computed expressions."""
    
    def test_profile_computed_y(self, sample_df):
        """Computed y expression works."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile('y + z : x')
        assert stats['n'] == 1000
    
    def test_profile_computed_both(self, sample_df):
        """Computed x and y expressions work."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile('y * 2 : x + z')
        assert stats['n'] == 1000


class TestProfileExistingAxes:
    """Test profile on existing axes."""
    
    def test_profile_on_existing_axes(self, sample_df):
        """Plotting on existing axes works."""
        fig, ax = plt.subplots()
        plotter = DFDraw(sample_df)
        returned_fig, returned_ax, _ = plotter.profile('y:x', ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig


class TestProfileSampling:
    """Test profile with sampling."""
    
    def test_profile_sampling(self, sample_df):
        """Sampling limits points."""
        plotter = DFDraw(sample_df)
        _, _, stats = plotter.profile('y:x', sample=100)
        assert stats['n'] <= 100


class TestProfileDrawDispatch:
    """Test draw() dispatch to profile."""
    
    def test_draw_profile_type(self, sample_df):
        """draw(type='profile') dispatches correctly."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.draw('y:x', type='profile')
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 1000
