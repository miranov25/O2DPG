"""
Tests for Phase 12.4b5: Statistics box and reference overlay methods.

Tests DFDraw.add_statistics_box() and DFDraw.add_reference_overlay().
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestStatisticsBox:
    """Test add_statistics_box() method."""
    
    def test_statistics_box_created(self, sample_df):
        """Verify statistics box is added to axis."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x')
        
        text_artist = plotter.add_statistics_box(ax, sample_df['x'].values)
        
        assert text_artist is not None
        text_content = text_artist.get_text()
        assert 'n =' in text_content
        # Check for statistics (handles both Unicode and ASCII)
        assert '=' in text_content  # Has value assignments
        lines = text_content.split('\n')
        assert len(lines) >= 3  # n, mean, std at minimum
    
    def test_statistics_box_with_expected(self, sample_df):
        """Verify delta values shown when expected provided."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x')
        
        text_artist = plotter.add_statistics_box(
            ax, sample_df['x'].values,
            expected_mean=0.0, expected_std=1.0
        )
        
        assert text_artist is not None
        text_content = text_artist.get_text()
        lines = text_content.split('\n')
        # Should have 5 lines: n, mu, sigma, delta_mu, delta_sigma
        assert len(lines) >= 5
    
    def test_statistics_box_positions(self, sample_df):
        """Verify different positions work."""
        plotter = DFDraw(sample_df)
        
        positions = ['upper right', 'upper left', 'lower right', 'lower left']
        for pos in positions:
            fig, ax, _ = plotter.hist('x')
            text_artist = plotter.add_statistics_box(
                ax, sample_df['x'].values, position=pos
            )
            assert text_artist is not None
            plt.close(fig)
    
    def test_statistics_box_empty_values(self, sample_df):
        """Verify returns None for empty values."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x')
        
        # Empty array
        result = plotter.add_statistics_box(ax, np.array([]))
        assert result is None
        
        # All NaN
        result = plotter.add_statistics_box(ax, np.array([np.nan, np.nan]))
        assert result is None


class TestReferenceOverlay:
    """Test add_reference_overlay() method."""
    
    def test_gaussian_overlay_created(self, sample_df):
        """Verify Gaussian overlay is created."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', bins=50)
        
        line_artist = plotter.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)
        
        assert line_artist is not None
    
    def test_overlay_has_label(self, sample_df):
        """Verify overlay has legend label."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', bins=50)
        
        line_artist = plotter.add_reference_overlay(
            ax, func='gaussian', mu=0, sigma=1, show_legend=True
        )
        
        assert line_artist is not None
        assert line_artist.get_label() is not None
    
    def test_custom_callable_overlay(self, sample_df):
        """Verify custom callable function works."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', bins=50)
        
        # Custom exponential function
        def custom_func(x):
            return np.exp(-np.abs(x))
        
        line_artist = plotter.add_reference_overlay(
            ax, func=custom_func, label='Custom'
        )
        
        assert line_artist is not None
        assert line_artist.get_label() == 'Custom'
    
    def test_overlay_returns_none_for_empty_axis(self, sample_df):
        """Verify returns None if no histogram patches."""
        plotter = DFDraw(sample_df)
        fig, ax = plt.subplots()  # Empty axis
        
        result = plotter.add_reference_overlay(ax, func='gaussian')
        
        assert result is None
    
    def test_overlay_custom_styling(self, sample_df):
        """Verify custom styling parameters work."""
        plotter = DFDraw(sample_df)
        fig, ax, _ = plotter.hist('x', bins=50)
        
        line_artist = plotter.add_reference_overlay(
            ax, func='gaussian',
            color='blue', linestyle=':', linewidth=2.0
        )
        
        assert line_artist is not None
        assert line_artist.get_color() == 'blue'
        assert line_artist.get_linestyle() == ':'
        assert line_artist.get_linewidth() == 2.0


class TestCombinedUsage:
    """Test combined usage of statistics box and overlay."""
    
    def test_pull_distribution_workflow(self, sample_df):
        """Test typical pull distribution QA workflow."""
        plotter = DFDraw(sample_df)
        
        # Create histogram
        fig, ax, stats = plotter.hist('x', bins=50, title='Pull Distribution')
        
        # Add reference Gaussian
        plotter.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)
        
        # Add statistics box with expected values
        plotter.add_statistics_box(
            ax, sample_df['x'].values,
            expected_mean=0.0, expected_std=1.0,
            position='upper right'
        )
        
        # Verify both elements present
        assert len(ax.lines) > 0  # Reference curve
        assert len(ax.texts) > 0  # Statistics box
