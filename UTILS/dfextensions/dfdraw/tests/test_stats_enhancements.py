"""
Tests for Phase 13.6.G.DF: dfdraw Statistics Enhancements

Test coverage:
- P0.1: ddof breaking change (population std)
- P0.2: 2D n counts both-valid pairs
- Issue 1: Default stats fields adapt to plot type
- Issue 2: Range-aware stats computation
- Issue 3: Robust statistics (median, quartiles, MAD)
"""

import pytest
import numpy as np
import pandas as pd

# Proper package imports (no sys.path manipulation)
from dfdraw.stats import (
    compute_stats,
    format_stats_box,
    get_default_stats_fields,
)

# Helper to test internal stats computation through public API
def _compute_single_stats_via_public_api(df, y_col, x_col=None, **kwargs):
    """Test helper: compute stats for single group via public API."""
    result = compute_stats(df, y_col, x_col, group_by=None, **kwargs)
    return result.iloc[0].to_dict()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame with random data."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000),
    })


@pytest.fixture
def linear_df():
    """Create DataFrame with known linear values for range testing."""
    return pd.DataFrame({
        'x': np.linspace(-5, 5, 1001),
        'y': np.linspace(-5, 5, 1001),
    })


@pytest.fixture
def independent_df():
    """Create DataFrame with independent x and y for correlation tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000),
    })


@pytest.fixture
def known_values_df():
    """Create DataFrame with known values for exact assertions."""
    return pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [2.0, 4.0, 6.0, 8.0, 10.0],
    })


# =============================================================================
# P0.1: Breaking Change - ddof
# =============================================================================

class TestBreakingChange_ddof:
    """Tests for P0.1: Verify std uses population (ddof=0)."""
    
    def test_std_uses_population_ddof(self, known_values_df):
        """Verify std uses population (ddof=0), not sample (ddof=1)."""
        stats = _compute_single_stats_via_public_api(known_values_df, 'x')
        
        # Population std (ddof=0) for [1,2,3,4,5]
        expected_std = np.std([1, 2, 3, 4, 5], ddof=0)  # ~1.4142
        assert np.isclose(stats['std'], expected_std), \
            f"Expected population std {expected_std}, got {stats['std']}"
        
        # NOT sample std (ddof=1)
        sample_std = np.std([1, 2, 3, 4, 5], ddof=1)  # ~1.5811
        assert not np.isclose(stats['std'], sample_std), \
            "std should NOT use sample std (ddof=1)"
    
    def test_std_x_uses_population_ddof(self, sample_df):
        """Verify std_x uses population (ddof=0)."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        
        # Compute expected population std
        expected_std_x = np.std(sample_df['x'].dropna(), ddof=0)
        assert np.isclose(stats['std_x'], expected_std_x, rtol=0.01)
    
    def test_std_y_uses_population_ddof(self, sample_df):
        """Verify std_y uses population (ddof=0)."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        
        # Compute expected population std
        expected_std_y = np.std(sample_df['y'].dropna(), ddof=0)
        assert np.isclose(stats['std_y'], expected_std_y, rtol=0.01)
    
    def test_std_population_vs_sample_difference(self):
        """Verify there's a measurable difference between ddof=0 and ddof=1."""
        # For small samples, difference is significant
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        stats = _compute_single_stats_via_public_api(df, 'x')
        
        pop_std = np.std([1, 2, 3], ddof=0)   # sqrt(2/3) ≈ 0.8165
        sample_std = np.std([1, 2, 3], ddof=1)  # 1.0
        
        assert np.isclose(stats['std'], pop_std)
        assert abs(pop_std - sample_std) > 0.1  # Difference should be noticeable


# =============================================================================
# P0.2: 2D n Semantics
# =============================================================================

class TestIssue2_2D_n_Semantics:
    """Tests for P0.2: Verify 2D n counts both-valid pairs."""
    
    def test_2d_n_counts_both_valid(self):
        """2D n should count pairs where both x and y are valid."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, np.nan, 4.0, 5.0],
            'y': [1.0, np.nan, 3.0, 4.0, 5.0],
        })
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        # Only 3 rows have both x and y valid: (1,1), (4,4), (5,5)
        assert stats['n'] == 3
    
    def test_2d_n_with_all_valid(self, known_values_df):
        """2D n equals row count when all values are valid."""
        stats = _compute_single_stats_via_public_api(known_values_df, 'y', 'x')
        assert stats['n'] == 5
    
    def test_2d_n_with_range_counts_both_valid_in_range(self):
        """2D n with range counts pairs where both are valid AND in range."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        # Range excludes x=1 and y=5
        stats = _compute_single_stats_via_public_api(df, 'y', 'x', 
                                      range_x=(2, 4), range_y=(2, 4))
        
        # Only (2,2), (3,3), (4,4) are in range
        assert stats['n'] == 3


# =============================================================================
# Issue 1: Default Fields Adapt to Plot Type
# =============================================================================

class TestIssue1_DefaultFieldsAdapt:
    """Tests for Issue 1: Default stats display adapts to plot type."""
    
    def test_format_stats_box_hist_defaults(self, sample_df):
        """1D hist default fields are n, mean, std."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x')
        formatted = format_stats_box(stats, plot_type='hist')
        
        assert 'N =' in formatted
        assert 'mean' in formatted
        assert 'std' in formatted
        assert 'mean_x' not in formatted
    
    def test_format_stats_box_hist2d_defaults(self, sample_df):
        """hist2d default fields include 2D stats and corr."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats, plot_type='hist2d')
        
        assert 'N =' in formatted
        assert 'mean_x' in formatted
        assert 'mean_y' in formatted
        assert 'std_x' in formatted
        assert 'std_y' in formatted
        assert 'corr' in formatted
        # Should NOT contain ambiguous 1D field
        assert 'mean =' not in formatted
    
    def test_format_stats_box_scatter_defaults(self, sample_df):
        """scatter default fields are n, mean_x, mean_y only (no corr)."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats, plot_type='scatter')
        
        assert 'N =' in formatted
        assert 'mean_x' in formatted
        assert 'mean_y' in formatted
        # Scatter default should NOT include corr
        assert 'corr' not in formatted
    
    def test_format_stats_box_profile_defaults(self, sample_df):
        """profile default fields are n, mean_x, mean_y only."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats, plot_type='profile')
        
        assert 'N =' in formatted
        assert 'mean_x' in formatted
        assert 'mean_y' in formatted
        assert 'corr' not in formatted
    
    def test_format_stats_box_hexbin_defaults(self, sample_df):
        """hexbin default fields are n, mean_x, mean_y only."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats, plot_type='hexbin')
        
        assert 'N =' in formatted
        assert 'mean_x' in formatted
        assert 'mean_y' in formatted
        assert 'corr' not in formatted
    
    def test_explicit_fields_override_defaults(self, sample_df):
        """Explicit fields override plot_type defaults."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats, fields=['n', 'corr'], plot_type='hist2d')
        
        assert 'N =' in formatted
        assert 'corr' in formatted
        assert 'mean_x' not in formatted
        assert 'std_x' not in formatted
    
    def test_format_stats_box_fallback_2d_detection(self, sample_df):
        """Without plot_type, detect 2D from stats keys."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x')
        formatted = format_stats_box(stats)  # No plot_type
        
        # Should detect 2D from presence of mean_x
        assert 'mean_x' in formatted
        assert 'corr' in formatted
    
    def test_format_stats_box_fallback_1d_detection(self, sample_df):
        """Without plot_type, detect 1D from stats keys."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x')
        formatted = format_stats_box(stats)  # No plot_type
        
        # Should detect 1D (no mean_x)
        assert 'mean' in formatted
        assert 'mean_x' not in formatted


# =============================================================================
# Issue 2: Range-Aware Stats
# =============================================================================

class TestIssue2_RangeAwareStats:
    """Tests for Issue 2: Statistics respect range parameter."""
    
    def test_1d_stats_respect_range(self, linear_df):
        """1D stats computed within range only."""
        # Full range
        stats_full = _compute_single_stats_via_public_api(linear_df, 'x')
        
        # Restricted to positive only [0, 5]
        stats_range = _compute_single_stats_via_public_api(linear_df, 'x', range_x=(0, 5))
        
        # Mean should be positive for positive range
        assert stats_range['mean'] > 0
        assert stats_range['mean'] > stats_full['mean']
        # Should have approximately half the data
        assert 400 < stats_range['n'] < 600
    
    def test_2d_stats_respect_range(self, linear_df):
        """2D stats computed within 2D range only."""
        # Restricted to positive quadrant
        # Note: linear_df has y=x, so filtering [0,5] on both axes
        # gives the same ~501 points (half the data)
        stats = _compute_single_stats_via_public_api(linear_df, 'y', 'x',
                                      range_x=(0, 5), range_y=(0, 5))
        
        # Means should be positive
        assert stats['mean_x'] > 2.0
        assert stats['mean_y'] > 2.0
        # For linear_df where y=x, should have approximately half the data
        assert 450 < stats['n'] < 550
    
    def test_range_x_only_for_2d(self):
        """2D stats with only range_x applied."""
        # Use independent data so y is NOT filtered when we filter x
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.linspace(-5, 5, 1001),
            'y': np.random.randn(1001),  # Independent of x
        })
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x', range_x=(0, 5))
        
        assert stats['mean_x'] > 0
        # y is not filtered directly, but rows are filtered by x range
        # so we get the y values corresponding to x in [0,5]
        # With independent random y, mean should still be near 0
        assert -0.5 < stats['mean_y'] < 0.5
    
    def test_range_y_only_for_2d(self, linear_df):
        """2D stats with only range_y applied."""
        stats = _compute_single_stats_via_public_api(linear_df, 'y', 'x', range_y=(0, 5))
        
        assert stats['mean_y'] > 0
        # For this linear case where y=x, filtering y also affects x
        assert stats['mean_x'] > 0
    
    def test_empty_range_1d(self, sample_df):
        """Empty range returns n=0 and NaN for 1D fields."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x', range_x=(100, 200))
        
        assert stats['n'] == 0
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['std'])
        assert np.isnan(stats['min'])
        assert np.isnan(stats['max'])
    
    def test_empty_range_2d(self, sample_df):
        """Empty 2D range returns n=0 and NaN for all 2D fields."""
        stats = _compute_single_stats_via_public_api(sample_df, 'y', 'x',
                                      range_x=(100, 200), range_y=(100, 200))
        
        assert stats['n'] == 0
        assert np.isnan(stats['mean_x'])
        assert np.isnan(stats['mean_y'])
        assert np.isnan(stats['std_x'])
        assert np.isnan(stats['std_y'])
        assert np.isnan(stats['corr'])
    
    def test_range_boundaries_inclusive(self):
        """Range boundaries are inclusive [min, max]."""
        df = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'x', range_x=(1.0, 2.0))
        
        assert stats['n'] == 2  # Includes both 1.0 and 2.0
        assert stats['mean'] == 1.5
    
    def test_range_with_group_by(self, linear_df):
        """Range filtering works with group_by."""
        # Add a category column
        df = linear_df.copy()
        df['cat'] = ['A', 'B'] * (len(df) // 2) + ['A'] * (len(df) % 2)
        
        result = compute_stats(df, 'x', group_by='cat', range_x=(0, 5))
        
        # Both groups should have positive means
        for _, row in result.iterrows():
            assert row['mean'] > 0


# =============================================================================
# Issue 3: Robust Statistics
# =============================================================================

class TestIssue3_RobustStats:
    """Tests for Issue 3: Robust statistics (median, quartiles, MAD)."""
    
    def test_median_computed_when_robust(self, sample_df):
        """Median field available when robust=True."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x', robust=True)
        
        assert 'median' in stats
        assert not np.isnan(stats['median'])
    
    def test_quartiles_computed_when_robust(self, sample_df):
        """Quartile fields available when robust=True."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x', robust=True)
        
        assert 'q25' in stats
        assert 'q75' in stats
        assert stats['q25'] < stats['q75']
    
    def test_mad_computed_when_robust(self, sample_df):
        """MAD field available when robust=True."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x', robust=True)
        
        assert 'mad' in stats
        assert stats['mad'] > 0
    
    def test_robust_values_correct(self):
        """Verify robust stats match expected values."""
        # Known data: [1, 2, 3, 4, 5]
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        stats = _compute_single_stats_via_public_api(df, 'x', robust=True)
        
        assert stats['median'] == 3.0
        assert stats['q25'] == 2.0
        assert stats['q75'] == 4.0
        # MAD = median(|x - 3|) = median([2, 1, 0, 1, 2]) = 1
        assert stats['mad'] == 1.0
    
    def test_robust_with_range(self, linear_df):
        """Robust stats work with range filtering."""
        stats = _compute_single_stats_via_public_api(linear_df, 'x', 
                                      range_x=(0, 5), robust=True)
        
        assert 400 < stats['n'] < 600
        assert stats['median'] > 0  # Positive range
        assert stats['q25'] > 0
        assert stats['q75'] > 0
    
    def test_robust_2d_applies_to_y_only(self, known_values_df):
        """For 2D plots, robust stats apply to y-axis only."""
        stats = _compute_single_stats_via_public_api(known_values_df, 'y', 'x', robust=True)
        
        # median should be median of y values [2,4,6,8,10] = 6
        assert stats['median'] == 6.0
        assert stats['q25'] == 4.0
        assert stats['q75'] == 8.0
    
    def test_robust_empty_range(self, sample_df):
        """Robust stats return NaN for empty range."""
        stats = _compute_single_stats_via_public_api(sample_df, 'x', 
                                      range_x=(100, 200), robust=True)
        
        assert stats['n'] == 0
        assert np.isnan(stats['median'])
        assert np.isnan(stats['q25'])
        assert np.isnan(stats['q75'])
        assert np.isnan(stats['mad'])
    
    def test_get_default_stats_fields_robust(self):
        """get_default_stats_fields returns robust defaults for 1D."""
        fields = get_default_stats_fields('hist', robust=True)
        assert fields == ['n', 'median', 'mad']
    
    def test_get_default_stats_fields_robust_2d_unchanged(self):
        """get_default_stats_fields: robust mode doesn't change 2D defaults."""
        fields_normal = get_default_stats_fields('hist2d', robust=False)
        fields_robust = get_default_stats_fields('hist2d', robust=True)
        
        # 2D defaults unchanged by robust mode
        assert fields_normal == fields_robust


# =============================================================================
# Edge Cases
# =============================================================================

class TestStatsEdgeCases:
    """Edge case tests for statistics computation."""
    
    def test_single_point_1d(self):
        """Stats with single data point (1D)."""
        df = pd.DataFrame({'x': [5.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'x')
        
        assert stats['n'] == 1
        assert stats['mean'] == 5.0
        assert stats['std'] == 0.0  # Population std of single point
    
    def test_single_point_2d(self):
        """Stats with single data point (2D)."""
        df = pd.DataFrame({'x': [5.0], 'y': [3.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        assert stats['n'] == 1
        assert stats['mean_x'] == 5.0
        assert stats['mean_y'] == 3.0
        assert stats['std_x'] == 0.0
        assert stats['std_y'] == 0.0
    
    def test_correlation_single_point(self):
        """Correlation undefined for single point."""
        df = pd.DataFrame({'x': [5.0], 'y': [3.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        assert stats['n'] == 1
        assert np.isnan(stats['corr'])
    
    def test_correlation_two_points(self):
        """Correlation defined for two points."""
        df = pd.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        assert stats['n'] == 2
        assert np.isclose(stats['corr'], 1.0)  # Perfect correlation
    
    def test_negative_correlation(self):
        """Negative correlation computed correctly."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [3.0, 2.0, 1.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        assert np.isclose(stats['corr'], -1.0)  # Perfect negative correlation
    
    def test_nan_values_excluded_1d(self, sample_df):
        """NaN values excluded from 1D stats computation."""
        df = sample_df.copy()
        df.loc[0:99, 'x'] = np.nan  # 100 NaN values
        
        stats = _compute_single_stats_via_public_api(df, 'x')
        
        assert stats['n'] == 900  # 1000 - 100 NaN
    
    def test_nan_values_excluded_2d(self):
        """NaN values excluded from 2D stats (both-valid counting)."""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0, 4.0, np.nan],
            'y': [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        
        stats = _compute_single_stats_via_public_api(df, 'y', 'x')
        
        # Only rows 0 and 3 have both valid: (1,1) and (4,4)
        assert stats['n'] == 2
    
    def test_empty_dataframe(self):
        """Empty DataFrame returns n=0 and NaN fields."""
        df = pd.DataFrame({'x': []})
        
        stats = _compute_single_stats_via_public_api(df, 'x')
        
        assert stats['n'] == 0
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['std'])
    
    def test_all_nan_values(self):
        """All-NaN column returns n=0 and NaN fields."""
        df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        
        stats = _compute_single_stats_via_public_api(df, 'x')
        
        assert stats['n'] == 0
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['std'])
    
    def test_expression_evaluation(self):
        """Stats work with computed expressions."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 30.0]})
        
        stats = _compute_single_stats_via_public_api(df, 'a + b')
        
        # a + b = [11, 22, 33]
        assert stats['n'] == 3
        assert stats['mean'] == 22.0
    
    def test_format_stats_box_nan_display(self):
        """NaN values display as 'N/A' in stats box."""
        stats = {'n': 0, 'mean': np.nan, 'std': np.nan}
        
        formatted = format_stats_box(stats, plot_type='hist')
        
        assert 'N = 0' in formatted
        assert 'mean = N/A' in formatted
        assert 'std = N/A' in formatted


# =============================================================================
# Compute Stats Function (High-Level)
# =============================================================================

class TestComputeStats:
    """Tests for the high-level compute_stats function."""
    
    def test_returns_dataframe(self, sample_df):
        """compute_stats returns a DataFrame."""
        result = compute_stats(sample_df, 'x')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_group_by_returns_multiple_rows(self, sample_df):
        """compute_stats with group_by returns multiple rows."""
        df = sample_df.copy()
        df['cat'] = ['A', 'B'] * 500
        
        result = compute_stats(df, 'x', group_by='cat')
        
        assert len(result) == 2
        assert 'group' in result.columns
        assert set(result['group']) == {'A', 'B'}
    
    def test_range_passed_through(self, linear_df):
        """compute_stats passes range parameters through."""
        result = compute_stats(linear_df, 'x', range_x=(0, 5))
        
        stats = result.iloc[0]
        assert stats['mean'] > 0
        assert 400 < stats['n'] < 600
    
    def test_robust_passed_through(self, sample_df):
        """compute_stats passes robust parameter through."""
        result = compute_stats(sample_df, 'x', robust=True)
        
        assert 'median' in result.columns
        assert 'mad' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
