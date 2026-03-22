"""
Tests for Phase 13.12.DF profile enhancements.

Features tested:
- F1: return_data=True → DataFrame export
- F2: min_entries=3 → suppress low-stats bins
- F3: group_by_bins/group_by_quantiles → auto-bin floats
- F4: sort_groups=True → sorted legend order
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
    x = np.random.uniform(0, 10, n)
    y = 2 * x + np.random.normal(0, 1, n)
    group = np.random.choice(['A', 'B', 'C'], n)
    float_group = np.random.uniform(0, 5, n)
    
    return pd.DataFrame({
        'x': x,
        'y': y,
        'group': group,
        'float_group': float_group,
    })


@pytest.fixture
def sparse_df():
    """Create DataFrame with sparse bins for min_entries testing."""
    np.random.seed(42)
    # Most data in center, sparse at edges
    x = np.concatenate([
        np.random.uniform(4, 6, 100),  # Dense center
        np.array([0.5, 1.5, 8.5, 9.5]),  # Sparse edges (1-2 per bin)
    ])
    y = 2 * x + np.random.normal(0, 0.5, len(x))
    
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestReturnData:
    """Tests for F1: return_data parameter."""
    
    def test_profile_return_data_structure(self, sample_df):
        """Verify profile_data DataFrame has correct structure."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            return_data=True,
        )
        
        assert 'profile_data' in stats
        df = stats['profile_data']
        
        # Check columns
        expected_cols = ['x_center', 'x_low', 'x_high', 'y_mean', 'y_std', 'y_sem', 'count']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check shape
        assert len(df) == 10, "Should have 10 bins"
        
        # Check bin edges are consistent
        assert np.allclose(df['x_center'], (df['x_low'] + df['x_high']) / 2)
    
    def test_profile_return_data_grouped(self, sample_df):
        """Verify group column present when group_by used."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='group',
            return_data=True,
        )
        
        assert 'profile_data' in stats
        df = stats['profile_data']
        
        assert 'group' in df.columns
        assert set(df['group'].unique()) == {'A', 'B', 'C'}
    
    def test_profile_return_data_false(self, sample_df):
        """Verify no profile_data when return_data=False."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            return_data=False,
        )
        
        assert 'profile_data' not in stats


class TestMinEntries:
    """Tests for F2: min_entries parameter."""
    
    def test_profile_min_entries_filter(self, sparse_df):
        """Verify bins with n<min_entries excluded from plot."""
        plotter = DFDraw(sparse_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=20,
            range=(0, 10),
            min_entries=3,
            return_data=True,
        )
        
        # Check that sparse bins are in data but not plotted
        df = stats['profile_data']
        sparse_bins = df[df['count'] < 3]
        dense_bins = df[df['count'] >= 3]
        
        assert len(sparse_bins) > 0, "Should have some sparse bins"
        assert len(dense_bins) > 0, "Should have some dense bins"
    
    def test_profile_min_entries_in_data(self, sparse_df):
        """Verify low-count bins still present in profile_data."""
        plotter = DFDraw(sparse_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=20,
            range=(0, 10),
            min_entries=10,  # High threshold
            return_data=True,
        )
        
        df = stats['profile_data']
        
        # All bins should be in data
        assert len(df) == 20
        
        # Some should have low counts
        assert (df['count'] < 10).any()
    
    def test_profile_min_entries_default(self, sample_df):
        """Verify default min_entries=3 works."""
        plotter = DFDraw(sample_df)
        # This should work without error
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
        )
        assert isinstance(fig, plt.Figure)


class TestGroupByBins:
    """Tests for F3: group_by_bins and group_by_quantiles."""
    
    def test_group_by_bins(self, sample_df):
        """Verify pd.cut binning with float column."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='float_group',
            group_by_bins=3,
            return_data=True,
        )
        
        df = stats['profile_data']
        groups = df['group'].unique()
        
        # Should have 3 groups
        assert len(groups) == 3
        
        # Groups should be interval-style labels
        for g in groups:
            assert '-' in str(g), f"Group label should contain '-': {g}"
    
    def test_group_by_quantiles(self, sample_df):
        """Verify pd.qcut binning."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='float_group',
            group_by_quantiles=4,
            return_data=True,
        )
        
        df = stats['profile_data']
        groups = df['group'].unique()
        
        # Should have up to 4 groups (qcut may produce fewer if duplicates)
        assert len(groups) <= 4
    
    def test_group_by_bins_label_format(self, sample_df):
        """Verify custom interval label format (AD-3)."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='float_group',
            group_by_bins=3,
            return_data=True,
        )
        
        df = stats['profile_data']
        groups = df['group'].unique()
        
        # Labels should be 'X.XX-Y.YY' format, not '(X.XX, Y.YY]'
        for g in groups:
            assert '(' not in str(g), f"Should not use pandas interval format: {g}"
            assert ']' not in str(g), f"Should not use pandas interval format: {g}"
    
    def test_group_by_mutual_exclusion(self, sample_df):
        """Verify error when both group_by_bins and group_by_quantiles specified."""
        plotter = DFDraw(sample_df)
        with pytest.raises(ValueError, match="Cannot specify both"):
            plotter.profile(
                'y:x',
                group_by='float_group',
                group_by_bins=3,
                group_by_quantiles=3,
            )


class TestSortGroups:
    """Tests for F4: sort_groups parameter."""
    
    def test_sort_groups_numeric(self):
        """Verify numeric groups sorted."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.uniform(0, 10, 300),
            'y': np.random.normal(0, 1, 300),
            'group': np.tile([3, 1, 2], 100),  # Unsorted order
        })
        
        plotter = DFDraw(df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='group',
            sort_groups=True,
            return_data=True,
        )
        
        # Check legend order
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        
        assert labels == ['1', '2', '3'], f"Expected sorted order, got {labels}"
    
    def test_sort_groups_string(self):
        """Verify string groups sorted alphabetically."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.uniform(0, 10, 300),
            'y': np.random.normal(0, 1, 300),
            'group': np.tile(['C', 'A', 'B'], 100),  # Unsorted order
        })
        
        plotter = DFDraw(df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='group',
            sort_groups=True,
        )
        
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        
        assert labels == ['A', 'B', 'C'], f"Expected sorted order, got {labels}"
    
    def test_sort_groups_false(self):
        """Verify occurrence order preserved when sort_groups=False."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.uniform(0, 10, 300),
            'y': np.random.normal(0, 1, 300),
            'group': np.tile(['C', 'A', 'B'], 100),  # This order
        })
        
        plotter = DFDraw(df)
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='group',
            sort_groups=False,
        )
        
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        
        # Should preserve unique() order: C, A, B
        assert labels == ['C', 'A', 'B'], f"Expected occurrence order, got {labels}"


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_existing_call_unchanged(self, sample_df):
        """Existing calls should produce valid output."""
        plotter = DFDraw(sample_df)
        # Old-style call without new parameters
        fig, ax, stats = plotter.profile(
            'y:x',
            bins=10,
            group_by='group',
        )
        
        assert fig is not None
        assert ax is not None
        assert 'n' in stats
        assert 'profile_data' not in stats  # return_data=False by default
    
    def test_all_original_parameters_work(self, sample_df):
        """All original parameters still work."""
        plotter = DFDraw(sample_df)
        fig, ax, stats = plotter.profile(
            'y:x',
            selection='x > 2',
            sample=500,
            bins=20,
            range=(2, 8),
            error='std',
            stats=True,
            title='Test Profile',
            xlabel='X Axis',
            ylabel='Y Axis',
            group_by='group',
            top_k=2,
        )
        
        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == 'Test Profile'
        assert ax.get_xlabel() == 'X Axis'
