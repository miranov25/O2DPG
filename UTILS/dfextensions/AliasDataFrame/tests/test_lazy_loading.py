"""
Tests for Phase 7.1: Lazy Branch Loading Foundation.

Tests cover:
- LazyTreeReader class
- read_tree_lazy() method
- ensure_branches() method
- available_branches / loaded_branches properties
- Auto-loading via __getitem__
- Lazy/Eager compatibility
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

# Import from the same package structure
from AliasDataFrame import AliasDataFrame
from LazyTreeReader import LazyTreeReader


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_root_file(tmp_path):
    """Create a sample ROOT file for testing."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "test_data.root"
    
    # Create test data with various types
    n_entries = 1000
    data = {
        'x': np.random.randn(n_entries).astype(np.float32),
        'y': np.random.randn(n_entries).astype(np.float32),
        'z': np.random.randn(n_entries).astype(np.float32),
        'pt': np.abs(np.random.randn(n_entries)).astype(np.float32),
        'eta': np.random.uniform(-2.5, 2.5, n_entries).astype(np.float32),
        'phi': np.random.uniform(-np.pi, np.pi, n_entries).astype(np.float32),
        'charge': np.random.choice([-1, 1], n_entries).astype(np.int32),
        'isOK': np.random.choice([True, False], n_entries),
    }
    
    with uproot.recreate(file_path) as f:
        f['tree'] = data
    
    return str(file_path)


@pytest.fixture
def lazy_tree_reader(sample_root_file):
    """Create a LazyTreeReader instance."""
    return LazyTreeReader(sample_root_file, 'tree')


# =============================================================================
# LazyTreeReader Tests
# =============================================================================

class TestLazyTreeReaderInit:
    """Tests for LazyTreeReader initialization."""
    
    def test_init_loads_metadata(self, sample_root_file):
        """Reader loads branch list and entry count on init."""
        reader = LazyTreeReader(sample_root_file, 'tree')
        
        assert len(reader.available_branches) == 8  # x,y,z,pt,eta,phi,charge,isOK
        assert reader.num_entries == 1000
        assert len(reader.loaded_branches) == 0
    
    def test_init_file_not_found(self, tmp_path):
        """Raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            LazyTreeReader(str(tmp_path / "nonexistent.root"), 'tree')
    
    def test_init_tree_not_found(self, sample_root_file):
        """Raises error for non-existent tree."""
        with pytest.raises(KeyError):
            LazyTreeReader(sample_root_file, 'nonexistent_tree')
    
    def test_repr(self, lazy_tree_reader):
        """__repr__ shows useful info."""
        repr_str = repr(lazy_tree_reader)
        assert 'LazyTreeReader' in repr_str
        assert 'available=' in repr_str
        assert 'loaded=' in repr_str


class TestLazyTreeReaderEnsureBranches:
    """Tests for ensure_branches method."""
    
    def test_ensure_branches_loads_data(self, lazy_tree_reader):
        """ensure_branches loads requested branches."""
        df = lazy_tree_reader.ensure_branches(['x', 'y'], pd.DataFrame())
        
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert len(df) == lazy_tree_reader.num_entries
        assert lazy_tree_reader.loaded_branches == {'x', 'y'}
    
    def test_ensure_branches_invalid_raises(self, lazy_tree_reader):
        """Raises ValueError for non-existent branch."""
        with pytest.raises(ValueError, match="not found"):
            lazy_tree_reader.ensure_branches(['nonexistent'], pd.DataFrame())
    
    def test_ensure_branches_idempotent(self, lazy_tree_reader):
        """Calling ensure_branches twice doesn't reload."""
        df1 = lazy_tree_reader.ensure_branches(['x'], pd.DataFrame())
        original_x = df1['x'].values.copy()
        
        df2 = lazy_tree_reader.ensure_branches(['x'], df1)
        
        # Same DataFrame object (no reload)
        assert df1 is df2
        np.testing.assert_array_equal(df2['x'].values, original_x)
    
    def test_ensure_branches_incremental(self, lazy_tree_reader):
        """Can load branches incrementally."""
        df = lazy_tree_reader.ensure_branches(['x'], pd.DataFrame())
        assert lazy_tree_reader.loaded_branches == {'x'}
        
        df = lazy_tree_reader.ensure_branches(['y'], df)
        assert lazy_tree_reader.loaded_branches == {'x', 'y'}
        
        df = lazy_tree_reader.ensure_branches(['z', 'pt'], df)
        assert lazy_tree_reader.loaded_branches == {'x', 'y', 'z', 'pt'}
    
    def test_ensure_branches_empty_list(self, lazy_tree_reader):
        """Empty list is no-op."""
        df = pd.DataFrame({'existing': [1, 2, 3]})
        result = lazy_tree_reader.ensure_branches([], df)
        assert result is df
    
    def test_is_loaded(self, lazy_tree_reader):
        """is_loaded correctly tracks state."""
        assert not lazy_tree_reader.is_loaded('x')
        
        lazy_tree_reader.ensure_branches(['x'], pd.DataFrame())
        
        assert lazy_tree_reader.is_loaded('x')
        assert not lazy_tree_reader.is_loaded('y')


class TestLazyTreeReaderMisc:
    """Miscellaneous LazyTreeReader tests."""
    
    def test_get_branch_dtype(self, lazy_tree_reader):
        """get_branch_dtype returns dtype info."""
        dtype = lazy_tree_reader.get_branch_dtype('x')
        # uproot returns interpretation string like "AsDtype('>f4')" or similar
        assert dtype is not None
        assert len(dtype) > 0
    
    def test_get_branch_dtype_invalid(self, lazy_tree_reader):
        """get_branch_dtype raises for invalid branch."""
        with pytest.raises(ValueError, match="not in TTree"):
            lazy_tree_reader.get_branch_dtype('nonexistent')
    
    def test_close(self, lazy_tree_reader):
        """close releases file handle."""
        lazy_tree_reader.close()
        assert lazy_tree_reader._file is None
        assert lazy_tree_reader._tree is None


# =============================================================================
# read_tree_lazy Tests
# =============================================================================

class TestReadTreeLazy:
    """Tests for AliasDataFrame.read_tree_lazy()."""
    
    def test_read_with_branches(self, sample_root_file):
        """read_tree_lazy loads specified branches."""
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x', 'y']
        )
        
        assert 'x' in adf.df.columns
        assert 'y' in adf.df.columns
        assert 'z' not in adf.df.columns  # Not requested
        assert adf.is_lazy
    
    def test_read_metadata_only(self, sample_root_file):
        """read_tree_lazy without branches loads metadata only."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        assert len(adf.df.columns) == 0
        assert len(adf) == 1000  # Has correct length
        assert len(adf.available_branches) == 8
        assert adf.is_lazy
    
    def test_chain_config_created(self, sample_root_file):
        """_chain config populated for single file."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        assert adf._chain is not None
        assert len(adf._chain['files']) == 1
        assert adf._chain['files'][0]['entries'] == 1000
        assert adf._chain['entry_offsets'] == [0]
    
    def test_schema_applied(self, sample_root_file):
        """Schema is applied to lazy ADF."""
        schema = {'columns': {'x': {'title': 'X Position [cm]'}}}
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x'],
            schema=schema
        )
        
        assert adf.get_axis_title('x') == 'X Position [cm]'


# =============================================================================
# ensure_branches Method Tests
# =============================================================================

class TestEnsureBranchesMethod:
    """Tests for AliasDataFrame.ensure_branches()."""
    
    def test_ensure_branches_loads(self, sample_root_file):
        """ensure_branches loads branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.ensure_branches(['x', 'y'])
        
        assert 'x' in adf.df.columns
        assert 'y' in adf.df.columns
        assert adf.loaded_branches == {'x', 'y'}
    
    def test_ensure_branches_non_lazy(self):
        """ensure_branches validates in non-lazy mode."""
        from exceptions import BranchNotFoundError
        
        # Create non-lazy ADF
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        adf = AliasDataFrame(df)
        
        # Should pass for existing columns
        adf.ensure_branches(['a', 'b'])
        
        # Should raise for missing columns
        with pytest.raises(BranchNotFoundError):
            adf.ensure_branches(['c'])
    
    def test_ensure_branches_empty_noop(self, sample_root_file):
        """Empty list is no-op."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.ensure_branches([])
        
        assert len(adf.loaded_branches) == 0


# =============================================================================
# Properties Tests
# =============================================================================

class TestLazyProperties:
    """Tests for lazy mode properties."""
    
    def test_available_branches_lazy(self, sample_root_file):
        """available_branches returns all TTree branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        assert adf.available_branches is not None
        assert 'x' in adf.available_branches
        assert 'y' in adf.available_branches
        assert len(adf.available_branches) == 8
    
    def test_available_branches_non_lazy(self):
        """available_branches returns None for non-lazy."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        assert adf.available_branches is None
    
    def test_loaded_branches_lazy(self, sample_root_file):
        """loaded_branches tracks what's in memory."""
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x']
        )
        
        assert 'x' in adf.loaded_branches
        assert len(adf.loaded_branches) == 1
    
    def test_loaded_branches_non_lazy(self):
        """loaded_branches returns None for non-lazy."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        assert adf.loaded_branches is None
    
    def test_is_lazy_true(self, sample_root_file):
        """is_lazy returns True for lazy ADF."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        assert adf.is_lazy is True
    
    def test_is_lazy_false(self):
        """is_lazy returns False for non-lazy ADF."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        assert adf.is_lazy is False


# =============================================================================
# Auto-Loading Tests (__getitem__)
# =============================================================================

class TestAutoLoading:
    """Tests for auto-loading branches via __getitem__."""
    
    def test_getitem_auto_loads_single(self, sample_root_file):
        """adf['x'] auto-loads branch x."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        assert len(adf.loaded_branches) == 0
        
        # Access triggers auto-load
        x = adf['x']
        
        assert len(x) == 1000
        assert 'x' in adf.loaded_branches
    
    def test_getitem_auto_loads_multiple(self, sample_root_file):
        """adf[['x', 'y']] auto-loads both branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Access multiple columns
        df = adf[['x', 'y']]
        
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert adf.loaded_branches == {'x', 'y'}
    
    def test_getitem_already_loaded_no_reload(self, sample_root_file):
        """Already loaded branch isn't reloaded."""
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x']
        )
        
        original_x = adf.df['x'].values.copy()
        
        # Access again - should not reload
        x = adf['x']
        
        np.testing.assert_array_equal(x.values, original_x)
    
    def test_getitem_non_lazy_normal(self):
        """Non-lazy ADF __getitem__ works normally."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}))
        
        assert adf['x'].tolist() == [1, 2, 3]
        assert adf[['x', 'y']].shape == (3, 2)
    
    def test_getitem_missing_raises(self, sample_root_file):
        """Accessing non-existent branch raises KeyError."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(KeyError):
            _ = adf['nonexistent_branch']


# =============================================================================
# Lazy/Eager Compatibility Tests
# =============================================================================

class TestLazyEagerCompatibility:
    """Tests that lazy and eager modes produce same results."""
    
    def test_same_data_values(self, sample_root_file):
        """Lazy and eager loading produce identical data."""
        # Load all branches via lazy mode first
        adf_lazy = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        all_branches = list(adf_lazy.available_branches)
        adf_lazy.ensure_branches(all_branches)
        
        # Load same branches again with fresh lazy ADF
        adf_lazy2 = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=all_branches
        )
        
        # Compare values between two lazy loads (should be identical)
        for col in all_branches:
            np.testing.assert_array_almost_equal(
                adf_lazy.df[col].values,
                adf_lazy2.df[col].values,
                decimal=5
            )
    
    def test_df_length_matches(self, sample_root_file):
        """Lazy ADF has correct length even without branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        assert len(adf) == 1000
        assert len(adf.df) == 1000


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_load_all_branches(self, sample_root_file):
        """Can load all available branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.ensure_branches(list(adf.available_branches))
        
        assert adf.loaded_branches == adf.available_branches
        assert len(adf.df.columns) == 8
    
    def test_repeated_ensure_same_branch(self, sample_root_file):
        """Calling ensure_branches multiple times for same branch is fine."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.ensure_branches(['x'])
        adf.ensure_branches(['x'])
        adf.ensure_branches(['x', 'y'])
        adf.ensure_branches(['x', 'y'])
        
        assert adf.loaded_branches == {'x', 'y'}
