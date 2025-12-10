"""Tests for Phase 7.4: Chain Mode.

Reviewer feedback incorporated:
- Test __file_idx__ NOT in available_branches (GPT)
- Test explicit NaN for missing branches in first/union mode (GPT)
- Test many files for LRU behavior (Claude)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame
from exceptions import BranchNotFoundError, ChainValidationError


class TestChainCreation:
    """Tests for chain creation and file parsing."""
    
    def test_glob_pattern(self, chain_root_files):
        """Create chain from glob pattern."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        assert adf.file_count == 3
        assert adf.is_chain
    
    def test_file_list(self, chain_root_files):
        """Create chain from explicit file list."""
        files = [f'{chain_root_files}/f{i}.root:tree' for i in range(3)]
        adf = AliasDataFrame.read_chain_lazy(files)
        assert adf.file_count == 3
    
    def test_separate_tree_name(self, chain_root_files):
        """Create chain with separate tree name."""
        files = [f'{chain_root_files}/f{i}.root' for i in range(3)]
        adf = AliasDataFrame.read_chain_lazy(files, tree_name='tree')
        assert adf.file_count == 3
    
    def test_empty_glob_raises(self, tmp_path):
        """Empty glob pattern raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AliasDataFrame.read_chain_lazy(f'{tmp_path}/nonexistent_*.root:tree')
    
    def test_missing_tree_raises(self, chain_root_files):
        """Missing tree name raises ValueError."""
        files = [f'{chain_root_files}/f0.root']  # No :tree
        with pytest.raises(ValueError, match="Tree name required"):
            AliasDataFrame.read_chain_lazy(files)
    
    def test_chain_info_property(self, chain_root_files):
        """chain_info returns correct metadata."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        info = adf.chain_info
        assert info is not None
        assert len(info['files']) == 3
        assert info['total_entries'] > 0
        assert len(info['entry_offsets']) == 3
    
    def test_single_file_not_chain(self, chain_root_files):
        """Single file is not considered a chain."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/f0.root:tree')
        assert adf.file_count == 1
        assert not adf.is_chain
    
    def test_sorted_glob_order(self, chain_root_files):
        """Glob results are sorted for reproducibility."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        files = [f['path'] for f in adf.chain_info['files']]
        assert files == sorted(files)
    
    def test_available_branches(self, chain_root_files):
        """available_branches returns correct set."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        branches = adf.available_branches
        assert 'x' in branches
        assert 'y' in branches
        assert 'z' in branches
    
    def test_total_entries_sum(self, chain_root_files):
        """Total entries equals sum of file entries."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        info = adf.chain_info
        # Files have 1000, 2000, 3000 entries = 6000 total
        assert info['total_entries'] == 6000
    
    def test_entry_offsets(self, chain_root_files):
        """Entry offsets are cumulative."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        offsets = adf.chain_info['entry_offsets']
        assert offsets == [0, 1000, 3000]  # 0, 1000, 1000+2000


class TestChainLoading:
    """Tests for branch loading from chains."""
    
    def test_lazy_no_data_initially(self, chain_root_files):
        """Lazy chain has no data initially."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        assert len(adf.df) == 0
        assert len(adf.loaded_branches) == 0
    
    def test_ensure_branches_loads(self, chain_root_files):
        """ensure_branches loads from all files."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        adf.ensure_branches(['x', 'y'])
        
        assert 'x' in adf.loaded_branches
        assert 'y' in adf.loaded_branches
        assert len(adf.df) == adf.chain_info['total_entries']
    
    def test_incremental_loading(self, chain_root_files):
        """Incremental branch loading works correctly."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        adf.ensure_branches(['x'])
        assert adf.loaded_branches == {'x'}
        rows_after_first = len(adf.df)
        
        adf.ensure_branches(['y', 'z'])
        assert adf.loaded_branches == {'x', 'y', 'z'}
        assert len(adf.df) == rows_after_first  # Same row count
    
    def test_draw_auto_loads(self, chain_root_files):
        """draw() auto-loads required branches."""
        matplotlib = pytest.importorskip("matplotlib")
        
        # Skip if dfdraw not available
        try:
            from dfdraw import DFDraw
        except ImportError:
            pytest.skip("dfdraw not available")
        
        import matplotlib.pyplot as plt
        
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        fig, ax, stats = adf.draw('y:x')
        
        assert 'x' in adf.loaded_branches
        assert 'y' in adf.loaded_branches
        plt.close(fig)
    
    def test_missing_branch_raises(self, chain_root_files):
        """Loading missing branch raises BranchNotFoundError."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        with pytest.raises(BranchNotFoundError) as exc_info:
            adf.ensure_branches(['nonexistent'])
        
        assert 'nonexistent' in exc_info.value.missing
    
    def test_file_index_column(self, chain_root_files):
        """add_file_index adds __file_idx__ column."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{chain_root_files}/*.root:tree',
            add_file_index=True
        )
        adf.ensure_branches(['x'])
        
        assert '__file_idx__' in adf.df.columns
        assert set(adf.df['__file_idx__'].unique()) == {0, 1, 2}
    
    def test_file_index_not_in_available_branches(self, chain_root_files):
        """__file_idx__ is NOT in available_branches (per GPT review)."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{chain_root_files}/*.root:tree',
            add_file_index=True
        )
        
        # Before loading
        assert '__file_idx__' not in adf.available_branches
        
        # After loading
        adf.ensure_branches(['x'])
        assert '__file_idx__' not in adf.available_branches
        assert '__file_idx__' in adf.loaded_branches  # But IS in loaded
    
    def test_eager_loads_all(self, chain_root_files):
        """read_chain (eager) loads all data."""
        adf = AliasDataFrame.read_chain(
            f'{chain_root_files}/*.root:tree',
            branches=['x', 'y']
        )
        
        assert 'x' in adf.df.columns
        assert 'y' in adf.df.columns
        assert len(adf.df) > 0
    
    def test_idempotent_ensure_branches(self, chain_root_files):
        """Calling ensure_branches twice is idempotent."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        adf.ensure_branches(['x'])
        x_values = adf.df['x'].copy()
        
        adf.ensure_branches(['x'])  # Same branch again
        
        # Should not change anything
        assert np.array_equal(adf.df['x'].values, x_values.values)
    
    def test_data_correct_after_load(self, chain_root_files):
        """Loaded data has expected properties."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        adf.ensure_branches(['x'])
        
        # Data should be float32
        assert adf.df['x'].dtype == np.float32
        
        # Should have 6000 entries
        assert len(adf.df) == 6000
    
    def test_column_access_auto_loads(self, chain_root_files):
        """adf['column'] auto-loads branch."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        # Access column should trigger load
        _ = adf['x']
        
        assert 'x' in adf.loaded_branches
    
    def test_string_ensure_branches(self, chain_root_files):
        """ensure_branches accepts string as well as list."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        adf.ensure_branches('x')  # String, not list
        
        assert 'x' in adf.loaded_branches


class TestValidationModes:
    """Tests for branch validation modes."""
    
    def test_strict_raises_on_mismatch(self, mismatched_chain_files):
        """strict mode raises on branch mismatch."""
        with pytest.raises(ChainValidationError):
            AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='strict'
            )
    
    def test_first_warns_on_mismatch(self, mismatched_chain_files):
        """first mode warns but continues."""
        with pytest.warns(UserWarning, match="Branch mismatch"):
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='first'
            )
        assert adf.file_count > 0
    
    def test_first_mode_fills_nan_for_missing(self, mismatched_chain_files):
        """first mode fills NaN for missing branches (per GPT review)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='first',
                add_file_index=True
            )
        
        # 'y' is in first file but missing in second file
        adf.ensure_branches(['y'])
        
        # Rows from second file should have NaN for 'y'
        file1_rows = adf.df[adf.df['__file_idx__'] == 1]
        assert file1_rows['y'].isna().all(), "Missing branch should be NaN"
    
    def test_intersection_uses_common(self, mismatched_chain_files):
        """intersection mode uses only common branches."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='intersection'
            )
        # Only branches in ALL files should be available
        assert 'common_branch' in adf.available_branches
        assert 'y' not in adf.available_branches  # Not in all files
    
    def test_union_includes_all(self, mismatched_chain_files):
        """union mode includes all branches."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='union'
            )
        # All branches from any file should be available
        assert 'y' in adf.available_branches  # Only in first file
        assert 'z' in adf.available_branches  # Only in second file
    
    def test_union_mode_fills_nan_for_missing(self, mismatched_chain_files):
        """union mode fills NaN for missing branches (per GPT review)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='union',
                add_file_index=True
            )
        
        # 'z' is only in second file
        adf.ensure_branches(['z'])
        
        # Rows from first file should have NaN for 'z'
        file0_rows = adf.df[adf.df['__file_idx__'] == 0]
        assert file0_rows['z'].isna().all(), "Missing branch should be NaN"
    
    def test_invalid_validation_mode_raises(self, chain_root_files):
        """Invalid validation mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown validation mode"):
            AliasDataFrame.read_chain_lazy(
                f'{chain_root_files}/*.root:tree',
                validate_branches='invalid_mode'
            )
    
    def test_first_mode_extra_branches_ignored(self, mismatched_chain_files):
        """first mode ignores extra branches in later files."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='first'
            )
        
        # 'z' is only in second file, should not be available in 'first' mode
        assert 'z' not in adf.available_branches
    
    def test_intersection_warns(self, mismatched_chain_files):
        """intersection mode warns when branches removed."""
        with pytest.warns(UserWarning, match="Intersection mode"):
            AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='intersection'
            )
    
    def test_union_warns(self, mismatched_chain_files):
        """union mode warns about NaN branches."""
        with pytest.warns(UserWarning, match="Union mode"):
            AliasDataFrame.read_chain_lazy(
                f'{mismatched_chain_files}/*.root:tree',
                validate_branches='union'
            )


class TestEntrySelection:
    """Tests for entry selection across files."""
    
    def test_global_entry_numbering(self, chain_root_files):
        """Entries are numbered 0..N-1 globally."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        adf.ensure_branches(['x'])
        
        assert adf.df.index[0] == 0
        assert adf.df.index[-1] == len(adf.df) - 1
        assert adf.df.index.is_unique
    
    def test_entry_selection_cross_file(self, chain_root_files):
        """Entry selection can span files."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{chain_root_files}/*.root:tree',
            add_file_index=True
        )
        adf.ensure_branches(['x'])
        
        # Select entries spanning file boundary
        offsets = adf.chain_info['entry_offsets']
        if len(offsets) > 1:
            mid = offsets[1]  # Start of second file (1000)
            
            selection = adf.df.iloc[mid-10:mid+10]
            assert len(selection) == 20
            assert len(selection['__file_idx__'].unique()) == 2  # Spans 2 files
    
    def test_get_file_for_entry(self, chain_root_files):
        """get_file_for_entry maps correctly."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        offsets = adf.chain_info['entry_offsets']
        reader = adf._lazy_reader
        
        # First entry of each file
        assert reader.get_file_for_entry(0) == 0
        assert reader.get_file_for_entry(999) == 0  # Last in first file
        assert reader.get_file_for_entry(1000) == 1  # First in second file
        assert reader.get_file_for_entry(2999) == 1  # Last in second file
        assert reader.get_file_for_entry(3000) == 2  # First in third file
    
    def test_entry_out_of_range_raises(self, chain_root_files):
        """Out of range entry raises IndexError."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        reader = adf._lazy_reader
        
        with pytest.raises(IndexError):
            reader.get_file_for_entry(-1)
        
        with pytest.raises(IndexError):
            reader.get_file_for_entry(reader.entries + 1)
    
    def test_file_index_distribution(self, chain_root_files):
        """File index column has correct distribution."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{chain_root_files}/*.root:tree',
            add_file_index=True
        )
        adf.ensure_branches(['x'])
        
        counts = adf.df['__file_idx__'].value_counts().sort_index()
        assert counts[0] == 1000
        assert counts[1] == 2000
        assert counts[2] == 3000


class TestResourceManagement:
    """Tests for resource cleanup."""
    
    def test_context_manager(self, chain_root_files):
        """Context manager closes resources."""
        with AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree') as adf:
            adf.ensure_branches(['x'])
            assert len(adf.df) > 0
        
        # After context, reader should be closed
        assert adf._lazy_reader is None
    
    def test_explicit_close(self, chain_root_files):
        """close() releases resources."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        adf.ensure_branches(['x'])
        adf.close()
        
        assert adf._lazy_reader is None
        assert adf._chain is None
    
    def test_lru_cache_eviction(self, chain_root_files):
        """LRU cache evicts oldest readers."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{chain_root_files}/*.root:tree',
            max_open_files=2
        )
        adf.ensure_branches(['x'])
        
        # With 3 files and max_open=2, one should be evicted
        assert len(adf._lazy_reader._readers) <= 2
    
    def test_many_files_lru(self, many_chain_files):
        """LRU handles many files correctly (per Claude review)."""
        adf = AliasDataFrame.read_chain_lazy(
            f'{many_chain_files}/*.root:tree',
            max_open_files=5
        )
        adf.ensure_branches(['x'])
        
        # With 20 files and max_open=5, should only have 5 handles
        assert len(adf._lazy_reader._readers) <= 5
        assert adf.file_count == 20
    
    def test_close_idempotent(self, chain_root_files):
        """close() can be called multiple times safely."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        adf.close()
        adf.close()  # Should not raise
        assert adf._lazy_reader is None


class TestMemoryEstimation:
    """Tests for memory estimation."""
    
    def test_estimate_memory(self, chain_root_files):
        """estimate_memory returns valid estimate."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        est = adf.estimate_memory(['x', 'y'])
        
        assert 'bytes' in est
        assert 'human' in est
        assert est['bytes'] > 0
        assert est['branches'] == 2
    
    def test_estimate_memory_all_branches(self, chain_root_files):
        """estimate_memory with no args estimates all branches."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        est = adf.estimate_memory()
        
        assert est['branches'] == len(adf.available_branches)
    
    def test_estimate_memory_eager_mode(self, chain_root_files):
        """estimate_memory works in eager mode."""
        adf = AliasDataFrame.read_chain(
            f'{chain_root_files}/*.root:tree',
            branches=['x', 'y']
        )
        
        est = adf.estimate_memory(['x'])
        
        assert est['bytes'] > 0
    
    def test_estimate_human_readable(self, chain_root_files):
        """estimate_memory returns human-readable format."""
        adf = AliasDataFrame.read_chain_lazy(f'{chain_root_files}/*.root:tree')
        
        est = adf.estimate_memory()
        
        # Should contain unit
        assert any(unit in est['human'] for unit in ['B', 'KB', 'MB', 'GB'])


class TestParseChainFiles:
    """Tests for _parse_chain_files helper."""
    
    def test_glob_with_tree(self, chain_root_files):
        """Parse glob:tree format."""
        specs = AliasDataFrame._parse_chain_files(f'{chain_root_files}/*.root:tree')
        assert len(specs) == 3
        assert all(s['tree'] == 'tree' for s in specs)
    
    def test_list_with_tree(self, chain_root_files):
        """Parse list with tree in each entry."""
        files = [f'{chain_root_files}/f0.root:tree', f'{chain_root_files}/f1.root:tree']
        specs = AliasDataFrame._parse_chain_files(files)
        assert len(specs) == 2
    
    def test_list_with_separate_tree(self, chain_root_files):
        """Parse list with separate tree_name."""
        files = [f'{chain_root_files}/f0.root', f'{chain_root_files}/f1.root']
        specs = AliasDataFrame._parse_chain_files(files, tree_name='tree')
        assert len(specs) == 2
        assert all(s['tree'] == 'tree' for s in specs)
    
    def test_single_file(self, chain_root_files):
        """Parse single file."""
        specs = AliasDataFrame._parse_chain_files(f'{chain_root_files}/f0.root:tree')
        assert len(specs) == 1


# ============== Fixtures ==============

@pytest.fixture
def chain_root_files(tmp_path):
    """Create 3 ROOT files for chain testing."""
    uproot = pytest.importorskip("uproot")
    
    for i in range(3):
        file_path = tmp_path / f"f{i}.root"
        n = 1000 * (i + 1)  # Different sizes: 1000, 2000, 3000
        data = {
            'x': np.random.randn(n).astype(np.float32),
            'y': np.random.randn(n).astype(np.float32),
            'z': np.random.randn(n).astype(np.float32),
        }
        with uproot.recreate(file_path) as f:
            f['tree'] = data
    
    return str(tmp_path)


@pytest.fixture
def mismatched_chain_files(tmp_path):
    """Create chain files with mismatched branches."""
    uproot = pytest.importorskip("uproot")
    
    # File 0: x, y, common_branch
    with uproot.recreate(tmp_path / "f0.root") as f:
        f['tree'] = {
            'x': np.random.randn(100).astype(np.float32),
            'y': np.random.randn(100).astype(np.float32),
            'common_branch': np.random.randn(100).astype(np.float32),
        }
    
    # File 1: x, z, common_branch (missing y, has z)
    with uproot.recreate(tmp_path / "f1.root") as f:
        f['tree'] = {
            'x': np.random.randn(100).astype(np.float32),
            'z': np.random.randn(100).astype(np.float32),
            'common_branch': np.random.randn(100).astype(np.float32),
        }
    
    return str(tmp_path)


@pytest.fixture
def many_chain_files(tmp_path):
    """Create 20 ROOT files for LRU testing (per Claude review)."""
    uproot = pytest.importorskip("uproot")
    
    for i in range(20):
        file_path = tmp_path / f"f{i:02d}.root"
        n = 100
        data = {
            'x': np.random.randn(n).astype(np.float32),
        }
        with uproot.recreate(file_path) as f:
            f['tree'] = data
    
    return str(tmp_path)
