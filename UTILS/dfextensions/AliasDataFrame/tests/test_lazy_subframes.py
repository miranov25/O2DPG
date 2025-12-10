"""
Phase 7.5a Tests: Lazy Single-File Subframes

Test Categories:
1. Registration (6 tests)
2. Lazy Loading Trigger (5 tests)
3. Join Behavior (5 tests)
4. Error Handling (4 tests)
5. Resource Management (3 tests)

Total: 23 tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_root_file(tmp_path):
    """Create sample ROOT file with main data."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "main.root"
    
    with uproot.create(file_path) as f:
        f["tree"] = {
            "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            "y": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            "sector": np.array([0, 1, 0, 1, 2], dtype=np.int32),
            "run": np.array([100, 100, 100, 100, 100], dtype=np.int32),
        }
    
    return str(file_path)


@pytest.fixture
def calib_root_file(tmp_path):
    """Create calibration ROOT file."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "calib.root"
    
    with uproot.create(file_path) as f:
        f["tree"] = {
            "sector": np.array([0, 1, 2], dtype=np.int32),
            "gain": np.array([1.0, 1.1, 0.9], dtype=np.float64),
            "offset": np.array([0.0, 0.1, -0.1], dtype=np.float64),
        }
    
    return str(file_path)


@pytest.fixture
def calib_root_file_extra_cols(tmp_path):
    """Create calibration ROOT file with extra columns."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "calib_extra.root"
    
    with uproot.create(file_path) as f:
        f["tree"] = {
            "sector": np.array([0, 1, 2], dtype=np.int32),
            "gain": np.array([1.0, 1.1, 0.9], dtype=np.float64),
            "offset": np.array([0.0, 0.1, -0.1], dtype=np.float64),
            "quality": np.array([1, 1, 0], dtype=np.int32),
            "timestamp": np.array([1000, 1001, 1002], dtype=np.int64),
        }
    
    return str(file_path)


@pytest.fixture
def sample_root_file_missing_sector(tmp_path):
    """Create sample ROOT file with sector values not in calibration."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "main_missing.root"
    
    with uproot.create(file_path) as f:
        f["tree"] = {
            "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            "y": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            "sector": np.array([0, 1, 5, 6, 7], dtype=np.int32),  # 5, 6, 7 not in calib
        }
    
    return str(file_path)


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestLazySubframeRegistration:
    """Test register_subframe_lazy() method."""
    
    def test_register_basic(self, sample_root_file, calib_root_file):
        """Basic registration stores config without loading."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file}:tree',
            index_columns=['sector']
        )
        
        # Should be registered but not loaded
        assert 'Calib' in adf.lazy_subframes
        assert 'Calib' not in adf.loaded_subframes
        assert adf._subframe_loaded['Calib'] == False
    
    def test_register_with_columns(self, sample_root_file, calib_root_file_extra_cols):
        """Registration with specific columns subset."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file_extra_cols}:tree',
            index_columns=['sector'],
            columns=['gain', 'offset']  # Only these columns, not quality/timestamp
        )
        
        config = adf._subframe_lazy_config['Calib']
        assert set(config['columns']) == {'gain', 'offset'}
    
    def test_register_validates_file_exists(self, sample_root_file):
        """Registration fails if file doesn't exist."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(FileNotFoundError):
            adf.register_subframe_lazy(
                'Calib',
                'nonexistent.root:tree',
                index_columns=['sector']
            )
    
    def test_register_validates_tree_name(self, sample_root_file, calib_root_file):
        """Registration fails if tree name not provided."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(ValueError, match="Tree name required"):
            adf.register_subframe_lazy(
                'Calib',
                calib_root_file,  # No :tree suffix
                index_columns=['sector']
            )
    
    def test_register_validates_index_columns(self, sample_root_file, calib_root_file):
        """Registration validates index columns exist in subframe."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(KeyError, match="missing index column"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['nonexistent_column']
            )
    
    def test_register_duplicate_name_fails(self, sample_root_file, calib_root_file):
        """Cannot register same name twice."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file}:tree',
            index_columns=['sector']
        )
        
        with pytest.raises(ValueError, match="already registered"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['sector']
            )


class TestLazyLoadingTrigger:
    """Test that lazy subframes load at the right time."""
    
    def test_load_on_materialize(self, sample_root_file, calib_root_file):
        """Subframe loads when alias is materialized."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        # Need to load main branches first
        adf.ensure_branches(['x', 'sector'])
        
        adf.add_alias('corrected', 'x * Calib.gain')
        
        # Not loaded yet
        assert not adf._subframe_loaded['Calib']
        
        # Materialize triggers load
        adf.materialize_aliases(names=['corrected'])
        
        # Now loaded
        assert adf._subframe_loaded['Calib']
        assert 'Calib' in adf.loaded_subframes
        assert 'corrected' in adf.df.columns
    
    def test_load_on_materialize_single(self, sample_root_file, calib_root_file):
        """Subframe loads when single alias is materialized."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.ensure_branches(['x', 'sector'])
        adf.add_alias('corrected', 'x * Calib.gain')
        
        # Not loaded yet
        assert not adf._subframe_loaded['Calib']
        
        # materialize_alias triggers load
        adf.materialize_alias('corrected')
        
        # Now loaded
        assert adf._subframe_loaded['Calib']
        assert 'corrected' in adf.df.columns
    
    def test_explicit_ensure_subframe(self, sample_root_file, calib_root_file):
        """ensure_subframe() explicitly triggers loading."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        # Explicit load
        adf.ensure_subframe('Calib')
        
        assert adf._subframe_loaded['Calib']
        assert 'Calib' in adf.loaded_subframes
    
    def test_ensure_subframe_idempotent(self, sample_root_file, calib_root_file):
        """Multiple ensure_subframe() calls are safe."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.ensure_subframe('Calib')
        adf.ensure_subframe('Calib')  # Should not error
        adf.ensure_subframe('Calib')
        
        assert adf._subframe_loaded['Calib']
    
    def test_ensure_eager_subframe_noop(self, sample_root_file):
        """ensure_subframe() on eager subframe is no-op."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Register eager subframe
        calib_df = pd.DataFrame({
            'sector': [0, 1, 2],
            'gain': [1.0, 1.1, 0.9]
        })
        calib_adf = AliasDataFrame(calib_df)
        adf.register_subframe('Calib', calib_adf, index_columns=['sector'])
        
        # Should not error
        adf.ensure_subframe('Calib')


class TestLazySubframeJoin:
    """Test join behavior after lazy loading."""
    
    def test_join_produces_correct_values(self, sample_root_file, calib_root_file):
        """Lazy subframe join produces correct results."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.ensure_branches(['x', 'sector'])
        adf.add_alias('corrected', 'x * Calib.gain')
        adf.materialize_aliases(names=['corrected'])
        
        # Expected: x * gain for each sector
        # sector 0: gain 1.0, sector 1: gain 1.1, sector 2: gain 0.9
        # x values: [1, 2, 3, 4, 5], sectors: [0, 1, 0, 1, 2]
        expected = np.array([1.0*1.0, 2.0*1.1, 3.0*1.0, 4.0*1.1, 5.0*0.9])
        np.testing.assert_array_almost_equal(adf.df['corrected'].values, expected)
    
    def test_missing_keys_fill_nan(self, sample_root_file_missing_sector, calib_root_file):
        """Missing keys in subframe produce NaN."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file_missing_sector, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.ensure_branches(['x', 'sector'])
        adf.add_alias('corrected', 'x * Calib.gain')
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['corrected'])
        
        # Sectors 5, 6, 7 not in calib, should produce NaN
        result = adf.df['corrected'].values
        assert not np.isnan(result[0])  # sector 0 exists
        assert not np.isnan(result[1])  # sector 1 exists
        assert np.isnan(result[2])      # sector 5 missing
        assert np.isnan(result[3])      # sector 6 missing
        assert np.isnan(result[4])      # sector 7 missing
    
    def test_column_subset_works(self, sample_root_file, calib_root_file_extra_cols):
        """Only requested columns are loaded."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file_extra_cols}:tree',
            index_columns=['sector'],
            columns=['gain']  # Only gain, not offset/quality/timestamp
        )
        
        adf.ensure_subframe('Calib')
        
        # Get the loaded subframe
        sf = adf.get_subframe('Calib')
        
        # Only gain and index should be loaded
        assert 'gain' in sf.df.columns
        assert 'sector' in sf.df.columns
        # Other columns should not be loaded
        assert 'quality' not in sf.df.columns
        assert 'timestamp' not in sf.df.columns
    
    def test_multiple_lazy_subframes(self, sample_root_file, calib_root_file, tmp_path):
        """Multiple lazy subframes work together."""
        uproot = pytest.importorskip("uproot")
        
        # Create second calibration file
        calib2_path = tmp_path / "calib2.root"
        with uproot.create(calib2_path) as f:
            f["tree"] = {
                "run": np.array([100], dtype=np.int32),
                "scale": np.array([2.0], dtype=np.float64),
            }
        
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Register two lazy subframes
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        adf.register_subframe_lazy(
            'RunScale', f'{str(calib2_path)}:tree', index_columns=['run']
        )
        
        assert len(adf.lazy_subframes) == 2
        assert 'Calib' in adf.lazy_subframes
        assert 'RunScale' in adf.lazy_subframes
        
        # Load main data
        adf.ensure_branches(['x', 'sector', 'run'])
        
        # Use both subframes
        adf.add_alias('result', 'x * Calib.gain * RunScale.scale')
        adf.materialize_aliases(names=['result'])
        
        # Both should be loaded now
        assert adf._subframe_loaded['Calib']
        assert adf._subframe_loaded['RunScale']
    
    def test_mixed_eager_lazy_subframes(self, sample_root_file, calib_root_file):
        """Can have both eager and lazy subframes."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Eager subframe
        eager_df = pd.DataFrame({'run': [100], 'scale': [2.0]})
        eager_adf = AliasDataFrame(eager_df)
        adf.register_subframe('Eager', eager_adf, index_columns=['run'])
        
        # Lazy subframe
        adf.register_subframe_lazy(
            'Lazy', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        assert 'Eager' in adf.loaded_subframes
        assert 'Lazy' not in adf.loaded_subframes
        assert 'Lazy' in adf.lazy_subframes


class TestLazySubframeErrors:
    """Test error handling."""
    
    def test_ensure_unknown_subframe_raises(self, sample_root_file):
        """ensure_subframe() raises for unknown name."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(KeyError):
            adf.ensure_subframe('NonExistent')
    
    def test_missing_index_columns_raises(self, sample_root_file, calib_root_file):
        """Registration without index_columns raises."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(ValueError, match="index_columns required"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=None
            )
    
    def test_invalid_columns_raises(self, sample_root_file, calib_root_file):
        """Registration with invalid columns raises."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(KeyError, match="missing column"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['sector'],
                columns=['nonexistent_col']
            )
    
    def test_eager_lazy_name_conflict(self, sample_root_file, calib_root_file):
        """Cannot register lazy subframe with same name as eager."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Register eager first
        eager_df = pd.DataFrame({'sector': [0, 1, 2], 'gain': [1.0, 1.1, 0.9]})
        eager_adf = AliasDataFrame(eager_df)
        adf.register_subframe('Calib', eager_adf, index_columns=['sector'])
        
        # Try to register lazy with same name
        with pytest.raises(ValueError, match="already registered"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['sector']
            )


class TestLazySubframeResources:
    """Test resource management."""
    
    def test_close_releases_readers(self, sample_root_file, calib_root_file):
        """close() releases subframe readers."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        assert len(adf._subframe_readers) == 1
        
        adf.close()
        
        assert len(adf._subframe_readers) == 0
        assert len(adf._subframe_loaded) == 0
        assert len(adf._subframe_lazy_config) == 0
    
    def test_context_manager_cleanup(self, sample_root_file, calib_root_file):
        """Context manager cleans up subframe readers."""
        with AliasDataFrame.read_tree_lazy(sample_root_file, 'tree') as adf:
            adf.register_subframe_lazy(
                'Calib', f'{calib_root_file}:tree', index_columns=['sector']
            )
            assert len(adf._subframe_readers) == 1
        
        # After context exit, readers should be cleared
        assert len(adf._subframe_readers) == 0
    
    def test_lazy_subframes_property(self, sample_root_file, calib_root_file, tmp_path):
        """lazy_subframes property returns correct names."""
        uproot = pytest.importorskip("uproot")
        
        # Create second calibration file
        calib2_path = tmp_path / "calib2.root"
        with uproot.create(calib2_path) as f:
            f["tree"] = {
                "run": np.array([100], dtype=np.int32),
                "scale": np.array([2.0], dtype=np.float64),
            }
        
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib1', f'{calib_root_file}:tree', index_columns=['sector']
        )
        adf.register_subframe_lazy(
            'Calib2', f'{str(calib2_path)}:tree', index_columns=['run']
        )
        
        assert set(adf.lazy_subframes) == {'Calib1', 'Calib2'}


class TestGetSubframesForAliases:
    """Test _get_subframes_for_aliases helper method."""
    
    def test_finds_direct_subframe_reference(self, sample_root_file, calib_root_file):
        """Finds subframe referenced directly in alias."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.add_alias('corrected', 'x * Calib.gain')
        
        needed = adf._get_subframes_for_aliases(['corrected'])
        assert 'Calib' in needed
    
    def test_finds_nested_subframe_reference(self, sample_root_file, calib_root_file):
        """Finds subframe referenced through alias dependency."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.register_subframe_lazy(
            'Calib', f'{calib_root_file}:tree', index_columns=['sector']
        )
        
        adf.add_alias('gain_factor', 'Calib.gain')
        adf.add_alias('corrected', 'x * gain_factor')  # References Calib indirectly
        
        needed = adf._get_subframes_for_aliases(['corrected'])
        assert 'Calib' in needed
    
    def test_no_subframes_needed(self, sample_root_file):
        """Returns empty set when no subframes referenced."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.add_alias('sum', 'x + y')
        
        needed = adf._get_subframes_for_aliases(['sum'])
        assert len(needed) == 0


class TestLazinessPreservation:
    """Test that lazy subframe operations don't break main laziness."""
    
    def test_ensure_subframe_does_not_load_main(self, sample_root_file, calib_root_file):
        """Loading lazy subframe should NOT trigger main tree load."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Main should be lazy (no data loaded yet)
        assert adf._lazy_reader is not None
        initial_loaded = len(adf._lazy_reader.loaded_branches)
        
        # Register and load lazy subframe
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file}:tree',
            index_columns=['sector']
        )
        adf.ensure_subframe('Calib')
        
        # Main should STILL have same loaded branches (not triggered full load)
        assert len(adf._lazy_reader.loaded_branches) == initial_loaded
        
        # Subframe should be loaded
        assert adf._subframe_loaded['Calib'] == True
    
    def test_get_subframe_does_not_load_main(self, sample_root_file, calib_root_file):
        """get_subframe() should not trigger main tree load."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file}:tree',
            index_columns=['sector']
        )
        
        initial_loaded = len(adf._lazy_reader.loaded_branches)
        
        # Access subframe
        sf = adf.get_subframe('Calib')
        
        # Main should not have loaded additional branches
        assert len(adf._lazy_reader.loaded_branches) == initial_loaded


class TestParameterValidation:
    """Test parameter validation in register_subframe_lazy()."""
    
    def test_invalid_alignment_raises(self, sample_root_file, calib_root_file):
        """Invalid alignment parameter should raise ValueError."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(ValueError, match="Invalid alignment"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['sector'],
                alignment='invalid_mode'
            )
    
    def test_invalid_join_type_raises(self, sample_root_file, calib_root_file):
        """Invalid join_type parameter should raise ValueError."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        with pytest.raises(ValueError, match="Invalid join_type"):
            adf.register_subframe_lazy(
                'Calib',
                f'{calib_root_file}:tree',
                index_columns=['sector'],
                join_type='invalid_type'
            )
    
    def test_valid_alignments_accepted(self, sample_root_file, calib_root_file):
        """All valid alignment values should be accepted."""
        for alignment in ['by_key', 'N:1', '1:1']:
            adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
            adf.register_subframe_lazy(
                f'Calib_{alignment}',
                f'{calib_root_file}:tree',
                index_columns=['sector'],
                alignment=alignment
            )
            assert adf._subframe_lazy_config[f'Calib_{alignment}']['alignment'] == alignment


class TestSchemaConsistency:
    """Test schema structure consistency between eager and lazy subframes."""
    
    def test_eager_subframe_has_both_index_keys(self, sample_root_file):
        """Eager subframe schema should have both 'index' and 'index_columns'."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        calib_df = pd.DataFrame({
            'sector': [0, 1, 2],
            'gain': [1.0, 1.1, 0.9]
        })
        calib_adf = AliasDataFrame(calib_df)
        adf.register_subframe('Calib', calib_adf, index_columns=['sector'])
        
        schema = adf._schema['subframes']['Calib']
        assert 'index' in schema
        assert 'index_columns' in schema
        assert schema['index'] == schema['index_columns']
    
    def test_lazy_subframe_has_index_columns(self, sample_root_file, calib_root_file):
        """Lazy subframe schema should have 'index_columns'."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.register_subframe_lazy(
            'Calib',
            f'{calib_root_file}:tree',
            index_columns=['sector']
        )
        
        schema = adf._schema['subframes']['Calib']
        assert 'index_columns' in schema
        assert schema['index_columns'] == ['sector']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
