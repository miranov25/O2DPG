#!/usr/bin/env python3
"""
test_draw_chain_integration.py - Chain drawing integration tests

Tests all 6 combinations of loading with Phase 7 features:

| Main Data          | Subframe           | Test Name                           |
|--------------------|--------------------|------------------------------------|
| Single file, eager | None               | test_single_eager_baseline         |
| Single file, lazy  | None               | test_single_lazy                   |
| Chain, lazy        | None               | test_chain_lazy                    |
| Single file, lazy  | Single file, lazy  | test_with_lazy_subframe            |
| Chain, lazy        | Single file, lazy  | test_chain_with_subframe_single    |
| Chain, lazy        | Chain, lazy        | test_chain_with_subframe_chain     |

Each combination tests both happy-path and basic error scenarios.
"""

import os
import sys
import subprocess
import glob
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame

# =============================================================================
# Dependency Checks
# =============================================================================

try:
    import ROOT
    HAS_ROOT = ROOT is not None
except ImportError:
    HAS_ROOT = False

try:
    from dfdraw import DFDraw
    HAS_DFDRAW = True
except ImportError:
    HAS_DFDRAW = False

requires_root = pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
requires_dfdraw = pytest.mark.skipif(not HAS_DFDRAW, reason="dfdraw not available")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def single_file(tmp_path_factory):
    """Generate a single synthetic ROOT file."""
    output_dir = tmp_path_factory.mktemp('single')
    output_file = output_dir / 'data.root'
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--rows', '3000',
        '--tracks', '300',
        '-o', str(output_file)
    ], check=True, capture_output=True)
    
    return str(output_file)


@pytest.fixture(scope='module')
def chain_files(tmp_path_factory):
    """Generate chain data (3 files) without subframe chain."""
    output_dir = tmp_path_factory.mktemp('chain')
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--chain', '3',
        '--rows', '1500',
        '--tracks', '150',
        '-o', str(output_dir)
    ], check=True, capture_output=True)
    
    return str(output_dir)


@pytest.fixture(scope='module')
def chain_with_subframe(tmp_path_factory):
    """Generate chain data with subframe chain."""
    output_dir = tmp_path_factory.mktemp('chain_sub')
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--chain', '3',
        '--subframe-chain', '2',
        '--rows', '1500',
        '--tracks', '150',
        '-o', str(output_dir)
    ], check=True, capture_output=True)
    
    return str(output_dir)


# =============================================================================
# Combination 1: Single file, eager - Baseline
# =============================================================================

class TestSingleEagerBaseline:
    """Baseline tests: Single file with eager loading."""
    
    def test_eager_load_basic(self, single_file):
        """Basic eager chain load works (single file as 'chain')."""
        # Use read_chain with single file pattern
        adf = AliasDataFrame.read_chain(single_file, 'tree')
        
        assert len(adf.df) > 0
        assert 'x' in adf.df.columns
    
    def test_eager_alias_works(self, single_file):
        """Alias on eager-loaded data works."""
        adf = AliasDataFrame.read_chain(single_file, 'tree')
        adf.add_alias('computed', 'x * 2 + y')
        adf.materialize_alias('computed')
        
        expected = adf.df['x'] * 2 + adf.df['y']
        np.testing.assert_allclose(
            adf.df['computed'].values, expected.values, rtol=1e-6
        )
    
    def test_eager_alias_chain(self, single_file):
        """Chained aliases on eager-loaded data."""
        adf = AliasDataFrame.read_chain(single_file, 'tree')
        adf.add_alias('a', 'x + 1')
        adf.add_alias('b', 'a * 2')
        adf.add_alias('c', 'b + y')
        adf.materialize_alias('c')
        
        expected = (adf.df['x'] + 1) * 2 + adf.df['y']
        np.testing.assert_allclose(
            adf.df['c'].values, expected.values, rtol=1e-6
        )
    
    def test_eager_subframe_works(self, single_file):
        """Eagerly loaded subframe works."""
        import uproot
        
        adf = AliasDataFrame.read_chain(single_file, 'tree')
        
        # Load SectorCalib from file
        with uproot.open(single_file) as f:
            calib_data = f['SectorCalib'].arrays(library='pd')
        
        calib_adf = AliasDataFrame(calib_data)
        adf.register_subframe('SectorCalib', calib_adf, index_columns=['sec'])
        
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.materialize_alias('calibrated')
        
        # Verify
        expected = adf.df['signal'] * (1.0 + 0.01 * adf.df['sec'])
        np.testing.assert_allclose(
            adf.df['calibrated'].values, expected.values, rtol=1e-5
        )


# =============================================================================
# Combination 2: Single file, lazy
# =============================================================================

class TestSingleLazy:
    """Single file with lazy loading."""
    
    def test_lazy_load_basic(self, single_file):
        """Lazy load creates reader without loading data."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        # Reader exists
        assert adf._lazy_reader is not None
        # No data loaded yet
        assert len(adf._lazy_reader.loaded_branches) == 0
    
    def test_lazy_ensure_branches(self, single_file):
        """ensure_branches loads specified columns."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        adf.ensure_branches(['x', 'y'])
        
        assert 'x' in adf._lazy_reader.loaded_branches
        assert 'y' in adf._lazy_reader.loaded_branches
        assert len(adf.df) > 0
    
    def test_lazy_alias_triggers_load(self, single_file):
        """Alias materialization works after loading required branches."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        adf.ensure_branches(['x', 'y'])  # Load required branches first
        adf.add_alias('sum_xy', 'x + y')
        adf.materialize_alias('sum_xy')
        
        # Branches should be loaded
        assert 'x' in adf._lazy_reader.loaded_branches
        assert 'y' in adf._lazy_reader.loaded_branches
        # Alias should be computed
        assert 'sum_xy' in adf.df.columns
    
    def test_lazy_only_loads_needed(self, single_file):
        """Lazy loading only loads required branches."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        adf.ensure_branches(['x'])
        
        loaded = adf._lazy_reader.loaded_branches
        available = adf._lazy_reader.available_branches
        
        assert 'x' in loaded
        assert len(loaded) < len(available)


# =============================================================================
# Combination 3: Chain, lazy
# =============================================================================

class TestChainLazy:
    """Chain of files with lazy loading."""
    
    def test_chain_lazy_basic(self, chain_files):
        """Lazy chain read works."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        assert adf._lazy_reader is not None
        assert len(adf._lazy_reader._files) == 3
    
    def test_chain_lazy_has_all_files(self, chain_files):
        """Chain includes data from all files."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        adf.ensure_branches(['file_idx'])
        
        unique_files = np.unique(adf.df['file_idx'].values)
        assert len(unique_files) == 3
    
    def test_chain_lazy_alias(self, chain_files):
        """Alias on lazy chain works."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        adf.ensure_branches(['x', 'y'])  # Load required branches first
        adf.add_alias('computed', 'x * 2 + y')
        adf.materialize_alias('computed')
        
        expected = adf.df['x'] * 2 + adf.df['y']
        np.testing.assert_allclose(
            adf.df['computed'].values, expected.values, rtol=1e-6
        )
    
    def test_chain_preserves_invariant(self, chain_files):
        """y_derived = 2*x preserved across chain."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        adf.ensure_branches(['x', 'y_derived'])
        
        expected = 2.0 * adf.df['x'].values
        actual = adf.df['y_derived'].values
        
        np.testing.assert_allclose(expected, actual, rtol=1e-6)


# =============================================================================
# Combination 4: Single file, lazy + lazy subframe
# =============================================================================

class TestWithLazySubframe:
    """Single lazy file with lazy subframe."""
    
    def test_lazy_subframe_registration(self, single_file):
        """Lazy subframe can be registered."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        adf.register_subframe_lazy(
            'SectorCalib',
            single_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        assert 'SectorCalib' in adf.lazy_subframes
    
    def test_lazy_subframe_loads_on_demand(self, single_file):
        """Lazy subframe loads when alias needs it."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        adf.register_subframe_lazy(
            'SectorCalib',
            single_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        # Not loaded yet
        assert 'SectorCalib' not in adf._subframes.subframes
        
        # Load required columns first (including index column for join)
        adf.ensure_branches(['signal', 'sec'])
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.materialize_alias('calibrated')
        
        # Now loaded
        assert 'SectorCalib' in adf._subframes.subframes
    
    def test_lazy_subframe_correct_values(self, single_file):
        """Lazy subframe join produces correct values."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        adf.register_subframe_lazy(
            'SectorCalib',
            single_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.ensure_branches(['signal', 'sec'])
        adf.materialize_alias('calibrated')
        
        # gain[sec] = 1.0 + 0.01 * sec
        expected = adf.df['signal'].values * (1.0 + 0.01 * adf.df['sec'].values)
        actual = adf.df['calibrated'].values
        
        np.testing.assert_allclose(expected, actual, rtol=1e-5)


# =============================================================================
# Combination 5: Chain, lazy + single file subframe
# =============================================================================

class TestChainWithSubframeSingle:
    """Chain with single-file subframe."""
    
    def test_chain_with_single_subframe(self, chain_files):
        """Chain with subframe from single file."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        # Get SectorCalib from first file
        first_file = sorted(glob.glob(os.path.join(chain_files, 'data_run*.root')))[0]
        
        adf.register_subframe_lazy(
            'SectorCalib',
            first_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        # Load required columns including join key
        adf.ensure_branches(['signal', 'sec'])
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.materialize_alias('calibrated')
        
        assert len(adf.df) > 0
        assert 'calibrated' in adf.df.columns
    
    def test_chain_subframe_applied_to_all_files(self, chain_files):
        """Single subframe calibration applies to all chain files."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        first_file = sorted(glob.glob(os.path.join(chain_files, 'data_run*.root')))[0]
        
        adf.register_subframe_lazy(
            'SectorCalib',
            first_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.ensure_branches(['file_idx', 'signal', 'sec'])
        adf.materialize_alias('calibrated')
        
        # Verify calibration correct for all files
        expected = adf.df['signal'].values * (1.0 + 0.01 * adf.df['sec'].values)
        actual = adf.df['calibrated'].values
        
        np.testing.assert_allclose(expected, actual, rtol=1e-5)


# =============================================================================
# Combination 6: Chain, lazy + Chain subframe
# =============================================================================

class TestChainWithSubframeChain:
    """Chain with subframe chain (N:M join)."""
    
    def test_chain_with_subframe_chain_basic(self, chain_with_subframe):
        """Chain with subframe chain registers successfully."""
        pattern = os.path.join(chain_with_subframe, 'data_run*.root:tree')
        calib_pattern = os.path.join(chain_with_subframe, 'calib_*.root:tree')
        
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        adf.register_subframe_chain(
            'RunCalib',
            calib_pattern,
            index_columns=['run_number', 'sec'],
            validate_branches='first'
        )
        
        assert 'RunCalib' in adf.chain_subframes
    
    def test_chain_subframe_chain_correct_values(self, chain_with_subframe):
        """Chain + subframe chain produces correct run-dependent calibration."""
        pattern = os.path.join(chain_with_subframe, 'data_run*.root:tree')
        calib_pattern = os.path.join(chain_with_subframe, 'calib_*.root:tree')
        
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        adf.register_subframe_chain(
            'RunCalib',
            calib_pattern,
            index_columns=['run_number', 'sec'],
            validate_branches='first'
        )
        
        adf.add_alias('calibrated', 'signal * RunCalib.gain')
        adf.ensure_branches(['signal', 'sec', 'run_number'])
        adf.materialize_alias('calibrated')
        
        # Verify: gain = 1.0 + 0.01*sec + 0.001*(run-1000)
        runs = adf.df['run_number'].values
        secs = adf.df['sec'].values
        signals = adf.df['signal'].values
        
        expected_gain = 1.0 + 0.01 * secs + 0.001 * (runs - 1000)
        expected = signals * expected_gain
        actual = adf.df['calibrated'].values
        
        np.testing.assert_allclose(expected, actual, rtol=1e-5)
    
    def test_chain_subframe_chain_per_run(self, chain_with_subframe):
        """Verify calibration is correct for each run separately."""
        pattern = os.path.join(chain_with_subframe, 'data_run*.root:tree')
        calib_pattern = os.path.join(chain_with_subframe, 'calib_*.root:tree')
        
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        adf.register_subframe_chain(
            'RunCalib',
            calib_pattern,
            index_columns=['run_number', 'sec'],
            validate_branches='first'
        )
        
        adf.add_alias('calibrated', 'signal * RunCalib.gain')
        adf.ensure_branches(['signal', 'sec', 'run_number'])
        adf.materialize_alias('calibrated')
        
        # Check each run separately
        for run in adf.df['run_number'].unique():
            mask = adf.df['run_number'] == run
            run_secs = adf.df.loc[mask, 'sec'].values
            run_signals = adf.df.loc[mask, 'signal'].values
            run_calibrated = adf.df.loc[mask, 'calibrated'].values
            
            expected_gain = 1.0 + 0.01 * run_secs + 0.001 * (run - 1000)
            expected = run_signals * expected_gain
            
            np.testing.assert_allclose(
                expected, run_calibrated, rtol=1e-5,
                err_msg=f"Calibration incorrect for run {run}"
            )


# =============================================================================
# Error Scenarios for Integration
# =============================================================================

class TestIntegrationErrors:
    """Error scenarios for chain integration."""
    
    def test_chain_missing_file_pattern_raises(self, tmp_path):
        """Non-matching pattern should raise."""
        pattern = str(tmp_path / 'nonexistent_*.root:tree')
        
        with pytest.raises((FileNotFoundError, ValueError)):
            AliasDataFrame.read_chain_lazy(pattern)
    
    def test_subframe_chain_missing_files_raises(self, chain_files):
        """Subframe chain with no matching files raises."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        bad_pattern = os.path.join(chain_files, 'nonexistent_*.root:tree')
        
        with pytest.raises(FileNotFoundError):
            adf.register_subframe_chain(
                'BadCalib',
                bad_pattern,
                index_columns=['sec']
            )
    
    def test_subframe_chain_missing_index_column_raises(self, chain_with_subframe):
        """Subframe chain missing index column raises."""
        pattern = os.path.join(chain_with_subframe, 'data_run*.root:tree')
        calib_pattern = os.path.join(chain_with_subframe, 'calib_*.root:tree')
        
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        with pytest.raises(KeyError):
            adf.register_subframe_chain(
                'RunCalib',
                calib_pattern,
                index_columns=['nonexistent_column']
            )


# =============================================================================
# Performance Sanity Checks
# =============================================================================

class TestPerformanceSanity:
    """Basic performance sanity checks."""
    
    def test_lazy_chain_doesnt_load_all_branches(self, chain_files):
        """Lazy chain should not load all branches for single column."""
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        # Load only 'x'
        adf.ensure_branches(['x'])
        
        loaded = adf._lazy_reader.loaded_branches
        available = adf._lazy_reader.available_branches
        
        assert 'x' in loaded
        assert len(loaded) < len(available)
    
    def test_subframe_not_loaded_if_not_needed(self, single_file):
        """Lazy subframe should not load if alias doesn't use it."""
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        adf.register_subframe_lazy(
            'SectorCalib',
            single_file,
            tree_name='SectorCalib',
            index_columns=['sec']
        )
        
        # Use alias that doesn't need subframe, but load branches first
        adf.ensure_branches(['x', 'y'])
        adf.add_alias('simple', 'x + y')
        adf.materialize_alias('simple')
        
        # Subframe should NOT be loaded
        assert 'SectorCalib' not in adf._subframes.subframes


# =============================================================================
# Draw-Specific Integration Tests (require dfdraw)
# =============================================================================

@requires_dfdraw
class TestDrawIntegration:
    """Tests that specifically require draw() functionality."""
    
    def test_draw_single_eager(self, single_file):
        """Draw on eager single file."""
        import matplotlib.pyplot as plt
        
        adf = AliasDataFrame.read_chain(single_file, 'tree')
        # draw() returns (fig, ax, stats_dict) where stats_dict has 'n', 'mean', etc.
        fig, ax, stats = adf.draw('x', bins=50)
        
        # stats['n'] should equal DataFrame length
        assert stats['n'] == len(adf.df), \
            f"stats['n']={stats['n']} should equal len(df)={len(adf.df)}"
        
        plt.close(fig)
    
    def test_draw_chain_lazy(self, chain_files):
        """Draw on lazy chain."""
        import matplotlib.pyplot as plt
        
        pattern = os.path.join(chain_files, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        # draw() returns (fig, ax, stats_dict)
        fig, ax, stats = adf.draw('x', bins=50)
        
        # Verify stats has entries
        assert stats['n'] > 0, "stats['n'] should be > 0"
        
        plt.close(fig)
    
    @pytest.mark.xfail(
        reason="BUG-2025-12-11: draw() in lazy mode treats subframe names as TTree branches. Fix in Phase 6.8a"
    )
    def test_draw_with_subframe(self, single_file):
        """Draw with subframe join.
        
        KNOWN BUG: draw() calls ensure_branches(['SectorCalib']) but SectorCalib
        is a subframe name, not a TTree branch. LazyTreeReader correctly raises
        ValueError. Fix: Phase 6.8a should filter subframe names from branch list.
        """
        import matplotlib.pyplot as plt
        import uproot
        
        adf = AliasDataFrame.read_tree_lazy(single_file, 'tree')
        
        # Load calibration subframe
        with uproot.open(single_file) as f:
            calib_data = f['SectorCalib'].arrays(library='pd')
        calib_adf = AliasDataFrame(calib_data)
        adf.register_subframe('SectorCalib', calib_adf, index_columns=['sec'])
        
        # Define alias using subframe
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        
        # Must load required branches and materialize alias before draw
        adf.ensure_branches(['signal', 'sec'])
        adf.materialize_alias('calibrated')
        
        # Now draw the materialized column
        # draw() returns (fig, ax, stats_dict)
        fig, ax, stats = adf.draw('calibrated', bins=50)
        
        # Verify draw succeeded
        assert stats['n'] > 0, "stats['n'] should be > 0"
        assert 'calibrated' in adf.df.columns, "calibrated should be materialized"
        
        plt.close(fig)
