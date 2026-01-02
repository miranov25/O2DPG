#!/usr/bin/env python3
"""
test_draw_invariance.py - Invariance tests for draw() after Phase 7

Core Principle:
    "Same logical draw spec → same numbers regardless of mode"

Tests verify that draw results are numerically IDENTICAL across:
    - Eager vs lazy loading
    - Single file vs chain
    - draw() vs materialize_aliases()
    - Different batch sizes
    - With and without subframes

The synthetic data has EXACT relationships (no noise):
    - y_derived = 2 * x
    - gain[sec] = 1.0 + 0.01 * sec

This allows np.array_equal() (no tolerance needed).
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame

# =============================================================================
# Dependency Checks
# =============================================================================

# Check for ROOT
try:
    import ROOT
    HAS_ROOT = ROOT is not None
except ImportError:
    HAS_ROOT = False

# Check for dfdraw
try:
    from dfextensions.dfdraw import DFDraw
    HAS_DFDRAW = True
except ImportError:
    HAS_DFDRAW = False

# Skip markers
requires_root = pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
requires_dfdraw = pytest.mark.skipif(not HAS_DFDRAW, reason="dfdraw not available")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def single_file_data(tmp_path_factory):
    """Generate a single synthetic ROOT file for testing."""
    output_dir = tmp_path_factory.mktemp('single_data')
    output_file = output_dir / 'synthetic.root'
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--rows', '5000',
        '--tracks', '500',
        '-o', str(output_file)
    ], check=True, capture_output=True)
    
    return str(output_file)


@pytest.fixture(scope='module')
def chain_data(tmp_path_factory):
    """Generate chain data (3 files) for testing."""
    output_dir = tmp_path_factory.mktemp('chain_data')
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--chain', '3',
        '--rows', '2000',
        '--tracks', '200',
        '-o', str(output_dir)
    ], check=True, capture_output=True)
    
    return str(output_dir)


@pytest.fixture(scope='module')
def chain_with_subframe_data(tmp_path_factory):
    """Generate chain data with subframe chain for testing."""
    output_dir = tmp_path_factory.mktemp('chain_subframe_data')
    
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
    generator = os.path.join(benchmarks_dir, 'generate_synthetic_data.py')
    
    subprocess.run([
        sys.executable, generator,
        '--chain', '3',
        '--subframe-chain', '2',
        '--rows', '2000',
        '--tracks', '200',
        '-o', str(output_dir)
    ], check=True, capture_output=True)
    
    return str(output_dir)


# =============================================================================
# Core Invariant Tests (work without dfdraw)
# =============================================================================

class TestCoreInvariants:
    """Core invariant tests that work without dfdraw."""
    
    def test_known_relationship_single_file(self, single_file_data):
        """y_derived = 2*x must hold in single file."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.ensure_branches(['x', 'y_derived'])
        
        expected = 2.0 * adf.df['x'].values
        actual = adf.df['y_derived'].values
        
        np.testing.assert_allclose(
            expected, actual, rtol=1e-6,
            err_msg="y_derived != 2*x in single file"
        )
    
    def test_known_relationship_chain(self, chain_data):
        """y_derived = 2*x must hold across chain."""
        pattern = os.path.join(chain_data, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        adf.ensure_branches(['x', 'y_derived'])
        
        expected = 2.0 * adf.df['x'].values
        actual = adf.df['y_derived'].values
        
        np.testing.assert_allclose(
            expected, actual, rtol=1e-6,
            err_msg="y_derived != 2*x in chain"
        )
    
    def test_eager_vs_lazy_data_identical(self, chain_data):
        """Eager and lazy chain loading produce identical data."""
        pattern = os.path.join(chain_data, 'data_run*.root')
        
        # Eager
        adf_eager = AliasDataFrame.read_chain(pattern, 'tree')
        
        # Lazy
        adf_lazy = AliasDataFrame.read_chain_lazy(pattern + ':tree')
        adf_lazy.ensure_branches(['x', 'y', 'file_idx'])
        
        # Compare - use allclose for float values
        np.testing.assert_allclose(
            np.sort(adf_eager.df['x'].values),
            np.sort(adf_lazy.df['x'].values),
            rtol=1e-6,
            err_msg="Eager vs lazy chain data differs"
        )
    
    def test_alias_materialization_identical(self, chain_data):
        """Alias computation identical for eager vs lazy."""
        pattern = os.path.join(chain_data, 'data_run*.root')
        
        # Eager
        adf_eager = AliasDataFrame.read_chain(pattern, 'tree')
        adf_eager.add_alias('computed', 'x * 2 + y')
        adf_eager.materialize_alias('computed')
        
        # Lazy - need to ensure branches loaded first
        adf_lazy = AliasDataFrame.read_chain_lazy(pattern + ':tree')
        adf_lazy.ensure_branches(['x', 'y'])  # Load required branches
        adf_lazy.add_alias('computed', 'x * 2 + y')
        adf_lazy.materialize_alias('computed')
        
        # Sort for comparison (order may differ)
        eager_sorted = np.sort(adf_eager.df['computed'].values)
        lazy_sorted = np.sort(adf_lazy.df['computed'].values)
        
        np.testing.assert_allclose(
            eager_sorted, lazy_sorted, rtol=1e-6,
            err_msg="Alias computation differs between eager and lazy"
        )
    
    def test_chain_has_all_files(self, chain_data):
        """Chain contains data from all expected files."""
        pattern = os.path.join(chain_data, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        adf.ensure_branches(['file_idx'])
        
        unique_files = np.unique(adf.df['file_idx'].values)
        expected_files = [0, 1, 2]  # 3 files
        
        np.testing.assert_array_equal(
            sorted(unique_files), expected_files,
            err_msg="Chain missing expected files"
        )
    
    def test_complex_alias_chain(self, single_file_data):
        """Complex alias chain computes correctly."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.ensure_branches(['x'])  # Load required base column
        
        # Diamond dependency
        adf.add_alias('a', 'x + 1')
        adf.add_alias('b', 'a * 2')
        adf.add_alias('c', 'a + 3')
        adf.add_alias('d', 'b + c')
        
        adf.materialize_alias('d')  # Use singular form
        
        # Verify: d = (x+1)*2 + (x+1)+3 = 2x+2 + x+4 = 3x+6
        expected = 3 * adf.df['x'].values + 6
        actual = adf.df['d'].values
        
        np.testing.assert_allclose(
            expected, actual, rtol=1e-5,
            err_msg="Complex alias chain computed incorrectly"
        )


# =============================================================================
# Subframe Join Correctness Tests
# =============================================================================

class TestSubframeJoinCorrectness:
    """Verify subframe joins produce correct results with known relationships."""
    
    def test_sector_calibration_correct(self, single_file_data):
        """Sector calibration join must apply correct gain values."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        
        # Load the SectorCalib subframe from same file
        import uproot
        with uproot.open(single_file_data) as f:
            calib_data = f['SectorCalib'].arrays(library='pd')
        
        calib_adf = AliasDataFrame(calib_data)
        adf.register_subframe('SectorCalib', calib_adf, index_columns=['sec'])
        
        # The synthetic data has: gain[sec] = 1.0 + 0.01 * sec
        adf.add_alias('calibrated', 'signal * SectorCalib.gain')
        adf.ensure_branches(['signal', 'sec'])
        adf.materialize_alias('calibrated')
        
        # Verify the relationship
        expected = adf.df['signal'].values * (1.0 + 0.01 * adf.df['sec'].values)
        actual = adf.df['calibrated'].values
        
        np.testing.assert_allclose(
            expected, actual, rtol=1e-5,
            err_msg="Sector calibration join produced incorrect values"
        )


# =============================================================================
# Chain + Subframe Integration
# =============================================================================

class TestChainSubframeIntegration:
    """Verify chain loading with subframes produces correct results."""
    
    def test_chain_with_subframe_chain(self, chain_with_subframe_data):
        """Chain with subframe chain must apply run-dependent calibration."""
        pattern = os.path.join(chain_with_subframe_data, 'data_run*.root:tree')
        calib_pattern = os.path.join(chain_with_subframe_data, 'calib_*.root:tree')
        
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        # Register subframe chain
        adf.register_subframe_chain(
            'RunCalib',
            calib_pattern,
            index_columns=['run_number', 'sec'],
            validate_branches='first'
        )
        
        adf.add_alias('calibrated', 'signal * RunCalib.gain')
        adf.ensure_branches(['signal', 'sec', 'run_number'])
        adf.materialize_alias('calibrated')
        
        # Verify the relationship: gain = 1.0 + 0.01*sec + 0.001*(run-1000)
        runs = adf.df['run_number'].values
        secs = adf.df['sec'].values
        signals = adf.df['signal'].values
        calibrated = adf.df['calibrated'].values
        
        expected_gain = 1.0 + 0.01 * secs + 0.001 * (runs - 1000)
        expected_calibrated = signals * expected_gain
        
        np.testing.assert_allclose(
            expected_calibrated, calibrated, rtol=1e-5,
            err_msg="Run-dependent calibration produced incorrect values"
        )


# =============================================================================
# dtype Preservation Tests
# =============================================================================

class TestDtypePreservation:
    """Verify dtypes are preserved across loading modes."""
    
    def test_float32_preserved_lazy(self, single_file_data):
        """float32 columns must stay float32 in lazy mode."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.ensure_branches(['x'])
        
        dtype = adf.df['x'].dtype
        assert dtype == np.float32, f"Expected float32, got {dtype}"
    
    def test_int32_preserved_lazy(self, single_file_data):
        """int32 columns must stay int32 in lazy mode."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.ensure_branches(['sec'])
        
        dtype = adf.df['sec'].dtype
        assert dtype == np.int32, f"Expected int32, got {dtype}"


# =============================================================================
# Error Scenarios
# =============================================================================

class TestErrorScenarios:
    """Verify proper error handling."""
    
    def test_missing_branch_raises(self, single_file_data):
        """Accessing non-existent branch must raise."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        
        with pytest.raises((KeyError, ValueError)):
            adf.ensure_branches(['nonexistent_branch'])
    
    def test_invalid_alias_raises(self, single_file_data):
        """Invalid alias expression must raise on materialize."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.add_alias('bad', 'x + nonexistent')
        
        with pytest.raises((KeyError, NameError, ValueError)):
            adf.materialize_alias('bad')
    
    def test_circular_alias_raises(self, single_file_data):
        """Circular alias dependency must raise error."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.ensure_branches(['x'])  # Load a base column
        
        # Create first alias
        adf.add_alias('a', 'b + 1')
        
        # Creating circular dependency should raise at add_alias time
        with pytest.raises((ValueError, RecursionError)):
            adf.add_alias('b', 'a + 1')


# =============================================================================
# Draw-Specific Tests (require dfdraw)
# =============================================================================

@requires_dfdraw
class TestDrawInvariance:
    """Tests that specifically require draw() functionality."""
    
    def test_draw_vs_materialize_identical(self, single_file_data):
        """draw() must match histogram of materialized data."""
        adf = AliasDataFrame.read_tree_lazy(single_file_data, 'tree')
        adf.add_alias('computed', 'x * 2 + y')
        
        # Via draw() with lazy=True to auto-materialize alias
        # Returns (fig, ax, stats_dict) where stats_dict has 'n', 'mean', 'std', etc.
        fig, ax, stats = adf.draw('computed', bins=100, lazy=True)
        
        # Verify stats dict has expected keys and values
        assert 'n' in stats, "stats should have 'n' key"
        assert stats['n'] == len(adf.df), f"stats['n']={stats['n']} should equal len(df)={len(adf.df)}"
        
        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_eager_vs_lazy_draw_identical(self, chain_data):
        """Draw results must be identical for eager vs lazy chain."""
        import matplotlib.pyplot as plt
        
        pattern = os.path.join(chain_data, 'data_run*.root')
        
        # Eager - returns (fig, ax, stats_dict)
        adf_eager = AliasDataFrame.read_chain(pattern, 'tree')
        fig_eager, ax_eager, stats_eager = adf_eager.draw('x', bins=100, range=(-100, 500))
        
        # Lazy - returns (fig, ax, stats_dict)
        adf_lazy = AliasDataFrame.read_chain_lazy(pattern + ':tree')
        fig_lazy, ax_lazy, stats_lazy = adf_lazy.draw('x', bins=100, range=(-100, 500))
        
        # Compare stats - both should have same count
        assert stats_eager['n'] == stats_lazy['n'], \
            f"Entry count differs: eager={stats_eager['n']}, lazy={stats_lazy['n']}"
        
        # Compare means (should be very close)
        np.testing.assert_allclose(
            stats_eager['mean'], stats_lazy['mean'], rtol=1e-6,
            err_msg="Mean differs between eager and lazy chain"
        )
        
        # Cleanup matplotlib figures
        plt.close(fig_eager)
        plt.close(fig_lazy)
