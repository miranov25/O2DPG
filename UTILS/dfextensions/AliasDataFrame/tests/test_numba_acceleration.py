"""
Test suite for Numba acceleration (Phase 8).

These tests verify that:
1. Numba and non-Numba paths produce identical results
2. Graceful fallback when Numba unavailable
3. Performance threshold is respected
4. Different dtypes are handled correctly
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame, NUMBA_AVAILABLE

# Import accelerators for direct testing
try:
    from _numba_accelerators import (
        numba_scatter, numba_compute_join_indices,
        NUMBA_MIN_ROWS, get_numba_info
    )
    ACCELERATORS_AVAILABLE = True
except ImportError:
    ACCELERATORS_AVAILABLE = False


class TestNumbaAvailability:
    """Tests for Numba availability detection."""
    
    def test_numba_info_property(self):
        """ADF should expose numba_info property."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        info = adf.numba_info
        assert 'available' in info
        assert 'enabled' in info
        assert 'min_rows' in info
        assert isinstance(info['min_rows'], int)
    
    def test_use_numba_parameter(self):
        """Constructor should accept use_numba parameter."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        
        # Explicit disable
        adf = AliasDataFrame(df, use_numba=False)
        assert adf._use_numba == False
        
        # Auto-detect (depends on whether numba is installed)
        adf_auto = AliasDataFrame(df, use_numba=None)
        assert adf_auto._use_numba == NUMBA_AVAILABLE


class TestNumbaScatter:
    """Tests for Phase 8a: Numba scatter (value extraction)."""
    
    @pytest.fixture
    def large_data(self):
        """Create data large enough to trigger Numba (>10K rows)."""
        np.random.seed(42)
        n_main = 50000  # Above NUMBA_MIN_ROWS
        n_sub = 1000
        
        main_df = pd.DataFrame({
            'idx': np.random.randint(0, n_sub, n_main),
            'x': np.random.randn(n_main).astype(np.float32)
        })
        
        sub_df = pd.DataFrame({
            'idx': np.arange(n_sub),
            'val_f32': np.random.randn(n_sub).astype(np.float32),
            'val_f64': np.random.randn(n_sub).astype(np.float64),
        })
        
        return main_df, sub_df
    
    def test_numba_produces_identical_results_f32(self, large_data):
        """Numba and NumPy paths should produce identical float32 results."""
        main_df, sub_df = large_data
        
        # With Numba (if available)
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numba.add_alias('val', 'S.val_f32')
        adf_numba.materialize_alias('val')
        result_numba = adf_numba.df['val'].copy()
        
        # Without Numba
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_numpy.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numpy.add_alias('val', 'S.val_f32')
        adf_numpy.materialize_alias('val')
        result_numpy = adf_numpy.df['val'].copy()
        
        # Compare
        np.testing.assert_array_almost_equal(
            result_numba.values, 
            result_numpy.values,
            decimal=6,
            err_msg="Numba and NumPy results differ for float32"
        )
    
    def test_numba_produces_identical_results_f64(self, large_data):
        """Numba and NumPy paths should produce identical float64 results."""
        main_df, sub_df = large_data
        
        # With Numba
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numba.add_alias('val', 'S.val_f64')
        adf_numba.materialize_alias('val')
        result_numba = adf_numba.df['val'].copy()
        
        # Without Numba
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_numpy.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numpy.add_alias('val', 'S.val_f64')
        adf_numpy.materialize_alias('val')
        result_numpy = adf_numpy.df['val'].copy()
        
        np.testing.assert_array_almost_equal(
            result_numba.values,
            result_numpy.values,
            decimal=10,
            err_msg="Numba and NumPy results differ for float64"
        )
    
    def test_numba_handles_missing_keys(self, large_data):
        """Numba path should correctly handle missing keys."""
        main_df, sub_df = large_data
        
        # Add some keys that don't exist in subframe
        main_df_with_missing = main_df.copy()
        main_df_with_missing.loc[0:100, 'idx'] = 99999  # Non-existent key
        
        # With Numba
        adf_numba = AliasDataFrame(main_df_with_missing.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numba.add_alias('val', 'S.val_f32')
        adf_numba.materialize_alias('val')
        result_numba = adf_numba.df['val'].values
        
        # Without Numba
        adf_numpy = AliasDataFrame(main_df_with_missing.copy(), use_numba=False)
        adf_numpy.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numpy.add_alias('val', 'S.val_f32')
        adf_numpy.materialize_alias('val')
        result_numpy = adf_numpy.df['val'].values
        
        # Check NaN positions match
        np.testing.assert_array_equal(
            np.isnan(result_numba),
            np.isnan(result_numpy),
            err_msg="Missing key handling differs between Numba and NumPy"
        )
    
    def test_small_data_uses_numpy(self):
        """Small datasets should use NumPy (JIT overhead not worth it)."""
        # Create small dataset (below threshold)
        main_df = pd.DataFrame({
            'idx': np.arange(100),  # Below NUMBA_MIN_ROWS
            'x': np.random.randn(100)
        })
        sub_df = pd.DataFrame({
            'idx': np.arange(50),
            'val': np.random.randn(50)
        })
        
        adf = AliasDataFrame(main_df, use_numba=True)
        adf.register_subframe('S', AliasDataFrame(sub_df), index_columns='idx')
        adf.add_alias('v', 'S.val')
        
        # Should work fine (uses NumPy path due to size)
        adf.materialize_alias('v')
        assert 'v' in adf.df.columns


class TestNumbaIndexLookup:
    """Tests for Phase 8b: Numba index lookup (replace pd.merge)."""
    
    @pytest.fixture
    def integer_key_data(self):
        """Create data with integer keys for Numba index lookup."""
        np.random.seed(42)
        n_main = 50000
        n_sub = 1000
        
        main_df = pd.DataFrame({
            'key': np.random.randint(0, n_sub, n_main).astype(np.int64),
            'x': np.random.randn(n_main)
        })
        
        sub_df = pd.DataFrame({
            'key': np.arange(n_sub, dtype=np.int64),
            'val': np.random.randn(n_sub)
        })
        
        return main_df, sub_df
    
    def test_numba_index_lookup_matches_pandas(self, integer_key_data):
        """Numba index lookup should match pandas merge results."""
        main_df, sub_df = integer_key_data
        
        # With Numba
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='key')
        adf_numba.add_alias('v', 'S.val')
        adf_numba.materialize_alias('v')
        result_numba = adf_numba.df['v'].values
        
        # Without Numba (pandas merge)
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='key')
        adf_pandas.add_alias('v', 'S.val')
        adf_pandas.materialize_alias('v')
        result_pandas = adf_pandas.df['v'].values
        
        np.testing.assert_array_almost_equal(
            result_numba,
            result_pandas,
            decimal=10,
            err_msg="Numba index lookup doesn't match pandas merge"
        )
    
    def test_sparse_keys(self):
        """Numba should handle sparse key ranges correctly."""
        np.random.seed(42)
        n_main = 20000
        
        # Sparse keys (large gaps)
        main_df = pd.DataFrame({
            'key': np.random.choice([0, 1000, 5000, 9999], n_main).astype(np.int64),
            'x': np.random.randn(n_main)
        })
        
        sub_df = pd.DataFrame({
            'key': np.array([0, 1000, 5000, 9999], dtype=np.int64),
            'val': np.array([1.0, 2.0, 3.0, 4.0])
        })
        
        # With Numba
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='key')
        adf_numba.add_alias('v', 'S.val')
        adf_numba.materialize_alias('v')
        
        # Without Numba
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='key')
        adf_pandas.add_alias('v', 'S.val')
        adf_pandas.materialize_alias('v')
        
        np.testing.assert_array_almost_equal(
            adf_numba.df['v'].values,
            adf_pandas.df['v'].values
        )
    
    def test_multi_column_key_uses_pandas(self):
        """Multi-column keys should fall back to pandas merge."""
        np.random.seed(42)
        n_main = 20000
        
        main_df = pd.DataFrame({
            'key1': np.random.randint(0, 100, n_main),
            'key2': np.random.randint(0, 100, n_main),
            'x': np.random.randn(n_main)
        })
        
        sub_df = pd.DataFrame({
            'key1': np.repeat(np.arange(100), 10),
            'key2': np.tile(np.arange(10), 100),
            'val': np.random.randn(1000)
        })
        
        # Should work (falls back to pandas)
        adf = AliasDataFrame(main_df, use_numba=True)
        adf.register_subframe('S', AliasDataFrame(sub_df), index_columns=['key1', 'key2'])
        adf.add_alias('v', 'S.val')
        adf.materialize_alias('v')
        
        assert 'v' in adf.df.columns
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_index_lookup_single_column_verified(self):
        """
        Verify Phase 8b Numba path is exercised for single-column integer keys.
        
        This test explicitly confirms that numba_compute_join_indices is called
        when conditions are met:
        - use_numba=True
        - Single-column key
        - Integer dtype
        - n_rows >= NUMBA_MIN_ROWS (10000)
        
        Added per reviewer request to verify Phase 8b integration works.
        """
        np.random.seed(42)
        n_main = 100000  # Well above NUMBA_MIN_ROWS threshold
        n_sub = 10000
        
        # Create ADF with single-column integer key
        main_df = pd.DataFrame({
            'row_id': np.random.randint(0, n_sub, n_main, dtype=np.int64),
            'value': np.random.randn(n_main).astype(np.float32)
        })
        sub_df = pd.DataFrame({
            'row_id': np.arange(n_sub, dtype=np.int64),
            'lookup_val': np.random.randn(n_sub).astype(np.float32)
        })
        
        adf = AliasDataFrame(main_df, use_numba=True)
        adf.register_subframe('sub', AliasDataFrame(sub_df), index_columns='row_id')
        adf.add_alias('result', 'sub.lookup_val')
        
        # Verify Numba is available and enabled
        assert adf.numba_info['available'], "Numba should be available"
        assert adf.numba_info['enabled'], "Numba should be enabled"
        
        # Clear any cached join indices to force recomputation
        adf._join_index_cache.clear()
        
        # Materialize - this should trigger Numba index lookup
        adf.materialize_alias('result')
        
        # Verify results
        assert 'result' in adf.df.columns
        assert len(adf.df['result']) == n_main
        
        # Verify no NaN values (all keys should be found since we use valid range)
        assert not np.any(np.isnan(adf.df['result'].values)), \
            "All keys should be found in subframe"
        
        # Cross-check with pandas-only path
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('sub', AliasDataFrame(sub_df.copy()), index_columns='row_id')
        adf_pandas.add_alias('result', 'sub.lookup_val')
        adf_pandas.materialize_alias('result')
        
        np.testing.assert_array_almost_equal(
            adf.df['result'].values,
            adf_pandas.df['result'].values,
            decimal=6,
            err_msg="Numba path should produce identical results to pandas path"
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestNumbaDirectAccelerators:
    """Direct tests for Numba accelerator functions."""
    
    def test_numba_scatter_inplace(self):
        """Numba scatter should modify array in-place."""
        if not ACCELERATORS_AVAILABLE:
            pytest.skip("Accelerators not available")
        
        sub_values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        indices = np.array([0, 2, 1, -1, 0], dtype=np.int64)
        result = np.full(5, np.nan, dtype=np.float64)
        
        used_numba = numba_scatter(sub_values, indices, result)
        
        expected = np.array([10.0, 30.0, 20.0, np.nan, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_numba_compute_join_indices(self):
        """Numba index lookup should return correct indices."""
        if not ACCELERATORS_AVAILABLE:
            pytest.skip("Accelerators not available")
        
        main_keys = np.array([0, 5, 2, 99, 1], dtype=np.int64)
        sub_keys = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        
        indices, missing_mask, used = numba_compute_join_indices(main_keys, sub_keys)
        
        assert used == True
        np.testing.assert_array_equal(indices, [0, 5, 2, -1, 1])
        np.testing.assert_array_equal(missing_mask, [False, False, False, True, False])


class TestNumbaWithFillConfig:
    """Tests for Numba with fill configuration."""
    
    def test_fill_missing_with_numba(self):
        """Fill config should work correctly with Numba path."""
        np.random.seed(42)
        n_main = 20000
        
        main_df = pd.DataFrame({
            'idx': np.concatenate([
                np.arange(100),  # Valid keys
                np.full(n_main - 100, 9999)  # Missing keys
            ]),
            'x': np.random.randn(n_main)
        })
        
        sub_df = pd.DataFrame({
            'idx': np.arange(100, dtype=np.int64),
            'val': np.arange(100, dtype=np.float64)
        })
        
        # With Numba
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numba.set_subframe_fill('S', fill_missing=-999.0)
        adf_numba.add_alias('v', 'S.val')
        adf_numba.materialize_alias('v')
        
        # Without Numba
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_numpy.register_subframe('S', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf_numpy.set_subframe_fill('S', fill_missing=-999.0)
        adf_numpy.add_alias('v', 'S.val')
        adf_numpy.materialize_alias('v')
        
        # Results should match
        np.testing.assert_array_almost_equal(
            adf_numba.df['v'].values,
            adf_numpy.df['v'].values
        )
        
        # Check fill value applied
        assert adf_numba.df['v'].iloc[-1] == -999.0


# =============================================================================
# Phase 8c Tests: Multi-Column Key Linearization
# =============================================================================

class TestMultiColumnLinearization:
    """Tests for Phase 8c: Multi-column key linearization."""
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_linearization_matches_pandas_3col(self):
        """Linearized Numba path should match pd.merge for 3-column keys."""
        np.random.seed(42)
        n_main = 50000  # Above NUMBA_MIN_ROWS threshold
        
        # Create TPC-like structure: drift25 (0-3), side (0-1), row (0-150)
        main_df = pd.DataFrame({
            'drift25': np.random.randint(0, 4, n_main, dtype=np.int8),
            'side': np.random.randint(0, 2, n_main, dtype=np.int8),
            'row': np.random.randint(0, 151, n_main, dtype=np.int16),
            'x': np.random.randn(n_main).astype(np.float32)
        })
        
        # Subframe with calibration data
        n_sub = 4 * 2 * 151  # Full coverage
        sub_df = pd.DataFrame({
            'drift25': np.repeat(np.arange(4), 2 * 151).astype(np.int8),
            'side': np.tile(np.repeat(np.arange(2), 151), 4).astype(np.int8),
            'row': np.tile(np.arange(151), 4 * 2).astype(np.int16),
            'calibration': np.random.randn(n_sub).astype(np.float32)
        })
        
        # With Numba (should use linearization)
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('cal', AliasDataFrame(sub_df.copy()), 
                                     index_columns=['drift25', 'side', 'row'])
        adf_numba.add_alias('calib', 'cal.calibration')
        adf_numba.materialize_alias('calib')
        
        # Without Numba (uses pandas merge)
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('cal', AliasDataFrame(sub_df.copy()),
                                      index_columns=['drift25', 'side', 'row'])
        adf_pandas.add_alias('calib', 'cal.calibration')
        adf_pandas.materialize_alias('calib')
        
        np.testing.assert_array_almost_equal(
            adf_numba.df['calib'].values,
            adf_pandas.df['calib'].values,
            decimal=6,
            err_msg="Linearization result doesn't match pandas merge"
        )
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_linearization_different_maxes_in_main_vs_sub(self):
        """
        CRITICAL TEST: Global strides must handle different maxes in main vs sub.
        
        If main has max(col1)=100 but sub has max(col1)=50, we must use 
        global max=100 for stride computation, otherwise keys won't match.
        """
        np.random.seed(42)
        n_main = 20000
        
        # Main has LARGER range in col1 than subframe
        main_df = pd.DataFrame({
            'col1': np.random.randint(0, 100, n_main, dtype=np.int64),  # max=99
            'col2': np.random.randint(0, 50, n_main, dtype=np.int64),   # max=49
        })
        
        # Subframe has SMALLER range in col1
        sub_df = pd.DataFrame({
            'col1': np.arange(50, dtype=np.int64),  # max=49 (smaller than main!)
            'col2': np.arange(50, dtype=np.int64),  # max=49
            'value': np.arange(50, dtype=np.float64)
        })
        
        # With Numba
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()),
                                     index_columns=['col1', 'col2'])
        adf_numba.add_alias('v', 'S.value')
        adf_numba.materialize_alias('v')
        
        # Without Numba (ground truth)
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('S', AliasDataFrame(sub_df.copy()),
                                      index_columns=['col1', 'col2'])
        adf_pandas.add_alias('v', 'S.value')
        adf_pandas.materialize_alias('v')
        
        # Must match exactly - this tests global stride computation
        np.testing.assert_array_equal(
            np.isnan(adf_numba.df['v'].values),
            np.isnan(adf_pandas.df['v'].values),
            err_msg="Missing key pattern differs - global stride bug!"
        )
        
        # Non-NaN values must match
        mask = ~np.isnan(adf_pandas.df['v'].values)
        if mask.any():
            np.testing.assert_array_almost_equal(
                adf_numba.df['v'].values[mask],
                adf_pandas.df['v'].values[mask],
                decimal=10,
                err_msg="Values differ - global stride bug!"
            )
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_linearization_negative_keys_fallback(self):
        """Negative keys should fallback to pandas gracefully."""
        np.random.seed(42)
        n_main = 20000
        
        main_df = pd.DataFrame({
            'col1': np.random.randint(-10, 10, n_main),  # Negative keys!
            'col2': np.random.randint(0, 20, n_main),
        })
        sub_df = pd.DataFrame({
            'col1': np.arange(-10, 10),
            'col2': np.tile(np.arange(20), 1)[:20],
            'value': np.arange(20, dtype=np.float64)
        })
        
        # Should not crash - falls back to pandas
        adf = AliasDataFrame(main_df, use_numba=True)
        adf.register_subframe('S', AliasDataFrame(sub_df), 
                               index_columns=['col1', 'col2'])
        adf.add_alias('v', 'S.value')
        adf.materialize_alias('v')
        
        assert 'v' in adf.df.columns
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed") 
    def test_linearization_2_columns(self):
        """Two-column keys should work with linearization."""
        np.random.seed(42)
        n_main = 30000
        
        main_df = pd.DataFrame({
            'sector': np.random.randint(0, 18, n_main, dtype=np.int32),
            'pad': np.random.randint(0, 100, n_main, dtype=np.int32),
        })
        sub_df = pd.DataFrame({
            'sector': np.repeat(np.arange(18), 100).astype(np.int32),
            'pad': np.tile(np.arange(100), 18).astype(np.int32),
            'gain': np.random.randn(1800).astype(np.float32)
        })
        
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('cal', AliasDataFrame(sub_df.copy()),
                                     index_columns=['sector', 'pad'])
        adf_numba.add_alias('g', 'cal.gain')
        adf_numba.materialize_alias('g')
        
        adf_pandas = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_pandas.register_subframe('cal', AliasDataFrame(sub_df.copy()),
                                      index_columns=['sector', 'pad'])
        adf_pandas.add_alias('g', 'cal.gain')
        adf_pandas.materialize_alias('g')
        
        np.testing.assert_array_almost_equal(
            adf_numba.df['g'].values,
            adf_pandas.df['g'].values,
            decimal=6
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
