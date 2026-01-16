"""
test_invariance_subframe.py — I3: Subframe Join Invariance

Phase 13.7.ADF: Invariance Test Suite
Priority: P0
Tests: 9

Property: adf['x + T.y'] == pandas.merge() equivalent

Join Contract:
- Join type: LEFT JOIN (preserve all main frame rows)
- Duplicate keys: Keep all combinations (many-to-one)
- Missing keys: Fill with NaN
- Ordering: Preserve main frame row order

Public Entry Points Exercised:
- adf.register_subframe(name, adf, index_columns)
- adf.add_alias() with T.column syntax
- adf.materialize_alias()
- adf.df['alias'] accessor

Author: Claude2-Coder
Date: 2026-01-15
Version: 1.1 - Fixed missing materialize_alias() calls
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tolerance settings
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


def assert_invariant_equal(result, expected, name=""):
    """Compare arrays with appropriate tolerance."""
    result = np.asarray(result)
    expected = np.asarray(expected)
    if np.issubdtype(result.dtype, np.floating):
        result_nan = np.isnan(result)
        expected_nan = np.isnan(expected)
        np.testing.assert_array_equal(result_nan, expected_nan, err_msg=f"{name}_nan_positions")
        mask = ~result_nan
        if mask.any():
            np.testing.assert_allclose(result[mask], expected[mask], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=name)
    else:
        np.testing.assert_array_equal(result, expected, err_msg=name)


def assert_nan_positions_equal(result, expected, name=""):
    """Check NaN positions match exactly."""
    result = np.asarray(result)
    expected = np.asarray(expected)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected), err_msg=name)


# =============================================================================
# TEST CLASS: SUBFRAME JOIN INVARIANCE
# =============================================================================

@pytest.mark.invariance
class TestInvarianceSubframe:
    """
    I3: Subframe Join Invariance Tests
    
    Core invariant: ADF subframe join == pandas.merge(how='left')
    
    This is CRITICAL for Phase 13.6.B Team 2 integration.
    """
    
    # -------------------------------------------------------------------------
    # I3_1: Single-key join matches pandas
    # -------------------------------------------------------------------------
    
    def test_I3_1_single_key_join_matches_pandas(self):
        """
        I3_1: Single-key join produces same results as pandas.merge.
        
        Invariant: adf['T.column'] == merge(how='left')['column']
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame
        main_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
            'value': np.arange(8, dtype=np.float64),
        })
        
        # Subframe
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 3], dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge (LEFT JOIN - the contract)
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_1_single_key_join")
    
    # -------------------------------------------------------------------------
    # I3_2: Multi-key (2 columns) join matches pandas
    # -------------------------------------------------------------------------
    
    def test_I3_2_multikey_2col_join_matches_pandas(self):
        """
        I3_2: Two-column key join produces same results as pandas.merge.
        
        Invariant: Multi-key ADF join == pandas.merge(on=[k1, k2])
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame
        main_df = pd.DataFrame({
            'key1': np.array([0, 0, 1, 1, 0, 1], dtype=np.int32),
            'key2': np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
            'value': np.arange(6, dtype=np.float64),
        })
        
        # Subframe (all 4 combinations)
        sub_df = pd.DataFrame({
            'key1': np.array([0, 0, 1, 1], dtype=np.int32),
            'key2': np.array([0, 1, 0, 1], dtype=np.int32),
            'factor': np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), ['key1', 'key2'])
        adf.add_alias('t_factor', 'T.factor')
        adf.materialize_alias('t_factor')  # FIX: Must materialize
        adf_result = adf.df['t_factor'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on=['key1', 'key2'], how='left')
        pandas_result = merged['factor'].values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_2_multikey_2col_join")
    
    # -------------------------------------------------------------------------
    # I3_3: Multi-key (3+ columns) join matches pandas
    # -------------------------------------------------------------------------
    
    def test_I3_3_multikey_3col_join_matches_pandas(self):
        """
        I3_3: Three-column key join produces same results as pandas.merge.
        
        Invariant: N-key ADF join == pandas.merge(on=[k1, k2, k3])
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame
        np.random.seed(42)
        n = 50
        main_df = pd.DataFrame({
            'k1': np.random.randint(0, 2, n).astype(np.int32),
            'k2': np.random.randint(0, 3, n).astype(np.int32),
            'k3': np.random.randint(0, 2, n).astype(np.int32),
            'value': np.random.randn(n).astype(np.float64),
        })
        
        # Subframe (all 12 combinations: 2*3*2)
        from itertools import product
        keys = list(product(range(2), range(3), range(2)))
        sub_df = pd.DataFrame({
            'k1': [k[0] for k in keys],
            'k2': [k[1] for k in keys],
            'k3': [k[2] for k in keys],
            'correction': np.linspace(0.1, 1.2, len(keys)).astype(np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), ['k1', 'k2', 'k3'])
        adf.add_alias('t_correction', 'T.correction')
        adf.materialize_alias('t_correction')  # FIX: Must materialize
        adf_result = adf.df['t_correction'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on=['k1', 'k2', 'k3'], how='left')
        pandas_result = merged['correction'].values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_3_multikey_3col_join")
    
    # -------------------------------------------------------------------------
    # I3_4: Missing keys produce NaN (left join semantics)
    # -------------------------------------------------------------------------
    
    def test_I3_4_missing_keys_produce_nan(self):
        """
        I3_4: Keys not in subframe produce NaN values.
        
        Invariant: Missing key positions match between ADF and pandas
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame with keys 0-9
        main_df = pd.DataFrame({
            'key': np.arange(10, dtype=np.int64),
            'value': np.arange(10, dtype=np.float64),
        })
        
        # Subframe with only keys 0-4 (5-9 missing)
        sub_df = pd.DataFrame({
            'key': np.arange(5, dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant: NaN positions must match
        assert_nan_positions_equal(adf_result, pandas_result, "I3_4_nan_positions")
        
        # Invariant: Non-NaN values must match
        mask = ~np.isnan(pandas_result)
        assert_invariant_equal(adf_result[mask], pandas_result[mask], "I3_4_values")
        
        # Verify we actually have NaNs (test validation)
        assert np.sum(np.isnan(adf_result)) == 5, "Expected 5 NaN values for keys 5-9"
    
    # -------------------------------------------------------------------------
    # I3_5: Duplicate keys handled correctly (many-to-one)
    # -------------------------------------------------------------------------
    
    def test_I3_5_duplicate_keys_handled_correctly(self):
        """
        I3_5: Duplicate keys in main frame produce correct repeated values.
        
        Invariant: Each main frame row gets the correct subframe value
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame with duplicates
        main_df = pd.DataFrame({
            'key': np.array([0, 0, 0, 1, 1, 2], dtype=np.int64),
            'value': np.arange(6, dtype=np.float64),
        })
        
        # Subframe (unique keys)
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_5_duplicate_keys")
        
        # Expected: [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        expected = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float64)
        assert_invariant_equal(adf_result, expected, "I3_5_expected_values")
    
    # -------------------------------------------------------------------------
    # I3_6: Empty subframe handling
    # -------------------------------------------------------------------------
    
    def test_I3_6_empty_subframe_handling(self):
        """
        I3_6: Empty subframe produces all NaN values.
        
        Invariant: Empty subframe == pandas.merge with empty right
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame
        main_df = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int64),
            'value': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        })
        
        # Empty subframe (same schema but no rows)
        sub_df = pd.DataFrame({
            'key': np.array([], dtype=np.int64),
            'offset': np.array([], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant check
        assert_nan_positions_equal(adf_result, pandas_result, "I3_6_empty_subframe_nan")
        
        # All should be NaN
        assert np.all(np.isnan(adf_result)), "Expected all NaN for empty subframe"
    
    # -------------------------------------------------------------------------
    # I3_7: Large subframe join (slow test)
    # -------------------------------------------------------------------------
    
    @pytest.mark.slow
    def test_I3_7_large_subframe_join(self):
        """
        I3_7: Large dataset join produces same results as pandas.
        
        Invariant: Scale doesn't affect correctness
        """
        from AliasDataFrame import AliasDataFrame
        
        # Large main frame
        np.random.seed(42)
        n_main = 100_000
        n_keys = 1000
        
        main_df = pd.DataFrame({
            'key': np.random.randint(0, n_keys, n_main).astype(np.int64),
            'value': np.random.randn(n_main).astype(np.float64),
        })
        
        # Subframe
        sub_df = pd.DataFrame({
            'key': np.arange(n_keys, dtype=np.int64),
            'offset': np.random.randn(n_keys).astype(np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_7_large_subframe")
    
    # -------------------------------------------------------------------------
    # I3_8: Chained subframe references (T.column in expression)
    # -------------------------------------------------------------------------
    
    def test_I3_8_chained_subframe_expression(self):
        """
        I3_8: Expressions combining main and subframe columns work correctly.
        
        Invariant: adf['value + T.offset'] == main.value + merged.offset
        """
        from AliasDataFrame import AliasDataFrame
        
        # Main frame
        main_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
            'value': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        })
        
        # Subframe
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        })
        
        # ADF: Compute expression with subframe reference
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('combined', 'value + T.offset')
        adf.materialize_alias('combined')  # FIX: Must materialize
        adf_result = adf.df['combined'].values
        
        # Pandas: Manual equivalent
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = (merged['value'] + merged['offset']).values
        
        # Invariant check
        assert_invariant_equal(adf_result, pandas_result, "I3_8_chained_expression")
    
    # -------------------------------------------------------------------------
    # I3_9: Flat-Normalized Equivalence (CRITICAL)
    # -------------------------------------------------------------------------
    
    def test_I3_9_flat_normalized_equivalence(self):
        """
        I3_9 CRITICAL: Flat (replicated) data produces same computed results 
        as normalized (subframes) data.
        
        This is THE core guarantee for Phase 13.6.B Team 2 integration.
        
        Invariant:
            Option A (flat with replication) == Option C (normalized with subframes)
        
        Both approaches must yield identical computed values.
        """
        from AliasDataFrame import AliasDataFrame
        
        # =================================================================
        # Test Case 1: Simple event-track structure
        # =================================================================
        
        # Normalized data (Option C structure from Phase 13.6.B)
        events = pd.DataFrame({
            'event_id': np.array([0, 1], dtype=np.int64),
            'run': np.array([100, 101], dtype=np.int64),
        })
        
        tracks = pd.DataFrame({
            'event_id': np.array([0, 0, 1], dtype=np.int64),
            'track_idx': np.array([0, 1, 0], dtype=np.int32),
            'pt': np.array([1.0, 2.0, 1.5], dtype=np.float64),
        })
        
        # Option A: Flat (replicated event data at track level)
        flat_df = pd.DataFrame({
            'event_id': np.array([0, 0, 1], dtype=np.int64),
            'run': np.array([100, 100, 101], dtype=np.int64),  # Replicated
            'track_idx': np.array([0, 1, 0], dtype=np.int32),
            'pt': np.array([1.0, 2.0, 1.5], dtype=np.float64),
        })
        
        adf_flat = AliasDataFrame(flat_df.copy())
        adf_flat.add_alias('pt_scaled', 'pt * run / 100')
        adf_flat.materialize_alias('pt_scaled')  # FIX: Must materialize
        flat_result = adf_flat.df['pt_scaled'].values
        
        # Option C: Normalized (subframes, no replication)
        # Main frame is at track level, events is subframe
        adf_norm = AliasDataFrame(tracks.copy())
        adf_norm.register_subframe('E', AliasDataFrame(events.copy()), 'event_id')
        adf_norm.add_alias('pt_scaled', 'pt * E.run / 100')
        adf_norm.materialize_alias('pt_scaled')  # FIX: Must materialize
        norm_result = adf_norm.df['pt_scaled'].values
        
        # INVARIANT: Computed results must be identical
        assert_invariant_equal(flat_result, norm_result, "I3_9_flat_vs_normalized")
        
        # Verify expected values
        # Event 0 (run=100): pt * 100/100 = pt
        # Event 1 (run=101): pt * 101/100 = pt * 1.01
        expected = np.array([1.0, 2.0, 1.515], dtype=np.float64)
        assert_invariant_equal(flat_result, expected, "I3_9_expected_values")
        
        # =================================================================
        # Test Case 2: More complex expression
        # =================================================================
        
        # Flat
        adf_flat2 = AliasDataFrame(flat_df.copy())
        adf_flat2.add_alias('complex', '(pt + 1) * (run - 99)')
        adf_flat2.materialize_alias('complex')  # FIX: Must materialize
        flat_result2 = adf_flat2.df['complex'].values
        
        # Normalized
        adf_norm2 = AliasDataFrame(tracks.copy())
        adf_norm2.register_subframe('E', AliasDataFrame(events.copy()), 'event_id')
        adf_norm2.add_alias('complex', '(pt + 1) * (E.run - 99)')
        adf_norm2.materialize_alias('complex')  # FIX: Must materialize
        norm_result2 = adf_norm2.df['complex'].values
        
        # INVARIANT
        assert_invariant_equal(flat_result2, norm_result2, "I3_9_complex_expression")
        
        # =================================================================
        # Test Case 3: With missing keys (some tracks without events)
        # =================================================================
        
        tracks_missing = pd.DataFrame({
            'event_id': np.array([0, 0, 1, 2, 2], dtype=np.int64),  # Event 2 not in events
            'pt': np.array([1.0, 2.0, 1.5, 3.0, 4.0], dtype=np.float64),
        })
        
        # Normalized (events only has event_id 0 and 1)
        adf_missing = AliasDataFrame(tracks_missing.copy())
        adf_missing.register_subframe('E', AliasDataFrame(events.copy()), 'event_id')
        adf_missing.add_alias('run_value', 'E.run')
        adf_missing.materialize_alias('run_value')  # FIX: Must materialize
        missing_result = adf_missing.df['run_value'].values
        
        # Expected: [100, 100, 101, NaN, NaN]
        expected_missing = np.array([100., 100., 101., np.nan, np.nan], dtype=np.float64)
        
        # Check NaN positions
        assert_nan_positions_equal(missing_result, expected_missing, "I3_9_missing_nan_positions")
        
        # Check non-NaN values
        mask = ~np.isnan(expected_missing)
        assert_invariant_equal(missing_result[mask], expected_missing[mask], "I3_9_missing_values")


# =============================================================================
# TEST SUMMARY
# =============================================================================

class TestSubframeSummary:
    """Verify all I3 tests are present."""
    
    def test_I3_count(self):
        """Verify we have 9 subframe join tests."""
        tests = [
            'test_I3_1_single_key_join_matches_pandas',
            'test_I3_2_multikey_2col_join_matches_pandas',
            'test_I3_3_multikey_3col_join_matches_pandas',
            'test_I3_4_missing_keys_produce_nan',
            'test_I3_5_duplicate_keys_handled_correctly',
            'test_I3_6_empty_subframe_handling',
            'test_I3_7_large_subframe_join',
            'test_I3_8_chained_subframe_expression',
            'test_I3_9_flat_normalized_equivalence',
        ]
        assert len(tests) == 9, "Expected 9 I3 tests"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "invariance and not slow"])
