"""
Test suite for join index caching (Phase 4 optimization).

These tests verify that:
1. Cache is populated on first subframe access
2. Cache is reused for subsequent columns from same subframe
3. Cache statistics are tracked correctly
4. Cache is cleared after materialize_aliases completes
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestJoinIndexCaching:
    """Tests for join index caching optimization."""
    
    @pytest.fixture
    def setup_with_subframe(self):
        """Create ADF with subframe containing multiple columns."""
        np.random.seed(42)
        main_df = pd.DataFrame({
            'idx': np.random.randint(0, 100, 1000),
            'x': np.random.randn(1000)
        })
        
        sub_df = pd.DataFrame({
            'idx': np.arange(100),
            'val_a': np.random.randn(100),
            'val_b': np.random.randn(100),
            'val_c': np.random.randn(100),
            'val_d': np.random.randn(100),
            'val_e': np.random.randn(100),
        })
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('T', AliasDataFrame(sub_df), index_columns='idx')
        
        return adf
    
    def test_cache_initialized_empty(self, setup_with_subframe):
        """Cache should be empty before any subframe access."""
        adf = setup_with_subframe
        assert adf._join_index_cache == {}
        assert adf._join_cache_hits == 0
        assert adf._join_cache_misses == 0
    
    def test_cache_populated_on_first_access(self, setup_with_subframe):
        """First subframe column access should populate cache."""
        adf = setup_with_subframe
        adf.add_alias('col_a', 'T.val_a')
        adf.materialize_alias('col_a')
        
        assert 'T' in adf._join_index_cache
        cache_entry = adf._join_index_cache['T']
        assert 'indices' in cache_entry
        assert 'missing_mask' in cache_entry
        assert cache_entry['n_rows'] == len(adf.df)
    
    def test_cache_hit_on_subsequent_access(self, setup_with_subframe):
        """Subsequent columns from same subframe should use cache."""
        adf = setup_with_subframe
        
        adf.add_alias('col_a', 'T.val_a')
        adf.add_alias('col_b', 'T.val_b')
        adf.add_alias('col_c', 'T.val_c')
        
        adf._join_cache_hits = 0
        adf._join_cache_misses = 0
        
        adf.materialize_aliases(pattern=r'col_.*')
        
        assert adf._join_cache_misses == 1, f"Expected 1 miss, got {adf._join_cache_misses}"
        assert adf._join_cache_hits == 2, f"Expected 2 hits, got {adf._join_cache_hits}"
    
    def test_cache_cleared_after_materialize_batch(self, setup_with_subframe):
        """Cache should SURVIVE after materialize_aliases completes.
        
        Phase 13.21.ADF: changed from "cache cleared" to "cache preserved".
        materialize_aliases only adds value columns — join key columns are
        unchanged, so cached join indices remain valid. Targeted invalidation
        happens in register_subframe instead of blanket clear.
        """
        adf = setup_with_subframe
        adf.add_alias('col_a', 'T.val_a')
        adf.materialize_aliases(pattern=r'col_.*')
        
        assert 'T' in adf._join_index_cache, (
            "Cache should survive after materialize_aliases (Phase 13.21.ADF). "
            "Join indices are still valid — only value columns were added."
        )
    
    def test_cache_stats_reset_on_new_batch(self, setup_with_subframe):
        """Cache stats should reset at start of each materialize_aliases call.
        
        Phase 13.21.ADF: cache now survives between materialize_aliases calls.
        Stats still reset (counters track per-batch diagnostics), but the
        second call sees cache HITS (not misses) because the join indices
        from the first call are still valid.
        """
        adf = setup_with_subframe
        
        adf.add_alias('col_a', 'T.val_a')
        adf.add_alias('col_b', 'T.val_b')
        adf.materialize_aliases(pattern=r'col_.*')
        
        adf.add_alias('col_c', 'T.val_c')
        adf.add_alias('col_d', 'T.val_d')
        
        adf.materialize_aliases(pattern=r'col_[cd]')
        
        # Phase 13.21.ADF: cache survives from first call.
        # Second call: col_c → HIT (T cached), col_d → HIT (T cached).
        assert adf._join_cache_misses == 0, (
            f"Expected 0 misses (cache survives), got {adf._join_cache_misses}"
        )
        assert adf._join_cache_hits == 2, (
            f"Expected 2 hits (cache survives), got {adf._join_cache_hits}"
        )
    
    def test_multiple_subframes_cached_separately(self):
        """Each subframe should have its own cache entry."""
        np.random.seed(42)
        main_df = pd.DataFrame({
            'idx1': np.random.randint(0, 50, 500),
            'idx2': np.random.randint(0, 50, 500),
            'x': np.random.randn(500)
        })
        
        sub1_df = pd.DataFrame({
            'idx1': np.arange(50),
            'val1': np.random.randn(50)
        })
        
        sub2_df = pd.DataFrame({
            'idx2': np.arange(50),
            'val2': np.random.randn(50)
        })
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('S1', AliasDataFrame(sub1_df), index_columns='idx1')
        adf.register_subframe('S2', AliasDataFrame(sub2_df), index_columns='idx2')
        
        adf.add_alias('from_s1', 'S1.val1')
        adf.add_alias('from_s2', 'S2.val2')
        
        adf._join_cache_hits = 0
        adf._join_cache_misses = 0
        
        adf.materialize_aliases()
        
        assert adf._join_cache_misses == 2
        assert adf._join_cache_hits == 0
    
    def test_cache_produces_correct_values(self, setup_with_subframe):
        """Cached and non-cached paths should produce identical results."""
        adf = setup_with_subframe
        
        sub_adf = adf.get_subframe('T')
        expected = adf.df.merge(
            sub_adf.df[['idx', 'val_a', 'val_b']], 
            on='idx', 
            how='left'
        )
        
        adf.add_alias('col_a', 'T.val_a')
        adf.add_alias('col_b', 'T.val_b')
        adf.materialize_aliases(pattern=r'col_.*')
        
        np.testing.assert_array_almost_equal(
            adf.df['col_a'].values,
            expected['val_a'].values,
            err_msg="Cached col_a values don't match expected"
        )
        np.testing.assert_array_almost_equal(
            adf.df['col_b'].values,
            expected['val_b'].values,
            err_msg="Cached col_b values don't match expected"
        )
    
    def test_cache_handles_missing_keys(self):
        """Cache should correctly handle missing keys with fill config."""
        main_df = pd.DataFrame({
            'idx': [0, 1, 2, 999, 998],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        sub_df = pd.DataFrame({
            'idx': [0, 1, 2],
            'val_a': [10.0, 20.0, 30.0],
            'val_b': [100.0, 200.0, 300.0]
        })
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('T', AliasDataFrame(sub_df), index_columns='idx')
        adf.set_subframe_fill('T', fill_missing=-999.0)
        
        adf.add_alias('col_a', 'T.val_a')
        adf.add_alias('col_b', 'T.val_b')
        adf.materialize_aliases(pattern=r'col_.*')
        
        assert adf.df['col_a'].iloc[3] == -999.0
        assert adf.df['col_a'].iloc[4] == -999.0
        assert adf.df['col_b'].iloc[3] == -999.0
        assert adf.df['col_b'].iloc[4] == -999.0
        
        assert adf.df['col_a'].iloc[0] == 10.0
        assert adf.df['col_b'].iloc[0] == 100.0
    
    def test_five_column_batch_cache_stats(self, setup_with_subframe):
        """Materializing 5 columns from one subframe should show 1 miss, 4 hits."""
        adf = setup_with_subframe
        
        adf.add_alias('col_a', 'T.val_a')
        adf.add_alias('col_b', 'T.val_b')
        adf.add_alias('col_c', 'T.val_c')
        adf.add_alias('col_d', 'T.val_d')
        adf.add_alias('col_e', 'T.val_e')
        
        adf._join_cache_hits = 0
        adf._join_cache_misses = 0
        
        adf.materialize_aliases(pattern=r'col_.*')
        
        assert adf._join_cache_misses == 1, f"Expected 1 miss for 5 columns, got {adf._join_cache_misses}"
        assert adf._join_cache_hits == 4, f"Expected 4 hits for 5 columns, got {adf._join_cache_hits}"
        
        assert adf._join_cache_misses + adf._join_cache_hits == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
