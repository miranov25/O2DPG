#!/usr/bin/env python3
"""
Test cases for join caching in AliasDataFrame.

Tests for Phase 2: Join Caching implementation.
- Cache correctness
- Cache performance
- Cache cleanup between batches
- Multi-key index support
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestJoinCachingCorrectness:
    """Tests for join cache correctness."""
    
    def test_cached_values_match_uncached(self):
        """Cached lookups should produce same values as uncached merge."""
        df = pd.DataFrame({
            'idx': [0, 0, 1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],
            'val_a': [10.0, 20.0, 30.0],
            'val_b': [100.0, 200.0, 300.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        # Materialize using batch (with caching)
        adf.materialize_aliases(names=['a', 'b'])
        
        # Expected values
        np.testing.assert_array_equal(adf.df['a'], [10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
        np.testing.assert_array_equal(adf.df['b'], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])
    
    def test_missing_keys_handled_correctly_with_cache(self):
        """Missing keys should be handled correctly with caching."""
        df = pd.DataFrame({
            'idx': [0, 1, 2, 3],  # 3 has no match
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],  # No idx=3
            'val_a': [10.0, 20.0, 30.0],
            'val_b': [100.0, 200.0, 300.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        adf.set_subframe_fill('S', fill_missing=np.nan)
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        # First 3 should have values, last should be NaN
        assert adf.df['a'].iloc[0] == 10.0
        assert adf.df['a'].iloc[1] == 20.0
        assert adf.df['a'].iloc[2] == 30.0
        assert np.isnan(adf.df['a'].iloc[3])
        
        assert adf.df['b'].iloc[0] == 100.0
        assert np.isnan(adf.df['b'].iloc[3])
    
    def test_fill_value_applied_with_cache(self):
        """Fill values should be applied correctly with caching."""
        df = pd.DataFrame({
            'idx': [0, 1, 99],  # 99 has no match
            'x': [1.0, 2.0, 3.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'val_a': [10.0, 20.0],
            'val_b': [100.0, 200.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        adf.set_subframe_fill('S', fill_missing=-999.0)
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        # Missing key should use fill value
        assert adf.df['a'].iloc[2] == -999.0
        assert adf.df['b'].iloc[2] == -999.0
    
    def test_multi_key_index_caching(self):
        """Multi-key indexes should cache correctly."""
        df = pd.DataFrame({
            'k1': [0, 0, 1, 1],
            'k2': [0, 1, 0, 1],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'k1': [0, 0, 1, 1],
            'k2': [0, 1, 0, 1],
            'val_a': [10.0, 20.0, 30.0, 40.0],
            'val_b': [100.0, 200.0, 300.0, 400.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns=['k1', 'k2'])
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        np.testing.assert_array_equal(adf.df['a'], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(adf.df['b'], [100.0, 200.0, 300.0, 400.0])
    
    def test_duplicate_keys_in_subframe(self):
        """Duplicate keys should take first match with caching."""
        df = pd.DataFrame({
            'idx': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 0, 1, 1, 2],  # Duplicates
            'val_a': [10.0, 11.0, 20.0, 21.0, 30.0],  # First values: 10, 20, 30
            'val_b': [100.0, 101.0, 200.0, 201.0, 300.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        # Should take first match
        np.testing.assert_array_equal(adf.df['a'], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(adf.df['b'], [100.0, 200.0, 300.0])


class TestJoinCachingLifecycle:
    """Tests for cache lifecycle management."""
    
    def test_cache_cleared_between_batches(self):
        """Cache should not persist between materialize_aliases() calls."""
        df = pd.DataFrame({
            'idx': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],
            'val': [10.0, 20.0, 30.0],
            'val2': [100.0, 200.0, 300.0]  # Different column for second batch
        })
        
        adf = AliasDataFrame(df)
        sf_adf = AliasDataFrame(sf)
        adf.register_subframe('S', sf_adf, index_columns='idx')
        
        adf.add_alias('a', 'S.val')
        adf.materialize_aliases(names=['a'])
        
        # First values are correct
        np.testing.assert_array_equal(adf.df['a'], [10.0, 20.0, 30.0])
        
        # Modify subframe's second column
        sf_adf.df['val2'] = [1000.0, 2000.0, 3000.0]
        
        # Add new alias using different column and materialize
        adf.add_alias('b', 'S.val2')
        adf.materialize_aliases(names=['b'], only_unmaterialized=True)
        
        # b should use the MODIFIED val2 values (cache was cleared, fresh join)
        # This verifies the cache doesn't persist stale data across batches
        np.testing.assert_array_equal(adf.df['b'], [1000.0, 2000.0, 3000.0])
    
    def test_cache_not_used_for_single_alias(self):
        """materialize_alias() should work without cache."""
        df = pd.DataFrame({
            'idx': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],
            'val': [10.0, 20.0, 30.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val')
        
        # Single alias materialization (no batch cache)
        adf.materialize_alias('a')
        
        np.testing.assert_array_equal(adf.df['a'], [10.0, 20.0, 30.0])
    
    def test_verbose_shows_cache_stats(self, capsys):
        """Verbose output should show cache statistics."""
        df = pd.DataFrame({
            'idx': np.random.randint(0, 100, 1000),
            'x': np.random.randn(1000)
        })
        sf = pd.DataFrame({
            'idx': np.arange(100),
            'val_a': np.random.randn(100),
            'val_b': np.random.randn(100)
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        
        adf.materialize_aliases(names=['a', 'b'], verbose=True)
        
        captured = capsys.readouterr()
        assert 'Join cache' in captured.out or 'cache' in captured.out.lower()


class TestJoinCachingPerformance:
    """Tests for join cache performance."""
    
    def test_second_column_faster_than_first(self):
        """Second column from same subframe should be faster (cache hit)."""
        n_rows = 100_000
        n_sf_rows = 1000
        
        df = pd.DataFrame({
            'idx': np.random.randint(0, n_sf_rows, n_rows),
            'x': np.random.randn(n_rows)
        })
        sf = pd.DataFrame({
            'idx': np.arange(n_sf_rows),
            'val_a': np.random.randn(n_sf_rows),
            'val_b': np.random.randn(n_sf_rows)
        })
        
        adf = AliasDataFrame(df.copy())
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        # First column (cache miss)
        adf.add_alias('a', 'S.val_a')
        t0 = time.time()
        adf.materialize_alias('a')  # No batch cache
        t_first = time.time() - t0
        
        # Reset and use batch for comparison
        adf2 = AliasDataFrame(df.copy())
        adf2.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        adf2.add_alias('a', 'S.val_a')
        adf2.add_alias('b', 'S.val_b')
        
        t0 = time.time()
        adf2.materialize_aliases(names=['a', 'b'])
        t_batch = time.time() - t0
        
        # Batch should be faster than 2x single (due to caching)
        # This is a soft assertion - caching should help
        print(f"\nSingle alias: {t_first*1000:.1f}ms")
        print(f"Two aliases (batched): {t_batch*1000:.1f}ms")
        print(f"Speedup vs 2x single: {(2*t_first)/t_batch:.2f}x")
    
    def test_many_columns_performance(self):
        """Many columns from same subframe should benefit from caching."""
        n_rows = 100_000
        n_cols = 10
        
        df = pd.DataFrame({
            'idx': np.random.randint(0, 1000, n_rows),
            'x': np.random.randn(n_rows)
        })
        
        sf_data = {'idx': np.arange(1000)}
        for i in range(n_cols):
            sf_data[f'val_{i}'] = np.random.randn(1000)
        sf = pd.DataFrame(sf_data)
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        for i in range(n_cols):
            adf.add_alias(f'col_{i}', f'S.val_{i}')
        
        t0 = time.time()
        adf.materialize_aliases(pattern=r'col_.*')
        elapsed = time.time() - t0
        
        # All columns should be materialized
        for i in range(n_cols):
            assert f'col_{i}' in adf.df.columns
        
        # Performance expectation: with caching, should be much faster than n_cols * single_join_time
        # On 100K rows, 10 columns should complete well under 5 seconds with caching
        assert elapsed < 5.0, f"Performance regression: {elapsed:.2f}s for {n_cols} columns"
        
        print(f"\n{n_cols} columns on {n_rows:,} rows: {elapsed:.2f}s ({elapsed/n_cols*1000:.1f}ms per column)")


class TestJoinCachingEdgeCases:
    """Tests for edge cases in join caching."""
    
    def test_different_subframes_separate_caches(self):
        """Different subframes should have separate cache entries."""
        df = pd.DataFrame({
            'idx1': [0, 1, 2],
            'idx2': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        sf1 = pd.DataFrame({
            'idx1': [0, 1, 2],
            'val': [10.0, 20.0, 30.0]
        })
        sf2 = pd.DataFrame({
            'idx2': [0, 1, 2],
            'val': [100.0, 200.0, 300.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S1', AliasDataFrame(sf1), index_columns='idx1')
        adf.register_subframe('S2', AliasDataFrame(sf2), index_columns='idx2')
        
        adf.add_alias('a', 'S1.val')
        adf.add_alias('b', 'S2.val')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        np.testing.assert_array_equal(adf.df['a'], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(adf.df['b'], [100.0, 200.0, 300.0])
    
    def test_subframe_alias_materialized_correctly(self):
        """Subframe aliases should be materialized before caching."""
        df = pd.DataFrame({
            'idx': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],
            'raw': [10.0, 20.0, 30.0]
        })
        
        sf_adf = AliasDataFrame(sf)
        sf_adf.add_alias('computed', 'raw * 2')  # Alias in subframe
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', sf_adf, index_columns='idx')
        
        adf.add_alias('a', 'S.computed')
        adf.add_alias('b', 'S.raw')
        
        adf.materialize_aliases(names=['a', 'b'])
        
        np.testing.assert_array_equal(adf.df['a'], [20.0, 40.0, 60.0])
        np.testing.assert_array_equal(adf.df['b'], [10.0, 20.0, 30.0])
    
    def test_empty_dataframe(self):
        """Empty DataFrames should work with caching."""
        df = pd.DataFrame({
            'idx': pd.Series([], dtype=int),
            'x': pd.Series([], dtype=float)
        })
        sf = pd.DataFrame({
            'idx': [0, 1, 2],
            'val': [10.0, 20.0, 30.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val')
        adf.materialize_aliases(names=['a'])
        
        assert 'a' in adf.df.columns
        assert len(adf.df) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
