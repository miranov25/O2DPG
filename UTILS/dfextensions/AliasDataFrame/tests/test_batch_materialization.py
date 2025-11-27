#!/usr/bin/env python3
"""
Test cases for BUG-2025-11-27-002: materialize_aliases() Performance Fix

Tests for batch materialization optimization.
"""

import pytest
import pandas as pd
import numpy as np
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestBatchMaterializationPerformance:
    """Tests for the batch materialization performance optimization."""
    
    def test_batch_materialize_faster_than_threshold(self):
        """Batch materialization of 50 aliases on 1M rows should complete in <10s."""
        df = pd.DataFrame({'x': np.random.randn(1_000_000)})
        adf = AliasDataFrame(df)
        
        # Add 50 aliases
        for i in range(50):
            adf.add_alias(f'col_{i}', f'x * {i}')
        
        t0 = time.time()
        result = adf.materialize_aliases(pattern=r'col_.*')
        elapsed = time.time() - t0
        
        assert len(result) == 50, f"Expected 50 aliases, got {len(result)}"
        assert elapsed < 10.0, f"Took {elapsed:.1f}s, expected <10s"
        
        # Verify columns exist
        for i in range(50):
            assert f'col_{i}' in adf.df.columns
    
    def test_no_fragmentation_warning(self):
        """Batch materialization should not trigger pandas PerformanceWarning."""
        df = pd.DataFrame({'x': np.random.randn(100_000)})
        adf = AliasDataFrame(df)
        
        # Add many aliases
        for i in range(30):
            adf.add_alias(f'col_{i}', f'x * {i}')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(pattern=r'col_.*')
            
            # Check for fragmentation warnings
            frag_warnings = [x for x in w if 'fragmented' in str(x.message).lower()]
            assert len(frag_warnings) == 0, f"Got fragmentation warning: {frag_warnings}"
    
    def test_batch_correctness_simple(self):
        """Batch results should be correct for simple aliases."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [10.0, 20.0, 30.0, 40.0]
        })
        adf = AliasDataFrame(df)
        
        adf.add_alias('sum', 'x + y')
        adf.add_alias('prod', 'x * y')
        adf.add_alias('diff', 'x - y')
        
        adf.materialize_aliases(names=['sum', 'prod', 'diff'])
        
        np.testing.assert_array_equal(adf.df['sum'], [11.0, 22.0, 33.0, 44.0])
        np.testing.assert_array_equal(adf.df['prod'], [10.0, 40.0, 90.0, 160.0])
        np.testing.assert_array_equal(adf.df['diff'], [-9.0, -18.0, -27.0, -36.0])
    
    def test_batch_with_dependencies(self):
        """Batch materialization should handle alias dependencies correctly."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0]})
        adf = AliasDataFrame(df)
        
        # Create dependency chain: x -> a -> b -> c
        adf.add_alias('a', 'x * 2')      # a = x * 2
        adf.add_alias('b', 'a + 10')     # b = a + 10 = x * 2 + 10
        adf.add_alias('c', 'b * 3')      # c = b * 3 = (x * 2 + 10) * 3
        
        # Materialize only 'c' with dependencies
        result = adf.materialize_aliases(names=['c'], with_dependencies=True, cleanTemporary=False)
        
        # All should be materialized
        assert 'a' in adf.df.columns
        assert 'b' in adf.df.columns
        assert 'c' in adf.df.columns
        
        # Verify values
        np.testing.assert_array_equal(adf.df['a'], [2.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_equal(adf.df['b'], [12.0, 14.0, 16.0, 18.0])
        np.testing.assert_array_equal(adf.df['c'], [36.0, 42.0, 48.0, 54.0])
    
    def test_batch_with_clean_temporary(self):
        """cleanTemporary should batch-drop intermediate columns."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0]})
        adf = AliasDataFrame(df)
        
        # Create dependency chain
        adf.add_alias('temp1', 'x * 2')
        adf.add_alias('temp2', 'temp1 + 10')
        adf.add_alias('final', 'temp2 * 3')
        
        # Materialize only 'final' with cleanTemporary=True
        result = adf.materialize_aliases(names=['final'], with_dependencies=True, cleanTemporary=True)
        
        # Only 'final' should remain (temps cleaned up)
        assert 'final' in adf.df.columns
        assert 'temp1' not in adf.df.columns
        assert 'temp2' not in adf.df.columns
        
        # Value should still be correct
        np.testing.assert_array_equal(adf.df['final'], [36.0, 42.0, 48.0, 54.0])
    
    def test_batch_with_subframe(self):
        """Batch materialization should work with subframe references."""
        df_main = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'idx': [0, 0, 1, 1]
        })
        df_sub = pd.DataFrame({
            'idx': [0, 1],
            'scale': [10.0, 100.0]
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns='idx')
        
        adf.add_alias('scaled', 'x * S.scale')
        
        adf.materialize_aliases(names=['scaled'])
        
        # idx=0 -> scale=10, idx=1 -> scale=100
        np.testing.assert_array_equal(adf.df['scaled'], [10.0, 20.0, 300.0, 400.0])
    
    def test_context_override_enables_chained_deps(self):
        """context_override should allow B to see A before A is in DataFrame."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        # A -> B -> C chain where each depends on previous
        adf.add_alias('A', 'x + 1')
        adf.add_alias('B', 'A * 2')
        adf.add_alias('C', 'B + A')  # Depends on both A and B
        
        # Materialize C with dependencies
        adf.materialize_aliases(names=['C'], with_dependencies=True, cleanTemporary=False)
        
        # Verify all values
        np.testing.assert_array_equal(adf.df['A'], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(adf.df['B'], [4.0, 6.0, 8.0])
        np.testing.assert_array_equal(adf.df['C'], [6.0, 9.0, 12.0])  # B + A
    
    def test_verbose_output(self, capsys):
        """verbose=True should print progress information."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        adf.add_alias('y', 'x * 2')
        adf.add_alias('z', 'y + 1')
        
        adf.materialize_aliases(names=['z'], with_dependencies=True, verbose=True)
        
        captured = capsys.readouterr()
        assert 'materialize_aliases' in captured.out
        assert 'Batch-added' in captured.out
    
    def test_dtype_preserved_in_batch(self):
        """Alias dtypes should be preserved during batch materialization."""
        df = pd.DataFrame({'x': [1.5, 2.5, 3.5]})
        adf = AliasDataFrame(df)
        
        adf.add_alias('y', 'x * 2', dtype='float32')
        adf.add_alias('z', 'x * 3', dtype='int32')
        
        adf.materialize_aliases(names=['y', 'z'])
        
        assert adf.df['y'].dtype == np.float32
        assert adf.df['z'].dtype == np.int32
    
    def test_empty_targets_returns_empty_list(self):
        """Empty targets should return empty list without error."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        result = adf.materialize_aliases(pattern=r'nonexistent.*')
        assert result == []


class TestEvalInNamespaceContextOverride:
    """Tests for the context_override parameter in _eval_in_namespace."""
    
    def test_context_override_provides_values(self):
        """context_override should provide values for evaluation."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        # Provide 'y' via context_override
        context = {'y': pd.Series([10.0, 20.0, 30.0])}
        result = adf._eval_in_namespace('x + y', context_override=context)
        
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])
    
    def test_context_override_shadows_columns(self):
        """context_override should shadow existing columns."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [100.0, 200.0, 300.0]  # Will be shadowed
        })
        adf = AliasDataFrame(df)
        
        # Override 'y' with different values
        context = {'y': pd.Series([10.0, 20.0, 30.0])}
        result = adf._eval_in_namespace('x + y', context_override=context)
        
        # Should use context_override values, not DataFrame column
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
