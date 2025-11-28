#!/usr/bin/env python3
"""
Test cases for BUG-2025-11-28-001: Cycle Detection and Index Column Materialization

Tests for:
- Fix A: Self-referential cycle detection (false positives)
- Fix B: Index column materialization in batched path
- Fix C: Better cycle error messages
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestCycleDetection:
    """Tests for dependency cycle detection and error messages."""
    
    def test_self_referential_subframe_alias_no_cycle(self):
        """Auto-generated subframe aliases should not create false cycles.
        
        Bug: alias 'val' with expr 'T.val' extracts token 'val', creating self-loop.
        Fix: Exclude token == alias from dependency graph.
        """
        df = pd.DataFrame({'x': [1, 2, 3], 'idx': [0, 0, 1]})
        sf = pd.DataFrame({'idx': [0, 1], 'val': [10.0, 20.0]})
        
        adf = AliasDataFrame(df)
        adf.register_subframe('T', AliasDataFrame(sf), index_columns='idx')
        
        # This pattern caused the bug: alias name appears in expression
        adf.add_alias('val', 'T.val')
        
        # Should NOT raise cycle error
        result = adf.select_aliases(names=['val'], with_dependencies=True)
        assert 'val' in result
    
    def test_multiple_self_referential_aliases_no_cycle(self):
        """Multiple auto-aliased columns should not create cycles."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'idx': [0, 0, 1, 1]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'col_a': [10.0, 20.0],
            'col_b': [100.0, 200.0],
            'col_c': [1000.0, 2000.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        # Multiple aliases where name == subframe column name
        adf.add_alias('col_a', 'S.col_a')
        adf.add_alias('col_b', 'S.col_b')
        adf.add_alias('col_c', 'S.col_c')
        
        # Should work without cycle error
        result = adf.select_aliases(names=['col_a', 'col_b', 'col_c'], with_dependencies=True)
        assert len(result) == 3
        
        # Should materialize without error
        adf.materialize_aliases(names=['col_a', 'col_b', 'col_c'])
        assert 'col_a' in adf.df.columns
        assert 'col_b' in adf.df.columns
        assert 'col_c' in adf.df.columns
    
    def test_real_cycle_detected_with_message(self):
        """Real cycles should be detected with helpful error message.
        
        Note: Cycles are caught at add_alias() time by _check_for_cycles().
        We test that cycle detection works during alias creation.
        """
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Create first alias (no cycle yet)
        adf.add_alias('a', 'b + 1')
        
        # Adding second alias that creates cycle should be caught
        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf.add_alias('b', 'a + 1')
    
    def test_cycle_error_shows_expressions(self):
        """Cycle error from select_aliases should show expressions.
        
        To test select_aliases cycle detection, we need to bypass add_alias
        cycle checking by manipulating _schema directly.
        """
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Bypass add_alias cycle detection by manipulating schema directly
        adf._schema['columns']['a'] = {'expr': 'b + 1'}
        adf._schema['columns']['b'] = {'expr': 'a + 1'}
        
        # select_aliases should catch it with helpful message
        with pytest.raises(ValueError, match=r"[Cc]ycle"):
            adf.select_aliases(names=['a'], with_dependencies=True)
    
    def test_cycle_error_shows_hint(self):
        """Cycle error should include diagnostic hint."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Bypass add_alias cycle detection
        adf._schema['columns']['a'] = {'expr': 'b + 1'}
        adf._schema['columns']['b'] = {'expr': 'a + 1'}
        
        with pytest.raises(ValueError, match="validate_no_cycles"):
            adf.select_aliases(names=['a'], with_dependencies=True)
    
    def test_three_way_cycle_detected(self):
        """Three-way cycles (a -> b -> c -> a) should be detected."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Bypass add_alias cycle detection
        adf._schema['columns']['a'] = {'expr': 'c + 1'}
        adf._schema['columns']['b'] = {'expr': 'a + 1'}
        adf._schema['columns']['c'] = {'expr': 'b + 1'}
        
        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf.select_aliases(names=['a'], with_dependencies=True)
    
    def test_non_cycle_chain_works(self):
        """Linear dependency chains should work correctly."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        # Chain: x -> a -> b -> c (no cycle)
        adf.add_alias('a', 'x * 2')
        adf.add_alias('b', 'a + 10')
        adf.add_alias('c', 'b * 3')
        
        result = adf.select_aliases(names=['c'], with_dependencies=True)
        assert result == ['a', 'b', 'c']  # Topological order
        
        adf.materialize_aliases(names=['c'], with_dependencies=True, cleanTemporary=False)
        np.testing.assert_array_equal(adf.df['c'], [36.0, 42.0, 48.0])


class TestIndexColumnMaterialization:
    """Tests for index column materialization in batched path."""
    
    def test_alias_index_column_materialized_in_batch(self):
        """Index columns that are aliases should be materialized before join.
        
        Bug: materialize_aliases() bypasses materialize_alias() and calls
        _eval_in_namespace() directly, skipping index column materialization.
        """
        df = pd.DataFrame({
            'raw_idx': [0, 0, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'scale': [10.0, 100.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        # Index column is an alias
        adf.add_alias('idx', 'raw_idx')
        adf.add_alias('scaled', 'x * S.scale')
        
        # Should work - idx should be materialized before join
        adf.materialize_aliases(names=['scaled'], with_dependencies=True)
        
        assert 'scaled' in adf.df.columns
        assert 'idx' in adf.df.columns  # Index was materialized
        
        # Values should be correct
        expected = np.array([10.0, 20.0, 300.0, 400.0])  # x * scale
        np.testing.assert_array_equal(adf.df['scaled'], expected)
    
    def test_batch_path_matches_single_path(self):
        """Batched materialization should produce same result as single."""
        df = pd.DataFrame({
            'raw_idx': [0, 0, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'scale': [10.0, 100.0]
        })
        
        # Test single path
        adf1 = AliasDataFrame(df.copy())
        adf1.register_subframe('S', AliasDataFrame(sf.copy()), index_columns='idx')
        adf1.add_alias('idx', 'raw_idx')
        adf1.add_alias('scaled', 'x * S.scale')
        adf1.materialize_alias('idx')
        adf1.materialize_alias('scaled')
        
        # Test batch path
        adf2 = AliasDataFrame(df.copy())
        adf2.register_subframe('S', AliasDataFrame(sf.copy()), index_columns='idx')
        adf2.add_alias('idx', 'raw_idx')
        adf2.add_alias('scaled', 'x * S.scale')
        adf2.materialize_aliases(names=['scaled'], with_dependencies=True)
        
        # Results should match
        np.testing.assert_array_equal(adf1.df['scaled'], adf2.df['scaled'])
    
    def test_multi_key_index_aliases_materialized(self):
        """Multiple index columns that are aliases should all be materialized."""
        df = pd.DataFrame({
            'raw_a': [0, 0, 1, 1],
            'raw_b': [0, 1, 0, 1],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx_a': [0, 0, 1, 1],
            'idx_b': [0, 1, 0, 1],
            'val': [10.0, 20.0, 30.0, 40.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns=['idx_a', 'idx_b'])
        
        # Both index columns are aliases
        adf.add_alias('idx_a', 'raw_a')
        adf.add_alias('idx_b', 'raw_b')
        adf.add_alias('result', 'x + S.val')
        
        # Should work - both indices should be materialized
        adf.materialize_aliases(names=['result'], with_dependencies=True)
        
        assert 'idx_a' in adf.df.columns
        assert 'idx_b' in adf.df.columns
        assert 'result' in adf.df.columns
        
        # Values should be correct
        expected = np.array([11.0, 22.0, 33.0, 44.0])
        np.testing.assert_array_equal(adf.df['result'], expected)
    
    def test_computed_index_column(self):
        """Index column computed from expression should be materialized."""
        df = pd.DataFrame({
            'a': [0, 0, 1, 1],
            'b': [0, 0, 0, 0],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'scale': [10.0, 100.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        # Index is computed expression
        adf.add_alias('idx', 'a + b')
        adf.add_alias('scaled', 'x * S.scale')
        
        adf.materialize_aliases(names=['scaled'], with_dependencies=True)
        
        assert 'idx' in adf.df.columns
        assert 'scaled' in adf.df.columns
        
        expected = np.array([10.0, 20.0, 300.0, 400.0])
        np.testing.assert_array_equal(adf.df['scaled'], expected)


class TestBatchOptimization:
    """Tests to verify batch optimization is working."""
    
    def test_verbose_output_shows_batch_operations(self, capsys):
        """Verbose output should show batch-added message."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        
        adf.add_alias('y', 'x * 2')
        adf.add_alias('z', 'y + 1')
        
        adf.materialize_aliases(names=['z'], with_dependencies=True, verbose=True)
        
        captured = capsys.readouterr()
        assert 'Batch-added' in captured.out
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
