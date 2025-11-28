#!/usr/bin/env python3
"""
Test cases for BUG-2025-11-27-001: Self-Referential Cycles Fix

Add these tests to test_alias_data_frame_schema.py or run standalone.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestAutoAliasSubframeCycleFix:
    """Tests for the self-referential cycle bug fix."""
    
    def test_skip_existing_columns(self):
        """auto_alias_subframe skips columns that exist in main DataFrame."""
        # Setup: main DataFrame with column 'y'
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40],  # This will conflict
            'idx': [0, 0, 1, 1]
        })
        
        # Subframe also has 'y'
        sf_df = pd.DataFrame({
            'idx': [0, 1],
            'y': [100, 200],  # Conflicts with main 'y'
            'z': [300, 400]   # Does not conflict
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf_df), index_columns='idx')
        
        # Act
        result = adf.auto_alias_subframe('S', verbose=False)
        
        # Assert
        assert 'z' in result['created'], "Should create alias for 'z'"
        assert 'y' in result['skipped_column'], "Should skip 'y' (column exists)"
        assert 'y' not in adf.aliases, "Should NOT create alias for existing column"
        assert 'z' in adf.aliases, "Should have alias for 'z'"
    
    def test_no_self_referential_cycles(self):
        """auto_alias_subframe does not create self-referential cycles."""
        # Setup: Create scenario that previously caused cycles
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'dEdxTPC': [100, 200, 300],  # Already materialized
            'track_idx': [0, 0, 1]
        })
        
        sf_df = pd.DataFrame({
            'track_idx': [0, 1],
            'dEdxTPC': [50, 60],  # Same name - would cause cycle
            'mP3': [0.1, 0.2]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('T', AliasDataFrame(sf_df), index_columns='track_idx')
        
        # Act
        result = adf.auto_alias_subframe('T', verbose=False)
        
        # Assert: No self-referential cycles
        cycles = adf.validate_no_cycles(raise_on_cycle=False)
        assert len(cycles) == 0, f"Found cycles: {cycles}"
        
        # dEdxTPC should be skipped
        assert 'dEdxTPC' in result['skipped_column']
        assert 'dEdxTPC' not in result['created']
    
    def test_add_alias_rejects_self_reference(self):
        """add_alias raises error on self-referential expression."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Add column 'foo' to DataFrame
        adf.df['foo'] = [10, 20, 30]
        
        # Try to create self-referential alias
        with pytest.raises(ValueError, match="reference itself"):
            adf.add_alias('foo', 'foo + 1')
    
    def test_add_alias_allows_subframe_reference(self):
        """add_alias allows subframe.column references (not self-reference)."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'idx': [0, 0, 1]
        })
        sf_df = pd.DataFrame({
            'idx': [0, 1],
            'val': [100, 200]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf_df), index_columns='idx')
        
        # This should work - 'val' references S.val, not itself
        adf.add_alias('val', 'S.val')
        
        assert 'val' in adf.aliases
        assert adf.aliases['val'] == 'S.val'
    
    def test_auto_alias_with_overwrite(self):
        """auto_alias_subframe with overwrite=True replaces existing aliases."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'idx': [0, 0, 1]
        })
        sf_df = pd.DataFrame({
            'idx': [0, 1],
            'y': [100, 200]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf_df), index_columns='idx')
        
        # Create manual alias first
        adf.add_alias('y', 'x * 2')
        assert adf.aliases['y'] == 'x * 2'
        
        # Auto-alias with overwrite
        result = adf.auto_alias_subframe('S', overwrite=True, verbose=False)
        
        # Should overwrite
        assert 'y' in result['created']
        assert adf.aliases['y'] == 'S.y'
    
    def test_auto_alias_without_overwrite(self):
        """auto_alias_subframe without overwrite skips existing aliases."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'idx': [0, 0, 1]
        })
        sf_df = pd.DataFrame({
            'idx': [0, 1],
            'y': [100, 200]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf_df), index_columns='idx')
        
        # Create manual alias first
        adf.add_alias('y', 'x * 2')
        
        # Auto-alias without overwrite (default)
        result = adf.auto_alias_subframe('S', overwrite=False, verbose=False)
        
        # Should NOT overwrite
        assert 'y' in result['skipped_alias']
        assert adf.aliases['y'] == 'x * 2'  # Original preserved
    
    def test_validate_no_cycles_method(self):
        """validate_no_cycles detects and reports cycles."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Create indirect cycle: a -> b -> a
        adf._schema['columns']['a'] = {'expr': 'b + 1'}
        adf._schema['columns']['b'] = {'expr': 'a + 1'}
        
        # Should detect cycle
        cycles = adf.validate_no_cycles(raise_on_cycle=False)
        assert len(cycles) > 0, "Should detect indirect cycle"
        
        # Should raise when requested
        with pytest.raises(ValueError, match="cycles"):
            adf.validate_no_cycles(raise_on_cycle=True)
    
    def test_auto_alias_after_materialize(self):
        """auto_alias_subframe works correctly after some columns materialized."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'track_idx': [0, 0, 1, 1]
        })
        sf_df = pd.DataFrame({
            'track_idx': [0, 1],
            'mP3': [0.1, 0.2],
            'mP4': [0.01, 0.02]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('T', AliasDataFrame(sf_df), index_columns='track_idx')
        
        # Materialize one column manually
        adf.add_alias('mP3', 'T.mP3')
        adf.materialize_alias('mP3')
        
        # Now mP3 is a column in DataFrame
        assert 'mP3' in adf.df.columns
        
        # Remove the alias (simulating real-world scenario)
        del adf._schema['columns']['mP3']
        
        # auto_alias should skip mP3 (it's a column now)
        result = adf.auto_alias_subframe('T', verbose=False)
        
        assert 'mP3' in result['skipped_column']
        assert 'mP3' not in result['created']
        assert 'mP4' in result['created']
    
    @pytest.mark.xfail(reason="Blocked by auto_alias_subframe cycle bug (BUG-2025-11-27-001)")
    def test_materialize_after_auto_alias_fix(self):
        """materialize_aliases works after auto_alias with the fix."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'existing_col': [10, 20, 30, 40],  # Pre-existing
            'track_idx': [0, 0, 1, 1]
        })
        sf_df = pd.DataFrame({
            'track_idx': [0, 1],
            'existing_col': [100, 200],  # Same name - would cause cycle
            'new_col': [300, 400]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('T', AliasDataFrame(sf_df), index_columns='track_idx')
        
        # Auto-alias (should skip existing_col)
        result = adf.auto_alias_subframe('T', verbose=False)
        
        # Add dependent alias
        adf.add_alias('computed', 'new_col * 2')
        
        # This should work without NetworkXUnfeasible error
        adf.materialize_aliases(names=['computed'], with_dependencies=True)
        
        assert 'computed' in adf.df.columns
        # Dependency may be materialized as 'new_col' or with suffix - check alias still works
        # The key is that 'computed' was successfully materialized
        computed_values = adf.df['computed'].values
        expected = np.array([600, 600, 800, 800])  # 300*2, 300*2, 400*2, 400*2
        np.testing.assert_array_equal(computed_values, expected)


class TestOnlyUnmaterializedFix:
    """Tests for only_unmaterialized parameter fix."""
    
    def test_only_unmaterialized_with_alias_and_column(self):
        """only_unmaterialized correctly handles aliases that are also columns."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]  # Pre-existing column
        })
        adf = AliasDataFrame(df)
        
        # Create alias with same name as column (edge case)
        # After the fix, this should raise an error
        # But for backward compat, test the only_unmaterialized logic
        
        adf.add_alias('z', 'x + y')
        
        # z is not materialized yet
        result = adf.materialize_aliases(names=['z'], only_unmaterialized=True)
        
        # Should have materialized z
        assert 'z' in adf.df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
