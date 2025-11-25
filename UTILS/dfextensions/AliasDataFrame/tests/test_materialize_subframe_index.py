"""
Test for automatic materialization of subframe index columns.

Bug: When materializing an alias that references a subframe, if the subframe's
index columns are aliases in the main frame, they weren't being materialized
automatically, causing KeyError during the pandas merge operation.
"""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from AliasDataFrame import AliasDataFrame


class TestSubframeIndexMaterialization:
    """Test automatic materialization of subframe index columns."""
    
    def test_materialize_with_alias_index_columns(self):
        """
        Test that materializing an alias with subframe reference auto-materializes
        the subframe's index columns if they're aliases.
        
        This is the exact bug scenario from the user's report.
        """
        # Create main frame with physical columns
        main_df = pd.DataFrame({
            'sec': np.array([0, 9, 18, 0], dtype=np.uint8),
            'z': np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32),
            'cosPhi': np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32),
            'sinPhi': np.array([0.0, 1.0, 0.0, 0.866], dtype=np.float32)
        })
        adf = AliasDataFrame(main_df)
        
        # Add aliases for index columns (these need to be materialized for join)
        adf.add_alias('drift25', 'int((1-abs(z)/250.)*25)', dtype=np.int8)
        adf.add_alias('side', 'sec>=18', dtype=np.int8)
        
        # Create subframe with index on the aliases
        sub_df = pd.DataFrame({
            'drift25': np.array([10, 20], dtype=np.int8),
            'side': np.array([0, 1], dtype=np.int8),
            'slope_cosPhi': np.array([0.1, 0.2], dtype=np.float32),
            'slope_sinPhi': np.array([0.05, 0.15], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        
        # Register subframe with alias index columns
        adf.register_subframe('DITS0FitSide', sub_adf, index_columns=['drift25', 'side'])
        
        # Create auto-aliases for subframe (no prefix for this test)
        adf.auto_alias_subframe('DITS0FitSide', validate=False)
        
        # Add alias that depends on subframe + main frame aliases
        adf.add_alias('AlignG1', 
                     'slope_cosPhi*cosPhi+slope_sinPhi*sinPhi',
                     dtype=np.float16)
        
        # Verify index columns are NOT yet materialized
        assert 'drift25' not in adf.df.columns
        assert 'side' not in adf.df.columns
        
        # THE FIX: This should auto-materialize drift25 and side before attempting join
        adf.materialize_alias('AlignG1')
        
        # Verify index columns were auto-materialized
        assert 'drift25' in adf.df.columns, "drift25 should be auto-materialized"
        assert 'side' in adf.df.columns, "side should be auto-materialized"
        
        # Verify the target alias was materialized successfully
        assert 'AlignG1' in adf.df.columns
        assert not np.all(np.isnan(adf.df['AlignG1']))  # Should have some non-NaN values
    
    def test_materialize_with_multi_key_alias_index(self):
        """Test with multiple index columns that are aliases."""
        # Main frame
        main_df = pd.DataFrame({
            'row': np.array([50, 100, 150], dtype=np.uint8),
            'sec': np.array([0, 9, 18], dtype=np.uint8),
            'z': np.array([100.0, -100.0, 0.0], dtype=np.float32)
        })
        adf = AliasDataFrame(main_df)
        
        # Index columns as aliases
        adf.add_alias('side', 'sec>=18', dtype=np.int8)
        adf.add_alias('row_group', 'row//50', dtype=np.int8)
        adf.add_alias('drift_bin', 'int((1-abs(z)/250.)*10)', dtype=np.int8)
        
        # Subframe
        sub_df = pd.DataFrame({
            'side': np.array([0, 1], dtype=np.int8),
            'row_group': np.array([1, 2], dtype=np.int8),
            'drift_bin': np.array([8, 10], dtype=np.int8),
            'correction': np.array([0.1, 0.2], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        
        # Register with 3-key alias index
        adf.register_subframe('Calib', sub_adf, 
                             index_columns=['side', 'row_group', 'drift_bin'])
        adf.auto_alias_subframe('Calib', validate=False)
        
        # Add alias using subframe
        adf.add_alias('corrected', 'row + correction', dtype=np.float32)
        
        # None should be materialized yet
        assert 'side' not in adf.df.columns
        assert 'row_group' not in adf.df.columns
        assert 'drift_bin' not in adf.df.columns
        
        # Materialize - should auto-materialize all 3 index columns
        adf.materialize_alias('corrected')
        
        # All index columns should now be materialized
        assert 'side' in adf.df.columns
        assert 'row_group' in adf.df.columns
        assert 'drift_bin' in adf.df.columns
        assert 'corrected' in adf.df.columns
    
    def test_materialize_chain_with_subframe_index(self):
        """
        Test dependency chain: alias1 -> alias2 -> subframe (with alias index).
        
        This tests that recursive materialization handles subframe index aliases.
        """
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'sec': np.array([0, 9, 18], dtype=np.uint8)
        })
        adf = AliasDataFrame(main_df)
        
        # Dependency chain
        adf.add_alias('side', 'sec>=18', dtype=np.int8)  # Index for subframe
        adf.add_alias('x2', 'x**2', dtype=np.float32)    # Intermediate alias
        
        # Subframe
        sub_df = pd.DataFrame({
            'side': np.array([0, 1], dtype=np.int8),
            'factor': np.array([1.5, 2.0], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('SF', sub_adf, index_columns='side')
        adf.auto_alias_subframe('SF', validate=False)
        
        # Top-level alias: depends on x2 (alias) and factor (subframe with alias index)
        adf.add_alias('result', 'x2 * factor', dtype=np.float32)
        
        # Nothing materialized yet
        assert 'side' not in adf.df.columns
        assert 'x2' not in adf.df.columns
        assert 'result' not in adf.df.columns
        
        # Materialize top-level - should cascade to x2 AND side
        adf.materialize_alias('result')
        
        # All dependencies should be materialized
        assert 'side' in adf.df.columns, "Subframe index should be auto-materialized"
        assert 'x2' in adf.df.columns, "Intermediate alias should be auto-materialized"
        assert 'result' in adf.df.columns
    
    def test_no_redundant_materialization(self):
        """Verify index columns aren't materialized twice if already present."""
        main_df = pd.DataFrame({
            'sec': np.array([0, 9, 18], dtype=np.uint8),
            'side': np.array([0, 0, 1], dtype=np.int8),  # Already physical
        })
        adf = AliasDataFrame(main_df)
        
        # Subframe using physical column as index
        sub_df = pd.DataFrame({
            'side': np.array([0, 1], dtype=np.int8),
            'value': np.array([10.0, 20.0], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('SF', sub_adf, index_columns='side')
        adf.auto_alias_subframe('SF', validate=False)
        
        # Add alias using subframe
        adf.add_alias('scaled', 'value * 2', dtype=np.float32)
        
        # Verify 'side' is already present
        assert 'side' in adf.df.columns
        
        # Materialize - 'side' should NOT be duplicated
        adf.materialize_alias('scaled')
        
        # Verify 'side' still appears only once
        assert list(adf.df.columns).count('side') == 1
        assert 'scaled' in adf.df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
