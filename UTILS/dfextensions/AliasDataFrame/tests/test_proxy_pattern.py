#!/usr/bin/env python3
"""
Test suite for AliasDataFrame proxy pattern.

Tests the DataFrame delegation that enables:
- adf['column'] instead of adf.df['column']
- len(adf) instead of len(adf.df)
- adf.head() instead of adf.df.head()
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_adf():
    """Simple AliasDataFrame for basic tests."""
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50],
        'z': [100, 200, 300, 400, 500],
    })
    return AliasDataFrame(df)


@pytest.fixture
def adf_with_aliases():
    """AliasDataFrame with some aliases defined."""
    df = pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    adf = AliasDataFrame(df)
    adf.add_alias('sum_xy', 'x + y')
    adf.add_alias('prod_xy', 'x * y')
    return adf


# =============================================================================
# __getitem__ Tests
# =============================================================================

class TestGetItem:
    """Tests for adf['column'] syntax."""
    
    def test_single_column_access(self, simple_adf):
        """adf['x'] should return the column."""
        result = simple_adf['x']
        expected = simple_adf.df['x']
        pd.testing.assert_series_equal(result, expected)
    
    def test_multiple_column_access(self, simple_adf):
        """adf[['x', 'y']] should return DataFrame with those columns."""
        result = simple_adf[['x', 'y']]
        expected = simple_adf.df[['x', 'y']]
        pd.testing.assert_frame_equal(result, expected)
    
    def test_column_chain_operations(self, simple_adf):
        """adf['x'].mean() should work."""
        assert simple_adf['x'].mean() == 3.0
        assert simple_adf['y'].sum() == 150
    
    def test_nonexistent_column_raises(self, simple_adf):
        """Accessing non-existent column should raise KeyError."""
        with pytest.raises(KeyError):
            _ = simple_adf['nonexistent']


# =============================================================================
# __setitem__ Tests (Write-through) — PHASE_13_62_ADF Stage 2a (Fix A)
# =============================================================================

class TestSetItemWriteThrough:
    """adf['col'] = value now writes through to the underlying frame.

    PHASE_13_62_ADF Stage 2a (Fix A) replaced the previous TypeError block with a
    write-through that also syncs the lazy reader's loaded_branches bookkeeping.
    These tests exercise the same public API (adf[col] = value) the bug used (FM#12).
    """

    def test_writethrough_numpy_array(self, simple_adf):
        """adf['new'] = np.array writes through and is read-back-able."""
        simple_adf['new'] = np.arange(5.0) * 2
        assert 'new' in simple_adf.df.columns
        assert np.allclose(simple_adf.df['new'], np.arange(5.0) * 2)
        assert np.allclose(simple_adf['new'], np.arange(5.0) * 2)   # read back via adf[]

    def test_writethrough_list(self, simple_adf):
        """adf['new'] = list of matching length writes through."""
        simple_adf['new'] = [1, 2, 3, 4, 5]
        assert list(simple_adf.df['new']) == [1, 2, 3, 4, 5]

    def test_writethrough_series_index_aligned(self, simple_adf):
        """adf['new'] = Series aligns on the frame index."""
        simple_adf['new'] = pd.Series([5, 4, 3, 2, 1], index=simple_adf.df.index)
        assert list(simple_adf.df['new']) == [5, 4, 3, 2, 1]

    def test_writethrough_scalar_broadcast(self, simple_adf):
        """adf['new'] = scalar broadcasts to all rows."""
        simple_adf['new'] = 7
        assert (simple_adf.df['new'] == 7).all()

    def test_overwrite_existing_column(self, simple_adf):
        """adf['x'] = value overwrites an existing column."""
        simple_adf['x'] = np.zeros(5)
        assert (simple_adf.df['x'] == 0).all()

    def test_non_string_key_raises(self, simple_adf):
        """A non-string key still raises TypeError (string column names only)."""
        with pytest.raises(TypeError):
            simple_adf[('bad',)] = [1, 2, 3, 4, 5]

    def test_length_mismatch_raises(self, simple_adf):
        """A length-mismatched value raises (from pandas); no silent corruption."""
        with pytest.raises(Exception):
            simple_adf['new'] = [1, 2, 3]

    def test_aliases_mapping_still_immutable(self, simple_adf):
        """Fix A touches column assignment only; adf.aliases stays immutable
        (the _ReadOnlyAliasDict guard is unrelated and untouched)."""
        with pytest.raises(TypeError):
            simple_adf.aliases['foo'] = 'x + 1'   # _ReadOnlyAliasDict guard


# =============================================================================
# __len__ Tests
# =============================================================================

class TestLen:
    """Tests for len(adf) syntax."""
    
    def test_len_returns_row_count(self, simple_adf):
        """len(adf) should return number of rows."""
        assert len(simple_adf) == 5
    
    def test_len_matches_df(self, simple_adf):
        """len(adf) should match len(adf.df)."""
        assert len(simple_adf) == len(simple_adf.df)
    
    def test_len_empty_dataframe(self):
        """len(adf) should return 0 for empty DataFrame."""
        adf = AliasDataFrame(pd.DataFrame())
        assert len(adf) == 0


# =============================================================================
# __iter__ Tests
# =============================================================================

class TestIter:
    """Tests for iteration over adf."""
    
    def test_iterate_columns(self, simple_adf):
        """Iterating over adf should yield column names."""
        columns = list(simple_adf)
        assert columns == ['x', 'y', 'z']
    
    def test_for_loop(self, simple_adf):
        """for col in adf should work."""
        result = []
        for col in simple_adf:
            result.append(col)
        assert result == ['x', 'y', 'z']


# =============================================================================
# __contains__ Tests
# =============================================================================

class TestContains:
    """Tests for 'column' in adf syntax."""
    
    def test_existing_column(self, simple_adf):
        """'x' in adf should return True for existing column."""
        assert 'x' in simple_adf
        assert 'y' in simple_adf
    
    def test_nonexistent_column(self, simple_adf):
        """'nonexistent' in adf should return False."""
        assert 'nonexistent' not in simple_adf
    
    def test_alias_not_in_columns(self, adf_with_aliases):
        """Unmaterialized alias should not be 'in' adf (not a column yet)."""
        # Before materialization, alias is not a column
        assert 'sum_xy' not in adf_with_aliases
        
        # After materialization, it becomes a column
        adf_with_aliases.materialize_alias('sum_xy')
        assert 'sum_xy' in adf_with_aliases


# =============================================================================
# Property Tests
# =============================================================================

class TestProperties:
    """Tests for proxy properties."""
    
    def test_columns_property(self, simple_adf):
        """adf.columns should return DataFrame columns."""
        pd.testing.assert_index_equal(simple_adf.columns, simple_adf.df.columns)
    
    def test_index_property(self, simple_adf):
        """adf.index should return DataFrame index."""
        pd.testing.assert_index_equal(simple_adf.index, simple_adf.df.index)
    
    def test_shape_property(self, simple_adf):
        """adf.shape should return (rows, cols) tuple."""
        assert simple_adf.shape == (5, 3)
        assert simple_adf.shape == simple_adf.df.shape
    
    def test_dtypes_property(self, simple_adf):
        """adf.dtypes should return column dtypes."""
        pd.testing.assert_series_equal(simple_adf.dtypes, simple_adf.df.dtypes)
    
    def test_loc_property(self, simple_adf):
        """adf.loc should work for label-based indexing."""
        result = simple_adf.loc[0, 'x']
        assert result == 1
    
    def test_iloc_property(self, simple_adf):
        """adf.iloc should work for integer-based indexing."""
        result = simple_adf.iloc[0, 0]
        assert result == 1


# =============================================================================
# __getattr__ Delegation Tests
# =============================================================================

class TestMethodDelegation:
    """Tests for delegated DataFrame methods."""
    
    def test_head(self, simple_adf):
        """adf.head() should work."""
        result = simple_adf.head(2)
        expected = simple_adf.df.head(2)
        pd.testing.assert_frame_equal(result, expected)
    
    def test_tail(self, simple_adf):
        """adf.tail() should work."""
        result = simple_adf.tail(2)
        expected = simple_adf.df.tail(2)
        pd.testing.assert_frame_equal(result, expected)
    
    def test_describe(self, simple_adf):
        """adf.describe() should work."""
        result = simple_adf.describe()
        expected = simple_adf.df.describe()
        pd.testing.assert_frame_equal(result, expected)
    
    def test_mean(self, simple_adf):
        """adf.mean() should work."""
        result = simple_adf.mean()
        expected = simple_adf.df.mean()
        pd.testing.assert_series_equal(result, expected)
    
    def test_groupby(self, simple_adf):
        """adf.groupby() should work."""
        simple_adf.df['group'] = ['a', 'a', 'b', 'b', 'b']
        result = simple_adf.groupby('group')['x'].sum()
        expected = simple_adf.df.groupby('group')['x'].sum()
        pd.testing.assert_series_equal(result, expected)
    
    def test_query(self, simple_adf):
        """adf.query() should work."""
        result = simple_adf.query('x > 2')
        expected = simple_adf.df.query('x > 2')
        pd.testing.assert_frame_equal(result, expected)
    
    def test_to_dict(self, simple_adf):
        """adf.to_dict() should work."""
        result = simple_adf.to_dict()
        expected = simple_adf.df.to_dict()
        assert result == expected


# =============================================================================
# Column vs Alias Priority Tests
# =============================================================================

class TestColumnAliasPriority:
    """Tests for priority between columns, aliases, and DataFrame methods."""
    
    def test_column_access_via_attribute(self, simple_adf):
        """adf.x should return column x."""
        result = simple_adf.x
        expected = simple_adf.df['x']
        pd.testing.assert_series_equal(result, expected)
    
    def test_alias_auto_materializes(self, adf_with_aliases):
        """adf.sum_xy should materialize and return the alias."""
        result = adf_with_aliases.sum_xy
        expected = adf_with_aliases.df['x'] + adf_with_aliases.df['y']
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_adf_method_takes_priority(self, adf_with_aliases):
        """AliasDataFrame's own methods should take priority over DataFrame methods."""
        # materialize_alias() is an AliasDataFrame method, not a DataFrame method
        # This verifies AliasDataFrame methods are called, not delegated
        adf_with_aliases.materialize_alias('sum_xy')
        assert 'sum_xy' in adf_with_aliases.df.columns
        
        # add_alias() is also AliasDataFrame-specific
        adf_with_aliases.add_alias('test', 'x + 1')
        assert 'test' in adf_with_aliases.aliases


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_dataframe(self):
        """Operations on empty DataFrame should work."""
        adf = AliasDataFrame(pd.DataFrame())
        assert len(adf) == 0
        assert adf.shape == (0, 0)
        assert list(adf.columns) == []
    
    def test_nonexistent_attribute_raises(self, simple_adf):
        """Accessing truly nonexistent attribute should raise AttributeError."""
        with pytest.raises(AttributeError):
            _ = simple_adf.totally_nonexistent_attribute_xyz
    
    def test_chained_operations(self, simple_adf):
        """Complex chained operations should work."""
        result = simple_adf[simple_adf['x'] > 2]['y'].mean()
        expected = simple_adf.df[simple_adf.df['x'] > 2]['y'].mean()
        assert result == expected


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing adf.df access still works."""
    
    def test_df_access_still_works(self, simple_adf):
        """adf.df['column'] should still work."""
        result = simple_adf.df['x']
        expected = pd.Series([1, 2, 3, 4, 5], name='x')
        pd.testing.assert_series_equal(result, expected)
    
    def test_df_modification_still_works(self, simple_adf):
        """adf.df['new'] = values should still work."""
        simple_adf.df['new'] = [5, 4, 3, 2, 1]
        assert 'new' in simple_adf.df.columns
        assert list(simple_adf.df['new']) == [5, 4, 3, 2, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
