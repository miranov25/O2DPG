"""
Tests for _composite_keys.py module.

These tests verify the composite key generation utilities work correctly
as a standalone module, independent of AliasDataFrameRDF.

The same functions are also tested via AliasDataFrameRDF re-exports in
test_AliasDataFrameRDF.py, ensuring backward compatibility.
"""

import pytest
import numpy as np
import pandas as pd


class TestGetCompositeKeyColumnName:
    """Test standard naming convention for composite key columns."""
    
    def test_basic_name(self):
        """Test basic subframe name."""
        from _composite_keys import get_composite_key_column_name
        
        assert get_composite_key_column_name('DTrack0') == '__adf_key_DTrack0__'
    
    def test_short_name(self):
        """Test short subframe name."""
        from _composite_keys import get_composite_key_column_name
        
        assert get_composite_key_column_name('S') == '__adf_key_S__'
    
    def test_long_name(self):
        """Test longer subframe name."""
        from _composite_keys import get_composite_key_column_name
        
        assert get_composite_key_column_name('calibration') == '__adf_key_calibration__'
    
    def test_underscore_name(self):
        """Test name with underscores."""
        from _composite_keys import get_composite_key_column_name
        
        assert get_composite_key_column_name('my_subframe') == '__adf_key_my_subframe__'


class TestCheckDenseOverflow:
    """Test overflow detection for dense linearization."""
    
    def test_small_values_safe(self):
        """Small values should be safe."""
        from _composite_keys import check_dense_overflow
        
        is_safe, compact_range = check_dense_overflow([10, 20, 30])
        assert is_safe
        assert compact_range == 10 * 20 * 30
    
    def test_typical_tpc_case(self):
        """Typical TPC case: side=2, row=152, drift=28."""
        from _composite_keys import check_dense_overflow
        
        is_safe, compact_range = check_dense_overflow([2, 152, 28])
        assert is_safe
        assert compact_range == 2 * 152 * 28
    
    def test_large_values_unsafe(self):
        """Values that would overflow int64 should be unsafe."""
        from _composite_keys import check_dense_overflow
        
        # 2^30 * 2^30 * 2^30 = 2^90 > 2^63
        is_safe, compact_range = check_dense_overflow([2**30, 2**30, 2**30])
        assert not is_safe
        assert compact_range == float('inf')
    
    def test_edge_case_just_safe(self):
        """Test values just below overflow threshold."""
        from _composite_keys import check_dense_overflow
        
        # 2^20 * 2^20 * 2^20 = 2^60 < 2^63
        is_safe, compact_range = check_dense_overflow([2**20, 2**20, 2**20])
        assert is_safe
        assert compact_range == 2**60
    
    def test_single_value(self):
        """Single key column is always safe."""
        from _composite_keys import check_dense_overflow
        
        is_safe, compact_range = check_dense_overflow([1000000])
        assert is_safe
        assert compact_range == 1000000


class TestShouldUseSparse:
    """Test sparse vs dense decision function."""
    
    def test_small_dense_data(self):
        """Dense data with small range should use dense."""
        from _composite_keys import should_use_sparse
        
        df = pd.DataFrame({
            'k1': np.array([0, 1, 2, 3, 4], dtype=np.int32),
            'k2': np.array([0, 0, 1, 1, 2], dtype=np.int32),
        })
        
        assert not should_use_sparse(df, ['k1', 'k2'])
    
    def test_large_sparse_data(self):
        """Sparse data with large gaps should use sparse."""
        from _composite_keys import should_use_sparse
        
        # Only 5 unique combinations but max values suggest huge range
        df = pd.DataFrame({
            'k1': np.array([0, 1000000, 2000000, 3000000, 4000000], dtype=np.int64),
            'k2': np.array([0, 1000000, 2000000, 3000000, 4000000], dtype=np.int64),
        })
        
        assert should_use_sparse(df, ['k1', 'k2'])
    
    def test_overflow_triggers_sparse(self):
        """Data that would overflow int32 should use sparse."""
        from _composite_keys import should_use_sparse
        
        # max = 100000, range = 100001^2 > 2^31
        df = pd.DataFrame({
            'k1': np.array([0, 100000], dtype=np.int64),
            'k2': np.array([0, 100000], dtype=np.int64),
        })
        
        assert should_use_sparse(df, ['k1', 'k2'])
    
    def test_wasteful_triggers_sparse(self):
        """Range >10x unique combinations should use sparse."""
        from _composite_keys import should_use_sparse
        
        # 2 unique combinations but range = 101 * 101 = 10201 > 10 * 2
        df = pd.DataFrame({
            'k1': np.array([0, 100], dtype=np.int32),
            'k2': np.array([0, 100], dtype=np.int32),
        })
        
        assert should_use_sparse(df, ['k1', 'k2'])


class TestGenerateDenseCppExpression:
    """Test C++ expression generation for dense linearization."""
    
    def test_single_key(self):
        """Single key returns just the column name."""
        from _composite_keys import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['key'], [100])
        assert result == 'key'
    
    def test_two_keys(self):
        """Two keys: k0 + k1 * max0."""
        from _composite_keys import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['side', 'row'], [2, 152])
        assert result == 'side + row * 2'
    
    def test_three_keys(self):
        """Three keys: k0 + k1 * max0 + k2 * max0 * max1."""
        from _composite_keys import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['a', 'b', 'c'], [10, 20, 30])
        assert result == 'a + b * 10 + c * 10 * 20'
    
    def test_four_keys(self):
        """Four keys: k0 + k1 * max0 + k2 * max0 * max1 + k3 * max0 * max1 * max2."""
        from _composite_keys import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['k1', 'k2', 'k3', 'k4'], [2, 3, 4, 5])
        assert result == 'k1 + k2 * 2 + k3 * 2 * 3 + k4 * 2 * 3 * 4'
    
    def test_tpc_style_keys(self):
        """TPC-style keys: side, row, drift."""
        from _composite_keys import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['side', 'row', 'driftTime'], [2, 152, 28])
        assert result == 'side + row * 2 + driftTime * 2 * 152'


class TestComputeCompositeKeyDense:
    """Test dense linearization computation."""
    
    def test_single_column(self):
        """Single column key is just the column values."""
        from _composite_keys import compute_composite_key_dense
        
        df = pd.DataFrame({'key': np.array([0, 5, 10, 15], dtype=np.int32)})
        result = compute_composite_key_dense(df, ['key'])
        
        np.testing.assert_array_equal(result, [0, 5, 10, 15])
    
    def test_two_columns(self):
        """Two column key: k0 + k1 * max0."""
        from _composite_keys import compute_composite_key_dense
        
        df = pd.DataFrame({
            'k1': np.array([0, 0, 1, 1], dtype=np.int32),
            'k2': np.array([0, 1, 0, 1], dtype=np.int32),
        })
        
        # max_values auto-computed: [2, 2]
        # k1=0,k2=0 -> 0 + 0*2 = 0
        # k1=0,k2=1 -> 0 + 1*2 = 2
        # k1=1,k2=0 -> 1 + 0*2 = 1
        # k1=1,k2=1 -> 1 + 1*2 = 3
        result = compute_composite_key_dense(df, ['k1', 'k2'])
        
        np.testing.assert_array_equal(result, [0, 2, 1, 3])
    
    def test_three_columns(self):
        """Three column key: k0 + k1*max0 + k2*max0*max1."""
        from _composite_keys import compute_composite_key_dense
        
        df = pd.DataFrame({
            'a': np.array([0, 1, 0, 1], dtype=np.int32),
            'b': np.array([0, 0, 1, 1], dtype=np.int32),
            'c': np.array([0, 0, 0, 1], dtype=np.int32),
        })
        
        # max_values: [2, 2, 2]
        # a=0,b=0,c=0 -> 0 + 0*2 + 0*2*2 = 0
        # a=1,b=0,c=0 -> 1 + 0*2 + 0*2*2 = 1
        # a=0,b=1,c=0 -> 0 + 1*2 + 0*2*2 = 2
        # a=1,b=1,c=1 -> 1 + 1*2 + 1*2*2 = 7
        result = compute_composite_key_dense(df, ['a', 'b', 'c'])
        
        np.testing.assert_array_equal(result, [0, 1, 2, 7])
    
    def test_with_explicit_max_values(self):
        """Test with explicitly provided max values."""
        from _composite_keys import compute_composite_key_dense
        
        df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 1], dtype=np.int32),
        })
        
        # Use larger max values than data requires
        result = compute_composite_key_dense(df, ['k1', 'k2'], max_values=[10, 10])
        
        # k1=0,k2=0 -> 0 + 0*10 = 0
        # k1=1,k2=1 -> 1 + 1*10 = 11
        np.testing.assert_array_equal(result, [0, 11])
    
    def test_result_is_int64(self):
        """Result should always be int64."""
        from _composite_keys import compute_composite_key_dense
        
        df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 1], dtype=np.int32),
        })
        
        result = compute_composite_key_dense(df, ['k1', 'k2'])
        assert result.dtype == np.int64


class TestComputeCompositeKeySparse:
    """Test sparse key mapping."""
    
    def test_basic_mapping(self):
        """Basic sparse mapping assigns sequential IDs."""
        from _composite_keys import compute_composite_key_sparse
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1, 2], dtype=np.int32),
            'k2': np.array([0, 0, 0], dtype=np.int32),
        })
        
        sub_df = pd.DataFrame({
            'k1': np.array([0, 1, 2], dtype=np.int32),
            'k2': np.array([0, 0, 0], dtype=np.int32),
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k1', 'k2'])
        
        # Same keys should get same IDs
        np.testing.assert_array_equal(main_keys, sub_keys)
    
    def test_shuffled_subframe(self):
        """Sparse mapping works with shuffled subframe."""
        from _composite_keys import compute_composite_key_sparse
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1, 2], dtype=np.int32),
            'k2': np.array([0, 0, 0], dtype=np.int32),
        })
        
        # Subframe in different order
        sub_df = pd.DataFrame({
            'k1': np.array([2, 0, 1], dtype=np.int32),
            'k2': np.array([0, 0, 0], dtype=np.int32),
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k1', 'k2'])
        
        # Key (0,0) should have same ID in both
        assert main_keys[0] == sub_keys[1]  # main row 0 = sub row 1
        # Key (1,0) should have same ID in both
        assert main_keys[1] == sub_keys[2]  # main row 1 = sub row 2
        # Key (2,0) should have same ID in both
        assert main_keys[2] == sub_keys[0]  # main row 2 = sub row 0
    
    def test_sparse_with_gaps(self):
        """Sparse mapping handles large gaps efficiently."""
        from _composite_keys import compute_composite_key_sparse
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1000000, 2000000], dtype=np.int64),
            'k2': np.array([0, 1000000, 2000000], dtype=np.int64),
        })
        
        sub_df = pd.DataFrame({
            'k1': np.array([0, 1000000, 2000000], dtype=np.int64),
            'k2': np.array([0, 1000000, 2000000], dtype=np.int64),
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k1', 'k2'])
        
        # Should get sequential IDs despite large gaps
        assert max(main_keys) < 10  # Only 3 unique combinations
    
    def test_result_is_int64(self):
        """Result should always be int64."""
        from _composite_keys import compute_composite_key_sparse
        
        main_df = pd.DataFrame({'k': np.array([0, 1], dtype=np.int32)})
        sub_df = pd.DataFrame({'k': np.array([0, 1], dtype=np.int32)})
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k'])
        assert main_keys.dtype == np.int64
        assert sub_keys.dtype == np.int64


class TestComputeCompositeKeyAuto:
    """Test automatic dense/sparse selection."""
    
    def test_auto_selects_dense_for_compact_data(self):
        """Auto should select dense for compact data."""
        from _composite_keys import compute_composite_key_auto
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 0, 1, 1], dtype=np.int32),
            'k2': np.array([0, 1, 0, 1], dtype=np.int32),
        })
        
        sub_df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 0], dtype=np.int32),
        })
        
        main_keys, sub_keys, method = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2'])
        
        assert method == 'dense'
        assert len(main_keys) == 4
        assert len(sub_keys) == 2
    
    def test_auto_selects_sparse_for_large_gaps(self):
        """Auto should select sparse for sparse data."""
        from _composite_keys import compute_composite_key_auto
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1000000], dtype=np.int64),
            'k2': np.array([0, 1000000], dtype=np.int64),
        })
        
        sub_df = pd.DataFrame({
            'k1': np.array([0, 1000000], dtype=np.int64),
            'k2': np.array([0, 1000000], dtype=np.int64),
        })
        
        main_keys, sub_keys, method = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2'])
        
        assert method == 'sparse'
    
    def test_auto_keys_match_correctly(self):
        """Auto-generated keys should match between main and sub."""
        from _composite_keys import compute_composite_key_auto
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1, 2, 3, 4], dtype=np.int32),
            'k2': np.array([0, 0, 0, 0, 0], dtype=np.int32),
        })
        
        # Shuffled subframe
        sub_df = pd.DataFrame({
            'k1': np.array([4, 2, 0, 3, 1], dtype=np.int32),
            'k2': np.array([0, 0, 0, 0, 0], dtype=np.int32),
        })
        
        main_keys, sub_keys, _ = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2'])
        
        # main row 0 (k1=0) should match sub row 2 (k1=0)
        assert main_keys[0] == sub_keys[2]
        # main row 1 (k1=1) should match sub row 4 (k1=1)
        assert main_keys[1] == sub_keys[4]
    
    def test_explicit_dense_method(self):
        """Forcing dense method."""
        from _composite_keys import compute_composite_key_auto
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 1], dtype=np.int32),
        })
        sub_df = main_df.copy()
        
        _, _, method = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2'], method='dense')
        assert method == 'dense'
    
    def test_explicit_sparse_method(self):
        """Forcing sparse method."""
        from _composite_keys import compute_composite_key_auto
        
        main_df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 1], dtype=np.int32),
        })
        sub_df = main_df.copy()
        
        _, _, method = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2'], method='sparse')
        assert method == 'sparse'


class TestModuleExports:
    """Test that module exports are correct."""
    
    def test_all_exports_exist(self):
        """Test that all __all__ exports exist."""
        import _composite_keys
        
        for name in _composite_keys.__all__:
            assert hasattr(_composite_keys, name), f"Missing export: {name}"
    
    def test_all_exports_are_callable(self):
        """Test that all exports are callable."""
        import _composite_keys
        
        for name in _composite_keys.__all__:
            obj = getattr(_composite_keys, name)
            assert callable(obj), f"Export {name} is not callable"


class TestAliasDataFrameRDFReexports:
    """Test that AliasDataFrameRDF re-exports work correctly."""
    
    def test_reexports_exist(self):
        """Test that all functions are available from AliasDataFrameRDF."""
        from AliasDataFrameRDF import (
            get_composite_key_column_name,
            check_dense_overflow,
            should_use_sparse,
            generate_dense_cpp_expression,
            compute_composite_key_dense,
            compute_composite_key_sparse,
            compute_composite_key_auto,
        )
        
        # Just verify they import without error
        assert callable(get_composite_key_column_name)
        assert callable(check_dense_overflow)
        assert callable(should_use_sparse)
        assert callable(generate_dense_cpp_expression)
        assert callable(compute_composite_key_dense)
        assert callable(compute_composite_key_sparse)
        assert callable(compute_composite_key_auto)
    
    def test_reexports_are_same_functions(self):
        """Test that re-exports are the same as _composite_keys functions."""
        import _composite_keys
        import AliasDataFrameRDF
        
        assert AliasDataFrameRDF.get_composite_key_column_name is _composite_keys.get_composite_key_column_name
        assert AliasDataFrameRDF.check_dense_overflow is _composite_keys.check_dense_overflow
        assert AliasDataFrameRDF.should_use_sparse is _composite_keys.should_use_sparse
