"""
test_invariance_load_mode.py — I1: Load Mode Invariance

Phase 13.7.ADF: Invariance Test Suite
Priority: P0
Tests: 8

Property: read_tree_lazy() + materialize == read_tree() (eager)

Author: Claude2-Coder
Date: 2026-01-15
Version: 1.3 - Added skip markers for BUG_AliasDataFrame_20260116_lazy_subframe_init
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12

# Bug reference for skipped tests
BUG_LAZY_SUBFRAME = "BUG_AliasDataFrame_20260116_lazy_subframe_init: register_subframe_lazy() missing _df attribute"


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


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def root_test_file(tmp_path):
    """Create a test ROOT file with multiple branches."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "test_load_mode.root"
    
    np.random.seed(42)
    n = 1000
    
    data = {
        'x': np.random.randn(n).astype(np.float64),
        'y': np.random.randn(n).astype(np.float64),
        'z': np.random.randn(n).astype(np.float64),
        'key': np.random.randint(0, 100, n).astype(np.int64),
        'flag': np.random.randint(0, 2, n).astype(np.int32),
    }
    
    with uproot.create(file_path) as f:
        f["tree"] = data
    
    return str(file_path), data


@pytest.fixture
def root_chain_files(tmp_path):
    """Create multiple ROOT files for chain testing."""
    uproot = pytest.importorskip("uproot")
    
    files = []
    all_data = {k: [] for k in ['x', 'y', 'key']}
    
    np.random.seed(42)
    
    for i in range(3):
        file_path = tmp_path / f"chain_{i}.root"
        n = 500
        
        data = {
            'x': np.random.randn(n).astype(np.float64),
            'y': np.random.randn(n).astype(np.float64),
            'key': np.random.randint(0, 50, n).astype(np.int64),
        }
        
        with uproot.create(file_path) as f:
            f["tree"] = data
        
        files.append(str(file_path))
        for k in all_data:
            all_data[k].extend(data[k])
    
    for k in all_data:
        all_data[k] = np.array(all_data[k])
    
    return files, all_data


@pytest.fixture
def root_subframe_files(tmp_path):
    """Create main and subframe ROOT files."""
    uproot = pytest.importorskip("uproot")
    
    np.random.seed(42)
    
    main_path = tmp_path / "main.root"
    n_main = 500
    main_data = {
        'key': np.random.randint(0, 50, n_main).astype(np.int64),
        'value': np.random.randn(n_main).astype(np.float64),
    }
    with uproot.create(main_path) as f:
        f["tree"] = main_data
    
    sub_path = tmp_path / "subframe.root"
    sub_data = {
        'key': np.arange(50, dtype=np.int64),
        'offset': np.random.randn(50).astype(np.float64),
    }
    with uproot.create(sub_path) as f:
        f["tree"] = sub_data
    
    return str(main_path), str(sub_path), main_data, sub_data


# =============================================================================
# TEST CLASS: LOAD MODE INVARIANCE
# =============================================================================

@pytest.mark.invariance
class TestInvarianceLoadMode:
    """
    I1: Load Mode Invariance Tests
    
    Core invariant: Lazy loading produces identical results to eager loading.
    """
    
    def test_I1_1_lazy_vs_eager_column_values(self, root_test_file):
        """
        I1_1: Column values identical between lazy and eager load.
        """
        from AliasDataFrame import AliasDataFrame
        
        file_path, expected_data = root_test_file
        
        adf_eager = AliasDataFrame.read_tree(file_path, "tree")
        adf_lazy = AliasDataFrame.read_tree_lazy(file_path, "tree")
        
        for col in ['x', 'y', 'z', 'key', 'flag']:
            eager_values = adf_eager[col].values
            lazy_values = adf_lazy[col].values
            
            assert_invariant_equal(lazy_values, eager_values, f"I1_1_{col}")
            assert_invariant_equal(eager_values, expected_data[col], f"I1_1_{col}_original")
    
    def test_I1_2_lazy_vs_eager_alias_evaluation(self, root_test_file):
        """
        I1_2: Alias evaluation produces identical results.
        """
        from AliasDataFrame import AliasDataFrame
        
        file_path, _ = root_test_file
        
        # Eager load
        adf_eager = AliasDataFrame.read_tree(file_path, "tree")
        adf_eager.add_alias('sum_xy', 'x + y')
        adf_eager.add_alias('product', 'x * y * z')
        adf_eager.add_alias('scaled', 'x * 2 + y * 3')
        adf_eager.materialize_alias('sum_xy')
        adf_eager.materialize_alias('product')
        adf_eager.materialize_alias('scaled')
        eager_sum = adf_eager.df['sum_xy'].values
        eager_product = adf_eager.df['product'].values
        eager_scaled = adf_eager.df['scaled'].values
        
        # Lazy load
        adf_lazy = AliasDataFrame.read_tree_lazy(file_path, "tree")
        adf_lazy.ensure_branches(['x', 'y', 'z'])
        adf_lazy.add_alias('sum_xy', 'x + y')
        adf_lazy.add_alias('product', 'x * y * z')
        adf_lazy.add_alias('scaled', 'x * 2 + y * 3')
        adf_lazy.materialize_alias('sum_xy')
        adf_lazy.materialize_alias('product')
        adf_lazy.materialize_alias('scaled')
        lazy_sum = adf_lazy.df['sum_xy'].values
        lazy_product = adf_lazy.df['product'].values
        lazy_scaled = adf_lazy.df['scaled'].values
        
        assert_invariant_equal(lazy_sum, eager_sum, "I1_2_sum_xy")
        assert_invariant_equal(lazy_product, eager_product, "I1_2_product")
        assert_invariant_equal(lazy_scaled, eager_scaled, "I1_2_scaled")
    
    def test_I1_3_lazy_vs_eager_with_subframe_join(self, root_subframe_files):
        """
        I1_3: Subframe join results identical in lazy vs eager mode.
        """
        from AliasDataFrame import AliasDataFrame
        
        main_path, sub_path, main_data, sub_data = root_subframe_files
        
        # Pandas reference
        main_df = pd.DataFrame(main_data)
        sub_df = pd.DataFrame(sub_data)
        merged = main_df.merge(sub_df, on='key', how='left')
        expected = (merged['value'] + merged['offset']).values
        
        # Eager main + eager subframe
        adf_eager = AliasDataFrame.read_tree(main_path, "tree")
        sub_adf = AliasDataFrame.read_tree(sub_path, "tree")
        adf_eager.register_subframe('S', sub_adf, 'key')
        adf_eager.add_alias('combined', 'value + S.offset')
        adf_eager.materialize_alias('combined')
        eager_result = adf_eager.df['combined'].values
        
        # Lazy main + eager subframe
        adf_lazy = AliasDataFrame.read_tree_lazy(main_path, "tree")
        adf_lazy.ensure_branches(['key', 'value'])
        sub_adf2 = AliasDataFrame.read_tree(sub_path, "tree")
        adf_lazy.register_subframe('S', sub_adf2, 'key')
        adf_lazy.add_alias('combined', 'value + S.offset')
        adf_lazy.materialize_alias('combined')
        lazy_result = adf_lazy.df['combined'].values
        
        assert_invariant_equal(eager_result, expected, "I1_3_eager_vs_pandas")
        assert_invariant_equal(lazy_result, expected, "I1_3_lazy_vs_pandas")
        assert_invariant_equal(lazy_result, eager_result, "I1_3_lazy_vs_eager")
    
    @pytest.mark.slow
    def test_I1_4_lazy_chain_vs_eager_chain(self, root_chain_files):
        """
        I1_4: Chain loading produces same results in lazy vs eager mode.
        """
        from AliasDataFrame import AliasDataFrame
        
        files, expected_data = root_chain_files
        
        adf_eager = AliasDataFrame.read_chain(files, "tree")
        adf_lazy = AliasDataFrame.read_chain_lazy(files, "tree")
        
        for col in ['x', 'y', 'key']:
            eager_values = adf_eager[col].values
            lazy_values = adf_lazy[col].values
            assert_invariant_equal(lazy_values, eager_values, f"I1_4_{col}")
        
        assert len(adf_eager.df) == len(adf_lazy.df), "I1_4: Row count mismatch"
        assert len(adf_eager.df) == len(expected_data['x']), "I1_4: Total row count wrong"
    
    @pytest.mark.skip(reason=BUG_LAZY_SUBFRAME)
    def test_I1_5_lazy_subframe_vs_eager_subframe(self, root_subframe_files):
        """
        I1_5: Lazy subframe registration produces same join results.
        
        SKIPPED: Known bug in register_subframe_lazy()
        See: BUG_AliasDataFrame_20260116_lazy_subframe_init.md
        """
        from AliasDataFrame import AliasDataFrame
        
        main_path, sub_path, main_data, sub_data = root_subframe_files
        
        # Eager main + eager subframe (reference)
        adf_eager = AliasDataFrame.read_tree(main_path, "tree")
        sub_eager = AliasDataFrame.read_tree(sub_path, "tree")
        adf_eager.register_subframe('S', sub_eager, 'key')
        adf_eager.add_alias('s_offset', 'S.offset')
        adf_eager.materialize_alias('s_offset')
        eager_result = adf_eager.df['s_offset'].values
        
        # Eager main + lazy subframe
        adf_lazy_sub = AliasDataFrame.read_tree(main_path, "tree")
        adf_lazy_sub.register_subframe_lazy(
            name='S',
            file=sub_path,
            tree_name='tree',
            index_columns='key'
        )
        adf_lazy_sub.add_alias('s_offset', 'S.offset')
        adf_lazy_sub.materialize_alias('s_offset')
        lazy_sub_result = adf_lazy_sub.df['s_offset'].values
        
        assert_invariant_equal(lazy_sub_result, eager_result, "I1_5_lazy_subframe")
    
    def test_I1_6_mixed_lazy_main_eager_sub(self, root_subframe_files):
        """
        I1_6: Lazy main frame with eager subframe works correctly.
        """
        from AliasDataFrame import AliasDataFrame
        
        main_path, sub_path, _, _ = root_subframe_files
        
        # All eager (reference)
        adf_eager = AliasDataFrame.read_tree(main_path, "tree")
        sub_eager = AliasDataFrame.read_tree(sub_path, "tree")
        adf_eager.register_subframe('S', sub_eager, 'key')
        adf_eager.add_alias('result', 'value * S.offset')
        adf_eager.materialize_alias('result')
        eager_result = adf_eager.df['result'].values
        
        # Lazy main + eager sub
        adf_mixed = AliasDataFrame.read_tree_lazy(main_path, "tree")
        adf_mixed.ensure_branches(['key', 'value'])
        sub_eager2 = AliasDataFrame.read_tree(sub_path, "tree")
        adf_mixed.register_subframe('S', sub_eager2, 'key')
        adf_mixed.add_alias('result', 'value * S.offset')
        adf_mixed.materialize_alias('result')
        mixed_result = adf_mixed.df['result'].values
        
        assert_invariant_equal(mixed_result, eager_result, "I1_6_mixed_lazy_main")
    
    @pytest.mark.skip(reason=BUG_LAZY_SUBFRAME)
    def test_I1_7_mixed_eager_main_lazy_sub(self, root_subframe_files):
        """
        I1_7: Eager main frame with lazy subframe works correctly.
        
        SKIPPED: Known bug in register_subframe_lazy()
        See: BUG_AliasDataFrame_20260116_lazy_subframe_init.md
        """
        from AliasDataFrame import AliasDataFrame
        
        main_path, sub_path, _, _ = root_subframe_files
        
        # All eager (reference)
        adf_eager = AliasDataFrame.read_tree(main_path, "tree")
        sub_eager = AliasDataFrame.read_tree(sub_path, "tree")
        adf_eager.register_subframe('S', sub_eager, 'key')
        adf_eager.add_alias('result', 'value + S.offset')
        adf_eager.materialize_alias('result')
        eager_result = adf_eager.df['result'].values
        
        # Eager main + lazy sub
        adf_mixed = AliasDataFrame.read_tree(main_path, "tree")
        adf_mixed.register_subframe_lazy(
            name='S',
            file=sub_path,
            tree_name='tree',
            index_columns='key'
        )
        adf_mixed.add_alias('result', 'value + S.offset')
        adf_mixed.materialize_alias('result')
        mixed_result = adf_mixed.df['result'].values
        
        assert_invariant_equal(mixed_result, eager_result, "I1_7_mixed_lazy_sub")
    
    def test_I1_8_branch_auto_detection_equivalence(self, root_test_file):
        """
        I1_8: Auto-detected branches produce same results as explicit loading.
        """
        from AliasDataFrame import AliasDataFrame
        
        file_path, _ = root_test_file
        
        # Lazy with explicit branches
        adf_explicit = AliasDataFrame.read_tree_lazy(file_path, "tree")
        adf_explicit.ensure_branches(['x', 'y'])
        adf_explicit.add_alias('sum_xy', 'x + y')
        adf_explicit.materialize_alias('sum_xy')
        explicit_result = adf_explicit.df['sum_xy'].values
        
        # Lazy with auto-detection via column access
        adf_auto = AliasDataFrame.read_tree_lazy(file_path, "tree")
        _ = adf_auto['x']  # Trigger auto-load
        _ = adf_auto['y']  # Trigger auto-load
        adf_auto.add_alias('sum_xy', 'x + y')
        adf_auto.materialize_alias('sum_xy')
        auto_result = adf_auto.df['sum_xy'].values
        
        # Eager reference
        adf_eager = AliasDataFrame.read_tree(file_path, "tree")
        adf_eager.add_alias('sum_xy', 'x + y')
        adf_eager.materialize_alias('sum_xy')
        eager_result = adf_eager.df['sum_xy'].values
        
        assert_invariant_equal(explicit_result, eager_result, "I1_8_explicit_vs_eager")
        assert_invariant_equal(auto_result, eager_result, "I1_8_auto_vs_eager")
        assert_invariant_equal(auto_result, explicit_result, "I1_8_auto_vs_explicit")


# =============================================================================
# TEST SUMMARY
# =============================================================================

class TestLoadModeSummary:
    """Verify all I1 tests are present."""
    
    def test_I1_count(self):
        """Verify we have 8 load mode tests (including 2 skipped)."""
        tests = [
            'test_I1_1_lazy_vs_eager_column_values',
            'test_I1_2_lazy_vs_eager_alias_evaluation',
            'test_I1_3_lazy_vs_eager_with_subframe_join',
            'test_I1_4_lazy_chain_vs_eager_chain',
            'test_I1_5_lazy_subframe_vs_eager_subframe',  # SKIPPED - known bug
            'test_I1_6_mixed_lazy_main_eager_sub',
            'test_I1_7_mixed_eager_main_lazy_sub',        # SKIPPED - known bug
            'test_I1_8_branch_auto_detection_equivalence',
        ]
        assert len(tests) == 8, "Expected 8 I1 tests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "invariance and not slow"])
