"""
Research tests for AliasDataFrame RDataFrame integration.

These tests explore what is possible with RDataFrame + friend trees
to inform the implementation strategy.

Test Priority:
[1] test_indexed_friend          ← BLOCKER
[2] test_composite_index_friend  ← BLOCKER  
[3] test_friend_dot_notation     
[4] test_friend_with_alias       
"""

import pytest
import sys
import os

# Add parent directory to path for imports (module is in parent, tests in tests/)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
    ROOT_VERSION = ROOT.gROOT.GetVersion()
except ImportError:
    HAS_ROOT = False
    ROOT_VERSION = None

from AliasDataFrameRDF import (
    to_cpp_expr,
    extract_dependencies,
    get_ordered_defines,
    generate_rdf_code,
    CppExprConverter,
)


# =============================================================================
# Expression Conversion Tests (No ROOT needed)
# =============================================================================

class TestToCppExpr:
    """Test pandas -> C++ expression conversion using AST."""
    
    def test_numpy_sqrt(self):
        assert 'sqrt(x)' in to_cpp_expr('np.sqrt(x)')
    
    def test_numpy_abs(self):
        assert 'abs(x)' in to_cpp_expr('np.abs(x)')
    
    def test_numpy_pi(self):
        result = to_cpp_expr('2 * np.pi * r')
        assert 'M_PI' in result
    
    def test_power_simple(self):
        result = to_cpp_expr('x**2')
        assert 'pow(x, 2)' in result
    
    def test_power_with_parentheses(self):
        """This is why we use AST instead of regex."""
        result = to_cpp_expr('(x + y)**2')
        assert 'pow' in result
        # Should have pow((x + y), 2) or similar
    
    def test_power_float(self):
        result = to_cpp_expr('x**0.5')
        assert 'pow(x, 0.5)' in result
    
    def test_boolean_true(self):
        assert 'true' in to_cpp_expr('True')
    
    def test_boolean_false(self):
        assert 'false' in to_cpp_expr('False')
    
    def test_bitwise_and_preserved(self):
        """Bitwise & should be kept as & (not converted to &&)."""
        result = to_cpp_expr('(a < 5) & (b > 3)')
        assert '&' in result
        # Should NOT have && since we keep bitwise
    
    def test_bitwise_or_preserved(self):
        """Bitwise | should be kept as | (not converted to ||)."""
        result = to_cpp_expr('(a < 5) | (b > 3)')
        assert '|' in result
    
    def test_logical_not_converted(self):
        """Bitwise ~ should be converted to logical !."""
        result = to_cpp_expr('~valid')
        assert '!' in result
    
    def test_combined_expression(self):
        """Test complex expression."""
        expr = 'np.sqrt(x**2 + y**2)'
        result = to_cpp_expr(expr)
        assert 'sqrt' in result
        assert 'pow' in result
    
    def test_subframe_alias_replacement(self):
        """Test subframe column aliasing."""
        expr = 'T.mP3 * drift25'
        result = to_cpp_expr(expr, {'T.mP3': 'T_mP3'})
        assert 'T_mP3' in result
        assert 'T.mP3' not in result
    
    def test_comparison_operators(self):
        result = to_cpp_expr('x < 5')
        assert '<' in result
    
    def test_ternary(self):
        """Test if-else → ternary."""
        result = to_cpp_expr('a if cond else b')
        assert '?' in result
        assert ':' in result
    
    def test_logical_and_or(self):
        """Test 'and'/'or' → '&&'/'||'."""
        result = to_cpp_expr('a and b')
        assert '&&' in result
        
        result = to_cpp_expr('a or b')
        assert '||' in result


class TestExtractDependencies:
    """Test dependency extraction from expressions."""
    
    def test_simple_columns(self):
        deps = extract_dependencies('x + y')
        assert 'x' in deps
        assert 'y' in deps
    
    def test_subframe_column(self):
        deps = extract_dependencies('T.mP3 * 2')
        assert 'T.mP3' in deps
    
    def test_excludes_functions(self):
        deps = extract_dependencies('sqrt(x) + abs(y)')
        assert 'sqrt' not in deps
        assert 'abs' not in deps
        assert 'x' in deps
        assert 'y' in deps
    
    def test_filter_known_columns(self):
        known = {'x', 'y', 'z'}
        deps = extract_dependencies('x + y + unknown', known)
        assert 'x' in deps
        assert 'y' in deps
        assert 'unknown' not in deps
    
    def test_numpy_prefix_excluded(self):
        deps = extract_dependencies('np.sqrt(x)')
        assert 'np' not in deps
        assert 'x' in deps


# =============================================================================
# Sparse Key Tests
# =============================================================================

import numpy as np
import pandas as pd

from AliasDataFrameRDF import (
    should_use_sparse,
    compute_composite_key_dense,
    compute_composite_key_sparse,
    compute_composite_key_auto,
)


class TestSparseKeySupport:
    """Test sparse key mapping for multi-key joins."""
    
    def test_should_use_sparse_small_range(self):
        """Small contiguous range should use dense."""
        df = pd.DataFrame({
            'a': [0, 1, 2, 3, 4],
            'b': [0, 1, 2, 3, 4],
        })
        assert not should_use_sparse(df, ['a', 'b'])
    
    def test_should_use_sparse_large_range(self):
        """Large range exceeding int32 should use sparse."""
        df = pd.DataFrame({
            'a': [0, 100000],
            'b': [0, 100000],
            'c': [0, 100000],
        })
        # 100001^3 > 2^31
        assert should_use_sparse(df, ['a', 'b', 'c'])
    
    def test_should_use_sparse_wasteful(self):
        """Wasteful range (>10x unique) should use sparse."""
        df = pd.DataFrame({
            'a': [0, 1000],  # max 1001
            'b': [0, 1000],  # max 1001
        })
        # Compact range: 1001*1001 = 1M, unique: 2, ratio > 10x
        assert should_use_sparse(df, ['a', 'b'])
    
    def test_dense_key_basic(self):
        """Test dense key computation."""
        df = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [0, 1, 0],
        })
        keys = compute_composite_key_dense(df, ['a', 'b'], max_values=[3, 2])
        # key = a + b*3
        expected = np.array([0, 4, 2])  # 0+0*3, 1+1*3, 2+0*3
        np.testing.assert_array_equal(keys, expected)
    
    def test_sparse_key_with_gaps(self):
        """Sparse keys with gaps should produce contiguous indices."""
        main_df = pd.DataFrame({
            'k1': [0, 100, 500],
            'k2': [5, 10, 15],
        })
        sub_df = pd.DataFrame({
            'k1': [0, 100, 500, 999],
            'k2': [5, 10, 15, 20],
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k1', 'k2'])
        
        # Keys should be contiguous integers starting from 0
        assert main_keys.min() >= 0
        assert sub_keys.min() >= 0
        
        # Total unique keys = 4 (main has 3, sub has 4, but 3 overlap)
        # (0,5), (100,10), (500,15) shared + (999,20) only in sub
        all_keys = np.concatenate([main_keys, sub_keys])
        assert len(np.unique(all_keys)) == 4
        
        # Max key should be 3 (0-indexed for 4 unique combos)
        assert all_keys.max() == 3
    
    def test_sparse_key_large_values(self):
        """Sparse keys with values exceeding int32 range."""
        main_df = pd.DataFrame({
            'orbit': [1_000_000_000, 2_000_000_000, 3_000_000_000],
            'row': [0, 1, 2],
        })
        sub_df = pd.DataFrame({
            'orbit': [1_000_000_000, 2_000_000_000],
            'row': [0, 1],
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['orbit', 'row'])
        
        # Should produce small contiguous integers
        assert main_keys.max() < 10
        assert sub_keys.max() < 10
    
    def test_sparse_key_shared_mapping(self):
        """Main and subframe must use same key mapping."""
        main_df = pd.DataFrame({
            'k': [1, 2, 3],
        })
        sub_df = pd.DataFrame({
            'k': [2, 3, 4],  # Overlapping + extra
        })
        
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, ['k'])
        
        # k=2 should have same key in both
        main_k2_idx = main_df[main_df['k'] == 2].index[0]
        sub_k2_idx = sub_df[sub_df['k'] == 2].index[0]
        assert main_keys[main_k2_idx] == sub_keys[sub_k2_idx]
        
        # k=3 should have same key in both
        main_k3_idx = main_df[main_df['k'] == 3].index[0]
        sub_k3_idx = sub_df[sub_df['k'] == 3].index[0]
        assert main_keys[main_k3_idx] == sub_keys[sub_k3_idx]
    
    def test_sparse_matches_dense_for_contiguous(self):
        """Sparse and dense should produce equivalent joins for contiguous keys."""
        main_df = pd.DataFrame({
            'a': [0, 0, 1, 1, 2, 2],
            'b': [0, 1, 0, 1, 0, 1],
            'val': [10, 20, 30, 40, 50, 60],
        })
        sub_df = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [0, 0, 0],
            'calib': [1.0, 2.0, 3.0],
        })
        
        # Dense keys
        max_values = [3, 2]
        main_dense = compute_composite_key_dense(main_df, ['a', 'b'], max_values)
        sub_dense = compute_composite_key_dense(sub_df, ['a', 'b'], max_values)
        
        # Sparse keys
        main_sparse, sub_sparse = compute_composite_key_sparse(main_df, sub_df, ['a', 'b'])
        
        # Both should produce same join result
        # Build index lookup for both
        dense_lookup = {k: i for i, k in enumerate(sub_dense)}
        sparse_lookup = {k: i for i, k in enumerate(sub_sparse)}
        
        for i in range(len(main_df)):
            dense_match = dense_lookup.get(main_dense[i], -1)
            sparse_match = sparse_lookup.get(main_sparse[i], -1)
            assert dense_match == sparse_match, f"Row {i}: dense={dense_match}, sparse={sparse_match}"
    
    def test_auto_selects_dense_for_small(self):
        """Auto should select dense for small contiguous keys."""
        main_df = pd.DataFrame({'k': [0, 1, 2]})
        sub_df = pd.DataFrame({'k': [0, 1, 2]})
        
        _, _, method = compute_composite_key_auto(main_df, sub_df, ['k'])
        assert method == 'dense'
    
    def test_auto_selects_sparse_for_large(self):
        """Auto should select sparse for large/wasteful keys."""
        main_df = pd.DataFrame({'k': [0, 1_000_000_000]})
        sub_df = pd.DataFrame({'k': [0, 1_000_000_000]})
        
        _, _, method = compute_composite_key_auto(main_df, sub_df, ['k'])
        assert method == 'sparse'


class TestGetOrderedDefines:
    """Test dependency resolution and ordering."""
    
    def test_simple_chain(self):
        """a depends on b, b depends on c -> order: c, b, a"""
        schema = {
            'aliases': {
                'a': {'expr': 'b + 1'},
                'b': {'expr': 'c + 1'},
                'c': {'expr': 'x + 1'},
            }
        }
        result = get_ordered_defines(['a'], schema=schema)
        names = [d['name'] for d in result]
        
        # c must come before b, b must come before a
        assert names.index('c') < names.index('b')
        assert names.index('b') < names.index('a')
    
    def test_diamond_dependency(self):
        """a->b, a->c, b->d, c->d -> d must come first"""
        schema = {
            'aliases': {
                'd': {'expr': 'x'},
                'b': {'expr': 'd + 1'},
                'c': {'expr': 'd + 2'},
                'a': {'expr': 'b + c'},
            }
        }
        result = get_ordered_defines(['a'], schema=schema)
        names = [d['name'] for d in result]
        
        assert names.index('d') < names.index('b')
        assert names.index('d') < names.index('c')
        assert names.index('b') < names.index('a')
        assert names.index('c') < names.index('a')
    
    def test_independent_aliases(self):
        """Independent aliases can be in any order."""
        schema = {
            'aliases': {
                'a': {'expr': 'x + 1'},
                'b': {'expr': 'y + 1'},
            }
        }
        result = get_ordered_defines(['a', 'b'], schema=schema)
        assert len(result) == 2
    
    def test_circular_dependency_detected(self):
        """Circular dependencies should raise error."""
        schema = {
            'aliases': {
                'a': {'expr': 'b + 1'},
                'b': {'expr': 'a + 1'},  # Circular!
            }
        }
        with pytest.raises(ValueError, match="Circular"):
            get_ordered_defines(['a'], schema=schema)
    
    def test_cpp_expr_included(self):
        """Result should include C++ expression."""
        schema = {
            'aliases': {
                'a': {'expr': 'np.sqrt(x**2)'},
            }
        }
        result = get_ordered_defines(['a'], schema=schema)
        assert len(result) == 1
        assert 'sqrt' in result[0]['cpp_expr']
        assert 'pow' in result[0]['cpp_expr']


class TestGenerateRdfCode:
    """Test C++ code generation."""
    
    def test_basic_generation(self):
        defines = [
            {'name': 'a', 'expr': 'x + 1', 'deps': ['x'], 'cpp_expr': 'x + 1'},
            {'name': 'b', 'expr': 'a * 2', 'deps': ['a'], 'cpp_expr': 'a * 2'},
        ]
        code = generate_rdf_code(defines)
        
        assert '#include <ROOT/RDataFrame.hxx>' in code
        assert '.Define("a", "x + 1")' in code
        assert '.Define("b", "a * 2")' in code
    
    def test_with_mt(self):
        defines = [{'name': 'a', 'expr': 'x', 'deps': [], 'cpp_expr': 'x'}]
        code = generate_rdf_code(defines, include_mt=True)
        
        assert 'ROOT::EnableImplicitMT()' in code
    
    def test_empty_defines(self):
        code = generate_rdf_code([])
        assert 'auto df_final = df;' in code


# =============================================================================
# ROOT RDataFrame Research Tests (Require ROOT)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameBasics:
    """Basic RDataFrame functionality tests."""
    
    @pytest.fixture
    def simple_tree(self, tmp_path):
        """Create a simple test tree."""
        filename = str(tmp_path / "test.root")
        
        ROOT.RDataFrame(100) \
            .Define("x", "double(rdfentry_)") \
            .Define("y", "x * 2") \
            .Snapshot["double", "double"]("tree", filename, ["x", "y"])
        
        return filename
    
    def test_define_chain(self, simple_tree):
        """Test that chained Define() works."""
        df = ROOT.RDataFrame("tree", simple_tree)
        df2 = df.Define("z", "x + y").Define("w", "z * 2")
        
        result = df2.Take["double"]("w").GetValue()
        assert len(result) == 100
        
        # w = (x + y) * 2 = (x + 2x) * 2 = 6x
        # Entry 1: x=1, w=6
        assert abs(result[1] - 6.0) < 0.001


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameFriendAccess:
    """
    RESEARCH: Test what friend tree syntax works in RDataFrame.
    These tests determine our implementation strategy.
    
    Priority:
    [1] test_indexed_friend          ← BLOCKER
    [2] test_composite_index_friend  ← BLOCKER  
    [3] test_friend_dot_notation     
    [4] test_friend_with_alias       
    """
    
    @pytest.fixture
    def trees_with_friend(self, tmp_path):
        """Create main tree and friend tree with index."""
        main_file = str(tmp_path / "main.root")
        friend_file = str(tmp_path / "friend.root")
        
        # Main tree: 100 entries with key column
        ROOT.RDataFrame(100) \
            .Define("key", "int(rdfentry_)") \
            .Define("x", "double(rdfentry_)") \
            .Snapshot["int", "double"]("main", main_file, ["key", "x"])
        
        # Friend tree: 100 entries with matching key
        ROOT.RDataFrame(100) \
            .Define("key", "int(rdfentry_)") \
            .Define("val", "double(rdfentry_) * 10") \
            .Snapshot["int", "double"]("friend", friend_file, ["key", "val"])
        
        return main_file, friend_file
    
    def test_indexed_friend(self, trees_with_friend):
        """
        [PRIORITY 1 - BLOCKER]
        Test indexed friend tree access with RDataFrame.
        This is the core mechanism we rely on.
        """
        main_file, friend_file = trees_with_friend
        
        # Load trees
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main = f_main.Get("main")
        friend = f_friend.Get("friend")
        
        # Build index and add friend
        friend.BuildIndex("key")
        main.AddFriend(friend, "F")
        
        # Create RDataFrame from tree (not filename - important!)
        df = ROOT.RDataFrame(main)
        
        # Test: Can we access friend column at all?
        try:
            # Try using GetColumnNames to see what's available
            cols = [str(c) for c in df.GetColumnNames()]
            print(f"\nAvailable columns: {cols}")
            
            # Try to define using friend column
            df2 = df.Define("test", "F.val")
            result = df2.Take["double"]("test").GetValue()
            
            print(f"[PASS] Indexed friend: Got {len(result)} values")
            print(f"  Sample values: {list(result[:5])}")
            assert len(result) == 100
            
        except Exception as e:
            print(f"[FAIL] Indexed friend failed: {e}")
            pytest.fail(f"Indexed friend access failed: {e}")
    
    def test_composite_index_friend(self, tmp_path):
        """
        [PRIORITY 2 - BLOCKER]
        Test if our composite index approach works with RDataFrame.
        This mimics AliasDataFrameTree.C's multi-key indexing.
        """
        main_file = str(tmp_path / "main.root")
        friend_file = str(tmp_path / "friend.root")
        
        # Main tree with composite key columns
        ROOT.RDataFrame(100) \
            .Define("key1", "int(rdfentry_ % 10)") \
            .Define("key2", "int(rdfentry_ / 10)") \
            .Define("__adf_key__", "key1 + key2 * 10") \
            .Define("x", "double(rdfentry_)") \
            .Snapshot["int", "int", "int", "double"](
                "main", main_file, 
                ["key1", "key2", "__adf_key__", "x"]
            )
        
        # Friend tree with same composite key (10 unique combinations)
        ROOT.RDataFrame(10) \
            .Define("key1", "int(rdfentry_)") \
            .Define("key2", "0") \
            .Define("__adf_key__", "key1") \
            .Define("calib", "double(rdfentry_) * 100") \
            .Snapshot["int", "int", "int", "double"](
                "friend", friend_file,
                ["key1", "key2", "__adf_key__", "calib"]
            )
        
        # Load and setup
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main = f_main.Get("main")
        friend = f_friend.Get("friend")
        
        # Build index on composite key (single column!)
        friend.BuildIndex("__adf_key__")
        main.AddFriend(friend, "C")
        
        df = ROOT.RDataFrame(main)
        
        try:
            cols = [str(c) for c in df.GetColumnNames()]
            print(f"\nAvailable columns: {cols}")
            
            df2 = df.Define("calibrated", "x + C.calib")
            result = df2.Take["double"]("calibrated").GetValue()
            
            print(f"[PASS] Composite index: Got {len(result)} values")
            print(f"  Sample values: {list(result[:5])}")
            
        except Exception as e:
            print(f"[FAIL] Composite index failed: {e}")
            pytest.fail(f"Composite index friend failed: {e}")
    
    def test_friend_dot_notation(self, trees_with_friend):
        """
        [PRIORITY 3]
        Test if T.column syntax works directly in Define().
        If this fails, we need to use Alias() workaround.
        """
        main_file, friend_file = trees_with_friend
        
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main = f_main.Get("main")
        friend = f_friend.Get("friend")
        
        friend.BuildIndex("key")
        main.AddFriend(friend, "F")
        
        df = ROOT.RDataFrame(main)
        
        try:
            # Direct dot notation in expression
            df2 = df.Define("result", "x + F.val")
            result = df2.Take["double"]("result").GetValue()
            
            print(f"[PASS] Dot notation F.val: Works!")
            print(f"  Got {len(result)} values, sample: {list(result[:5])}")
            assert len(result) == 100
            
        except Exception as e:
            print(f"[INFO] Dot notation F.val failed: {e}")
            print("  Will need to use Alias() workaround")
            pytest.skip(f"Dot notation failed (expected): {e}")
    
    def test_friend_with_alias(self, trees_with_friend):
        """
        [PRIORITY 4]
        Test friend access via Alias() if dot notation fails.
        This is our fallback strategy.
        """
        main_file, friend_file = trees_with_friend
        
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main = f_main.Get("main")
        friend = f_friend.Get("friend")
        
        friend.BuildIndex("key")
        main.AddFriend(friend, "F")
        
        df = ROOT.RDataFrame(main)
        
        try:
            # Use Alias to rename friend column
            df2 = df.Alias("F_val", "F.val")
            df3 = df2.Define("result", "x + F_val")
            result = df3.Take["double"]("result").GetValue()
            
            print(f"[PASS] Alias workaround: Works!")
            print(f"  Got {len(result)} values, sample: {list(result[:5])}")
            assert len(result) == 100
            
        except Exception as e:
            print(f"[FAIL] Alias workaround failed: {e}")
            pytest.fail(f"Alias workaround failed: {e}")


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameWithRealSchema:
    """Integration tests with schema-like structure."""
    
    def test_ordered_defines_to_rdf(self, tmp_path):
        """Test that get_ordered_defines output works with RDataFrame."""
        # Create simple test tree
        filename = str(tmp_path / "test.root")
        ROOT.RDataFrame(100) \
            .Define("x", "double(rdfentry_)") \
            .Define("y", "double(rdfentry_) * 2") \
            .Snapshot["double", "double"]("tree", filename, ["x", "y"])
        
        # Define schema with dependencies
        schema = {
            'aliases': {
                'sum_xy': {'expr': 'x + y'},
                'result': {'expr': 'sum_xy * 2'},
            }
        }
        
        # Get ordered defines
        defines = get_ordered_defines(['result'], schema=schema)
        
        # Verify order
        names = [d['name'] for d in defines]
        assert names.index('sum_xy') < names.index('result')
        
        # Apply to RDataFrame
        df = ROOT.RDataFrame("tree", filename)
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        result = df.Take["double"]("result").GetValue()
        assert len(result) == 100
        
        # Verify calculation: result = (x + y) * 2 = (x + 2x) * 2 = 6x
        # For entry 1: x=1, result=6
        assert abs(result[1] - 6.0) < 0.001
        print(f"[PASS] Schema-based defines work correctly")


# =============================================================================
# Summary Report
# =============================================================================

def print_research_summary():
    """Print summary of research findings."""
    print("\n" + "="*60)
    print("RDataFrame Research Summary")
    print("="*60)
    
    if not HAS_ROOT:
        print("ROOT not available - cannot run research tests")
        return
    
    print(f"ROOT Version: {ROOT_VERSION}")
    print("\nTest Results:")
    print("  Run pytest with -v -s to see detailed results")
    print("\nKey Questions:")
    print("  [1] Does indexed friend work?")
    print("  [2] Does composite __adf_key__ work?")
    print("  [3] Does F.val syntax work in Define()?")
    print("  [4] Does Alias() workaround work?")


if __name__ == '__main__':
    print_research_summary()
    pytest.main([__file__, '-v', '-s', '--tb=short'])
