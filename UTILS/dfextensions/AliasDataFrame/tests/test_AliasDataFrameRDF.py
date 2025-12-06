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
    
    def test_numpy_pi_full_module(self):
        """Test numpy.pi (full module name) → M_PI."""
        result = to_cpp_expr('2 * numpy.pi * r')
        assert 'M_PI' in result
        assert 'numpy' not in result
    
    def test_math_pi(self):
        """Test math.pi → M_PI."""
        result = to_cpp_expr('2 * math.pi * r')
        assert 'M_PI' in result
        assert 'math' not in result
    
    def test_np_e(self):
        """Test np.e → M_E."""
        result = to_cpp_expr('np.exp(1) == np.e')
        assert 'M_E' in result
        assert 'np.e' not in result
    
    def test_numpy_e_full_module(self):
        """Test numpy.e (full module name) → M_E."""
        result = to_cpp_expr('x * numpy.e')
        assert 'M_E' in result
        assert 'numpy' not in result
    
    def test_math_e(self):
        """Test math.e → M_E."""
        result = to_cpp_expr('x * math.e')
        assert 'M_E' in result
        assert 'math' not in result
    
    def test_pi_and_e_combined(self):
        """Test expression with both pi and e constants."""
        result = to_cpp_expr('np.pi * np.e')
        assert 'M_PI' in result
        assert 'M_E' in result
        assert 'np.' not in result
    
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
# Composite Key Tests
# =============================================================================

class TestCompositeKeyHelpers:
    """Test composite key utility functions."""
    
    def test_get_composite_key_column_name(self):
        """Test standard naming convention."""
        from AliasDataFrameRDF import get_composite_key_column_name
        
        assert get_composite_key_column_name('DTrack0') == '__adf_key_DTrack0__'
        assert get_composite_key_column_name('S') == '__adf_key_S__'
        assert get_composite_key_column_name('calibration') == '__adf_key_calibration__'
    
    def test_check_dense_overflow_safe(self):
        """Test overflow check for safe values."""
        from AliasDataFrameRDF import check_dense_overflow
        
        # Small values - definitely safe
        is_safe, compact_range = check_dense_overflow([10, 20, 30])
        assert is_safe
        assert compact_range == 10 * 20 * 30
        
        # Typical TPC case: side=2, row=152, drift=28
        is_safe, compact_range = check_dense_overflow([2, 152, 28])
        assert is_safe
        assert compact_range == 2 * 152 * 28
    
    def test_check_dense_overflow_unsafe(self):
        """Test overflow check for large values."""
        from AliasDataFrameRDF import check_dense_overflow
        
        # Values that would overflow int64: 2^30 * 2^30 * 2^30 = 2^90 > 2^63
        is_safe, _ = check_dense_overflow([2**30, 2**30, 2**30])
        assert not is_safe
    
    def test_generate_dense_cpp_expression_single_key(self):
        """Test C++ expression for single key (trivial case)."""
        from AliasDataFrameRDF import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['key'], [100])
        assert result == 'key'
    
    def test_generate_dense_cpp_expression_two_keys(self):
        """Test C++ expression for two keys."""
        from AliasDataFrameRDF import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['side', 'row'], [2, 152])
        assert result == 'side + row * 2'
    
    def test_generate_dense_cpp_expression_three_keys(self):
        """Test C++ expression for three keys."""
        from AliasDataFrameRDF import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['a', 'b', 'c'], [10, 20, 30])
        assert result == 'a + b * 10 + c * 10 * 20'
    
    def test_generate_dense_cpp_expression_four_keys(self):
        """Test C++ expression for four keys."""
        from AliasDataFrameRDF import generate_dense_cpp_expression
        
        result = generate_dense_cpp_expression(['k1', 'k2', 'k3', 'k4'], [2, 3, 4, 5])
        assert result == 'k1 + k2 * 2 + k3 * 2 * 3 + k4 * 2 * 3 * 4'


class TestShouldUseSparse:
    """Test sparse vs dense decision function."""
    
    def test_small_dense_data(self):
        """Dense data with small range should use dense."""
        from AliasDataFrameRDF import should_use_sparse
        
        df = pd.DataFrame({
            'k1': np.array([0, 1, 2, 3, 4], dtype=np.int32),
            'k2': np.array([0, 0, 1, 1, 2], dtype=np.int32),
        })
        
        assert not should_use_sparse(df, ['k1', 'k2'])
    
    def test_large_sparse_data(self):
        """Sparse data with large gaps should use sparse."""
        from AliasDataFrameRDF import should_use_sparse
        
        # Only 5 unique combinations but max values suggest huge range
        df = pd.DataFrame({
            'k1': np.array([0, 1000000, 2000000, 3000000, 4000000], dtype=np.int64),
            'k2': np.array([0, 1000000, 2000000, 3000000, 4000000], dtype=np.int64),
        })
        
        assert should_use_sparse(df, ['k1', 'k2'])
    
    def test_overflow_triggers_sparse(self):
        """Data that would overflow int32 should use sparse."""
        from AliasDataFrameRDF import should_use_sparse
        
        # max = 100000, range = 100001^2 > 2^31
        df = pd.DataFrame({
            'k1': np.array([0, 100000], dtype=np.int64),
            'k2': np.array([0, 100000], dtype=np.int64),
        })
        
        assert should_use_sparse(df, ['k1', 'k2'])


class TestComputeCompositeKeyDense:
    """Test dense linearization."""
    
    def test_single_column(self):
        """Single column key is just the column values."""
        from AliasDataFrameRDF import compute_composite_key_dense
        
        df = pd.DataFrame({'key': np.array([0, 5, 10, 15], dtype=np.int32)})
        result = compute_composite_key_dense(df, ['key'])
        
        np.testing.assert_array_equal(result, [0, 5, 10, 15])
    
    def test_two_columns(self):
        """Two column key: k0 + k1 * max0."""
        from AliasDataFrameRDF import compute_composite_key_dense
        
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
        from AliasDataFrameRDF import compute_composite_key_dense
        
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
        from AliasDataFrameRDF import compute_composite_key_dense
        
        df = pd.DataFrame({
            'k1': np.array([0, 1], dtype=np.int32),
            'k2': np.array([0, 1], dtype=np.int32),
        })
        
        # Use larger max values than data requires
        result = compute_composite_key_dense(df, ['k1', 'k2'], max_values=[10, 10])
        
        # k1=0,k2=0 -> 0 + 0*10 = 0
        # k1=1,k2=1 -> 1 + 1*10 = 11
        np.testing.assert_array_equal(result, [0, 11])


class TestComputeCompositeKeySparse:
    """Test sparse key mapping."""
    
    def test_basic_mapping(self):
        """Basic sparse mapping assigns sequential IDs."""
        from AliasDataFrameRDF import compute_composite_key_sparse
        
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
        from AliasDataFrameRDF import compute_composite_key_sparse
        
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
        from AliasDataFrameRDF import compute_composite_key_sparse
        
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


class TestComputeCompositeKeyAuto:
    """Test automatic dense/sparse selection."""
    
    def test_auto_selects_dense_for_compact_data(self):
        """Auto should select dense for compact data."""
        from AliasDataFrameRDF import compute_composite_key_auto
        
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
        from AliasDataFrameRDF import compute_composite_key_auto
        
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
        from AliasDataFrameRDF import compute_composite_key_auto
        
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
# Modular RDataFrame API Tests
# =============================================================================

import pandas as pd
import numpy as np
from AliasDataFrame import AliasDataFrame


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestModularRDFSetup:
    """Test modular RDataFrame setup functions."""
    
    @pytest.fixture
    def sample_adf_with_file(self, tmp_path):
        """Create sample AliasDataFrame and ROOT file."""
        df = pd.DataFrame({
            'x': np.random.randn(1000).astype(np.float32),
            'y': np.random.randn(1000).astype(np.float32),
            'row': np.random.randint(0, 100, 1000).astype(np.int32),
        })
        sub_df = pd.DataFrame({
            'row': np.arange(100).astype(np.int32),
            'calib': np.random.randn(100).astype(np.float32),
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('T', AliasDataFrame(sub_df), index_columns='row')
        adf.add_alias('r2', 'x**2 + y**2')
        
        filepath = str(tmp_path / "test_rdf.root")
        adf.export_tree(filepath, 'tree')
        return adf, filepath
    
    def test_setup_rdf_returns_tuple(self, sample_adf_with_file):
        """Test setup_rdf_with_friends returns (rdf, file_handle)."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, filepath = sample_adf_with_file
        
        result = setup_rdf_with_friends(adf, filepath)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        rdf, file_handle = result
        assert rdf is not None
        assert file_handle is not None
        assert not file_handle.IsZombie()
    
    def test_setup_rdf_can_count(self, sample_adf_with_file):
        """Test RDataFrame can perform actions."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, filepath = sample_adf_with_file
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        count = rdf.Count().GetValue()
        assert count == 1000
    
    def test_setup_rdf_file_not_found(self, sample_adf_with_file):
        """Test raises OSError for missing file."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, _ = sample_adf_with_file
        
        with pytest.raises(OSError):
            setup_rdf_with_friends(adf, "nonexistent.root")
    
    def test_setup_rdf_has_columns(self, sample_adf_with_file):
        """Test RDataFrame has expected columns."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, filepath = sample_adf_with_file
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        columns = [str(c) for c in rdf.GetColumnNames()]
        assert 'x' in columns
        assert 'y' in columns
        assert 'row' in columns


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTMemFileBranch:
    """
    Test TBranch::SetFile() approach for adding branches to read-only trees.
    
    This is a CRITICAL verification test for Phase 5.3 runtime composite keys.
    The approach: use SetFile() to store composite key branches in TMemFile
    while keeping original data in read-only file.
    """
    
    def test_setfile_basic(self, tmp_path):
        """
        Test that TBranch::SetFile() allows adding branch to read-only tree.
        
        Approach:
        1. Create file with tree, close it
        2. Reopen in READ mode (read-only)
        3. Create TMemFile
        4. Add branch to tree, redirect storage via SetFile()
        5. Fill branch, verify data accessible
        """
        import ROOT
        
        # Step 1: Create original file with data
        filepath = str(tmp_path / "readonly_test.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        tree = ROOT.TTree("T", "Test tree")
        x = np.array([0.0], dtype=np.float32)
        key = np.array([0], dtype=np.int32)
        tree.Branch("x", x, "x/F")
        tree.Branch("key", key, "key/I")
        
        for i in range(100):
            x[0] = float(i)
            key[0] = i
            tree.Fill()
        
        tree.Write()
        f_create.Close()
        
        # Step 2: Reopen in READ mode
        f_readonly = ROOT.TFile.Open(filepath, "READ")
        tree = f_readonly.Get("T")
        
        assert tree is not None
        assert tree.GetEntries() == 100
        
        # Step 3: Create TMemFile
        memfile = ROOT.TMemFile("branch_storage", "RECREATE")
        
        # Step 4: Add new branch, redirect to TMemFile
        composite_key = np.array([0], dtype=np.int64)
        new_branch = tree.Branch("__adf_key__", composite_key, "__adf_key__/L")
        
        # CRITICAL: Redirect branch storage to TMemFile
        new_branch.SetFile(memfile)
        
        # Step 5: Fill the new branch
        for i in range(int(tree.GetEntries())):
            tree.GetEntry(i)
            composite_key[0] = i * 10  # Some computed value
            new_branch.Fill()
        
        # Step 6: Verify branch is accessible
        assert tree.GetBranch("__adf_key__") is not None
        
        # Verify we can read back values
        # Reset and read entry 5
        new_branch.GetEntry(5)
        assert composite_key[0] == 50, f"Expected 50, got {composite_key[0]}"
        
        print("✅ test_setfile_basic PASSED")
        
        # Cleanup
        f_readonly.Close()
    
    def test_setfile_with_buildindex(self, tmp_path):
        """
        Test that BuildIndex() works on a branch stored in TMemFile.
        
        Creates shuffled friend tree to verify index lookup actually works.
        """
        import ROOT
        
        filepath = str(tmp_path / "buildindex_test.root")
        
        # Create file with main and SHUFFLED friend tree
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree: keys 0-9 in order
        main_tree = ROOT.TTree("main", "Main tree")
        key = np.array([0], dtype=np.int32)
        value = np.array([0.0], dtype=np.float32)
        main_tree.Branch("key", key, "key/I")
        main_tree.Branch("value", value, "value/F")
        
        for i in range(10):
            key[0] = i
            value[0] = float(i)
            main_tree.Fill()
        
        main_tree.Write()
        
        # Friend tree with SHUFFLED order
        friend_tree = ROOT.TTree("friend", "Friend tree")
        f_key = np.array([0], dtype=np.int32)
        f_data = np.array([0.0], dtype=np.float32)
        friend_tree.Branch("key", f_key, "key/I")
        friend_tree.Branch("data", f_data, "data/F")
        
        # Shuffle: data = key * 100, but stored in shuffled order
        shuffled_keys = [9, 7, 5, 3, 1, 8, 6, 4, 2, 0]
        for k in shuffled_keys:
            f_key[0] = k
            f_data[0] = k * 100.0  # data = key * 100
            friend_tree.Fill()
        
        friend_tree.Write()
        f_create.Close()
        
        # Reopen in READ mode
        f = ROOT.TFile.Open(filepath, "READ")
        main_tree = f.Get("main")
        friend_tree = f.Get("friend")
        
        # Build index on friend tree's key column
        friend_tree.BuildIndex("key")
        main_tree.AddFriend(friend_tree, "F")
        
        # Verify join correctness: F.data should = key * 100
        errors = []
        for i in range(int(main_tree.GetEntries())):
            main_tree.GetEntry(i)
            expected = main_tree.key * 100.0
            actual = friend_tree.data
            
            if abs(actual - expected) > 0.001:
                errors.append(f"Entry {i}: key={main_tree.key}, expected {expected}, got {actual}")
        
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            pytest.fail(f"BuildIndex join incorrect: {errors}")
        
        print("✅ test_setfile_with_buildindex PASSED")
        
        f.Close()
    
    def test_setfile_composite_key_3keys(self, tmp_path):
        """
        CRITICAL: Test full composite key scenario with 3 index columns.
        
        This tests the actual use case:
        - Main tree with 3 key columns
        - Friend tree with 3 key columns + data (SHUFFLED)
        - Runtime composite key added via SetFile()
        - BuildIndex on composite key
        - Verify join returns correct values: F.calib == k1*100 + k2*10 + k3
        """
        import ROOT
        
        filepath = str(tmp_path / "composite_3key.root")
        
        # Create file with main and shuffled friend tree
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree: 5 entries with known key combinations
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("value", val, "value/F")
        
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree - SHUFFLED order, calib = k1*100 + k2*10 + k3
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        # All 8 combinations (2x2x2), SHUFFLED
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0), 
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend = f.Get("friend")
        
        # Create TMemFile for composite key branches
        memfile_main = ROOT.TMemFile("main_key", "RECREATE")
        memfile_friend = ROOT.TMemFile("friend_key", "RECREATE")
        
        # Add composite key to FRIEND tree
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        friend_key_branch.SetFile(memfile_friend)
        
        # Compute composite key: k1 + k2*2 + k3*2*2 (max values: 2, 2, 2)
        max_k1, max_k2 = 2, 2
        
        for i in range(int(friend.GetEntries())):
            friend.GetEntry(i)
            friend_ckey[0] = int(friend.k1 + friend.k2 * max_k1 + friend.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        # BuildIndex on composite key
        friend.BuildIndex("__adf_key__")
        
        # Add composite key to MAIN tree
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        # Add friend
        main.AddFriend(friend, "F")
        
        # VERIFY CORRECTNESS
        # For each main entry, F.calib should equal k1*100 + k2*10 + k3
        errors = []
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            expected = float(main.k1 * 100 + main.k2 * 10 + main.k3)
            actual = friend.calib
            
            if abs(actual - expected) > 0.001:
                errors.append(f"Entry {i}: keys=({main.k1},{main.k2},{main.k3}), expected {expected}, got {actual}")
        
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            pytest.fail(f"Composite key join incorrect: {errors}")
        
        print("✅ test_setfile_composite_key_3keys PASSED")
        
        f.Close()
    
    def test_setfile_composite_key_with_rdf(self, tmp_path):
        """
        Test that SetFile composite keys work with RDataFrame.
        
        Same setup as test_setfile_composite_key_3keys but verify via RDF.
        This is the ultimate test - if this passes, we can implement runtime
        composite keys for RDataFrame friend joins.
        """
        import ROOT
        
        filepath = str(tmp_path / "composite_rdf.root")
        
        # Create file with main and shuffled friend tree
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("value", val, "value/F")
        
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree - SHUFFLED
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0), 
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend = f.Get("friend")
        
        # Create TMemFile for composite key branches
        memfile_main = ROOT.TMemFile("main_key_rdf", "RECREATE")
        memfile_friend = ROOT.TMemFile("friend_key_rdf", "RECREATE")
        
        # Add composite key to FRIEND tree
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        friend_key_branch.SetFile(memfile_friend)
        
        max_k1, max_k2 = 2, 2
        
        for i in range(int(friend.GetEntries())):
            friend.GetEntry(i)
            friend_ckey[0] = int(friend.k1 + friend.k2 * max_k1 + friend.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend.BuildIndex("__adf_key__")
        
        # Add composite key to MAIN tree
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        # Add friend
        main.AddFriend(friend, "F")
        
        # Create RDataFrame and verify
        rdf = ROOT.RDataFrame(main)
        
        # Define expected and compute difference (cast to float to match F.calib type)
        rdf = rdf.Define("expected", "(float)(k1 * 100 + k2 * 10 + k3)")
        rdf = rdf.Define("actual", "F.calib")
        rdf = rdf.Define("diff", "F.calib - expected")
        
        # Get results
        diffs = list(rdf.Take['float']("diff").GetValue())
        expected_vals = list(rdf.Take['float']("expected").GetValue())
        actual_vals = list(rdf.Take['float']("actual").GetValue())
        
        print(f"\nRDataFrame composite key test:")
        print(f"  Expected: {expected_vals}")
        print(f"  Actual:   {actual_vals}")
        print(f"  Diffs:    {diffs}")
        
        # All differences should be 0
        assert all(abs(d) < 0.001 for d in diffs), \
            f"RDataFrame join incorrect! Expected: {expected_vals}, Actual: {actual_vals}"
        
        print("✅ test_setfile_composite_key_with_rdf PASSED")
        
        f.Close()
    
    # =========================================================================
    # Isolation Tests: Determine which tree causes RDataFrame failure
    # =========================================================================
    
    def test_setfile_friend_only_with_rdf(self, tmp_path):
        """
        Test A: SetFile on FRIEND tree only, main tree has key in file.
        
        If this passes: Only friend tree needs TMemFile handling.
        If this fails: SetFile doesn't work for friend trees with RDF.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_friend_setfile.root")
        
        # Create file with pre-computed composite key IN MAIN TREE
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree WITH __adf_key__ branch (pre-computed, in file)
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        main_key = np.array([0], dtype=np.int64)  # Pre-computed in file
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("__adf_key__", main_key, "__adf_key__/L")
        main.Branch("value", val, "value/F")
        
        max_k1, max_k2 = 2, 2
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            main_key[0] = a + b * max_k1 + c * max_k1 * max_k2
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree WITHOUT __adf_key__ branch (will add via SetFile)
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        # Shuffled friend data
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0),
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend = f.Get("friend")
        
        # ONLY friend tree uses SetFile
        memfile_friend = ROOT.TMemFile("friend_key_only", "RECREATE")
        
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        friend_key_branch.SetFile(memfile_friend)
        
        for i in range(int(friend.GetEntries())):
            friend.GetEntry(i)
            friend_ckey[0] = int(friend.k1 + friend.k2 * max_k1 + friend.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend.BuildIndex("__adf_key__")
        main.AddFriend(friend, "F")
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("diff", "F.calib - expected")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        assert all(abs(d) < 0.001 for d in diffs), f"Wrong values: {diffs}"
        print("✅ test_setfile_friend_only_with_rdf PASSED")
        
        f.Close()
    
    def test_setfile_main_only_with_rdf(self, tmp_path):
        """
        Test B: SetFile on MAIN tree only, friend tree has key in file.
        
        If this passes: Only main tree can use SetFile.
        If this fails: SetFile doesn't work for main tree with RDF.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_main_setfile.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree WITHOUT __adf_key__ (will add via SetFile)
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("value", val, "value/F")
        
        max_k1, max_k2 = 2, 2
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree WITH __adf_key__ branch (pre-computed, in file)
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        friend_key = np.array([0], dtype=np.int64)  # Pre-computed in file
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("__adf_key__", friend_key, "__adf_key__/L")
        friend.Branch("calib", calib, "calib/F")
        
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0),
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            friend_key[0] = a + b * max_k1 + c * max_k1 * max_k2
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend = f.Get("friend")
        
        # ONLY main tree uses SetFile
        memfile_main = ROOT.TMemFile("main_key_only", "RECREATE")
        
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        # BuildIndex on friend (which has key in file)
        friend.BuildIndex("__adf_key__")
        main.AddFriend(friend, "F")
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("diff", "F.calib - expected")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        assert all(abs(d) < 0.001 for d in diffs), f"Wrong values: {diffs}"
        print("✅ test_setfile_main_only_with_rdf PASSED")
        
        f.Close()
    
    def test_clone_friend_to_memfile_with_rdf(self, tmp_path):
        """
        Test C: Clone ENTIRE friend tree to TMemFile (like AliasDataFrameTree.C).
        
        If this passes: Full clone works for RDataFrame - viable fallback for small trees.
        If this fails: TMemFile doesn't work with RDataFrame at all.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_clone_friend.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree WITH __adf_key__ (pre-computed)
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        main_key = np.array([0], dtype=np.int64)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("__adf_key__", main_key, "__adf_key__/L")
        main.Branch("value", val, "value/F")
        
        max_k1, max_k2 = 2, 2
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            main_key[0] = a + b * max_k1 + c * max_k1 * max_k2
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree WITHOUT __adf_key__ (will clone and add)
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0),
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend_orig = f.Get("friend")
        
        # Clone ENTIRE friend tree to TMemFile
        memfile = ROOT.TMemFile("friend_clone", "RECREATE")
        memfile.cd()
        
        friend_clone = friend_orig.CloneTree(-1, "fast")
        friend_clone.SetDirectory(memfile)
        
        # Add composite key branch to clone
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend_clone.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        
        for i in range(int(friend_clone.GetEntries())):
            friend_clone.GetEntry(i)
            friend_ckey[0] = int(friend_clone.k1 + friend_clone.k2 * max_k1 + friend_clone.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend_clone.BuildIndex("__adf_key__")
        main.AddFriend(friend_clone, "F")
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("diff", "F.calib - expected")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        assert all(abs(d) < 0.001 for d in diffs), f"Wrong values: {diffs}"
        print("✅ test_clone_friend_to_memfile_with_rdf PASSED")
        print(f"   (Cloned {friend_clone.GetEntries()} entries to TMemFile)")
        
        f.Close()
    
    def test_clone_friend_setfile_main_with_rdf(self, tmp_path):
        """
        Test D: Clone friend to TMemFile, SetFile on main tree.
        
        This tests the realistic scenario:
        - Main tree is read-only and large (cannot clone)
        - Main tree key added via SetFile
        - Friend tree is small, fully cloned to TMemFile
        
        If this passes: We have a viable solution for runtime composite keys.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_combined.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree WITHOUT __adf_key__ (will use SetFile)
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("value", val, "value/F")
        
        max_k1, max_k2 = 2, 2
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree WITHOUT __adf_key__ (will clone fully)
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        friend_keys = [(1,1,1), (0,0,0), (1,0,1), (0,1,0),
                      (1,1,0), (0,0,1), (1,0,0), (0,1,1)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend_orig = f.Get("friend")
        
        # SetFile for MAIN tree key
        memfile_main = ROOT.TMemFile("main_key_combined", "RECREATE")
        
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        # Clone ENTIRE friend tree to TMemFile
        memfile_friend = ROOT.TMemFile("friend_clone_combined", "RECREATE")
        memfile_friend.cd()
        
        friend_clone = friend_orig.CloneTree(-1, "fast")
        friend_clone.SetDirectory(memfile_friend)
        
        # Add composite key branch to clone
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend_clone.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        
        for i in range(int(friend_clone.GetEntries())):
            friend_clone.GetEntry(i)
            friend_ckey[0] = int(friend_clone.k1 + friend_clone.k2 * max_k1 + friend_clone.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend_clone.BuildIndex("__adf_key__")
        main.AddFriend(friend_clone, "F")
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("diff", "F.calib - expected")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        assert all(abs(d) < 0.001 for d in diffs), f"Wrong values: {diffs}"
        print("✅ test_clone_friend_setfile_main_with_rdf PASSED")
        print("   Main: SetFile for key branch")
        print(f"   Friend: Full clone ({friend_clone.GetEntries()} entries)")
        
        f.Close()
    
    # =========================================================================
    # Robustness Tests E, F, G: Verify correctness before implementation
    # =========================================================================
    
    def test_clone_friend_setfile_main_shuffled_correctness(self, tmp_path):
        """
        Test E (CRITICAL): Verify Test D approach with SHUFFLED friend data.
        
        This is the definitive correctness test. If this passes,
        we can confidently implement runtime composite keys.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_shuffled_correctness.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree: keys in sequential order
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        val = np.array([0.0], dtype=np.float32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        main.Branch("value", val, "value/F")
        
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for i, (a,b,c) in enumerate(main_keys):
            k1[0], k2[0], k3[0] = a, b, c
            val[0] = float(i)
            main.Fill()
        main.Write()
        
        # Friend tree: DELIBERATELY SHUFFLED order
        # calib = k1*100 + k2*10 + k3 (deterministic, verifiable)
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        # Shuffled: NOT in same order as main (reversed + shuffled)
        friend_keys = [(1,1,0), (0,0,1), (1,0,0), (0,1,0), (0,0,0)]
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)  # Deterministic formula
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Reopen READ-ONLY
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend_orig = f.Get("friend")
        
        max_k1, max_k2 = 2, 2
        
        # Main tree: SetFile for key branch
        memfile_main = ROOT.TMemFile("main_key_shuffled", "RECREATE")
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        # Friend tree: Full clone to TMemFile
        memfile_friend = ROOT.TMemFile("friend_clone_shuffled", "RECREATE")
        memfile_friend.cd()
        friend_clone = friend_orig.CloneTree(-1, "fast")
        friend_clone.SetDirectory(memfile_friend)
        
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend_clone.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        
        for i in range(int(friend_clone.GetEntries())):
            friend_clone.GetEntry(i)
            friend_ckey[0] = int(friend_clone.k1 + friend_clone.k2 * max_k1 + friend_clone.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend_clone.BuildIndex("__adf_key__")
        main.AddFriend(friend_clone, "F")
        
        # Test with RDataFrame (cast to float to match F.calib type)
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected_calib", "(float)(k1 * 100 + k2 * 10 + k3)")
        rdf = rdf.Define("diff", "F.calib - expected_calib")
        
        expected_vals = list(rdf.Take['float']("expected_calib").GetValue())
        actual_vals = list(rdf.Take['float']("F.calib").GetValue())
        diffs = list(rdf.Take['float']("diff").GetValue())
        
        print(f"\nShuffled correctness test:")
        print(f"  Main keys:  {main_keys}")
        print(f"  Expected:   {expected_vals}")
        print(f"  Actual:     {actual_vals}")
        print(f"  Diffs:      {diffs}")
        
        # CORRECTNESS CHECK: All diffs must be 0
        assert all(abs(d) < 0.001 for d in diffs), \
            f"SHUFFLED JOIN INCORRECT! Expected: {expected_vals}, Actual: {actual_vals}"
        
        print("✅ test_clone_friend_setfile_main_shuffled_correctness PASSED")
        
        f.Close()
    
    def test_multiple_subframes_runtime_composite(self, tmp_path):
        """
        Test F: Runtime composite keys with MULTIPLE subframes.
        
        Real TPC calibration has DTrack0, DITS0FitSide, etc.
        This verifies the approach scales to multiple friend trees.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_multi_subframe.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        
        main_keys = [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
        for (a,b,c) in main_keys:
            k1[0], k2[0], k3[0] = a, b, c
            main.Fill()
        main.Write()
        
        # Subframe 1: S1 with calib1 = k1*100 + k2*10 + k3
        s1 = ROOT.TTree("S1", "Subframe 1")
        s1_k1 = np.array([0], dtype=np.int32)
        s1_k2 = np.array([0], dtype=np.int32)
        s1_k3 = np.array([0], dtype=np.int32)
        s1_calib = np.array([0.0], dtype=np.float32)
        s1.Branch("k1", s1_k1, "k1/I")
        s1.Branch("k2", s1_k2, "k2/I")
        s1.Branch("k3", s1_k3, "k3/I")
        s1.Branch("calib", s1_calib, "calib/F")
        
        s1_keys = [(1,1,0), (0,0,0), (1,0,0), (0,1,0)]  # Shuffled
        for (a,b,c) in s1_keys:
            s1_k1[0], s1_k2[0], s1_k3[0] = a, b, c
            s1_calib[0] = float(a*100 + b*10 + c)
            s1.Fill()
        s1.Write()
        
        # Subframe 2: S2 with calib2 = k1*1000 + k2*100 + k3*10
        s2 = ROOT.TTree("S2", "Subframe 2")
        s2_k1 = np.array([0], dtype=np.int32)
        s2_k2 = np.array([0], dtype=np.int32)
        s2_k3 = np.array([0], dtype=np.int32)
        s2_calib = np.array([0.0], dtype=np.float32)
        s2.Branch("k1", s2_k1, "k1/I")
        s2.Branch("k2", s2_k2, "k2/I")
        s2.Branch("k3", s2_k3, "k3/I")
        s2.Branch("calib", s2_calib, "calib/F")
        
        s2_keys = [(0,1,0), (1,0,0), (0,0,0), (1,1,0)]  # Different shuffle
        for (a,b,c) in s2_keys:
            s2_k1[0], s2_k2[0], s2_k3[0] = a, b, c
            s2_calib[0] = float(a*1000 + b*100 + c*10)
            s2.Fill()
        s2.Write()
        
        f_create.Close()
        
        # Setup with runtime composite keys for both subframes
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        s1_orig = f.Get("S1")
        s2_orig = f.Get("S2")
        
        max_k1, max_k2 = 2, 2
        
        # Main tree: SetFile for BOTH subframe keys
        memfile_main = ROOT.TMemFile("main_keys_multi", "RECREATE")
        
        main_key_s1 = np.array([0], dtype=np.int64)
        main_key_s2 = np.array([0], dtype=np.int64)
        main_branch_s1 = main.Branch("__adf_key_S1__", main_key_s1, "__adf_key_S1__/L")
        main_branch_s2 = main.Branch("__adf_key_S2__", main_key_s2, "__adf_key_S2__/L")
        main_branch_s1.SetFile(memfile_main)
        main_branch_s2.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            key_val = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_s1[0] = key_val
            main_key_s2[0] = key_val  # Same key formula for both
            main_branch_s1.Fill()
            main_branch_s2.Fill()
        
        # Clone S1
        memfile_s1 = ROOT.TMemFile("s1_clone", "RECREATE")
        memfile_s1.cd()
        s1_clone = s1_orig.CloneTree(-1, "fast")
        s1_clone.SetDirectory(memfile_s1)
        
        s1_ckey = np.array([0], dtype=np.int64)
        s1_key_branch = s1_clone.Branch("__adf_key_S1__", s1_ckey, "__adf_key_S1__/L")
        for i in range(int(s1_clone.GetEntries())):
            s1_clone.GetEntry(i)
            s1_ckey[0] = int(s1_clone.k1 + s1_clone.k2 * max_k1 + s1_clone.k3 * max_k1 * max_k2)
            s1_key_branch.Fill()
        s1_clone.BuildIndex("__adf_key_S1__")
        
        # Clone S2
        memfile_s2 = ROOT.TMemFile("s2_clone", "RECREATE")
        memfile_s2.cd()
        s2_clone = s2_orig.CloneTree(-1, "fast")
        s2_clone.SetDirectory(memfile_s2)
        
        s2_ckey = np.array([0], dtype=np.int64)
        s2_key_branch = s2_clone.Branch("__adf_key_S2__", s2_ckey, "__adf_key_S2__/L")
        for i in range(int(s2_clone.GetEntries())):
            s2_clone.GetEntry(i)
            s2_ckey[0] = int(s2_clone.k1 + s2_clone.k2 * max_k1 + s2_clone.k3 * max_k1 * max_k2)
            s2_key_branch.Fill()
        s2_clone.BuildIndex("__adf_key_S2__")
        
        # Add friends
        main.AddFriend(s1_clone, "S1")
        main.AddFriend(s2_clone, "S2")
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(main)
        rdf = rdf.Define("expected_s1", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("expected_s2", "k1 * 1000 + k2 * 100 + k3 * 10")
        rdf = rdf.Define("diff_s1", "S1.calib - expected_s1")
        rdf = rdf.Define("diff_s2", "S2.calib - expected_s2")
        
        diffs_s1 = list(rdf.Take['float']("diff_s1").GetValue())
        diffs_s2 = list(rdf.Take['float']("diff_s2").GetValue())
        
        print(f"\nMultiple subframes test:")
        print(f"  S1 diffs: {diffs_s1}")
        print(f"  S2 diffs: {diffs_s2}")
        
        assert all(abs(d) < 0.001 for d in diffs_s1), f"S1 join incorrect: {diffs_s1}"
        assert all(abs(d) < 0.001 for d in diffs_s2), f"S2 join incorrect: {diffs_s2}"
        
        print("✅ test_multiple_subframes_runtime_composite PASSED")
        
        f.Close()
    
    def test_missing_keys_in_friend(self, tmp_path):
        """
        Test G: Behavior when main tree has keys not present in friend tree.
        
        ROOT friend join returns default values (0) for missing keys.
        We verify this doesn't crash and document the behavior.
        """
        import ROOT
        
        filepath = str(tmp_path / "test_missing_keys.root")
        
        f_create = ROOT.TFile.Open(filepath, "RECREATE")
        
        # Main tree: 5 key combinations
        main = ROOT.TTree("main", "Main")
        k1 = np.array([0], dtype=np.int32)
        k2 = np.array([0], dtype=np.int32)
        k3 = np.array([0], dtype=np.int32)
        main.Branch("k1", k1, "k1/I")
        main.Branch("k2", k2, "k2/I")
        main.Branch("k3", k3, "k3/I")
        
        main_keys = [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1)]
        for (a,b,c) in main_keys:
            k1[0], k2[0], k3[0] = a, b, c
            main.Fill()
        main.Write()
        
        # Friend tree: Only 3 of the 5 keys (missing (1,0,0) and (0,0,1))
        friend = ROOT.TTree("friend", "Friend")
        fk1 = np.array([0], dtype=np.int32)
        fk2 = np.array([0], dtype=np.int32)
        fk3 = np.array([0], dtype=np.int32)
        calib = np.array([0.0], dtype=np.float32)
        friend.Branch("k1", fk1, "k1/I")
        friend.Branch("k2", fk2, "k2/I")
        friend.Branch("k3", fk3, "k3/I")
        friend.Branch("calib", calib, "calib/F")
        
        friend_keys = [(0,0,0), (0,1,0), (1,1,0)]  # Missing (1,0,0) and (0,0,1)
        for (a,b,c) in friend_keys:
            fk1[0], fk2[0], fk3[0] = a, b, c
            calib[0] = float(a*100 + b*10 + c)
            friend.Fill()
        friend.Write()
        
        f_create.Close()
        
        # Setup runtime composite key
        f = ROOT.TFile.Open(filepath, "READ")
        main = f.Get("main")
        friend_orig = f.Get("friend")
        
        max_k1, max_k2 = 2, 2
        
        memfile_main = ROOT.TMemFile("main_missing", "RECREATE")
        main_ckey = np.array([0], dtype=np.int64)
        main_key_branch = main.Branch("__adf_key__", main_ckey, "__adf_key__/L")
        main_key_branch.SetFile(memfile_main)
        
        for i in range(int(main.GetEntries())):
            main.GetEntry(i)
            main_ckey[0] = int(main.k1 + main.k2 * max_k1 + main.k3 * max_k1 * max_k2)
            main_key_branch.Fill()
        
        memfile_friend = ROOT.TMemFile("friend_missing", "RECREATE")
        memfile_friend.cd()
        friend_clone = friend_orig.CloneTree(-1, "fast")
        friend_clone.SetDirectory(memfile_friend)
        
        friend_ckey = np.array([0], dtype=np.int64)
        friend_key_branch = friend_clone.Branch("__adf_key__", friend_ckey, "__adf_key__/L")
        for i in range(int(friend_clone.GetEntries())):
            friend_clone.GetEntry(i)
            friend_ckey[0] = int(friend_clone.k1 + friend_clone.k2 * max_k1 + friend_clone.k3 * max_k1 * max_k2)
            friend_key_branch.Fill()
        
        friend_clone.BuildIndex("__adf_key__")
        main.AddFriend(friend_clone, "F")
        
        # Get values via RDF
        rdf = ROOT.RDataFrame(main)
        calib_vals = list(rdf.Take['float']("F.calib").GetValue())
        
        # Expected: 
        # (0,0,0) -> 0, (0,1,0) -> 10, (1,0,0) -> MISSING, (1,1,0) -> 110, (0,0,1) -> MISSING
        # Missing keys typically return 0 or last-read value
        
        print(f"\nMissing keys test:")
        print(f"  Main keys:   {main_keys}")
        print(f"  Friend keys: {friend_keys}")
        print(f"  Calib values: {calib_vals}")
        
        # Document the behavior (don't assert specific values for missing keys)
        # Just verify it doesn't crash and returns correct count
        assert len(calib_vals) == 5, "Should have 5 values"
        
        # Verify the keys that DO exist return correct values
        # Index 0: (0,0,0) -> expected 0
        # Index 1: (0,1,0) -> expected 10
        # Index 3: (1,1,0) -> expected 110
        assert abs(calib_vals[0] - 0.0) < 0.001, f"Key (0,0,0) incorrect: {calib_vals[0]}"
        assert abs(calib_vals[1] - 10.0) < 0.001, f"Key (0,1,0) incorrect: {calib_vals[1]}"
        assert abs(calib_vals[3] - 110.0) < 0.001, f"Key (1,1,0) incorrect: {calib_vals[3]}"
        
        print("✅ test_missing_keys_in_friend PASSED")
        print("   Note: Missing keys return default/stale values (ROOT behavior)")
        
        f.Close()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRuntimeCompositeKey:
    """Test runtime composite key generation for >2 index columns."""
    
    @pytest.fixture
    def adf_with_3key_subframe(self, tmp_path):
        """Create ADF with 3-key subframe for testing composite key generation."""
        # Main dataframe with 3 key columns
        np.random.seed(42)
        n_main = 100
        df_main = pd.DataFrame({
            'k1': np.random.randint(0, 3, n_main).astype(np.int32),   # 0-2
            'k2': np.random.randint(0, 5, n_main).astype(np.int32),   # 0-4
            'k3': np.random.randint(0, 4, n_main).astype(np.int32),   # 0-3
            'value': np.random.randn(n_main).astype(np.float32),
        })
        
        # Subframe with unique key combinations
        # Create all possible combinations for 3x5x4 = 60 entries
        k1_vals, k2_vals, k3_vals = [], [], []
        calib_vals = []
        for k1 in range(3):
            for k2 in range(5):
                for k3 in range(4):
                    k1_vals.append(k1)
                    k2_vals.append(k2)
                    k3_vals.append(k3)
                    calib_vals.append(k1 * 100 + k2 * 10 + k3)  # Deterministic
        
        df_sub = pd.DataFrame({
            'k1': np.array(k1_vals, dtype=np.int32),
            'k2': np.array(k2_vals, dtype=np.int32),
            'k3': np.array(k3_vals, dtype=np.int32),
            'calib': np.array(calib_vals, dtype=np.float32),
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns=['k1', 'k2', 'k3'])
        adf.add_alias('calibrated', 'value + S.calib')
        
        filepath = str(tmp_path / "test_3key.root")
        adf.export_tree(filepath, 'tree')
        
        return adf, filepath
    
    def test_return_composite_info_flag(self, tmp_path):
        """Test return_composite_info=True returns 3-tuple."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        
        # Create simple 1-key subframe (no composite key needed)
        df_main = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int32),
            'value': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        })
        df_sub = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int32),
            'calib': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns='key')
        
        filepath = str(tmp_path / "test_flag.root")
        adf.export_tree(filepath, 'tree')
        
        # Default should return 2-tuple
        result = setup_rdf_with_friends(adf, filepath)
        assert len(result) == 2
        
        # With flag should return 3-tuple
        result = setup_rdf_with_friends(adf, filepath, return_composite_info=True)
        assert len(result) == 3
        rdf, f, info = result
        assert isinstance(info, dict)
    
    def test_3key_without_precomputed_uses_runtime_generation(self, adf_with_3key_subframe):
        """Test that >2 keys without pre-computed composite key uses runtime generation."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, filepath = adf_with_3key_subframe
        
        # Should succeed with runtime composite key generation
        rdf, fh, info = setup_rdf_with_friends(adf, filepath, return_composite_info=True)
        
        # Verify runtime generation was used
        assert 'S' in info
        assert info['S']['method'] == 'runtime_dense'
        assert info['S']['n_keys'] == 3
        
        # Verify the RDF is usable
        count = rdf.Count().GetValue()
        assert count > 0
    
    def test_runtime_composite_key_join_correctness(self, adf_with_3key_subframe):
        """Test that runtime composite key join returns correct values."""
        from AliasDataFrameRDF import setup_rdf_with_friends
        adf, filepath = adf_with_3key_subframe
        
        rdf, fh, info = setup_rdf_with_friends(adf, filepath, return_composite_info=True)
        
        # Verify runtime generation was used
        assert info['S']['method'] == 'runtime_dense'
        
        # CRITICAL: Verify join correctness
        # calib = k1*100 + k2*10 + k3
        rdf = rdf.Define("expected_calib", "(float)(k1 * 100 + k2 * 10 + k3)")
        rdf = rdf.Define("diff", "S.calib - expected_calib")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        
        # All differences should be 0 (within floating point tolerance)
        assert all(abs(d) < 0.001 for d in diffs), \
            f"Runtime composite key join returned wrong values! Diffs: {diffs}"
    
    def test_precomputed_composite_key_used(self, tmp_path):
        """
        Test that pre-computed composite key branch is used when present.
        
        CRITICAL: Both main tree AND subframe tree must have the composite key
        branch for friend indexed lookup to work.
        """
        from AliasDataFrameRDF import setup_rdf_with_friends, get_composite_key_column_name
        import uproot
        
        # Create data with 3 keys
        # Key formula: k1 + k2*2 + k3*4 (with max k1=2, k2=2, k3=1)
        # Main tree rows:
        #   row 0: (0,0,0) -> key=0, expected calib=0
        #   row 1: (1,0,0) -> key=1, expected calib=100
        #   row 2: (0,1,0) -> key=2, expected calib=10
        #   row 3: (1,1,0) -> key=3, expected calib=110
        df_main = pd.DataFrame({
            'k1': np.array([0, 1, 0, 1], dtype=np.int32),
            'k2': np.array([0, 0, 1, 1], dtype=np.int32),
            'k3': np.array([0, 0, 0, 0], dtype=np.int32),
            '__adf_key_S__': np.array([0, 1, 2, 3], dtype=np.int64),  # MUST be in main too!
            'value': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        })
        
        # Subframe: calib = k1*100 + k2*10 + k3
        # Rows in SAME order as main (for simplicity)
        df_sub = pd.DataFrame({
            'k1': np.array([0, 1, 0, 1], dtype=np.int32),
            'k2': np.array([0, 0, 1, 1], dtype=np.int32),
            'k3': np.array([0, 0, 0, 0], dtype=np.int32),
            '__adf_key_S__': np.array([0, 1, 2, 3], dtype=np.int64),
            'calib': np.array([0.0, 100.0, 10.0, 110.0], dtype=np.float32),
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns=['k1', 'k2', 'k3'])
        
        filepath = str(tmp_path / "test_precomputed.root")
        
        # Export manually to include composite key branch in BOTH trees
        with uproot.recreate(filepath) as f:
            f['tree'] = {col: df_main[col].values for col in df_main.columns}
            f['tree__subframe__S'] = {col: df_sub[col].values for col in df_sub.columns}
        
        rdf, fh, info = setup_rdf_with_friends(adf, filepath, return_composite_info=True)
        
        # Should use pre-computed branch, not runtime generation
        assert 'S' in info
        assert info['S']['method'] == 'from_file'
        assert info['S']['branch'] == '__adf_key_S__'
        
        # CRITICAL: Verify join correctness
        rdf = rdf.Define("expected_calib", "k1 * 100 + k2 * 10 + k3")
        rdf = rdf.Define("diff", "S.calib - expected_calib")
        
        diffs = list(rdf.Take['float']("diff").GetValue())
        
        # All differences should be 0 (within floating point tolerance)
        assert all(abs(d) < 0.001 for d in diffs), \
            f"Pre-computed composite key join returned wrong values! Diffs: {diffs}"
    
    def test_precomputed_composite_key_shuffled(self, tmp_path):
        """
        Test pre-computed composite key works with SHUFFLED subframe.
        
        This is the critical test - subframe rows are in different order
        than main tree, requiring actual index lookup.
        """
        from AliasDataFrameRDF import setup_rdf_with_friends
        import uproot
        
        # Main tree: key 0,1,2,3 in order
        df_main = pd.DataFrame({
            'k1': np.array([0, 1, 0, 1], dtype=np.int32),
            'k2': np.array([0, 0, 1, 1], dtype=np.int32),
            'k3': np.array([0, 0, 0, 0], dtype=np.int32),
            '__adf_key_S__': np.array([0, 1, 2, 3], dtype=np.int64),
            'value': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        })
        
        # Subframe: SHUFFLED order (key 3,1,2,0)
        # calib = k1*100 + k2*10 + k3
        df_sub = pd.DataFrame({
            'k1': np.array([1, 1, 0, 0], dtype=np.int32),  # Shuffled!
            'k2': np.array([1, 0, 1, 0], dtype=np.int32),
            'k3': np.array([0, 0, 0, 0], dtype=np.int32),
            '__adf_key_S__': np.array([3, 1, 2, 0], dtype=np.int64),  # Matches shuffled keys
            'calib': np.array([110.0, 100.0, 10.0, 0.0], dtype=np.float32),  # Shuffled!
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns=['k1', 'k2', 'k3'])
        
        filepath = str(tmp_path / "test_shuffled.root")
        
        with uproot.recreate(filepath) as f:
            f['tree'] = {col: df_main[col].values for col in df_main.columns}
            f['tree__subframe__S'] = {col: df_sub[col].values for col in df_sub.columns}
        
        rdf, fh, info = setup_rdf_with_friends(adf, filepath, return_composite_info=True)
        
        assert info['S']['method'] == 'from_file'
        
        # CRITICAL: Verify join correctness with shuffled data
        rdf = rdf.Define("expected_calib", "(float)(k1 * 100 + k2 * 10 + k3)")
        rdf = rdf.Define("diff", "S.calib - expected_calib")
        
        expected = list(rdf.Take['float']("expected_calib").GetValue())
        actual = list(rdf.Take['float']("S.calib").GetValue())
        diffs = list(rdf.Take['float']("diff").GetValue())
        
        print(f"\nShuffled join test:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        print(f"  Diffs:    {diffs}")
        
        assert all(abs(d) < 0.001 for d in diffs), \
            f"Shuffled composite key join returned wrong values! Expected: {expected}, Actual: {actual}"


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestAddDefinesToRDF:
    """Test add_defines_to_rdf function."""
    
    @pytest.fixture
    def adf_with_aliases(self, tmp_path):
        """Create ADF with aliases and ROOT file."""
        df = pd.DataFrame({
            'x': np.random.randn(500).astype(np.float32),
            'y': np.random.randn(500).astype(np.float32),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('r2', 'x**2 + y**2')
        adf.add_alias('r', 'np.sqrt(r2)')
        
        filepath = str(tmp_path / "test_defines.root")
        adf.export_tree(filepath, 'tree')
        return adf, filepath
    
    def test_add_defines_creates_column(self, adf_with_aliases):
        """Test add_defines_to_rdf creates defined column."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_aliases
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        rdf = add_defines_to_rdf(rdf, adf, ['r2'])
        
        columns = [str(c) for c in rdf.GetColumnNames()]
        assert 'r2' in columns
    
    def test_add_defines_resolves_dependencies(self, adf_with_aliases):
        """Test add_defines_to_rdf resolves alias dependencies."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_aliases
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        # r depends on r2, both should be defined
        rdf = add_defines_to_rdf(rdf, adf, ['r'])
        
        columns = [str(c) for c in rdf.GetColumnNames()]
        assert 'r2' in columns  # Dependency
        assert 'r' in columns   # Target
    
    def test_add_defines_computes_values(self, adf_with_aliases):
        """Test computed values are correct."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_aliases
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        rdf = add_defines_to_rdf(rdf, adf, ['r2'])
        
        mean_r2 = rdf.Mean('r2').GetValue()
        # x**2 + y**2 with standard normal: E[X^2] + E[Y^2] ≈ 1 + 1 = 2
        assert 1.5 < mean_r2 < 2.5


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestChainSetup:
    """Test chain setup for multiple files."""
    
    @pytest.fixture
    def multi_file_setup(self, tmp_path):
        """Create multiple ROOT files."""
        filepaths = []
        for i in range(3):
            df = pd.DataFrame({
                'x': np.random.randn(100).astype(np.float32),
                'file_id': np.full(100, i, dtype=np.int32),
            })
            adf = AliasDataFrame(df)
            filepath = str(tmp_path / f"data_{i}.root")
            adf.export_tree(filepath, 'tree')
            filepaths.append(filepath)
        return adf, filepaths, tmp_path
    
    def test_chain_with_list(self, multi_file_setup):
        """Test setup_chain_with_friends with file list."""
        from AliasDataFrameRDF import setup_chain_with_friends
        adf, filepaths, _ = multi_file_setup
        
        rdf, chain, files = setup_chain_with_friends(adf, filepaths)
        count = rdf.Count().GetValue()
        assert count == 300  # 3 files × 100 rows
    
    def test_chain_with_glob(self, multi_file_setup):
        """Test setup_chain_with_friends with glob pattern."""
        from AliasDataFrameRDF import setup_chain_with_friends
        adf, _, tmp_path = multi_file_setup
        
        pattern = str(tmp_path / "data_*.root")
        rdf, chain, files = setup_chain_with_friends(adf, pattern)
        count = rdf.Count().GetValue()
        assert count == 300


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestJoinColumnsForSnapshot:
    """Test get_join_columns_for_snapshot function."""
    
    def test_returns_index_columns(self):
        """Test returns subframe index columns."""
        from AliasDataFrameRDF import get_join_columns_for_snapshot
        
        df = pd.DataFrame({'x': [1, 2, 3], 'row': [0, 1, 2]})
        sub_df = pd.DataFrame({'row': [0, 1, 2], 'val': [10, 20, 30]})
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sub_df), index_columns='row')
        
        join_cols = get_join_columns_for_snapshot(adf)
        assert 'row' in join_cols


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")  
class TestCacheToSnapshot:
    """Test cache_to_snapshot convenience function."""
    
    def test_cache_creates_file(self, tmp_path):
        """Test cache_to_snapshot creates output file."""
        from AliasDataFrameRDF import cache_to_snapshot
        import os
        
        df = pd.DataFrame({
            'x': np.random.randn(100).astype(np.float32),
            'y': np.random.randn(100).astype(np.float32),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('r2', 'x**2 + y**2')
        
        input_file = str(tmp_path / "input.root")
        output_file = str(tmp_path / "cache.root")
        adf.export_tree(input_file, 'tree')
        
        result = cache_to_snapshot(adf, input_file, output_file, ['r2'])
        
        assert os.path.exists(output_file)
        assert result['entries'] == 100
        assert 'r2' in result['columns']
        assert result['time_s'] > 0


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestAddDefinesCollision:
    """Test add_defines_to_rdf collision handling."""
    
    @pytest.fixture
    def adf_with_collision(self, tmp_path):
        """Create ADF with physical column that matches an alias name."""
        df = pd.DataFrame({
            'x': np.array([1, 2, 3], dtype=np.float32),
            'y': np.array([4, 5, 6], dtype=np.float32),
        })
        adf = AliasDataFrame(df)
        
        # Export FIRST (so 'x' is a physical branch in tree)
        filepath = str(tmp_path / "collision.root")
        adf.export_tree(filepath, 'tree')
        
        # THEN add alias 'x' (collides with physical branch 'x')
        adf.add_alias('x', 'y * 2')
        
        return adf, filepath
    
    def test_collision_error(self, adf_with_collision):
        """Test on_collision='error' raises ValueError."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_collision
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        with pytest.raises(ValueError, match="already exists"):
            add_defines_to_rdf(rdf, adf, ['x'], on_collision='error')
    
    def test_collision_skip(self, adf_with_collision):
        """Test on_collision='skip' skips silently."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_collision
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        rdf = add_defines_to_rdf(rdf, adf, ['x'], on_collision='skip')
        
        # Should use original branch value (1,2,3), not alias (8,10,12)
        mean = rdf.Mean('x').GetValue()
        assert abs(mean - 2.0) < 0.1  # Original data mean
    
    def test_collision_warn(self, adf_with_collision):
        """Test on_collision='warn' skips with warning."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_collision
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        with pytest.warns(UserWarning, match="already exist"):
            rdf = add_defines_to_rdf(rdf, adf, ['x'], on_collision='warn')
    
    def test_collision_redefine(self, adf_with_collision):
        """Test on_collision='redefine' overwrites with alias."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        adf, filepath = adf_with_collision
        
        rdf, f = setup_rdf_with_friends(adf, filepath)
        rdf = add_defines_to_rdf(rdf, adf, ['x'], on_collision='redefine')
        
        # Should use alias value (y*2 = 8,10,12), not original (1,2,3)
        mean = rdf.Mean('x').GetValue()
        assert abs(mean - 10.0) < 0.1  # Alias computation mean
    
    def test_collision_from_friend_tree(self, tmp_path):
        """Test collision detection works for friend tree columns."""
        from AliasDataFrameRDF import setup_rdf_with_friends, add_defines_to_rdf
        
        # Create main dataframe
        df_main = pd.DataFrame({
            'x': np.array([1, 2, 3], dtype=np.float32),
            'row': np.array([0, 1, 2], dtype=np.int32),
        })
        
        # Create subframe with column 'val'
        df_sub = pd.DataFrame({
            'row': np.array([0, 1, 2], dtype=np.int32),
            'val': np.array([10, 20, 30], dtype=np.float32),
        })
        
        adf = AliasDataFrame(df_main)
        adf.register_subframe('S', AliasDataFrame(df_sub), index_columns='row')
        
        # Materialize subframe column to main tree
        adf.add_alias('val', 'S.val')
        adf.materialize_alias('val')
        
        # Export - 'val' now exists as physical branch
        filepath = str(tmp_path / "friend_collision.root")
        adf.export_tree(filepath, 'tree')
        
        # Add alias that collides with materialized column
        adf.add_alias('val', 'x * 2')  # Different expression
        
        # Should handle collision gracefully with warn (default)
        rdf, f = setup_rdf_with_friends(adf, filepath)
        with pytest.warns(UserWarning, match="already exist"):
            rdf = add_defines_to_rdf(rdf, adf, ['val'], on_collision='warn')
        
        # Should use original branch value (10, 20, 30), not alias (2, 4, 6)
        mean = rdf.Mean('val').GetValue()
        assert abs(mean - 20.0) < 0.1


# =============================================================================
# RDataFrame Index Verification Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameIndexVerification:
    """
    Critical verification that RDataFrame uses BuildIndex for friend joins.
    
    These tests use SHUFFLED friend data to distinguish between:
    - Index-based join (correct): uses key values to match
    - Row-by-row join (incorrect): just matches by row number
    
    If these tests fail, our entire friend tree approach is broken.
    """
    
    def test_rdf_uses_single_key_index(self, tmp_path):
        """
        CRITICAL: Verify RDataFrame uses BuildIndex for single-key friend joins.
        
        Main:   row 0 → key=0, row 1 → key=1, ...
        Friend: row 0 → key=4, row 1 → key=2, row 2 → key=0, ... (SHUFFLED!)
        
        If index-based: main.key=0 → friend where key=0 → friend_val=0
        If row-based:   main.row=0 → friend.row=0 → friend_val=400 (WRONG!)
        """
        main_file = str(tmp_path / "main.root")
        friend_file = str(tmp_path / "friend.root")
        
        # Main tree: sequential keys
        main_data = {
            'key': np.array([0, 1, 2, 3, 4], dtype=np.int32),
            'main_val': np.array([0, 10, 20, 30, 40], dtype=np.float64),
        }
        
        # Friend tree: SHUFFLED key order!
        # key = [4, 2, 0, 3, 1], friend_val = key * 100
        friend_data = {
            'key': np.array([4, 2, 0, 3, 1], dtype=np.int32),
            'friend_val': np.array([400, 200, 0, 300, 100], dtype=np.float64),
        }
        
        # Export to ROOT files using uproot
        import uproot
        with uproot.recreate(main_file) as f:
            f["tree"] = main_data
        
        with uproot.recreate(friend_file) as f:
            f["tree"] = friend_data
        
        # Setup with BuildIndex
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main_tree = f_main.Get("tree")
        friend_tree = f_friend.Get("tree")
        
        # Build index on friend tree
        friend_tree.BuildIndex("key")
        
        # Add as friend
        main_tree.AddFriend(friend_tree, "F")
        
        # Create RDataFrame
        rdf = ROOT.RDataFrame(main_tree)
        
        # Define check: friend_val should equal key * 100 if index is used
        rdf = rdf.Define("expected", "key * 100.0")
        rdf = rdf.Define("actual", "F.friend_val")
        rdf = rdf.Define("diff", "F.friend_val - expected")
        
        # Get results
        diffs = list(rdf.Take['double']("diff").GetValue())
        
        # All differences should be 0 if index is used
        assert all(abs(d) < 0.001 for d in diffs), \
            f"RDataFrame NOT using single-key index! Diffs: {diffs}"
    
    def test_rdf_uses_two_key_index(self, tmp_path):
        """
        CRITICAL: Verify RDataFrame uses BuildIndex for two-key friend joins.
        
        Uses composite index with two columns.
        """
        main_file = str(tmp_path / "main.root")
        friend_file = str(tmp_path / "friend.root")
        
        # Main tree: k1, k2 combinations
        main_data = {
            'k1': np.array([0, 0, 1, 1, 2], dtype=np.int32),
            'k2': np.array([0, 1, 0, 1, 0], dtype=np.int32),
            'main_val': np.array([0, 1, 2, 3, 4], dtype=np.float64),
        }
        
        # Friend tree: SHUFFLED order!
        # Matching: (0,0)→100, (0,1)→101, (1,0)→102, (1,1)→103, (2,0)→104
        friend_data = {
            'k1': np.array([2, 1, 0, 1, 0], dtype=np.int32),       # Shuffled!
            'k2': np.array([0, 1, 0, 0, 1], dtype=np.int32),       # Shuffled!
            'friend_val': np.array([104, 103, 100, 102, 101], dtype=np.float64),
        }
        
        # Export to ROOT files
        import uproot
        with uproot.recreate(main_file) as f:
            f["tree"] = main_data
        
        with uproot.recreate(friend_file) as f:
            f["tree"] = friend_data
        
        # Setup with BuildIndex (two columns)
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main_tree = f_main.Get("tree")
        friend_tree = f_friend.Get("tree")
        
        # Build index with two keys
        friend_tree.BuildIndex("k1", "k2")
        
        main_tree.AddFriend(friend_tree, "F")
        
        rdf = ROOT.RDataFrame(main_tree)
        
        # Expected: friend_val = 100 + main_val
        rdf = rdf.Define("expected", "100.0 + main_val")
        rdf = rdf.Define("actual", "F.friend_val")
        rdf = rdf.Define("diff", "F.friend_val - expected")
        
        # Get results
        diffs = list(rdf.Take['double']("diff").GetValue())
        
        assert all(abs(d) < 0.001 for d in diffs), \
            f"RDataFrame NOT using 2-key index! Diffs: {diffs}"
    
    def test_ttree_draw_uses_index_for_comparison(self, tmp_path):
        """
        Sanity check: Verify TTree::Draw DOES use the index correctly.
        
        This confirms our test data is correct - if TTree::Draw works
        but RDataFrame doesn't, then RDataFrame has the limitation.
        """
        main_file = str(tmp_path / "main.root")
        friend_file = str(tmp_path / "friend.root")
        
        # Same data as single-key test
        main_data = {
            'key': np.array([0, 1, 2, 3, 4], dtype=np.int32),
            'main_val': np.array([0, 10, 20, 30, 40], dtype=np.float64),
        }
        
        friend_data = {
            'key': np.array([4, 2, 0, 3, 1], dtype=np.int32),
            'friend_val': np.array([400, 200, 0, 300, 100], dtype=np.float64),
        }
        
        import uproot
        with uproot.recreate(main_file) as f:
            f["tree"] = main_data
        
        with uproot.recreate(friend_file) as f:
            f["tree"] = friend_data
        
        # Setup
        f_main = ROOT.TFile.Open(main_file)
        f_friend = ROOT.TFile.Open(friend_file)
        
        main_tree = f_main.Get("tree")
        friend_tree = f_friend.Get("tree")
        
        friend_tree.BuildIndex("key")
        main_tree.AddFriend(friend_tree, "F")
        
        # Use TTree::Draw to get values
        n = main_tree.Draw("F.friend_val:key*100", "", "goff")
        
        v1 = main_tree.GetV1()  # F.friend_val
        v2 = main_tree.GetV2()  # key*100
        
        actual = [v1[i] for i in range(n)]
        expected = [v2[i] for i in range(n)]
        diffs = [actual[i] - expected[i] for i in range(n)]
        
        assert all(abs(d) < 0.001 for d in diffs), \
            "TTree::Draw should use index - test data may be wrong"


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
