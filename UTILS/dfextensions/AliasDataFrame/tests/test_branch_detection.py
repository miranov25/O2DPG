"""
Tests for Phase 7.2: Branch Auto-Detection from Expressions.

Tests cover:
- _parse_selection_columns(): AST-based selection parsing
- _parse_selection_columns_regex(): Fallback regex parsing  
- _resolve_to_base_branches(): Alias chain resolution
- get_required_branches(): Public API
- Integration with lazy loading
"""

import pytest
import pandas as pd
import numpy as np

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_adf():
    """Create sample ADF for testing."""
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100),
        'pt': np.abs(np.random.randn(100)),
        'eta': np.random.uniform(-2.5, 2.5, 100),
        'phi': np.random.uniform(-np.pi, np.pi, 100),
        'charge': np.random.choice([-1, 1], 100),
        'isOK': np.random.choice([True, False], 100),
        'nHits': np.random.randint(0, 100, 100),
        'signal': np.random.uniform(0, 100, 100),
        'trackLength': np.random.uniform(1, 10, 100),
        'expected': np.random.uniform(1, 5, 100),
        'p': np.random.uniform(0, 10, 100),
    })
    return AliasDataFrame(df)


@pytest.fixture
def sample_root_file(tmp_path):
    """Create a sample ROOT file for testing."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "test_data.root"
    
    n_entries = 1000
    data = {
        'x': np.random.randn(n_entries).astype(np.float32),
        'y': np.random.randn(n_entries).astype(np.float32),
        'pt': np.abs(np.random.randn(n_entries)).astype(np.float32),
        'eta': np.random.uniform(-2.5, 2.5, n_entries).astype(np.float32),
        'isOK': np.random.choice([True, False], n_entries),
        'charge': np.random.choice([-1, 1], n_entries).astype(np.int32),
    }
    
    with uproot.recreate(file_path) as f:
        f['tree'] = data
    
    return str(file_path)


# =============================================================================
# _parse_selection_columns Tests
# =============================================================================

class TestParseSelectionColumns:
    """Tests for _parse_selection_columns()."""
    
    def test_simple_comparison(self, sample_adf):
        """Parse simple comparison."""
        result = sample_adf._parse_selection_columns('pt > 0.5')
        assert result == {'pt'}
    
    def test_multiple_conditions_and_cstyle(self, sample_adf):
        """Parse C-style AND conditions."""
        result = sample_adf._parse_selection_columns('isOK && pt > 0.5')
        assert result == {'isOK', 'pt'}
    
    def test_multiple_conditions_or_cstyle(self, sample_adf):
        """Parse C-style OR conditions."""
        result = sample_adf._parse_selection_columns('x > 0 || y < 10')
        assert result == {'x', 'y'}
    
    def test_python_style_operators(self, sample_adf):
        """Parse Python-style operators."""
        result = sample_adf._parse_selection_columns('(x > 0) and (y < 10) or isOK')
        assert result == {'x', 'y', 'isOK'}
    
    def test_bitwise_operators(self, sample_adf):
        """Parse bitwise operators."""
        result = sample_adf._parse_selection_columns('(x > 0) & (y < 10) | isOK')
        assert result == {'x', 'y', 'isOK'}
    
    def test_negation_cstyle(self, sample_adf):
        """Parse C-style negation."""
        result = sample_adf._parse_selection_columns('!isOK && pt > 0')
        assert result == {'isOK', 'pt'}
    
    def test_negation_python(self, sample_adf):
        """Parse Python-style negation."""
        result = sample_adf._parse_selection_columns('not isOK and pt > 0')
        assert result == {'isOK', 'pt'}
    
    def test_not_equal_preserved(self, sample_adf):
        """!= should not be converted."""
        result = sample_adf._parse_selection_columns('x != 0 && y > 0')
        assert result == {'x', 'y'}
    
    def test_numpy_function_excluded(self, sample_adf):
        """Exclude numpy functions."""
        result = sample_adf._parse_selection_columns('np.abs(eta) < 2.5')
        assert result == {'eta'}
        assert 'np' not in result
        assert 'abs' not in result
    
    def test_numpy_sqrt_excluded(self, sample_adf):
        """Exclude numpy.sqrt."""
        result = sample_adf._parse_selection_columns('np.sqrt(x**2 + y**2) < 10')
        assert result == {'x', 'y'}
        assert 'np' not in result
        assert 'sqrt' not in result
    
    def test_math_functions_excluded(self, sample_adf):
        """Exclude standalone math functions."""
        result = sample_adf._parse_selection_columns('sqrt(x**2 + y**2) < 10')
        assert result == {'x', 'y'}
        assert 'sqrt' not in result
    
    def test_nested_function_calls(self, sample_adf):
        """Handle nested function calls."""
        result = sample_adf._parse_selection_columns('np.sqrt(np.abs(x)) < 5')
        assert result == {'x'}
    
    def test_complex_physics_expression(self, sample_adf):
        """Parse complex physics selection."""
        selection = '(np.abs(eta) < 2.5) && (pt > 0.5) && isOK && (nHits >= 10)'
        result = sample_adf._parse_selection_columns(selection)
        assert result == {'eta', 'pt', 'isOK', 'nHits'}
    
    def test_empty_selection(self, sample_adf):
        """Empty selection returns empty set."""
        assert sample_adf._parse_selection_columns('') == set()
        assert sample_adf._parse_selection_columns(None) == set()
    
    def test_none_selection(self, sample_adf):
        """None selection returns empty set."""
        assert sample_adf._parse_selection_columns(None) == set()
    
    def test_numeric_literals_excluded(self, sample_adf):
        """Numeric literals not included."""
        result = sample_adf._parse_selection_columns('pt > 0.5 && eta < 2')
        assert result == {'pt', 'eta'}
        # Numbers shouldn't appear
        assert '0' not in result
        assert '5' not in result
        assert '2' not in result
    
    def test_string_comparison(self, sample_adf):
        """Handle string in expression (falls back to regex)."""
        # This might not parse with AST, should fall back gracefully
        result = sample_adf._parse_selection_columns('status == "good"')
        assert 'status' in result
    
    def test_builtin_functions_excluded(self, sample_adf):
        """Exclude builtin functions."""
        result = sample_adf._parse_selection_columns('abs(x) > 0 and len(y) > 0')
        # Note: 'len' on a scalar might be weird, but we test filtering
        assert 'x' in result
        assert 'abs' not in result


# =============================================================================
# _parse_selection_columns_regex Tests
# =============================================================================

class TestParseSelectionColumnsRegex:
    """Tests for regex fallback."""
    
    def test_basic_extraction(self, sample_adf):
        """Regex extracts basic identifiers."""
        result = sample_adf._parse_selection_columns_regex('x > 0 && y < 10')
        assert 'x' in result
        assert 'y' in result
    
    def test_filters_keywords(self, sample_adf):
        """Regex filters Python keywords."""
        result = sample_adf._parse_selection_columns_regex('x > 0 and y < 10 or z')
        assert 'and' not in result
        assert 'or' not in result
        assert {'x', 'y', 'z'} <= result


# =============================================================================
# _resolve_to_base_branches Tests
# =============================================================================

class TestResolveToBaseBranches:
    """Tests for _resolve_to_base_branches()."""
    
    def test_no_aliases(self, sample_adf):
        """Columns without aliases return unchanged."""
        result = sample_adf._resolve_to_base_branches({'x', 'y', 'pt'})
        assert result == {'x', 'y', 'pt'}
    
    def test_simple_alias(self, sample_adf):
        """Resolve simple alias."""
        sample_adf.add_alias('r', 'np.sqrt(x**2 + y**2)')
        result = sample_adf._resolve_to_base_branches({'r', 'pt'})
        assert result == {'x', 'y', 'pt'}
    
    def test_alias_with_single_dep(self, sample_adf):
        """Alias with single dependency."""
        sample_adf.add_alias('pt_scaled', 'pt * 1000')
        result = sample_adf._resolve_to_base_branches({'pt_scaled'})
        assert result == {'pt'}
    
    def test_nested_aliases(self, sample_adf):
        """Resolve nested alias chain."""
        sample_adf.add_alias('dEdx', 'signal / trackLength')
        sample_adf.add_alias('normalized', 'dEdx / expected')
        result = sample_adf._resolve_to_base_branches({'normalized'})
        assert result == {'signal', 'trackLength', 'expected'}
    
    def test_deep_nested_aliases(self, sample_adf):
        """Resolve deeply nested alias chain."""
        sample_adf.add_alias('a', 'x + 1')
        sample_adf.add_alias('b', 'a + y')
        sample_adf.add_alias('c', 'b + z')
        result = sample_adf._resolve_to_base_branches({'c'})
        assert result == {'x', 'y', 'z'}
    
    def test_circular_alias_raises(self, sample_adf):
        """Circular alias dependency raises error.
        
        Note: add_alias() has built-in cycle detection via _topological_sort().
        The error is raised at add_alias time when the cycle is created.
        """
        sample_adf.add_alias('A', 'B + 1')
        # Second add_alias triggers cycle detection
        with pytest.raises(ValueError, match="[Cc]ycle"):
            sample_adf.add_alias('B', 'A + 1')
    
    def test_self_referencing_alias_raises(self, sample_adf):
        """Self-referencing alias raises error.
        
        Note: add_alias() catches this with a specific self-reference check.
        """
        with pytest.raises(ValueError, match="reference itself"):
            sample_adf.add_alias('x_norm', 'x_norm / 10')
    
    def test_three_way_circular(self, sample_adf):
        """Three-way circular dependency.
        
        Note: add_alias() catches this when the cycle is completed.
        """
        sample_adf.add_alias('A', 'B + 1')
        sample_adf.add_alias('B', 'C + 1')
        # Third add_alias completes the cycle
        with pytest.raises(ValueError, match="[Cc]ycle"):
            sample_adf.add_alias('C', 'A + 1')
    
    def test_diamond_dependency(self, sample_adf):
        """Diamond dependency (not circular) resolves correctly."""
        # A depends on B and C, both depend on D
        sample_adf.add_alias('D', 'x + 1')
        sample_adf.add_alias('B', 'D + 2')
        sample_adf.add_alias('C', 'D + 3')
        sample_adf.add_alias('A', 'B + C')
        result = sample_adf._resolve_to_base_branches({'A'})
        assert result == {'x'}
    
    def test_empty_input(self, sample_adf):
        """Empty input returns empty set."""
        result = sample_adf._resolve_to_base_branches(set())
        assert result == set()


# =============================================================================
# get_required_branches Tests
# =============================================================================

class TestGetRequiredBranches:
    """Tests for get_required_branches()."""
    
    def test_simple_expr(self, sample_adf):
        """Parse simple expression."""
        result = sample_adf.get_required_branches(expr='y:x')
        assert result == {'x', 'y'}
    
    def test_single_var_expr(self, sample_adf):
        """Parse single variable expression."""
        result = sample_adf.get_required_branches(expr='pt')
        assert result == {'pt'}
    
    def test_expr_with_selection(self, sample_adf):
        """Parse expression with selection."""
        result = sample_adf.get_required_branches(
            expr='y:x',
            selection='pt > 0.5 && isOK'
        )
        assert result == {'x', 'y', 'pt', 'isOK'}
    
    def test_with_group_by(self, sample_adf):
        """Include group_by column."""
        result = sample_adf.get_required_branches(
            expr='y:x',
            group_by='charge'
        )
        assert result == {'x', 'y', 'charge'}
    
    def test_with_color(self, sample_adf):
        """Include color column."""
        result = sample_adf.get_required_branches(
            expr='y:x',
            color='pt'
        )
        assert result == {'x', 'y', 'pt'}
    
    def test_with_group_by_and_color(self, sample_adf):
        """Include both group_by and color."""
        result = sample_adf.get_required_branches(
            expr='y:x',
            group_by='charge',
            color='pt'
        )
        assert result == {'x', 'y', 'charge', 'pt'}
    
    def test_with_alias_resolution(self, sample_adf):
        """Resolve alias in expression."""
        sample_adf.add_alias('dEdx', 'signal / trackLength')
        result = sample_adf.get_required_branches(expr='dEdx:p')
        assert result == {'signal', 'trackLength', 'p'}
    
    def test_alias_in_selection(self, sample_adf):
        """Resolve alias used in selection."""
        sample_adf.add_alias('r', 'np.sqrt(x**2 + y**2)')
        result = sample_adf.get_required_branches(
            expr='pt',
            selection='r < 10'
        )
        assert result == {'pt', 'x', 'y'}
    
    def test_full_physics_query(self, sample_adf):
        """Complete physics query."""
        sample_adf.add_alias('dEdx', 'signal / trackLength')
        result = sample_adf.get_required_branches(
            expr='dEdx:p',
            selection='isOK && pt > 0.5 && np.abs(eta) < 2.5',
            group_by='charge'
        )
        expected = {'signal', 'trackLength', 'p', 'isOK', 'pt', 'eta', 'charge'}
        assert result == expected
    
    def test_explicit_aliases_param(self, sample_adf):
        """Include explicit aliases."""
        sample_adf.add_alias('r', 'np.sqrt(x**2 + y**2)')
        result = sample_adf.get_required_branches(
            expr='pt',
            aliases=['r']
        )
        assert result == {'pt', 'x', 'y'}
    
    def test_no_inputs_returns_empty(self, sample_adf):
        """No inputs returns empty set."""
        result = sample_adf.get_required_branches()
        assert result == set()
    
    def test_validate_eager_mode(self, sample_adf):
        """validate=True filters in eager mode."""
        result = sample_adf.get_required_branches(
            expr='x:nonexistent_column',
            validate=True
        )
        assert 'x' in result
        assert 'nonexistent_column' not in result
    
    def test_validate_includes_aliases(self, sample_adf):
        """validate=True keeps columns that exist as aliases."""
        sample_adf.add_alias('my_alias', 'x + y')
        # my_alias exists as alias, so should resolve
        result = sample_adf.get_required_branches(
            expr='my_alias',
            validate=True
        )
        # Should contain base branches x, y
        assert result == {'x', 'y'}


# =============================================================================
# Integration with Lazy Loading Tests
# =============================================================================

class TestIntegrationWithLazy:
    """Integration tests with lazy loading."""
    
    def test_required_branches_before_load(self, sample_root_file):
        """Can detect required branches before loading."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # No branches loaded yet
        assert len(adf.loaded_branches) == 0
        
        # But we can detect what we need
        required = adf.get_required_branches(expr='y:x', selection='pt > 0')
        assert required == {'x', 'y', 'pt'}
    
    def test_validate_lazy_mode(self, sample_root_file):
        """validate=True filters against available branches in lazy mode."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        result = adf.get_required_branches(
            expr='x:nonexistent',
            validate=True
        )
        assert 'x' in result
        assert 'nonexistent' not in result
    
    def test_ensure_required_branches(self, sample_root_file):
        """Use get_required_branches() to load only what's needed."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        required = adf.get_required_branches(expr='y:x')
        adf.ensure_branches(list(required))
        
        assert adf.loaded_branches == {'x', 'y'}
    
    def test_workflow_detect_then_load(self, sample_root_file):
        """Full workflow: detect branches, validate, load."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Detect what we need
        required = adf.get_required_branches(
            expr='y:x',
            selection='pt > 0.5 && isOK',
            validate=True
        )
        
        # Should be subset of available
        assert required <= adf.available_branches
        
        # Load them
        adf.ensure_branches(list(required))
        
        # Verify loaded
        assert required <= adf.loaded_branches


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_spaces_in_expr(self, sample_adf):
        """Handle spaces in expression."""
        result = sample_adf.get_required_branches(expr='y : x')
        assert result == {'x', 'y'}
    
    def test_complex_selection_with_parentheses(self, sample_adf):
        """Handle complex nested parentheses."""
        selection = '((x > 0) && (y < 10)) || ((pt > 1) && (eta < 2))'
        result = sample_adf.get_required_branches(selection=selection)
        assert result == {'x', 'y', 'pt', 'eta'}
    
    def test_alias_not_found_passes_through(self, sample_adf):
        """Unknown identifiers pass through (treated as base columns)."""
        result = sample_adf.get_required_branches(expr='unknown:x')
        assert result == {'unknown', 'x'}
    
    def test_color_non_string_ignored(self, sample_adf):
        """Non-string color value ignored."""
        result = sample_adf.get_required_branches(
            expr='y:x',
            color=['red', 'blue']  # List, not string column name
        )
        assert result == {'x', 'y'}
