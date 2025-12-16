"""
Phase 12.6.DSL: to_aliasdf() export tests.

Tests the DSL definition export to AliasDataFrame schema format.
"""

import pytest
import warnings


class TestToAliasdf:
    """Test to_aliasdf() export functionality."""
    
    @pytest.fixture
    def dsl(self):
        """Create DSL compiler for testing."""
        from RDataFrameDSL import DSLCompiler
        # Use simple schema format: {"col": "ctype"}
        schema = {
            "trackPt": "double", 
            "trackEta": "double", 
            "nHits": "int",
            "isGood": "bool",
        }
        return DSLCompiler(schema)
    
    def test_basic_export_structure(self, dsl):
        """Basic export produces valid schema structure."""
        dsl.define("pt_gev", "trackPt / 1000")
        schema = dsl.to_aliasdf()
        
        assert 'columns' in schema
        assert '__meta__' in schema
        assert schema['__meta__']['source'] == 'RDataFrameDSL'
        assert schema['__meta__']['export_version'] == '1.0'
        assert 'exported_at' in schema['__meta__']
    
    def test_simple_definition_exported(self, dsl):
        """Simple definition appears in columns with expr field."""
        dsl.define("pt_gev", "trackPt / 1000")
        schema = dsl.to_aliasdf()
        
        assert 'pt_gev' in schema['columns']
        assert schema['columns']['pt_gev']['expr'] == 'trackPt / 1000'
    
    def test_and_operator_with_parentheses(self, dsl):
        """C++ && converted to Python & with parentheses."""
        # Test the conversion helper directly
        result = dsl._cpp_to_python_expr("trackPt > 0.5 && nHits > 5")
        
        # Must have & operator
        assert ' & ' in result
        assert '&&' not in result
        
        # Must have parentheses for precedence
        assert '(trackPt > 0.5)' in result
        assert '(nHits > 5)' in result
    
    def test_or_operator_with_parentheses(self, dsl):
        """C++ || converted to Python | with parentheses."""
        # Test the conversion helper directly
        result = dsl._cpp_to_python_expr("trackPt > 1.0 || trackEta < 1.0")
        
        assert ' | ' in result
        assert '||' not in result
        assert '(' in result and ')' in result
    
    def test_unary_not_conversion(self, dsl):
        """C++ ! converted to Python ~."""
        # Test the conversion helper directly
        result = dsl._cpp_to_python_expr("!isGood")
        
        assert '~' in result
        assert '!' not in result
    
    def test_not_equal_preserved(self, dsl):
        """!= is preserved (not converted)."""
        dsl.define("nonzero", "nHits != 0")
        schema = dsl.to_aliasdf()
        
        assert '!=' in schema['columns']['nonzero']['expr']
        assert '~=' not in schema['columns']['nonzero']['expr']
    
    def test_include_filter(self, dsl):
        """Include filter selects specific definitions."""
        dsl.define("a", "trackPt + 1")
        dsl.define("b", "trackPt + 2")
        dsl.define("c", "trackPt + 3")
        schema = dsl.to_aliasdf(include=['a', 'c'])
        
        assert 'a' in schema['columns']
        assert 'c' in schema['columns']
        assert 'b' not in schema['columns']
    
    def test_exclude_filter(self, dsl):
        """Exclude filter removes specific definitions."""
        dsl.define("a", "trackPt + 1")
        dsl.define("b", "trackPt + 2")
        dsl.define("c", "trackPt + 3")
        schema = dsl.to_aliasdf(exclude=['b'])
        
        assert 'a' in schema['columns']
        assert 'c' in schema['columns']
        assert 'b' not in schema['columns']
    
    def test_dtype_map(self, dsl):
        """dtype_map sets column dtypes."""
        dsl.define("pt_gev", "trackPt / 1000")
        dsl.define("is_good", "nHits > 5")
        schema = dsl.to_aliasdf(dtype_map={
            'pt_gev': 'float32',
            'is_good': 'bool'
        })
        
        assert schema['columns']['pt_gev']['dtype'] == 'float32'
        assert schema['columns']['is_good']['dtype'] == 'bool'
    
    def test_warns_on_tmath(self, dsl):
        """Warning issued for TMath functions."""
        # Test the conversion helper directly for warning
        with pytest.warns(UserWarning, match='TMath'):
            dsl._cpp_to_python_expr("TMath::Sin(phi)")
    
    def test_warns_on_root_namespace(self, dsl):
        """Warning issued for ROOT namespace."""
        # Test the conversion helper directly for warning
        with pytest.warns(UserWarning, match='ROOT'):
            dsl._cpp_to_python_expr("ROOT::VecOps::Sum(arr)")
    
    def test_empty_definitions(self, dsl):
        """Empty definitions produce empty columns."""
        schema = dsl.to_aliasdf()
        
        assert schema['columns'] == {}


class TestBooleanPrecedence:
    """Test boolean operator precedence handling."""
    
    @pytest.fixture
    def dsl(self):
        from RDataFrameDSL import DSLCompiler
        return DSLCompiler({})
    
    def test_single_and(self, dsl):
        """Single && produces correct output."""
        # Test the conversion helper directly (not through define)
        result = dsl._cpp_to_python_expr('a > 0 && b < 1')
        assert result == '(a > 0) & (b < 1)'
    
    def test_single_or(self, dsl):
        """Single || produces correct output."""
        result = dsl._cpp_to_python_expr('a > 0 || b < 1')
        assert result == '(a > 0) | (b < 1)'
    
    def test_multiple_and(self, dsl):
        """Multiple && chained correctly."""
        result = dsl._cpp_to_python_expr('a > 0 && b > 1 && c > 2')
        assert '&&' not in result
        assert result.count(' & ') == 2
    
    def test_mixed_and_or_precedence(self, dsl):
        """Mixed && and || respects C++ precedence (&& binds tighter)."""
        # C++: a && b || c && d  means  (a && b) || (c && d)
        result = dsl._cpp_to_python_expr('a > 0 && b > 0 || c > 0 && d > 0')
        
        assert '&&' not in result
        assert '||' not in result
        assert ' | ' in result  # OR operator present
        assert ' & ' in result  # AND operator present
        # The OR should separate two AND groups
        assert result.count(' | ') == 1
        assert result.count(' & ') == 2


class TestGetDefinitions:
    """Test get_definitions() helper."""
    
    @pytest.fixture
    def dsl(self):
        from RDataFrameDSL import DSLCompiler
        return DSLCompiler({"x": "double"})
    
    def test_empty_definitions(self, dsl):
        """Empty DSL returns empty dict."""
        assert dsl.get_definitions() == {}
    
    def test_returns_dict(self, dsl):
        """Definitions returned as dict."""
        dsl.define("a", "x + 1")
        dsl.define("b", "x + 2")
        
        defs = dsl.get_definitions()
        
        assert isinstance(defs, dict)
        assert defs == {'a': 'x + 1', 'b': 'x + 2'}


class TestToAliasdfIntegration:
    """Integration tests with AliasDataFrame.
    
    These tests verify the exported schema works with AliasDataFrame.
    """
    
    def test_schema_applies_to_aliasdf(self):
        """Exported schema works with AliasDataFrame.apply_schema()."""
        pd = pytest.importorskip("pandas")
        
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            pytest.skip("AliasDataFrame not available")
        
        from RDataFrameDSL import DSLCompiler
        
        # Use simple schema format
        schema = {"trackPt": "double"}
        dsl = DSLCompiler(schema)
        dsl.define("pt_gev", "trackPt / 1000")
        
        adf_schema = dsl.to_aliasdf()
        
        # Verify schema structure is correct for AliasDataFrame
        assert 'columns' in adf_schema
        assert 'pt_gev' in adf_schema['columns']
        assert adf_schema['columns']['pt_gev']['expr'] == 'trackPt / 1000'
        
        # Test that schema can be applied and alias materialized
        df = pd.DataFrame({'trackPt': [1000.0, 2000.0, 3000.0]})
        adf = AliasDataFrame(df)
        adf.apply_schema(adf_schema)
        
        # materialize_aliases with pattern matching (regex)
        adf.materialize_aliases('pt_gev')
        
        assert 'pt_gev' in adf.df.columns
        assert list(adf.df['pt_gev']) == [1.0, 2.0, 3.0]
    
    def test_boolean_expression_evaluates_correctly(self):
        """Boolean expression with & evaluates correctly in pandas."""
        pd = pytest.importorskip("pandas")
        
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            pytest.skip("AliasDataFrame not available")
        
        from RDataFrameDSL import DSLCompiler
        
        # Use simple schema format and Python-compatible syntax
        schema = {"trackPt": "double", "nHits": "int"}
        dsl = DSLCompiler(schema)
        # Use Python bitwise operators (DSL supports these)
        dsl.define("good_track", "(trackPt > 0.5) & (nHits > 5)")
        
        adf_schema = dsl.to_aliasdf()
        
        # Verify the expression is exported correctly
        assert 'good_track' in adf_schema['columns']
        expr = adf_schema['columns']['good_track']['expr']
        assert '&' in expr
        
        df = pd.DataFrame({
            'trackPt': [0.3, 0.6, 1.0],
            'nHits': [3, 6, 10]
        })
        adf = AliasDataFrame(df)
        adf.apply_schema(adf_schema)
        
        # Materialize and verify
        adf.materialize_aliases('good_track')
        
        # Row 0: pt=0.3 (F) & nHits=3 (F) = False
        # Row 1: pt=0.6 (T) & nHits=6 (T) = True
        # Row 2: pt=1.0 (T) & nHits=10 (T) = True
        expected = [False, True, True]
        assert list(adf.df['good_track']) == expected
