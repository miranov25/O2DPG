# =============================================================================
# tests/test_api_define.py
# =============================================================================
# API smoke tests for define() and apply() methods.
#
# Purpose: Verify DSLCompiler API works correctly. These tests answer "does the
#          method work?" not "are the results mathematically correct?"
#
# Phase: 13.6.F
# Date: 2026-01-23
# =============================================================================

import pytest
import numpy as np

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError, IRErrorKind


# =============================================================================
# TestDefine - Basic API functionality
# =============================================================================

class TestDefine:
    """API smoke tests for define() method."""
    
    @pytest.fixture
    def basic_schema(self):
        """Simple schema for define() tests."""
        return {
            'event_id': 'long',
            'x': 'double',
            'y': 'double',
            'pt': 'RVec<double>',
            'eta': 'RVec<double>',
        }
    
    @pytest.mark.feature("api_define")
    def test_define_arithmetic(self, basic_schema):
        """
        define() works with basic arithmetic expressions.
        
        API Contract:
        - Returns self for chaining
        - Expression is stored
        """
        dsl = DSLCompiler(basic_schema)
        
        result = dsl.define("r", "sqrt(x**2 + y**2)")
        
        # API contract: returns self for chaining
        assert result is dsl
        
        # API contract: definition stored
        definitions = dsl.list_definitions()
        assert len(definitions) == 1
        assert definitions[0][0] == "r"
        assert definitions[0][1] == "sqrt(x**2 + y**2)"
    
    @pytest.mark.feature("api_define")
    def test_define_method_call(self, basic_schema):
        """
        define() works with method calls on RVec objects.
        
        Note: This is a schema-only test (no ROOT execution).
        """
        # Add TLorentzVector to schema
        schema = basic_schema.copy()
        schema['tracks'] = 'RVec<TLorentzVector>'
        
        dsl = DSLCompiler(schema)
        
        # Method call on RVec<Object> - should not raise
        dsl.define("track_pt", "tracks.Pt()")
        
        definitions = dsl.list_definitions()
        assert definitions[0][0] == "track_pt"
    
    @pytest.mark.feature("api_define")
    def test_define_slicing(self, basic_schema):
        """
        define() works with slice expressions.
        """
        dsl = DSLCompiler(basic_schema)
        
        dsl.define("first_two", "pt[:2]")
        
        definitions = dsl.list_definitions()
        assert definitions[0][0] == "first_two"
    
    @pytest.mark.feature("api_define")
    def test_define_chaining(self, basic_schema):
        """
        define() supports method chaining.
        
        API Contract:
        - Multiple define() calls can be chained
        - All definitions stored in order
        """
        dsl = DSLCompiler(basic_schema)
        
        dsl.define("r", "sqrt(x**2 + y**2)") \
           .define("phi", "atan2(y, x)") \
           .define("pt_sum", "Sum(pt)")
        
        definitions = dsl.list_definitions()
        assert len(definitions) == 3
        assert definitions[0][0] == "r"
        assert definitions[1][0] == "phi"
        assert definitions[2][0] == "pt_sum"
    
    @pytest.mark.feature("api_define")
    def test_define_alias_reference(self, basic_schema):
        """
        define() can reference previously defined aliases.
        """
        dsl = DSLCompiler(basic_schema)
        
        dsl.define("r", "sqrt(x**2 + y**2)")
        dsl.define("r_squared", "r**2")  # References 'r' alias
        
        definitions = dsl.list_definitions()
        assert len(definitions) == 2
        assert definitions[1][1] == "r**2"


# =============================================================================
# TestDefineErrors - Error detection
# =============================================================================

class TestDefineErrors:
    """Error handling tests for define() - validates Layer 1 error detection."""
    
    @pytest.fixture
    def basic_schema(self):
        """Simple schema for error tests."""
        return {
            'event_id': 'long',
            'x': 'double',
            'pt': 'RVec<double>',
        }
    
    @pytest.mark.feature("error_missing_column")
    def test_define_missing_column(self, basic_schema):
        """
        define() raises IRError for column not in schema.
        
        Layer 1 Requirement: Reject before ROOT compilation.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "missing_column + 1")
        
        # Should be TYPE_ERROR (unknown variable)
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
    
    @pytest.mark.feature("error_suggestions")
    def test_define_suggestions(self, basic_schema):
        """
        define() error includes suggestions for similar column names.
        
        Layer 1 Requirement: Helpful error messages.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "p")  # Similar to 'pt'
        
        # Should have suggestions
        assert len(exc_info.value.suggestions) > 0
    
    @pytest.mark.feature("error_name_conflict")
    def test_define_name_conflict(self, basic_schema):
        """
        define() raises error when name conflicts with schema column.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("x", "pt[0]")  # 'x' already in schema
        
        assert exc_info.value.kind == IRErrorKind.VALIDATION_ERROR
    
    @pytest.mark.feature("error_duplicate_definition")
    def test_define_duplicate(self, basic_schema):
        """
        define() raises error for duplicate definition name.
        """
        dsl = DSLCompiler(basic_schema)
        
        dsl.define("calc", "x + 1")
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("calc", "x + 2")  # Same name again
        
        assert exc_info.value.kind == IRErrorKind.VALIDATION_ERROR
    
    @pytest.mark.feature("error_syntax")
    def test_define_syntax_error(self, basic_schema):
        """
        define() raises IRError for invalid Python syntax.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "x +")  # Incomplete expression
        
        assert exc_info.value.kind == IRErrorKind.PARSE_ERROR
    
    @pytest.mark.feature("error_unknown_function")
    def test_define_unknown_function(self, basic_schema):
        """
        define() raises error for unknown function.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "unknown_func(x)")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
    
    @pytest.mark.feature("error_slice_step_zero")
    def test_define_slice_step_zero(self, basic_schema):
        """
        define() raises error for slice with step=0.
        """
        dsl = DSLCompiler(basic_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "pt[::0]")  # Step cannot be 0
        
        assert exc_info.value.kind == IRErrorKind.VALIDATION_ERROR


# =============================================================================
# TestApply - RDataFrame integration
# =============================================================================

class TestApply:
    """API smoke tests for apply() method."""
    
    @pytest.mark.feature("api_apply")
    @pytest.mark.root_serial
    def test_apply_single_definition(self, nd_2d_rdf, nd_2d_schema):
        """
        apply() works with single definition.
        
        API Contract:
        - Returns modified RDataFrame
        - New column accessible via AsNumpy
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("pt_squared", "track_pt * track_pt")
        
        rdf = dsl.apply(nd_2d_rdf)
        
        # API contract: new column available
        columns = rdf.GetColumnNames()
        assert "pt_squared" in [str(c) for c in columns]
    
    @pytest.mark.feature("api_apply")
    @pytest.mark.root_serial
    def test_apply_multiple_definitions(self, nd_2d_rdf, nd_2d_schema):
        """
        apply() works with multiple definitions.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("pt_squared", "track_pt * track_pt")
        dsl.define("eta_abs", "abs(track_eta)")
        
        rdf = dsl.apply(nd_2d_rdf)
        
        columns = [str(c) for c in rdf.GetColumnNames()]
        assert "pt_squared" in columns
        assert "eta_abs" in columns
    
    @pytest.mark.feature("api_apply")
    @pytest.mark.root_serial
    def test_apply_chained_aliases(self, nd_2d_rdf, nd_2d_schema):
        """
        apply() works with chained alias references.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("pt_squared", "track_pt * track_pt")
        dsl.define("pt_quad", "pt_squared * pt_squared")  # Uses pt_squared
        
        rdf = dsl.apply(nd_2d_rdf)
        
        columns = [str(c) for c in rdf.GetColumnNames()]
        assert "pt_squared" in columns
        assert "pt_quad" in columns


# =============================================================================
# TestPreview - Code generation preview
# =============================================================================

class TestPreview:
    """API tests for preview() method."""
    
    @pytest.mark.feature("api_define")
    def test_preview_returns_cpp(self):
        """
        preview() returns generated C++ code.
        """
        schema = {'x': 'double', 'y': 'double'}
        dsl = DSLCompiler(schema)
        dsl.define("r", "sqrt(x**2 + y**2)")
        
        code = dsl.preview()
        
        # API contract: returns string with C++ code
        assert isinstance(code, str)
        assert "sqrt" in code
        assert "x" in code
        assert "y" in code
    
    @pytest.mark.feature("api_define")
    def test_preview_empty(self):
        """
        preview() works with no definitions.
        """
        schema = {'x': 'double'}
        dsl = DSLCompiler(schema)
        
        code = dsl.preview()
        
        assert isinstance(code, str)
