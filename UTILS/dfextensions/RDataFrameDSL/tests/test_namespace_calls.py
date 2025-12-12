"""
Phase 11.1: Namespace function call tests.

Tests for namespace functions like TMath.Pi(), TMath.Sin(x),
and ROOT.Math.VectorUtil.DeltaPhi(v1, v2).

Also tests preprocessing of C++ :: syntax to Python dot syntax.
"""

import pytest
from RDataFrameDSL import DSLCompiler


# =============================================================================
# Preprocessing Tests (No ROOT needed)
# =============================================================================

class TestPreprocessing:
    """Test :: to . preprocessing."""
    
    def test_simple_namespace(self):
        """TMath::Pi() -> TMath.Pi()"""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression("TMath::Pi()")
        assert result == "TMath.Pi()"
    
    def test_nested_namespace(self):
        """ROOT::Math::VectorUtil::DeltaPhi -> dots"""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression("ROOT::Math::VectorUtil::DeltaPhi(a, b)")
        assert result == "ROOT.Math.VectorUtil.DeltaPhi(a, b)"
    
    def test_preserves_string_literals(self):
        """Don't replace :: inside strings."""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression('x + "Error::Message"')
        assert result == 'x + "Error::Message"'
    
    def test_preserves_single_quote_strings(self):
        """Don't replace :: inside single-quoted strings."""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression("x + 'Error::Code'")
        assert result == "x + 'Error::Code'"
    
    def test_preserves_slice_syntax(self):
        """Don't replace :: inside square brackets (slice syntax)."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        # pt[::-1] should NOT become pt[.-1]
        result = dsl._preprocess_expression("pt[::-1]")
        assert result == "pt[::-1]"
        # Combined: namespace outside brackets, slice inside
        result2 = dsl._preprocess_expression("TMath::Sqrt(pt[::-1])")
        assert result2 == "TMath.Sqrt(pt[::-1])"
    
    def test_mixed_expression(self):
        """Mix of :: and regular code."""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression("x * TMath::Pi() + TMath::E()")
        assert result == "x * TMath.Pi() + TMath.E()"
    
    def test_no_change_without_namespace(self):
        """Regular expressions pass through unchanged."""
        dsl = DSLCompiler({"x": "double"})
        result = dsl._preprocess_expression("x + y * 2")
        assert result == "x + y * 2"


# =============================================================================
# Code Generation Tests (No ROOT compilation needed)
# =============================================================================

class TestNamespaceCodeGeneration:
    """Test C++ code generation for namespace calls."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"x": "double", "y": "double"})
    
    def test_tmath_pi_generates_cpp(self, dsl):
        """TMath.Pi() generates TMath::Pi()"""
        dsl.define("scaled", "x * TMath.Pi()")
        code = dsl.preview()
        assert "TMath::Pi()" in code
    
    def test_tmath_sin_generates_cpp(self, dsl):
        """TMath.Sin(x) generates TMath::Sin(x)"""
        dsl.define("sin_x", "TMath.Sin(x)")
        code = dsl.preview()
        assert "TMath::Sin(" in code
    
    def test_nested_namespace_generates_cpp(self):
        """ROOT.Math.VectorUtil.DeltaPhi generates correct C++"""
        dsl = DSLCompiler({
            "v1": "ROOT::Math::XYZTVector",
            "v2": "ROOT::Math::XYZTVector"
        })
        dsl.define("dphi", "ROOT.Math.VectorUtil.DeltaPhi(v1, v2)")
        code = dsl.preview()
        assert "ROOT::Math::VectorUtil::DeltaPhi(" in code
    
    def test_cpp_syntax_also_works(self, dsl):
        """TMath::Pi() (C++ syntax) also works after preprocessing."""
        dsl.define("scaled", "x * TMath::Pi()")
        code = dsl.preview()
        assert "TMath::Pi()" in code
    
    def test_header_included(self, dsl):
        """TMath functions include TMath.h header."""
        dsl.define("scaled", "x * TMath.Pi()")
        func = dsl.get_function("scaled")
        assert "<TMath.h>" in func.headers


# =============================================================================
# Return Type Tests
# =============================================================================

class TestNamespaceReturnTypes:
    """Test return type inference for namespace functions."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"x": "double"})
    
    def test_tmath_pi_returns_double(self, dsl):
        """TMath.Pi() returns double."""
        dsl.define("pi", "TMath.Pi()")
        func = dsl.get_function("pi")
        assert "double" in func.return_type.lower()
    
    def test_tmath_sin_returns_double(self, dsl):
        """TMath.Sin(x) returns double."""
        dsl.define("sin_x", "TMath.Sin(x)")
        func = dsl.get_function("sin_x")
        assert "double" in func.return_type.lower()
    
    def test_arithmetic_with_namespace(self, dsl):
        """x * TMath.Pi() returns double."""
        dsl.define("scaled", "x * TMath.Pi()")
        func = dsl.get_function("scaled")
        assert "double" in func.return_type.lower()


# =============================================================================
# RVec Broadcasting Tests (Phase 11.1 fix)
# =============================================================================

class TestNamespaceRVecBroadcasting:
    """Test RVec broadcasting with namespace functions."""
    
    def test_tmath_sqrt_broadcasts(self):
        """TMath.Sqrt(pt) broadcasts over RVec."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        func = dsl.get_function("sqrt_pt")
        assert "RVec" in func.return_type
        code = dsl.preview()
        assert "TMath::Sqrt" in code
    
    def test_tmath_sin_broadcasts(self):
        """TMath.Sin(x) broadcasts over RVec."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("sin_x", "TMath.Sin(x)")
        func = dsl.get_function("sin_x")
        assert "RVec" in func.return_type
    
    def test_nested_tmath_calls(self):
        """TMath.Sin(TMath.Pi()) works with nested calls."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x + TMath.Sin(TMath.Pi())")
        code = dsl.preview()
        assert "TMath::Sin(TMath::Pi())" in code


# =============================================================================
# Disambiguation Tests
# =============================================================================

class TestNamespaceDisambiguation:
    """Test that namespace calls are distinguished from object methods."""
    
    def test_namespace_not_confused_with_variable(self):
        """TMath.Pi() works even if 'TMath' could be a variable name."""
        # TMath is NOT in schema, so it's treated as namespace
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "TMath.Pi()")
        code = dsl.preview()
        assert "TMath::Pi()" in code
    
    @pytest.fixture(autouse=True)
    def require_root_for_object_tests(self, request):
        """Some disambiguation tests need ROOT for object method resolution."""
        # These tests verify namespace detection logic, not full compilation
        pass
    
    def test_object_method_not_confused_with_namespace(self):
        """track.Pt() is method call, not namespace - test namespace detection."""
        # Test that namespace detection returns None for schema variables
        from RDataFrameDSL.ir_builder import IRBuilder
        from RDataFrameDSL.type_inferrer import TypeInferrer
        import ast
        
        schema = {'columns': {'track': {'dtype': 'TLorentzVector', 'rank': 0, 'cpp_type': 'TLorentzVector'}}}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        tree = ast.parse('track.Pt()', mode='eval')
        namespace_info = builder._extract_namespace_chain(tree.body.func)
        
        # Should return None because 'track' is in schema
        assert namespace_info is None, f"Expected None but got {namespace_info}"
    
    def test_schema_variable_takes_priority(self):
        """Variable in schema takes priority over namespace - test detection."""
        from RDataFrameDSL.ir_builder import IRBuilder
        from RDataFrameDSL.type_inferrer import TypeInferrer
        import ast
        
        # If someone has a variable named TMath (unusual but possible)
        schema = {'columns': {'TMath': {'dtype': 'TLorentzVector', 'rank': 0, 'cpp_type': 'TLorentzVector'}}}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        tree = ast.parse('TMath.M()', mode='eval')
        namespace_info = builder._extract_namespace_chain(tree.body.func)
        
        # Should return None because 'TMath' is in schema (takes priority)
        assert namespace_info is None, f"Expected None but got {namespace_info}"
    
    def test_namespace_detected_when_not_in_schema(self):
        """TMath.Pi() IS a namespace when TMath not in schema."""
        from RDataFrameDSL.ir_builder import IRBuilder
        from RDataFrameDSL.type_inferrer import TypeInferrer
        import ast
        
        schema = {'columns': {'x': {'dtype': 'double', 'rank': 0}}}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        tree = ast.parse('TMath.Pi()', mode='eval')
        namespace_info = builder._extract_namespace_chain(tree.body.func)
        
        # Should detect namespace
        assert namespace_info == ('TMath', 'Pi'), f"Expected ('TMath', 'Pi') but got {namespace_info}"


# =============================================================================
# Compilation Tests (Require ROOT)
# =============================================================================


class TestNamespaceCompilation:
    """ROOT compilation tests for namespace calls."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        pytest.importorskip("ROOT")
    
    def test_tmath_pi_compiles(self):
        """TMath.Pi() compiles successfully."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("scaled", "x * TMath.Pi()")
        dsl.compile_all()  # Should not raise
    
    def test_tmath_sin_compiles(self):
        """TMath.Sin(x) compiles successfully."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sin_x", "TMath.Sin(x)")
        dsl.compile_all()
    
    def test_tmath_sqrt_compiles(self):
        """TMath.Sqrt(x) compiles successfully."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sqrt_x", "TMath.Sqrt(x)")
        dsl.compile_all()
    
    def test_multiple_tmath_compiles(self):
        """Multiple TMath calls compile."""
        dsl = DSLCompiler({"x": "double", "y": "double"})
        dsl.define("result", "TMath.Sqrt(x*x + y*y) * TMath.Pi()")
        dsl.compile_all()
    
    def test_cpp_syntax_compiles(self):
        """TMath::Pi() (C++ syntax) compiles."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("scaled", "x * TMath::Pi()")
        dsl.compile_all()


# =============================================================================
# Execution Tests (Require ROOT)
# =============================================================================

class TestNamespaceExecution:
    """Execution tests with actual values."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        ROOT = pytest.importorskip("ROOT")
        self.ROOT = ROOT
    
    def test_tmath_pi_value(self):
        """TMath.Pi() returns correct value."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("pi", "TMath.Pi()")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("x", "1.0")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("pi").GetValue()
        assert abs(result - 3.14159265) < 0.0001
    
    def test_tmath_sin_value(self):
        """TMath.Sin(x) returns correct value."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sin_x", "TMath.Sin(x)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("x", "0.0")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("sin_x").GetValue()
        assert abs(result - 0.0) < 0.0001
    
    def test_tmath_combined_value(self):
        """Combined TMath expression returns correct value."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "TMath.Sin(x) + TMath.Cos(x)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("x", "0.0")  # sin(0) + cos(0) = 0 + 1 = 1
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("result").GetValue()
        assert abs(result - 1.0) < 0.0001
    
    def test_multiplication_with_pi(self):
        """x * TMath.Pi() multiplies correctly."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("scaled", "x * TMath.Pi()")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("x", "2.0")  # 2 * pi ≈ 6.283
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("scaled").GetValue()
        assert abs(result - 6.283185) < 0.001
    
    def test_tmath_e_value(self):
        """TMath.E() returns Euler's number."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("e", "TMath.E()")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("x", "1.0")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("e").GetValue()
        assert abs(result - 2.71828) < 0.0001


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestNamespaceErrors:
    """Test error handling for namespace calls."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        pytest.importorskip("ROOT")
    
    def test_unknown_namespace_function_error(self):
        """Unknown function in known namespace gives error at compile time."""
        dsl = DSLCompiler({"x": "double"})
        # TMath.NonExistent should fail at compile time
        dsl.define("bad", "TMath.NonExistentFunction(x)")
        with pytest.raises(Exception):
            dsl.compile_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
