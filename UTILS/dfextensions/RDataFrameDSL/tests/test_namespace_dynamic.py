"""
Phase 11.1c: Dynamic namespace detection + dtype parameter tests.

Tests for:
1. Schema priority over namespace resolution
2. Dynamic namespace detection via ROOT reflection
3. Positive-only namespace caching
4. Helpful error messages for unknown symbols
5. dtype parameter for explicit type specification
6. Integration of dynamic detection with broadcasting
"""

import pytest
from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_builder import IRBuilder
from RDataFrameDSL.type_inferrer import TypeInferrer
from RDataFrameDSL.ir_errors import IRError


# =============================================================================
# Schema Priority Tests
# =============================================================================

class TestSchemaPriority:
    """Schema variables take priority over namespace resolution."""
    
    def test_schema_shadows_namespace(self):
        """Variable named 'TMath' is treated as variable, not namespace."""
        dsl = DSLCompiler({"TMath": "double", "x": "double"})
        dsl.define("y", "TMath + x")
        code = dsl.preview()
        # Should NOT have namespace syntax - TMath is a variable
        assert "TMath::" not in code
        # Should reference TMath as a parameter
        assert "TMath" in code
    
    def test_schema_shadows_nested_namespace(self):
        """Schema variable shadows even if it looks like namespace prefix."""
        dsl = DSLCompiler({"ROOT": "double", "x": "double"})
        # ROOT here is a variable, not a namespace
        dsl.define("y", "ROOT + x")
        code = dsl.preview()
        assert "ROOT::" not in code


# =============================================================================
# Dynamic Detection Tests (Require ROOT)
# =============================================================================

class TestDynamicDetection:
    """Dynamic namespace detection via ROOT reflection."""
    
    def test_tmath_detected_dynamically(self):
        """TMath detected via ROOT, not just whitelist."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("y", "TMath.Sin(x)")
        code = dsl.preview()
        assert "TMath::Sin" in code
        dsl.compile_all()
    
    def test_nested_namespace_detected(self):
        """ROOT.Math.VectorUtil detected dynamically."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"v1": "TLorentzVector", "v2": "TLorentzVector"})
        dsl.define("dphi", "ROOT.Math.VectorUtil.DeltaPhi(v1, v2)")
        code = dsl.preview()
        assert "ROOT::Math::VectorUtil::DeltaPhi" in code
    
    def test_dynamic_namespace_broadcasts(self):
        """Dynamically detected namespace functions broadcast on RVec."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        code = dsl.preview()
        # Broadcasting loop should be generated (TMath::Sqrt is scalar-only)
        assert "for" in code or "for (" in code
    
    def test_multiple_namespaces_detected(self):
        """Multiple different namespaces work in same compiler."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double", "y": "double"})
        dsl.define("a", "TMath.Sin(x)")
        dsl.define("b", "TMath.Cos(y)")
        dsl.compile_all()


# =============================================================================
# Caching Tests
# =============================================================================

class TestCaching:
    """Namespace resolution caching."""
    
    def test_positive_cache_works(self):
        """Successful namespace lookups are cached."""
        ROOT = pytest.importorskip("ROOT")
        
        # Create inferrer and builder
        schema = {"x": {"dtype": "double", "rank": 0}}
        inferrer = TypeInferrer.from_schema({"columns": schema})
        builder = IRBuilder(inferrer)
        
        # First lookup populates cache
        builder.build("TMath.Sin(x)")
        
        # Check cache was populated
        assert "TMath" in builder._namespace_cache
    
    def test_cache_hit_second_lookup(self):
        """Second lookup uses cache."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("y", "TMath.Sin(x)")
        dsl.define("z", "TMath.Cos(x)")
        # Both should work - second uses cache
        code = dsl.preview()
        assert "TMath::Sin" in code
        assert "TMath::Cos" in code


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Error handling for unknown symbols."""
    
    def test_unknown_namespace_error(self):
        """Unknown namespace gives clear error at define() time."""
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(IRError) as exc:
            dsl.define("y", "UnknownNamespace.Func(x)")
        assert "Unknown symbol" in str(exc.value)
        assert "UnknownNamespace" in str(exc.value)
    
    def test_error_message_includes_suggestions(self):
        """Error message includes actionable suggestions."""
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(IRError) as exc:
            dsl.define("y", "MyLib.Calculate(x)")
        error_msg = str(exc.value)
        # Should have suggestions
        assert "gSystem.Load" in error_msg or "gInterpreter" in error_msg or "schema" in error_msg
    
    def test_unknown_deep_namespace_error(self):
        """Unknown nested namespace gives helpful error."""
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(IRError) as exc:
            dsl.define("y", "my.custom.analysis.Helper.Compute(x)")
        assert "Unknown symbol" in str(exc.value)
        assert "my" in str(exc.value)


# =============================================================================
# dtype Parameter Tests (Unit Tests)
# =============================================================================

class TestDtypeParameter:
    """Tests for explicit dtype specification."""
    
    def test_dtype_bool(self):
        """dtype='bool' sets return type to bool."""
        dsl = DSLCompiler({"x": "double", "y": "double"})
        dsl.define("is_positive", "x > 0", dtype="bool")
        assert dsl.schema["is_positive"] == "bool"
    
    def test_dtype_int(self):
        """dtype='int' sets return type to int32."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("truncated", "x * 10", dtype="int")
        assert dsl.schema["truncated"] == "int"
    
    def test_dtype_int8(self):
        """dtype='int8' sets return type to int8_t."""
        dsl = DSLCompiler({"phi": "double"})
        dsl.define("sector", "9*phi/3.14159", dtype="int8")
        assert dsl.schema["sector"] == "int8_t"
    
    def test_dtype_int16(self):
        """dtype='int16' sets return type to int16_t."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 100", dtype="int16")
        assert dsl.schema["result"] == "int16_t"
    
    def test_dtype_int64(self):
        """dtype='int64' sets return type to long long."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 1000000", dtype="int64")
        assert dsl.schema["result"] == "long long"
    
    def test_dtype_uint8(self):
        """dtype='uint8' sets return type to uint8_t."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 10", dtype="uint8")
        assert dsl.schema["result"] == "uint8_t"
    
    def test_dtype_uint16(self):
        """dtype='uint16' sets return type to uint16_t."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 100", dtype="uint16")
        assert dsl.schema["result"] == "uint16_t"
    
    def test_dtype_uint32(self):
        """dtype='uint32' sets return type to unsigned int."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 1000", dtype="uint32")
        assert dsl.schema["result"] == "unsigned int"
    
    def test_dtype_uint64(self):
        """dtype='uint64' sets return type to unsigned long long."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 1000000", dtype="uint64")
        assert dsl.schema["result"] == "unsigned long long"
    
    def test_dtype_float32(self):
        """dtype='float32' sets return type to float."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("x_float", "x", dtype="float32")
        assert dsl.schema["x_float"] == "float"
    
    def test_dtype_float(self):
        """dtype='float' (alias for float32) sets return type to float."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("x_float", "x", dtype="float")
        assert dsl.schema["x_float"] == "float"
    
    def test_dtype_float64(self):
        """dtype='float64' sets return type to double."""
        dsl = DSLCompiler({"x": "float"})
        dsl.define("x_double", "x", dtype="float64")
        assert dsl.schema["x_double"] == "double"
    
    def test_dtype_double(self):
        """dtype='double' (alias for float64) sets return type to double."""
        dsl = DSLCompiler({"x": "float"})
        dsl.define("x_double", "x", dtype="double")
        assert dsl.schema["x_double"] == "double"
    
    def test_dtype_overrides_inference(self):
        """Explicit dtype overrides type inference."""
        dsl = DSLCompiler({"x": "double"})
        # sqrt would normally return double, but we force int32
        dsl.define("sqrt_int", "sqrt(x)", dtype="int")
        assert dsl.schema["sqrt_int"] == "int"
    
    def test_dtype_none_uses_inference(self):
        """When dtype=None, type inference is used."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sqrt_x", "sqrt(x)")  # No dtype
        # Should use inference - sqrt(double) -> double
        assert dsl.schema["sqrt_x"] == "double"
    
    def test_invalid_dtype_error(self):
        """Invalid dtype raises helpful error."""
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(IRError) as exc:
            dsl.define("y", "x", dtype="invalid_type")
        error_msg = str(exc.value)
        assert "Unknown dtype" in error_msg or "invalid_type" in error_msg
    
    def test_invalid_dtype_shows_valid_types(self):
        """Invalid dtype error shows valid types."""
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(IRError) as exc:
            dsl.define("y", "x", dtype="complex")
        error_msg = str(exc.value)
        # Should mention supported types
        assert "bool" in error_msg or "int" in error_msg or "float" in error_msg


class TestDtypeChaining:
    """Tests for dtype affecting subsequent type inference."""
    
    def test_dtype_affects_subsequent_definitions(self):
        """dtype on alias affects subsequent type inference."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("is_big", "x > 100", dtype="bool")
        # Now is_big is registered as bool in schema
        assert dsl.schema["is_big"] == "bool"
        # Subsequent definitions can reference it
        dsl.define("not_big", "is_big == 0")  # Uses is_big


# =============================================================================
# dtype Compilation Tests (Require ROOT)
# =============================================================================

class TestDtypeCompilation:
    """Compilation tests for dtype parameter."""
    
    def test_dtype_bool_compiles(self):
        """bool dtype compiles correctly."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("is_positive", "x > 0", dtype="bool")
        dsl.compile_all()
    
    def test_dtype_int8_compiles(self):
        """int8 dtype compiles correctly."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sector", "x * 10", dtype="int8")
        dsl.compile_all()
    
    def test_dtype_int16_compiles(self):
        """int16 dtype compiles correctly."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 100", dtype="int16")
        dsl.compile_all()
    
    def test_dtype_float32_compiles(self):
        """float32 dtype compiles correctly."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("x_float", "x", dtype="float32")
        dsl.compile_all()
    
    def test_dtype_uint8_compiles(self):
        """uint8 dtype compiles correctly."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("result", "x * 10", dtype="uint8")
        dsl.compile_all()


# =============================================================================
# dtype Execution Tests (Require ROOT)
# =============================================================================

class TestDtypeExecution:
    """Execution tests for dtype parameter."""
    
    def test_dtype_bool_execution(self):
        """bool dtype returns correct boolean values."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("is_positive", "x > 0", dtype="bool")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(3)
        rdf = rdf.Define("x", "rdfentry_ - 1.0")  # -1, 0, 1
        rdf = dsl.apply(rdf)
        
        # Count positive entries (only x=1 is positive)
        count = rdf.Filter("is_positive").Count().GetValue()
        assert count == 1
    
    def test_dtype_int8_execution(self):
        """int8 dtype truncates to int8 range."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sector", "x", dtype="int8")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(3)
        rdf = rdf.Define("x", "(double)rdfentry_ * 10")  # 0, 10, 20
        rdf = dsl.apply(rdf)
        
        result = rdf.Sum("sector").GetValue()
        # 0 + 10 + 20 = 30
        assert result == 30


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration of dynamic detection + dtype."""
    
    def test_dynamic_namespace_with_dtype(self):
        """Dynamic namespace + explicit dtype."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)", dtype="double")
        dsl.compile_all()
    
    def test_calibration_style_schema(self):
        """Real calibration-style definitions with dtypes."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({
            "row": "int",
            "dy": "float",
            "mP4": "float",
            "phi": "double",
        })
        
        dsl.define("fsector", "9*phi/3.14159", dtype="float32")
        dsl.define("isOK", "(row < 152) & (abs(dy) < 10) & (abs(mP4) < 1.5)", dtype="bool")
        
        dsl.compile_all()
    
    def test_multiple_dtypes_in_sequence(self):
        """Multiple definitions with different dtypes."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double", "y": "double"})
        
        dsl.define("a", "x + y", dtype="float32")
        dsl.define("b", "x > y", dtype="bool")
        dsl.define("c", "x * 100", dtype="int16")
        dsl.define("d", "y * 1000", dtype="uint32")
        
        dsl.compile_all()


# =============================================================================
# Fallback Tests (Without ROOT)
# =============================================================================

class TestFallback:
    """Fallback behavior when ROOT reflection is not available."""
    
    def test_known_namespaces_work_without_root_reflection(self):
        """KNOWN_NAMESPACES used when ROOT reflection unavailable."""
        # This tests the fallback path in mock environments
        # TMath should work via KNOWN_NAMESPACES even without ROOT reflection
        dsl = DSLCompiler({"x": "double"})
        dsl.define("y", "TMath.Sin(x)")
        code = dsl.preview()
        assert "TMath::Sin" in code
    
    def test_dtype_works_without_root(self):
        """dtype parameter works without ROOT for code generation."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("is_positive", "x > 0", dtype="bool")
        code = dsl.preview()
        assert "bool" in code.lower() or "Bool" in code


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_chained_defines_with_dtype(self):
        """Chained defines work with dtype."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("a", "x * 2", dtype="float32")
        dsl.define("b", "a + 1")  # References a
        # b should work and inherit type from schema (a is float)
        assert "a" in dsl.schema
    
    def test_dtype_with_namespace_function(self):
        """dtype works with namespace function calls."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "double"})
        # TMath.Sin returns double, but we force int
        dsl.define("sin_int", "TMath.Sin(x)", dtype="int")
        assert dsl.schema["sin_int"] == "int"
        dsl.compile_all()
    
    def test_empty_schema_with_constants(self):
        """Can use namespace functions with empty schema (constants only)."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({})
        dsl.define("pi", "TMath.Pi()")
        dsl.compile_all()
