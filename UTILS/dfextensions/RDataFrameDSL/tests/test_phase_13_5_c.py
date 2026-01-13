"""
Phase 13.5.C Tests: DSL Integration for Registered Functions

Tests overload resolution, define_raw(), and registered function usage in dsl.define().

Test Count: 19 tests
- OV1-OV10: Overload resolution (10 tests)
- AC1-AC3: Core acceptance (3 tests)
- DR1-DR3: define_raw (3 tests)
- VAL1-VAL3: Validation (3 tests)

v0.5 Specification:
- P0-1: Zero-param functions allowed (param_types=[] is valid)
- P0-2: Test assertions use regex call-site checks
- P0-3: return_type=None warns and defaults to Object
"""

import pytest
import re
import warnings

# Import test subjects
try:
    from RDataFrameDSL.dsl_compiler import DSLCompiler
    from RDataFrameDSL.ir_builder import IRBuilder
    from RDataFrameDSL.ir_types import IRTypeKind
    from RDataFrameDSL.ir_errors import IRError
    from RDataFrameDSL.type_inferrer import TypeInferrer
except ImportError:
    from dsl_compiler import DSLCompiler
    from ir_builder import IRBuilder
    from ir_types import IRTypeKind
    from ir_errors import IRError
    from type_inferrer import TypeInferrer


# =============================================================================
# Overload Resolution Tests (OV1-OV10)
# =============================================================================

class TestPhase13_5_C_Overloads:
    """
    Phase 13.5.C v0.5: Overload resolution tests.
    
    10 required tests for P0 coverage.
    """
    
    def test_OV1_scalar_overload_selected(self):
        """OV1: Scalar arguments select scalar overload."""
        dsl = DSLCompiler({
            "px": "double",
            "py": "double",
            "px_vec": "RVec<double>",
            "py_vec": "RVec<double>",
        })
        
        # Register scalar version
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        # Register RVec version
        dsl.register_function_cpp('''
            RVec<double> pt(const RVec<double>& px, const RVec<double>& py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        # Use with scalar args — must select scalar overload
        dsl.define("track_pt", "pt(px, py)")
        
        func = dsl._functions["track_pt"]
        # Verify scalar version cpp_name is in generated code
        scalar_cpp_name = dsl._dsl_registered_functions['pt'][0]['cpp_name']
        vector_cpp_name = dsl._dsl_registered_functions['pt'][1]['cpp_name']
        
        # v0.5 FIX (P0-2): Check for actual function CALL, not just presence
        scalar_call = rf'\b{re.escape(scalar_cpp_name)}\s*\('
        assert re.search(scalar_call, func.code), \
            f"Scalar overload {scalar_cpp_name} not called in: {func.code}"
        
        # Ensure wrong overload is NOT called
        vector_call = rf'\b{re.escape(vector_cpp_name)}\s*\('
        assert not re.search(vector_call, func.code), \
            f"Vector overload {vector_cpp_name} incorrectly called"
    
    def test_OV2_rvec_overload_selected(self):
        """OV2: RVec arguments select RVec overload."""
        dsl = DSLCompiler({
            "px": "double",
            "py": "double",
            "px_vec": "RVec<double>",
            "py_vec": "RVec<double>",
        })
        
        # Register scalar version first
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        # Register RVec version
        dsl.register_function_cpp('''
            RVec<double> pt(const RVec<double>& px, const RVec<double>& py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        # Use with RVec args — must select RVec overload
        dsl.define("track_pts", "pt(px_vec, py_vec)")
        
        func = dsl._functions["track_pts"]
        scalar_cpp_name = dsl._dsl_registered_functions['pt'][0]['cpp_name']
        vector_cpp_name = dsl._dsl_registered_functions['pt'][1]['cpp_name']
        
        # v0.5 FIX (P0-2): Check for actual function CALL
        vector_call = rf'\b{re.escape(vector_cpp_name)}\s*\('
        assert re.search(vector_call, func.code), \
            f"Vector overload {vector_cpp_name} not called in: {func.code}"
        
        # Ensure wrong overload is NOT called
        scalar_call = rf'\b{re.escape(scalar_cpp_name)}\s*\('
        assert not re.search(scalar_call, func.code), \
            f"Scalar overload {scalar_cpp_name} incorrectly called"
    
    def test_OV3_wrong_arity_error(self):
        """OV3: Wrong arity produces clear error with available arities."""
        dsl = DSLCompiler({"px": "double", "py": "double", "pz": "double"})
        
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("wrong", "pt(px, py, pz)")  # 3 args, expects 2
        
        error_msg = str(exc_info.value)
        assert "3" in error_msg  # Mentions wrong arity
        assert "2" in error_msg or "arity" in error_msg.lower()
    
    def test_OV4_rank_mismatch_error(self):
        """OV4: Rank mismatch produces clear error."""
        dsl = DSLCompiler({
            "px_vec": "RVec<double>",
            "py_vec": "RVec<double>",
        })
        
        # Only register scalar version
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("wrong", "pt(px_vec, py_vec)")  # RVec args, only scalar available
        
        error_msg = str(exc_info.value)
        assert "match" in error_msg.lower() or "type" in error_msg.lower()
    
    def test_OV5_same_signature_ambiguity_error(self):
        """
        OV5: Same signature registered twice raises ambiguity error.
        
        Phase 13.5.D Q3 Decision (7/7 unanimous): Error on ambiguity.
        When multiple overloads have the same rank, raise an error
        instead of silently picking one.
        """
        dsl = DSLCompiler({"px": "double", "py": "double"})
        
        # Register first version
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        # Register same signature again (different body = different hash)
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return px + py;
            }
        ''')
        
        # Phase 13.5.D: Same signature = same rank = ambiguity error
        with pytest.raises(IRError) as exc_info:
            dsl.define("track_pt", "pt(px, py)")
        
        error_msg = str(exc_info.value)
        assert "ambiguous" in error_msg.lower(), \
            f"Expected 'ambiguous' in error, got: {error_msg}"
    
    def test_OV6_execution_smoke_test(self):
        """OV6: End-to-end execution with correct overload."""
        try:
            import ROOT
        except ImportError:
            pytest.skip("ROOT not available")
        
        dsl = DSLCompiler({
            "px": "double",
            "py": "double",
        })
        
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        dsl.define("track_pt", "pt(px, py)")
        
        rdf = ROOT.RDataFrame(10)
        rdf = rdf.Define("px", "3.0")
        rdf = rdf.Define("py", "4.0")
        rdf = dsl.apply(rdf)
        
        # sqrt(9+16) = 5, × 10 events = 50
        result = rdf.Sum("track_pt").GetValue()
        assert abs(result - 50.0) < 0.001
    
    def test_OV7_int_vs_double_exact_match(self):
        """OV7: f(int,int) vs f(double,double) — exact kind match."""
        dsl = DSLCompiler({
            "x": "int",
            "y": "int",
            "a": "double",
            "b": "double",
        })
        
        dsl.register_function_cpp('''
            double f(int a, int b) { return a + b; }
        ''')
        int_cpp_name = dsl._dsl_registered_functions['f'][0]['cpp_name']
        
        dsl.register_function_cpp('''
            double f(double a, double b) { return a * b; }
        ''')
        double_cpp_name = dsl._dsl_registered_functions['f'][1]['cpp_name']
        
        # Use with int args — must select int overload
        dsl.define("int_result", "f(x, y)")
        int_func = dsl._functions["int_result"]
        
        # v0.5 FIX (P0-2): Check actual call
        int_call = rf'\b{re.escape(int_cpp_name)}\s*\('
        double_call = rf'\b{re.escape(double_cpp_name)}\s*\('
        
        assert re.search(int_call, int_func.code), \
            f"Int overload not called"
        assert not re.search(double_call, int_func.code), \
            f"Double overload incorrectly called for int args"
        
        # Use with double args — must select double overload
        dsl.define("double_result", "f(a, b)")
        double_func = dsl._functions["double_result"]
        
        assert re.search(double_call, double_func.code), \
            f"Double overload not called"
        assert not re.search(int_call, double_func.code), \
            f"Int overload incorrectly called for double args"
    
    def test_OV8_rvec_int_vs_rvec_double_overload(self):
        """
        OV8: RVec<int> vs RVec<double> — exact kind match with SAME function name.
        
        v0.5 FIX (P0-2): Now uses same function name 'sum' to properly test overload resolution.
        """
        dsl = DSLCompiler({
            "int_vec": "RVec<int>",
            "double_vec": "RVec<double>",
        })
        
        # Register SAME NAME with different element types
        dsl.register_function_cpp('''
            int mysum(const RVec<int>& v) { return Sum(v); }
        ''')
        int_cpp_name = dsl._dsl_registered_functions['mysum'][0]['cpp_name']
        
        dsl.register_function_cpp('''
            double mysum(const RVec<double>& v) { return Sum(v); }
        ''')
        double_cpp_name = dsl._dsl_registered_functions['mysum'][1]['cpp_name']
        
        # Use with RVec<int> — must select int overload
        dsl.define("int_sum", "mysum(int_vec)")
        
        int_call = rf'\b{re.escape(int_cpp_name)}\s*\('
        double_call = rf'\b{re.escape(double_cpp_name)}\s*\('
        
        assert re.search(int_call, dsl._functions["int_sum"].code), \
            "Int overload not called for RVec<int>"
        
        # Use with RVec<double> — must select double overload
        dsl.define("double_sum", "mysum(double_vec)")
        
        assert re.search(double_call, dsl._functions["double_sum"].code), \
            "Double overload not called for RVec<double>"
    
    def test_OV9_error_message_quality(self):
        """
        OV9: Error messages include helpful diagnostics.
        
        Phase 13.5.D: Tests narrowing rejection (double→float is forbidden).
        Note: int→double now works via conversion (rank 2).
        """
        dsl = DSLCompiler({
            "x": "double",  # Float64
            "y": "double",
        })
        
        # Only float overload available - narrowing required
        dsl.register_function_cpp('''
            float f(float a, float b) { return a * b; }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("wrong", "f(x, y)")  # double args, only float available
        
        error_msg = str(exc_info.value)
        # Error should mention:
        # - Function name
        assert "f" in error_msg
        # - What was expected vs got (type mismatch info)
        assert "narrowing" in error_msg.lower() or "no overload" in error_msg.lower()
    
    def test_OV10_zero_parameter_function(self):
        """
        OV10: Zero-parameter functions work correctly.
        
        v0.5 NEW: Tests that param_types=[] is allowed for zero-arg functions.
        """
        dsl = DSLCompiler({"x": "double"})
        
        # Register zero-parameter function
        dsl.register_function_cpp('''
            double pi() { return 3.14159265359; }
        ''')
        
        # Use in expression
        dsl.define("circle_area", "pi() * x * x")
        
        # Should work
        assert "circle_area" in [n for n, _ in dsl._definitions]
        
        # Verify function registered with empty param_types
        assert 'pi' in dsl._dsl_registered_functions
        assert len(dsl._dsl_registered_functions['pi'][0]['param_types']) == 0


# =============================================================================
# Acceptance Tests (AC1-AC3)
# =============================================================================

class TestPhase13_5_C_Acceptance:
    """Core functionality tests."""
    
    def test_AC1_registered_function_in_define(self):
        """AC1: User can use registered function in dsl.define()."""
        dsl = DSLCompiler({"px": "double", "py": "double"})
        
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        dsl.define("track_pt", "pt(px, py)")
        
        assert "track_pt" in [n for n, _ in dsl._definitions]
        func = dsl._functions["track_pt"]
        assert "dsl_pt_" in func.code
    
    def test_AC2_registered_function_in_expression(self):
        """AC2: Registered function works in complex expressions."""
        dsl = DSLCompiler({"px": "double", "py": "double", "pz": "double"})
        
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        dsl.define("p_total", "sqrt(pt(px, py)**2 + pz**2)")
        dsl.define("high_pt", "pt(px, py) > 10.0")
        
        names = [n for n, _ in dsl._definitions]
        assert "p_total" in names
        assert "high_pt" in names
    
    def test_AC3_multiple_functions(self):
        """AC3: Multiple registered functions work together."""
        dsl = DSLCompiler({"px": "double", "py": "double", "pz": "double"})
        
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        dsl.register_function_cpp('''
            double eta(double px, double py, double pz) {
                double p = sqrt(px*px + py*py + pz*pz);
                return 0.5 * log((p + pz) / (p - pz));
            }
        ''')
        
        dsl.define("pt_eta", "pt(px, py) * eta(px, py, pz)")
        assert "pt_eta" in [n for n, _ in dsl._definitions]


# =============================================================================
# define_raw() Tests (DR1-DR3)
# =============================================================================

class TestPhase13_5_C_DefineRaw:
    """define_raw() escape hatch tests."""
    
    def test_DR1_basic(self):
        """DR1: define_raw() creates column."""
        dsl = DSLCompiler({"E": "double", "px": "double", "py": "double", "pz": "double"})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dsl.define_raw("mass", "std::sqrt(E*E - px*px - py*py - pz*pz)")
            
            # v0.5: Verify warning emitted (tolerant assertion per GPT3 Team2)
            assert any("bypass" in str(warning.message).lower() for warning in w), \
                "Expected warning about bypassing DSL parsing"
        
        assert "mass" in [n for n, _ in dsl._definitions]
        assert dsl._functions["mass"].is_raw == True
    
    def test_DR2_lambda_rejected(self):
        """DR2: define_raw() rejects lambda (FROZEN RULE #1)."""
        dsl = DSLCompiler({"x": "double"})
        
        with pytest.raises(IRError) as exc_info:
            dsl.define_raw("result", "[](double x) { return x * 2; }(x)")
        
        assert "FROZEN RULE #1" in str(exc_info.value)
    
    def test_DR3_schema_collision(self):
        """DR3: define_raw() rejects schema column names."""
        dsl = DSLCompiler({"px": "double"})
        
        with pytest.raises(IRError) as exc_info:
            dsl.define_raw("px", "px * 2")
        
        assert "conflict" in str(exc_info.value).lower()


# =============================================================================
# Validation Tests (VAL1-VAL3)
# =============================================================================

class TestPhase13_5_C_Validation:
    """v0.5 validation tests."""
    
    def test_VAL1_param_types_none_rejected(self):
        """VAL1: register_function() requires param_types (rejects None)."""
        inferrer = TypeInferrer.from_schema({"x": "double"})
        builder = IRBuilder(inferrer)
        
        with pytest.raises(ValueError) as exc_info:
            builder.register_function(
                name="bad_func",
                cpp_name="dsl_bad_xxx",
                return_type=IRTypeKind.Float64,
                headers=[],
                param_types=None,  # Missing!
            )
        
        error_msg = str(exc_info.value)
        assert "param_types" in error_msg
        assert "bad_func" in error_msg
    
    def test_VAL2_empty_param_types_allowed(self):
        """
        VAL2: Empty param_types allowed for zero-parameter functions.
        
        v0.5 FIX: Changed from rejecting [] to allowing it.
        """
        inferrer = TypeInferrer.from_schema({"x": "double"})
        builder = IRBuilder(inferrer)
        
        # Zero-parameter function should work
        builder.register_function(
            name="pi",
            cpp_name="dsl_pi_xxx",
            return_type=IRTypeKind.Float64,
            headers=[],
            param_types=[],  # Valid for zero-param functions
        )
        
        # Should register successfully
        assert "pi" in builder._custom_functions
        assert len(builder._custom_functions["pi"]) == 1
    
    def test_VAL3_return_type_none_warns(self):
        """
        VAL3: register_function() warns and defaults to Object when return_type=None.
        
        v0.5 FIX (P0-3): Clarified return_type contract.
        """
        inferrer = TypeInferrer.from_schema({"x": "double"})
        builder = IRBuilder(inferrer)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.register_function(
                name="unknown_return",
                cpp_name="dsl_unknown_xxx",
                return_type=None,  # Should warn
                headers=[],
                param_types=[{"name": "x", "cpp_type": "double", "rank": 0, "ir_kind": IRTypeKind.Float64}],
            )
            
            # v0.5: Tolerant assertion (per GPT3 Team2 feedback)
            assert any("return_type" in str(warning.message).lower() or 
                      "object" in str(warning.message).lower() for warning in w), \
                "Expected warning about missing return_type"
        
        # Should still register with Object default
        assert "unknown_return" in builder._custom_functions
        registered = builder._custom_functions["unknown_return"][0]
        assert registered["return_type"] == IRTypeKind.Object


# =============================================================================
# Test Count Summary
# =============================================================================
# OV1-OV10: 10 tests
# AC1-AC3: 3 tests
# DR1-DR3: 3 tests
# VAL1-VAL3: 3 tests
# Total: 19 tests
