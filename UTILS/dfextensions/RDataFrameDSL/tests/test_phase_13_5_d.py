"""
Phase 13.5.D Tests: Numeric Widening for Overload Resolution

Tests conversion ranking, promotion, and ambiguity detection.

Test Count: 13 tests (WD1-WD13)
- WD1-WD3: Basic conversions
- WD4-WD5: Rank preference
- WD6-WD7: Forbidden conversions
- WD8: Ambiguity detection
- WD9: Multi-argument ranking
- WD10: RVec exact match
- WD11-WD13: Edge cases

v0.2 Specification (7/7 Approved):
- Q1: Int32→Float32 FORBIDDEN (lossy)
- Q2: Exact match (rank 0) always wins
- Q3: Ambiguity raises error
"""

import pytest
import warnings

# Import test subjects
try:
    from RDataFrameDSL.dsl_compiler import DSLCompiler
    from RDataFrameDSL.ir_builder import IRBuilder, CONVERSION_MATRIX
    from RDataFrameDSL.ir_types import IRTypeKind
    from RDataFrameDSL.ir_errors import IRError
    from RDataFrameDSL.type_inferrer import TypeInferrer
except ImportError:
    from dsl_compiler import DSLCompiler
    from ir_builder import IRBuilder, CONVERSION_MATRIX
    from ir_types import IRTypeKind
    from ir_errors import IRError
    from type_inferrer import TypeInferrer


# =============================================================================
# Conversion Matrix Sanity Tests
# =============================================================================

class TestConversionMatrix:
    """Verify CONVERSION_MATRIX is correctly defined."""
    
    def test_matrix_exact_match_diagonal(self):
        """All types have exact match with themselves (rank 0)."""
        for from_kind in CONVERSION_MATRIX:
            assert CONVERSION_MATRIX[from_kind][from_kind] == 0, \
                f"{from_kind.name} should match itself with rank 0"
    
    def test_matrix_float_promotion(self):
        """Float32 → Float64 is promotion (rank 1)."""
        assert CONVERSION_MATRIX[IRTypeKind.Float32][IRTypeKind.Float64] == 1
    
    def test_matrix_int_widening(self):
        """Integer widening is promotion (rank 1)."""
        assert CONVERSION_MATRIX[IRTypeKind.Int8][IRTypeKind.Int16] == 1
        assert CONVERSION_MATRIX[IRTypeKind.Int16][IRTypeKind.Int32] == 1
        assert CONVERSION_MATRIX[IRTypeKind.Int32][IRTypeKind.Int64] == 1
    
    def test_matrix_int_to_double(self):
        """Int → Float64 is conversion (rank 2)."""
        assert CONVERSION_MATRIX[IRTypeKind.Int32][IRTypeKind.Float64] == 2
        assert CONVERSION_MATRIX[IRTypeKind.Int64][IRTypeKind.Float64] == 2
    
    def test_matrix_narrowing_forbidden(self):
        """Narrowing is forbidden (rank -1)."""
        assert CONVERSION_MATRIX[IRTypeKind.Float64][IRTypeKind.Float32] == -1
        assert CONVERSION_MATRIX[IRTypeKind.Int64][IRTypeKind.Int32] == -1
    
    def test_matrix_int_to_float32_forbidden(self):
        """Int → Float32 is forbidden (Q1 decision)."""
        assert CONVERSION_MATRIX[IRTypeKind.Int32][IRTypeKind.Float32] == -1
        assert CONVERSION_MATRIX[IRTypeKind.Int64][IRTypeKind.Float32] == -1
    
    def test_matrix_signed_unsigned_forbidden(self):
        """Signed ↔ Unsigned is forbidden."""
        assert CONVERSION_MATRIX[IRTypeKind.Int32][IRTypeKind.UInt32] == -1
        assert CONVERSION_MATRIX[IRTypeKind.UInt32][IRTypeKind.Int32] == -1


# =============================================================================
# Widening Tests (WD1-WD13)
# =============================================================================

class TestPhase13_5_D_Widening:
    """Phase 13.5.D: Numeric widening tests."""
    
    def test_WD1_float32_to_float64_promotion(self):
        """WD1: Float32 → Float64 promotion works (rank 1)."""
        dsl = DSLCompiler({"x": "float"})  # Float32
        
        dsl.register_function_cpp('''
            double f(double x) { return x * 2; }
        ''')
        
        # Should work with promotion
        dsl.define("result", "f(x)")
        assert "result" in [n for n, _ in dsl._definitions]
    
    def test_WD2_int32_to_int64_widening(self):
        """WD2: Int32 → Int64 widening works (rank 1)."""
        dsl = DSLCompiler({"n": "int"})  # Int32
        
        dsl.register_function_cpp('''
            long f(long x) { return x * 2; }
        ''')
        
        # Should work with widening
        dsl.define("result", "f(n)")
        assert "result" in [n for n, _ in dsl._definitions]
    
    def test_WD3_int32_to_float64_conversion(self):
        """WD3: Int32 → Float64 conversion works (rank 2)."""
        dsl = DSLCompiler({"n": "int"})  # Int32
        
        dsl.register_function_cpp('''
            double f(double x) { return x * 2; }
        ''')
        
        # Should work with conversion
        dsl.define("result", "f(n)")
        assert "result" in [n for n, _ in dsl._definitions]
    
    def test_WD4_exact_preferred_over_promotion(self):
        """WD4: Exact match (rank 0) preferred over promotion (rank 1)."""
        dsl = DSLCompiler({"x": "float"})  # Float32
        
        # Register both overloads
        dsl.register_function_cpp('''
            double f(float x) { return x; }
        ''')
        float_cpp_name = dsl._dsl_registered_functions['f'][0]['cpp_name']
        
        dsl.register_function_cpp('''
            double f(double x) { return x * 2; }
        ''')
        double_cpp_name = dsl._dsl_registered_functions['f'][1]['cpp_name']
        
        dsl.define("result", "f(x)")
        
        # Should select float overload (exact match)
        import re
        func = dsl._functions["result"]
        float_call = rf'\b{re.escape(float_cpp_name)}\s*\('
        double_call = rf'\b{re.escape(double_cpp_name)}\s*\('
        
        assert re.search(float_call, func.code), "Exact match should be selected"
        assert not re.search(double_call, func.code), "Promotion should not be selected"
    
    def test_WD5_promotion_preferred_over_conversion(self):
        """WD5: Promotion (rank 1) preferred over conversion (rank 2)."""
        dsl = DSLCompiler({"x": "float"})  # Float32
        
        # Register conversion overload first
        dsl.register_function_cpp('''
            int g(int x) { return x; }
        ''')
        
        # Register promotion overload
        dsl.register_function_cpp('''
            double g(double x) { return x * 2; }
        ''')
        double_cpp_name = dsl._dsl_registered_functions['g'][1]['cpp_name']
        
        dsl.define("result", "g(x)")
        
        # Should select double overload (promotion rank 1, not int which would be -1)
        import re
        func = dsl._functions["result"]
        double_call = rf'\b{re.escape(double_cpp_name)}\s*\('
        assert re.search(double_call, func.code), "Promotion should be selected"
    
    def test_WD6_narrowing_rejected(self):
        """WD6: Narrowing (double → float) rejected."""
        dsl = DSLCompiler({"x": "double"})  # Float64
        
        dsl.register_function_cpp('''
            float f(float x) { return x; }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("result", "f(x)")
        
        error_msg = str(exc_info.value)
        assert "narrowing" in error_msg.lower() or "no overload" in error_msg.lower()
    
    def test_WD7_signed_unsigned_rejected(self):
        """WD7: Signed ↔ Unsigned mixing rejected."""
        dsl = DSLCompiler({"n": "int"})  # Int32 (signed)
        
        dsl.register_function_cpp('''
            unsigned int f(unsigned int x) { return x; }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("result", "f(n)")
        
        error_msg = str(exc_info.value)
        assert "no overload" in error_msg.lower() or "signed" in error_msg.lower()
    
    def test_WD8_ambiguity_error(self):
        """WD8: Ambiguous overloads raise clear error."""
        dsl = DSLCompiler({"x": "int", "y": "int"})  # Both Int32
        
        # Two overloads with same total rank
        dsl.register_function_cpp('''
            double f(double a, int b) { return a + b; }
        ''')
        
        dsl.register_function_cpp('''
            double f(int a, double b) { return a + b; }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("result", "f(x, y)")
        
        error_msg = str(exc_info.value)
        assert "ambiguous" in error_msg.lower()
    
    def test_WD9_multi_argument_ranking(self):
        """WD9: Multi-argument ranking sums correctly."""
        dsl = DSLCompiler({"x": "float", "y": "int"})  # Float32, Int32
        
        # Overload A: total rank = 1 + 2 = 3
        dsl.register_function_cpp('''
            double f(double a, double b) { return a + b; }
        ''')
        double_cpp_name = dsl._dsl_registered_functions['f'][0]['cpp_name']
        
        # Overload B: NOT VIABLE (int→float is forbidden)
        dsl.register_function_cpp('''
            double f(float a, float b) { return a * b; }
        ''')
        
        dsl.define("result", "f(x, y)")
        
        # Should select overload A (only viable)
        import re
        func = dsl._functions["result"]
        double_call = rf'\b{re.escape(double_cpp_name)}\s*\('
        assert re.search(double_call, func.code), "Only viable overload should be selected"
    
    def test_WD10_rvec_exact_match_only(self):
        """WD10: RVec types require exact match (no element widening)."""
        dsl = DSLCompiler({"pts": "RVec<float>"})  # RVec<Float32>
        
        dsl.register_function_cpp('''
            double sum_pts(const RVec<double>& v) { return Sum(v); }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("total", "sum_pts(pts)")
        
        error_msg = str(exc_info.value)
        assert "no overload" in error_msg.lower()
    
    def test_WD11_int32_to_float32_forbidden(self):
        """WD11: Int32 → Float32 explicitly forbidden (Q1 decision)."""
        dsl = DSLCompiler({"n": "int"})  # Int32
        
        dsl.register_function_cpp('''
            float f(float x) { return x; }
        ''')
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("result", "f(n)")
        
        error_msg = str(exc_info.value)
        assert "no overload" in error_msg.lower()
    
    def test_WD12_zero_parameter_unaffected(self):
        """WD12: Zero-parameter functions work as before."""
        dsl = DSLCompiler({"x": "double"})
        
        dsl.register_function_cpp('''
            double pi() { return 3.14159265359; }
        ''')
        
        dsl.define("circle_area", "pi() * x * x")
        assert "circle_area" in [n for n, _ in dsl._definitions]
    
    def test_WD13_single_viable_high_rank(self):
        """WD13: Single viable candidate selected even with high rank."""
        dsl = DSLCompiler({"n": "int"})  # Int32
        
        # Only double overload available (rank 2 conversion)
        dsl.register_function_cpp('''
            double compute(double x) { return x * 2; }
        ''')
        
        # Should work (single candidate, rank 2)
        dsl.define("result", "compute(n)")
        assert "result" in [n for n, _ in dsl._definitions]


# =============================================================================
# Execution Tests
# =============================================================================

class TestPhase13_5_D_Execution:
    """End-to-end execution tests."""
    
    def test_promotion_execution(self):
        """Verify promotion works at execution time."""
        try:
            import ROOT
        except ImportError:
            pytest.skip("ROOT not available")
        
        dsl = DSLCompiler({"x": "float"})
        
        dsl.register_function_cpp('''
            double double_it(double x) { return x * 2; }
        ''')
        
        dsl.define("result", "double_it(x)")
        
        rdf = ROOT.RDataFrame(10)
        rdf = rdf.Define("x", "(float)3.5")
        rdf = dsl.apply(rdf)
        
        # 3.5 * 2 = 7.0, × 10 events = 70.0
        total = rdf.Sum("result").GetValue()
        assert abs(total - 70.0) < 0.001


# =============================================================================
# Test Count Summary
# =============================================================================
# Matrix sanity: 7 tests
# WD1-WD13: 13 tests
# Execution: 1 test
# Total: 21 tests
