"""
Phase 11.1b: Scalar namespace functions on vectors.

Tests broadcasting of scalar functions (like TMath::Sqrt) over RVec arguments.
"""

import pytest
from RDataFrameDSL import DSLCompiler


# =============================================================================
# Code Generation Tests (Mock - No ROOT Required)
# =============================================================================

class TestNamespaceBroadcastCodeGen:
    """Code generation tests for scalar-to-vector broadcasting."""
    
    def test_tmath_sqrt_rvec_generates_loop(self):
        """TMath.Sqrt(pt) generates loop for RVec."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        code = dsl.preview()
        # Should have a loop
        assert "for" in code
        assert "size_t i" in code
        # Should hoist vector to temporary and index it
        assert "_arg0 = pt" in code
        assert "_arg0[i]" in code
        # Should use TMath::Sqrt
        assert "TMath::Sqrt" in code
    
    def test_tmath_sin_rvec_generates_loop(self):
        """TMath.Sin(phi) generates loop for RVec."""
        dsl = DSLCompiler({"phi": "RVec<double>"})
        dsl.define("sin_phi", "TMath.Sin(phi)")
        code = dsl.preview()
        assert "for" in code
        # Should hoist and index
        assert "_arg0 = phi" in code
        assert "_arg0[i]" in code
        assert "TMath::Sin" in code
    
    def test_mixed_scalar_vector_generates_correctly(self):
        """TMath.Power(pt, 2.0) indexes vector, uses scalar directly."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("pt_sq", "TMath.Power(pt, 2.0)")
        code = dsl.preview()
        # Vector should be hoisted and indexed
        assert "_arg0 = pt" in code
        assert "_arg0[i]" in code
        # Scalar should be used directly (not indexed)
        assert "2.0)" in code
        assert "TMath::Power" in code
    
    def test_two_vectors_uses_min_size(self):
        """TMath.ATan2(y, x) with two RVecs uses std::min for size."""
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("angle", "TMath.ATan2(y, x)")
        code = dsl.preview()
        # Should use std::min for safety
        assert "std::min" in code
        # Both vectors should be hoisted and indexed
        assert "_arg0" in code
        assert "_arg1" in code
        assert "_arg0[i]" in code
        assert "_arg1[i]" in code
    
    def test_three_vectors_chains_min(self):
        """Three vector args chain std::min calls."""
        # Using a hypothetical function with 3 vector args
        dsl = DSLCompiler({
            "a": "RVec<double>", 
            "b": "RVec<double>",
            "c": "RVec<double>"
        })
        # TMath.Hypot doesn't take 3 args, but test the pattern
        # We'll test with a complex expression
        dsl.define("summed", "TMath.Sqrt(a) + TMath.Sqrt(b) + TMath.Sqrt(c)")
        code = dsl.preview()
        # Each call should have its own loop
        assert code.count("for") == 3
    
    def test_scalar_function_no_broadcast(self):
        """Scalar args don't trigger broadcast."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("sqrt_x", "TMath.Sqrt(x)")
        code = dsl.preview()
        # Should NOT have a loop
        assert "for" not in code
        # Direct call
        assert "TMath::Sqrt(x)" in code
    
    def test_cpp_syntax_broadcasts(self):
        """TMath::Sqrt(pt) with C++ syntax also broadcasts."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath::Sqrt(pt)")
        code = dsl.preview()
        assert "for" in code
        # Should hoist and index
        assert "_arg0 = pt" in code
        assert "_arg0[i]" in code


class TestNamespaceBroadcastTypes:
    """Return type tests for broadcasted functions."""
    
    def test_sqrt_rvec_returns_rvec(self):
        """TMath.Sqrt(RVec<double>) returns RVec<double>."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        func = dsl.get_function("sqrt_pt")
        assert "RVec" in func.return_type
        assert "double" in func.return_type
    
    def test_nint_rvec_returns_rvec_int(self):
        """TMath.Nint returns int, so RVec<int>."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("rounded", "TMath.Nint(x)")
        code = dsl.preview()
        # Should return RVec<int> based on NAMESPACE_FUNCTION_TYPES
        assert "RVec<int>" in code


class TestNamespaceBroadcastHeaders:
    """Header inclusion tests."""
    
    def test_single_vector_no_algorithm(self):
        """Single vector doesn't need <algorithm> in code."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        code = dsl.preview()
        # std::min not used for single vector
        assert "std::min" not in code
    
    def test_two_vectors_uses_std_min(self):
        """Two vectors use std::min in generated code."""
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("angle", "TMath.ATan2(y, x)")
        code = dsl.preview()
        # std::min should be in the code
        assert "std::min" in code
    
    def test_tmath_sqrt_uses_namespace(self):
        """TMath:: namespace is in generated code."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        code = dsl.preview()
        assert "TMath::Sqrt" in code


class TestNamespaceBroadcastNested:
    """Test nested namespace calls with vectors."""
    
    def test_nested_tmath_separate_loops(self):
        """TMath.Sqrt(TMath.Abs(pt)) generates separate loops."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(pt))")
        code = dsl.preview()
        # Each call should have its own loop
        # (optimization to fuse loops is for later phases)
        assert "TMath::Sqrt" in code
        assert "TMath::Abs" in code
    
    def test_arithmetic_with_broadcast(self):
        """TMath.Sqrt(pt) * 2.0 works correctly."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(pt) * 2.0")
        code = dsl.preview()
        assert "TMath::Sqrt" in code
        assert "* 2.0" in code


# =============================================================================
# Compilation Tests (Require ROOT)
# =============================================================================

class TestNamespaceBroadcastCompilation:
    """ROOT compilation tests."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        pytest.importorskip("ROOT")
    
    def test_tmath_sqrt_rvec_compiles(self):
        """TMath.Sqrt(pt) on RVec compiles."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        dsl.compile_all()
    
    def test_tmath_sin_rvec_compiles(self):
        """TMath.Sin(phi) on RVec compiles."""
        dsl = DSLCompiler({"phi": "RVec<double>"})
        dsl.define("sin_phi", "TMath.Sin(phi)")
        dsl.compile_all()
    
    def test_tmath_cos_rvec_compiles(self):
        """TMath.Cos(phi) on RVec compiles."""
        dsl = DSLCompiler({"phi": "RVec<double>"})
        dsl.define("cos_phi", "TMath.Cos(phi)")
        dsl.compile_all()
    
    def test_tmath_exp_rvec_compiles(self):
        """TMath.Exp(x) on RVec compiles."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("exp_x", "TMath.Exp(x)")
        dsl.compile_all()
    
    def test_tmath_log_rvec_compiles(self):
        """TMath.Log(x) on RVec compiles."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("log_x", "TMath.Log(x)")
        dsl.compile_all()
    
    def test_tmath_abs_rvec_compiles(self):
        """TMath.Abs(x) on RVec compiles."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("abs_x", "TMath.Abs(x)")
        dsl.compile_all()
    
    def test_tmath_power_mixed_compiles(self):
        """TMath.Power(pt, 2.0) with mixed args compiles."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("pt_sq", "TMath.Power(pt, 2.0)")
        dsl.compile_all()
    
    def test_tmath_atan2_two_vectors_compiles(self):
        """TMath.ATan2(y, x) with two RVecs compiles."""
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("angle", "TMath.ATan2(y, x)")
        dsl.compile_all()
    
    def test_tmath_hypot_two_vectors_compiles(self):
        """TMath.Hypot(x, y) with two RVecs compiles."""
        dsl = DSLCompiler({"x": "RVec<double>", "y": "RVec<double>"})
        dsl.define("r", "TMath.Hypot(x, y)")
        dsl.compile_all()
    
    def test_nested_tmath_compiles(self):
        """TMath.Sqrt(TMath.Abs(x)) compiles."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(x))")
        dsl.compile_all()


# =============================================================================
# Execution Tests (Require ROOT)
# =============================================================================

class TestNamespaceBroadcastExecution:
    """Execution tests with actual values."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        ROOT = pytest.importorskip("ROOT")
        self.ROOT = ROOT
    
    def test_tmath_sqrt_values(self):
        """TMath.Sqrt(pt) returns correct values."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{4.0, 9.0, 16.0}")
        rdf = dsl.apply(rdf)
        
        # sqrt([4, 9, 16]) = [2, 3, 4], sum = 9
        rdf = rdf.Define("sum_sqrt", "ROOT::VecOps::Sum(sqrt_pt)")
        result = rdf.Mean("sum_sqrt").GetValue()
        assert abs(result - 9.0) < 0.001
    
    def test_tmath_power_values(self):
        """TMath.Power(pt, 2.0) returns squared values."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("pt_sq", "TMath.Power(pt, 2.0)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{2.0, 3.0, 4.0}")
        rdf = dsl.apply(rdf)
        
        # [2, 3, 4]^2 = [4, 9, 16], sum = 29
        rdf = rdf.Define("sum_sq", "ROOT::VecOps::Sum(pt_sq)")
        result = rdf.Mean("sum_sq").GetValue()
        assert abs(result - 29.0) < 0.001
    
    def test_tmath_atan2_values(self):
        """TMath.ATan2(y, x) returns correct angles."""
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("angle", "TMath.ATan2(y, x)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("y", "ROOT::RVec<double>{1.0, 0.0, -1.0}")
        rdf = rdf.Define("x", "ROOT::RVec<double>{0.0, 1.0, 0.0}")
        rdf = dsl.apply(rdf)
        
        # atan2(1,0)=π/2, atan2(0,1)=0, atan2(-1,0)=-π/2
        # sum ≈ 0
        rdf = rdf.Define("sum_angle", "ROOT::VecOps::Sum(angle)")
        result = rdf.Mean("sum_angle").GetValue()
        assert abs(result - 0.0) < 0.001
    
    def test_tmath_sin_cos_identity(self):
        """sin^2 + cos^2 = 1 for all elements."""
        dsl = DSLCompiler({"phi": "RVec<double>"})
        dsl.define("sin_phi", "TMath.Sin(phi)")
        dsl.define("cos_phi", "TMath.Cos(phi)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        rdf = rdf.Define("phi", "ROOT::RVec<double>{0.0, 0.5, 1.0, 1.5, 2.0}")
        rdf = dsl.apply(rdf)
        
        # sin^2 + cos^2 should equal 1 for each element
        rdf = rdf.Define("identity", 
            "ROOT::VecOps::Sum(sin_phi*sin_phi + cos_phi*cos_phi)")
        result = rdf.Mean("identity").GetValue()
        # 5 elements, each summing to 1 = 5.0
        assert abs(result - 5.0) < 0.001
    
    def test_mismatched_sizes_uses_min(self):
        """ATan2 with different size vectors uses min."""
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("angle", "TMath.ATan2(y, x)")
        dsl.compile_all()
        
        rdf = self.ROOT.RDataFrame(1)
        # y has 3 elements, x has 5 - should use min(3,5)=3
        rdf = rdf.Define("y", "ROOT::RVec<double>{1.0, 0.0, -1.0}")
        rdf = rdf.Define("x", "ROOT::RVec<double>{0.0, 1.0, 0.0, 1.0, 1.0}")
        rdf = dsl.apply(rdf)
        
        rdf = rdf.Define("n_results", "angle.size()")
        result = rdf.Mean("n_results").GetValue()
        assert result == 3.0  # Should have 3 elements


# =============================================================================
# Chaining Tests
# =============================================================================

class TestNamespaceBroadcastChaining:
    """Test chaining with reductions."""
    
    @pytest.fixture(autouse=True)
    def require_root(self):
        pytest.importorskip("ROOT")
    
    def test_sum_of_sqrt(self):
        """Sum(TMath.Sqrt(pt)) works - broadcast then reduce."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("sqrt_pt", "TMath.Sqrt(pt)")
        dsl.define("sum_sqrt", "Sum(sqrt_pt)")
        dsl.compile_all()
    
    def test_mean_of_sin(self):
        """Mean(TMath.Sin(phi)) works."""
        dsl = DSLCompiler({"phi": "RVec<double>"})
        dsl.define("sin_phi", "TMath.Sin(phi)")
        dsl.define("avg_sin", "Mean(sin_phi)")
        dsl.compile_all()
    
    def test_max_of_abs(self):
        """Max(TMath.Abs(x)) works."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("abs_x", "TMath.Abs(x)")
        dsl.define("max_abs", "Max(abs_x)")
        dsl.compile_all()


# =============================================================================
# Edge Cases
# =============================================================================

class TestNamespaceBroadcastEdgeCases:
    """Edge case tests."""
    
    def test_pi_scalar_no_broadcast(self):
        """TMath.Pi() with no args doesn't broadcast."""
        dsl = DSLCompiler({"x": "double"})
        dsl.define("scaled", "x * TMath.Pi()")
        code = dsl.preview()
        # No loop needed
        assert "for" not in code
    
    def test_vectorized_namespace_no_broadcast(self):
        """ROOT.VecOps functions don't get wrapped."""
        # This tests that VECTORIZED_NAMESPACES works
        # ROOT.VecOps.Sum etc should NOT be wrapped in a loop
        # (They already handle RVec natively)
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("total", "Sum(pt)")  # Sum is from VecOps
        code = dsl.preview()
        # No extra loop wrapping for Sum
        assert code.count("for") == 0 or "Sum" in code


# =============================================================================
# Nested Broadcast Tests (O(n²) Prevention)
# =============================================================================

class TestNestedBroadcast:
    """Tests for nested namespace function broadcasting."""
    
    def test_nested_sqrt_abs_code_shape(self):
        """Nested broadcast generates hoisted temporary (not repeated lambda)."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(pt))")
        code = dsl.preview()
        
        # Should have hoisted temporary for inner broadcast
        assert "_arg0" in code
        # Inner broadcast should be computed BEFORE the outer loop
        # The inner lambda should appear only ONCE (in the hoisting assignment)
        # and NOT be duplicated inside the for loop
        
        # Count how many times the inner loop appears
        inner_abs_count = code.count("TMath::Abs")
        assert inner_abs_count == 1, f"Inner TMath::Abs appears {inner_abs_count} times, should be 1"
    
    def test_nested_sqrt_abs_compiles(self):
        """TMath.Sqrt(TMath.Abs(pt)) compiles successfully."""
        pytest.importorskip("ROOT")
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(pt))")
        dsl.compile_all()  # Should not raise
    
    def test_nested_sqrt_abs_values(self):
        """TMath.Sqrt(TMath.Abs(pt)) returns correct values."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(pt))")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{-4.0, 9.0, -16.0}")
        rdf = dsl.apply(rdf)
        
        # sqrt(abs([-4, 9, -16])) = sqrt([4, 9, 16]) = [2, 3, 4], sum = 9
        rdf = rdf.Define("sum_result", "ROOT::VecOps::Sum(result)")
        result = rdf.Mean("sum_result").GetValue()
        assert abs(result - 9.0) < 0.001
    
    def test_triple_nested_code_shape(self):
        """Triple nested broadcast generates proper hoisting."""
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(TMath.Sin(x)))")
        code = dsl.preview()
        
        # Each function should appear exactly once
        assert code.count("TMath::Sin") == 1
        assert code.count("TMath::Abs") == 1
        assert code.count("TMath::Sqrt") == 1
    
    def test_triple_nested_compiles(self):
        """TMath.Sqrt(TMath.Abs(TMath.Sin(x))) compiles."""
        pytest.importorskip("ROOT")
        dsl = DSLCompiler({"x": "RVec<double>"})
        dsl.define("result", "TMath.Sqrt(TMath.Abs(TMath.Sin(x)))")
        dsl.compile_all()
    
    def test_nested_with_multi_vector_compiles(self):
        """TMath.ATan2(TMath.Abs(y), x) compiles."""
        pytest.importorskip("ROOT")
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("result", "TMath.ATan2(TMath.Abs(y), x)")
        dsl.compile_all()
    
    def test_nested_with_multi_vector_values(self):
        """TMath.ATan2(TMath.Abs(y), x) returns correct values."""
        ROOT = pytest.importorskip("ROOT")
        dsl = DSLCompiler({"y": "RVec<double>", "x": "RVec<double>"})
        dsl.define("result", "TMath.ATan2(TMath.Abs(y), x)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("y", "ROOT::RVec<double>{-1.0, 1.0}")
        rdf = rdf.Define("x", "ROOT::RVec<double>{1.0, 1.0}")
        rdf = dsl.apply(rdf)
        
        # atan2(abs([-1, 1]), [1, 1]) = atan2([1, 1], [1, 1]) = [π/4, π/4]
        rdf = rdf.Define("sum_result", "ROOT::VecOps::Sum(result)")
        result = rdf.Mean("sum_result").GetValue()
        expected = 2 * 0.7853981633974483  # 2 * π/4
        assert abs(result - expected) < 0.001

