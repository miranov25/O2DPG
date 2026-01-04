"""
RDataFrameDSL Invariant Testing — DSL Expression Tests

Phase 13.2.2.DSL: Test DSL compiler correctness using mathematical invariants.

This module tests that DSL-compiled C++ expressions preserve mathematical
invariants when executed via ROOT RDataFrame. The test data is generated
with invariants that hold by construction (Phase 13.2.1.DSL).

Test Categories:
    - Arithmetic: (A + B) - A == B, A * 1 == A
    - Precedence: A + B * C vs (A + B) * C
    - Boolean: De Morgan's laws, double negation
    - Comparison: (A > B) == not (A <= B)
    - RVec Element-wise: vec_a + vec_b == vec_sum
    - RVec Reductions: Sum, Mean, length preservation
    - C-Array/Legacy: arr_d[idx] == picked
"""

import pytest
import numpy as np

try:
    from .invariant_schema import (
        TOLERANCE,
        INVARIANT_SCHEMA,
        DEFAULT_N_EVENTS,
    )
except ImportError:
    from invariant_schema import (
        TOLERANCE,
        INVARIANT_SCHEMA,
        DEFAULT_N_EVENTS,
    )

# Check ROOT availability
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

pytestmark = pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dsl():
    """Create DSLCompiler with invariant schema."""
    try:
        from RDataFrameDSL.dsl_compiler import DSLCompiler
    except ImportError:
        from RDataFrameDSL.dsl_compiler import DSLCompiler
    
    # Schema for DSL (excludes C-array raw pointers which need special handling)
    schema = {
        # Scalars
        "A": "double", "B": "double", "C": "double",
        "sum_ab": "double", "prod_ab": "double", "A_positive": "double",
        "prec_add_mul": "double", "prec_mul_add": "double",
        "Af": "float", "Bf": "float", "sum_f": "float",
        "Ai": "int", "Bi": "int", "sum_i": "int",
        "Au": "unsigned int", "Bu": "unsigned int", "sum_u": "unsigned int",
        # Booleans
        "flag_a": "bool", "flag_b": "bool", "flag_c": "bool",
        "flag_and": "bool", "flag_or": "bool", "flag_not_a": "bool",
        # RVecs
        "vec_a": "RVec<double>", "vec_b": "RVec<double>", "vec_sum": "RVec<double>",
        "vec_len": "int",
        "vec_i": "RVec<int>", "vec_i_doubled": "RVec<int>",
        # C-array support columns
        "n_arr": "int", "idx": "int", "picked": "double",
        # Edge cases
        "near_zero": "double", "large_val": "double",
    }
    
    return DSLCompiler(schema)


@pytest.fixture
def test_rdf(invariant_tree_path):
    """Create RDataFrame from invariant test tree."""
    if invariant_tree_path is None:
        pytest.skip("Invariant tree not available")
    
    return ROOT.RDataFrame("invariants", str(invariant_tree_path))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_no_violations(rdf, filter_expr: str, context: str = ""):
    """Assert that no events match the violation filter."""
    count = rdf.Filter(filter_expr).Count().GetValue()
    assert count == 0, f"{context}: {count} violations found with filter '{filter_expr}'"


def assert_column_equal(rdf, col1: str, col2: str, tol: float = 1e-12, context: str = ""):
    """Assert two columns are equal within tolerance."""
    filter_expr = f"abs({col1} - {col2}) > {tol}"
    count = rdf.Filter(filter_expr).Count().GetValue()
    assert count == 0, f"{context}: {col1} != {col2} in {count} events (tol={tol})"


# =============================================================================
# ARITHMETIC INVARIANT TESTS
# =============================================================================

class TestArithmeticInvariants:
    """Test arithmetic expression invariants via DSL."""
    
    def test_addition_subtraction_inverse(self, dsl, test_rdf):
        """Invariant: (A + B) - A == B"""
        dsl.define("computed_b", "sum_ab - A")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(computed_b - B) > 1e-12",
            "(A + B) - A == B"
        )
    
    def test_subtraction_addition_inverse(self, dsl, test_rdf):
        """Invariant: (A - B) + B == A"""
        dsl.define("a_minus_b", "A - B")
        dsl.define("reconstructed_a", "a_minus_b + B")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(reconstructed_a - A) > 1e-12",
            "(A - B) + B == A"
        )
    
    def test_multiplication_division_inverse(self, dsl, test_rdf):
        """Invariant: (A * B) / B == A (for B != 0)"""
        # Use A_positive which is guaranteed > 0
        dsl.define("prod", "A * A_positive")
        dsl.define("reconstructed", "prod / A_positive")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(reconstructed - A) > 1e-10",
            "(A * B) / B == A"
        )
    
    def test_multiplication_identity(self, dsl, test_rdf):
        """Invariant: A * 1 == A"""
        dsl.define("a_times_one", "A * 1.0")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(a_times_one - A) > 1e-14",
            "A * 1 == A"
        )
    
    def test_addition_identity(self, dsl, test_rdf):
        """Invariant: A + 0 == A"""
        dsl.define("a_plus_zero", "A + 0.0")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(a_plus_zero - A) > 1e-14",
            "A + 0 == A"
        )
    
    def test_double_negation(self, dsl, test_rdf):
        """Invariant: -(-A) == A"""
        dsl.define("neg_neg_a", "-(-A)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(neg_neg_a - A) > 1e-14",
            "-(-A) == A"
        )


# =============================================================================
# OPERATOR PRECEDENCE TESTS
# =============================================================================

class TestPrecedenceInvariants:
    """Test operator precedence is correctly preserved by DSL."""
    
    def test_add_mul_precedence(self, dsl, test_rdf):
        """Invariant: A + B * C follows standard precedence (mul before add)."""
        dsl.define("computed", "A + B * C")
        rdf = dsl.apply(test_rdf)
        
        # prec_add_mul was computed as A + (B * C) in generator
        assert_no_violations(
            rdf,
            "abs(computed - prec_add_mul) > 1e-10",
            "A + B * C == A + (B * C)"
        )
    
    def test_explicit_parentheses_add_first(self, dsl, test_rdf):
        """Invariant: (A + B) * C is different from A + B * C."""
        dsl.define("grouped", "(A + B) * C")
        rdf = dsl.apply(test_rdf)
        
        # prec_mul_add was computed as (A + B) * C in generator
        assert_no_violations(
            rdf,
            "abs(grouped - prec_mul_add) > 1e-10",
            "(A + B) * C"
        )
    
    def test_precedence_difference(self, dsl, test_rdf):
        """Verify A + B * C != (A + B) * C in general."""
        dsl.define("standard", "A + B * C")
        dsl.define("grouped", "(A + B) * C")
        rdf = dsl.apply(test_rdf)
        
        # These should be different (except by coincidence)
        # Count where they ARE equal - should be rare
        same_count = rdf.Filter("abs(standard - grouped) < 1e-10").Count().GetValue()
        total = rdf.Count().GetValue()
        
        # Should be different in most cases (allow up to 10% coincidental equality)
        assert same_count < total * 0.1, (
            f"Precedence test: {same_count}/{total} events have equal results - "
            "suggests precedence may not be working"
        )
    
    def test_bool_and_or_precedence(self, dsl, test_rdf):
        """Invariant:  and  has higher precedence than  or """
        # a  or  b  and  c should equal a  or  (b  and  c), not (a  or  b)  and  c
        dsl.define("standard", "flag_a  or  flag_b  and  flag_c")
        dsl.define("and_first", "flag_a  or  (flag_b  and  flag_c)")
        dsl.define("or_first", "(flag_a  or  flag_b)  and  flag_c")
        rdf = dsl.apply(test_rdf)
        
        # Standard should match and_first (correct precedence)
        assert_no_violations(
            rdf,
            "standard != and_first",
            "a  or  b  and  c == a  or  (b  and  c)"
        )


# =============================================================================
# BOOLEAN INVARIANT TESTS
# =============================================================================

class TestBooleanInvariants:
    """Test boolean expression invariants via DSL."""
    
    def test_demorgan_and(self, dsl, test_rdf):
        """Invariant: !(A  and  B) == (!A  or  !B) (De Morgan's law)"""
        dsl.define("not_and", "not (flag_a and flag_b)")
        dsl.define("or_nots", "(not flag_a) or (not flag_b)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "not_and != or_nots",
            "De Morgan: !(A  and  B) == !A  or  !B"
        )
    
    def test_demorgan_or(self, dsl, test_rdf):
        """Invariant: !(A  or  B) == (!A  and  !B) (De Morgan's law)"""
        dsl.define("not_or", "not (flag_a or flag_b)")
        dsl.define("and_nots", "(not flag_a) and (not flag_b)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "not_or != and_nots",
            "De Morgan: !(A  or  B) == !A  and  !B"
        )
    
    def test_bool_double_negation(self, dsl, test_rdf):
        """Invariant: !!A == A"""
        dsl.define("double_neg", "not (not flag_a)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "double_neg != flag_a",
            "!!A == A"
        )
    
    def test_and_with_true(self, dsl, test_rdf):
        """Invariant: A  and  true == A"""
        dsl.define("and_true", "flag_a and True")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "and_true != flag_a",
            "A  and  true == A"
        )
    
    def test_or_with_false(self, dsl, test_rdf):
        """Invariant: A  or  false == A"""
        dsl.define("or_false", "flag_a or False")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "or_false != flag_a",
            "A  or  false == A"
        )


# =============================================================================
# COMPARISON INVARIANT TESTS
# =============================================================================

class TestComparisonInvariants:
    """Test comparison expression invariants via DSL."""
    
    def test_greater_lessequal_inverse(self, dsl, test_rdf):
        """Invariant: (A > B) == not (A <= B)"""
        dsl.define("gt", "A > B")
        dsl.define("not_le", "not (A <= B)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "gt != not_le",
            "(A > B) == not (A <= B)"
        )
    
    def test_less_greaterequal_inverse(self, dsl, test_rdf):
        """Invariant: (A < B) == not (A >= B)"""
        dsl.define("lt", "A < B")
        dsl.define("not_ge", "not (A >= B)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "lt != not_ge",
            "(A < B) == not (A >= B)"
        )
    
    def test_equal_notequal_inverse(self, dsl, test_rdf):
        """Invariant: (A == B) == not (A != B)"""
        dsl.define("eq", "A == B")
        dsl.define("not_ne", "not (A != B)")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "eq != not_ne",
            "(A == B) == not (A != B)"
        )
    
    def test_comparison_transitivity(self, dsl, test_rdf):
        """Invariant: (A > B)  and  (B > C) implies (A > C)"""
        dsl.define("premise", "(A > B)  and  (B > C)")
        dsl.define("conclusion", "A > C")
        rdf = dsl.apply(test_rdf)
        
        # If premise is true, conclusion must be true
        # Violation: premise  and  !conclusion
        assert_no_violations(
            rdf,
            "premise  and  (!conclusion)",
            "Transitivity: (A > B)  and  (B > C) => (A > C)"
        )


# =============================================================================
# RVEC ELEMENT-WISE INVARIANT TESTS
# =============================================================================

class TestRVecElementWiseInvariants:
    """Test RVec element-wise operation invariants via DSL."""
    
    def test_vec_addition(self, dsl, test_rdf):
        """Invariant: vec_a + vec_b == vec_sum (element-wise)"""
        dsl.define("computed_sum", "vec_a + vec_b")
        rdf = dsl.apply(test_rdf)
        
        # Compare using Sum of absolute differences
        assert_no_violations(
            rdf,
            "Sum(abs(computed_sum - vec_sum)) > 1e-10",
            "vec_a + vec_b == vec_sum"
        )
    
    def test_vec_scalar_multiply(self, dsl, test_rdf):
        """Invariant: vec_i * 2 == vec_i_doubled"""
        dsl.define("computed_doubled", "vec_i * 2")
        rdf = dsl.apply(test_rdf)
        
        # Integer comparison should be exact
        assert_no_violations(
            rdf,
            "Sum(abs(computed_doubled - vec_i_doubled)) > 0",
            "vec_i * 2 == vec_i_doubled"
        )
    
    def test_length_preserved_after_scalar_multiply(self, dsl, test_rdf):
        """Invariant: Size(vec_a * 2) == Size(vec_a)"""
        dsl.define("scaled", "vec_a * 2.0")
        dsl.define("len_original", "vec_a.size()")
        dsl.define("len_scaled", "scaled.size()")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "len_original != len_scaled",
            "Size(vec_a * 2) == Size(vec_a)"
        )
    
    def test_length_preserved_after_addition(self, dsl, test_rdf):
        """Invariant: Size(vec_a + vec_b) == Size(vec_a)"""
        dsl.define("summed", "vec_a + vec_b")
        dsl.define("len_a", "vec_a.size()")
        dsl.define("len_summed", "summed.size()")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "len_a != len_summed",
            "Size(vec_a + vec_b) == Size(vec_a)"
        )


# =============================================================================
# RVEC REDUCTION INVARIANT TESTS
# =============================================================================

class TestRVecReductionInvariants:
    """Test RVec reduction operation invariants via DSL."""
    
    def test_sum_distributive(self, dsl, test_rdf):
        """Invariant: Sum(vec_a) + Sum(vec_b) == Sum(vec_a + vec_b)"""
        dsl.define("sum_a", "Sum(vec_a)")
        dsl.define("sum_b", "Sum(vec_b)")
        dsl.define("sum_of_sums", "sum_a + sum_b")
        dsl.define("sum_of_sum_vec", "Sum(vec_sum)")
        rdf = dsl.apply(test_rdf)
        
        # Use aggregation tolerance
        assert_no_violations(
            rdf,
            "abs(sum_of_sums - sum_of_sum_vec) > 1e-10",
            "Sum(A) + Sum(B) == Sum(A + B)"
        )
    
    def test_sum_scalar_factor(self, dsl, test_rdf):
        """Invariant: Sum(vec_a * k) == k * Sum(vec_a)"""
        k = 3.5
        dsl.define("scaled_vec", f"vec_a * {k}")
        dsl.define("sum_scaled", "Sum(scaled_vec)")
        dsl.define("scaled_sum", f"Sum(vec_a) * {k}")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(sum_scaled - scaled_sum) > 1e-10",
            "Sum(k * A) == k * Sum(A)"
        )
    
    def test_mean_definition(self, dsl, test_rdf):
        """Invariant: Mean(vec_a) == Sum(vec_a) / Size(vec_a) (for non-empty)"""
        dsl.define("mean_val", "Mean(vec_a)")
        dsl.define("manual_mean", "Sum(vec_a) / vec_a.size()")
        dsl.define("is_nonempty", "vec_a.size() > 0")
        rdf = dsl.apply(test_rdf)
        
        # Only check non-empty vectors
        assert_no_violations(
            rdf.Filter("is_nonempty"),
            "abs(mean_val - manual_mean) > 1e-10",
            "Mean(A) == Sum(A) / Size(A)"
        )
    
    def test_min_max_bounds(self, dsl, test_rdf):
        """Invariant: Min(vec_a) <= Mean(vec_a) <= Max(vec_a) (for non-empty)"""
        dsl.define("min_val", "Min(vec_a)")
        dsl.define("max_val", "Max(vec_a)")
        dsl.define("mean_val", "Mean(vec_a)")
        dsl.define("is_nonempty", "vec_a.size() > 0")
        rdf = dsl.apply(test_rdf)
        
        # Min <= Mean
        assert_no_violations(
            rdf.Filter("is_nonempty"),
            "min_val > mean_val + 1e-12",
            "Min(A) <= Mean(A)"
        )
        
        # Mean <= Max
        assert_no_violations(
            rdf.Filter("is_nonempty"),
            "mean_val > max_val + 1e-12",
            "Mean(A) <= Max(A)"
        )


# =============================================================================
# RVEC EDGE CASE TESTS
# =============================================================================

class TestRVecEdgeCases:
    """Test RVec edge cases: empty, singleton, variable-length."""
    
    def test_empty_vector_size(self, dsl, test_rdf):
        """Empty vectors have size 0."""
        dsl.define("is_empty", "vec_a.size() == 0")
        rdf = dsl.apply(test_rdf)
        
        # Should have some empty vectors (from RVEC_LENGTH_PATTERN)
        empty_count = rdf.Filter("is_empty").Count().GetValue()
        assert empty_count > 0, "Test data should include empty vectors"
    
    def test_singleton_vector(self, dsl, test_rdf):
        """Singleton vectors have size 1."""
        dsl.define("is_singleton", "vec_a.size() == 1")
        rdf = dsl.apply(test_rdf)
        
        singleton_count = rdf.Filter("is_singleton").Count().GetValue()
        assert singleton_count > 0, "Test data should include singleton vectors"
    
    def test_variable_length_coverage(self, dsl, test_rdf):
        """Vectors have variable lengths across events."""
        dsl.define("vec_size", "vec_a.size()")
        rdf = dsl.apply(test_rdf)
        
        # Get unique lengths
        sizes = rdf.Take["unsigned long"]("vec_size").GetValue()
        unique_sizes = set(sizes)
        
        # Should have at least 5 different lengths
        assert len(unique_sizes) >= 5, (
            f"Only {len(unique_sizes)} unique vector lengths - "
            "expected more variety"
        )
    
    def test_sum_empty_vector(self, dsl, test_rdf):
        """Sum of empty vector is 0."""
        dsl.define("sum_vec", "Sum(vec_a)")
        dsl.define("is_empty", "vec_a.size() == 0")
        rdf = dsl.apply(test_rdf)
        
        # For empty vectors, Sum should be 0
        assert_no_violations(
            rdf.Filter("is_empty"),
            "abs(sum_vec) > 1e-14",
            "Sum(empty) == 0"
        )


# =============================================================================
# C-ARRAY / LEGACY BRANCH TESTS
# =============================================================================

class TestCArrayInvariants:
    """Test C-array (legacy TTree) access invariants via DSL."""
    
    @pytest.mark.skip(reason="C-array branch access deferred to V2 scope")
    def test_index_access(self, dsl, test_rdf):
        """Invariant: arr_d[idx] == picked (pre-computed in generator)"""
        # Note: arr_d is a C-array branch, accessed via raw TTree
        # The DSL should support this pattern
        dsl.define("accessed", "arr_d[idx]")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(accessed - picked) > 1e-12",
            "arr_d[idx] == picked"
        )
    
    def test_index_bounds(self, dsl, test_rdf):
        """Index is always within bounds."""
        # idx should be < n_arr
        rdf = test_rdf  # Direct check without DSL
        
        assert_no_violations(
            rdf,
            "idx >= n_arr  or  idx < 0",
            "0 <= idx < n_arr"
        )
    
    @pytest.mark.skip(reason="C-array branch access deferred to V2 scope")
    def test_carray_element_invariant(self, dsl, test_rdf):
        """Invariant: arr_sum[0] == arr_d[0] + 1.0"""
        dsl.define("first_d", "arr_d[0]")
        dsl.define("first_sum", "arr_sum[0]")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "abs(first_sum - (first_d + 1.0)) > 1e-12",
            "arr_sum[i] == arr_d[i] + 1.0"
        )


# =============================================================================
# INTEGER TYPE TESTS
# =============================================================================

class TestIntegerInvariants:
    """Test integer arithmetic preserves exact values."""
    
    def test_int_addition_exact(self, dsl, test_rdf):
        """Invariant: Ai + Bi == sum_i (exact)"""
        dsl.define("computed_sum", "Ai + Bi")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "computed_sum != sum_i",
            "Ai + Bi == sum_i (exact)"
        )
    
    def test_uint_addition_exact(self, dsl, test_rdf):
        """Invariant: Au + Bu == sum_u (exact)"""
        dsl.define("computed_sum", "Au + Bu")
        rdf = dsl.apply(test_rdf)
        
        assert_no_violations(
            rdf,
            "computed_sum != sum_u",
            "Au + Bu == sum_u (exact)"
        )


# =============================================================================
# FLOAT TYPE TESTS (Tolerance)
# =============================================================================

class TestFloatInvariants:
    """Test float arithmetic with appropriate tolerance."""
    
    def test_float_addition(self, dsl, test_rdf):
        """Invariant: Af + Bf == sum_f (float tolerance)"""
        dsl.define("computed_sum", "Af + Bf")
        rdf = dsl.apply(test_rdf)
        
        # Float needs 1e-6 tolerance
        assert_no_violations(
            rdf,
            "abs(computed_sum - sum_f) > 1e-5",
            "Af + Bf == sum_f (float)"
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
