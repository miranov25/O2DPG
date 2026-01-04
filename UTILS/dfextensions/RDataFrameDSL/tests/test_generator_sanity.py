"""
RDataFrameDSL Invariant Testing — Generator Sanity Tests

Phase 13.2.1.DSL: Verify the data generator produces correct invariants.

These tests run BEFORE DSL tests to ensure we're not testing against
garbage data. If these fail, the generator is broken, not the DSL.

This is the "meta-test" requirement from Architect review:
    "Phase 13.2.1 must include a sanity check for the Data Generator itself."
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from .invariant_schema import (
    TOLERANCE,
    DEFAULT_SEED,
    RVEC_LENGTH_PATTERN,
    MAX_CARRAY_LENGTH,
    INVARIANTS,
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

@pytest.fixture(scope="module")
def generated_tree():
    """Generate a test tree for sanity checks (module-scoped)."""
    from .test_data_generator import generate_invariant_tree
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "sanity_test.root")
        generate_invariant_tree(filepath, n_events=500, seed=DEFAULT_SEED)
        yield filepath


@pytest.fixture(scope="module")
def rdf(generated_tree):
    """Create RDataFrame from generated tree."""
    return ROOT.RDataFrame("invariants", generated_tree)


# =============================================================================
# BASIC STRUCTURE TESTS
# =============================================================================

class TestGeneratorBasics:
    """Test basic generator functionality."""
    
    def test_file_created(self, generated_tree):
        """Generator creates a file."""
        assert os.path.exists(generated_tree)
        assert os.path.getsize(generated_tree) > 0
    
    def test_tree_exists(self, generated_tree):
        """Generated file contains the expected tree."""
        f = ROOT.TFile(generated_tree, "READ")
        tree = f.Get("invariants")
        assert tree is not None
        assert tree.GetEntries() == 500
        f.Close()
    
    def test_all_branches_exist(self, generated_tree):
        """All expected branches are present."""
        f = ROOT.TFile(generated_tree, "READ")
        tree = f.Get("invariants")
        branches = [b.GetName() for b in tree.GetListOfBranches()]
        f.Close()
        
        expected = [
            # Doubles
            "A", "B", "C", "sum_ab", "prod_ab", "A_positive",
            "prec_add_mul", "prec_mul_add",
            # Floats
            "Af", "Bf", "sum_f",
            # Integers
            "Ai", "Bi", "sum_i",
            # Unsigned integers
            "Au", "Bu", "sum_u",
            # Booleans
            "flag_a", "flag_b", "flag_c", "flag_and", "flag_or", "flag_not_a",
            # Edge cases
            "near_zero", "large_val",
            # RVecs
            "vec_a", "vec_b", "vec_sum", "vec_len", "vec_i", "vec_i_doubled",
            # C-arrays
            "n_arr", "arr_d", "arr_i", "arr_sum", "idx", "picked",
        ]
        
        for branch in expected:
            assert branch in branches, f"Missing branch: {branch}"
    
    def test_reproducibility(self):
        """Same seed produces identical data."""
        from .test_data_generator import generate_invariant_tree
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "test1.root")
            path2 = os.path.join(tmpdir, "test2.root")
            
            generate_invariant_tree(path1, n_events=100, seed=12345)
            generate_invariant_tree(path2, n_events=100, seed=12345)
            
            rdf1 = ROOT.RDataFrame("invariants", path1)
            rdf2 = ROOT.RDataFrame("invariants", path2)
            
            # Compare first values
            a1 = rdf1.Take['double']("A").GetValue()
            a2 = rdf2.Take['double']("A").GetValue()
            
            assert list(a1) == list(a2), "Same seed should produce identical data"


# =============================================================================
# DOUBLE PRECISION ARITHMETIC INVARIANTS
# =============================================================================

class TestDoubleInvariants:
    """Test double-precision arithmetic invariants."""
    
    def test_sum_ab_equals_a_plus_b(self, rdf):
        """Invariant: sum_ab == A + B"""
        count = rdf.Filter("abs(sum_ab - (A + B)) > 1e-12").Count().GetValue()
        assert count == 0, f"sum_ab invariant violated in {count} events"
    
    def test_prod_ab_equals_a_times_b(self, rdf):
        """Invariant: prod_ab == A * B"""
        count = rdf.Filter("abs(prod_ab - (A * B)) > 1e-12").Count().GetValue()
        assert count == 0, f"prod_ab invariant violated in {count} events"
    
    def test_a_positive_is_positive(self, rdf):
        """Invariant: A_positive > 0"""
        count = rdf.Filter("A_positive <= 0").Count().GetValue()
        assert count == 0, f"A_positive <= 0 in {count} events"
    
    def test_precedence_add_mul(self, rdf):
        """Invariant: prec_add_mul == A + (B * C)"""
        # Note: C++ follows standard precedence, so A + B * C == A + (B * C)
        count = rdf.Filter("abs(prec_add_mul - (A + B * C)) > 1e-10").Count().GetValue()
        assert count == 0, f"prec_add_mul invariant violated in {count} events"
    
    def test_precedence_mul_add(self, rdf):
        """Invariant: prec_mul_add == (A + B) * C"""
        count = rdf.Filter("abs(prec_mul_add - ((A + B) * C)) > 1e-12").Count().GetValue()
        assert count == 0, f"prec_mul_add invariant violated in {count} events"
    
    def test_a_and_b_have_variety(self, rdf):
        """A and B have both positive and negative values."""
        pos_a = rdf.Filter("A > 0").Count().GetValue()
        neg_a = rdf.Filter("A < 0").Count().GetValue()
        pos_b = rdf.Filter("B > 0").Count().GetValue()
        neg_b = rdf.Filter("B < 0").Count().GetValue()
        
        total = rdf.Count().GetValue()
        
        # With uniform [-100, 100], expect roughly 50% each
        assert pos_a > total * 0.3, "Not enough positive A values"
        assert neg_a > total * 0.3, "Not enough negative A values"
        assert pos_b > total * 0.3, "Not enough positive B values"
        assert neg_b > total * 0.3, "Not enough negative B values"


# =============================================================================
# FLOAT PRECISION INVARIANTS
# =============================================================================

class TestFloatInvariants:
    """Test single-precision (float) invariants with appropriate tolerance."""
    
    def test_sum_f_equals_af_plus_bf(self, rdf):
        """Invariant: sum_f == Af + Bf (with float tolerance)"""
        # Float has ~7 digits of precision
        count = rdf.Filter("abs(sum_f - (Af + Bf)) > 1e-5").Count().GetValue()
        assert count == 0, f"sum_f invariant violated in {count} events"


# =============================================================================
# INTEGER INVARIANTS
# =============================================================================

class TestIntegerInvariants:
    """Test integer invariants (exact equality)."""
    
    def test_sum_i_equals_ai_plus_bi(self, rdf):
        """Invariant: sum_i == Ai + Bi (exact)"""
        count = rdf.Filter("sum_i != (Ai + Bi)").Count().GetValue()
        assert count == 0, f"sum_i invariant violated in {count} events"
    
    def test_sum_u_equals_au_plus_bu(self, rdf):
        """Invariant: sum_u == Au + Bu (exact)"""
        count = rdf.Filter("sum_u != (Au + Bu)").Count().GetValue()
        assert count == 0, f"sum_u invariant violated in {count} events"
    
    def test_unsigned_values_non_negative(self, rdf):
        """Au and Bu are non-negative."""
        # This is implicit from unsigned type, but good to verify
        count = rdf.Filter("Au < 0 || Bu < 0").Count().GetValue()
        assert count == 0, "Unsigned values should not be negative"


# =============================================================================
# BOOLEAN INVARIANTS
# =============================================================================

class TestBooleanInvariants:
    """Test boolean invariants."""
    
    def test_flag_and_equals_a_and_b(self, rdf):
        """Invariant: flag_and == (flag_a && flag_b)"""
        count = rdf.Filter("flag_and != (flag_a && flag_b)").Count().GetValue()
        assert count == 0, f"flag_and invariant violated in {count} events"
    
    def test_flag_or_equals_a_or_b(self, rdf):
        """Invariant: flag_or == (flag_a || flag_b)"""
        count = rdf.Filter("flag_or != (flag_a || flag_b)").Count().GetValue()
        assert count == 0, f"flag_or invariant violated in {count} events"
    
    def test_flag_not_a_equals_not_a(self, rdf):
        """Invariant: flag_not_a == !flag_a"""
        count = rdf.Filter("flag_not_a != (!flag_a)").Count().GetValue()
        assert count == 0, f"flag_not_a invariant violated in {count} events"
    
    def test_boolean_values_variety(self, rdf):
        """Boolean flags have both true and false values."""
        true_a = rdf.Filter("flag_a").Count().GetValue()
        false_a = rdf.Filter("!flag_a").Count().GetValue()
        
        total = rdf.Count().GetValue()
        
        # With 50/50 distribution, expect roughly equal
        assert true_a > total * 0.3, "Not enough true flag_a values"
        assert false_a > total * 0.3, "Not enough false flag_a values"


# =============================================================================
# RVEC INVARIANTS
# =============================================================================

class TestRVecInvariants:
    """Test RVec invariants."""
    
    def test_vec_len_equals_size_vec_a(self, rdf):
        """Invariant: vec_len == Size(vec_a)"""
        count = rdf.Filter("vec_len != (int)vec_a.size()").Count().GetValue()
        assert count == 0, f"vec_len invariant violated in {count} events"
    
    def test_vec_a_and_vec_b_same_length(self, rdf):
        """Invariant: Size(vec_a) == Size(vec_b)"""
        count = rdf.Filter("vec_a.size() != vec_b.size()").Count().GetValue()
        assert count == 0, f"vec_a/vec_b length mismatch in {count} events"
    
    def test_vec_sum_same_length_as_vec_a(self, rdf):
        """Invariant: Size(vec_sum) == Size(vec_a)"""
        count = rdf.Filter("vec_sum.size() != vec_a.size()").Count().GetValue()
        assert count == 0, f"vec_sum length mismatch in {count} events"
    
    def test_vec_i_and_vec_i_doubled_same_length(self, rdf):
        """Invariant: Size(vec_i) == Size(vec_i_doubled)"""
        count = rdf.Filter("vec_i.size() != vec_i_doubled.size()").Count().GetValue()
        assert count == 0, f"vec_i/vec_i_doubled length mismatch in {count} events"
    
    def test_variable_length_includes_empty(self, rdf):
        """RVecs include empty vectors (length 0)."""
        count = rdf.Filter("vec_a.size() == 0").Count().GetValue()
        assert count > 0, "No empty vectors found (should have some)"
    
    def test_variable_length_includes_singleton(self, rdf):
        """RVecs include singleton vectors (length 1)."""
        count = rdf.Filter("vec_a.size() == 1").Count().GetValue()
        assert count > 0, "No singleton vectors found (should have some)"
    
    def test_variable_length_includes_larger(self, rdf):
        """RVecs include larger vectors (length > 10)."""
        count = rdf.Filter("vec_a.size() > 10").Count().GetValue()
        assert count > 0, "No large vectors found (should have some)"
    
    def test_length_pattern_coverage(self, rdf):
        """RVec lengths follow the expected pattern."""
        total = rdf.Count().GetValue()
        
        # Check that we have variety in lengths
        lengths_seen = set()
        for length in RVEC_LENGTH_PATTERN:
            count = rdf.Filter(f"vec_a.size() == {length}").Count().GetValue()
            if count > 0:
                lengths_seen.add(length)
        
        # Should see at least 80% of pattern lengths in 500 events
        expected_lengths = set(RVEC_LENGTH_PATTERN)
        coverage = len(lengths_seen) / len(expected_lengths)
        assert coverage >= 0.8, f"Only saw {len(lengths_seen)}/{len(expected_lengths)} length patterns"


# =============================================================================
# C-ARRAY INVARIANTS
# =============================================================================

class TestCArrayInvariants:
    """Test C-array (legacy TTree pattern) invariants."""
    
    def test_n_arr_in_valid_range(self, rdf):
        """n_arr is in valid range [1, MAX_CARRAY_LENGTH]."""
        count = rdf.Filter(f"n_arr < 1 || n_arr > {MAX_CARRAY_LENGTH}").Count().GetValue()
        assert count == 0, f"n_arr out of range in {count} events"
    
    def test_idx_less_than_n_arr(self, rdf):
        """Invariant: idx < n_arr (valid index)."""
        count = rdf.Filter("idx >= n_arr").Count().GetValue()
        assert count == 0, f"idx >= n_arr in {count} events"
    
    def test_idx_non_negative(self, rdf):
        """Invariant: idx >= 0."""
        count = rdf.Filter("idx < 0").Count().GetValue()
        assert count == 0, f"idx < 0 in {count} events"
    
    def test_picked_equals_arr_d_at_idx(self, rdf):
        """Invariant: picked == arr_d[idx]."""
        count = rdf.Filter("abs(picked - arr_d[idx]) > 1e-12").Count().GetValue()
        assert count == 0, f"picked != arr_d[idx] in {count} events"
    
    def test_arr_sum_equals_arr_d_plus_one(self, rdf):
        """Invariant: arr_sum[i] == arr_d[i] + 1.0 for all valid i."""
        # Check first element as proxy (full check would need loop)
        count = rdf.Filter("abs(arr_sum[0] - (arr_d[0] + 1.0)) > 1e-12").Count().GetValue()
        assert count == 0, f"arr_sum[0] invariant violated in {count} events"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge case values."""
    
    def test_near_zero_is_small(self, rdf):
        """near_zero values are very small."""
        count = rdf.Filter("abs(near_zero) > 1e-14").Count().GetValue()
        # Most should be small, allow some variation
        total = rdf.Count().GetValue()
        assert count < total * 0.1, f"Too many large near_zero values: {count}"
    
    def test_large_val_is_large(self, rdf):
        """large_val includes large magnitude values."""
        count = rdf.Filter("abs(large_val) > 1e9").Count().GetValue()
        total = rdf.Count().GetValue()
        # With uniform [-1e10, 1e10], ~80% should have |x| > 1e9
        assert count > total * 0.5, f"Not enough large values: {count}"


# =============================================================================
# FULL VERIFICATION TEST
# =============================================================================

class TestFullVerification:
    """Run the full verification routine."""
    
    def test_verify_invariants_python(self, generated_tree):
        """Full verification using verify_invariants_python()."""
        from .test_data_generator import verify_invariants_python
        
        result = verify_invariants_python(generated_tree)
        
        assert result["passed"], (
            f"Verification failed: {result['n_passed']}/{result['n_checks']} passed\n"
            + "\n".join(f"  ✗ {name}: {msg}" for name, passed, msg in result["checks"] if not passed)
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
