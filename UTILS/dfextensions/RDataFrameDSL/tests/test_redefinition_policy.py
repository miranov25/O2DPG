# =============================================================================
# tests/test_redefinition_policy.py
# =============================================================================
# Tests for redefinition parameter - Phase 13.6.G+
#
# Date: 2026-01-26
#
# Tests verify:
# - Redefinition policy handling (error, allow, skip, ifneeded)
# - Physical column guard
# - Expression normalization
# - Apply with Define/Redefine
# =============================================================================

import pytest

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError

# File-level marker: ALL tests in this file run serially (ROOT dependency)
pytestmark = pytest.mark.root_serial


# =============================================================================
# Test Class: Basic Policy Tests (Pure Python)
# =============================================================================

class TestRedefinitionPolicyBasic:
    """Basic tests for redefinition parameter - no RDataFrame needed."""
    
    @pytest.mark.feature("redefinition_policy")
    def test_default_policy_is_error(self):
        """Default redefinition policy is 'error'."""
        dsl = DSLCompiler({'px': 'double', 'py': 'double'})
        assert dsl._redefinition == "error"
    
    @pytest.mark.feature("redefinition_policy")
    def test_invalid_policy_raises(self):
        """Invalid redefinition policy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DSLCompiler({'px': 'double'}, redefinition="invalid")
        
        assert "redefinition" in str(exc_info.value).lower()
        assert "invalid" in str(exc_info.value).lower()
    
    @pytest.mark.feature("redefinition_policy")
    def test_error_policy_raises_on_duplicate(self):
        """error: duplicate define raises IRError."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="error")
        dsl.define("pt", "px * 2")
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("pt", "px * 3")
        
        assert "already defined" in str(exc_info.value).lower()
    
    @pytest.mark.feature("redefinition_policy")
    def test_skip_policy_keeps_original(self):
        """skip: duplicate define keeps original expression."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="skip")
        dsl.define("pt", "px * 2")
        dsl.define("pt", "px * 3")  # Should be ignored
        
        # Original expression should be preserved
        assert dsl._expressions_raw["pt"] == "px * 2"
        assert len(dsl._definitions) == 1
    
    @pytest.mark.feature("redefinition_policy")
    def test_allow_policy_always_replaces(self):
        """allow: duplicate define always replaces."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="allow")
        dsl.define("pt", "px * 2")
        dsl.define("pt", "px * 3")  # Should replace
        
        assert dsl._expressions_raw["pt"] == "px * 3"
        assert len(dsl._definitions) == 1
    
    @pytest.mark.feature("redefinition_policy")
    def test_ifneeded_same_expr_skips(self):
        """ifneeded: same expression skips redefinition."""
        dsl = DSLCompiler({'px': 'double', 'py': 'double'}, redefinition="ifneeded")
        
        # First define
        dsl.define("pt", "sqrt(px**2 + py**2)")
        assert len(dsl._definitions) == 1
        assert dsl._dirty["pt"] == True  # Set by first define
        
        # Second define (same expr) should skip
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        # Verify skip: definition unchanged, only one definition
        assert len(dsl._definitions) == 1
        assert dsl._expressions_norm["pt"] == "sqrt(px**2 + py**2)"
        
        # dirty flag remains True (only cleared by apply)
        assert dsl._dirty["pt"] == True
    
    @pytest.mark.feature("redefinition_policy")
    def test_ifneeded_same_expr_whitespace_skips(self):
        """ifneeded: same expression with different whitespace skips."""
        dsl = DSLCompiler({'px': 'double', 'py': 'double'}, redefinition="ifneeded")
        dsl.define("pt", "sqrt(px**2  +   py**2)")
        
        # Manually clear dirty to simulate after apply
        dsl._dirty["pt"] = False
        
        # Same expression, different whitespace - should skip
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        # dirty should NOT be set (expression unchanged after normalization)
        assert dsl._dirty.get("pt", False) == False
    
    @pytest.mark.feature("redefinition_policy")
    def test_ifneeded_different_expr_marks_dirty(self):
        """ifneeded: different expression marks dirty for redefine."""
        dsl = DSLCompiler({'px': 'double', 'py': 'double'}, redefinition="ifneeded")
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        # Simulate after apply
        dsl._dirty["pt"] = False
        
        # Different expression - should mark dirty
        dsl.define("pt", "sqrt(px**2 + py**2) * 1.1")
        
        assert dsl._dirty["pt"] == True
        assert dsl._expressions_raw["pt"] == "sqrt(px**2 + py**2) * 1.1"
    
    @pytest.mark.feature("redefinition_policy")
    def test_physical_column_guard(self):
        """Cannot redefine physical schema columns regardless of policy."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="ifneeded")
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("px", "px * 2")
        
        assert "physical column" in str(exc_info.value).lower()
    
    @pytest.mark.feature("redefinition_policy")
    def test_physical_column_guard_with_allow(self):
        """Physical column guard applies even with redefinition='allow'."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="allow")
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("px", "px * 2")
        
        assert "physical column" in str(exc_info.value).lower()
    
    @pytest.mark.feature("redefinition_policy")
    def test_alias_stays_strict_with_ifneeded(self):
        """alias() ignores redefinition policy - always strict."""
        dsl = DSLCompiler({'track_pt': 'double'}, redefinition="ifneeded")
        dsl.alias("high_pt", "track_pt > 1.0")
        
        with pytest.raises(IRError):
            dsl.alias("high_pt", "track_pt > 2.0")  # Must error


# =============================================================================
# Test Class: Integration Tests with RDataFrame
# =============================================================================

class TestRedefinitionIntegration:
    """Integration tests with RDataFrame - Phase 13.6.G+"""
    
    @pytest.mark.feature("redefinition_policy")
    def test_apply_defines_new_columns(self, nd_2d_rdf, nd_2d_schema):
        """apply() uses Define() for new columns."""
        dsl = DSLCompiler(nd_2d_schema, redefinition="ifneeded")
        dsl.define("double_weight", "event_weight * 2")
        
        rdf = dsl.apply(nd_2d_rdf)
        
        assert "double_weight" in list(rdf.GetColumnNames())
        assert dsl._dirty.get("double_weight", True) == False  # Cleared after apply
    
    @pytest.mark.feature("redefinition_policy")
    def test_apply_skips_unchanged(self, nd_2d_rdf, nd_2d_schema):
        """apply() skips unchanged columns on second call."""
        dsl = DSLCompiler(nd_2d_schema, redefinition="ifneeded")
        dsl.define("double_weight", "event_weight * 2")
        
        # First apply
        rdf1 = dsl.apply(nd_2d_rdf)
        assert "double_weight" in list(rdf1.GetColumnNames())
        
        # Second apply (no change) - should not crash
        rdf2 = dsl.apply(rdf1)
        assert "double_weight" in list(rdf2.GetColumnNames())
    
    @pytest.mark.feature("redefinition_policy")
    def test_apply_redefines_changed(self, nd_2d_rdf, nd_2d_schema):
        """apply() uses Redefine() for changed expressions."""
        dsl = DSLCompiler(nd_2d_schema, redefinition="ifneeded")
        
        # First definition and apply
        dsl.define("double_weight", "event_weight * 2")
        rdf1 = dsl.apply(nd_2d_rdf)
        
        # Change expression
        dsl.define("double_weight", "event_weight * 3")
        
        # Second apply should use Redefine (not crash)
        rdf2 = dsl.apply(rdf1)
        
        # Verify column exists and dirty cleared
        assert "double_weight" in list(rdf2.GetColumnNames())
        assert dsl._dirty.get("double_weight", True) == False
    
    @pytest.mark.feature("redefinition_policy")
    def test_draw_after_apply_works_with_ifneeded(self, nd_2d_rdf, nd_2d_schema):
        """draw() after apply() works with ifneeded policy."""
        dfdraw = pytest.importorskip("dfdraw")
        
        dsl = DSLCompiler(nd_2d_schema, redefinition="ifneeded")
        dsl.define("double_weight", "event_weight * 2")
        
        # Apply then draw (draw internally calls apply again)
        rdf = dsl.apply(nd_2d_rdf)
        fig, ax, stats = dsl.draw("double_weight", rdf)  # Should not crash
        
        assert fig is not None


# =============================================================================
# Test Class: Expression Normalization
# =============================================================================

class TestExpressionNormalization:
    """Tests for expression normalization - Phase 13.6.G+"""
    
    @pytest.mark.feature("redefinition_policy")
    def test_normalize_collapses_whitespace(self):
        """Normalization collapses internal whitespace."""
        dsl = DSLCompiler({'px': 'double'})
        
        assert dsl._normalize_expr("px  *   2") == "px * 2"
        assert dsl._normalize_expr("  px * 2  ") == "px * 2"
        assert dsl._normalize_expr("px*2") == "px*2"  # No space, stays same
    
    @pytest.mark.feature("redefinition_policy")
    def test_normalize_handles_multiline(self):
        """Normalization handles multiline expressions."""
        dsl = DSLCompiler({'px': 'double'})
        
        expr = """px
        *
        2"""
        assert dsl._normalize_expr(expr) == "px * 2"
    
    @pytest.mark.feature("redefinition_policy")
    def test_normalized_comparison_in_ifneeded(self):
        """ifneeded uses normalized comparison."""
        dsl = DSLCompiler({'px': 'double'}, redefinition="ifneeded")
        
        dsl.define("pt", "px  *  2")
        initial_count = len(dsl._definitions)
        
        # Same after normalization - should skip
        dsl.define("pt", "px * 2")
        
        assert len(dsl._definitions) == initial_count
