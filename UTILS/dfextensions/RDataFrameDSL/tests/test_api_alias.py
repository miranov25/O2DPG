# =============================================================================
# tests/test_api_alias.py
# =============================================================================
# API tests for alias() method - Pool-based deferred validation.
#
# Phase: 13.6.F
# Date: 2026-01-24
#
# Tests verify:
# - alias() stores expressions without validation
# - Pool-based: only needed aliases compiled
# - Dependency resolution with cycle detection
# - from_rdf() and update_schema_from_rdf()
# =============================================================================

import pytest
import ast

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError, IRErrorKind

# File-level marker: ALL tests in this file run serially (ROOT dependency)
pytestmark = pytest.mark.root_serial


# =============================================================================
# Test Class: Basic alias() functionality
# =============================================================================

class TestAliasBasic:
    """Basic tests for alias() method."""
    
    @pytest.mark.feature("api_alias")
    def test_alias_basic_store(self):
        """
        alias() stores expression without validation.
        
        Phase 13.6.F: Deferred validation - no schema check at alias() time.
        """
        dsl = DSLCompiler()  # Empty schema
        
        # Should NOT raise - just stores
        dsl.alias("pt", "sqrt(px**2 + py**2)")
        
        # Verify stored in pool
        assert "pt" in dsl._aliases
        assert dsl._aliases["pt"] == "sqrt(px**2 + py**2)"
        
        # NOT in schema yet
        assert "pt" not in dsl.schema
    
    @pytest.mark.feature("api_alias")
    def test_alias_chaining(self):
        """
        alias() returns self for method chaining.
        """
        dsl = DSLCompiler()
        
        result = dsl.alias("a", "x + 1").alias("b", "a + 2").alias("c", "b + 3")
        
        assert result is dsl
        assert len(dsl._aliases) == 3
    
    @pytest.mark.feature("api_alias")
    def test_alias_conflict_with_schema(self):
        """
        alias() raises error if name conflicts with schema column.
        """
        dsl = DSLCompiler({'px': 'double'})
        
        with pytest.raises(IRError) as exc_info:
            dsl.alias("px", "py + 1")  # Conflicts with schema
        
        assert "conflicts with existing column" in str(exc_info.value)
    
    @pytest.mark.feature("api_alias")
    def test_alias_conflict_with_alias(self):
        """
        alias() raises error if name already defined as alias.
        """
        dsl = DSLCompiler()
        dsl.alias("pt", "sqrt(px**2 + py**2)")
        
        with pytest.raises(IRError) as exc_info:
            dsl.alias("pt", "different_expression")  # Already defined
        
        assert "already defined" in str(exc_info.value)


# =============================================================================
# Test Class: Pool-based compilation
# =============================================================================

class TestAliasPoolBased:
    """Tests for pool-based compilation model."""
    
    @pytest.mark.feature("api_alias")
    def test_alias_unused_ignored(self):
        """
        Unused aliases are never validated.
        
        Phase 13.6.F: Pool-based - only compile what's needed.
        """
        dsl = DSLCompiler({'px': 'double', 'py': 'double'})
        
        # Define valid alias
        dsl.alias("pt", "sqrt(px**2 + py**2)")
        
        # Define INVALID alias (references nonexistent column)
        dsl.alias("bad", "nonexistent_column + 1")
        
        # Get needed aliases for "pt" only
        needed = dsl._get_needed_aliases(["pt"])
        
        # Only "pt" should be needed, not "bad"
        assert "pt" in needed
        assert "bad" not in needed
    
    @pytest.mark.feature("api_alias")
    def test_alias_dependency_chain(self):
        """
        Alias dependencies are traced correctly.
        
        high_pt → pt → px, py
        """
        dsl = DSLCompiler()
        
        dsl.alias("pt", "sqrt(px**2 + py**2)")
        dsl.alias("high_pt", "pt > 10")
        
        needed = dsl._get_needed_aliases(["high_pt"])
        
        # Both should be needed
        assert "pt" in needed
        assert "high_pt" in needed
    
    @pytest.mark.feature("api_alias")
    def test_extract_dependencies(self):
        """
        _extract_dependencies() correctly extracts variable names.
        """
        dsl = DSLCompiler()
        
        deps = dsl._extract_dependencies("sqrt(px**2 + py**2)")
        assert "px" in deps
        assert "py" in deps
        assert "sqrt" in deps  # Functions are also Name nodes
        
        deps2 = dsl._extract_dependencies("a + b * c")
        assert deps2 == {"a", "b", "c"}


# =============================================================================
# Test Class: Cycle detection
# =============================================================================

class TestAliasCycleDetection:
    """Tests for circular dependency detection."""
    
    @pytest.mark.feature("api_alias")
    def test_alias_cycle_detection(self):
        """
        Circular dependencies raise CYCLE_ERROR.
        
        a → b → a
        """
        dsl = DSLCompiler()
        
        dsl.alias("a", "b + 1")
        dsl.alias("b", "a + 1")  # Circular!
        
        with pytest.raises(IRError) as exc_info:
            dsl._get_needed_aliases(["a"])
        
        error = exc_info.value
        assert "Circular dependency" in str(error)
    
    @pytest.mark.feature("api_alias")
    def test_alias_self_reference(self):
        """
        Self-referencing alias raises CYCLE_ERROR.
        
        a → a
        """
        dsl = DSLCompiler()
        
        dsl.alias("a", "a + 1")  # Self-reference!
        
        with pytest.raises(IRError) as exc_info:
            dsl._get_needed_aliases(["a"])
        
        error = exc_info.value
        assert "Circular dependency" in str(error)
    
    @pytest.mark.feature("api_alias")
    def test_alias_no_cycle_reuse(self):
        """
        Reusing an alias in multiple places is NOT a cycle.
        
        a → x
        b → a
        c → a (reuse, not cycle)
        """
        dsl = DSLCompiler()
        
        dsl.alias("a", "x + 1")
        dsl.alias("b", "a + 2")
        dsl.alias("c", "a + 3")  # Reuses 'a', but not a cycle
        
        # Should NOT raise
        needed = dsl._get_needed_aliases(["b", "c"])
        
        assert "a" in needed
        assert "b" in needed
        assert "c" in needed


# =============================================================================
# Test Class: from_rdf and update_schema_from_rdf
# =============================================================================

class TestFromRdf:
    """Tests for from_rdf() and update_schema_from_rdf()."""
    
    @pytest.mark.feature("api_from_rdf")
    def test_from_rdf_basic(self, nd_2d_rdf, nd_2d_schema):
        """
        from_rdf() infers schema from RDataFrame.
        """
        # Create DSLCompiler from RDF
        dsl = DSLCompiler.from_rdf(nd_2d_rdf)
        
        # Schema should have been populated
        assert len(dsl.schema) > 0
        
        # Should have some expected columns (from nd_2d fixture)
        # Note: actual columns depend on fixture
        assert dsl._rdf is nd_2d_rdf
    
    @pytest.mark.feature("api_from_rdf")
    def test_update_schema_from_rdf(self, nd_2d_rdf, nd_2d_schema):
        """
        update_schema_from_rdf() adds RDF columns to existing schema.
        """
        # Start with partial schema
        dsl = DSLCompiler({'custom_col': 'double'})
        
        # Update from RDF
        result = dsl.update_schema_from_rdf(nd_2d_rdf)
        
        # Should return self for chaining
        assert result is dsl
        
        # Custom column should still be there
        assert 'custom_col' in dsl.schema
        
        # RDF columns should have been added
        assert len(dsl.schema) > 1
    
    @pytest.mark.feature("api_from_rdf")
    def test_update_schema_preserves_manual(self, nd_2d_rdf, nd_2d_schema):
        """
        update_schema_from_rdf() doesn't overwrite manual schema entries.
        """
        # If nd_2d has 'event_id', we can test preservation
        dsl = DSLCompiler({'event_id': 'MANUAL_TYPE'})
        
        dsl.update_schema_from_rdf(nd_2d_rdf)
        
        # Manual entry should NOT be overwritten
        assert dsl.schema['event_id'] == 'MANUAL_TYPE'


# =============================================================================
# Test Class: Integration with to_pandas
# =============================================================================

class TestAliasIntegration:
    """Integration tests for alias() with to_pandas()."""
    
    @pytest.mark.feature("api_alias")
    def test_alias_compile_on_demand(self, nd_2d_rdf, nd_2d_schema):
        """
        Aliases are compiled only when needed for to_pandas().
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Add alias that uses existing column
        dsl.alias("double_weight", "event_weight * 2")
        
        # Alias should be in pool, not schema
        assert "double_weight" in dsl._aliases
        assert "double_weight" not in dsl.schema
        
        # Call to_pandas with the alias
        df = dsl.to_pandas(nd_2d_rdf, columns=['event_id', 'double_weight'])
        
        # Should have materialized and worked
        assert 'double_weight' in df.columns
        
        # Alias should now be in schema (materialized)
        assert "double_weight" in dsl.schema
        assert "double_weight" not in dsl._aliases
    
    @pytest.mark.feature("api_alias")
    def test_alias_missing_column_error(self, nd_2d_rdf, nd_2d_schema):
        """
        Missing column in alias raises error with suggestions at materialization.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Alias with typo
        dsl.alias("bad", "nonexistent_column + 1")
        
        # Should raise when trying to use it
        with pytest.raises(IRError) as exc_info:
            dsl.to_pandas(nd_2d_rdf, columns=['event_id', 'bad'])
        
        # Error should mention the bad column
        assert "nonexistent_column" in str(exc_info.value).lower() or \
               "not found" in str(exc_info.value).lower()
    
    @pytest.mark.feature("api_alias")
    def test_alias_type_registration(self, nd_2d_rdf, nd_2d_schema):
        """
        After compilation, alias output type is registered in schema.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Define alias chain
        dsl.alias("double_weight", "event_weight * 2")
        dsl.alias("triple_weight", "double_weight + event_weight")
        
        # Materialize by requesting triple_weight
        dsl._materialize_aliases(['triple_weight'], nd_2d_rdf)
        
        # Both should now be in schema
        assert "double_weight" in dsl.schema
        assert "triple_weight" in dsl.schema


# =============================================================================
# Test Class: Empty schema workflow
# =============================================================================

class TestEmptySchemaWorkflow:
    """Tests for schema-less workflow using alias()."""
    
    @pytest.mark.feature("api_alias")
    def test_empty_schema_allowed(self):
        """
        DSLCompiler() with no arguments creates empty schema.
        """
        dsl = DSLCompiler()
        
        assert dsl.schema == {}
        assert len(dsl._aliases) == 0
    
    @pytest.mark.feature("api_alias")
    def test_empty_schema_with_none(self):
        """
        DSLCompiler(None) also creates empty schema.
        """
        dsl = DSLCompiler(None)
        
        assert dsl.schema == {}
    
    @pytest.mark.feature("api_alias")
    def test_define_with_empty_schema_fails(self):
        """
        define() with empty schema raises error (need schema for immediate validation).
        """
        dsl = DSLCompiler()
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("pt", "sqrt(px**2 + py**2)")
        
        # Should fail because px, py not in schema
        error = exc_info.value
        assert "px" in str(error).lower() or "unknown" in str(error).lower()
# =============================================================================
# ADDITION to tests/test_api_alias.py
# =============================================================================
# Add this class to test_api_alias.py after TestAliasIntegration
#
# Phase: 13.6.G+
# Date: 2026-01-27
#
# BUG: define() does not resolve aliases
# - to_pandas(columns=['alias'])  ✅ works
# - draw("alias")                 ✅ works
# - define("x", "alias")          ❌ FAILS
#
# Priority: P0 - Basic API consistency
# =============================================================================


# =============================================================================
# Test Class: define() should resolve aliases (BUG FIX NEEDED)
# =============================================================================

class TestDefineAliasResolution:
    """
    Tests for define() alias resolution.
    
    BUG: define("x", "alias") fails with "Unknown variable 'alias'"
    
    These tests document expected behavior and will FAIL until bug is fixed.
    """
    
    @pytest.mark.feature("api_define")
    @pytest.mark.feature("api_alias")
    @pytest.mark.p0
    @pytest.mark.root_serial
    def test_define_resolves_simple_alias(self, nd_2d_rdf, nd_2d_schema):
        """
        define() must resolve aliases - basic consistency requirement.
        
        BUG: Currently fails with IRError: "Unknown variable 'my_alias'"
        
        Alias resolution must work consistently across:
        - to_pandas(columns=['alias'])  ✅ works
        - draw("alias")                 ✅ works
        - define("x", "alias")          ❌ FAILS (BUG)
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("my_alias", "track_pt[0]")
        
        # define() should resolve alias
        dsl.define("final", "my_alias")
        
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["final"])
        
        assert "final" in data
        assert "my_alias" in dsl.schema  # Alias should be materialized
    
    @pytest.mark.feature("api_define")
    @pytest.mark.feature("api_alias")
    @pytest.mark.p0
    @pytest.mark.root_serial
    def test_define_resolves_chained_alias(self, nd_2d_rdf, nd_2d_schema):
        """
        define() must resolve chained aliases.
        
        Chain: c → b → a → track_pt[0]
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("a", "track_pt[0]")
        dsl.alias("b", "a + 1")
        dsl.alias("c", "b + 1")
        
        # Should resolve c → b → a → track_pt[0]
        dsl.define("result", "c")
        
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["result"])
        
        assert "result" in data
        # All aliases in chain should be materialized
        assert "a" in dsl.schema
        assert "b" in dsl.schema
        assert "c" in dsl.schema
    
    @pytest.mark.feature("api_define")
    @pytest.mark.feature("api_alias")
    @pytest.mark.p0
    @pytest.mark.root_serial
    def test_define_with_alias_in_expression(self, nd_2d_rdf, nd_2d_schema):
        """
        define() should resolve aliases used within expressions.
        
        Example: define("x", "my_alias * 2")
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("pt0", "track_pt[0]")
        
        # Alias used in expression
        dsl.define("double_pt", "pt0 * 2")
        
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["double_pt"])
        
        assert "double_pt" in data
        assert "pt0" in dsl.schema
    
    @pytest.mark.feature("api_define")
    @pytest.mark.feature("api_alias")
    @pytest.mark.p1
    @pytest.mark.root_serial
    def test_define_with_multiple_aliases(self, nd_2d_rdf, nd_2d_schema):
        """
        define() should resolve multiple aliases in one expression.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("pt0", "track_pt[0]")
        dsl.alias("eta0", "track_eta[0]")
        
        # Multiple aliases in expression
        dsl.define("combined", "pt0 + eta0")
        
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["combined"])
        
        assert "combined" in data
        assert "pt0" in dsl.schema
        assert "eta0" in dsl.schema
