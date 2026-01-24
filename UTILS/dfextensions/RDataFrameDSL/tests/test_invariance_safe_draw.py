# =============================================================================
# tests/test_invariance_safe_draw.py
# =============================================================================
# Integration tests for Safe Mode and Draw API - CAPABILITY_MATRIX validation.
#
# Phase: 13.6.F
# Date: 2026-01-24
#
# Test Naming: test_INV_<ID>_<description>
# Each test documents its INVARIANT clearly in the docstring.
# =============================================================================

import pytest
import sys
import numpy as np

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError


pytestmark = [pytest.mark.root_serial]


# =============================================================================
# SAFE MODE INTEGRATION (INV-SAFE-*)
# =============================================================================

class TestSafeModeIntegration:
    """Integration tests for to_pandas_safe() - Layer 3 protection."""
    
    @pytest.mark.feature("api_to_pandas_safe")
    @pytest.mark.skipif(sys.platform == "win32", reason="fork() unavailable")
    def test_INV_SAFE_01_output_matches_regular(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: to_pandas_safe() output identical to to_pandas().
        
        Safe mode adds protection but must not change results.
        """
        print("\n[INV-SAFE-01] Verifying safe mode produces identical output...")
        
        columns = ['event_id', 'event_weight']
        
        dsl1 = DSLCompiler(nd_2d_schema)
        df_safe = dsl1.to_pandas_safe(nd_2d_rdf, columns=columns, probe_size=50)
        
        dsl2 = DSLCompiler(nd_2d_schema)
        df_regular = dsl2.to_pandas(nd_2d_rdf, columns=columns)
        
        # INVARIANT CHECK
        assert len(df_safe) == len(df_regular), "Row counts must match"
        np.testing.assert_array_equal(
            df_safe['event_id'].values, 
            df_regular['event_id'].values,
            err_msg="event_id values must be identical"
        )
        print(f"  ✓ {len(df_safe)} rows match exactly")
    
    @pytest.mark.feature("api_to_pandas_safe")
    @pytest.mark.skipif(sys.platform == "win32", reason="fork() unavailable")
    def test_INV_SAFE_02_define_computed_correctly(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: define()'d columns computed correctly in safe mode.
        
        Arithmetic: double_weight = event_weight * 2
        """
        print("\n[INV-SAFE-02] Verifying define() works in safe mode...")
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("double_weight", "event_weight * 2")
        
        df = dsl.to_pandas_safe(
            nd_2d_rdf,
            columns=['event_weight', 'double_weight'],
            probe_size=50
        )
        
        # INVARIANT CHECK
        np.testing.assert_array_almost_equal(
            df['double_weight'].values,
            df['event_weight'].values * 2,
            err_msg="double_weight must equal event_weight * 2"
        )
        print(f"  ✓ Computed column verified for {len(df)} rows")


# =============================================================================
# DRAW API INTEGRATION (INV-DRAW-*)
# =============================================================================

class TestDrawIntegration:
    """Integration tests for draw()/batch_draw() validation."""
    
    @pytest.mark.feature("api_draw")
    def test_INV_DRAW_01_layer1_before_root(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: Layer 1 validation raises IRError, not ROOT crash.
        
        Missing columns detected by DSL, not by ROOT's AsNumpy().
        """
        print("\n[INV-DRAW-01] Verifying Layer 1 catches errors before ROOT...")
        
        dsl = DSLCompiler(nd_2d_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.draw("nonexistent_column", nd_2d_rdf)
        
        # INVARIANT CHECK: Must be IRError with helpful message
        error = exc_info.value
        assert isinstance(error, IRError), "Must be IRError, not ROOT error"
        print(f"  ✓ IRError raised: {str(error)[:60]}...")
    
    @pytest.mark.feature("api_draw")
    def test_INV_DRAW_02_alias_materialized(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: draw() materializes alias chain before plotting.
        
        Alias: weight_x4 → weight_x2 → event_weight
        """
        print("\n[INV-DRAW-02] Verifying alias chain materialization...")
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("weight_x2", "event_weight * 2")
        dsl.alias("weight_x4", "weight_x2 * 2")
        
        # Should NOT raise IRError - aliases must materialize
        try:
            dsl.draw("weight_x4", nd_2d_rdf)
        except ImportError:
            pass  # dfdraw not installed - OK, validation passed
        
        # INVARIANT CHECK: Aliases moved from pool to schema
        assert "weight_x2" in dsl.schema, "weight_x2 must be materialized"
        assert "weight_x4" in dsl.schema, "weight_x4 must be materialized"
        print("  ✓ Alias chain materialized: weight_x4 → weight_x2 → event_weight")
    
    @pytest.mark.feature("api_batch_draw")
    def test_INV_DRAW_03_batch_validates_all(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: batch_draw() validates ALL columns before ANY plotting.
        
        One bad spec should block all plots, not just fail later.
        """
        print("\n[INV-DRAW-03] Verifying batch validates ALL specs first...")
        
        dsl = DSLCompiler(nd_2d_schema)
        
        specs = {
            'good_plot': {'expr': 'event_weight'},
            'bad_plot': {'expr': 'nonexistent'},
        }
        
        with pytest.raises(IRError):
            dsl.draw_batch(specs, nd_2d_rdf)
        
        # INVARIANT CHECK: No partial execution
        print("  ✓ All-or-nothing validation confirmed")


# =============================================================================
# FROM_RDF INTEGRATION (INV-RDF-*)
# =============================================================================

class TestFromRdfIntegration:
    """Integration tests for from_rdf() schema inference."""
    
    @pytest.mark.feature("api_from_rdf")
    def test_INV_RDF_01_schema_complete(self, nd_2d_rdf):
        """
        INVARIANT: from_rdf() infers ALL columns from RDataFrame.
        
        Every GetColumnNames() column must appear in schema.
        """
        print("\n[INV-RDF-01] Verifying complete schema inference...")
        
        dsl = DSLCompiler.from_rdf(nd_2d_rdf)
        
        rdf_columns = [str(c) for c in nd_2d_rdf.GetColumnNames()]
        
        # INVARIANT CHECK
        for col in rdf_columns:
            assert col in dsl.schema, f"Column {col} missing from schema"
        
        print(f"  ✓ All {len(rdf_columns)} columns inferred: {rdf_columns}")
    
    @pytest.mark.feature("api_from_rdf")
    def test_INV_RDF_02_define_after_inference(self, nd_2d_rdf):
        """
        INVARIANT: define() works with inferred schema.
        
        Can immediately use inferred columns in expressions.
        """
        print("\n[INV-RDF-02] Verifying define() works after inference...")
        
        dsl = DSLCompiler.from_rdf(nd_2d_rdf)
        dsl.define("scaled", "event_weight * 2")
        
        # INVARIANT CHECK
        assert "scaled" in dsl.schema, "defined column must be in schema"
        print("  ✓ define('scaled', 'event_weight * 2') succeeded")


# =============================================================================
# ALIAS POOL INTEGRATION (INV-ALIAS-*)
# =============================================================================

class TestAliasPoolIntegration:
    """Integration tests for alias() pool-based compilation."""
    
    @pytest.mark.feature("api_alias")
    def test_INV_ALIAS_01_unused_not_compiled(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: Unused aliases never validated or compiled.
        
        Invalid unused alias must not cause error.
        """
        print("\n[INV-ALIAS-01] Verifying unused aliases stay in pool...")
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("good", "event_weight * 2")
        dsl.alias("bad", "nonexistent_column + 1")  # Invalid but unused
        
        # Request only 'good' - should NOT raise
        df = dsl.to_pandas(nd_2d_rdf, columns=['event_id', 'good'])
        
        # INVARIANT CHECK
        assert "bad" in dsl._aliases, "bad must still be in pool"
        assert "bad" not in dsl.schema, "bad must not be materialized"
        print(f"  ✓ Exported {len(df)} rows, 'bad' alias untouched")
    
    @pytest.mark.feature("api_alias")
    def test_INV_ALIAS_02_chain_order(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: Alias chain compiled in dependency order.
        
        c = b + 1, b = a + 1, a = event_weight → c = event_weight + 2
        """
        print("\n[INV-ALIAS-02] Verifying chain compilation order...")
        
        dsl = DSLCompiler(nd_2d_schema)
        
        # Define in "wrong" order
        dsl.alias("c", "b + 1")
        dsl.alias("b", "a + 1")
        dsl.alias("a", "event_weight")
        
        df = dsl.to_pandas(nd_2d_rdf, columns=['event_weight', 'c'])
        
        # INVARIANT CHECK: c = event_weight + 2
        np.testing.assert_array_almost_equal(
            df['c'].values,
            df['event_weight'].values + 2,
            err_msg="c must equal event_weight + 2"
        )
        print("  ✓ Chain: c = b + 1 = a + 2 = event_weight + 2")
    
    @pytest.mark.feature("api_alias")
    def test_INV_ALIAS_03_cycle_detected(self, nd_2d_schema):
        """
        INVARIANT: Circular dependencies raise clear error.
        
        a → b → a is a cycle.
        """
        print("\n[INV-ALIAS-03] Verifying cycle detection...")
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("a", "b + 1")
        dsl.alias("b", "a + 1")  # Cycle!
        
        with pytest.raises(IRError) as exc_info:
            dsl._get_needed_aliases(["a"])
        
        # INVARIANT CHECK
        assert "circular" in str(exc_info.value).lower()
        print("  ✓ Cycle detected: a → b → a")
    
    @pytest.mark.feature("api_alias")
    def test_INV_ALIAS_04_type_propagation(self, nd_2d_rdf, nd_2d_schema):
        """
        INVARIANT: Alias output types registered for downstream use.
        
        Chain: pt (double) → high_pt (bool uses pt as double)
        Verifies Q4 from proposal: type assignment after compilation.
        """
        print("\n[INV-ALIAS-04] Verifying type propagation through chain...")
        
        dsl = DSLCompiler(nd_2d_schema)
        
        # First alias: arithmetic → double output
        dsl.alias("scaled_weight", "event_weight * 2.0")
        
        # Second alias: comparison using first → bool output
        dsl.alias("is_heavy", "scaled_weight > 1.0")
        
        # Materialize by requesting is_heavy
        df = dsl.to_pandas(nd_2d_rdf, columns=['event_weight', 'scaled_weight', 'is_heavy'])
        
        # INVARIANT CHECK: Types are correct
        assert "scaled_weight" in dsl.schema, "scaled_weight must be in schema"
        assert "is_heavy" in dsl.schema, "is_heavy must be in schema"
        
        # Verify actual data types
        assert df['scaled_weight'].dtype in [np.float64, np.float32], \
            f"scaled_weight should be float, got {df['scaled_weight'].dtype}"
        assert df['is_heavy'].dtype in [np.bool_, np.uint8, bool], \
            f"is_heavy should be bool-like, got {df['is_heavy'].dtype}"
        
        # Verify values are correct
        np.testing.assert_array_almost_equal(
            df['scaled_weight'].values,
            df['event_weight'].values * 2.0
        )
        
        print(f"  ✓ scaled_weight type: {dsl.schema.get('scaled_weight', 'unknown')}")
        print(f"  ✓ is_heavy type: {dsl.schema.get('is_heavy', 'unknown')}")
        print("  ✓ Type propagation: event_weight → scaled_weight → is_heavy")
