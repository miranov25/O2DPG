# =============================================================================
# tests/test_api_draw.py
# =============================================================================
# API tests for draw() and batch_draw() methods.
#
# Phase: 13.6.F
# Date: 2026-01-24
#
# Tests verify:
# - Layer 1 validation before plotting
# - Alias materialization for draw expressions
# - Safe mode integration
# - Error reporting (ALL errors, not just first)
#
# Note: These tests require dfdraw to be installed for full execution.
# Tests gracefully skip if dfdraw is not available.
# =============================================================================

import pytest
import sys

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError, IRErrorKind


# =============================================================================
# Helper: Check if dfdraw is available
# =============================================================================

def dfdraw_available():
    """Check if dfdraw is installed."""
    try:
        from dfextensions.dfdraw import DFDraw
        return True
    except ImportError:
        return False


# =============================================================================
# Test Class: draw() Layer 1 Validation
# =============================================================================

class TestDrawValidation:
    """Tests for draw() Layer 1 validation."""
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_validates_columns(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() raises IRError for missing columns (Layer 1).
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.draw("nonexistent_column", nd_2d_rdf)
        
        error = exc_info.value
        assert "nonexistent" in str(error).lower() or "not found" in str(error).lower()
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_validates_before_dfdraw(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() validates columns before attempting to import dfdraw.
        
        Even without dfdraw installed, validation should occur first.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # This should raise IRError for missing column, not ImportError for dfdraw
        with pytest.raises(IRError):
            dsl.draw("bad_column", nd_2d_rdf)
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_with_alias(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() materializes aliases before plotting.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("double_weight", "event_weight * 2")
        
        # Should NOT raise - alias should be materialized
        # (Will raise ImportError if dfdraw not installed, which is fine)
        try:
            dsl.draw("double_weight", nd_2d_rdf)
        except ImportError:
            # dfdraw not installed - that's OK, validation passed
            pass
        except IRError:
            # Should NOT happen - alias should have been materialized
            pytest.fail("draw() should materialize aliases before validation")
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_error_has_suggestions(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() error includes "Did you mean...?" suggestions.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Typo: "event_wieght" instead of "event_weight"
        with pytest.raises(IRError) as exc_info:
            dsl.draw("event_wieght", nd_2d_rdf)
        
        error = exc_info.value
        # Should have suggestions
        assert error.suggestions is not None
        suggestions_str = ' '.join(error.suggestions or [])
        assert 'event_weight' in suggestions_str.lower() or 'did you mean' in suggestions_str.lower()


# =============================================================================
# Test Class: batch_draw() Validation
# =============================================================================

class TestBatchDrawValidation:
    """Tests for batch_draw() Layer 1 validation."""
    
    @pytest.mark.feature("api_batch_draw")
    @pytest.mark.root_serial
    def test_batch_draw_validates_all(self, nd_2d_rdf, nd_2d_schema):
        """
        batch_draw() validates ALL columns before any plotting.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        specs = {
            'good_plot': {'expr': 'event_weight'},
            'bad_plot': {'expr': 'nonexistent_column'},
        }
        
        with pytest.raises(IRError) as exc_info:
            dsl.draw_batch(specs, nd_2d_rdf)
        
        error = exc_info.value
        assert "nonexistent" in str(error).lower() or "not found" in str(error).lower()
    
    @pytest.mark.feature("api_batch_draw")
    @pytest.mark.root_serial
    def test_batch_draw_empty_specs(self, nd_2d_rdf, nd_2d_schema):
        """
        batch_draw() returns empty dict for empty specs.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        result = dsl.draw_batch({}, nd_2d_rdf)
        
        assert result == {}
    
    @pytest.mark.feature("api_batch_draw")
    @pytest.mark.root_serial
    def test_batch_draw_with_aliases(self, nd_2d_rdf, nd_2d_schema):
        """
        batch_draw() materializes all needed aliases.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("scaled_weight", "event_weight * 10")
        dsl.alias("many_tracks", "n_tracks > 5")  # Use n_tracks, not multiplicity
        
        specs = {
            'weight_plot': {'expr': 'scaled_weight'},
            'tracks_plot': {'expr': 'many_tracks'},
        }
        
        # Should NOT raise IRError - aliases should be materialized
        try:
            dsl.draw_batch(specs, nd_2d_rdf)
        except ImportError:
            # dfdraw not installed - validation passed
            pass
        except IRError:
            pytest.fail("batch_draw() should materialize aliases before validation")


# =============================================================================
# Test Class: Safe Mode Integration
# =============================================================================

class TestDrawSafeMode:
    """Tests for draw() safe mode integration."""
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    @pytest.mark.skipif(sys.platform == "win32", reason="fork() not available on Windows")
    def test_draw_safe_mode(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() with safe_mode=True uses probe-run.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Should pass probe and either:
        # - Draw successfully (if dfdraw installed)
        # - Raise ImportError (if dfdraw not installed)
        try:
            dsl.draw("event_weight", nd_2d_rdf, safe_mode=True, probe_size=100)
        except ImportError:
            # dfdraw not installed - probe passed, that's fine
            pass
    
    @pytest.mark.feature("api_batch_draw")
    @pytest.mark.root_serial
    @pytest.mark.skipif(sys.platform == "win32", reason="fork() not available on Windows")
    def test_batch_draw_safe_mode(self, nd_2d_rdf, nd_2d_schema):
        """
        batch_draw() with safe_mode=True uses probe-run.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        specs = {
            'weight': {'expr': 'event_weight'},
        }
        
        try:
            dsl.draw_batch(specs, nd_2d_rdf, safe_mode=True, probe_size=100)
        except ImportError:
            # dfdraw not installed - probe passed
            pass


# =============================================================================
# Test Class: Full draw() execution (requires dfdraw)
# =============================================================================

@pytest.mark.skipif(not dfdraw_available(), reason="dfdraw not installed")
class TestDrawExecution:
    """Full execution tests for draw() - requires dfdraw."""
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_histogram(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() creates histogram for scalar column.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        fig, ax, stats = dsl.draw("event_weight", nd_2d_rdf, bins=50)
        
        assert fig is not None
        assert ax is not None
        assert stats is not None
    
    @pytest.mark.feature("api_draw")
    @pytest.mark.root_serial
    def test_draw_with_define(self, nd_2d_rdf, nd_2d_schema):
        """
        draw() works with define()'d columns.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("scaled", "event_weight * 2")
        
        fig, ax, stats = dsl.draw("scaled", nd_2d_rdf)
        
        assert fig is not None
    
    @pytest.mark.feature("api_batch_draw")
    @pytest.mark.root_serial
    def test_batch_draw_multiple(self, nd_2d_rdf, nd_2d_schema):
        """
        batch_draw() creates multiple plots.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        specs = {
            'weight': {'expr': 'event_weight', 'bins': 30},
            'tracks': {'expr': 'n_tracks', 'bins': 20},
        }
        
        results = dsl.draw_batch(specs, nd_2d_rdf)
        
        assert 'weight' in results
        assert 'tracks' in results
        assert results['weight']['fig'] is not None
        assert results['tracks']['fig'] is not None
