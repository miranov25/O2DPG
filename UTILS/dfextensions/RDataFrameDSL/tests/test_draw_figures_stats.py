"""
Phase 12.5.DSL: Statistical annotation tests for draw_figures().

Tests the show_statistics and show_expected parameters added to draw_figures().
"""

import pytest
import numpy as np


class TestDrawFiguresStatistics:
    """Test show_statistics and show_expected parameters."""
    
    @pytest.fixture
    def dsl_with_data(self):
        """Create DSL with deterministic test data."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        # Use deterministic expressions (avoid gRandom for CI stability)
        rdf = ROOT.RDataFrame(1000)
        rdf = rdf.Define("pt", "50.0 + (rdfentry_ % 100) * 0.2")
        rdf = rdf.Define("dy_pull", "((int)(rdfentry_ % 200) - 100) * 0.01")
        rdf = rdf.Define("dz_pull", "((int)(rdfentry_ % 200) - 100) * 0.011 + 0.1")
        
        # Use simple schema format: {"col": "ctype"}
        schema = {
            "pt": "double", 
            "dy_pull": "double", 
            "dz_pull": "double"
        }
        dsl = DSLCompiler(schema)
        
        return dsl, rdf
    
    def test_show_statistics_adds_content(self, dsl_with_data):
        """Statistics box contains statistical values."""
        dsl, rdf = dsl_with_data
        
        specs = [{'name': 'test', 'plots': [{'expr': 'pt', 'bins': 50}]}]
        results = dsl.draw_figures(specs, rdf, show_statistics=True)
        
        ax = results['test']['axes'][0]
        texts = [t.get_text() for t in ax.texts]
        text_content = ' '.join(texts)
        
        # Verify statistics markers present (μ, σ, n or mean, std, count)
        has_stats = any(marker in text_content for marker in ['n', 'μ', 'σ', 'mean', 'std', '='])
        assert has_stats, f"Statistics box missing. Got texts: {texts}"
    
    def test_show_expected_adds_overlay_for_pull(self, dsl_with_data):
        """Gaussian overlay with label added for pull expressions."""
        dsl, rdf = dsl_with_data
        
        specs = [{'name': 'test', 'plots': [{'expr': 'dy_pull', 'bins': 50}]}]
        results = dsl.draw_figures(specs, rdf, show_expected=True)
        
        ax = results['test']['axes'][0]
        
        # Check for labeled overlay line
        has_gaussian = False
        
        # Check legend
        handles, labels = ax.get_legend_handles_labels()
        if any('N(0' in str(lbl) for lbl in labels):
            has_gaussian = True
        
        # Check line labels directly
        for line in ax.get_lines():
            if 'N(0' in str(line.get_label()):
                has_gaussian = True
                break
        
        assert has_gaussian, "Gaussian overlay with N(0,1) label not found"
    
    def test_no_overlay_for_non_pull(self, dsl_with_data):
        """No Gaussian overlay for non-pull expressions."""
        dsl, rdf = dsl_with_data
        
        specs = [{'name': 'test', 'plots': [{'expr': 'pt', 'bins': 50}]}]
        results = dsl.draw_figures(specs, rdf, show_expected=True)
        
        ax = results['test']['axes'][0]
        
        # Verify N(0,1) label is NOT present
        has_gaussian = False
        handles, labels = ax.get_legend_handles_labels()
        if any('N(0' in str(lbl) for lbl in labels):
            has_gaussian = True
        for line in ax.get_lines():
            if 'N(0' in str(line.get_label()):
                has_gaussian = True
        
        assert not has_gaussian, "Unexpected Gaussian overlay on non-pull expression"
    
    def test_explicit_is_pull_override_true(self, dsl_with_data):
        """Explicit is_pull=True forces overlay on non-pull column."""
        dsl, rdf = dsl_with_data
        
        # pt doesn't contain "pull" but we force it
        specs = [{'name': 'test', 'plots': [
            {'expr': 'pt', 'bins': 50, 'is_pull': True}
        ]}]
        results = dsl.draw_figures(specs, rdf, show_expected=True)
        
        ax = results['test']['axes'][0]
        
        has_overlay = False
        for line in ax.get_lines():
            if 'N(0' in str(line.get_label()):
                has_overlay = True
        
        assert has_overlay, "Overlay missing when is_pull=True explicitly set"
    
    def test_explicit_is_pull_override_false(self, dsl_with_data):
        """Explicit is_pull=False suppresses overlay on pull column."""
        dsl, rdf = dsl_with_data
        
        # dy_pull contains "pull" but we suppress
        specs = [{'name': 'test', 'plots': [
            {'expr': 'dy_pull', 'bins': 50, 'is_pull': False}
        ]}]
        results = dsl.draw_figures(specs, rdf, show_expected=True)
        
        ax = results['test']['axes'][0]
        
        has_overlay = False
        for line in ax.get_lines():
            if 'N(0' in str(line.get_label()):
                has_overlay = True
        
        assert not has_overlay, "Overlay present despite is_pull=False"
    
    def test_backward_compatible_defaults(self, dsl_with_data):
        """Default behavior unchanged (no annotations)."""
        dsl, rdf = dsl_with_data
        
        specs = [{'name': 'test', 'plots': [{'expr': 'pt', 'bins': 50}]}]
        results = dsl.draw_figures(specs, rdf)  # No new params
        
        assert results is not None
        assert 'test' in results
        assert results['test']['fig'] is not None
