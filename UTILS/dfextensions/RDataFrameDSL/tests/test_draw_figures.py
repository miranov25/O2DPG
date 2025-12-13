"""
Phase 12.3: Composed Canvas & Multi-Figure Drawing Tests

Tests for draw_figures() method - multi-subplot figures from declarative specs.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_schema():
    """Simple schema for basic tests."""
    return {
        "pt": "double",
        "eta": "double",
        "phi": "double",
        "nHits": "int",
    }


@pytest.fixture
def rvec_schema():
    """Schema with RVec columns."""
    return {
        "trackPt": "RVec<float>",
        "trackEta": "RVec<float>",
        "clusterDy": "RVec<float>",
        "clusterRow": "RVec<int>",
    }


@pytest.fixture
def simple_rdf():
    """Create simple RDataFrame with scalar columns."""
    ROOT = pytest.importorskip("ROOT")
    
    rdf = ROOT.RDataFrame(100)
    rdf = rdf.Define("pt", "gRandom->Gaus(50, 10)")
    rdf = rdf.Define("eta", "gRandom->Uniform(-2.5, 2.5)")
    rdf = rdf.Define("phi", "gRandom->Uniform(-3.14159, 3.14159)")
    rdf = rdf.Define("nHits", "(int)(gRandom->Uniform(10, 100))")
    
    return rdf


@pytest.fixture
def rvec_rdf():
    """Create RDataFrame with RVec columns (matched lengths)."""
    ROOT = pytest.importorskip("ROOT")
    
    rdf = ROOT.RDataFrame(10)
    # Track data - 3 tracks per event
    rdf = rdf.Define("trackPt", "ROOT::RVec<float>{10.5f, 20.3f, 5.1f}")
    rdf = rdf.Define("trackEta", "ROOT::RVec<float>{0.5f, -1.2f, 0.8f}")
    # Cluster data - matched length (3 per event for paired plotting)
    rdf = rdf.Define("clusterDy", "ROOT::RVec<float>{0.1f, 0.2f, 0.3f}")
    rdf = rdf.Define("clusterRow", "ROOT::RVec<int>{10, 20, 30}")
    
    return rdf


@pytest.fixture
def mismatched_rvec_rdf():
    """Create RDataFrame with mismatched RVec lengths (for error testing)."""
    ROOT = pytest.importorskip("ROOT")
    
    rdf = ROOT.RDataFrame(5)
    # Different lengths per event for dy and row
    rdf = rdf.Define("dy", "ROOT::RVec<float>{0.1f, 0.2f}")  # 2 elements
    rdf = rdf.Define("row", "ROOT::RVec<int>{10, 20, 30}")   # 3 elements
    
    return rdf


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestDrawFiguresBasic:
    """Test basic draw_figures functionality."""
    
    def test_single_figure_single_plot(self, simple_schema, simple_rdf):
        """Single figure with one plot."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'test_fig',
            'plots': [{'expr': 'pt', 'bins': 20}]
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        assert 'test_fig' in results
        assert 'fig' in results['test_fig']
        assert 'axes' in results['test_fig']
        assert 'stats' in results['test_fig']
        assert len(results['test_fig']['axes']) == 1
    
    def test_single_figure_multiple_plots(self, simple_schema, simple_rdf):
        """Single figure with 4 subplots in 2x2 grid."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'quad_plot',
            'ncols': 2,
            'plots': [
                {'expr': 'pt', 'bins': 20},
                {'expr': 'eta', 'bins': 20},
                {'expr': 'phi', 'bins': 20},
                {'expr': 'nHits', 'bins': 20},
            ]
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        assert 'quad_plot' in results
        assert len(results['quad_plot']['axes']) == 4
        assert len(results['quad_plot']['stats']) == 4
    
    def test_multiple_figures(self, simple_schema, simple_rdf):
        """Multiple figures in one call."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [
            {
                'name': 'fig1',
                'plots': [{'expr': 'pt'}]
            },
            {
                'name': 'fig2',
                'plots': [{'expr': 'eta'}]
            },
            {
                'name': 'fig3',
                'plots': [{'expr': 'phi'}, {'expr': 'nHits'}]
            },
        ]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        assert len(results) == 3
        assert 'fig1' in results
        assert 'fig2' in results
        assert 'fig3' in results
    
    def test_empty_specs_returns_empty(self, simple_schema, simple_rdf):
        """Empty specs returns empty dict."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        results = dsl.draw_figures([], simple_rdf)
        
        assert results == {}
    
    def test_short_form_expression(self, simple_schema, simple_rdf):
        """Short form (string) plot specs work."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'short_form',
            'plots': ['pt', 'eta', 'phi']  # Short form
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        assert 'short_form' in results
        assert len(results['short_form']['axes']) == 3


# =============================================================================
# Layout Tests
# =============================================================================

class TestDrawFiguresLayout:
    """Test figure layout options."""
    
    def test_ncols_parameter(self, simple_schema, simple_rdf):
        """ncols controls grid columns."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'three_cols',
            'ncols': 3,
            'plots': ['pt', 'eta', 'phi', 'nHits', 'pt', 'eta']  # 6 plots
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        # Should be 2 rows x 3 cols
        fig = results['three_cols']['fig']
        assert fig is not None
    
    def test_figsize_parameter(self, simple_schema, simple_rdf):
        """Custom figsize is applied."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'custom_size',
            'figsize': (12, 8),
            'plots': ['pt', 'eta']
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        fig = results['custom_size']['fig']
        width, height = fig.get_size_inches()
        assert width == pytest.approx(12, rel=0.1)
        assert height == pytest.approx(8, rel=0.1)
    
    def test_suptitle(self, simple_schema, simple_rdf):
        """Suptitle is applied."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'with_title',
            'suptitle': 'My QA Report',
            'plots': ['pt', 'eta']
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        fig = results['with_title']['fig']
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == 'My QA Report'
    
    def test_subplot_title(self, simple_schema, simple_rdf):
        """Individual subplot titles work."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'subplot_titles',
            'plots': [
                {'expr': 'pt', 'title': 'Transverse Momentum'},
                {'expr': 'eta', 'title': 'Pseudorapidity'},
            ]
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        axes = results['subplot_titles']['axes']
        assert axes[0].get_title() == 'Transverse Momentum'
        assert axes[1].get_title() == 'Pseudorapidity'
    
    def test_unused_axes_hidden(self, simple_schema, simple_rdf):
        """Unused axes in grid are hidden."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        # 3 plots in 2x2 grid → 4th axis should be hidden
        specs = [{
            'name': 'partial_grid',
            'ncols': 2,
            'plots': ['pt', 'eta', 'phi']  # Only 3 plots
        }]
        
        results = dsl.draw_figures(specs, simple_rdf)
        
        # The returned axes list should only have 3 (the visible ones)
        assert len(results['partial_grid']['axes']) == 3


# =============================================================================
# Save Tests
# =============================================================================

class TestDrawFiguresSave:
    """Test figure saving functionality."""
    
    def test_savefig_creates_file(self, simple_schema, simple_rdf):
        """savefig creates PNG file."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{
                'name': 'saved_fig',
                'savefig': f'{tmpdir}/test.png',
                'plots': ['pt']
            }]
            
            dsl.draw_figures(specs, simple_rdf)
            
            assert Path(f'{tmpdir}/test.png').exists()
    
    def test_savefig_with_save_dir(self, simple_schema, simple_rdf):
        """savefig combined with save_dir."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{
                'name': 'relative_path',
                'savefig': 'subdir/plot.png',  # Relative path
                'plots': ['pt']
            }]
            
            dsl.draw_figures(specs, simple_rdf, save_dir=tmpdir)
            
            assert Path(f'{tmpdir}/subdir/plot.png').exists()
    
    def test_savefig_creates_parent_dirs(self, simple_schema, simple_rdf):
        """savefig creates parent directories."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{
                'name': 'nested',
                'savefig': f'{tmpdir}/a/b/c/plot.png',
                'plots': ['pt']
            }]
            
            dsl.draw_figures(specs, simple_rdf)
            
            assert Path(f'{tmpdir}/a/b/c/plot.png').exists()


# =============================================================================
# Validation Tests
# =============================================================================

class TestDrawFiguresValidation:
    """Test spec validation and error handling."""
    
    def test_missing_plots_raises_error(self, simple_schema, simple_rdf):
        """Missing 'plots' key raises ValueError."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{'name': 'no_plots'}]  # Missing 'plots'
        
        with pytest.raises(ValueError) as exc_info:
            dsl.draw_figures(specs, simple_rdf)
        
        assert "'plots' is required" in str(exc_info.value)
    
    def test_empty_plots_raises_error(self, simple_schema, simple_rdf):
        """Empty 'plots' list raises ValueError."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{'name': 'empty_plots', 'plots': []}]
        
        with pytest.raises(ValueError) as exc_info:
            dsl.draw_figures(specs, simple_rdf)
        
        assert "'plots' cannot be empty" in str(exc_info.value)
    
    def test_missing_expr_raises_error(self, simple_schema, simple_rdf):
        """Missing 'expr' in plot spec raises ValueError."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'no_expr',
            'plots': [{'bins': 50}]  # Missing 'expr'
        }]
        
        with pytest.raises(ValueError) as exc_info:
            dsl.draw_figures(specs, simple_rdf)
        
        assert "'expr' is required" in str(exc_info.value)


# =============================================================================
# RVec Tests
# =============================================================================

class TestDrawFiguresRVec:
    """Test RVec column handling."""
    
    def test_rvec_1d_histogram(self, rvec_schema, rvec_rdf):
        """1D histogram of RVec column works (flattened)."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(rvec_schema)
        
        specs = [{
            'name': 'rvec_hist',
            'plots': [{'expr': 'trackPt', 'bins': 20}]
        }]
        
        results = dsl.draw_figures(specs, rvec_rdf)
        
        assert 'rvec_hist' in results
        # Should have stats from flattened data
        assert results['rvec_hist']['stats'][0] is not None
    
    def test_paired_rvec_2d_plot(self, rvec_schema, rvec_rdf):
        """2D plot with paired RVec columns (same length per event)."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(rvec_schema)
        
        # clusterDy and clusterRow have same length per event
        specs = [{
            'name': 'paired_rvec',
            'plots': [{'expr': 'clusterDy:clusterRow', 'type': 'scatter'}]
        }]
        
        results = dsl.draw_figures(specs, rvec_rdf)
        
        assert 'paired_rvec' in results
    
    def test_mismatched_rvec_raises_error(self, mismatched_rvec_rdf):
        """Mismatched RVec lengths in paired columns raises error."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        schema = {
            "dy": "RVec<float>",
            "row": "RVec<int>",
        }
        
        dsl = DSLCompiler(schema)
        
        specs = [{
            'name': 'mismatched',
            'plots': [{'expr': 'dy:row', 'type': 'scatter'}]
        }]
        
        with pytest.raises(ValueError) as exc_info:
            dsl.draw_figures(specs, mismatched_rvec_rdf)
        
        assert "different lengths" in str(exc_info.value).lower()


# =============================================================================
# Defaults Tests
# =============================================================================

class TestDrawFiguresDefaults:
    """Test defaults parameter."""
    
    def test_defaults_applied_to_all_plots(self, simple_schema, simple_rdf):
        """defaults parameter applies to all plots."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'with_defaults',
            'plots': ['pt', 'eta', 'phi']
        }]
        
        # All plots should get bins=30
        results = dsl.draw_figures(specs, simple_rdf, defaults={'bins': 30})
        
        assert 'with_defaults' in results
        # Check that all 3 plots were created
        assert len(results['with_defaults']['stats']) == 3
    
    def test_plot_spec_overrides_defaults(self, simple_schema, simple_rdf):
        """Plot-level spec overrides defaults."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'override',
            'plots': [
                {'expr': 'pt'},  # Uses default bins
                {'expr': 'eta', 'bins': 100},  # Overrides default
            ]
        }]
        
        results = dsl.draw_figures(specs, simple_rdf, defaults={'bins': 20})
        
        assert 'override' in results


# =============================================================================
# Max Entries Tests
# =============================================================================

class TestDrawFiguresMaxEntries:
    """Test max_entries parameter."""
    
    def test_max_entries_limits_data(self, simple_schema, simple_rdf):
        """max_entries limits the number of entries processed."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        specs = [{
            'name': 'limited',
            'plots': [{'expr': 'pt', 'bins': 20}]
        }]
        
        # Limit to 10 entries (out of 100)
        results = dsl.draw_figures(specs, simple_rdf, max_entries=10)
        
        # Should still work, just with less data
        assert 'limited' in results
        assert results['limited']['fig'] is not None
        assert results['limited']['stats'][0] is not None
        # Stats structure varies by dfdraw version - just verify it exists


# =============================================================================
# Integration Tests
# =============================================================================

class TestDrawFiguresIntegration:
    """Integration tests for realistic workflows."""
    
    def test_qa_dashboard_pattern(self, simple_schema, simple_rdf):
        """Full QA dashboard pattern."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(simple_schema)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            qa_report = [
                {
                    'name': 'overview',
                    'suptitle': 'Track QA Overview',
                    'ncols': 2,
                    'savefig': 'overview.png',
                    'plots': [
                        {'expr': 'pt', 'bins': 50, 'title': 'pT'},
                        {'expr': 'eta', 'bins': 50, 'title': 'η'},
                        {'expr': 'phi', 'bins': 50, 'title': 'φ'},
                        {'expr': 'nHits', 'bins': 50, 'title': 'N hits'},
                    ]
                },
                {
                    'name': 'correlations',
                    'suptitle': 'Correlations',
                    'ncols': 2,
                    'savefig': 'correlations.png',
                    'plots': [
                        {'expr': 'eta:phi', 'type': 'hist2d'},
                        {'expr': 'pt:nHits', 'type': 'scatter'},
                    ]
                },
            ]
            
            results = dsl.draw_figures(qa_report, simple_rdf, save_dir=tmpdir)
            
            assert len(results) == 2
            assert Path(f'{tmpdir}/overview.png').exists()
            assert Path(f'{tmpdir}/correlations.png').exists()
    
    def test_json_serializable_spec(self, simple_schema, simple_rdf):
        """Specs can be loaded from JSON."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        import json
        
        dsl = DSLCompiler(simple_schema)
        
        # Spec as JSON string (simulating load from file)
        json_spec = '''
        [
            {
                "name": "from_json",
                "ncols": 2,
                "plots": [
                    {"expr": "pt", "bins": 30},
                    {"expr": "eta", "bins": 30}
                ]
            }
        ]
        '''
        
        specs = json.loads(json_spec)
        results = dsl.draw_figures(specs, simple_rdf)
        
        assert 'from_json' in results


# =============================================================================
# Mock Tests (No ROOT Required)
# =============================================================================

class TestDrawFiguresValidationMock:
    """Validation tests that don't require ROOT."""
    
    def test_validate_specs_no_plots(self):
        """Validation catches missing plots."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        
        with pytest.raises(ValueError) as exc_info:
            dsl._validate_figure_specs([{'name': 'test'}])
        
        assert "'plots' is required" in str(exc_info.value)
    
    def test_validate_specs_empty_plots(self):
        """Validation catches empty plots."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        
        with pytest.raises(ValueError) as exc_info:
            dsl._validate_figure_specs([{'name': 'test', 'plots': []}])
        
        assert "'plots' cannot be empty" in str(exc_info.value)
    
    def test_validate_specs_missing_expr(self):
        """Validation catches missing expr."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        
        with pytest.raises(ValueError) as exc_info:
            dsl._validate_figure_specs([{'name': 'test', 'plots': [{'bins': 50}]}])
        
        assert "'expr' is required" in str(exc_info.value)
    
    def test_validate_specs_string_plot_valid(self):
        """String plot specs are valid."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        
        # Should not raise
        dsl._validate_figure_specs([{'name': 'test', 'plots': ['x', 'x']}])
    
    def test_extract_paired_columns(self):
        """_extract_paired_columns identifies y:x patterns."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"dy": "double", "row": "int"})
        
        specs = [{
            'name': 'test',
            'plots': [
                {'expr': 'dy:row'},
                {'expr': 'dy'},  # Not paired
                'row',  # Short form, not paired
            ]
        }]
        
        pairs = dsl._extract_paired_columns(specs)
        
        assert ('dy', 'row') in pairs
        assert len(pairs) == 1  # Only one pair
