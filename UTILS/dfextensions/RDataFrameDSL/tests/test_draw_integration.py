"""Phase 12.1: dfdraw integration tests with synthetic data."""

import pytest
import warnings
import numpy as np

# File-level marker: ALL tests in this file run serially (ROOT dependency)
pytestmark = pytest.mark.root_serial


# =============================================================================
# Dependency Collection Tests (IR-based)
# =============================================================================

class TestDependencyCollection:
    """Test _collect_dependencies using IRBuilder."""
    
    def test_simple_variable(self, scalar_schema):
        """Single variable extracted."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_dependencies("pt")
        assert deps == {"pt"}
    
    def test_binary_expression(self):
        """Variables from binary expression."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"px": "double", "py": "double"})
        deps = dsl._collect_dependencies("sqrt(px**2 + py**2)")
        assert deps == {"px", "py"}
    
    def test_function_not_included(self):
        """Function names (sqrt, abs) not included as dependencies."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"x": "double"})
        deps = dsl._collect_dependencies("abs(sqrt(x))")
        assert "abs" not in deps
        assert "sqrt" not in deps
        assert "x" in deps
    
    def test_namespace_not_included(self):
        """Namespace names not included as dependencies."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"x": "double"})
        deps = dsl._collect_dependencies("TMath.Sin(x)")
        assert "TMath" not in deps
        assert "x" in deps
    
    def test_multiple_variables(self):
        """Multiple variables from complex expression."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"row": "int", "dy": "float", "mP4": "float"})
        deps = dsl._collect_dependencies("(row < 152) & (abs(dy) < 10) & (abs(mP4) < 1.5)")
        assert deps == {"row", "dy", "mP4"}
    
    def test_rvec_column(self, track_cluster_schema):
        """RVec column dependency extraction."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(track_cluster_schema)
        deps = dsl._collect_dependencies("trackPt")
        assert deps == {"trackPt"}
    
    def test_rvec_expression(self, track_cluster_schema):
        """RVec expression with multiple columns."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(track_cluster_schema)
        deps = dsl._collect_dependencies("sqrt(clusterDy**2 + clusterDz**2)")
        assert deps == {"clusterDy", "clusterDz"}


class TestDrawDependencies:
    """Test _collect_draw_dependencies including selection parsing."""
    
    def test_simple_expr(self, scalar_schema):
        """Simple expression dependency."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_draw_dependencies("pt")
        assert "pt" in deps
    
    def test_y_x_syntax(self):
        """'y:x' syntax extracts both variables."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"dy": "float", "row": "int"})
        deps = dsl._collect_draw_dependencies("dy:row")
        assert deps == {"dy", "row"}
    
    def test_with_selection(self, scalar_schema):
        """Selection string adds dependencies."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_draw_dependencies("pt", selection="isOK && eta < 1.0")
        assert deps == {"pt", "isOK", "eta"}
    
    def test_with_group_by(self):
        """group_by adds dependency."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"pt": "double", "sector": "int"})
        deps = dsl._collect_draw_dependencies("pt", group_by="sector")
        assert deps == {"pt", "sector"}
    
    def test_with_color(self, scalar_schema):
        """color adds dependency."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_draw_dependencies("pt:eta", color="charge")
        assert deps == {"pt", "eta", "charge"}
    
    def test_unknown_column_warning(self):
        """Unknown column triggers warning."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler({"pt": "double"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deps = dsl._collect_draw_dependencies("pt:unknown_col")
            assert len(w) == 1
            assert "not in schema" in str(w[0].message)
            assert "unknown_col" in str(w[0].message)
    
    def test_rvec_y_x_syntax(self, track_cluster_schema):
        """'y:x' with RVec columns."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(track_cluster_schema)
        deps = dsl._collect_draw_dependencies("clusterDy:clusterZ")
        assert deps == {"clusterDy", "clusterZ"}
    
    def test_rvec_with_selection(self, track_cluster_schema):
        """RVec columns with selection."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(track_cluster_schema)
        deps = dsl._collect_draw_dependencies("trackPt", selection="trackIsOK")
        assert deps == {"trackPt", "trackIsOK"}


# =============================================================================
# draw() Method Tests
# =============================================================================

class TestDrawMethod:
    """Test draw() method."""
    
    def test_draw_import_error(self, scalar_schema):
        """Helpful error when dfdraw not installed."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        
        # Mock RDataFrame
        class MockRDF:
            def Range(self, n):
                return self
            def AsNumpy(self, cols):
                return {c: np.array([1.0, 2.0, 3.0]) for c in cols}
        
        try:
            dsl.draw("pt", MockRDF())
        except ImportError as e:
            assert "dfdraw" in str(e)
            assert "pip install" in str(e)
    
    def test_draw_with_max_entries(self, scalar_schema):
        """max_entries parameter limits data."""
        pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def __init__(self):
                self.range_called = False
                self.range_value = None
            
            def Range(self, n):
                self.range_called = True
                self.range_value = n
                return self
            
            def AsNumpy(self, cols):
                return {c: np.array([1.0, 2.0, 3.0]) for c in cols}
        
        rdf = MockRDF()
        try:
            dsl.draw("pt", rdf, max_entries=1000)
        except Exception:
            pass  # dfdraw may fail on mock data
        
        assert rdf.range_called
        assert rdf.range_value == 1000
    
    def test_draw_auto_detects_columns(self, scalar_schema):
        """Columns auto-detected from expression."""
        pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def __init__(self):
                self.columns_requested = None
            
            def Range(self, n):
                return self
            
            def AsNumpy(self, cols):
                self.columns_requested = set(cols)
                return {c: np.array([1.0, 2.0, 3.0]) for c in cols}
        
        rdf = MockRDF()
        try:
            dsl.draw("pt:eta", rdf, selection="isOK")
        except Exception:
            pass
        
        assert rdf.columns_requested == {"pt", "eta", "isOK"}


# =============================================================================
# to_dataframe() Tests
# =============================================================================

class TestToDataFrame:
    """Test to_dataframe() method."""
    
    def test_to_dataframe_all_columns(self, scalar_schema):
        """Export all schema columns."""
        from RDataFrameDSL import DSLCompiler
        import pandas as pd
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def AsNumpy(self, cols):
                return {c: np.array([1.0, 2.0, 3.0]) for c in cols}
        
        df = dsl.to_dataframe(MockRDF())
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(scalar_schema.keys())
        assert len(df) == 3
    
    def test_to_dataframe_specific_columns(self, scalar_schema):
        """Export specific columns only."""
        from RDataFrameDSL import DSLCompiler
        import pandas as pd
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def AsNumpy(self, cols):
                return {c: np.array([1.0, 2.0]) for c in cols}
        
        df = dsl.to_dataframe(MockRDF(), columns=["pt", "eta"])
        assert set(df.columns) == {"pt", "eta"}
        assert "phi" not in df.columns
    
    def test_to_dataframe_with_max_entries(self, scalar_schema):
        """max_entries limits rows."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def __init__(self):
                self.range_called = False
            
            def Range(self, n):
                self.range_called = True
                return self
            
            def AsNumpy(self, cols):
                return {c: np.array([1.0, 2.0]) for c in cols}
        
        rdf = MockRDF()
        dsl.to_dataframe(rdf, max_entries=100)
        assert rdf.range_called
    
    def test_to_dataframe_rvec_columns(self, track_cluster_schema):
        """RVec columns exported as arrays."""
        from RDataFrameDSL import DSLCompiler
        import pandas as pd
        
        dsl = DSLCompiler(track_cluster_schema)
        
        class MockRDF:
            def AsNumpy(self, cols):
                result = {}
                for c in cols:
                    # Simulate RVec: each row is an array
                    if "RVec" in track_cluster_schema.get(c, ""):
                        result[c] = np.array([
                            np.array([1.0, 2.0, 3.0]),
                            np.array([4.0, 5.0]),
                        ], dtype=object)
                    else:
                        result[c] = np.array([10, 20])
                return result
        
        df = dsl.to_dataframe(MockRDF(), columns=["nTracks", "trackPt"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # trackPt should be arrays
        assert isinstance(df["trackPt"].iloc[0], np.ndarray)


# =============================================================================
# draw_batch() Tests
# =============================================================================

class TestDrawBatch:
    """Test draw_batch() method."""
    
    def test_batch_collects_all_columns(self, track_cluster_schema):
        """All columns from all specs collected."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        
        specs = {
            'plot1': {'expr': 'trackPt:trackEta'},
            'plot2': {'expr': 'clusterDy:clusterZ', 'selection': 'trackIsOK'},
        }
        
        # Collect what columns would be needed
        all_cols = set()
        for spec in specs.values():
            all_cols.update(dsl._collect_draw_dependencies(
                spec.get('expr', ''),
                spec.get('selection')
            ))
        
        assert all_cols == {"trackPt", "trackEta", "clusterDy", "clusterZ", "trackIsOK"}
    
    def test_batch_single_asnumpy_call(self, scalar_schema):
        """Verify only one AsNumpy call for efficiency."""
        pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            def __init__(self):
                self.asnumpy_calls = 0
            
            def Range(self, n):
                return self
            
            def AsNumpy(self, cols):
                self.asnumpy_calls += 1
                return {c: np.array([1.0, 2.0, 3.0]) for c in cols}
        
        specs = {
            'plot1': {'expr': 'pt'},
            'plot2': {'expr': 'eta'},
            'plot3': {'expr': 'pt:eta'},
        }
        
        rdf = MockRDF()
        try:
            dsl.draw_batch(specs, rdf)
        except Exception:
            pass  # dfdraw may fail
        
        # Should be exactly 1 AsNumpy call, not 3
        assert rdf.asnumpy_calls == 1
    
    def test_batch_empty_specs(self, scalar_schema):
        """Empty specs returns empty dict."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        
        class MockRDF:
            pass
        
        result = dsl.draw_batch({}, MockRDF())
        assert result == {}


# =============================================================================
# Integration Tests (with ROOT)
# =============================================================================

class TestScalarIntegration:
    """Integration tests with scalar RDataFrame."""
    
    def test_draw_with_real_rdf(self, synthetic_scalar_rdf, scalar_schema):
        """Test draw with actual RDataFrame."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        fig, ax, stats = dsl.draw("pt", synthetic_scalar_rdf, selection="isOK")
        
        assert fig is not None
        assert ax is not None
        assert stats is not None
    
    def test_draw_2d(self, synthetic_scalar_rdf, scalar_schema):
        """Test 2D plot."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(scalar_schema)
        fig, ax, stats = dsl.draw("pt:eta", synthetic_scalar_rdf, type="scatter")
        
        assert fig is not None
    
    def test_to_dataframe_real(self, synthetic_scalar_rdf, scalar_schema):
        """Test to_dataframe with real RDataFrame."""
        from RDataFrameDSL import DSLCompiler
        import pandas as pd
        
        dsl = DSLCompiler(scalar_schema)
        df = dsl.to_dataframe(synthetic_scalar_rdf, max_entries=100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "pt" in df.columns
    
    def test_draw_defined_column(self, synthetic_scalar_rdf, scalar_schema):
        """Test draw with a column defined via dsl.define().
        
        Phase 13.6.G+: Uses redefinition='ifneeded' since draw() internally
        calls apply() again, and the column is already defined.
        """
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        # Add px, py to schema
        schema = dict(scalar_schema)
        schema["px"] = "double"
        schema["py"] = "double"
        
        # Create RDF with px, py
        ROOT = pytest.importorskip("ROOT")
        rdf = ROOT.RDataFrame(100)
        rdf = rdf.Define("px", "gRandom->Gaus(0, 1)")
        rdf = rdf.Define("py", "gRandom->Gaus(0, 1)")
        rdf = rdf.Define("pt", "gRandom->Gaus(10, 2)")
        rdf = rdf.Define("eta", "gRandom->Uniform(-2, 2)")
        rdf = rdf.Define("phi", "gRandom->Uniform(-3.14159, 3.14159)")
        rdf = rdf.Define("isOK", "abs(eta) < 1.0")
        rdf = rdf.Define("charge", "gRandom->Rndm() > 0.5 ? 1 : -1")
        
        # Phase 13.6.G+: Use redefinition='ifneeded' for interactive workflow
        dsl = DSLCompiler(schema, redefinition="ifneeded")
        dsl.define("pt_calc", "sqrt(px**2 + py**2)")
        
        rdf = dsl.apply(rdf)
        
        # Draw the DEFINED column - draw() calls apply() again, but with
        # redefinition='ifneeded', it skips since expression unchanged
        fig, ax, stats = dsl.draw("pt_calc", rdf)
        assert fig is not None


class TestRVecIntegration:
    """Integration tests with RVec (Track → Cluster) RDataFrame."""
    
    def test_rvec_to_dataframe(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Export RVec columns to DataFrame."""
        from RDataFrameDSL import DSLCompiler
        import pandas as pd
        
        dsl = DSLCompiler(track_cluster_schema)
        df = dsl.to_dataframe(
            synthetic_track_cluster_rdf, 
            columns=["nTracks", "trackPt", "trackEta"],
            max_entries=50
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        # trackPt should contain arrays
        assert hasattr(df["trackPt"].iloc[0], '__len__')
    
    def test_rvec_draw_histogram(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Draw histogram of RVec column."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        
        # This should flatten RVec for histogram
        fig, ax, stats = dsl.draw("trackPt", synthetic_track_cluster_rdf, max_entries=100)
        
        assert fig is not None
    
    def test_rvec_draw_2d(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Draw 2D plot with RVec columns."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        
        fig, ax, stats = dsl.draw(
            "clusterDy:clusterZ", 
            synthetic_track_cluster_rdf,
            type="hist2d",
            max_entries=100
        )
        
        assert fig is not None
    
    def test_rvec_batch_draw(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Batch draw with RVec columns."""
        dfdraw = pytest.importorskip("dfdraw")
        from RDataFrameDSL import DSLCompiler
        import tempfile
        
        dsl = DSLCompiler(track_cluster_schema)
        
        specs = {
            'track_pt': {'expr': 'trackPt', 'bins': 50},
            'track_eta': {'expr': 'trackEta', 'bins': 40},
            'cluster_dy_z': {'expr': 'clusterDy:clusterZ', 'type': 'hist2d'},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = dsl.draw_batch(
                specs, 
                synthetic_track_cluster_rdf,
                save_dir=tmpdir,
                max_entries=100
            )
        
        assert len(results) == 3
        assert all('fig' in r for r in results.values())


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_selection(self, scalar_schema):
        """Empty selection string handled."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_draw_dependencies("pt", selection="")
        assert deps == {"pt"}
    
    def test_none_selection(self, scalar_schema):
        """None selection handled."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        deps = dsl._collect_draw_dependencies("pt", selection=None)
        assert deps == {"pt"}
    
    def test_complex_selection_string(self, track_cluster_schema):
        """Complex selection with multiple operators."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(track_cluster_schema)
        deps = dsl._collect_draw_dependencies(
            "trackPt",
            selection="trackIsOK && trackEta > -1 && trackEta < 1 && nTracks > 0"
        )
        assert deps == {"trackPt", "trackIsOK", "trackEta", "nTracks"}
    
    def test_fallback_tokenization(self):
        """Fallback tokenization handles unknown expressions."""
        from RDataFrameDSL import DSLCompiler
        # Use empty schema to force fallback
        dsl = DSLCompiler({})
        # This will use fallback because 'unknown' isn't in schema
        deps = dsl._collect_dependencies("unknown_var + 5")
        assert "unknown_var" in deps
        assert "5" not in deps  # Numbers not included
    
    def test_defined_column_in_dependencies(self, scalar_schema):
        """Dependencies include columns defined via define()."""
        from RDataFrameDSL import DSLCompiler
        dsl = DSLCompiler(scalar_schema)
        dsl.define("pt2", "pt * 2")
        
        # pt2 should now be in schema
        deps = dsl._collect_draw_dependencies("pt2:eta")
        assert "pt2" in deps
        assert "eta" in deps
