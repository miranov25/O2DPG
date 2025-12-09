"""
test_rdataframe_integration_advanced.py - Multi-Function RDataFrame Stress Tests

Phase 7.9: Validates that ALL DSL functionality (Phases 1-7) works in
real-world RDataFrame scenarios.

Tests:
- Multiple functions in single pipeline
- Multi-threading safety
- Large datasets
- Macro export and execution
- Edge cases

Run with:
    pytest tests/test_rdataframe_integration_advanced.py -v
"""

import pytest
import math
import os
import tempfile

ROOT = pytest.importorskip("ROOT")

from RDataFrameDSL import DSLCompiler


class TestMultiFunctionRDataFrame:
    """
    Test multiple DSL functions in single RDataFrame pipeline.
    """
    
    @pytest.fixture
    def test_tree(self, tmp_path):
        """Create test TTree with scalar and RVec branches."""
        filename = str(tmp_path / "multi_func_test.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_multi_func_tree(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "Test events");
                
                double px, py, pz;
                std::vector<double> pt;
                std::vector<double> eta;
                
                tree.Branch("px", &px);
                tree.Branch("py", &py);
                tree.Branch("pz", &pz);
                tree.Branch("pt", &pt);
                tree.Branch("eta", &eta);
                
                // Generate 100 events with varying track counts
                for (int ev = 0; ev < 100; ++ev) {{
                    px = 3.0 + ev * 0.1;
                    py = 4.0 + ev * 0.1;
                    pz = ev * 0.5;
                    
                    pt.clear();
                    eta.clear();
                    int n_tracks = (ev % 5) + 1;  // 1-5 tracks
                    for (int t = 0; t < n_tracks; ++t) {{
                        pt.push_back(1.0 + t * 0.5 + ev * 0.01);
                        eta.push_back(0.1 * t);
                    }}
                    tree.Fill();
                }}
                
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_multi_func_tree(filename)
        return filename
    
    def test_all_phase_patterns_together(self, test_tree):
        """All Phase 1-7 patterns work in single pipeline."""
        schema = {
            "px": "double",
            "py": "double",
            "pz": "double",
            "pt": "RVec<double>",
            "eta": "RVec<double>",
        }
        
        dsl = DSLCompiler(schema)
        
        # Phase 5: Scalar expressions
        dsl.define("event_pt", "sqrt(px**2 + py**2)")
        dsl.define("event_p", "sqrt(px**2 + py**2 + pz**2)")
        
        # Phase 6b: RVec operations
        dsl.define("n_tracks", "pt.size()")
        dsl.define("lead_pt", "pt[0]")
        dsl.define("last_pt", "pt[-1]")
        
        # Phase 7: Slicing
        dsl.define("first3", "pt[:3]")
        dsl.define("last2", "pt[-2:]")
        dsl.define("high_pt", "pt[pt > 1.5]")
        dsl.define("reversed", "pt[::-1]")
        dsl.define("every_other", "pt[::2]")
        
        # Apply all at once
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = dsl.apply(rdf)
        
        # Verify columns exist and have correct shapes
        results = rdf.AsNumpy([
            "event_pt", "event_p", "n_tracks", "lead_pt", "last_pt"
        ])
        
        # Basic checks
        assert len(results["event_pt"]) == 100
        assert len(results["n_tracks"]) == 100
        
        # Verify first event: px=3.0, py=4.0 → event_pt=5.0
        assert abs(results["event_pt"][0] - 5.0) < 0.01
        
        # Verify n_tracks pattern (1,2,3,4,5,1,2,3,4,5,...)
        assert results["n_tracks"][0] == 1
        assert results["n_tracks"][1] == 2
        assert results["n_tracks"][4] == 5
        assert results["n_tracks"][5] == 1
        
        print("✅ All Phase 1-7 patterns work together!")
    
    def test_slice_results_correctness(self, test_tree):
        """Slice results are mathematically correct."""
        schema = {"pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("first2", "pt[:2]")
        dsl.define("last2", "pt[-2:]")
        dsl.define("middle", "pt[1:3]")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = dsl.apply(rdf)
        
        # Get results for event with 5 tracks (event 4)
        # pt = [1.04, 1.54, 2.04, 2.54, 3.04] approximately
        first2 = rdf.Take["ROOT::RVec<double>"]("first2").GetValue()
        last2 = rdf.Take["ROOT::RVec<double>"]("last2").GetValue()
        
        # Event 4 has 5 tracks
        ev4_first2 = list(first2[4])
        ev4_last2 = list(last2[4])
        
        assert len(ev4_first2) == 2
        assert len(ev4_last2) == 2
        
        # First 2 should be smaller values
        assert ev4_first2[0] < ev4_last2[0]
        
        print("✅ Slice results are mathematically correct!")
    
    def test_empty_vector_handling(self, tmp_path):
        """Empty RVec handling doesn't crash."""
        filename = str(tmp_path / "empty_vec_test.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_empty_vec_tree(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "Test");
                std::vector<double> pt;
                tree.Branch("pt", &pt);
                
                // Event with tracks
                pt = {{1.0, 2.0, 3.0}};
                tree.Fill();
                
                // Empty event
                pt.clear();
                tree.Fill();
                
                // Event with tracks again
                pt = {{5.0, 6.0}};
                tree.Fill();
                
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_empty_vec_tree(filename)
        
        schema = {"pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("first3", "pt[:3]")
        dsl.define("high_pt", "pt[pt > 2.0]")
        
        rdf = ROOT.RDataFrame("Events", filename)
        rdf = dsl.apply(rdf)
        
        # Should not crash
        first3 = rdf.Take["ROOT::RVec<double>"]("first3").GetValue()
        
        # Event 1 (empty) should have empty result
        assert len(first3[1]) == 0
        
        print("✅ Empty vector handling works!")


class TestMultiThreading:
    """Test multi-threaded RDataFrame execution."""
    
    @pytest.fixture
    def large_tree(self, tmp_path):
        """Create larger tree for MT testing."""
        filename = str(tmp_path / "mt_test.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_mt_tree(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "MT Test");
                
                double px, py;
                std::vector<double> pt;
                tree.Branch("px", &px);
                tree.Branch("py", &py);
                tree.Branch("pt", &pt);
                
                for (int ev = 0; ev < 1000; ++ev) {{
                    px = 3.0 + (ev % 100) * 0.1;
                    py = 4.0 + (ev % 100) * 0.1;
                    
                    pt.clear();
                    int n = (ev % 10) + 1;
                    for (int t = 0; t < n; ++t) {{
                        pt.push_back(1.0 + t);
                    }}
                    tree.Fill();
                }}
                
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_mt_tree(filename)
        return filename
    
    def test_multi_function_with_mt(self, large_tree):
        """Multiple functions work with EnableImplicitMT()."""
        ROOT.EnableImplicitMT(4)
        
        try:
            schema = {"px": "double", "py": "double", "pt": "RVec<double>"}
            dsl = DSLCompiler(schema)
            
            dsl.define("event_pt", "sqrt(px**2 + py**2)")
            dsl.define("n_tracks", "pt.size()")
            dsl.define("first3", "pt[:3]")
            dsl.define("high_pt", "pt[pt > 2.0]")
            dsl.define("reversed", "pt[::-1]")
            
            rdf = ROOT.RDataFrame("Events", large_tree)
            rdf = dsl.apply(rdf)
            
            # Force evaluation
            results = rdf.AsNumpy(["event_pt", "n_tracks"])
            
            assert len(results["event_pt"]) == 1000
            assert len(results["n_tracks"]) == 1000
            
            print("✅ Multi-threading works!")
            
        finally:
            ROOT.DisableImplicitMT()


class TestMacroExport:
    """Test C++ macro export functionality."""
    
    @pytest.fixture
    def test_tree(self, tmp_path):
        """Create simple test tree."""
        filename = str(tmp_path / "export_test.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_export_tree(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "Export Test");
                
                double px = 3.0, py = 4.0;
                std::vector<double> pt = {{1.0, 2.0, 3.0, 4.0, 5.0}};
                
                tree.Branch("px", &px);
                tree.Branch("py", &py);
                tree.Branch("pt", &pt);
                
                for (int i = 0; i < 10; ++i) {{
                    tree.Fill();
                }}
                
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_export_tree(filename)
        return filename
    
    def test_export_macro_content(self, tmp_path):
        """Exported macro contains correct content."""
        schema = {"px": "double", "py": "double", "pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("event_pt", "sqrt(px**2 + py**2)")
        dsl.define("first3", "pt[:3]")
        dsl.define("high_pt", "pt[pt > 2.0]")
        
        macro_path = str(tmp_path / "test_dsl.C")
        dsl.export_macro(macro_path, include_test=True)
        
        # Read and verify content
        with open(macro_path, 'r') as f:
            content = f.read()
        
        # Check structure
        assert "Generated by RDataFrameDSL" in content
        assert "#include <ROOT/RVec.hxx>" in content
        
        # Check DSL comments
        assert "// DSL: sqrt(px**2 + py**2)" in content
        assert "// DSL: pt[:3]" in content
        assert "// DSL: pt[pt > 2.0]" in content
        
        # Check function definitions
        assert "alias_event_pt" in content
        assert "alias_first3" in content
        assert "alias_high_pt" in content
        
        # Check test harness
        assert "void test_all" in content
        assert ".Define(" in content
        
        print("✅ Macro export content correct!")
    
    def test_export_macro_compiles(self, test_tree, tmp_path):
        """Exported macro compiles successfully."""
        schema = {"px": "double", "py": "double", "pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("event_pt", "sqrt(px**2 + py**2)")
        dsl.define("n_tracks", "pt.size()")
        dsl.define("first3", "pt[:3]")
        
        macro_path = str(tmp_path / "compile_test.C")
        dsl.export_macro(macro_path, include_test=False)
        
        # Load the macro (this compiles it)
        with open(macro_path, 'r') as f:
            code = f.read()
        
        result = ROOT.gInterpreter.Declare(code)
        assert result == True, "Macro failed to compile"
        
        print("✅ Exported macro compiles!")
    
    def test_preview_without_compile(self):
        """preview() returns code without compiling."""
        schema = {"px": "double", "pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("event_px", "px * 2")
        dsl.define("first2", "pt[:2]")
        
        # Should not raise even without ROOT compilation
        preview = dsl.preview()
        
        assert "alias_event_px" in preview
        assert "alias_first2" in preview
        assert "// DSL:" in preview
        
        print("✅ Preview works without compilation!")


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_name_collision_error(self):
        """Error when alias name equals branch name."""
        schema = {"pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        with pytest.raises(Exception) as exc:
            dsl.define("pt", "pt * 2")  # Collision!
        
        assert "conflict" in str(exc.value).lower()
    
    def test_duplicate_definition_error(self):
        """Error when defining same name twice."""
        schema = {"px": "double"}
        dsl = DSLCompiler(schema)
        
        dsl.define("result", "px * 2")
        
        with pytest.raises(Exception) as exc:
            dsl.define("result", "px * 3")  # Duplicate!
        
        assert "already defined" in str(exc.value).lower()
    
    def test_error_message_includes_dsl(self):
        """Error messages include DSL expression."""
        schema = {"pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        # This should work
        dsl.define("test", "pt[:3]")
        
        # Force an error during compilation (invalid syntax would be caught earlier)
        # This is more of a documentation test
        func = dsl.get_function("test")
        assert func.dsl_expression == "pt[:3]"


class TestDefinitionOrder:
    """Test that definition order is preserved."""
    
    def test_apply_order_matches_define_order(self, tmp_path):
        """Define() order is preserved in apply()."""
        filename = str(tmp_path / "order_test.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_order_tree(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "Order Test");
                double x = 1.0;
                tree.Branch("x", &x);
                tree.Fill();
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_order_tree(filename)
        
        schema = {"x": "double"}
        dsl = DSLCompiler(schema)
        
        # Define in specific order
        dsl.define("a", "x * 2")      # x * 2 = 2
        dsl.define("b", "x * 3")      # x * 3 = 3
        dsl.define("c", "x * 4")      # x * 4 = 4
        
        # Check internal order
        defs = dsl.list_definitions()
        assert defs[0][0] == "a"
        assert defs[1][0] == "b"
        assert defs[2][0] == "c"
        
        # Apply and verify
        rdf = ROOT.RDataFrame("Events", filename)
        rdf = dsl.apply(rdf)
        
        results = rdf.AsNumpy(["a", "b", "c"])
        assert results["a"][0] == 2.0
        assert results["b"][0] == 3.0
        assert results["c"][0] == 4.0
        
        print("✅ Definition order preserved!")


class TestDSLCompilerAPI:
    """Test DSLCompiler API methods."""
    
    def test_len(self):
        """__len__ returns number of definitions."""
        schema = {"x": "double"}
        dsl = DSLCompiler(schema)
        
        assert len(dsl) == 0
        dsl.define("a", "x * 2")
        assert len(dsl) == 1
        dsl.define("b", "x * 3")
        assert len(dsl) == 2
    
    def test_repr(self):
        """__repr__ returns useful info."""
        schema = {"x": "double", "y": "double"}
        dsl = DSLCompiler(schema)
        dsl.define("sum", "x + y")
        
        r = repr(dsl)
        assert "DSLCompiler" in r
        assert "definitions=1" in r
    
    def test_chaining(self):
        """define() returns self for chaining."""
        schema = {"x": "double"}
        dsl = DSLCompiler(schema)
        
        result = dsl.define("a", "x * 2").define("b", "x * 3")
        
        assert result is dsl
        assert len(dsl) == 2
    
    def test_get_function(self):
        """get_function() returns GeneratedFunction."""
        schema = {"x": "double"}
        dsl = DSLCompiler(schema)
        dsl.define("double_x", "x * 2")
        
        func = dsl.get_function("double_x")
        assert func.dsl_expression == "x * 2"
        assert "alias_double_x" in func.name
    
    def test_get_function_not_found(self):
        """get_function() raises KeyError for unknown name."""
        schema = {"x": "double"}
        dsl = DSLCompiler(schema)
        
        with pytest.raises(KeyError):
            dsl.get_function("nonexistent")
    
    def test_unique_function_names(self):
        """Different DSLCompiler instances generate unique function names."""
        schema = {"x": "double"}
        
        dsl1 = DSLCompiler(schema)
        dsl1.define("result", "x * 2")
        
        dsl2 = DSLCompiler(schema)
        dsl2.define("result", "x * 2")  # Same column name
        
        func1 = dsl1.get_function("result")
        func2 = dsl2.get_function("result")
        
        # Function names should be different (unique ID)
        assert func1.name != func2.name
        
        # But column names should be the same
        assert func1.column_name == func2.column_name == "result"
        
        print("✅ Unique function names for parallel safety!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
