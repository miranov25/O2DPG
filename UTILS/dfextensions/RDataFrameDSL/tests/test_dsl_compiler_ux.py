"""
Phase 10: UX/API Sugar tests.

Tests for DSLCompiler convenience methods:
- from_tree(): Auto-infer schema from ROOT file
- show_types(): Display inferred types
- validate(): Fast consistency check
"""

import pytest

# Skip all tests if ROOT not available
ROOT = pytest.importorskip("ROOT")

from RDataFrameDSL import DSLCompiler


class TestFromTree:
    """DSLCompiler.from_tree() tests."""
    
    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample ROOT file with various branch types."""
        filepath = str(tmp_path / "test.root")
        
        # Create file with RDataFrame
        rdf = ROOT.RDataFrame(10)
        rdf = rdf.Define("px", "gRandom->Gaus(0, 1)")
        rdf = rdf.Define("py", "gRandom->Gaus(0, 1)")
        rdf = rdf.Define("pz", "gRandom->Gaus(0, 1)")
        rdf = rdf.Define("flag", "px > 0")  # bool branch
        rdf.Snapshot("Events", filepath)
        
        return filepath
    
    @pytest.fixture
    def vector_file(self, tmp_path):
        """Create a ROOT file with vector branches."""
        filepath = str(tmp_path / "vector_test.root")
        
        # Create file with vector branch
        rdf = ROOT.RDataFrame(10)
        rdf = rdf.Define("pt_vec", "ROOT::RVec<float>{1.0f, 2.0f, 3.0f}")
        rdf = rdf.Define("eta", "gRandom->Gaus(0, 1)")
        rdf.Snapshot("Events", filepath)
        
        return filepath
    
    def test_from_tree_basic(self, sample_file):
        """from_tree creates compiler with correct schema."""
        dsl = DSLCompiler.from_tree(sample_file, "Events")
        
        assert "px" in dsl.schema
        assert "py" in dsl.schema
        assert "pz" in dsl.schema
    
    def test_from_tree_infers_double(self, sample_file):
        """from_tree infers double type correctly."""
        dsl = DSLCompiler.from_tree(sample_file, "Events")
        
        # px should be double (or Double_t)
        px_type = dsl.schema["px"].lower()
        assert "double" in px_type or "float64" in px_type
    
    def test_from_tree_infers_bool(self, sample_file):
        """from_tree infers bool type correctly."""
        dsl = DSLCompiler.from_tree(sample_file, "Events")
        
        # flag should be bool (ROOT may store as int or bool)
        assert "flag" in dsl.schema
        flag_type = dsl.schema["flag"].lower()
        # ROOT may represent bool as int, bool, or Bool_t
        assert any(t in flag_type for t in ["bool", "int", "char"])
    
    def test_from_tree_infers_vector(self, vector_file):
        """from_tree infers vector types correctly."""
        dsl = DSLCompiler.from_tree(vector_file, "Events")
        
        # pt_vec should be RVec<float> or similar
        assert "pt_vec" in dsl.schema
        pt_type = dsl.schema["pt_vec"]
        assert "RVec" in pt_type or "vector" in pt_type
    
    def test_from_tree_with_override(self, sample_file):
        """from_tree applies overrides (adds new column)."""
        dsl = DSLCompiler.from_tree(
            sample_file, "Events",
            overrides={"custom": "RVec<TLorentzVector>"}
        )
        
        assert dsl.schema["custom"] == "RVec<TLorentzVector>"
    
    def test_from_tree_override_replaces(self, sample_file):
        """Override replaces auto-detected type."""
        dsl = DSLCompiler.from_tree(
            sample_file, "Events",
            overrides={"px": "float"}  # Override double → float
        )
        
        assert dsl.schema["px"] == "float"
    
    def test_from_tree_stores_source(self, sample_file):
        """from_tree stores source file and tree info."""
        dsl = DSLCompiler.from_tree(sample_file, "Events")
        
        assert dsl._source_file == sample_file
        assert dsl._source_tree == "Events"
    
    def test_from_tree_file_not_found(self):
        """from_tree raises on missing file."""
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            DSLCompiler.from_tree("nonexistent.root", "Events")
    
    def test_from_tree_tree_not_found(self, sample_file):
        """from_tree raises on missing tree."""
        with pytest.raises((KeyError, Exception)):
            DSLCompiler.from_tree(sample_file, "NonexistentTree")
    
    def test_from_tree_full_workflow(self, sample_file):
        """Full workflow: from_tree → define → apply."""
        dsl = DSLCompiler.from_tree(sample_file, "Events")
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        rdf = ROOT.RDataFrame("Events", sample_file)
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("pt").GetValue()
        assert result > 0  # Should have positive mean


class TestShowTypes:
    """show_types() tests."""
    
    def test_show_types_schema_only(self):
        """show_types displays schema columns."""
        dsl = DSLCompiler({"px": "double", "py": "double"})
        output = dsl.show_types()
        
        assert "Schema columns:" in output
        assert "px" in output
        assert "double" in output
    
    def test_show_types_with_definitions(self):
        """show_types includes defined columns."""
        dsl = DSLCompiler({"px": "double", "py": "double"})
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        output = dsl.show_types()
        
        assert "Defined columns:" in output
        assert "pt" in output
        assert "sqrt(px**2 + py**2)" in output
    
    def test_show_types_without_definitions(self):
        """show_types can exclude definitions."""
        dsl = DSLCompiler({"px": "double", "py": "double"})
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        output = dsl.show_types(include_definitions=False)
        
        assert "px" in output
        assert "Defined columns" not in output


class TestValidate:
    """validate() tests."""
    
    def test_validate_success(self):
        """validate returns empty list on success."""
        dsl = DSLCompiler({"px": "double", "py": "double"})
        dsl.define("pt", "sqrt(px**2 + py**2)")
        
        errors = dsl.validate()
        assert errors == []
    
    def test_validate_no_definitions(self):
        """validate with no definitions returns empty."""
        dsl = DSLCompiler({"px": "double"})
        errors = dsl.validate()
        assert errors == []
    
    def test_validate_multiple_definitions(self):
        """validate checks all definitions."""
        dsl = DSLCompiler({"px": "double", "py": "double", "pz": "double"})
        dsl.define("pt", "sqrt(px**2 + py**2)")
        dsl.define("p", "sqrt(px**2 + py**2 + pz**2)")
        dsl.define("ratio", "pt / p")
        
        errors = dsl.validate()
        assert errors == []


class TestFromTreeMock:
    """Mock tests (minimal ROOT dependency)."""
    
    def test_from_tree_classmethod_exists(self):
        """from_tree is a classmethod."""
        assert hasattr(DSLCompiler, 'from_tree')
        assert callable(getattr(DSLCompiler, 'from_tree'))
    
    def test_show_types_method_exists(self):
        """show_types is a method."""
        dsl = DSLCompiler({"x": "double"})
        assert hasattr(dsl, 'show_types')
        assert callable(getattr(dsl, 'show_types'))
    
    def test_validate_method_exists(self):
        """validate is a method."""
        dsl = DSLCompiler({"x": "double"})
        assert hasattr(dsl, 'validate')
        assert callable(getattr(dsl, 'validate'))


class TestToSimpleSchema:
    """TypeInferrer.to_simple_schema() tests."""
    
    def test_to_simple_schema_scalars(self):
        """to_simple_schema handles scalar types."""
        from RDataFrameDSL.type_inferrer import TypeInferrer
        
        schema = {
            "columns": {
                "px": {"dtype": "double", "rank": 0},
                "count": {"dtype": "int", "rank": 0},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        simple = inferrer.to_simple_schema()
        
        assert "px" in simple
        assert "count" in simple
    
    def test_to_simple_schema_vectors(self):
        """to_simple_schema handles vector types."""
        from RDataFrameDSL.type_inferrer import TypeInferrer
        
        schema = {
            "columns": {
                "pt": {"dtype": "double", "rank": 1, "cpp_type": "RVec<double>"},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        simple = inferrer.to_simple_schema()
        
        assert "pt" in simple
        assert "RVec" in simple["pt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
