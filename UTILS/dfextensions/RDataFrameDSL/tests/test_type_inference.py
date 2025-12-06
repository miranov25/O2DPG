"""
Tests for type inference (Phase 2).

This module contains two categories of tests:
1. Mock tests - Test TypeInferrer logic without ROOT
2. Real ROOT tests - Integration tests with actual ROOT trees

Mock tests run everywhere. ROOT tests are skipped if ROOT is unavailable.
"""

import pytest
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.type_inferrer import (
    TypeInferrer, VariableInfo,
    extract_inner_type, is_vector_type, is_rvec_type,
    is_collection_type, normalize_cpp_type
)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    ROOT = None


# =============================================================================
# Helper Function Tests (No ROOT Required)
# =============================================================================

class TestExtractInnerType:
    """Tests for extract_inner_type helper."""
    
    def test_simple_type(self):
        """Non-container type returns depth 0."""
        inner, depth = extract_inner_type("float")
        assert inner == "float"
        assert depth == 0
    
    def test_vector_float(self):
        """vector<float> returns depth 1."""
        inner, depth = extract_inner_type("vector<float>")
        assert inner == "float"
        assert depth == 1
    
    def test_std_vector(self):
        """std::vector<double> returns depth 1."""
        inner, depth = extract_inner_type("std::vector<double>")
        assert inner == "double"
        assert depth == 1
    
    def test_rvec_int(self):
        """RVec<int> returns depth 1."""
        inner, depth = extract_inner_type("RVec<int>")
        assert inner == "int"
        assert depth == 1
    
    def test_nested_vector(self):
        """vector<vector<float>> returns depth 2."""
        inner, depth = extract_inner_type("vector<vector<float>>")
        assert inner == "float"
        assert depth == 2
    
    def test_nested_rvec(self):
        """RVec<RVec<double>> returns depth 2."""
        inner, depth = extract_inner_type("RVec<RVec<double>>")
        assert inner == "double"
        assert depth == 2
    
    def test_vecops_rvec(self):
        """ROOT::VecOps::RVec<float> returns depth 1."""
        inner, depth = extract_inner_type("ROOT::VecOps::RVec<float>")
        assert inner == "float"
        assert depth == 1
    
    def test_vector_of_object(self):
        """vector<TParticle> returns object type."""
        inner, depth = extract_inner_type("vector<TParticle>")
        assert inner == "TParticle"
        assert depth == 1
    
    def test_object_type(self):
        """Plain object type returns depth 0."""
        inner, depth = extract_inner_type("TParticle")
        assert inner == "TParticle"
        assert depth == 0


class TestIsVectorType:
    """Tests for type detection helpers."""
    
    def test_std_vector(self):
        assert is_vector_type("std::vector<float>")
        assert is_vector_type("vector<int>")
    
    def test_rvec(self):
        assert is_rvec_type("RVec<float>")
        assert is_rvec_type("ROOT::VecOps::RVec<int>")
    
    def test_not_vector(self):
        assert not is_vector_type("float")
        assert not is_vector_type("TParticle")
    
    def test_collection(self):
        assert is_collection_type("vector<float>")
        assert is_collection_type("RVec<int>")
        assert not is_collection_type("double")


class TestNormalizeCppType:
    """Tests for C++ type normalization."""
    
    def test_strip_const(self):
        assert normalize_cpp_type("const float") == "float"
        assert normalize_cpp_type("float const") == "float"
    
    def test_strip_reference(self):
        assert normalize_cpp_type("float&") == "float"
        assert normalize_cpp_type("const float&") == "float"
    
    def test_strip_pointer(self):
        assert normalize_cpp_type("float*") == "float"
    
    def test_strip_whitespace(self):
        assert normalize_cpp_type("  float  ") == "float"


# =============================================================================
# Mock Tests (No ROOT Required)
# =============================================================================

class TestTypeInferrerFromSchema:
    """Tests for TypeInferrer using schema (no ROOT)."""
    
    def test_simple_schema(self):
        """Basic schema with primitive types."""
        schema = {
            "columns": {
                "px": {"dtype": "float", "rank": 0},
                "py": {"dtype": "float", "rank": 0},
                "nTracks": {"dtype": "int", "rank": 0},
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        info = inferrer.get_variable_info("px")
        assert info.dtype.kind == IRTypeKind.Float32
        assert info.rank == 0
        
        info = inferrer.get_variable_info("nTracks")
        assert info.dtype.kind == IRTypeKind.Int32
    
    def test_vector_types(self):
        """Schema with vector types."""
        schema = {
            "columns": {
                "track_pt": {"dtype": "float", "rank": 1, "is_jagged": True},
                "cluster_x": {"dtype": "double", "rank": 1, "is_jagged": False},
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        info = inferrer.get_variable_info("track_pt")
        assert info.dtype.kind == IRTypeKind.Float32
        assert info.rank == 1
        assert info.is_jagged
        
        info = inferrer.get_variable_info("cluster_x")
        assert info.dtype.kind == IRTypeKind.Float64
        assert info.rank == 1
        assert not info.is_jagged
    
    def test_object_types(self):
        """Schema with object types."""
        schema = {
            "columns": {
                "track": {"dtype": "o2::tpc::TrackTPC", "rank": 0},
                "tracks": {"dtype": "TParticle", "rank": 1, "is_jagged": True},
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        info = inferrer.get_variable_info("track")
        assert info.dtype.kind == IRTypeKind.Object
        assert info.dtype.cpp_type == "o2::tpc::TrackTPC"
        
        info = inferrer.get_variable_info("tracks")
        assert info.dtype.kind == IRTypeKind.Object
        assert info.dtype.cpp_type == "TParticle"
        assert info.rank == 1
    
    def test_simple_dtype_strings(self):
        """Schema with just dtype strings (shorthand)."""
        schema = {
            "columns": {
                "x": "float",
                "n": "int",
                "flag": "bool",
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.get_type("x").kind == IRTypeKind.Float32
        assert inferrer.get_type("n").kind == IRTypeKind.Int32
        assert inferrer.get_type("flag").kind == IRTypeKind.Bool
    
    def test_dtype_aliases(self):
        """Various dtype string aliases."""
        schema = {
            "columns": {
                "a": "float32",
                "b": "float64",
                "c": "int32",
                "d": "int64",
                "e": "double",
                "f": "long",
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.get_type("a").kind == IRTypeKind.Float32
        assert inferrer.get_type("b").kind == IRTypeKind.Float64
        assert inferrer.get_type("c").kind == IRTypeKind.Int32
        assert inferrer.get_type("d").kind == IRTypeKind.Int64
        assert inferrer.get_type("e").kind == IRTypeKind.Float64
        assert inferrer.get_type("f").kind == IRTypeKind.Int64
    
    def test_unknown_variable_error(self):
        """Accessing unknown variable raises error."""
        schema = {"columns": {"px": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        
        with pytest.raises(IRError) as exc_info:
            inferrer.get_variable_info("nonexistent")
        
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
        assert "nonexistent" in exc_info.value.message
    
    def test_similar_name_suggestions(self):
        """Error includes similar name suggestions."""
        schema = {
            "columns": {
                "trackPt": "float",
                "trackEta": "float",
                "trackPhi": "float",
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        
        with pytest.raises(IRError) as exc_info:
            inferrer.get_variable_info("trackP")
        
        # Should suggest trackPt, trackPhi (both start with trackP)
        assert len(exc_info.value.suggestions) > 0
    
    def test_has_variable(self):
        """has_variable method."""
        schema = {"columns": {"px": "float", "py": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.has_variable("px")
        assert inferrer.has_variable("py")
        assert not inferrer.has_variable("pz")
    
    def test_get_rank(self):
        """get_rank convenience method."""
        schema = {
            "columns": {
                "scalar": {"dtype": "float", "rank": 0},
                "vector": {"dtype": "float", "rank": 1},
                "matrix": {"dtype": "float", "rank": 2},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.get_rank("scalar") == 0
        assert inferrer.get_rank("vector") == 1
        assert inferrer.get_rank("matrix") == 2
    
    def test_is_jagged(self):
        """is_jagged convenience method."""
        schema = {
            "columns": {
                "fixed": {"dtype": "float", "rank": 1, "is_jagged": False},
                "jagged": {"dtype": "float", "rank": 1, "is_jagged": True},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        
        assert not inferrer.is_jagged("fixed")
        assert inferrer.is_jagged("jagged")
    
    def test_register_alias(self):
        """Register computed alias types."""
        schema = {"columns": {"px": "float", "py": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        
        # Register pt alias
        inferrer.register_alias("pt", IRType(IRTypeKind.Float64), rank=0)
        
        # Now pt is accessible
        assert inferrer.has_variable("pt")
        info = inferrer.get_variable_info("pt")
        assert info.dtype.kind == IRTypeKind.Float64
        assert info.source == "alias"
    
    def test_get_all_variables(self):
        """Get all registered variables."""
        schema = {"columns": {"px": "float", "py": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        inferrer.register_alias("pt", IRType(IRTypeKind.Float64))
        
        all_vars = inferrer.get_all_variables()
        assert "px" in all_vars
        assert "py" in all_vars
        assert "pt" in all_vars
    
    def test_get_column_names(self):
        """Get list of column names."""
        schema = {"columns": {"px": "float", "py": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        
        names = inferrer.get_column_names()
        assert "px" in names
        assert "py" in names
    
    def test_to_schema(self):
        """Export to schema dict."""
        schema = {
            "columns": {
                "px": {"dtype": "float", "rank": 0},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        
        exported = inferrer.to_schema()
        assert "columns" in exported
        assert "px" in exported["columns"]
    
    def test_describe(self):
        """describe() returns string summary."""
        schema = {"columns": {"px": "float", "py": "float"}}
        inferrer = TypeInferrer.from_schema(schema)
        
        desc = inferrer.describe()
        assert "px" in desc
        assert "py" in desc


class TestVariableInfo:
    """Tests for VariableInfo dataclass."""
    
    def test_basic_info(self):
        """Basic VariableInfo creation."""
        info = VariableInfo(
            name="px",
            dtype=IRType(IRTypeKind.Float32),
            rank=0,
        )
        assert info.name == "px"
        assert info.dtype.kind == IRTypeKind.Float32
        assert info.rank == 0
        assert not info.is_jagged
    
    def test_repr(self):
        """VariableInfo repr includes key info."""
        info = VariableInfo(
            name="tracks",
            dtype=IRType(IRTypeKind.Object, "TParticle"),
            rank=1,
            is_jagged=True,
        )
        r = repr(info)
        assert "tracks" in r
        assert "TParticle" in r
        assert "rank=1" in r
        assert "jagged" in r


# =============================================================================
# Real ROOT Tests (Require ROOT)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTypeInferrerFromTree:
    """Integration tests with real ROOT trees."""
    
    @pytest.fixture
    def simple_tree(self):
        """Create a simple TTree with various branch types."""
        import ROOT
        import tempfile
        import os
        
        # Create temp file
        tmpfile = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmpfile.close()
        
        f = ROOT.TFile(tmpfile.name, "RECREATE")
        tree = ROOT.TTree("test", "Test tree")
        
        # Create branches with different types
        import array
        
        px = array.array('f', [0.0])
        py = array.array('d', [0.0])
        n = array.array('i', [0])
        flag = array.array('b', [0])
        
        tree.Branch("px", px, "px/F")
        tree.Branch("py", py, "py/D")
        tree.Branch("n", n, "n/I")
        tree.Branch("flag", flag, "flag/B")
        
        # Fill with some data
        for i in range(10):
            px[0] = float(i)
            py[0] = float(i) * 0.5
            n[0] = i
            flag[0] = i % 2
            tree.Fill()
        
        tree.Write()
        
        yield tree, tmpfile.name
        
        # Cleanup
        f.Close()
        os.unlink(tmpfile.name)
    
    def test_scalar_branches(self, simple_tree):
        """Infer types from scalar branches."""
        tree, _ = simple_tree
        
        inferrer = TypeInferrer.from_tree(tree)
        
        # Float branch
        info = inferrer.get_variable_info("px")
        assert info.dtype.kind == IRTypeKind.Float32
        assert info.rank == 0
        
        # Double branch
        info = inferrer.get_variable_info("py")
        assert info.dtype.kind == IRTypeKind.Float64
        assert info.rank == 0
        
        # Int branch
        info = inferrer.get_variable_info("n")
        assert info.dtype.kind == IRTypeKind.Int32
        assert info.rank == 0
    
    def test_describe_tree(self, simple_tree):
        """describe() works with real tree."""
        tree, _ = simple_tree
        
        inferrer = TypeInferrer.from_tree(tree)
        desc = inferrer.describe()
        
        assert "px" in desc
        assert "py" in desc


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTypeInferrerVectorBranches:
    """Tests for vector/RVec branches with real ROOT."""
    
    @pytest.fixture
    def vector_tree(self):
        """Create TTree with vector branches."""
        import ROOT
        import tempfile
        import os
        
        # Ensure RVec dictionary is loaded
        ROOT.gInterpreter.Declare("""
            #include <ROOT/RVec.hxx>
        """)
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmpfile.close()
        
        f = ROOT.TFile(tmpfile.name, "RECREATE")
        tree = ROOT.TTree("test", "Test tree with vectors")
        
        # Create std::vector<float> branch
        vec_float = ROOT.std.vector['float']()
        tree.Branch("track_pt", vec_float)
        
        # Create RVec<double> branch
        rvec_double = ROOT.ROOT.VecOps.RVec['double']()
        tree.Branch("cluster_x", rvec_double)
        
        # Fill with some data
        for i in range(5):
            vec_float.clear()
            rvec_double.clear()
            for j in range(i + 1):
                vec_float.push_back(float(j))
                rvec_double.push_back(float(j) * 0.1)
            tree.Fill()
        
        tree.Write()
        
        yield tree, tmpfile.name
        
        f.Close()
        os.unlink(tmpfile.name)
    
    def test_std_vector_branch(self, vector_tree):
        """Infer types from std::vector branches."""
        tree, _ = vector_tree
        
        inferrer = TypeInferrer.from_tree(tree)
        
        info = inferrer.get_variable_info("track_pt")
        assert info.dtype.kind == IRTypeKind.Float32
        assert info.rank == 1
        assert info.is_jagged  # Vector is potentially jagged
    
    def test_rvec_branch(self, vector_tree):
        """Infer types from RVec branches."""
        tree, _ = vector_tree
        
        inferrer = TypeInferrer.from_tree(tree)
        
        info = inferrer.get_variable_info("cluster_x")
        assert info.dtype.kind == IRTypeKind.Float64
        assert info.rank == 1


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")  
class TestTypeInferrerWithSchema:
    """Tests for tree + schema override."""
    
    @pytest.fixture
    def tree_with_schema(self):
        """Create tree and matching schema."""
        import ROOT
        import tempfile
        import os
        import array
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmpfile.close()
        
        f = ROOT.TFile(tmpfile.name, "RECREATE")
        tree = ROOT.TTree("test", "Test tree")
        
        px = array.array('f', [0.0])
        tree.Branch("px", px, "px/F")
        
        for i in range(5):
            px[0] = float(i)
            tree.Fill()
        
        tree.Write()
        
        schema = {
            "columns": {
                # Override px to be double (schema wins)
                "px": {"dtype": "double", "rank": 0},
                # Add virtual column not in tree
                "pt": {"dtype": "float", "rank": 0},
            }
        }
        
        yield tree, schema, tmpfile.name
        
        f.Close()
        os.unlink(tmpfile.name)
    
    def test_schema_override(self, tree_with_schema):
        """Schema overrides tree-inferred types."""
        tree, schema, _ = tree_with_schema
        
        inferrer = TypeInferrer.from_tree(tree, schema=schema)
        
        # px should be double from schema, not float from tree
        info = inferrer.get_variable_info("px")
        assert info.dtype.kind == IRTypeKind.Float64
        assert info.source == "schema"
    
    def test_schema_adds_columns(self, tree_with_schema):
        """Schema can add columns not in tree."""
        tree, schema, _ = tree_with_schema
        
        inferrer = TypeInferrer.from_tree(tree, schema=schema)
        
        # pt is only in schema, not in tree
        assert inferrer.has_variable("pt")
        info = inferrer.get_variable_info("pt")
        assert info.dtype.kind == IRTypeKind.Float32


# =============================================================================
# Integration Tests
# =============================================================================

class TestTypeInferrerIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow(self):
        """Test complete type inference workflow."""
        # Start with schema
        schema = {
            "columns": {
                "px": {"dtype": "float", "rank": 0},
                "py": {"dtype": "float", "rank": 0},
                "tracks": {"dtype": "TParticle", "rank": 1, "is_jagged": True},
            },
            "aliases": {
                "pt": "sqrt(px**2 + py**2)",
            }
        }
        
        inferrer = TypeInferrer.from_schema(schema)
        
        # Check column types
        assert inferrer.get_type("px").kind == IRTypeKind.Float32
        assert inferrer.get_type("tracks").kind == IRTypeKind.Object
        assert inferrer.get_rank("tracks") == 1
        assert inferrer.is_jagged("tracks")
        
        # Register computed alias
        inferrer.register_alias("pt", IRType(IRTypeKind.Float64), rank=0)
        
        # Verify alias is accessible
        assert inferrer.has_variable("pt")
        info = inferrer.get_variable_info("pt")
        assert info.source == "alias"
        
        # Export and verify
        exported = inferrer.to_schema()
        assert "px" in exported["columns"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
