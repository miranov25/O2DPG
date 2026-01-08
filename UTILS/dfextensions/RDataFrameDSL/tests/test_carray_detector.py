"""
Phase 13.4.DSL - Tests for C-Array Detector (D2 + D3)

Tests:
- Branch title parsing
- Fixed-size array detection
- Variable-length array detection
- Counter branch discovery
- Multi-dimensional arrays

Run with: pytest tests/test_carray_detector.py -v
"""

import pytest
from RDataFrameDSL.carray_detector import (
    CArrayInfo,
    CArraySchema,
    BranchTitleParser,
    CArrayDetector,
    detect_carrays,
)


# =============================================================================
# Test CArrayInfo
# =============================================================================

class TestCArrayInfo:
    """Test CArrayInfo dataclass."""
    
    def test_fixed_1d_total_size(self):
        """Fixed 1D array calculates total size."""
        info = CArrayInfo(
            name="arr",
            shape=(10,),
            dtype="float",
            fixed=True
        )
        assert info.total_size == 10
        assert info.ndim == 1
        assert not info.is_variable
    
    def test_fixed_2d_total_size(self):
        """Fixed 2D array calculates total size."""
        info = CArrayInfo(
            name="mat",
            shape=(3, 4),
            dtype="float",
            fixed=True
        )
        assert info.total_size == 12
        assert info.ndim == 2
    
    def test_fixed_3d_total_size(self):
        """Fixed 3D array calculates total size."""
        info = CArrayInfo(
            name="tensor",
            shape=(2, 3, 4),
            dtype="double",
            fixed=True
        )
        assert info.total_size == 24
        assert info.ndim == 3
    
    def test_variable_no_total_size(self):
        """Variable-length array has no total size."""
        info = CArrayInfo(
            name="arr",
            shape=("n",),
            dtype="float",
            fixed=False,
            counter="n"
        )
        assert info.total_size is None
        assert info.is_variable
    
    def test_strides_1d(self):
        """1D array has stride [1]."""
        info = CArrayInfo(name="arr", shape=(10,), dtype="float", fixed=True)
        assert info.get_strides() == [1]
    
    def test_strides_2d(self):
        """2D array (3,4) has strides [4, 1]."""
        info = CArrayInfo(name="mat", shape=(3, 4), dtype="float", fixed=True)
        assert info.get_strides() == [4, 1]
    
    def test_strides_3d(self):
        """3D array (2,3,4) has strides [12, 4, 1]."""
        info = CArrayInfo(name="tensor", shape=(2, 3, 4), dtype="float", fixed=True)
        assert info.get_strides() == [12, 4, 1]
    
    def test_strides_variable_raises(self):
        """Variable-length array cannot compute strides."""
        info = CArrayInfo(name="arr", shape=("n",), dtype="float", fixed=False)
        with pytest.raises(ValueError):
            info.get_strides()


# =============================================================================
# Test BranchTitleParser
# =============================================================================

class TestBranchTitleParser:
    """Test branch title parsing."""
    
    def test_parse_1d_fixed_float(self):
        """Parse arr[10]/F."""
        info = BranchTitleParser.parse("arr[10]/F")
        assert info is not None
        assert info.name == "arr"
        assert info.shape == (10,)
        assert info.dtype == "float"
        assert info.fixed is True
        assert info.counter is None
    
    def test_parse_1d_fixed_double(self):
        """Parse arr[20]/D."""
        info = BranchTitleParser.parse("arr[20]/D")
        assert info is not None
        assert info.dtype == "double"
        assert info.shape == (20,)
    
    def test_parse_1d_fixed_int(self):
        """Parse arr[5]/I."""
        info = BranchTitleParser.parse("arr[5]/I")
        assert info is not None
        assert info.dtype == "int"
    
    def test_parse_2d_fixed(self):
        """Parse mat[3][4]/F."""
        info = BranchTitleParser.parse("mat[3][4]/F")
        assert info is not None
        assert info.name == "mat"
        assert info.shape == (3, 4)
        assert info.fixed is True
        assert info.total_size == 12
    
    def test_parse_3d_fixed(self):
        """Parse tensor[2][3][4]/D."""
        info = BranchTitleParser.parse("tensor[2][3][4]/D")
        assert info is not None
        assert info.shape == (2, 3, 4)
        assert info.total_size == 24
    
    def test_parse_1d_variable(self):
        """Parse arr[n]/F - variable length."""
        info = BranchTitleParser.parse("arr[n]/F")
        assert info is not None
        assert info.shape == ("n",)
        assert info.fixed is False
        assert info.counter == "n"
    
    def test_parse_2d_hybrid(self):
        """Parse mat[n][3]/F - first dim variable, second fixed."""
        info = BranchTitleParser.parse("mat[n][3]/F")
        assert info is not None
        assert info.shape == ("n", 3)
        assert info.fixed is False
        assert info.counter == "n"
    
    def test_parse_variable_with_long_counter(self):
        """Parse arr[nTracks]/F."""
        info = BranchTitleParser.parse("arr[nTracks]/F")
        assert info is not None
        assert info.counter == "nTracks"
    
    def test_parse_with_override_name(self):
        """Branch name can be overridden."""
        info = BranchTitleParser.parse("arr[10]/F", branch_name="myArray")
        assert info.name == "myArray"
    
    def test_parse_invalid_no_type(self):
        """Invalid: no type specifier."""
        info = BranchTitleParser.parse("arr[10]")
        assert info is None
    
    def test_parse_invalid_no_dims(self):
        """Invalid: no dimensions."""
        info = BranchTitleParser.parse("arr/F")
        assert info is None
    
    def test_parse_scalar_returns_none(self):
        """Scalar branch returns None."""
        info = BranchTitleParser.parse("x/F")
        assert info is None
    
    def test_parse_whitespace_handling(self):
        """Handles whitespace in title."""
        info = BranchTitleParser.parse("  arr[10]/F  ")
        assert info is not None
        assert info.shape == (10,)


# =============================================================================
# Test CArraySchema
# =============================================================================

class TestCArraySchema:
    """Test CArraySchema container."""
    
    def test_add_and_get(self):
        """Add and retrieve arrays."""
        schema = CArraySchema()
        info = CArrayInfo(name="arr", shape=(10,), dtype="float", fixed=True)
        schema.add(info)
        
        assert "arr" in schema
        assert schema["arr"] == info
    
    def test_get_default(self):
        """Get with default for missing."""
        schema = CArraySchema()
        assert schema.get("missing") is None
        assert schema.get("missing", "default") == "default"
    
    def test_is_carray(self):
        """Check if name is C-array."""
        schema = CArraySchema()
        schema.add(CArrayInfo(name="arr", shape=(10,), dtype="float", fixed=True))
        
        assert schema.is_carray("arr")
        assert not schema.is_carray("other")
    
    def test_list_arrays(self):
        """List all array names."""
        schema = CArraySchema()
        schema.add(CArrayInfo(name="arr1", shape=(10,), dtype="float", fixed=True))
        schema.add(CArrayInfo(name="arr2", shape=(20,), dtype="double", fixed=True))
        
        names = schema.list_arrays()
        assert "arr1" in names
        assert "arr2" in names
    
    def test_list_fixed_and_variable(self):
        """Separate fixed and variable arrays."""
        schema = CArraySchema()
        schema.add(CArrayInfo(name="fixed1", shape=(10,), dtype="float", fixed=True))
        schema.add(CArrayInfo(name="var1", shape=("n",), dtype="float", fixed=False))
        schema.add(CArrayInfo(name="fixed2", shape=(3, 4), dtype="double", fixed=True))
        
        fixed = schema.list_fixed()
        variable = schema.list_variable()
        
        assert "fixed1" in fixed
        assert "fixed2" in fixed
        assert "var1" in variable
        assert len(fixed) == 2
        assert len(variable) == 1
    
    def test_counter_tracking(self):
        """Tracks counter relationships."""
        schema = CArraySchema()
        info = CArrayInfo(name="arr", shape=("n",), dtype="float", fixed=False, counter="n")
        schema.add(info)
        
        assert schema.counters["arr"] == "n"


# =============================================================================
# Test CArrayDetector (requires ROOT)
# =============================================================================

class TestCArrayDetectorWithROOT:
    """Test CArrayDetector with actual ROOT TTree."""
    
    @pytest.fixture(scope="class")
    def root_module(self):
        """Import ROOT."""
        ROOT = pytest.importorskip("ROOT")
        return ROOT
    
    @pytest.fixture(scope="class")
    def test_tree(self, root_module, tmp_path_factory):
        """Create TTree with various C-array types."""
        ROOT = root_module
        
        tmpdir = tmp_path_factory.mktemp("detector_test")
        filename = str(tmpdir / "detector_test.root")
        
        import uuid
        func_name = f"create_detector_test_tree_{uuid.uuid4().hex[:8]}"
        
        ROOT.gInterpreter.ProcessLine(f'''
            void {func_name}(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("tree", "test");
                
                // Fixed 1D
                Float_t arr1d[10];
                tree.Branch("arr1d", arr1d, "arr1d[10]/F");
                
                // Fixed 2D
                Float_t mat[3][4];
                tree.Branch("mat", mat, "mat[3][4]/F");
                
                // Fixed 3D
                Double_t tensor[2][3][4];
                tree.Branch("tensor", tensor, "tensor[2][3][4]/D");
                
                // Variable length with counter
                Int_t n;
                Float_t varr[100];
                tree.Branch("n", &n, "n/I");
                tree.Branch("varr", varr, "varr[n]/F");
                
                // Scalar (not C-array)
                Float_t x;
                tree.Branch("x", &x, "x/F");
                
                // Fill some data
                n = 5;
                for (int i = 0; i < 10; i++) arr1d[i] = i;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 4; j++)
                        mat[i][j] = i * 10 + j;
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 3; j++)
                        for (int k = 0; k < 4; k++)
                            tensor[i][j][k] = i * 100 + j * 10 + k;
                for (int i = 0; i < n; i++) varr[i] = i * 2;
                x = 3.14f;
                
                tree.Fill();
                tree.Write();
                f.Close();
            }}
        ''')
        getattr(ROOT, func_name)(filename)
        
        # Open and return tree
        f = ROOT.TFile.Open(filename)
        tree = f.Get("tree")
        
        yield tree, filename
        
        f.Close()
    
    def test_detect_fixed_1d(self, test_tree):
        """Detect fixed 1D array."""
        tree, _ = test_tree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        assert "arr1d" in schema
        info = schema["arr1d"]
        assert info.shape == (10,)
        assert info.dtype == "float"
        assert info.fixed is True
    
    def test_detect_fixed_2d(self, test_tree):
        """Detect fixed 2D array."""
        tree, _ = test_tree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        assert "mat" in schema
        info = schema["mat"]
        assert info.shape == (3, 4)
        assert info.fixed is True
        assert info.total_size == 12
    
    def test_detect_fixed_3d(self, test_tree):
        """Detect fixed 3D array."""
        tree, _ = test_tree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        assert "tensor" in schema
        info = schema["tensor"]
        assert info.shape == (2, 3, 4)
        assert info.dtype == "double"
        assert info.total_size == 24
    
    def test_detect_variable_with_counter(self, test_tree):
        """Detect variable-length array with counter (D3)."""
        tree, _ = test_tree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        assert "varr" in schema
        info = schema["varr"]
        assert info.fixed is False
        assert info.counter == "n"
    
    def test_scalar_not_detected(self, test_tree):
        """Scalar branch is not detected as C-array."""
        tree, _ = test_tree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        assert "x" not in schema
    
    def test_convenience_function(self, test_tree):
        """Test detect_carrays convenience function."""
        tree, _ = test_tree
        schema = detect_carrays(tree)
        
        assert "arr1d" in schema
        assert "mat" in schema
        assert "tensor" in schema
        assert "varr" in schema
    
    def test_detect_from_rdf(self, root_module, test_tree):
        """Detect from RDataFrame (limited)."""
        ROOT = root_module
        tree, filename = test_tree
        
        rdf = ROOT.RDataFrame("tree", filename)
        detector = CArrayDetector.from_rdf(rdf)
        schema = detector.detect()
        
        # RDF detection is limited - can only detect RVec columns
        # All C-arrays should be detected as they become RVec
        assert "arr1d" in schema or len(schema.arrays) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
