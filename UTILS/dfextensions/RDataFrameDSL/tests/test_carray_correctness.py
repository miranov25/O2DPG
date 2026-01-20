"""
Phase 13.4.DSL Bundle 2: ROOT Correctness Tests

Tests verify ACTUAL NUMERICAL CORRECTNESS using RDataFrame.
Uses mathematical invariants to validate:
- Row-major stride calculation
- Bounds checking (NaN on OOB)
- Negative index normalization
- Slice clamping behavior

Uses C++ macros for proper TTree creation with C-arrays.
"""

import pytest
import numpy as np
import tempfile
import os

try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

from RDataFrameDSL.ir_nodes_carray import (
    make_carray_element_access,
    make_carray_slice_access,
    make_carray_row_access,
    make_carray_column_access,
)
from RDataFrameDSL.ir_nodes_linalg import SliceParams
from RDataFrameDSL.backend_carray import generate_carray_code


def apply_carray_code(rdf, col_name: str, result):
    """Apply C-array code to RDataFrame with proper JIT declaration.
    
    FROZEN RULE: Must declare JIT functions before using in Define().
    """
    import ROOT
    if result.jit_declarations:
        ROOT.gInterpreter.Declare(result.jit_declarations)
    return rdf.Define(col_name, result.code)


def create_test_tree_1d_cpp(filename: str):
    """Create TTree with 1D C-array using C++ macro.
    
    Array: arr[10] where arr[i] = i * 10 + 1
    So: arr = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    """
    import ROOT
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_carray_tree_1d(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            float arr[10];
            tree.Branch("arr", arr, "arr[10]/F");
            
            // Fill with arr[i] = i * 10 + 1
            for (int i = 0; i < 10; i++) {{
                arr[i] = i * 10.0f + 1.0f;
            }}
            tree.Fill();
            
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_carray_tree_1d(filename)


def create_test_tree_2d_cpp(filename: str, rows: int = 3, cols: int = 4):
    """Create TTree with 2D C-array (stored as flat 1D).
    
    Values: mat[i * cols + j] = i * 100 + j * 10 + 1
    For 3x4: flat indices 0-11
    """
    import ROOT
    
    size = rows * cols
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_carray_tree_2d(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            float mat[{size}];
            tree.Branch("mat", mat, "mat[{size}]/F");
            
            // Fill with mat[i*cols + j] = i*100 + j*10 + 1
            for (int i = 0; i < {rows}; i++) {{
                for (int j = 0; j < {cols}; j++) {{
                    mat[i * {cols} + j] = i * 100.0f + j * 10.0f + 1.0f;
                }}
            }}
            tree.Fill();
            
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_carray_tree_2d(filename)


def create_test_tree_3d_cpp(filename: str, d0: int = 2, d1: int = 3, d2: int = 4):
    """Create TTree with 3D C-array (stored as flat 1D).
    
    Values: tensor[i*d1*d2 + j*d2 + k] = i*1000 + j*100 + k*10 + 1
    """
    import ROOT
    
    size = d0 * d1 * d2
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_carray_tree_3d(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            float tensor[{size}];
            tree.Branch("tensor", tensor, "tensor[{size}]/F");
            
            // Fill with tensor[i*d1*d2 + j*d2 + k] = i*1000 + j*100 + k*10 + 1
            for (int i = 0; i < {d0}; i++) {{
                for (int j = 0; j < {d1}; j++) {{
                    for (int k = 0; k < {d2}; k++) {{
                        int idx = i * {d1} * {d2} + j * {d2} + k;
                        tensor[idx] = i * 1000.0f + j * 100.0f + k * 10.0f + 1.0f;
                    }}
                }}
            }}
            tree.Fill();
            
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_carray_tree_3d(filename)


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestCArray1DCorrectness:
    """Test 1D C-array access produces correct numerical values."""
    
    @pytest.fixture
    def tree_file(self, tmp_path):
        """Create test TTree file."""
        filename = str(tmp_path / "test_1d.root")
        create_test_tree_1d_cpp(filename)
        return filename
    
    def test_element_access_correct_value(self, tree_file):
        """arr[5] returns correct value (expected: 51)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [5])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem5", result)
        values = rdf2.AsNumpy(["elem5"])["elem5"]
        
        # Expected: arr[5] = 5 * 10 + 1 = 51
        assert len(values) == 1
        assert abs(values[0] - 51.0) < 0.001
    
    def test_first_element_correct(self, tree_file):
        """arr[0] returns first element (expected: 1)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [0])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "first", result)
        values = rdf2.AsNumpy(["first"])["first"]
        
        # Expected: arr[0] = 0 * 10 + 1 = 1
        assert abs(values[0] - 1.0) < 0.001
    
    def test_last_element_correct(self, tree_file):
        """arr[9] returns last element (expected: 91)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [9])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "last", result)
        values = rdf2.AsNumpy(["last"])["last"]
        
        # Expected: arr[9] = 9 * 10 + 1 = 91
        assert abs(values[0] - 91.0) < 0.001
    
    def test_negative_index_correct(self, tree_file):
        """arr[-1] returns last element (expected: 91)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [-1])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg1", result)
        values = rdf2.AsNumpy(["neg1"])["neg1"]
        
        # arr[-1] should equal arr[9] = 91
        assert abs(values[0] - 91.0) < 0.001
    
    def test_negative_index_minus2(self, tree_file):
        """arr[-2] returns second-to-last (expected: 81)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [-2])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg2", result)
        values = rdf2.AsNumpy(["neg2"])["neg2"]
        
        # arr[-2] should equal arr[8] = 81
        assert abs(values[0] - 81.0) < 0.001
    
    def test_oob_returns_nan(self, tree_file):
        """arr[100] (out of bounds) returns NaN."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [100])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "oob", result)
        values = rdf2.AsNumpy(["oob"])["oob"]
        
        assert np.isnan(values[0])
    
    def test_negative_oob_returns_nan(self, tree_file):
        """arr[-100] (out of bounds negative) returns NaN."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("arr", "float", [(10, True)], [-100])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg_oob", result)
        values = rdf2.AsNumpy(["neg_oob"])["neg_oob"]
        
        assert np.isnan(values[0])


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestCArray2DCorrectness:
    """Test 2D C-array access produces correct numerical values."""
    
    @pytest.fixture
    def tree_file(self, tmp_path):
        """Create test TTree with 3x4 matrix."""
        filename = str(tmp_path / "test_2d.root")
        create_test_tree_2d_cpp(filename, rows=3, cols=4)
        return filename
    
    def test_element_0_0_correct(self, tree_file):
        """mat[0, 0] returns correct value (expected: 1)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [0, 0])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem00", result)
        values = rdf2.AsNumpy(["elem00"])["elem00"]
        
        # mat[0][0] = 0*100 + 0*10 + 1 = 1
        assert abs(values[0] - 1.0) < 0.001
    
    def test_element_1_2_correct(self, tree_file):
        """mat[1, 2] returns correct value (expected: 121)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [1, 2])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem12", result)
        values = rdf2.AsNumpy(["elem12"])["elem12"]
        
        # mat[1][2] = 1*100 + 2*10 + 1 = 121
        assert abs(values[0] - 121.0) < 0.001
    
    def test_element_2_3_correct(self, tree_file):
        """mat[2, 3] returns correct value (expected: 231)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [2, 3])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem23", result)
        values = rdf2.AsNumpy(["elem23"])["elem23"]
        
        # mat[2][3] = 2*100 + 3*10 + 1 = 231
        assert abs(values[0] - 231.0) < 0.001
    
    def test_row_major_stride_invariant(self, tree_file):
        """Verify row-major stride: mat[i,j] == flat[i*cols + j]."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        # Test multiple positions
        test_cases = [(0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3)]
        
        for i, j in test_cases:
            node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [i, j])
            result = generate_carray_code(node)
            
            rdf2 = apply_carray_code(rdf, f"elem_{i}_{j}", result)
            values = rdf2.AsNumpy([f"elem_{i}_{j}"])[f"elem_{i}_{j}"]
            
            # Expected value based on our encoding
            expected = i * 100 + j * 10 + 1
            assert abs(values[0] - expected) < 0.001, f"mat[{i},{j}] expected {expected}, got {values[0]}"
    
    def test_negative_index_row(self, tree_file):
        """mat[-1, 0] returns last row, first column."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [-1, 0])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg_row", result)
        values = rdf2.AsNumpy(["neg_row"])["neg_row"]
        
        # mat[-1][0] = mat[2][0] = 2*100 + 0*10 + 1 = 201
        assert abs(values[0] - 201.0) < 0.001
    
    def test_negative_index_col(self, tree_file):
        """mat[0, -1] returns first row, last column."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [0, -1])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg_col", result)
        values = rdf2.AsNumpy(["neg_col"])["neg_col"]
        
        # mat[0][-1] = mat[0][3] = 0*100 + 3*10 + 1 = 31
        assert abs(values[0] - 31.0) < 0.001
    
    def test_oob_row_returns_nan(self, tree_file):
        """mat[10, 0] (row OOB) returns NaN."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [10, 0])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "oob_row", result)
        values = rdf2.AsNumpy(["oob_row"])["oob_row"]
        
        assert np.isnan(values[0])
    
    def test_oob_col_returns_nan(self, tree_file):
        """mat[0, 10] (col OOB) returns NaN."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [0, 10])
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "oob_col", result)
        values = rdf2.AsNumpy(["oob_col"])["oob_col"]
        
        assert np.isnan(values[0])


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestCArray3DCorrectness:
    """Test 3D C-array access produces correct numerical values."""
    
    @pytest.fixture
    def tree_file(self, tmp_path):
        """Create test TTree with 2x3x4 tensor."""
        filename = str(tmp_path / "test_3d.root")
        create_test_tree_3d_cpp(filename, d0=2, d1=3, d2=4)
        return filename
    
    def test_element_0_0_0_correct(self, tree_file):
        """tensor[0,0,0] returns correct value (expected: 1)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            [0, 0, 0]
        )
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem000", result)
        values = rdf2.AsNumpy(["elem000"])["elem000"]
        
        # tensor[0][0][0] = 0*1000 + 0*100 + 0*10 + 1 = 1
        assert abs(values[0] - 1.0) < 0.001
    
    def test_element_1_2_3_correct(self, tree_file):
        """tensor[1,2,3] returns correct value (expected: 1231)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            [1, 2, 3]
        )
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "elem123", result)
        values = rdf2.AsNumpy(["elem123"])["elem123"]
        
        # tensor[1][2][3] = 1*1000 + 2*100 + 3*10 + 1 = 1231
        assert abs(values[0] - 1231.0) < 0.001
    
    def test_3d_row_major_stride_invariant(self, tree_file):
        """Verify 3D row-major stride: tensor[i,j,k] == flat[i*d1*d2 + j*d2 + k]."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        # Test multiple positions
        test_cases = [
            (0, 0, 0), (0, 0, 3), (0, 2, 0), (0, 2, 3),
            (1, 0, 0), (1, 1, 2), (1, 2, 3),
        ]
        
        for i, j, k in test_cases:
            node = make_carray_element_access(
                "tensor", "float",
                [(2, True), (3, True), (4, True)],
                [i, j, k]
            )
            result = generate_carray_code(node)
            
            rdf2 = apply_carray_code(rdf, f"elem_{i}_{j}_{k}", result)
            values = rdf2.AsNumpy([f"elem_{i}_{j}_{k}"])[f"elem_{i}_{j}_{k}"]
            
            # Expected value based on our encoding
            expected = i * 1000 + j * 100 + k * 10 + 1
            assert abs(values[0] - expected) < 0.001, \
                f"tensor[{i},{j},{k}] expected {expected}, got {values[0]}"
    
    def test_negative_indices_3d(self, tree_file):
        """tensor[-1, -1, -1] returns last element."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            [-1, -1, -1]
        )
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "neg111", result)
        values = rdf2.AsNumpy(["neg111"])["neg111"]
        
        # tensor[-1][-1][-1] = tensor[1][2][3] = 1231
        assert abs(values[0] - 1231.0) < 0.001
    
    def test_3d_oob_returns_nan(self, tree_file):
        """tensor[10, 0, 0] (OOB) returns NaN."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_element_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            [10, 0, 0]
        )
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "oob", result)
        values = rdf2.AsNumpy(["oob"])["oob"]
        
        assert np.isnan(values[0])


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestCArraySliceCorrectness:
    """Test slice operations produce correct results."""
    
    @pytest.fixture
    def tree_file(self, tmp_path):
        """Create test TTree."""
        filename = str(tmp_path / "test_slice.root")
        create_test_tree_1d_cpp(filename)
        return filename
    
    def test_slice_first_3_correct(self, tree_file):
        """arr[:3] returns first 3 elements."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=3))
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "slice3", result)
        
        # Check the size using Take
        size_check = rdf2.Define("slice_size", "(int)slice3.size()")
        sizes = size_check.AsNumpy(["slice_size"])["slice_size"]
        assert sizes[0] == 3
    
    def test_slice_from_5_correct(self, tree_file):
        """arr[5:] returns elements from index 5."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(start=5))
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "slice_from_5", result)
        size_check = rdf2.Define("slice_size", "(int)slice_from_5.size()")
        sizes = size_check.AsNumpy(["slice_size"])["slice_size"]
        
        # Should have 5 elements (indices 5,6,7,8,9)
        assert sizes[0] == 5
    
    def test_slice_oob_clamps(self, tree_file):
        """arr[:100] clamps to array size (returns all 10 elements)."""
        import ROOT
        
        rdf = ROOT.RDataFrame("tree", tree_file)
        
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=100))
        result = generate_carray_code(node)
        
        rdf2 = apply_carray_code(rdf, "slice_oob", result)
        size_check = rdf2.Define("slice_size", "(int)slice_oob.size()")
        sizes = size_check.AsNumpy(["slice_size"])["slice_size"]
        
        # Should clamp to 10 elements
        assert sizes[0] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
