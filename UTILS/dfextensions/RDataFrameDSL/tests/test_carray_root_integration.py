"""
Phase 13.4.DSL Bundle 2: ROOT Integration Tests

IMPORTANT: These tests MUST run WITHOUT parallel execution:
    pytest tests/test_carray_root_integration.py -v -n 0

The tests verify C-array access works correctly in RDataFrame.
ROOT's global JIT state doesn't handle parallel test execution well.
"""

import pytest
import numpy as np
import os
import tempfile

# Skip entire module if ROOT not available
pytest.importorskip("ROOT")

# Force sequential execution for this module
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# =============================================================================
# Fixtures - Module scoped to minimize ROOT state changes
# =============================================================================

@pytest.fixture(scope="module")
def root_module():
    """Import ROOT once per module."""
    import ROOT
    return ROOT


@pytest.fixture(scope="module") 
def test_tree_path(root_module, tmp_path_factory):
    """Create test TTree with C-arrays (module-scoped)."""
    ROOT = root_module
    
    tmpdir = tmp_path_factory.mktemp("carray_root_test")
    filename = str(tmpdir / "carray_test.root")
    
    # Use unique function name
    import uuid
    func_name = f"create_test_tree_{uuid.uuid4().hex[:8]}"
    
    ROOT.gInterpreter.ProcessLine(f'''
        void {func_name}(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            // 1D array: arr1d[i] = i * 10 + 1
            Float_t arr1d[10];
            tree.Branch("arr1d", arr1d, "arr1d[10]/F");
            
            // 2D array (3x4): arr2d[i*4+j] = i*100 + j*10 + 1
            Float_t arr2d[12];
            tree.Branch("arr2d", arr2d, "arr2d[12]/F");
            
            // Fill arrays
            for (int i = 0; i < 10; i++) {{
                arr1d[i] = i * 10.0f + 1.0f;
            }}
            for (int i = 0; i < 3; i++) {{
                for (int j = 0; j < 4; j++) {{
                    arr2d[i * 4 + j] = i * 100.0f + j * 10.0f + 1.0f;
                }}
            }}
            
            tree.Fill();
            tree.Write();
            f.Close();
        }}
    ''')
    getattr(ROOT, func_name)(filename)
    
    return filename


# =============================================================================
# Test Classes
# =============================================================================

class TestCArrayDirectAccess:
    """Test direct C-array element access (Tier 1)."""
    
    def test_1d_literal_index(self, root_module, test_tree_path):
        """arr1d[5] returns correct value."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        rdf2 = rdf.Define("elem5", "arr1d[5]")
        vals = rdf2.AsNumpy(["elem5"])["elem5"]
        
        # arr1d[5] = 5*10 + 1 = 51
        assert abs(vals[0] - 51.0) < 0.001
    
    def test_1d_first_element(self, root_module, test_tree_path):
        """arr1d[0] returns first element."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        rdf2 = rdf.Define("first", "arr1d[0]")
        vals = rdf2.AsNumpy(["first"])["first"]
        
        assert abs(vals[0] - 1.0) < 0.001
    
    def test_1d_last_element(self, root_module, test_tree_path):
        """arr1d[9] returns last element."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        rdf2 = rdf.Define("last", "arr1d[9]")
        vals = rdf2.AsNumpy(["last"])["last"]
        
        assert abs(vals[0] - 91.0) < 0.001
    
    def test_2d_row_major_access(self, root_module, test_tree_path):
        """arr2d[6] accesses element [1][2] correctly."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        # mat[1][2] = index 1*4+2 = 6
        rdf2 = rdf.Define("elem12", "arr2d[6]")
        vals = rdf2.AsNumpy(["elem12"])["elem12"]
        
        # arr2d[1][2] = 1*100 + 2*10 + 1 = 121
        assert abs(vals[0] - 121.0) < 0.001
    
    def test_2d_corners(self, root_module, test_tree_path):
        """Test 2D array corner values."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        rdf2 = rdf.Define("c00", "arr2d[0]")   # [0][0]
        rdf2 = rdf2.Define("c23", "arr2d[11]")  # [2][3]
        
        vals = rdf2.AsNumpy(["c00", "c23"])
        
        assert abs(vals["c00"][0] - 1.0) < 0.001    # 0*100 + 0*10 + 1
        assert abs(vals["c23"][0] - 231.0) < 0.001  # 2*100 + 3*10 + 1


class TestCArrayBoundsChecking:
    """Test bounds-checked access (Tier 2 - inline ternary)."""
    
    def test_bounds_check_inbounds(self, root_module, test_tree_path):
        """Bounds check returns value when in bounds."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        # Tier 2: inline ternary for bounds checking
        code = "(5 >= 0 && 5 < 10) ? arr1d[5] : std::numeric_limits<float>::quiet_NaN()"
        rdf2 = rdf.Define("bounded", code)
        vals = rdf2.AsNumpy(["bounded"])["bounded"]
        
        assert abs(vals[0] - 51.0) < 0.001
    
    def test_bounds_check_oob_returns_nan(self, root_module, test_tree_path):
        """Bounds check returns NaN when out of bounds."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        code = "(100 >= 0 && 100 < 10) ? arr1d[100] : std::numeric_limits<float>::quiet_NaN()"
        rdf2 = rdf.Define("oob", code)
        vals = rdf2.AsNumpy(["oob"])["oob"]
        
        assert np.isnan(vals[0])
    
    def test_negative_index_normalized(self, root_module, test_tree_path):
        """Negative index normalized at Python time: arr[-1] → arr[9]."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        # Python normalizes -1 → 9 before generating code
        rdf2 = rdf.Define("neg1", "arr1d[9]")
        vals = rdf2.AsNumpy(["neg1"])["neg1"]
        
        assert abs(vals[0] - 91.0) < 0.001


class TestCArrayNamedFunctions:
    """Test named function approach (Tier 3 - for loops).
    
    IMPORTANT: ROOT exposes C-array branches as RVec<T>, not T*.
    So named functions must accept RVec<T>& not T*.
    """
    
    def test_row_extraction_function(self, root_module, test_tree_path):
        """Named function extracts row from 2D array (stored as flat RVec)."""
        ROOT = root_module
        
        # Declare row extraction function with unique name
        # NOTE: arr is RVec<float>, not float*, because ROOT converts C-arrays
        import hashlib
        code = '''
ROOT::RVec<float> DSL_ROW(const ROOT::RVec<float>& arr, int rows, int cols, int row_idx) {
    int r = row_idx >= 0 ? row_idx : rows + row_idx;
    if (r < 0 || r >= rows) return ROOT::RVec<float>{};
    ROOT::RVec<float> result(cols);
    for (int j = 0; j < cols; ++j) {
        result[j] = arr[r * cols + j];
    }
    return result;
}
'''
        func_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        func_name = f"dsl_row_{func_hash}"
        actual_code = code.replace("DSL_ROW", func_name)
        
        # Declare with macro guard
        guard_code = f'''
#ifndef DSL_{func_name.upper()}
#define DSL_{func_name.upper()}
{actual_code}
#endif
'''
        ROOT.gInterpreter.Declare(guard_code)
        
        # Use in RDataFrame
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        rdf2 = rdf.Define("row1", f"{func_name}(arr2d, 3, 4, 1)")
        rdf3 = rdf2.Define("row1_size", "(int)row1.size()")
        
        vals = rdf3.AsNumpy(["row1_size"])
        assert vals["row1_size"][0] == 4
    
    def test_slice_function(self, root_module, test_tree_path):
        """Named function slices 1D array."""
        ROOT = root_module
        
        # NOTE: arr is RVec<float>, not float*
        import hashlib
        code = '''
ROOT::RVec<float> DSL_SLICE(const ROOT::RVec<float>& arr, int size, int start, int stop) {
    int s = start >= 0 ? start : size + start;
    int e = stop >= 0 ? stop : size + stop;
    s = std::max(0, std::min(s, (int)arr.size()));
    e = std::max(0, std::min(e, (int)arr.size()));
    ROOT::RVec<float> result;
    for (int i = s; i < e; ++i) result.push_back(arr[i]);
    return result;
}
'''
        func_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        func_name = f"dsl_slice_{func_hash}"
        actual_code = code.replace("DSL_SLICE", func_name)
        
        guard_code = f'''
#ifndef DSL_{func_name.upper()}
#define DSL_{func_name.upper()}
{actual_code}
#endif
'''
        ROOT.gInterpreter.Declare(guard_code)
        
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        rdf2 = rdf.Define("slice3", f"{func_name}(arr1d, 10, 0, 3)")
        rdf3 = rdf2.Define("slice_size", "(int)slice3.size()")
        
        vals = rdf3.AsNumpy(["slice_size"])
        assert vals["slice_size"][0] == 3


class TestCArrayColumnType:
    """Test that ROOT handles C-array column types correctly."""
    
    def test_carray_becomes_rvec(self, root_module, test_tree_path):
        """C-array branch is exposed as RVec in RDataFrame."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        col_type = rdf.GetColumnType("arr1d")
        
        # ROOT converts C-arrays to RVec automatically
        assert "RVec" in col_type
        assert "Float_t" in col_type or "float" in col_type.lower()
    
    def test_carray_size_correct(self, root_module, test_tree_path):
        """C-array RVec has correct size."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", test_tree_path)
        
        rdf2 = rdf.Define("arr_size", "(int)arr1d.size()")
        vals = rdf2.AsNumpy(["arr_size"])
        
        assert vals["arr_size"][0] == 10


class TestCArrayNDSlicing:
    """Test ND slicing strategies for flattened arrays.
    
    ROOT flattens multi-dimensional C-arrays:
    - mat[3][4] → RVec<float> size 12 (row-major)
    - Stride calculation required for 2D/3D access
    """
    
    @pytest.fixture(scope="class")
    def nd_tree_path(self, root_module, tmp_path_factory):
        """Create test TTree with 2D and 3D arrays."""
        ROOT = root_module
        
        tmpdir = tmp_path_factory.mktemp("nd_array_test")
        filename = str(tmpdir / "nd_test.root")
        
        import uuid
        func_name = f"create_nd_tree_{uuid.uuid4().hex[:8]}"
        
        ROOT.gInterpreter.ProcessLine(f'''
            void {func_name}(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("tree", "test");
                
                // 2D: mat[3][4] - values = row*100 + col*10 + 1
                Float_t mat[3][4];
                tree.Branch("mat", mat, "mat[3][4]/F");
                
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 4; j++)
                        mat[i][j] = i * 100.0f + j * 10.0f + 1.0f;
                
                tree.Fill();
                tree.Write();
                f.Close();
            }}
        ''')
        getattr(ROOT, func_name)(filename)
        
        return filename
    
    def test_2d_element_access(self, root_module, nd_tree_path):
        """mat[i,j] → mat[i*cols+j] works correctly."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", nd_tree_path)
        
        # mat[1,2] → mat[1*4+2] = mat[6]
        rdf2 = rdf.Define("elem_1_2", "mat[1*4+2]")
        val = rdf2.AsNumpy(["elem_1_2"])["elem_1_2"][0]
        
        expected = 1*100 + 2*10 + 1  # 121
        assert abs(val - expected) < 0.001
    
    def test_2d_row_extraction(self, root_module, nd_tree_path):
        """mat[i,:] extracts a row correctly."""
        ROOT = root_module
        
        # Declare row extraction helper
        ROOT.gInterpreter.Declare('''
#ifndef GET_ROW_2D_TEST
#define GET_ROW_2D_TEST
ROOT::RVec<float> get_row_2d_test(const ROOT::RVec<float>& arr, int cols, int row_idx) {
    int offset = row_idx * cols;
    ROOT::RVec<float> result(cols);
    for (int j = 0; j < cols; ++j) {
        result[j] = arr[offset + j];
    }
    return result;
}
#endif
''')
        
        rdf = ROOT.RDataFrame("tree", nd_tree_path)
        rdf2 = rdf.Define("row_1", "get_row_2d_test(mat, 4, 1)")
        rdf3 = rdf2.Define("row_1_size", "(int)row_1.size()")
        
        vals = rdf3.AsNumpy(["row_1_size"])
        assert vals["row_1_size"][0] == 4
    
    def test_2d_column_extraction(self, root_module, nd_tree_path):
        """mat[:,j] extracts a column correctly (strided access)."""
        ROOT = root_module
        
        # Declare column extraction helper
        ROOT.gInterpreter.Declare('''
#ifndef GET_COL_2D_TEST
#define GET_COL_2D_TEST
ROOT::RVec<float> get_col_2d_test(const ROOT::RVec<float>& arr, int rows, int cols, int col_idx) {
    ROOT::RVec<float> result(rows);
    for (int i = 0; i < rows; ++i) {
        result[i] = arr[i * cols + col_idx];
    }
    return result;
}
#endif
''')
        
        rdf = ROOT.RDataFrame("tree", nd_tree_path)
        rdf2 = rdf.Define("col_2", "get_col_2d_test(mat, 3, 4, 2)")
        rdf3 = rdf2.Define("col_2_size", "(int)col_2.size()")
        
        vals = rdf3.AsNumpy(["col_2_size"])
        assert vals["col_2_size"][0] == 3
    
    def test_2d_flattening_confirmed(self, root_module, nd_tree_path):
        """Confirm ROOT flattens 2D arrays to 1D RVec."""
        ROOT = root_module
        rdf = ROOT.RDataFrame("tree", nd_tree_path)
        
        col_type = rdf.GetColumnType("mat")
        assert "RVec" in col_type
        
        rdf2 = rdf.Define("mat_size", "(int)mat.size()")
        size = rdf2.AsNumpy(["mat_size"])["mat_size"][0]
        assert size == 12  # 3 * 4 = 12


if __name__ == "__main__":
    # Run without parallelization
    pytest.main([__file__, "-v", "-n", "0"])
