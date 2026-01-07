#!/usr/bin/env python3
"""
ND Slicing Strategies for Flattened C-Arrays in ROOT

Since ROOT flattens mat[3][4] to RVec<float> of size 12,
we need different strategies for different slice patterns.

Run with: python tests/test_nd_slicing_strategies.py
"""

def main():
    import ROOT
    import tempfile
    import os
    import hashlib
    
    print("=" * 70)
    print("ND SLICING STRATEGIES FOR FLATTENED ARRAYS")
    print("=" * 70)
    
    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "test_nd.root")
    
    # Create test tree with 2D and 3D arrays
    ROOT.gInterpreter.ProcessLine(f'''
        void create_nd_tree(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            // 2D: mat[3][4] - values = row*100 + col*10 + 1
            Float_t mat[3][4];
            tree.Branch("mat", mat, "mat[3][4]/F");
            
            // 3D: tensor[2][3][4] - values = plane*1000 + row*100 + col*10 + 1
            Float_t tensor[2][3][4];
            tree.Branch("tensor", tensor, "tensor[2][3][4]/F");
            
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    mat[i][j] = i * 100.0f + j * 10.0f + 1.0f;
            
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        tensor[i][j][k] = i * 1000.0f + j * 100.0f + k * 10.0f + 1.0f;
            
            tree.Fill();
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_nd_tree(filename)
    
    rdf = ROOT.RDataFrame("tree", filename)
    
    # =========================================================================
    # STRATEGY 1: Element Access (Direct Index Calculation)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 1: Element Access - mat[i,j] → mat[i*cols+j]")
    print("-" * 70)
    
    # mat[1,2] → mat[1*4+2] = mat[6]
    rdf2 = rdf.Define("elem_1_2", "mat[1*4+2]")
    val = rdf2.AsNumpy(["elem_1_2"])["elem_1_2"][0]
    expected = 1*100 + 2*10 + 1  # 121
    print(f"  mat[1,2] = mat[6] = {val} (expected {expected}) {'✅' if val == expected else '❌'}")
    
    # tensor[1,2,3] → tensor[1*3*4 + 2*4 + 3] = tensor[23]
    rdf2 = rdf.Define("elem_1_2_3", "tensor[1*3*4 + 2*4 + 3]")
    val = rdf2.AsNumpy(["elem_1_2_3"])["elem_1_2_3"][0]
    expected = 1*1000 + 2*100 + 3*10 + 1  # 1231
    print(f"  tensor[1,2,3] = tensor[23] = {val} (expected {expected}) {'✅' if val == expected else '❌'}")
    
    # =========================================================================
    # STRATEGY 2: Row Access (Contiguous - Use RVec Slice)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 2: Row Access - mat[i,:] → Contiguous slice")
    print("-" * 70)
    
    # mat[1,:] = elements at indices 4,5,6,7
    # Declare helper for row extraction
    row_code = '''
#ifndef GET_ROW_2D_DEFINED
#define GET_ROW_2D_DEFINED
ROOT::RVec<float> get_row_2d(const ROOT::RVec<float>& arr, int cols, int row_idx) {
    int offset = row_idx * cols;
    ROOT::RVec<float> result(cols);
    for (int j = 0; j < cols; ++j) {
        result[j] = arr[offset + j];
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(row_code)
    
    rdf2 = rdf.Define("row_1", "get_row_2d(mat, 4, 1)")
    rdf3 = rdf2.Define("row_1_size", "(int)row_1.size()")
    
    # Get values directly
    vals = rdf2.AsNumpy(["row_1"])["row_1"][0]
    print(f"  mat[1,:] = {list(vals)}")
    print(f"  Expected: [101, 111, 121, 131] {'✅' if list(vals) == [101, 111, 121, 131] else '❌'}")
    
    # =========================================================================
    # STRATEGY 3: Column Access (Strided - Need Loop)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 3: Column Access - mat[:,j] → Strided (need loop)")
    print("-" * 70)
    
    col_code = '''
#ifndef GET_COL_2D_DEFINED
#define GET_COL_2D_DEFINED
ROOT::RVec<float> get_col_2d(const ROOT::RVec<float>& arr, int rows, int cols, int col_idx) {
    ROOT::RVec<float> result(rows);
    for (int i = 0; i < rows; ++i) {
        result[i] = arr[i * cols + col_idx];
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(col_code)
    
    rdf2 = rdf.Define("col_2", "get_col_2d(mat, 3, 4, 2)")
    vals = rdf2.AsNumpy(["col_2"])["col_2"][0]
    print(f"  mat[:,2] = {list(vals)}")
    print(f"  Expected: [21, 121, 221] {'✅' if list(vals) == [21, 121, 221] else '❌'}")
    
    # =========================================================================
    # STRATEGY 4: 3D Plane Access - tensor[i,:,:]
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 4: 3D Plane Access - tensor[i,:,:] → 2D slice")
    print("-" * 70)
    
    plane_code = '''
#ifndef GET_PLANE_3D_DEFINED
#define GET_PLANE_3D_DEFINED
ROOT::RVec<float> get_plane_3d(const ROOT::RVec<float>& arr, int d1, int d2, int plane_idx) {
    // Returns flattened plane (d1 x d2 elements)
    int offset = plane_idx * d1 * d2;
    ROOT::RVec<float> result(d1 * d2);
    for (int i = 0; i < d1 * d2; ++i) {
        result[i] = arr[offset + i];
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(plane_code)
    
    rdf2 = rdf.Define("plane_1", "get_plane_3d(tensor, 3, 4, 1)")
    vals = rdf2.AsNumpy(["plane_1"])["plane_1"][0]
    vals_list = list(vals)
    print(f"  tensor[1,:,:] size = {len(vals_list)} (expected 12)")
    print(f"  First 4 values (row 0): {vals_list[:4]}")
    print(f"  Expected: [1001, 1011, 1021, 1031] {'✅' if vals_list[:4] == [1001, 1011, 1021, 1031] else '❌'}")
    
    # =========================================================================
    # STRATEGY 5: 3D Row Access - tensor[i,j,:]
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 5: 3D Row Access - tensor[i,j,:] → Contiguous")
    print("-" * 70)
    
    row3d_code = '''
#ifndef GET_ROW_3D_DEFINED
#define GET_ROW_3D_DEFINED
ROOT::RVec<float> get_row_3d(const ROOT::RVec<float>& arr, int d1, int d2, int plane_idx, int row_idx) {
    int offset = plane_idx * d1 * d2 + row_idx * d2;
    ROOT::RVec<float> result(d2);
    for (int k = 0; k < d2; ++k) {
        result[k] = arr[offset + k];
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(row3d_code)
    
    rdf2 = rdf.Define("row_1_2", "get_row_3d(tensor, 3, 4, 1, 2)")
    vals = rdf2.AsNumpy(["row_1_2"])["row_1_2"][0]
    print(f"  tensor[1,2,:] = {list(vals)}")
    print(f"  Expected: [1201, 1211, 1221, 1231] {'✅' if list(vals) == [1201, 1211, 1221, 1231] else '❌'}")
    
    # =========================================================================
    # STRATEGY 6: 3D Column Access - tensor[i,:,k] → Strided
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 6: 3D Column Access - tensor[i,:,k] → Strided")
    print("-" * 70)
    
    col3d_code = '''
#ifndef GET_COL_3D_DEFINED
#define GET_COL_3D_DEFINED
ROOT::RVec<float> get_col_3d(const ROOT::RVec<float>& arr, int d1, int d2, int plane_idx, int col_idx) {
    int plane_offset = plane_idx * d1 * d2;
    ROOT::RVec<float> result(d1);
    for (int j = 0; j < d1; ++j) {
        result[j] = arr[plane_offset + j * d2 + col_idx];
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(col3d_code)
    
    rdf2 = rdf.Define("col_1_3", "get_col_3d(tensor, 3, 4, 1, 3)")
    vals = rdf2.AsNumpy(["col_1_3"])["col_1_3"][0]
    print(f"  tensor[1,:,3] = {list(vals)}")
    print(f"  Expected: [1031, 1131, 1231] {'✅' if list(vals) == [1031, 1131, 1231] else '❌'}")
    
    # =========================================================================
    # STRATEGY 7: Subarray/Range Slicing - mat[0:2, 1:3]
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 7: Subarray Slicing - mat[0:2, 1:3]")
    print("-" * 70)
    
    subarray_code = '''
#ifndef GET_SUBARRAY_2D_DEFINED
#define GET_SUBARRAY_2D_DEFINED
ROOT::RVec<float> get_subarray_2d(const ROOT::RVec<float>& arr, int cols,
                                   int row_start, int row_end,
                                   int col_start, int col_end) {
    int out_rows = row_end - row_start;
    int out_cols = col_end - col_start;
    ROOT::RVec<float> result(out_rows * out_cols);
    int idx = 0;
    for (int i = row_start; i < row_end; ++i) {
        for (int j = col_start; j < col_end; ++j) {
            result[idx++] = arr[i * cols + j];
        }
    }
    return result;
}
#endif
'''
    ROOT.gInterpreter.Declare(subarray_code)
    
    rdf2 = rdf.Define("sub", "get_subarray_2d(mat, 4, 0, 2, 1, 3)")
    vals = rdf2.AsNumpy(["sub"])["sub"][0]
    print(f"  mat[0:2, 1:3] = {list(vals)}")
    print(f"  Expected: [11, 21, 111, 121] (2x2 subarray flattened)")
    expected = [11, 21, 111, 121]
    print(f"  {'✅' if list(vals) == expected else '❌'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: ND SLICING IMPLEMENTATION STRATEGIES")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ OPERATION          │ PATTERN      │ IMPLEMENTATION                  │
├─────────────────────────────────────────────────────────────────────┤
│ mat[i,j]           │ Element      │ Direct: mat[i*cols+j]           │
│ tensor[i,j,k]      │ Element      │ Direct: tensor[i*d1*d2+j*d2+k]  │
├─────────────────────────────────────────────────────────────────────┤
│ mat[i,:]           │ Row (contig) │ Loop from offset (simple)       │
│ tensor[i,j,:]      │ Row (contig) │ Loop from offset (simple)       │
├─────────────────────────────────────────────────────────────────────┤
│ mat[:,j]           │ Col (stride) │ Loop with stride                │
│ tensor[i,:,k]      │ Col (stride) │ Loop with stride                │
├─────────────────────────────────────────────────────────────────────┤
│ tensor[i,:,:]      │ Plane        │ Copy contiguous block           │
│ mat[a:b, c:d]      │ Subarray     │ Nested loop                     │
└─────────────────────────────────────────────────────────────────────┘

CODE GENERATION APPROACH:
1. Parse slice expression to determine pattern
2. Generate appropriate helper function based on pattern
3. Use content-hash for function naming (avoid redeclaration)
4. Call function in RDataFrame.Define()

FUNCTION SIGNATURES (all use RVec, not T*):
- Element:   float get_elem_Nd(const RVec<float>& arr, dims..., indices...)
- Row:       RVec<float> get_row_Nd(const RVec<float>& arr, dims..., fixed_indices...)
- Column:    RVec<float> get_col_Nd(const RVec<float>& arr, dims..., fixed_indices...)
- Plane:     RVec<float> get_plane_Nd(const RVec<float>& arr, dims..., plane_idx)
- Subarray:  RVec<float> get_sub_Nd(const RVec<float>& arr, dims..., ranges...)
""")


if __name__ == "__main__":
    main()
