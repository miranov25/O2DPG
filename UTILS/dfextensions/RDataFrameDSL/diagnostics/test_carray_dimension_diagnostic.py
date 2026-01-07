#!/usr/bin/env python3
"""
Diagnostic: How does ROOT handle 2D/3D C-arrays in RDataFrame?

Questions to answer:
1. Does mat[3][4] become RVec<float> of size 12 (flattened)?
2. Is stride calculation (i*cols+j) still needed?
3. What about 3D arrays?

Run with: python tests/test_carray_dimension_diagnostic.py
"""

def main():
    import ROOT
    import tempfile
    import os
    
    print("=" * 70)
    print("C-ARRAY DIMENSION DIAGNOSTIC")
    print("=" * 70)
    
    tmpdir = tempfile.mkdtemp()
    
    # =========================================================================
    # TEST 1: True 2D C-array (C-style declaration)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: True 2D C-array - float mat[3][4]")
    print("=" * 70)
    
    filename_2d = os.path.join(tmpdir, "test_2d.root")
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_true_2d_tree(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            // True 2D array in C++
            Float_t mat[3][4];
            tree.Branch("mat", mat, "mat[3][4]/F");
            
            // Fill with mat[i][j] = i*100 + j*10 + 1
            for (int i = 0; i < 3; i++) {{
                for (int j = 0; j < 4; j++) {{
                    mat[i][j] = i * 100.0f + j * 10.0f + 1.0f;
                }}
            }}
            tree.Fill();
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_true_2d_tree(filename_2d)
    
    # Inspect the branch
    f = ROOT.TFile(filename_2d)
    tree = f.Get("tree")
    branch = tree.GetBranch("mat")
    print(f"  Branch title: {branch.GetTitle()}")
    
    leaf = tree.GetLeaf("mat")
    print(f"  Leaf type: {leaf.GetTypeName()}")
    print(f"  Leaf len: {leaf.GetLen()}")
    f.Close()
    
    # Check RDataFrame type
    rdf = ROOT.RDataFrame("tree", filename_2d)
    col_type = rdf.GetColumnType("mat")
    print(f"  RDF column type: {col_type}")
    
    # Check size
    rdf2 = rdf.Define("mat_size", "(int)mat.size()")
    size = rdf2.AsNumpy(["mat_size"])["mat_size"][0]
    print(f"  RVec size: {size} (expected: 12 if flattened)")
    
    # Check if we can access mat[6] to get mat[1][2]
    rdf3 = rdf.Define("elem_1_2", "mat[6]")  # 1*4+2 = 6
    val = rdf3.AsNumpy(["elem_1_2"])["elem_1_2"][0]
    expected = 1*100 + 2*10 + 1  # = 121
    print(f"  mat[6] (should be mat[1][2]): {val} (expected: {expected})")
    print(f"  ✅ FLATTENED" if abs(val - expected) < 0.001 else f"  ❌ NOT FLATTENED")
    
    # =========================================================================
    # TEST 2: Flat 1D array with 2D semantics (what we've been testing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Flat 1D array with 2D semantics - float mat[12]")
    print("=" * 70)
    
    filename_flat = os.path.join(tmpdir, "test_flat.root")
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_flat_2d_tree(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            // Flat 1D array
            Float_t mat[12];
            tree.Branch("mat", mat, "mat[12]/F");
            
            // Fill as if 2D: mat[i*4+j] = i*100 + j*10 + 1
            for (int i = 0; i < 3; i++) {{
                for (int j = 0; j < 4; j++) {{
                    mat[i * 4 + j] = i * 100.0f + j * 10.0f + 1.0f;
                }}
            }}
            tree.Fill();
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_flat_2d_tree(filename_flat)
    
    rdf = ROOT.RDataFrame("tree", filename_flat)
    col_type = rdf.GetColumnType("mat")
    print(f"  RDF column type: {col_type}")
    
    rdf2 = rdf.Define("mat_size", "(int)mat.size()")
    size = rdf2.AsNumpy(["mat_size"])["mat_size"][0]
    print(f"  RVec size: {size}")
    
    # =========================================================================
    # TEST 3: True 3D C-array
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: True 3D C-array - float tensor[2][3][4]")
    print("=" * 70)
    
    filename_3d = os.path.join(tmpdir, "test_3d.root")
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_true_3d_tree(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            // True 3D array in C++
            Float_t tensor[2][3][4];
            tree.Branch("tensor", tensor, "tensor[2][3][4]/F");
            
            // Fill with tensor[i][j][k] = i*1000 + j*100 + k*10 + 1
            for (int i = 0; i < 2; i++) {{
                for (int j = 0; j < 3; j++) {{
                    for (int k = 0; k < 4; k++) {{
                        tensor[i][j][k] = i * 1000.0f + j * 100.0f + k * 10.0f + 1.0f;
                    }}
                }}
            }}
            tree.Fill();
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_true_3d_tree(filename_3d)
    
    # Inspect the branch
    f = ROOT.TFile(filename_3d)
    tree = f.Get("tree")
    branch = tree.GetBranch("tensor")
    print(f"  Branch title: {branch.GetTitle()}")
    
    leaf = tree.GetLeaf("tensor")
    print(f"  Leaf type: {leaf.GetTypeName()}")
    print(f"  Leaf len: {leaf.GetLen()}")
    f.Close()
    
    # Check RDataFrame type
    rdf = ROOT.RDataFrame("tree", filename_3d)
    col_type = rdf.GetColumnType("tensor")
    print(f"  RDF column type: {col_type}")
    
    # Check size
    rdf2 = rdf.Define("tensor_size", "(int)tensor.size()")
    size = rdf2.AsNumpy(["tensor_size"])["tensor_size"][0]
    print(f"  RVec size: {size} (expected: 24 if flattened)")
    
    # Check if we can access tensor[17] to get tensor[1][1][1]
    # Index = 1*3*4 + 1*4 + 1 = 12 + 4 + 1 = 17
    rdf3 = rdf.Define("elem_1_1_1", "tensor[17]")
    val = rdf3.AsNumpy(["elem_1_1_1"])["elem_1_1_1"][0]
    expected = 1*1000 + 1*100 + 1*10 + 1  # = 1111
    print(f"  tensor[17] (should be tensor[1][1][1]): {val} (expected: {expected})")
    print(f"  ✅ FLATTENED" if abs(val - expected) < 0.001 else f"  ❌ NOT FLATTENED")
    
    # =========================================================================
    # TEST 4: Variable-length array (counter branch pattern)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Variable-length array with counter branch")
    print("=" * 70)
    
    filename_var = os.path.join(tmpdir, "test_var.root")
    
    ROOT.gInterpreter.ProcessLine(f'''
        void create_var_length_tree(const char* fname) {{
            TFile f(fname, "RECREATE");
            TTree tree("tree", "test");
            
            Int_t n;
            Float_t arr[100];  // Max size
            
            tree.Branch("n", &n, "n/I");
            tree.Branch("arr", arr, "arr[n]/F");  // Variable length!
            
            // Event 1: 5 elements
            n = 5;
            for (int i = 0; i < n; i++) arr[i] = i * 10.0f + 1.0f;
            tree.Fill();
            
            // Event 2: 3 elements
            n = 3;
            for (int i = 0; i < n; i++) arr[i] = i * 10.0f + 100.0f;
            tree.Fill();
            
            // Event 3: 8 elements
            n = 8;
            for (int i = 0; i < n; i++) arr[i] = i * 10.0f + 200.0f;
            tree.Fill();
            
            tree.Write();
            f.Close();
        }}
    ''')
    ROOT.create_var_length_tree(filename_var)
    
    # Inspect the branch
    f = ROOT.TFile(filename_var)
    tree = f.Get("tree")
    branch = tree.GetBranch("arr")
    print(f"  Branch title: {branch.GetTitle()}")
    f.Close()
    
    # Check RDataFrame
    rdf = ROOT.RDataFrame("tree", filename_var)
    col_type = rdf.GetColumnType("arr")
    print(f"  RDF column type: {col_type}")
    
    # Check sizes per event
    rdf2 = rdf.Define("arr_size", "(int)arr.size()")
    sizes = rdf2.AsNumpy(["arr_size"])["arr_size"]
    print(f"  RVec sizes per event: {list(sizes)} (expected: [5, 3, 8])")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS:

1. True 2D arrays (float mat[3][4]) are FLATTENED to RVec<float> of size 12
   - Access requires stride calculation: mat[i,j] → mat[i*cols+j]
   
2. True 3D arrays (float tensor[2][3][4]) are FLATTENED to RVec<float> of size 24
   - Access requires stride calculation: tensor[i,j,k] → tensor[i*d1*d2+j*d2+k]

3. Variable-length arrays (arr[n]/F) work correctly
   - RVec size matches counter branch value
   - Size varies per event

IMPLICATIONS FOR PHASE 13.4:

✅ STILL NEEDED:
   - D1 Schema Parser: Parse float[n][3] notation
   - D2-D3 Auto-detection: Discover counter branches
   - D4-D7 Stride calculation: mat[i,j] → mat[i*cols+j]
   
✅ REUSABLE FROM PHASE 13.3:
   - RVec element access: arr[idx]
   - RVec bounds checking: (idx<size) ? arr[idx] : NaN
   - RVec slicing: Take(arr, n)
   
❌ NOT NEEDED:
   - T* pointer handling (ROOT gives us RVec)
   - Lambda expressions
""")


if __name__ == "__main__":
    main()
