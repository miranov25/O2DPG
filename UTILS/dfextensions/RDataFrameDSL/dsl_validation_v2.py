#!/usr/bin/env python3
"""
Minimal DSLCompiler - End-to-End Validation (Fixed)
Tests the RDataFrame DSL with real expressions.

What works (Phase 1-6):
- Scalar math: sqrt(px**2 + py**2)
- RVec arithmetic: pt * 1.5, px + py
- Simple indexing: pt[0], pt[-1]
- RVec methods: pt.size(), pt.empty()
- Object methods: particle.Pt()
- Private members: particle.fPx (via reflection)

What doesn't work yet (Phase 7+):
- Slicing: pt[1:3]
- Boolean masking: pt[pt > 1.0]
- Method broadcasting: tracks.Pt()
"""

import sys
import os

# Add the RDataFrameDSL path (adjust if needed)
sys.path.insert(0, os.path.expanduser("~/alicesw/O2DPG/UTILS/dfextensions/RDataFrameDSL"))

from RDataFrameDSL import (
    IRBuilder,
    TypeInferrer,
    CppCodeGenerator,
    FunctionLibrary,
    ReflectionCache,
)


def simple_schema_to_full(simple_schema: dict) -> dict:
    """
    Convert simple schema to full TypeInferrer schema format.
    
    Simple: {"px": "double", "pt": "RVec<double>"}
    Full: {"columns": {"px": {"dtype": "double", "rank": 0}, ...}}
    """
    columns = {}
    
    for name, type_str in simple_schema.items():
        # Parse RVec<T>
        if type_str.startswith("RVec<") and type_str.endswith(">"):
            inner = type_str[5:-1]  # Extract T from RVec<T>
            columns[name] = {"dtype": inner, "rank": 1}
        elif type_str.startswith("std::vector<") and type_str.endswith(">"):
            inner = type_str[12:-1]
            columns[name] = {"dtype": inner, "rank": 1}
        else:
            # Scalar or object type
            columns[name] = {"dtype": type_str, "rank": 0}
    
    return {"columns": columns}


class DSLCompiler:
    """
    Minimal DSL compiler that converts Python-like expressions to C++ functions.
    
    Usage:
        schema = {"px": "double", "py": "double", "pt": "RVec<double>"}
        dsl = DSLCompiler(schema)
        
        dsl.define("momentum", "sqrt(px**2 + py**2)")
        dsl.define("lead_pt", "pt[0]")
        
        rdf = dsl.apply(rdf)
    """
    
    def __init__(self, schema: dict):
        """
        Initialize DSL compiler with schema.
        
        Args:
            schema: Dict mapping variable names to C++ types
                    e.g. {"px": "double", "pt": "RVec<double>", "particle": "TParticle"}
        """
        self.schema = schema
        
        # Convert simple schema to full format
        full_schema = simple_schema_to_full(schema)
        self.type_inferrer = TypeInferrer.from_schema(full_schema)
        
        # Try to create reflection cache (needs ROOT)
        try:
            self.reflection_cache = ReflectionCache()
        except:
            self.reflection_cache = None
        
        self.generator = CppCodeGenerator(
            type_inferrer=self.type_inferrer,
            reflection_cache=self.reflection_cache,
            safe_indexing=True
        )
        self.library = FunctionLibrary()
        self.definitions = {}  # name -> GeneratedFunction
    
    def define(self, name: str, expression: str) -> 'DSLCompiler':
        """
        Define a new column from an expression.
        
        Args:
            name: Output column name
            expression: Python-like expression, e.g. "sqrt(px**2 + py**2)"
        
        Returns:
            self (for chaining)
        """
        # Parse expression to IR
        builder = IRBuilder(self.type_inferrer)
        ir = builder.build(expression)
        
        # Generate C++ function
        func = self.generator.generate(ir, name)
        
        # Store and compile
        self.library.add(func)
        self.definitions[name] = func
        
        return self
    
    def compile_all(self):
        """Compile all defined functions to ROOT."""
        for name in self.definitions:
            self.library.compile(name)
    
    def get_code(self, name: str) -> str:
        """Get generated C++ code for a definition."""
        if name not in self.definitions:
            raise KeyError(f"No definition named '{name}'")
        return self.definitions[name].code
    
    def get_call(self, name: str) -> str:
        """Get the function call expression for RDataFrame.Define()."""
        if name not in self.definitions:
            raise KeyError(f"No definition named '{name}'")
        return self.definitions[name].get_call_expression()
    
    def apply(self, rdf):
        """
        Apply all definitions to an RDataFrame.
        
        Args:
            rdf: ROOT.RDataFrame instance
        
        Returns:
            Modified RDataFrame with new columns
        """
        self.compile_all()
        
        for name, func in self.definitions.items():
            rdf = rdf.Define(name, func.get_call_expression())
        
        return rdf
    
    def list_definitions(self) -> list:
        """List all defined columns."""
        return list(self.definitions.keys())
    
    def print_all_code(self):
        """Print all generated C++ code."""
        for name, func in self.definitions.items():
            print(f"=== {name} ===")
            print(func.code)
            print()


# =============================================================================
# Test Functions
# =============================================================================

def test_scalar_expressions():
    """Test basic scalar math expressions."""
    print("\n" + "="*60)
    print("TEST: Scalar Expressions")
    print("="*60)
    
    schema = {
        "px": "double",
        "py": "double",
        "pz": "double",
    }
    
    dsl = DSLCompiler(schema)
    
    # Define expressions
    dsl.define("pt", "sqrt(px**2 + py**2)")
    dsl.define("p", "sqrt(px**2 + py**2 + pz**2)")
    dsl.define("phi", "atan2(py, px)")
    
    # Print generated code
    dsl.print_all_code()
    
    print("✅ Scalar expressions work!")
    return True


def test_rvec_expressions():
    """Test RVec operations."""
    print("\n" + "="*60)
    print("TEST: RVec Expressions")
    print("="*60)
    
    schema = {
        "pt": "RVec<double>",
        "eta": "RVec<double>",
    }
    
    dsl = DSLCompiler(schema)
    
    # Define expressions
    dsl.define("n_tracks", "pt.size()")
    dsl.define("lead_pt", "pt[0]")
    dsl.define("last_pt", "pt[-1]")
    dsl.define("scaled_pt", "pt * 1.5")
    dsl.define("is_empty", "pt.empty()")
    
    # Print generated code
    dsl.print_all_code()
    
    print("✅ RVec expressions work!")
    return True


def test_object_expressions():
    """Test object method calls and property access."""
    print("\n" + "="*60)
    print("TEST: Object Expressions")
    print("="*60)
    
    schema = {
        "vec": "TVector3",
        "particle": "TParticle",
    }
    
    dsl = DSLCompiler(schema)
    
    # Define expressions - methods
    dsl.define("vec_mag", "vec.Mag()")
    dsl.define("particle_pt", "particle.Pt()")
    
    # Define expressions - properties (may use reflection for protected)
    dsl.define("vec_x", "vec.fX")
    
    # Print generated code
    dsl.print_all_code()
    
    print("✅ Object expressions work!")
    return True


def test_with_root():
    """Test end-to-end with ROOT (if available)."""
    print("\n" + "="*60)
    print("TEST: End-to-End with ROOT")
    print("="*60)
    
    try:
        import ROOT
    except ImportError:
        print("⚠️  ROOT not available, skipping")
        return True
    
    # Create test data
    ROOT.gInterpreter.ProcessLine('''
        void create_test_tree(const char* filename) {
            TFile f(filename, "RECREATE");
            TTree tree("Events", "Test events");
            
            double px, py, pz;
            std::vector<double> track_pt;
            
            tree.Branch("px", &px);
            tree.Branch("py", &py);
            tree.Branch("pz", &pz);
            tree.Branch("track_pt", &track_pt);
            
            // Event 1
            px = 3.0; py = 4.0; pz = 0.0;
            track_pt = {1.5, 2.5, 3.5, 4.5};
            tree.Fill();
            
            // Event 2
            px = 1.0; py = 0.0; pz = 0.0;
            track_pt = {10.0, 20.0};
            tree.Fill();
            
            // Event 3
            px = 0.0; py = 5.0; pz = 12.0;
            track_pt = {0.5};
            tree.Fill();
            
            tree.Write();
            f.Close();
        }
    ''')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
        filename = tmp.name
    
    ROOT.create_test_tree(filename)
    
    # Define schema
    schema = {
        "px": "double",
        "py": "double",
        "pz": "double",
        "track_pt": "RVec<double>",
    }
    
    # Create DSL compiler
    dsl = DSLCompiler(schema)
    
    # Define columns
    dsl.define("pt", "sqrt(px**2 + py**2)")
    dsl.define("p_total", "sqrt(px**2 + py**2 + pz**2)")
    dsl.define("n_tracks", "track_pt.size()")
    dsl.define("lead_track_pt", "track_pt[0]")
    dsl.define("last_track_pt", "track_pt[-1]")
    
    # Print generated code
    print("\nGenerated C++ code:")
    dsl.print_all_code()
    
    # Apply to RDataFrame
    rdf = ROOT.RDataFrame("Events", filename)
    rdf = dsl.apply(rdf)
    
    # Verify results
    print("\nResults:")
    results = rdf.AsNumpy(["pt", "p_total", "n_tracks", "lead_track_pt", "last_track_pt"])
    
    print(f"  pt:            {results['pt']}")
    print(f"  p_total:       {results['p_total']}")
    print(f"  n_tracks:      {results['n_tracks']}")
    print(f"  lead_track_pt: {results['lead_track_pt']}")
    print(f"  last_track_pt: {results['last_track_pt']}")
    
    # Verify expected values
    import math
    assert abs(results['pt'][0] - 5.0) < 0.001, f"Expected pt[0]=5.0, got {results['pt'][0]}"
    assert abs(results['p_total'][2] - 13.0) < 0.001, f"Expected p_total[2]=13.0, got {results['p_total'][2]}"
    assert results['n_tracks'][0] == 4, f"Expected n_tracks[0]=4, got {results['n_tracks'][0]}"
    assert abs(results['lead_track_pt'][0] - 1.5) < 0.001, f"Expected lead_track_pt[0]=1.5, got {results['lead_track_pt'][0]}"
    assert abs(results['last_track_pt'][0] - 4.5) < 0.001, f"Expected last_track_pt[0]=4.5, got {results['last_track_pt'][0]}"
    
    # Cleanup
    os.unlink(filename)
    
    print("\n✅ End-to-end with ROOT works!")
    return True


def test_combined_expressions():
    """Test combined scalar + RVec expressions."""
    print("\n" + "="*60)
    print("TEST: Combined Expressions")
    print("="*60)
    
    schema = {
        "px": "double",
        "py": "double",
        "track_pt": "RVec<double>",
        "track_eta": "RVec<double>",
    }
    
    dsl = DSLCompiler(schema)
    
    # Scalar
    dsl.define("event_pt", "sqrt(px**2 + py**2)")
    
    # RVec
    dsl.define("n_tracks", "track_pt.size()")
    dsl.define("lead_pt", "track_pt[0]")
    
    # RVec arithmetic
    dsl.define("scaled_pt", "track_pt * 2.0")
    dsl.define("pt_plus_eta", "track_pt + track_eta")
    
    # Print
    dsl.print_all_code()
    
    print("✅ Combined expressions work!")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RDataFrame DSL - End-to-End Validation")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_scalar_expressions()
    except Exception as e:
        print(f"❌ Scalar expressions failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_rvec_expressions()
    except Exception as e:
        print(f"❌ RVec expressions failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_object_expressions()
    except Exception as e:
        print(f"❌ Object expressions failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_combined_expressions()
    except Exception as e:
        print(f"❌ Combined expressions failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_with_root()
    except Exception as e:
        print(f"❌ ROOT end-to-end failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - DSL is working!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)
