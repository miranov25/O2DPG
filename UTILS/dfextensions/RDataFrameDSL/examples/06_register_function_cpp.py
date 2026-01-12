#!/usr/bin/env python3
"""
Example 06: Custom C++ Function Registration

This example demonstrates how to register custom C++ functions
that can be used in RDataFrame expressions.

Phase 13.5.B feature: register_function_cpp()

Key features:
- Register C++ functions with natural syntax
- Hash-based naming prevents collisions
- Thread-safe declaration
- Auto-detection of required headers
- FROZEN RULE #1: Lambda expressions are prohibited
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT
ROOT.gROOT.SetBatch(True)

from RDataFrameDSL import DSLCompiler


def main():
    print("=" * 60)
    print("Example 06: Custom C++ Function Registration")
    print("=" * 60)
    
    # =========================================================================
    # Setup: Create synthetic data
    # =========================================================================
    
    print("\nCreating synthetic data...")
    rdf = ROOT.RDataFrame(1000)
    rdf = rdf.Define("px", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("py", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("pz", "gRandom->Gaus(0, 50)")
    rdf = rdf.Define("energy", "gRandom->Uniform(50, 150)")
    rdf = rdf.Define("eta_raw", "gRandom->Uniform(-2.5, 2.5)")
    
    # Schema
    schema = {
        "px": "double",
        "py": "double", 
        "pz": "double",
        "energy": "double",
        "eta_raw": "double",
    }
    
    print("\nSchema:")
    for name, dtype in schema.items():
        print(f"  {name}: {dtype}")
    
    # =========================================================================
    # Part 1: Basic Function Registration
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 1: Basic Function Registration")
    print("=" * 60)
    
    dsl = DSLCompiler(schema)
    
    # Register a simple function
    dsl.register_function_cpp('''
        double pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    # Check what was registered
    func = dsl.get_registered_function("pt")
    print(f"\nRegistered function:")
    print(f"  Name:        {func.name}")
    print(f"  C++ name:    {func.cpp_name}")
    print(f"  Hash:        {func.hash}")
    print(f"  Return type: {func.return_type}")
    print(f"  Parameters:  {func.params}")
    print(f"  Declared:    {func.declared}")
    print(f"  Headers:     {func.headers}")
    
    # =========================================================================
    # Part 2: Multiple Functions with Chaining
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 2: Multiple Functions (Method Chaining)")
    print("=" * 60)
    
    # Register multiple functions using chaining
    dsl.register_function_cpp('''
        double p_total(double px, double py, double pz) {
            return sqrt(px*px + py*py + pz*pz);
        }
    ''').register_function_cpp('''
        double eta_calc(double px, double py, double pz) {
            double p = sqrt(px*px + py*py + pz*pz);
            double pt = sqrt(px*px + py*py);
            if (pt < 1e-10) return 0.0;
            return 0.5 * log((p + pz) / (p - pz + 1e-10));
        }
    ''').register_function_cpp('''
        double phi_calc(double px, double py) {
            return atan2(py, px);
        }
    ''')
    
    print(f"\nAll registered functions: {dsl.list_registered_functions()}")
    
    # =========================================================================
    # Part 3: Using Registered Functions in RDataFrame
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 3: Using Functions in RDataFrame")
    print("=" * 60)
    
    # Apply DSL (declares functions in ROOT)
    rdf = dsl.apply(rdf)
    
    # Get C++ names for Define
    pt_func = dsl.get_registered_function("pt")
    eta_func = dsl.get_registered_function("eta_calc")
    phi_func = dsl.get_registered_function("phi_calc")
    
    # Define computed columns using registered functions
    # NOTE: Use rdf.Define() directly with the C++ function name
    rdf = rdf.Define("track_pt", f"{pt_func.cpp_name}(px, py)")
    rdf = rdf.Define("track_eta", f"{eta_func.cpp_name}(px, py, pz)")
    rdf = rdf.Define("track_phi", f"{phi_func.cpp_name}(px, py)")
    
    # Calculate statistics
    mean_pt = rdf.Mean("track_pt").GetValue()
    mean_eta = rdf.Mean("track_eta").GetValue()
    mean_phi = rdf.Mean("track_phi").GetValue()
    
    print(f"\nComputed column statistics:")
    print(f"  Mean track_pt:  {mean_pt:.4f}")
    print(f"  Mean track_eta: {mean_eta:.4f}")
    print(f"  Mean track_phi: {mean_phi:.4f}")
    
    # =========================================================================
    # Part 4: Using with Filter and Selection
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 4: Using with Filter and Selection")
    print("=" * 60)
    
    # NOTE: Registered functions are used directly in RDataFrame expressions,
    # not through dsl.define(). The DSL parser doesn't recognize them.
    # Use rdf.Define() and rdf.Filter() with the C++ function names.
    
    # Define selection columns using C++ expressions directly
    rdf = rdf.Define("high_pt_cut", f"{pt_func.cpp_name}(px, py) > 10.0")
    rdf = rdf.Define("central_eta", f"abs({eta_func.cpp_name}(px, py, pz)) < 1.0")
    
    # Count events passing cuts
    n_high_pt = rdf.Filter("high_pt_cut").Count().GetValue()
    n_central = rdf.Filter("central_eta").Count().GetValue()
    n_both = rdf.Filter("high_pt_cut && central_eta").Count().GetValue()
    
    print(f"\nSelection results (out of 1000 events):")
    print(f"  high_pt_cut (pT > 10):     {n_high_pt}")
    print(f"  central_eta (|η| < 1.0):   {n_central}")
    print(f"  Both cuts:                 {n_both}")
    
    # =========================================================================
    # Part 5: Custom Headers
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 5: Custom Headers")
    print("=" * 60)
    
    # Register function that needs specific headers
    dsl.register_function_cpp('''
        double clamp_value(double x, double lo, double hi) {
            return std::max(lo, std::min(x, hi));
        }
    ''', headers=["<algorithm>"])
    
    clamp_func = dsl.get_registered_function("clamp_value")
    print(f"\nclamp_value headers: {clamp_func.headers}")
    
    # Use clamped value
    rdf = rdf.Define("clamped_pt", f"{clamp_func.cpp_name}(track_pt, 0.0, 50.0)")
    mean_clamped = rdf.Mean("clamped_pt").GetValue()
    print(f"Mean clamped_pt (0-50): {mean_clamped:.4f}")
    
    # =========================================================================
    # Part 6: FROZEN RULE #1 - Lambda Rejection
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 6: FROZEN RULE #1 - Lambda Rejection")
    print("=" * 60)
    
    print("\nLambda expressions are PROHIBITED due to ROOT stability issues.")
    print("Attempting to register a lambda...")
    
    try:
        dsl.register_function_cpp("[](double x) { return x * 2; }")
        print("  ERROR: Lambda was accepted (this is a bug!)")
    except ValueError as e:
        print(f"  ✓ Lambda correctly rejected")
        print(f"    Error: {str(e)[:50]}...")
    
    print("\nUse named functions instead:")
    print("  ✗ [](double x) { return x * 2; }")
    print("  ✓ double double_it(double x) { return x * 2; }")
    
    # =========================================================================
    # Part 7: Show Generated C++ Code
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 7: Generated C++ Code")
    print("=" * 60)
    
    print("\n// pt function:")
    print(pt_func.full_cpp)
    
    print("\n// eta_calc function:")
    print(eta_func.full_cpp)
    
    # =========================================================================
    # Part 8: Combining DSL define() with Registered Functions
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Part 8: Workflow - DSL + Registered Functions")
    print("=" * 60)
    
    print("""
Recommended workflow:

1. Use dsl.define() for DSL expressions (Python-like syntax):
   dsl.define("pt_squared", "px**2 + py**2")
   dsl.define("high_pt_mask", "pt > 10.0")

2. Use register_function_cpp() for complex C++ logic:
   dsl.register_function_cpp('''
       double complex_calc(double x, double y) {
           // Complex logic here
           return result;
       }
   ''')

3. Use rdf.Define() to call registered functions:
   rdf = rdf.Define("result", "dsl_complex_calc_xxx(x, y)")

4. Combine both in filters:
   rdf.Filter("high_pt_mask && dsl_complex_calc_xxx(x, y) > 0")
""")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"""
register_function_cpp() allows you to:
- Register custom C++ functions using natural syntax
- Use them in RDataFrame Define/Filter expressions
- Auto-detect required headers

Key points:
- Internal naming: dsl_{{name}}_{{hash16}}
- Thread-safe (class-level lock)
- FROZEN RULE #1: No lambda expressions
- Use rdf.Define() to call registered functions (not dsl.define())

Registered functions: {dsl.list_registered_functions()}
""")
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
