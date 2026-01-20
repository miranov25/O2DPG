#!/usr/bin/env python3
"""
Example: register_function_cpp() with TTree::Draw-like functionality

Phase 13.5.B — C++ Function Registration

This example demonstrates:
1. Registering custom C++ functions
2. Using them in RDataFrame expressions
3. TTree::Draw-like plotting with dsl.draw()
"""

import ROOT

# Suppress ROOT startup messages
ROOT.gROOT.SetBatch(True)

from RDataFrameDSL import DSLCompiler


def example_basic():
    """Basic example: register and use a simple function."""
    print("=" * 60)
    print("Example 1: Basic Function Registration")
    print("=" * 60)
    
    # Create synthetic data
    rdf = ROOT.RDataFrame(1000)
    rdf = rdf.Define("px", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("py", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("pz", "gRandom->Gaus(0, 50)")
    
    # Create DSL compiler with schema
    dsl = DSLCompiler({
        "px": "double",
        "py": "double",
        "pz": "double",
    })
    
    # Register custom C++ function
    dsl.register_function_cpp('''
        double pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    # Check registration
    func = dsl.get_registered_function("pt")
    print(f"✅ Registered: {func.name}")
    print(f"   C++ name:   {func.cpp_name}")
    print(f"   Hash:       {func.hash}")
    print(f"   Declared:   {func.declared}")
    print(f"   Headers:    {func.headers}")
    
    # Apply to RDataFrame
    rdf = dsl.apply(rdf)
    
    # Use registered function in Define
    rdf = rdf.Define("track_pt", f"{func.cpp_name}(px, py)")
    
    # Verify it works
    mean_pt = rdf.Mean("track_pt").GetValue()
    print(f"\n   Mean pt:    {mean_pt:.2f}")
    
    return dsl, rdf


def example_multiple_functions():
    """Register multiple functions and chain them."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Functions")
    print("=" * 60)
    
    # Create synthetic data
    rdf = ROOT.RDataFrame(1000)
    rdf = rdf.Define("px", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("py", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("pz", "gRandom->Gaus(0, 50)")
    rdf = rdf.Define("energy", "gRandom->Gaus(100, 20)")
    
    dsl = DSLCompiler({
        "px": "double",
        "py": "double",
        "pz": "double",
        "energy": "double",
    })
    
    # Register multiple functions with chaining
    dsl.register_function_cpp('''
        double pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''').register_function_cpp('''
        double p(double px, double py, double pz) {
            return sqrt(px*px + py*py + pz*pz);
        }
    ''').register_function_cpp('''
        double eta(double px, double py, double pz) {
            double pmag = sqrt(px*px + py*py + pz*pz);
            if (pmag == pz) return 0;
            return 0.5 * log((pmag + pz) / (pmag - pz));
        }
    ''').register_function_cpp('''
        double phi(double px, double py) {
            return atan2(py, px);
        }
    ''')
    
    # List registered functions
    print(f"✅ Registered functions: {dsl.list_registered_functions()}")
    
    # Apply and use
    rdf = dsl.apply(rdf)
    
    pt_func = dsl.get_registered_function("pt")
    eta_func = dsl.get_registered_function("eta")
    phi_func = dsl.get_registered_function("phi")
    
    rdf = rdf.Define("track_pt", f"{pt_func.cpp_name}(px, py)")
    rdf = rdf.Define("track_eta", f"{eta_func.cpp_name}(px, py, pz)")
    rdf = rdf.Define("track_phi", f"{phi_func.cpp_name}(px, py)")
    
    # Stats
    print(f"\n   Mean pt:  {rdf.Mean('track_pt').GetValue():.2f}")
    print(f"   Mean eta: {rdf.Mean('track_eta').GetValue():.2f}")
    print(f"   Mean phi: {rdf.Mean('track_phi').GetValue():.2f}")
    
    return dsl, rdf


def example_rvec():
    """Register function working with RVec."""
    print("\n" + "=" * 60)
    print("Example 3: RVec Functions")
    print("=" * 60)
    
    # Create synthetic data with RVec columns
    rdf = ROOT.RDataFrame(100)
    rdf = rdf.Define("nTracks", "(int)(gRandom->Uniform(1, 10))")
    rdf = rdf.Define("trackPt", 
        "ROOT::VecOps::RVec<double> v(nTracks); "
        "for(int i=0; i<nTracks; i++) v[i] = gRandom->Gaus(10, 3); "
        "return v;")
    
    dsl = DSLCompiler({
        "nTracks": "int",
        "trackPt": "RVec<double>",
    })
    
    # Register RVec function
    dsl.register_function_cpp('''
        double sum_pt(const RVec<double>& pts) {
            return Sum(pts);
        }
    ''')
    
    dsl.register_function_cpp('''
        double max_pt(const RVec<double>& pts) {
            return pts.size() > 0 ? Max(pts) : 0.0;
        }
    ''')
    
    func_sum = dsl.get_registered_function("sum_pt")
    func_max = dsl.get_registered_function("max_pt")
    
    print(f"✅ Registered: sum_pt → {func_sum.cpp_name}")
    print(f"✅ Registered: max_pt → {func_max.cpp_name}")
    
    rdf = dsl.apply(rdf)
    rdf = rdf.Define("total_pt", f"{func_sum.cpp_name}(trackPt)")
    rdf = rdf.Define("leading_pt", f"{func_max.cpp_name}(trackPt)")
    
    print(f"\n   Mean total pt:   {rdf.Mean('total_pt').GetValue():.2f}")
    print(f"   Mean leading pt: {rdf.Mean('leading_pt').GetValue():.2f}")
    
    return dsl, rdf


def example_draw_like():
    """TTree::Draw-like functionality."""
    print("\n" + "=" * 60)
    print("Example 4: TTree::Draw-like Usage")
    print("=" * 60)
    
    # Create synthetic data
    rdf = ROOT.RDataFrame(1000)
    rdf = rdf.Define("px", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("py", "gRandom->Gaus(0, 10)")
    rdf = rdf.Define("eta", "gRandom->Uniform(-2.5, 2.5)")
    rdf = rdf.Define("isGood", "abs(eta) < 1.0")
    
    dsl = DSLCompiler({
        "px": "double",
        "py": "double",
        "eta": "double",
        "isGood": "bool",
    })
    
    # Register pt function
    dsl.register_function_cpp('''
        double pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    # Define computed column using registered function
    func = dsl.get_registered_function("pt")
    dsl.define("track_pt", f"{func.cpp_name}(px, py)")
    
    # Apply
    rdf = dsl.apply(rdf)
    
    print("✅ Ready for TTree::Draw-like plotting")
    print("\n   Available expressions:")
    print("   - dsl.draw('track_pt', rdf)")
    print("   - dsl.draw('track_pt:eta', rdf, type='hist2d')")
    print("   - dsl.draw('track_pt', rdf, selection='isGood')")
    
    # Test if dfdraw is available
    try:
        import dfdraw
        print("\n   dfdraw available - can run draw()")
        
        # 1D histogram
        fig, ax, stats = dsl.draw("track_pt", rdf)
        print(f"   ✅ 1D histogram: mean={stats.get('mean', 'N/A'):.2f}")
        
    except ImportError:
        print("\n   ⚠️ dfdraw not installed - skipping draw() demo")
        print("   Install with: pip install dfdraw")
    
    return dsl, rdf


def example_lambda_rejection():
    """Demonstrate FROZEN RULE #1: Lambda rejection."""
    print("\n" + "=" * 60)
    print("Example 5: FROZEN RULE #1 - Lambda Rejection")
    print("=" * 60)
    
    dsl = DSLCompiler({"x": "double"})
    
    # Try to register a lambda (should fail)
    print("Attempting to register lambda expression...")
    try:
        dsl.register_function_cpp("[](double x) { return x * 2; }")
        print("❌ Lambda was accepted (BUG!)")
    except ValueError as e:
        print("✅ Lambda correctly rejected:")
        print(f"   {str(e)[:60]}...")
    
    # Named function works
    print("\nRegistering named function instead...")
    dsl.register_function_cpp("double double_it(double x) { return x * 2; }")
    func = dsl.get_registered_function("double_it")
    print(f"✅ Named function accepted: {func.cpp_name}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RDataFrameDSL — register_function_cpp() Examples")
    print("Phase 13.5.B Implementation")
    print("=" * 60)
    
    example_basic()
    example_multiple_functions()
    example_rvec()
    example_draw_like()
    example_lambda_rejection()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
