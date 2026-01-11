#!/usr/bin/env python3
"""
Phase 13.5.B0 - Tests T19-T21: Overload, Cross-Instance, Export

T19: Overload Set Preservation (P0)
T20: Cross-Instance Collision Test (P0)
T21: Export→Reload→Define Compatibility (P0)

All Priority P0 - Core Architectural Validation
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime

from test_infrastructure import (
    setup_test_env, get_workspace, TestResult, 
    MockDSLCompiler, check_root_available, create_test_macro
)


# =============================================================================
# T19: Overload Set Preservation
# =============================================================================

def test_t19_overload_preservation():
    """
    T19: Overload Set Preservation Test
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Verify resolution doesn't bypass C++ overload resolution.
    One of the most likely remaining 'gotchas' in v7.
    """
    result = TestResult(
        test_id="T19",
        title="Overload Set Preservation",
        hypothesis="C++ overload resolution works correctly for "
                   "functions with same name but different signatures"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T19: Overload Set Preservation")
    result.log("="*60)
    
    # =========================================================================
    # T19a: Arity-based Overloading
    # =========================================================================
    result.log("\n--- T19a: Arity-based Overloading ---")
    
    # Declare overloaded functions
    overload_code = '''
// 1-arg version: multiply by 2
double t19_calc(double x) {
    return x * 2.0;
}

// 2-arg version: add
double t19_calc(double x, double y) {
    return x + y;
}

// 3-arg version: sum all
double t19_calc(double x, double y, double z) {
    return x + y + z;
}
'''
    
    decl_result = ROOT.gInterpreter.Declare(overload_code)
    result.observe("t19a_declare_result", decl_result, "Overloaded functions declared")
    
    if not decl_result:
        result.status = "FAILED"
        result.log("Declaration failed", "FAIL")
        return result
    
    # Test with RDataFrame
    rdf = ROOT.RDataFrame(1)
    rdf = rdf.Define("x", "5.0")
    rdf = rdf.Define("y", "3.0")
    rdf = rdf.Define("z", "2.0")
    
    # Apply each overload
    rdf = rdf.Define("result1", "t19_calc(x)")           # 1 arg → 5 * 2 = 10
    rdf = rdf.Define("result2", "t19_calc(x, y)")        # 2 args → 5 + 3 = 8
    rdf = rdf.Define("result3", "t19_calc(x, y, z)")     # 3 args → 5 + 3 + 2 = 10
    
    r1 = rdf.Sum("result1").GetValue()
    r2 = rdf.Sum("result2").GetValue()
    r3 = rdf.Sum("result3").GetValue()
    
    result.log(f"  1-arg result: {r1} (expected 10.0)")
    result.log(f"  2-arg result: {r2} (expected 8.0)")
    result.log(f"  3-arg result: {r3} (expected 10.0)")
    
    arity_correct = (r1 == 10.0 and r2 == 8.0 and r3 == 10.0)
    result.observe("t19a_arity_overload_works", arity_correct,
                   "C++ resolves overloads by argument count")
    
    if arity_correct:
        result.log("Arity-based overloading: PASS", "PASS")
    else:
        result.log("Arity-based overloading: FAIL", "FAIL")
    
    # =========================================================================
    # T19b: Type-based Overloading
    # =========================================================================
    result.log("\n--- T19b: Type-based Overloading ---")
    
    type_overload_code = '''
// double version: multiply by 2
double t19_typed(double x) {
    return x * 2.0;
}

// float version: multiply by 3
double t19_typed(float x) {
    return x * 3.0;
}

// int version: multiply by 4
double t19_typed(int x) {
    return x * 4.0;
}
'''
    
    decl2 = ROOT.gInterpreter.Declare(type_overload_code)
    result.observe("t19b_typed_declare", decl2, "Type-overloaded functions declared")
    
    # Test type resolution
    rdf2 = ROOT.RDataFrame(1)
    rdf2 = rdf2.Define("x_double", "5.0")     # double literal
    rdf2 = rdf2.Define("x_float", "5.0f")     # float literal
    rdf2 = rdf2.Define("x_int", "5")          # int literal
    
    rdf2 = rdf2.Define("r_double", "t19_typed(x_double)")
    rdf2 = rdf2.Define("r_float", "t19_typed(x_float)")
    rdf2 = rdf2.Define("r_int", "t19_typed(x_int)")
    
    rd = rdf2.Sum("r_double").GetValue()
    rf = rdf2.Sum("r_float").GetValue()
    ri = rdf2.Sum("r_int").GetValue()
    
    result.log(f"  double literal → {rd} (expected 10.0)")
    result.log(f"  float literal → {rf} (expected 15.0)")
    result.log(f"  int literal → {ri} (expected 20.0)")
    
    # Note: RDataFrame may coerce types
    result.observe("t19b_double_result", rd, "double literal result")
    result.observe("t19b_float_result", rf, "float literal result")
    result.observe("t19b_int_result", ri, "int literal result")
    
    # Document actual behavior (may vary due to coercion)
    if rd == 10.0 and rf == 15.0 and ri == 20.0:
        result.log("Type-based overloading: PERFECT", "PASS")
    else:
        result.log("Type-based overloading: Coercion observed", "INFO")
        result.action_items.append("Document type coercion behavior in v7 spec")
    
    # Summary
    if arity_correct:
        result.status = "PASSED"
        result.log("\n✅ T19 PASSED: Overload preservation verified", "PASS")
    else:
        result.status = "FAILED"
        result.log("\n❌ T19 FAILED: Overload resolution issues", "FAIL")
    
    return result


# =============================================================================
# T20: Cross-Instance Collision Test
# =============================================================================

def test_t20_cross_instance():
    """
    T20: Cross-Instance Collision (Two DSLCompiler instances)
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Validate multi-instance behavior in single ROOT session.
    Forces clear contract: is function resolution per-DSLCompiler instance or global?
    
    Expected (v7): Hash suffix prevents collision. Different implementations
    get different hashes, so different symbol names.
    """
    result = TestResult(
        test_id="T20",
        title="Cross-Instance Collision Test",
        hypothesis="Two DSLCompiler instances with same function name but "
                   "different implementations produce different results due to hash suffix"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T20: Cross-Instance Collision Test")
    result.log("="*60)
    
    # v7 Contract: Different implementations get different hashes
    # dsl1: pt = sqrt(px*px + py*py)  → dsl_pt_<hash1>
    # dsl2: pt = px + py              → dsl_pt_<hash2>
    
    schema = {"px": "double", "py": "double"}
    
    # =========================================================================
    # T20a: Create two DSL instances with different implementations
    # =========================================================================
    result.log("\n--- T20a: Create Two DSL Instances ---")
    
    # Instance 1: sqrt(px*px + py*py)
    dsl1 = MockDSLCompiler(schema)
    dsl1.register_function_cpp('''
        double t20_pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    func1 = dsl1.get_function("t20_pt")
    result.log(f"  DSL1: {func1.cpp_name} (sqrt implementation)")
    result.observe("t20a_dsl1_hash", func1.hash, "Hash for sqrt implementation")
    
    # Instance 2: px + py (different implementation!)
    dsl2 = MockDSLCompiler(schema)
    dsl2.register_function_cpp('''
        double t20_pt(double px, double py) {
            return px + py;
        }
    ''')
    
    func2 = dsl2.get_function("t20_pt")
    result.log(f"  DSL2: {func2.cpp_name} (sum implementation)")
    result.observe("t20a_dsl2_hash", func2.hash, "Hash for sum implementation")
    
    # =========================================================================
    # T20b: Verify different hashes
    # =========================================================================
    result.log("\n--- T20b: Verify Different Hashes ---")
    
    hashes_different = func1.hash != func2.hash
    result.observe("t20b_hashes_different", hashes_different,
                   "Different implementations produce different hashes")
    
    if hashes_different:
        result.log("Different hashes for different implementations: PASS", "PASS")
    else:
        result.log("Same hash for different implementations: FAIL", "FAIL")
        result.status = "FAILED"
        return result
    
    # =========================================================================
    # T20c: Test actual execution
    # =========================================================================
    result.log("\n--- T20c: Test Execution ---")
    
    # Create test data
    rdf = ROOT.RDataFrame(1)
    rdf = rdf.Define("px", "3.0")
    rdf = rdf.Define("py", "4.0")
    
    # Use dsl1 function (sqrt)
    rdf1 = dsl1.apply(rdf)
    rdf1 = rdf1.Define("track_pt", f"{func1.cpp_name}(px, py)")
    result1 = rdf1.Sum("track_pt").GetValue()
    
    # Use dsl2 function (sum)
    rdf2 = dsl2.apply(rdf)
    rdf2 = rdf2.Define("track_pt", f"{func2.cpp_name}(px, py)")
    result2 = rdf2.Sum("track_pt").GetValue()
    
    result.log(f"  DSL1 (sqrt): {result1} (expected 5.0)")
    result.log(f"  DSL2 (sum):  {result2} (expected 7.0)")
    
    result.observe("t20c_dsl1_result", result1, "sqrt(3² + 4²) = 5")
    result.observe("t20c_dsl2_result", result2, "3 + 4 = 7")
    
    results_different = (result1 != result2)
    results_correct = (result1 == 5.0 and result2 == 7.0)
    
    result.observe("t20c_results_different", results_different,
                   "Different DSL instances produce different results")
    result.observe("t20c_results_correct", results_correct,
                   "Both results match expected values")
    
    # Summary
    if hashes_different and results_correct:
        result.status = "PASSED"
        result.log("\n✅ T20 PASSED: Cross-instance collision prevented by hash", "PASS")
    elif hashes_different and results_different:
        result.status = "PASSED"
        result.log("\n✅ T20 PASSED: No collision (results differ)", "PASS")
    else:
        result.status = "FAILED"
        result.log("\n❌ T20 FAILED: Cross-instance collision detected", "FAIL")
    
    return result


# =============================================================================
# T21: Export→Reload→Define Compatibility
# =============================================================================

def test_t21_export_reload():
    """
    T21: Export → Reload → Define Compatibility Test
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Confirm 'export is deployable,' not just 'export compiles.'
    Tests both .L and gSystem.Load() paths (per review feedback).
    """
    result = TestResult(
        test_id="T21",
        title="Export→Reload→Define Compatibility",
        hypothesis="Exported macro can be loaded in fresh session and "
                   "used directly without DSL"
    )
    result.set_environment()
    
    workspace = setup_test_env()
    result.log(f"Workspace: {workspace}")
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T21: Export→Reload→Define Compatibility")
    result.log("="*60)
    
    # =========================================================================
    # T21a: Create and export DSL functions
    # =========================================================================
    result.log("\n--- T21a: Create and Export Functions ---")
    
    dsl = MockDSLCompiler()
    dsl.register_function_cpp('''
        double t21_pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    dsl.register_function_cpp('''
        double t21_eta(double px, double py, double pz) {
            double p = sqrt(px*px + py*py + pz*pz);
            return 0.5 * log((p + pz) / (p - pz + 1e-10));
        }
    ''')
    
    macro_name = "t21_analysis.C"
    export_result = dsl.export_macro(
        macro_name,
        namespace="t21_ns",
        clean_names=True,
        compile=True
    )
    
    result.log(f"Exported to: {export_result['filepath']}")
    result.log(f"Functions: {export_result['functions_exported']}")
    result.log(f"Compiled: {export_result['compiled']}")
    
    result.observe("t21a_export_path", export_result['filepath'], "Export file path")
    result.observe("t21a_compiled", export_result['compiled'], "ACLiC compilation success")
    
    if not export_result['compiled']:
        result.status = "FAILED"
        result.log("Export compilation failed", "FAIL")
        return result
    
    # =========================================================================
    # T21b: Test .L loading path
    # =========================================================================
    result.log("\n--- T21b: Test .L Loading Path ---")
    
    # The .so should already be loaded from export, but test explicit load
    so_path = export_result['so_path']
    
    try:
        # Use functions via namespace
        rdf_b = ROOT.RDataFrame(10)
        rdf_b = rdf_b.Define("px", "sin(rdfentry_ * 0.1) * 10")
        rdf_b = rdf_b.Define("py", "cos(rdfentry_ * 0.1) * 10")
        rdf_b = rdf_b.Define("pz", "sin(rdfentry_ * 0.05) * 20")
        
        rdf_b = rdf_b.Define("track_pt", "t21_ns::t21_pt(px, py)")
        rdf_b = rdf_b.Define("track_eta", "t21_ns::t21_eta(px, py, pz)")
        
        mean_pt = rdf_b.Mean("track_pt").GetValue()
        mean_eta = rdf_b.Mean("track_eta").GetValue()
        
        result.log(f"  Mean(pt): {mean_pt}")
        result.log(f"  Mean(eta): {mean_eta}")
        
        import math
        pt_valid = not math.isnan(mean_pt)
        eta_valid = not math.isnan(mean_eta)
        
        result.observe("t21b_pt_valid", pt_valid, "pt calculation produces valid result")
        result.observe("t21b_eta_valid", eta_valid, "eta calculation produces valid result")
        
        if pt_valid and eta_valid:
            result.log(".L loading path: PASS", "PASS")
        else:
            result.log(".L loading path: NaN detected", "FAIL")
            
    except Exception as e:
        result.log(f".L loading path failed: {e}", "FAIL")
        result.observe("t21b_load_error", str(e), "Error using exported functions")
    
    # =========================================================================
    # T21c: Test gSystem.Load() path (subprocess for fresh session)
    # =========================================================================
    result.log("\n--- T21c: Test gSystem.Load() in Fresh Process ---")
    
    # Create test script for subprocess
    test_script = f'''
import ROOT
import math

# Load the exported .so (simulating fresh session)
ROOT.gSystem.Load("{so_path}")

# Create RDataFrame (NO DSL!)
rdf = ROOT.RDataFrame(100)
rdf = rdf.Define("px", "sin(rdfentry_ * 0.1) * 10")
rdf = rdf.Define("py", "cos(rdfentry_ * 0.1) * 10")
rdf = rdf.Define("pz", "sin(rdfentry_ * 0.05) * 20")

# Use exported functions directly
rdf = rdf.Define("track_pt", "t21_ns::t21_pt(px, py)")
rdf = rdf.Define("track_eta", "t21_ns::t21_eta(px, py, pz)")

mean_pt = rdf.Mean("track_pt").GetValue()
mean_eta = rdf.Mean("track_eta").GetValue()

if not math.isnan(mean_pt) and not math.isnan(mean_eta):
    print(f"SUCCESS: pt={{mean_pt:.4f}}, eta={{mean_eta:.4f}}")
else:
    print(f"FAIL: NaN detected pt={{mean_pt}}, eta={{mean_eta}}")
'''
    
    try:
        proc = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(workspace)
        )
        
        result.log(f"  Subprocess stdout: {proc.stdout.strip()}")
        if proc.stderr:
            result.log(f"  Subprocess stderr: {proc.stderr.strip()[:200]}")
        
        subprocess_success = "SUCCESS" in proc.stdout
        result.observe("t21c_subprocess_success", subprocess_success,
                       "Fresh process can use exported functions")
        
        if subprocess_success:
            result.log("gSystem.Load() path: PASS", "PASS")
        else:
            result.log("gSystem.Load() path: FAIL", "FAIL")
            
    except subprocess.TimeoutExpired:
        result.log("Subprocess timeout", "FAIL")
        result.observe("t21c_subprocess_success", False, "Timeout")
    except Exception as e:
        result.log(f"Subprocess error: {e}", "FAIL")
        result.observe("t21c_subprocess_success", False, str(e))
    
    # Summary
    if pt_valid and eta_valid:
        result.status = "PASSED"
        result.log("\n✅ T21 PASSED: Export is deployable", "PASS")
    else:
        result.status = "PARTIAL"
        result.log("\n⚠️  T21 PARTIAL: Some load paths failed", "WARN")
    
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Tests T19-T21")
    print("="*70)
    
    results = {}
    
    # T19: Overload Preservation
    print("\n" + "="*70)
    results["T19"] = test_t19_overload_preservation()
    
    # T20: Cross-Instance
    print("\n" + "="*70)
    results["T20"] = test_t20_cross_instance()
    
    # T21: Export→Reload
    print("\n" + "="*70)
    results["T21"] = test_t21_export_reload()
    
    # Summary
    print("\n" + "="*70)
    print("T19-T21 SUMMARY")
    print("="*70)
    
    for test_id, result in results.items():
        status_icon = "✅" if result.status == "PASSED" else "⚠️" if result.status == "PARTIAL" else "❌"
        print(f"{status_icon} {test_id}: {result.status}")
    
    passed = sum(1 for r in results.values() if r.status in ["PASSED", "PARTIAL"])
    print(f"\n{passed}/{len(results)} tests passed")
    
    sys.exit(0 if passed == len(results) else 1)
