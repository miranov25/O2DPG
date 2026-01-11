#!/usr/bin/env python3
"""
Phase 13.5.B0 - Tests T22-T24: Overload Edge, Idempotency, Schema

T22: Same-Arity Overload Ambiguity (P0)
T23: Registry Idempotency (Repeated apply) (P0)
T24: Schema-RDF Mismatch (P0)

All Priority P0 - Core Architectural Validation
"""

import os
import sys
from datetime import datetime

from test_infrastructure import (
    setup_test_env, get_workspace, TestResult, 
    MockDSLCompiler, check_root_available
)


# =============================================================================
# T22: Same-Arity Overload Ambiguity
# =============================================================================

def test_t22_same_arity_overload():
    """
    T22: Same-Arity Overload Ambiguity
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Test sharp edge of v7 resolution strategy.
    Same-arity overloads (different types) are the sharp edge.
    """
    result = TestResult(
        test_id="T22",
        title="Same-Arity Overload Ambiguity",
        hypothesis="C++ compiler resolves same-arity overloads by type, "
                   "or coercion behavior is documented"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T22: Same-Arity Overload Ambiguity")
    result.log("="*60)
    
    # =========================================================================
    # T22a: double vs float (same arity)
    # =========================================================================
    result.log("\n--- T22a: double vs float Overload ---")
    
    code = '''
// double version: multiply by 2
double t22_calc_d(double x) {
    return x * 2.0;
}

// float version: multiply by 3
double t22_calc_f(float x) {
    return x * 3.0;
}
'''
    
    decl_result = ROOT.gInterpreter.Declare(code)
    result.observe("t22a_declare", decl_result, "Functions declared")
    
    # Test with explicit types
    rdf = ROOT.RDataFrame(1)
    rdf = rdf.Define("x_double", "5.0")     # double literal
    rdf = rdf.Define("x_float", "5.0f")     # float literal
    
    # Call with double
    rdf = rdf.Define("r_d", "t22_calc_d(x_double)")
    rd = rdf.Sum("r_d").GetValue()
    result.log(f"  t22_calc_d(double): {rd} (expected 10.0)")
    
    # Call with float
    rdf2 = ROOT.RDataFrame(1)
    rdf2 = rdf2.Define("x_float", "5.0f")
    rdf2 = rdf2.Define("r_f", "t22_calc_f(x_float)")
    rf = rdf2.Sum("r_f").GetValue()
    result.log(f"  t22_calc_f(float): {rf} (expected 15.0)")
    
    result.observe("t22a_double_result", rd, "double function result")
    result.observe("t22a_float_result", rf, "float function result")
    
    # =========================================================================
    # T22b: Ambiguity test with single overloaded name
    # =========================================================================
    result.log("\n--- T22b: Ambiguous Overload Test ---")
    
    # Try to create truly ambiguous overloads
    ambig_code = '''
double t22_ambig(double x) { return x * 2.0; }
double t22_ambig(float x) { return x * 3.0; }
'''
    
    decl2 = ROOT.gInterpreter.Declare(ambig_code)
    result.observe("t22b_ambig_declare", decl2, "Ambiguous overloads declared")
    
    if decl2:
        # Test which one gets called with double literal
        rdf3 = ROOT.RDataFrame(1)
        rdf3 = rdf3.Define("val", "5.0")  # double literal
        
        try:
            rdf3 = rdf3.Define("r", "t22_ambig(val)")
            r = rdf3.Sum("r").GetValue()
            result.log(f"  t22_ambig(5.0): {r}")
            
            if r == 10.0:
                result.log("  → Resolved to double version", "INFO")
                result.observe("t22b_resolution", "double", "double literal → double overload")
            elif r == 15.0:
                result.log("  → Resolved to float version", "INFO")
                result.observe("t22b_resolution", "float", "double literal → float overload")
            else:
                result.log(f"  → Unexpected result: {r}", "WARN")
                result.observe("t22b_resolution", "unknown", f"Unexpected: {r}")
                
        except Exception as e:
            result.log(f"  Ambiguity caused error: {e}", "INFO")
            result.observe("t22b_resolution", "error", "Compiler rejects ambiguity")
    
    # =========================================================================
    # T22c: Document type coercion behavior
    # =========================================================================
    result.log("\n--- T22c: Type Coercion Documentation ---")
    
    # Test various type combinations
    coercion_tests = [
        ("5.0", "double literal"),
        ("5.0f", "float literal"),
        ("5", "int literal"),
        ("(double)5", "explicit double cast"),
        ("(float)5", "explicit float cast"),
    ]
    
    for expr, desc in coercion_tests:
        try:
            rdf_t = ROOT.RDataFrame(1)
            rdf_t = rdf_t.Define("v", expr)
            rdf_t = rdf_t.Define("r", "t22_ambig(v)")
            val = rdf_t.Sum("r").GetValue()
            result.log(f"  {desc} ({expr}): {val}")
        except Exception as e:
            result.log(f"  {desc} ({expr}): ERROR - {e}", "WARN")
    
    # Summary
    result.status = "PASSED"
    result.log("\n✅ T22 PASSED: Same-arity overload behavior documented", "PASS")
    result.action_items = [
        "Document type coercion rules in v7 spec",
        "Consider explicit type hints in DSL API",
    ]
    
    return result


# =============================================================================
# T23: Registry Idempotency
# =============================================================================

def test_t23_registry_idempotency():
    """
    T23: Registry Idempotency Under Repeated apply()
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: No re-declare attempts when apply() called multiple times.
    Practical failure mode - users call apply() iteratively.
    """
    result = TestResult(
        test_id="T23",
        title="Registry Idempotency (Repeated apply)",
        hypothesis="Multiple apply() calls do not cause re-declaration errors"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T23: Registry Idempotency")
    result.log("="*60)
    
    # =========================================================================
    # T23a: Register function once, apply multiple times
    # =========================================================================
    result.log("\n--- T23a: Multiple apply() Calls ---")
    
    dsl = MockDSLCompiler()
    dsl.register_function_cpp('''
        double t23_pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    func = dsl.get_function("t23_pt")
    result.log(f"Registered: {func.cpp_name}")
    
    # Create base RDataFrame
    rdf_base = ROOT.RDataFrame(100)
    rdf_base = rdf_base.Define("px", "sin(rdfentry_ * 0.1) * 10")
    rdf_base = rdf_base.Define("py", "cos(rdfentry_ * 0.1) * 10")
    
    # Apply multiple times
    results = []
    errors = []
    
    for i in range(5):
        try:
            rdf_i = dsl.apply(rdf_base)
            rdf_i = rdf_i.Define("track_pt", f"{func.cpp_name}(px, py)")
            mean_pt = rdf_i.Mean("track_pt").GetValue()
            results.append(mean_pt)
            result.log(f"  apply() #{i+1}: Mean(pt) = {mean_pt:.4f}", "PASS")
        except Exception as e:
            errors.append(str(e))
            result.log(f"  apply() #{i+1}: ERROR - {e}", "FAIL")
    
    result.observe("t23a_apply_count", len(results), "Successful apply() calls")
    result.observe("t23a_error_count", len(errors), "Failed apply() calls")
    
    # Verify all results are consistent
    if results:
        all_same = all(abs(r - results[0]) < 1e-10 for r in results)
        result.observe("t23a_results_consistent", all_same,
                       "All apply() calls produce same result")
    
    # =========================================================================
    # T23b: Check for re-declaration warnings
    # =========================================================================
    result.log("\n--- T23b: Re-declaration Check ---")
    
    # The MockDSL tracks declared functions
    declared_count = len(dsl._declared_cpp_names)
    result.log(f"  Declared function count: {declared_count}")
    result.observe("t23b_declared_once", declared_count == 1,
                   "Function declared only once despite multiple apply()")
    
    # =========================================================================
    # T23c: Derived RDataFrames
    # =========================================================================
    result.log("\n--- T23c: Derived RDataFrames ---")
    
    # Create chain of derived RDFs
    rdf1 = dsl.apply(rdf_base)
    rdf1 = rdf1.Define("pt1", f"{func.cpp_name}(px, py)")
    
    rdf2 = rdf1.Filter("pt1 > 5.0")
    rdf2 = rdf2.Define("pt2", f"{func.cpp_name}(px, py)")  # Same function
    
    rdf3 = rdf2.Filter("pt2 > 7.0")
    rdf3 = rdf3.Define("pt3", f"{func.cpp_name}(px, py)")  # Same function again
    
    try:
        count1 = rdf1.Count().GetValue()
        count2 = rdf2.Count().GetValue()
        count3 = rdf3.Count().GetValue()
        
        result.log(f"  RDF1 count: {count1}")
        result.log(f"  RDF2 count (pt > 5): {count2}")
        result.log(f"  RDF3 count (pt > 7): {count3}")
        
        result.observe("t23c_chain_works", True, "Derived RDFs work correctly")
    except Exception as e:
        result.log(f"  Derived RDF error: {e}", "FAIL")
        result.observe("t23c_chain_works", False, str(e))
    
    # Summary
    if len(errors) == 0:
        result.status = "PASSED"
        result.log("\n✅ T23 PASSED: Registry is idempotent", "PASS")
    else:
        result.status = "FAILED"
        result.log("\n❌ T23 FAILED: Re-declaration errors occurred", "FAIL")
    
    return result


# =============================================================================
# T24: Schema-RDF Mismatch
# =============================================================================

def test_t24_schema_mismatch():
    """
    T24: Schema-RDF Mismatch (Missing Column Error)
    
    Proposed By: Claude2
    Priority: P0
    
    Goal: Document error timing and message quality.
    Common user error - error timing affects UX.
    """
    result = TestResult(
        test_id="T24",
        title="Schema-RDF Mismatch",
        hypothesis="Missing column errors occur at predictable time "
                   "with clear error messages"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T24: Schema-RDF Mismatch")
    result.log("="*60)
    
    # =========================================================================
    # T24a: Missing column in Define expression
    # =========================================================================
    result.log("\n--- T24a: Missing Column in Define ---")
    
    # Create RDF without expected column
    rdf = ROOT.RDataFrame(10)
    rdf = rdf.Define("px", "1.0")  # Only px, no py
    
    # Try to define using missing column
    error_timing_a = "none"
    error_msg_a = ""
    
    try:
        # Define-time?
        rdf2 = rdf.Define("pt", "sqrt(px*px + py*py)")  # py doesn't exist
        result.log("  Define() accepted (lazy)", "INFO")
        
        try:
            # Evaluate-time?
            val = rdf2.Sum("pt").GetValue()
            result.log(f"  Sum() returned: {val}", "WARN")
            error_timing_a = "never"  # No error (unexpected)
        except Exception as e:
            error_timing_a = "evaluate"
            error_msg_a = str(e)
            result.log(f"  Error at evaluate time: {e}", "INFO")
            
    except Exception as e:
        error_timing_a = "define"
        error_msg_a = str(e)
        result.log(f"  Error at define time: {e}", "INFO")
    
    result.observe("t24a_error_timing", error_timing_a,
                   "When missing column error occurs")
    result.observe("t24a_error_msg", error_msg_a[:100] if error_msg_a else "none",
                   "Error message (truncated)")
    
    # Rate error message quality
    if error_msg_a:
        has_column_name = "py" in error_msg_a.lower()
        has_hint = "column" in error_msg_a.lower() or "not found" in error_msg_a.lower()
        result.observe("t24a_msg_has_column_name", has_column_name,
                       "Error mentions missing column name")
        result.observe("t24a_msg_has_hint", has_hint,
                       "Error provides helpful hint")
    
    # =========================================================================
    # T24b: Type mismatch
    # =========================================================================
    result.log("\n--- T24b: Type Mismatch ---")
    
    # Declare function expecting specific type
    ROOT.gInterpreter.Declare('''
        double t24_expects_vec(const ROOT::VecOps::RVec<double>& v) {
            return v.size();
        }
    ''')
    
    rdf3 = ROOT.RDataFrame(10)
    rdf3 = rdf3.Define("scalar_val", "5.0")  # Scalar, not vector
    
    error_timing_b = "none"
    error_msg_b = ""
    
    try:
        rdf3 = rdf3.Define("result", "t24_expects_vec(scalar_val)")
        
        try:
            val = rdf3.Sum("result").GetValue()
            error_timing_b = "never"
            result.log(f"  No error! Result: {val}", "WARN")
        except Exception as e:
            error_timing_b = "evaluate"
            error_msg_b = str(e)
            result.log(f"  Error at evaluate: {e}", "INFO")
            
    except Exception as e:
        error_timing_b = "define"
        error_msg_b = str(e)
        result.log(f"  Error at define: {e}", "INFO")
    
    result.observe("t24b_error_timing", error_timing_b,
                   "When type mismatch error occurs")
    
    # =========================================================================
    # T24c: Wrong column type
    # =========================================================================
    result.log("\n--- T24c: Wrong Column Type ---")
    
    # Function expecting double, column is string-ish
    ROOT.gInterpreter.Declare('''
        double t24_double_func(double x) {
            return x * 2;
        }
    ''')
    
    rdf4 = ROOT.RDataFrame(10)
    # Create a vector column, try to use with scalar function
    rdf4 = rdf4.Define("vec_col", "ROOT::VecOps::RVec<double>{1.0, 2.0}")
    
    error_timing_c = "none"
    
    try:
        rdf4 = rdf4.Define("result", "t24_double_func(vec_col)")
        
        try:
            val = rdf4.Sum("result").GetValue()
            error_timing_c = "never"
            result.log(f"  No error! Result: {val}", "WARN")
        except Exception as e:
            error_timing_c = "evaluate"
            result.log(f"  Error at evaluate: {e}", "INFO")
            
    except Exception as e:
        error_timing_c = "define"
        result.log(f"  Error at define: {e}", "INFO")
    
    result.observe("t24c_error_timing", error_timing_c,
                   "When wrong type error occurs")
    
    # =========================================================================
    # Summary
    # =========================================================================
    result.log("\n--- T24 Summary ---")
    result.log(f"  Missing column: Error at {error_timing_a}")
    result.log(f"  Type mismatch:  Error at {error_timing_b}")
    result.log(f"  Wrong type:     Error at {error_timing_c}")
    
    # Document for v7 spec
    result.action_items = [
        f"Document: Missing column errors occur at {error_timing_a} time",
        "Consider adding DSL-level validation for better error messages",
        "Add examples to user documentation",
    ]
    
    result.status = "PASSED"
    result.log("\n✅ T24 PASSED: Error timing documented", "PASS")
    
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Tests T22-T24")
    print("="*70)
    
    results = {}
    
    # T22: Same-Arity Overload
    print("\n" + "="*70)
    results["T22"] = test_t22_same_arity_overload()
    
    # T23: Registry Idempotency
    print("\n" + "="*70)
    results["T23"] = test_t23_registry_idempotency()
    
    # T24: Schema Mismatch
    print("\n" + "="*70)
    results["T24"] = test_t24_schema_mismatch()
    
    # Summary
    print("\n" + "="*70)
    print("T22-T24 SUMMARY")
    print("="*70)
    
    for test_id, result in results.items():
        status_icon = "✅" if result.status == "PASSED" else "⚠️" if result.status == "PARTIAL" else "❌"
        print(f"{status_icon} {test_id}: {result.status}")
    
    passed = sum(1 for r in results.values() if r.status in ["PASSED", "PARTIAL"])
    print(f"\n{passed}/{len(results)} tests passed")
    
    sys.exit(0 if passed == len(results) else 1)
