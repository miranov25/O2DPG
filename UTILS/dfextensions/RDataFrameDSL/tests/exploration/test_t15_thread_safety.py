#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T15: Thread Safety (ImplicitMT)

Priority: P0 - CRITICAL (Run FIRST)
Proposed By: Gemini1
Category: Parallelism

Rationale:
    "T15 is a 'Crash Test'. If it fails, it might require fundamental 
    changes to code generation. Discovering this in B1 is too late."

Goal: Verify DSL-generated functions are thread-safe under EnableImplicitMT.

Key Requirements (from review):
    - Use deterministic columns (NOT gRandom - not thread-safe)
    - Run single-thread baseline first for comparison
    - Test with 4/8 threads
    - Verify no segfaults, no races, deterministic results
"""

import os
import sys
import math
from datetime import datetime

# Import test infrastructure
from test_infrastructure import (
    setup_test_env, get_workspace, TestResult,
    MockDSLCompiler, check_root_available, get_root_version
)


def test_t15_thread_safety():
    """
    T15: Thread Safety Under ROOT::EnableImplicitMT
    
    Hypothesis:
        - Generated JIT functions are stateless
        - No static state or globals
        - Results are deterministic under multithreading
    """
    result = TestResult(
        test_id="T15",
        title="Thread Safety (ImplicitMT)",
        hypothesis="DSL-generated functions are thread-safe and produce "
                   "deterministic results under EnableImplicitMT"
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
    result.log("T15: Thread Safety Under ROOT::EnableImplicitMT")
    result.log("="*60)
    
    # =========================================================================
    # T15a: Single-Thread Baseline
    # =========================================================================
    result.log("\n--- T15a: Single-Thread Baseline ---")
    
    # Ensure single-threaded first
    ROOT.ROOT.DisableImplicitMT()
    result.observe("t15a_mt_disabled", True, "Start with single-thread baseline")
    
    # Create test functions via gInterpreter (simulating DSL)
    func_code = '''
#include <cmath>

// Thread-safe: no static/global state
double t15_pt(double px, double py) {
    return sqrt(px*px + py*py);
}

double t15_complex_calc(double a, double b, double c) {
    return sqrt(a*a + b*b) + c;
}

double t15_eta(double px, double py, double pz) {
    double pt = sqrt(px*px + py*py);
    double p = sqrt(px*px + py*py + pz*pz);
    if (p == pt) return 0.0;  // Avoid division issues
    return 0.5 * log((p + pz) / (p - pz + 1e-10));
}
'''
    
    decl_result = ROOT.gInterpreter.Declare(func_code)
    result.observe("t15a_declare_result", decl_result, "Function declaration success")
    
    if not decl_result:
        result.status = "FAILED"
        result.log("Function declaration failed", "FAIL")
        return result
    
    # Create RDataFrame with DETERMINISTIC columns (NOT gRandom!)
    # Use rdfentry_ for reproducibility
    N_ENTRIES = 100000
    
    rdf_st = ROOT.RDataFrame(N_ENTRIES)
    
    # Deterministic columns based on entry number
    rdf_st = rdf_st.Define("px", "sin(rdfentry_ * 0.1) * 10.0")
    rdf_st = rdf_st.Define("py", "cos(rdfentry_ * 0.1) * 10.0")
    rdf_st = rdf_st.Define("pz", "sin(rdfentry_ * 0.05) * 20.0")
    
    # Apply test functions
    rdf_st = rdf_st.Define("track_pt", "t15_pt(px, py)")
    rdf_st = rdf_st.Define("track_eta", "t15_eta(px, py, pz)")
    rdf_st = rdf_st.Define("result", "t15_complex_calc(track_pt, px, py)")
    
    # Get single-thread results
    sum_st = rdf_st.Sum("result").GetValue()
    mean_pt_st = rdf_st.Mean("track_pt").GetValue()
    
    result.log(f"Single-thread Sum(result): {sum_st}", "PASS")
    result.log(f"Single-thread Mean(pt): {mean_pt_st}", "PASS")
    result.observe("t15a_sum_single_thread", sum_st, "Baseline for comparison")
    result.observe("t15a_mean_pt_single_thread", mean_pt_st, "Baseline pt")
    
    # =========================================================================
    # T15b: Multi-Thread Test (4 threads)
    # =========================================================================
    result.log("\n--- T15b: Multi-Thread Test (4 threads) ---")
    
    ROOT.ROOT.EnableImplicitMT(4)
    n_threads = ROOT.ROOT.GetThreadPoolSize()
    result.observe("t15b_thread_count", n_threads, "Actual thread pool size")
    
    # Run multiple times to detect races
    mt_results = []
    NUM_RUNS = 5
    
    for i in range(NUM_RUNS):
        rdf_mt = ROOT.RDataFrame(N_ENTRIES)
        rdf_mt = rdf_mt.Define("px", "sin(rdfentry_ * 0.1) * 10.0")
        rdf_mt = rdf_mt.Define("py", "cos(rdfentry_ * 0.1) * 10.0")
        rdf_mt = rdf_mt.Define("pz", "sin(rdfentry_ * 0.05) * 20.0")
        rdf_mt = rdf_mt.Define("track_pt", "t15_pt(px, py)")
        rdf_mt = rdf_mt.Define("track_eta", "t15_eta(px, py, pz)")
        rdf_mt = rdf_mt.Define("result", "t15_complex_calc(track_pt, px, py)")
        
        sum_mt = rdf_mt.Sum("result").GetValue()
        mt_results.append(sum_mt)
        result.log(f"  Run {i+1}: Sum = {sum_mt}")
    
    # Check determinism
    all_same = len(set(mt_results)) == 1
    result.observe("t15b_all_results_identical", all_same, 
                   "All MT runs produce identical results")
    
    if all_same:
        result.log("Multi-thread results are deterministic", "PASS")
    else:
        result.log(f"NON-DETERMINISTIC: {mt_results}", "FAIL")
        result.status = "FAILED"
        return result
    
    # Compare with single-thread
    diff = abs(mt_results[0] - sum_st)
    tolerance = abs(sum_st) * 1e-10  # Floating point tolerance
    
    result.observe("t15b_st_mt_difference", diff, "Difference between ST and MT")
    
    if diff < tolerance:
        result.log(f"MT result matches ST (diff={diff})", "PASS")
    else:
        result.log(f"MT result differs from ST! diff={diff}", "WARN")
    
    # =========================================================================
    # T15c: Higher Thread Count (8 threads)
    # =========================================================================
    result.log("\n--- T15c: Stress Test (8 threads) ---")
    
    ROOT.ROOT.DisableImplicitMT()
    ROOT.ROOT.EnableImplicitMT(8)
    n_threads_8 = ROOT.ROOT.GetThreadPoolSize()
    result.observe("t15c_thread_count_8", n_threads_8, "8-thread pool size")
    
    # Larger dataset for stress
    N_STRESS = 500000
    
    rdf_stress = ROOT.RDataFrame(N_STRESS)
    rdf_stress = rdf_stress.Define("px", "sin(rdfentry_ * 0.1) * 10.0")
    rdf_stress = rdf_stress.Define("py", "cos(rdfentry_ * 0.1) * 10.0")
    rdf_stress = rdf_stress.Define("pz", "sin(rdfentry_ * 0.05) * 20.0")
    rdf_stress = rdf_stress.Define("track_pt", "t15_pt(px, py)")
    rdf_stress = rdf_stress.Define("result", "t15_complex_calc(track_pt, px, py)")
    
    try:
        sum_stress = rdf_stress.Sum("result").GetValue()
        result.log(f"Stress test Sum: {sum_stress}", "PASS")
        result.observe("t15c_stress_test_passed", True, "No crash under 8 threads")
    except Exception as e:
        result.log(f"Stress test failed: {e}", "FAIL")
        result.observe("t15c_stress_test_passed", False, str(e))
        result.status = "FAILED"
        return result
    
    # =========================================================================
    # T15d: Negative Test - Mutate After Apply (GPT5 enhancement)
    # =========================================================================
    result.log("\n--- T15d: Registration During Computation ---")
    
    # This tests whether registering new functions during computation is safe
    # In real DSL, this would be prevented by apply() locking
    
    rdf_neg = ROOT.RDataFrame(10000)
    rdf_neg = rdf_neg.Define("x", "sin(rdfentry_ * 0.1)")
    rdf_neg = rdf_neg.Define("y", "t15_pt(x, x)")
    
    # Start lazy computation
    action = rdf_neg.Sum("y")
    
    # Try to declare new function while action pending
    new_func = '''
double t15_new_func(double x) {
    return x * 3.0;
}
'''
    
    try:
        # This should work - Cling handles this
        ROOT.gInterpreter.Declare(new_func)
        result.log("New registration during pending action: allowed", "INFO")
        result.observe("t15d_registration_during_compute", "ALLOWED",
                       "Cling allows registration during pending computation")
    except Exception as e:
        result.log(f"Registration rejected: {e}", "INFO")
        result.observe("t15d_registration_during_compute", "REJECTED", str(e))
    
    # Complete the action
    try:
        sum_val = action.GetValue()
        result.log(f"Action completed: {sum_val}", "PASS")
    except Exception as e:
        result.log(f"Action failed: {e}", "FAIL")
    
    # =========================================================================
    # Summary
    # =========================================================================
    ROOT.ROOT.DisableImplicitMT()  # Clean up
    
    result.log("\n" + "="*60)
    result.log("T15 SUMMARY: Thread Safety")
    result.log("="*60)
    
    # Determine final status
    if all_same and diff < tolerance:
        result.status = "PASSED"
        result.log("✅ T15 PASSED: Thread-safe, deterministic under ImplicitMT", "PASS")
    else:
        result.status = "FAILED"
        result.log("❌ T15 FAILED: Thread safety issues detected", "FAIL")
    
    result.action_items = [
        "Ensure all DSL functions are stateless",
        "Document thread safety requirements",
        "Consider thread-local storage if state needed",
    ]
    
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T15: Thread Safety (ImplicitMT)")
    print("="*70)
    print("⚠️  This test should run FIRST - may crash if thread-unsafe")
    print()
    
    result = test_t15_thread_safety()
    
    print("\n")
    print(result.to_markdown())
    
    # Exit code
    sys.exit(0 if result.status == "PASSED" else 1)
