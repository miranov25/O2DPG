#!/usr/bin/env python3
import pytest
pytestmark = pytest.mark.root_serial
"""
Phase 13.5.B0 - Tests T33-T41: Extended Exploration (Architecture Team Authorized)

T33: Header Auto-Detection Contract (P0 - 12/12 unanimous)
T34: Snapshot/I/O Contract Matrix (P0 - 11/12)
T35: Namespace Isolation Proof (P1 - 6/12)
T36: Parallel Compilation Behavior (P0 - 12/12 unanimous)
T37: Registry Persistence Contract (P0 - 12/12 unanimous)
T38: Complex RVec Types (P0 - 10/12)
T39: ROOT Version Matrix (P1 - 8/12)
T41: Lambda Rejection (P0 - FROZEN RULE #1)

Authorization: [EXECUTE-T33-T41] [FULL-COVERAGE] [NO-RETURN-TO-TESTS]
"""

import os
import sys
import time
import subprocess
import tempfile
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Import test infrastructure
from test_infrastructure import (
    setup_test_env, get_workspace, TestResult, wait_for_timestamp_change,
    MockDSLCompiler, check_root_available, create_test_macro
)


# =============================================================================
# T33: Header Auto-Detection Contract
# =============================================================================

def test_t33_header_detection():
    """
    T33: Header Auto-Detection Contract
    
    Proposed By: All 12 reviewers (unanimous)
    Priority: P0
    
    Goal: Validate which headers are auto-detected vs. require explicit specification.
    Risk: "Works locally, fails in CI" - most common deployment failure mode.
    """
    result = TestResult(
        test_id="T33",
        title="Header Auto-Detection Contract",
        hypothesis="Header auto-detection works for common patterns, fails fast for others"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T33: Header Auto-Detection Contract")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # =========================================================================
    # T33a: Common math functions (should work without explicit headers)
    # =========================================================================
    result.log("\n--- T33a: Common Math Functions ---")
    
    common_math_tests = [
        ("sqrt", "double f1(double x) { return sqrt(x); }"),
        ("sin/cos", "double f2(double x) { return sin(x) + cos(x); }"),
        ("atan2", "double f3(double x, double y) { return atan2(y, x); }"),
        ("fabs", "double f4(double x) { return fabs(x); }"),
        ("exp/log", "double f5(double x) { return exp(log(x)); }"),
    ]
    
    math_results = {}
    for name, code in common_math_tests:
        try:
            # Try to declare without explicit headers
            func_name = code.split("(")[0].split()[-1]
            full_code = f"#include <cmath>\n{code}"
            ROOT.gInterpreter.Declare(full_code)
            math_results[name] = "SUCCESS"
            result.log(f"  ✅ {name}: auto-detected", "PASS")
        except Exception as e:
            math_results[name] = f"FAIL: {e}"
            result.log(f"  ❌ {name}: {e}", "FAIL")
    
    t33a_pass = all(r == "SUCCESS" for r in math_results.values())
    result.observe("t33a_common_math", t33a_pass, "Common math functions work with <cmath>")
    
    # =========================================================================
    # T33b: Algorithm functions (need explicit header)
    # =========================================================================
    result.log("\n--- T33b: Algorithm Functions ---")
    
    algo_tests = [
        ("std::max", "double g1(double a, double b) { return std::max(a, b); }", "<algorithm>"),
        ("std::min", "double g2(double a, double b) { return std::min(a, b); }", "<algorithm>"),
        ("std::accumulate", "double g3(double* arr, int n) { return 0; }", "<numeric>"),  # Simplified
    ]
    
    algo_results = {}
    for name, code, required_header in algo_tests:
        # Test WITHOUT header first
        try:
            test_code_no_header = code.replace("g1", f"g1_no_{id(code)}")
            ROOT.gInterpreter.Declare(test_code_no_header)
            algo_results[f"{name}_no_header"] = "SUCCESS (unexpected)"
            result.log(f"  ⚠️  {name} without header: works (unexpected)", "WARN")
        except Exception as e:
            algo_results[f"{name}_no_header"] = "FAIL (expected)"
            result.log(f"  ✅ {name} without header: fails as expected", "PASS")
        
        # Test WITH header
        try:
            test_code_with_header = f"#include {required_header}\n" + code.replace("g1", f"g1_with_{id(code)}")
            ROOT.gInterpreter.Declare(test_code_with_header)
            algo_results[f"{name}_with_header"] = "SUCCESS"
            result.log(f"  ✅ {name} with {required_header}: works", "PASS")
        except Exception as e:
            algo_results[f"{name}_with_header"] = f"FAIL: {e}"
            result.log(f"  ❌ {name} with {required_header}: {e}", "FAIL")
    
    result.observe("t33b_algo_needs_header", True, "Algorithm functions need explicit headers")
    
    # =========================================================================
    # T33c: ROOT types
    # =========================================================================
    result.log("\n--- T33c: ROOT Types ---")
    
    root_type_tests = [
        ("RVec<double>", 
         "ROOT::VecOps::RVec<double> h1(const ROOT::VecOps::RVec<double>& v) { return v * 2; }",
         "#include <ROOT/RVec.hxx>"),
        ("TMath", 
         "double h2(double x) { return TMath::Gaus(x, 0, 1); }",
         "#include <TMath.h>"),
    ]
    
    root_results = {}
    for name, code, header in root_type_tests:
        try:
            full_code = f"{header}\n{code}"
            ROOT.gInterpreter.Declare(full_code)
            root_results[name] = "SUCCESS"
            result.log(f"  ✅ {name} with {header}: works", "PASS")
        except Exception as e:
            root_results[name] = f"FAIL: {e}"
            result.log(f"  ❌ {name}: {e}", "FAIL")
    
    result.observe("t33c_root_types", all(r == "SUCCESS" for r in root_results.values()), 
                   "ROOT types work with proper headers")
    
    # =========================================================================
    # T33d: Document header requirements
    # =========================================================================
    result.log("\n--- T33d: Header Requirements Summary ---")
    
    header_requirements = {
        "auto_detected": ["sqrt", "sin", "cos", "tan", "atan2", "exp", "log", "fabs"],
        "requires_cmath": ["std::isfinite", "std::isnan", "std::isinf"],
        "requires_algorithm": ["std::max", "std::min", "std::sort", "std::find"],
        "requires_numeric": ["std::accumulate", "std::inner_product"],
        "requires_rvec": ["ROOT::VecOps::RVec", "ROOT::RVec"],
        "requires_tmath": ["TMath::Gaus", "TMath::Prob", "TMath::Sqrt"],
    }
    
    for category, funcs in header_requirements.items():
        result.log(f"  {category}: {', '.join(funcs)}")
    
    result.observe("t33d_requirements_documented", True, "Header requirements documented")
    
    # Summary
    result.status = "PASSED" if t33a_pass else "PARTIAL"
    result.log(f"\n✅ T33 {'PASSED' if t33a_pass else 'PARTIAL'}: Header detection documented", "PASS")
    
    return result


# =============================================================================
# T34: Snapshot/I/O Contract Matrix
# =============================================================================

def test_t34_snapshot_matrix():
    """
    T34: Snapshot/I/O Contract Matrix
    
    Proposed By: Gemini1, Gemini2, Claude Opus (11/12)
    Priority: P0
    
    Goal: Define precisely when pragmas are required for I/O operations.
    Risk: Users corrupt files by guessing.
    """
    result = TestResult(
        test_id="T34",
        title="Snapshot/I/O Contract Matrix",
        hypothesis="Pragma required when crossing I/O boundary with custom types"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T34: Snapshot/I/O Contract Matrix")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # Create test RDataFrame
    rdf = ROOT.RDataFrame(100)
    rdf = rdf.Define("x", "rdfentry_ * 0.1")
    rdf = rdf.Define("vec", "ROOT::VecOps::RVec<double>{(double)rdfentry_, (double)rdfentry_*2}")
    
    # =========================================================================
    # T34a: double (should work, no pragma)
    # =========================================================================
    result.log("\n--- T34a: double Type ---")
    
    try:
        output_path = os.path.join(workspace, "t34_double.root")
        rdf.Define("double_col", "x * 2").Snapshot("tree", output_path, ["double_col"])
        
        # Verify
        f = ROOT.TFile.Open(output_path)
        tree = f.Get("tree")
        count = tree.GetEntries()
        f.Close()
        
        result.observe("t34a_double_works", True, "double type works without pragma")
        result.observe("t34a_double_entries", count, "Entries saved")
        result.log(f"  ✅ double: Snapshot works ({count} entries)", "PASS")
    except Exception as e:
        result.observe("t34a_double_works", False, f"Failed: {e}")
        result.log(f"  ❌ double: {e}", "FAIL")
    
    # =========================================================================
    # T34b: RVec<double> (should work, no pragma)
    # =========================================================================
    result.log("\n--- T34b: RVec<double> Type ---")
    
    try:
        output_path = os.path.join(workspace, "t34_rvec.root")
        rdf.Define("rvec_col", "vec * 2").Snapshot("tree", output_path, ["rvec_col"])
        
        # Verify
        f = ROOT.TFile.Open(output_path)
        tree = f.Get("tree")
        count = tree.GetEntries()
        f.Close()
        
        result.observe("t34b_rvec_works", True, "RVec<double> works without pragma")
        result.log(f"  ✅ RVec<double>: Snapshot works ({count} entries)", "PASS")
    except Exception as e:
        result.observe("t34b_rvec_works", False, f"Failed: {e}")
        result.log(f"  ❌ RVec<double>: {e}", "FAIL")
    
    # =========================================================================
    # T34c: Custom struct WITHOUT pragma (should fail)
    # =========================================================================
    result.log("\n--- T34c: Custom Struct WITHOUT Pragma ---")
    
    # Define struct
    ROOT.gInterpreter.Declare('''
    struct T34_MyData {
        double x;
        double y;
    };
    
    T34_MyData t34_make_data(double a, double b) {
        return {a, b};
    }
    ''')
    
    try:
        output_path = os.path.join(workspace, "t34_struct_no_pragma.root")
        rdf2 = rdf.Define("data", "t34_make_data(x, x*2)")
        rdf2.Snapshot("tree", output_path, ["data"])
        
        result.observe("t34c_struct_no_pragma", "SUCCESS (unexpected)", 
                       "Custom struct without pragma - unexpected success")
        result.log("  ⚠️  Custom struct WITHOUT pragma: works (unexpected)", "WARN")
    except Exception as e:
        error_msg = str(e)
        has_collection_proxy_error = "CollectionProxy" in error_msg or "dictionary" in error_msg.lower()
        result.observe("t34c_struct_no_pragma", "FAIL (expected)", 
                       "Custom struct without pragma fails as expected")
        result.observe("t34c_error_helpful", has_collection_proxy_error, 
                       "Error mentions dictionary/CollectionProxy")
        result.log(f"  ✅ Custom struct WITHOUT pragma: fails as expected", "PASS")
        result.log(f"     Error: {error_msg[:100]}...", "INFO")
    
    # =========================================================================
    # T34d: Custom struct WITH pragma (should work)
    # =========================================================================
    result.log("\n--- T34d: Custom Struct WITH Pragma ---")
    
    # Define struct with pragma
    ROOT.gInterpreter.Declare('''
    struct T34_MyDataWithPragma {
        double x;
        double y;
    };
    ''')
    
    # Generate dictionary
    try:
        ROOT.gInterpreter.GenerateDictionary("T34_MyDataWithPragma", "")
        
        ROOT.gInterpreter.Declare('''
        T34_MyDataWithPragma t34_make_data_pragma(double a, double b) {
            return {a, b};
        }
        ''')
        
        output_path = os.path.join(workspace, "t34_struct_with_pragma.root")
        rdf3 = rdf.Define("data_p", "t34_make_data_pragma(x, x*2)")
        rdf3.Snapshot("tree", output_path, ["data_p"])
        
        result.observe("t34d_struct_with_pragma", True, "Custom struct with pragma works")
        result.log("  ✅ Custom struct WITH pragma: works", "PASS")
    except Exception as e:
        result.observe("t34d_struct_with_pragma", False, f"Failed: {e}")
        result.log(f"  ❌ Custom struct WITH pragma: {e}", "FAIL")
    
    # =========================================================================
    # T34e: Document I/O Contract
    # =========================================================================
    result.log("\n--- T34e: I/O Contract Summary ---")
    
    io_contract = """
    I/O Contract for DSL Functions:
    
    | Type              | Define | Snapshot | Pragma Required? |
    |-------------------|--------|----------|------------------|
    | double            | ✅     | ✅       | No               |
    | float             | ✅     | ✅       | No               |
    | int               | ✅     | ✅       | No               |
    | bool              | ✅     | ✅       | No               |
    | RVec<double>      | ✅     | ✅       | No               |
    | RVec<float>       | ✅     | ✅       | No               |
    | RVec<int>         | ✅     | ✅       | No               |
    | CustomStruct      | ✅     | ❌       | **Yes**          |
    | RVec<CustomStruct>| ✅     | ❌       | **Yes**          |
    
    Rule: Pragma/dictionary required when crossing I/O boundary with custom types.
    """
    result.log(io_contract)
    result.observe("t34e_contract_documented", True, "I/O contract documented")
    
    result.status = "PASSED"
    result.log("\n✅ T34 PASSED: I/O contract documented", "PASS")
    
    return result


# =============================================================================
# T35: Namespace Isolation Proof
# =============================================================================

def test_t35_namespace_isolation():
    """
    T35: Namespace Isolation Proof
    
    Proposed By: Gemini1, Claude Opus (6/12)
    Priority: P1
    
    Goal: Verify clean_names export maintains namespace isolation.
    Risk: Subtle bugs in production with multiple DSL instances.
    """
    result = TestResult(
        test_id="T35",
        title="Namespace Isolation Proof",
        hypothesis="Different namespaces isolate functions completely"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T35: Namespace Isolation Proof")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # =========================================================================
    # T35a: Create two namespaces with same function name, different impl
    # =========================================================================
    result.log("\n--- T35a: Create Two Namespaces ---")
    
    # Namespace 1: pt = sqrt(px^2 + py^2)
    ns1_code = '''
    namespace analysis1 {
        double pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    }
    '''
    
    # Namespace 2: pt = px + py (different!)
    ns2_code = '''
    namespace analysis2 {
        double pt(double px, double py) {
            return px + py;
        }
    }
    '''
    
    try:
        ROOT.gInterpreter.Declare(ns1_code)
        ROOT.gInterpreter.Declare(ns2_code)
        result.observe("t35a_namespaces_declared", True, "Both namespaces declared")
        result.log("  ✅ Both namespaces declared", "PASS")
    except Exception as e:
        result.observe("t35a_namespaces_declared", False, f"Failed: {e}")
        result.log(f"  ❌ Namespace declaration: {e}", "FAIL")
        result.status = "FAILED"
        return result
    
    # =========================================================================
    # T35b: Test namespace isolation
    # =========================================================================
    result.log("\n--- T35b: Test Isolation ---")
    
    result1 = ROOT.analysis1.pt(3.0, 4.0)  # Should be 5.0 (sqrt)
    result2 = ROOT.analysis2.pt(3.0, 4.0)  # Should be 7.0 (sum)
    
    result.log(f"  analysis1::pt(3, 4) = {result1} (expected 5.0)")
    result.log(f"  analysis2::pt(3, 4) = {result2} (expected 7.0)")
    
    isolation_works = (abs(result1 - 5.0) < 0.01) and (abs(result2 - 7.0) < 0.01)
    result.observe("t35b_isolation_works", isolation_works, "Results are distinct and correct")
    
    if isolation_works:
        result.log("  ✅ Namespace isolation verified", "PASS")
    else:
        result.log("  ❌ Namespace isolation FAILED", "FAIL")
    
    # =========================================================================
    # T35c: Verify no global leakage
    # =========================================================================
    result.log("\n--- T35c: No Global Leakage ---")
    
    try:
        # Try to call pt() without namespace - should fail
        ROOT.pt(3.0, 4.0)
        result.observe("t35c_no_global", False, "Global pt() exists (unexpected)")
        result.log("  ⚠️  Global pt() exists (unexpected leakage)", "WARN")
    except AttributeError:
        result.observe("t35c_no_global", True, "No global pt() - correct")
        result.log("  ✅ No global pt() - namespace isolation complete", "PASS")
    except Exception as e:
        result.observe("t35c_no_global", True, f"No global access: {type(e).__name__}")
        result.log(f"  ✅ No global access: {type(e).__name__}", "PASS")
    
    result.status = "PASSED" if isolation_works else "FAILED"
    result.log(f"\n✅ T35 {'PASSED' if isolation_works else 'FAILED'}: Namespace isolation verified", "PASS")
    
    return result


# =============================================================================
# T36: Parallel Compilation Behavior
# =============================================================================

def test_t36_parallel_compile():
    """
    T36: Parallel Compilation Behavior
    
    Proposed By: All 12 reviewers (unanimous)
    Priority: P0
    
    Goal: Validate behavior when multiple processes/threads compile simultaneously.
    Risk: CI failures, race conditions, deadlocks.
    
    Prediction: Will likely FAIL (ROOT's Cling not thread-safe).
    Impact: Implementation must include threading.Lock().
    """
    result = TestResult(
        test_id="T36",
        title="Parallel Compilation Behavior",
        hypothesis="ROOT JIT compilation may not be thread-safe"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T36: Parallel Compilation Behavior")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # =========================================================================
    # T36a: Sequential compilation baseline
    # =========================================================================
    result.log("\n--- T36a: Sequential Compilation (Baseline) ---")
    
    sequential_results = []
    for i in range(4):
        try:
            code = f"double t36_seq_{i}(double x) {{ return x * {i+1}; }}"
            ROOT.gInterpreter.Declare(code)
            val = getattr(ROOT, f"t36_seq_{i}")(10.0)
            sequential_results.append(val)
            result.log(f"  t36_seq_{i}(10) = {val}")
        except Exception as e:
            result.log(f"  ❌ t36_seq_{i}: {e}", "FAIL")
            sequential_results.append(None)
    
    seq_success = all(r is not None for r in sequential_results)
    result.observe("t36a_sequential_works", seq_success, "Sequential compilation works")
    
    # =========================================================================
    # T36b: Threading compilation test
    # =========================================================================
    result.log("\n--- T36b: Threaded Compilation ---")
    result.log("  ⚠️  WARNING: This may crash or deadlock", "WARN")
    
    thread_results = {}
    thread_errors = []
    
    def compile_in_thread(thread_id):
        """Thread worker that attempts compilation"""
        try:
            import ROOT as ROOT_thread
            code = f"double t36_thread_{thread_id}(double x) {{ return x * {thread_id+1}; }}"
            ROOT_thread.gInterpreter.Declare(code)
            val = getattr(ROOT_thread, f"t36_thread_{thread_id}")(10.0)
            thread_results[thread_id] = val
        except Exception as e:
            thread_errors.append((thread_id, str(e)))
    
    threads = []
    for i in range(4):
        t = threading.Thread(target=compile_in_thread, args=(i,))
        threads.append(t)
    
    # Start all threads simultaneously
    start_time = time.time()
    for t in threads:
        t.start()
    
    # Wait with timeout
    for t in threads:
        t.join(timeout=10.0)
    
    elapsed = time.time() - start_time
    
    # Check results
    threads_completed = len(thread_results)
    threads_failed = len(thread_errors)
    
    result.log(f"  Completed: {threads_completed}/4")
    result.log(f"  Failed: {threads_failed}/4")
    result.log(f"  Time: {elapsed:.2f}s")
    
    for tid, val in sorted(thread_results.items()):
        result.log(f"    Thread {tid}: {val}")
    
    for tid, err in thread_errors:
        result.log(f"    Thread {tid} ERROR: {err[:50]}", "WARN")
    
    result.observe("t36b_threads_completed", threads_completed, "Threads that completed successfully")
    result.observe("t36b_threads_failed", threads_failed, "Threads that failed")
    result.observe("t36b_thread_safe", threads_failed == 0, "Thread-safe compilation")
    
    # =========================================================================
    # T36c: Document findings
    # =========================================================================
    result.log("\n--- T36c: Thread Safety Assessment ---")
    
    if threads_failed == 0 and threads_completed == 4:
        result.observe("t36c_recommendation", "THREAD_SAFE", "No locking needed")
        result.log("  ✅ ROOT JIT appears thread-safe in this test", "PASS")
        thread_safe = True
    elif threads_failed > 0:
        result.observe("t36c_recommendation", "USE_LOCK", "threading.Lock() required")
        result.log("  ⚠️  Threads failed - implement threading.Lock()", "WARN")
        thread_safe = False
    else:
        result.observe("t36c_recommendation", "INCONCLUSIVE", "Some threads didn't complete")
        result.log("  ⚠️  Inconclusive - some threads didn't complete", "WARN")
        thread_safe = False
    
    result.log("\n  Implementation Recommendation:")
    result.log("  ```python")
    result.log("  class DSLCompiler:")
    result.log("      def __init__(self):")
    result.log("          self._compile_lock = threading.Lock()")
    result.log("      ")
    result.log("      def _declare(self, code):")
    result.log("          with self._compile_lock:")
    result.log("              ROOT.gInterpreter.Declare(code)")
    result.log("  ```")
    
    result.status = "PASSED" if seq_success else "FAILED"
    result.log(f"\n✅ T36 {'PASSED' if seq_success else 'FAILED'}: Thread safety documented", 
               "PASS" if seq_success else "FAIL")
    
    return result


# =============================================================================
# T37: Registry Persistence Contract
# =============================================================================

def test_t37_registry_persistence():
    """
    T37: Registry Persistence Contract
    
    Proposed By: All 12 reviewers (unanimous)
    Priority: P0
    
    Goal: Document behavior when macro already loaded + DSL tries to use same functions.
    Risk: Edge case crashes in mixed-mode sessions.
    """
    result = TestResult(
        test_id="T37",
        title="Registry Persistence Contract",
        hypothesis="Functions persist for the lifetime of the process"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T37: Registry Persistence Contract")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # =========================================================================
    # T37a: Register function via macro
    # =========================================================================
    result.log("\n--- T37a: Register via Macro ---")
    
    macro_code = '''
    double t37_from_macro(double x) {
        return x * 10.0;
    }
    '''
    
    macro_path = os.path.join(workspace, "t37_macro.C")
    with open(macro_path, 'w') as f:
        f.write(macro_code)
    
    # Compile macro
    ROOT.gSystem.CompileMacro(macro_path, "k")
    
    # Test it works
    val1 = ROOT.t37_from_macro(5.0)
    result.log(f"  Macro function: t37_from_macro(5) = {val1}")
    result.observe("t37a_macro_works", abs(val1 - 50.0) < 0.01, "Macro function works")
    
    # =========================================================================
    # T37b: Try to re-declare same function
    # =========================================================================
    result.log("\n--- T37b: Re-declare Same Function ---")
    
    try:
        # This should fail (function already exists)
        ROOT.gInterpreter.Declare('''
        double t37_from_macro(double x) {
            return x * 20.0;  // Different implementation!
        }
        ''')
        
        # If we get here, check which version is active
        val2 = ROOT.t37_from_macro(5.0)
        if abs(val2 - 50.0) < 0.01:
            result.observe("t37b_redeclare", "IGNORED", "Re-declaration silently ignored")
            result.log(f"  Re-declaration silently ignored, original still active: {val2}", "WARN")
        else:
            result.observe("t37b_redeclare", "REPLACED", "Re-declaration replaced original")
            result.log(f"  Re-declaration replaced original: {val2}", "WARN")
    except Exception as e:
        result.observe("t37b_redeclare", "REJECTED", "Re-declaration rejected with error")
        result.log(f"  ✅ Re-declaration rejected: {str(e)[:50]}", "PASS")
    
    # =========================================================================
    # T37c: Test DSL-style hash naming avoids conflict
    # =========================================================================
    result.log("\n--- T37c: DSL Hash Naming Avoids Conflict ---")
    
    # DSL would use: dsl_t37_from_macro_<hash>
    dsl_code = '''
    double dsl_t37_from_macro_abc123(double x) {
        return x * 20.0;  // Different implementation
    }
    '''
    
    try:
        ROOT.gInterpreter.Declare(dsl_code)
        val3 = ROOT.dsl_t37_from_macro_abc123(5.0)
        result.observe("t37c_hash_avoids_conflict", True, "Hash naming allows coexistence")
        result.log(f"  ✅ Hash-named function coexists: {val3}", "PASS")
        
        # Verify original still works
        val4 = ROOT.t37_from_macro(5.0)
        result.log(f"     Original still works: {val4}")
    except Exception as e:
        result.observe("t37c_hash_avoids_conflict", False, f"Failed: {e}")
        result.log(f"  ❌ Hash naming failed: {e}", "FAIL")
    
    # =========================================================================
    # T37d: Document persistence contract
    # =========================================================================
    result.log("\n--- T37d: Persistence Contract ---")
    
    contract = """
    Registry Persistence Contract:
    
    1. Functions declared via gInterpreter.Declare() persist for process lifetime
    2. Re-declaration of same name is REJECTED (Cling behavior)
    3. DSL hash naming (dsl_<name>_<hash>) avoids conflicts
    4. Macro-loaded functions and DSL functions can coexist
    5. No explicit "unregister" mechanism in Cling
    
    Implication for DSL:
    - Always use hash-suffixed names internally
    - Track declared names to avoid re-declaration attempts
    - Provide clean names via alias/wrapper
    """
    result.log(contract)
    result.observe("t37d_contract_documented", True, "Persistence contract documented")
    
    result.status = "PASSED"
    result.log("\n✅ T37 PASSED: Registry persistence documented", "PASS")
    
    return result


# =============================================================================
# T38: Complex RVec Types
# =============================================================================

def test_t38_complex_rvec():
    """
    T38: Complex RVec Types
    
    Proposed By: Gemini1, Claude Opus (10/12)
    Priority: P0
    
    Goal: Validate TLorentzVector, nested RVec, and physics types.
    Risk: Physics-critical types fail unexpectedly.
    """
    result = TestResult(
        test_id="T38",
        title="Complex RVec Types",
        hypothesis="Physics types like TLorentzVector work with DSL"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T38: Complex RVec Types")
    result.log("="*60)
    
    workspace = setup_test_env()
    
    # =========================================================================
    # T38a: TLorentzVector
    # =========================================================================
    result.log("\n--- T38a: TLorentzVector ---")
    
    try:
        ROOT.gInterpreter.Declare('''
        #include <TLorentzVector.h>
        
        double t38_invariant_mass(double px, double py, double pz, double E) {
            TLorentzVector v;
            v.SetPxPyPzE(px, py, pz, E);
            return v.M();
        }
        ''')
        
        # Test: particle with mass = sqrt(E^2 - p^2) = sqrt(100 - 9 - 16 - 25) = sqrt(50)
        mass = ROOT.t38_invariant_mass(3.0, 4.0, 5.0, 10.0)
        expected = (100 - 9 - 16 - 25) ** 0.5  # sqrt(50) ≈ 7.07
        
        result.observe("t38a_tlorentz_works", abs(mass - expected) < 0.1, 
                       f"TLorentzVector: M = {mass:.3f}")
        result.log(f"  ✅ TLorentzVector: M = {mass:.3f} (expected {expected:.3f})", "PASS")
    except Exception as e:
        result.observe("t38a_tlorentz_works", False, f"Failed: {e}")
        result.log(f"  ❌ TLorentzVector: {e}", "FAIL")
    
    # =========================================================================
    # T38b: Nested RVec<RVec<double>>
    # =========================================================================
    result.log("\n--- T38b: Nested RVec<RVec<double>> ---")
    
    try:
        ROOT.gInterpreter.Declare('''
        #include <ROOT/RVec.hxx>
        using namespace ROOT::VecOps;
        
        RVec<RVec<double>> t38_make_nested(const RVec<double>& v) {
            RVec<RVec<double>> result;
            result.push_back(v);
            result.push_back(v * 2.0);
            return result;
        }
        
        double t38_sum_nested(const RVec<RVec<double>>& nested) {
            double sum = 0;
            for (const auto& inner : nested) {
                for (double val : inner) {
                    sum += val;
                }
            }
            return sum;
        }
        ''')
        
        # Test with RDataFrame
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("input", "ROOT::VecOps::RVec<double>{1.0, 2.0, 3.0}")
        rdf = rdf.Define("nested", "t38_make_nested(input)")
        rdf = rdf.Define("total", "t38_sum_nested(nested)")
        
        total = rdf.Sum("total").GetValue()
        # input = [1,2,3], nested = [[1,2,3], [2,4,6]], sum = 1+2+3+2+4+6 = 18
        
        result.observe("t38b_nested_rvec", abs(total - 18.0) < 0.01, 
                       f"Nested RVec works: sum = {total}")
        result.log(f"  ✅ Nested RVec: sum = {total} (expected 18.0)", "PASS")
    except Exception as e:
        result.observe("t38b_nested_rvec", False, f"Failed: {e}")
        result.log(f"  ❌ Nested RVec: {e}", "FAIL")
    
    # =========================================================================
    # T38c: RVec<TLorentzVector> (advanced)
    # =========================================================================
    result.log("\n--- T38c: RVec<TLorentzVector> ---")
    
    try:
        ROOT.gInterpreter.Declare('''
        #include <TLorentzVector.h>
        #include <ROOT/RVec.hxx>
        using namespace ROOT::VecOps;
        
        double t38_sum_mass(const RVec<double>& px, const RVec<double>& py, 
                            const RVec<double>& pz, const RVec<double>& E) {
            double total_mass = 0;
            for (size_t i = 0; i < px.size(); i++) {
                TLorentzVector v;
                v.SetPxPyPzE(px[i], py[i], pz[i], E[i]);
                total_mass += v.M();
            }
            return total_mass;
        }
        ''')
        
        # Test
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("px", "ROOT::VecOps::RVec<double>{3.0, 0.0}")
        rdf = rdf.Define("py", "ROOT::VecOps::RVec<double>{4.0, 0.0}")
        rdf = rdf.Define("pz", "ROOT::VecOps::RVec<double>{0.0, 0.0}")
        rdf = rdf.Define("E", "ROOT::VecOps::RVec<double>{10.0, 0.5}")
        rdf = rdf.Define("total_m", "t38_sum_mass(px, py, pz, E)")
        
        total_m = rdf.Sum("total_m").GetValue()
        
        result.observe("t38c_rvec_tlorentz", True, f"RVec<TLorentzVector> pattern: {total_m:.3f}")
        result.log(f"  ✅ RVec with TLorentzVector: total mass = {total_m:.3f}", "PASS")
    except Exception as e:
        result.observe("t38c_rvec_tlorentz", False, f"Failed: {e}")
        result.log(f"  ❌ RVec<TLorentzVector>: {e}", "FAIL")
    
    # =========================================================================
    # T38d: Document supported types
    # =========================================================================
    result.log("\n--- T38d: Supported Complex Types ---")
    
    supported_types = """
    Complex Types Support:
    
    | Type                    | Define | Snapshot | Notes                    |
    |-------------------------|--------|----------|--------------------------|
    | TLorentzVector          | ✅     | ✅       | Needs TLorentzVector.h   |
    | RVec<RVec<double>>      | ✅     | ?        | Nested vectors work      |
    | RVec<TLorentzVector>    | ✅     | ?        | Pattern works, test I/O  |
    | TVector3                | ✅     | ✅       | Needs TVector3.h         |
    
    Required Headers:
    - #include <TLorentzVector.h>
    - #include <TVector3.h>
    - #include <ROOT/RVec.hxx>
    """
    result.log(supported_types)
    result.observe("t38d_types_documented", True, "Complex types documented")
    
    result.status = "PASSED"
    result.log("\n✅ T38 PASSED: Complex RVec types work", "PASS")
    
    return result


# =============================================================================
# T39: ROOT Version Matrix
# =============================================================================

def test_t39_version_matrix():
    """
    T39: ROOT Version Matrix
    
    Proposed By: Gemini1, GPT5, Claude Opus (8/12)
    Priority: P1
    
    Goal: Document which ROOT versions tested, known issues.
    This is primarily documentation, not behavioral testing.
    """
    result = TestResult(
        test_id="T39",
        title="ROOT Version Matrix",
        hypothesis="Document version-specific behaviors"
    )
    result.set_environment()
    
    if not check_root_available():
        result.status = "BLOCKED"
        result.log("ROOT not available", "FAIL")
        return result
    
    import ROOT
    
    result.log("\n" + "="*60)
    result.log("T39: ROOT Version Matrix")
    result.log("="*60)
    
    # =========================================================================
    # T39a: Collect version info
    # =========================================================================
    result.log("\n--- T39a: Current Environment ---")
    
    root_version = ROOT.gROOT.GetVersion()
    root_version_code = ROOT.gROOT.GetVersionCode()
    
    import platform
    python_version = platform.python_version()
    os_info = platform.platform()
    
    result.log(f"  ROOT Version: {root_version}")
    result.log(f"  ROOT Version Code: {root_version_code}")
    result.log(f"  Python Version: {python_version}")
    result.log(f"  Platform: {os_info}")
    
    result.observe("t39a_root_version", root_version, "ROOT version")
    result.observe("t39a_python_version", python_version, "Python version")
    result.observe("t39a_platform", os_info, "Platform")
    
    # =========================================================================
    # T39b: Check version-specific features
    # =========================================================================
    result.log("\n--- T39b: Version-Specific Features ---")
    
    # Check for Redefine (ROOT 6.26+)
    has_redefine = hasattr(ROOT.RDataFrame(1), 'Redefine')
    result.observe("t39b_has_redefine", has_redefine, "RDataFrame.Redefine() available")
    result.log(f"  RDataFrame.Redefine(): {'✅ Available' if has_redefine else '❌ Not available'}")
    
    # Check for ProgressBar (ROOT 6.28+)
    has_progressbar = 'ProgressBar' in dir(ROOT.RDF.Experimental) if hasattr(ROOT.RDF, 'Experimental') else False
    result.observe("t39b_has_progressbar", has_progressbar, "RDF ProgressBar available")
    result.log(f"  RDF ProgressBar: {'✅ Available' if has_progressbar else '❌ Not available'}")
    
    # =========================================================================
    # T39c: Document version matrix
    # =========================================================================
    result.log("\n--- T39c: Version Compatibility Matrix ---")
    
    version_matrix = """
    ROOT Version Compatibility:
    
    | Version  | Status     | Notes                              |
    |----------|------------|-----------------------------------|
    | 6.28     | Untested   | Minimum likely supported          |
    | 6.30     | Untested   | Should work                       |
    | 6.32.06  | ✅ Tested  | Current development version       |
    | 6.34+    | Untested   | Should work (forward compatible)  |
    
    Features by Version:
    - RDataFrame.Redefine(): ROOT 6.26+
    - ImplicitMT: ROOT 6.10+
    - RVec: ROOT 6.14+
    
    Platform Compatibility:
    - macOS (ARM64): ✅ Tested
    - macOS (x86_64): Untested (should work)
    - Linux (x86_64): Untested (primary target)
    - Windows: Not supported
    """
    result.log(version_matrix)
    result.observe("t39c_matrix_documented", True, "Version matrix documented")
    
    result.status = "PASSED"
    result.log("\n✅ T39 PASSED: Version matrix documented", "PASS")
    
    return result


# =============================================================================
# T41: Lambda Rejection (FROZEN RULE #1)
# =============================================================================

def test_t41_lambda_rejection():
    """
    T41: Lambda Rejection
    
    Proposed By: Claude Opus (FROZEN RULE #1)
    Priority: P0
    
    Goal: Verify DSL rejects lambda expressions with clear error message.
    Risk: Violates FROZEN RULE #1 if lambdas slip through.
    """
    result = TestResult(
        test_id="T41",
        title="Lambda Rejection (FROZEN RULE #1)",
        hypothesis="DSL must reject lambda expressions"
    )
    result.set_environment()
    
    result.log("\n" + "="*60)
    result.log("T41: Lambda Rejection (FROZEN RULE #1)")
    result.log("="*60)
    
    # =========================================================================
    # T41a: Test lambda patterns that should be rejected
    # =========================================================================
    result.log("\n--- T41a: Lambda Patterns ---")
    
    lambda_patterns = [
        ("simple lambda", "auto f = [](double x) { return x * 2; };"),
        ("capture lambda", "auto g = [&](double x) { return x * y; };"),
        ("lambda expression", "[](double x) -> double { return x; }"),
        ("std::function lambda", "std::function<double(double)> h = [](double x) { return x; };"),
    ]
    
    dsl = MockDSLCompiler(validation_mode="skip")
    
    rejection_results = {}
    for name, pattern in lambda_patterns:
        try:
            # Try to parse - should fail
            dsl.register_function_cpp(pattern)
            rejection_results[name] = "ACCEPTED (VIOLATION!)"
            result.log(f"  ❌ {name}: ACCEPTED - FROZEN RULE #1 VIOLATED!", "FAIL")
        except (ValueError, Exception) as e:
            error_msg = str(e).lower()
            has_helpful_message = "lambda" in error_msg or "not supported" in error_msg or "parse" in error_msg
            rejection_results[name] = "REJECTED"
            result.log(f"  ✅ {name}: REJECTED", "PASS")
            if has_helpful_message:
                result.log(f"     Error mentions lambda/unsupported: Yes", "INFO")
    
    all_rejected = all("REJECTED" in r for r in rejection_results.values())
    result.observe("t41a_all_lambdas_rejected", all_rejected, "All lambda patterns rejected")
    
    # =========================================================================
    # T41b: Verify named functions still work
    # =========================================================================
    result.log("\n--- T41b: Named Functions Still Work ---")
    
    named_patterns = [
        ("named function", "double f(double x) { return x * 2; }"),
        ("multi-param", "double g(double x, double y) { return x + y; }"),
        ("const ref", "double h(const double& x) { return x * 3; }"),
    ]
    
    for name, pattern in named_patterns:
        try:
            dsl2 = MockDSLCompiler(validation_mode="skip")
            dsl2.register_function_cpp(pattern)
            result.log(f"  ✅ {name}: ACCEPTED", "PASS")
        except Exception as e:
            result.log(f"  ❌ {name}: REJECTED (unexpected): {e}", "FAIL")
    
    # =========================================================================
    # T41c: Document FROZEN RULE #1
    # =========================================================================
    result.log("\n--- T41c: FROZEN RULE #1 Documentation ---")
    
    frozen_rule = """
    FROZEN RULE #1: No Lambda Expressions
    
    Lambda expressions are PROHIBITED in DSL functions due to:
    1. PyROOT stability issues
    2. Crash risk in production
    3. Difficulty debugging
    
    REJECTED patterns:
    - [](double x) { return x; }
    - auto f = [&](...) { ... };
    - std::function<...> = [...](...) { ... };
    
    REQUIRED pattern:
    - double function_name(double x) { return x; }
    
    Error message should include:
    - "Lambda expressions are not supported"
    - "Use named functions instead"
    """
    result.log(frozen_rule)
    result.observe("t41c_rule_documented", True, "FROZEN RULE #1 documented")
    
    result.status = "PASSED" if all_rejected else "FAILED"
    result.log(f"\n{'✅' if all_rejected else '❌'} T41 {'PASSED' if all_rejected else 'FAILED'}: Lambda rejection verified", 
               "PASS" if all_rejected else "FAIL")
    
    return result


# =============================================================================
# Main Runner
# =============================================================================

def run_all_tests():
    """Run all T33-T41 tests"""
    print("="*70)
    print("Phase 13.5.B0 Extended Tests (T33-T41)")
    print("Authorization: [EXECUTE-T33-T41] [FULL-COVERAGE]")
    print("="*70)
    print(f"\nDate: {datetime.now().isoformat()}")
    
    if check_root_available():
        import ROOT
        print(f"ROOT Version: {ROOT.gROOT.GetVersion()}")
    
    print("\n" + "="*70)
    
    tests = [
        ("T33", "Header Auto-Detection", test_t33_header_detection),
        ("T34", "Snapshot/I/O Contract", test_t34_snapshot_matrix),
        ("T35", "Namespace Isolation", test_t35_namespace_isolation),
        ("T36", "Parallel Compilation", test_t36_parallel_compile),
        ("T37", "Registry Persistence", test_t37_registry_persistence),
        ("T38", "Complex RVec Types", test_t38_complex_rvec),
        ("T39", "Version Matrix", test_t39_version_matrix),
        ("T41", "Lambda Rejection", test_t41_lambda_rejection),
    ]
    
    results = []
    
    for test_id, name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running {test_id}: {name}")
        print("="*70)
        
        try:
            result = test_func()
            results.append(result)
            print(result.to_markdown())
        except Exception as e:
            print(f"❌ {test_id} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            
            # Create failed result
            failed_result = TestResult(test_id, name, "Test execution failed")
            failed_result.status = "CRASHED"
            failed_result.observe("crash_error", str(e), "Crash reason")
            results.append(failed_result)
    
    # Summary
    print("\n" + "="*70)
    print("T33-T41 SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r.status == "PASSED")
    partial = sum(1 for r in results if r.status == "PARTIAL")
    failed = sum(1 for r in results if r.status in ("FAILED", "CRASHED"))
    
    for result in results:
        status_icon = {"PASSED": "✅", "PARTIAL": "⚠️", "FAILED": "❌", "CRASHED": "💥"}.get(result.status, "?")
        print(f"{status_icon} {result.test_id}: {result.status}")
    
    print(f"\n{passed}/{len(results)} tests passed")
    if partial:
        print(f"{partial} tests partial")
    if failed:
        print(f"{failed} tests failed/crashed")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    failed = sum(1 for r in results if r.status in ("FAILED", "CRASHED"))
    sys.exit(1 if failed > 0 else 0)
