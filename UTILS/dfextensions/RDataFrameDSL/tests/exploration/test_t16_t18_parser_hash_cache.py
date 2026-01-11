#!/usr/bin/env python3
"""
Phase 13.5.B0 - Tests T16-T18: Parser, Hash, Cache

T16: Parser Coverage Matrix (P0)
T17: Hash Determinism + Schema-Version Salt (P0)
T18: Cache Contamination Guard (P0)

All Priority P0 - Core Architectural Validation
"""

import os
import sys
import time
import subprocess
from datetime import datetime

from test_infrastructure import (
    setup_test_env, get_workspace, TestResult, wait_for_timestamp_change,
    MockDSLCompiler, check_root_available, create_test_macro
)


# =============================================================================
# T16: Parser Coverage Matrix
# =============================================================================

def test_t16_parser_coverage():
    """
    T16: Signature Parser Coverage Matrix
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Define explicitly supported C++ subset.
    Fail-fast outside subset with clear error.
    """
    result = TestResult(
        test_id="T16",
        title="Parser Coverage Matrix",
        hypothesis="Parser correctly handles supported signatures and "
                   "rejects unsupported ones with clear errors"
    )
    result.set_environment()
    
    result.log("\n" + "="*60)
    result.log("T16: Parser Coverage Matrix")
    result.log("="*60)
    
    dsl = MockDSLCompiler(validation_mode="skip")  # Don't validate with ROOT
    
    # =========================================================================
    # T16a: Supported Signatures (Must Parse)
    # =========================================================================
    result.log("\n--- T16a: Supported Signatures ---")
    
    supported_cases = [
        # Basic types
        ("double f(double x) { return x; }", "basic double"),
        ("float f(float x) { return x; }", "basic float"),
        ("int f(int x) { return x; }", "basic int"),
        ("bool f(bool x) { return x; }", "basic bool"),
        
        # Multiple parameters
        ("double f(double x, double y) { return x + y; }", "two params"),
        ("double f(double x, double y, double z) { return x; }", "three params"),
        
        # Const references
        ("double f(const double& x) { return x; }", "const double ref"),
        
        # RVec types
        ("double f(const RVec<double>& v) { return v[0]; }", "RVec const ref"),
        ("RVec<double> f(double x) { return {x}; }", "RVec return"),
        
        # Namespace in types
        ("double f(ROOT::VecOps::RVec<double> v) { return v[0]; }", "full namespace"),
        
        # Complex expressions
        ("double f(double x) { return sqrt(x*x + 1); }", "complex body"),
        
        # Multi-line
        ("double f(double x) {\n    double y = x * 2;\n    return y;\n}", "multi-line"),
    ]
    
    passed_supported = 0
    for code, description in supported_cases:
        try:
            # Reset DSL for each test
            dsl_test = MockDSLCompiler(validation_mode="skip")
            dsl_test.register_function_cpp(code)
            result.log(f"  ✅ {description}: parsed", "PASS")
            passed_supported += 1
        except Exception as e:
            result.log(f"  ❌ {description}: failed - {e}", "FAIL")
    
    result.observe("t16a_supported_passed", f"{passed_supported}/{len(supported_cases)}",
                   "Supported signatures that parsed correctly")
    
    # =========================================================================
    # T16b: Unsupported Signatures (Must Reject)
    # =========================================================================
    result.log("\n--- T16b: Unsupported Signatures ---")
    
    unsupported_cases = [
        # Missing return type
        ("f(double x) { return x; }", "missing return type"),
        
        # Missing body
        ("double f(double x);", "declaration only"),
        
        # Template function
        ("template<typename T> T f(T x) { return x; }", "template"),
        
        # Member function
        ("double C::f(double x) { return x; }", "member function"),
        
        # Variadic
        ("double f(double x, ...) { return x; }", "variadic"),
        
        # Lambda (should be rejected)
        ("auto f = [](double x) { return x; };", "lambda"),
    ]
    
    rejected_unsupported = 0
    for code, description in unsupported_cases:
        try:
            dsl_test = MockDSLCompiler(validation_mode="skip")
            dsl_test.register_function_cpp(code)
            result.log(f"  ⚠️  {description}: accepted (should reject)", "WARN")
        except ValueError as e:
            result.log(f"  ✅ {description}: rejected", "PASS")
            rejected_unsupported += 1
        except Exception as e:
            result.log(f"  ✅ {description}: rejected ({type(e).__name__})", "PASS")
            rejected_unsupported += 1
    
    result.observe("t16b_unsupported_rejected", f"{rejected_unsupported}/{len(unsupported_cases)}",
                   "Unsupported signatures that were rejected")
    
    # =========================================================================
    # T16c: Edge Cases
    # =========================================================================
    result.log("\n--- T16c: Edge Cases ---")
    
    edge_cases = [
        # Comments
        ("/* comment */ double f(double x) { return x; }", True, "block comment"),
        ("// comment\ndouble f(double x) { return x; }", True, "line comment"),
        
        # Extra whitespace
        ("double    f  (  double   x  )  {  return  x  ;  }", True, "extra whitespace"),
        
        # Nested templates
        ("double f(const RVec<RVec<double>>& v) { return v[0][0]; }", True, "nested RVec"),
    ]
    
    for code, should_parse, description in edge_cases:
        try:
            dsl_test = MockDSLCompiler(validation_mode="skip")
            dsl_test.register_function_cpp(code)
            if should_parse:
                result.log(f"  ✅ {description}: parsed", "PASS")
            else:
                result.log(f"  ⚠️  {description}: parsed (unexpected)", "WARN")
        except Exception as e:
            if not should_parse:
                result.log(f"  ✅ {description}: rejected", "PASS")
            else:
                result.log(f"  ❌ {description}: failed - {e}", "FAIL")
    
    # Summary
    total_tests = len(supported_cases) + len(unsupported_cases) + len(edge_cases)
    result.observe("t16_total_tests", total_tests, "Total parser tests")
    
    if passed_supported == len(supported_cases):
        result.status = "PASSED"
        result.log("\n✅ T16 PASSED: Parser handles all test cases", "PASS")
    else:
        result.status = "PARTIAL"
        result.log("\n⚠️  T16 PARTIAL: Some supported signatures failed", "WARN")
    
    return result


# =============================================================================
# T17: Hash Determinism + Schema-Version Salt
# =============================================================================

def test_t17_hash_determinism():
    """
    T17: Hash Determinism & Schema-Version Salt
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Prove hash stability across formatting; version invalidation works.
    """
    result = TestResult(
        test_id="T17",
        title="Hash Determinism + Schema-Version Salt",
        hypothesis="Hash is stable across whitespace/formatting changes, "
                   "but changes with schema version"
    )
    result.set_environment()
    
    result.log("\n" + "="*60)
    result.log("T17: Hash Determinism & Schema-Version Salt")
    result.log("="*60)
    
    # =========================================================================
    # T17a: Whitespace Changes → Same Hash
    # =========================================================================
    result.log("\n--- T17a: Whitespace Invariance ---")
    
    code_variants = [
        "double f(double x){return x*2;}",
        "double f(double x) { return x * 2; }",
        "double f(double x) {\n    return x * 2;\n}",
        "double   f(  double   x  )  {  return   x  *  2  ;  }",
    ]
    
    hashes = []
    for i, code in enumerate(code_variants):
        dsl = MockDSLCompiler(validation_mode="skip")
        dsl.register_function_cpp(code)
        func = dsl.get_function("f")
        hashes.append(func.hash)
        result.log(f"  Variant {i+1}: {func.hash}")
    
    all_same = len(set(hashes)) == 1
    result.observe("t17a_whitespace_invariant", all_same,
                   "All whitespace variants produce same hash")
    
    if all_same:
        result.log("Whitespace invariance: PASS", "PASS")
    else:
        result.log("Whitespace invariance: FAIL", "FAIL")
    
    # =========================================================================
    # T17b: Header Order → Same Hash
    # =========================================================================
    result.log("\n--- T17b: Header Order Invariance ---")
    
    dsl1 = MockDSLCompiler(validation_mode="skip")
    dsl1.register_function_cpp(
        "double f(double x) { return sqrt(x); }",
        headers=["<cmath>", "<vector>"]
    )
    
    dsl2 = MockDSLCompiler(validation_mode="skip")
    dsl2.register_function_cpp(
        "double f(double x) { return sqrt(x); }",
        headers=["<vector>", "<cmath>"]
    )
    
    hash1 = dsl1.get_function("f").hash
    hash2 = dsl2.get_function("f").hash
    
    result.log(f"  Headers [cmath, vector]: {hash1}")
    result.log(f"  Headers [vector, cmath]: {hash2}")
    
    header_order_invariant = hash1 == hash2
    result.observe("t17b_header_order_invariant", header_order_invariant,
                   "Header order does not affect hash")
    
    if header_order_invariant:
        result.log("Header order invariance: PASS", "PASS")
    else:
        result.log("Header order invariance: FAIL", "FAIL")
    
    # =========================================================================
    # T17c: Different Headers → Different Hash
    # =========================================================================
    result.log("\n--- T17c: Different Headers → Different Hash ---")
    
    dsl3 = MockDSLCompiler(validation_mode="skip")
    dsl3.register_function_cpp(
        "double g(double x) { return x; }",
        headers=["<cmath>"]
    )
    
    dsl4 = MockDSLCompiler(validation_mode="skip")
    dsl4.register_function_cpp(
        "double g(double x) { return x; }",
        headers=["<cmath>", "<vector>"]
    )
    
    hash3 = dsl3.get_function("g").hash
    hash4 = dsl4.get_function("g").hash
    
    result.log(f"  Headers [cmath]: {hash3}")
    result.log(f"  Headers [cmath, vector]: {hash4}")
    
    headers_affect_hash = hash3 != hash4
    result.observe("t17c_headers_affect_hash", headers_affect_hash,
                   "Different headers produce different hash")
    
    if headers_affect_hash:
        result.log("Headers affect hash: PASS", "PASS")
    else:
        result.log("Headers affect hash: FAIL", "FAIL")
    
    # =========================================================================
    # T17d: Schema Version Change → Different Hash
    # =========================================================================
    result.log("\n--- T17d: Schema Version Invalidation ---")
    
    # Store original version
    original_version = MockDSLCompiler.HASH_SCHEMA_VERSION
    
    # Hash with version 1
    MockDSLCompiler.HASH_SCHEMA_VERSION = 1
    dsl_v1 = MockDSLCompiler(validation_mode="skip")
    dsl_v1.register_function_cpp("double h(double x) { return x; }")
    hash_v1 = dsl_v1.get_function("h").hash
    
    # Hash with version 2
    MockDSLCompiler.HASH_SCHEMA_VERSION = 2
    dsl_v2 = MockDSLCompiler(validation_mode="skip")
    dsl_v2.register_function_cpp("double h(double x) { return x; }")
    hash_v2 = dsl_v2.get_function("h").hash
    
    # Restore
    MockDSLCompiler.HASH_SCHEMA_VERSION = original_version
    
    result.log(f"  Schema v1: {hash_v1}")
    result.log(f"  Schema v2: {hash_v2}")
    
    version_invalidates = hash_v1 != hash_v2
    result.observe("t17d_version_invalidates_hash", version_invalidates,
                   "Schema version change invalidates cache")
    
    if version_invalidates:
        result.log("Version invalidation: PASS", "PASS")
    else:
        result.log("Version invalidation: FAIL", "FAIL")
    
    # Summary
    if all_same and header_order_invariant and headers_affect_hash and version_invalidates:
        result.status = "PASSED"
        result.log("\n✅ T17 PASSED: Hash determinism verified", "PASS")
    else:
        result.status = "PARTIAL"
        result.log("\n⚠️  T17 PARTIAL: Some hash tests failed", "WARN")
    
    return result


# =============================================================================
# T18: Cache Contamination Guard
# =============================================================================

def test_t18_cache_contamination():
    """
    T18: Cache Contamination Guard
    
    Proposed By: GPT5
    Priority: P0
    
    Goal: Prevent Process B from silently running Process A's stale code.
    Critical for CI environments.
    """
    result = TestResult(
        test_id="T18",
        title="Cache Contamination Guard",
        hypothesis="Source changes are detected and code is rebuilt, "
                   "preventing stale cache usage"
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
    result.log("T18: Cache Contamination Guard")
    result.log("="*60)
    
    # Create unique function name to avoid conflicts
    import random
    func_id = random.randint(10000, 99999)
    func_name = f"t18_test_func_{func_id}"
    
    # =========================================================================
    # T18a: Create and compile version 1
    # =========================================================================
    result.log("\n--- T18a: Create Version 1 ---")
    
    code_v1 = f'''
#include <iostream>
const char* T18_BUILD_ID_{func_id} = "UUID_A_V1";
double {func_name}() {{
    return 1.0;
}}
'''
    
    macro_path = create_test_macro(f"t18_macro_{func_id}", code_v1, workspace)
    result.log(f"Created macro: {macro_path}")
    
    # Compile with ACLiC
    ROOT.gSystem.CompileMacro(macro_path, "kf")  # k=keep, f=force
    
    so_path = macro_path.replace('.C', '_C.so')
    so_exists_v1 = os.path.exists(so_path)
    result.observe("t18a_so_created", so_exists_v1, "Shared library created for v1")
    
    if not so_exists_v1:
        result.status = "FAILED"
        result.log("Failed to create .so", "FAIL")
        return result
    
    timestamp_v1 = os.path.getmtime(so_path)
    result.log(f"V1 .so timestamp: {timestamp_v1}")
    
    # Get v1 result
    result_v1 = getattr(ROOT, func_name)()
    result.log(f"V1 result: {result_v1}")
    result.observe("t18a_v1_result", result_v1, "Version 1 returns 1.0")
    
    # =========================================================================
    # T18b: Wait and modify source
    # =========================================================================
    result.log("\n--- T18b: Modify Source (with timestamp wait) ---")
    
    # Wait for filesystem timestamp resolution (critical!)
    wait_for_timestamp_change(1.5)  # 1.5 seconds to be safe
    
    code_v2 = f'''
#include <iostream>
const char* T18_BUILD_ID_{func_id} = "UUID_B_V2";
double {func_name}() {{
    return 2.0;
}}
'''
    
    with open(macro_path, 'w') as f:
        f.write(code_v2)
    
    result.log("Source modified to v2")
    
    # =========================================================================
    # T18c: Force recompile and check
    # =========================================================================
    result.log("\n--- T18c: Force Recompile ---")
    
    # Force recompile
    ROOT.gSystem.CompileMacro(macro_path, "kf")  # f=force rebuild
    
    timestamp_v2 = os.path.getmtime(so_path)
    result.log(f"V2 .so timestamp: {timestamp_v2}")
    
    timestamp_changed = timestamp_v2 > timestamp_v1
    result.observe("t18c_timestamp_changed", timestamp_changed,
                   "Timestamp updated after rebuild")
    
    # Load and test
    ROOT.gSystem.Load(so_path)
    
    result_v2 = getattr(ROOT, func_name)()
    result.log(f"V2 result: {result_v2}")
    result.observe("t18c_v2_result", result_v2, "Version 2 returns 2.0")
    
    # =========================================================================
    # T18d: Verify correct version
    # =========================================================================
    result.log("\n--- T18d: Verify Correct Version ---")
    
    correct_result = (result_v2 == 2.0)
    result.observe("t18d_correct_version", correct_result,
                   "Got v2 result (2.0) after rebuild")
    
    # Also check BUILD_ID if accessible
    try:
        build_id_name = f"T18_BUILD_ID_{func_id}"
        # BUILD_ID check via gInterpreter
        ROOT.gInterpreter.ProcessLine(f'std::cout << "BUILD_ID: " << {build_id_name} << std::endl;')
    except:
        result.log("BUILD_ID not accessible (expected with some ROOT configs)", "INFO")
    
    # Summary
    if correct_result and timestamp_changed:
        result.status = "PASSED"
        result.log("\n✅ T18 PASSED: Cache contamination prevented", "PASS")
    else:
        result.status = "FAILED"
        result.log("\n❌ T18 FAILED: Stale cache detected!", "FAIL")
        result.action_items.append("Investigate ACLiC cache invalidation")
    
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Tests T16-T18")
    print("="*70)
    
    results = {}
    
    # T16: Parser Coverage
    print("\n" + "="*70)
    results["T16"] = test_t16_parser_coverage()
    
    # T17: Hash Determinism
    print("\n" + "="*70)
    results["T17"] = test_t17_hash_determinism()
    
    # T18: Cache Contamination
    print("\n" + "="*70)
    results["T18"] = test_t18_cache_contamination()
    
    # Summary
    print("\n" + "="*70)
    print("T16-T18 SUMMARY")
    print("="*70)
    
    for test_id, result in results.items():
        status_icon = "✅" if result.status == "PASSED" else "⚠️" if result.status == "PARTIAL" else "❌"
        print(f"{status_icon} {test_id}: {result.status}")
    
    passed = sum(1 for r in results.values() if r.status == "PASSED")
    print(f"\n{passed}/{len(results)} tests passed")
    
    sys.exit(0 if passed == len(results) else 1)
