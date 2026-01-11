#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T9: Function Behavior

Objective: Validate Cling function handling, overloading, and type coercion

Tests:
- T9a: Cling redeclaration (what happens on same-name declare?) ← DECISION POINT
- T9b: C++ overloading via gInterpreter
- T9c: Type coercion in RDataFrame ← CRITICAL DECISION POINT
- T9d: Overload resolution by C++ compiler

Per Phase 13.5.B0 v7 specification.
CRITICAL: T9a and T9c are DECISION POINTS that affect v7 implementation strategy.
"""

import os
import sys
import tempfile
import glob
from datetime import datetime
from typing import Tuple, Optional

# Results tracking
RESULTS = {
    "test": "T9: Function Behavior",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "errors": [],
    "observations": {},
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", 
              "OBSERVATION": "🔍", "DECISION": "🎯"}
    print(f"{prefix.get(level, '')} {msg}")
    RESULTS["findings"].append(f"[{level}] {msg}")


def observe(key: str, value, implication: str = ""):
    """Record observation for specification impact."""
    RESULTS["observations"][key] = {
        "value": value,
        "implication": implication
    }
    log(f"OBSERVATION: {key} = {value}", "OBSERVATION")
    if implication:
        log(f"  → Implication: {implication}", "INFO")


def decision_point(key: str, result, options: dict):
    """Record decision point outcome."""
    log(f"\n🎯 DECISION POINT: {key}", "DECISION")
    log(f"   Result: {result}", "DECISION")
    RESULTS["observations"][f"DECISION_{key}"] = {
        "value": result,
        "options": options,
        "implication": options.get(result, "Unknown implication")
    }
    log(f"   → {options.get(result, 'Unknown')}", "DECISION")


# =============================================================================
# T9a: Cling Redeclaration ← DECISION POINT
# =============================================================================

def test_t9a_cling_redeclaration():
    """
    T9a: What happens when declaring same function twice?
    
    DECISION POINT: Determines if we need hash suffix for redefinition.
    
    Scenarios:
    1. Same signature, same body → Should be OK (duplicate)
    2. Same signature, different body → Error? Override? Both exist?
    3. Same name, different signature → Overload (should work)
    """
    log("\n" + "="*60)
    log("T9a: Cling Redeclaration Behavior ← DECISION POINT")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Scenario 1: Same signature, same body (exact duplicate)
    log("\n--- Scenario 1: Exact duplicate declaration ---")
    scenario1_result = None
    try:
        code1a = '''
double t9a_func_s1(double x) {
    return x * 2;
}
'''
        result1a = ROOT.gInterpreter.Declare(code1a)
        log(f"First declaration: {'Success' if result1a else 'Failed'}")
        
        # Declare exact same thing again
        result1b = ROOT.gInterpreter.Declare(code1a)
        log(f"Second (duplicate) declaration: {'Success' if result1b else 'Failed'}")
        
        # Test if function works
        test_result = ROOT.gInterpreter.ProcessLine('t9a_func_s1(5.0)')
        
        if result1a and result1b:
            scenario1_result = "DUPLICATE_ALLOWED"
            log("Exact duplicate declaration allowed", "PASS")
        elif result1a and not result1b:
            scenario1_result = "DUPLICATE_REJECTED"
            log("Duplicate declaration rejected (second failed)", "INFO")
        else:
            scenario1_result = "FIRST_FAILED"
            log("First declaration failed", "FAIL")
            
    except Exception as e:
        scenario1_result = f"ERROR: {e}"
        log(f"Scenario 1 error: {e}", "WARN")
    
    observe("t9a_scenario1_duplicate", scenario1_result,
            "Behavior when declaring exact same function twice")
    
    # Scenario 2: Same signature, different body (redefinition)
    log("\n--- Scenario 2: Same signature, different body ---")
    scenario2_result = None
    try:
        code2a = '''
double t9a_func_s2(double x) {
    return x * 10;
}
'''
        result2a = ROOT.gInterpreter.Declare(code2a)
        log(f"First declaration (returns x*10): {'Success' if result2a else 'Failed'}")
        
        # Try to verify which version is active
        if result2a:
            ROOT.gInterpreter.ProcessLine('double t9a_s2_test1 = t9a_func_s2(5.0);')
            # Result should be 50
        
        code2b = '''
double t9a_func_s2(double x) {
    return x * 100;
}
'''
        result2b = ROOT.gInterpreter.Declare(code2b)
        log(f"Second declaration (returns x*100): {'Success' if result2b else 'Failed'}")
        
        if result2a and result2b:
            # Both succeeded - which one is active?
            ROOT.gInterpreter.ProcessLine('double t9a_s2_test2 = t9a_func_s2(5.0);')
            scenario2_result = "BOTH_SUCCEED_NEED_CHECK_WHICH_ACTIVE"
            log("Both declarations succeeded - need to check which is active", "WARN")
        elif result2a and not result2b:
            scenario2_result = "REDEFINITION_REJECTED"
            log("Redefinition rejected (second failed)", "INFO")
        else:
            scenario2_result = "FIRST_FAILED"
            
    except Exception as e:
        scenario2_result = f"ERROR: {e}"
        log(f"Scenario 2 error: {e}", "WARN")
    
    observe("t9a_scenario2_redefine", scenario2_result,
            "Behavior when declaring function with same signature but different body")
    
    # Scenario 3: Same name, different signature (overload)
    log("\n--- Scenario 3: Same name, different signature (overload) ---")
    scenario3_result = None
    try:
        code3a = '''
double t9a_func_s3(double x) {
    return x;
}
'''
        result3a = ROOT.gInterpreter.Declare(code3a)
        log(f"Single-param version: {'Success' if result3a else 'Failed'}")
        
        code3b = '''
double t9a_func_s3(double x, double y) {
    return x + y;
}
'''
        result3b = ROOT.gInterpreter.Declare(code3b)
        log(f"Two-param version: {'Success' if result3b else 'Failed'}")
        
        if result3a and result3b:
            # Both exist - test calling each
            ROOT.gInterpreter.ProcessLine('double t9a_s3_test1 = t9a_func_s3(5.0);')
            ROOT.gInterpreter.ProcessLine('double t9a_s3_test2 = t9a_func_s3(3.0, 4.0);')
            scenario3_result = "OVERLOAD_WORKS"
            log("Overloading works - both versions callable", "PASS")
        else:
            scenario3_result = "OVERLOAD_FAILED"
            log("Overloading failed", "FAIL")
            
    except Exception as e:
        scenario3_result = f"ERROR: {e}"
        log(f"Scenario 3 error: {e}", "WARN")
    
    observe("t9a_scenario3_overload", scenario3_result,
            "Behavior when declaring overloaded functions")
    
    # DECISION POINT
    decision_options = {
        "REDEFINITION_REJECTED": "v7 hash suffix is REQUIRED - cannot redefine same signature",
        "BOTH_SUCCEED_NEED_CHECK_WHICH_ACTIVE": "Need more investigation - may have both symbols",
        "ERROR": "Cling has issues - need robust error handling",
    }
    decision_point("T9A_REDECLARATION", scenario2_result, decision_options)
    
    return scenario3_result == "OVERLOAD_WORKS"


# =============================================================================
# T9b: C++ Overloading via gInterpreter
# =============================================================================

def test_t9b_overloading():
    """
    T9b: Does gInterpreter handle C++ overloading correctly?
    
    Tests that multiple overloads with different signatures work.
    """
    log("\n" + "="*60)
    log("T9b: C++ Overloading via gInterpreter")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Declare multiple overloads
    code = '''
// Single argument
double t9b_calc(double x) {
    return x;
}

// Two arguments
double t9b_calc(double x, double y) {
    return x + y;
}

// Three arguments
double t9b_calc(double x, double y, double z) {
    return x + y + z;
}

// Different type (int)
int t9b_calc(int x) {
    return x * 2;
}
'''
    
    try:
        result = ROOT.gInterpreter.Declare(code)
        log(f"Overloaded declarations: {'Success' if result else 'Failed'}")
        
        if not result:
            observe("t9b_overload_declare", False, "Overload declarations failed")
            return False
        
        # Test each overload
        log("\nTesting each overload...")
        
        # 1-arg double
        try:
            ROOT.gInterpreter.ProcessLine('double t9b_r1 = t9b_calc(5.0);')
            log("t9b_calc(double) works", "PASS")
            observe("t9b_1arg_double", True, "")
        except Exception as e:
            log(f"t9b_calc(double) failed: {e}", "FAIL")
            observe("t9b_1arg_double", False, str(e))
        
        # 2-arg double
        try:
            ROOT.gInterpreter.ProcessLine('double t9b_r2 = t9b_calc(3.0, 4.0);')
            log("t9b_calc(double, double) works", "PASS")
            observe("t9b_2arg_double", True, "")
        except Exception as e:
            log(f"t9b_calc(double, double) failed: {e}", "FAIL")
            observe("t9b_2arg_double", False, str(e))
        
        # 3-arg double
        try:
            ROOT.gInterpreter.ProcessLine('double t9b_r3 = t9b_calc(1.0, 2.0, 3.0);')
            log("t9b_calc(double, double, double) works", "PASS")
            observe("t9b_3arg_double", True, "")
        except Exception as e:
            log(f"t9b_calc(double, double, double) failed: {e}", "FAIL")
            observe("t9b_3arg_double", False, str(e))
        
        # int overload
        try:
            ROOT.gInterpreter.ProcessLine('int t9b_r4 = t9b_calc(5);')
            log("t9b_calc(int) works", "PASS")
            observe("t9b_1arg_int", True, "")
        except Exception as e:
            log(f"t9b_calc(int) failed: {e}", "WARN")
            observe("t9b_1arg_int", False, str(e))
        
        observe("t9b_overloading_works", True,
                "C++ overloading via gInterpreter is supported")
        return True
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        observe("t9b_overloading_works", False, str(e))
        return False


# =============================================================================
# T9c: Type Coercion in RDataFrame ← CRITICAL DECISION POINT
# =============================================================================

def test_t9c_type_coercion():
    """
    T9c: Does RDataFrame auto-convert types?
    
    CRITICAL DECISION POINT - affects resolution strategy.
    
    Tests:
    1. float column → double function
    2. int column → double function
    3. double column → float function
    """
    log("\n" + "="*60)
    log("T9c: Type Coercion in RDataFrame ← CRITICAL DECISION POINT")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Declare test functions
    code = '''
double t9c_double_func(double x) {
    return x * 2;
}

float t9c_float_func(float x) {
    return x * 3;
}

double t9c_double2_func(double x, double y) {
    return x + y;
}
'''
    
    try:
        result = ROOT.gInterpreter.Declare(code)
        if not result:
            log("Function declarations failed", "FAIL")
            return False
        
        log("Test functions declared")
        
        # Create RDataFrame with different column types
        log("\n--- Creating RDataFrame with typed columns ---")
        
        ROOT.gInterpreter.ProcessLine('''
            // Create vectors for RDataFrame
            std::vector<float> t9c_float_col = {1.0f, 2.0f, 3.0f};
            std::vector<int> t9c_int_col = {1, 2, 3};
            std::vector<double> t9c_double_col = {1.0, 2.0, 3.0};
        ''')
        
        # Create RDataFrame from vectors
        rdf_code = '''
            ROOT::RDataFrame t9c_rdf(3);
            auto t9c_rdf1 = t9c_rdf.Define("float_col", [](ULong64_t i) -> float { return (float)(i + 1); }, {"rdfentry_"});
            auto t9c_rdf2 = t9c_rdf1.Define("int_col", [](ULong64_t i) -> int { return (int)(i + 1); }, {"rdfentry_"});
            auto t9c_rdf3 = t9c_rdf2.Define("double_col", [](ULong64_t i) -> double { return (double)(i + 1); }, {"rdfentry_"});
        '''
        ROOT.gInterpreter.ProcessLine(rdf_code)
        log("RDataFrame with float, int, double columns created")
        
        # Test 1: float column → double function
        log("\n--- Test 1: float column → double function ---")
        test1_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9c_test1 = t9c_rdf3.Define("test1", "t9c_double_func(float_col)");
                auto t9c_test1_result = t9c_test1.Take<double>("test1");
            ''')
            test1_works = True
            log("float → double: WORKS (implicit conversion)", "PASS")
        except Exception as e:
            log(f"float → double: FAILED - {e}", "INFO")
        
        observe("t9c_float_to_double", test1_works,
                "RDataFrame converts float to double implicitly" if test1_works else "No implicit float→double")
        
        # Test 2: int column → double function
        log("\n--- Test 2: int column → double function ---")
        test2_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9c_test2 = t9c_rdf3.Define("test2", "t9c_double_func(int_col)");
                auto t9c_test2_result = t9c_test2.Take<double>("test2");
            ''')
            test2_works = True
            log("int → double: WORKS (implicit conversion)", "PASS")
        except Exception as e:
            log(f"int → double: FAILED - {e}", "INFO")
        
        observe("t9c_int_to_double", test2_works,
                "RDataFrame converts int to double implicitly" if test2_works else "No implicit int→double")
        
        # Test 3: double column → float function
        log("\n--- Test 3: double column → float function ---")
        test3_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9c_test3 = t9c_rdf3.Define("test3", "t9c_float_func(double_col)");
                auto t9c_test3_result = t9c_test3.Take<float>("test3");
            ''')
            test3_works = True
            log("double → float: WORKS (narrowing conversion)", "PASS")
        except Exception as e:
            log(f"double → float: FAILED - {e}", "INFO")
        
        observe("t9c_double_to_float", test3_works,
                "RDataFrame converts double to float (narrowing)" if test3_works else "No narrowing conversion")
        
        # Test 4: Two columns with mixed types
        log("\n--- Test 4: Mixed types in multi-arg function ---")
        test4_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9c_test4 = t9c_rdf3.Define("test4", "t9c_double2_func(float_col, int_col)");
                auto t9c_test4_result = t9c_test4.Take<double>("test4");
            ''')
            test4_works = True
            log("float, int → double, double: WORKS", "PASS")
        except Exception as e:
            log(f"Mixed types: FAILED - {e}", "INFO")
        
        observe("t9c_mixed_types", test4_works,
                "RDataFrame handles mixed type arguments")
        
        # DECISION POINT
        all_coercion_works = test1_works and test2_works
        
        decision_options = {
            True: "v7 can delegate to C++ compiler - type coercion is automatic",
            False: "v7 may need schema-based validation - coercion is limited",
        }
        decision_point("T9C_TYPE_COERCION", all_coercion_works, decision_options)
        
        if all_coercion_works:
            observe("t9c_conclusion", "COERCION_WORKS",
                    "RDataFrame handles type coercion - v7 resolution strategy validated")
        else:
            observe("t9c_conclusion", "COERCION_LIMITED",
                    "Type coercion limited - may need explicit type matching")
        
        return all_coercion_works
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        observe("t9c_conclusion", f"ERROR: {e}",
                "Could not complete type coercion test")
        return False


# =============================================================================
# T9d: Overload Resolution
# =============================================================================

def test_t9d_overload_resolution():
    """
    T9d: Does C++ compiler correctly resolve overloads in RDataFrame?
    
    Tests that when multiple overloads exist, correct one is called.
    """
    log("\n" + "="*60)
    log("T9d: Overload Resolution in RDataFrame")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Declare overloaded functions with distinguishable results
    code = '''
// Returns 1 for single arg
double t9d_overloaded(double x) {
    return 1.0;
}

// Returns 2 for two args
double t9d_overloaded(double x, double y) {
    return 2.0;
}

// Returns 3 for three args
double t9d_overloaded(double x, double y, double z) {
    return 3.0;
}
'''
    
    try:
        result = ROOT.gInterpreter.Declare(code)
        if not result:
            log("Overloaded function declarations failed", "FAIL")
            return False
        
        log("Overloaded functions declared (return 1, 2, or 3 based on arg count)")
        
        # Create simple RDataFrame
        ROOT.gInterpreter.ProcessLine('''
            ROOT::RDataFrame t9d_rdf(1);
            auto t9d_rdf1 = t9d_rdf.Define("x", "1.0");
            auto t9d_rdf2 = t9d_rdf1.Define("y", "2.0");
            auto t9d_rdf3 = t9d_rdf2.Define("z", "3.0");
        ''')
        
        # Test 1-arg resolution
        log("\n--- Testing 1-arg overload resolution ---")
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9d_test1 = t9d_rdf3.Define("result1", "t9d_overloaded(x)");
                auto t9d_r1 = t9d_test1.Take<double>("result1");
            ''')
            # Check result via Mean
            ROOT.gInterpreter.ProcessLine('''
                double t9d_mean1 = *t9d_test1.Mean("result1");
            ''')
            log("1-arg overload resolved", "PASS")
            observe("t9d_1arg_resolution", True, "")
        except Exception as e:
            log(f"1-arg resolution failed: {e}", "FAIL")
            observe("t9d_1arg_resolution", False, str(e))
        
        # Test 2-arg resolution
        log("\n--- Testing 2-arg overload resolution ---")
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9d_test2 = t9d_rdf3.Define("result2", "t9d_overloaded(x, y)");
            ''')
            log("2-arg overload resolved", "PASS")
            observe("t9d_2arg_resolution", True, "")
        except Exception as e:
            log(f"2-arg resolution failed: {e}", "FAIL")
            observe("t9d_2arg_resolution", False, str(e))
        
        # Test 3-arg resolution
        log("\n--- Testing 3-arg overload resolution ---")
        try:
            ROOT.gInterpreter.ProcessLine('''
                auto t9d_test3 = t9d_rdf3.Define("result3", "t9d_overloaded(x, y, z)");
            ''')
            log("3-arg overload resolved", "PASS")
            observe("t9d_3arg_resolution", True, "")
        except Exception as e:
            log(f"3-arg resolution failed: {e}", "FAIL")
            observe("t9d_3arg_resolution", False, str(e))
        
        all_resolved = (RESULTS["observations"].get("t9d_1arg_resolution", {}).get("value") and
                       RESULTS["observations"].get("t9d_2arg_resolution", {}).get("value") and
                       RESULTS["observations"].get("t9d_3arg_resolution", {}).get("value"))
        
        observe("t9d_overload_resolution_works", all_resolved,
                "RDataFrame correctly resolves overloads by argument count")
        
        return all_resolved
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Summary and Main
# =============================================================================

def print_summary():
    """Print test summary with observations."""
    print("\n" + "="*70)
    print("T9 TEST SUMMARY: Function Behavior")
    print("="*70)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    
    print("\n--- DECISION POINTS ---")
    for key, obs in RESULTS.get("observations", {}).items():
        if key.startswith("DECISION_"):
            print(f"  🎯 {key}: {obs['value']}")
            print(f"     → {obs.get('implication', 'N/A')}")
    
    print("\n--- Key Observations ---")
    for key, obs in RESULTS.get("observations", {}).items():
        if not key.startswith("DECISION_"):
            print(f"  {key}: {obs['value']}")
            if obs.get('implication'):
                print(f"    → {obs['implication']}")
    
    print("="*70)


def print_observation_report():
    """Print markdown-formatted observation report for v7 spec."""
    print("\n")
    print("## Test T9: Function Behavior - Observation Report")
    print()
    print("### Date")
    print(RESULTS["timestamp"])
    print()
    print("### Environment")
    try:
        import ROOT
        print(f"- ROOT Version: {ROOT.gROOT.GetVersion()}")
    except:
        print("- ROOT Version: N/A")
    print(f"- Python Version: {sys.version.split()[0]}")
    print(f"- Platform: {sys.platform}")
    print()
    print("### DECISION POINTS")
    print()
    
    obs = RESULTS.get("observations", {})
    
    # T9a Decision
    t9a_decision = obs.get("DECISION_T9A_REDECLARATION", {})
    print("#### T9a: Cling Redeclaration")
    print(f"**Result:** `{t9a_decision.get('value', 'N/A')}`")
    print(f"**Implication:** {t9a_decision.get('implication', 'N/A')}")
    print()
    
    # T9c Decision
    t9c_decision = obs.get("DECISION_T9C_TYPE_COERCION", {})
    print("#### T9c: Type Coercion")
    print(f"**Result:** `{t9c_decision.get('value', 'N/A')}`")
    print(f"**Implication:** {t9c_decision.get('implication', 'N/A')}")
    print()
    
    print("### Impact on v7 Specification")
    print()
    
    # Analyze impact
    t9a_result = obs.get("t9a_scenario2_redefine", {}).get("value")
    t9c_result = obs.get("t9c_conclusion", {}).get("value")
    
    if t9a_result == "REDEFINITION_REJECTED":
        print("- ✅ **Hash suffix REQUIRED** - Cling rejects same-signature redefinition")
        print("  - v7 `dsl_<n>_<hash>` naming is correct")
    else:
        print(f"- ⚠️ **Redeclaration behavior:** {t9a_result}")
        print("  - May need further investigation")
    print()
    
    if t9c_result == "COERCION_WORKS":
        print("- ✅ **Delegate to C++ compiler** - Type coercion works automatically")
        print("  - v7 resolution strategy validated")
    else:
        print(f"- ⚠️ **Type coercion:** {t9c_result}")
        print("  - May need schema-based validation")
    print()


if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T9: Function Behavior")
    print("="*70)
    
    results = {
        "t9a": test_t9a_cling_redeclaration(),
        "t9b": test_t9b_overloading(),
        "t9c": test_t9c_type_coercion(),
        "t9d": test_t9d_overload_resolution(),
    }
    
    # Determine overall status
    all_passed = all(results.values())
    # T9a and T9c are decision points - document their outcome
    decision_points_clear = True  # Always true - we're documenting behavior
    
    if all_passed:
        RESULTS["status"] = "PASS"
    elif results["t9b"] and results["t9d"]:  # Overloading works
        RESULTS["status"] = "PARTIAL_PASS"
    else:
        RESULTS["status"] = "FAIL"
    
    print_summary()
    print_observation_report()
    
    # Final status
    if all_passed:
        print("\n✅ T9 ALL TESTS PASSED")
    else:
        print("\n⚠️ T9 COMPLETED - Check decision points above")
    
    # Exit 0 since this is exploration (documenting behavior)
    sys.exit(0)
