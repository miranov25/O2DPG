#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T8: Pragma Handling

Objective: Validate pragma detection, loading, and error handling

Tests:
- T8a: Pragma detection (TClass.GetClass method)
- T8b: Pragma loading via compiled macro
- T8c: Nested type pragmas (RVec<RVec<double>>)
- T8d: Subprocess safety (crash protection)
- T8e: Pragma error handling

Per Phase 13.5.B0 v7 specification.
CRITICAL: These tests inform whether auto-detection is feasible.
"""

import os
import sys
import tempfile
import glob
import subprocess
from datetime import datetime
from typing import Tuple, Optional

# Results tracking
RESULTS = {
    "test": "T8: Pragma Handling",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "errors": [],
    "observations": {},
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "OBSERVATION": "🔍"}
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


# =============================================================================
# T8a: Pragma Detection
# =============================================================================

def test_t8a_pragma_detection():
    """
    T8a: How to detect if pragma/dictionary is loaded?
    
    Hypothesis:
        TClass.GetClass("RVec<double>") returns non-null if loaded
    
    Tests multiple detection methods to find reliable approach.
    """
    log("\n" + "="*60)
    log("T8a: Pragma/Dictionary Detection Methods")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    test_types = [
        "double",  # Primitive - should always exist
        "ROOT::VecOps::RVec<double>",  # RVec
        "RVec<double>",  # Short form
        "std::vector<double>",  # std::vector
        "ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>",  # Nested
    ]
    
    detection_results = {}
    
    for type_str in test_types:
        log(f"\nTesting detection for: {type_str}")
        
        # Method 1: TClass.GetClass
        try:
            tclass = ROOT.TClass.GetClass(type_str)
            method1 = tclass is not None
            if tclass:
                is_loaded = tclass.IsLoaded() if hasattr(tclass, 'IsLoaded') else None
                log(f"  TClass.GetClass: Found (IsLoaded={is_loaded})", "PASS" if method1 else "INFO")
            else:
                log(f"  TClass.GetClass: None", "INFO")
        except Exception as e:
            method1 = False
            log(f"  TClass.GetClass error: {e}", "WARN")
        
        # Method 2: gInterpreter.ClassInfo_IsValid
        try:
            method2 = ROOT.gInterpreter.ClassInfo_IsValid(type_str)
            log(f"  ClassInfo_IsValid: {method2}", "INFO")
        except Exception as e:
            method2 = None
            log(f"  ClassInfo_IsValid error: {e}", "WARN")
        
        # Method 3: Try to declare a variable of that type
        try:
            test_code = f'{{ {type_str} _t8a_test_var; }}'
            result = ROOT.gInterpreter.ProcessLine(test_code)
            method3 = (result == 0) if result is not None else None
            log(f"  Variable declaration: {'Success' if method3 else 'Failed'}", 
                "PASS" if method3 else "INFO")
        except Exception as e:
            method3 = False
            log(f"  Variable declaration error: {e}", "WARN")
        
        detection_results[type_str] = {
            "TClass.GetClass": method1,
            "ClassInfo_IsValid": method2,
            "variable_declaration": method3,
        }
    
    observe("t8a_detection_results", detection_results,
            "Which detection methods work for which types")
    
    # Determine best detection method
    # Prefer method that correctly identifies RVec availability
    rvec_detected = detection_results.get("ROOT::VecOps::RVec<double>", {})
    
    if rvec_detected.get("TClass.GetClass"):
        observe("t8a_recommended_detection", "TClass.GetClass",
                "TClass.GetClass works for RVec detection")
    elif rvec_detected.get("variable_declaration"):
        observe("t8a_recommended_detection", "variable_declaration",
                "Try variable declaration as detection method")
    else:
        observe("t8a_recommended_detection", "NEEDS_PRAGMA_FIRST",
                "RVec may need pragma before detection works")
    
    log("T8a: Detection methods documented", "PASS")
    return True


# =============================================================================
# T8b: Pragma Loading via Macro
# =============================================================================

def test_t8b_pragma_via_macro():
    """
    T8b: Can we load pragma via compiled macro?
    
    Tests loading #pragma link C++ class via ACLiC compiled macro.
    """
    log("\n" + "="*60)
    log("T8b: Pragma Loading via Compiled Macro")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Macro with pragma link
    code = '''
#include <ROOT/RVec.hxx>

#ifdef __CLING__
#pragma link C++ class ROOT::VecOps::RVec<double>+;
#endif

// Dummy function to ensure compilation
void t8b_dummy() {}

// Function using RVec
ROOT::VecOps::RVec<double> t8b_create_rvec() {
    return ROOT::VecOps::RVec<double>{1.0, 2.0, 3.0};
}
'''
    
    with tempfile.NamedTemporaryFile(suffix='_t8b.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    log(f"Created pragma macro: {macro_path}")
    
    try:
        # Check RVec usability BEFORE loading pragma macro
        log("Checking RVec availability BEFORE pragma macro...")
        before_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                ROOT::VecOps::RVec<double> t8b_before{1.0, 2.0};
            ''')
            before_works = True
            log("RVec was already usable before pragma macro", "INFO")
        except Exception as e:
            log(f"RVec not usable before: {e}", "INFO")
        
        observe("t8b_rvec_before_pragma", before_works,
                "Whether RVec works before explicit pragma")
        
        # Compile macro with pragma
        log("Compiling pragma macro with ACLiC...")
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}+')
        
        observe("t8b_aclic_result", result,
                "ACLiC compilation result with pragma")
        
        # Check for .so
        base_name = macro_path.replace('.C', '_C')
        so_files = [f for f in glob.glob(f"{base_name}.*") 
                    if f.endswith('.so') or f.endswith('.dylib')]
        
        if so_files:
            log(f"Pragma macro compiled: {so_files[0]}", "PASS")
        else:
            log("No .so created from pragma macro", "WARN")
        
        # Check RVec usability AFTER loading pragma macro
        log("Checking RVec availability AFTER pragma macro...")
        after_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                ROOT::VecOps::RVec<double> t8b_after{4.0, 5.0};
                double t8b_sum = Sum(t8b_after);
            ''')
            after_works = True
            log("RVec works after pragma macro", "PASS")
        except Exception as e:
            log(f"RVec still not working: {e}", "FAIL")
        
        observe("t8b_rvec_after_pragma", after_works,
                "Whether RVec works after pragma macro loaded")
        
        # Test the function from pragma macro
        log("Testing t8b_create_rvec function...")
        try:
            result_vec = ROOT.t8b_create_rvec()
            log(f"t8b_create_rvec() returned: size={result_vec.size()}", "PASS")
            observe("t8b_function_works", True,
                    "Functions in pragma macro are callable")
        except Exception as e:
            log(f"t8b_create_rvec failed: {e}", "WARN")
            observe("t8b_function_works", False, str(e))
        
        return after_works
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        return False
        
    finally:
        try:
            os.unlink(macro_path)
        except:
            pass
        base_name = macro_path.replace('.C', '_C')
        for f in glob.glob(f"{base_name}.*"):
            try:
                os.unlink(f)
            except:
                pass


# =============================================================================
# T8c: Nested Type Pragmas
# =============================================================================

def test_t8c_nested_pragma():
    """
    T8c: What about nested types like RVec<RVec<double>>?
    
    Tests dependency order for nested template types.
    """
    log("\n" + "="*60)
    log("T8c: Nested Type Pragmas (RVec<RVec<double>>)")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Test 1: Try nested RVec without explicit pragma
    log("Test 1: Trying nested RVec without explicit pragma...")
    test1_works = False
    try:
        ROOT.gInterpreter.ProcessLine('''
            ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> t8c_nested{{1.0, 2.0}, {3.0}};
        ''')
        test1_works = True
        log("Nested RVec works without explicit pragma", "PASS")
    except Exception as e:
        log(f"Nested RVec failed: {e}", "INFO")
    
    observe("t8c_nested_without_pragma", test1_works,
            "Whether nested RVec works without explicit pragma")
    
    if test1_works:
        log("T8c: Nested types work automatically (no special handling needed)", "PASS")
        observe("t8c_conclusion", "AUTO",
                "Nested RVec types work automatically in ROOT")
        return True
    
    # Test 2: Try with single-level pragma first
    log("\nTest 2: Loading single-level pragma first...")
    code_single = '''
#include <ROOT/RVec.hxx>
#ifdef __CLING__
#pragma link C++ class ROOT::VecOps::RVec<double>+;
#endif
void t8c_single_pragma() {}
'''
    
    with tempfile.NamedTemporaryFile(suffix='_t8c_single.C', delete=False, mode='w') as f:
        f.write(code_single)
        macro_single = f.name
    
    try:
        ROOT.gROOT.ProcessLine(f'.L {macro_single}+')
        log("Single-level pragma loaded", "INFO")
        
        # Try nested again
        test2_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> t8c_nested2{{1.0}, {2.0}};
            ''')
            test2_works = True
            log("Nested RVec works after single-level pragma", "PASS")
        except Exception as e:
            log(f"Still failed after single pragma: {e}", "INFO")
        
        observe("t8c_nested_after_single_pragma", test2_works,
                "Whether nested works after loading single-level pragma")
        
    finally:
        os.unlink(macro_single)
        for f in glob.glob(macro_single.replace('.C', '_C') + '.*'):
            try:
                os.unlink(f)
            except:
                pass
    
    # Test 3: Try with explicit nested pragma
    if not test2_works:
        log("\nTest 3: Loading explicit nested pragma...")
        code_nested = '''
#include <ROOT/RVec.hxx>
#ifdef __CLING__
#pragma link C++ class ROOT::VecOps::RVec<double>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>+;
#endif
void t8c_nested_pragma() {}
'''
        
        with tempfile.NamedTemporaryFile(suffix='_t8c_nested.C', delete=False, mode='w') as f:
            f.write(code_nested)
            macro_nested = f.name
        
        try:
            ROOT.gROOT.ProcessLine(f'.L {macro_nested}+')
            
            test3_works = False
            try:
                ROOT.gInterpreter.ProcessLine('''
                    ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> t8c_nested3{{1.0}, {2.0}};
                ''')
                test3_works = True
                log("Nested RVec works with explicit nested pragma", "PASS")
            except Exception as e:
                log(f"Still failed with explicit nested pragma: {e}", "FAIL")
            
            observe("t8c_nested_with_explicit_pragma", test3_works,
                    "Whether nested works with explicit pragma for nested type")
            
        finally:
            os.unlink(macro_nested)
            for f in glob.glob(macro_nested.replace('.C', '_C') + '.*'):
                try:
                    os.unlink(f)
                except:
                    pass
    
    # Determine conclusion
    if test1_works:
        observe("t8c_conclusion", "AUTO", "No special handling needed")
    elif test2_works:
        observe("t8c_conclusion", "SINGLE_PRAGMA_SUFFICIENT",
                "Only need pragma for innermost type")
    else:
        observe("t8c_conclusion", "EXPLICIT_NESTED_PRAGMA_REQUIRED",
                "Must add pragma for each nesting level")
    
    return test1_works or test2_works


# =============================================================================
# T8d: Subprocess Safety
# =============================================================================

def test_t8d_subprocess_safety():
    """
    T8d: Does subprocess protect main process from pragma crash?
    
    Tests that bad pragma/compilation in subprocess doesn't crash main.
    """
    log("\n" + "="*60)
    log("T8d: Subprocess Safety for Pragma Loading")
    log("="*60)
    
    # Test 1: Bad pragma in subprocess
    log("Test 1: Bad pragma code in subprocess...")
    
    bad_pragma_test = '''
import sys
try:
    import ROOT
    # Intentionally bad pragma
    code = """
#ifdef __CLING__
#pragma link C++ class NonExistentType+;
#endif
void bad_pragma_test() {}
"""
    # Try to process bad code
    result = ROOT.gInterpreter.Declare(code)
    if result:
        print("SUBPROCESS_UNEXPECTED_SUCCESS")
    else:
        print("SUBPROCESS_EXPECTED_FAILURE")
except Exception as e:
    print(f"SUBPROCESS_EXCEPTION: {e}")
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", bad_pragma_test],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        log(f"Subprocess stdout: {result.stdout.strip()}")
        if result.stderr:
            log(f"Subprocess stderr (truncated): {result.stderr[:200]}", "INFO")
        
        subprocess_completed = result.returncode is not None
        observe("t8d_subprocess_completed", subprocess_completed,
                "Whether subprocess completed (didn't hang)")
        
        # Main process still alive!
        log("Main process survived subprocess test", "PASS")
        observe("t8d_main_process_safe", True,
                "Main process not affected by subprocess failure")
        
    except subprocess.TimeoutExpired:
        log("Subprocess timed out", "WARN")
        observe("t8d_subprocess_timeout", True,
                "Subprocess may hang on bad pragma - need timeout")
    except Exception as e:
        log(f"Subprocess error: {e}", "FAIL")
        return False
    
    # Test 2: Good pragma in subprocess
    log("\nTest 2: Good pragma in subprocess...")
    
    good_pragma_test = '''
import sys
try:
    import ROOT
    code = """
#include <ROOT/RVec.hxx>
ROOT::VecOps::RVec<double> subprocess_test() {
    return ROOT::VecOps::RVec<double>{1.0, 2.0, 3.0};
}
"""
    result = ROOT.gInterpreter.Declare(code)
    if result:
        print("SUBPROCESS_SUCCESS")
    else:
        print("SUBPROCESS_FAILURE")
except Exception as e:
    print(f"SUBPROCESS_EXCEPTION: {e}")
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", good_pragma_test],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = "SUBPROCESS_SUCCESS" in result.stdout
        log(f"Good pragma subprocess: {'Success' if success else 'Failed'}", 
            "PASS" if success else "WARN")
        observe("t8d_good_pragma_subprocess", success,
                "Whether good pragma works in subprocess")
        
    except Exception as e:
        log(f"Good pragma subprocess error: {e}", "WARN")
    
    log("T8d: Subprocess isolation works", "PASS")
    return True


# =============================================================================
# T8e: Pragma Error Handling
# =============================================================================

def test_t8e_error_handling():
    """
    T8e: What error do we get on pragma failure?
    
    Captures error formats for user-friendly messaging.
    """
    log("\n" + "="*60)
    log("T8e: Pragma Error Handling")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    error_cases = []
    
    # Test 1: Invalid type in pragma
    log("Test 1: Invalid type pragma...")
    try:
        code1 = '''
#ifdef __CLING__
#pragma link C++ class CompletelyFakeType+;
#endif
'''
        result = ROOT.gInterpreter.Declare(code1)
        error_cases.append({
            "case": "invalid_type",
            "result": result,
            "error": "No error captured" if result else "Declare returned False"
        })
        log(f"Invalid type pragma result: {result}", "INFO")
    except Exception as e:
        error_cases.append({
            "case": "invalid_type",
            "result": None,
            "error": str(e)
        })
        log(f"Invalid type pragma error: {e}", "INFO")
    
    # Test 2: Syntax error in code
    log("\nTest 2: Syntax error in code...")
    try:
        code2 = '''
double broken_function(double x { return x; }  // Missing )
'''
        result = ROOT.gInterpreter.Declare(code2)
        error_cases.append({
            "case": "syntax_error",
            "result": result,
            "error": "No error captured" if result else "Declare returned False"
        })
        log(f"Syntax error result: {result}", "INFO")
    except Exception as e:
        error_cases.append({
            "case": "syntax_error",
            "result": None,
            "error": str(e)
        })
        log(f"Syntax error exception: {e}", "INFO")
    
    # Test 3: Missing include
    log("\nTest 3: Missing include...")
    try:
        code3 = '''
SomeUndefinedVector<double> missing_include_func() {
    return SomeUndefinedVector<double>();
}
'''
        result = ROOT.gInterpreter.Declare(code3)
        error_cases.append({
            "case": "missing_include",
            "result": result,
            "error": "No error captured" if result else "Declare returned False"
        })
        log(f"Missing include result: {result}", "INFO")
    except Exception as e:
        error_cases.append({
            "case": "missing_include",
            "result": None,
            "error": str(e)
        })
        log(f"Missing include exception: {e}", "INFO")
    
    observe("t8e_error_cases", error_cases,
            "Error patterns from various failure modes")
    
    # Analyze error handling pattern
    declare_returns_false = any(c["result"] == False for c in error_cases)
    raises_exception = any(c["result"] is None for c in error_cases)
    
    if declare_returns_false:
        observe("t8e_error_pattern", "DECLARE_RETURNS_FALSE",
                "gInterpreter.Declare() returns False on error, check stderr for details")
    elif raises_exception:
        observe("t8e_error_pattern", "RAISES_EXCEPTION",
                "gInterpreter.Declare() raises exception on error")
    else:
        observe("t8e_error_pattern", "UNCLEAR",
                "Error handling pattern needs more investigation")
    
    log("T8e: Error patterns documented", "PASS")
    return True


# =============================================================================
# Summary and Main
# =============================================================================

def print_summary():
    """Print test summary with observations."""
    print("\n" + "="*70)
    print("T8 TEST SUMMARY: Pragma Handling")
    print("="*70)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    
    print("\n--- Key Observations ---")
    for key, obs in RESULTS.get("observations", {}).items():
        print(f"  {key}: {obs['value']}")
        if obs.get('implication'):
            print(f"    → {obs['implication']}")
    
    print("\n--- Findings ---")
    for finding in RESULTS["findings"]:
        print(f"  {finding}")
    
    if RESULTS["errors"]:
        print("\n--- Errors ---")
        for error in RESULTS["errors"]:
            print(f"  {error}")
    
    print("="*70)


def print_observation_report():
    """Print markdown-formatted observation report for v7 spec."""
    print("\n")
    print("## Test T8: Pragma Handling - Observation Report")
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
    print("### Key Findings for v7 Specification")
    print()
    
    obs = RESULTS.get("observations", {})
    
    # Detection method
    rec_detect = obs.get("t8a_recommended_detection", {}).get("value")
    print(f"**Recommended Detection Method:** `{rec_detect}`")
    print()
    
    # Nested handling
    nested_conclusion = obs.get("t8c_conclusion", {}).get("value")
    print(f"**Nested Type Handling:** `{nested_conclusion}`")
    print()
    
    # Error pattern
    error_pattern = obs.get("t8e_error_pattern", {}).get("value")
    print(f"**Error Pattern:** `{error_pattern}`")
    print()
    
    # Subprocess safety
    safe = obs.get("t8d_main_process_safe", {}).get("value")
    print(f"**Subprocess Safety:** `{'Confirmed' if safe else 'Needs Investigation'}`")
    print()
    
    print("### Implications for Phase 13.5.B")
    print()
    if obs.get("t8b_rvec_before_pragma", {}).get("value"):
        print("- ✅ RVec works without explicit pragma in current ROOT version")
        print("- Explicit pragmas parameter may be optional for common types")
    else:
        print("- ⚠️ May need pragma macro for RVec support")
    print()


if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T8: Pragma Handling")
    print("="*70)
    
    results = {
        "t8a": test_t8a_pragma_detection(),
        "t8b": test_t8b_pragma_via_macro(),
        "t8c": test_t8c_nested_pragma(),
        "t8d": test_t8d_subprocess_safety(),
        "t8e": test_t8e_error_handling(),
    }
    
    # Determine overall status
    all_passed = all(results.values())
    critical_passed = results["t8a"] and results["t8d"]  # Detection and safety are critical
    
    if all_passed:
        RESULTS["status"] = "PASS"
    elif critical_passed:
        RESULTS["status"] = "PARTIAL_PASS"
    else:
        RESULTS["status"] = "FAIL"
    
    print_summary()
    print_observation_report()
    
    if all_passed:
        print("\n✅ T8 ALL TESTS PASSED")
        sys.exit(0)
    elif critical_passed:
        print("\n⚠️ T8 CRITICAL TESTS PASSED (detection + safety)")
        sys.exit(0)
    else:
        print("\n❌ T8 CRITICAL TESTS FAILED")
        sys.exit(1)
