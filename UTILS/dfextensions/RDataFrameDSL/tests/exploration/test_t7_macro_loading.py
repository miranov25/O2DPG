#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T7: Macro Loading

Objective: Validate macro loading, namespace access, and persistence

Tests:
- T7a: Basic macro loading (.L file.C)
- T7b: ACLiC compilation (.L file.C+)
- T7c: Namespace access from Python
- T7d: .so persistence across sessions

Per Phase 13.5.B0 v7 specification.
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
    "test": "T7: Macro Loading",
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
# T7a: Basic Macro Loading
# =============================================================================

def test_t7a_basic_loading():
    """
    T7a: Can we load a simple .C macro?
    
    Hypothesis:
        ROOT.gInterpreter.ProcessLine('.L file.C') loads macro
        Functions become available via ROOT.funcname()
    
    Validates:
        - Basic .L loading works
        - Functions callable from Python
    """
    log("\n" + "="*60)
    log("T7a: Basic Macro Loading (.L file.C)")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available - cannot run this test", "FAIL")
        return False
    
    # Simple macro without namespace
    code = '''
// T7a Test Macro - Basic Loading
#include <cmath>

double t7a_add(double a, double b) {
    return a + b;
}

double t7a_pt(double px, double py) {
    return std::sqrt(px * px + py * py);
}
'''
    
    with tempfile.NamedTemporaryFile(suffix='_t7a.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    log(f"Created test macro: {macro_path}")
    
    try:
        # Load without compilation
        log("Loading macro with .L (interpret mode)...")
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}')
        
        observe("t7a_load_result", result, 
                "Return value of .L command (0 = success typically)")
        
        # Test function availability
        log("Testing function availability...")
        try:
            add_result = ROOT.t7a_add(2.0, 3.0)
            observe("t7a_add_callable", True, "Functions accessible via ROOT.funcname()")
            
            if abs(add_result - 5.0) < 0.001:
                log(f"t7a_add(2, 3) = {add_result} (correct)", "PASS")
            else:
                log(f"t7a_add(2, 3) = {add_result} (expected 5.0)", "FAIL")
                return False
                
        except AttributeError as e:
            observe("t7a_add_callable", False, f"Function not accessible: {e}")
            log(f"Function not available via ROOT.t7a_add: {e}", "FAIL")
            return False
        
        # Test second function
        try:
            pt_result = ROOT.t7a_pt(3.0, 4.0)
            if abs(pt_result - 5.0) < 0.001:
                log(f"t7a_pt(3, 4) = {pt_result} (correct)", "PASS")
            else:
                log(f"t7a_pt(3, 4) = {pt_result} (expected 5.0)", "WARN")
        except Exception as e:
            log(f"t7a_pt call failed: {e}", "WARN")
        
        log("T7a PASSED: Basic macro loading works", "PASS")
        return True
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        return False
        
    finally:
        try:
            os.unlink(macro_path)
        except:
            pass


# =============================================================================
# T7b: ACLiC Compilation
# =============================================================================

def test_t7b_aclic():
    """
    T7b: Does ACLiC compilation work?
    
    Hypothesis:
        '.L file.C+' compiles to .so
        Faster execution after compilation
    
    Validates:
        - ACLiC compilation succeeds
        - .so file created
        - Function callable after compilation
    """
    log("\n" + "="*60)
    log("T7b: ACLiC Compilation (.L file.C+)")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    code = '''
// T7b Test Macro - ACLiC Compilation
#include <cmath>

double t7b_multiply(double a, double b) {
    return a * b;
}

double t7b_eta(double px, double py, double pz) {
    double pt = std::sqrt(px * px + py * py);
    double p = std::sqrt(px * px + py * py + pz * pz);
    if (p < 1e-10) return 0.0;
    return 0.5 * std::log((p + pz) / (p - pz));
}
'''
    
    with tempfile.NamedTemporaryFile(suffix='_t7b.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    log(f"Created test macro: {macro_path}")
    so_files = []
    
    try:
        # Compile with ACLiC
        log("Compiling with ACLiC (.L macro.C+)...")
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}+')
        
        observe("t7b_aclic_result", result, 
                "Return value of ACLiC compilation")
        
        # Check for .so file
        base_name = macro_path.replace('.C', '_C')
        so_files = [f for f in glob.glob(f"{base_name}.*") 
                    if f.endswith('.so') or f.endswith('.dylib')]
        
        observe("t7b_so_created", len(so_files) > 0, 
                "Whether ACLiC creates shared library")
        
        if so_files:
            log(f"Shared library created: {so_files[0]}", "PASS")
            RESULTS["so_path"] = so_files[0]
        else:
            log("No .so/.dylib created", "WARN")
            # Check what files were created
            all_files = glob.glob(f"{base_name}.*")
            observe("t7b_files_created", all_files, "Files created by ACLiC")
        
        # Test function callable
        log("Testing compiled function...")
        try:
            mult_result = ROOT.t7b_multiply(3.0, 4.0)
            observe("t7b_function_callable", True, 
                    "Compiled functions accessible via ROOT.funcname()")
            
            if abs(mult_result - 12.0) < 0.001:
                log(f"t7b_multiply(3, 4) = {mult_result} (correct)", "PASS")
            else:
                log(f"t7b_multiply(3, 4) = {mult_result} (expected 12.0)", "FAIL")
                return False
                
        except Exception as e:
            observe("t7b_function_callable", False, str(e))
            log(f"Compiled function not callable: {e}", "FAIL")
            return False
        
        log("T7b PASSED: ACLiC compilation works", "PASS")
        return True
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        return False
        
    finally:
        try:
            os.unlink(macro_path)
        except:
            pass
        # Keep .so for T7d test
        RESULTS["t7b_so_files"] = so_files


# =============================================================================
# T7c: Namespace Access
# =============================================================================

def test_t7c_namespace():
    """
    T7c: How to access namespaced functions from Python?
    
    Hypothesis:
        namespace analysis { double f() {...} }
        Access via ROOT.analysis.f()
    
    Validates:
        - Namespace in macro works
        - Python access pattern (ROOT.ns.func or other)
    """
    log("\n" + "="*60)
    log("T7c: Namespace Access from Python")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    code = '''
// T7c Test Macro - Namespace
#include <cmath>

namespace t7c_analysis {

double compute_pt(double px, double py) {
    return std::sqrt(px * px + py * py);
}

double compute_mass(double e, double px, double py, double pz) {
    return std::sqrt(e*e - px*px - py*py - pz*pz);
}

} // namespace t7c_analysis
'''
    
    with tempfile.NamedTemporaryFile(suffix='_t7c.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    log(f"Created test macro: {macro_path}")
    
    try:
        # Load macro
        log("Loading namespaced macro...")
        ROOT.gROOT.ProcessLine(f'.L {macro_path}+')
        
        # Test Method 1: ROOT.namespace.function()
        log("Testing access via ROOT.t7c_analysis.compute_pt()...")
        method1_works = False
        try:
            result1 = ROOT.t7c_analysis.compute_pt(3.0, 4.0)
            method1_works = True
            log(f"ROOT.t7c_analysis.compute_pt(3, 4) = {result1}", "PASS")
        except Exception as e:
            log(f"ROOT.namespace.func() failed: {e}", "WARN")
        
        observe("t7c_method_ROOT_ns_func", method1_works,
                "Whether ROOT.namespace.function() syntax works")
        
        # Test Method 2: gInterpreter.ProcessLine
        log("Testing access via gInterpreter.ProcessLine...")
        method2_works = False
        try:
            ROOT.gInterpreter.ProcessLine('''
                double t7c_result = t7c_analysis::compute_pt(3.0, 4.0);
            ''')
            # Retrieve result
            result2 = ROOT.gInterpreter.ProcessLine('t7c_result')
            method2_works = True
            log(f"gInterpreter with t7c_analysis::compute_pt works", "PASS")
        except Exception as e:
            log(f"gInterpreter method failed: {e}", "WARN")
        
        observe("t7c_method_gInterpreter", method2_works,
                "Whether gInterpreter.ProcessLine with ns::func works")
        
        # Test Method 3: GetGlobalFunction
        log("Testing ROOT.gROOT.GetGlobalFunction...")
        method3_works = False
        try:
            func = ROOT.gROOT.GetGlobalFunction("t7c_analysis::compute_pt")
            if func:
                method3_works = True
                log(f"GetGlobalFunction found: {func}", "PASS")
            else:
                log("GetGlobalFunction returned None", "WARN")
        except Exception as e:
            log(f"GetGlobalFunction failed: {e}", "WARN")
        
        observe("t7c_method_GetGlobalFunction", method3_works,
                "Whether GetGlobalFunction can find namespaced functions")
        
        # Summary of access patterns
        log("\n--- T7c Summary: Namespace Access Patterns ---")
        log(f"ROOT.namespace.function():     {'✅ Works' if method1_works else '❌ Failed'}")
        log(f"gInterpreter ns::function:     {'✅ Works' if method2_works else '❌ Failed'}")
        log(f"GetGlobalFunction:             {'✅ Works' if method3_works else '❌ Failed'}")
        
        # Determine recommended approach
        if method1_works:
            observe("t7c_recommended_access", "ROOT.namespace.function()",
                    "Direct Python-style access works - recommended for v7")
        elif method2_works:
            observe("t7c_recommended_access", "gInterpreter.ProcessLine",
                    "Must use gInterpreter for namespace access")
        else:
            observe("t7c_recommended_access", "NONE_WORKED",
                    "Namespace access problematic - may need workaround")
        
        return method1_works or method2_works
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        return False
        
    finally:
        try:
            os.unlink(macro_path)
        except:
            pass
        # Cleanup .so
        base_name = macro_path.replace('.C', '_C')
        for f in glob.glob(f"{base_name}.*"):
            try:
                os.unlink(f)
            except:
                pass


# =============================================================================
# T7d: .so Persistence
# =============================================================================

def test_t7d_persistence():
    """
    T7d: Does .so survive Python session restart?
    
    Tests:
        1. Compile macro to .so
        2. Record .so path
        3. Test loading .so in subprocess (simulates restart)
        4. Verify functions available
    
    Note: We use subprocess to simulate session restart.
    """
    log("\n" + "="*60)
    log("T7d: .so Persistence Across Sessions")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    code = '''
// T7d Test Macro - Persistence
#include <cmath>

double t7d_persistent_func(double x) {
    return x * x;
}
'''
    
    # Use a fixed temp location
    macro_path = "/tmp/test_t7d_persist.C"
    with open(macro_path, 'w') as f:
        f.write(code)
    
    log(f"Created test macro: {macro_path}")
    
    try:
        # Step 1: Compile to .so
        log("Step 1: Compiling macro to .so...")
        ROOT.gROOT.ProcessLine(f'.L {macro_path}+')
        
        # Find .so file
        base_name = macro_path.replace('.C', '_C')
        so_files = [f for f in glob.glob(f"{base_name}.*") 
                    if f.endswith('.so') or f.endswith('.dylib')]
        
        if not so_files:
            log("No .so created - cannot test persistence", "FAIL")
            return False
        
        so_path = so_files[0]
        log(f"Shared library at: {so_path}", "PASS")
        observe("t7d_so_path", so_path, "Path to compiled shared library")
        
        # Step 2: Verify function works in current session
        log("Step 2: Verifying function in current session...")
        result_current = ROOT.t7d_persistent_func(5.0)
        log(f"t7d_persistent_func(5) = {result_current} in current session", "PASS")
        
        # Step 3: Test in subprocess (simulates new session)
        log("Step 3: Testing .so load in new Python session (subprocess)...")
        
        subprocess_code = f'''
import sys
try:
    import ROOT
    # Load the .so directly
    ROOT.gSystem.Load("{so_path}")
    # Try to call function
    result = ROOT.t7d_persistent_func(5.0)
    if abs(result - 25.0) < 0.001:
        print("SUBPROCESS_SUCCESS")
        print(f"Result: {{result}}")
    else:
        print("SUBPROCESS_WRONG_VALUE")
        print(f"Result: {{result}}")
except Exception as e:
    print(f"SUBPROCESS_ERROR: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, "-c", subprocess_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        log(f"Subprocess stdout: {result.stdout.strip()}")
        if result.stderr:
            log(f"Subprocess stderr: {result.stderr.strip()}", "WARN")
        
        subprocess_success = "SUBPROCESS_SUCCESS" in result.stdout
        observe("t7d_subprocess_load", subprocess_success,
                "Whether .so can be loaded in new session via gSystem.Load")
        
        if subprocess_success:
            log("T7d PASSED: .so persists and loads in new session", "PASS")
            return True
        else:
            log("T7d: .so exists but load in new session failed", "WARN")
            observe("t7d_subprocess_error", result.stdout + result.stderr,
                    "Error details from subprocess")
            return False
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        return False
        
    finally:
        # Cleanup
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
# Summary and Main
# =============================================================================

def print_summary():
    """Print test summary with observations."""
    print("\n" + "="*70)
    print("T7 TEST SUMMARY: Macro Loading")
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
    """Print markdown-formatted observation report."""
    print("\n")
    print("## Test T7: Macro Loading - Observation Report")
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
    print("### Observations")
    print()
    for key, obs in RESULTS.get("observations", {}).items():
        print(f"**{key}:** `{obs['value']}`")
        if obs.get('implication'):
            print(f"- Implication: {obs['implication']}")
        print()
    print("### Implications for v7 Specification")
    print()
    
    # Analyze observations for spec implications
    obs = RESULTS.get("observations", {})
    
    if obs.get("t7c_recommended_access", {}).get("value") == "ROOT.namespace.function()":
        print("- ✅ Namespace access works via `ROOT.namespace.function()` - export_macro with namespace is viable")
    else:
        print("- ⚠️ Namespace access may need alternative approach")
    
    if obs.get("t7d_subprocess_load", {}).get("value"):
        print("- ✅ .so persistence confirmed - compiled macros can be loaded in new sessions")
    else:
        print("- ⚠️ .so persistence needs investigation")
    print()


if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T7: Macro Loading")
    print("="*70)
    
    results = {
        "t7a": test_t7a_basic_loading(),
        "t7b": test_t7b_aclic(),
        "t7c": test_t7c_namespace(),
        "t7d": test_t7d_persistence(),
    }
    
    # Determine overall status
    all_passed = all(results.values())
    critical_passed = results["t7a"] and results["t7b"]
    
    if all_passed:
        RESULTS["status"] = "PASS"
    elif critical_passed:
        RESULTS["status"] = "PARTIAL_PASS"
    else:
        RESULTS["status"] = "FAIL"
    
    print_summary()
    print_observation_report()
    
    if all_passed:
        print("\n✅ T7 ALL TESTS PASSED")
        sys.exit(0)
    elif critical_passed:
        print("\n⚠️ T7 CRITICAL TESTS PASSED (some optional tests failed)")
        sys.exit(0)
    else:
        print("\n❌ T7 CRITICAL TESTS FAILED")
        sys.exit(1)
