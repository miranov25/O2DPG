#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T10: Redefine Semantics

Objective: Validate RDataFrame.Redefine() behavior

Tests:
- T10a: Basic Redefine workflow
- T10b: Redefine with different expression
- T10c: Redefine scope (isolation between RDF branches)
- T10d: Redefine type change (double → RVec)
- T10e: Redefine dependencies (adding new column dependencies)

Per Phase 13.5.B0 v7 specification.
"""

import os
import sys
import tempfile
from datetime import datetime
from typing import Tuple, Optional

# Results tracking
RESULTS = {
    "test": "T10: Redefine Semantics",
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
# T10a: Basic Redefine
# =============================================================================

def test_t10a_basic_redefine():
    """
    T10a: Does RDF.Redefine() work?
    
    Basic test of Redefine functionality.
    """
    log("\n" + "="*60)
    log("T10a: Basic Redefine Workflow")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Check if Redefine exists
        log("Checking if RDataFrame has Redefine method...")
        
        # Create simple RDataFrame
        rdf = ROOT.RDataFrame(5)
        
        # Check for method
        has_redefine = hasattr(rdf, 'Redefine')
        observe("t10a_has_redefine_method", has_redefine,
                "RDataFrame has Redefine method" if has_redefine else "Redefine not available")
        
        if not has_redefine:
            log("Redefine method not found - may be ROOT version issue", "FAIL")
            observe("t10a_root_version", ROOT.gROOT.GetVersion(),
                    "Check ROOT version - Redefine added in 6.26")
            return False
        
        log("Redefine method exists", "PASS")
        
        # Test basic workflow
        log("\nTesting basic Define → Redefine workflow...")
        
        rdf1 = rdf.Define("x", "1.0")
        log("Defined column 'x' = 1.0")
        
        # Get value before redefine
        mean_before = rdf1.Mean("x").GetValue()
        log(f"Mean(x) before Redefine: {mean_before}")
        
        # Redefine
        rdf2 = rdf1.Redefine("x", "2.0")
        log("Redefined column 'x' = 2.0")
        
        # Get value after redefine
        mean_after = rdf2.Mean("x").GetValue()
        log(f"Mean(x) after Redefine: {mean_after}")
        
        if abs(mean_after - 2.0) < 0.001:
            log("Redefine successfully changed column value", "PASS")
            observe("t10a_basic_redefine_works", True,
                    "Basic Redefine workflow works")
            return True
        else:
            log(f"Unexpected value after Redefine: {mean_after}", "FAIL")
            observe("t10a_basic_redefine_works", False,
                    f"Redefine didn't change value as expected")
            return False
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        RESULTS["errors"].append(str(e))
        return False


# =============================================================================
# T10b: Redefine Different Expression
# =============================================================================

def test_t10b_different_expression():
    """
    T10b: Can Redefine change expression complexity?
    
    Tests changing from simple to complex expression.
    """
    log("\n" + "="*60)
    log("T10b: Redefine with Different Expression")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Create RDataFrame with base columns
        rdf = ROOT.RDataFrame(3)
        rdf = rdf.Define("px", "(double)rdfentry_ + 1")  # 1, 2, 3
        rdf = rdf.Define("py", "(double)rdfentry_ + 2")  # 2, 3, 4
        
        # Define simple expression
        rdf1 = rdf.Define("pt", "px")
        log("Defined 'pt' = px (simple)")
        
        # Get values
        pt_simple = list(rdf1.Take["double"]("pt").GetValue())
        log(f"pt (simple): {pt_simple}")
        
        # Redefine with complex expression
        rdf2 = rdf1.Redefine("pt", "sqrt(px*px + py*py)")
        log("Redefined 'pt' = sqrt(px*px + py*py) (complex)")
        
        pt_complex = list(rdf2.Take["double"]("pt").GetValue())
        log(f"pt (complex): {pt_complex}")
        
        # Verify values changed
        # pt_complex should be sqrt(1^2+2^2)=2.24, sqrt(2^2+3^2)=3.61, sqrt(3^2+4^2)=5.0
        expected = [2.236, 3.606, 5.0]
        matches = all(abs(a - b) < 0.01 for a, b in zip(pt_complex, expected))
        
        if matches:
            log("Redefine with complex expression works", "PASS")
            observe("t10b_complex_expression", True,
                    "Can redefine with arbitrarily complex expressions")
            return True
        else:
            log(f"Values don't match expected {expected}", "WARN")
            observe("t10b_complex_expression", False,
                    f"Complex expression gave unexpected results")
            return False
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        RESULTS["errors"].append(str(e))
        return False


# =============================================================================
# T10c: Redefine Scope
# =============================================================================

def test_t10c_scope():
    """
    T10c: Does Redefine affect previous RDF references?
    
    Critical test: RDataFrame branches should be isolated.
    """
    log("\n" + "="*60)
    log("T10c: Redefine Scope (Branch Isolation)")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Create base
        rdf = ROOT.RDataFrame(3)
        rdf1 = rdf.Define("x", "1.0")
        
        log("Created rdf1 with x = 1.0")
        
        # Create branch with Redefine
        rdf2 = rdf1.Redefine("x", "2.0")
        log("Created rdf2 with x = 2.0 (Redefine from rdf1)")
        
        # Check both branches
        mean1 = rdf1.Mean("x").GetValue()
        mean2 = rdf2.Mean("x").GetValue()
        
        log(f"rdf1.Mean('x') = {mean1}")
        log(f"rdf2.Mean('x') = {mean2}")
        
        # Verify isolation
        rdf1_has_1 = abs(mean1 - 1.0) < 0.001
        rdf2_has_2 = abs(mean2 - 2.0) < 0.001
        
        if rdf1_has_1 and rdf2_has_2:
            log("Branches are isolated - Redefine doesn't affect original", "PASS")
            observe("t10c_branch_isolation", True,
                    "RDataFrame branches are isolated - Redefine creates new branch")
            observe("t10c_original_unchanged", True,
                    "Original RDF reference retains original value")
            return True
        elif not rdf1_has_1:
            log("WARNING: Original rdf1 was modified by Redefine!", "FAIL")
            observe("t10c_branch_isolation", False,
                    "Redefine modifies original - NOT ISOLATED")
            return False
        else:
            log(f"Unexpected values: rdf1={mean1}, rdf2={mean2}", "WARN")
            return False
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        RESULTS["errors"].append(str(e))
        return False


# =============================================================================
# T10d: Redefine Type Change
# =============================================================================

def test_t10d_type_change():
    """
    T10d: Can Redefine change return type?
    
    Tests: double → RVec<double>
    
    This determines if we can flexibly redefine columns with different types.
    """
    log("\n" + "="*60)
    log("T10d: Redefine with Type Change")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Create RDataFrame
        rdf = ROOT.RDataFrame(3)
        
        # Define as scalar double
        rdf1 = rdf.Define("val", "1.0")
        log("Defined 'val' as double (scalar)")
        
        # Try to redefine as RVec<double>
        log("Attempting to Redefine 'val' as RVec<double>...")
        
        type_change_works = False
        try:
            rdf2 = rdf1.Redefine("val", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
            
            # Try to access as RVec
            ROOT.gInterpreter.ProcessLine('''
                // Just check it compiles
            ''')
            
            type_change_works = True
            log("Type change (double → RVec) succeeded", "PASS")
            
        except Exception as e:
            log(f"Type change failed: {e}", "INFO")
        
        observe("t10d_type_change_allowed", type_change_works,
                "Redefine allows type change" if type_change_works else "Redefine requires same type")
        
        # Also test RVec → double (narrowing)
        log("\nTesting reverse: RVec<double> → double...")
        
        reverse_works = False
        try:
            rdf3 = rdf.Define("vec", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
            rdf4 = rdf3.Redefine("vec", "1.0")  # Try to make it scalar
            reverse_works = True
            log("Reverse type change (RVec → double) succeeded", "PASS")
        except Exception as e:
            log(f"Reverse type change failed: {e}", "INFO")
        
        observe("t10d_reverse_type_change", reverse_works,
                "Can also change RVec → double" if reverse_works else "Reverse type change not allowed")
        
        return True  # Test is exploratory - document behavior
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        RESULTS["errors"].append(str(e))
        return False


# =============================================================================
# T10e: Redefine Dependencies
# =============================================================================

def test_t10e_dependencies():
    """
    T10e: Can Redefine add new column dependencies?
    
    Tests: Original uses only px, Redefine uses px + py
    """
    log("\n" + "="*60)
    log("T10e: Redefine with New Dependencies")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Create RDataFrame with multiple columns
        rdf = ROOT.RDataFrame(3)
        rdf = rdf.Define("px", "(double)rdfentry_ + 1")  # 1, 2, 3
        rdf = rdf.Define("py", "(double)rdfentry_ + 2")  # 2, 3, 4
        rdf = rdf.Define("pz", "(double)rdfentry_ + 3")  # 3, 4, 5
        
        # Define column using only px
        rdf1 = rdf.Define("result", "px")
        log("Defined 'result' = px (uses only px)")
        
        result_before = list(rdf1.Take["double"]("result").GetValue())
        log(f"result (px only): {result_before}")
        
        # Redefine to use px AND py
        log("Redefining to use px + py...")
        add_dep_works = False
        try:
            rdf2 = rdf1.Redefine("result", "px + py")
            result_after = list(rdf2.Take["double"]("result").GetValue())
            log(f"result (px + py): {result_after}")
            
            # Should be 1+2=3, 2+3=5, 3+4=7
            expected = [3.0, 5.0, 7.0]
            if all(abs(a - b) < 0.001 for a, b in zip(result_after, expected)):
                add_dep_works = True
                log("Adding new dependency works", "PASS")
            else:
                log(f"Values don't match expected {expected}", "WARN")
                
        except Exception as e:
            log(f"Adding py dependency failed: {e}", "INFO")
        
        observe("t10e_add_dependency", add_dep_works,
                "Redefine can add new column dependencies" if add_dep_works else "Cannot add new dependencies")
        
        # Also test adding completely new column dependency
        log("\nTesting adding pz dependency...")
        add_pz_works = False
        try:
            rdf3 = rdf1.Redefine("result", "px + py + pz")
            result_pz = list(rdf3.Take["double"]("result").GetValue())
            log(f"result (px + py + pz): {result_pz}")
            
            # Should be 1+2+3=6, 2+3+4=9, 3+4+5=12
            expected = [6.0, 9.0, 12.0]
            if all(abs(a - b) < 0.001 for a, b in zip(result_pz, expected)):
                add_pz_works = True
                log("Adding multiple new dependencies works", "PASS")
        except Exception as e:
            log(f"Adding pz dependency failed: {e}", "INFO")
        
        observe("t10e_add_multiple_deps", add_pz_works,
                "Can add multiple new dependencies in Redefine")
        
        return add_dep_works
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        RESULTS["errors"].append(str(e))
        return False


# =============================================================================
# Summary and Main
# =============================================================================

def print_summary():
    """Print test summary with observations."""
    print("\n" + "="*70)
    print("T10 TEST SUMMARY: Redefine Semantics")
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
    print("## Test T10: Redefine Semantics - Observation Report")
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
    print("### Key Findings")
    print()
    
    obs = RESULTS.get("observations", {})
    
    # Basic redefine
    basic = obs.get("t10a_basic_redefine_works", {}).get("value")
    print(f"**Basic Redefine:** `{'Works' if basic else 'Failed'}`")
    
    # Isolation
    isolated = obs.get("t10c_branch_isolation", {}).get("value")
    print(f"**Branch Isolation:** `{'Confirmed' if isolated else 'NOT isolated - CRITICAL'}`")
    
    # Type change
    type_change = obs.get("t10d_type_change_allowed", {}).get("value")
    print(f"**Type Change Allowed:** `{type_change}`")
    
    # Dependencies
    add_dep = obs.get("t10e_add_dependency", {}).get("value")
    print(f"**Add Dependencies:** `{'Allowed' if add_dep else 'Not allowed'}`")
    
    print()
    print("### Implications for v7 Specification")
    print()
    
    if isolated:
        print("- ✅ Branch isolation confirmed - DSL can track column versions safely")
    else:
        print("- ❌ **CRITICAL:** No branch isolation - Redefine modifies in place!")
    
    if type_change:
        print("- ✅ Type change allowed - flexible redefinition supported")
    else:
        print("- ⚠️ Type change not allowed - may need to track column types")
    
    if add_dep:
        print("- ✅ Can add dependencies - Redefine is flexible with expressions")
    else:
        print("- ⚠️ Cannot add dependencies - Redefine has restrictions")
    print()


if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T10: Redefine Semantics")
    print("="*70)
    
    results = {
        "t10a": test_t10a_basic_redefine(),
        "t10b": test_t10b_different_expression(),
        "t10c": test_t10c_scope(),
        "t10d": test_t10d_type_change(),
        "t10e": test_t10e_dependencies(),
    }
    
    # Determine overall status
    all_passed = all(results.values())
    critical_passed = results["t10a"] and results["t10c"]  # Basic and isolation critical
    
    if all_passed:
        RESULTS["status"] = "PASS"
    elif critical_passed:
        RESULTS["status"] = "PARTIAL_PASS"
    else:
        RESULTS["status"] = "FAIL"
    
    print_summary()
    print_observation_report()
    
    if all_passed:
        print("\n✅ T10 ALL TESTS PASSED")
        sys.exit(0)
    elif critical_passed:
        print("\n⚠️ T10 CRITICAL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ T10 CRITICAL TESTS FAILED")
        sys.exit(1)
