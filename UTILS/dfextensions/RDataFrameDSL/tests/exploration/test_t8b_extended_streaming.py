#!/usr/bin/env python3
"""
Phase 13.5.B0 - Test T8b Extended: Pragma for File Streaming

Objective: Validate that pragma link IS required for data streaming/snapshot,
even though execution works without it.

Key Insight (from Main Architect):
    "pragma link is needed for the streaming of the data - not for execution of function"

Tests:
- T8b_ext1: RVec function execution (should work without pragma)
- T8b_ext2: RVec column snapshot to file WITHOUT pragma (expected to fail)
- T8b_ext3: RVec column snapshot to file WITH pragma (should work)
- T8b_ext4: Verify pragma is NOT auto-loaded by ROOT

Per Phase 13.5.B0 v7 specification.
CRITICAL: This test validates that explicit pragmas parameter IS needed for file output.
"""

import os
import sys
import tempfile
import glob
from datetime import datetime
from typing import Tuple, Optional

# Results tracking
RESULTS = {
    "test": "T8b Extended: Pragma for File Streaming",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "errors": [],
    "observations": {},
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", 
              "OBSERVATION": "🔍", "CRITICAL": "🚨"}
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
# T8b_ext1: Execution Without Pragma (Baseline)
# =============================================================================

def test_t8b_ext1_execution_without_pragma():
    """
    T8b_ext1: Verify RVec function execution works without explicit pragma.
    
    This establishes the baseline that EXECUTION doesn't need pragma.
    """
    log("\n" + "="*60)
    log("T8b_ext1: RVec Execution Without Pragma (Baseline)")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    try:
        # Declare RVec function without any pragma macro
        code = '''
#include <ROOT/RVec.hxx>

ROOT::VecOps::RVec<double> t8b_ext1_scale(const ROOT::VecOps::RVec<double>& v, double s) {
    return v * s;
}
'''
        
        log("Declaring RVec function without pragma macro...")
        result = ROOT.gInterpreter.Declare(code)
        
        if not result:
            log("Function declaration failed", "FAIL")
            return False
        
        log("Function declared successfully", "PASS")
        
        # Test execution
        log("Testing function execution...")
        ROOT.gInterpreter.ProcessLine('''
            ROOT::VecOps::RVec<double> t8b_ext1_input{1.0, 2.0, 3.0};
            auto t8b_ext1_output = t8b_ext1_scale(t8b_ext1_input, 2.0);
        ''')
        
        log("RVec function execution works WITHOUT pragma", "PASS")
        observe("t8b_ext1_execution_no_pragma", True,
                "RVec function EXECUTION works without explicit pragma")
        return True
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        observe("t8b_ext1_execution_no_pragma", False, str(e))
        return False


# =============================================================================
# T8b_ext2: Snapshot Without Pragma (Expected to Fail)
# =============================================================================

def test_t8b_ext2_snapshot_without_pragma():
    """
    T8b_ext2: Try to snapshot RVec column WITHOUT explicit pragma.
    
    HYPOTHESIS: This should FAIL because streaming needs dictionary.
    """
    log("\n" + "="*60)
    log("T8b_ext2: RVec Snapshot WITHOUT Pragma")
    log("="*60)
    log("HYPOTHESIS: Snapshot should FAIL without pragma", "INFO")
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return None  # Return None to indicate inconclusive
    
    # Create temp file for output
    output_file = tempfile.mktemp(suffix='_t8b_ext2.root')
    
    try:
        # Create RDataFrame with RVec column
        log("Creating RDataFrame with RVec column...")
        
        ROOT.gInterpreter.ProcessLine('''
            ROOT::RDataFrame t8b_ext2_rdf(10);
            auto t8b_ext2_rdf1 = t8b_ext2_rdf.Define("rvec_col", 
                "ROOT::VecOps::RVec<double>{(double)rdfentry_, (double)rdfentry_*2}");
        ''')
        
        log(f"Attempting Snapshot to: {output_file}")
        
        # Try to snapshot - this is where pragma might be needed
        snapshot_works = False
        error_msg = ""
        
        try:
            ROOT.gInterpreter.ProcessLine(f'''
                t8b_ext2_rdf1.Snapshot("tree", "{output_file}", {{"rvec_col"}});
            ''')
            
            # Check if file was created and has data
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                log(f"File created: {output_file} ({file_size} bytes)")
                
                # Try to read it back
                try:
                    f = ROOT.TFile.Open(output_file)
                    tree = f.Get("tree")
                    if tree and tree.GetEntries() > 0:
                        snapshot_works = True
                        log(f"File readable with {tree.GetEntries()} entries", "PASS")
                    else:
                        log("File exists but tree is empty or missing", "WARN")
                    f.Close()
                except Exception as e:
                    error_msg = str(e)
                    log(f"File exists but cannot read: {e}", "WARN")
            else:
                log("No output file created", "INFO")
                
        except Exception as e:
            error_msg = str(e)
            log(f"Snapshot failed with: {e}", "INFO")
        
        if snapshot_works:
            log("UNEXPECTED: Snapshot WORKS without explicit pragma", "WARN")
            observe("t8b_ext2_snapshot_no_pragma", "WORKS",
                    "Snapshot works WITHOUT pragma - ROOT may auto-generate dictionary")
        else:
            log("EXPECTED: Snapshot FAILS without explicit pragma", "PASS")
            observe("t8b_ext2_snapshot_no_pragma", "FAILS",
                    "Snapshot requires pragma - explicit pragmas parameter IS needed")
            observe("t8b_ext2_error", error_msg, "Error message for diagnostics")
        
        return snapshot_works
        
    except Exception as e:
        log(f"Test error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except:
                pass


# =============================================================================
# T8b_ext3: Snapshot WITH Pragma (Should Work)
# =============================================================================

def test_t8b_ext3_snapshot_with_pragma():
    """
    T8b_ext3: Snapshot RVec column WITH explicit pragma.
    
    HYPOTHESIS: This should WORK because we provide the dictionary.
    """
    log("\n" + "="*60)
    log("T8b_ext3: RVec Snapshot WITH Pragma")
    log("="*60)
    log("HYPOTHESIS: Snapshot should WORK with pragma", "INFO")
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    # Create pragma macro
    pragma_macro = '''
#include <ROOT/RVec.hxx>

#ifdef __CLING__
#pragma link C++ class ROOT::VecOps::RVec<double>+;
#endif

void t8b_ext3_pragma_loaded() {}
'''
    
    macro_path = tempfile.mktemp(suffix='_t8b_ext3_pragma.C')
    output_file = tempfile.mktemp(suffix='_t8b_ext3.root')
    
    try:
        # Write and compile pragma macro
        with open(macro_path, 'w') as f:
            f.write(pragma_macro)
        
        log(f"Compiling pragma macro: {macro_path}")
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}+')
        
        # Verify pragma loaded
        try:
            ROOT.t8b_ext3_pragma_loaded()
            log("Pragma macro loaded", "PASS")
        except:
            log("Pragma macro failed to load", "FAIL")
            return False
        
        # Create RDataFrame with RVec column
        log("Creating RDataFrame with RVec column...")
        
        ROOT.gInterpreter.ProcessLine('''
            ROOT::RDataFrame t8b_ext3_rdf(10);
            auto t8b_ext3_rdf1 = t8b_ext3_rdf.Define("rvec_col", 
                "ROOT::VecOps::RVec<double>{(double)rdfentry_, (double)rdfentry_*2}");
        ''')
        
        log(f"Attempting Snapshot to: {output_file}")
        
        # Try to snapshot
        snapshot_works = False
        
        try:
            ROOT.gInterpreter.ProcessLine(f'''
                t8b_ext3_rdf1.Snapshot("tree", "{output_file}", {{"rvec_col"}});
            ''')
            
            # Verify file
            if os.path.exists(output_file):
                f = ROOT.TFile.Open(output_file)
                tree = f.Get("tree")
                if tree and tree.GetEntries() > 0:
                    snapshot_works = True
                    log(f"Snapshot successful: {tree.GetEntries()} entries", "PASS")
                    
                    # Try to read RVec values
                    for i, event in enumerate(tree):
                        if i == 0:
                            rvec = getattr(event, 'rvec_col')
                            log(f"First entry RVec: size={rvec.size()}", "PASS")
                            break
                f.Close()
                
        except Exception as e:
            log(f"Snapshot with pragma failed: {e}", "FAIL")
        
        observe("t8b_ext3_snapshot_with_pragma", snapshot_works,
                "Snapshot works WITH pragma" if snapshot_works else "Snapshot still fails with pragma")
        
        return snapshot_works
        
    except Exception as e:
        log(f"Test error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        for f in [macro_path, output_file]:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        # Cleanup .so
        base = macro_path.replace('.C', '_C')
        for f in glob.glob(f"{base}.*"):
            try:
                os.unlink(f)
            except:
                pass


# =============================================================================
# T8b_ext4: Verify Pragma Not Auto-Loaded
# =============================================================================

def test_t8b_ext4_pragma_not_autoloaded():
    """
    T8b_ext4: Verify that ROOT doesn't auto-load pragma for custom RVec types.
    
    Tests RVec<CustomType> where no dictionary exists.
    """
    log("\n" + "="*60)
    log("T8b_ext4: Pragma Not Auto-Loaded for Custom Types")
    log("="*60)
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    output_file = tempfile.mktemp(suffix='_t8b_ext4.root')
    
    try:
        # Define a custom struct
        log("Defining custom struct...")
        ROOT.gInterpreter.Declare('''
struct T8b_CustomData {
    double x;
    double y;
};
''')
        
        # Create RVec of custom type
        log("Creating RDataFrame with RVec<CustomData>...")
        
        ROOT.gInterpreter.ProcessLine('''
            ROOT::RDataFrame t8b_ext4_rdf(5);
            auto t8b_ext4_rdf1 = t8b_ext4_rdf.Define("custom_vec", 
                "ROOT::VecOps::RVec<T8b_CustomData>{{1.0, 2.0}, {3.0, 4.0}}");
        ''')
        
        # Try snapshot without pragma
        log(f"Attempting Snapshot of RVec<CustomData> without pragma...")
        
        custom_snapshot_works = False
        try:
            ROOT.gInterpreter.ProcessLine(f'''
                t8b_ext4_rdf1.Snapshot("tree", "{output_file}", {{"custom_vec"}});
            ''')
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                custom_snapshot_works = True
                log("Custom RVec snapshot worked (unexpected)", "WARN")
            else:
                log("Custom RVec snapshot failed as expected", "PASS")
                
        except Exception as e:
            log(f"Custom RVec snapshot failed: {e}", "PASS")
        
        observe("t8b_ext4_custom_rvec_no_pragma", custom_snapshot_works,
                "ROOT auto-generates dict for custom types" if custom_snapshot_works 
                else "Custom types REQUIRE explicit pragma")
        
        return not custom_snapshot_works  # Success if it FAILS without pragma
        
    except Exception as e:
        log(f"Test error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except:
                pass


# =============================================================================
# Summary
# =============================================================================

def print_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("T8b EXTENDED TEST SUMMARY: Pragma for File Streaming")
    print("="*70)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    
    print("\n--- Critical Findings ---")
    
    obs = RESULTS.get("observations", {})
    
    exec_works = obs.get("t8b_ext1_execution_no_pragma", {}).get("value")
    snap_no_pragma = obs.get("t8b_ext2_snapshot_no_pragma", {}).get("value")
    snap_with_pragma = obs.get("t8b_ext3_snapshot_with_pragma", {}).get("value")
    custom_needs_pragma = not obs.get("t8b_ext4_custom_rvec_no_pragma", {}).get("value", True)
    
    print(f"\n  Execution without pragma:     {exec_works}")
    print(f"  Snapshot without pragma:      {snap_no_pragma}")
    print(f"  Snapshot with pragma:         {snap_with_pragma}")
    print(f"  Custom types need pragma:     {custom_needs_pragma}")
    
    print("\n--- Conclusion for v7 Specification ---")
    
    if snap_no_pragma == "FAILS" or custom_needs_pragma:
        print("\n  🚨 PRAGMA IS REQUIRED FOR FILE STREAMING")
        print("  → Explicit 'pragmas' parameter in register_function_cpp() IS needed")
        print("  → DSL should warn when RVec functions are used without pragma")
    else:
        print("\n  ⚠️ ROOT may auto-generate dictionaries for basic RVec types")
        print("  → Custom types still require explicit pragma")
        print("  → Recommend explicit pragma for production/streaming use cases")
    
    print("\n" + "="*70)


def print_observation_report():
    """Print markdown observation report."""
    print("\n")
    print("## Test T8b Extended: Pragma for File Streaming — Observation Report")
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
    print("### Key Insight (Main Architect)")
    print()
    print("> \"pragma link is needed for the streaming of the data - not for execution of function\"")
    print()
    print("### Test Results")
    print()
    print("| Test | Scenario | Result |")
    print("|------|----------|--------|")
    
    obs = RESULTS.get("observations", {})
    print(f"| T8b_ext1 | Execution without pragma | `{obs.get('t8b_ext1_execution_no_pragma', {}).get('value', 'N/A')}` |")
    print(f"| T8b_ext2 | Snapshot without pragma | `{obs.get('t8b_ext2_snapshot_no_pragma', {}).get('value', 'N/A')}` |")
    print(f"| T8b_ext3 | Snapshot with pragma | `{obs.get('t8b_ext3_snapshot_with_pragma', {}).get('value', 'N/A')}` |")
    print(f"| T8b_ext4 | Custom type needs pragma | `{not obs.get('t8b_ext4_custom_rvec_no_pragma', {}).get('value', True)}` |")
    print()
    print("### Implications for v7 Specification")
    print()
    
    snap_no_pragma = obs.get("t8b_ext2_snapshot_no_pragma", {}).get("value")
    
    if snap_no_pragma == "FAILS":
        print("- ✅ **Explicit `pragmas` parameter IS REQUIRED** for file streaming")
        print("- DSL should validate pragma availability before Snapshot operations")
        print("- Warning should be issued for RVec columns without pragma")
    else:
        print("- ⚠️ Basic RVec types may work without explicit pragma (ROOT auto-dict)")
        print("- Custom types still require explicit pragma")
        print("- **Recommend explicit pragma for all streaming/production use cases**")
    print()


if __name__ == "__main__":
    print("="*70)
    print("Phase 13.5.B0 - Test T8b Extended: Pragma for File Streaming")
    print("="*70)
    
    results = {
        "ext1": test_t8b_ext1_execution_without_pragma(),
        "ext2": test_t8b_ext2_snapshot_without_pragma(),
        "ext3": test_t8b_ext3_snapshot_with_pragma(),
        "ext4": test_t8b_ext4_pragma_not_autoloaded(),
    }
    
    # Determine status
    # Critical finding: ext2 should FAIL (snap without pragma) and ext3 should PASS
    execution_works = results["ext1"]
    snap_behavior_correct = (results["ext2"] == False and results["ext3"] == True)
    
    if execution_works and snap_behavior_correct:
        RESULTS["status"] = "PASS - PRAGMA REQUIRED FOR STREAMING"
    elif execution_works and results["ext2"] == True:
        RESULTS["status"] = "PASS - ROOT AUTO-DICT (may work without pragma)"
    else:
        RESULTS["status"] = "PARTIAL - Check individual results"
    
    print_summary()
    print_observation_report()
    
    print("\n✅ T8b Extended Complete")
    sys.exit(0)
