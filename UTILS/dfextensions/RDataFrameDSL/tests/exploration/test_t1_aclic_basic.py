#!/usr/bin/env python3
"""
Phase 13.5.A - Test T1: ACLiC Basic Compilation

Objective: Verify that DSL-generated code compiles with ACLiC

Success Criteria:
- ✅ Compilation succeeds (return code 0)
- ✅ Function callable from ROOT (primary criterion)
- ⚠️ Shared library created (platform-dependent, informational only)

Per Phase 13.5 v0.3 specification.
"""

import os
import sys
import tempfile
import glob
from datetime import datetime

# Results tracking
RESULTS = {
    "test": "T1: ACLiC Basic Compilation",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "errors": [],
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
    print(f"{prefix.get(level, '')} {msg}")
    RESULTS["findings"].append(f"[{level}] {msg}")


def test_aclic_basic():
    """Test basic ACLiC compilation workflow."""
    
    log("Starting T1: ACLiC Basic Compilation Test")
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available - cannot run this test", "FAIL")
        RESULTS["status"] = "SKIP_NO_ROOT"
        return False
    
    # Generate simple DSL code (manually, to test ACLiC independently of DSLCompiler)
    code = '''
// T1 Test Macro - Phase 13.5.A Exploration
// Tests ACLiC compilation workflow

#include <cmath>
#include <iostream>

// Simple scalar function
double dsl_pt(double px, double py) {
    return std::sqrt(px * px + py * py);
}

// Function with more complex math
double dsl_eta(double pt, double pz) {
    if (pt < 1e-10) return 0.0;
    return -std::log(std::tan(std::atan2(pt, pz) / 2.0));
}

// Test function
void test_t1() {
    std::cout << "T1 Test Results:" << std::endl;
    std::cout << "  dsl_pt(3, 4) = " << dsl_pt(3.0, 4.0) << " (expected: 5.0)" << std::endl;
    std::cout << "  dsl_eta(5, 5) = " << dsl_eta(5.0, 5.0) << " (expected: ~0.88)" << std::endl;
}
'''
    
    # Write to temp file
    so_pattern = None
    with tempfile.NamedTemporaryFile(suffix='.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    log(f"Created test macro: {macro_path}")
    
    try:
        # Test 1: Compile with ACLiC (force recompile)
        log("Compiling with ACLiC (.L macro.C++)...")
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}++')
        
        if result != 0:
            log(f"ACLiC compilation failed with code {result}", "FAIL")
            RESULTS["status"] = "FAIL"
            return False
        
        log("ACLiC compilation succeeded", "PASS")
        
        # Test 2: Verify function is callable (PRIMARY SUCCESS CRITERION)
        log("Testing function callability...")
        try:
            result_val = ROOT.dsl_pt(3.0, 4.0)
            if abs(result_val - 5.0) < 0.001:
                log(f"dsl_pt(3.0, 4.0) = {result_val} (correct!)", "PASS")
            else:
                log(f"dsl_pt(3.0, 4.0) = {result_val} (expected 5.0)", "FAIL")
                RESULTS["status"] = "FAIL"
                return False
        except Exception as e:
            log(f"Function call failed: {e}", "FAIL")
            RESULTS["status"] = "FAIL"
            return False
        
        # Test 3: Test eta function
        # Note: eta = -log(tan(atan2(pt,pz)/2))
        # For pt=5, pz=5: theta=pi/4, eta=-log(tan(pi/8)) ≈ 0.8814
        try:
            eta_val = ROOT.dsl_eta(5.0, 5.0)
            expected_eta = 0.8814  # -log(tan(atan2(5,5)/2)) = 0.8814
            if abs(eta_val - expected_eta) < 0.01:
                log(f"dsl_eta(5.0, 5.0) = {eta_val:.4f} (correct!)", "PASS")
            else:
                log(f"dsl_eta(5.0, 5.0) = {eta_val:.4f} (expected ~{expected_eta})", "WARN")
        except Exception as e:
            log(f"dsl_eta call failed: {e}", "WARN")
        
        # Test 4: Check for shared library (OPTIONAL - platform dependent)
        base_dir = os.path.dirname(macro_path)
        base_name = os.path.basename(macro_path).replace('.C', '_C')
        so_pattern = os.path.join(base_dir, f"{base_name}.*")
        so_files = glob.glob(so_pattern)
        
        if so_files:
            log(f"Shared library created: {so_files[0]}", "PASS")
            RESULTS["shared_library"] = so_files[0]
        else:
            log("No shared library found (may be platform-specific)", "WARN")
        
        # Test 5: Run built-in test function
        log("Running test_t1() from macro...")
        try:
            ROOT.test_t1()
            log("test_t1() executed successfully", "PASS")
        except Exception as e:
            log(f"test_t1() failed: {e}", "WARN")
        
        RESULTS["status"] = "PASS"
        return True
        
    except Exception as e:
        log(f"Unexpected error: {e}", "FAIL")
        RESULTS["errors"].append(str(e))
        RESULTS["status"] = "FAIL"
        return False
        
    finally:
        # Cleanup macro file only - keep .so until after T1b to avoid ACLiC cache issues
        try:
            os.unlink(macro_path)
            log(f"Cleaned up macro file")
        except:
            pass
        
        # Store pattern for deferred cleanup (done after T1b)
        RESULTS["cleanup_pattern"] = so_pattern


def test_aclic_with_rvec():
    """
    Test ACLiC compilation with RVec types.
    
    This test verifies RVec function compilation and execution.
    If compilation fails, we capture diagnostic info to understand why.
    """
    
    log("\n" + "="*60)
    log("T1b: ACLiC with RVec types")
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available", "FAIL")
        return False
    
    code = '''
#include <cmath>
#include <ROOT/RVec.hxx>

using namespace ROOT::VecOps;

// Vectorized pt calculation using RVec element-wise operations
RVec<double> dsl_track_pts_v6(const RVec<double>& px, const RVec<double>& py) {
    return sqrt(px * px + py * py);
}

// Size calculation
int dsl_n_tracks_v6(const RVec<double>& px) {
    return static_cast<int>(px.size());
}
'''
    
    with tempfile.NamedTemporaryFile(suffix='.C', delete=False, mode='w') as f:
        f.write(code)
        macro_path = f.name
    
    try:
        log(f"Compiling RVec macro: {macro_path}")
        
        # Get compilation verbosity
        old_level = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kInfo  # Show all messages
        
        result = ROOT.gROOT.ProcessLine(f'.L {macro_path}++')
        
        ROOT.gErrorIgnoreLevel = old_level
        
        # Check for .so file (more reliable than return code)
        base_name = macro_path.replace('.C', '_C')
        so_files = [f for f in glob.glob(f"{base_name}.*") if f.endswith('.so') or f.endswith('.dylib')]
        
        if not so_files:
            log(f"ACLiC return code: {result}", "WARN")
            log("No .so/.dylib file created - compilation actually failed!", "FAIL")
            log("This is likely an ACLiC shell quoting issue with complex include paths", "WARN")
            
            # Show what files WERE created
            all_files = glob.glob(f"{base_name}.*")
            if all_files:
                log(f"Files created: {all_files}", "INFO")
            
            return False
        
        log(f"Shared library created: {so_files[0]}", "PASS")
        
        # Verify function is actually available
        try:
            ROOT.gInterpreter.ProcessLine("dsl_track_pts_v6;")
            log("Function symbol available in interpreter", "PASS")
        except:
            log("Function symbol NOT available despite .so creation", "FAIL")
            return False
        
        # Test execution via interpreter (robust method)
        log("Testing RVec function execution...")
        try:
            ROOT.gInterpreter.ProcessLine('''
                ROOT::RVec<double> test_px = {3.0, 4.0, 5.0};
                ROOT::RVec<double> test_py = {4.0, 3.0, 12.0};
                auto test_pts = dsl_track_pts_v6(test_px, test_py);
                if (std::abs(test_pts[0] - 5.0) > 0.001 ||
                    std::abs(test_pts[1] - 5.0) > 0.001 ||
                    std::abs(test_pts[2] - 13.0) > 0.001) {
                    throw std::runtime_error("RVec results incorrect");
                }
            ''')
            log("RVec function returns correct values [5.0, 5.0, 13.0]", "PASS")
            return True
            
        except Exception as e:
            log(f"RVec execution failed: {e}", "FAIL")
            return False
        
    except Exception as e:
        log(f"RVec test error: {e}", "FAIL")
        import traceback
        traceback.print_exc()
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


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("T1 TEST SUMMARY")
    print("="*60)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    print("\nFindings:")
    for finding in RESULTS["findings"]:
        print(f"  {finding}")
    if RESULTS["errors"]:
        print("\nErrors:")
        for error in RESULTS["errors"]:
            print(f"  {error}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("Phase 13.5.A - Test T1: ACLiC Basic Compilation")
    print("="*60)
    
    success1 = test_aclic_basic()
    success2 = test_aclic_with_rvec()
    
    # Deferred cleanup - clean up T1's .so files AFTER T1b completes
    # This avoids ACLiC cache issues where T1b tries to link against T1's deleted .so
    if "cleanup_pattern" in RESULTS and RESULTS["cleanup_pattern"]:
        for so_file in glob.glob(RESULTS["cleanup_pattern"]):
            try:
                os.unlink(so_file)
                log(f"Deferred cleanup: {so_file}")
            except:
                pass
    
    print_summary()
    
    if success1 and success2:
        print("\n✅ T1 PASS: ACLiC compilation works for scalar and RVec types")
        sys.exit(0)
    else:
        print("\n❌ T1 FAIL: Some tests failed")
        sys.exit(1)
