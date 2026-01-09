#!/bin/bash
# Phase 13.5.A - Test T1: ACLiC Basic Compilation (Shell Script Version)
#
# Main Architect suggestion: Test via external ROOT command
# This is the simplest way to verify ACLiC works.
#
# Usage: ./test_t1_aclic_basic.sh

set -e

echo "============================================================"
echo "Phase 13.5.A - Test T1: ACLiC Basic Compilation"
echo "============================================================"

# Create temp directory
TMPDIR=$(mktemp -d)
MACRO_PATH="${TMPDIR}/test_t1.C"

echo "Creating test macro: ${MACRO_PATH}"

# Generate test macro
cat > "${MACRO_PATH}" << 'EOF'
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

// Test harness
void run_tests() {
    bool all_pass = true;
    
    // Test 1: dsl_pt
    double pt = dsl_pt(3.0, 4.0);
    if (std::abs(pt - 5.0) < 0.001) {
        std::cout << "✅ dsl_pt(3, 4) = " << pt << " (PASS)" << std::endl;
    } else {
        std::cout << "❌ dsl_pt(3, 4) = " << pt << " (FAIL, expected 5.0)" << std::endl;
        all_pass = false;
    }
    
    // Test 2: dsl_eta
    // Note: eta = -log(tan(atan2(pt,pz)/2))
    // For pt=5, pz=5: theta=pi/4, eta=-log(tan(pi/8)) ≈ 0.8814
    double eta = dsl_eta(5.0, 5.0);
    if (std::abs(eta - 0.8814) < 0.01) {
        std::cout << "✅ dsl_eta(5, 5) = " << eta << " (PASS)" << std::endl;
    } else {
        std::cout << "❌ dsl_eta(5, 5) = " << eta << " (FAIL, expected ~0.88)" << std::endl;
        all_pass = false;
    }
    
    if (all_pass) {
        std::cout << "\n✅ T1 PASS: All function tests passed" << std::endl;
    } else {
        std::cout << "\n❌ T1 FAIL: Some tests failed" << std::endl;
    }
}
EOF

echo "Macro created. Compiling with ACLiC..."

# Test compilation via external ROOT command
# Using ++ to force recompile
root -l -b -q << ROOTCMD
std::cout << "Loading and compiling macro..." << std::endl;
int result = gROOT->ProcessLine(".L ${MACRO_PATH}++");
if (result != 0) {
    std::cerr << "❌ ACLiC compilation FAILED with code " << result << std::endl;
    gSystem->Exit(1);
}
std::cout << "✅ ACLiC compilation succeeded" << std::endl;

// Run tests
run_tests();

// Check if shared library was created
TString so_path = "${MACRO_PATH}";
so_path.ReplaceAll(".C", "_C.so");
if (gSystem->AccessPathName(so_path)) {
    // Try .dylib for macOS
    so_path.ReplaceAll(".so", ".dylib");
    if (!gSystem->AccessPathName(so_path)) {
        std::cout << "✅ Shared library created: " << so_path << std::endl;
    } else {
        std::cout << "⚠️  No shared library found (may be platform-specific)" << std::endl;
    }
} else {
    std::cout << "✅ Shared library created: " << so_path << std::endl;
}

.q
ROOTCMD

EXIT_CODE=$?

# Cleanup
echo ""
echo "Cleaning up..."
rm -rf "${TMPDIR}"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ T1 PASS: ACLiC basic compilation works"
    echo "============================================================"
    exit 0
else
    echo ""
    echo "============================================================"
    echo "❌ T1 FAIL: ACLiC compilation failed"
    echo "============================================================"
    exit 1
fi
