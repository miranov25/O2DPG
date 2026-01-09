#!/bin/bash
# Phase 13.5.A - Test T2-Full: Debug Symbols (Interactive)
#
# This is for MANUAL validation on developer workstation.
# Tests actual GDB/LLDB breakpoint and inspection workflow.
#
# Success Criteria:
# - Can attach GDB to ROOT process
# - Can set breakpoint in dsl_buggy
# - Can inspect variables (x, result, i)
# - Can step through loop
# - Line numbers match source file
#
# Platform Notes:
# - Linux: Usually works (may need: sudo sysctl kernel.yama.ptrace_scope=0)
# - macOS: Requires disabling SIP or code signing
# - Containers: May have ptrace restrictions
#
# Usage: ./test_t2_debug_interactive.sh

echo "============================================================"
echo "Phase 13.5.A - Test T2-Full: Debug Symbols (Interactive)"
echo "============================================================"

# Create persistent directory for this test
TEST_DIR="${HOME}/.phase13_5_exploration"
mkdir -p "${TEST_DIR}"

MACRO_PATH="${TEST_DIR}/debug_test.C"

echo "Creating debug test macro: ${MACRO_PATH}"

# Generate macro with clear debug landmarks
cat > "${MACRO_PATH}" << 'EOF'
// T2-Full: Debug Symbols Test Macro
// Phase 13.5.A Exploration
//
// To debug:
//   $ gdb --args root -l
//   (gdb) break dsl_buggy
//   (gdb) run
//   root [0] .L debug_test.C++g
//   root [1] dsl_buggy(1.0)
//   (gdb) print x
//   (gdb) print result
//   (gdb) step

#include <cmath>
#include <iostream>

double dsl_buggy(double x) {
    double result = 0.0;  // <-- Set breakpoint here (line ~19)
    
    for (int i = 0; i < 10; i++) {
        double term = std::sin(x + i);  // <-- Step through here
        result += term;
        
        // Debug print for verification
        // std::cout << "  i=" << i << " term=" << term << " result=" << result << std::endl;
    }
    
    return result;  // <-- Should be able to inspect final result
}

// Simple test function
void test_debug() {
    std::cout << "Testing dsl_buggy(1.0)..." << std::endl;
    double r = dsl_buggy(1.0);
    std::cout << "Result: " << r << std::endl;
}
EOF

echo ""
echo "Step 1: Compile with debug symbols (++g)..."
echo ""

# First, just compile to make sure it works
root -l -b -q << ROOTCMD
std::cout << "Compiling ${MACRO_PATH} with debug symbols..." << std::endl;
int result = gROOT->ProcessLine(".L ${MACRO_PATH}++g");
if (result != 0) {
    std::cerr << "❌ Compilation failed!" << std::endl;
    gSystem->Exit(1);
}
std::cout << "✅ Compilation succeeded" << std::endl;
std::cout << std::endl;
std::cout << "Testing function..." << std::endl;
test_debug();
.q
ROOTCMD

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed - cannot proceed with debug test"
    exit 1
fi

echo ""
echo "============================================================"
echo "MANUAL DEBUGGING INSTRUCTIONS"
echo "============================================================"
echo ""
echo "The macro has been compiled with debug symbols."
echo "Now test the debugger workflow manually:"
echo ""
echo "Option 1: GDB (Linux)"
echo "  $ gdb --args root -l"
echo "  (gdb) break dsl_buggy"
echo "  (gdb) run"
echo "  root [0] .L ${MACRO_PATH}++g"
echo "  root [1] dsl_buggy(1.0)"
echo "  (gdb) print x"
echo "  (gdb) print result"
echo "  (gdb) print i"
echo "  (gdb) step"
echo "  (gdb) continue"
echo ""
echo "Option 2: LLDB (macOS)"
echo "  $ lldb -- root -l"
echo "  (lldb) breakpoint set --name dsl_buggy"
echo "  (lldb) run"
echo "  root [0] .L ${MACRO_PATH}++g"
echo "  root [1] dsl_buggy(1.0)"
echo "  (lldb) frame variable"
echo "  (lldb) step"
echo ""
echo "Expected behavior:"
echo "  ✅ Breakpoint hits in dsl_buggy"
echo "  ✅ Can print x (should be 1.0)"
echo "  ✅ Can print result (starts at 0.0)"
echo "  ✅ Can print i (loop counter)"
echo "  ✅ Can step through loop iterations"
echo "  ✅ Line numbers match source file"
echo ""
echo "============================================================"
echo "Files for manual testing:"
echo "  Macro: ${MACRO_PATH}"
echo "  Library: ${MACRO_PATH//.C/_C.so} (or .dylib on macOS)"
echo "============================================================"
echo ""
echo "After manual testing, report results:"
echo "  ✅ T2-Full PASS - if all expected behaviors work"
echo "  ❌ T2-Full FAIL - if debugger cannot inspect variables"
echo "  ⚠️  T2-Full PARTIAL - if some features work"
