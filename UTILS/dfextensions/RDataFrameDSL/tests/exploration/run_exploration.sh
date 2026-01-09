#!/bin/bash
# Phase 13.5.A - Exploration Test Runner
#
# Runs all T1-T6 exploration tests in Main Architect priority order:
#   T1 (ACLiC) → T3 (Pure ROOT) → T2 (Debug) → T5 (Benchmark)
#   T4 (Headers) and T6 (Perf) are supplementary
#
# Usage: ./run_exploration.sh [--all | --priority | --quick]
#   --all      Run all tests T1-T6
#   --priority Run priority tests only (T1, T3, T2, T5)
#   --quick    Run quick tests only (T1, T4)
#
# Per Phase 13.5 v0.3 specification.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_FILE="${SCRIPT_DIR}/exploration_report.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
MODE="${1:---priority}"

echo ""
echo "============================================================"
echo " Phase 13.5.A - Exploration Test Suite"
echo " Mode: ${MODE}"
echo "============================================================"
echo ""

# Initialize report
cat > "${REPORT_FILE}" << 'EOF'
# Phase 13.5.A - Exploration Test Report

**Generated:** $(date -Iseconds)
**Status:** IN_PROGRESS

## Test Results

| Test | Status | Notes |
|------|--------|-------|
EOF

# Function to run a test and capture result
run_test() {
    local TEST_NAME=$1
    local TEST_CMD=$2
    local TEST_FILE=$3
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running: ${TEST_NAME}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    local START_TIME=$(date +%s)
    local EXIT_CODE=0
    
    if [ -f "${TEST_FILE}" ]; then
        if [[ "${TEST_FILE}" == *.py ]]; then
            python3 "${TEST_FILE}" || EXIT_CODE=$?
        elif [[ "${TEST_FILE}" == *.sh ]]; then
            bash "${TEST_FILE}" || EXIT_CODE=$?
        fi
    else
        echo -e "${YELLOW}⚠️  Test file not found: ${TEST_FILE}${NC}"
        EXIT_CODE=2
    fi
    
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ ${TEST_NAME}: PASS (${DURATION}s)${NC}"
        echo "| ${TEST_NAME} | ✅ PASS | ${DURATION}s |" >> "${REPORT_FILE}"
    elif [ $EXIT_CODE -eq 2 ]; then
        echo -e "${YELLOW}⚠️  ${TEST_NAME}: SKIP (file not found)${NC}"
        echo "| ${TEST_NAME} | ⚠️ SKIP | File not found |" >> "${REPORT_FILE}"
    else
        echo -e "${RED}❌ ${TEST_NAME}: FAIL (exit code: ${EXIT_CODE})${NC}"
        echo "| ${TEST_NAME} | ❌ FAIL | Exit code: ${EXIT_CODE} |" >> "${REPORT_FILE}"
    fi
    echo ""
    
    return $EXIT_CODE
}

# Track overall status
PASSED=0
FAILED=0
SKIPPED=0

# Run tests based on mode
case "${MODE}" in
    --quick)
        echo "Running quick tests (T1, T4)..."
        echo ""
        
        run_test "T1: ACLiC Basic" "python3" "${SCRIPT_DIR}/test_t1_aclic_basic.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T4: Header Detection" "python3" "${SCRIPT_DIR}/test_t4_header_detection.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        ;;
        
    --priority)
        echo "Running priority tests (T1, T3, T2, T5)..."
        echo ""
        
        run_test "T1: ACLiC Basic" "python3" "${SCRIPT_DIR}/test_t1_aclic_basic.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T3: Standalone ROOT" "bash" "${SCRIPT_DIR}/test_t3_standalone.sh" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T2: Debug Symbols" "python3" "${SCRIPT_DIR}/test_t2_debug_symbols.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T5: Multicore JIT" "python3" "${SCRIPT_DIR}/test_t5_multicore_jit.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        ;;
        
    --all)
        echo "Running all tests (T1-T6)..."
        echo ""
        
        run_test "T1: ACLiC Basic" "python3" "${SCRIPT_DIR}/test_t1_aclic_basic.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T2: Debug Symbols" "python3" "${SCRIPT_DIR}/test_t2_debug_symbols.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T3: Standalone ROOT" "bash" "${SCRIPT_DIR}/test_t3_standalone.sh" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T4: Header Detection" "python3" "${SCRIPT_DIR}/test_t4_header_detection.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T5: Multicore JIT" "python3" "${SCRIPT_DIR}/test_t5_multicore_jit.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        run_test "T6: Performance" "python3" "${SCRIPT_DIR}/test_t6_performance.py" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
        ;;
        
    *)
        echo "Usage: $0 [--all | --priority | --quick]"
        echo "  --all      Run all tests T1-T6"
        echo "  --priority Run priority tests only (T1, T3, T2, T5) [default]"
        echo "  --quick    Run quick tests only (T1, T4)"
        exit 1
        ;;
esac

# Generate summary
echo ""
echo "============================================================"
echo " EXPLORATION TEST SUMMARY"
echo "============================================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASSED}"
echo -e "  ${RED}Failed:${NC}  ${FAILED}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED}"
echo ""

# Finalize report
cat >> "${REPORT_FILE}" << EOF

## Summary

- **Passed:** ${PASSED}
- **Failed:** ${FAILED}
- **Skipped:** ${SKIPPED}

## Next Steps

EOF

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo "1. ✅ All tests passed - proceed to Phase 13.5.B" >> "${REPORT_FILE}"
else
    echo -e "${RED}❌ Some tests failed - review results above${NC}"
    echo "1. ❌ Some tests failed - review and fix before proceeding" >> "${REPORT_FILE}"
fi

echo ""
echo "Report saved to: ${REPORT_FILE}"
echo ""

# Manual testing reminder
if [[ "${MODE}" == "--priority" || "${MODE}" == "--all" ]]; then
    echo "============================================================"
    echo " MANUAL TESTING REQUIRED"
    echo "============================================================"
    echo ""
    echo "T2-Full (Interactive Debugging) requires manual validation:"
    echo "  Run: ${SCRIPT_DIR}/test_t2_debug_interactive.sh"
    echo ""
    echo "Then test with GDB/LLDB as instructed."
    echo ""
fi

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
