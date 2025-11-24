#!/bin/bash
# run_tests.sh - Run all ROOT macro tests for AliasDataFrameTree.C
#
# Usage: ./run_tests.sh [test_name]
#   No args: run all tests
#   test_name: run specific test (composite_index, AliasDataFrameTree)
#
# Exit codes:
#   0 - All tests passed
#   1 - Some tests failed
#   2 - Test file not found

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "AliasDataFrameTree.C Test Suite"
echo "=============================================="
echo ""

FAILED=0
PASSED=0

run_test() {
    local test_name=$1
    local test_file="test_${test_name}.C"
    
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}SKIP${NC}: $test_file not found"
        return 0
    fi
    
    echo -n "Running $test_file... "
    
    # Run test and capture output
    if root.exe -b -q "$test_file" > "${test_name}.log" 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Log: ${test_name}.log"
        ((FAILED++))
        return 1
    fi
}

# Determine which tests to run
if [ -n "$1" ]; then
    # Run specific test
    run_test "$1" || true
else
    # Run all tests
    echo "Running all tests..."
    echo ""
    
    run_test "composite_index" || true
    run_test "AliasDataFrameTree" || true
fi

echo ""
echo "=============================================="
echo "Results: ${PASSED} passed, ${FAILED} failed"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed test logs:"
    for log in *.log; do
        if grep -q "FAILED\|ERROR\|FAIL:" "$log" 2>/dev/null; then
            echo "  - $log"
        fi
    done
    exit 1
else
    # Cleanup logs on success
    rm -f *.log
    echo "All tests passed!"
    exit 0
fi
