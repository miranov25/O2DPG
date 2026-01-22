#!/bin/bash
# run_tests.sh - Unified test runner for RDataFrameDSL
#
# Phase 13.6.E: Two-phase execution for ROOT JIT isolation
# 1. Parallel: Non-ROOT tests (pytest-xdist, -n 12)
# 2. Parallel: ROOT tests (GNU parallel with TMPDIR isolation)
#
# Usage:
#   ./run_tests.sh              # Default: summary output
#   ./run_tests.sh -v           # Verbose: show each test
#   ./run_tests.sh --tb=long    # Long tracebacks
#
# Output logs:
#   test_logs/test_parallel_TIMESTAMP.log
#   test_logs/test_serial_TIMESTAMP.log

set -e

EXTRA_ARGS="$@"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

PARALLEL_LOG="$LOG_DIR/test_parallel_${TIMESTAMP}.log"
SERIAL_LOG="$LOG_DIR/test_serial_${TIMESTAMP}.log"

echo "=== RDataFrameDSL Test Suite ==="
echo "Timestamp: $TIMESTAMP"
echo "Extra args: ${EXTRA_ARGS:-'(none)'}"
echo ""

# Phase 1: Parallel tests (non-ROOT)
echo "=============================================="
echo "Phase 1: Non-ROOT tests (pytest -n 12)"
echo "=============================================="
pytest tests/ -n 12 -m "not root_serial" --tb=short $EXTRA_ARGS 2>&1 | tee "$PARALLEL_LOG"
PARALLEL_EXIT=${PIPESTATUS[0]}

echo ""

# Phase 2: ROOT tests (parallel with TMPDIR isolation)
echo "=============================================="
echo "Phase 2: ROOT tests (GNU parallel, TMPDIR isolated)"
echo "=============================================="

# List of test files containing root_serial tests
# Add new ROOT test files here as needed
ROOT_TEST_FILES=(
    tests/test_invariance_nd.py
    tests/test_invariance_join_e2e.py
    tests/test_invariance_udf.py
    tests/test_carray_correctness.py
    tests/test_carray_root_integration.py
    tests/test_root_broadcast_integration.py
    tests/test_root_integration.py
)

printf '%s\n' "${ROOT_TEST_FILES[@]}" | \
    parallel -j 4 "TMPDIR=\$(mktemp -d) pytest {} -n 0 -m 'root_serial' --tb=short $EXTRA_ARGS" 2>&1 | tee "$SERIAL_LOG"
SERIAL_EXIT=${PIPESTATUS[0]}

# Count results from logs
PARALLEL_PASSED=$(grep -E '[0-9]+ passed' "$PARALLEL_LOG" | grep -oE '[0-9]+(?= passed)' | tail -1 || echo 0)
SERIAL_PASSED=$(grep -E '[0-9]+ passed' "$SERIAL_LOG" | grep -oE '[0-9]+(?= passed)' | awk '{s+=$1} END {print s+0}')
TOTAL_PASSED=$((PARALLEL_PASSED + SERIAL_PASSED))

echo ""
echo "=============================================="
echo "=== Summary ==="
echo "=============================================="
echo "Phase 1 (non-ROOT): $PARALLEL_PASSED passed (exit: $PARALLEL_EXIT)"
echo "Phase 2 (ROOT):     $SERIAL_PASSED passed (exit: $SERIAL_EXIT)"
echo "Total:              $TOTAL_PASSED passed"
echo ""
echo "Logs: $PARALLEL_LOG"
echo "      $SERIAL_LOG"
echo ""

if [ $PARALLEL_EXIT -eq 0 ] && [ $SERIAL_EXIT -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
