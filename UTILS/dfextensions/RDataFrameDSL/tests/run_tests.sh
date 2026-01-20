#!/bin/bash
# run_tests.sh - Unified test runner for RDataFrameDSL
#
# Handles ROOT JIT isolation by running tests in two phases:
# 1. Parallel: Non-ROOT tests (fast)
# 2. Serial: ROOT tests (requires JIT isolation)

set -e

echo "=== RDataFrameDSL Test Suite ==="
echo ""

# Phase 1: Parallel tests (non-ROOT)
echo "Phase 1: Running parallel tests (non-ROOT)..."
pytest tests/ -n auto -m "not root_serial" -v
PARALLEL_EXIT=$?

echo ""

# Phase 2: Serial ROOT tests
echo "Phase 2: Running serial ROOT tests..."
pytest tests/ -m "root_serial" -n 0 -v
SERIAL_EXIT=$?

echo ""
echo "=== Summary ==="
if [ $PARALLEL_EXIT -eq 0 ] && [ $SERIAL_EXIT -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Some tests failed"
    echo "   Parallel exit code: $PARALLEL_EXIT"
    echo "   Serial exit code: $SERIAL_EXIT"
    exit 1
fi
