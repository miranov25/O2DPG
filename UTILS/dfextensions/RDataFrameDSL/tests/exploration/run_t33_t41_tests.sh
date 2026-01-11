#!/bin/bash
# =============================================================================
# Phase 13.5.B0 Extended Tests Runner (T33-T41)
# Authorization: [EXECUTE-T33-T41] [FULL-COVERAGE] [NO-RETURN-TO-TESTS]
# =============================================================================

set -e

echo "========================================================================"
echo "Phase 13.5.B0 Extended Tests (T33-T41)"
echo "========================================================================"
echo ""
echo "Date: $(date)"
echo "Directory: $(pwd)"
echo ""

# Check ROOT availability
if command -v root-config &> /dev/null; then
    echo "ROOT Version: $(root-config --version)"
else
    echo "WARNING: ROOT not found in PATH"
fi

echo ""
echo "========================================================================"

# Parse arguments
RUN_MODE="${1:-all}"

case "$RUN_MODE" in
    all|t33-t41)
        echo "Running ALL T33-T41 tests..."
        python3 test_t33_t41_extended.py
        ;;
    t33)
        echo "Running T33 only..."
        python3 -c "
from test_t33_t41_extended import test_t33_header_detection
result = test_t33_header_detection()
print(result.to_markdown())
"
        ;;
    t34)
        echo "Running T34 only..."
        python3 -c "
from test_t33_t41_extended import test_t34_snapshot_matrix
result = test_t34_snapshot_matrix()
print(result.to_markdown())
"
        ;;
    t35)
        echo "Running T35 only..."
        python3 -c "
from test_t33_t41_extended import test_t35_namespace_isolation
result = test_t35_namespace_isolation()
print(result.to_markdown())
"
        ;;
    t36)
        echo "Running T36 only..."
        python3 -c "
from test_t33_t41_extended import test_t36_parallel_compile
result = test_t36_parallel_compile()
print(result.to_markdown())
"
        ;;
    t37)
        echo "Running T37 only..."
        python3 -c "
from test_t33_t41_extended import test_t37_registry_persistence
result = test_t37_registry_persistence()
print(result.to_markdown())
"
        ;;
    t38)
        echo "Running T38 only..."
        python3 -c "
from test_t33_t41_extended import test_t38_complex_rvec
result = test_t38_complex_rvec()
print(result.to_markdown())
"
        ;;
    t39)
        echo "Running T39 only..."
        python3 -c "
from test_t33_t41_extended import test_t39_version_matrix
result = test_t39_version_matrix()
print(result.to_markdown())
"
        ;;
    t41)
        echo "Running T41 only..."
        python3 -c "
from test_t33_t41_extended import test_t41_lambda_rejection
result = test_t41_lambda_rejection()
print(result.to_markdown())
"
        ;;
    p0)
        echo "Running P0 tests only (T33, T36, T37, T38)..."
        python3 -c "
from test_t33_t41_extended import (
    test_t33_header_detection,
    test_t36_parallel_compile,
    test_t37_registry_persistence,
    test_t38_complex_rvec,
)
results = []
for name, test in [
    ('T33', test_t33_header_detection),
    ('T36', test_t36_parallel_compile),
    ('T37', test_t37_registry_persistence),
    ('T38', test_t38_complex_rvec),
]:
    print(f'\\n=== Running {name} ===')
    result = test()
    results.append(result)
    print(result.to_markdown())

passed = sum(1 for r in results if r.status == 'PASSED')
print(f'\\n{passed}/{len(results)} P0 tests passed')
"
        ;;
    *)
        echo "Usage: $0 [all|p0|t33|t34|t35|t36|t37|t38|t39|t41]"
        echo ""
        echo "Options:"
        echo "  all     Run all T33-T41 tests (default)"
        echo "  p0      Run P0 tests only (T33, T36, T37, T38)"
        echo "  t33     Header Auto-Detection"
        echo "  t34     Snapshot/I/O Contract"
        echo "  t35     Namespace Isolation"
        echo "  t36     Parallel Compilation"
        echo "  t37     Registry Persistence"
        echo "  t38     Complex RVec Types"
        echo "  t39     Version Matrix"
        echo "  t41     Lambda Rejection"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Test run complete"
echo "========================================================================"
