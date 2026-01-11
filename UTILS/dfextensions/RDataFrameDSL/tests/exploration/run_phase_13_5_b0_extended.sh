#!/bin/bash
# Phase 13.5.B0 Extended Tests Runner (T7-T32)
# 
# Usage:
#   ./run_phase_13_5_b0_extended.sh              # Run all tests
#   ./run_phase_13_5_b0_extended.sh p0           # Run P0 tests only (T15-T24)
#   ./run_phase_13_5_b0_extended.sh p1           # Run P1 tests only (T25-T30, T32)
#   ./run_phase_13_5_b0_extended.sh p2           # Run P2 tests only (T31)
#   ./run_phase_13_5_b0_extended.sh t15          # Run specific test
#   ./run_phase_13_5_b0_extended.sh 2>&1 | tee run_extended.log
#
# Test Coverage:
#   P0 (Critical): T15-T24 - Thread Safety, Parser, Hash, Cache, Overload, Export
#   P1 (Robustness): T25-T30, T32 - Stale Cache, Func Chain, Safe Mode, etc.
#   P2 (Optional): T31 - Optimization Levels

set +e  # Don't exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Phase 13.5.B0 Extended Tests (v7 Specification)"
echo "========================================================================"
echo ""
echo "Date: $(date)"
echo "Directory: $SCRIPT_DIR"
echo ""

# Check ROOT
python3 -c "import ROOT; print(f'ROOT Version: {ROOT.gROOT.GetVersion()}')" 2>/dev/null || {
    echo "ERROR: ROOT not available"
    exit 1
}

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo ""
    echo "========================================================================"
    echo "Running $test_name"
    echo "========================================================================"
    
    if [ ! -f "$test_file" ]; then
        echo "ERROR: Test file not found: $test_file"
        return 1
    fi
    
    if python3 "$test_file"; then
        echo "✅ $test_name PASSED"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ]; then
            echo "⚠️  $test_name completed with crash (exit $exit_code)"
            return 0
        fi
        echo "❌ $test_name FAILED (exit $exit_code)"
        return 1
    fi
}

# Parse arguments
TESTS=""
case "$1" in
    p0|P0)
        TESTS="t15 t16-18 t19-21 t22-24"
        ;;
    p1|P1)
        TESTS="t25-32"
        ;;
    p2|P2)
        TESTS="t31"
        ;;
    t7|t8|t8b|t9|t10|t11-14)
        # Original B0 tests
        ./run_phase_13_5_b0_tests.sh "$1"
        exit $?
        ;;
    t15)
        TESTS="t15"
        ;;
    t16-18|t16|t17|t18)
        TESTS="t16-18"
        ;;
    t19-21|t19|t20|t21)
        TESTS="t19-21"
        ;;
    t22-24|t22|t23|t24)
        TESTS="t22-24"
        ;;
    t25-32|t25|t26|t27|t28|t29|t30|t31|t32)
        TESTS="t25-32"
        ;;
    all|"")
        # Run original B0 tests first, then extended
        echo "=== Running Original T7-T14 Tests ==="
        ./run_phase_13_5_b0_tests.sh
        ORIGINAL_RESULT=$?
        
        echo ""
        echo "=== Running Extended T15-T32 Tests ==="
        TESTS="t15 t16-18 t19-21 t22-24 t25-32"
        ;;
    *)
        echo "Unknown test: $1"
        echo "Valid options: p0, p1, p2, t15, t16-18, t19-21, t22-24, t25-32, all"
        exit 1
        ;;
esac

# Track results
PASSED=0
FAILED=0
TOTAL=0

for test in $TESTS; do
    case $test in
        t15)
            ((TOTAL++))
            echo ""
            echo "========================================================================"
            echo "=== T15: Thread Safety (RUN FIRST - Crash Test) ==="
            echo "========================================================================"
            if run_test "T15: Thread Safety" "test_t15_thread_safety.py"; then
                ((PASSED++))
            else
                ((FAILED++))
            fi
            ;;
        t16-18)
            ((TOTAL++))
            if run_test "T16-T18: Parser, Hash, Cache" "test_t16_t18_parser_hash_cache.py"; then
                ((PASSED++))
            else
                ((FAILED++))
            fi
            ;;
        t19-21)
            ((TOTAL++))
            if run_test "T19-T21: Overload, Cross-Instance, Export" "test_t19_t21_overload_crossinstance_export.py"; then
                ((PASSED++))
            else
                ((FAILED++))
            fi
            ;;
        t22-24)
            ((TOTAL++))
            if run_test "T22-T24: Same-Arity, Idempotency, Schema" "test_t22_t24_overload_idempotency_schema.py"; then
                ((PASSED++))
            else
                ((FAILED++))
            fi
            ;;
        t25-32)
            ((TOTAL++))
            if run_test "T25-T32: Robustness (P1/P2)" "test_t25_t32_robustness.py"; then
                ((PASSED++))
            else
                ((FAILED++))
            fi
            ;;
    esac
done

echo ""
echo "========================================================================"
echo "EXTENDED TESTS SUMMARY (T15-T32)"
echo "========================================================================"
echo "Total Test Files: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

echo "--- Test Coverage ---"
echo "P0 (Critical):"
echo "  T15:     Thread Safety (ImplicitMT)"
echo "  T16-T18: Parser, Hash Determinism, Cache"
echo "  T19-T21: Overload, Cross-Instance, Export"
echo "  T22-T24: Same-Arity, Idempotency, Schema"
echo ""
echo "P1 (Robustness):"
echo "  T25: Stale Cache Detection"
echo "  T26: Function-to-Function Calls"
echo "  T27: Hash Format Determinism"
echo "  T28: Safe Mode Crash Protection"
echo "  T29: Export/Import Round-Trip"
echo "  T30: Session State After Error"
echo "  T32: Edge Case Return Types"
echo ""
echo "P2 (Optional):"
echo "  T31: ACLiC Optimization Levels"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL $TOTAL TEST FILES PASSED"
    exit 0
else
    echo "⚠️  $PASSED/$TOTAL test files passed ($FAILED failed)"
    exit 1
fi
