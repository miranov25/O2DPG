#!/bin/bash
# run_tests.sh - Unified test runner for RDataFrameDSL
#
# Phase 13.6.E: Two-phase execution for ROOT JIT isolation
# 0. Pre-generate ROOT dictionaries (avoids parallel compilation race)
# 1. Parallel: Non-ROOT tests (pytest-xdist, -n 12)
# 2. Serial: ROOT tests (GNU parallel with TMPDIR isolation)
#
# NOTE: Exploration tests are excluded - run via tests/exploration/run_exploration.sh
#
# Usage:
#   ./run_tests.sh              # Default: summary output
#   ./run_tests.sh -v           # Verbose: show each test
#   ./run_tests.sh --tb=long    # Long tracebacks
#
# Output logs:
#   test_logs/test_parallel_TIMESTAMP.log
#   test_logs/test_serial_TIMESTAMP.log
#   test_logs/test_crashes_TIMESTAMP.log (if crashes detected)

set -e

EXTRA_ARGS="$@"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

PARALLEL_LOG="$LOG_DIR/test_parallel_${TIMESTAMP}.log"
SERIAL_LOG="$LOG_DIR/test_serial_${TIMESTAMP}.log"
CRASH_LOG="$LOG_DIR/test_crashes_${TIMESTAMP}.log"

echo "=== RDataFrameDSL Test Suite ==="
echo "Timestamp: $TIMESTAMP"
echo "Extra args: ${EXTRA_ARGS:-'(none)'}"
echo ""

# Phase 0: Pre-generate ROOT dictionaries
# This avoids race conditions when multiple parallel processes try to generate
# the same AutoDict_* files simultaneously
echo "=============================================="
echo "Phase 0: Pre-generating ROOT dictionaries"
echo "=============================================="
python -c "
import sys
sys.path.insert(0, 'tests')
sys.path.insert(0, 'tests/generators')
try:
    from toy_nd import _ensure_rvec_dictionaries
    _ensure_rvec_dictionaries()
    print('✅ RVec dictionaries ready')
except Exception as e:
    print(f'⚠️  Dictionary generation warning: {e}')
    print('   Tests will generate dictionaries on demand')
"
echo ""

# Phase 1: Parallel tests (non-ROOT)
# Note: exploration/ excluded via pytest.ini norecursedirs
echo "=============================================="
echo "Phase 1: Non-ROOT tests (pytest -n 12)"
echo "=============================================="
PARALLEL_JSON="$LOG_DIR/test_parallel_${TIMESTAMP}.json"
pytest tests/ -n 12 -m "not root_serial" --ignore=tests/exploration --tb=short \
    --json-report --json-report-file="$PARALLEL_JSON" \
    $EXTRA_ARGS 2>&1 | tee "$PARALLEL_LOG"
PARALLEL_EXIT=${PIPESTATUS[0]}

echo ""

# Phase 2: ROOT tests (serial with TMPDIR isolation)
# Exploration tests excluded - run via tests/exploration/run_exploration.sh
echo "=============================================="
echo "Phase 2: ROOT tests (GNU parallel, TMPDIR isolated)"
echo "=============================================="

# List of test files containing root_serial tests
# Exploration tests excluded - they have their own runner
ROOT_TEST_FILES=(
    tests/test_api_alias.py
    tests/test_api_define.py
    tests/test_api_draw.py
    tests/test_api_to_pandas.py
    tests/test_carray_correctness.py
    tests/test_carray_root_integration.py
    tests/test_helix_generator.py
    tests/test_invariance_join_e2e.py
    tests/test_invariance_nd.py
    tests/test_invariance_safe_draw.py
    tests/test_invariance_udf.py
    tests/test_nested_slicing.py
    tests/test_root_broadcast_integration.py
    tests/test_root_integration.py
    tests/test_rvec_selection.py
    tests/test_safe_mode.py
)

# Create directory for serial JSON reports
SERIAL_JSON_DIR="$LOG_DIR/serial_json_${TIMESTAMP}"
mkdir -p "$SERIAL_JSON_DIR"

printf '%s\n' "${ROOT_TEST_FILES[@]}" | \
    parallel -j 4 "TMPDIR=\$(mktemp -d) pytest {} -n 0 -m 'root_serial' --tb=short \
        --json-report --json-report-file='$SERIAL_JSON_DIR/{/.}.json' \
        $EXTRA_ARGS" 2>&1 | tee "$SERIAL_LOG"
SERIAL_EXIT=${PIPESTATUS[0]}

# Count results from logs (macOS compatible - no -P flag)
PARALLEL_PASSED=$(grep -oE '[0-9]+ passed' "$PARALLEL_LOG" | tail -1 | grep -oE '^[0-9]+' || echo "0")
SERIAL_PASSED=$(grep -oE '[0-9]+ passed' "$SERIAL_LOG" | grep -oE '^[0-9]+' | awk '{s+=$1} END {print s+0}')
TOTAL_PASSED=$((PARALLEL_PASSED + SERIAL_PASSED))

# Detect crashes in BOTH parallel and serial logs
CRASH_COUNT_PARALLEL=$(grep -c "Fatal Python error" "$PARALLEL_LOG" 2>/dev/null || echo "0")
CRASH_COUNT_PARALLEL="${CRASH_COUNT_PARALLEL%%[^0-9]*}"
CRASH_COUNT_SERIAL=$(grep -c "Fatal Python error" "$SERIAL_LOG" 2>/dev/null || echo "0")
CRASH_COUNT_SERIAL="${CRASH_COUNT_SERIAL%%[^0-9]*}"
CRASH_COUNT=$((CRASH_COUNT_PARALLEL + CRASH_COUNT_SERIAL))

# Extract crash locations if any
if [ "$CRASH_COUNT" -gt 0 ]; then
    echo ""
    echo "=============================================="
    echo "Extracting crash locations..."
    echo "=============================================="
    
    # Extract crash locations from BOTH logs
    CRASH_LOCATIONS_PARALLEL=""
    CRASH_LOCATIONS_SERIAL=""
    
    if [ "$CRASH_COUNT_PARALLEL" -gt 0 ]; then
        CRASH_LOCATIONS_PARALLEL=$(grep -A 30 "Fatal Python error" "$PARALLEL_LOG" | \
            grep -E "tests/.*\.py.*line [0-9]+ in " | \
            sed 's/.*File "\([^"]*\)", line \([0-9]*\) in \(.*\)/\1:\2  \3/' | \
            sort -u)
    fi
    
    if [ "$CRASH_COUNT_SERIAL" -gt 0 ]; then
        CRASH_LOCATIONS_SERIAL=$(grep -A 30 "Fatal Python error" "$SERIAL_LOG" | \
            grep -E "tests/.*\.py.*line [0-9]+ in " | \
            sed 's/.*File "\([^"]*\)", line \([0-9]*\) in \(.*\)/\1:\2  \3/' | \
            sort -u)
    fi
    
    # Combine and dedupe
    CRASH_LOCATIONS=$(echo -e "${CRASH_LOCATIONS_PARALLEL}\n${CRASH_LOCATIONS_SERIAL}" | grep -v '^$' | sort -u)
    
    # Create crash report
    {
        echo "# Crash Report - $TIMESTAMP"
        echo "# Detected $CRASH_COUNT total crashes ($CRASH_COUNT_PARALLEL parallel, $CRASH_COUNT_SERIAL serial)"
        echo "#"
        echo "# Root Cause: ROOT gInterpreter is not thread-safe."
        echo "# Parallel crashes: Multiple workers call ROOT simultaneously."
        echo "# Serial crashes: Test or teardown triggers ROOT interpreter issue."
        echo "#"
        echo "# Fix for parallel: Mark tests with @pytest.mark.root_serial"
        echo "# Fix for serial: Investigate specific test - may need xfail or code fix"
        echo "#"
        echo "# Crash Locations:"
        echo "# ================"
        if [ -n "$CRASH_LOCATIONS_PARALLEL" ]; then
            echo "# From parallel tests:"
            echo "$CRASH_LOCATIONS_PARALLEL"
        fi
        if [ -n "$CRASH_LOCATIONS_SERIAL" ]; then
            echo "# From serial tests:"
            echo "$CRASH_LOCATIONS_SERIAL"
        fi
        echo ""
        echo "# Unique files with crashes:"
        echo "# =========================="
        echo "$CRASH_LOCATIONS" | sed 's/:.*//g' | sort -u
    } > "$CRASH_LOG"
    
    # Display crash summary
    echo ""
    echo "Crash locations found:"
    if [ -n "$CRASH_LOCATIONS_PARALLEL" ]; then
        echo "  [PARALLEL]:"
        echo "$CRASH_LOCATIONS_PARALLEL" | sed 's/^/    /'
    fi
    if [ -n "$CRASH_LOCATIONS_SERIAL" ]; then
        echo "  [SERIAL]:"
        echo "$CRASH_LOCATIONS_SERIAL" | sed 's/^/    /'
    fi
    echo ""
    echo "Full crash report: $CRASH_LOG"
fi

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

# Phase 3: Generate Capability Matrix from test results
echo ""
echo "=============================================="
echo "Phase 3: Generating Capability Matrix"
echo "=============================================="
if [ -f "scripts/generate_capability_matrix.py" ]; then
    python scripts/generate_capability_matrix.py \
        --from-reports "$PARALLEL_JSON" "$SERIAL_JSON_DIR"/*.json \
        --output docs/CAPABILITY_MATRIX.md
    if [ $? -eq 0 ]; then
        echo "✅ Capability matrix updated: docs/CAPABILITY_MATRIX.md"
    else
        echo "⚠️  Capability matrix generation failed (non-fatal)"
    fi
else
    echo "⚠️  scripts/generate_capability_matrix.py not found, skipping matrix generation"
fi

if [ "$CRASH_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: $CRASH_COUNT crashes detected ($CRASH_COUNT_PARALLEL parallel, $CRASH_COUNT_SERIAL serial)"
    echo "   Crash report: $CRASH_LOG"
    if [ "$CRASH_COUNT_PARALLEL" -gt 0 ]; then
        echo "   Parallel fix: Add 'pytestmark = pytest.mark.root_serial' to affected files"
    fi
    if [ "$CRASH_COUNT_SERIAL" -gt 0 ]; then
        echo "   Serial fix: Investigate test - may need xfail or ROOT cleanup fix"
    fi
fi

# Phase 4: Check for orphaned root_serial tests
echo ""
echo "=============================================="
echo "Phase 4: Checking for orphaned root_serial tests"
echo "=============================================="
ORPHANED_COUNT=0
for f in tests/test_*.py; do
    if grep -q "root_serial" "$f" 2>/dev/null; then
        # Check if file is in ROOT_TEST_FILES
        FOUND=0
        for root_file in "${ROOT_TEST_FILES[@]}"; do
            if [ "$root_file" = "$f" ]; then
                FOUND=1
                break
            fi
        done
        if [ $FOUND -eq 0 ]; then
            echo "⚠️  ORPHANED: $f has root_serial tests but is not in ROOT_TEST_FILES!"
            ORPHANED_COUNT=$((ORPHANED_COUNT + 1))
        fi
    fi
done
if [ $ORPHANED_COUNT -eq 0 ]; then
    echo "✅ No orphaned root_serial tests found"
else
    echo ""
    echo "⚠️  Found $ORPHANED_COUNT orphaned test file(s)"
    echo "   Add them to ROOT_TEST_FILES in run_tests.sh"
fi

# Note: AutoDict_* files are intentionally NOT cleaned up.
# They are cached ROOT dictionaries needed for subsequent runs.
# Cleaning them causes race conditions in parallel test execution.

echo ""
echo "NOTE: Exploration tests excluded. Run separately:"
echo "      ./tests/exploration/run_exploration.sh"
echo ""

if [ $PARALLEL_EXIT -eq 0 ] && [ $SERIAL_EXIT -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
