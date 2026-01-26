#!/bin/bash
# =============================================================================
# run_tests.sh — RDataFrameDSL Test Runner
# =============================================================================
#
# Location: tests/run_tests.sh (run from project root or tests/ directory)
#
# Phase 13.6.G: Improved failure reporting
#   - Shows FAILED count prominently (not just passed)
#   - Lists failed test names explicitly  
#   - Saves summary to test_logs/SUMMARY_${TIMESTAMP}.txt
#   - Failure summary BEFORE capability matrix
#   - Color output for failures
#
# Fixes in this version:
#   - Priority 1: Use --from-reports instead of --json-dir for capability matrix
#   - Priority 2: Remove -m root_serial from serial phase (run ALL tests in file)
#
# Usage:
#   ./tests/run_tests.sh          # From project root
#   ./run_tests.sh                # From tests/ directory
#   ./tests/run_tests.sh --quick  # Skip capability matrix
#
# =============================================================================

set -e

# =============================================================================
# Path Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -d "$SCRIPT_DIR/tests" ]]; then
    PROJECT_ROOT="$SCRIPT_DIR"
    TESTS_DIR="$SCRIPT_DIR/tests"
elif [[ "$(basename "$SCRIPT_DIR")" == "tests" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    TESTS_DIR="$SCRIPT_DIR"
else
    echo "ERROR: Cannot determine project structure"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "Tests dir: $TESTS_DIR"

# =============================================================================
# Configuration
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

# Colors
if [[ -t 1 ]] && command -v tput &>/dev/null; then
    RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
    BOLD=$(tput bold); RESET=$(tput sgr0)
else
    RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
fi

SUMMARY_FILE="$LOG_DIR/SUMMARY_${TIMESTAMP}.txt"
PARALLEL_JSON_DIR="$LOG_DIR/parallel_json_${TIMESTAMP}"
SERIAL_JSON_DIR="$LOG_DIR/serial_json_${TIMESTAMP}"
mkdir -p "$PARALLEL_JSON_DIR" "$SERIAL_JSON_DIR"

PARALLEL_LOG="$LOG_DIR/test_parallel_${TIMESTAMP}.log"
SERIAL_LOG="$LOG_DIR/test_serial_${TIMESTAMP}.log"
CRASH_LOG="$LOG_DIR/test_crashes_${TIMESTAMP}.log"

PARALLEL_PASSED=0; PARALLEL_FAILED=0; PARALLEL_SKIPPED=0
SERIAL_PASSED=0; SERIAL_FAILED=0; SERIAL_SKIPPED=0
PARALLEL_EXIT=0; SERIAL_EXIT=0

FAILED_TESTS_FILE=$(mktemp)
trap "rm -f $FAILED_TESTS_FILE" EXIT

# =============================================================================
# ROOT Test Files (run serially)
# =============================================================================

ROOT_TEST_FILES=(
    "tests/test_api_define.py"
    "tests/test_api_alias.py"
    "tests/test_api_to_pandas.py"
    "tests/test_api_draw.py"
    "tests/test_draw_integration.py"
    "tests/test_redefinition_policy.py"
    "tests/test_carray_root_integration.py"
    "tests/test_carray_correctness.py"
    "tests/test_invariance_join_e2e.py"
    "tests/test_helix_generator.py"
    "tests/test_invariance_safe_draw.py"
    "tests/test_nested_slicing.py"
    "tests/test_invariance_udf.py"
    "tests/test_invariance_nd.py"
    "tests/test_root_broadcast_integration.py"
    "tests/test_root_integration.py"
    "tests/test_rvec_selection.py"
    "tests/test_safe_mode.py"
    "tests/test_dsl_api.py"
)

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

aggregate_json_reports() {
    local json_dir="$1"
    local failed_file="$2"
    
    python3 << PYEOF
import json, os, sys
json_dir, failed_file = '$json_dir', '$failed_file'
total_passed = total_failed = total_skipped = 0
failed_tests = []

if not os.path.exists(json_dir):
    print("0 0 0"); sys.exit(0)

for fn in sorted(os.listdir(json_dir)):
    if not fn.endswith('.json'): continue
    try:
        with open(os.path.join(json_dir, fn)) as f:
            data = json.load(f)
        s = data.get('summary', {})
        total_passed += s.get('passed', 0)
        total_failed += s.get('failed', 0)
        total_skipped += s.get('skipped', 0)
        for t in data.get('tests', []):
            if t.get('outcome') == 'failed':
                failed_tests.append(t.get('nodeid', 'unknown'))
    except Exception as e:
        print(f"# Warning: {e}", file=sys.stderr)

print(f"{total_passed} {total_failed} {total_skipped}")
if failed_tests:
    with open(failed_file, 'a') as f:
        for t in failed_tests: f.write(t + '\n')
PYEOF
}

# =============================================================================
# Phase 0: Pre-generate ROOT dictionaries
# =============================================================================

print_header "Phase 0: Pre-generating ROOT dictionaries"
python3 -c "
import sys; sys.path.insert(0, 'tests/generators'); sys.path.insert(0, 'tests')
try:
    from generators.toy_nd import _ensure_rvec_dictionaries
    _ensure_rvec_dictionaries(); print('✅ ROOT dictionaries pre-generated')
except Exception as e: print(f'⚠️  {e}')
" 2>/dev/null || echo "⚠️  Dictionary pre-generation skipped"

# =============================================================================
# Phase 1: Parallel Tests (non-ROOT)
# =============================================================================

print_header "Phase 1: Running parallel tests (non-ROOT)"

EXCLUDE_PATTERN=""
for f in "${ROOT_TEST_FILES[@]}"; do
    EXCLUDE_PATTERN="$EXCLUDE_PATTERN --ignore=tests/$(basename "$f")"
done

[[ ! -d "tests" ]] && echo "ERROR: tests/ not found" && exit 1

set +e
python3 -m pytest tests/ -v -m "not root_serial" $EXCLUDE_PATTERN \
    --tb=short --json-report \
    --json-report-file="$PARALLEL_JSON_DIR/test_parallel_${TIMESTAMP}.json" \
    2>&1 | tee "$PARALLEL_LOG"
PARALLEL_EXIT=${PIPESTATUS[0]}
set -e

read PARALLEL_PASSED PARALLEL_FAILED PARALLEL_SKIPPED < <(aggregate_json_reports "$PARALLEL_JSON_DIR" "$FAILED_TESTS_FILE")
echo ""; echo "Phase 1 complete: ${PARALLEL_PASSED} passed, ${PARALLEL_FAILED} failed, ${PARALLEL_SKIPPED} skipped"

# =============================================================================
# Phase 2: Serial Tests (ROOT) - NO MARKER FILTER
# =============================================================================

print_header "Phase 2: Running serial tests (ROOT)"
> "$CRASH_LOG"

for test_file in "${ROOT_TEST_FILES[@]}"; do
    [[ ! -f "$test_file" ]] && echo "⚠️  Skipping missing: $test_file" && continue
    
    basename_f=$(basename "$test_file" .py)
    json_file="$SERIAL_JSON_DIR/${basename_f}.json"
    echo "Running: $test_file"
    
    set +e
    # NO -m filter: run ALL tests in these files serially
    python3 -m pytest "$test_file" -v --tb=short \
        --json-report --json-report-file="$json_file" \
        2>&1 | tee -a "$SERIAL_LOG"
    exit_code=${PIPESTATUS[0]}
    set -e
    
    [[ $exit_code -gt 1 ]] && echo "⚠️  Crash in $test_file (exit: $exit_code)" >> "$CRASH_LOG"
done

read SERIAL_PASSED SERIAL_FAILED SERIAL_SKIPPED < <(aggregate_json_reports "$SERIAL_JSON_DIR" "$FAILED_TESTS_FILE")
[[ $SERIAL_FAILED -gt 0 ]] && SERIAL_EXIT=1 || SERIAL_EXIT=0
echo ""; echo "Phase 2 complete: ${SERIAL_PASSED} passed, ${SERIAL_FAILED} failed, ${SERIAL_SKIPPED} skipped"

# =============================================================================
# FAILURE SUMMARY
# =============================================================================

TOTAL_PASSED=$((PARALLEL_PASSED + SERIAL_PASSED))
TOTAL_FAILED=$((PARALLEL_FAILED + SERIAL_FAILED))
TOTAL_SKIPPED=$((PARALLEL_SKIPPED + SERIAL_SKIPPED))

FAILED_TESTS=()
[[ -f "$FAILED_TESTS_FILE" && -s "$FAILED_TESTS_FILE" ]] && \
    while IFS= read -r line; do [[ -n "$line" ]] && FAILED_TESTS+=("$line"); done < "$FAILED_TESTS_FILE"

cat > "$SUMMARY_FILE" << EOF
================================================================================
RDataFrameDSL Test Summary
================================================================================
Date: $(date '+%Y-%m-%d %H:%M:%S')
Timestamp: ${TIMESTAMP}

RESULTS:
  PASSED:  ${TOTAL_PASSED}
  FAILED:  ${TOTAL_FAILED}
  SKIPPED: ${TOTAL_SKIPPED}

Phase 1 (Parallel/non-ROOT): ${PARALLEL_PASSED} passed, ${PARALLEL_FAILED} failed
Phase 2 (Serial/ROOT):       ${SERIAL_PASSED} passed, ${SERIAL_FAILED} failed
EOF

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo "" >> "$SUMMARY_FILE"
    echo "FAILED TESTS:" >> "$SUMMARY_FILE"
    for t in "${FAILED_TESTS[@]}"; do echo "  - $t" >> "$SUMMARY_FILE"; done
fi
echo "" >> "$SUMMARY_FILE"
echo "Logs:" >> "$SUMMARY_FILE"
echo "  Parallel: $PARALLEL_LOG" >> "$SUMMARY_FILE"
echo "  Serial:   $SERIAL_LOG" >> "$SUMMARY_FILE"
echo "  Summary:  $SUMMARY_FILE" >> "$SUMMARY_FILE"

print_header "TEST RESULTS SUMMARY"

if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo ""; echo "${RED}${BOLD}❌ FAILURES DETECTED${RESET}"; echo ""
    echo "  ${BOLD}PASSED:${RESET}  ${GREEN}${TOTAL_PASSED}${RESET}"
    echo "  ${BOLD}FAILED:${RESET}  ${RED}${TOTAL_FAILED}${RESET}"
    echo "  ${BOLD}SKIPPED:${RESET} ${YELLOW}${TOTAL_SKIPPED}${RESET}"; echo ""
    if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
        echo "${RED}${BOLD}FAILED TESTS:${RESET}"
        for t in "${FAILED_TESTS[@]}"; do echo "  ${RED}- $t${RESET}"; done
        echo ""
    fi
else
    echo ""; echo "${GREEN}${BOLD}✅ ALL TESTS PASSED${RESET}"; echo ""
    echo "  ${BOLD}PASSED:${RESET}  ${GREEN}${TOTAL_PASSED}${RESET}"
    echo "  ${BOLD}SKIPPED:${RESET} ${YELLOW}${TOTAL_SKIPPED}${RESET}"; echo ""
fi

echo "Phase 1 (non-ROOT): ${PARALLEL_PASSED} passed, ${PARALLEL_FAILED} failed (exit: ${PARALLEL_EXIT})"
echo "Phase 2 (ROOT):     ${SERIAL_PASSED} passed, ${SERIAL_FAILED} failed (exit: ${SERIAL_EXIT})"
echo ""; echo "Summary saved to: ${BOLD}${SUMMARY_FILE}${RESET}"

CRASH_COUNT=0
[[ -f "$CRASH_LOG" ]] && CRASH_COUNT=$(wc -l < "$CRASH_LOG" | tr -d ' ')
[[ "$CRASH_COUNT" -gt 0 ]] && echo "" && echo "${YELLOW}⚠️  WARNING: ${CRASH_COUNT} crashes detected${RESET}" && echo "   See: $CRASH_LOG"

# =============================================================================
# Phase 3: Capability Matrix
# =============================================================================

if [[ "$1" != "--quick" ]]; then
    print_header "Phase 3: Generating Capability Matrix"
    
    MATRIX_SCRIPT=""
    for c in "scripts/generate_capability_matrix.py" "tests/scripts/generate_capability_matrix.py"; do
        [[ -f "$c" ]] && MATRIX_SCRIPT="$c" && break
    done
    
    if [[ -n "$MATRIX_SCRIPT" ]]; then
        echo "Using: $MATRIX_SCRIPT"
        # FIX: Use --from-reports with glob patterns
        python3 "$MATRIX_SCRIPT" \
            --from-reports "$PARALLEL_JSON_DIR"/*.json "$SERIAL_JSON_DIR"/*.json \
            2>&1 || echo "⚠️  Capability matrix generation had errors"
    else
        echo "⚠️  generate_capability_matrix.py not found"
    fi
fi

# =============================================================================
# Phase 4: Orphaned root_serial check
# =============================================================================

print_header "Phase 4: Checking for orphaned root_serial tests"
orphaned=0
for tf in tests/test_*.py; do
    [[ -f "$tf" ]] || continue
    if grep -q "root_serial" "$tf" 2>/dev/null; then
        bn=$(basename "$tf"); found=0
        for rf in "${ROOT_TEST_FILES[@]}"; do
            [[ "$(basename "$rf")" == "$bn" ]] && found=1 && break
        done
        [[ $found -eq 0 ]] && echo "⚠️  ORPHANED: $tf" && orphaned=$((orphaned + 1))
    fi
done
[[ $orphaned -eq 0 ]] && echo "✅ No orphaned root_serial tests found"

# =============================================================================
# Phase 5: Generate review diffs
# =============================================================================

print_header "Phase 5: Generating review diffs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# 1. Diff of last commit (for multi-commit approval)
git diff HEAD~1..HEAD > "$LOG_DIR/diff_last_commit_${TIMESTAMP}.txt" 2>/dev/null && \
    echo "✅ diff_last_commit_${TIMESTAMP}.txt (last commit)" || \
    echo "⚠️  Could not generate last commit diff"

# 2. Diff to PHASE_BEGIN tag (for phase review)
if git rev-parse --verify PHASE_BEGIN &>/dev/null; then
    git diff PHASE_BEGIN..HEAD > "$LOG_DIR/diff_to_phase_${TIMESTAMP}.txt" 2>/dev/null && \
        echo "✅ diff_to_phase_${TIMESTAMP}.txt (since PHASE_BEGIN)" || \
        echo "⚠️  Could not generate phase diff"
else
    # Create helpful error file
    cat > "$LOG_DIR/diff_to_phase_${TIMESTAMP}.txt" << 'NOTAGEOF'
PHASE_BEGIN tag not found.

To enable phase diffs, create the tag at the start of each phase:
    git tag -f PHASE_BEGIN <commit-hash>

Example (tag last commit of previous phase):
    git log --oneline -5
    git tag -f PHASE_BEGIN abc1234

Then re-run tests to generate the diff.
NOTAGEOF
    echo "⚠️  No PHASE_BEGIN tag (see diff_to_phase_${TIMESTAMP}.txt for instructions)"
fi

# =============================================================================
# Final Status
# =============================================================================

print_header "Final Status"
echo "Summary file: $SUMMARY_FILE"; echo ""; cat "$SUMMARY_FILE"; echo ""

# List generated diff files
echo "Review diffs:"
[[ -f "$LOG_DIR/diff_last_commit_${TIMESTAMP}.txt" ]] && echo "  - $LOG_DIR/diff_last_commit_${TIMESTAMP}.txt"
[[ -f "$LOG_DIR/diff_to_phase_${TIMESTAMP}.txt" ]] && echo "  - $LOG_DIR/diff_to_phase_${TIMESTAMP}.txt"
echo ""

echo "NOTE: Exploration tests excluded. Run: ./tests/exploration/run_exploration.sh"; echo ""

[[ $TOTAL_FAILED -gt 0 ]] && echo "${RED}${BOLD}❌ Some tests failed${RESET}" && exit 1
echo "${GREEN}${BOLD}✅ All tests passed${RESET}" && exit 0
