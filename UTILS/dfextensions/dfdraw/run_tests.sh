#!/bin/bash
# =============================================================================
# run_tests.sh — dfdraw Test Runner
# =============================================================================
#
# Usage:
#   ./run_tests.sh              # Run all tests + generate matrix + diffs
#   ./run_tests.sh --quick      # Run tests only (no matrix)
#   ./run_tests.sh --matrix     # Generate matrix only (skip tests)
#   ./run_tests.sh --help       # Show help
#
# Environment:
#   PYTEST_WORKERS=N    Parallel workers (default: auto)
#
# Output (in test_logs/):
#   SUMMARY_<ts>.txt               Test summary
#   test_full_<ts>.log             Full pytest output
#   test_failures_<ts>.log         Failures only
#   CAPABILITY_MATRIX_<ts>.md      Auto-generated matrix snapshot
#   diff_last_commit_<ts>.txt      Git diff since last commit
#   diff_to_phase_<ts>.txt         Git diff to PHASE_BEGIN tag
#   git_status_<ts>.txt            Git status snapshot
#   reviewer_<ts>.zip              Review package

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# dfdraw root — run_tests.sh lives at dfdraw/
PROJECT_ROOT="$SCRIPT_DIR"
PROJECT_NAME="dfdraw"
SUBPROJECT="dfdraw"

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << 'EOF'
dfdraw Test Runner
===================

Usage:
  ./run_tests.sh [OPTIONS]

Options:
  --help       Show this help
  --quick      Run tests only (no matrix, no diffs)
  --matrix     Generate matrix only (skip tests)
  --verbose    Verbose pytest output

Environment:
  PYTEST_WORKERS=N    Parallel workers (default: auto)

Output:
  test_logs/SUMMARY_<ts>.txt            Test summary
  test_logs/CAPABILITY_MATRIX_<ts>.md   Feature matrix
  test_logs/diff_to_phase_<ts>.txt      Phase diff
  test_logs/git_status_<ts>.txt         Git status
  test_logs/reviewer_<ts>.zip           Review package

EOF
    exit 0
}

# =============================================================================
# Parse args
# =============================================================================

MODE="full"
PYTEST_VERBOSITY="-v"

for arg in "$@"; do
    case "$arg" in
        --help|-h)    show_help ;;
        --quick)      MODE="quick" ;;
        --matrix)     MODE="matrix" ;;
        --verbose)    PYTEST_VERBOSITY="-v --tb=long" ;;
    esac
done

# =============================================================================
# Configuration
# =============================================================================

TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

# dfdraw tests are fast — use auto or specified workers
PYTEST_WORKERS=${PYTEST_WORKERS:-auto}

# Colors
if [[ -t 1 ]] && command -v tput &>/dev/null; then
    RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
    BOLD=$(tput bold); RESET=$(tput sgr0)
else
    RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
fi

# File paths
LOG_FILE="$LOG_DIR/test_full_${TS}.log"
FAIL_FILE="$LOG_DIR/test_failures_${TS}.log"
JSON_DIR="$LOG_DIR/json_report_${TS}"
JSON_REPORT="$JSON_DIR/.pytest_report.json"
MATRIX_MD="$LOG_DIR/CAPABILITY_MATRIX_${TS}.md"
SUMMARY_FILE="$LOG_DIR/SUMMARY_${TS}.txt"
DIFF_COMMIT="$LOG_DIR/diff_last_commit_${TS}.txt"
DIFF_PHASE="$LOG_DIR/diff_to_phase_${TS}.txt"
GIT_STATUS="$LOG_DIR/git_status_${TS}.txt"

echo "========================================"
echo "dfdraw Test Runner"
echo "Mode: $MODE"
echo "Timestamp: $TS"
echo "========================================"
echo ""

# =============================================================================
# Git diffs + status
# =============================================================================

echo "--- Capturing git info ---"
GIT_HASH="unknown"
GIT_BRANCH="unknown"

if git rev-parse --is-inside-work-tree &>/dev/null; then
    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

    # Last commit diff (scoped to dfdraw)
    git diff HEAD~1..HEAD -- "$PROJECT_ROOT" > "$DIFF_COMMIT" 2>/dev/null || \
        echo "(could not generate)" > "$DIFF_COMMIT"
    echo "  Last commit diff: $DIFF_COMMIT"

    # Phase diff
    PHASE_TAG=""
    for tag in PHASE_BEGIN_dfdraw; do
        if git rev-parse --verify "$tag" &>/dev/null; then
            PHASE_TAG="$tag"
            break
        fi
    done

    if [[ -n "$PHASE_TAG" ]]; then
        git diff "$PHASE_TAG"..HEAD -- "$PROJECT_ROOT" > "$DIFF_PHASE" 2>/dev/null || true
        echo "  Phase tag: $PHASE_TAG"
    else
        echo "(No PHASE_BEGIN_dfdraw tag found)" > "$DIFF_PHASE"
        echo "  ⚠️  No phase tag — create with: source scripts/phase_tag.sh && phase_begin 13_15_DF"
    fi

    # Git status snapshot
    git status --short -- "$PROJECT_ROOT" > "$GIT_STATUS" 2>/dev/null
    echo "  Git status: $GIT_STATUS"
else
    echo "(not a git repository)" > "$DIFF_COMMIT"
    echo "(not a git repository)" > "$DIFF_PHASE"
    echo "(not a git repository)" > "$GIT_STATUS"
fi
echo ""

# =============================================================================
# Run tests
# =============================================================================

TEST_EXIT=0
PASSED=0; FAILED=0; ERRORS=0; SKIPPED=0
DURATION_STR="N/A"

if [[ "$MODE" != "matrix" ]]; then
    echo "--- Running tests ---"
    mkdir -p "$JSON_DIR"

    START_TIME=$(date +%s)

    # dfdraw single-phase test run (no ROOT serial phase needed)
    python3 -m pytest tests/ \
        $PYTEST_VERBOSITY \
        --tb=short \
        --json-report --json-report-file="$JSON_REPORT" \
        2>&1 | tee "$LOG_FILE"
    TEST_EXIT=${PIPESTATUS[0]}

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_STR="$((DURATION / 60))m $((DURATION % 60))s"

    # Extract counts from pytest output
    SUMMARY_LINE=$(tail -5 "$LOG_FILE" | grep -E "[0-9]+ (passed|failed|error|skipped)" | tail -1)
    PASSED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
    FAILED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
    ERRORS=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo 0)
    SKIPPED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)

    # Extract failures
    grep -E "^FAILED |^ERROR " "$LOG_FILE" > "$FAIL_FILE" 2>/dev/null || true

    echo ""
    echo "⏱️  Tests completed in $DURATION_STR"
    echo ""
fi

# =============================================================================
# Generate capability matrix
# =============================================================================

if [[ "$MODE" != "quick" ]]; then
    echo "--- Generating capability matrix ---"

    MATRIX_SCRIPT=""
    for candidate in \
        "scripts/generate_capability_matrix.py"; do
        [[ -f "$candidate" ]] && MATRIX_SCRIPT="$candidate" && break
    done

    if [[ -n "$MATRIX_SCRIPT" ]]; then
        MATRIX_ARGS=""
        [[ -f "$JSON_REPORT" ]] && MATRIX_ARGS="--test-results $JSON_REPORT"

        python3 "$MATRIX_SCRIPT" $MATRIX_ARGS --phase "13.15.DF" 2>&1 || \
            echo "⚠️  Capability matrix generation had errors"

        # Copy timestamped snapshot
        [[ -f "docs/CAPABILITY_MATRIX.md" ]] && cp "docs/CAPABILITY_MATRIX.md" "$MATRIX_MD"
    else
        echo "⚠️  generate_capability_matrix.py not found"
        echo "    Expected at: scripts/generate_capability_matrix.py"
    fi
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================

{
    echo "========================================"
    echo "SUMMARY — dfdraw Test Run"
    echo "========================================"
    echo ""
    echo "Timestamp:    $TS"
    echo "Date:         $(date)"
    echo "Mode:         $MODE"
    echo "Git branch:   $GIT_BRANCH"
    echo "Git commit:   $GIT_HASH"
    echo "Python:       $(python3 --version 2>&1)"
    echo "Platform:     $(uname -s) $(uname -m)"
    echo "Duration:     ${DURATION_STR}"
    echo ""
    echo "── Test Results ──"
    echo "  Passed:   $PASSED"
    echo "  Failed:   $FAILED"
    echo "  Errors:   $ERRORS"
    echo "  Skipped:  $SKIPPED"
    echo "  Exit:     $TEST_EXIT"
    echo ""
    if [[ -s "$FAIL_FILE" ]]; then
        echo "── Failures ──"
        cat "$FAIL_FILE"
        echo ""
    fi
    if [[ -f "$MATRIX_MD" ]]; then
        echo "── Capability Matrix Summary ──"
        sed -n '/^## Summary/,/^## /p' "$MATRIX_MD" | head -12
        echo ""
    fi
    echo "── Files ──"
    echo "  Log:      $LOG_FILE"
    echo "  Failures: $FAIL_FILE"
    echo "  Matrix:   $MATRIX_MD"
    echo "  Summary:  $SUMMARY_FILE"
    echo "  Diff:     $DIFF_COMMIT"
    echo "  Phase:    $DIFF_PHASE"
    echo "  Status:   $GIT_STATUS"
    echo "========================================"
} | tee "$SUMMARY_FILE"

# =============================================================================
# Package reviewer.zip
# =============================================================================

echo ""
echo "--- Packaging reviewer.zip ---"

REVIEWER_ZIP="$LOG_DIR/reviewer_${TS}.zip"

(
    cd "$PROJECT_ROOT"
    ZIP_FILES=""
    for f in \
        "$SUMMARY_FILE" \
        "$FAIL_FILE" \
        "$MATRIX_MD" \
        "$DIFF_COMMIT" \
        "$DIFF_PHASE" \
        "$GIT_STATUS" \
        "docs/CAPABILITY_MATRIX.md"
    do
        [[ -f "$f" ]] && ZIP_FILES="$ZIP_FILES $f"
    done

    if [[ -n "$ZIP_FILES" ]]; then
        zip -q "$REVIEWER_ZIP" $ZIP_FILES 2>/dev/null || true
        echo "  Reviewer package: $REVIEWER_ZIP"
    fi
)

# =============================================================================
# Final
# =============================================================================

echo ""
if [[ $FAILED -gt 0 || $ERRORS -gt 0 ]]; then
    echo "${RED}${BOLD}❌ Some tests failed${RESET}"
    exit 1
else
    echo "${GREEN}${BOLD}✅ All tests passed${RESET}"
    exit 0
fi
