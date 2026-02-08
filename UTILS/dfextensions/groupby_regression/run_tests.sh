#!/bin/bash
# Phase 13.7.GB — Unified test runner with capability matrix generation
#
# Usage:
#   source run_tests.sh              # Run all tests + generate matrix + log
#   source run_tests.sh quick        # Run tests only (no matrix)
#   source run_tests.sh matrix       # Generate matrix only (no tests)
#   source run_tests.sh json         # Generate JSON matrix
#   source run_tests.sh invariance   # Run only invariance/integration tests
#
# Environment variables:
#   PYTEST_WORKERS=4  source run_tests.sh   # Use 4 parallel workers (default: auto)
#
# Output structure (in test_logs/):
#   test_logs/
#   ├── test_full_20260207_163248.log        # Full test output
#   ├── test_failures_20260207_163248.log    # Failures only (empty = all pass)
#   ├── json_report_20260207_163248/         # pytest JSON report
#   │   └── .pytest_report.json
#   ├── CAPABILITY_MATRIX_20260207_163248.md # Auto-generated matrix snapshot
#   ├── capability_matrix_20260207_163248.json
#   ├── SUMMARY_20260207_163248.txt          # One-page summary
#   ├── diff_last_commit_20260207_163248.txt # Git diff since last commit
#   └── diff_to_phase_20260207_163248.txt    # Git diff to phase tag (if exists)

# Don't use set -e — we want to continue after test failures
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Numba threading fallback — use 'safe' if no OpenMP/TBB available
export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-safe}"

# Timestamp
TS=$(date +"%Y%m%d_%H%M%S")

MODE="full"
JSON_FLAG=""

for arg in "$@"; do
    # Strip leading dashes for flexibility (accept both "invariance" and "--invariance")
    clean_arg="${arg#--}"
    clean_arg="${clean_arg#-}"
    case "$clean_arg" in
        quick)       MODE="quick" ;;
        matrix)      MODE="matrix" ;;
        json)        JSON_FLAG="--json" ;;
        invariance)  MODE="invariance" ;;
        h|help)
            echo "Usage: source $0 [quick|matrix|json|invariance]"
            echo "  (default):   Run all tests + generate matrix + log to test_logs/"
            echo "  quick:       Run tests only (no matrix)"
            echo "  matrix:      Generate matrix only (skip tests)"
            echo "  json:        Output JSON matrix"
            echo "  invariance:  Run only invariance/integration layer tests"
            echo ""
            echo "Environment variables:"
            echo "  PYTEST_WORKERS=N  Set parallel workers (default: auto = all cores)"
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run 'source $0 help' for usage."
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

# Create test_logs directory
LOG_DIR="$SCRIPT_DIR/test_logs"
mkdir -p "$LOG_DIR"

# File paths
LOG_FILE="$LOG_DIR/test_${MODE}_${TS}.log"
FAIL_FILE="$LOG_DIR/test_failures_${TS}.log"
JSON_DIR="$LOG_DIR/json_report_${TS}"
MATRIX_MD="$LOG_DIR/CAPABILITY_MATRIX_${TS}.md"
MATRIX_JSON="$LOG_DIR/capability_matrix_${TS}.json"
SUMMARY_FILE="$LOG_DIR/SUMMARY_${TS}.txt"
DIFF_COMMIT="$LOG_DIR/diff_last_commit_${TS}.txt"
DIFF_PHASE="$LOG_DIR/diff_to_phase_${TS}.txt"

echo "========================================"
echo "Phase 13.7.GB — Test Runner"
echo "Mode: $MODE"
echo "Timestamp: $TS"
echo "NUMBA_THREADING_LAYER=$NUMBA_THREADING_LAYER"
echo "PYTEST_WORKERS=${PYTEST_WORKERS:-auto}"
echo "Log dir: $LOG_DIR"
echo "========================================"
echo ""

# ── Git diffs ─────────────────────────────────────────────────────
echo "--- Capturing git diffs ---"
if git rev-parse --is-inside-work-tree &>/dev/null; then
    git diff HEAD -- . > "$DIFF_COMMIT" 2>/dev/null || true
    # Try diff to phase tag — search both PHASE_* and phase_* conventions
    PHASE_TAG=$(git tag --list 'PHASE_BEGIN_GroupByRegression' 2>/dev/null | head -1)
    if [ -z "$PHASE_TAG" ]; then
        PHASE_TAG=$(git tag --list 'PHASE_*' --sort=-version:refname | head -1)
    fi
    if [ -n "$PHASE_TAG" ]; then
        git diff "$PHASE_TAG" -- . > "$DIFF_PHASE" 2>/dev/null || true
        echo "  Phase tag: $PHASE_TAG"
    else
        echo "  No phase tag found — skipping diff_to_phase"
        echo "(no phase tag found)" > "$DIFF_PHASE"
    fi
    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
else
    echo "(not a git repository)" > "$DIFF_COMMIT"
    echo "(not a git repository)" > "$DIFF_PHASE"
    GIT_HASH="unknown"
    GIT_BRANCH="unknown"
fi
echo ""

# ── Run tests ─────────────────────────────────────────────────────
TEST_EXIT=0
PASSED=0
FAILED=0
ERRORS=0
SKIPPED=0

if [ "$MODE" != "matrix" ]; then
    echo "--- Running tests ---"

    mkdir -p "$JSON_DIR"
    JSON_REPORT_FILE="$JSON_DIR/.pytest_report.json"

    # Parallel execution: use -n auto for all cores, or set PYTEST_WORKERS=N
    N_WORKERS="${PYTEST_WORKERS:-auto}"
    PYTEST_ARGS="-v --tb=short -n $N_WORKERS --json-report --json-report-file=$JSON_REPORT_FILE"

    if [ "$MODE" = "invariance" ]; then
        echo "NOTE: Running invariance/integration tests only"
        PYTEST_ARGS="$PYTEST_ARGS -m 'layer(\"invariance\") or layer(\"integration\")'"
    fi

    # Run tests, capture output + exit code
    eval python -m pytest $PYTEST_ARGS tests/ 2>&1 | tee "$LOG_FILE"
    TEST_EXIT=${PIPESTATUS[0]}

    # Extract failure lines
    grep -E "^FAILED |^ERROR " "$LOG_FILE" > "$FAIL_FILE" 2>/dev/null || true

    # Parse counts from pytest output
    SUMMARY_LINE=$(tail -5 "$LOG_FILE" | grep -E "[0-9]+ (passed|failed|error|skipped)" | tail -1)
    PASSED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
    FAILED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
    ERRORS=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo 0)
    SKIPPED=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)

    # Also copy JSON report to working dir for matrix generator
    cp "$JSON_REPORT_FILE" "$SCRIPT_DIR/.pytest_report.json" 2>/dev/null || true

    echo ""
fi

# ── Generate matrix ───────────────────────────────────────────────
if [ "$MODE" != "quick" ]; then
    echo "--- Generating capability matrix ---"

    MATRIX_ARGS=""
    if [ -f "$SCRIPT_DIR/.pytest_report.json" ]; then
        MATRIX_ARGS="--test-results $SCRIPT_DIR/.pytest_report.json"
    fi

    # Generate markdown
    python scripts/generate_capability_matrix.py $MATRIX_ARGS
    # Copy timestamped snapshot
    cp "$SCRIPT_DIR/docs/CAPABILITY_MATRIX.md" "$MATRIX_MD" 2>/dev/null || true

    # Generate JSON
    python scripts/generate_capability_matrix.py --json $MATRIX_ARGS
    cp "$SCRIPT_DIR/docs/capability_matrix.json" "$MATRIX_JSON" 2>/dev/null || true

    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────
{
    echo "========================================"
    echo "SUMMARY — Phase 13.7.GB Test Run"
    echo "========================================"
    echo ""
    echo "Timestamp:    $TS"
    echo "Date:         $(date)"
    echo "Mode:         $MODE"
    echo "Git branch:   $GIT_BRANCH"
    echo "Git commit:   $GIT_HASH"
    echo "Python:       $(python --version 2>&1)"
    echo "Platform:     $(uname -s) $(uname -m)"
    echo "NUMBA_THREADING_LAYER: $NUMBA_THREADING_LAYER"
    echo "PYTEST_WORKERS: ${PYTEST_WORKERS:-auto}"
    echo ""
    echo "── Test Results ──"
    echo "  Passed:   $PASSED"
    echo "  Failed:   $FAILED"
    echo "  Errors:   $ERRORS"
    echo "  Skipped:  $SKIPPED"
    echo "  Exit:     $TEST_EXIT"
    echo ""
    if [ -s "$FAIL_FILE" ]; then
        echo "── Failures ──"
        cat "$FAIL_FILE"
        echo ""
    fi
    echo "── Capability Matrix ──"
    if [ -f "$MATRIX_MD" ]; then
        # Extract summary table from matrix
        sed -n '/^## Summary/,/^## /p' "$MATRIX_MD" | head -12
    fi
    echo ""
    echo "── Files ──"
    echo "  Log:       $LOG_FILE"
    echo "  Failures:  $FAIL_FILE"
    echo "  JSON:      $JSON_DIR/"
    echo "  Matrix:    $MATRIX_MD"
    echo "  Summary:   $SUMMARY_FILE"
    echo "  Diff HEAD: $DIFF_COMMIT"
    echo "  Diff tag:  $DIFF_PHASE"
    echo "  Reviewer:  $LOG_DIR/reviewer_${TS}.zip"
    echo "========================================"
} | tee "$SUMMARY_FILE"

# ── Package reviewer.zip ──────────────────────────────────────────────────
echo ""
echo "--- Packaging reviewer.zip ---"

REVIEWER_ZIP="$LOG_DIR/reviewer_${TS}.zip"
REVIEWER_LATEST="$SCRIPT_DIR/reviewer.zip"

# Collect files into zip (relative paths for clean extraction)
(
    cd "$SCRIPT_DIR"

    ZIP_FILES=""

    # 1. Timestamped logs and diffs
    for f in \
        "test_logs/SUMMARY_${TS}.txt" \
        "test_logs/test_failures_${TS}.log" \
        "test_logs/test_${MODE}_${TS}.log" \
        "test_logs/CAPABILITY_MATRIX_${TS}.md" \
        "test_logs/capability_matrix_${TS}.json" \
        "test_logs/diff_last_commit_${TS}.txt" \
        "test_logs/diff_to_phase_${TS}.txt"
    do
        [ -f "$f" ] && ZIP_FILES="$ZIP_FILES $f"
    done

    # 2. Infrastructure files (for reviewer to read code)
    for f in \
        tests/feature_taxonomy.py \
        tests/test_layer_classification.py \
        tests/conftest.py \
        tests/README.md \
        scripts/generate_capability_matrix.py \
        pytest.ini \
        run_tests.sh
    do
        [ -f "$f" ] && ZIP_FILES="$ZIP_FILES $f"
    done

    # 3. Proposal and review docs (if present in docs/)
    for f in docs/PHASE_13_7_GB*.md docs/PHASE_*.md; do
        [ -f "$f" ] && ZIP_FILES="$ZIP_FILES $f"
    done

    # 4. Generated matrix (live copy in docs/)
    for f in docs/CAPABILITY_MATRIX.md docs/capability_matrix.json; do
        [ -f "$f" ] && ZIP_FILES="$ZIP_FILES $f"
    done

    if [ -n "$ZIP_FILES" ]; then
        zip -q "$REVIEWER_ZIP" $ZIP_FILES
        cp "$REVIEWER_ZIP" "$REVIEWER_LATEST"
        echo "  Reviewer package: $REVIEWER_ZIP"
        echo "  Also copied to:   $REVIEWER_LATEST"
        echo "  Contents:"
        zipinfo -1 "$REVIEWER_ZIP" | sed 's/^/    /'
    else
        echo "  WARNING: No files to package"
    fi
)

echo ""
echo "========================================"
echo "Done. Logs in: $LOG_DIR"
echo "  reviewer.zip: $REVIEWER_LATEST"
echo "========================================"
