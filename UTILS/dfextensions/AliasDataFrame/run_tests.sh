#!/bin/bash
# =============================================================================
# run_tests.sh — AliasDataFrame Test Runner
# =============================================================================
#
# Usage:
#   ./run_tests.sh              # Run all tests + generate matrix + diffs
#   ./run_tests.sh --quick      # Run tests only (no matrix)
#   ./run_tests.sh --matrix     # Generate matrix only (skip tests)
#   ./run_tests.sh --help       # Show help
#
# Environment:
#   PYTEST_WORKERS=N    Parallel workers (default: 12)
#
# Output (in test_logs/):
#   SUMMARY_<ts>.txt               Test summary
#   test_full_<ts>.log             Full pytest output
#   test_focused_<ts>.log          Phase suite only
#   runxfail_focused_<ts>.log      Phase suite with xfail DISABLED (failures expected)
#   focused_nodes_<ts>.txt         Exact focused collection manifest
#   md5_manifest_<ts>.txt          Fingerprints of the reviewed FINAL bytes
#   test_failures_<ts>.log         Failures only
#   CAPABILITY_MATRIX_<ts>.md      Auto-generated Markdown matrix snapshot
#   CAPABILITY_MATRIX_<ts>.html    Auto-generated HTML matrix snapshot
#   diff_last_commit_<ts>.txt      Uncommitted diff + last commit diff
#   diff_to_phase_<ts>.txt         Diff since PHASE_BEGIN_AliasDataFrame tag
#   git_status_<ts>.txt            Working tree state (git status --porcelain)
#   reviewer_<ts>.zip              Review package (zip)
#   reviewer_<ts>.tar              Review package (tar, same file list)
#   reviewer_<ts>.llmbundle.txt    Review package (plain text, same file list)
#   reviewer_digests_<ts>.txt      MD5/SHA256 of ALL packages
#   timing_<ts>.txt                Final human/machine-readable stage timings
#   timing_packet_<ts>.txt         Pre-package timing snapshot shipped in packet
#   reviewer_manifest_<ts>.json     Versioned logical packet manifest + digest
#   reviewer_profile_<ts>.txt       Read-only packet byte/character/token-estimate profile

# Don't exit on test failures
# set -e

RUN_TESTS_TOTAL_START_EPOCH=$(date +%s)
RUN_TESTS_TOTAL_START_WALL=$(date +"%Y-%m-%d %H:%M:%S %z")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to AliasDataFrame root (parent of tests/)
if [[ "$(basename "$SCRIPT_DIR")" == "tests" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
elif [[ -d "$SCRIPT_DIR/tests" ]]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "ERROR: Cannot determine project structure"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << 'EOF'
AliasDataFrame Test Runner
==========================

Usage:
  ./run_tests.sh [OPTIONS]

Options:
  --help       Show this help
  --quick      Run tests only (no matrix)
  --matrix     Generate matrix only (skip tests)
  --verbose    Verbose pytest output

Environment:
  PYTEST_WORKERS=N    Parallel workers (default: 12)

Output:
  test_logs/SUMMARY_<ts>.txt            Test summary
  test_logs/test_focused_<ts>.log       Phase suite only
  test_logs/runxfail_focused_<ts>.log   Phase suite, xfail disabled (failures expected)
  test_logs/focused_nodes_<ts>.txt      Exact focused collection manifest
  test_logs/md5_manifest_<ts>.txt       Fingerprints of reviewed final bytes
  test_logs/CAPABILITY_MATRIX_<ts>.md   Markdown feature matrix
  test_logs/CAPABILITY_MATRIX_<ts>.html HTML feature matrix
  test_logs/diff_last_commit_<ts>.txt   Uncommitted + HEAD~1 diffs
  test_logs/diff_to_phase_<ts>.txt      Diff since PHASE_BEGIN tag
  test_logs/git_status_<ts>.txt         Working tree state snapshot
  test_logs/reviewer_<ts>.zip           Review package (zip)
  test_logs/reviewer_<ts>.tar           Review package (tar, same file list)
  test_logs/reviewer_<ts>.llmbundle.txt Review package (plain text, same file list)
  test_logs/reviewer_digests_<ts>.txt   MD5/SHA256 of ALL packages
  test_logs/timing_<ts>.txt             Final stage timings
  test_logs/timing_packet_<ts>.txt      Pre-package timing snapshot shipped in packet
  test_logs/reviewer_manifest_<ts>.json  Logical packet manifest/duplicate groups
  test_logs/reviewer_profile_<ts>.txt    Packet volume/profile summary

Environment:
  ADF_REVIEWER_TAR=0     Disable the .tar companion (zip only)
  ADF_REVIEWER_BUNDLE=0  Disable the plain-text LLMBUNDLE companion

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

# PHASE_13_80 A14/A15: human-visible timing instrumentation.
# IMPORTANT: these helpers only RECORD timing.  They must never execute a
# pytest/package stage.  In particular, TEST_EXIT=${PIPESTATUS[0]} must remain
# immediately after the full pytest pipeline, before timing_end is called.
TIMING_FILE="$PROJECT_ROOT/$LOG_DIR/timing_${TS}.txt"
TIMING_PACKET_SNAPSHOT="$PROJECT_ROOT/$LOG_DIR/timing_packet_${TS}.txt"
TIMING_ROWS_FILE="$PROJECT_ROOT/$LOG_DIR/.timing_rows_${TS}.tsv"
TIMING_TOTAL_START_EPOCH=$RUN_TESTS_TOTAL_START_EPOCH
TIMING_TOTAL_START_WALL="$RUN_TESTS_TOTAL_START_WALL"
TIMING_STAGE_NAME=""
TIMING_STAGE_START_EPOCH=0
TIMING_STAGE_START_WALL=""

timing_start() {
    local stage="$1"
    TIMING_STAGE_NAME="$stage"
    TIMING_STAGE_START_EPOCH=$(date +%s)
    TIMING_STAGE_START_WALL=$(date +"%Y-%m-%d %H:%M:%S %z")
    printf '[%s] START %s\n' "$TIMING_STAGE_START_WALL" "$stage"
    {
        printf 'stage: %s\n' "$stage"
        printf 'start: %s\n' "$TIMING_STAGE_START_WALL"
    } >> "$TIMING_FILE"
}

timing_end() {
    local stage="$1"
    local result="${2:-0}"
    local end_epoch end_wall elapsed
    end_epoch=$(date +%s)
    end_wall=$(date +"%Y-%m-%d %H:%M:%S %z")
    elapsed=$((end_epoch - TIMING_STAGE_START_EPOCH))
    printf '[%s] END %s elapsed=%ss result=%s\n' \
        "$end_wall" "$stage" "$elapsed" "$result"
    {
        printf 'end: %s\n' "$end_wall"
        printf 'elapsed_seconds: %s\n' "$elapsed"
        printf 'result: %s\n\n' "$result"
    } >> "$TIMING_FILE"
    printf '%s\t%s\n' "$stage" "$elapsed" >> "$TIMING_ROWS_FILE"
    TIMING_STAGE_NAME=""
    TIMING_STAGE_START_EPOCH=0
    TIMING_STAGE_START_WALL=""
}

timing_snapshot_for_packet() {
    {
        cat "$TIMING_FILE"
        echo "packet_snapshot: true"
        echo "packet_snapshot_note: packaging-stage timings continue in timing_${TS}.txt outside the packet to avoid self-referential packet-byte drift"
    } > "$TIMING_PACKET_SNAPSHOT"
}

timing_finish_total() {
    local total_end_epoch total_end_wall total_seconds stage_sum other
    total_end_epoch=$(date +%s)
    total_end_wall=$(date +"%Y-%m-%d %H:%M:%S %z")
    total_seconds=$((total_end_epoch - TIMING_TOTAL_START_EPOCH))
    stage_sum=$(awk -F '\t' '{s += $2} END {print s+0}' "$TIMING_ROWS_FILE" 2>/dev/null)
    stage_sum=${stage_sum:-0}
    other=$((total_seconds - stage_sum))
    if [[ $other -lt 0 ]]; then
        other=0
    fi

    echo ""
    echo "TIMING SUMMARY"
    echo ""
    if [[ -s "$TIMING_ROWS_FILE" ]]; then
        awk -F '\t' '{printf "  %-34s %6s s\n", $1, $2}' "$TIMING_ROWS_FILE"
    fi
    printf '  %-34s %6s s\n' "unclassified/other" "$other"
    echo "  -------------------------------------------"
    printf '  %-34s %6s s\n' "TOTAL" "$total_seconds"
    echo ""

    {
        printf 'end: %s\n' "$total_end_wall"
        printf 'total_seconds: %s\n' "$total_seconds"
        printf 'classified_stage_seconds: %s\n' "$stage_sum"
        printf 'unclassified_other_seconds: %s\n' "$other"
    } >> "$TIMING_FILE"
    rm -f "$TIMING_ROWS_FILE"
}

{
    echo "ADF_RUN_TIMING/1"
    echo "run_timestamp: $TS"
    echo "start: $TIMING_TOTAL_START_WALL"
    echo ""
} > "$TIMING_FILE"
: > "$TIMING_ROWS_FILE"

printf '[%s] START run_tests.sh\n' "$TIMING_TOTAL_START_WALL"

PYTEST_WORKERS=${PYTEST_WORKERS:-12}

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
MATRIX_HTML="$LOG_DIR/CAPABILITY_MATRIX_${TS}.html"
MATRIX_AI_JSON="$LOG_DIR/CAPABILITY_MATRIX_${TS}.json"
SUMMARY_FILE="$LOG_DIR/SUMMARY_${TS}.txt"
DIFF_COMMIT="$LOG_DIR/diff_last_commit_${TS}.txt"
DIFF_PHASE="$LOG_DIR/diff_to_phase_${TS}.txt"
GIT_STATUS="$LOG_DIR/git_status_${TS}.txt"

# Focused (phase) suite, logged SEPARATELY and shipped in the packet.
# Override the pattern per phase, e.g.:
#   FOCUSED_TESTS="tests/test_phase_13_77_*.py" bash run_tests.sh
FOCUSED_TESTS="${FOCUSED_TESTS:-tests/test_phase_13_79_*.py}"
FOCUSED_LOG="$LOG_DIR/test_focused_${TS}.log"

# RUNNER-FOCUS-1: exact focused collection evidence. The manifest is the
# execution authority: collect once, then both normal and raw lanes execute
# exactly those collected node IDs with the same xdist configuration.
FOCUSED_NODE_LOG="$LOG_DIR/focused_nodes_${TS}.txt"

# --runxfail evidence for the focused suite.
RUNXFAIL_LOG="$LOG_DIR/runxfail_focused_${TS}.log"

# Candidate fingerprints.
MD5_MANIFEST="$LOG_DIR/md5_manifest_${TS}.txt"
REVIEWER_ZIP="$LOG_DIR/reviewer_${TS}.zip"

# Companion .tar of the SAME file list, built in the SAME step.  Some reviewer
# environments cannot read the zip; a second container removes that as a
# blocker WITHOUT changing the canonical zip that every other consumer uses.
# Set ADF_REVIEWER_TAR=0 to skip it.
REVIEWER_TAR="$LOG_DIR/reviewer_${TS}.tar"

# Plain-text companion.  Some reviewer runtimes cannot open ANY archive: the
# attachment -> analysis-runtime handoff fails for zip and tar alike, so the
# container was never the problem.  A text bundle removes the container.
# It is built FROM THE ZIP, never from a directory, so it cannot drift from
# the reviewed file list (finding D-1), and it carries a header/manifest/
# trailer so a truncated bundle is detectable rather than silently short.
# Set ADF_REVIEWER_BUNDLE=0 to skip it.
REVIEWER_BUNDLE="$LOG_DIR/reviewer_${TS}.llmbundle.txt"
REVIEWER_DIGESTS="$LOG_DIR/reviewer_digests_${TS}.txt"

# PHASE_13_80 PART-A A1-A8: one logical packet, lossless exact-content
# duplicate measurement, consumer-safe archive delivery, shared ZIP->LLMBUNDLE custody and read-only profiler.
PACKET_TOOL="scripts/reviewer_packet.py"
REVIEWER_INPUTS="$LOG_DIR/reviewer_inputs_${TS}.txt"
REVIEWER_MANIFEST="$LOG_DIR/reviewer_manifest_${TS}.json"
REVIEWER_PROFILE="$LOG_DIR/reviewer_profile_${TS}.txt"
REVIEWER_PROFILE_JSON="$LOG_DIR/reviewer_profile_${TS}.json"

# Absolute paths, computed before packaging so the summary can print them.
REVIEWER_ZIP_ABS="$(realpath -m "$REVIEWER_ZIP" 2>/dev/null || echo "$PROJECT_ROOT/$REVIEWER_ZIP")"
REVIEWER_TAR_ABS="$(realpath -m "$REVIEWER_TAR" 2>/dev/null || echo "$PROJECT_ROOT/$REVIEWER_TAR")"
REVIEWER_BUNDLE_ABS="$(realpath -m "$REVIEWER_BUNDLE" 2>/dev/null || echo "$PROJECT_ROOT/$REVIEWER_BUNDLE")"

echo "========================================"
echo "AliasDataFrame Test Runner"
echo "Mode: $MODE"
echo "Timestamp: $TS"
echo "PYTEST_WORKERS: $PYTEST_WORKERS"
echo "========================================"
echo ""

# =============================================================================
# Git diffs
# =============================================================================

echo "--- Capturing git diffs ---"
GIT_HASH="unknown"
GIT_BRANCH="unknown"

if git rev-parse --is-inside-work-tree &>/dev/null; then
    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    if [[ -z "$(git status --porcelain -- "$PROJECT_ROOT" 2>/dev/null)" ]]; then
        GIT_TREE_STATE="CLEAN"
    else
        GIT_TREE_STATE="DIRTY (uncommitted changes — not anchored to $GIT_HASH)"
    fi

    timing_start "diff_last_commit"
    {
        echo "=== Uncommitted changes (git diff HEAD) ==="
        echo "=== staged + unstaged, relative to last commit ==="
        echo ""
        git diff --relative HEAD -- . 2>/dev/null || echo "(no uncommitted changes)"
        echo ""
        echo "=== Previous commit (git diff HEAD~1..HEAD) ==="
        echo ""
        git diff --relative HEAD~1..HEAD -- . 2>/dev/null || echo "(no previous commit)"
    } > "$DIFF_COMMIT"
    timing_end "diff_last_commit" 0
    echo "  Last commit diff: $(realpath "$DIFF_COMMIT" 2>/dev/null || echo "$DIFF_COMMIT")"

    PHASE_TAG=""
    for tag in PHASE_BEGIN_AliasDataFrame PHASE_BEGIN_ADF; do
        if git rev-parse --verify "$tag" &>/dev/null; then
            PHASE_TAG="$tag"
            break
        fi
    done

    timing_start "diff_to_phase"
    if [[ -n "$PHASE_TAG" ]]; then
        git diff --relative "$PHASE_TAG" -- . > "$DIFF_PHASE" 2>/dev/null || true
        echo "  Phase tag: $PHASE_TAG"
    else
        echo "(No PHASE_BEGIN_* tag found — searched: PHASE_BEGIN_AliasDataFrame, PHASE_BEGIN_ADF)" > "$DIFF_PHASE"
        echo "  ⚠️  No phase tag — create with: source scripts/phase_tag.sh && phase_begin <id>"
    fi
    timing_end "diff_to_phase" 0

    timing_start "environment / provenance"
    {
        echo "=== git status --porcelain (scoped to cwd) ==="
        git status --porcelain -- . 2>/dev/null || echo "(git status failed)"
        echo ""
        echo "=== git status (human-readable) ==="
        git status -- . 2>/dev/null || echo "(git status failed)"
        echo ""
        echo "=== Current HEAD ==="
        git log -1 --oneline 2>/dev/null || echo "(git log failed)"
        echo ""
        echo "=== Branch ==="
        git branch --show-current 2>/dev/null || echo "(git branch failed)"
        echo ""
        echo "=== PHASE_* tags (newest first, last 10) ==="
        git tag --list 'PHASE_*' --sort=-creatordate 2>/dev/null | head -10 \
            || echo "(no PHASE_* tags found)"
        echo ""
        echo "=== Recent commits with tag decorations (last 5) ==="
        git log --oneline --decorate -5 2>/dev/null || echo "(git log failed)"
    } > "$GIT_STATUS"
    timing_end "environment / provenance" 0
    echo "  Working tree: $(realpath "$GIT_STATUS" 2>/dev/null || echo "$GIT_STATUS")"
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

if [[ "$MODE" != "matrix" ]]; then
    echo "--- Running tests ---"
    mkdir -p "$JSON_DIR"

    START_TIME=$(date +%s)
    timing_start "full pytest"

    python3 -m pytest tests/ \
        $PYTEST_VERBOSITY \
        --tb=short \
        -n "$PYTEST_WORKERS" \
        --continue-on-collection-errors \
        --json-report --json-report-file="$JSON_REPORT" \
        2>&1 | tee "$LOG_FILE"
    TEST_EXIT=${PIPESTATUS[0]}
    timing_end "full pytest" "$TEST_EXIT"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_STR="$((DURATION / 60))m $((DURATION % 60))s"

    read PASSED FAILED ERRORS SKIPPED < <(python3 - "$JSON_REPORT" << 'PYCOUNT'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print("0 0 0 0"); sys.exit()
s = r.get("summary", {})
passed = s.get("passed", 0)
failed = s.get("failed", 0)
skipped = s.get("skipped", 0)
errors = s.get("error", 0)
errors += sum(
    1 for c in r.get("collectors", [])
    if c.get("outcome") in ("failed", "error")
)
print(passed, failed, errors, skipped)
PYCOUNT
)
    PASSED=${PASSED:-0}; FAILED=${FAILED:-0}; ERRORS=${ERRORS:-0}; SKIPPED=${SKIPPED:-0}

    grep -E "^FAILED |^ERROR " "$LOG_FILE" 2>/dev/null | sort -u > "$FAIL_FILE" || true

    # Focused suite: collect one node set, then execute the SAME selection
    # normally and with --runxfail. Both execution lanes use xdist.
    if compgen -G "$FOCUSED_TESTS" > /dev/null 2>&1; then
        echo "--- Collecting focused node set: $FOCUSED_TESTS ---"
        timing_start "focused collection"
        python3 -m pytest $FOCUSED_TESTS --collect-only -q 2>/dev/null \
            | grep '::' | sort -u > "$FOCUSED_NODE_LOG" || true
        timing_end "focused collection" 0
        FOCUSED_NODE_COUNT=$(grep -c '::' "$FOCUSED_NODE_LOG" 2>/dev/null || echo 0)
        echo "  focused nodes: $FOCUSED_NODE_COUNT"

        mapfile -t FOCUSED_NODES < "$FOCUSED_NODE_LOG"
        if [[ ${#FOCUSED_NODES[@]} -eq 0 ]]; then
            echo "ERROR: focused collection produced zero executable node IDs" >&2
            return 1 2>/dev/null || false
        fi

        echo "--- Running exact collected focused node set ---"
        timing_start "focused pytest"
        python3 -m pytest "${FOCUSED_NODES[@]}" -n "$PYTEST_WORKERS" -q --tb=short \
            2>&1 | tee "$FOCUSED_LOG"
        FOCUSED_EXIT=${PIPESTATUS[0]}
        timing_end "focused pytest" "$FOCUSED_EXIT"
        FOCUSED_LINE=$(grep -E "^[0-9]+ (passed|failed)" "$FOCUSED_LOG" | tail -1)
        echo "  focused: ${FOCUSED_LINE:-<no summary line>}"

        echo "--- Recording --runxfail evidence for the exact same focused node set ---"
        timing_start "raw-XFAIL pytest"
        python3 -m pytest "${FOCUSED_NODES[@]}" --runxfail -n "$PYTEST_WORKERS" \
            -q --tb=line -p no:warnings > "$RUNXFAIL_LOG" 2>&1
        RUNXFAIL_EXIT=$?
        timing_end "raw-XFAIL pytest" "$RUNXFAIL_EXIT"
        RUNXFAIL_LINE=$(grep -E "^[0-9]+ (passed|failed)" "$RUNXFAIL_LOG" | tail -1)
        echo "  runxfail: ${RUNXFAIL_LINE:-<no summary line>} (failures here are EXPECTED)"
    else
        echo "--- No focused suite matched: $FOCUSED_TESTS ---"
    fi

    echo ""
    echo "⏱️  Tests completed in $DURATION_STR"
    echo ""
fi

# =============================================================================
# Generate capability matrix — shared semantics, established HTML renderer
# =============================================================================

if [[ "$MODE" != "quick" ]]; then
    echo "--- Generating capability matrix ---"

    MATRIX_SCRIPT=""
    for candidate in \
        "scripts/generate_capability_matrix.py" \
        "tests/scripts/generate_capability_matrix.py"; do
        [[ -f "$candidate" ]] && MATRIX_SCRIPT="$candidate" && break
    done

    HTML_SCRIPT=""
    for candidate in \
        "scripts/generate_matrix_html.py" \
        "tests/scripts/generate_matrix_html.py"; do
        [[ -f "$candidate" ]] && HTML_SCRIPT="$candidate" && break
    done

    # Both renderers consume the same pytest JSON evidence.  The HTML script
    # imports build_matrix_model() from generate_capability_matrix.py, so the
    # historical rich renderer is presentation-only and cannot independently
    # redefine feature status.
    MATRIX_JSON=""
    if [[ -f "$JSON_REPORT" ]]; then
        MATRIX_JSON="$JSON_REPORT"
    elif [[ -f ".pytest_report.json" ]]; then
        MATRIX_JSON=".pytest_report.json"
    fi

    # MATRIX-PHASE-1: matrix provenance is the current ADF work phase, not
    # the latest historical BEGIN tag.  Override explicitly when needed.
    PHASE_FOR_MATRIX="${ADF_MATRIX_PHASE:-PHASE_13_79_ADF}"

    if [[ -n "$MATRIX_SCRIPT" && -n "$MATRIX_JSON" ]]; then
        timing_start "Capability Matrix Markdown/JSON generation"
        python3 "$MATRIX_SCRIPT" \
            --test-results "$MATRIX_JSON" \
            --phase "$PHASE_FOR_MATRIX" \
            --json-output "docs/CAPABILITY_MATRIX.json" 2>&1 || \
            echo "⚠️  Capability matrix Markdown generation had errors"

        [[ -f "docs/CAPABILITY_MATRIX.md" ]] && \
            cp "docs/CAPABILITY_MATRIX.md" "$MATRIX_MD"
        [[ -f "docs/CAPABILITY_MATRIX.json" ]] && \
            cp "docs/CAPABILITY_MATRIX.json" "$MATRIX_AI_JSON"
        timing_end "Capability Matrix Markdown/JSON generation" 0
    elif [[ -z "$MATRIX_SCRIPT" ]]; then
        echo "⚠️  generate_capability_matrix.py not found"
        echo "    Expected at: scripts/generate_capability_matrix.py"
    else
        echo "⚠️  pytest JSON report not found; Capability Matrix not regenerated"
    fi

    if [[ -n "$HTML_SCRIPT" && -n "$MATRIX_JSON" ]]; then
        timing_start "Capability Matrix HTML generation"
        # MATRIX-PARITY-1 + HTML-PRESENTATION-1:
        # keep the established interactive HTML renderer, but feed it the
        # shared normalized semantic model via the same pytest JSON evidence.
        python3 "$HTML_SCRIPT" \
            --test-results "$MATRIX_JSON" \
            --output "docs/CAPABILITY_MATRIX.html" \
            --snapshot "$MATRIX_HTML" \
            --phase "$PHASE_FOR_MATRIX" 2>&1 || \
            echo "⚠️  Capability matrix HTML generation had errors"
        timing_end "Capability Matrix HTML generation" 0
    elif [[ -z "$HTML_SCRIPT" ]]; then
        echo "⚠️  generate_matrix_html.py not found"
        echo "    Expected at: scripts/generate_matrix_html.py"
    fi
    echo ""
fi

# Environment stamp into Markdown only.  The established HTML renderer keeps
# its own environment/off-gate banner.
ENV_STAMP="*Environment: $(hostname) · $(uname -s)-$(uname -m) · Python $(python3 -c 'import platform; print(platform.python_version())') · stamped by run_tests.sh*"
for mdf in "docs/CAPABILITY_MATRIX.md" "$MATRIX_MD"; do
    if [[ -f "$mdf" ]] && ! grep -q '^\*Environment: ' "$mdf"; then
        printf '\n%s\n' "$ENV_STAMP" >> "$mdf"
    fi
done

# =============================================================================
# Final candidate fingerprints — AFTER matrix generation/copy/render
# =============================================================================
# REVIEWER-CUSTODY: candidate hashes must describe the final bytes that are
# about to enter reviewer.zip, not pre-generation matrix bytes.

timing_start "candidate hashing / MD5 manifest"
CANDIDATE_FILES=$(
    {
        echo "AliasDataFrame.py"
        for t in $FOCUSED_TESTS; do echo "$t"; done
        if git rev-parse --is-inside-work-tree &>/dev/null; then
            git diff --name-only --relative HEAD 2>/dev/null
            git diff --cached --name-only --relative HEAD 2>/dev/null
        fi
    } | sed 's|^\./||' | sort -u
)

{
    echo "=== MD5 of the candidate files ==="
    echo "(the FINAL bytes this run measured; compare against the CRR)"
    echo "(computed after Markdown/HTML generation and final copies)"
    echo ""
    for f in $CANDIDATE_FILES; do
        [[ -f "$f" ]] && md5sum "$f" 2>/dev/null
    done
    echo ""
    echo "=== staged blob MD5 (what a commit would record) ==="
    if git rev-parse --is-inside-work-tree &>/dev/null; then
        for f in $CANDIDATE_FILES; do
            if git ls-files --error-unmatch "$f" &>/dev/null; then
                printf '%s  %s\n' \
                    "$(git show ":0:./$f" 2>/dev/null | md5sum | cut -d' ' -f1)" "$f"
            fi
        done
    fi
} > "$MD5_MANIFEST" 2>/dev/null || true
timing_end "candidate hashing / MD5 manifest" 0

# =============================================================================
# Summary
# =============================================================================

timing_start "summary generation"
{
    echo "========================================"
    echo "SUMMARY — AliasDataFrame Test Run"
    echo "========================================"
    echo ""
    echo "Timestamp:    $TS"
    echo "Date:         $(date)"
    echo "Mode:         $MODE"
    echo "Git branch:   $GIT_BRANCH"
    echo "Git commit:   $GIT_HASH"
    echo "Tree state:   $GIT_TREE_STATE"
    echo "Python:       $(python3 --version 2>&1)"
    echo "Platform:     $(uname -s) $(uname -m)"
    echo "Packages:     $(python3 -c 'import pandas,numpy;print(f"pandas {pandas.__version__} numpy {numpy.__version__}",end="")' 2>/dev/null || echo "pandas ? numpy ?")$(python3 -c 'import pyarrow;print(f" pyarrow {pyarrow.__version__}",end="")' 2>/dev/null)$(python3 -c 'import uproot;print(f" uproot {uproot.__version__}",end="")' 2>/dev/null)"
    echo "Workers:      $PYTEST_WORKERS"
    echo "Duration:     ${DURATION_STR:-N/A}"
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
        sed -n '/^## Summary/,/^## /p' "$MATRIX_MD" | head -16
        echo ""
    fi
    echo "── Files ──"
    echo "  Log:      $(realpath "$LOG_FILE" 2>/dev/null || echo "$LOG_FILE")"
    echo "  Failures: $(realpath "$FAIL_FILE" 2>/dev/null || echo "$FAIL_FILE")"
    echo "  Matrix:   $(realpath "$MATRIX_MD" 2>/dev/null || echo "$MATRIX_MD")"
    if [[ -f "$MATRIX_AI_JSON" ]]; then
        echo "  MatrixAI: $(realpath "$MATRIX_AI_JSON" 2>/dev/null || echo "$MATRIX_AI_JSON")"
    fi
    echo "  Summary:  $(realpath "$SUMMARY_FILE" 2>/dev/null || echo "$SUMMARY_FILE")"
    echo "  Diff:     $(realpath "$DIFF_COMMIT" 2>/dev/null || echo "$DIFF_COMMIT")"
    echo "  Phase:    $(realpath "$DIFF_PHASE" 2>/dev/null || echo "$DIFF_PHASE")"
    echo "  Status:   $(realpath "$GIT_STATUS" 2>/dev/null || echo "$GIT_STATUS")"
    if [[ -f "$FOCUSED_NODE_LOG" ]]; then
        echo "  Nodes:    $(realpath "$FOCUSED_NODE_LOG" 2>/dev/null || echo "$FOCUSED_NODE_LOG")"
        echo "            $(grep -c '::' "$FOCUSED_NODE_LOG" 2>/dev/null || echo 0) collected focused nodes"
    fi
    if [[ -f "$FOCUSED_LOG" ]]; then
        echo "  Focused:  $(realpath "$FOCUSED_LOG" 2>/dev/null || echo "$FOCUSED_LOG")"
        echo "            $(grep -E "^[0-9]+ (passed|failed)" "$FOCUSED_LOG" | tail -1)"
    fi
    if [[ -f "$RUNXFAIL_LOG" ]]; then
        echo "  Runxfail: $(realpath "$RUNXFAIL_LOG" 2>/dev/null || echo "$RUNXFAIL_LOG")"
        echo "            $(grep -E "^[0-9]+ (passed|failed)" "$RUNXFAIL_LOG" | tail -1)  <- failures EXPECTED"
    fi
    if [[ -f "$MD5_MANIFEST" ]]; then
        echo "  MD5:      $(realpath "$MD5_MANIFEST" 2>/dev/null || echo "$MD5_MANIFEST")"
    fi
    echo "  Timing:   $TIMING_FILE"
    echo "  TimingPkt:$TIMING_PACKET_SNAPSHOT"
    echo "  Manifest: $(realpath "$REVIEWER_MANIFEST" 2>/dev/null || echo "$REVIEWER_MANIFEST")"
    echo "  Profile:  $(realpath "$REVIEWER_PROFILE" 2>/dev/null || echo "$REVIEWER_PROFILE")"
    echo ""
    echo "── Reviewer package ──"
    echo "  $REVIEWER_ZIP_ABS"
    if [[ "${ADF_REVIEWER_TAR:-1}" != "0" ]]; then
        echo "  $REVIEWER_TAR_ABS   (same file list)"
    fi
    if [[ "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
        echo "  $REVIEWER_BUNDLE_ABS   (plain text, built FROM the zip)"
    fi
    if [[ "${ADF_REVIEWER_TAR:-1}" != "0" || "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
        echo "  digests written to reviewer_digests_${TS}.txt AFTER packaging"
        echo "  (the digest file is deliberately NOT inside the packages —"
        echo "   an archive cannot contain its own hash)"
    fi
    echo "========================================"
} | tee "$SUMMARY_FILE"
SUMMARY_PIPE_EXIT=${PIPESTATUS[0]}
timing_end "summary generation" "$SUMMARY_PIPE_EXIT"

# =============================================================================
# Pre-bundle staging check
# =============================================================================

timing_start "pre-bundle custody checks"
if git rev-parse --is-inside-work-tree &>/dev/null; then
    UNTRACKED_TESTS=$(git status --porcelain tests/ 2>/dev/null | grep "^?? " | grep "\.py$" || true)
    if [[ -n "$UNTRACKED_TESTS" ]] && [[ -z "$ADF_SKIP_STAGING_CHECK" ]]; then
        echo ""
        echo "${RED}${BOLD}❌ BUNDLE BLOCKED — untracked Python files in tests/:${RESET}"
        echo "$UNTRACKED_TESTS" | sed 's/^/    /'
        echo ""
        echo "${YELLOW}These files exist in the working tree but are NOT in any commit.${RESET}"
        echo "${YELLOW}pytest found and ran them (gate above), but the committed test count${RESET}"
        echo "${YELLOW}does not include them. Shipping this bundle is a false positive.${RESET}"
        echo ""
        echo "Stage them before bundling:"
        echo "    git add tests/<file>.py"
        echo ""
        echo "Or, if intentionally local for development, add to .gitignore."
        echo ""
        echo "Override (development only): ADF_SKIP_STAGING_CHECK=1 bash run_tests.sh"
        echo ""
        echo "Test results from this run are saved to:"
        echo "    $LOG_DIR/"
        echo "Bundle .zip was NOT created."
        timing_end "pre-bundle custody checks" 1
        timing_finish_total
        exit 1
    fi
fi

# =============================================================================
# Pre-bundle PHASE_HISTORY <-> git-tag drift check
# =============================================================================

PHASE_HISTORY_FILE="docs/PHASE_HISTORY.md"
if git rev-parse --is-inside-work-tree &>/dev/null \
        && [[ -f "$PHASE_HISTORY_FILE" ]] \
        && [[ -z "$ADF_SKIP_TAG_DRIFT_CHECK" ]]; then

    DOC_END_TAGS=$(grep -ioE 'tag[^`]{0,12}`PHASE_[A-Z0-9_]+_END`' "$PHASE_HISTORY_FILE" \
                       | grep -oE 'PHASE_[A-Z0-9_]+_END' \
                       | sort -u || true)
    REPO_END_TAGS=$(git tag --list 'PHASE_*_END' | sort -u || true)

    MISSING_TAGS=$(comm -23 \
        <(printf '%s\n' "$DOC_END_TAGS") \
        <(printf '%s\n' "$REPO_END_TAGS") | grep -v '^$' || true)

    if [[ -n "$MISSING_TAGS" ]]; then
        {
            echo ""
            echo "⚠️  TAG DRIFT (non-blocking) — PHASE_HISTORY.md declares phase-closure"
            echo "   tags that have no matching git tag:"
            echo "$MISSING_TAGS" | sed 's/^/      /'
            echo "   (These appear as \"tag \`PHASE_..._END\`\" in docs/PHASE_HISTORY.md.)"
            echo "   Resolve by tagging the closure (git tag <PHASE_..._END> <commit>)"
            echo "   or correcting the history entry. Reported, NOT blocked."
        } | tee -a "$SUMMARY_FILE"
        echo "${YELLOW}(Tag drift reported in SUMMARY; bundle still built.)${RESET}"
    fi

    TAG_PLACEMENT_WARNINGS=""
    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        tag_fix=$(printf '%s' "$tag" | grep -oE 'FIX[0-9]+' | head -1 || true)
        [[ -z "$tag_fix" ]] && continue
        subject=$(git log -1 --format='%s' "$tag" 2>/dev/null || true)
        subj_fix=$(printf '%s' "$subject" | grep -oE 'FIX[0-9]+' | head -1 || true)
        if [[ -n "$subj_fix" ]] && [[ "$subj_fix" != "$tag_fix" ]]; then
            TAG_PLACEMENT_WARNINGS+="    $tag -> commit subject mentions $subj_fix (\"$subject\")"$'\n'
        fi
    done <<< "$REPO_END_TAGS"

    if [[ -n "$TAG_PLACEMENT_WARNINGS" ]]; then
        echo ""
        echo "${YELLOW}${BOLD}⚠️  TAG PLACEMENT WARNING — FIX tag(s) may be on the wrong commit:${RESET}"
        printf '%s' "$TAG_PLACEMENT_WARNINGS"
        echo ""
        echo "${YELLOW}The tag name's FIX number does not match the tagged commit's${RESET}"
        echo "${YELLOW}subject. Verify with:  git log --oneline -1 <tag>${RESET}"
        echo "${YELLOW}(Warning only — bundle still created.)${RESET}"
        echo ""
    fi
fi

timing_end "pre-bundle custody checks" 0

# Freeze a packet-safe timing snapshot before packaging.  The final timing file
# continues to record ZIP/TAR/LLMBUNDLE/digest durations outside the packet;
# freezing the snapshot avoids self-referential packet-byte drift.
timing_snapshot_for_packet

# =============================================================================
# Package optimized logical reviewer packet
# Custody invariant: reviewer ZIP is the reviewed container; LLMBUNDLE/3 is
# derived ONLY from that completed ZIP and never rereads worktree payloads.
# =============================================================================

echo ""
echo "--- Packaging reviewer.zip ---"

(
    cd "$PROJECT_ROOT"

    if [[ ! -f "$PACKET_TOOL" ]]; then
        echo "${RED}${BOLD}❌ reviewer packet tool not found: $PACKET_TOOL${RESET}"
        exit 1
    fi

    ZIP_FILES=""
    for f in \
        "$SUMMARY_FILE" \
        "$FAIL_FILE" \
        "$LOG_FILE" \
        "$FOCUSED_LOG" \
        "$RUNXFAIL_LOG" \
        "$FOCUSED_NODE_LOG" \
        "$MD5_MANIFEST" \
        "$MATRIX_MD" \
        "$MATRIX_HTML" \
        "$DIFF_COMMIT" \
        "$DIFF_PHASE" \
        "$GIT_STATUS" \
        "$LOG_DIR/timing_packet_${TS}.txt" \
        "docs/CAPABILITY_MATRIX.md" \
        "docs/CAPABILITY_MATRIX.html" \
        "docs/CAPABILITY_MATRIX.json" \
        "docs/ARCHITECT_DECISIONS.md" \
        "tests/phase_13_79_slot_grid_contract.json" \
        "$MATRIX_AI_JSON"
    do
        [[ -f "$f" ]] && ZIP_FILES="$ZIP_FILES $f"
    done

    # Reviewer source custody: ship the exact changed/focused candidate files
    # too. Every logical reviewer path is stored as an ordinary archive file;
    # exact duplicate relationships remain recorded in the logical manifest.
    for f in $CANDIDATE_FILES; do
        [[ -f "$f" ]] && ZIP_FILES="$ZIP_FILES $f"
    done
    ZIP_FILES=$(printf '%s\n' $ZIP_FILES | sort -u)
    printf '%s\n' $ZIP_FILES > "$REVIEWER_INPUTS"

    if [[ -s "$REVIEWER_INPUTS" ]]; then
        N_LOGICAL=$(grep -c . "$REVIEWER_INPUTS" 2>/dev/null || echo 0)

        timing_start "reviewer manifest / dedup planning"
        MANIFEST_LINE=$(python3 "$PACKET_TOOL" manifest \
            --root "$PROJECT_ROOT" \
            --files-file "$REVIEWER_INPUTS" \
            --manifest "$REVIEWER_MANIFEST" 2>&1)
        MANIFEST_EXIT=$?
        timing_end "reviewer manifest / dedup planning" "$MANIFEST_EXIT"
        if [[ "$MANIFEST_EXIT" -ne 0 ]]; then
            echo "${RED}${BOLD}❌ reviewer logical manifest generation failed${RESET}"
            echo "$MANIFEST_LINE"
            exit 1
        fi
        echo "  logical manifest: $MANIFEST_LINE"
        LOGICAL_MANIFEST_SHA256=$(python3 - "$REVIEWER_MANIFEST" <<'PYMANIFEST'
import json, sys
m=json.load(open(sys.argv[1]))
print(m["logical_manifest_sha256"])
PYMANIFEST
)
        N_PHYSICAL=$(python3 - "$REVIEWER_MANIFEST" <<'PYMANIFEST'
import json, sys
m=json.load(open(sys.argv[1]))
print(m["physical_payload_count"])
PYMANIFEST
)
        N_ALIASES=$(python3 - "$REVIEWER_MANIFEST" <<'PYMANIFEST'
import json, sys
m=json.load(open(sys.argv[1]))
print(m["alias_count"])
PYMANIFEST
)

        timing_start "reviewer ZIP creation"
        python3 "$PACKET_TOOL" zip \
            --root "$PROJECT_ROOT" \
            --manifest "$REVIEWER_MANIFEST" \
            --output "$REVIEWER_ZIP"
        ZIP_EXIT=$?
        timing_end "reviewer ZIP creation" "$ZIP_EXIT"
        if [[ "$ZIP_EXIT" -ne 0 || ! -s "$REVIEWER_ZIP" ]]; then
            echo "${RED}${BOLD}❌ optimized reviewer ZIP creation failed${RESET}"
            exit 1
        fi
        echo "  Reviewer package (zip): $REVIEWER_ZIP_ABS"
        N_ZIP=$(unzip -Z1 "$REVIEWER_ZIP" 2>/dev/null | grep -cv '/$' || echo 0)
        EXPECTED_CONTAINER_MEMBERS=$((N_LOGICAL + 1))  # every logical path + REVIEW_PACKET_MANIFEST.json
        if [[ "$N_ZIP" -ne "$EXPECTED_CONTAINER_MEMBERS" ]]; then
            echo "${RED}${BOLD}❌ zip holds $N_ZIP physical members; expected $EXPECTED_CONTAINER_MEMBERS${RESET}"
            exit 1
        else
            echo "  logical/content: $N_LOGICAL ordinary reviewer files; $N_PHYSICAL unique content payloads; $N_ALIASES exact-duplicate logical paths + manifest"
        fi

        if ! unzip -l "$REVIEWER_ZIP" 2>/dev/null | grep -q 'docs/CAPABILITY_MATRIX\.html$'; then
            echo "${YELLOW}${BOLD}⚠️  $REVIEWER_ZIP missing canonical docs/CAPABILITY_MATRIX.html payload${RESET}"
        fi

        if [[ "${ADF_REVIEWER_TAR:-1}" != "0" ]]; then
            timing_start "reviewer TAR creation"
            python3 "$PACKET_TOOL" tar \
                --root "$PROJECT_ROOT" \
                --manifest "$REVIEWER_MANIFEST" \
                --output "$REVIEWER_TAR"
            TAR_EXIT=$?
            timing_end "reviewer TAR creation" "$TAR_EXIT"
            if [[ "$TAR_EXIT" -eq 0 && -s "$REVIEWER_TAR" ]]; then
                N_TAR=$(tar -tf "$REVIEWER_TAR" 2>/dev/null | grep -cv '/$' || echo 0)
                echo "  Reviewer package (tar): $REVIEWER_TAR_ABS"
                if [[ "$N_TAR" -ne "$EXPECTED_CONTAINER_MEMBERS" ]]; then
                    echo "${RED}${BOLD}❌ tar holds $N_TAR physical members; expected $EXPECTED_CONTAINER_MEMBERS${RESET}"
                    exit 1
                fi
            else
                echo "${RED}${BOLD}❌ optimized reviewer TAR creation failed${RESET}"
                exit 1
            fi
        fi

        if [[ "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
            # Reuse the established shared ZIP -> LLMBUNDLE implementation.
            # The completed reviewer ZIP is the ONLY packet input; no ADF-local
            # converter exists and no worktree payload is reread here.
            BUNDLE_SCRIPT=""
            for candidate in \
                "../scripts/make_llm_bundle.py" \
                "scripts/make_llm_bundle.py" \
                "tests/scripts/make_llm_bundle.py"; do
                [[ -f "$candidate" ]] && BUNDLE_SCRIPT="$candidate" && break
            done
            if [[ -z "$BUNDLE_SCRIPT" ]]; then
                echo "${RED}${BOLD}❌ make_llm_bundle.py not found — cannot build reviewer text representation${RESET}"
                exit 1
            fi

            timing_start "LLMBUNDLE conversion"
            BUNDLE_ERR="$LOG_DIR/reviewer_bundle_stderr_${TS}.txt"
            python3 "$BUNDLE_SCRIPT" "$REVIEWER_ZIP" -o "$REVIEWER_BUNDLE" \
                --allow-delimiters >/dev/null 2>"$BUNDLE_ERR"
            BUNDLE_EXIT=$?
            timing_end "LLMBUNDLE conversion" "$BUNDLE_EXIT"
            if [[ "$BUNDLE_EXIT" -eq 0 && -s "$REVIEWER_BUNDLE" ]]; then
                rm -f "$BUNDLE_ERR"
                N_BUNDLE=$(grep -c '^===== LLMBUNDLE ENTRY BEGIN =====$' "$REVIEWER_BUNDLE" 2>/dev/null || echo 0)
                BUNDLE_VERSION=$(head -1 "$REVIEWER_BUNDLE" 2>/dev/null || true)
                echo "  Reviewer package (text): $REVIEWER_BUNDLE_ABS"
                echo "  LLMBUNDLE representation: $BUNDLE_VERSION"
                if [[ "$N_BUNDLE" -ne "$EXPECTED_CONTAINER_MEMBERS" ]]; then
                    echo "${RED}${BOLD}❌ LLMBUNDLE entries=$N_BUNDLE expected=$EXPECTED_CONTAINER_MEMBERS (logical paths + manifest)${RESET}"
                    exit 1
                fi
                if ! tail -1 "$REVIEWER_BUNDLE" | grep -q '^===== LLMBUNDLE END '; then
                    echo "${RED}${BOLD}❌ LLMBUNDLE has no trailer — truncated${RESET}"
                    exit 1
                fi
            else
                echo "${RED}${BOLD}❌ make_llm_bundle.py FAILED — no text bundle${RESET}"
                if [[ -s "$BUNDLE_ERR" ]]; then
                    echo "${YELLOW}--- bundler stderr ---${RESET}"
                    sed 's/^/    /' "$BUNDLE_ERR"
                    echo "${YELLOW}--- end bundler stderr ---${RESET}"
                fi
                rm -f "$REVIEWER_BUNDLE"
                exit 1
            fi
        fi

        timing_start "reviewer packet round-trip verification"
        VERIFY_ARGS=(
            verify
            --root "$PROJECT_ROOT"
            --manifest "$REVIEWER_MANIFEST"
            --zip "$REVIEWER_ZIP"
        )
        if [[ "${ADF_REVIEWER_TAR:-1}" != "0" ]]; then
            VERIFY_ARGS+=(--tar "$REVIEWER_TAR")
        fi
        if [[ "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
            VERIFY_ARGS+=(--bundle "$REVIEWER_BUNDLE")
        fi
        VERIFY_LINE=$(python3 "$PACKET_TOOL" "${VERIFY_ARGS[@]}" 2>&1)
        VERIFY_EXIT=$?
        timing_end "reviewer packet round-trip verification" "$VERIFY_EXIT"
        if [[ "$VERIFY_EXIT" -ne 0 ]]; then
            echo "${RED}${BOLD}❌ reviewer packet logical round-trip verification failed${RESET}"
            echo "$VERIFY_LINE"
            exit 1
        fi
        echo "  round-trip: $VERIFY_LINE"

        timing_start "reviewer packet profiling"
        PROFILE_ARGS=(
            profile
            --root "$PROJECT_ROOT"
            --manifest "$REVIEWER_MANIFEST"
            --zip "$REVIEWER_ZIP"
            --output "$REVIEWER_PROFILE"
            --json-output "$REVIEWER_PROFILE_JSON"
        )
        if [[ "${ADF_REVIEWER_TAR:-1}" != "0" ]]; then
            PROFILE_ARGS+=(--tar "$REVIEWER_TAR")
        fi
        if [[ "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
            PROFILE_ARGS+=(--bundle "$REVIEWER_BUNDLE")
        fi
        PROFILE_LINE=$(python3 "$PACKET_TOOL" "${PROFILE_ARGS[@]}" 2>&1)
        PROFILE_EXIT=$?
        timing_end "reviewer packet profiling" "$PROFILE_EXIT"
        if [[ "$PROFILE_EXIT" -ne 0 ]]; then
            echo "${RED}${BOLD}❌ reviewer packet profiler failed${RESET}"
            echo "$PROFILE_LINE"
            exit 1
        fi
        echo "  profile: $PROFILE_LINE"
        echo "  profile file: $(realpath "$REVIEWER_PROFILE" 2>/dev/null || echo "$REVIEWER_PROFILE")"

        timing_start "reviewer package digest generation"
        {
            echo "=== reviewer package digests — run $TS ==="
            echo "(logical packet identity is shared across enabled encodings)"
            echo ""
            echo "logical-manifest-sha256 $LOGICAL_MANIFEST_SHA256"
            echo "logical-paths $N_LOGICAL"
            echo "unique-payloads $N_PHYSICAL"
            echo "aliases $N_ALIASES"
            echo "manifest-md5 $(md5sum "$REVIEWER_MANIFEST" | cut -d' ' -f1)"
            echo "manifest-sha256 $(sha256sum "$REVIEWER_MANIFEST" | cut -d' ' -f1)"
            echo "profile-md5 $(md5sum "$REVIEWER_PROFILE" | cut -d' ' -f1)"
            echo "profile-sha256 $(sha256sum "$REVIEWER_PROFILE" | cut -d' ' -f1)"
            echo ""
            for pkg in "$REVIEWER_ZIP" "$REVIEWER_TAR" "$REVIEWER_BUNDLE"; do
                [[ -f "$pkg" ]] || continue
                echo "$(basename "$pkg")"
                echo "  md5     $(md5sum    "$pkg" | cut -d' ' -f1)"
                echo "  sha256  $(sha256sum "$pkg" | cut -d' ' -f1)"
                case "$pkg" in
                    *.zip)
                        echo "  physical-members $(unzip -Z1 "$pkg" 2>/dev/null | grep -cv '/$')"
                        echo "  logical-paths $N_LOGICAL"
                        echo "  logical-manifest-sha256 $LOGICAL_MANIFEST_SHA256" ;;
                    *.tar)
                        echo "  physical-members $(tar -tf "$pkg" 2>/dev/null | grep -cv '/$')"
                        echo "  logical-paths $N_LOGICAL"
                        echo "  logical-manifest-sha256 $LOGICAL_MANIFEST_SHA256" ;;
                    *)
                        echo "  representation $(head -1 "$pkg")"
                        echo "  entries $(grep -c '^===== LLMBUNDLE ENTRY BEGIN =====$' "$pkg" 2>/dev/null || echo 0)"
                        echo "  symlinks $(grep -c '^type: symlink$' "$pkg" 2>/dev/null || echo 0)"
                        echo "  logical-paths $N_LOGICAL"
                        echo "  logical-manifest-sha256 $LOGICAL_MANIFEST_SHA256"
                        echo "  (identity comes from embedded REVIEW_PACKET_MANIFEST.json and verified round-trip)" ;;
                esac
                echo ""
            done
        } > "$REVIEWER_DIGESTS"
        timing_end "reviewer package digest generation" 0

        echo ""
        cat "$REVIEWER_DIGESTS" | sed 's/^/  /'
    fi
)
PACKAGE_EXIT=$?
if [[ "$PACKAGE_EXIT" -ne 0 ]]; then
    echo ""
    echo "${RED}${BOLD}❌ Reviewer package construction/custody failed (exit $PACKAGE_EXIT)${RESET}"
    timing_finish_total
    printf '[%s] END run_tests.sh — package failure\n' "$(date +"%Y-%m-%d %H:%M:%S %z")"
    echo "Timing artifact: $TIMING_FILE"
    echo "Packet timing snapshot: $TIMING_PACKET_SNAPSHOT"
    exit "$PACKAGE_EXIT"
fi

# =============================================================================
# Working-tree warnings
# =============================================================================

if git rev-parse --is-inside-work-tree &>/dev/null; then
    if [[ -n "$(git status --porcelain -- docs/CAPABILITY_MATRIX.md 2>/dev/null)" ]]; then
        echo ""
        echo "${YELLOW}${BOLD}⚠️  docs/CAPABILITY_MATRIX.md is modified but not staged.${RESET}"
        echo "${YELLOW}   If committing this phase, include it in the commit:${RESET}"
        echo "       git add docs/CAPABILITY_MATRIX.md"
    fi
fi

# =============================================================================
# Final
# =============================================================================

timing_finish_total
printf '[%s] END run_tests.sh\n' "$(date +"%Y-%m-%d %H:%M:%S %z")"
echo "Timing artifact: $TIMING_FILE"
echo "Packet timing snapshot: $TIMING_PACKET_SNAPSHOT"

echo ""
if [[ $FAILED -gt 0 || $ERRORS -gt 0 ]]; then
    echo "${RED}${BOLD}❌ Some tests failed${RESET}"
    exit 1
else
    echo "${GREEN}${BOLD}✅ All tests passed${RESET}"
    exit 0
fi
