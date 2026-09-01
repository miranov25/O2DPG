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

# Don't exit on test failures
# set -e

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
SUMMARY_FILE="$LOG_DIR/SUMMARY_${TS}.txt"
DIFF_COMMIT="$LOG_DIR/diff_last_commit_${TS}.txt"
DIFF_PHASE="$LOG_DIR/diff_to_phase_${TS}.txt"
GIT_STATUS="$LOG_DIR/git_status_${TS}.txt"

# Focused (phase) suite, logged SEPARATELY and shipped in the packet.
# Override the pattern per phase, e.g.:
#   FOCUSED_TESTS="tests/test_phase_13_77_*.py" bash run_tests.sh
FOCUSED_TESTS="${FOCUSED_TESTS:-tests/test_phase_13_76_draw_path_characterization.py}"
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
    echo "  Last commit diff: $(realpath "$DIFF_COMMIT" 2>/dev/null || echo "$DIFF_COMMIT")"

    PHASE_TAG=""
    for tag in PHASE_BEGIN_AliasDataFrame PHASE_BEGIN_ADF; do
        if git rev-parse --verify "$tag" &>/dev/null; then
            PHASE_TAG="$tag"
            break
        fi
    done

    if [[ -n "$PHASE_TAG" ]]; then
        git diff --relative "$PHASE_TAG" -- . > "$DIFF_PHASE" 2>/dev/null || true
        echo "  Phase tag: $PHASE_TAG"
    else
        echo "(No PHASE_BEGIN_* tag found — searched: PHASE_BEGIN_AliasDataFrame, PHASE_BEGIN_ADF)" > "$DIFF_PHASE"
        echo "  ⚠️  No phase tag — create with: source scripts/phase_tag.sh && phase_begin <id>"
    fi

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

    python3 -m pytest tests/ \
        $PYTEST_VERBOSITY \
        --tb=short \
        -n "$PYTEST_WORKERS" \
        --continue-on-collection-errors \
        --json-report --json-report-file="$JSON_REPORT" \
        2>&1 | tee "$LOG_FILE"
    TEST_EXIT=${PIPESTATUS[0]}

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
        python3 -m pytest $FOCUSED_TESTS --collect-only -q 2>/dev/null \
            | grep '::' | sort -u > "$FOCUSED_NODE_LOG" || true
        FOCUSED_NODE_COUNT=$(grep -c '::' "$FOCUSED_NODE_LOG" 2>/dev/null || echo 0)
        echo "  focused nodes: $FOCUSED_NODE_COUNT"

        mapfile -t FOCUSED_NODES < "$FOCUSED_NODE_LOG"
        if [[ ${#FOCUSED_NODES[@]} -eq 0 ]]; then
            echo "ERROR: focused collection produced zero executable node IDs" >&2
            return 1 2>/dev/null || false
        fi

        echo "--- Running exact collected focused node set ---"
        python3 -m pytest "${FOCUSED_NODES[@]}" -n "$PYTEST_WORKERS" -q --tb=short \
            2>&1 | tee "$FOCUSED_LOG"
        FOCUSED_LINE=$(grep -E "^[0-9]+ (passed|failed)" "$FOCUSED_LOG" | tail -1)
        echo "  focused: ${FOCUSED_LINE:-<no summary line>}"

        echo "--- Recording --runxfail evidence for the exact same focused node set ---"
        python3 -m pytest "${FOCUSED_NODES[@]}" --runxfail -n "$PYTEST_WORKERS" \
            -q --tb=line -p no:warnings > "$RUNXFAIL_LOG" 2>&1 || true
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

    # MATRIX-PHASE-1: this is the AliasDataFrame matrix.  Resolve phase
    # provenance only from ADF phase tags; never reuse dfdraw *_DF*_END tags.
    PHASE_FOR_MATRIX=$(git tag --merged HEAD --list 'PHASE_*_ADF_BEGIN' --sort=-creatordate 2>/dev/null | head -1)
    if [[ -z "$PHASE_FOR_MATRIX" ]]; then
        PHASE_FOR_MATRIX=$(git tag --merged HEAD --list 'PHASE_BEGIN_AliasDataFrame' --sort=-creatordate 2>/dev/null | head -1)
    fi
    if [[ -z "$PHASE_FOR_MATRIX" ]]; then
        PHASE_FOR_MATRIX=$(git tag --merged HEAD --list 'PHASE_BEGIN_ADF' --sort=-creatordate 2>/dev/null | head -1)
    fi
    [[ -n "$PHASE_FOR_MATRIX" ]] || PHASE_FOR_MATRIX="PHASE_13_76_ADF"

    if [[ -n "$MATRIX_SCRIPT" && -n "$MATRIX_JSON" ]]; then
        python3 "$MATRIX_SCRIPT" \
            --test-results "$MATRIX_JSON" \
            --phase "$PHASE_FOR_MATRIX" 2>&1 || \
            echo "⚠️  Capability matrix Markdown generation had errors"

        [[ -f "docs/CAPABILITY_MATRIX.md" ]] && \
            cp "docs/CAPABILITY_MATRIX.md" "$MATRIX_MD"
    elif [[ -z "$MATRIX_SCRIPT" ]]; then
        echo "⚠️  generate_capability_matrix.py not found"
        echo "    Expected at: scripts/generate_capability_matrix.py"
    else
        echo "⚠️  pytest JSON report not found; Capability Matrix not regenerated"
    fi

    if [[ -n "$HTML_SCRIPT" && -n "$MATRIX_JSON" ]]; then
        # MATRIX-PARITY-1 + HTML-PRESENTATION-1:
        # keep the established interactive HTML renderer, but feed it the
        # shared normalized semantic model via the same pytest JSON evidence.
        python3 "$HTML_SCRIPT" \
            --test-results "$MATRIX_JSON" \
            --output "docs/CAPABILITY_MATRIX.html" \
            --snapshot "$MATRIX_HTML" \
            --phase "$PHASE_FOR_MATRIX" 2>&1 || \
            echo "⚠️  Capability matrix HTML generation had errors"
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

# =============================================================================
# Summary
# =============================================================================

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

# =============================================================================
# Pre-bundle staging check
# =============================================================================

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

# =============================================================================
# Package reviewer.zip
# =============================================================================

echo ""
echo "--- Packaging reviewer.zip ---"

(
    cd "$PROJECT_ROOT"
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
        "docs/CAPABILITY_MATRIX.md" \
        "docs/CAPABILITY_MATRIX.html"
    do
        [[ -f "$f" ]] && ZIP_FILES="$ZIP_FILES $f"
    done

    if [[ -n "$ZIP_FILES" ]]; then
        N_WANT=$(printf '%s\n' $ZIP_FILES | grep -c .)

        zip -q "$REVIEWER_ZIP" $ZIP_FILES 2>/dev/null || true
        echo "  Reviewer package (zip): $REVIEWER_ZIP_ABS"

        N_ZIP=$(unzip -Z1 "$REVIEWER_ZIP" 2>/dev/null | grep -cv '/$' || echo 0)
        if [[ "$N_ZIP" -ne "$N_WANT" ]]; then
            echo "${RED}${BOLD}❌ zip holds $N_ZIP of $N_WANT intended files${RESET}"
        fi

        if ! unzip -l "$REVIEWER_ZIP" 2>/dev/null | grep -q 'docs/CAPABILITY_MATRIX\.html$'; then
            echo "${YELLOW}${BOLD}⚠️  $REVIEWER_ZIP missing docs/CAPABILITY_MATRIX.html — reviewers cannot navigate the rendered matrix${RESET}"
        fi

        # ---------------------------------------------------------------
        # Companion .tar — SAME file list, SAME relative paths, SAME step.
        # Two packets built from two lists is finding D-1 in a new costume,
        # so the list is computed once above and reused verbatim here.
        # Uncompressed on purpose: the architect asked for .tar, and it
        # removes gzip as a second thing that can fail in a reader.
        # ---------------------------------------------------------------
        if [[ "${ADF_REVIEWER_TAR:-1}" != "0" ]]; then
            if command -v tar >/dev/null 2>&1; then
                # --sort=name and fixed ownership keep the member order and
                # uid/gid stable across machines.  Not required for review,
                # cheap insurance if anyone ever diffs two tars.
                tar --sort=name --owner=0 --group=0 --numeric-owner \
                    -cf "$REVIEWER_TAR" $ZIP_FILES 2>/dev/null \
                    || tar -cf "$REVIEWER_TAR" $ZIP_FILES 2>/dev/null || true

                if [[ -f "$REVIEWER_TAR" ]]; then
                    N_TAR=$(tar -tf "$REVIEWER_TAR" 2>/dev/null | grep -cv '/$' || echo 0)
                    echo "  Reviewer package (tar): $REVIEWER_TAR_ABS"
                    if [[ "$N_TAR" -ne "$N_WANT" ]]; then
                        echo "${RED}${BOLD}❌ tar holds $N_TAR of $N_WANT intended files${RESET}"
                    fi
                    if [[ "$N_TAR" -ne "$N_ZIP" ]]; then
                        echo "${RED}${BOLD}❌ PACKAGE MISMATCH: zip=$N_ZIP tar=$N_TAR — the two packets are NOT the same content${RESET}"
                    else
                        echo "  content check: zip and tar both hold $N_TAR files"
                    fi
                else
                    echo "${YELLOW}⚠️  tar companion was requested but not produced${RESET}"
                fi
            else
                echo "${YELLOW}⚠️  tar not available; zip only${RESET}"
            fi
        fi

        # ---------------------------------------------------------------
        # Plain-text LLMBUNDLE companion — SAME content, no container.
        # Built from "$REVIEWER_ZIP" and NOT from a directory: the zip IS
        # the reviewed file list, so the bundle cannot drift from it.
        # Pointing the bundler at a working directory sweeps up .git,
        # coverage output and test data — measured at 2386 entries and
        # 298 MB for dfdraw, which no reviewer runtime can read.
        # ---------------------------------------------------------------
        if [[ "${ADF_REVIEWER_BUNDLE:-1}" != "0" ]]; then
            BUNDLE_SCRIPT=""
            for candidate in \
                "../scripts/make_llm_bundle.py" \
                "scripts/make_llm_bundle.py" \
                "tests/scripts/make_llm_bundle.py"; do
                [[ -f "$candidate" ]] && BUNDLE_SCRIPT="$candidate" && break
            done

            # --allow-delimiters is REQUIRED here, not optional.  The packet
            # always contains diff_to_phase/diff_last_commit, and any diff that
            # touches make_llm_bundle.py necessarily quotes that script's own
            # DELIMITERS constants.  Refusing on that would make the packet
            # permanently un-bundleable.  The bundle stays unambiguous for any
            # reader that uses the "bytes:" length prefix, which is what the
            # header/manifest/trailer exist to support.
            BUNDLE_ERR="$LOG_DIR/reviewer_bundle_stderr_${TS}.txt"
            if [[ -z "$BUNDLE_SCRIPT" ]]; then
                echo "${YELLOW}⚠️  make_llm_bundle.py not found (../scripts/, scripts/); no text bundle${RESET}"
            elif python3 "$BUNDLE_SCRIPT" "$REVIEWER_ZIP" -o "$REVIEWER_BUNDLE" \
                        --allow-delimiters >/dev/null 2>"$BUNDLE_ERR" \
                    && [[ -s "$REVIEWER_BUNDLE" ]]; then
                rm -f "$BUNDLE_ERR"
                N_BUNDLE=$(grep -c '^===== LLMBUNDLE ENTRY BEGIN =====$' "$REVIEWER_BUNDLE" 2>/dev/null || echo 0)
                echo "  Reviewer package (text): $REVIEWER_BUNDLE_ABS"
                if [[ "$N_BUNDLE" -ne "$N_ZIP" ]]; then
                    echo "${RED}${BOLD}❌ PACKAGE MISMATCH: zip=$N_ZIP bundle=$N_BUNDLE — not the same content${RESET}"
                else
                    echo "  content check: zip and text bundle both hold $N_BUNDLE files"
                fi
                if ! tail -1 "$REVIEWER_BUNDLE" | grep -q '^===== LLMBUNDLE END '; then
                    echo "${RED}${BOLD}❌ text bundle has no trailer — it is truncated${RESET}"
                fi
            else
                echo "${RED}${BOLD}❌ make_llm_bundle.py FAILED — no text bundle (zip/tar unaffected)${RESET}"
                if [[ -s "$BUNDLE_ERR" ]]; then
                    echo "${YELLOW}--- bundler stderr ---${RESET}"
                    sed 's/^/    /' "$BUNDLE_ERR"
                    echo "${YELLOW}--- end bundler stderr ---${RESET}"
                fi
                rm -f "$REVIEWER_BUNDLE"
            fi
        fi

        # ---------------------------------------------------------------
        # Digests of ALL packages, in one file, written AFTER packaging.
        # Deliberately NOT inside either archive: an archive cannot carry
        # its own hash, and a digest file that is inside one packet but
        # describes both is exactly the custody confusion to avoid.
        # Declare BOTH lines in the CRR; reviewers state which they opened.
        # ---------------------------------------------------------------
        {
            echo "=== reviewer package digests — run $TS ==="
            echo "(declare BOTH in the CRR; a reviewer must state which they opened)"
            echo ""
            for pkg in "$REVIEWER_ZIP" "$REVIEWER_TAR" "$REVIEWER_BUNDLE"; do
                [[ -f "$pkg" ]] || continue
                echo "$(basename "$pkg")"
                echo "  md5     $(md5sum    "$pkg" | cut -d' ' -f1)"
                echo "  sha256  $(sha256sum "$pkg" | cut -d' ' -f1)"
                case "$pkg" in
                    *.zip)
                        echo "  files   $(unzip -Z1 "$pkg" 2>/dev/null | grep -cv '/$')" ;;
                    *.tar)
                        echo "  files   $(tar -tf "$pkg" 2>/dev/null | grep -cv '/$')" ;;
                    *)
                        echo "  files   $(grep -c '^===== LLMBUNDLE ENTRY BEGIN =====$' "$pkg" 2>/dev/null || echo 0)"
                        echo "  entries $(grep -m1 '^entries: ' "$pkg" 2>/dev/null | cut -d' ' -f2)"
                        echo "  manifest-sha256 $(grep -m1 '^manifest-sha256: ' "$pkg" 2>/dev/null | cut -d' ' -f2)"
                        echo "  (a Source-Read claim must quote entries + manifest-sha256"
                        echo "   from BOTH the header and the trailer; no trailer = truncated)" ;;
                esac
                echo ""
            done
        } > "$REVIEWER_DIGESTS"

        echo ""
        cat "$REVIEWER_DIGESTS" | sed 's/^/  /'
    fi
)

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

echo ""
if [[ $FAILED -gt 0 || $ERRORS -gt 0 ]]; then
    echo "${RED}${BOLD}❌ Some tests failed${RESET}"
    exit 1
else
    echo "${GREEN}${BOLD}✅ All tests passed${RESET}"
    exit 0
fi
