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
#   test_failures_<ts>.log         Failures only
#   CAPABILITY_MATRIX_<ts>.md      Auto-generated matrix snapshot
#   diff_last_commit_<ts>.txt      Uncommitted diff + last commit diff
#   diff_to_phase_<ts>.txt         Diff since PHASE_BEGIN_AliasDataFrame tag
#   git_status_<ts>.txt            Working tree state (git status --porcelain)
#   reviewer_<ts>.zip              Review package

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
  --quick      Run tests only (no matrix, no diffs)
  --matrix     Generate matrix only (skip tests)
  --verbose    Verbose pytest output

Environment:
  PYTEST_WORKERS=N    Parallel workers (default: 12)

Output:
  test_logs/SUMMARY_<ts>.txt            Test summary
  test_logs/CAPABILITY_MATRIX_<ts>.md   Feature matrix
  test_logs/diff_last_commit_<ts>.txt   Uncommitted + HEAD~1 diffs
  test_logs/diff_to_phase_<ts>.txt      Diff since PHASE_BEGIN tag
  test_logs/git_status_<ts>.txt         Working tree state snapshot
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
SUMMARY_FILE="$LOG_DIR/SUMMARY_${TS}.txt"
DIFF_COMMIT="$LOG_DIR/diff_last_commit_${TS}.txt"
DIFF_PHASE="$LOG_DIR/diff_to_phase_${TS}.txt"
GIT_STATUS="$LOG_DIR/git_status_${TS}.txt"
# Focused (phase) suite, logged SEPARATELY and shipped in the packet.
# GPT30 P2-1, round 6: the CRR quoted a focused-suite count that the packet
# contained no evidence for, so a reviewer could verify the full run and not
# the number the CRR actually led with. Override the pattern per phase:
#   FOCUSED_TESTS="tests/test_phase_13_77_*.py" bash run_tests.sh
FOCUSED_TESTS="${FOCUSED_TESTS:-tests/test_phase_13_76_draw_path_characterization.py}"
FOCUSED_LOG="$LOG_DIR/test_focused_${TS}.log"
REVIEWER_ZIP="$LOG_DIR/reviewer_${TS}.zip"
# Absolute path, computed HERE rather than at packaging time, so the summary
# block below can print it BEFORE the file exists. `realpath -m` resolves a
# not-yet-created path; the fallback covers platforms without it.
REVIEWER_ZIP_ABS="$(realpath -m "$REVIEWER_ZIP" 2>/dev/null || echo "$PROJECT_ROOT/$REVIEWER_ZIP")"

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

    # Combined diff: uncommitted work (reviewer's primary interest)
    # + last committed change (for context).
    # Phase 13.16.DF fix: was 'git diff HEAD~1..HEAD' which misses uncommitted work.
    # --relative makes paths cwd-relative (drawer.py instead of UTILS/.../drawer.py).
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

    # Phase diff — AliasDataFrame tag names (this project's own tags)
    PHASE_TAG=""
    for tag in PHASE_BEGIN_AliasDataFrame PHASE_BEGIN_ADF; do
        if git rev-parse --verify "$tag" &>/dev/null; then
            PHASE_TAG="$tag"
            break
        fi
    done

    if [[ -n "$PHASE_TAG" ]]; then
        # Phase 13.16.DF fix: was '$PHASE_TAG..HEAD' which misses uncommitted work.
        # 'git diff $PHASE_TAG' without range includes working tree.
        # --relative scopes paths to cwd.
        git diff --relative "$PHASE_TAG" -- . > "$DIFF_PHASE" 2>/dev/null || true
        echo "  Phase tag: $PHASE_TAG"
    else
        echo "(No PHASE_BEGIN_* tag found — searched: PHASE_BEGIN_AliasDataFrame, PHASE_BEGIN_ADF)" > "$DIFF_PHASE"
        echo "  ⚠️  No phase tag — create with: source scripts/phase_tag.sh && phase_begin <id>"
    fi
    
    # Working tree snapshot — reviewers use this to verify repo state
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
        # Phase 13.27 Commit 2 FIX1.FIX1 follow-up — Sonet50 reviewer-packet
        # gap: tag listing was missing from reviewer.zip across multiple
        # phases, forcing reviewers to round-trip the architect for tag
        # verification. Append PHASE_* tags (newest first, last 10) and
        # the last 5 commits with decorations.
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

    # Counts come from the SAME JSON the capability matrix consumes (single source of
    # truth — no terminal scraping). pytest-json-report records collection errors (e.g. a
    # module that fails to import) in collectors[].outcome, NOT summary.error, so ERRORS
    # sums both. See BUG_20260630_run_tests for why the scrape/two-source design was wrong.
    read PASSED FAILED ERRORS SKIPPED < <(python3 - "$JSON_REPORT" << 'PYCOUNT'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print("0 0 0 0"); sys.exit()
s = r.get("summary", {})
passed = s.get("passed", 0); failed = s.get("failed", 0); skipped = s.get("skipped", 0)
# test-level errors (summary.error) + collection errors (collectors with non-passed outcome)
errors = s.get("error", 0)
errors += sum(1 for c in r.get("collectors", []) if c.get("outcome") in ("failed", "error"))
print(passed, failed, errors, skipped)
PYCOUNT
)
    PASSED=${PASSED:-0}; FAILED=${FAILED:-0}; ERRORS=${ERRORS:-0}; SKIPPED=${SKIPPED:-0}

    # Extract failures
    grep -E "^FAILED |^ERROR " "$LOG_FILE" 2>/dev/null | sort -u > "$FAIL_FILE" || true

    # --- focused (phase) suite, its own log so the CRR's headline number is
    # verifiable from the packet rather than taken on trust.
    if compgen -G "$FOCUSED_TESTS" > /dev/null 2>&1; then
        echo "--- Running focused suite: $FOCUSED_TESTS ---"
        python3 -m pytest $FOCUSED_TESTS -q --tb=short 2>&1 | tee "$FOCUSED_LOG"
        FOCUSED_LINE=$(grep -E "^[0-9]+ (passed|failed)" "$FOCUSED_LOG" | tail -1)
        echo "  focused: ${FOCUSED_LINE:-<no summary line>}"
    else
        echo "--- No focused suite matched: $FOCUSED_TESTS ---"
    fi

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
        "scripts/generate_capability_matrix.py" \
        "tests/scripts/generate_capability_matrix.py"; do
        [[ -f "$candidate" ]] && MATRIX_SCRIPT="$candidate" && break
    done

    if [[ -n "$MATRIX_SCRIPT" ]]; then
        MATRIX_ARGS=""
        [[ -f "$JSON_REPORT" ]] && MATRIX_ARGS="--test-results $JSON_REPORT"

        # Phase 13.49.DF §9 D-B: derive --phase from the most recent _END tag.
        # NB: the glob is 'PHASE_[0-9]*_DF*_END' (no underscore between _DF and *).
        # The natural-looking pattern '_DF_*_END' would require >=1 char between
        # _DF_ and _END and silently miss the plain _END tags (most phases).
        # Verified at v1.2 implementation against the real tag set.
        PHASE_FOR_MATRIX=$(git tag --list 'PHASE_[0-9]*_DF*_END' --sort=-creatordate 2>/dev/null | head -1)
        if [[ -n "$PHASE_FOR_MATRIX" ]]; then
            MATRIX_ARGS="$MATRIX_ARGS --phase $PHASE_FOR_MATRIX"
        else
            MATRIX_ARGS="$MATRIX_ARGS --phase unknown"
        fi

        python3 "$MATRIX_SCRIPT" $MATRIX_ARGS 2>&1 || \
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
# BUG_ADF_20260705_matrix_html — HTML matrix (dfdraw parity) + env stamps
# Architect decisions 2026-07-05: phase = latest ADF BEGIN tag (in the script);
# off-gate: red banner only (#2 not needed); md env stamp (#3 yes); BUG commit (#4).
# =============================================================================
HTML_SCRIPT=""
for cand in "scripts/generate_matrix_html.py" "tests/scripts/generate_matrix_html.py"; do
    [[ -f "$cand" ]] && HTML_SCRIPT="$cand" && break
done
if [[ -n "$HTML_SCRIPT" ]]; then
    python3 "$HTML_SCRIPT" --log "$LOG_FILE" \
        --output "docs/CAPABILITY_MATRIX.html" \
        --snapshot "$LOG_DIR/CAPABILITY_MATRIX_${TS}.html" 2>&1 \
        || echo "${YELLOW}⚠️  generate_matrix_html.py failed (non-blocking)${RESET}"
else
    echo "${YELLOW}⚠️  generate_matrix_html.py not found — docs/CAPABILITY_MATRIX.html not regenerated${RESET}"
fi
# Environment stamp into the md (idempotent; one line, grep-guarded)
ENV_STAMP="*Environment: $(hostname) · $(uname -s)-$(uname -m) · Python $(python3 -c 'import platform; print(platform.python_version())') · stamped by run_tests.sh*"
for mdf in "docs/CAPABILITY_MATRIX.md" "$MATRIX_MD"; do
    if [[ -f "$mdf" ]] && ! grep -q '^\*Environment: ' "$mdf"; then
        printf '\n%s\n' "$ENV_STAMP" >> "$mdf"
    fi
done

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
        sed -n '/^## Summary/,/^## /p' "$MATRIX_MD" | head -12
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
    if [[ -f "$FOCUSED_LOG" ]]; then
        echo "  Focused:  $(realpath "$FOCUSED_LOG" 2>/dev/null || echo "$FOCUSED_LOG")"
        echo "            $(grep -E "^[0-9]+ (passed|failed)" "$FOCUSED_LOG" | tail -1)"
    fi
    echo ""
    echo "── Reviewer package ──"
    echo "  $REVIEWER_ZIP_ABS"
    echo "========================================"
} | tee "$SUMMARY_FILE"

# =============================================================================
# Pre-bundle staging check (inherited from the shared runner)
# =============================================================================
# Catches a class of bugs where new test files are created in the working tree,
# pytest finds them and reports "all tests pass", but the file is untracked and
# never enters the commit. The bundle then ships a gate (e.g., 822/0/0) that
# doesn't match the committed test count (e.g., 817).
#
# Rationale: a new test file in the working tree that is untracked will be run
# by pytest (gate looks green) yet is absent from the commit — a false-positive bundle.
#
# This check blocks BUNDLE creation when any *.py file in tests/ is untracked.
# Test results are still saved to test_logs/ (already written above) — only
# the .zip artifact is prevented. That's the artifact reviewers consume.
#
# Override: ADF_SKIP_STAGING_CHECK=1 bash run_tests.sh
#   (for development runs where untracked test files are intentional).

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
# Pre-bundle PHASE_HISTORY <-> git-tag drift check (inherited from the shared runner)
# =============================================================================
# Catches a class of bugs where docs/PHASE_HISTORY.md and the actual git tags
# disagree about phase closures. Two real incidents motivated this (both found
# (generic check; not tied to any one project's history):
#
#   (1) DOCUMENTED-BUT-UNTAGGED: PHASE_HISTORY.md recorded
#       "FIX2 ... tag PHASE_13_25_DF_FIX2_END" but no such tag existed in the
#       repo (the FIX2 commit existed; the tag was never created). The history
#       claimed a closure the repo could not prove.
#
#   (2) MISPLACED FIX TAG: PHASE_13_25_DF_FIX1_END was sitting on the FIX2
#       commit (subject "Phase 13.25.DF FIX2: ...") instead of the FIX1 commit.
#       Anyone checking out the FIX1 tag would have gotten FIX2 code.
#
# Check (1) is a BLOCK: any PHASE_*_END string mentioned in PHASE_HISTORY.md
#   that is NOT a real git tag stops bundle creation. This is unambiguous —
#   the doc asserts a closure the repo doesn't have.
# Check (2) is a WARNING (heuristic): for each repo PHASE_*_FIX<N>_END tag,
#   if the tagged commit's subject line mentions a DIFFERENT FIX<M> (M != N),
#   the tag is likely on the wrong commit. Warn (don't block) because commit
#   subjects are free-form; promote to a block later if it proves reliable.
#
# Note: the reverse direction (repo tags NOT cited in PHASE_HISTORY.md) is NOT
# flagged — many tags (GB/ADF/older phases) are intentionally not cited by
# exact string in the PHASE_HISTORY narrative. That direction is noise.
#
# Override: ADF_SKIP_TAG_DRIFT_CHECK=1 bash run_tests.sh
#   (for development runs before PHASE_HISTORY.md has been updated).

PHASE_HISTORY_FILE="docs/PHASE_HISTORY.md"
if git rev-parse --is-inside-work-tree &>/dev/null \
        && [[ -f "$PHASE_HISTORY_FILE" ]] \
        && [[ -z "$ADF_SKIP_TAG_DRIFT_CHECK" ]]; then

    # _END tags claimed in the history doc vs _END tags actually in the repo.
    # Phase 13.48 grep tightening — scope to tag-DECLARATION context only.
    # A phase closure is *claimed* by writing "tag `PHASE_X_END`" or
    # "**Tag:** `PHASE_X_END`". The earlier loose grep matched any PHASE_*_END
    # token anywhere in the doc, including prose mentions (e.g. an example tag
    # in a sentence describing this very check) -> false positives.
    DOC_END_TAGS=$(grep -ioE 'tag[^`]{0,12}`PHASE_[A-Z0-9_]+_END`' "$PHASE_HISTORY_FILE" \
                       | grep -oE 'PHASE_[A-Z0-9_]+_END' \
                       | sort -u || true)
    REPO_END_TAGS=$(git tag --list 'PHASE_*_END' | sort -u || true)

    # (1) Documented-but-untagged: lines in DOC not in REPO.
    MISSING_TAGS=$(comm -23 \
        <(printf '%s\n' "$DOC_END_TAGS") \
        <(printf '%s\n' "$REPO_END_TAGS") | grep -v '^$' || true)

    if [[ -n "$MISSING_TAGS" ]]; then
        # NON-BLOCKING (Phase 13.48 design — committed as df3057a3): doc<->tag drift
        # is a documentation-hygiene signal, not a test result. Hard-blocking the
        # bundle on a heuristic grep of a prose file is fragile and creates override
        # pressure (the bypass would become invisible -> dead-weight check). Instead:
        # WARN loudly AND record the warning in SUMMARY so the drift itself travels
        # in reviewer.zip for architect/reviewers to see and resolve. Bundle still
        # builds. A false negative here is low-harm — a missed warning, not a false
        # sense of a passed gate. (Restored at 13.49 implementation after a regression.)
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

    # (2) Misplaced FIX tag (heuristic warning, non-blocking).
    TAG_PLACEMENT_WARNINGS=""
    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        # Extract FIX<N> from the tag name, if present.
        tag_fix=$(printf '%s' "$tag" | grep -oE 'FIX[0-9]+' | head -1 || true)
        [[ -z "$tag_fix" ]] && continue
        subject=$(git log -1 --format='%s' "$tag" 2>/dev/null || true)
        # Find any FIX<M> mentioned in the tagged commit's subject.
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
        "$MATRIX_MD" \
        "$DIFF_COMMIT" \
        "$DIFF_PHASE" \
        "$GIT_STATUS" \
        "docs/CAPABILITY_MATRIX.md" \
        "docs/CAPABILITY_MATRIX.html"
    do
        [[ -f "$f" ]] && ZIP_FILES="$ZIP_FILES $f"
    done

    if [[ -n "$ZIP_FILES" ]]; then
        zip -q "$REVIEWER_ZIP" $ZIP_FILES 2>/dev/null || true
        # ABSOLUTE path. It used to print "test_logs/reviewer_<ts>.zip", which
        # is unusable for copy/paste from a terminal whose cwd the reader does
        # not share — and the zip is the ONE artifact of this script that
        # leaves the machine.
        echo "  Reviewer package: $REVIEWER_ZIP_ABS"

        # (inherited) assert the HTML matrix
        # made it into the zip. Three consecutive phases (13.49, 13.49-FIX1,
        # 13.50) shipped reviewer.zip without docs/CAPABILITY_MATRIX.html
        # because the file list above forgot the .html line. Mechanical
        # guard so voluntary discipline isn't relied on for a fourth time.
        if ! unzip -l "$REVIEWER_ZIP" 2>/dev/null | grep -q 'docs/CAPABILITY_MATRIX\.html$'; then
            echo "${YELLOW}${BOLD}⚠️  $REVIEWER_ZIP missing docs/CAPABILITY_MATRIX.html — reviewers cannot navigate the rendered matrix${RESET}"
        fi
    fi
)

# =============================================================================
# Working-tree warnings (Phase 13.27 Commit 2 FIX1.FIX1 follow-up)
# =============================================================================
# After 6 reproductions across Phases 13.25 / 13.26 / 13.28 / 13.30 / 13.32 /
# 13.27-FIX1, docs/CAPABILITY_MATRIX.md is the chronic miss in commits — it is
# regenerated by this script (above) and then forgotten in the next git add.
# Warn loudly so the coder doesn't ship a phase tag with the matrix orphaned
# in the working tree. Purely informational — does not affect exit code.

if git rev-parse --is-inside-work-tree &>/dev/null; then
    # --porcelain prefix: ' M' (unstaged), 'M ' (staged), 'MM' (both),
    # '??' (untracked). We warn on ANY unstaged modification of the matrix.
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
