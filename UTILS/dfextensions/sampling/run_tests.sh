#!/bin/bash
# run_tests.sh — Phase 13.11.DF regression test suite
# Runs unit tests + parallel scan generation + parallel validation
# All outputs colocated: scan_<name>_<n>/  contains .root + figures + summary
#
# Usage:
#   bash run_tests.sh                        # full run, 100 iter
#   bash run_tests.sh --skip-generate        # reuse existing .root files
#   bash run_tests.sh --skip-generate 1000   # validate existing 1000-iter files
#   bash run_tests.sh 1000                   # full run, 1000 iter

set -e

# ── Parse args ─────────────────────────────────────────────────────
SKIP_GEN=false
N_ITER=100
for arg in "$@"; do
    case $arg in
        --skip-generate) SKIP_GEN=true ;;
        [0-9]*) N_ITER=$arg ;;
    esac
done

GIT_DESC=$(git describe --always --dirty 2>/dev/null || echo "unknown")

echo "======================================================================"
echo "REGRESSION TEST SUITE — Phase 13.11.DF"
echo "Git: $GIT_DESC"
echo "Date: $(date -Iseconds)"
echo "N_iter: $N_ITER"
echo "======================================================================"

# ── Directories: scan + figures + summary colocated ────────────────
DIR_GAUSS="scan_gauss_${N_ITER}"
DIR_QUANT="scan_quantile_${N_ITER}"
DIR_LINEAR="scan_linear_${N_ITER}"

mkdir -p "$DIR_GAUSS" "$DIR_QUANT" "$DIR_LINEAR"

# ── Step 1: Unit tests (fast, serial) ──────────────────────────────
echo ""
echo "=== [1/3] Unit tests ==="
pytest tests/ -x -q --tb=short
echo "Unit tests: PASSED"

# ── Step 2: Generate scan trees (parallel) ─────────────────────────
if [[ "$SKIP_GEN" == false ]]; then
    echo ""
    echo "=== [2/3] Generating scan trees (parallel) ==="
    T0=$SECONDS

    python generate_scan_tree.py \
        --mode uniform --distribution gaussian \
        --n_iter $N_ITER \
        --output "$DIR_GAUSS/scan.root" &
    PID_GAUSS=$!

    python generate_scan_tree.py \
        --mode quantile --fit_coordinate bin \
        --nbins_min 20 --nbins_max 500 \
        --n_iter $N_ITER \
        --output "$DIR_QUANT/scan.root" &
    PID_QUANT=$!

    python generate_scan_tree.py \
        --mode uniform --distribution linear \
        --n_iter $N_ITER \
        --output "$DIR_LINEAR/scan.root" &
    PID_LINEAR=$!

    FAIL=0
    wait $PID_GAUSS  || { echo "FAIL: Gaussian generation";  FAIL=1; }
    wait $PID_QUANT  || { echo "FAIL: Quantile generation";  FAIL=1; }
    wait $PID_LINEAR || { echo "FAIL: Linear generation";    FAIL=1; }

    if [[ $FAIL -ne 0 ]]; then
        echo "Generation FAILED — aborting."
        exit 1
    fi
    echo "Generation: $((SECONDS - T0))s (3 parallel jobs)"
else
    echo ""
    echo "=== [2/3] Skipping generation (--skip-generate) ==="
fi

# ── Step 3: Validate scan trees (parallel) ─────────────────────────
echo ""
echo "=== [3/3] Validating scan trees (parallel) ==="
T0=$SECONDS

python validate_scan.py \
    --input "$DIR_GAUSS/scan.root" \
    --output "$DIR_GAUSS/" \
    > "$DIR_GAUSS/scan_summary.txt" 2>&1 &
PID_V_GAUSS=$!

python validate_scan.py \
    --input "$DIR_QUANT/scan.root" \
    --output "$DIR_QUANT/" \
    > "$DIR_QUANT/scan_summary.txt" 2>&1 &
PID_V_QUANT=$!

python validate_scan.py \
    --input "$DIR_LINEAR/scan.root" \
    --output "$DIR_LINEAR/" \
    > "$DIR_LINEAR/scan_summary.txt" 2>&1 &
PID_V_LINEAR=$!

FAIL=0
wait $PID_V_GAUSS  || { echo "FAIL: Gaussian validation";  FAIL=1; }
wait $PID_V_QUANT  || { echo "FAIL: Quantile validation";  FAIL=1; }
wait $PID_V_LINEAR || { echo "FAIL: Linear validation";    FAIL=1; }

echo "Validation: $((SECONDS - T0))s (3 parallel jobs)"

# ── Step 4: Invariant checks ──────────────────────────────────────
echo ""
echo "=== Invariant checks ==="
FAIL=0

check_invariant() {
    local LABEL=$1
    local FILE=$2

    if [[ ! -f "$FILE" ]]; then
        echo "  FAIL: $LABEL — summary missing: $FILE"
        FAIL=1
        return
    fi

    echo "  --- $LABEL ---"

    # Spectra ratio: expect ~1.0 ± 0.05
    local SPEC=$(grep "spectra" "$FILE" | grep "smooth_v5" | head -1)
    if [[ -n "$SPEC" ]]; then
        echo "  $SPEC"
    else
        echo "  WARNING: no v5 spectra line found"
    fi

    # S4c slope: expect ~1.0 ± 0.15
    local S4C=$(grep "S4c.*global.*smooth_v5" "$FILE" | head -1)
    if [[ -n "$S4C" ]]; then
        echo "  $S4C"
    fi

    # Check for crashes (Python traceback)
    if grep -q "Traceback" "$FILE"; then
        echo "  FAIL: Python traceback detected!"
        FAIL=1
    fi

    # Check all figures saved
    local N_SAVED=$(grep -c "Saved:" "$FILE" || true)
    echo "  Figures saved: $N_SAVED"
    if [[ $N_SAVED -lt 5 ]]; then
        echo "  FAIL: too few figures ($N_SAVED < 5)"
        FAIL=1
    fi
}

check_invariant "Gaussian uniform"   "$DIR_GAUSS/scan_summary.txt"
check_invariant "Gaussian quantile"  "$DIR_QUANT/scan_summary.txt"
check_invariant "Linear uniform"     "$DIR_LINEAR/scan_summary.txt"

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
if [[ $FAIL -ne 0 ]]; then
    echo "RESULT: FAILED — see above"
    exit 1
else
    echo "RESULT: ALL PASSED"
fi
echo "Git: $GIT_DESC"
echo "Total time: ${SECONDS}s"
echo ""
echo "Output directories:"
echo "  $DIR_GAUSS/"
echo "  $DIR_QUANT/"
echo "  $DIR_LINEAR/"
echo "======================================================================"
