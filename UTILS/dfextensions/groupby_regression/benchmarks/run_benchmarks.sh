#!/bin/bash
# run_benchmarks.sh - Run groupby_regression benchmark suite
#
# Usage:
#   source ./benchmarks/run_benchmarks.sh              # Quick suite (default)
#   source ./benchmarks/run_benchmarks.sh --full       # Full suite
#   source ./benchmarks/run_benchmarks.sh --json       # Save JSON results
#   source ./benchmarks/run_benchmarks.sh --review     # Generate review artifacts
#
# Phase 12.14b.GB

# Don't use set -e with source - track errors manually instead
# set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Inherit library paths for OpenMP (critical on macOS)
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Parse arguments
QUICK_FLAG="--quick"
JSON_FLAG=""
FULL_MODE=false
REVIEW_MODE=false

for arg in "$@"; do
    case $arg in
        --full)
            QUICK_FLAG=""
            FULL_MODE=true
            shift
            ;;
        --json)
            JSON_FLAG="--json"
            shift
            ;;
        --review)
            REVIEW_MODE=true
            shift
            ;;
        *)
            ;;
    esac
done

# Review mode: generate both log and JSON for reviewers
if [ "$REVIEW_MODE" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    REVIEW_DIR="${SCRIPT_DIR}/../benchmark_review_${TIMESTAMP}"
    mkdir -p "$REVIEW_DIR"
    
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  GENERATING REVIEW ARTIFACTS                                       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Output directory: $REVIEW_DIR"
    echo ""
    
    # Kernel benchmarks (standalone)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: Standalone Kernel Benchmarks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPT_DIR}/bench_groupby_regression_kernels.py" --quick \
        --json "$REVIEW_DIR/kernel_results.json" \
        2>&1 | tee "$REVIEW_DIR/kernel_results.log"
    
    echo ""
    
    # Memory benchmarks (standalone)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: Standalone Memory Benchmarks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPT_DIR}/bench_groupby_regression_memory.py" --quick \
        --json "$REVIEW_DIR/memory_results.json" \
        2>&1 | tee "$REVIEW_DIR/memory_results.log"
    
    echo ""
    
    # BF Runner (integration test)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: Benchmark Framework Runner (Integration)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # Note: BF runner saves results automatically to $BENCHMARK_PREFIX
    # We just capture the console output for review
    python -m dfextensions.benchmarks.runner \
        --subproject groupby_regression \
        --suite quick \
        2>&1 | tee "$REVIEW_DIR/bf_results.log" || echo "BF runner failed (see log for details)"
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  REVIEW ARTIFACTS GENERATED                                        ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Files for reviewers:"
    ls -la "$REVIEW_DIR/"
    echo ""
    echo "Submit these files for review:"
    echo "  Standalone:"
    echo "    - kernel_results.log / kernel_results.json"
    echo "    - memory_results.log / memory_results.json"
    echo "  BF Integration:"
    echo "    - bf_results.log (proves BF discovery works)"
    
    return 0 2>/dev/null || exit 0
fi

# Timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/../benchmark_results_${TIMESTAMP}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  GROUPBY REGRESSION BENCHMARK SUITE                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Date: $(date)"
echo "Mode: $([ "$FULL_MODE" = true ] && echo "FULL" || echo "QUICK")"
echo ""

# Track overall status
ALL_PASS=true

# -----------------------------------------------------------------------------
# Kernel Benchmarks
# -----------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Kernel Benchmarks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$JSON_FLAG" ]; then
    mkdir -p "$RESULTS_DIR"
    python "${SCRIPT_DIR}/bench_groupby_regression_kernels.py" $QUICK_FLAG --json "$RESULTS_DIR/kernel_results.json" || ALL_PASS=false
else
    python "${SCRIPT_DIR}/bench_groupby_regression_kernels.py" $QUICK_FLAG || ALL_PASS=false
fi

echo ""

# -----------------------------------------------------------------------------
# Memory Benchmarks
# -----------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Memory Benchmarks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ITER_FLAG=""
if [ "$FULL_MODE" = true ]; then
    ITER_FLAG="--iterations 50"
else
    ITER_FLAG="--quick"
fi

if [ -n "$JSON_FLAG" ]; then
    python "${SCRIPT_DIR}/bench_groupby_regression_memory.py" $ITER_FLAG --json "$RESULTS_DIR/memory_results.json" || ALL_PASS=false
else
    python "${SCRIPT_DIR}/bench_groupby_regression_memory.py" $ITER_FLAG || ALL_PASS=false
fi

echo ""

# -----------------------------------------------------------------------------
# V5 API Benchmarks (optional in full mode)
# -----------------------------------------------------------------------------
if [ "$FULL_MODE" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: V5 API Benchmarks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -n "$JSON_FLAG" ]; then
        python "${SCRIPT_DIR}/bench_v5.py" --json "$RESULTS_DIR/v5_results.json" || ALL_PASS=false
    else
        python "${SCRIPT_DIR}/bench_v5.py" || ALL_PASS=false
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ "$ALL_PASS" = true ]; then
    echo "║  ✓ ALL BENCHMARKS PASSED                                          ║"
else
    echo "║  ✗ SOME BENCHMARKS FAILED                                         ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"

if [ -n "$JSON_FLAG" ]; then
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    ls -la "$RESULTS_DIR/"
fi

# Exit with appropriate code (use return if sourced, exit if run)
if [ "$ALL_PASS" = true ]; then
    echo ""
    # Don't exit/return - just set status
    true
else
    echo ""
    # Don't exit/return - just set status  
    false
fi
