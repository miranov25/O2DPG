#!/bin/bash
#
# run_benchmark.sh - Run ALL AliasDataFrame benchmarks with pytest-style output
#
# Runs all benchmarks and provides unified pass/fail summary.
# Similar to pytest but for performance benchmarks.
#
# Usage:
#   ./run_benchmark.sh                    # Run all benchmarks
#   ./run_benchmark.sh --quick            # Quick mode (smaller data)
#   ./run_benchmark.sh --synthetic-only   # Only synthetic benchmarks (no ROOT files)
#   ./run_benchmark.sh --generate-data    # Generate synthetic ROOT file first
#   ./run_benchmark.sh --save-baseline    # Save results as new baseline
#   ./run_benchmark.sh --compare-baseline # Compare against baseline (detect regressions)
#   ./run_benchmark.sh --threshold 15     # Set regression threshold (default: 20%)
#
# Exit Codes:
#   0 - All benchmarks passed
#   1 - Some benchmarks failed or regression detected (when --strict)
#   0 - Always (default, failures reported but not fatal)
#

# Force C locale for numeric formatting (use . as decimal separator, not ,)
export LC_NUMERIC=C

# Strict mode would exit with 1 on failure
STRICT_MODE=false

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"
OUTPUT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SYNTHETIC_DATA="${SCRIPT_DIR}/synthetic_data.root"

# Benchmark configuration
QUICK_MODE=""
SYNTHETIC_ONLY=false
GENERATE_DATA=false
VERBOSE=false

# Baseline comparison configuration
SAVE_BASELINE=false
COMPARE_BASELINE=false
THRESHOLD=20
BASELINE_FILE="${SCRIPT_DIR}/baseline.json"
PROFILE_FLAG=""
FULL_FLAG=""

# Results tracking
declare -a BENCHMARK_NAMES
declare -a BENCHMARK_STATUS
declare -a BENCHMARK_TIMES
declare -a BENCHMARK_MESSAGES

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
MERGED_JSON=""

# =============================================================================
# Helper Functions
# =============================================================================

# Get current time with sub-second precision (works on macOS and Linux)
get_time() {
    python3 -c "import time; print(f'{time.time():.6f}')"
}

# Calculate elapsed time
calc_elapsed() {
    local start="$1"
    local end="$2"
    python3 -c "print(f'{$end - $start:.2f}')"
}

log_result() {
    local name="$1"
    local status="$2"
    local time="$3"
    local message="$4"
    
    BENCHMARK_NAMES+=("$name")
    BENCHMARK_STATUS+=("$status")
    BENCHMARK_TIMES+=("$time")
    BENCHMARK_MESSAGES+=("$message")
    
    case "$status" in
        PASSED) ((TOTAL_PASSED++)) ;;
        FAILED) ((TOTAL_FAILED++)) ;;
        SKIPPED) ((TOTAL_SKIPPED++)) ;;
    esac
}

print_status() {
    local name="$1"
    local status="$2"
    local time="$3"
    
    local color=""
    local symbol=""
    
    case "$status" in
        PASSED)
            color="\033[32m"  # Green
            symbol="✓"
            ;;
        FAILED)
            color="\033[31m"  # Red
            symbol="✗"
            ;;
        SKIPPED)
            color="\033[33m"  # Yellow
            symbol="○"
            ;;
    esac
    
    local reset="\033[0m"
    
    if [[ -n "$time" ]]; then
        printf "${color}${symbol} %-40s %s${reset} (%.2fs)\n" "$name" "$status" "$time"
    else
        printf "${color}${symbol} %-40s %s${reset}\n" "$name" "$status"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE="--quick"
            shift
            ;;
        --synthetic-only)
            SYNTHETIC_ONLY=true
            shift
            ;;
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --profile)
            PROFILE_FLAG="--profile"
            shift
            ;;
        --full)
            PROFILE_FLAG="--profile"
            FULL_FLAG="--full"
            COMPARE_BASELINE=true
            shift
            ;;
        --save-baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --compare-baseline)
            COMPARE_BASELINE=true
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick            Quick mode (smaller data, faster)"
            echo "  --synthetic-only   Only run synthetic benchmarks (no ROOT file needed)"
            echo "  --generate-data    Generate synthetic ROOT file before running"
            echo "  --strict           Exit with code 1 if any benchmark fails or regression detected"
            echo "  --verbose, -v      Show detailed output"
            echo "  --profile          Save profiler output (.prof and .txt) for analysis"
            echo "  --full             Full analysis: profiling + baseline comparison + history archive"
            echo "  --output DIR       Output directory (default: benchmarks/results)"
            echo ""
            echo "Regression Detection:"
            echo "  --save-baseline    Save current results as new baseline"
            echo "  --compare-baseline Compare results against baseline (detect regressions)"
            echo "  --threshold PCT    Regression threshold percentage (default: 20)"
            echo "  --baseline FILE    Baseline file path (default: benchmarks/baseline.json)"
            echo ""
            echo "Benchmarks:"
            echo "  benchmark_performance.py          Synthetic data tests (always runs)"
            echo "  benchmark_materialize_aliases.py  Alias & subframe materialization (always runs)"
            echo "  benchmark_read_tree.py            ROOT file read tests (needs data)"
            echo "  benchmark_subframe.py             Subframe tests (needs data)"
            echo "  benchmark_parallel.py             Parallel scaling tests (needs data)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run all benchmarks"
            echo "  $0 --save-baseline              # Run and save as baseline"
            echo "  $0 --compare-baseline --strict  # CI mode: fail on regression"
            echo "  $0 --compare-baseline --threshold 15  # Custom threshold"
            echo "  $0 --full                       # Full run with profiling and history"
            echo "  $0 --profile                    # Run with profiler output only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "==================== BENCHMARK SESSION STARTS ===================="
echo "Timestamp:  $(date -Iseconds)"
echo "Host:       $(hostname)"
echo "Python:     $(python3 --version 2>&1 | head -1)"
echo "Directory:  ${SCRIPT_DIR}"
echo "Mode:       ${QUICK_MODE:-full}"
echo ""

cd "${PARENT_DIR}"

# =============================================================================
# Generate Synthetic Data (if requested or needed)
# =============================================================================

if [[ "$GENERATE_DATA" = true ]] || { [[ "$SYNTHETIC_ONLY" = false ]] && [[ ! -f "$SYNTHETIC_DATA" ]]; }; then
    echo "--- Generating synthetic data ---"
    
    START_TIME=$(get_time)
    
    if python3 "${SCRIPT_DIR}/generate_synthetic_data.py" --output "$SYNTHETIC_DATA" --verify 2>&1; then
        END_TIME=$(get_time)
        ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")
        log_result "generate_synthetic_data" "PASSED" "$ELAPSED" ""
        print_status "generate_synthetic_data" "PASSED" "$ELAPSED"
    else
        log_result "generate_synthetic_data" "FAILED" "" "Failed to generate"
        print_status "generate_synthetic_data" "FAILED" ""
        
        if [[ "$SYNTHETIC_ONLY" = false ]]; then
            echo "WARNING: Cannot run ROOT-based benchmarks without data"
            SYNTHETIC_ONLY=true
        fi
    fi
    echo ""
fi

# =============================================================================
# Benchmark 1: Performance (Synthetic)
# Tests: AliasDataFrame creation, alias definition, schema operations,
#        materialization, compression, and export. No I/O - pure computation.
# =============================================================================

echo "--- benchmark_performance.py ${QUICK_MODE} ---"
if [[ "$VERBOSE" = true ]]; then
    echo "    Tests: create_adf, add_aliases, validate_schema, materialize, compress, export"
fi

START_TIME=$(get_time)
PERF_JSON="${OUTPUT_DIR}/benchmark_performance_${TIMESTAMP}.json"

if [[ "$VERBOSE" = true ]]; then
    OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_performance.py" ${QUICK_MODE} --json "$PERF_JSON" 2>&1)
    PERF_STATUS=$?
    echo "$OUTPUT"
else
    OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_performance.py" ${QUICK_MODE} --json "$PERF_JSON" --quiet 2>&1)
    PERF_STATUS=$?
fi

END_TIME=$(get_time)
ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")

if [[ $PERF_STATUS -eq 0 ]]; then
    # Check if all sub-benchmarks passed
    if command -v jq &> /dev/null && [[ -f "$PERF_JSON" ]]; then
        ALL_PASSED=$(jq -r '.all_passed' "$PERF_JSON" 2>/dev/null || echo "true")
        if [[ "$ALL_PASSED" = "true" ]]; then
            log_result "benchmark_performance.py" "PASSED" "$ELAPSED" ""
            print_status "benchmark_performance.py" "PASSED" "$ELAPSED"
        else
            log_result "benchmark_performance.py" "FAILED" "$ELAPSED" "Threshold exceeded"
            print_status "benchmark_performance.py" "FAILED" "$ELAPSED"
        fi
    else
        log_result "benchmark_performance.py" "PASSED" "$ELAPSED" ""
        print_status "benchmark_performance.py" "PASSED" "$ELAPSED"
    fi
else
    log_result "benchmark_performance.py" "FAILED" "$ELAPSED" "Exit code $PERF_STATUS"
    print_status "benchmark_performance.py" "FAILED" "$ELAPSED"
fi

echo ""

# =============================================================================
# Benchmark 2: Materialize Aliases (Synthetic)
# Tests: Alias DAG materialization with subframe joins.
#        Compares fill_mode='safe' vs 'direct' performance.
# =============================================================================

echo "--- benchmark_materialize_aliases.py ${QUICK_MODE} ${FULL_FLAG} ${PROFILE_FLAG} ---"
if [[ "$VERBOSE" = true ]]; then
    echo "    Tests: simple (no subframe), safe (full NaN/Inf checks), direct (fast mode)"
fi

START_TIME=$(get_time)
MATERIALIZE_JSON="${OUTPUT_DIR}/benchmark_materialize_aliases_${TIMESTAMP}.json"

if [[ "$VERBOSE" = true ]]; then
    OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_materialize_aliases.py" ${QUICK_MODE} ${FULL_FLAG} ${PROFILE_FLAG} --json "$MATERIALIZE_JSON" 2>&1)
    MAT_STATUS=$?
    echo "$OUTPUT"
else
    OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_materialize_aliases.py" ${QUICK_MODE} ${FULL_FLAG} ${PROFILE_FLAG} --json "$MATERIALIZE_JSON" --quiet 2>&1)
    MAT_STATUS=$?
fi

END_TIME=$(get_time)
ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")

if [[ $MAT_STATUS -eq 0 ]]; then
    log_result "benchmark_materialize_aliases.py" "PASSED" "$ELAPSED" ""
    print_status "benchmark_materialize_aliases.py" "PASSED" "$ELAPSED"
else
    log_result "benchmark_materialize_aliases.py" "FAILED" "$ELAPSED" "Exit code $MAT_STATUS"
    print_status "benchmark_materialize_aliases.py" "FAILED" "$ELAPSED"
fi

echo ""

# =============================================================================
# Benchmark 3: Read Tree (needs ROOT file)
# Tests: ROOT file I/O with uproot, threaded branch reading,
#        dtype preservation (float16), and subframe loading.
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_read_tree.py ---"
    if [[ "$VERBOSE" = true ]]; then
        echo "    Tests: read_tree speed, dtype preservation, subframe auto-detection"
    fi
    
    ENTRIES_ARG=""
    if [[ -n "$QUICK_MODE" ]]; then
        ENTRIES_ARG="50000"  # Smaller for quick mode
    fi
    
    START_TIME=$(get_time)
    
    if [[ "$VERBOSE" = true ]]; then
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_read_tree.py" "$SYNTHETIC_DATA" tree $ENTRIES_ARG 2>&1)
        READ_STATUS=$?
        echo "$OUTPUT"
    else
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_read_tree.py" "$SYNTHETIC_DATA" tree $ENTRIES_ARG 2>&1)
        READ_STATUS=$?
    fi
    
    END_TIME=$(get_time)
    ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")
    
    if [[ $READ_STATUS -eq 0 ]]; then
        log_result "benchmark_read_tree.py" "PASSED" "$ELAPSED" ""
        print_status "benchmark_read_tree.py" "PASSED" "$ELAPSED"
    else
        log_result "benchmark_read_tree.py" "FAILED" "$ELAPSED" "Exit code $READ_STATUS"
        print_status "benchmark_read_tree.py" "FAILED" "$ELAPSED"
    fi
    
    echo ""
else
    log_result "benchmark_read_tree.py" "SKIPPED" "" "No ROOT file"
    print_status "benchmark_read_tree.py" "SKIPPED" ""
    echo ""
fi

# =============================================================================
# Benchmark 4: Subframe (needs ROOT file)
# Tests: Subframe registration, join correctness (invariant validation),
#        alias materialization with subframe lookups, missing key statistics.
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_subframe.py ---"
    if [[ "$VERBOSE" = true ]]; then
        echo "    Tests: subframe loading, join correctness, alias speed, missing key stats"
    fi
    
    ENTRIES_ARG=""
    if [[ -n "$QUICK_MODE" ]]; then
        ENTRIES_ARG="--entries 50000"
    fi
    
    START_TIME=$(get_time)
    
    if [[ "$VERBOSE" = true ]]; then
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_subframe.py" "$SYNTHETIC_DATA" --treename tree $ENTRIES_ARG 2>&1)
        SUB_STATUS=$?
        echo "$OUTPUT"
    else
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_subframe.py" "$SYNTHETIC_DATA" --treename tree $ENTRIES_ARG 2>&1)
        SUB_STATUS=$?
    fi
    
    END_TIME=$(get_time)
    ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")
    
    if [[ $SUB_STATUS -eq 0 ]]; then
        log_result "benchmark_subframe.py" "PASSED" "$ELAPSED" ""
        print_status "benchmark_subframe.py" "PASSED" "$ELAPSED"
    else
        log_result "benchmark_subframe.py" "FAILED" "$ELAPSED" "Exit code $SUB_STATUS"
        print_status "benchmark_subframe.py" "FAILED" "$ELAPSED"
    fi
    
    echo ""
else
    log_result "benchmark_subframe.py" "SKIPPED" "" "No ROOT file"
    print_status "benchmark_subframe.py" "SKIPPED" ""
    echo ""
fi

# =============================================================================
# Benchmark 5: Parallel (needs ROOT file)
# Tests: read_tree scaling with num_workers (1, 2, 4, 8).
#        Identifies optimal worker count and detects hangs/instability.
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_parallel.py ---"
    if [[ "$VERBOSE" = true ]]; then
        echo "    Tests: parallel read scaling, optimal num_workers, timeout detection"
    fi
    
    PARALLEL_ARGS="--repeats 2 --timeout 30"
    if [[ -n "$QUICK_MODE" ]]; then
        PARALLEL_ARGS="--repeats 1 --timeout 15 --max-workers 4"
    fi
    
    # Save JSON output for baseline comparison
    PARALLEL_JSON="${OUTPUT_DIR}/benchmark_parallel_${TIMESTAMP}.json"
    
    START_TIME=$(get_time)
    
    if [[ "$VERBOSE" = true ]]; then
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_parallel.py" "$SYNTHETIC_DATA" --treename tree $PARALLEL_ARGS --json "$PARALLEL_JSON" 2>&1)
        PAR_STATUS=$?
        echo "$OUTPUT"
    else
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_parallel.py" "$SYNTHETIC_DATA" --treename tree $PARALLEL_ARGS --json "$PARALLEL_JSON" --quiet 2>&1)
        PAR_STATUS=$?
    fi
    
    END_TIME=$(get_time)
    ELAPSED=$(calc_elapsed "$START_TIME" "$END_TIME")
    
    if [[ $PAR_STATUS -eq 0 ]]; then
        log_result "benchmark_parallel.py" "PASSED" "$ELAPSED" ""
        print_status "benchmark_parallel.py" "PASSED" "$ELAPSED"
    else
        log_result "benchmark_parallel.py" "FAILED" "$ELAPSED" "Exit code $PAR_STATUS"
        print_status "benchmark_parallel.py" "FAILED" "$ELAPSED"
    fi
    
    echo ""
else
    log_result "benchmark_parallel.py" "SKIPPED" "" "No ROOT file"
    print_status "benchmark_parallel.py" "SKIPPED" ""
    echo ""
fi

# =============================================================================
# Baseline Operations
# =============================================================================

# Save baseline if requested
if [[ "$SAVE_BASELINE" = true ]]; then
    echo "--- Saving Baseline ---"
    
    if python3 "${SCRIPT_DIR}/baseline_utils.py" merge "${OUTPUT_DIR}" "${BASELINE_FILE}" 2>&1; then
        echo ""
        echo -e "\033[32m✓ Baseline saved to: ${BASELINE_FILE}\033[0m"
        echo "  Commit this file to track performance over time."
    else
        echo -e "\033[31m✗ Failed to save baseline\033[0m"
        ((TOTAL_FAILED++))
    fi
    echo ""
fi

# Compare against baseline if requested
REGRESSION_DETECTED=false
if [[ "$COMPARE_BASELINE" = true ]]; then
    echo "--- Comparing Against Baseline ---"
    
    if [[ -f "$BASELINE_FILE" ]]; then
        # First, merge current results into a single file for comparison
        MERGED_JSON="${OUTPUT_DIR}/benchmark_merged_${TIMESTAMP}.json"
        
        echo "Merging current results..."
        if python3 "${SCRIPT_DIR}/baseline_utils.py" merge "${OUTPUT_DIR}" "$MERGED_JSON" --timestamp "$TIMESTAMP" 2>&1; then
            echo "Merged results to: $MERGED_JSON"
        else
            echo "Warning: Could not merge results, comparing individual files"
        fi
        
        COMPARE_ARGS="--threshold $THRESHOLD"
        if [[ "$STRICT_MODE" = true ]]; then
            COMPARE_ARGS="$COMPARE_ARGS --strict"
        fi
        
        # Export comparison results
        COMPARISON_JSON="${OUTPUT_DIR}/comparison_${TIMESTAMP}.json"
        
        # Compare merged results (or latest if merge failed)
        if [[ -f "$MERGED_JSON" ]]; then
            python3 "${SCRIPT_DIR}/baseline_utils.py" compare "$MERGED_JSON" "$BASELINE_FILE" $COMPARE_ARGS --json "$COMPARISON_JSON"
        else
            python3 "${SCRIPT_DIR}/baseline_utils.py" compare "${OUTPUT_DIR}" "$BASELINE_FILE" $COMPARE_ARGS --latest --json "$COMPARISON_JSON"
        fi
        COMPARE_STATUS=$?
        
        if [[ $COMPARE_STATUS -eq 1 ]]; then
            REGRESSION_DETECTED=true
            ((TOTAL_FAILED++))
            log_result "regression_check" "FAILED" "" "Regression detected"
        elif [[ $COMPARE_STATUS -eq 0 ]]; then
            log_result "regression_check" "PASSED" "" "Within threshold"
        else
            echo -e "\033[33m⚠️  Comparison error (exit code $COMPARE_STATUS)\033[0m"
        fi
    else
        echo -e "\033[33m⚠️  No baseline.json found at: ${BASELINE_FILE}\033[0m"
        echo "   Run with --save-baseline to create one."
        log_result "regression_check" "SKIPPED" "" "No baseline"
        ((TOTAL_SKIPPED++))
    fi
    echo ""
fi

# =============================================================================
# Archive to History (always runs)
# Archives every benchmark run for time series tracking
# =============================================================================

echo "--- Archiving to History ---"

# Ensure we have a merged JSON file
if [[ ! -f "$MERGED_JSON" ]]; then
    MERGED_JSON="${OUTPUT_DIR}/benchmark_merged_${TIMESTAMP}.json"
    echo "Merging results for archive..."
    python3 "${SCRIPT_DIR}/baseline_utils.py" merge "${OUTPUT_DIR}" "$MERGED_JSON" --timestamp "$TIMESTAMP" 2>&1 || true
fi

if [[ -f "$MERGED_JSON" ]]; then
    if python3 "${SCRIPT_DIR}/baseline_utils.py" archive "$MERGED_JSON" --history-dir "${OUTPUT_DIR}/history" 2>&1; then
        echo -e "\033[32m✓ Results archived to history\033[0m"
    else
        echo -e "\033[33m⚠️  Failed to archive to history (non-fatal)\033[0m"
    fi
else
    echo -e "\033[33m⚠️  No merged results to archive\033[0m"
fi
echo ""

# =============================================================================
# Summary (pytest-style)
# =============================================================================

echo "==================== BENCHMARK SUMMARY ===================="
echo ""

# Calculate total time
TOTAL_TIME=0
for time in "${BENCHMARK_TIMES[@]}"; do
    if [[ -n "$time" ]]; then
        TOTAL_TIME=$(python3 -c "print(f'{$TOTAL_TIME + $time:.2f}')")
    fi
done

# Status line (pytest-style)
STATUS_PARTS=""
if [[ $TOTAL_PASSED -gt 0 ]]; then
    STATUS_PARTS="${TOTAL_PASSED} passed"
fi
if [[ $TOTAL_FAILED -gt 0 ]]; then
    [[ -n "$STATUS_PARTS" ]] && STATUS_PARTS="${STATUS_PARTS}, "
    STATUS_PARTS="${STATUS_PARTS}${TOTAL_FAILED} failed"
fi
if [[ $TOTAL_SKIPPED -gt 0 ]]; then
    [[ -n "$STATUS_PARTS" ]] && STATUS_PARTS="${STATUS_PARTS}, "
    STATUS_PARTS="${STATUS_PARTS}${TOTAL_SKIPPED} skipped"
fi

# Color the final line
if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo -e "\033[31m==================== ${STATUS_PARTS} in ${TOTAL_TIME}s ====================\033[0m"
else
    echo -e "\033[32m==================== ${STATUS_PARTS} in ${TOTAL_TIME}s ====================\033[0m"
fi

echo ""

# Save summary to file
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
{
    echo "Benchmark Summary - $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo ""
    for i in "${!BENCHMARK_NAMES[@]}"; do
        printf "%-40s %s\n" "${BENCHMARK_NAMES[$i]}" "${BENCHMARK_STATUS[$i]}"
    done
    echo ""
    echo "Total: ${STATUS_PARTS}"
} > "$SUMMARY_FILE"

echo "Results saved to: ${OUTPUT_DIR}/"

# =============================================================================
# Exit Code
# =============================================================================

if [[ "$STRICT_MODE" = true ]]; then
    if [[ $TOTAL_FAILED -gt 0 ]]; then
        exit 1
    fi
    if [[ "$REGRESSION_DETECTED" = true ]]; then
        exit 1
    fi
fi

exit 0
