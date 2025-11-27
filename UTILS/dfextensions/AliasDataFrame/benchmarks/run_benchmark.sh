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
#
# Exit Codes:
#   0 - All benchmarks passed
#   1 - Some benchmarks failed (when --strict)
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

# Results tracking
declare -a BENCHMARK_NAMES
declare -a BENCHMARK_STATUS
declare -a BENCHMARK_TIMES
declare -a BENCHMARK_MESSAGES

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

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
            echo "  --strict           Exit with code 1 if any benchmark fails"
            echo "  --verbose, -v      Show detailed output"
            echo "  --output DIR       Output directory (default: benchmarks/results)"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Benchmarks:"
            echo "  benchmark_performance.py   Synthetic data tests (always runs)"
            echo "  benchmark_read_tree.py     ROOT file read tests (needs data)"
            echo "  benchmark_subframe.py      Subframe tests (needs data)"
            echo "  benchmark_parallel.py      Parallel scaling tests (needs data)"
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
# =============================================================================

echo "--- benchmark_performance.py ${QUICK_MODE} ---"

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
# Benchmark 2: Read Tree (needs ROOT file)
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_read_tree.py ---"
    
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
# Benchmark 3: Subframe (needs ROOT file)
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_subframe.py ---"
    
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
# Benchmark 4: Parallel (needs ROOT file)
# =============================================================================

if [[ "$SYNTHETIC_ONLY" = false ]] && [[ -f "$SYNTHETIC_DATA" ]]; then
    echo "--- benchmark_parallel.py ---"
    
    PARALLEL_ARGS="--repeats 2 --timeout 30"
    if [[ -n "$QUICK_MODE" ]]; then
        PARALLEL_ARGS="--repeats 1 --timeout 15 --max-workers 4"
    fi
    
    START_TIME=$(get_time)
    
    if [[ "$VERBOSE" = true ]]; then
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_parallel.py" "$SYNTHETIC_DATA" --treename tree $PARALLEL_ARGS 2>&1)
        PAR_STATUS=$?
        echo "$OUTPUT"
    else
        OUTPUT=$(python3 "${SCRIPT_DIR}/benchmark_parallel.py" "$SYNTHETIC_DATA" --treename tree $PARALLEL_ARGS --quiet 2>&1)
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

if [[ "$STRICT_MODE" = true ]] && [[ $TOTAL_FAILED -gt 0 ]]; then
    exit 1
fi

exit 0
