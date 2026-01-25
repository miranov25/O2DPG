#!/bin/bash
# Run debug notebooks and verify they execute without errors
# Phase 13.6.G: Debug notebook validation
#
# Usage:
#   ./run_notebooks.sh              # Run all (quiet)
#   ./run_notebooks.sh -v           # Run all (verbose - show errors)
#   ./run_notebooks.sh 08a          # Run specific notebook
#   ./run_notebooks.sh -v 08a       # Run specific notebook (verbose)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/tmp/notebook_tests"
VERBOSE=0

# Parse options
while getopts "v" opt; do
    case $opt in
        v) VERBOSE=1 ;;
    esac
done
shift $((OPTIND-1))

mkdir -p "$OUTPUT_DIR"

# Notebooks to test (P0 debug notebooks)
NOTEBOOKS=(
    "08a_debug_scalar.ipynb"
    "08b_debug_1d.ipynb"
    "08c_debug_2d.ipynb"
    "08f_invariance_visual.ipynb"
)

# Filter if argument provided
if [ -n "$1" ]; then
    NOTEBOOKS=($(printf '%s\n' "${NOTEBOOKS[@]}" | grep "$1"))
    if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
        echo "No notebooks matching '$1'"
        exit 1
    fi
fi

echo "=============================================="
echo "Phase 13.6.G: Debug Notebook Validation"
echo "=============================================="
echo ""

PASSED=0
FAILED=0

for nb in "${NOTEBOOKS[@]}"; do
    nb_path="$SCRIPT_DIR/$nb"
    output_name="${nb%.ipynb}"
    
    if [ ! -f "$nb_path" ]; then
        echo "⚠️  Skipping $nb (not found)"
        continue
    fi
    
    echo -n "Testing $nb... "
    
    if [ $VERBOSE -eq 1 ]; then
        # Verbose: show all output
        if jupyter nbconvert --execute "$nb_path" \
            --to html \
            --output "$OUTPUT_DIR/$output_name.html" \
            --ExecutePreprocessor.timeout=300; then
            echo "✓ PASSED"
            ((PASSED++))
        else
            echo "✗ FAILED"
            ((FAILED++))
        fi
    else
        # Quiet: suppress output, show only on failure
        if jupyter nbconvert --execute "$nb_path" \
            --to html \
            --output "$OUTPUT_DIR/$output_name.html" \
            --ExecutePreprocessor.timeout=300 \
            2>/dev/null; then
            echo "✓ PASSED"
            ((PASSED++))
        else
            echo "✗ FAILED"
            ((FAILED++))
            # Show error details
            jupyter nbconvert --execute "$nb_path" \
                --to html \
                --output "$OUTPUT_DIR/$output_name.html" \
                --ExecutePreprocessor.timeout=300 2>&1 | tail -20
        fi
    fi
done

echo ""
echo "=============================================="
echo "Results: $PASSED passed, $FAILED failed"
echo "HTML outputs: $OUTPUT_DIR/"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
