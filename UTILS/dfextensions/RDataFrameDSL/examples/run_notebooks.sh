#!/bin/bash
# Run debug notebooks and verify they execute without errors
# Phase 13.6.G: Debug notebook validation
#
# Usage:
#   ./run_notebooks.sh              # Run all (default verbose)
#   ./run_notebooks.sh -q           # Run all (quiet - less output)
#   ./run_notebooks.sh 08a          # Run specific notebook
#   ./run_notebooks.sh -q 08a       # Run specific notebook (quiet)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/tmp/notebook_tests"
QUIET=0

# Parse options
while getopts "qv" opt; do
    case $opt in
        q) QUIET=1 ;;
        v) QUIET=0 ;;  # Kept for backward compatibility
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
    
    if [ $QUIET -eq 1 ]; then
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
                --ExecutePreprocessor.timeout=300 2>&1 | tail -30
        fi
    else
        # Default verbose: show all output on failure, summary on success
        OUTPUT=$(jupyter nbconvert --execute "$nb_path" \
            --to html \
            --output "$OUTPUT_DIR/$output_name.html" \
            --ExecutePreprocessor.timeout=300 2>&1)
        STATUS=$?
        
        if [ $STATUS -eq 0 ]; then
            echo "✓ PASSED"
            ((PASSED++))
            # Show output path
            echo "   → $OUTPUT_DIR/$output_name.html"
        else
            echo "✗ FAILED"
            ((FAILED++))
            # Show error details
            echo "----------------------------------------"
            echo "$OUTPUT" | tail -40
            echo "----------------------------------------"
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
