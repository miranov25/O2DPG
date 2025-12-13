#!/bin/bash
# =============================================================================
# Tutorial Test Script
# =============================================================================
#
# Tests all tutorials in sequence to verify they work correctly.
#
# Usage:
#     cd /path/to/AliasDataFrame
#     bash tutorials/test_tutorials.sh
#
# =============================================================================

set -e  # Exit on first error

echo "============================================================"
echo "AliasDataFrame Tutorial Test Suite"
echo "============================================================"
echo ""

# Get script directory and ADF directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADF_DIR="$(dirname "$SCRIPT_DIR")"

echo "Tutorial directory: $SCRIPT_DIR"
echo "AliasDataFrame directory: $ADF_DIR"
echo ""

# =============================================================================
# CRITICAL: Set PYTHONPATH
# =============================================================================
# AliasDataFrame and dfdraw are siblings under dfextensions/
# We need the parent directory so "from dfdraw import ..." works
DFEXTENSIONS_DIR="$(dirname "$ADF_DIR")"
export PYTHONPATH="$ADF_DIR:$DFEXTENSIONS_DIR:$PYTHONPATH"
echo "PYTHONPATH includes:"
echo "  - $ADF_DIR (for AliasDataFrame)"
echo "  - $DFEXTENSIONS_DIR (for dfdraw, etc.)"

# Verify dfdraw directory exists
if [ -d "$DFEXTENSIONS_DIR/dfdraw" ]; then
    echo "  ✓ dfdraw found at: $DFEXTENSIONS_DIR/dfdraw"
else
    echo "  ⚠ dfdraw NOT found at: $DFEXTENSIONS_DIR/dfdraw"
fi
echo ""

# =============================================================================
# Step 0: Check prerequisites
# =============================================================================

echo "============================================================"
echo "Step 0: Checking prerequisites..."
echo "============================================================"

python3 -c "from AliasDataFrame import AliasDataFrame; print('✓ AliasDataFrame available')" || {
    echo "✗ AliasDataFrame not found."
    echo "Make sure you run from the AliasDataFrame directory:"
    echo "  cd /path/to/AliasDataFrame"
    echo "  bash tutorials/test_tutorials.sh"
    exit 1
}

python3 -c "import uproot; print('✓ uproot available')" || {
    echo "✗ uproot not found. Install with: pip install uproot"
    exit 1
}

python3 -c "import numpy; print('✓ numpy available')"
python3 -c "import pandas; print('✓ pandas available')"

python3 -c "from dfdraw import DFDraw; print('✓ dfdraw available')" 2>/dev/null || {
    echo "⚠ dfdraw not found. Tutorials will run without plots."
}

echo ""

# =============================================================================
# Step 1: Generate synthetic data
# =============================================================================

echo "============================================================"
echo "Step 1: Generating synthetic data..."
echo "============================================================"

DATA_DIR="$SCRIPT_DIR/data"
DATA_FILE="$DATA_DIR/synthetic_data.root"

mkdir -p "$DATA_DIR"

if [ -f "$DATA_FILE" ]; then
    echo "Data file exists: $DATA_FILE"
    echo "Skipping generation (delete file to regenerate)"
else
    echo "Generating: $DATA_FILE"
    python3 "$ADF_DIR/benchmarks/generate_synthetic_data.py" \
        -o "$DATA_FILE" \
        --rows 10000 \
        --tracks 1000 \
        --verify
    
    if [ $? -eq 0 ]; then
        echo "✓ Data generation successful"
    else
        echo "✗ Data generation failed"
        exit 1
    fi
fi

echo ""

# =============================================================================
# Step 2: Test histograms.py
# =============================================================================

echo "============================================================"
echo "Step 2: Testing histograms.py..."
echo "============================================================"

cd "$SCRIPT_DIR/drawing"
python3 histograms.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ histograms.py passed"
else
    echo "✗ histograms.py failed"
    exit 1
fi

echo ""

# =============================================================================
# Step 3: Test scatter_profile.py
# =============================================================================

echo "============================================================"
echo "Step 3: Testing scatter_profile.py..."
echo "============================================================"

python3 scatter_profile.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ scatter_profile.py passed"
else
    echo "✗ scatter_profile.py failed"
    exit 1
fi

echo ""

# =============================================================================
# Step 4: Test selections_groupby.py
# =============================================================================

echo "============================================================"
echo "Step 4: Testing selections_groupby.py..."
echo "============================================================"

python3 selections_groupby.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ selections_groupby.py passed"
else
    echo "✗ selections_groupby.py failed"
    exit 1
fi

echo ""

# =============================================================================
# Step 5: Test with_subframes.py
# =============================================================================

echo "============================================================"
echo "Step 5: Testing with_subframes.py..."
echo "============================================================"

python3 with_subframes.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ with_subframes.py passed"
else
    echo "✗ with_subframes.py failed"
    exit 1
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
echo "TUTORIAL TEST SUMMARY"
echo "============================================================"
echo ""
echo "✓ Step 0: Prerequisites checked"
echo "✓ Step 1: Synthetic data ready"
echo "✓ Step 2: histograms.py"
echo "✓ Step 3: scatter_profile.py"
echo "✓ Step 4: selections_groupby.py"
echo "✓ Step 5: with_subframes.py"
echo ""
echo "============================================================"
echo "ALL TUTORIALS PASSED"
echo "============================================================"
