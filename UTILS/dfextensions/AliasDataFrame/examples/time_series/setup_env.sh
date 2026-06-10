#!/bin/bash
# =============================================================================
# Environment setup for AliasDataFrame examples
# =============================================================================
#
# Usage:
#     source setup_env.sh
#
# After sourcing, `python`, `ipython`, and `pytest` will all find:
#   - AliasDataFrame  (from .../dfextensions/AliasDataFrame)
#   - dfdraw          (from .../dfextensions/dfdraw)
#
# Mirrors the PYTHONPATH convention from tutorials/test_tutorials.sh.
# =============================================================================

# Resolve where this script lives, even when sourced
# (BASH_SOURCE[0] is the script path; $0 would be "bash" when sourced)
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Layout: dfextensions/AliasDataFrame/examples/time_series/setup_env.sh
#   ../../  →  .../AliasDataFrame
#   ../../../  →  .../dfextensions
ADF_DIR="$(cd "$_HERE/../.." && pwd)"
DFEXTENSIONS_DIR="$(cd "$_HERE/../../.." && pwd)"

export PYTHONPATH="$ADF_DIR:$DFEXTENSIONS_DIR:$PYTHONPATH"

echo "Environment configured:"
echo "  ADF_DIR           = $ADF_DIR"
echo "  DFEXTENSIONS_DIR  = $DFEXTENSIONS_DIR"
echo "  PYTHONPATH        = $PYTHONPATH"
echo ""

# Verify imports work
python3 -c "from AliasDataFrame import AliasDataFrame; print('  ✓ AliasDataFrame importable')" 2>&1 || \
    echo "  ✗ AliasDataFrame import FAILED — check ADF_DIR"

python3 -c "from dfdraw import DFDraw; print('  ✓ dfdraw importable')" 2>&1 || \
    echo "  ⚠ dfdraw import failed (tutorials work without it but plots won't render)"

unset _HERE
