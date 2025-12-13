#!/usr/bin/env python3
"""
Diagnostic script to check dfdraw import path issues.

Run from AliasDataFrame directory:
    python tutorials/diagnose_dfdraw.py
"""

import sys
import os

print("=" * 60)
print("dfdraw Import Diagnostic")
print("=" * 60)
print()

# Show Python path
print("Python executable:", sys.executable)
print()
print("sys.path:")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")
print()

# Try to import dfdraw
print("Attempting to import dfdraw...")
try:
    import dfdraw
    print(f"✓ dfdraw imported successfully")
    print(f"  Location: {dfdraw.__file__}")
    print(f"  Has DFDraw class: {hasattr(dfdraw, 'DFDraw')}")
except ImportError as e:
    print(f"✗ dfdraw import failed: {e}")
    print()
    print("Possible fixes:")
    print("  1. Check if dfdraw is in PYTHONPATH")
    print("  2. Add dfdraw location to test_tutorials.sh")
    print()
    
    # Try to find dfdraw
    print("Searching for dfdraw...")
    
    # Check common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adf_dir = os.path.dirname(script_dir)
    parent_dir = os.path.dirname(adf_dir)
    
    possible_locations = [
        os.path.join(parent_dir, 'dfdraw'),
        os.path.join(adf_dir, '..', 'dfdraw'),
        os.path.join(adf_dir, '..', '..', 'dfdraw'),
        '/Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/dfdraw',
    ]
    
    for loc in possible_locations:
        loc = os.path.abspath(loc)
        if os.path.exists(loc):
            print(f"  Found: {loc}")
            if os.path.exists(os.path.join(loc, 'drawer.py')) or os.path.exists(os.path.join(loc, '__init__.py')):
                print(f"    ✓ Looks like dfdraw directory")
        else:
            print(f"  Not found: {loc}")

print()

# Check AliasDataFrame import
print("Checking AliasDataFrame import...")
try:
    from AliasDataFrame import AliasDataFrame
    print(f"✓ AliasDataFrame imported successfully")
except ImportError as e:
    print(f"✗ AliasDataFrame import failed: {e}")

print()
print("=" * 60)
print("Recommendation")
print("=" * 60)
print()
print("If dfdraw is at:")
print("  /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/dfdraw")
print()
print("Add this to test_tutorials.sh after the PYTHONPATH line:")
print('  export PYTHONPATH="$ADF_DIR/../dfdraw:$PYTHONPATH"')
print()
print("Or run tutorials with:")
print("  PYTHONPATH=/path/to/dfdraw:$PYTHONPATH ./tutorials/test_tutorials.sh")
