#!/usr/bin/env python3
"""
Example 02: RVec Operations

Demonstrates:
- RVec size and indexing
- Python-like slicing ([:3], [-2:], [::2])
- Boolean masking
- Safe out-of-bounds handling (returns NaN)
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT
from RDataFrameDSL import DSLCompiler

print("=" * 60)
print("Example 02: RVec Operations")
print("=" * 60)

# =============================================================================
# Schema Definition
# =============================================================================

schema = {
    "pt": "RVec<double>",  # Track pT values
}

print("\nSchema:")
print("  pt: RVec<double>  (variable-length array of pT values)")

# =============================================================================
# Create DSLCompiler and Define Expressions
# =============================================================================

dsl = DSLCompiler(schema)

# Size operations
dsl.define("n_tracks", "pt.size()")

# Safe indexing (returns NaN if out of bounds)
dsl.define("lead_pt", "pt[0]")      # First track
dsl.define("sublead_pt", "pt[1]")   # Second track
dsl.define("last_pt", "pt[-1]")     # Last track

# Slicing
dsl.define("first3", "pt[:3]")      # First 3 tracks
dsl.define("last2", "pt[-2:]")      # Last 2 tracks
dsl.define("middle", "pt[1:4]")     # Tracks 1, 2, 3
dsl.define("every_other", "pt[::2]") # Even indices
dsl.define("reversed", "pt[::-1]")  # Reversed order

# Boolean masking
dsl.define("high_pt", "pt[pt > 5.0]")  # Tracks with pT > 5 GeV
dsl.define("n_high_pt", "pt[pt > 5.0].size()")

print("\nDefined expressions:")
print("  n_tracks    = pt.size()")
print("  lead_pt     = pt[0]           # First (NaN if empty)")
print("  sublead_pt  = pt[1]           # Second (NaN if < 2)")
print("  last_pt     = pt[-1]          # Last")
print("  first3      = pt[:3]          # First 3")
print("  last2       = pt[-2:]         # Last 2")
print("  middle      = pt[1:4]         # Elements 1,2,3")
print("  every_other = pt[::2]         # Even indices")
print("  reversed    = pt[::-1]        # Reversed")
print("  high_pt     = pt[pt > 5.0]    # Boolean mask")
print("  n_high_pt   = pt[pt > 5.0].size()")

# =============================================================================
# Preview Generated C++
# =============================================================================

print("\n" + "=" * 60)
print("Generated C++ Code (selected):")
print("=" * 60)

# Show a few interesting examples
preview = dsl.preview()
print(preview)

# =============================================================================
# Apply to RDataFrame
# =============================================================================

print("=" * 60)
print("Execution with RDataFrame:")
print("=" * 60)

try:
    rdf = ROOT.RDataFrame("Events", "test_tracks.root")
    rdf = dsl.apply(rdf)
    
    # Get scalar results
    scalars = rdf.AsNumpy(["n_tracks", "lead_pt", "sublead_pt", "last_pt", "n_high_pt"])
    
    print(f"\nProcessed {len(scalars['n_tracks'])} events")
    
    print(f"\nFirst 10 events (scalar values):")
    print(f"  {'n_tracks':>10} {'lead_pt':>10} {'sublead_pt':>10} {'last_pt':>10} {'n_high_pt':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(min(10, len(scalars['n_tracks']))):
        n = scalars['n_tracks'][i]
        lead = scalars['lead_pt'][i]
        sub = scalars['sublead_pt'][i]
        last = scalars['last_pt'][i]
        nhigh = scalars['n_high_pt'][i]
        print(f"  {n:10.0f} {lead:10.4f} {sub:10.4f} {last:10.4f} {nhigh:10.0f}")
    
    print(f"\nStatistics:")
    print(f"  Mean tracks/event: {scalars['n_tracks'].mean():.2f}")
    print(f"  Mean lead pT: {scalars['lead_pt'].mean():.4f}")
    print(f"  Mean high-pT tracks/event: {scalars['n_high_pt'].mean():.2f}")
    
    # Show NaN handling
    import numpy as np
    nan_count = np.isnan(scalars['sublead_pt']).sum()
    print(f"\n  Events with NaN sublead_pt (single-track events): {nan_count}")
    
except Exception as e:
    print(f"\nNote: Could not run on data file: {e}")
    print("Run 'python create_test_data.py' first to generate test files.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
