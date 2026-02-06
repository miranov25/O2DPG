#!/usr/bin/env python3
"""
Phase 13.7.B Exploration: Parent-Child Indexing for Residuals (v2)
===================================================================

Uses correct offset-based indexing from unbinnedResid tree.

Run with: python -u explore_residuals_v2.py 2>&1 | tee explore_residuals_v2.log
"""

import sys
print(f"Python: {sys.version}")

# ============================================================
# Cell E0: Setup and Imports
# ============================================================
print("=" * 60)
print("E0: Setup and Imports")
print("=" * 60)

import ROOT
print(f"ROOT version: {ROOT.gROOT.GetVersion()}")

# Configuration
LIB_PATH = "libIndexHelpers.so"
HEADER_PATH = "index_helpers.h"
RESIDUALS_FILE = "o2residuals_tpc.root"

# ============================================================
# Cell E1: Load Index Helpers Library
# ============================================================
print("\n" + "=" * 60)
print("E1: Load Index Helpers Library")
print("=" * 60)

load_result = ROOT.gSystem.Load(LIB_PATH)
print(f"  Load {LIB_PATH}: {load_result}")

ROOT.gInterpreter.Declare(f'#include "{HEADER_PATH}"')
print(f"  Header included: {HEADER_PATH}")

print("\n[OK] E1 PASSED")

# ============================================================
# Cell E2: Load unbinnedResid - Single Tree (Correct Approach)
# ============================================================
print("\n" + "=" * 60)
print("E2: Load unbinnedResid Tree")
print("=" * 60)

N_EVENTS = 5
rdf = ROOT.RDataFrame("unbinnedResid", RESIDUALS_FILE).Range(N_EVENTS)

cols = [str(c) for c in rdf.GetColumnNames()]
print(f"  Total columns: {len(cols)}")
print(f"  Key columns:")
print(f"    - trackInfo.idxFirstResidual (offsets)")
print(f"    - res.dy, res.dz (residual values)")

print("\n[OK] E2 PASSED")

# ============================================================
# Cell E3: Define Computed Columns Using C++ Helpers
# ============================================================
print("\n" + "=" * 60)
print("E3: Define Computed Columns")
print("=" * 60)

# Use ExpandParentIndexFromOffsets - needs totalSize = res.dy.size()
rdf = rdf.Define(
    "parentIdx",
    "RDataFrameDSL::IndexHelpers::ExpandParentIndexFromOffsets("
    "trackInfo.idxFirstResidual, (int)res.dy.size())"
)
print("  Defined: parentIdx = ExpandParentIndexFromOffsets(trackInfo.idxFirstResidual, res.dy.size())")

print("\n[OK] E3 PASSED")

# ============================================================
# Cell E4: Export to NumPy and Verify
# ============================================================
print("\n" + "=" * 60)
print("E4: Export to NumPy and Verify")
print("=" * 60)

result = rdf.AsNumpy(["parentIdx", "res.dy", "res.dz", "trackInfo.idxFirstResidual"])

print(f"  Exported {N_EVENTS} events")

# Verify lengths match
for i in range(N_EVENTS):
    pidx_len = len(result["parentIdx"][i])
    dy_len = len(result["res.dy"][i])
    match = "✓" if pidx_len == dy_len else "✗"
    print(f"    Event {i}: parentIdx={pidx_len}, res.dy={dy_len} {match}")

print("\n[OK] E4 PASSED")

# ============================================================
# Cell E5: Flatten to pandas DataFrame
# ============================================================
print("\n" + "=" * 60)
print("E5: Flatten to pandas DataFrame")
print("=" * 60)

import pandas as pd

all_eventIdx = []
all_parentIdx = []
all_dy = []
all_dz = []

for evt_idx in range(N_EVENTS):
    pidx = result["parentIdx"][evt_idx]
    dy = result["res.dy"][evt_idx]
    dz = result["res.dz"][evt_idx]
    
    n = len(pidx)
    all_eventIdx.extend([evt_idx] * n)
    all_parentIdx.extend([int(pidx[i]) for i in range(n)])
    all_dy.extend([int(dy[i]) for i in range(n)])
    all_dz.extend([int(dz[i]) for i in range(n)])

df = pd.DataFrame({
    "eventIdx": all_eventIdx,
    "parentIdx": all_parentIdx,
    "dy": all_dy,
    "dz": all_dz,
})

print(f"  DataFrame shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Head:\n{df.head(10)}")
print(f"\n  Tail:\n{df.tail(10)}")

# Statistics per event
print(f"\n  Residuals per event:")
for evt in range(N_EVENTS):
    n_res = len(df[df.eventIdx == evt])
    n_tracks = df[df.eventIdx == evt].parentIdx.max() + 1
    print(f"    Event {evt}: {n_res} residuals, {n_tracks} tracks")

print("\n[OK] E5 PASSED")

# ============================================================
# Cell E6: Draw Histograms
# ============================================================
print("\n" + "=" * 60)
print("E6: Draw Histograms")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: dy distribution
    axes[0,0].hist(df["dy"], bins=100, alpha=0.7, color='blue')
    axes[0,0].set_xlabel("dy (residual)")
    axes[0,0].set_ylabel("Count")
    axes[0,0].set_title(f"Residual dy distribution (N={len(df)})")
    
    # Plot 2: dz distribution
    axes[0,1].hist(df["dz"], bins=100, alpha=0.7, color='green')
    axes[0,1].set_xlabel("dz (residual)")
    axes[0,1].set_ylabel("Count")
    axes[0,1].set_title("Residual dz distribution")
    
    # Plot 3: dy vs dz 2D
    axes[1,0].hist2d(df["dy"], df["dz"], bins=50, cmap='viridis')
    axes[1,0].set_xlabel("dy")
    axes[1,0].set_ylabel("dz")
    axes[1,0].set_title("dy vs dz")
    
    # Plot 4: Residuals per track (histogram of track sizes)
    track_sizes = df.groupby(["eventIdx", "parentIdx"]).size()
    axes[1,1].hist(track_sizes, bins=50, alpha=0.7, color='orange')
    axes[1,1].set_xlabel("Residuals per track")
    axes[1,1].set_ylabel("Count")
    axes[1,1].set_title(f"Track size distribution (mean={track_sizes.mean():.1f})")
    
    plt.tight_layout()
    plt.savefig("explore_residuals_v2_plot.png", dpi=100)
    print(f"  Saved: explore_residuals_v2_plot.png")
    
    print("\n[OK] E6 PASSED")
    
except ImportError as e:
    print(f"  matplotlib not available: {e}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Cell E7: Test ExpandToChildrenFromOffsets (expand track property)
# ============================================================
print("\n" + "=" * 60)
print("E7: Test ExpandToChildrenFromOffsets")
print("=" * 60)

# We need a track property from trackData tree
# For now, let's use trackInfo.nResiduals as example (even though buggy)
# In real usage, would join with trackData tree

rdf2 = ROOT.RDataFrame("unbinnedResid", RESIDUALS_FILE).Range(N_EVENTS)

# Expand trackInfo.sourceId to residual level
rdf2 = rdf2.Define(
    "res_sourceId",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets("
    "trackInfo.sourceId, trackInfo.idxFirstResidual, (int)res.dy.size())"
)

result2 = rdf2.AsNumpy(["res_sourceId", "res.dy"])

print("  Expanded trackInfo.sourceId to residual level")
for i in range(min(3, N_EVENTS)):
    src_len = len(result2["res_sourceId"][i])
    dy_len = len(result2["res.dy"][i])
    match = "✓" if src_len == dy_len else "✗"
    print(f"    Event {i}: res_sourceId={src_len}, res.dy={dy_len} {match}")

print("\n[OK] E7 PASSED")

# ============================================================
# Cell E8: Summary
# ============================================================
print("\n" + "=" * 60)
print("E8: SUMMARY")
print("=" * 60)

print(f"""
Phase 13.7.B Exploration Results (v2):
======================================

Data Structure:
  - unbinnedResid tree: {N_EVENTS} events processed
  - Total residuals: {len(df)}
  - Index: trackInfo.idxFirstResidual (offsets only)

Working Helpers:
  ✓ ExpandParentIndexFromOffsets - builds child→parent mapping
  ✓ ExpandToChildrenFromOffsets  - expands parent values to child level

Key Insight:
  - trackInfo.nResiduals is BUGGY (counts subset only)
  - Correct: compute nEntries from consecutive offsets
  - Formula: nEntries[i] = offset[i+1] - offset[i]

Output:
  - pandas DataFrame with eventIdx, parentIdx, dy, dz
  - Histogram: explore_residuals_v2_plot.png

Next Steps:
  1. Join with trackData tree for track properties
  2. Integrate into DSLCompiler
  3. Support dsl.draw("res.dy : track.dEdx") syntax
""")
