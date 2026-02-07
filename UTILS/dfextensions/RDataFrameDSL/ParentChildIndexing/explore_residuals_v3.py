#!/usr/bin/env python3
"""
Phase 13.7.B Exploration: Flattened Export (v3)
===============================================

Working version with:
- Friend tree (trackData + unbinnedResid)
- Track properties expanded to residual level
- Cluster charge extracted using O2 interface

Run with: python -u explore_residuals_v3.py 2>&1 | tee explore_residuals_v3.log
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
import numpy as np
print(f"ROOT version: {ROOT.gROOT.GetVersion()}")

# Configuration
LIB_PATH = "libIndexHelpers.so"
HEADER_PATH = "index_helpers.h"
RESIDUALS_FILE = "o2residuals_tpc.root"
N_EVENTS = 5

# ============================================================
# Cell E1: Load Index Helpers and O2 Residual Helpers
# ============================================================
print("\n" + "=" * 60)
print("E1: Load Index Helpers and O2 Residual Helpers")
print("=" * 60)

# Load IndexHelpers (for ExpandParentIndexFromOffsets, etc.)
load_result = ROOT.gSystem.Load(LIB_PATH)
print(f"  Load {LIB_PATH}: {load_result}")

ROOT.gInterpreter.Declare(f'#include "{HEADER_PATH}"')
print(f"  Header included: {HEADER_PATH}")

# Load O2ResidualHelpers (for ExtractQMaxTPC, ExtractQTotTPC)
O2_HELPERS_LIB = "libO2ResidualHelpers.so"
O2_HELPERS_HEADER = "o2_residual_helpers.h"

load_result2 = ROOT.gSystem.Load(O2_HELPERS_LIB)
print(f"  Load {O2_HELPERS_LIB}: {load_result2}")

ROOT.gInterpreter.Declare(f'#include "{O2_HELPERS_HEADER}"')
print(f"  Header included: {O2_HELPERS_HEADER}")

# Verify functions are available
print(f"  ExtractQMaxTPC: {ROOT.RDataFrameDSL.O2ResidualHelpers.ExtractQMaxTPC}")
print(f"  ExtractQTotTPC: {ROOT.RDataFrameDSL.O2ResidualHelpers.ExtractQTotTPC}")

print("\n[OK] E1 PASSED")

# ============================================================
# Cell E2: Create Friend Tree RDataFrame
# ============================================================
print("\n" + "=" * 60)
print("E2: Create Friend Tree RDataFrame")
print("=" * 60)

chain_resid = ROOT.TChain("unbinnedResid")
chain_resid.Add(RESIDUALS_FILE)

chain_tracks = ROOT.TChain("trackData")
chain_tracks.Add(RESIDUALS_FILE)

print(f"  unbinnedResid entries: {chain_resid.GetEntries()}")
print(f"  trackData entries: {chain_tracks.GetEntries()}")

chain_resid.AddFriend(chain_tracks, "td")
print("  Added trackData as friend 'td'")

rdf = ROOT.RDataFrame(chain_resid).Range(N_EVENTS)
print(f"  RDataFrame created with Range({N_EVENTS})")

print("\n[OK] E2 PASSED")

# ============================================================
# Cell E3: Define Flattened Columns
# ============================================================
print("\n" + "=" * 60)
print("E3: Define Flattened Columns")
print("=" * 60)

# Parent index (which track each residual belongs to)
rdf = rdf.Define("parentIdx", 
    "RDataFrameDSL::IndexHelpers::ExpandParentIndexFromOffsets("
    "trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Defined: parentIdx")

# Expand track dEdxTPC to residual level
rdf = rdf.Define("res_dEdxTPC",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
    "td.trk.dEdxTPC, trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Defined: res_dEdxTPC (track dEdx expanded to residuals)")

# Expand track chi2TPC to residual level
rdf = rdf.Define("res_chi2TPC",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
    "td.trk.chi2TPC, trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Defined: res_chi2TPC (track chi2 expanded to residuals)")

# Expand track nClsTPC to residual level
rdf = rdf.Define("res_nClsTPC",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<unsigned char>("
    "td.trk.nClsTPC, trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Defined: res_nClsTPC (track nCls expanded to residuals)")

# Extract cluster charge using O2 interface (from library)
rdf = rdf.Define("qMaxTPC", "RDataFrameDSL::O2ResidualHelpers::ExtractQMaxTPC(detInfo)")
rdf = rdf.Define("qTotTPC", "RDataFrameDSL::O2ResidualHelpers::ExtractQTotTPC(detInfo)")
print("  Defined: qMaxTPC, qTotTPC (cluster charge)")

print("\n[OK] E3 PASSED")

# ============================================================
# Cell E4: Export to NumPy
# ============================================================
print("\n" + "=" * 60)
print("E4: Export to NumPy")
print("=" * 60)

export_cols = ["res.dy", "res.dz", "res.row", "res.sec", 
               "parentIdx", "res_dEdxTPC", "res_chi2TPC", "res_nClsTPC",
               "qMaxTPC", "qTotTPC"]

result = rdf.AsNumpy(export_cols)

print(f"  Exported {len(export_cols)} columns")
for i in range(min(3, N_EVENTS)):
    print(f"    Event {i}: {len(result['res.dy'][i])} residuals")

print("\n[OK] E4 PASSED")

# ============================================================
# Cell E5: Flatten to pandas DataFrame
# ============================================================
print("\n" + "=" * 60)
print("E5: Flatten to pandas DataFrame")
print("=" * 60)

import pandas as pd

all_data = {col: [] for col in export_cols}
all_data["eventIdx"] = []

n_events_actual = len(result["res.dy"])
total_residuals = 0

for evt_idx in range(n_events_actual):
    n_residuals = len(result["res.dy"][evt_idx])
    total_residuals += n_residuals
    all_data["eventIdx"].extend([evt_idx] * n_residuals)
    
    for col in export_cols:
        arr = result[col][evt_idx]
        all_data[col].extend([float(arr[i]) for i in range(len(arr))])

df = pd.DataFrame(all_data)

# Rename columns for convenience
df = df.rename(columns={
    "res.dy": "dy",
    "res.dz": "dz",
    "res.row": "row",
    "res.sec": "sec",
    "res_dEdxTPC": "dEdx",
    "res_chi2TPC": "chi2",
    "res_nClsTPC": "nCls",
})

print(f"  DataFrame shape: {df.shape}")
print(f"  Total residuals: {total_residuals}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Head:\n{df.head(10)}")
print(f"\n  Describe:\n{df[['dy', 'dz', 'dEdx', 'chi2', 'qMaxTPC', 'qTotTPC']].describe()}")

print("\n[OK] E5 PASSED")

# ============================================================
# Cell E6: Draw Basic Distributions
# ============================================================
print("\n" + "=" * 60)
print("E6: Draw Basic Distributions")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# dy distribution
axes[0,0].hist(df["dy"], bins=100, range=(-5000, 5000), alpha=0.7, color='blue')
axes[0,0].set_xlabel("dy (residual)")
axes[0,0].set_ylabel("Count")
axes[0,0].set_title(f"Residual dy (N={len(df)})")

# dz distribution
axes[0,1].hist(df["dz"], bins=100, range=(-5000, 5000), alpha=0.7, color='green')
axes[0,1].set_xlabel("dz (residual)")
axes[0,1].set_ylabel("Count")
axes[0,1].set_title("Residual dz")

# dEdx distribution
axes[0,2].hist(df["dEdx"], bins=100, range=(0, 200), alpha=0.7, color='red')
axes[0,2].set_xlabel("dEdx TPC")
axes[0,2].set_ylabel("Count")
axes[0,2].set_title("Track dEdx")

# qMaxTPC distribution
axes[1,0].hist(df["qMaxTPC"], bins=100, range=(0, 200), alpha=0.7, color='orange')
axes[1,0].set_xlabel("qMaxTPC")
axes[1,0].set_ylabel("Count")
axes[1,0].set_title("Cluster qMax")

# qTotTPC distribution
axes[1,1].hist(df["qTotTPC"], bins=100, range=(0, 500), alpha=0.7, color='purple')
axes[1,1].set_xlabel("qTotTPC")
axes[1,1].set_ylabel("Count")
axes[1,1].set_title("Cluster qTot")

# row (padrow) distribution
axes[1,2].hist(df["row"], bins=152, range=(0, 152), alpha=0.7, color='brown')
axes[1,2].set_xlabel("row (padrow)")
axes[1,2].set_ylabel("Count")
axes[1,2].set_title("Padrow distribution")

plt.tight_layout()
plt.savefig("explore_residuals_v3_basic.png", dpi=100)
print("  Saved: explore_residuals_v3_basic.png")

print("\n[OK] E6 PASSED")

# ============================================================
# Cell E7: Draw Correlations
# ============================================================
print("\n" + "=" * 60)
print("E7: Draw Correlations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# dy vs dEdx
axes[0,0].hist2d(df["dEdx"], df["dy"], bins=[50, 100], 
                 range=[[20, 100], [-3000, 3000]], cmap='viridis')
axes[0,0].set_xlabel("dEdx TPC")
axes[0,0].set_ylabel("dy")
axes[0,0].set_title("dy vs dEdx (track)")

# dy vs qMaxTPC
axes[0,1].hist2d(df["qMaxTPC"], df["dy"], bins=[50, 100], 
                 range=[[0, 100], [-3000, 3000]], cmap='viridis')
axes[0,1].set_xlabel("qMaxTPC")
axes[0,1].set_ylabel("dy")
axes[0,1].set_title("dy vs qMax (cluster)")

# dy vs row (padrow)
axes[0,2].hist2d(df["row"], df["dy"], bins=[152, 100], 
                 range=[[0, 152], [-3000, 3000]], cmap='viridis')
axes[0,2].set_xlabel("row (padrow)")
axes[0,2].set_ylabel("dy")
axes[0,2].set_title("dy vs padrow")

# dz vs row
axes[1,0].hist2d(df["row"], df["dz"], bins=[152, 100], 
                 range=[[0, 152], [-3000, 3000]], cmap='viridis')
axes[1,0].set_xlabel("row (padrow)")
axes[1,0].set_ylabel("dz")
axes[1,0].set_title("dz vs padrow")

# qMaxTPC vs row
axes[1,1].hist2d(df["row"], df["qMaxTPC"], bins=[152, 50], 
                 range=[[0, 152], [0, 100]], cmap='viridis')
axes[1,1].set_xlabel("row (padrow)")
axes[1,1].set_ylabel("qMaxTPC")
axes[1,1].set_title("qMax vs padrow")

# qMaxTPC vs dEdx
axes[1,2].hist2d(df["dEdx"], df["qMaxTPC"], bins=[50, 50], 
                 range=[[20, 100], [0, 100]], cmap='viridis')
axes[1,2].set_xlabel("dEdx TPC")
axes[1,2].set_ylabel("qMaxTPC")
axes[1,2].set_title("qMax vs dEdx")

plt.tight_layout()
plt.savefig("explore_residuals_v3_correlations.png", dpi=100)
print("  Saved: explore_residuals_v3_correlations.png")

print("\n[OK] E7 PASSED")

# ============================================================
# Cell E8: Save to CSV
# ============================================================
print("\n" + "=" * 60)
print("E8: Save to CSV")
print("=" * 60)

csv_file = "flattened_residuals.csv"
df.to_csv(csv_file, index=False)
print(f"  Saved: {csv_file} ({len(df)} rows)")

print("\n[OK] E8 PASSED")

# ============================================================
# Cell E9: Summary
# ============================================================
print("\n" + "=" * 60)
print("E9: SUMMARY")
print("=" * 60)

print(f"""
Phase 13.7.B Exploration Results (v3):
======================================

Data Processed:
  - Events: {N_EVENTS}
  - Total residuals: {len(df)}
  - Unique tracks: {df.groupby(['eventIdx', 'parentIdx']).ngroups}

Flattened Columns:
  Residual level: dy, dz, row, sec, qMaxTPC, qTotTPC
  Track level (expanded): dEdx, chi2, nCls, parentIdx

Outputs:
  - flattened_residuals.csv
  - explore_residuals_v3_basic.png
  - explore_residuals_v3_correlations.png

Key Findings:
  - Friend tree approach works (unbinnedResid + trackData)
  - ExpandToChildrenFromOffsets maps track properties to residuals
  - ExtractQMaxTPC/ExtractQTotTPC use proper O2 interface

Next Steps:
  1. Integrate into DSLCompiler
  2. Add more track properties (qPt, tgl from td.trk.par.*)
  3. Support dsl.draw("dy : dEdx") syntax
""")
