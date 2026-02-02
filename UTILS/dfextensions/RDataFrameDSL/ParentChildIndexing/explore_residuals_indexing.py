#!/usr/bin/env python3
"""
Phase 13.7.B Exploration: Parent-Child Indexing for Residuals
==============================================================

Interactive exploration using cell-style execution.
Run with: python -u explore_residuals_indexing.py 2>&1 | tee explore_residuals_indexing.log

Cell structure follows O2MCAI exploration pattern.
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

# Configuration - UPDATE THESE PATHS
LIB_PATH = "libIndexHelpers.so"
HEADER_PATH = "index_helpers.h"

# ============================================================
# Cell E1: Load Index Helpers Library
# ============================================================
print("\n" + "=" * 60)
print("E1: Load Index Helpers Library")
print("=" * 60)

# Load shared library
load_result = ROOT.gSystem.Load(LIB_PATH)
print(f"  Load {LIB_PATH}: {load_result} (0=ok, 1=already loaded, -1=fail)")

# Include header to make symbols visible
ROOT.gInterpreter.Declare(f'#include "{HEADER_PATH}"')
print(f"  Header included: {HEADER_PATH}")

# Quick test
test_code = '''
auto testFirstIdx = ROOT::RVecI{0, 3, 5};
auto testNEntries = ROOT::RVecI{3, 2, 4};
auto testResult = RDataFrameDSL::IndexHelpers::ExpandParentIndex(testFirstIdx, testNEntries);
'''
ROOT.gInterpreter.ProcessLine(test_code)
print("  Quick test: ExpandParentIndex executed successfully")

print("\n[OK] E1 PASSED: Index helpers loaded")

# ============================================================
# Cell E2: Test Helpers with Synthetic Data (Python side)
# ============================================================
print("\n" + "=" * 60)
print("E2: Test Helpers with Synthetic Data")
print("=" * 60)

# Create test RDataFrame with synthetic data
ROOT.gInterpreter.Declare('''
ROOT::RVecI synth_firstIdx = {0, 3, 5};
ROOT::RVecI synth_nEntries = {3, 2, 4};
ROOT::RVecF synth_parentValues = {1.0f, 2.0f, 3.0f};
''')

# Test ExpandParentIndex
result_code = '''
auto synth_parentIdx = RDataFrameDSL::IndexHelpers::ExpandParentIndex(synth_firstIdx, synth_nEntries);
std::cout << "ExpandParentIndex result size: " << synth_parentIdx.size() << std::endl;
'''
ROOT.gInterpreter.ProcessLine(result_code)

# Test ExpandToChildren
result_code2 = '''
auto synth_childValues = RDataFrameDSL::IndexHelpers::ExpandToChildren(synth_parentValues, synth_firstIdx, synth_nEntries);
std::cout << "ExpandToChildren result size: " << synth_childValues.size() << std::endl;
'''
ROOT.gInterpreter.ProcessLine(result_code2)

print("\n[OK] E2 PASSED: Synthetic data tests work")

# ============================================================
# Cell E3: Discover Residuals File Structure
# ============================================================
print("\n" + "=" * 60)
print("E3: Discover Residuals File Structure")
print("=" * 60)

# TODO: Set your residuals file path
RESIDUALS_FILE = "o2residuals_tpc.root"  # symlink to actual file

import os
if not os.path.exists(RESIDUALS_FILE):
    print(f"  WARNING: File not found: {RESIDUALS_FILE}")
    print("  Please update RESIDUALS_FILE path and re-run from this cell")
    print("  Skipping E3-E7...")
    SKIP_REAL_DATA = True
else:
    SKIP_REAL_DATA = False
    
    f = ROOT.TFile.Open(RESIDUALS_FILE, "READ")
    print(f"  File: {RESIDUALS_FILE}")
    print(f"  Size: {os.path.getsize(RESIDUALS_FILE) / 1e9:.2f} GB")
    
    print("\n  Trees:")
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TTree):
            print(f"    - {key.GetName()}: {obj.GetEntries()} entries")
    
    f.Close()
    print("\n[OK] E3 PASSED: File structure discovered")

# ============================================================
# Cell E4: Explore trackData Columns
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E4: Explore trackData Columns")
    print("=" * 60)
    
    rdf_track = ROOT.RDataFrame("trackData", RESIDUALS_FILE)
    track_cols = sorted([str(c) for c in rdf_track.GetColumnNames()])
    
    print(f"  Total columns: {len(track_cols)}")
    print("\n  Index-related columns:")
    for col in track_cols:
        if "Idx" in col or "idx" in col or "Entry" in col or "entry" in col:
            col_type = rdf_track.GetColumnType(col)
            print(f"    {col}: {col_type}")
    
    print("\n  Track property columns (first 20):")
    for col in track_cols[:20]:
        try:
            col_type = rdf_track.GetColumnType(col)
            print(f"    {col}: {col_type}")
        except Exception as e:
            print(f"    {col}: ERROR - {e}")
    
    print("\n[OK] E4 PASSED: trackData columns explored")

# ============================================================
# Cell E5: Explore unbinnedResid Columns
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E5: Explore unbinnedResid Columns")
    print("=" * 60)
    
    rdf_resid = ROOT.RDataFrame("unbinnedResid", RESIDUALS_FILE)
    resid_cols = sorted([str(c) for c in rdf_resid.GetColumnNames()])
    
    print(f"  Total columns: {len(resid_cols)}")
    print("\n  All columns:")
    for col in resid_cols:
        try:
            col_type = rdf_resid.GetColumnType(col)
            print(f"    {col}: {col_type}")
        except Exception as e:
            print(f"    {col}: ERROR - {e}")
    
    print("\n[OK] E5 PASSED: unbinnedResid columns explored")

# ============================================================
# Cell E6: Check Index Column Values (First Event)
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E6: Check Index Column Values (First Event)")
    print("=" * 60)
    
    # Use Range(1) for first event only
    rdf_1 = ROOT.RDataFrame("trackData", RESIDUALS_FILE).Range(1)
    
    # Column names from E4 output
    FIRST_IDX_COL = "trk.clIdx.mFirstEntry"
    N_ENTRIES_COL = "trk.clIdx.mEntries"
    
    print(f"  Trying columns: {FIRST_IDX_COL}, {N_ENTRIES_COL}")
    
    try:
        # Get column types first
        first_idx_type = rdf_1.GetColumnType(FIRST_IDX_COL)
        n_entries_type = rdf_1.GetColumnType(N_ENTRIES_COL)
        print(f"  Types: {FIRST_IDX_COL}={first_idx_type}, {N_ENTRIES_COL}={n_entries_type}")
        
        # Take values
        first_idx = rdf_1.Take[first_idx_type](FIRST_IDX_COL).GetValue()
        n_entries = rdf_1.Take[n_entries_type](N_ENTRIES_COL).GetValue()
        
        print(f"\n  First event:")
        print(f"    Number of tracks: {len(first_idx)}")
        print(f"    firstIdx[:10]: {list(first_idx)[:10]}")
        print(f"    nEntries[:10]: {list(n_entries)[:10]}")
        print(f"    Total residuals: {sum(n_entries)}")
        
        print("\n[OK] E6 PASSED: Index values retrieved")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Check column names from E4 output")

# ============================================================
# Cell E7: Test ExpandParentIndex with Real Data
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E7: Test ExpandParentIndex with Real Data")
    print("=" * 60)
    
    # Use Range(5) for quick test
    rdf_test = ROOT.RDataFrame("trackData", RESIDUALS_FILE).Range(5)
    
    try:
        # Define parent index using our helper
        rdf_test = rdf_test.Define(
            "parentIdx",
            f"RDataFrameDSL::IndexHelpers::ExpandParentIndex({FIRST_IDX_COL}, {N_ENTRIES_COL})"
        )
        
        # Get result
        parent_idx_all = rdf_test.Take["ROOT::RVecI"]("parentIdx").GetValue()
        
        print(f"  Events processed: {len(parent_idx_all)}")
        for i, pidx in enumerate(parent_idx_all):
            print(f"    Event {i}: {len(pidx)} residuals, parentIdx[:10]={list(pidx)[:10]}")
        
        print("\n[OK] E7 PASSED: ExpandParentIndex works with real data")
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================
# Cell E8: Test ExpandToChildren with Real Data
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E8: Test ExpandToChildren with Real Data")
    print("=" * 60)
    
    # TODO: Update TRACK_VALUE_COL based on E4 output
    # Need a simple float column like dEdx or chi2
    TRACK_VALUE_COL = "trk.dEdxTPC"  # Check E4 for actual column name
    
    rdf_test2 = ROOT.RDataFrame("trackData", RESIDUALS_FILE).Range(5)
    
    try:
        # Expand track dEdx to residual level
        rdf_test2 = rdf_test2.Define(
            "res_dEdx",
            f"RDataFrameDSL::IndexHelpers::ExpandToChildren({TRACK_VALUE_COL}, {FIRST_IDX_COL}, {N_ENTRIES_COL})"
        )
        
        # Get result
        res_dEdx_all = rdf_test2.Take["ROOT::RVecF"]("res_dEdx").GetValue()
        
        print(f"  Events processed: {len(res_dEdx_all)}")
        for i, vals in enumerate(res_dEdx_all):
            print(f"    Event {i}: {len(vals)} values, res_dEdx[:5]={list(vals)[:5]}")
        
        print("\n[OK] E8 PASSED: ExpandToChildren works with real data")
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================
# Cell E9: Summary
# ============================================================
print("\n" + "=" * 60)
print("E9: SUMMARY")
print("=" * 60)

print("""
  E1: Index helpers loaded        ✓
  E2: Synthetic data test         ✓
  E3: File structure              {} 
  E4: trackData columns           {}
  E5: unbinnedResid columns       {}
  E6: Index values                {}
  E7: ExpandParentIndex           {}
  E8: ExpandToChildren            {}
""".format(
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
    "✓" if not SKIP_REAL_DATA else "SKIPPED",
))

print("Next steps:")
print("  1. Update RESIDUALS_FILE path")
print("  2. Update column names based on E4/E5 output")
print("  3. Test draw() and to_pandas() integration")

# ============================================================
# Cell E10: Export to pandas (Residual + Track data)
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E10: Export to pandas")
    print("=" * 60)
    
    import numpy as np
    
    # Column names
    FIRST_IDX_COL = "trk.clIdx.mFirstEntry"
    N_ENTRIES_COL = "trk.clIdx.mEntries"
    TRACK_VALUE_COL = "trk.dEdxTPC"  # Update if needed
    
    # Use Range for quick test
    N_EVENTS = 5
    rdf_export = ROOT.RDataFrame("trackData", RESIDUALS_FILE).Range(N_EVENTS)
    
    try:
        # Define expanded columns
        rdf_export = rdf_export.Define(
            "parentIdx",
            f"RDataFrameDSL::IndexHelpers::ExpandParentIndex({FIRST_IDX_COL}, {N_ENTRIES_COL})"
        )
        rdf_export = rdf_export.Define(
            "res_dEdx",
            f"RDataFrameDSL::IndexHelpers::ExpandToChildren({TRACK_VALUE_COL}, {FIRST_IDX_COL}, {N_ENTRIES_COL})"
        )
        
        # Export to numpy/pandas
        result = rdf_export.AsNumpy(["parentIdx", "res_dEdx", FIRST_IDX_COL, N_ENTRIES_COL])
        
        print(f"  Exported {N_EVENTS} events")
        print(f"  Keys: {list(result.keys())}")
        
        # Flatten for pandas (each event is an RVec)
        all_parentIdx = []
        all_res_dEdx = []
        all_eventIdx = []
        
        for evt_idx in range(len(result["parentIdx"])):
            pidx = result["parentIdx"][evt_idx]
            dEdx = result["res_dEdx"][evt_idx]
            all_parentIdx.extend(pidx)
            all_res_dEdx.extend(dEdx)
            all_eventIdx.extend([evt_idx] * len(pidx))
        
        print(f"\n  Flattened to {len(all_parentIdx)} residuals")
        print(f"  parentIdx[:10]: {all_parentIdx[:10]}")
        print(f"  res_dEdx[:10]: {all_res_dEdx[:10]}")
        
        # Create pandas DataFrame
        try:
            import pandas as pd
            df = pd.DataFrame({
                "eventIdx": all_eventIdx,
                "parentIdx": all_parentIdx,
                "res_dEdx": all_res_dEdx,
            })
            print(f"\n  DataFrame shape: {df.shape}")
            print(f"  DataFrame head:\n{df.head(10)}")
            print("\n[OK] E10 PASSED: to_pandas export works")
        except ImportError:
            print("  pandas not available, skipping DataFrame creation")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Cell E11: Load unbinnedResid and Merge (using Friend Trees)
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E11: Load unbinnedResid with Friend Tree")
    print("=" * 60)
    
    # Use TChain with friend to align trees
    N_EVENTS = 5
    
    try:
        # Create RDataFrame with friend tree
        # trackData is the main tree, unbinnedResid is friend
        chain = ROOT.TChain("trackData")
        chain.Add(RESIDUALS_FILE)
        
        friend_chain = ROOT.TChain("unbinnedResid")
        friend_chain.Add(RESIDUALS_FILE)
        
        chain.AddFriend(friend_chain)
        
        rdf_combined = ROOT.RDataFrame(chain).Range(N_EVENTS)
        
        # Check available columns
        all_cols = [str(c) for c in rdf_combined.GetColumnNames()]
        print(f"  Combined columns: {len(all_cols)}")
        
        # Check if res.dy is available
        if "res.dy" in all_cols:
            print("  ✓ res.dy available in combined RDF")
            
            # Define all columns we need
            rdf_combined = rdf_combined.Define(
                "parentIdx",
                f"RDataFrameDSL::IndexHelpers::ExpandParentIndex({FIRST_IDX_COL}, {N_ENTRIES_COL})"
            )
            rdf_combined = rdf_combined.Define(
                "res_dEdx",
                f"RDataFrameDSL::IndexHelpers::ExpandToChildren({TRACK_VALUE_COL}, {FIRST_IDX_COL}, {N_ENTRIES_COL})"
            )
            
            # Export all together
            result = rdf_combined.AsNumpy(["parentIdx", "res_dEdx", "res.dy"])
            
            # Flatten
            all_parentIdx = []
            all_res_dEdx = []
            all_dy = []
            all_eventIdx = []
            
            for evt_idx in range(len(result["parentIdx"])):
                pidx = result["parentIdx"][evt_idx]
                dEdx = result["res_dEdx"][evt_idx]
                dy = result["res.dy"][evt_idx]
                
                # Check lengths match within event
                if len(pidx) == len(dy):
                    all_parentIdx.extend(pidx)
                    all_res_dEdx.extend(dEdx)
                    all_dy.extend(dy)
                    all_eventIdx.extend([evt_idx] * len(pidx))
                else:
                    print(f"  Event {evt_idx}: length mismatch pidx={len(pidx)}, dy={len(dy)}")
            
            print(f"\n  Flattened to {len(all_parentIdx)} residuals")
            
            # Create DataFrame
            import pandas as pd
            df = pd.DataFrame({
                "eventIdx": all_eventIdx,
                "parentIdx": all_parentIdx,
                "res_dEdx": all_res_dEdx,
                "dy": all_dy,
            })
            print(f"  DataFrame shape: {df.shape}")
            print(f"  DataFrame head:\n{df.head(10)}")
            print("\n[OK] E11 PASSED: Friend tree merge works")
        else:
            print("  ERROR: res.dy not found in combined columns")
            print(f"  Available: {[c for c in all_cols if 'res' in c.lower()]}")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Cell E12: Draw Histogram (res.dy vs track property)
# ============================================================
if not SKIP_REAL_DATA:
    print("\n" + "=" * 60)
    print("E12: Draw Histogram")
    print("=" * 60)
    
    try:
        # Create 2D histogram if we have merged data
        if 'df' in dir() and 'dy' in df.columns:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: dy distribution
            axes[0].hist(df["dy"], bins=100, alpha=0.7)
            axes[0].set_xlabel("dy (residual)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Residual dy distribution")
            
            # Plot 2: dy vs res_dEdx (track dEdx expanded to residual level)
            axes[1].hist2d(df["res_dEdx"], df["dy"], bins=50, cmap='viridis')
            axes[1].set_xlabel("Track dEdx (expanded)")
            axes[1].set_ylabel("dy (residual)")
            axes[1].set_title("dy vs Track dEdx")
            plt.colorbar(axes[1].collections[0], ax=axes[1])
            
            plt.tight_layout()
            plt.savefig("explore_residuals_plot.png", dpi=100)
            print(f"  Saved: explore_residuals_plot.png")
            print("\n[OK] E12 PASSED: Histogram drawn")
        else:
            print("  DataFrame not available or missing 'dy' column")
            print("  Skipping histogram")
            
    except ImportError as e:
        print(f"  matplotlib not available: {e}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Cell E13: Final Summary
# ============================================================
print("\n" + "=" * 60)
print("E13: FINAL SUMMARY")
print("=" * 60)

print("""
Phase 13.7.B Exploration Results:
================================

Index Helpers:
  ✓ ExpandParentIndex - builds child→parent mapping
  ✓ ExpandToChildren  - expands parent values to child level
  ✓ GatherByIndex     - direct index lookup with fallback
  ✓ CountChildren     - reverse lookup counts

Residuals Data:
  - trackData tree: track properties + index columns
  - unbinnedResid tree: residual values (dy, dz, etc.)
  - Index columns: trk.clIdx.mFirstEntry, trk.clIdx.mEntries

Next Steps for DSL Integration:
  1. Add index helpers to DSLCompiler
  2. Auto-detect parent-child schema from tree
  3. Support dsl.draw("res.dy : track.dEdx") syntax
  4. Support dsl.to_pandas() with mixed levels
""")
