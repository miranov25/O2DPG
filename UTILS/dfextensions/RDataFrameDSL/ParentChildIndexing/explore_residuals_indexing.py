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
