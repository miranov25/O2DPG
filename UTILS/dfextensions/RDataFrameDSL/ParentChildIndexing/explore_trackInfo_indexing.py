#!/usr/bin/env python3
"""
IPython exploration: Verify trackInfo indexing in unbinnedResid
================================================================

Run interactively:
    ipython -i explore_trackInfo_indexing.py

Or cell by cell in Jupyter.
"""

# %% Cell 1: Setup
import ROOT
print(f"ROOT version: {ROOT.gROOT.GetVersion()}")

RESIDUALS_FILE = "o2residuals_tpc.root"

# %% Cell 2: Load unbinnedResid tree - first event
rdf = ROOT.RDataFrame("unbinnedResid", RESIDUALS_FILE).Range(1)

# Get columns we need
cols = [str(c) for c in rdf.GetColumnNames()]
print("Available columns with 'trackInfo' or 'res':")
for c in cols:
    if "trackInfo" in c or c.startswith("res."):
        print(f"  {c}")

# %% Cell 3: Get trackInfo.idxFirstResidual and res.dy for first event
result = rdf.AsNumpy(["trackInfo.idxFirstResidual", "trackInfo.nResiduals", "res.dy"])

idxFirst = result["trackInfo.idxFirstResidual"][0]  # First event
nResiduals = result["trackInfo.nResiduals"][0]
dy = result["res.dy"][0]

print(f"\nFirst event:")
print(f"  Number of tracks: {len(idxFirst)}")
print(f"  Number of residuals (res.dy): {len(dy)}")

# %% Cell 4: Check the indexing
print(f"\nFirst 10 tracks:")
print(f"  idxFirstResidual: {[int(idxFirst[i]) for i in range(min(10, len(idxFirst)))]}")
# nResiduals is UChar_t - use ord() or direct byte value
print(f"  nResiduals:       {[ord(nResiduals[i]) if isinstance(nResiduals[i], str) else int(nResiduals[i]) for i in range(min(10, len(nResiduals)))]}")

# Compute nEntries from consecutive idxFirstResidual
nEntries_computed = []
for i in range(len(idxFirst) - 1):
    nEntries_computed.append(int(idxFirst[i+1]) - int(idxFirst[i]))
# Last track: goes to end of res.dy
nEntries_computed.append(len(dy) - int(idxFirst[len(idxFirst)-1]))

print(f"\nComputed nEntries (from consecutive idx):")
print(f"  First 10: {nEntries_computed[:10]}")

# %% Cell 5: Verify total
total_from_computed = sum(nEntries_computed)
print(f"\nVerification:")
print(f"  sum(nEntries_computed) = {total_from_computed}")
print(f"  len(res.dy)            = {len(dy)}")
print(f"  Match: {total_from_computed == len(dy)}")

# %% Cell 6: Compare with nResiduals (the buggy count)
# nResiduals is UChar_t
total_from_nResiduals = sum(ord(nResiduals[i]) if isinstance(nResiduals[i], str) else int(nResiduals[i]) for i in range(len(nResiduals)))
print(f"\nBuggy nResiduals:")
print(f"  sum(nResiduals) = {total_from_nResiduals}")
print(f"  This is less than res.dy because nResiduals counts only subset")

# %% Cell 7: Test ExpandParentIndex with correct indexing
print("\n" + "="*60)
print("Testing ExpandParentIndex with corrected indexing")
print("="*60)

# Load index helpers
ROOT.gSystem.Load("libIndexHelpers.so")
ROOT.gInterpreter.Declare('#include "index_helpers.h"')

# We need a modified helper that takes only idxFirstResidual and total size
# Or compute nEntries in Python and pass to existing helper

import numpy as np
idxFirst_arr = np.array(idxFirst, dtype=np.int32)
nEntries_arr = np.array(nEntries_computed, dtype=np.int32)

# Convert to RVec
ROOT.gInterpreter.ProcessLine(f'''
ROOT::RVecI py_idxFirst = {{{", ".join(map(str, idxFirst_arr[:10]))}}};
ROOT::RVecI py_nEntries = {{{", ".join(map(str, nEntries_arr[:10]))}}};
auto py_parentIdx = RDataFrameDSL::IndexHelpers::ExpandParentIndex(py_idxFirst, py_nEntries);
std::cout << "parentIdx size: " << py_parentIdx.size() << std::endl;
std::cout << "parentIdx[0:20]: ";
for (int i = 0; i < 20 && i < py_parentIdx.size(); i++) std::cout << py_parentIdx[i] << " ";
std::cout << std::endl;
''')

# %% Cell 8: Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Correct indexing approach:
1. Use trackInfo.idxFirstResidual from unbinnedResid tree
2. Compute nEntries[i] = idxFirstResidual[i+1] - idxFirstResidual[i]
3. For last track: nEntries[-1] = len(res.dy) - idxFirstResidual[-1]
4. Do NOT use trackInfo.nResiduals (buggy, counts subset only)

This needs a new helper or preprocessing step.
""")
