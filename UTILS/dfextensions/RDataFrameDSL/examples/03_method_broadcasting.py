#!/usr/bin/env python3
"""
Example 03: Method Broadcasting (Phase 8)

★★★ KEY DEMO FOR ROOT TEAM ★★★

Demonstrates:
- Element-wise method calls on RVec<Object>
- tracks.Pt() → RVec<double>
- Slice then broadcast: tracks[:3].Pt()
- Filter then broadcast: tracks[tracks.Pt() > 1.0].Eta()
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT
from RDataFrameDSL import DSLCompiler

print("=" * 60)
print("Example 03: Method Broadcasting (Phase 8)")
print("=" * 60)

# =============================================================================
# The 6-Line Demo (as suggested by Architecture Reviewer)
# =============================================================================

print("\n★★★ THE 6-LINE DEMO ★★★\n")

schema = {'tracks': 'RVec<TLorentzVector>'}
dsl = DSLCompiler(schema)
dsl.define("track_pts", "tracks.Pt()")
dsl.define("lead_pt", "tracks[:1].Pt()")
dsl.define("high_pt_eta", "tracks[tracks.Pt() > 1.0].Eta()")

print("Code:")
print('  schema = {"tracks": "RVec<TLorentzVector>"}')
print('  dsl = DSLCompiler(schema)')
print('  dsl.define("track_pts", "tracks.Pt()")')
print('  dsl.define("lead_pt", "tracks[:1].Pt()")')
print('  dsl.define("high_pt_eta", "tracks[tracks.Pt() > 1.0].Eta()")')

print("\n" + "=" * 60)
print("Generated C++ Code:")
print("=" * 60)
print(dsl.preview())

# =============================================================================
# Extended Demo: All Phase 8 Features
# =============================================================================

print("=" * 60)
print("Extended Demo: All Phase 8 Features")
print("=" * 60)

# Fresh compiler with more examples
schema = {'tracks': 'RVec<TLorentzVector>'}
dsl = DSLCompiler(schema)

# Basic method broadcasting
dsl.define("all_pt", "tracks.Pt()")
dsl.define("all_eta", "tracks.Eta()")
dsl.define("all_phi", "tracks.Phi()")
dsl.define("all_mass", "tracks.M()")

# Component access
dsl.define("all_px", "tracks.Px()")
dsl.define("all_py", "tracks.Py()")
dsl.define("all_pz", "tracks.Pz()")
dsl.define("all_e", "tracks.E()")

# Slice then broadcast
dsl.define("lead3_pt", "tracks[:3].Pt()")
dsl.define("last2_eta", "tracks[-2:].Eta()")

# Filter then broadcast
dsl.define("central_pt", "tracks[tracks.Eta() < 1.0].Pt()")
dsl.define("high_pt_tracks_eta", "tracks[tracks.Pt() > 2.0].Eta()")

# Method returning object (TVector3)
dsl.define("all_vect", "tracks.Vect()")

print("\nDefined expressions:")
print("  # Basic broadcasting")
print('  all_pt   = tracks.Pt()    → RVec<double>')
print('  all_eta  = tracks.Eta()   → RVec<double>')
print('  all_phi  = tracks.Phi()   → RVec<double>')
print()
print("  # Slice then broadcast")
print('  lead3_pt = tracks[:3].Pt()     → First 3 tracks\' pT')
print('  last2_eta = tracks[-2:].Eta()  → Last 2 tracks\' eta')
print()
print("  # Filter then broadcast")
print('  central_pt = tracks[tracks.Eta() < 1.0].Pt()')
print('  high_pt_tracks_eta = tracks[tracks.Pt() > 2.0].Eta()')
print()
print("  # Method returning object")
print('  all_vect = tracks.Vect()  → RVec<TVector3>')

# =============================================================================
# Compare: DSL vs Raw C++
# =============================================================================

print("\n" + "=" * 60)
print("Comparison: DSL vs Raw C++")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ DSL (Python-like)                │ Raw C++ (verbose)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ tracks.Pt()                      │ ROOT::RVec<double> result;               │
│                                  │ result.reserve(tracks.size());           │
│                                  │ for (const auto& t : tracks)             │
│                                  │     result.push_back(t.Pt());            │
│                                  │ return result;                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ tracks[:3].Pt()                  │ auto sliced = Take(tracks, 3);           │
│                                  │ ROOT::RVec<double> result;               │
│                                  │ for (const auto& t : sliced)             │
│                                  │     result.push_back(t.Pt());            │
├─────────────────────────────────────────────────────────────────────────────┤
│ tracks[tracks.Pt() > 1.0].Eta()  │ ROOT::RVec<double> mask_result;          │
│                                  │ for (const auto& t : tracks)             │
│                                  │     if (t.Pt() > 1.0)                    │
│                                  │         mask_result.push_back(t.Eta());  │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# Apply to RDataFrame
# =============================================================================

print("=" * 60)
print("Execution with RDataFrame:")
print("=" * 60)

try:
    rdf = ROOT.RDataFrame("Events", "test_tracks.root")
    rdf = dsl.apply(rdf)
    
    # Get results
    results = rdf.AsNumpy(["all_pt", "lead3_pt", "all_eta"])
    
    print(f"\nProcessed {len(results['all_pt'])} events")
    
    print(f"\nFirst 5 events:")
    for i in range(min(5, len(results['all_pt']))):
        pt_arr = list(results['all_pt'][i])
        lead3 = list(results['lead3_pt'][i])
        eta_arr = list(results['all_eta'][i])
        
        print(f"\n  Event {i}:")
        print(f"    tracks.Pt()     = [{', '.join(f'{x:.2f}' for x in pt_arr[:5])}{'...' if len(pt_arr) > 5 else ''}]")
        print(f"    tracks[:3].Pt() = [{', '.join(f'{x:.2f}' for x in lead3)}]")
        print(f"    tracks.Eta()    = [{', '.join(f'{x:.2f}' for x in eta_arr[:5])}{'...' if len(eta_arr) > 5 else ''}]")

except Exception as e:
    print(f"\nNote: Could not run on data file: {e}")
    print("Run 'python create_test_data.py' first to generate test files.")

print("\n" + "=" * 60)
print("Key Takeaway: Phase 8 enables Python-style method broadcasting")
print("on RVec<Object> with automatic C++ code generation!")
print("=" * 60)
