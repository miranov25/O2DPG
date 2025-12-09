#!/usr/bin/env python3
"""
Example 04: DSL vs Raw RDataFrame Comparison

Side-by-side comparison showing:
- Code verbosity reduction
- Safety improvements (NaN vs crash)
- Equivalent results
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT
from RDataFrameDSL import DSLCompiler

print("=" * 60)
print("Example 04: DSL vs Raw RDataFrame Comparison")
print("=" * 60)

# =============================================================================
# Test Case 1: Scalar pT Calculation
# =============================================================================

print("\n" + "-" * 60)
print("Test 1: Scalar pT Calculation")
print("-" * 60)

# DSL approach
print("\n[DSL Approach]")
print('  dsl.define("pt", "sqrt(px**2 + py**2)")')

schema = {"px": "double", "py": "double"}
dsl = DSLCompiler(schema)
dsl.define("pt", "sqrt(px**2 + py**2)")

print("\n[Generated C++]")
print(dsl.preview())

# Raw approach
print("[Raw C++ in RDataFrame]")
print('  rdf.Define("pt", "std::sqrt(px*px + py*py)")')

# =============================================================================
# Test Case 2: Safe First Element Access
# =============================================================================

print("\n" + "-" * 60)
print("Test 2: Safe First Element Access")
print("-" * 60)

# DSL approach - safe
print("\n[DSL Approach - SAFE]")
print('  dsl.define("lead_pt", "pt[0]")')
print('  → Returns NaN if vector is empty')

schema = {"pt": "RVec<double>"}
dsl = DSLCompiler(schema)
dsl.define("lead_pt", "pt[0]")

print("\n[Generated C++]")
print(dsl.preview())

# Raw approach - DANGEROUS
print("[Raw C++ - DANGEROUS]")
print('  rdf.Define("lead_pt", "pt[0]")')
print('  → CRASHES if vector is empty!')
print()
print('  Safe raw version:')
print('  rdf.Define("lead_pt", "pt.size() > 0 ? pt[0] : std::nan(\\"\\")")')

# =============================================================================
# Test Case 3: Slicing
# =============================================================================

print("\n" + "-" * 60)
print("Test 3: First N Elements (Slicing)")
print("-" * 60)

# DSL approach
print("\n[DSL Approach]")
print('  dsl.define("first3", "pt[:3]")')

schema = {"pt": "RVec<double>"}
dsl = DSLCompiler(schema)
dsl.define("first3", "pt[:3]")

print("\n[Generated C++]")
print(dsl.preview())

# Raw approach
print("[Raw C++ - Verbose]")
print('  rdf.Define("first3", "ROOT::VecOps::Take(pt, std::min((size_t)3, pt.size()))")')

# =============================================================================
# Test Case 4: Boolean Masking
# =============================================================================

print("\n" + "-" * 60)
print("Test 4: Boolean Masking")
print("-" * 60)

# DSL approach
print("\n[DSL Approach]")
print('  dsl.define("high_pt", "pt[pt > 5.0]")')

schema = {"pt": "RVec<double>"}
dsl = DSLCompiler(schema)
dsl.define("high_pt", "pt[pt > 5.0]")

print("\n[Generated C++]")
print(dsl.preview())

# Raw approach
print("[Raw C++ - Verbose]")
print('  rdf.Define("high_pt", "pt[pt > 5.0]")')
print('  → Actually similar! RVec supports this natively.')

# =============================================================================
# Test Case 5: Method Broadcasting (Phase 8)
# =============================================================================

print("\n" + "-" * 60)
print("Test 5: Method Broadcasting (Phase 8) ★")
print("-" * 60)

# DSL approach
print("\n[DSL Approach]")
print('  dsl.define("track_pts", "tracks.Pt()")')

schema = {"tracks": "RVec<TLorentzVector>"}
dsl = DSLCompiler(schema)
dsl.define("track_pts", "tracks.Pt()")

print("\n[Generated C++]")
print(dsl.preview())

# Raw approach
print("[Raw C++ - Very Verbose]")
raw_code = '''
rdf.Define("track_pts", R"(
    ROOT::RVec<double> result;
    result.reserve(tracks.size());
    for (const auto& t : tracks) {
        result.push_back(t.Pt());
    }
    return result;
)")'''
print(raw_code)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("Summary: Lines of Code Comparison")
print("=" * 60)

summary = """
┌────────────────────────────────┬──────────┬──────────┐
│ Operation                      │ DSL      │ Raw C++  │
├────────────────────────────────┼──────────┼──────────┤
│ Scalar math: sqrt(px**2+py**2) │ 1 line   │ 1 line   │
│ Safe indexing: pt[0]           │ 1 line   │ 3 lines  │
│ Slicing: pt[:3]                │ 1 line   │ 1 line*  │
│ Negative index: pt[-1]         │ 1 line   │ 2 lines  │
│ Boolean mask: pt[pt > 5]       │ 1 line   │ 1 line   │
│ Method broadcast: tracks.Pt()  │ 1 line   │ 6 lines  │
│ Slice+broadcast: tracks[:3].Pt()│ 1 line   │ 8 lines  │
│ Filter+broadcast               │ 1 line   │ 8 lines  │
└────────────────────────────────┴──────────┴──────────┘
* Using ROOT::VecOps::Take (but more verbose)

Key Benefits:
✓ Shorter code
✓ Safer (NaN instead of crash)
✓ Python-like syntax (familiar to physicists)
✓ Generated C++ is debuggable
✓ Can export to .C macro for sharing
"""
print(summary)

print("=" * 60)
print("Done!")
print("=" * 60)
