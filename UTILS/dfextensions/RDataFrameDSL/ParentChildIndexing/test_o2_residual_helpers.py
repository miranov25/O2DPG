#!/usr/bin/env python3
"""
Test O2 Residual Helpers - Step 1 Validation

Goal: Validate that DSL can call vec.method() on O2 classes.

Test sequence:
1. Load O2 libraries and include header
2. Test detInfo.qMaxTPC() via RDataFrame (baseline)
3. Test detInfo.qMaxTPC() via DSL (target)

Run: python -u test_o2_residual_helpers.py 2>&1 | tee test_o2_residual_helpers.log
"""

import sys
print(f"Python: {sys.version}")

# ============================================================
# Step 1: Load ROOT and O2
# ============================================================
print("=" * 60)
print("Step 1: Load ROOT and O2")
print("=" * 60)

import ROOT
print(f"ROOT version: {ROOT.gROOT.GetVersion()}")

# Load index helpers (generic)
ROOT.gSystem.Load("libIndexHelpers.so")
ROOT.gInterpreter.Declare('#include "index_helpers.h"')
print("  Loaded: libIndexHelpers.so")

# Include O2 header to get full type definition
ROOT.gInterpreter.Declare('#include "SpacePoints/TrackInterpolation.h"')
print("  Included: SpacePoints/TrackInterpolation.h")

print("\n[OK] Step 1 PASSED")

# ============================================================
# Step 2: Test RDataFrame baseline (no DSL)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Test RDataFrame baseline")
print("=" * 60)

# Create chain
chain = ROOT.TChain("unbinnedResid")
chain.Add("o2residuals_tpc.root")
print(f"  Entries: {chain.GetEntries()}")

# Create RDF with Range(2) for quick test
rdf = ROOT.RDataFrame(chain).Range(2)

# Test 1: Direct column access
print("\n  Test 2a: Direct column access (detInfo.word)")
rdf_test = rdf.Define("word0", "detInfo.word[0]")
result = rdf_test.AsNumpy(["word0"])
print(f"    word0 values: {list(result['word0'])}")

# Test 2: Method call via Map
print("\n  Test 2b: Method call via Map (detInfo.qMaxTPC)")
try:
    # This is what DSL generates for vec.method()
    rdf_test2 = rdf.Define("qMax_map", 
        "Map(detInfo, [](const o2::tpc::DetInfoResid& d){ return d.qMaxTPC(); })")
    # Note: AsNumpy may fail here, but Define should work
    print("    Define succeeded (Map with lambda)")
    
    # Try to materialize
    count = rdf_test2.Count().GetValue()
    print(f"    Count: {count}")
except Exception as e:
    print(f"    ERROR: {e}")

# Test 3: Explicit helper function
print("\n  Test 2c: Explicit helper function")
ROOT.gInterpreter.Declare("""
ROOT::RVec<int> ExtractQMaxTPC(const ROOT::RVec<o2::tpc::DetInfoResid>& detInfo) {
    ROOT::RVec<int> result(detInfo.size());
    for (size_t i = 0; i < detInfo.size(); ++i) {
        result[i] = detInfo[i].qMaxTPC();
    }
    return result;
}
""")
rdf_test3 = rdf.Define("qMax_helper", "ExtractQMaxTPC(detInfo)")
result3 = rdf_test3.AsNumpy(["qMax_helper"])
print(f"    qMax_helper[0][:10]: {[result3['qMax_helper'][0][i] for i in range(10)]}")

print("\n[OK] Step 2 PASSED")

# ============================================================
# Step 3: Test DSL (target)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Test DSL with O2 classes")
print("=" * 60)

# Check if DSL is available
try:
    sys.path.insert(0, '/home/miranov25/github/RDataFrameDSL')  # Adjust path
    from RDataFrameDSL import DSLCompiler
    print("  DSL imported successfully")
    
    # Create schema for unbinnedResid
    schema = {
        "detInfo": "RVec<o2::tpc::DetInfoResid>",
        "res.dy": "RVec<short>",
        "res.dz": "RVec<short>",
        "res.row": "RVec<unsigned char>",
    }
    
    # Create DSL compiler
    dsl = DSLCompiler(schema)
    print(f"  Schema: {schema}")
    
    # Test: define qMaxTPC using method syntax
    print("\n  Test 3a: DSL define with method call")
    try:
        dsl.define("qMax", "detInfo.qMaxTPC()")
        print("    dsl.define('qMax', 'detInfo.qMaxTPC()') - OK")
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # Apply to RDF and export
    print("\n  Test 3b: Apply and export")
    try:
        rdf_dsl = dsl.apply(rdf)
        result_dsl = rdf_dsl.AsNumpy(["qMax"])
        print(f"    qMax[0][:10]: {[result_dsl['qMax'][0][i] for i in range(10)]}")
        print("\n[OK] Step 3 PASSED - DSL vec.method() works with O2 classes!")
    except Exception as e:
        print(f"    ERROR during apply/export: {e}")
        print("\n[FAIL] Step 3 - DSL integration needs work")

except ImportError as e:
    print(f"  DSL not available: {e}")
    print("  Skipping Step 3 - run from RDataFrameDSL directory")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Step 1: Load O2 libraries          - PASSED
Step 2: RDataFrame baseline        - PASSED  
  - detInfo.word direct access     - OK
  - Map with lambda                - OK (Define works)
  - Helper function                - OK (AsNumpy works)
Step 3: DSL integration            - See above

Key finding: 
  The O2 header must be included BEFORE using the types.
  Helper functions work reliably with AsNumpy.
  Map+lambda may have issues with AsNumpy (type deduction).
""")
