#!/usr/bin/env python3
"""
Example 01: Basic Scalar Expressions

Demonstrates:
- Creating DSLCompiler with schema
- Defining scalar expressions
- Viewing generated C++ code
- Applying to RDataFrame
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ROOT
from RDataFrameDSL import DSLCompiler

print("=" * 60)
print("Example 01: Basic Scalar Expressions")
print("=" * 60)

# =============================================================================
# Schema Definition
# =============================================================================

schema = {
    "px": "double",
    "py": "double",
    "pz": "double",
    "energy": "double",
}

print("\nSchema:")
for name, dtype in schema.items():
    print(f"  {name}: {dtype}")

# =============================================================================
# Create DSLCompiler and Define Expressions
# =============================================================================

dsl = DSLCompiler(schema)

# Transverse momentum
dsl.define("pt", "sqrt(px**2 + py**2)")

# Total momentum
dsl.define("p", "sqrt(px**2 + py**2 + pz**2)")

# Pseudorapidity (using theta = atan2(pt, pz), but inline pt calculation)
dsl.define("eta", "-log(tan(atan2(sqrt(px**2 + py**2), pz)/2))")

# Azimuthal angle
dsl.define("phi", "atan2(py, px)")

# Invariant mass
dsl.define("mass", "sqrt(energy**2 - px**2 - py**2 - pz**2)")

# Boolean cut (inline pt calculation)
dsl.define("is_central", "abs(-log(tan(atan2(sqrt(px**2 + py**2), pz)/2))) < 2.5")

print("\nDefined expressions:")
print("  pt         = sqrt(px**2 + py**2)")
print("  p          = sqrt(px**2 + py**2 + pz**2)")
print("  eta        = -log(tan(atan2(sqrt(px**2+py**2), pz)/2))")
print("  phi        = atan2(py, px)")
print("  mass       = sqrt(energy**2 - px**2 - py**2 - pz**2)")
print("  is_central = abs(eta) < 2.5")

# =============================================================================
# Preview Generated C++
# =============================================================================

print("\n" + "=" * 60)
print("Generated C++ Code:")
print("=" * 60)
print(dsl.preview())

# =============================================================================
# Apply to RDataFrame
# =============================================================================

print("=" * 60)
print("Execution with RDataFrame:")
print("=" * 60)

# Create RDataFrame from test file
try:
    rdf = ROOT.RDataFrame("Events", "test_scalars.root")
    
    # Apply DSL definitions
    rdf = dsl.apply(rdf)
    
    # Get some results
    results = rdf.AsNumpy(["pt", "eta", "phi", "mass", "is_central"])
    
    print(f"\nProcessed {len(results['pt'])} events")
    print(f"\nFirst 5 events:")
    print(f"  {'pt':>10} {'eta':>10} {'phi':>10} {'mass':>10} {'central':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for i in range(min(5, len(results['pt']))):
        print(f"  {results['pt'][i]:10.4f} {results['eta'][i]:10.4f} "
              f"{results['phi'][i]:10.4f} {results['mass'][i]:10.4f} "
              f"{results['is_central'][i]:>8}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean pT:   {results['pt'].mean():.4f}")
    print(f"  Mean eta:  {results['eta'].mean():.4f}")
    print(f"  Central fraction: {results['is_central'].mean()*100:.1f}%")
    
except Exception as e:
    print(f"\nNote: Could not run on data file: {e}")
    print("Run 'python create_test_data.py' first to generate test files.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
