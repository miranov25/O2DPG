# Quick Start Guide

Get up and running with RDataFrameDSL in 5 minutes.

## Installation

```bash
# Add to your Python path
export PYTHONPATH=/path/to/O2DPG/UTILS/dfextensions:$PYTHONPATH
```

## Basic Usage

### 1. Import and Create Compiler

```python
import ROOT
from RDataFrameDSL import DSLCompiler

# Define your schema: column_name → C++ type
schema = {
    "px": "double",
    "py": "double",
    "pz": "double",
    "pt": "RVec<double>",
    "tracks": "RVec<TLorentzVector>",
}

dsl = DSLCompiler(schema)
```

### 2. Define Computed Columns

```python
# Scalar expressions
dsl.define("event_pt", "sqrt(px**2 + py**2)")
dsl.define("event_p", "sqrt(px**2 + py**2 + pz**2)")

# RVec operations
dsl.define("n_tracks", "pt.size()")
dsl.define("lead_pt", "pt[0]")           # First element (NaN if empty)
dsl.define("last_pt", "pt[-1]")          # Last element

# Slicing
dsl.define("first3", "pt[:3]")           # First 3 elements
dsl.define("high_pt", "pt[pt > 1.0]")    # Boolean mask

# Method broadcasting (Phase 8)
dsl.define("track_pts", "tracks.Pt()")   # Element-wise Pt()
dsl.define("track_etas", "tracks.Eta()") # Element-wise Eta()
```

### 3. Apply to RDataFrame

```python
# Create RDataFrame
rdf = ROOT.RDataFrame("Events", "data.root")

# Apply all definitions at once
rdf = dsl.apply(rdf)

# Use the new columns
results = rdf.AsNumpy(["event_pt", "n_tracks", "track_pts"])
print(f"First event pt: {results['event_pt'][0]}")
```

### 4. Export for Debugging (Optional)

```python
# Preview generated C++ without compiling
print(dsl.preview())

# Export to .C macro file
dsl.export_macro("my_analysis.C", include_test=True)
```

## Complete Example

```python
import ROOT
from RDataFrameDSL import DSLCompiler

# Schema
schema = {
    "px": "double",
    "py": "double",
    "tracks": "RVec<TLorentzVector>",
}

# Create compiler and define columns
dsl = DSLCompiler(schema)
dsl.define("event_pt", "sqrt(px**2 + py**2)")
dsl.define("n_tracks", "tracks.size()")
dsl.define("track_pts", "tracks.Pt()")
dsl.define("lead_track_pt", "tracks[:1].Pt()")
dsl.define("high_pt_tracks", "tracks[tracks.Pt() > 2.0]")

# Apply to data
rdf = ROOT.RDataFrame("Events", "data.root")
rdf = dsl.apply(rdf)

# Make histogram
hist = rdf.Histo1D(("h_pt", "Event pT;pT [GeV];Events", 100, 0, 50), "event_pt")
hist.Draw()
```

## Common Patterns

### Physics Calculations

```python
# Transverse momentum
dsl.define("pt", "sqrt(px**2 + py**2)")

# Pseudorapidity
dsl.define("eta", "-log(tan(theta/2))")

# Invariant mass (if you have E, px, py, pz)
dsl.define("mass", "sqrt(E**2 - px**2 - py**2 - pz**2)")
```

### Track Selection

```python
# Get Pt of all tracks
dsl.define("track_pts", "tracks.Pt()")

# Select high-pT tracks
dsl.define("high_pt_tracks", "tracks[tracks.Pt() > 1.0]")

# Get eta of high-pT tracks
dsl.define("high_pt_eta", "tracks[tracks.Pt() > 1.0].Eta()")

# Leading tracks
dsl.define("lead3_pt", "tracks[:3].Pt()")
```

### Safe Indexing

```python
# These return NaN instead of crashing on empty vectors
dsl.define("first", "pt[0]")
dsl.define("last", "pt[-1]")
dsl.define("tenth", "pt[9]")  # NaN if fewer than 10 elements
```

## Error Messages

The DSL provides helpful error messages:

```python
dsl.define("bad", "trakcs.Pt()")  # Typo!
# Error: Unknown variable 'trakcs'
# Suggestions: Did you mean 'tracks'?

dsl.define("bad", "tracks.Unknown()")
# Error: Method 'Unknown' not found on element type 'TLorentzVector'
# Suggestions: Did you mean 'Pt', 'Eta', 'Phi'?
```

## Alias Referencing

Aliases can reference other previously defined aliases:

```python
schema = {"px": "double", "py": "double", "pz": "double"}
dsl = DSLCompiler(schema)

# Define pt
dsl.define("pt", "sqrt(px**2 + py**2)")

# Use pt in subsequent definition - WORKS!
dsl.define("eta", "-log(tan(atan2(pt, pz)/2))")

# Chain further
dsl.define("is_central", "abs(eta) < 2.5")
```

This makes expressions much cleaner and more readable.

## Next Steps

- See [expressions.md](expressions.md) for complete DSL syntax
- See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details
- See [examples/](../examples/) for more complex examples
