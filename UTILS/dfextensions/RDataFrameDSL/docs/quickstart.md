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

> **Note:** The schema is a flat dictionary mapping column names to C++ types. Values can be scalar types (`double`, `int`) or vector types (`RVec<T>`).

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

# Method broadcasting
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

---

## Visualization (Phase 12)

Generate QA plots with statistical annotations:

```python
# Define pull distributions
dsl.define("dy_pull", "(dy - dy_fit) / dy_err")
dsl.define("dz_pull", "(dz - dz_fit) / dz_err")

# Create plots with stats
specs = [
    {'expr': 'dy_pull', 'bins': 50, 'range': (-5, 5)},
    {'expr': 'dz_pull', 'bins': 50, 'range': (-5, 5)},
]

results = dsl.draw_figures(
    specs, rdf,
    show_statistics=True,   # Add μ, σ, n box
    show_expected=True,     # Add N(0,1) overlay for pulls
)
```

**Pull Detection:**
- Auto-detected if `'pull'` appears in expression name
- Override with `is_pull` in plot spec: `{'expr': 'residual', 'is_pull': True}`

---

## Export to AliasDataFrame (Phase 12)

Migrate DSL definitions to pandas-based workflows:

```python
# Export DSL definitions as schema
schema = dsl.to_aliasdf(
    include=['pt_gev', 'good_track'],
    dtype_map={'pt_gev': 'float32'}
)

# Apply to AliasDataFrame
from AliasDataFrame import AliasDataFrame
adf = AliasDataFrame(df)
adf.apply_schema(schema)
```

> **Note:** C++ operators are converted to Python equivalents (`&&` → `&`, `||` → `|`). Complex expressions may need manual review.

---

## Arrow Integration (Phase 13)

Export/import data via PyArrow for memory-efficient workflows:

```python
import pyarrow.parquet as pq

# Export RDataFrame to Arrow (pass rdf explicitly)
table = dsl.to_arrow(rdf=rdf, columns=['pt', 'eta'])

# Save as Parquet
pq.write_table(table, 'output.parquet')

# Load and create new DSL
table = pq.read_table('output.parquet')
new_dsl = DSLCompiler.from_arrow(table)
```

**RVec Handling:**
- `flatten_rvec=False` (default): Preserve as Arrow ListArray (jagged structure)
- `flatten_rvec=True`: Flatten to 1D array (loses event structure)

> **Note:** Round-trip for jagged (RVec) columns may have limitations. Use `flatten_rvec=True` for guaranteed compatibility.

---

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

---

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

---

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

---

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

---

## Next Steps

- See [API Reference](api_reference.md) for complete method documentation
- See [Expression Reference](expressions.md) for complete DSL syntax
- See [Architecture](ARCHITECTURE.md) for implementation details
- See [examples/](../examples/) for more complex examples
