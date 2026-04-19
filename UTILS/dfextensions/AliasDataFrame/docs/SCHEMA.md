# AliasDataFrame Schema Guide

## Overview

The **schema** is the **single source of truth** for all AliasDataFrame metadata. It centralizes column definitions, aliases, compression settings, subframe relationships, and versioning into one unified structure.

**Key Benefits:**

- Reproducibility: Schema defines exact dtypes, aliases, and compression formulas
- Persistence: Schema round-trips through JSON, ROOT, and Parquet formats
- C++ Compatibility: Schema can be embedded in ROOT files for use with TTree::Draw
- Memory Optimization: Schema-driven dtype restoration prevents float16→float32 bloat

### Schema Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python (AliasDataFrame)                     │
│  ┌─────────────┐                                                    │
│  │   _schema   │  ← Single source of truth                         │
│  │  (in-memory)│                                                    │
│  └──────┬──────┘                                                    │
│         │                                                           │
│    export_schema()                                                  │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │    JSON     │  ← Portable, human-readable                       │
│  │   (dict)    │                                                    │
│  └──────┬──────┘                                                    │
└─────────┼───────────────────────────────────────────────────────────┘
          │
          ├──────────────────────┬──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │  .json file │       │  ROOT file  │       │  Parquet    │
   │             │       │  (TNamed)   │       │  (.schema)  │
   └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
          │                      │                      │
          │              load_schema_from_root()        │
          │                      │         load_schema_from_parquet_metadata()
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    apply_schema()                           │
   │              Restores dtypes, aliases, compression          │
   └─────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌─────────────┐
   │   C++ ROOT  │  ← TTree::Draw with aliases
   │  (optional) │
   └─────────────┘
```

---

## Schema Structure

The `_schema` dictionary has three main sections:

```python
_schema = {
    "columns": {
        # Physical columns and aliases
        "pt": {"dtype": "float32", "expr": None},           # Physical column
        "eta": {"dtype": "float32", "expr": "log(tan(theta/2))", "constant": False},  # Alias
        "scale": {"dtype": "float32", "expr": "1.5", "constant": True}   # Constant alias
    },
    "compression": {
        "__meta__": {
            "schema_version": 1,
            "state_machine": "CompressionState.v1"
        },
        "dy": {
            "compress": "round(asinh(dy)*40)",
            "decompress": "sinh(dy_c/40.)",
            "compressed_dtype": "int16",
            "decompressed_dtype": "float16",
            "compressed_col": "dy_c",
            "state": "compressed"
        }
    },
    "subframes": {
        "track": {"index": "track_index"},
        "collision": {"index": "collision_index"}
    }
}
```

### Columns Section

Each entry in `columns` describes either a physical column or an alias:

| Field | Type | Description |
|-------|------|-------------|
| `dtype` | str | Column data type (e.g., "float32", "int16") |
| `expr` | str or None | Expression for aliases; `None` for physical columns |
| `constant` | bool | If `True`, alias is a constant value |

**Physical columns** have `expr: None` and represent actual DataFrame columns.

**Aliases** have `expr: "<expression>"` and are computed on-demand.

### Compression Section

The `compression` section tracks compressed columns and their formulas:

| Field | Type | Description |
|-------|------|-------------|
| `compress` | str | Expression to compress (e.g., "round(asinh(dy)*40)") |
| `decompress` | str | Expression to decompress (e.g., "sinh(dy_c/40.)") |
| `compressed_dtype` | str | Storage dtype after compression |
| `decompressed_dtype` | str | Restored dtype after decompression |
| `compressed_col` | str | Name of compressed column (e.g., "dy_c") |
| `state` | str | Current state: "compressed", "decompressed", "schema_only" |

The `__meta__` key contains schema metadata:

```python
"__meta__": {
    "schema_version": 1,
    "state_machine": "CompressionState.v1"
}
```

### Subframes Section

The `subframes` section defines hierarchical relationships:

```python
"subframes": {
    "track": {"index": "track_index"},
    "collision": {"index": ["collision_index"]}  # Can be string or list
}
```

| Field | Type | Description |
|-------|------|-------------|
| `index` | str or list | Column(s) used for joining to parent frame |

---

## Schema Methods

### Core Methods

```python
# Get read-only copy of schema
schema = adf.schema        # Returns deep copy of _schema

# Export JSON-safe schema (converts dtypes to strings)
schema = adf.export_schema()

# Save schema to JSON file
adf.save_schema("output_schema.json")

# Load schema from JSON file (static method)
schema = AliasDataFrame.load_schema("output_schema.json")

# Apply schema to current ADF (dtypes, aliases, compression)
adf.apply_schema(schema, validate=True, warn_missing=True)

# Partial update of schema sections
adf.update_schema({
    "columns": {"pt": {"dtype": "float32"}},
    "compression": {"dy": {...}},
    "subframes": {"track": {"index": "track_index"}}
})
```

### ROOT Integration

```python
# Embed schema in ROOT file (as TObjString)
adf.save_schema_to_root("output.root", tree_name="tree")

# Load embedded schema from ROOT file
success = adf.load_schema_from_root("input.root")
```

### Parquet Integration

```python
# Save schema alongside Parquet file (creates .schema.json)
adf.save_schema_to_parquet_metadata("output.parquet")

# Load schema from Parquet metadata file
success = adf.load_schema_from_parquet_metadata("input.parquet")
```

---

## Workflows

### Workflow 1: Calibration Pipeline (ALICE Use Case)

This is the primary workflow for physics analysis:

```python
# Step 1: Load raw data from ROOT
adf = AliasDataFrame.read_tree("clusters.root", "tree")

# Step 2: Drop any previously materialized columns
adf.dematerialize()

# Step 3: Register subframes and auto-alias
adf_tracks = AliasDataFrame.read_tree("tracks.root", "tree")
adf.auto_alias_subframe("track", adf_tracks, index_columns="track_index")

# Step 4: Add calibration aliases
adf.add_alias("dX", "mX - track.mX")
adf.add_alias("dY", "mY - track.mY")

# Step 5: Apply compression
adf.compress_columns(compression_spec)

# Step 6: Export to ROOT with embedded schema
adf.export_tree("calibrated.root", "tree")
adf.save_schema_to_root("calibrated.root", "tree")

# Step 7: In C++ - use schema for TTree::Draw
# The embedded schema allows:
#   tree->Draw("dy")  // Uses TTree alias for decompression
```

### Workflow 2: Schema Round-Trip (Python → ROOT → Python)

```python
# === Export Phase ===
adf = AliasDataFrame(df)
adf.add_alias("pt", "sqrt(px**2 + py**2)")
adf.compress_columns(spec)

# Export tree with embedded schema
adf.export_tree("data.root", "tree")
adf.save_schema_to_root("data.root", "tree")

# === Import Phase ===
adf2 = AliasDataFrame.read_tree("data.root", "tree")

# Load embedded schema
if adf2.load_schema_from_root("data.root"):
    print("Schema restored from ROOT file")

# Now adf2 has same aliases, dtypes, compression info
print(adf2.aliases)           # {'pt': 'sqrt(px**2 + py**2)'}
print(adf2.compression_info)  # Compression formulas restored
```

### Workflow 3: Schema-Based Dtype Normalization

Ensures consistent dtypes across different file sources:

```python
# Define reference schema once
reference_schema = {
    "columns": {
        "x": {"dtype": "float32", "expr": None},
        "y": {"dtype": "float32", "expr": None},
        "z": {"dtype": "float32", "expr": None},
        "pt": {"dtype": "float32", "expr": "sqrt(px**2 + py**2)"}
    },
    "compression": {},
    "subframes": {}
}

# Apply to any loaded data
def load_normalized(filepath):
    adf = AliasDataFrame.read_tree(filepath, "tree")
    adf.apply_schema(reference_schema)
    return adf

# Now all files have consistent dtypes regardless of source
adf1 = load_normalized("run1.root")
adf2 = load_normalized("run2.root")
```

### Workflow 4: Parquet Round-Trip

Schema persistence with Parquet files:

```python
# === Export Phase ===
adf = AliasDataFrame(df)
adf.add_alias("pt", "sqrt(px**2 + py**2)", dtype=np.float32)
adf.compress_columns(compression_spec)

# Save data as Parquet
adf.save("output")  # Creates output.parquet

# Save schema alongside (creates output.schema.json)
adf.save_schema_to_parquet_metadata("output.parquet")

# === Import Phase ===
adf2 = AliasDataFrame.load("output")

# Load schema from metadata file
if adf2.load_schema_from_parquet_metadata("output.parquet"):
    print("Schema loaded from Parquet metadata")

# Aliases and compression info restored
print(adf2.aliases)           # {'pt': 'sqrt(px**2 + py**2)'}
print(adf2.compression_info)  # Compression formulas restored
```

**File structure:**
```
output.parquet       # DataFrame data
output.schema.json   # Schema metadata (created by save_schema_to_parquet_metadata)
```

### Workflow 5: Schema Export for C++ Analysis

```python
# Export schema as standalone JSON
adf.save_schema("analysis_schema.json")

# In C++ code, parse JSON and use for TTree access:
# - Get column dtypes
# - Get alias expressions for TTree::SetAlias
# - Get compression formulas for manual decompression
```

---

## Schema Versioning

### Version Field

The schema includes version information in `compression.__meta__`:

```python
"__meta__": {
    "schema_version": 1,
    "state_machine": "CompressionState.v1"
}
```

### Backward Compatibility

- `load_schema()` handles older schema versions automatically
- Missing fields get default values
- New fields are added without breaking old schemas
- Version 1 is current; future versions will include migration logic

### Migration Strategy

When loading older schemas:

```python
def _deserialize_schema(serialized):
    version = serialized.get("schema_version", 1)
    
    if version == 1:
        # Current version - no migration needed
        return serialized
    
    # Future: version 2+ migration
    if version < 2:
        # Add new fields with defaults
        serialized["new_field"] = default_value
        serialized["schema_version"] = 2
    
    return serialized
```

---

## Integration with Other Systems

### With Aliases

Schema is the source of truth for aliases:

```python
# Adding alias updates schema
adf.add_alias("pt", "sqrt(px**2 + py**2)", dtype=np.float32)

# Schema now contains:
# _schema["columns"]["pt"] = {
#     "expr": "sqrt(px**2 + py**2)",
#     "dtype": "float32"
# }

# Backward-compatible access
adf.aliases       # Returns {"pt": "sqrt(px**2 + py**2)"}
adf.alias_dtypes  # Returns {"pt": dtype('float32')}
```

### With Compression

Compression info lives in schema:

```python
adf.compress_columns(spec)

# Schema now contains compression details:
# _schema["compression"]["dy"] = {
#     "compress": "round(asinh(dy)*40)",
#     "decompress": "sinh(dy_c/40.)",
#     "state": "compressed",
#     ...
# }

# Access via property
adf.compression_info  # Returns _schema["compression"]
```

For complete compression documentation including state machine, precision measurement, and compression patterns, see **[COMPRESSION.md](COMPRESSION.md)**.

### With Subframes

Subframe registration updates schema:

```python
adf.register_subframe("track", adf_tracks, index_columns="track_index")

# Schema now contains:
# _schema["subframes"]["track"] = {"index": "track_index"}
```

---

## Examples

### Complete Example: TPC Residuals Analysis

```python
import numpy as np
from dfextensions.AliasDataFrame import AliasDataFrame

# === Define compression schema ===
compression_spec = {
    'dy': {
        'compress': 'round(asinh(dy)*40)',
        'decompress': 'sinh(dy_c/40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    },
    'dz': {
        'compress': 'round(asinh(dz)*40)',
        'decompress': 'sinh(dz_c/40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    }
}

# === Load and process ===
adf = AliasDataFrame.read_tree("residuals.root", "tree")

# Register subframe
adf_tracks = AliasDataFrame.read_tree("tracks.root", "tree")
adf.auto_alias_subframe("track", adf_tracks, index_columns="fClTrackIndex")

# Add calibration aliases
adf.add_alias("dY", "fClY - track.fY")
adf.add_alias("dZ", "fClZ - track.fZ")

# Compress
adf.compress_columns(compression_spec, measure_precision=True)

# === Export with schema ===
adf.export_tree("calibrated_residuals.root", "tree")
adf.save_schema_to_root("calibrated_residuals.root", "tree")

# Save schema separately for reference
adf.save_schema("calibrated_residuals_schema.json")

# === Verify round-trip ===
adf2 = AliasDataFrame.read_tree("calibrated_residuals.root", "tree")
adf2.load_schema_from_root("calibrated_residuals.root")

print("Aliases restored:", adf2.aliases)
print("Compression restored:", list(adf2.compression_info.keys()))
```

### Example Schema JSON

```json
{
  "columns": {
    "fClX": {"dtype": "float32", "expr": null},
    "fClY": {"dtype": "float32", "expr": null},
    "fClZ": {"dtype": "float32", "expr": null},
    "dy_c": {"dtype": "int16", "expr": null},
    "dz_c": {"dtype": "int16", "expr": null},
    "dy": {"dtype": "float16", "expr": "sinh(dy_c/40.)"},
    "dz": {"dtype": "float16", "expr": "sinh(dz_c/40.)"}
  },
  "compression": {
    "__meta__": {
      "schema_version": 1,
      "state_machine": "CompressionState.v1"
    },
    "dy": {
      "compress": "round(asinh(dy)*40)",
      "decompress": "sinh(dy_c/40.)",
      "compressed_dtype": "int16",
      "decompressed_dtype": "float16",
      "compressed_col": "dy_c",
      "state": "compressed"
    },
    "dz": {
      "compress": "round(asinh(dz)*40)",
      "decompress": "sinh(dz_c/40.)",
      "compressed_dtype": "int16",
      "decompressed_dtype": "float16",
      "compressed_col": "dz_c",
      "state": "compressed"
    }
  },
  "subframes": {
    "track": {"index": "fClTrackIndex"}
  }
}
```

---

## Troubleshooting

### Error: "Schema version mismatch"

```python
# Problem: Loading schema from older/newer version
# Solution: Schema loading handles version migration automatically
adf.apply_schema(schema)  # Will migrate older schemas

# If persistent issues:
print(schema.get("compression", {}).get("__meta__", {}).get("schema_version"))
```

### Error: "Missing index column for subframe"

```python
# Problem: Subframe index column not in DataFrame
# Check available columns:
print(adf.df.columns.tolist())

# Check required index:
print(adf._schema["subframes"])

# Solution: Ensure index column exists before registering subframe
if "track_index" in adf.df.columns:
    adf.register_subframe("track", adf_tracks, index_columns="track_index")
```

### Error: "Failed to apply dtype"

```python
# Problem: Schema dtype incompatible with data
# Solution: Use warn_missing=True to see warnings without failing
adf.apply_schema(schema, warn_missing=True)

# Or handle specific columns:
try:
    adf.df["col"] = adf.df["col"].astype(np.float32)
except (ValueError, TypeError) as e:
    print(f"Cannot convert column: {e}")
```

### Corrupted or Invalid JSON Schema

```python
# Problem: JSON file is malformed
# Solution: Validate JSON before loading
import json

try:
    with open("schema.json", "r") as f:
        schema = json.load(f)
    print("Schema is valid JSON")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")

# Re-export from valid AliasDataFrame:
adf.save_schema("schema_fixed.json")
```

### Schema Not Found in ROOT File

```python
# Problem: load_schema_from_root() returns False
# Solution: Check if schema was saved
import ROOT
f = ROOT.TFile.Open("file.root", "READ")
schema_obj = f.Get("ADF_SCHEMA")
if schema_obj:
    print("Schema exists:", schema_obj.GetString().Data()[:100])
else:
    print("No embedded schema - use save_schema_to_root() when exporting")
f.Close()
```

---

## Best Practices

### ✅ DO

1. **Use `apply_schema()` for restoration** - It handles dtypes, aliases, and compression together
2. **Save schema alongside data** - Always call `save_schema_to_root()` or `save_schema_to_parquet_metadata()` after export
3. **Version your schema files** - Include date or version in filename: `schema_v1.2.json`
4. **Validate after loading** - Check that aliases and compression info are restored correctly
5. **Use reference schemas** - Define canonical schemas for consistent processing across files
6. **Export schema for debugging** - Use `save_schema()` to inspect current state

### ❌ DON'T

1. **Don't manually edit embedded schema in ROOT files** - Use Python API to modify and re-export
2. **Don't mix schema versions carelessly** - Older schemas may lack fields expected by newer code
3. **Don't skip dtype specification** - Always include `dtype` in schema for reproducibility
4. **Don't rely on implicit schema** - Explicitly save and load schema for production workflows
5. **Don't ignore warnings from `apply_schema()`** - They indicate potential data issues
6. **Don't modify `_schema` directly** - Use `update_schema()` or specific methods

### Schema Hygiene

```python
# Good: Use explicit schema management
adf.add_alias("pt", "sqrt(px**2 + py**2)", dtype=np.float32)
adf.save_schema("analysis_schema.json")

# Bad: Modify internal structure directly
adf._schema["columns"]["pt"] = {"expr": "sqrt(px**2 + py**2)"}  # DON'T DO THIS
```

### Production Workflow Pattern

```python
# 1. Load with schema restoration
adf = AliasDataFrame.read_tree("data.root", "tree")
if not adf.load_schema_from_root("data.root"):
    raise ValueError("Missing schema - cannot process")

# 2. Process data
# ... analysis code ...

# 3. Export with schema preservation
adf.export_tree("output.root", "tree")
adf.save_schema_to_root("output.root", "tree")

# 4. Verify round-trip
adf_verify = AliasDataFrame.read_tree("output.root", "tree")
adf_verify.load_schema_from_root("output.root")
assert adf_verify.aliases == adf.aliases, "Schema round-trip failed"
```

---

## API Reference

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `adf.schema` | dict | Deep copy of `_schema` |
| `adf.aliases` | dict | `{name: expr}` for all aliases |
| `adf.alias_dtypes` | dict | `{name: dtype}` for aliases with dtype |
| `adf.compression_info` | dict | Reference to `_schema["compression"]` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `export_schema()` | dict | JSON-safe schema with string dtypes |
| `save_schema(path)` | None | Write schema to JSON file |
| `load_schema(path)` | dict | (Static) Read schema from JSON file |
| `apply_schema(schema)` | None | Apply dtypes, aliases, compression from schema |
| `update_schema(update)` | None | Partial update of schema sections |
| `save_schema_to_root(file, tree)` | None | Embed schema in ROOT file |
| `load_schema_from_root(file)` | bool | Load embedded schema from ROOT |
| `save_schema_to_parquet_metadata(file)` | None | Save schema alongside Parquet |
| `load_schema_from_parquet_metadata(file)` | bool | Load schema from Parquet metadata |

---

## See Also

- **USER_GUIDE.md** - Complete feature overview
- **COMPRESSION.md** - Compression system details
- **CHANGELOG.md** - Version history
## Composite Index for N > 2 Keys (C++ Only)

When subframes have more than 2 index columns, the C++ macro automatically
creates a composite key using cardinality-based packing.

### Schema Format

```json
{
  "subframes": {
    "Calib": {
      "index": ["row", "drift25", "side", "firstTFOrbit"]
    }
  }
}
```

### How It Works

The C++ `LoadADFTree()` function will:

1. **Map values to codes:** Each column's distinct values are mapped to 
   compact codes [0, 1, 2, ...]. This handles sparse indices correctly
   (e.g., orbit values like 547832001, 547832045, ...).

2. **Pack codes into single key:** Using cardinality as base:
   ```
   key = code[0] + code[1]*card[0] + code[2]*card[0]*card[1] + ...
   ```

3. **Create key branches:** A branch `__adf_key_<subframeName>__` is created
   in both the main tree and subframe.

4. **Build index on subframe:** Only the subframe tree gets `BuildIndex()`.
   ROOT's friend mechanism uses the subframe's index for lookups.

### Requirements

- All index columns must be **integer types** (Int_t, Long64_t, etc.)
- Float/Double columns will be rejected with an error message
- Maximum key space is checked to prevent overflow

### Multiple Subframes

Each subframe can have its own independent index definition:

```json
{
  "subframes": {
    "CalibA": {"index": ["row", "drift", "side"]},
    "CalibB": {"index": ["sector", "timeframe"]}
  }
}
```

Both subframes will work correctly and independently.

### Behavior with Edge Cases

| Case | Behavior |
|------|----------|
| No matching keys | Draw returns 0 entries for subframe columns |
| Empty subframe | No crash; main tree still accessible |
| Duplicate keys | ROOT uses first matching entry |
| Non-integer columns | Error message; graceful degradation (no index) |

### Example Usage

```cpp
// Load tree with 4-key composite index
TTree* tree = LoadADFTree("data.root", "tree");

// Access calibration data via standard ROOT syntax
tree->Draw("mX * Calib.gain + Calib.offset");

// Both subframes accessible independently
tree->Draw("CalibA.value");
tree->Draw("CalibB.correction");
```

### Performance Note

The composite index provides O(log n) lookup instead of O(n) linear scan,
which is critical for large calibration tables (>10k entries).
