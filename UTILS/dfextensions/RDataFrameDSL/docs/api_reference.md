# DSLCompiler API Reference

Complete API documentation for the `DSLCompiler` class.

## Table of Contents

1. [Constructor](#constructor)
2. [Core Methods](#core-methods)
3. [Visualization Methods](#visualization-methods) *(Phase 12)*
4. [Export Methods](#export-methods) *(Phase 12)*
5. [Arrow Integration](#arrow-integration) *(Phase 13)*
6. [Debugging Methods](#debugging-methods)

---

## Constructor

### `DSLCompiler(schema)`

Create a new DSL compiler with the given schema.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `Dict[str, str]` | Column name → C++ type mapping |

**Schema Format:**
```python
schema = {
    "column_name": "ctype",
    # Examples:
    "px": "double",
    "py": "float", 
    "n": "int",
    "flag": "bool",
    "pt": "RVec<double>",
    "tracks": "RVec<TLorentzVector>",
}
```

> **Note:** The canonical Team 2 schema format is a flat dictionary mapping column names to C++ type strings. Values may be scalar types (`double`, `int`, `bool`) or vector types (`RVec<T>`).

**Example:**
```python
from RDataFrameDSL import DSLCompiler

schema = {
    "px": "double",
    "py": "double",
    "tracks": "RVec<TLorentzVector>",
}
dsl = DSLCompiler(schema)
```

---

## Core Methods

### `define(name, expression)`

Define a new computed column.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name for the new column |
| `expression` | `str` | DSL expression (Python-like syntax) |

**Returns:** `DSLCompiler` (self, for chaining)

**Example:**
```python
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("track_pts", "tracks.Pt()")
dsl.define("high_pt", "tracks[tracks.Pt() > 1.0]")
```

**Chaining:**
```python
dsl.define("pt", "sqrt(px**2 + py**2)") \
   .define("eta", "-log(tan(theta/2))") \
   .define("is_central", "abs(eta) < 2.5")
```

---

### `apply(rdf)`

Apply all defined columns to an RDataFrame.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `rdf` | `ROOT.RDataFrame` | Input RDataFrame |

**Returns:** `ROOT.RDataFrame` with new columns defined

**Example:**
```python
import ROOT
rdf = ROOT.RDataFrame("Events", "data.root")
rdf = dsl.apply(rdf)

# Now use the new columns
hist = rdf.Histo1D("pt")
```

---

## Visualization Methods

*Added in Phase 12.1-12.5*

### `draw_figures(specs, rdf, **kwargs)`

Generate multiple plots from DSL definitions.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `specs` | `List[dict]` | required | Plot specifications |
| `rdf` | `ROOT.RDataFrame` | required | Data source |
| `show_statistics` | `bool` | `False` | Add μ, σ, n stats box |
| `show_expected` | `bool` | `False` | Add reference overlay for pulls |
| `expected_mean` | `float` | `0.0` | Expected mean for overlay |
| `expected_std` | `float` | `1.0` | Expected std for overlay |

**Plot Specification:**
```python
spec = {
    'expr': 'pt_residual',      # Column name or expression
    'bins': 100,                 # Number of bins
    'range': (-5, 5),           # X-axis range
    'title': 'pT Residual',     # Plot title
    'xlabel': 'Residual [GeV]', # X-axis label
    'type': 'hist',             # 'hist', 'scatter', 'profile'
    'is_pull': True,            # Override pull detection
}
```

**Returns:** `dict` with figure, axes, and statistics

**Example:**
```python
specs = [
    {'expr': 'dy_pull', 'bins': 50, 'range': (-5, 5)},
    {'expr': 'dz_pull', 'bins': 50, 'range': (-5, 5)},
    {'expr': 'pt_residual', 'bins': 100, 'is_pull': False},
]

results = dsl.draw_figures(
    specs, rdf,
    show_statistics=True,
    show_expected=True,
)
```

**Pull Detection:**
- Auto-detected if `'pull'` in expression name (case-insensitive)
- Override with `is_pull` in plot spec
- Pulls get N(0,1) Gaussian overlay when `show_expected=True`

---

## Export Methods

*Added in Phase 12.6*

### `to_aliasdf(include=None, exclude=None, dtype_map=None)`

Export DSL definitions to AliasDataFrame schema format.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include` | `List[str]` | `None` | Only export these columns |
| `exclude` | `List[str]` | `None` | Exclude these columns |
| `dtype_map` | `Dict[str, str]` | `None` | Override dtypes |

**Returns:** `dict` — AliasDataFrame-compatible schema

**Schema Format:**
```python
{
    'columns': {
        'pt_gev': {'expr': 'trackPt / 1000', 'dtype': 'float32'},
        'good_track': {'expr': '(trackPt > 0.5) & (nHits > 5)'}
    },
    '__meta__': {
        'source': 'RDataFrameDSL',
        'export_version': '1.0',
        'exported_at': '2025-12-16T...'
    }
}
```

> **Note:** This export format is specifically for AliasDataFrame interoperability. It differs from the canonical DSL schema format (flat dict) and includes Python-converted expressions.

**Operator Conversion:**

The following C++ operators are converted to Python equivalents:

| C++ | Python | Notes |
|-----|--------|-------|
| `&&` | `&` | Wrapped in parentheses for precedence |
| `\|\|` | `\|` | Wrapped in parentheses for precedence |
| `!` | `~` | Preserves `!=` |

**Example:**
```python
# Export all definitions
schema = dsl.to_aliasdf()

# Export with filters
schema = dsl.to_aliasdf(
    include=['pt_gev', 'eta'],
    dtype_map={'pt_gev': 'float32'}
)

# Apply to AliasDataFrame
from AliasDataFrame import AliasDataFrame
adf = AliasDataFrame(df)
adf.apply_schema(schema)
```

**Warnings:**
- TMath:: functions cannot be converted (warning issued)
- ROOT:: namespace expressions (warning issued)
- Method calls like `.Pt()` (warning issued)

> **Note:** Operator conversion is best-effort. Complex boolean expressions may require manual review after export.

---

### `get_definitions()`

Get all defined expressions as a simple dictionary.

**Returns:** `Dict[str, str]` — Column name → C++ expression

**Example:**
```python
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("good", "pt > 1.0 && abs(eta) < 2.5")

defs = dsl.get_definitions()
# {'pt': 'sqrt(px**2 + py**2)', 'good': 'pt > 1.0 && abs(eta) < 2.5'}
```

---

## Arrow Integration

*Added in Phase 13.2*

### `to_arrow(rdf=None, columns=None, flatten_rvec=False, include_schema=True)`

Export RDataFrame data to PyArrow Table.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rdf` | `ROOT.RDataFrame` | `None` | RDataFrame to export (uses internal if None) |
| `columns` | `List[str]` | `None` | Columns to export (all if None) |
| `flatten_rvec` | `bool` | `False` | Flatten RVec to 1D array |
| `include_schema` | `bool` | `True` | Embed DSL schema in Arrow metadata |

**Returns:** `pyarrow.Table`

**Recommended Usage:**
```python
# Option 1: Pass rdf explicitly (recommended)
table = dsl.to_arrow(rdf=rdf, columns=['pt', 'eta'])

# Option 2: Set internal reference first
dsl._rdf = rdf
table = dsl.to_arrow(columns=['pt', 'eta'])
```

> **Note:** The `rdf=None` default uses an internal reference that must be set separately (e.g., via `dsl._rdf = rdf`). For clarity, passing `rdf` explicitly is recommended.

**RVec Handling:**
| `flatten_rvec` | RVec Behavior |
|----------------|---------------|
| `False` | Preserve as Arrow ListArray (jagged) |
| `True` | Flatten to 1D array (loses event structure) |

**Example:**
```python
import pyarrow.parquet as pq

# Export to Arrow
table = dsl.to_arrow(rdf=rdf, columns=['pt', 'eta', 'tracks'])

# Save as Parquet
pq.write_table(table, 'output.parquet')

# Export with flattening
flat_table = dsl.to_arrow(rdf=rdf, flatten_rvec=True)
```

**Schema Metadata:**
When `include_schema=True`, the Arrow table metadata contains:
- `dsl_schema`: JSON-encoded schema from `to_aliasdf()`
- `rvec_columns`: List of columns that were RVec

**Requirements:**
- `pyarrow>=12.0` (optional dependency)
- Raises `ImportError` if PyArrow not installed

**Notes:**
- Phase 1 implementation uses numpy as intermediate layer (copy-based)
- Large RVec columns (>1M events) trigger a memory warning

---

### `DSLCompiler.from_arrow(table, apply_schema=True)` *(classmethod)*

Create a DSLCompiler from a PyArrow Table.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `pyarrow.Table` | required | Input Arrow table |
| `apply_schema` | `bool` | `True` | Apply schema from metadata if present |

**Returns:** `DSLCompiler` — New instance with RDataFrame from table data

**Type Inference:**
| Arrow Type | C++ Type |
|------------|----------|
| `float64` | `double` |
| `float32` | `float` |
| `int32` | `int` |
| `int64` | `long` |
| `bool` | `bool` |
| `list<T>` | `RVec<T>` |

> **Platform Note:** C++ integer widths are platform-dependent. This mapping assumes typical Linux/ROOT environments where `long` is 64-bit.

**Example:**
```python
import pyarrow.parquet as pq

# Load from Parquet
table = pq.read_table('data.parquet')

# Create DSL from Arrow
dsl = DSLCompiler.from_arrow(table, apply_schema=True)

# Definitions from metadata are available
dsl.define("new_var", "pt * 2")
```

**Round-Trip Example:**
```python
# Export
table = dsl.to_arrow(rdf=rdf, include_schema=True)

# Import
new_dsl = DSLCompiler.from_arrow(table)

# Schema preserved (for scalar columns)
assert 'pt_gev' in new_dsl.get_definitions()
```

**Limitations:**
- Expression conversion is best-effort (warning issued for complex expressions)
- Round-trip for `list<T>` (RVec) columns may require `flatten_rvec=True` on export for full compatibility
- Complex expressions with TMath/ROOT functions may need manual review after import

**Notes:**
- Creates a copy of the data (Phase 1 implementation)
- Uses numpy as intermediate for Arrow → RDataFrame conversion

---

## Debugging Methods

### `preview()`

Preview generated C++ code without compiling.

**Returns:** `str` — Human-readable preview of all functions

**Example:**
```python
dsl.define("pt", "sqrt(px**2 + py**2)")
print(dsl.preview())
```

**Output:**
```
=== Function: pt ===
DSL Expression: sqrt(px**2 + py**2)

double alias_pt_abc123(double px, double py) {
    return std::sqrt((std::pow(px, 2) + std::pow(py, 2)));
}
```

---

### `export_macro(filepath, include_test=False)`

Export all definitions to a C++ macro file.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | required | Output file path (`.C`) |
| `include_test` | `bool` | `False` | Include test code |

**Example:**
```python
dsl.export_macro("my_analysis.C", include_test=True)
```

**Generated File:**
```cpp
// Generated by RDataFrameDSL
// DSL: sqrt(px**2 + py**2)

#include <cmath>

double alias_pt(double px, double py) {
    return std::sqrt((std::pow(px, 2) + std::pow(py, 2)));
}

// ... more functions ...
```

---

## Complete Example

```python
from RDataFrameDSL import DSLCompiler
import ROOT

# Define schema
schema = {
    "px": "double",
    "py": "double",
    "pz": "double",
    "tracks": "RVec<TLorentzVector>",
}

# Create compiler
dsl = DSLCompiler(schema)

# Define computed columns
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("eta", "-log(tan(atan2(pt, pz)/2))")
dsl.define("track_pts", "tracks.Pt()")
dsl.define("n_tracks", "tracks.size()")
dsl.define("high_pt_tracks", "tracks[tracks.Pt() > 1.0]")

# Apply to RDataFrame
rdf = ROOT.RDataFrame("Events", "data.root")
rdf = dsl.apply(rdf)

# Make plots with statistics
results = dsl.draw_figures(
    [{'expr': 'pt', 'bins': 100}],
    rdf,
    show_statistics=True
)

# Export to Arrow
table = dsl.to_arrow(rdf=rdf)

# Export schema for AliasDataFrame
schema = dsl.to_aliasdf()
```

---

## Error Handling

All DSL errors raise `IRError` with helpful messages:

```python
try:
    dsl.define("bad", "trakcs.Pt()")  # Typo
except IRError as e:
    print(e.message)
    # Unknown variable 'trakcs'
    print(e.suggestions)
    # ['Did you mean: tracks?']
```

---

## Requirements

| Feature | Required Package | Version |
|---------|------------------|---------|
| Core | ROOT | 6.26+ |
| Arrow export | pyarrow | 12.0+ |
| Visualization | matplotlib | any |
| AliasDataFrame | pandas | any |

---

## See Also

- [Quick Start Guide](quickstart.md) — Get started in 5 minutes
- [Expression Reference](expressions.md) — Complete DSL syntax
- [Architecture](ARCHITECTURE.md) — Internal design
- [Phase History](PHASE_HISTORY.md) — Development history
