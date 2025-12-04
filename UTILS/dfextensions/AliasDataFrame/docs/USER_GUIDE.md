# AliasDataFrame – Hierarchical Lazy Evaluation for Pandas + ROOT

`AliasDataFrame` is an extension of `pandas.DataFrame` that enables **named expression-based columns (aliases)** with:

* ✅ **Lazy evaluation** (on-demand computation)
* ✅ **Automatic dependency resolution** (topological sort, cycle detection)
* ✅ **Hierarchical aliasing** across **linked subframes** (e.g. clusters referencing tracks via index joins)
* ✅ **Persistence** to Parquet and ROOT TTree formats, including full alias metadata

It is designed for physics and data analysis workflows where derived quantities, calibration constants, and multi-table joins should remain symbolic until final export.

---

## ✨ Core Features

### ✅ Alias Definition & Lazy Evaluation

Define symbolic columns as expressions involving other columns or aliases:

```python
adf.add_alias("pt", "sqrt(px**2 + py**2)")
adf.materialize_alias("pt")  # Adds 'pt' column to DataFrame
```

**Evaluate aliases without storing:**

```python
# Get alias as pandas Series (doesn't add column)
pt_series = adf.get_alias_series("pt")

# Get alias as numpy array (doesn't add column) 
pt_array = adf.get_alias_array("pt")
```

This is particularly useful for:
- **Boolean selections** without polluting DataFrame with temporary mask columns
- **Temporary computations** for analysis or visualization
- **Progressive workflows** where you want clean DataFrames

**Example: Selection masks**

```python
# Define selection as boolean alias
adf.add_alias("highPt", "pt > 5.0", dtype=bool)

# Get mask without adding column to DataFrame
mask = adf.get_alias_array("highPt")
selected_data = adf.df[mask]

# DataFrame stays clean: 'highPt' is NOT a column
assert 'highPt' not in adf.df.columns  # ✓ True
```

**Return types:**
- `get_alias_series(name)` always returns a pandas Series aligned with the DataFrame index.
- `get_alias_array(name)` always returns a NumPy array.

**Important:** Only the *dependencies* of the alias may be materialized as columns; the alias itself is never added as a column unless `materialize_alias()` is explicitly called. This enables clean boolean selections ideal for SoA-style (Structure of Arrays) selection specifications.

### ✅ Subframe Support (Hierarchical Dependencies)

Reference a subframe (e.g. per-cluster frame linked to a per-track frame):

```python
adf_clusters.register_subframe("track", adf_tracks, index_columns="track_index")
adf_tracks.register_subframe("collision", adf_collisions, index_columns="collision_index")

adf_clusters.add_alias("dX", "mX - track.mX")
adf_clusters.add_alias("vertexZ", "track.collision.z")
```

Under the hood, this performs joins using index columns such as `track_index` and `collision_index`, rewrites dotted expressions like `track.mX` and `track.collision.z` to joined columns, and evaluates in that context.

For example, in ALICE data:

- clusters reference tracks: `cluster → track`
- tracks reference collisions: `track → collision`
- V0s reference two tracks: `v0 → track1`, `v0 → track2`

These relations can be declared using `register_subframe()` and used symbolically in aliases.

### ✅ Explicit Subframe API (Phase B)

Phase B adds explicit control over subframe alias creation, tracking, and removal. This enables memory optimization workflows where materialized subframe columns can be replaced with lazy aliases.

#### Method Summary

| Category | Method | Description |
|----------|--------|-------------|
| **Core** | `subframe(name)` | Access subframe by name |
| | `list_subframes()` | List available subframes |
| **Auto-Alias** | `auto_alias_subframe(name)` | Create aliases for subframe columns |
| | `auto_alias_all_subframes()` | Auto-alias all subframes |
| | `get_auto_alias_candidates()` | Find optimization opportunities |
| **Removal** | `remove_alias(name)` | Remove single alias |
| | `remove_aliases(names)` | Remove multiple aliases |
| **Tracking** | `is_auto_alias(name)` | Check if auto-created |
| | `get_auto_aliases()` | Get all auto-aliases |
| | `list_auto_aliases()` | List auto-alias names |
| | `remove_auto_aliases()` | Remove auto-aliases |
| **Schema I/O** | `save_schema_to_root()` | Embed schema in ROOT file |
| | `load_schema_from_root()` | Load schema from ROOT file |
| | `save_schema_to_parquet_metadata()` | Save schema alongside Parquet |
| | `load_schema_from_parquet_metadata()` | Load schema from Parquet metadata |

#### When to Use Auto-Alias vs Manual Alias

| Use Case | Approach |
|----------|----------|
| Access all subframe columns uniformly | `auto_alias_subframe()` |
| Define derived quantities (e.g., `dX = mX - track.mX`) | `add_alias()` (manual) |
| Memory optimization (drop materialized columns) | `auto_alias_subframe()` + `get_auto_alias_candidates()` |
| Temporary calculations | `add_alias()` then `remove_alias()` |
| Calibration corrections | `add_alias()` with explicit expressions |

**Rule of thumb:** Use `auto_alias_subframe()` when you want direct access to subframe columns. Use `add_alias()` when you need computed expressions.

#### Examples

**Auto-alias a subframe:**
```python
# Create aliases: column_name → subframe.column_name
aliases = adf.auto_alias_subframe('DITS0FitSide', validate=True)
# Output: {'dyC1_intercept': 'DITS0FitSide.dyC1_intercept', ...}
```

**Remove aliases safely:**
```python
# Strict mode (raises KeyError if not found)
adf.remove_alias('temp_alias')

# Lenient mode (no error if missing)
adf.remove_alias('maybe_exists', strict=False)

# Batch removal
adf.remove_aliases(['alias1', 'alias2', 'alias3'], strict=False)
```

**Track auto-created aliases:**
```python
# Check if auto-created
if adf.is_auto_alias('dyC1_intercept'):
    print("This was auto-created from a subframe")

# Get all auto-aliases for a specific subframe
dits_aliases = adf.get_auto_aliases('DITS0FitSide')

# Remove only calibration auto-aliases (keep others)
adf.remove_auto_aliases('DITS0FitSide')
```

**Embed schema in ROOT file:**
```python
# Save schema (for later reconstruction)
adf.save_schema_to_root('output.root', 'tree')

# Load schema (into fresh AliasDataFrame)
adf2 = AliasDataFrame(df)
if adf2.load_schema_from_root('output.root'):
    print("Schema loaded - all aliases restored")
```

**Multiple subframes with collision handling:**
```python
# Auto-alias both subframes
adf.auto_alias_subframe('calib1')
adf.auto_alias_subframe('calib2')  # If same column name, calib2 wins

# Remove only calib1 aliases (calib2 remains)
adf.remove_auto_aliases('calib1')
```

### 📊 Memory Optimization Workflow

A complete workflow for reducing DataFrame memory using subframe aliases:

#### Workflow Checklist

1. ☐ Load data with `read_tree()`
2. ☐ Create auto-aliases with `auto_alias_subframe(validate=True)`
3. ☐ Find candidates with `get_auto_alias_candidates()`
4. ☐ Drop materialized columns from DataFrame
5. ☐ Save schema with `save_schema_to_root()`
6. ☐ Export with `export_tree()`

#### Complete Example

```python
# 1. Load data with subframes
adf = AliasDataFrame.read_tree('tpc_data.root', 'tree', num_workers=12)
print(f"Initial memory: {adf.df.memory_usage().sum()/1e9:.2f} GB")

# 2. Create auto-aliases for calibration subframe
aliases = adf.auto_alias_subframe('DITS0FitSide', validate=True)

# 3. Find materialized columns that can be dropped
candidates = adf.get_auto_alias_candidates()
print(f"Can drop: {candidates}")

# 4. Drop materialized columns (now accessed via aliases)
for cols in candidates.values():
    adf.df.drop(columns=cols, inplace=True)

print(f"Final memory: {adf.df.memory_usage().sum()/1e9:.2f} GB")
# Typical result: 8 GB → 1.5 GB (5x reduction)

# 5. Save with embedded schema
adf.save_schema_to_root('optimized.root', 'tree')

# 6. Export final tree
adf.export_tree('optimized.root', 'tree')
```

#### Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | 8 GB | 1.5 GB | 5x reduction |
| Columns | 70+ | 20 | Lazy evaluation |
| Schema | Lost | Embedded | Reproducible |

### 🖥️ C++ Compatibility (ROOT TTree::Draw)

AliasDataFrame ensures full compatibility with the low-level **ROOT TTree::Draw** query environment, bridging the flexibility of Python with the efficiency of C++.

#### Key Features

* **Schema Embedding:** The full symbolic schema (aliases, subframes, dependencies) is automatically embedded in the TTree's UserInfo list as `TNamed("ADF_SCHEMA")`, allowing C++ macros to access and interpret the Python schema definitions.

* **Alias Translation:** All standard Python/NumPy expressions are translated into valid ROOT TTree expressions (`atan2` → `atan2`, `np.sqrt` → `sqrt`, etc.) via `TTree::SetAlias`.

* **Multi-Key Joins (ROOT Constraint):** ROOT's native `TTree::BuildIndex` is limited to two index columns. To enable 3D and 4D joins on the C++ side, the Python exporter creates a **1D composite key** based on index hashing. The C++ loading macro leverages this composite key to perform efficient indexed joins in the ROOT environment.

> **Note:** Multi-key joins in C++ currently rely on 2-key `BuildIndex`. Generic N-key support (`BuildCompositeIndex`) is planned for the next phase.

#### C++ Usage

```cpp
// Load the helper macro
.L AliasDataFrameTree.C

// Load TTree with embedded schema
TFile* f = TFile::Open("output.root");
TTree* tree = (TTree*)f->Get("tree");

// Aliases defined in Python are available in C++
tree->Draw("dy");           // Uses decompression alias if compressed
tree->Draw("dX:dY");        // Subframe aliases work transparently
tree->Draw("pt", "highPt"); // Boolean selection aliases

// Multi-key joins (planned - requires BuildCompositeIndex)
// tree->Draw("calib.gain_factor");  // 3D join: (drift, side, sector)
```

#### Required C++ Macro

To enable alias translation, use the `AliasDataFrameTree.C` macro:

```cpp
// AliasDataFrameTree.C provides:
// - LoadAliasDataFrame(): Load tree with schema reconstruction
// - ApplyAliases(): Set TTree aliases from embedded schema
// 
// Planned (next phase):
// - BuildCompositeIndex(): Create multi-key index from hash
```

#### Workflow: Python → C++

```
┌─────────────────────────────────────────────────────────────┐
│  Python                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ AliasDataFrame  │───▶│ export_tree()   │                │
│  │ + aliases       │    │ + schema embed  │                │
│  │ + subframes     │    │ + alias export  │                │
│  └─────────────────┘    └────────┬────────┘                │
└──────────────────────────────────┼──────────────────────────┘
                                   │
                              output.root
                                   │
┌──────────────────────────────────┼──────────────────────────┐
│  C++ (ROOT)                      ▼                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ AliasDataFrame  │◀───│ TFile::Open()   │                │
│  │ Tree.C          │    │ + TNamed schema │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ TTree::Draw()   │  ← Standard aliases work now          │
│  │ with aliases    │  ← Multi-key joins: planned           │
│  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

This enables seamless analysis workflows where data preparation is done in Python and final visualization/fitting is done in C++ ROOT.

### 🚀 RDataFrame Integration (AliasDataFrameRDF)

Generate RDataFrame C++ code from AliasDataFrame expressions for high-performance analysis.

#### Quick Start

```python
from AliasDataFrameRDF import generate_rdf_code_with_friends

# Generate RDF code from AliasDataFrame
code = generate_rdf_code_with_friends(
    adf,
    target_aliases=["L10", "correctedY"],
    enable_mt=True,
    num_threads=8
)
print(code)  # C++ code ready to compile
```

#### Key Functions

| Function | Purpose |
|----------|---------|
| `to_cpp_expr(expr)` | Convert Python expression to C++ |
| `extract_dependencies(expr)` | Find column dependencies |
| `get_ordered_defines(adf, aliases)` | Topologically sorted Define() chain |
| `generate_rdf_code_with_friends(adf, ...)` | Full RDF code with friend trees |

#### Expression Translation

| Python | C++ (ROOT) |
|--------|------------|
| `x**2` | `pow(x, 2)` |
| `np.sqrt(x)` | `sqrt(x)` |
| `np.abs(x)` | `abs(x)` |
| `True/False` | `true/false` |
| `track.mP3` | `T.mP3` (friend tree) |

#### Sparse Key Support

For multi-key joins with sparse distributions:

```python
from AliasDataFrameRDF import compute_composite_key_auto

main_keys, sub_keys, method = compute_composite_key_auto(
    main_df, sub_df, ['orbit', 'row', 'drift']
)
# method = 'dense' or 'sparse' (auto-selected)
```

#### Performance

Benchmark results (1M rows, 10-level alias chain):

| Method | Time | Speedup |
|--------|------|---------|
| AliasDataFrame | 0.04s | 25x |
| TTree::Draw | 0.96s | 1.0x |
| RDataFrame | 1.36s | 0.7x |

> **Note:** RDataFrame slower at this scale due to JIT overhead. RDF advantage shows at larger scales (>10M rows) with complex DAGs.

### ✅ Dependency Graph & Cycle Detection

* Automatically resolves dependency order
* Detects and raises on circular alias definitions
* Visualize with:

```python
adf.plot_alias_dependencies()
```

### ✅ Constant Aliases & Dtype Enforcement

```python
adf.add_alias("scale", "1.5", dtype=np.float32, is_constant=True)
```

### ✅ Attribute Access for Aliases and Subframes

Access aliases and subframe members with convenient dot notation:

```python
adf.cutHighPt  # equivalent to adf["cutHighPt"]
adf.track.pt   # evaluates pt from registered subframe "track"
```

---

## 💾 Persistence

### ➤ Save to Parquet

```python
adf.save("data/my_frame")  # Saves data + alias metadata
```

### ➤ Load from Parquet

```python
adf2 = AliasDataFrame.load("data/my_frame")
```

### ➤ Export to ROOT TTree (with aliases!)

```python
adf.export_tree("output.root", treename="MyTree")
```

### ➤ Import from ROOT TTree

```python
adf = AliasDataFrame.read_tree("output.root", treename="MyTree")
```

Subframe alias metadata (including join indices) is preserved recursively.

---

## 🧪 Unit-Tested Features

Tests included for:

* Basic alias chaining and materialization
* Dtype conversion
* Constant and hierarchical aliasing
* Partial materialization
* Subframe joins on index columns
* Chained access via `adf.attr` and `adf.subframe.alias`
* Persistence round-trips for `.parquet` and `.root`
* Error detection: cycles, invalid expressions, undefined symbols
* **Phase B:** Explicit subframe API (37 tests in `test_subframe_alias_api.py`)

---

## 🧠 Internals

* Expression evaluation via `eval()` with math/Numpy-safe scope
* Dependency analysis via `networkx`
* Subframes stored in a registry (`SubframeRegistry`) with index-aware entries
* Subframe alias resolution is performed via on-the-fly joins using provided index columns
* **Auto-alias tracking** via `_auto_aliases` dict: `{alias_name: subframe_name}`
* Metadata embedded into:

  * `.parquet` via Arrow schema metadata
  * `.root` via `TTree::SetAlias` and `TNamed("ADF_SCHEMA")`

---

## 📋 Schema System

The **schema** is the single source of truth for all AliasDataFrame metadata. It unifies column definitions, aliases, compression settings, and subframe relationships into one structure.

### What the Schema Contains

| Section | Contents |
|---------|----------|
| `columns` | Physical column dtypes and aliases (expressions + dtypes) |
| `compression` | Compression formulas, dtypes, and state per column |
| `subframes` | Registered subframes and their index columns |

### Why Schema Matters

- **Reproducibility**: Schema defines exact dtypes and aliases for consistent results
- **Persistence**: Schema round-trips through JSON, ROOT, and Parquet formats
- **C++ Compatibility**: Embedded schema enables TTree::Draw with aliases
- **Memory Optimization**: Dtype restoration prevents float16→float32 bloat

### Basic Usage

```python
# Export schema to JSON
schema = adf.export_schema()
adf.save_schema("my_schema.json")

# Load and apply schema
schema = AliasDataFrame.load_schema("my_schema.json")
adf.apply_schema(schema)

# Embed schema in ROOT file
adf.save_schema_to_root("output.root", "tree")

# Load schema from ROOT file
adf.load_schema_from_root("input.root")
```

For complete schema documentation including structure details, versioning, and workflows, see **[SCHEMA.md](SCHEMA.md)**.

---

## 🔍 Introspection & Debugging

```python
adf.describe_aliases()       # Print aliases, dependencies, broken ones
adf.validate_aliases()       # List broken/inconsistent aliases
adf.list_subframes()         # List available subframes
adf.get_auto_aliases()       # Get auto-created alias tracking
```

---

## 🧩 Requirements

* `pandas`, `numpy`, `pyarrow`, `uproot`, `networkx`, `matplotlib`, `ROOT`

---

## 🔁 Comparison with Other Tools

| Feature                       | AliasDataFrame | pandas    | Vaex     | Awkward Arrays | polars    | Dask      |
| ----------------------------- | -------------- | --------- | -------- | -------------- | --------- | --------- |
| Lazy alias columns            | ✅ Yes          | ⚠️ Manual | ✅ Yes    | ❌              | ✅ Partial | ✅ Partial |
| Non-materializing evaluation  | ✅ Yes          | ⚠️ eval() | ⚠️ Partial | ❌            | ⚠️ Partial | ⚠️ Partial |
| Dependency tracking           | ✅ Full graph   | ❌         | ⚠️ Basic | ❌              | ❌         | ❌         |
| Subframe hierarchy (joins)    | ✅ Index-based  | ❌         | ❌        | ⚠️ Nested only | ❌         | ⚠️ Manual |
| Explicit subframe aliasing    | ✅ Auto-alias   | ❌         | ❌        | ❌              | ❌         | ❌         |
| Constant alias support        | ✅ With dtype   | ❌         | ❌        | ❌              | ❌         | ❌         |
| Visualization of dependencies | ✅ `networkx`   | ❌         | ❌        | ❌              | ❌         | ❌         |
| Export to ROOT TTree          | ✅ Optional     | ❌         | ❌        | ✅ via uproot   | ❌         | ❌         |
| Schema embedding (ROOT/Parquet) | ✅ TNamed     | ❌         | ❌        | ❌              | ❌         | ❌         |

---

## ❓ Why AliasDataFrame?

In many data workflows, users recreate the same patterns again and again:

* Manually compute derived columns with ad hoc logic
* Scatter constants and correction factors in multiple files
* Perform fragile joins between tables (e.g. clusters ↔ tracks) with little traceability
* Lose transparency into what each column actually means

**AliasDataFrame** turns these practices into a formalized, symbolic layer over your DataFrames:

* 📐 Define all derived quantities as symbolic expressions
* 🔗 Keep relations between DataFrames declarative, index-based, and reusable
* 📊 Visualize dependency structures and broken logic automatically
* 📦 Export the full state of your workflow (including symbolic metadata)

It brings the clarity of a computation graph to structured table analysis — a common but under-supported need in `pandas`, `vaex`, or `polars` workflows.

---

## 🛣 Roadmap Ideas

* [x] Explicit subframe API (Phase B - vmi-1-3-0)
* [x] Schema embedding in ROOT/Parquet
* [ ] **C++ BuildCompositeIndex** for N-key joins (high priority)
* [ ] Secure expression parser (no raw `eval`)
* [ ] Aliased column caching / invalidation strategy
* [ ] Inter-subframe join strategies (e.g., key-based, 1: n)
* [ ] Jupyter widget or CLI tool for alias graph exploration
* [ ] Broadcasting-aware joins or 2D index support

---

## 🧑‍🔬 Designed for...

* Physics workflows (e.g. ALICE Physics analysis V0 ↔ tracks ↔ collisions)
* Symbolic calibration / correction workflows
* Structured data exports with traceable metadata

---

**Author:** Marian Ivanov

MIT License
