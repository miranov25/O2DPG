# AliasDataFrame Technical Summary

**Document ID:** `AliasDataFrame_Technical_Summary_v13_27_ADF_v1_6.md`  
**Author:** Claude37 (AliasDataFrame Reviewer / Main Reviewer)  
**Date:** 2026-05-14  
**Version:** 1.6  
**Phase:** 13.27.ADF (`read_tree` skip_branches) — base for active queue  
**Audience:** All teams — architecture reviewers, cross-team coders (ORecoAI, RootInteractive, RDataFrameDSL, GBAI), and direct users  
**Purpose:** Complete public API reference, data model, hierarchical data representation, dependencies, limitations

**Revision notes (v1.6):** Phases 13.22 through 13.27, plus bug fixes and FIX1 queue:

- Updated header: version 1.6, date 2026-05-14, phase target 13.27.ADF
- Updated §1.4: test count **1606** (was 1521), invariance count **177** (was 125), Capability Matrix **47 features / 28 ✅ / 14 ☑️ / 4 🧨 / 1 📋**
- Updated §4 (`read_tree`): added `dtype_overrides={regex: np.dtype}` and `skip_branches=[regex]` parameters with usage examples
- Updated §13.1 Bug Fixes: 4 new resolutions (BUG_GroupBy_Expression_Materialization, BUG_validate_aliases_false_positives, BUG_save_load_compression_regression, plus prior pending)
- Updated §13.2 Resolved Items: G1-G4, B1, D1-D14 tests shipped; baseline preserved
- Updated §14 Planned Work: Phase 13.25.DF FIX1 (active priority), Phase 14 ADFStore (PyArrow-backed storage, motivated by production fragmentation evidence)
- NEW §14.3: Methodology/governance lessons from BUG_GroupBy cycle (anti-fabrication firewall validation, multi-model panel diversity, mandatory baseline differential)
- All v1.5.2 content retained

**Revision notes (v1.5.2):** Phase 13.21.ADF additions:
- NEW in §4.3: `dematerialize(drop=, keep=)` — memory reclamation with raw-column protection; replaces removed `drop_materialized()`
- Updated §2 Join Contract: join index cache row (content-based validation, survives `materialize_aliases`, invalidated on `register_subframe`)
- Updated §12.1: 1521 tests, 125 invariance

---

## Table of Contents

1. [Overview](#1-overview)
2. [Contract Snapshot (Cross-Team Quick Reference)](#2-contract-snapshot-cross-team-quick-reference)
3. [Module Architecture](#3-module-architecture)
4. [Core Public API](#4-core-public-api)
5. [Data Model & Hierarchical Representation](#5-data-model--hierarchical-representation)
6. [Alias & Lazy Evaluation System](#6-alias--lazy-evaluation-system)
7. [Subframe Registration & Join Semantics](#7-subframe-registration--join-semantics)
8. [Draw Integration with dfdraw](#8-draw-integration-with-dfdraw)
9. [Backend Strategy](#9-backend-strategy)
10. [Compression System](#10-compression-system)
11. [Cross-Subproject Dependencies](#11-cross-subproject-dependencies)
12. [Test Coverage](#12-test-coverage)
13. [Known Limitations](#13-known-limitations)
14. [Planned Work](#14-planned-work)
15. [Quick Reference](#15-quick-reference)

---

## 1. Overview

### 1.1 What is AliasDataFrame?

AliasDataFrame is a high-performance Python data analysis framework for ALICE particle physics at CERN. It provides:

- **Schema-driven aliases** — derived columns via expression strings (e.g., `'pt * cos(phi)'`)
- **Hierarchical joins** — parent/child table relationships via `register_subframe()`
- **Lazy evaluation** — on-demand branch loading from ROOT files (90%+ memory savings)
- **Multi-file chains** — seamless handling of split datasets across ROOT files
- **Compression** — user-defined quantization with idempotent compress/decompress
- **Draw integration** — declarative plotting via dfdraw with auto-materialization
- **Registered functions** — custom callables, polynomials, and evaluators as aliases (Phase 13.9/13.10)

### 1.2 Architecture Overview

```
ROOT / Parquet Files
        ↓
┌──────────────────────┐
│  LazyTreeReader      │  ← On-demand branch loading
│  LazyChainReader     │  ← Multi-file composition
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  AliasDataFrame      │  ← Core: schema, aliases, subframes, compression
│  (~12,100 lines)     │
│  PolynomialSpec      │  ← N-dimensional polynomial specification (Phase 13.9)
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Backend Dispatch     │  ← Numba (primary) → NumPy → Pandas fallback
│  _numba_accelerators  │
│  _arrow_compute       │  ← Arrow for scatter only (compute disabled)
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  dfdraw (DFDraw)     │  ← Visualization via draw()
│  AliasDataFrameRDF   │  ← RDataFrame export
│  GroupByRegression   │  ← Evaluator integration (Phase 13.10)
└──────────────────────┘
```

### 1.3 File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `AliasDataFrame.py` | ~12,100 | Core module: schema, aliases, subframes, compression, draw, function registry |
| `PolynomialSpec.py` | 337 | N-dimensional polynomial specification (Phase 13.9) |
| `AliasDataFrameRDF.py` | ~800 | RDataFrame integration (setup_rdf, add_defines, export_tree) |
| `AliasDataFrameTree.C` | ~200 | ROOT C++ macro for TTree operations |
| `LazyTreeReader.py` | ~500 | On-demand ROOT branch loading |
| `LazyChainReader.py` | ~600 | Multi-file chain with LRU cache |
| `_numba_accelerators.py` | ~300 | JIT-compiled scatter/join kernels |
| `_arrow_compute.py` | ~400 | PyArrow expression compiler (scatter only) |
| `_composite_keys.py` | ~200 | Composite key linearization |
| `exceptions.py` | ~100 | Custom exception hierarchy |
| `run_tests.sh` | ~320 | Automated test runner with reviewer.zip (Phase 13.11) |
| `scripts/generate_capability_matrix.py` | ~200 | Taxonomy-based matrix generator v2 (Phase 13.11.B) |
| `scripts/phase_tag.sh` | ~54 | PHASE_BEGIN tag management (Phase 13.11) |
| `tests/feature_taxonomy.py` | ~200 | 41-feature taxonomy definition (Phase 13.11.B) |

**Total source:** ~16,300 lines  
**Test files:** 48+ (Python + C++)

### 1.4 Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Test suite | 1441 passed, 6 failed, 7 skipped | `pytest tests/` (2026-04-02) |
| Runtime | ~27s (12 workers, ROOT enabled) | Same run |
| Capability Matrix | 41 features (9 verified, 27 smoke, 4 broken, 1 planned) | Phase 13.11.B |
| Invariance tests | 64 | `@pytest.mark.invariance` markers |
| Read speedup | 60–770× (threaded branch-by-branch vs original) | Phase 1 benchmark, commit `60d0e5d` |
| Join speedup | 10.8× vs Phase 3 baseline | Phase 8c benchmark, commit `46d2320` |
| Polynomial eval speedup | 42× (Numba JIT vs pandas.eval) | Phase 13.9 benchmark |
| Peak memory | 74–79% reduction (3894 MB → 853 MB for 12M rows) | Phase 1 `benchmark_read_tree.py` |
| Lazy memory savings | 90%+ (5–10 of 100+ branches loaded) | Phase 7.1 measurement |
| Efficiency vs hardware | 6.8% (NumPy), 1.8% (Numba), 1.4% (memcpy) | Phase 3 roofline analysis |

---

## 2. Contract Snapshot (Cross-Team Quick Reference)

> **For Team 2 (RDataFrameDSL), GBAI team, and cross-team integration — the critical API surface:**

### Constructor

```python
# ACTUAL SIGNATURE (source: AliasDataFrame.py line 773):
AliasDataFrame(df, schema_id=None, use_numba=None, use_arrow=None)

# ❌ NOT SUPPORTED (will raise TypeError):
AliasDataFrame(df, schema=...)     # No schema= parameter
AliasDataFrame(df, metadata=...)   # No metadata= parameter
```

**Post-construction schema enrichment:**
```python
adf = AliasDataFrame(df, schema_id="my_analysis_v1")
adf.update_schema({'columns': {'pt': {'dtype': 'float64', 'unit': 'GeV'}}})
```

### Subframe Registration

```python
adf.register_subframe(name, adf, index_columns)
# name: str — short name for dot-notation (e.g., 'T')
# adf: AliasDataFrame — child table (eager, in-memory)
# index_columns: str or list — join key(s), must exist in both tables
```

### Join Contract

| Property | Behavior |
|----------|----------|
| Join type | LEFT JOIN (preserve all parent rows) |
| Missing keys | NaN in child columns |
| Duplicate child keys | Parent rows expanded (many-to-one replication) |
| Ordering | Parent row order preserved |
| Pandas equivalent | `pd.merge(main, sub, on=keys, how='left', sort=False)` |
| **Cache** | **Join indices cached across `materialize_aliases` calls; invalidated only on `register_subframe` or index-column change (Phase 13.21.ADF). O(1) content-based validation via first/last/dtype signature.** |

### Function Registration (Phase 13.9/13.10)

```python
# Generic callable
adf.register_function('myFunc', callable, overwrite=False)

# Polynomial from subframe coefficients
adf.register_polynomial_from_subframe('polyCorr', poly_spec, 'CoeffSF', coeff_cols, overwrite=False)

# Evaluator (e.g., GroupByRegressionEvaluator)
adf.register_evaluator('corr', evaluator, ['x', 'y', 'z'], predictor_columns=['dy'], overwrite=False)

# Usage via alias
adf.add_alias('corrected', 'y - corr(x, y, z)')
```

### Access Pattern

```python
adf['T.column']                     # → left-join lookup
adf.add_alias('x', 'T.pt * gain')  # → cross-table expression
adf.add_alias('corr', 'polyCorr(xM, driftM)')  # → registered function
```

---

## 3. Module Architecture

### 3.1 Core Data Flow

```python
# 1. Load data (eager or lazy)
adf = AliasDataFrame.read_tree("file.root", "TreeName")          # Eager
adf = AliasDataFrame.read_tree_lazy("file.root", "TreeName")     # Lazy
adf = AliasDataFrame.read_chain(["f1.root", "f2.root"], "Tree")  # Chain

# 2. Register subframes (hierarchical data)
tracks_adf = AliasDataFrame.read_tree("file.root", "Tracks")
adf.register_subframe("T", tracks_adf, index_columns="event_id")

# 3. Define aliases (derived columns)
adf.add_alias("pt", "sqrt(px**2 + py**2)")
adf.add_alias("scaled_pt", "T.pt * gain")   # Cross-table expression

# 4. Access data (auto-materialization)
values = adf['pt']              # Triggers: branch load → alias eval → return
values = adf['T.column']       # Triggers: join → scatter → return

# 5. Visualize
fig, ax, stats = adf.draw("pt")
adf.draw_figures([{"x": "pt"}, {"y:x": "T.pt:eta"}])
```

### 3.2 Class Relationships

```
AliasDataFrame (main)
├── .df → pd.DataFrame (underlying data)
├── ._schema → dict (v2 schema with column groups, metadata)
├── ._subframes → dict[str, SubframeInfo]
│   └── SubframeInfo: (adf, index_columns, join_cache)
├── ._aliases → dict[str, str] (name → expression)
├── ._registered_functions → dict[str, callable] (Phase 13.9/13.10)
├── ._compression_info → dict (column → compression spec)
├── ._fit_metadata → dict (registered fit results)
├── ._lazy_reader → LazyTreeReader | None
├── ._use_numba → bool (auto-detected or explicit)
└── ._data_source → object | None (for dfdraw label lookup)

PolynomialSpec (Phase 13.9)
├── .dimensions → list[dict] (column, degree, mode)
├── .basis_expressions() → list[(key, expr)]
├── .numba_evaluator(coeff_subframe) → callable
├── .to_schema() / .from_schema() → JSON roundtrip
└── .to_root_expression() → C++ string for TTree::Draw

LazyTreeReader
├── .file_path, .tree_name
├── .available_branches → list[str]
├── .loaded_branches → set[str]
└── .ensure_branches(branches) → loads on demand

LazyChainReader (wraps LazyTreeReader)
├── .file_paths → list[str]
├── .validation_mode → 'first' | 'strict' | 'intersection' | 'union'
├── ._file_handle_cache → LRU(K=8)
└── Composition: creates LazyTreeReader per file
```

---

## 4. Complete Public API

> **Organization:** Methods are grouped by workflow (what you're trying to do), not by implementation module. All methods are on the `AliasDataFrame` class unless noted otherwise.

### 4.1 Constructor & Factory Methods

```python
class AliasDataFrame:
    def __init__(self, df, schema_id=None, use_numba=None, use_arrow=None):
        """
        Create from existing pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        schema_id : str, optional
            Human-readable schema identifier (e.g., "miranov_TPC_calib_v3")
        use_numba : bool, optional
            Enable Numba JIT acceleration. Default: auto-detect availability.
        use_arrow : bool, optional
            Enable PyArrow for scatter operations. Default: auto-detect.

        NOT ACCEPTED (will raise TypeError):
            schema=, metadata=
            Use update_schema() post-construction instead.

        Example
        -------
        >>> adf = AliasDataFrame(df, schema_id='my_analysis_v1')
        """

    @staticmethod
    def read_tree(file_path, tree_name,
                  entry_start=None, entry_stop=None,
                  num_workers=8, load_subframes=True,
                  dtype_overrides=None, skip_branches=None) -> 'AliasDataFrame':
        """Read ROOT TTree into eager AliasDataFrame (all branches loaded).
        Uses threaded branch-by-branch reading (Phase 1).
        Auto-restores dtypes from metadata if available (float16 roundtrip).

        Phase 13.26: dtype_overrides={regex: np.dtype} — on-the-fly type
        conversion during read. First match wins. Overflow warned.
        Phase 13.27: skip_branches=[regex] — exclude matched branches.

        Example
        -------
        >>> adf = AliasDataFrame.read_tree("data.root", "tree")
        >>> adf = AliasDataFrame.read_tree("data.root", "tree", entry_stop=10000)
        >>> adf = AliasDataFrame.read_tree("data.root", "tree",
        ...     dtype_overrides={r'.*_PIter\\d+': np.float16},
        ...     skip_branches=[r'quality_flag.*'])
        """

    @classmethod
    def read_tree_lazy(cls, file_path, tree_name,
                       schema=None) -> 'AliasDataFrame':
        """Read ROOT TTree lazily — branches loaded on demand via ensure_branches().
        90%+ memory savings for selective access (5 of 100+ branches).

        Example
        -------
        >>> adf = AliasDataFrame.read_tree_lazy("data.root", "tree")
        >>> adf.ensure_branches(['pt', 'eta'])  # loads only these 2
        """

    @classmethod
    def read_chain(cls, file_paths, tree_name,
                   validation_mode='first', **kwargs) -> 'AliasDataFrame':
        """Read multiple ROOT files into single AliasDataFrame (eager).

        Parameters
        ----------
        file_paths : list[str] or str
            File paths or glob pattern (e.g., '*.root:tree')
        validation_mode : str
            'first' | 'strict' | 'intersection' | 'union'

        Example
        -------
        >>> adf = AliasDataFrame.read_chain(["run1.root", "run2.root"], "tree")
        >>> adf = AliasDataFrame.read_chain("data_*.root:tree")
        """

    @classmethod
    def read_chain_lazy(cls, file_paths, tree_name,
                        validation_mode='first') -> 'AliasDataFrame':
        """Read multiple ROOT files lazily. LRU cache (K=8) for file handles.

        Example
        -------
        >>> adf = AliasDataFrame.read_chain_lazy("data_*.root:tree")
        """

    @classmethod
    def from_schema(cls, schema_dict) -> 'AliasDataFrame':
        """Create empty AliasDataFrame from schema template (no data).

        Example
        -------
        >>> schema = AliasDataFrame.load_schema('calib_schema.json')
        >>> adf_template = AliasDataFrame.from_schema(schema)
        """
```

### 4.2 I/O & Export

```python
    # ── ROOT TTree Export ──
    def export_tree(self, filename_or_file, treename='tree',
                    dropAliasColumns=True, compression=uproot.ZLIB(level=1),
                    columns=None):
        """
        Export to ROOT TTree via uproot. Requires uproot.
        Stores column dtypes as metadata for roundtrip preservation.

        Parameters
        ----------
        filename_or_file : str or file-like
            Output .root file path or open file handle
        treename : str, default 'tree'
            TTree name in output file
        dropAliasColumns : bool, default True
            If True, exclude alias columns from export (only physical columns).
            Set False to include materialized aliases in output.
        compression : uproot compression, default uproot.ZLIB(level=1)
            ROOT compression algorithm and level
        columns : list[str], optional
            Subset of columns to export. Default: all columns in .df

        IMPORTANT:
            - Aliases must be materialized BEFORE export if dropAliasColumns=False
            - Schema metadata is NOT embedded in ROOT file — save as sidecar JSON
            - float16 columns are written as float32 (ROOT limitation) but dtype
              metadata enables automatic restoration on read_tree()

        Example
        -------
        >>> adf.export_tree('output.root')
        >>> adf.export_tree('output.root', treename='recoVertices')
        >>> adf.materialize_aliases()
        >>> adf.export_tree('full.root', dropAliasColumns=False)
        >>> adf.export_tree('slim.root', columns=['pt', 'eta', 'pull_z'])
        """

    # ── RDataFrame Integration (AliasDataFrameRDF.py) ──
    def setup_rdf_with_friends(self) -> tuple:
        """Set up RDataFrame with friend trees for subframes.
        Returns (rdf, file_handle) — keep file_handle alive."""

    def add_defines_to_rdf(self, rdf, on_collision='warn'):
        """Add alias definitions to RDataFrame as Define() calls.
        Translates Python expressions to C++ equivalents."""

    # ── Schema Serialization ──
    def save_schema(self, path):
        """Save schema to JSON file (v2 format).
        Simple wrapper — calls save_schema_v2() with default settings.
        For advanced control (compression stats, formatting), use save_schema_v2() directly.

        Example
        -------
        >>> adf.save_schema('calib_schema.json')
        """

    @staticmethod
    def load_schema(path) -> dict:
        """Load schema from JSON file. Auto-detects v1 and v2 formats.

        Example
        -------
        >>> schema = AliasDataFrame.load_schema('calib_schema.json')
        >>> adf.apply_schema(schema)
        """
```

### 4.3 Alias Management

```python
    def add_alias(self, name, expression, fill_value=None, **kwargs):
        """
        Define a derived column (computed on access).

        Parameters
        ----------
        name : str
            Alias name (accessible as adf['name'])
        expression : str
            Python expression. Supports: column names, np.* functions,
            cross-table references ('SubframeName.column'),
            registered functions ('funcName(arg1, arg2)')
        fill_value : scalar, optional
            Replace inf/NaN after evaluation (e.g., fill_value=0 for division)

        Examples
        --------
        >>> adf.add_alias('pt', 'sqrt(px**2 + py**2)')
        >>> adf.add_alias('ratio', 'T.pt / pt')                    # Cross-table
        >>> adf.add_alias('side', 'sec >= 18', dtype=np.int8)      # With dtype
        >>> adf.add_alias('sector', '18*((y+dy)/x)/pi', fill_value=0)  # Safe div
        >>> adf.add_alias('corr', 'myEvaluator(x, y, z)')          # Registered func
        """

    def materialize_alias(self, name) -> pd.Series:
        """Force evaluation of a single alias. Result stored in .df."""

    def materialize_aliases(self, names=None):
        """Force evaluation of multiple aliases (batch). If names=None, all aliases."""

    def get_required_branches(self, expression) -> set:
        """AST-based detection of branches needed for expression."""

    def describe_aliases(self, verbosity=0x07, pattern=None):
        """Print alias definitions, dependencies, materialization status."""

    def dematerialize(self, drop=None, keep=None) -> list:
        """Drop materialized alias columns to reclaim memory.
        
        Raw columns always protected. Aliases/subframes preserved — 
        re-materialization via materialize_aliases() recovers values.
        Three modes: drop=[...], keep=[...], no args (drop all).
        Mutually exclusive drop/keep (ValueError if both).
        Composes with join index caching: re-materialization reuses
        cached join indices since index columns are unchanged.
        Phase 13.21.ADF. Replaces drop_materialized() (removed).
        """
```

### 4.4 Schema Management

```python
    def update_schema(self, schema_dict):
        """Merge schema updates into existing _schema (deep merge, recommended pattern).

        Example
        -------
        >>> adf.update_schema({
        ...     '__meta__': {'source': 'ORecoAI', 'version': 'v1.0'},
        ...     'columns': {'vtx_z': {'dtype': 'float64', 'unit': 'cm'}},
        ... })
        """

    def export_schema(self) -> dict:
        """Export full schema (v2) as JSON-serializable dict.
        Includes: columns, aliases, subframes, compression, fit metadata, groups."""

    def apply_schema(self, schema, validate=True, warn_missing=True):
        """Apply schema to existing AliasDataFrame.
        Applies: dtypes, aliases, compression info, metadata, groups."""

    def export_definition_schema(self) -> dict:
        """Export blueprint schema (structure only, no runtime state)."""

    def export_record_schema(self) -> dict:
        """Export snapshot schema (full state including current data stats)."""

    def validate_schema(self, mode='strict') -> bool:
        """Validate current state against schema. Returns True if valid."""
```

### 4.5 Subframe Registration

```python
    def register_subframe(self, name, adf, index_columns, pre_index=False):
        """
        Register a child table for hierarchical joins.

        Parameters
        ----------
        name : str
            Short name for dot-notation access (e.g., 'T' for tracks)
        adf : AliasDataFrame
            Child table (must be an AliasDataFrame, not plain DataFrame)
        index_columns : str or list[str]
            Column(s) used for left join (must exist in both tables)
        pre_index : bool, default False
            If True, pre-compute join indices at registration time.
            Faster for repeated access, but uses more memory.

        Join contract:
            - LEFT JOIN: all parent rows preserved
            - Missing keys → NaN in child columns
            - Duplicate child keys → parent rows expanded (many-to-one, Cartesian)
            - Ordering: parent row order preserved
            - Equivalent: pd.merge(main, sub, on=keys, how='left', sort=False)

        After registration:
            adf['T.column'] → left-join lookup from child table
            adf.add_alias('x', 'T.pt * gain') → cross-table expression

        Example
        -------
        >>> tracks_adf = AliasDataFrame(tracks_df)
        >>> adf.register_subframe('T', tracks_adf, index_columns=['event_id'])
        >>> adf['T.pt']  # left-join lookup
        """

    def register_subframe_chain(self, name, file_paths, tree_name,
                                index_columns, validation_mode='first'):
        """Register subframe from multiple ROOT files (lazy chain).

        Example
        -------
        >>> adf.register_subframe_chain('Calib', 'calib_*.root', 'tree',
        ...                              index_columns=['run', 'sector'])
        """
```

> ⚠️ **BROKEN API — Do not use:**
> ```python
> def register_subframe_lazy(self, name, file_path, tree_name, index_columns):
>     """⚠️ KNOWN BUG (BUG_AliasDataFrame_20260116_lazy_subframe_init):
>     Missing _df attribute after initialization.
>     Use register_subframe() (eager) instead.
>     Tracked: docs/BUG_AliasDataFrame_20260116_lazy_subframe_init.md
>     Covered by: I1_5, I1_7 (skipped invariance tests)"""
> ```

### 4.6 Lazy Loading

```python
    # ── Properties ──
    @property
    def is_lazy(self) -> bool:
        """True if this ADF was created with read_tree_lazy/read_chain_lazy."""

    @property
    def available_branches(self) -> list:
        """All branches available in the ROOT file (lazy mode only)."""

    @property
    def loaded_branches(self) -> set:
        """Branches that have been loaded so far (lazy mode only)."""

    # ── Loading ──
    def ensure_branches(self, branches):
        """
        Load specific branches from ROOT file (lazy mode).
        No-op if branch already loaded. No-op if ADF is eager.

        Parameters
        ----------
        branches : str or list[str]
            Branch name(s) to load

        Example
        -------
        >>> adf = AliasDataFrame.read_tree_lazy("data.root", "tree")
        >>> adf.ensure_branches(['pt', 'eta', 'phi'])
        >>> adf.df[['pt', 'eta', 'phi']]  # now available

        Note: adf['column'] auto-calls ensure_branches, so explicit
        calls are only needed for batch pre-loading.
        """
```

### 4.7 Data Access (Proxy Pattern)

```python
    # Direct access (proxy delegates to .df)
    adf['column']        # → auto-loads branch (if lazy), then returns .df[column]
                         # NOTE: Does NOT auto-materialize aliases or auto-join subframes.
                         # Use materialize_alias('name') first, then adf['name'].
    adf.columns          # → .df.columns
    adf.shape            # → .df.shape
    len(adf)             # → len(.df)
    'col' in adf         # → True if col in .df.columns (physical columns only)
                         # Aliases and subframe refs are NOT checked.
    adf.loc[...]         # → .df.loc[...]
    adf.iloc[...]        # → .df.iloc[...]
    adf.df               # → direct access to underlying pd.DataFrame

    # Blocked (to prevent accidental overwrites):
    adf['x'] = val       # → raises error with helpful message
    # Use adf.df['x'] = val for direct DataFrame mutation
```

### 4.8 Column Metadata & Axis Titles

```python
    # ── Axis Titles (used by dfdraw for automatic label resolution) ──
    def set_axis_title(self, column, title):
        """Set display title for plotting. Stored in schema, used by dfdraw.

        Example
        -------
        >>> adf.set_axis_title('vtx_z', 'z_{vtx} (cm)')
        >>> adf.set_axis_title('pull_z', 'Pull z = Δz/σ_z')
        """

    def get_axis_title(self, column) -> str | None:
        """Get display title for column/alias, or None if not set.
        Returns value from _schema['columns'][column]['title']."""

    # ── Rich Column Metadata (arbitrary key-value) ──
    def set_column_metadata(self, column, **metadata):
        """Set metadata for one column. Supports any key-value pairs.
        Note: axis title is stored under key 'title' (not 'axisLabel').

        Example
        -------
        >>> adf.set_column_metadata('vtx_z',
        ...     unit='cm',
        ...     title='z_{vtx} (cm)',
        ...     description='Reconstructed vertex Z position',
        ...     range=[-15, 15]
        ... )
        """

    def set_columns_metadata(self, metadata_dict):
        """Set metadata for multiple columns at once.

        Example
        -------
        >>> adf.set_columns_metadata({
        ...     'vtx_z': {'unit': 'cm', 'title': 'z_{vtx} (cm)'},
        ...     'sigma_vtx_z': {'unit': 'cm', 'title': 'σ_{z} (cm)'},
        ...     'dz_truth': {'unit': 'cm', 'title': 'Δz_{truth} (cm)'},
        ... })
        """

    def get_column_metadata(self, column) -> dict | None:
        """Get all metadata for a column, or None."""
```

### 4.9 Data Inspection & Description

```python
    def describe_data(self, verbosity=0x07, pattern=None, names=None,
                      only_physical=False, only_aliases=False,
                      sort_by='name', as_dict=False):
        """Print column summary with dtype, memory, nulls, metadata.

        Verbosity flags (bitmask):
            0x01 DATA_SHOW_CORE   — name, dtype, memory
            0x02 DATA_SHOW_STATS  — null count, unique values
            0x04 DATA_SHOW_META   — user metadata (unit, description)
            0x08 DATA_SHOW_SOURCE — physical vs alias vs subframe_ref
            0x0F DATA_SHOW_ALL

        Example
        -------
        >>> adf.describe_data()
        >>> adf.describe_data(pattern=r'vtx_.*', sort_by='memory')
        """

    def describe_structure(self):
        """Print comprehensive structure summary: columns, aliases, compression, subframes.

        Example output
        --------------
        AliasDataFrame Structure
        ========================
        DataFrame: 12,035,611 rows × 70 columns
        Memory: 3,179 MB
        Aliases: 12 defined
        Compression: 8 columns
        Subframes: 4
        """

    def describe_schema(self, sections=None, as_dict=False):
        """Print schema overview: column counts, groups, metadata coverage.

        Example
        -------
        >>> adf.describe_schema(sections=['columns', 'compression', 'subframes'])
        """

    def describe_compression(self, verbosity=0x07, pattern=None):
        """Print compression state for compressed columns."""

    def select_data(self, pattern=None, dtype=None, only_physical=True,
                    min_memory_mb=None) -> list:
        """Filter column names by criteria. Returns list of column names.

        Example
        -------
        >>> float32_cols = adf.select_data(dtype=np.float32)
        >>> large_cols = adf.select_data(min_memory_mb=50)
        """

    def select_schema(self, dtype=None, has_metadata=None,
                      is_subframe_ref=None) -> list:
        """Filter schema entries by criteria. Returns list of column names."""
```

### 4.10 Dtype Management

```python
    def convert_dtypes(self, dtype_map):
        """Batch convert column dtypes. Updates schema automatically.

        Parameters
        ----------
        dtype_map : dict
            {column_name: target_dtype}

        Example
        -------
        >>> adf.convert_dtypes({
        ...     'dy': np.float16,
        ...     'dz': np.float16,
        ...     'tgSlp': np.float16,
        ... })
        """

    def convert_dtypes_pattern(self, pattern, target_dtype):
        """Convert all columns matching regex pattern to target dtype.

        Example
        -------
        >>> adf.convert_dtypes_pattern(r'dy.*', np.float16)
        """
```

### 4.11 Schema Groups

```python
    def set_groups(self, groups):
        """Define column groups for schema organization and export ordering.

        Example
        -------
        >>> adf.set_groups({
        ...     'coordinates': ['x', 'y', 'z', 'r', 'phi'],
        ...     'calibtrack': ['dy_TrackFit0', 'dz_TrackFit0'],
        ...     'cuts': ['isOK', 'isOKGB'],
        ... })
        """

    def get_groups(self) -> dict:
        """Get current column groups."""
```

### 4.12 Compression

```python
    def compress_columns(self, spec) -> dict:
        """
        Compress columns with user-defined quantization.

        Parameters
        ----------
        spec : dict
            {column_name: {'formula': str, 'bits': int, ...}}
            Supported formulas: 'linear', 'log', 'sqrt', 'asinh', custom

        Returns
        -------
        dict : compression summary with RMSE per column

        Example
        -------
        >>> adf.compress_columns({
        ...     'pt': {'formula': 'linear', 'bits': 12, 'range': [0, 100]},
        ...     'phi': {'formula': 'linear', 'bits': 10, 'range': [-3.15, 3.15]},
        ... })
        """

    def decompress_columns(self, columns=None):
        """Decompress columns (idempotent). If columns=None, decompress all."""

    def get_compression_info(self) -> dict:
        """Return current compression state for all compressed columns."""
```

### 4.13 Fit Registration

```python
    def register_fit_result(self, name, fit_result, **metadata):
        """Register a fit result (e.g., from GroupBy regression) with metadata.
        Enables draw_fit_summary() for QA dashboards.

        Example
        -------
        >>> adf.register_fit_result('driftV_calib', dfGB,
        ...     fit_columns=['driftV'], linear_columns=['spaceCharge'],
        ...     suffix='_fit')
        """
```

### 4.14 Drawing & Visualization

```python
    def draw(self, expression, selection=None, group_by=None, **kwargs) -> tuple:
        """
        Plot expression using dfdraw. Auto-loads branches, auto-materializes aliases.
        Axis labels auto-populated from schema metadata (set_axis_title).

        Parameters
        ----------
        expression : str
            Plot expression (e.g., 'pt', 'y:x', 'dz:r:phi')
        selection : str, optional
            Row filter expression (e.g., 'eta > 0')
        group_by : str, optional
            Group by column for overlaid plots

        Returns
        -------
        (fig, ax, stats_dict) : matplotlib figure, axes, statistics

        Label precedence: explicit kwarg > schema title > column name

        Example
        -------
        >>> fig, ax, stats = adf.draw('dz_truth:vtx_z')
        >>> fig, ax, stats = adf.draw('pull_z', selection='n_contributors > 100')
        """

    def draw_figures(self, specs, **kwargs):
        """Batch plotting with composed canvas (multi-subplot QA dashboards).

        Example
        -------
        >>> adf.draw_figures([
        ...     {'x': 'pt'},
        ...     {'y:x': 'dz:eta'},
        ...     {'x': 'pull_z', 'selection': 'is_fake == 0'},
        ... ])
        """

    def draw_fit_summary(self, fit_name, **kwargs):
        """Generate QA dashboard for registered fit result (see register_fit_result)."""
```

### 4.15 Registered Functions (Phase 13.9/13.10)

```python
    def register_function(self, name, func, overwrite=False):
        """
        Register a callable as an alias function.

        Parameters
        ----------
        name : str
            Function name (usable in alias expressions)
        func : callable
            Function that takes positional array arguments, returns array
        overwrite : bool, default False
            If True, replace existing function with same name

        Example
        -------
        >>> adf.register_function('scale', lambda x, factor: x * factor)
        >>> adf.add_alias('scaled', 'scale(pt, 1.5)')
        """

    def register_polynomial_from_subframe(self, func_name, poly_spec,
                                          coefficients_subframe, coeff_select,
                                          overwrite=False):
        """
        Register a polynomial evaluator using coefficients from a subframe.

        Parameters
        ----------
        func_name : str
            Function name for alias expressions
        poly_spec : PolynomialSpec
            Polynomial specification defining basis terms
        coefficients_subframe : str
            Name of registered subframe containing coefficient columns
        coeff_select : list[str]
            Column names in subframe that contain the coefficients
            (one per basis term, in order matching poly_spec.basis_expressions())
        overwrite : bool, default False
            Replace existing function if True

        Architecture:
            - Coefficients stay at subframe size (e.g., 36 sectors × 96 terms = 27 KB)
            - No materialization to main frame length (13.5M rows)
            - Numba JIT evaluator: 42× faster than pandas.eval

        Example
        -------
        >>> from PolynomialSpec import PolynomialSpec
        >>> spec = PolynomialSpec(columns=['xM', 'driftM', 'dsecM', 'tgSlp'],
        ...                       degrees=(3, 3, 2, 1))
        >>> coeff_cols = [f'T{i}' for i in range(spec.n_terms)]
        >>> adf.register_polynomial_from_subframe('polyCorr', spec, 'CoeffSF', coeff_cols)
        >>> adf.add_alias('dy_corr', 'dy - polyCorr(xM, driftM, dsecM, tgSlp)')
        """

    def register_evaluator(self, name, evaluator, coord_columns,
                           predictor_columns=None, overwrite=False):
        """
        Register an evaluator object (duck-typed) as an alias function.

        Parameters
        ----------
        name : str
            Function name for alias expressions
        evaluator : object
            Any object with .evaluate(positions: dict) -> ndarray | dict
            (e.g., GroupByRegressionEvaluator from GBAI team)
        coord_columns : list[str]
            Column names that map to evaluator's position dict keys
        predictor_columns : list[str], optional
            For multi-predictor evaluators: which output(s) to use.
            If evaluator returns dict with multiple keys and this is None,
            raises ValueError.
        overwrite : bool, default False
            Replace existing function if True

        Schema:
            Stores interface contract only (coord_columns, predictor_columns).
            Evaluator object is NOT serialized — must re-register after load.

        Example
        -------
        >>> # evaluator.evaluate({'xM': arr, 'driftM': arr}) -> {'dy': arr, 'dz': arr}
        >>> adf.register_evaluator('corr', evaluator, ['xM', 'driftM'],
        ...                         predictor_columns=['dy'])
        >>> adf.add_alias('dy_I1', 'dy - corr(xM, driftM)')
        """
```

---

## 5. Data Model & Hierarchical Representation

### 5.1 Flat vs Normalized

AliasDataFrame supports two data representations that produce identical computed results:

```
Option A: Flat (replicated parent data)
┌─────────┬─────┬───────┬────┐
│event_id │ run │ track │ pt │    ← parent columns replicated per child row
├─────────┼─────┼───────┼────┤
│ 0       │ 100 │  0    │1.0 │
│ 0       │ 100 │  1    │2.0 │    run=100 appears twice (once per track)
│ 1       │ 101 │  0    │1.5 │
└─────────┴─────┴───────┴────┘

Option C: Normalized (subframes, no replication)
Events:                    Tracks:
┌─────────┬─────┐         ┌─────────┬───────┬────┐
│event_id │ run │         │event_id │ track │ pt │
├─────────┼─────┤         ├─────────┼───────┼────┤
│ 0       │ 100 │         │ 0       │  0    │1.0 │
│ 1       │ 101 │         │ 0       │  1    │2.0 │
└─────────┴─────┘         │ 1       │  0    │1.5 │
                          └─────────┴───────┴────┘
    adf.register_subframe('T', tracks_adf, 'event_id')
    adf['T.pt']  ← produces [1.0, 2.0, 1.5] via left join
```

**Critical invariant (I3_9):** `flat_adf['pt * run'] == normalized_adf['pt * E.run']`  
This is validated by the invariance test suite and guarantees Team 2 integration correctness.

### 5.2 Join Semantics (Precise)

When accessing `adf['SubframeName.column']`:

1. **Index lookup:** Find join indices via `index_columns` (left join semantics)
2. **Scatter:** Extract values from child table at those indices
3. **Backend dispatch:** Numba (primary) → NumPy → Pandas fallback
4. **Caching:** Join indices cached per subframe

**Join contract (tested by I3 invariance suite):**

| Property | Behavior | Pandas Equivalent |
|----------|----------|-------------------|
| Join type | LEFT JOIN | `how='left'` |
| Missing keys | NaN in child columns | Same |
| Duplicate child keys | Parent rows expanded (many-to-one replication, Cartesian product) | `validate=None` (allow duplicates) |
| Ordering | Parent row order preserved | `sort=False` |
| Multi-key | AND condition on all keys | `on=['key1', 'key2']` |

```python
# Precise pandas equivalent of adf['T.value']:
pd.merge(main_df, sub_df, on=index_columns, how='left', sort=False, validate=None)['value']
```

**Cache performance (Phase 8 benchmark):** 87.5% hit rate on representative workloads (Phase 3 roofline, commit `46d2320`).

### 5.3 Multi-Column Keys

For composite join keys (e.g., `[event_id, sector]`):
- Keys linearized to single int64: `key = event_id * max_sector + sector`
- Uses GLOBAL max values from both tables (Phase 8c, commit `46d2320`)
- Auto fallback for overflow, negative, or non-integer keys

---

## 6. Alias & Lazy Evaluation System

### 6.1 Alias Resolution Chain

```
adf.add_alias('pt', 'sqrt(px**2 + py**2)')
adf.add_alias('pt_gev', 'pt / 1000')
adf.add_alias('scaled', 'T.pt * gain')

adf['pt_gev']  →  resolves: pt_gev → pt → px, py
                   loads: px, py branches (if lazy)
                   evaluates: pt, then pt_gev
                   returns: pd.Series

adf['scaled']  →  resolves: scaled → T.pt (subframe), gain (local)
                   loads: T.pt from child, gain from parent
                   joins: left join on index_columns
                   evaluates: T.pt * gain at child granularity
                   returns: pd.Series
```

### 6.2 Lazy Loading Contract

After `read_tree_lazy()`:
- `adf.is_lazy == True`
- `adf.available_branches` — all branches in ROOT file
- `adf.loaded_branches` — only what's been accessed
- Any access triggers automatic load: `adf['x']` calls `ensure_branches(['x'])`
- After materialization: lazy ADF behaves identically to eager ADF

**Invariant (I1):** `read_tree_lazy() + materialize == read_tree()`

### 6.3 Expression Evaluation

Aliases use `pandas.eval()` with Python engine, plus custom handling for:
- `np.*` functions (sin, cos, sqrt, exp, log, abs, etc.)
- Cross-table references (`SubframeName.column` → join + scatter)
- Subframe expressions like `SubframeName.column * local_column`

### 6.4 Slice-First Optimization (Phase 6.8a)

For draw operations with selection:
```python
adf.draw("pt", selection="eta > 0")
# 1. Apply selection FIRST → get mask
# 2. Evaluate alias on subset only
# 3. 12M rows with 10K mask → evaluates on 10K only
```

### 6.5 Registered Function System (Phase 13.9/13.10)

AliasDataFrame supports three types of registered functions:

#### 6.5.1 Generic Callables (`register_function`)

```python
# Simple lambda or function
adf.register_function('scale', lambda x, factor: x * factor)
adf.add_alias('scaled_pt', 'scale(pt, 2.0)')
```

#### 6.5.2 Polynomial Evaluators (`register_polynomial_from_subframe`)

For calibration workflows with polynomial corrections stored in subframes:

```python
# PolynomialSpec defines N-dimensional polynomial basis
spec = PolynomialSpec(columns=['xM', 'driftM', 'dsecM', 'tgSlp'],
                      degrees=(3, 3, 2, 1))

# Coefficients subframe: 36 sectors × n_terms coefficients
# coeff_cols must match order of spec.basis_expressions()
coeff_cols = [f'T{i}' for i in range(spec.n_terms)]
adf.register_polynomial_from_subframe('polIter0', spec, 'CoeffSF', coeff_cols)

# Usage — evaluates polynomial at each row's position
adf.add_alias('dy_corr', 'dy - polIter0(xM, driftM, dsecM, tgSlp)')
```

**Architecture decision:** `tgSlp` is a standard polynomial dimension (degree 1), not a special "coupling column". Coupling is expressed via normal alias algebra (`dy * tgSlp`), not via a dedicated parameter.

**Memory efficiency:** Coefficient matrix stays at subframe size (36 × 96 = 3,456 floats = 27 KB). No broadcast to main frame length (13.5M rows × 96 terms = 5.2 GB avoided).

#### 6.5.3 Evaluator Objects (`register_evaluator`)

For integration with GroupByRegressionEvaluator and similar objects:

```python
# evaluator has .evaluate(positions: dict) -> ndarray or dict
# positions = {'xM': array, 'driftM': array, 'sector': array}

adf.register_evaluator('corr_I1', evaluator, 
                       coord_columns=['xM', 'driftM', 'sector'])

# Multi-predictor evaluator returns {'dy': arr, 'dz': arr}
adf.register_evaluator('corr_dz', evaluator, 
                       coord_columns=['xM', 'driftM'],
                       predictor_columns=['dz'])

# Usage
adf.add_alias('dy_I1', 'dy - corr_I1(xM, driftM, sector)')
adf.add_alias('dz_I1', 'dz - corr_dz(xM, driftM)')
```

**Schema serialization:** Only the interface contract is stored (`coord_columns`, `predictor_columns`, `type='evaluator'`). The evaluator object itself is NOT serialized — user must re-register after load. GBAI team handles evaluator persistence separately.

---

## 7. Subframe Registration & Join Semantics

### 7.1 Registration Methods

| Method | Data Source | Loading | Status |
|--------|-----------|---------|--------|
| `register_subframe(name, adf, index_columns)` | In-memory ADF | Eager | ✅ Working |
| `register_subframe_chain(name, paths, tree, index_columns, validation_mode)` | Multiple ROOT files | Lazy | ✅ Working |
| `register_subframe_lazy(name, path, tree, index_columns)` | Single ROOT file | Lazy | ⚠️ **BROKEN** (see §4.4) |

### 7.2 Join Implementation

```python
# Internal join flow for adf['T.column']:
def _get_subframe_column(self, subframe_name, column_name):
    info = self._subframes[subframe_name]
    
    # 1. Compute or retrieve cached join indices
    if not info.has_cached_indices():
        indices = compute_join_indices(
            self.df[info.index_columns],      # parent keys
            info.adf.df[info.index_columns]   # child keys
        )
        info.cache_indices(indices)
    
    # 2. Scatter: extract child values at parent positions
    values = scatter(info.adf.df[column_name], indices)
    
    # 3. Return as Series aligned with parent index
    return pd.Series(values, index=self.df.index)
```

### 7.3 Backend Dispatch for Join/Scatter Operations

**Implemented dispatch order** (source: `_extract_subframe_values_cached`):

| Operation | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|-----------|-----------|
| **Scatter** (extract values at indices) | Numba `numba_scatter()` | Arrow `pc.take()` (if `use_arrow`) | NumPy advanced indexing |
| **Join index** (compute indices) | Numba `numba_compute_join_indices()` | NumPy | Pandas `pd.merge` |
| **Multi-key linearization** | Numba | NumPy | Pandas (multi-column merge) |
| **Expression eval** | NumPy (via `pandas.eval`) | Pandas | — |

> **Note:** Arrow compute for expressions was evaluated in Phase 9c and **disabled** (16× slower than NumPy due to lack of expression fusion). Arrow is used only for `pc.take()` scatter operations where it is competitive.

### 7.4 Comparison with RootInteractive CDSJoin

| Feature | AliasDataFrame | RootInteractive CDSJoin |
|---------|---------------|------------------------|
| Join type | Left join (pandas semantics) | Inner/left/right/outer |
| Key types | Integer, multi-column composite | Sorted merge, index lookup |
| Execution | Python/Numba server-side | TypeScript client-side |
| Caching | Join index cache | Recompute on change |
| Sparse ND→1D | ✅ (composite key linearization) | ❌ (proposed Phase 1.1.A) |
| Auto-alias | ✅ (dot notation: `T.column`) | ❌ (proposed) |
| Tolerance join | ❌ | ✅ (approximate matching) |

---

## 8. Draw Integration with dfdraw

### 8.1 Integration Pattern

```python
# AliasDataFrame.draw() → dfdraw.DFDraw
def draw(self, expression, selection=None, group_by=None, **kwargs):
    # 1. Parse expression for required branches
    branches = self.get_required_branches(expression)
    
    # 2. Load branches (if lazy)
    self.ensure_branches(branches)
    
    # 3. Materialize required aliases
    self.materialize_aliases(required_aliases)
    
    # 4. Create DFDraw and plot
    from dfextensions.dfdraw import DFDraw
    drawer = DFDraw(self.df, data_source=self)
    return drawer.draw(expression, selection=selection, group_by=group_by, **kwargs)
```

### 8.2 Return Contract

```python
fig, ax, stats = adf.draw("pt")
# fig: matplotlib.Figure
# ax: matplotlib.Axes
# stats: {'n': int, 'mean': float, 'std': float, 'min': float, 'max': float}
```

### 8.3 Label Resolution

dfdraw uses duck-typed label lookup from AliasDataFrame's schema:
```
Label precedence: explicit parameter > schema metadata (title) > column name
```

---

## 9. Backend Strategy

### 9.1 Decision Matrix (from Phase 9, commit `a788a68`)

| Operation | Backend | Rationale |
|-----------|---------|-----------|
| Scatter/Gather | Numba (primary), Arrow `pc.take` (secondary) | Numba 8× faster; Arrow competitive for large arrays |
| Expression eval | NumPy via `pandas.eval()` | Arrow compute 16× slower (no expression fusion, Phase 9e) |
| Join index | Numba JIT | 8× faster than NumPy for index computation (Phase 8a) |
| Multi-column key | Numba (linearize to int64) | 1.5× faster with int64 composition (Phase 8c) |

### 9.2 Implemented Fallback Chains

```
Scatter:          Numba → Arrow pc.take (if use_arrow) → NumPy
Join index:       Numba → NumPy → Pandas pd.merge
Expression eval:  NumPy (pandas.eval) → Pandas
I/O:              uproot (ROOT), pyarrow (Parquet)
```

### 9.3 Constructor-Level Backend Selection

```python
adf = AliasDataFrame(df, use_numba=True)   # Force Numba (default: auto-detect)
adf = AliasDataFrame(df, use_numba=False)  # Force NumPy fallback
adf = AliasDataFrame(df, use_arrow=True)   # Enable Arrow scatter path
```

---

## 10. Compression System

### 10.1 Architecture

```python
# Compress with user-defined formula
adf.compress_columns({
    'pt': {'formula': 'linear', 'bits': 12, 'range': [0, 100]},
    'phi': {'formula': 'linear', 'bits': 10, 'range': [-3.15, 3.15]},
    'dEdx': {'formula': 'log', 'bits': 8}
})

# Decompress (idempotent)
adf.decompress_columns(['pt', 'phi'])
```

### 10.2 Supported Formulas

| Formula | Transform | Use Case |
|---------|-----------|----------|
| `linear` | `(x - min) / (max - min) * (2^bits - 1)` | Uniform distributions |
| `log` | `log(x)` then linear | Exponential distributions |
| `sqrt` | `sqrt(x)` then linear | Poisson-like |
| `asinh` | `asinh(x/scale)` then linear | Signed with wide dynamic range |
| Custom | User string expression | Any transform |

### 10.3 Schema Integration

Compression state stored in schema and preserved through export/import:
```python
schema['compression'] = {
    'pt': {'formula': 'linear', 'bits': 12, ...},
    'phi': {'formula': 'linear', 'bits': 10, ...}
}
```

---

## 11. Cross-Subproject Dependencies

### 11.1 What AliasDataFrame Requires

| Dependency | From | Purpose |
|-----------|------|---------|
| `uproot` | External | ROOT file reading |
| `pandas` | External | Core data storage |
| `numpy` | External | Numerical operations |
| `numba` | External (optional) | JIT acceleration |
| `pyarrow` | External (optional) | Arrow scatter, Parquet I/O |
| `matplotlib` | External | Drawing (via dfdraw) |
| `dfdraw` (DFDraw) | Team 3 | Visualization backend |

### 11.2 What Depends on AliasDataFrame

| Consumer | Interface Used | Contract |
|----------|---------------|----------|
| **RDataFrameDSL** (Team 2) | `AliasDataFrame()` constructor, `register_subframe()` | `export_to_aliasdf()` creates ADF with subframes from flattened DSL output |
| **dfdraw** (Team 3) | `.draw()` return contract `(fig, ax, stats)` | ADF is the primary upstream data provider |
| **RootInteractive** | Indirect (shared data model concepts) | AliasDataFrame's join model informs RootInteractive CDSJoin design |
| **O2DPG pipeline** | `read_tree()`, `export_tree()`, schema | Production data analysis workflows |

### 11.3 Team 2 Integration Contract (Phase 13.6.B)

```python
# Team 2 (RDataFrameDSL) calls:
adf = AliasDataFrame(df, schema_id="...")       # ✅ Tested (6 contract tests)
adf.register_subframe("T", tracks_adf,          # ✅ Tested
                       index_columns="event_id")

# Parameter names are API contract:
# Constructor: schema_id ✅, use_numba ✅, use_arrow ✅
# NOT: schema=, metadata= (will raise TypeError)
# register_subframe: name, adf, index_columns ✅
# NOT: subframe=, on= (will raise TypeError)
```

### 11.4 Upcoming: O2 SOA Integration (Phase 11, Planned)

Vision: automatic conversion of C++ SOA (Structure of Arrays) table definitions to AliasDataFrame schemas.
```
*.h (O2 SOA headers) → schema.json → AliasDataFrame
```

---

## 12. Test Coverage

### 12.1 Current Status (2026-04-19)

**Command:** `pytest tests/` (with ROOT enabled, Numba available, PyArrow available)

| Metric | Value |
|--------|-------|
| Collected | ~1537 |
| Passed | 1521 |
| Failed | 7 |
| Skipped | 8 |
| Runtime | ~26s (12 workers) |
| Invariance tests | 125 |

**Environment assumptions:** ROOT 6.x installed, `numba` available, `pyarrow` available, `dfdraw` available. Without ROOT, ~50 tests are skipped (ROOT-dependent integration tests).

### 12.2 Phase 13 Test Additions

| Phase | Tests Added | Total After | Description |
|-------|-------------|-------------|-------------|
| 13.9.ADF PolynomialSpec | 26 | 1402 | Polynomial specification, register_function, register_polynomial_from_subframe |
| 13.10.ADF register_evaluator | 24 | 1425 | Evaluator integration, multi-predictor, schema contract |
| BUG draw_subframe_resolution | 23 | 1392 (at fix) | draw/draw_batch/draw_figures subframe resolution |
| BUG fill_value_dependency (P0) | 7 | 1432 | fill_value in batch dependency chain |
| 13.11.ADF test infrastructure | — | 1432 | run_tests.sh, CAPABILITY_MATRIX, @invariance markers |
| 13.9.Fix1 polynomial persistence | 9 | 1441 | Schema export/import roundtrip for registered_functions |
| BUG draw_lazy_compound (P1) | 12 | 1441 | Compound expression alias resolution in draw_lazy |
| 13.11.B taxonomy | — | 1441 | 41-feature taxonomy, generate_capability_matrix v2 |

### 12.3 Invariance Test Suite (Phase 13.7.ADF)

#### Committed (Phase 13.7.ADF.1)

| Category | File | Tests | Passed | Skipped | Status |
|----------|------|-------|--------|---------|--------|
| I0 Smoke | `test_invariance_smoke.py` | 6 | 6 | 0 | ✅ |
| I1 Load Mode | `test_invariance_load_mode.py` | 8 | 6 | 2 | ✅ (skips: lazy subframe bug) |
| I3 Subframe | `test_invariance_subframe.py` | 9 | 9 | 0 | ✅ |

#### In Validation

| Category | File | Tests | Passed | Failed | Status |
|----------|------|-------|--------|--------|--------|
| I2 Backend | `test_invariance_backend.py` | 11 | 10 | 1 | 🔄 I2_6 failure |
| I4 Compression | `test_invariance_compression.py` | 9 | 7 | 2 | 🔄 I4_2, I4_3 failures |

### 12.4 Pre-existing Failures (6)

| Test File | Failures | Category |
|-----------|----------|----------|
| `test_AliasDataFrameRDF.py` | 3 | RDF friend access, collision, missing keys |
| `test_invariance_backend.py` | 1 | I2_6: chained subframe Numba vs NumPy |
| `test_invariance_compression.py` | 2 | I4_2, I4_3: scaled compression roundtrip |

### 12.5 Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| Core functionality | `test_alias_dataframe.py` | Main module tests |
| Constructor contract | `test_constructor_contract.py` | Phase 13.6.B API lock (6 tests) |
| Schema | `test_alias_data_frame_schema*.py` | Schema v1/v2, validation |
| Lazy loading | `test_lazy_*.py` | Branch, chain, subframe |
| RDF integration | `test_*rdf*.py` | RDataFrame export/import |
| Drawing | `test_draw_*.py` | dfdraw integration |
| Fit validation | `test_register_fit_result.py` | Phase 12 fit framework |
| Invariance | `test_invariance_*.py` | Refactoring safety net |
| Registered functions | `test_register_evaluator.py` | Phase 13.10 evaluator tests |
| Polynomial persistence | `test_polynomial_persistence.py` | Phase 13.9.Fix1 schema roundtrip |
| Draw lazy compound | `test_draw_lazy_compound.py` | Phase 13.11 compound expression parsing |

### 12.6 Capability Matrix Infrastructure (Phase 13.11/13.11.B)

**Location:** `docs/CAPABILITY_MATRIX.md` (auto-generated by `run_tests.sh`)

**Infrastructure files:**
- `run_tests.sh` — automated test runner producing SUMMARY, diffs, CAPABILITY_MATRIX, reviewer.zip
- `scripts/generate_capability_matrix.py` v2 — taxonomy-based matrix generation from pytest JSON
- `tests/feature_taxonomy.py` — 41 curated features across 12 modules (PHASE_13_11_B approved, 3 reviewers)
- `scripts/phase_tag.sh` — PHASE_BEGIN tag management
- `@pytest.mark.invariance` markers on invariance test classes (62+ tests)

**Current Matrix (2026-04-02):**

| Status | Count | Description |
|--------|------:|-------------|
| ✅ Verified | 9 | Has `@pytest.mark.invariance` tests |
| ☑️ Smoke-only | 27 | Tests pass but no invariance verification |
| 🧨 Broken | 4 | Pre-existing failures (I2_6, I4_2/3, RDF, Parquet) |
| 📋 Planned | 1 | SUB.clone (not implemented) |
| **Total** | **41** | 1509 matched tests, 42 unmatched |

**Module breakdown:** CORE (8), SUBFRAMES (6), DRAWING (5), SCHEMA (4), REGISTERED_FUNCTIONS (4), COMPRESSION (3), BACKEND (3), LAZY_LOADING (3), FIT_REGISTRATION (2), RDATAFRAME (2), INVARIANCE (1)

---

## 13. Known Limitations

### 13.1 Current Bugs

| Bug ID | Severity | Description | Workaround | Status |
|--------|----------|-------------|------------|--------|
| BUG_AliasDataFrame_20260116_lazy_subframe_init | P1 | `register_subframe_lazy()` missing `_df` | Use eager `register_subframe()` | Open |
| BUG_AliasDataFrame_20260324_draw_subframe_resolution | — | `draw("Sub.col")` failed | — | ✅ Fixed |
| BUG_AliasDataFrame_20260331_fill_value_dependency | P0 | fill_value skipped in batch dependency chain (7.4% NaN in production) | — | ✅ Fixed (`06d2d611`) |
| BUG_AliasDataFrame_20260401_draw_lazy_compound | P1 | Aliases in abs()/sqrt() not auto-materialized by draw_lazy | materialize before draw | ✅ Fixed (`ef9d263c`) |
| Phase 13.9.Fix1 polynomial persistence | P1 | registered_functions lost on export_tree/read_tree | Manual re-registration | ✅ Fixed (`189b7682` + fix1b) |
| describe_aliases false BROKEN | P2 | Subframe columns reported as missing (cosmetic) | Aliases work when materialized | Open |
| I2_6 | P2 | Chained subframe Numba/NumPy mismatch | Use single backend | Open |
| I4_2, I4_3 | P2 | Scaled compression roundtrip | Use linear/sqrt formulas | Open |
| RDF failures | P2 | 3 RDF friend access failures | Under investigation | Open |

### 13.2 Architectural Limitations

| Limitation | Impact | Resolution |
|-----------|--------|------------|
| Monolithic module (~12,100 lines) | MTTU concern, hard to navigate | Refactoring planned (after invariance coverage ≥80%) |
| Code duplication: _serialize/_deserialize vs export/apply schema | Two parallel JSON/ROOT serialization paths | Technical debt — caused 13.9.Fix1 incident |
| draw_lazy=False default | Silent empty plots for unmaterialized aliases | Decision pending (O2DistAI request: change to True) |
| Left join only | Cannot do inner/outer joins | Sufficient for physics use case |
| Expression eval via pandas.eval | Limited function set | Custom dispatch for np.* functions |
| Evaluator not serialized | Must re-register after load | By design (GBAI handles persistence) |

### 13.3 Design Constraints

| Constraint | Rationale |
|-----------|-----------|
| Python-side join only | Server-side computation; data too large for client |
| Left join semantics | Physics data model: events have tracks, not vice versa |
| pandas.eval for expressions | Proven correctness; custom AST too risky for physics |
| Numba optional | Not all environments have Numba; graceful fallback |
| No `schema=` in constructor | Schema built internally from DataFrame dtypes; enrichment via `update_schema()` |
| Schema stores contract only (evaluators) | Evaluator serialization is GBAI responsibility |
| Multi-predictor requires explicit selection | Silent default on multi-predictor is unsafe |

---

## 14. Planned Work

### 14.1 Immediate

| Task | Status | Priority |
|------|--------|----------|
| draw_lazy=True default decision | Pending (O2DistAI request) | P1 |
| clone_with_selection() method | Pending (O2DistAI request) | P1 |
| GB tuple support for linear_columns | Pending (Gemini3 ruling) | P1 — blocks end-to-end PolynomialSpec |
| Refactor serialize/deserialize duplication | Pending | P1 — caused 13.9.Fix1 incident |
| Fix I2_6, I4_2, I4_3 failures | Pending | P2 |
| .gitignore cleanup | Pending | P2 |
| describe_aliases false BROKEN fix | Pending | P2 |
| Increase invariance coverage (9/41 → 30+) | Pending | P2 — prerequisite for refactoring |

### 14.2 Future Phases

- Phase 14 — ADFStore concept (shared storage layer, no data copies)
- Evaluator persistence (GBAI interface for schema reconstruction)
- PolynomialSpec.project() helper (dx decomposition)
- DFDRAW.md creation
- AliasDataFrame.C documentation
- Phase 10/11: Production ALICE data validation, O2 SOA integration

---

## 15. Quick Reference

### 15.1 Common Patterns

```python
# Load + subframe + alias + draw
adf = AliasDataFrame.read_tree("data.root", "Events")
tracks = AliasDataFrame.read_tree("data.root", "Tracks")
adf.register_subframe("T", tracks, "event_id")
adf.add_alias("pt_ratio", "T.pt / pt")
fig, ax, stats = adf.draw("pt_ratio", selection="eta > 0")

# Evaluator integration (Phase 13.10)
adf.register_evaluator('corr', evaluator, ['xM', 'driftM', 'sector'])
adf.add_alias('dy_I1', 'dy - corr(xM, driftM, sector)')

# Polynomial from subframe (Phase 13.9)
spec = PolynomialSpec(columns=['xM', 'driftM', 'dsecM', 'tgSlp'], degrees=(3, 3, 2, 1))
coeff_cols = [f'T{i}' for i in range(spec.n_terms)]
adf.register_polynomial_from_subframe('poly', spec, 'Coeffs', coeff_cols)
adf.add_alias('dy_poly', 'dy - poly(xM, driftM, dsecM, tgSlp)')
```

### 15.2 Key Imports

```python
from AliasDataFrame import AliasDataFrame
from PolynomialSpec import PolynomialSpec  # Phase 13.9

# RDataFrame integration (only if using ROOT RDataFrame)
from AliasDataFrameRDF import (setup_rdf_with_friends,
                                add_defines_to_rdf)

# export_tree() is a method on AliasDataFrame, not a separate import:
#   adf.export_tree('output.root')
```

### 15.3 Schema V2 Structure

```json
{
  "version": 2,
  "__meta__": {"schema_id": "my_analysis", "created": "..."},
  "columns": {"pt": {"dtype": "float64", "group": "kinematics", "unit": "GeV", "title": "p_{T} (GeV/c)"}},
  "aliases": {"pt_gev": {"expression": "pt / 1000", "fill_value": null}},
  "subframes": {"T": {"tree": "Tracks", "index_columns": ["event_id"]}},
  "registered_functions": {
    "corr": {"type": "evaluator", "coord_columns": ["xM", "driftM"], "predictor_columns": ["dy"]}
  },
  "compression": {"pt": {"formula": "linear", "bits": 12}},
  "fit_metadata": {}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-02-14 | Initial technical summary for architecture review |
| v1.1 | 2026-02-14 | Post-review: fixed constructor signature (P0), reconciled dispatch order (P0), split committed vs in-validation tests (P1), sourced performance numbers (P1), precise join semantics (P1), marked broken API (P1), added Contract Snapshot (P2) |
| v1.2 | 2026-02-19 | **Complete public API** (P0): §4 rewritten with ALL public methods. Added: I/O & Export, Lazy Loading, Metadata, Data Inspection, Dtype Management, Schema Groups, Fit Registration. Origin: ORecoAI integration failure — 3 wasted iterations due to missing API docs. |
| v1.3 | 2026-02-19 | **Source-verified corrections** from 4-reviewer consolidated review (CLAUDE3, GPT, GPT3, source verification against 11,557-line AliasDataFrame.py). P0-1: `export_tree()` signature corrected. P0-2: `get_axis_unit()` removed. P1-1: `__getitem__` auto-materialization claim removed. P1-2: `__contains__` corrected. P1-3: schema key `axisLabel` → `title`. P1-4: `save_schema()` simplified. P2-1: `register_subframe(pre_index)` added. P2-2: `read_tree` → `@staticmethod`. |
| v1.4 | 2026-03-27 | **Phase 13.9/13.10 additions:** NEW §4.15 (register_function, register_polynomial_from_subframe, register_evaluator), NEW §6.5 (Registered Function System with PolynomialSpec architecture, evaluator schema contract). Updated: §1.1 feature list, §1.2 architecture diagram, §1.3 file structure (+PolynomialSpec.py), §1.4 metrics (1425 tests, 42× polynomial speedup), §2 contract snapshot, §4.3 add_alias fill_value, §12 test coverage, §13 bugs (draw_subframe_resolution ✅ fixed), §15.3 schema with registered_functions. |
| v1.5 | 2026-04-06 | **Phase 13.11/13.11.B + bug fixes.** Updated §1.2–1.4 (12,100 lines, 1441 tests, 41-feature matrix). NEW §12.6 Capability Matrix infrastructure (run_tests.sh, taxonomy-based generator, 41 features, 12 modules). Updated §12.2 (5 new test addition entries). Updated §13.1: fill_value P0 ✅, draw_lazy compound ✅, polynomial persistence ✅. Updated §13.2: CAPABILITY_MATRIX ✅, run_tests.sh ✅, PHASE_BEGIN ✅ resolved; serialization duplication added as tech debt. Updated §14 planned work. |
| v1.5.2 | 2026-04-19 | **Phase 13.21.ADF additions.** NEW in §4.3: `dematerialize(drop=, keep=)` — memory reclamation with raw-column protection; replaces removed `drop_materialized()`. Updated §2 Join Contract: join index cache row (content-based validation, survives `materialize_aliases`, invalidated on `register_subframe`). Updated §12.1: 1521 tests, 125 invariance. |
| v1.6 | 2026-05-14 | **Phases 13.22 through 13.27 + bug fixes + FIX1 queue.** Updated §1.4: 1606 tests, 177 invariance, 47-feature matrix (28 ✅ / 14 ☑️ / 4 🧨 / 1 📋). NEW §4 `read_tree` parameters: `dtype_overrides={regex: np.dtype}` (Phase 13.26.ADF, on-the-fly type conversion with overflow detection) and `skip_branches=[regex]` (Phase 13.27.ADF, selective branch exclusion). Updated §13.1: BUG_GroupBy_Expression_Materialization ✅, BUG_validate_aliases_false_positives ✅, BUG_save_load_compression_regression ✅. Updated §14: Phase 13.25.DF FIX1 (active priority, 4 P1s open since 2026-04-30); Phase 14 ADFStore (PyArrow-backed storage motivated by production fragmentation evidence — 60M-row × 14-column dataset at 2.30 GB physical hits BlockManager fragmentation ceiling). NEW §14.3: methodology lessons from BUG_GroupBy review cycle (matrix-history differential mandatory, multi-model panel diversity validated as Anti-Fabrication firewall). |

---

**END OF TECHNICAL SUMMARY**

**Document Version:** 1.6  
**Phase:** 13.27.ADF (`read_tree` skip_branches)  
**Total Source Lines:** ~13,625 (AliasDataFrame.py) + 337 (PolynomialSpec.py) + ~800 (scripts/infrastructure)  
**Total Test Files:** 60+  
**Test Results:** 1606 passed, 7 failed, 1 error, 8 skipped (2026-05-14 baseline; matches PHASE_HISTORY documented state)  
**Capability Matrix:** 47 features — 28 ✅ Verified, 14 ☑️ Smoke-only, 4 🧨 Broken (K2_3 intermittent + 3 pre-existing I2_6/I4_2/I4_3 + 3 RDF.export friend-tree tests), 1 📋 Planned. 177 invariance tests.

---

## v1.6 Addendum: Active Queue + Methodology Notes

### Active phases (priority order)

1. **Phase 13.25.DF FIX1** (dfdraw cross-subproject; spec at `PHASE_13_25_DF_v1.3_Proposal.md`). Proposal `PHASE_13_25_DF_FIX1_v1.0_Proposal.md` drafted 2026-05-14 by Claude37. Closes 4 P1s from Claude40 consolidated code review of 2026-04-30. **15 days open**; two correctness P1s affect production users today (AD-52 silent rebind for `error="sem"`; `error="none"` empty render in `quantile_mode='error_bars'`). Effort: 5–6 hr Coder + 2-day review cycle.

2. **Phase 14 ADFStore** (formal architect-review proposal required). Concept: shared storage layer where AliasDataFrame operates as a view over relational tables with no copies. Motivated by (a) nested subframe export crash, (b) `hadd` file-size problem, (c) **production memory fragmentation evidence** from Phase 13.27.ADF — 60M-row TPC calibration dataset at 2.30 GB physical hits pandas BlockManager fragmentation ceiling that `dtype_overrides` + `skip_branches` cannot escape. PyArrow-backed storage is the structural fix.

### Methodology lessons (from BUG_GroupBy_Expression_Materialization cycle)

This bug-fix cycle (2026-05-12 to 2026-05-13) demonstrated that the Anti-Fabrication firewall works as designed when reviewer panels have model-family diversity:

1. **Internal-consistency-without-external-anchor failure mode.** Initial Claude37 review approved ✅ based on architectural-isolation reasoning (the 12-line `draw()` fix cannot cascade into save/load/compression subsystems) without running the differential against the documented PHASE_HISTORY baseline. The reasoning was correct but incomplete.

2. **Multi-model diversity caught the consensus blind spot.** Sonnet1 + Sonnet2 (different model family, fresh methodology read) independently performed the matrix-history differential and identified 3 newly-broken tests (`test_save_and_load_integrity`, `test_backward_compatibility_no_compression_info`, `test_roundtrip_save_load`). Same-model panels (Claude-only) would have produced consensus on the wrong answer.

3. **Outcome.** Main Reviewer (Claude37) synthesis overrode the initial ✅ to ❌ CHANGES REQUESTED. Regressions resolved during commit `b9c28663` (en passant with B1 fix). Failure count returned to documented 7F+1E baseline.

### Governance recommendations (for next Org-structure / MTTU_Reviewer revision)

| Recommendation | Origin |
|---|---|
| Bug-fix proposal template: mandatory "Baseline test state vs PHASE_HISTORY" field in Evidence Anchor | BUG_GroupBy cycle |
| Reviewer Card Rule 5c: matrix-history differential mechanically required, not judgment-driven | BUG_GroupBy cycle |
| Reviewer Card Rule: "no commit, no review" — if implementation not in `diff_last_commit_*.txt`, verdict is automatically [X] BLOCKED with no further analysis | Phase 13.26.ADF F1 enforcement |
| Anti-Library entry: "Merging features without specification" (analogous to existing "Reasoning about performance without profiling") | Phase 13.26.ADF v1.1 cycle |
| Phase 13.26.ADF v1.0: production datapoint provides quantified Phase 14 motivation (60M-row × 14-column = 2.30 GB physical, ~+1.5–2× RSS overhead from fragmentation) | Production telemetry |
| Coder rotation: Claude48 → Reviewer paired-test (Phase 13.25.DF FIX1 is the natural slot) | Phase 13.25.DF v1.0_END findings |
