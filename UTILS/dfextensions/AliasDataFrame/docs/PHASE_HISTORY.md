# AliasDataFrame Phase History

> **Purpose**: Development history for architecture reviews and restart prompts.  
> **Last Updated**: 2025-12-11  
> **Maintained By**: Marian Ivanov (miranov25)

## How to Use This File

This file is intended for AI reviewers and human collaborators as a **restart context**.

**For new review sessions:**
1. Skim **Overview** + the most recent in-progress phase + **Architecture Decisions**
2. For detailed commit-level context, see `git log` or referenced commit hashes
3. Status values are relative to "Last Updated" date above

**Important**: Phases are grouped by **logical topic**, not strict calendar order. Later-numbered phases (e.g., Phase 9) may precede earlier ones (e.g., Phase 7) in time. See dates in each section for actual chronology.

---

## Table of Contents

- [Overview](#overview)
- [Phase 9: PyArrow Acceleration](#phase-9-pyarrow-acceleration)
- [Phase 8: Numba Acceleration](#phase-8-numba-acceleration)
- [Phase 7: Lazy Loading](#phase-7-lazy-loading)
- [Phase 6: dfdraw Integration & Drawing Validation](#phase-6-dfdraw-integration--drawing-validation)
- [Phase 5: RDataFrame Integration](#phase-5-rdataframe-integration)
- [Phase 4: Modular RDF API](#phase-4-modular-rdf-api)
- [Phase 3: Benchmark Infrastructure](#phase-3-benchmark-infrastructure)
- [Phase 2: Batch Optimization](#phase-2-batch-optimization)
- [Phase 1: Foundation & Schema](#phase-1-foundation--schema)
- [Architecture Decisions](#architecture-decisions)
- [Performance Summary](#performance-summary)

---

## Overview

AliasDataFrame is a high-performance data analysis framework for particle physics research at CERN's ALICE experiment. It provides schema-driven, lazy-evaluated columns and hierarchical joins for ROOT/Parquet data.

**Key Metrics:**
- Performance: 60-770x speedups achieved
- Test Coverage: 1178+ tests passing
- Lines of Code: ~10,000 (AliasDataFrame.py)

**Development Team:**
- Coordinator: Marian Ivanov (miranov25)
- Primary Coder: Claude (Anthropic)
- Reviewers: GPT, Gemini, Claude instances

---

## Phase 9: PyArrow Acceleration

**Dates**: 2025-12-01 to 2025-12-02  
**Status**: ✅ Complete (selective adoption)

### Phase 9a: ArrowComputeMapper Foundation
**Date**: 2025-12-01  
**Commit**: `d72465c6797625c1b3eba98d1c6dc4df82b7f9ef`

- AST-based expression parsing (mirrors convert_expr_to_root)
- 38 function mappings (trig, hyperbolic, exp/log, rounding)
- Expression caching for performance
- Graceful fallback with warnings

### Phase 9b: Arrow Scatter Integration
**Date**: 2025-12-01  
**Commit**: `de9d765b1507a30b11109b63f6ae4a10ddfd3511`

- `_extract_subframe_values_arrow()` using `pc.take()`
- Missing key handling via null masking
- Priority: Arrow → Numba → NumPy fallback
- **Status**: ENABLED (beneficial)

### Phase 9c: Arrow Compute Path
**Date**: 2025-12-01  
**Commit**: `6c44ae6d84c4c37fe1f7d052dc86ff008bd5cb2a`

- Infrastructure in place but disabled
- Per-expression conversion overhead too high
- **Status**: DISABLED

### Phase 9e: Final Decision
**Date**: 2025-12-02  
**Commit**: `a788a687c6d0684fbaa71427e443451dd09445b7`

**Finding**: PyArrow compute 16x slower than NumPy for element-wise math
- NumPy: 2.6ms vs Arrow: 41.2ms
- Root cause: Arrow lacks expression fusion

**Design Decision**: Keep Arrow for I/O + scatter operations, use NumPy for compute. This hybrid approach gets the best of both worlds.

---

## Phase 8: Numba Acceleration

**Dates**: 2025-11-30 to 2025-12-01  
**Status**: ✅ Complete

### Phase 8a: Numba Scatter
**Date**: 2025-12-01  
**Commit**: `adac6b457e552c8ccadb7e2c77e548a779366229`

- JIT-compiled `numba_scatter()` with f32/f64/i64 dispatch
- Parallel execution with prange
- Scatter: 0.032s → 0.004s (8x faster)

### Phase 8b: Numba Index Lookup
**Date**: 2025-12-01  
**Commit**: `adac6b457e552c8ccadb7e2c77e548a779366229`

- JIT-compiled `numba_compute_join_indices()`
- Auto-select: direct addressing vs hash lookup
- Falls back to pandas for multi-column keys

### Phase 8c: Multi-Column Key Linearization
**Date**: 2025-12-01  
**Commit**: `46d2320b8a90b4fe33a480314ddc856236a13bbc`

- Linearize composite keys into single int64
- Use GLOBAL max values from both main and subframe
- Auto fallback for overflow/negative/non-integer keys

**Performance**:
- Phase 7: 0.34s → Phase 8c: 0.228s (1.5x faster)
- Total speedup vs baseline: 10.8x
- Efficiency vs theoretical: 6.8%

---

## Phase 7: Lazy Loading

**Dates**: 2025-12-09 to 2025-12-10  
**Status**: ✅ Complete

### Phase 7.1: Lazy Branch Loading Foundation
**Date**: 2025-12-09  
**Commit**: `60d0e5daf7214ea13edbf6edb3f49c4b7588f15a`

- `read_tree_lazy()` for on-demand branch loading
- `ensure_branches()` to load specific branches
- Auto-load on `adf['x']` access
- Properties: `is_lazy`, `available_branches`, `loaded_branches`
- LazyTreeReader class for ROOT I/O
- **Memory savings**: 90%+ (5-10 of 100+ branches)

**Tests**: 35 new tests

### Phase 7.2: Branch Auto-Detection
**Date**: 2025-12-09  
**Commit**: `0c15c5bbbb8de1a1f9c94bbead853b6f6de71165`

- `get_required_branches()` public API
- AST-based selection string parsing
- Alias chain resolution to base branches
- Regex fallback for unparseable expressions

**Tests**: 51 new tests, 1014 total

### Phase 7.3: Draw Integration
**Date**: 2025-12-09  
**Commit**: `37aed3d5f4a054df87a1fdeea9d9343d7b59567d`

- `draw()` auto-loads required branches
- `draw_batch()` pre-scans specs, batch-loads once
- Handles selection, group_by, color parameters

**Tests**: 22 new tests, 1037 total

### Phase 7.4: Chain Mode
**Date**: 2025-12-10  
**Commit**: `8dca62767978f813056b0c5f746c5df464b43595`

- LazyChainReader with composition pattern
- LRU file handle cache (K=8 default)
- `read_chain()` and `read_chain_lazy()` class methods
- Validation modes: first, strict, intersection, union
- `__file_idx__` column for file provenance
- Custom exception hierarchy

**Architecture**:
- Composition pattern ✅
- LRU cache K=8 ✅
- Single pd.concat with ignore_index=True ✅

**Tests**: 51 new chain tests, 136 total lazy/chain

### Phase 7.5a: Lazy Single-File Subframes
**Date**: 2025-12-10  
**Commits**: `e0dbd99c38f0715117ecd5bd6eb375521f9b268b`, `87be90d10fdd762ff961abda791d30a672ffa98a`

- `register_subframe_lazy()` for single ROOT files
- `ensure_subframe()` for explicit loading
- Automatic loading on alias materialization
- Unification: loaded lazy subframes identical to eager

**Tests**: 33 lazy subframe tests

### Phase 7.5b: Subframe Chains
**Date**: 2025-12-10  
**Commit**: `b394323772951bd4e10c0a44008478b84038f6a7`

- `register_subframe_chain()` for multi-file subframes
- Reuses LazyChainReader from Phase 7.4
- Validation modes: first, strict, intersection, union

**Architecture Decisions** (4-reviewer consensus):
- Add 'type': 'file' to single-file config ✅
- No file index for subframes ✅
- Trust reader protocol ✅
- Schema has both 'index' and 'index_columns' ✅

**Tests**: 19 new, 52 total lazy subframe tests

---

## Phase 6: dfdraw Integration & Drawing Validation

**Dates**: 2025-12-07 to 2025-12-11  
**Status**: 🔄 In Progress (6.8e/c)

> **Note**: The "6.8" label has been extended. Original 6.8 (dfdraw integration: axis titles + slice-first evaluation) is now part of the broader "Phase 6.8: Drawing Validation & Benchmarks" package.

### Phase 6.8 Core: Duck-Typed Axis Titles
**Date**: 2025-12-08  
**Commit**: `314fde98e747306a7903bda0d03eab01cf768dfe`

- `_data_source` storage for duck-typed lookups
- `_get_label()` method for axis title resolution
- Label precedence: explicit > schema > default

**Tests**: 222 passed (+19 new)

### Phase 6.8a: Slice-First Lazy Evaluation
**Date**: 2025-12-08  
**Commit**: `b7d713af0229d08d186b66377d05c95f8e1cba52`

- Entry selection happens BEFORE alias evaluation
- `_eval_alias_on_df()` for arbitrary DataFrame subset
- Performance: 12M rows with 10K mask evaluates on 10K only

### Phase 6.8d: Synthetic Data Generator
**Date**: 2025-12-11

- Extended `generate_synthetic_data.py` with `--chain` and `--subframe-chain` flags
- Created examples/ directory with four documented examples for draw invariance and lazy loading
- Known relationships for testing:
  - `y_derived = 2 * x` (exact, no noise)
  - `gain[sec] = 1.0 + 0.01 * sec`
  - `gain[run,sec] = 1.0 + 0.01*sec + 0.001*(run-1000)`

### Phase 6.8e/c: Draw Invariance Tests (Current)
**Date**: 2025-12-11

- `test_draw_invariance.py`: Core invariants, subframe joins, dtype preservation
- `test_draw_chain_integration.py`: Chain loading modes, lazy subframes, error handling
- Tests validate eager vs lazy equivalence across all loading modes

**draw() API** (from dfdraw):
- Returns: `(fig, ax, stats_dict)` tuple
- `stats_dict` keys: `'n'`, `'mean'`, `'std'`, `'min'`, `'max'`
- Use `lazy=True` or call `materialize_alias()` before draw

### Phase 6.8 Remaining (per spec)
- **6.8a**: Execution-order fixes (if needed from test failures)
- **6.8b**: Per-spec selection (nice-to-have)
- **6.8f**: Performance benchmarks (nice-to-have)

---

## Phase 5: RDataFrame Integration

**Dates**: 2025-12-03 to 2025-12-06  
**Status**: ✅ Complete

### Phase 5.1: Constants Mapping
**Date**: 2025-12-05  
**Commit**: `631258f53826a071e3140a604470cbca355668d4`

- np.pi, numpy.pi, math.pi → M_PI
- np.e, numpy.e, math.e → M_E

### Phase 5.2: Composite Key Infrastructure
**Date**: 2025-12-05  
**Commit**: `fb97c5471926991ddd510fe8d1b1386d0e6b57df`

- `get_composite_key_column_name()`: Standard naming
- `check_dense_overflow()`: Safe overflow detection
- `generate_dense_cpp_expression()`: C++ for RDF Define()

**Tests**: 27 new, 834 total

### Phase 5.3: Runtime Composite Keys
**Date**: 2025-12-06  
**Commit**: `d5591a07b252fe432afefe68b94ceced68cbfbb1`

- Runtime composite key generation for >2 index columns
- TMemFile + SetFile approach
- Main tree: SetFile for `__adf_key__` branch only
- Safety: friend size limit (1M), int64 overflow detection

**Tests**: 861 passing

### Phase 5 (Full): RDataFrame with Benchmarks
**Date**: 2025-12-04  
**Commit**: `5595ec550899685a541e862ed382372799eaec51`

- Sparse key support (np.unique vectorized)
- Auto-selection dense/sparse based on key distribution
- `benchmark_rdf.py`: Compare RDF vs TTree::Draw vs ADF

**Results** (1M rows, 10-level alias chain):
- AliasDataFrame: 0.039s (25x faster than TTree::Draw)
- TTree::Draw: 0.964s (baseline)
- RDataFrame: 1.357s (JIT overhead)

---

## Phase 4: Modular RDF API

**Dates**: 2025-12-04  
**Status**: ✅ Complete

**Commit**: `cf68ba49e9f8f7296ed0afe18cda775199e591ca`

- `setup_rdf_with_friends()` - returns (rdf, file_handle)
- `setup_chain_with_friends()` - TChain for multiple files
- `add_defines_to_rdf(on_collision='warn')` - handles existing columns
- `get_join_columns_for_snapshot()` - index columns helper
- `cache_to_snapshot()` - convenience function
- `export_tree(columns=...)` - snapshot mode

**Tests**: 46 RDF tests passing

---

## Phase 3: Benchmark Infrastructure

**Dates**: 2025-11-27 to 2025-11-30  
**Status**: ✅ Complete

### Benchmark Suite
**Commit**: `a8de5c6bf3b71217da0177132102cba906ad3964`

- `run_benchmark.sh` as main entry point
- `generate_synthetic_data.py` for test files
- Row counts: quick=500K, default=1M, full=2M
- JSON output, memory tracking, baseline comparison

### Roofline Analysis
**Commit**: `12372b429fcfeae4dbc652b4ab320028b3bad418`

3-Level Reference Model:
- L1 Hardware (memcpy): 0.003s (33 GB/s)
- L2 Numba (@njit parallel): 0.004s (16 GB/s)
- L3 NumPy (C backend): 0.015s (4 GB/s)

---

## Phase 2: Batch Optimization

**Dates**: 2025-11-27 to 2025-11-28  
**Status**: ✅ Complete

### BUG-2025-11-27-002: DataFrame Fragmentation
**Commit**: `4312270cccd7ebd8f0b97a726b5f68a29d7072ef`

- Problem: 216s for 58 aliases on 13.5M rows
- Fix: Single pd.concat() instead of sequential insertions
- Result: 216s → <20s (>10× speedup)

### BUG-2025-11-27-003: Fill Handling
**Commit**: `f9df9cf6f380ae78872e0263ca5397714d210594`

- `set_global_fill()` / `set_subframe_fill()` API
- fill_missing, fill_nan, fill_inf, fill_invalid
- fill_mode: 'safe' (default) or 'direct'
- Aggregated warnings (24 → 1 summary)

### Join Caching
**Commit**: `e8ba0c486f6b18b41bb52e588213b2fb65b2ab60`

- Add `sort=False` to merge
- Cache hit rate: 87.5%
- Total: 2.46s → 0.344s (-86%)

---

## Phase 1: Foundation & Schema

**Dates**: 2025-11-15 to 2025-11-26  
**Status**: ✅ Complete

### Schema V2
**Commit**: `62ca3fe691a8141c7e480fd12f29cfe9b47a0f33`

- Column groups for logical organization
- Arbitrary column metadata (unit, axisLabel, description)
- Smart JSON formatting
- Recursive subframe schema export
- v1→v2 backward compatible loading

### Compression Engine
**Commit**: `d5962723860d8a44da173782e3fed948ef054c12`

- on_missing: 'warn', 'error', 'ignore'
- return_summary parameter
- Idempotent compress/decompress operations

### Definition vs Record Schema
**Commit**: `acf6fbc3cd8657a2fe238de4f14fc4044ad91d04`

- Definition schema (blueprint): no state
- Record schema (snapshot): full state
- `export_definition_schema()` / `export_record_schema()`
- `validate_schema()` with modes

---

## Architecture Decisions

### Multi-Reviewer Consensus Process

All major decisions require consensus from 3+ AI reviewers:
- Claude (Architect/Coder)
- GPT (Performance)
- Gemini (C++/ROOT)

### Key Design Principles

1. **Unification**: Lazy→Eager after load (identical behavior)
2. **Composition**: LazyChainReader wraps LazyTreeReader
3. **Fallback Chain**: Arrow → Numba → NumPy → Pandas
4. **Schema-Driven**: All metadata in unified schema
5. **Backward Compatible**: All changes maintain existing API

### Validation Modes (Chains)

| Mode | Behavior |
|------|----------|
| first | Use first file as reference |
| strict | All files must match exactly |
| intersection | Common branches only |
| union | All branches, fill missing |

---

## Performance Summary

### Speedups Achieved

| Phase | Improvement | Method |
|-------|-------------|--------|
| Phase 2 | 10× | Batch materialization |
| Phase 7 | 86% | Join caching |
| Phase 8 | 10.8× | Numba acceleration |
| Phase 5 | 25× | vs TTree::Draw |

### Memory Savings

| Feature | Savings |
|---------|---------|
| Lazy loading | 90%+ |
| Compression | 50-80% |

### Current Efficiency

- vs L3 (NumPy): 6.8%
- vs L2 (Numba): 1.8%
- vs L1 (memcpy): 1.4%

Remaining overhead is Python/Pandas framework cost.

---

## Appendix: Test Counts by Phase

| Phase | New Tests | Total |
|-------|-----------|-------|
| 7.1 | 35 | 963 |
| 7.2 | 51 | 1014 |
| 7.3 | 22 | 1037 |
| 7.4 | 51 | 136 (lazy/chain) |
| 7.5a | 33 | 33 (lazy subframe) |
| 7.5b | 19 | 52 (lazy subframe) |
| 6.8e/c | 43 | 1178+ |

---

*Document generated from git history. For updates, run:*
```bash
git log --oneline --since="2025-11-01" > history.log
```
