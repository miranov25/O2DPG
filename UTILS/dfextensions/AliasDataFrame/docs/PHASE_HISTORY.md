# AliasDataFrame Phase History

> **Purpose**: Development history for architecture reviews and restart prompts.  
> **Last Updated**: 2026-04-12  
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
- [Phase 13: Advanced Features](#phase-13-advanced-features)
- [Phase 9: PyArrow Acceleration](#phase-9-pyarrow-acceleration)
- [Phase 8: Numba Acceleration](#phase-8-numba-acceleration)
- [Phase 7: Lazy Loading](#phase-7-lazy-loading)
- [Phase 6: dfdraw Integration & Drawing Validation](#phase-6-dfdraw-integration--drawing-validation)
- [Phase 5: RDataFrame Integration](#phase-5-rdataframe-integration)
- [Phase 4: Modular RDF API](#phase-4-modular-rdf-api)
- [Phase 3: Benchmark Infrastructure](#phase-3-benchmark-infrastructure)
- [Phase 2: Batch Optimization](#phase-2-batch-optimization)
- [Phase 1: Foundation & Schema](#phase-1-foundation--schema)
- [Bug Fixes](#bug-fixes)
- [Architecture Decisions](#architecture-decisions)
- [Performance Summary](#performance-summary)

---

## Overview

AliasDataFrame is a high-performance data analysis framework for particle physics research at CERN's ALICE experiment. It provides schema-driven, lazy-evaluated columns and hierarchical joins for ROOT/Parquet data.

**Key Metrics:**
- Performance: 60-770x speedups achieved; production pipeline 2× faster (1452s → 722s)
- Test Coverage: 1521 tests passing, 125 invariance tests
- Lines of Code: ~12,800 (AliasDataFrame.py)
- Features: 44 in taxonomy (26 verified, 14 smoke-only, 3 broken, 1 planned)

**Development Team:**
- Coordinator: Marian Ivanov (miranov25)
- Primary Coder: Claude (Anthropic)
- Reviewers: GPT, Gemini, Claude instances

---

## Phase 13: Advanced Features

**Dates**: 2026-03-22 to 2026-03-27  
**Status**: 🔄 In Progress

### Phase 13.12.DF: Profile Enhancements
**Date**: 2026-03-22  
**Commit**: `8e6e5d87d60206ad24b39af5847424d1d5db3c9d`

dfdraw profile() enhancements:
- F1: `return_data=True` exports profile statistics as DataFrame
- F2: `min_entries=3` suppresses low-statistics bins
- F3: `group_by_bins`/`group_by_quantiles` auto-bins float columns
- F4: `sort_groups=True` sorts legend numerically/alphabetically
- v1.1: `weights` parameter for weighted mean/std/sem

**Tests**: 19 tests (16 original + 3 weights)

### Phase 13.9.ADF: PolynomialSpec
**Date**: 2026-03-23  
**Commit**: `eb54d3bc16e216e057da32c50a7a67e9a5dac568`

N-dimensional polynomial specification for calibration workflows:
- `PolynomialSpec.py` (337 lines) — polynomial term generation
- `basis_expressions()` → (key_name, expression) tuples for GB Regression
- `numba_evaluator()` → Numba JIT with subframe coefficient lookup (42× vs eval)
- `to_schema()`/`from_schema()` — JSON serialization
- `to_root_expression()` — C++ expression for TTree::Draw

New methods on AliasDataFrame:
- `register_function(name, func, overwrite=False)` — generic callable registry
- `register_polynomial_from_subframe()` — polynomial alias from subframe coefficients

**Architecture**: Coefficient matrix stays at subframe size (e.g., 36×96 = 27 KB). No materialization to main frame length.

**Tests**: 26 new tests, 1402 total passed

### Phase 13.10.ADF: register_evaluator
**Date**: 2026-03-27  
**Commit**: `66164aadd6dd8216d18e545e1df5dac87ba8a087`

GroupByRegressionEvaluator integration:
- `register_evaluator(name, evaluator, coord_columns, predictor_columns, overwrite)`
- Wraps any object with `.evaluate(positions: dict) → ndarray`
- Enables lazy evaluation via `add_alias('dy_corr', 'corr_I1(xM, driftM, dsectorM)')`
- Multi-predictor: auto-selects single, raises ValueError if multiple without `predictor_columns`
- Schema stores interface contract only — evaluator must be re-registered after load

**Tests**: 24 new tests, 1425 total passed

### Phase 13.12.ADF: Public API Invariance Test Suite
**Dates**: 2026-04-08 to 2026-04-12 (4 calendar days)
**Commits**:
- `037767e8` — Batch 1: I5, I6, I7 (10 tests; schema roundtrip, subframe NaN, draw paths)
- `1c4d6038` — Batch 2: I8, I9, I10 (7 tests; composition, register_function, lazy/eager)
- `bc7fd19e` — Batch 3: I11, I12 (4 tests; linear compression, metadata persistence)
- `d812d871` — Batch 4: I13, I14, I15, I16, I17 (10 tests; backend, join, order, dtype, pipeline)
- `77a8fa22` — Close: feature_taxonomy + Capability Matrix regeneration

**Proposal**: `PHASE_13_12_ADF_v1.2_Proposal.md` (approved 2026-04-08)
**Code Review Request**: `PHASE_13_12_ADF_v1.0_Code_Review_Request.md`

31 invariance tests across 13 new standalone files in `tests/`, all named
`test_I{N}_*_invariance.py` per §5.1 architect waiver, all marked
`@pytest.mark.invariance`. Every test carries a path-explicit docstring
naming its production entry point and source line number (per Failure
Mode #11 discipline).

**Coverage added**:
- I5 (4 tests): JSON + ROOT schema roundtrip preserves aliases and subframes
- I6 (3 tests): subframe-missing-key NaN propagation + fill_value (BUG_20260331 regression guard, production fix `06d2d611`)
- I7 (3 tests): draw() path invariance with draw_lazy (BUG_20260324, BUG_20260401 regression guards)
- I8 (3 tests): subframe + alias composition, composite-key join vs pd.merge reference, auto_alias_subframe
- I9 (2 tests): register_function one-arg and two-arg identity
- I10 (2 tests): read_tree eager vs read_tree_lazy+ensure_branches equivalence
- I11 (2 tests): linear compression roundtrip within bit-budget (working paths only; asinh/scaled still 🧨)
- I12 (2 tests): set_axis_title + set_columns_metadata schema persistence
- I13 (3 tests): numba vs numpy backend equivalence (arithmetic, compound expr, subframe scatter)
- I14 (2 tests): single-key and composite-key join == pd.merge
- I15 (2 tests): batch vs sequential, forward vs reversed materialization order
- I16 (2 tests): explicit dtype pinning + no silent narrowing
- I17 (1 test): end-to-end integration (register → alias → metadata → export_tree → read_tree → materialize)

**Scope deviations** (all flagged upfront in module docstrings, none silent):
- §5.1: standalone-files model accepted in lieu of "extend existing files"
- §4.2 I9_2: two-arg register_function replaces polynomial test (duplicate coverage already in `test_polynomial_persistence.py`)
- §4.3 I11_1: v1.2 shorthand `{formula,bits}` does not exist in source; real API uses explicit compress/decompress expressions

**Tests**: 1463 → 1480 passed (+17 net after skip accounting); same 6 pre-existing failures; same 1 pre-existing xdist parallel-collection artifact; zero regressions; zero xfails filed; zero bug reports filed throughout phase.

**Lessons learned**:
- **Failure Mode #11 (path-explicit discipline)** remains the single highest-value coder rule. Caught one test-writing bug in Batch 1 I5 (fixed via `_build_fresh_adf_with_subframe` helper) and one silent-pass hazard in Batch 2 I8_3 (Claude33 source-line 9305 verification).
- **Scope changes must be flagged upfront, not in commit messages.** Batch 2 I9_2 silent deviation cost two review rounds. From Batch 3 onward, deviations were documented in module docstrings at delivery time.
- **Reviewer package must contain committed code**, not staged. Reviewer-package generation before `git commit` produced an empty `diff_last_commit` in two cases; workflow corrected mid-phase.
- **Independent reference paths catch silent-pass hazards**. Direct fixture-array slicing in I6_1 was replaced with `pd.merge` reference after Claude32 P2 feedback; the pattern was reused in I8_2 and I14 from the start.

**Reviewer performance**: Claude33 consistently produced the highest-signal findings via source-line verification. Claude30 tracked cross-round spec fidelity. Claude32 provided the cleanest per-test verification tables. Main Reviewer Claude1 consolidated each batch and the end-of-phase review.

**Follow-up items** (tracked elsewhere, not blocking phase closure):
- Technical Summary v1.6 full public API documentation (~90 methods) — deferred to separate future phase per Phase 13.11 decision
- `draw_lazy=True` default change — pending architect decision (pre-existing, not a 13.12 deliverable)

---

### Phase 13.18.ADF: Regression Metadata Bridge
**Dates**: 2026-04-12 to 2026-04-13  
**Status**: ✅ Merged  
**Commits**: `2bc65bb`, `2662cee`, `d0d020a`

**Deliverables**:
- 3 new public API methods: `register_regression_metadata()`, `update_regression_metadata()`, `register_evaluator_from_metadata()`
- `describe_regression()` for lazy-status-aware introspection
- Schema persistence for regression_metadata (mirrors registered_functions pattern)
- Natural-label → compact-index remap in `_bridge_eval_func` (cross-team Q&A with GB team)
- feature_taxonomy.py: +3 entries (FUNC.regression_metadata, FUNC.evaluator_from_metadata, FUNC.regression_persistence), 41 → 44 features

**Tests**: 11/11 passing (7 pass + 4 xfail resolved after bridge fix)

**Key decision**: `_bridge_eval_func` uses `register_function` directly (not `register_evaluator` wrapper) because evaluator requires compact-index input, not natural labels. Bridge handles the remap.

**Lesson learned**: Reviewer-discipline failure — `_eval_lookup` docstring says "raw grid indices" but coder read "raw" as "natural values." Recovery required cross-team Q&A. Future: hand-trace one example through the target method before escalating.

---

### Phase 13.19.ADF.FIX1: Vector Draw Kwarg Diagnostic
**Dates**: 2026-04-14 to 2026-04-18  
**Status**: ✅ Merged  
**Commits**: `f9679f7`, `1fa5145b` (combined with 13.20)

**Problem**: `adf.draw('[y1..y6]:staveITS', group_by='mP3', group_by_bins=6)` produced 421 legend entries instead of 6.

**Diagnostic approach** (architect direction: *"Make tests first"*):
- K1 tests (boundary diagnostic): monkey-patch DFDraw methods to capture kwargs at ADF→DFDraw boundary. **Conclusion: ADF forwards kwargs correctly. Bug was dfdraw-internal.**
- K2 tests (end-to-end): count rendered legend entries post-dfdraw FIX1 (`fe007b7c`)

**Tests**: K1 (4 pass, 1 skip) + K2 (4 pass) = 8 permanent regression tests

**Lesson learned**: K1 ruled out wrong hypothesis; K2 (output counting) should have been v0.2 from the start. *"Diagnostic tests must include the user-visible symptom, not just intermediate-layer plumbing."*

---

### Phase 13.20.ADF: export_tree Metadata Batching
**Dates**: 2026-04-18 to 2026-04-19  
**Status**: ✅ Merged  
**Commit**: `1fa5145b` (combined with 13.19.FIX1)

**Problem**: `export_tree` with N subframes opened ROOT file N+1 times for metadata writing. Profile: 159s / 11% of 1452s total.

**Fix**: Separate data-write (uproot) from metadata-write (ROOT). 4 new methods: `_write_all_data_to_uproot`, `_collect_metadata_targets`, `_write_all_metadata_to_root`, `_write_metadata_to_tree`. 1 `TFile.Open` instead of 16+.

**Measured savings**: ~29s (not 80-130s predicted — O2DistAI's single-TF filter reduced file sizes, making each TFile.Open cheaper). Fix is structurally correct; savings scale with file size.

**Tests**: E1 (4 pass) + E2 (3 pass, 1 xfail for pre-existing `read_tree` nested subframe limitation)

---

### Phase 13.21.ADF: Join Index Caching + dematerialize() API
**Dates**: 2026-04-19  
**Status**: ✅ Merged  
**Commits**: `fa7cd11d` (v1.0 caching), `4c269c3c` (v1.1 dematerialize + remove drop_materialized)

**Origin**: PHASE_13_19_ADF_PERF_Summary.md item A1, unanimous ADF team agreement (Claude31, Claude30, Claude32).

**Fix A — Join index caching** (4 changes, +35 lines):
- Root cause: line 4411 cleared ENTIRE `_join_index_cache` after every `materialize_aliases` call, even though only value columns changed
- Added `_index_column_signature()` — O(1) content-based cache validation (first/last/dtype)
- Added targeted cache invalidation in `register_subframe` instead of blanket clear
- Profile evidence: 56s / 141 cache misses → expected ~5-10s / ~15 misses

**Fix B — `dematerialize(drop=, keep=)` API** (+72 lines):
- Three modes: `drop=[...]`, `keep=[...]`, no args (drop all)
- Raw columns always protected. Mutually exclusive drop/keep (`ValueError`)
- Replaced `drop_materialized()` — strict superset, 3 internal call sites updated
- Composes with join caching: re-materialization reuses cached indices

**Tests**: J1_1..J1_10 (correctness + dematerialize) + J2_1 (performance gate) = 11 tests. Updated 2 pre-existing `test_join_index_caching.py` tests to expect new cache-survives behavior.

**Test results**: 1521 passed, 7 failed (all pre-existing), 1 error (pre-existing)

---

## Bug Fixes

### BUG_AliasDataFrame_20260324_draw_subframe_resolution
**Dates**: 2026-03-25  
**Status**: ✅ Fixed

**Problem**: `adf.draw("Side.dy:row")` fails — dfdraw receives unresolved `Side.dy`, pandas eval interprets dots as attribute access.

**Commits**:
- `75609ea2` — draw() fix (5 tests)
- `3c069a55` — draw_batch() fix
- `b4c845c1` — consolidated fix (19 tests, 1392 passed)
- `0f4fd63f` — draw_figures() extension (4 tests, 23 total draw subframe tests)

**Solution**: Detect `Subframe.column` patterns in expr/selection/group_by, materialize via `pd.merge` on index columns only, rename to underscore format (`Side_dy`) for pandas eval safety, rewrite all expressions to match.

**Key insight**: `pd.merge` operates only on index columns, NOT full DataFrame — memory is O(N × index_cols), not O(N × all_cols).

### Production Fixes (2026-03-24 to 2026-03-26)
**Commits**:
- `4c45c4dd`, `abd5d518`, `cef9bfef` — `fill_value` parameter for `add_alias()` (division-by-zero handling)
- `f7d2335c` — `register_polynomial_from_subframe()` accepts `overwrite` parameter
- `f7d2335c` — `_draw_single_figure()` prints errors to console when verbose=True
- `486ff33c`, `6930690d` — `draw_figures()` per-figure defaults cascade (temporary fix)

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

**Finding**: PyArrow compute 8-10× slower than NumPy for element-wise math
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

**Tests**: 19 new, 52 total lazy subframe tests

---

## Phase 6: dfdraw Integration & Drawing Validation

**Dates**: 2025-12-07 to 2025-12-11  
**Status**: ✅ Complete

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
**Commit**: `d5591a07b252fe432afefe68b94ceced68cbfbb11`

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

### Phase 13 Decisions

| Decision | Rationale |
|----------|-----------|
| PolynomialSpec `tgSlp` as standard dimension | Coupling via alias algebra, not special parameter |
| Evaluator schema stores contract only | User must re-register after load; GBAI handles serialization |
| Multi-predictor requires explicit selection | Silent default on multi-predictor is P1 violation |
| pd.merge on index columns only | Memory O(N × index_cols), not O(N × all_cols) |

---

## Performance Summary

### Speedups Achieved

| Phase | Improvement | Method |
|-------|-------------|--------|
| Phase 2 | 10× | Batch materialization |
| Phase 7 | 86% | Join caching |
| Phase 8 | 10.8× | Numba acceleration |
| Phase 5 | 25× | vs TTree::Draw |
| Phase 13.9 | 42× | Numba polynomial evaluator vs eval |
| Phase 13.20 | ~29s saved | export_tree metadata batching (1 TFile.Open vs N+1) |
| Phase 13.21 | ~40-50s expected | Join cache survives materialize_aliases |
| **Production** | **2× (1452→722s)** | **Cross-team: GB + ADF + O2DistAI fixes combined** |

### Memory Savings

| Feature | Savings |
|---------|---------|
| Lazy loading | 90%+ |
| Compression | 50-80% |
| Coefficient matrix (Phase 13.9) | 36×96 = 27 KB vs 13.5M rows |

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
| 6.8 | 43 | 1178+ |
| 13.9.ADF | 26 | 1402 |
| 13.10.ADF | 24 | 1425 |
| BUG draw_subframe | 23 | 1392 (at fix time) |
| BUG fill_value | 7 | 1432 |
| BUG draw_lazy_compound | 13 | 1441 |
| 13.11.ADF | infra | 1441 |
| 13.11.B | taxonomy | 1441 |
| 13.12.ADF | 31 | 1480 |
| 13.18.ADF | 11 | 1491 |
| 13.19.ADF.FIX1 | 8 (K1+K2) | 1499 |
| 13.20.ADF | 8 (E1+E2) | 1510 |
| 13.21.ADF | 11 (J1+J2) | 1521 |

---

## Pending Items

- [x] ~~CAPABILITY_MATRIX.md creation~~ (Phase 13.11)
- [x] ~~PHASE_BEGIN_AliasDataFrame tag~~ (Phase 13.20 close)
- [ ] A2 — LZ4 default compression (one-line + compat test, ~15-20s savings)
- [ ] A3 — Batch metadata serialization (~20-30s savings)
- [ ] `read_tree` recursive subframe loading (line 5172 `load_subframes=False` → `True`)
- [ ] GB tuple support for `linear_columns` (PolynomialSpec production blocker)
- [ ] Technical Summary v1.6 full public API documentation (~90 methods)
- [ ] P1 tests: I2_6, I4_2, I4_3 fixes
- [ ] Fix `register_subframe_lazy()` bug (BUG_AliasDataFrame_20260116)
- [ ] Axis title lookup for subframe columns (`Sub_dy` vs `Side.dy`)

---

*Document generated from git history. For updates, run:*
```bash
git log --oneline --since="2025-11-01" > history.log
```
