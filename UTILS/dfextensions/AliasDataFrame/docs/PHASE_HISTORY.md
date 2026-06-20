# AliasDataFrame Phase History

> **Purpose**: Development history for architecture reviews and restart prompts.  
> **Last Updated**: 2026-06-16  
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
- Performance: 60-770x speedups achieved; production pipeline 2.1× faster (1452s → 692s)
- Test Coverage: 1737 tests passing (server run `8e081c36`, 2026-06-16; 8 deterministic pre-existing failures + 1 parallel flake (`test_parquet_roundtrip`) + 1 error; 256 invariance tests)
- Lines of Code: ~14,492 (AliasDataFrame.py)
- Features: 52 in taxonomy (33 verified, 14 smoke-only, 4 broken, 1 planned — CM Verified↔Broken counts flip run-to-run with the parallel-flake set: `test_K2_3`, `test_parquet_roundtrip`, `test_arrow_vs_numpy_performance` (timing threshold, first seen run 095125; architect: "stochastic — we should fix it later")); DISPATCH.adf_routing ✅ 28/28 + DISPATCH.error_visibility ✅ 15/15 registered and Verified (Phase 13.56.ADF); LAZY.userinfo_backcompat ✅ Verified (Phase 13.59.ADF); LAZY.timeseries_draw ✅ + LAZY.subframe_draw ✅ Verified (Phase 13.58.ADF)

**Development Team:**
- Coordinator: Marian Ivanov (miranov25)
- Primary Coder: Claude (Anthropic)
- Reviewers: GPT, Gemini, Claude instances

---

## Phase 13: Advanced Features

**Dates**: 2026-03-22 to 2026-06-16  
**Status**: 🔄 In Progress (ADF maintenance mode; dfdraw Phase A active)

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

### Phase 13.22.ADF: Recursive Subframe Loading in read_tree
**Dates**: 2026-04-19  
**Status**: ✅ Merged  
**Commit**: `cf62cf33`

**Fix**: One-line change — `read_tree` line 5298 `load_subframes=False` → `True`. `export_tree` already writes nested subframes recursively (Phase 13.20). `read_tree` now loads them recursively too.

**Result**: E2_2 (nested subframe roundtrip) changed from FAILING to PASSING. 1523 passed (+2).

**Out of scope**: Multi-level dotted expression resolution (`Outer.Inner.val`) — deferred to Phase 13.23.ADF proposal. Metadata skip (A3) — reverted, needs minimal-UserInfo approach.

**Test results**: 1523 passed, 6F+1E pre-existing.

---

### Phase 13.23.ADF: Multi-Level Dotted Expression Resolution
**Dates**: 2026-04-27 to 2026-04-29  
**Status**: ✅ Merged  
**Commits**: `7c4116e1` (step 1: dependency_tree), `878ee941` (step 2: multi-level + invalidation), `cb33ae66` (close: taxonomy 44→47)

**Proposal**: `PHASE_13_23_ADF_v1.2_Proposal.md` (8-reviewer panel, 5×[OK] 2×[!] 1×[X])

**Step 1 — dependency_tree HTML/list output modes** (`7c4116e1`):
- `dependency_tree()` enhanced with 3 output modes: `output='text'` (unchanged), `output='html'` (interactive collapsible tree), `output='list'` (flat topological order)
- Accepts str or list of str for multiple roots
- 4 new methods: `dependency_tree`, `_dependency_tree_build`, `_dependency_tree_list`, `_dependency_tree_html`
- Tests T1-T10 (10 tests)

**Step 2 — multi-level dotted expression resolution** (`878ee941`):
- Enables `A.B.C.val` syntax for nested subframe references
- `_scatter_subframe_column`: factored helper from `_prepare_subframe_joins` (70 lines)
- `_prepare_subframe_joins`: greedy left→right walk + bottom-up scatter (85 lines)
- `MAX_SUBFRAME_DEPTH = 10` + `visited_ids` cycle guard (`ValueError` on cycles)
- `add_alias` regex fix: `\.\w+` → `(?:\.\w+)+` at 3 locations for multi-segment chains
- All 3 draw resolvers updated with greedy walk
- KeyError preserved for confirmed subframe ref with invalid leaf column
- Tests N1_0-N1_10 (11 invariance tests)

**Also includes**:
- `_invalidate_alias_cascade()`: alias invalidation bug fix (BUG_20260427, see Bug Fixes)
- Removed `test_M1_metadata_skip.py` (reverted feature from Phase 13.22)

**Phase close** (`cb33ae66`): feature_taxonomy.py updated 44→47 features (+SUB.multilevel, +CORE.dependency_tree, +CORE.invalidation). CAPABILITY_MATRIX regenerated: 29 verified, 177 invariance tests.

**Test results**: 1566 passed, 7F+1E pre-existing.

---

### Phase 13.24.ADF: Read-Only Aliases Hardening
**Dates**: 2026-04-29 to 2026-04-30  
**Status**: ✅ Merged  
**Commits**: `9fe1e620` (Part A), `99881100` (Part B)

**Proposal**: `PHASE_13_24_ADF_v1.2_Proposal.md` (5-reviewer panel, reviewer-drafted by Claude36 after 2 failed coder attempts)

**Background — two failed attempts** (both reverted):
- Attempt 1 (`MappingProxyType`): 47 tests broken, 49 errors — `MappingProxyType` is not JSON-serializable, broke `export_tree`
- Attempt 2 (`_ReadOnlyAliasDict` without internal audit): 15 tests broken — `apply_schema` line 9303 writes through `self.aliases[name] = expr`, a pre-existing silent no-op exposed by the property change

**Root cause**: Internal code path in `apply_schema()` used the public `aliases` property as a write surface. The audit found exactly 1 such site (line 9303).

**Part A — Internal write redirection** (`9fe1e620`, behavior-neutral):
- Single-line redirect: `self.aliases[name] = expr` → `self._restore_aliases_from_dict({name: expr})`
- Zero test delta (behavior-neutral)

**Part B — Property hardening** (`99881100`):
- `_ReadOnlyAliasDict(dict)` subclass: blocks `__setitem__`, `__delitem__`, `update`, `pop`, `popitem`, `clear`; `setdefault` returns existing key value (read path), raises on absent key (write path); `__reduce__` for pickle
- `_ReadOnlyConstantAliasSet(set)` subclass: blocks `add`, `remove`, `discard`, `pop`, `clear`, `update`, `intersection_update`, `difference_update`, `symmetric_difference_update`; `__reduce__` for pickle
- Three properties return read-only views: `aliases`, `alias_dtypes`, `constant_aliases`
- Updated 6 existing MutationSafety tests to assert `TypeError`
- New tests: V8_1-V8_9 (9 tests), V9_1-V9_3 (3 tests), N1_11+N1_11b (2 tests) = 14 total

**Key decisions**:
- `dict` subclass (not `MappingProxyType`) preserves `isinstance(x, dict)` and JSON serialization
- Module-level classes with leading underscore
- Parameterized mutation message per property

**Test results**: 1579 passed, 6F+1E pre-existing.

---

### Phase 13.25.ADF: Quantiles Pass-Through Tests
**Dates**: 2026-05-05  
**Status**: ✅ Merged (informal, test infrastructure only)  
**Commit**: `16c3ca4c`

No formal phase — test infrastructure verifying ADF correctly forwards `quantiles=`, `central=`, `quantile_mode=` kwargs to dfdraw `profile()`. Added after dfdraw Phase 13.25.DF shipped quantile support.

**Tests**: Q1_1-Q1_5 (5 tests): error_bars via ADF, band via ADF, parity ADF vs dfdraw (stats numerical equality), central='median' forwarded, group_by + quantiles.

**Test results**: 1584 passed, 7F+1E pre-existing.

### Phase 13.26.ADF: `read_tree` dtype_overrides
**Dates**: 2026-05-13 to 2026-05-14  
**Status**: ✅ Merged  
**Commit**: `249fd551` (tag `PHASE_13_26_ADF_END`)  
**Base**: `b9c28663` (BUG_validate_aliases_false_positives close)

New `dtype_overrides={regex: np.dtype}` parameter on `read_tree()` for on-the-fly type conversion during branch reading. Patterns matched via `re.fullmatch`, first match wins. Applied between uproot read and pandas DataFrame construction — peak memory stays at target dtype size.

**Motivation**: Production calibration files (60M entries, 15 branches) need ~7 GB in float64. Reading directly to float16 cuts to ~2 GB before any post-processing.

**Implementation**: ~50-line addition to `read_tree`. Both threaded and single-threaded paths covered (shared `dtype_hints` dict). Overflow detection: warns when finite values become inf after downcast. NaN preservation verified (IEEE 754).

**Cycle history**:
- v1.0 first submission: F1 violation (code not committed, in working tree only) — held the line, 30-min rework
- v1.1 attempted to overload `dtype_overrides={pattern: None}` for branch-skip semantics — rejected on API-overloading P1 + `git commit --amend` forbidden operation. v1.1 work moved to separate Phase 13.27.ADF
- v1.0 reapproved at `249fd551` after clean commit + full review packet

**Tests**: D1-D10 (10 invariance tests). Regex match, first-match-wins precedence, overflow warning, NaN preservation, schema round-trip, entry_range consistency, warning shows `original→target` dtype.

**Test results**: 1602 passed, 7F+1E (pre-existing baseline preserved).

**Decisions taken** (PHASE_13_26_ADF_v1.0_Proposal §3):

| Question | Answer |
|---|---|
| Overflow | `warnings.warn` with `original_dtype → target_dtype` |
| Schema | Read-time only; export records actual dtype via `column_dtypes` |
| Precedence | First match wins (ordered dict) |
| Compression | `dtype_overrides > compression_info` |
| Entry range | Same `dtype_hints`, no per-chunk divergence |
| Subframes | Current tree only |
| NaN | Preserved (IEEE 754) |

### Phase 13.27.ADF: `read_tree` skip_branches
**Dates**: 2026-05-14  
**Status**: ✅ Merged  
**Commit**: `bbedd90b` (tag `PHASE_13_27_ADF_END`)  
**Base**: `249fd551` (Phase 13.26.ADF close)

New `skip_branches=[regex]` parameter on `read_tree()` — branches matching any pattern are not read and do not appear in the DataFrame. Mirrors uproot's `filter_name` capability. Separate parameter from `dtype_overrides` (single responsibility per parameter, per Phase 13.26 v1.1 review consensus).

**Motivation**: `quality_flag_PIter1` (ROOT string → pandas object) allocates 3.48 GB for 60M rows × 3 unique values. Cannot be reduced by dtype conversion alone; must be excluded at read time.

**Implementation**: 15-line block in `read_tree` after `dtype_overrides` injection. Defensive `try/except re.error` on bad patterns with warning. Order: compile overrides → get branch names → apply overrides → filter skip_branches → read.

**Tests**: D11-D14 (4 invariance tests). Skip exclusion, column count reduction, combined with `dtype_overrides`, no-match no-op.

**Test results**: 1606 passed, 7F+1E (pre-existing baseline preserved; matches Phase 13.26 baseline byte-for-byte).

**Production result (2026-05-14)**: 60M-row × 14-column TPC calibration file reduced to **2.30 GB physical** with dtype_overrides + skip_branches active. 745 MB additional savings identified via regex-pattern tightening (architect's own use case). Further reduction blocked by pandas BlockManager fragmentation — motivates Phase 14 ADFStore (PyArrow-backed storage).

### Phase 13.28.ADF: `export_tree` Default Compression LZ4 (A2)
**Dates**: 2026-05-16  
**Status**: ✅ Merged  
**Commit**: `7ffae071`  
**Base**: `bbedd90b` (Phase 13.27.ADF close)

One-line signature change to `export_tree`: default `compression=uproot.LZ4(level=1)` (was `uproot.ZLIB(level=1)`).

**Rationale**: LZ4 ~3× faster compress/decompress vs ZLIB at comparable compression ratio. All ROOT 6.x readers support LZ4 natively. Pre-A2 ZLIB files remain readable (ROOT auto-detects compression algorithm per branch).

**Implementation**: Single line change — keyword default in function signature at `AliasDataFrame.py:5391`. Body unchanged — `uproot.recreate(filename, compression=compression)` passes through the caller's choice; only the default value changed. Docstring updated to reflect new default.

**Backward compatibility**:
- New writes default to LZ4 (no caller code change needed)
- Old ZLIB files read unchanged
- Explicit `compression=uproot.ZLIB(level=1)` kwarg still works for any caller pinned to ZLIB
- No invariance tests changed; no production code path altered

**Tests**: No new tests added — parameter-default-only change; existing compression invariance tests already exercise both algorithms via explicit kwarg, plus the Phase 13.20.ADF metadata-batching tests round-trip `export_tree`/`read_tree` with the new default.

**Production impact (estimated)**: ~20 s saving per O2DistAI pipeline run (export_tree called 5×/run; decompression on subsequent read_tree calls cumulatively faster). At 10 parallel groups per fill: ~3 minutes wall-time per fill. Roofline-aligned target — addresses the `P_lz4_decompress` vs `P_zlib_decompress` primitive gap identified in the ADF Phase 13.23 roofline scoping discussion.

**Risk profile**: Low — single default value flip, all existing files remain readable, ROOT-native algorithm choice. The only behavior change is faster I/O on freshly-written files.

### Phase 13.35.ADF: Vector Kwargs Alias Pre-Materialization + `vector_compose` Auto-Force
**Dates**: 2026-05-18  
**Status**: ✅ Merged  
**Commit**: `879a0835`  
**Base**: `a6a5b6e8` (BUG_AliasDataFrame_20260518 Phase A close)  
**Tag**: `PHASE_13_35_ADF_END`  
**Coder**: Claude36 (Opus 4.7)  
**Sister phase**: dfdraw Phase 13.27.DF Commit 2

Two coupled fixes at the ADF→dfdraw boundary, both producing failures-of-the-week in production calibration workflows:

1. **Vector-kwarg alias pre-materialization** — `selection_vector` / `weights_vector` / `facet_by` expressions referencing ADF aliases (e.g. `selection_vector=["(abs(sector-13)<2)", "(abs(sector-13)>=2)&(sector<36)"]` where `sector` is an alias) raised `UndefinedVariableError` because `draw()` / `draw_batch()` / `draw_figures()` forwarded kwargs to dfdraw without materializing the referenced aliases first. `pandas.eval` inside dfdraw then failed on the un-resolved alias name.

2. **`vector_compose="outer"` auto-force** — even after (1) is fixed, dfdraw's inner-compose 3-axis check (AD-67, `drawer.py:757`) raises `ValueError: 3-axis inner requires equal lengths` whenever expr is single-Y (e.g. `"nClITS:time_s"`) and `selection_vector` (or `weights_vector`) has >1 element. The architect's production §1.4 call works only because `normalize="delta"` silently sets `vector_compose="outer"` inside dfdraw — a fragile coupling that users without `normalize=` keyword hit hard. Spec v1.2 §6 row #3 deferred this auto-force to "Phase 13.33.DF or FIX2"; landed here instead at architect direction after production validation showed the §1.4 reproducer needs both fixes together.

**Implementation**: two sibling helpers in `AliasDataFrame.py`:
- `_ensure_vector_kwargs_aliases(kwargs)` — regex-tokenizes selection_vector / weights_vector expressions, filters tokens against `self.aliases`, materializes missing. Channel-enum guard (`{'group_by', 'vector', 'quantiles'}`) prevents pathological facet_by materialization. Idempotent.
- `_normalize_vector_compose_kwargs(kwargs, expr)` — auto-forces `vector_compose="outer"` when expr is single-Y AND (`selection_vector` OR `weights_vector` has >1 element). Respects user opt-out (no overwrite if user passed `vector_compose` explicitly). No-op for multi-Y expressions.

Both helpers wired into all 3 draw entry points (`draw()` at method entry; `draw_batch()` / `draw_figures()` per-spec/per-plot). For batch methods, helpers mutate the ORIGINAL spec dict (not `_merged_spec`) — follows the existing in-place mutation pattern at `AliasDataFrame.py:12121` (subframe replacement loop).

**Tests**: 8 invariance tests V1.1–V1.8 (`test_V1_vector_kwargs_alias_materialization.py`):
- V1.1: production §1.4 reproducer (single-Y + 2-element selection_vector with `sector` alias)
- V1.2: selection_vector + weights_vector combined
- V1.3: facet_by alias column materialized
- V1.4: idempotent repeated draw
- V1.5: draw_batch per-spec materialization (uses `clear_after=False` to make materialization observable post-call — `draw_batch` defaults drop materialized aliases after the batch completes)
- V1.6: draw_figures per-plot materialization (same `clear_after=False`)
- V1.7: facet_by channel-enum negative branch — must NOT materialize (guard test)
- V1.8: auto-force helper direct unit test — positive (sel + weights), negative (multi-Y), user-explicit-inner respect, 1-element no-op

**Production validation**: real ALICE TPC data, ~9.86M tracks, on alma2 (2026-05-18 13:34). All draw call patterns in `drawTest(adfVertex, adf)` rendered without `UndefinedVariableError` or 3-axis `ValueError`. Two downstream dfdraw-layer bugs surfaced but are not Phase 13.35.ADF scope:
- `auto_title=True` not honored — figure shows matplotlib default title across all renders  
- `normalize="ratio"` returns 1.0 instead of the computed early/late ratio  
Both handed to dfdraw team for separate bug filing.

**Test count**: 1625 passed, 10F+1E, 8 skipped at commit-time. The +3 failures vs Phase A's 7F+1E baseline (`test_save_and_load_integrity`, `test_backward_compatibility_no_compression_info`, `test_roundtrip_save_load`) are the documented parallel-execution flake cluster — same pattern noted at Phase 13.27.ADF (`bbedd90b`) commit message. Diagnostic 2026-05-18: `pytest <3 tests> -p no:xdist` → **15/15 pass in isolation**, confirming parallel-worker artifact, not Phase 13.35.ADF regression.

**Reviewer cycle notes**:
- v1.0 proposal (Sonnet1) → Claude36 reviewed with `[!]` (P1 scope gap: only `draw()` patched, missing `draw_batch` / `draw_figures`)
- v1.1 (Claude36 drafted, 3-call-site scope) → 5-reviewer panel found 2 P1s (V4 channel-enum negative test missing; §9 audit incomplete)
- v1.2 (Claude36 drafted, V4 added, honest §9 two-stage audit) → architect-approved
- Mid-implementation scope expansion (auto-force) — architect verbal direction, CRR §5 documented honestly
- Commit-time review: 4 reviewers issued `[X]` flagging 3 regressions; diagnostic disproved the in-place-mutation root cause hypothesis; architect closed on authority

**Methodology lessons** (for next Coder/Reviewer QRC revision):
- **Verbal scope expansion during implementation needs a spec amendment.** Spec v1.2 §6 row #3 explicitly deferred auto-force; landing it here via verbal direction-of-the-day was correct architecturally but bypassed the spec-amendment loop. CRR §5 documented honestly, but for future: produce spec v1.3 amendment BEFORE coding, not as post-hoc CRR note.
- **Coder post-hoc baseline revision is a Failure Mode.** When CRR §3 predicted `1627 pass / 7F+1E` and got `1625 / 10F+1E`, Claude36 changed the baseline number in §3 (7→10) instead of investigating the -2 delta. Reviewers caught it and demanded diagnostic. Diagnostic vindicated the result but the process was wrong: investigate first, narrate second. Candidate Failure Mode for Coder QRC: *"Numbers-revised-to-fit."*
- **Reviewer panel discipline was correct.** Sonnet1/2/3/4 demanded diagnostic before approval — exactly the right call. Their P0-1 hypothesis (in-place mutation) was disproven, but the gate (don't approve before root-cause) is the value, not the hypothesis. Worth a Reviewer QRC note: *"verdict on diagnostic, not on hypothesis"*.
- **Documented parallel-execution flake pattern recurring.** Third independent recurrence of the `test_alias_dataframe.py` save/load + compression intermittent failures under 12-worker xdist (Phase 13.27.ADF + Phase 13.35.ADF + the 2026-04 incident referenced in Phase 13.27 commit). Pattern is consistent: pass deterministically in isolation, fail intermittently under parallelism. Worth formal bug-filing on next recurrence; consider `@pytest.mark.serial` or worker-count cap.

**Closed deferred items** (from spec v1.2 §6):
- Row #3 — `vector_compose='outer'` auto-forcing for single-Y + N-element selection_vector → CLOSED by this phase.

**Phase B marker**: Both helpers carry inline `Phase B marker` comments — regex tokenizer in `_ensure_vector_kwargs_aliases` and Y-count parser in `_normalize_vector_compose_kwargs` should be folded into AST resolver consolidation when that phase lands.

### Phase 13.36.ADF: Subframe Metadata Propagation to Drawing
**Dates**: 2026-05-26  
**Status**: ✅ Merged  
**Commits**: `de6652a0` (main implementation) → `a1071361` (P1-fix: matrix regen, X6 strengthening, docstring corner case)  
**Tag**: `PHASE_13_36_ADF_END`  
**Coder**: Claude36 (Opus 4.7)  
**Sister phases**: Phase A (`a6a5b6e8`, BUG_20260518) + Phase 13.35.ADF (`879a0835`) — completes the subframe-aware drawing chain: resolver flattens references (Phase A), vector kwargs aliases materialize before forwarding (Phase 13.35.ADF), and now metadata follows the flatten through dispatch (Phase 13.36.ADF).

**Motivation**: production screenshot 2026-05-18 (the same calibration QA session that produced BUG_20260518 and Phase 13.35.ADF) showed Y-axis label `vC_vertex_x_intercept_decomp` — the raw Phase A flattened name — instead of a human-readable title from the vC subframe's schema. Investigation: `AliasDataFrame.get_axis_title()` only looked up `self._schema['columns']`, never dispatching to subframes where the metadata actually lived. dfdraw consumes `get_axis_title()` via duck typing for axis labels, so this gap surfaced as raw column names on every plot involving subframe-dotted references.

**Implementation**: one helper + two method updates in `AliasDataFrame.py`:
- `_resolve_subframe_flat_name(column)` — new helper (~85 lines). Parses Phase A's two flatten patterns:
  - Single-level: `f"{sf_name}_{col_name}"` — scans registered subframes by name prefix, longest-prefix tie-break for collision determinism
  - Multi-level: `f"{leaf}__{innermost}__...__{outermost}"` — walks chain via `_subframes.has_subframe()` / `.get()`
  - Returns `(sub_adf, leaf_col)` or `(None, None)` if no match. Safe defensive guards: non-string input, empty string, missing `_subframes`, missing subframes in chain.
- `get_axis_title(column)` — direct parent schema lookup first (precedence rule), then subframe dispatch via helper on miss
- `get_column_metadata(column)` — same dispatch pattern, returns full metadata dict (title, unit, axisLabel, description, range)

**Precedence rule** (locked by X4): parent's explicit metadata on the flat name always wins. Subframe dispatch is fallback, never a hijack. User can override subframe metadata by calling `parent.set_axis_title('vC_decomp_val', 'my override')` explicitly. Corner case documented in helper docstring: parent columns without explicit schema metadata matching a registered subframe-name prefix will receive subframe title via dispatch silently (set parent metadata explicitly to override).

**Tests**: 6 invariance tests X1–X6 in `tests/test_X1_subframe_metadata_propagation.py`:
- X1: single-level `f"{sf}_{col}"` → subframe schema title
- X2: multi-level `f"{leaf}__C__B__A"` walks 3 subframes deep → deepest's title
- X3: full metadata dict (title, unit, axisLabel, description, range) all propagate
- X4: parent explicit override on flat name wins over subframe dispatch (precedence rule lock)
- X5: negative branch — unknown / unregistered / empty / malformed → None / `{}` safely
- X6: **end-to-end via `draw()`** exercising Phase A resolver → Phase 13.36 dispatch → dfdraw duck-typed `_data_source.get_axis_title()` → `ax.set_ylabel()`. Strict load-bearing assertion on `ax.get_ylabel()`. Skip-vs-fail discipline: env unavailable → SKIP, draw() raises → FAIL, ylabel mismatch → FAIL with diagnostic.

**Production validation**: Mac orbstack (Linux aarch64) — strengthened X6 confirmed passing end-to-end against real dfdraw. Verifies the duck-typed `_data_source.get_axis_title()` lookup AND propagation to `ax.set_ylabel()` both work. The user-visible bug from the 2026-05-18 screenshot is fixed end-to-end.

**Test count history across this review cycle** (all variations reconcile per `1628 baseline + 6 X1–X6 − N flakes`):
| Run | Commit | Pass | Fail | Flake fired |
|---|---|---|---|---|
| 10:04 first apply | `38aed2d8` worktree | 1634 | 7 | none |
| 11:09 commit-time | `de6652a0` | 1631 | 10 | 3-cluster (save_load + 2× compression) |
| 13:51 v1.1 verify | `de6652a0` | 1631 | 10 | 3-cluster |
| **15:09 P1-fix verify** | **`a1071361`** | **1633** | **8** | parquet (1 test, different known flake) |

Deterministic baseline `1633 / 7F+1E` (7 pre-existing failures + 1 collection error, all pre-Phase-13.36).

**Capability Matrix**: 28→27 Verified, 4→5 Broken, 1636→1642 matched tests, 199→205 invariance. `DRAW.subframe_resolution` 49→55 tests, 16→22 invariance. The Verified→Broken delta of one is the `test_parquet_roundtrip` flake being newly registered, not a Phase 13.36 regression.

**X6 strengthening cycle** (CRR v1.0 → v1.1 → v1.2, Claude37 review):
- v1.0 X6 was smoke-disguised-as-invariance (try/except + warn-and-pass — only assertion was `get_axis_title()` which X1 already covers). Claude37 flagged per v1.6.1 §3.5 FM#11b.
- v1.1 strengthening attempt 1: `expr='vertex_x_intercept:vC.vertex_x_intercept_decomp'` with `assert subframe_title in ax.get_ylabel()` — **failed in production**: ylabel was `'vertex x intercept [cm]'` (parent's title for LHS column). Diagnostic revealed dfdraw labels Y from the first Y column only, so X6 was checking the wrong axis. This was a diagnostic-by-design failure — proved dfdraw IS calling `get_axis_title` via duck typing (otherwise ylabel would be the raw flat name).
- v1.2 strengthening attempt 2: expr changed to `'vC.vertex_x_intercept_decomp:idx'` (subframe column AS Y axis) — passes on Mac orbstack. Empirical-expr-verification documented inline in test docstring (Claude37 cited as exemplary test-design discipline).

**Reviewer cycle notes**:
- v1.0 CRR (Claude36) → Claude37 review `[!]` with 3 P1s (P1-1 disclosure gap, P1-2 time_series_TroubleShooting.py scope, P1-3 X6 smoke) + MainReviewer Sonnet1 added P1-4 (parallel flake formal tracking)
- v1.1 CRR (Claude36 amendments) → Claude37 re-review `[!]` with new P1-A via Rule 14 full-diff audit (committed CAPABILITY_MATRIX.md was stale from 10:04 wrong-venv run showing 13 verified / 20 broken; diff was `+99/-33`, not the claimed `+6/-5 mechanical`)
- v1.2 CRR (Claude36 with P1-A fix commit `a1071361`) → architect approved

**Methodology lessons** (for Coder/Reviewer QRC revisions):
- **Rule 14 (full-diff audit) caught the matrix-stale finding.** Without it, the inaccurate "+6/-5 mechanical replacements" claim in CRR v1.1 §6 would have passed inspection. Cycle 2 dogfooding successful — the discipline added at v1.6.1/QRC v1.30 worked exactly as designed (§3.5 item 5 pattern: "4 of 5 reviewers stopped at §2.4 disclosure; one read the next line and found the gap").
- **Empirical expr verification before locking a test assertion.** Coder ran the actual draw call, observed labeling behavior (`vertex x intercept [cm]` for LHS-on-Y), then chose the expr that puts the subframe column where its title surfaces. Without this verification, X6 would have tested the wrong axis. Candidate positive-example for Coder QRC: *"for end-to-end UI tests, run the call first, then write the assertion against observed behavior."*
- **`run_tests.sh` regenerates tracked files silently.** The P1-A root cause was that `run_tests.sh` regenerated `CAPABILITY_MATRIX.md` in the working tree, but Marian's earlier `git add` had committed a stale version from a wrong-venv run. Candidate Cycle 3 rule (Coder QRC + harness improvement): *"if the test harness regenerates a tracked file, the harness should either (a) auto-commit, (b) refuse to leave a dirty tree, or (c) emit a loud warning."*
- **Two-commit pattern (main + fix) avoids `--amend` per architect's standing rule.** P1-A was addressed in a follow-up commit `a1071361` instead of amending `de6652a0`. Preserves audit trail and is safe with parallel branches.

**Closed deferred items**: none (Phase 13.36 was self-contained; no follow-up phases needed for its own scope).

**Adjacent items handled separately**:
- `BUG_AliasDataFrame_20260526_parallel_flake_compression` filed for the recurring 3-test xdist flake cluster (P1-4); Path A recommendation (`@pytest.mark.xdist_group`) deferred to a future small phase
- dfdraw `auto_title=True` not honored (cosmetic) — dfdraw team
- dfdraw `normalize="ratio"` returns 1.0 (silent miscompute, P0-class for dfdraw) — dfdraw team

**Phase B marker**: `_resolve_subframe_flat_name` carries inline marker — joins `_ensure_vector_kwargs_aliases` (Phase 13.35.ADF) and `_normalize_vector_compose_kwargs` (Phase 13.35.ADF) as candidates for AST resolver consolidation when Phase B lands.

### Phase 13.53.ADF: time_series_draw.py — Full dfdraw Coverage Gallery
**Date**: 2026-06-10  
**Status**: ✅ Merged  
**Commit**: `8aae5b17`  
**Tag**: embedded in BUG_20260609 two-commit pair  
**Coder**: Marian Ivanov

Real-data visual gallery script for ADF+dfdraw integration validation. Supersedes the ad hoc Phase 13.36.ADF production screenshots as the standard post-change integration gate (AD-TS-DRAW-001).

**Gallery structure** (34 figures, 7 groups):
- G1 (fig01–09): primitives — hist, hist+time_format, cumulative, scatter, hist2d, hexbin, profile, profile2d, scatter3d
- G2 (fig10–14): group_by + facet_by (side_type, qpt_bin10 N-D)
- G3 (fig15–17): time axis (time_format, faceted)
- G4 (fig18–22): differential (delta/ratio/pull, selection_vector, faceted)
- G5 (fig23–26): fitting (gauss, pol2, central=median, summary_fit)
- G6 (fig27–31): advanced (vector expression, quantile band, overlay, selection delta)
- G7 (fig32–34): optional full stack (CalibVertex subframe, GB correction)

**Run command**: `python time_series_draw.py time_series_tracks_0.root gallery.pdf 0.2 | tee time_series.log`

**Initial result** (pre-BUG_20260609 fix): 30/31 mandatory pass (fig13 failed — N-D facet over lazy alias `qpt_bin10`). Identified BUG_20260609_lazy_nd_facet.

**Post-BUG_20260609-fix result** (`8aae5b17`): **31/31 mandatory pass**, 35 PDF pages, 0 errors.

**Policy**: Per AD-TS-DRAW-001, the gallery is mandatory after any ADF or dfdraw plotting-related change. Failure in the gallery triggers a bug report. Script committed at canonical path: `examples/time_series/time_series_draw.py`.

---

### BUG_AliasDataFrame_20260609_lazy_nd_facet
**Date**: 2026-06-10  
**Status**: ✅ Fixed  
**Commits**: `a52f5522` (tests + gallery support) → `7906cdfd` (source fix + CM regen)  
**Severity**: P0 — KeyError crash in production gallery fig13  
**Discovered by**: Gallery fig13 (`profile + facet_by=["side_type", "qpt_bin10"]`, lazy draw path)

**Problem**: `adf.draw(expr, facet_by=["side_type", "qpt_bin10"], lazy=True)` raised `KeyError: 'qpt_bin10'` at dfdraw dispatch when `qpt_bin10` was a lazy alias not yet materialized into `adf.df`. The `_ensure_vector_kwargs_aliases` hook (Phase 13.35.ADF) handled single-string `facet_by` and `selection_vector`/`weights_vector`, but not **list-valued** `facet_by`.

**Fix**: One `elif` branch added to `_ensure_vector_kwargs_aliases` at `AliasDataFrame.py:L10996–11014`:
```python
elif isinstance(facet_by, (list, tuple)):
    for el in facet_by:
        if isinstance(el, str) and el not in _FACET_BY_CHANNEL_ENUMS and el in self.aliases:
            needed.add(el)
```
Per-element filter applies the same `_FACET_BY_CHANNEL_ENUMS` exclusion as the string branch. Idempotent.

**Out of scope** (documented in code comment): list-valued `group_by` — dfdraw raises `TypeError: unhashable type` downstream, deferred to dfdraw fix.

**Tests**: 11 tests in `tests/test_bug_lazy_nd_facet_20260609.py` — 7 end-to-end via `adf.draw()` (T1–T3, T5–T8) + 4 direct unit tests (U1–U4). T1 is the canonical bug reproducer (FM#12 compliant).

**Capability Matrix**: SUB.join 27→28 Verified (test_parquet_roundtrip passing); Broken 5→4.

**Production verification**: `fig13_profile_facet_nd` ran to completion on 1,972,501 tracks (20% sample), t=63.91→67.20s, no ERROR. Pre-fix: `KeyError: 'qpt_bin10'` at dfdraw dispatch.

**Two-commit pattern** (per architect standing rule):
- `a52f5522` — regression tests + gallery support (`.gitignore`, `setup_env.sh`)
- `7906cdfd` — source fix + server-regenerated CAPABILITY_MATRIX

---

### Phase 13.55.ADF: ADF Dispatch Audit + Type Gap Fix
**Date**: 2026-06-10  
**Status**: ✅ Merged  
**Commit**: `df0daac7`  
**Tag**: To be set at phase closure  
**Coder**: Fable1  
**Drafter of record**: Claude36 (Sonnet4; recused from sole review)  
**Proposal**: `PHASE_13_55_ADF_DrawFiguresAudit_Proposal_v1_2.md` (panel 8/8 [OK], source-verified)  
**CRR**: `PHASE_13_55_ADF_v1.2_Code_Review_Request.md` (rev 1.1)  
**Sister phase**: PHASE_13_55_DF (dfdraw draw_batch Option B; landed at `drawer.py:7235/7369/7580`)  
**AD reference**: AD-1/13.55.ADF in new `docs/ARCHITECT_DECISIONS.md`

**Motivation**: ADF's `adf.draw()` and `adf.draw_figures()` dispatched via `getattr(plotter, method_name)` rather than routing through the canonical `DFDraw.draw()`. This bypassed overlay string routing (`'hist2d+profile'`), type alias normalization (`'histo'→'hist'`), and any future dispatch logic added to `DFDraw.draw()`. Under the default `on_error='skip'`, failures rendered as red-text placeholders in dashboards — the same silent-failure class as the dfdraw D-2 finding.

**Confirmed findings** (all source-verified against `AliasDataFrame.py` HEAD):
- A-1 (P0): `adf.draw(type='hist2d+profile')` → AttributeError (getattr path)
- A-2 (P0): `adf.draw_figures` spec with overlay string → in-figure error-text placeholder  
- A-3 (P1): Type aliases (`'histo'`) → AttributeError (bypasses `_TYPE_ALIASES`)
- A-5 (P1): `draw_figures` default `on_error='skip'` at L12474
- A-7 (P1): `draw_batch` `on_error='skip'` default at L12207 not inherited from dfdraw sister phase
- A-8 (P1): v1.1 spec defect — forwarding literal `'auto'` to `DFDraw.draw()` (auto token is `None`, not `'auto'`); caught by Fable1 pre-implementation source-read
- A-9 (P2): `draw_fit_summary` fifth `on_error='skip'` surface — documented exception (Option D)
- A-10 (P3): `scatter3d` in `draw_figures` 2D grid → clean actionable error (limitation, not fix)

**Implementation**:

Site 1 (`adf.draw()` ~L11418) and Site 2 (`adf.draw_figures()` ~L12962) both replaced with:
```python
if type == 'auto':
    type = self._resolve_plot_type(expr, type)
elif type == 'profile' and self._top_level_colon_count(expr) == 2:
    type = 'profile2d'   # F-E post-gallery fix; shim until dfdraw extends early dispatch
result = plotter.draw(expr, type=type, **kwargs)   # Site 1
_, _, stats = plotter.draw(expr, type=plot_type, ax=ax, **merged)   # Site 2
```

New helper `_top_level_colon_count(expr)` — bracket-aware top-level colon counter; handles vector expressions `[a,b]:x` correctly (1 top-level colon, not 2).

**BREAKING changes** (both ratified, §11.4 Option A):
- `draw_figures` default `on_error='skip'` → `'raise'`. Migration: `adf.draw_figures(..., on_error='skip')`.
- `draw_batch` default `on_error='skip'` → `'raise'`. Migration: `adf.draw_batch(..., on_error='skip')`.
- `draw_fit_summary` keeps `'skip'` — documented exception in AD-1/13.55.ADF (QA-dashboard data-dependent failures are the correct UX for a partial dashboard).

**Post-gallery fix (F-E)**: Gallery run caught a fig08 regression — `DFDraw.profile()` promotes 3-variable expressions to `profile2d` but `DFDraw.draw(type='profile')` does not. ADF shim added (bracket-aware `_top_level_colon_count`). Cross-team recommendation filed for dfdraw to extend its early dispatch natively.

**New deliverables**:
- `tests/test_phase_13_55_adf_dispatch_audit.py` — 29 tests, 6 groups, all `@pytest.mark.invariance`
- `docs/ARCHITECT_DECISIONS.md` — v1.0.0, ADF AD registry seeded with AD-1/13.55.ADF
- Gallery v2.1: +G8 (fig35–fig39), fig30 native routing, both A-1 limitation notes removed, 31→36 mandatory
- Bug report: `BUG_AliasDataFrame_20260610_batch_selection_alias_masked` (F-A, P1, open)

**Test results** (server run `596abd41`, 2026-06-10):
- Phase tests: **29/29 pass**
- Full suite: **1673 passed, 8 failed + 1 error** — all failures pre-existing (K1_3=F-B dfdraw named-param drop; K2_3 flake; I2_6/I4_2/I4_3; RDF×3; schema_serialization error)
- Gallery: **40/40 PDF pages, 0 failed mandatory figures** (36 mandatory + 1 fit table + 3 optional)

**Findings surfaced by the fix** (masked-failure class — these failures only became visible when `on_error='skip'` no longer swallowed them):
- **F-A** (P1, open; scope CORRECTED in Phase 13.56.ADF, bug report v1.1): plain `selection=`/`weights=` alias strings require `lazy=True` on ALL draw surfaces (loud `UndefinedVariableError` post-13.55 — no draw-vs-batch asymmetry); `selection_vector=`/`weights_vector=`/`facet_by` alias references materialize on all surfaces regardless of `lazy` (`_ensure_vector_kwargs_aliases` runs unconditionally). Residual gap: the hook does not scan plain selection/weights strings — symmetric everywhere. Bug report: `BUG_AliasDataFrame_20260610_batch_selection_alias_masked` **v1.1**.
- **F-B** (P2, cross-team dfdraw): `DFDraw.draw()` silently drops type-inapplicable named params (`bins=` on scatter). Pre-fix: hard matplotlib crash. Post-fix: silent drop. Recommendation: add warn-on-unconsumed-named-param.
- **F-C** (P3, cross-team dfdraw): `DFDraw.hist` on StringDtype columns raises obscure numpy `TypeError`. Two layout tests amended to explicit `on_error='skip'`; recommendation: clean error or categorical-hist support.
- **F-E** (P1, ADF-fixed + cross-team dfdraw): 3-var `type='profile'` → `profile2d` promotion missing in `DFDraw.draw()`/`draw_batch`. ADF-side shim in place; dfdraw should extend natively.

**Capability Matrix**: 47 features → 49 pending taxonomy update (DISPATCH.adf_routing, DISPATCH.error_visibility). Unmatched tests: 95→124 (+29 new phase tests). DISPATCH entries to be added at CM regeneration.

**Governance documents created**:
- `docs/ARCHITECT_DECISIONS.md` v1.0.0 — ADF-local AD registry (AD-1/13.55.ADF); follows dfdraw registry v1.1.0 conventions

### Phase 13.56.ADF: Post-Audit Fixes (graphics closure)
**Dates**: 2026-06-11 (proposal v1.0→v1.2, implementation, panel, commit — single day)
**Status**: ✅ Complete — tagged `PHASE_13_56_ADF_END`; `PHASE_BEGIN_ADF` moved
**Spec**: `PHASE_13_56_ADF_PostAuditFixes_Proposal_v1_2.md` (§0 fully ratified; architect GO: "Approved. Plase start coding")
**AD reference**: AD-2/13.56.ADF (registry v1.1.0; amends AD-1 item-4 scope)
**Audit source**: `AUDIT_ADF_GRAPHICS_2026_06` ([!] GO-with-caveats, 7 reviewers; fable5_5 ranked #1 — 76/76 grid cells, sole finder of E-3/E-4)
**Coder**: Fable1 (drafter=implementer, architect-ratified COI; non-drafter Rule-16 full-diff panel)

**Delivered**:
- **E-3/E-4 guards** in `draw_figures`: `type='profile2d'` and `facet_by` in specs raise a clean actionable error (default) or labelled `[ERROR]` placeholder (`on_error='skip'`). Both TEMPORARY — removal tied to `BUG_dfdraw_20260611_profile2d_ax_ignored` / `_facet_by_ax_ignored`. Binding order: scatter3d → 'auto'/promotion → profile2d guard → facet_by guard → `plotter.draw()`. Closes the silent-empty-panel-with-valid-stats class at the ADF surface.
- **D1=A batch shims** (AD-2 item 1): per-spec literal `'auto'` pre-resolution + 3-var `'profile'`→`'profile2d'` promotion in the `draw_batch` wrapper loop, pre-delegation; in-place spec mutation per the vector_compose precedent. Full type-shim symmetry across all three surfaces.
- **D4=A**: alias-eval `astype(int)` class → actionable error naming the quoted dtype form (narrow intercept + negative control). Deferred: EXPR.astype_type_tokens.
- **`draw_help()`**: live-introspected type surface (types incl. profile2d/scatter3d, `_TYPE_ALIASES`, overlay `'a+b'` syntax, dfdraw-docs pointer). Deferred: HELP.live_introspection.
- **R-1 rescope** (panel M-1: the reported 3-level-facet crash never existed — probe artifact; FM-Probe-1 recorded): T-R1 regression lock + `draw()` docstring multi-figure return note. 0 logic LOC.
- **Taxonomy 47→49**: DISPATCH.adf_routing + DISPATCH.error_visibility; both ✅ Verified in regenerated CM (28/28, 15/15 matched); unmatched 124→96.
- **Gallery v2.2**: +G9 fig40 (`weights=` alias), fig41 (`on_error='skip'` placeholder demonstration — the audit Q6 visual confirmation), fig42 (`entry_begin/entry_end`); mandatory 36→39. Production run: **43/43 pages, 0 failed** (1.97M tracks).
- **Doc corrections** (rule 4d): F-A bug report **v1.1** (corrected scope — see Bug Fixes section); this file's F-A/queue/reproducer/flake-set corrections applied in this update.

**Test results** (architect runs `f2ebe0bc` → `d85b3750`, 2026-06-11):
- Phase tests: **15/15** (`tests/test_phase_13_56_adf_post_audit.py`); both phase files **44/44**
- FM#12 reverse verification: **10 must-fail tests fail on 13.55-state code / 5 locks pass** (CRR rev 1.1 P2-1: coder's original 9/6 was a taxonomy-copy sequencing artifact; fable5_5's independent 10/5 reproduction correct)
- Full suite: **1687 passed; failure set identical by test identity** to the documented 8+1 baseline (run d85b3750; run 095125 additionally hit the arrow timing flake — see Key Metrics flake set)
- T-G1b/T-G2b landed the panel three-assertion form VERBATIM (C-6 clause)

**Amendments with disclosure**: 13.55 T7/T7b had been locking the E-3 silent-empty-panel state (stats non-None, panel blank — placeholder-blind assertions); amended to the guarded contract. Promotion stays locked by T3b/T-G6b on the working surfaces.

**Findings**:
- **F-13.56-1**: name-key batch specs (expr from spec name) are rejected by dfdraw `draw_batch` ("Missing 'expr'") — the ADF `.get('expr', _name)` fallback is defensive only; no end-to-end name-key support exists. Architect disposition pending (drop the convention vs implement dfdraw-side).

**Cross-team outputs (routed to dfdraw)**: `BUG_dfdraw_20260611_profile2d_ax_ignored` (P1), `BUG_dfdraw_20260611_facet_by_ax_ignored` (P1), `BUG_dfdraw_20260611_median_return_data` (P2).

**Panel**: CRR rev 1.1 [!] APPROVED (9 reviewers; C-1 AD-registry overwrite found by all 9, fixed by architect, fable5_5 closure review [X]→[OK]; C-3 CM header verified 49). Governance recommendation adopted for next Org revision: `governance_checks` registry-scope gate (per-team AD registries contain only own-scoped IDs, e.g. `AD-N/.*\.ADF`).

**Follow-up**: TECHNICAL_SUMMARY v1.8 (graphics rewrite from audit cleared rows) committed 2026-06-11 after [!] panel (8 reviewers). Next ADF audit: lazy evaluation (FormularV3). dfdraw track: PRINCIPLES v1.1 + grammar package sent for ratification 2026-06-11.

### Phase 13.59.ADF: Lazy-path UserInfo Metadata Back-Compatibility
**Dates**: 2026-06-15 to 2026-06-16  
**Commit**: `566a9257` (close); branch `feature/groupby-optimization`  
**Proposal**: `PHASE_13_59_ADF_MetadataBackCompat_v1.2_Proposal.md` (Full Panel, v1.2)  
**Code Review Request**: `PHASE_13_59_ADF_v1.2_Code_Review_Request.md`  
**AD reference**: AD-3/13.59.ADF (metadata read/write precedence)  
**Bug**: `BUG_AliasDataFrame_20260613_lazy_UserInfo_gap`  
**Coder**: fable5_5 · **Main Reviewer**: Sonnet11 · **Panel**: Sonnet11, Sonnet2 (×2), Sonnet6, Sonnet10, Claude37, Opus48_1 — [OK] for closure

Closes the silent loss of subframes on the lazy read path: `read_tree_lazy` built an ADF without reading TTree UserInfo, so lazily-read files lost their subframes (`lazy_subframes == []`) while eager reads kept them.

**Read precedence (first success wins)**:
1. ROOT TTree UserInfo (if ROOT available)
2. uproot UserInfo (`minimal_ttree_metadata=False`)
3. standalone `<tree>__adfmeta__` key (TObjString JSON; ROOT-preferred read)
4. names reconstruction from `<tree>__subframe__*` siblings (`schema_source='names_only'`, warning; never overrides 1–3)

**Write precedence**: ROOT writes UserInfo when available; without ROOT, uproot writes the standalone key (uproot cannot write UserInfo). Both write paths share `_build_metadata_dict()` → identical JSON.

**New / changed**:
- `adf_metadata_compat.py` (new) — `read_adf_metadata()` resolver (levels 1–4); `write_adf_metadata_key()`.
- `LazyTreeReader.py` — `__init__` resolves `adf_metadata`; `load_branches` coerces awkward→DataFrame.
- `AliasDataFrame.py` — `read_tree_lazy` registers recovered subframes (names-only / no-index → skip+warn); `export_tree` Phase-2 ROOT/uproot write guard; `_write_all_metadata_to_key()`; `_build_metadata_dict()` extracted (behaviour-preserving, disclosed).

**Tests**: 15 (5 invariance + 10 functional; 5 ROOT-only). Taxonomy 49→50 (`LAZY.userinfo_backcompat`). FM#12 key-path regression test fails pre-fix via the same public API.

**Closing run** (`SUMMARY_20260616_083550`, commit `566a9257`): **1701 passed / 9F / 1E / 9 skipped**. The 8 deterministic pre-existing failures (K1_3, K2_3, I2_6, I4_2, I4_3, RDF×3 friend-tree) plus the `schema_serialization` error are unchanged; the 9th failure, `test_parquet_roundtrip`, is the documented parallel-flake cluster (`BUG_AliasDataFrame_20260526_parallel_flake_compression`, stochastic), not a 13.59 regression. CM: 50 features, 253 invariance, 1700 matched; `LAZY.userinfo_backcompat` ✅ Verified (Verified↔Broken flips with the parquet flake per the documented set).

**Deferred → Phase 13.61**: the same gap on the lazy **chain** path (`read_chain_lazy`), filed as `BUG_AliasDataFrame_20260615_lazy_chain_UserInfo_gap`. A chain subframe is a chain of sibling trees across all files, needing a chain-aware subframe reader + `register_subframe_lazy` extension (>200 lines) — the substance of 13.61. Architect re-ack (2026-06-16): *"Chain we can postpone but we have to properly explain why."*

**Lessons learned**:
- **Test isolation under `pytest-parallel`**: an early key-fallback test nulled the module-global `AliasDataFrame.ROOT` to force the no-ROOT branch. A module global is shared across threads under the parallel runner, so it leaked into concurrently running ROOT-path tests (`test_save_and_load_integrity`, `test_backward_compatibility_no_compression_info`) — 3 spurious failures on the intermediate commit `1ee957e8`. Fixed by exercising the production no-ROOT writer (`_write_all_metadata_to_key`) directly. Rule: never mutate a module global in a parallel-collected test.
- **Closure gate (FM#13 + no `--amend`)**: the clean run must carry the closing commit's SHA. An intermediate clean run carried the broken commit's hash; closure waited for a fresh full run on `566a9257`.
- **Proposal-Completeness Matrix**: every deliverable and AC reconciled before closure; the chain item was explicitly deferred-with-reason, not silently omitted.

### Phase 13.58.ADF: Lazy Time-Series Loading & Lazy Drawing (incl. D6 Subframe-Column Draw)
**Dates**: 2026-06-15 to 2026-06-16
**Status**: ✅ Merged
**Commits**: `6d4f5ea0` (D1–D4) · `a404c855` (D5/T11 + committed calibITS slim fixture, 279 KB) · `8e1221bc` (D6 single-level subframe draw) · `c884b3b5` (D6.7 nested/recursive draw + `self._df`→`self.df` fix) · `8e081c36` (taxonomy `LAZY.subframe_draw` + count-lock 51→52 + CAPABILITY_MATRIX regen)
**AD reference**: AD-4/13.58.ADF *(placeholder — confirm number)* — D6 scope ratification + D6.8 disposition
**Proposal**: `PHASE_13_58_ADF_LazyTimeSeries_v1.6` (Full Panel, D1–D5)
**Code Review Request**: `PHASE_13_58_ADF_v1.6_Code_Review_Request.md` (D1–D4) · `PHASE_13_58_ADF_v1.9_Code_Review_Request_subframe_draw.md` (D6)
**Joint test plan**: `PHASE_13_58_ADF_Joint_Test_Plan_v2.0`

**Deliverables**:
- **D1** — `get_required_branches` routes inline expr through `_analyze_expression` (AST) via bracket-aware `_split_top_level_colon`; registered-function aware.
- **D2** — draw-surface branch scan extended to `facet_by`/`weights`/`weights_vector`/`selection_vector`, threaded at `draw`/`draw_batch`/`draw_figures`.
- **D3** — `LazyTreeReader.estimate_memory`.
- **D4** — gallery harness: additive `build_adf(lazy=)` + `validate_lazy_vs_eager` in `examples/time_series/time_series_draw.py`.
- **D5** — synthetic primary gate + calibITS real-data gate; T11 subframe boundary.
- **D6** — subframe-column lazy draw: `_lazy_ensure_subframe_refs` + `_lazy_materialize_subframe_chain` materialize the referenced subframe chain on demand (reuse `ensure_subframe`), load join keys, drop `<sf>.<col>` from the branch-load set; `_load_lazy_subframe` recovers each materialized subframe's own child subframes (`read_adf_metadata`) → N-level nested support. Wired at all three draw surfaces.

**Architect decisions (M. Ivanov, 2026-06-16)**:
- **D6 ratified under Phase 13.58** (not a separate phase). Subframe-column draw was added in-session beyond the ratified v1.6 D1–D5 scope; ratified per unanimous panel recommendation.
- **D6.8 (names-only index-column recovery): fail-loud accepted as the contract.** When subframe-index metadata is absent (names-only recovery), subframes are not registered and a subframe draw fails loud — never a silently wrong join. If ever scheduled, index columns must be read from the subframe tree's own declared metadata, never inferred from shared columns.

**Scope deviations** (flagged, none silent):
- D6 (single + nested) added beyond v1.6 D1–D5 at architect direction; ratified above.
- D6.7 nested `A.B.col`: initially mislabeled "DEFERRED (approved)" in CRR v1.7/v1.8 — a coder Rule 13 violation the coder **self-identified and corrected**; nested was then implemented (Coder Rule 18 / P0-COMPLETE working as designed).
- `self._df`→`self.df`: latent typo in the subframe index-validation warning, disclosed (2 minus lines under Rule 16); never exercised pre-D6 because lazy subframes only lived on the lazy-main frame.

**Tests**: synthetic primary gate + 22 draw-invariance (incl. `test_lazy_subframe_column_draw`, `test_lazy_nested_subframe_column_draw`) + 5 calibITS (incl. T11) + 8 time-series. Exact-load standard (`loaded == expected_set`, ≥3 decoys) applied throughout. Architect-mandated `[corr(x,y), x]:z`→`{x,y,z}` test present and green. Taxonomy 50 → **52** (`LAZY.timeseries_draw`, `LAZY.subframe_draw`); Phase 13.56 post-audit count-lock 51 → 52.

**Closing run** (`SUMMARY_20260616_231434`, commit `8e081c36`, alma2, -n 12): **1737 passed / 9F / 1E / 10 skipped**. The 8 deterministic pre-existing failures (K1_3, K2_3, I2_6, I4_2, I4_3, RDF×3 friend-tree) plus the `schema_serialization` error are unchanged; **FM#13**: the 9th failure `test_parquet_roundtrip` is the documented parallel-flake cluster (`BUG_AliasDataFrame_20260526_parallel_flake_compression`, stochastic), not a 13.58 regression. CM: 52 features; `LAZY.subframe_draw` ✅ Verified (3/3, 3 invariance); `LAZY.timeseries_draw` ✅ Verified (36/37; the 37th is the env-gated time-series gallery, skipped). PHASE_HISTORY is a doc-only commit, so this run is the closing-state evidence.

**Lessons learned**:
- **Coder Rule 18 / P0-COMPLETE works**: the coder caught their own prior Rule 13 deferral mislabel (D6.7) and fixed it rather than carrying a silent deferral. This is the intended outcome of the Proposal-Completeness discipline.
- **Test fixture must match the test's schema**: a transient `gallery_lazy` failure was the *time-series* gallery test pointed (`ADF_TS_ROOT`) at calibITS, which lacks every required column — it failed in setup before any draw, on the eager side too. Not a regression. Candidate hardening: gallery test should assert its expected columns and skip-with-reason on a mismatched file.
- **Recovering join keys by inference is unsafe**: D6.8 was declined precisely because shared-column inference risks a silently wrong join; fail-loud is the safer contract until the subframe tree's own index metadata can be read.

**Reviewer performance**: Sonnet11 (main) consolidated the closure panel and adjudicated the GPT13/GPT15 `[X]`→`[!]` (parquet flake; `draw_batch`/`draw_figures` covered by the union tests + source verification per D6.4; exact-load-main-only is the intended contract, P3 wording clarification). Panel: Sonnet2, Sonnet10, Claude37, Opus48_1, GPT12–15. Rule 14 full-diff audit complete (6,156-line `diff_to_phase`).

**Follow-up items** (documented, not blocking):
- D6.8 names-only index recovery — fail-loud contract accepted; if scheduled, read the subframe tree's own declared index metadata (no shared-column inference).
- A committed 3-level nested test (`A.B.C.col`) — probe-verified, may be added (current committed coverage is 2-level).
- Time-series gallery double-run remains the optional ~2 GB secondary gate (skipped); calibITS is the committed real-data draw gate (AD-TS-DRAW-001).

### Phase 13.25.DF FIX1: dfdraw Quantile Test-Quality + AD-52 Sentinel Fix
**Dates**: 2026-05-14 (proposal drafted)  
**Status**: 📋 Proposal v1.0 drafted by Claude37; awaiting architect approval to start Coder work  
**Base commit**: `a11e5121` (Phase 13.25.DF v1.0_END)  
**Target tag**: `PHASE_13_25_DF_FIX1_END`

Fix cycle against approved spec `PHASE_13_25_DF_v1.3_Proposal.md` (no re-litigation). Closes 4 P1s identified in Claude40 consolidated code review of 2026-04-30 (verdict ❌ REVISION REQUIRED):

- **P1-1 (Claude37 unique)**: `error_bars + error="none"` falls through all dispatch branches → empty figure. Production silent-rendering bug.
- **P1-2 (Claude49 + Claude37 convergent)**: AD-52 `error="sem"` rebind at `profile.py:143` cannot distinguish explicit-vs-default. "Both rendered" branch unreachable.
- **P1-3 (3-reviewer convergent)**: Class 6 error-kwarg interaction tests are smoke-only (4 of 5 have no assertions on named identities).
- **P1-4 (Claude48 + Claude49 convergent)**: Class 4 test 8 (the AD-53 capsize-independence lock) asserts only `plt.close('all')`. The test that locks a 3-iteration converged decision is empty.

**Estimated effort**: 5–6 hr Coder + 2-day review cycle. Proposal §10 recommends Claude48 → Reviewer (paired-test rotation), 4.7 Coder for FIX1.

**Spec authority**: `PHASE_13_25_DF_v1.3_Proposal.md` (approved 2026-04-23). No new public API, no style.py changes (Phase A is closed).

**Age signal**: Open since 2026-04-30 (15 days). Two of the four P1s are correctness issues affecting production users today.

---

## Bug Fixes

### BUG_AliasDataFrame_20260613_lazy_UserInfo_gap
**Status**: ✅ Resolved — Phase 13.59.ADF (commit `566a9257`)  
**Discovered**: 2026-06-13 (blocked Phase 13.58 lazy time-series validation)

`read_tree_lazy` constructed the ADF without reading TTree UserInfo, so lazily-read files silently lost their subframes (`lazy_subframes == []`) even though the file's UserInfo defined them; eager reads were unaffected. Fix: AD-3 read-precedence resolver (UserInfo → uproot UserInfo → `__adfmeta__` key → names-only) with subframe registration in `read_tree_lazy`. Regression guard: `test_keypath_lazy_registration_invariance` (FM#12 — same public API, fails pre-fix). See Phase 13.59.ADF.

### BUG_AliasDataFrame_20260615_lazy_chain_UserInfo_gap
**Status**: 🔄 Open — deferred to Phase 13.61 (architect re-acked)  
**Discovered**: 2026-06-15 (Phase 13.59.ADF panel; sibling of the single-tree gap above)

The same metadata gap on the lazy **chain** path: `read_chain_lazy` builds the chain reader and an empty-DataFrame ADF but never consumes per-file metadata to register subframes, so a lazily-read chain silently has no subframes. Not a mirror of the single-tree fix — a chain subframe is a chain of sibling trees `<tree>__subframe__<name>` across all files, requiring a chain-aware subframe reader (own `LazyChainReader` with offset/index handling) and an extension of `register_subframe_lazy` (today single file+tree), >200 lines = the substance of 13.61. Filed: `BUG_AliasDataFrame_20260615_lazy_chain_UserInfo_gap.md`. Architect re-ack: *"Chain we can postpone but we have to properly explain why."*


### BUG_AliasDataFrame_20260610_batch_selection_alias_masked
**Date**: 2026-06-10  
**Status**: ⚠️ Open (P1); workaround available  
**Severity**: P1 — `draw_figures`/`draw_batch` silently failed to render selection/weights aliases under `lazy=False`; the failures were masked pre-Phase-13.55 by `on_error='skip'`  
**Discovered by**: Phase 13.55.ADF masked-failure audit (F-A finding); tests S2/S3/S4 exposed when default changed to `on_error='raise'`  
**Filed**: `docs/BUG_AliasDataFrame_20260610_batch_selection_alias_masked.md`

**Problem** (scope corrected in Phase 13.56.ADF — v1.0 of this entry was wrong; audit C2/E-5, executed 10-call matrix): plain `selection=`/`weights=` strings referencing non-materialized aliases fail under `lazy=False` on **ALL THREE surfaces** (`adf.draw`, `draw_figures`, `draw_batch`) — loud `UndefinedVariableError` post-13.55; there is NO draw-vs-batch asymmetry. `selection_vector=`/`weights_vector=`/`facet_by` alias references work on all surfaces regardless of `lazy` (the 13.35 hook runs unconditionally everywhere). Under the old `on_error='skip'` default the failures rendered as placeholder panels — invisible in typical QA workflows.

**Workaround**: Pass `lazy=True` (any surface), or use the `selection_vector=`/`weights_vector=` forms.

**Recommended fix**: Extend the `_ensure_vector_kwargs_aliases` scan set to plain `selection=`/`weights=` strings (one scan site; all surfaces inherit). Regression net: Phase 13.56 T-G3 locks the corrected six-cell behavior matrix with fresh instances. Bug report: **v1.1** (`BUG_AliasDataFrame_20260610_batch_selection_alias_masked_v1_1.md`).

**Tests**: S2/S3/S4 amended in Phase 13.55.ADF to use `lazy=True` + placeholder-proof assertions (`_errors=={}`, `stats[0] is not None`).

---

### BUG_AliasDataFrame_20260609_lazy_nd_facet
*(Full entry in Phase history above — standalone section between Phase 13.53.ADF and Phase 13.55.ADF)*  
**Date**: 2026-06-10  
**Status**: ✅ Fixed  
**Commits**: `a52f5522` (tests) → `7906cdfd` (fix + CM regen)

---

### BUG_AliasDataFrame_20260526_parallel_flake_compression
**Dates**: 2026-05-26 (formal filing); first documented 2026-05-14 at Phase 13.27.ADF `bbedd90b`  
**Status**: ⚠️ Tracked (no code fix yet); Path A recommended  
**Severity**: P2 — pre-existing intermittent flake under `pytest -n 12` (xdist parallel); tests pass deterministically in isolation (`-p no:xdist`). No production correctness impact.  
**Detected via**: 4 consecutive review packages this month surfaced the cluster

**Cluster** (3 tests in `tests/test_alias_dataframe.py`):
1. `TestAliasDataFrameWithSubframes::test_save_and_load_integrity`
2. `TestAliasDataFrameCompression::test_backward_compatibility_no_compression_info`
3. `TestAliasDataFrameCompression::test_roundtrip_save_load`

**Recurrence pattern**: ~30–50% of parallel runs on alma2 Linux aarch64. Failure rate non-deterministic.

**Diagnostic confirmation** (Phase 13.35.ADF closure, 2026-05-18): `pytest <3 tests> -p no:xdist` → 15/15 pass in isolation. Phase 13.36.ADF cycle showed the cluster fire in `reviewer_20260526_110908.zip` and `reviewer_20260526_135123.zip` (commit-time + v1.1 verify) but not in `reviewer_20260526_100433.zip` or `reviewer_20260526_150919.zip` — same source, same alma2 environment, different run outcomes.

**Recommended resolution paths** (architect's call):
- **Path A** (~15 min, recommended): add `@pytest.mark.xdist_group(name="alias_dataframe_save_load")` to all 3 tests — pins them to the same xdist worker, eliminating the race
- **Path B**: investigate root cause (suspects: shared temp dirs, parquet write-then-read races, pandas global state, uproot file-handle sharing) — 2–4 hr debug session
- **Path C**: document and accept; add reviewer-card rule for fast recognition

**Review-process impact**: estimated 2–4 review-hours wasted across ~4 recurrences (each forces a context-less reviewer to investigate as potential regression). Path A's 15-min investment pays back immediately.

**Filed**: `BUG_AliasDataFrame_20260526_parallel_flake_compression.md` (separate doc with reproducer + evidence anchor + 3 resolution paths)

**Tracked from**: Phase 13.36.ADF review cycle (Claude37 P1-4, Sonnet1 MainReviewer summary P1-4)

### BUG_AliasDataFrame_20260518_draw_subframe_alias_not_materialized (Phase A, S10–S19)
**Dates**: 2026-05-18  
**Status**: ✅ Fixed  
**Commit**: `a6a5b6e8`  
**Predecessor**: `c1f77b06` (BUG_draw_silent_swallow)  
**Severity**: P0 — cold draw of `Subframe.aliased_column` raised `UndefinedVariableError` whenever the subframe column was an ADF alias not yet materialized into the subframe's DataFrame. Production reproducer: `adfVertex.draw("vertex_x_intercept:vC.vertex_x_intercept_decomp")` — fails cold because `vC.vertex_x_intercept_decomp` is an alias on the vC subframe, not a raw column.

**Problem**: Four draw-time resolver sites assumed that any `Subframe.col` reference resolves to a raw column on the subframe's DataFrame. When `col` is an ADF alias on that subframe (the common pattern for compressed/decompressed columns in calibration QA), the lookup miss propagated as `UndefinedVariableError`. The error pointed at the rewritten flat reference (e.g. `vertex_x_intercept_decomp__vC`), never at the actual cause (the column needed lazy materialization on the subframe).

**Sites patched** (all in `AliasDataFrame.py`):
| Method | Source line (approx) | Level |
|---|---|---|
| `draw()` | 11036 | Single-level |
| `draw_batch()` | 12060 | Single-level |
| `draw_figures()` | 12360 | Single-level |
| `_scatter_subframe_column` | 3108 | Multi-level |

Each site now calls `sf_adf.materialize_aliases([col_name])` on the subframe before the join.

**Silent-swallow cleanup** (completes the remediation begun at `c1f77b06`): 4× `except Exception: pass` blocks in the draw resolver paths were replaced with `warnings.warn(...)` to surface previously-masked errors. Aligns with the diagnostic-improvement direction of BUG_20260517 — drawer paths no longer hide their failures.

**Tests**: S10–S19 (10 invariance tests in `tests/test_S10_draw_subframe_alias.py`):
- S10–S12: single-level / multi-level / compound-expression alias resolution
- S13–S14: alias in arithmetic / alias in selection
- S15: raw column still works (regression guard)
- S16–S17: `draw_batch` / `draw_figures` paths
- S18: multilevel alias on inner subframe
- S19: cold draw, no workaround (production reproducer)

10/10 pass in 6.45s parallel.

**Production validation**: cold draw on alma2 with real ALICE TPC data (~9.86M tracks, 986 quantile bins) — `adfVertex.draw("vertex_x_intercept:vC.vertex_x_intercept_decomp")` produces correlation 0.9999 between signal and decompressed reference, no workaround (no pre-call `materialize_aliases([...])` needed).

**Taxonomy**: `DRAW.subframe_resolution` 33→49 tests, 2→16 invariance.

**Test count delta**: 1606 → 1620 passed (+14 — 10 new S10–S19 + 4 indirect from taxonomy regrouping). 7F+1E baseline identical (no regressions).

**Reviewer cycle**: Sonnet1 (MainReviewer), Sonnet2, Sonnet3, GPT7 — all `[!]` APPROVED WITH COMMENTS. P1 items addressed pre-commit: test file `git add`'d, feature_taxonomy.py updated, `_scatter_subframe_column` docstring revised. P2 items deferred to Phase B (AST resolver consolidation).

### BUG_AliasDataFrame_20260517_draw_silent_swallow (S6–S9)
**Dates**: 2026-05-17  
**Status**: ✅ Fixed  
**Commit**: `c1f77b06`  
**Severity**: P0 — silent miscomputation (subframe merge failures hidden as misleading `UndefinedVariableError`) plus length-unstable joins on duplicate-index subframes

**Problem**: Two failure modes in `draw()` subframe resolution, both producing wrong-looking output without surfacing the real cause:

1. **Silent swallow of subframe-resolution exceptions.** Two `try/except Exception: pass` blocks (single-level subframe ref at `AliasDataFrame.py:11042`; multi-level at `:11057`) swallowed every exception raised during subframe lookup. Downstream `pandas.eval` then failed with `UndefinedVariableError` because the unresolved alias never got rewritten — the surfaced error pointed at the symptom (alias not in df), never at the cause (e.g., schema mismatch, missing index column, dtype-loss at join boundary).

2. **Cartesian merge expansion on duplicate-index subframes.** `df_subset[index_cols].merge(sf_keys, on=index_cols, how='left')` did not deduplicate the subframe side before merging. Subframes with duplicate index entries (quantile-bin subframes; per-iteration coefficient tables before deduplication; certain `register_subframe` callers that don't pre-dedupe) produced row-count expansion → length-mismatch when the merged column was assigned back. Users saw silently-inflated downstream draw results before catching the row-count drift.

**Root cause**: 
- For (1): bare `pass` made every subframe-resolution failure indistinguishable. Conservative `try/except` placed defensively to keep draw resilient to schema mismatches, but the absence of any warning made debug intractable.
- For (2): assumption that subframes are unique on their declared index columns. Holds for most subframes but not all — particularly not for QA/diagnostic subframes registered with multi-row-per-key intentional grouping.

**Fix**: 
1. Both `except Exception: pass` replaced with `except Exception as e: warnings.warn(f"[draw] Failed to resolve subframe ref '...': {e}")`. Real errors now surfaced as warnings; draw continues with best-effort resolution (preserves existing resilience while making cause visible).
2. `sf_keys = sf_keys.drop_duplicates(subset=index_cols, keep='first')` inserted immediately before the merge. Subframes with duplicate index keys now produce length-stable joins (first occurrence wins, consistent with `set_subframe_fill` semantics).

**Tests**: S6, S7, S8, S9 (4 invariance tests in `tests/test_S6_draw_subframe_expression.py`):
- **S6**: `draw()` resolves dotted subframe ref inside arithmetic expression — calls public `adf.draw()` API (entry-point rule, Failure Mode #12)
- **S7**: Same for selection arithmetic (`selection="Sub.col > 0"` patterns)
- **S8**: Standalone dotted subframe ref still resolves — regression guard against the fix breaking pre-existing single-ref usage
- **S9**: Duplicate index keys in subframe → no merge expansion (length-stable assertion: `len(result_df) == len(input_df)`)

**Production impact**: Two classes of bug closed simultaneously:
- Silent miscomputation (the duplicate-index expansion) previously produced silently inflated result sets in some O2DistAI calibration QA workflows — the kind of bias that takes weeks to detect (echoes the ADF parser bracket-bug pattern documented by O2DistAI coder feedback 2026-05-15)
- Misleading-error class (the swallowed exception) routinely wasted developer debug time across multiple teams. The new `warnings.warn` surface gives the real exception text immediately.

**Methodology notes**:
- Both fixes are minimal and localized — the diagnostic-improvement fix (1) and the correctness fix (2) are independent and could ship separately if needed; bundled because both touch the same merge path.
- S6/S9 are entry-point tests (call `adf.draw()` directly); S7/S8 are entry-point + regression guards. Pattern aligns with bug-fix-test entry-point rule.
- Future work: production sites that depend on subframe-uniqueness (now defensively dedupe'd by draw) should declare uniqueness explicitly via `register_subframe(..., unique=True)` once that API lands — to convert "silent dedupe with warning" into "fail-loud on contract violation". Tracked as a candidate API enhancement, no commitment yet.

### BUG_AliasDataFrame_20260512_groupby_expression_materialization
**Dates**: 2026-05-12 to 2026-05-13  
**Status**: ✅ Fixed  
**Commits**: `d377a7b1` (initial fix) → consolidated through to `249fd551` (Phase 13.26.ADF close, where attribution audit closed)  
**Severity**: P0 — production crash after dfdraw Phase 13.30 deployment  
**Regression catch attribution**: Sonnet1 + Sonnet2 (matrix-history differential against PHASE_HISTORY baseline)

**Problem**: `adf.draw(..., group_by="row%3")` raised `ValueError` after dfdraw Phase 13.30 added validation that `group_by` must be a real column. Previously silently produced ungrouped output (wrong results). Now crashes loud.

**Root cause**: `ADF.draw()` materializes plot expressions (e.g., `dy_I5T` in `"dy_I5T:row"`) via `_parse_expr_aliases` but has no parallel materialization for `group_by` expressions. Expression like `"row%3"` isn't tokenized as an alias and falls through to dfdraw unchanged.

**Fix**: 12-line surgical insertion in `AliasDataFrame.draw()` at line 10987 (before `DFDraw(df_subset)` construction). Four-guard precondition: `group_by is not None and isinstance(str) and not in df_subset.columns and not in self.aliases`. Materializes via `df_subset.copy() + df_subset.eval(group_by)` — per-call temp column on the copy, no persistent alias pollution.

**Tests**: G1-G4 (4 invariance tests). G1 calls `adf.draw(group_by='row%3')` — exact public API user hit (Failure Mode #12). G4 locks "no persistent alias pollution" with before-and-after assertion (`'row%3' not in adf.aliases`).

**Cycle notes**:
- Initial Claude37 review approved ✅, missed the differential-against-prior-matrix check
- Sonnet1 + Sonnet2 caught regression: 3 newly-broken tests (SUB.register `test_save_and_load_integrity` + COMP.roundtrip `test_backward_compatibility_no_compression_info` + `test_roundtrip_save_load`)
- Verdict overridden to ❌ CHANGES REQUESTED in Main Reviewer consolidation
- The 3 regressions resolved at commit `b9c28663` (BUG_validate_aliases_false_positives) — root cause attribution: Phase 13.24 `apply_schema()` line 9424 interaction. Cleaned up implicitly with the B1 commit.
- Reinforces methodology lesson: matrix-history differential is mandatory, not optional

### BUG_AliasDataFrame_20260512_validate_aliases_false_positives (B1)
**Dates**: 2026-05-12 to 2026-05-13  
**Status**: ✅ Fixed  
**Commit**: `b9c28663`  
**Drafter**: Sonnet1  
**Severity**: P1 — false positives in `validate_aliases()` annoyance, not data corruption

**Problem**: `describe_aliases()` reported 62 broken aliases; **53 were false positives.** Three false-positive classes:
- B1_1: `np.pi` and other numpy constants flagged as missing subframes
- B1_2: `subframe.column` tokens (e.g., `Side.dy`) rechecked as bare unknowns
- B1_3: Mid-chain multi-level references (alias `A` defined as `B + C` where `B`, `C` are themselves aliases) escaping the guard

**Fix**: `validate_aliases()` rewritten to delegate to `_analyze_expression()` — the same AST-based walker used by `dependency_tree()`. No new logic introduced; eliminates parallel-implementation drift.

**Production result**: **62 broken → 9 broken** (53 false positives eliminated). 9 remaining are genuine (CTPLumi.* — dots in R subframe column names, deferred).

**Tests**: B1_1 through B1_5 (5 invariance tests). All under `tests/test_B1_validate_aliases_false_positives.py`.

**Test results**: At close, 1599 passed, 7F+1E baseline preserved.

### BUG_AliasDataFrame_save_load_compression_regression (resolved en passant)
**Dates**: Active 2026-05-12 (caught) to 2026-05-13 (resolved)  
**Status**: ✅ Fixed (no formal bug report — caught by Sonnet1/Sonnet2 during BUG_GroupBy review)  
**Severity**: P0 — silent regression of 3 documented-passing tests

**Problem**: Between Phase 13.25.ADF baseline (`16c3ca4c`, 1584 passed / 7F+1E) and BUG_GroupBy first-fix submission (`d377a7b1`), 3 tests regressed from passing to failing:
- `test_save_and_load_integrity` (SUB.register) ✅ → 🧨
- `test_backward_compatibility_no_compression_info` (COMP.roundtrip) — new failure
- `test_roundtrip_save_load` (COMP.roundtrip) — new failure

**Root cause (suspected)**: Phase 13.24 Part A change at `apply_schema()` line 9424 — `self._restore_aliases_from_dict({name: expr})` replaced `self.aliases[name] = expr` after read-only property introduction. Interaction with save/load round-trip path was unaudited.

**Resolution**: Resolved by commit `b9c28663` (BUG_validate_aliases_false_positives). Mechanism not formally attributed; the 3 regressions disappeared en passant. Failure count returned to documented 7F+1E baseline.

**Methodology lesson**: This bug was the trigger for several governance recommendations:
- Bug-fix proposal template must include "Baseline test state vs PHASE_HISTORY" field
- Reviewer Card Rule 5c proposed: matrix-history differential is mechanically required, not optional
- Multi-model panel diversity (Sonnet + Claude) caught what same-model panel would have missed

### BUG_AliasDataFrame_20260420_draw_selection_alias
**Dates**: 2026-04-20  
**Status**: ✅ Fixed  
**Commit**: `6f93e2d1`

**Problem**: `draw_batch()` and `draw_figures()` do not pass `selection` or `weights` to `_parse_expr_aliases()`. Aliases used only in selections (e.g., `isNotEdge`) are never auto-materialized. `pandas.eval` fails with `name 'isNotEdge' is not defined`.

**Root cause**: Two call sites pass `(expr, group_by, color)` but omit `selection=` and `weights=`. `draw()` was correct (already passes all params).

**Fix**: 2 lines per method — pass `selection` and `weights`.

**Tests**: S1 (xfail — draw lazy=False limitation), S1b, S2, S3, S4.

**Discovered**: Production QA (`makeIterationFit123_QA`) on gr17.

### BUG_AliasDataFrame_20260424_dtype_loss_subframe_join
**Dates**: 2026-04-24  
**Status**: ✅ Fixed  
**Commit**: `bfb4d22f`

**Problem**: Aliases declared with integer/bool dtype lose their dtype through subframe joins. The join produces NaN for missing keys; pandas raises `IntCastingNaNError` on `.astype(int8)`; the cast fails silently; result stays float32. Downstream bool operators (`isPrimITS & isNotEdge`) fail.

**Root cause**: Three `.astype()` calls caught only `AttributeError`, not `IntCastingNaNError`.

**Fix**: New `_safe_dtype_cast()` helper fills NaN with 0 (int) or False (bool) before casting, with `RuntimeWarning`. Replaces all 3 raw `.astype()` calls.

**Tests**: D1-D5 (int8, bool, float unaffected, no-NaN no-warning, bool & bool production pattern).

### BUG_AliasDataFrame_20260426_draw_index_col_collision
**Dates**: 2026-04-26  
**Status**: ✅ Fixed  
**Commit**: `d3188527`

**Problem**: `adf.draw('Sub.col:Sub.index_col')` raises `KeyError` when the plotted column is also one of the subframe's `index_columns`. The draw resolver selects the column twice via `sf.df[index_cols + [col_name]]`, then `.rename()` renames both copies (pandas rename is name-based), destroying the join key.

**Root cause**: No guard for `col_name in index_cols` in two draw resolvers.

**Fix**: 5 lines per resolver — guard `col_name in index_cols`, copy + add instead of select + rename. Two locations: `draw()` and `draw_figures()`. `draw_batch()` uses direct index lookup — not affected.

**Tests**: S5_1-S5_6 (index col on x/y axis, selection, correctness, draw_figures, both axes as index cols).

**Discovered**: O2DistAI Phase 0.3 (`makeTrackPairGB` QA plots).

### BUG_AliasDataFrame_20260427_alias_invalidation
**Dates**: 2026-04-27  
**Status**: ✅ Fixed (included in Phase 13.23.ADF step 2 commit)  
**Commit**: `878ee941` (part of Phase 13.23.ADF)

**Problem**: `add_alias()` updates the expression in `_schema["columns"]` but does NOT drop the old materialized column from `self.df`. Stale values persist silently. Aliases that transitively depend on the redefined alias also keep their stale materialized values. Production impact: iterative calibration with coefficient-swap pattern produces wrong corrections.

**Root cause**: No invalidation mechanism — `add_alias()` overwrites the schema entry but never checks if the old value was materialized.

**Fix**: New `_invalidate_alias_cascade(name)` method — builds reverse dependency map via `_resolve_dependencies()`, BFS from changed alias to find all transitive dependents, drops all stale materialized columns. Raw columns never dropped. Called in `add_alias()` after schema write.

**Tests**: V1-V7 (7 invariance tests): basic redefine, cascade, 3-level cascade, unrelated not dropped, raw columns protected, new alias no drop, production pattern (iterative calibration with subframe coefficient swap).

**Severity**: P0 Safety — silent wrong results in iterative calibration workflows.

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
| `_ReadOnlyAliasDict(dict)` not `MappingProxyType` | JSON-serializable + `isinstance(x, dict)` True; MappingProxyType broke 47 tests |
| Two-cycle merge for property changes | Audit internal writes (Part A) before changing return type (Part B); validated by 2 failed single-cycle attempts |
| `_restore_aliases_from_dict` as sanctioned write path | 3 independent reviewers (GPT10, GPT11, Claude37) converged; avoids spreading `setdefault({})["expr"]` across file |
| Greedy left→right walk for multi-level subframes | `A.B.C.val` parsed segment-by-segment; first non-subframe segment = leaf column |
| MAX_SUBFRAME_DEPTH = 10 + visited_ids cycle guard | Prevents infinite recursion on self-referential subframe registration |
| ADF dispatch routes through DFDraw.draw() (AD-1/13.55.ADF) | `getattr(plotter, method_name)` replaced at `adf.draw()` + `adf.draw_figures()` dispatch sites; overlay strings and type aliases now work at ADF surface; `'auto'` pre-resolved via `_resolve_plot_type` before routing (ADF convention); 3-var `type='profile'` promoted to `'profile2d'` ADF-side until dfdraw extends its early dispatch (F-E shim). Architect: "OK. Approve." (2026-06-10) |
| draw_figures + draw_batch default on_error='raise' (AD-1/13.55.ADF) | All general-purpose ADF batch surfaces default to raise; silent-skip failure mode closed. draw_fit_summary keeps 'skip' (documented exception — data-dependent per-panel fit failures; QA-dashboard UX). Architect: "Skip" for draw_fit_summary. |
| ADF ARCHITECT_DECISIONS.md registry created | Phase 13.55.ADF; mirrors dfdraw registry v1.1.0 conventions. Seeded with AD-1/13.55.ADF. |
| Batch type shims pre-delegation; figures guards temporary (AD-2/13.56.ADF) | `draw_batch` per-spec 'auto' resolution + 3-var profile promotion in the ADF wrapper loop (dispatch stays dfdraw's — AD-1 rationale preserved); profile2d/facet_by guards in `draw_figures` removed when the dfdraw `ax=` bugs land. Architect: "D1: A · D2: B" (2026-06-11). |

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
| Phase 13.21 | ~37s saved | Join cache survives materialize_aliases (56s → 19.5s) |
| **Production** | **2.1× (1452→692s)** | **Cross-team: GB + ADF + O2DistAI fixes combined** |

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
| 13.22.ADF | 0 (E2_2 fixed) | 1523 |
| BUG draw_selection_alias | 5 (S1-S4) | 1528 |
| BUG dtype_loss_subframe | 5 (D1-D5) | 1533 |
| BUG draw_index_col | 6 (S5_1-S5_6) | 1538 |
| BUG alias_invalidation | 7 (V1-V7) | 1545 |
| 13.23.ADF step 1 | 10 (T1-T10) | 1548 |
| 13.23.ADF step 2 | 11 (N1_0-N1_10) | 1566 |
| 13.24.ADF Part B | 14 (V8+V9+N1_11) | 1579 |
| 13.25.ADF (Q1) | 5 (Q1_1-Q1_5) | 1584 |
| BUG_GroupBy_Expression_Materialization | 4 (G1-G4) | 1588 |
| BUG_validate_aliases_false_positives | 5 (B1_1-B1_5) | 1599 (clean baseline) |
| 13.26.ADF (dtype_overrides) | 10 (D1-D10) | 1602 |
| 13.27.ADF (skip_branches) | 4 (D11-D14) | 1606 |
| BUG_draw_subframe_alias (Phase A) | 10 (S10-S19) | 1620 |
| 13.35.ADF (vector kwargs + compose auto-force) | 8 (V1.1-V1.8) | 1625 |
| 13.36.ADF (subframe metadata to draw) | 6 (X1-X6) | 1633 |
| BUG_20260609_lazy_nd_facet | 11 (T1-T8 + U1-U4) | 1644 |
| 13.53.ADF (gallery script) | 0 (no unit tests) | 1644 |
| 13.55.ADF (dispatch audit) | 29 (T1–T7.5+T-auto+T8–T23, 6 groups) | 1673 |

---

## Pending Items

- [x] ~~CAPABILITY_MATRIX.md creation~~ (Phase 13.11)
- [x] ~~PHASE_BEGIN_AliasDataFrame tag~~ (Phase 13.20 close)
- [x] ~~`read_tree` recursive subframe loading~~ (Phase 13.22)
- [x] ~~Phase 13.23.ADF — Multi-level dotted expression resolution~~ (v1.2 approved, merged 2026-04-29)
- [x] ~~Phase 13.24.ADF — Read-only aliases hardening~~ (v1.2 approved, merged 2026-04-30)
- [x] ~~BUG_GroupBy_Expression_Materialization~~ (fix at `d377a7b1`; baseline regressions resolved by `b9c28663`)
- [x] ~~Phase 13.26.ADF — read_tree dtype_overrides~~ (merged 2026-05-13 at `249fd551`)
- [x] ~~Phase 13.27.ADF — read_tree skip_branches~~ (merged 2026-05-14 at `bbedd90b`)
- [x] ~~BUG_AliasDataFrame_20260518 (Phase A) — draw subframe alias not materialized~~ (merged 2026-05-18 at `a6a5b6e8`)
- [x] ~~Phase 13.35.ADF — vector kwargs alias pre-materialization + `vector_compose` auto-force~~ (merged 2026-05-18 at `879a0835`; closes spec v1.2 §6 row #3)

### Active queue (priority order)

- [ ] **BUG_AliasDataFrame_20260610_batch_selection_alias_masked v1.1** (F-A, P1, scope corrected 13.56) — plain `selection=`/`weights=` alias strings need `lazy=True` on ALL draw surfaces; vector kwargs unaffected. Fix: extend the hook scan set to plain strings (one site, all surfaces inherit; T-G3 regression net in place).
- [ ] **Parallel-flake cluster** (`BUG_AliasDataFrame_20260526_parallel_flake_compression` scope extension) — `test_K2_3`, `test_parquet_roundtrip`, `test_arrow_vs_numpy_performance` (timing threshold, run 095125) flip under 12-worker runs; CM Verified↔Broken flips accordingly. Path A (`pytest xdist_group`) recommended. Architect: "stochastic — we should fix it later."
- [ ] **dfdraw E-3: profile2d ax= ignored** (`BUG_dfdraw_20260611_profile2d_ax_ignored`, P1, routed to dfdraw) — ADF carries a temporary draw_figures guard; remove on fix.
- [ ] **dfdraw E-4: facet_by ax= ignored / figure leak** (`BUG_dfdraw_20260611_facet_by_ax_ignored`, P1, routed to dfdraw) — ADF temporary guard; remove on fix (dfdraw next step: nested sub-gridspec / figID paging).
- [ ] **dfdraw E-2: median return_data carries mean** (`BUG_dfdraw_20260611_median_return_data`, P2, routed to dfdraw) — TS §8.2 carries the limitation line until fixed.
- [ ] **dfdraw F-E: 3-var profile promotion** — `DFDraw.draw(type='profile', expr='z:y:x')` lacks the `DFDraw.profile()` promotion to profile2d. ADF shim in place (Phase 13.55.ADF). dfdraw team should extend early dispatch natively so ADF shim can be removed.
- [ ] **dfdraw F-B: named-param silent drop** — `DFDraw.draw()` silently drops type-inapplicable named params (e.g. `bins=` on scatter). Surfaced by Phase 13.55.ADF masked-failure audit. Recommendation: warn-on-unconsumed-named-param in `draw()`.
- [ ] **dfdraw F-C: StringDtype hist crash** — `DFDraw.hist` on StringDtype columns raises obscure numpy `TypeError`. Two ADF layout tests amended to explicit `on_error='skip'` as workaround. dfdraw recommendation: clean error or categorical-hist support.
- [ ] **Phase 13.25.DF FIX1** — Quantile test-quality + AD-52 sentinel + `error="none"` dispatch (proposal v1.0 drafted 2026-05-14; **27 days open**, 2 correctness P1s)
- [ ] **TS update for Phase 13.55.ADF** (§8 items) — dispatch routing behavior, `on_error` migration lines for both surfaces, `draw_fit_summary` exception, scatter3d limitation, `draw_help()` type list caveat
- [ ] **CM regeneration for Phase 13.55.ADF** — add DISPATCH.adf_routing + DISPATCH.error_visibility to `feature_taxonomy.py`; regenerate `CAPABILITY_MATRIX.md` (47→49 features expected)
- [ ] **dfdraw `auto_title` scatter fix** (`BUG_dfdraw_20260609_scatter_auto_title`) — surfaced in gallery fig04; scatter `auto_title=True` raises `PathCollection.set()` unexpected keyword. Phase test T10 targets profile instead of scatter as workaround (panel correction C-2).
- [ ] **dfdraw hist2d time_format y-range anomaly** (`BUG_dfdraw_20260610_hist2d_time_format_epoch`) — gallery fig16 y-range shows ±0.4 instead of correct nClITS range 5–7 when using `pd.to_datetime` pre-conversion workaround. P2 follow-up.
- [ ] **Parallel-execution flake cluster** — `test_save_and_load_integrity`, `test_backward_compatibility_no_compression_info`, `test_roundtrip_save_load` intermittently fail under 12-worker xdist; pass deterministically in isolation. Formally tracked as `BUG_AliasDataFrame_20260526_parallel_flake_compression.md` (filed 2026-05-26); Path A `@pytest.mark.xdist_group` recommended. `test_parquet_roundtrip` added as intermittent candidate (Phase 13.55.ADF cycle).
- [ ] **Phase 13.26.ADF P2 follow-ups** — `PHASE_13_26_ADF_v1.0_Proposal.md` upload to docs; D11/D12 compression+subframe interaction tests
- [ ] **Phase 13.27.ADF P3 follow-ups** — `feature_taxonomy.py` update for G1-G4, B1, D1-D14 (currently in Unmatched Tests)
- [ ] **A3** — Batch metadata serialization (~50-55s savings, needs minimal-UserInfo approach)
- [ ] **Phase 14** — ADFStore concept (formal architect review proposal needed; PyArrow-backed storage to escape pandas BlockManager fragmentation — see Phase 13.27.ADF production datapoint)
- [ ] **AD-50 Option C1b** — `_cached_last_ax` for Drawer state (~15 lines, deferred, not urgent)
- [ ] **Technical Summary v1.6** full public API documentation (~90 methods)
- [ ] **P1 tests**: I2_6, I4_2, I4_3 fixes
- [ ] Fix `register_subframe_lazy()` bug (BUG_AliasDataFrame_20260116)
- [x] ~~Axis title lookup for subframe columns~~ (closed by Phase 13.36.ADF at `de6652a0` + `a1071361`)
- [x] ~~Gallery post-change integration gate~~ (closed by Phase 13.53.ADF + AD-TS-DRAW-001 policy)
- [x] ~~BUG_AliasDataFrame_20260609_lazy_nd_facet~~ (fixed at `7906cdfd`, 2026-06-10)
- [x] ~~Phase 13.55.ADF dispatch routing + on_error defaults~~ (merged at `df0daac7`, 2026-06-10)
- [ ] `draw()` lazy=False doesn't materialize selection aliases (S1 xfail) — broader form of F-A above
- [ ] `draw()` resolver unification (3 parallel implementations)
- [ ] Feature taxonomy: 124 unmatched tests (post Phase 13.55.ADF; includes 29 new dispatch-audit tests)
- [ ] `test_K2_3_production_reproducer_mirror` intermittent — investigation pending (B-Q1; parallel-flake cluster candidate)
- [ ] `test_schema_serialization.py` collection error (1E pre-existing in baseline; unattributed)

### Cross-team queue (informational)

- [ ] **dfdraw F-E** — 3-var `type='profile'` → `profile2d` promotion missing in `DFDraw.draw()`/`draw_batch()`; ADF shim in place until fixed natively (Phase 13.55.ADF)
- [ ] **dfdraw F-B** — `DFDraw.draw()` silently drops type-inapplicable named params (bins= on scatter); K1_3 pre-existing failure class; recommend warn-on-unconsumed
- [ ] **dfdraw F-C** — `DFDraw.hist` on StringDtype raises obscure numpy TypeError; recommend clean error or categorical support
- [ ] **dfdraw Phase 13.55.DF follow-up** — draw_batch G8 gallery figures (fig_batch_profile2d, fig_batch_overlay) now validated end-to-end in ADF gallery; dfdraw batch path confirmed clean
- [ ] **dfdraw Phase 13.31.DF** — `facet_by` column-name support (AD-78, commit `f3ca432a` on `feature/groupby-optimization`; pending review packet completion)
- [ ] **dfdraw Phase 13.27.DF Commit 2** — `selection_delta` / `weights_delta` facet (deferred per current phasing)
- [ ] **O2DistAI** — LZ4 compression rollout, ITS dematerialization (~2 GB savings)
- [ ] **GBRegression** — GB V4 median batching (~35–40s)

### Reviewer-quality process items (for next governance-doc revision)

- [ ] Bug-fix proposal template: add mandatory "Baseline test state vs PHASE_HISTORY" field in Evidence Anchor (per BUG_GroupBy methodology lesson)
- [ ] Reviewer Card Rule 5c: matrix-history differential mechanically required (not judgment-driven)
- [ ] Reviewer Card Rule "no commit, no review": codify Phase 13.26.ADF F1 enforcement
- [ ] Anti-Library entry: "merging features without specification" (analogous to existing "Reasoning about performance without profiling")
- [ ] Claude48 → Reviewer paired-test rotation: Phase 13.25.DF FIX1 is the natural slot
- [ ] Claude37 anchoring-pattern signal: 3 instances logged across 3 cycles; one-line reminder pre-review recommended
- [ ] **Coder QRC Failure Mode candidate (from Phase 13.35.ADF)**: *"Numbers-revised-to-fit"* — when CRR-predicted test count misses actual, do not rewrite the prediction to match the result; investigate the delta first, narrate second. Phase 13.35.ADF: Claude36 changed baseline 7F+1E → 10F+1E in CRR §3 to match the observed result instead of investigating why the prediction missed. Reviewer panel correctly caught it.
- [ ] **Coder QRC reminder (from Phase 13.35.ADF)**: verbal architect direction mid-implementation that expands scope beyond the approved spec should produce a spec amendment (v1.x → v1.x+1) BEFORE coding, not as a post-hoc CRR §5 note. Phase 13.35.ADF auto-force was architecturally correct but the process bypassed the spec-amendment loop.
- [ ] **Reviewer QRC reminder (from Phase 13.35.ADF)**: "verdict on diagnostic, not on hypothesis" — when reviewers hypothesize a root cause for an anomaly, the gate is the diagnostic that confirms/refutes it, not the hypothesis itself. Phase 13.35.ADF: Sonnet4 hypothesized in-place mutation; diagnostic disproved it; reviewer discipline (demanding diagnostic before approval) was the value, not the specific hypothesis.
- [ ] **Source-read mandatory for all panel reviewers** (from Phase 13.55.ADF round 1): GPT12–15 all explicitly declared no source read in round 1; GPT13 only found the critical `draw_batch` on_error P1 (A-7) in round 2 after performing an actual source read. Process rule reinforced: for own-team reviews, proposal-level reviewers must either (a) read source or (b) explicitly declare proposal-only scope.
- [ ] **Coder pre-implementation source-read requirement** (from Phase 13.55.ADF): Fable1 found the A-8 `'auto'` sentinel defect (would have broken all default-type ADF calls) by running the source-read listed in §3 before writing code. The 7-reviewer panel missed it at the proposal stage. Candidate process rule: coder source-read is not optional; findings must be reported in the CRR even if the spec was wrong.
- [ ] **Gallery-driven masked-failure discovery** (from Phase 13.55.ADF): Changing `on_error='skip'`→`'raise'` by default revealed 6 tests that were unknowingly relying on the skip default (F-A/F-B/F-C/F-D findings). The gallery further caught the fig08 F-E regression (3-var profile promotion gap). AD-TS-DRAW-001 policy justified — the gallery found a real bug that the 29-test matrix would not have caught alone.

---

*Document generated from git history. For updates, run:*
```bash
git log --oneline --since="2025-11-01" > history.log
```
