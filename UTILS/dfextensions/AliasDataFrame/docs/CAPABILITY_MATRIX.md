# Capability Matrix — AliasDataFrame

**Generated:** 2026-09-02 06:48 UTC
**Phase:** PHASE_13_76_ADF_BEGIN
**Taxonomy:** 68 features
**Generator:** `scripts/generate_capability_matrix.py` v4 (shared semantic model)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 42 | 61% |
| ☑️ Smoke-only | 15 | 22% |
| 🧨 Broken | 10 | 14% |
| 📋 Planned | 1 | 1% |
| **Total features** | **68** | |
| **Unique matched tests** | **2283** | |
| **Feature-test associations** | **2593** | |
| **Invariance tests** | **559** | |
| **Mapped XFAIL evidence** | **29** | |
| **Mapped XPASS evidence** | **3** | |
| **Mapped skipped tests** | **17** | |

**Unmatched tests:** 1096 (not mapped to any feature)

## CORE

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **CORE.alias_definition** — Alias definition & expression evaluation | 44 | 43 | 0 | 0 | 0 | 1 | 0 | 2 |
| ✅ | **CORE.materialization** — Alias materialization (single + batch) | 28 | 28 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **CORE.dependency_resolution** — Dynamic dependency resolution — deferred unresolved definitions, dependency chains, fill_value propagation, late namespace completion, and use-time resolution | 66 | 65 | 0 | 0 | 0 | 1 | 0 | 9 |
| ✅ | **CORE.dtypes** — Dtype handling & casting | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 7 |
| ☑️ | **CORE.constructor** — DataFrame creation & initialization | 6 | 6 | 0 | 0 | 0 | 0 | 0 |  |
| 🧨 | **CORE.describe** — Structure & alias inspection — resolved/unresolved logical definitions and validation visibility | 7 | 6 | 0 | 0 | 1 | 0 | 0 |  |
| ☑️ | **CORE.cleanup** — Column cleanup & temporary management | 4 | 4 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **CORE.api_contract** — Public API stability | 8 | 8 | 0 | 0 | 0 | 0 | 0 |  |
| ✅ | **CORE.vector_alias** — Vector (group) aliases & multi-output prediction (PHASE_13_70) — add_alias(list-of-names, tuple-or-2D expression, dtype=list) defines k scalar member columns from ONE expression evaluated ONCE via the function-generic group engine (D0, shared with register_model) and split by slot; accepted shapes are a k-tuple/list of 1-D or an (n_rows,k) 2-D ndarray (V-6); evaluate-once across siblings with the same cache/invalidation contract as ML prediction (frame length, __setitem__ write hook, release/reload, re-registration); per-name collision refused across all namespaces (CF-5); dtype-list length and arity mismatches are loud errors (CF-8/V-6); single-name list = ordinary alias. Not scope this phase: per-row vector members, dot-sugar. | 27 | 27 | 0 | 0 | 0 | 0 | 0 | 8 |
| ☑️ | **WRITE.column_assignment** — Direct column write-through via adf[col] = value (PHASE_13_62 Stage 2a / Fix A) — writes to the frame and syncs the lazy reader's loaded_branches so a hand-added column is present and never re-requested from the TTree; supports numpy/Series/list/scalar (awkward via explicit conversion); non-string key raises; bad shape raises before bookkeeping; adf.aliases immutability (_ReadOnlyAliasDict) unaffected | 51 | 51 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **CORE.dependency_tree** — Dependency tree output (text/html/list) | 16 | 16 | 0 | 0 | 0 | 0 | 0 |  |
| 🧨 | **CORE.invalidation** — Interactive invalidation — alias redefinition, final-state/fresh-instance equivalence, materialization-history independence, subframe replacement, and rejected-mutation state integrity | 31 | 12 | 0 | 0 | 19 | 0 | 0 | 12 |

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **SUB.register** — Subframe registration & replacement — registration/re-registration semantics, current logical ownership, and dependent-state invalidation obligations | 54 | 50 | 0 | 0 | 4 | 0 | 0 | 3 |
| 🧨 | **SUB.join** — Subframe join & column resolution — key matching, missing-key/declared-neutral fill semantics, conditional calibration composition, and cache/order invariance | 74 | 72 | 0 | 0 | 2 | 0 | 0 | 23 |
| ✅ | **SUB.composite_key** — Composite key operations | 38 | 38 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **SUB.auto_alias** — Auto-aliasing subframe columns | 11 | 10 | 0 | 0 | 0 | 1 | 0 | 1 |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ✅ | **SUB.nested** — Nested subframe export | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 8 |
| ✅ | **SUB.multilevel** — Multi-level dotted subframe resolution (A.B.C.val) | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 11 |

## SCHEMA

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **SCHEMA.export_import** — Schema export & import (JSON) | 242 | 242 | 0 | 0 | 0 | 0 | 0 | 6 |
| ✅ | **SCHEMA.root_persistence** — ROOT file persistence | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| ☑️ | **SCHEMA.validation** — Schema validation | 22 | 22 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **SCHEMA.versioning** — Schema versioning & migration | 6 | 6 | 0 | 0 | 0 | 0 | 0 |  |

## REGISTERED_FUNCTIONS

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **FUNC.register_function** — register_function API | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **FUNC.polynomial** — PolynomialSpec & register_polynomial_from_subframe | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 3 |
| ✅ | **FUNC.evaluator** — register_evaluator | 24 | 24 | 0 | 0 | 0 | 0 | 0 | 3 |
| ✅ | **FUNC.ml_model** — ML model registration, lazy prediction alias, persistence (embed/external/load-from-ROOT), integrity, multi-output cache, chain recovery (PHASE_13_69) — register_model register+alias in one call; ONNX canonical + native xgboost-JSON path; format='auto' byte-sniff (ROOT->JSON->ONNX); single float32 input tensor in feature order; multi-output = sibling aliases sharing ONE evaluation via a prediction cache invalidated by the __setitem__ write-event hook / release / re-registration; embed (default, ADF_ML/ blob+descriptor via uproot, UserInfo untouched) + external relative-path + load-from-ROOT persistence, MD5-verified; missing runtime -> loud refuse. Not scope: training, CCDB, GPU, full RNTuple verification, subframe-column inputs (Phase-1 deferral). | 27 | 27 | 0 | 0 | 0 | 0 | 0 | 6 |
| ✅ | **FUNC.persistence** — Function persistence through schema | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **FUNC.regression_metadata** — Regression metadata registration & update | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| ✅ | **FUNC.evaluator_from_metadata** — Bridge: metadata → evaluator binding | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 6 |
| ✅ | **FUNC.regression_persistence** — Regression metadata schema roundtrip | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

## ALIAS

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **ALIAS.source_scoped** — Source-scoped alias resolution (PHASE_13_73) — add_alias with source=<subframe> binds BARE fit formulas (e.g. GB meta['formulas']) to a registered subframe by rewriting bare names to Subframe.name. Resolution is AST-based (names are tokens, not text), so the substring-collision class fixed in 13.72 is structurally impossible. Per-name order: Attribute/Call-func untouched; source index_columns exempt (R1a); present in BOTH source and parent -> loud shadow refusal (R1); in source -> qualify; in parent-universe -> leave bare; otherwise -> refuse at registration (R2). The parent-universe includes lazy available-but-unloaded branches, struct members and registered functions (F-1), so binding works on a lazy frame before any branch is loaded. Batch binding is a loop over add_alias (PHASE_13_73_FIX removed the add_aliases helper: it could not express per-alias dtypes and was a second entry point for one concept). The rewritten alias stays user-readable and source-qualified; the existing subframe-join path performs the join (no second evaluation route). source=None is bit-identical to pre-13.73. Composes with 13.70 vector aliases. | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 3 |

## DIAGNOSTICS

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **DIAGNOSTICS.lazy_state** — User-facing lazy-state diagnostic (PHASE_13_71) — adf.describe_lazy() prints (or returns via as_dict) the main lazy reader's entries, available/loaded branches, DataFrame columns, and available-but-not-loaded set, plus a per-lazy-subframe block (available/loaded counts + index columns from _subframe_lazy_config); diagnostic-only with NO loading or materialization side effects; bounded output via max_items with a '... (+N more)' suffix; tolerant of LazyTreeReader/LazyChainReader attribute differences via getattr defaults; reports the subframe block even when the main frame is eager. Not scope this phase: loading/ensuring branches, HTML/JSON output, rich repr. | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 1 |

## DRAWING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **DRAW.execution** — draw() with auto-materialization | 70 | 67 | 0 | 0 | 2 | 0 | 1 | 17 |
| 🧨 | **DRAW.batch** — draw_batch() & draw_figures() | 45 | 44 | 0 | 0 | 1 | 0 | 0 |  |
| ✅ | **DRAW.subframe_resolution** — Subframe resolution in draw — expression-slot discovery, owner-qualified materialization/cleanup, request-level/spec-slot attribution, per-public-call evidence isolation, and terminal plan/state reconciliation | 88 | 88 | 0 | 0 | 0 | 0 | 0 | 41 |
| ✅ | **DRAW.compound_expr** — Lazy materialization of compound expressions | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **DRAW.invariance** — Draw vs materialize invariance | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 15 |

## COMPRESSION

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **COMP.roundtrip** — Compress/decompress roundtrip | 72 | 70 | 2 | 0 | 0 | 0 | 0 | 8 |
| ☑️ | **COMP.selection** — Compression method selection | 10 | 10 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **COMP.monitoring** — Compression quality monitoring | 15 | 15 | 0 | 0 | 0 | 0 | 0 |  |

## BACKEND

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **BACK.arrow** — PyArrow compute & scatter | 120 | 120 | 0 | 0 | 0 | 0 | 0 |  |
| ✅ | **BACK.numba** — Numba JIT acceleration | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 3 |
| 🧨 | **BACK.invariance** — Backend equivalence (numpy vs arrow vs numba) | 14 | 13 | 1 | 0 | 0 | 0 | 0 | 12 |

## LAZY_LOADING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **LAZY.read_tree** — Lazy branch loading from ROOT | 88 | 88 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **LAZY.chain** — Chain loading (multiple files) | 60 | 58 | 0 | 0 | 0 | 0 | 2 | 6 |
| ✅ | **LAZY.materialization** — Lazy subframe & alias evaluation — on-demand physical/logical resolution across eager/lazy parent-child modes, no-metadata operation, and raise vs warn/skip resolution policies | 70 | 70 | 0 | 0 | 0 | 0 | 0 | 18 |
| ✅ | **LAZY.userinfo_backcompat** — Lazy-path UserInfo metadata back-compatibility (AD-3 precedence) | 15 | 14 | 0 | 0 | 0 | 0 | 1 | 5 |
| ✅ | **LAZY.chain_metadata** — Chain lazy metadata recovery (PHASE_13_67) — first-file UserInfo canonical, applied by DEFAULT (aliases+dtypes+compression, 0a); raise on cross-file incompatibility (0b); union/intersection -> error by default via metadata_conflict policy (parametrizable, off-switch in message); lazy application loads zero columns (INV-1); D4 pre-sized chain frame; names_only is a valid sparse case; real public-API read_chain_lazy recovery (alias+dtype+eval) verified on export_tree fixtures; subframe parity test runs on full-metadata fixtures, skips on names-only slim | 23 | 22 | 0 | 0 | 0 | 0 | 1 | 5 |
| ✅ | **LAZY.timeseries_draw** — Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory) | 40 | 38 | 0 | 0 | 0 | 0 | 2 | 32 |
| ✅ | **LAZY.subframe_draw** — Subframe-column lazy draw — single/nested qualified references, eager/lazy parent-child compositions, structural join-key setup, post-load aliases, and on-demand ensure_subframe resolution | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 19 |
| ✅ | **LAZY.alias_autoload** — Alias resolution auto-loads lazy branches (materialize_aliases / validate_aliases / describe_aliases bridge to the lazy reader; LAZY status) | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 4 |
| ✅ | **LAZY.expression_autoload** — Expression/column lazy autoload via ensure_columns() — bridges df.eval()/direct-access paths on a lazy ADF (get_required_branches → ensure_branches; branches-only, subframe-name + dotted-ref filtered; eager no-op) | 93 | 93 | 0 | 0 | 0 | 0 | 0 | 23 |
| ✅ | **LAZY.release** — Explicit lazy-branch/struct release (PHASE_13_68) — release_branches()/release_struct() symmetric evict: drop frame columns AND unbook the physical branch(es) on the lazy reader so a later access re-reads from file; struct members translated internal->physical forward from the registry; all-or-nothing loud refuse for eager frames (DD-alpha), aliases (DD-gamma -> dematerialize), written/__file_idx__ non-branch names (DD-beta), parent-side subframe join keys (DD-delta), and names in a materialized alias's dependency closure (C-6); memory_policy surface accepts 'keep' only ('bounded'/'drop' reserved); purely additive, no automatic eviction | 17 | 15 | 0 | 0 | 0 | 0 | 2 | 3 |

## SUBFRAME

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **SUBFRAME.asymmetric_join_keys** — Asymmetric subframe join keys (PHASE_13_65) — register_subframe(right_index_columns=[...]) lets parent/child join columns differ in name (pandas left_on/right_on); name-aware across all three _compute_join_indices paths (single-col numba, Phase 8c multi-col linearization via rename-before-linearize, merge fallback); right_index_columns=None is byte-identical to the prior same-name behavior; schema-persisted with absent-field back-compat | 9 | 9 | 0 | 0 | 0 | 0 | 0 |  |

## OBJECT

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **OBJECT.struct_1to1** — 1:1 struct/object branch support (PHASE_13_66) — ROOT struct members (parent/member) usable via dot grammar (dedxTPC.dEdxTotIROC); three-name mapping (physical slash / internal member__struct / logical dot, anchor 0i); reference-driven load with A-1 rename-on-load handling bare-leaf or slash reader keys; public adf.eval() with Step-0 syntax gate; struct-aware across the 7 analysis surfaces, get_required_branches (physical form), the 5 dispatch sites, and all 3 draw surfaces; alias-over-struct (direct + nested) via _get_structs_for_aliases + _do_materialize hook; auto-detection with scalar/jagged guard (never auto-flatten 1:N, anchor 0h); schema-persisted (export + apply) with absent-key back-compat | 123 | 120 | 0 | 0 | 0 | 0 | 3 | 30 |

## FIT_REGISTRATION

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **FIT.registration** — Fit metadata storage & retrieval | 98 | 98 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **FIT.visualization** — Fit summary visualization | 18 | 18 | 0 | 0 | 0 | 0 | 0 |  |

## RDATAFRAME

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **RDF.export** — Export to RDataFrame | 123 | 115 | 3 | 0 | 0 | 0 | 5 |  |
| ☑️ | **RDF.composite** — RDataFrame composite key support | 10 | 10 | 0 | 0 | 0 | 0 | 0 |  |

## INVARIANCE

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **INV.cross_module** — Cross-module invariance tests | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 7 |
| ✅ | **INV.draw_surface_consistency** — Same-spec numerical consistency across draw(), draw_batch(), and draw_figures(), including explicit supported-surface refusals and closure reconciliation (A3) | 27 | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| ✅ | **INV.eager_lazy_slot_symmetry** — Eager/lazy expression-slot causality and exact dependency-load symmetry across selection/expression/weights/group_by/facet/vector/subframe compositions (A4) | 24 | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| ✅ | **INV.realdata_acceptance** — Deterministic real-data/gallery acceptance and state invariance — full-stack composition, provenance, environment contracts, G7.32/G7.33/G7.34 evidence, GB prepared-state reuse, and logical-state mutation falsifiers (A5) | 42 | 42 | 0 | 0 | 0 | 0 | 0 | 42 |

## DISPATCH

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **DISPATCH.adf_routing** — adf.draw/draw_figures route through DFDraw.draw() (auto pre-resolution, overlay strings, type aliases, 3-var profile promotion) | 28 | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| ✅ | **DISPATCH.error_visibility** — Batch-surface error visibility (on_error='raise' defaults; A-10/E-3/E-4 guards; draw_fit_summary documented exception) | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 15 |
| 🧨 | **DISPATCH.dict_dispatch** — Draw-path dict dispatch frame: draw()/draw_batch()/draw_figures() hand dfdraw only the needed columns (get_required_branches ∪ materialized alias names ∪ subframe index cols); structural column-count gate + peak-RSS + volume-invariance memory gates + dict≡full-frame equivalence (AC-1/1a/1b incl. subframe single+multi-level) + loud no-silent-full-frame fallback | 102 | 101 | 1 | 0 | 0 | 0 | 0 | 41 |

## TESTING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **TESTING.phase13_77_harness** — PHASE_13_77 acceptance-harness integrity — CaseSpec/FigureContract registry, comparator/gate fail-closed behavior, manifest reconciliation, environment gating, and anti-false-green controls (A1+A2) | 155 | 155 | 0 | 0 | 0 | 0 | 0 |  |

## 🧨 Broken Features — Details

### CORE.describe
- 🧨 `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InspectionContract::test_ev_8_malformed_scalar_syntax_is_visible_to_inspection` — `xfailed`

### SUB.register
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed`
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed`
- 🧨 `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_3_subframe_reregistration_invalidates_sourced_alias` — `xfailed`
- 🧨 `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_subframe_reregistration_final_state_matches_fresh_instance` — `xfailed`

### SUB.join
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed`
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed`

### DRAW.execution
- 🧨 `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_3_draw_batch_forwards_batch_kwargs` — `xfailed`
- 🧨 `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_3_production_reproducer_mirror` — `xfailed`

### DRAW.batch
- 🧨 `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S1_draw_materializes_selection_alias` — `xfailed`

### COMP.roundtrip
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip` — `failed`
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip` — `failed`

### BACK.invariance
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy` — `failed`

### RDF.export
- ❌ `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree` — `failed`
- ❌ `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend` — `failed`
- ❌ `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend` — `failed`

### CORE.invalidation
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed101]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed137]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed17]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed211]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed29]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed307]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed419]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed43]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed557]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed71]` — `xfailed`
- 🧨 `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m2_indirect_cycle_rejection_is_state_atomic` — `xfailed`
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed`
- 🧨 `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed`
- 🧨 `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_2_virtual_upstream_redefine_invalidates_materialized_dependent` — `xfailed`
- 🧨 `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_3_subframe_reregistration_invalidates_sourced_alias` — `xfailed`
- 🧨 `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_7_rejected_cycle_redefinition_is_state_atomic` — `xfailed`
- 🧨 `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_alias_redefinition_final_state_matches_fresh_instance` — `xfailed`
- 🧨 `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_subframe_reregistration_final_state_matches_fresh_instance` — `xfailed`
- 🧨 `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o2_materialization_history_converges_to_same_final_oracle[sink_warm_upstream_virtual-DYN-P0-1]` — `xfailed`

### DISPATCH.dict_dispatch
- ❌ `test_phase1361_dict.py::test_peak_rss_dict_below_full_frame` — `failed`

## Unmatched Tests

1096 tests not mapped to any feature.

- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_1_np_pi_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_2_subframe_column_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_3_arithmetic_expression_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_4_genuinely_broken_still_detected`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_5_truly_missing_bare_token_detected`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestBatchFiguresSymmetry::test_draw_batch_weights_resolves`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestBatchFiguresSymmetry::test_draw_figures_weights_resolves`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_string_slot_resolves_subframe_ref[color]`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_string_slot_resolves_subframe_ref[facet_by]`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_string_slot_resolves_subframe_ref[group_by]`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_string_slot_resolves_subframe_ref[selection]`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_string_slot_resolves_subframe_ref[weights]`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestStringSlotSymmetry::test_weights_broadcast_matches_manual`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestVectorSlotGuard::test_non_subframe_dotted_token_does_not_raise`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestVectorSlotGuard::test_none_is_noop`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestVectorSlotGuard::test_plain_column_does_not_raise`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestVectorSlotGuard::test_selection_vector_subframe_ref_raises_loud`
- `test_BUG_20260701_subframe_ref_slot_symmetry.py::TestVectorSlotGuard::test_weights_vector_subframe_ref_raises_loud`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D10_override_warning_shows_correct_dtypes`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D1_regex_converts_float64_to_float16`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D2_first_match_wins`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D3_no_override_columns_unchanged`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D4_overflow_warns`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D5_nan_preserved`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D6_no_overrides_matches_baseline`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D7_roundtrip_values_within_tolerance`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D8_schema_roundtrip_preserves_overridden_dtype`
- `test_D1_dtype_overrides.py::TestDtypeOverrides::test_D9_entry_range_with_overrides`
- `test_D1_dtype_overrides.py::TestSkipBranches::test_D11_skip_branch_not_in_dataframe`
- `test_D1_dtype_overrides.py::TestSkipBranches::test_D12_skip_reduces_column_count`
- ... +1066 more

---
*Generated from pytest JSON + feature_taxonomy.py using the shared Capability Matrix semantic model.*
*Environment: alma2 · Linux-aarch64 · Python 3.10.19 · stamped by run_tests.sh*
