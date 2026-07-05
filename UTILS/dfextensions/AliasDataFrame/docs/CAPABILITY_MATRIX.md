# Capability Matrix — AliasDataFrame

**Generated:** 2026-07-05 07:07 UTC
**Phase:** PHASE_13_57_DF_END
**Taxonomy:** 60 features (PHASE_13_11_B approved)
**Generator:** `scripts/generate_capability_matrix.py` v2 (taxonomy-based)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 37 | 61% |
| ☑️ Smoke-only | 17 | 28% |
| 🧨 Broken | 5 | 8% |
| 📋 Planned | 1 | 1% |
| **Total features** | **60** | |
| **Matched tests** | **1921** | |
| **Invariance tests** | **328** | |

**Unmatched tests:** 110 (not mapped to any feature)

## CORE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **CORE.alias_definition** — Alias definition & expression evaluation | 44 | 43 | 0 | 2 |
| ✅ | **CORE.materialization** — Alias materialization (single + batch) | 28 | 28 | 0 | 2 |
| ✅ | **CORE.dependency_resolution** — Dependency chain & fill_value resolution | 60 | 59 | 0 | 4 |
| ✅ | **CORE.dtypes** — Dtype handling & casting | 12 | 12 | 0 | 7 |
| ☑️ | **CORE.constructor** — DataFrame creation & initialization | 6 | 6 | 0 |  |
| ☑️ | **CORE.describe** — Structure & alias inspection | 6 | 6 | 0 |  |
| ☑️ | **CORE.cleanup** — Column cleanup & temporary management | 4 | 4 | 0 |  |
| ☑️ | **CORE.api_contract** — Public API stability | 8 | 8 | 0 |  |
| ☑️ | **WRITE.column_assignment** — Direct column write-through via adf[col] = value (PHASE_13_62 Stage 2a / Fix A) — writes to the frame and syncs the lazy reader's loaded_branches so a hand-added column is present and never re-requested from the TTree; supports numpy/Series/list/scalar (awkward via explicit conversion); non-string key raises; bad shape raises before bookkeeping; adf.aliases immutability (_ReadOnlyAliasDict) unaffected | 51 | 51 | 0 |  |
| ☑️ | **CORE.dependency_tree** — Dependency tree output (text/html/list) | 16 | 16 | 0 |  |
| ✅ | **CORE.invalidation** — Alias invalidation on expression redefine | 7 | 7 | 0 | 7 |

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **SUB.register** — Subframe registration | 50 | 50 | 0 | 3 |
| 🧨 | **SUB.join** — Subframe join & column resolution | 70 | 69 | 1 | 22 |
| ✅ | **SUB.composite_key** — Composite key operations | 38 | 38 | 0 | 2 |
| ✅ | **SUB.auto_alias** — Auto-aliasing subframe columns | 11 | 10 | 0 | 1 |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 |  |
| ✅ | **SUB.nested** — Nested subframe export | 13 | 13 | 0 | 8 |
| ✅ | **SUB.multilevel** — Multi-level dotted subframe resolution (A.B.C.val) | 11 | 11 | 0 | 11 |

## SCHEMA

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **SCHEMA.export_import** — Schema export & import (JSON) | 242 | 242 | 0 | 6 |
| ✅ | **SCHEMA.root_persistence** — ROOT file persistence | 5 | 5 | 0 | 5 |
| ☑️ | **SCHEMA.validation** — Schema validation | 22 | 22 | 0 |  |
| ☑️ | **SCHEMA.versioning** — Schema versioning & migration | 6 | 6 | 0 |  |

## REGISTERED_FUNCTIONS

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **FUNC.register_function** — register_function API | 8 | 8 | 0 | 2 |
| ✅ | **FUNC.polynomial** — PolynomialSpec & register_polynomial_from_subframe | 20 | 20 | 0 | 3 |
| ✅ | **FUNC.evaluator** — register_evaluator | 24 | 24 | 0 | 3 |
| ✅ | **FUNC.persistence** — Function persistence through schema | 9 | 9 | 0 | 2 |
| ✅ | **FUNC.regression_metadata** — Regression metadata registration & update | 4 | 4 | 0 | 4 |
| ✅ | **FUNC.evaluator_from_metadata** — Bridge: metadata → evaluator binding | 6 | 6 | 0 | 6 |
| ✅ | **FUNC.regression_persistence** — Regression metadata schema roundtrip | 1 | 1 | 0 | 1 |

## DRAWING

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **DRAW.execution** — draw() with auto-materialization | 70 | 67 | 2 | 20 |
| ☑️ | **DRAW.batch** — draw_batch() & draw_figures() | 45 | 44 | 0 |  |
| ✅ | **DRAW.subframe_resolution** — Subframe column resolution in draw | 55 | 55 | 0 | 22 |
| ✅ | **DRAW.compound_expr** — Lazy materialization of compound expressions | 13 | 13 | 0 | 2 |
| ✅ | **DRAW.invariance** — Draw vs materialize invariance | 18 | 18 | 0 | 15 |

## COMPRESSION

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **COMP.roundtrip** — Compress/decompress roundtrip | 72 | 70 | 2 | 10 |
| ☑️ | **COMP.selection** — Compression method selection | 10 | 10 | 0 |  |
| ☑️ | **COMP.monitoring** — Compression quality monitoring | 15 | 15 | 0 |  |

## BACKEND

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **BACK.arrow** — PyArrow compute & scatter | 120 | 120 | 0 |  |
| ✅ | **BACK.numba** — Numba JIT acceleration | 20 | 20 | 0 | 3 |
| 🧨 | **BACK.invariance** — Backend equivalence (numpy vs arrow vs numba) | 14 | 13 | 1 | 13 |

## LAZY_LOADING

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **LAZY.read_tree** — Lazy branch loading from ROOT | 88 | 88 | 0 | 2 |
| ✅ | **LAZY.chain** — Chain loading (multiple files) | 60 | 58 | 0 | 8 |
| ✅ | **LAZY.materialization** — Lazy subframe & alias evaluation | 54 | 54 | 0 | 2 |
| ✅ | **LAZY.userinfo_backcompat** — Lazy-path UserInfo metadata back-compatibility (AD-3 precedence) | 15 | 14 | 0 | 5 |
| ✅ | **LAZY.chain_metadata** — Chain lazy metadata recovery (PHASE_13_67) — first-file UserInfo canonical, applied by DEFAULT (aliases+dtypes+compression, 0a); raise on cross-file incompatibility (0b); union/intersection -> error by default via metadata_conflict policy (parametrizable, off-switch in message); lazy application loads zero columns (INV-1); D4 pre-sized chain frame; names_only is a valid sparse case; real public-API read_chain_lazy recovery (alias+dtype+eval) verified on export_tree fixtures; subframe parity test runs on full-metadata fixtures, skips on names-only slim | 23 | 22 | 0 | 6 |
| ✅ | **LAZY.timeseries_draw** — Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory) | 40 | 38 | 0 | 33 |
| ✅ | **LAZY.subframe_draw** — Subframe-column lazy draw (single-level A.col + nested A.B.col; on-demand materialization via ensure_subframe + recursive chain walk) | 3 | 3 | 0 | 3 |
| ✅ | **LAZY.alias_autoload** — Alias resolution auto-loads lazy branches (materialize_aliases / validate_aliases / describe_aliases bridge to the lazy reader; LAZY status) | 6 | 6 | 0 | 4 |
| ☑️ | **LAZY.expression_autoload** — Expression/column lazy autoload via ensure_columns() — bridges df.eval()/direct-access paths on a lazy ADF (get_required_branches → ensure_branches; branches-only, subframe-name + dotted-ref filtered; eager no-op) | 11 | 11 | 0 |  |
| ✅ | **LAZY.release** — Explicit lazy-branch/struct release (PHASE_13_68) — release_branches()/release_struct() symmetric evict: drop frame columns AND unbook the physical branch(es) on the lazy reader so a later access re-reads from file; struct members translated internal->physical forward from the registry; all-or-nothing loud refuse for eager frames (DD-alpha), aliases (DD-gamma -> dematerialize), written/__file_idx__ non-branch names (DD-beta), parent-side subframe join keys (DD-delta), and names in a materialized alias's dependency closure (C-6); memory_policy surface accepts 'keep' only ('bounded'/'drop' reserved); purely additive, no automatic eviction | 17 | 15 | 0 | 4 |

## SUBFRAME

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **SUBFRAME.asymmetric_join_keys** — Asymmetric subframe join keys (PHASE_13_65) — register_subframe(right_index_columns=[...]) lets parent/child join columns differ in name (pandas left_on/right_on); name-aware across all three _compute_join_indices paths (single-col numba, Phase 8c multi-col linearization via rename-before-linearize, merge fallback); right_index_columns=None is byte-identical to the prior same-name behavior; schema-persisted with absent-field back-compat | 9 | 9 | 0 |  |

## OBJECT

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **OBJECT.struct_1to1** — 1:1 struct/object branch support (PHASE_13_66) — ROOT struct members (parent/member) usable via dot grammar (dedxTPC.dEdxTotIROC); three-name mapping (physical slash / internal member__struct / logical dot, anchor 0i); reference-driven load with A-1 rename-on-load handling bare-leaf or slash reader keys; public adf.eval() with Step-0 syntax gate; struct-aware across the 7 analysis surfaces, get_required_branches (physical form), the 5 dispatch sites, and all 3 draw surfaces; alias-over-struct (direct + nested) via _get_structs_for_aliases + _do_materialize hook; auto-detection with scalar/jagged guard (never auto-flatten 1:N, anchor 0h); schema-persisted (export + apply) with absent-key back-compat | 41 | 38 | 0 | 7 |

## FIT_REGISTRATION

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **FIT.registration** — Fit metadata storage & retrieval | 98 | 98 | 0 |  |
| ☑️ | **FIT.visualization** — Fit summary visualization | 18 | 18 | 0 |  |

## RDATAFRAME

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **RDF.export** — Export to RDataFrame | 123 | 115 | 3 |  |
| ☑️ | **RDF.composite** — RDataFrame composite key support | 10 | 10 | 0 |  |

## INVARIANCE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **INV.cross_module** — Cross-module invariance tests | 8 | 8 | 0 | 7 |

## DISPATCH

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **DISPATCH.adf_routing** — adf.draw/draw_figures route through DFDraw.draw() (auto pre-resolution, overlay strings, type aliases, 3-var profile promotion) | 28 | 28 | 0 | 28 |
| ✅ | **DISPATCH.error_visibility** — Batch-surface error visibility (on_error='raise' defaults; A-10/E-3/E-4 guards; draw_fit_summary documented exception) | 15 | 15 | 0 | 15 |
| ✅ | **DISPATCH.dict_dispatch** — Draw-path dict dispatch frame: draw()/draw_batch()/draw_figures() hand dfdraw only the needed columns (get_required_branches ∪ materialized alias names ∪ subframe index cols); structural column-count gate + peak-RSS + volume-invariance memory gates + dict≡full-frame equivalence (AC-1/1a/1b incl. subframe single+multi-level) + loud no-silent-full-frame fallback | 20 | 20 | 0 | 18 |

## 🧨 Broken Features — Details

### SUB.join
- ❌ `test_alias_subframe.py::TestSubframeRoundtrip::test_parquet_roundtrip`

### DRAW.execution
- ❌ `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_3_production_reproducer_mirror`
- ❌ `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_3_draw_batch_forwards_batch_kwargs`

### COMP.roundtrip
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip`
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip`

### BACK.invariance
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy`

### RDF.export
- ❌ `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend`
- ❌ `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend`
- ❌ `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree`

## Unmatched Tests

110 tests not mapped to any feature.

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
- ... +80 more

---
*Generated from pytest JSON + feature_taxonomy.py (v2 taxonomy-based).*
*Environment: alma2 · Linux-aarch64 · Python 3.10.19 · stamped by run_tests.sh*
