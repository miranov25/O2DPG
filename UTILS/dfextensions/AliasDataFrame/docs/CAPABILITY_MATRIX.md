# Capability Matrix — AliasDataFrame

**Generated:** 2026-04-26 07:48 UTC
**Phase:** 13.11.B
**Taxonomy:** 44 features (PHASE_13_11_B approved)
**Generator:** `scripts/generate_capability_matrix.py` v2 (taxonomy-based)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 26 | 59% |
| ☑️ Smoke-only | 14 | 31% |
| 🧨 Broken | 3 | 6% |
| 📋 Planned | 1 | 2% |
| **Total features** | **44** | |
| **Matched tests** | **1536** | |
| **Invariance tests** | **125** | |

**Unmatched tests:** 88 (not mapped to any feature)

## CORE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **CORE.alias_definition** — Alias definition & expression evaluation | 44 | 43 | 0 | 2 |
| ✅ | **CORE.materialization** — Alias materialization (single + batch) | 28 | 28 | 0 | 2 |
| ✅ | **CORE.dependency_resolution** — Dependency chain & fill_value resolution | 60 | 59 | 0 | 4 |
| ✅ | **CORE.dtypes** — Dtype handling & casting | 7 | 7 | 0 | 2 |
| ☑️ | **CORE.constructor** — DataFrame creation & initialization | 6 | 6 | 0 |  |
| ☑️ | **CORE.describe** — Structure & alias inspection | 6 | 6 | 0 |  |
| ☑️ | **CORE.cleanup** — Column cleanup & temporary management | 4 | 4 | 0 |  |
| ☑️ | **CORE.api_contract** — Public API stability | 8 | 8 | 0 |  |

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **SUB.register** — Subframe registration | 50 | 50 | 0 | 3 |
| ✅ | **SUB.join** — Subframe join & column resolution | 59 | 59 | 0 | 11 |
| ✅ | **SUB.composite_key** — Composite key operations | 38 | 38 | 0 | 2 |
| ✅ | **SUB.auto_alias** — Auto-aliasing subframe columns | 11 | 10 | 0 | 1 |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 |  |
| ☑️ | **SUB.nested** — Nested subframe export | 5 | 5 | 0 |  |

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
| ✅ | **DRAW.execution** — draw() with auto-materialization | 53 | 53 | 0 | 3 |
| ☑️ | **DRAW.batch** — draw_batch() & draw_figures() | 40 | 40 | 0 |  |
| ✅ | **DRAW.subframe_resolution** — Subframe column resolution in draw | 29 | 29 | 0 | 1 |
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

## 🧨 Broken Features — Details

### COMP.roundtrip
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip`
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip`

### BACK.invariance
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy`

### RDF.export
- ❌ `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree`
- ❌ `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend`
- ❌ `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend`

## Unmatched Tests

88 tests not mapped to any feature.

- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D1_int8_dtype_preserved_through_join`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D2_bool_dtype_preserved_through_join`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D3_float_dtype_unaffected`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D4_no_missing_keys_no_warning`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D5_boolean_and_operator_works`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_1_basic_roundtrip_no_subframes`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_2_roundtrip_3_subframes`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_3_roundtrip_production_scale_subframes`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_4_roundtrip_timing_report`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_1_single_tfile_open_per_export`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_2_nested_subframe_roundtrip`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_3_read_tree_backward_compatibility`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_4_standalone_write_metadata_to_root`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_10_dematerialize_drop_keep_mutual_exclusion`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_1_cached_equals_uncached`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_2_cache_survives_materialize_aliases`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_3_cache_invalidates_on_register_subframe`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_4_cache_invalidates_on_subframe_data_change`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_5_multi_subframe_pipeline`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_6_dematerialize_drop_and_recover`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_7_dematerialize_keep`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_8_dematerialize_all`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_9_dematerialize_ignores_raw_columns`
- `test_J1_join_cache.py::TestJ2JoinCachePerformance::test_J2_1_cache_hit_count_across_materialize_calls`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_1_draw_accepts_all_documented_kwargs`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_2_draw_forwards_kwargs_to_dfdraw`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_3_draw_batch_forwards_batch_kwargs`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_4_draw_figures_forwards_figure_kwargs`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_5_vector_expression_each_call_gets_full_kwargs`
- `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_1_scalar_groupby_baseline`
- ... +58 more

---
*Generated from pytest JSON + feature_taxonomy.py (v2 taxonomy-based).*