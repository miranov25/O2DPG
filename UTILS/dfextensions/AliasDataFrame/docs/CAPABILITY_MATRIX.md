# Capability Matrix — AliasDataFrame

**Generated:** 2026-04-02 14:12 UTC
**Phase:** 13.11.B
**Taxonomy:** 41 features (PHASE_13_11_B approved)
**Generator:** `scripts/generate_capability_matrix.py` v2 (taxonomy-based)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 9 | 21% |
| ☑️ Smoke-only | 27 | 65% |
| 🧨 Broken | 4 | 9% |
| 📋 Planned | 1 | 2% |
| **Total features** | **41** | |
| **Matched tests** | **1509** | |
| **Invariance tests** | **64** | |

**Unmatched tests:** 42 (not mapped to any feature)

## CORE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **CORE.alias_definition** — Alias definition & expression evaluation | 42 | 41 | 0 |  |
| ☑️ | **CORE.materialization** — Alias materialization (single + batch) | 26 | 26 | 0 |  |
| ✅ | **CORE.dependency_resolution** — Dependency chain & fill_value resolution | 57 | 56 | 0 | 1 |
| ☑️ | **CORE.dtypes** — Dtype handling & casting | 5 | 5 | 0 |  |
| ☑️ | **CORE.constructor** — DataFrame creation & initialization | 6 | 6 | 0 |  |
| ☑️ | **CORE.describe** — Structure & alias inspection | 6 | 6 | 0 |  |
| ☑️ | **CORE.cleanup** — Column cleanup & temporary management | 4 | 4 | 0 |  |
| ☑️ | **CORE.api_contract** — Public API stability | 8 | 8 | 0 |  |

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **SUB.register** — Subframe registration | 47 | 47 | 0 |  |
| ✅ | **SUB.join** — Subframe join & column resolution | 57 | 57 | 0 | 9 |
| ☑️ | **SUB.composite_key** — Composite key operations | 36 | 36 | 0 |  |
| ☑️ | **SUB.auto_alias** — Auto-aliasing subframe columns | 10 | 9 | 0 |  |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 |  |
| ☑️ | **SUB.nested** — Nested subframe export | 5 | 5 | 0 |  |

## SCHEMA

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **SCHEMA.export_import** — Schema export & import (JSON) | 236 | 236 | 0 |  |
| 🧨 | **SCHEMA.root_persistence** — ROOT file persistence | 35 | 31 | 4 | 1 |
| ☑️ | **SCHEMA.validation** — Schema validation | 22 | 22 | 0 |  |
| ☑️ | **SCHEMA.versioning** — Schema versioning & migration | 6 | 6 | 0 |  |

## REGISTERED_FUNCTIONS

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **FUNC.register_function** — register_function API | 6 | 6 | 0 |  |
| ✅ | **FUNC.polynomial** — PolynomialSpec & register_polynomial_from_subframe | 20 | 20 | 0 | 3 |
| ✅ | **FUNC.evaluator** — register_evaluator | 24 | 24 | 0 | 3 |
| ✅ | **FUNC.persistence** — Function persistence through schema | 9 | 9 | 0 | 2 |

## DRAWING

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **DRAW.execution** — draw() with auto-materialization | 50 | 50 | 0 |  |
| ☑️ | **DRAW.batch** — draw_batch() & draw_figures() | 40 | 40 | 0 |  |
| ☑️ | **DRAW.subframe_resolution** — Subframe column resolution in draw | 28 | 28 | 0 |  |
| ✅ | **DRAW.compound_expr** — Lazy materialization of compound expressions | 12 | 12 | 0 | 1 |
| ✅ | **DRAW.invariance** — Draw vs materialize invariance | 15 | 15 | 0 | 12 |

## COMPRESSION

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **COMP.roundtrip** — Compress/decompress roundtrip | 70 | 68 | 2 | 8 |
| ☑️ | **COMP.selection** — Compression method selection | 10 | 10 | 0 |  |
| ☑️ | **COMP.monitoring** — Compression quality monitoring | 15 | 15 | 0 |  |

## BACKEND

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **BACK.arrow** — PyArrow compute & scatter | 120 | 120 | 0 |  |
| ☑️ | **BACK.numba** — Numba JIT acceleration | 17 | 17 | 0 |  |
| 🧨 | **BACK.invariance** — Backend equivalence (numpy vs arrow vs numba) | 11 | 10 | 1 | 10 |

## LAZY_LOADING

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **LAZY.read_tree** — Lazy branch loading from ROOT | 86 | 86 | 0 |  |
| ✅ | **LAZY.chain** — Chain loading (multiple files) | 60 | 58 | 0 | 8 |
| ☑️ | **LAZY.materialization** — Lazy subframe & alias evaluation | 52 | 52 | 0 |  |

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
| ✅ | **INV.cross_module** — Cross-module invariance tests | 7 | 7 | 0 | 6 |

## 🧨 Broken Features — Details

### SCHEMA.root_persistence
- ❌ `test_schema_serialization.py::TestParquetSchemaRoundtrip::test_subframe_metadata_preserved`
- ❌ `test_schema_serialization.py::TestParquetSchemaRoundtrip::test_column_dtypes_preserved`
- ❌ `test_schema_serialization.py::TestParquetSchemaRoundtrip::test_invariance_materialize_before_after`
- ❌ `test_schema_serialization.py::TestParquetSchemaRoundtrip::test_subframe_data_preserved`

### COMP.roundtrip
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip`
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip`

### BACK.invariance
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy`

### RDF.export
- ❌ `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend`
- ❌ `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree`
- ❌ `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend`

## Unmatched Tests

42 tests not mapped to any feature.

- `test_alias_dataframe.py::TestReadTreeOptimized::test_aliases_preserved`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_backward_compatibility_no_metadata`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_basic_read`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_entry_range_start_stop`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_entry_range_stop`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_invalid_tree_raises_error`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_subframe_loaded`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_subframe_warning_with_entry_range`
- `test_alias_dataframe.py::TestReadTreeOptimized::test_threaded_vs_unthreaded_equivalence`
- `test_alias_dataframe.py::TestReadTreeWithCompression::test_compressed_columns_dtype_restored`
- `test_alias_dataframe.py::TestReadTreeWithCompression::test_compression_info_preserved`
- `test_alias_dataframe.py::TestReadTreeWithCompression::test_decompression_alias_works`
- `test_alias_dataframe.py::TestReadTreeWithCompression::test_entry_range_with_compression`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_groups_roundtrip`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_groups_simple_lists`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_order_agnostic_load`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_order_canonical`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_order_full`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_order_strict_positions`
- `test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_without_groups`
- `test_fill_handling.py::TestModeComparison::test_direct_and_safe_give_same_values_when_no_missing`
- `test_fill_handling.py::TestMultipleSubframes::test_different_fill_per_subframe`
- `test_fill_handling.py::TestSubframeFillConfig::test_clear_subframe_fill`
- `test_fill_handling.py::TestSubframeFillConfig::test_clear_subframe_fill_validates_subframe`
- `test_fill_handling.py::TestSubframeFillConfig::test_set_subframe_fill_partial_override`
- `test_fill_handling.py::TestSubframeFillConfig::test_set_subframe_fill_rejects_unknown_fill_mode`
- `test_fill_handling.py::TestSubframeFillConfig::test_set_subframe_fill_stores_config`
- `test_fill_handling.py::TestSubframeFillConfig::test_set_subframe_fill_unknown_subframe_raises`
- `test_fill_handling.py::TestSubframeFillConfig::test_subframe_config_overrides_global`
- `test_fill_handling.py::TestSubframeFillConfig::test_subframe_fill_mode_overrides_global`
- ... +12 more

---
*Generated from pytest JSON + feature_taxonomy.py (v2 taxonomy-based).*