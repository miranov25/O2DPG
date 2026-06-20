# Capability Matrix — AliasDataFrame

**Generated:** 2026-06-20 13:59 UTC
**Phase:** 13.11.B
**Taxonomy:** 53 features (PHASE_13_11_B approved)
**Generator:** `scripts/generate_capability_matrix.py` v2 (taxonomy-based)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 20 | 37% |
| ☑️ Smoke-only | 12 | 22% |
| 🧨 Broken | 19 | 35% |
| 📋 Planned | 2 | 3% |
| **Total features** | **53** | |
| **Matched tests** | **1780** | |
| **Invariance tests** | **290** | |

**Unmatched tests:** 99 (not mapped to any feature)

## CORE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **CORE.alias_definition** — Alias definition & expression evaluation | 44 | 42 | 1 | 2 |
| ✅ | **CORE.materialization** — Alias materialization (single + batch) | 28 | 28 | 0 | 2 |
| ✅ | **CORE.dependency_resolution** — Dependency chain & fill_value resolution | 60 | 59 | 0 | 4 |
| 🧨 | **CORE.dtypes** — Dtype handling & casting | 12 | 7 | 5 | 7 |
| ☑️ | **CORE.constructor** — DataFrame creation & initialization | 6 | 6 | 0 |  |
| ☑️ | **CORE.describe** — Structure & alias inspection | 6 | 6 | 0 |  |
| ☑️ | **CORE.cleanup** — Column cleanup & temporary management | 4 | 4 | 0 |  |
| ☑️ | **CORE.api_contract** — Public API stability | 8 | 8 | 0 |  |
| ☑️ | **CORE.dependency_tree** — Dependency tree output (text/html/list) | 16 | 16 | 0 |  |
| ✅ | **CORE.invalidation** — Alias invalidation on expression redefine | 7 | 7 | 0 | 7 |

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **SUB.register** — Subframe registration | 50 | 47 | 2 | 3 |
| 🧨 | **SUB.join** — Subframe join & column resolution | 70 | 65 | 3 | 22 |
| ✅ | **SUB.composite_key** — Composite key operations | 38 | 38 | 0 | 2 |
| ✅ | **SUB.auto_alias** — Auto-aliasing subframe columns | 11 | 10 | 0 | 1 |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 |  |
| 🧨 | **SUB.nested** — Nested subframe export | 13 | 4 | 1 | 8 |
| ✅ | **SUB.multilevel** — Multi-level dotted subframe resolution (A.B.C.val) | 11 | 10 | 0 | 11 |

## SCHEMA

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **SCHEMA.export_import** — Schema export & import (JSON) | 242 | 240 | 2 | 6 |
| 🧨 | **SCHEMA.root_persistence** — ROOT file persistence | 39 | 28 | 2 | 5 |
| ☑️ | **SCHEMA.validation** — Schema validation | 22 | 22 | 0 |  |
| ☑️ | **SCHEMA.versioning** — Schema versioning & migration | 6 | 6 | 0 |  |

## REGISTERED_FUNCTIONS

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **FUNC.register_function** — register_function API | 8 | 8 | 0 | 2 |
| 🧨 | **FUNC.polynomial** — PolynomialSpec & register_polynomial_from_subframe | 20 | 17 | 3 | 3 |
| ✅ | **FUNC.evaluator** — register_evaluator | 24 | 24 | 0 | 3 |
| 🧨 | **FUNC.persistence** — Function persistence through schema | 9 | 6 | 3 | 2 |
| 🧨 | **FUNC.regression_metadata** — Regression metadata registration & update | 4 | 3 | 1 | 4 |
| ✅ | **FUNC.evaluator_from_metadata** — Bridge: metadata → evaluator binding | 6 | 6 | 0 | 6 |
| 🧨 | **FUNC.regression_persistence** — Regression metadata schema roundtrip | 1 | 0 | 1 | 1 |

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
| 🧨 | **COMP.roundtrip** — Compress/decompress roundtrip | 72 | 69 | 3 | 10 |
| ☑️ | **COMP.selection** — Compression method selection | 10 | 10 | 0 |  |
| ☑️ | **COMP.monitoring** — Compression quality monitoring | 15 | 15 | 0 |  |

## BACKEND

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **BACK.arrow** — PyArrow compute & scatter | 120 | 120 | 0 |  |
| 🧨 | **BACK.numba** — Numba JIT acceleration | 20 | 7 | 13 | 3 |
| 🧨 | **BACK.invariance** — Backend equivalence (numpy vs arrow vs numba) | 14 | 12 | 2 | 13 |

## LAZY_LOADING

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **LAZY.read_tree** — Lazy branch loading from ROOT | 88 | 86 | 0 | 2 |
| 🧨 | **LAZY.chain** — Chain loading (multiple files) | 60 | 53 | 5 | 8 |
| ✅ | **LAZY.materialization** — Lazy subframe & alias evaluation | 54 | 52 | 0 | 2 |
| ✅ | **LAZY.userinfo_backcompat** — Lazy-path UserInfo metadata back-compatibility (AD-3 precedence) | 15 | 9 | 0 | 5 |
| ✅ | **LAZY.timeseries_draw** — Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory) | 37 | 36 | 0 | 30 |
| ✅ | **LAZY.subframe_draw** — Subframe-column lazy draw (single-level A.col + nested A.B.col; on-demand materialization via ensure_subframe + recursive chain walk) | 3 | 3 | 0 | 3 |
| ✅ | **LAZY.alias_autoload** — Alias resolution auto-loads lazy branches (materialize_aliases / validate_aliases / describe_aliases bridge to the lazy reader; LAZY status) | 6 | 6 | 0 | 4 |

## FIT_REGISTRATION

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **FIT.registration** — Fit metadata storage & retrieval | 98 | 97 | 1 |  |
| 🧨 | **FIT.visualization** — Fit summary visualization | 18 | 17 | 1 |  |

## RDATAFRAME

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ☑️ | **RDF.export** — Export to RDataFrame | 123 | 65 | 0 |  |
| 📋 | **RDF.composite** — RDataFrame composite key support | 10 | 0 | 0 |  |

## INVARIANCE

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| 🧨 | **INV.cross_module** — Cross-module invariance tests | 8 | 5 | 2 | 7 |

## DISPATCH

| Status | Feature | Tests | Pass | Fail | Inv |
|--------|---------|------:|-----:|-----:|:---:|
| ✅ | **DISPATCH.adf_routing** — adf.draw/draw_figures route through DFDraw.draw() (auto pre-resolution, overlay strings, type aliases, 3-var profile promotion) | 28 | 28 | 0 | 28 |
| ✅ | **DISPATCH.error_visibility** — Batch-surface error visibility (on_error='raise' defaults; A-10/E-3/E-4 guards; draw_fit_summary documented exception) | 15 | 15 | 0 | 15 |

## 🧨 Broken Features — Details

### CORE.alias_definition
- ❌ `test_alias_dataframe.py::TestAliasDataFrame::test_export_import_tree_roundtrip`

### CORE.dtypes
- ❌ `test_alias_dataframe.py::TestDtypeRestoration::test_backward_compat_no_column_dtypes`
- ❌ `test_alias_dataframe.py::TestDtypeRestoration::test_compression_info_priority_over_column_dtypes`
- ❌ `test_alias_dataframe.py::TestDtypeRestoration::test_dtype_roundtrip_compressed_and_uncompressed`
- ❌ `test_alias_dataframe.py::TestDtypeRestoration::test_column_dtypes_stored_in_metadata`
- ❌ `test_alias_dataframe.py::TestDtypeRestoration::test_dtype_roundtrip_noncompressed`

### SUB.register
- ❌ `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_save_and_load_integrity`
- ❌ `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_alias_cluster_track_dx`

### SUB.join
- ❌ `test_join_caching.py::TestJoinCachingPerformance::test_second_column_faster_than_first`
- ❌ `test_join_caching.py::TestJoinCachingPerformance::test_many_columns_performance`
- ❌ `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_7_large_subframe_join`

### SUB.nested
- ❌ `test_alias_dataframe.py::TestExportTreeColumns::test_export_tree_columns_subset`

### SCHEMA.export_import
- ❌ `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_3_json_and_root_paths_produce_semantically_equal_schemas`
- ❌ `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_2_root_tree_roundtrip_preserves_full_schema`

### SCHEMA.root_persistence
- ❌ `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip`
- ❌ `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_2_root_tree_roundtrip_preserves_full_schema`

### FUNC.polynomial
- ❌ `test_polynomial_spec.py::TestRegisterPolynomial::test_end_to_end`
- ❌ `test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_flat_vs_subframe_coeffs`
- ❌ `test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_polynomial_vs_numpy`

### FUNC.persistence
- ❌ `test_polynomial_persistence.py::TestPolynomialPersistence::test_reconstruct_polynomial_from_schema`
- ❌ `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip`
- ❌ `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_before_after_schema_roundtrip`

### DRAW.execution
- ❌ `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_3_draw_batch_forwards_batch_kwargs`
- ❌ `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_3_production_reproducer_mirror`

### COMP.roundtrip
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip`
- ❌ `test_alias_dataframe.py::TestAliasDataFrameCompression::test_roundtrip_export_import_tree`
- ❌ `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip`

### BACK.numba
- ❌ `test_numba_acceleration.py::TestNumbaScatter::test_numba_handles_missing_keys`
- ❌ `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_2_columns`
- ❌ `test_numba_acceleration.py::TestNumbaIndexLookup::test_multi_column_key_uses_pandas`
- ❌ `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_different_maxes_in_main_vs_sub`
- ❌ `test_numba_acceleration.py::TestNumbaScatter::test_numba_produces_identical_results_f64`
- ❌ `test_numba_acceleration.py::TestNumbaScatter::test_numba_produces_identical_results_f32`
- ❌ `test_numba_acceleration.py::TestNumbaDirectAccelerators::test_numba_scatter_inplace`
- ❌ `test_numba_acceleration.py::TestNumbaIndexLookup::test_numba_index_lookup_matches_pandas`
- ❌ `test_numba_acceleration.py::TestNumbaIndexLookup::test_numba_index_lookup_single_column_verified`
- ❌ `test_numba_acceleration.py::TestNumbaIndexLookup::test_sparse_keys`
- ❌ `test_numba_acceleration.py::TestNumbaDirectAccelerators::test_numba_compute_join_indices`
- ❌ `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_matches_pandas_3col`
- ❌ `test_numba_acceleration.py::TestNumbaWithFillConfig::test_fill_missing_with_numba`

### BACK.invariance
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_10_large_dataset_numba_vs_numpy`
- ❌ `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy`

### LAZY.chain
- ❌ `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_2_lazy_vs_eager_alias_evaluation`
- ❌ `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_8_branch_auto_detection_equivalence`
- ❌ `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_3_lazy_vs_eager_with_subframe_join`
- ❌ `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_6_mixed_lazy_main_eager_sub`
- ❌ `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_1_lazy_vs_eager_column_values`

### FIT.registration
- ❌ `test_register_fit_result.py::TestDrawFitSummaryBasic::test_large_dataset_warns`

### FIT.visualization
- ❌ `test_register_fit_result.py::TestDrawFitSummaryBasic::test_large_dataset_warns`

### INV.cross_module
- ❌ `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I1_load_mode`
- ❌ `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I5_schema`

### FUNC.regression_metadata
- ❌ `test_R1_metadata_persistence_invariance.py::TestR1RegressionMetadataPersistence::test_R1_1_metadata_dict_survives_schema_roundtrip`

### FUNC.regression_persistence
- ❌ `test_R1_2_evaluator_roundtrip_invariance.py::TestR1_2EvaluatorRoundtripInvariance::test_R1_2_evaluator_equivalence_after_roundtrip`

## Unmatched Tests

99 tests not mapped to any feature.

- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_1_np_pi_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_2_subframe_column_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_3_arithmetic_expression_not_broken`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_4_genuinely_broken_still_detected`
- `test_B1_validate_aliases_false_positives.py::TestB1ValidateAliasesFalsePositives::test_B1_5_truly_missing_bare_token_detected`
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
- `test_D1_dtype_overrides.py::TestSkipBranches::test_D13_skip_and_dtype_override_combined`
- `test_D1_dtype_overrides.py::TestSkipBranches::test_D14_skip_no_match_is_noop`
- `test_G1_groupby_expression.py::TestGroupByExpressionMaterialization::test_G1_arithmetic_expression_materializes`
- `test_G1_groupby_expression.py::TestGroupByExpressionMaterialization::test_G2_existing_column_unchanged`
- `test_G1_groupby_expression.py::TestGroupByExpressionMaterialization::test_G3_alias_works`
- `test_G1_groupby_expression.py::TestGroupByExpressionMaterialization::test_G4_no_alias_pollution`
- `test_N1_11_missing_column_keyerror.py::TestN1_11_MissingColumnKeyError::test_N1_11_two_level_missing_column_raises`
- `test_N1_11_missing_column_keyerror.py::TestN1_11_MissingColumnKeyError::test_N1_11b_single_level_missing_column_raises`
- `test_Q1_quantiles_profile_adf.py::TestQ1QuantilesADFPassthrough::test_Q1_1_error_bars_via_adf`
- `test_Q1_quantiles_profile_adf.py::TestQ1QuantilesADFPassthrough::test_Q1_2_band_via_adf`
- `test_Q1_quantiles_profile_adf.py::TestQ1QuantilesADFPassthrough::test_Q1_3_parity_adf_vs_dfdraw`
- `test_Q1_quantiles_profile_adf.py::TestQ1QuantilesADFPassthrough::test_Q1_4_central_median_forwarded`
- `test_Q1_quantiles_profile_adf.py::TestQ1QuantilesADFPassthrough::test_Q1_5_groupby_with_quantiles`
- ... +69 more

---
*Generated from pytest JSON + feature_taxonomy.py (v2 taxonomy-based).*