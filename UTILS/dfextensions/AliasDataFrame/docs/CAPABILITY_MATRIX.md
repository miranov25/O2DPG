# Capability Matrix — AliasDataFrame

**Generated:** 2026-09-04 08:46 UTC
**Phase:** PHASE_13_79_ADF
**Taxonomy:** 70 features
**Generator:** `scripts/generate_capability_matrix.py` v4 (shared semantic model)

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 44 | 62% |
| ☑️ Smoke-only | 14 | 20% |
| 🧨 Broken | 11 | 15% |
| 📋 Planned | 1 | 1% |
| **Total features** | **70** | |
| **Unique matched tests** | **2550** | |
| **Feature-test associations** | **2860** | |
| **Invariance tests** | **837** | |
| **Mapped XFAIL evidence** | **67** | |
| **Mapped XPASS evidence** | **3** | |
| **Mapped skipped tests** | **17** | |

**Unmatched tests:** 1081 (not mapped to any feature)

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

### Supporting tests

<details>
<summary><code>CORE.alias_definition</code> — 44 owned pytest nodes</summary>

- `test_I16_dtype_preservation_invariance.py::TestI16DtypePreservationInvariance::test_I16_1_explicit_dtype_float32_pinned` — `passed` — `invariance` — `tests/test_I16_dtype_preservation_invariance.py:L27`
- `test_I16_dtype_preservation_invariance.py::TestI16DtypePreservationInvariance::test_I16_2_default_dtype_is_not_reduced_below_input` — `passed` — `invariance` — `tests/test_I16_dtype_preservation_invariance.py:L53`
- `test_alias_dataframe.py::TestAliasDataFrame::test_basic_alias` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L18`
- `test_alias_dataframe.py::TestAliasDataFrame::test_bidirectional_atan2_support` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L113`
- `test_alias_dataframe.py::TestAliasDataFrame::test_circular_dependency_raises_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L51`
- `test_alias_dataframe.py::TestAliasDataFrame::test_constant` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L29`
- `test_alias_dataframe.py::TestAliasDataFrame::test_dependency_order` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L36`
- `test_alias_dataframe.py::TestAliasDataFrame::test_dtype` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L24`
- `test_alias_dataframe.py::TestAliasDataFrame::test_export_import_tree_roundtrip` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L75`
- `test_alias_dataframe.py::TestAliasDataFrame::test_getattr_column_and_alias_access` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L96`
- `test_alias_dataframe.py::TestAliasDataFrame::test_invalid_syntax_raises_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L61`
- `test_alias_dataframe.py::TestAliasDataFrame::test_log_rate_with_constant` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L43`
- `test_alias_dataframe.py::TestAliasDataFrame::test_partial_materialization` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L66`
- `test_alias_dataframe.py::TestAliasDataFrame::test_undefined_function_helpful_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L136`
- `test_alias_dataframe.py::TestAliasDataFrame::test_undefined_symbol_raises_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L56`
- `test_batch_materialization.py::TestEvalInNamespaceContextOverride::test_context_override_provides_values` — `passed` — `smoke` — `tests/test_batch_materialization.py:L201`
- `test_batch_materialization.py::TestEvalInNamespaceContextOverride::test_context_override_shadows_columns` — `passed` — `smoke` — `tests/test_batch_materialization.py:L212`
- `test_fill_handling.py::TestComplexExpressions::test_auto_generated_aliases_respect_fill_config` — `xpassed` — `smoke` — `tests/test_fill_handling.py:L597`
- `test_fill_handling.py::TestComplexExpressions::test_fill_applies_to_complex_expression` — `passed` — `smoke` — `tests/test_fill_handling.py:L583`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_adf_method_takes_priority` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L309`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_alias_auto_materializes` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L303`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_column_access_via_attribute` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L297`
- `test_proxy_pattern.py::TestContains::test_alias_not_in_columns` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L194`
- `test_proxy_pattern.py::TestContains::test_existing_column` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L185`
- `test_proxy_pattern.py::TestContains::test_nonexistent_column` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L190`
- `test_proxy_pattern.py::TestEdgeCases::test_chained_operations` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L340`
- `test_proxy_pattern.py::TestEdgeCases::test_empty_dataframe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L328`
- `test_proxy_pattern.py::TestEdgeCases::test_nonexistent_attribute_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L335`
- `test_proxy_pattern.py::TestGetItem::test_column_chain_operations` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L70`
- `test_proxy_pattern.py::TestGetItem::test_multiple_column_access` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L64`
- `test_proxy_pattern.py::TestGetItem::test_nonexistent_column_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L75`
- `test_proxy_pattern.py::TestGetItem::test_single_column_access` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L58`
- `test_proxy_pattern.py::TestIter::test_for_loop` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L170`
- `test_proxy_pattern.py::TestIter::test_iterate_columns` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L165`
- `test_proxy_pattern.py::TestLen::test_len_empty_dataframe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L152`
- `test_proxy_pattern.py::TestLen::test_len_matches_df` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L148`
- `test_proxy_pattern.py::TestLen::test_len_returns_row_count` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L144`
- `test_proxy_pattern.py::TestMethodDelegation::test_describe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L258`
- `test_proxy_pattern.py::TestMethodDelegation::test_groupby` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L270`
- `test_proxy_pattern.py::TestMethodDelegation::test_head` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L246`
- `test_proxy_pattern.py::TestMethodDelegation::test_mean` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L264`
- `test_proxy_pattern.py::TestMethodDelegation::test_query` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L277`
- `test_proxy_pattern.py::TestMethodDelegation::test_tail` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L252`
- `test_proxy_pattern.py::TestMethodDelegation::test_to_dict` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L283`

</details>

<details>
<summary><code>CORE.materialization</code> — 28 owned pytest nodes</summary>

- `test_I15_materialization_order_invariance.py::TestI15MaterializationOrderInvariance::test_I15_1_batch_equals_sequential_independent_aliases` — `passed` — `invariance` — `tests/test_I15_materialization_order_invariance.py:L36`
- `test_I15_materialization_order_invariance.py::TestI15MaterializationOrderInvariance::test_I15_2_reversed_order_equals_forward_order` — `passed` — `invariance` — `tests/test_I15_materialization_order_invariance.py:L68`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_batch_correctness_simple` — `passed` — `smoke` — `tests/test_batch_materialization.py:L61`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_batch_materialize_faster_than_threshold` — `passed` — `smoke` — `tests/test_batch_materialization.py:L24`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_batch_with_clean_temporary` — `passed` — `smoke` — `tests/test_batch_materialization.py:L102`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_batch_with_dependencies` — `passed` — `smoke` — `tests/test_batch_materialization.py:L79`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_batch_with_subframe` — `passed` — `smoke` — `tests/test_batch_materialization.py:L123`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_context_override_enables_chained_deps` — `passed` — `smoke` — `tests/test_batch_materialization.py:L144`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_dtype_preserved_in_batch` — `passed` — `smoke` — `tests/test_batch_materialization.py:L176`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_empty_targets_returns_empty_list` — `passed` — `smoke` — `tests/test_batch_materialization.py:L189`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_no_fragmentation_warning` — `passed` — `smoke` — `tests/test_batch_materialization.py:L44`
- `test_batch_materialization.py::TestBatchMaterializationPerformance::test_verbose_output` — `passed` — `smoke` — `tests/test_batch_materialization.py:L162`
- `test_cycle_detection.py::TestBatchOptimization::test_context_override_enables_chained_deps` — `passed` — `smoke` — `tests/test_cycle_detection.py:L290`
- `test_cycle_detection.py::TestBatchOptimization::test_verbose_output_shows_batch_operations` — `passed` — `smoke` — `tests/test_cycle_detection.py:L277`
- `test_fill_handling.py::TestMaterializationBehavior::test_fill_applied_during_materialization` — `passed` — `smoke` — `tests/test_fill_handling.py:L525`
- `test_fill_handling.py::TestMaterializationBehavior::test_fill_applied_in_dependency_chain` — `passed` — `smoke` — `tests/test_fill_handling.py:L560`
- `test_fill_handling.py::TestMaterializationBehavior::test_update_config_affects_subsequent_materialization` — `passed` — `smoke` — `tests/test_fill_handling.py:L540`
- `test_profiling.py::TestProfiling::test_materialize_alias_singular_profiling` — `passed` — `smoke` — `tests/test_profiling.py:L119`
- `test_profiling.py::TestProfiling::test_no_profiling_no_output` — `passed` — `smoke` — `tests/test_profiling.py:L110`
- `test_profiling.py::TestProfiling::test_profile_binary_only_no_stdout` — `passed` — `smoke` — `tests/test_profiling.py:L99`
- `test_profiling.py::TestProfiling::test_profile_binary_output` — `passed` — `smoke` — `tests/test_profiling.py:L48`
- `test_profiling.py::TestProfiling::test_profile_both_outputs` — `passed` — `smoke` — `tests/test_profiling.py:L58`
- `test_profiling.py::TestProfiling::test_profile_stdout_only` — `passed` — `smoke` — `tests/test_profiling.py:L31`
- `test_profiling.py::TestProfiling::test_profile_text_only_no_stdout` — `passed` — `smoke` — `tests/test_profiling.py:L86`
- `test_profiling.py::TestProfiling::test_profile_text_output` — `passed` — `smoke` — `tests/test_profiling.py:L39`
- `test_profiling.py::test_profile_binary_output` — `passed` — `smoke` — `tests/test_profiling.py:L162`
- `test_profiling.py::test_profile_both_outputs` — `passed` — `smoke` — `tests/test_profiling.py:L186`
- `test_profiling.py::test_profile_text_output` — `passed` — `smoke` — `tests/test_profiling.py:L138`

</details>

<details>
<summary><code>CORE.dependency_resolution</code> — 66 owned pytest nodes</summary>

- `test_I6_subframe_missing_key_invariance.py::TestI6SubframeMissingKeyNaNPropagation::test_I6_1_subframe_missing_key_fill_value_propagates_through_chain` — `passed` — `invariance` — `tests/test_I6_subframe_missing_key_invariance.py:L113`
- `test_I6_subframe_missing_key_invariance.py::TestI6SubframeMissingKeyNaNPropagation::test_I6_2_sequential_equals_batch_materialization` — `passed` — `invariance` — `tests/test_I6_subframe_missing_key_invariance.py:L195`
- `test_I6_subframe_missing_key_invariance.py::TestI6SubframeMissingKeyNaNPropagation::test_I6_3_materialization_order_does_not_affect_final_values` — `passed` — `invariance` — `tests/test_I6_subframe_missing_key_invariance.py:L263`
- `test_cycle_detection.py::TestCycleDetection::test_cycle_error_shows_expressions` — `passed` — `smoke` — `tests/test_cycle_detection.py:L91`
- `test_cycle_detection.py::TestCycleDetection::test_cycle_error_shows_hint` — `passed` — `smoke` — `tests/test_cycle_detection.py:L108`
- `test_cycle_detection.py::TestCycleDetection::test_multiple_self_referential_aliases_no_cycle` — `passed` — `smoke` — `tests/test_cycle_detection.py:L44`
- `test_cycle_detection.py::TestCycleDetection::test_non_cycle_chain_works` — `passed` — `smoke` — `tests/test_cycle_detection.py:L133`
- `test_cycle_detection.py::TestCycleDetection::test_real_cycle_detected_with_message` — `passed` — `smoke` — `tests/test_cycle_detection.py:L75`
- `test_cycle_detection.py::TestCycleDetection::test_self_referential_subframe_alias_no_cycle` — `passed` — `smoke` — `tests/test_cycle_detection.py:L25`
- `test_cycle_detection.py::TestCycleDetection::test_three_way_cycle_detected` — `passed` — `smoke` — `tests/test_cycle_detection.py:L120`
- `test_cycle_detection.py::TestIndexColumnMaterialization::test_alias_index_column_materialized_in_batch` — `passed` — `smoke` — `tests/test_cycle_detection.py:L153`
- `test_cycle_detection.py::TestIndexColumnMaterialization::test_batch_path_matches_single_path` — `passed` — `smoke` — `tests/test_cycle_detection.py:L185`
- `test_cycle_detection.py::TestIndexColumnMaterialization::test_computed_index_column` — `passed` — `smoke` — `tests/test_cycle_detection.py:L246`
- `test_cycle_detection.py::TestIndexColumnMaterialization::test_multi_key_index_aliases_materialized` — `passed` — `smoke` — `tests/test_cycle_detection.py:L214`
- `test_dependency_tree.py::test_dependency_tree_basic` — `passed` — `smoke` — `tests/test_dependency_tree.py:L12`
- `test_dependency_tree.py::test_dependency_tree_max_depth` — `passed` — `smoke` — `tests/test_dependency_tree.py:L42`
- `test_dependency_tree.py::test_dependency_tree_show_expr_false` — `passed` — `smoke` — `tests/test_dependency_tree.py:L73`
- `test_dependency_tree.py::test_dependency_tree_with_subframe` — `passed` — `smoke` — `tests/test_dependency_tree.py:L98`
- `test_dependency_tree.py::test_describe_aliases_expr_width` — `passed` — `smoke` — `tests/test_dependency_tree.py:L132`
- `test_dependency_tree.py::test_describe_aliases_expr_width_none` — `passed` — `smoke` — `tests/test_dependency_tree.py:L159`
- `test_fill_handling.py::TestBackwardCompatibility::test_backward_compatibility_no_config` — `passed` — `smoke` — `tests/test_fill_handling.py:L658`
- `test_fill_handling.py::TestCalibrationWorkflow::test_calibration_workflow_example` — `passed` — `smoke` — `tests/test_fill_handling.py:L747`
- `test_fill_handling.py::TestEdgeCases::test_all_keys_present` — `passed` — `smoke` — `tests/test_fill_handling.py:L696`
- `test_fill_handling.py::TestEdgeCases::test_empty_subframe` — `passed` — `smoke` — `tests/test_fill_handling.py:L680`
- `test_fill_handling.py::TestEdgeCases::test_fill_value_zero_vs_none` — `passed` — `smoke` — `tests/test_fill_handling.py:L719`
- `test_fill_handling.py::TestFillModeDirect::test_V3_2_unmatched_key_gets_declared_neutral_in_masked_correction` — `passed` — `smoke` — `tests/test_fill_handling.py:L292`
- `test_fill_handling.py::TestFillModeDirect::test_direct_mode_does_not_fill_original_nan` — `passed` — `smoke` — `tests/test_fill_handling.py:L320`
- `test_fill_handling.py::TestFillModeDirect::test_direct_mode_fills_missing` — `passed` — `smoke` — `tests/test_fill_handling.py:L264`
- `test_fill_handling.py::TestFillModeDirect::test_direct_mode_no_fill` — `passed` — `smoke` — `tests/test_fill_handling.py:L278`
- `test_fill_handling.py::TestFillModeSafe::test_fill_missing_replaces_nan` — `passed` — `smoke` — `tests/test_fill_handling.py:L342`
- `test_fill_handling.py::TestFillModeSafe::test_safe_mode_does_not_touch_non_subframe_aliases` — `passed` — `smoke` — `tests/test_fill_handling.py:L406`
- `test_fill_handling.py::TestFillModeSafe::test_safe_mode_fills_all_invalid` — `passed` — `smoke` — `tests/test_fill_handling.py:L391`
- `test_fill_handling.py::TestFillModeSafe::test_safe_mode_fills_inf` — `passed` — `smoke` — `tests/test_fill_handling.py:L380`
- `test_fill_handling.py::TestFillModeSafe::test_safe_mode_fills_missing_only` — `passed` — `smoke` — `tests/test_fill_handling.py:L353`
- `test_fill_handling.py::TestFillModeSafe::test_safe_mode_fills_nan` — `passed` — `smoke` — `tests/test_fill_handling.py:L367`
- `test_fill_handling.py::TestGlobalFillConfig::test_clear_global_fill` — `passed` — `smoke` — `tests/test_fill_handling.py:L173`
- `test_fill_handling.py::TestGlobalFillConfig::test_default_fill_mode_is_safe` — `passed` — `smoke` — `tests/test_fill_handling.py:L85`
- `test_fill_handling.py::TestGlobalFillConfig::test_fast_mode_raises_not_implemented` — `passed` — `smoke` — `tests/test_fill_handling.py:L135`
- `test_fill_handling.py::TestGlobalFillConfig::test_fill_accepts_any_scalar_and_rejects_containers` — `passed` — `smoke` — `tests/test_fill_handling.py:L141`
- `test_fill_handling.py::TestGlobalFillConfig::test_set_global_fill_defaults` — `passed` — `smoke` — `tests/test_fill_handling.py:L91`
- `test_fill_handling.py::TestGlobalFillConfig::test_set_global_fill_invalid_expands` — `passed` — `smoke` — `tests/test_fill_handling.py:L111`
- `test_fill_handling.py::TestGlobalFillConfig::test_set_global_fill_missing` — `passed` — `smoke` — `tests/test_fill_handling.py:L103`
- `test_fill_handling.py::TestGlobalFillConfig::test_set_global_fill_mode_validation` — `passed` — `smoke` — `tests/test_fill_handling.py:L129`
- `test_fill_handling.py::TestGlobalFillConfig::test_set_global_fill_specific_overrides_invalid` — `passed` — `smoke` — `tests/test_fill_handling.py:L120`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_batch_materialize_preserves_direct_fill_value` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L98`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_dependency_chain_applies_fill_value` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L63`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_dependency_materialize_applies_fill_value` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L51`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_direct_materialize_applies_fill_value` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L40`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_fill_value_with_multiplication_guard` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L88`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_invariance_direct_vs_dependency` — `passed` — `invariance` — `tests/test_fill_value_dependency.py:L106`
- `test_fill_value_dependency.py::TestFillValueDependencyResolution::test_multiple_fill_value_dependencies` — `passed` — `smoke` — `tests/test_fill_value_dependency.py:L75`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12DependencyResolutionContract::test_ev_1_three_level_chain_matches_independent_numpy_oracle` — `passed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L78`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12DependencyResolutionContract::test_ev_4_unresolved_definition_is_legal_and_inspectable` — `passed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L88`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12DependencyResolutionContract::test_ev_5_unresolved_failure_occurs_at_use_unrelated_eval_stays_healthy` — `passed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L97`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12DependencyResolutionContract::test_ev_6_definition_time_structural_boundary` — `passed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L106`
- `test_phase_13_76_v12_history_invariance.py::TestV12DeferredRetryInvariance::test_o5_unresolved_alias_becomes_resolvable_without_redefinition` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L332`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_add_alias_allows_subframe_reference` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L92`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_add_alias_rejects_self_reference` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L80`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_after_materialize` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L178`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_with_overwrite` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L112`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_without_overwrite` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L137`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_materialize_after_auto_alias_fix` — `xpassed` — `smoke` — `tests/test_self_referential_cycles.py:L211`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_no_self_referential_cycles` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L51`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_skip_existing_columns` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L23`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_validate_no_cycles_method` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L161`
- `test_self_referential_cycles.py::TestOnlyUnmaterializedFix::test_only_unmaterialized_with_alias_and_column` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L247`

</details>

<details>
<summary><code>CORE.dtypes</code> — 12 owned pytest nodes</summary>

- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D1_int8_dtype_preserved_through_join` — `passed` — `invariance` — `tests/test_D1_dtype_subframe_join.py:L29`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D2_bool_dtype_preserved_through_join` — `passed` — `invariance` — `tests/test_D1_dtype_subframe_join.py:L84`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D3_float_dtype_unaffected` — `passed` — `invariance` — `tests/test_D1_dtype_subframe_join.py:L123`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D4_no_missing_keys_no_warning` — `passed` — `invariance` — `tests/test_D1_dtype_subframe_join.py:L150`
- `test_D1_dtype_subframe_join.py::TestDtypeLossSubframeJoin::test_D5_boolean_and_operator_works` — `passed` — `invariance` — `tests/test_D1_dtype_subframe_join.py:L176`
- `test_I16_dtype_preservation_invariance.py::TestI16DtypePreservationInvariance::test_I16_1_explicit_dtype_float32_pinned` — `passed` — `invariance` — `tests/test_I16_dtype_preservation_invariance.py:L27`
- `test_I16_dtype_preservation_invariance.py::TestI16DtypePreservationInvariance::test_I16_2_default_dtype_is_not_reduced_below_input` — `passed` — `invariance` — `tests/test_I16_dtype_preservation_invariance.py:L53`
- `test_alias_dataframe.py::TestDtypeRestoration::test_backward_compat_no_column_dtypes` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2096`
- `test_alias_dataframe.py::TestDtypeRestoration::test_column_dtypes_stored_in_metadata` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1955`
- `test_alias_dataframe.py::TestDtypeRestoration::test_compression_info_priority_over_column_dtypes` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2064`
- `test_alias_dataframe.py::TestDtypeRestoration::test_dtype_roundtrip_compressed_and_uncompressed` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2025`
- `test_alias_dataframe.py::TestDtypeRestoration::test_dtype_roundtrip_noncompressed` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2001`

</details>

<details>
<summary><code>CORE.constructor</code> — 6 owned pytest nodes</summary>

- `test_constructor_contract.py::TestAPIContract::test_constructor_accepts_schema_id` — `passed` — `smoke` — `tests/test_constructor_contract.py:L41`
- `test_constructor_contract.py::TestAPIContract::test_constructor_rejects_metadata_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L65`
- `test_constructor_contract.py::TestAPIContract::test_constructor_rejects_schema_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L51`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_correct_params` — `passed` — `smoke` — `tests/test_constructor_contract.py:L83`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_rejects_on_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L123`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_rejects_subframe_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L105`

</details>

<details>
<summary><code>CORE.describe</code> — 7 owned pytest nodes</summary>

- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_bitmask_basic_only` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2215`
- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_bitmask_combined` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2240`
- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_prints` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2164`
- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_return_dict` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2187`
- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_values` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2202`
- `test_alias_dataframe.py::TestDescribeStructure::test_describe_structure_verbose_full` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2264`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InspectionContract::test_ev_8_malformed_scalar_syntax_is_visible_to_inspection` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L198`

</details>

<details>
<summary><code>CORE.cleanup</code> — 4 owned pytest nodes</summary>

- `test_clean_temporary.py::test_clean_temporary_multiple_subframes` — `passed` — `smoke` — `tests/test_clean_temporary.py:L124`
- `test_clean_temporary.py::test_clean_temporary_preserves_targets` — `passed` — `smoke` — `tests/test_clean_temporary.py:L77`
- `test_clean_temporary.py::test_clean_temporary_subframe_columns` — `passed` — `smoke` — `tests/test_clean_temporary.py:L15`
- `test_clean_temporary.py::test_no_cleanup_when_disabled` — `passed` — `smoke` — `tests/test_clean_temporary.py:L174`

</details>

<details>
<summary><code>CORE.api_contract</code> — 8 owned pytest nodes</summary>

- `test_constructor_contract.py::TestAPIContract::test_constructor_accepts_schema_id` — `passed` — `smoke` — `tests/test_constructor_contract.py:L41`
- `test_constructor_contract.py::TestAPIContract::test_constructor_rejects_metadata_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L65`
- `test_constructor_contract.py::TestAPIContract::test_constructor_rejects_schema_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L51`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_correct_params` — `passed` — `smoke` — `tests/test_constructor_contract.py:L83`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_rejects_on_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L123`
- `test_constructor_contract.py::TestAPIContract::test_register_subframe_rejects_subframe_param` — `passed` — `smoke` — `tests/test_constructor_contract.py:L105`
- `test_proxy_pattern.py::TestBackwardCompatibility::test_df_access_still_works` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L354`
- `test_proxy_pattern.py::TestBackwardCompatibility::test_df_modification_still_works` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L360`

</details>

<details>
<summary><code>CORE.vector_alias</code> — 27 owned pytest nodes</summary>

- `test_phase_13_70_vector_alias.py::test_VEC_10_tuple_and_2d_equivalent` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L161`
- `test_phase_13_70_vector_alias.py::test_VEC_11_parquet_roundtrip_recovers_group` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L174`
- `test_phase_13_70_vector_alias.py::test_VEC_12_dematerialize_all_or_none` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L191`
- `test_phase_13_70_vector_alias.py::test_VEC_12b_dematerialize_keep_keeps_group` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L204`
- `test_phase_13_70_vector_alias.py::test_VEC_13_self_cycle_refused` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L215`
- `test_phase_13_70_vector_alias.py::test_VEC_14_dict_return_refused` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L222`
- `test_phase_13_70_vector_alias.py::test_VEC_14b_jagged_return_refused` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L232`
- `test_phase_13_70_vector_alias.py::test_VEC_1_tuple_expression_members` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L36`
- `test_phase_13_70_vector_alias.py::test_VEC_2_2d_function_members_equal_columns` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L46`
- `test_phase_13_70_vector_alias.py::test_VEC_3_evaluate_once_across_siblings` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L58`
- `test_phase_13_70_vector_alias.py::test_VEC_4_cache_invalidation` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L73`
- `test_phase_13_70_vector_alias.py::test_VEC_4_root_export_and_read_tree_lazy_recover` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L253`
- `test_phase_13_70_vector_alias.py::test_VEC_4b_direct_recover_from_file` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L273`
- `test_phase_13_70_vector_alias.py::test_VEC_4c_chain_first_file_canonical_recover` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L287`
- `test_phase_13_70_vector_alias.py::test_VEC_4d_invalidation_release_reregister_chain` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L434`
- `test_phase_13_70_vector_alias.py::test_VEC_5_arity_mismatch_2d_func` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L98`
- `test_phase_13_70_vector_alias.py::test_VEC_5_arity_mismatch_refused` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L92`
- `test_phase_13_70_vector_alias.py::test_VEC_6_collision_each_namespace` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L107`
- `test_phase_13_70_vector_alias.py::test_VEC_6_members_in_draw_slots[eager]` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L384`
- `test_phase_13_70_vector_alias.py::test_VEC_6_members_in_draw_slots[lazy]` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L384`
- `test_phase_13_70_vector_alias.py::test_VEC_6b_member_exact_load_closure` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L411`
- `test_phase_13_70_vector_alias.py::test_VEC_7_dtype_applied` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L131`
- `test_phase_13_70_vector_alias.py::test_VEC_7_dtype_length_and_duplicate_names` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L122`
- `test_phase_13_70_vector_alias.py::test_VEC_8_single_name_list_is_ordinary_alias` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L137`
- `test_phase_13_70_vector_alias.py::test_VEC_8b_release_branches_lazy_names_group` — `passed` — `smoke` — `tests/test_phase_13_70_vector_alias.py:L303`
- `test_phase_13_70_vector_alias.py::test_VEC_9_member_equals_prefilled_column` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L145`
- `test_phase_13_70_vector_alias.py::test_VEC_9_onnx_2d_flagship` — `passed` — `invariance` — `tests/test_phase_13_70_vector_alias.py:L331`

</details>

<details>
<summary><code>WRITE.column_assignment</code> — 51 owned pytest nodes</summary>

- `test_phase_13_62_adf_s2a.py::test_P3_1_backtick_leak_bare_leaf_not_raised` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L141`
- `test_phase_13_62_adf_s2a.py::test_S2A_1_eager_numpy_writethrough` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L55`
- `test_phase_13_62_adf_s2a.py::test_S2A_2_eager_series_index_aligned` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L63`
- `test_phase_13_62_adf_s2a.py::test_S2A_3_eager_scalar_broadcast` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L69`
- `test_phase_13_62_adf_s2a.py::test_S2A_4_non_string_key_raises` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L75`
- `test_phase_13_62_adf_s2a.py::test_S2A_5_aliases_mapping_still_immutable` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L81`
- `test_phase_13_62_adf_s2a.py::test_S2A_6_lazy_writethrough_and_loaded_branches_sync` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L93`
- `test_phase_13_62_adf_s2a.py::test_S2A_7_lazy_handadded_readable_no_rerequest` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L104`
- `test_phase_13_62_adf_s2a.py::test_S2A_8_lazy_length_mismatch_raises_and_not_registered` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L115`
- `test_phase_13_62_adf_s2a.py::test_S2A_9_awkward_regular_via_to_numpy` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s2a.py:L130`
- `test_proxy_pattern.py::TestBackwardCompatibility::test_df_access_still_works` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L354`
- `test_proxy_pattern.py::TestBackwardCompatibility::test_df_modification_still_works` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L360`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_adf_method_takes_priority` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L309`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_alias_auto_materializes` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L303`
- `test_proxy_pattern.py::TestColumnAliasPriority::test_column_access_via_attribute` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L297`
- `test_proxy_pattern.py::TestContains::test_alias_not_in_columns` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L194`
- `test_proxy_pattern.py::TestContains::test_existing_column` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L185`
- `test_proxy_pattern.py::TestContains::test_nonexistent_column` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L190`
- `test_proxy_pattern.py::TestEdgeCases::test_chained_operations` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L340`
- `test_proxy_pattern.py::TestEdgeCases::test_empty_dataframe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L328`
- `test_proxy_pattern.py::TestEdgeCases::test_nonexistent_attribute_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L335`
- `test_proxy_pattern.py::TestGetItem::test_column_chain_operations` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L70`
- `test_proxy_pattern.py::TestGetItem::test_multiple_column_access` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L64`
- `test_proxy_pattern.py::TestGetItem::test_nonexistent_column_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L75`
- `test_proxy_pattern.py::TestGetItem::test_single_column_access` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L58`
- `test_proxy_pattern.py::TestIter::test_for_loop` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L170`
- `test_proxy_pattern.py::TestIter::test_iterate_columns` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L165`
- `test_proxy_pattern.py::TestLen::test_len_empty_dataframe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L152`
- `test_proxy_pattern.py::TestLen::test_len_matches_df` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L148`
- `test_proxy_pattern.py::TestLen::test_len_returns_row_count` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L144`
- `test_proxy_pattern.py::TestMethodDelegation::test_describe` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L258`
- `test_proxy_pattern.py::TestMethodDelegation::test_groupby` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L270`
- `test_proxy_pattern.py::TestMethodDelegation::test_head` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L246`
- `test_proxy_pattern.py::TestMethodDelegation::test_mean` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L264`
- `test_proxy_pattern.py::TestMethodDelegation::test_query` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L277`
- `test_proxy_pattern.py::TestMethodDelegation::test_tail` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L252`
- `test_proxy_pattern.py::TestMethodDelegation::test_to_dict` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L283`
- `test_proxy_pattern.py::TestProperties::test_columns_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L211`
- `test_proxy_pattern.py::TestProperties::test_dtypes_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L224`
- `test_proxy_pattern.py::TestProperties::test_iloc_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L233`
- `test_proxy_pattern.py::TestProperties::test_index_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L215`
- `test_proxy_pattern.py::TestProperties::test_loc_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L228`
- `test_proxy_pattern.py::TestProperties::test_shape_property` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L219`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_aliases_mapping_still_immutable` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L130`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_length_mismatch_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L125`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_non_string_key_raises` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L120`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_overwrite_existing_column` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L115`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_writethrough_list` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L100`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_writethrough_numpy_array` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L93`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_writethrough_scalar_broadcast` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L110`
- `test_proxy_pattern.py::TestSetItemWriteThrough::test_writethrough_series_index_aligned` — `passed` — `smoke` — `tests/test_proxy_pattern.py:L105`

</details>

<details>
<summary><code>CORE.dependency_tree</code> — 16 owned pytest nodes</summary>

- `test_T1_dependency_tree.py::TestDependencyTreeHTML::test_T10_html_max_depth` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L144`
- `test_T1_dependency_tree.py::TestDependencyTreeHTML::test_T6_html_returns_string` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L104`
- `test_T1_dependency_tree.py::TestDependencyTreeHTML::test_T7_html_writes_file` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L114`
- `test_T1_dependency_tree.py::TestDependencyTreeHTML::test_T8_html_multiple_roots` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L126`
- `test_T1_dependency_tree.py::TestDependencyTreeHTML::test_T9_html_subframe_tagged` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L137`
- `test_T1_dependency_tree.py::TestDependencyTreeList::test_T3_list_output_contains_all_deps` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L73`
- `test_T1_dependency_tree.py::TestDependencyTreeList::test_T4_list_output_unique` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L86`
- `test_T1_dependency_tree.py::TestDependencyTreeList::test_T5_list_output_subframe` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L94`
- `test_T1_dependency_tree.py::TestDependencyTreeText::test_T1_single_alias_text` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L50`
- `test_T1_dependency_tree.py::TestDependencyTreeText::test_T2_list_input_text` — `passed` — `smoke` — `tests/test_T1_dependency_tree.py:L60`
- `test_dependency_tree.py::test_dependency_tree_basic` — `passed` — `smoke` — `tests/test_dependency_tree.py:L12`
- `test_dependency_tree.py::test_dependency_tree_max_depth` — `passed` — `smoke` — `tests/test_dependency_tree.py:L42`
- `test_dependency_tree.py::test_dependency_tree_show_expr_false` — `passed` — `smoke` — `tests/test_dependency_tree.py:L73`
- `test_dependency_tree.py::test_dependency_tree_with_subframe` — `passed` — `smoke` — `tests/test_dependency_tree.py:L98`
- `test_dependency_tree.py::test_describe_aliases_expr_width` — `passed` — `smoke` — `tests/test_dependency_tree.py:L132`
- `test_dependency_tree.py::test_describe_aliases_expr_width_none` — `passed` — `smoke` — `tests/test_dependency_tree.py:L159`

</details>

<details>
<summary><code>CORE.invalidation</code> — 31 owned pytest nodes</summary>

- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V1_redefine_alias_drops_stale_column` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L28`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V2_cascade_invalidates_dependents` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L57`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V3_deep_cascade_3_levels` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L86`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V4_unrelated_alias_not_dropped` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L115`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V5_raw_column_never_dropped` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L134`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V6_new_alias_no_invalidation` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L155`
- `test_V1_alias_invalidation.py::TestAliasInvalidation::test_V7_production_pattern_iterative_calibration` — `passed` — `invariance` — `tests/test_V1_alias_invalidation.py:L171`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed101]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed137]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed17]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed211]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed29]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed307]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed419]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed43]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed557]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m1_seeded_alias_dag_virtual_upstream_invalidation[seed71]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L130`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m2_indirect_cycle_rejection_is_state_atomic` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L192`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m2_self_reference_rejection_is_state_atomic` — `passed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L167`
- `test_phase_13_76_v12_alias_mutation_falsifiers.py::TestV12AliasMutationFalsifiers::test_m2_vector_arity_rejection_is_state_atomic` — `passed` — `invariance` — `tests/test_phase_13_76_v12_alias_mutation_falsifiers.py:L179`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L86`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L103`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_2_virtual_upstream_redefine_invalidates_materialized_dependent` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L127`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_3_subframe_reregistration_invalidates_sourced_alias` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L148`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_7_rejected_cycle_redefinition_is_state_atomic` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L178`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_alias_redefinition_final_state_matches_fresh_instance` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L83`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_subframe_reregistration_final_state_matches_fresh_instance` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L112`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o2_materialization_history_converges_to_same_final_oracle[cold]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L161`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o2_materialization_history_converges_to_same_final_oracle[fully_warm]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L161`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o2_materialization_history_converges_to_same_final_oracle[sink_warm_upstream_virtual-DYN-P0-1]` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L161`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o2_materialization_history_converges_to_same_final_oracle[upstream_warm]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L161`

</details>

## SUBFRAMES

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **SUB.register** — Subframe registration & replacement — registration/re-registration semantics, current logical ownership, and dependent-state invalidation obligations | 54 | 50 | 0 | 0 | 4 | 0 | 0 | 3 |
| 🧨 | **SUB.join** — Subframe join & column resolution — key matching, missing-key/declared-neutral fill semantics, conditional calibration composition, and cache/order invariance | 74 | 72 | 0 | 0 | 2 | 0 | 0 | 24 |
| ✅ | **SUB.composite_key** — Composite key operations | 38 | 38 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **SUB.auto_alias** — Auto-aliasing subframe columns | 11 | 10 | 0 | 0 | 0 | 1 | 0 | 1 |
| 📋 | **SUB.clone** — Clone with selection (planned) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ✅ | **SUB.nested** — Nested subframe export | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 8 |
| ✅ | **SUB.multilevel** — Multi-level dotted subframe resolution (A.B.C.val) | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 11 |

### Supporting tests

<details>
<summary><code>SUB.register</code> — 54 owned pytest nodes</summary>

- `test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_1_inline_subframe_access_equals_alias_composition` — `passed` — `invariance` — `tests/test_I8_subframe_alias_composition_invariance.py:L70`
- `test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_2_composite_key_join_equals_pandas_merge` — `passed` — `invariance` — `tests/test_I8_subframe_alias_composition_invariance.py:L101`
- `test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_3_auto_aliased_subframe_column_equals_manual_alias` — `passed` — `invariance` — `tests/test_I8_subframe_alias_composition_invariance.py:L130`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_alias_cluster_track_dx` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L185`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_getattr_chained_subframe_access` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L255`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_getattr_subframe_alias_access` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L232`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_multi_column_index_join` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L269`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_save_and_load_integrity` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L207`
- `test_alias_dataframe.py::TestAliasDataFrameWithSubframes::test_subframe_invalid_alias_raises` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L195`
- `test_alias_subframe.py::TestSubframeBasicJoin::test_multi_key_join_2keys` — `passed` — `smoke` — `tests/test_alias_subframe.py:L83`
- `test_alias_subframe.py::TestSubframeBasicJoin::test_multi_key_join_3keys` — `passed` — `smoke` — `tests/test_alias_subframe.py:L105`
- `test_alias_subframe.py::TestSubframeBasicJoin::test_single_key_join` — `passed` — `smoke` — `tests/test_alias_subframe.py:L63`
- `test_alias_subframe.py::TestSubframeBasicJoin::test_subframe_alias_in_expression` — `passed` — `smoke` — `tests/test_alias_subframe.py:L129`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L86`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L103`
- `test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_3_subframe_reregistration_invalidates_sourced_alias` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_dynamic_alias_contract.py:L148`
- `test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_subframe_reregistration_final_state_matches_fresh_instance` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L112`
- `test_subframe_alias_api.py::test_add_alias_basic` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L67`
- `test_subframe_alias_api.py::test_alias_chain` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L469`
- `test_subframe_alias_api.py::test_alias_lifecycle` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L155`
- `test_subframe_alias_api.py::test_alias_name_collision` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L400`
- `test_subframe_alias_api.py::test_alias_with_complex_expression` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L452`
- `test_subframe_alias_api.py::test_auto_alias_tracking_manual` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L261`
- `test_subframe_alias_api.py::test_auto_aliases_dict_exists` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L84`
- `test_subframe_alias_api.py::test_backward_compatibility_no_auto_aliases` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L224`
- `test_subframe_alias_api.py::test_concurrent_modifications` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L514`
- `test_subframe_alias_api.py::test_empty_alias_dict_operations` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L410`
- `test_subframe_alias_api.py::test_get_auto_aliases_empty` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L96`
- `test_subframe_alias_api.py::test_get_auto_aliases_filtered` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L276`
- `test_subframe_alias_api.py::test_indirect_subframe_reference` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L686`
- `test_subframe_alias_api.py::test_is_auto_alias_false_for_manual` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L90`
- `test_subframe_alias_api.py::test_large_number_of_aliases` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L437`
- `test_subframe_alias_api.py::test_list_auto_aliases_empty` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L241`
- `test_subframe_alias_api.py::test_list_auto_aliases_filtered` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L303`
- `test_subframe_alias_api.py::test_list_subframes_empty` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L60`
- `test_subframe_alias_api.py::test_materialize_alias_works` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L74`
- `test_subframe_alias_api.py::test_multi_subframe_auto_aliasing` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L723`
- `test_subframe_alias_api.py::test_multi_subframe_column_collision` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L792`
- `test_subframe_alias_api.py::test_multiple_alias_operations` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L202`
- `test_subframe_alias_api.py::test_remove_alias_basic` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L105`
- `test_subframe_alias_api.py::test_remove_alias_from_auto_aliases` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L141`
- `test_subframe_alias_api.py::test_remove_alias_keep_in_schema` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L386`
- `test_subframe_alias_api.py::test_remove_alias_lenient_no_error` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L120`
- `test_subframe_alias_api.py::test_remove_alias_schema_sync` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L363`
- `test_subframe_alias_api.py::test_remove_alias_strict_error` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L114`
- `test_subframe_alias_api.py::test_remove_aliases_multiple` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L128`
- `test_subframe_alias_api.py::test_remove_auto_aliases_all` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L343`
- `test_subframe_alias_api.py::test_remove_auto_aliases_empty` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L248`
- `test_subframe_alias_api.py::test_remove_auto_aliases_selective` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L324`
- `test_subframe_alias_api.py::test_schema_embedding_root_roundtrip` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L589`
- `test_subframe_alias_api.py::test_schema_export_import_basic` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L172`
- `test_subframe_alias_api.py::test_schema_roundtrip_preserves_auto_alias_info` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L488`
- `test_subframe_alias_api.py::test_special_characters_in_alias_name` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L422`
- `test_subframe_alias_api.py::test_subframe_self_reference_regression` — `passed` — `smoke` — `tests/test_subframe_alias_api.py:L638`

</details>

<details>
<summary><code>SUB.join</code> — 74 owned pytest nodes</summary>

- `test_I14_join_index_invariance.py::TestI14JoinIndexInvariance::test_I14_1_single_key_join_equals_pandas_merge` — `passed` — `invariance` — `tests/test_I14_join_index_invariance.py:L34`
- `test_I14_join_index_invariance.py::TestI14JoinIndexInvariance::test_I14_2_composite_key_order_does_not_affect_values` — `passed` — `invariance` — `tests/test_I14_join_index_invariance.py:L59`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_10_dematerialize_drop_keep_mutual_exclusion` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L384`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_1_cached_equals_uncached` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L57`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_2_cache_survives_materialize_aliases` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L89`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_3_cache_invalidates_on_register_subframe` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L147`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_4_cache_invalidates_on_subframe_data_change` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L181`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_5_multi_subframe_pipeline` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L221`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_6_dematerialize_drop_and_recover` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L274`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_7_dematerialize_keep` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L307`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_8_dematerialize_all` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L346`
- `test_J1_join_cache.py::TestJ1JoinCacheCorrectness::test_J1_9_dematerialize_ignores_raw_columns` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L367`
- `test_J1_join_cache.py::TestJ2JoinCachePerformance::test_J2_1_cache_hit_count_across_materialize_calls` — `passed` — `invariance` — `tests/test_J1_join_cache.py:L397`
- `test_alias_subframe.py::TestMultiKeySubframeJoins::test_3d_subframe_join_matches_pandas` — `passed` — `smoke` — `tests/test_alias_subframe.py:L620`
- `test_alias_subframe.py::TestMultiKeySubframeJoins::test_3d_subframe_with_missing_keys` — `passed` — `smoke` — `tests/test_alias_subframe.py:L700`
- `test_alias_subframe.py::TestMultiKeySubframeJoins::test_4d_subframe_join_matches_pandas` — `passed` — `smoke` — `tests/test_alias_subframe.py:L658`
- `test_alias_subframe.py::TestSubframeEdgeCases::test_duplicate_keys_in_subframe` — `passed` — `smoke` — `tests/test_alias_subframe.py:L560`
- `test_alias_subframe.py::TestSubframeEdgeCases::test_empty_subframe` — `passed` — `smoke` — `tests/test_alias_subframe.py:L542`
- `test_alias_subframe.py::TestSubframeEdgeCases::test_missing_column_in_subframe_raises` — `passed` — `smoke` — `tests/test_alias_subframe.py:L513`
- `test_alias_subframe.py::TestSubframeEdgeCases::test_missing_join_key_in_main_raises` — `passed` — `smoke` — `tests/test_alias_subframe.py:L529`
- `test_alias_subframe.py::TestSubframeEdgeCases::test_undefined_subframe_raises` — `passed` — `smoke` — `tests/test_alias_subframe.py:L497`
- `test_alias_subframe.py::TestSubframeLazyEvaluation::test_alias_not_materialized_until_requested` — `passed` — `smoke` — `tests/test_alias_subframe.py:L309`
- `test_alias_subframe.py::TestSubframeLazyEvaluation::test_getattr_triggers_materialization` — `passed` — `smoke` — `tests/test_alias_subframe.py:L359`
- `test_alias_subframe.py::TestSubframeLazyEvaluation::test_materialization_idempotent` — `passed` — `smoke` — `tests/test_alias_subframe.py:L325`
- `test_alias_subframe.py::TestSubframeLazyEvaluation::test_no_accidental_flattening` — `passed` — `smoke` — `tests/test_alias_subframe.py:L343`
- `test_alias_subframe.py::TestSubframeMissingKeys::test_missing_keys_multi_key_join` — `passed` — `smoke` — `tests/test_alias_subframe.py:L258`
- `test_alias_subframe.py::TestSubframeMissingKeys::test_missing_keys_produce_nan_not_dropped` — `passed` — `smoke` — `tests/test_alias_subframe.py:L152`
- `test_alias_subframe.py::TestSubframeMissingKeys::test_no_warning_when_all_keys_present` — `passed` — `smoke` — `tests/test_alias_subframe.py:L237`
- `test_alias_subframe.py::TestSubframeMissingKeys::test_warning_shows_count` — `passed` — `smoke` — `tests/test_alias_subframe.py:L192`
- `test_alias_subframe.py::TestSubframeMissingKeys::test_warning_suppression` — `passed` — `smoke` — `tests/test_alias_subframe.py:L217`
- `test_alias_subframe.py::TestSubframeRoundtrip::test_parquet_roundtrip` — `passed` — `smoke` — `tests/test_alias_subframe.py:L461`
- `test_alias_subframe.py::TestSubframeRoundtrip::test_root_roundtrip_basic` — `passed` — `smoke` — `tests/test_alias_subframe.py:L393`
- `test_alias_subframe.py::TestSubframeRoundtrip::test_root_roundtrip_with_entry_range` — `passed` — `smoke` — `tests/test_alias_subframe.py:L430`
- `test_alias_subframe.py::TestSubframeVsFlattened::test_subframe_column_equals_direct_column` — `passed` — `smoke` — `tests/test_alias_subframe.py:L582`
- `test_fill_handling.py::TestFillModeDirect::test_V3_2_unmatched_key_gets_declared_neutral_in_masked_correction` — `passed` — `smoke` — `tests/test_fill_handling.py:L292`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_1_single_key_join_matches_pandas` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L81`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_2_multikey_2col_join_matches_pandas` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L119`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_3_multikey_3col_join_matches_pandas` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L159`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_4_missing_keys_produce_nan` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L205`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_5_duplicate_keys_handled_correctly` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L250`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_6_empty_subframe_handling` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L292`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_7_large_subframe_join` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L334`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_8_chained_subframe_expression` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L376`
- `test_invariance_subframe.py::TestInvarianceSubframe::test_I3_9_flat_normalized_equivalence` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L414`
- `test_invariance_subframe.py::TestSubframeSummary::test_I3_count` — `passed` — `invariance` — `tests/test_invariance_subframe.py:L545`
- `test_join_caching.py::TestJoinCachingCorrectness::test_cached_values_match_uncached` — `passed` — `smoke` — `tests/test_join_caching.py:L27`
- `test_join_caching.py::TestJoinCachingCorrectness::test_duplicate_keys_in_subframe` — `passed` — `smoke` — `tests/test_join_caching.py:L132`
- `test_join_caching.py::TestJoinCachingCorrectness::test_fill_value_applied_with_cache` — `passed` — `smoke` — `tests/test_join_caching.py:L82`
- `test_join_caching.py::TestJoinCachingCorrectness::test_missing_keys_handled_correctly_with_cache` — `passed` — `smoke` — `tests/test_join_caching.py:L52`
- `test_join_caching.py::TestJoinCachingCorrectness::test_multi_key_index_caching` — `passed` — `smoke` — `tests/test_join_caching.py:L107`
- `test_join_caching.py::TestJoinCachingEdgeCases::test_different_subframes_separate_caches` — `passed` — `smoke` — `tests/test_join_caching.py:L320`
- `test_join_caching.py::TestJoinCachingEdgeCases::test_empty_dataframe` — `passed` — `smoke` — `tests/test_join_caching.py:L373`
- `test_join_caching.py::TestJoinCachingEdgeCases::test_subframe_alias_materialized_correctly` — `passed` — `smoke` — `tests/test_join_caching.py:L348`
- `test_join_caching.py::TestJoinCachingLifecycle::test_cache_cleared_between_batches` — `passed` — `smoke` — `tests/test_join_caching.py:L160`
- `test_join_caching.py::TestJoinCachingLifecycle::test_cache_not_used_for_single_alias` — `passed` — `smoke` — `tests/test_join_caching.py:L193`
- `test_join_caching.py::TestJoinCachingLifecycle::test_verbose_shows_cache_stats` — `passed` — `smoke` — `tests/test_join_caching.py:L214`
- `test_join_caching.py::TestJoinCachingPerformance::test_many_columns_performance` — `passed` — `smoke` — `tests/test_join_caching.py:L281`
- `test_join_caching.py::TestJoinCachingPerformance::test_second_column_faster_than_first` — `passed` — `smoke` — `tests/test_join_caching.py:L241`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_cleared_after_materialize_batch` — `passed` — `smoke` — `tests/test_join_index_caching.py:L82`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_handles_missing_keys` — `passed` — `smoke` — `tests/test_join_index_caching.py:L187`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_hit_on_subsequent_access` — `passed` — `smoke` — `tests/test_join_index_caching.py:L66`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_initialized_empty` — `passed` — `smoke` — `tests/test_join_index_caching.py:L47`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_populated_on_first_access` — `passed` — `smoke` — `tests/test_join_index_caching.py:L54`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_produces_correct_values` — `passed` — `smoke` — `tests/test_join_index_caching.py:L161`
- `test_join_index_caching.py::TestJoinIndexCaching::test_cache_stats_reset_on_new_batch` — `passed` — `smoke` — `tests/test_join_index_caching.py:L99`
- `test_join_index_caching.py::TestJoinIndexCaching::test_five_column_batch_cache_stats` — `passed` — `smoke` — `tests/test_join_index_caching.py:L216`
- `test_join_index_caching.py::TestJoinIndexCaching::test_multiple_subframes_cached_separately` — `passed` — `smoke` — `tests/test_join_index_caching.py:L127`
- `test_materialize_subframe_index.py::TestSubframeIndexMaterialization::test_materialize_chain_with_subframe_index` — `passed` — `smoke` — `tests/test_materialize_subframe_index.py:L120`
- `test_materialize_subframe_index.py::TestSubframeIndexMaterialization::test_materialize_with_alias_index_columns` — `passed` — `smoke` — `tests/test_materialize_subframe_index.py:L19`
- `test_materialize_subframe_index.py::TestSubframeIndexMaterialization::test_materialize_with_multi_key_alias_index` — `passed` — `smoke` — `tests/test_materialize_subframe_index.py:L74`
- `test_materialize_subframe_index.py::TestSubframeIndexMaterialization::test_no_redundant_materialization` — `passed` — `smoke` — `tests/test_materialize_subframe_index.py:L161`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_3_recalibration_updates_target_rows_only` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L86`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration::test_v3_4_mask_fill_and_recalibration_compose` — `xfailed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L103`
- `test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationScope::test_v3_1_tpc_scope_corrected_non_tpc_untouched` — `passed` — `invariance` — `tests/test_phase_13_76_v12_calibration_scope.py:L73`

</details>

<details>
<summary><code>SUB.composite_key</code> — 38 owned pytest nodes</summary>

- `test_I14_join_index_invariance.py::TestI14JoinIndexInvariance::test_I14_2_composite_key_order_does_not_affect_values` — `passed` — `invariance` — `tests/test_I14_join_index_invariance.py:L59`
- `test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_2_composite_key_join_equals_pandas_merge` — `passed` — `invariance` — `tests/test_I8_subframe_alias_composition_invariance.py:L101`
- `test_composite_keys.py::TestAliasDataFrameRDFReexports::test_reexports_are_same_functions` — `passed` — `smoke` — `tests/test_composite_keys.py:L468`
- `test_composite_keys.py::TestAliasDataFrameRDFReexports::test_reexports_exist` — `passed` — `smoke` — `tests/test_composite_keys.py:L447`
- `test_composite_keys.py::TestCheckDenseOverflow::test_edge_case_just_safe` — `passed` — `smoke` — `tests/test_composite_keys.py:L72`
- `test_composite_keys.py::TestCheckDenseOverflow::test_large_values_unsafe` — `passed` — `smoke` — `tests/test_composite_keys.py:L63`
- `test_composite_keys.py::TestCheckDenseOverflow::test_single_value` — `passed` — `smoke` — `tests/test_composite_keys.py:L81`
- `test_composite_keys.py::TestCheckDenseOverflow::test_small_values_safe` — `passed` — `smoke` — `tests/test_composite_keys.py:L47`
- `test_composite_keys.py::TestCheckDenseOverflow::test_typical_tpc_case` — `passed` — `smoke` — `tests/test_composite_keys.py:L55`
- `test_composite_keys.py::TestComputeCompositeKeyAuto::test_auto_keys_match_correctly` — `passed` — `smoke` — `tests/test_composite_keys.py:L376`
- `test_composite_keys.py::TestComputeCompositeKeyAuto::test_auto_selects_dense_for_compact_data` — `passed` — `smoke` — `tests/test_composite_keys.py:L338`
- `test_composite_keys.py::TestComputeCompositeKeyAuto::test_auto_selects_sparse_for_large_gaps` — `passed` — `smoke` — `tests/test_composite_keys.py:L358`
- `test_composite_keys.py::TestComputeCompositeKeyAuto::test_explicit_dense_method` — `passed` — `smoke` — `tests/test_composite_keys.py:L398`
- `test_composite_keys.py::TestComputeCompositeKeyAuto::test_explicit_sparse_method` — `passed` — `smoke` — `tests/test_composite_keys.py:L411`
- `test_composite_keys.py::TestComputeCompositeKeyDense::test_result_is_int64` — `passed` — `smoke` — `tests/test_composite_keys.py:L245`
- `test_composite_keys.py::TestComputeCompositeKeyDense::test_single_column` — `passed` — `smoke` — `tests/test_composite_keys.py:L183`
- `test_composite_keys.py::TestComputeCompositeKeyDense::test_three_columns` — `passed` — `smoke` — `tests/test_composite_keys.py:L210`
- `test_composite_keys.py::TestComputeCompositeKeyDense::test_two_columns` — `passed` — `smoke` — `tests/test_composite_keys.py:L192`
- `test_composite_keys.py::TestComputeCompositeKeyDense::test_with_explicit_max_values` — `passed` — `smoke` — `tests/test_composite_keys.py:L229`
- `test_composite_keys.py::TestComputeCompositeKeySparse::test_basic_mapping` — `passed` — `smoke` — `tests/test_composite_keys.py:L261`
- `test_composite_keys.py::TestComputeCompositeKeySparse::test_result_is_int64` — `passed` — `smoke` — `tests/test_composite_keys.py:L323`
- `test_composite_keys.py::TestComputeCompositeKeySparse::test_shuffled_subframe` — `passed` — `smoke` — `tests/test_composite_keys.py:L280`
- `test_composite_keys.py::TestComputeCompositeKeySparse::test_sparse_with_gaps` — `passed` — `smoke` — `tests/test_composite_keys.py:L304`
- `test_composite_keys.py::TestGenerateDenseCppExpression::test_four_keys` — `passed` — `smoke` — `tests/test_composite_keys.py:L165`
- `test_composite_keys.py::TestGenerateDenseCppExpression::test_single_key` — `passed` — `smoke` — `tests/test_composite_keys.py:L144`
- `test_composite_keys.py::TestGenerateDenseCppExpression::test_three_keys` — `passed` — `smoke` — `tests/test_composite_keys.py:L158`
- `test_composite_keys.py::TestGenerateDenseCppExpression::test_tpc_style_keys` — `passed` — `smoke` — `tests/test_composite_keys.py:L172`
- `test_composite_keys.py::TestGenerateDenseCppExpression::test_two_keys` — `passed` — `smoke` — `tests/test_composite_keys.py:L151`
- `test_composite_keys.py::TestGetCompositeKeyColumnName::test_basic_name` — `passed` — `smoke` — `tests/test_composite_keys.py:L19`
- `test_composite_keys.py::TestGetCompositeKeyColumnName::test_long_name` — `passed` — `smoke` — `tests/test_composite_keys.py:L31`
- `test_composite_keys.py::TestGetCompositeKeyColumnName::test_short_name` — `passed` — `smoke` — `tests/test_composite_keys.py:L25`
- `test_composite_keys.py::TestGetCompositeKeyColumnName::test_underscore_name` — `passed` — `smoke` — `tests/test_composite_keys.py:L37`
- `test_composite_keys.py::TestModuleExports::test_all_exports_are_callable` — `passed` — `smoke` — `tests/test_composite_keys.py:L435`
- `test_composite_keys.py::TestModuleExports::test_all_exports_exist` — `passed` — `smoke` — `tests/test_composite_keys.py:L428`
- `test_composite_keys.py::TestShouldUseSparse::test_large_sparse_data` — `passed` — `smoke` — `tests/test_composite_keys.py:L104`
- `test_composite_keys.py::TestShouldUseSparse::test_overflow_triggers_sparse` — `passed` — `smoke` — `tests/test_composite_keys.py:L116`
- `test_composite_keys.py::TestShouldUseSparse::test_small_dense_data` — `passed` — `smoke` — `tests/test_composite_keys.py:L93`
- `test_composite_keys.py::TestShouldUseSparse::test_wasteful_triggers_sparse` — `passed` — `smoke` — `tests/test_composite_keys.py:L128`

</details>

<details>
<summary><code>SUB.auto_alias</code> — 11 owned pytest nodes</summary>

- `test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_3_auto_aliased_subframe_column_equals_manual_alias` — `passed` — `invariance` — `tests/test_I8_subframe_alias_composition_invariance.py:L130`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_add_alias_allows_subframe_reference` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L92`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_add_alias_rejects_self_reference` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L80`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_after_materialize` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L178`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_with_overwrite` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L112`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_auto_alias_without_overwrite` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L137`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_materialize_after_auto_alias_fix` — `xpassed` — `smoke` — `tests/test_self_referential_cycles.py:L211`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_no_self_referential_cycles` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L51`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_skip_existing_columns` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L23`
- `test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix::test_validate_no_cycles_method` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L161`
- `test_self_referential_cycles.py::TestOnlyUnmaterializedFix::test_only_unmaterialized_with_alias_and_column` — `passed` — `smoke` — `tests/test_self_referential_cycles.py:L247`

</details>

<details>
<summary><code>SUB.clone</code> — 0 owned pytest nodes</summary>

_(no tests claimed yet)_

</details>

<details>
<summary><code>SUB.nested</code> — 13 owned pytest nodes</summary>

- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_1_basic_roundtrip_no_subframes` — `passed` — `invariance` — `tests/test_E1_export_tree_roundtrip.py:L145`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_2_roundtrip_3_subframes` — `passed` — `invariance` — `tests/test_E1_export_tree_roundtrip.py:L160`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_3_roundtrip_production_scale_subframes` — `passed` — `invariance` — `tests/test_E1_export_tree_roundtrip.py:L184`
- `test_E1_export_tree_roundtrip.py::TestExportTreeMetadataRoundtrip::test_E1_4_roundtrip_timing_report` — `passed` — `invariance` — `tests/test_E1_export_tree_roundtrip.py:L243`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_1_single_tfile_open_per_export` — `passed` — `invariance` — `tests/test_E2_export_tree_fix_a.py:L73`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_2_nested_subframe_roundtrip` — `passed` — `invariance` — `tests/test_E2_export_tree_fix_a.py:L115`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_3_read_tree_backward_compatibility` — `passed` — `invariance` — `tests/test_E2_export_tree_fix_a.py:L197`
- `test_E2_export_tree_fix_a.py::TestE2ExportTreeFixA::test_E2_4_standalone_write_metadata_to_root` — `passed` — `invariance` — `tests/test_E2_export_tree_fix_a.py:L245`
- `test_alias_dataframe.py::TestExportTreeColumns::test_export_tree_columns_missing_raises` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2513`
- `test_alias_dataframe.py::TestExportTreeColumns::test_export_tree_columns_subset` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2488`
- `test_alias_dataframe.py::TestExportTreeColumns::test_export_tree_columns_warns_subframes` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2526`
- `test_alias_dataframe.py::TestExportTreeColumns::test_fill_value_none_preserves_inf` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2566`
- `test_alias_dataframe.py::TestExportTreeColumns::test_fill_value_replaces_inf_nan` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L2555`

</details>

<details>
<summary><code>SUB.multilevel</code> — 11 owned pytest nodes</summary>

- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_0_single_level_backward_compat` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L106`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_10_add_alias_no_false_self_ref` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L354`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_1_two_level_resolution` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L130`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_2_nested_roundtrip` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L159`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_3_missing_keys_nan` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L185`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_4_dematerialize_then_recover` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L229`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_5_nested_in_compound_expression` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L252`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_6_method_chain_preserved` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L271`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_7_cycle_detection` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L297`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_8_three_level_chain` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L327`
- `test_N1_recursive_subframe_invariance.py::TestN1MultiLevelResolution::test_N1_9_draw_nested_subframe` — `passed` — `invariance` — `tests/test_N1_recursive_subframe_invariance.py:L344`

</details>

## SCHEMA

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **SCHEMA.export_import** — Schema export & import (JSON) | 242 | 242 | 0 | 0 | 0 | 0 | 0 | 6 |
| ✅ | **SCHEMA.root_persistence** — ROOT file persistence | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| ☑️ | **SCHEMA.validation** — Schema validation | 22 | 22 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **SCHEMA.versioning** — Schema versioning & migration | 6 | 6 | 0 | 0 | 0 | 0 | 0 |  |

### Supporting tests

<details>
<summary><code>SCHEMA.export_import</code> — 242 owned pytest nodes</summary>

- `test_I12_metadata_persistence_invariance.py::TestI12MetadataPersistenceInvariance::test_I12_1_axis_title_survives_schema_roundtrip` — `passed` — `invariance` — `tests/test_I12_metadata_persistence_invariance.py:L51`
- `test_I12_metadata_persistence_invariance.py::TestI12MetadataPersistenceInvariance::test_I12_2_bulk_metadata_equals_individual_metadata_calls` — `passed` — `invariance` — `tests/test_I12_metadata_persistence_invariance.py:L104`
- `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_1_json_schema_roundtrip_preserves_aliases_and_subframes` — `passed` — `invariance` — `tests/test_I5_schema_roundtrip_invariance.py:L164`
- `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_2_root_tree_roundtrip_preserves_full_schema` — `passed` — `invariance` — `tests/test_I5_schema_roundtrip_invariance.py:L245`
- `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_3_json_and_root_paths_produce_semantically_equal_schemas` — `passed` — `invariance` — `tests/test_I5_schema_roundtrip_invariance.py:L323`
- `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_4_export_schema_is_idempotent` — `passed` — `invariance` — `tests/test_I5_schema_roundtrip_invariance.py:L392`
- `test_alias_data_frame_schema.py::TestAddAliasSchema::test_add_alias_overwrites_existing` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L307`
- `test_alias_data_frame_schema.py::TestAddAliasSchema::test_add_alias_with_constant_writes_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L302`
- `test_alias_data_frame_schema.py::TestAddAliasSchema::test_add_alias_with_dtype_writes_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L297`
- `test_alias_data_frame_schema.py::TestAddAliasSchema::test_add_alias_writes_to_schema_columns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L291`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L172`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_excludes_aliases_without_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L182`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L168`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_reflects_add_alias_with_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L176`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L195`
- `test_alias_data_frame_schema.py::TestAliasDtypesProperty::test_alias_dtypes_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L189`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_only_includes_columns_with_expr` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L154`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_property_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L121`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_property_reads_from_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L131`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_property_reflects_add_alias` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L125`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L117`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L143`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L138`
- `test_alias_data_frame_schema.py::TestAliasesProperty::test_aliases_setter_raises_attribute_error_with_replacement` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L149`
- `test_alias_data_frame_schema.py::TestApplyAliases::test_apply_aliases_registers_multiple` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L483`
- `test_alias_data_frame_schema.py::TestApplyAliases::test_apply_aliases_requires_expr` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L507`
- `test_alias_data_frame_schema.py::TestApplyAliases::test_apply_aliases_sets_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L500`
- `test_alias_data_frame_schema.py::TestApplyAliases::test_apply_aliases_sets_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L493`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_converts_column` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L522`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_missing_column_ignores` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L557`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_missing_column_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L547`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_missing_column_warns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L552`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_multiple_columns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L528`
- `test_alias_data_frame_schema.py::TestApplyDtypes::test_apply_dtypes_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L540`
- `test_alias_data_frame_schema.py::TestCompressionInfoProperty::test_compression_info_has_meta_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L254`
- `test_alias_data_frame_schema.py::TestCompressionInfoProperty::test_compression_info_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L250`
- `test_alias_data_frame_schema.py::TestCompressionInfoProperty::test_compression_info_references_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L258`
- `test_alias_data_frame_schema.py::TestCompressionInfoProperty::test_compression_info_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L274`
- `test_alias_data_frame_schema.py::TestCompressionInfoProperty::test_compression_info_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L265`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L213`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_includes_schema_constants` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L234`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_property_returns_set` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L209`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_reflects_add_alias_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L217`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_restore_method_works` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L228`
- `test_alias_data_frame_schema.py::TestConstantAliasesProperty::test_constant_aliases_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L222`
- `test_alias_data_frame_schema.py::TestEdgeCases::test_aliases_setter_raises_with_empty_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L635`
- `test_alias_data_frame_schema.py::TestEdgeCases::test_constant_aliases_setter_raises_with_empty_set` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L640`
- `test_alias_data_frame_schema.py::TestEdgeCases::test_empty_dataframe` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L622`
- `test_alias_data_frame_schema.py::TestEdgeCases::test_update_schema_empty_update` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L628`
- `test_alias_data_frame_schema.py::TestMutationSafety::test_alias_dtypes_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L704`
- `test_alias_data_frame_schema.py::TestMutationSafety::test_aliases_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L691`
- `test_alias_data_frame_schema.py::TestMutationSafety::test_compression_info_is_reference_by_design` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L734`
- `test_alias_data_frame_schema.py::TestMutationSafety::test_constant_aliases_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L719`
- `test_alias_data_frame_schema.py::TestMutationSafety::test_schema_property_is_deep_copy` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L747`
- `test_alias_data_frame_schema.py::TestRegisterSubframeSchema::test_register_subframe_with_list_index` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L332`
- `test_alias_data_frame_schema.py::TestRegisterSubframeSchema::test_register_subframe_writes_to_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L321`
- `test_alias_data_frame_schema.py::TestSchemaIntegration::test_aliases_property_works_with_materialize` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L654`
- `test_alias_data_frame_schema.py::TestSchemaIntegration::test_schema_preserved_after_get_alias_series` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L675`
- `test_alias_data_frame_schema.py::TestSchemaIntegration::test_schema_with_chained_aliases` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L664`
- `test_alias_data_frame_schema.py::TestSchemaRoundtrip::test_add_alias_then_read_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L570`
- `test_alias_data_frame_schema.py::TestSchemaRoundtrip::test_schema_survives_materialize` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L584`
- `test_alias_data_frame_schema.py::TestSchemaStructure::test_schema_compression_has_meta` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L83`
- `test_alias_data_frame_schema.py::TestSchemaStructure::test_schema_initialized_with_correct_structure` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L73`
- `test_alias_data_frame_schema.py::TestSchemaStructure::test_schema_property_includes_all_sections` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L102`
- `test_alias_data_frame_schema.py::TestSchemaStructure::test_schema_property_returns_deep_copy` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L90`
- `test_alias_data_frame_schema.py::TestSubframeRegistryHasSubframe::test_has_subframe_false_when_empty` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L602`
- `test_alias_data_frame_schema.py::TestSubframeRegistryHasSubframe::test_has_subframe_true_after_register` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L606`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_adds_column_spec` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L347`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_applies_dtype_to_physical_column` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L369`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_errors_warn` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L432`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_no_apply_keeps_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L380`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_partial_update` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L357`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_updates_compression` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L391`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_updates_subframes` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L400`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_validate_false_skips_validation` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L423`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_with_invalid_dtype_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L409`
- `test_alias_data_frame_schema.py::TestUpdateSchema::test_update_schema_with_non_string_expr_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L416`
- `test_alias_data_frame_schema.py::TestValidateSchemaUpdate::test_validate_accepts_defined_subframe_reference` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L458`
- `test_alias_data_frame_schema.py::TestValidateSchemaUpdate::test_validate_rejects_undefined_subframe_reference` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L451`
- `test_alias_data_frame_schema.py::TestValidateSchemaUpdate::test_validate_subframe_requires_index` — `passed` — `smoke` — `tests/test_alias_data_frame_schema.py:L468`
- `test_alias_data_frame_schema_v2.py::TestAddAliasSchema::test_add_alias_overwrites_existing` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L307`
- `test_alias_data_frame_schema_v2.py::TestAddAliasSchema::test_add_alias_with_constant_writes_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L302`
- `test_alias_data_frame_schema_v2.py::TestAddAliasSchema::test_add_alias_with_dtype_writes_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L297`
- `test_alias_data_frame_schema_v2.py::TestAddAliasSchema::test_add_alias_writes_to_schema_columns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L291`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L172`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_excludes_aliases_without_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L182`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L168`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_reflects_add_alias_with_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L176`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L195`
- `test_alias_data_frame_schema_v2.py::TestAliasDtypesProperty::test_alias_dtypes_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L189`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_only_includes_columns_with_expr` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L154`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_property_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L121`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_property_reads_from_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L131`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_property_reflects_add_alias` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L125`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L117`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L143`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L138`
- `test_alias_data_frame_schema_v2.py::TestAliasesProperty::test_aliases_setter_raises_attribute_error_with_replacement` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L149`
- `test_alias_data_frame_schema_v2.py::TestApplyAliases::test_apply_aliases_registers_multiple` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L483`
- `test_alias_data_frame_schema_v2.py::TestApplyAliases::test_apply_aliases_requires_expr` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L507`
- `test_alias_data_frame_schema_v2.py::TestApplyAliases::test_apply_aliases_sets_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L500`
- `test_alias_data_frame_schema_v2.py::TestApplyAliases::test_apply_aliases_sets_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L493`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_converts_column` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L522`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_missing_column_ignores` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L557`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_missing_column_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L547`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_missing_column_warns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L552`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_multiple_columns` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L528`
- `test_alias_data_frame_schema_v2.py::TestApplyDtypes::test_apply_dtypes_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L540`
- `test_alias_data_frame_schema_v2.py::TestCompressionInfoProperty::test_compression_info_has_meta_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L254`
- `test_alias_data_frame_schema_v2.py::TestCompressionInfoProperty::test_compression_info_property_returns_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L250`
- `test_alias_data_frame_schema_v2.py::TestCompressionInfoProperty::test_compression_info_references_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L258`
- `test_alias_data_frame_schema_v2.py::TestCompressionInfoProperty::test_compression_info_restore_method_updates_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L274`
- `test_alias_data_frame_schema_v2.py::TestCompressionInfoProperty::test_compression_info_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L265`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_empty_initially` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L213`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_includes_schema_constants` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L234`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_property_returns_set` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L209`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_reflects_add_alias_constant` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L217`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_restore_method_works` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L228`
- `test_alias_data_frame_schema_v2.py::TestConstantAliasesProperty::test_constant_aliases_setter_raises_attribute_error` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L222`
- `test_alias_data_frame_schema_v2.py::TestEdgeCases::test_aliases_setter_raises_with_empty_dict` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L635`
- `test_alias_data_frame_schema_v2.py::TestEdgeCases::test_constant_aliases_setter_raises_with_empty_set` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L640`
- `test_alias_data_frame_schema_v2.py::TestEdgeCases::test_empty_dataframe` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L622`
- `test_alias_data_frame_schema_v2.py::TestEdgeCases::test_update_schema_empty_update` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L628`
- `test_alias_data_frame_schema_v2.py::TestMutationSafety::test_alias_dtypes_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L704`
- `test_alias_data_frame_schema_v2.py::TestMutationSafety::test_aliases_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L691`
- `test_alias_data_frame_schema_v2.py::TestMutationSafety::test_compression_info_is_reference_by_design` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L734`
- `test_alias_data_frame_schema_v2.py::TestMutationSafety::test_constant_aliases_mutation_does_not_affect_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L719`
- `test_alias_data_frame_schema_v2.py::TestMutationSafety::test_schema_property_is_deep_copy` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L747`
- `test_alias_data_frame_schema_v2.py::TestRegisterSubframeSchema::test_register_subframe_with_list_index` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L332`
- `test_alias_data_frame_schema_v2.py::TestRegisterSubframeSchema::test_register_subframe_writes_to_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L321`
- `test_alias_data_frame_schema_v2.py::TestSchemaIntegration::test_aliases_property_works_with_materialize` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L654`
- `test_alias_data_frame_schema_v2.py::TestSchemaIntegration::test_schema_preserved_after_get_alias_series` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L675`
- `test_alias_data_frame_schema_v2.py::TestSchemaIntegration::test_schema_with_chained_aliases` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L664`
- `test_alias_data_frame_schema_v2.py::TestSchemaRoundtrip::test_add_alias_then_read_schema` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L570`
- `test_alias_data_frame_schema_v2.py::TestSchemaRoundtrip::test_schema_survives_materialize` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L584`
- `test_alias_data_frame_schema_v2.py::TestSchemaStructure::test_schema_compression_has_meta` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L83`
- `test_alias_data_frame_schema_v2.py::TestSchemaStructure::test_schema_initialized_with_correct_structure` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L73`
- `test_alias_data_frame_schema_v2.py::TestSchemaStructure::test_schema_property_includes_all_sections` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L102`
- `test_alias_data_frame_schema_v2.py::TestSchemaStructure::test_schema_property_returns_deep_copy` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L90`
- `test_alias_data_frame_schema_v2.py::TestSubframeRegistryHasSubframe::test_has_subframe_false_when_empty` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L602`
- `test_alias_data_frame_schema_v2.py::TestSubframeRegistryHasSubframe::test_has_subframe_true_after_register` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L606`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_adds_column_spec` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L347`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_applies_dtype_to_physical_column` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L369`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_errors_warn` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L432`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_no_apply_keeps_dtype` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L380`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_partial_update` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L357`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_updates_compression` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L391`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_updates_subframes` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L400`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_validate_false_skips_validation` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L423`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_with_invalid_dtype_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L409`
- `test_alias_data_frame_schema_v2.py::TestUpdateSchema::test_update_schema_with_non_string_expr_raises` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L416`
- `test_alias_data_frame_schema_v2.py::TestValidateSchemaUpdate::test_validate_accepts_defined_subframe_reference` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L458`
- `test_alias_data_frame_schema_v2.py::TestValidateSchemaUpdate::test_validate_rejects_undefined_subframe_reference` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L451`
- `test_alias_data_frame_schema_v2.py::TestValidateSchemaUpdate::test_validate_subframe_requires_index` — `passed` — `smoke` — `tests/test_alias_data_frame_schema_v2.py:L468`
- `test_data_schema.py::TestApplySchema::test_apply_schema_aliases` — `passed` — `smoke` — `tests/test_data_schema.py:L202`
- `test_data_schema.py::TestApplySchema::test_apply_schema_dtypes` — `passed` — `smoke` — `tests/test_data_schema.py:L183`
- `test_data_schema.py::TestConvertDtypes::test_convert_dtypes_batch` — `passed` — `smoke` — `tests/test_data_schema.py:L237`
- `test_data_schema.py::TestConvertDtypes::test_convert_dtypes_pattern` — `passed` — `smoke` — `tests/test_data_schema.py:L252`
- `test_data_schema.py::TestDescribeData::test_describe_core_only` — `passed` — `smoke` — `tests/test_data_schema.py:L79`
- `test_data_schema.py::TestDescribeData::test_describe_sort_by_memory` — `passed` — `smoke` — `tests/test_data_schema.py:L99`
- `test_data_schema.py::TestDescribeData::test_describe_with_source` — `passed` — `smoke` — `tests/test_data_schema.py:L89`
- `test_data_schema.py::TestDescribeSchema::test_describe_schema_overview` — `passed` — `smoke` — `tests/test_data_schema.py:L113`
- `test_data_schema.py::TestDescribeSchema::test_describe_schema_with_verbosity` — `passed` — `smoke` — `tests/test_data_schema.py:L120`
- `test_data_schema.py::TestFromSchema::test_from_schema_creates_empty_df` — `passed` — `smoke` — `tests/test_data_schema.py:L224`
- `test_data_schema.py::TestSchemaSerialization::test_export_schema` — `passed` — `smoke` — `tests/test_data_schema.py:L146`
- `test_data_schema.py::TestSchemaSerialization::test_save_load_roundtrip` — `passed` — `smoke` — `tests/test_data_schema.py:L163`
- `test_data_schema.py::TestSelectData::test_mutually_exclusive_filters` — `passed` — `smoke` — `tests/test_data_schema.py:L70`
- `test_data_schema.py::TestSelectData::test_select_all` — `passed` — `smoke` — `tests/test_data_schema.py:L39`
- `test_data_schema.py::TestSelectData::test_select_by_dtype` — `passed` — `smoke` — `tests/test_data_schema.py:L45`
- `test_data_schema.py::TestSelectData::test_select_by_pattern` — `passed` — `smoke` — `tests/test_data_schema.py:L65`
- `test_data_schema.py::TestSelectData::test_select_multiple_dtypes` — `passed` — `smoke` — `tests/test_data_schema.py:L50`
- `test_data_schema.py::TestSelectData::test_select_only_aliases` — `passed` — `smoke` — `tests/test_data_schema.py:L60`
- `test_data_schema.py::TestSelectData::test_select_only_physical` — `passed` — `smoke` — `tests/test_data_schema.py:L55`
- `test_data_schema.py::TestSelectSchema::test_select_by_dtype` — `passed` — `smoke` — `tests/test_data_schema.py:L132`
- `test_data_schema.py::TestSelectSchema::test_select_subframe_refs` — `passed` — `smoke` — `tests/test_data_schema.py:L137`
- `test_data_schema.py::TestSubframeAliasVsMaterialized::test_subframe_alias_handles_missing_keys` — `passed` — `smoke` — `tests/test_data_schema.py:L300`
- `test_data_schema.py::TestSubframeAliasVsMaterialized::test_subframe_alias_matches_materialized` — `passed` — `smoke` — `tests/test_data_schema.py:L271`
- `test_schema_definition_vs_record.py::TestCycleDetection::test_cycle_detected_when_compression_target_is_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L275`
- `test_schema_definition_vs_record.py::TestCycleDetection::test_implicit_compression_detection` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L310`
- `test_schema_definition_vs_record.py::TestCycleDetection::test_no_cycle_after_proper_compression` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L298`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_convenience_methods_match_explicit_params` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L149`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_definition_schema_before_compression` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L135`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_definition_schema_compression_target_is_physical` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L83`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_definition_schema_no_state_fields` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L100`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_definition_schema_no_storage_columns` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L92`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_record_schema_compression_target_is_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L109`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_record_schema_has_state_fields` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L126`
- `test_schema_definition_vs_record.py::TestDefinitionVsRecordExport::test_record_schema_has_storage_columns` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L118`
- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_new_format_schema_no_warning` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L365`
- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_old_format_schema_emits_warning` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L335`
- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_old_schema_still_loads` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L393`
- `test_schema_definition_vs_record.py::TestFullWorkflow::test_definition_schema_apply_compress_workflow` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L537`
- `test_schema_definition_vs_record.py::TestFullWorkflow::test_strict_validation_before_export` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L566`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_different_schema_raises_error` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L428`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_reuse_mode_idempotent` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L461`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_same_schema_idempotent` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L449`
- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_false_skips_data_validation` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L179`
- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_false_still_validates_schema_structure` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L190`
- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_true_detects_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L169`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_permissive_allows_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L233`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_overrides_explicit_params` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L243`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_rejects_missing_columns` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L222`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_rejects_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L212`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_valid_schema_passes_strict` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L257`
- `test_schema_definition_vs_record.py::TestValidationOutputStructure::test_compression_targets_populated` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L506`
- `test_schema_definition_vs_record.py::TestValidationOutputStructure::test_info_has_required_keys` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L491`
- `test_schema_definition_vs_record.py::TestValidationOutputStructure::test_output_has_required_keys` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L480`
- `test_schema_definition_vs_record.py::TestValidationOutputStructure::test_pending_aliases_structure` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L516`
- `test_schema_export_v2.py::TestColumnOrderingByGroups::test_alphabetic_within_group_sort` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L152`
- `test_schema_export_v2.py::TestColumnOrderingByGroups::test_empty_groups_returns_original` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L191`
- `test_schema_export_v2.py::TestColumnOrderingByGroups::test_groups_ordered_first` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L129`
- `test_schema_export_v2.py::TestColumnOrderingByGroups::test_missing_columns_in_group_ignored` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L182`
- `test_schema_export_v2.py::TestColumnOrderingByGroups::test_schema_order_preserves_group_list` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L167`
- `test_schema_export_v2.py::TestColumnSpecExport::test_alias_column_has_expr` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L216`
- `test_schema_export_v2.py::TestColumnSpecExport::test_constant_only_if_true` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L240`
- `test_schema_export_v2.py::TestColumnSpecExport::test_dtype_from_dataframe` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L251`
- `test_schema_export_v2.py::TestColumnSpecExport::test_metadata_preserved` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L225`
- `test_schema_export_v2.py::TestColumnSpecExport::test_numpy_dtype_conversion` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L259`
- `test_schema_export_v2.py::TestColumnSpecExport::test_physical_column_no_expr_null` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L207`
- `test_schema_export_v2.py::TestIndexColumnRepair::test_does_not_repair_single_element` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L298`
- `test_schema_export_v2.py::TestIndexColumnRepair::test_does_not_repair_valid_multikey` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L290`
- `test_schema_export_v2.py::TestIndexColumnRepair::test_handles_string_input` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L306`
- `test_schema_export_v2.py::TestIndexColumnRepair::test_repairs_corrupted_index` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L281`
- `test_schema_export_v2.py::TestJqQueryability::test_filter_aliases_by_expr` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L475`
- `test_schema_export_v2.py::TestJqQueryability::test_filter_columns_with_units` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L490`
- `test_schema_export_v2.py::TestJqQueryability::test_uniform_dtype_access` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L462`
- `test_schema_export_v2.py::TestSchemaLoading::test_converts_string_index_to_list` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L386`
- `test_schema_export_v2.py::TestSchemaLoading::test_loads_v2_schema_unchanged` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L320`
- `test_schema_export_v2.py::TestSchemaLoading::test_normalizes_v1_schema` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L340`
- `test_schema_export_v2.py::TestSchemaLoading::test_repairs_corrupted_subframe_index` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L364`
- `test_schema_export_v2.py::TestSchemaRoundTrip::test_round_trip_preserves_data` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L414`
- `test_schema_export_v2.py::TestSmartJsonFormatting::test_groups_format_correctly` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L91`
- `test_schema_export_v2.py::TestSmartJsonFormatting::test_long_entries_are_expanded` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L76`
- `test_schema_export_v2.py::TestSmartJsonFormatting::test_short_entries_stay_on_one_line` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L62`
- `test_schema_export_v2.py::TestSmartJsonFormatting::test_valid_json_output` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L105`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_export_schema_v2_roundtrip_with_subframes` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L691`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_multiple_subframes_all_populated` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L727`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_subframe_columns_in_exported_schema` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L607`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_subframe_describe_schema_works` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L641`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_subframe_empty_schema_simulates_root_load` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L559`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_subframe_schema_populated_on_register` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L521`
- `test_schema_export_v2.py::TestSubframeSchemaPopulation::test_subframe_with_existing_schema_not_overwritten` — `passed` — `smoke` — `tests/test_schema_export_v2.py:L663`

</details>

<details>
<summary><code>SCHEMA.root_persistence</code> — 5 owned pytest nodes</summary>

- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_1_read_tree_eager_equals_read_tree_lazy` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L59`
- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_2_explicit_ensure_branches_equals_implicit_access` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L111`
- `test_I17_full_pipeline_invariance.py::TestI17FullPipelineInvariance::test_I17_1_register_export_read_materialize_roundtrip` — `passed` — `invariance` — `tests/test_I17_full_pipeline_invariance.py:L47`
- `test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_2_root_tree_roundtrip_preserves_full_schema` — `passed` — `invariance` — `tests/test_I5_schema_roundtrip_invariance.py:L245`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip` — `passed` — `invariance` — `tests/test_polynomial_persistence.py:L184`

</details>

<details>
<summary><code>SCHEMA.validation</code> — 22 owned pytest nodes</summary>

- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_false_skips_data_validation` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L179`
- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_false_still_validates_schema_structure` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L190`
- `test_schema_definition_vs_record.py::TestValidateSchemaCheckData::test_check_data_true_detects_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L169`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_permissive_allows_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L233`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_overrides_explicit_params` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L243`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_rejects_missing_columns` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L222`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_strict_rejects_pending_alias` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L212`
- `test_schema_definition_vs_record.py::TestValidateSchemaStrict::test_valid_schema_passes_strict` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L257`
- `test_validation_display_adf.py::TestAddValidationIndicator::test_fail_indicator_red` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L157`
- `test_validation_display_adf.py::TestAddValidationIndicator::test_pass_indicator_green` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L144`
- `test_validation_display_adf.py::TestAddValidationSummary::test_summary_contains_overall_status` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L178`
- `test_validation_display_adf.py::TestAddValidationSummary::test_summary_contains_per_column_metrics` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L197`
- `test_validation_display_adf.py::TestComputeStatistics::test_delta_statistics_included` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L115`
- `test_validation_display_adf.py::TestComputeStatistics::test_pull_values_approximately_normal` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L103`
- `test_validation_display_adf.py::TestComputeStatistics::test_returns_dict_structure` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L89`
- `test_validation_display_adf.py::TestComputeStatistics::test_unknown_fit_returns_empty` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L127`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_all_parameters_combined` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L342`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_default_no_annotations` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L321`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_statistics_adds_annotations` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L238`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_summary_adds_panel` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L294`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_validation_adds_indicator` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L266`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_statistics_returned_in_results` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L225`

</details>

<details>
<summary><code>SCHEMA.versioning</code> — 6 owned pytest nodes</summary>

- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_new_format_schema_no_warning` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L365`
- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_old_format_schema_emits_warning` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L335`
- `test_schema_definition_vs_record.py::TestDeprecationWarning::test_old_schema_still_loads` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L393`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_different_schema_raises_error` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L428`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_reuse_mode_idempotent` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L461`
- `test_schema_definition_vs_record.py::TestSchemaChangeProtection::test_same_schema_idempotent` — `passed` — `smoke` — `tests/test_schema_definition_vs_record.py:L449`

</details>

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

### Supporting tests

<details>
<summary><code>FUNC.register_function</code> — 8 owned pytest nodes</summary>

- `test_I9_registered_function_invariance.py::TestI9RegisteredFunctionInvariance::test_I9_1_register_function_one_arg_equals_direct_computation` — `passed` — `invariance` — `tests/test_I9_registered_function_invariance.py:L66`
- `test_I9_registered_function_invariance.py::TestI9RegisteredFunctionInvariance::test_I9_2_register_function_two_arg_equals_direct_computation` — `passed` — `invariance` — `tests/test_I9_registered_function_invariance.py:L107`
- `test_polynomial_spec.py::TestRegisterFunction::test_backward_compatibility` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L202`
- `test_polynomial_spec.py::TestRegisterFunction::test_basic` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L171`
- `test_polynomial_spec.py::TestRegisterFunction::test_collision_raises` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L184`
- `test_polynomial_spec.py::TestRegisterFunction::test_overwrite` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L192`
- `test_polynomial_spec.py::TestRepr::test_full` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L262`
- `test_polynomial_spec.py::TestRepr::test_sparse` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L268`

</details>

<details>
<summary><code>FUNC.polynomial</code> — 20 owned pytest nodes</summary>

- `test_polynomial_spec.py::TestBasisExpressions::test_2d` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L41`
- `test_polynomial_spec.py::TestBasisExpressions::test_4d` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L36`
- `test_polynomial_spec.py::TestBasisExpressions::test_columns_degrees_mismatch_raises` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L82`
- `test_polynomial_spec.py::TestBasisExpressions::test_constant_term_present` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L75`
- `test_polynomial_spec.py::TestBasisExpressions::test_degree_332` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L31`
- `test_polynomial_spec.py::TestBasisExpressions::test_expression_format` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L65`
- `test_polynomial_spec.py::TestBasisExpressions::test_key_name_format` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L55`
- `test_polynomial_spec.py::TestBasisExpressions::test_sparse` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L46`
- `test_polynomial_spec.py::TestBasisExpressions::test_tuple_format` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L87`
- `test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_flat_vs_subframe_coeffs` — `passed` — `invariance` — `tests/test_polynomial_spec.py:L318`
- `test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_polynomial_vs_numpy` — `passed` — `invariance` — `tests/test_polynomial_spec.py:L283`
- `test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_schema_roundtrip_values` — `passed` — `invariance` — `tests/test_polynomial_spec.py:L345`
- `test_polynomial_spec.py::TestRegisterPolynomial::test_end_to_end` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L235`
- `test_polynomial_spec.py::TestRootExpression::test_basic` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L149`
- `test_polynomial_spec.py::TestRootExpression::test_zero_coeffs_skipped` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L158`
- `test_polynomial_spec.py::TestSchemaRoundtrip::test_full_roundtrip` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L102`
- `test_polynomial_spec.py::TestSchemaRoundtrip::test_schema_is_json_serializable` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L135`
- `test_polynomial_spec.py::TestSchemaRoundtrip::test_sparse_roundtrip` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L112`
- `test_polynomial_spec.py::TestSchemaRoundtrip::test_terms_omitted_for_full` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L122`
- `test_polynomial_spec.py::TestSchemaRoundtrip::test_terms_present_for_sparse` — `passed` — `smoke` — `tests/test_polynomial_spec.py:L128`

</details>

<details>
<summary><code>FUNC.evaluator</code> — 24 owned pytest nodes</summary>

- `test_register_evaluator.py::TestRegisterEvaluatorBasic::test_evaluator_in_expression` — `passed` — `smoke` — `tests/test_register_evaluator.py:L142`
- `test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_and_evaluate` — `passed` — `smoke` — `tests/test_register_evaluator.py:L109`
- `test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_single_column` — `passed` — `smoke` — `tests/test_register_evaluator.py:L120`
- `test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_three_columns` — `passed` — `smoke` — `tests/test_register_evaluator.py:L131`
- `test_register_evaluator.py::TestRegisterEvaluatorCollision::test_collision_raises` — `passed` — `smoke` — `tests/test_register_evaluator.py:L161`
- `test_register_evaluator.py::TestRegisterEvaluatorCollision::test_no_collision_with_columns` — `passed` — `smoke` — `tests/test_register_evaluator.py:L182`
- `test_register_evaluator.py::TestRegisterEvaluatorCollision::test_overwrite_replaces` — `passed` — `smoke` — `tests/test_register_evaluator.py:L169`
- `test_register_evaluator.py::TestRegisterEvaluatorComposition::test_chained_corrections` — `passed` — `smoke` — `tests/test_register_evaluator.py:L323`
- `test_register_evaluator.py::TestRegisterEvaluatorComposition::test_evaluator_minus_column` — `passed` — `smoke` — `tests/test_register_evaluator.py:L299`
- `test_register_evaluator.py::TestRegisterEvaluatorComposition::test_evaluator_with_alias_dependency` — `passed` — `smoke` — `tests/test_register_evaluator.py:L310`
- `test_register_evaluator.py::TestRegisterEvaluatorComposition::test_two_evaluators_summed` — `passed` — `smoke` — `tests/test_register_evaluator.py:L286`
- `test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_backward_compatibility` — `passed` — `smoke` — `tests/test_register_evaluator.py:L438`
- `test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_evaluator_with_nan_input` — `passed` — `smoke` — `tests/test_register_evaluator.py:L405`
- `test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_schema_stores_contract` — `passed` — `smoke` — `tests/test_register_evaluator.py:L418`
- `test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_schema_with_predictor_columns` — `passed` — `smoke` — `tests/test_register_evaluator.py:L429`
- `test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_alias_vs_direct` — `passed` — `invariance` — `tests/test_register_evaluator.py:L346`
- `test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_large_data` — `passed` — `invariance` — `tests/test_register_evaluator.py:L362`
- `test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_multi_predictor` — `passed` — `invariance` — `tests/test_register_evaluator.py:L378`
- `test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_multi_predictor_dz` — `passed` — `smoke` — `tests/test_register_evaluator.py:L240`
- `test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_multi_predictor_no_selection_raises` — `passed` — `smoke` — `tests/test_register_evaluator.py:L255`
- `test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_multi_predictor_with_selection` — `passed` — `smoke` — `tests/test_register_evaluator.py:L225`
- `test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_single_dict_predictor_no_selection_ok` — `passed` — `smoke` — `tests/test_register_evaluator.py:L267`
- `test_register_evaluator.py::TestRegisterEvaluatorValidation::test_no_evaluate_method_raises` — `passed` — `smoke` — `tests/test_register_evaluator.py:L202`
- `test_register_evaluator.py::TestRegisterEvaluatorValidation::test_wrong_arg_count_raises` — `passed` — `smoke` — `tests/test_register_evaluator.py:L208`

</details>

<details>
<summary><code>FUNC.ml_model</code> — 27 owned pytest nodes</summary>

- `test_phase_13_69_ml_model.py::test_ML_10_multi_output_predict_once` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L272`
- `test_phase_13_69_ml_model.py::test_ML_11_cache_invalidation_on_release` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L304`
- `test_phase_13_69_ml_model.py::test_ML_11_cache_invalidation_on_write` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L285`
- `test_phase_13_69_ml_model.py::test_ML_12_deregister_full_cleanup_and_reregister` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L320`
- `test_phase_13_69_ml_model.py::test_ML_13_save_model_canonical_bytes` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L336`
- `test_phase_13_69_ml_model.py::test_ML_14_format_auto_disambiguation` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L350`
- `test_phase_13_69_ml_model.py::test_ML_15_scikit_randomforest_via_onnx` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L370`
- `test_phase_13_69_ml_model.py::test_ML_16_invariance_alias_vs_prefilled_column` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L397`
- `test_phase_13_69_ml_model.py::test_ML_17_per_row_vector_output_refused` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L415`
- `test_phase_13_69_ml_model.py::test_ML_18_draw_prediction_in_slots[eager]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L460`
- `test_phase_13_69_ml_model.py::test_ML_18_draw_prediction_in_slots[lazy]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L460`
- `test_phase_13_69_ml_model.py::test_ML_19_prediction_across_all_draw_surfaces[eager]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L481`
- `test_phase_13_69_ml_model.py::test_ML_19_prediction_across_all_draw_surfaces[lazy]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L481`
- `test_phase_13_69_ml_model.py::test_ML_1_register_creates_usable_lazy_alias` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L77`
- `test_phase_13_69_ml_model.py::test_ML_2_native_json_path_matches_booster` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L93`
- `test_phase_13_69_ml_model.py::test_ML_2_prediction_equals_direct_onnxruntime` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L85`
- `test_phase_13_69_ml_model.py::test_ML_3_surface_symmetry_with_ordinary_alias` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L103`
- `test_phase_13_69_ml_model.py::test_ML_4_embed_roundtrip[eager]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L117`
- `test_phase_13_69_ml_model.py::test_ML_4_embed_roundtrip[lazy]` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L117`
- `test_phase_13_69_ml_model.py::test_ML_5_external_reference_roundtrip_and_relocation` — `passed` — `invariance` — `tests/test_phase_13_69_ml_model.py:L143`
- `test_phase_13_69_ml_model.py::test_ML_6_load_from_root_and_missing_name` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L184`
- `test_phase_13_69_ml_model.py::test_ML_8_duplicate_refuse` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L201`
- `test_phase_13_69_ml_model.py::test_ML_8_md5_mismatch_refuse` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L225`
- `test_phase_13_69_ml_model.py::test_ML_8_missing_runtime_message_shape` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L240`
- `test_phase_13_69_ml_model.py::test_ML_8_subframe_input_deferral` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L217`
- `test_phase_13_69_ml_model.py::test_ML_8_unresolvable_auto_format` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L209`
- `test_phase_13_69_ml_model.py::test_ML_9_lazy_exact_load_of_input_closure` — `passed` — `smoke` — `tests/test_phase_13_69_ml_model.py:L256`

</details>

<details>
<summary><code>FUNC.persistence</code> — 9 owned pytest nodes</summary>

- `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_before_after_schema_roundtrip` — `passed` — `invariance` — `tests/test_polynomial_persistence.py:L156`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip` — `passed` — `invariance` — `tests/test_polynomial_persistence.py:L184`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_reconstruct_missing_subframe_skips` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L135`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_reconstruct_polynomial_from_schema` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L87`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_reconstruct_skips_evaluator` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L111`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_save_load_schema_roundtrip` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L77`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_schema_contains_registered_functions` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L53`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_schema_has_polynomial_spec` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L61`
- `test_polynomial_persistence.py::TestPolynomialPersistence::test_schema_json_serializable` — `passed` — `smoke` — `tests/test_polynomial_persistence.py:L69`

</details>

<details>
<summary><code>FUNC.regression_metadata</code> — 4 owned pytest nodes</summary>

- `test_R1_metadata_persistence_invariance.py::TestR1RegressionMetadataPersistence::test_R1_1_metadata_dict_survives_schema_roundtrip` — `passed` — `invariance` — `tests/test_R1_metadata_persistence_invariance.py:L65`
- `test_R2_recalibration_invariance.py::TestR2RecalibrationInvariance::test_R2_1_subframe_swap_produces_new_calibration_values` — `passed` — `invariance` — `tests/test_R2_recalibration_invariance.py:L65`
- `test_R2_recalibration_invariance.py::TestR2RecalibrationInvariance::test_R2_2_revert_to_previous_subframe_bit_exact` — `passed` — `invariance` — `tests/test_R2_recalibration_invariance.py:L114`
- `test_R2_recalibration_invariance.py::TestR2RecalibrationInvariance::test_R2_3_lazy_subframe_reference_validated_at_register` — `passed` — `invariance` — `tests/test_R2_recalibration_invariance.py:L161`

</details>

<details>
<summary><code>FUNC.evaluator_from_metadata</code> — 6 owned pytest nodes</summary>

- `test_R3_missing_bin_safety_invariance.py::TestR3MissingBinSafetyInvariance::test_R3_1_unpopulated_bin_returns_nan_not_silent_value` — `passed` — `invariance` — `tests/test_R3_missing_bin_safety_invariance.py:L55`
- `test_R3_missing_bin_safety_invariance.py::TestR3MissingBinSafetyInvariance::test_R3_2_bounds_clamp_still_nans_interior_missing_bins` — `passed` — `invariance` — `tests/test_R3_missing_bin_safety_invariance.py:L105`
- `test_R3_missing_bin_safety_invariance.py::TestR3MissingBinSafetyInvariance::test_R3_3_bounds_nan_out_of_range_returns_nan` — `passed` — `invariance` — `tests/test_R3_missing_bin_safety_invariance.py:L145`
- `test_R4_registration_contract_invariance.py::TestR4RegistrationContractInvariance::test_R4_1_missing_required_coefficient_column_raises` — `passed` — `invariance` — `tests/test_R4_registration_contract_invariance.py:L39`
- `test_R4_registration_contract_invariance.py::TestR4RegistrationContractInvariance::test_R4_2_group_columns_mismatch_raises` — `passed` — `invariance` — `tests/test_R4_registration_contract_invariance.py:L93`
- `test_R5_remap_correctness_invariance.py::TestR5RemapCorrectnessInvariance::test_R5_1_noncontiguous_natural_labels_match_pd_merge` — `passed` — `invariance` — `tests/test_R5_remap_correctness_invariance.py:L30`

</details>

<details>
<summary><code>FUNC.regression_persistence</code> — 1 owned pytest nodes</summary>

- `test_R1_2_evaluator_roundtrip_invariance.py::TestR1_2EvaluatorRoundtripInvariance::test_R1_2_evaluator_equivalence_after_roundtrip` — `passed` — `invariance` — `tests/test_R1_2_evaluator_roundtrip_invariance.py:L34`

</details>

## ALIAS

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **ALIAS.source_scoped** — Source-scoped alias resolution (PHASE_13_73) — add_alias with source=<subframe> binds BARE fit formulas (e.g. GB meta['formulas']) to a registered subframe by rewriting bare names to Subframe.name. Resolution is AST-based (names are tokens, not text), so the substring-collision class fixed in 13.72 is structurally impossible. Per-name order: Attribute/Call-func untouched; source index_columns exempt (R1a); present in BOTH source and parent -> loud shadow refusal (R1); in source -> qualify; in parent-universe -> leave bare; otherwise -> refuse at registration (R2). The parent-universe includes lazy available-but-unloaded branches, struct members and registered functions (F-1), so binding works on a lazy frame before any branch is loaded. Batch binding is a loop over add_alias (PHASE_13_73_FIX removed the add_aliases helper: it could not express per-alias dtypes and was a second entry point for one concept). The rewritten alias stays user-readable and source-qualified; the existing subframe-join path performs the join (no second evaluation route). source=None is bit-identical to pre-13.73. Composes with 13.70 vector aliases. | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 3 |

### Supporting tests

<details>
<summary><code>ALIAS.source_scoped</code> — 13 owned pytest nodes</summary>

- `test_phase_13_73_source_scoped_alias.py::test_T10_flagship_matches_handwritten` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L173`
- `test_phase_13_73_source_scoped_alias.py::test_T11_vector_alias_composition` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L196`
- `test_phase_13_73_source_scoped_alias.py::test_T1_resolution_call_args_and_attribute_passthrough` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L33`
- `test_phase_13_73_source_scoped_alias.py::test_T2_lazy_unloaded_branch_resolves` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L44`
- `test_phase_13_73_source_scoped_alias.py::test_T3_R1_shadow_raises` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L62`
- `test_phase_13_73_source_scoped_alias.py::test_T4_R1a_index_exemption_by_membership` — `passed` — `invariance` — `tests/test_phase_13_73_source_scoped_alias.py:L72`
- `test_phase_13_73_source_scoped_alias.py::test_T5_R2_unknown_token_raises` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L86`
- `test_phase_13_73_source_scoped_alias.py::test_T6_backward_compat_source_none` — `passed` — `invariance` — `tests/test_phase_13_73_source_scoped_alias.py:L95`
- `test_phase_13_73_source_scoped_alias.py::test_T7_prefix_collision_structurally_impossible` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L106`
- `test_phase_13_73_source_scoped_alias.py::test_T8_batch_binding_via_loop_with_per_alias_dtype` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L119`
- `test_phase_13_73_source_scoped_alias.py::test_T8b_bad_formula_in_a_loop_raises_at_that_formula` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L140`
- `test_phase_13_73_source_scoped_alias.py::test_T9_eager_lazy_chain_parity` — `passed` — `invariance` — `tests/test_phase_13_73_source_scoped_alias.py:L155`
- `test_phase_13_73_source_scoped_alias.py::test_unknown_source_refused` — `passed` — `smoke` — `tests/test_phase_13_73_source_scoped_alias.py:L207`

</details>

## DIAGNOSTICS

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **DIAGNOSTICS.lazy_state** — User-facing lazy-state diagnostic (PHASE_13_71) — adf.describe_lazy() prints (or returns via as_dict) the main lazy reader's entries, available/loaded branches, DataFrame columns, and available-but-not-loaded set, plus a per-lazy-subframe block (available/loaded counts + index columns from _subframe_lazy_config); diagnostic-only with NO loading or materialization side effects; bounded output via max_items with a '... (+N more)' suffix; tolerant of LazyTreeReader/LazyChainReader attribute differences via getattr defaults; reports the subframe block even when the main frame is eager. Not scope this phase: loading/ensuring branches, HTML/JSON output, rich repr. | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 1 |

### Supporting tests

<details>
<summary><code>DIAGNOSTICS.lazy_state</code> — 7 owned pytest nodes</summary>

- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_1_eager_reports_not_lazy` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L44`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_2_lazy_reports_branches` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L52`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_3_no_side_effects` — `passed` — `invariance` — `tests/test_phase_13_71_describe_lazy.py:L66`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_4_max_items_truncates` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L80`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_5_lazy_subframe_block` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L88`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_6_chain_reader_tolerant` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L101`
- `test_phase_13_71_describe_lazy.py::test_LAZYDESC_7_as_dict_form` — `passed` — `smoke` — `tests/test_phase_13_71_describe_lazy.py:L114`

</details>

## DRAWING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **DRAW.execution** — draw() with auto-materialization | 70 | 67 | 0 | 0 | 2 | 0 | 1 | 17 |
| 🧨 | **DRAW.batch** — draw_batch() & draw_figures() | 45 | 44 | 0 | 0 | 1 | 0 | 0 |  |
| ✅ | **DRAW.subframe_resolution** — Subframe resolution in draw — expression-slot discovery, owner-qualified materialization/cleanup, request-level/spec-slot attribution, per-public-call evidence isolation, and terminal plan/state reconciliation | 88 | 88 | 0 | 0 | 0 | 0 | 0 | 41 |
| ✅ | **DRAW.compound_expr** — Lazy materialization of compound expressions | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **DRAW.invariance** — Draw vs materialize invariance | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 🧨 | **DRAW.slot_grid** — Systematic draw slot symmetry grid<br><sub>Checks every architect-supported draw slot across column, alias, expression, subframe and struct forms in eager/lazy modes. Smoke records reachability and live gaps; invariance compares executable cells against an independent plain-data reference and checks eager/lazy equivalence.</sub><br><sub>Surface: 88 cells; KNOWN_GAP=18, PASSING=70; seams=4</sub> | 245 | 207 | 0 | 0 | 38 | 0 | 0 | 110 |

### Supporting tests

<details>
<summary><code>DRAW.execution</code> — 70 owned pytest nodes</summary>

- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_1_draw_lazy_compound_expression_equals_explicit_materialize` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L168`
- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_2_draw_subframe_column_equals_explicit_alias` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L243`
- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_3_draw_with_selection_equals_pre_filtered_draw` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L328`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_1_draw_accepts_all_documented_kwargs` — `passed` — `invariance` — `tests/test_K1_vector_draw_kwarg_diagnostic.py:L145`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_2_draw_forwards_kwargs_to_dfdraw` — `passed` — `invariance` — `tests/test_K1_vector_draw_kwarg_diagnostic.py:L181`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_3_draw_batch_forwards_batch_kwargs` — `xfailed` — `invariance` — `tests/test_K1_vector_draw_kwarg_diagnostic.py:L311`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_4_draw_figures_forwards_figure_kwargs` — `skipped` — `invariance` — `tests/test_K1_vector_draw_kwarg_diagnostic.py:L358`
- `test_K1_vector_draw_kwarg_diagnostic.py::TestK1VectorDrawKwargDiagnostic::test_K1_5_vector_expression_each_call_gets_full_kwargs` — `passed` — `invariance` — `tests/test_K1_vector_draw_kwarg_diagnostic.py:L418`
- `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_1_scalar_groupby_baseline` — `passed` — `invariance` — `tests/test_K2_vector_draw_end_to_end.py:L147`
- `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_2_vector_groupby_main_legend_bounded` — `passed` — `invariance` — `tests/test_K2_vector_draw_end_to_end.py:L193`
- `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_3_production_reproducer_mirror` — `xfailed` — `invariance` — `tests/test_K2_vector_draw_end_to_end.py:L260`
- `test_K2_vector_draw_end_to_end.py::TestK2VectorDrawEndToEnd::test_K2_4_title_not_duplicated_per_vector_iteration` — `passed` — `invariance` — `tests/test_K2_vector_draw_end_to_end.py:L313`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_1_selection_vector_alias_production_expression` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L109`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_2_weights_vector_with_selection_vector` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L136`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_3_facet_by_alias_column_materialized` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L161`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_4_idempotent_repeated_draw` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L182`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_5_draw_batch_selection_vector_alias` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L210`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_6_draw_figures_weights_vector_alias` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L249`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_7_facet_by_channel_enum_not_materialized` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L344`
- `test_V1_vector_kwargs_alias_materialization.py::TestV1_VectorKwargsAliasMaterialization::test_V1_8_auto_force_vector_compose_outer_for_single_Y` — `passed` — `invariance` — `tests/test_V1_vector_kwargs_alias_materialization.py:L289`
- `test_draw_chain_integration.py::TestChainLazy::test_chain_lazy_alias` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L250`
- `test_draw_chain_integration.py::TestChainLazy::test_chain_lazy_basic` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L233`
- `test_draw_chain_integration.py::TestChainLazy::test_chain_lazy_has_all_files` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L241`
- `test_draw_chain_integration.py::TestChainLazy::test_chain_preserves_invariant` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L264`
- `test_draw_chain_integration.py::TestChainWithSubframeChain::test_chain_subframe_chain_correct_values` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L418`
- `test_draw_chain_integration.py::TestChainWithSubframeChain::test_chain_subframe_chain_per_run` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L447`
- `test_draw_chain_integration.py::TestChainWithSubframeChain::test_chain_with_subframe_chain_basic` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L402`
- `test_draw_chain_integration.py::TestChainWithSubframeSingle::test_chain_subframe_applied_to_all_files` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L370`
- `test_draw_chain_integration.py::TestChainWithSubframeSingle::test_chain_with_single_subframe` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L347`
- `test_draw_chain_integration.py::TestDrawIntegration::test_draw_chain_lazy` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L587`
- `test_draw_chain_integration.py::TestDrawIntegration::test_draw_single_eager` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L573`
- `test_draw_chain_integration.py::TestDrawIntegration::test_draw_with_subframe` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L601`
- `test_draw_chain_integration.py::TestIntegrationErrors::test_chain_missing_file_pattern_raises` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L488`
- `test_draw_chain_integration.py::TestIntegrationErrors::test_subframe_chain_missing_files_raises` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L495`
- `test_draw_chain_integration.py::TestIntegrationErrors::test_subframe_chain_missing_index_column_raises` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L509`
- `test_draw_chain_integration.py::TestPerformanceSanity::test_lazy_chain_doesnt_load_all_branches` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L531`
- `test_draw_chain_integration.py::TestPerformanceSanity::test_subframe_not_loaded_if_not_needed` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L545`
- `test_draw_chain_integration.py::TestSingleEagerBaseline::test_eager_alias_chain` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L140`
- `test_draw_chain_integration.py::TestSingleEagerBaseline::test_eager_alias_works` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L129`
- `test_draw_chain_integration.py::TestSingleEagerBaseline::test_eager_load_basic` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L121`
- `test_draw_chain_integration.py::TestSingleEagerBaseline::test_eager_subframe_works` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L153`
- `test_draw_chain_integration.py::TestSingleLazy::test_lazy_alias_triggers_load` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L201`
- `test_draw_chain_integration.py::TestSingleLazy::test_lazy_ensure_branches` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L192`
- `test_draw_chain_integration.py::TestSingleLazy::test_lazy_load_basic` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L183`
- `test_draw_chain_integration.py::TestSingleLazy::test_lazy_only_loads_needed` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L214`
- `test_draw_chain_integration.py::TestWithLazySubframe::test_lazy_subframe_correct_values` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L318`
- `test_draw_chain_integration.py::TestWithLazySubframe::test_lazy_subframe_loads_on_demand` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L296`
- `test_draw_chain_integration.py::TestWithLazySubframe::test_lazy_subframe_registration` — `passed` — `smoke` — `tests/test_draw_chain_integration.py:L283`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_no_reload_existing` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L303`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_pre_loads_all_branches` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L222`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_single_io_operation` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L252`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_verbose_output` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L276`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_with_defaults` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L287`
- `test_draw_lazy_integration.py::TestDrawBatchLazyLoading::test_batch_with_selections` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L238`
- `test_draw_lazy_integration.py::TestDrawEagerUnchanged::test_eager_draw_batch_works` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L354`
- `test_draw_lazy_integration.py::TestDrawEagerUnchanged::test_eager_draw_works` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L348`
- `test_draw_lazy_integration.py::TestDrawEagerUnchanged::test_eager_no_lazy_reader` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L361`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_auto_loads_branches` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L104`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_doesnt_reload_existing` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L174`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_incremental_loading` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L155`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_single_var_loads_branch` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L118`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_with_alias_loads_dependencies` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L190`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_with_color_loads_column` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L146`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_with_group_by_loads_column` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L137`
- `test_draw_lazy_integration.py::TestDrawLazyLoading::test_draw_with_selection_loads_all` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L127`
- `test_draw_lazy_integration.py::TestEdgeCases::test_draw_complex_expression` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L398`
- `test_draw_lazy_integration.py::TestEdgeCases::test_draw_empty_lazy_then_load` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L386`
- `test_draw_lazy_integration.py::TestLazyLoadingWithoutDraw::test_ensure_branches_loads_correctly` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L434`
- `test_draw_lazy_integration.py::TestLazyLoadingWithoutDraw::test_get_required_branches_for_draw` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L421`
- `test_draw_lazy_integration.py::TestLazyLoadingWithoutDraw::test_incremental_ensure` — `passed` — `smoke` — `tests/test_draw_lazy_integration.py:L449`

</details>

<details>
<summary><code>DRAW.batch</code> — 45 owned pytest nodes</summary>

- `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S1_draw_materializes_selection_alias` — `xfailed` — `smoke` — `tests/test_S1_draw_selection_alias.py:L62`
- `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S1b_draw_lazy_materializes_selection_alias` — `passed` — `smoke` — `tests/test_S1_draw_selection_alias.py:L77`
- `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S2_draw_batch_materializes_selection_alias` — `passed` — `smoke` — `tests/test_S1_draw_selection_alias.py:L88`
- `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S3_draw_figures_materializes_selection_alias` — `passed` — `smoke` — `tests/test_S1_draw_selection_alias.py:L113`
- `test_S1_draw_selection_alias.py::TestDrawSelectionAliasBug::test_S4_draw_batch_materializes_weights_alias` — `passed` — `smoke` — `tests/test_S1_draw_selection_alias.py:L147`
- `test_draw_figures.py::TestBasic::test_auto_name_generation` — `passed` — `smoke` — `tests/test_draw_figures.py:L171`
- `test_draw_figures.py::TestBasic::test_multiple_figures` — `passed` — `smoke` — `tests/test_draw_figures.py:L141`
- `test_draw_figures.py::TestBasic::test_returns_stats` — `passed` — `smoke` — `tests/test_draw_figures.py:L161`
- `test_draw_figures.py::TestBasic::test_short_form_expansion` — `passed` — `smoke` — `tests/test_draw_figures.py:L154`
- `test_draw_figures.py::TestBasic::test_short_form_valid` — `passed` — `smoke` — `tests/test_draw_figures.py:L117`
- `test_draw_figures.py::TestBasic::test_single_figure_multiple_plots` — `passed` — `smoke` — `tests/test_draw_figures.py:L133`
- `test_draw_figures.py::TestBasic::test_single_figure_single_plot` — `passed` — `smoke` — `tests/test_draw_figures.py:L123`
- `test_draw_figures.py::TestEntrySelection::test_entry_begin_end` — `passed` — `smoke` — `tests/test_draw_figures.py:L325`
- `test_draw_figures.py::TestEntrySelection::test_entry_mask_boolean` — `passed` — `smoke` — `tests/test_draw_figures.py:L334`
- `test_draw_figures.py::TestEntrySelection::test_entry_mask_integer` — `passed` — `smoke` — `tests/test_draw_figures.py:L343`
- `test_draw_figures.py::TestEntrySelection::test_max_entries_limits_rows` — `passed` — `smoke` — `tests/test_draw_figures.py:L315`
- `test_draw_figures.py::TestErrorHandling::test_error_text_on_failed_plot` — `passed` — `smoke` — `tests/test_draw_figures.py:L399`
- `test_draw_figures.py::TestErrorHandling::test_figure_error_returns_error_dict` — `passed` — `smoke` — `tests/test_draw_figures.py:L386`
- `test_draw_figures.py::TestErrorHandling::test_on_error_raise_stops` — `passed` — `smoke` — `tests/test_draw_figures.py:L379`
- `test_draw_figures.py::TestErrorHandling::test_on_error_skip_continues` — `passed` — `smoke` — `tests/test_draw_figures.py:L361`
- `test_draw_figures.py::TestFileIO::test_save_dir_creates_parents` — `passed` — `smoke` — `tests/test_draw_figures.py:L433`
- `test_draw_figures.py::TestFileIO::test_save_dir_prepended` — `passed` — `smoke` — `tests/test_draw_figures.py:L425`
- `test_draw_figures.py::TestFileIO::test_savefig_creates_file` — `passed` — `smoke` — `tests/test_draw_figures.py:L417`
- `test_draw_figures.py::TestFileIO::test_savefig_without_save_dir` — `passed` — `smoke` — `tests/test_draw_figures.py:L441`
- `test_draw_figures.py::TestIntegration::test_json_serializable_specs` — `passed` — `smoke` — `tests/test_draw_figures.py:L509`
- `test_draw_figures.py::TestIntegration::test_multi_figure_workflow` — `passed` — `smoke` — `tests/test_draw_figures.py:L488`
- `test_draw_figures.py::TestIntegration::test_qa_dashboard_pattern` — `passed` — `smoke` — `tests/test_draw_figures.py:L462`
- `test_draw_figures.py::TestLayout::test_figsize_custom` — `passed` — `smoke` — `tests/test_draw_figures.py:L234`
- `test_draw_figures.py::TestLayout::test_ncols_custom` — `passed` — `smoke` — `tests/test_draw_figures.py:L204`
- `test_draw_figures.py::TestLayout::test_ncols_default_2` — `passed` — `smoke` — `tests/test_draw_figures.py:L186`
- `test_draw_figures.py::TestLayout::test_suptitle_displayed` — `passed` — `smoke` — `tests/test_draw_figures.py:L225`
- `test_draw_figures.py::TestLayout::test_unused_axes_hidden` — `passed` — `smoke` — `tests/test_draw_figures.py:L212`
- `test_draw_figures.py::TestOptions::test_clear_after_cleanup` — `passed` — `smoke` — `tests/test_draw_figures.py:L284`
- `test_draw_figures.py::TestOptions::test_defaults_applied` — `passed` — `smoke` — `tests/test_draw_figures.py:L252`
- `test_draw_figures.py::TestOptions::test_kwargs_as_defaults` — `passed` — `smoke` — `tests/test_draw_figures.py:L300`
- `test_draw_figures.py::TestOptions::test_lazy_materialization` — `passed` — `smoke` — `tests/test_draw_figures.py:L273`
- `test_draw_figures.py::TestOptions::test_plot_overrides_defaults` — `passed` — `smoke` — `tests/test_draw_figures.py:L261`
- `test_draw_figures.py::TestOptions::test_verbose_false_quiet` — `passed` — `smoke` — `tests/test_draw_figures.py:L294`
- `test_draw_figures.py::TestValidation::test_plot_must_have_expr` — `passed` — `smoke` — `tests/test_draw_figures.py:L104`
- `test_draw_figures.py::TestValidation::test_plots_cannot_be_empty` — `passed` — `smoke` — `tests/test_draw_figures.py:L99`
- `test_draw_figures.py::TestValidation::test_plots_must_be_list` — `passed` — `smoke` — `tests/test_draw_figures.py:L94`
- `test_draw_figures.py::TestValidation::test_spec_must_be_dict` — `passed` — `smoke` — `tests/test_draw_figures.py:L84`
- `test_draw_figures.py::TestValidation::test_spec_must_have_plots` — `passed` — `smoke` — `tests/test_draw_figures.py:L89`
- `test_draw_figures.py::TestValidation::test_specs_cannot_be_empty` — `passed` — `smoke` — `tests/test_draw_figures.py:L79`
- `test_draw_figures.py::TestValidation::test_specs_must_be_list` — `passed` — `smoke` — `tests/test_draw_figures.py:L74`

</details>

<details>
<summary><code>DRAW.subframe_resolution</code> — 88 owned pytest nodes</summary>

- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_2_draw_subframe_column_equals_explicit_alias` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L243`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S10_alias_in_xy_expression` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L140`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S11_alias_as_y_in_expression` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L152`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S12_alias_as_sole_expression_histogram` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L161`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S13_alias_in_arithmetic_expression` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L170`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S14_alias_in_selection` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L179`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S15_raw_column_still_works_regression_guard` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L194`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S16_draw_batch_with_alias_subframe` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L211`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S17_draw_figures_with_alias_subframe` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L231`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S18_multilevel_alias_on_inner_subframe` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L251`
- `test_S10_draw_subframe_alias.py::TestS10_DrawSubframeAliasNotMaterialized::test_S19_cold_draw_no_workaround_used` — `passed` — `invariance` — `tests/test_S10_draw_subframe_alias.py:L269`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_1_draw_index_col_on_x_axis` — `passed` — `smoke` — `tests/test_S5_draw_index_col_collision.py:L60`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_2_draw_index_col_on_y_axis` — `passed` — `smoke` — `tests/test_S5_draw_index_col_collision.py:L69`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_3_draw_index_col_with_selection` — `passed` — `smoke` — `tests/test_S5_draw_index_col_collision.py:L78`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_4_draw_values_match_subframe` — `passed` — `invariance` — `tests/test_S5_draw_index_col_collision.py:L88`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_5_draw_figures_index_col` — `passed` — `smoke` — `tests/test_S5_draw_index_col_collision.py:L124`
- `test_S5_draw_index_col_collision.py::TestDrawIndexColCollision::test_S5_6_draw_both_axes_are_index_cols` — `passed` — `smoke` — `tests/test_S5_draw_index_col_collision.py:L138`
- `test_S6_draw_subframe_expression.py::TestDrawSubframeExpression::test_S6_dotted_ref_in_arithmetic_expr` — `passed` — `invariance` — `tests/test_S6_draw_subframe_expression.py:L73`
- `test_S6_draw_subframe_expression.py::TestDrawSubframeExpression::test_S7_dotted_ref_in_selection_arithmetic` — `passed` — `invariance` — `tests/test_S6_draw_subframe_expression.py:L84`
- `test_S6_draw_subframe_expression.py::TestDrawSubframeExpression::test_S8_standalone_dotted_ref_still_works` — `passed` — `invariance` — `tests/test_S6_draw_subframe_expression.py:L96`
- `test_S6_draw_subframe_expression.py::TestDrawSubframeExpression::test_S9_duplicate_index_no_expansion` — `passed` — `invariance` — `tests/test_S6_draw_subframe_expression.py:L106`
- `test_X1_subframe_metadata_propagation.py::TestX1_SingleLevelTitle::test_X1_single_level_title_via_flat_name` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L102`
- `test_X1_subframe_metadata_propagation.py::TestX2_MultiLevelTitle::test_X2_multi_level_title_via_flat_name` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L119`
- `test_X1_subframe_metadata_propagation.py::TestX3_GetColumnMetadataDispatch::test_X3_full_metadata_dispatched` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L136`
- `test_X1_subframe_metadata_propagation.py::TestX4_ParentPrecedence::test_X4_parent_override_wins` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L161`
- `test_X1_subframe_metadata_propagation.py::TestX5_NegativeBranch::test_X5_unknown_flat_name_returns_none` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L180`
- `test_X1_subframe_metadata_propagation.py::TestX6_EndToEndDraw::test_X6_draw_axis_label_from_subframe` — `passed` — `invariance` — `tests/test_X1_subframe_metadata_propagation.py:L200`
- `test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_multikey` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L206`
- `test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_multiple_plots` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L182`
- `test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_subframe_basic` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L157`
- `test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_subframe_conflict` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L194`
- `test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_subframe_in_selection` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L169`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_defaults_cascade` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L316`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_layout_tuple` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L330`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_mixed_subframe_and_local` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L289`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_multikey` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L303`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_multiple_figures` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L357`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_preserves_df` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L376`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_subframe_basic` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L264`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_subframe_conflict` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L277`
- `test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_subframe_in_selection` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L344`
- `test_draw_subframe_resolution.py::TestDrawSubframeEdgeCases::test_dot_in_non_subframe_name` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L240`
- `test_draw_subframe_resolution.py::TestDrawSubframeEdgeCases::test_draw_preserves_original_df` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L248`
- `test_draw_subframe_resolution.py::TestDrawSubframeEdgeCases::test_no_subframe_registered` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L226`
- `test_draw_subframe_resolution.py::TestDrawSubframeEdgeCases::test_subframe_column_not_found` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L233`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_auto_alias_subframe_conflict_skips` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L142`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_multiple_subframe_refs` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L122`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_no_conflict` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L72`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_values_correct` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L85`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_with_conflict` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L79`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_in_selection` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L97`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_multikey` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L129`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_multikey_with_entry_end` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L135`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_with_entry_end` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L104`
- `test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_with_entry_end_values` — `passed` — `smoke` — `tests/test_draw_subframe_resolution.py:L111`
- `test_phase_13_76_v12_history_invariance.py::TestV12RequestHistoryInvariance::test_o3_spec_permutation_preserves_valid_failure_semantics` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L265`
- `test_phase_13_76_v12_history_invariance.py::TestV12RequestHistoryInvariance::test_o4_tolerated_request_evidence_is_per_public_call[clean_then_tolerated]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L292`
- `test_phase_13_76_v12_history_invariance.py::TestV12RequestHistoryInvariance::test_o4_tolerated_request_evidence_is_per_public_call[tolerated_then_clean]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_history_invariance.py:L292`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_meta_1_source_label_alone_does_not_prove_alias_namespace_complete` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L321`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_meta_2_runtime_tolerated_unresolved_evidence_is_request_exact` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L410`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r1_child_alias_dependency_closure_is_planned[2]` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L124`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r1_child_alias_dependency_closure_is_planned[3]` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L124`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r1_child_alias_dependency_closure_is_planned[4]` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L124`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r1_negative_control_one_level_child_alias_remains_green` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L154`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r2_two_valid_same_owner_requests_cannot_hide_one_runtime_failure` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L164`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r3_metadata_free_bad_only_warn_uses_runtime_unresolved_evidence` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L183`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r4_mixed_valid_and_bad_warn_normal_execution_passes` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L216`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r5_cross_spec_tolerance_cannot_discharge_faulted_valid_spec` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L230`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r6_cross_slot_success_cannot_hide_faulted_valid_selection` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L250`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r7_exact_owned_reconciliation_still_rejects_missing_and_extra_owner_effects` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L272`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r8_unreferenced_unresolved_alias_does_not_enter_unrelated_plan` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L291`
- `test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract::test_r9_projection_failure_precedes_dfdraw_on_error_skip` — `passed` — `smoke` — `tests/test_phase_13_76_v12_reconciliation_contract.py:L306`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`

</details>

<details>
<summary><code>DRAW.compound_expr</code> — 13 owned pytest nodes</summary>

- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_1_draw_lazy_compound_expression_equals_explicit_materialize` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L168`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_draw_abs_alias_works` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L91`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_draw_arithmetic_alias_works` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L100`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_draw_selection_with_alias` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L108`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_invariance_lazy_vs_explicit` — `passed` — `invariance` — `tests/test_draw_lazy_compound.py:L117`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_alias_arithmetic` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L57`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_alias_in_abs` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L45`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_alias_in_group_by` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L70`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_alias_in_selection` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L64`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_alias_in_sqrt` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L51`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_excludes_functions` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L76`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_excludes_physical_columns` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L84`
- `test_draw_lazy_compound.py::TestDrawLazyCompoundExpr::test_parse_simple_alias` — `passed` — `smoke` — `tests/test_draw_lazy_compound.py:L39`

</details>

<details>
<summary><code>DRAW.invariance</code> — 18 owned pytest nodes</summary>

- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_1_draw_lazy_compound_expression_equals_explicit_materialize` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L168`
- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_2_draw_subframe_column_equals_explicit_alias` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L243`
- `test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_3_draw_with_selection_equals_pre_filtered_draw` — `passed` — `invariance` — `tests/test_I7_draw_path_invariance.py:L328`
- `test_draw_invariance.py::TestChainSubframeIntegration::test_chain_with_subframe_chain` — `passed` — `invariance` — `tests/test_draw_invariance.py:L277`
- `test_draw_invariance.py::TestCoreInvariants::test_alias_materialization_identical` — `passed` — `invariance` — `tests/test_draw_invariance.py:L173`
- `test_draw_invariance.py::TestCoreInvariants::test_chain_has_all_files` — `passed` — `invariance` — `tests/test_draw_invariance.py:L197`
- `test_draw_invariance.py::TestCoreInvariants::test_complex_alias_chain` — `passed` — `invariance` — `tests/test_draw_invariance.py:L211`
- `test_draw_invariance.py::TestCoreInvariants::test_eager_vs_lazy_data_identical` — `passed` — `invariance` — `tests/test_draw_invariance.py:L154`
- `test_draw_invariance.py::TestCoreInvariants::test_known_relationship_chain` — `passed` — `invariance` — `tests/test_draw_invariance.py:L140`
- `test_draw_invariance.py::TestCoreInvariants::test_known_relationship_single_file` — `passed` — `invariance` — `tests/test_draw_invariance.py:L127`
- `test_draw_invariance.py::TestDrawInvariance::test_draw_vs_materialize_identical` — `passed` — `invariance` — `tests/test_draw_invariance.py:L380`
- `test_draw_invariance.py::TestDrawInvariance::test_eager_vs_lazy_draw_identical` — `passed` — `invariance` — `tests/test_draw_invariance.py:L397`
- `test_draw_invariance.py::TestDtypePreservation::test_float32_preserved_lazy` — `passed` — `invariance` — `tests/test_draw_invariance.py:L319`
- `test_draw_invariance.py::TestDtypePreservation::test_int32_preserved_lazy` — `passed` — `invariance` — `tests/test_draw_invariance.py:L327`
- `test_draw_invariance.py::TestErrorScenarios::test_circular_alias_raises` — `passed` — `invariance` — `tests/test_draw_invariance.py:L358`
- `test_draw_invariance.py::TestErrorScenarios::test_invalid_alias_raises` — `passed` — `invariance` — `tests/test_draw_invariance.py:L350`
- `test_draw_invariance.py::TestErrorScenarios::test_missing_branch_raises` — `passed` — `invariance` — `tests/test_draw_invariance.py:L343`
- `test_draw_invariance.py::TestSubframeJoinCorrectness::test_sector_calibration_correct` — `passed` — `invariance` — `tests/test_draw_invariance.py:L242`

</details>

<details>
<summary><code>DRAW.slot_grid</code> — 245 owned pytest nodes</summary>

- `test_phase_13_79_slot_grid.py::test_b3_gate0_declared_product_is_complete` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L275`
- `test_phase_13_79_slot_grid.py::test_b3_gate0_falsifier_delete_one_row_is_caught` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L279`
- `test_phase_13_79_slot_grid.py::test_b3_gate1_grid_slots_equal_production_slot_authority` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L286`
- `test_phase_13_79_slot_grid.py::test_b3_gate2_all_five_dfdraw_forwarding_authorities_exist` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L300`
- `test_phase_13_79_slot_grid.py::test_b3_gate2_falsifier_new_column_reference_is_caught` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L294`
- `test_phase_13_79_slot_grid.py::test_b3_gate2_live_column_reference_inventory_is_fully_classified` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L290`
- `test_phase_13_79_slot_grid.py::test_b3_gate3_b0_equals_smoke_parameter_ids` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L304`
- `test_phase_13_79_slot_grid.py::test_b3_ratified_b0_schema_and_four_b4_seams_are_complete` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L310`
- `test_phase_13_79_slot_grid.py::test_slot_grid_contract_has_expected_known_gap_ownership` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L645`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[color-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[expr-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[facet_by-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[facet_by-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[facet_by-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[group_by-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection_vector-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection_vector-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection_vector-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection_vector-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[selection_vector-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights_vector-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights_vector-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights_vector-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights_vector-subframe-eager_parent_lazy_child]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_known_gap_signature[weights_vector-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L637`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-expression-eager]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-expression-lazy]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-struct-eager]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-struct-lazy]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-eager]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-lazy]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-struct-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-struct-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-subframe-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-subframe-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-alias-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-alias-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-column-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-column-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-expression-eager]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-expression-lazy]` — `passed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-struct-eager]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-struct-lazy]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-eager]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-eager_parent_lazy_child]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-lazy]` — `xfailed` — `smoke` — `tests/test_phase_13_79_slot_grid.py:L598`
- `test_phase_13_79_slot_grid_invariance.py::test_b3_gate3_b0_equals_smoke_equals_invariance_ids` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L445`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_facet_seam_falsifier_swapped_branch_order_is_caught[selection_vector]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L594`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_facet_seam_falsifier_swapped_branch_order_is_caught[weights_vector]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L594`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[selection_vector-facet_by]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L567`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[selection_vector-group_by]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L567`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[weights_vector-facet_by]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L567`
- `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[weights_vector-group_by]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L567`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[color-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[color-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[color-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[color-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[color-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[expr-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[expr-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[expr-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[expr-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[expr-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[facet_by-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[facet_by-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[facet_by-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[facet_by-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[group_by-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[group_by-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[group_by-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[group_by-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[group_by-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection_vector-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection_vector-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[selection_vector-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights-struct]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights-subframe]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights_vector-alias]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights_vector-column]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_cross_mode[weights_vector-expression]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L434`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-expression-eager]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-expression-lazy]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-struct-eager]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-struct-lazy]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-eager]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-lazy]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-struct-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-struct-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-subframe-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-subframe-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-alias-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-alias-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-column-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-column-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-expression-eager]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-expression-lazy]` — `passed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-struct-eager]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-struct-lazy]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-eager]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-eager_parent_lazy_child]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`
- `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-lazy]` — `xfailed` — `invariance` — `tests/test_phase_13_79_slot_grid_invariance.py:L405`

</details>

## COMPRESSION

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **COMP.roundtrip** — Compress/decompress roundtrip | 72 | 70 | 2 | 0 | 0 | 0 | 0 | 9 |
| ☑️ | **COMP.selection** — Compression method selection | 10 | 10 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **COMP.monitoring** — Compression quality monitoring | 15 | 15 | 0 | 0 | 0 | 0 | 0 |  |

### Supporting tests

<details>
<summary><code>COMP.roundtrip</code> — 72 owned pytest nodes</summary>

- `test_I11_compression_working_invariance.py::TestI11CompressionWorkingPathInvariance::test_I11_1_linear_compress_decompress_max_error_within_bit_budget` — `passed` — `invariance` — `tests/test_I11_compression_working_invariance.py:L79`
- `test_I11_compression_working_invariance.py::TestI11CompressionWorkingPathInvariance::test_I11_2_mixed_compressed_uncompressed_arithmetic` — `passed` — `invariance` — `tests/test_I11_compression_working_invariance.py:L121`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_backward_compatibility_no_compression_info` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L679`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_basic_compression_decompression` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L306`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_compress_alias_source` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L368`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_compressed_column_name_collision_raises_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L410`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_compression_is_idempotent` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L388`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_compression_with_precision_measurement` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L342`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_decompress_inplace` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L430`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_decompress_keep_compressed_false` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L454`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_get_compression_info` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L655`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_missing_compressed_column_raises_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L478`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_multiple_columns_compression` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L616`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_partial_failure_handling` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L501`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_roundtrip_export_import_tree` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L583`
- `test_alias_dataframe.py::TestAliasDataFrameCompression::test_roundtrip_save_load` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L538`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_all_columns_missing_error` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1605`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_all_columns_missing_warn` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1594`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_default_warn_mode` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1511`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_explicit_columns_subset` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1560`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_method_chaining_still_works` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1629`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_partial_missing_with_columns_param` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1614`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_return_summary_default_false` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1579`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_silent_ignore_mode` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1543`
- `test_alias_dataframe.py::TestCompressionOnMissing::test_strict_error_mode` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1532`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_backward_compatibility_old_files` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1216`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_collision_foreign_column` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1123`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_collision_other_schema` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1134`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_collision_same_schema_recompression` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1109`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_compress_all_schema_only_columns` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1147`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_compress_decompress_roundtrip_idempotent` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L872`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_decompress_with_keep_schema_false` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L827`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_decompression_is_idempotent` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L844`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_definition_schema_export_no_alias_for_compression_targets` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1026`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_definition_schema_infers_schema_only_state` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L997`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_definition_schema_roundtrip_compress` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1057`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_direct_compression_without_schema` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L779`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_double_compression_is_idempotent` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L858`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_export_schema_include_state_false` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L948`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_export_schema_include_state_false_no_leakage` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L967`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_export_schema_include_state_true` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L934`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_full_compression_cycle` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L793`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_get_compression_info_excludes_meta` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1171`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_invalid_state_transition_schema_only_to_decompress` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1204`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_is_compressed_helper` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1161`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_metadata_versioning` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L723`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_multiple_selective_calls` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1291`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_pattern1_pattern2_mixing` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1443`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_precision_measurement_with_state` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1185`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_real_world_incremental_compression_pattern2` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1408`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_record_schema_includes_aliases` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1088`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_schema_compressed_but_data_not` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L897`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_schema_from_info_helper` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1194`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_schema_only_definition` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L730`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_schema_only_then_compress` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L754`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_mode_errors_on_schema_change_when_compressed` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1326`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_mode_skips_same_schema_compressed` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1311`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_mode_updates_schema_for_schema_only` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1375`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_mode_validates_column_exists` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1351`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_mode_validates_columns_in_spec` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1367`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_selective_registration_from_spec` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1272`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_state_invariants_after_compress` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1226`
- `test_alias_dataframe.py::TestCompressionStateMachine::test_state_invariants_after_decompress` — `passed` — `smoke` — `tests/test_alias_dataframe.py:L1248`
- `test_invariance_compression.py::TestCompressionSummary::test_I4_count` — `passed` — `invariance` — `tests/test_invariance_compression.py:L366`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_1_linear_compression_roundtrip` — `passed` — `invariance` — `tests/test_invariance_compression.py:L158`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip` — `failed` — `invariance` — `tests/test_invariance_compression.py:L186`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip` — `failed` — `invariance` — `tests/test_invariance_compression.py:L213`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_4_sqrt_compression_roundtrip` — `passed` — `invariance` — `tests/test_invariance_compression.py:L241`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_5_compress_decompress_idempotent` — `passed` — `invariance` — `tests/test_invariance_compression.py:L263`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_6_schema_preserved_after_roundtrip` — `passed` — `invariance` — `tests/test_invariance_compression.py:L287`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_7_compressed_dtype_correct` — `passed` — `invariance` — `tests/test_invariance_compression.py:L308`
- `test_invariance_compression.py::TestInvarianceCompression::test_I4_8_multi_column_compression` — `passed` — `invariance` — `tests/test_invariance_compression.py:L325`

</details>

<details>
<summary><code>COMP.selection</code> — 10 owned pytest nodes</summary>

- `test_compression_pytest.py::TestCompressionConstants::test_verbosity_constants_defined` — `passed` — `smoke` — `tests/test_compression_pytest.py:L308`
- `test_compression_pytest.py::TestCompressionConstants::test_verbosity_constants_values` — `passed` — `smoke` — `tests/test_compression_pytest.py:L316`
- `test_compression_pytest.py::TestCompressionSelection::test_invalid_pattern` — `passed` — `smoke` — `tests/test_compression_pytest.py:L172`
- `test_compression_pytest.py::TestCompressionSelection::test_mutually_exclusive_filters` — `passed` — `smoke` — `tests/test_compression_pytest.py:L164`
- `test_compression_pytest.py::TestCompressionSelection::test_select_all` — `passed` — `smoke` — `tests/test_compression_pytest.py:L128`
- `test_compression_pytest.py::TestCompressionSelection::test_select_compressed_only` — `passed` — `smoke` — `tests/test_compression_pytest.py:L134`
- `test_compression_pytest.py::TestCompressionSelection::test_select_decompressed_only` — `passed` — `smoke` — `tests/test_compression_pytest.py:L140`
- `test_compression_pytest.py::TestCompressionSelection::test_select_failed_only` — `passed` — `smoke` — `tests/test_compression_pytest.py:L146`
- `test_compression_pytest.py::TestCompressionSelection::test_select_names` — `passed` — `smoke` — `tests/test_compression_pytest.py:L158`
- `test_compression_pytest.py::TestCompressionSelection::test_select_pattern` — `passed` — `smoke` — `tests/test_compression_pytest.py:L152`

</details>

<details>
<summary><code>COMP.monitoring</code> — 15 owned pytest nodes</summary>

- `test_compression_pytest.py::TestDescribeCompression::test_describe_all` — `passed` — `smoke` — `tests/test_compression_pytest.py:L242`
- `test_compression_pytest.py::TestDescribeCompression::test_describe_as_dict` — `passed` — `smoke` — `tests/test_compression_pytest.py:L259`
- `test_compression_pytest.py::TestDescribeCompression::test_describe_failed_as_dict` — `passed` — `smoke` — `tests/test_compression_pytest.py:L266`
- `test_compression_pytest.py::TestDescribeCompression::test_describe_failed_only` — `passed` — `smoke` — `tests/test_compression_pytest.py:L250`
- `test_compression_pytest.py::TestDescribeCompression::test_describe_with_pattern` — `passed` — `smoke` — `tests/test_compression_pytest.py:L275`
- `test_compression_pytest.py::TestDescribeCompression::test_verbosity_flags` — `passed` — `smoke` — `tests/test_compression_pytest.py:L284`
- `test_compression_pytest.py::TestMonitorChecks::test_absolute_monitor_fail` — `passed` — `smoke` — `tests/test_compression_pytest.py:L186`
- `test_compression_pytest.py::TestMonitorChecks::test_absolute_monitor_pass` — `passed` — `smoke` — `tests/test_compression_pytest.py:L181`
- `test_compression_pytest.py::TestMonitorChecks::test_function_monitor_fail` — `passed` — `smoke` — `tests/test_compression_pytest.py:L196`
- `test_compression_pytest.py::TestMonitorChecks::test_no_monitor` — `passed` — `smoke` — `tests/test_compression_pytest.py:L201`
- `test_compression_pytest.py::TestMonitorChecks::test_relative_monitor_pass` — `passed` — `smoke` — `tests/test_compression_pytest.py:L191`
- `test_compression_pytest.py::TestMonitorValues::test_absolute_monitor_value` — `passed` — `smoke` — `tests/test_compression_pytest.py:L210`
- `test_compression_pytest.py::TestMonitorValues::test_function_monitor_value` — `passed` — `smoke` — `tests/test_compression_pytest.py:L224`
- `test_compression_pytest.py::TestMonitorValues::test_no_monitor_value` — `passed` — `smoke` — `tests/test_compression_pytest.py:L231`
- `test_compression_pytest.py::TestMonitorValues::test_relative_monitor_value` — `passed` — `smoke` — `tests/test_compression_pytest.py:L217`

</details>

## BACKEND

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **BACK.arrow** — PyArrow compute & scatter | 120 | 120 | 0 | 0 | 0 | 0 | 0 |  |
| ✅ | **BACK.numba** — Numba JIT acceleration | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 3 |
| 🧨 | **BACK.invariance** — Backend equivalence (numpy vs arrow vs numba) | 14 | 13 | 1 | 0 | 0 | 0 | 0 | 13 |

### Supporting tests

<details>
<summary><code>BACK.arrow</code> — 120 owned pytest nodes</summary>

- `test_arrow_compute.py::TestArrowComputeMapper::test_arctan2_alias` — `passed` — `smoke` — `tests/test_arrow_compute.py:L145`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x * y-expected2]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x ** 2-expected4]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x ** y-expected5]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x + y-expected0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x - y-expected1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_arithmetic_operators[x / y-expected3]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L71`
- `test_arrow_compute.py::TestArrowComputeMapper::test_atan2` — `passed` — `smoke` — `tests/test_arrow_compute.py:L134`
- `test_arrow_compute.py::TestArrowComputeMapper::test_cache_hit` — `passed` — `smoke` — `tests/test_arrow_compute.py:L513`
- `test_arrow_compute.py::TestArrowComputeMapper::test_chained_comparison_raises` — `passed` — `smoke` — `tests/test_arrow_compute.py:L345`
- `test_arrow_compute.py::TestArrowComputeMapper::test_clear_cache` — `passed` — `smoke` — `tests/test_arrow_compute.py:L503`
- `test_arrow_compute.py::TestArrowComputeMapper::test_clip` — `passed` — `smoke` — `tests/test_arrow_compute.py:L278`
- `test_arrow_compute.py::TestArrowComputeMapper::test_clip_with_variables` — `passed` — `smoke` — `tests/test_arrow_compute.py:L286`
- `test_arrow_compute.py::TestArrowComputeMapper::test_clip_wrong_args_raises` — `passed` — `smoke` — `tests/test_arrow_compute.py:L298`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x != y-expected5]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x < y-expected1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x <= y-expected3]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x == y-expected4]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x > y-expected0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_comparison_operators[x >= y-expected2]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L339`
- `test_arrow_compute.py::TestArrowComputeMapper::test_complex_expression` — `passed` — `smoke` — `tests/test_arrow_compute.py:L361`
- `test_arrow_compute.py::TestArrowComputeMapper::test_exp` — `passed` — `smoke` — `tests/test_arrow_compute.py:L186`
- `test_arrow_compute.py::TestArrowComputeMapper::test_float_constant` — `passed` — `smoke` — `tests/test_arrow_compute.py:L399`
- `test_arrow_compute.py::TestArrowComputeMapper::test_floor_division` — `passed` — `smoke` — `tests/test_arrow_compute.py:L77`
- `test_arrow_compute.py::TestArrowComputeMapper::test_get_supported_functions` — `passed` — `smoke` — `tests/test_arrow_compute.py:L494`
- `test_arrow_compute.py::TestArrowComputeMapper::test_hyperbolic_functions[cosh]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L158`
- `test_arrow_compute.py::TestArrowComputeMapper::test_hyperbolic_functions[sinh]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L158`
- `test_arrow_compute.py::TestArrowComputeMapper::test_hyperbolic_functions[tanh]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L158`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[acosh-arccosh-test_values2]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[arccosh-arccosh-test_values3]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[arcsinh-arcsinh-test_values1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[arctanh-arctanh-test_values5]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[asinh-arcsinh-test_values0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_hyperbolic[atanh-arctanh-test_values4]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L174`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[acos-arccos]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[arccos-arccos]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[arcsin-arcsin]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[arctan-arctan]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[asin-arcsin]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_inverse_trig[atan-arctan]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L126`
- `test_arrow_compute.py::TestArrowComputeMapper::test_is_supported` — `passed` — `smoke` — `tests/test_arrow_compute.py:L486`
- `test_arrow_compute.py::TestArrowComputeMapper::test_log10` — `passed` — `smoke` — `tests/test_arrow_compute.py:L208`
- `test_arrow_compute.py::TestArrowComputeMapper::test_log1p` — `passed` — `smoke` — `tests/test_arrow_compute.py:L224`
- `test_arrow_compute.py::TestArrowComputeMapper::test_log2` — `passed` — `smoke` — `tests/test_arrow_compute.py:L216`
- `test_arrow_compute.py::TestArrowComputeMapper::test_log_maps_to_ln` — `passed` — `smoke` — `tests/test_arrow_compute.py:L194`
- `test_arrow_compute.py::TestArrowComputeMapper::test_math_e_constant` — `passed` — `smoke` — `tests/test_arrow_compute.py:L415`
- `test_arrow_compute.py::TestArrowComputeMapper::test_missing_variable_raises` — `passed` — `smoke` — `tests/test_arrow_compute.py:L475`
- `test_arrow_compute.py::TestArrowComputeMapper::test_modulo` — `passed` — `smoke` — `tests/test_arrow_compute.py:L85`
- `test_arrow_compute.py::TestArrowComputeMapper::test_nested_sqrt_sum_squares` — `passed` — `smoke` — `tests/test_arrow_compute.py:L354`
- `test_arrow_compute.py::TestArrowComputeMapper::test_nested_trig_expression` — `passed` — `smoke` — `tests/test_arrow_compute.py:L378`
- `test_arrow_compute.py::TestArrowComputeMapper::test_numeric_constant` — `passed` — `smoke` — `tests/test_arrow_compute.py:L391`
- `test_arrow_compute.py::TestArrowComputeMapper::test_numpy_pi_constant` — `passed` — `smoke` — `tests/test_arrow_compute.py:L407`
- `test_arrow_compute.py::TestArrowComputeMapper::test_pow_alias` — `passed` — `smoke` — `tests/test_arrow_compute.py:L248`
- `test_arrow_compute.py::TestArrowComputeMapper::test_power` — `passed` — `smoke` — `tests/test_arrow_compute.py:L240`
- `test_arrow_compute.py::TestArrowComputeMapper::test_real_tpc_expression` — `passed` — `smoke` — `tests/test_arrow_compute.py:L370`
- `test_arrow_compute.py::TestArrowComputeMapper::test_rounding_functions[abs-input_vals3-expected3]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L267`
- `test_arrow_compute.py::TestArrowComputeMapper::test_rounding_functions[ceil-input_vals1-expected1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L267`
- `test_arrow_compute.py::TestArrowComputeMapper::test_rounding_functions[floor-input_vals0-expected0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L267`
- `test_arrow_compute.py::TestArrowComputeMapper::test_rounding_functions[sign-input_vals4-expected4]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L267`
- `test_arrow_compute.py::TestArrowComputeMapper::test_rounding_functions[trunc-input_vals2-expected2]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L267`
- `test_arrow_compute.py::TestArrowComputeMapper::test_sqrt` — `passed` — `smoke` — `tests/test_arrow_compute.py:L232`
- `test_arrow_compute.py::TestArrowComputeMapper::test_subframe_column_dotted_key` — `passed` — `smoke` — `tests/test_arrow_compute.py:L427`
- `test_arrow_compute.py::TestArrowComputeMapper::test_subframe_column_nested_dict` — `passed` — `smoke` — `tests/test_arrow_compute.py:L437`
- `test_arrow_compute.py::TestArrowComputeMapper::test_syntax_error_raises` — `passed` — `smoke` — `tests/test_arrow_compute.py:L470`
- `test_arrow_compute.py::TestArrowComputeMapper::test_ternary_expression` — `passed` — `smoke` — `tests/test_arrow_compute.py:L453`
- `test_arrow_compute.py::TestArrowComputeMapper::test_trig_functions[cos]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L110`
- `test_arrow_compute.py::TestArrowComputeMapper::test_trig_functions[sin]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L110`
- `test_arrow_compute.py::TestArrowComputeMapper::test_trig_functions[tan]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L110`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_division[a_type0-b_type0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L530`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_division[a_type1-b_type1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L530`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_division[a_type2-b_type2]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L530`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_division[a_type3-b_type3]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L530`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_multiplication[a_type0-b_type0]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L545`
- `test_arrow_compute.py::TestArrowComputeMapper::test_type_promotion_multiplication[a_type1-b_type1]` — `passed` — `smoke` — `tests/test_arrow_compute.py:L545`
- `test_arrow_compute.py::TestArrowComputeMapper::test_unary_negation` — `passed` — `smoke` — `tests/test_arrow_compute.py:L91`
- `test_arrow_compute.py::TestArrowComputeMapper::test_unary_plus` — `passed` — `smoke` — `tests/test_arrow_compute.py:L98`
- `test_arrow_compute.py::TestArrowComputeMapper::test_unsupported_function_raises` — `passed` — `smoke` — `tests/test_arrow_compute.py:L465`
- `test_arrow_compute.py::TestArrowComputeMapper::test_where` — `passed` — `smoke` — `tests/test_arrow_compute.py:L307`
- `test_arrow_compute.py::TestArrowComputeMapper::test_where_with_comparison` — `passed` — `smoke` — `tests/test_arrow_compute.py:L319`
- `test_arrow_compute.py::TestEvaluateExpressionArrow::test_basic_evaluation` — `passed` — `smoke` — `tests/test_arrow_compute.py:L569`
- `test_arrow_compute.py::TestEvaluateExpressionArrow::test_fallback_called` — `passed` — `smoke` — `tests/test_arrow_compute.py:L576`
- `test_arrow_compute.py::TestEvaluateExpressionArrow::test_pandas_series_input` — `passed` — `smoke` — `tests/test_arrow_compute.py:L591`
- `test_arrow_compute.py::TestIsArrowAvailable::test_is_arrow_available` — `passed` — `smoke` — `tests/test_arrow_compute.py:L604`
- `test_arrow_expression.py::TestArrowExpressionEdgeCases::test_inf_handling` — `passed` — `smoke` — `tests/test_arrow_expression.py:L352`
- `test_arrow_expression.py::TestArrowExpressionEdgeCases::test_nan_propagation` — `passed` — `smoke` — `tests/test_arrow_expression.py:L361`
- `test_arrow_expression.py::TestArrowExpressionEdgeCases::test_zero_handling` — `passed` — `smoke` — `tests/test_arrow_expression.py:L369`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_arrow_info_includes_compute` — `passed` — `smoke` — `tests/test_arrow_expression.py:L70`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_comparison_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L175`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_complex_arithmetic` — `passed` — `smoke` — `tests/test_arrow_expression.py:L91`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_disabled_arrow` — `passed` — `smoke` — `tests/test_arrow_expression.py:L236`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_division_type_promotion` — `passed` — `smoke` — `tests/test_arrow_expression.py:L184`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_exp_log_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L135`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_fallback_for_unsupported` — `passed` — `smoke` — `tests/test_arrow_expression.py:L215`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_hyperbolic_functions` — `passed` — `smoke` — `tests/test_arrow_expression.py:L258`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_nested_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L200`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_power_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L157`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_simple_arithmetic` — `passed` — `smoke` — `tests/test_arrow_expression.py:L78`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_small_array_uses_eval` — `passed` — `smoke` — `tests/test_arrow_expression.py:L224`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_sqrt_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L103`
- `test_arrow_expression.py::TestArrowExpressionEvaluation::test_trig_expression` — `passed` — `smoke` — `tests/test_arrow_expression.py:L117`
- `test_arrow_expression.py::TestArrowExpressionPerformance::test_expression_speed` — `passed` — `smoke` — `tests/test_arrow_expression.py:L299`
- `test_arrow_expression.py::TestArrowExpressionPerformance::test_multiple_expressions_speed` — `passed` — `smoke` — `tests/test_arrow_expression.py:L311`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_all_missing_keys` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L292`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_empty_arrays` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L325`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_matches_numpy_simple` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L248`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_matches_numpy_with_missing` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L263`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_no_missing_keys` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L312`
- `test_arrow_scatter.py::TestArrowScatterCorrectness::test_single_element` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L336`
- `test_arrow_scatter.py::TestArrowScatterIntegration::test_multi_column_extract` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L395`
- `test_arrow_scatter.py::TestArrowScatterIntegration::test_tpc_like_join` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L351`
- `test_arrow_scatter.py::TestArrowScatterPerformance::test_arrow_vs_numpy_performance` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L212`
- `test_arrow_scatter.py::TestArrowScatterPerformance::test_scatter_large_array_speed` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L158`
- `test_arrow_scatter.py::TestArrowScatterPerformance::test_scatter_with_missing_keys_speed` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L180`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_basic_take` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L35`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_int_with_nulls_becomes_float` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L120`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_missing_key_handling` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L57`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_preserves_float32_dtype` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L87`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_preserves_float64_dtype` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L98`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_preserves_int32_dtype` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L109`
- `test_arrow_scatter.py::TestArrowScatterPrimitives::test_take_with_duplicates` — `passed` — `smoke` — `tests/test_arrow_scatter.py:L46`

</details>

<details>
<summary><code>BACK.numba</code> — 20 owned pytest nodes</summary>

- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_1_simple_arithmetic_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L62`
- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_2_compound_expression_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L79`
- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_3_subframe_scatter_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L96`
- `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_2_columns` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L561`
- `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_different_maxes_in_main_vs_sub` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L481`
- `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_matches_pandas_3col` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L437`
- `test_numba_acceleration.py::TestMultiColumnLinearization::test_linearization_negative_keys_fallback` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L536`
- `test_numba_acceleration.py::TestNumbaAvailability::test_numba_info_property` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L34`
- `test_numba_acceleration.py::TestNumbaAvailability::test_use_numba_parameter` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L45`
- `test_numba_acceleration.py::TestNumbaDirectAccelerators::test_numba_compute_join_indices` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L369`
- `test_numba_acceleration.py::TestNumbaDirectAccelerators::test_numba_scatter_inplace` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L355`
- `test_numba_acceleration.py::TestNumbaIndexLookup::test_multi_column_key_uses_pandas` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L262`
- `test_numba_acceleration.py::TestNumbaIndexLookup::test_numba_index_lookup_matches_pandas` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L204`
- `test_numba_acceleration.py::TestNumbaIndexLookup::test_numba_index_lookup_single_column_verified` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L288`
- `test_numba_acceleration.py::TestNumbaIndexLookup::test_sparse_keys` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L229`
- `test_numba_acceleration.py::TestNumbaScatter::test_numba_handles_missing_keys` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L132`
- `test_numba_acceleration.py::TestNumbaScatter::test_numba_produces_identical_results_f32` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L81`
- `test_numba_acceleration.py::TestNumbaScatter::test_numba_produces_identical_results_f64` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L107`
- `test_numba_acceleration.py::TestNumbaScatter::test_small_data_uses_numpy` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L161`
- `test_numba_acceleration.py::TestNumbaWithFillConfig::test_fill_missing_with_numba` — `passed` — `smoke` — `tests/test_numba_acceleration.py:L387`

</details>

<details>
<summary><code>BACK.invariance</code> — 14 owned pytest nodes</summary>

- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_1_simple_arithmetic_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L62`
- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_2_compound_expression_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L79`
- `test_I13_backend_equivalence_invariance.py::TestI13BackendEquivalenceInvariance::test_I13_3_subframe_scatter_numba_equals_numpy` — `passed` — `invariance` — `tests/test_I13_backend_equivalence_invariance.py:L96`
- `test_invariance_backend.py::TestBackendSummary::test_I2_count` — `passed` — `invariance` — `tests/test_invariance_backend.py:L474`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_10_large_dataset_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L425`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_1_single_key_join_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L106`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_2_join_with_arithmetic_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L135`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_3_multikey_join_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L164`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_4_missing_keys_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L208`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_5_duplicate_keys_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L248`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy` — `failed` — `invariance` — `tests/test_invariance_backend.py:L287`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_7_multiple_subframes_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L320`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_8_math_functions_in_join_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L366`
- `test_invariance_backend.py::TestInvarianceBackend::test_I2_9_boolean_conditions_numba_vs_numpy` — `passed` — `invariance` — `tests/test_invariance_backend.py:L395`

</details>

## LAZY_LOADING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **LAZY.read_tree** — Lazy branch loading from ROOT | 88 | 88 | 0 | 0 | 0 | 0 | 0 | 2 |
| ✅ | **LAZY.chain** — Chain loading (multiple files) | 60 | 58 | 0 | 0 | 0 | 0 | 2 | 7 |
| ✅ | **LAZY.materialization** — Lazy subframe & alias evaluation — on-demand physical/logical resolution across eager/lazy parent-child modes, no-metadata operation, and raise vs warn/skip resolution policies | 70 | 70 | 0 | 0 | 0 | 0 | 0 | 18 |
| ✅ | **LAZY.userinfo_backcompat** — Lazy-path UserInfo metadata back-compatibility (AD-3 precedence) | 15 | 14 | 0 | 0 | 0 | 0 | 1 | 5 |
| ✅ | **LAZY.chain_metadata** — Chain lazy metadata recovery (PHASE_13_67) — first-file UserInfo canonical, applied by DEFAULT (aliases+dtypes+compression, 0a); raise on cross-file incompatibility (0b); union/intersection -> error by default via metadata_conflict policy (parametrizable, off-switch in message); lazy application loads zero columns (INV-1); D4 pre-sized chain frame; names_only is a valid sparse case; real public-API read_chain_lazy recovery (alias+dtype+eval) verified on export_tree fixtures; subframe parity test runs on full-metadata fixtures, skips on names-only slim | 23 | 22 | 0 | 0 | 0 | 0 | 1 | 5 |
| ✅ | **LAZY.timeseries_draw** — Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory) | 40 | 38 | 0 | 0 | 0 | 0 | 2 | 35 |
| ✅ | **LAZY.subframe_draw** — Subframe-column lazy draw — single/nested qualified references, eager/lazy parent-child compositions, structural join-key setup, post-load aliases, and on-demand ensure_subframe resolution | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 19 |
| ✅ | **LAZY.alias_autoload** — Alias resolution auto-loads lazy branches (materialize_aliases / validate_aliases / describe_aliases bridge to the lazy reader; LAZY status) | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 4 |
| ✅ | **LAZY.expression_autoload** — Expression/column lazy autoload via ensure_columns() — bridges df.eval()/direct-access paths on a lazy ADF (get_required_branches → ensure_branches; branches-only, subframe-name + dotted-ref filtered; eager no-op) | 93 | 93 | 0 | 0 | 0 | 0 | 0 | 23 |
| ✅ | **LAZY.release** — Explicit lazy-branch/struct release (PHASE_13_68) — release_branches()/release_struct() symmetric evict: drop frame columns AND unbook the physical branch(es) on the lazy reader so a later access re-reads from file; struct members translated internal->physical forward from the registry; all-or-nothing loud refuse for eager frames (DD-alpha), aliases (DD-gamma -> dematerialize), written/__file_idx__ non-branch names (DD-beta), parent-side subframe join keys (DD-delta), and names in a materialized alias's dependency closure (C-6); memory_policy surface accepts 'keep' only ('bounded'/'drop' reserved); purely additive, no automatic eviction | 17 | 15 | 0 | 0 | 0 | 0 | 2 | 3 |

### Supporting tests

<details>
<summary><code>LAZY.read_tree</code> — 88 owned pytest nodes</summary>

- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_1_read_tree_eager_equals_read_tree_lazy` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L59`
- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_2_explicit_ensure_branches_equals_implicit_access` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L111`
- `test_branch_detection.py::TestEdgeCases::test_alias_not_found_passes_through` — `passed` — `smoke` — `tests/test_branch_detection.py:L471`
- `test_branch_detection.py::TestEdgeCases::test_color_non_string_ignored` — `passed` — `smoke` — `tests/test_branch_detection.py:L476`
- `test_branch_detection.py::TestEdgeCases::test_complex_selection_with_parentheses` — `passed` — `smoke` — `tests/test_branch_detection.py:L465`
- `test_branch_detection.py::TestEdgeCases::test_spaces_in_expr` — `passed` — `smoke` — `tests/test_branch_detection.py:L460`
- `test_branch_detection.py::TestGetRequiredBranches::test_alias_in_selection` — `passed` — `smoke` — `tests/test_branch_detection.py:L339`
- `test_branch_detection.py::TestGetRequiredBranches::test_explicit_aliases_param` — `passed` — `smoke` — `tests/test_branch_detection.py:L359`
- `test_branch_detection.py::TestGetRequiredBranches::test_expr_with_selection` — `passed` — `smoke` — `tests/test_branch_detection.py:L300`
- `test_branch_detection.py::TestGetRequiredBranches::test_full_physics_query` — `passed` — `smoke` — `tests/test_branch_detection.py:L348`
- `test_branch_detection.py::TestGetRequiredBranches::test_no_inputs_returns_empty` — `passed` — `smoke` — `tests/test_branch_detection.py:L368`
- `test_branch_detection.py::TestGetRequiredBranches::test_simple_expr` — `passed` — `smoke` — `tests/test_branch_detection.py:L290`
- `test_branch_detection.py::TestGetRequiredBranches::test_single_var_expr` — `passed` — `smoke` — `tests/test_branch_detection.py:L295`
- `test_branch_detection.py::TestGetRequiredBranches::test_validate_eager_mode` — `passed` — `smoke` — `tests/test_branch_detection.py:L373`
- `test_branch_detection.py::TestGetRequiredBranches::test_validate_includes_aliases` — `passed` — `smoke` — `tests/test_branch_detection.py:L382`
- `test_branch_detection.py::TestGetRequiredBranches::test_with_alias_resolution` — `passed` — `smoke` — `tests/test_branch_detection.py:L333`
- `test_branch_detection.py::TestGetRequiredBranches::test_with_color` — `passed` — `smoke` — `tests/test_branch_detection.py:L316`
- `test_branch_detection.py::TestGetRequiredBranches::test_with_group_by` — `passed` — `smoke` — `tests/test_branch_detection.py:L308`
- `test_branch_detection.py::TestGetRequiredBranches::test_with_group_by_and_color` — `passed` — `smoke` — `tests/test_branch_detection.py:L324`
- `test_branch_detection.py::TestIntegrationWithLazy::test_ensure_required_branches` — `passed` — `smoke` — `tests/test_branch_detection.py:L423`
- `test_branch_detection.py::TestIntegrationWithLazy::test_required_branches_before_load` — `passed` — `smoke` — `tests/test_branch_detection.py:L401`
- `test_branch_detection.py::TestIntegrationWithLazy::test_validate_lazy_mode` — `passed` — `smoke` — `tests/test_branch_detection.py:L412`
- `test_branch_detection.py::TestIntegrationWithLazy::test_workflow_detect_then_load` — `passed` — `smoke` — `tests/test_branch_detection.py:L432`
- `test_branch_detection.py::TestParseSelectionColumns::test_bitwise_operators` — `passed` — `smoke` — `tests/test_branch_detection.py:L94`
- `test_branch_detection.py::TestParseSelectionColumns::test_builtin_functions_excluded` — `passed` — `smoke` — `tests/test_branch_detection.py:L169`
- `test_branch_detection.py::TestParseSelectionColumns::test_complex_physics_expression` — `passed` — `smoke` — `tests/test_branch_detection.py:L139`
- `test_branch_detection.py::TestParseSelectionColumns::test_empty_selection` — `passed` — `smoke` — `tests/test_branch_detection.py:L145`
- `test_branch_detection.py::TestParseSelectionColumns::test_math_functions_excluded` — `passed` — `smoke` — `tests/test_branch_detection.py:L128`
- `test_branch_detection.py::TestParseSelectionColumns::test_multiple_conditions_and_cstyle` — `passed` — `smoke` — `tests/test_branch_detection.py:L79`
- `test_branch_detection.py::TestParseSelectionColumns::test_multiple_conditions_or_cstyle` — `passed` — `smoke` — `tests/test_branch_detection.py:L84`
- `test_branch_detection.py::TestParseSelectionColumns::test_negation_cstyle` — `passed` — `smoke` — `tests/test_branch_detection.py:L99`
- `test_branch_detection.py::TestParseSelectionColumns::test_negation_python` — `passed` — `smoke` — `tests/test_branch_detection.py:L104`
- `test_branch_detection.py::TestParseSelectionColumns::test_nested_function_calls` — `passed` — `smoke` — `tests/test_branch_detection.py:L134`
- `test_branch_detection.py::TestParseSelectionColumns::test_none_selection` — `passed` — `smoke` — `tests/test_branch_detection.py:L150`
- `test_branch_detection.py::TestParseSelectionColumns::test_not_equal_preserved` — `passed` — `smoke` — `tests/test_branch_detection.py:L109`
- `test_branch_detection.py::TestParseSelectionColumns::test_numeric_literals_excluded` — `passed` — `smoke` — `tests/test_branch_detection.py:L154`
- `test_branch_detection.py::TestParseSelectionColumns::test_numpy_function_excluded` — `passed` — `smoke` — `tests/test_branch_detection.py:L114`
- `test_branch_detection.py::TestParseSelectionColumns::test_numpy_sqrt_excluded` — `passed` — `smoke` — `tests/test_branch_detection.py:L121`
- `test_branch_detection.py::TestParseSelectionColumns::test_python_style_operators` — `passed` — `smoke` — `tests/test_branch_detection.py:L89`
- `test_branch_detection.py::TestParseSelectionColumns::test_simple_comparison` — `passed` — `smoke` — `tests/test_branch_detection.py:L74`
- `test_branch_detection.py::TestParseSelectionColumns::test_string_comparison` — `passed` — `smoke` — `tests/test_branch_detection.py:L163`
- `test_branch_detection.py::TestParseSelectionColumnsRegex::test_basic_extraction` — `passed` — `smoke` — `tests/test_branch_detection.py:L184`
- `test_branch_detection.py::TestParseSelectionColumnsRegex::test_filters_keywords` — `passed` — `smoke` — `tests/test_branch_detection.py:L190`
- `test_branch_detection.py::TestResolveToBaseBranches::test_alias_with_single_dep` — `passed` — `smoke` — `tests/test_branch_detection.py:L216`
- `test_branch_detection.py::TestResolveToBaseBranches::test_circular_alias_raises` — `passed` — `smoke` — `tests/test_branch_detection.py:L237`
- `test_branch_detection.py::TestResolveToBaseBranches::test_deep_nested_aliases` — `passed` — `smoke` — `tests/test_branch_detection.py:L229`
- `test_branch_detection.py::TestResolveToBaseBranches::test_diamond_dependency` — `passed` — `smoke` — `tests/test_branch_detection.py:L267`
- `test_branch_detection.py::TestResolveToBaseBranches::test_empty_input` — `passed` — `smoke` — `tests/test_branch_detection.py:L277`
- `test_branch_detection.py::TestResolveToBaseBranches::test_nested_aliases` — `passed` — `smoke` — `tests/test_branch_detection.py:L222`
- `test_branch_detection.py::TestResolveToBaseBranches::test_no_aliases` — `passed` — `smoke` — `tests/test_branch_detection.py:L205`
- `test_branch_detection.py::TestResolveToBaseBranches::test_self_referencing_alias_raises` — `passed` — `smoke` — `tests/test_branch_detection.py:L248`
- `test_branch_detection.py::TestResolveToBaseBranches::test_simple_alias` — `passed` — `smoke` — `tests/test_branch_detection.py:L210`
- `test_branch_detection.py::TestResolveToBaseBranches::test_three_way_circular` — `passed` — `smoke` — `tests/test_branch_detection.py:L256`
- `test_lazy_loading.py::TestAutoLoading::test_getitem_already_loaded_no_reload` — `passed` — `smoke` — `tests/test_lazy_loading.py:L334`
- `test_lazy_loading.py::TestAutoLoading::test_getitem_auto_loads_multiple` — `passed` — `smoke` — `tests/test_lazy_loading.py:L323`
- `test_lazy_loading.py::TestAutoLoading::test_getitem_auto_loads_single` — `passed` — `smoke` — `tests/test_lazy_loading.py:L312`
- `test_lazy_loading.py::TestAutoLoading::test_getitem_missing_raises` — `passed` — `smoke` — `tests/test_lazy_loading.py:L355`
- `test_lazy_loading.py::TestAutoLoading::test_getitem_non_lazy_normal` — `passed` — `smoke` — `tests/test_lazy_loading.py:L348`
- `test_lazy_loading.py::TestEdgeCases::test_load_all_branches` — `passed` — `smoke` — `tests/test_lazy_loading.py:L406`
- `test_lazy_loading.py::TestEdgeCases::test_repeated_ensure_same_branch` — `passed` — `smoke` — `tests/test_lazy_loading.py:L414`
- `test_lazy_loading.py::TestEnsureBranchesMethod::test_ensure_branches_empty_noop` — `passed` — `smoke` — `tests/test_lazy_loading.py:L250`
- `test_lazy_loading.py::TestEnsureBranchesMethod::test_ensure_branches_loads` — `passed` — `smoke` — `tests/test_lazy_loading.py:L226`
- `test_lazy_loading.py::TestEnsureBranchesMethod::test_ensure_branches_non_lazy` — `passed` — `smoke` — `tests/test_lazy_loading.py:L235`
- `test_lazy_loading.py::TestLazyEagerCompatibility::test_df_length_matches` — `passed` — `smoke` — `tests/test_lazy_loading.py:L391`
- `test_lazy_loading.py::TestLazyEagerCompatibility::test_same_data_values` — `passed` — `smoke` — `tests/test_lazy_loading.py:L370`
- `test_lazy_loading.py::TestLazyProperties::test_available_branches_lazy` — `passed` — `smoke` — `tests/test_lazy_loading.py:L265`
- `test_lazy_loading.py::TestLazyProperties::test_available_branches_non_lazy` — `passed` — `smoke` — `tests/test_lazy_loading.py:L274`
- `test_lazy_loading.py::TestLazyProperties::test_is_lazy_false` — `passed` — `smoke` — `tests/test_lazy_loading.py:L299`
- `test_lazy_loading.py::TestLazyProperties::test_is_lazy_true` — `passed` — `smoke` — `tests/test_lazy_loading.py:L294`
- `test_lazy_loading.py::TestLazyProperties::test_loaded_branches_lazy` — `passed` — `smoke` — `tests/test_lazy_loading.py:L279`
- `test_lazy_loading.py::TestLazyProperties::test_loaded_branches_non_lazy` — `passed` — `smoke` — `tests/test_lazy_loading.py:L289`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_ensure_branches_empty_list` — `passed` — `smoke` — `tests/test_lazy_loading.py:L132`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_ensure_branches_idempotent` — `passed` — `smoke` — `tests/test_lazy_loading.py:L110`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_ensure_branches_incremental` — `passed` — `smoke` — `tests/test_lazy_loading.py:L121`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_ensure_branches_invalid_raises` — `passed` — `smoke` — `tests/test_lazy_loading.py:L105`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_ensure_branches_loads_data` — `passed` — `smoke` — `tests/test_lazy_loading.py:L96`
- `test_lazy_loading.py::TestLazyTreeReaderEnsureBranches::test_is_loaded` — `passed` — `smoke` — `tests/test_lazy_loading.py:L138`
- `test_lazy_loading.py::TestLazyTreeReaderInit::test_init_file_not_found` — `passed` — `smoke` — `tests/test_lazy_loading.py:L75`
- `test_lazy_loading.py::TestLazyTreeReaderInit::test_init_loads_metadata` — `passed` — `smoke` — `tests/test_lazy_loading.py:L67`
- `test_lazy_loading.py::TestLazyTreeReaderInit::test_init_tree_not_found` — `passed` — `smoke` — `tests/test_lazy_loading.py:L80`
- `test_lazy_loading.py::TestLazyTreeReaderInit::test_repr` — `passed` — `smoke` — `tests/test_lazy_loading.py:L85`
- `test_lazy_loading.py::TestLazyTreeReaderMisc::test_close` — `passed` — `smoke` — `tests/test_lazy_loading.py:L163`
- `test_lazy_loading.py::TestLazyTreeReaderMisc::test_get_branch_dtype` — `passed` — `smoke` — `tests/test_lazy_loading.py:L151`
- `test_lazy_loading.py::TestLazyTreeReaderMisc::test_get_branch_dtype_invalid` — `passed` — `smoke` — `tests/test_lazy_loading.py:L158`
- `test_lazy_loading.py::TestReadTreeLazy::test_chain_config_created` — `passed` — `smoke` — `tests/test_lazy_loading.py:L198`
- `test_lazy_loading.py::TestReadTreeLazy::test_read_metadata_only` — `passed` — `smoke` — `tests/test_lazy_loading.py:L189`
- `test_lazy_loading.py::TestReadTreeLazy::test_read_with_branches` — `passed` — `smoke` — `tests/test_lazy_loading.py:L177`
- `test_lazy_loading.py::TestReadTreeLazy::test_schema_applied` — `passed` — `smoke` — `tests/test_lazy_loading.py:L207`

</details>

<details>
<summary><code>LAZY.chain</code> — 60 owned pytest nodes</summary>

- `test_chain_loading.py::TestChainCreation::test_available_branches` — `passed` — `smoke` — `tests/test_chain_loading.py:L77`
- `test_chain_loading.py::TestChainCreation::test_chain_info_property` — `passed` — `smoke` — `tests/test_chain_loading.py:L56`
- `test_chain_loading.py::TestChainCreation::test_empty_glob_raises` — `passed` — `smoke` — `tests/test_chain_loading.py:L45`
- `test_chain_loading.py::TestChainCreation::test_entry_offsets` — `passed` — `smoke` — `tests/test_chain_loading.py:L92`
- `test_chain_loading.py::TestChainCreation::test_file_list` — `passed` — `smoke` — `tests/test_chain_loading.py:L33`
- `test_chain_loading.py::TestChainCreation::test_glob_pattern` — `passed` — `smoke` — `tests/test_chain_loading.py:L27`
- `test_chain_loading.py::TestChainCreation::test_missing_tree_raises` — `passed` — `smoke` — `tests/test_chain_loading.py:L50`
- `test_chain_loading.py::TestChainCreation::test_separate_tree_name` — `passed` — `smoke` — `tests/test_chain_loading.py:L39`
- `test_chain_loading.py::TestChainCreation::test_single_file_not_chain` — `passed` — `smoke` — `tests/test_chain_loading.py:L65`
- `test_chain_loading.py::TestChainCreation::test_sorted_glob_order` — `passed` — `smoke` — `tests/test_chain_loading.py:L71`
- `test_chain_loading.py::TestChainCreation::test_total_entries_sum` — `passed` — `smoke` — `tests/test_chain_loading.py:L85`
- `test_chain_loading.py::TestChainLoading::test_column_access_auto_loads` — `passed` — `smoke` — `tests/test_chain_loading.py:L232`
- `test_chain_loading.py::TestChainLoading::test_data_correct_after_load` — `passed` — `smoke` — `tests/test_chain_loading.py:L221`
- `test_chain_loading.py::TestChainLoading::test_draw_auto_loads` — `passed` — `smoke` — `tests/test_chain_loading.py:L143`
- `test_chain_loading.py::TestChainLoading::test_eager_loads_all` — `passed` — `smoke` — `tests/test_chain_loading.py:L198`
- `test_chain_loading.py::TestChainLoading::test_ensure_branches_loads` — `passed` — `smoke` — `tests/test_chain_loading.py:L122`
- `test_chain_loading.py::TestChainLoading::test_file_index_column` — `passed` — `smoke` — `tests/test_chain_loading.py:L172`
- `test_chain_loading.py::TestChainLoading::test_file_index_not_in_available_branches` — `passed` — `smoke` — `tests/test_chain_loading.py:L183`
- `test_chain_loading.py::TestChainLoading::test_idempotent_ensure_branches` — `passed` — `smoke` — `tests/test_chain_loading.py:L209`
- `test_chain_loading.py::TestChainLoading::test_incremental_loading` — `passed` — `smoke` — `tests/test_chain_loading.py:L131`
- `test_chain_loading.py::TestChainLoading::test_lazy_no_data_initially` — `passed` — `smoke` — `tests/test_chain_loading.py:L102`
- `test_chain_loading.py::TestChainLoading::test_missing_branch_raises` — `passed` — `smoke` — `tests/test_chain_loading.py:L163`
- `test_chain_loading.py::TestChainLoading::test_string_ensure_branches` — `passed` — `smoke` — `tests/test_chain_loading.py:L241`
- `test_chain_loading.py::TestEntrySelection::test_entry_out_of_range_raises` — `passed` — `smoke` — `tests/test_chain_loading.py:L408`
- `test_chain_loading.py::TestEntrySelection::test_entry_selection_cross_file` — `passed` — `smoke` — `tests/test_chain_loading.py:L377`
- `test_chain_loading.py::TestEntrySelection::test_file_index_distribution` — `passed` — `smoke` — `tests/test_chain_loading.py:L419`
- `test_chain_loading.py::TestEntrySelection::test_get_file_for_entry` — `passed` — `smoke` — `tests/test_chain_loading.py:L394`
- `test_chain_loading.py::TestEntrySelection::test_global_entry_numbering` — `passed` — `smoke` — `tests/test_chain_loading.py:L368`
- `test_chain_loading.py::TestMemoryEstimation::test_estimate_human_readable` — `passed` — `smoke` — `tests/test_chain_loading.py:L518`
- `test_chain_loading.py::TestMemoryEstimation::test_estimate_memory` — `passed` — `smoke` — `tests/test_chain_loading.py:L488`
- `test_chain_loading.py::TestMemoryEstimation::test_estimate_memory_all_branches` — `passed` — `smoke` — `tests/test_chain_loading.py:L499`
- `test_chain_loading.py::TestMemoryEstimation::test_estimate_memory_eager_mode` — `passed` — `smoke` — `tests/test_chain_loading.py:L507`
- `test_chain_loading.py::TestParseChainFiles::test_glob_with_tree` — `passed` — `smoke` — `tests/test_chain_loading.py:L531`
- `test_chain_loading.py::TestParseChainFiles::test_list_with_separate_tree` — `passed` — `smoke` — `tests/test_chain_loading.py:L543`
- `test_chain_loading.py::TestParseChainFiles::test_list_with_tree` — `passed` — `smoke` — `tests/test_chain_loading.py:L537`
- `test_chain_loading.py::TestParseChainFiles::test_single_file` — `passed` — `smoke` — `tests/test_chain_loading.py:L550`
- `test_chain_loading.py::TestResourceManagement::test_close_idempotent` — `passed` — `smoke` — `tests/test_chain_loading.py:L477`
- `test_chain_loading.py::TestResourceManagement::test_context_manager` — `passed` — `smoke` — `tests/test_chain_loading.py:L436`
- `test_chain_loading.py::TestResourceManagement::test_explicit_close` — `passed` — `smoke` — `tests/test_chain_loading.py:L445`
- `test_chain_loading.py::TestResourceManagement::test_lru_cache_eviction` — `passed` — `smoke` — `tests/test_chain_loading.py:L454`
- `test_chain_loading.py::TestResourceManagement::test_many_files_lru` — `passed` — `smoke` — `tests/test_chain_loading.py:L465`
- `test_chain_loading.py::TestValidationModes::test_first_mode_extra_branches_ignored` — `passed` — `smoke` — `tests/test_chain_loading.py:L336`
- `test_chain_loading.py::TestValidationModes::test_first_mode_fills_nan_for_missing` — `passed` — `smoke` — `tests/test_chain_loading.py:L270`
- `test_chain_loading.py::TestValidationModes::test_first_warns_on_mismatch` — `passed` — `smoke` — `tests/test_chain_loading.py:L261`
- `test_chain_loading.py::TestValidationModes::test_intersection_uses_common` — `passed` — `smoke` — `tests/test_chain_loading.py:L287`
- `test_chain_loading.py::TestValidationModes::test_intersection_warns` — `passed` — `smoke` — `tests/test_chain_loading.py:L348`
- `test_chain_loading.py::TestValidationModes::test_invalid_validation_mode_raises` — `passed` — `smoke` — `tests/test_chain_loading.py:L328`
- `test_chain_loading.py::TestValidationModes::test_strict_raises_on_mismatch` — `passed` — `smoke` — `tests/test_chain_loading.py:L253`
- `test_chain_loading.py::TestValidationModes::test_union_includes_all` — `passed` — `smoke` — `tests/test_chain_loading.py:L299`
- `test_chain_loading.py::TestValidationModes::test_union_mode_fills_nan_for_missing` — `passed` — `smoke` — `tests/test_chain_loading.py:L311`
- `test_chain_loading.py::TestValidationModes::test_union_warns` — `passed` — `smoke` — `tests/test_chain_loading.py:L356`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_1_lazy_vs_eager_column_values` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L145`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_2_lazy_vs_eager_alias_evaluation` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L163`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_3_lazy_vs_eager_with_subframe_join` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L200`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_4_lazy_chain_vs_eager_chain` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L236`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_5_lazy_subframe_vs_eager_subframe` — `skipped` — `invariance` — `tests/test_invariance_load_mode.py:L256`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_6_mixed_lazy_main_eager_sub` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L289`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_7_mixed_eager_main_lazy_sub` — `skipped` — `invariance` — `tests/test_invariance_load_mode.py:L317`
- `test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_8_branch_auto_detection_equivalence` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L350`
- `test_invariance_load_mode.py::TestLoadModeSummary::test_I1_count` — `passed` — `invariance` — `tests/test_invariance_load_mode.py:L391`

</details>

<details>
<summary><code>LAZY.materialization</code> — 70 owned pytest nodes</summary>

- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_1_read_tree_eager_equals_read_tree_lazy` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L59`
- `test_I10_lazy_eager_invariance.py::TestI10LazyEagerInvariance::test_I10_2_explicit_ensure_branches_equals_implicit_access` — `passed` — `invariance` — `tests/test_I10_lazy_eager_invariance.py:L111`
- `test_lazy_subframes.py::TestChainPlusChainIntegration::test_main_chain_with_subframe_chain` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L925`
- `test_lazy_subframes.py::TestChainSchemaConsistency::test_chain_schema_has_both_index_keys` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L1119`
- `test_lazy_subframes.py::TestConfigTypeField::test_chain_has_type_chain` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L1098`
- `test_lazy_subframes.py::TestConfigTypeField::test_single_file_has_type_file` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L1086`
- `test_lazy_subframes.py::TestGetSubframesForAliases::test_finds_direct_subframe_reference` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L504`
- `test_lazy_subframes.py::TestGetSubframesForAliases::test_finds_nested_subframe_reference` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L516`
- `test_lazy_subframes.py::TestGetSubframesForAliases::test_no_subframes_needed` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L529`
- `test_lazy_subframes.py::TestLazinessPreservation::test_ensure_subframe_does_not_load_main` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L542`
- `test_lazy_subframes.py::TestLazinessPreservation::test_get_subframe_does_not_load_main` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L564`
- `test_lazy_subframes.py::TestLazyLoadingTrigger::test_ensure_eager_subframe_noop` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L262`
- `test_lazy_subframes.py::TestLazyLoadingTrigger::test_ensure_subframe_idempotent` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L249`
- `test_lazy_subframes.py::TestLazyLoadingTrigger::test_explicit_ensure_subframe` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L236`
- `test_lazy_subframes.py::TestLazyLoadingTrigger::test_load_on_materialize` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L193`
- `test_lazy_subframes.py::TestLazyLoadingTrigger::test_load_on_materialize_single` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L216`
- `test_lazy_subframes.py::TestLazySubframeErrors::test_eager_lazy_name_conflict` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L430`
- `test_lazy_subframes.py::TestLazySubframeErrors::test_ensure_unknown_subframe_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L400`
- `test_lazy_subframes.py::TestLazySubframeErrors::test_invalid_columns_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L418`
- `test_lazy_subframes.py::TestLazySubframeErrors::test_missing_index_columns_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L407`
- `test_lazy_subframes.py::TestLazySubframeJoin::test_column_subset_works` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L320`
- `test_lazy_subframes.py::TestLazySubframeJoin::test_join_produces_correct_values` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L281`
- `test_lazy_subframes.py::TestLazySubframeJoin::test_missing_keys_fill_nan` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L298`
- `test_lazy_subframes.py::TestLazySubframeJoin::test_mixed_eager_lazy_subframes` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L378`
- `test_lazy_subframes.py::TestLazySubframeJoin::test_multiple_lazy_subframes` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L341`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_basic` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L110`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_duplicate_name_fails` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L172`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_validates_file_exists` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L139`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_validates_index_columns` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L161`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_validates_tree_name` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L150`
- `test_lazy_subframes.py::TestLazySubframeRegistration::test_register_with_columns` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L125`
- `test_lazy_subframes.py::TestLazySubframeResources::test_close_releases_readers` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L451`
- `test_lazy_subframes.py::TestLazySubframeResources::test_context_manager_cleanup` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L466`
- `test_lazy_subframes.py::TestLazySubframeResources::test_lazy_subframes_property` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L477`
- `test_lazy_subframes.py::TestParameterValidation::test_invalid_alignment_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L586`
- `test_lazy_subframes.py::TestParameterValidation::test_invalid_join_type_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L598`
- `test_lazy_subframes.py::TestParameterValidation::test_valid_alignments_accepted` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L610`
- `test_lazy_subframes.py::TestSchemaConsistency::test_eager_subframe_has_both_index_keys` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L626`
- `test_lazy_subframes.py::TestSchemaConsistency::test_lazy_subframe_has_index_columns` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L642`
- `test_lazy_subframes.py::TestSubframeChainLoading::test_chain_data_concatenated_correctly` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L824`
- `test_lazy_subframes.py::TestSubframeChainLoading::test_chain_join_produces_correct_values` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L841`
- `test_lazy_subframes.py::TestSubframeChainLoading::test_chain_loads_on_materialize` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L802`
- `test_lazy_subframes.py::TestSubframeChainLoading::test_chain_with_column_subset` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L879`
- `test_lazy_subframes.py::TestSubframeChainLoading::test_ensure_subframe_idempotent` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L898`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_basic` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L707`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_duplicate_name_raises` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L776`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_invalid_validation_mode` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L763`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_no_files_raises_filenotfound` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L740`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_validates_index_columns` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L751`
- `test_lazy_subframes.py::TestSubframeChainRegistration::test_register_chain_with_file_list` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L724`
- `test_lazy_subframes.py::TestSubframeChainResources::test_chain_subframes_property` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L1051`
- `test_lazy_subframes.py::TestSubframeChainResources::test_close_releases_chain_readers` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L1033`
- `test_lazy_subframes.py::TestSubframeChainValidation::test_first_mode_accepts_different_branches` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L963`
- `test_lazy_subframes.py::TestSubframeChainValidation::test_strict_mode_rejects_different_branches` — `passed` — `smoke` — `tests/test_lazy_subframes.py:L995`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`

</details>

<details>
<summary><code>LAZY.userinfo_backcompat</code> — 15 owned pytest nodes</summary>

- `test_phase1359_lazy_userinfo.py::test_G1_real_fixture_recovery_ITS` — `skipped` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L295`
- `test_phase1359_lazy_userinfo.py::test_G2_dtype_recovery` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L211`
- `test_phase1359_lazy_userinfo.py::test_G3_alias_recovery` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L224`
- `test_phase1359_lazy_userinfo.py::test_G4_malformed_userinfo_fallthrough` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L253`
- `test_phase1359_lazy_userinfo.py::test_G5_registration_failure_path` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L237`
- `test_phase1359_lazy_userinfo.py::test_conflict_userinfo_beats_key` — `passed` — `invariance` — `tests/test_phase1359_lazy_userinfo.py:L176`
- `test_phase1359_lazy_userinfo.py::test_keypath_lazy_registration_invariance` — `passed` — `invariance` — `tests/test_phase1359_lazy_userinfo.py:L85`
- `test_phase1359_lazy_userinfo.py::test_lazy_eager_subframe_invariance` — `passed` — `invariance` — `tests/test_phase1359_lazy_userinfo.py:L104`
- `test_phase1359_lazy_userinfo.py::test_load_guard_coerces_awkward` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L338`
- `test_phase1359_lazy_userinfo.py::test_multientry_userinfo_loop` — `passed` — `invariance` — `tests/test_phase1359_lazy_userinfo.py:L188`
- `test_phase1359_lazy_userinfo.py::test_names_only_no_registration` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L160`
- `test_phase1359_lazy_userinfo.py::test_precedence_key_when_only_key` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L133`
- `test_phase1359_lazy_userinfo.py::test_precedence_names_only_carries_indicator` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L145`
- `test_phase1359_lazy_userinfo.py::test_root_uproot_userinfo_invariance` — `passed` — `invariance` — `tests/test_phase1359_lazy_userinfo.py:L121`
- `test_phase1359_lazy_userinfo.py::test_write_key_fallback_no_root` — `passed` — `smoke` — `tests/test_phase1359_lazy_userinfo.py:L309`

</details>

<details>
<summary><code>LAZY.chain_metadata</code> — 23 owned pytest nodes</summary>

- `test_phase1358_lazy_calibITS.py::test_calibITS_lazy_applies_same_metadata_as_eager` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L178`
- `test_phase1358_lazy_calibITS.py::test_calibITS_lazy_value_parity_with_metadata_eager` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L196`
- `test_phase1358_lazy_calibITS.py::test_calibITS_subframe_lazy_eager_value_parity` — `skipped` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L151`
- `test_phase_13_67_chain_metadata.py::TestApplyRealStructure::test_applies_aliases` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L65`
- `test_phase_13_67_chain_metadata.py::TestApplyRealStructure::test_names_only_apply_is_noop` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L70`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_all_bare_proceeds` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L46`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_identical_ok` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L42`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_mismatch_raises[<lambda>0]` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L54`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_mismatch_raises[<lambda>1]` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L54`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_mismatch_raises[<lambda>2]` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L54`
- `test_phase_13_67_chain_metadata.py::TestComparatorRealStructure::test_mixed_presence_raises` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L59`
- `test_phase_13_67_chain_metadata.py::TestNormalizeRealStructure::test_names_only_is_valid_sparse_not_refused` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L27`
- `test_phase_13_67_chain_metadata.py::TestNormalizeRealStructure::test_none_metadata` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L37`
- `test_phase_13_67_chain_metadata.py::TestNormalizeRealStructure::test_subframes_list_no_crash` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L22`
- `test_phase_13_67_chain_metadata.py::TestValidateMetadataParam::test_bad_value_raises` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L80`
- `test_phase_13_67_chain_metadata.py::TestValidateMetadataParam::test_selector_rejects_bad_value` — `passed` — `smoke` — `tests/test_phase_13_67_chain_metadata.py:L84`
- `test_phase_13_67_chain_recovery_public_api.py::test_apply_recovered_metadata_bad_alias_does_not_abort_rest` — `passed` — `smoke` — `tests/test_phase_13_67_chain_recovery_public_api.py:L123`
- `test_phase_13_67_chain_recovery_public_api.py::test_apply_recovered_metadata_warns_on_bad_dtype` — `passed` — `smoke` — `tests/test_phase_13_67_chain_recovery_public_api.py:L113`
- `test_phase_13_67_chain_recovery_public_api.py::test_metadata_conflict_error_raises_with_offswitch` — `passed` — `invariance` — `tests/test_phase_13_67_chain_recovery_public_api.py:L73`
- `test_phase_13_67_chain_recovery_public_api.py::test_metadata_conflict_skip_proceeds_silently` — `passed` — `smoke` — `tests/test_phase_13_67_chain_recovery_public_api.py:L91`
- `test_phase_13_67_chain_recovery_public_api.py::test_metadata_conflict_warn_proceeds` — `passed` — `smoke` — `tests/test_phase_13_67_chain_recovery_public_api.py:L82`
- `test_phase_13_67_chain_recovery_public_api.py::test_read_chain_lazy_recovers_alias_public_api` — `passed` — `invariance` — `tests/test_phase_13_67_chain_recovery_public_api.py:L59`
- `test_phase_13_67_chain_recovery_public_api.py::test_union_with_metadata_default_errors` — `passed` — `invariance` — `tests/test_phase_13_67_chain_recovery_public_api.py:L102`

</details>

<details>
<summary><code>LAZY.timeseries_draw</code> — 40 owned pytest nodes</summary>

- `test_phase1358_gallery_lazy.py::test_AC1_AC2_gallery_lazy_vs_eager` — `skipped` — `smoke` — `tests/test_phase1358_gallery_lazy.py:L32`
- `test_phase1358_lazy_calibITS.py::test_calibITS_groupby_exact_load_and_equiv` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L96`
- `test_phase1358_lazy_calibITS.py::test_calibITS_hist_exact_load` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L107`
- `test_phase1358_lazy_calibITS.py::test_calibITS_lazy_applies_same_metadata_as_eager` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L178`
- `test_phase1358_lazy_calibITS.py::test_calibITS_lazy_construct_loads_nothing` — `passed` — `smoke` — `tests/test_phase1358_lazy_calibITS.py:L77`
- `test_phase1358_lazy_calibITS.py::test_calibITS_lazy_value_parity_with_metadata_eager` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L196`
- `test_phase1358_lazy_calibITS.py::test_calibITS_profile_exact_load_and_equiv` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L84`
- `test_phase1358_lazy_calibITS.py::test_calibITS_subframe_column_lazy_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L115`
- `test_phase1358_lazy_calibITS.py::test_calibITS_subframe_lazy_eager_value_parity` — `skipped` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L151`
- `test_phase1358_lazy_draw_invariance.py::test_d2_missing_wiring_raises_not_silent` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L252`
- `test_phase1358_lazy_draw_invariance.py::test_d2_resolver_isolation_function_level` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L269`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_color_branch_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L188`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_batch_loads_union` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L139`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_figures_loads_union` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L154`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[hist]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[hist_cumulative]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[hist_selection]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[profile]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[profile_facet]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[profile_groupby]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[profile_regfunc]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[profile_weights]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_draw_invariance[scatter]` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L110`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_logical_selection_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L208`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_multidraw_no_reload` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L228`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_nested_alias_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L178`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_nested_subframe_column_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L326`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_nonexistent_branch_raises` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L243`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_selection_side_registered_function` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L218`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_selection_vector_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L167`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_subframe_column_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L282`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_weights_vector_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L198`
- `test_phase1358_lazy_timeseries.py::test_AC2_lazy_eager_data_identity` — `passed` — `smoke` — `tests/test_phase1358_lazy_timeseries.py:L115`
- `test_phase1358_lazy_timeseries.py::test_AC3_facet_weights_preload` — `passed` — `smoke` — `tests/test_phase1358_lazy_timeseries.py:L106`
- `test_phase1358_lazy_timeseries.py::test_AC4_bracket_vector_resolver` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L175`
- `test_phase1358_lazy_timeseries.py::test_AC4_resolver_regression` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L142`
- `test_phase1358_lazy_timeseries.py::test_AC5_estimate_memory_exact` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L158`
- `test_phase1358_lazy_timeseries.py::test_AC6_resolver_registered_function` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L128`
- `test_phase1358_lazy_timeseries.py::test_AC7_only_needed_branches_load` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L77`
- `test_phase1358_lazy_timeseries.py::test_AC7_registered_function_alias_loads` — `passed` — `invariance` — `tests/test_phase1358_lazy_timeseries.py:L91`

</details>

<details>
<summary><code>LAZY.subframe_draw</code> — 19 owned pytest nodes</summary>

- `test_phase1358_lazy_calibITS.py::test_calibITS_subframe_column_lazy_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_calibITS.py:L115`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_nested_subframe_column_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L326`
- `test_phase1358_lazy_draw_invariance.py::test_lazy_subframe_column_draw` — `passed` — `invariance` — `tests/test_phase1358_lazy_draw_invariance.py:L282`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_alias_added_after_child_data_load[alias_postload-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L174`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_physical_existing[physical-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L154`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_raise_policy[bad_raise-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L161`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_eager-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_eager]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`
- `test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix::test_r10_unresolved_bad_warn_skip_policy[bad_warn-parent_lazy-child_lazy]` — `passed` — `invariance` — `tests/test_phase_13_76_v12_subframe_mode_matrix.py:L167`

</details>

<details>
<summary><code>LAZY.alias_autoload</code> — 6 owned pytest nodes</summary>

- `test_phase1360_alias_lazy_bridge.py::test_T1_materialize_over_unloaded_branch_equals_eager` — `passed` — `invariance` — `tests/test_phase1360_alias_lazy_bridge.py:L58`
- `test_phase1360_alias_lazy_bridge.py::test_T2_exact_load_decoys_not_loaded` — `passed` — `smoke` — `tests/test_phase1360_alias_lazy_bridge.py:L75`
- `test_phase1360_alias_lazy_bridge.py::test_T3_validate_and_describe_lazy_not_broken` — `passed` — `invariance` — `tests/test_phase1360_alias_lazy_bridge.py:L88`
- `test_phase1360_alias_lazy_bridge.py::test_T4_genuine_missing_still_reported_and_registered_fn` — `passed` — `smoke` — `tests/test_phase1360_alias_lazy_bridge.py:L102`
- `test_phase1360_alias_lazy_bridge.py::test_T7_chained_alias_transitive_autoload_equals_eager` — `passed` — `invariance` — `tests/test_phase1360_alias_lazy_bridge.py:L161`
- `test_phase1360_alias_lazy_bridge.py::test_T8_dtype_bearing_alias_lazy_equals_eager` — `passed` — `invariance` — `tests/test_phase1360_alias_lazy_bridge.py:L173`

</details>

<details>
<summary><code>LAZY.expression_autoload</code> — 93 owned pytest nodes</summary>

- `test_bug20260624_ensure_columns.py::test_T1_eval_fails_before_succeeds_after` — `passed` — `smoke` — `tests/test_bug20260624_ensure_columns.py:L38`
- `test_bug20260624_ensure_columns.py::test_T2_mixed_args` — `passed` — `smoke` — `tests/test_bug20260624_ensure_columns.py:L52`
- `test_bug20260624_ensure_columns.py::test_T3_eager_noop` — `passed` — `smoke` — `tests/test_bug20260624_ensure_columns.py:L61`
- `test_bug20260624_ensure_columns.py::test_T4_math_constants_not_loaded` — `passed` — `smoke` — `tests/test_bug20260624_ensure_columns.py:L69`
- `test_phase_13_62_adf_s1.py::test_C1_subframe_alias_lazy_draw_end_to_end` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L97`
- `test_phase_13_62_adf_s1.py::test_T1_hand_added_column_readable` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L41`
- `test_phase_13_62_adf_s1.py::test_T2_genuine_missing_raises_cause_naming_error` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L49`
- `test_phase_13_62_adf_s1.py::test_T3_alias_and_subframe_columns_not_raised` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L60`
- `test_phase_13_62_adf_s1.py::test_T4_exact_load` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L70`
- `test_phase_13_62_adf_s1.py::test_T5_eager_unchanged` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L80`
- `test_phase_13_62_adf_s1.py::test_T6_typo_negative_control` — `passed` — `smoke` — `tests/test_phase_13_62_adf_s1.py:L90`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_auto_struct_gains_member_on_refresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L439`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_explicit_struct_never_broadened` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L450`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_ensure_columns` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L418`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_get_required_branches` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L413`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_release_then_reload` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L478`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_leaf_two_parents_no_crosstalk` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L487`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_size_catalog_change_detected` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L423`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_schema_roundtrip_provenance_and_precedence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L460`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_chain_initial_branches_normalized_and_full` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L507`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_tree_initial_branches_full_structure` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L515`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_two_file_chain_value_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L500`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_batch_frame_capture_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L547`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_draw_frame_capture` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L540`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_figures_frame_capture_kwargs_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L554`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_composite` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L592`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_minimal` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L569`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_selection` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L583`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_reused_spec_object_second_call_works` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L563`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_lazy_reports_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L613`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_structure_contract_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L605`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_fault_injected_projection_raises_adf_error` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L623`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_batch_capture_excludes_physical_and_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L751`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_chain` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L671`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_tree` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L660`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_defaults_autoload_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L709`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_diagnostics_full_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L769`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_batch_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L726`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_figures_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L738`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_figures_per_figure_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L716`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fp_not_cached_on_failed_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L680`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_full_structure_completion_failure_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L693`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_refresh_member_usable_via_dot_grammar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L641`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_second_call_different_requirements_fresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L761`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_caller_spec_never_mutated_three_contexts` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L788`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_classifier_exception_reports_context` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L862`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_describe_structure_print_path` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L829`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_guard_ignores_titles_matching_internals` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L856`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_per_bin_profile_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L838`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_validate_aliases_fresh_lazy` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L817`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_chain_reader_delegates` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L84`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_tree_reader_three_valued` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L78`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_c1_unknown_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L110`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_clean_registration_and_provenance` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L94`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_explicit_register_precedence_survives` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L122`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h2_count_helper_never_a_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L100`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h3_collision_blocks_auto_member` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L104`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_chain_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L136`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_tree_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L132`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_eager_noop` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L159`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_fp_cache_prevents_rescan` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L150`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_idempotence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L140`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_tree_initial_branches_normalized` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L164`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_exact_loading_decoy_control` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L179`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_full_struct_loading_documented_behavior` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L173`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_alias_over_struct_equals_direct` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L199`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_auto_equals_explicit_registration` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L214`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_builtin_named_member` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L226`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_internal_column` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L194`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_physical_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L189`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_tree_equals_chain` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L207`
- `test_phase_13_75_lazy_struct_repair.py::TestStage6_ProjectionDispatch::test_reduced_frame_keeps_internal_drops_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L234`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_composite_production_expression` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L251`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_batch_surface` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L297`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_figures_surface_dict_and_short_form` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L303`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_reduced_equals_full_stats` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L311`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_entry_sliced_draw` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L318`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_minimal_profile_default_dispatch` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L247`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_persisted_schema_control` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L323`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_production_composition_facet_quantiles` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L289`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_slot_composition` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L256`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_color_facet_weights_slots` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L280`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_facet_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L274`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_group_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L268`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_selection_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L262`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_no_pandas_undefinedvariable_reaches_user` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L336`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_unknown_member_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L330`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_dtype_drift_warns_still_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L368`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_fixed_size_array_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L404`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_chain_detection_raises_loudly` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L359`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_scalar_jagged_raises` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L351`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_scalar_scalar_ok` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L347`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_union_absence_not_scalar_proof` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L377`

</details>

<details>
<summary><code>LAZY.release</code> — 17 owned pytest nodes</summary>

- `test_phase_13_68_release.py::test_REL_1_plain_release_reaccess_identity[chain]` — `passed` — `invariance` — `tests/test_phase_13_68_release.py:L66`
- `test_phase_13_68_release.py::test_REL_1_plain_release_reaccess_identity[tree]` — `passed` — `invariance` — `tests/test_phase_13_68_release.py:L66`
- `test_phase_13_68_release.py::test_REL_1_struct_member_end_to_end` — `skipped` — `invariance` — `tests/test_phase_13_68_release.py:L237`
- `test_phase_13_68_release.py::test_REL_2_c6_dependency_refuse_plain` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L93`
- `test_phase_13_68_release.py::test_REL_2_c6_dependency_refuse_struct_member` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L104`
- `test_phase_13_68_release.py::test_REL_3_plain_roundtrip_returns_and_gc` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L83`
- `test_phase_13_68_release.py::test_REL_4_dd_alpha_eager_raises` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L118`
- `test_phase_13_68_release.py::test_REL_4_dd_beta_unknown_name_refuse` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L125`
- `test_phase_13_68_release.py::test_REL_4_dd_gamma_alias_name_refuse` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L131`
- `test_phase_13_68_release.py::test_REL_5_memory_policy_surface` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L141`
- `test_phase_13_68_release.py::test_REL_6_partial_struct_release_exact_complement` — `passed` — `invariance` — `tests/test_phase_13_68_release.py:L152`
- `test_phase_13_68_release.py::test_REL_6_release_struct_sugar_and_unknown` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L171`
- `test_phase_13_68_release.py::test_REL_7_dematerialize_unaffected` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L187`
- `test_phase_13_68_release.py::test_REL_8_dd_delta_join_key_refuse[chain]` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L199`
- `test_phase_13_68_release.py::test_REL_8_dd_delta_join_key_refuse[in_memory]` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L199`
- `test_phase_13_68_release.py::test_REL_9_file_idx_marker_distinct_message` — `skipped` — `smoke` — `tests/test_phase_13_68_release.py:L225`
- `test_phase_13_68_release.py::test_REL_9_written_column_distinct_message` — `passed` — `smoke` — `tests/test_phase_13_68_release.py:L215`

</details>

## SUBFRAME

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **SUBFRAME.asymmetric_join_keys** — Asymmetric subframe join keys (PHASE_13_65) — register_subframe(right_index_columns=[...]) lets parent/child join columns differ in name (pandas left_on/right_on); name-aware across all three _compute_join_indices paths (single-col numba, Phase 8c multi-col linearization via rename-before-linearize, merge fallback); right_index_columns=None is byte-identical to the prior same-name behavior; schema-persisted with absent-field back-compat | 9 | 9 | 0 | 0 | 0 | 0 | 0 |  |

### Supporting tests

<details>
<summary><code>SUBFRAME.asymmetric_join_keys</code> — 9 owned pytest nodes</summary>

- `test_phase_13_65_adf_asymmetric_keys.py::test_A10_symmetric_unchanged` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L125`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A1_asymmetric_basic` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L141`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A4_singlecol_numba_vs_merge_asymmetric` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L87`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A4b_multicol_numba_vs_merge_asymmetric` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L62`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A4c_multicol_merge_asymmetric` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L112`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A5_missing_child_key` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L171`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A6_validation_length_and_names` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L153`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A7_schema_roundtrip_and_absent_field` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L218`
- `test_phase_13_65_adf_asymmetric_keys.py::test_A9_asymmetric_pre_index` — `passed` — `smoke` — `tests/test_phase_13_65_adf_asymmetric_keys.py:L234`

</details>

## OBJECT

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **OBJECT.struct_1to1** — 1:1 struct/object branch support (PHASE_13_66) — ROOT struct members (parent/member) usable via dot grammar (dedxTPC.dEdxTotIROC); three-name mapping (physical slash / internal member__struct / logical dot, anchor 0i); reference-driven load with A-1 rename-on-load handling bare-leaf or slash reader keys; public adf.eval() with Step-0 syntax gate; struct-aware across the 7 analysis surfaces, get_required_branches (physical form), the 5 dispatch sites, and all 3 draw surfaces; alias-over-struct (direct + nested) via _get_structs_for_aliases + _do_materialize hook; auto-detection with scalar/jagged guard (never auto-flatten 1:N, anchor 0h); schema-persisted (export + apply) with absent-key back-compat | 123 | 120 | 0 | 0 | 0 | 0 | 3 | 30 |

### Supporting tests

<details>
<summary><code>OBJECT.struct_1to1</code> — 123 owned pytest nodes</summary>

- `test_phase_13_66_adf_struct_foundation.py::TestAdfEval::test_alias_over_struct_eval` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L77`
- `test_phase_13_66_adf_struct_foundation.py::TestAdfEval::test_result_is_series_aligned` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L82`
- `test_phase_13_66_adf_struct_foundation.py::TestAdfEval::test_step0_syntax_gate` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L72`
- `test_phase_13_66_adf_struct_foundation.py::TestAdfEval::test_struct_member_comparison` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L61`
- `test_phase_13_66_adf_struct_foundation.py::TestAdfEval::test_struct_member_ratio` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L66`
- `test_phase_13_66_adf_struct_foundation.py::TestAliasOverStruct::test_direct_alias` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L163`
- `test_phase_13_66_adf_struct_foundation.py::TestAliasOverStruct::test_nested_alias` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L168`
- `test_phase_13_66_adf_struct_foundation.py::TestAnalysisSurfaces::test_materialize_struct_ref_alias` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L113`
- `test_phase_13_66_adf_struct_foundation.py::TestAnalysisSurfaces::test_only_physical_member_in_column_refs` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L107`
- `test_phase_13_66_adf_struct_foundation.py::TestAnalysisSurfaces::test_struct_ref_is_supported` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L102`
- `test_phase_13_66_adf_struct_foundation.py::TestDR71Message::test_bare_struct_name_hint` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L178`
- `test_phase_13_66_adf_struct_foundation.py::TestEvalErrorContract::test_malformed_syntaxerror_step0` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L212`
- `test_phase_13_66_adf_struct_foundation.py::TestEvalErrorContract::test_unresolvable_eager_nameerror` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L217`
- `test_phase_13_66_adf_struct_foundation.py::TestLazyStructPaths::test_detect_structs_auto_registers_scalar` — `skipped` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L230`
- `test_phase_13_66_adf_struct_foundation.py::TestLazyStructPaths::test_ensure_struct_loads_internal_names` — `skipped` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L234`
- `test_phase_13_66_adf_struct_foundation.py::TestLazyStructPaths::test_eval_autoloads_on_lazy` — `skipped` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L239`
- `test_phase_13_66_adf_struct_foundation.py::TestLoadPathViaMockReader::test_detect_structs_from_reader` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L303`
- `test_phase_13_66_adf_struct_foundation.py::TestLoadPathViaMockReader::test_ensure_struct_renames_bare_leaf_to_internal` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L287`
- `test_phase_13_66_adf_struct_foundation.py::TestLoadPathViaMockReader::test_eval_autoloads_struct_member_via_reader` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L293`
- `test_phase_13_66_adf_struct_foundation.py::TestLoadPathViaMockReader::test_eval_ratio_autoloads_both_members` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L298`
- `test_phase_13_66_adf_struct_foundation.py::TestRenameOnLoadHook::test_dataframe_rename` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L133`
- `test_phase_13_66_adf_struct_foundation.py::TestRenameOnLoadHook::test_dict_rename` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L126`
- `test_phase_13_66_adf_struct_foundation.py::TestRenameOnLoadHook::test_no_structs_noop` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L139`
- `test_phase_13_66_adf_struct_foundation.py::TestSchemaPersistence::test_structs_in_schema` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L185`
- `test_phase_13_66_adf_struct_foundation.py::TestSchemaRoundtrip::test_export_carries_structs` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L196`
- `test_phase_13_66_adf_struct_foundation.py::TestSchemaRoundtrip::test_from_schema_reconstructs_structs` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L202`
- `test_phase_13_66_adf_struct_foundation.py::TestSelectionLegStructAware::test_plain_selection_unaffected` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L155`
- `test_phase_13_66_adf_struct_foundation.py::TestSelectionLegStructAware::test_struct_member_to_physical` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L151`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_alias_over_struct_invariant` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L355`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_branch_resolution_invariant_dotted_vs_internal` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L336`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_dot_grammar_equals_internal_column[NHitsIROC]` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L328`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_dot_grammar_equals_internal_column[dEdxMaxTPC]` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L328`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_dot_grammar_equals_internal_column[dEdxTotIROC]` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L328`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_dot_grammar_equals_internal_column[dEdxTotOROC1]` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L328`
- `test_phase_13_66_adf_struct_foundation.py::TestStructInvariance::test_expression_invariant_across_members` — `passed` — `invariance` — `tests/test_phase_13_66_adf_struct_foundation.py:L343`
- `test_phase_13_66_adf_struct_foundation.py::TestStructRegistry::test_cross_namespace_collision_raises` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L38`
- `test_phase_13_66_adf_struct_foundation.py::TestStructRegistry::test_empty_members_raises` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L43`
- `test_phase_13_66_adf_struct_foundation.py::TestStructRegistry::test_three_name_mapping` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L32`
- `test_phase_13_66_adf_struct_foundation.py::TestStructRewrite::test_logical_to_internal` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L50`
- `test_phase_13_66_adf_struct_foundation.py::TestStructRewrite::test_word_boundary_safe` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L54`
- `test_phase_13_66_adf_struct_foundation.py::TestVectorGuardStructs::test_struct_ref_in_weights_vector_raises` — `passed` — `smoke` — `tests/test_phase_13_66_adf_struct_foundation.py:L90`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_auto_struct_gains_member_on_refresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L439`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_explicit_struct_never_broadened` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L450`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_ensure_columns` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L418`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_get_required_branches` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L413`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_release_then_reload` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L478`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_leaf_two_parents_no_crosstalk` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L487`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_size_catalog_change_detected` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L423`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_schema_roundtrip_provenance_and_precedence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L460`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_chain_initial_branches_normalized_and_full` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L507`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_tree_initial_branches_full_structure` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L515`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_two_file_chain_value_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L500`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_batch_frame_capture_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L547`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_draw_frame_capture` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L540`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_figures_frame_capture_kwargs_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L554`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_composite` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L592`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_minimal` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L569`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_selection` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L583`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_reused_spec_object_second_call_works` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L563`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_lazy_reports_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L613`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_structure_contract_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L605`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_fault_injected_projection_raises_adf_error` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L623`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_batch_capture_excludes_physical_and_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L751`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_chain` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L671`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_tree` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L660`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_defaults_autoload_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L709`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_diagnostics_full_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L769`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_batch_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L726`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_figures_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L738`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_figures_per_figure_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L716`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fp_not_cached_on_failed_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L680`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_full_structure_completion_failure_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L693`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_refresh_member_usable_via_dot_grammar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L641`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_second_call_different_requirements_fresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L761`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_caller_spec_never_mutated_three_contexts` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L788`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_classifier_exception_reports_context` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L862`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_describe_structure_print_path` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L829`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_guard_ignores_titles_matching_internals` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L856`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_per_bin_profile_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L838`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_validate_aliases_fresh_lazy` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L817`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_chain_reader_delegates` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L84`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_tree_reader_three_valued` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L78`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_c1_unknown_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L110`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_clean_registration_and_provenance` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L94`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_explicit_register_precedence_survives` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L122`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h2_count_helper_never_a_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L100`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h3_collision_blocks_auto_member` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L104`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_chain_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L136`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_tree_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L132`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_eager_noop` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L159`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_fp_cache_prevents_rescan` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L150`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_idempotence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L140`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_tree_initial_branches_normalized` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L164`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_exact_loading_decoy_control` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L179`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_full_struct_loading_documented_behavior` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L173`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_alias_over_struct_equals_direct` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L199`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_auto_equals_explicit_registration` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L214`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_builtin_named_member` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L226`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_internal_column` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L194`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_physical_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L189`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_tree_equals_chain` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L207`
- `test_phase_13_75_lazy_struct_repair.py::TestStage6_ProjectionDispatch::test_reduced_frame_keeps_internal_drops_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L234`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_composite_production_expression` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L251`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_batch_surface` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L297`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_figures_surface_dict_and_short_form` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L303`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_reduced_equals_full_stats` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L311`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_entry_sliced_draw` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L318`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_minimal_profile_default_dispatch` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L247`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_persisted_schema_control` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L323`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_production_composition_facet_quantiles` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L289`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_slot_composition` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L256`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_color_facet_weights_slots` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L280`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_facet_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L274`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_group_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L268`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_selection_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L262`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_no_pandas_undefinedvariable_reaches_user` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L336`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_unknown_member_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L330`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_dtype_drift_warns_still_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L368`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_fixed_size_array_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L404`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_chain_detection_raises_loudly` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L359`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_scalar_jagged_raises` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L351`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_scalar_scalar_ok` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L347`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_union_absence_not_scalar_proof` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L377`

</details>

## FIT_REGISTRATION

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ☑️ | **FIT.registration** — Fit metadata storage & retrieval | 98 | 98 | 0 | 0 | 0 | 0 | 0 |  |
| ☑️ | **FIT.visualization** — Fit summary visualization | 18 | 18 | 0 | 0 | 0 | 0 | 0 |  |

### Supporting tests

<details>
<summary><code>FIT.registration</code> — 98 owned pytest nodes</summary>

- `test_register_fit_result.py::TestApplyPullTransform::test_asinh_creates_alias` — `passed` — `smoke` — `tests/test_register_fit_result.py:L585`
- `test_register_fit_result.py::TestApplyPullTransform::test_idempotent` — `passed` — `smoke` — `tests/test_register_fit_result.py:L607`
- `test_register_fit_result.py::TestApplyPullTransform::test_invalid_transform_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L618`
- `test_register_fit_result.py::TestApplyPullTransform::test_none_returns_original` — `passed` — `smoke` — `tests/test_register_fit_result.py:L576`
- `test_register_fit_result.py::TestApplyPullTransform::test_tanh_creates_alias` — `passed` — `smoke` — `tests/test_register_fit_result.py:L596`
- `test_register_fit_result.py::TestCategoryFiltering::test_exclude_removes_categories` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1716`
- `test_register_fit_result.py::TestCategoryFiltering::test_include_limits_categories` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1701`
- `test_register_fit_result.py::TestComputeFitValidation::test_bad_fit_fails` — `passed` — `smoke` — `tests/test_register_fit_result.py:L659`
- `test_register_fit_result.py::TestComputeFitValidation::test_basic_validation` — `passed` — `smoke` — `tests/test_register_fit_result.py:L635`
- `test_register_fit_result.py::TestComputeFitValidation::test_good_fit_passes` — `passed` — `smoke` — `tests/test_register_fit_result.py:L648`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_generates_figures` — `passed` — `smoke` — `tests/test_register_fit_result.py:L701`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_large_dataset_warns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L755`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_returns_validation` — `passed` — `smoke` — `tests/test_register_fit_result.py:L744`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_saves_pdf` — `passed` — `smoke` — `tests/test_register_fit_result.py:L724`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_saves_png` — `passed` — `smoke` — `tests/test_register_fit_result.py:L712`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_unknown_fit_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L736`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_entry_end_limits_data` — `passed` — `smoke` — `tests/test_register_fit_result.py:L819`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_exclude_filter` — `passed` — `smoke` — `tests/test_register_fit_result.py:L798`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_fit_columns_subset` — `passed` — `smoke` — `tests/test_register_fit_result.py:L831`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_include_filter` — `passed` — `smoke` — `tests/test_register_fit_result.py:L786`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_pull_transform_asinh` — `passed` — `smoke` — `tests/test_register_fit_result.py:L808`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_unknown_category_warns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L843`
- `test_register_fit_result.py::TestEdgeCases::test_empty_dfGB` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1010`
- `test_register_fit_result.py::TestEdgeCases::test_missing_optional_metadata_keys` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1030`
- `test_register_fit_result.py::TestEdgeCases::test_single_fit_column` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1021`
- `test_register_fit_result.py::TestEdgeCases::test_special_characters_in_name` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1050`
- `test_register_fit_result.py::TestErrorSurfacing::test_invalid_column_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1480`
- `test_register_fit_result.py::TestErrorSurfacing::test_no_density_kwarg_duplication` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1458`
- `test_register_fit_result.py::TestErrorSurfacing::test_on_error_raise_surfaces_errors` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1446`
- `test_register_fit_result.py::TestFitMetadataAccessors::test_get_fit_metadata_all` — `passed` — `smoke` — `tests/test_register_fit_result.py:L533`
- `test_register_fit_result.py::TestFitMetadataAccessors::test_get_fit_metadata_single` — `passed` — `smoke` — `tests/test_register_fit_result.py:L524`
- `test_register_fit_result.py::TestFitMetadataAccessors::test_get_fit_metadata_unknown_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L544`
- `test_register_fit_result.py::TestFitMetadataAccessors::test_list_fit_results_empty` — `passed` — `smoke` — `tests/test_register_fit_result.py:L553`
- `test_register_fit_result.py::TestFitMetadataAccessors::test_list_fit_results_multiple` — `passed` — `smoke` — `tests/test_register_fit_result.py:L558`
- `test_register_fit_result.py::TestMultiGroupFit::test_multivar_layout_correct` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1339`
- `test_register_fit_result.py::TestMultiGroupFit::test_pull_distribution_multigroup` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1357`
- `test_register_fit_result.py::TestMultiGroupFit::test_quality_plot_shows_distribution` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1316`
- `test_register_fit_result.py::TestNumericalCorrectness::test_delta_equals_y_minus_prediction` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1408`
- `test_register_fit_result.py::TestNumericalCorrectness::test_prediction_formula_correct` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1391`
- `test_register_fit_result.py::TestNumericalCorrectness::test_pull_equals_delta_over_rms` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1377`
- `test_register_fit_result.py::TestNumericalCorrectness::test_validation_metrics_bounds` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1423`
- `test_register_fit_result.py::TestPlotContent::test_file_output_exists` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1567`
- `test_register_fit_result.py::TestPlotContent::test_gaussian_overlay_present` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1544`
- `test_register_fit_result.py::TestPlotContent::test_histograms_have_bars` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1507`
- `test_register_fit_result.py::TestPlotContent::test_no_error_text_in_plots` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1524`
- `test_register_fit_result.py::TestRegisterFitResultBackwardCompat::test_duplicate_overwrites` — `passed` — `smoke` — `tests/test_register_fit_result.py:L500`
- `test_register_fit_result.py::TestRegisterFitResultBackwardCompat::test_duplicate_registration_warns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L490`
- `test_register_fit_result.py::TestRegisterFitResultBackwardCompat::test_no_metadata_no_index_columns_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L482`
- `test_register_fit_result.py::TestRegisterFitResultBackwardCompat::test_no_metadata_with_index_columns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L470`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_metadata_stored` — `passed` — `smoke` — `tests/test_register_fit_result.py:L258`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_prediction_alias_created` — `passed` — `smoke` — `tests/test_register_fit_result.py:L267`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_pull_alias_created` — `passed` — `smoke` — `tests/test_register_fit_result.py:L283`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_residual_alias_created` — `passed` — `smoke` — `tests/test_register_fit_result.py:L275`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_returns_aliasdf` — `passed` — `smoke` — `tests/test_register_fit_result.py:L249`
- `test_register_fit_result.py::TestRegisterFitResultBasic::test_subframe_registered` — `passed` — `smoke` — `tests/test_register_fit_result.py:L241`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_dtype_applied` — `passed` — `smoke` — `tests/test_register_fit_result.py:L370`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_multi_column_predictions` — `passed` — `smoke` — `tests/test_register_fit_result.py:L337`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_prediction_evaluates_correctly` — `passed` — `smoke` — `tests/test_register_fit_result.py:L299`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_pull_distribution_properties` — `passed` — `smoke` — `tests/test_register_fit_result.py:L325`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_residual_evaluates_correctly` — `passed` — `smoke` — `tests/test_register_fit_result.py:L312`
- `test_register_fit_result.py::TestRegisterFitResultFormulas::test_subframe_coefficients_accessible` — `passed` — `smoke` — `tests/test_register_fit_result.py:L356`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_add_predictions_false` — `passed` — `smoke` — `tests/test_register_fit_result.py:L427`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_add_pulls_false` — `passed` — `smoke` — `tests/test_register_fit_result.py:L443`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_add_residuals_false` — `passed` — `smoke` — `tests/test_register_fit_result.py:L435`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_auto_alias_subframe_false` — `passed` — `smoke` — `tests/test_register_fit_result.py:L452`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_pull_type_both` — `passed` — `smoke` — `tests/test_register_fit_result.py:L407`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_pull_type_from_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L416`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_pull_type_mad_only` — `passed` — `smoke` — `tests/test_register_fit_result.py:L398`
- `test_register_fit_result.py::TestRegisterFitResultOptions::test_pull_type_rms_only` — `passed` — `smoke` — `tests/test_register_fit_result.py:L389`
- `test_register_fit_result.py::TestResultVerification::test_deepcopy_prevents_mutation` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1180`
- `test_register_fit_result.py::TestResultVerification::test_draw_fit_summary_plots_render_without_error` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1068`
- `test_register_fit_result.py::TestResultVerification::test_gaussian_overlay_added_to_pull_histogram` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1094`
- `test_register_fit_result.py::TestResultVerification::test_histograms_have_data` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1198`
- `test_register_fit_result.py::TestResultVerification::test_schema_roundtrip_metadata_values_match` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1152`
- `test_register_fit_result.py::TestResultVerification::test_validation_metrics_match_expected_values` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1120`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_export_mutation_isolated` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1629`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_import_mutation_isolated` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1641`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_json_strict_for_fit_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1679`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_no_phantom_fit_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1617`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_overwrite_replaces_not_merges` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1592`
- `test_register_fit_result.py::TestSchemaEdgeCases::test_roundtrip_then_draw` — `passed` — `smoke` — `tests/test_register_fit_result.py:L1659`
- `test_register_fit_result.py::TestSchemaPersistence::test_apply_schema_overwrites_with_warning` — `passed` — `smoke` — `tests/test_register_fit_result.py:L891`
- `test_register_fit_result.py::TestSchemaPersistence::test_apply_schema_restores_fit_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L871`
- `test_register_fit_result.py::TestSchemaPersistence::test_export_includes_fit_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L860`
- `test_register_fit_result.py::TestSchemaPersistence::test_exported_schema_is_json_serializable` — `passed` — `smoke` — `tests/test_register_fit_result.py:L922`
- `test_register_fit_result.py::TestSchemaPersistence::test_schema_roundtrip_preserves_formulas` — `passed` — `smoke` — `tests/test_register_fit_result.py:L902`
- `test_register_fit_result.py::TestSchemaPersistence::test_schema_without_fit_metadata_backward_compat` — `passed` — `smoke` — `tests/test_register_fit_result.py:L937`
- `test_register_fit_result.py::TestValidateFitMetadata::test_missing_columns_key` — `passed` — `smoke` — `tests/test_register_fit_result.py:L188`
- `test_register_fit_result.py::TestValidateFitMetadata::test_missing_formulas_key` — `passed` — `smoke` — `tests/test_register_fit_result.py:L181`
- `test_register_fit_result.py::TestValidateFitMetadata::test_missing_gb_columns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L195`
- `test_register_fit_result.py::TestValidateFitMetadata::test_non_dict_metadata` — `passed` — `smoke` — `tests/test_register_fit_result.py:L227`
- `test_register_fit_result.py::TestValidateFitMetadata::test_valid_metadata_passes` — `passed` — `smoke` — `tests/test_register_fit_result.py:L175`
- `test_register_fit_result.py::TestValidateFitMetadata::test_validate_raise_mode` — `passed` — `smoke` — `tests/test_register_fit_result.py:L202`
- `test_register_fit_result.py::TestValidateFitMetadata::test_validate_skip_mode` — `passed` — `smoke` — `tests/test_register_fit_result.py:L219`
- `test_register_fit_result.py::TestValidateFitMetadata::test_validate_warn_mode` — `passed` — `smoke` — `tests/test_register_fit_result.py:L209`
- `test_register_fit_result.py::TestValidationThresholds::test_custom_validation_thresholds` — `passed` — `smoke` — `tests/test_register_fit_result.py:L957`
- `test_register_fit_result.py::TestValidationThresholds::test_draw_fit_summary_with_thresholds` — `passed` — `smoke` — `tests/test_register_fit_result.py:L984`
- `test_register_fit_result.py::TestValidationThresholds::test_partial_threshold_override` — `passed` — `smoke` — `tests/test_register_fit_result.py:L970`

</details>

<details>
<summary><code>FIT.visualization</code> — 18 owned pytest nodes</summary>

- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_generates_figures` — `passed` — `smoke` — `tests/test_register_fit_result.py:L701`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_large_dataset_warns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L755`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_returns_validation` — `passed` — `smoke` — `tests/test_register_fit_result.py:L744`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_saves_pdf` — `passed` — `smoke` — `tests/test_register_fit_result.py:L724`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_saves_png` — `passed` — `smoke` — `tests/test_register_fit_result.py:L712`
- `test_register_fit_result.py::TestDrawFitSummaryBasic::test_unknown_fit_raises` — `passed` — `smoke` — `tests/test_register_fit_result.py:L736`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_entry_end_limits_data` — `passed` — `smoke` — `tests/test_register_fit_result.py:L819`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_exclude_filter` — `passed` — `smoke` — `tests/test_register_fit_result.py:L798`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_fit_columns_subset` — `passed` — `smoke` — `tests/test_register_fit_result.py:L831`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_include_filter` — `passed` — `smoke` — `tests/test_register_fit_result.py:L786`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_pull_transform_asinh` — `passed` — `smoke` — `tests/test_register_fit_result.py:L808`
- `test_register_fit_result.py::TestDrawFitSummaryOptions::test_unknown_category_warns` — `passed` — `smoke` — `tests/test_register_fit_result.py:L843`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_all_parameters_combined` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L342`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_default_no_annotations` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L321`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_statistics_adds_annotations` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L238`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_summary_adds_panel` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L294`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_show_validation_adds_indicator` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L266`
- `test_validation_display_adf.py::TestDrawFitSummaryIntegration::test_statistics_returned_in_results` — `passed` — `smoke` — `tests/test_validation_display_adf.py:L225`

</details>

## RDATAFRAME

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| 🧨 | **RDF.export** — Export to RDataFrame | 123 | 115 | 3 | 0 | 0 | 0 | 5 |  |
| ☑️ | **RDF.composite** — RDataFrame composite key support | 10 | 10 | 0 | 0 | 0 | 0 | 0 |  |

### Supporting tests

<details>
<summary><code>RDF.export</code> — 123 owned pytest nodes</summary>

- `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_error` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2473`
- `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree` — `failed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2515`
- `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_redefine` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2503`
- `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_skip` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2482`
- `test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_warn` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2494`
- `test_AliasDataFrameRDF.py::TestAddDefinesToRDF::test_add_defines_computes_values` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2353`
- `test_AliasDataFrameRDF.py::TestAddDefinesToRDF::test_add_defines_creates_column` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2329`
- `test_AliasDataFrameRDF.py::TestAddDefinesToRDF::test_add_defines_resolves_dependencies` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2340`
- `test_AliasDataFrameRDF.py::TestCacheToSnapshot::test_cache_creates_file` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2427`
- `test_AliasDataFrameRDF.py::TestChainSetup::test_chain_with_glob` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2394`
- `test_AliasDataFrameRDF.py::TestChainSetup::test_chain_with_list` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2385`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_check_dense_overflow_safe` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L212`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_check_dense_overflow_unsafe` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L226`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_generate_dense_cpp_expression_four_keys` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L255`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_generate_dense_cpp_expression_single_key` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L234`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_generate_dense_cpp_expression_three_keys` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L248`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_generate_dense_cpp_expression_two_keys` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L241`
- `test_AliasDataFrameRDF.py::TestCompositeKeyHelpers::test_get_composite_key_column_name` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L204`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyAuto::test_auto_keys_match_correctly` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L475`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyAuto::test_auto_selects_dense_for_compact_data` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L437`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyAuto::test_auto_selects_sparse_for_large_gaps` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L457`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyDense::test_single_column` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L305`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyDense::test_three_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L332`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyDense::test_two_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L314`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeyDense::test_with_explicit_max_values` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L351`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeySparse::test_basic_mapping` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L371`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeySparse::test_shuffled_subframe` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L390`
- `test_AliasDataFrameRDF.py::TestComputeCompositeKeySparse::test_sparse_with_gaps` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L414`
- `test_AliasDataFrameRDF.py::TestExtractDependencies::test_excludes_functions` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L177`
- `test_AliasDataFrameRDF.py::TestExtractDependencies::test_filter_known_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L184`
- `test_AliasDataFrameRDF.py::TestExtractDependencies::test_numpy_prefix_excluded` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L191`
- `test_AliasDataFrameRDF.py::TestExtractDependencies::test_simple_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L168`
- `test_AliasDataFrameRDF.py::TestExtractDependencies::test_subframe_column` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L173`
- `test_AliasDataFrameRDF.py::TestGenerateRdfCode::test_basic_generation` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L573`
- `test_AliasDataFrameRDF.py::TestGenerateRdfCode::test_empty_defines` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L590`
- `test_AliasDataFrameRDF.py::TestGenerateRdfCode::test_with_mt` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L584`
- `test_AliasDataFrameRDF.py::TestGetOrderedDefines::test_circular_dependency_detected` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L546`
- `test_AliasDataFrameRDF.py::TestGetOrderedDefines::test_cpp_expr_included` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L557`
- `test_AliasDataFrameRDF.py::TestGetOrderedDefines::test_diamond_dependency` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L517`
- `test_AliasDataFrameRDF.py::TestGetOrderedDefines::test_independent_aliases` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L535`
- `test_AliasDataFrameRDF.py::TestGetOrderedDefines::test_simple_chain` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L501`
- `test_AliasDataFrameRDF.py::TestJoinColumnsForSnapshot::test_returns_index_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2409`
- `test_AliasDataFrameRDF.py::TestModularRDFSetup::test_setup_rdf_can_count` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L916`
- `test_AliasDataFrameRDF.py::TestModularRDFSetup::test_setup_rdf_file_not_found` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L925`
- `test_AliasDataFrameRDF.py::TestModularRDFSetup::test_setup_rdf_has_columns` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L933`
- `test_AliasDataFrameRDF.py::TestModularRDFSetup::test_setup_rdf_returns_tuple` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L902`
- `test_AliasDataFrameRDF.py::TestRDataFrameBasics::test_define_chain` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L615`
- `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend` — `failed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L701`
- `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_friend_dot_notation` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L759`
- `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_friend_with_alias` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L792`
- `test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_indexed_friend` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L661`
- `test_AliasDataFrameRDF.py::TestRDataFrameIndexVerification::test_rdf_uses_single_key_index` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2571`
- `test_AliasDataFrameRDF.py::TestRDataFrameIndexVerification::test_rdf_uses_two_key_index` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2633`
- `test_AliasDataFrameRDF.py::TestRDataFrameIndexVerification::test_ttree_draw_uses_index_for_comparison` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2690`
- `test_AliasDataFrameRDF.py::TestRDataFrameWithRealSchema::test_ordered_defines_to_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L830`
- `test_AliasDataFrameRDF.py::TestRuntimeCompositeKey::test_3key_without_precomputed_uses_runtime_generation` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2151`
- `test_AliasDataFrameRDF.py::TestRuntimeCompositeKey::test_precomputed_composite_key_shuffled` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2251`
- `test_AliasDataFrameRDF.py::TestRuntimeCompositeKey::test_precomputed_composite_key_used` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2189`
- `test_AliasDataFrameRDF.py::TestRuntimeCompositeKey::test_return_composite_info_flag` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2121`
- `test_AliasDataFrameRDF.py::TestRuntimeCompositeKey::test_runtime_composite_key_join_correctness` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L2168`
- `test_AliasDataFrameRDF.py::TestShouldUseSparse::test_large_sparse_data` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L277`
- `test_AliasDataFrameRDF.py::TestShouldUseSparse::test_overflow_triggers_sparse` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L289`
- `test_AliasDataFrameRDF.py::TestShouldUseSparse::test_small_dense_data` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L266`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_clone_friend_setfile_main_shuffled_correctness` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1709`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_clone_friend_setfile_main_with_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1599`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_clone_friend_to_memfile_with_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1507`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend` — `failed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1965`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_multiple_subframes_runtime_composite` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1820`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_basic` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L955`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_composite_key_3keys` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1094`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_composite_key_with_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1210`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_friend_only_with_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1330`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_main_only_with_rdf` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1419`
- `test_AliasDataFrameRDF.py::TestTMemFileBranch::test_setfile_with_buildindex` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L1022`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_bitwise_and_preserved` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L116`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_bitwise_or_preserved` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L122`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_boolean_false` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L113`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_boolean_true` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L110`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_combined_expression` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L132`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_comparison_operators` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L146`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_logical_and_or` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L156`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_logical_not_converted` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L127`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_math_e` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L83`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_math_pi` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L65`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_np_e` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L71`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_abs` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L52`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_e_full_module` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L77`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_pi` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L55`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_pi_full_module` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L59`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_sqrt` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L49`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_pi_and_e_combined` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L89`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_power_float` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L106`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_power_simple` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L96`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_power_with_parentheses` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L100`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_subframe_alias_replacement` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L139`
- `test_AliasDataFrameRDF.py::TestToCppExpr::test_ternary` — `passed` — `smoke` — `tests/test_AliasDataFrameRDF.py:L150`
- `test_rdf_integration.py::TestOrderedDefinesWithSchema::test_cpp_expr_conversion` — `passed` — `integration` — `tests/test_rdf_integration.py:L89`
- `test_rdf_integration.py::TestOrderedDefinesWithSchema::test_isvalid_deep_dependency` — `passed` — `integration` — `tests/test_rdf_integration.py:L73`
- `test_rdf_integration.py::TestOrderedDefinesWithSchema::test_multiple_alias_ordering` — `passed` — `integration` — `tests/test_rdf_integration.py:L57`
- `test_rdf_integration.py::TestOrderedDefinesWithSchema::test_simple_alias_ordering` — `passed` — `integration` — `tests/test_rdf_integration.py:L46`
- `test_rdf_integration.py::TestOrderedDefinesWithSchema::test_subframe_dependencies_identified` — `passed` — `integration` — `tests/test_rdf_integration.py:L99`
- `test_rdf_integration.py::TestRDFIntegration::test_1key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration.py:L186`
- `test_rdf_integration.py::TestRDFIntegration::test_2key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration.py:L204`
- `test_rdf_integration.py::TestRDFIntegration::test_3key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration.py:L170`
- `test_rdf_integration.py::TestRDFIntegration::test_ordered_defines_with_real_schema` — `passed` — `integration` — `tests/test_rdf_integration.py:L121`
- `test_rdf_integration.py::TestRDFIntegration::test_rdf_define_chain` — `passed` — `integration` — `tests/test_rdf_integration.py:L149`
- `test_rdf_integration.py::TestRDFIntegration::test_setup_tree_with_friends` — `passed` — `integration` — `tests/test_rdf_integration.py:L132`
- `test_rdf_integration_final.py::TestOrderedDefinesWithSchema::test_cpp_expr_conversion` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L89`
- `test_rdf_integration_final.py::TestOrderedDefinesWithSchema::test_isvalid_deep_dependency` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L73`
- `test_rdf_integration_final.py::TestOrderedDefinesWithSchema::test_multiple_alias_ordering` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L57`
- `test_rdf_integration_final.py::TestOrderedDefinesWithSchema::test_simple_alias_ordering` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L46`
- `test_rdf_integration_final.py::TestOrderedDefinesWithSchema::test_subframe_dependencies_identified` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L99`
- `test_rdf_integration_final.py::TestRDFIntegration::test_1key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L186`
- `test_rdf_integration_final.py::TestRDFIntegration::test_2key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L204`
- `test_rdf_integration_final.py::TestRDFIntegration::test_3key_subframe_access` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L170`
- `test_rdf_integration_final.py::TestRDFIntegration::test_ordered_defines_with_real_schema` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L121`
- `test_rdf_integration_final.py::TestRDFIntegration::test_rdf_define_chain` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L149`
- `test_rdf_integration_final.py::TestRDFIntegration::test_setup_tree_with_friends` — `passed` — `integration` — `tests/test_rdf_integration_final.py:L132`
- `test_rdf_real_data.py::TestRDFMultiThread::test_mt_histogram` — `skipped` — `smoke` — `tests/test_rdf_real_data.py:L205`
- `test_rdf_real_data.py::TestRDFRealData::test_2d_histogram` — `skipped` — `smoke` — `tests/test_rdf_real_data.py:L139`
- `test_rdf_real_data.py::TestRDFRealData::test_correlation_coefficient` — `skipped` — `smoke` — `tests/test_rdf_real_data.py:L166`
- `test_rdf_real_data.py::TestRDFRealData::test_dyC2_histogram` — `skipped` — `smoke` — `tests/test_rdf_real_data.py:L63`
- `test_rdf_real_data.py::TestRDFRealData::test_dzC2_histogram` — `skipped` — `smoke` — `tests/test_rdf_real_data.py:L108`

</details>

<details>
<summary><code>RDF.composite</code> — 10 owned pytest nodes</summary>

- `test_ttree_draw_subframe.py::TestTTreeDrawMultiKeySubframe::test_multikey_expression` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L405`
- `test_ttree_draw_subframe.py::TestTTreeDrawMultiKeySubframe::test_multikey_friend_index` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L385`
- `test_ttree_draw_subframe.py::TestTTreeDrawStrictAccuracy::test_strict_elementwise_equality` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L307`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_alias_matches_draw_expression` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L173`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_draw_2d_main_vs_friend` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L216`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_draw_expression_with_friend` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L149`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_draw_main_column` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L100`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_draw_with_cut_on_friend` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L242`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_draw_with_friend_tree` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L122`
- `test_ttree_draw_subframe.py::TestTTreeDrawSubframe::test_file_structure` — `passed` — `smoke` — `tests/test_ttree_draw_subframe.py:L83`

</details>

## INVARIANCE

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **INV.cross_module** — Cross-module invariance tests | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| ✅ | **INV.draw_surface_consistency** — Same-spec numerical consistency across draw(), draw_batch(), and draw_figures(), including explicit supported-surface refusals and closure reconciliation (A3) | 27 | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| ✅ | **INV.eager_lazy_slot_symmetry** — Eager/lazy expression-slot causality and exact dependency-load symmetry across selection/expression/weights/group_by/facet/vector/subframe compositions (A4) | 24 | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| ✅ | **INV.realdata_acceptance** — Deterministic real-data/gallery acceptance and state invariance — full-stack composition, provenance, environment contracts, G7.32/G7.33/G7.34 evidence, GB prepared-state reuse, and logical-state mutation falsifiers (A5) | 42 | 42 | 0 | 0 | 0 | 0 | 0 | 42 |

### Supporting tests

<details>
<summary><code>INV.cross_module</code> — 8 owned pytest nodes</summary>

- `test_I17_full_pipeline_invariance.py::TestI17FullPipelineInvariance::test_I17_1_register_export_read_materialize_roundtrip` — `passed` — `invariance` — `tests/test_I17_full_pipeline_invariance.py:L47`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I1_load_mode` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L81`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I2_backend` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L116`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I3_subframe_join` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L160`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I4_compression` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L198`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I5_schema` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L240`
- `test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I6_order` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L276`
- `test_invariance_smoke.py::TestSmokeSummary::test_smoke_test_count` — `passed` — `invariance` — `tests/test_invariance_smoke.py:L320`

</details>

<details>
<summary><code>INV.draw_surface_consistency</code> — 27 owned pytest nodes</summary>

- `test_phase_13_77_realdata_invariance_harness.py::test_a3_01_hist_case_declares_one_spec_for_all_three_surfaces` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1307`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_02_hist_same_spec_passes_draw_draw_batch_draw_figures` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1320`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_03_draw_figures_is_an_executed_surface_not_a_documented_exception` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1338`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_04_profile_case_declares_one_spec_for_all_three_surfaces` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1347`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_05_profile_same_spec_passes_draw_draw_batch_draw_figures` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1362`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_06_surface_stats_corruption_changes_strict_gate_zero_to_one` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1381`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_07_groupby_case_declares_one_spec_and_group_resolved_observables` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1438`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_08_groupby_same_spec_passes_draw_draw_batch_draw_figures` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1461`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_09_profile_data_missing_column_fails_loudly` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1477`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_10_group_specific_mismatch_is_not_hidden_by_global_stats` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1485`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_11_facet_case_declares_supported_surfaces_and_separate_refusal_contract` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1548`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_12_facet_same_spec_passes_supported_surfaces_and_refuses_draw_figures` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1584`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_13_facet_specific_mismatch_reaches_strict_gate` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1613`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_14_subframe_case_declares_one_spec_and_keyed_observables` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1715`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_15_subframe_same_spec_passes_draw_draw_batch_draw_figures` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1739`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_16_subframe_specific_mismatch_reaches_strict_gate` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1753`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_17_selection_vector_case_declares_branch_resolved_observables` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1853`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_18_selection_vector_same_spec_passes_all_three_surfaces` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1883`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_19_selection_vector_branch_mismatch_is_not_hidden_by_derived_value` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1899`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_20_profile_bin_case_and_histogram_disposition_are_explicit` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1975`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_21_sparse_profile_bins_match_all_three_surfaces` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2027`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_22_profile_bin_error_mismatch_reaches_strict_gate` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2054`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_23_closure_reconciliation_covers_every_required_family_without_orphans` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2117`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_24_closure_reconciliation_fails_closed_on_missing_family_or_histogram_upgrade` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2160`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_25_complete_a3_manifest_carries_closure_record` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2200`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_26_closure_contract_map_blocks_legal_required_case_drift` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2229`
- `test_phase_13_77_realdata_invariance_harness.py::test_a3_27_missing_family_manifest_persists_blocked_closure` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2289`

</details>

<details>
<summary><code>INV.eager_lazy_slot_symmetry</code> — 24 owned pytest nodes</summary>

- `test_phase_13_77_realdata_invariance_harness.py::test_a4_01_selection_slot_contract_is_executable_and_manifest_visible` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2387`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_02_selection_slot_both_proves_materialization_and_exact_lazy_loads` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2412`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_03_m2_preload_contamination_is_invalid_fixture_not_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2436`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_04_scalar_catalogue_and_runner_binding_are_machine_authoritative` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2502`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-COMPOUND-EXPR-01-expected_loaded4]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2540`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-EXPR-01-expected_loaded0]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2540`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-FACET-BY-01-expected_loaded3]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2540`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-GROUP-BY-01-expected_loaded2]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2540`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-WEIGHTS-01-expected_loaded1]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2540`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_10_source_contract_is_checked_before_observable_path_resolution` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2562`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_11_vector_catalogue_and_refusal_contracts_are_machine_visible` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2651`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_12_13_vector_slots_prove_eager_materialization_and_exact_lazy_loads[I3-SELECTION-VECTOR-01-expected_loaded0]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2698`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_12_13_vector_slots_prove_eager_materialization_and_exact_lazy_loads[I3-WEIGHTS-VECTOR-01-expected_loaded1]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2698`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_14_15_subframe_vector_refusals_hold_in_eager_and_lazy_modes` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2724`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_16_complete_catalogue_and_contract_reconciliation_is_bidirectional` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2786`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_17_slot_alias_exclusivity_is_machine_locked_and_second_slot_fails` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2828`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_18_eager_target_only_materialization_is_proven_with_shared_dependency_alias` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2845`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_19_shared_physical_dependency_all_alias_materialization_false_green_is_caught` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2860`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_20_subframe_scalar_causality_and_vector_refusal_ownership_are_complete` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2879`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_21_historical_closure_ledger_has_no_implicit_obligation_and_blocks_on_loss` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2911`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_22_closure_ledger_status_and_owner_semantics_are_locked` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_23_refusal_contract_locks_exact_bug_identity_and_proof_scope` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L2989`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_24_qualified_reference_matching_is_token_exact_and_near_name_replacement_blocks` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3025`
- `test_phase_13_77_realdata_invariance_harness.py::test_a4_25_duplicate_historical_ledger_ids_block_and_persist` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3064`

</details>

<details>
<summary><code>INV.realdata_acceptance</code> — 42 owned pytest nodes</summary>

- `test_phase_13_77_realdata_invariance_harness.py::test_a5_01_full_stack_case_is_bounded_and_registry_valid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3200`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_02_full_stack_matches_independent_oracle_in_eager_and_lazy_modes` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3228`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_03_independent_oracle_catches_one_lazy_group_bin_corruption` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3262`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_04_preloaded_unrelated_branch_is_invalid_fixture_not_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3309`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_05_unavailable_realdata_environment_skips_without_running_product` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3413`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_06_realdata_case_pins_eager_fraction_20pct_seed42` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3428`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_07_realdata_runner_records_actual_sample_identity_and_g7_evidence` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3446`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_08_realdata_gate_persists_sample_provenance_in_manifest` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3479`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_09_applicable_optional_gallery_none_is_fail_not_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3498`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_10_applicable_g7_exception_fails_closed_and_wrong_seed_is_invalid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3511`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_11_missing_required_gallery_callable_is_not_a_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3530`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_12_unexpected_gallery_import_exception_is_not_environment_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3548`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_13_finite_bookkeeping_cannot_hide_all_nan_profile_y` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3568`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_14_sample_monkeypatch_restored_when_build_fails` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3584`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_15_lazy_full_case_is_bounded_and_registry_valid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3670`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_16_lazy_full_g7_records_real_lazy_branch_expansion` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3688`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_17_lazy_full_eager_in_disguise_is_invalid_fixture` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3716`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_18_lazy_full_requires_on_demand_branch_expansion` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3730`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_19_lazy_full_forbids_sampling_and_restores_sample` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3744`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_20_realdata_lazy_setup_exact_timems_blocker_is_error_contract_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3776`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_21_realdata_lazy_setup_wrong_key_does_not_false_green` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3793`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_22_realdata_lazy_setup_success_forces_contract_review` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3807`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_23_g7_33_case_is_bounded_eager_fraction_and_registry_valid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3902`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_24_g7_33_records_sample_gb_subframe_predicted_and_public_evidence` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3922`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_25_g7_33_nonfinite_predicted_column_fails_closed` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3961`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_26_g7_33_missing_calibbias1_subframe_fails_closed` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3975`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_27_g7_33_optional_none_is_fail_not_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L3989`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_28_g7_33_sequence_stats_profile_means_are_valid_public_evidence` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4001`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_29_g7_33_sequence_stats_bookkeeping_cannot_hide_nan_profiles` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4016`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_30_g7_34_case_declares_two_real_state_consistency_observables` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4100`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_31_g7_34_reuses_prepared_state_and_executes_two_comparisons` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4120`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_32_g7_34_refit_attempt_is_poisoned_and_fails` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4151`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_33_g7_34_predicted_state_mutation_fails_consistency` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4166`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_34_g7_34_calibbias1_mutation_fails_consistency` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4182`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_35_g7_34_failure_restores_exact_calibration_binding` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_36_g7_34_logical_state_case_declares_four_exact_observables` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4297`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_37_g7_34_healthy_logical_state_executes_four_comparisons` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4323`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_38_g7_34_alias_definition_mutation_fails` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4349`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_39_g7_34_subframe_index_definition_mutation_fails` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4364`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_40_g7_34_parent_structure_mutation_fails_and_fingerprint_is_deterministic` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4379`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_41_definition_digest_ignores_only_export_created_at` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4404`
- `test_phase_13_77_realdata_invariance_harness.py::test_a5_42_phase_13_77_capability_taxonomy_registration_is_exact` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L4454`

</details>

## DISPATCH

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **DISPATCH.adf_routing** — adf.draw/draw_figures route through DFDraw.draw() (auto pre-resolution, overlay strings, type aliases, 3-var profile promotion) | 28 | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| ✅ | **DISPATCH.error_visibility** — Batch-surface error visibility (on_error='raise' defaults; A-10/E-3/E-4 guards; draw_fit_summary documented exception) | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 15 |
| 🧨 | **DISPATCH.dict_dispatch** — Draw-path dict dispatch frame: draw()/draw_batch()/draw_figures() hand dfdraw only the needed columns (get_required_branches ∪ materialized alias names ∪ subframe index cols); structural column-count gate + peak-RSS + volume-invariance memory gates + dict≡full-frame equivalence (AC-1/1a/1b incl. subframe single+multi-level) + loud no-silent-full-frame fallback | 102 | 101 | 1 | 0 | 0 | 0 | 0 | 42 |

### Supporting tests

<details>
<summary><code>DISPATCH.adf_routing</code> — 28 owned pytest nodes</summary>

- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T1_draw_overlay_string` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L117`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T2_draw_histo_alias` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L122`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T3_draw_profile2d_no_regression` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L128`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T4_draw_scatter3d_no_regression` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L133`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T5_figures_overlay_in_spec` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L139`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T6_figures_histo_alias_in_spec` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L148`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T7_5_figures_scatter3d_clean_error` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L171`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage::test_T7_figures_profile2d_in_spec` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L157`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_T10_auto_title_profile_via_figures` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L230`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_T11_fit_gauss_via_figures` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L240`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_T12_selection_vector_via_figures` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L249`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_T8_central_median_via_figures` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L196`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_T9_time_format_via_figures` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L217`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks::test_Tauto_sentinel_regression` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L260`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T16_draw_return_shape` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L320`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T17_figures_return_shape` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L327`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T18_facet_by_lazy_alias_regression_lock` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L336`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T19_subframe_rewrite_regression_lock` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L345`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T20_data_source_preserved` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L350`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression::test_T21_cleanup_after_dispatch` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L357`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup6ProfilePromotion::test_T3b_draw_3var_profile_promotion` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L403`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup6ProfilePromotion::test_T3c_draw_vector_expr_profile_not_promoted` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L423`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup6ProfilePromotion::test_T3d_colon_counter_units` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L439`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup6ProfilePromotion::test_T7b_figures_3var_profile_promotion` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L428`
- `test_phase_13_56_adf_post_audit.py::TestG6BatchShims::test_TG6a_batch_literal_auto` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L242`
- `test_phase_13_56_adf_post_audit.py::TestG6BatchShims::test_TG6b_batch_three_var_profile_promotion` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L259`
- `test_phase_13_56_adf_post_audit.py::TestH1DrawHelp::test_TH1_draw_help_full_surface` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L349`
- `test_phase_13_56_adf_post_audit.py::TestR1ThreeLevelFacetLock::test_TR1_three_level_facet_multi_figure_return` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L323`

</details>

<details>
<summary><code>DISPATCH.error_visibility</code> — 15 owned pytest nodes</summary>

- `test_phase_13_55_adf_dispatch_audit.py::TestGroup3OnError::test_T13_invalid_type_default_raises` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L289`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup3OnError::test_T14_invalid_type_explicit_skip` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L296`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup3OnError::test_T15_valid_type_raise_regression` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L305`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup5DrawBatchOnError::test_T22_batch_invalid_default_raises` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L376`
- `test_phase_13_55_adf_dispatch_audit.py::TestGroup5DrawBatchOnError::test_T23_batch_explicit_skip` — `passed` — `invariance` — `tests/test_phase_13_55_adf_dispatch_audit.py:L382`
- `test_phase_13_56_adf_post_audit.py::TestG1Profile2dGuard::test_TG1a_profile2d_spec_default_raises` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L53`
- `test_phase_13_56_adf_post_audit.py::TestG1Profile2dGuard::test_TG1b_profile2d_spec_skip_placeholder` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L62`
- `test_phase_13_56_adf_post_audit.py::TestG1Profile2dGuard::test_TG1c_three_var_profile_promotion_reaches_guard` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L75`
- `test_phase_13_56_adf_post_audit.py::TestG2FacetByGuard::test_TG2a_facet_by_spec_default_raises_no_leak` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L94`
- `test_phase_13_56_adf_post_audit.py::TestG2FacetByGuard::test_TG2b_facet_by_spec_skip_placeholder_no_leak` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L111`
- `test_phase_13_56_adf_post_audit.py::TestG3SelectionAliasMatrix::test_TG3_selection_alias_behavior_matrix` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L136`
- `test_phase_13_56_adf_post_audit.py::TestG5AstypeTypeTokens::test_TG5_astype_int_actionable_error` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L217`
- `test_phase_13_56_adf_post_audit.py::TestG5AstypeTypeTokens::test_TG5_negative_control_other_typeerrors_unchanged` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L225`
- `test_phase_13_56_adf_post_audit.py::TestG7CoverageLocks::test_TG7a_weights_alias_figures_and_batch` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L279`
- `test_phase_13_56_adf_post_audit.py::TestG7CoverageLocks::test_TG7b_entry_window_via_figures` — `passed` — `invariance` — `tests/test_phase_13_56_adf_post_audit.py:L307`

</details>

<details>
<summary><code>DISPATCH.dict_dispatch</code> — 102 owned pytest nodes</summary>

- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw0]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw1]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw2]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw3]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw4]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw5]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1_dict_equals_full_frame[kw6]` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L271`
- `test_phase1361_dict.py::test_ac1a_subframe_dict_equals_full_frame` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L281`
- `test_phase1361_dict.py::test_ac1b_batch_dict_equals_full_frame` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L332`
- `test_phase1361_dict.py::test_ac1b_batch_subframe_dict_equals_full_frame` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L305`
- `test_phase1361_dict.py::test_ac1b_figures_dict_equals_full_frame` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L350`
- `test_phase1361_dict.py::test_block_count_self_df_unchanged` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L373`
- `test_phase1361_dict.py::test_color_list_values_not_projected` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L168`
- `test_phase1361_dict.py::test_decoy_named_like_tokens_not_overprojected` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L153`
- `test_phase1361_dict.py::test_peak_rss_dict_below_full_frame` — `failed` — `smoke` — `tests/test_phase1361_dict.py:L437`
- `test_phase1361_dict.py::test_resolver_failure_warns_and_not_fullframe` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L129`
- `test_phase1361_dict.py::test_structural_dict_drops_decoys` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L182`
- `test_phase1361_dict.py::test_structural_negative_control_full_frame_keeps_decoys` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L197`
- `test_phase1361_dict.py::test_volume_invariance_dispatch_excludes_decoys` — `passed` — `invariance` — `tests/test_phase1361_dict.py:L397`
- `test_phase1361_memory.py::test_draw_path_peak_rss_bounded` — `passed` — `invariance` — `tests/test_phase1361_memory.py:L53`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_auto_struct_gains_member_on_refresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L439`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_explicit_struct_never_broadened` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L450`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_ensure_columns` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L418`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_fresh_get_required_branches` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L413`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_release_then_reload` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L478`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_leaf_two_parents_no_crosstalk` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L487`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_same_size_catalog_change_detected` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L423`
- `test_phase_13_75_lazy_struct_repair.py::TestStage10_BridgesAndRefresh::test_schema_roundtrip_provenance_and_precedence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L460`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_chain_initial_branches_normalized_and_full` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L507`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_tree_initial_branches_full_structure` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L515`
- `test_phase_13_75_lazy_struct_repair.py::TestStage11_ChainWorkflow::test_two_file_chain_value_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L500`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_batch_frame_capture_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L547`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_draw_frame_capture` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L540`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_figures_frame_capture_kwargs_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L554`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_composite` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L592`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_minimal` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L569`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_returned_stats_oracle_selection` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L583`
- `test_phase_13_75_lazy_struct_repair.py::TestStage12_EffectiveSpecAndOracles::test_reused_spec_object_second_call_works` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L563`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_lazy_reports_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L613`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_describe_structure_contract_and_no_side_effects` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L605`
- `test_phase_13_75_lazy_struct_repair.py::TestStage13_DiagnosticsAndErrors::test_fault_injected_projection_raises_adf_error` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L623`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_batch_capture_excludes_physical_and_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L751`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_chain` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L671`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_constructor_schema_precedence_tree` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L660`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_defaults_autoload_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L709`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_diagnostics_full_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L769`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_batch_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L726`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fault_injection_figures_surface` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L738`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_figures_per_figure_defaults_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L716`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_fp_not_cached_on_failed_reconciliation` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L680`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_full_structure_completion_failure_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L693`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_refresh_member_usable_via_dot_grammar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L641`
- `test_phase_13_75_lazy_struct_repair.py::TestStage14_FinalCrrCorrections::test_second_call_different_requirements_fresh` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L761`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_caller_spec_never_mutated_three_contexts` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L788`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_classifier_exception_reports_context` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L862`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_describe_structure_print_path` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L829`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_guard_ignores_titles_matching_internals` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L856`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_per_bin_profile_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L838`
- `test_phase_13_75_lazy_struct_repair.py::TestStage15_Delta2::test_validate_aliases_fresh_lazy` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L817`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_chain_reader_delegates` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L84`
- `test_phase_13_75_lazy_struct_repair.py::TestStage1_ReaderClassification::test_tree_reader_three_valued` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L78`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_c1_unknown_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L110`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_clean_registration_and_provenance` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L94`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_explicit_register_precedence_survives` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L122`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h2_count_helper_never_a_struct` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L100`
- `test_phase_13_75_lazy_struct_repair.py::TestStage2_DetectorPolicy::test_h3_collision_blocks_auto_member` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L104`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_chain_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L136`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_auto_at_tree_construction` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L132`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_eager_noop` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L159`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_fp_cache_prevents_rescan` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L150`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_idempotence` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L140`
- `test_phase_13_75_lazy_struct_repair.py::TestStage3_Lifecycle::test_tree_initial_branches_normalized` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L164`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_exact_loading_decoy_control` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L179`
- `test_phase_13_75_lazy_struct_repair.py::TestStage4_LoadRename::test_full_struct_loading_documented_behavior` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L173`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_alias_over_struct_equals_direct` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L199`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_auto_equals_explicit_registration` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L214`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_builtin_named_member` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L226`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_internal_column` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L194`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_dot_eval_equals_physical_oracle` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L189`
- `test_phase_13_75_lazy_struct_repair.py::TestStage5_RewriteEval::test_tree_equals_chain` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L207`
- `test_phase_13_75_lazy_struct_repair.py::TestStage6_ProjectionDispatch::test_reduced_frame_keeps_internal_drops_decoys` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L234`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_composite_production_expression` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L251`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_batch_surface` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L297`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_figures_surface_dict_and_short_form` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L303`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_draw_reduced_equals_full_stats` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L311`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_entry_sliced_draw` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L318`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_minimal_profile_default_dispatch` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L247`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_persisted_schema_control` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L323`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_production_composition_facet_quantiles` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L289`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_slot_composition` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L256`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_color_facet_weights_slots` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L280`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_facet_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L274`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_group_by_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L268`
- `test_phase_13_75_lazy_struct_repair.py::TestStage7_DrawSurfaces::test_struct_in_selection_slot` — `passed` — `invariance` — `tests/test_phase_13_75_lazy_struct_repair.py:L262`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_no_pandas_undefinedvariable_reaches_user` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L336`
- `test_phase_13_75_lazy_struct_repair.py::TestStage8_ErrorContract::test_unknown_member_is_loud` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L330`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_dtype_drift_warns_still_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L368`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_fixed_size_array_never_scalar` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L404`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_chain_detection_raises_loudly` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L359`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_mixed_scalar_jagged_raises` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L351`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_scalar_scalar_ok` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L347`
- `test_phase_13_75_lazy_struct_repair.py::TestStage9_ChainShapeContract::test_union_absence_not_scalar_proof` — `passed` — `smoke` — `tests/test_phase_13_75_lazy_struct_repair.py:L377`

</details>

## TESTING

| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |
|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|
| ✅ | **TESTING.capability_matrix_index** — Capability Matrix diagnostic index tooling<br><sub>Proves the shared Capability Matrix semantic model, Markdown/HTML parity, AI-readable JSON export, node source locators, phase/focused-run evidence and reviewer-packet custody rules.</sub> | 22 | 22 | 0 | 0 | 0 | 0 | 0 | 1 |
| ✅ | **TESTING.phase13_77_harness** — PHASE_13_77 acceptance-harness integrity — CaseSpec/FigureContract registry, comparator/gate fail-closed behavior, manifest reconciliation, environment gating, and anti-false-green controls (A1+A2) | 155 | 155 | 0 | 0 | 0 | 0 | 0 | 155 |

### Supporting tests

<details>
<summary><code>TESTING.capability_matrix_index</code> — 22 owned pytest nodes</summary>

- `test_phase_13_76_capability_matrix_tooling.py::TestFocusedRunnerContract::test_T15_manifest_is_execution_authority_for_both_focused_lanes` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L396`
- `test_phase_13_76_capability_matrix_tooling.py::TestFocusedRunnerContract::test_T8_focused_normal_and_raw_use_same_selection` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L364`
- `test_phase_13_76_capability_matrix_tooling.py::TestFocusedRunnerContract::test_T9_both_focused_lanes_use_configured_workers` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L381`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixAccountingAndPhase::test_T13_unique_matched_nodes_are_distinct_from_feature_associations` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L318`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixAccountingAndPhase::test_T14_runner_phase_provenance_is_adf_specific` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L353`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixOutcomeSemantics::test_T1_mapped_xfail_cannot_be_verified` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L141`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixOutcomeSemantics::test_T2_clean_invariance_feature_remains_verified` — `passed` — `invariance` — `tests/test_phase_13_76_capability_matrix_tooling.py:L159`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixOutcomeSemantics::test_T3_smoke_plus_mapped_xfail_is_broken` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L173`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixOutcomeSemantics::test_T4_skip_is_neutral_and_does_not_promote` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L190`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixOutcomeSemantics::test_T5_xpass_is_visible_but_does_not_promote` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L204`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixRendererParity::test_T11_historical_interactive_html_surface_is_preserved` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L262`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixRendererParity::test_T12_feature_permalink_expands_supporting_tests` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L287`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixRendererParity::test_T6_markdown_html_summary_parity` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L219`
- `test_phase_13_76_capability_matrix_tooling.py::TestMatrixRendererParity::test_T7_per_feature_markdown_html_status_parity` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L245`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379DiagnosticIndex::test_T16_json_projection_is_ai_readable_and_has_source_locators` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L434`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379DiagnosticIndex::test_T17_markdown_contains_description_and_expandable_owned_nodes` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L460`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379DiagnosticIndex::test_T18_html_contains_description_and_source_locator_surface` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L477`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379DiagnosticIndex::test_T19_runner_defaults_focus_to_phase_13_79_and_packages_candidates` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L495`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379MachineSurfaceExport::test_T21_json_carries_full_slot_grid_cells_and_seams` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L520`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379MachineSurfaceExport::test_T22_semantic_payload_digest_is_deterministic_and_surface_sensitive` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L543`
- `test_phase_13_76_capability_matrix_tooling.py::TestPhase1379SlotGridRegistration::test_T20_slot_grid_contract_summary_is_machine_derived` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L504`
- `test_phase_13_76_capability_matrix_tooling.py::TestReviewerPacketCustody::test_T10_manifest_and_zip_are_finalized_after_matrix_rendering` — `passed` — `smoke` — `tests/test_phase_13_76_capability_matrix_tooling.py:L409`

</details>

<details>
<summary><code>TESTING.phase13_77_harness</code> — 155 owned pytest nodes</summary>

- `test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_raises_on_unresolvable_path` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L158`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_rejects_array_declared_flat` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L165`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_resolves_flat` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L152`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_never_returns_a_silent_default` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L135`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_bad_figure_key` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L124`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_bad_plot_index` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L118`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_wrong_batch_case_key` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L112`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_wrong_return_shape` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L130`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L102`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw_batch]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L102`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw_figures]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L102`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_consistency_case_passes_across_three_surfaces` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L218`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_correctness_case_agrees_with_numpy` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L237`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_error_contract_draw_figures_refuses_facet_by` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L256`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_faceted_stats_have_a_different_shape` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L173`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_mutation_corrupted_surface_cannot_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L274`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_1_case_comparing_nothing_cannot_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L319`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_2_anchor_is_immune_to_product_mutation` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L330`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_environment_gated_skip_is_still_legal` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L363`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_mandatory_skip_fails_closed` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L356`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_registry_rejects_a_mandatory_case_that_can_only_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L378`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_1_unknown_comparator_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L383`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_2_unimplemented_oracle_source_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L393`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_3_error_contract_failure_always_gates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L401`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_4_manifest_carries_the_comparison_contract` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L410`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_accepts_a_clean_case` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L186`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_duplicate_case_id` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L211`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_floating_comparator_without_rationale` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L204`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over0-no claim]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over1-owner_on_failure]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over2-sampled-lazy]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over3-known_bug_id]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over4-conflated]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over5-declares no observable]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L198`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_always_gates_on_invalid_fixture` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L310`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_ignores_a_failing_known_bug` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L303`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_is_non_zero_on_mandatory_failure` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L297`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_undeclarable_observable_is_invalid_fixture_not_pass` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L228`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[accepted_envelope]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_group_count]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_panels]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_traces]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[panel_roles]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[primary_comparison]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[proof_kind]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[residual_definition]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L448`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_footer_and_manifest_share_one_source` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L465`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_footer_carries_the_three_required_blocks` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L457`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_manifest_carries_the_new_contract_fields` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L498`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over0-no setup_contract]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L439`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over1-no preconditions]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L439`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over2-no figure_contract]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L439`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over3-applicability_reason]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L439`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over4-always applicable]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L439`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_aligned_figure_contract_is_accepted` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L560`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L573`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[FAMILY_MUTATION:hist-anchor-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L573`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[GLOBAL_MUTATION:-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L573`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[GLOBAL_MUTATION:M1-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L573`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[corrupt one surface -> FAIL-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L573`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over0-case_ids is empty]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L545`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over1-case_ids]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L545`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over2-proof_kind]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L545`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_inapplicable_case_never_invokes_the_product` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L511`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_inapplicable_short_circuits_every_runner` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L528`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_applicable_failure_gates_whatever_the_gate_class` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L587`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_audit_detects_a_planted_orphan` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L645`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_correctness_runner_rejects_a_stats_label` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L615`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_declared_source_must_match_the_executed_path` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L604`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_future_staged_fields_are_recorded_with_their_stage` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L657`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_matching_source_labels_still_execute` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L626`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_no_caseSpec_field_is_declared_but_unread` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L631`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_unavailable_environment_may_still_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L596`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_applicable_skip_always_gates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L677`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_audit_detects_a_field_with_no_reader` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L740`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_audit_is_derived_not_declared` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L727`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_environment_blocked_may_not_be_applicable` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L694`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-DIAGNOSTIC]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-FAIL]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-INVALID_FIXTURE]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-PASS]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-SKIP]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-DIAGNOSTIC]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-FAIL]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-INVALID_FIXTURE]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-PASS]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-SKIP]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L708`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_manifest_carries_title_and_case_schema_version` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L756`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_unavailable_environment_may_still_skip` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L687`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_unknown_status_fails_closed` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L720`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_audit_detects_a_name_colliding_field` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L820`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_audit_is_receiver_narrowed` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L809`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_complete_declared_set_still_passes` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L794`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_coverage_gaps_names_the_dropped_case` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L803`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_declared_case_with_no_result_gates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L772`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_duplicate_result_gates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L780`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-DIAGNOSTIC-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-FAIL-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-INVALID_FIXTURE-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-PASS-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-SKIP-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-DIAGNOSTIC-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-FAIL-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-INVALID_FIXTURE-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-PASS-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-SKIP-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-SOMETHING_NEW-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L869`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_iterating_a_field_does_not_promote_its_element` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L836`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_no_per_case_schema_version` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L847`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_result_for_an_undeclared_case_gates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L787`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseResult-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpec | None-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpec-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpecView-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[FakeCaseSpec-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[NotACaseSpec-False]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[Sequence[CaseSpec]-True]` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L936`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_complete_run_still_passes` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L978`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_duplicate_declared_id_is_caught_by_reconcile` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L968`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_empty_declared_set_is_refused_at_both_doors` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L889`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_gate_and_coverage_derive_from_one_authority` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L957`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_missing_declared_case_appears_in_the_manifest` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L944`
- `test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_receiver_evidence_is_scope_local` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L898`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_01_scalar_exact_pass_and_fail` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L991`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_02_scalar_close_atol_boundary` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L997`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_03_scalar_close_rtol_boundary` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1004`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_04_scalar_nan_and_infinity_rules` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1011`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_05_scalar_refuses_array_nonnumeric_and_unknown` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1021`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_06_array_shape_is_exact` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1030`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_07_array_exact_reports_mismatch_coordinates` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1035`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_08_array_close_elementwise_and_nan` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1043`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_09_array_close_rejects_non_numeric` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1053`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_10_a1_comparator_api_remains_compatible` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1058`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_11_tolerance_exact_contract_is_valid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1064`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_12_tolerance_close_contract_is_valid` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1069`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_13_negative_atol_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1076`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_14_negative_rtol_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1082`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_15_nan_tolerance_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1088`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_16_infinite_tolerance_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1094`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_17_exact_cannot_carry_ignored_tolerance` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1100`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_18_close_zero_tolerance_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1105`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_19_close_requires_rationale` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1111`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_20_unknown_comparator_is_refused` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1117`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_21_compare_observable_uses_declared_scalar_contract` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1122`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_22_compare_observable_uses_declared_array_contract` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1128`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_23_comparison_evidence_is_json_ready` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1134`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_24_consistency_runner_records_structured_evidence` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1147`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_25_correctness_runner_records_structured_evidence` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1163`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_26_pass_without_comparison_fails_strict_gate_and_manifest` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1183`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_27_intentional_numeric_corruption_changes_gate_zero_to_one` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1193`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_28_direct_exact_comparison_rejects_nonzero_tolerance` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1212`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_29_invalid_comparator_is_rejected_before_shape_comparison` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1219`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_30_extended_precision_scalar_mismatch_survives_close_and_gate` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1240`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_31_extended_precision_array_mismatch_survives_close` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1268`
- `test_phase_13_77_realdata_invariance_harness.py::test_a2_32_close_refuses_nonfloating_numeric_families_without_coercion` — `passed` — `invariance` — `tests/test_phase_13_77_realdata_invariance_harness.py:L1288`

</details>

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

### DRAW.slot_grid
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[color-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[expr-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-expression-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-expression-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[facet_by-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[group_by-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-struct-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-struct-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[selection_vector-subframe-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-struct-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-struct-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid.py::test_slot_grid_smoke[weights_vector-subframe-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[selection_vector-facet_by]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_b4_vector_composition_seam[weights_vector-facet_by]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[color-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[expr-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-expression-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-expression-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[facet_by-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[group_by-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-struct-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-struct-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[selection_vector-subframe-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-struct-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-struct-lazy]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-eager]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-eager_parent_lazy_child]` — `xfailed`
- 🧨 `test_phase_13_79_slot_grid_invariance.py::test_slot_grid_invariance[weights_vector-subframe-lazy]` — `xfailed`

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

1081 tests not mapped to any feature.

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
- ... +1051 more

---
*Generated from pytest JSON + feature_taxonomy.py using the shared Capability Matrix semantic model.*
*Environment: alma2 · Linux-aarch64 · Python 3.10.19 · stamped by run_tests.sh*
