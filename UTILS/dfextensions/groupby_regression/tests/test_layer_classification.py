"""
Test Layer Classification — Phase 13.7.GB

Complete mapping of every test function to its layer, following the
classification decision tree from v02 proposal §3.3.

Rules applied:
  1. pytest.raises as primary assertion → validation
  2. Compares TWO code paths (v4 vs v5, Numba vs NumPy, chunks=1 vs N) → invariance
  3. assert_allclose against KNOWN TRUE VALUE (MC, analytic) → integration
  4. Speedup ratios, RSS limits, timing gates → performance
  5. Shapes, columns, types, len()>0, runs without crash → smoke
  6. None of above → smoke (fail-closed)

Edge case rules (§3.4):
  E1: Zero assertions = smoke
  E2: Shape/length-only assertions = smoke
  E3: Performance-only ≠ Verified
  E4: Property tests with tolerance = integration
  E5: Structural parity (column names match) ≠ invariance
"""

# fmt: off
TEST_LAYERS = {
    # ==================================================================
    # test_groupby_regression_kernels.py (28 tests)
    # ==================================================================
    # -- StatusCode tests (structural checks)
    "test_groupby_regression_kernels.py::TestStatusCode::test_decode_ok": "smoke",
    "test_groupby_regression_kernels.py::TestStatusCode::test_decode_single_bits": "smoke",
    "test_groupby_regression_kernels.py::TestStatusCode::test_decode_combined_bits": "smoke",
    "test_groupby_regression_kernels.py::TestStatusCode::test_decode_vectorized": "smoke",
    # -- SingleFit: allclose against known coefficients [1.0, 2.0, 3.0] → integration
    "test_groupby_regression_kernels.py::TestSingleFitCorrectness::test_basic_fit": "integration",
    # -- SingleFit: finite check only → smoke
    "test_groupby_regression_kernels.py::TestSingleFitCorrectness::test_weighted_fit": "smoke",
    # -- SingleFit: status check → smoke
    "test_groupby_regression_kernels.py::TestSingleFitCorrectness::test_insufficient_data": "smoke",
    # -- NaN handling: status/detection checks → smoke
    "test_groupby_regression_kernels.py::TestSingleFitCorrectness::test_nan_detection": "smoke",
    "test_groupby_regression_kernels.py::TestSingleFitCorrectness::test_nan_filtering": "smoke",
    # -- Multi-target: finite + RMS < 1e-10 (no-noise implies property) → smoke (RMS check is shape/existence)
    "test_groupby_regression_kernels.py::TestMultiFitCorrectness::test_multi_target_fit": "smoke",
    # -- PARITY: single-fit vs multi-fit produce identical results → INVARIANCE (A≡B)
    "test_groupby_regression_kernels.py::TestKernelParity::test_single_vs_multi_parity": "invariance",
    # -- Dispatch: checks which kernel function is called → smoke
    "test_groupby_regression_kernels.py::TestKernelDispatch::test_single_target_uses_single_kernel": "smoke",
    "test_groupby_regression_kernels.py::TestKernelDispatch::test_multi_target_uses_multi_kernel": "smoke",
    "test_groupby_regression_kernels.py::TestKernelDispatch::test_filter_mode_uses_single_kernel": "smoke",
    # -- MC validation: allclose against TRUE coefficients → integration
    "test_groupby_regression_kernels.py::TestMCValidation::test_single_fit_mc_validation_no_noise": "integration",
    "test_groupby_regression_kernels.py::TestMCValidation::test_single_fit_mc_validation_with_noise": "integration",
    "test_groupby_regression_kernels.py::TestMCValidation::test_multi_fit_mc_validation_no_noise": "integration",
    "test_groupby_regression_kernels.py::TestMCValidation::test_mc_validation_varying_features": "integration",
    # -- PARITY: Numba vs NumPy exact parity → INVARIANCE (A≡B)
    "test_groupby_regression_kernels.py::TestNumbaNumpyParity::test_numba_numpy_exact_parity": "invariance",
    "test_groupby_regression_kernels.py::TestNumbaNumpyParity::test_numba_numpy_parity_with_noise": "invariance",
    # -- Large scale: correctness + Numba/NumPy parity at scale → INVARIANCE (contains A≡B parity check)
    "test_groupby_regression_kernels.py::TestLargeScale::test_large_scale_correctness_50k_groups": "invariance",
    # -- Large scale multi-target: allclose to true coefficients → integration
    "test_groupby_regression_kernels.py::TestLargeScale::test_large_scale_multi_target": "integration",
    # -- Streaming RSS stability → performance
    "test_groupby_regression_kernels.py::TestStreaming::test_streaming_chunk_loop_rss_stability": "performance",
    "test_groupby_regression_kernels.py::TestStreaming::test_streaming_multifit_rss_stability": "performance",
    # -- Speedup ratios → performance
    "test_groupby_regression_kernels.py::TestPerformance::test_numba_vs_numpy_ratio": "performance",
    "test_groupby_regression_kernels.py::TestPerformance::test_multifit_speedup": "performance",
    # -- JIT signature checks → smoke
    "test_groupby_regression_kernels.py::TestJITSignatures::test_single_fit_jit_signatures": "smoke",
    "test_groupby_regression_kernels.py::TestJITSignatures::test_multi_fit_jit_signatures": "smoke",

    # ==================================================================
    # test_phase_12_8_gb.py (33 tests) — V5 batch API
    # ==================================================================
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_single_fit_returns_dataframe": "smoke",
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_multiple_fits_same_target": "smoke",
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_different_targets": "smoke",
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_different_weights_per_fit": "smoke",
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_different_linear_columns_per_fit": "smoke",
    "test_phase_12_8_gb.py::TestV5BasicFunctionality::test_no_intercept": "smoke",
    # -- V5 single fit matches V4 → INVARIANCE (A≡B cross-version)
    "test_phase_12_8_gb.py::TestV5V4Parity::test_single_fit_matches_v4": "invariance",
    # -- V5 multiple fits match V4 merged → INVARIANCE (A≡B)
    "test_phase_12_8_gb.py::TestV5V4Parity::test_multiple_fits_match_v4_merged": "invariance",
    # -- Precision check: finite only → smoke
    "test_phase_12_8_gb.py::TestV5V4Parity::test_precision_float32_input": "smoke",
    # -- chunks=1 vs chunks=4 identical → INVARIANCE (A≡B)
    "test_phase_12_8_gb.py::TestV5ChunkInvariance::test_chunks_1_vs_4_identical": "invariance",
    # -- chunks=1 vs chunks=8: assert_frame_equal → INVARIANCE (A≡B)
    "test_phase_12_8_gb.py::TestV5ChunkInvariance::test_chunks_1_vs_8_identical": "invariance",
    # -- Chunk boundary: structural check → smoke
    "test_phase_12_8_gb.py::TestV5ChunkInvariance::test_chunk_boundaries_never_split_groups": "smoke",
    # -- Validation errors
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_shared_suffix_duplicate_targets_raises": "validation",
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_duplicate_output_columns_raises": "validation",
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_mismatched_lengths_raises": "validation",
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_empty_fit_columns_raises": "validation",
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_missing_columns_raises": "validation",
    "test_phase_12_8_gb.py::TestV5ErrorHandling::test_null_in_gb_columns_raises": "validation",
    # -- Edge cases: structural checks → smoke
    "test_phase_12_8_gb.py::TestV5EdgeCases::test_empty_dataframe": "smoke",
    # -- Single group: allclose to known slope=2.0 → integration
    "test_phase_12_8_gb.py::TestV5EdgeCases::test_single_group": "integration",
    "test_phase_12_8_gb.py::TestV5EdgeCases::test_insufficient_data_group": "smoke",
    "test_phase_12_8_gb.py::TestV5EdgeCases::test_selection_parameter": "smoke",
    # -- Diagnostics and metadata → smoke (structural)
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_compute_mad_false": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_return_metadata": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_metadata_structure": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_metadata_formulas": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_diag_true_adds_columns": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_per_fit_diagnostics": "smoke",
    "test_phase_12_8_gb.py::TestV5Diagnostics::test_diag_n_total_correct": "smoke",
    # -- Internal helpers → smoke
    "test_phase_12_8_gb.py::TestV5InternalHelpers::test_sort_indices_single_key": "smoke",
    "test_phase_12_8_gb.py::TestV5InternalHelpers::test_sort_indices_multi_key": "smoke",
    "test_phase_12_8_gb.py::TestV5InternalHelpers::test_group_boundaries_correct": "smoke",
    # -- Performance → performance
    "test_phase_12_8_gb.py::TestV5Performance::test_v5_faster_than_v4_multiple": "performance",

    # ==================================================================
    # test_phase_12_9_gb.py (20 tests) — V5 Numba parallel
    # ==================================================================
    # -- V5 Numba matches V4: allclose → INVARIANCE (A≡B cross-version)
    "test_phase_12_9_gb.py::TestV5NumbaParallel::test_single_fit_matches": "invariance",
    "test_phase_12_9_gb.py::TestV5NumbaParallel::test_multi_fit_matches": "invariance",
    # -- Unweighted V5 matches V4: allclose → INVARIANCE
    "test_phase_12_9_gb.py::TestV5NumbaParallel::test_unweighted_matches": "invariance",
    # -- No-intercept V5 matches V4 → INVARIANCE
    "test_phase_12_9_gb.py::TestV5NumbaParallel::test_no_intercept_matches": "invariance",
    # -- Backend selection: structural → smoke
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_sequential": "smoke",
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_numba": "smoke",
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_numba_unavailable": "validation",
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_auto_numba": "smoke",
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_auto_sequential": "smoke",
    "test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_auto_no_numba": "smoke",
    # -- Status codes and diagnostics → smoke
    "test_phase_12_9_gb.py::TestV5Diagnostics::test_status_code_constants": "smoke",
    "test_phase_12_9_gb.py::TestV5Diagnostics::test_status_to_string_mapping": "smoke",
    "test_phase_12_9_gb.py::TestV5Diagnostics::test_status_codes_in_output": "smoke",
    "test_phase_12_9_gb.py::TestV5Diagnostics::test_per_fit_n_valid_different_nans": "smoke",
    "test_phase_12_9_gb.py::TestV5Diagnostics::test_shared_n_total_only": "smoke",
    # -- Parallel edge cases → smoke
    "test_phase_12_9_gb.py::TestV5Parallel::test_n_jobs_greater_than_groups": "smoke",
    "test_phase_12_9_gb.py::TestV5Parallel::test_single_group": "smoke",
    "test_phase_12_9_gb.py::TestV5Parallel::test_insufficient_data_groups": "smoke",
    "test_phase_12_9_gb.py::TestV5Parallel::test_scaling_diagnostic": "smoke",
    "test_phase_12_9_gb.py::TestV5Parallel::test_threading_env_restored": "smoke",

    # ==================================================================
    # test_groupby_regression_optimized.py (41 tests) — v2/v3/v4
    # ==================================================================
    "test_groupby_regression_optimized.py::test_basic_fit_serial": "smoke",
    "test_groupby_regression_optimized.py::test_basic_fit_parallel": "smoke",
    "test_groupby_regression_optimized.py::test_prediction_accuracy": "smoke",
    "test_groupby_regression_optimized.py::test_missing_values": "smoke",
    "test_groupby_regression_optimized.py::test_exact_coefficient_recovery": "smoke",  # no allclose, just checks finite/shape
    "test_groupby_regression_optimized.py::test_robust_outlier_resilience": "smoke",
    "test_groupby_regression_optimized.py::test_batch_strategy_auto": "smoke",
    "test_groupby_regression_optimized.py::test_batch_strategy_size_bucketing": "smoke",
    "test_groupby_regression_optimized.py::test_multiple_targets": "smoke",
    "test_groupby_regression_optimized.py::test_cast_dtype": "smoke",
    "test_groupby_regression_optimized.py::test_statistical_precision": "smoke",
    "test_groupby_regression_optimized.py::test_insufficient_data": "smoke",
    "test_groupby_regression_optimized.py::test_single_group": "smoke",
    "test_groupby_regression_optimized.py::test_empty_after_selection": "smoke",
    "test_groupby_regression_optimized.py::test_parallel_speedup": "smoke",  # timing ratio, soft gate
    # -- Threading: loky vs threading → INVARIANCE (allclose between backends)
    "test_groupby_regression_optimized.py::test_threading_backend_small_groups": "invariance",
    "test_groupby_regression_optimized.py::test_threading_backend_tiny_groups": "smoke",
    # -- Backend consistency: v4 fast vs v4 standard → INVARIANCE (diff < 1e-6)
    "test_groupby_regression_optimized.py::test_fast_backend_consistency": "invariance",
    # -- Numba backend consistency: numba vs standard → INVARIANCE (diff < 1e-6)
    "test_groupby_regression_optimized.py::test_numba_backend_consistency": "invariance",
    # -- Multi-column groupby v4 vs v2 → INVARIANCE (nanmax < tol)
    "test_groupby_regression_optimized.py::test_numba_multicol_groupby_v4_matches_v2": "invariance",
    "test_groupby_regression_optimized.py::test_numba_multicol_weighted_v4_matches_v2": "invariance",
    # -- Numba diagnostics → smoke
    "test_groupby_regression_optimized.py::test_numba_diagnostics_v4": "smoke",
    # -- V2 group rows bug check → smoke (structural)
    "test_groupby_regression_optimized.py::test_v2_group_rows_not_multiplied_by_targets": "smoke",
    # -- V2/V3/V4 identical groups 3col → smoke (E5: structural parity — checks column sets, group counts, no numerical comparison)
    "test_groupby_regression_optimized.py::test_v2_v3_v4_identical_groups_3col": "smoke",
    # -- V3 specific tests → smoke
    "test_groupby_regression_optimized.py::test_v3_fit_intercept_true": "smoke",
    "test_groupby_regression_optimized.py::test_v3_fit_intercept_false": "smoke",
    "test_groupby_regression_optimized.py::test_v3_parameter_errors": "smoke",
    "test_groupby_regression_optimized.py::test_v3_inf_nan_filtering": "smoke",
    "test_groupby_regression_optimized.py::test_v3_multiple_predictors": "smoke",
    "test_groupby_regression_optimized.py::test_v3_diagnostics_control": "smoke",
    "test_groupby_regression_optimized.py::test_v3_weighted_fits": "smoke",
    "test_groupby_regression_optimized.py::test_v3_singular_matrix_handling": "smoke",
    # -- V4 specific tests → smoke
    "test_groupby_regression_optimized.py::test_v4_fit_intercept_true": "smoke",
    "test_groupby_regression_optimized.py::test_v4_fit_intercept_false": "smoke",
    "test_groupby_regression_optimized.py::test_v4_parameter_errors": "smoke",
    "test_groupby_regression_optimized.py::test_v4_inf_nan_filtering": "smoke",
    "test_groupby_regression_optimized.py::test_v4_multiple_predictors": "smoke",
    "test_groupby_regression_optimized.py::test_v4_diagnostics_control": "smoke",
    "test_groupby_regression_optimized.py::test_v4_weighted_fits": "smoke",
    "test_groupby_regression_optimized.py::test_v4_singular_matrix_handling": "smoke",
    # -- V3/V4 parity: max_diff < 1e-8 → INVARIANCE (A≡B)
    "test_groupby_regression_optimized.py::test_v3_v4_parity": "invariance",

    # ==================================================================
    # test_groupby_regression.py (15 tests) — robust implementation
    # ==================================================================
    "test_groupby_regression.py::test_make_linear_fit_basic": "smoke",
    "test_groupby_regression.py::test_make_parallel_fit_robust": "smoke",
    "test_groupby_regression.py::test_insufficient_data": "smoke",
    "test_groupby_regression.py::test_prediction_accuracy": "smoke",
    "test_groupby_regression.py::test_missing_values": "smoke",
    "test_groupby_regression.py::test_cast_dtype_effect": "smoke",
    "test_groupby_regression.py::test_robust_outlier_resilience": "smoke",
    # -- Exact coefficient recovery: isclose to [2.0, 3.0] → integration
    "test_groupby_regression.py::test_exact_coefficient_recovery": "integration",
    "test_groupby_regression.py::test_exact_coefficient_recovery_parallel": "smoke",  # does NOT have allclose — need to verify
    "test_groupby_regression.py::test_min_stat_per_predictor": "smoke",
    "test_groupby_regression.py::test_sigma_cut_impact": "smoke",
    "test_groupby_regression.py::test_make_parallel_fit_robust_v2": "smoke",
    "test_groupby_regression.py::test_make_parallel_fit_with_linear_regression": "smoke",
    # -- Custom fitter: allclose(predicted, 42) → integration (property: constant fitter)
    "test_groupby_regression.py::test_make_parallel_fit_with_custom_fitter": "integration",
    "test_groupby_regression.py::test_diagnostics_columns_present": "smoke",

    # ==================================================================
    # test_cross_validation.py (3 tests)
    # ==================================================================
    # -- Robust vs v4 numerical parity: slope diff + intercept diff → INVARIANCE
    "test_cross_validation.py::test_robust_vs_v4_numerical_parity": "invariance",
    # -- Robust vs v2: structural agreement (column checks) → smoke (E5)
    "test_cross_validation.py::test_robust_vs_v2_structural_agreement": "smoke",
    # -- Robust vs v4 on common groups: slope diff < 1e-5 → INVARIANCE
    "test_cross_validation.py::test_robust_vs_v4_agreement_on_common_groups": "invariance",

    # ==================================================================
    # test_fit_metadata.py (65 tests)
    # All are structural/validation — no numerical checks
    # ==================================================================
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_simple_names": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_with_underscores": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_with_numbers": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_starting_with_underscore": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_mixed_case": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_valid_physics_names": "smoke",
    "test_fit_metadata.py::TestColumnNameValidation::test_invalid_starts_with_number": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_invalid_hyphen": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_invalid_space": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_invalid_special_chars": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_invalid_brackets": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_context_in_error_message": "validation",
    "test_fit_metadata.py::TestColumnNameValidation::test_empty_list_valid": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_single_predictor_with_intercept": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_single_predictor_without_intercept": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_two_predictors_with_intercept": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_two_predictors_without_intercept": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_many_predictors": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_preserves_predictor_order": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_empty_predictors_with_intercept": "smoke",
    "test_fit_metadata.py::TestFormulaGeneration::test_empty_predictors_without_intercept_raises": "validation",
    "test_fit_metadata.py::TestFormulaGeneration::test_different_suffixes": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_simple_residual": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_parentheses_around_prediction": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_complex_prediction": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_simple_pull": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_parentheses_around_residual": "smoke",
    "test_fit_metadata.py::TestDerivedColumnFormulas::test_mad_based_pull": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_schema_version_present": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_all_top_level_keys_present": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_columns_subsections_present": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_parameters_complete": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_single_target_formulas": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_multiple_targets": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_fit_intercept_true_includes_intercept": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_fit_intercept_false_excludes_intercept": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_columns_categorization": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_diagnostics_empty_when_diag_false": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_diagnostics_populated_when_diag_true": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_custom_diag_prefix": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_medians_empty_when_not_provided": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_medians_populated_when_provided": "smoke",
    "test_fit_metadata.py::TestMetadataSchema::test_invalid_fit_column_raises": "validation",
    "test_fit_metadata.py::TestMetadataSchema::test_invalid_linear_column_raises": "validation",
    "test_fit_metadata.py::TestMetadataIntegration::test_pull_both_rms_and_mad": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_formula_columns_match_coefficient_list": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_residual_references_original_target": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_pull_references_quality_column": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_empty_suffix": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_long_suffix": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_many_group_columns": "smoke",
    "test_fit_metadata.py::TestMetadataIntegration::test_single_character_names": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_prediction_formula_evaluates": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_residual_formula_evaluates": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_pull_rms_formula_evaluates": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_pull_mad_formula_evaluates": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_no_intercept_formula_evaluates": "smoke",
    "test_fit_metadata.py::TestFormulaEvaluation::test_multiple_targets_all_evaluate": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_column_selection_basic": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v3_column_selection_basic": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_includes_weights": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_includes_median_columns": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_deterministic_column_order": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_fit_results_unchanged": "smoke",
    "test_fit_metadata.py::TestV4ColumnSelection::test_v4_missing_median_column_raises": "validation",

    # ==================================================================
    # test_groupby_regression_sliding_window.py (28 tests)
    # ==================================================================
    "test_groupby_regression_sliding_window.py::test_sliding_window_basic_3d_verbose": "smoke",
    "test_groupby_regression_sliding_window.py::test_sliding_window_aggregation_verbose": "smoke",
    # -- Recover slope ≈ 2.0: abs(mean - 2.0) < 0.1 → integration (property with tolerance)
    "test_groupby_regression_sliding_window.py::test_sliding_window_linear_fit_recover_slope": "integration",
    "test_groupby_regression_sliding_window.py::test_empty_window_handling_no_crash": "smoke",
    "test_groupby_regression_sliding_window.py::test_min_entries_enforcement_flag_or_drop": "smoke",
    # -- Validation tests
    "test_groupby_regression_sliding_window.py::test_invalid_window_spec_rejected": "validation",
    "test_groupby_regression_sliding_window.py::test_missing_columns_raise_valueerror": "validation",
    "test_groupby_regression_sliding_window.py::test_float_bins_rejected_in_m71": "validation",
    "test_groupby_regression_sliding_window.py::test_min_entries_must_be_positive_int": "validation",
    "test_groupby_regression_sliding_window.py::test_invalid_fit_formula_raises": "validation",
    "test_groupby_regression_sliding_window.py::test_selection_mask_length_and_dtype": "validation",
    "test_groupby_regression_sliding_window.py::test_wls_requires_weights_column": "validation",
    # -- Zero-assert tests → smoke (E1)
    "test_groupby_regression_sliding_window.py::test_numpy_fallback_emits_performance_warning": "smoke",
    # -- Smoke tests
    "test_groupby_regression_sliding_window.py::test_single_bin_dataset_ok": "smoke",
    "test_groupby_regression_sliding_window.py::test_all_bins_below_threshold": "smoke",
    "test_groupby_regression_sliding_window.py::test_boundary_bins_truncation_counts": "smoke",
    "test_groupby_regression_sliding_window.py::test_multi_target_fit_output_schema": "smoke",
    "test_groupby_regression_sliding_window.py::test_weighted_vs_unweighted_coefficients_differ": "smoke",
    "test_groupby_regression_sliding_window.py::test_selection_mask_filters_pre_windowing": "smoke",
    "test_groupby_regression_sliding_window.py::test_metadata_presence_in_attrs": "smoke",
    "test_groupby_regression_sliding_window.py::test_backend_numba_request_warns_numpy_fallback": "smoke",
    "test_groupby_regression_sliding_window.py::test_statsmodels_fitters_basic": "smoke",
    "test_groupby_regression_sliding_window.py::test_statsmodels_formula_rich_syntax_relaxed": "smoke",
    "test_groupby_regression_sliding_window.py::test_statsmodels_not_available_doc_behavior": "validation",
    # -- Window size zero parity with v4 → INVARIANCE (A≡B)
    "test_groupby_regression_sliding_window.py::test_window_size_zero_parity_with_v4_relaxed": "invariance",
    # -- Internal helper contracts → smoke
    "test_groupby_regression_sliding_window.py::test__build_bin_index_map_contract": "smoke",
    "test_groupby_regression_sliding_window.py::test__generate_offsets_and_get_neighbors_truncate_contract": "smoke",
    "test_groupby_regression_sliding_window.py::test_realistic_smoke_normalised_residuals_gate": "smoke",

    # ==================================================================
    # test_groupby_regression_sliding_window_verbose.py (28 tests)
    # Identical assertions to non-verbose — same layer classification
    # Generator deduplicates: verbose copy cannot independently promote status
    # ==================================================================
    "test_groupby_regression_sliding_window_verbose.py::test_sliding_window_basic_3d_verbose": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_sliding_window_aggregation_verbose": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_sliding_window_linear_fit_recover_slope": "integration",
    "test_groupby_regression_sliding_window_verbose.py::test_empty_window_handling_no_crash": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_min_entries_enforcement_flag_or_drop": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_invalid_window_spec_rejected": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_missing_columns_raise_valueerror": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_float_bins_rejected_in_m71": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_min_entries_must_be_positive_int": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_invalid_fit_formula_raises": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_selection_mask_length_and_dtype": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_wls_requires_weights_column": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_numpy_fallback_emits_performance_warning": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_single_bin_dataset_ok": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_all_bins_below_threshold": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_boundary_bins_truncation_counts": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_multi_target_fit_output_schema": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_weighted_vs_unweighted_coefficients_differ": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_selection_mask_filters_pre_windowing": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_metadata_presence_in_attrs": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_backend_numba_request_warns_numpy_fallback": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_statsmodels_fitters_basic": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_statsmodels_formula_rich_syntax_relaxed": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_statsmodels_not_available_doc_behavior": "validation",
    "test_groupby_regression_sliding_window_verbose.py::test_window_size_zero_parity_with_v4_relaxed": "invariance",
    "test_groupby_regression_sliding_window_verbose.py::test__build_bin_index_map_contract": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test__generate_offsets_and_get_neighbors_truncate_contract": "smoke",
    "test_groupby_regression_sliding_window_verbose.py::test_realistic_smoke_normalised_residuals_gate": "smoke",

    # ==================================================================
    # test_invariance_sliding_window.py (21 tests)
    # All are invariance or integration by design (Phase 13.8.GB)
    # ==================================================================
    # -- Value recovery: nsigma-gated coefficient recovery → invariance
    "test_invariance_sliding_window.py::TestSWValueRecovery::test_sw_slope_nsigma_recovery": "invariance",
    "test_invariance_sliding_window.py::TestSWValueRecovery::test_sw_intercept_nsigma_recovery": "invariance",
    # -- Error estimator: consistency check → invariance
    "test_invariance_sliding_window.py::TestSWErrorEstimator::test_sw_error_estimator_consistency": "invariance",
    # -- Pull distribution: chi2 goodness-of-fit → invariance
    "test_invariance_sliding_window.py::TestSWPullDistribution::test_sw_pull_distribution": "invariance",
    # -- RMSE: noise recovery → invariance
    "test_invariance_sliding_window.py::TestSWRmse::test_sw_rmse_vs_known_noise": "invariance",
    # -- Structural invariants: window=0 parity, neighbor counts → invariance
    "test_invariance_sliding_window.py::TestSWStructuralInvariants::test_sw_window0_entries_equals_bin": "invariance",
    "test_invariance_sliding_window.py::TestSWStructuralInvariants::test_sw_window0_neighbors_equals_one": "invariance",
    "test_invariance_sliding_window.py::TestSWStructuralInvariants::test_sw_interior_entries_27x": "invariance",
    # -- Metamorphic: permutation invariance, determinism → invariance
    "test_invariance_sliding_window.py::TestSWMetamorphic::test_sw_permutation_invariance": "invariance",
    "test_invariance_sliding_window.py::TestSWMetamorphic::test_sw_determinism": "invariance",
    # -- Numba backend parity: V2 ≡ V1 → invariance
    "test_invariance_sliding_window.py::TestSWNumba::test_sw_numba_backend_used": "smoke",
    "test_invariance_sliding_window.py::TestSWNumba::test_sw_numba_equals_numpy": "invariance",
    "test_invariance_sliding_window.py::TestSWNumba::test_sw_numba_nsigma_recovery": "invariance",
    "test_invariance_sliding_window.py::TestSWNumba::test_sw_numba_multi_predictor": "invariance",
    # -- Multi-predictor: nsigma recovery with x + x² → invariance
    "test_invariance_sliding_window.py::TestSWMultiPredictor::test_sw_multi_predictor_nsigma_recovery": "invariance",
    # -- Oracle parity: window=0 ≡ per-bin OLS → invariance
    "test_invariance_sliding_window.py::TestSWOracleParity::test_sw_window0_equals_per_bin_ols": "invariance",

    # -- V3 incremental algorithm parity: V3 ≡ V1 → invariance
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_slope_matches_v1": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_multi_predictor_matches_v1": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_errors_match_v1": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_diagnostics_match_v1": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_stats_from_sufficient": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_metadata_algorithm": "smoke",
    "test_invariance_sliding_window.py::TestSWV3Parity::test_v3_nsigma_recovery": "invariance",

    # -- V3b boundary handling and bin weights → invariance/smoke
    "test_invariance_sliding_window.py::TestSWV3bBackwardCompat::test_v3b_defaults_equal_v3": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBackwardCompat::test_v3b_defaults_equal_v3_stats": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_reduces_corner_window": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_interior_equals_full": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_per_dimension": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_periodic_wraps_at_edges": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_periodic_too_few_bins_raises": "smoke",
    "test_invariance_sliding_window.py::TestSWV3bBoundary::test_invalid_boundary_raises": "smoke",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_weight_scale_invariance": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_err_nan_for_nonuniform_kernel": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_err_valid_for_uniform_kernel": "smoke",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_gaussian_recovers_slope": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_epanechnikov_kernel_zeros_distant_bins": "invariance",
    "test_invariance_sliding_window.py::TestSWV3bKernel::test_invalid_kernel_raises": "smoke",
    "test_invariance_sliding_window.py::TestSWV3bInteraction::test_symmetric_gaussian_interior_same_as_full_gaussian": "invariance",

    # -- V3b timing benchmarks (relative, not absolute)
    "test_invariance_sliding_window.py::TestSWV3bTiming::test_v3_numpy_faster_than_v1_numpy": "performance",
    "test_invariance_sliding_window.py::TestSWV3bTiming::test_v1_numpy_slower_than_v2_numba": "performance",
    "test_invariance_sliding_window.py::TestSWV3bTiming::test_v3_numba_faster_than_v2_numba": "performance",

    # -- V3-Numba parity: incremental_numba ≡ incremental_numpy
    "test_invariance_sliding_window.py::TestSWV3Numba::test_v3_numba_coeffs_match_numpy": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Numba::test_v3_numba_errors_match_numpy": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Numba::test_v3_numba_diagnostics_match_numpy": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Numba::test_v3_numba_gaussian_err_nan": "invariance",
    "test_invariance_sliding_window.py::TestSWV3Numba::test_v3_numba_multi_predictor": "invariance",

    # ==================================================================
    # test_invariance_kernels.py (4 tests)
    # All are invariance by design (Phase 13.8.GB)
    # ==================================================================
    "test_invariance_kernels.py::TestKernelSingleFitTruth::test_kernel_single_nsigma_recovery": "invariance",
    "test_invariance_kernels.py::TestKernelMultiFitTruth::test_kernel_multi_nsigma_recovery": "invariance",
    "test_invariance_kernels.py::TestKernelNumbaNumpyParity::test_kernel_numba_equals_numpy_lstsq": "invariance",
    "test_invariance_kernels.py::TestKernelSingleMultiParity::test_kernel_single_equals_multi_target0": "invariance",

    # ==================================================================
    # test_groupby_regression_standardization.py (5 tests)
    # All structural/column checks → smoke
    # ==================================================================
    "test_groupby_regression_standardization.py::test_v2_v3_v4_columns_without_diag": "smoke",
    "test_groupby_regression_standardization.py::test_v2_v3_v4_columns_with_diag": "smoke",
    "test_groupby_regression_standardization.py::test_suffix_applied_to_all_output_columns": "smoke",
    "test_groupby_regression_standardization.py::test_rms_mad_always_present": "smoke",
    "test_groupby_regression_standardization.py::test_merge_multiple_fits_with_suffix": "smoke",

    # ==================================================================
    # test_pyarrow_backend.py (24 tests)
    # ==================================================================
    "test_pyarrow_backend.py::TestPyArrowBackend::test_backend_pandas_explicit": "smoke",
    "test_pyarrow_backend.py::TestPyArrowBackend::test_backend_pyarrow_explicit": "smoke",
    "test_pyarrow_backend.py::TestPyArrowBackend::test_backend_auto_below_threshold": "smoke",
    "test_pyarrow_backend.py::TestPyArrowBackend::test_backend_auto_above_threshold": "smoke",
    "test_pyarrow_backend.py::TestPyArrowBackend::test_backend_invalid_raises": "validation",
    # -- Results match: assert_frame_equal pandas vs pyarrow → INVARIANCE (A≡B)
    "test_pyarrow_backend.py::TestPyArrowParity::test_results_match_simple": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_results_match_multicolumn_groupby": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_results_match_with_weights": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_results_match_multiple_targets": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_results_match_no_intercept": "invariance",
    # -- Sort/order parity → INVARIANCE (assert_frame_equal/assert_series_equal)
    "test_pyarrow_backend.py::TestPyArrowParity::test_sort_stability_matches_pandas": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_deterministic_column_order": "smoke",  # column order only
    "test_pyarrow_backend.py::TestPyArrowParity::test_deterministic_results_across_runs": "invariance",
    "test_pyarrow_backend.py::TestPyArrowParity::test_row_order_within_groups_matches_pandas": "invariance",
    # -- Arrow pool/memory → smoke/performance
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_arrow_pool_bytes_check": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_pyarrow_backend_functional": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_memory_reduction_large_dataset": "smoke",  # zero asserts (E1)
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_empty_dataframe": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_single_group": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_single_row": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_with_nan_values": "smoke",
    "test_pyarrow_backend.py::TestPyArrowEdgeCases::test_with_selection": "smoke",
    # -- Metadata: structural → smoke
    "test_pyarrow_backend.py::TestPyArrowMetadata::test_metadata_export_pyarrow": "smoke",
    "test_pyarrow_backend.py::TestPyArrowMetadata::test_metadata_matches_pandas": "smoke",  # formula string comparison, not numerical

    # ==================================================================
    # test_tpc_distortion_recovery.py (1 test)
    # Zero assertions (E1) → smoke
    # ==================================================================
    "test_tpc_distortion_recovery.py::test_tpc_distortion_recovery": "smoke",
}
# fmt: on


def get_layer_summary():
    """Return summary counts by layer."""
    from collections import Counter
    counts = Counter(TEST_LAYERS.values())
    return dict(sorted(counts.items()))


def get_layers_for_file(filename):
    """Return {test_name: layer} for all tests in a given file."""
    prefix = filename + "::"
    return {
        k.split("::")[-1]: v
        for k, v in TEST_LAYERS.items()
        if k.startswith(prefix)
    }


if __name__ == "__main__":
    summary = get_layer_summary()
    total = sum(summary.values())
    print(f"Total tests classified: {total}")
    print()
    for layer, count in summary.items():
        pct = 100.0 * count / total
        print(f"  {layer:15s}: {count:3d} ({pct:5.1f}%)")

    # Count unique (excluding verbose duplicates)
    non_verbose = {k: v for k, v in TEST_LAYERS.items()
                   if "verbose" not in k}
    from collections import Counter
    nv_counts = Counter(non_verbose.values())
    print(f"\nExcluding verbose duplicates ({total - len(non_verbose)} removed):")
    print(f"  Effective total: {len(non_verbose)}")
    inv_int = nv_counts.get("invariance", 0) + nv_counts.get("integration", 0)
    print(f"  Invariance + Integration: {inv_int} ({100*inv_int/len(non_verbose):.1f}%)")
