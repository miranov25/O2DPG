"""
Test Layer Classification — dfdraw

Determines whether each test is invariance (A ≡ B) or smoke (no crash).
Used by generate_capability_matrix.py to assign ✅ Verified vs ☑️ Smoke-only.

Classification rule:
  - invariance: test checks A == B (two paths produce same result)
  - smoke: test checks code runs without error, or structural properties

Everything not listed defaults to "smoke".

Phase 13.15.DF
"""

TEST_LAYERS = {
    # PHASE_13_76_DF (v1.1) — B=invariance (dispatcher vs oracle), A=integration, S=smoke
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_B1_dispatch_cell_counts_match_oracle_bins": "invariance",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_B2_dispatch_cell_counts_match_oracle_quantiles": "invariance",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_A1_float16_public_dispatch_bins": "integration",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_A2_float16_public_dispatch_quantiles": "integration",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S1_float16_upcast_preserves_binning": "smoke",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S2_qcut_duplicates_drop_preserved": "smoke",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S3_discrete_facet_unaffected": "smoke",
    "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S4_correlated_fixture_is_nonvacuous": "smoke",
    "test_profile_overflow.py::test_profile_mean_range_invariance": "invariance",
    "test_profile_overflow.py::test_profile_std_range_invariance": "invariance",
    "test_profile_overflow.py::test_profile_median_range_invariance": "invariance",
    "test_profile_overflow.py::test_profile_quantile_range_invariance": "invariance",
    "test_profile_overflow.py::test_profile_grouped_range_invariance": "invariance",
    "test_profile_overflow.py::test_profile_mean_no_edge_pileup": "invariance",

    # ── Invariance: batch ≡ standalone (numerical equality) ──
    "test_batch_groups.py::test_value_correctness": "invariance",

    # ── Invariance: same=True identity (ax1 is ax2) ──
    "test_same.py::TestSameBasic::test_same_reuses_axes_profile": "invariance",
    "test_same.py::TestSameBasic::test_same_reuses_axes_hist": "invariance",
    "test_same.py::TestSameBasic::test_same_three_overlays": "invariance",
    "test_same.py::TestSameOverride::test_ax_precedence": "invariance",
    "test_same.py::TestSameAcrossMethods::test_profile_on_hist2d": "invariance",
    "test_same.py::TestSameAcrossMethods::test_profile_on_hexbin": "invariance",
    "test_same.py::TestSameAcrossMethods::test_hist_overlay": "invariance",
    "test_same.py::TestSameAcrossMethods::test_draw_dispatch_same": "invariance",

    # ── Invariance: PyArrow ≡ pandas parity ──
    "test_pyarrow_input.py::TestResultsParity::test_hist_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_scatter_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_profile_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_hist2d_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_hexbin_parity": "invariance",

    # ── Invariance: vector path ≡ scalar same-loop (Phase 13.16.DF) ──
    # True A≡B comparisons: stats + line count + colors + xydata must match.
    "test_vector.py::TestVectorInvariance::test_vector_N1_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_1N_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_NN_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_hist_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_through_adf_equivalent_to_direct": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_chain_continuity_equivalent_to_full_scalar_loop": "invariance",
    "test_vector.py::TestVectorInvariance::test_vector_determinism": "invariance",

    # ── Invariance: Phase 13.16.DF FIX1 — vector kwarg propagation ──
    # Strong A≡B: vector(expr, kwarg=X) ≡ scalar_loop(expr, kwarg=X) for every
    # scalar-mode kwarg that was silently dropped in Phase 13.16.DF.
    "test_vector.py::TestVectorKwargPropagation::test_vector_group_by_bins_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_group_by_quantiles_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_min_entries_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_sort_groups_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_linestyle_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_weights_equivalent_to_scalar_loop": "invariance",
    "test_vector.py::TestVectorKwargPropagation::test_vector_return_data_equivalent_to_scalar_loop": "invariance",


    # ── Auto-classified §9-marked invariance tests (Phase 13.34.DF refresh) ──
    # Generated from test docstrings via §9.* marker grep.

    # ── §9 invariance tests in test_normalize.py ──
    "test_normalize.py::TestNormalizeBackwardCompat::test_NBC_1_default_normalize_none_unchanged": "invariance",
    "test_normalize.py::TestNormalizeCallable::test_NC_1_callable_with_errors": "invariance",
    "test_normalize.py::TestNormalizeCallable::test_NC_2_callable_values_only": "invariance",
    "test_normalize.py::TestNormalizeDelta::test_ND_1_basic_delta": "invariance",
    "test_normalize.py::TestNormalizeDelta::test_ND_2_delta_error_formula": "invariance",
    "test_normalize.py::TestNormalizeDelta::test_ND_3_stats_dict_structure": "invariance",
    "test_normalize.py::TestNormalizeFacetBy::test_NF_1_k_by_2_grid": "invariance",
    "test_normalize.py::TestNormalizeFacetBy::test_NF_2_per_facet_independent_computation": "invariance",
    "test_normalize.py::TestNormalizeFacetBy::test_NF_3_facet_by_bins_with_normalize_raises_with_workaround_hint": "invariance",
    "test_normalize.py::TestNormalizeGroupBy::test_NG_1_basic_group_by_3_fills": "invariance",
    "test_normalize.py::TestNormalizeGroupBy::test_NG_2_per_group_delta_recovers_offset": "invariance",
    "test_normalize.py::TestNormalizeGroupBy::test_NG_3_stats_dict_grouped_structure": "invariance",
    "test_normalize.py::TestNormalizeLayout::test_NLY_1_overlay_diff_creates_two_panels": "invariance",
    "test_normalize.py::TestNormalizeLayout::test_NLY_2_diff_only_single_panel": "invariance",
    "test_normalize.py::TestNormalizeLayout::test_NLY_3_height_ratio_style_respected": "invariance",
    "test_normalize.py::TestNormalizeLogRatio::test_NL_1_basic_log_ratio": "invariance",
    "test_normalize.py::TestNormalizeLogRatio::test_NL_2_non_positive_mean_masked": "invariance",
    "test_normalize.py::TestNormalizePull::test_NP_1_basic_pull_dimensionless": "invariance",
    "test_normalize.py::TestNormalizePull::test_NP_2_pull_bands_rendered": "invariance",
    "test_normalize.py::TestNormalizeRatio::test_NR_1_basic_ratio": "invariance",
    "test_normalize.py::TestNormalizeRatio::test_NR_2_zero_denominator_masked": "invariance",
    "test_normalize.py::TestNormalizeRatio::test_NR_3_ratio_error_formula": "invariance",
    "test_normalize.py::TestNormalizeSignConvention::test_NSC_1_signal_above_reference_delta_positive": "invariance",
    "test_normalize.py::TestNormalizeSingleYConvention::test_NSY_1_single_y_default_inner_works": "invariance",
    "test_normalize.py::TestNormalizeSingleYConvention::test_NSY_2_single_y_explicit_outer_equivalent": "invariance",
    "test_normalize.py::TestNormalizeValidation::test_NV_1_wrong_vector_count_raises": "invariance",
    "test_normalize.py::TestNormalizeValidation::test_NV_2_invalid_mode_string_raises": "invariance",
    "test_normalize.py::TestNormalizeValidation::test_NV_3_same_true_with_normalize_raises": "invariance",

    # ── §9 invariance tests in test_phase_13_27_commit2_selection_weights.py ──
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_1_2axis_inner_matched_lengths": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_2_2axis_inner_mismatched_raises": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_3_2axis_outer_creates_mxn": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_4_3axis_inner_all_equal": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_5_3axis_inner_mismatched_raises": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_6_3axis_outer_creates_mxnxp": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_7_1element_degrades_to_scalar": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_8_empty_list_raises": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_1_facet_by_quartile_with_selection_vector": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_2_facet_by_quartile_with_weights_vector": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_3_facet_inner_compose_no_crash": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_4_facet_by_bins_with_selection_vector": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_5_column_mode_facet_by_with_selection_vector_AD78": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_6_facet_by_quantiles_with_selection_vector_AD79": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_1_profile_vector_path_idempotent": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_2_profile_scalar_path_no_extra_assign": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_3_hist_vector_path_idempotent": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_4_scatter_vector_path_idempotent": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_5_facet_dispatch_idempotent": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_1_per_curve_sanitize_stats_in_stats_list": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_2_nan_policy_filter_default_no_crash": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_3_nan_policy_warn_emits_warning": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_4_no_nan_data_no_sanitize_warning": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_1_hist_weights_column_renders_weighted": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_2_hist_weights_expression_renders_weighted": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_3_hist_weights_with_norm_probability": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_4_hist_weights_with_group_by_raises": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_5_hist_no_weights_backward_compat": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_6_single_y_selection_vector_outer": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_7_single_y_selection_vector_inner_raises_actionable": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_8_no_fix1_userwarning_fires": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_9_inner_raise_message_names_lengths": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_WDH_4_single_x_weights_vector_outer": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_1_no_vec_iteration_indices_backward_compat": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_2_profile_no_vector_unchanged": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_3_hist_no_vector_unchanged": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_4_scatter_no_vector_unchanged": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_1_hist_accepts_selection_vector": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_2_hist_selection_vector_bins_shared": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_3_hist2d_with_selection_vector_typeerror": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_1_1ch_selection_alone_gets_color": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_2_2ch_selection_vector_inner": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_3_selection_vector_with_global_selection": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_4_selection_labels_override_accepted": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_5_selection_truncate_style_key_registered": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_1_scatter_accepts_selection_vector": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_2_scatter_selection_with_facet_by": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_1_selection_weights_explicit_rule": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_2_5channel_refuses_without_facet": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_3_3channel_with_selection_delta_resolves": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_4_explicit_rules_count": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_5_combination_with_quantiles": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_1_hist_weights_vector_runs": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_2_hist_weights_vector_with_global": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_3_hist2d_with_weights_vector_typeerror": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_1_1ch_weights_alone_gets_color": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_2_2ch_weights_vector_inner": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_3_weights_vector_with_global_weights": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_4_weights_labels_kwarg_plumbed": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_5_weights_categorical_kwarg_accepted": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_1_scatter_weights_vector_warns_once": "invariance",
    "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_2_scatter_weights_vector_silently_dropped": "invariance",

    # ── §9 invariance tests in test_phase_13_27_facet_refactor.py ──
    "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_groupby_dispatches_correctly": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_invalid_channel_name_raises": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_quantiles_one_subplot_per_quantile": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_vector_creates_n_subplots": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_capacity_fires": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_warn_mode": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_existing_test_facetstar_unchanged": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_facet_true_eqivalent_to_facet_by_groupby": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_old_facet_profile_kwargs_preserved": "invariance",
    "test_phase_13_27_facet_refactor.py::TestFacetSameTrueExclusion::test_facet_by_and_same_true_raises": "invariance",

    # ── §9 invariance tests in test_phase_13_28_df_fix1_autorange_style_keys.py ──
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_combined_autorange_overrides": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_k_robust_override": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_strategy_override": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_outlier_in_default_style": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_robust_in_default_style": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_percentile_in_default_style": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_strategy_in_default_style": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_profile_marker_still_present": "invariance",
    "test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_quantile_band_alpha_still_present": "invariance",

    # ── §9 invariance tests in test_phase_13_30_column_reference_validation.py ──
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist2d_no_groupby_param_unchanged": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist_groupby_existing_unchanged": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist_groupby_missing_raises": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_existing_column_unchanged": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_missing_column_raises_clear_error": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_none_unchanged": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_weights_expression_still_works": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Scatter::test_scatter_groupby_existing_unchanged": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Scatter::test_scatter_groupby_missing_raises": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestProductionReproducer::test_production_reproducer_now_raises_not_silent": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestR6ColumnReferenceValidator::test_r6_catches_invented_param_in_tuple": "invariance",
    "test_phase_13_30_column_reference_validation.py::TestR6ColumnReferenceValidator::test_r6_tuples_are_subset_of_forwarded": "invariance",

    # ── §9 invariance tests in test_phase_13_31_facet_by_column.py ──
    "test_phase_13_31_facet_by_column.py::TestAmbiguityError::test_error_message_lists_available_columns": "invariance",
    "test_phase_13_31_facet_by_column.py::TestAmbiguityError::test_typo_facet_by_raises_with_both_alternatives": "invariance",
    "test_phase_13_31_facet_by_column.py::TestCardinalityCap::test_column_facet_too_many_unique_values_triggers_cap": "invariance",
    "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_float_column_facet_works": "invariance",
    "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_int_column_facet_produces_n_subplots": "invariance",
    "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_subplot_titles_show_facet_value": "invariance",
    "test_phase_13_31_facet_by_column.py::TestColumnNameMode_String::test_string_column_facet_works_without_quoting_issue": "invariance",
    "test_phase_13_31_facet_by_column.py::TestOrthogonalComposition::test_facet_by_column_AND_group_by_compose": "invariance",
    "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_group_by_does_not_raise_p1330": "invariance",
    "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_quantiles_does_not_raise_p1330": "invariance",
    "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_vector_does_not_raise_p1330": "invariance",
    "test_phase_13_31_facet_by_column.py::TestRegressionCommit1::test_channel_group_by_mode_suppresses_inner_group_by": "invariance",

    # ── §9 invariance tests in test_phase_13_32_df_fix1.py ──
    "test_phase_13_32_df_fix1.py::TestBUG001FacetBinNameLeak::test_BUG001_facet_bin_col_name_not_in_subplot_titles": "invariance",
    "test_phase_13_32_df_fix1.py::TestBUG002AutoTitleInFacetedMode::test_BUG002_auto_title_channel_mode_facet_no_NameError": "invariance",
    "test_phase_13_32_df_fix1.py::TestBUG002AutoTitleInFacetedMode::test_BUG002_auto_title_sets_suptitle_in_faceted_mode": "invariance",
    "test_phase_13_32_df_fix1.py::TestBUG003FacetBinNumericSort::test_BUG003_facet_bins_sorted_numerically_not_lexicographically": "invariance",

    # ── §9 invariance tests in test_phase_13_32_groupby_quantiles_facet.py ──
    "test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_126_overlay_with_quantiles": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_129_facet_with_groupby_bins": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_cap_still_fires_without_binning": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_bins_honors_n": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_quantiles_honors_n": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_discrete_grouped_uses_group_color_dashed": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_nested_band_with_groupby_raises": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_quantile_band_per_group_color": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_quantile_discrete_per_group": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_stats_dict_quantile_keys_present": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_weights_compose_with_groupby_quantiles": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_hist": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_hist2d": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_scatter": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_on_channel_facet_raises": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_on_column_facet": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_quantiles_mutual_exclusion": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_without_facet_by_raises": "invariance",
    "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_quantiles_on_column_facet": "invariance",

    # ── Phase 13.34.DF M2 — Robustness gap §9 invariance tests ──
    "test_phase_13_34_df_m2_robustness.py::TestMedianMADSigma::test_MED_1_median_uses_mad_sigma_for_errors": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_profile_keys_present": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_normalize_keys_single_curve": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_normalize_keys_grouped": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_auto_title": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_subplot_titles": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_group_by_composition": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_normalize_with_facet_by_column_mode": "invariance",
    "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_channel_mode_facet_with_auto_title": "invariance",

    # ── Phase 13.37.DF FIX1 — Phase 13.36 backward-compat invariance locks ──
    # These promote 3 features from ☑️ Smoke to ✅ Verified by adding A≡B
    # behavioral assertions on top of the pre-existing smoke tests:
    # - SO.COMPAT.1: A≡B determinism across two identical render calls
    #   (locks _user_*=None sentinel transparency)
    # - SO.COMPAT.2: A≡B preservation of first call's colors across same=True
    #   second call + invariance on no-auto-color collapse
    # - SO.COMPAT.3: Invariance on vector+group_by rendering matrix shape
    #   (N_groups × N_vector lines; N_groups colors; N_vector linestyles)
    "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_1_group_by_bins_default_cycle_byte_identical": "invariance",
    "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_2_same_true_group_by_no_auto_color_injection": "invariance",
    "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_3_vector_group_by_palette_and_channels_preserved": "invariance",

    # ── Phase 13.38.DF — Scatter enhancements: BUG-017 + xerr/yerr + expression ──
    # All 19 new tests are A≡B invariance locks (not smoke). Coverage:
    # - BUG-017 facet_by float guard (positive + negative)
    # - xerr/yerr extents ≡ column/eval values to 1e-9
    # - SE.5 dispatch invariance (PathCollection vs ErrorbarContainer)
    # - SE.6 three-tier NaN policy (silent/warn/raise)
    # - ECM.1/6 colormap array ≡ expression/column values
    # - ECM.6 CP0-1 BACKWARD-COMPAT LOCK (column 'b' wins over named-color)
    # - ECM.4 concrete marker-path comparison + point count
    # - ECM.8 no-spurious-legend invariant for per-point marker rendering
    "test_phase_13_38_df_scatter_enhancements.py::TestFacetByFloatGuard::test_FBGUARD_1_float_facet_by_no_bins_raises": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestFacetByFloatGuard::test_FBGUARD_2_float_facet_by_with_bins_no_error": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_1_yerr_column_extents_match": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_2_xerr_column_extents_match": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_3_both_xerr_yerr_simultaneously": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_4_yerr_dfeval_expression": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_5_default_dispatch_invariance": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_6_nan_policy_three_tiers": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_7_style_keys_registered": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_8_group_by_plus_yerr": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_9_xerr_plus_facet_by": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_1_expression_color_colormap_applied": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_2_column_color_byte_identical_backward_compat": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_3_invalid_expression_actionable_error": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_6_column_name_collision_with_named_color": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_7_expression_color_plus_group_by_behavior_locked": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_4_boolean_marker_two_marker_encoding": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_5_expression_color_plus_expression_marker_compose": "invariance",
    "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_8_per_point_marker_legend_no_duplicates": "invariance",

    # ── Phase 13.39.DF: 2D Profile + Time Axis + Scatter3D (24 invariance) ──
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_1_quadmesh_rendered": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_2_per_cell_mean_correctness": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_3_min_entries_masks_low_count_cells": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_4_dfeval_expression_for_z": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_5_bins_list_vs_bins2_shape_invariance": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_6_colorbar_labeled_single_key": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_7_selection_applied_before_binning": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_8_backward_compat_1d_profile": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_10_group_by_raises_value_error": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_1_profile_time_format_pct_HM": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_2_profile_time_format_auto": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_3_default_no_time_format_backward_compat": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_4_scatter_time_format": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_5_hist_time_format_realistic_timestamps": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_6_profile2d_x_axis_date_formatter": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_7_datetime64_column_no_crash": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_1_basic_render": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_2_color_expression": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_3_size_column": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_4_selection_reduces_count": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_5_two_variable_expr_raises": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_6_stats_dict_locks_all_three_means": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_7_group_by_raises_value_error": "invariance",
    "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_8_same_true_non_3d_axes_raises": "invariance",

    # ── Phase 13.40.DF: Cumulative Histogram (10 invariance) ──
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_1_monotone_and_total_N_lock": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_2_ECDF_last_value_is_one": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_3_survival_starts_at_one": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_4_cumulative_false_backward_compat": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_5_group_by_per_group_ecdf": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_6_hist_errors_plus_cumulative_raises": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_7_vector_dispatch_propagates_cumulative": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_8_facet_by_per_facet_ecdf": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_9_histtype_step_plus_cumulative_polygon_safe": "invariance",
    "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_10_group_by_stacked_cumulative_regression_lock": "invariance",

    # ── Phase 13.41.DF: N-D Faceting via facet_by=List[str] (19 invariance) ──
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_1_string_equals_list_of_one": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_2_2d_grid_rows_by_cols": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_3_2d_bins_per_dim": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_4_2d_quantiles_per_dim": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_5_mixed_bins_None_and_int": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_6_convention_lock_2d_row_col": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_7_length_mismatch_raises": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_8_group_by_inside_cells": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_9_cumulative_per_cell": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_10_four_dimensions_raises": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_11_convention_lock_3d_list_of_figures": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_12_share_x_row_regression_lock": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_13_share_across_figures_3d_scatter": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_14_share_none_data_divergence": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_15_empty_cell_no_crash": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_16_3d_share_across_hist_dispatch": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_17_3d_share_across_profile_dispatch": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_18_share_across_figures_false_independence": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_19_share_x_col_symmetry": "invariance",
    # Phase 13.41.DF FIX1 (3 additional regression locks)
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_20_share_y_row_symmetry": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_21_share_x_invalid_raises": "invariance",
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_22_auto_title_suptitle_lock": "invariance",
    # Phase 13.41.DF FIX2 (1 additional regression lock)
    "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_23_3d_auto_title_combined_suptitle": "invariance",

    # ── Phase 13.42.DF: Inline fits — invariance battery (F.1–F.27) ──
    # All 27 F.* tests assert canonical contracts (§3.5 list-of-lists,
    # numerical recovery within tolerance, propagation through dispatch,
    # silent-consume policy, etc.) and are invariance tests by construction.
    # NB v2-fix1: 6 names corrected after Sonnet54 caught taxonomy mismatch
    # (proposed names from v1.4 spec did not match implementation).
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f1_gauss_recovers_center_sigma": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f2_pol2_recovers_coefficients": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f3_user_callable_with_initial": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f4_pol1_alias_equals_linear": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f5_gauss_registry_heuristic_converges": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f23_user_guess_callable": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f6_str_shorthand_equiv_to_dict": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f7_callable_shorthand_equiv_to_dict": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f8_length1_list_broadcasts": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f9_length_match_pair_per_channel": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f10_length_mismatch_raises": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f11_vector_expr_scalar_fit_broadcast": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f12_vector_expr_vector_fit_pair": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f13_compound_on_single_curve": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f14_group_by_list_of_lists_per_group": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f15_facet_by_2d_tuple_keys": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f16_range_restricts_fit_domain": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f17_bounds_constrain_params": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f18_unknown_dict_key_raises": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsDisplayRendering::test_f19_show_params_false_hides_one_block": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsDisplayRendering::test_f20_multi_fit_distinct_linestyles": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsFailureHandling::test_f21_default_failure_does_not_raise": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsFailureHandling::test_f22_raise_on_failure_true_raises": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f24_register_fit_public_export": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f25_groupby_facet_fit_deep_composition": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f26_normalize_plus_fit_silent_consume": "invariance",
    # Phase 13.42.DF P1-B regression lock (Sonnet54 finding, CRR §2 D7)
    "test_phase_13_42_df_inline_fits.py::TestFitsP1BProfileGroupedRegression::test_f27_profile_group_by_fit_returns_dict_keyed_by_group": "invariance",
    # Phase 13.42.DF FIX1 production-gate regression locks (B1-B7, D5/D8/D9)
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f28_grouped_fit_quantile_binning": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f28b_skipped_empty_does_not_crash": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f29_hist_fit_redchi_physically_correct": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f30_set_style_fit_textbox_fontsize_facet": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f31_vector_fit_pairing": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f32_stacked_hist_per_group_fits": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f33_fit_textbox_kwargs_fontsize_override": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f33_fit_textbox_kwargs_precedence_over_style": "invariance",
    # Phase 13.42.DF FIX2 regression locks (close items deferred at FIX1 close)
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f59_suptitle_top_for_title_helper": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f60_facet_fit_no_crash": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f61_weighted_hist_fit_userwarning": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f61_unweighted_hist_fit_no_warning": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f62_stacked_selection_vector_fit_raises": "invariance",
    "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f63_fit_textbox_kwargs_signature_and_forwarded_names": "invariance",

    # Phase 13.43.DF v1.2 — Summary Fit (26 tests: F.34-F.48 + F.38a +
    # F.47b + F.49-F.56 + F.56b)
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f34_summary_fit_table": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f35_summary_fit_figure": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f36_summary_fit_both": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f37_summary_fit_omitted_is_scenario_a": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f38_table_row_count": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f38a_selection_vector_composition": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f39_params_figure_auto_layout": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f40_overlay_mode": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f41_annotate_mode": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f42_no_group_no_facet_figure_is_scenario_e": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f43_precision_in_table_cells": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f44_2d_facet_composition": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f45_normalize_plus_summary_fit_is_scenario_e": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f46_quantile_band_profile_is_scenario_e": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f47_cumulative_hist": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f47b_stacked_grouped_fit": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f48_summary_fit_without_fit": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f49_auto_title_content": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f50_title_overflow_truncate": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f51_default_data_format_is_list_dict": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f52_pandas_via_style_key": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f53_per_call_data_format": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f54_same_true_replace_mode": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f55_placement_invariants": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56_faceted_aggregation_shape3": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56b_faceted_no_group_by": "invariance",
    "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56c_draw_scalar_forwards_fit_and_summary_fit": "invariance",

    # Phase 13.46.DF — audit bucket ① fixes
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f64_gaus_root_alias": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f65_histo_type_alias": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f66_kwarg_typo_guard": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f67_scatter_range_minmax_nonfaceted": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f68_scatter_range_minmax_faceted": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f69a_scatter_range_strategy_parity": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f69b_profile_hist_range_minmax_no_unpack_error": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f70_get_suptitle_live_path": "invariance",
    "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f71_scatter_range_removes_out_of_range_points": "invariance",

    # ── Phase 13.49.DF — meta-tests for the capability matrix ──
    # Invariance: assert structural properties of the taxonomy/classification/
    # generator (every claimed test resolves, classification coverage, HTML
    # emitter integrity, no orphan visual tests). See PHASE_13_49_DF v1.2 §5.
    "test_meta_capability_matrix.py::test_taxonomy_tests_resolve": "invariance",
    "test_meta_capability_matrix.py::test_classification_coverage": "invariance",
    "test_meta_capability_matrix.py::test_html_emitter_parseable": "invariance",
    "test_meta_capability_matrix.py::test_no_orphan_visual_tests": "invariance",

    # ── Phase 13.48.DF — visual_primitive layer (renderer-free figure checks) ──
    # A distinct layer from invariance/smoke: these assert structural properties
    # of the rendered figure (points in bounds, per-cell artist counts, color
    # distinctness, grid shape, title de-duplication) against figure-derived /
    # filtered-df oracles. The matrix currently displays these as Smoke-only;
    # a dedicated "visually checked" status is the deferred §3 schema task.
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v1_scatter_points_in_bounds": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v2_every_visible_cell_nonempty": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v2_padding_safe_on_ragged_grid": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v3_artist_count_matches_groups": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v4_per_group_colors_distinct": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v5_facet_grid_shape": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v6_shared_axis_consistency": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v7_legend_entries_match_groups": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v8_color_cycle_distinct": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v9_auto_title_not_duplicated": "visual_primitive",
    "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v10_title_content": "visual_primitive",

    # Phase 13.50.DF — Fit-rendering visual primitives (step 1 of 7).
    # F1-F3 lock the _DISPLAY_NAMES map in plots/_fit_render.py:
    #   slope→p1, intercept→p0, center→$\mu$, sigma→$\sigma$, decay→$\tau$.
    # Steps 2-6 will add F4-F16 (visual_primitive layer) and step 4 will add
    # F17, F18, test_normalize_legend_spec_idempotent (invariance layer).
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F1_linear_short_names": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F2_gauss_mu_mathtext": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F3_exponential_tau_mathtext": "visual_primitive",
    # Phase 13.50.DF step 2 — precision keys (visual_primitive layer)
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitPrecision::test_F4_default_precision": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitPrecision::test_F5_precision_mode_physics": "visual_primitive",
    # Phase 13.50.DF step 3 — fit_textbox_kwargs extensions (visual_primitive layer)
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitTextboxKwargsExtensions::test_F6_rename_params_overrides_display_map": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350FitTextboxKwargsExtensions::test_F16_value_format_per_call_override": "visual_primitive",
    # Phase 13.50.DF step 4 — legend= polymorphic kwarg + show_legend= parallel.
    # F7/F8/F9/F10 inspect fig.legends + ax.get_legend() → renderer-free
    # primitive checks → visual_primitive. F18 compares two legend_topology
    # 4-tuples + the normalizer test is a pure function → both invariance.
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F7_shared_one_fig_zero_per_axes": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F8_first_only_axes_0_kept": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F9_false_no_legends": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F10_dict_forwards_loc_and_ncol": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F18_show_legend_legend_behavioral_equivalence": "invariance",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendNormalizerInvariance::test_normalize_legend_spec_idempotent": "invariance",
    # Phase 13.50.DF step 5 — summary_fit.placement axis.
    # F11/F12/F13 inspect placement_topology via fig.axes / fig.subfigures /
    # stats['summary_fit'] keys → renderer-free → visual_primitive.
    # F17 compares the keyed-dict topologies across all 3 placements;
    # invariance because the assertion is on dict equality of topology dicts
    # rendered from the SAME dispatch, no per-pixel rasterization.
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F11_placement_figure_default_unchanged": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F12_placement_subfigure_per_panel_inset_table": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F13_placement_pad_uses_extra_row_axes": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F17_placement_variants_produce_equivalent_tables": "invariance",
    # Phase 13.50.DF step 6 — summary_fit.orientation axis. F14/F15 inspect
    # the matplotlib Table object's cell shape via table.get_celld() →
    # renderer-free primitive checks → visual_primitive.
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitOrientation::test_F14_orientation_row_default_unchanged": "visual_primitive",
    "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitOrientation::test_F15_orientation_column_transposes_shape": "visual_primitive",

    # ── Phase 13.51.DF — Post-audit fix-pass tests (gate-4 closure) ──
    # 12 of the 23 tests in test_phase_13_51_post_audit.py are classified
    # here so the matching feature_taxonomy.py entries (registered in
    # commit 8f6f93f8, Phase 13.52 distribution prep) land in the right
    # capability layer. The remaining 11 (T1, T7, T10a, T11, T12, T13,
    # T15, T16/16b, T17/17b) are smoke / no-crash regressions and stay
    # at the default "smoke" classification.
    #
    # Without this block, the 5 Phase 13.51 feature claims (DRAW.R2_forwarding,
    # PROFILE.central_median_1d, PROFILE2D.central_median_mesh,
    # PROFILE.central_median_fit, AUTORANGE.datetime64_guard) were
    # classified Smoke-only, leaving CAPABILITY_MATRIX gate 4
    # (Verified >=64) open at 59. With this block, Verified moves to 64
    # and gate 4 closes. ADF reviewer pre-distribution finding.
    #
    # Invariance class (A == B / explicit-value assertions):
    # T3/T5 use "matches_direct" — vector path equals direct call.
    # T9a/T9b/T9c assert central='median' produces DIFFERENT artefacts
    # from the mean path (negative-equivalence check on data values).
    # T10b/T10c assert datetime64 input yields a finite (xmin, xmax)
    # tuple across two distinct autorange strategies.
    "test_phase_13_51_post_audit.py::test_T3_draw_type_profile_selection_vector_matches_direct": "invariance",
    "test_phase_13_51_post_audit.py::test_T5_draw_type_hist_selection_vector_matches_direct": "invariance",
    "test_phase_13_51_post_audit.py::test_T9a_profile2d_central_median_mesh_differs_from_mean": "invariance",
    "test_phase_13_51_post_audit.py::test_T9b_profile_1d_central_median_line_differs_from_mean": "invariance",
    "test_phase_13_51_post_audit.py::test_T9c_profile_fit_central_median_fit_center_differs": "invariance",
    "test_phase_13_51_post_audit.py::test_T10b_compute_autorange_datetime64_robust_3mad": "invariance",
    "test_phase_13_51_post_audit.py::test_T10c_compute_autorange_datetime64_percentile_99": "invariance",
    #
    # Visual primitive class (renderer-free figure-state checks):
    # T2/T4/T6/T8 inspect populated panels via ax.collections / ax.lines
    # / ax.patches per faceted cell (R-2 forwarding visual check).
    # T14 inspects hist2d's tick label text for date-formatted output.
    "test_phase_13_51_post_audit.py::test_T2_draw_type_profile_facet_by_sec_populated_panels": "visual_primitive",
    "test_phase_13_51_post_audit.py::test_T4_draw_type_hist_facet_by_sec_populated_panels": "visual_primitive",
    "test_phase_13_51_post_audit.py::test_T6_draw_type_scatter_facet_by_sec_populated_panels": "visual_primitive",
    "test_phase_13_51_post_audit.py::test_T8_draw_type_hist2d_facet_by_sec_populated_panels": "visual_primitive",
    "test_phase_13_51_post_audit.py::test_T14_hist2d_time_format_auto_date_tick_labels": "visual_primitive",

    # ── Phase 13.55.DF — draw_batch audit + Phase 13.51/13.52 type-gap fix ──
    # 7 of the 17 tests in test_phase_13_55_df_draw_batch_audit.py are
    # classified here; the remaining 10 (T1-T6, T11, T16, T17) are smoke
    # / no-crash regressions and stay at the default "smoke" classification.
    #
    # Invariance class (A == B / explicit-value assertions):
    # T7 — central='median' vs 'mean' under outlier fixture differ
    #      measurably (mean != median at multiple bins); Phase 13.51 T9b
    #      pattern (10% rows @ 100x).
    # T10 — fit='gauss' produces a non-empty stats['fit'] block;
    #       data-structure existence + non-emptiness.
    # T12 — invalid type raises ValueError (explicit exception-type check).
    # T13 — invalid type under on_error='skip' counted in _errors dict
    #       (explicit key + summary['failed']==1).
    # T14 — valid type under on_error='raise' returns success + stats
    #       (explicit success state).
    # T15 — same=True within group reuses axis (len(axes)==1,
    #       len(stats)==2 explicit counts).
    "test_phase_13_55_df_draw_batch_audit.py::test_T7_central_median_via_batch_with_outlier_fixture": "invariance",
    "test_phase_13_55_df_draw_batch_audit.py::test_T10_fit_gauss_profile_via_batch": "invariance",
    "test_phase_13_55_df_draw_batch_audit.py::test_T12_invalid_type_raises_under_new_default": "invariance",
    "test_phase_13_55_df_draw_batch_audit.py::test_T13_invalid_type_with_explicit_skip_is_counted_not_raised": "invariance",
    "test_phase_13_55_df_draw_batch_audit.py::test_T14_valid_type_under_new_default_raise_succeeds": "invariance",
    "test_phase_13_55_df_draw_batch_audit.py::test_T15_same_overlay_within_group_reuses_axis": "invariance",
    #
    # Visual primitive class (renderer-free figure-state checks):
    # T8 inspects hist2d x-axis tick label text for HH:MM date-formatted
    #    output via ax.get_xticklabels().
    # T9 inspects scatter title text via ax.get_title().
    "test_phase_13_55_df_draw_batch_audit.py::test_T8_time_format_via_batch": "visual_primitive",
    "test_phase_13_55_df_draw_batch_audit.py::test_T9_auto_title_scatter_via_batch": "visual_primitive",

    # Everything else defaults to "smoke"

    # ── Phase 13.57.DF — equivalence / effect-assertion tests (A≡B) ──
    "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_equivalence_typed_surface": "invariance",
    "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_equivalence_draw_surface": "invariance",
    "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_p0_equivalent_to_initial": "invariance",
    "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_old_key_equivalence_regression": "invariance",
    "test_phase_13_57_df_fixes.py::TestE2YCentral::test_y_central_present_and_is_median": "invariance",
    "test_phase_13_57_df_fixes.py::TestE2YCentral::test_y_central_equals_mean_when_central_default": "invariance",
    "test_phase_13_57_df_fixes.py::TestE2YCentral::test_grouped_export_has_y_central": "invariance",
    "test_phase_13_57_df_fixes.py::TestE3Matrix::test_e3a_draw_direct_honors_ax": "invariance",
    "test_phase_13_57_df_fixes.py::TestE3Matrix::test_e3b_typed_direct_honors_ax_regression_lock": "invariance",
    "test_phase_13_57_df_fixes.py::TestE3Matrix::test_k4_named_params_reach_profile2d": "invariance",
    "test_phase_13_57_df_fixes.py::TestE3Matrix::test_k4_selection_effect_assertion": "invariance",
    "test_phase_13_57_df_fixes.py::TestK5ZeroWarningGate::test_corpus_form_calls_emit_zero_warnings": "invariance",

    # ── PHASE_13_86_DF — explicit layer classification (98 claims) ──
    # All 97 collected tests in the six PHASE_13_86 files are classified
    # explicitly; C-a also changed one historical auto-title regression node.
    # invariance: 58
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A5_scatter_selection_vector_facet_is_exact_or_refuses": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A6_scatter_weights_vector_facet_refuses_until_semantics_exist": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A7_scatter_errorbar_plus_expression_color_is_exact_or_refuses": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A7s_single_panel_errorbar_plus_expression_color_is_exact_or_refuses": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A8_internal_suppress_color_cycle_never_warns_as_user_kwarg": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[hist2d]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[hist]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[profile2d]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[profile]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[scatter]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10e_draw_signature_has_no_unclassified_global_parameter": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[hist2d-fit-gauss]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[profile2d-color-magenta]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[profile2d-marker-s]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A11_internal_vector_control_is_registered_with_typo_guard": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[hist2d]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[hist]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[profile]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[scatter]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[hist-x]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[hist2d-y:x]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[profile-y:x]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[scatter-y:x]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9d_scatter_refused_multi_branch_vector_semantics_are_loud[selection_vector]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9d_scatter_refused_multi_branch_vector_semantics_are_loud[weights_vector]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9e_profile_refused_same_plus_facet_is_loud": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9f2_hist_refused_normalize_plus_facet_is_loud": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9f_hist2d_refused_fit_is_loud": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist-x-ax]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist-x-same]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist2d-y:x-ax]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist2d-y:x-same]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[profile-y:x-ax]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[scatter-y:x-ax]": "invariance",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[scatter-y:x-same]": "invariance",
    "test_phase_13_86_stage_ba.py::test_Ba1_draw_profile_facet_vector_matches_typed_owner": "invariance",
    "test_phase_13_86_stage_ba.py::test_Ba2_vector_facets_keep_expression_components_distinct": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB1_typed_profile_vector_ordinary_facet_matches_independent_oracle": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB2_draw_group_vector_ordinary_facet_matches_oracle_and_typed_owner": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB4_expression_vector_selection_vector_composes_within_each_facet[inner-2]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB4_expression_vector_selection_vector_composes_within_each_facet[outer-4]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB5_expression_vector_weights_vector_composes_within_each_facet[inner-2]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB5_expression_vector_weights_vector_composes_within_each_facet[outer-4]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[hist-[y0,y1]-draw]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[hist-[y0,y1]-typed]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[scatter-[y0,y1]:x-draw]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[scatter-[y0,y1]:x-typed]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB7_profile_expression_vector_nd_facet_positive_contract[draw]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB7_profile_expression_vector_nd_facet_positive_contract[typed]": "invariance",
    "test_phase_13_86_stage_bb.py::test_BB8_public_same_true_neighbor_remains_invalid": "invariance",
    "test_phase_13_86_stage_bc.py::test_BC1_computed_y_component_works_with_vector_facets[draw]": "invariance",
    "test_phase_13_86_stage_bc.py::test_BC1_computed_y_component_works_with_vector_facets[typed]": "invariance",
    "test_phase_13_86_stage_bc.py::test_BC2_computed_y_component_works_inside_ordinary_facets[draw]": "invariance",
    "test_phase_13_86_stage_bc.py::test_BC2_computed_y_component_works_inside_ordinary_facets[typed]": "invariance",
    "test_phase_13_86_stage_bd.py::test_BD2_explicit_ylabel_still_wins_over_metadata": "invariance",
    "test_phase_13_86_stage_bd.py::test_BD3_scalar_reference_behavior_unchanged": "invariance",
    "test_phase_13_86_stage_ca.py::test_CA2_build_auto_title_preserves_selection_until_render_width_is_known": "invariance",
    "test_auto_title.py::TestAutoTitleParts::test_long_selection_preserved_until_render": "invariance",
    # integration: 30
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[hist-x-required0]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[hist2d-y:x-required3]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[profile-y:x-required2]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[scatter-y:x-required1]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10b_draw_profile_normalize_vector_special_delegation_preserves_request": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[hist-x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[hist2d-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[profile-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[profile2d-z:y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[scatter-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist-x-draw]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist-x-typed]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist2d-y:x-draw]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist2d-y:x-typed]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[profile-y:x-draw]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[profile-y:x-typed]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[scatter-y:x-draw]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[scatter-y:x-typed]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[hist-x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[hist2d-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[profile-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[scatter-y:x]": "integration",
    "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10c_draw_profile_facet_vector_preserves_intact_expression_for_typed_owner": "integration",
    "test_phase_13_86_stage_bb.py::test_BB3_realistic_group_quantiles_and_facet_bins_compose_inside_panels": "integration",
    "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[draw-facet]": "integration",
    "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[draw-vector]": "integration",
    "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[typed-facet]": "integration",
    "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[typed-vector]": "integration",
    "test_phase_13_86_stage_bd.py::test_BD1_vector_facets_use_per_component_axis_title_metadata[draw]": "integration",
    "test_phase_13_86_stage_bd.py::test_BD1_vector_facets_use_per_component_axis_title_metadata[typed]": "integration",
    # visual_primitive: 10
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A1_scatter_facet_preserves_real_errorbars[xerr-has_xerr]": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A1_scatter_facet_preserves_real_errorbars[yerr-has_yerr]": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[hist-time_s-extra2]": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[profile-y:time_s-extra1]": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[scatter-y:time_s-extra0]": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A3_hist_facet_preserves_hist_errors": "visual_primitive",
    "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A4_profile_group_facet_preserves_explicit_color": "visual_primitive",
    "test_phase_13_86_stage_bb.py::test_BB9_expression_vector_channel_identity_is_stable_across_facets": "visual_primitive",
    "test_phase_13_86_stage_ca.py::test_CA1_wide_figure_keeps_full_selection_but_narrow_figure_elides": "visual_primitive",
    "test_phase_13_86_stage_ca.py::test_CA3_faceted_wide_figure_keeps_full_selection_in_figure_title": "visual_primitive",
    # ── PHASE_13_88_DF — normalization bug-fix proof layers ──
    # Bug A: exact canonical shared-range/bin assertions.
    "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[None]": "invariance",
    "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[hybrid]": "invariance",
    "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[robust_4mad]": "invariance",
    "test_phase_13_88_bug_a_hist_autorange.py::test_A2_explicit_numeric_range_remains_exact_green_control": "invariance",
    "test_phase_13_88_bug_a_hist_autorange.py::test_A3_weights_vector_same_rows_remains_green_control": "invariance",
    "test_phase_13_88_bug_a_hist_autorange.py::test_A4_two_expression_vector_uses_union_domain": "invariance",

    # Bug B: exact 0.5 numerical ratio oracle across owners/expressions.
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[plain-[y,2*y]:x]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[plain-[y,2*y]:2*x]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[grouped-[y,2*y]:x]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[grouped-[y,2*y]:2*x]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[faceted-[y,2*y]:x]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[faceted-[y,2*y]:2*x]": "invariance",
    # B2 proves the integrated selection-vector + computed-x path is live,
    # but does not independently assert a numerical equality.
    "test_phase_13_88_bug_b_profile_expressions.py::test_B2_single_y_computed_x_supported_with_selection_vector": "integration",

    # F7: every owner must reproduce the canonical resolve_range_1d centers.
    "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[plain]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[grouped]": "invariance",
    "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[faceted]": "invariance",

    # Bug C: public draw-door / typed-owner equality plus independent NumPy truth.
    "test_phase_13_88_bug_c_draw_hist_normalize.py::test_C1_draw_hist_vector_normalize_matches_typed_owner": "invariance",
    "test_phase_13_88_bug_c_draw_hist_normalize.py::test_C2_draw_hist_computed_vector_element_is_materialized": "invariance",

    # ── PHASE_13_89_DF — explicit-axis normalize ownership ──
    # Supported embedding: ownership/geometry invariants plus exact numerical
    # parity against the established standalone normalize owner.
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[delta-typed]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[delta-draw]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[ratio-typed]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[ratio-draw]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R2_draw_selection_vector_production_shape_embeds_and_preserves_numbers": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R4_diff_only_uses_exact_supplied_axis[delta]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R4_diff_only_uses_exact_supplied_axis[ratio]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R5_multisubplot_embedding_changes_only_target_slot": "invariance",

    # Standalone control proves the neighboring integrated owners remain live,
    # but does not independently rederive their numerical statistics.
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R10_profile_and_hist_standalone_controls_remain_supported": "integration",

    # Fail-loud behavior is itself part of the public ownership contract:
    # unsupported combinations must refuse before caller mutation or leakage.
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R6_custom_add_axes_refuses_before_mutation_or_leak": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R7_nonempty_placeholder_refuses_before_mutation_or_leak": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R8_grouped_or_faceted_normalize_with_ax_refuses_without_leak[grouped]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R8_grouped_or_faceted_normalize_with_ax_refuses_without_leak[faceted]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R9_hist_normalize_ax_preserves_existing_loud_refusal[typed]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R9_hist_normalize_ax_preserves_existing_loud_refusal[draw]": "invariance",
    "test_phase_13_89_explicit_ax_normalize_red.py::test_R11_ax_plus_save_refuses_without_mutating_caller": "invariance",

    # ── PHASE_13_91_DF — histogram expression-vector facets ──
    # Exact independent NumPy histogram counts and public-door equivalence.
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R1_R2_computed_expression_vector_facets_match_numpy_truth[typed]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R1_R2_computed_expression_vector_facets_match_numpy_truth[draw]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R3_typed_and_draw_doors_have_identical_panel_values": "invariance",

    # Metadata assertions prove integrated ADF/dfdraw label propagation, not
    # independent histogram arithmetic.
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R4_physical_column_vector_facets_preserve_component_x_metadata[typed]": "integration",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R4_physical_column_vector_facets_preserve_component_x_metadata[draw]": "integration",

    # Loud refusal before Figure allocation is part of the bounded public
    # semantic contract and therefore counts as invariance evidence.
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R5_duplicate_expression_identity_refuses_before_figure_allocation[typed]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R5_duplicate_expression_identity_refuses_before_figure_allocation[draw]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[selection-vector]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[weights-vector]": "invariance",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[normalize]": "invariance",

    # Neighboring/backward-compatibility controls are integrated path checks.
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R7_nonfaceted_hist_expression_vector_overlay_remains_supported": "integration",
    "test_phase_13_91_hist_expression_vector_facets_red.py::test_R8_profile_expression_vector_facets_remain_supported": "integration",

    # ── PHASE_13_92_DF — mapped scatter color + error-bar composition ──
    # Independent geometry, row-domain, ownership, z-order, normalization,
    # non-mutation and fail-loud assertions are semantic invariance evidence.
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R1_numeric_mapped_color_plus_yerr_two_layer": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R2_numeric_mapped_color_plus_xerr_two_layer": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R3_numeric_mapped_color_plus_both_errors_preserves_both_geometries": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R4_R5_faceted_numeric_mapped_color_preserves_points_and_errors[yerr]": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R4_R5_faceted_numeric_mapped_color_preserves_points_and_errors[xerr]": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R6_computed_color_expression_plus_errorbar_and_colorbar": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R7_colorbar_mappable_is_exact_point_collection_object": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R8_asymmetric_row_alignment_after_selection_nan_and_range_filter": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R9_errorbars_have_explicitly_lower_zorder_than_points": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R10_per_point_marker_expression_plus_yerr_keeps_marker_owner": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_G1_string_categorical_color_plus_errors_remains_fail_loud": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R11_caller_dataframe_not_mutated_by_composed_path": "invariance",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R12_marker_split_preserves_full_numeric_color_domain_and_colorbar_owner": "invariance",

    # Explicit backward-compatibility controls exercise established neighboring
    # owners rather than independently rederiving their full numerical semantics.
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_C1_fixed_scalar_color_plus_errors_keeps_legacy_errorbar_owner": "integration",
    "test_phase_13_92_scatter_mapped_color_errorbars.py::test_C2_numeric_color_only_keeps_ordinary_scatter_path": "integration",

    # ── PHASE_13_93_DF — dense N-D Figure presentation and categorical legend ──
    # Layout-call identity, exact ordered keys, value/error preservation,
    # Figure-artist counts/ownership, color equivalence across facets/pages,
    # and refusal contracts are invariance proofs. Four neighboring/legacy
    # controls are integration coverage rather than independent oracle proofs.
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R1_2d_nd_facet_has_exactly_one_whole_figure_layout_pass': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R2_3d_nd_facet_has_one_layout_pass_per_generated_figure': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R3_ordered_categorical_rows_preserve_declared_order': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R4_ordered_categorical_columns_preserve_declared_order': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R5_explicit_facet_order_overrides_categorical_order': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R5b_invalid_facet_order_refuses_clearly': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R6_outer_column_headings_appear_once_each': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R7_outer_row_headings_appear_once_each': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R8_outer_mode_removes_outer_dimension_text_from_panels': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R9_shared_xlabel_is_one_figure_semantic_label': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R10_shared_ylabel_preserves_independent_limits_and_ticks': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R11_explicit_figure_margins_survive_final_layout': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R12_explicit_subplot_spacing_survives_final_layout': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R13_legacy_2d_facet_presentation_unchanged_without_new_options': 'integration',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H1_public_draw_retains_unscaled_scatter_errorbar_values': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[scatter-coefficient:region-extra0]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[hist-coefficient-extra1]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[profile-coefficient:region-extra2]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[hist2d-coefficient:region-extra3]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H3_3D_categorical_figure_order_and_outer_headings': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad0]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad1]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad2]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad3]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H5_scalar_facet_rejects_ND_only_presentation_options': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H6_shared_ylabel_without_outer_mode_keeps_row_identity': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[figure_margins-value0]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[figure_margins-value1]': 'invariance',
    'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[subplot_spacing-value2]': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b1_omitted_color_display_keeps_original_colorbars': 'integration',
    'test_phase_13_93_stage_b_color_display_red.py::test_b2_b3_binary_numeric_side_one_figure_legend[auto]': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b2_b3_binary_numeric_side_one_figure_legend[legend]': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b4_explicit_colorbar_is_not_silently_changed_to_legend': 'integration',
    'test_phase_13_93_stage_b_color_display_red.py::test_b5_same_side_same_rgba_in_all_facets_even_singleton': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b6_no_repeated_axes_legend': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b8_matching_marker_partition_produces_combined_legend': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b8b_non_equivalent_marker_partition_refuses': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b9_uncertainties_are_retained_with_two_marker_classes': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b10_auto_threshold_controls_discrete_heuristic': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b10b_explicit_legend_limit_refuses': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b11_bad_mode_rejected_before_plotting': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b12_invalid_threshold_rejected': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b13_real_grid_40_cells_one_legend': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b14_layout_still_one_owner': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b15_max_legend_alone_does_not_silently_do_nothing': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b16_string_color_with_errorbars_preserves_explicit_refusal': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b17_three_dimensional_facets_keep_one_color_domain_across_figures': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b18_numeric_color_expression_three_categories_with_one_legend': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b19_grouped_color_owner_does_not_silently_fabricate_side_legend': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b20_nonfaceted_explicit_none_is_legacy_compatible': 'integration',
    'test_phase_13_93_stage_b_color_display_red.py::test_b21_nonfaceted_legend_refuses_before_matplotlib': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R14_rendered_left_decorations_do_not_intersect[0.06]': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R14_rendered_left_decorations_do_not_intersect[0.08]': 'invariance',
    'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R15_too_narrow_left_gutter_warns_without_changing_margins': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b_R11_legend_disjoint_from_column_headings_and_title[0.89]': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b_R11_legend_disjoint_from_column_headings_and_title[0.92]': 'invariance',
    'test_phase_13_93_stage_b_color_display_red.py::test_b_R12_after_return_manual_annotation_edit_is_not_reversed': 'invariance',

    # ── Candidate N-D retained-state independent RED/GREEN oracles ──
    'test_phase_13_82_retained_state_red.py::test_t1_weighted_histogram_against_numpy': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t2_cancelled_weight_is_not_empty': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t3_weighted_profile_against_independent_raw_sums': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t4_categorical_coordinate_is_semantic_dimension': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t5_schema_name_and_partition_contract': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t6_sparse_underflow_overflow_nan_and_last_edge': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t7_one_node_class_different_sufficient_state': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t8_logical_payload_round_trip_with_metadata': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t9_invalid_resolved_edges_refused': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t10_nonfinite_weight_and_profile_value_not_accumulated': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t11_numeric_categories_preserve_identity': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t12_corrupt_payload_fractional_entries_refused': 'invariance',
    'test_phase_13_82_retained_state_red.py::test_t13_two_named_categorical_dimensions_form_sparse_product': 'invariance',


    # PHASE_13_94_DF — N-D profile expression-vector composition.
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R1_two_branches_in_each_2d_facet_without_group[typed]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R1_two_branches_in_each_2d_facet_without_group[draw]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R2_computed_vector_residuals_equal_materialized_truth': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R3_numeric_3x5_facet_classification_matches_global_pandas_membership': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R4_categorical_group_composes_2_vectors_times_3_groups': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R5_global_group_bins_maintain_physical_identity_across_cells': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R6_group_quantile_boundaries_resolved_globally': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R7_vector_group_identity_and_single_figure_legend': 'visual_primitive',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R8_strong_numeric_decomposition_with_selection_and_all_coordinates': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R9_missing_facet_cell_preserves_empty_cell_schema': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R10_forwarding_nondefault_min_entries_50_masks_short_profiles': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R11_typed_and_public_draw_nested_stats_equivalent': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R12_existing_one_dimensional_profile_vector_facet_still_works': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R13_neighbor_hist_scatter_nd_vector_still_refuse_loudly[hist-[y0,y1]]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R13_neighbor_hist_scatter_nd_vector_still_refuse_loudly[scatter-[y0,y1]:x]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R14_existing_scalar_nd_facet_is_not_changed_by_new_global_policy': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R15_missing_intermediate_group_keeps_global_color_marker_and_label': 'visual_primitive',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted0]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted1]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted2]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted3]': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R17_figure_channel_assignment_is_resolved_once': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R18_explicit_group_color_override_remains_authoritative': 'invariance',
    'test_phase_13_94_profile_vector_nd_facet_red.py::test_R19_group_bins_without_group_column_fail_before_figure': 'invariance',


    # PHASE_13_95_DF — strong independent numerical/semantic oracles.
    'test_phase_13_95_group_scope_title_red.py::test_R1_public_signature_and_invalid_values': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R2_auto_preserves_historical_1d_faceted_numbers': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R3_explicit_global_1d_reuses_exact_physical_groups': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R4_explicit_local_1d_matches_auto': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R5_global_quantiles_in_1d': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R6_global_missing_group_2d_has_stable_identity': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R7_global_figure_legend_obeys_explicit_hide': 'visual_primitive',
    'test_phase_13_95_group_scope_title_red.py::test_R8_facet_local_has_no_false_global_numeric_legend': 'visual_primitive',
    'test_phase_13_95_group_scope_title_red.py::test_R9_13_94_auto_matches_omitted_policy': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R10_13_94_explicit_facet_policy_changes_group_boundaries': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R11_1d_quantiles_appear_once_in_auto_title': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R12_title_false_and_explicit_title_precedence': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R13_typed_draw_parity_for_explicit_global[typed]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R13_typed_draw_parity_for_explicit_global[draw]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R14_categorical_no_binning_no_numeric_changes': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R15_single_global_cut_event': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R16_selection_domain_global_bin_membership_independent_pandas': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R17_single_dimension_list_is_equivalent_to_scalar_facet': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R18_global_quantile_boundaries_derived_from_selection_not_raw_input': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R19_more_than_two_quantiles_are_listed_not_misrepresented_as_range': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R20_no_scope_widening_to_other_plot_families': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R21_global_top_k_does_not_claim_unsupported_global_ranking': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R22_2d_scalar_global_matches_selected_pandas_counts': 'invariance',

    'test_phase_13_95_group_scope_title_red.py::test_R23_auto_vector_13_94_figure_legend_obeys_explicit_hide': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R24_explicit_all_overrides_default_shared_global_legend': 'visual_primitive',
    'test_phase_13_95_group_scope_title_red.py::test_R25_vector_1d_global_bins_match_full_selected_pandas_and_scalar[draw]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R25_vector_1d_global_bins_match_full_selected_pandas_and_scalar[typed]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R26_vector_1d_global_quantile_scope_reuses_selected_boundaries[draw]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R26_vector_1d_global_quantile_scope_reuses_selected_boundaries[typed]': 'invariance',
    'test_phase_13_95_group_scope_title_red.py::test_R27_vector_1d_auto_and_facet_retain_legacy_local_semantics': 'invariance',

}
