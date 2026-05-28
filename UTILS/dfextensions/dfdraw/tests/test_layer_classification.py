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

    # Everything else defaults to "smoke"
}
