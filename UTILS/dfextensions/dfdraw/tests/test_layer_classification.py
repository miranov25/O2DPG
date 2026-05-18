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

    # Everything else defaults to "smoke"
}
