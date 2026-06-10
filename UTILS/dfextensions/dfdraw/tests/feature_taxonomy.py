"""
Feature Taxonomy — dfdraw

Maps feature IDs to their test references.
Used by generate_capability_matrix.py to build the capability matrix.

Format follows GroupByRegression reference implementation.

Phase 13.15.DF — DRAFT for team vote
"""

FEATURES = [

    # ── Core Drawing ──

    {
        "id": "CORE.constructor",
        "name": "DFDraw constructor and data normalization",
        "category": "CORE",
        "tests": [
            "test_drawer.py::TestDrawDispatch::test_draw_1d_dispatches_to_hist",
        ],
    },
    {
        "id": "CORE.sampling",
        "name": "Random sampling (reproducible)",
        "category": "CORE",
        "tests": [
            "test_drawer.py::TestSampling::test_sampling_is_reproducible",
        ],
    },

    # ── Plot Types ──

    {
        "id": "PLOT.histogram",
        "name": "1D histogram",
        "category": "PLOT",
        "tests": [
            "test_histogram.py::TestBasicHistogram::test_hist_returns_tuple",
            "test_histogram.py::TestHistogramNormalization::test_hist_norm_count",
        ],
    },
    {
        "id": "PLOT.scatter",
        "name": "Scatter plot",
        "category": "PLOT",
        "tests": [
            "test_scatter.py::TestBasicScatter::test_scatter_returns_tuple",
        ],
    },
    {
        "id": "PLOT.profile",
        "name": "Profile plot (mean per bin)",
        "category": "PLOT",
        "tests": [
            "test_profile.py::TestBasicProfile::test_profile_returns_tuple",
        ],
    },
    {
        "id": "PLOT.hist2d",
        "name": "2D histogram (density heatmap)",
        "category": "PLOT",
        "tests": [
            "test_hist2d.py::TestBasicHist2D::test_hist2d_returns_tuple",
        ],
    },
    {
        "id": "PLOT.hexbin",
        "name": "Hexbin plot (hexagonal binning)",
        "category": "PLOT",
        "tests": [
            "test_hexbin.py::TestBasicHexbin::test_hexbin_returns_tuple",
        ],
    },

    # ── Profile Enhancements (Phase 13.12.DF) ──

    {
        "id": "PROFILE.return_data",
        "name": "Profile data export (return_data=True)",
        "category": "PROFILE",
        "tests": [
            "test_profile_phase13_12.py::TestReturnData::test_profile_return_data_structure",
            "test_profile_phase13_12.py::TestReturnData::test_profile_return_data_grouped",
            "test_profile_phase13_12.py::TestReturnData::test_profile_return_data_false",
        ],
    },
    {
        "id": "PROFILE.min_entries",
        "name": "Minimum entries filter (min_entries=3)",
        "category": "PROFILE",
        "tests": [
            "test_profile_phase13_12.py::TestMinEntries::test_profile_min_entries_filter",
            "test_profile_phase13_12.py::TestMinEntries::test_profile_min_entries_in_data",
            "test_profile_phase13_12.py::TestMinEntries::test_profile_min_entries_default",
        ],
    },
    {
        "id": "PROFILE.group_by_bins",
        "name": "Auto-bin float group_by (bins/quantiles)",
        "category": "PROFILE",
        "tests": [
            "test_profile_phase13_12.py::TestGroupByBins::test_group_by_bins",
            "test_profile_phase13_12.py::TestGroupByBins::test_group_by_quantiles",
            "test_profile_phase13_12.py::TestGroupByBins::test_group_by_bins_label_format",
            "test_profile_phase13_12.py::TestGroupByBins::test_group_by_mutual_exclusion",
            # Phase 13.37 FIX1: Phase 13.36 backward-compat invariance lock
            "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_1_group_by_bins_default_cycle_byte_identical",
        ],
    },
    {
        "id": "PROFILE.sort_groups",
        "name": "Sorted group order (negative-safe intervals)",
        "category": "PROFILE",
        "tests": [
            "test_profile_phase13_12.py::TestSortGroups::test_sort_groups_numeric",
            "test_profile_phase13_12.py::TestSortGroups::test_sort_groups_string",
            "test_profile_phase13_12.py::TestSortGroups::test_sort_groups_false",
        ],
    },
    {
        "id": "PROFILE.weights",
        "name": "Weighted profile (column or expression)",
        "category": "PROFILE",
        "tests": [
            "test_profile_phase13_12.py::TestWeightExpressions::test_weight_column_name",
            "test_profile_phase13_12.py::TestWeightExpressions::test_weight_expression",
            "test_profile_phase13_12.py::TestWeightExpressions::test_weight_expression_grouped",
            "test_profile_phase13_12.py::TestWeightExpressions::test_weight_invalid_raises",
            "test_profile_phase13_12.py::TestWeightExpressions::test_weight_none_unchanged",
        ],
    },

    # ── Auto Title (Phase 13.12.DF v1.2) ──

    {
        "id": "TITLE.auto_title",
        "name": "Automatic title from plot parameters",
        "category": "TITLE",
        "tests": [
            "test_auto_title.py::TestBuildAutoTitle::test_all_parts",
            "test_auto_title.py::TestBuildAutoTitle::test_basic_2d",
            "test_auto_title.py::TestBuildAutoTitle::test_basic_1d",
            "test_auto_title.py::TestAutoTitleProfile::test_auto_title_true",
            "test_auto_title.py::TestAutoTitleProfile::test_explicit_title_overrides",
            "test_auto_title.py::TestAutoTitleHist::test_auto_title_hist",
            "test_auto_title.py::TestAutoTitleHist2d::test_auto_title_hist2d",
            "test_auto_title.py::TestAutoTitleStyle::test_per_call_overrides_style",
            "test_auto_title.py::TestParseAutoTitleParts::test_true",
            "test_auto_title.py::TestParseAutoTitleParts::test_expr",
        ],
    },

    # ── Superposition — same=True (Phase 13.13.DF) ──

    {
        "id": "SAME.axes_reuse",
        "name": "same=True reuses last axes",
        "category": "SAME",
        "tests": [
            "test_same.py::TestSameBasic::test_same_reuses_axes_profile",
            "test_same.py::TestSameBasic::test_same_reuses_axes_hist",
            "test_same.py::TestSameBasic::test_same_false_creates_new",
            "test_same.py::TestSameBasic::test_same_stores_last_ax",
            "test_same.py::TestSameBasic::test_same_three_overlays",
            "test_same.py::TestSameFallback::test_fallback_to_gca",
            "test_same.py::TestSameFallback::test_fallback_no_axes_creates_new",
        ],
    },
    {
        "id": "SAME.auto_features",
        "name": "same=True auto-color, auto-label, legend",
        "category": "SAME",
        "tests": [
            "test_same.py::TestColorCycleReset::test_new_figure_resets_cycle",
            "test_same.py::TestColorCycleReset::test_colors_differ",
            "test_same.py::TestSameLegend::test_legend_auto_shown",
            "test_same.py::TestSameLegend::test_auto_label_content",
            # Phase 13.37 FIX1: Phase 13.36 same=True + group_by auto-color guard
            "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_2_same_true_group_by_no_auto_color_injection",
        ],
    },
    {
        "id": "SAME.title_append",
        "name": "same=True title append + subtitle merge",
        "category": "SAME",
        "tests": [
            "test_same.py::TestSameWithAutoTitle::test_title_append",
            "test_same.py::TestSameWithAutoTitle::test_title_multiline",
            "test_same.py::TestSameWithAutoTitle::test_explicit_title_overrides",
            "test_same.py::TestSameWithAutoTitle::test_subtitle_merge",
        ],
    },
    {
        "id": "SAME.override",
        "name": "same=True precedence (ax= wins, explicit overrides)",
        "category": "SAME",
        "tests": [
            "test_same.py::TestSameOverride::test_ax_precedence",
            "test_same.py::TestSameOverride::test_explicit_color",
            "test_same.py::TestSameOverride::test_explicit_label",
        ],
    },
    {
        "id": "SAME.cross_method",
        "name": "same=True across plot types (profile on hist2d)",
        "category": "SAME",
        "tests": [
            "test_same.py::TestSameAcrossMethods::test_profile_on_hist2d",
            "test_same.py::TestSameAcrossMethods::test_profile_on_hexbin",
            "test_same.py::TestSameAcrossMethods::test_hist_overlay",
            "test_same.py::TestSameAcrossMethods::test_draw_dispatch_same",
        ],
    },

    # ── Batch Processing (Phase 6.9 + 13.14.DF) ──

    {
        "id": "BATCH.dict_format",
        "name": "draw_batch dict format (original)",
        "category": "BATCH",
        "tests": [
            "test_batch.py::TestDrawBatchBasic::test_summary_keys_present",
            "test_batch.py::TestDrawBatchBasic::test_success_count_correct",
            "test_batch.py::TestDrawBatchDefaults::test_spec_overrides_defaults",
            "test_batch.py::TestDrawBatchPlotTypes::test_all_plot_types",
            "test_batch_groups.py::test_old_dict_format_unchanged",
        ],
    },
    {
        "id": "BATCH.group_format",
        "name": "draw_batch group format with defaults hierarchy",
        "category": "BATCH",
        "tests": [
            "test_batch_groups.py::test_list_format_single_group",
            "test_batch_groups.py::test_defaults_cascade",
            "test_batch_groups.py::test_defaults_override",
            "test_batch_groups.py::test_multiple_groups",
            "test_batch_groups.py::test_value_correctness",
            "test_batch_groups.py::test_mixed_old_new_format",
        ],
    },
    {
        "id": "BATCH.subplot_grid",
        "name": "Subplot grid (ncols, layout, figsize, suptitle)",
        "category": "BATCH",
        "tests": [
            "test_batch_groups.py::test_ncols_layout",
            "test_batch_groups.py::test_layout_explicit",
            "test_batch_groups.py::test_layout_overrides_ncols",
            "test_batch_groups.py::test_empty_subplot_hidden",
            "test_batch_groups.py::test_suptitle_applied",
            "test_batch_groups.py::test_figsize_applied",
            "test_batch_groups.py::test_savefig_output",
        ],
    },
    {
        "id": "BATCH.same_in_group",
        "name": "same=True within batch groups",
        "category": "BATCH",
        "tests": [
            "test_batch_groups.py::test_same_true_within_group",
            "test_batch_groups.py::test_same_true_first_plot_raises",
        ],
    },
    {
        "id": "BATCH.verbose",
        "name": "Verbose levels (0/1/2)",
        "category": "BATCH",
        "tests": [
            "test_batch_groups.py::test_verbose_false_silent",
            "test_batch_groups.py::test_verbose_true_progress",
            "test_batch_groups.py::test_verbose_2_debug",
        ],
    },
    {
        "id": "BATCH.save",
        "name": "Batch save and close figures",
        "category": "BATCH",
        "tests": [
            "test_batch.py::TestDrawBatchSaveDir::test_creates_directory",
            "test_batch.py::TestDrawBatchCloseFigures::test_close_figures_true_clears_fig",
        ],
    },

    # ── Style System ──

    {
        "id": "STYLE.predefined",
        "name": "Predefined styles (default, publication, presentation)",
        "category": "STYLE",
        "tests": [
            "test_style.py::TestSetStyle::test_set_style_predefined",
            "test_style.py::TestSetStyle::test_set_style_none_resets",
            "test_style.py::TestListStyles::test_list_styles_returns_list",
        ],
    },
    {
        "id": "STYLE.custom",
        "name": "Custom style dict and JSON persistence",
        "category": "STYLE",
        "tests": [
            "test_style.py::TestSetStyle::test_set_style_custom_dict",
            "test_style.py::TestSaveLoadStyle::test_save_and_load_style",
            "test_style.py::TestSaveLoadStyle::test_saved_style_is_valid_json",
            "test_style.py::TestDefaultStyle::test_get_style_returns_copy",
        ],
    },

    # ── Statistics (Phase 13.6.G.DF) ──

    {
        "id": "STATS.default_fields",
        "name": "Auto-detect default stats fields by plot type",
        "category": "STATS",
        "tests": [
            "test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_fallback_1d_detection",
        ],
    },
    {
        "id": "STATS.range_aware",
        "name": "Range-aware statistics (range_x, range_y)",
        "category": "STATS",
        "tests": [
            "test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_range_y_only_for_2d",
            "test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_empty_range_2d",
            "test_stats_enhancements.py::TestIssue2_2D_n_Semantics::test_2d_n_counts_both_valid",
        ],
    },
    {
        "id": "STATS.robust",
        "name": "Robust statistics (median, MAD)",
        "category": "STATS",
        "tests": [
            "test_stats_enhancements.py::TestIssue3_RobustStats::test_robust_empty_range",
            "test_stats_enhancements.py::TestIssue3_RobustStats::test_robust_2d_applies_to_y_only",
        ],
    },

    # ── Annotations (Phase 12.4b5) ──

    {
        "id": "ANNOT.statistics_box",
        "name": "Statistics annotation box",
        "category": "ANNOT",
        "tests": [
            "test_validation_display.py::TestStatisticsBox::test_statistics_box_created",
            "test_validation_display.py::TestStatisticsBox::test_statistics_box_with_expected",
            "test_validation_display.py::TestStatisticsBox::test_statistics_box_positions",
        ],
    },
    {
        "id": "ANNOT.reference_overlay",
        "name": "Reference function overlay (Gaussian, callable)",
        "category": "ANNOT",
        "tests": [
            "test_validation_display.py::TestReferenceOverlay::test_gaussian_overlay_created",
            "test_validation_display.py::TestReferenceOverlay::test_custom_callable_overlay",
            "test_validation_display.py::TestReferenceOverlay::test_overlay_has_label",
            "test_validation_display.py::TestCombinedUsage::test_pull_distribution_workflow",
        ],
    },

    # ── ADF Integration (Phase 6.8) ──

    {
        "id": "ADF.axis_titles",
        "name": "Duck-typed axis titles from AliasDataFrame",
        "category": "ADF",
        "tests": [
            "test_adf_integration.py::TestDuckTypedAxisTitles::test_hist_uses_axis_title",
            "test_adf_integration.py::TestDuckTypedAxisTitles::test_scatter_uses_axis_titles",
            "test_adf_integration.py::TestGetLabelMethod::test_get_label_with_title",
            "test_adf_integration.py::TestGetLabelMethod::test_get_label_without_title",
            "test_adf_integration.py::TestDFDrawDataSourceStorage::test_data_source_stored",
        ],
    },

    # ── PyArrow (Phase 13.1.DF) ──

    {
        "id": "PYARROW.input",
        "name": "PyArrow Table input support",
        "category": "PYARROW",
        "tests": [
            "test_pyarrow_input.py::TestPyArrowInputAcceptance::test_init_pyarrow_table",
            "test_pyarrow_input.py::TestBackendProperty::test_backend_pyarrow",
            "test_pyarrow_input.py::TestResultsParity::test_hist_parity",
            "test_pyarrow_input.py::TestResultsParity::test_scatter_parity",
            "test_pyarrow_input.py::TestResultsParity::test_profile_parity",
            "test_pyarrow_input.py::TestResultsParity::test_hist2d_parity",
            "test_pyarrow_input.py::TestResultsParity::test_hexbin_parity",
        ],
    },

    # ── Faceting (Phase 6.4) ──

    {
        "id": "FACET.grid",
        "name": "Facet subplot grids (group_by + facet=True)",
        "category": "FACET",
        "tests": [
            "test_profile_phase13_12.py::TestFacetMode::test_facet_profile_with_new_params",
        ],
    },

    # ── Backward Compatibility ──

    {
        "id": "COMPAT.profile",
        "name": "Profile backward compatibility",
        "category": "COMPAT",
        "tests": [
            "test_profile_phase13_12.py::TestBackwardCompatibility::test_existing_call_unchanged",
            "test_profile_phase13_12.py::TestBackwardCompatibility::test_all_original_parameters_work",
        ],
    },

    # ── Vector Expression Interface (Phase 13.16.DF) ──

    {
        "id": "VECTOR.parse",
        "name": "Bracket syntax parsing with paren-aware split",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestParserScalar::test_parse_scalar_2d_unchanged",
            "test_vector.py::TestParserScalar::test_parse_scalar_with_paren_y",
            "test_vector.py::TestParserScalar::test_parse_three_colons_still_raises",
            "test_vector.py::TestParserVectorBroadcast::test_N1",
            "test_vector.py::TestParserVectorBroadcast::test_1N",
            "test_vector.py::TestParserVectorBroadcast::test_NN_equal",
            "test_vector.py::TestParserVectorBroadcast::test_NM_broadcast_mismatch_raises",
            "test_vector.py::TestParserParenInsideBracket::test_paren_comma_inside_bracket",
        ],
    },
    {
        "id": "VECTOR.dispatch",
        "name": "Vector dispatch across profile/hist/scatter/draw",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorDispatch::test_profile_N1",
            "test_vector.py::TestVectorDispatch::test_hist_1d_vector",
            "test_vector.py::TestVectorDispatch::test_scatter_NN",
            "test_vector.py::TestVectorDispatch::test_draw_dispatches_profile",
            "test_vector.py::TestVectorDispatch::test_draw_1d_auto_dispatches_hist",
        ],
    },
    {
        "id": "VECTOR.fail_fast",
        "name": "Fail-fast guards on hist2d/hexbin; per-pair for stats",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorFailFast::test_hist2d_vector_raises",
            "test_vector.py::TestVectorFailFast::test_hexbin_vector_raises",
            "test_vector.py::TestVectorFailFast::test_stats_vector_returns_list",
        ],
    },
    {
        "id": "VECTOR.style_channels",
        "name": "Vector + group_by style channel decomposition (P1-2)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorChannels::test_default_vector_only_uses_color",
            "test_vector.py::TestVectorChannels::test_vector_style_linestyle_explicit",
            "test_vector.py::TestVectorChannels::test_same_channel_collision_raises",
            "test_vector.py::TestVectorChannels::test_vector_style_invalid_raises",
        ],
    },
    {
        "id": "VECTOR.contract",
        "name": "Return contract: stats_list, ylabel, auto_title",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorContract::test_stats_list_length",
            "test_vector.py::TestVectorContract::test_ylabel_common_prefix",
            "test_vector.py::TestVectorContract::test_auto_title_default_on_for_vector",
        ],
    },
    {
        "id": "VECTOR.color_cycle",
        "name": "Color cycle continuity with outer same=True (GPT5 fix)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorColorCycle::test_fresh_vector_draw_resets_cycle",
            "test_vector.py::TestVectorColorCycle::test_vector_same_chain_continues_color_cycle",
            "test_vector.py::TestVectorColorCycle::test_outer_same_true_not_swallowed",
            # Phase 13.37 FIX1: vector+group_by rendering matrix invariance lock
            "test_phase_13_37_df_hist_robustness.py::TestPhase1336BackwardCompat::test_SO_COMPAT_3_vector_group_by_palette_and_channels_preserved",
        ],
    },
    {
        "id": "VECTOR.adf_integration",
        "name": "Vector through AliasDataFrame entry point (P0-3)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorThroughADF::test_vector_through_adf_draw",
        ],
    },
    {
        "id": "VECTOR.invariance",
        "name": "Vector ≡ scalar same-loop semantic invariance (strong A≡B)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorInvariance::test_vector_N1_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorInvariance::test_vector_1N_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorInvariance::test_vector_NN_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorInvariance::test_vector_hist_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorInvariance::test_vector_through_adf_equivalent_to_direct",
            "test_vector.py::TestVectorInvariance::test_vector_chain_continuity_equivalent_to_full_scalar_loop",
            "test_vector.py::TestVectorInvariance::test_vector_determinism",
        ],
    },

    # ── Phase 13.16.DF FIX1 (2026-04-14) — Vector kwarg propagation bug fix ──
    # Permanent capability matrix entries. Pre-fix (d662c0a5): all tests FAIL.
    # Post-fix (PHASE_13_16_DF_FIX1_END): all tests PASS.
    # Diagnostic role: localizes bugs to dfdraw vs ADF subproject.

    {
        "id": "VECTOR.kwarg_propagation",
        "name": "Vector path forwards all scalar-mode kwargs (FIX1)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorKwargPropagation::test_vector_group_by_bins_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_group_by_quantiles_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_min_entries_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_sort_groups_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_linestyle_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_weights_equivalent_to_scalar_loop",
            "test_vector.py::TestVectorKwargPropagation::test_vector_return_data_equivalent_to_scalar_loop",
        ],
    },
    {
        "id": "VECTOR.groupby_polish",
        "name": "Vector + group_by deduplicated legend, title, layout (FIX1)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorGroupBy::test_vector_groupby_main_legend_dedup_count",
            "test_vector.py::TestVectorGroupBy::test_vector_groupby_secondary_legend_count",
            "test_vector.py::TestVectorGroupBy::test_vector_groupby_title_one_line",
            "test_vector.py::TestVectorGroupBy::test_vector_groupby_no_layout_warnings",
            "test_vector.py::TestVectorGroupBy::test_vector_groupby_real_world_reproducer",
        ],
    },
    {
        "id": "VECTOR.kwarg_surface",
        "name": "Vector dispatch forwards named-parameter surface + facet guard (FIX1)",
        "category": "VECTOR",
        "tests": [
            "test_vector.py::TestVectorKwargSurface::test_vector_profile_kwarg_surface_enumeration",
            "test_vector.py::TestVectorKwargSurface::test_vector_hist_kwarg_surface_enumeration",
            "test_vector.py::TestVectorKwargSurface::test_vector_scatter_kwarg_surface_enumeration",
            "test_vector.py::TestVectorKwargSurface::test_vector_draw_kwarg_surface_enumeration",
            "test_vector.py::TestVectorKwargSurface::test_vector_facet_with_vector_raises",
            "test_vector.py::TestVectorKwargSurface::test_all_forwarded_names_are_valid_signature_params",
        ],
    },
    # =========================================================================
    # Phase 13.25.DF FIX1: Quantile rendering on profile() (Phase A)
    # =========================================================================
    {
        "id": "QUANTILE.error_bars",
        "name": "Quantile error_bars mode (asymmetric bars from symmetric pair)",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_q16_q84_per_bin_matches_nanpercentile",
            "test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_q25_q75_per_bin_matches_nanpercentile",
            "test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_q05_q95_per_bin_matches_nanpercentile",
            "test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_yerr_is_asymmetric",
            "test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_q_lower_below_mean",
            "test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_q_upper_above_mean",
            "test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_with_central_mean",
            "test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_with_central_median",
        ],
    },
    {
        "id": "QUANTILE.band",
        "name": "Quantile band mode (fill_between from symmetric triple)",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_renders_polycollection",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_alpha_default_is_025",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_color_matches_central_line",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_y_lower_equals_q_lower",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_y_upper_equals_q_upper",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_with_central_none",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_alpha_from_style",
            "test_quantiles_profile.py::TestQuantileBandRendering::test_band_hatch_from_style",
        ],
    },
    {
        "id": "QUANTILE.central",
        "name": "Quantile central= parameter (mean/median/both/none)",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_mean_matches_existing_gb_mean",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_median_matches_nanmedian",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_median_computed_when_0_5_not_in_quantiles",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_both_renders_two_lines",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_none_with_band_omits_central_line",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_none_with_error_bars_raises_valueerror",
            "test_quantiles_profile.py::TestQuantileCentralLine::test_central_None_resolves_to_style_key",
        ],
    },
    {
        "id": "QUANTILE.auto_detection",
        "name": "Quantile mode auto-detection from list shape",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_symmetric_pair_no_05_returns_error_bars",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_symmetric_triple_with_05_returns_band",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_symmetric_pair_p25_p75_returns_error_bars",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_asymmetric_returns_discrete",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_multi_pair_returns_nested_band",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_single_value_returns_discrete",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_out_of_range_raises_valueerror",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_empty_list_raises_valueerror",
        ],
    },
    {
        "id": "QUANTILE.style_keys",
        "name": "Quantile style keys (band.alpha, band.hatch, error_bars.capsize, central_default)",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_band_alpha_default",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_band_hatch_default",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_capsize_default",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_central_default",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_override_band",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_override_capsize",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_save_load_roundtrip",
            "test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_namespace_integrity",
        ],
    },
    {
        "id": "QUANTILE.parity",
        "name": "Quantile determinism + backward-compat regression-lock",
        "category": "QUANTILE",
        "tests": [
            "test_quantiles_profile.py::TestQuantileDeterminism::test_det_error_bars_mean",
            "test_quantiles_profile.py::TestQuantileDeterminism::test_det_band_mean",
            "test_quantiles_profile.py::TestQuantileDeterminism::test_det_quantile_arrays",
            "test_quantiles_profile.py::TestQuantileBackwardCompat::test_no_quantiles_produces_one_errorbar_container",
            "test_quantiles_profile.py::TestQuantileBackwardCompat::test_no_quantiles_stats_match_reference",
            "test_quantiles_profile.py::TestQuantileBackwardCompat::test_production_pattern_unchanged",
        ],
    },
    # =========================================================================
    # BUG_dfdraw_20260505: Boolean expression crash fix
    # =========================================================================
    # =========================================================================
    # BUG_dfdraw_20260505: Boolean expression crash fix (Phase 13.25.DF FIX1)
    # =========================================================================
    {
        "id": "COMPAT.bool_expression",
        "name": "Boolean expression input (==, !=, >, <, &, |, ~) on all plot functions",
        "category": "COMPAT",
        "tests": [
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_equality_bool_histogram",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_inequality_bool",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_greater_than_bool",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_logical_and_bool",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_logical_or_bool",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_logical_not_bool",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_bool_group_by",
            "test_quantiles_profile.py::TestBoolExpressionHistogram::test_bool_values_correct",
            "test_quantiles_profile.py::TestBoolExpressionProfile::test_bool_y_profile",
            "test_quantiles_profile.py::TestBoolExpressionProfile::test_bool_x_profile",
            "test_quantiles_profile.py::TestBoolExpressionProfile::test_bool_y_with_quantiles",
        ],
    },
    # =========================================================================
    # Phase 13.26.DF (Phase B): N-Channel Framework — CHANNEL.* features
    # =========================================================================
    # Status: Planned (Commit 1 scaffolding; tests are pytest-skip stubs).
    # Will flip to Verified after Commit 2 implementation lands.
    {
        "id": "CHANNEL.assignment",
        "name": "Algorithm A: automatic visual-channel assignment for N data channels",
        "category": "CHANNEL",
        "tests": [
            "test_channel_assignment.py::TestChannelAssignment1Active::test_1ch_vector_alone_gets_color",
            "test_channel_assignment.py::TestChannelAssignment1Active::test_1ch_groupby_alone_gets_color",
            "test_channel_assignment.py::TestChannelAssignment1Active::test_1ch_quantiles_discrete_alone_gets_linestyle",
            "test_channel_assignment.py::TestChannelAssignment1Active::test_1ch_quantiles_band_no_channel",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_vector_groupby",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_vector_groupby_unconditional_on_cardinality",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_groupby_quantiles_discrete",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_vector_quantiles_discrete",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_groupby_quantiles_band",
            "test_channel_assignment.py::TestChannelAssignment2Active::test_2ch_vector_quantiles_band",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_default",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_with_band_is_2ch",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_unconditional_on_cardinality",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_renders_correctly",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_with_central_mean",
            "test_channel_assignment.py::TestChannelAssignment3Active::test_3ch_with_central_none",
            "test_channel_assignment.py::TestChannelCollision::test_collision_vector_group_same_channel",
            "test_channel_assignment.py::TestChannelCollision::test_collision_vector_quantile_same_channel",
            "test_channel_assignment.py::TestChannelCollision::test_collision_group_quantile_same_channel",
            "test_channel_assignment.py::TestChannelCollision::test_no_collision_when_zero_cost",
            "test_channel_assignment.py::TestChannelCapacity::test_overflow_color_gt_10",
            "test_channel_assignment.py::TestChannelCapacity::test_overflow_linestyle_gt_4",
            "test_channel_assignment.py::TestChannelCapacity::test_overflow_marker_gt_8",
            "test_channel_assignment.py::TestChannelCapacity::test_overflow_warn_mode",
            "test_channel_assignment.py::TestChannelUserOverride::test_percall_wins_over_style_default",
            "test_channel_assignment.py::TestChannelUserOverride::test_style_default_wins_over_explicit_rule",
            "test_channel_assignment.py::TestChannelUserOverride::test_quantile_style_override",
            "test_channel_assignment.py::TestChannelUserOverride::test_all_three_overridden",
            "test_channel_assignment.py::TestChannelStyleOverride::test_set_priority_changes_assignment",
            "test_channel_assignment.py::TestChannelStyleOverride::test_set_cycles_changes_capacity",
            "test_channel_assignment.py::TestChannelStyleOverride::test_save_load_round_trips_channels_keys",
            "test_channel_assignment.py::TestChannelStyleOverride::test_set_style_validates_list_values",
            "test_channel_assignment.py::TestChannelStyleOverride::test_default_style_has_all_channels_keys",
            "test_channel_assignment.py::TestChannelStyleOverride::test_namespace_integrity",
            "test_channel_assignment.py::TestProductionPatternBackwardCompat::test_groupby_quantiles8_structurally_equal",
            "test_channel_assignment.py::TestProductionPatternBackwardCompat::test_groupby_quantiles8_stats_rtol_1e_12",
            "test_channel_assignment.py::TestProductionPatternBackwardCompat::test_simple_profile_with_auto_title_unchanged",
            "test_channel_assignment.py::TestProductionPatternBackwardCompat::test_quantiles_016_084_error_bars_unchanged",
            "test_channel_assignment.py::TestProductionPatternBackwardCompat::test_makeSmoothMaps_kwargs_signature_unchanged",
            "test_channel_assignment.py::TestIdempotency::test_vector_path_calls_assign_once",
            "test_channel_assignment.py::TestIdempotency::test_scalar_path_calls_assign_once",
        ],
    },
    {
        "id": "CHANNEL.nested_band",
        "name": "Nested-band detection (>=2 symmetric pairs, central optional) and rendering",
        "category": "CHANNEL",
        "tests": [
            "test_channel_assignment.py::TestNestedBand::test_nested_band_detection_5_entry",
            "test_channel_assignment.py::TestNestedBand::test_nested_band_detection_no_central",
            "test_channel_assignment.py::TestNestedBand::test_nested_band_renders_polycollections",
            "test_channel_assignment.py::TestNestedBand::test_nested_band_outer_alpha_lt_inner",
            "test_channel_assignment.py::TestNestedBand::test_nested_band_max_3",
        ],
    },
    {
        "id": "CHANNEL.factored_legend",
        "name": "Factored legend with section headers (sum-not-product entry count)",
        "category": "CHANNEL",
        "tests": [
            "test_channel_assignment.py::TestFactoredLegend::test_factored_legend_entry_count_is_sum_not_product",
            "test_channel_assignment.py::TestFactoredLegend::test_factored_legend_has_section_headers",
            "test_channel_assignment.py::TestFactoredLegend::test_factored_false_uses_flat_dedup",
            "test_channel_assignment.py::TestFactoredLegend::test_2ch_legend_also_factored",
        ],
    },

    # ── Phase 13.27.DF Commit 2 (Phase D): Selection/Weights Vectors + delta_facet ──

    {
        "id": "CHANNEL.selection_delta",
        "name": "selection_delta channel (per-curve selection_vector) — AD-61, AD-66, AD-67",
        "category": "CHANNEL",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_1_1ch_selection_alone_gets_color",
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_2_2ch_selection_vector_inner",
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_3_selection_vector_with_global_selection",
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_3_hist2d_with_selection_vector_typeerror",
        ],
    },
    {
        "id": "CHANNEL.weights_delta",
        "name": "weights_delta channel (per-curve weights_vector) — AD-61, AD-66, AD-67",
        "category": "CHANNEL",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_1_1ch_weights_alone_gets_color",
            "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_3_weights_vector_with_global_weights",
            "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_1_scatter_weights_vector_warns_once",
            "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_3_hist2d_with_weights_vector_typeerror",
        ],
    },
    {
        "id": "CHANNEL.compose_inner",
        "name": "vector_compose='inner' — element-wise pairing (AD-62)",
        "category": "CHANNEL",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_1_2axis_inner_matched_lengths",
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_2_2axis_inner_mismatched_raises",
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_4_3axis_inner_all_equal",
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_5_3axis_inner_mismatched_raises",
        ],
    },
    {
        "id": "CHANNEL.compose_outer",
        "name": "vector_compose='outer' — cross-product (AD-62)",
        "category": "CHANNEL",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_3_2axis_outer_creates_mxn",
            "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_6_3axis_outer_creates_mxnxp",
        ],
    },
    {
        "id": "CHANNEL.delta_facet_label",
        "name": "Per-curve label management for selection_delta/weights_delta channels",
        "category": "CHANNEL",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_4_selection_labels_override_accepted",
            "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Profile::test_SDP_5_selection_truncate_style_key_registered",
            "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_4_weights_labels_kwarg_plumbed",
        ],
    },

    # ── Phase 13.28.DF: Robust Data Handling (NaN/Inf + Hybrid Autorange) ──

    {
        "id": "DATA.nan_policy",
        "name": "Optional NaN/inf filter with nan_policy parameter",
        "category": "DATA",
        "tests": [
            "test_data_sanitize_autorange.py::TestNaNInfFilter::test_inf_filtered_silently_with_default_policy",
            "test_data_sanitize_autorange.py::TestNaNInfFilter::test_nan_filtered_silently_with_default_policy",
            "test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_raise",
            "test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_warn_emits_warning_then_filters",
            "test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_filter_preserves_pre_phase_behavior",
            "test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_invalid_value_raises",
        ],
    },
    {
        "id": "DATA.counters",
        "name": "Stats dict counters: n_input, n_filtered, n_inf_*, n_nan_*",
        "category": "DATA",
        "tests": [
            "test_data_sanitize_autorange.py::TestNaNInfFilter::test_inf_y_only_counted_correctly",
            "test_data_sanitize_autorange.py::TestNaNInfFilter::test_no_finite_data_warns_with_filter_policy",
            "test_data_sanitize_autorange.py::TestNaNInfFilter::test_clean_data_no_counters_change",
            "test_data_sanitize_autorange.py::TestStatsDictAdditive::test_sanitize_stats_keys_present_and_well_typed",
            "test_data_sanitize_autorange.py::TestStatsDictAdditive::test_sanitize_counters_arithmetic_consistent",
        ],
    },
    {
        "id": "AUTORANGE.hybrid",
        "name": "Hybrid autorange (outlier-aware: robust + minmax combined)",
        "category": "AUTORANGE",
        "tests": [
            "test_data_sanitize_autorange.py::TestHybridAutorange::test_clean_gaussian_uses_minmax",
            "test_data_sanitize_autorange.py::TestHybridAutorange::test_outlier_high_clips_to_robust",
            "test_data_sanitize_autorange.py::TestHybridAutorange::test_outlier_low_clips_to_robust",
            "test_data_sanitize_autorange.py::TestHybridAutorange::test_asymmetric_distribution_one_sided_clip",
            "test_data_sanitize_autorange.py::TestHybridAutorange::test_constant_data_returns_unit_range",
        ],
    },
    {
        "id": "AUTORANGE.minmax",
        "name": "Minmax autorange strategy (backward compat preset)",
        "category": "AUTORANGE",
        "tests": [
            "test_data_sanitize_autorange.py::TestAutorangeStrategies::test_minmax_strategy_equals_data_min_max",
            "test_data_sanitize_autorange.py::TestAutorangeStrategies::test_invalid_strategy_raises",
        ],
    },
    {
        "id": "AUTORANGE.percentile",
        "name": "Percentile autorange strategies (percentile_99, percentile_95)",
        "category": "AUTORANGE",
        "tests": [
            "test_data_sanitize_autorange.py::TestAutorangeStrategies::test_percentile_99_clips_to_quantiles",
        ],
    },
    {
        "id": "AUTORANGE.diagnostics",
        "name": "Stats keys autorange_used + autorange_strategy (AD-77)",
        "category": "AUTORANGE",
        "tests": [
            "test_data_sanitize_autorange.py::TestStatsDictAdditive::test_compute_autorange_returns_finite_tuple",
        ],
    },

    # ── Phase 13.27.DF Commit 2 FIX1 — Histogram weights (HSW) ──

    {
        "id": "HIST.weights",
        "name": "hist() weights= column or expression — Phase 13.27 Commit 2 FIX1",
        "category": "DATA",
        "tests": [
            "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_1_hist_weights_column_renders_weighted",
            "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_2_hist_weights_expression_renders_weighted",
            "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_3_hist_weights_with_norm_probability",
            "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_4_hist_weights_with_group_by_raises",
            "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_HSW_5_hist_no_weights_backward_compat",
        ],
    },

    # ── Phase 13.30.DF — Column reference parameter validation (Class-2) ──

    {
        "id": "COLUMN_REF.validation",
        "name": "Column-reference parameter validation (Class-2 actionable errors) — Phase 13.30",
        "category": "COLUMN_REF",
        "tests": [
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_missing_column_raises_clear_error",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_existing_column_unchanged",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_groupby_none_unchanged",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Profile::test_weights_expression_still_works",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist_groupby_missing_raises",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist_groupby_existing_unchanged",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Hist::test_hist2d_no_groupby_param_unchanged",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Scatter::test_scatter_groupby_missing_raises",
            "test_phase_13_30_column_reference_validation.py::TestColumnReferenceValidation_Scatter::test_scatter_groupby_existing_unchanged",
            "test_phase_13_30_column_reference_validation.py::TestR6ColumnReferenceValidator::test_r6_catches_invented_param_in_tuple",
            "test_phase_13_30_column_reference_validation.py::TestR6ColumnReferenceValidator::test_r6_tuples_are_subset_of_forwarded",
            "test_phase_13_30_column_reference_validation.py::TestProductionReproducer::test_production_reproducer_now_raises_not_silent",
        ],
    },

    # ── Phase 13.31.DF — facet_by column-name mode (AD-78) ──

    {
        "id": "FACET.column_mode",
        "name": "facet_by accepts DataFrame column name (AD-78) — Phase 13.31",
        "category": "FACET",
        "tests": [
            "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_group_by_does_not_raise_p1330",
            "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_quantiles_does_not_raise_p1330",
            "test_phase_13_31_facet_by_column.py::TestPhase1330Safe::test_channel_facet_by_vector_does_not_raise_p1330",
            "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_int_column_facet_produces_n_subplots",
            "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_float_column_facet_works",
            "test_phase_13_31_facet_by_column.py::TestColumnNameMode_Numeric::test_subplot_titles_show_facet_value",
            "test_phase_13_31_facet_by_column.py::TestColumnNameMode_String::test_string_column_facet_works_without_quoting_issue",
            "test_phase_13_31_facet_by_column.py::TestOrthogonalComposition::test_facet_by_column_AND_group_by_compose",
            "test_phase_13_31_facet_by_column.py::TestAmbiguityError::test_typo_facet_by_raises_with_both_alternatives",
            "test_phase_13_31_facet_by_column.py::TestAmbiguityError::test_error_message_lists_available_columns",
            "test_phase_13_31_facet_by_column.py::TestCardinalityCap::test_column_facet_too_many_unique_values_triggers_cap",
            "test_phase_13_31_facet_by_column.py::TestRegressionCommit1::test_channel_group_by_mode_suppresses_inner_group_by",
        ],
    },

    # ── Phase 13.32.DF — group_by × quantiles × facet_by binning (AD-79) ──

    {
        "id": "FACET.column_mode_binning",
        "name": "facet_by_bins / facet_by_quantiles auto-binning of float column facets (AD-79) — Phase 13.32 Sub-fix 3",
        "category": "FACET",
        "tests": [
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_on_column_facet",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_quantiles_on_column_facet",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_on_channel_facet_raises",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_quantiles_mutual_exclusion",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3Profile::test_facet_by_bins_without_facet_by_raises",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_hist",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_hist2d",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix3AllPlots::test_facet_by_bins_for_scatter",
        ],
    },
    {
        "id": "PROFILE.quantiles_grouped",
        "name": "Per-group quantile band/discrete rendering on profile() — Phase 13.32 Sub-fix 2",
        "category": "QUANTILE",
        "tests": [
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_quantile_band_per_group_color",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_quantile_discrete_per_group",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_stats_dict_quantile_keys_present",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_weights_compose_with_groupby_quantiles",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_nested_band_with_groupby_raises",
            "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix2::test_discrete_grouped_uses_group_color_dashed",
        ],
    },

    # ── Phase 13.32.DF FIX1 — Faceted rendering bug fixes (BUG-001/002/003) ──

    {
        "id": "FACET.title_display_name",
        "name": "Subplot titles show original facet_by name, never internal __dfdraw_facet_bin__ (BUG-001) — Phase 13.32 FIX1",
        "category": "FACET",
        "tests": [
            "test_phase_13_32_df_fix1.py::TestBUG001FacetBinNameLeak::test_BUG001_facet_bin_col_name_not_in_subplot_titles",
        ],
    },
    {
        "id": "FACET.auto_title",
        "name": "auto_title=True produces fig.suptitle in faceted mode (BUG-002) — Phase 13.32 FIX1",
        "category": "FACET",
        "tests": [
            "test_phase_13_32_df_fix1.py::TestBUG002AutoTitleInFacetedMode::test_BUG002_auto_title_sets_suptitle_in_faceted_mode",
            "test_phase_13_32_df_fix1.py::TestBUG002AutoTitleInFacetedMode::test_BUG002_auto_title_channel_mode_facet_no_NameError",
        ],
    },
    {
        "id": "FACET.numeric_bin_sort",
        "name": "Facet bin panels in numeric order, not lexicographic (BUG-003) — Phase 13.32 FIX1",
        "category": "FACET",
        "tests": [
            "test_phase_13_32_df_fix1.py::TestBUG003FacetBinNumericSort::test_BUG003_facet_bins_sorted_numerically_not_lexicographically",
        ],
    },

    # ── Phase 13.33.DF — Normalized Differential Profiles (AD-80/81/82) ──

    {
        "id": "NORMALIZE.delta",
        "name": "normalize='delta': v[0]-v[1] per bin with SEM error propagation — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeDelta::test_ND_1_basic_delta",
            "test_normalize.py::TestNormalizeDelta::test_ND_2_delta_error_formula",
            "test_normalize.py::TestNormalizeDelta::test_ND_3_stats_dict_structure",
        ],
    },
    {
        "id": "NORMALIZE.ratio",
        "name": "normalize='ratio': v[0]/v[1] with delta-method error, zero-denom mask — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeRatio::test_NR_1_basic_ratio",
            "test_normalize.py::TestNormalizeRatio::test_NR_2_zero_denominator_masked",
            "test_normalize.py::TestNormalizeRatio::test_NR_3_ratio_error_formula",
        ],
    },
    {
        "id": "NORMALIZE.log_ratio",
        "name": "normalize='log_ratio': ln(v[0]/v[1]) with non-positive mean mask — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeLogRatio::test_NL_1_basic_log_ratio",
            "test_normalize.py::TestNormalizeLogRatio::test_NL_2_non_positive_mean_masked",
        ],
    },
    {
        "id": "NORMALIZE.pull",
        "name": "normalize='pull': (v[0]-v[1])/sigma with +/-1sigma/+/-2sigma bands (AD-82) — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizePull::test_NP_1_basic_pull_dimensionless",
            "test_normalize.py::TestNormalizePull::test_NP_2_pull_bands_rendered",
        ],
    },
    {
        "id": "NORMALIZE.callable",
        "name": "normalize=callable: user-supplied f(stats_0, stats_1) -> (values, errors) — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeCallable::test_NC_1_callable_with_errors",
            "test_normalize.py::TestNormalizeCallable::test_NC_2_callable_values_only",
        ],
    },
    {
        "id": "NORMALIZE.layout",
        "name": "normalize_layout: overlay+diff (2-panel) vs diff_only (single panel) — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeLayout::test_NLY_1_overlay_diff_creates_two_panels",
            "test_normalize.py::TestNormalizeLayout::test_NLY_2_diff_only_single_panel",
            "test_normalize.py::TestNormalizeLayout::test_NLY_3_height_ratio_style_respected",
        ],
    },
    {
        "id": "NORMALIZE.sign_convention",
        "name": "AD-80 sign convention: vector[0]=signal, vector[1]=reference; delta=signal-reference — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeSignConvention::test_NSC_1_signal_above_reference_delta_positive",
        ],
    },
    {
        "id": "NORMALIZE.single_y_convention",
        "name": "Single-Y + selection_vector + normalize forces vector_compose='outer' internally (§6 directive) — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeSingleYConvention::test_NSY_1_single_y_default_inner_works",
            "test_normalize.py::TestNormalizeSingleYConvention::test_NSY_2_single_y_explicit_outer_equivalent",
        ],
    },
    {
        "id": "NORMALIZE.backward_compat",
        "name": "normalize=None preserves pre-Phase-13.33 behavior bit-identical — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeBackwardCompat::test_NBC_1_default_normalize_none_unchanged",
        ],
    },
    {
        "id": "NORMALIZE.validation",
        "name": "normalize input validation: wrong vector count, invalid mode, same=True conflict — Phase 13.33 M1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeValidation::test_NV_1_wrong_vector_count_raises",
            "test_normalize.py::TestNormalizeValidation::test_NV_2_invalid_mode_string_raises",
            "test_normalize.py::TestNormalizeValidation::test_NV_3_same_true_with_normalize_raises",
        ],
    },
    {
        "id": "NORMALIZE.group_by_compose",
        "name": "group_by + normalize: per-group differential rendering — Phase 13.33 M2",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeGroupBy::test_NG_1_basic_group_by_3_fills",
            "test_normalize.py::TestNormalizeGroupBy::test_NG_2_per_group_delta_recovers_offset",
            "test_normalize.py::TestNormalizeGroupBy::test_NG_3_stats_dict_grouped_structure",
        ],
    },
    {
        "id": "NORMALIZE.facet_by_compose",
        "name": "facet_by + normalize: K x 2 grid with per-facet independent differential; facet_by_bins/_quantiles raises (NF.3 workaround-hint lock) — Phase 13.33 M2 + FIX1",
        "category": "NORMALIZE",
        "tests": [
            "test_normalize.py::TestNormalizeFacetBy::test_NF_1_k_by_2_grid",
            "test_normalize.py::TestNormalizeFacetBy::test_NF_2_per_facet_independent_computation",
            "test_normalize.py::TestNormalizeFacetBy::test_NF_3_facet_by_bins_with_normalize_raises_with_workaround_hint",
        ],
    },

    # ── Phase 13.34.DF M2 — Robustness gap invariance tests ──

    {
        "id": "ROBUSTNESS.median_mad_sigma",
        "name": "central='median' must use MAD-sigma error bars (Phase 13.33 CRR §11 pre-existing inconsistency lock; xfail until source-side fix) — Phase 13.34 M2",
        "category": "ROBUSTNESS",
        "tests": [
            "test_phase_13_34_df_m2_robustness.py::TestMedianMADSigma::test_MED_1_median_uses_mad_sigma_for_errors",
        ],
    },
    {
        "id": "ROBUSTNESS.stats_schema",
        "name": "Stats dict key contract per plot kind — locks against silent renames breaking ADF/RootInteractive — Phase 13.34 M2",
        "category": "ROBUSTNESS",
        "tests": [
            "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_profile_keys_present",
            "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_normalize_keys_single_curve",
            "test_phase_13_34_df_m2_robustness.py::TestStatsDictSchema::test_STATS_normalize_keys_grouped",
        ],
    },
    {
        "id": "ROBUSTNESS.kwarg_composition",
        "name": "Feature interaction tests — would have caught BUG-001/002/003 at delivery; locks 5 known kwarg interaction pairs — Phase 13.34 M2",
        "category": "ROBUSTNESS",
        "tests": [
            "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_auto_title",
            "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_subplot_titles",
            "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_facet_bins_with_group_by_composition",
            "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_normalize_with_facet_by_column_mode",
            "test_phase_13_34_df_m2_robustness.py::TestKwargComposition::test_X_channel_mode_facet_with_auto_title",
        ],
    },

    # ── Phase 13.37.DF — Histogram Robustness (BUG-014/015/016 + hist_errors + linestyle_cycle) ──

    {
        "id": "HIST.step_per_group_color",
        "name": "histtype='step' renders distinct per-group edgecolors (BUG-014 closed; extends Phase 13.36 sentinel to edgecolor) — Phase 13.37.DF",
        "category": "HIST",
        "tests": [
            "test_phase_13_37_df_hist_robustness.py::TestBUG014StepColor::test_STEP1_default_per_group_step_colors",
            "test_phase_13_37_df_hist_robustness.py::TestBUG014StepColor::test_STEP2_bar_histtype_unchanged",
            "test_phase_13_37_df_hist_robustness.py::TestBUG014StepColor::test_STEP3_user_edgecolor_wins_over_step_cycle",
            "test_phase_13_37_df_hist_robustness.py::TestBUG014StepColor::test_STEP4_stepfilled_edgecolor_untouched",
        ],
    },
    {
        "id": "PROFILE.float_group_by_guard",
        "name": "profile() float group_by with no bins + nunique>20 raises ValueError with group_by_bins=N guidance (BUG-015; mirrors Phase 13.35 hist BUG-012) — Phase 13.37.DF",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_37_df_hist_robustness.py::TestBUG015ProfileGuard::test_GUARD1_float_no_bins_raises",
            "test_phase_13_37_df_hist_robustness.py::TestBUG015ProfileGuard::test_GUARD2_with_bins_no_error",
        ],
    },
    {
        "id": "HIST.interval_sort_numeric",
        "name": "pd.Interval group legend sorted by numeric .left (BUG-016; hasattr-guard extension of _interval_sort_key) — Phase 13.37.DF",
        "category": "HIST",
        "tests": [
            "test_phase_13_37_df_hist_robustness.py::TestBUG016IntervalSort::test_SORT1_bins_crossing_10_numeric_order",
            "test_phase_13_37_df_hist_robustness.py::TestBUG016IntervalSort::test_SORT2_positive_control_lexicographic_would_fail",
            "test_phase_13_37_df_hist_robustness.py::TestBUG016IntervalSort::test_SORT3_categorical_string_groups_unaffected",
        ],
    },
    {
        "id": "HIST.hist_errors",
        "name": "Poisson error bar overlay (hist_errors=True): √n raw / √n/N probability / √n/(N·bw_i) density (per-bin); weighted Poisson via Σw²; zero-bin masking; ungrouped+bins=int safe; Phase 13.36 color sentinel preserved — Phase 13.37.DF",
        "category": "HIST",
        "tests": [
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE1_unnormalized_sqrt_n",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE2_probability_norm",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE3_density_equal_width",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE4_default_no_errorbars",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE5_zero_count_bins_skipped",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE6_color_override_phase_13_36",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE7_min_entries_filters_errorbars",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE8_density_variable_bin_widths",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HE9_ungrouped_bins_int_no_crash",
            "test_phase_13_37_df_hist_robustness.py::TestHistErrors::test_HEstyle_keys_registered",
        ],
    },
    {
        "id": "PROFILE_HIST.linestyle_cycle",
        "name": "linestyle_cycle=True cycles per-group linestyles from channels.cycles.linestyle on profile() and hist(); user-explicit linestyle= wins via _ud_user_linestyle sentinel (extends Phase 13.36 Edit 17 pattern) — Phase 13.37.DF",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_37_df_hist_robustness.py::TestLinestyleCycle::test_LC1_profile_distinct_linestyles",
            "test_phase_13_37_df_hist_robustness.py::TestLinestyleCycle::test_LC2_default_unchanged",
            "test_phase_13_37_df_hist_robustness.py::TestLinestyleCycle::test_LC3_user_linestyle_wins_over_cycle",
            "test_phase_13_37_df_hist_robustness.py::TestLinestyleCycle::test_LC4_same_true_overlay_distinct_linestyles",
            "test_phase_13_37_df_hist_robustness.py::TestLinestyleCycle::test_LC5_hist_step_per_group_linestyle",
        ],
    },
    # ====================================================================== #
    # Phase 13.38.DF — Scatter enhancements: BUG-017 + xerr/yerr + expression #
    # ====================================================================== #
    {
        "id": "FACET.float_facet_by_guard",
        "name": "_dispatch_faceted_render() float facet_by + no bins + nunique>20 raises ValueError with facet_by_bins=N / facet_by_quantiles=N guidance (BUG-017; third instance of BUG-012/BUG-015 float-guard class) — Phase 13.38.DF",
        "category": "FACET",
        "tests": [
            "test_phase_13_38_df_scatter_enhancements.py::TestFacetByFloatGuard::test_FBGUARD_1_float_facet_by_no_bins_raises",
            "test_phase_13_38_df_scatter_enhancements.py::TestFacetByFloatGuard::test_FBGUARD_2_float_facet_by_with_bins_no_error",
        ],
    },
    {
        "id": "SCATTER.xerr_yerr",
        "name": "scatter() xerr/yerr from column name or df.eval() expression. Render via ax.errorbar() when either provided; ax.scatter() otherwise (dispatch invariance). Three-tier NaN policy: raise on 100%, warn at >50%, silent zeroing at ≤50%. nanfrac in stats dict. Style keys: scatter.error_capsize=2, scatter.error_elinewidth=1.0, scatter.error_ecolor=None — Phase 13.38.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_1_yerr_column_extents_match",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_2_xerr_column_extents_match",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_3_both_xerr_yerr_simultaneously",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_4_yerr_dfeval_expression",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_5_default_dispatch_invariance",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_6_nan_policy_three_tiers",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_7_style_keys_registered",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_8_group_by_plus_yerr",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterErrorBars::test_SE_9_xerr_plus_facet_by",
        ],
    },
    {
        "id": "SCATTER.expression_color",
        "name": "scatter() color= accepts df.eval() expression (e.g. color='abs(tgl)'). _process_color() dispatch reordered (CP0-1): None → array → column → fixed-color (to_rgba) → df.eval → terminal. Column-name check precedes to_rgba() to preserve backward compat for columns named after matplotlib colors ('b', 'r', 'k') — Phase 13.38.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_1_expression_color_colormap_applied",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_2_column_color_byte_identical_backward_compat",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_3_invalid_expression_actionable_error",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_6_column_name_collision_with_named_color",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionColor::test_ECM_7_expression_color_plus_group_by_behavior_locked",
        ],
    },
    {
        "id": "SCATTER.expression_marker",
        "name": "scatter() marker= accepts boolean df.eval() expression (e.g. marker='ncl > 100'); True → 's', False → 'o'. Per-point rendering via np.unique loop with label='_nolegend_' (no spurious legend entries) — Phase 13.38.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_4_boolean_marker_two_marker_encoding",
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_8_per_point_marker_legend_no_duplicates",
        ],
    },
    {
        "id": "SCATTER.expr_compose",
        "name": "scatter() expression color + expression marker composition: both encodings simultaneously on single-path scatter. Each marker subgroup carries its own colormap array — Phase 13.38.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_38_df_scatter_enhancements.py::TestScatterExpressionMarker::test_ECM_5_expression_color_plus_expression_marker_compose",
        ],
    },
    # ── Phase 13.39.DF: 2D Profile + Time Axis + Scatter3D ──
    {
        "id": "PROFILE.profile2d",
        "name": "profile('z:y:x') → 2D mean heatmap via scipy.stats.binned_statistic_2d + ax.pcolormesh. Supports bins=[nx,ny] or bins=nx+bins2=ny, min_entries_2d=N masking, norm='log', colorbar+clabel. Dispatch in DFDraw.profile() at colon_count==2 (CP1-5: after _apply_selection/_apply_sampling, before _parse_expr). z/y/x accept column names or df.eval() expressions. Backward compat: 'y:x' (colon_count==1) unchanged — Phase 13.39.DF",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_1_quadmesh_rendered",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_2_per_cell_mean_correctness",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_3_min_entries_masks_low_count_cells",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_4_dfeval_expression_for_z",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_5_bins_list_vs_bins2_shape_invariance",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_6_colorbar_labeled_single_key",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_7_selection_applied_before_binning",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_8_backward_compat_1d_profile",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestProfile2D::test_P2D_10_group_by_raises_value_error",
        ],
    },
    {
        "id": "PROFILE.time_axis",
        "name": "profile() time_format= pre-conversion: x_data converted to matplotlib date numbers via mdates.date2num() before binning. CP1-4 auto-detect: datetime64 column dtype detected BEFORE astype(float) (else int64-nanosecond cast becomes ~1.7e15 → pd.to_datetime crashes 'year out of range'). DateFormatter / AutoDateFormatter applied post-render — Phase 13.39.DF",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_1_profile_time_format_pct_HM",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_2_profile_time_format_auto",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_3_default_no_time_format_backward_compat",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_6_profile2d_x_axis_date_formatter",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_7_datetime64_column_no_crash",
        ],
    },
    {
        "id": "HIST.time_axis",
        "name": "hist() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.hist(). Post-hoc rewrite would be no-op for Patches (lesson from Phase 13.39 v1.0 P1). CP1-1 regression-lock: §9.TA.5 uses realistic timestamps (~1.7e9), as epoch-0 made both pre-conv (0.0) and raw (0) paths pass — Phase 13.39.DF",
        "category": "HIST",
        "tests": [
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_5_hist_time_format_realistic_timestamps",
        ],
    },
    {
        "id": "SCATTER.time_axis",
        "name": "scatter() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.scatter/ax.errorbar — Phase 13.39.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_39_df_profile2d_timeaxis.py::TestTimeAxis::test_TA_4_scatter_time_format",
        ],
    },
    {
        "id": "SCATTER.scatter3d",
        "name": "draw('z:y:x', type='scatter3d') → 3D point cloud via mpl_toolkits.mplot3d. Reuses Phase 13.38 _process_color() + _process_size() unchanged. color=/size= accept column names or df.eval() expressions. elev=/azim= for ax.view_init(). Stats dict locks mean_x AND mean_y AND mean_z to 1e-9 (CP1-3). Scope boundaries: group_by + scatter3d raises (CP2-1); same=True onto non-3D axes raises (CP2-2). 'y:x' (colon!=2) with type='scatter3d' raises with actionable message — Phase 13.39.DF",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_1_basic_render",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_2_color_expression",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_3_size_column",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_4_selection_reduces_count",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_5_two_variable_expr_raises",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_6_stats_dict_locks_all_three_means",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_7_group_by_raises_value_error",
            "test_phase_13_39_df_profile2d_timeaxis.py::TestScatter3D::test_SC3D_8_same_true_non_3d_axes_raises",
        ],
    },
    # ── Phase 13.40.DF: Cumulative Histogram ──
    {
        "id": "HIST.cumulative",
        "name": "hist() cumulative=True/-1/False — ROOT TH1::Draw('cumulative') equivalent. Three values: True (ascending CDF/ECDF), False (default, byte-identical backward compat), -1 (descending/survival, ROOT convention). matplotlib native cumulative= forwarded explicitly at 4 internal call sites (Phase 13.39 §2.2 lesson applied recursively: DFDraw.hist → draw_hist → _draw_hist_grouped → ax.hist; ALSO through _dispatch_faceted_render for facet_by composition). Composes with: norm='probability' (→ ECDF 0-1), group_by overlaid (per-group ECDFs), group_by stacked (CP2-1 regression lock for 3rd call site), facet_by (per-facet cumulative), histtype='step' (HEP-standard step ECDF). Correctness guard (M5): hist_errors+cumulative → NotImplementedError (Poisson per-bin errors are independent; cumulative counts are correlated). Vector dispatch [x,y] propagates cumulative correctly (Phase 13.16.DF FIX1 bug class lock) — Phase 13.40.DF",
        "category": "HIST",
        "tests": [
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_1_monotone_and_total_N_lock",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_2_ECDF_last_value_is_one",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_3_survival_starts_at_one",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_4_cumulative_false_backward_compat",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_5_group_by_per_group_ecdf",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_6_hist_errors_plus_cumulative_raises",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_7_vector_dispatch_propagates_cumulative",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_8_facet_by_per_facet_ecdf",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_9_histtype_step_plus_cumulative_polygon_safe",
            "test_phase_13_40_df_cumulative_hist.py::TestCumulativeHist::test_CH_10_group_by_stacked_cumulative_regression_lock",
        ],
    },
    # ── Phase 13.41.DF: N-D Faceting via facet_by=List[str] ──
    {
        "id": "FACET.list_grid",
        "name": "facet_by accepts Union[str, List[str]] for 1D/2D/3D faceting. Convention LOCKED matching numpy/pandas (n_rows, n_cols, ...) shape: facet_by[0]=ROW (vertical within figure), facet_by[1]=COLUMN (horizontal within figure), facet_by[2]=FIGID (separate figures, one per value). facet_by[3+] raises NotImplementedError. 3D returns (List[Figure], List[axes_2d], List[stats_dict]) — DEVIATES from standard (fig, ax, stats) contract; documented prominently in inline help. New params: share_x/share_y ∈ {'all','row','col','none'} (within-figure axis sharing), share_across_figures: bool (3D global range lock). Per-plot-kind lock for share_across_figures (CP1-2): scatter locks x AND y; hist/profile locks x only (y auto-scales per figure to handle sparse-figID variance). New helpers: _normalize_facet_args, _to_mpl_share (symmetric {'all':True,'row':'row','col':'col','none':False} — v1.2 CP0-1 fix for Hard Constraint #3), _validate_share_axis_value, _resolve_facet_values (discrete or pd.cut/qcut Interval), _filter_facet_value (CP1-3 discrete vs binned), _compute_global_ranges. dfdraw is FIRST major plotting library with unified API where Nth faceting dimension generates separate figures (seaborn/ggplot2/plotly/altair all require manual loops). _validate_facet_by_binning guard for list input (v1.3 P1-A). Per-plot-kind dispatch: hist uses range= (matplotlib convention); profile uses range= which DFDraw.profile remaps to draw_profile's x_range= internally; scatter uses ax.set_xlim/set_ylim post-draw (no native range params); hist also locks ax.set_xlim post-draw (range= only locks bins, not axis xlim). Empty cell handling: '(no data)' diagnostic + stats={'n':0,'empty':True} — Phase 13.41.DF",
        "category": "FACET",
        "tests": [
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_1_string_equals_list_of_one",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_2_2d_grid_rows_by_cols",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_3_2d_bins_per_dim",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_4_2d_quantiles_per_dim",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_5_mixed_bins_None_and_int",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_6_convention_lock_2d_row_col",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_7_length_mismatch_raises",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_8_group_by_inside_cells",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_9_cumulative_per_cell",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_10_four_dimensions_raises",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_11_convention_lock_3d_list_of_figures",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_12_share_x_row_regression_lock",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_13_share_across_figures_3d_scatter",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_14_share_none_data_divergence",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_15_empty_cell_no_crash",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_16_3d_share_across_hist_dispatch",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_17_3d_share_across_profile_dispatch",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_18_share_across_figures_false_independence",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_19_share_x_col_symmetry",
            # Phase 13.41.DF FIX1 (3 additional regression locks)
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_20_share_y_row_symmetry",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_21_share_x_invalid_raises",
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_22_auto_title_suptitle_lock",
            # Phase 13.41.DF FIX2 (1 additional regression lock)
            "test_phase_13_41_df_2d_faceting.py::TestFacetByListGrid::test_FBY_23_3d_auto_title_combined_suptitle",
        ],
    },

    # ── Phase 13.42.DF: Inline fits ──

    {
        "id": "FIT.inline",
        "name": "Inline fits (fit= parameter on hist/profile/scatter/draw)",
        "category": "FIT",
        "tests": [
            # Numerical correctness (TestFitsNumericalCorrectness — includes f23)
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f1_gauss_recovers_center_sigma",
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f2_pol2_recovers_coefficients",
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f3_user_callable_with_initial",
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f4_pol1_alias_equals_linear",
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f5_gauss_registry_heuristic_converges",
            "test_phase_13_42_df_inline_fits.py::TestFitsNumericalCorrectness::test_f23_user_guess_callable",
            # Spec normalization (actual implemented names per source)
            "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f6_str_shorthand_equiv_to_dict",
            "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f7_callable_shorthand_equiv_to_dict",
            "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f8_length1_list_broadcasts",
            "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f9_length_match_pair_per_channel",
            "test_phase_13_42_df_inline_fits.py::TestFitsSpecNormalization::test_f10_length_mismatch_raises",
            # Composition (vector / group_by / facet_by)
            "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f11_vector_expr_scalar_fit_broadcast",
            "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f12_vector_expr_vector_fit_pair",
            "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f13_compound_on_single_curve",
            "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f14_group_by_list_of_lists_per_group",
            "test_phase_13_42_df_inline_fits.py::TestFitsComposition::test_f15_facet_by_2d_tuple_keys",
            # Dict-key parsing
            "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f16_range_restricts_fit_domain",
            "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f17_bounds_constrain_params",
            "test_phase_13_42_df_inline_fits.py::TestFitsDictKeyParsing::test_f18_unknown_dict_key_raises",
            # Display rendering
            "test_phase_13_42_df_inline_fits.py::TestFitsDisplayRendering::test_f19_show_params_false_hides_one_block",
            "test_phase_13_42_df_inline_fits.py::TestFitsDisplayRendering::test_f20_multi_fit_distinct_linestyles",
            # Failure handling (only f21, f22; f23 lives in NumericalCorrectness)
            "test_phase_13_42_df_inline_fits.py::TestFitsFailureHandling::test_f21_default_failure_does_not_raise",
            "test_phase_13_42_df_inline_fits.py::TestFitsFailureHandling::test_f22_raise_on_failure_true_raises",
            # Panel closures
            "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f24_register_fit_public_export",
            "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f25_groupby_facet_fit_deep_composition",
            "test_phase_13_42_df_inline_fits.py::TestFitsPanelClosures::test_f26_normalize_plus_fit_silent_consume",
            # Phase 13.42.DF P1-B regression (Sonnet54 finding, CRR §2 D7)
            "test_phase_13_42_df_inline_fits.py::TestFitsP1BProfileGroupedRegression::test_f27_profile_group_by_fit_returns_dict_keyed_by_group",
            # Phase 13.42.DF FIX1 production-gate regressions (B1-B7, D5/D8/D9)
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f28_grouped_fit_quantile_binning",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f28b_skipped_empty_does_not_crash",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f29_hist_fit_redchi_physically_correct",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f30_set_style_fit_textbox_fontsize_facet",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f31_vector_fit_pairing",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f32_stacked_hist_per_group_fits",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f33_fit_textbox_kwargs_fontsize_override",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX1Regressions::test_f33_fit_textbox_kwargs_precedence_over_style",
            # Phase 13.42.DF FIX2 regressions (close items deferred at FIX1 close)
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f59_suptitle_top_for_title_helper",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f60_facet_fit_no_crash",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f61_weighted_hist_fit_userwarning",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f61_unweighted_hist_fit_no_warning",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f62_stacked_selection_vector_fit_raises",
            "test_phase_13_42_df_inline_fits.py::TestPhase1342FIX2Regressions::test_f63_fit_textbox_kwargs_signature_and_forwarded_names",
        ],
    },
    # ========================================================================
    # Phase 13.43.DF — Summary Fit (standalone figures for fit results)
    # ========================================================================
    {
        "id": "FIT.summary",
        "name": "Summary fit — standalone table + params figure",
        "category": "FIT",
        "tests": [
            # Basic scenarios A/B/C/D
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f34_summary_fit_table",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f35_summary_fit_figure",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f36_summary_fit_both",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f37_summary_fit_omitted_is_scenario_a",
            # Row count + composition
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f38_table_row_count",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f38a_selection_vector_composition",
            # Params figure layout
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f39_params_figure_auto_layout",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f40_overlay_mode",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f41_annotate_mode",
            # Scenario E edge cases
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f42_no_group_no_facet_figure_is_scenario_e",
            # Precision
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f43_precision_in_table_cells",
            # Faceted composition
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f44_2d_facet_composition",
            # Composition with other modes
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f45_normalize_plus_summary_fit_is_scenario_e",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f46_quantile_band_profile_is_scenario_e",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f47_cumulative_hist",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f47b_stacked_grouped_fit",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f48_summary_fit_without_fit",
            # Title
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f49_auto_title_content",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f50_title_overflow_truncate",
            # Data format
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f51_default_data_format_is_list_dict",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f52_pandas_via_style_key",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f53_per_call_data_format",
            # same=True Replace mode
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f54_same_true_replace_mode",
            # Placement invariants
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f55_placement_invariants",
            # Faceted aggregation §4.2.0
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56_faceted_aggregation_shape3",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56b_faceted_no_group_by",
            "test_phase_13_43.py::TestPhase1343SummaryFit::test_f56c_draw_scalar_forwards_fit_and_summary_fit",
        ],
    },
    {
        "id": "FIT.root_aliases",
        "name": "ROOT-convention aliases (fit='gaus', type='histo')",
        "category": "FIT",
        "tests": [
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f64_gaus_root_alias",
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f65_histo_type_alias",
        ],
    },
    {
        # Phase 13.50.DF — Display-name map (render-only): canonical names in
        # plots/fits.py (slope, intercept, amplitude, center, sigma, decay) are
        # unchanged; renderer consults _DISPLAY_NAMES in plots/_fit_render.py
        # to produce short (p0/p1/A) or Greek-mathtext ($\mu$/$\sigma$/$\tau$)
        # display strings. Layer: visual_primitive (F1-F3 added in step 1;
        # full set F1-F6 will land by step 3 once fit_textbox_kwargs is
        # extended with the rename_params override sub-key).
        "id": "FIT.display_names",
        "name": "Display-name map for fit parameters (render-only short/Greek names)",
        "category": "FIT",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F1_linear_short_names",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F2_gauss_mu_mathtext",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitDisplayNames::test_F3_exponential_tau_mathtext",
        ],
    },
    {
        # Phase 13.50.DF step 2 — Precision keys replace single fit.text_format
        # ([BREACH] under Coder QRC R14; architect approval in v2.5 §3.3).
        # New style keys: fit.value_format='.2g', fit.error_format='.1g',
        # fit.precision_mode=None|'physics'|'uniform'. _format_value_error_pair
        # helper in plots/_fit_render.py performs the physics-mode alignment.
        # Layer: visual_primitive. F16 (per-call override via
        # fit_textbox_kwargs={'value_format': ...}) will be added in step 3
        # and will live on FIT.textbox_kwargs_extensions per panel decision.
        "id": "FIT.precision_modes",
        "name": "Separate value/error precision keys + physics alignment mode",
        "category": "FIT",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitPrecision::test_F4_default_precision",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitPrecision::test_F5_precision_mode_physics",
        ],
    },
    {
        # Phase 13.50.DF step 3 — fit_textbox_kwargs gets four new sub-keys:
        # rename_params (dict per-call override of _DISPLAY_NAMES),
        # value_format / error_format (per-call format-spec overrides of the
        # style defaults), precision_mode (per-call physics/uniform/None).
        # _allowed_sub_keys in plots/_fit_render.py extended; per-call values
        # take precedence over style defaults (same precedence as the existing
        # 'fontsize' sub-key from Phase 13.42 FIX1). Layer: visual_primitive.
        # F16 (value_format override) lives here per panel P2-6 decision in v2.4
        # review — per-call override tests belong with the kwarg surface they
        # test, not with the underlying style key.
        "id": "FIT.textbox_kwargs_extensions",
        "name": "fit_textbox_kwargs extensions: rename_params / value_format / error_format / precision_mode",
        "category": "FIT",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitTextboxKwargsExtensions::test_F6_rename_params_overrides_display_map",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350FitTextboxKwargsExtensions::test_F16_value_format_per_call_override",
        ],
    },
    {
        # Phase 13.50.DF step 4 — legend= polymorphic kwarg + show_legend= bool
        # parallel kwarg. Both Pattern A (popped at top-level dispatcher entry
        # in plots/_legend.py:_normalize_legend_spec, never in
        # _*_FORWARDED_NAMES). The applier (_apply_legend_mode) is hooked at
        # the main return of hist/scatter/profile in drawer.py.
        #
        # show_legend= is technically NEW at the public dispatcher level
        # (R16 verified: previously only on add_reference_overlay helper);
        # the v2.5 proposal's "back-compat parallel" framing meant
        # "alongside legend=", not "preserves existing surface".
        #
        # Mixed-layer feature row per v2.4 P2-NEW-1 panel decision:
        # F7/F8/F9/F10 are visual_primitive (renderer-free fig.legends /
        # ax.get_legend() inspection); F18 + normalizer idempotency are
        # invariance (no fig needed for the normalizer; F18 compares two
        # legend_topology 4-tuples).
        "id": "LEGEND.modes",
        "name": "legend= polymorphic kwarg (bool|str|dict) + show_legend= bool parallel + four modes (all|none|shared|first)",
        "category": "LEGEND",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F7_shared_one_fig_zero_per_axes",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F8_first_only_axes_0_kept",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F9_false_no_legends",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F10_dict_forwards_loc_and_ncol",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendModes::test_F18_show_legend_legend_behavioral_equivalence",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350LegendNormalizerInvariance::test_normalize_legend_spec_idempotent",
        ],
    },
    {
        # Phase 13.50.DF step 5 — summary_fit.placement axis: where the
        # summary_fit content is rendered.
        #   'figure'    → separate matplotlib Figure (default; Phase 13.43
        #                 behavior preserved exactly)
        #   'subfigure' → SubFigure inside the main fig
        #   'pad'       → extra GridSpec row inside the main fig (single axes)
        #
        # GridSpec is immutable post-creation (v2.3 panel P1-A), so the
        # dispatcher MUST pre-plan the slot before plt.subplots(). The
        # pre-planning intercept is at the top of _dispatch_faceted_render;
        # the slot SubplotSpec is stashed on fig._dfdraw_summary_fit_slot
        # for _maybe_attach_summary_fit to consume.
        #
        # Mixed-layer feature row: F11/F12/F13 are visual_primitive
        # (renderer-free placement_topology inspection); F17 is invariance
        # (keyed-dict comparison across all 3 placements per v2.4 P2-NEW-1
        # — count-tuple insufficient because 'pad' and 'subfigure' both
        # produce one host axes; keyed dict catches the mode-swap bug).
        #
        # Non-faceted callers: placement='pad'/'subfigure' currently
        # requires a faceted dispatch path (group_by or facet_by). Calls
        # without faceting raise ValueError with a clear fix message at
        # _maybe_attach_summary_fit time. Broader N-D faceting + non-
        # faceted support is deferred to Phase 13.50 FIX1 if requested.
        "id": "SUMMARY_FIT.placement",
        "name": "summary_fit.placement axis: figure (default) / subfigure / pad with GridSpec pre-planning",
        "category": "SUMMARY_FIT",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F11_placement_figure_default_unchanged",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F12_placement_subfigure_per_panel_inset_table",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F13_placement_pad_uses_extra_row_axes",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitPlacement::test_F17_placement_variants_produce_equivalent_tables",
        ],
    },
    {
        # Phase 13.50.DF step 6 — summary_fit.orientation axis: table layout
        # direction. Default 'row' (one row per fit, columns = id keys + params)
        # preserves the Phase 13.43 table shape; 'column' transposes (one row
        # per id key + param, columns = fits). Honored by BOTH the in-slot
        # renderer (_render_table_in_axes — placements 'pad' and 'subfigure')
        # AND the Phase 13.43 'figure' placement renderer (_render_table_figure).
        # Phase 13.50.DF step 7b spec-conformance: token rename
        # ('horizontal'/'vertical' → 'row'/'column') per v2.5 §3.4(b) canonical
        # naming. Step 7b R1: orientation extended to the 'figure' placement
        # path; v2.5 §3.4(b) put no placement restriction on orientation, the
        # original step-6 limitation to slot-renderer was an undisclosed partial.
        # Layer: visual_primitive — F14/F15 inspect the matplotlib Table
        # object's cell shape via table.get_celld(), no rasterization needed.
        "id": "SUMMARY_FIT.orientation",
        "name": "summary_fit.orientation axis: row (default) / column (transpose) — honored by all placements (figure, pad, subfigure)",
        "category": "SUMMARY_FIT",
        "tests": [
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitOrientation::test_F14_orientation_row_default_unchanged",
            "test_phase_13_50_df_fit_visual.py::TestPhase1350SummaryFitOrientation::test_F15_orientation_column_transposes_shape",
        ],
    },
    {
        "id": "API.kwarg_typo_guard",
        "name": "Kwarg-typo guard (difflib did-you-mean at draw() entry)",
        "category": "API",
        "tests": [
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f66_kwarg_typo_guard",
        ],
    },
    {
        "id": "RANGE.scatter",
        "name": "range= on scatter via shared 2D resolver",
        "category": "RANGE",
        "tests": [
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f67_scatter_range_minmax_nonfaceted",
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f68_scatter_range_minmax_faceted",
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f69a_scatter_range_strategy_parity",
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f69b_profile_hist_range_minmax_no_unpack_error",
        ],
    },
    {
        "id": "RANGE.scatter_filter",
        "name": "range= on scatter removes out-of-range points (FIX1 semantic)",
        "category": "RANGE",
        "tests": [
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f71_scatter_range_removes_out_of_range_points",
        ],
    },
    {
        "id": "TITLE.get_suptitle",
        "name": "_get_suptitle public-API helper (mpl >= 3.8 + fallback)",
        "category": "TITLE",
        "tests": [
            "test_phase_13_46_df_audit_fixes.py::TestPhase1346AuditFixes::test_f70_get_suptitle_live_path",
        ],
    },
    # ── Phase 13.48.DF — Tier-1 automated visual checks (visual_primitive) ──
    {
        "id": "VISUAL.data_bounds",
        "name": "plotted points lie within axes limits (C-9 regression lock)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v1_scatter_points_in_bounds",
        ],
    },
    {
        "id": "VISUAL.cell_population",
        "name": "every visible facet cell drew data; ragged-grid padding excluded",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v2_every_visible_cell_nonempty",
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v2_padding_safe_on_ragged_grid",
        ],
    },
    {
        "id": "VISUAL.artist_count",
        "name": "per-cell data-series / legend count matches per-cell filtered groups",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v3_artist_count_matches_groups",
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v7_legend_entries_match_groups",
        ],
    },
    {
        "id": "VISUAL.color_distinct",
        "name": "per-group colors distinct — color-cycle not reset (AD-37 class)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v4_per_group_colors_distinct",
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v8_color_cycle_distinct",
        ],
    },
    {
        "id": "VISUAL.facet_grid",
        "name": "facet grid shape + shared-axis consistency (figure-derived)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v5_facet_grid_shape",
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v6_shared_axis_consistency",
        ],
    },
    {
        "id": "VISUAL.title",
        "name": "suptitle populated, not duplicated per-cell (C-3); content (I-4)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v9_auto_title_not_duplicated",
            "test_phase_13_48_df_visual_testing.py::TestPhase1348VisualPrimitive::test_v10_title_content",
        ],
    },
    # ── Phase 13.51.DF — Audit fix feature registrations ──
    # The 23 tests in tests/test_phase_13_51_post_audit.py landed in-tree at
    # `b1db4740` (v1.5) + `2235caef` (FIX1). They cover post-audit fixes:
    # R-2 explicit forwarding at all 4 draw() dispatch branches, central=
    # median routing, datetime64 guard at compute_autorange entry, and
    # hist2d time_format= symmetry. This block registers them as feature
    # claims so CAPABILITY_MATRIX gates 4 (Verified ≥64) and 7
    # (VISUAL.* ≥9) close ahead of distribution to the 5 audiences.
    # No new tests — registration-only pass.
    {
        "id": "DRAW.R2_forwarding",
        "name": "draw() scalar dispatch forwards Phase 13.27–13.41 named params (selection_vector, weights_vector, share_*, facet_by, etc.) to all 4 typed methods — A≡B with direct calls",
        "category": "DRAW",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T3_draw_type_profile_selection_vector_matches_direct",
            "test_phase_13_51_post_audit.py::test_T5_draw_type_hist_selection_vector_matches_direct",
        ],
    },
    {
        "id": "PROFILE.central_median_1d",
        "name": "profile() central='median' renders the median line on the 1D non-grouped path (Phase 13.51 V-3 fix at profile.py:879; bin_means → _central_values)",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T9b_profile_1d_central_median_line_differs_from_mean",
        ],
    },
    {
        "id": "PROFILE2D.central_median_mesh",
        "name": "profile2d() central='median' produces a mesh that differs from mean (Phase 13.51 R-2: conditional central= forward into draw_profile2d)",
        "category": "PROFILE2D",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T9a_profile2d_central_median_mesh_differs_from_mean",
        ],
    },
    {
        "id": "PROFILE.central_median_fit",
        "name": "profile() central='median', fit=… — fit center reflects median data, not mean (Phase 13.51 P1-B fix at profile.py:907 fit curve _central_values)",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T9c_profile_fit_central_median_fit_center_differs",
        ],
    },
    {
        "id": "AUTORANGE.datetime64_guard",
        "name": "compute_autorange() handles datetime64 input across robust_3mad / percentile_99 (and by symmetry all 5) strategies without crashing (Phase 13.51 V-2 guard at _autorange.py entry)",
        "category": "AUTORANGE",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T10b_compute_autorange_datetime64_robust_3mad",
            "test_phase_13_51_post_audit.py::test_T10c_compute_autorange_datetime64_percentile_99",
        ],
    },
    {
        "id": "VISUAL.facet_r2_profile",
        "name": "draw(type='profile', facet_by=) populates every panel — R-2 forwarding visual check",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T2_draw_type_profile_facet_by_sec_populated_panels",
        ],
    },
    {
        "id": "VISUAL.facet_r2_hist",
        "name": "draw(type='hist', facet_by=) populates every panel — R-2 forwarding visual check",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T4_draw_type_hist_facet_by_sec_populated_panels",
        ],
    },
    {
        "id": "VISUAL.facet_r2_scatter",
        "name": "draw(type='scatter', facet_by=) populates every panel — R-2 forwarding visual check",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T6_draw_type_scatter_facet_by_sec_populated_panels",
        ],
    },
    {
        "id": "VISUAL.facet_r2_hist2d",
        "name": "draw(type='hist2d', facet_by=) populates every panel — R-2 forwarding visual check (Phase 13.51 S-3 closure)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T8_draw_type_hist2d_facet_by_sec_populated_panels",
        ],
    },
    {
        "id": "VISUAL.hist2d_datetime_labels",
        "name": "hist2d() time_format='auto' renders readable date tick labels on the appropriate axis (Phase 13.51 S-8 symmetry with hist/scatter/profile)",
        "category": "VISUAL",
        "tests": [
            "test_phase_13_51_post_audit.py::test_T14_hist2d_time_format_auto_date_tick_labels",
        ],
    },
    # ── Phase 13.52.DF — Declarative Overlay Capability ──
    {
        "id": "OVERLAY.engine_layers_form",
        "name": "overlay() engine with explicit layers=[...] (density base + overlays on one shared axes)",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T1_overlay_hist2d_profile_layers_form",
            "test_phase_13_52_df_overlay.py::test_T1h_overlay_hexbin_profile_alternative_density_base",
            "test_phase_13_52_df_overlay.py::test_T2_overlay_quantiles_kwarg_on_profile_layer",
            "test_phase_13_52_df_overlay.py::test_T3_overlay_scatter_overlay_onto_hist2d",
        ],
    },
    {
        "id": "OVERLAY.engine_guards",
        "name": "engine ValueError guards (empty, non-method type, non-allowed type, >1 density, non-density base, 3D) raise BEFORE any draw",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T_D1_overlay_two_density_layers_raises",
            "test_phase_13_52_df_overlay.py::test_T_D2_overlay_3d_scatter3d_layer_raises",
            "test_phase_13_52_df_overlay.py::test_T_D3_overlay_empty_layers_raises",
            "test_phase_13_52_df_overlay.py::test_T_D4_overlay_non_method_type_token_raises",
            "test_phase_13_52_df_overlay.py::test_T_D5_overlay_disallowed_type_outside_whitelist_raises",
            "test_phase_13_52_df_overlay.py::test_T_S5b_string_non_density_base_raises",
            "test_phase_13_52_df_overlay.py::test_T_S4_string_routing_unroutable_kwarg_raises",
        ],
    },
    {
        "id": "OVERLAY.string_sugar_desugar",
        "name": "draw(type='A+B') string sugar desugars to overlay engine with §1.4 routing policy (named-param splice complete per v1.5.1 P1-B)",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T_S1_string_sugar_equals_layers_form",
            "test_phase_13_52_df_overlay.py::test_T_S2_string_routing_bins_to_base_fit_to_profile_negative_control",
            "test_phase_13_52_df_overlay.py::test_T_S2b_string_routing_color_to_scatter_overlay",
            "test_phase_13_52_df_overlay.py::test_T_S3_string_routing_selection_replicates_to_all_layers",
            "test_phase_13_52_df_overlay.py::test_T_S3b_string_routing_selection_vector_replicates_to_accepting_layers",
        ],
    },
    {
        "id": "OVERLAY.faceting_rejected",
        "name": "faceting/share_* params rejected at engine + sugar level (Phase 13.53 deferred)",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T_F_overlay_facet_by_on_layer_raises",
            "test_phase_13_52_df_overlay.py::test_T_S5c_string_facet_by_kwarg_raises",
        ],
    },
    {
        "id": "OVERLAY.range_lock_zorder",
        "name": "post-draw range lock holds after plt.draw(); base-before-overlay z-order",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T4_overlay_range_lock_after_plt_draw",
            "test_phase_13_52_df_overlay.py::test_T5_overlay_z_order_base_before_overlay",
        ],
    },
    {
        "id": "OVERLAY.summary_fit_profile_layer",
        "name": "summary_fit on profile overlay layer renders (Phase 13.51 + 13.52 interaction)",
        "category": "OVERLAY",
        "tests": [
            "test_phase_13_52_df_overlay.py::test_T_S6_overlay_summary_fit_on_profile_layer_renders",
        ],
    },
    # ── Phase 13.54.DF — Gallery-found bug fixes (BUG_dfdraw 20260609/20260610) ──
    # Surfaced by the ADF time_series_draw.py real-data visual gallery per
    # AD-TS-DRAW-001 (gallery as mandatory pre-tag validation). The two bugs
    # were not caught by the existing unit-test suite because the test
    # fixtures used synthetic data shapes (gaussian floats, small datetime64
    # ranges) instead of realistic float64 epoch-seconds.
    {
        "id": "SCATTER.auto_title",
        "name": "DFDraw.scatter() / draw(type='scatter') honors auto_title= without crashing PathCollection.set() — symmetry with hist/profile/scatter3d (BUG_dfdraw_20260609 close)",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_54_df_gallery_fixes.py::test_T1_scatter_direct_auto_title_sets_title",
            "test_phase_13_54_df_gallery_fixes.py::test_T2_draw_type_scatter_auto_title_routes_through_dispatch",
            "test_phase_13_54_df_gallery_fixes.py::test_T3_scatter_no_auto_title_regression_baseline",
        ],
    },
    {
        "id": "HIST2D.time_format_epoch",
        "name": "hist2d() time_format= handles float64 epoch-second timestamps (~1.776e9) without OverflowError; mirrors draw_hist() epoch-second branch (BUG_dfdraw_20260610 close)",
        "category": "HIST2D",
        "tests": [
            "test_phase_13_54_df_gallery_fixes.py::test_T4_hist2d_float_epoch_time_format_no_overflow",
            "test_phase_13_54_df_gallery_fixes.py::test_T5_hist2d_datetime64_time_format_phase_13_51_regression",
            "test_phase_13_54_df_gallery_fixes.py::test_T6_hist2d_non_time_float_no_time_format_else_else_branch",
        ],
    },
    # ── Phase 13.49.DF — meta-tests for the capability matrix itself ──
    {
        "id": "META.capability_matrix",
        "name": "capability matrix integrity (taxonomy resolves; coverage; HTML; no orphan visuals)",
        "category": "META",
        "tests": [
            "test_meta_capability_matrix.py::test_taxonomy_tests_resolve",
            "test_meta_capability_matrix.py::test_classification_coverage",
            "test_meta_capability_matrix.py::test_html_emitter_parseable",
            "test_meta_capability_matrix.py::test_no_orphan_visual_tests",
        ],
    },
]
