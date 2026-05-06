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
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_asymmetric_raises_notimplementederror_phaseb",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_multi_pair_raises_notimplementederror_phaseb",
            "test_quantiles_profile.py::TestQuantileAutoDetection::test_single_value_raises_valueerror",
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
            "test_channel_assignment.py::TestChannelStyleOverride::test_default_style_has_all_10_keys",
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
]
