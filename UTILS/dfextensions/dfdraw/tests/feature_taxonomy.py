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
            "test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_style_key_default",
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
]
