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
]
