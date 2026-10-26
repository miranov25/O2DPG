"""
Feature Taxonomy — dfdraw

Maps feature IDs to their test references.
Used by generate_capability_matrix.py to build the capability matrix.

Format follows GroupByRegression reference implementation.

Phase 13.15.DF — DRAFT for team vote
"""

FEATURES = [
    {
        "id": "PROFILE.range_invariance",
        "name": "Profile per-bin statistics are range-invariant; out-of-range "
                "rows excluded, not folded into edge bins (overflow-fold fix). "
                "Covers mean, std, median, quantile band, and the grouped path.",
        "category": "PROFILE",
        "tests": [
            "test_profile_overflow.py::test_profile_mean_range_invariance",
            "test_profile_overflow.py::test_profile_std_range_invariance",
            "test_profile_overflow.py::test_profile_median_range_invariance",
            "test_profile_overflow.py::test_profile_quantile_range_invariance",
            "test_profile_overflow.py::test_profile_grouped_range_invariance",
            "test_profile_overflow.py::test_profile_mean_no_edge_pileup",
        ],
    },

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
    # ── Phase 13.55.DF — draw_batch audit + Phase 13.51/13.52 type-gap fix ──
    # Triggered by architect's question "Did we audit draw_batch?" after
    # closing Phase 13.54.DF. 7-reviewer panel [OK] APPROVED. Architect
    # ratified Option B (route draw_batch through draw()) + on_error
    # default 'skip'->'raise' (BREAKING). Implementation discovered
    # architectural asymmetry between profile2d (delegates to self.profile)
    # and scatter3d (routes through self.draw) — closed via B-fix-draw
    # (architect-ratified amendment): profile2d early-dispatch added to
    # draw() mirroring scatter3d, plus _suppress_layout whitelist in
    # draw() kwarg validator and overlay-engine exemption.
    {
        "id": "BATCH.engine_routing",
        "name": "draw_batch routes through draw() — profile2d/scatter3d/aliases/overlay strings accepted; central=/time_format=/auto_title=/fit= regression-locked (D-1, D-3, D-4 closure)",
        "category": "BATCH",
        "tests": [
            "test_phase_13_55_df_draw_batch_audit.py::test_T1_dict_profile2d",
            "test_phase_13_55_df_draw_batch_audit.py::test_T2_dict_scatter3d",
            "test_phase_13_55_df_draw_batch_audit.py::test_T3_dict_histo_alias",
            "test_phase_13_55_df_draw_batch_audit.py::test_T4_dict_overlay_string",
            "test_phase_13_55_df_draw_batch_audit.py::test_T5_list_profile2d_in_group",
            "test_phase_13_55_df_draw_batch_audit.py::test_T6_list_overlay_in_group",
            "test_phase_13_55_df_draw_batch_audit.py::test_T7_central_median_via_batch_with_outlier_fixture",
            "test_phase_13_55_df_draw_batch_audit.py::test_T8_time_format_via_batch",
            "test_phase_13_55_df_draw_batch_audit.py::test_T9_auto_title_scatter_via_batch",
            "test_phase_13_55_df_draw_batch_audit.py::test_T10_fit_gauss_profile_via_batch",
            "test_phase_13_55_df_draw_batch_audit.py::test_T11_selection_vector_hist_via_batch",
            "test_phase_13_55_df_draw_batch_audit.py::test_T15_same_overlay_within_group_reuses_axis",
            "test_phase_13_55_df_draw_batch_audit.py::test_T16_suppress_layout_flows_to_typed_method",
            "test_phase_13_55_df_draw_batch_audit.py::test_T17_group_keys_stripped_before_draw_positive_test",
        ],
    },
    {
        "id": "BATCH.error_visibility",
        "name": "draw_batch on_error default changed 'skip'->'raise' (BREAKING); invalid types now surface as exceptions instead of silent-skip into _errors dict (D-2 closure)",
        "category": "BATCH",
        "tests": [
            "test_phase_13_55_df_draw_batch_audit.py::test_T12_invalid_type_raises_under_new_default",
            "test_phase_13_55_df_draw_batch_audit.py::test_T13_invalid_type_with_explicit_skip_is_counted_not_raised",
            "test_phase_13_55_df_draw_batch_audit.py::test_T14_valid_type_under_new_default_raise_succeeds",
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
        "name": "Duck-typed axis titles from AliasDataFrame, including per-component labels in vector profile facets",
        "category": "ADF",
        "tests": [
            "test_adf_integration.py::TestDuckTypedAxisTitles::test_hist_uses_axis_title",
            "test_adf_integration.py::TestDuckTypedAxisTitles::test_scatter_uses_axis_titles",
            "test_adf_integration.py::TestGetLabelMethod::test_get_label_with_title",
            "test_adf_integration.py::TestGetLabelMethod::test_get_label_without_title",
            "test_adf_integration.py::TestDFDrawDataSourceStorage::test_data_source_stored",
            # PHASE_13_86_DF Stage B-d: vector-facet axis metadata parity
            "test_phase_13_86_stage_bd.py::test_BD1_vector_facets_use_per_component_axis_title_metadata[draw]",
            "test_phase_13_86_stage_bd.py::test_BD1_vector_facets_use_per_component_axis_title_metadata[typed]",
            "test_phase_13_86_stage_bd.py::test_BD2_explicit_ylabel_still_wins_over_metadata",
            "test_phase_13_86_stage_bd.py::test_BD3_scalar_reference_behavior_unchanged",
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
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R7_nonfaceted_hist_expression_vector_overlay_remains_supported",
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
            # PHASE_13_76_DF (v1.1): float16 crash (Bug A) + 2D cell-integrity (Bug B)
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_B1_dispatch_cell_counts_match_oracle_bins",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_B2_dispatch_cell_counts_match_oracle_quantiles",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_A1_float16_public_dispatch_bins",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_A2_float16_public_dispatch_quantiles",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S1_float16_upcast_preserves_binning",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S2_qcut_duplicates_drop_preserved",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S3_discrete_facet_unaffected",
            "test_phase_13_76_facet2d_bugfix.py::test_FBY16_S4_correlated_fixture_is_nonvacuous",
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
    # ── Phase 13.57.DF — kwarg grammar audit fix phase ──
    # AF-2'/K-3 guard machinery, D4/DR-5 aliases, AF-1/AF-3/FX-4 guards,
    # K-4 forwarding parity (E-3 fix), F-E promotion, E-2 y_central,
    # AF-4 live-registry message, K-5 zero-warning corpus gate.
    {
        "id": "DISPATCH.kwarg_validation",
        "name": "C-7 typo guard on all surfaces: close-match raises naming the proper parameter (K-2 Option A); inapplicable dfdraw-semantic kwargs raise cleanly on typed methods (crash paths) and warn on draw() named-param silent drops (FB-1); P-5 matplotlib vocabulary silent; live-introspected known set",
        "category": "DISPATCH",
        "tests": [
            "test_phase_13_57_df_fixes.py::TestTypedSurfaceTypoGuard::test_typed_close_match_names_proper_parameter",
            "test_phase_13_57_df_fixes.py::TestTypedSurfaceTypoGuard::test_typed_inapplicable_semantic_kwarg_raises_clean",
            "test_phase_13_57_df_fixes.py::TestTypedSurfaceTypoGuard::test_typed_negative_control_no_matplotlib_frames",
            "test_phase_13_57_df_fixes.py::TestTypedSurfaceTypoGuard::test_typed_unknown_kwarg_warns_not_silent",
            "test_phase_13_57_df_fixes.py::TestTypedSurfaceTypoGuard::test_typed_style_passthrough_stays_silent",
            "test_phase_13_57_df_fixes.py::TestDrawSurfaceSilentDropWarns::test_fb1_bins_on_scatter_warns",
            "test_phase_13_57_df_fixes.py::TestDrawSurfaceSilentDropWarns::test_draw_close_match_still_raises_with_proper_name",
            "test_phase_13_57_df_fixes.py::TestDrawSurfaceSilentDropWarns::test_draw_applicable_params_do_not_warn",
            "test_phase_13_57_df_fixes.py::TestAF4Message::test_unknown_type_message_includes_profile2d",
            "test_phase_13_57_df_fixes.py::TestK5ZeroWarningGate::test_corpus_form_calls_emit_zero_warnings",
        ],
    },
    {
        "id": "DISPATCH.kwarg_aliases",
        "name": "Kept-forever aliases (D4 quantiles_mode->quantile_mode both surfaces, scope-limited per FX-4; DR-5 fit p0->initial, guess untouched); equivalence-tested incl. old-key forms",
        "category": "DISPATCH",
        "tests": [
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_equivalence_typed_surface",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_equivalence_draw_surface",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_no_longer_rejected_where_typo_guard_fired",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_alias_scope_per_method[profile-True]",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_alias_scope_per_method[hist-False]",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_alias_scope_per_method[hist2d-False]",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_alias_scope_per_method[hexbin-False]",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_alias_scope_per_method[scatter-False]",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_fx4_draw_surface_alias_clean_raise_on_inapplicable_type",
            "test_phase_13_57_df_fixes.py::TestQuantilesModeAlias::test_alias_conflict_raises",
            "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_p0_equivalent_to_initial",
            "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_p0_initial_conflict_raises",
            "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_guess_untouched",
            "test_phase_13_57_df_fixes.py::TestDR5P0Alias::test_old_key_equivalence_regression",
        ],
    },
    {
        "id": "DISPATCH.input_guards",
        "name": "Clean input guards: time_format/quantiles dict rejection (AF-1), non-numeric binning subjects (AF-3, profile + dispatch twins) — negative controls assert dfdraw raises, never the backend",
        "category": "DISPATCH",
        "tests": [
            "test_phase_13_57_df_fixes.py::TestAF1Guards::test_time_format_dict_clean_error",
            "test_phase_13_57_df_fixes.py::TestAF1Guards::test_time_format_dict_negative_control_not_strftime",
            "test_phase_13_57_df_fixes.py::TestAF1Guards::test_quantiles_dict_clean_error",
            "test_phase_13_57_df_fixes.py::TestAF1Guards::test_quantiles_string_elements_clean_error",
            "test_phase_13_57_df_fixes.py::TestAF1Guards::test_valid_forms_unaffected",
            "test_phase_13_57_df_fixes.py::TestAF3Guard::test_string_group_by_bins_clean_error",
            "test_phase_13_57_df_fixes.py::TestAF3Guard::test_negative_control_not_dtype_promotion_error",
            "test_phase_13_57_df_fixes.py::TestAF3Guard::test_string_discrete_grouping_still_works",
            "test_phase_13_57_df_fixes.py::TestAF3Guard::test_string_facet_by_bins_clean_error",
        ],
    },
    {
        "id": "DISPATCH.E3_parity",
        "name": "draw() entry-form parity with typed methods on the profile2d/scatter3d early-dispatch: named params (ax/selection/bins/save/...) forwarded (K-4, E-3/E3-D fix); F-E native 3-var profile promotion on draw()+draw_batch",
        "category": "DISPATCH",
        "tests": [
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_e3a_draw_direct_honors_ax",
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_e3b_typed_direct_honors_ax_regression_lock",
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_e3d_draw_faceted_no_orphan_blank_figure",
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_k4_named_params_reach_profile2d",
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_k4_selection_effect_assertion",
            "test_phase_13_57_df_fixes.py::TestE3Matrix::test_scatter3d_save_forwarded",
            "test_phase_13_57_df_fixes.py::TestFEPromotion::test_draw_profile_3var_promotes",
            "test_phase_13_57_df_fixes.py::TestFEPromotion::test_draw_batch_profile_3var_promotes",
            "test_phase_13_57_df_fixes.py::TestFEPromotion::test_profile_2var_unchanged",
        ],
    },
    {
        "id": "STATS.y_central",
        "name": "profile_data y_central column = rendered central values (E-2/D5 additive fix; y_mean unchanged forever; grouped path mirrors render, FX-1 documented)",
        "category": "STATS",
        "tests": [
            "test_phase_13_57_df_fixes.py::TestE2YCentral::test_y_central_present_and_is_median",
            "test_phase_13_57_df_fixes.py::TestE2YCentral::test_y_central_equals_mean_when_central_default",
            "test_phase_13_57_df_fixes.py::TestE2YCentral::test_grouped_export_has_y_central",
            "test_phase_13_57_df_fixes.py::TestE2YCentral::test_pre_existing_columns_unchanged",
        ],
    },

    # ── PHASE_13_86_DF — facet/vector correctness and bounded title repair ──

    {
        "id": "DISPATCH.facet_owner_contract",
        "name": "Explicit draw→typed→facet ownership contract: forwarded/consumed/refused parameters preserve semantics or fail loudly (Stage A)",
        "category": "DISPATCH",
        "tests": [
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A1_scatter_facet_preserves_real_errorbars[xerr-has_xerr]",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A1_scatter_facet_preserves_real_errorbars[yerr-has_yerr]",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[hist-time_s-extra2]",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[profile-y:time_s-extra1]",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A2_faceted_time_format_reaches_each_cell[scatter-y:time_s-extra0]",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A3_hist_facet_preserves_hist_errors",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A4_profile_group_facet_preserves_explicit_color",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A5_scatter_selection_vector_facet_is_exact_or_refuses",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A6_scatter_weights_vector_facet_refuses_until_semantics_exist",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A7_scatter_errorbar_plus_expression_color_is_exact_or_refuses",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A7s_single_panel_errorbar_plus_expression_color_is_exact_or_refuses",
            "test_phase_13_86_stage_a_red.py::TestStageAPropertyReds::test_A8_internal_suppress_color_cycle_never_warns_as_user_kwarg",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[hist-x-required0]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[hist2d-y:x-required3]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[profile-y:x-required2]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10a_draw_to_typed_plain_expression_green_guard[scatter-y:x-required1]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10b_draw_profile_normalize_vector_special_delegation_preserves_request",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[hist2d]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[hist]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[profile2d]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[profile]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10d_draw_to_typed_disposition_is_complete_and_exclusive[scatter]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10e_draw_signature_has_no_unclassified_global_parameter",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[hist-x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[hist2d-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[profile-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[profile2d-z:y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10f_draw_to_typed_forwarded_names_reach_owner[scatter-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[hist2d-fit-gauss]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[profile2d-color-magenta]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10g_draw_to_typed_refused_names_fail_loudly[profile2d-marker-s]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A11_internal_vector_control_is_registered_with_typo_guard",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist-x-draw]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist-x-typed]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist2d-y:x-draw]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[hist2d-y:x-typed]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[profile-y:x-draw]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[profile-y:x-typed]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[scatter-y:x-draw]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A12_legacy_facet_true_does_not_leak_facet_layout_controls[scatter-y:x-typed]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[hist2d]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[hist]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[profile]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9a_typed_to_facet_disposition_is_complete_and_exclusive[scatter]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[hist-x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[hist2d-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[profile-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9b_typed_to_facet_forwarded_names_reach_dispatch[scatter-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[hist-x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[hist2d-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[profile-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9c_consumed_names_do_not_leak_to_facet_dispatch[scatter-y:x]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9d_scatter_refused_multi_branch_vector_semantics_are_loud[selection_vector]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9d_scatter_refused_multi_branch_vector_semantics_are_loud[weights_vector]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9e_profile_refused_same_plus_facet_is_loud",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9f2_hist_refused_normalize_plus_facet_is_loud",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9f_hist2d_refused_fit_is_loud",
        ],
    },

    {
        "id": "FACET.known_silent_gap",
        "name": "PLANNED gap: facet_by with ax=/same=True must fail loudly instead of being silently ignored; strict XFAIL inventory, not supported behavior",
        "category": "FACET",
        "tests": [
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist-x-ax]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist-x-same]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist2d-y:x-ax]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[hist2d-y:x-same]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[profile-y:x-ax]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[scatter-y:x-ax]",
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A9g_known_silent_gap_is_loud[scatter-y:x-same]",
        ],
    },

    {
        "id": "VECTOR.facet_routing",
        "name": "Profile expression-vector facet_by=\"vector\" routing preserves intact expression and draw()≡typed-owner semantics (Stage B-a)",
        "category": "VECTOR",
        "tests": [
            "test_phase_13_86_stage_a_red.py::TestStageAStructuralBoundaryInventory::test_A10c_draw_profile_facet_vector_preserves_intact_expression_for_typed_owner",
            "test_phase_13_86_stage_ba.py::test_Ba1_draw_profile_facet_vector_matches_typed_owner",
            "test_phase_13_86_stage_ba.py::test_Ba2_vector_facets_keep_expression_components_distinct",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R8_profile_expression_vector_facets_remain_supported",
        ],
    },

    {
        "id": "VECTOR.ordinary_facet_compose",
        "name": "Profile expression vectors compose inside ordinary facets with group/selection/weights channels; unsupported neighboring compositions refuse loudly (Stage B-b)",
        "category": "VECTOR",
        "tests": [
            "test_phase_13_86_stage_bb.py::test_BB1_typed_profile_vector_ordinary_facet_matches_independent_oracle",
            "test_phase_13_86_stage_bb.py::test_BB2_draw_group_vector_ordinary_facet_matches_oracle_and_typed_owner",
            "test_phase_13_86_stage_bb.py::test_BB3_realistic_group_quantiles_and_facet_bins_compose_inside_panels",
            "test_phase_13_86_stage_bb.py::test_BB4_expression_vector_selection_vector_composes_within_each_facet[inner-2]",
            "test_phase_13_86_stage_bb.py::test_BB4_expression_vector_selection_vector_composes_within_each_facet[outer-4]",
            "test_phase_13_86_stage_bb.py::test_BB5_expression_vector_weights_vector_composes_within_each_facet[inner-2]",
            "test_phase_13_86_stage_bb.py::test_BB5_expression_vector_weights_vector_composes_within_each_facet[outer-4]",
            "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[hist-[y0,y1]-draw]",
            "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[hist-[y0,y1]-typed]",
            "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[scatter-[y0,y1]:x-draw]",
            "test_phase_13_86_stage_bb.py::test_BB6_neighbor_hist_scatter_expression_vector_facet_refuse_loudly[scatter-[y0,y1]:x-typed]",
            "test_phase_13_86_stage_bb.py::test_BB7_profile_expression_vector_nd_facet_positive_contract[draw]",
            "test_phase_13_86_stage_bb.py::test_BB7_profile_expression_vector_nd_facet_positive_contract[typed]",
            "test_phase_13_86_stage_bb.py::test_BB8_public_same_true_neighbor_remains_invalid",
            "test_phase_13_86_stage_bb.py::test_BB9_expression_vector_channel_identity_is_stable_across_facets",
        ],
    },

    {
        "id": "VECTOR.expression_components",
        "name": "Computed y components and shared x expressions are materialized correctly in vector and ordinary profile facets through typed/draw doors (Stage B-c)",
        "category": "VECTOR",
        "tests": [
            "test_phase_13_86_stage_bc.py::test_BC1_computed_y_component_works_with_vector_facets[draw]",
            "test_phase_13_86_stage_bc.py::test_BC1_computed_y_component_works_with_vector_facets[typed]",
            "test_phase_13_86_stage_bc.py::test_BC2_computed_y_component_works_inside_ordinary_facets[draw]",
            "test_phase_13_86_stage_bc.py::test_BC2_computed_y_component_works_inside_ordinary_facets[typed]",
            "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[draw-facet]",
            "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[draw-vector]",
            "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[typed-facet]",
            "test_phase_13_86_stage_bc.py::test_BC3_computed_shared_x_expression_is_materialized[typed-vector]",
        ],
    },

    {
        "id": "TITLE.width_aware_selection",
        "name": "Automatic-title selection text is preserved semantically and fitted at render time to physical figure width; no fixed 72-character cutoff (Stage C-a)",
        "category": "TITLE",
        "tests": [
            "test_auto_title.py::TestAutoTitleParts::test_long_selection_preserved_until_render",
            "test_phase_13_86_stage_ca.py::test_CA1_wide_figure_keeps_full_selection_but_narrow_figure_elides",
            "test_phase_13_86_stage_ca.py::test_CA2_build_auto_title_preserves_selection_until_render_width_is_known",
            "test_phase_13_86_stage_ca.py::test_CA3_faceted_wide_figure_keeps_full_selection_in_figure_title",
        ],
    },

    # ── PHASE_13_88_DF — normalization correctness repairs ──

    {
        "id": "NORMALIZE.hist_shared_autorange",
        "name": "Histogram ratio normalization resolves one canonical pooled range before branch rendering; explicit ranges remain exact and edge compatibility stays guarded (PHASE_13_88 Bug A)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[None]",
            "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[hybrid]",
            "test_phase_13_88_bug_a_hist_autorange.py::test_A1_hist_selection_vector_uses_one_canonical_pooled_range[robust_4mad]",
            "test_phase_13_88_bug_a_hist_autorange.py::test_A2_explicit_numeric_range_remains_exact_green_control",
            "test_phase_13_88_bug_a_hist_autorange.py::test_A3_weights_vector_same_rows_remains_green_control",
            "test_phase_13_88_bug_a_hist_autorange.py::test_A4_two_expression_vector_uses_union_domain",
        ],
    },
    {
        "id": "NORMALIZE.profile_expression_materialization",
        "name": "Normalized profiles materialize supported computed x/y positional expressions on the active plain/grouped/faceted curve frame before low-level rendering (PHASE_13_88 Bug B)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[plain-[y,2*y]:x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[plain-[y,2*y]:2*x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[grouped-[y,2*y]:x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[grouped-[y,2*y]:2*x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[faceted-[y,2*y]:x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B1_profile_normalize_materializes_computed_positional_expressions[faceted-[y,2*y]:2*x]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_B2_single_y_computed_x_supported_with_selection_vector",
        ],
    },
    {
        "id": "NORMALIZE.profile_shared_autorange",
        "name": "Plain, grouped and faceted normalized-profile owners share the canonical configured 1-D autorange instead of local raw min/max (PHASE_13_88 F7)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[plain]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[grouped]",
            "test_phase_13_88_bug_b_profile_expressions.py::test_F7_profile_normalize_uses_canonical_shared_autorange_not_raw_minmax[faceted]",
        ],
    },
    {
        "id": "NORMALIZE.hist_draw_parity",
        "name": "Public draw() preserves atomic two-expression histogram normalize semantics and computed branch expressions exactly as the typed hist() owner (PHASE_13_88 Bug C/follow-up)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_88_bug_c_draw_hist_normalize.py::test_C1_draw_hist_vector_normalize_matches_typed_owner",
            "test_phase_13_88_bug_c_draw_hist_normalize.py::test_C2_draw_hist_computed_vector_element_is_materialized",
        ],
    },

    # ── PHASE_13_89_DF — explicit-axis normalize ownership ──

    {
        "id": "NORMALIZE.profile_explicit_ax_embedding",
        "name": "Plain normalized profiles embed into caller-owned subplot Axes without changing numerical normalization or unrelated Figure ownership (PHASE_13_89)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[delta-typed]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[delta-draw]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[ratio-typed]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R1_R3_plain_overlay_diff_embeds_in_caller_slot_and_preserves_numbers[ratio-draw]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R2_draw_selection_vector_production_shape_embeds_and_preserves_numbers",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R4_diff_only_uses_exact_supplied_axis[delta]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R4_diff_only_uses_exact_supplied_axis[ratio]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R5_multisubplot_embedding_changes_only_target_slot",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R10_profile_and_hist_standalone_controls_remain_supported",
        ],
    },
    {
        "id": "NORMALIZE.explicit_ax_refusal_guards",
        "name": "Unsupported normalize + explicit-Axes ownership combinations refuse before caller mutation or persistent Figure leakage (PHASE_13_89)",
        "category": "NORMALIZE",
        "tests": [
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R6_custom_add_axes_refuses_before_mutation_or_leak",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R7_nonempty_placeholder_refuses_before_mutation_or_leak",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R8_grouped_or_faceted_normalize_with_ax_refuses_without_leak[grouped]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R8_grouped_or_faceted_normalize_with_ax_refuses_without_leak[faceted]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R9_hist_normalize_ax_preserves_existing_loud_refusal[typed]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R9_hist_normalize_ax_preserves_existing_loud_refusal[draw]",
            "test_phase_13_89_explicit_ax_normalize_red.py::test_R11_ax_plus_save_refuses_without_mutating_caller",
        ],
    },

    # ── PHASE_13_91_DF — histogram expression-vector facets ──

    {
        "id": "HIST.expression_vector_facets",
        "name": "Histogram bracket-expression vectors render one facet per component through typed/draw doors, with independent counts, stable expression identity, and per-component x-axis metadata (PHASE_13_91)",
        "category": "HIST",
        "tests": [
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R1_R2_computed_expression_vector_facets_match_numpy_truth[typed]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R1_R2_computed_expression_vector_facets_match_numpy_truth[draw]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R3_typed_and_draw_doors_have_identical_panel_values",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R4_physical_column_vector_facets_preserve_component_x_metadata[typed]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R4_physical_column_vector_facets_preserve_component_x_metadata[draw]",
        ],
    },
    {
        "id": "HIST.expression_vector_facet_guards",
        "name": "Histogram expression-vector facet composition rejects duplicate identities and deferred vector/normalize cross-products before Figure allocation (PHASE_13_91)",
        "category": "HIST",
        "tests": [
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R5_duplicate_expression_identity_refuses_before_figure_allocation[typed]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R5_duplicate_expression_identity_refuses_before_figure_allocation[draw]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[selection-vector]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[weights-vector]",
            "test_phase_13_91_hist_expression_vector_facets_red.py::test_R6_deferred_cross_products_refuse_before_figure_allocation[normalize]",
        ],
    },


    # ── PHASE_13_92_DF — mapped scatter color + error-bar composition ──
    {
        "id": "SCATTER.mapped_color_errorbars",
        "name": "Numeric mapped scatter color composes with x/y error bars while preserving one common row domain, scatter-owned colorbar/marker semantics, lower-zorder marker-free uncertainty geometry, per-point marker composition, full-domain shared color normalization, categorical fail-loud behavior, and legacy neighboring owners (PHASE_13_92)",
        "category": "SCATTER",
        "tests": [
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R1_numeric_mapped_color_plus_yerr_two_layer",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R2_numeric_mapped_color_plus_xerr_two_layer",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R3_numeric_mapped_color_plus_both_errors_preserves_both_geometries",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R4_R5_faceted_numeric_mapped_color_preserves_points_and_errors[yerr]",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R4_R5_faceted_numeric_mapped_color_preserves_points_and_errors[xerr]",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R6_computed_color_expression_plus_errorbar_and_colorbar",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R7_colorbar_mappable_is_exact_point_collection_object",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R8_asymmetric_row_alignment_after_selection_nan_and_range_filter",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R9_errorbars_have_explicitly_lower_zorder_than_points",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R10_per_point_marker_expression_plus_yerr_keeps_marker_owner",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_G1_string_categorical_color_plus_errors_remains_fail_loud",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_C1_fixed_scalar_color_plus_errors_keeps_legacy_errorbar_owner",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_C2_numeric_color_only_keeps_ordinary_scatter_path",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R11_caller_dataframe_not_mutated_by_composed_path",
            "test_phase_13_92_scatter_mapped_color_errorbars.py::test_R12_marker_split_preserves_full_numeric_color_domain_and_colorbar_owner",
        ],
    },

    # ── PHASE_13_93_DF — opt-in dense N-D facet presentation (Stage A) ──
    {
        "id": "FACET.nd_dense_presentation",
        "name": (
            "N-dimensional facet figures use one layout owner per Figure, "
            "preserve ordered pandas categorical coordinates and explicit facet_order, "
            "provide outer row/column headings and figure-level semantic axis labels "
            "independently of numerical axis sharing, and honor explicit margins/spacing "
            "without scaling values or errors; unsupported inputs refuse and legacy "
            "presentation remains compatible (PHASE_13_93 Stage A)"
        ),
        "category": "FACET",
        "tests": [
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R1_2d_nd_facet_has_exactly_one_whole_figure_layout_pass',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R2_3d_nd_facet_has_one_layout_pass_per_generated_figure',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R3_ordered_categorical_rows_preserve_declared_order',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R4_ordered_categorical_columns_preserve_declared_order',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R5_explicit_facet_order_overrides_categorical_order',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R5b_invalid_facet_order_refuses_clearly',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R6_outer_column_headings_appear_once_each',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R7_outer_row_headings_appear_once_each',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R8_outer_mode_removes_outer_dimension_text_from_panels',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R9_shared_xlabel_is_one_figure_semantic_label',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R10_shared_ylabel_preserves_independent_limits_and_ticks',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R11_explicit_figure_margins_survive_final_layout',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R12_explicit_subplot_spacing_survives_final_layout',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R13_legacy_2d_facet_presentation_unchanged_without_new_options',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R14_rendered_left_decorations_do_not_intersect[0.06]',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R14_rendered_left_decorations_do_not_intersect[0.08]',
            'test_phase_13_93_stage_a_dense_facet_red.py::test_A_R15_too_narrow_left_gutter_warns_without_changing_margins',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H1_public_draw_retains_unscaled_scatter_errorbar_values',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[scatter-coefficient:region-extra0]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[hist-coefficient-extra1]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[profile-coefficient:region-extra2]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H2_all_ND_plot_kinds_share_the_single_layout_owner[hist2d-coefficient:region-extra3]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H3_3D_categorical_figure_order_and_outer_headings',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad0]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad1]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad2]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H4_invalid_facet_order_never_degrades_silently[bad3]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H5_scalar_facet_rejects_ND_only_presentation_options',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H6_shared_ylabel_without_outer_mode_keeps_row_identity',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[figure_margins-value0]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[figure_margins-value1]',
            'test_phase_13_93_stage_a_handover_regressions.py::test_A_H7_invalid_layout_parameters_raise_before_figure_allocation[subplot_spacing-value2]',
        ],
    },

    # ── PHASE_13_93_DF — figure-wide categorical scatter display (Stage B) ──
    {
        "id": "SCATTER.figure_categorical_color_legend",
        "name": (
            "Opt-in N-D mapped numeric scatter colors yield one figure-level categorical "
            "legend with stable category colors and equivalent markers across facets "
            "and 3-D figure pages; keep point/error-bar geometry, legacy colorbars, "
            "explicit colorbar choice and strict refusal boundaries (PHASE_13_93 Stage B)"
        ),
        "category": "SCATTER",
        "tests": [
            'test_phase_13_93_stage_b_color_display_red.py::test_b1_omitted_color_display_keeps_original_colorbars',
            'test_phase_13_93_stage_b_color_display_red.py::test_b2_b3_binary_numeric_side_one_figure_legend[auto]',
            'test_phase_13_93_stage_b_color_display_red.py::test_b2_b3_binary_numeric_side_one_figure_legend[legend]',
            'test_phase_13_93_stage_b_color_display_red.py::test_b4_explicit_colorbar_is_not_silently_changed_to_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b5_same_side_same_rgba_in_all_facets_even_singleton',
            'test_phase_13_93_stage_b_color_display_red.py::test_b6_no_repeated_axes_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b8_matching_marker_partition_produces_combined_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b8b_non_equivalent_marker_partition_refuses',
            'test_phase_13_93_stage_b_color_display_red.py::test_b9_uncertainties_are_retained_with_two_marker_classes',
            'test_phase_13_93_stage_b_color_display_red.py::test_b10_auto_threshold_controls_discrete_heuristic',
            'test_phase_13_93_stage_b_color_display_red.py::test_b10b_explicit_legend_limit_refuses',
            'test_phase_13_93_stage_b_color_display_red.py::test_b11_bad_mode_rejected_before_plotting',
            'test_phase_13_93_stage_b_color_display_red.py::test_b12_invalid_threshold_rejected',
            'test_phase_13_93_stage_b_color_display_red.py::test_b13_real_grid_40_cells_one_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b14_layout_still_one_owner',
            'test_phase_13_93_stage_b_color_display_red.py::test_b15_max_legend_alone_does_not_silently_do_nothing',
            'test_phase_13_93_stage_b_color_display_red.py::test_b16_string_color_with_errorbars_preserves_explicit_refusal',
            'test_phase_13_93_stage_b_color_display_red.py::test_b17_three_dimensional_facets_keep_one_color_domain_across_figures',
            'test_phase_13_93_stage_b_color_display_red.py::test_b18_numeric_color_expression_three_categories_with_one_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b19_grouped_color_owner_does_not_silently_fabricate_side_legend',
            'test_phase_13_93_stage_b_color_display_red.py::test_b20_nonfaceted_explicit_none_is_legacy_compatible',
            'test_phase_13_93_stage_b_color_display_red.py::test_b21_nonfaceted_legend_refuses_before_matplotlib',
            'test_phase_13_93_stage_b_color_display_red.py::test_b_R11_legend_disjoint_from_column_headings_and_title[0.89]',
            'test_phase_13_93_stage_b_color_display_red.py::test_b_R11_legend_disjoint_from_column_headings_and_title[0.92]',
            'test_phase_13_93_stage_b_color_display_red.py::test_b_R12_after_return_manual_annotation_edit_is_not_reversed',
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
    # ── Candidate N-D retained-state first slice (phase owner under review) ──
    {
        "id": "STATS.retained_histogram_state",
        "name": 'Internal generic sparse weighted histogram retained state, named categorical coordinates and strict edge/provenance/schema guards (first-slice candidate)',
        "category": "STATS",
        "tests": [
            'test_phase_13_82_retained_state_red.py::test_t1_weighted_histogram_against_numpy',
            'test_phase_13_82_retained_state_red.py::test_t2_cancelled_weight_is_not_empty',
            'test_phase_13_82_retained_state_red.py::test_t4_categorical_coordinate_is_semantic_dimension',
            'test_phase_13_82_retained_state_red.py::test_t5_schema_name_and_partition_contract',
            'test_phase_13_82_retained_state_red.py::test_t6_sparse_underflow_overflow_nan_and_last_edge',
            'test_phase_13_82_retained_state_red.py::test_t9_invalid_resolved_edges_refused',
            'test_phase_13_82_retained_state_red.py::test_t11_numeric_categories_preserve_identity',
            'test_phase_13_82_retained_state_red.py::test_t13_two_named_categorical_dimensions_form_sparse_product',
        ],
    },
    {
        "id": "STATS.retained_profile_state",
        "name": 'Internal generic weighted profile retained moments and logical payload round-trip with corruption guards (first-slice candidate)',
        "category": "STATS",
        "tests": [
            'test_phase_13_82_retained_state_red.py::test_t3_weighted_profile_against_independent_raw_sums',
            'test_phase_13_82_retained_state_red.py::test_t7_one_node_class_different_sufficient_state',
            'test_phase_13_82_retained_state_red.py::test_t8_logical_payload_round_trip_with_metadata',
            'test_phase_13_82_retained_state_red.py::test_t10_nonfinite_weight_and_profile_value_not_accumulated',
            'test_phase_13_82_retained_state_red.py::test_t12_corrupt_payload_fractional_entries_refused',
        ],
    },

    {
        "id": "STATS.exact_retained_state_merge",
        "name": (
            "Exact pooled merge of compatible retained histogram/profile state, "
            "with duplicate-partition protection, categorical outer alignment, "
            "provenance retention and pooled-data/finalization equivalence "
            "(PHASE_13_82 continuation)"
        ),
        "category": "STATS",
        "tests": [
            'test_phase_13_82_exact_merge_red.py::test_m1_weighted_histogram_merge_equals_direct_pooled_state',
            'test_phase_13_82_exact_merge_red.py::test_m2_weighted_profile_merge_equals_direct_pooled_state',
            'test_phase_13_82_exact_merge_red.py::test_m3_signed_weight_cancellation_across_partitions_stays_occupied',
            'test_phase_13_82_exact_merge_red.py::test_m4_sparse_categorical_union_matches_direct_pooled_state',
            'test_phase_13_82_exact_merge_red.py::test_m5_duplicate_partition_refused_including_nested_contributors',
            'test_phase_13_82_exact_merge_red.py::test_m6_different_name_refused',
            'test_phase_13_82_exact_merge_red.py::test_m7_different_binned_axis_refused_without_rebinning',
            'test_phase_13_82_exact_merge_red.py::test_m8_statistic_kind_and_state_contract_mismatch_refused',
            'test_phase_13_82_exact_merge_red.py::test_m9_semantic_source_definition_mismatch_refused',
            'test_phase_13_82_exact_merge_red.py::test_m10_metadata_mismatch_refused_for_first_exact_merge_slice',
            'test_phase_13_82_exact_merge_red.py::test_m11_requires_at_least_two_nodes_and_explicit_output_partition',
            'test_phase_13_82_exact_merge_red.py::test_m12_merged_profile_finalization_matches_direct_pooled_raw_data',
            'test_phase_13_82_exact_merge_red.py::test_m13_payload_roundtrip_states_merge_to_direct_pooled_truth',
        ],
    },


    # ── PHASE_13_94_DF — profile expression-vector × 2D N-D facet ──
    {
        "id": "PROFILE.expression_vector_nd_facet",
        "name": (
            "Profile expression-vectors compose with N-D facet partitions and "
            "group overlays, global selected-frame group bins, stable physical "
            "group/vector identity and one Figure-level legend; preserve "
            "independent scalar numerical oracle and neighboring refusals "
            "(PHASE_13_94_DF)"
        ),
        "category": "PROFILE",
        "tests": [
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R1_two_branches_in_each_2d_facet_without_group[typed]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R1_two_branches_in_each_2d_facet_without_group[draw]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R2_computed_vector_residuals_equal_materialized_truth',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R3_numeric_3x5_facet_classification_matches_global_pandas_membership',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R4_categorical_group_composes_2_vectors_times_3_groups',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R5_global_group_bins_maintain_physical_identity_across_cells',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R6_group_quantile_boundaries_resolved_globally',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R7_vector_group_identity_and_single_figure_legend',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R8_strong_numeric_decomposition_with_selection_and_all_coordinates',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R9_missing_facet_cell_preserves_empty_cell_schema',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R10_forwarding_nondefault_min_entries_50_masks_short_profiles',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R11_typed_and_public_draw_nested_stats_equivalent',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R12_existing_one_dimensional_profile_vector_facet_still_works',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R13_neighbor_hist_scatter_nd_vector_still_refuse_loudly[hist-[y0,y1]]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R13_neighbor_hist_scatter_nd_vector_still_refuse_loudly[scatter-[y0,y1]:x]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R14_existing_scalar_nd_facet_is_not_changed_by_new_global_policy',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R15_missing_intermediate_group_keeps_global_color_marker_and_label',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted0]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted1]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted2]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure[restricted3]',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R17_figure_channel_assignment_is_resolved_once',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R18_explicit_group_color_override_remains_authoritative',
            'test_phase_13_94_profile_vector_nd_facet_red.py::test_R19_group_bins_without_group_column_fail_before_figure',
        ],
    },


    # PHASE_13_95_DF — explicit global/facet scope and quantile titles.
    {
        "id": "PROFILE.group_by_scope",
        "name": 'Public deterministic auto/global/facet group-by domain for 1-D and 2-D grouped profiles; selected-frame bins and quantiles, global physical identities, truthful legends and legacy compatibility (PHASE_13_95_DF)',
        "category": 'PROFILE',
        "tests": [
            'test_phase_13_95_group_scope_title_red.py::test_R1_public_signature_and_invalid_values',
            'test_phase_13_95_group_scope_title_red.py::test_R2_auto_preserves_historical_1d_faceted_numbers',
            'test_phase_13_95_group_scope_title_red.py::test_R3_explicit_global_1d_reuses_exact_physical_groups',
            'test_phase_13_95_group_scope_title_red.py::test_R4_explicit_local_1d_matches_auto',
            'test_phase_13_95_group_scope_title_red.py::test_R5_global_quantiles_in_1d',
            'test_phase_13_95_group_scope_title_red.py::test_R6_global_missing_group_2d_has_stable_identity',
            'test_phase_13_95_group_scope_title_red.py::test_R7_global_figure_legend_obeys_explicit_hide',
            'test_phase_13_95_group_scope_title_red.py::test_R8_facet_local_has_no_false_global_numeric_legend',
            'test_phase_13_95_group_scope_title_red.py::test_R9_13_94_auto_matches_omitted_policy',
            'test_phase_13_95_group_scope_title_red.py::test_R10_13_94_explicit_facet_policy_changes_group_boundaries',
            'test_phase_13_95_group_scope_title_red.py::test_R13_typed_draw_parity_for_explicit_global[typed]',
            'test_phase_13_95_group_scope_title_red.py::test_R13_typed_draw_parity_for_explicit_global[draw]',
            'test_phase_13_95_group_scope_title_red.py::test_R14_categorical_no_binning_no_numeric_changes',
            'test_phase_13_95_group_scope_title_red.py::test_R15_single_global_cut_event',
            'test_phase_13_95_group_scope_title_red.py::test_R16_selection_domain_global_bin_membership_independent_pandas',
            'test_phase_13_95_group_scope_title_red.py::test_R17_single_dimension_list_is_equivalent_to_scalar_facet',
            'test_phase_13_95_group_scope_title_red.py::test_R18_global_quantile_boundaries_derived_from_selection_not_raw_input',
            'test_phase_13_95_group_scope_title_red.py::test_R20_no_scope_widening_to_other_plot_families',
            'test_phase_13_95_group_scope_title_red.py::test_R21_global_top_k_does_not_claim_unsupported_global_ranking',
            'test_phase_13_95_group_scope_title_red.py::test_R22_2d_scalar_global_matches_selected_pandas_counts',
            'test_phase_13_95_group_scope_title_red.py::test_R23_auto_vector_13_94_figure_legend_obeys_explicit_hide',
            'test_phase_13_95_group_scope_title_red.py::test_R24_explicit_all_overrides_default_shared_global_legend',
            'test_phase_13_95_group_scope_title_red.py::test_R25_vector_1d_global_bins_match_full_selected_pandas_and_scalar[draw]',
            'test_phase_13_95_group_scope_title_red.py::test_R25_vector_1d_global_bins_match_full_selected_pandas_and_scalar[typed]',
            'test_phase_13_95_group_scope_title_red.py::test_R26_vector_1d_global_quantile_scope_reuses_selected_boundaries[draw]',
            'test_phase_13_95_group_scope_title_red.py::test_R26_vector_1d_global_quantile_scope_reuses_selected_boundaries[typed]',
            'test_phase_13_95_group_scope_title_red.py::test_R27_vector_1d_auto_and_facet_retain_legacy_local_semantics',
        ],
    },
    {
        "id": "TITLE.profile_quantile_semantics",
        "name": 'Automatic profile title reports non-default two-value and multi-value quantiles once, preserving explicit titles and auto_title=False (PHASE_13_95_DF)',
        "category": 'TITLE',
        "tests": [
            'test_phase_13_95_group_scope_title_red.py::test_R11_1d_quantiles_appear_once_in_auto_title',
            'test_phase_13_95_group_scope_title_red.py::test_R12_title_false_and_explicit_title_precedence',
            'test_phase_13_95_group_scope_title_red.py::test_R19_more_than_two_quantiles_are_listed_not_misrepresented_as_range',
        ],
    },


    # PHASE_13_96_DF — Stage A/B/C retained-statistic rendering.
    {
        "id": 'PROFILE.median_resolved_bin_edges',
        "name": 'Existing median uses precisely the resolved profile x-bin edges (PHASE_13_96 Stage A)',
        "category": 'PROFILE',
        "tests": [
            'test_phase_13_96_median_bin_owner_red.py::test_M1_both_median_curve_agrees_with_independent_exact_bin_truth',
            'test_phase_13_96_median_bin_owner_red.py::test_M2_median_exported_y_central_agrees_with_same_bin_truth',
            'test_phase_13_96_median_bin_owner_red.py::test_M3_explicit_xrange_median_remains_exact',
        ],
    },
    {
        "id": 'PROFILE.stat_list_basic',
        "name": 'Mean/std/SEM statistic curves from one profile computation; independent numerical truth, compatibility and supported boundaries (PHASE_13_96)',
        "category": 'PROFILE',
        "tests": [
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B01_stat_curves_match_independent_numpy[mean]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B01_stat_curves_match_independent_numpy[std]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B01_stat_curves_match_independent_numpy[sem]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B02_two_stat_curves_in_order_and_without_mean_uncertainties',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B03_draw_and_typed_profile_equivalent',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B04_historical_default_none_is_unchanged',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B05_compute_profile_called_once',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B06_weighted_statistics_equal_existing_profile_data',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B07_future_or_conflicting_combinations_fail_loudly[bad0-median]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B07_future_or_conflicting_combinations_fail_loudly[bad1-quantiles]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B07_future_or_conflicting_combinations_fail_loudly[bad2-group]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B07_future_or_conflicting_combinations_fail_loudly[bad3-facet]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B07_future_or_conflicting_combinations_fail_loudly[bad4-central]',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B08_weighted_statistics_agree_with_direct_weighted_numpy',
            'test_phase_13_96_stage_b_stat_curves_red.py::test_B09_refusal_never_allocates_a_new_figure',
        ],
    },

    {
        "id": 'PROFILE.stat_list_median_quantiles',
        "name": 'Median and existing quantile bundles in independent profile statistic curves; exact-bin NumPy truth, central-line de-duplication, unsupported combinations fail-loud (PHASE_13_96 Stage C)',
        "category": 'PROFILE',
        "tests": [
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C01_median_matches_numpy_at_same_bin_centers_and_has_no_errorbars',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C02_mean_median_std_sem_four_distinct_curves_and_one_profile_computation',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C03_quantiles_only_reuses_legacy_mode_and_reports_values[error_bars-qs0]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C03_quantiles_only_reuses_legacy_mode_and_reports_values[band-qs1]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C03_quantiles_only_reuses_legacy_mode_and_reports_values[discrete-qs2]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C03_quantiles_only_reuses_legacy_mode_and_reports_values[nested_band-qs3]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C04_mean_and_band_quantile_have_one_central_mean_curve',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C05_mean_quantile_asym_errorbars_overlay_without_duplicate_mean_curve',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C06_median_discrete_quantile_half_is_not_duplicated',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra0-quantiles=]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra1-median]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra2-group]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra3-weighted]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra4-central]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C07_refuse_unsupported_before_new_figure[extra5-weighted]',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C08_draw_door_median_and_quantile_parity',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C09_quantile_values_match_raw_numpy_on_actual_resolved_autorange_edges',
            'test_phase_13_96_stage_c_median_quantiles_red.py::test_C10_returned_statistic_identity_exists_without_return_data',
        ],
    },

    {
        "id": 'PROFILE.stat_list_vector_composition',
        "name": 'Existing vector branches B compose independently with statistic curves S, preserving B × S identity, numerical truth, one Figure legend and fail-loud finite channels (PHASE_13_96 Stage D)',
        "category": 'PROFILE',
        "tests": [
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D01_expression_vector_stat_outer_product_draw_typed_and_numpy',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D02_selection_vector_stat_outer_preserves_selections',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D03_weight_vector_stat_outer_matches_scalar_reference',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D04_mean_colors_follow_branch_not_statistic',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D05_channel_collision_refused_before_creating_figure',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D06_legacy_vector_payload_unchanged_when_stat_list_none',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D07_explicit_show_legend_false_and_marker_identity',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D08_expression_and_selection_branch_cross_product',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D09_quantile_band_composes_with_expression_vector_without_extra_mean',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D10_discrete_quantile_channel_refuses_and_no_plot',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D11_weight_vector_unsupported_statistics_refuse_before_artists[median]',
            'test_phase_13_96_stage_d_vector_stat_red.py::test_D11_weight_vector_unsupported_statistics_refuse_before_artists[quantiles]',
        ],
    },

    {
        "id": "PROFILE.stat_list_group_facet",
        "name": "Statistic curves mean/std/SEM reuse existing grouped and faceted profile owners with independent numerical truth, stable global group colors, finite legends, compute-once and explicit unsupported-surface refusals (PHASE_13_96 Stage E)",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_96_stage_e_group_facet_red.py::test_E1_grouped_identity_and_direct_numpy_truth",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E2_grouped_draw_typed_equivalence",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E3_one_dimensional_facet_stats_truth",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E4_refusals_before_figure_allocation",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E5_grouped_weighted_numpy_truth",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E6_global_group_bins_1d_facets_semantic_legend",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E7_global_group_quantiles_2d_facets_one_figure_legend",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E8_ungrouped_facets_single_figure_stat_legend",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E9_legacy_grouped_and_faceted_payload_when_none",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E10_grouped_count_compute_once",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E11_global_group_sparse_cell_colors_stable",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E12_grouped_explicit_legend_false",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E13_two_dimensional_global_group_statistics_match_raw_pandas",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E14_unsupported_3d_facet_and_group_color_refuse",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E15_draw_nonfaceted_explicit_global_bins_against_pandas",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E16_typed_nonfaceted_explicit_global_quantiles_equal_auto",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E17_nonfaceted_global_without_stat_list_preserves_legacy",
            "test_phase_13_96_stage_e_group_facet_red.py::test_E18_nonfaceted_scope_refusal_boundaries_before_figure_allocation",
        ],
    },

    {
        "id": "PROFILE.stat_list_vector_groupfacet_single_std",
        "name": "Two expression profile branches compose with five Figure-global groups, 2-D facets and singleton std using group colors, expression markers, factored Figure legend and independent NumPy/pandas truth (PHASE_13_96 Stage F)",
        "category": "PROFILE",
        "tests": [
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F01_exact_stagef_query_renders_through_both_public_doors",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F02_every_occupied_expression_group_facet_xbin_matches_pandas_numpy_truth",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F03_sparse_groups_and_min_entries_do_not_invent_rendered_curves_or_points",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F04_global_group_colors_stay_physical_through_sparse_middle_group",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F05_two_expression_markers_own_identity_and_all_curves_are_marker_only_std",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F06_one_factored_figure_legend_truthful_and_nonoverlapping",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F07_public_result_shape_remains_facet_keyed_with_ordered_expression_branches",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F08_stat_list_none_preserves_existing_nd_vector_group_facet_route",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override0]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override1]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override2]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override3]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override4]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override5]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09_unsupported_neighboring_combinations_refuse_before_figure_allocation[override6]",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F09b_private_stagef_coordinator_token_is_not_a_public_escape_hatch",
            "test_phase_13_96_stage_f_vector_groupfacet_single_std_red.py::test_F10_global_group_resolution_and_figure_layout_have_single_owners",
        ],
    },

]
