"""
Feature Taxonomy — AliasDataFrame

44 features + 3 new (PHASE_13_23_ADF). Patterns match actual pytest node IDs.
Updated from .pytest_report.json (2026-04-02, 1456 tests).
Phase 13.12.ADF (2026-04-12): Added test_I5..I17 patterns for the 31 new
invariance tests. No new features added — existing features gain
verification coverage.
Phase 13.18.ADF (2026-04-13): Added 3 new features under REGISTERED_FUNCTIONS
category (FUNC.regression_metadata, FUNC.evaluator_from_metadata,
FUNC.regression_persistence) covered by 11 R-tests. Total features: 41 → 44.
Phase 13.23.ADF (2026-04-29): Added 3 new features (SUB.multilevel,
CORE.dependency_tree, CORE.invalidation). Extended 6 existing features
with new test patterns (J1, K1, K2, E1, E2, S1, D1, S5). Total: 44 → 47.
Phase 13.56.ADF (2026-06-11): Added 2 new features under DISPATCH category
(DISPATCH.adf_routing, DISPATCH.error_visibility) covered by the 29
Phase-13.55 tests + 14 Phase-13.56 tests. Total: 47 → 49.
Phase 13.59.ADF (2026-06-15): Added 1 new feature LAZY.userinfo_backcompat
(lazy-path UserInfo metadata back-compatibility, AD-3 read precedence) covered by
the 13 test_phase1359_lazy_userinfo.py tests. Total: 49 → 50.
Phase 13.58.ADF (2026-06-16): Added 1 new feature LAZY.timeseries_draw (single-tree
lazy time-series loading & lazy drawing — D1 resolver, D2 draw-surface branch scan,
D3 estimate_memory) covered by test_phase1358_lazy_timeseries.py (7),
test_phase1358_lazy_draw_invariance.py (14), test_phase1358_gallery_lazy.py (1,
env-gated). Total: 50 → 51.
Phase 13.58.ADF (2026-06-16): Added 1 new feature LAZY.subframe_draw (subframe-column
lazy draw — single-level A.col + nested/recursive A.B.col, on-demand materialization via
ensure_subframe) covered by test_lazy_subframe_column_draw, test_lazy_nested_subframe_
column_draw, and test_calibITS_subframe_column_lazy_draw. Total: 51 → 52.
Phase 13.61.ADF (2026-06-24): Added 1 new feature DISPATCH.dict_dispatch (draw-path dict
dispatch frame — transient channel projection across draw/draw_batch/draw_figures, with
structural + peak-RSS + volume-invariance memory gates and dict≡full-frame equivalence)
covered by test_phase1361_dict.py (19) and test_phase1361_memory.py (1). Total: 52 → 53.
Phase 13.61.ADF Fix (2026-06-24, BUG_20260624_lazy_bridge): Added 1 new feature
LAZY.expression_autoload (ensure_columns lazy bridge) covered by
test_bug20260624_ensure_columns.py (4). Total: 54 → 55.
"""

FEATURES = [
    # ── CORE (8) ──
    {"id": "CORE.alias_definition", "name": "Alias definition & expression evaluation", "category": "CORE",
     "test_patterns": [
         "test_alias_dataframe.py::TestAliasDataFrame",
         "test_proxy_pattern.py::TestColumnAliasPriority",
         "test_proxy_pattern.py::TestContains",
         "test_proxy_pattern.py::TestGetItem",
         "test_proxy_pattern.py::TestMethodDelegation",
         "test_proxy_pattern.py::TestIter",
         "test_proxy_pattern.py::TestLen",
         "test_proxy_pattern.py::TestEdgeCases",
         "test_fill_handling.py::TestComplexExpressions",
         "test_batch_materialization.py::TestEvalInNamespaceContextOverride",
         # Phase 13.12.ADF
         "test_I16_dtype_preservation_invariance.py",
     ]},
    {"id": "CORE.materialization", "name": "Alias materialization (single + batch)", "category": "CORE",
     "test_patterns": [
         "test_batch_materialization.py::TestBatchMaterializationPerformance",
         "test_cycle_detection.py::TestBatchOptimization",
         "test_fill_handling.py::TestMaterializationBehavior",
         "test_profiling.py",
         # Phase 13.12.ADF
         "test_I15_materialization_order_invariance.py",
     ]},
    {"id": "CORE.dependency_resolution", "name": "Dependency chain & fill_value resolution", "category": "CORE",
     "test_patterns": [
         "test_cycle_detection.py::TestCycleDetection",
         "test_cycle_detection.py::TestIndexColumnMaterialization",
         "test_dependency_tree.py",
         "test_self_referential_cycles.py",
         "test_fill_value_dependency.py",
         "test_fill_handling.py::TestFillModeDirect",
         "test_fill_handling.py::TestFillModeSafe",
         "test_fill_handling.py::TestGlobalFillConfig",
         "test_fill_handling.py::TestEdgeCases",
         "test_fill_handling.py::TestBackwardCompatibility",
         "test_fill_handling.py::TestCalibrationWorkflow",
         # Phase 13.12.ADF — fill_value propagation + missing-key NaN
         "test_I6_subframe_missing_key_invariance.py",
     ]},
    {"id": "CORE.dtypes", "name": "Dtype handling & casting", "category": "CORE",
     "test_patterns": [
         "test_alias_dataframe.py::TestDtypeRestoration",
         # Phase 13.12.ADF
         "test_I16_dtype_preservation_invariance.py",
         # BUG FIX 20260424 — dtype loss through subframe join
         "test_D1_dtype_subframe_join.py",
     ]},
    {"id": "CORE.constructor", "name": "DataFrame creation & initialization", "category": "CORE",
     "test_patterns": [
         "test_constructor_contract.py",
     ]},
    {"id": "CORE.describe", "name": "Structure & alias inspection", "category": "CORE",
     "test_patterns": [
         "test_alias_dataframe.py::TestDescribeStructure",
     ]},
    {"id": "CORE.cleanup", "name": "Column cleanup & temporary management", "category": "CORE",
     "test_patterns": [
         "test_clean_temporary.py",
     ]},
    {"id": "CORE.api_contract", "name": "Public API stability", "category": "CORE",
     "test_patterns": [
         "test_constructor_contract.py::TestAPIContract",
         "test_proxy_pattern.py::TestBackwardCompatibility",
     ]},

    # ── SUBFRAMES (6) ──
    {"id": "SUB.register", "name": "Subframe registration", "category": "SUBFRAMES",
     "test_patterns": [
         "test_alias_dataframe.py::TestAliasDataFrameWithSubframes",
         "test_alias_subframe.py::TestSubframeBasicJoin",
         "test_subframe_alias_api.py",
         # Phase 13.12.ADF
         "test_I8_subframe_alias_composition_invariance.py",
     ]},
    {"id": "SUB.join", "name": "Subframe join & column resolution", "category": "SUBFRAMES",
     "test_patterns": [
         "test_alias_subframe.py::TestMultiKeySubframeJoins",
         "test_alias_subframe.py::TestSubframeMissingKeys",
         "test_alias_subframe.py::TestSubframeEdgeCases",
         "test_alias_subframe.py::TestSubframeLazyEvaluation",
         "test_alias_subframe.py::TestSubframeVsFlattened",
         "test_alias_subframe.py::TestSubframeRoundtrip",
         "test_join_caching.py",
         "test_join_index_caching.py",
         "test_materialize_subframe_index.py",
         "test_invariance_subframe.py",
         # Phase 13.12.ADF
         "test_I14_join_index_invariance.py",
         # Phase 13.21.ADF — join index caching
         "test_J1_join_cache.py",
     ]},
    {"id": "SUB.composite_key", "name": "Composite key operations", "category": "SUBFRAMES",
     "test_patterns": [
         "test_composite_keys.py",
         # Phase 13.12.ADF — composite-key join + order independence
         "test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_2_composite_key_join_equals_pandas_merge",
         "test_I14_join_index_invariance.py::TestI14JoinIndexInvariance::test_I14_2_composite_key_order_does_not_affect_values",
     ]},
    {"id": "SUB.auto_alias", "name": "Auto-aliasing subframe columns", "category": "SUBFRAMES",
     "test_patterns": [
         "test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix",
         "test_self_referential_cycles.py::TestOnlyUnmaterializedFix",
         # Phase 13.12.ADF
         "test_I8_subframe_alias_composition_invariance.py::TestI8SubframeAliasCompositionInvariance::test_I8_3_auto_aliased_subframe_column_equals_manual_alias",
     ]},
    {"id": "SUB.clone", "name": "Clone with selection (planned)", "category": "SUBFRAMES",
     "test_patterns": []},
    {"id": "SUB.nested", "name": "Nested subframe export", "category": "SUBFRAMES",
     "test_patterns": [
         "test_alias_dataframe.py::TestExportTreeColumns",
         # Phase 13.20/13.22.ADF — export_tree roundtrip + recursive loading
         "test_E1_export_tree_roundtrip.py",
         "test_E2_export_tree_fix_a.py",
     ]},

    # ── SCHEMA (4) ──
    {"id": "SCHEMA.export_import", "name": "Schema export & import (JSON)", "category": "SCHEMA",
     "test_patterns": [
         "test_alias_data_frame_schema.py",
         "test_alias_data_frame_schema_v2.py",
         "test_schema_export_v2.py",
         "test_data_schema.py",
         "test_schema_definition_vs_record.py",
         # Phase 13.12.ADF — schema roundtrip + metadata persistence
         "test_I5_schema_roundtrip_invariance.py",
         "test_I12_metadata_persistence_invariance.py",
     ]},
    {"id": "SCHEMA.root_persistence", "name": "ROOT file persistence", "category": "SCHEMA",
     "test_patterns": [
         "test_schema_serialization.py",
         "test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip",
         # Phase 13.12.ADF — ROOT roundtrip + lazy/eager equivalence
         "test_I5_schema_roundtrip_invariance.py::TestI5SchemaRoundtripInvariance::test_I5_2_root_tree_roundtrip_preserves_full_schema",
         "test_I10_lazy_eager_invariance.py",
         "test_I17_full_pipeline_invariance.py",
     ]},
    {"id": "SCHEMA.validation", "name": "Schema validation", "category": "SCHEMA",
     "test_patterns": [
         "test_validation_display_adf.py",
         "test_schema_definition_vs_record.py::TestValidateSchemaCheckData",
         "test_schema_definition_vs_record.py::TestValidateSchemaStrict",
     ]},
    {"id": "SCHEMA.versioning", "name": "Schema versioning & migration", "category": "SCHEMA",
     "test_patterns": [
         "test_schema_definition_vs_record.py::TestSchemaChangeProtection",
         "test_schema_definition_vs_record.py::TestDeprecationWarning",
     ]},

    # ── REGISTERED_FUNCTIONS (4) ──
    {"id": "FUNC.register_function", "name": "register_function API", "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_polynomial_spec.py::TestRegisterFunction",
         "test_polynomial_spec.py::TestRepr",
         # Phase 13.12.ADF — one-arg + two-arg register_function identity
         "test_I9_registered_function_invariance.py",
     ]},
    {"id": "FUNC.polynomial", "name": "PolynomialSpec & register_polynomial_from_subframe", "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_polynomial_spec.py::TestBasisExpressions",
         "test_polynomial_spec.py::TestSchemaRoundtrip",
         "test_polynomial_spec.py::TestRootExpression",
         "test_polynomial_spec.py::TestRegisterPolynomial",
         "test_polynomial_spec.py::TestInvariancePolynomial",
     ]},
    {"id": "FUNC.evaluator", "name": "register_evaluator", "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_register_evaluator.py",
     ]},
    {"id": "FUNC.persistence", "name": "Function persistence through schema", "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_polynomial_persistence.py",
     ]},

    # ── DRAWING (5) ──
    {"id": "DRAW.execution", "name": "draw() with auto-materialization", "category": "DRAWING",
     "test_patterns": [
         "test_draw_lazy_integration.py",
         "test_draw_chain_integration.py",
         # Phase 13.12.ADF
         "test_I7_draw_path_invariance.py",
         # Phase 13.19.ADF.FIX1 — vector draw kwarg forwarding
         "test_K1_vector_draw_kwarg_diagnostic.py",
         "test_K2_vector_draw_end_to_end.py",
         # PHASE_13_35_ADF — vector kwargs alias pre-materialization
         "test_V1_vector_kwargs_alias_materialization.py",
     ]},
    {"id": "DRAW.batch", "name": "draw_batch() & draw_figures()", "category": "DRAWING",
     "test_patterns": [
         "test_draw_figures.py",
         # BUG FIX 20260420 — selection/weights alias materialization
         "test_S1_draw_selection_alias.py",
     ]},
    {"id": "DRAW.subframe_resolution", "name": "Subframe column resolution in draw", "category": "DRAWING",
     "test_patterns": [
         "test_draw_subframe_resolution.py",
         # Phase 13.12.ADF
         "test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_2_draw_subframe_column_equals_explicit_alias",
         # BUG FIX 20260426 — index col collision in draw
         "test_S5_draw_index_col_collision.py",
         # BUG_AliasDataFrame_20260517_draw_silent_swallow (commit c1f77b06)
         "test_S6_draw_subframe_expression.py",
         # BUG_AliasDataFrame_20260518_draw_subframe_alias_not_materialized (Phase A)
         "test_S10_draw_subframe_alias.py",
         # PHASE_13_36_ADF — subframe metadata propagation to drawing
         "test_X1_subframe_metadata_propagation.py",
     ]},
    {"id": "DRAW.compound_expr", "name": "Lazy materialization of compound expressions", "category": "DRAWING",
     "test_patterns": [
         "test_draw_lazy_compound.py",
         # Phase 13.12.ADF
         "test_I7_draw_path_invariance.py::TestI7DrawPathEquivalence::test_I7_1_draw_lazy_compound_expression_equals_explicit_materialize",
     ]},
    {"id": "DRAW.invariance", "name": "Draw vs materialize invariance", "category": "DRAWING",
     "test_patterns": [
         "test_draw_invariance.py",
         # Phase 13.12.ADF
         "test_I7_draw_path_invariance.py",
     ]},

    # ── COMPRESSION (3) ──
    {"id": "COMP.roundtrip", "name": "Compress/decompress roundtrip", "category": "COMPRESSION",
     "test_patterns": [
         "test_alias_dataframe.py::TestCompressionStateMachine",
         "test_alias_dataframe.py::TestAliasDataFrameCompression",
         "test_alias_dataframe.py::TestCompressionOnMissing",
         "test_invariance_compression.py",
         # Phase 13.12.ADF — linear working path only
         "test_I11_compression_working_invariance.py",
     ]},
    {"id": "COMP.selection", "name": "Compression method selection", "category": "COMPRESSION",
     "test_patterns": [
         "test_compression_pytest.py::TestCompressionSelection",
         "test_compression_pytest.py::TestCompressionConstants",
     ]},
    {"id": "COMP.monitoring", "name": "Compression quality monitoring", "category": "COMPRESSION",
     "test_patterns": [
         "test_compression_pytest.py::TestDescribeCompression",
         "test_compression_pytest.py::TestMonitorChecks",
         "test_compression_pytest.py::TestMonitorValues",
     ]},

    # ── BACKEND (3) ──
    {"id": "BACK.arrow", "name": "PyArrow compute & scatter", "category": "BACKEND",
     "test_patterns": [
         "test_arrow_compute.py",
         "test_arrow_expression.py",
         "test_arrow_scatter.py",
     ]},
    {"id": "BACK.numba", "name": "Numba JIT acceleration", "category": "BACKEND",
     "test_patterns": [
         "test_numba_acceleration.py",
         # Phase 13.12.ADF — numba vs numpy equivalence
         "test_I13_backend_equivalence_invariance.py",
     ]},
    {"id": "BACK.invariance", "name": "Backend equivalence (numpy vs arrow vs numba)", "category": "BACKEND",
     "test_patterns": [
         "test_invariance_backend.py",
         # Phase 13.12.ADF
         "test_I13_backend_equivalence_invariance.py",
     ]},

    # ── LAZY_LOADING (5) ──
    {"id": "LAZY.read_tree", "name": "Lazy branch loading from ROOT", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_lazy_loading.py",
         "test_branch_detection.py",
         # Phase 13.12.ADF
         "test_I10_lazy_eager_invariance.py",
     ]},
    {"id": "LAZY.chain", "name": "Chain loading (multiple files)", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_chain_loading.py",
         "test_invariance_load_mode.py",
     ]},
    {"id": "LAZY.materialization", "name": "Lazy subframe & alias evaluation", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_lazy_subframes.py",
         # Phase 13.12.ADF — lazy vs eager full pipeline
         "test_I10_lazy_eager_invariance.py",
     ]},
    {"id": "LAZY.userinfo_backcompat", "name": "Lazy-path UserInfo metadata back-compatibility (AD-3 precedence)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.59.ADF — BUG_20260613 lazy UserInfo gap + AD-3 read precedence
         "test_phase1359_lazy_userinfo.py",
     ]},
    {"id": "LAZY.timeseries_draw", "name": "Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.58.ADF — single-tree lazy time-series (use case 1)
         "test_phase1358_lazy_timeseries.py",        # loader mechanism + resolver/estimator gates
         "test_phase1358_lazy_draw_invariance.py",   # real lazy draw() / draw_batch / draw_figures vs eager
         "test_phase1358_lazy_calibITS.py",           # real-data lazy invariance on calibITS (committed 4 MB fixture)
         "test_phase1358_gallery_lazy.py",            # time-series gallery double-run (env-gated)
     ]},
    {"id": "LAZY.subframe_draw", "name": "Subframe-column lazy draw (single-level A.col + nested A.B.col; on-demand materialization via ensure_subframe + recursive chain walk)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.58.ADF — subframe-column lazy draw (single-level + nested/recursive)
         "test_phase1358_lazy_draw_invariance.py::test_lazy_subframe_column_draw",
         "test_phase1358_lazy_draw_invariance.py::test_lazy_nested_subframe_column_draw",
         "test_phase1358_lazy_calibITS.py::test_calibITS_subframe_column_lazy_draw",
     ]},
    {"id": "LAZY.alias_autoload", "name": "Alias resolution auto-loads lazy branches (materialize_aliases / validate_aliases / describe_aliases bridge to the lazy reader; LAZY status)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.60.ADF — alias <-> lazy-branch bridge (U-1 first instance)
         "test_phase1360_alias_lazy_bridge.py::test_T1_materialize_over_unloaded_branch_equals_eager",
         "test_phase1360_alias_lazy_bridge.py::test_T2_exact_load_decoys_not_loaded",
         "test_phase1360_alias_lazy_bridge.py::test_T3_validate_and_describe_lazy_not_broken",
         "test_phase1360_alias_lazy_bridge.py::test_T4_genuine_missing_still_reported_and_registered_fn",
         "test_phase1360_alias_lazy_bridge.py::test_T7_chained_alias_transitive_autoload_equals_eager",
         "test_phase1360_alias_lazy_bridge.py::test_T8_dtype_bearing_alias_lazy_equals_eager",
     ]},
    {"id": "LAZY.expression_autoload",
     "name": "Expression/column lazy autoload via ensure_columns() — bridges df.eval()/direct-access paths on a lazy ADF (get_required_branches → ensure_branches; branches-only, subframe-name + dotted-ref filtered; eager no-op)",
     "category": "LAZY_LOADING",
     "test_patterns": [
         "test_bug20260624_ensure_columns.py",
     ]},

    # ── FIT_REGISTRATION (2) ──
    {"id": "FIT.registration", "name": "Fit metadata storage & retrieval", "category": "FIT_REGISTRATION",
     "test_patterns": [
         "test_register_fit_result.py",
     ]},
    {"id": "FIT.visualization", "name": "Fit summary visualization", "category": "FIT_REGISTRATION",
     "test_patterns": [
         "test_register_fit_result.py::TestDrawFitSummaryBasic",
         "test_register_fit_result.py::TestDrawFitSummaryOptions",
         "test_validation_display_adf.py::TestDrawFitSummaryIntegration",
     ]},

    # ── RDATAFRAME (2) ──
    {"id": "RDF.export", "name": "Export to RDataFrame", "category": "RDATAFRAME",
     "test_patterns": [
         "test_AliasDataFrameRDF.py",
         "test_rdf_integration.py",
         "test_rdf_integration_final.py",
         "test_rdf_real_data.py",
     ]},
    {"id": "RDF.composite", "name": "RDataFrame composite key support", "category": "RDATAFRAME",
     "test_patterns": [
         "test_ttree_draw_subframe.py",
     ]},

    # ── INVARIANCE (1) ──
    {"id": "INV.cross_module", "name": "Cross-module invariance tests", "category": "INVARIANCE",
     "test_patterns": [
         "test_invariance_smoke.py",
         # Phase 13.12.ADF — full integration pipeline
         "test_I17_full_pipeline_invariance.py",
     ]},

    # ── Phase 13.18.ADF — Regression Metadata Bridge (3 new features) ──
    {"id": "FUNC.regression_metadata",
     "name": "Regression metadata registration & update",
     "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_R1_metadata_persistence_invariance.py",
         "test_R2_recalibration_invariance.py",
     ]},
    {"id": "FUNC.evaluator_from_metadata",
     "name": "Bridge: metadata → evaluator binding",
     "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_R3_missing_bin_safety_invariance.py",
         "test_R4_registration_contract_invariance.py",
         "test_R5_remap_correctness_invariance.py",
     ]},
    {"id": "FUNC.regression_persistence",
     "name": "Regression metadata schema roundtrip",
     "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_R1_2_evaluator_roundtrip_invariance.py",
     ]},

    # ── Phase 13.23.ADF — New features (3) ──
    {"id": "SUB.multilevel",
     "name": "Multi-level dotted subframe resolution (A.B.C.val)",
     "category": "SUBFRAMES",
     "test_patterns": [
         "test_N1_recursive_subframe_invariance.py",
     ]},
    {"id": "CORE.dependency_tree",
     "name": "Dependency tree output (text/html/list)",
     "category": "CORE",
     "test_patterns": [
         "test_T1_dependency_tree.py",
         "test_dependency_tree.py",
     ]},
    {"id": "CORE.invalidation",
     "name": "Alias invalidation on expression redefine",
     "category": "CORE",
     "test_patterns": [
         "test_V1_alias_invalidation.py",
     ]},
    # ── DISPATCH (2) — PHASE_13_55_ADF / PHASE_13_56_ADF ──
    {"id": "DISPATCH.adf_routing",
     "name": "adf.draw/draw_figures route through DFDraw.draw() (auto pre-resolution, overlay strings, type aliases, 3-var profile promotion)",
     "category": "DISPATCH",
     "test_patterns": [
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup1TypeCoverage",
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup2KwargLocks",
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup4RoutingRegression",
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup6ProfilePromotion",
         "test_phase_13_56_adf_post_audit.py::TestG6BatchShims",
         "test_phase_13_56_adf_post_audit.py::TestR1ThreeLevelFacetLock",
         "test_phase_13_56_adf_post_audit.py::TestH1DrawHelp",
     ]},
    {"id": "DISPATCH.error_visibility",
     "name": "Batch-surface error visibility (on_error='raise' defaults; A-10/E-3/E-4 guards; draw_fit_summary documented exception)",
     "category": "DISPATCH",
     "test_patterns": [
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup3OnError",
         "test_phase_13_55_adf_dispatch_audit.py::TestGroup5DrawBatchOnError",
         "test_phase_13_56_adf_post_audit.py::TestG1Profile2dGuard",
         "test_phase_13_56_adf_post_audit.py::TestG2FacetByGuard",
         "test_phase_13_56_adf_post_audit.py::TestG3SelectionAliasMatrix",
         "test_phase_13_56_adf_post_audit.py::TestG5AstypeTypeTokens",
         "test_phase_13_56_adf_post_audit.py::TestG7CoverageLocks",
     ]},
    # ── DISPATCH (1) — PHASE_13_61_ADF (transient draw channels / dict dispatch) ──
    {"id": "DISPATCH.dict_dispatch",
     "name": "Draw-path dict dispatch frame: draw()/draw_batch()/draw_figures() hand dfdraw only the needed columns (get_required_branches ∪ materialized alias names ∪ subframe index cols); structural column-count gate + peak-RSS + volume-invariance memory gates + dict≡full-frame equivalence (AC-1/1a/1b incl. subframe single+multi-level) + loud no-silent-full-frame fallback",
     "category": "DISPATCH",
     "test_patterns": [
         "test_phase1361_dict.py",
         "test_phase1361_memory.py",
     ]},
]
