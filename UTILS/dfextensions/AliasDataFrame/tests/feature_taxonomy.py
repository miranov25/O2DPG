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
Phase 13.77.ADF A5 closure (2026-09-02): Added 4 acceptance-catalogue features: TESTING.phase13_77_harness, INV.draw_surface_consistency, INV.eager_lazy_slot_symmetry, and INV.realdata_acceptance. Exact collected-node ownership covers all 248 current PHASE_13_77 harness nodes; A3/A4/A5 nodes are marked invariance. Total features: 64 → 68.
Phase 13.79.ADF Part A registration (2026-09-03): Added DRAW.slot_grid for the architect-ratified B-SMOKE/B-INVARIANCE systematic surface and TESTING.capability_matrix_index for the shared Markdown/HTML/JSON diagnostic-index tooling.
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
    {"id": "CORE.dependency_resolution", "name": "Dynamic dependency resolution — deferred unresolved definitions, dependency chains, fill_value propagation, late namespace completion, and use-time resolution", "category": "CORE",
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
         "test_phase_13_76_v12_dynamic_alias_contract.py::TestV12DependencyResolutionContract",
         "test_phase_13_76_v12_history_invariance.py::TestV12DeferredRetryInvariance",
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
    {"id": "CORE.describe", "name": "Structure & alias inspection — resolved/unresolved logical definitions and validation visibility", "category": "CORE",
     "test_patterns": [
         "test_alias_dataframe.py::TestDescribeStructure",
         "test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InspectionContract",
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
    {"id": "SUB.register", "name": "Subframe registration & replacement — registration/re-registration semantics, current logical ownership, and dependent-state invalidation obligations", "category": "SUBFRAMES",
     "test_patterns": [
         "test_alias_dataframe.py::TestAliasDataFrameWithSubframes",
         "test_alias_subframe.py::TestSubframeBasicJoin",
         "test_subframe_alias_api.py",
         # Phase 13.12.ADF
         "test_I8_subframe_alias_composition_invariance.py",
         # PHASE_13_76_ADF — interactive replacement/invalidation contract
         "test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract::test_ev_3_subframe_reregistration_invalidates_sourced_alias",
         "test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration",
         "test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance::test_o1_subframe_reregistration_final_state_matches_fresh_instance",
     ]},
    {"id": "SUB.join", "name": "Subframe join & column resolution — key matching, missing-key/declared-neutral fill semantics, conditional calibration composition, and cache/order invariance", "category": "SUBFRAMES",
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
         "test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationScope",
         "test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration",
         "test_fill_handling.py::TestFillModeDirect::test_V3_2_unmatched_key_gets_declared_neutral_in_masked_correction",
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
    {"id": "ALIAS.source_scoped",
     "name": "Source-scoped alias resolution (PHASE_13_73) — add_alias with source=<subframe> binds BARE fit formulas (e.g. GB meta['formulas']) to a registered subframe by rewriting bare names to Subframe.name. Resolution is AST-based (names are tokens, not text), so the substring-collision class fixed in 13.72 is structurally impossible. Per-name order: Attribute/Call-func untouched; source index_columns exempt (R1a); present in BOTH source and parent -> loud shadow refusal (R1); in source -> qualify; in parent-universe -> leave bare; otherwise -> refuse at registration (R2). The parent-universe includes lazy available-but-unloaded branches, struct members and registered functions (F-1), so binding works on a lazy frame before any branch is loaded. Batch binding is a loop over add_alias (PHASE_13_73_FIX removed the add_aliases helper: it could not express per-alias dtypes and was a second entry point for one concept). The rewritten alias stays user-readable and source-qualified; the existing subframe-join path performs the join (no second evaluation route). source=None is bit-identical to pre-13.73. Composes with 13.70 vector aliases.",
     "category": "ALIAS",
     "test_patterns": [
         "test_phase_13_73_source_scoped_alias.py",
     ]},
    {"id": "DIAGNOSTICS.lazy_state",
     "name": "User-facing lazy-state diagnostic (PHASE_13_71) — adf.describe_lazy() prints (or returns via as_dict) the main lazy reader's entries, available/loaded branches, DataFrame columns, and available-but-not-loaded set, plus a per-lazy-subframe block (available/loaded counts + index columns from _subframe_lazy_config); diagnostic-only with NO loading or materialization side effects; bounded output via max_items with a '... (+N more)' suffix; tolerant of LazyTreeReader/LazyChainReader attribute differences via getattr defaults; reports the subframe block even when the main frame is eager. Not scope this phase: loading/ensuring branches, HTML/JSON output, rich repr.",
     "category": "DIAGNOSTICS",
     "test_patterns": [
         "test_phase_13_71_describe_lazy.py",
     ]},
    {"id": "CORE.vector_alias",
     "name": "Vector (group) aliases & multi-output prediction (PHASE_13_70) — add_alias(list-of-names, tuple-or-2D expression, dtype=list) defines k scalar member columns from ONE expression evaluated ONCE via the function-generic group engine (D0, shared with register_model) and split by slot; accepted shapes are a k-tuple/list of 1-D or an (n_rows,k) 2-D ndarray (V-6); evaluate-once across siblings with the same cache/invalidation contract as ML prediction (frame length, __setitem__ write hook, release/reload, re-registration); per-name collision refused across all namespaces (CF-5); dtype-list length and arity mismatches are loud errors (CF-8/V-6); single-name list = ordinary alias. Not scope this phase: per-row vector members, dot-sugar.",
     "category": "CORE",
     "test_patterns": [
         "test_phase_13_70_vector_alias.py",
     ]},
    {"id": "FUNC.ml_model",
     "name": "ML model registration, lazy prediction alias, persistence (embed/external/load-from-ROOT), integrity, multi-output cache, chain recovery (PHASE_13_69) — register_model register+alias in one call; ONNX canonical + native xgboost-JSON path; format='auto' byte-sniff (ROOT->JSON->ONNX); single float32 input tensor in feature order; multi-output = sibling aliases sharing ONE evaluation via a prediction cache invalidated by the __setitem__ write-event hook / release / re-registration; embed (default, ADF_ML/ blob+descriptor via uproot, UserInfo untouched) + external relative-path + load-from-ROOT persistence, MD5-verified; missing runtime -> loud refuse. Not scope: training, CCDB, GPU, full RNTuple verification, subframe-column inputs (Phase-1 deferral).",
     "category": "REGISTERED_FUNCTIONS",
     "test_patterns": [
         "test_phase_13_69_ml_model.py",
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
    {"id": "DRAW.subframe_resolution", "name": "Subframe resolution in draw — expression-slot discovery, owner-qualified materialization/cleanup, request-level/spec-slot attribution, per-public-call evidence isolation, and terminal plan/state reconciliation", "category": "DRAWING",
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
         "test_phase_13_76_v12_reconciliation_contract.py::TestV12B3ReconciliationContract",
         "test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix",
         "test_phase_13_76_v12_history_invariance.py::TestV12RequestHistoryInvariance",
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
    {"id": "DRAW.slot_grid",
     "name": "Systematic draw slot symmetry grid",
     "description": "Checks every architect-supported draw slot across column, alias, expression, subframe and struct forms in eager/lazy modes. Smoke records reachability and live gaps; invariance compares executable cells against an independent plain-data reference and checks eager/lazy equivalence.",
     "category": "DRAWING",
     "surface": {
         "slots": 8,
         "forms": 5,
         "base_modes": 2,
         "base_cells": 80,
         "bounded_mixed_mode_rows": 8,
         "declared_cells": 88,
         "evidence": ["B-SMOKE", "B-INVARIANCE"],
     },
     "contract_file": "tests/phase_13_79_slot_grid_contract.json",
     "test_patterns": [
         "test_phase_13_79_slot_grid.py",
         "test_phase_13_79_slot_grid_invariance.py",
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
    {"id": "LAZY.materialization", "name": "Lazy subframe & alias evaluation — on-demand physical/logical resolution across eager/lazy parent-child modes, no-metadata operation, and raise vs warn/skip resolution policies", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_lazy_subframes.py",
         # Phase 13.12.ADF — lazy vs eager full pipeline
         "test_I10_lazy_eager_invariance.py",
         "test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix",
     ]},
    {"id": "LAZY.userinfo_backcompat", "name": "Lazy-path UserInfo metadata back-compatibility (AD-3 precedence)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.59.ADF — BUG_20260613 lazy UserInfo gap + AD-3 read precedence
         "test_phase1359_lazy_userinfo.py",
     ]},
    {"id": "LAZY.chain_metadata", "name": "Chain lazy metadata recovery (PHASE_13_67) — first-file UserInfo canonical, applied by DEFAULT (aliases+dtypes+compression, 0a); raise on cross-file incompatibility (0b); union/intersection -> error by default via metadata_conflict policy (parametrizable, off-switch in message); lazy application loads zero columns (INV-1); D4 pre-sized chain frame; names_only is a valid sparse case; real public-API read_chain_lazy recovery (alias+dtype+eval) verified on export_tree fixtures; subframe parity test runs on full-metadata fixtures, skips on names-only slim", "category": "LAZY_LOADING",
     "test_patterns": [
         # PHASE_13_67_ADF — chain metadata comparator/apply (real adf_metadata structure)
         "test_phase_13_67_chain_metadata.py",
         "test_phase_13_67_chain_recovery_public_api.py",
         # real-data back-compat: lazy applies same UserInfo as eager; subframe lazy==eager
         "test_phase1358_lazy_calibITS.py::test_calibITS_lazy_applies_same_metadata_as_eager",
         "test_phase1358_lazy_calibITS.py::test_calibITS_lazy_value_parity_with_metadata_eager",
         "test_phase1358_lazy_calibITS.py::test_calibITS_subframe_lazy_eager_value_parity",
     ]},
    {"id": "LAZY.timeseries_draw", "name": "Single-tree lazy time-series loading & lazy drawing (D1 resolver + D2 draw-surface branch scan + D3 estimate_memory)", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.58.ADF — single-tree lazy time-series (use case 1)
         "test_phase1358_lazy_timeseries.py",        # loader mechanism + resolver/estimator gates
         "test_phase1358_lazy_draw_invariance.py",   # real lazy draw() / draw_batch / draw_figures vs eager
         "test_phase1358_lazy_calibITS.py",           # real-data lazy invariance on calibITS (committed 4 MB fixture)
         "test_phase1358_gallery_lazy.py",            # time-series gallery double-run (env-gated)
     ]},
    {"id": "LAZY.subframe_draw", "name": "Subframe-column lazy draw — single/nested qualified references, eager/lazy parent-child compositions, structural join-key setup, post-load aliases, and on-demand ensure_subframe resolution", "category": "LAZY_LOADING",
     "test_patterns": [
         # Phase 13.58.ADF — subframe-column lazy draw (single-level + nested/recursive)
         "test_phase1358_lazy_draw_invariance.py::test_lazy_subframe_column_draw",
         "test_phase1358_lazy_draw_invariance.py::test_lazy_nested_subframe_column_draw",
         "test_phase1358_lazy_calibITS.py::test_calibITS_subframe_column_lazy_draw",
         "test_phase_13_76_v12_subframe_mode_matrix.py::TestR10SubframeModeMatrix",
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
         "test_phase_13_62_adf_s1.py",
          "test_phase_13_75_lazy_struct_repair.py",
     ]},

    {"id": "LAZY.release",
     "name": "Explicit lazy-branch/struct release (PHASE_13_68) — release_branches()/release_struct() symmetric evict: drop frame columns AND unbook the physical branch(es) on the lazy reader so a later access re-reads from file; struct members translated internal->physical forward from the registry; all-or-nothing loud refuse for eager frames (DD-alpha), aliases (DD-gamma -> dematerialize), written/__file_idx__ non-branch names (DD-beta), parent-side subframe join keys (DD-delta), and names in a materialized alias's dependency closure (C-6); memory_policy surface accepts 'keep' only ('bounded'/'drop' reserved); purely additive, no automatic eviction",
     "category": "LAZY_LOADING",
     "test_patterns": [
         "test_phase_13_68_release.py",
     ]},

    {"id": "SUBFRAME.asymmetric_join_keys",
     "name": "Asymmetric subframe join keys (PHASE_13_65) — register_subframe(right_index_columns=[...]) lets parent/child join columns differ in name (pandas left_on/right_on); name-aware across all three _compute_join_indices paths (single-col numba, Phase 8c multi-col linearization via rename-before-linearize, merge fallback); right_index_columns=None is byte-identical to the prior same-name behavior; schema-persisted with absent-field back-compat",
     "category": "SUBFRAME",
     "test_patterns": [
         "test_phase_13_65_adf_asymmetric_keys.py",
     ]},

    {"id": "OBJECT.struct_1to1",
     "name": "1:1 struct/object branch support (PHASE_13_66) — ROOT struct members (parent/member) usable via dot grammar (dedxTPC.dEdxTotIROC); three-name mapping (physical slash / internal member__struct / logical dot, anchor 0i); reference-driven load with A-1 rename-on-load handling bare-leaf or slash reader keys; public adf.eval() with Step-0 syntax gate; struct-aware across the 7 analysis surfaces, get_required_branches (physical form), the 5 dispatch sites, and all 3 draw surfaces; alias-over-struct (direct + nested) via _get_structs_for_aliases + _do_materialize hook; auto-detection with scalar/jagged guard (never auto-flatten 1:N, anchor 0h); schema-persisted (export + apply) with absent-key back-compat",
     "category": "OBJECT",
     "test_patterns": [
         "test_phase_13_66_adf_struct_foundation.py",
          "test_phase_13_75_lazy_struct_repair.py",
     ]},

    {"id": "WRITE.column_assignment",
     "name": "Direct column write-through via adf[col] = value (PHASE_13_62 Stage 2a / Fix A) — writes to the frame and syncs the lazy reader's loaded_branches so a hand-added column is present and never re-requested from the TTree; supports numpy/Series/list/scalar (awkward via explicit conversion); non-string key raises; bad shape raises before bookkeeping; adf.aliases immutability (_ReadOnlyAliasDict) unaffected",
     "category": "CORE",
     "test_patterns": [
         "test_phase_13_62_adf_s2a.py",
         "test_proxy_pattern.py",
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
     "name": "Interactive invalidation — alias redefinition, final-state/fresh-instance equivalence, materialization-history independence, subframe replacement, and rejected-mutation state integrity",
     "category": "CORE",
     "test_patterns": [
         "test_V1_alias_invalidation.py",
         "test_phase_13_76_v12_dynamic_alias_contract.py::TestV12InvalidationContract",
         "test_phase_13_76_v12_alias_mutation_falsifiers.py",
         "test_phase_13_76_v12_calibration_scope.py::TestV12CalibrationRecalibration",
         "test_phase_13_76_v12_history_invariance.py::TestV12HistoryInvariance",
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
          "test_phase_13_75_lazy_struct_repair.py",
     ]},

    # ── PHASE_13_77_ADF A5 closure — acceptance catalogue (4) ──
    {"id": "TESTING.capability_matrix_index",
     "name": "Capability Matrix diagnostic index tooling",
     "description": "Proves the shared Capability Matrix semantic model, Markdown/HTML parity, AI-readable JSON export, node source locators, phase/focused-run evidence and reviewer-packet custody rules.",
     "category": "TESTING",
     "test_patterns": [
         "test_phase_13_76_capability_matrix_tooling.py",
     ]},
    {"id": "TESTING.phase13_77_harness",
     "name": "PHASE_13_77 acceptance-harness integrity — CaseSpec/FigureContract registry, comparator/gate fail-closed behavior, manifest reconciliation, environment gating, and anti-false-green controls (A1+A2)",
     "category": "TESTING",
     "test_patterns": [
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_raises_on_unresolvable_path",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_rejects_array_declared_flat",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_accessor_resolves_flat",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_never_returns_a_silent_default",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_bad_figure_key",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_bad_plot_index",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_wrong_batch_case_key",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_raises_on_wrong_return_shape",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw_batch]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_adapter_unwraps_each_public_surface[draw_figures]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_consistency_case_passes_across_three_surfaces",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_correctness_case_agrees_with_numpy",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_error_contract_draw_figures_refuses_facet_by",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_faceted_stats_have_a_different_shape",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_mutation_corrupted_surface_cannot_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_1_case_comparing_nothing_cannot_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_2_anchor_is_immune_to_product_mutation",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_environment_gated_skip_is_still_legal",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_mandatory_skip_fails_closed",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p0_3_registry_rejects_a_mandatory_case_that_can_only_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_1_unknown_comparator_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_2_unimplemented_oracle_source_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_3_error_contract_failure_always_gates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_p1_4_manifest_carries_the_comparison_contract",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_accepts_a_clean_case",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_duplicate_case_id",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_floating_comparator_without_rationale",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over0-no claim]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over1-owner_on_failure]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over2-sampled-lazy]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over3-known_bug_id]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over4-conflated]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_registry_rejects_violation[over5-declares no observable]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_always_gates_on_invalid_fixture",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_ignores_a_failing_known_bug",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_strict_exit_code_is_non_zero_on_mandatory_failure",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_undeclarable_observable_is_invalid_fixture_not_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[accepted_envelope]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_group_count]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_panels]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[expected_traces]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[panel_roles]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[primary_comparison]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[proof_kind]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_figure_contract_requires_each_of_its_nine_fields[residual_definition]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_footer_and_manifest_share_one_source",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_footer_carries_the_three_required_blocks",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_manifest_carries_the_new_contract_fields",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over0-no setup_contract]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over1-no preconditions]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over2-no figure_contract]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over3-applicability_reason]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v03_registry_requires_the_ratified_minimum[over4-always applicable]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_aligned_figure_contract_is_accepted",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[FAMILY_MUTATION:hist-anchor-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[GLOBAL_MUTATION:-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[GLOBAL_MUTATION:M1-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_every_machine_gated_case_has_a_falsification_owner[corrupt one surface -> FAIL-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over0-case_ids is empty]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over1-case_ids]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_figure_contract_may_not_contradict_its_case[over2-proof_kind]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_inapplicable_case_never_invokes_the_product",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v04_inapplicable_short_circuits_every_runner",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_applicable_failure_gates_whatever_the_gate_class",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_audit_detects_a_planted_orphan",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_correctness_runner_rejects_a_stats_label",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_declared_source_must_match_the_executed_path",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_future_staged_fields_are_recorded_with_their_stage",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_matching_source_labels_still_execute",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_no_caseSpec_field_is_declared_but_unread",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v05_unavailable_environment_may_still_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_applicable_skip_always_gates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_audit_detects_a_field_with_no_reader",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_audit_is_derived_not_declared",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_environment_blocked_may_not_be_applicable",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-DIAGNOSTIC]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-FAIL]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-INVALID_FIXTURE]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-PASS]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[False-SKIP]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-DIAGNOSTIC]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-FAIL]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-INVALID_FIXTURE]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-PASS]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_every_gate_state_has_a_defined_verdict[True-SKIP]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_manifest_carries_title_and_case_schema_version",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_unavailable_environment_may_still_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v06_unknown_status_fails_closed",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_audit_detects_a_name_colliding_field",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_audit_is_receiver_narrowed",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_complete_declared_set_still_passes",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_coverage_gaps_names_the_dropped_case",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_declared_case_with_no_result_gates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_duplicate_result_gates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-DIAGNOSTIC-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-FAIL-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-INVALID_FIXTURE-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-PASS-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[False-SKIP-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-DIAGNOSTIC-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-FAIL-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-INVALID_FIXTURE-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-PASS-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-SKIP-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_gate_matrix_asserts_the_expected_verdict[True-SOMETHING_NEW-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_iterating_a_field_does_not_promote_its_element",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_no_per_case_schema_version",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v07_result_for_an_undeclared_case_gates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseResult-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpec | None-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpec-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[CaseSpecView-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[FakeCaseSpec-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[NotACaseSpec-False]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_annotation_match_is_exact_not_substring[Sequence[CaseSpec]-True]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_complete_run_still_passes",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_duplicate_declared_id_is_caught_by_reconcile",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_empty_declared_set_is_refused_at_both_doors",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_gate_and_coverage_derive_from_one_authority",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_missing_declared_case_appears_in_the_manifest",
         "test_phase_13_77_realdata_invariance_harness.py::test_a1_v08_receiver_evidence_is_scope_local",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_01_scalar_exact_pass_and_fail",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_02_scalar_close_atol_boundary",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_03_scalar_close_rtol_boundary",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_04_scalar_nan_and_infinity_rules",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_05_scalar_refuses_array_nonnumeric_and_unknown",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_06_array_shape_is_exact",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_07_array_exact_reports_mismatch_coordinates",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_08_array_close_elementwise_and_nan",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_09_array_close_rejects_non_numeric",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_10_a1_comparator_api_remains_compatible",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_11_tolerance_exact_contract_is_valid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_12_tolerance_close_contract_is_valid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_13_negative_atol_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_14_negative_rtol_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_15_nan_tolerance_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_16_infinite_tolerance_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_17_exact_cannot_carry_ignored_tolerance",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_18_close_zero_tolerance_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_19_close_requires_rationale",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_20_unknown_comparator_is_refused",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_21_compare_observable_uses_declared_scalar_contract",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_22_compare_observable_uses_declared_array_contract",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_23_comparison_evidence_is_json_ready",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_24_consistency_runner_records_structured_evidence",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_25_correctness_runner_records_structured_evidence",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_26_pass_without_comparison_fails_strict_gate_and_manifest",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_27_intentional_numeric_corruption_changes_gate_zero_to_one",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_28_direct_exact_comparison_rejects_nonzero_tolerance",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_29_invalid_comparator_is_rejected_before_shape_comparison",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_30_extended_precision_scalar_mismatch_survives_close_and_gate",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_31_extended_precision_array_mismatch_survives_close",
         "test_phase_13_77_realdata_invariance_harness.py::test_a2_32_close_refuses_nonfloating_numeric_families_without_coercion",
     ]},
    {"id": "INV.draw_surface_consistency",
     "name": "Same-spec numerical consistency across draw(), draw_batch(), and draw_figures(), including explicit supported-surface refusals and closure reconciliation (A3)",
     "category": "INVARIANCE",
     "test_patterns": [
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_01_hist_case_declares_one_spec_for_all_three_surfaces",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_02_hist_same_spec_passes_draw_draw_batch_draw_figures",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_03_draw_figures_is_an_executed_surface_not_a_documented_exception",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_04_profile_case_declares_one_spec_for_all_three_surfaces",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_05_profile_same_spec_passes_draw_draw_batch_draw_figures",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_06_surface_stats_corruption_changes_strict_gate_zero_to_one",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_07_groupby_case_declares_one_spec_and_group_resolved_observables",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_08_groupby_same_spec_passes_draw_draw_batch_draw_figures",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_09_profile_data_missing_column_fails_loudly",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_10_group_specific_mismatch_is_not_hidden_by_global_stats",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_11_facet_case_declares_supported_surfaces_and_separate_refusal_contract",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_12_facet_same_spec_passes_supported_surfaces_and_refuses_draw_figures",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_13_facet_specific_mismatch_reaches_strict_gate",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_14_subframe_case_declares_one_spec_and_keyed_observables",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_15_subframe_same_spec_passes_draw_draw_batch_draw_figures",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_16_subframe_specific_mismatch_reaches_strict_gate",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_17_selection_vector_case_declares_branch_resolved_observables",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_18_selection_vector_same_spec_passes_all_three_surfaces",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_19_selection_vector_branch_mismatch_is_not_hidden_by_derived_value",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_20_profile_bin_case_and_histogram_disposition_are_explicit",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_21_sparse_profile_bins_match_all_three_surfaces",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_22_profile_bin_error_mismatch_reaches_strict_gate",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_23_closure_reconciliation_covers_every_required_family_without_orphans",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_24_closure_reconciliation_fails_closed_on_missing_family_or_histogram_upgrade",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_25_complete_a3_manifest_carries_closure_record",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_26_closure_contract_map_blocks_legal_required_case_drift",
         "test_phase_13_77_realdata_invariance_harness.py::test_a3_27_missing_family_manifest_persists_blocked_closure",
     ]},
    {"id": "INV.eager_lazy_slot_symmetry",
     "name": "Eager/lazy expression-slot causality and exact dependency-load symmetry across selection/expression/weights/group_by/facet/vector/subframe compositions (A4)",
     "category": "INVARIANCE",
     "test_patterns": [
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_01_selection_slot_contract_is_executable_and_manifest_visible",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_02_selection_slot_both_proves_materialization_and_exact_lazy_loads",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_03_m2_preload_contamination_is_invalid_fixture_not_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_04_scalar_catalogue_and_runner_binding_are_machine_authoritative",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-COMPOUND-EXPR-01-expected_loaded4]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-EXPR-01-expected_loaded0]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-FACET-BY-01-expected_loaded3]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-GROUP-BY-01-expected_loaded2]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads[I3-WEIGHTS-01-expected_loaded1]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_10_source_contract_is_checked_before_observable_path_resolution",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_11_vector_catalogue_and_refusal_contracts_are_machine_visible",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_12_13_vector_slots_prove_eager_materialization_and_exact_lazy_loads[I3-SELECTION-VECTOR-01-expected_loaded0]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_12_13_vector_slots_prove_eager_materialization_and_exact_lazy_loads[I3-WEIGHTS-VECTOR-01-expected_loaded1]",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_14_15_subframe_vector_refusals_hold_in_eager_and_lazy_modes",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_16_complete_catalogue_and_contract_reconciliation_is_bidirectional",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_17_slot_alias_exclusivity_is_machine_locked_and_second_slot_fails",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_18_eager_target_only_materialization_is_proven_with_shared_dependency_alias",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_19_shared_physical_dependency_all_alias_materialization_false_green_is_caught",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_20_subframe_scalar_causality_and_vector_refusal_ownership_are_complete",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_21_historical_closure_ledger_has_no_implicit_obligation_and_blocks_on_loss",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_22_closure_ledger_status_and_owner_semantics_are_locked",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_23_refusal_contract_locks_exact_bug_identity_and_proof_scope",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_24_qualified_reference_matching_is_token_exact_and_near_name_replacement_blocks",
         "test_phase_13_77_realdata_invariance_harness.py::test_a4_25_duplicate_historical_ledger_ids_block_and_persist",
     ]},
    {"id": "INV.realdata_acceptance",
     "name": "Deterministic real-data/gallery acceptance and state invariance — full-stack composition, provenance, environment contracts, G7.32/G7.33/G7.34 evidence, GB prepared-state reuse, and logical-state mutation falsifiers (A5)",
     "category": "INVARIANCE",
     "test_patterns": [
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_01_full_stack_case_is_bounded_and_registry_valid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_02_full_stack_matches_independent_oracle_in_eager_and_lazy_modes",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_03_independent_oracle_catches_one_lazy_group_bin_corruption",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_04_preloaded_unrelated_branch_is_invalid_fixture_not_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_05_unavailable_realdata_environment_skips_without_running_product",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_06_realdata_case_pins_eager_fraction_20pct_seed42",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_07_realdata_runner_records_actual_sample_identity_and_g7_evidence",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_08_realdata_gate_persists_sample_provenance_in_manifest",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_09_applicable_optional_gallery_none_is_fail_not_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_10_applicable_g7_exception_fails_closed_and_wrong_seed_is_invalid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_11_missing_required_gallery_callable_is_not_a_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_12_unexpected_gallery_import_exception_is_not_environment_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_13_finite_bookkeeping_cannot_hide_all_nan_profile_y",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_14_sample_monkeypatch_restored_when_build_fails",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_15_lazy_full_case_is_bounded_and_registry_valid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_16_lazy_full_g7_records_real_lazy_branch_expansion",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_17_lazy_full_eager_in_disguise_is_invalid_fixture",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_18_lazy_full_requires_on_demand_branch_expansion",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_19_lazy_full_forbids_sampling_and_restores_sample",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_20_realdata_lazy_setup_exact_timems_blocker_is_error_contract_pass",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_21_realdata_lazy_setup_wrong_key_does_not_false_green",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_22_realdata_lazy_setup_success_forces_contract_review",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_23_g7_33_case_is_bounded_eager_fraction_and_registry_valid",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_24_g7_33_records_sample_gb_subframe_predicted_and_public_evidence",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_25_g7_33_nonfinite_predicted_column_fails_closed",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_26_g7_33_missing_calibbias1_subframe_fails_closed",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_27_g7_33_optional_none_is_fail_not_skip",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_28_g7_33_sequence_stats_profile_means_are_valid_public_evidence",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_29_g7_33_sequence_stats_bookkeeping_cannot_hide_nan_profiles",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_30_g7_34_case_declares_two_real_state_consistency_observables",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_31_g7_34_reuses_prepared_state_and_executes_two_comparisons",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_32_g7_34_refit_attempt_is_poisoned_and_fails",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_33_g7_34_predicted_state_mutation_fails_consistency",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_34_g7_34_calibbias1_mutation_fails_consistency",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_35_g7_34_failure_restores_exact_calibration_binding",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_36_g7_34_logical_state_case_declares_four_exact_observables",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_37_g7_34_healthy_logical_state_executes_four_comparisons",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_38_g7_34_alias_definition_mutation_fails",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_39_g7_34_subframe_index_definition_mutation_fails",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_40_g7_34_parent_structure_mutation_fails_and_fingerprint_is_deterministic",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_41_definition_digest_ignores_only_export_created_at",
         "test_phase_13_77_realdata_invariance_harness.py::test_a5_42_phase_13_77_capability_taxonomy_registration_is_exact",
     ]},

]
