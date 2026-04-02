"""
Feature Taxonomy — AliasDataFrame

41 features (PHASE_13_11_B approved). Patterns match actual pytest node IDs.
Updated from .pytest_report.json (2026-04-02, 1456 tests).
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
     ]},
    {"id": "CORE.materialization", "name": "Alias materialization (single + batch)", "category": "CORE",
     "test_patterns": [
         "test_batch_materialization.py::TestBatchMaterializationPerformance",
         "test_cycle_detection.py::TestBatchOptimization",
         "test_fill_handling.py::TestMaterializationBehavior",
         "test_profiling.py",
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
     ]},
    {"id": "CORE.dtypes", "name": "Dtype handling & casting", "category": "CORE",
     "test_patterns": [
         "test_alias_dataframe.py::TestDtypeRestoration",
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
     ]},
    {"id": "SUB.composite_key", "name": "Composite key operations", "category": "SUBFRAMES",
     "test_patterns": [
         "test_composite_keys.py",
     ]},
    {"id": "SUB.auto_alias", "name": "Auto-aliasing subframe columns", "category": "SUBFRAMES",
     "test_patterns": [
         "test_self_referential_cycles.py::TestAutoAliasSubframeCycleFix",
         "test_self_referential_cycles.py::TestOnlyUnmaterializedFix",
     ]},
    {"id": "SUB.clone", "name": "Clone with selection (planned)", "category": "SUBFRAMES",
     "test_patterns": []},
    {"id": "SUB.nested", "name": "Nested subframe export", "category": "SUBFRAMES",
     "test_patterns": [
         "test_alias_dataframe.py::TestExportTreeColumns",
     ]},

    # ── SCHEMA (4) ──
    {"id": "SCHEMA.export_import", "name": "Schema export & import (JSON)", "category": "SCHEMA",
     "test_patterns": [
         "test_alias_data_frame_schema.py",
         "test_alias_data_frame_schema_v2.py",
         "test_schema_export_v2.py",
         "test_data_schema.py",
         "test_schema_definition_vs_record.py",
     ]},
    {"id": "SCHEMA.root_persistence", "name": "ROOT file persistence", "category": "SCHEMA",
     "test_patterns": [
         "test_schema_serialization.py",
         "test_polynomial_persistence.py::TestPolynomialPersistence::test_invariance_export_tree_read_tree_roundtrip",
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
     ]},
    {"id": "DRAW.batch", "name": "draw_batch() & draw_figures()", "category": "DRAWING",
     "test_patterns": [
         "test_draw_figures.py",
     ]},
    {"id": "DRAW.subframe_resolution", "name": "Subframe column resolution in draw", "category": "DRAWING",
     "test_patterns": [
         "test_draw_subframe_resolution.py",
     ]},
    {"id": "DRAW.compound_expr", "name": "Lazy materialization of compound expressions", "category": "DRAWING",
     "test_patterns": [
         "test_draw_lazy_compound.py",
     ]},
    {"id": "DRAW.invariance", "name": "Draw vs materialize invariance", "category": "DRAWING",
     "test_patterns": [
         "test_draw_invariance.py",
     ]},

    # ── COMPRESSION (3) ──
    {"id": "COMP.roundtrip", "name": "Compress/decompress roundtrip", "category": "COMPRESSION",
     "test_patterns": [
         "test_alias_dataframe.py::TestCompressionStateMachine",
         "test_alias_dataframe.py::TestAliasDataFrameCompression",
         "test_alias_dataframe.py::TestCompressionOnMissing",
         "test_invariance_compression.py",
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
     ]},
    {"id": "BACK.invariance", "name": "Backend equivalence (numpy vs arrow vs numba)", "category": "BACKEND",
     "test_patterns": [
         "test_invariance_backend.py",
     ]},

    # ── LAZY_LOADING (3) ──
    {"id": "LAZY.read_tree", "name": "Lazy branch loading from ROOT", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_lazy_loading.py",
         "test_branch_detection.py",
     ]},
    {"id": "LAZY.chain", "name": "Chain loading (multiple files)", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_chain_loading.py",
         "test_invariance_load_mode.py",
     ]},
    {"id": "LAZY.materialization", "name": "Lazy subframe & alias evaluation", "category": "LAZY_LOADING",
     "test_patterns": [
         "test_lazy_subframes.py",
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
     ]},
]
