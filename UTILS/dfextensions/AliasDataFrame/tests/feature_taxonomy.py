"""
Feature Taxonomy — Phase 13.11.ADF

Defines capability features for the AliasDataFrame subproject.
Each feature maps to its associated tests (proof) and benchmark checks (bench_proof).
The test layer classification in test_layer_classification.py determines
whether a feature is Verified (invariance/integration) or Smoke-only.

Phase 13.11.ADF — DRAFT for team vote
"""

# fmt: off

FEATURE_TAXONOMY = {

    # ====== CORE — Constructor, aliases, proxy, data access ======

    "CORE.constructor": {
        "name": "Constructor and factory methods",
        "description": "AliasDataFrame() constructor, read_tree, read_tree_lazy, read_chain, from_schema",
        "module": "Core",
        "proof": [
            "tests/test_constructor_contract.py::TestConstructorSchema::test_schema_id_stored",
            "tests/test_constructor_contract.py::TestConstructorSchema::test_no_schema_param",
            "tests/test_constructor_contract.py::TestConstructorSchema::test_no_metadata_param",
            "tests/test_constructor_contract.py::TestConstructorSubframe::test_subframe_register_contract",
            "tests/test_constructor_contract.py::TestConstructorSubframe::test_subframe_no_on_param",
            "tests/test_constructor_contract.py::TestConstructorSubframe::test_subframe_no_subframe_param",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.alias_definition": {
        "name": "Alias definition and materialization",
        "description": "add_alias, materialize_alias, materialize_aliases, fill_value, dtype casting",
        "module": "Core",
        "proof": [
            "tests/test_alias_dataframe.py::TestExportTreeColumns::test_fill_value_replaces_inf_nan",
            "tests/test_alias_dataframe.py::TestExportTreeColumns::test_fill_value_none_preserves_inf",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.batch_materialization": {
        "name": "Batch alias materialization",
        "description": "materialize_aliases batch path, dependency resolution",
        "module": "Core",
        "proof": [
            "tests/test_batch_materialization.py::TestBatchMaterialization::test_batch_materialize_basic",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.proxy_pattern": {
        "name": "Proxy pattern (data access delegation)",
        "description": "__getitem__, __contains__, columns, shape, loc, iloc proxy to .df",
        "module": "Core",
        "proof": [
            "tests/test_proxy_pattern.py::TestProxyGetItem::test_getitem_column",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.cycle_detection": {
        "name": "Alias cycle detection",
        "description": "Circular alias references detected and reported",
        "module": "Core",
        "proof": [
            "tests/test_cycle_detection.py::TestCycleDetection::test_simple_cycle",
            "tests/test_self_referential_cycles.py::TestSelfReferentialCycle::test_self_reference_detected",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.branch_detection": {
        "name": "AST-based branch detection",
        "description": "get_required_branches parses expressions for needed columns",
        "module": "Core",
        "proof": [
            "tests/test_branch_detection.py::TestBranchDetection::test_simple_expression",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "CORE.dependency_tree": {
        "name": "Alias dependency tree",
        "description": "Dependency resolution order for chained aliases",
        "module": "Core",
        "proof": [
            "tests/test_dependency_tree.py::TestDependencyTree::test_simple_chain",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== SCHEMA — v1/v2, export, validation, groups ======

    "SCHEMA.v1_v2": {
        "name": "Schema v1/v2 format support",
        "description": "Schema creation, loading, v1→v2 migration, validation",
        "module": "Schema",
        "proof": [
            "tests/test_alias_data_frame_schema.py::TestSchemaBasic::test_schema_id",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SCHEMA.v2_export": {
        "name": "Schema v2 export and groups",
        "description": "export_schema, save_schema_v2, column groups, ordering",
        "module": "Schema",
        "proof": [
            "tests/test_schema_export_v2.py::TestSchemaExportV2::test_export_schema_v2",
            "tests/test_alias_dataframe.py::TestSchemaV2Ordering::test_schema_v2_groups_roundtrip",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SCHEMA.serialization": {
        "name": "Schema JSON serialization roundtrip",
        "description": "save_schema / load_schema JSON roundtrip, format detection",
        "module": "Schema",
        "proof": [
            "tests/test_schema_serialization.py::TestSchemaSerialization::test_save_load_roundtrip",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SCHEMA.definition_vs_record": {
        "name": "Definition schema vs record schema",
        "description": "export_definition_schema (blueprint) vs export_record_schema (snapshot)",
        "module": "Schema",
        "proof": [
            "tests/test_schema_definition_vs_record.py::TestDefinitionVsRecord::test_definition_has_no_data",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SCHEMA.data_schema": {
        "name": "Data schema and column metadata",
        "description": "set_column_metadata, set_axis_title, get_column_metadata",
        "module": "Schema",
        "proof": [
            "tests/test_data_schema.py::TestDataSchema::test_set_axis_title",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SCHEMA.validation": {
        "name": "Schema validation",
        "description": "validate_schema, apply_schema, describe_schema",
        "module": "Schema",
        "proof": [
            "tests/test_validation_display_adf.py::TestValidation::test_validate_strict",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== SUBFRAMES — Join, multi-key, caching ======

    "SUBFRAME.join_basic": {
        "name": "Single-key subframe join",
        "description": "register_subframe + left join on single key",
        "module": "Subframes",
        "proof": [
            "tests/test_alias_subframe.py::TestSubframeBasicJoin::test_single_key_join",
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_1_single_key_join_matches_pandas",
        ],
        "bench_proof": ["Phase 8c benchmark: 10.8× join speedup"],
        "impl_tag": "NUMBA",
    },
    "SUBFRAME.join_multikey": {
        "name": "Multi-key subframe join",
        "description": "Composite key linearization, 2-key and 3-key joins",
        "module": "Subframes",
        "proof": [
            "tests/test_alias_subframe.py::TestSubframeBasicJoin::test_multi_key_join_2keys",
            "tests/test_alias_subframe.py::TestSubframeBasicJoin::test_multi_key_join_3keys",
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_2_multikey_2col_join_matches_pandas",
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_3_multikey_3col_join_matches_pandas",
            "tests/test_composite_keys.py::TestCompositeKeys::test_2key_linearization",
        ],
        "bench_proof": [],
        "impl_tag": "NUMBA",
    },
    "SUBFRAME.join_edge_cases": {
        "name": "Join edge cases (missing keys, duplicates, empty)",
        "description": "NaN for missing, Cartesian for duplicates, empty subframe handling",
        "module": "Subframes",
        "proof": [
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_4_missing_keys_produce_nan",
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_5_duplicate_keys_handled_correctly",
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_6_empty_subframe_handling",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SUBFRAME.flat_normalized_equivalence": {
        "name": "Flat vs normalized data equivalence",
        "description": "Critical invariant: flat['pt * run'] == normalized['pt * E.run']",
        "module": "Subframes",
        "proof": [
            "tests/test_invariance_subframe.py::TestInvarianceSubframe::test_I3_9_flat_normalized_equivalence",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SUBFRAME.alias_api": {
        "name": "Subframe alias API (dot notation)",
        "description": "adf['Sub.col'] access, cross-table expressions in add_alias",
        "module": "Subframes",
        "proof": [
            "tests/test_subframe_alias_api.py::TestSubframeAliasAPI::test_dot_notation_access",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "SUBFRAME.join_caching": {
        "name": "Join index caching",
        "description": "Cached join indices for repeated subframe access",
        "module": "Subframes",
        "proof": [
            "tests/test_join_caching.py::TestJoinCaching::test_cache_hit",
            "tests/test_join_index_caching.py::TestJoinIndexCaching::test_cache_performance",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== LAZY — Branch loading, chains ======

    "LAZY.basic": {
        "name": "Lazy branch loading",
        "description": "read_tree_lazy, ensure_branches, on-demand loading",
        "module": "Lazy Loading",
        "proof": [
            "tests/test_lazy_loading.py::TestLazyLoading::test_lazy_load_basic",
            "tests/test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_1_lazy_vs_eager_column_values",
            "tests/test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_2_lazy_vs_eager_alias_evaluation",
        ],
        "bench_proof": ["Phase 1 benchmark: 60-770× read speedup"],
        "impl_tag": None,
    },
    "LAZY.chain": {
        "name": "Multi-file chain loading",
        "description": "read_chain, read_chain_lazy, LRU cache, validation modes",
        "module": "Lazy Loading",
        "proof": [
            "tests/test_chain_loading.py::TestChainLoading::test_chain_basic",
            "tests/test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_4_lazy_chain_vs_eager_chain",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "LAZY.subframe": {
        "name": "Lazy subframe loading",
        "description": "register_subframe with lazy/chain subframes, mixed lazy/eager",
        "module": "Lazy Loading",
        "proof": [
            "tests/test_lazy_subframes.py::TestLazySubframe::test_lazy_subframe_basic",
            "tests/test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_3_lazy_vs_eager_with_subframe_join",
            "tests/test_invariance_load_mode.py::TestInvarianceLoadMode::test_I1_6_mixed_lazy_main_eager_sub",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== BACKEND — Numba, Arrow, NumPy dispatch ======

    "BACKEND.numba_scatter": {
        "name": "Numba scatter/join acceleration",
        "description": "JIT-compiled scatter, join index computation, multi-key linearization",
        "module": "Backend",
        "proof": [
            "tests/test_numba_acceleration.py::TestNumbaAcceleration::test_scatter_basic",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_1_single_key_join_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_2_join_with_arithmetic_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_10_large_dataset_numba_vs_numpy",
        ],
        "bench_proof": ["Phase 8a: 8× faster index computation"],
        "impl_tag": "NUMBA",
    },
    "BACKEND.arrow_scatter": {
        "name": "Arrow scatter (pc.take)",
        "description": "PyArrow scatter path for large arrays",
        "module": "Backend",
        "proof": [
            "tests/test_arrow_scatter.py::TestArrowScatter::test_scatter_basic",
        ],
        "bench_proof": [],
        "impl_tag": "PYARROW",
    },
    "BACKEND.arrow_compute": {
        "name": "Arrow compute (disabled)",
        "description": "PyArrow expression eval — disabled (16× slower than NumPy)",
        "module": "Backend",
        "proof": [
            "tests/test_arrow_compute.py::TestArrowCompute::test_basic_expression",
            "tests/test_arrow_expression.py::TestArrowExpression::test_basic",
        ],
        "bench_proof": ["Phase 9e: 16× slower than NumPy — disabled"],
        "impl_tag": "PYARROW",
    },
    "BACKEND.numba_numpy_parity": {
        "name": "Numba vs NumPy backend parity",
        "description": "All join operations produce identical results regardless of backend",
        "module": "Backend",
        "proof": [
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_3_multikey_join_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_4_missing_keys_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_5_duplicate_keys_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_7_multiple_subframes_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_8_math_functions_in_join_numba_vs_numpy",
            "tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_9_boolean_conditions_numba_vs_numpy",
        ],
        "bench_proof": [],
        "impl_tag": "NUMBA",
    },

    # ====== COMPRESSION — Roundtrip, formulas ======

    "COMPRESSION.roundtrip": {
        "name": "Compression roundtrip correctness",
        "description": "compress_columns / decompress_columns roundtrip for all formulas",
        "module": "Compression",
        "proof": [
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_1_linear_compression_roundtrip",
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_4_sqrt_compression_roundtrip",
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_5_compress_decompress_idempotent",
            "tests/test_compression_pytest.py::TestCompressionBasic::test_linear_compression",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "COMPRESSION.schema_preservation": {
        "name": "Compression schema preservation",
        "description": "Schema metadata preserved through compress/decompress cycle",
        "module": "Compression",
        "proof": [
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_6_schema_preserved_after_roundtrip",
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_7_compressed_dtype_correct",
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_8_multi_column_compression",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "COMPRESSION.known_failures": {
        "name": "Compression known failures (scaled/asinh)",
        "description": "I4_2 scaled linear, I4_3 asinh — known roundtrip failures",
        "module": "Compression",
        "proof": [
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip",
            "tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== DRAWING — draw, draw_batch, draw_figures ======

    "DRAW.basic": {
        "name": "draw() basic plotting",
        "description": "AliasDataFrame.draw() delegates to DFDraw with auto-materialization",
        "module": "Drawing",
        "proof": [
            "tests/test_draw_invariance.py::TestDrawInvariance::test_draw_vs_materialize_identical",
            "tests/test_draw_invariance.py::TestDrawInvariance::test_eager_vs_lazy_draw_identical",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "DRAW.subframe_resolution": {
        "name": "draw() subframe column resolution",
        "description": "Sub.col → Sub_col via pd.merge in draw/draw_batch/draw_figures",
        "module": "Drawing",
        "proof": [
            "tests/test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_values_correct",
            "tests/test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_no_conflict",
            "tests/test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_column_with_conflict",
            "tests/test_draw_subframe_resolution.py::TestDrawSubframeResolution::test_draw_subframe_in_selection",
            "tests/test_draw_subframe_resolution.py::TestDrawBatchSubframeResolution::test_draw_batch_subframe_basic",
            "tests/test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_subframe_basic",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "DRAW.figures": {
        "name": "draw_figures() multi-subplot dashboard",
        "description": "draw_figures with layout, defaults cascade, savefig",
        "module": "Drawing",
        "proof": [
            "tests/test_draw_figures.py::TestDrawFigures::test_basic_figure",
            "tests/test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_defaults_cascade",
            "tests/test_draw_subframe_resolution.py::TestDrawFiguresSubframeResolution::test_draw_figures_layout_tuple",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "DRAW.chain_integration": {
        "name": "Draw with chain and lazy loading",
        "description": "draw() on lazy chain ADF with auto-loading",
        "module": "Drawing",
        "proof": [
            "tests/test_draw_chain_integration.py::TestDrawChainIntegration::test_draw_chain_basic",
            "tests/test_draw_lazy_integration.py::TestDrawLazyIntegration::test_draw_lazy_basic",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "DRAW.data_invariance": {
        "name": "Draw data correctness invariants",
        "description": "Eager vs lazy draw identical, alias chains correct, dtype preserved",
        "module": "Drawing",
        "proof": [
            "tests/test_draw_invariance.py::TestCoreInvariants::test_known_relationship_single_file",
            "tests/test_draw_invariance.py::TestCoreInvariants::test_known_relationship_chain",
            "tests/test_draw_invariance.py::TestCoreInvariants::test_eager_vs_lazy_data_identical",
            "tests/test_draw_invariance.py::TestCoreInvariants::test_alias_materialization_identical",
            "tests/test_draw_invariance.py::TestCoreInvariants::test_complex_alias_chain",
            "tests/test_draw_invariance.py::TestSubframeJoinCorrectness::test_sector_calibration_correct",
            "tests/test_draw_invariance.py::TestChainSubframeIntegration::test_chain_with_subframe_chain",
            "tests/test_draw_invariance.py::TestDtypePreservation::test_float32_preserved_lazy",
            "tests/test_draw_invariance.py::TestDtypePreservation::test_int32_preserved_lazy",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== REGISTERED FUNCTIONS — PolynomialSpec, evaluator ======

    "FUNC.register_function": {
        "name": "Generic function registration",
        "description": "register_function(name, callable, overwrite) for custom alias functions",
        "module": "Registered Functions",
        "proof": [
            "tests/test_polynomial_spec.py::TestRegisterFunction::test_basic",
            "tests/test_polynomial_spec.py::TestRegisterFunction::test_collision_raises",
            "tests/test_polynomial_spec.py::TestRegisterFunction::test_overwrite",
            "tests/test_polynomial_spec.py::TestRegisterFunction::test_backward_compatibility",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "FUNC.polynomial_spec": {
        "name": "PolynomialSpec N-dimensional basis",
        "description": "PolynomialSpec(columns, degrees), basis_expressions, to_schema, from_schema",
        "module": "Registered Functions",
        "proof": [
            "tests/test_polynomial_spec.py::TestBasisExpressions::test_degree_332",
            "tests/test_polynomial_spec.py::TestBasisExpressions::test_4d",
            "tests/test_polynomial_spec.py::TestBasisExpressions::test_tuple_format",
            "tests/test_polynomial_spec.py::TestSchemaRoundtrip::test_full_roundtrip",
            "tests/test_polynomial_spec.py::TestSchemaRoundtrip::test_sparse_roundtrip",
            "tests/test_polynomial_spec.py::TestRootExpression::test_basic",
            "tests/test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_polynomial_vs_numpy",
            "tests/test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_flat_vs_subframe_coeffs",
            "tests/test_polynomial_spec.py::TestInvariancePolynomial::test_invariance_schema_roundtrip_values",
        ],
        "bench_proof": ["Phase 13.9: 42× Numba vs pandas.eval"],
        "impl_tag": "NUMBA",
    },
    "FUNC.polynomial_subframe": {
        "name": "Polynomial from subframe coefficients",
        "description": "register_polynomial_from_subframe with Numba JIT evaluator",
        "module": "Registered Functions",
        "proof": [
            "tests/test_polynomial_spec.py::TestRegisterPolynomial::test_end_to_end",
        ],
        "bench_proof": ["Phase 13.9: 27 KB coefficient matrix, no broadcast to 13.5M rows"],
        "impl_tag": "NUMBA",
    },
    "FUNC.evaluator": {
        "name": "Evaluator registration (duck-typed)",
        "description": "register_evaluator for GroupByRegressionEvaluator and similar objects",
        "module": "Registered Functions",
        "proof": [
            "tests/test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_and_evaluate",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_single_column",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorBasic::test_register_three_columns",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorCollision::test_collision_raises",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorCollision::test_overwrite_replaces",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorValidation::test_no_evaluate_method_raises",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorValidation::test_wrong_arg_count_raises",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_multi_predictor_with_selection",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorMultiPredictor::test_multi_predictor_no_selection_raises",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorComposition::test_two_evaluators_summed",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorComposition::test_chained_corrections",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_alias_vs_direct",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_large_data",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorInvariance::test_invariance_multi_predictor",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_schema_stores_contract",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "FUNC.evaluator_composition": {
        "name": "Evaluator composition via alias chaining",
        "description": "Multiple evaluators combined via add_alias('total', 'f1(x) + f2(x)')",
        "module": "Registered Functions",
        "proof": [
            "tests/test_register_evaluator.py::TestRegisterEvaluatorComposition::test_evaluator_minus_column",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorComposition::test_evaluator_with_alias_dependency",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorComposition::test_chained_corrections",
            "tests/test_register_evaluator.py::TestRegisterEvaluatorEdgeCases::test_backward_compatibility",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== RDATAFRAME — C++ export, friend trees ======

    "RDF.cpp_expression": {
        "name": "Python to C++ expression translation",
        "description": "to_cpp_expr for RDataFrame Define() calls",
        "module": "RDataFrame",
        "proof": [
            "tests/test_AliasDataFrameRDF.py::TestToCppExpr::test_numpy_sqrt",
            "tests/test_AliasDataFrameRDF.py::TestToCppExpr::test_combined_expression",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
    "RDF.integration": {
        "name": "RDataFrame setup and friend trees",
        "description": "setup_rdf_with_friends, add_defines_to_rdf",
        "module": "RDataFrame",
        "proof": [
            "tests/test_rdf_integration.py::TestRDFIntegration::test_basic_setup",
            "tests/test_rdf_integration_final.py::TestRDFIntegrationFinal::test_basic_setup",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== FIT — Fit registration, profiling ======

    "FIT.register_result": {
        "name": "Fit result registration",
        "description": "register_fit_result with metadata for QA dashboards",
        "module": "Fit Registration",
        "proof": [
            "tests/test_register_fit_result.py::TestRegisterFitResult::test_register_basic",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },

    # ====== INVARIANCE — Cross-cutting invariance suites ======

    "INV.smoke": {
        "name": "Invariance smoke tests (I0)",
        "description": "Quick smoke checks for all invariance categories",
        "module": "Invariance",
        "proof": [
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I1_load_mode",
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I2_backend",
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I3_subframe_join",
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I4_compression",
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I5_schema",
            "tests/test_invariance_smoke.py::TestInvarianceSmoke::test_smoke_I6_order",
        ],
        "bench_proof": [],
        "impl_tag": None,
    },
}

# fmt: on

IMPL_TAGS = {
    "NUMBA": "Requires Numba JIT compilation",
    "PYARROW": "Requires PyArrow library",
}

VERBOSE_DUPLICATES = {}  # No verbose duplicate files in ADF


def get_feature_count():
    return len(FEATURE_TAXONOMY)


def get_all_proof_tests():
    tests = set()
    for feat in FEATURE_TAXONOMY.values():
        tests.update(feat["proof"])
    return tests


if __name__ == "__main__":
    print(f"Feature count: {get_feature_count()}")
    modules = {}
    for feat in FEATURE_TAXONOMY.values():
        m = feat["module"]
        modules[m] = modules.get(m, 0) + 1
    for m, c in sorted(modules.items()):
        print(f"  {m}: {c} features")
    all_proofs = get_all_proof_tests()
    print(f"\nTotal unique proof tests: {len(all_proofs)}")
    inv_count = sum(1 for t in all_proofs if "invariance" in t.lower())
    print(f"Invariance proof tests: {inv_count}")
