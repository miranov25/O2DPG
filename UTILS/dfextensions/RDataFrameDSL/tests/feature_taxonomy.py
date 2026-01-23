# =============================================================================
# tests/feature_taxonomy.py
# =============================================================================
# Feature Taxonomy v1.0 — RDataFrameDSL
#
# Purpose: Define canonical feature IDs for capability tracking
# Location: Separate file to avoid conftest.py interference
#
# Usage:
#   @pytest.mark.feature("feature_id")
#   from feature_taxonomy import FEATURE_TAXONOMY, KNOWN_LIMITATIONS
#
# IMPORTANT: Feature IDs are IMMUTABLE. If renaming needed, use FEATURE_ALIASES.
#
# Phase: 13.6.B.fix
# Date: 2026-01-18
# =============================================================================

FEATURE_TAXONOMY_VERSION = "2.0"  # Phase 13.6.F: Error detection and API features

# =============================================================================
# FEATURE ALIASES (for backward compatibility if renaming needed)
# =============================================================================
# Format: {"old_id": "new_id"}
# Usage: If feature ID must change, add mapping here instead of breaking tests
FEATURE_ALIASES = {}


FEATURE_TAXONOMY = {
    # -------------------------------------------------------------------------
    # FLATTENING (INV-S*)
    # -------------------------------------------------------------------------
    "flatten_scalar": {
        "name": "Scalar flattening",
        "description": "Scalar columns preserved during flattening",
        "tests": [
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S00_ENG_scalar_scalar_arithmetic",
        ],
        "proof": "examples/01_basic_flatten.py",
    },
    "flatten_1d": {
        "name": "1D array flattening",
        "description": "RVec<T> columns flattened to rows",
        "tests": [
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S01a_ENG_scalar_1d_replication",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S01b_ENG_scalar_1d_consistency",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S11a_ENG_1d_1d_alignment",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S11b_ENG_1d_1d_index_match",
        ],
        "proof": "examples/01_basic_flatten.py",
    },
    "flatten_2d": {
        "name": "2D array flattening",
        "description": "RVec<RVec<T>> columns flattened with track replication",
        "tests": [
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S02a_ENG_scalar_2d_replication",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S02b_ENG_scalar_2d_consistency",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S22a_ENG_2d_2d_alignment",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S22b_ENG_2d_2d_index_match",
        ],
        "proof": "examples/02_nested_flatten.py",
    },
    "flatten_mixed": {
        "name": "Mixed-depth flattening",
        "description": "Scalar + 1D + 2D columns in single query",
        "tests": [
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S12a_ENG_1d_2d_replication",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S12b_ENG_1d_2d_consistency",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S12c_DSL_main_architect_expression",
            "tests/test_invariance_combinations.py::TestDepthCombinations::test_INV_S12_ENG_main_architect_expression_engine",
        ],
        "proof": "examples/02_nested_flatten.py",
    },

    # -------------------------------------------------------------------------
    # METHOD BROADCASTING (INV-F*)
    # -------------------------------------------------------------------------
    "method_broadcast": {
        "name": "Method broadcasting (Phase 8)",
        "description": "Method calls on RVec<T> (e.g., tracks.Pt())",
        "tests": [
            "tests/test_invariance_functions.py::TestMemberFunctionsDSLTests::test_INV_F1_PT_DSL_phase8_method",
            "tests/test_invariance_functions.py::TestMemberFunctionsDSLTests::test_INV_F1_PT_TOY_exact",
            "tests/test_invariance_functions.py::TestMemberFunctionsDSLTests::test_INV_F1_PXPY_DSL_consistency",
            "tests/test_invariance_functions.py::TestMemberFunctionsDSLTests::test_INV_F_DUAL_access_paths",
        ],
        "proof": "examples/03_method_broadcasting.py",
    },
    "method_trig": {
        "name": "Trigonometric functions",
        "description": "sin, cos, sqrt, atan2 on arrays",
        "tests": [
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F1_SQRT_ENG_1d",
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F1_TRIG_ENG_identity",
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F2_SQRT_ENG_2d",
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F2_ATAN2_ENG_reconstruction",
        ],
        "proof": None,
    },
    "method_geometry": {
        "name": "Geometric computations",
        "description": "Radius, phi reconstruction from x,y",
        "tests": [
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F2_R_ENG_radius",
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F12_MIXED_ENG_function",
            "tests/test_invariance_functions.py::TestMemberFunctionsEngineTests::test_INV_F12_WEIGHTED_ENG_cluster_sum",
        ],
        "proof": None,
    },
    "udf_member_function": {
        "name": "Custom class member functions",
        "description": "Member function calls on user-defined objects (ToyTrack.Pt(), ToyCluster.getQ())",
        "tests": [
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_exact_pt",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_sliced_length",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_commutation",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_reduction_monotonicity",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_nested_exact",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Invariances::test_INV_L2_UDF_sliced_nested",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Validation::test_INV_L2_UDF_pythagorean_all_tracks",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Validation::test_INV_L2_UDF_cluster_x_offset",
            "tests/test_invariance_udf.py::TestND_L2_UDF_Validation::test_INV_L2_UDF_total_charge_sum",
        ],
        "proof": None,
        "phase": "13.6.D",
    },
    "pragma_management": {
        "name": "ROOT pragma management",
        "description": "Global pragma registry for custom class dictionaries with deduplication",
        "tests": [
            "tests/test_pragma_registry.py::TestPragmaRegistry::test_register_pragma_new",
            "tests/test_pragma_registry.py::TestPragmaRegistry::test_register_pragma_duplicate",
            "tests/test_pragma_registry.py::TestPragmaRegistry::test_is_pragma_registered",
            "tests/test_pragma_registry.py::TestPragmaRegistry::test_register_pragma_invalid",
            "tests/test_pragma_registry.py::TestPragmaRegistry::test_register_pragma_whitespace",
            "tests/test_pragma_registry.py::TestSchemaPragmas::test_schema_pragma_extraction",
            "tests/test_pragma_registry.py::TestSchemaPragmas::test_schema_pragma_registered",
            "tests/test_pragma_registry.py::TestSchemaPragmas::test_schema_no_pragmas",
            "tests/test_pragma_registry.py::TestSchemaPragmas::test_schema_multiple_pragmas",
            "tests/test_pragma_registry.py::TestPragmaThreadSafety::test_concurrent_registration",
            "tests/test_pragma_registry.py::TestRegisterFunctionCppPragmas::test_register_function_cpp_with_pragmas",
        ],
        "proof": None,
        "phase": "13.6.D",
    },

    # -------------------------------------------------------------------------
    # SLICING & RANGES (INV-R*)
    # -------------------------------------------------------------------------
    "slice_1d": {
        "name": "1D array slicing",
        "description": "Slice syntax on RVec<T> (e.g., track_pt[:2])",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R1_SLICE_dsl_slicing",
        ],
        "proof": "examples/03_method_broadcasting.py",
    },
    "slice_2d": {
        "name": "2D array slicing",
        "description": "Slice syntax on nested arrays",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R2_SLICE_2d_slicing",
        ],
        "proof": None,
        # L1 resolved: Join path OK; full-vs-sliced same-column in single call remains constrained
    },
    "slice_chain": {
        "name": "Slice chain export",
        "description": "Mixed full + sliced columns in single to_pandas()",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R1_SLICE_preserves_order",
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R5_SUM_dsl_sum_preservation",
        ],
        "proof": None,
        # L1 resolved: Join path OK; full-vs-sliced same-column in single call remains constrained
    },
    "filter_export": {
        "name": "Filter and export",
        "description": "Threshold filtering before export",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R4_FILTER_dsl_filtering",
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R4_FILTER_low_threshold",
        ],
        "proof": None,
    },
    "boundary_access": {
        "name": "Boundary element access",
        "description": "First/last element access",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R6_BOUNDARY_first_element",
        ],
        "proof": None,
    },

    # -------------------------------------------------------------------------
    # N-D SLICING (Phase 13.6.C) (INV-ND*)
    # -------------------------------------------------------------------------
    "nd_slice_2d": {
        "name": "2D N-dimensional slicing",
        "description": "N-D slice syntax on 2D arrays (e.g., cluster_Q[0:2, 0:3])",
        "tests": [
            "tests/test_invariance_nd.py::TestND1_2DSlicingBasic::test_INV_ND1a_first_n_both_dims",
            "tests/test_invariance_nd.py::TestND1_2DSlicingBasic::test_INV_ND1b_full_first_dim",
            "tests/test_invariance_nd.py::TestND1_2DSlicingBasic::test_INV_ND1c_full_both_dims",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2a_mixed_index_slice",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2b_slice_index",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2c_negative_outer",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2d_negative_inner",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2e_reverse_inner",
            "tests/test_invariance_nd.py::TestND2_2DSlicingAdvanced::test_INV_ND2f_step_outer",
        ],
        "proof": None,
        "phase": "13.6.C",
    },
    "nd_slice_3d": {
        "name": "3D N-dimensional slicing",
        "description": "N-D slice syntax on 3D arrays (e.g., hit_E[0:2, :, 0:3])",
        "tests": [
            "tests/test_invariance_nd.py::TestND3_3DSlicing::test_INV_ND3a_first_n_all_dims",
            "tests/test_invariance_nd.py::TestND3_3DSlicing::test_INV_ND3b_full_middle",
            "tests/test_invariance_nd.py::TestND3_3DSlicing::test_INV_ND3c_single_element",
            "tests/test_invariance_nd.py::TestND3_3DSlicing::test_INV_ND3d_mixed_3d",
        ],
        "proof": None,
        "phase": "13.6.C",
    },
    "nd_slice_invariance": {
        "name": "N-D slicing mathematical invariance",
        "description": "Mathematical properties: sum consistency, identity, partitioning",
        "tests": [
            "tests/test_invariance_nd.py::TestND4_MathematicalInvariance::test_INV_ND4a_sum_consistency_2d",
            "tests/test_invariance_nd.py::TestND4_MathematicalInvariance::test_INV_ND4b_identity_slice_2d",
            "tests/test_invariance_nd.py::TestND4_MathematicalInvariance::test_INV_ND4c_negative_index_equivalence",
            "tests/test_invariance_nd.py::TestND4_MathematicalInvariance::test_INV_ND4d_slice_sum_partition",
        ],
        "proof": None,
        "phase": "13.6.C",
    },
    "nd_slice_edge": {
        "name": "N-D slicing edge cases",
        "description": "Edge cases: OOB clipping, empty results, dimension limits, jagged arrays",
        "tests": [
            "tests/test_invariance_nd.py::TestND5_EdgeCases::test_INV_ND5a_oob_clipping",
            "tests/test_invariance_nd.py::TestND5_EdgeCases::test_INV_ND5b_empty_result",
            "tests/test_invariance_nd.py::TestND5_EdgeCases::test_INV_ND5c_dimension_limit",
            "tests/test_invariance_nd.py::TestND5_EdgeCases::test_INV_ND5d_jagged_structure",
        ],
        "proof": None,
        "phase": "13.6.C",
    },
    "nd_slice_arithmetic": {
        "name": "N-D slicing arithmetic (same-slice)",
        "description": "Arithmetic operations on uniformly sliced columns (no join needed)",
        "tests": [
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic::test_INV_ND_SAME_SLICE_diff_exact",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic::test_INV_ND_SAME_SLICE_xy_diff_exact",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic::test_INV_ND_SAME_SLICE_sum_partition",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic::test_INV_ND_SAME_SLICE_scalar_multiply",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic_DSL::test_INV_ND_DSL_same_slice_diff",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic_DSL::test_INV_ND_DSL_same_slice_scalar_mult",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic_DSL::test_INV_ND_DSL_chained_slice_arithmetic",
            "tests/test_invariance_nd.py::TestND_SameSliceArithmetic_DSL::test_INV_ND_DSL_slice_order_equivalence",
        ],
        "proof": None,
        "phase": "13.6.C",
    },
    "nd_slice_reduction": {
        "name": "N-D slicing reductions (Sum, Mean)",
        "description": "Reduction operations (Sum, Mean) on sliced N-D columns",
        "tests": [
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_sliced_exact",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_sliced_le_full",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_MEAN_sliced_bounds",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_inner_slice_exact",
            # L2 RESOLVED in Phase 13.6.D - DSL tests now pass
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sum_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_nested_sum",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_mean_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sqrt_sliced",
        ],
        "proof": None,
        "phase": "13.6.C",
        # L2 resolved in Phase 13.6.D - removed limitation marker
    },
    "nd_slice_order": {
        "name": "N-D slice order equivalence",
        "description": "Verify slice-first vs operate-first produce identical results",
        "tests": [
            "tests/test_invariance_nd.py::TestND_SliceOrderEquivalence::test_INV_ND_ORDER_diff_equivalence",
            "tests/test_invariance_nd.py::TestND_SliceOrderEquivalence::test_INV_ND_ORDER_sum_equivalence",
        ],
        "proof": None,
        "phase": "13.6.C",
    },

    "nd_join_strategy": {
        "name": "N-D join strategy for mixed-depth columns",
        "description": "Join strategies (inner/outer/left/right) for combining columns of different nesting depths",
        "tests": [
            "tests/test_invariance_join_e2e.py::TestJoinWithRDataFrame::test_E2E_join_cluster_track",
            "tests/test_invariance_join_e2e.py::TestJoinWithRDataFrame::test_E2E_join_cluster_event",
            "tests/test_invariance_join_e2e.py::TestJoinWithRDataFrame::test_E2E_join_track_event",
            "tests/test_invariance_join_e2e.py::TestJoinWithRDataFrame::test_E2E_join_three_depths",
            "tests/test_invariance_join_e2e.py::TestJoinStrategiesE2E::test_E2E_join_inner",
            "tests/test_invariance_join_e2e.py::TestJoinStrategiesE2E::test_E2E_join_outer",
            "tests/test_invariance_join_e2e.py::TestJoinStrategiesE2E::test_E2E_join_left",
            "tests/test_invariance_join_e2e.py::TestJoinStrategiesE2E::test_E2E_join_right",
            "tests/test_invariance_join_e2e.py::TestJoinInvarianceE2E::test_INV_E2E_broadcast_event_weight",
            "tests/test_invariance_join_e2e.py::TestJoinInvarianceE2E::test_INV_E2E_broadcast_track_pt",
            "tests/test_invariance_join_e2e.py::TestJoinInvarianceE2E::test_INV_E2E_cluster_Q_preserved",
            "tests/test_invariance_join_e2e.py::TestJoinInvarianceE2E::test_INV_E2E_weighted_cluster_sum",
            "tests/test_invariance_join_e2e.py::TestJoinInvarianceE2E::test_INV_E2E_row_count_consistency",
            "tests/test_invariance_join_e2e.py::TestBackwardCompatibilityE2E::test_E2E_no_join_param_default",
            "tests/test_invariance_join_e2e.py::TestBackwardCompatibilityE2E::test_E2E_single_column_still_works",
        ],
        "proof": None,
        "phase": "13.6.D",  # Updated with invariance tests
    },

    # -------------------------------------------------------------------------
    # STRUCTURAL INVARIANTS (INV-X*)
    # -------------------------------------------------------------------------
    "struct_sum": {
        "name": "Sum preservation",
        "description": "Sums preserved through flattening",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X1_SUM_preservation",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X1_SUM_preservation_1d",
        ],
        "proof": None,
    },
    "struct_count": {
        "name": "Count consistency",
        "description": "Row counts match expected",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X2_COUNT_consistency_1d",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X2_COUNT_consistency_2d",
        ],
        "proof": None,
    },
    "struct_index": {
        "name": "Index uniqueness",
        "description": "Unique indices per hierarchy level",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X3_UNIQUE_index_1d",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X3_UNIQUE_index_2d",
        ],
        "proof": None,
    },
    "struct_groupby": {
        "name": "GroupBy operations",
        "description": "Aggregation by event/track",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X4_GROUPBY_per_event_sum",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X4_GROUPBY_count",
        ],
        "proof": None,
    },
    "struct_dtype": {
        "name": "Dtype preservation",
        "description": "Float64 preserved, no NaN introduced",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X5_DTYPE_float64_preserved",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X5_DTYPE_index_int",
            "tests/test_invariance_structural.py::TestStructuralInvariance::test_INV_X5_DTYPE_no_nan",
        ],
        "proof": None,
    },

    # -------------------------------------------------------------------------
    # TOY DATA VALIDATION
    # -------------------------------------------------------------------------
    "toy_pythagorean": {
        "name": "Pythagorean triple validation",
        "description": "Exact validation with known integer values",
        "tests": [
            "tests/test_invariance_functions.py::TestMemberFunctionsToyData::test_INV_F_TOY_pythagorean_identity",
            "tests/test_invariance_functions.py::TestMemberFunctionsToyData::test_INV_F_TOY_known_values",
            "tests/test_invariance_functions.py::TestMemberFunctionsToyData::test_INV_F_TOY_phi_consistency",
            "tests/test_invariance_combinations.py::TestDepthCombinationsToy::test_INV_S_toy_1d_2d_replication",
            "tests/test_invariance_combinations.py::TestDepthCombinationsToy::test_INV_S_toy_pythagorean_identity",
        ],
        "proof": "tests/generators/toy_lorentz.py",
    },

    # -------------------------------------------------------------------------
    # SIMPLE/UTILITY TESTS
    # -------------------------------------------------------------------------
    "simple_ranges": {
        "name": "Simple range operations",
        "description": "Fixed window sums and track/cluster aggregations",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesSimple::test_INV_R_simple_fixed_window_1d_sum",
            "tests/test_invariance_ranges.py::TestSlidingRangesSimple::test_INV_R_simple_fixed_window_2d_sum",
            "tests/test_invariance_ranges.py::TestSlidingRangesSimple::test_INV_R_simple_event_cluster_sum",
            "tests/test_invariance_ranges.py::TestSlidingRangesSimple::test_INV_R_simple_track_cluster_sum",
        ],
        "proof": None,
    },
    "simple_structural": {
        "name": "Simple structural operations",
        "description": "Range sum and first/last element access",
        "tests": [
            "tests/test_invariance_structural.py::TestStructuralInvarianceSimple::test_INV_X_simple_range_sum",
            "tests/test_invariance_structural.py::TestStructuralInvarianceSimple::test_INV_X_first_last_elements",
        ],
        "proof": None,
    },
    # -------------------------------------------------------------------------
    # ERROR DETECTION (Phase 13.6.F - Layer 1)
    # -------------------------------------------------------------------------
    "error_missing_column": {
        "name": "Missing column detection",
        "description": "DSL rejects expressions referencing columns not in schema",
        "tests": [
            "tests/test_ir_builder.py::TestVariables::test_unknown_variable_error",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_suggestions": {
        "name": "Error message suggestions",
        "description": "Error messages include 'Did you mean...?' suggestions for similar names",
        "tests": [
            "tests/test_ir_builder.py::TestVariables::test_unknown_variable_suggestions",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_unknown_function": {
        "name": "Unknown function detection",
        "description": "DSL rejects calls to functions not in KNOWN_FUNCTIONS",
        "tests": [
            "tests/test_ir_builder.py::TestFunctionCalls::test_unknown_function_error",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_syntax": {
        "name": "Syntax error detection",
        "description": "DSL reports Python AST parse errors with PARSE_ERROR kind",
        "tests": [
            "tests/test_ir_builder.py::TestMiscellaneous::test_syntax_error",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_location": {
        "name": "Error source location",
        "description": "Errors include source_location with alias name and position",
        "tests": [
            "tests/test_ir_builder.py::TestMiscellaneous::test_error_has_location",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_slice_step_zero": {
        "name": "Slice step zero detection",
        "description": "DSL rejects slice with step=0",
        "tests": [
            "tests/test_ir_builder.py::TestSlicing::test_slice_step_zero_error",
        ],
        "proof": None,
        "phase": "13.2",
    },
    "error_invalid_method_args": {
        "name": "Invalid method arguments detection",
        "description": "DSL rejects method calls with unsupported argument patterns",
        "tests": [
            "tests/test_backend_cpp_objects.py::TestMethodCallGeneration::test_method_with_arguments_error",
        ],
        "proof": None,
        "phase": "8",
    },
    "error_rank_mismatch": {
        "name": "Rank mismatch detection",
        "description": "DSL detects scalar/vector rank incompatibilities",
        "tests": [
            "tests/test_backend_cpp.py::TestCppCodeGeneratorArithmetic::test_unknown_variable_error",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_name_conflict": {
        "name": "Name conflict detection",
        "description": "DSL rejects define() when name conflicts with schema column",
        "tests": [
            "tests/test_api_define.py::TestDefineErrors::test_define_name_conflict",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_duplicate_definition": {
        "name": "Duplicate definition detection",
        "description": "DSL rejects define() when name already defined",
        "tests": [
            "tests/test_api_define.py::TestDefineErrors::test_define_duplicate",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "error_lambda_rejected": {
        "name": "Lambda rejection",
        "description": "DSL rejects lambda expressions in define_raw() (FROZEN RULE #1)",
        "tests": [
            "tests/test_register_function_cpp.py::TestRegisterFunctionCpp::test_lambda_rejected",
        ],
        "proof": None,
        "phase": "13.5.B",
    },

    # -------------------------------------------------------------------------
    # USER API METHODS (Phase 13.6.F)
    # -------------------------------------------------------------------------
    "api_to_pandas": {
        "name": "to_pandas() export",
        "description": "Export RDataFrame to flat pandas DataFrame with TTree::Draw semantics",
        "tests": [
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_scalar",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_1d",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_2d",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_mixed_depth",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_with_selection",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_join_inner",
            "tests/test_api_to_pandas.py::TestToPandas::test_to_pandas_join_outer",
        ],
        "proof": None,
        "phase": "13.6.B",
    },
    "api_define": {
        "name": "define() column creation",
        "description": "Define computed columns with DSL expressions",
        "tests": [
            "tests/test_api_define.py::TestDefine::test_define_arithmetic",
            "tests/test_api_define.py::TestDefine::test_define_method_call",
            "tests/test_api_define.py::TestDefine::test_define_slicing",
            "tests/test_api_define.py::TestDefine::test_define_chaining",
            "tests/test_api_define.py::TestDefine::test_define_alias_reference",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "api_apply": {
        "name": "apply() RDataFrame integration",
        "description": "Apply all definitions to an RDataFrame",
        "tests": [
            "tests/test_api_define.py::TestApply::test_apply_single_definition",
            "tests/test_api_define.py::TestApply::test_apply_multiple_definitions",
            "tests/test_api_define.py::TestApply::test_apply_chained_aliases",
        ],
        "proof": None,
        "phase": "7.9",
    },
    "api_register_function_cpp": {
        "name": "register_function_cpp() UDF registration",
        "description": "Register custom C++ functions for use in DSL expressions",
        "tests": [
            "tests/test_register_function_cpp.py::TestRegisterFunctionCpp::test_register_simple_function",
            "tests/test_register_function_cpp.py::TestRegisterFunctionCpp::test_register_with_headers",
            "tests/test_register_function_cpp.py::TestRegisterFunctionCpp::test_lambda_rejected",
        ],
        "proof": None,
        "phase": "13.5.B",
    },
}


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================

KNOWN_LIMITATIONS = {
    "L1": {
        "name": "Slice chain validation",
        "status": "✅ Resolved",
        "description": "Mixed-depth joins (2D+1D+0D) now fully supported in to_pandas()",
        "workaround": None,  # No longer needed for mixed-depth joins
        "remaining_constraint": "Mixing full and sliced of same column (e.g., track_pt + track_pt[:2]) in single call remains unsupported",
        "bug_report": "BUG_RDataFrameDSL_20260116_dsl_slice_chain.md",
        "resolution": "Phase 13.6.D - Mixed-depth join invariance tests added (15/15 pass)",
        "resolution_date": "2026-01-21",
        "tests_affected": [],  # No longer affected - tests fixed
    },
    "L2": {
        "name": "Reductions on sliced 2D columns",
        "status": "✅ Resolved",
        "description": "Sum/Mean/sqrt on sliced 2D columns previously caused ROOT JIT crash.",
        "workaround": None,  # No longer needed
        "bug_report": None,
        "resolution": "Phase 13.6.D - Added explicit nested loop generation in backend_cpp.py",
        "resolution_date": "2026-01-21",
        "tests_affected": [],  # No longer affected - all tests pass
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_ids():
    """Return sorted list of all feature IDs."""
    return sorted(FEATURE_TAXONOMY.keys())


def get_limitation_ids():
    """Return sorted list of all limitation IDs."""
    return sorted(KNOWN_LIMITATIONS.keys())


def validate_feature_id(feature_id: str) -> bool:
    """Check if feature ID is valid."""
    return feature_id in FEATURE_TAXONOMY


def get_features_with_limitations():
    """Return dict of feature_id -> limitation_id for features with limitations."""
    return {
        fid: feature.get("limitation")
        for fid, feature in FEATURE_TAXONOMY.items()
        if feature.get("limitation")
    }
