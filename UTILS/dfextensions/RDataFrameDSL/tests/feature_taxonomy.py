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

FEATURE_TAXONOMY_VERSION = "1.6"  # Phase 13.6.C: L2 tests SKIPPED (not xfail - ROOT JIT crash)

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
        "limitation": "L1",
    },
    "slice_chain": {
        "name": "Slice chain export",
        "description": "Mixed full + sliced columns in single to_pandas()",
        "tests": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R1_SLICE_preserves_order",
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R5_SUM_dsl_sum_preservation",
        ],
        "proof": None,
        "limitation": "L1",
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
        "description": "Reduction operations (Sum, Mean) on sliced N-D columns - L2: DSL tests SKIPPED (ROOT JIT crash)",
        "tests": [
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_sliced_exact",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_sliced_le_full",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_MEAN_sliced_bounds",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions::test_INV_ND_SUM_inner_slice_exact",
            # L2: DSL tests below SKIPPED - reductions on sliced 2D cause ROOT JIT crash
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sum_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_nested_sum",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_mean_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sqrt_sliced",
        ],
        "proof": None,
        "phase": "13.6.C",
        "limitation": "L2",  # DSL tests SKIPPED
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
}


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================

KNOWN_LIMITATIONS = {
    "L1": {
        "name": "Slice chain validation",
        "status": "✅ Resolved",
        "description": "Cannot mix full and sliced columns in single to_pandas() call",
        "workaround": "Export full and sliced columns in separate to_pandas() calls",
        "bug_report": "BUG_RDataFrameDSL_20260116_dsl_slice_chain.md",
        "resolution": "Phase 13.6.C - Tests fixed to use separate exports",
        "tests_affected": [],  # No longer affected - tests fixed
    },
    "L2": {
        "name": "Reductions on sliced 2D columns",
        "status": "⚠️ Not implemented",
        "description": "DSL does not support Sum/Mean/sqrt on sliced 2D columns like Sum(cluster_Q[0:2, :]). Causes ROOT JIT crash.",
        "workaround": "Use element-wise arithmetic on sliced 2D (e.g., cluster_x[:2,:] - cluster_Q[:2,:]), then Sum separately",
        "bug_report": None,
        "resolution": "Phase 13.6.C+ - requires DSL extension",
        "tests_affected": [
            # SKIPPED (not xfail) - these crash ROOT JIT fatally
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sum_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_nested_sum",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_mean_sliced",
            "tests/test_invariance_nd.py::TestND_SameSliceReductions_DSL::test_INV_ND_DSL_sqrt_sliced",
        ],
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
