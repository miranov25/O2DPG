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

FEATURE_TAXONOMY_VERSION = "1.0"

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
        "status": "⚠️ Partial",
        "description": "Cannot mix full and sliced columns in single to_pandas() call",
        "workaround": "Export full and sliced columns in separate to_pandas() calls",
        "bug_report": "BUG_RDataFrameDSL_20260116_dsl_slice_chain.md",
        "resolution": "Phase 13.6.C",
        "tests_affected": [
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R1_SLICE_preserves_order",
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R2_SLICE_2d_slicing",
            "tests/test_invariance_ranges.py::TestSlidingRangesDSL::test_INV_R5_SUM_dsl_sum_preservation",
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
