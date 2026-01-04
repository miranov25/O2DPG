"""
RDataFrameDSL Invariant Testing — Schema Constants

Phase 13.2.1.DSL: Test data schema with mathematical invariants.

This module defines the test data schema where mathematical invariants
hold by construction. Tests verify the DSL compiler preserves these
invariants when generating C++ code for ROOT RDataFrame.

Invariant Categories:
    - Arithmetic: (A + B) - A == B
    - Boolean: De Morgan's laws
    - Comparison: (A > B) == !(A <= B)
    - RVec: Sum(A) + Sum(B) == Sum(A + B)
    - Legacy Arrays: arr[idx] == picked
"""

from typing import Dict, Any

# =============================================================================
# SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION = "13.2.1"

# =============================================================================
# TOLERANCE POLICY (FROZEN)
# =============================================================================

TOLERANCE = {
    "double": {"atol": 1e-12, "rtol": 1e-12},
    "float": {"atol": 1e-6, "rtol": 1e-6},
    "int": {"atol": 0, "rtol": 0},        # Exact
    "uint": {"atol": 0, "rtol": 0},       # Exact
    "bool": {"atol": 0, "rtol": 0},       # Exact
    "aggregation": {"atol": 1e-10, "rtol": 1e-10},  # Sum, Mean over vectors
}

# =============================================================================
# DSL SCHEMA FOR INVARIANT TESTS
# =============================================================================

INVARIANT_SCHEMA: Dict[str, str] = {
    # =========================================================================
    # SCALARS — Double precision (baseline)
    # =========================================================================
    "A": "double",              # Random [-100, 100]
    "B": "double",              # Random [-100, 100]
    "C": "double",              # Random [-100, 100] for precedence tests
    "sum_ab": "double",         # = A + B (computed at generation)
    "prod_ab": "double",        # = A * B
    "A_positive": "double",     # = abs(A) + 0.001 (always > 0)
    
    # Operator precedence testing
    "prec_add_mul": "double",   # = A + (B * C)
    "prec_mul_add": "double",   # = (A + B) * C
    
    # =========================================================================
    # SCALARS — Single precision (tolerance testing)
    # =========================================================================
    "Af": "float",              # Random [-100, 100]
    "Bf": "float",              # Random [-100, 100]
    "sum_f": "float",           # = Af + Bf
    
    # =========================================================================
    # SCALARS — Signed integers
    # =========================================================================
    "Ai": "int",                # Random [-1000, 1000]
    "Bi": "int",                # Random [-1000, 1000]
    "sum_i": "int",             # = Ai + Bi
    
    # =========================================================================
    # SCALARS — Unsigned integers (promotion testing)
    # =========================================================================
    "Au": "unsigned int",       # Random [0, 1000]
    "Bu": "unsigned int",       # Random [0, 1000]
    "sum_u": "unsigned int",    # = Au + Bu
    
    # =========================================================================
    # BOOLEANS — De Morgan's laws, precedence
    # =========================================================================
    "flag_a": "bool",
    "flag_b": "bool",
    "flag_c": "bool",           # For precedence: a || b && c
    "flag_and": "bool",         # = flag_a && flag_b
    "flag_or": "bool",          # = flag_a || flag_b
    "flag_not_a": "bool",       # = !flag_a
    
    # =========================================================================
    # RVecs — Variable length, element-wise operations
    # =========================================================================
    "vec_a": "RVec<double>",    # Variable length [0, 20] per event
    "vec_b": "RVec<double>",    # Same length as vec_a per event
    "vec_sum": "RVec<double>",  # = vec_a + vec_b (element-wise)
    "vec_len": "int",           # = len(vec_a) (for verification)
    
    "vec_i": "RVec<int>",       # Integer RVec
    "vec_i_doubled": "RVec<int>",  # = vec_i * 2
    
    # =========================================================================
    # C-ARRAY / POINTER-LIKE — Legacy TTree pattern
    # =========================================================================
    "n_arr": "int",             # Length of C-arrays [1, 10]
    # Note: C-arrays are defined in TTree branch format, not DSL schema
    # arr_d[n_arr]/D, arr_i[n_arr]/I
    
    # Index for access testing
    "idx": "int",               # Valid index into arr_d [0, n_arr-1]
    "picked": "double",         # = arr_d[idx] (for verification)
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    "near_zero": "double",      # Very small values (~1e-15)
    "large_val": "double",      # Large values (~1e10)
}

# =============================================================================
# TTREE BRANCH SPECIFICATION (for C-arrays via leaflist)
# =============================================================================

# These branches use the TTree leaflist format, not stored in DSL schema
LEGACY_BRANCHES = {
    "n_arr": {"leaflist": "n_arr/I", "description": "Length of C-arrays"},
    "arr_d": {"leaflist": "arr_d[n_arr]/D", "description": "C-array of doubles"},
    "arr_i": {"leaflist": "arr_i[n_arr]/I", "description": "C-array of ints"},
    "arr_sum": {"leaflist": "arr_sum[n_arr]/D", "description": "= arr_d + 1.0"},
}

# =============================================================================
# RVEC LENGTH PATTERNS (for variable-length testing)
# =============================================================================

# Cycle through these lengths to ensure variable-length coverage
RVEC_LENGTH_PATTERN = [0, 1, 2, 3, 5, 8, 13, 20, 1, 0, 7, 15, 4, 0, 2]

# =============================================================================
# DATA GENERATION PARAMETERS
# =============================================================================

DEFAULT_N_EVENTS = 1000
DEFAULT_SEED = 42
MAX_RVEC_LENGTH = 20
MAX_CARRAY_LENGTH = 10

# Value ranges for different types
VALUE_RANGES = {
    "double": (-100.0, 100.0),
    "float": (-100.0, 100.0),
    "int": (-1000, 1000),
    "uint": (0, 1000),
}

# =============================================================================
# INVARIANT DEFINITIONS (for documentation and sanity checks)
# =============================================================================

INVARIANTS = {
    # Arithmetic invariants
    "sum_ab": "A + B",
    "prod_ab": "A * B",
    "A_positive": "abs(A) + 0.001",
    "prec_add_mul": "A + (B * C)",
    "prec_mul_add": "(A + B) * C",
    "sum_f": "Af + Bf",
    "sum_i": "Ai + Bi",
    "sum_u": "Au + Bu",
    
    # Boolean invariants
    "flag_and": "flag_a && flag_b",
    "flag_or": "flag_a || flag_b",
    "flag_not_a": "!flag_a",
    
    # RVec invariants
    "vec_sum": "vec_a + vec_b (element-wise)",
    "vec_len": "len(vec_a)",
    "vec_i_doubled": "vec_i * 2",
    
    # C-array invariants
    "picked": "arr_d[idx]",
    "arr_sum": "arr_d + 1.0 (element-wise)",
}

# =============================================================================
# V1 SCOPE BOUNDARY (FROZEN)
# =============================================================================

V1_IN_SCOPE = [
    "Scalar types: double, float, int32, uint32, bool",
    "Vector types: RVec<double>, RVec<int>",
    "Legacy arrays: C-array branches (leaflist pattern)",
    "Access patterns: Index-based (arr[idx])",
    "Arrow export: RDF → Arrow verification",
    "Arrow import: Fail-closed rejection of unsupported types",
    "Operators: Arithmetic, Boolean, Comparison, Precedence",
]

V1_OUT_OF_SCOPE = [
    "Full Arrow round-trip (Arrow → RDF → Arrow)",
    "Object branches (POD class member access)",
    "Nested containers (RVec<RVec<T>>)",
    "String operations",
    "Algorithm-based pointer access (Sum(ptr, n))",
]

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "SCHEMA_VERSION",
    # Policy
    "TOLERANCE",
    # Schema
    "INVARIANT_SCHEMA",
    "LEGACY_BRANCHES",
    # Patterns
    "RVEC_LENGTH_PATTERN",
    # Parameters
    "DEFAULT_N_EVENTS",
    "DEFAULT_SEED",
    "MAX_RVEC_LENGTH",
    "MAX_CARRAY_LENGTH",
    "VALUE_RANGES",
    # Documentation
    "INVARIANTS",
    "V1_IN_SCOPE",
    "V1_OUT_OF_SCOPE",
]
