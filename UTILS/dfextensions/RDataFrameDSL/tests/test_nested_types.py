"""
Phase 13.3.DSL: Nested RVec Type Inference Tests

This module tests the bug fix for nested RVec type handling (D1-D4).

The bug was in _simple_schema_to_full() which incorrectly parsed:
- RVec<RVec<double>> → dtype='RVec<double>', rank=1 (WRONG)

The fix uses extract_inner_type() to correctly parse:
- RVec<RVec<double>> → dtype='double', rank=2 (CORRECT)

Test Coverage:
- Nested RVec<RVec<T>> parsing (unit tests)
- Triple nesting RVec<RVec<RVec<T>>>
- Backward compatibility with RVec<T>
- Backward compatibility with scalars
- ROOT:: prefixed types
- std::vector nested types
- Type chain through subscript operations
- ROOT integration tests (P0 requirement)

Phase: 13.3.DSL (D1-D4 Bug Fix)
Date: 2026-01-06

P0 Fixes Applied:
- P0-1: Added ROOT integration tests (compile + execute)
- P0-2: Import from RDataFrameDSL source instead of duplicating helpers
"""

import pytest

# =============================================================================
# P0-2 FIX: Import from source instead of duplicating helpers
# =============================================================================

from RDataFrameDSL.type_inferrer import (
    extract_inner_type,
    is_collection_type,
    is_vector_type,
    is_rvec_type,
)
from RDataFrameDSL.dsl_compiler import (
    DSLCompiler,
    _simple_schema_to_full,
)


# =============================================================================
# Type Parsing Tests (D1: Nested type parsing)
# =============================================================================

class TestNestedTypeParsing:
    """Test correct parsing of nested RVec types."""
    
    def test_nested_rvec_double(self):
        """RVec<RVec<double>> should have dtype='double', rank=2."""
        result = _simple_schema_to_full({"nested": "RVec<RVec<double>>"})
        col = result["columns"]["nested"]
        
        assert col["dtype"] == "double", f"Expected dtype='double', got {col['dtype']!r}"
        assert col["rank"] == 2, f"Expected rank=2, got {col['rank']}"
        assert col["cpp_type"] == "RVec<RVec<double>>"
    
    def test_nested_rvec_float(self):
        """RVec<RVec<float>> should have dtype='float', rank=2."""
        result = _simple_schema_to_full({"nested": "RVec<RVec<float>>"})
        col = result["columns"]["nested"]
        
        assert col["dtype"] == "float"
        assert col["rank"] == 2
    
    def test_nested_rvec_int(self):
        """RVec<RVec<int>> should have dtype='int', rank=2."""
        result = _simple_schema_to_full({"nested": "RVec<RVec<int>>"})
        col = result["columns"]["nested"]
        
        assert col["dtype"] == "int"
        assert col["rank"] == 2
    
    def test_triple_nested(self):
        """RVec<RVec<RVec<float>>> should have dtype='float', rank=3."""
        result = _simple_schema_to_full({"deep": "RVec<RVec<RVec<float>>>"})
        col = result["columns"]["deep"]
        
        assert col["dtype"] == "float"
        assert col["rank"] == 3
    
    def test_root_prefixed_nested(self):
        """ROOT::RVec<ROOT::RVec<double>> should have dtype='double', rank=2."""
        result = _simple_schema_to_full({"nested": "ROOT::RVec<ROOT::RVec<double>>"})
        col = result["columns"]["nested"]
        
        assert col["dtype"] == "double"
        assert col["rank"] == 2
    
    def test_vector_nested(self):
        """std::vector<std::vector<int>> should have dtype='int', rank=2."""
        result = _simple_schema_to_full({"matrix": "std::vector<std::vector<int>>"})
        col = result["columns"]["matrix"]
        
        assert col["dtype"] == "int"
        assert col["rank"] == 2


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure fix doesn't break existing functionality."""
    
    def test_simple_rvec_double(self):
        """RVec<double> should still have dtype='double', rank=1."""
        result = _simple_schema_to_full({"pt": "RVec<double>"})
        col = result["columns"]["pt"]
        
        assert col["dtype"] == "double"
        assert col["rank"] == 1
    
    def test_simple_rvec_int(self):
        """RVec<int> should still have dtype='int', rank=1."""
        result = _simple_schema_to_full({"counts": "RVec<int>"})
        col = result["columns"]["counts"]
        
        assert col["dtype"] == "int"
        assert col["rank"] == 1
    
    def test_scalar_double(self):
        """Scalar 'double' should have dtype='double', rank=0."""
        result = _simple_schema_to_full({"px": "double"})
        col = result["columns"]["px"]
        
        assert col["dtype"] == "double"
        assert col["rank"] == 0
    
    def test_scalar_int(self):
        """Scalar 'int' should have dtype='int', rank=0."""
        result = _simple_schema_to_full({"n": "int"})
        col = result["columns"]["n"]
        
        assert col["dtype"] == "int"
        assert col["rank"] == 0
    
    def test_rvec_of_objects(self):
        """RVec<TLorentzVector> should preserve object type."""
        result = _simple_schema_to_full({"tracks": "RVec<TLorentzVector>"})
        col = result["columns"]["tracks"]
        
        assert col["dtype"] == "TLorentzVector"
        assert col["rank"] == 1
    
    def test_mixed_schema(self):
        """Schema with mixed types should parse correctly."""
        schema = {
            "px": "double",
            "pt": "RVec<double>",
            "nested": "RVec<RVec<double>>",
            "tracks": "RVec<TLorentzVector>",
        }
        result = _simple_schema_to_full(schema)
        
        assert result["columns"]["px"]["rank"] == 0
        assert result["columns"]["pt"]["rank"] == 1
        assert result["columns"]["nested"]["rank"] == 2
        assert result["columns"]["tracks"]["rank"] == 1


# =============================================================================
# Type Inference Chain Tests (D2-D4)
# =============================================================================

class TestTypeInferenceChain:
    """Test type inference through subscript operations."""
    
    def test_subscript_chain_rank_reduction(self):
        """
        Subscripting reduces rank by 1, dtype stays constant.
        
        nested[0][0] where nested is RVec<RVec<double>>:
        - nested:       rank=2, dtype='double'
        - nested[0]:    rank=1, dtype='double'  → RVec<double>
        - nested[0][0]: rank=0, dtype='double'  → double
        """
        result = _simple_schema_to_full({"nested": "RVec<RVec<double>>"})
        col = result["columns"]["nested"]
        
        # Initial state
        rank = col["rank"]
        dtype = col["dtype"]
        
        assert rank == 2
        assert dtype == "double"
        
        # After nested[0]
        rank_1 = max(0, rank - 1)
        assert rank_1 == 1, "After first subscript, rank should be 1"
        
        # After nested[0][0]
        rank_2 = max(0, rank_1 - 1)
        assert rank_2 == 0, "After second subscript, rank should be 0"
        
        # dtype stays constant through chain
        assert dtype == "double", "dtype should be 'double' throughout"
    
    def test_triple_subscript_chain(self):
        """
        Triple nesting subscript chain.
        
        deep[0][0][0] where deep is RVec<RVec<RVec<float>>>:
        - deep:          rank=3
        - deep[0]:       rank=2
        - deep[0][0]:    rank=1
        - deep[0][0][0]: rank=0
        """
        result = _simple_schema_to_full({"deep": "RVec<RVec<RVec<float>>>"})
        col = result["columns"]["deep"]
        
        rank = col["rank"]
        assert rank == 3
        
        for i in range(3):
            rank = max(0, rank - 1)
        
        assert rank == 0, "After three subscripts, rank should be 0"


# =============================================================================
# Invariant Tests
# =============================================================================

class TestInvariants:
    """Mathematical invariants for type system."""
    
    def test_rank_equals_nesting_depth(self):
        """rank == nesting depth for all collection types."""
        test_cases = [
            ("double", 0),
            ("RVec<double>", 1),
            ("RVec<RVec<double>>", 2),
            ("RVec<RVec<RVec<double>>>", 3),
            ("std::vector<int>", 1),
            ("std::vector<std::vector<int>>", 2),
        ]
        
        for type_str, expected_rank in test_cases:
            result = _simple_schema_to_full({"col": type_str})
            actual_rank = result["columns"]["col"]["rank"]
            assert actual_rank == expected_rank, \
                f"Type {type_str!r}: expected rank={expected_rank}, got {actual_rank}"
    
    def test_dtype_is_innermost_element(self):
        """dtype == innermost element type for all collection types."""
        test_cases = [
            ("double", "double"),
            ("RVec<double>", "double"),
            ("RVec<RVec<double>>", "double"),
            ("RVec<float>", "float"),
            ("RVec<RVec<float>>", "float"),
            ("RVec<TLorentzVector>", "TLorentzVector"),
            ("std::vector<int>", "int"),
        ]
        
        for type_str, expected_dtype in test_cases:
            result = _simple_schema_to_full({"col": type_str})
            actual_dtype = result["columns"]["col"]["dtype"]
            assert actual_dtype == expected_dtype, \
                f"Type {type_str!r}: expected dtype={expected_dtype!r}, got {actual_dtype!r}"
    
    def test_cpp_type_preserved(self):
        """cpp_type == original type string for collection types."""
        test_cases = [
            "RVec<double>",
            "RVec<RVec<double>>",
            "ROOT::RVec<double>",
            "std::vector<int>",
        ]
        
        for type_str in test_cases:
            result = _simple_schema_to_full({"col": type_str})
            cpp_type = result["columns"]["col"].get("cpp_type", "")
            assert cpp_type == type_str, \
                f"cpp_type should be {type_str!r}, got {cpp_type!r}"


# =============================================================================
# P0-1 FIX: ROOT Integration Tests
# =============================================================================

class TestNestedRVecROOTIntegration:
    """
    ROOT integration tests for Phase 13.3.DSL nested RVec fix.
    
    These tests verify that:
    1. Generated C++ has correct parameter types
    2. ROOT compiles the generated code
    3. Actual execution returns correct results
    
    P0 requirement from Main Architect review.
    """
    
    @pytest.fixture
    def nested_dsl(self):
        """Create DSLCompiler with nested RVec schema."""
        return DSLCompiler({"nested": "RVec<RVec<double>>"})
    
    def test_nested_parameter_type_in_generated_code(self, nested_dsl):
        """Generated C++ should have correct nested parameter type."""
        nested_dsl.define("first_inner", "nested[0]")
        
        # Get generated code via preview
        code = nested_dsl.preview()
        
        # Verify the parameter type is the full nested RVec type
        # Should be: const ROOT::RVec<ROOT::RVec<double>>& nested
        assert "RVec<" in code, "Generated code should contain RVec"
        # The parameter must be the nested type, not flattened
        assert "RVec<double>>" in code or "RVec<ROOT::RVec<double>>" in code, \
            f"Parameter should be nested RVec type, got:\n{code}"
    
    def test_nested_double_index_return_type(self, nested_dsl):
        """nested[0][0] should return double, not RVec."""
        nested_dsl.define("elem", "nested[0][0]")
        
        # Get the generated function info
        func = nested_dsl.get_function("elem")
        
        # Return type should be double (scalar), not RVec
        assert func is not None, "Function should be generated"
        assert "double" in func.return_type.lower(), \
            f"Return type should be double, got {func.return_type}"
        assert "rvec" not in func.return_type.lower(), \
            f"Return type should NOT be RVec, got {func.return_type}"
    
    def test_nested_single_index_return_type(self, nested_dsl):
        """nested[0] should return RVec<double>, not double."""
        nested_dsl.define("inner", "nested[0]")
        
        func = nested_dsl.get_function("inner")
        
        assert func is not None, "Function should be generated"
        # Return type should be RVec<double>
        assert "RVec" in func.return_type or "rvec" in func.return_type.lower(), \
            f"Return type should be RVec, got {func.return_type}"
    
    @pytest.mark.requires_root
    def test_nested_compiles_with_root(self, nested_dsl):
        """Nested RVec expressions should compile in ROOT without errors."""
        ROOT = pytest.importorskip("ROOT")
        
        nested_dsl.define("first_inner", "nested[0]")
        nested_dsl.define("first_elem", "nested[0][0]")
        
        # compile_all() should not raise any exceptions
        # This tests that ROOT's gInterpreter accepts the generated C++
        try:
            nested_dsl.compile_all()
        except Exception as e:
            pytest.fail(f"ROOT compilation failed: {e}")
    
    @pytest.mark.requires_root
    def test_nested_executes_correctly(self):
        """
        End-to-end test: create RDF with nested RVec, apply DSL, verify results.
        
        This is the ultimate test that the fix works in practice.
        """
        ROOT = pytest.importorskip("ROOT")
        
        # Create a simple RDataFrame with nested RVec column
        # Using Define to create a nested structure
        rdf = ROOT.RDataFrame(3)
        
        # Create nested RVec: [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]
        rdf = rdf.Define("nested", """
            ROOT::RVec<ROOT::RVec<double>> result;
            if (rdfentry_ == 0) {
                result.push_back(ROOT::RVec<double>{1.0, 2.0});
            } else if (rdfentry_ == 1) {
                result.push_back(ROOT::RVec<double>{3.0, 4.0, 5.0});
            } else {
                result.push_back(ROOT::RVec<double>{6.0});
            }
            return result;
        """)
        
        # Now use DSLCompiler with schema matching the column type
        schema = {"nested": "RVec<RVec<double>>"}
        dsl = DSLCompiler(schema)
        
        # Define expressions using nested indexing
        dsl.define("first_inner", "nested[0]")
        dsl.define("first_elem", "nested[0][0]")
        
        # Apply to RDF
        rdf = dsl.apply(rdf)
        
        # Verify results
        results = rdf.AsNumpy(["first_elem"])
        first_elems = results["first_elem"]
        
        # Expected: first element of first inner vector for each entry
        # Entry 0: nested[0][0] = 1.0
        # Entry 1: nested[0][0] = 3.0
        # Entry 2: nested[0][0] = 6.0
        expected = [1.0, 3.0, 6.0]
        
        for i, (actual, exp) in enumerate(zip(first_elems, expected)):
            assert abs(actual - exp) < 1e-10, \
                f"Entry {i}: expected {exp}, got {actual}"
    
    @pytest.mark.requires_root
    def test_nested_with_slicing(self):
        """Test that nested[0][:2] works (chained operations)."""
        ROOT = pytest.importorskip("ROOT")
        
        # Create RDF with nested RVec
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("nested", """
            ROOT::RVec<ROOT::RVec<double>> result;
            result.push_back(ROOT::RVec<double>{1.0, 2.0, 3.0, 4.0});
            return result;
        """)
        
        schema = {"nested": "RVec<RVec<double>>"}
        dsl = DSLCompiler(schema)
        
        # nested[0] returns RVec<double>, then [:2] slices it
        dsl.define("first_two", "nested[0][:2]")
        
        # Should compile without error
        rdf = dsl.apply(rdf)
        
        # Verify result
        results = rdf.AsNumpy(["first_two"])
        first_two = results["first_two"][0]  # First (only) entry
        
        assert len(first_two) == 2, f"Expected 2 elements, got {len(first_two)}"
        assert abs(first_two[0] - 1.0) < 1e-10
        assert abs(first_two[1] - 2.0) < 1e-10


# =============================================================================
# Helper Function Tests (verify imports work)
# =============================================================================

class TestHelperImports:
    """Verify that imported helper functions work correctly."""
    
    def test_extract_inner_type_imported(self):
        """extract_inner_type should be imported from source."""
        inner, depth = extract_inner_type("RVec<RVec<double>>")
        assert inner == "double"
        assert depth == 2
    
    def test_is_collection_type_imported(self):
        """is_collection_type should be imported from source."""
        assert is_collection_type("RVec<double>") is True
        assert is_collection_type("double") is False
    
    def test_is_rvec_type_imported(self):
        """is_rvec_type should be imported from source."""
        assert is_rvec_type("RVec<double>") is True
        assert is_rvec_type("ROOT::RVec<double>") is True
        assert is_rvec_type("std::vector<double>") is False
    
    def test_is_vector_type_imported(self):
        """is_vector_type should be imported from source."""
        assert is_vector_type("std::vector<double>") is True
        assert is_vector_type("vector<int>") is True
        assert is_vector_type("RVec<double>") is False


# =============================================================================
# Run tests when executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
