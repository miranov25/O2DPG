"""
Phase 13.4.DSL D1: Tests for Schema Parser

Tests for C-array notation parsing:
- Fixed 1D: float[10]
- Variable 1D: float[n]
- Fixed 2D: float[3][3]
- Fixed 3D: float[2][3][4]
- Hybrid: float[n][3]

All tests verify CORRECT RESULTS, not just no-crash.
"""

import pytest
from RDataFrameDSL.schema_parser import (
    CArrayType,
    Dimension,
    DimensionKind,
    parse_carray_type,
    is_carray_notation,
    SchemaParser,
    parse_schema,
)


# =============================================================================
# is_carray_notation Tests
# =============================================================================

class TestIsCArrayNotation:
    """Test C-array notation detection."""
    
    def test_fixed_1d(self):
        """float[10] is C-array notation."""
        assert is_carray_notation("float[10]") == True
    
    def test_variable_1d(self):
        """float[n] is C-array notation."""
        assert is_carray_notation("float[n]") == True
    
    def test_fixed_2d(self):
        """float[3][3] is C-array notation."""
        assert is_carray_notation("float[3][3]") == True
    
    def test_fixed_3d(self):
        """double[2][3][4] is C-array notation."""
        assert is_carray_notation("double[2][3][4]") == True
    
    def test_hybrid(self):
        """float[n][3] is C-array notation."""
        assert is_carray_notation("float[n][3]") == True
    
    def test_rvec_not_carray(self):
        """RVec<float> is NOT C-array notation."""
        assert is_carray_notation("RVec<float>") == False
    
    def test_scalar_not_carray(self):
        """float is NOT C-array notation."""
        assert is_carray_notation("float") == False
    
    def test_tmatrix_not_carray(self):
        """TMatrixD is NOT C-array notation."""
        assert is_carray_notation("TMatrixD") == False
    
    def test_nested_rvec_not_carray(self):
        """RVec<RVec<float>> is NOT C-array notation."""
        assert is_carray_notation("RVec<RVec<float>>") == False
    
    def test_whitespace_handling(self):
        """Whitespace is stripped."""
        assert is_carray_notation("  float[10]  ") == True


# =============================================================================
# Dimension Parsing Tests
# =============================================================================

class TestDimensionParsing:
    """Test individual dimension parsing."""
    
    def test_fixed_dimension_value(self):
        """Fixed dimension has correct integer value."""
        result = parse_carray_type("float[10]")
        assert result.dims[0].value == 10
        assert result.dims[0].is_fixed == True
    
    def test_variable_dimension_value(self):
        """Variable dimension has correct counter name."""
        result = parse_carray_type("float[n]")
        assert result.dims[0].value == "n"
        assert result.dims[0].is_variable == True
    
    def test_large_fixed_dimension(self):
        """Large fixed dimensions parsed correctly."""
        result = parse_carray_type("float[1000]")
        assert result.dims[0].value == 1000
    
    def test_underscore_counter_name(self):
        """Counter names with underscores allowed."""
        result = parse_carray_type("float[n_clusters]")
        assert result.dims[0].value == "n_clusters"
    
    def test_camelcase_counter_name(self):
        """CamelCase counter names allowed."""
        result = parse_carray_type("float[nClusters]")
        assert result.dims[0].value == "nClusters"


# =============================================================================
# Fixed Array Parsing Tests
# =============================================================================

class TestFixedArrayParsing:
    """Test parsing of fully fixed arrays."""
    
    def test_fixed_1d_base_type(self):
        """float[10] has base type float."""
        result = parse_carray_type("float[10]")
        assert result.base == "float"
    
    def test_fixed_1d_rank(self):
        """float[10] has rank 1."""
        result = parse_carray_type("float[10]")
        assert result.rank == 1
    
    def test_fixed_1d_is_fixed(self):
        """float[10] is fully fixed."""
        result = parse_carray_type("float[10]")
        assert result.is_fixed == True
        assert result.is_variable == False
    
    def test_fixed_1d_no_counter(self):
        """float[10] has no counter branch."""
        result = parse_carray_type("float[10]")
        assert result.counter_branch is None
    
    def test_fixed_2d_dimensions(self):
        """float[3][4] has correct dimensions."""
        result = parse_carray_type("float[3][4]")
        assert result.rank == 2
        assert result.dims[0].value == 3
        assert result.dims[1].value == 4
    
    def test_fixed_2d_total_size(self):
        """float[3][4] has total size 12."""
        result = parse_carray_type("float[3][4]")
        assert result.total_fixed_size == 12
    
    def test_fixed_3d_dimensions(self):
        """double[2][3][4] has correct dimensions."""
        result = parse_carray_type("double[2][3][4]")
        assert result.rank == 3
        assert result.dims[0].value == 2
        assert result.dims[1].value == 3
        assert result.dims[2].value == 4
    
    def test_fixed_3d_total_size(self):
        """double[2][3][4] has total size 24."""
        result = parse_carray_type("double[2][3][4]")
        assert result.total_fixed_size == 24
    
    def test_double_base_type(self):
        """double[10] has base type double."""
        result = parse_carray_type("double[10]")
        assert result.base == "double"
    
    def test_int_base_type(self):
        """int[5] has base type int."""
        result = parse_carray_type("int[5]")
        assert result.base == "int"


# =============================================================================
# Variable Array Parsing Tests
# =============================================================================

class TestVariableArrayParsing:
    """Test parsing of variable-length arrays."""
    
    def test_variable_1d_counter(self):
        """float[n] has counter branch n."""
        result = parse_carray_type("float[n]")
        assert result.counter_branch == "n"
    
    def test_variable_1d_is_variable(self):
        """float[n] is variable."""
        result = parse_carray_type("float[n]")
        assert result.is_variable == True
        assert result.is_fixed == False
    
    def test_variable_1d_no_total_size(self):
        """float[n] has no total fixed size."""
        result = parse_carray_type("float[n]")
        assert result.total_fixed_size is None
    
    def test_hybrid_2d_counter(self):
        """float[n][3] has counter branch n."""
        result = parse_carray_type("float[n][3]")
        assert result.counter_branch == "n"
    
    def test_hybrid_2d_dimensions(self):
        """float[n][3] has correct dimensions."""
        result = parse_carray_type("float[n][3]")
        assert result.rank == 2
        assert result.dims[0].is_variable == True
        assert result.dims[0].value == "n"
        assert result.dims[1].is_fixed == True
        assert result.dims[1].value == 3
    
    def test_hybrid_2d_inner_fixed_size(self):
        """float[n][3] has inner fixed size 3."""
        result = parse_carray_type("float[n][3]")
        assert result.inner_fixed_size == 3
    
    def test_hybrid_3d(self):
        """float[n][3][4] has correct structure."""
        result = parse_carray_type("float[n][3][4]")
        assert result.rank == 3
        assert result.counter_branch == "n"
        assert result.dims[0].value == "n"
        assert result.dims[1].value == 3
        assert result.dims[2].value == 4
        assert result.inner_fixed_size == 12  # 3 × 4


# =============================================================================
# Stride Calculation Tests
# =============================================================================

class TestStrideCalculation:
    """Test row-major stride calculations."""
    
    def test_1d_stride(self):
        """1D array has stride 1."""
        result = parse_carray_type("float[10]")
        assert result.get_stride(0) == "1"
    
    def test_2d_stride_row(self):
        """2D array row stride is number of columns."""
        result = parse_carray_type("float[3][4]")
        assert result.get_stride(0) == "4"
    
    def test_2d_stride_col(self):
        """2D array column stride is 1."""
        result = parse_carray_type("float[3][4]")
        assert result.get_stride(1) == "1"
    
    def test_3d_stride_dim0(self):
        """3D array dim0 stride is D1 * D2."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_stride(0) == "3 * 4"
    
    def test_3d_stride_dim1(self):
        """3D array dim1 stride is D2."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_stride(1) == "4"
    
    def test_3d_stride_dim2(self):
        """3D array dim2 stride is 1."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_stride(2) == "1"
    
    def test_hybrid_stride(self):
        """Hybrid array uses counter name in stride."""
        result = parse_carray_type("float[n][3]")
        # dim0 stride should reference the fixed dimension
        assert result.get_stride(0) == "3"


# =============================================================================
# Result Type Tests
# =============================================================================

class TestResultType:
    """Test result type inference for indexing."""
    
    def test_1d_element_access(self):
        """arr[i] on float[10] returns float."""
        result = parse_carray_type("float[10]")
        assert result.get_result_type(access_rank=1) == "float"
    
    def test_2d_row_access(self):
        """arr[i] on float[3][4] returns RVec<float>."""
        result = parse_carray_type("float[3][4]")
        assert result.get_result_type(access_rank=1) == "ROOT::RVec<float>"
    
    def test_2d_element_access(self):
        """arr[i,j] on float[3][4] returns float."""
        result = parse_carray_type("float[3][4]")
        assert result.get_result_type(access_rank=2) == "float"
    
    def test_3d_slice_access(self):
        """arr[i] on float[2][3][4] returns RVec<RVec<float>>."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_result_type(access_rank=1) == "ROOT::RVec<ROOT::RVec<float>>"
    
    def test_3d_row_access(self):
        """arr[i,j] on float[2][3][4] returns RVec<float>."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_result_type(access_rank=2) == "ROOT::RVec<float>"
    
    def test_3d_element_access(self):
        """arr[i,j,k] on float[2][3][4] returns float."""
        result = parse_carray_type("float[2][3][4]")
        assert result.get_result_type(access_rank=3) == "float"
    
    def test_double_base_type_preserved(self):
        """double base type preserved in result."""
        result = parse_carray_type("double[3][4]")
        assert result.get_result_type(access_rank=1) == "ROOT::RVec<double>"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling for invalid notations."""
    
    def test_empty_string_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            parse_carray_type("")
    
    def test_not_carray_error(self):
        """Non-C-array notation raises ValueError."""
        with pytest.raises(ValueError, match="Not a C-array"):
            parse_carray_type("RVec<float>")
    
    def test_zero_dimension_error(self):
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            parse_carray_type("float[0]")
    
    def test_negative_dimension_error(self):
        """Negative dimension raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            parse_carray_type("float[-1]")
    
    def test_multiple_variable_dims_error(self):
        """Multiple variable dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Multiple variable"):
            parse_carray_type("float[n][m]")
    
    def test_variable_not_outermost_error(self):
        """Variable dimension not outermost raises ValueError."""
        with pytest.raises(ValueError, match="outermost"):
            parse_carray_type("float[3][n]")
    
    def test_invalid_dimension_syntax_error(self):
        """Invalid dimension syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            parse_carray_type("float[3.5]")


# =============================================================================
# SchemaParser Tests
# =============================================================================

class TestSchemaParser:
    """Test SchemaParser class."""
    
    def test_parse_mixed_schema(self):
        """Mixed schema parsed correctly."""
        schema = {
            "n": "int",
            "arr": "float[n]",
            "mat": "float[3][3]",
            "vec": "RVec<double>",
        }
        parser = SchemaParser()
        parsed = parser.parse(schema)
        
        # Scalars/RVec remain strings
        assert parsed["n"] == "int"
        assert parsed["vec"] == "RVec<double>"
        
        # C-arrays become CArrayType
        assert isinstance(parsed["arr"], CArrayType)
        assert isinstance(parsed["mat"], CArrayType)
    
    def test_get_counter_branches(self):
        """Counter branches extracted correctly."""
        schema = {
            "n": "int",
            "arr": "float[n]",
            "hits": "float[n][3]",
            "mat": "float[3][3]",
        }
        parser = SchemaParser()
        counters = parser.get_counter_branches(schema)
        
        assert "n" in counters
        assert set(counters["n"]) == {"arr", "hits"}
    
    def test_validate_missing_counter(self):
        """Validation detects missing counter branch."""
        schema = {
            "arr": "float[n]",  # n not defined!
        }
        parser = SchemaParser()
        issues = parser.validate_schema(schema)
        
        assert len(issues) > 0
        assert "n" in issues[0]
    
    def test_validate_valid_schema(self):
        """Valid schema has no issues."""
        schema = {
            "n": "int",
            "arr": "float[n]",
        }
        parser = SchemaParser()
        issues = parser.validate_schema(schema)
        
        assert len(issues) == 0
    
    def test_caching(self):
        """Parser caches results."""
        parser = SchemaParser()
        
        result1 = parser.parse_type("float[10]")
        result2 = parser.parse_type("float[10]")
        
        # Should be same object (cached)
        assert result1 is result2


# =============================================================================
# Integration Tests
# =============================================================================

class TestSchemaParserIntegration:
    """Integration tests for complete parsing workflow."""
    
    def test_physics_schema(self):
        """Parse realistic physics schema."""
        schema = {
            "nTracks": "int",
            "trackPt": "float[nTracks]",
            "trackCov": "float[nTracks][6]",  # 6-element covariance per track
            "vertex": "float[3]",
            "eventMatrix": "double[4][4]",
        }
        parser = SchemaParser()
        parsed = parser.parse(schema)
        
        # Verify structure
        assert parsed["nTracks"] == "int"
        
        assert isinstance(parsed["trackPt"], CArrayType)
        assert parsed["trackPt"].counter_branch == "nTracks"
        assert parsed["trackPt"].rank == 1
        
        assert isinstance(parsed["trackCov"], CArrayType)
        assert parsed["trackCov"].counter_branch == "nTracks"
        assert parsed["trackCov"].rank == 2
        assert parsed["trackCov"].inner_fixed_size == 6
        
        assert isinstance(parsed["vertex"], CArrayType)
        assert parsed["vertex"].is_fixed == True
        assert parsed["vertex"].total_fixed_size == 3
        
        assert isinstance(parsed["eventMatrix"], CArrayType)
        assert parsed["eventMatrix"].base == "double"
        assert parsed["eventMatrix"].total_fixed_size == 16
    
    def test_original_preserved(self):
        """Original notation string preserved."""
        result = parse_carray_type("float[n][3]")
        assert result.original == "float[n][3]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
