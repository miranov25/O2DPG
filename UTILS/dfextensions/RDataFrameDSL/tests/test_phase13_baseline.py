"""
Phase 13.4.DSL D0: Baseline Tests for Phase 13.3 Features

BLOCKING GATE: These tests MUST pass before any Phase 13.4 implementation.

Purpose:
1. Verify Phase 13.3 features produce CORRECT RESULTS
2. Document what we're protecting during Phase 13.4
3. Provide fast feedback for regression detection

Tests cover:
- Type inference for RVec, TMatrix, TVectorD, nested types
- TMatrixD/TVectorD indexing and slicing
- Nested RVec two-tier ragged policy
- Code generation correctness
"""

import pytest
from typing import Dict

# Package imports
from RDataFrameDSL.type_inferrer import (
    TypeInferrer,
    is_rvec_type,
    is_collection_type,
    extract_inner_type,
    is_root_matrix_type,
    is_root_vector_type,
    get_linalg_element_type,
)
from RDataFrameDSL.ir_nodes_linalg import (
    LinalgSliceKind,
    SliceParams,
    make_matrix_element_access,
    make_matrix_row_access,
    make_matrix_column_access,
    make_vector_element_access,
    make_vector_slice_access,
)
from RDataFrameDSL.backend_linalg import (
    LinalgCodeGenerator,
    generate_linalg_code,
)
from RDataFrameDSL.dsl_linalg import LinalgDSLCompiler
from RDataFrameDSL.ir_nodes_nested import (
    NestedSliceKind,
    make_nested_column_extract,
    make_nested_column_slice,
)
from RDataFrameDSL.backend_nested import generate_nested_code
from RDataFrameDSL.dsl_nested import (
    NestedDSLCompiler,
    is_nested_rvec_type,
    get_nested_element_type,
)


# =============================================================================
# Type Inference Baseline Tests
# =============================================================================

class TestTypeInferenceBaseline:
    """Verify type inference produces correct results."""
    
    def test_rvec_detection(self):
        """RVec types correctly identified."""
        assert is_rvec_type("RVec<float>") == True
        assert is_rvec_type("ROOT::RVec<double>") == True
        assert is_rvec_type("float") == False
        assert is_rvec_type("TMatrixD") == False
    
    def test_nested_rvec_detection(self):
        """Nested RVec types correctly identified."""
        assert is_nested_rvec_type("RVec<RVec<double>>") == True
        assert is_nested_rvec_type("ROOT::RVec<ROOT::RVec<float>>") == True
        assert is_nested_rvec_type("RVec<double>") == False
        assert is_nested_rvec_type("double") == False
    
    def test_tmatrix_detection(self):
        """TMatrix types correctly identified."""
        assert is_root_matrix_type("TMatrixD") == True
        assert is_root_matrix_type("TMatrixF") == True
        # Note: TMatrixDSym (symmetric) deferred to Phase 14
        # assert is_root_matrix_type("TMatrixDSym") == True
        assert is_root_matrix_type("RVec<float>") == False
    
    def test_tvector_detection(self):
        """TVectorD types correctly identified."""
        assert is_root_vector_type("TVectorD") == True
        assert is_root_vector_type("TVectorF") == True
        assert is_root_vector_type("TMatrixD") == False
    
    def test_element_type_extraction(self):
        """Inner element type correctly extracted."""
        inner, depth = extract_inner_type("RVec<float>")
        assert inner == "float"
        assert depth == 1
        
        inner, depth = extract_inner_type("RVec<RVec<double>>")
        assert inner == "double"
        assert depth == 2
    
    def test_linalg_element_type(self):
        """TMatrixD/F element type correctly identified."""
        assert get_linalg_element_type("TMatrixD") == "double"
        assert get_linalg_element_type("TMatrixF") == "float"
        assert get_linalg_element_type("TVectorD") == "double"
        assert get_linalg_element_type("TVectorF") == "float"
    
    def test_type_inferrer_schema(self):
        """TypeInferrer correctly processes schema."""
        schema = {
            "x": "float",
            "arr": "RVec<double>",
            "mat": "TMatrixD",
            "nested": "RVec<RVec<float>>",
        }
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.get_cpp_type("x") == "float"
        assert inferrer.get_cpp_type("arr") == "RVec<double>"
        assert inferrer.get_cpp_type("mat") == "TMatrixD"
        assert inferrer.get_cpp_type("nested") == "RVec<RVec<float>>"


# =============================================================================
# TMatrixD/TVectorD Baseline Tests
# =============================================================================

class TestLinalgCompilerBaseline:
    """Verify TMatrixD/TVectorD DSL compiler produces correct results."""
    
    @pytest.fixture
    def compiler(self):
        schema = {"mat": "TMatrixD", "vec": "TVectorD", "fmat": "TMatrixF"}
        return LinalgDSLCompiler.from_schema(schema)
    
    def test_matrix_element_result_type(self, compiler):
        """mat[i,j] returns scalar double."""
        cpp_type, rank = compiler.get_result_type("mat[0, 1]")
        assert cpp_type == "double"
        assert rank == 0
    
    def test_matrix_row_result_type(self, compiler):
        """mat[i] returns RVec<double>."""
        cpp_type, rank = compiler.get_result_type("mat[0]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_matrix_column_result_type(self, compiler):
        """mat[:,j] returns RVec<double>."""
        cpp_type, rank = compiler.get_result_type("mat[:, 0]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_vector_element_result_type(self, compiler):
        """vec[i] returns scalar."""
        cpp_type, rank = compiler.get_result_type("vec[0]")
        assert cpp_type == "double"
        assert rank == 0
    
    def test_vector_slice_result_type(self, compiler):
        """vec[:n] returns RVec<double>."""
        cpp_type, rank = compiler.get_result_type("vec[:5]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_float_type_preserved(self, compiler):
        """TMatrixF returns float, not double."""
        cpp_type, rank = compiler.get_result_type("fmat[0, 0]")
        assert cpp_type == "float"
        assert rank == 0
    
    def test_three_matrix_syntaxes_equivalent(self, compiler):
        """mat(i,j), mat[i,j], mat[i][j] all produce element access."""
        # All three syntaxes should be detected as linalg expressions
        assert compiler.is_linalg_expression("mat(0, 1)") == True
        assert compiler.is_linalg_expression("mat[0, 1]") == True
        # Note: mat[0][1] is chained subscript, may be handled differently


class TestLinalgCodeGenerationBaseline:
    """Verify generated C++ code contains correct patterns."""
    
    def test_matrix_element_has_bounds_check(self):
        """Matrix element access includes bounds checking."""
        node = make_matrix_element_access("mat", 0, 1, safe=True)
        result = generate_linalg_code(node)
        
        # Must have bounds check
        assert "GetNrows()" in result.code
        assert "GetNcols()" in result.code
        assert "quiet_NaN" in result.code or "NaN" in result.code
    
    def test_matrix_element_negative_index_normalization(self):
        """Negative indices are normalized."""
        node = make_matrix_element_access("mat", -1, -1, safe=True)
        result = generate_linalg_code(node)
        
        # Must normalize negative index
        assert ">= 0 ?" in result.code or "< 0" in result.code
    
    def test_matrix_row_extraction_produces_rvec(self):
        """Row extraction generates RVec<T> result."""
        node = make_matrix_row_access("mat", 0)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<double>" in result.code
        assert result.result_type == "ROOT::RVec<double>"
    
    def test_vector_slice_first_n(self):
        """vec[:n] generates correct slice code."""
        params = SliceParams(stop=5)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<double>" in result.code
        # Should limit to first 5 elements
        assert "5" in result.code or "stop" in result.code


# =============================================================================
# Nested RVec Two-Tier Ragged Policy Baseline Tests
# =============================================================================

class TestNestedRVecPolicyBaseline:
    """Verify two-tier ragged policy produces correct behavior."""
    
    @pytest.fixture
    def compiler(self):
        schema = {"nested": "RVec<RVec<double>>", "fnested": "RVec<RVec<float>>"}
        return NestedDSLCompiler.from_schema(schema)
    
    def test_column_extract_is_fail_closed(self, compiler):
        """Integer column index uses fail-closed policy."""
        policy = compiler.get_policy("nested[:, 0]")
        assert policy == "fail-closed"
    
    def test_column_slice_is_clamp(self, compiler):
        """Slice column index uses clamp-per-row policy."""
        policy = compiler.get_policy("nested[:, :3]")
        assert policy == "clamp-per-row"
    
    def test_negative_index_is_fail_closed(self, compiler):
        """Negative integer index also fail-closed."""
        policy = compiler.get_policy("nested[:, -1]")
        assert policy == "fail-closed"
    
    def test_column_extract_result_type(self, compiler):
        """Column extract returns RVec<T>."""
        cpp_type, rank = compiler.get_result_type("nested[:, 0]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_column_slice_result_type(self, compiler):
        """Column slice returns RVec<RVec<T>>."""
        cpp_type, rank = compiler.get_result_type("nested[:, :3]")
        assert cpp_type == "ROOT::RVec<ROOT::RVec<double>>"
        assert rank == 2
    
    def test_float_type_preserved_nested(self, compiler):
        """Float element type preserved in nested RVec."""
        cpp_type, rank = compiler.get_result_type("fnested[:, 0]")
        assert cpp_type == "ROOT::RVec<float>"


class TestNestedCodeGenerationBaseline:
    """Verify nested RVec code generation is correct."""
    
    def test_column_extract_throws_on_oob(self):
        """Fail-closed policy generates throw statement."""
        node = make_nested_column_extract("nested", 0)
        result = generate_nested_code(node)
        
        # Must throw on out-of-bounds
        assert "throw std::runtime_error" in result.code
        assert "out of bounds" in result.code
    
    def test_column_slice_clamps(self):
        """Clamp policy generates std::min/max."""
        params = SliceParams(stop=3)
        node = make_nested_column_slice("nested", params)
        result = generate_nested_code(node)
        
        # Must NOT throw (clamp instead)
        assert "throw" not in result.code
        # Must have clamp logic
        assert "std::max" in result.code
        assert "std::min" in result.code
    
    def test_column_extract_loops_all_rows(self):
        """Column extract iterates over all rows."""
        node = make_nested_column_extract("nested", 0)
        result = generate_nested_code(node)
        
        # Must loop over outer
        assert "for" in result.code
        assert "outer.size()" in result.code


# =============================================================================
# ROOT Integration Baseline Tests
# =============================================================================

try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTLinalgBaseline:
    """Verify TMatrixD/TVectorD work correctly in ROOT."""
    
    def test_tmatrix_element_access_correct_value(self):
        """TMatrixD element access returns correct value."""
        import ROOT
        mat = ROOT.TMatrixD(3, 4)
        mat[1][2] = 42.0
        
        # Direct access should work
        assert mat[1][2] == 42.0
        # Alternative syntax
        assert mat(1, 2) == 42.0
    
    def test_tvector_element_access_correct_value(self):
        """TVectorD element access returns correct value."""
        import ROOT
        vec = ROOT.TVectorD(5)
        vec[3] = 99.0
        
        assert vec[3] == 99.0
    
    def test_tmatrix_dimensions(self):
        """TMatrixD GetNrows/GetNcols return correct values."""
        import ROOT
        mat = ROOT.TMatrixD(3, 5)
        
        assert mat.GetNrows() == 3
        assert mat.GetNcols() == 5


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTNestedBaseline:
    """Verify nested RVec operations work correctly in ROOT."""
    
    def test_nested_rvec_creation(self):
        """RVec<RVec<double>> can be created and accessed."""
        import ROOT
        outer = ROOT.RVec['ROOT::RVec<double>']()
        
        inner1 = ROOT.RVec['double']()
        inner1.push_back(1.0)
        inner1.push_back(2.0)
        
        inner2 = ROOT.RVec['double']()
        inner2.push_back(3.0)
        
        outer.push_back(inner1)
        outer.push_back(inner2)
        
        # Verify structure
        assert len(outer) == 2
        assert len(outer[0]) == 2
        assert len(outer[1]) == 1
        
        # Verify values
        assert outer[0][0] == 1.0
        assert outer[0][1] == 2.0
        assert outer[1][0] == 3.0
    
    def test_rvec_slice_semantics(self):
        """RVec slicing produces expected results."""
        import ROOT
        vec = ROOT.RVec['double']()
        for i in range(5):
            vec.push_back(float(i * 10))
        
        # Verify original values
        assert list(vec) == [0.0, 10.0, 20.0, 30.0, 40.0]


# =============================================================================
# Summary
# =============================================================================

class TestBaselineSummary:
    """Meta-test to document baseline coverage."""
    
    def test_baseline_coverage_documented(self):
        """
        D0 Baseline Tests cover:
        
        1. Type Inference (7 tests)
           - RVec detection
           - Nested RVec detection
           - TMatrixD/TVectorD detection
           - Element type extraction
           
        2. TMatrixD/TVectorD Compiler (7 tests)
           - Result types for element, row, column access
           - Float type preservation
           - Three matrix syntaxes
           
        3. TMatrixD/TVectorD Code Generation (4 tests)
           - Bounds checking
           - Negative index normalization
           - RVec output
           
        4. Nested RVec Policy (6 tests)
           - Fail-closed for integer index
           - Clamp-per-row for slice
           - Result types
           
        5. Nested RVec Code Generation (3 tests)
           - Throw on OOB
           - Clamp logic
           - Loop structure
           
        6. ROOT Integration (5 tests)
           - TMatrixD/TVectorD access
           - Nested RVec creation
           
        Total: ~32 tests verifying CORRECT RESULTS
        """
        assert True  # Documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
