"""
Phase 13.3.DSL D5+D6: ROOT Linear Algebra Type Detection Tests

Tests for TMatrixD/TVectorD type detection functions.

To run: pytest tests/test_linalg_type_detection.py -v
"""

import pytest
from typing import Dict

# Import from package
from RDataFrameDSL.type_inferrer import (
    TypeInferrer,
    VariableInfo,
    is_root_matrix_type,
    is_root_vector_type,
    is_root_linalg_type,
    get_linalg_element_type,
    ROOT_MATRIX_TYPES,
    ROOT_VECTOR_TYPES,
)


# =============================================================================
# Type Detection Function Tests
# =============================================================================

class TestRootMatrixTypeDetection:
    """Test is_root_matrix_type function."""
    
    def test_tmatrixd(self):
        assert is_root_matrix_type("TMatrixD") is True
    
    def test_tmatrixf(self):
        assert is_root_matrix_type("TMatrixF") is True
    
    def test_tmatrix_template_double(self):
        assert is_root_matrix_type("TMatrixT<double>") is True
    
    def test_tmatrix_template_float(self):
        assert is_root_matrix_type("TMatrixT<float>") is True
    
    def test_tmatrixsym(self):
        assert is_root_matrix_type("TMatrixTSym<double>") is True
    
    def test_negative_double(self):
        assert is_root_matrix_type("double") is False
    
    def test_negative_rvec(self):
        assert is_root_matrix_type("RVec<double>") is False
    
    def test_negative_tvector(self):
        assert is_root_matrix_type("TVectorD") is False
    
    def test_negative_empty(self):
        assert is_root_matrix_type("") is False
    
    def test_with_whitespace(self):
        assert is_root_matrix_type("  TMatrixD  ") is True


class TestRootVectorTypeDetection:
    """Test is_root_vector_type function."""
    
    def test_tvectord(self):
        assert is_root_vector_type("TVectorD") is True
    
    def test_tvectorf(self):
        assert is_root_vector_type("TVectorF") is True
    
    def test_tvector_template_double(self):
        assert is_root_vector_type("TVectorT<double>") is True
    
    def test_tvector_template_float(self):
        assert is_root_vector_type("TVectorT<float>") is True
    
    def test_negative_double(self):
        assert is_root_vector_type("double") is False
    
    def test_negative_rvec(self):
        assert is_root_vector_type("RVec<double>") is False
    
    def test_negative_tmatrix(self):
        assert is_root_vector_type("TMatrixD") is False
    
    def test_negative_std_vector(self):
        # std::vector is NOT a ROOT linear algebra vector
        assert is_root_vector_type("std::vector<double>") is False


class TestRootLinalgTypeDetection:
    """Test is_root_linalg_type function."""
    
    def test_matrix_is_linalg(self):
        assert is_root_linalg_type("TMatrixD") is True
        assert is_root_linalg_type("TMatrixF") is True
    
    def test_vector_is_linalg(self):
        assert is_root_linalg_type("TVectorD") is True
        assert is_root_linalg_type("TVectorF") is True
    
    def test_scalar_not_linalg(self):
        assert is_root_linalg_type("double") is False
    
    def test_rvec_not_linalg(self):
        assert is_root_linalg_type("RVec<double>") is False


class TestLinalgElementType:
    """Test get_linalg_element_type function."""
    
    def test_tmatrixd_double(self):
        assert get_linalg_element_type("TMatrixD") == "double"
    
    def test_tmatrixf_float(self):
        assert get_linalg_element_type("TMatrixF") == "float"
    
    def test_tvectord_double(self):
        assert get_linalg_element_type("TVectorD") == "double"
    
    def test_tvectorf_float(self):
        assert get_linalg_element_type("TVectorF") == "float"
    
    def test_template_double(self):
        assert get_linalg_element_type("TMatrixT<double>") == "double"
    
    def test_template_float(self):
        assert get_linalg_element_type("TVectorT<float>") == "float"
    
    def test_default_double(self):
        # Unknown types default to double
        assert get_linalg_element_type("unknown") == "double"


class TestTypeConstants:
    """Test type constant sets."""
    
    def test_matrix_types_contain_tmatrixd(self):
        assert "TMatrixD" in ROOT_MATRIX_TYPES
    
    def test_matrix_types_contain_tmatrixf(self):
        assert "TMatrixF" in ROOT_MATRIX_TYPES
    
    def test_vector_types_contain_tvectord(self):
        assert "TVectorD" in ROOT_VECTOR_TYPES
    
    def test_vector_types_contain_tvectorf(self):
        assert "TVectorF" in ROOT_VECTOR_TYPES
    
    def test_no_overlap(self):
        """Matrix and vector type sets should not overlap."""
        overlap = ROOT_MATRIX_TYPES & ROOT_VECTOR_TYPES
        assert len(overlap) == 0


# =============================================================================
# TypeInferrer Method Tests
# =============================================================================

class TestTypeInferrerLinalgMethods:
    """Test TypeInferrer linalg helper methods."""
    
    def test_is_linalg_type_for_matrix(self):
        """TypeInferrer.is_linalg_type returns True for TMatrixD."""
        schema = {"columns": {"mat": {"dtype": "double", "cpp_type": "TMatrixD", "rank": 0}}}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.is_linalg_type("mat") is True
    
    def test_is_linalg_type_for_vector(self):
        """TypeInferrer.is_linalg_type returns True for TVectorD."""
        schema = {"columns": {"vec": {"dtype": "double", "cpp_type": "TVectorD", "rank": 0}}}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.is_linalg_type("vec") is True
    
    def test_is_linalg_type_for_scalar(self):
        """TypeInferrer.is_linalg_type returns False for scalar."""
        schema = {"columns": {"x": {"dtype": "double", "cpp_type": "double", "rank": 0}}}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.is_linalg_type("x") is False
    
    def test_is_matrix_type(self):
        """TypeInferrer.is_matrix_type works correctly."""
        schema = {"columns": {
            "mat": {"dtype": "double", "cpp_type": "TMatrixD", "rank": 0},
            "vec": {"dtype": "double", "cpp_type": "TVectorD", "rank": 0},
        }}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.is_matrix_type("mat") is True
        assert inferrer.is_matrix_type("vec") is False
    
    def test_is_vector_linalg_type(self):
        """TypeInferrer.is_vector_linalg_type works correctly."""
        schema = {"columns": {
            "mat": {"dtype": "double", "cpp_type": "TMatrixD", "rank": 0},
            "vec": {"dtype": "double", "cpp_type": "TVectorD", "rank": 0},
        }}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.is_vector_linalg_type("mat") is False
        assert inferrer.is_vector_linalg_type("vec") is True
    
    def test_get_cpp_type(self):
        """TypeInferrer.get_cpp_type returns cpp_type string."""
        schema = {"columns": {"mat": {"dtype": "double", "cpp_type": "TMatrixD", "rank": 0}}}
        inferrer = TypeInferrer.from_schema(schema)
        
        assert inferrer.get_cpp_type("mat") == "TMatrixD"


# =============================================================================
# Simple Schema Tests (Phase 13.3.DSL addition)
# =============================================================================

class TestSimpleSchemaLinalgParsing:
    """Test that simple schema parses TMatrixD/TVectorD correctly."""
    
    def test_simple_schema_tmatrixd(self):
        """Simple schema with TMatrixD is parsed correctly."""
        schema = {"mat": "TMatrixD", "x": "double"}
        inferrer = TypeInferrer.from_schema(schema)
        
        mat_info = inferrer.get_variable_info("mat")
        assert mat_info.cpp_type == "TMatrixD"
        assert mat_info.rank == 0  # Treated as container, not expanded
        
    def test_simple_schema_tvectord(self):
        """Simple schema with TVectorD is parsed correctly."""
        schema = {"vec": "TVectorD"}
        inferrer = TypeInferrer.from_schema(schema)
        
        vec_info = inferrer.get_variable_info("vec")
        assert vec_info.cpp_type == "TVectorD"
        assert vec_info.rank == 0  # Treated as container
    
    def test_simple_schema_tmatrixf(self):
        """Simple schema with TMatrixF preserves float type."""
        schema = {"err": "TMatrixF"}
        inferrer = TypeInferrer.from_schema(schema)
        
        err_info = inferrer.get_variable_info("err")
        assert err_info.cpp_type == "TMatrixF"
        # Element type should be float
        assert err_info.dtype.to_cpp() == "float"
    
    def test_simple_schema_tvectorf(self):
        """Simple schema with TVectorF preserves float type."""
        schema = {"weights": "TVectorF"}
        inferrer = TypeInferrer.from_schema(schema)
        
        weights_info = inferrer.get_variable_info("weights")
        assert weights_info.cpp_type == "TVectorF"
        assert weights_info.dtype.to_cpp() == "float"
    
    def test_simple_schema_rvec(self):
        """Simple schema with RVec still works."""
        schema = {"pts": "RVec<double>"}
        inferrer = TypeInferrer.from_schema(schema)
        
        pts_info = inferrer.get_variable_info("pts")
        assert pts_info.cpp_type == "RVec<double>"
        assert pts_info.rank == 1
        assert pts_info.is_jagged is True
    
    def test_simple_schema_scalar(self):
        """Simple schema with scalar still works."""
        schema = {"x": "double"}
        inferrer = TypeInferrer.from_schema(schema)
        
        x_info = inferrer.get_variable_info("x")
        assert x_info.cpp_type == "double"
        assert x_info.rank == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
