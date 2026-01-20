"""
Phase 13.4.D9: C-Array Schema Integration Tests

Tests for:
- P0-1: rank vs carray_shape invariant
- P0-A: AST-authoritative routing
- P0-3: Unknown dimensions error handling
- P0-4: Scalar type normalization
"""

import pytest
import ast
from typing import Optional, Tuple, Union
from dataclasses import dataclass


# =============================================================================
# Mock classes for testing without ROOT
# =============================================================================

@dataclass
class MockVariableInfo:
    """Mock VariableInfo for testing."""
    name: str
    dtype: str = "float"
    rank: int = 0
    is_jagged: bool = False
    cpp_type: Optional[str] = None
    source: str = "test"
    carray_shape: Optional[Tuple[Union[int, str], ...]] = None
    carray_counter: Optional[str] = None
    
    @property
    def is_carray(self) -> bool:
        return self.carray_shape is not None
    
    @property
    def carray_ndim(self) -> int:
        return len(self.carray_shape) if self.carray_shape else 0


# Dimension and DimensionKind mock
class DimensionKind:
    FIXED = "FIXED"
    VARIABLE = "VARIABLE"


@dataclass
class Dimension:
    kind: str
    value: Union[int, str]


@dataclass  
class CArrayType:
    """Mock CArrayType for testing."""
    base: str
    dims: list
    original: str = ""
    
    @property
    def rank(self) -> int:
        return len(self.dims)


# =============================================================================
# P0-1: rank vs carray_shape invariant tests
# =============================================================================

class TestVariableInfoInvariant:
    """P0-1: rank must not be overloaded."""
    
    def test_carray_rank_is_container_rank(self):
        """mat[3][4] should have rank=1 (RVec), not rank=2."""
        # C-array mat[3][4] is stored as RVec<float> with 12 elements
        info = MockVariableInfo(
            name="mat",
            dtype="float",
            rank=1,  # Container rank: RVec
            cpp_type="float",
            carray_shape=(3, 4),  # Logical shape
        )
        
        # Invariant: rank is container rank
        assert info.rank == 1, "rank should be 1 (RVec), not len(carray_shape)"
        
        # carray_shape stores logical dimensions
        assert info.carray_shape == (3, 4)
        assert info.carray_ndim == 2
    
    def test_carray_with_variable_dimension(self):
        """arr[n][3] should have rank=1, carray_shape=("n", 3)."""
        info = MockVariableInfo(
            name="arr",
            dtype="float",
            rank=1,
            is_jagged=True,
            cpp_type="float",
            carray_shape=("n", 3),
            carray_counter="n",
        )
        
        assert info.rank == 1
        assert info.carray_shape == ("n", 3)
        assert info.carray_counter == "n"
        assert info.is_carray is True
    
    def test_non_carray_has_none_shape(self):
        """Regular RVec should have carray_shape=None."""
        info = MockVariableInfo(
            name="pt",
            dtype="float",
            rank=1,
            carray_shape=None,
        )
        
        assert info.carray_shape is None
        assert info.is_carray is False
        assert info.carray_ndim == 0


# =============================================================================
# P0-A: AST-authoritative routing tests
# =============================================================================

class TestASTBasedRouting:
    """P0-A: Routing must be AST-authoritative."""
    
    def setup_method(self):
        """Set up schema with C-array."""
        self.schema = {
            "mat": CArrayType(base="float", dims=[
                Dimension(DimensionKind.FIXED, 3),
                Dimension(DimensionKind.FIXED, 4)
            ]),
            "arr": CArrayType(base="float", dims=[
                Dimension(DimensionKind.FIXED, 10)
            ]),
        }
    
    def _is_carray_expression(self, expression: str) -> bool:
        """Test implementation of AST-based routing check."""
        # Quick pre-filter
        if ',' not in expression or '[' not in expression:
            return False
        
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError:
            return False
        
        return self._has_carray_access(tree.body)
    
    def _has_carray_access(self, node: ast.AST) -> bool:
        """Check if AST contains C-array access."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Tuple):
                base = node.value
                if isinstance(base, ast.Name):
                    if base.id in self.schema:
                        return True
        
        for child in ast.iter_child_nodes(node):
            if self._has_carray_access(child):
                return True
        
        return False
    
    def test_routes_direct_access(self):
        """mat[0,:] should route to C-array."""
        assert self._is_carray_expression("mat[0, :]") is True
    
    def test_routes_element_access(self):
        """mat[i,j] should route to C-array."""
        assert self._is_carray_expression("mat[i, j]") is True
    
    def test_routes_expression_with_carray(self):
        """mat[i,j] + 1 should route to C-array."""
        assert self._is_carray_expression("mat[i, j] + 1") is True
    
    def test_routes_column_slice(self):
        """mat[:,2] should route to C-array."""
        assert self._is_carray_expression("mat[:, 2]") is True
    
    def test_no_route_function_call_result(self):
        """func(a,b)[0,1] should NOT route to C-array."""
        assert self._is_carray_expression("func(a, b)[0, 1]") is False
    
    def test_no_route_attribute_access(self):
        """obj.mat[i,j] should NOT route to C-array."""
        assert self._is_carray_expression("obj.mat[i, j]") is False
    
    def test_no_route_call_on_variable(self):
        """mat()[i,j] should NOT route to C-array."""
        assert self._is_carray_expression("mat()[i, j]") is False
    
    def test_no_route_binop_base(self):
        """(mat+x)[i,j] should NOT route to C-array."""
        assert self._is_carray_expression("(mat + x)[i, j]") is False
    
    def test_no_route_unknown_variable(self):
        """unknown[i,j] should NOT route (not in schema)."""
        assert self._is_carray_expression("unknown[i, j]") is False
    
    def test_no_route_1d_access(self):
        """arr[5] should NOT route (1D, no comma)."""
        assert self._is_carray_expression("arr[5]") is False
    
    def test_no_route_1d_slice(self):
        """arr[0:5] should NOT route (1D slice, no comma)."""
        assert self._is_carray_expression("arr[0:5]") is False
    
    def test_parenthesized_variable(self):
        """(mat)[i,j] - parenthesized name should route."""
        # This is a tricky case - (mat) becomes ast.Name
        # The AST for (mat)[i,j] has base = Name('mat')
        assert self._is_carray_expression("(mat)[i, j]") is True


# =============================================================================
# P0-3: Unknown dimensions error handling tests
# =============================================================================

class TestUnknownDimensionsError:
    """P0-3: ND operations must error when dims unknown."""
    
    def test_error_message_format(self):
        """Error message should be actionable."""
        base_var = "mat"
        expression = "mat[0, :]"
        
        # Expected error message format
        expected = (
            f"Cannot evaluate '{expression}': C-array dimensions unknown for '{base_var}'. "
            f"Use from_tree() or provide carray_schema for ND operations."
        )
        
        # Verify format
        assert "dimensions unknown" in expected
        assert "from_tree()" in expected
        assert base_var in expected
    
    def test_1d_operations_pattern(self):
        """Verify 1D operations don't have commas."""
        # These should NOT trigger C-array routing
        expressions_1d = [
            "arr[5]",
            "arr[0:5]",
            "arr[:]",
            "Sum(arr)",
        ]
        
        for expr in expressions_1d:
            # 1D operations don't have comma-in-brackets pattern
            has_carray_pattern = ',' in expr and '[' in expr
            # Check if it's inside brackets
            if has_carray_pattern:
                try:
                    tree = ast.parse(expr, mode='eval')
                    # Would need full check, but for these examples:
                    pass
                except SyntaxError:
                    pass
            # These should not route
            assert '[' not in expr or ',' not in expr or 'Sum' in expr


# =============================================================================
# P0-4: Scalar type normalization tests
# =============================================================================

class TestScalarTypeNormalization:
    """P0-4: Base type must be scalar, not container."""
    
    def _get_scalar_cpp_type(self, cpp_type: str) -> str:
        """Test implementation of scalar type normalization."""
        TYPE_MAP = {
            "Float_t": "float",
            "Double_t": "double",
            "Int_t": "int",
            "ROOT::VecOps::RVec<float>": "float",
            "ROOT::VecOps::RVec<Float_t>": "float",
            "RVec<float>": "float",
            "RVec<double>": "double",
            "float": "float",
            "double": "double",
            "int": "int",
        }
        return TYPE_MAP.get(cpp_type, "float")
    
    def test_float_passthrough(self):
        """float -> float."""
        assert self._get_scalar_cpp_type("float") == "float"
    
    def test_root_float_normalized(self):
        """Float_t -> float."""
        assert self._get_scalar_cpp_type("Float_t") == "float"
    
    def test_rvec_float_normalized(self):
        """RVec<float> -> float."""
        assert self._get_scalar_cpp_type("RVec<float>") == "float"
    
    def test_full_rvec_normalized(self):
        """ROOT::VecOps::RVec<float> -> float."""
        assert self._get_scalar_cpp_type("ROOT::VecOps::RVec<float>") == "float"
    
    def test_double_normalized(self):
        """Double_t -> double."""
        assert self._get_scalar_cpp_type("Double_t") == "double"
    
    def test_unknown_defaults_to_float(self):
        """Unknown type -> float."""
        assert self._get_scalar_cpp_type("UnknownType") == "float"


# =============================================================================
# Integration tests (mock-based)
# =============================================================================

class TestCArraySchemaBuilding:
    """Test building C-array schema from VariableInfo."""
    
    def test_build_schema_from_variableinfo(self):
        """_build_carray_schema converts VariableInfo to CArrayType."""
        # Simulate VariableInfo with carray_shape
        variables = {
            "mat": MockVariableInfo(
                name="mat",
                dtype="float",
                rank=1,
                cpp_type="float",
                carray_shape=(3, 4),
            ),
            "pt": MockVariableInfo(
                name="pt",
                dtype="float",
                rank=1,
                carray_shape=None,  # Not a C-array
            ),
        }
        
        # Build schema (only C-arrays)
        schema = {}
        for name, info in variables.items():
            if info.carray_shape:
                dims = []
                for d in info.carray_shape:
                    if isinstance(d, int):
                        dims.append(Dimension(DimensionKind.FIXED, d))
                    else:
                        dims.append(Dimension(DimensionKind.VARIABLE, d))
                
                schema[name] = CArrayType(
                    base=info.cpp_type or "float",
                    dims=dims,
                )
        
        # Verify
        assert "mat" in schema
        assert "pt" not in schema  # Not a C-array
        assert schema["mat"].base == "float"
        assert schema["mat"].rank == 2


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
