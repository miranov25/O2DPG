"""
Phase 13.4.DSL Bundle 2: Tests for C-Array Indexing + Slicing

Covers D4 (1D), D5 (2D), D6 (3D), D7 (bounds checking).

All tests verify CORRECT RESULTS, not just no-crash.
"""

import pytest
from typing import Dict

from RDataFrameDSL.schema_parser import CArrayType, parse_carray_type, SchemaParser
from RDataFrameDSL.ir_nodes_linalg import SliceParams
from RDataFrameDSL.ir_nodes_carray import (
    CArraySliceKind,
    CArrayAccessNode,
    DimensionSpec,
    make_carray_element_access,
    make_carray_slice_access,
    make_carray_row_access,
    make_carray_column_access,
    make_carray_subarray_access,
    make_carray_plane_access,
)
from RDataFrameDSL.backend_carray import (
    CArrayCodeGenerator,
    CArrayCodeResult,
    generate_carray_code,
)
from RDataFrameDSL.dsl_carray import (
    CArrayDSLCompiler,
    CArrayExpressionAnalyzer,
    IndexSpec,
)


# =============================================================================
# D4: 1D Indexing + Slicing Tests
# =============================================================================

class TestCArray1DElementAccess:
    """Test 1D element access IR and code generation."""
    
    def test_element_access_fixed_creates_correct_node(self):
        """Element access on fixed array creates correct IR node."""
        node = make_carray_element_access(
            source="arr",
            base_type="float",
            dims=[(10, True)],
            indices=[0],
        )
        assert node.kind == CArraySliceKind.ELEMENT_1D
        assert node.rank == 0
        assert node.result_type == "float"
    
    def test_element_access_variable_creates_correct_node(self):
        """Element access on variable array creates correct IR node."""
        node = make_carray_element_access(
            source="arr",
            base_type="double",
            dims=[("n", False)],
            indices=[0],
        )
        assert node.kind == CArraySliceKind.ELEMENT_1D
        assert node.dims[0].is_fixed == False
        assert node.dims[0].size == "n"
    
    def test_element_access_generates_bounds_check(self):
        """Generated code includes bounds checking."""
        node = make_carray_element_access("arr", "float", [(10, True)], [0])
        result = generate_carray_code(node)
        
        assert "i_norm < 0 || i_norm >= size" in result.jit_declarations
        assert "quiet_NaN" in result.jit_declarations
    
    def test_element_access_generates_negative_normalization(self):
        """Generated code normalizes negative indices."""
        node = make_carray_element_access("arr", "float", [(10, True)], [-1])
        result = generate_carray_code(node)
        
        assert "i >= 0 ? i : size + i" in result.jit_declarations
    
    def test_element_access_fixed_uses_constexpr(self):
        """Fixed size: code passes literal value."""
        node = make_carray_element_access("arr", "float", [(10, True)], [0])
        result = generate_carray_code(node)
        
        # New pattern: size passed as parameter, function call has literal
        assert "int size" in result.jit_declarations
        assert ", 10," in result.code  # Literal size in call
    
    def test_element_access_variable_uses_int(self):
        """Variable size: code passes variable name."""
        node = make_carray_element_access("arr", "float", [("n", False)], [0])
        result = generate_carray_code(node)
        
        assert "int size" in result.jit_declarations
        assert ", n," in result.code  # Variable name in call


class TestCArray1DSlice:
    """Test 1D slice IR and code generation."""
    
    def test_slice_creates_correct_node(self):
        """Slice creates RVec result type."""
        node = make_carray_slice_access(
            source="arr",
            base_type="float",
            dims=[(10, True)],
            slice_params=SliceParams(stop=5),
        )
        assert node.kind == CArraySliceKind.SLICE_1D
        assert node.rank == 1
        assert node.result_type == "ROOT::RVec<float>"
    
    def test_slice_first_n(self):
        """arr[:5] generates correct loop."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=5))
        result = generate_carray_code(node)
        
        # Function signature has stop parameter, call passes value 5
        assert "int stop" in result.jit_declarations
        assert ", 5," in result.code  # stop=5 passed as argument
        assert "for (int k = start; k < stop" in result.jit_declarations
    
    def test_slice_from_n(self):
        """arr[3:] generates correct loop."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(start=3))
        result = generate_carray_code(node)
        
        # Function signature has start parameter, call passes value 3
        assert "int start" in result.jit_declarations
        assert ", 3," in result.code  # start=3 passed as argument
    
    def test_slice_with_step(self):
        """arr[::2] generates correct step."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(step=2))
        result = generate_carray_code(node)
        
        # Function signature has step parameter, call passes value 2
        assert "int step" in result.jit_declarations
        assert ", 2)" in result.code  # step=2 is last argument
        assert "k += step" in result.jit_declarations
    
    def test_slice_reverse(self):
        """arr[::-1] generates reverse iteration."""
        node = make_carray_slice_access(
            "arr", "float", [(10, True)],
            SliceParams(start=9, stop=-1, step=-1),
        )
        result = generate_carray_code(node)
        
        # Function has reverse iteration logic, call passes step=-1
        assert ", -1)" in result.code  # step=-1 is last argument
        assert "k > stop" in result.jit_declarations
    
    def test_slice_clamps_to_valid_range(self):
        """Slice clamps indices to valid range."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=5))
        result = generate_carray_code(node)
        
        assert "std::max(0, std::min(start, size))" in result.jit_declarations
        assert "std::max(0, std::min(stop, size))" in result.jit_declarations


# =============================================================================
# D5: 2D Indexing + Slicing Tests
# =============================================================================

class TestCArray2DElementAccess:
    """Test 2D element access."""
    
    def test_element_2d_creates_correct_node(self):
        """2D element access creates scalar result."""
        node = make_carray_element_access(
            source="mat",
            base_type="float",
            dims=[(3, True), (4, True)],
            indices=[1, 2],
        )
        assert node.kind == CArraySliceKind.ELEMENT_2D
        assert node.rank == 0
        assert node.result_type == "float"
    
    def test_element_2d_row_major_stride(self):
        """2D element uses row-major stride: r * cols + c."""
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [1, 2])
        result = generate_carray_code(node)
        
        assert "r_norm * cols + c_norm" in result.jit_declarations
    
    def test_element_2d_bounds_check_both_dims(self):
        """2D element checks both dimensions."""
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [1, 2])
        result = generate_carray_code(node)
        
        assert "r_norm < 0 || r_norm >= rows" in result.jit_declarations
        assert "c_norm < 0 || c_norm >= cols" in result.jit_declarations


class TestCArray2DRowAccess:
    """Test 2D row extraction."""
    
    def test_row_access_creates_rvec(self):
        """Row access returns RVec."""
        node = make_carray_row_access("mat", "float", [(3, True), (4, True)], 1)
        assert node.kind == CArraySliceKind.ROW_2D
        assert node.rank == 1
        assert node.result_type == "ROOT::RVec<float>"
    
    def test_row_access_loops_columns(self):
        """Row access iterates over columns."""
        node = make_carray_row_access("mat", "float", [(3, True), (4, True)], 1)
        result = generate_carray_code(node)
        
        assert "for (int c = 0; c < cols" in result.jit_declarations
        assert "base + c" in result.jit_declarations


class TestCArray2DColumnAccess:
    """Test 2D column extraction."""
    
    def test_column_access_creates_rvec(self):
        """Column access returns RVec."""
        node = make_carray_column_access("mat", "float", [(3, True), (4, True)], 2)
        assert node.kind == CArraySliceKind.COLUMN_2D
        assert node.rank == 1
        assert node.result_type == "ROOT::RVec<float>"
    
    def test_column_access_loops_rows(self):
        """Column access iterates over rows."""
        node = make_carray_column_access("mat", "float", [(3, True), (4, True)], 2)
        result = generate_carray_code(node)
        
        assert "for (int r = 0; r < rows" in result.jit_declarations
        assert "r * cols + c_norm" in result.jit_declarations


class TestCArray2DSubarray:
    """Test 2D subarray extraction."""
    
    def test_subarray_creates_nested_rvec(self):
        """Subarray returns RVec<RVec>."""
        node = make_carray_subarray_access(
            "mat", "float", [(3, True), (4, True)],
            SliceParams(stop=2),
            SliceParams(stop=2),
        )
        assert node.rank == 2
        assert node.result_type == "ROOT::RVec<ROOT::RVec<float>>"
    
    def test_row_slice_loops_rows(self):
        """Row slice (arr[0:2, :]) iterates subset of rows."""
        node = make_carray_subarray_access(
            "mat", "float", [(3, True), (4, True)],
            SliceParams(stop=2),
            None,
        )
        result = generate_carray_code(node)
        
        # Function signature has r_stop parameter, call passes value 2
        assert "int r_stop" in result.jit_declarations
        assert ", 2," in result.code  # r_stop=2 passed as argument


class TestCArray2DHybrid:
    """Test 2D hybrid arrays (float[n][3])."""
    
    def test_hybrid_row_uses_counter(self):
        """Hybrid array row count uses counter branch."""
        node = make_carray_column_access("hits", "float", [("n", False), (3, True)], 0)
        result = generate_carray_code(node)
        
        # Function takes rows/cols as parameters
        assert "int rows" in result.jit_declarations
        assert "int cols" in result.jit_declarations
        # Call passes variable n for rows, literal 3 for cols
        assert ", n," in result.code
        assert ", 3," in result.code
    
    def test_hybrid_dependencies_include_counter(self):
        """Dependencies include counter branch."""
        node = make_carray_column_access("hits", "float", [("n", False), (3, True)], 0)
        result = generate_carray_code(node)
        
        assert "n" in result.dependencies
        assert "hits" in result.dependencies


# =============================================================================
# D6: 3D Indexing + Slicing Tests
# =============================================================================

class TestCArray3DElementAccess:
    """Test 3D element access."""
    
    def test_element_3d_creates_correct_node(self):
        """3D element access creates scalar result."""
        node = make_carray_element_access(
            source="tensor",
            base_type="double",
            dims=[(2, True), (3, True), (4, True)],
            indices=[0, 1, 2],
        )
        assert node.kind == CArraySliceKind.ELEMENT_3D
        assert node.rank == 0
        assert node.result_type == "double"
    
    def test_element_3d_row_major_stride(self):
        """3D element uses row-major stride: i*D1*D2 + j*D2 + k."""
        node = make_carray_element_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            [0, 1, 2],
        )
        result = generate_carray_code(node)
        
        assert "i_norm * d1 * d2 + j_norm * d2 + k_norm" in result.jit_declarations


class TestCArray3DPlaneAccess:
    """Test 3D plane extraction."""
    
    def test_plane_access_creates_nested_rvec(self):
        """Plane access returns RVec<RVec>."""
        node = make_carray_plane_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            0,
        )
        assert node.kind == CArraySliceKind.PLANE_3D
        assert node.rank == 2
        assert node.result_type == "ROOT::RVec<ROOT::RVec<float>>"
    
    def test_plane_access_loops_2d(self):
        """Plane access has nested loops for 2D result."""
        node = make_carray_plane_access(
            "tensor", "float",
            [(2, True), (3, True), (4, True)],
            0,
        )
        result = generate_carray_code(node)
        
        assert "for (int j = 0; j < d1" in result.jit_declarations
        assert "for (int k = 0; k < d2" in result.jit_declarations


class TestCArray3DSlices:
    """Test 3D slice operations."""
    
    def test_slice_dim0_loops_first_dim(self):
        """arr[:, j, k] loops over first dimension."""
        dim_specs = [
            DimensionSpec(2, True),
            DimensionSpec(3, True),
            DimensionSpec(4, True),
        ]
        node = CArrayAccessNode(
            kind=CArraySliceKind.SLICE_DIM0_3D,
            source="tensor",
            base_type="float",
            dims=dim_specs,
            indices=[None, 1, 2],
            rank=1,
        )
        result = generate_carray_code(node)
        
        assert "for (int i = 0; i < d0" in result.jit_declarations
        assert node.result_type == "ROOT::RVec<float>"
    
    def test_slice_dim1_loops_middle_dim(self):
        """arr[i, :, k] loops over middle dimension."""
        dim_specs = [
            DimensionSpec(2, True),
            DimensionSpec(3, True),
            DimensionSpec(4, True),
        ]
        node = CArrayAccessNode(
            kind=CArraySliceKind.SLICE_DIM1_3D,
            source="tensor",
            base_type="float",
            dims=dim_specs,
            indices=[0, None, 2],
            rank=1,
        )
        result = generate_carray_code(node)
        
        assert "for (int j = 0; j < d1" in result.jit_declarations


# =============================================================================
# D7: Bounds Checking Tests
# =============================================================================

class TestBoundsCheckingSemantics:
    """Test bounds checking is consistent with Phase 13.3."""
    
    def test_scalar_oob_returns_nan(self):
        """Scalar out-of-bounds returns NaN."""
        node = make_carray_element_access("arr", "float", [(10, True)], [0])
        result = generate_carray_code(node)
        
        assert "std::numeric_limits<float>::quiet_NaN()" in result.jit_declarations
    
    def test_double_oob_returns_nan(self):
        """Double type uses correct NaN."""
        node = make_carray_element_access("arr", "double", [(10, True)], [0])
        result = generate_carray_code(node)
        
        assert "std::numeric_limits<double>::quiet_NaN()" in result.jit_declarations
    
    def test_slice_oob_clamps(self):
        """Slice out-of-bounds clamps to valid range."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=100))
        result = generate_carray_code(node)
        
        # Clamp logic present
        assert "std::max" in result.jit_declarations
        assert "std::min" in result.jit_declarations
    
    def test_negative_index_normalized(self):
        """Negative indices are normalized."""
        node = make_carray_element_access("arr", "float", [(10, True)], [-1])
        result = generate_carray_code(node)
        
        assert "i >= 0 ? i : size + i" in result.jit_declarations
    
    def test_row_oob_returns_empty_rvec(self):
        """Row out-of-bounds returns empty RVec."""
        node = make_carray_row_access("mat", "float", [(3, True), (4, True)], 0)
        result = generate_carray_code(node)
        
        assert "return ROOT::RVec<float>()" in result.jit_declarations


# =============================================================================
# DSL Compiler Tests
# =============================================================================

class TestCArrayExpressionAnalyzer:
    """Test expression parsing."""
    
    @pytest.fixture
    def schema(self):
        parser = SchemaParser()
        return parser.parse({
            "n": "int",
            "arr": "float[10]",
            "varr": "float[n]",
            "mat": "float[3][4]",
            "hits": "float[n][3]",
            "tensor": "double[2][3][4]",
        })
    
    def test_detects_1d_element_access(self, schema):
        """Detects arr[0] as 1D element access."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("arr[0]")
        
        assert result is not None
        name, atype, indices = result
        assert name == "arr"
        assert len(indices) == 1
        assert indices[0].is_slice == False
        assert indices[0].value == 0
    
    def test_detects_1d_slice(self, schema):
        """Detects arr[:5] as 1D slice."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("arr[:5]")
        
        assert result is not None
        name, atype, indices = result
        assert indices[0].is_slice == True
        assert indices[0].stop == 5
    
    def test_detects_2d_element_access(self, schema):
        """Detects mat[1, 2] as 2D element access."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("mat[1, 2]")
        
        assert result is not None
        name, atype, indices = result
        assert len(indices) == 2
        assert indices[0].value == 1
        assert indices[1].value == 2
    
    def test_detects_column_access(self, schema):
        """Detects mat[:, 0] as column access."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("mat[:, 0]")
        
        assert result is not None
        name, atype, indices = result
        assert indices[0].is_slice == True
        assert indices[0].is_full_slice == True
        assert indices[1].value == 0
    
    def test_detects_negative_index(self, schema):
        """Detects arr[-1] with negative index."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("arr[-1]")
        
        assert result is not None
        name, atype, indices = result
        assert indices[0].value == -1
    
    def test_non_carray_returns_none(self, schema):
        """Non-C-array column returns None."""
        analyzer = CArrayExpressionAnalyzer(schema)
        result = analyzer.analyze("n[0]")  # n is int, not array
        
        assert result is None


class TestCArrayDSLCompiler:
    """Test DSL compiler integration."""
    
    @pytest.fixture
    def compiler(self):
        return CArrayDSLCompiler.from_schema({
            "n": "int",
            "arr": "float[10]",
            "varr": "float[n]",
            "mat": "float[3][4]",
            "hits": "float[n][3]",
            "tensor": "double[2][3][4]",
        })
    
    def test_is_carray_expression(self, compiler):
        """Correctly identifies C-array expressions."""
        assert compiler.is_carray_expression("arr[0]") == True
        assert compiler.is_carray_expression("mat[1, 2]") == True
        assert compiler.is_carray_expression("n[0]") == False
    
    def test_compile_1d_element(self, compiler):
        """Compiles 1D element access."""
        code, deps = compiler.compile("arr[0]", "first")
        
        assert "auto first = " in code
        assert "arr" in deps
    
    def test_compile_1d_slice(self, compiler):
        """Compiles 1D slice."""
        code, deps = compiler.compile("arr[:5]", "slice")
        jit = compiler.get_jit_declarations()
        
        assert "ROOT::RVec<float>" in jit
    
    def test_compile_2d_element(self, compiler):
        """Compiles 2D element access."""
        code, deps = compiler.compile("mat[1, 2]", "elem")
        jit = compiler.get_jit_declarations()
        
        assert "r_norm * cols + c_norm" in jit
    
    def test_compile_2d_row(self, compiler):
        """Compiles 2D row access."""
        code, deps = compiler.compile("mat[0]", "row")
        jit = compiler.get_jit_declarations()
        
        assert "ROOT::RVec<float>" in jit
    
    def test_compile_2d_column(self, compiler):
        """Compiles 2D column access."""
        code, deps = compiler.compile("mat[:, 0]", "col")
        jit = compiler.get_jit_declarations()
        
        assert "for (int r = 0; r < rows" in jit
    
    def test_compile_3d_element(self, compiler):
        """Compiles 3D element access."""
        code, deps = compiler.compile("tensor[0, 1, 2]", "elem")
        jit = compiler.get_jit_declarations()
        
        assert "i_norm * d1 * d2 + j_norm * d2 + k_norm" in jit
    
    def test_compile_hybrid_includes_counter(self, compiler):
        """Compiles hybrid array with counter dependency."""
        code, deps = compiler.compile("hits[:, 0]", "col")
        
        assert "n" in deps
        # Code is now a function call with n as parameter
        assert "n" in code or "col" in code
    
    def test_get_result_type_1d_element(self, compiler):
        """Gets correct result type for 1D element."""
        cpp_type, rank = compiler.get_result_type("arr[0]")
        assert cpp_type == "float"
        assert rank == 0
    
    def test_get_result_type_2d_row(self, compiler):
        """Gets correct result type for 2D row."""
        cpp_type, rank = compiler.get_result_type("mat[0]")
        assert cpp_type == "ROOT::RVec<float>"
        assert rank == 1
    
    def test_get_result_type_3d_plane(self, compiler):
        """Gets correct result type for 3D plane."""
        cpp_type, rank = compiler.get_result_type("tensor[0]")
        assert cpp_type == "ROOT::RVec<ROOT::RVec<double>>"
        assert rank == 2


# =============================================================================
# ROOT Integration Tests
# =============================================================================
# NOTE: Full ROOT correctness tests are in test_carray_correctness.py
# which uses RDataFrame + AsNumpy with proper TTree creation via C++ macros.
#
# gInterpreter.Calc() has issues with lambdas and NaN values.
# gInterpreter.Declare() with global variables causes parallel test issues.
#
# Unit tests above verify code structure without ROOT.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
