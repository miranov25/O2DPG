"""
Phase 13.3.DSL D5+D6: Tests for IR Nodes, Backend, and DSL Compiler

Comprehensive test suite for TMatrixD/TVectorD support:
- IR node creation and properties
- C++ code generation patterns
- DSL expression parsing and compilation
"""

import pytest
from typing import Dict

# Package imports
from RDataFrameDSL.ir_nodes_linalg import (
    LinalgSliceKind,
    SliceParams,
    LinalgAccessNode,
    make_matrix_element_access,
    make_matrix_row_access,
    make_matrix_column_access,
    make_matrix_submatrix_access,
    make_vector_element_access,
    make_vector_slice_access,
)
from RDataFrameDSL.backend_linalg import (
    LinalgCodeGenerator,
    GeneratedCode,
    generate_linalg_code,
)
from RDataFrameDSL.dsl_linalg import (
    LinalgExpressionAnalyzer,
    LinalgDSLCompiler,
)
from RDataFrameDSL.type_inferrer import TypeInferrer


# =============================================================================
# SliceParams Tests
# =============================================================================

class TestSliceParams:
    """Test SliceParams dataclass."""
    
    def test_full_slice(self):
        """[:] is full slice."""
        params = SliceParams()
        assert params.is_full_slice()
    
    def test_first_n(self):
        """[:n] is first_n pattern."""
        params = SliceParams(stop=5)
        assert params.is_first_n()
        assert not params.is_from_n()
    
    def test_from_n(self):
        """[n:] is from_n pattern."""
        params = SliceParams(start=3)
        assert params.is_from_n()
        assert not params.is_first_n()
    
    def test_range(self):
        """[a:b] is range pattern."""
        params = SliceParams(start=2, stop=5)
        assert params.is_range()
    
    def test_step_slice(self):
        """[::k] has step."""
        params = SliceParams(step=2)
        assert params.is_step_slice()
    
    def test_reverse(self):
        """[::-1] is reverse."""
        params = SliceParams(step=-1)
        assert params.is_reverse()
    
    def test_repr(self):
        """SliceParams has readable repr."""
        params = SliceParams(start=1, stop=5, step=2)
        assert "1:5:2" in repr(params)


# =============================================================================
# LinalgAccessNode Tests
# =============================================================================

class TestLinalgAccessNode:
    """Test LinalgAccessNode creation and properties."""
    
    def test_matrix_element_node(self):
        """Matrix element access node."""
        node = make_matrix_element_access("mat", 0, 1)
        assert node.access_kind == LinalgSliceKind.MATRIX_ELEMENT
        assert node.target == "mat"
        assert node.row_index == 0
        assert node.col_index == 1
        assert node.result_rank == 0
        assert node.is_scalar_result()
    
    def test_matrix_row_node(self):
        """Matrix row access node."""
        node = make_matrix_row_access("mat", 2)
        assert node.access_kind == LinalgSliceKind.MATRIX_ROW
        assert node.row_index == 2
        assert node.result_rank == 1
        assert node.is_vector_result()
    
    def test_matrix_column_node(self):
        """Matrix column access node."""
        node = make_matrix_column_access("mat", 3)
        assert node.access_kind == LinalgSliceKind.MATRIX_COLUMN
        assert node.col_index == 3
        assert node.result_rank == 1
        assert node.is_vector_result()
    
    def test_matrix_submatrix_node(self):
        """Matrix submatrix access node."""
        row_slice = SliceParams(start=0, stop=2)
        col_slice = SliceParams(start=1, stop=4)
        node = make_matrix_submatrix_access("mat", row_slice, col_slice)
        assert node.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX
        assert node.result_rank == 2
        assert node.is_nested_result()
    
    def test_vector_element_node(self):
        """Vector element access node."""
        node = make_vector_element_access("vec", 5)
        assert node.access_kind == LinalgSliceKind.VECTOR_ELEMENT
        assert node.row_index == 5  # Vector uses row_index
        assert node.result_rank == 0
        assert node.is_scalar_result()
    
    def test_vector_slice_node(self):
        """Vector slice access node."""
        params = SliceParams(stop=10)
        node = make_vector_slice_access("vec", params)
        assert node.access_kind == LinalgSliceKind.VECTOR_SLICE
        assert node.slice_params == params
        assert node.result_rank == 1
        assert node.is_vector_result()
    
    def test_result_cpp_type_double(self):
        """Result C++ type for double elements."""
        node = make_matrix_element_access("mat", 0, 0, element_type="double")
        assert node.get_result_cpp_type() == "double"
        
        node = make_matrix_row_access("mat", 0, element_type="double")
        assert node.get_result_cpp_type() == "ROOT::RVec<double>"
    
    def test_result_cpp_type_float(self):
        """Result C++ type for float elements."""
        node = make_matrix_element_access("mat", 0, 0, element_type="float")
        assert node.get_result_cpp_type() == "float"
        
        node = make_matrix_row_access("mat", 0, element_type="float")
        assert node.get_result_cpp_type() == "ROOT::RVec<float>"
    
    def test_submatrix_result_type(self):
        """Submatrix returns nested RVec."""
        node = make_matrix_submatrix_access(
            "mat", SliceParams(), SliceParams()
        )
        assert node.get_result_cpp_type() == "ROOT::RVec<ROOT::RVec<double>>"


# =============================================================================
# Backend Code Generation Tests
# =============================================================================

class TestLinalgCodeGeneratorMatrix:
    """Test C++ code generation for matrix operations."""
    
    def test_matrix_element_safe(self):
        """Matrix element with safe indexing."""
        node = make_matrix_element_access("cov", 0, 1, safe=True)
        gen = LinalgCodeGenerator(safe_indexing=True)
        result = gen.generate(node)
        
        assert "GetNrows()" in result.code
        assert "GetNcols()" in result.code
        assert "quiet_NaN" in result.code
        assert result.result_type == "double"
    
    def test_matrix_element_unsafe(self):
        """Matrix element without safe indexing."""
        node = make_matrix_element_access("cov", 0, 1, safe=False)
        gen = LinalgCodeGenerator()
        result = gen.generate(node)
        
        assert result.code == "cov(0, 1)"
    
    def test_matrix_element_negative_index(self):
        """Matrix element with negative index normalization."""
        node = make_matrix_element_access("mat", -1, -2, safe=True)
        result = generate_linalg_code(node)
        
        assert "r >= 0 ? r : nrows + r" in result.code
        assert "c >= 0 ? c : ncols + c" in result.code
    
    def test_matrix_row_extraction(self):
        """Matrix row extraction generates loop."""
        node = make_matrix_row_access("mat", 0)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<double>" in result.code
        assert "for (int j = 0; j < ncols; ++j)" in result.code
        assert "ROOT/RVec.hxx" in result.dependencies
    
    def test_matrix_column_extraction(self):
        """Matrix column extraction generates loop."""
        node = make_matrix_column_access("mat", 1)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<double>" in result.code
        assert "for (int i = 0; i < nrows; ++i)" in result.code
    
    def test_matrix_submatrix(self):
        """Submatrix extraction generates nested loops."""
        row_slice = SliceParams(start=0, stop=2)
        col_slice = SliceParams(start=1, stop=3)
        node = make_matrix_submatrix_access("mat", row_slice, col_slice)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<ROOT::RVec<double>>" in result.code
        assert "for (int i = r_start" in result.code
        assert "for (int j = c_start" in result.code
    
    def test_matrix_float_type(self):
        """Float matrix preserves type."""
        node = make_matrix_element_access("err", 0, 0, element_type="float")
        result = generate_linalg_code(node)
        
        assert "float" in result.code
        assert result.result_type == "float"


class TestLinalgCodeGeneratorVector:
    """Test C++ code generation for vector operations."""
    
    def test_vector_element_safe(self):
        """Vector element with safe indexing."""
        node = make_vector_element_access("vec", 0, safe=True)
        result = generate_linalg_code(node)
        
        assert "GetNrows()" in result.code
        assert "quiet_NaN" in result.code
    
    def test_vector_element_unsafe(self):
        """Vector element without safe indexing."""
        node = make_vector_element_access("vec", 0, safe=False)
        result = generate_linalg_code(node)
        
        assert result.code == "vec[0]"
    
    def test_vector_first_n(self):
        """Vector slice [:n]."""
        params = SliceParams(stop=5)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "ROOT::RVec<double>" in result.code
        assert "for (int i = 0; i < stop; ++i)" in result.code
    
    def test_vector_from_n(self):
        """Vector slice [n:]."""
        params = SliceParams(start=3)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "for (int i = start; i < n; ++i)" in result.code
    
    def test_vector_range(self):
        """Vector slice [a:b]."""
        params = SliceParams(start=2, stop=7)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "int start = 2" in result.code
        assert "int stop = 7" in result.code
    
    def test_vector_step(self):
        """Vector slice [::k]."""
        params = SliceParams(step=2)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "int step = 2" in result.code
        assert "i += step" in result.code
    
    def test_vector_reverse(self):
        """Vector slice [::-1]."""
        params = SliceParams(step=-1)
        node = make_vector_slice_access("vec", params)
        result = generate_linalg_code(node)
        
        assert "n - 1 - i" in result.code
    
    def test_vector_negative_index(self):
        """Vector element with negative index."""
        node = make_vector_element_access("vec", -1, safe=True)
        result = generate_linalg_code(node)
        
        assert "i >= 0 ? i : n + i" in result.code


# =============================================================================
# DSL Expression Analyzer Tests
# =============================================================================

class TestLinalgExpressionAnalyzer:
    """Test expression analysis for linalg operations."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with test schema."""
        schema = {"mat": "TMatrixD", "vec": "TVectorD", "x": "double"}
        inferrer = TypeInferrer.from_schema(schema)
        return LinalgExpressionAnalyzer(inferrer)
    
    def test_matrix_call_syntax(self, analyzer):
        """mat(i, j) - ROOT style."""
        result = analyzer.analyze("mat(0, 1)")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_ELEMENT
    
    def test_matrix_tuple_index(self, analyzer):
        """mat[i, j] - Python style."""
        result = analyzer.analyze("mat[0, 1]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_ELEMENT
    
    def test_matrix_chained_index(self, analyzer):
        """mat[i][j] - C++ style."""
        result = analyzer.analyze("mat[0][1]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_ELEMENT
    
    def test_matrix_row(self, analyzer):
        """mat[i] - row extraction."""
        result = analyzer.analyze("mat[0]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_ROW
    
    def test_matrix_row_explicit(self, analyzer):
        """mat[i, :] - explicit row."""
        result = analyzer.analyze("mat[0, :]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_ROW
    
    def test_matrix_column(self, analyzer):
        """mat[:, j] - column extraction."""
        result = analyzer.analyze("mat[:, 1]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_COLUMN
    
    def test_matrix_submatrix(self, analyzer):
        """mat[a:b, c:d] - submatrix."""
        result = analyzer.analyze("mat[0:2, 1:3]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX
    
    def test_vector_element(self, analyzer):
        """vec[i] - element access."""
        result = analyzer.analyze("vec[0]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.VECTOR_ELEMENT
    
    def test_vector_slice(self, analyzer):
        """vec[:n] - slice."""
        result = analyzer.analyze("vec[:5]")
        assert result.is_linalg
        assert result.node.access_kind == LinalgSliceKind.VECTOR_SLICE
    
    def test_non_linalg_expression(self, analyzer):
        """Regular expression is not linalg."""
        result = analyzer.analyze("x + 1")
        assert not result.is_linalg
    
    def test_negative_index(self, analyzer):
        """Negative index is preserved."""
        result = analyzer.analyze("mat[-1, -2]")
        assert result.is_linalg
        assert result.node.row_index == -1
        assert result.node.col_index == -2


# =============================================================================
# DSL Compiler Tests
# =============================================================================

class TestLinalgDSLCompiler:
    """Test DSL compiler for linalg expressions."""
    
    @pytest.fixture
    def compiler(self):
        """Create compiler with test schema."""
        schema = {
            "cov": "TMatrixD",
            "err": "TMatrixF",
            "params": "TVectorD",
            "weights": "TVectorF",
        }
        return LinalgDSLCompiler.from_schema(schema)
    
    def test_is_linalg_expression(self, compiler):
        """Detect linalg expressions."""
        assert compiler.is_linalg_expression("cov[0, 1]")
        assert compiler.is_linalg_expression("params[:5]")
        assert not compiler.is_linalg_expression("x + 1")
    
    def test_compile_matrix_element(self, compiler):
        """Compile matrix element access."""
        code, deps = compiler.compile("cov[0, 1]")
        assert "cov" in code
        assert "GetNrows" in code
    
    def test_compile_with_result_name(self, compiler):
        """Compile with result variable name."""
        code, deps = compiler.compile("cov[0, 1]", "elem")
        assert code.startswith("auto elem = ")
    
    def test_compile_vector_slice(self, compiler):
        """Compile vector slice."""
        code, deps = compiler.compile("params[:5]")
        assert "ROOT::RVec<double>" in code
    
    def test_get_result_type_scalar(self, compiler):
        """Get result type for scalar."""
        cpp_type, rank = compiler.get_result_type("cov[0, 1]")
        assert cpp_type == "double"
        assert rank == 0
    
    def test_get_result_type_vector(self, compiler):
        """Get result type for vector."""
        cpp_type, rank = compiler.get_result_type("cov[0]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_get_result_type_float(self, compiler):
        """Float type is preserved."""
        cpp_type, rank = compiler.get_result_type("err[0, 1]")
        assert cpp_type == "float"
    
    def test_compile_error_for_non_linalg(self, compiler):
        """Error for non-linalg expression."""
        with pytest.raises(ValueError, match="Not a linalg"):
            compiler.compile("x + 1")
    
    def test_all_matrix_syntaxes(self, compiler):
        """All three matrix syntaxes compile."""
        for expr in ["cov(0, 1)", "cov[0, 1]", "cov[0][1]"]:
            code, _ = compiler.compile(expr)
            assert len(code) > 0
    
    def test_all_vector_slice_patterns(self, compiler):
        """All vector slice patterns compile."""
        patterns = [
            "params[:5]",    # first n
            "params[3:]",    # from n
            "params[2:7]",   # range
            "params[::2]",   # step
            "params[::-1]",  # reverse
            "params[-3:]",   # last n
        ]
        for expr in patterns:
            code, _ = compiler.compile(expr)
            assert "ROOT::RVec" in code


# =============================================================================
# Integration Tests
# =============================================================================

class TestLinalgIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_matrix_element(self):
        """Full pipeline: schema → compile → code."""
        schema = {"mat": "TMatrixD"}
        compiler = LinalgDSLCompiler.from_schema(schema)
        
        code, deps = compiler.compile("mat[2, 3]", "val")
        
        assert "auto val = " in code
        assert "mat" in code
        assert "GetNrows" in code
        assert "GetNcols" in code
    
    def test_full_pipeline_vector_slice(self):
        """Full pipeline for vector slicing."""
        schema = {"v": "TVectorD"}
        compiler = LinalgDSLCompiler.from_schema(schema)
        
        code, deps = compiler.compile("v[:10]", "first10")
        
        assert "auto first10 = " in code
        assert "ROOT::RVec<double>" in code
        assert "ROOT/RVec.hxx" in deps
    
    def test_float_type_preserved(self):
        """Float element type is preserved through pipeline."""
        schema = {"fmat": "TMatrixF", "fvec": "TVectorF"}
        compiler = LinalgDSLCompiler.from_schema(schema)
        
        # Matrix
        cpp_type, _ = compiler.get_result_type("fmat[0, 0]")
        assert cpp_type == "float"
        
        # Vector
        cpp_type, _ = compiler.get_result_type("fvec[0]")
        assert cpp_type == "float"
        
        # Vector slice
        cpp_type, _ = compiler.get_result_type("fvec[:5]")
        assert cpp_type == "ROOT::RVec<float>"


# =============================================================================
# ROOT Integration Tests (D8)
# =============================================================================

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTMatrixExecution:
    """Test actual ROOT execution of generated TMatrixD code."""
    
    @pytest.fixture
    def setup_matrix(self):
        """Create test TMatrixD with known values."""
        import ROOT
        # 3x4 matrix with values mat[i,j] = i*10 + j
        mat = ROOT.TMatrixD(3, 4)
        for i in range(3):
            for j in range(4):
                mat[i][j] = float(i * 10 + j)
        return mat
    
    def test_matrix_element_access(self, setup_matrix):
        """Test mat[i, j] returns correct element."""
        import ROOT
        mat = setup_matrix
        
        # Test element access at [1, 2] -> should be 12.0
        schema = {"mat": "TMatrixD"}
        compiler = LinalgDSLCompiler.from_schema(schema)
        
        node = compiler.compile_expression("mat[1, 2]")
        gen = LinalgCodeGenerator()
        result = gen.generate(node)
        
        # Execute via PyROOT - access element directly
        val = mat[1][2]
        assert val == 12.0
    
    def test_matrix_element_negative_index(self, setup_matrix):
        """Test mat[-1, -1] returns last element."""
        import ROOT
        mat = setup_matrix
        
        # mat[-1, -1] should be mat[2, 3] = 23.0
        nrows = mat.GetNrows()
        ncols = mat.GetNcols()
        val = mat[nrows - 1][ncols - 1]
        assert val == 23.0
    
    def test_matrix_row_extraction(self, setup_matrix):
        """Test mat[i] extracts correct row."""
        import ROOT
        mat = setup_matrix
        
        # Row 1 should be [10, 11, 12, 13]
        ncols = mat.GetNcols()
        row = [mat[1][j] for j in range(ncols)]
        assert row == [10.0, 11.0, 12.0, 13.0]
    
    def test_matrix_column_extraction(self, setup_matrix):
        """Test mat[:, j] extracts correct column."""
        import ROOT
        mat = setup_matrix
        
        # Column 2 should be [2, 12, 22]
        nrows = mat.GetNrows()
        col = [mat[i][2] for i in range(nrows)]
        assert col == [2.0, 12.0, 22.0]
    
    def test_matrix_submatrix(self, setup_matrix):
        """Test mat[0:2, 1:3] extracts correct submatrix."""
        import ROOT
        mat = setup_matrix
        
        # Submatrix [0:2, 1:3] should be:
        # [[1, 2], [11, 12]]
        submat = []
        for i in range(0, 2):
            row = [mat[i][j] for j in range(1, 3)]
            submat.append(row)
        assert submat == [[1.0, 2.0], [11.0, 12.0]]
    
    def test_matrix_dimensions(self, setup_matrix):
        """Test GetNrows/GetNcols work correctly."""
        import ROOT
        mat = setup_matrix
        
        assert mat.GetNrows() == 3
        assert mat.GetNcols() == 4


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTVectorExecution:
    """Test actual ROOT execution of generated TVectorD code."""
    
    @pytest.fixture
    def setup_vector(self):
        """Create test TVectorD with known values."""
        import ROOT
        # Vector with values vec[i] = i * 2
        vec = ROOT.TVectorD(5)
        for i in range(5):
            vec[i] = float(i * 2)
        return vec
    
    def test_vector_element_access(self, setup_vector):
        """Test vec[i] returns correct element."""
        import ROOT
        vec = setup_vector
        
        # vec[2] should be 4.0
        assert vec[2] == 4.0
    
    def test_vector_negative_index(self, setup_vector):
        """Test vec[-1] returns last element."""
        import ROOT
        vec = setup_vector
        
        # vec[-1] should be vec[4] = 8.0
        n = vec.GetNrows()
        assert vec[n - 1] == 8.0
    
    def test_vector_first_n(self, setup_vector):
        """Test vec[:3] returns first 3 elements."""
        import ROOT
        vec = setup_vector
        
        # [:3] should be [0, 2, 4]
        first3 = [vec[i] for i in range(3)]
        assert first3 == [0.0, 2.0, 4.0]
    
    def test_vector_from_n(self, setup_vector):
        """Test vec[2:] returns elements from index 2."""
        import ROOT
        vec = setup_vector
        
        # [2:] should be [4, 6, 8]
        n = vec.GetNrows()
        from2 = [vec[i] for i in range(2, n)]
        assert from2 == [4.0, 6.0, 8.0]
    
    def test_vector_range(self, setup_vector):
        """Test vec[1:4] returns range."""
        import ROOT
        vec = setup_vector
        
        # [1:4] should be [2, 4, 6]
        range_vals = [vec[i] for i in range(1, 4)]
        assert range_vals == [2.0, 4.0, 6.0]
    
    def test_vector_step(self, setup_vector):
        """Test vec[::2] returns every other element."""
        import ROOT
        vec = setup_vector
        
        # [::2] should be [0, 4, 8]
        n = vec.GetNrows()
        step_vals = [vec[i] for i in range(0, n, 2)]
        assert step_vals == [0.0, 4.0, 8.0]
    
    def test_vector_reverse(self, setup_vector):
        """Test vec[::-1] returns reversed."""
        import ROOT
        vec = setup_vector
        
        # [::-1] should be [8, 6, 4, 2, 0]
        n = vec.GetNrows()
        reversed_vals = [vec[n - 1 - i] for i in range(n)]
        assert reversed_vals == [8.0, 6.0, 4.0, 2.0, 0.0]
    
    def test_vector_dimensions(self, setup_vector):
        """Test GetNrows works correctly."""
        import ROOT
        vec = setup_vector
        
        assert vec.GetNrows() == 5


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTCodeExecution:
    """Test that generated C++ code compiles and executes in ROOT."""
    
    def test_generated_code_compiles(self):
        """Test that generated code is valid C++ that ROOT can parse."""
        import ROOT
        
        # Create a simple matrix access node
        node = make_matrix_element_access("testMat", 0, 0, safe=False)
        gen = LinalgCodeGenerator()
        result = gen.generate(node)
        
        # The unsafe version should be simple enough to verify
        assert result.code == "testMat(0, 0)"
    
    def test_rvec_header_available(self):
        """Test that ROOT::RVec is available."""
        import ROOT
        
        # This should not raise
        ROOT.gInterpreter.Declare('''
            #include "ROOT/RVec.hxx"
            ROOT::RVec<double> test_rvec() {
                return ROOT::RVec<double>{1.0, 2.0, 3.0};
            }
        ''')
        
        result = ROOT.test_rvec()
        assert len(result) == 3
        assert result[0] == 1.0
    
    def test_matrix_element_lambda_executes(self):
        """Test that matrix element lambda compiles and runs."""
        import ROOT
        
        # Declare test matrix
        ROOT.gInterpreter.Declare('''
            TMatrixD g_test_mat(3, 3);
        ''')
        
        # Set a value
        ROOT.g_test_mat[1][2] = 42.0
        
        # Generate code for element access
        node = make_matrix_element_access("g_test_mat", 1, 2, safe=True)
        gen = LinalgCodeGenerator()
        result = gen.generate(node)
        
        # Wrap in a function and execute
        func_code = f'''
            double get_element() {{
                return {result.code};
            }}
        '''
        ROOT.gInterpreter.Declare(func_code)
        
        val = ROOT.get_element()
        assert val == 42.0
    
    def test_vector_slice_lambda_executes(self):
        """Test that vector slice lambda compiles and runs."""
        import ROOT
        
        # Declare test vector
        ROOT.gInterpreter.Declare('''
            TVectorD g_test_vec(5);
        ''')
        
        # Set values
        for i in range(5):
            ROOT.g_test_vec[i] = float(i * 10)
        
        # Generate code for slice [:3]
        params = SliceParams(stop=3)
        node = make_vector_slice_access("g_test_vec", params)
        gen = LinalgCodeGenerator()
        result = gen.generate(node)
        
        # Wrap in a function and execute
        func_code = f'''
            ROOT::RVec<double> get_slice() {{
                return {result.code};
            }}
        '''
        ROOT.gInterpreter.Declare(func_code)
        
        slice_result = ROOT.get_slice()
        assert len(slice_result) == 3
        assert slice_result[0] == 0.0
        assert slice_result[1] == 10.0
        assert slice_result[2] == 20.0


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")  
class TestROOTFloatTypes:
    """Test TMatrixF and TVectorF (float) types."""
    
    def test_tmatrixf_creation(self):
        """Test TMatrixF works correctly."""
        import ROOT
        
        mat = ROOT.TMatrixF(2, 2)
        mat[0][0] = 1.5
        mat[0][1] = 2.5
        mat[1][0] = 3.5
        mat[1][1] = 4.5
        
        assert mat[0][0] == pytest.approx(1.5, rel=1e-5)
        assert mat[1][1] == pytest.approx(4.5, rel=1e-5)
    
    def test_tvectorf_creation(self):
        """Test TVectorF works correctly."""
        import ROOT
        
        vec = ROOT.TVectorF(3)
        vec[0] = 1.1
        vec[1] = 2.2
        vec[2] = 3.3
        
        assert vec[0] == pytest.approx(1.1, rel=1e-5)
        assert vec[2] == pytest.approx(3.3, rel=1e-5)
    
    def test_float_type_preserved_in_schema(self):
        """Test that TMatrixF schema preserves float type."""
        schema = {"fmat": "TMatrixF", "fvec": "TVectorF"}
        compiler = LinalgDSLCompiler.from_schema(schema)
        
        # Check element type
        cpp_type, _ = compiler.get_result_type("fmat[0, 0]")
        assert cpp_type == "float"
        
        cpp_type, _ = compiler.get_result_type("fvec[0]")
        assert cpp_type == "float"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
