"""
Phase 13.3.DSL D7: Tests for Nested RVec 2D Slicing

Comprehensive test suite for RVec<RVec<T>> operations:
- IR node creation and properties
- C++ code generation patterns
- DSL expression parsing
- Two-tier ragged policy verification
- ROOT integration tests
"""

import pytest
pytestmark = pytest.mark.root_serial
from typing import Dict, List

# Package imports
from RDataFrameDSL.ir_nodes_nested import (
    NestedSliceKind,
    SliceParams,
    NestedAccessNode,
    make_nested_column_extract,
    make_nested_column_slice,
    make_nested_element_access,
    make_nested_row_access,
)
from RDataFrameDSL.backend_nested import (
    NestedCodeGenerator,
    GeneratedCode,
    generate_nested_code,
)
from RDataFrameDSL.dsl_nested import (
    NestedExpressionAnalyzer,
    NestedDSLCompiler,
    is_nested_rvec_type,
    get_nested_element_type,
)
from RDataFrameDSL.type_inferrer import TypeInferrer


# =============================================================================
# Type Detection Tests
# =============================================================================

class TestNestedRVecTypeDetection:
    """Test detection of nested RVec types."""
    
    def test_is_nested_rvec_double(self):
        """RVec<RVec<double>> is nested."""
        assert is_nested_rvec_type("RVec<RVec<double>>")
    
    def test_is_nested_rvec_float(self):
        """RVec<RVec<float>> is nested."""
        assert is_nested_rvec_type("RVec<RVec<float>>")
    
    def test_is_nested_root_rvec(self):
        """ROOT::RVec<ROOT::RVec<double>> is nested."""
        assert is_nested_rvec_type("ROOT::RVec<ROOT::RVec<double>>")
    
    def test_simple_rvec_not_nested(self):
        """RVec<double> is NOT nested."""
        assert not is_nested_rvec_type("RVec<double>")
    
    def test_scalar_not_nested(self):
        """double is NOT nested."""
        assert not is_nested_rvec_type("double")
    
    def test_vector_not_nested(self):
        """std::vector<double> is NOT nested (rank 1)."""
        assert not is_nested_rvec_type("std::vector<double>")
    
    def test_nested_vector(self):
        """vector<vector<double>> IS nested."""
        assert is_nested_rvec_type("vector<vector<double>>")


class TestNestedElementType:
    """Test element type extraction from nested RVec."""
    
    def test_nested_double(self):
        """Extract double from RVec<RVec<double>>."""
        assert get_nested_element_type("RVec<RVec<double>>") == "double"
    
    def test_nested_float(self):
        """Extract float from RVec<RVec<float>>."""
        assert get_nested_element_type("RVec<RVec<float>>") == "float"
    
    def test_nested_int(self):
        """Extract int from RVec<RVec<int>>."""
        assert get_nested_element_type("RVec<RVec<int>>") == "int"


# =============================================================================
# IR Node Tests
# =============================================================================

class TestNestedAccessNode:
    """Test NestedAccessNode creation and properties."""
    
    def test_column_extract_node(self):
        """Column extract node has correct properties."""
        node = make_nested_column_extract("tracks", 0)
        assert node.access_kind == NestedSliceKind.NESTED_COLUMN_EXTRACT
        assert node.target == "tracks"
        assert node.col_index == 0
        assert node.result_rank == 1
        assert node.fail_closed == True  # FAIL-CLOSED for integer
    
    def test_column_extract_negative_index(self):
        """Column extract with negative index."""
        node = make_nested_column_extract("tracks", -1)
        assert node.col_index == -1
        assert node.fail_closed == True
    
    def test_column_slice_node(self):
        """Column slice node has correct properties."""
        params = SliceParams(stop=3)
        node = make_nested_column_slice("tracks", params)
        assert node.access_kind == NestedSliceKind.NESTED_COLUMN_SLICE
        assert node.result_rank == 2
        assert node.fail_closed == False  # CLAMP for slice
    
    def test_element_access_node(self):
        """Element access node has correct properties."""
        node = make_nested_element_access("tracks", 1, 2)
        assert node.access_kind == NestedSliceKind.NESTED_ELEMENT
        assert node.row_index == 1
        assert node.col_index == 2
        assert node.result_rank == 0
    
    def test_row_access_node(self):
        """Row access node has correct properties."""
        node = make_nested_row_access("tracks", 0)
        assert node.access_kind == NestedSliceKind.NESTED_ROW_ACCESS
        assert node.row_index == 0
        assert node.result_rank == 1
    
    def test_result_cpp_type_column_extract(self):
        """Column extract returns RVec<T>."""
        node = make_nested_column_extract("tracks", 0, element_type="double")
        assert node.get_result_cpp_type() == "ROOT::RVec<double>"
    
    def test_result_cpp_type_column_slice(self):
        """Column slice returns RVec<RVec<T>>."""
        node = make_nested_column_slice("tracks", SliceParams())
        assert node.get_result_cpp_type() == "ROOT::RVec<ROOT::RVec<double>>"
    
    def test_result_cpp_type_element(self):
        """Element access returns scalar."""
        node = make_nested_element_access("tracks", 0, 0, element_type="float")
        assert node.get_result_cpp_type() == "float"


# =============================================================================
# Backend Code Generation Tests
# =============================================================================

class TestNestedCodeGeneratorColumnExtract:
    """Test code generation for column extraction (fail-closed)."""
    
    def test_column_extract_basic(self):
        """Basic column extract generates correct code."""
        node = make_nested_column_extract("tracks", 0)
        result = generate_nested_code(node)
        
        assert "throw std::runtime_error" in result.code
        assert "Column index" in result.code
        assert "out of bounds" in result.code
        assert result.result_type == "ROOT::RVec<double>"
    
    def test_column_extract_negative_index(self):
        """Negative index generates normalization code."""
        node = make_nested_column_extract("tracks", -1)
        result = generate_nested_code(node)
        
        assert "j >= 0 ? j : row_size + j" in result.code
    
    def test_column_extract_has_loop(self):
        """Column extract loops over all rows."""
        node = make_nested_column_extract("tracks", 0)
        result = generate_nested_code(node)
        
        assert "for (size_t i = 0; i < outer.size(); ++i)" in result.code
        assert "result.push_back(row[j_norm])" in result.code
    
    def test_column_extract_float_type(self):
        """Float element type preserved."""
        node = make_nested_column_extract("tracks", 0, element_type="float")
        result = generate_nested_code(node)
        
        assert "ROOT::RVec<float>" in result.code
        assert result.result_type == "ROOT::RVec<float>"


class TestNestedCodeGeneratorColumnSlice:
    """Test code generation for column slicing (clamp per-row)."""
    
    def test_column_slice_first_n(self):
        """[:, :3] generates clamp code."""
        params = SliceParams(stop=3)
        node = make_nested_column_slice("tracks", params)
        result = generate_nested_code(node)
        
        # Should NOT have throw (clamp policy)
        assert "throw" not in result.code
        # Should have clamp logic
        assert "std::max" in result.code
        assert "std::min" in result.code
        assert result.result_type == "ROOT::RVec<ROOT::RVec<double>>"
    
    def test_column_slice_range(self):
        """[:, 1:4] generates range slice."""
        params = SliceParams(start=1, stop=4)
        node = make_nested_column_slice("tracks", params)
        result = generate_nested_code(node)
        
        assert "int start = 1" in result.code
        assert "int stop = 4" in result.code
    
    def test_column_slice_step(self):
        """[:, ::2] generates step slice."""
        params = SliceParams(step=2)
        node = make_nested_column_slice("tracks", params)
        result = generate_nested_code(node)
        
        assert "int step = 2" in result.code
        assert "k += step" in result.code
    
    def test_column_slice_reverse(self):
        """[:, ::-1] generates reverse code."""
        params = SliceParams(step=-1)
        node = make_nested_column_slice("tracks", params)
        result = generate_nested_code(node)
        
        assert "row_size - 1" in result.code


class TestNestedCodeGeneratorElement:
    """Test code generation for element access."""
    
    def test_element_access_basic(self):
        """Element access generates bounds check."""
        node = make_nested_element_access("tracks", 0, 1)
        result = generate_nested_code(node)
        
        assert "i_norm" in result.code
        assert "j_norm" in result.code
        assert "quiet_NaN" in result.code  # NaN fallback for out-of-bounds
    
    def test_element_access_negative_indices(self):
        """Negative indices normalized."""
        node = make_nested_element_access("tracks", -1, -2)
        result = generate_nested_code(node)
        
        assert "i >= 0 ? i : outer_size + i" in result.code
        assert "j >= 0 ? j : inner_size + j" in result.code


class TestNestedCodeGeneratorRow:
    """Test code generation for row access."""
    
    def test_row_access_basic(self):
        """Row access returns inner RVec."""
        node = make_nested_row_access("tracks", 0)
        result = generate_nested_code(node)
        
        assert "return outer[i_norm]" in result.code
        assert result.result_type == "ROOT::RVec<double>"


# =============================================================================
# DSL Expression Analyzer Tests
# =============================================================================

class TestNestedExpressionAnalyzer:
    """Test expression analysis for nested RVec operations."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with test schema."""
        schema = {
            "tracks": "RVec<RVec<double>>",
            "hits": "RVec<RVec<float>>",
            "simple": "RVec<double>",
            "x": "double",
        }
        inferrer = TypeInferrer.from_schema(schema)
        return NestedExpressionAnalyzer(inferrer)
    
    def test_column_extract_detected(self, analyzer):
        """tracks[:, 0] detected as column extract."""
        result = analyzer.analyze("tracks[:, 0]")
        assert result.is_nested
        assert result.node.access_kind == NestedSliceKind.NESTED_COLUMN_EXTRACT
    
    def test_column_extract_negative(self, analyzer):
        """tracks[:, -1] detected with negative index."""
        result = analyzer.analyze("tracks[:, -1]")
        assert result.is_nested
        assert result.node.col_index == -1
    
    def test_column_slice_detected(self, analyzer):
        """tracks[:, :3] detected as column slice."""
        result = analyzer.analyze("tracks[:, :3]")
        assert result.is_nested
        assert result.node.access_kind == NestedSliceKind.NESTED_COLUMN_SLICE
    
    def test_element_access_detected(self, analyzer):
        """tracks[0, 1] detected as element access."""
        result = analyzer.analyze("tracks[0, 1]")
        assert result.is_nested
        assert result.node.access_kind == NestedSliceKind.NESTED_ELEMENT
    
    def test_row_access_detected(self, analyzer):
        """tracks[0] detected as row access."""
        result = analyzer.analyze("tracks[0]")
        assert result.is_nested
        assert result.node.access_kind == NestedSliceKind.NESTED_ROW_ACCESS
    
    def test_simple_rvec_not_nested(self, analyzer):
        """simple[:, 0] NOT detected (not nested RVec)."""
        result = analyzer.analyze("simple[:, 0]")
        assert not result.is_nested
    
    def test_scalar_not_nested(self, analyzer):
        """x + 1 NOT detected."""
        result = analyzer.analyze("x + 1")
        assert not result.is_nested
    
    def test_float_type_preserved(self, analyzer):
        """hits[:, 0] preserves float type."""
        result = analyzer.analyze("hits[:, 0]")
        assert result.is_nested
        assert result.node.element_type == "float"


# =============================================================================
# DSL Compiler Tests
# =============================================================================

class TestNestedDSLCompiler:
    """Test DSL compiler for nested expressions."""
    
    @pytest.fixture
    def compiler(self):
        """Create compiler with test schema."""
        schema = {
            "tracks_pt": "RVec<RVec<double>>",
            "tracks_eta": "RVec<RVec<float>>",
        }
        return NestedDSLCompiler.from_schema(schema)
    
    def test_is_nested_expression(self, compiler):
        """Detect nested expressions."""
        assert compiler.is_nested_expression("tracks_pt[:, 0]")
        assert compiler.is_nested_expression("tracks_pt[0, 1]")
        assert not compiler.is_nested_expression("x + 1")
    
    def test_compile_column_extract(self, compiler):
        """Compile column extract."""
        code, deps = compiler.compile("tracks_pt[:, 0]")
        assert "throw std::runtime_error" in code
        assert "ROOT/RVec.hxx" in deps
    
    def test_compile_with_result_name(self, compiler):
        """Compile with result variable name."""
        code, deps = compiler.compile("tracks_pt[:, 0]", "first_pt")
        assert code.startswith("auto first_pt = ")
    
    def test_compile_column_slice(self, compiler):
        """Compile column slice."""
        code, deps = compiler.compile("tracks_pt[:, :3]")
        assert "ROOT::RVec<ROOT::RVec<double>>" in code
        assert "throw" not in code  # Clamp policy
    
    def test_get_result_type_column_extract(self, compiler):
        """Get result type for column extract."""
        cpp_type, rank = compiler.get_result_type("tracks_pt[:, 0]")
        assert cpp_type == "ROOT::RVec<double>"
        assert rank == 1
    
    def test_get_result_type_column_slice(self, compiler):
        """Get result type for column slice."""
        cpp_type, rank = compiler.get_result_type("tracks_pt[:, :3]")
        assert cpp_type == "ROOT::RVec<ROOT::RVec<double>>"
        assert rank == 2
    
    def test_get_policy_column_extract(self, compiler):
        """Column extract uses fail-closed policy."""
        policy = compiler.get_policy("tracks_pt[:, 0]")
        assert policy == "fail-closed"
    
    def test_get_policy_column_slice(self, compiler):
        """Column slice uses clamp-per-row policy."""
        policy = compiler.get_policy("tracks_pt[:, :3]")
        assert policy == "clamp-per-row"
    
    def test_compile_error_for_non_nested(self, compiler):
        """Error for non-nested expression."""
        with pytest.raises(ValueError, match="Not a nested"):
            compiler.compile("x + 1")
    
    def test_all_column_slice_patterns(self, compiler):
        """All column slice patterns compile."""
        patterns = [
            "tracks_pt[:, :3]",   # first n
            "tracks_pt[:, 1:]",   # from n
            "tracks_pt[:, 1:4]",  # range
            "tracks_pt[:, ::2]",  # step
            "tracks_pt[:, ::-1]", # reverse
        ]
        for expr in patterns:
            code, _ = compiler.compile(expr)
            assert "ROOT::RVec" in code


# =============================================================================
# Two-Tier Ragged Policy Tests
# =============================================================================

class TestTwoTierRaggedPolicy:
    """Test the two-tier ragged policy implementation."""
    
    def test_integer_index_fail_closed(self):
        """Integer column index uses fail-closed policy."""
        node = make_nested_column_extract("tracks", 0)
        assert node.fail_closed == True
        
        result = generate_nested_code(node)
        assert "throw std::runtime_error" in result.code
    
    def test_negative_integer_fail_closed(self):
        """Negative integer index also fail-closed."""
        node = make_nested_column_extract("tracks", -1)
        assert node.fail_closed == True
        
        result = generate_nested_code(node)
        assert "throw std::runtime_error" in result.code
    
    def test_slice_clamp_per_row(self):
        """Slice uses clamp-per-row policy."""
        node = make_nested_column_slice("tracks", SliceParams(stop=3))
        assert node.fail_closed == False
        
        result = generate_nested_code(node)
        assert "throw" not in result.code
        assert "std::max" in result.code
        assert "std::min" in result.code
    
    def test_range_clamp_per_row(self):
        """Range slice uses clamp-per-row."""
        node = make_nested_column_slice("tracks", SliceParams(start=1, stop=4))
        assert node.fail_closed == False
    
    def test_step_slice_clamp_per_row(self):
        """Step slice uses clamp-per-row."""
        node = make_nested_column_slice("tracks", SliceParams(step=2))
        assert node.fail_closed == False


# =============================================================================
# Integration Tests
# =============================================================================

class TestNestedIntegration:
    """Integration tests for the full nested pipeline."""
    
    def test_full_pipeline_column_extract(self):
        """Full pipeline: schema → compile → code."""
        schema = {"nested": "RVec<RVec<double>>"}
        compiler = NestedDSLCompiler.from_schema(schema)
        
        code, deps = compiler.compile("nested[:, 0]", "col0")
        
        assert "auto col0 = " in code
        assert "throw std::runtime_error" in code
        assert "ROOT/RVec.hxx" in deps
    
    def test_full_pipeline_column_slice(self):
        """Full pipeline for column slicing."""
        schema = {"nested": "RVec<RVec<double>>"}
        compiler = NestedDSLCompiler.from_schema(schema)
        
        code, deps = compiler.compile("nested[:, :3]", "first3")
        
        assert "auto first3 = " in code
        assert "ROOT::RVec<ROOT::RVec<double>>" in code
        assert "throw" not in code
    
    def test_float_type_preserved(self):
        """Float element type preserved through pipeline."""
        schema = {"nested": "RVec<RVec<float>>"}
        compiler = NestedDSLCompiler.from_schema(schema)
        
        cpp_type, _ = compiler.get_result_type("nested[:, 0]")
        assert cpp_type == "ROOT::RVec<float>"
        
        code, _ = compiler.compile("nested[:, 0]")
        assert "ROOT::RVec<float>" in code


# =============================================================================
# ROOT Integration Tests
# =============================================================================

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTNestedExecution:
    """Test actual ROOT execution of generated nested RVec code."""
    
    @pytest.fixture
    def setup_nested_rvec(self):
        """Create test RVec<RVec<double>> with known values."""
        import ROOT
        # Create nested RVec:
        # [[0, 1, 2], [10, 11], [20, 21, 22, 23]]
        outer = ROOT.RVec['ROOT::RVec<double>']()
        
        row0 = ROOT.RVec['double']()
        for v in [0.0, 1.0, 2.0]:
            row0.push_back(v)
        
        row1 = ROOT.RVec['double']()
        for v in [10.0, 11.0]:
            row1.push_back(v)
        
        row2 = ROOT.RVec['double']()
        for v in [20.0, 21.0, 22.0, 23.0]:
            row2.push_back(v)
        
        outer.push_back(row0)
        outer.push_back(row1)
        outer.push_back(row2)
        
        return outer
    
    def test_nested_rvec_structure(self, setup_nested_rvec):
        """Verify test data structure."""
        outer = setup_nested_rvec
        assert len(outer) == 3
        assert len(outer[0]) == 3
        assert len(outer[1]) == 2
        assert len(outer[2]) == 4
    
    def test_column_extract_first_element(self, setup_nested_rvec):
        """Manual column extract: [:, 0] should return [0, 10, 20]."""
        outer = setup_nested_rvec
        
        # Manual extraction
        result = [outer[i][0] for i in range(len(outer))]
        assert result == [0.0, 10.0, 20.0]
    
    def test_column_extract_last_element(self, setup_nested_rvec):
        """Manual column extract: [:, -1] should return [2, 11, 23]."""
        outer = setup_nested_rvec
        
        # Manual extraction with negative index
        result = [outer[i][len(outer[i]) - 1] for i in range(len(outer))]
        assert result == [2.0, 11.0, 23.0]
    
    def test_column_extract_fails_on_missing(self, setup_nested_rvec):
        """
        Column extract [:, 3] - row 1 only has 2 elements.
        
        Note: Raw ROOT RVec access does NOT raise Python IndexError for
        out-of-bounds access (it's undefined behavior in C++).
        Our generated code with fail-closed policy handles this with
        std::runtime_error instead. This test documents ROOT's behavior.
        """
        outer = setup_nested_rvec
        
        # ROOT RVec does NOT raise IndexError for out-of-bounds
        # It returns garbage or crashes (undefined behavior)
        # So we just verify the row lengths to confirm our test data is correct
        assert len(outer[0]) == 3  # row 0 has 3 elements
        assert len(outer[1]) == 2  # row 1 only has 2 elements (would fail at index 3)
        assert len(outer[2]) == 4  # row 2 has 4 elements
        
        # Verify that index 3 would be out of bounds for row 1
        # (This is what our generated code would catch with fail-closed policy)
    
    def test_column_slice_first_two(self, setup_nested_rvec):
        """Manual column slice: [:, :2] should return [[0,1], [10,11], [20,21]]."""
        outer = setup_nested_rvec
        
        # Manual clamp slice
        result = []
        for i in range(len(outer)):
            row = outer[i]
            stop = min(2, len(row))
            result.append([row[j] for j in range(stop)])
        
        assert result == [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]
    
    def test_element_access(self, setup_nested_rvec):
        """Element access [1, 0] should return 10."""
        outer = setup_nested_rvec
        assert outer[1][0] == 10.0
    
    def test_row_access(self, setup_nested_rvec):
        """Row access [1] should return [10, 11]."""
        outer = setup_nested_rvec
        row = outer[1]
        assert list(row) == [10.0, 11.0]


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTGeneratedCodeExecution:
    """Test that generated C++ code compiles and executes in ROOT."""
    
    def test_rvec_nested_type_available(self):
        """Test that RVec<RVec<double>> type is available."""
        import ROOT
        
        # This should not raise
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<ROOT::RVec<double>> make_nested_test() {
                ROOT::RVec<ROOT::RVec<double>> outer;
                outer.push_back(ROOT::RVec<double>{1.0, 2.0});
                outer.push_back(ROOT::RVec<double>{3.0, 4.0, 5.0});
                return outer;
            }
        ''')
        
        result = ROOT.make_nested_test()
        assert len(result) == 2
    
    def test_column_extract_lambda_compiles(self):
        """Test that column extract lambda compiles."""
        import ROOT
        
        # Declare test nested RVec
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<ROOT::RVec<double>> g_nested;
        ''')
        
        # Initialize (need to do at runtime)
        ROOT.g_nested.clear()
        r0 = ROOT.RVec['double']()
        r0.push_back(1.0)
        r0.push_back(2.0)
        ROOT.g_nested.push_back(r0)
        r1 = ROOT.RVec['double']()
        r1.push_back(3.0)
        r1.push_back(4.0)
        ROOT.g_nested.push_back(r1)
        
        # Generate and compile column extract code
        node = make_nested_column_extract("g_nested", 0)
        result = generate_nested_code(node)
        
        func_code = f'''
            ROOT::RVec<double> extract_column() {{
                return {result.code};
            }}
        '''
        ROOT.gInterpreter.Declare(func_code)
        
        col = ROOT.extract_column()
        assert len(col) == 2
        assert col[0] == 1.0
        assert col[1] == 3.0
    
    def test_column_slice_lambda_compiles(self):
        """Test that column slice lambda compiles."""
        import ROOT
        
        # Declare test nested RVec for this test
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<ROOT::RVec<double>> g_nested_slice;
        ''')
        
        # Initialize
        ROOT.g_nested_slice.clear()
        r0 = ROOT.RVec['double']()
        r0.push_back(1.0)
        r0.push_back(2.0)
        ROOT.g_nested_slice.push_back(r0)
        r1 = ROOT.RVec['double']()
        r1.push_back(3.0)
        r1.push_back(4.0)
        ROOT.g_nested_slice.push_back(r1)
        
        # Generate and compile column slice code
        params = SliceParams(stop=1)
        node = make_nested_column_slice("g_nested_slice", params)
        result = generate_nested_code(node)
        
        func_code = f'''
            ROOT::RVec<ROOT::RVec<double>> slice_column_test() {{
                return {result.code};
            }}
        '''
        ROOT.gInterpreter.Declare(func_code)
        
        sliced = ROOT.slice_column_test()
        assert len(sliced) == 2
        assert len(sliced[0]) == 1
        assert sliced[0][0] == 1.0


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestROOTFailClosedBehavior:
    """Test fail-closed behavior with actual ROOT execution."""
    
    def test_column_extract_throws_on_missing(self):
        """Column extract throws when row is too short."""
        import ROOT
        
        # Create ragged nested RVec
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<ROOT::RVec<double>> g_ragged;
        ''')
        
        ROOT.g_ragged.clear()
        r0 = ROOT.RVec['double']()
        r0.push_back(1.0)
        r0.push_back(2.0)
        r0.push_back(3.0)
        ROOT.g_ragged.push_back(r0)
        r1 = ROOT.RVec['double']()
        r1.push_back(10.0)  # Only 1 element!
        ROOT.g_ragged.push_back(r1)
        
        # Try to access index 2 - should throw on row 1
        node = make_nested_column_extract("g_ragged", 2)
        result = generate_nested_code(node)
        
        func_code = f'''
            ROOT::RVec<double> extract_will_throw() {{
                return {result.code};
            }}
        '''
        ROOT.gInterpreter.Declare(func_code)
        
        # This should throw
        with pytest.raises(Exception):  # ROOT wraps as cppyy.gbl.std.runtime_error
            ROOT.extract_will_throw()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
