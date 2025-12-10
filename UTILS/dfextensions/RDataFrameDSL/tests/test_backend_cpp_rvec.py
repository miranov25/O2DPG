"""
Tests for Phase 6b: RVec Operations in C++ Code Generation.

This module tests:
- RVec parameter type generation (const ROOT::RVec<T>&)
- RVec return type generation (ROOT::RVec<T>)
- RVec arithmetic code generation
- Simple indexing with safe bounds checking
- Negative literal indexing
- RVec methods (size, empty, at)
- Header collection (<ROOT/RVec.hxx>, <limits>)

Test Categories:
- Mock Tests: Code generation patterns (no ROOT required)
- ROOT Integration Tests: Compile and execute (requires ROOT)
- RDataFrame Tests: End-to-end with vector branches
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional

from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_nodes import (
    VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode,
    CallNode, MethodCallNode, SubscriptNode, SliceNode,
    BinaryOp, UnaryOp
)
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.backend_cpp import (
    CppCodeGenerator, FunctionLibrary, GeneratedFunction,
    RVEC_HEADER, RVEC_METHODS
)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


# =============================================================================
# Test Helpers
# =============================================================================

def make_rvec_var(name: str, inner_type: IRTypeKind = IRTypeKind.Float64) -> VariableNode:
    """Create a VariableNode representing an RVec."""
    return VariableNode(
        name=name,
        dtype=IRType(inner_type),
        rank=1  # rank 1 = RVec
    )

def make_scalar_var(name: str, dtype_kind: IRTypeKind = IRTypeKind.Float64) -> VariableNode:
    """Create a scalar VariableNode."""
    return VariableNode(
        name=name,
        dtype=IRType(dtype_kind),
        rank=0
    )

def make_int_const(value: int) -> ConstantNode:
    """Create an integer constant."""
    return ConstantNode(
        value=value,
        dtype=IRType(IRTypeKind.Int32),
        rank=0
    )

def make_float_const(value: float) -> ConstantNode:
    """Create a float constant."""
    return ConstantNode(
        value=value,
        dtype=IRType(IRTypeKind.Float64),
        rank=0
    )


# =============================================================================
# Mock Tests - RVec Parameter Types
# =============================================================================

class TestRVecParameterType:
    """Tests for RVec parameter type generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_rvec_float64_parameter(self, generator):
        """RVec<double> is passed as const reference."""
        pt = make_rvec_var("pt", IRTypeKind.Float64)
        
        # Simple expression: pt.size()
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n_tracks")
        
        assert "const ROOT::RVec<double>& pt" in func.code
    
    def test_rvec_float32_parameter(self, generator):
        """RVec<float> is passed as const reference."""
        pt = make_rvec_var("pt", IRTypeKind.Float32)
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n")
        
        assert "const ROOT::RVec<float>& pt" in func.code
    
    def test_rvec_int32_parameter(self, generator):
        """RVec<int> is passed as const reference."""
        ids = make_rvec_var("ids", IRTypeKind.Int32)
        
        ir = MethodCallNode(
            object=ids,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n_ids")
        
        assert "const ROOT::RVec<int>& ids" in func.code


# =============================================================================
# Mock Tests - RVec Return Types
# =============================================================================

class TestRVecReturnType:
    """Tests for RVec return type generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_rvec_arithmetic_return_type(self, generator):
        """RVec arithmetic returns RVec."""
        pt = make_rvec_var("pt")
        scalar = make_float_const(1.5)
        
        # pt * 1.5 -> RVec<double>
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=pt,
            right=scalar,
            dtype=IRType(IRTypeKind.Float64),
            rank=1  # Result is RVec
        )
        
        func = generator.generate(ir, "scaled_pt")
        
        assert func.return_type == "ROOT::RVec<double>"
    
    def test_scalar_indexing_return_type(self, generator):
        """Indexing RVec returns scalar."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        # pt[0] -> double
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0  # Result is scalar
        )
        
        func = generator.generate(ir, "first_pt")
        
        assert func.return_type == "double"


# =============================================================================
# Mock Tests - RVec Arithmetic
# =============================================================================

class TestRVecArithmeticCodeGeneration:
    """Tests for RVec arithmetic code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_rvec_multiply_scalar(self, generator):
        """pt * 1.5 generates correct code."""
        pt = make_rvec_var("pt")
        scalar = make_float_const(1.5)
        
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=pt,
            right=scalar,
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        func = generator.generate(ir, "scaled")
        
        assert "(pt * 1.5)" in func.code
    
    def test_rvec_add_rvec(self, generator):
        """px + py generates correct code."""
        px = make_rvec_var("px")
        py = make_rvec_var("py")
        
        ir = BinaryOpNode(
            op=BinaryOp.ADD,
            left=px,
            right=py,
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        func = generator.generate(ir, "sum")
        
        assert "(px + py)" in func.code
        assert "const ROOT::RVec<double>& px" in func.code
        assert "const ROOT::RVec<double>& py" in func.code
    
    def test_rvec_divide_scalar(self, generator):
        """pt / 2.0 generates correct code."""
        pt = make_rvec_var("pt")
        scalar = make_float_const(2.0)
        
        ir = BinaryOpNode(
            op=BinaryOp.DIV,
            left=pt,
            right=scalar,
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        func = generator.generate(ir, "half_pt")
        
        assert "(pt / 2.0)" in func.code
    
    def test_rvec_comparison(self, generator):
        """pt > 1.0 generates correct code."""
        pt = make_rvec_var("pt")
        threshold = make_float_const(1.0)
        
        ir = BinaryOpNode(
            op=BinaryOp.GT,
            left=pt,
            right=threshold,
            dtype=IRType(IRTypeKind.Bool),
            rank=1  # Result is RVec<bool>
        )
        
        func = generator.generate(ir, "high_pt")
        
        assert "(pt > 1.0)" in func.code
        assert func.return_type == "ROOT::RVec<bool>"


# =============================================================================
# Mock Tests - Simple Indexing
# =============================================================================

class TestRVecIndexingCodeGeneration:
    """Tests for RVec indexing code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def unsafe_generator(self):
        return CppCodeGenerator(safe_indexing=False)
    
    def test_safe_index_literal(self, generator):
        """pt[0] generates safe bounds-checked code."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "first")
        
        # Check for bounds check pattern
        assert "0 >= 0" in func.code
        assert "pt.size()" in func.code
        assert "quiet_NaN" in func.code
        assert "<limits>" in func.headers
    
    def test_safe_index_variable(self, generator):
        """pt[i] generates safe bounds-checked code."""
        pt = make_rvec_var("pt")
        i = make_scalar_var("i", IRTypeKind.Int32)
        
        ir = SubscriptNode(
            value=pt,
            indices=[i],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "at_i")
        
        # Check for bounds check pattern
        assert "i >= 0" in func.code
        assert "pt.size()" in func.code
        assert "quiet_NaN" in func.code
    
    def test_unsafe_index_literal(self, unsafe_generator):
        """pt[0] with safe_indexing=False generates direct access."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = unsafe_generator.generate(ir, "first_fast")
        
        # Direct access without bounds check
        assert "pt[0]" in func.code
        assert "quiet_NaN" not in func.code
        assert "<limits>" not in func.headers


# =============================================================================
# Mock Tests - Negative Indexing
# =============================================================================

class TestNegativeIndexingCodeGeneration:
    """Tests for negative literal index code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def unsafe_generator(self):
        return CppCodeGenerator(safe_indexing=False)
    
    def test_negative_one_safe(self, generator):
        """pt[-1] generates safe last element access."""
        pt = make_rvec_var("pt")
        idx = make_int_const(-1)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "last")
        
        # Check for size check and size-1 access
        assert "pt.size() >= 1" in func.code
        assert "pt.size() - 1" in func.code
        assert "quiet_NaN" in func.code
    
    def test_negative_two_safe(self, generator):
        """pt[-2] generates safe second-to-last access."""
        pt = make_rvec_var("pt")
        idx = make_int_const(-2)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "second_last")
        
        assert "pt.size() >= 2" in func.code
        assert "pt.size() - 2" in func.code
    
    def test_negative_one_unsafe(self, unsafe_generator):
        """pt[-1] with safe_indexing=False generates direct access."""
        pt = make_rvec_var("pt")
        idx = make_int_const(-1)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = unsafe_generator.generate(ir, "last_fast")
        
        assert "pt.size() - 1" in func.code
        assert "quiet_NaN" not in func.code


# =============================================================================
# Mock Tests - RVec Methods
# =============================================================================

class TestRVecMethodsCodeGeneration:
    """Tests for RVec methods code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_size_method(self, generator):
        """tracks.size() generates correct code."""
        tracks = make_rvec_var("tracks")
        
        ir = MethodCallNode(
            object=tracks,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n_tracks")
        
        assert "tracks.size()" in func.code
        # size_t is typically unsigned long
        assert func.return_type in ("unsigned long", "size_t")
    
    def test_empty_method(self, generator):
        """tracks.empty() generates correct code."""
        tracks = make_rvec_var("tracks")
        
        ir = MethodCallNode(
            object=tracks,
            method_name="empty",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "is_empty")
        
        assert "tracks.empty()" in func.code
        assert func.return_type == "bool"
    
    def test_at_method(self, generator):
        """tracks.at(i) generates correct code."""
        tracks = make_rvec_var("tracks")
        i = make_scalar_var("i", IRTypeKind.Int32)
        
        ir = MethodCallNode(
            object=tracks,
            method_name="at",
            args=[i],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "track_at")
        
        assert "tracks.at(i)" in func.code


# =============================================================================
# Mock Tests - Header Collection
# =============================================================================

class TestRVecHeaderCollection:
    """Tests for header collection with RVec operations."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_rvec_header_included(self, generator):
        """RVec operations include <ROOT/RVec.hxx>."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n")
        
        assert RVEC_HEADER in func.headers
    
    def test_limits_header_with_safe_indexing(self, generator):
        """Safe indexing includes <limits>."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "first")
        
        assert "<limits>" in func.headers
        assert RVEC_HEADER in func.headers
    
    def test_no_limits_without_indexing(self, generator):
        """Operations without indexing don't include <limits>."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n")
        
        assert "<limits>" not in func.headers


# =============================================================================
# Mock Tests - Vectorized Math
# =============================================================================

class TestVectorizedMathCodeGeneration:
    """Tests for vectorized math function code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_sqrt_rvec(self, generator):
        """sqrt(pt) generates code using ADL."""
        pt = make_rvec_var("pt")
        
        ir = CallNode(
            func="sqrt",
            args=[pt],
            dtype=IRType(IRTypeKind.Float64),
            rank=1  # Result is RVec
        )
        
        func = generator.generate(ir, "sqrt_pt")
        
        # Phase 9: Use unqualified sqrt for ADL to find ROOT::VecOps::sqrt
        assert "sqrt(pt)" in func.code
        assert "std::sqrt" not in func.code  # Should NOT use std:: for RVec
        assert func.return_type == "ROOT::RVec<double>"
    
    def test_abs_rvec(self, generator):
        """abs(eta) generates code using ADL."""
        eta = make_rvec_var("eta")
        
        ir = CallNode(
            func="abs",
            args=[eta],
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        func = generator.generate(ir, "abs_eta")
        
        # Phase 9: Use unqualified abs for ADL to find ROOT::VecOps::abs
        assert "abs(eta)" in func.code
        assert "std::abs" not in func.code  # Should NOT use std:: for RVec


# =============================================================================
# Mock Tests - Error Cases
# =============================================================================

class TestRVecErrorCases:
    """Tests for error cases in RVec operations."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_slicing_error(self, generator):
        """Slicing raises unsupported error."""
        pt = make_rvec_var("pt")
        
        slice_node = SliceNode(
            start=make_int_const(1),
            stop=make_int_const(3),
            step=None,
            dtype=IRType(IRTypeKind.Unknown),
            rank=0
        )
        
        ir = SubscriptNode(
            value=pt,
            indices=[slice_node],
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "slicing" in exc_info.value.message.lower()
    
    def test_multi_index_error(self, generator):
        """Multi-dimensional indexing raises error."""
        arr = make_rvec_var("arr")
        
        ir = SubscriptNode(
            value=arr,
            indices=[make_int_const(0), make_int_const(1)],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "multi-dimensional" in exc_info.value.message.lower()
    
    def test_boolean_mask_error(self, generator):
        """Boolean mask indexing raises error."""
        pt = make_rvec_var("pt")
        mask = make_rvec_var("mask")
        mask.dtype = IRType(IRTypeKind.Bool)
        
        ir = SubscriptNode(
            value=pt,
            indices=[mask],
            is_boolean_mask=True,
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "boolean" in exc_info.value.message.lower()
    
    def test_nested_rvec_error(self, generator):
        """Nested RVec (rank > 1) raises error."""
        nested = VariableNode(
            name="nested",
            dtype=IRType(IRTypeKind.Float64),
            rank=2  # Nested RVec
        )
        
        ir = MethodCallNode(
            object=nested,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "nested" in exc_info.value.message.lower()


# =============================================================================
# ROOT Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRVecROOTCompilation:
    """ROOT integration tests - compile generated code."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_compile_rvec_size(self, generator, library):
        """RVec.size() compiles successfully."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "rvec_size")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_compile_rvec_empty(self, generator, library):
        """RVec.empty() compiles successfully."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="empty",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "rvec_empty")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_compile_rvec_index_safe(self, generator, library):
        """Safe indexing compiles successfully."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "rvec_first_safe")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_compile_rvec_negative_index(self, generator, library):
        """Negative indexing compiles successfully."""
        pt = make_rvec_var("pt")
        idx = make_int_const(-1)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "rvec_last")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_compile_rvec_arithmetic(self, generator, library):
        """RVec arithmetic compiles successfully."""
        pt = make_rvec_var("pt")
        scalar = make_float_const(2.0)
        
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=pt,
            right=scalar,
            dtype=IRType(IRTypeKind.Float64),
            rank=1
        )
        
        func = generator.generate(ir, "rvec_scaled")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRVecROOTExecution:
    """ROOT integration tests - execute compiled functions."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_execute_rvec_size(self, generator, library):
        """RVec.size() executes correctly."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "exec_size")
        library.add(func)
        library.compile(func.name)
        
        result = ROOT.gInterpreter.Calc('''
            ROOT::RVec<double> v = {1.0, 2.0, 3.0};
            alias_exec_size(v)
        ''')
        
        assert result == 3
    
    def test_execute_rvec_index_first(self, generator, library):
        """pt[0] returns first element."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "exec_first")
        library.add(func)
        library.compile(func.name)
        
        # Create RVec directly in Python
        v = ROOT.RVec('double')([1.5, 2.5, 3.5])
        
        # Call via PyROOT (preserves double type)
        result = getattr(ROOT, func.name)(v)
        
        assert abs(result - 1.5) < 0.001
    
    def test_execute_rvec_negative_index(self, generator, library):
        """pt[-1] returns last element."""
        pt = make_rvec_var("pt")
        idx = make_int_const(-1)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "exec_last")
        library.add(func)
        library.compile(func.name)
        
        # Create RVec directly in Python
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        
        # Call via PyROOT (preserves double type)
        result = getattr(ROOT, func.name)(v)
        
        assert abs(result - 3.0) < 0.001
    
    def test_execute_rvec_oob_returns_nan(self, generator, library):
        """Out-of-bounds access returns NaN."""
        import math
        
        pt = make_rvec_var("pt")
        idx = make_int_const(999)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "exec_oob")
        library.add(func)
        library.compile(func.name)
        
        # Create RVec with only 3 elements
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        
        # Call via PyROOT - index 999 is out of bounds
        result = getattr(ROOT, func.name)(v)
        
        # Should return NaN for out-of-bounds access
        assert math.isnan(result), f"Expected NaN, got {result}"
    
    def test_execute_empty_vec_returns_nan(self, generator, library):
        """Indexing empty vector returns NaN."""
        import math
        
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "exec_empty_idx")
        library.add(func)
        library.compile(func.name)
        
        # Create empty RVec
        v = ROOT.RVec('double')()
        
        # Call via PyROOT - index 0 on empty vector is out of bounds
        result = getattr(ROOT, func.name)(v)
        
        # Should return NaN for empty vector access
        assert math.isnan(result), f"Expected NaN, got {result}"


# =============================================================================
# RDataFrame Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRVecRDataFrameIntegration:
    """RDataFrame end-to-end tests with RVec branches."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    @pytest.fixture
    def rdf_with_rvec(self, tmp_path):
        """Create RDataFrame with RVec columns."""
        import ROOT
        
        # Create RDataFrame with RVec columns using FromSpec
        rdf = ROOT.RDataFrame(3)
        
        # Define vector columns
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = rdf.Define("eta", "ROOT::RVec<double>{0.1, 0.2, 0.3}")
        
        return rdf
    
    def test_rdataframe_rvec_size(self, generator, library, rdf_with_rvec):
        """RDataFrame.Define with RVec.size()."""
        pt = make_rvec_var("pt")
        
        ir = MethodCallNode(
            object=pt,
            method_name="size",
            args=[],
            dtype=IRType(IRTypeKind.UInt64),
            rank=0
        )
        
        func = generator.generate(ir, "n_pt")
        library.add(func)
        library.compile(func.name)
        
        # Use the function in RDataFrame
        rdf2 = rdf_with_rvec.Define("n_tracks", func.get_call_expression())
        
        # Get results
        result = rdf2.Take["unsigned long"]("n_tracks").GetValue()
        
        assert list(result) == [3, 3, 3]
    
    def test_rdataframe_rvec_first_element(self, generator, library, rdf_with_rvec):
        """RDataFrame.Define with pt[0]."""
        pt = make_rvec_var("pt")
        idx = make_int_const(0)
        
        ir = SubscriptNode(
            value=pt,
            indices=[idx],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "first_pt")
        library.add(func)
        library.compile(func.name)
        
        rdf2 = rdf_with_rvec.Define("lead_pt", func.get_call_expression())
        result = rdf2.Take["double"]("lead_pt").GetValue()
        
        for val in result:
            assert abs(val - 1.0) < 0.001


# =============================================================================
# Phase 7: Slice Code Generation Tests
# =============================================================================

class TestSliceCodeGeneration:
    """Tests for Phase 7 slice code generation patterns."""
    
    @pytest.fixture
    def slice_setup(self):
        """Set up builder and generator for slice tests."""
        from RDataFrameDSL import IRBuilder, TypeInferrer, CppCodeGenerator
        
        schema = {
            "columns": {
                "pt": {"dtype": "double", "rank": 1, "cpp_type": "RVec<double>"},
                "mask": {"dtype": "bool", "rank": 1, "cpp_type": "RVec<bool>"},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        generator = CppCodeGenerator(inferrer)
        
        return builder, generator
    
    def test_gen_first_n(self, slice_setup):
        """[:3] generates Take with clamping."""
        builder, generator = slice_setup
        ir = builder.build("pt[:3]")
        func = generator.generate(ir, "first_3")
        
        assert "ROOT::VecOps::Take" in func.code
        assert "std::min" in func.code  # Clamps to vector size
    
    def test_gen_last_n(self, slice_setup):
        """[-3:] generates Take with clamping."""
        builder, generator = slice_setup
        ir = builder.build("pt[-3:]")
        func = generator.generate(ir, "last_3")
        
        assert "ROOT::VecOps::Take" in func.code
        assert "std::min" in func.code  # Clamps to vector size
    
    def test_gen_from_index(self, slice_setup):
        """[2:] generates lambda with Range."""
        builder, generator = slice_setup
        ir = builder.build("pt[2:]")
        func = generator.generate(ir, "from_2")
        
        assert "Range" in func.code
        assert "start = 2" in func.code
    
    def test_gen_range_has_clamp(self, slice_setup):
        """[1:3] includes std::min clamp."""
        builder, generator = slice_setup
        ir = builder.build("pt[1:3]")
        func = generator.generate(ir, "range_1_3")
        
        assert "std::min" in func.code
        assert "start >= stop" in func.code  # Empty check
    
    def test_gen_step_has_loop(self, slice_setup):
        """[::2] includes loop-based indexing."""
        builder, generator = slice_setup
        ir = builder.build("pt[::2]")
        func = generator.generate(ir, "step_2")
        
        assert "for (size_t i" in func.code
        assert "i += 2" in func.code
        assert "indices.push_back" in func.code
    
    def test_gen_step_with_start(self, slice_setup):
        """[1::2] includes loop starting at 1."""
        builder, generator = slice_setup
        ir = builder.build("pt[1::2]")
        func = generator.generate(ir, "step_1_2")
        
        assert "i = 1" in func.code
        assert "i += 2" in func.code
    
    def test_gen_step_with_range(self, slice_setup):
        """[1:5:2] includes loop with bounds and step."""
        builder, generator = slice_setup
        ir = builder.build("pt[1:5:2]")
        func = generator.generate(ir, "step_1_5_2")
        
        assert "i = 1" in func.code
        assert "std::min" in func.code
        assert "i += 2" in func.code
    
    def test_gen_reverse_no_vecops_reverse(self, slice_setup):
        """[::-1] does NOT use VecOps::Reverse."""
        builder, generator = slice_setup
        ir = builder.build("pt[::-1]")
        func = generator.generate(ir, "reverse")
        
        assert "Reverse" not in func.code  # We don't use VecOps::Reverse
        assert "i-- > 0" in func.code  # Manual loop
        assert "reserve" in func.code
    
    def test_gen_boolean_mask_native(self, slice_setup):
        """[pt > 1.0] generates native v[mask]."""
        builder, generator = slice_setup
        ir = builder.build("pt[pt > 1.0]")
        func = generator.generate(ir, "gt_1")
        
        # Should use native boolean indexing
        assert "pt[" in func.code
        assert "pt > 1.0" in func.code
    
    def test_gen_boolean_mask_variable(self, slice_setup):
        """[mask] generates native v[mask]."""
        builder, generator = slice_setup
        ir = builder.build("pt[mask]")
        func = generator.generate(ir, "with_mask")
        
        assert "pt[mask]" in func.code
    
    def test_gen_full_slice(self, slice_setup):
        """[:] generates full copy via Range."""
        builder, generator = slice_setup
        ir = builder.build("pt[:]")
        func = generator.generate(ir, "full")
        
        assert "Range" in func.code or "Take" in func.code
    
    def test_slice_return_type(self, slice_setup):
        """Slice return type is RVec<T>."""
        builder, generator = slice_setup
        ir = builder.build("pt[:3]")
        func = generator.generate(ir, "ret_type")
        
        assert "ROOT::RVec<double>" in func.code
    
    def test_slice_needs_algorithm_header(self, slice_setup):
        """Range slice needs <algorithm> for std::min."""
        builder, generator = slice_setup
        ir = builder.build("pt[1:3]")
        func = generator.generate(ir, "headers")
        
        assert "<algorithm>" in func.headers
    
    def test_slice_needs_rvec_header(self, slice_setup):
        """Slices need RVec header."""
        builder, generator = slice_setup
        ir = builder.build("pt[:3]")
        func = generator.generate(ir, "rvec_header")
        
        # Check for RVec header (may be in different forms)
        has_rvec = any("RVec" in h for h in func.headers)
        assert has_rvec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
