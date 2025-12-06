"""
Tests for C++ code generation backend (Phase 5).

This module contains:
1. Mock tests - Test code generation logic without ROOT
2. ROOT integration tests - Test actual compilation
3. End-to-end RDataFrame test - Verify full pipeline

Mock tests run everywhere. ROOT tests are skipped if ROOT unavailable.
"""

import pytest
import os
import tempfile
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_nodes import (
    ConstantNode, VariableNode, UnaryOpNode, BinaryOpNode, TernaryOpNode,
    CallNode, MethodCallNode, PropertyAccessNode, SubscriptNode, SliceNode,
    UnaryOp, BinaryOp
)
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.backend_cpp import (
    CppCodeGenerator, GeneratedFunction, FunctionLibrary,
    FUNCTION_HEADERS, FUNCTION_CPP_NAMES
)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    ROOT = None


# =============================================================================
# Helper Functions for Test Setup
# =============================================================================

def make_int_var(name: str) -> VariableNode:
    """Create an int variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Int32),
        rank=0
    )

def make_float_var(name: str) -> VariableNode:
    """Create a float variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Float32),
        rank=0
    )

def make_double_var(name: str) -> VariableNode:
    """Create a double variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Float64),
        rank=0
    )

def make_bool_var(name: str) -> VariableNode:
    """Create a bool variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Bool),
        rank=0
    )

def make_int_const(value: int) -> ConstantNode:
    """Create an int constant node (note: becomes Int64 via ConstantNode)."""
    return ConstantNode(value=value)
    # Note: ConstantNode.__post_init__ sets dtype to Int64 for Python ints

def make_float_const(value: float) -> ConstantNode:
    """Create a float constant node (note: becomes Float64 via ConstantNode)."""
    return ConstantNode(value=value)
    # Note: ConstantNode.__post_init__ sets dtype to Float64 for Python floats

def make_bool_const(value: bool) -> ConstantNode:
    """Create a bool constant node."""
    return ConstantNode(value=value)
    # Note: ConstantNode.__post_init__ sets dtype to Bool for Python bools


# =============================================================================
# GeneratedFunction Tests
# =============================================================================

class TestGeneratedFunction:
    """Tests for GeneratedFunction dataclass."""
    
    def test_get_call_expression(self):
        """get_call_expression returns correct format."""
        func = GeneratedFunction(
            name="alias_pt",
            code="double alias_pt(double px, double py) { return px + py; }",
            inputs=[("px", "double"), ("py", "double")],
            return_type="double",
            headers=["<cmath>"]
        )
        assert func.get_call_expression() == "alias_pt(px, py)"
    
    def test_get_call_expression_no_args(self):
        """get_call_expression with no arguments."""
        func = GeneratedFunction(
            name="alias_pi",
            code="double alias_pi() { return 3.14159; }",
            inputs=[],
            return_type="double",
            headers=[]
        )
        assert func.get_call_expression() == "alias_pi()"
    
    def test_get_signature(self):
        """get_signature returns function signature."""
        func = GeneratedFunction(
            name="alias_sum",
            code="...",
            inputs=[("a", "int"), ("b", "float")],
            return_type="float",
            headers=[]
        )
        assert func.get_signature() == "float alias_sum(int a, float b)"


# =============================================================================
# CppCodeGenerator - Constant Tests
# =============================================================================

class TestCppCodeGeneratorConstants:
    """Tests for constant code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_int_constant(self, generator):
        """Integer constant generates correctly."""
        ir = make_int_const(42)
        func = generator.generate(ir, "answer")
        
        assert "return 42;" in func.code
        # Note: Python integers become Int64 in IR (see ConstantNode.__post_init__)
        assert func.return_type == "long long"
        assert func.inputs == []
    
    def test_float_constant(self, generator):
        """Float constant includes decimal point."""
        ir = make_float_const(3.14)
        func = generator.generate(ir, "pi")
        
        assert "3.14" in func.code
        assert func.return_type == "double"
    
    def test_float_constant_whole_number(self, generator):
        """Float constant with whole number includes decimal."""
        ir = make_float_const(3.0)
        func = generator.generate(ir, "three")
        
        # Should have decimal point
        assert "3.0" in func.code or "3." in func.code
    
    def test_bool_constant_true(self, generator):
        """Boolean true constant."""
        ir = make_bool_const(True)
        func = generator.generate(ir, "flag")
        
        assert "return true;" in func.code
        assert func.return_type == "bool"
    
    def test_bool_constant_false(self, generator):
        """Boolean false constant."""
        ir = make_bool_const(False)
        func = generator.generate(ir, "flag")
        
        assert "return false;" in func.code
    
    def test_negative_int_constant(self, generator):
        """Negative integer constant."""
        ir = make_int_const(-42)
        func = generator.generate(ir, "neg")
        
        assert "-42" in func.code


# =============================================================================
# CppCodeGenerator - Variable Tests
# =============================================================================

class TestCppCodeGeneratorVariables:
    """Tests for variable code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_single_variable(self, generator):
        """Single variable becomes function argument."""
        ir = make_double_var("x")
        func = generator.generate(ir, "identity")
        
        assert "double alias_identity(double x)" in func.code
        assert "return x;" in func.code
        assert func.inputs == [("x", "double")]
    
    def test_multiple_variables_alphabetical(self, generator):
        """Multiple variables are sorted alphabetically."""
        # Create expression: z + a + m
        z = make_double_var("z")
        a = make_double_var("a")
        m = make_double_var("m")
        
        za = BinaryOpNode(
            op=BinaryOp.ADD,
            left=z,
            right=a,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        ir = BinaryOpNode(
            op=BinaryOp.ADD,
            left=za,
            right=m,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sum")
        
        # Should be alphabetical: a, m, z
        assert func.inputs == [("a", "double"), ("m", "double"), ("z", "double")]
        assert "double a, double m, double z" in func.code
    
    def test_variable_used_multiple_times(self, generator):
        """Variable used multiple times appears once in signature."""
        x = make_double_var("x")
        
        # x * x
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=x,
            right=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "square")
        
        assert func.inputs == [("x", "double")]
        assert func.code.count("double x") == 1  # Only in signature
    
    def test_unknown_variable_error(self, generator):
        """Unknown variable type raises error."""
        ir = VariableNode(
            name="unknown",
            dtype=IRType(IRTypeKind.Unknown),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
        assert "unknown" in exc_info.value.message.lower()


# =============================================================================
# CppCodeGenerator - Arithmetic Tests
# =============================================================================

class TestCppCodeGeneratorArithmetic:
    """Tests for arithmetic operation code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_addition(self, generator):
        """Addition generates correctly."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.ADD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sum")
        assert "(x + y)" in func.code
    
    def test_subtraction(self, generator):
        """Subtraction generates correctly."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.SUB,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "diff")
        assert "(x - y)" in func.code
    
    def test_multiplication(self, generator):
        """Multiplication generates correctly."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "product")
        assert "(x * y)" in func.code
    
    def test_division(self, generator):
        """Division generates correctly."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.DIV,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "ratio")
        assert "(x / y)" in func.code
    
    def test_power(self, generator):
        """Power uses std::pow."""
        x = make_double_var("x")
        two = make_int_const(2)
        ir = BinaryOpNode(
            op=BinaryOp.POW,
            left=x,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "square")
        assert "std::pow(x, 2)" in func.code
        assert "<cmath>" in func.headers
    
    def test_floor_division(self, generator):
        """Floor division uses static_cast."""
        x = make_int_var("x")
        y = make_int_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.FLOORDIV,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Int64),
            rank=0
        )
        
        func = generator.generate(ir, "floordiv")
        assert "static_cast<long long>(x / y)" in func.code
    
    def test_modulo_int(self, generator):
        """Integer modulo generates correctly."""
        x = make_int_var("x")
        y = make_int_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.MOD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "mod")
        assert "(x % y)" in func.code
    
    def test_modulo_float_error(self, generator):
        """Float modulo raises error."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.MOD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "mod")
        
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
        assert "modulo" in exc_info.value.message.lower()
    
    def test_nested_arithmetic(self, generator):
        """Nested arithmetic expression."""
        # (x + y) * z
        x = make_double_var("x")
        y = make_double_var("y")
        z = make_double_var("z")
        
        add = BinaryOpNode(
            op=BinaryOp.ADD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        ir = BinaryOpNode(
            op=BinaryOp.MUL,
            left=add,
            right=z,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "expr")
        assert "((x + y) * z)" in func.code


# =============================================================================
# CppCodeGenerator - Comparison Tests
# =============================================================================

class TestCppCodeGeneratorComparisons:
    """Tests for comparison operation code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_less_than(self, generator):
        """Less than comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.LT,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x < y)" in func.code
        assert func.return_type == "bool"
    
    def test_less_equal(self, generator):
        """Less than or equal comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.LE,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x <= y)" in func.code
    
    def test_greater_than(self, generator):
        """Greater than comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.GT,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x > y)" in func.code
    
    def test_greater_equal(self, generator):
        """Greater than or equal comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.GE,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x >= y)" in func.code
    
    def test_equal(self, generator):
        """Equality comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.EQ,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x == y)" in func.code
    
    def test_not_equal(self, generator):
        """Inequality comparison."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.NE,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cmp")
        assert "(x != y)" in func.code


# =============================================================================
# CppCodeGenerator - Logical Tests
# =============================================================================

class TestCppCodeGeneratorLogical:
    """Tests for logical operation code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_logical_and(self, generator):
        """Logical AND."""
        x = make_bool_var("x")
        y = make_bool_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.AND,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "both")
        assert "(x && y)" in func.code
    
    def test_logical_or(self, generator):
        """Logical OR."""
        x = make_bool_var("x")
        y = make_bool_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.OR,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "either")
        assert "(x || y)" in func.code
    
    def test_logical_not(self, generator):
        """Logical NOT."""
        x = make_bool_var("x")
        ir = UnaryOpNode(
            op=UnaryOp.NOT,
            operand=x,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "negate")
        assert "(!x)" in func.code


# =============================================================================
# CppCodeGenerator - Unary Tests
# =============================================================================

class TestCppCodeGeneratorUnary:
    """Tests for unary operation code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_negation(self, generator):
        """Arithmetic negation."""
        x = make_double_var("x")
        ir = UnaryOpNode(
            op=UnaryOp.NEG,
            operand=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "neg")
        assert "(-x)" in func.code
    
    def test_positive(self, generator):
        """Arithmetic positive (no-op)."""
        x = make_double_var("x")
        ir = UnaryOpNode(
            op=UnaryOp.POS,
            operand=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "pos")
        assert "(+x)" in func.code
    
    def test_bitwise_not_int(self, generator):
        """Bitwise NOT on integer."""
        x = make_int_var("x")
        ir = UnaryOpNode(
            op=UnaryOp.BITNOT,
            operand=x,
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "bitnot")
        assert "(~x)" in func.code
    
    def test_bitwise_not_float_error(self, generator):
        """Bitwise NOT on float raises error."""
        x = make_double_var("x")
        ir = UnaryOpNode(
            op=UnaryOp.BITNOT,
            operand=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "bitnot")
        
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
        assert "bitwise" in exc_info.value.message.lower()


# =============================================================================
# CppCodeGenerator - Bitwise Tests
# =============================================================================

class TestCppCodeGeneratorBitwise:
    """Tests for bitwise operation code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_bitwise_and(self, generator):
        """Bitwise AND on integers."""
        x = make_int_var("x")
        y = make_int_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.BITAND,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "bitand")
        assert "(x & y)" in func.code
    
    def test_bitwise_or(self, generator):
        """Bitwise OR on integers."""
        x = make_int_var("x")
        y = make_int_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.BITOR,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "bitor")
        assert "(x | y)" in func.code
    
    def test_bitwise_xor(self, generator):
        """Bitwise XOR on integers."""
        x = make_int_var("x")
        y = make_int_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.BITXOR,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "bitxor")
        assert "(x ^ y)" in func.code
    
    def test_bitwise_float_error(self, generator):
        """Bitwise ops on floats raise error."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.BITAND,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "bitand")
        
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR


# =============================================================================
# CppCodeGenerator - Function Call Tests
# =============================================================================

class TestCppCodeGeneratorFunctionCalls:
    """Tests for function call code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_sqrt(self, generator):
        """sqrt maps to std::sqrt."""
        x = make_double_var("x")
        ir = CallNode(
            func="sqrt",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "root")
        assert "std::sqrt(x)" in func.code
        assert "<cmath>" in func.headers
    
    def test_abs(self, generator):
        """abs maps to std::abs."""
        x = make_double_var("x")
        ir = CallNode(
            func="abs",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "absolute")
        assert "std::abs(x)" in func.code
    
    def test_sin(self, generator):
        """sin maps to std::sin."""
        x = make_double_var("x")
        ir = CallNode(
            func="sin",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sine")
        assert "std::sin(x)" in func.code
    
    def test_cos(self, generator):
        """cos maps to std::cos."""
        x = make_double_var("x")
        ir = CallNode(
            func="cos",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "cosine")
        assert "std::cos(x)" in func.code
    
    def test_pow_function(self, generator):
        """pow maps to std::pow."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = CallNode(
            func="pow",
            args=[x, y],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "power")
        assert "std::pow(x, y)" in func.code
    
    def test_tmath_gaus(self, generator):
        """TMath.Gaus maps to TMath::Gaus."""
        x = make_double_var("x")
        mu = make_double_var("mu")
        sigma = make_double_var("sigma")
        
        ir = CallNode(
            func="TMath.Gaus",
            args=[x, mu, sigma],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "gaus")
        assert "TMath::Gaus(mu, sigma, x)" in func.code or "TMath::Gaus(x, mu, sigma)" in func.code
    
    def test_tmath_with_namespace(self, generator):
        """TMath function with explicit namespace."""
        x = make_double_var("x")
        ir = CallNode(
            func="Sqrt",
            namespace="TMath",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tsqrt")
        assert "TMath::Sqrt(x)" in func.code
    
    def test_nested_function_calls(self, generator):
        """Nested function calls."""
        x = make_double_var("x")
        
        # sqrt(abs(x))
        abs_call = CallNode(
            func="abs",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        ir = CallNode(
            func="sqrt",
            args=[abs_call],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "safe_sqrt")
        assert "std::sqrt(std::abs(x))" in func.code


# =============================================================================
# CppCodeGenerator - Ternary Tests
# =============================================================================

class TestCppCodeGeneratorTernary:
    """Tests for ternary/conditional code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_simple_ternary(self, generator):
        """Simple conditional expression."""
        cond = make_bool_var("cond")
        x = make_double_var("x")
        y = make_double_var("y")
        
        ir = TernaryOpNode(
            condition=cond,
            if_true=x,
            if_false=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "select")
        assert "((cond) ? (x) : (y))" in func.code
    
    def test_ternary_with_comparison(self, generator):
        """Conditional with comparison condition."""
        x = make_double_var("x")
        zero = make_int_const(0)
        
        cond = BinaryOpNode(
            op=BinaryOp.GT,
            left=x,
            right=zero,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        neg_x = UnaryOpNode(
            op=UnaryOp.NEG,
            operand=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        ir = TernaryOpNode(
            condition=cond,
            if_true=x,
            if_false=neg_x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "abs_manual")
        assert "(x > 0)" in func.code
        assert "(-x)" in func.code


# =============================================================================
# CppCodeGenerator - Complex Expressions
# =============================================================================

class TestCppCodeGeneratorComplexExpressions:
    """Tests for complex expression code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_pt_calculation(self, generator):
        """sqrt(px**2 + py**2) - typical physics calculation."""
        px = make_double_var("px")
        py = make_double_var("py")
        two = make_int_const(2)
        
        px_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=px,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        py_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=py,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        sum_sq = BinaryOpNode(
            op=BinaryOp.ADD,
            left=px_sq,
            right=py_sq,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        ir = CallNode(
            func="sqrt",
            args=[sum_sq],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "pt")
        
        assert "std::sqrt" in func.code
        assert "std::pow(px, 2)" in func.code
        assert "std::pow(py, 2)" in func.code
        assert "<cmath>" in func.headers
    
    def test_boolean_cut(self, generator):
        """x > 0 and y < 10 - typical analysis cut."""
        x = make_double_var("x")
        y = make_double_var("y")
        zero = make_int_const(0)
        ten = make_int_const(10)
        
        cmp1 = BinaryOpNode(
            op=BinaryOp.GT,
            left=x,
            right=zero,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        cmp2 = BinaryOpNode(
            op=BinaryOp.LT,
            left=y,
            right=ten,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        ir = BinaryOpNode(
            op=BinaryOp.AND,
            left=cmp1,
            right=cmp2,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "cut")
        
        assert "(x > 0)" in func.code
        assert "(y < 10)" in func.code
        assert "&&" in func.code
        assert func.return_type == "bool"


# =============================================================================
# CppCodeGenerator - Function Naming
# =============================================================================

class TestCppCodeGeneratorNaming:
    """Tests for function naming."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_simple_name(self, generator):
        """Simple name sanitization."""
        ir = make_int_const(42)
        func = generator.generate(ir, "my_alias")
        
        assert func.name == "alias_my_alias"
    
    def test_special_characters(self, generator):
        """Special characters replaced with underscore."""
        ir = make_int_const(42)
        func = generator.generate(ir, "my-alias.v2")
        
        assert func.name == "alias_my_alias_v2"
    
    def test_collision_handling(self, generator):
        """Name collision adds hash suffix."""
        ir = make_int_const(42)
        
        func1 = generator.generate(ir, "test")
        func2 = generator.generate(ir, "test")
        
        assert func1.name == "alias_test"
        assert func2.name.startswith("alias_test_")
        assert func1.name != func2.name


# =============================================================================
# CppCodeGenerator - Header Collection
# =============================================================================

class TestCppCodeGeneratorHeaders:
    """Tests for header collection."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_cmath_from_sqrt(self, generator):
        """sqrt includes <cmath>."""
        x = make_double_var("x")
        ir = CallNode(
            func="sqrt",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "root")
        assert "<cmath>" in func.headers
    
    def test_cmath_from_pow_operator(self, generator):
        """Power operator includes <cmath>."""
        x = make_double_var("x")
        ir = BinaryOpNode(
            op=BinaryOp.POW,
            left=x,
            right=make_int_const(2),
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "square")
        assert "<cmath>" in func.headers
    
    def test_no_headers_for_arithmetic(self, generator):
        """Basic arithmetic needs no headers."""
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.ADD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sum")
        assert func.headers == []


# =============================================================================
# CppCodeGenerator - Error Cases
# =============================================================================

class TestCppCodeGeneratorErrors:
    """Tests for error handling."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_method_call_error(self, generator):
        """Method calls raise unsupported error."""
        obj = make_double_var("obj")
        obj.dtype = IRType(IRTypeKind.Object, "MyClass")
        
        ir = MethodCallNode(
            object=obj,
            method_name="getValue",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "method" in exc_info.value.message.lower()
    
    def test_property_access_error(self, generator):
        """Property access raises unsupported error."""
        obj = make_double_var("obj")
        obj.dtype = IRType(IRTypeKind.Object, "MyClass")
        
        ir = PropertyAccessNode(
            object=obj,
            property_name="value",
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
    
    def test_subscript_error(self, generator):
        """Subscript raises unsupported error."""
        arr = make_double_var("arr")
        arr.dtype = IRType(IRTypeKind.Float64)
        arr.rank = 1  # Vector
        
        ir = SubscriptNode(
            value=arr,
            indices=[make_int_const(0)],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
    
    def test_string_constant_error(self, generator):
        """String constants raise unsupported error."""
        ir = ConstantNode(
            value="hello",
            dtype=IRType(IRTypeKind.Unknown),
            rank=0
        )
        
        with pytest.raises(IRError):
            generator.generate(ir, "test")


# =============================================================================
# FunctionLibrary Tests
# =============================================================================

class TestFunctionLibrary:
    """Tests for FunctionLibrary (mock tests, no ROOT)."""
    
    def test_add_function(self):
        """Add function to library."""
        library = FunctionLibrary()
        func = GeneratedFunction(
            name="alias_test",
            code="double alias_test(double x) { return x; }",
            inputs=[("x", "double")],
            return_type="double",
            headers=[]
        )
        
        library.add(func)
        
        assert "alias_test" in library
        assert len(library) == 1
    
    def test_get_function(self):
        """Get function from library."""
        library = FunctionLibrary()
        func = GeneratedFunction(
            name="alias_test",
            code="...",
            inputs=[("x", "double")],
            return_type="double",
            headers=[]
        )
        library.add(func)
        
        retrieved = library.get_function("alias_test")
        assert retrieved is func
    
    def test_get_define_expression(self):
        """Get expression for RDataFrame.Define()."""
        library = FunctionLibrary()
        func = GeneratedFunction(
            name="alias_sum",
            code="...",
            inputs=[("a", "double"), ("b", "double")],
            return_type="double",
            headers=[]
        )
        library.add(func)
        
        expr = library.get_define_expression("alias_sum")
        assert expr == "alias_sum(a, b)"
    
    def test_list_functions(self):
        """List all functions in library."""
        library = FunctionLibrary()
        
        library.add(GeneratedFunction("alias_b", "...", [], "double", []))
        library.add(GeneratedFunction("alias_a", "...", [], "double", []))
        
        names = library.list_functions()
        assert names == ["alias_a", "alias_b"]  # Sorted
    
    def test_clear(self):
        """Clear library."""
        library = FunctionLibrary()
        library.add(GeneratedFunction("alias_test", "...", [], "double", []))
        
        library.clear()
        
        assert len(library) == 0
    
    def test_save_to_file(self):
        """Save functions to file."""
        library = FunctionLibrary()
        library.add(GeneratedFunction(
            name="alias_a",
            code="double alias_a(double x) { return x; }",
            inputs=[("x", "double")],
            return_type="double",
            headers=["<cmath>"]
        ))
        library.add(GeneratedFunction(
            name="alias_b",
            code="int alias_b(int n) { return n * 2; }",
            inputs=[("n", "int")],
            return_type="int",
            headers=[]
        ))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.C', delete=False) as f:
            path = f.name
        
        try:
            library.save_to_file(path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert "#include <cmath>" in content
            assert "double alias_a(double x)" in content
            assert "int alias_b(int n)" in content
        finally:
            os.unlink(path)
    
    def test_function_not_found_error(self):
        """Get non-existent function raises error."""
        library = FunctionLibrary()
        
        with pytest.raises(IRError) as exc_info:
            library.get_function("nonexistent")
        
        assert exc_info.value.kind == IRErrorKind.VALIDATION_ERROR


# =============================================================================
# ROOT Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestCppCodeGeneratorROOT:
    """ROOT integration tests for code generation and compilation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_compile_simple_function(self, generator, library):
        """Compile simple function via gInterpreter."""
        ir = make_double_var("x")
        func = generator.generate(ir, "identity")
        
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
        assert library.is_compiled(func.name)
    
    def test_compile_with_cmath(self, generator, library):
        """Compile function requiring <cmath>."""
        x = make_double_var("x")
        ir = CallNode(
            func="sqrt",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "root")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_execute_compiled_function(self, generator, library):
        """Execute compiled function via Calc."""
        # x + y
        x = make_double_var("x")
        y = make_double_var("y")
        ir = BinaryOpNode(
            op=BinaryOp.ADD,
            left=x,
            right=y,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sum_test")
        library.add(func)
        library.compile(func.name)
        
        # Execute via ROOT
        result = ROOT.gInterpreter.Calc(f"{func.name}(3.0, 4.0)")
        assert abs(result - 7.0) < 0.001
    
    def test_execute_sqrt(self, generator, library):
        """Execute sqrt function."""
        x = make_double_var("x")
        ir = CallNode(
            func="sqrt",
            args=[x],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "sqrt_test")
        library.add(func)
        library.compile(func.name)
        
        result = ROOT.gInterpreter.Calc(f"{func.name}(16.0)")
        assert abs(result - 4.0) < 0.001
    
    def test_execute_pow(self, generator, library):
        """Execute power function."""
        x = make_double_var("x")
        ir = BinaryOpNode(
            op=BinaryOp.POW,
            left=x,
            right=make_int_const(2),
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "square_test")
        library.add(func)
        library.compile(func.name)
        
        result = ROOT.gInterpreter.Calc(f"{func.name}(5.0)")
        assert abs(result - 25.0) < 0.001
    
    def test_execute_ternary(self, generator, library):
        """Execute conditional expression."""
        x = make_double_var("x")
        zero = make_int_const(0)
        
        cond = BinaryOpNode(
            op=BinaryOp.GT,
            left=x,
            right=zero,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        neg_x = UnaryOpNode(
            op=UnaryOp.NEG,
            operand=x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        ir = TernaryOpNode(
            condition=cond,
            if_true=x,
            if_false=neg_x,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "abs_ternary")
        library.add(func)
        library.compile(func.name)
        
        result_pos = ROOT.gInterpreter.Calc(f"{func.name}(5.0)")
        result_neg = ROOT.gInterpreter.Calc(f"{func.name}(-3.0)")
        
        assert abs(result_pos - 5.0) < 0.001
        assert abs(result_neg - 3.0) < 0.001
    
    def test_compile_all(self, generator, library):
        """compile_all compiles multiple functions."""
        for i in range(3):
            ir = make_int_const(i)
            func = generator.generate(ir, f"const_{i}")
            library.add(func)
        
        errors = library.compile_all()
        
        assert errors == []
        assert all(library.is_compiled(f"alias_const_{i}") for i in range(3))


# =============================================================================
# End-to-End RDataFrame Test
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameIntegration:
    """End-to-end test with RDataFrame."""
    
    def test_rdataframe_define(self):
        """Full pipeline: generate, compile, use in RDataFrame."""
        # Create simple tree with px, py
        tmpfile = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        tmpfile.close()
        
        try:
            # Create test data
            ROOT.gInterpreter.ProcessLine('''
                void create_test_tree(const char* filename) {
                    TFile f(filename, "RECREATE");
                    TTree tree("tree", "test");
                    double px, py;
                    tree.Branch("px", &px);
                    tree.Branch("py", &py);
                    
                    // Add some test data
                    px = 3.0; py = 4.0; tree.Fill();  // pt = 5
                    px = 6.0; py = 8.0; tree.Fill();  // pt = 10
                    px = 5.0; py = 12.0; tree.Fill(); // pt = 13
                    
                    tree.Write();
                    f.Close();
                }
            ''')
            ROOT.create_test_tree(tmpfile.name)
            
            # Generate pt calculation: sqrt(px**2 + py**2)
            generator = CppCodeGenerator()
            
            px = make_double_var("px")
            py = make_double_var("py")
            two = make_int_const(2)
            
            px_sq = BinaryOpNode(
                op=BinaryOp.POW,
                left=px,
                right=two,
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            py_sq = BinaryOpNode(
                op=BinaryOp.POW,
                left=py,
                right=two,
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            sum_sq = BinaryOpNode(
                op=BinaryOp.ADD,
                left=px_sq,
                right=py_sq,
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            ir = CallNode(
                func="sqrt",
                args=[sum_sq],
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            
            func = generator.generate(ir, "pt")
            
            # Compile
            library = FunctionLibrary()
            library.add(func)
            library.compile(func.name)
            
            # Use in RDataFrame
            rdf = ROOT.RDataFrame("tree", tmpfile.name)
            rdf = rdf.Define("pt", library.get_define_expression(func.name))
            
            # Get results
            pt_values = list(rdf.Take["double"]("pt").GetValue())
            
            # Verify
            assert len(pt_values) == 3
            assert abs(pt_values[0] - 5.0) < 0.001
            assert abs(pt_values[1] - 10.0) < 0.001
            assert abs(pt_values[2] - 13.0) < 0.001
            
        finally:
            os.unlink(tmpfile.name)
    
    def test_rdataframe_filter(self):
        """Use generated function for RDataFrame.Filter()."""
        tmpfile = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        tmpfile.close()
        
        try:
            # Create test data
            ROOT.gInterpreter.ProcessLine('''
                void create_filter_tree(const char* filename) {
                    TFile f(filename, "RECREATE");
                    TTree tree("tree", "test");
                    double x;
                    tree.Branch("x", &x);
                    
                    x = -5.0; tree.Fill();
                    x = 0.0; tree.Fill();
                    x = 5.0; tree.Fill();
                    x = 10.0; tree.Fill();
                    
                    tree.Write();
                    f.Close();
                }
            ''')
            ROOT.create_filter_tree(tmpfile.name)
            
            # Generate filter: x > 0
            generator = CppCodeGenerator()
            
            x = make_double_var("x")
            zero = make_int_const(0)
            ir = BinaryOpNode(
                op=BinaryOp.GT,
                left=x,
                right=zero,
                dtype=IRType(IRTypeKind.Bool),
                rank=0
            )
            
            func = generator.generate(ir, "positive")
            
            library = FunctionLibrary()
            library.add(func)
            library.compile(func.name)
            
            # Use in RDataFrame filter
            rdf = ROOT.RDataFrame("tree", tmpfile.name)
            rdf_filtered = rdf.Filter(library.get_define_expression(func.name))
            
            count = rdf_filtered.Count().GetValue()
            assert count == 2  # Only 5.0 and 10.0
            
        finally:
            os.unlink(tmpfile.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
