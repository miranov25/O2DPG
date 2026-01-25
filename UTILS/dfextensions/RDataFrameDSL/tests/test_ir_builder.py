"""
Tests for IRBuilder (Phase 3).

Tests the conversion of Python AST to IR nodes with type inference.
"""

import pytest
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_nodes import (
    ConstantNode, VariableNode, BinaryOpNode, UnaryOpNode,
    CallNode, MethodCallNode, PropertyAccessNode,
    SubscriptNode, SliceNode, TernaryOpNode,
    BinaryOp, UnaryOp, RVecSliceNode, SliceKind
)
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.type_inferrer import TypeInferrer
from RDataFrameDSL.ir_builder import IRBuilder, BuildContext


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_schema():
    """Schema with basic numeric columns."""
    return {
        "columns": {
            "px": {"dtype": "float", "rank": 0},
            "py": {"dtype": "float", "rank": 0},
            "pz": {"dtype": "float", "rank": 0},
            "energy": {"dtype": "double", "rank": 0},
            "n": {"dtype": "int", "rank": 0},
            "flag": {"dtype": "bool", "rank": 0},
        }
    }


@pytest.fixture
def vector_schema():
    """Schema with vector columns."""
    return {
        "columns": {
            "track_pt": {"dtype": "float", "rank": 1, "is_jagged": True},
            "track_eta": {"dtype": "float", "rank": 1, "is_jagged": True},
            "cluster_x": {"dtype": "double", "rank": 1, "is_jagged": False},
            "hit_idx": {"dtype": "int", "rank": 1, "is_jagged": True},
        }
    }


@pytest.fixture
def object_schema():
    """Schema with object columns."""
    return {
        "columns": {
            "track": {"dtype": "TParticle", "rank": 0},
            "tracks": {"dtype": "TParticle", "rank": 1, "is_jagged": True},
        }
    }


@pytest.fixture
def basic_builder(basic_schema):
    """Builder with basic schema."""
    inferrer = TypeInferrer.from_schema(basic_schema)
    return IRBuilder(inferrer)


@pytest.fixture
def vector_builder(vector_schema):
    """Builder with vector schema."""
    inferrer = TypeInferrer.from_schema(vector_schema)
    return IRBuilder(inferrer)


@pytest.fixture
def object_builder(object_schema):
    """Builder with object schema."""
    inferrer = TypeInferrer.from_schema(object_schema)
    return IRBuilder(inferrer)


# =============================================================================
# Constant Tests
# =============================================================================

class TestConstants:
    """Tests for literal constant handling."""
    
    def test_int_constant(self, basic_builder):
        """Integer constant."""
        node = basic_builder.build("42")
        assert isinstance(node, ConstantNode)
        assert node.value == 42
        assert node.dtype.kind == IRTypeKind.Int64
    
    def test_float_constant(self, basic_builder):
        """Float constant."""
        node = basic_builder.build("3.14")
        assert isinstance(node, ConstantNode)
        assert node.value == 3.14
        assert node.dtype.kind == IRTypeKind.Float64
    
    def test_bool_true(self, basic_builder):
        """Boolean True constant."""
        node = basic_builder.build("True")
        assert isinstance(node, ConstantNode)
        assert node.value is True
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_bool_false(self, basic_builder):
        """Boolean False constant."""
        node = basic_builder.build("False")
        assert isinstance(node, ConstantNode)
        assert node.value is False
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_negative_int(self, basic_builder):
        """Negative integer - constant folding produces ConstantNode with negative value."""
        node = basic_builder.build("-42")
        # Phase 6.9: Negative literals are now folded to ConstantNode(-42)
        assert isinstance(node, ConstantNode)
        assert node.value == -42
        assert node.dtype.kind == IRTypeKind.Int64


# =============================================================================
# Variable Tests
# =============================================================================

class TestVariables:
    """Tests for variable reference handling."""
    
    def test_simple_variable(self, basic_builder):
        """Simple variable reference."""
        node = basic_builder.build("px")
        assert isinstance(node, VariableNode)
        assert node.name == "px"
        assert node.dtype.kind == IRTypeKind.Float32
        assert node.rank == 0
    
    def test_vector_variable(self, vector_builder):
        """Vector variable reference."""
        node = vector_builder.build("track_pt")
        assert isinstance(node, VariableNode)
        assert node.name == "track_pt"
        assert node.dtype.kind == IRTypeKind.Float32
        assert node.rank == 1
        assert node.is_jagged
    
    def test_unknown_variable_error(self, basic_builder):
        """Unknown variable raises error."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("unknown_var")
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR
    
    def test_unknown_variable_suggestions(self, basic_builder):
        """Unknown variable error includes suggestions."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("p")  # Should suggest px, py, pz
        assert len(exc_info.value.suggestions) > 0


# =============================================================================
# Binary Operation Tests
# =============================================================================

class TestBinaryOps:
    """Tests for binary operations."""
    
    def test_addition(self, basic_builder):
        """Addition of two variables."""
        node = basic_builder.build("px + py")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.ADD
        assert node.dtype.kind == IRTypeKind.Float32
    
    def test_subtraction(self, basic_builder):
        """Subtraction."""
        node = basic_builder.build("px - py")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.SUB
    
    def test_multiplication(self, basic_builder):
        """Multiplication."""
        node = basic_builder.build("px * py")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.MUL
    
    def test_division(self, basic_builder):
        """Division promotes to float."""
        node = basic_builder.build("n / 2")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.DIV
        assert node.dtype.is_float()
    
    def test_floor_division(self, basic_builder):
        """Floor division."""
        node = basic_builder.build("n // 2")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.FLOORDIV
    
    def test_modulo(self, basic_builder):
        """Modulo operation."""
        node = basic_builder.build("n % 2")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.MOD
    
    def test_power(self, basic_builder):
        """Power operation."""
        node = basic_builder.build("px ** 2")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.POW
    
    def test_type_promotion_float_int(self, basic_builder):
        """Float + int promotes to float."""
        node = basic_builder.build("px + n")
        assert node.dtype.is_float()
    
    def test_type_promotion_double_float(self, basic_builder):
        """Double + float promotes to double."""
        node = basic_builder.build("energy + px")
        assert node.dtype.kind == IRTypeKind.Float64
    
    def test_nested_operations(self, basic_builder):
        """Nested binary operations."""
        node = basic_builder.build("px + py * pz")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.ADD
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.op == BinaryOp.MUL
    
    def test_parenthesized(self, basic_builder):
        """Parenthesized expression."""
        node = basic_builder.build("(px + py) * pz")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.MUL
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.op == BinaryOp.ADD


class TestBroadcasting:
    """Tests for rank broadcasting in binary operations."""
    
    def test_scalar_vector_broadcast(self, vector_builder):
        """Scalar + vector broadcasts to vector."""
        # Add scalar column to schema
        vector_builder.inferrer._variables["scale"] = \
            vector_builder.inferrer._variables["track_pt"].__class__(
                name="scale", dtype=IRType(IRTypeKind.Float32),
                rank=0, source="test"
            )
        
        node = vector_builder.build("track_pt * scale")
        assert node.rank == 1
        assert node.is_jagged
    
    def test_vector_vector_same_rank(self, vector_builder):
        """Vector + vector preserves rank."""
        node = vector_builder.build("track_pt + track_eta")
        assert node.rank == 1


# =============================================================================
# Comparison Tests
# =============================================================================

class TestComparisons:
    """Tests for comparison operations."""
    
    def test_less_than(self, basic_builder):
        """Less than comparison."""
        node = basic_builder.build("px < py")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.LT
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_less_equal(self, basic_builder):
        """Less than or equal."""
        node = basic_builder.build("px <= py")
        assert node.op == BinaryOp.LE
    
    def test_greater_than(self, basic_builder):
        """Greater than."""
        node = basic_builder.build("px > py")
        assert node.op == BinaryOp.GT
    
    def test_greater_equal(self, basic_builder):
        """Greater than or equal."""
        node = basic_builder.build("px >= py")
        assert node.op == BinaryOp.GE
    
    def test_equal(self, basic_builder):
        """Equality comparison."""
        node = basic_builder.build("px == py")
        assert node.op == BinaryOp.EQ
    
    def test_not_equal(self, basic_builder):
        """Not equal comparison."""
        node = basic_builder.build("px != py")
        assert node.op == BinaryOp.NE
    
    def test_chained_comparison(self, basic_builder):
        """Chained comparison: a < b < c."""
        node = basic_builder.build("px < py < pz")
        # Should become (px < py) and (py < pz)
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.AND
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.op == BinaryOp.LT
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.op == BinaryOp.LT


# =============================================================================
# Boolean Operation Tests
# =============================================================================

class TestBooleanOps:
    """Tests for boolean operations."""
    
    def test_logical_and(self, basic_builder):
        """Logical AND."""
        node = basic_builder.build("flag and (px > 0)")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.AND
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_logical_or(self, basic_builder):
        """Logical OR."""
        node = basic_builder.build("flag or (px > 0)")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.OR
    
    def test_logical_not(self, basic_builder):
        """Logical NOT."""
        node = basic_builder.build("not flag")
        assert isinstance(node, UnaryOpNode)
        assert node.op == UnaryOp.NOT
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_combined_boolean(self, basic_builder):
        """Combined boolean expression."""
        node = basic_builder.build("(px > 0) and (py > 0) or flag")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.OR


# =============================================================================
# Unary Operation Tests
# =============================================================================

class TestUnaryOps:
    """Tests for unary operations."""
    
    def test_negation(self, basic_builder):
        """Unary negation."""
        node = basic_builder.build("-px")
        assert isinstance(node, UnaryOpNode)
        assert node.op == UnaryOp.NEG
        assert node.dtype.kind == IRTypeKind.Float32
    
    def test_positive(self, basic_builder):
        """Unary positive."""
        node = basic_builder.build("+px")
        assert isinstance(node, UnaryOpNode)
        assert node.op == UnaryOp.POS
    
    def test_bitwise_not(self, basic_builder):
        """Bitwise NOT on integer."""
        node = basic_builder.build("~n")
        assert isinstance(node, UnaryOpNode)
        assert node.op == UnaryOp.BITNOT
    
    def test_bitwise_not_requires_int(self, basic_builder):
        """Bitwise NOT on float raises error."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("~px")
        assert exc_info.value.kind == IRErrorKind.TYPE_ERROR


# =============================================================================
# Function Call Tests
# =============================================================================

class TestFunctionCalls:
    """Tests for function call handling."""
    
    def test_sqrt(self, basic_builder):
        """sqrt function."""
        node = basic_builder.build("sqrt(px)")
        assert isinstance(node, CallNode)
        assert node.func == "sqrt"
        assert node.cpp_name == "std::sqrt"
        assert node.dtype.kind == IRTypeKind.Float64
    
    def test_abs(self, basic_builder):
        """abs function preserves type."""
        node = basic_builder.build("abs(px)")
        assert isinstance(node, CallNode)
        assert node.func == "abs"
        assert node.dtype.kind == IRTypeKind.Float32  # Same as input
    
    def test_trig_functions(self, basic_builder):
        """Trigonometric functions."""
        for func in ["sin", "cos", "tan", "asin", "acos", "atan"]:
            node = basic_builder.build(f"{func}(px)")
            assert isinstance(node, CallNode)
            assert node.func == func
            assert node.dtype.kind == IRTypeKind.Float64
    
    def test_pow_function(self, basic_builder):
        """pow function with two arguments."""
        node = basic_builder.build("pow(px, 2)")
        assert isinstance(node, CallNode)
        assert node.func == "pow"
        assert len(node.args) == 2
    
    def test_min_max(self, basic_builder):
        """min/max functions."""
        node = basic_builder.build("min(px, py)")
        assert isinstance(node, CallNode)
        assert node.func == "min"
        assert len(node.args) == 2
    
    def test_tmath_gaus(self, basic_builder):
        """TMath.Gaus function."""
        node = basic_builder.build("TMath.Gaus(px, 0, 1)")
        assert isinstance(node, CallNode)
        assert node.cpp_name == "TMath::Gaus"
        assert node.dtype.kind == IRTypeKind.Float64
    
    def test_unknown_function_error(self, basic_builder):
        """Unknown function raises error."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("unknown_func(px)")
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
    
    def test_nested_calls(self, basic_builder):
        """Nested function calls."""
        node = basic_builder.build("sqrt(abs(px))")
        assert isinstance(node, CallNode)
        assert node.func == "sqrt"
        assert isinstance(node.args[0], CallNode)
        assert node.args[0].func == "abs"
    
    def test_function_with_expression_arg(self, basic_builder):
        """Function with expression argument."""
        node = basic_builder.build("sqrt(px**2 + py**2)")
        assert isinstance(node, CallNode)
        assert node.func == "sqrt"
        assert isinstance(node.args[0], BinaryOpNode)


class TestCustomFunctions:
    """Tests for custom function registration."""
    
    def test_register_custom_function(self, basic_builder):
        """Register and use custom function."""
        # Phase 13.5.C: param_types required for overload resolution
        # Note: px is defined as 'float' (Float32) in basic_schema
        basic_builder.register_function(
            "myFunc", "MyNamespace::myFunc", 
            return_type=IRTypeKind.Float64,
            param_types=[{'name': 'x', 'cpp_type': 'float', 'rank': 0, 'ir_kind': IRTypeKind.Float32}]
        )
        
        node = basic_builder.build("myFunc(px)")
        assert isinstance(node, CallNode)
        assert node.cpp_name == "MyNamespace::myFunc"
        assert node.dtype.kind == IRTypeKind.Float64


# =============================================================================
# Method Call Tests
# =============================================================================

class TestMethodCalls:
    """Tests for method call handling."""
    
    def test_object_method(self, object_builder):
        """Method call on object."""
        node = object_builder.build("track.GetPx()")
        assert isinstance(node, MethodCallNode)
        assert node.method_name == "GetPx"
        assert node.class_name == "TParticle"
    
    def test_method_with_args(self, object_builder):
        """Method call with arguments."""
        node = object_builder.build("track.Distance(0, 0, 0)")
        assert isinstance(node, MethodCallNode)
        assert len(node.args) == 3
    
    def test_rvec_size_method(self, vector_builder):
        """RVec.size() method."""
        node = vector_builder.build("track_pt.size()")
        assert isinstance(node, MethodCallNode)
        assert node.method_name == "size"
        assert node.dtype.kind == IRTypeKind.UInt64
        assert node.rank == 0  # size() returns scalar
    
    def test_rvec_empty_method(self, vector_builder):
        """RVec.empty() method."""
        node = vector_builder.build("track_pt.empty()")
        assert isinstance(node, MethodCallNode)
        assert node.method_name == "empty"
        assert node.dtype.kind == IRTypeKind.Bool


# =============================================================================
# Attribute Access Tests
# =============================================================================

class TestAttributeAccess:
    """Tests for attribute/property access."""
    
    def test_object_property(self, object_builder):
        """Property access on object."""
        node = object_builder.build("track.mPx")
        assert isinstance(node, PropertyAccessNode)
        assert node.property_name == "mPx"
        assert node.class_name == "TParticle"
    
    def test_tmath_constants(self, basic_builder):
        """TMath constants like TMath.Pi."""
        node = basic_builder.build("TMath.Pi")
        assert isinstance(node, ConstantNode)
        assert abs(node.value - 3.14159265358979323846) < 1e-10
    
    def test_attribute_on_non_object_error(self, basic_builder):
        """Attribute access on non-object raises error."""
        with pytest.raises(IRError):
            basic_builder.build("px.something")


# =============================================================================
# Subscript Tests
# =============================================================================

class TestSubscripts:
    """Tests for subscript operations."""
    
    def test_scalar_index(self, vector_builder):
        """Scalar indexing: arr[0]."""
        node = vector_builder.build("track_pt[0]")
        assert isinstance(node, SubscriptNode)
        assert node.rank == 0  # Reduced from 1
        assert isinstance(node.indices[0], ConstantNode)
    
    def test_variable_index(self, vector_builder):
        """Variable indexing: arr[i]."""
        vector_builder.inferrer._variables["i"] = \
            vector_builder.inferrer._variables["track_pt"].__class__(
                name="i", dtype=IRType(IRTypeKind.Int32),
                rank=0, source="test"
            )
        
        node = vector_builder.build("track_pt[i]")
        assert isinstance(node, SubscriptNode)
        assert isinstance(node.indices[0], VariableNode)
    
    def test_slice_full(self, vector_builder):
        """Full slice: arr[:] - Phase 7: returns RVecSliceNode."""
        node = vector_builder.build("track_pt[:]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.FROM_INDEX
        assert node.start is None  # Full slice has no start
        assert node.stop is None
        assert node.rank == 1  # Preserved
    
    def test_slice_range(self, vector_builder):
        """Range slice: arr[1:3] - Phase 7: returns RVecSliceNode."""
        node = vector_builder.build("track_pt[1:3]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.RANGE
        assert node.start.value == 1
        assert node.stop.value == 3
        assert node.rank == 1
    
    def test_negative_index(self, vector_builder):
        """Negative index: arr[-1] - constant folding produces ConstantNode(-1)."""
        node = vector_builder.build("track_pt[-1]")
        assert isinstance(node, SubscriptNode)
        # Phase 6.9: Negative indices are now folded to ConstantNode(-1)
        assert isinstance(node.indices[0], ConstantNode)
        assert node.indices[0].value == -1
        assert node.rank == 0


# =============================================================================
# Conditional Expression Tests
# =============================================================================

class TestConditionals:
    """Tests for conditional expressions."""
    
    def test_simple_conditional(self, basic_builder):
        """Simple conditional: x if cond else y."""
        node = basic_builder.build("px if flag else py")
        assert isinstance(node, TernaryOpNode)
        assert isinstance(node.condition, VariableNode)
        assert node.dtype.kind == IRTypeKind.Float32
    
    def test_conditional_with_comparison(self, basic_builder):
        """Conditional with comparison."""
        node = basic_builder.build("px if px > 0 else -px")
        assert isinstance(node, TernaryOpNode)
        assert isinstance(node.condition, BinaryOpNode)
    
    def test_conditional_type_promotion(self, basic_builder):
        """Conditional promotes branch types."""
        node = basic_builder.build("px if flag else n")
        # Float32 and Int32 should promote
        assert node.dtype.is_float()
    
    def test_nested_conditional(self, basic_builder):
        """Nested conditionals."""
        node = basic_builder.build("px if flag else (py if n > 0 else pz)")
        assert isinstance(node, TernaryOpNode)
        assert isinstance(node.if_false, TernaryOpNode)


# =============================================================================
# Complex Expression Tests
# =============================================================================

class TestComplexExpressions:
    """Tests for complex combined expressions."""
    
    def test_pt_calculation(self, basic_builder):
        """Classic pt calculation: sqrt(px**2 + py**2)."""
        node = basic_builder.build("sqrt(px**2 + py**2)")
        assert isinstance(node, CallNode)
        assert node.func == "sqrt"
        # Verify structure
        arg = node.args[0]
        assert isinstance(arg, BinaryOpNode)
        assert arg.op == BinaryOp.ADD
    
    def test_eta_calculation(self, basic_builder):
        """Eta calculation with atanh."""
        # Phase 13.5.C: param_types required for overload resolution
        basic_builder.register_function(
            "atanh", "std::atanh", 
            return_type=IRTypeKind.Float64,
            param_types=[{'name': 'x', 'cpp_type': 'double', 'rank': 0, 'ir_kind': IRTypeKind.Float64}]
        )
        
        # First add pt alias
        basic_builder.inferrer.register_alias("pt", IRType(IRTypeKind.Float64))
        
        node = basic_builder.build("atanh(pz / energy)")
        assert isinstance(node, CallNode)
    
    def test_mass_formula(self, basic_builder):
        """Mass formula: sqrt(E² - p²)."""
        node = basic_builder.build("sqrt(energy**2 - px**2 - py**2 - pz**2)")
        assert isinstance(node, CallNode)
        assert node.func == "sqrt"
    
    def test_boolean_cut(self, basic_builder):
        """Complex boolean cut."""
        node = basic_builder.build("(px > 0) and (py > 0) and (energy > 10)")
        assert isinstance(node, BinaryOpNode)
        assert node.op == BinaryOp.AND
        assert node.dtype.kind == IRTypeKind.Bool
    
    def test_conditional_with_math(self, basic_builder):
        """Conditional with math operations."""
        node = basic_builder.build("abs(px) if px < 0 else px")
        assert isinstance(node, TernaryOpNode)
        assert isinstance(node.if_true, CallNode)


# =============================================================================
# Multiple Alias Tests
# =============================================================================

class TestMultipleAliases:
    """Tests for building multiple interdependent aliases."""
    
    def test_build_multiple_independent(self, basic_builder):
        """Build multiple independent aliases."""
        aliases = {
            "pt": "sqrt(px**2 + py**2)",
            "p": "sqrt(px**2 + py**2 + pz**2)",
        }
        
        results = basic_builder.build_multiple(aliases)
        
        assert "pt" in results
        assert "p" in results
    
    def test_build_multiple_dependent(self, basic_builder):
        """Build aliases that depend on each other."""
        aliases = {
            "pt": "sqrt(px**2 + py**2)",
            "pt2": "pt * 2",  # Depends on pt
        }
        
        results = basic_builder.build_multiple(aliases)
        
        assert "pt" in results
        assert "pt2" in results
        # pt2 should use pt
        assert basic_builder.inferrer.has_variable("pt")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_syntax_error(self, basic_builder):
        """Syntax error in expression."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("px +")
        assert exc_info.value.kind == IRErrorKind.PARSE_ERROR
    
    def test_type_mismatch_in_conditional(self, basic_builder):
        """Type mismatch in conditional branches."""
        # Object + float should fail to promote
        with pytest.raises(IRError):
            basic_builder.build("track if flag else px", )
    
    def test_error_has_location(self, basic_builder):
        """Error includes source location."""
        with pytest.raises(IRError) as exc_info:
            basic_builder.build("unknown_var", alias_name="test_alias")
        
        assert exc_info.value.source_location is not None
        assert exc_info.value.source_location.expr_name == "test_alias"


# =============================================================================
# BuildContext Tests
# =============================================================================

class TestBuildContext:
    """Tests for BuildContext."""
    
    def test_basic_context(self):
        """Basic context creation."""
        ctx = BuildContext(alias_name="test", expression_text="px + py")
        assert ctx.alias_name == "test"
        assert ctx.expression_text == "px + py"
    
    def test_with_namespace(self):
        """Context with namespace."""
        ctx = BuildContext(alias_name="test")
        new_ctx = ctx.with_namespace("tracks")
        
        assert new_ctx.namespace == "tracks"
        assert new_ctx.alias_name == "test"  # Preserved
    
    def test_with_subscript(self):
        """Context in subscript mode."""
        ctx = BuildContext()
        new_ctx = ctx.with_subscript()
        
        assert new_ctx.in_subscript
        assert not ctx.in_subscript  # Original unchanged


# =============================================================================
# Phase 7: Slice Parsing Tests
# =============================================================================

class TestSliceParsing:
    """Tests for Phase 7 slice parsing."""
    
    @pytest.fixture
    def slice_builder(self):
        """Builder with RVec column for slice tests."""
        schema = {
            "columns": {
                "pt": {"dtype": "double", "rank": 1, "cpp_type": "RVec<double>"},
                "mask": {"dtype": "bool", "rank": 1, "cpp_type": "RVec<bool>"},
            }
        }
        inferrer = TypeInferrer.from_schema(schema)
        return IRBuilder(inferrer)
    
    def test_slice_first_n(self, slice_builder):
        """[:3] parses to RVecSliceNode with FIRST_N."""
        node = slice_builder.build("pt[:3]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.FIRST_N
        assert node.stop.value == 3
        assert node.start is None
        assert node.step is None
    
    def test_slice_last_n(self, slice_builder):
        """[-3:] parses to RVecSliceNode with LAST_N."""
        node = slice_builder.build("pt[-3:]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.LAST_N
        assert node.start.value == -3
        assert node.stop is None
    
    def test_slice_from_index(self, slice_builder):
        """[2:] parses to RVecSliceNode with FROM_INDEX."""
        node = slice_builder.build("pt[2:]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.FROM_INDEX
        assert node.start.value == 2
        assert node.stop is None
    
    def test_slice_range(self, slice_builder):
        """[1:3] parses to RVecSliceNode with RANGE."""
        node = slice_builder.build("pt[1:3]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.RANGE
        assert node.start.value == 1
        assert node.stop.value == 3
    
    def test_slice_step(self, slice_builder):
        """[::2] parses to RVecSliceNode with STEP."""
        node = slice_builder.build("pt[::2]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.STEP
        assert node.step.value == 2
    
    def test_slice_step_with_start(self, slice_builder):
        """[1::2] parses to RVecSliceNode with STEP."""
        node = slice_builder.build("pt[1::2]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.STEP
        assert node.start.value == 1
        assert node.step.value == 2
    
    def test_slice_step_with_range(self, slice_builder):
        """[1:5:2] parses to RVecSliceNode with STEP."""
        node = slice_builder.build("pt[1:5:2]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.STEP
        assert node.start.value == 1
        assert node.stop.value == 5
        assert node.step.value == 2
    
    def test_slice_reverse(self, slice_builder):
        """[::-1] parses to RVecSliceNode with REVERSE."""
        node = slice_builder.build("pt[::-1]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.REVERSE
        assert node.step.value == -1
    
    def test_slice_step_zero_error(self, slice_builder):
        """[::0] raises IRError."""
        with pytest.raises(IRError) as exc:
            slice_builder.build("pt[::0]")
        assert "step cannot be zero" in str(exc.value).lower()
    
    def test_slice_mixed_negative_supported(self, slice_builder):
        """[-3:-1] now supported with RANGE_NEG (Phase 13.6.G)."""
        node = slice_builder.build("pt[-3:-1]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.RANGE_NEG
    
    def test_boolean_mask_comparison(self, slice_builder):
        """[pt > 1.0] parses to RVecSliceNode with BOOLEAN."""
        node = slice_builder.build("pt[pt > 1.0]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.BOOLEAN
    
    def test_boolean_mask_variable(self, slice_builder):
        """[mask] parses to RVecSliceNode with BOOLEAN."""
        node = slice_builder.build("pt[mask]")
        assert isinstance(node, RVecSliceNode)
        assert node.slice_kind == SliceKind.BOOLEAN
    
    def test_slice_preserves_dtype(self, slice_builder):
        """Slicing preserves element type."""
        node = slice_builder.build("pt[:3]")
        assert node.rank == 1  # Still RVec
        # dtype should be double (the element type)
        assert str(node.dtype) == "double" or "Float64" in str(node.dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
