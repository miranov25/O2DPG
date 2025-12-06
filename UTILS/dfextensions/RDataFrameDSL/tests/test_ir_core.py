"""
Tests for IR core classes (Phase 1).

This module tests the foundational IR classes:
- ir_types.py: Type system and promotion rules
- ir_nodes.py: IR node construction and traversal
- ir_errors.py: Error handling and collection

These tests do NOT require ROOT - they test the pure Python IR layer.
"""

import pytest
from RDataFrameDSL.ir_types import (
    IRType, IRTypeKind, promote_types, comparison_result_type,
    cpp_type_to_ir, CPP_TO_IR_TYPE, division_result_type
)
from RDataFrameDSL.ir_nodes import (
    IRNode, ConstantNode, VariableNode, 
    UnaryOpNode, UnaryOp,
    BinaryOpNode, BinaryOp,
    TernaryOpNode,
    CallNode, MethodCallNode, PropertyAccessNode,
    SliceNode, SubscriptNode, CollectionIndexNode,
    BroadcastInfo,
    make_constant, make_variable, make_binary_op, make_call
)
from RDataFrameDSL.ir_errors import (
    IRError, IRErrorKind, SourceLocation, 
    ErrorRecoveryMode, ErrorCollector,
    type_mismatch_error, unknown_variable_error, method_not_found_error
)


# =============================================================================
# Test IRType
# =============================================================================

class TestIRType:
    """Tests for IRType class."""
    
    def test_float32_type(self):
        """Float32 should be float and numeric."""
        t = IRType(IRTypeKind.Float32)
        assert t.is_float()
        assert t.is_numeric()
        assert not t.is_int()
        assert not t.is_bool()
        assert str(t) == "float"
    
    def test_float64_type(self):
        """Float64 should be double."""
        t = IRType(IRTypeKind.Float64)
        assert t.is_float()
        assert t.is_numeric()
        assert str(t) == "double"
        assert t.bit_width() == 64
    
    def test_int32_type(self):
        """Int32 should be int."""
        t = IRType(IRTypeKind.Int32)
        assert t.is_int()
        assert t.is_signed_int()
        assert t.is_numeric()
        assert not t.is_float()
        assert str(t) == "int"
    
    def test_int64_type(self):
        """Int64 should be long long."""
        t = IRType(IRTypeKind.Int64)
        assert t.is_int()
        assert t.is_signed_int()
        assert str(t) == "long long"
        assert t.bit_width() == 64
    
    def test_uint32_type(self):
        """UInt32 should be unsigned int."""
        t = IRType(IRTypeKind.UInt32)
        assert t.is_int()
        assert t.is_unsigned_int()
        assert not t.is_signed_int()
        assert str(t) == "unsigned int"
    
    def test_bool_type(self):
        """Bool type checks."""
        t = IRType(IRTypeKind.Bool)
        assert t.is_bool()
        assert not t.is_numeric()  # Bool is not considered numeric
        assert str(t) == "bool"
    
    def test_object_type(self):
        """Object type should include cpp_type."""
        t = IRType(IRTypeKind.Object, "TParticle")
        assert t.is_object()
        assert not t.is_numeric()
        assert str(t) == "TParticle"
        assert t.to_cpp() == "TParticle"
    
    def test_unknown_type(self):
        """Unknown type checks."""
        t = IRType(IRTypeKind.Unknown)
        assert t.is_unknown()
        assert not t.is_numeric()
    
    def test_type_equality(self):
        """Type equality checks."""
        t1 = IRType(IRTypeKind.Float32)
        t2 = IRType(IRTypeKind.Float32)
        t3 = IRType(IRTypeKind.Float64)
        
        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)
    
    def test_object_type_equality(self):
        """Object types with same cpp_type should be equal."""
        t1 = IRType(IRTypeKind.Object, "TParticle")
        t2 = IRType(IRTypeKind.Object, "TParticle")
        t3 = IRType(IRTypeKind.Object, "TLorentzVector")
        
        assert t1 == t2
        assert t1 != t3


class TestTypePromotion:
    """Tests for type promotion rules."""
    
    def test_promote_float_int(self):
        """Float + Int → Float."""
        f = IRType(IRTypeKind.Float32)
        i = IRType(IRTypeKind.Int32)
        result = promote_types(f, i)
        assert result.kind == IRTypeKind.Float32
    
    def test_promote_double_float(self):
        """Double + Float → Double (wider wins)."""
        d = IRType(IRTypeKind.Float64)
        f = IRType(IRTypeKind.Float32)
        result = promote_types(d, f)
        assert result.kind == IRTypeKind.Float64
    
    def test_promote_int32_int64(self):
        """Int32 + Int64 → Int64 (wider wins)."""
        i32 = IRType(IRTypeKind.Int32)
        i64 = IRType(IRTypeKind.Int64)
        result = promote_types(i32, i64)
        assert result.kind == IRTypeKind.Int64
    
    def test_promote_signed_unsigned(self):
        """Signed + Unsigned → Signed (safer)."""
        s = IRType(IRTypeKind.Int32)
        u = IRType(IRTypeKind.UInt32)
        result = promote_types(s, u)
        assert result.kind == IRTypeKind.Int32
    
    def test_promote_unknown_propagates(self):
        """Unknown + anything → Unknown."""
        unknown = IRType(IRTypeKind.Unknown)
        i = IRType(IRTypeKind.Int32)
        result = promote_types(unknown, i)
        assert result.kind == IRTypeKind.Unknown
    
    def test_promote_objects_fail(self):
        """Object + anything → Unknown (can't do arithmetic on objects)."""
        obj = IRType(IRTypeKind.Object, "TParticle")
        i = IRType(IRTypeKind.Int32)
        result = promote_types(obj, i)
        assert result.kind == IRTypeKind.Unknown
    
    def test_comparison_result_type(self):
        """Comparison always returns Bool."""
        result = comparison_result_type()
        assert result.kind == IRTypeKind.Bool


class TestCppTypeMapping:
    """Tests for C++ type mapping."""
    
    def test_cpp_float_to_ir(self):
        """C++ 'float' maps to Float32."""
        t = cpp_type_to_ir("float")
        assert t.kind == IRTypeKind.Float32
    
    def test_cpp_double_to_ir(self):
        """C++ 'double' maps to Float64."""
        t = cpp_type_to_ir("double")
        assert t.kind == IRTypeKind.Float64
    
    def test_root_typedef_to_ir(self):
        """ROOT typedefs should map correctly."""
        assert cpp_type_to_ir("Float_t").kind == IRTypeKind.Float32
        assert cpp_type_to_ir("Double_t").kind == IRTypeKind.Float64
        assert cpp_type_to_ir("Int_t").kind == IRTypeKind.Int32
        assert cpp_type_to_ir("Long64_t").kind == IRTypeKind.Int64
    
    def test_unknown_class_to_object(self):
        """Unknown class names become Object type."""
        t = cpp_type_to_ir("TParticle")
        assert t.kind == IRTypeKind.Object
        assert t.cpp_type == "TParticle"
    
    def test_const_stripped(self):
        """const qualifiers should be stripped."""
        t = cpp_type_to_ir("const float")
        assert t.kind == IRTypeKind.Float32


# =============================================================================
# Test IR Nodes
# =============================================================================

class TestConstantNode:
    """Tests for ConstantNode."""
    
    def test_int_constant(self):
        """Integer constant should have Int64 type."""
        n = ConstantNode(value=42)
        assert n.dtype.kind == IRTypeKind.Int64
        assert n.rank == 0
        assert n.value == 42
        assert n.to_cpp() == "42"
    
    def test_float_constant(self):
        """Float constant should have Float64 type."""
        n = ConstantNode(value=3.14)
        assert n.dtype.kind == IRTypeKind.Float64
        assert n.rank == 0
        assert "3.14" in n.to_cpp()
    
    def test_bool_constant(self):
        """Bool constant should have Bool type."""
        n = ConstantNode(value=True)
        assert n.dtype.kind == IRTypeKind.Bool
        assert n.to_cpp() == "true"
        
        n2 = ConstantNode(value=False)
        assert n2.to_cpp() == "false"
    
    def test_factory_function(self):
        """make_constant should work correctly."""
        n = make_constant(42)
        assert isinstance(n, ConstantNode)
        assert n.value == 42


class TestVariableNode:
    """Tests for VariableNode."""
    
    def test_simple_variable(self):
        """Simple variable node."""
        n = VariableNode(name="px")
        assert n.name == "px"
        assert n.namespace is None
        assert n.full_name() == "px"
        assert n.to_cpp() == "px"
    
    def test_namespaced_variable(self):
        """Variable with namespace (subframe)."""
        n = VariableNode(name="offset", namespace="calib")
        assert n.name == "offset"
        assert n.namespace == "calib"
        assert n.full_name() == "calib.offset"
        assert n.to_cpp() == "calib.offset"
    
    def test_alias_variable(self):
        """Variable marked as alias."""
        n = VariableNode(name="pt", is_alias=True)
        assert n.is_alias
    
    def test_factory_function(self):
        """make_variable should work correctly."""
        n = make_variable("px")
        assert isinstance(n, VariableNode)
        assert n.name == "px"


class TestBinaryOpNode:
    """Tests for BinaryOpNode."""
    
    def test_add_node(self):
        """Addition node structure."""
        left = ConstantNode(value=1)
        right = ConstantNode(value=2)
        op = BinaryOpNode(op=BinaryOp.ADD, left=left, right=right)
        
        assert op.op == BinaryOp.ADD
        assert op.left is left
        assert op.right is right
        assert len(op.children()) == 2
    
    def test_comparison_node(self):
        """Comparison node structure."""
        x = VariableNode(name="x")
        zero = ConstantNode(value=0)
        op = BinaryOpNode(op=BinaryOp.GT, left=x, right=zero)
        
        assert op.op == BinaryOp.GT
        assert op.op.is_comparison()
    
    def test_operator_to_cpp(self):
        """Binary operators should convert to C++ correctly."""
        assert BinaryOp.ADD.to_cpp() == "+"
        assert BinaryOp.AND.to_cpp() == "&&"
        assert BinaryOp.OR.to_cpp() == "||"
        assert BinaryOp.EQ.to_cpp() == "=="
    
    def test_walk_binary_tree(self):
        """Walk should visit all nodes in binary tree."""
        left = ConstantNode(value=1)
        right = ConstantNode(value=2)
        op = BinaryOpNode(op=BinaryOp.ADD, left=left, right=right)
        
        nodes = list(op.walk())
        assert len(nodes) == 3
        assert op in nodes
        assert left in nodes
        assert right in nodes
    
    def test_factory_function(self):
        """make_binary_op should work correctly."""
        left = make_constant(1)
        right = make_constant(2)
        op = make_binary_op(BinaryOp.ADD, left, right)
        assert isinstance(op, BinaryOpNode)


class TestUnaryOpNode:
    """Tests for UnaryOpNode."""
    
    def test_negation_node(self):
        """Negation node structure."""
        x = VariableNode(name="x")
        neg = UnaryOpNode(op=UnaryOp.NEG, operand=x)
        
        assert neg.op == UnaryOp.NEG
        assert neg.operand is x
        assert len(neg.children()) == 1
    
    def test_logical_not(self):
        """Logical not node."""
        x = VariableNode(name="flag")
        not_x = UnaryOpNode(op=UnaryOp.NOT, operand=x)
        
        assert not_x.op == UnaryOp.NOT
        assert UnaryOp.NOT.to_cpp() == "!"


class TestCallNode:
    """Tests for CallNode."""
    
    def test_simple_call(self):
        """Simple function call."""
        x = VariableNode(name="x")
        call = CallNode(func="sqrt", args=[x])
        
        assert call.func == "sqrt"
        assert len(call.args) == 1
        assert len(call.children()) == 1
        assert call.full_cpp_name() == "sqrt"
    
    def test_namespaced_call(self):
        """Namespaced function call."""
        x = VariableNode(name="x")
        call = CallNode(func="Gaus", namespace="TMath", args=[x])
        
        assert call.namespace == "TMath"
        assert call.full_cpp_name() == "TMath::Gaus"
    
    def test_factory_function(self):
        """make_call should work correctly."""
        x = make_variable("x")
        call = make_call("sqrt", [x])
        assert isinstance(call, CallNode)


class TestMethodCallNode:
    """Tests for MethodCallNode."""
    
    def test_method_call(self):
        """Method call node structure."""
        track = VariableNode(name="track")
        call = MethodCallNode(object=track, method_name="getX", args=[])
        
        assert call.object is track
        assert call.method_name == "getX"
        assert len(call.children()) == 1  # Just the object
    
    def test_method_with_args(self):
        """Method call with arguments."""
        track = VariableNode(name="track")
        arg = ConstantNode(value=1)
        call = MethodCallNode(object=track, method_name="getNthCluster", args=[arg])
        
        assert len(call.args) == 1
        assert len(call.children()) == 2  # Object + arg


class TestPropertyAccessNode:
    """Tests for PropertyAccessNode."""
    
    def test_property_access(self):
        """Property access node structure."""
        particle = VariableNode(name="particle")
        prop = PropertyAccessNode(object=particle, property_name="fPdgCode")
        
        assert prop.object is particle
        assert prop.property_name == "fPdgCode"


class TestSliceNode:
    """Tests for SliceNode."""
    
    def test_full_slice(self):
        """Full slice [:]."""
        s = SliceNode()
        assert s.is_full_slice()
        assert not s.has_step()
    
    def test_range_slice(self):
        """Range slice [1:10]."""
        s = SliceNode(
            start=ConstantNode(value=1),
            stop=ConstantNode(value=10)
        )
        assert not s.is_full_slice()
        assert not s.has_step()
        assert len(s.children()) == 2
    
    def test_step_slice(self):
        """Step slice [::2]."""
        s = SliceNode(step=ConstantNode(value=2))
        assert s.has_step()


class TestSubscriptNode:
    """Tests for SubscriptNode."""
    
    def test_scalar_index(self):
        """Scalar indexing arr[5]."""
        arr = VariableNode(name="arr")
        idx = ConstantNode(value=5)
        sub = SubscriptNode(value=arr, indices=[idx])
        
        assert sub.is_scalar_index()
        assert not sub.is_slice()
    
    def test_slice_index(self):
        """Slice indexing arr[:]."""
        arr = VariableNode(name="arr")
        s = SliceNode()
        sub = SubscriptNode(value=arr, indices=[s])
        
        assert not sub.is_scalar_index()
        assert sub.is_slice()
        assert sub.slice_dimensions() == 1
    
    def test_2d_slice(self):
        """2D slicing arr[:, 0]."""
        arr = VariableNode(name="arr")
        s = SliceNode()
        idx = ConstantNode(value=0)
        sub = SubscriptNode(value=arr, indices=[s, idx])
        
        assert sub.is_slice()
        assert sub.slice_dimensions() == 1


class TestTreeTraversal:
    """Tests for IR tree traversal."""
    
    def test_depth_simple(self):
        """Depth of simple expression."""
        n = ConstantNode(value=42)
        assert n.depth() == 1
    
    def test_depth_binary(self):
        """Depth of binary expression."""
        left = ConstantNode(value=1)
        right = ConstantNode(value=2)
        op = BinaryOpNode(op=BinaryOp.ADD, left=left, right=right)
        assert op.depth() == 2
    
    def test_depth_nested(self):
        """Depth of nested expression: (1 + 2) * 3."""
        one = ConstantNode(value=1)
        two = ConstantNode(value=2)
        three = ConstantNode(value=3)
        add = BinaryOpNode(op=BinaryOp.ADD, left=one, right=two)
        mul = BinaryOpNode(op=BinaryOp.MUL, left=add, right=three)
        assert mul.depth() == 3
    
    def test_node_count(self):
        """Node count of expression tree."""
        one = ConstantNode(value=1)
        two = ConstantNode(value=2)
        op = BinaryOpNode(op=BinaryOp.ADD, left=one, right=two)
        assert op.node_count() == 3
    
    def test_collect_variables(self):
        """Collect all variables in expression."""
        x = VariableNode(name="x")
        y = VariableNode(name="y")
        c = ConstantNode(value=1)
        expr = BinaryOpNode(
            op=BinaryOp.ADD,
            left=BinaryOpNode(op=BinaryOp.MUL, left=x, right=c),
            right=y
        )
        
        vars = expr.collect_variables()
        names = [v.name for v in vars]
        assert "x" in names
        assert "y" in names
        assert len(vars) == 2
    
    def test_postorder_walk(self):
        """Post-order traversal visits children before parent."""
        left = ConstantNode(value=1)
        right = ConstantNode(value=2)
        op = BinaryOpNode(op=BinaryOp.ADD, left=left, right=right)
        
        nodes = list(op.walk_postorder())
        # Children should come before parent in post-order
        assert nodes.index(left) < nodes.index(op)
        assert nodes.index(right) < nodes.index(op)


# =============================================================================
# Test IR Errors
# =============================================================================

class TestSourceLocation:
    """Tests for SourceLocation."""
    
    def test_basic_location(self):
        """Basic source location."""
        loc = SourceLocation(expr_name="test", snippet="x + y")
        s = str(loc)
        assert "test" in s
        assert "x + y" in s
    
    def test_with_line_column(self):
        """Location with line and column."""
        loc = SourceLocation(expr_name="test", line=10, column=5)
        s = str(loc)
        assert "line 10" in s
        assert "column 5" in s


class TestIRError:
    """Tests for IRError."""
    
    def test_basic_error(self):
        """Basic error creation."""
        err = IRError(
            kind=IRErrorKind.TYPE_ERROR,
            message="Type mismatch"
        )
        assert err.kind == IRErrorKind.TYPE_ERROR
        assert "Type mismatch" in str(err)
    
    def test_error_with_location(self):
        """Error with source location."""
        loc = SourceLocation(expr_name="test_alias", snippet="x + y")
        err = IRError(
            kind=IRErrorKind.TYPE_ERROR,
            message="Type mismatch",
            source_location=loc
        )
        s = err.format_error()
        assert "test_alias" in s
        assert "x + y" in s
    
    def test_error_with_suggestions(self):
        """Error with suggestions."""
        err = IRError(
            kind=IRErrorKind.REFLECTION_ERROR,
            message="Method 'getpt' not found",
            suggestions=["Did you mean 'GetPt'?", "Check spelling"]
        )
        s = err.format_error()
        assert "Suggestions" in s
        assert "GetPt" in s
    
    def test_error_with_location_method(self):
        """with_location returns new error."""
        err = IRError(IRErrorKind.TYPE_ERROR, "Original")
        loc = SourceLocation(expr_name="test")
        
        new_err = err.with_location(loc)
        assert new_err.source_location is loc
        assert err.source_location is None  # Original unchanged
    
    def test_error_with_suggestion_method(self):
        """with_suggestion returns new error."""
        err = IRError(IRErrorKind.TYPE_ERROR, "Original")
        new_err = err.with_suggestion("Try this")
        
        assert "Try this" in new_err.suggestions
        assert len(err.suggestions) == 0  # Original unchanged


class TestErrorHelperFunctions:
    """Tests for error helper functions."""
    
    def test_type_mismatch_error(self):
        """type_mismatch_error helper."""
        err = type_mismatch_error("int", "float", "in addition")
        assert err.kind == IRErrorKind.TYPE_ERROR
        assert "int" in err.message
        assert "float" in err.message
    
    def test_unknown_variable_error(self):
        """unknown_variable_error helper."""
        err = unknown_variable_error("xyz", similar_names=["x", "xy", "xz"])
        assert err.kind == IRErrorKind.TYPE_ERROR
        assert "xyz" in err.message
        assert len(err.suggestions) > 0
    
    def test_method_not_found_error(self):
        """method_not_found_error helper."""
        err = method_not_found_error(
            "TParticle", "getpt",
            similar_methods=["GetPt", "Pt"]
        )
        assert err.kind == IRErrorKind.REFLECTION_ERROR
        assert "TParticle" in err.message
        assert "getpt" in err.message


class TestErrorCollector:
    """Tests for ErrorCollector."""
    
    def test_empty_collector(self):
        """Empty collector has no errors."""
        collector = ErrorCollector()
        assert not collector.has_errors()
        assert collector.error_count() == 0
        assert "No errors" in collector.format_report()
    
    def test_add_error(self):
        """Adding errors."""
        collector = ErrorCollector()
        err = IRError(IRErrorKind.TYPE_ERROR, "Bad type")
        collector.add(err)
        
        assert collector.has_errors()
        assert collector.error_count() == 1
    
    def test_failed_aliases_tracked(self):
        """Failed aliases should be tracked."""
        collector = ErrorCollector()
        loc = SourceLocation(expr_name="bad_alias")
        err = IRError(IRErrorKind.TYPE_ERROR, "Bad type", source_location=loc)
        collector.add(err)
        
        assert "bad_alias" in collector.failed_aliases
    
    def test_fail_all_mode(self):
        """FAIL_ALL mode stops on first error."""
        collector = ErrorCollector(ErrorRecoveryMode.FAIL_ALL)
        
        # Before any error
        assert collector.should_process("alias1", set())
        
        # Add error
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Error"))
        
        # After error, nothing should process
        assert not collector.should_process("alias2", set())
    
    def test_skip_continue_mode(self):
        """SKIP_CONTINUE mode skips failed, continues others."""
        collector = ErrorCollector(ErrorRecoveryMode.SKIP_CONTINUE)
        
        loc = SourceLocation(expr_name="bad_alias")
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Error", source_location=loc))
        
        # Failed alias shouldn't process
        assert not collector.should_process("bad_alias", set())
        # Other alias should process
        assert collector.should_process("good_alias", set())
    
    def test_fail_chain_mode(self):
        """FAIL_CHAIN mode fails dependent aliases."""
        collector = ErrorCollector(ErrorRecoveryMode.FAIL_CHAIN)
        
        loc = SourceLocation(expr_name="base_alias")
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Error", source_location=loc))
        
        # Independent alias should process
        assert collector.should_process("independent", set())
        
        # Dependent alias should NOT process
        assert not collector.should_process("derived", {"base_alias"})
    
    def test_clear(self):
        """Clear should reset collector."""
        collector = ErrorCollector()
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Error"))
        collector.clear()
        
        assert not collector.has_errors()
        assert collector.error_count() == 0
    
    def test_raise_if_errors(self):
        """raise_if_errors should raise first error."""
        collector = ErrorCollector()
        err = IRError(IRErrorKind.TYPE_ERROR, "First error")
        collector.add(err)
        collector.add(IRError(IRErrorKind.PARSE_ERROR, "Second error"))
        
        with pytest.raises(IRError) as exc_info:
            collector.raise_if_errors()
        assert exc_info.value.message == "First error"
    
    def test_get_errors_by_kind(self):
        """Filter errors by kind."""
        collector = ErrorCollector()
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Type error 1"))
        collector.add(IRError(IRErrorKind.PARSE_ERROR, "Parse error"))
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Type error 2"))
        
        type_errors = collector.get_errors_by_kind(IRErrorKind.TYPE_ERROR)
        assert len(type_errors) == 2
    
    def test_format_report(self):
        """Format report with multiple errors."""
        collector = ErrorCollector()
        collector.add(IRError(IRErrorKind.TYPE_ERROR, "Error 1"))
        collector.add(IRError(IRErrorKind.PARSE_ERROR, "Error 2"))
        
        report = collector.format_report()
        assert "2 error(s)" in report
        assert "Error 1" in report
        assert "Error 2" in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestIRConstruction:
    """Integration tests for building complete IR trees."""
    
    def test_build_sqrt_px2_py2(self):
        """Build IR for sqrt(px**2 + py**2)."""
        px = VariableNode(name="px", dtype=IRType(IRTypeKind.Float64))
        py = VariableNode(name="py", dtype=IRType(IRTypeKind.Float64))
        two = ConstantNode(value=2)
        
        px_squared = BinaryOpNode(op=BinaryOp.POW, left=px, right=two)
        py_squared = BinaryOpNode(op=BinaryOp.POW, left=py, right=two)
        sum_squares = BinaryOpNode(op=BinaryOp.ADD, left=px_squared, right=py_squared)
        sqrt_call = CallNode(func="sqrt", args=[sum_squares])
        
        # Check structure
        assert sqrt_call.func == "sqrt"
        assert isinstance(sqrt_call.args[0], BinaryOpNode)
        
        # Check traversal
        vars = sqrt_call.collect_variables()
        assert len(vars) == 2
        
        # Check depth: sqrt -> add -> pow -> variable = 4 levels
        assert sqrt_call.depth() == 4
    
    def test_build_method_chain(self):
        """Build IR for track.getCluster(0).getX()."""
        track = VariableNode(name="track", dtype=IRType(IRTypeKind.Object, "Track"))
        zero = ConstantNode(value=0)
        
        get_cluster = MethodCallNode(
            object=track,
            method_name="getCluster",
            args=[zero]
        )
        get_x = MethodCallNode(
            object=get_cluster,
            method_name="getX",
            args=[]
        )
        
        # Check structure
        assert get_x.method_name == "getX"
        assert get_x.object is get_cluster
        assert get_cluster.object is track
        
        # Check traversal finds track
        vars = get_x.collect_variables()
        assert len(vars) == 1
        assert vars[0].name == "track"
    
    def test_build_conditional(self):
        """Build IR for x if x > 0 else -x."""
        x = VariableNode(name="x", dtype=IRType(IRTypeKind.Float64))
        zero = ConstantNode(value=0)
        
        condition = BinaryOpNode(op=BinaryOp.GT, left=x, right=zero)
        neg_x = UnaryOpNode(op=UnaryOp.NEG, operand=x)
        
        ternary = TernaryOpNode(
            condition=condition,
            if_true=x,
            if_false=neg_x
        )
        
        # Check structure
        assert ternary.condition is condition
        assert ternary.if_true is x
        assert isinstance(ternary.if_false, UnaryOpNode)
        
        # Variable appears 3 times (condition, true branch, false branch)
        vars = ternary.collect_variables()
        assert len(vars) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
