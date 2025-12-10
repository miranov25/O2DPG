"""
C++ code generation backend for RDataFrame DSL.

This module generates C++ helper functions from IR trees for scalar and vector
expressions. The generated functions can be compiled via ROOT's gInterpreter
and used in RDataFrame.Define() calls.

Architecture:
    IR Tree → CppCodeGenerator → GeneratedFunction → FunctionLibrary → gInterpreter.Declare()

Example:
    >>> generator = CppCodeGenerator(type_inferrer)
    >>> ir = builder.build("sqrt(px**2 + py**2)")
    >>> func = generator.generate(ir, "pt")
    >>> 
    >>> library = FunctionLibrary()
    >>> library.add(func)
    >>> library.compile("pt")
    >>> 
    >>> rdf.Define("pt", "alias_pt(px, py)")

Phase 5 Scope (Scalars):
- Arithmetic operations (+, -, *, /, %, **)
- Comparisons (<, <=, >, >=, ==, !=)
- Logical operations (and, or, not)
- Bitwise operations (&, |, ^, ~) for int/bool
- Function calls (sqrt, sin, cos, abs, TMath::*)
- Conditionals (ternary)
- Constants (numeric, boolean) and variables

Phase 6a Scope (Objects):
- Method calls on objects (particle.Px())
- Property access on objects (vec.fX)

Phase 6b Scope (RVec Operations):
- RVec arithmetic (pt * 1.5, px + py)
- RVec comparisons (pt > 1.0)
- Vectorized math via ADL (sqrt(pt))
- Simple indexing (pt[0], pt[i])
- Negative literal indexing (pt[-1])
- Safe bounds checking (default ON, returns NaN)
- RVec methods: size(), empty(), at()

Phase 6c Scope (Private/Protected Member Access):
- Detect member access level (public/protected/private) via TClass
- Direct access for public members (obj.member)
- Reflection-based access for protected/private members via GetOffset()
- IsBasic() validation (reject non-basic members)
- IsaPointer() validation (reject pointer members)
- Thread-safe via C++11 magic statics
"""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from .ir_types import IRType, IRTypeKind, IR_TO_CPP_TYPE
from .ir_nodes import (
    IRNode, ConstantNode, VariableNode, UnaryOpNode, BinaryOpNode,
    TernaryOpNode, CallNode, MethodCallNode, PropertyAccessNode,
    MethodBroadcastNode, PropertyBroadcastNode,  # Phase 8
    SubscriptNode, SliceNode, UnaryOp, BinaryOp, RVecSliceNode, SliceKind
)
from .ir_errors import IRError, IRErrorKind

__all__ = [
    'CppCodeGenerator',
    'GeneratedFunction',
    'FunctionLibrary',
    'FUNCTION_HEADERS',
    'CLASS_HEADERS',
    'RVEC_HEADER',
    'RVEC_METHODS',
    'REFLECTION_HEADERS',
]


# =============================================================================
# Header Registry
# =============================================================================

FUNCTION_HEADERS: Dict[str, List[str]] = {
    # Standard math (std:: functions from <cmath>)
    "sqrt": ["<cmath>"],
    "sin": ["<cmath>"],
    "cos": ["<cmath>"],
    "tan": ["<cmath>"],
    "asin": ["<cmath>"],
    "acos": ["<cmath>"],
    "atan": ["<cmath>"],
    "atan2": ["<cmath>"],
    "sinh": ["<cmath>"],
    "cosh": ["<cmath>"],
    "tanh": ["<cmath>"],
    "log": ["<cmath>"],
    "log10": ["<cmath>"],
    "log2": ["<cmath>"],
    "exp": ["<cmath>"],
    "exp2": ["<cmath>"],
    "pow": ["<cmath>"],
    "abs": ["<cmath>"],
    "fabs": ["<cmath>"],
    "floor": ["<cmath>"],
    "ceil": ["<cmath>"],
    "round": ["<cmath>"],
    "trunc": ["<cmath>"],
    "fmod": ["<cmath>"],
    "hypot": ["<cmath>"],
    
    # TMath functions
    "TMath::Gaus": ["<TMath.h>"],
    "TMath::Landau": ["<TMath.h>"],
    "TMath::Sqrt": ["<TMath.h>"],
    "TMath::Abs": ["<TMath.h>"],
    "TMath::Sin": ["<TMath.h>"],
    "TMath::Cos": ["<TMath.h>"],
    "TMath::Tan": ["<TMath.h>"],
    "TMath::Log": ["<TMath.h>"],
    "TMath::Exp": ["<TMath.h>"],
    "TMath::Power": ["<TMath.h>"],
    "TMath::Pi": ["<TMath.h>"],
    "TMath::E": ["<TMath.h>"],
    "TMath::TwoPi": ["<TMath.h>"],
    "TMath::PiOver2": ["<TMath.h>"],
    "TMath::PiOver4": ["<TMath.h>"],
    "TMath::DegToRad": ["<TMath.h>"],
    "TMath::RadToDeg": ["<TMath.h>"],
    "TMath::ATan2": ["<TMath.h>"],
    "TMath::Hypot": ["<TMath.h>"],
    "TMath::Sign": ["<TMath.h>"],
    "TMath::Min": ["<TMath.h>"],
    "TMath::Max": ["<TMath.h>"],
    "TMath::Range": ["<TMath.h>"],
}

# Mapping from Python/DSL function names to C++ equivalents
FUNCTION_CPP_NAMES: Dict[str, str] = {
    # Standard math -> std:: versions
    "sqrt": "std::sqrt",
    "sin": "std::sin",
    "cos": "std::cos",
    "tan": "std::tan",
    "asin": "std::asin",
    "acos": "std::acos",
    "atan": "std::atan",
    "atan2": "std::atan2",
    "sinh": "std::sinh",
    "cosh": "std::cosh",
    "tanh": "std::tanh",
    "log": "std::log",
    "log10": "std::log10",
    "log2": "std::log2",
    "exp": "std::exp",
    "exp2": "std::exp2",
    "pow": "std::pow",
    "abs": "std::abs",
    "fabs": "std::fabs",
    "floor": "std::floor",
    "ceil": "std::ceil",
    "round": "std::round",
    "trunc": "std::trunc",
    "fmod": "std::fmod",
    "hypot": "std::hypot",
    
    # min/max -> std:: versions
    "min": "std::min",
    "max": "std::max",
}


# Class headers for object types
CLASS_HEADERS: Dict[str, str] = {
    # Physics objects
    "TParticle": "<TParticle.h>",
    "TLorentzVector": "<TLorentzVector.h>",
    "TVector3": "<TVector3.h>",
    "TVector2": "<TVector2.h>",
    
    # String types
    "TString": "<TString.h>",
    "TObjString": "<TObjString.h>",
    
    # Base classes
    "TObject": "<TObject.h>",
    "TNamed": "<TNamed.h>",
    
    # Math objects
    "TMatrixD": "<TMatrixD.h>",
    "TMatrixF": "<TMatrixF.h>",
    "TVectorD": "<TVectorD.h>",
    "TVectorF": "<TVectorF.h>",
}


# RVec header for vectorized operations (Phase 6b)
RVEC_HEADER = "<ROOT/RVec.hxx>"

# RVec methods supported in Phase 6b
RVEC_METHODS = {"size", "empty", "at"}

# Reflection headers for private/protected member access (Phase 6c)
REFLECTION_HEADERS = ["<TClass.h>", "<TDataMember.h>"]


# =============================================================================
# GeneratedFunction
# =============================================================================

@dataclass
class GeneratedFunction:
    """
    Holds a generated C++ helper function and its metadata.
    
    Attributes:
        name: Function name (e.g., "alias_pt")
        code: Complete C++ function code
        inputs: List of (column_name, cpp_type) sorted alphabetically
        return_type: C++ return type (e.g., "double")
        headers: Required #includes
        ir: Original IR tree (for debugging)
        dsl_expression: Original DSL expression string (for comments/debugging)
        column_name: User-friendly column name for RDataFrame.Define()
    """
    name: str
    code: str
    inputs: List[Tuple[str, str]]
    return_type: str
    headers: List[str]
    ir: Optional[IRNode] = None
    dsl_expression: str = ""
    column_name: str = ""  # User-friendly name for Define()
    
    def get_call_expression(self) -> str:
        """Get expression for RDataFrame.Define()."""
        args = ", ".join(name for name, _ in self.inputs)
        return f"{self.name}({args})"
    
    def get_define_expression(self) -> str:
        """Alias for get_call_expression() for clarity."""
        return self.get_call_expression()
    
    def get_signature(self) -> str:
        """Get function signature without body."""
        params = ", ".join(f"{cpp_type} {name}" for name, cpp_type in self.inputs)
        return f"{self.return_type} {self.name}({params})"
    
    def get_full_code_with_comment(self) -> str:
        """Return full C++ code with DSL expression as comment."""
        if self.dsl_expression:
            comment = f"// DSL: {self.dsl_expression}\n"
        else:
            comment = ""
        return comment + self.code
    
    def __repr__(self) -> str:
        return f"GeneratedFunction({self.name}, inputs={self.inputs}, return_type={self.return_type})"


# =============================================================================
# CppCodeGenerator
# =============================================================================

class CppCodeGenerator:
    """
    Generates C++ code from IR trees.
    
    This generator handles scalar expressions (rank 0). Vector operations
    (rank > 0) are handled in Phase 6.
    
    Example:
        >>> generator = CppCodeGenerator(type_inferrer)
        >>> ir = builder.build("sqrt(px**2 + py**2)")
        >>> func = generator.generate(ir, "pt")
        >>> print(func.code)
        double alias_pt(double px, double py) {
            return std::sqrt(std::pow(px, 2) + std::pow(py, 2));
        }
    """
    
    def __init__(self,
                 type_inferrer: Any = None,
                 reflection_cache: Any = None,
                 error_detail: str = "full",
                 safe_indexing: bool = True,
                 use_reflection: bool = True):
        """
        Initialize code generator.
        
        Args:
            type_inferrer: TypeInferrer for looking up variable types
            reflection_cache: ReflectionCache for method/property types
            error_detail: Level of detail in error messages ("full", "summary", "minimal")
            safe_indexing: If True (default), generate bounds-checked indexing that
                          returns NaN for out-of-bounds access. If False, use direct
                          indexing (faster but undefined behavior on out-of-bounds).
            use_reflection: If True (default), use TClass reflection to access
                           protected/private members. If False, only allow public
                           member access (let C++ compiler enforce access rules).
        """
        self.type_inferrer = type_inferrer
        self.reflection_cache = reflection_cache
        self.error_detail = error_detail
        self.safe_indexing = safe_indexing
        self.use_reflection = use_reflection
        self._existing_names: Set[str] = set()
        self._uses_reflection_access = False  # Track if reflection access was generated
    
    def generate(self, ir: IRNode, name: str) -> GeneratedFunction:
        """
        Generate C++ helper function from IR tree.
        
        Args:
            ir: IR tree representing the expression
            name: Alias name for the function
            
        Returns:
            GeneratedFunction with code and metadata
            
        Raises:
            IRError: If code generation fails
        """
        # Reset per-generation state
        self._uses_reflection_access = False
        
        # Check for unsupported node types
        self._validate_ir(ir)
        
        # Collect inputs (variables) alphabetically
        inputs = self._collect_inputs(ir)
        
        # Determine return type
        return_type = self._get_cpp_return_type(ir)
        
        # Generate function body
        body_expr = self._generate_body(ir)
        
        # Collect required headers
        headers = self._collect_headers(ir)
        
        # Generate unique function name
        func_name = self._generate_function_name(name)
        self._existing_names.add(func_name)
        
        # Format complete function
        code = self._format_function(func_name, inputs, return_type, body_expr)
        
        return GeneratedFunction(
            name=func_name,
            code=code,
            inputs=inputs,
            return_type=return_type,
            headers=headers,
            ir=ir
        )
    
    def _validate_ir(self, ir: IRNode) -> None:
        """Validate IR tree for supported operations."""
        for node in ir.walk():
            # Check for Unknown types
            if node.dtype.kind == IRTypeKind.Unknown:
                if isinstance(node, VariableNode):
                    raise IRError(
                        IRErrorKind.TYPE_ERROR,
                        f"Variable '{node.name}' has Unknown type",
                        suggestions=["Ensure all variables are defined in the schema"]
                    )
                elif not isinstance(node, SliceNode):  # SliceNode has Unknown type by design
                    raise IRError(
                        IRErrorKind.TYPE_ERROR,
                        "Expression contains Unknown type",
                        suggestions=["Check that all sub-expressions have valid types"]
                    )
            
            # Phase 6a: Method calls supported, but not with arguments (except RVec.at())
            if isinstance(node, MethodCallNode):
                # RVec methods: size(), empty() have no args; at() has one arg
                if node.args and len(node.args) > 0:
                    # Allow at() with one argument for RVec
                    if node.method_name == "at" and len(node.args) == 1:
                        pass  # OK - at(i) is allowed
                    else:
                        raise IRError(
                            IRErrorKind.UNSUPPORTED_OP,
                            f"Method arguments not yet supported: {node.method_name}(...)",
                            suggestions=["Use no-argument methods or at(i) for RVec"]
                        )
            
            # Phase 6a: Property access supported
            # (PropertyAccessNode is now allowed)
            
            # Phase 6b: Subscript supported for RVec (single index only, no slicing)
            if isinstance(node, SubscriptNode):
                # Check for slicing (deferred to Phase 7)
                if node.is_slice():
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Slicing operations (e.g., pt[1:3]) are not supported yet",
                        suggestions=["Slicing support will be added in Phase 7"]
                    )
                # Check for multi-dimensional indexing (deferred)
                if len(node.indices) > 1:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Multi-dimensional indexing is not supported yet",
                        suggestions=["Use single index for Phase 6b"]
                    )
                # Check for boolean mask (deferred)
                if node.is_boolean_mask:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Boolean mask indexing is not supported yet",
                        suggestions=["Boolean masking will be added in Phase 7"]
                    )
            
            # Check rank - allow rank 1 for RVec operations, reject rank > 1
            if node.rank > 1 and not isinstance(node, (SliceNode,)):
                raise IRError(
                    IRErrorKind.UNSUPPORTED_OP,
                    f"Nested vector operations (rank > 1) are not supported yet",
                    suggestions=["Nested RVec support will be added in Phase 8"]
                )
    
    def _collect_inputs(self, ir: IRNode) -> List[Tuple[str, str]]:
        """
        Collect all variables and their C++ types.
        
        Returns list sorted alphabetically for deterministic signatures.
        """
        variables: Dict[str, str] = {}
        
        for node in ir.walk():
            if isinstance(node, VariableNode):
                if node.name not in variables:
                    cpp_type = self._get_cpp_type_for_variable(node)
                    variables[node.name] = cpp_type
        
        # Sort alphabetically for determinism
        return sorted(variables.items(), key=lambda x: x[0])
    
    def _get_cpp_type_for_variable(self, node: VariableNode) -> str:
        """Get C++ type string for a variable node."""
        if node.dtype.kind == IRTypeKind.Unknown:
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"Cannot determine C++ type for variable '{node.name}'",
                suggestions=["Ensure variable is defined in the schema"]
            )
        
        # RVec types (rank 1) use const reference - check rank FIRST
        # This handles both RVec<double> and RVec<TLorentzVector>
        if node.rank == 1:
            # Check if cpp_type is already a collection type
            cpp_type = node.dtype.cpp_type or ""
            if cpp_type.startswith("RVec<") or cpp_type.startswith("ROOT::RVec<"):
                # Already have full RVec type
                if not cpp_type.startswith("ROOT::"):
                    cpp_type = f"ROOT::{cpp_type}"
                return f"const {cpp_type}&"
            elif cpp_type.startswith("std::vector<"):
                return f"const {cpp_type}&"
            else:
                # cpp_type is the element type, wrap with RVec
                inner_type = cpp_type if cpp_type else node.dtype.to_cpp()
                return f"const ROOT::RVec<{inner_type}>&"
        
        # Object types (rank 0) use const reference
        if node.dtype.kind == IRTypeKind.Object:
            cpp_type = node.dtype.cpp_type
            return f"const {cpp_type}&"
        
        # Scalar types use value
        return node.dtype.to_cpp()
    
    def _get_cpp_return_type(self, ir: IRNode) -> str:
        """Get C++ return type for the expression."""
        if ir.dtype.kind == IRTypeKind.Unknown:
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                "Cannot determine return type for expression with Unknown type",
                suggestions=["Check that all sub-expressions have valid types"]
            )
        
        # Phase 8: Broadcast nodes already have RVec<T> as dtype
        if isinstance(ir, (MethodBroadcastNode, PropertyBroadcastNode)):
            # dtype is already RVec<result_element_type>
            cpp_type = ir.dtype.cpp_type or ir.dtype.to_cpp()
            # Ensure ROOT:: prefix
            if cpp_type.startswith("RVec<"):
                return f"ROOT::{cpp_type}"
            elif "RVec<" in cpp_type and not cpp_type.startswith("ROOT::"):
                return f"ROOT::{cpp_type}"
            return cpp_type
        
        # RVec return type (rank 1)
        if ir.rank == 1:
            inner_type = ir.dtype.to_cpp()
            # Avoid double-wrapping if inner_type is already RVec
            if inner_type.startswith("RVec<") or inner_type.startswith("ROOT::RVec<"):
                if inner_type.startswith("ROOT::"):
                    return inner_type
                return f"ROOT::{inner_type}"
            return f"ROOT::RVec<{inner_type}>"
        
        # Scalar return type
        return ir.dtype.to_cpp()
    
    def _generate_body(self, ir: IRNode) -> str:
        """Generate C++ expression from IR tree."""
        return self._visit(ir)
    
    def _visit(self, node: IRNode) -> str:
        """Dispatch to appropriate visitor method."""
        if isinstance(node, ConstantNode):
            return self._visit_constant(node)
        elif isinstance(node, VariableNode):
            return self._visit_variable(node)
        elif isinstance(node, UnaryOpNode):
            return self._visit_unary(node)
        elif isinstance(node, BinaryOpNode):
            return self._visit_binary(node)
        elif isinstance(node, TernaryOpNode):
            return self._visit_ternary(node)
        elif isinstance(node, CallNode):
            return self._visit_call(node)
        elif isinstance(node, MethodCallNode):
            return self._visit_method_call(node)
        elif isinstance(node, PropertyAccessNode):
            return self._visit_property_access(node)
        # Phase 8: Broadcasting nodes
        elif isinstance(node, MethodBroadcastNode):
            return self._visit_method_broadcast(node)
        elif isinstance(node, PropertyBroadcastNode):
            return self._visit_property_broadcast(node)
        elif isinstance(node, SubscriptNode):
            return self._visit_subscript(node)
        elif isinstance(node, RVecSliceNode):
            return self._visit_rvec_slice(node)
        else:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Unsupported node type: {type(node).__name__}",
                suggestions=["This node type may be supported in a later phase"]
            )
    
    def _visit_constant(self, node: ConstantNode) -> str:
        """Generate C++ for constant value."""
        value = node.value
        
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float):
            # Ensure decimal point for float literals
            s = repr(value)
            if '.' not in s and 'e' not in s.lower():
                s = s + ".0"
            return s
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                "String constants are not supported in Phase 5",
                suggestions=["String support may be added in a later phase"]
            )
        else:
            return str(value)
    
    def _visit_variable(self, node: VariableNode) -> str:
        """Generate C++ for variable reference."""
        return node.name
    
    def _visit_unary(self, node: UnaryOpNode) -> str:
        """Generate C++ for unary operation."""
        operand = self._visit(node.operand)
        
        if node.op == UnaryOp.NEG:
            return f"(-{operand})"
        elif node.op == UnaryOp.POS:
            return f"(+{operand})"
        elif node.op == UnaryOp.NOT:
            return f"(!{operand})"
        elif node.op == UnaryOp.BITNOT:
            # Bitwise NOT only for int/bool
            if not (node.operand.dtype.is_int() or node.operand.dtype.is_bool()):
                raise IRError(
                    IRErrorKind.TYPE_ERROR,
                    "Bitwise NOT (~) requires integer or boolean operand",
                    suggestions=[f"Operand has type {node.operand.dtype}"]
                )
            return f"(~{operand})"
        else:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Unsupported unary operator: {node.op}"
            )
    
    def _visit_binary(self, node: BinaryOpNode) -> str:
        """Generate C++ for binary operation."""
        left = self._visit(node.left)
        right = self._visit(node.right)
        
        op = node.op
        
        # Arithmetic operators
        if op == BinaryOp.ADD:
            return f"({left} + {right})"
        elif op == BinaryOp.SUB:
            return f"({left} - {right})"
        elif op == BinaryOp.MUL:
            return f"({left} * {right})"
        elif op == BinaryOp.DIV:
            return f"({left} / {right})"
        elif op == BinaryOp.POW:
            # Phase 9: Use unqualified pow for RVec operations to enable ADL
            if node.rank > 0:
                return f"pow({left}, {right})"
            return f"std::pow({left}, {right})"
        elif op == BinaryOp.FLOORDIV:
            # C++ truncation toward zero
            return f"static_cast<long long>({left} / {right})"
        elif op == BinaryOp.MOD:
            # Only integer modulo in Phase 5
            if node.left.dtype.is_float() or node.right.dtype.is_float():
                raise IRError(
                    IRErrorKind.TYPE_ERROR,
                    "Modulo (%) with float operands is not supported in Phase 5",
                    suggestions=["Use integer operands or use std::fmod() explicitly"]
                )
            return f"({left} % {right})"
        
        # Comparison operators
        elif op == BinaryOp.LT:
            return f"({left} < {right})"
        elif op == BinaryOp.LE:
            return f"({left} <= {right})"
        elif op == BinaryOp.GT:
            return f"({left} > {right})"
        elif op == BinaryOp.GE:
            return f"({left} >= {right})"
        elif op == BinaryOp.EQ:
            return f"({left} == {right})"
        elif op == BinaryOp.NE:
            return f"({left} != {right})"
        
        # Logical operators
        elif op == BinaryOp.AND:
            return f"({left} && {right})"
        elif op == BinaryOp.OR:
            return f"({left} || {right})"
        
        # Bitwise operators (int/bool only)
        elif op == BinaryOp.BITAND:
            self._check_bitwise_types(node)
            return f"({left} & {right})"
        elif op == BinaryOp.BITOR:
            self._check_bitwise_types(node)
            return f"({left} | {right})"
        elif op == BinaryOp.BITXOR:
            self._check_bitwise_types(node)
            return f"({left} ^ {right})"
        
        else:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Unsupported binary operator: {op}"
            )
    
    def _check_bitwise_types(self, node: BinaryOpNode) -> None:
        """Check that bitwise operation has valid operand types."""
        left_ok = node.left.dtype.is_int() or node.left.dtype.is_bool()
        right_ok = node.right.dtype.is_int() or node.right.dtype.is_bool()
        
        if not (left_ok and right_ok):
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"Bitwise operator {node.op.value} requires integer or boolean operands",
                suggestions=[
                    f"Left operand has type {node.left.dtype}",
                    f"Right operand has type {node.right.dtype}"
                ]
            )
    
    def _visit_ternary(self, node: TernaryOpNode) -> str:
        """Generate C++ for conditional expression."""
        cond = self._visit(node.condition)
        if_true = self._visit(node.if_true)
        if_false = self._visit(node.if_false)
        
        # Wrap in extra parentheses to avoid precedence issues
        return f"(({cond}) ? ({if_true}) : ({if_false}))"
    
    def _visit_call(self, node: CallNode) -> str:
        """Generate C++ for function call."""
        args = ", ".join(self._visit(arg) for arg in node.args)
        cpp_name = self._cpp_function_name(node)
        
        return f"{cpp_name}({args})"
    
    def _visit_method_call(self, node: MethodCallNode) -> str:
        """
        Generate C++ for method call on object.
        
        Example: particle.GetPx() → "particle.GetPx()"
        
        Phase 6a: Object methods (no arguments).
        Phase 6b: RVec methods (size, empty, at).
        """
        # Generate code for the object
        object_code = self._visit(node.object)
        
        # Check if this is an RVec method (Phase 6b)
        if node.object.rank == 1 and node.method_name in RVEC_METHODS:
            if node.method_name == "at":
                # at(i) has one argument
                if node.args and len(node.args) == 1:
                    idx_code = self._visit(node.args[0])
                    return f"{object_code}.at({idx_code})"
                else:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "RVec.at() requires exactly one argument",
                        suggestions=["Use vec.at(i) with a single index"]
                    )
            else:
                # size() and empty() have no arguments
                return f"{object_code}.{node.method_name}()"
        
        # Optionally validate via reflection for object types
        if self.reflection_cache and node.object.dtype.kind == IRTypeKind.Object:
            class_name = node.object.dtype.cpp_type
            try:
                method_info = self.reflection_cache.resolve_method(
                    class_name,
                    node.method_name
                )
                # Check for pointer return types (not supported in Phase 6a)
                if method_info and method_info.return_type:
                    ret_type = method_info.return_type.strip()
                    if ret_type.endswith('*'):
                        raise IRError(
                            IRErrorKind.UNSUPPORTED_OP,
                            f"Method '{node.method_name}' returns pointer type '{ret_type}' which is not supported",
                            suggestions=["Pointer return types will be supported in Phase 8+"]
                        )
            except IRError as e:
                # Re-raise pointer type errors
                if "pointer type" in str(e.message):
                    raise
                # Other reflection errors - proceed anyway, let C++ compiler catch
                pass
        
        # Generate method call (no arguments for object methods)
        return f"{object_code}.{node.method_name}()"
    
    def _visit_property_access(self, node: PropertyAccessNode) -> str:
        """
        Generate C++ for property access on object.
        
        Phase 6c supports:
        - Direct access for public members: obj.member
        - Reflection-based access for protected/private members via GetOffset()
        
        Example:
            vec.fX (public) → "vec.fX"
            particle.fPx (protected) → lambda with TClass reflection
        
        The reflection approach uses C++11 "magic statics" which are thread-safe.
        """
        # Generate code for the object
        object_code = self._visit(node.object)
        class_name = node.object.dtype.cpp_type if node.object.dtype.kind == IRTypeKind.Object else None
        member_name = node.property_name
        
        # If reflection is disabled or no class info, use direct access
        if not self.use_reflection or not class_name:
            return f"{object_code}.{member_name}"
        
        # Try to get data member info via reflection
        dm_info = self._get_data_member_info(class_name, member_name)
        
        if dm_info is None:
            # Member not found - fall back to direct access (let C++ compiler handle it)
            # This allows mock tests without ROOT to still work
            return f"{object_code}.{member_name}"
        
        # Validate member type
        if not dm_info['is_basic']:
            access_str = dm_info.get('access_level', 'unknown')
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Non-basic member '{member_name}' ({access_str}) cannot be accessed via reflection",
                suggestions=[f"Use getter method instead of direct member access"]
            )
        
        if dm_info['is_pointer']:
            access_str = dm_info.get('access_level', 'unknown')
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Pointer member '{member_name}' ({access_str}) not supported",
                suggestions=["Pointer dereferencing is unsafe; use getter method"]
            )
        
        # Check access level
        if dm_info['is_public']:
            # Public member - use direct access
            return f"{object_code}.{member_name}"
        else:
            # Protected/private member - use reflection access
            self._uses_reflection_access = True
            return self._generate_reflection_access(
                object_code, class_name, member_name, dm_info['type_name']
            )
    
    def _get_data_member_info(self, class_name: str, member_name: str) -> Optional[dict]:
        """
        Get data member info using TClass reflection.
        
        Returns:
            dict with keys: is_public, is_basic, is_pointer, type_name, access_level
            None if TClass or member not found (e.g., ROOT not available)
        """
        try:
            import ROOT
            
            tclass = ROOT.TClass.GetClass(class_name)
            if not tclass:
                return None
            
            dm = tclass.GetDataMember(member_name)
            if not dm:
                return None
            
            props = dm.Property()
            
            return {
                'is_public': bool(props & ROOT.kIsPublic),
                'is_basic': dm.IsBasic(),
                'is_pointer': dm.IsaPointer(),
                'type_name': dm.GetTypeName(),
                'access_level': self._access_level_str(props),
            }
        except (ImportError, AttributeError):
            # ROOT not available - return None to fall back to direct access
            return None
    
    def _access_level_str(self, props: int) -> str:
        """Convert property bits to access level string."""
        try:
            import ROOT
            if props & ROOT.kIsPublic:
                return "public"
            if props & ROOT.kIsProtected:
                return "protected"
            if props & ROOT.kIsPrivate:
                return "private"
        except (ImportError, AttributeError):
            pass
        return "unknown"
    
    def _generate_reflection_access(self, object_code: str, class_name: str,
                                     member_name: str, member_type: str) -> str:
        """
        Generate C++ code for reflection-based member access.
        
        Uses a lambda with static variables for one-time lookup (cached per function).
        C++11 guarantees thread-safe initialization of function-local statics
        ("magic statics"), so this approach is thread-safe.
        
        Example output:
            [&]() -> double {
                static TClass* cls = TClass::GetClass("TParticle");
                static TDataMember* dm = cls->GetDataMember("fPx");
                static Long_t offset = dm->GetOffset();
                return *reinterpret_cast<const double*>(
                    reinterpret_cast<const char*>(&particle) + offset);
            }()
        """
        return (
            f"[&]() -> {member_type} {{ "
            f"static TClass* cls = TClass::GetClass(\"{class_name}\"); "
            f"static TDataMember* dm = cls->GetDataMember(\"{member_name}\"); "
            f"static Long_t offset = dm->GetOffset(); "
            f"return *reinterpret_cast<const {member_type}*>("
            f"reinterpret_cast<const char*>(&{object_code}) + offset); "
            f"}}()"
        )
    
    # =========================================================================
    # Phase 8: Broadcasting Visitors
    # =========================================================================
    
    def _visit_method_broadcast(self, node: MethodBroadcastNode) -> str:
        """
        Generate C++ for element-wise method call on RVec<Object>.
        
        Generates a loop that calls the method on each element and collects
        results into a new RVec.
        
        Example for tracks.Pt() where tracks is RVec<TLorentzVector>:
            [&]() -> ROOT::RVec<double> {
                ROOT::RVec<double> result;
                result.reserve(tracks.size());
                for (const auto& elem : tracks) {
                    result.push_back(elem.Pt());
                }
                return result;
            }()
        """
        target = self._visit(node.target)
        method = node.method_name
        result_type = node.result_element_type
        
        # Build the RVec result type
        rvec_result_type = f"ROOT::RVec<{result_type}>"
        
        return (
            f"[&]() -> {rvec_result_type} {{\n"
            f"    {rvec_result_type} result;\n"
            f"    result.reserve({target}.size());\n"
            f"    for (const auto& elem : {target}) {{\n"
            f"        result.push_back(elem.{method}());\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}()"
        )
    
    def _visit_property_broadcast(self, node: PropertyBroadcastNode) -> str:
        """
        Generate C++ for element-wise property access on RVec<Object>.
        
        Generates a loop that accesses the property on each element and collects
        results into a new RVec.
        
        Example for particles.fPx where particles is RVec<TParticle>:
            [&]() -> ROOT::RVec<double> {
                ROOT::RVec<double> result;
                result.reserve(particles.size());
                for (const auto& elem : particles) {
                    result.push_back(elem.fPx);
                }
                return result;
            }()
        """
        target = self._visit(node.target)
        property_name = node.property_name
        result_type = node.result_element_type
        
        # Build the RVec result type
        rvec_result_type = f"ROOT::RVec<{result_type}>"
        
        # Determine how to access the property
        if node.access_mode == "reflection":
            # Use reflection for protected/private members
            accessor = self._generate_reflection_access(
                "elem", node.element_type, property_name, result_type
            )
        else:
            # Direct access for public members
            accessor = f"elem.{property_name}"
        
        return (
            f"[&]() -> {rvec_result_type} {{\n"
            f"    {rvec_result_type} result;\n"
            f"    result.reserve({target}.size());\n"
            f"    for (const auto& elem : {target}) {{\n"
            f"        result.push_back({accessor});\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}()"
        )
    
    def _visit_subscript(self, node: SubscriptNode) -> str:
        """
        Generate C++ for subscript/indexing operation.
        
        Phase 6b supports:
        - Simple indexing: pt[0], pt[i]
        - Negative literal indexing: pt[-1], pt[-2]
        - Safe bounds checking (default ON): returns NaN on out-of-bounds
        
        Examples:
            pt[0] (safe mode) → (0 >= 0 && static_cast<size_t>(0) < pt.size()) 
                                  ? pt[0] : std::numeric_limits<float>::quiet_NaN()
            pt[-1] (safe mode) → (pt.size() > 0) 
                                   ? pt[pt.size() - 1] : std::numeric_limits<float>::quiet_NaN()
            pt[0] (unsafe mode) → pt[0]
        """
        # Generate code for the value being indexed
        value_code = self._visit(node.value)
        
        # Get the index (single index only in Phase 6b)
        if not node.indices:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                "Subscript requires at least one index",
                suggestions=["Use pt[0] or pt[i] syntax"]
            )
        
        idx = node.indices[0]
        
        # Get the result type for NaN generation
        result_cpp_type = node.dtype.to_cpp()
        
        # Check for negative literal index
        if isinstance(idx, ConstantNode) and isinstance(idx.value, int) and idx.value < 0:
            return self._generate_negative_index(value_code, idx.value, result_cpp_type)
        
        # Generate index code
        idx_code = self._visit(idx)
        
        if self.safe_indexing:
            return self._generate_safe_index(value_code, idx_code, result_cpp_type)
        else:
            return f"{value_code}[{idx_code}]"
    
    def _generate_negative_index(self, value_code: str, neg_idx: int, result_type: str) -> str:
        """
        Generate C++ for negative index access.
        
        pt[-1] → last element
        pt[-2] → second to last
        
        With safe mode, checks that the vector has enough elements.
        """
        abs_idx = abs(neg_idx)
        
        if self.safe_indexing:
            # Safe: check size >= abs_idx
            return (f"({value_code}.size() >= {abs_idx}) "
                    f"? {value_code}[{value_code}.size() - {abs_idx}] "
                    f": std::numeric_limits<{result_type}>::quiet_NaN()")
        else:
            # Unsafe: direct access
            return f"{value_code}[{value_code}.size() - {abs_idx}]"
    
    def _generate_safe_index(self, value_code: str, idx_code: str, result_type: str) -> str:
        """
        Generate C++ for safe bounds-checked index access.
        
        Returns NaN if index is out of bounds.
        """
        return (f"({idx_code} >= 0 && static_cast<size_t>({idx_code}) < {value_code}.size()) "
                f"? {value_code}[{idx_code}] "
                f": std::numeric_limits<{result_type}>::quiet_NaN()")
    
    # =========================================================================
    # Phase 7: RVec Slice Operations
    # =========================================================================
    
    def _visit_rvec_slice(self, node: RVecSliceNode) -> str:
        """
        Generate C++ for RVec slice operations.
        
        Phase 7 supports:
        - First N: [:n] → Take(v, n)
        - Last N: [-n:] → Take(v, -n)
        - From index: [n:] → Take(v, Range(n, size))
        - Range: [a:b] → Take(v, Range(a, min(b, size)))
        - Step: [::step] → loop-based index generation
        - Reverse: [::-1] → manual reverse loop
        - Boolean mask: [mask] → native v[mask]
        """
        target = self._visit(node.target)
        
        if node.slice_kind == SliceKind.FIRST_N:
            return self._gen_slice_first_n(target, node)
        elif node.slice_kind == SliceKind.LAST_N:
            return self._gen_slice_last_n(target, node)
        elif node.slice_kind == SliceKind.FROM_INDEX:
            return self._gen_slice_from_index(target, node)
        elif node.slice_kind == SliceKind.RANGE:
            return self._gen_slice_range(target, node)
        elif node.slice_kind == SliceKind.STEP:
            return self._gen_slice_step(target, node)
        elif node.slice_kind == SliceKind.REVERSE:
            return self._gen_slice_reverse(target, node)
        elif node.slice_kind == SliceKind.BOOLEAN:
            return self._gen_slice_boolean(target, node)
        else:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Unsupported slice kind: {node.slice_kind}",
                suggestions=["This slice pattern may be supported in a later phase"]
            )
    
    def _gen_slice_first_n(self, target: str, node: RVecSliceNode) -> str:
        """[:n] → Take(v, min(n, size)) - first n elements with clamping."""
        n = self._visit(node.stop)
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    size_t n = std::min(static_cast<size_t>({n}), {target}.size());
    return ROOT::VecOps::Take({target}, n);
}}()'''
    
    def _gen_slice_last_n(self, target: str, node: RVecSliceNode) -> str:
        """[-n:] → Take(v, -min(n, size)) - last n elements with clamping."""
        # node.start contains the negative index, e.g., -3
        # We need to extract the absolute value and clamp it
        neg_n = self._visit(node.start)  # e.g., "-3"
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    size_t abs_n = static_cast<size_t>(-({neg_n}));
    size_t clamped = std::min(abs_n, {target}.size());
    if (clamped == 0) return {elem_type}();
    return ROOT::VecOps::Take({target}, -static_cast<int>(clamped));
}}()'''
    
    def _gen_slice_from_index(self, target: str, node: RVecSliceNode) -> str:
        """[n:] or [:] → Take(v, Range(n, size)) - from index to end."""
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        # Handle full slice [:] where start is None
        if node.start is None:
            start = "0"
        else:
            start = self._visit(node.start)
        
        return f'''[&]() -> {elem_type} {{
    size_t start = {start};
    if (start >= {target}.size()) return {elem_type}();
    return ROOT::VecOps::Take({target}, 
        ROOT::VecOps::Range(start, {target}.size()));
}}()'''
    
    def _gen_slice_range(self, target: str, node: RVecSliceNode) -> str:
        """[a:b] → Take(v, Range(a, min(b, size))) - range with clamping."""
        start = self._visit(node.start)
        stop = self._visit(node.stop)
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    size_t start = {start};
    size_t stop = std::min(static_cast<size_t>({stop}), {target}.size());
    if (start >= stop) return {elem_type}();
    return ROOT::VecOps::Take({target}, 
        ROOT::VecOps::Range(start, stop));
}}()'''
    
    def _gen_slice_step(self, target: str, node: RVecSliceNode) -> str:
        """[::step] or [start::step] or [start:stop:step] → loop-based indices."""
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        # Get start (default 0)
        if node.start is not None:
            start = self._visit(node.start)
        else:
            start = "0"
        
        # Get step
        step = self._visit(node.step)
        
        # Get stop (default: size)
        if node.stop is not None:
            stop = self._visit(node.stop)
            stop_expr = f"std::min(static_cast<size_t>({stop}), {target}.size())"
        else:
            stop_expr = f"{target}.size()"
        
        return f'''[&]() -> {elem_type} {{
    ROOT::RVec<size_t> indices;
    size_t stop = {stop_expr};
    for (size_t i = {start}; i < stop; i += {step}) {{
        indices.push_back(i);
    }}
    return ROOT::VecOps::Take({target}, indices);
}}()'''
    
    def _gen_slice_reverse(self, target: str, node: RVecSliceNode) -> str:
        """[::-1] → manual reverse loop (not using VecOps::Reverse)."""
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    {elem_type} result;
    result.reserve({target}.size());
    for (size_t i = {target}.size(); i-- > 0; ) {{
        result.push_back({target}[i]);
    }}
    return result;
}}()'''
    
    def _gen_slice_boolean(self, target: str, node: RVecSliceNode) -> str:
        """[mask] → native RVec boolean indexing."""
        # mask is stored in node.start
        mask = self._visit(node.start)
        return f"{target}[{mask}]"
    
    def _cpp_type_for_rvec(self, dtype) -> str:
        """Get C++ RVec<T> type string from dtype."""
        if isinstance(dtype, IRType):
            inner_type = dtype.to_cpp()
        else:
            inner_type = str(dtype)
        return f"ROOT::RVec<{inner_type}>"
    
    def _cpp_function_name(self, node: CallNode) -> str:
        """Convert DSL function name to C++ function name."""
        # === PHASE 9: For RVec operations, use unqualified names to enable ADL ===
        # ROOT provides vectorized functions like sqrt, sin, cos via ROOT::VecOps
        # ADL (Argument Dependent Lookup) finds them when arguments are ROOT::RVec
        if node.rank > 0:
            # Use unqualified name for ADL with RVec
            func_name = node.func
            if "." in func_name:
                func_name = func_name.replace(".", "::")
            # Don't use std:: prefix for RVec operations
            return func_name
        
        # Check custom cpp_name first - it takes priority
        if node.cpp_name:
            # cpp_name already contains full qualified name (e.g., "std::sqrt")
            # Don't add namespace again even if node.namespace is set
            return node.cpp_name
        
        # Check if it's a namespaced function (e.g., TMath.Gaus)
        if node.namespace:
            # TMath.Gaus -> TMath::Gaus
            return f"{node.namespace}::{node.func}"
        
        # Use Python-dot notation to C++ double-colon
        func_name = node.func
        if "." in func_name:
            # TMath.Gaus -> TMath::Gaus
            func_name = func_name.replace(".", "::")
        
        # Check known function mappings
        if func_name in FUNCTION_CPP_NAMES:
            return FUNCTION_CPP_NAMES[func_name]
        
        # Check if it's a TMath function (case-insensitive prefix)
        if func_name.startswith("TMath::"):
            return func_name
        
        # Return as-is (user-defined or unknown function)
        return func_name
    
    def _collect_headers(self, ir: IRNode) -> List[str]:
        """Collect required headers from IR tree."""
        headers: Set[str] = set()
        needs_limits = False  # Track if we need <limits> for safe indexing
        needs_rvec = False    # Track if we need RVec header
        
        for node in ir.walk():
            if isinstance(node, CallNode):
                cpp_name = self._cpp_function_name(node)
                
                # Check header registry
                if cpp_name in FUNCTION_HEADERS:
                    headers.update(FUNCTION_HEADERS[cpp_name])
                elif node.func in FUNCTION_HEADERS:
                    headers.update(FUNCTION_HEADERS[node.func])
                
                # Add headers from node itself
                if node.headers:
                    headers.update(node.headers)
            
            elif isinstance(node, BinaryOpNode):
                if node.op == BinaryOp.POW:
                    headers.add("<cmath>")
            
            # Phase 6a: Add class headers for object types
            elif isinstance(node, VariableNode):
                if node.dtype.kind == IRTypeKind.Object:
                    class_name = node.dtype.cpp_type
                    if class_name in CLASS_HEADERS:
                        headers.add(CLASS_HEADERS[class_name])
                # Phase 6b: RVec variables need RVec header
                if node.rank == 1:
                    needs_rvec = True
            
            # Phase 6b: Subscript with safe indexing needs <limits>
            elif isinstance(node, SubscriptNode):
                if self.safe_indexing:
                    needs_limits = True
            
            # Phase 7: RVec slice operations need headers
            elif isinstance(node, RVecSliceNode):
                needs_rvec = True
                # These slice kinds use std::min for clamping
                if node.slice_kind in (SliceKind.RANGE, SliceKind.STEP, 
                                       SliceKind.FROM_INDEX, SliceKind.FIRST_N,
                                       SliceKind.LAST_N):
                    headers.add("<algorithm>")  # for std::min
            
            # Phase 8: Broadcasting needs RVec header and element type headers
            elif isinstance(node, (MethodBroadcastNode, PropertyBroadcastNode)):
                needs_rvec = True
                # Add header for element type
                element_type = node.element_type
                if element_type in CLASS_HEADERS:
                    headers.add(CLASS_HEADERS[element_type])
        
        # Add RVec header if needed
        if needs_rvec:
            headers.add(RVEC_HEADER)
        
        # Add limits header if safe indexing is used
        if needs_limits:
            headers.add("<limits>")
        
        # Phase 6c: Add reflection headers if private/protected member access is used
        if self._uses_reflection_access:
            headers.update(REFLECTION_HEADERS)
        
        return sorted(headers)
    
    def _generate_function_name(self, alias_name: str) -> str:
        """Generate unique C++ function name."""
        # Sanitize: replace non-alphanumeric with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', alias_name)
        
        # Ensure doesn't start with digit
        if safe_name and safe_name[0].isdigit():
            safe_name = "_" + safe_name
        
        base_name = f"alias_{safe_name}"
        
        # Handle collisions
        if base_name in self._existing_names:
            hash_suffix = hashlib.md5(alias_name.encode()).hexdigest()[:6]
            return f"{base_name}_{hash_suffix}"
        
        return base_name
    
    def _format_function(self, name: str, inputs: List[Tuple[str, str]],
                         return_type: str, body_expr: str) -> str:
        """Format complete C++ function."""
        params = ", ".join(f"{cpp_type} {var_name}" for var_name, cpp_type in inputs)
        
        return f"""{return_type} {name}({params}) {{
    return {body_expr};
}}"""


# =============================================================================
# FunctionLibrary
# =============================================================================

class FunctionLibrary:
    """
    Compiles and manages C++ helper functions.
    
    This class handles:
    - Storing generated functions
    - Loading required headers
    - Compiling functions via gInterpreter.Declare()
    - Providing expressions for RDataFrame.Define()
    - Saving functions to .C macro files
    
    Example:
        >>> library = FunctionLibrary()
        >>> library.add(func)
        >>> library.compile("alias_pt")
        >>> rdf.Define("pt", library.get_define_expression("alias_pt"))
    """
    
    def __init__(self):
        """Initialize empty function library."""
        self.functions: Dict[str, GeneratedFunction] = {}
        self.compiled: Set[str] = set()
        self.headers_loaded: Set[str] = set()
    
    def add(self, func: GeneratedFunction) -> None:
        """Add function to library."""
        self.functions[func.name] = func
    
    def compile(self, name: str) -> bool:
        """
        Compile function via gInterpreter.Declare().
        
        Args:
            name: Function name to compile
            
        Returns:
            True if compilation succeeded
            
        Raises:
            IRError: If compilation fails
        """
        if name not in self.functions:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Function '{name}' not found in library",
                suggestions=["Add the function first with library.add()"]
            )
        
        if name in self.compiled:
            return True  # Already compiled
        
        func = self.functions[name]
        
        try:
            import ROOT
        except ImportError:
            raise IRError(
                IRErrorKind.COMPILE_ERROR,
                "ROOT is not available",
                suggestions=["Install ROOT to compile C++ functions"]
            )
        
        # Load headers first (once per session)
        for header in func.headers:
            if header not in self.headers_loaded:
                ROOT.gInterpreter.ProcessLine(f'#include {header}')
                self.headers_loaded.add(header)
        
        # Compile function
        result = ROOT.gInterpreter.Declare(func.code)
        
        if not result:
            error_msg = f"Failed to compile helper function '{name}'"
            suggestions = ["Check generated C++ code below:"]
            
            if self.error_detail != "minimal":
                suggestions.append(func.code)
            
            raise IRError(
                IRErrorKind.COMPILE_ERROR,
                error_msg,
                suggestions=suggestions
            )
        
        self.compiled.add(name)
        return True
    
    @property
    def error_detail(self) -> str:
        """Error detail level (for compatibility)."""
        return "full"
    
    def compile_all(self) -> List[IRError]:
        """
        Compile all uncompiled functions.
        
        Returns:
            List of errors for failed compilations
        """
        errors = []
        for name in self.functions:
            if name not in self.compiled:
                try:
                    self.compile(name)
                except IRError as e:
                    errors.append(e)
        return errors
    
    def get_define_expression(self, name: str) -> str:
        """
        Get expression for RDataFrame.Define().
        
        Args:
            name: Function name
            
        Returns:
            Call expression (e.g., "alias_pt(px, py)")
        """
        if name not in self.functions:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Function '{name}' not found in library"
            )
        return self.functions[name].get_call_expression()
    
    def get_function(self, name: str) -> GeneratedFunction:
        """Get function by name."""
        if name not in self.functions:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Function '{name}' not found in library"
            )
        return self.functions[name]
    
    def is_compiled(self, name: str) -> bool:
        """Check if function is compiled."""
        return name in self.compiled
    
    def list_functions(self) -> List[str]:
        """List all function names in library."""
        return sorted(self.functions.keys())
    
    def save_to_file(self, path: str) -> None:
        """
        Save all functions to .C macro file (basic format).
        
        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            # Header comment
            f.write("// Generated by RDataFrameDSL\n")
            f.write("// This file contains helper functions for RDataFrame\n\n")
            
            # Collect all headers
            all_headers: Set[str] = set()
            for func in self.functions.values():
                all_headers.update(func.headers)
            
            # Write header block once
            for header in sorted(all_headers):
                f.write(f'#include {header}\n')
            
            if all_headers:
                f.write('\n')
            
            # Write all functions
            for func in self.functions.values():
                f.write(func.code)
                f.write('\n\n')
    
    def export_to_file(self, filepath: str, include_test: bool = True,
                       tree_name: str = "Events") -> None:
        """
        Export all functions to a .C macro file with DSL comments and test harness.
        
        Args:
            filepath: Output file path
            include_test: Include test_all() harness
            tree_name: TTree name for test harness
        """
        # Collect all unique headers
        all_headers: Set[str] = set()
        for func in self.functions.values():
            all_headers.update(func.headers)
        
        # Always include these for RDataFrame macros
        all_headers.add("<ROOT/RDataFrame.hxx>")
        all_headers.add("<ROOT/RVec.hxx>")
        
        lines = []
        
        # Banner
        lines.append("// " + "=" * 60)
        lines.append("// Generated by RDataFrameDSL")
        lines.append(f"// Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// Functions: {len(self.functions)}")
        lines.append("// " + "=" * 60)
        lines.append("")
        
        # Headers
        for header in sorted(all_headers):
            lines.append(f"#include {header}")
        lines.append("")
        
        # Functions with DSL comments
        for name, func in self.functions.items():
            lines.append("// " + "-" * 60)
            if func.dsl_expression:
                lines.append(f"// DSL: {func.dsl_expression}")
            lines.append("// " + "-" * 60)
            lines.append(func.code)
            lines.append("")
        
        # Test harness
        if include_test:
            lines.append("// " + "=" * 60)
            lines.append("// Test harness - run with: root -l 'macro.C(\"data.root\")'")
            lines.append("// " + "=" * 60)
            lines.append(f'void test_all(const char* filename = "data.root") {{')
            lines.append(f'    ROOT::RDataFrame df("{tree_name}", filename);')
            lines.append("")
            lines.append("    auto df2 = df")
            
            # Chain all Define() calls
            func_list = list(self.functions.values())
            for i, func in enumerate(func_list):
                call_expr = func.get_call_expression()
                # Use column_name if available, otherwise extract from function name
                if func.column_name:
                    col_name = func.column_name
                else:
                    col_name = func.name
                    if col_name.startswith("alias_"):
                        col_name = col_name[6:]
                comma = ";" if i == len(func_list) - 1 else ""
                lines.append(f'        .Define("{col_name}", "{call_expr}"){comma}')
            
            lines.append("")
            
            # Display results - use column_name if available
            col_names = []
            for f in func_list[:5]:
                if f.column_name:
                    col_names.append(f.column_name)
                elif f.name.startswith("alias_"):
                    col_names.append(f.name[6:])
                else:
                    col_names.append(f.name)
            col_str = ", ".join(f'"{c}"' for c in col_names)
            lines.append(f"    auto display = df2.Display({{{col_str}}}, 5);")
            lines.append("    display->Print();")
            lines.append("")
            lines.append('    std::cout << "SUCCESS: All DSL functions compiled and executed." << std::endl;')
            lines.append("}")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    def preview(self) -> str:
        """Return all generated C++ code without compiling."""
        lines = []
        for name, func in self.functions.items():
            lines.append(func.get_full_code_with_comment())
            lines.append("")
        return '\n'.join(lines)
    
    def clear(self) -> None:
        """Clear all functions (does not unload from interpreter)."""
        self.functions.clear()
        self.compiled.clear()
    
    def __len__(self) -> int:
        return len(self.functions)
    
    def __contains__(self, name: str) -> bool:
        return name in self.functions
