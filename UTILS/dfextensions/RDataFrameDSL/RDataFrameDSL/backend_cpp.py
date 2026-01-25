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
from .constants import NAMESPACE_HEADERS, VECTORIZED_NAMESPACES, NAMESPACE_FUNCTION_TYPES  # Phase 11.1/11.1b

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
        is_raw: True if created via define_raw() (Phase 13.5.C)
    """
    name: str
    code: str
    inputs: List[Tuple[str, str]]
    return_type: str
    headers: List[str]
    ir: Optional[IRNode] = None
    dsl_expression: str = ""
    column_name: str = ""  # User-friendly name for Define()
    is_raw: bool = False  # Phase 13.5.C: True if from define_raw()
    
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
    
    def _is_numeric_type(self, type_str: str) -> bool:
        """
        Check if a type is numeric (supports std::numeric_limits::quiet_NaN).
        
        Phase 13.6.D+: Added to distinguish numeric types from custom classes.
        Custom classes need default constructor T() instead of quiet_NaN().
        """
        # Basic numeric types
        NUMERIC_TYPES = {
            'double', 'float', 'int', 'long', 'short', 'char',
            'unsigned int', 'unsigned long', 'unsigned short', 'unsigned char',
            'long long', 'unsigned long long', 'size_t',
            'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
            # ROOT types
            'Int_t', 'Long_t', 'Short_t', 'Char_t',
            'UInt_t', 'ULong_t', 'UShort_t', 'UChar_t',
            'Float_t', 'Double_t', 'Long64_t', 'ULong64_t',
            'Bool_t', 'bool'
        }
        
        # Remove const, &, *, and whitespace for comparison
        clean_type = type_str.replace('const', '').replace('&', '').replace('*', '').strip()
        
        return clean_type in NUMERIC_TYPES
    
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
                # Phase 13.6.D+: Allow SliceNode and MethodCallNode with Unknown type
                # - SliceNode: Unknown type by design (slice semantics)
                # - MethodCallNode: Method return type may not be in signature database,
                #   but C++ compiler can still resolve the call at compile time
                elif not isinstance(node, (SliceNode, MethodCallNode)):
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
                # Phase 13.6.C: N-D slicing now supported - removed is_slice() blocker
                # The _visit_subscript method handles both indexing and slicing
                
                # Phase 13.6.C: N-D indexing supported (up to 5D)
                if len(node.indices) > 5:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        f"Indexing with {len(node.indices)} dimensions exceeds limit (max 5D)",
                        suggestions=["Reduce number of dimensions"]
                    )
                # Check for boolean mask (deferred)
                if node.is_boolean_mask:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Boolean mask indexing is not supported yet",
                        suggestions=["Boolean masking will be added in Phase 7"]
                    )
            
            # Phase 13.3.DSL: Nested RVec (rank > 1) now supported
            # The type system correctly handles nested types, and SubscriptNode
            # properly reduces rank through chained indexing operations.
            # Example: nested[0][0] where nested is RVec<RVec<double>>
            #   - VariableNode('nested'): rank=2, dtype=double
            #   - SubscriptNode(nested[0]): rank=1, dtype=double
            #   - SubscriptNode(nested[0][0]): rank=0, dtype=double
    
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
        
        # Phase 13.3.DSL: Handle nested RVec (rank > 1)
        # For rank=2: ROOT::RVec<ROOT::RVec<dtype>>
        # For rank=3: ROOT::RVec<ROOT::RVec<ROOT::RVec<dtype>>>
        if node.rank > 1:
            # Try to use stored cpp_type first (from schema)
            cpp_type = node.dtype.cpp_type or ""
            if cpp_type and ("RVec<" in cpp_type or "vector<" in cpp_type):
                # Normalize to ROOT:: prefix
                if "RVec<" in cpp_type and not cpp_type.startswith("ROOT::"):
                    cpp_type = cpp_type.replace("RVec<", "ROOT::RVec<")
                return f"const {cpp_type}&"
            else:
                # Reconstruct nested RVec type from rank and dtype
                inner = node.dtype.to_cpp()
                for _ in range(node.rank):
                    inner = f"ROOT::RVec<{inner}>"
                return f"const {inner}&"
        
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
        
        # Phase 13.3.DSL: Handle nested RVec return (rank > 1)
        if ir.rank > 1:
            inner_type = ir.dtype.to_cpp()
            for _ in range(ir.rank):
                inner_type = f"ROOT::RVec<{inner_type}>"
            return inner_type
        
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
        # Phase 11.1b: Check if we need to broadcast scalar namespace function over vectors
        if self._needs_scalar_broadcast(node):
            return self._generate_broadcast_loop(node)
        
        # Phase 13.6.D: Check if we need special handling for nested RVec (rank > 1)
        if self._needs_nested_rvec_handling(node):
            return self._generate_nested_rvec_call(node)
        
        args = ", ".join(self._visit(arg) for arg in node.args)
        cpp_name = self._cpp_function_name(node)
        
        return f"{cpp_name}({args})"
    
    def _needs_nested_rvec_handling(self, node: CallNode) -> bool:
        """
        Check if function call needs special handling for nested RVec (rank > 1).
        
        Phase 13.6.D: ROOT's reduction functions (Sum, Mean, etc.) and elementwise
        math functions (sqrt, abs, etc.) don't natively support RVec<RVec<T>>.
        We need to generate explicit nested loops.
        
        Args:
            node: CallNode to check
            
        Returns:
            True if we need nested RVec handling
        """
        # Check if any argument is nested RVec (rank > 1)
        has_nested_arg = any(arg.rank > 1 for arg in node.args)
        
        if not has_nested_arg:
            return False
        
        # Functions that natively support nested RVec (currently none in ROOT)
        # In future, if ROOT adds support, add them here
        native_nested_functions: set = set()
        
        func_lower = node.func.lower()
        return func_lower not in native_nested_functions
    
    def _generate_nested_rvec_call(self, node: CallNode) -> str:
        """
        Generate C++ for function call on nested RVec (rank > 1).
        
        Phase 13.6.D: Handles two categories of functions:
        1. Reductions (Sum, Mean, Min, Max) - flatten and reduce all elements
        2. Elementwise (sqrt, abs, sin, cos, etc.) - apply to each leaf element
        
        Args:
            node: CallNode with nested RVec argument(s)
            
        Returns:
            C++ code with explicit nested loops
        """
        func_lower = node.func.lower()
        
        # Reduction functions - collapse to scalar
        reduction_functions = {'sum', 'mean', 'min', 'max', 'stddev', 'variance'}
        
        if func_lower in reduction_functions:
            return self._generate_nested_reduction(node, func_lower)
        else:
            # Elementwise function (sqrt, abs, sin, cos, log, exp, etc.)
            return self._generate_nested_elementwise(node)
    
    def _generate_nested_reduction(self, node: CallNode, func_name: str) -> str:
        """
        Generate C++ for reduction function on nested RVec.
        
        Phase 13.6.D: Generates explicit nested loops to reduce all elements.
        
        Example for Sum(cluster_Q[:2, :]):
            [&]() {
                auto nested = <slice_code>;
                double total = 0.0;
                for (size_t i = 0; i < nested.size(); ++i) {
                    for (size_t j = 0; j < nested[i].size(); ++j) {
                        total += nested[i][j];
                    }
                }
                return total;
            }()
        
        Args:
            node: CallNode for reduction
            func_name: Lowercase function name (sum, mean, min, max)
            
        Returns:
            C++ code with nested reduction loop
        """
        if len(node.args) != 1:
            # Fall back to default for multi-arg functions
            args = ", ".join(self._visit(arg) for arg in node.args)
            cpp_name = self._cpp_function_name(node)
            return f"{cpp_name}({args})"
        
        arg = node.args[0]
        arg_code = self._visit(arg)
        rank = arg.rank
        
        # Get the scalar type
        scalar_type = arg.dtype.to_cpp() if hasattr(arg, 'dtype') else 'double'
        
        # Build nested type string for auto declaration
        nested_type = scalar_type
        for _ in range(rank):
            nested_type = f"ROOT::RVec<{nested_type}>"
        
        # Generate nested loop based on rank
        if rank == 2:
            return self._generate_2d_reduction(arg_code, func_name, scalar_type, nested_type)
        elif rank == 3:
            return self._generate_3d_reduction(arg_code, func_name, scalar_type, nested_type)
        else:
            # For rank > 3, fall back to recursive flattening
            return self._generate_generic_nested_reduction(arg_code, func_name, scalar_type, rank)
    
    def _generate_2d_reduction(self, arg_code: str, func_name: str, 
                                scalar_type: str, nested_type: str) -> str:
        """Generate 2D nested reduction (rank 2)."""
        
        if func_name == 'sum':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} total = 0;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                total += nested[i][j];
            }}
        }}
        return total;
    }}()"""
        
        elif func_name == 'mean':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} total = 0;
        size_t count = 0;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                total += nested[i][j];
                ++count;
            }}
        }}
        return count > 0 ? total / static_cast<{scalar_type}>(count) : std::numeric_limits<{scalar_type}>::quiet_NaN();
    }}()"""
        
        elif func_name == 'min':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} result = std::numeric_limits<{scalar_type}>::max();
        bool found = false;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                if (!found || nested[i][j] < result) {{
                    result = nested[i][j];
                    found = true;
                }}
            }}
        }}
        return found ? result : std::numeric_limits<{scalar_type}>::quiet_NaN();
    }}()"""
        
        elif func_name == 'max':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} result = std::numeric_limits<{scalar_type}>::lowest();
        bool found = false;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                if (!found || nested[i][j] > result) {{
                    result = nested[i][j];
                    found = true;
                }}
            }}
        }}
        return found ? result : std::numeric_limits<{scalar_type}>::quiet_NaN();
    }}()"""
        
        else:
            # Fallback for other reductions (stddev, variance)
            # Use Sum of Sum approach
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} total = 0;
        for (size_t i = 0; i < nested.size(); ++i) {{
            total += Sum(nested[i]);
        }}
        return total;
    }}()"""
    
    def _generate_3d_reduction(self, arg_code: str, func_name: str,
                                scalar_type: str, nested_type: str) -> str:
        """Generate 3D nested reduction (rank 3)."""
        
        if func_name == 'sum':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} total = 0;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                for (size_t k = 0; k < nested[i][j].size(); ++k) {{
                    total += nested[i][j][k];
                }}
            }}
        }}
        return total;
    }}()"""
        
        elif func_name == 'mean':
            return f"""[&]() {{
        auto nested = {arg_code};
        {scalar_type} total = 0;
        size_t count = 0;
        for (size_t i = 0; i < nested.size(); ++i) {{
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                for (size_t k = 0; k < nested[i][j].size(); ++k) {{
                    total += nested[i][j][k];
                    ++count;
                }}
            }}
        }}
        return count > 0 ? total / static_cast<{scalar_type}>(count) : std::numeric_limits<{scalar_type}>::quiet_NaN();
    }}()"""
        
        else:
            # Min/Max/other for 3D
            return self._generate_generic_nested_reduction(arg_code, func_name, scalar_type, 3)
    
    def _generate_generic_nested_reduction(self, arg_code: str, func_name: str,
                                            scalar_type: str, rank: int) -> str:
        """Generate generic nested reduction for rank > 3."""
        # Use recursive Sum approach
        if func_name == 'sum':
            inner = "nested"
            for _ in range(rank - 1):
                inner = f"Sum({inner})"
            return f"""[&]() {{
        auto nested = {arg_code};
        return Sum({inner});
    }}()"""
        else:
            # For other functions, flatten first then apply
            # This is a simplified fallback
            return f"""[&]() {{
        auto nested = {arg_code};
        // Fallback: flatten and reduce
        ROOT::RVec<{scalar_type}> flat;
        // TODO: implement generic flattening for rank {rank}
        return {func_name.capitalize()}(flat);
    }}()"""
    
    def _generate_nested_elementwise(self, node: CallNode) -> str:
        """
        Generate C++ for elementwise function on nested RVec.
        
        Phase 13.6.D: Applies function to each leaf element, preserving structure.
        
        Example for sqrt(cluster_Q[:2, :]):
            [&]() -> ROOT::RVec<ROOT::RVec<double>> {
                auto nested = <slice_code>;
                ROOT::RVec<ROOT::RVec<double>> result;
                result.reserve(nested.size());
                for (size_t i = 0; i < nested.size(); ++i) {
                    ROOT::RVec<double> inner;
                    inner.reserve(nested[i].size());
                    for (size_t j = 0; j < nested[i].size(); ++j) {
                        inner.push_back(std::sqrt(nested[i][j]));
                    }
                    result.push_back(inner);
                }
                return result;
            }()
        
        Args:
            node: CallNode for elementwise function
            
        Returns:
            C++ code with nested elementwise loop
        """
        if len(node.args) != 1:
            # Multi-arg elementwise functions - fall back to default
            args = ", ".join(self._visit(arg) for arg in node.args)
            cpp_name = self._cpp_function_name(node)
            return f"{cpp_name}({args})"
        
        arg = node.args[0]
        arg_code = self._visit(arg)
        rank = arg.rank
        cpp_func = self._cpp_function_name(node)
        
        # Get the scalar type
        scalar_type = arg.dtype.to_cpp() if hasattr(arg, 'dtype') else 'double'
        
        # Build result type (same structure as input)
        result_type = scalar_type
        for _ in range(rank):
            result_type = f"ROOT::RVec<{result_type}>"
        
        if rank == 2:
            return self._generate_2d_elementwise(arg_code, cpp_func, scalar_type, result_type)
        elif rank == 3:
            return self._generate_3d_elementwise(arg_code, cpp_func, scalar_type, result_type)
        else:
            # For higher ranks, fall back to a generic approach
            return self._generate_generic_nested_elementwise(arg_code, cpp_func, scalar_type, rank)
    
    def _generate_2d_elementwise(self, arg_code: str, cpp_func: str,
                                  scalar_type: str, result_type: str) -> str:
        """Generate 2D nested elementwise (rank 2)."""
        inner_type = f"ROOT::RVec<{scalar_type}>"
        
        return f"""[&]() -> {result_type} {{
        auto nested = {arg_code};
        {result_type} result;
        result.reserve(nested.size());
        for (size_t i = 0; i < nested.size(); ++i) {{
            {inner_type} inner;
            inner.reserve(nested[i].size());
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                inner.push_back({cpp_func}(nested[i][j]));
            }}
            result.push_back(inner);
        }}
        return result;
    }}()"""
    
    def _generate_3d_elementwise(self, arg_code: str, cpp_func: str,
                                  scalar_type: str, result_type: str) -> str:
        """Generate 3D nested elementwise (rank 3)."""
        inner_type_2 = f"ROOT::RVec<{scalar_type}>"
        inner_type_1 = f"ROOT::RVec<{inner_type_2}>"
        
        return f"""[&]() -> {result_type} {{
        auto nested = {arg_code};
        {result_type} result;
        result.reserve(nested.size());
        for (size_t i = 0; i < nested.size(); ++i) {{
            {inner_type_1} mid;
            mid.reserve(nested[i].size());
            for (size_t j = 0; j < nested[i].size(); ++j) {{
                {inner_type_2} inner;
                inner.reserve(nested[i][j].size());
                for (size_t k = 0; k < nested[i][j].size(); ++k) {{
                    inner.push_back({cpp_func}(nested[i][j][k]));
                }}
                mid.push_back(inner);
            }}
            result.push_back(mid);
        }}
        return result;
    }}()"""
    
    def _generate_generic_nested_elementwise(self, arg_code: str, cpp_func: str,
                                              scalar_type: str, rank: int) -> str:
        """Generate generic nested elementwise for rank > 3."""
        # Build result type
        result_type = scalar_type
        for _ in range(rank):
            result_type = f"ROOT::RVec<{result_type}>"
        
        # For very deep nesting, use Map if available, otherwise fallback
        return f"""[&]() -> {result_type} {{
        auto nested = {arg_code};
        // TODO: implement generic elementwise for rank {rank}
        // Using ROOT::VecOps::Map would be ideal but doesn't support nested RVec
        return nested;  // Placeholder
    }}()"""
    
    def _needs_scalar_broadcast(self, node: CallNode) -> bool:
        """
        Check if namespace call needs loop broadcasting.
        
        Phase 11.1b: Returns True if:
        - The function has a namespace (e.g., TMath::)
        - The namespace is NOT in VECTORIZED_NAMESPACES (which handle RVec natively)
        - At least one argument has rank > 0 (is a vector)
        
        Args:
            node: CallNode to check
            
        Returns:
            True if we need to wrap in a loop for broadcasting
        """
        # Only applies to namespace functions
        if not node.namespace:
            return False
        
        # Vectorized namespaces handle RVec natively - no loop needed
        # Check both dot and :: notation
        ns_dot = node.namespace.replace("::", ".")
        ns_cpp = node.namespace.replace(".", "::")
        if ns_dot in VECTORIZED_NAMESPACES or ns_cpp in VECTORIZED_NAMESPACES:
            return False
        
        # If any argument is a vector, we need to broadcast
        return any(arg.rank > 0 for arg in node.args)
    
    def _generate_broadcast_loop(self, node: CallNode) -> str:
        """
        Generate loop to broadcast scalar namespace function over vectors.
        
        Phase 11.1b: Wraps scalar functions (like TMath::Sqrt) in a loop
        when applied to RVec arguments. Vector arguments are hoisted to
        local temporaries to ensure O(n) complexity even with nested broadcasts.
        
        Example:
            TMath.Sqrt(pt) where pt is RVec<double>
            Generates:
            [&]() {
                auto _arg0 = pt;
                ROOT::RVec<double> result;
                size_t n = _arg0.size();
                result.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    result.push_back(TMath::Sqrt(_arg0[i]));
                }
                return result;
            }()
            
        For nested broadcasts like TMath.Sqrt(TMath.Abs(pt)):
            [&]() {
                auto _arg0 = [inner broadcast]();  // Computed ONCE
                ROOT::RVec<double> result;
                size_t n = _arg0.size();
                ...
                    result.push_back(TMath::Sqrt(_arg0[i]));
                ...
            }()
        
        Args:
            node: CallNode with namespace function and vector args
            
        Returns:
            C++ code with loop-based broadcasting (O(n) guaranteed)
        """
        # Get the C++ function name (with namespace)
        cpp_name = self._cpp_function_name(node)
        
        # Determine return type
        return_type = self._get_broadcast_return_type(node)
        
        # Phase 1: Hoist all vector-valued arguments to temporaries
        # This ensures nested broadcasts are computed ONCE, not N times
        hoisted_vars = []
        call_args = []
        size_exprs = []
        
        for i, arg in enumerate(node.args):
            arg_code = self._visit(arg)
            
            if arg.rank > 0:
                # Hoist vector argument to temporary (computed ONCE)
                temp_name = f"_arg{i}"
                hoisted_vars.append(f"auto {temp_name} = {arg_code};")
                call_args.append(f"{temp_name}[i]")
                size_exprs.append(f"{temp_name}.size()")
            else:
                # Scalar used directly (no hoisting needed)
                call_args.append(arg_code)
        
        if not size_exprs:
            # Should not happen if _needs_scalar_broadcast returned True
            raise RuntimeError("_generate_broadcast_loop called with no vector args")
        
        # Phase 2: Build size expression with min() for safety
        if len(size_exprs) == 1:
            size_expr = size_exprs[0]
        else:
            # Multiple vectors: chain std::min for safety
            size_expr = size_exprs[0]
            for s in size_exprs[1:]:
                size_expr = f"std::min({size_expr}, {s})"
        
        # Phase 3: Generate the loop
        func_call = f"{cpp_name}({', '.join(call_args)})"
        hoisted_code = "\n        ".join(hoisted_vars)
        
        return f"""[&]() {{
        {hoisted_code}
        {return_type} result;
        size_t n = {size_expr};
        result.reserve(n);
        for (size_t i = 0; i < n; ++i) {{
            result.push_back({func_call});
        }}
        return result;
    }}()"""
    
    def _get_broadcast_return_type(self, node: CallNode) -> str:
        """
        Get C++ return type for broadcasted function.
        
        Phase 11.1b: Looks up scalar return type from NAMESPACE_FUNCTION_TYPES
        and wraps it in RVec<>.
        
        Args:
            node: CallNode being broadcasted
            
        Returns:
            C++ type string like "ROOT::RVec<double>"
        """
        # Get scalar return type from NAMESPACE_FUNCTION_TYPES
        scalar_type = "double"  # Default
        
        # Convert namespace to dot notation for lookup
        ns_dot = node.namespace.replace("::", ".") if node.namespace else ""
        
        if ns_dot in NAMESPACE_FUNCTION_TYPES:
            if node.func in NAMESPACE_FUNCTION_TYPES[ns_dot]:
                type_str = NAMESPACE_FUNCTION_TYPES[ns_dot][node.func]
                # Map type strings to C++ types
                type_map = {
                    "double": "double",
                    "float": "float",
                    "int": "int",
                    "bool": "bool",
                    "Int_t": "int",
                    "Double_t": "double",
                    "Float_t": "float",
                    "Bool_t": "bool",
                }
                scalar_type = type_map.get(type_str, type_str)
        
        return f"ROOT::RVec<{scalar_type}>"
    
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
        # Phase 13.6.D+: Wrap ternary expressions in parentheses for correct precedence
        # Without this: condition ? obj : fallback.method() ← Wrong!
        # With this: (condition ? obj : fallback).method() ← Correct!
        if '?' in object_code:
            # Always wrap ternary to ensure correct precedence
            # Strip outer spaces and wrap
            object_code = f"({object_code.strip()})"
        
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
        
        Phase 13.6.C: Extended to support N-dimensional slicing.
        
        Supports:
        - Simple indexing: pt[0], pt[-1]
        - N-D indexing: nested[0, 1] or nested[0][1]
        - N-D slicing: cluster_Q[0:2, 0:3]
        - Mixed: cluster_Q[0, :], cluster_Q[:, 0]
        
        Examples:
            cluster_Q[0:2, :]     → first 2 tracks, all clusters
            cluster_Q[:, 0:3]     → all tracks, first 3 clusters  
            cluster_Q[0:2, 0:3]   → first 2 tracks, first 3 clusters
            hit_E[0:2, :, 0:3]    → 3D slicing
        """
        # Check if this has N-D indices
        if len(node.indices) > 1:
            # Check if any index is a slice
            has_slice = any(isinstance(idx, SliceNode) for idx in node.indices)
            if has_slice:
                return self._visit_nd_slice(node)
            else:
                return self._visit_nd_index(node)
        
        # Single dimension - check for slice (shouldn't happen, uses RVecSliceNode)
        if len(node.indices) == 1 and isinstance(node.indices[0], SliceNode):
            return self._visit_1d_slice_fallback(node)
        
        # Original simple indexing logic
        return self._visit_subscript_simple(node)
    
    def _visit_subscript_simple(self, node: SubscriptNode) -> str:
        """
        Original _visit_subscript logic for simple single-index access.
        
        Handles: arr[0], arr[-1], arr[i]
        """
        value_code = self._visit(node.value)
        
        if not node.indices:
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                "Subscript requires at least one index",
                suggestions=["Use pt[0] or pt[i] syntax"]
            )
        
        idx = node.indices[0]
        result_cpp_type = node.dtype.to_cpp()
        result_rank = node.rank
        
        # Check for negative literal index
        if isinstance(idx, ConstantNode) and isinstance(idx.value, int) and idx.value < 0:
            return self._generate_negative_index(value_code, idx.value, result_cpp_type, result_rank)
        
        idx_code = self._visit(idx)
        
        if self.safe_indexing:
            return self._generate_safe_index(value_code, idx_code, result_cpp_type, result_rank)
        else:
            return f"{value_code}[{idx_code}]"
    
    def _generate_negative_index(self, value_code: str, neg_idx: int, result_type: str, result_rank: int = 0) -> str:
        """
        Generate C++ for negative index access.
        
        pt[-1] → last element
        pt[-2] → second to last
        
        With safe mode, checks that the vector has enough elements.
        
        Phase 13.3.DSL: Added result_rank to generate correct fallback for nested RVec.
        Also handles chained subscripts where value_code is a complex expression.
        """
        abs_idx = abs(neg_idx)
        
        if self.safe_indexing:
            # Phase 13.3.DSL: Use empty vector fallback for vector results
            if result_rank > 0:
                # Result is a vector - fallback is empty vector
                inner = result_type
                for _ in range(result_rank):
                    inner = f"ROOT::RVec<{inner}>"
                fallback = f"{inner}{{}}"
            else:
                # Result is scalar - fallback depends on type
                # Phase 13.6.D+: Use default constructor for custom classes
                if self._is_numeric_type(result_type):
                    fallback = f"std::numeric_limits<{result_type}>::quiet_NaN()"
                else:
                    fallback = f"{result_type}()"  # Default constructor for custom classes
            
            # Phase 13.3.DSL: Handle chained subscripts - if value_code is complex
            # (contains ternary), wrap in lambda to extract to variable
            if '?' in value_code:
                return f'''[&]() {{
    auto v = {value_code};
    return (v.size() >= {abs_idx}) ? v[v.size() - {abs_idx}] : {fallback};
}}()'''
            else:
                # Safe: check size >= abs_idx
                return (f"({value_code}.size() >= {abs_idx}) "
                        f"? {value_code}[{value_code}.size() - {abs_idx}] "
                        f": {fallback}")
        else:
            # Unsafe: direct access
            return f"{value_code}[{value_code}.size() - {abs_idx}]"
    
    def _generate_safe_index(self, value_code: str, idx_code: str, result_type: str, result_rank: int = 0) -> str:
        """
        Generate C++ for safe bounds-checked index access.
        
        Returns NaN if index is out of bounds for scalars,
        or empty vector for vector results.
        
        Phase 13.3.DSL: Added result_rank to generate correct fallback for nested RVec.
        Also handles chained subscripts where value_code is a complex expression.
        """
        # Phase 13.3.DSL: Use empty vector fallback for vector results
        if result_rank > 0:
            # Result is a vector - fallback is empty vector
            inner = result_type
            for _ in range(result_rank):
                inner = f"ROOT::RVec<{inner}>"
            fallback = f"{inner}{{}}"
        else:
            # Result is scalar - fallback depends on type
            # Phase 13.6.D+: Use default constructor for custom classes
            if self._is_numeric_type(result_type):
                fallback = f"std::numeric_limits<{result_type}>::quiet_NaN()"
            else:
                fallback = f"{result_type}()"  # Default constructor for custom classes
        
        # Phase 13.3.DSL: Handle chained subscripts - if value_code is complex
        # (contains ternary), wrap in lambda to extract to variable
        if '?' in value_code:
            return f'''[&]() {{
    auto v = {value_code};
    return ({idx_code} >= 0 && static_cast<size_t>({idx_code}) < v.size()) ? v[{idx_code}] : {fallback};
}}()'''
        else:
            # Phase 13.6.G: Wrap entire ternary in parentheses for correct C++ precedence
            # Without this: A + (condition) ? B : fallback → parsed as (A + condition) ? B : fallback
            # With this:    A + ((condition) ? B : fallback) → correct!
            return (f"(({idx_code} >= 0 && static_cast<size_t>({idx_code}) < {value_code}.size()) "
                    f"? {value_code}[{idx_code}] "
                    f": {fallback})")
    
    # =========================================================================
    # Phase 13.6.C: N-D Slicing Operations
    # =========================================================================
    
    def _visit_nd_index(self, node: SubscriptNode) -> str:
        """
        Handle N-dimensional indexing without slices.
        
        cluster_Q[0, 1] → cluster_Q[0][1] with bounds checking
        """
        value_code = self._visit(node.value)
        result_cpp_type = node.dtype.to_cpp()
        
        if self.safe_indexing:
            return self._generate_nd_index_safe(node, value_code, result_cpp_type)
        else:
            # Simple chained access
            for idx in node.indices:
                idx_code = self._visit(idx)
                value_code = f"{value_code}[{idx_code}]"
            return value_code
    
    def _generate_nd_index_safe(self, node: SubscriptNode, value_code: str, 
                                 result_cpp_type: str) -> str:
        """Generate safe N-D index access with bounds checking."""
        num_dims = len(node.indices)
        result_rank = node.rank
        fallback = self._make_nd_fallback(result_cpp_type, result_rank)
        
        code_lines = []
        code_lines.append(f"[&]() {{")
        code_lines.append(f"    auto d0 = {value_code};")
        
        for i, idx in enumerate(node.indices):
            d_var = f"d{i}"
            next_d_var = f"d{i+1}" if i < num_dims - 1 else "result"
            
            if isinstance(idx, ConstantNode) and isinstance(idx.value, int) and idx.value < 0:
                abs_idx = abs(idx.value)
                code_lines.append(f"    if ({d_var}.size() < {abs_idx}) return {fallback};")
                if i < num_dims - 1:
                    code_lines.append(f"    auto {next_d_var} = {d_var}[{d_var}.size() - {abs_idx}];")
                else:
                    code_lines.append(f"    return {d_var}[{d_var}.size() - {abs_idx}];")
            else:
                idx_code = self._visit(idx)
                code_lines.append(f"    if ({idx_code} < 0 || static_cast<size_t>({idx_code}) >= {d_var}.size()) return {fallback};")
                if i < num_dims - 1:
                    code_lines.append(f"    auto {next_d_var} = {d_var}[{idx_code}];")
                else:
                    code_lines.append(f"    return {d_var}[{idx_code}];")
        
        code_lines.append(f"}}()")
        return "\n".join(code_lines)
    
    def _visit_nd_slice(self, node: SubscriptNode) -> str:
        """
        Generate C++ for N-dimensional slicing.
        
        Examples:
            cluster_Q[0:2, :]     → slice first 2 tracks, keep all clusters
            cluster_Q[:, 0:3]     → keep all tracks, slice first 3 clusters
            cluster_Q[0:2, 0:3]   → slice both dimensions
            hit_E[0:2, :, 0:3]    → 3D slicing
        """
        num_dims = len(node.indices)
        target = self._visit(node.value)
        result_type = self._get_nd_result_type(node)
        
        # Analyze each dimension's slice/index
        slice_infos = []
        for i, idx in enumerate(node.indices):
            slice_infos.append(self._analyze_slice_dim(idx, i))
        
        # Generate code based on dimensionality
        if num_dims == 2:
            return self._generate_2d_slice(target, slice_infos, result_type, node)
        elif num_dims == 3:
            return self._generate_3d_slice(target, slice_infos, result_type, node)
        else:
            return self._generate_generic_nd_slice(target, slice_infos, result_type, num_dims, node)
    
    def _analyze_slice_dim(self, idx, dim: int) -> dict:
        """
        Analyze a single dimension's slice/index.
        
        Returns dict with kind, start, stop, step, dim, and for indices: index, is_negative.
        """
        if isinstance(idx, SliceNode):
            start = self._visit(idx.start) if idx.start else None
            stop = self._visit(idx.stop) if idx.stop else None
            step = self._visit(idx.step) if idx.step else None
            
            # Classify the slice kind
            if start is None and stop is None and step is None:
                return {'kind': 'full', 'start': None, 'stop': None, 'step': None, 'dim': dim}
            
            if step is not None:
                if isinstance(idx.step, ConstantNode) and idx.step.value == -1 and start is None and stop is None:
                    return {'kind': 'reverse', 'start': None, 'stop': None, 'step': step, 'dim': dim}
                return {'kind': 'step', 'start': start, 'stop': stop, 'step': step, 'dim': dim}
            
            if start is None and stop is not None:
                return {'kind': 'first_n', 'start': None, 'stop': stop, 'step': None, 'dim': dim}
            
            if start is not None and stop is None:
                if isinstance(idx.start, ConstantNode) and idx.start.value < 0:
                    return {'kind': 'last_n', 'start': start, 'stop': None, 'step': None, 'dim': dim}
                return {'kind': 'from_idx', 'start': start, 'stop': None, 'step': None, 'dim': dim}
            
            return {'kind': 'range', 'start': start, 'stop': stop, 'step': None, 'dim': dim}
        
        else:
            idx_code = self._visit(idx)
            is_negative = isinstance(idx, ConstantNode) and isinstance(idx.value, int) and idx.value < 0
            return {'kind': 'index', 'index': idx_code, 'dim': dim, 'is_negative': is_negative,
                    'neg_value': idx.value if is_negative else None}
    
    def _get_nd_result_type(self, node: SubscriptNode) -> str:
        """Get the full C++ type for the result of N-D slicing."""
        base_type = node.dtype.to_cpp()
        rank = node.rank
        
        result = base_type
        for _ in range(rank):
            result = f"ROOT::RVec<{result}>"
        return result
    
    def _make_nd_fallback(self, base_type: str, rank: int) -> str:
        """
        Generate fallback value for out-of-bounds access.
        
        Phase 13.6.D+: Use default constructor for custom classes.
        """
        if rank > 0:
            inner = base_type
            for _ in range(rank):
                inner = f"ROOT::RVec<{inner}>"
            return f"{inner}{{}}"
        else:
            # Scalar fallback - depends on type
            if self._is_numeric_type(base_type):
                return f"std::numeric_limits<{base_type}>::quiet_NaN()"
            else:
                return f"{base_type}()"  # Default constructor for custom classes
    
    def _generate_2d_slice(self, target: str, slice_infos: list, 
                           result_type: str, node: SubscriptNode) -> str:
        """Generate optimized C++ for 2D slicing."""
        outer = slice_infos[0]
        inner = slice_infos[1]
        
        # Get inner element type (one RVec layer removed)
        inner_type = result_type
        if inner_type.startswith("ROOT::RVec<"):
            inner_type = inner_type[len("ROOT::RVec<"):-1]
        
        elem_type = node.dtype.to_cpp()
        
        code = []
        code.append(f"[&]() -> {result_type} {{")
        code.append(f"    auto src = {target};")
        code.append(f"    {result_type} result;")
        
        # Handle outer dimension
        if outer['kind'] == 'index':
            idx = outer['index']
            if outer.get('is_negative'):
                neg_val = outer['neg_value']
                abs_val = abs(neg_val)
                code.append(f"    if (src.size() < {abs_val}) return result;")
                code.append(f"    auto row = src[src.size() - {abs_val}];")
            else:
                code.append(f"    if (static_cast<size_t>({idx}) >= src.size()) return result;")
                code.append(f"    auto row = src[{idx}];")
            
            inner_result = self._gen_inner_slice_expr("row", inner, inner_type)
            code.append(f"    return {inner_result};")
        else:
            outer_loop = self._gen_loop_header("src", outer, "oi")
            if outer_loop['setup']:
                code.append(f"    {outer_loop['setup']}")
            code.append(f"    for ({outer_loop['header']}) {{")
            code.append(f"        auto row = src[{outer_loop['index']}];")
            
            inner_result = self._gen_inner_slice_expr("row", inner, inner_type)
            code.append(f"        result.push_back({inner_result});")
            code.append("    }")
            code.append("    return result;")
        
        code.append("}()")
        return "\n".join(code)
    
    def _generate_3d_slice(self, target: str, slice_infos: list,
                           result_type: str, node: SubscriptNode) -> str:
        """Generate C++ for 3D slicing."""
        d0 = slice_infos[0]
        d1 = slice_infos[1]
        d2 = slice_infos[2]
        
        # Type at each level
        type_d0 = result_type
        type_d1 = type_d0[len("ROOT::RVec<"):-1] if type_d0.startswith("ROOT::RVec<") else type_d0
        type_d2 = type_d1[len("ROOT::RVec<"):-1] if type_d1.startswith("ROOT::RVec<") else type_d1
        
        code = []
        code.append(f"[&]() -> {result_type} {{")
        code.append(f"    auto src = {target};")
        code.append(f"    {result_type} result;")
        
        # Level 0
        if d0['kind'] == 'index':
            idx = d0['index']
            if d0.get('is_negative'):
                abs_val = abs(d0['neg_value'])
                code.append(f"    if (src.size() < {abs_val}) return result;")
                code.append(f"    auto lv0 = src[src.size() - {abs_val}];")
            else:
                code.append(f"    if (static_cast<size_t>({idx}) >= src.size()) return result;")
                code.append(f"    auto lv0 = src[{idx}];")
            indent0 = "    "
            result_var0 = "result"
            close_loop0 = False
        else:
            loop0 = self._gen_loop_header("src", d0, "i0")
            if loop0['setup']:
                code.append(f"    {loop0['setup']}")
            code.append(f"    for ({loop0['header']}) {{")
            code.append(f"        auto lv0 = src[{loop0['index']}];")
            code.append(f"        {type_d1} res0;")
            indent0 = "        "
            result_var0 = "res0"
            close_loop0 = True
        
        # Level 1
        if d1['kind'] == 'index':
            idx = d1['index']
            if d1.get('is_negative'):
                abs_val = abs(d1['neg_value'])
                code.append(f"{indent0}if (lv0.size() < {abs_val}) {{ }}")
                code.append(f"{indent0}else {{ auto lv1 = lv0[lv0.size() - {abs_val}];")
            else:
                code.append(f"{indent0}if (static_cast<size_t>({idx}) < lv0.size()) {{")
                code.append(f"{indent0}    auto lv1 = lv0[{idx}];")
            
            inner_result = self._gen_inner_slice_expr("lv1", d2, type_d2)
            if d0['kind'] == 'index':
                code.append(f"{indent0}    return {inner_result};")
            else:
                code.append(f"{indent0}    {result_var0} = {inner_result};")
            code.append(f"{indent0}}}")
        else:
            loop1 = self._gen_loop_header("lv0", d1, "i1")
            if loop1['setup']:
                code.append(f"{indent0}{loop1['setup']}")
            code.append(f"{indent0}for ({loop1['header']}) {{")
            code.append(f"{indent0}    auto lv1 = lv0[{loop1['index']}];")
            
            inner_result = self._gen_inner_slice_expr("lv1", d2, type_d2)
            code.append(f"{indent0}    {result_var0}.push_back({inner_result});")
            code.append(f"{indent0}}}")
        
        # Close loops and return
        if close_loop0:
            code.append(f"        result.push_back(res0);")
            code.append("    }")
        code.append("    return result;")
        code.append("}()")
        
        return "\n".join(code)
    
    def _generate_generic_nd_slice(self, target: str, slice_infos: list,
                                    result_type: str, num_dims: int, 
                                    node: SubscriptNode) -> str:
        """Generate C++ for generic N-D slicing (4D+)."""
        # For 4D+, generate explicit nested loops
        code = []
        code.append(f"[&]() -> {result_type} {{")
        code.append(f"    auto src = {target};")
        code.append(f"    {result_type} result;")
        
        # Build nested type strings
        types = [result_type]
        t = result_type
        for _ in range(num_dims - 1):
            if t.startswith("ROOT::RVec<"):
                t = t[len("ROOT::RVec<"):-1]
            types.append(t)
        
        # For simplicity, generate nested loops (may not be fully optimized for all patterns)
        # This handles the general case
        indent = "    "
        src_var = "src"
        
        for d in range(num_dims - 1):
            si = slice_infos[d]
            lv_var = f"lv{d}"
            
            if si['kind'] == 'index':
                idx = si['index']
                code.append(f"{indent}if (static_cast<size_t>({idx}) >= {src_var}.size()) return result;")
                code.append(f"{indent}auto {lv_var} = {src_var}[{idx}];")
            else:
                loop = self._gen_loop_header(src_var, si, f"i{d}")
                res_var = f"res{d}"
                if loop['setup']:
                    code.append(f"{indent}{loop['setup']}")
                code.append(f"{indent}{types[d+1]} {res_var};")
                code.append(f"{indent}for ({loop['header']}) {{")
                code.append(f"{indent}    auto {lv_var} = {src_var}[{loop['index']}];")
                indent += "    "
            
            src_var = lv_var
        
        # Innermost slice
        inner_si = slice_infos[-1]
        inner_result = self._gen_inner_slice_expr(src_var, inner_si, types[-1])
        
        # Build result chain (simplified)
        code.append(f"{indent}result.push_back({inner_result});")
        
        # Close loops
        for d in range(num_dims - 2, -1, -1):
            si = slice_infos[d]
            if si['kind'] != 'index':
                indent = indent[:-4]
                code.append(f"{indent}}}")
        
        code.append("    return result;")
        code.append("}()")
        
        return "\n".join(code)
    
    def _gen_loop_header(self, vec_var: str, si: dict, idx_var: str) -> dict:
        """Generate loop setup and header for a slice dimension."""
        kind = si['kind']
        
        if kind == 'full':
            return {
                'setup': "",
                'header': f"size_t {idx_var} = 0; {idx_var} < {vec_var}.size(); ++{idx_var}",
                'index': idx_var
            }
        
        elif kind == 'first_n':
            stop = si['stop']
            return {
                'setup': f"size_t {idx_var}_n = std::min(static_cast<size_t>({stop}), {vec_var}.size());",
                'header': f"size_t {idx_var} = 0; {idx_var} < {idx_var}_n; ++{idx_var}",
                'index': idx_var
            }
        
        elif kind == 'range':
            start, stop = si['start'], si['stop']
            return {
                'setup': f"size_t {idx_var}_start = std::min(static_cast<size_t>({start}), {vec_var}.size()); size_t {idx_var}_stop = std::min(static_cast<size_t>({stop}), {vec_var}.size());",
                'header': f"size_t {idx_var} = {idx_var}_start; {idx_var} < {idx_var}_stop; ++{idx_var}",
                'index': idx_var
            }
        
        elif kind == 'from_idx':
            start = si['start']
            return {
                'setup': f"size_t {idx_var}_start = std::min(static_cast<size_t>({start}), {vec_var}.size());",
                'header': f"size_t {idx_var} = {idx_var}_start; {idx_var} < {vec_var}.size(); ++{idx_var}",
                'index': idx_var
            }
        
        elif kind == 'last_n':
            start = si['start']
            return {
                'setup': f"size_t {idx_var}_n = std::min(static_cast<size_t>(-({start})), {vec_var}.size()); size_t {idx_var}_start = {vec_var}.size() - {idx_var}_n;",
                'header': f"size_t {idx_var} = {idx_var}_start; {idx_var} < {vec_var}.size(); ++{idx_var}",
                'index': idx_var
            }
        
        elif kind == 'step':
            start = si['start'] or "0"
            stop = si['stop']
            step = si['step']
            stop_expr = f"std::min(static_cast<size_t>({stop}), {vec_var}.size())" if stop else f"{vec_var}.size()"
            return {
                'setup': f"size_t {idx_var}_stop = {stop_expr};",
                'header': f"size_t {idx_var} = {start}; {idx_var} < {idx_var}_stop; {idx_var} += {step}",
                'index': idx_var
            }
        
        elif kind == 'reverse':
            return {
                'setup': "",
                'header': f"size_t {idx_var} = 0; {idx_var} < {vec_var}.size(); ++{idx_var}",
                'index': f"{vec_var}.size() - 1 - {idx_var}"
            }
        
        else:
            raise ValueError(f"Unknown slice kind for loop: {kind}")
    
    def _gen_inner_slice_expr(self, vec_var: str, si: dict, result_type: str) -> str:
        """Generate slice expression for innermost dimension using ROOT::VecOps."""
        kind = si['kind']
        
        if kind == 'full':
            return vec_var
        
        elif kind == 'index':
            idx = si['index']
            if si.get('is_negative'):
                abs_val = abs(si['neg_value'])
                return f"({vec_var}.size() >= {abs_val} ? {vec_var}[{vec_var}.size() - {abs_val}] : {result_type}{{}})"
            else:
                return f"(static_cast<size_t>({idx}) < {vec_var}.size() ? {vec_var}[{idx}] : {result_type}{{}})"
        
        elif kind == 'first_n':
            stop = si['stop']
            return f"ROOT::VecOps::Take({vec_var}, static_cast<int>(std::min(static_cast<size_t>({stop}), {vec_var}.size())))"
        
        elif kind == 'range':
            start, stop = si['start'], si['stop']
            return f"[&]() -> {result_type} {{ size_t s = std::min(static_cast<size_t>({start}), {vec_var}.size()); size_t e = std::min(static_cast<size_t>({stop}), {vec_var}.size()); if (s >= e) return {result_type}{{}}; return ROOT::VecOps::Take({vec_var}, ROOT::VecOps::Range(s, e)); }}()"
        
        elif kind == 'from_idx':
            start = si['start']
            return f"[&]() -> {result_type} {{ size_t s = std::min(static_cast<size_t>({start}), {vec_var}.size()); if (s >= {vec_var}.size()) return {result_type}{{}}; return ROOT::VecOps::Take({vec_var}, ROOT::VecOps::Range(s, {vec_var}.size())); }}()"
        
        elif kind == 'last_n':
            start = si['start']
            return f"ROOT::VecOps::Take({vec_var}, -static_cast<int>(std::min(static_cast<size_t>(-({start})), {vec_var}.size())))"
        
        elif kind == 'step':
            start = si['start'] or "0"
            stop = si['stop']
            step = si['step']
            stop_expr = f"std::min(static_cast<size_t>({stop}), {vec_var}.size())" if stop else f"{vec_var}.size()"
            return f"[&]() -> {result_type} {{ ROOT::RVec<size_t> indices; for (size_t i = {start}; i < {stop_expr}; i += {step}) indices.push_back(i); return ROOT::VecOps::Take({vec_var}, indices); }}()"
        
        elif kind == 'reverse':
            return f"ROOT::VecOps::Reverse({vec_var})"
        
        else:
            return vec_var
    
    def _visit_1d_slice_fallback(self, node: SubscriptNode) -> str:
        """Fallback for 1D slice in SubscriptNode (should normally use RVecSliceNode)."""
        target = self._visit(node.value)
        si = node.indices[0]
        result_type = self._get_nd_result_type(node)
        slice_info = self._analyze_slice_dim(si, 0)
        return self._gen_inner_slice_expr(target, slice_info, result_type)
    
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
        elif node.slice_kind == SliceKind.RANGE_NEG:
            return self._gen_slice_range_neg(target, node)
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
        """[:n] → Take(v, min(n, size)) - first n elements with clamping.
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions
        like safe-indexed nested RVec.
        """
        n = self._visit(node.stop)
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    size_t n = std::min(static_cast<size_t>({n}), target_val.size());
    return ROOT::VecOps::Take(target_val, n);
}}()'''
    
    def _gen_slice_last_n(self, target: str, node: RVecSliceNode) -> str:
        """[-n:] → Take(v, -min(n, size)) - last n elements with clamping.
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions.
        """
        # node.start contains the negative index, e.g., -3
        # We need to extract the absolute value and clamp it
        neg_n = self._visit(node.start)  # e.g., "-3"
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    size_t abs_n = static_cast<size_t>(-({neg_n}));
    size_t clamped = std::min(abs_n, target_val.size());
    if (clamped == 0) return {elem_type}();
    return ROOT::VecOps::Take(target_val, -static_cast<int>(clamped));
}}()'''
    
    def _gen_slice_from_index(self, target: str, node: RVecSliceNode) -> str:
        """[n:] or [:] → Take(v, Range(n, size)) - from index to end.
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions.
        """
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        # Handle full slice [:] where start is None
        if node.start is None:
            start = "0"
        else:
            start = self._visit(node.start)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    size_t start = {start};
    if (start >= target_val.size()) return {elem_type}();
    return ROOT::VecOps::Take(target_val, 
        ROOT::VecOps::Range(start, target_val.size()));
}}()'''
    
    def _gen_slice_range(self, target: str, node: RVecSliceNode) -> str:
        """[a:b] → Take(v, Range(a, min(b, size))) - range with clamping.
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions.
        """
        start = self._visit(node.start)
        stop = self._visit(node.stop)
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    size_t start = {start};
    size_t stop = std::min(static_cast<size_t>({stop}), target_val.size());
    if (start >= stop) return {elem_type}();
    return ROOT::VecOps::Take(target_val, 
        ROOT::VecOps::Range(start, stop));
}}()'''
    
    def _gen_slice_range_neg(self, target: str, node: RVecSliceNode) -> str:
        """[a:b] with negative indices → Range with length-dependent translation.
        
        Phase 13.6.G: Support mixed negative indices like [1:-1], [-3:-1], etc.
        Negative indices are translated: neg_idx -> size + neg_idx
        Then clamped to [0, size] range.
        """
        start_expr = self._visit(node.start) if node.start is not None else "0"
        stop_expr = self._visit(node.stop) if node.stop is not None else "target_val.size()"
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    long long sz = static_cast<long long>(target_val.size());
    
    // Translate negative indices: neg_idx -> size + neg_idx
    long long raw_start = {start_expr};
    long long raw_stop = {stop_expr};
    
    // Handle negative indices
    long long start = raw_start < 0 ? std::max(0LL, sz + raw_start) : raw_start;
    long long stop = raw_stop < 0 ? std::max(0LL, sz + raw_stop) : raw_stop;
    
    // Clamp to valid range
    start = std::min(start, sz);
    stop = std::min(stop, sz);
    
    // Return empty if start >= stop
    if (start >= stop) return {elem_type}();
    
    return ROOT::VecOps::Take(target_val, 
        ROOT::VecOps::Range(static_cast<size_t>(start), static_cast<size_t>(stop)));
}}()'''
    
    def _gen_slice_step(self, target: str, node: RVecSliceNode) -> str:
        """[::step] or [start::step] or [start:stop:step] → loop-based indices.
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions.
        """
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
            stop_expr = f"std::min(static_cast<size_t>({stop}), target_val.size())"
        else:
            stop_expr = "target_val.size()"
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    ROOT::RVec<size_t> indices;
    size_t stop = {stop_expr};
    for (size_t i = {start}; i < stop; i += {step}) {{
        indices.push_back(i);
    }}
    return ROOT::VecOps::Take(target_val, indices);
}}()'''
    
    def _gen_slice_reverse(self, target: str, node: RVecSliceNode) -> str:
        """[::-1] → manual reverse loop (not using VecOps::Reverse).
        
        Phase 13.3.DSL: Extract target to variable to handle complex expressions.
        """
        elem_type = self._cpp_type_for_rvec(node.dtype)
        
        return f'''[&]() -> {elem_type} {{
    auto target_val = {target};
    {elem_type} result;
    result.reserve(target_val.size());
    for (size_t i = target_val.size(); i-- > 0; ) {{
        result.push_back(target_val[i]);
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
        # Note: This applies to std:: functions but NOT TMath:: (which need explicit namespace)
        if node.rank > 0:
            # Phase 11.1: For non-std namespace functions (like TMath::Sin), preserve the namespace
            # cpp_name contains full qualified name for namespace functions
            is_std_function = (node.namespace == "std" or 
                              (node.cpp_name and node.cpp_name.startswith("std::")))
            
            if not is_std_function and (node.namespace or node.cpp_name):
                # Non-std namespace function - use cpp_name which has proper qualification
                if node.cpp_name:
                    return node.cpp_name
                if node.namespace:
                    return f"{node.namespace}::{node.func}"
            
            # Standard math functions - use unqualified name for ADL
            func_name = node.func
            if "." in func_name:
                func_name = func_name.replace(".", "::")
            # Don't use std:: prefix for RVec operations
            return func_name
        
        # Check custom cpp_name first - it takes priority
        if node.cpp_name:
            # cpp_name already contains full qualified name (e.g., "std::sqrt", "TMath::Sin")
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
                
                # Phase 13.6.D: Nested RVec reductions need <limits> for NaN handling
                if self._needs_nested_rvec_handling(node):
                    func_lower = node.func.lower()
                    if func_lower in {'mean', 'min', 'max', 'stddev', 'variance'}:
                        needs_limits = True
                    needs_rvec = True
                
                # Phase 11.1: Add namespace headers
                if node.namespace:
                    # Convert C++ namespace back to dot notation for lookup
                    ns_dot = node.namespace.replace("::", ".")
                    if ns_dot in NAMESPACE_HEADERS:
                        headers.add(NAMESPACE_HEADERS[ns_dot])
                    # Check parent namespaces
                    parts = ns_dot.split(".")
                    for i in range(len(parts), 0, -1):
                        parent = ".".join(parts[:i])
                        if parent in NAMESPACE_HEADERS:
                            headers.add(NAMESPACE_HEADERS[parent])
                            break
                    
                    # Phase 11.1b: Add <algorithm> if broadcasting with multiple vectors
                    if self._needs_scalar_broadcast(node):
                        vec_count = sum(1 for arg in node.args if arg.rank > 0)
                        if vec_count > 1:
                            headers.add("<algorithm>")  # for std::min
                        needs_rvec = True  # Broadcasting produces RVec
            
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
    
    Phase 13.3.DSL: Added support for nested RVec type declarations via #pragma link.
    
    Example:
        >>> library = FunctionLibrary()
        >>> library.add(func)
        >>> library.compile("alias_pt")
        >>> rdf.Define("pt", library.get_define_expression("alias_pt"))
    """
    
    # Phase 13.3.DSL: Track declared nested RVec types (class-level for session persistence)
    _nested_rvec_declared: Set[str] = set()
    
    def __init__(self):
        """Initialize empty function library."""
        self.functions: Dict[str, GeneratedFunction] = {}
        self.compiled: Set[str] = set()
        self.headers_loaded: Set[str] = set()
    
    def add(self, func: GeneratedFunction) -> None:
        """Add function to library."""
        self.functions[func.name] = func
    
    def _declare_nested_rvec_types(self, code: str) -> None:
        """
        Phase 13.3.DSL: Declare nested RVec types via #pragma link before compilation.
        
        ROOT requires explicit instantiation of nested RVec templates.
        This detects patterns like ROOT::RVec<ROOT::RVec<double>> and
        declares them if not already done.
        
        Args:
            code: C++ code to scan for nested RVec types
        """
        import re
        try:
            import ROOT
        except ImportError:
            return
        
        # Pattern to match nested RVec types: RVec<RVec<type>>
        # Handles: ROOT::RVec<ROOT::RVec<double>>, RVec<RVec<float>>, etc.
        pattern = r'(?:ROOT::)?(?:VecOps::)?RVec<\s*(?:ROOT::)?(?:VecOps::)?RVec<\s*(\w+)\s*>\s*>'
        
        for match in re.finditer(pattern, code):
            inner_type = match.group(1)
            type_key = f"RVec<RVec<{inner_type}>>"
            
            if type_key not in FunctionLibrary._nested_rvec_declared:
                # Declare the nested RVec type
                pragma1 = f'#pragma link C++ class ROOT::RVec<ROOT::RVec<{inner_type}>>+;'
                pragma2 = f'#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<{inner_type}>>+;'
                
                # Process the pragmas
                ROOT.gInterpreter.ProcessLine(pragma1)
                ROOT.gInterpreter.ProcessLine(pragma2)
                
                FunctionLibrary._nested_rvec_declared.add(type_key)
    
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
        
        # Phase 13.3.DSL: Declare nested RVec types if needed
        self._declare_nested_rvec_types(func.code)
        
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
