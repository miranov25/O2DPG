"""
C++ code generation backend for RDataFrame DSL.

This module generates C++ helper functions from IR trees for scalar expressions.
The generated functions can be compiled via ROOT's gInterpreter and used in
RDataFrame.Define() calls.

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

Phase 5 Scope (Scalars Only):
- Arithmetic operations (+, -, *, /, %, **)
- Comparisons (<, <=, >, >=, ==, !=)
- Logical operations (and, or, not)
- Bitwise operations (&, |, ^, ~) for int/bool
- Function calls (sqrt, sin, cos, abs, TMath::*)
- Conditionals (ternary)
- Constants (numeric, boolean) and variables
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from .ir_types import IRType, IRTypeKind, IR_TO_CPP_TYPE
from .ir_nodes import (
    IRNode, ConstantNode, VariableNode, UnaryOpNode, BinaryOpNode,
    TernaryOpNode, CallNode, MethodCallNode, PropertyAccessNode,
    SubscriptNode, SliceNode, UnaryOp, BinaryOp
)
from .ir_errors import IRError, IRErrorKind

__all__ = [
    'CppCodeGenerator',
    'GeneratedFunction',
    'FunctionLibrary',
    'FUNCTION_HEADERS',
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
    """
    name: str
    code: str
    inputs: List[Tuple[str, str]]
    return_type: str
    headers: List[str]
    ir: Optional[IRNode] = None
    
    def get_call_expression(self) -> str:
        """Get expression for RDataFrame.Define()."""
        args = ", ".join(name for name, _ in self.inputs)
        return f"{self.name}({args})"
    
    def get_signature(self) -> str:
        """Get function signature without body."""
        params = ", ".join(f"{cpp_type} {name}" for name, cpp_type in self.inputs)
        return f"{self.return_type} {self.name}({params})"
    
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
                 error_detail: str = "full"):
        """
        Initialize code generator.
        
        Args:
            type_inferrer: TypeInferrer for looking up variable types
            reflection_cache: ReflectionCache for method/property types
            error_detail: Level of detail in error messages ("full", "summary", "minimal")
        """
        self.type_inferrer = type_inferrer
        self.reflection_cache = reflection_cache
        self.error_detail = error_detail
        self._existing_names: Set[str] = set()
    
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
        """Validate IR tree for Phase 5 scalar support."""
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
            
            # Check for unsupported node types in Phase 5
            if isinstance(node, MethodCallNode):
                raise IRError(
                    IRErrorKind.UNSUPPORTED_OP,
                    "Method calls on objects are not supported in Phase 5",
                    suggestions=["Method call support will be added in Phase 6"]
                )
            
            if isinstance(node, PropertyAccessNode):
                raise IRError(
                    IRErrorKind.UNSUPPORTED_OP,
                    "Property access on objects is not supported in Phase 5",
                    suggestions=["Property access support will be added in Phase 6"]
                )
            
            if isinstance(node, SubscriptNode):
                raise IRError(
                    IRErrorKind.UNSUPPORTED_OP,
                    "Subscript/slicing operations are not supported in Phase 5",
                    suggestions=["Subscript support will be added in Phase 6"]
                )
            
            # Check rank
            if node.rank > 0 and not isinstance(node, (SliceNode,)):
                raise IRError(
                    IRErrorKind.UNSUPPORTED_OP,
                    f"Vector operations (rank > 0) are not supported in Phase 5",
                    suggestions=["Vector support will be added in Phase 6"]
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
        
        return node.dtype.to_cpp()
    
    def _get_cpp_return_type(self, ir: IRNode) -> str:
        """Get C++ return type for the expression."""
        if ir.dtype.kind == IRTypeKind.Unknown:
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                "Cannot determine return type for expression with Unknown type",
                suggestions=["Check that all sub-expressions have valid types"]
            )
        
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
    
    def _cpp_function_name(self, node: CallNode) -> str:
        """Convert DSL function name to C++ function name."""
        # Check if it's a namespaced function (e.g., TMath.Gaus)
        if node.namespace:
            # TMath.Gaus -> TMath::Gaus
            return f"{node.namespace}::{node.cpp_name or node.func}"
        
        # Check custom cpp_name first
        if node.cpp_name:
            return node.cpp_name
        
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
        Save all functions to .C macro file.
        
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
    
    def clear(self) -> None:
        """Clear all functions (does not unload from interpreter)."""
        self.functions.clear()
        self.compiled.clear()
    
    def __len__(self) -> int:
        return len(self.functions)
    
    def __contains__(self, name: str) -> bool:
        return name in self.functions
