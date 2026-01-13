"""
IRBuilder: Convert Python AST to IR nodes.

This module provides the IRBuilder class which:
1. Parses Python expression strings
2. Converts AST nodes to IR nodes
3. Infers types during construction
4. Validates operations (rank compatibility, etc.)

Usage:
    inferrer = TypeInferrer.from_schema(schema)
    builder = IRBuilder(inferrer)
    
    ir_node = builder.build("sqrt(px**2 + py**2)")
    print(ir_node.dtype, ir_node.rank)

Supported Python syntax:
- Variables: px, py, track.pt
- Literals: 42, 3.14, True
- Arithmetic: +, -, *, /, //, %, **
- Comparison: <, <=, >, >=, ==, !=
- Logical: and, or, not
- Bitwise: &, |, ^, ~
- Calls: sqrt(x), TMath.Gaus(x, 0, 1)
- Methods: track.getX(), track.Pt()
- Attributes: track.mPx
- Subscripts: arr[0], arr[1:3], arr[:]
- Conditionals: x if cond else y
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from .ir_types import (
    IRType, IRTypeKind, 
    promote_types, comparison_result_type, division_result_type,
    cpp_type_to_ir
)
from .ir_nodes import (
    IRNode, ConstantNode, VariableNode,
    UnaryOp, UnaryOpNode, BinaryOp, BinaryOpNode, TernaryOpNode,
    CallNode, MethodCallNode, PropertyAccessNode,
    MethodBroadcastNode, PropertyBroadcastNode,  # Phase 8
    SliceNode, SubscriptNode, CollectionIndexNode,
    SliceKind, RVecSliceNode,
    make_constant, make_variable, make_binary_op, make_unary_op,
    make_call, make_method_call, make_subscript, make_rvec_slice
)
from .type_inferrer import extract_inner_type, is_collection_type
from .reflection import (
    ReflectionCache, 
    get_method_return_type_fallback, 
    get_property_type_fallback,
    get_known_methods
)
from .ir_errors import (
    IRError, IRErrorKind, SourceLocation, ErrorCollector,
    type_mismatch_error, unknown_variable_error, 
    unsupported_operation_error, rank_mismatch_error
)
from .type_inferrer import TypeInferrer, VariableInfo

__all__ = [
    'IRBuilder',
    'BuildContext',
]


# =============================================================================
# Known Functions
# =============================================================================

# Math functions that map to C++ std:: or ROOT::Math::
KNOWN_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    # Standard math (C++ std::)
    "sqrt": {"cpp_name": "std::sqrt", "return_type": IRTypeKind.Float64},
    "abs": {"cpp_name": "std::abs", "return_type": None},  # Same as input
    "fabs": {"cpp_name": "std::fabs", "return_type": IRTypeKind.Float64},
    "exp": {"cpp_name": "std::exp", "return_type": IRTypeKind.Float64},
    "log": {"cpp_name": "std::log", "return_type": IRTypeKind.Float64},
    "log10": {"cpp_name": "std::log10", "return_type": IRTypeKind.Float64},
    "log2": {"cpp_name": "std::log2", "return_type": IRTypeKind.Float64},
    "sin": {"cpp_name": "std::sin", "return_type": IRTypeKind.Float64},
    "cos": {"cpp_name": "std::cos", "return_type": IRTypeKind.Float64},
    "tan": {"cpp_name": "std::tan", "return_type": IRTypeKind.Float64},
    "asin": {"cpp_name": "std::asin", "return_type": IRTypeKind.Float64},
    "acos": {"cpp_name": "std::acos", "return_type": IRTypeKind.Float64},
    "atan": {"cpp_name": "std::atan", "return_type": IRTypeKind.Float64},
    "atan2": {"cpp_name": "std::atan2", "return_type": IRTypeKind.Float64},
    "sinh": {"cpp_name": "std::sinh", "return_type": IRTypeKind.Float64},
    "cosh": {"cpp_name": "std::cosh", "return_type": IRTypeKind.Float64},
    "tanh": {"cpp_name": "std::tanh", "return_type": IRTypeKind.Float64},
    "pow": {"cpp_name": "std::pow", "return_type": IRTypeKind.Float64},
    "floor": {"cpp_name": "std::floor", "return_type": IRTypeKind.Float64},
    "ceil": {"cpp_name": "std::ceil", "return_type": IRTypeKind.Float64},
    "round": {"cpp_name": "std::round", "return_type": IRTypeKind.Float64},
    "min": {"cpp_name": "std::min", "return_type": None},  # Promoted type
    "max": {"cpp_name": "std::max", "return_type": None},  # Promoted type
    
    # ROOT TMath functions
    "TMath.Gaus": {"cpp_name": "TMath::Gaus", "return_type": IRTypeKind.Float64, 
                   "headers": ["TMath.h"]},
    "TMath.Landau": {"cpp_name": "TMath::Landau", "return_type": IRTypeKind.Float64,
                    "headers": ["TMath.h"]},
    "TMath.Abs": {"cpp_name": "TMath::Abs", "return_type": None},
    "TMath.Sqrt": {"cpp_name": "TMath::Sqrt", "return_type": IRTypeKind.Float64},
    
    # RVec operations (will be expanded in Phase 7)
    # Phase 10.5: Reduction functions return scalar (rank=0)
    "Sum": {"cpp_name": "ROOT::VecOps::Sum", "return_type": None, "is_reduction": True},  # Element type
    "Mean": {"cpp_name": "ROOT::VecOps::Mean", "return_type": IRTypeKind.Float64, "is_reduction": True},
    "StdDev": {"cpp_name": "ROOT::VecOps::StdDev", "return_type": IRTypeKind.Float64, "is_reduction": True},
    "Var": {"cpp_name": "ROOT::VecOps::Var", "return_type": IRTypeKind.Float64, "is_reduction": True},
    "Min": {"cpp_name": "ROOT::VecOps::Min", "return_type": None, "is_reduction": True},
    "Max": {"cpp_name": "ROOT::VecOps::Max", "return_type": None, "is_reduction": True},
    "Any": {"cpp_name": "ROOT::VecOps::Any", "return_type": IRTypeKind.Bool, "is_reduction": True},
    "All": {"cpp_name": "ROOT::VecOps::All", "return_type": IRTypeKind.Bool, "is_reduction": True},
    "ArgMin": {"cpp_name": "ROOT::VecOps::ArgMin", "return_type": IRTypeKind.UInt64, "is_reduction": True},
    "ArgMax": {"cpp_name": "ROOT::VecOps::ArgMax", "return_type": IRTypeKind.UInt64, "is_reduction": True},
    "Sort": {"cpp_name": "ROOT::VecOps::Sort", "return_type": None},  # Same as input (not reduction)
    "Reverse": {"cpp_name": "ROOT::VecOps::Reverse", "return_type": None},
    
    # Phase 12.2: RVec selection functions
    "Take": {"cpp_name": "ROOT::VecOps::Take", "return_type": None, "is_selection": True},
    "Range": {"cpp_name": "ROOT::VecOps::Range", "return_type": None, "is_selection": True},
    "Where": {"cpp_name": "ROOT::VecOps::Where", "return_type": None, "is_selection": True},
    "IndicesFromOffsets": {"cpp_name": "IndicesFromOffsets", "return_type": None, "is_selection": True},
}


# =============================================================================
# Phase 13.5.D: Numeric Widening Conversion Matrix
# =============================================================================

# Conversion ranks:
#   0  = Exact match
#   1  = Promotion (safe widening within type family)
#   2  = Conversion (cross-type, e.g., int→double)
#   -1 = Forbidden (narrowing, lossy, or incompatible)
#
# Key decisions (unanimous 7/7 reviewers):
#   - Int→Float32 is FORBIDDEN (lossy for values > 16,777,216)
#   - Signed↔Unsigned is FORBIDDEN (ambiguous semantics)
#   - Bool conversions are FORBIDDEN (exact match only)
#   - Narrowing is ALWAYS FORBIDDEN

CONVERSION_MATRIX: Dict[IRTypeKind, Dict[IRTypeKind, int]] = {
    # Signed integers: can widen within family, convert to Float64
    IRTypeKind.Int8: {
        IRTypeKind.Int8: 0, IRTypeKind.Int16: 1, IRTypeKind.Int32: 1, IRTypeKind.Int64: 1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.Int16: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: 0, IRTypeKind.Int32: 1, IRTypeKind.Int64: 1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.Int32: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: 0, IRTypeKind.Int64: 1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.Int64: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: 0,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    # Unsigned integers: can widen within family, convert to Float64
    IRTypeKind.UInt8: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: 0, IRTypeKind.UInt16: 1, IRTypeKind.UInt32: 1, IRTypeKind.UInt64: 1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.UInt16: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: 0, IRTypeKind.UInt32: 1, IRTypeKind.UInt64: 1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.UInt32: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: 0, IRTypeKind.UInt64: 1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.UInt64: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: 0,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 2, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    # Floating point: Float32→Float64 is promotion, no narrowing
    IRTypeKind.Float32: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: 0, IRTypeKind.Float64: 1, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    IRTypeKind.Float64: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: 0, IRTypeKind.Bool: -1, IRTypeKind.Object: -1,
    },
    # Bool: exact match only
    IRTypeKind.Bool: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: -1, IRTypeKind.Bool: 0, IRTypeKind.Object: -1,
    },
    # Object: exact match only
    IRTypeKind.Object: {
        IRTypeKind.Int8: -1, IRTypeKind.Int16: -1, IRTypeKind.Int32: -1, IRTypeKind.Int64: -1,
        IRTypeKind.UInt8: -1, IRTypeKind.UInt16: -1, IRTypeKind.UInt32: -1, IRTypeKind.UInt64: -1,
        IRTypeKind.Float32: -1, IRTypeKind.Float64: -1, IRTypeKind.Bool: -1, IRTypeKind.Object: 0,
    },
}

@dataclass
class BuildContext:
    """
    Context for building IR from expressions.
    
    Tracks current alias being built, namespace, and error state.
    """
    alias_name: Optional[str] = None  # Name of alias being defined
    namespace: Optional[str] = None   # Subframe namespace (e.g., "tracks")
    expression_text: str = ""         # Original expression for error messages
    in_subscript: bool = False        # Currently inside subscript
    dependencies: Set[str] = field(default_factory=set)  # Variables used
    
    def with_namespace(self, ns: str) -> 'BuildContext':
        """Return new context with updated namespace."""
        return BuildContext(
            alias_name=self.alias_name,
            namespace=ns,
            expression_text=self.expression_text,
            in_subscript=self.in_subscript,
            dependencies=self.dependencies,
        )
    
    def with_subscript(self) -> 'BuildContext':
        """Return new context marking we're in a subscript."""
        return BuildContext(
            alias_name=self.alias_name,
            namespace=self.namespace,
            expression_text=self.expression_text,
            in_subscript=True,
            dependencies=self.dependencies,
        )


# =============================================================================
# AST Operator Mapping
# =============================================================================

# Map Python AST operators to IR operators
AST_BINOP_MAP = {
    ast.Add: BinaryOp.ADD,
    ast.Sub: BinaryOp.SUB,
    ast.Mult: BinaryOp.MUL,
    ast.Div: BinaryOp.DIV,
    ast.FloorDiv: BinaryOp.FLOORDIV,
    ast.Mod: BinaryOp.MOD,
    ast.Pow: BinaryOp.POW,
    ast.BitAnd: BinaryOp.BITAND,
    ast.BitOr: BinaryOp.BITOR,
    ast.BitXor: BinaryOp.BITXOR,
}

AST_CMPOP_MAP = {
    ast.Lt: BinaryOp.LT,
    ast.LtE: BinaryOp.LE,
    ast.Gt: BinaryOp.GT,
    ast.GtE: BinaryOp.GE,
    ast.Eq: BinaryOp.EQ,
    ast.NotEq: BinaryOp.NE,
}

AST_BOOLOP_MAP = {
    ast.And: BinaryOp.AND,
    ast.Or: BinaryOp.OR,
}

AST_UNARYOP_MAP = {
    ast.UAdd: UnaryOp.POS,
    ast.USub: UnaryOp.NEG,
    ast.Not: UnaryOp.NOT,
    ast.Invert: UnaryOp.BITNOT,
}


# =============================================================================
# IRBuilder
# =============================================================================

class IRBuilder:
    """
    Build IR trees from Python expressions.
    
    The IRBuilder parses Python expression strings and converts them
    to IR node trees, inferring types along the way.
    
    Example:
        >>> inferrer = TypeInferrer.from_schema({"columns": {"px": "float"}})
        >>> builder = IRBuilder(inferrer)
        >>> node = builder.build("sqrt(px**2)")
        >>> print(node.dtype.kind)  # Float64
    """
    
    def __init__(self, type_inferrer: TypeInferrer, 
                 error_collector: ErrorCollector = None):
        """
        Initialize builder with type information.
        
        Args:
            type_inferrer: TypeInferrer with column/alias types
            error_collector: Optional error collector for batch processing
        """
        self.inferrer = type_inferrer
        self.errors = error_collector or ErrorCollector()
        self._custom_functions: Dict[str, List[Dict]] = {}  # Phase 13.5.C: List for overloads
        # Phase 11.1c: Cache for namespace resolution (positive results only)
        self._namespace_cache: Dict[str, bool] = {}
    
    def register_function(
        self, 
        name: str, 
        cpp_name: str,
        return_type: 'IRTypeKind' = None,
        headers: List[str] = None,
        param_types: List[Dict] = None,
    ) -> None:
        """
        Register a custom function mapping.
        
        Phase 13.5.C v0.5: Supports multiple overloads per name.
        
        Args:
            name: Python function name (e.g., "pt")
            cpp_name: C++ function name (e.g., "dsl_pt_abc123")
            return_type: Return type. If None, defaults to Object with warning.
            headers: Required C++ headers
            param_types: List of parameter type info for overload resolution.
                         Each entry: {'name', 'cpp_type', 'rank', 'ir_kind'}
                         For zero-parameter functions, pass empty list [].
                         
        Raises:
            ValueError: If param_types is None (must be explicit list)
        """
        import warnings
        
        # Phase 13.5.C: param_types is REQUIRED for overload resolution
        if param_types is None:
            raise ValueError(
                f"Custom function '{name}' requires param_types for overload resolution. "
                f"Got: None. Note: For zero-parameter functions, pass empty list []."
            )
        
        # v0.5 FIX (P0-3): Handle None return_type with warning
        if return_type is None:
            warnings.warn(
                f"Function '{name}' registered without return_type. "
                f"Defaulting to Object. Specify return_type for better type inference.",
                UserWarning
            )
            return_type = IRTypeKind.Object
        
        if name not in self._custom_functions:
            self._custom_functions[name] = []
        
        self._custom_functions[name].append({
            "cpp_name": cpp_name,
            "return_type": return_type,
            "headers": headers or [],
            "param_types": param_types,
        })
    
    def build(self, expression: str, 
              alias_name: str = None,
              namespace: str = None) -> IRNode:
        """
        Parse expression and build IR tree.
        
        Args:
            expression: Python expression string
            alias_name: Name of alias being defined (for error messages)
            namespace: Optional namespace for variable lookup
            
        Returns:
            Root IRNode of the expression tree
            
        Raises:
            IRError: If parsing or type inference fails
        """
        context = BuildContext(
            alias_name=alias_name,
            namespace=namespace,
            expression_text=expression,
        )
        
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise IRError(
                IRErrorKind.PARSE_ERROR,
                f"Invalid Python syntax: {e.msg}",
                source_location=SourceLocation(
                    expr_name=alias_name or "<expression>",
                    text_span=expression,
                    line=e.lineno or 1,
                    column=e.offset or 0,
                )
            )
        
        return self._visit(tree.body, context)
    
    def build_multiple(self, aliases: Dict[str, str]) -> Dict[str, IRNode]:
        """
        Build IR for multiple aliases, handling dependencies.
        
        Args:
            aliases: Dict of {name: expression}
            
        Returns:
            Dict of {name: IRNode}
        """
        results = {}
        
        # Simple approach: try each alias, retry failed ones
        remaining = dict(aliases)
        max_iterations = len(aliases) + 1
        
        for _ in range(max_iterations):
            if not remaining:
                break
            
            progress = False
            failed = {}
            
            for name, expr in remaining.items():
                try:
                    node = self.build(expr, alias_name=name)
                    results[name] = node
                    # Register alias type for dependent aliases
                    self.inferrer.register_alias(
                        name, node.dtype, node.rank, node.is_jagged
                    )
                    progress = True
                except IRError as e:
                    if self.errors.should_process(name, set()):
                        failed[name] = expr
                        self.errors.add(e)
            
            remaining = failed
            if not progress:
                break
        
        return results
    
    # =========================================================================
    # Broadcasting Detection Helpers (Phase 8)
    # =========================================================================
    
    # Scalar types that don't broadcast
    _SCALAR_TYPES = frozenset({
        'double', 'float', 'int', 'long', 'short', 'char',
        'unsigned int', 'unsigned long', 'unsigned short', 'unsigned char',
        'bool', 'size_t', 'int32_t', 'int64_t', 'uint32_t', 'uint64_t',
        'Double_t', 'Float_t', 'Int_t', 'Long_t', 'Bool_t',
    })
    
    def _is_rvec_of_objects(self, node: IRNode) -> bool:
        """
        Check if node is RVec<Object> (not RVec<scalar>).
        
        Returns True for RVec<TLorentzVector>, False for RVec<double>.
        
        Handles two schema formats:
        - Full: dtype.cpp_type = 'RVec<TLorentzVector>'
        - DSLCompiler: dtype.cpp_type = 'TLorentzVector', rank = 1
        """
        if node.rank != 1:
            return False
        
        # Get cpp_type - try multiple sources
        cpp_type = ""
        if node.dtype.cpp_type:
            cpp_type = node.dtype.cpp_type
        elif hasattr(node.dtype, 'to_cpp'):
            cpp_type = node.dtype.to_cpp()
        
        if not cpp_type:
            return False
        
        # Case 1: cpp_type is already a collection type (RVec<T>, vector<T>)
        if is_collection_type(cpp_type):
            element_type, _ = extract_inner_type(cpp_type)
            return element_type not in self._SCALAR_TYPES
        
        # Case 2: cpp_type is the element type itself (rank=1 implies RVec)
        # This happens when DSLCompiler passes dtype as element type
        return cpp_type not in self._SCALAR_TYPES
    
    def _get_rvec_element_type(self, node: IRNode) -> Optional[str]:
        """
        Extract element type from RVec node.
        
        For RVec<TLorentzVector>, returns "TLorentzVector".
        For RVec<double>, returns "double".
        
        Handles two schema formats:
        - Full: dtype.cpp_type = 'RVec<TLorentzVector>'
        - DSLCompiler: dtype.cpp_type = 'TLorentzVector', rank = 1
        """
        # Get cpp_type - try multiple sources
        cpp_type = ""
        if node.dtype.cpp_type:
            cpp_type = node.dtype.cpp_type
        elif hasattr(node.dtype, 'to_cpp'):
            cpp_type = node.dtype.to_cpp()
        
        if not cpp_type:
            return None
        
        # Case 1: cpp_type is already a collection type
        if is_collection_type(cpp_type):
            element_type, _ = extract_inner_type(cpp_type)
            return element_type
        
        # Case 2: cpp_type is the element type itself (rank=1 implies it's the element)
        if node.rank == 1:
            return cpp_type
        
        return None
    
    def _resolve_broadcast_method(self, element_type: str, method_name: str,
                                   ctx: BuildContext, node: ast.AST) -> Tuple[str, IRType]:
        """
        Resolve method return type for broadcasting.
        
        Args:
            element_type: C++ type of RVec elements (e.g., "TLorentzVector")
            method_name: Method name (e.g., "Pt")
            ctx: Build context for error location
            node: AST node for error location
            
        Returns:
            (return_type_str, ir_type) tuple
            
        Raises:
            IRError: If method not found
        """
        # Try reflection cache first
        try:
            cache = ReflectionCache()
            method_info = cache.resolve_method(element_type, method_name)
            return_type_str = method_info.return_type
        except IRError:
            # Fall back to hardcoded map
            return_type_str = get_method_return_type_fallback(element_type, method_name)
            
            if return_type_str is None:
                # Method not found - give helpful error
                known = get_known_methods(element_type)
                suggestions = []
                if known:
                    suggestions.append(f"Did you mean: {', '.join(known[:7])}?")
                suggestions.append(f"Note: Broadcasting '{method_name}()' on RVec<{element_type}>")
                
                raise IRError(
                    IRErrorKind.TYPE_ERROR,
                    f"Method '{method_name}' not found on element type '{element_type}'",
                    source_location=self._make_location(node, ctx),
                    suggestions=suggestions
                )
        
        # Convert to IRType
        ir_type = cpp_type_to_ir(return_type_str)
        return return_type_str, ir_type
    
    def _resolve_broadcast_property(self, element_type: str, property_name: str,
                                     ctx: BuildContext, node: ast.AST) -> Tuple[str, IRType, str]:
        """
        Resolve property type for broadcasting.
        
        Args:
            element_type: C++ type of RVec elements
            property_name: Property name (e.g., "fPx")
            ctx: Build context for error location
            node: AST node for error location
            
        Returns:
            (property_type_str, ir_type, access_mode) tuple
            
        Raises:
            IRError: If property not found
        """
        access_mode = "direct"
        property_type_str = None
        
        # Try reflection cache first
        try:
            cache = ReflectionCache()
            prop_info = cache.resolve_property(element_type, property_name)
            property_type_str = prop_info.property_type
            # Note: access_mode could be determined by reflection in future
        except IRError:
            # Fall back to hardcoded map
            property_type_str = get_property_type_fallback(element_type, property_name)
        
        if property_type_str is None:
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"Property '{property_name}' not found on element type '{element_type}'",
                source_location=self._make_location(node, ctx),
                suggestions=[
                    f"Note: Broadcasting '{property_name}' on RVec<{element_type}>",
                    "Check that the property name is correct"
                ]
            )
        
        ir_type = cpp_type_to_ir(property_type_str)
        return property_type_str, ir_type, access_mode
    
    # =========================================================================
    # AST Visitor Methods
    # =========================================================================
    
    def _visit(self, node: ast.AST, ctx: BuildContext) -> IRNode:
        """Dispatch to appropriate visitor method."""
        method_name = f"_visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, None)
        
        if visitor is None:
            raise unsupported_operation_error(
                f"Unsupported syntax: {node.__class__.__name__}",
                location=self._make_location(node, ctx),
            )
        
        return visitor(node, ctx)
    
    def _visit_Constant(self, node: ast.Constant, ctx: BuildContext) -> IRNode:
        """Handle literal constants (42, 3.14, True, "str")."""
        value = node.value
        
        if isinstance(value, bool):
            return make_constant(value)
        elif isinstance(value, int):
            return make_constant(value)
        elif isinstance(value, float):
            return make_constant(value)
        elif isinstance(value, str):
            # String constants - rare but possible
            return ConstantNode(
                value=value,
                dtype=IRType(IRTypeKind.Object, "std::string"),
                source_location=self._make_location(node, ctx),
            )
        else:
            raise unsupported_operation_error(
                f"Unsupported constant type: {type(value).__name__}",
                location=self._make_location(node, ctx),
            )
    
    def _visit_Num(self, node: ast.Num, ctx: BuildContext) -> IRNode:
        """Handle numbers (Python 3.7 compatibility)."""
        return make_constant(node.n)
    
    def _visit_Name(self, node: ast.Name, ctx: BuildContext) -> IRNode:
        """Handle variable references."""
        name = node.id
        ctx.dependencies.add(name)
        
        # Check if variable exists
        if not self.inferrer.has_variable(name, ctx.namespace):
            similar = self.inferrer._find_similar_names(name)
            raise unknown_variable_error(
                name, 
                location=self._make_location(node, ctx),
                similar_names=similar
            )
        
        info = self.inferrer.get_variable_info(name, ctx.namespace)
        
        return VariableNode(
            name=name,
            dtype=info.dtype,
            rank=info.rank,
            is_jagged=info.is_jagged,
            namespace=ctx.namespace,
            cpp_type=info.cpp_type,
            source_location=self._make_location(node, ctx),
        )
    
    def _visit_BinOp(self, node: ast.BinOp, ctx: BuildContext) -> IRNode:
        """Handle binary operations (+, -, *, /, etc.)."""
        left = self._visit(node.left, ctx)
        right = self._visit(node.right, ctx)
        
        op_type = type(node.op)
        if op_type not in AST_BINOP_MAP:
            raise unsupported_operation_error(
                f"Unsupported operator: {op_type.__name__}",
                location=self._make_location(node, ctx),
            )
        
        ir_op = AST_BINOP_MAP[op_type]
        
        # Type inference for result
        if ir_op == BinaryOp.DIV:
            result_type = division_result_type(left.dtype, right.dtype)
        else:
            result_type = promote_types(left.dtype, right.dtype)
        
        if result_type is None:
            raise type_mismatch_error(
                str(left.dtype), str(right.dtype),
                f"Cannot apply {ir_op.name} to these types",
                location=self._make_location(node, ctx),
            )
        
        # Rank broadcasting
        result_rank = max(left.rank, right.rank)
        result_jagged = left.is_jagged or right.is_jagged
        
        return BinaryOpNode(
            op=ir_op,
            left=left,
            right=right,
            dtype=result_type,
            rank=result_rank,
            is_jagged=result_jagged,
            source_location=self._make_location(node, ctx),
        )
    
    def _visit_Compare(self, node: ast.Compare, ctx: BuildContext) -> IRNode:
        """Handle comparison operations (<, <=, >, >=, ==, !=)."""
        # Handle chained comparisons: a < b < c → (a < b) and (b < c)
        if len(node.ops) > 1:
            return self._build_chained_comparison(node, ctx)
        
        left = self._visit(node.left, ctx)
        right = self._visit(node.comparators[0], ctx)
        
        op_type = type(node.ops[0])
        if op_type not in AST_CMPOP_MAP:
            raise unsupported_operation_error(
                f"Unsupported comparison: {op_type.__name__}",
                location=self._make_location(node, ctx),
            )
        
        ir_op = AST_CMPOP_MAP[op_type]
        result_type = comparison_result_type()
        
        # Rank broadcasting
        result_rank = max(left.rank, right.rank)
        result_jagged = left.is_jagged or right.is_jagged
        
        return BinaryOpNode(
            op=ir_op,
            left=left,
            right=right,
            dtype=result_type,
            rank=result_rank,
            is_jagged=result_jagged,
            source_location=self._make_location(node, ctx),
        )
    
    def _build_chained_comparison(self, node: ast.Compare, 
                                   ctx: BuildContext) -> IRNode:
        """Build chained comparison: a < b < c → (a < b) and (b < c)."""
        parts = []
        left = self._visit(node.left, ctx)
        
        for op, comparator in zip(node.ops, node.comparators):
            right = self._visit(comparator, ctx)
            
            op_type = type(op)
            if op_type not in AST_CMPOP_MAP:
                raise unsupported_operation_error(
                    f"Unsupported comparison: {op_type.__name__}",
                    location=self._make_location(node, ctx),
                )
            
            ir_op = AST_CMPOP_MAP[op_type]
            
            cmp_node = BinaryOpNode(
                op=ir_op,
                left=left,
                right=right,
                dtype=comparison_result_type(),
                rank=max(left.rank, right.rank),
                is_jagged=left.is_jagged or right.is_jagged,
            )
            parts.append(cmp_node)
            left = right
        
        # Combine with AND
        result = parts[0]
        for part in parts[1:]:
            result = BinaryOpNode(
                op=BinaryOp.AND,
                left=result,
                right=part,
                dtype=comparison_result_type(),
                rank=max(result.rank, part.rank),
                is_jagged=result.is_jagged or part.is_jagged,
            )
        
        return result
    
    def _visit_BoolOp(self, node: ast.BoolOp, ctx: BuildContext) -> IRNode:
        """Handle boolean operations (and, or)."""
        op_type = type(node.op)
        if op_type not in AST_BOOLOP_MAP:
            raise unsupported_operation_error(
                f"Unsupported boolean op: {op_type.__name__}",
                location=self._make_location(node, ctx),
            )
        
        ir_op = AST_BOOLOP_MAP[op_type]
        
        # Build left-to-right chain
        values = [self._visit(v, ctx) for v in node.values]
        
        result = values[0]
        for right in values[1:]:
            result = BinaryOpNode(
                op=ir_op,
                left=result,
                right=right,
                dtype=IRType(IRTypeKind.Bool),
                rank=max(result.rank, right.rank),
                is_jagged=result.is_jagged or right.is_jagged,
                source_location=self._make_location(node, ctx),
            )
        
        return result
    
    def _visit_UnaryOp(self, node: ast.UnaryOp, ctx: BuildContext) -> IRNode:
        """Handle unary operations (-, +, not, ~)."""
        operand = self._visit(node.operand, ctx)
        
        op_type = type(node.op)
        if op_type not in AST_UNARYOP_MAP:
            raise unsupported_operation_error(
                f"Unsupported unary op: {op_type.__name__}",
                location=self._make_location(node, ctx),
            )
        
        ir_op = AST_UNARYOP_MAP[op_type]
        
        # Constant folding for numeric literals (Phase 6.9 bug fix)
        # Fold -1 → ConstantNode(-1) instead of UnaryOpNode(NEG, ConstantNode(1))
        # This is critical for negative indexing: pt[-1] must recognize -1 as negative
        if isinstance(operand, ConstantNode):
            if ir_op == UnaryOp.NEG and isinstance(operand.value, (int, float)):
                # -constant → constant with negated value
                return ConstantNode(
                    value=-operand.value,
                    dtype=operand.dtype,
                    rank=operand.rank,
                    is_jagged=operand.is_jagged,
                    source_location=self._make_location(node, ctx),
                )
            elif ir_op == UnaryOp.POS and isinstance(operand.value, (int, float)):
                # +constant → constant (no change)
                return operand
            elif ir_op == UnaryOp.NOT and isinstance(operand.value, bool):
                # not True → False, not False → True
                return ConstantNode(
                    value=not operand.value,
                    dtype=IRType(IRTypeKind.Bool),
                    rank=0,
                    is_jagged=False,
                    source_location=self._make_location(node, ctx),
                )
        
        # Type inference for non-constant operands
        if ir_op == UnaryOp.NOT:
            result_type = IRType(IRTypeKind.Bool)
        elif ir_op == UnaryOp.BITNOT:
            if not operand.dtype.is_int():
                raise type_mismatch_error(
                    "integer", str(operand.dtype),
                    "Bitwise NOT requires integer type",
                    location=self._make_location(node, ctx),
                )
            result_type = operand.dtype
        else:  # NEG, POS
            result_type = operand.dtype
        
        return UnaryOpNode(
            op=ir_op,
            operand=operand,
            dtype=result_type,
            rank=operand.rank,
            is_jagged=operand.is_jagged,
            source_location=self._make_location(node, ctx),
        )
    
    def _visit_Call(self, node: ast.Call, ctx: BuildContext) -> IRNode:
        """Handle function calls: sqrt(x), TMath.Gaus(x, 0, 1), TMath.Pi()."""
        # Phase 11.1: Check if this is a namespace call (e.g., TMath.Pi())
        if isinstance(node.func, ast.Attribute):
            namespace_info = self._extract_namespace_chain(node.func)
            if namespace_info:
                namespace, func_name = namespace_info
                return self._build_namespace_call(namespace, func_name, node.args, ctx, node)
            
            # Phase 11.1c: Check if this looks like an unknown namespace call
            # If the base name is not in schema and not resolved by ROOT, give helpful error
            chain = self._collect_attribute_chain(node.func)
            if chain and len(chain) >= 2:
                base_name = chain[0]
                if not self.inferrer.has_variable(base_name):
                    # This looks like a namespace call but wasn't resolved
                    raise IRError(
                        IRErrorKind.REFLECTION_ERROR,
                        f"Unknown symbol '{base_name}'.\n"
                        f"Not found in schema or ROOT.\n"
                        f"Did you forget to load the library?",
                        suggestions=[
                            f"ROOT.gSystem.Load('lib{base_name}')",
                            f"ROOT.gInterpreter.ProcessLine('#include \"{base_name}.h\"')",
                            f"Or add '{base_name}' to the schema if it's a variable",
                        ],
                        source_location=self._make_location(node, ctx),
                    )
        
        # Get function name
        func_name = self._get_call_name(node.func)
        
        # Phase 13.5.C: Build argument nodes FIRST for overload resolution
        args = [self._visit(arg, ctx) for arg in node.args]
        
        # Check if it's a known function (pass args for overload resolution)
        func_info = self._get_function_info(func_name, args) if func_name else None
        
        if func_info is None:
            # Not a known function - treat as method call
            return self._visit_method_call(node, ctx)
        
        # Phase 12.2: Special handling for selection functions
        if func_info.get("is_selection"):
            result_type, result_rank = self._infer_selection_function_type(
                func_name, args, ctx, node
            )
            result_jagged = any(arg.is_jagged for arg in args) if args else False
            
            # Parse namespace from cpp_name
            cpp_name = func_info["cpp_name"]
            namespace = None
            if "::" in cpp_name:
                parts = cpp_name.rsplit("::", 1)
                namespace = parts[0]
            
            return CallNode(
                func=func_name,
                args=args,
                dtype=result_type,
                rank=result_rank,
                is_jagged=result_jagged,
                namespace=namespace,
                cpp_name=func_info["cpp_name"],
                headers=func_info.get("headers", []),
                source_location=self._make_location(node, ctx),
            )
        
        # Infer return type
        return_kind = func_info.get("return_type")
        if return_kind is None and args:
            # Infer from arguments (e.g., abs, min, max, Sum, Min, Max)
            result_type = args[0].dtype
            # For reductions, extract element type if input is RVec
            if func_info.get("is_reduction") and args[0].rank > 0:
                # Extract element type from RVec<T>
                cpp_type = args[0].dtype.cpp_type or ""
                inner = None
                if cpp_type.startswith("RVec<") or "RVec<" in cpp_type:
                    if cpp_type.startswith("ROOT::RVec<"):
                        inner = cpp_type[11:-1]
                    elif cpp_type.startswith("RVec<"):
                        inner = cpp_type[5:-1]
                    else:
                        inner = cpp_type
                    # Map to IRType
                    from .ir_types import cpp_type_to_ir
                    result_type = cpp_type_to_ir(inner)
                else:
                    # No cpp_type string, but rank > 0 means it's a vector
                    # Use the dtype directly as the element type
                    result_type = args[0].dtype
                    
                # Special case: Sum(RVec<bool>) returns int (count of true values)
                if func_name == "Sum" and result_type.kind == IRTypeKind.Bool:
                    result_type = IRType(IRTypeKind.Int32)
            else:
                for arg in args[1:]:
                    result_type = promote_types(result_type, arg.dtype) or result_type
        else:
            result_type = IRType(return_kind) if return_kind else IRType(IRTypeKind.Float64)
        
        # Rank from arguments (unless it's a reduction)
        if func_info.get("is_reduction"):
            result_rank = 0  # Reductions always return scalar
            result_jagged = False
        else:
            result_rank = max((arg.rank for arg in args), default=0)
            result_jagged = any(arg.is_jagged for arg in args)
        
        # Parse namespace from cpp_name
        cpp_name = func_info["cpp_name"]
        namespace = None
        if "::" in cpp_name:
            parts = cpp_name.rsplit("::", 1)
            namespace = parts[0]
            cpp_name_short = parts[1]
        else:
            cpp_name_short = cpp_name
        
        return CallNode(
            func=func_name,
            args=args,
            dtype=result_type,
            rank=result_rank,
            is_jagged=result_jagged,
            namespace=namespace,
            cpp_name=func_info["cpp_name"],
            headers=func_info.get("headers", []),
            source_location=self._make_location(node, ctx),
        )
    
    def _get_call_name(self, node: ast.AST) -> Optional[str]:
        """Extract function name from call target."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # TMath.Gaus → "TMath.Gaus"
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None
    
    def _get_function_info(self, name: str, args: List['IRNode'] = None) -> Optional[Dict]:
        """
        Look up function info by name, with optional overload resolution.
        
        Phase 13.5.C v0.5: Selects correct overload based on argument types.
        
        Args:
            name: Function name to look up
            args: Optional argument IRNodes for overload resolution
            
        Returns:
            Dict with function info, or None if not found
            
        Note:
            When args=None, returns latest registered candidate (consistent with
            "latest wins" versioning rule per v0.4 P0-3 fix).
        """
        # Check custom functions first
        if name in self._custom_functions:
            candidates = self._custom_functions[name]
            
            if args is None:
                # v0.5 FIX (P0-3): Return latest for consistency with "latest wins" rule
                return candidates[-1] if candidates else None
            
            # v0.5: Overload resolution
            return self._select_overload(name, candidates, args)
        
        # Check known functions (no overloading for builtins)
        if name in KNOWN_FUNCTIONS:
            return KNOWN_FUNCTIONS[name]
        
        return None
    
    def _select_overload(
        self, 
        name: str, 
        candidates: List[Dict], 
        args: List['IRNode']
    ) -> Dict:
        """
        Select the correct overload based on argument types.
        
        Phase 13.5.D: Ranked selection with numeric widening.
        
        Selection algorithm:
        1. Filter by arity (number of arguments)
        2. Score each candidate by conversion rank sum
        3. Select lowest total rank (exact=0 > promotion=1 > conversion=2)
        4. Error on ambiguity (multiple candidates with same rank)
        5. Error if no viable candidates (all rank -1)
        """
        # Step 1: Filter by arity
        by_arity = [c for c in candidates if len(c['param_types']) == len(args)]
        
        if not by_arity:
            available_arities = sorted(set(len(c['param_types']) for c in candidates))
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"No overload of '{name}' accepts {len(args)} argument(s).\n"
                f"Available arities: {available_arities}",
                suggestions=[
                    f"Check the number of arguments passed to '{name}'",
                ]
            )
        
        # Step 2: Score each candidate by conversion rank
        scored = []  # List of (total_rank, candidate, rank_details)
        not_viable_reasons = []
        
        for candidate in by_arity:
            total_rank, viable, details = self._compute_conversion_rank(
                candidate['param_types'], args
            )
            if viable:
                scored.append((total_rank, candidate, details))
            else:
                sig = [(p['rank'], p['ir_kind'].name) for p in candidate['param_types']]
                not_viable_reasons.append((str(sig), details))
        
        # Step 3: Handle no viable candidates
        if not scored:
            arg_sig = [(arg.rank, arg.dtype.kind.name) for arg in args]
            reason_lines = []
            for sig, reason in not_viable_reasons:
                reason_lines.append(f"    {sig} - not viable: {reason}")
            
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"No overload of '{name}' matches argument types.\n"
                f"  Arguments: {arg_sig}\n"
                f"  Available overloads:\n" + "\n".join(reason_lines),
                suggestions=[
                    "Register an overload with matching parameter types",
                    "Check argument types (scalar vs RVec, int vs double)",
                ]
            )
        
        # Step 4: Sort by total rank (lowest wins)
        scored.sort(key=lambda x: x[0])
        
        # Step 5: Check for ambiguity (Q3 decision: error on same rank)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            min_rank = scored[0][0]
            ambiguous = [s for s in scored if s[0] == min_rank]
            arg_sig = [(arg.rank, arg.dtype.kind.name) for arg in args]
            
            amb_lines = []
            for total, cand, details in ambiguous:
                sig = [(p['rank'], p['ir_kind'].name) for p in cand['param_types']]
                amb_lines.append(f"    - {sig} (rank {total})")
            
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"Ambiguous overload for '{name}' with argument types {arg_sig}.\n"
                f"  Multiple candidates with rank {min_rank}:\n" + "\n".join(amb_lines),
                suggestions=[
                    "Register an overload with exact parameter types",
                    "Remove one of the ambiguous overloads",
                ]
            )
        
        # Return best match
        return scored[0][1]
    
    def _compute_conversion_rank(
        self, 
        param_types: List[Dict], 
        args: List['IRNode']
    ) -> Tuple[int, bool, str]:
        """
        Compute total conversion rank for a candidate overload.
        
        Phase 13.5.D: Numeric widening with ranked matching.
        
        Returns:
            (total_rank, viable, details)
            - total_rank: Sum of per-argument conversion ranks
            - viable: True if all conversions are allowed (no rank -1)
            - details: Human-readable explanation
        """
        total_rank = 0
        details_parts = []
        
        for i, (param, arg) in enumerate(zip(param_types, args)):
            rank = self._conversion_rank(arg, param)
            
            if rank == -1:
                # Not viable - explain why
                reason = self._explain_conversion_failure(arg, param)
                return (-1, False, f"arg[{i}]: {reason}")
            
            total_rank += rank
            if rank > 0:
                details_parts.append(f"arg[{i}]:{arg.dtype.kind.name}→{param['ir_kind'].name}=rank{rank}")
        
        details = ", ".join(details_parts) if details_parts else "exact match"
        return (total_rank, True, details)
    
    def _conversion_rank(self, arg: 'IRNode', param: Dict) -> int:
        """
        Compute conversion rank from argument type to parameter type.
        
        Phase 13.5.D: Scalar-only widening per spec.
        
        Returns:
            0  = Exact match
            1  = Promotion (float→double, int→long)
            2  = Conversion (int→double)
            -1 = Not viable (narrowing, rank mismatch, etc.)
        """
        # P0-2: Check rank first (scalar-only widening)
        if arg.rank != param['rank']:
            return -1  # Rank mismatch is never viable
        
        if arg.rank != 0:
            # Non-scalar: exact kind match only (Phase 13.5.C behavior)
            return 0 if arg.dtype.kind == param['ir_kind'] else -1
        
        # Scalar: apply widening rules from CONVERSION_MATRIX
        from_kind = arg.dtype.kind
        to_kind = param['ir_kind']
        
        if from_kind not in CONVERSION_MATRIX:
            return -1
        if to_kind not in CONVERSION_MATRIX[from_kind]:
            return -1
        
        return CONVERSION_MATRIX[from_kind][to_kind]
    
    def _explain_conversion_failure(self, arg: 'IRNode', param: Dict) -> str:
        """Generate human-readable explanation for conversion failure."""
        if arg.rank != param['rank']:
            return f"rank mismatch ({arg.rank} vs {param['rank']})"
        
        from_kind = arg.dtype.kind
        to_kind = param['ir_kind']
        
        # Check specific forbidden cases
        if from_kind in (IRTypeKind.Float64,) and to_kind in (IRTypeKind.Float32,):
            return f"narrowing {from_kind.name}→{to_kind.name} forbidden"
        
        if from_kind in (IRTypeKind.Int32, IRTypeKind.Int64) and to_kind == IRTypeKind.Float32:
            return f"lossy {from_kind.name}→Float32 forbidden (precision loss > 16.7M)"
        
        if (from_kind.name.startswith('Int') and to_kind.name.startswith('UInt')) or \
           (from_kind.name.startswith('UInt') and to_kind.name.startswith('Int')):
            return f"signed↔unsigned {from_kind.name}→{to_kind.name} forbidden"
        
        return f"no conversion {from_kind.name}→{to_kind.name}"
    
    # =========================================================================
    # Phase 12.2: Selection Function Type Inference
    # =========================================================================
    
    def _infer_selection_function_type(self, func_name: str, args: List[IRNode],
                                        ctx: BuildContext, node: ast.AST) -> Tuple[IRType, int]:
        """
        Infer return type for RVec selection functions (Take, Range, Where, IndicesFromOffsets).
        
        Phase 12.2: These functions have special type inference rules:
        - Take(vec, n/indices) → same element type as vec, rank=1
        - Range(start, end, step) → RVec<int>, args must be scalar
        - Where(cond, if_true, if_false) → promoted type of if_true/if_false
        - IndicesFromOffsets(first, count) → RVec<int>, args must be RVec
        
        Args:
            func_name: Function name
            args: List of IR nodes for arguments
            ctx: Build context
            node: AST node for error location
            
        Returns:
            Tuple of (dtype, rank)
            
        Raises:
            IRError: If type constraints are violated
        """
        if func_name == "Take":
            # Take(vec, n) or Take(vec, indices) → same element type as vec, rank=1
            if len(args) >= 1:
                vec_type = args[0].dtype
                # Return same type, always rank=1 (RVec)
                return vec_type, 1
            return IRType(IRTypeKind.Unknown), 1
        
        elif func_name == "Range":
            # Range MUST have scalar arguments (rank=0)
            for i, arg in enumerate(args):
                if arg.rank != 0:
                    raise IRError(
                        IRErrorKind.RANK_ERROR,
                        f"Range() argument {i+1} must be scalar (got rank={arg.rank}). "
                        f"Range does not support RVec arguments.",
                        suggestions=[
                            "Use Range with scalar bounds: Range(start, end)",
                            "For vector-of-ranges, use: IndicesFromOffsets(first_vec, count_vec)",
                            "Or index into RVec first: Range(arr[0], arr[0] + n[0])"
                        ],
                        source_location=self._make_location(node, ctx),
                    )
            # Range always returns RVec<int>
            return IRType(IRTypeKind.Object, cpp_type="RVec<int>"), 1
        
        elif func_name == "Where":
            # Where(cond, if_true, if_false) → promoted type of if_true and if_false
            if len(args) >= 3:
                cond, if_true, if_false = args[0], args[1], args[2]
                
                # Result type is promoted common type of if_true and if_false
                result_type = promote_types(if_true.dtype, if_false.dtype)
                if result_type is None:
                    result_type = if_true.dtype  # Fallback to first arg
                
                # Result rank: max of all argument ranks (broadcast semantics)
                result_rank = max(cond.rank, if_true.rank, if_false.rank)
                
                return result_type, result_rank
            return IRType(IRTypeKind.Unknown), 0
        
        elif func_name == "IndicesFromOffsets":
            # IndicesFromOffsets(first_vec, count_vec) → RVec<int>
            # Both args must be RVec<int> (rank=1)
            if len(args) >= 2:
                for i, arg in enumerate(args):
                    if arg.rank != 1:
                        raise IRError(
                            IRErrorKind.RANK_ERROR,
                            f"IndicesFromOffsets() argument {i+1} must be RVec (got rank={arg.rank}). "
                            f"Both first and count must be RVec<int> arrays.",
                            suggestions=[
                                "Pass RVec<int> arrays for first and count",
                                "Example: IndicesFromOffsets(trackClusterFirst, trackClusterN)"
                            ],
                            source_location=self._make_location(node, ctx),
                        )
            # Always returns RVec<int>
            return IRType(IRTypeKind.Object, cpp_type="RVec<int>"), 1
        
        # Fallback
        return IRType(IRTypeKind.Unknown), 0
    
    # =========================================================================
    # Phase 11.1: Namespace Call Support
    # =========================================================================
    
    def _extract_namespace_chain(self, node: ast.Attribute) -> Optional[Tuple[str, str]]:
        """
        Extract namespace and function from chained attribute access.
        
        Phase 11.1c: Uses ROOT reflection to find longest valid namespace prefix.
        Schema variables take priority over namespaces.
        
        Handles:
        - TMath.Pi -> ("TMath", "Pi")
        - ROOT.Math.VectorUtil.DeltaPhi -> ("ROOT.Math.VectorUtil", "DeltaPhi")
        - o2.tpc.TrackTPC.GetParam -> ("o2.tpc.TrackTPC", "GetParam") if loaded
        
        Returns None if this is not a namespace call (e.g., track.Pt()).
        Raises IRError if symbol cannot be resolved in schema or ROOT.
        
        Args:
            node: AST Attribute node representing the call target
            
        Returns:
            Tuple of (namespace, function_name) or None
            
        Raises:
            IRError: If base symbol not found in schema or ROOT
        """
        # Collect the chain: [ROOT, Math, VectorUtil, DeltaPhi]
        chain = []
        current = node
        
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            chain.append(current.id)
        else:
            return None  # Complex expression, not a simple namespace
        
        chain.reverse()  # Now: [ROOT, Math, VectorUtil, DeltaPhi]
        
        if len(chain) < 2:
            return None
        
        # The last element is the function, rest is namespace
        func_name = chain[-1]
        
        # Check if base name is in schema (then it's an object method, not namespace)
        base_name = chain[0]
        if self.inferrer.has_variable(base_name):
            return None  # This is track.Pt(), not a namespace
        
        # Try progressively longer namespace prefixes (longest match first)
        # For [ROOT, Math, VectorUtil, DeltaPhi]:
        # Try: ROOT.Math.VectorUtil, ROOT.Math, ROOT
        for i in range(len(chain) - 1, 0, -1):
            namespace = ".".join(chain[:i])
            
            # Check if this namespace is known (builtin, ROOT reflection, or registered)
            if self._is_known_namespace(namespace):
                # The function name is everything after the namespace
                func_name = chain[i]
                # Cache positive result
                self._namespace_cache[namespace] = True
                return (namespace, func_name)
        
        # Phase 11.1c: Unknown symbol - raise helpful error
        # (Only if we got here via a call, not attribute access)
        # We return None here and let the caller decide whether to error
        # This preserves backward compatibility for non-call attribute access
        return None
    
    def _collect_attribute_chain(self, node: ast.AST) -> Optional[List[str]]:
        """
        Collect the chain of names from an attribute access.
        
        Phase 11.1c: Helper for error messages.
        
        Args:
            node: AST node (Name or Attribute)
            
        Returns:
            List of names like ["TMath", "Sin"] or None if not a simple chain
        """
        chain = []
        current = node
        
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            chain.append(current.id)
            chain.reverse()
            return chain
        
        return None
    
    def _is_known_namespace(self, namespace: str) -> bool:
        """
        Check if namespace is known (via ROOT reflection or fallback list).
        
        Phase 11.1c: Dynamic namespace detection using ROOT reflection.
        
        Resolution order:
        1. Check positive cache (fast path)
        2. Try ROOT reflection via hasattr
        3. Fallback to KNOWN_NAMESPACES (for non-ROOT environments)
        
        Only positive results are cached to allow library loading after
        a failed lookup.
        
        Args:
            namespace: Namespace string (e.g., "TMath", "ROOT.Math.VectorUtil", "o2.tpc")
            
        Returns:
            True if namespace is recognized
        """
        # Check cache first (positive results only)
        if namespace in self._namespace_cache:
            return True
        
        # Try ROOT reflection
        try:
            import ROOT
            parts = namespace.replace("::", ".").split(".")
            obj = ROOT
            
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    # Chain broken - not fully resolvable via ROOT
                    break
            else:
                # Successfully resolved entire chain - cache positive result
                self._namespace_cache[namespace] = True
                return True
        except ImportError:
            pass  # ROOT not available, fall through to KNOWN_NAMESPACES
        
        # Fallback to hardcoded list (for non-ROOT environments / mock tests)
        from .constants import KNOWN_NAMESPACES
        if namespace in KNOWN_NAMESPACES:
            return True
        
        # Phase 11.2: Check user-registered namespaces (guard for now)
        if hasattr(self, '_registry') and self._registry:
            if self._registry.is_registered_namespace(namespace):
                return True
        
        return False
    
    def _build_namespace_call(self, namespace: str, func_name: str,
                              args: List[ast.expr], ctx: BuildContext,
                              node: ast.AST) -> CallNode:
        """
        Build a CallNode for a namespace function call.
        
        Phase 11.1: Creates CallNode with namespace information.
        
        Args:
            namespace: Namespace string (e.g., "TMath")
            func_name: Function name (e.g., "Pi", "Sin")
            args: AST argument nodes
            ctx: Build context
            node: Original AST node for location info
            
        Returns:
            CallNode with namespace and type information
        """
        # Visit arguments
        visited_args = [self._visit(arg, ctx) for arg in args]
        
        # Get return type from registry
        return_type = self._get_namespace_function_type(namespace, func_name)
        
        # Phase 11.1 fix: Propagate rank from arguments (enables RVec broadcasting)
        # TMath.Sqrt(pt) where pt is RVec<double> → returns RVec<double>
        arg_rank = max((arg.rank for arg in visited_args), default=0)
        arg_jagged = any(arg.is_jagged for arg in visited_args)
        
        # Convert namespace to C++ format for cpp_name
        cpp_namespace = namespace.replace(".", "::")
        cpp_name = f"{cpp_namespace}::{func_name}"
        
        # Get headers for this namespace
        from .constants import NAMESPACE_HEADERS
        headers = []
        if namespace in NAMESPACE_HEADERS:
            headers.append(NAMESPACE_HEADERS[namespace])
        # Check parent namespaces too
        parts = namespace.split(".")
        for i in range(len(parts), 0, -1):
            parent = ".".join(parts[:i])
            if parent in NAMESPACE_HEADERS:
                headers.append(NAMESPACE_HEADERS[parent])
                break
        
        return CallNode(
            func=func_name,
            args=visited_args,
            namespace=cpp_namespace,  # Store as C++ style (TMath, ROOT::Math)
            cpp_name=cpp_name,
            dtype=return_type,
            rank=arg_rank,  # Propagate rank for broadcasting
            is_jagged=arg_jagged,
            headers=headers,
            source_location=self._make_location(node, ctx),
        )
    
    def _get_namespace_function_type(self, namespace: str, func_name: str) -> IRType:
        """
        Get return type for namespace function.
        
        Phase 11.1: Uses NAMESPACE_FUNCTION_TYPES constant.
        
        Args:
            namespace: Namespace string (e.g., "TMath")
            func_name: Function name (e.g., "Sin", "Pi")
            
        Returns:
            IRType for the function return value
        """
        from .constants import NAMESPACE_FUNCTION_TYPES
        
        # Check builtin types
        if namespace in NAMESPACE_FUNCTION_TYPES:
            if func_name in NAMESPACE_FUNCTION_TYPES[namespace]:
                type_str = NAMESPACE_FUNCTION_TYPES[namespace][func_name]
                if type_str == "double":
                    return IRType(IRTypeKind.Float64)
                elif type_str == "int":
                    return IRType(IRTypeKind.Int32)
                elif type_str == "bool":
                    return IRType(IRTypeKind.Bool)
                # Add more mappings as needed
                return IRType(IRTypeKind.Float64)
        
        # Phase 11.2: Check user-registered functions
        if hasattr(self, '_registry') and self._registry:
            func_info = self._registry.lookup(f"{namespace}.{func_name}")
            if func_info:
                # Parse return type from func_info
                return IRType(IRTypeKind.Float64)  # Placeholder
        
        # Default to double for unknown functions
        return IRType(IRTypeKind.Float64)
    
    def _visit_method_call(self, node: ast.Call, ctx: BuildContext) -> IRNode:
        """Handle method calls: track.getX(), obj.Method(args), tracks.Pt()."""
        if not isinstance(node.func, ast.Attribute):
            raise unsupported_operation_error(
                "Invalid method call syntax",
                location=self._make_location(node, ctx),
            )
        
        obj = self._visit(node.func.value, ctx)
        method_name = node.func.attr
        args = [self._visit(arg, ctx) for arg in node.args]
        
        # Phase 8: Check for method broadcasting on RVec<Object>
        if self._is_rvec_of_objects(obj):
            # Broadcasting: tracks.Pt() → RVec<double>
            element_type = self._get_rvec_element_type(obj)
            
            # Resolve method return type
            return_type_str, ir_type = self._resolve_broadcast_method(
                element_type, method_name, ctx, node
            )
            
            # Determine result dtype - it's RVec<return_type>
            if ir_type.kind == IRTypeKind.Object:
                # Method returns object (e.g., Vect() → TVector3)
                result_dtype = IRType(IRTypeKind.Object, f"RVec<{return_type_str}>")
            else:
                # Method returns scalar (e.g., Pt() → double)
                result_dtype = IRType(IRTypeKind.Object, f"RVec<{return_type_str}>")
            
            return MethodBroadcastNode(
                target=obj,
                method_name=method_name,
                element_type=element_type,
                result_element_type=return_type_str,
                dtype=result_dtype,
                rank=1,  # Result is always RVec
                is_jagged=obj.is_jagged,
                source_location=self._make_location(node, ctx),
            )
        
        # For scalar object types, use direct method call (Phase 6a)
        if obj.dtype.kind == IRTypeKind.Object and obj.rank == 0:
            return MethodCallNode(
                object=obj,
                method_name=method_name,
                args=args,
                dtype=IRType(IRTypeKind.Unknown),  # Will be resolved in Phase 4
                rank=obj.rank,
                is_jagged=obj.is_jagged,
                class_name=obj.dtype.cpp_type,
                source_location=self._make_location(node, ctx),
            )
        
        # For non-object types (e.g., RVec<double>), handle RVec methods
        rvec_methods = {"size", "at", "front", "back", "empty"}
        if method_name in rvec_methods:
            if method_name == "size":
                return MethodCallNode(
                    object=obj,
                    method_name=method_name,
                    args=args,
                    dtype=IRType(IRTypeKind.UInt64),
                    rank=0 if obj.rank > 0 else obj.rank,
                    is_jagged=False,
                    source_location=self._make_location(node, ctx),
                )
            elif method_name == "empty":
                return MethodCallNode(
                    object=obj,
                    method_name=method_name,
                    args=args,
                    dtype=IRType(IRTypeKind.Bool),
                    rank=0 if obj.rank > 0 else obj.rank,
                    is_jagged=False,
                    source_location=self._make_location(node, ctx),
                )
            else:  # at, front, back
                return MethodCallNode(
                    object=obj,
                    method_name=method_name,
                    args=args,
                    dtype=obj.dtype,
                    rank=max(0, obj.rank - 1),
                    is_jagged=False,
                    source_location=self._make_location(node, ctx),
                )
        
        # Phase 10.5: RVec aggregation methods (sum, mean, max, min, etc.)
        # Maps method-style to function-style: pt.sum() → Sum(pt)
        rvec_aggregation_methods = {
            'sum': 'Sum',
            'mean': 'Mean',
            'max': 'Max',
            'min': 'Min',
            'any': 'Any',
            'all': 'All',
            'std': 'StdDev',
            'var': 'Var',
        }
        
        if obj.rank > 0 and method_name in rvec_aggregation_methods:
            func_name = rvec_aggregation_methods[method_name]
            func_info = KNOWN_FUNCTIONS.get(func_name)
            
            if func_info:
                # Determine return type
                return_kind = func_info.get("return_type")
                if return_kind is None:
                    # Infer element type from RVec<T>
                    cpp_type = obj.dtype.cpp_type or ""
                    if cpp_type.startswith("ROOT::RVec<"):
                        inner = cpp_type[11:-1]
                    elif cpp_type.startswith("RVec<"):
                        inner = cpp_type[5:-1]
                    else:
                        inner = obj.dtype.to_cpp()
                    from .ir_types import cpp_type_to_ir
                    result_type = cpp_type_to_ir(inner)
                else:
                    result_type = IRType(return_kind)
                
                # All aggregations return scalar (rank=0)
                return CallNode(
                    func=func_name,
                    args=[obj],
                    dtype=result_type,
                    rank=0,
                    is_jagged=False,
                    namespace="ROOT::VecOps",
                    cpp_name=func_info["cpp_name"],
                    headers=["<ROOT/RVec.hxx>"],
                    source_location=self._make_location(node, ctx),
                )
        
        raise unsupported_operation_error(
            f"Unknown method: {method_name}",
            location=self._make_location(node, ctx),
        )
    
    def _visit_Attribute(self, node: ast.Attribute, ctx: BuildContext) -> IRNode:
        """Handle attribute access: track.mPx, obj.field."""
        # Check if this is a namespace reference (TMath.Pi)
        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{node.attr}"
            
            # Check if it's a known constant
            known_constants = {
                "TMath.Pi": (3.14159265358979323846, IRTypeKind.Float64),
                "TMath.E": (2.71828182845904523536, IRTypeKind.Float64),
                "TMath.TwoPi": (6.28318530717958647692, IRTypeKind.Float64),
            }
            
            if full_name in known_constants:
                value, kind = known_constants[full_name]
                return ConstantNode(
                    value=value,
                    dtype=IRType(kind),
                    source_location=self._make_location(node, ctx),
                )
            
            # Could be a qualified variable name
            if self.inferrer.has_variable(full_name):
                info = self.inferrer.get_variable_info(full_name)
                return VariableNode(
                    name=full_name,
                    dtype=info.dtype,
                    rank=info.rank,
                    is_jagged=info.is_jagged,
                    source_location=self._make_location(node, ctx),
                )
        
        # Property access on object
        obj = self._visit(node.value, ctx)
        attr_name = node.attr
        
        # Phase 8: Check for property broadcasting on RVec<Object>
        if self._is_rvec_of_objects(obj):
            # Broadcasting: particles.fPx → RVec<double>
            element_type = self._get_rvec_element_type(obj)
            
            # Resolve property type
            property_type_str, ir_type, access_mode = self._resolve_broadcast_property(
                element_type, attr_name, ctx, node
            )
            
            # Determine result dtype - it's RVec<property_type>
            result_dtype = IRType(IRTypeKind.Object, f"RVec<{property_type_str}>")
            
            return PropertyBroadcastNode(
                target=obj,
                property_name=attr_name,
                element_type=element_type,
                result_element_type=property_type_str,
                access_mode=access_mode,
                dtype=result_dtype,
                rank=1,  # Result is always RVec
                is_jagged=obj.is_jagged,
                source_location=self._make_location(node, ctx),
            )
        
        # For scalar object types, use direct property access (existing)
        if obj.dtype.kind == IRTypeKind.Object:
            return PropertyAccessNode(
                object=obj,
                property_name=attr_name,
                dtype=IRType(IRTypeKind.Unknown),  # Resolve in Phase 4
                rank=obj.rank,
                is_jagged=obj.is_jagged,
                class_name=obj.dtype.cpp_type,
                source_location=self._make_location(node, ctx),
            )
        
        raise unsupported_operation_error(
            f"Cannot access attribute '{attr_name}' on non-object type",
            location=self._make_location(node, ctx),
        )
    
    def _visit_Subscript(self, node: ast.Subscript, ctx: BuildContext) -> IRNode:
        """Handle subscript operations: arr[0], arr[1:3], arr[:]."""
        value = self._visit(node.value, ctx)
        sub_ctx = ctx.with_subscript()
        
        # Handle slice vs index
        slice_node = node.slice
        
        # Python 3.9+ uses slice directly, older uses Index wrapper
        if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index):
            slice_node = slice_node.value
        
        if isinstance(slice_node, ast.Slice):
            return self._build_slice_subscript(value, slice_node, sub_ctx)
        elif isinstance(slice_node, ast.Tuple):
            # Multi-dimensional: arr[i, j] or arr[:, 0]
            return self._build_multi_subscript(value, slice_node, sub_ctx)
        else:
            # Scalar index: arr[i]
            index = self._visit(slice_node, sub_ctx)
            return self._build_scalar_subscript(value, index, ctx)
    
    def _build_slice_subscript(self, value: IRNode, 
                                slice_node: ast.Slice,
                                ctx: BuildContext) -> IRNode:
        """Build subscript with slice: arr[1:3], arr[:], etc.
        
        Phase 7: Creates RVecSliceNode for RVec slicing with proper classification.
        Phase 8: Detects and errors on broadcast-then-slice pattern.
        """
        # Phase 8: Check for forbidden broadcast-then-slice pattern
        if isinstance(value, (MethodBroadcastNode, PropertyBroadcastNode)):
            if isinstance(value, MethodBroadcastNode):
                original = f"tracks.{value.method_name}()"
                suggestion = f"tracks[:n].{value.method_name}()"
            else:
                original = f"particles.{value.property_name}"
                suggestion = f"particles[:n].{value.property_name}"
            
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                f"Slicing after broadcasting is not supported: '{original}[...]'",
                source_location=self._make_location(slice_node, ctx),
                suggestions=[
                    f"Use '{suggestion}' instead of '{original}[:n]'",
                    "Slice before broadcasting for better performance"
                ]
            )
        
        start = self._visit(slice_node.lower, ctx) if slice_node.lower else None
        stop = self._visit(slice_node.upper, ctx) if slice_node.upper else None
        step = self._visit(slice_node.step, ctx) if slice_node.step else None
        
        # Validate step != 0
        if step and isinstance(step, ConstantNode) and step.value == 0:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                "Slice step cannot be zero",
                source_location=self._make_location(slice_node, ctx),
                suggestions=["Use a non-zero step value, e.g., [::1] or [::2]"]
            )
        
        # If this is an RVec (rank=1), use RVecSliceNode
        if value.rank == 1:
            slice_kind = self._classify_slice(start, stop, step, ctx, slice_node)
            
            return RVecSliceNode(
                target=value,
                start=start,
                stop=stop,
                step=step,
                slice_kind=slice_kind,
                dtype=value.dtype,
                rank=1,  # Slicing RVec returns RVec
                source_location=self._make_location(slice_node, ctx),
            )
        
        # Fallback: For non-RVec (rank != 1), use existing SubscriptNode
        slice_ir = SliceNode(
            start=start,
            stop=stop,
            step=step,
            source_location=self._make_location(slice_node, ctx),
        )
        
        return SubscriptNode(
            value=value,
            indices=[slice_ir],
            dtype=value.dtype,
            rank=value.rank,
            is_jagged=value.is_jagged,
            source_location=self._make_location(slice_node, ctx),
        )
    
    def _classify_slice(self, start: Optional[IRNode], 
                        stop: Optional[IRNode],
                        step: Optional[IRNode],
                        ctx: BuildContext,
                        ast_node: ast.Slice) -> SliceKind:
        """Classify slice pattern for code generation.
        
        Returns appropriate SliceKind based on start/stop/step values.
        Raises IRError for unsupported patterns.
        """
        # Helper to check if a node is a constant with specific value
        def is_const(node: Optional[IRNode], check_fn) -> bool:
            return isinstance(node, ConstantNode) and check_fn(node.value)
        
        def is_positive_const(node: Optional[IRNode]) -> bool:
            return is_const(node, lambda v: isinstance(v, int) and v > 0)
        
        def is_negative_const(node: Optional[IRNode]) -> bool:
            return is_const(node, lambda v: isinstance(v, int) and v < 0)
        
        def is_nonneg_const(node: Optional[IRNode]) -> bool:
            return is_const(node, lambda v: isinstance(v, int) and v >= 0)
        
        # Check for reverse: [::-1]
        if step and is_const(step, lambda v: v == -1):
            if start is None and stop is None:
                return SliceKind.REVERSE
            # Negative step with start/stop not supported
            raise IRError(
                IRErrorKind.UNSUPPORTED_OP,
                "Negative step with start/stop not supported",
                source_location=self._make_location(ast_node, ctx),
                suggestions=["Use [::-1] for simple reverse, or implement in Python"]
            )
        
        # Check for step slicing: [::n] or [start::n] or [start:stop:n]
        if step and is_positive_const(step):
            return SliceKind.STEP
        
        # No step cases
        if step is None:
            # [:] - full slice (copy all)
            if start is None and stop is None:
                # Full slice is essentially FROM_INDEX with start=0
                return SliceKind.FROM_INDEX
            
            # [:n] - first n (stop must be positive)
            if start is None and stop is not None:
                if is_positive_const(stop):
                    return SliceKind.FIRST_N
                # [:n] with negative n - treat as range with clamping
                if is_negative_const(stop):
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Slice '[:negative]' not yet supported",
                        source_location=self._make_location(ast_node, ctx),
                        suggestions=["Use [:-1] equivalent with explicit indices"]
                    )
            
            # [-n:] - last n (start must be negative, stop must be None)
            if stop is None and start is not None:
                if is_negative_const(start):
                    return SliceKind.LAST_N
                # [n:] - from index (start must be non-negative)
                if is_nonneg_const(start):
                    return SliceKind.FROM_INDEX
            
            # [a:b] - range (both start and stop present)
            if start is not None and stop is not None:
                # Check for mixed negative indices
                start_neg = is_negative_const(start)
                stop_neg = is_negative_const(stop)
                
                if start_neg or stop_neg:
                    raise IRError(
                        IRErrorKind.UNSUPPORTED_OP,
                        "Mixed negative indices in slice not yet supported",
                        source_location=self._make_location(ast_node, ctx),
                        suggestions=["Supported: [:n], [-n:], [n:], [a:b] with positive indices, [::step], [::-1]"]
                    )
                
                return SliceKind.RANGE
        
        # Fallback - unsupported complex pattern
        raise IRError(
            IRErrorKind.UNSUPPORTED_OP,
            "Complex slice pattern not yet supported",
            source_location=self._make_location(ast_node, ctx),
            suggestions=["Supported: [:n], [-n:], [n:], [a:b], [::step], [::-1], [mask]"]
        )
    
    def _build_scalar_subscript(self, value: IRNode, 
                                 index: IRNode,
                                 ctx: BuildContext) -> IRNode:
        """Build subscript with scalar index: arr[i] or boolean mask: arr[mask]."""
        
        # Check for boolean masking: arr[arr > 1.0] or arr[mask]
        # Boolean mask must be rank=1 and dtype=bool
        if self._is_boolean_mask(index, value):
            return RVecSliceNode(
                target=value,
                start=index,  # Store mask in start field
                stop=None,
                step=None,
                slice_kind=SliceKind.BOOLEAN,
                dtype=value.dtype,
                rank=1,  # Boolean masking returns RVec
                source_location=index.source_location,
            )
        
        # Regular scalar indexing reduces rank by 1
        new_rank = max(0, value.rank - 1)
        
        return SubscriptNode(
            value=value,
            indices=[index],
            dtype=value.dtype,
            rank=new_rank,
            is_jagged=value.is_jagged if new_rank > 0 else False,
            source_location=index.source_location,
        )
    
    def _is_boolean_mask(self, index: IRNode, target: IRNode) -> bool:
        """Check if index is a boolean mask for the target.
        
        Boolean masking requires:
        - Target is rank=1 (RVec)
        - Index is rank=1 (RVec)
        - Index has element type=bool (RVec<bool>)
        """
        if target.rank != 1:
            return False
        if index.rank != 1:
            return False
        
        # Check for boolean type - could be direct Bool or RVec<bool>
        if isinstance(index.dtype, IRType):
            # Direct bool dtype
            if index.dtype.kind == IRTypeKind.Bool:
                return True
            # RVec<bool> has kind=Object with cpp_type containing "bool"
            if index.dtype.kind == IRTypeKind.Object:
                cpp_type = index.dtype.cpp_type or ""
                # Extract inner type and check if it's bool
                inner_type, _ = extract_inner_type(cpp_type)
                return inner_type.lower() in ('bool', 'bool_t')
        
        return str(index.dtype).lower() == 'bool'
    
    def _build_multi_subscript(self, value: IRNode,
                                tuple_node: ast.Tuple,
                                ctx: BuildContext) -> IRNode:
        """Build multi-dimensional subscript: arr[i, j] or arr[:, 0]."""
        indices = []
        rank_reduction = 0
        
        for elt in tuple_node.elts:
            if isinstance(elt, ast.Slice):
                start = self._visit(elt.lower, ctx) if elt.lower else None
                stop = self._visit(elt.upper, ctx) if elt.upper else None
                step = self._visit(elt.step, ctx) if elt.step else None
                indices.append(SliceNode(start=start, stop=stop, step=step))
            else:
                indices.append(self._visit(elt, ctx))
                rank_reduction += 1
        
        new_rank = max(0, value.rank - rank_reduction)
        
        return SubscriptNode(
            value=value,
            indices=indices,
            dtype=value.dtype,
            rank=new_rank,
            is_jagged=value.is_jagged if new_rank > 0 else False,
            source_location=self._make_location(tuple_node, ctx),
        )
    
    def _visit_IfExp(self, node: ast.IfExp, ctx: BuildContext) -> IRNode:
        """Handle conditional expressions: x if cond else y."""
        condition = self._visit(node.test, ctx)
        if_true = self._visit(node.body, ctx)
        if_false = self._visit(node.orelse, ctx)
        
        # Result type is promoted type of branches
        result_type = promote_types(if_true.dtype, if_false.dtype)
        if result_type is None:
            raise type_mismatch_error(
                str(if_true.dtype), str(if_false.dtype),
                "Conditional branches must have compatible types",
                location=self._make_location(node, ctx),
            )
        
        # Rank is max of all components
        result_rank = max(condition.rank, if_true.rank, if_false.rank)
        result_jagged = condition.is_jagged or if_true.is_jagged or if_false.is_jagged
        
        return TernaryOpNode(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
            dtype=result_type,
            rank=result_rank,
            is_jagged=result_jagged,
            source_location=self._make_location(node, ctx),
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _make_location(self, node: ast.AST, ctx: BuildContext) -> SourceLocation:
        """Create SourceLocation from AST node."""
        return SourceLocation(
            expr_name=ctx.alias_name or "<expression>",
            text_span=ctx.expression_text,
            line=getattr(node, 'lineno', 1),
            column=getattr(node, 'col_offset', 0),
        )
