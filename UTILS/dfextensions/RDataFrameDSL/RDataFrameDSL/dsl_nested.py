"""
Phase 13.3.DSL D7: DSL Compiler Extension for Nested RVec 2D Slicing

Extends DSL compiler to support RVec<RVec<T>> operations:
- nested[:, j]   → Column extract (fail-closed)
- nested[:, a:b] → Column slice (clamp per-row)
- nested[i, j]   → Element access
- nested[i]      → Row access

Detects nested RVec types from schema and routes to appropriate handlers.
"""

import ast
import re
from typing import Optional, Dict, Tuple, Union, List, Any
from dataclasses import dataclass

from .type_inferrer import (
    TypeInferrer,
    is_rvec_type,
    is_vector_type,
    is_collection_type,
    extract_inner_type,
)
from .ir_nodes_nested import (
    NestedAccessNode, NestedSliceKind, SliceParams,
    make_nested_column_extract,
    make_nested_column_slice,
    make_nested_element_access,
    make_nested_row_access,
)
from .backend_nested import NestedCodeGenerator, GeneratedCode


__all__ = [
    'NestedExpressionAnalyzer',
    'NestedDSLCompiler',
    'is_nested_rvec_type',
    'get_nested_element_type',
]


def is_nested_rvec_type(cpp_type: str) -> bool:
    """
    Check if C++ type is a nested RVec (RVec<RVec<T>>).
    
    Args:
        cpp_type: C++ type string
        
    Returns:
        True if type is RVec<RVec<T>>
        
    Examples:
        >>> is_nested_rvec_type("RVec<RVec<double>>")
        True
        >>> is_nested_rvec_type("RVec<double>")
        False
    """
    if not is_collection_type(cpp_type):
        return False
    
    inner, depth = extract_inner_type(cpp_type)
    return depth >= 2


def get_nested_element_type(cpp_type: str) -> str:
    """
    Get the innermost element type from a nested RVec.
    
    Args:
        cpp_type: C++ type string like "RVec<RVec<double>>"
        
    Returns:
        Innermost element type (e.g., "double")
    """
    inner, _ = extract_inner_type(cpp_type)
    return inner


@dataclass
class AnalysisResult:
    """Result of analyzing an expression for nested operations."""
    is_nested: bool
    node: Optional[NestedAccessNode] = None
    error: Optional[str] = None


class NestedExpressionAnalyzer:
    """
    Analyzes Python expressions to detect and parse nested RVec operations.
    
    Handles:
    - nested[:, j]   → Column extract (fail-closed)
    - nested[:, a:b] → Column slice (clamp per-row)
    - nested[i, j]   → Element access
    - nested[i]      → Row access
    
    Phase 13.3.DSL D7.
    """
    
    def __init__(self, type_inferrer: TypeInferrer):
        """
        Initialize analyzer.
        
        Args:
            type_inferrer: TypeInferrer with schema information
        """
        self.type_inferrer = type_inferrer
    
    def analyze(self, expr: str) -> AnalysisResult:
        """
        Analyze an expression for nested RVec operations.
        
        Args:
            expr: Python expression string
            
        Returns:
            AnalysisResult with is_nested flag and optional IR node
        """
        try:
            tree = ast.parse(expr, mode='eval')
            return self._analyze_node(tree.body, expr)
        except SyntaxError as e:
            return AnalysisResult(is_nested=False, error=f"Syntax error: {e}")
    
    def _analyze_node(self, node: ast.AST, source: str) -> AnalysisResult:
        """Analyze an AST node."""
        # Check for subscript syntax: nested[...]
        if isinstance(node, ast.Subscript):
            return self._analyze_subscript(node, source)
        
        return AnalysisResult(is_nested=False)
    
    def _analyze_subscript(self, node: ast.Subscript, source: str) -> AnalysisResult:
        """
        Analyze subscript expression: nested[...].
        
        Handles:
        - nested[i, j] → element access
        - nested[:, j] → column extract (fail-closed)
        - nested[:, a:b] → column slice (clamp)
        - nested[i] → row access
        """
        # Get variable name
        if not isinstance(node.value, ast.Name):
            return AnalysisResult(is_nested=False)
        
        name = node.value.id
        
        # Check if it's a nested RVec variable
        if not self._is_nested_var(name):
            return AnalysisResult(is_nested=False)
        
        elem_type = self._get_element_type(name)
        
        # Check for tuple (2D indexing): nested[i, j] or nested[:, j]
        if isinstance(node.slice, ast.Tuple):
            if len(node.slice.elts) != 2:
                return AnalysisResult(
                    is_nested=True,
                    error=f"Nested RVec {name}[...] requires exactly 2 indices"
                )
            
            row_node = node.slice.elts[0]
            col_node = node.slice.elts[1]
            
            row_is_slice = isinstance(row_node, ast.Slice)
            col_is_slice = isinstance(col_node, ast.Slice)
            
            # nested[:, j] - column extract (integer) → FAIL-CLOSED
            if row_is_slice and self._is_full_slice(row_node) and not col_is_slice:
                col_index = self._ast_to_index(col_node)
                ir_node = make_nested_column_extract(
                    target=name,
                    col_index=col_index,
                    element_type=elem_type,
                    source=source
                )
                return AnalysisResult(is_nested=True, node=ir_node)
            
            # nested[:, a:b] - column slice → CLAMP PER-ROW
            if row_is_slice and self._is_full_slice(row_node) and col_is_slice:
                col_slice = self._ast_to_slice_params(col_node)
                ir_node = make_nested_column_slice(
                    target=name,
                    col_slice=col_slice,
                    element_type=elem_type,
                    source=source
                )
                return AnalysisResult(is_nested=True, node=ir_node)
            
            # nested[i, j] - element access
            if not row_is_slice and not col_is_slice:
                row = self._ast_to_index(row_node)
                col = self._ast_to_index(col_node)
                ir_node = make_nested_element_access(
                    target=name,
                    row=row,
                    col=col,
                    element_type=elem_type,
                    source=source
                )
                return AnalysisResult(is_nested=True, node=ir_node)
            
            # nested[i, :] - same as nested[i]
            if not row_is_slice and col_is_slice and self._is_full_slice(col_node):
                row = self._ast_to_index(row_node)
                ir_node = make_nested_row_access(
                    target=name,
                    row=row,
                    element_type=elem_type,
                    source=source
                )
                return AnalysisResult(is_nested=True, node=ir_node)
            
            # Other combinations not supported yet
            return AnalysisResult(
                is_nested=True,
                error=f"Unsupported nested indexing pattern: {source}"
            )
        
        # Single index: nested[i] → row access
        if isinstance(node.slice, ast.Slice):
            # nested[:n] - row slice (not column operation)
            # This returns RVec<RVec<T>> (subset of rows)
            row_slice = self._ast_to_slice_params(node.slice)
            # For now, treat as row slice (not a D7 column operation)
            return AnalysisResult(is_nested=False)  # Let existing RVec handler deal with it
        
        row = self._ast_to_index(node.slice)
        ir_node = make_nested_row_access(
            target=name,
            row=row,
            element_type=elem_type,
            source=source
        )
        
        return AnalysisResult(is_nested=True, node=ir_node)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _is_nested_var(self, name: str) -> bool:
        """Check if variable is a nested RVec type."""
        try:
            cpp_type = self.type_inferrer.get_cpp_type(name)
            if cpp_type is None:
                return False
            return is_nested_rvec_type(cpp_type)
        except:
            return False
    
    def _get_element_type(self, name: str) -> str:
        """Get inner element type for a nested RVec variable."""
        try:
            cpp_type = self.type_inferrer.get_cpp_type(name)
            return get_nested_element_type(cpp_type)
        except:
            return "double"
    
    def _is_full_slice(self, node: ast.Slice) -> bool:
        """Check if slice is [:] (full slice)."""
        return node.lower is None and node.upper is None and node.step is None
    
    def _ast_to_index(self, node: ast.AST) -> Union[int, str]:
        """Convert AST node to index value (int or expression string)."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._ast_to_index(node.operand)
            if isinstance(inner, int):
                return -inner
            return f"-{inner}"
        # For complex expressions, convert to string
        return ast.unparse(node) if hasattr(ast, 'unparse') else self._node_to_str(node)
    
    def _ast_to_slice_params(self, node: ast.Slice) -> SliceParams:
        """Convert AST Slice to SliceParams."""
        start = self._ast_to_index(node.lower) if node.lower else None
        stop = self._ast_to_index(node.upper) if node.upper else None
        step = self._ast_to_index(node.step) if node.step else None
        return SliceParams(start=start, stop=stop, step=step)
    
    def _node_to_str(self, node: ast.AST) -> str:
        """Convert AST node to string (fallback for Python < 3.9)."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Num):
            return str(node.n)
        return "expr"


class NestedDSLCompiler:
    """
    DSL compiler for RVec<RVec<T>> (nested RVec) expressions.
    
    Compiles Python expressions to C++ code for RDataFrame.
    
    Example:
        >>> schema = {"tracks_pt": "RVec<RVec<double>>"}
        >>> compiler = NestedDSLCompiler.from_schema(schema)
        >>> code, deps = compiler.compile("tracks_pt[:, 0]", "first_pt")
        >>> print(code)  # C++ lambda for column extract
    
    Phase 13.3.DSL D7.
    """
    
    def __init__(self, type_inferrer: TypeInferrer):
        """
        Initialize compiler.
        
        Args:
            type_inferrer: TypeInferrer with schema information
        """
        self.type_inferrer = type_inferrer
        self.analyzer = NestedExpressionAnalyzer(type_inferrer)
        self.code_gen = NestedCodeGenerator()
    
    @classmethod
    def from_schema(cls, schema: Dict[str, str]) -> 'NestedDSLCompiler':
        """
        Create compiler from schema dict.
        
        Args:
            schema: Dict mapping names to C++ types
            
        Returns:
            NestedDSLCompiler instance
        """
        inferrer = TypeInferrer.from_schema(schema)
        return cls(inferrer)
    
    def is_nested_expression(self, expr: str) -> bool:
        """
        Check if expression involves nested RVec operations.
        
        Args:
            expr: Python expression string
            
        Returns:
            True if expression contains nested RVec operations
        """
        result = self.analyzer.analyze(expr)
        return result.is_nested
    
    def compile_expression(self, expr: str) -> Optional[NestedAccessNode]:
        """
        Compile expression to IR node.
        
        Args:
            expr: Python expression string
            
        Returns:
            NestedAccessNode or None if not a nested expression
            
        Raises:
            ValueError: If expression has errors
        """
        result = self.analyzer.analyze(expr)
        
        if not result.is_nested:
            return None
        
        if result.error:
            raise ValueError(result.error)
        
        return result.node
    
    def compile(self, expr: str, result_name: str = None) -> Tuple[str, List[str]]:
        """
        Compile expression to C++ code.
        
        Args:
            expr: Python expression string
            result_name: Optional name for result variable
            
        Returns:
            Tuple of (C++ code, list of required headers)
            
        Raises:
            ValueError: If expression is not a nested expression or has errors
        """
        node = self.compile_expression(expr)
        
        if node is None:
            raise ValueError(f"Not a nested RVec expression: {expr}")
        
        generated = self.code_gen.generate(node)
        
        code = generated.code
        if result_name:
            code = f"auto {result_name} = {code};"
        
        return code, generated.dependencies
    
    def get_result_type(self, expr: str) -> Tuple[str, int]:
        """
        Get result type for an expression.
        
        Args:
            expr: Python expression string
            
        Returns:
            Tuple of (C++ type string, rank)
        """
        node = self.compile_expression(expr)
        
        if node is None:
            raise ValueError(f"Not a nested RVec expression: {expr}")
        
        return node.get_result_cpp_type(), node.result_rank
    
    def get_policy(self, expr: str) -> str:
        """
        Get the ragged policy for an expression.
        
        Args:
            expr: Python expression string
            
        Returns:
            "fail-closed" or "clamp-per-row"
        """
        node = self.compile_expression(expr)
        
        if node is None:
            raise ValueError(f"Not a nested RVec expression: {expr}")
        
        return "fail-closed" if node.fail_closed else "clamp-per-row"


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    schema = {
        "tracks_pt": "RVec<RVec<double>>",
        "tracks_eta": "RVec<RVec<float>>",
        "hits_x": "RVec<RVec<double>>",
    }
    
    compiler = NestedDSLCompiler.from_schema(schema)
    
    test_exprs = [
        # Column extract (fail-closed)
        "tracks_pt[:, 0]",
        "tracks_pt[:, -1]",
        "tracks_pt[:, 2]",
        # Column slice (clamp)
        "tracks_pt[:, :3]",
        "tracks_pt[:, 1:4]",
        "tracks_pt[:, ::2]",
        "tracks_pt[:, ::-1]",
        # Element access
        "tracks_pt[0, 1]",
        "tracks_pt[-1, -1]",
        # Row access
        "tracks_pt[0]",
        "tracks_pt[-1]",
    ]
    
    print("Phase 13.3.DSL D7: NestedDSLCompiler Demo\n")
    print("=" * 60)
    
    for expr in test_exprs:
        try:
            code, deps = compiler.compile(expr)
            cpp_type, rank = compiler.get_result_type(expr)
            policy = compiler.get_policy(expr)
            print(f"\nExpression: {expr}")
            print(f"Result type: {cpp_type} (rank={rank})")
            print(f"Policy: {policy}")
            print(f"Code preview: {code[:60]}...")
        except Exception as e:
            print(f"\nExpression: {expr}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Self-test complete.")
