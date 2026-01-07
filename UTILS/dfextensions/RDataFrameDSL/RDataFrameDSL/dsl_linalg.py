"""
Phase 13.3.DSL D5+D6: DSL Compiler for ROOT Linear Algebra Types

This module provides the DSL compiler integration for TMatrixD/TVectorD:
- LinalgExpressionAnalyzer: Parses Python expressions to detect linalg ops
- LinalgDSLCompiler: Compiles expressions to C++ code

Supported syntax:
- mat(i, j)     → scalar (ROOT-style function call)
- mat[i, j]     → scalar (Python tuple indexing)  
- mat[i][j]     → scalar (C++ chained indexing)
- mat[i]        → RVec<T> (row extraction)
- mat[i, :]     → RVec<T> (explicit row extraction)
- mat[:, j]     → RVec<T> (column extraction)
- mat[a:b, c:d] → RVec<RVec<T>> (submatrix)
- vec[i]        → scalar
- vec[:n]       → RVec<T> (first n)
- vec[n:]       → RVec<T> (from n)
- vec[::k]      → RVec<T> (step k)
- vec[::-1]     → RVec<T> (reverse)
"""

import ast
import re
from typing import Optional, Dict, Tuple, Union, List, Any
from dataclasses import dataclass

from .type_inferrer import (
    TypeInferrer,
    is_root_matrix_type,
    is_root_vector_type,
    is_root_linalg_type,
    get_linalg_element_type,
)
from .ir_nodes_linalg import (
    LinalgAccessNode, LinalgSliceKind, SliceParams,
    make_matrix_element_access,
    make_matrix_row_access,
    make_matrix_column_access,
    make_matrix_submatrix_access,
    make_vector_element_access,
    make_vector_slice_access,
)
from .backend_linalg import LinalgCodeGenerator, GeneratedCode


__all__ = [
    'LinalgExpressionAnalyzer',
    'LinalgDSLCompiler',
]


@dataclass
class AnalysisResult:
    """Result of analyzing an expression for linalg operations."""
    is_linalg: bool
    node: Optional[LinalgAccessNode] = None
    error: Optional[str] = None


class LinalgExpressionAnalyzer:
    """
    Analyzes Python expressions to detect and parse linalg operations.
    
    Handles all three TMatrixD access syntaxes:
    1. mat(i, j)   - ROOT-style function call
    2. mat[i, j]   - Python tuple indexing
    3. mat[i][j]   - C++ chained indexing
    
    Also handles TVectorD slicing patterns.
    
    Phase 13.3.DSL D5+D6.
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
        Analyze an expression for linear algebra operations.
        
        Args:
            expr: Python expression string
            
        Returns:
            AnalysisResult with is_linalg flag and optional IR node
        """
        try:
            tree = ast.parse(expr, mode='eval')
            return self._analyze_node(tree.body, expr)
        except SyntaxError as e:
            return AnalysisResult(is_linalg=False, error=f"Syntax error: {e}")
    
    def _analyze_node(self, node: ast.AST, source: str) -> AnalysisResult:
        """Analyze an AST node."""
        # Check for function call syntax: mat(i, j)
        if isinstance(node, ast.Call):
            return self._analyze_call(node, source)
        
        # Check for subscript syntax: mat[...] or vec[...]
        if isinstance(node, ast.Subscript):
            return self._analyze_subscript(node, source)
        
        return AnalysisResult(is_linalg=False)
    
    def _analyze_call(self, node: ast.Call, source: str) -> AnalysisResult:
        """
        Analyze function call: mat(i, j).
        
        This is the ROOT-style matrix element access.
        """
        # Get function name
        if not isinstance(node.func, ast.Name):
            return AnalysisResult(is_linalg=False)
        
        name = node.func.id
        
        # Check if it's a matrix variable
        if not self._is_matrix_var(name):
            return AnalysisResult(is_linalg=False)
        
        # Must have exactly 2 arguments for mat(i, j)
        if len(node.args) != 2:
            return AnalysisResult(
                is_linalg=True,
                error=f"Matrix call {name}() requires exactly 2 arguments"
            )
        
        row = self._ast_to_index(node.args[0])
        col = self._ast_to_index(node.args[1])
        elem_type = self._get_element_type(name)
        
        ir_node = make_matrix_element_access(
            target=name,
            row=row,
            col=col,
            element_type=elem_type,
            source=source
        )
        
        return AnalysisResult(is_linalg=True, node=ir_node)
    
    def _analyze_subscript(self, node: ast.Subscript, source: str) -> AnalysisResult:
        """
        Analyze subscript expression: mat[...] or vec[...].
        
        Handles:
        - mat[i, j] (tuple index)
        - mat[i][j] (chained subscript)
        - mat[i] or mat[i, :] (row)
        - mat[:, j] (column)
        - mat[a:b, c:d] (submatrix)
        - vec[i] (element)
        - vec[:n], vec[::k], etc. (slices)
        """
        # Check for chained subscript: mat[i][j]
        if isinstance(node.value, ast.Subscript):
            return self._analyze_chained_subscript(node, source)
        
        # Get variable name
        if not isinstance(node.value, ast.Name):
            return AnalysisResult(is_linalg=False)
        
        name = node.value.id
        
        # Check if it's a linalg variable
        if self._is_matrix_var(name):
            return self._analyze_matrix_subscript(name, node.slice, source)
        elif self._is_vector_var(name):
            return self._analyze_vector_subscript(name, node.slice, source)
        
        return AnalysisResult(is_linalg=False)
    
    def _analyze_chained_subscript(self, node: ast.Subscript, 
                                    source: str) -> AnalysisResult:
        """
        Analyze chained subscript: mat[i][j].
        
        This is equivalent to mat[i, j] for element access.
        """
        # node.value is the first subscript (mat[i])
        inner = node.value
        if not isinstance(inner, ast.Subscript):
            return AnalysisResult(is_linalg=False)
        
        # Get the variable name
        if not isinstance(inner.value, ast.Name):
            return AnalysisResult(is_linalg=False)
        
        name = inner.value.id
        
        if not self._is_matrix_var(name):
            return AnalysisResult(is_linalg=False)
        
        row = self._ast_to_index(inner.slice)
        col = self._ast_to_index(node.slice)
        elem_type = self._get_element_type(name)
        
        ir_node = make_matrix_element_access(
            target=name,
            row=row,
            col=col,
            element_type=elem_type,
            source=source
        )
        
        return AnalysisResult(is_linalg=True, node=ir_node)
    
    def _analyze_matrix_subscript(self, name: str, slice_node: ast.AST,
                                   source: str) -> AnalysisResult:
        """Analyze matrix subscript: mat[...]."""
        elem_type = self._get_element_type(name)
        
        # Check for tuple (2D indexing): mat[i, j] or mat[i, :] or mat[:, j]
        if isinstance(slice_node, ast.Tuple):
            if len(slice_node.elts) != 2:
                return AnalysisResult(
                    is_linalg=True,
                    error=f"Matrix {name}[...] requires exactly 2 indices"
                )
            
            row_node = slice_node.elts[0]
            col_node = slice_node.elts[1]
            
            row_is_slice = isinstance(row_node, ast.Slice)
            col_is_slice = isinstance(col_node, ast.Slice)
            
            if row_is_slice and col_is_slice:
                # Submatrix: mat[a:b, c:d]
                row_slice = self._ast_to_slice_params(row_node)
                col_slice = self._ast_to_slice_params(col_node)
                ir_node = make_matrix_submatrix_access(
                    target=name,
                    row_slice=row_slice,
                    col_slice=col_slice,
                    element_type=elem_type,
                    source=source
                )
            elif row_is_slice:
                # Column extraction: mat[:, j]
                col = self._ast_to_index(col_node)
                ir_node = make_matrix_column_access(
                    target=name,
                    col=col,
                    element_type=elem_type,
                    source=source
                )
            elif col_is_slice:
                # Row extraction: mat[i, :]
                row = self._ast_to_index(row_node)
                ir_node = make_matrix_row_access(
                    target=name,
                    row=row,
                    element_type=elem_type,
                    source=source
                )
            else:
                # Element access: mat[i, j]
                row = self._ast_to_index(row_node)
                col = self._ast_to_index(col_node)
                ir_node = make_matrix_element_access(
                    target=name,
                    row=row,
                    col=col,
                    element_type=elem_type,
                    source=source
                )
            
            return AnalysisResult(is_linalg=True, node=ir_node)
        
        # Single index: mat[i] → row extraction
        if isinstance(slice_node, ast.Slice):
            # mat[:] - doesn't make sense for matrix, return error
            return AnalysisResult(
                is_linalg=True,
                error=f"Matrix {name}[:] is ambiguous. Use mat[:, :] for full matrix."
            )
        
        row = self._ast_to_index(slice_node)
        ir_node = make_matrix_row_access(
            target=name,
            row=row,
            element_type=elem_type,
            source=source
        )
        
        return AnalysisResult(is_linalg=True, node=ir_node)
    
    def _analyze_vector_subscript(self, name: str, slice_node: ast.AST,
                                   source: str) -> AnalysisResult:
        """Analyze vector subscript: vec[...]."""
        elem_type = self._get_element_type(name)
        
        # Check for slice: vec[:n], vec[::k], etc.
        if isinstance(slice_node, ast.Slice):
            params = self._ast_to_slice_params(slice_node)
            ir_node = make_vector_slice_access(
                target=name,
                slice_params=params,
                element_type=elem_type,
                source=source
            )
            return AnalysisResult(is_linalg=True, node=ir_node)
        
        # Single index: vec[i] → element access
        index = self._ast_to_index(slice_node)
        ir_node = make_vector_element_access(
            target=name,
            index=index,
            element_type=elem_type,
            source=source
        )
        
        return AnalysisResult(is_linalg=True, node=ir_node)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _is_matrix_var(self, name: str) -> bool:
        """Check if variable is a matrix type."""
        try:
            return self.type_inferrer.is_matrix_type(name)
        except:
            return False
    
    def _is_vector_var(self, name: str) -> bool:
        """Check if variable is a linalg vector type."""
        try:
            return self.type_inferrer.is_vector_linalg_type(name)
        except:
            return False
    
    def _get_element_type(self, name: str) -> str:
        """Get element type for a linalg variable."""
        try:
            cpp_type = self.type_inferrer.get_cpp_type(name)
            return get_linalg_element_type(cpp_type)
        except:
            return "double"
    
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
        if isinstance(node, ast.BinOp):
            left = self._node_to_str(node.left)
            right = self._node_to_str(node.right)
            op = {
                ast.Add: '+', ast.Sub: '-', ast.Mult: '*', 
                ast.Div: '/', ast.Mod: '%'
            }.get(type(node.op), '?')
            return f"({left} {op} {right})"
        return "expr"


class LinalgDSLCompiler:
    """
    DSL compiler for TMatrixD/TVectorD expressions.
    
    Compiles Python expressions to C++ code for RDataFrame.
    
    Example:
        >>> schema = {"cov": "TMatrixD", "params": "TVectorD"}
        >>> compiler = LinalgDSLCompiler.from_schema(schema)
        >>> code, deps = compiler.compile("cov[0, 1]", "cov_01")
        >>> print(code)  # C++ lambda for element access
    
    Phase 13.3.DSL D5+D6.
    """
    
    def __init__(self, type_inferrer: TypeInferrer, safe_indexing: bool = True):
        """
        Initialize compiler.
        
        Args:
            type_inferrer: TypeInferrer with schema information
            safe_indexing: Whether to generate bounds-checked code
        """
        self.type_inferrer = type_inferrer
        self.analyzer = LinalgExpressionAnalyzer(type_inferrer)
        self.code_gen = LinalgCodeGenerator(safe_indexing=safe_indexing)
    
    @classmethod
    def from_schema(cls, schema: Dict[str, str], 
                    safe_indexing: bool = True) -> 'LinalgDSLCompiler':
        """
        Create compiler from schema dict.
        
        Args:
            schema: Dict mapping names to C++ types
            safe_indexing: Whether to generate bounds-checked code
            
        Returns:
            LinalgDSLCompiler instance
        """
        inferrer = TypeInferrer.from_schema(schema)
        return cls(inferrer, safe_indexing)
    
    def is_linalg_expression(self, expr: str) -> bool:
        """
        Check if expression involves linalg operations.
        
        Args:
            expr: Python expression string
            
        Returns:
            True if expression contains linalg operations
        """
        result = self.analyzer.analyze(expr)
        return result.is_linalg
    
    def compile_expression(self, expr: str) -> Optional[LinalgAccessNode]:
        """
        Compile expression to IR node.
        
        Args:
            expr: Python expression string
            
        Returns:
            LinalgAccessNode or None if not a linalg expression
            
        Raises:
            ValueError: If expression has errors
        """
        result = self.analyzer.analyze(expr)
        
        if not result.is_linalg:
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
            ValueError: If expression is not a linalg expression or has errors
        """
        node = self.compile_expression(expr)
        
        if node is None:
            raise ValueError(f"Not a linalg expression: {expr}")
        
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
            raise ValueError(f"Not a linalg expression: {expr}")
        
        return node.get_result_cpp_type(), node.result_rank


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    schema = {
        "cov": "TMatrixD",
        "err": "TMatrixF", 
        "params": "TVectorD",
        "weights": "TVectorF",
    }
    
    compiler = LinalgDSLCompiler.from_schema(schema)
    
    test_exprs = [
        # Matrix element access (3 syntaxes)
        "cov(0, 1)",
        "cov[0, 1]",
        "cov[0][1]",
        # Matrix row/column
        "cov[0]",
        "cov[0, :]",
        "cov[:, 1]",
        # Matrix submatrix
        "cov[0:2, 1:3]",
        # Vector element
        "params[0]",
        # Vector slices
        "params[:3]",
        "params[2:]",
        "params[::2]",
        "params[::-1]",
        "params[-3:]",
    ]
    
    print("Phase 13.3.DSL D5+D6: LinalgDSLCompiler Demo\n")
    print("=" * 60)
    
    for expr in test_exprs:
        try:
            code, deps = compiler.compile(expr)
            cpp_type, rank = compiler.get_result_type(expr)
            print(f"\nExpression: {expr}")
            print(f"Result type: {cpp_type} (rank={rank})")
            print(f"Dependencies: {deps}")
            print(f"Code preview: {code[:80]}...")
        except Exception as e:
            print(f"\nExpression: {expr}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Self-test complete.")
