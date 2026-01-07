"""
Phase 13.4.DSL D4-D7: DSL Compiler for C-Array Expressions

Compiles Python-like expressions into C++ code for C-array access.
Integrates with schema parser from D1.

Supported expressions:
- arr[i]           → element access (1D)
- arr[a:b]         → slice (1D)
- arr[i, j]        → element access (2D)
- arr[i]           → row access (2D)
- arr[:, j]        → column access (2D)
- arr[a:b, c:d]    → subarray (2D)
- arr[i, j, k]     → element access (3D)
- arr[i, :, :]     → plane access (3D)
- Similar patterns for 3D
"""

import ast
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any

from RDataFrameDSL.schema_parser import (
    CArrayType,
    SchemaParser,
    is_carray_notation,
    parse_carray_type,
)
from RDataFrameDSL.ir_nodes_linalg import SliceParams
from RDataFrameDSL.ir_nodes_carray import (
    CArraySliceKind,
    CArrayAccessNode,
    DimensionSpec,
    make_carray_element_access,
    make_carray_slice_access,
    make_carray_row_access,
    make_carray_column_access,
    make_carray_subarray_access,
    make_carray_plane_access,
)
from RDataFrameDSL.backend_carray import generate_carray_code, CArrayCodeResult


__all__ = [
    'CArrayDSLCompiler',
    'CArrayExpressionAnalyzer',
]


@dataclass
class IndexSpec:
    """Specification for a single index in an access expression."""
    is_slice: bool
    value: Optional[Union[int, str]] = None  # For element access
    start: Optional[Union[int, str]] = None  # For slice
    stop: Optional[Union[int, str]] = None
    step: Optional[Union[int, str]] = None
    is_full_slice: bool = False  # True for [:] (all elements)
    
    @classmethod
    def from_element(cls, value: Union[int, str]) -> 'IndexSpec':
        """Create element index spec."""
        return cls(is_slice=False, value=value)
    
    @classmethod
    def from_slice(cls, start=None, stop=None, step=None) -> 'IndexSpec':
        """Create slice index spec."""
        is_full = start is None and stop is None and step is None
        return cls(
            is_slice=True,
            start=start,
            stop=stop,
            step=step,
            is_full_slice=is_full,
        )
    
    def to_slice_params(self) -> SliceParams:
        """Convert to SliceParams for code generation."""
        return SliceParams(
            start=self.start,
            stop=self.stop,
            step=self.step,
        )


class CArrayExpressionAnalyzer:
    """
    Analyzes Python expressions to extract C-array access patterns.
    
    Parses expressions like:
    - arr[0]
    - arr[:5]
    - arr[1, 2]
    - arr[:, 0]
    - arr[0:2, 1:3]
    - arr[0, :, 2]
    """
    
    def __init__(self, schema: Dict[str, Union[CArrayType, str]]):
        """
        Initialize analyzer with parsed schema.
        
        Args:
            schema: Dict mapping column names to types (CArrayType or str)
        """
        self.schema = schema
    
    def analyze(self, expr: str) -> Optional[Tuple[str, CArrayType, List[IndexSpec]]]:
        """
        Analyze expression and extract access pattern.
        
        Args:
            expr: Expression string like "arr[0, 1]"
            
        Returns:
            Tuple of (array_name, array_type, index_specs) or None if not C-array
        """
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return None
        
        return self._analyze_node(tree.body)
    
    def _analyze_node(self, node: ast.AST) -> Optional[Tuple[str, CArrayType, List[IndexSpec]]]:
        """Analyze AST node."""
        if isinstance(node, ast.Subscript):
            return self._analyze_subscript(node)
        return None
    
    def _analyze_subscript(self, node: ast.Subscript) -> Optional[Tuple[str, CArrayType, List[IndexSpec]]]:
        """Analyze subscript expression."""
        # Get array name
        if isinstance(node.value, ast.Name):
            array_name = node.value.id
        elif isinstance(node.value, ast.Subscript):
            # Chained subscript like arr[i][j]
            result = self._analyze_chained_subscript(node)
            return result
        else:
            return None
        
        # Check if it's a C-array
        if array_name not in self.schema:
            return None
        
        array_type = self.schema[array_name]
        if not isinstance(array_type, CArrayType):
            return None
        
        # Parse indices
        indices = self._parse_indices(node.slice)
        
        return (array_name, array_type, indices)
    
    def _analyze_chained_subscript(self, node: ast.Subscript) -> Optional[Tuple[str, CArrayType, List[IndexSpec]]]:
        """
        Analyze chained subscript like arr[i][j].
        
        Converts arr[i][j] to equivalent of arr[i, j].
        """
        indices = []
        current = node
        
        while isinstance(current, ast.Subscript):
            idx = self._parse_single_index(current.slice)
            indices.insert(0, idx)
            current = current.value
        
        if not isinstance(current, ast.Name):
            return None
        
        array_name = current.id
        
        if array_name not in self.schema:
            return None
        
        array_type = self.schema[array_name]
        if not isinstance(array_type, CArrayType):
            return None
        
        return (array_name, array_type, indices)
    
    def _parse_indices(self, slice_node: ast.AST) -> List[IndexSpec]:
        """Parse index/slice specification from AST."""
        if isinstance(slice_node, ast.Tuple):
            # Multiple indices: arr[i, j] or arr[:, j]
            return [self._parse_single_index(elt) for elt in slice_node.elts]
        else:
            # Single index: arr[i] or arr[:5]
            return [self._parse_single_index(slice_node)]
    
    def _parse_single_index(self, node: ast.AST) -> IndexSpec:
        """Parse a single index or slice."""
        if isinstance(node, ast.Slice):
            start = self._eval_index_expr(node.lower)
            stop = self._eval_index_expr(node.upper)
            step = self._eval_index_expr(node.step)
            return IndexSpec.from_slice(start, stop, step)
        else:
            value = self._eval_index_expr(node)
            return IndexSpec.from_element(value)
    
    def _eval_index_expr(self, node: Optional[ast.AST]) -> Optional[Union[int, str]]:
        """Evaluate index expression to int or variable name."""
        if node is None:
            return None
        
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Negative number
            inner = self._eval_index_expr(node.operand)
            if isinstance(inner, int):
                return -inner
            return f"-{inner}"
        elif isinstance(node, ast.BinOp):
            # Expression like n-1
            left = self._eval_index_expr(node.left)
            right = self._eval_index_expr(node.right)
            op = self._get_op_str(node.op)
            return f"({left} {op} {right})"
        else:
            # Fallback: use ast.unparse if available
            try:
                return ast.unparse(node)
            except AttributeError:
                return str(node)
    
    def _get_op_str(self, op: ast.operator) -> str:
        """Convert AST operator to string."""
        ops = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
        }
        return ops.get(type(op), '?')


class CArrayDSLCompiler:
    """
    DSL compiler for C-array expressions.
    
    Compiles Python-like array access expressions to C++ code.
    
    Example:
        >>> schema = {"n": "int", "arr": "float[n]", "mat": "float[3][4]"}
        >>> compiler = CArrayDSLCompiler.from_schema(schema)
        >>> code, deps = compiler.compile("arr[0]", "first")
        >>> print(code)
        auto first = [&]() { ... }();
    """
    
    def __init__(self, parsed_schema: Dict[str, Union[CArrayType, str]]):
        """
        Initialize compiler with parsed schema.
        
        Args:
            parsed_schema: Dict from SchemaParser.parse()
        """
        self.schema = parsed_schema
        self.analyzer = CArrayExpressionAnalyzer(parsed_schema)
    
    @classmethod
    def from_schema(cls, schema: Dict[str, str]) -> 'CArrayDSLCompiler':
        """
        Create compiler from raw schema dictionary.
        
        Args:
            schema: Dict mapping column names to type strings
            
        Returns:
            CArrayDSLCompiler instance
        """
        parser = SchemaParser()
        parsed = parser.parse(schema)
        return cls(parsed)
    
    def is_carray_expression(self, expr: str) -> bool:
        """
        Check if expression accesses a C-array.
        
        Args:
            expr: Expression string
            
        Returns:
            True if expression accesses a C-array column
        """
        result = self.analyzer.analyze(expr)
        return result is not None
    
    def compile(self, expr: str, result_name: str) -> Tuple[str, set]:
        """
        Compile expression to C++ code.
        
        Args:
            expr: Expression string like "arr[0, 1]"
            result_name: Name for the result variable
            
        Returns:
            Tuple of (C++ code, set of dependencies)
            
        Raises:
            ValueError: If expression is invalid or not a C-array access
        """
        analysis = self.analyzer.analyze(expr)
        if analysis is None:
            raise ValueError(f"Not a valid C-array expression: {expr}")
        
        array_name, array_type, indices = analysis
        
        # Create IR node
        ir_node = self._create_ir_node(array_name, array_type, indices)
        
        # Generate code
        result = generate_carray_code(ir_node)
        
        # Format as variable assignment
        code = f"auto {result_name} = {result.code};"
        
        return code, result.dependencies
    
    def get_result_type(self, expr: str) -> Tuple[str, int]:
        """
        Get result type for expression.
        
        Args:
            expr: Expression string
            
        Returns:
            Tuple of (C++ type string, rank)
        """
        analysis = self.analyzer.analyze(expr)
        if analysis is None:
            raise ValueError(f"Not a valid C-array expression: {expr}")
        
        array_name, array_type, indices = analysis
        ir_node = self._create_ir_node(array_name, array_type, indices)
        
        return ir_node.result_type, ir_node.rank
    
    def _create_ir_node(
        self,
        array_name: str,
        array_type: CArrayType,
        indices: List[IndexSpec],
    ) -> CArrayAccessNode:
        """Create IR node from analyzed expression."""
        
        # Convert CArrayType dims to tuples for factory functions
        dims = [(d.value, d.is_fixed) for d in array_type.dims]
        base_type = array_type.base
        rank = array_type.rank
        
        # Determine access pattern based on array rank and indices
        if rank == 1:
            return self._create_1d_node(array_name, base_type, dims, indices)
        elif rank == 2:
            return self._create_2d_node(array_name, base_type, dims, indices)
        else:  # rank == 3
            return self._create_3d_node(array_name, base_type, dims, indices)
    
    def _create_1d_node(
        self,
        name: str,
        base: str,
        dims: List[Tuple],
        indices: List[IndexSpec],
    ) -> CArrayAccessNode:
        """Create IR node for 1D array access."""
        if len(indices) != 1:
            raise ValueError(f"1D array requires 1 index, got {len(indices)}")
        
        idx = indices[0]
        
        if idx.is_slice:
            return make_carray_slice_access(name, base, dims, idx.to_slice_params())
        else:
            return make_carray_element_access(name, base, dims, [idx.value])
    
    def _create_2d_node(
        self,
        name: str,
        base: str,
        dims: List[Tuple],
        indices: List[IndexSpec],
    ) -> CArrayAccessNode:
        """Create IR node for 2D array access."""
        
        if len(indices) == 1:
            # arr[i] → row access
            idx = indices[0]
            if idx.is_slice:
                # arr[:] or arr[a:b] on 2D → row slice
                return make_carray_subarray_access(
                    name, base, dims,
                    idx.to_slice_params(),
                    None,
                )
            else:
                return make_carray_row_access(name, base, dims, idx.value)
        
        elif len(indices) == 2:
            row_idx, col_idx = indices
            
            if not row_idx.is_slice and not col_idx.is_slice:
                # arr[i, j] → element
                return make_carray_element_access(
                    name, base, dims,
                    [row_idx.value, col_idx.value],
                )
            
            elif row_idx.is_slice and not col_idx.is_slice:
                # arr[:, j] → column
                if row_idx.is_full_slice:
                    return make_carray_column_access(name, base, dims, col_idx.value)
                else:
                    # arr[a:b, j] → partial column
                    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
                    return CArrayAccessNode(
                        kind=CArraySliceKind.COLUMN_2D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[None, col_idx.value],
                        slices=[row_idx.to_slice_params(), None],
                        rank=1,
                    )
            
            elif not row_idx.is_slice and col_idx.is_slice:
                # arr[i, :] or arr[i, a:b] → row or partial row
                if col_idx.is_full_slice:
                    return make_carray_row_access(name, base, dims, row_idx.value)
                else:
                    # Partial row - create custom node
                    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
                    return CArrayAccessNode(
                        kind=CArraySliceKind.ROW_2D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[row_idx.value],
                        slices=[None, col_idx.to_slice_params()],
                        rank=1,
                    )
            
            else:
                # arr[a:b, c:d] → subarray
                return make_carray_subarray_access(
                    name, base, dims,
                    row_idx.to_slice_params() if not row_idx.is_full_slice else None,
                    col_idx.to_slice_params() if not col_idx.is_full_slice else None,
                )
        
        raise ValueError(f"Invalid index count for 2D array: {len(indices)}")
    
    def _create_3d_node(
        self,
        name: str,
        base: str,
        dims: List[Tuple],
        indices: List[IndexSpec],
    ) -> CArrayAccessNode:
        """Create IR node for 3D array access."""
        dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
        
        if len(indices) == 1:
            idx = indices[0]
            if idx.is_slice:
                # arr[:] on 3D - slice along first dimension
                raise ValueError("Full 3D copy not supported; use arr[a:b] for plane slice")
            else:
                # arr[i] → plane
                return make_carray_plane_access(name, base, dims, idx.value)
        
        elif len(indices) == 2:
            i_idx, j_idx = indices
            
            if not i_idx.is_slice and not j_idx.is_slice:
                # arr[i, j] → row (innermost dimension)
                node = CArrayAccessNode(
                    kind=CArraySliceKind.ROW_3D,
                    source=name,
                    base_type=base,
                    dims=dim_specs,
                    indices=[i_idx.value, j_idx.value],
                    rank=1,
                )
                return node
            
            elif i_idx.is_slice and not j_idx.is_slice:
                # arr[:, j] → slice along dim0 at fixed j
                node = CArrayAccessNode(
                    kind=CArraySliceKind.PLANE_SLICE_3D,
                    source=name,
                    base_type=base,
                    dims=dim_specs,
                    indices=[None, j_idx.value],
                    rank=2,
                )
                return node
            
            elif not i_idx.is_slice and j_idx.is_slice:
                # arr[i, :] → 2D plane at fixed i (same as arr[i])
                return make_carray_plane_access(name, base, dims, i_idx.value)
            
            else:
                raise ValueError("Slice on both dims requires 3 indices for 3D")
        
        elif len(indices) == 3:
            i_idx, j_idx, k_idx = indices
            
            # Count slices
            slices = [idx.is_slice for idx in indices]
            n_slices = sum(slices)
            
            if n_slices == 0:
                # arr[i, j, k] → element
                return make_carray_element_access(
                    name, base, dims,
                    [i_idx.value, j_idx.value, k_idx.value],
                )
            
            elif n_slices == 1:
                # One slice → 1D result
                if i_idx.is_slice:
                    # arr[:, j, k]
                    return CArrayAccessNode(
                        kind=CArraySliceKind.SLICE_DIM0_3D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[None, j_idx.value, k_idx.value],
                        rank=1,
                    )
                elif j_idx.is_slice:
                    # arr[i, :, k]
                    return CArrayAccessNode(
                        kind=CArraySliceKind.SLICE_DIM1_3D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[i_idx.value, None, k_idx.value],
                        rank=1,
                    )
                else:
                    # arr[i, j, :]
                    return CArrayAccessNode(
                        kind=CArraySliceKind.ROW_3D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[i_idx.value, j_idx.value],
                        slices=[None, None, k_idx.to_slice_params()],
                        rank=1,
                    )
            
            elif n_slices == 2:
                # Two slices → 2D result
                if not i_idx.is_slice:
                    # arr[i, :, :] → plane
                    return make_carray_plane_access(name, base, dims, i_idx.value)
                elif not j_idx.is_slice:
                    # arr[:, j, :] → plane slice
                    return CArrayAccessNode(
                        kind=CArraySliceKind.PLANE_SLICE_3D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[None, j_idx.value, None],
                        rank=2,
                    )
                else:
                    # arr[:, :, k] → 2D slice at fixed k
                    return CArrayAccessNode(
                        kind=CArraySliceKind.PLANE_SLICE_3D,
                        source=name,
                        base_type=base,
                        dims=dim_specs,
                        indices=[None, None, k_idx.value],
                        rank=2,
                    )
            
            else:
                # All slices → 3D copy (not supported)
                raise ValueError("Full 3D copy (arr[:, :, :]) not supported")
        
        raise ValueError(f"Invalid index count for 3D array: {len(indices)}")
