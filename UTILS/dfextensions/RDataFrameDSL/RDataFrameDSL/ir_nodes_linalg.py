"""
Phase 13.3.DSL D5+D6: IR Nodes for ROOT Linear Algebra Types

This module defines IR node classes for TMatrixD/TVectorD operations:
- LinalgSliceKind: Enum for different access patterns
- SliceParams: Represents [start:stop:step] slice parameters
- LinalgAccessNode: IR node for matrix/vector access operations

These nodes are used by the DSL compiler to represent linalg operations
before C++ code generation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union, Any

from .ir_types import IRType, IRTypeKind


__all__ = [
    'LinalgSliceKind',
    'SliceParams',
    'LinalgAccessNode',
    # Factory functions
    'make_matrix_element_access',
    'make_matrix_row_access',
    'make_matrix_column_access',
    'make_matrix_submatrix_access',
    'make_vector_element_access',
    'make_vector_slice_access',
]


class LinalgSliceKind(Enum):
    """
    Types of linear algebra access operations.
    
    Phase 13.3.DSL D5+D6.
    """
    # Matrix operations (D5)
    MATRIX_ELEMENT = auto()      # mat(i, j) or mat[i, j] → scalar
    MATRIX_ROW = auto()          # mat[i] or mat[i, :] → RVec<T>
    MATRIX_COLUMN = auto()       # mat[:, j] → RVec<T>
    MATRIX_SUBMATRIX = auto()    # mat[a:b, c:d] → RVec<RVec<T>>
    
    # Vector operations (D6)
    VECTOR_ELEMENT = auto()      # vec[i] → scalar
    VECTOR_SLICE = auto()        # vec[:n], vec[::2], etc. → RVec<T>


@dataclass
class SliceParams:
    """
    Represents Python slice parameters [start:stop:step].
    
    All fields are optional to handle cases like:
    - [:n]  → start=None, stop=n, step=None
    - [n:]  → start=n, stop=None, step=None
    - [::2] → start=None, stop=None, step=2
    - [::-1] → start=None, stop=None, step=-1
    
    Values can be:
    - int: Literal index
    - str: Variable name or expression
    - None: Not specified (use default)
    """
    start: Optional[Union[int, str]] = None
    stop: Optional[Union[int, str]] = None
    step: Optional[Union[int, str]] = None
    
    def is_full_slice(self) -> bool:
        """Check if this is [:] (select all)."""
        return self.start is None and self.stop is None and self.step is None
    
    def is_first_n(self) -> bool:
        """Check if this is [:n] pattern."""
        return self.start is None and self.stop is not None and self.step is None
    
    def is_from_n(self) -> bool:
        """Check if this is [n:] pattern."""
        return self.start is not None and self.stop is None and self.step is None
    
    def is_range(self) -> bool:
        """Check if this is [a:b] pattern."""
        return self.start is not None and self.stop is not None and self.step is None
    
    def is_step_slice(self) -> bool:
        """Check if this has a step (includes reverse)."""
        return self.step is not None
    
    def is_reverse(self) -> bool:
        """Check if this is [::-1] (reverse)."""
        return self.step == -1 or self.step == "-1"
    
    def __repr__(self) -> str:
        parts = []
        if self.start is not None:
            parts.append(str(self.start))
        parts.append(":")
        if self.stop is not None:
            parts.append(str(self.stop))
        if self.step is not None:
            parts.append(":")
            parts.append(str(self.step))
        return f"SliceParams([{''.join(parts)}])"


@dataclass
class LinalgAccessNode:
    """
    IR node representing a linear algebra access operation.
    
    This node captures all information needed to generate C++ code
    for accessing elements, rows, columns, or slices of TMatrixD/TVectorD.
    
    Attributes:
        target: The matrix/vector variable name
        access_kind: Type of access (element, row, column, slice, etc.)
        row_index: Row index for matrix operations (int, str, or SliceParams)
        col_index: Column index for matrix operations (int, str, or SliceParams)
        slice_params: Slice parameters for vector slicing
        element_type: C++ element type ("double" or "float")
        result_type: IRType of the result
        result_rank: Rank of result (0=scalar, 1=vector, 2=nested)
        safe_indexing: Whether to use bounds-checked access
        source_expr: Original Python expression (for error messages)
    
    Phase 13.3.DSL D5+D6.
    """
    target: str
    access_kind: LinalgSliceKind
    row_index: Optional[Union[int, str, SliceParams]] = None
    col_index: Optional[Union[int, str, SliceParams]] = None
    slice_params: Optional[SliceParams] = None
    element_type: str = "double"
    result_type: Optional[IRType] = None
    result_rank: int = 0
    safe_indexing: bool = True
    source_expr: Optional[str] = None
    
    def __post_init__(self):
        """Compute result_type if not provided."""
        if self.result_type is None:
            self.result_type = self._infer_result_type()
    
    def _infer_result_type(self) -> IRType:
        """Infer the result IRType based on access kind."""
        # Determine base type kind
        if self.element_type == "float":
            base_kind = IRTypeKind.Float32
        else:
            base_kind = IRTypeKind.Float64
        
        # Scalar results
        if self.access_kind in (LinalgSliceKind.MATRIX_ELEMENT, 
                                 LinalgSliceKind.VECTOR_ELEMENT):
            return IRType(base_kind)
        
        # Vector results (RVec<T>)
        if self.access_kind in (LinalgSliceKind.MATRIX_ROW,
                                 LinalgSliceKind.MATRIX_COLUMN,
                                 LinalgSliceKind.VECTOR_SLICE):
            return IRType(base_kind)  # Element type, rank handled separately
        
        # Nested vector results (RVec<RVec<T>>)
        if self.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX:
            return IRType(base_kind)
        
        return IRType(IRTypeKind.Unknown)
    
    def get_result_cpp_type(self) -> str:
        """Get the C++ type string for the result."""
        elem = self.element_type
        
        if self.access_kind in (LinalgSliceKind.MATRIX_ELEMENT,
                                 LinalgSliceKind.VECTOR_ELEMENT):
            return elem
        
        if self.access_kind in (LinalgSliceKind.MATRIX_ROW,
                                 LinalgSliceKind.MATRIX_COLUMN,
                                 LinalgSliceKind.VECTOR_SLICE):
            return f"ROOT::RVec<{elem}>"
        
        if self.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX:
            return f"ROOT::RVec<ROOT::RVec<{elem}>>"
        
        return elem
    
    def is_scalar_result(self) -> bool:
        """Check if result is a scalar."""
        return self.access_kind in (LinalgSliceKind.MATRIX_ELEMENT,
                                     LinalgSliceKind.VECTOR_ELEMENT)
    
    def is_vector_result(self) -> bool:
        """Check if result is RVec<T>."""
        return self.access_kind in (LinalgSliceKind.MATRIX_ROW,
                                     LinalgSliceKind.MATRIX_COLUMN,
                                     LinalgSliceKind.VECTOR_SLICE)
    
    def is_nested_result(self) -> bool:
        """Check if result is RVec<RVec<T>>."""
        return self.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX


# =============================================================================
# Factory Functions
# =============================================================================

def make_matrix_element_access(
    target: str,
    row: Union[int, str],
    col: Union[int, str],
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for matrix element access: mat(i, j) or mat[i, j].
    
    Args:
        target: Matrix variable name
        row: Row index (int or expression string)
        col: Column index (int or expression string)
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression for error messages
        
    Returns:
        LinalgAccessNode configured for element access
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.MATRIX_ELEMENT,
        row_index=row,
        col_index=col,
        element_type=element_type,
        result_rank=0,
        safe_indexing=safe,
        source_expr=source
    )


def make_matrix_row_access(
    target: str,
    row: Union[int, str],
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for matrix row extraction: mat[i] or mat[i, :].
    
    Args:
        target: Matrix variable name
        row: Row index
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression
        
    Returns:
        LinalgAccessNode configured for row extraction
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.MATRIX_ROW,
        row_index=row,
        col_index=SliceParams(),  # Full slice ":"
        element_type=element_type,
        result_rank=1,
        safe_indexing=safe,
        source_expr=source
    )


def make_matrix_column_access(
    target: str,
    col: Union[int, str],
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for matrix column extraction: mat[:, j].
    
    Args:
        target: Matrix variable name
        col: Column index
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression
        
    Returns:
        LinalgAccessNode configured for column extraction
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.MATRIX_COLUMN,
        row_index=SliceParams(),  # Full slice ":"
        col_index=col,
        element_type=element_type,
        result_rank=1,
        safe_indexing=safe,
        source_expr=source
    )


def make_matrix_submatrix_access(
    target: str,
    row_slice: SliceParams,
    col_slice: SliceParams,
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for submatrix extraction: mat[a:b, c:d].
    
    Args:
        target: Matrix variable name
        row_slice: Row slice parameters
        col_slice: Column slice parameters
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression
        
    Returns:
        LinalgAccessNode configured for submatrix extraction
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.MATRIX_SUBMATRIX,
        row_index=row_slice,
        col_index=col_slice,
        element_type=element_type,
        result_rank=2,
        safe_indexing=safe,
        source_expr=source
    )


def make_vector_element_access(
    target: str,
    index: Union[int, str],
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for vector element access: vec[i].
    
    Args:
        target: Vector variable name
        index: Element index
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression
        
    Returns:
        LinalgAccessNode configured for element access
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.VECTOR_ELEMENT,
        row_index=index,  # Use row_index for vector index
        element_type=element_type,
        result_rank=0,
        safe_indexing=safe,
        source_expr=source
    )


def make_vector_slice_access(
    target: str,
    slice_params: SliceParams,
    element_type: str = "double",
    safe: bool = True,
    source: str = None
) -> LinalgAccessNode:
    """
    Create IR node for vector slicing: vec[:n], vec[::2], etc.
    
    Args:
        target: Vector variable name
        slice_params: Slice parameters
        element_type: "double" or "float"
        safe: Use bounds-checked access
        source: Original expression
        
    Returns:
        LinalgAccessNode configured for slicing
    """
    return LinalgAccessNode(
        target=target,
        access_kind=LinalgSliceKind.VECTOR_SLICE,
        slice_params=slice_params,
        element_type=element_type,
        result_rank=1,
        safe_indexing=safe,
        source_expr=source
    )
