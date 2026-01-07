"""
Phase 13.3.DSL D7: IR Nodes Extension for Nested RVec 2D Slicing

Extends ir_nodes_linalg.py with support for RVec<RVec<T>> operations:
- NESTED_COLUMN_EXTRACT: nested[:, j] → RVec<T> (fail-closed)
- NESTED_COLUMN_SLICE: nested[:, a:b] → RVec<RVec<T>> (clamp per-row)

Two-Tier Ragged Policy:
- Integer index ([:, j]): Fail-closed - throws if ANY row missing element j
- Slice/range ([:, a:b]): Clamp per-row - variable-length results OK
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union, List

from .ir_types import IRType, IRTypeKind


__all__ = [
    # New slice kinds for nested RVec
    'NestedSliceKind',
    'NestedAccessNode',
    # Factory functions
    'make_nested_column_extract',
    'make_nested_column_slice',
    'make_nested_element_access',
    'make_nested_row_access',
    # Re-export SliceParams for convenience
    'SliceParams',
]

# Import SliceParams from existing module
from .ir_nodes_linalg import SliceParams


class NestedSliceKind(Enum):
    """
    Types of nested RVec (RVec<RVec<T>>) access operations.
    
    Phase 13.3.DSL D7.
    
    Two-Tier Ragged Policy:
    - Integer column index: Fail-closed (throw if any row too short)
    - Slice column index: Clamp per-row (variable results OK)
    """
    # Row operations (same as existing RVec)
    NESTED_ROW_ACCESS = auto()       # nested[i] → RVec<T>
    NESTED_ROW_SLICE = auto()        # nested[:n] → RVec<RVec<T>>
    
    # Column operations (NEW in D7)
    NESTED_COLUMN_EXTRACT = auto()   # nested[:, j] → RVec<T> (FAIL-CLOSED)
    NESTED_COLUMN_SLICE = auto()     # nested[:, a:b] → RVec<RVec<T>> (CLAMP)
    
    # Element access
    NESTED_ELEMENT = auto()          # nested[i, j] → T (scalar)


@dataclass
class NestedAccessNode:
    """
    IR node representing a nested RVec (RVec<RVec<T>>) access operation.
    
    This node captures all information needed to generate C++ code
    for accessing elements, rows, or columns of RVec<RVec<T>>.
    
    Attributes:
        target: The nested RVec variable name
        access_kind: Type of access (row, column extract, column slice, etc.)
        row_index: Row index or slice (int, str, or SliceParams)
        col_index: Column index or slice (int, str, or SliceParams)
        element_type: C++ inner element type (e.g., "double", "float", "int")
        result_type: IRType of the result
        result_rank: Rank of result (0=scalar, 1=RVec<T>, 2=RVec<RVec<T>>)
        fail_closed: If True, throw on out-of-bounds; if False, clamp/skip
        source_expr: Original Python expression (for error messages)
    
    Phase 13.3.DSL D7.
    """
    target: str
    access_kind: NestedSliceKind
    row_index: Optional[Union[int, str, SliceParams]] = None
    col_index: Optional[Union[int, str, SliceParams]] = None
    element_type: str = "double"
    result_type: Optional[IRType] = None
    result_rank: int = 0
    fail_closed: bool = True  # Default: fail-closed for safety
    source_expr: Optional[str] = None
    
    def __post_init__(self):
        """Compute result_type and fail_closed based on access kind."""
        if self.result_type is None:
            self.result_type = self._infer_result_type()
        
        # Set fail_closed based on access kind (per two-tier policy)
        if self.access_kind == NestedSliceKind.NESTED_COLUMN_EXTRACT:
            self.fail_closed = True  # Integer index → fail-closed
        elif self.access_kind == NestedSliceKind.NESTED_COLUMN_SLICE:
            self.fail_closed = False  # Slice → clamp per-row
    
    def _infer_result_type(self) -> IRType:
        """Infer the result IRType based on access kind."""
        # Determine base type kind
        type_map = {
            "float": IRTypeKind.Float32,
            "double": IRTypeKind.Float64,
            "int": IRTypeKind.Int32,
            "long": IRTypeKind.Int64,
            "bool": IRTypeKind.Bool,
        }
        base_kind = type_map.get(self.element_type, IRTypeKind.Float64)
        
        # Scalar results
        if self.access_kind == NestedSliceKind.NESTED_ELEMENT:
            return IRType(base_kind)
        
        # RVec<T> results
        if self.access_kind in (NestedSliceKind.NESTED_ROW_ACCESS,
                                 NestedSliceKind.NESTED_COLUMN_EXTRACT):
            return IRType(base_kind)
        
        # RVec<RVec<T>> results
        if self.access_kind in (NestedSliceKind.NESTED_ROW_SLICE,
                                 NestedSliceKind.NESTED_COLUMN_SLICE):
            return IRType(base_kind)
        
        return IRType(IRTypeKind.Unknown)
    
    def get_result_cpp_type(self) -> str:
        """Get the C++ type string for the result."""
        elem = self.element_type
        
        # Scalar
        if self.access_kind == NestedSliceKind.NESTED_ELEMENT:
            return elem
        
        # RVec<T>
        if self.access_kind in (NestedSliceKind.NESTED_ROW_ACCESS,
                                 NestedSliceKind.NESTED_COLUMN_EXTRACT):
            return f"ROOT::RVec<{elem}>"
        
        # RVec<RVec<T>>
        if self.access_kind in (NestedSliceKind.NESTED_ROW_SLICE,
                                 NestedSliceKind.NESTED_COLUMN_SLICE):
            return f"ROOT::RVec<ROOT::RVec<{elem}>>"
        
        return elem
    
    def is_scalar_result(self) -> bool:
        """Check if result is a scalar."""
        return self.access_kind == NestedSliceKind.NESTED_ELEMENT
    
    def is_vector_result(self) -> bool:
        """Check if result is RVec<T>."""
        return self.access_kind in (NestedSliceKind.NESTED_ROW_ACCESS,
                                     NestedSliceKind.NESTED_COLUMN_EXTRACT)
    
    def is_nested_result(self) -> bool:
        """Check if result is RVec<RVec<T>>."""
        return self.access_kind in (NestedSliceKind.NESTED_ROW_SLICE,
                                     NestedSliceKind.NESTED_COLUMN_SLICE)


# =============================================================================
# Factory Functions
# =============================================================================

def make_nested_column_extract(
    target: str,
    col_index: Union[int, str],
    element_type: str = "double",
    source: str = None
) -> NestedAccessNode:
    """
    Create IR node for nested column extraction: nested[:, j] → RVec<T>.
    
    Uses FAIL-CLOSED policy: throws if ANY row is missing element j.
    
    Args:
        target: Nested RVec variable name
        col_index: Column index (int or expression)
        element_type: Inner element type
        source: Original expression for errors
        
    Returns:
        NestedAccessNode configured for column extraction
        
    Example:
        >>> node = make_nested_column_extract("tracks", 0)
        >>> # Extracts first element from each inner RVec
        >>> # Throws if any inner RVec is empty
    """
    return NestedAccessNode(
        target=target,
        access_kind=NestedSliceKind.NESTED_COLUMN_EXTRACT,
        row_index=SliceParams(),  # Full slice [:] for rows
        col_index=col_index,
        element_type=element_type,
        result_rank=1,
        fail_closed=True,  # FAIL-CLOSED for integer index
        source_expr=source
    )


def make_nested_column_slice(
    target: str,
    col_slice: SliceParams,
    element_type: str = "double",
    source: str = None
) -> NestedAccessNode:
    """
    Create IR node for nested column slicing: nested[:, a:b] → RVec<RVec<T>>.
    
    Uses CLAMP-PER-ROW policy: each row is clamped independently,
    resulting in variable-length inner RVecs.
    
    Args:
        target: Nested RVec variable name
        col_slice: Column slice parameters
        element_type: Inner element type
        source: Original expression for errors
        
    Returns:
        NestedAccessNode configured for column slicing
        
    Example:
        >>> node = make_nested_column_slice("tracks", SliceParams(stop=3))
        >>> # Extracts first 3 elements from each inner RVec (or fewer)
    """
    return NestedAccessNode(
        target=target,
        access_kind=NestedSliceKind.NESTED_COLUMN_SLICE,
        row_index=SliceParams(),  # Full slice [:] for rows
        col_index=col_slice,
        element_type=element_type,
        result_rank=2,
        fail_closed=False,  # CLAMP for slice
        source_expr=source
    )


def make_nested_element_access(
    target: str,
    row: Union[int, str],
    col: Union[int, str],
    element_type: str = "double",
    source: str = None
) -> NestedAccessNode:
    """
    Create IR node for nested element access: nested[i, j] → T.
    
    Args:
        target: Nested RVec variable name
        row: Row index
        col: Column index
        element_type: Inner element type
        source: Original expression
        
    Returns:
        NestedAccessNode configured for element access
    """
    return NestedAccessNode(
        target=target,
        access_kind=NestedSliceKind.NESTED_ELEMENT,
        row_index=row,
        col_index=col,
        element_type=element_type,
        result_rank=0,
        fail_closed=True,
        source_expr=source
    )


def make_nested_row_access(
    target: str,
    row: Union[int, str],
    element_type: str = "double",
    source: str = None
) -> NestedAccessNode:
    """
    Create IR node for nested row access: nested[i] → RVec<T>.
    
    Args:
        target: Nested RVec variable name
        row: Row index
        element_type: Inner element type
        source: Original expression
        
    Returns:
        NestedAccessNode configured for row access
    """
    return NestedAccessNode(
        target=target,
        access_kind=NestedSliceKind.NESTED_ROW_ACCESS,
        row_index=row,
        col_index=None,
        element_type=element_type,
        result_rank=1,
        fail_closed=True,
        source_expr=source
    )
