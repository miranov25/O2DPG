"""
Phase 13.4.DSL D4-D7: IR Nodes for C-Array Access

IR nodes representing C-array indexing and slicing operations.
Supports 1D, 2D, 3D arrays with fixed and variable dimensions.

Reuses SliceParams from Phase 13.3 for slicing semantics.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Union, Tuple

# Reuse SliceParams from Phase 13.3
from RDataFrameDSL.ir_nodes_linalg import SliceParams


__all__ = [
    'CArraySliceKind',
    'CArrayAccessNode',
    'make_carray_element_access',
    'make_carray_slice_access',
    'make_carray_row_access',
    'make_carray_column_access',
    'make_carray_subarray_access',
]


class CArraySliceKind(Enum):
    """Types of C-array access operations."""
    
    # 1D operations
    ELEMENT_1D = auto()       # arr[i] → scalar
    SLICE_1D = auto()         # arr[a:b] → RVec<T>
    
    # 2D operations  
    ELEMENT_2D = auto()       # arr[i, j] → scalar
    ROW_2D = auto()           # arr[i] or arr[i, :] → RVec<T>
    COLUMN_2D = auto()        # arr[:, j] → RVec<T>
    ROW_SLICE_2D = auto()     # arr[a:b, :] → RVec<RVec<T>>
    COLUMN_SLICE_2D = auto()  # arr[:, a:b] → RVec<RVec<T>>
    SUBARRAY_2D = auto()      # arr[a:b, c:d] → RVec<RVec<T>>
    
    # 3D operations
    ELEMENT_3D = auto()       # arr[i, j, k] → scalar
    PLANE_3D = auto()         # arr[i] or arr[i, :, :] → RVec<RVec<T>>
    ROW_3D = auto()           # arr[i, j] or arr[i, j, :] → RVec<T>
    SLICE_DIM0_3D = auto()    # arr[:, j, k] → RVec<T>
    SLICE_DIM1_3D = auto()    # arr[i, :, k] → RVec<T>
    SLICE_DIM2_3D = auto()    # arr[i, j, :] → RVec<T> (same as ROW_3D)
    PLANE_SLICE_3D = auto()   # arr[:, j, :] or similar → RVec<RVec<T>>


@dataclass
class DimensionSpec:
    """
    Specification for a single dimension in C-array.
    
    Attributes:
        size: Size expression (int for fixed, str for counter branch)
        is_fixed: True if compile-time constant
    """
    size: Union[int, str]
    is_fixed: bool
    
    def get_size_expr(self) -> str:
        """Get C++ expression for this dimension's size."""
        return str(self.size)
    
    def get_decl(self, var_name: str) -> str:
        """Get C++ declaration for size variable."""
        if self.is_fixed:
            return f"constexpr int {var_name} = {self.size}"
        else:
            return f"int {var_name} = {self.size}"


@dataclass
class CArrayAccessNode:
    """
    IR node representing a C-array access operation.
    
    Attributes:
        kind: Type of access operation
        source: Source array name
        base_type: Element type (float, double, int)
        dims: List of dimension specifications
        indices: Integer indices for element/row access
        slices: Slice parameters for range access
        result_type: C++ result type
        rank: Result rank (0=scalar, 1=RVec, 2=RVec<RVec>)
    """
    kind: CArraySliceKind
    source: str
    base_type: str
    dims: List[DimensionSpec]
    indices: List[Optional[Union[int, str]]] = field(default_factory=list)
    slices: List[Optional[SliceParams]] = field(default_factory=list)
    result_type: str = ""
    rank: int = 0
    
    def __post_init__(self):
        """Compute result type if not provided."""
        if not self.result_type:
            self.result_type = self._compute_result_type()
    
    def _compute_result_type(self) -> str:
        """Compute C++ result type based on kind."""
        base = self.base_type
        
        if self.rank == 0:
            return base
        elif self.rank == 1:
            return f"ROOT::RVec<{base}>"
        elif self.rank == 2:
            return f"ROOT::RVec<ROOT::RVec<{base}>>"
        else:
            # Rank 3+ (rare)
            result = base
            for _ in range(self.rank):
                result = f"ROOT::RVec<{result}>"
            return result
    
    @property
    def array_rank(self) -> int:
        """Rank of the source array."""
        return len(self.dims)
    
    def get_stride(self, dim: int) -> str:
        """
        Get row-major stride expression for dimension.
        
        For arr[D0][D1][D2]:
        - stride(0) = D1 * D2
        - stride(1) = D2
        - stride(2) = 1
        """
        if dim >= len(self.dims) - 1:
            return "1"
        
        parts = []
        for i in range(dim + 1, len(self.dims)):
            parts.append(self.dims[i].get_size_expr())
        
        if not parts:
            return "1"
        return " * ".join(parts)


# =============================================================================
# Factory Functions
# =============================================================================

def make_carray_element_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    indices: List[Union[int, str]],
) -> CArrayAccessNode:
    """
    Create IR node for element access.
    
    Args:
        source: Array name
        base_type: Element type (float, double)
        dims: List of (size, is_fixed) tuples
        indices: List of index expressions
        
    Returns:
        CArrayAccessNode for element access
        
    Examples:
        arr[0] on float[10] → 1D element
        arr[1, 2] on float[3][4] → 2D element
        arr[0, 1, 2] on float[2][3][4] → 3D element
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    rank = len(dims)
    
    if rank == 1:
        kind = CArraySliceKind.ELEMENT_1D
    elif rank == 2:
        kind = CArraySliceKind.ELEMENT_2D
    else:
        kind = CArraySliceKind.ELEMENT_3D
    
    return CArrayAccessNode(
        kind=kind,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        indices=list(indices),
        rank=0,
    )


def make_carray_slice_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    slice_params: SliceParams,
) -> CArrayAccessNode:
    """
    Create IR node for 1D slice access.
    
    Args:
        source: Array name
        base_type: Element type
        dims: Dimension specs (should be 1D)
        slice_params: Slice parameters
        
    Returns:
        CArrayAccessNode for 1D slice
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    
    return CArrayAccessNode(
        kind=CArraySliceKind.SLICE_1D,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        slices=[slice_params],
        rank=1,
    )


def make_carray_row_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    row_index: Union[int, str],
) -> CArrayAccessNode:
    """
    Create IR node for row access (2D or 3D).
    
    Args:
        source: Array name
        base_type: Element type
        dims: Dimension specs
        row_index: Row index
        
    Returns:
        CArrayAccessNode for row extraction
        
    Examples:
        arr[i] on float[3][4] → RVec<float> size 4
        arr[i, j] on float[2][3][4] → RVec<float> size 4
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    rank = len(dims)
    
    if rank == 2:
        kind = CArraySliceKind.ROW_2D
        result_rank = 1
        indices = [row_index]
    else:  # 3D - need two indices
        kind = CArraySliceKind.ROW_3D
        result_rank = 1
        indices = [row_index]  # Will be extended by caller for 3D
    
    return CArrayAccessNode(
        kind=kind,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        indices=indices,
        rank=result_rank,
    )


def make_carray_column_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    col_index: Union[int, str],
) -> CArrayAccessNode:
    """
    Create IR node for column access.
    
    Args:
        source: Array name
        base_type: Element type
        dims: Dimension specs (2D)
        col_index: Column index
        
    Returns:
        CArrayAccessNode for column extraction
        
    Example:
        arr[:, j] on float[3][4] → RVec<float> size 3
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    
    return CArrayAccessNode(
        kind=CArraySliceKind.COLUMN_2D,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        indices=[None, col_index],  # None for sliced dimension
        rank=1,
    )


def make_carray_subarray_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    row_slice: Optional[SliceParams],
    col_slice: Optional[SliceParams],
) -> CArrayAccessNode:
    """
    Create IR node for 2D subarray access.
    
    Args:
        source: Array name
        base_type: Element type
        dims: Dimension specs (2D)
        row_slice: Row slice params (None = all rows)
        col_slice: Column slice params (None = all columns)
        
    Returns:
        CArrayAccessNode for subarray
        
    Examples:
        arr[0:2, :] → first 2 rows
        arr[:, 0:2] → first 2 columns
        arr[0:2, 0:2] → 2x2 subarray
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    
    # Determine kind based on which dimensions are sliced
    if row_slice and col_slice:
        kind = CArraySliceKind.SUBARRAY_2D
    elif row_slice:
        kind = CArraySliceKind.ROW_SLICE_2D
    else:
        kind = CArraySliceKind.COLUMN_SLICE_2D
    
    return CArrayAccessNode(
        kind=kind,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        slices=[row_slice, col_slice],
        rank=2,
    )


def make_carray_plane_access(
    source: str,
    base_type: str,
    dims: List[Tuple[Union[int, str], bool]],
    plane_index: Union[int, str],
) -> CArrayAccessNode:
    """
    Create IR node for 3D plane access.
    
    Args:
        source: Array name
        base_type: Element type
        dims: Dimension specs (3D)
        plane_index: First dimension index
        
    Returns:
        CArrayAccessNode for plane extraction
        
    Example:
        arr[i] on float[2][3][4] → RVec<RVec<float>> (3x4)
    """
    dim_specs = [DimensionSpec(size=s, is_fixed=f) for s, f in dims]
    
    return CArrayAccessNode(
        kind=CArraySliceKind.PLANE_3D,
        source=source,
        base_type=base_type,
        dims=dim_specs,
        indices=[plane_index],
        rank=2,
    )
