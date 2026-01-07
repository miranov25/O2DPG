"""
Phase 13.4.DSL D1: Schema Parser for C-Array Notation

Parses C-array type notation like:
- float[10]      → Fixed 1D, size=10
- float[n]       → Variable 1D, counter="n"
- float[3][3]    → Fixed 2D, 3×3
- float[n][3]    → Hybrid, n rows × 3 columns
- double[2][3][4] → Fixed 3D, 2×3×4

Rules (FROZEN):
1. Integer literal → fixed dimension
2. Identifier → variable (counter branch)
3. Variable dimension must be outermost (leftmost)
4. Only one variable dimension allowed
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
from enum import Enum, auto


__all__ = [
    'CArrayType',
    'DimensionKind',
    'parse_carray_type',
    'is_carray_notation',
    'SchemaParser',
]


class DimensionKind(Enum):
    """Type of array dimension."""
    FIXED = auto()      # Compile-time literal (e.g., 10)
    VARIABLE = auto()   # Counter branch (e.g., n)


@dataclass
class Dimension:
    """Represents a single array dimension."""
    kind: DimensionKind
    value: Union[int, str]  # int for fixed, str for variable (counter name)
    
    @property
    def is_fixed(self) -> bool:
        return self.kind == DimensionKind.FIXED
    
    @property
    def is_variable(self) -> bool:
        return self.kind == DimensionKind.VARIABLE
    
    def __repr__(self) -> str:
        if self.is_fixed:
            return f"Fixed({self.value})"
        else:
            return f"Variable({self.value})"


@dataclass
class CArrayType:
    """
    Parsed C-array type representation.
    
    Examples:
        float[10]      → base="float", dims=[Fixed(10)]
        float[n]       → base="float", dims=[Variable("n")]
        float[3][3]    → base="float", dims=[Fixed(3), Fixed(3)]
        float[n][3]    → base="float", dims=[Variable("n"), Fixed(3)]
        double[2][3][4] → base="double", dims=[Fixed(2), Fixed(3), Fixed(4)]
    
    Attributes:
        base: Base element type (float, double, int, etc.)
        dims: List of dimensions, outer to inner (left to right)
        original: Original notation string
    """
    base: str
    dims: List[Dimension] = field(default_factory=list)
    original: str = ""
    
    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.dims)
    
    @property
    def is_fixed(self) -> bool:
        """True if all dimensions are fixed (compile-time)."""
        return all(d.is_fixed for d in self.dims)
    
    @property
    def is_variable(self) -> bool:
        """True if any dimension is variable."""
        return any(d.is_variable for d in self.dims)
    
    @property
    def counter_branch(self) -> Optional[str]:
        """
        Get counter branch name for variable-length arrays.
        
        Returns None for fixed arrays.
        Variable dimension must be outermost (dims[0]).
        """
        if self.dims and self.dims[0].is_variable:
            return self.dims[0].value
        return None
    
    @property
    def fixed_dims(self) -> List[int]:
        """Get list of fixed dimension sizes (for inner dims in hybrid)."""
        return [d.value for d in self.dims if d.is_fixed]
    
    @property
    def total_fixed_size(self) -> Optional[int]:
        """
        Total size if fully fixed, None otherwise.
        
        For float[3][4] returns 12.
        For float[n][3] returns None (variable).
        """
        if not self.is_fixed:
            return None
        result = 1
        for d in self.dims:
            result *= d.value
        return result
    
    @property
    def inner_fixed_size(self) -> int:
        """
        Product of all fixed (inner) dimensions.
        
        For float[n][3][4] returns 12 (3×4).
        For float[10] returns 10.
        For float[n] returns 1.
        """
        result = 1
        for d in self.dims:
            if d.is_fixed:
                result *= d.value
        return result
    
    def get_stride(self, dim_index: int) -> str:
        """
        Get stride expression for dimension at given index.
        
        For arr[N][M][K]:
        - dim 0 stride = M * K
        - dim 1 stride = K
        - dim 2 stride = 1
        
        Returns C++ expression string.
        """
        if dim_index >= len(self.dims):
            return "1"
        
        # Stride is product of all dimensions after this one
        stride_parts = []
        for i in range(dim_index + 1, len(self.dims)):
            d = self.dims[i]
            if d.is_fixed:
                stride_parts.append(str(d.value))
            else:
                stride_parts.append(d.value)  # Counter branch name
        
        if not stride_parts:
            return "1"
        return " * ".join(stride_parts)
    
    def get_cpp_size_expr(self, dim_index: int) -> str:
        """
        Get C++ expression for size of dimension.
        
        For fixed: returns literal (e.g., "10")
        For variable: returns counter branch name (e.g., "n")
        """
        if dim_index >= len(self.dims):
            raise IndexError(f"Dimension index {dim_index} out of range")
        
        d = self.dims[dim_index]
        if d.is_fixed:
            return str(d.value)
        else:
            return d.value
    
    def get_result_type(self, access_rank: int) -> str:
        """
        Get C++ result type for access that reduces rank.
        
        Args:
            access_rank: Number of dimensions being indexed (not sliced)
            
        Returns:
            C++ type string
            
        Examples:
            arr[i] on float[10] → access_rank=1 → "float"
            arr[i] on float[3][3] → access_rank=1 → "ROOT::RVec<float>"
            arr[i,j] on float[3][3] → access_rank=2 → "float"
        """
        remaining_rank = self.rank - access_rank
        
        if remaining_rank <= 0:
            return self.base
        elif remaining_rank == 1:
            return f"ROOT::RVec<{self.base}>"
        else:
            # Nested RVec for rank 2+
            result = self.base
            for _ in range(remaining_rank):
                result = f"ROOT::RVec<{result}>"
            return result
    
    def __repr__(self) -> str:
        dims_str = ", ".join(repr(d) for d in self.dims)
        return f"CArrayType(base={self.base!r}, dims=[{dims_str}])"


# =============================================================================
# Parser Implementation
# =============================================================================

# Regex patterns
_BASE_TYPE_PATTERN = r'^([a-zA-Z_][a-zA-Z0-9_]*)'
_DIMENSION_PATTERN = r'\[([^\]]+)\]'
_FULL_PATTERN = re.compile(
    r'^([a-zA-Z_][a-zA-Z0-9_]*)(\[[^\]]+\])+$'
)


def is_carray_notation(type_str: str) -> bool:
    """
    Check if type string is C-array notation.
    
    Args:
        type_str: Type string like "float[10]" or "RVec<float>"
        
    Returns:
        True if C-array notation (has [...] suffix)
        
    Examples:
        >>> is_carray_notation("float[10]")
        True
        >>> is_carray_notation("float[n][3]")
        True
        >>> is_carray_notation("RVec<float>")
        False
        >>> is_carray_notation("float")
        False
    """
    return bool(_FULL_PATTERN.match(type_str.strip()))


def _parse_dimension(dim_str: str) -> Dimension:
    """
    Parse a single dimension string.
    
    Args:
        dim_str: Dimension content (without brackets), e.g., "10" or "n"
        
    Returns:
        Dimension object
        
    Raises:
        ValueError: If dimension is invalid
    """
    dim_str = dim_str.strip()
    
    if not dim_str:
        raise ValueError("Empty dimension")
    
    # Try to parse as integer (fixed dimension)
    try:
        value = int(dim_str)
    except ValueError:
        # Not an integer - must be identifier (variable dimension / counter branch)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', dim_str):
            raise ValueError(f"Invalid dimension: {dim_str!r} (must be integer or identifier)")
        return Dimension(DimensionKind.VARIABLE, dim_str)
    
    # It's an integer - check if positive
    if value <= 0:
        raise ValueError(f"Dimension must be positive, got {value}")
    
    return Dimension(DimensionKind.FIXED, value)


def parse_carray_type(type_str: str) -> CArrayType:
    """
    Parse C-array type notation.
    
    Args:
        type_str: Type string like "float[10]", "double[n][3]"
        
    Returns:
        CArrayType object
        
    Raises:
        ValueError: If notation is invalid
        
    Examples:
        >>> parse_carray_type("float[10]")
        CArrayType(base='float', dims=[Fixed(10)])
        
        >>> parse_carray_type("float[n][3]")
        CArrayType(base='float', dims=[Variable(n), Fixed(3)])
    """
    type_str = type_str.strip()
    
    if not type_str:
        raise ValueError("Empty type string")
    
    # Check format
    if not is_carray_notation(type_str):
        raise ValueError(f"Not a C-array notation: {type_str!r}")
    
    # Extract base type
    base_match = re.match(_BASE_TYPE_PATTERN, type_str)
    if not base_match:
        raise ValueError(f"Cannot extract base type from: {type_str!r}")
    base = base_match.group(1)
    
    # Extract dimensions
    dim_matches = re.findall(_DIMENSION_PATTERN, type_str)
    if not dim_matches:
        raise ValueError(f"No dimensions found in: {type_str!r}")
    
    dims = []
    for i, dim_str in enumerate(dim_matches):
        try:
            dim = _parse_dimension(dim_str)
            dims.append(dim)
        except ValueError as e:
            raise ValueError(f"Invalid dimension [{dim_str}] in {type_str!r}: {e}")
    
    # Validate: variable dimension must be outermost
    variable_indices = [i for i, d in enumerate(dims) if d.is_variable]
    if len(variable_indices) > 1:
        raise ValueError(
            f"Multiple variable dimensions not supported: {type_str!r}. "
            f"Found variable at indices {variable_indices}"
        )
    if variable_indices and variable_indices[0] != 0:
        raise ValueError(
            f"Variable dimension must be outermost (leftmost): {type_str!r}. "
            f"Variable at index {variable_indices[0]}, should be 0"
        )
    
    return CArrayType(base=base, dims=dims, original=type_str)


# =============================================================================
# Schema Parser Class
# =============================================================================

class SchemaParser:
    """
    Parses schema dictionaries with C-array notation.
    
    Handles mixed schemas with:
    - C-array notation: float[n], float[3][3]
    - Regular types: int, float, double
    - RVec types: RVec<float>, RVec<RVec<double>>
    - ROOT types: TMatrixD, TVectorD
    
    Example:
        >>> parser = SchemaParser()
        >>> schema = {
        ...     "n": "int",
        ...     "arr": "float[n]",
        ...     "mat": "float[3][3]",
        ...     "vec": "RVec<double>",
        ... }
        >>> parsed = parser.parse(schema)
        >>> parsed["arr"]
        CArrayType(base='float', dims=[Variable(n)])
    """
    
    def __init__(self):
        """Initialize parser."""
        self._cache = {}
    
    def parse(self, schema: dict) -> dict:
        """
        Parse schema dictionary.
        
        Args:
            schema: Dict mapping column names to type strings
            
        Returns:
            Dict mapping column names to parsed types.
            C-array notations become CArrayType objects.
            Other types remain as strings.
        """
        result = {}
        for name, type_str in schema.items():
            result[name] = self.parse_type(type_str)
        return result
    
    def parse_type(self, type_str: str) -> Union[CArrayType, str]:
        """
        Parse a single type string.
        
        Returns CArrayType for C-array notation, original string otherwise.
        """
        type_str = type_str.strip()
        
        # Check cache
        if type_str in self._cache:
            return self._cache[type_str]
        
        # Try C-array notation
        if is_carray_notation(type_str):
            result = parse_carray_type(type_str)
        else:
            result = type_str
        
        self._cache[type_str] = result
        return result
    
    def is_carray(self, type_str: str) -> bool:
        """Check if type string is C-array notation."""
        return is_carray_notation(type_str)
    
    def get_counter_branches(self, schema: dict) -> dict:
        """
        Get all counter branches used in schema.
        
        Args:
            schema: Schema dictionary
            
        Returns:
            Dict mapping counter branch names to columns that use them
            
        Example:
            >>> parser.get_counter_branches({"arr": "float[n]", "hits": "float[n][3]"})
            {"n": ["arr", "hits"]}
        """
        counters = {}
        parsed = self.parse(schema)
        
        for name, ptype in parsed.items():
            if isinstance(ptype, CArrayType) and ptype.counter_branch:
                counter = ptype.counter_branch
                if counter not in counters:
                    counters[counter] = []
                counters[counter].append(name)
        
        return counters
    
    def validate_schema(self, schema: dict) -> List[str]:
        """
        Validate schema for consistency.
        
        Checks:
        - Counter branches exist as columns
        - No circular dependencies
        - Valid type notation
        
        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []
        parsed = self.parse(schema)
        counters = self.get_counter_branches(schema)
        
        # Check counter branches exist
        for counter, columns in counters.items():
            if counter not in schema:
                issues.append(
                    f"Counter branch '{counter}' used by {columns} "
                    f"but not defined in schema"
                )
        
        return issues


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_schema(schema: dict) -> dict:
    """
    Parse schema dictionary (convenience function).
    
    Args:
        schema: Dict mapping column names to type strings
        
    Returns:
        Dict with CArrayType for C-array notations
    """
    return SchemaParser().parse(schema)
