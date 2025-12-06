"""
Type system for RDataFrame DSL IR.

This module defines the type representation used throughout the IR.
Types map between Python concepts and C++ types for RDataFrame code generation.

Type Hierarchy:
- Numeric: Float32, Float64, Int32, Int64, UInt32, UInt64
- Boolean: Bool
- Object: C++ class types (TParticle, o2::tpc::TrackTPC, etc.)
- Unknown: Unresolved type (error state)

Type Promotion Rules:
- Float + anything numeric = Float (wider wins)
- Int + Int = Int (wider wins)
- Comparison ops always return Bool
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict

__all__ = [
    'IRTypeKind',
    'IRType',
    'promote_types',
    'comparison_result_type',
    'CPP_TO_IR_TYPE',
    'IR_TO_CPP_TYPE',
]


class IRTypeKind(Enum):
    """Enumeration of IR type kinds."""
    Float32 = "float"
    Float64 = "double"
    Int32 = "int"
    Int64 = "long long"
    UInt32 = "unsigned int"
    UInt64 = "unsigned long"
    Bool = "bool"
    Object = "object"      # C++ class type
    Unknown = "unknown"    # Unresolved/error state


@dataclass
class IRType:
    """
    Represents a type in the IR.
    
    Attributes:
        kind: The type kind (Float32, Int64, Object, etc.)
        cpp_type: Full C++ type name for Object kinds (e.g., "o2::tpc::TrackTPC")
        
    Examples:
        >>> IRType(IRTypeKind.Float64)  # double
        >>> IRType(IRTypeKind.Object, "TParticle")  # TParticle object
    """
    kind: IRTypeKind
    cpp_type: Optional[str] = None
    
    def __str__(self) -> str:
        """Return human-readable type string."""
        if self.kind == IRTypeKind.Object:
            return self.cpp_type or "object"
        return self.kind.value
    
    def __repr__(self) -> str:
        if self.kind == IRTypeKind.Object:
            return f"IRType({self.kind.name}, cpp_type={self.cpp_type!r})"
        return f"IRType({self.kind.name})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, IRType):
            return False
        if self.kind != other.kind:
            return False
        if self.kind == IRTypeKind.Object:
            return self.cpp_type == other.cpp_type
        return True
    
    def __hash__(self) -> int:
        if self.kind == IRTypeKind.Object:
            return hash((self.kind, self.cpp_type))
        return hash(self.kind)
    
    def is_numeric(self) -> bool:
        """Check if type is numeric (float or int)."""
        return self.kind in (
            IRTypeKind.Float32, IRTypeKind.Float64,
            IRTypeKind.Int32, IRTypeKind.Int64,
            IRTypeKind.UInt32, IRTypeKind.UInt64
        )
    
    def is_float(self) -> bool:
        """Check if type is floating-point."""
        return self.kind in (IRTypeKind.Float32, IRTypeKind.Float64)
    
    def is_int(self) -> bool:
        """Check if type is integer (signed or unsigned)."""
        return self.kind in (
            IRTypeKind.Int32, IRTypeKind.Int64,
            IRTypeKind.UInt32, IRTypeKind.UInt64
        )
    
    def is_signed_int(self) -> bool:
        """Check if type is signed integer."""
        return self.kind in (IRTypeKind.Int32, IRTypeKind.Int64)
    
    def is_unsigned_int(self) -> bool:
        """Check if type is unsigned integer."""
        return self.kind in (IRTypeKind.UInt32, IRTypeKind.UInt64)
    
    def is_bool(self) -> bool:
        """Check if type is boolean."""
        return self.kind == IRTypeKind.Bool
    
    def is_object(self) -> bool:
        """Check if type is a C++ object."""
        return self.kind == IRTypeKind.Object
    
    def is_unknown(self) -> bool:
        """Check if type is unknown/unresolved."""
        return self.kind == IRTypeKind.Unknown
    
    def bit_width(self) -> int:
        """Return bit width of numeric types, 0 for non-numeric."""
        widths = {
            IRTypeKind.Float32: 32,
            IRTypeKind.Float64: 64,
            IRTypeKind.Int32: 32,
            IRTypeKind.Int64: 64,
            IRTypeKind.UInt32: 32,
            IRTypeKind.UInt64: 64,
            IRTypeKind.Bool: 8,
        }
        return widths.get(self.kind, 0)
    
    def to_cpp(self) -> str:
        """Return C++ type string for code generation."""
        if self.kind == IRTypeKind.Object:
            return self.cpp_type or "/* unknown object */"
        return self.kind.value


def promote_types(left: IRType, right: IRType) -> IRType:
    """
    Determine result type of binary arithmetic operation.
    
    Promotion rules:
    1. If either is float, result is float (wider wins)
    2. If both are int, result is int (wider wins, signed wins over unsigned)
    3. If either is unknown, result is unknown
    4. Object types cannot be promoted (returns Unknown)
    
    Args:
        left: Left operand type
        right: Right operand type
        
    Returns:
        Result type after promotion
        
    Examples:
        >>> promote_types(IRType(IRTypeKind.Float32), IRType(IRTypeKind.Int32))
        IRType(Float32)
        >>> promote_types(IRType(IRTypeKind.Int32), IRType(IRTypeKind.Int64))
        IRType(Int64)
    """
    # Unknown propagates
    if left.is_unknown() or right.is_unknown():
        return IRType(IRTypeKind.Unknown)
    
    # Objects cannot be promoted arithmetically
    if left.is_object() or right.is_object():
        return IRType(IRTypeKind.Unknown)
    
    # Bool treated as int for arithmetic
    if left.is_bool() and right.is_bool():
        return IRType(IRTypeKind.Bool)
    
    # Float + anything numeric = Float (wider wins)
    if left.is_float() or right.is_float():
        if left.kind == IRTypeKind.Float64 or right.kind == IRTypeKind.Float64:
            return IRType(IRTypeKind.Float64)
        return IRType(IRTypeKind.Float32)
    
    # Int + Int = wider Int
    if left.is_int() and right.is_int():
        # 64-bit wins
        if left.kind in (IRTypeKind.Int64, IRTypeKind.UInt64) or \
           right.kind in (IRTypeKind.Int64, IRTypeKind.UInt64):
            # Prefer signed if either is signed
            if left.is_signed_int() or right.is_signed_int():
                return IRType(IRTypeKind.Int64)
            return IRType(IRTypeKind.UInt64)
        # 32-bit
        if left.is_signed_int() or right.is_signed_int():
            return IRType(IRTypeKind.Int32)
        return IRType(IRTypeKind.UInt32)
    
    # Bool + numeric
    if left.is_bool() or right.is_bool():
        other = right if left.is_bool() else left
        return other
    
    # Fallback
    return IRType(IRTypeKind.Unknown)


def comparison_result_type() -> IRType:
    """Return the type of comparison operations (always Bool)."""
    return IRType(IRTypeKind.Bool)


def division_result_type(left: IRType, right: IRType) -> IRType:
    """
    Determine result type of division operation.
    
    Division always promotes to float to avoid integer truncation issues.
    Use floor division (//) explicitly for integer division.
    """
    # Unknown propagates
    if left.is_unknown() or right.is_unknown():
        return IRType(IRTypeKind.Unknown)
    
    # Division result is always float
    if left.kind == IRTypeKind.Float64 or right.kind == IRTypeKind.Float64:
        return IRType(IRTypeKind.Float64)
    
    # For int/int or float32, use float32 at minimum
    if left.bit_width() <= 32 and right.bit_width() <= 32:
        return IRType(IRTypeKind.Float32)
    
    return IRType(IRTypeKind.Float64)


# =============================================================================
# Type Mapping Tables
# =============================================================================

# C++ type string → IRTypeKind
CPP_TO_IR_TYPE: Dict[str, IRTypeKind] = {
    # Standard C++ floats
    "float": IRTypeKind.Float32,
    "double": IRTypeKind.Float64,
    "long double": IRTypeKind.Float64,  # Map to double for simplicity
    
    # ROOT typedefs for floats
    "Float_t": IRTypeKind.Float32,
    "Double_t": IRTypeKind.Float64,
    
    # Standard C++ signed integers
    "int": IRTypeKind.Int32,
    "long": IRTypeKind.Int64,
    "long long": IRTypeKind.Int64,
    "short": IRTypeKind.Int32,  # Promote to Int32
    "char": IRTypeKind.Int32,   # Promote to Int32
    
    # ROOT typedefs for signed integers
    "Int_t": IRTypeKind.Int32,
    "Long_t": IRTypeKind.Int64,
    "Long64_t": IRTypeKind.Int64,
    "Short_t": IRTypeKind.Int32,
    "Char_t": IRTypeKind.Int32,
    
    # Standard C++ unsigned integers
    "unsigned int": IRTypeKind.UInt32,
    "unsigned long": IRTypeKind.UInt64,
    "unsigned long long": IRTypeKind.UInt64,
    "unsigned short": IRTypeKind.UInt32,
    "unsigned char": IRTypeKind.UInt32,
    "size_t": IRTypeKind.UInt64,
    
    # ROOT typedefs for unsigned integers
    "UInt_t": IRTypeKind.UInt32,
    "ULong_t": IRTypeKind.UInt64,
    "ULong64_t": IRTypeKind.UInt64,
    "UShort_t": IRTypeKind.UInt32,
    "UChar_t": IRTypeKind.UInt32,
    
    # Boolean
    "bool": IRTypeKind.Bool,
    "Bool_t": IRTypeKind.Bool,
}

# IRTypeKind → C++ type string (for code generation)
IR_TO_CPP_TYPE: Dict[IRTypeKind, str] = {
    IRTypeKind.Float32: "float",
    IRTypeKind.Float64: "double",
    IRTypeKind.Int32: "int",
    IRTypeKind.Int64: "long long",
    IRTypeKind.UInt32: "unsigned int",
    IRTypeKind.UInt64: "unsigned long long",
    IRTypeKind.Bool: "bool",
    IRTypeKind.Unknown: "/* unknown */",
}


def cpp_type_to_ir(cpp_type: str) -> IRType:
    """
    Convert C++ type string to IRType.
    
    Args:
        cpp_type: C++ type name (e.g., "float", "Int_t", "TParticle")
        
    Returns:
        Corresponding IRType. Returns Object type for unknown class names.
    """
    # Strip whitespace and const/reference qualifiers
    clean_type = cpp_type.strip()
    clean_type = clean_type.replace("const ", "").replace("&", "").strip()
    
    # Check known primitive types
    if clean_type in CPP_TO_IR_TYPE:
        return IRType(CPP_TO_IR_TYPE[clean_type])
    
    # Check for pointer types (treat as object)
    if clean_type.endswith("*"):
        return IRType(IRTypeKind.Object, clean_type)
    
    # Assume it's a class/object type
    return IRType(IRTypeKind.Object, clean_type)
