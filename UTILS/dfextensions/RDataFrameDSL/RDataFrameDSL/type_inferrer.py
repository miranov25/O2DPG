"""
Type inference from ROOT tree reflection.

This module provides TypeInferrer which extracts type information from:
- TTree branches (TLeaf, TBranch)
- C++ class reflection (TClass)
- User-provided schema overrides

The inferrer builds a type registry that maps column names to:
- IRType (the data type)
- rank (0=scalar, 1=vector, 2=nested vector)
- is_jagged (whether inner dimensions vary)

Usage:
    # From ROOT tree
    inferrer = TypeInferrer.from_tree(tree)
    dtype, rank, is_jagged = inferrer.get_variable_info("px")
    
    # From schema (for testing without ROOT)
    inferrer = TypeInferrer.from_schema(schema_dict)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Union

from .ir_types import IRType, IRTypeKind, CPP_TO_IR_TYPE, cpp_type_to_ir
from .ir_errors import (
    IRError, IRErrorKind, SourceLocation,
    missing_dictionary_error, unknown_variable_error
)

__all__ = [
    'TypeInferrer',
    'VariableInfo',
    'extract_inner_type',
    'is_vector_type',
    'is_rvec_type',
]


# =============================================================================
# Variable Info
# =============================================================================

@dataclass
class VariableInfo:
    """
    Complete type information for a variable/column.
    
    Attributes:
        name: Variable name
        dtype: IR type
        rank: Dimensionality (0=scalar, 1=vector, 2=nested)
        is_jagged: Whether inner dimensions vary in size
        cpp_type: Original C++ type string
        source: Where the type came from ('tree', 'schema', 'alias')
    """
    name: str
    dtype: IRType
    rank: int = 0
    is_jagged: bool = False
    cpp_type: Optional[str] = None
    source: str = "unknown"
    
    def __repr__(self) -> str:
        jagged_str = ", jagged" if self.is_jagged else ""
        return f"VariableInfo({self.name}: {self.dtype}, rank={self.rank}{jagged_str})"


# =============================================================================
# Helper Functions
# =============================================================================

def is_vector_type(cpp_type: str) -> bool:
    """Check if C++ type is std::vector."""
    clean = cpp_type.strip()
    return (clean.startswith("vector<") or 
            clean.startswith("std::vector<"))


def is_rvec_type(cpp_type: str) -> bool:
    """Check if C++ type is ROOT::VecOps::RVec or ROOT::RVec."""
    clean = cpp_type.strip()
    return ("RVec<" in clean or 
            "VecOps::RVec<" in clean or
            clean.startswith("RVec<"))


def is_collection_type(cpp_type: str) -> bool:
    """Check if C++ type is any collection type."""
    return is_vector_type(cpp_type) or is_rvec_type(cpp_type)


def extract_inner_type(cpp_type: str) -> Tuple[str, int]:
    """
    Extract inner type from container and count nesting depth.
    
    Args:
        cpp_type: C++ type string like "vector<float>" or "RVec<RVec<int>>"
        
    Returns:
        (inner_type, depth) where depth is nesting level
        
    Examples:
        >>> extract_inner_type("vector<float>")
        ('float', 1)
        >>> extract_inner_type("RVec<RVec<double>>")
        ('double', 2)
        >>> extract_inner_type("int")
        ('int', 0)
    """
    clean = cpp_type.strip()
    depth = 0
    
    while is_collection_type(clean):
        depth += 1
        # Find the inner type
        if clean.startswith("std::vector<"):
            clean = clean[12:-1].strip()
        elif clean.startswith("vector<"):
            clean = clean[7:-1].strip()
        elif "VecOps::RVec<" in clean:
            # Handle ROOT::VecOps::RVec<T>
            start = clean.find("VecOps::RVec<") + 13
            # Find matching >
            clean = _extract_template_arg(clean[start:])
        elif clean.startswith("RVec<"):
            clean = clean[5:-1].strip()
        elif "RVec<" in clean:
            start = clean.find("RVec<") + 5
            clean = _extract_template_arg(clean[start:])
        else:
            break
    
    return clean, depth


def _extract_template_arg(s: str) -> str:
    """Extract template argument handling nested <> properly."""
    depth = 0
    end = 0
    for i, c in enumerate(s):
        if c == '<':
            depth += 1
        elif c == '>':
            if depth == 0:
                end = i
                break
            depth -= 1
        end = i
    return s[:end].strip()


def normalize_cpp_type(cpp_type: str) -> str:
    """
    Normalize C++ type for consistent comparison.
    
    - Removes const, &, *
    - Strips whitespace
    - Handles ROOT typedefs
    """
    result = cpp_type.strip()
    result = result.replace("const ", "").replace(" const", "")
    result = result.replace("&", "").replace("*", "")
    result = result.strip()
    return result


# =============================================================================
# Type Inferrer
# =============================================================================

class TypeInferrer:
    """
    Infers types from ROOT tree reflection or schema.
    
    The inferrer builds a registry of variable information that can be
    queried during IR construction.
    
    Priority order for type resolution:
    1. User schema override (highest)
    2. Alias definitions (computed columns)
    3. Tree branch reflection (from ROOT)
    
    Example:
        >>> inferrer = TypeInferrer.from_tree(tree)
        >>> info = inferrer.get_variable_info("px")
        >>> print(info.dtype, info.rank)
    """
    
    def __init__(self):
        """Initialize empty inferrer. Use from_tree() or from_schema()."""
        self._variables: Dict[str, VariableInfo] = {}
        self._aliases: Dict[str, str] = {}  # name -> expression
        self._alias_types: Dict[str, VariableInfo] = {}  # Computed alias types
        self._tree = None
        self._schema: Dict[str, Any] = {}
    
    @classmethod
    def from_tree(cls, tree, schema: Dict = None) -> 'TypeInferrer':
        """
        Create inferrer from ROOT TTree.
        
        Args:
            tree: ROOT.TTree object
            schema: Optional schema dict for overrides
            
        Returns:
            TypeInferrer with types extracted from tree
        """
        inferrer = cls()
        inferrer._tree = tree
        inferrer._schema = schema or {}
        inferrer._scan_tree(tree)
        
        # Apply schema overrides
        if schema:
            inferrer._apply_schema(schema)
        
        return inferrer
    
    @classmethod
    def from_schema(cls, schema: Dict) -> 'TypeInferrer':
        """
        Create inferrer from schema dict (for testing without ROOT).
        
        Schema format:
        {
            "columns": {
                "px": {"dtype": "float", "rank": 0},
                "tracks": {"dtype": "TParticle", "rank": 1, "is_jagged": True},
            },
            "aliases": {
                "pt": "sqrt(px**2 + py**2)"
            }
        }
        
        Args:
            schema: Schema dictionary
            
        Returns:
            TypeInferrer with types from schema
        """
        inferrer = cls()
        inferrer._schema = schema
        inferrer._apply_schema(schema)
        return inferrer
    
    def _scan_tree(self, tree) -> None:
        """
        Scan TTree and extract branch types.
        
        Handles:
        - Leaf branches (primitive types)
        - Object branches (C++ classes)
        - Vector/RVec branches
        - Fixed and variable-length arrays
        """
        # Import ROOT here to allow module to load without ROOT
        try:
            import ROOT
        except ImportError:
            raise IRError(
                IRErrorKind.COMPILE_ERROR,
                "ROOT is required for tree scanning"
            )
        
        branches = tree.GetListOfBranches()
        if not branches:
            return
        
        for i in range(branches.GetEntries()):
            branch = branches.At(i)
            name = branch.GetName()
            
            try:
                info = self._analyze_branch(branch)
                if info:
                    self._variables[name] = info
            except Exception as e:
                # Log but continue - don't fail on one bad branch
                import warnings
                warnings.warn(f"Could not analyze branch '{name}': {e}")
    
    def _analyze_branch(self, branch) -> Optional[VariableInfo]:
        """
        Analyze a single branch and return VariableInfo.
        
        Args:
            branch: ROOT.TBranch object
            
        Returns:
            VariableInfo or None if branch cannot be analyzed
        """
        import ROOT
        
        name = branch.GetName()
        
        # Check for object class first
        class_name = branch.GetClassName()
        if class_name:
            return self._handle_object_branch(name, class_name)
        
        # Try leaf type
        leaf = branch.GetLeaf(name)
        if not leaf:
            # Try first leaf
            leaves = branch.GetListOfLeaves()
            if leaves and leaves.GetEntries() > 0:
                leaf = leaves.At(0)
        
        if leaf:
            return self._handle_leaf_branch(name, leaf)
        
        return None
    
    def _handle_object_branch(self, name: str, class_name: str) -> VariableInfo:
        """
        Handle branch with C++ class type.
        
        Handles vector<T>, RVec<T>, and plain objects.
        """
        import ROOT
        
        # Check if it's a collection type
        if is_collection_type(class_name):
            inner_type, depth = extract_inner_type(class_name)
            
            # Get inner type's IR type
            inner_ir = cpp_type_to_ir(inner_type)
            
            return VariableInfo(
                name=name,
                dtype=inner_ir,
                rank=depth,
                is_jagged=True,  # Collections are potentially jagged
                cpp_type=class_name,
                source="tree"
            )
        
        # Plain object - verify dictionary exists
        tclass = ROOT.TClass.GetClass(class_name)
        if tclass is None:
            raise missing_dictionary_error(class_name)
        
        return VariableInfo(
            name=name,
            dtype=IRType(IRTypeKind.Object, class_name),
            rank=0,
            is_jagged=False,
            cpp_type=class_name,
            source="tree"
        )
    
    def _handle_leaf_branch(self, name: str, leaf) -> VariableInfo:
        """
        Handle primitive leaf branch.
        
        Handles scalars, fixed-length arrays, and variable-length arrays.
        """
        type_name = leaf.GetTypeName()
        ir_type = cpp_type_to_ir(type_name)
        
        # Check for array
        leaf_count = leaf.GetLeafCount()
        if leaf_count:
            # Variable-length array (jagged)
            return VariableInfo(
                name=name,
                dtype=ir_type,
                rank=1,
                is_jagged=True,
                cpp_type=type_name,
                source="tree"
            )
        
        array_len = leaf.GetLen()
        if array_len > 1:
            # Fixed-length array
            return VariableInfo(
                name=name,
                dtype=ir_type,
                rank=1,
                is_jagged=False,
                cpp_type=type_name,
                source="tree"
            )
        
        # Scalar
        return VariableInfo(
            name=name,
            dtype=ir_type,
            rank=0,
            is_jagged=False,
            cpp_type=type_name,
            source="tree"
        )
    
    def _apply_schema(self, schema: Dict) -> None:
        """
        Apply schema to override/add type information.
        
        Schema format:
        {
            "columns": {
                "name": {"dtype": "float", "rank": 0, ...}
            },
            "aliases": {
                "name": "expression"
            }
        }
        """
        # Process columns
        columns = schema.get("columns", {})
        for name, info in columns.items():
            if isinstance(info, dict):
                # Full column specification
                dtype_str = info.get("dtype", "unknown")
                dtype = self._parse_dtype(dtype_str)
                
                var_info = VariableInfo(
                    name=name,
                    dtype=dtype,
                    rank=info.get("rank", 0),
                    is_jagged=info.get("is_jagged", False),
                    cpp_type=info.get("cpp_type", dtype_str),
                    source="schema"
                )
                self._variables[name] = var_info
            elif isinstance(info, str):
                # Just dtype string
                dtype = self._parse_dtype(info)
                self._variables[name] = VariableInfo(
                    name=name,
                    dtype=dtype,
                    rank=0,
                    is_jagged=False,
                    cpp_type=info,
                    source="schema"
                )
        
        # Store aliases for later resolution
        aliases = schema.get("aliases", {})
        for name, expr in aliases.items():
            self._aliases[name] = expr
    
    def _parse_dtype(self, dtype_str: str) -> IRType:
        """Parse dtype string to IRType."""
        # Check direct mapping
        if dtype_str in CPP_TO_IR_TYPE:
            return IRType(CPP_TO_IR_TYPE[dtype_str])
        
        # Check IRTypeKind names
        dtype_lower = dtype_str.lower()
        kind_map = {
            "float32": IRTypeKind.Float32,
            "float": IRTypeKind.Float32,
            "float64": IRTypeKind.Float64,
            "double": IRTypeKind.Float64,
            "int32": IRTypeKind.Int32,
            "int": IRTypeKind.Int32,
            "int64": IRTypeKind.Int64,
            "long": IRTypeKind.Int64,
            "uint32": IRTypeKind.UInt32,
            "uint64": IRTypeKind.UInt64,
            "bool": IRTypeKind.Bool,
            "unknown": IRTypeKind.Unknown,
        }
        
        if dtype_lower in kind_map:
            return IRType(kind_map[dtype_lower])
        
        # Assume it's an object type
        return IRType(IRTypeKind.Object, dtype_str)
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_variable_info(self, name: str, 
                          namespace: str = None) -> VariableInfo:
        """
        Get complete type information for a variable.
        
        Args:
            name: Variable name
            namespace: Optional namespace (subframe name)
            
        Returns:
            VariableInfo for the variable
            
        Raises:
            IRError: If variable not found
        """
        # Check with namespace
        if namespace:
            full_name = f"{namespace}.{name}"
            if full_name in self._variables:
                return self._variables[full_name]
        
        # Check direct name
        if name in self._variables:
            return self._variables[name]
        
        # Check aliases
        if name in self._alias_types:
            return self._alias_types[name]
        
        # Not found
        similar = self._find_similar_names(name)
        raise unknown_variable_error(name, similar_names=similar)
    
    def has_variable(self, name: str, namespace: str = None) -> bool:
        """Check if variable exists."""
        if namespace:
            full_name = f"{namespace}.{name}"
            if full_name in self._variables:
                return True
        return name in self._variables or name in self._alias_types
    
    def get_type(self, name: str, namespace: str = None) -> IRType:
        """Get just the IRType for a variable."""
        return self.get_variable_info(name, namespace).dtype
    
    def get_rank(self, name: str, namespace: str = None) -> int:
        """Get rank for a variable."""
        return self.get_variable_info(name, namespace).rank
    
    def is_jagged(self, name: str, namespace: str = None) -> bool:
        """Check if variable is jagged."""
        return self.get_variable_info(name, namespace).is_jagged
    
    def register_alias(self, name: str, dtype: IRType, 
                       rank: int = 0, is_jagged: bool = False) -> None:
        """
        Register a computed alias with its inferred type.
        
        Called by IRBuilder after analyzing alias expression.
        """
        self._alias_types[name] = VariableInfo(
            name=name,
            dtype=dtype,
            rank=rank,
            is_jagged=is_jagged,
            source="alias"
        )
    
    def get_all_variables(self) -> Dict[str, VariableInfo]:
        """Get all registered variables."""
        result = dict(self._variables)
        result.update(self._alias_types)
        return result
    
    def get_column_names(self) -> List[str]:
        """Get list of all column names."""
        names = set(self._variables.keys())
        names.update(self._alias_types.keys())
        return sorted(names)
    
    def _find_similar_names(self, target: str, max_results: int = 3) -> List[str]:
        """
        Find similar variable names for error suggestions.
        
        Uses simple substring matching (per Phase 1 review guidance).
        """
        target_lower = target.lower()
        all_names = self.get_column_names()
        
        matches = []
        for name in all_names:
            name_lower = name.lower()
            # Check substring match (both directions)
            if target_lower in name_lower or name_lower in target_lower:
                matches.append(name)
            # Check prefix match
            elif name_lower.startswith(target_lower[:3]) if len(target_lower) >= 3 else False:
                matches.append(name)
        
        return matches[:max_results]
    
    # =========================================================================
    # Debug/Introspection
    # =========================================================================
    
    def describe(self) -> str:
        """Return human-readable description of inferred types."""
        lines = ["TypeInferrer contents:"]
        lines.append(f"  Variables: {len(self._variables)}")
        lines.append(f"  Aliases: {len(self._alias_types)}")
        
        if self._variables:
            lines.append("\n  Columns:")
            for name, info in sorted(self._variables.items()):
                lines.append(f"    {info}")
        
        if self._alias_types:
            lines.append("\n  Computed Aliases:")
            for name, info in sorted(self._alias_types.items()):
                lines.append(f"    {info}")
        
        return "\n".join(lines)
    
    def to_schema(self) -> Dict:
        """Export current state as schema dict."""
        columns = {}
        for name, info in self._variables.items():
            columns[name] = {
                "dtype": str(info.dtype),
                "rank": info.rank,
                "is_jagged": info.is_jagged,
                "cpp_type": info.cpp_type,
            }
        
        aliases = {}
        for name, expr in self._aliases.items():
            aliases[name] = expr
        
        return {
            "columns": columns,
            "aliases": aliases,
        }
    
    def to_simple_schema(self) -> Dict[str, str]:
        """
        Convert internal schema to simple {name: cpp_type} format.
        
        Used by DSLCompiler.from_tree() to get schema in expected format.
        
        Returns:
            Dict mapping column names to C++ type strings
            
        Example:
            >>> inferrer = TypeInferrer.from_tree(tree)
            >>> schema = inferrer.to_simple_schema()
            >>> print(schema)
            {'px': 'double', 'py': 'double', 'tracks': 'RVec<TLorentzVector>'}
        """
        result = {}
        for name, info in self._variables.items():
            if info.rank == 1:
                # Vector type
                if info.cpp_type and ('RVec' in info.cpp_type or 'vector' in info.cpp_type):
                    result[name] = info.cpp_type
                else:
                    # Wrap element type in RVec
                    elem_type = info.cpp_type or info.dtype.to_cpp()
                    result[name] = f"RVec<{elem_type}>"
            else:
                # Scalar type
                result[name] = info.cpp_type or info.dtype.to_cpp()
        
        return result
