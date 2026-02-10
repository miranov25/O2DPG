"""
Parent-child indexing support for RDataFrameDSL.

Phase 13.7.B: Offset pattern (residuals).

This module provides:
- ParentChildConfig: dataclass for one registered relationship
- ParentChildRegistry: stores and queries registered relationships
- expand_parent_columns(): generates expansion Define() calls on RDataFrame
- ensure_index_helpers_loaded(): loads C++ IndexHelpers library (idempotent)

Usage:
    registry = ParentChildRegistry()
    registry.register(parent="td.trk", child="res",
                      offset_column="trackInfo.idxFirstResidual")

    # At to_pandas()/draw() time:
    rdf, rename_map = expand_parent_columns(rdf, registry, columns, schema)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger("RDataFrameDSL")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ParentChildConfig:
    """
    One registered parent-child relationship.

    Attributes:
        parent_prefix: Actual branch prefix for parent columns (e.g., "td.trk")
        child_prefix: Actual branch prefix for child columns (e.g., "res")
        offset_column: Branch name with RVec<int> of first-child offsets
                       (e.g., "trackInfo.idxFirstResidual")
    """
    parent_prefix: str
    child_prefix: str
    offset_column: str


# =============================================================================
# Registry
# =============================================================================

class ParentChildRegistry:
    """
    Registry of parent-child relationships. Supports multiple relationships.

    Constraints:
    - Same parent with different children: allowed
    - Same child with different parents: rejected (ambiguous)
    - offset_column must not share prefix with parent or child
    """

    def __init__(self):
        self._relations: List[ParentChildConfig] = []
        self._child_prefixes: Dict[str, int] = {}  # child_prefix → index in _relations

    def register(self, parent: str, child: str, offset_column: str) -> None:
        """
        Register a parent-child relationship.

        Args:
            parent: Actual branch prefix for parent-level columns.
            child: Actual branch prefix for child-level columns.
            offset_column: Branch name containing RVec<int> offsets.

        Raises:
            ValueError: On missing parameters, duplicate child, or prefix overlap.
        """
        if not parent:
            raise ValueError("parent prefix must be non-empty")
        if not child:
            raise ValueError("child prefix must be non-empty")
        if not offset_column:
            raise ValueError("Must provide offset_column")

        # Check duplicate child prefix
        if child in self._child_prefixes:
            existing = self._relations[self._child_prefixes[child]]
            raise ValueError(
                f"Child '{child}' already registered with parent "
                f"'{existing.parent_prefix}'"
            )

        # Check duplicate (parent, child) pair
        for rel in self._relations:
            if rel.parent_prefix == parent and rel.child_prefix == child:
                raise ValueError(
                    f"Relationship '{parent}'→'{child}' already registered"
                )

        # Check offset_column prefix doesn't overlap parent or child
        parent_dot = parent + "."
        child_dot = child + "."
        if offset_column.startswith(parent_dot):
            raise ValueError(
                f"offset_column '{offset_column}' shares prefix with "
                f"parent '{parent}'. Use a column from a different branch."
            )
        if offset_column.startswith(child_dot):
            raise ValueError(
                f"offset_column '{offset_column}' shares prefix with "
                f"child '{child}'. Use a column from a different branch."
            )

        config = ParentChildConfig(
            parent_prefix=parent,
            child_prefix=child,
            offset_column=offset_column,
        )
        self._child_prefixes[child] = len(self._relations)
        self._relations.append(config)
        logger.debug(
            f"[parent-child] Registered: {parent}.* → {child}.* "
            f"via {offset_column}"
        )

    def classify_column(self, column_name: str) -> Optional[Tuple[str, ParentChildConfig]]:
        """
        Classify a column as parent, child, or unregistered.

        Args:
            column_name: The column name to classify.

        Returns:
            ('parent', config) if column matches a parent prefix,
            ('child', config) if column matches a child prefix,
            None if unregistered.
        """
        for rel in self._relations:
            if column_name.startswith(rel.parent_prefix + "."):
                return ('parent', rel)
            if column_name.startswith(rel.child_prefix + "."):
                return ('child', rel)
        return None

    def has_registrations(self) -> bool:
        """True if any relationships are registered."""
        return len(self._relations) > 0

    def get_expansion_info(self, columns: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze requested columns for expansion needs.

        Returns None if no expansion needed.
        Returns dict with expansion details if expansion should trigger.

        Expansion triggers when at least one parent AND at least one child
        column from the same relationship appear in the requested columns.

        Returns:
            None if no expansion, or dict with:
                'config': ParentChildConfig
                'parent_columns': list of parent column names to expand
                'child_columns': list of child column names
                'child_size_column': first child column (for .size())
        """
        if not self._relations:
            return None

        for rel in self._relations:
            parent_dot = rel.parent_prefix + "."
            child_dot = rel.child_prefix + "."

            parent_cols = [c for c in columns if c.startswith(parent_dot)]
            child_cols = [c for c in columns if c.startswith(child_dot)]

            if parent_cols and child_cols:
                return {
                    'config': rel,
                    'parent_columns': parent_cols,
                    'child_columns': child_cols,
                    'child_size_column': child_cols[0],
                }

        return None


# =============================================================================
# C++ Helper Loading
# =============================================================================

_INDEX_HELPERS_LOADED = False


def ensure_index_helpers_loaded() -> bool:
    """
    Load libIndexHelpers.so and include index_helpers.h. Idempotent.

    Must be called before any expansion Define() calls.
    Domain-specific libraries (e.g., libO2TPCCalibration) must be loaded
    BEFORE calling this function.

    The caller is responsible for ensuring that libIndexHelpers.so and
    index_helpers.h are findable by ROOT — either via LD_LIBRARY_PATH,
    ROOT.gSystem.AddDynamicPath(), and ROOT.gInterpreter.AddIncludePath(),
    or by running from the directory containing them.

    Returns:
        True if helpers are loaded (or were already loaded).

    Raises:
        RuntimeError: If the library or header cannot be loaded.
    """
    global _INDEX_HELPERS_LOADED
    if _INDEX_HELPERS_LOADED:
        return True

    try:
        import ROOT
    except ImportError:
        raise RuntimeError(
            "[parent-child] ROOT (PyROOT) is not available. "
            "Install ROOT or activate the environment with ROOT."
        )

    # --- Load shared library ---
    load_result = ROOT.gSystem.Load("libIndexHelpers.so")
    if load_result < 0:
        # Collect diagnostic info
        dyn_path = ROOT.gSystem.GetDynamicPath()
        raise RuntimeError(
            f"[parent-child] Failed to load libIndexHelpers.so "
            f"(gSystem.Load returned {load_result}).\n"
            f"  ROOT dynamic path: {dyn_path}\n"
            f"  Fix: either:\n"
            f"    1. Run from directory containing libIndexHelpers.so, or\n"
            f"    2. Add its directory to LD_LIBRARY_PATH, or\n"
            f"    3. Call ROOT.gSystem.AddDynamicPath('/path/to/dir') "
            f"before register_parent_child()"
        )

    # --- Include header ---
    decl_ok = ROOT.gInterpreter.Declare('#include "index_helpers.h"')
    if not decl_ok:
        inc_path = ROOT.gInterpreter.GetIncludePath()
        raise RuntimeError(
            f"[parent-child] Failed to include index_helpers.h.\n"
            f"  ROOT include path: {inc_path}\n"
            f"  Fix: ROOT.gInterpreter.AddIncludePath('/path/to/dir')"
        )

    # --- Verify symbols exist ---
    try:
        _ = ROOT.RDataFrameDSL.IndexHelpers.ExpandParentIndexFromOffsets
    except AttributeError:
        raise RuntimeError(
            "[parent-child] libIndexHelpers.so loaded but "
            "RDataFrameDSL::IndexHelpers::ExpandParentIndexFromOffsets "
            "not found. Library may be corrupt or from wrong build."
        )

    _INDEX_HELPERS_LOADED = True
    logger.debug("[parent-child] C++ IndexHelpers loaded and verified")
    return True


def reset_helpers_loaded():
    """Reset the loaded state. For testing only."""
    global _INDEX_HELPERS_LOADED
    _INDEX_HELPERS_LOADED = False


# =============================================================================
# Type Resolution
# =============================================================================

# Map from schema type strings to C++ template types
_SCHEMA_TO_CPP_TYPE = {
    "float": "float",
    "Float_t": "float",
    "double": "double",
    "Double_t": "double",
    "int": "int",
    "Int_t": "int",
    "short": "short",
    "Short_t": "short",
    "unsigned char": "unsigned char",
    "UChar_t": "unsigned char",
    "long": "long",
    "Long64_t": "long long",
}


def _resolve_cpp_type(column_name: str, schema: Dict[str, str]) -> str:
    """
    Resolve the C++ element type for a parent column.

    Looks up the column in the schema. If the schema says RVec<float>,
    extracts 'float'. Falls back to 'float' if not found.

    Args:
        column_name: The parent column name.
        schema: DSL schema dict {column_name: type_string}.

    Returns:
        C++ type string suitable for template parameter.
    """
    if column_name not in schema:
        logger.debug(
            f"[parent-child] Column '{column_name}' not in schema, "
            "defaulting to float"
        )
        return "float"

    type_str = schema[column_name]

    # Extract inner type from RVec<T>
    if type_str.startswith("RVec<") and type_str.endswith(">"):
        inner = type_str[5:-1].strip()
    elif type_str.startswith("ROOT::VecOps::RVec<") and type_str.endswith(">"):
        inner = type_str[len("ROOT::VecOps::RVec<"):-1].strip()
    else:
        inner = type_str

    # Map to C++ type
    return _SCHEMA_TO_CPP_TYPE.get(inner, inner)


# =============================================================================
# Expansion Logic
# =============================================================================

def expand_parent_columns(
    rdf,
    registry: ParentChildRegistry,
    requested_columns: List[str],
    schema: Dict[str, str],
) -> Tuple[Any, Dict[str, str]]:
    """
    Inject RDataFrame Define() calls for parent columns that need expansion.

    This is the core of Phase 13.7.B: for each parent column in the
    requested list, generate a Define() call using
    ExpandToChildrenFromOffsets to expand parent values to child level.

    Args:
        rdf: RDataFrame (after apply()).
        registry: ParentChildRegistry with registered relationships.
        requested_columns: Columns requested by the user.
        schema: DSL schema dict for type resolution.

    Returns:
        (modified_rdf, rename_map)
        rename_map: {internal_expanded_name: original_column_name}
        The caller should:
        1. Replace original names with expanded names in AsNumpy()
        2. After flatten, rename back to original names
    """
    info = registry.get_expansion_info(requested_columns)
    if info is None:
        return rdf, {}

    config = info['config']
    parent_cols = info['parent_columns']
    child_size_col = info['child_size_column']
    rename_map = {}

    logger.debug(
        f"[parent-child] Expanding {len(parent_cols)} parent columns "
        f"using offset={config.offset_column}, "
        f"child_size={child_size_col}.size()"
    )

    for pcol in parent_cols:
        cpp_type = _resolve_cpp_type(pcol, schema)
        expanded_name = f"__pc_expanded_{pcol.replace('.', '_')}"

        cpp_expr = (
            f'RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<{cpp_type}>'
            f'({pcol}, {config.offset_column}, '
            f'(int){child_size_col}.size())'
        )

        logger.debug(f"[parent-child] Define('{expanded_name}', '{cpp_expr}')")
        rdf = rdf.Define(expanded_name, cpp_expr)
        rename_map[expanded_name] = pcol

    return rdf, rename_map
