import sys
import os; sys.path.insert(1, os.environ.get("O2DPG", "") + "/UTILS/dfextensions")
import pandas as pd
import numpy as np
import json
import uproot
import copy
import warnings
import glob
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Set, Optional, Union
try:
    import ROOT  # type: ignore
except ImportError as e:
    print(f"[AliasDataFrame] WARNING: ROOT import failed: {e}")
    ROOT = None
import matplotlib.pyplot as plt
import networkx as nx
import re
import ast

# Phase 7.4: Custom exceptions
try:
    from exceptions import (
        AliasDataFrameError,
        BranchNotFoundError,
        ChainValidationError,
        CircularAliasError
    )
except ImportError:
    # Define inline if module not found
    class AliasDataFrameError(Exception):
        pass
    class BranchNotFoundError(AliasDataFrameError):
        def __init__(self, missing, available=None, message=None):
            self.missing = missing
            self.available = available
            super().__init__(message or f"Branches not found: {sorted(missing)}")
    class ChainValidationError(AliasDataFrameError):
        pass
    class CircularAliasError(AliasDataFrameError):
        pass

# Numba acceleration (optional)
try:
    from _numba_accelerators import (
        NUMBA_AVAILABLE, NUMBA_MIN_ROWS,
        numba_scatter, numba_compute_join_indices, get_numba_info,
        linearize_multi_column_keys_pair
    )
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_MIN_ROWS = 10000
    numba_scatter = None
    numba_compute_join_indices = None
    get_numba_info = lambda: {'available': False, 'version': None}
    linearize_multi_column_keys_pair = None

# PyArrow acceleration (optional) - Phase 9
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pc = None

# Arrow compute mapper (optional) - Phase 9a/9c
try:
    from _arrow_compute import ArrowComputeMapper, evaluate_expression_arrow
    ARROW_COMPUTE_AVAILABLE = PYARROW_AVAILABLE
except ImportError:
    ArrowComputeMapper = None
    evaluate_expression_arrow = None
    ARROW_COMPUTE_AVAILABLE = False

# =============================================================================
# SECTION 0: Schema & Metadata Constants
# =============================================================================
#
# The _schema dict is the SINGLE SOURCE OF TRUTH for all AliasDataFrame metadata.
# See the canonical structure comment in AliasDataFrame.__init__().
#
# Key concepts:
# - columns: physical column dtypes AND computed aliases (expr + dtype)
# - compression: state machine tracking compressed columns
# - subframes: registered child DataFrames with join keys
# - __meta__: schema versioning, timestamps, user-defined IDs
#
# =============================================================================

# =============================================================================
# Verbosity Bitmask Constants for describe_structure()
# =============================================================================
VERBOSITY_BASIC        = 0x01  # rows/columns/memory
VERBOSITY_DTYPES       = 0x02  # columns grouped by dtype
VERBOSITY_ALIASES      = 0x04  # list aliases (short)
VERBOSITY_ALIASES_FULL = 0x08  # full alias definitions
VERBOSITY_COMPRESSION  = 0x10  # compression summary
VERBOSITY_COMP_FULL    = 0x20  # full compression info
VERBOSITY_SUBFRAMES    = 0x40  # list subframes
VERBOSITY_METADATA     = 0x80  # raw metadata dump

# Presets
VERBOSE_MINIMAL = VERBOSITY_BASIC
VERBOSE_DEFAULT = (VERBOSITY_BASIC | VERBOSITY_DTYPES | VERBOSITY_ALIASES |
                   VERBOSITY_COMPRESSION | VERBOSITY_SUBFRAMES)
VERBOSE_FULL = 0xFF  # All flags


class SubframeRegistry:
    """
    Registry to manage subframes (nested AliasDataFrame instances).
    """
    def __init__(self):
        self.subframes = {}  # name → {'frame': adf, 'index': index_columns}

    def add_subframe(self, name, alias_df, index_columns, pre_index=False):
        # Convert string to list (defensive - prevents "track_tf_uid" → ['t','r','a','c','k',...])
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        if pre_index and not alias_df.df.index.names == index_columns:
            alias_df.df.set_index(index_columns, inplace=True)
        self.subframes[name] = {'frame': alias_df, 'index': index_columns}

    def get(self, name):
        return self.subframes.get(name, {}).get('frame', None)

    def get_entry(self, name):
        return self.subframes.get(name, None)

    def items(self):
        return self.subframes.items()

    def has_subframe(self, name):
        """Check if a subframe with given name is registered."""
        return name in self.subframes


def convert_expr_to_root(expr):
    class RootTransformer(ast.NodeTransformer):
        FUNC_MAP = {
            "arctan2": "atan2",
            "mod": "fmod",
            "sqrt": "sqrt",
            "log": "log",
            "log10": "log10",
            "exp": "exp",
            "abs": "abs",
            "power": "pow",
            "maximum": "TMath::Max",
            "minimum": "TMath::Min"
        }

        def visit_Call(self, node):
            def get_func_name(n):
                if isinstance(n, ast.Attribute):
                    return n.attr
                elif isinstance(n, ast.Name):
                    return n.id
                return ""

            func_name = get_func_name(node.func)

            # Use NumpyRootMapper for function name translation
            root_func = NumpyRootMapper.get_root_name(func_name)
            # Fallback to old FUNC_MAP for backward compatibility
            if root_func == func_name:
                root_func = self.FUNC_MAP.get(func_name, func_name)

            node.args = [self.visit(arg) for arg in node.args]
            node.func = ast.Name(id=root_func, ctx=ast.Load())
            return node

    try:
        expr_clean = re.sub(r"\bnp\\.", "", expr)
        tree = ast.parse(expr_clean, mode='eval')
        tree = RootTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return expr
# Add BEFORE class AliasDataFrame:

class NumpyRootMapper:
    """Maps NumPy function names to ROOT C++ equivalents (bidirectional)"""

    # Maps function names to (numpy_attr, root_name)
    MAPPING = {
        # Hyperbolic functions
        'sinh': ('sinh', 'sinh'),
        'cosh': ('cosh', 'cosh'),
        'tanh': ('tanh', 'tanh'),
        'arcsinh': ('arcsinh', 'asinh'),
        'arccosh': ('arccosh', 'acosh'),
        'arctanh': ('arctanh', 'atanh'),
        'asinh': ('arcsinh', 'asinh'),
        'acosh': ('arccosh', 'acosh'),
        'atanh': ('arctanh', 'atanh'),

        # Trigonometric
        'sin': ('sin', 'sin'),
        'cos': ('cos', 'cos'),
        'tan': ('tan', 'tan'),
        'arcsin': ('arcsin', 'asin'),
        'arccos': ('arccos', 'acos'),
        'arctan': ('arctan', 'atan'),
        'arctan2': ('arctan2', 'atan2'),
        'asin': ('arcsin', 'asin'),
        'acos': ('arccos', 'acos'),
        'atan': ('arctan', 'atan'),
        'atan2': ('arctan2', 'atan2'),  # ← NEW: ROOT name maps to numpy

        # Exponential/log
        'exp': ('exp', 'exp'),
        'log': ('log', 'log'),
        'log10': ('log10', 'log10'),
        'sqrt': ('sqrt', 'sqrt'),
        'pow': ('power', 'pow'),
        'power': ('power', 'pow'),

        # Rounding
        'round': ('round', 'round'),
        'floor': ('floor', 'floor'),
        'ceil': ('ceil', 'ceil'),
        'abs': ('abs', 'abs'),
    }

    @classmethod
    def get_numpy_functions_for_eval(cls):
        """Get dict of function_name → numpy_function for evaluation

        Includes both Python names (arctan2) and ROOT names (atan2)
        for bidirectional compatibility when reading ROOT files.
        """
        funcs = {}
        for name, (np_attr, _) in cls.MAPPING.items():
            if hasattr(np, np_attr):
                funcs[name] = getattr(np, np_attr)
        return funcs

    @classmethod
    def get_root_name(cls, name):
        """Get ROOT C++ equivalent name for a function"""
        entry = cls.MAPPING.get(name)
        return entry[1] if entry else name

class CompressionState:
    """
    Compression state constants for column compression lifecycle.

    States:
        COMPRESSED: Physical compressed column exists, original is alias
        DECOMPRESSED: Decompressed column exists physically, schema retained
        SCHEMA_ONLY: Metadata defined but no data compressed yet
    """
    COMPRESSED = "compressed"
    DECOMPRESSED = "decompressed"
    SCHEMA_ONLY = "schema_only"


# =============================================================================
# Phase 4b: Schema Serialization Constants and Helpers
# =============================================================================

# Dedicated metadata key to avoid collisions
SCHEMA_METADATA_KEY = "__alias_dataframe_schema__"
SCHEMA_VERSION = 1
SCHEMA_VERSION_V2 = 2  # New v2 format with groups, metadata, smart formatting


def _repair_index_columns(index_cols, sf_name=None):
    """
    Repair corrupted index_columns that were stored as individual characters.
    
    Bug: If index_columns="track_tf_uid" was passed as string instead of list,
    iteration yields ['t','r','a','c','k','_','t','f','_','u','i','d'].
    
    Parameters
    ----------
    index_cols : list
        The index columns list (possibly corrupted)
    sf_name : str, optional
        Subframe name for warning message
        
    Returns
    -------
    list
        Repaired index columns
    """
    if (isinstance(index_cols, list) and 
        len(index_cols) > 1 and 
        all(isinstance(c, str) and len(c) == 1 for c in index_cols)):
        # Repair: rejoin characters back to original column name
        repaired_name = "".join(index_cols)
        sf_msg = f"Subframe '{sf_name}': " if sf_name else ""
        warnings.warn(
            f"{sf_msg}Repaired corrupted index {index_cols[:5]}... → ['{repaired_name}']"
        )
        return [repaired_name]
    return index_cols


def _serialize_schema(schema):
    """
    Serialize _schema dict to JSON-safe format.
    
    Handles:
    - numpy dtype objects → string representation
    - Sets → lists
    - Ensures all values are JSON-serializable
    - Preserves __meta__ (schema_version, created_at, schema_id)
    
    Parameters
    ----------
    schema : dict
        The _schema dict with columns/compression/subframes
        
    Returns
    -------
    dict
        JSON-serializable version of schema
    """
    # Preserve __meta__ if present, otherwise create default
    meta = schema.get("__meta__", {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_id": None
    })
    
    result = {
        "__meta__": meta,
        "schema_version": meta.get("schema_version", SCHEMA_VERSION),  # Also at top level for backward compat
        "columns": {},
        "compression": schema.get("compression", {}),
        "subframes": schema.get("subframes", {}),
    }
    
    # Serialize columns section - convert dtypes to strings
    for name, spec in schema.get("columns", {}).items():
        serialized_spec = {}
        for key, value in spec.items():
            if key == "dtype":
                # Convert numpy dtype to string
                if hasattr(value, 'name'):
                    serialized_spec[key] = value.name
                elif hasattr(value, '__name__'):
                    serialized_spec[key] = value.__name__
                else:
                    serialized_spec[key] = str(value)
            else:
                serialized_spec[key] = value
        result["columns"][name] = serialized_spec
    
    return result


def _deserialize_schema(serialized):
    """
    Deserialize JSON schema back to _schema format.
    
    Handles:
    - String dtype names → numpy dtype types
    - Schema version migration (future-proofing)
    - Restores __meta__ (schema_version, created_at, schema_id)
    - Repairs corrupted subframe indices (string iterated as chars)
    
    Parameters
    ----------
    serialized : dict
        JSON-parsed schema dict
        
    Returns
    -------
    dict
        Restored _schema dict with proper types
    """
    # Get version from __meta__ or top-level for backward compat
    meta = serialized.get("__meta__", {})
    version = meta.get("schema_version", serialized.get("schema_version", 1))
    
    # Future: Add migration logic here
    if version > SCHEMA_VERSION:
        warnings.warn(
            f"Schema version {version} is newer than supported version {SCHEMA_VERSION}. "
            f"Some features may not work correctly."
        )
    
    result = {
        "__meta__": {
            "schema_version": version,
            "created_at": meta.get("created_at"),  # May be None for old schemas
            "schema_id": meta.get("schema_id")  # May be None
        },
        "columns": {},
        "compression": serialized.get("compression", {
            "__meta__": {
                "schema_version": 1,
                "state_machine": "CompressionState.v1"
            }
        }),
        "subframes": {},
    }
    
    # Ensure compression has __meta__
    if "__meta__" not in result["compression"]:
        result["compression"]["__meta__"] = {
            "schema_version": 1,
            "state_machine": "CompressionState.v1"
        }
    
    # Deserialize subframes with repair for corrupted indices
    # Bug: "track_tf_uid" passed as string → stored as ['t','r','a','c','k','_',...]
    for sf_name, sf_spec in serialized.get("subframes", {}).items():
        repaired_spec = dict(sf_spec)
        index_cols = sf_spec.get("index", [])
        repaired_spec["index"] = _repair_index_columns(index_cols, sf_name)
        result["subframes"][sf_name] = repaired_spec
    
    # Deserialize columns section - convert dtype strings to numpy types
    for name, spec in serialized.get("columns", {}).items():
        deserialized_spec = {}
        for key, value in spec.items():
            if key == "dtype" and isinstance(value, str):
                # Convert string back to numpy dtype type
                try:
                    deserialized_spec[key] = np.dtype(value).type
                except TypeError:
                    # Fallback: try getattr on np
                    deserialized_spec[key] = getattr(np, value, None)
            else:
                deserialized_spec[key] = value
        result["columns"][name] = deserialized_spec
    
    return result


# =============================================================================
# Schema Export v2: Enhanced JSON export with groups, metadata, smart formatting
# =============================================================================

def _format_json_smart(data, indent=2, max_line_length=100):
    """
    Format JSON with smart line splitting.
    
    Short entries stay on one line, long entries are expanded.
    
    Parameters
    ----------
    data : dict
        Data to format as JSON
    indent : int
        Indentation level (spaces)
    max_line_length : int
        Maximum line length before splitting
        
    Returns
    -------
    str
        Formatted JSON string
    """
    def format_dict_smart(d, level=0):
        """Recursively format a dict with smart line decisions."""
        if not d:
            return '{}'
        
        base_indent = ' ' * (level * indent)
        item_indent = ' ' * ((level + 1) * indent)
        
        lines = ['{']
        items = list(d.items())
        
        for i, (key, value) in enumerate(items):
            key_str = json.dumps(key)
            comma = ',' if i < len(items) - 1 else ''
            
            # Check if this is a section that should always be expanded
            if key in ('columns', '__meta__', 'groups', 'subframes', 'compression', 'schemas'):
                if isinstance(value, dict) and value:
                    nested = format_dict_smart(value, level + 1)
                    lines.append(f'{item_indent}{key_str}: {nested}{comma}')
                else:
                    lines.append(f'{item_indent}{key_str}: {json.dumps(value)}{comma}')
            elif isinstance(value, dict):
                # Try compact first for column entries
                compact = json.dumps(value, separators=(', ', ': '))
                full_line = f'{item_indent}{key_str}: {compact}{comma}'
                
                if len(full_line) <= max_line_length:
                    lines.append(full_line)
                else:
                    # Need to expand
                    nested = format_dict_smart(value, level + 1)
                    lines.append(f'{item_indent}{key_str}: {nested}{comma}')
            elif isinstance(value, list):
                compact = json.dumps(value, separators=(', ', ': '))
                full_line = f'{item_indent}{key_str}: {compact}{comma}'
                
                if len(full_line) <= max_line_length:
                    lines.append(full_line)
                else:
                    # Expand list
                    lines.append(f'{item_indent}{key_str}: {json.dumps(value, indent=indent)}{comma}')
            else:
                lines.append(f'{item_indent}{key_str}: {json.dumps(value)}{comma}')
        
        lines.append(f'{base_indent}}}')
        return '\n'.join(lines)
    
    return format_dict_smart(data)


def _order_columns_by_groups(columns, groups, within_group_sort="schema"):
    """
    Order columns: grouped first (in group order), then ungrouped.
    
    Parameters
    ----------
    columns : dict
        Column specifications
    groups : dict
        {group_name: [column_names], ...}
    within_group_sort : str
        "schema" - preserve order as defined in groups dict (default)
        "alphabetic" - sort alphabetically within each group
    
    Returns
    -------
    dict
        Ordered columns dict
    """
    if not groups:
        return columns
    
    ordered = {}
    seen = set()
    
    # First: columns in group order
    for group_name, group_cols in groups.items():
        cols_to_add = list(group_cols)
        if within_group_sort == "alphabetic":
            cols_to_add = sorted(cols_to_add)
        
        for col in cols_to_add:
            if col in columns:
                ordered[col] = columns[col]
                seen.add(col)
    
    # Then: ungrouped columns (preserve original order from schema)
    for col, spec in columns.items():
        if col not in seen:
            ordered[col] = spec
    
    return ordered


def _dtype_to_str(dtype):
    """
    Convert a dtype to its string representation.
    
    Handles:
    - numpy dtype instances: np.dtype('float32') → 'float32'
    - numpy type classes: np.float32 → 'float32'
    - strings: 'float32' → 'float32'
    - None: None → None
    
    Parameters
    ----------
    dtype : various
        A dtype in various formats
        
    Returns
    -------
    str or None
        String representation of dtype
    """
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype
    # np.dtype instances have .name
    if hasattr(dtype, 'name'):
        return dtype.name
    # numpy type classes like np.float32
    if hasattr(dtype, '__name__'):
        return dtype.__name__
    # Fallback - try to convert via np.dtype
    try:
        return np.dtype(dtype).name
    except (TypeError, AttributeError):
        return str(dtype)


def _export_column_spec_v2(col_name, col_info, df=None):
    """
    Export a single column specification in v2 format.
    
    Always returns object format: {"dtype": "..."} 
    Excludes "expr": null for physical columns.
    Preserves all metadata (unit, axisLabel, etc.)
    
    Parameters
    ----------
    col_name : str
        Column name
    col_info : dict
        Column specification from schema
    df : pd.DataFrame, optional
        DataFrame to get dtype from if not in schema
        
    Returns
    -------
    dict
        Clean column specification
    """
    result = {}
    
    # Get dtype
    dtype = col_info.get('dtype')
    if dtype is not None:
        if hasattr(dtype, 'name'):
            result['dtype'] = dtype.name
        elif hasattr(dtype, '__name__'):
            result['dtype'] = dtype.__name__
        else:
            result['dtype'] = str(dtype)
    elif df is not None and col_name in df.columns:
        result['dtype'] = str(df[col_name].dtype)
    
    # Add expr only if it's not None (aliases only)
    expr = col_info.get('expr')
    if expr is not None:
        result['expr'] = expr
    
    # Copy all other metadata (unit, axisLabel, description, etc.)
    # Skip internal keys
    skip_keys = {'dtype', 'expr', 'constant'}
    for key, value in col_info.items():
        if key not in skip_keys and key not in result:
            result[key] = value
    
    # Add constant only if True
    if col_info.get('constant', False):
        result['constant'] = True
    
    return result


def _export_subframe_schema_v2(subframe_entry, include_precision_stats=False, include_state=True):
    """
    Export a subframe's full schema recursively.
    
    Parameters
    ----------
    subframe_entry : dict
        Entry from SubframeRegistry: {'frame': adf, 'index': [...]}
    include_precision_stats : bool
        Whether to include precision statistics in compression section
    include_state : bool
        Whether to include runtime state fields (state, original_removed)
        
    Returns
    -------
    dict
        Subframe schema with index and columns
    """
    result = {}
    
    # Index columns (ensure list format)
    index_cols = subframe_entry.get('index', [])
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    result['index'] = index_cols
    
    # Get the subframe AliasDataFrame
    sf_adf = subframe_entry.get('frame')
    if sf_adf is None:
        return result
    
    # Export columns
    columns = {}
    sf_schema = sf_adf._schema if hasattr(sf_adf, '_schema') else {}
    sf_df = sf_adf.df if hasattr(sf_adf, 'df') else None
    compression_info = sf_schema.get('compression', {})
    
    # Build set of compressed column names to exclude in definition mode
    compressed_col_names = set()
    compression_targets = set()
    if not include_state:
        for orig_col, comp_info in compression_info.items():
            if orig_col == '__meta__':
                continue
            compressed_col_names.add(comp_info.get('compressed_col', f'{orig_col}_c'))
            compression_targets.add(orig_col)
    
    # Physical columns from DataFrame
    if sf_df is not None:
        for col in sf_df.columns:
            # In definition mode, skip compressed storage columns
            if not include_state and col in compressed_col_names:
                continue
            col_info = sf_schema.get('columns', {}).get(col, {})
            columns[col] = _export_column_spec_v2(col, col_info, sf_df)
    
    # Aliases from schema
    for col, col_info in sf_schema.get('columns', {}).items():
        if col not in columns:
            # In definition mode, compression targets as physical columns
            if not include_state and col in compression_targets:
                decompressed_dtype = compression_info.get(col, {}).get('decompressed_dtype')
                physical_info = col_info.copy()
                physical_info.pop('expr', None)
                if decompressed_dtype:
                    physical_info['dtype'] = decompressed_dtype
                columns[col] = _export_column_spec_v2(col, physical_info)
            else:
                columns[col] = _export_column_spec_v2(col, col_info)
    
    if columns:
        result['columns'] = columns
    
    # Groups (if present)
    if sf_schema.get('groups'):
        result['groups'] = sf_schema['groups']
    
    # Compression - always include when present (required for data interpretation)
    if sf_schema.get('compression'):
        comp = {}
        for name, info in sf_schema['compression'].items():
            if name == '__meta__':
                comp['__meta__'] = copy.deepcopy(info)
                continue
            
            entry = {}
            # Required fields (always included)
            for field in ['compressed_col', 'compress_expr', 'decompress_expr']:
                if field in info:
                    entry[field] = info[field]
            
            # State fields (optional)
            if include_state:
                for field in ['state', 'original_removed']:
                    if field in info:
                        entry[field] = info[field]
            
            # Convert dtypes to strings using helper
            for dtype_field in ['compressed_dtype', 'decompressed_dtype']:
                if dtype_field in info and info[dtype_field] is not None:
                    entry[dtype_field] = _dtype_to_str(info[dtype_field])
            
            # Optional precision stats
            if include_precision_stats and 'precision' in info:
                entry['precision'] = copy.deepcopy(info['precision'])
            
            comp[name] = entry
        
        if comp:
            result['compression'] = comp
    
    return result


class AliasDataFrame:
    """
    AliasDataFrame allows for defining and evaluating lazy-evaluated column aliases
    on top of a pandas DataFrame, including nested subframes with hierarchical indexing.
    
    Phase 4: Uses unified _schema dict as single source of truth.
    """
    
    def __init__(self, df, schema_id=None, use_numba=None, use_arrow=None):
        """
        Initialize AliasDataFrame with unified schema structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            The underlying pandas DataFrame
        schema_id : str, optional
            User-defined identifier for this schema (e.g., "miranov_lxplus_TPC_calib_v3").
            Useful for parameter scans, test studies, and provenance tracking.
        use_numba : bool, optional
            Enable/disable Numba acceleration for subframe joins.
            If None (default), auto-detect: use Numba if available.
            Set to False to force pure NumPy/Pandas operations.
        
        The _schema dict is the single source of truth for:
        - __meta__: schema version, timestamps, user-defined ID
        - columns: physical column dtypes and aliases (expr + dtype + constant)
        - compression: compression formulas per column
        - subframes: registered subframes with index info
        
        _schema canonical structure:
        {
            "__meta__": {
                "schema_version": 1,
                "created_at": "2025-01-15T10:30:00+00:00",  # ISO timestamp
                "schema_id": None  # User-defined, optional
            },
            "columns": {
                "x": {"dtype": "float32", "expr": None},           # Physical column
                "pt": {"dtype": "float32", "expr": "sqrt(px**2+py**2)", "constant": False}  # Alias
            },
            "compression": {
                "__meta__": {"schema_version": 1, "state_machine": "CompressionState.v1"},
                "dy": {
                    "compress": "round(asinh(dy)*40)",
                    "decompress": "sinh(dy_c/40.)",
                    "compressed_dtype": "int16",
                    "decompressed_dtype": "float16",
                    "compressed_col": "dy_c",
                    "state": "compressed"  # One of: compressed, decompressed, schema_only
                }
            },
            "subframes": {
                "track": {"index": ["track_index"]}
            }
        }
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"AliasDataFrame must be initialized with a pandas.DataFrame. "
                f"Received type: {type(df)}"
            )
        self.df = df
        
        # Unified schema (Phase 4)
        self._schema = {
            "__meta__": {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "schema_id": schema_id  # User-defined, optional
            },
            "columns": {},      # {name: {"dtype": ..., "expr": ..., "constant": ...}}
            "compression": {
                "__meta__": {
                    "schema_version": 1,
                    "state_machine": "CompressionState.v1"
                }
            },
            "subframes": {},    # {name: {"index": ...}}
        }
        
        # Subframe registry (keeps actual ADF objects)
        self._subframes = SubframeRegistry()
        
        # Phase B: Auto-alias tracking
        self._auto_aliases = {}  # {alias_name: subframe_name}
        self.index_columns = {}  # {subframe_name: [index_cols]}
        
        # Fill configuration for subframe joins (Phase 1: scalars only)
        self._global_fill_config = {
            'fill_missing': None,    # Fill value for missing keys (None = NaN)
            'fill_nan': None,        # Fill value for NaN in subframe data
            'fill_inf': None,        # Fill value for Inf in subframe data
            'fill_invalid': None,    # Shortcut for both NaN and Inf
            'warn_missing_keys': True,
            'warn_threshold': 0.01,  # Warn if > 1% missing
            'fill_mode': 'safe',     # 'safe' or 'direct'
        }
        self._subframe_fill_config = {}  # {subframe_name: {...}}
        
        # For aggregated warnings during materialization
        # NOTE: _missing_key_stats is not thread-safe. If parallel 
        # materialization is added, use thread-local storage.
        self._missing_key_stats = {}  # {subframe_name: {'count': n, 'total': N, ...}}
        
        # Phase 4: Join index cache for subframe lookups
        # Caches precomputed indices to avoid repeated pd.merge() operations
        self._join_index_cache = {}  # {sf_name: {indices, missing_mask, n_rows, subframe_id}}
        self._join_cache_hits = 0
        self._join_cache_misses = 0
        
        # Phase 8: Numba acceleration configuration
        # Auto-detect if not specified: use Numba when available
        if use_numba is None:
            self._use_numba = NUMBA_AVAILABLE
        else:
            self._use_numba = use_numba and NUMBA_AVAILABLE
        
        # Phase 9: PyArrow acceleration configuration
        # Auto-detect if not specified: use PyArrow when available
        if use_arrow is None:
            self._use_arrow = PYARROW_AVAILABLE
        else:
            self._use_arrow = use_arrow and PYARROW_AVAILABLE
        
        # Phase 6.8: Draw integration properties
        # These control default behavior for draw methods
        self.draw_lazy = False              # Default: require explicit materialization
        self.draw_keep_materialized = True  # Default: keep after single draw
        self.draw_clear_after = True        # Default: clear after batch
        
        # Phase 7.1: Lazy branch loading support
        # _lazy_reader: LazyTreeReader instance for on-demand branch loading
        # _chain: Runtime config for file chain (separate from _schema)
        self._lazy_reader = None  # Set by read_tree_lazy()
        self._chain = None        # Set by read_tree_lazy() or read_chain()
        self._df_access_warned = False  # Track if we've warned about .df access
        
        # Phase 7.5a: Lazy subframe support
        # _subframe_readers: LazyTreeReader for each lazy subframe
        # _subframe_loaded: Whether each lazy subframe has been loaded
        # _subframe_lazy_config: Config for lazy subframes (index_columns, columns, etc.)
        self._subframe_readers = {}   # {name: LazyTreeReader}
        self._subframe_loaded = {}    # {name: bool}
        self._subframe_lazy_config = {}  # {name: {file, tree, index_columns, ...}}

    # =========================================================================
    # SECTION 0b: Proxy Pattern - DataFrame Delegation
    # =========================================================================
    #
    # Enable direct DataFrame-like access: adf['column'], len(adf), adf.head()
    # instead of adf.df['column'], len(adf.df), adf.df.head()
    #
    # =========================================================================

    def __getitem__(self, key):
        """
        Enable adf['column'] and adf[['col1', 'col2']] syntax.
        
        In lazy mode, auto-loads branches if available in TTree.
        
        Examples
        --------
        >>> adf['x']           # Single column (auto-loads if lazy)
        >>> adf[['x', 'y']]    # Multiple columns
        >>> adf['x'].mean()    # Chain with pandas methods
        """
        # Phase 7.1: Auto-load branches in lazy mode
        if self._lazy_reader is not None:
            if isinstance(key, str):
                # Single column access
                if key not in self.df.columns and key in self._lazy_reader.available_branches:
                    self.ensure_branches([key])
            elif isinstance(key, list):
                # Multiple column access
                to_load = [k for k in key 
                          if k not in self.df.columns 
                          and k in self._lazy_reader.available_branches]
                if to_load:
                    self.ensure_branches(to_load)
        
        return self.df[key]
    
    def __setitem__(self, key, value):
        """
        Block direct assignment to prevent confusion between aliases and columns.
        
        Use add_alias() for computed columns or adf.df['column'] = value for direct assignment.
        """
        raise TypeError(
            "Direct assignment via adf['column'] = value is not supported.\n"
            "Use one of:\n"
            "  adf.add_alias('name', 'expression')  # For computed columns\n"
            "  adf.df['column'] = value             # For direct DataFrame modification"
        )
    
    def __len__(self):
        """Enable len(adf) to return number of rows."""
        return len(self.df)
    
    def __iter__(self):
        """Enable iteration over column names."""
        return iter(self.df)
    
    def __contains__(self, key):
        """Enable 'column' in adf syntax."""
        return key in self.df.columns
    
    @property
    def columns(self):
        """DataFrame columns (read-only access)."""
        return self.df.columns
    
    @property
    def index(self):
        """DataFrame index (read-only access)."""
        return self.df.index
    
    @property
    def shape(self):
        """DataFrame shape as (rows, columns) tuple."""
        return self.df.shape
    
    @property
    def dtypes(self):
        """DataFrame column dtypes."""
        return self.df.dtypes
    
    @property
    def loc(self):
        """Label-based indexer for DataFrame rows/columns."""
        return self.df.loc
    
    @property
    def iloc(self):
        """Integer-based indexer for DataFrame rows/columns."""
        return self.df.iloc

    # =========================================================================
    # SECTION 1: Core DataFrame Operations & Schema Properties
    # =========================================================================
    #
    # Core properties and methods for accessing/modifying the DataFrame and schema.
    # Includes backward-compatible property accessors for aliases, dtypes, etc.
    #
    # =========================================================================
    
    @property
    def schema_id(self):
        """Get the user-defined schema identifier."""
        return self._schema.get("__meta__", {}).get("schema_id")
    
    @schema_id.setter
    def schema_id(self, value):
        """Set the user-defined schema identifier."""
        if "__meta__" not in self._schema:
            self._schema["__meta__"] = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        self._schema["__meta__"]["schema_id"] = value
    
    def set_schema_id(self, schema_id):
        """
        Set the user-defined schema identifier.
        
        Parameters
        ----------
        schema_id : str
            Identifier for this schema (e.g., "miranov_lxplus_TPC_calib_v3")
        
        Returns
        -------
        self : AliasDataFrame
            For method chaining
        
        Example
        -------
        >>> adf.set_schema_id("TPC_residuals_run3_v2")
        """
        self.schema_id = schema_id
        return self
    
    @property
    def numba_info(self):
        """
        Get information about Numba acceleration status.
        
        Returns
        -------
        dict
            Contains:
            - available: bool - whether Numba is installed
            - enabled: bool - whether this ADF instance uses Numba
            - version: str or None - Numba version if available
            - min_rows: int - minimum rows to use Numba (JIT overhead threshold)
        
        Example
        -------
        >>> adf.numba_info
        {'available': True, 'enabled': True, 'version': '0.57.0', 'min_rows': 10000}
        """
        info = get_numba_info()
        info['enabled'] = self._use_numba
        info['min_rows'] = NUMBA_MIN_ROWS
        return info
    
    @property
    def arrow_info(self):
        """
        Get information about PyArrow acceleration status.
        
        Returns
        -------
        dict
            Contains:
            - available: bool - whether PyArrow is installed
            - enabled: bool - whether this ADF instance uses PyArrow
            - version: str or None - PyArrow version if available
            - min_rows: int - minimum rows to use PyArrow (overhead threshold)
            - compute_available: bool - whether ArrowComputeMapper is available
        
        Example
        -------
        >>> adf.arrow_info
        {'available': True, 'enabled': True, 'version': '14.0.2', 'min_rows': 10000, 'compute_available': True}
        """
        info = {
            'available': PYARROW_AVAILABLE,
            'enabled': self._use_arrow,
            'version': pa.__version__ if PYARROW_AVAILABLE else None,
            'min_rows': NUMBA_MIN_ROWS,  # Reuse same threshold
            'compute_available': ARROW_COMPUTE_AVAILABLE
        }
        return info
    
    # =========================================================================
    # Phase 4: Backward Compatibility Properties
    # =========================================================================
    
    @property
    def aliases(self):
        """
        Backward compatible: returns {name: expr} for all aliases.
        Read-only view over _schema["columns"].
        """
        return {k: v["expr"] for k, v in self._schema["columns"].items() if "expr" in v}

    @aliases.setter
    def aliases(self, value):
        """
        Phase 4b: Raises AttributeError - use add_alias() or update_schema() instead.
        
        For bulk loading from serialized data, use _restore_aliases_from_serialized().
        """
        raise AttributeError(
            "Direct assignment to 'aliases' is no longer supported. "
            "Use 'add_alias()' or 'update_schema({\"columns\": {...}})' instead."
        )
    
    def _restore_aliases_from_dict(self, aliases_dict):
        """
        Internal method to restore aliases from serialized data.
        Used by read_tree() and load() for deserialization.
        
        Parameters
        ----------
        aliases_dict : dict
            {alias_name: expression_string, ...}
        """
        for name, expr in aliases_dict.items():
            if name not in self._schema["columns"]:
                self._schema["columns"][name] = {}
            self._schema["columns"][name]["expr"] = expr

    @property
    def alias_dtypes(self):
        """
        Backward compatible: returns {name: dtype} for aliases with dtype.
        Read-only view over _schema["columns"].
        """
        return {k: v.get("dtype") for k, v in self._schema["columns"].items() 
                if "expr" in v and "dtype" in v}

    @alias_dtypes.setter
    def alias_dtypes(self, value):
        """
        Phase 4b: Raises AttributeError - use add_alias() with dtype parameter instead.
        """
        raise AttributeError(
            "Direct assignment to 'alias_dtypes' is no longer supported. "
            "Use 'add_alias(name, expr, dtype=...)' or 'update_schema()' instead."
        )
    
    def _restore_alias_dtypes_from_dict(self, dtypes_dict):
        """
        Internal method to restore alias dtypes from serialized data.
        Used by read_tree() and load() for deserialization.
        
        Parameters
        ----------
        dtypes_dict : dict
            {alias_name: dtype_type, ...}
        """
        for name, dtype in dtypes_dict.items():
            if name not in self._schema["columns"]:
                self._schema["columns"][name] = {}
            self._schema["columns"][name]["dtype"] = dtype

    @property
    def constant_aliases(self):
        """
        Backward compatible: returns set of constant alias names.
        Phase 4b: derives only from _schema.
        """
        return {k for k, v in self._schema["columns"].items() 
                if v.get("constant", False)}

    @constant_aliases.setter
    def constant_aliases(self, value):
        """
        Phase 4b: Raises AttributeError - use add_alias() instead.
        """
        raise AttributeError(
            "Direct assignment to 'constant_aliases' is no longer supported. "
            "Use 'add_alias(name, expr, is_constant=True)' instead."
        )
    
    def _restore_constant_aliases(self, constants_list):
        """
        Internal method to restore constant alias flags from serialized data.
        Used by read_tree() and load() for deserialization.
        
        Parameters
        ----------
        constants_list : list
            List of alias names that are constants
        """
        for name in constants_list:
            if name in self._schema["columns"]:
                self._schema["columns"][name]["constant"] = True

    @property
    def compression_info(self):
        """
        Backward compatible: returns compression dict.
        Direct reference to _schema["compression"].
        """
        return self._schema["compression"]

    @compression_info.setter
    def compression_info(self, value):
        """
        Phase 4b: Raises AttributeError - use update_schema() instead.
        """
        raise AttributeError(
            "Direct assignment to 'compression_info' is no longer supported. "
            "Use 'update_schema({\"compression\": {...}})' instead."
        )
    
    def _restore_compression_info(self, compression_dict):
        """
        Internal method to restore compression info from serialized data.
        Used by read_tree() and load() for deserialization.
        
        Parameters
        ----------
        compression_dict : dict
            Compression metadata dict including __meta__
        """
        self._schema["compression"] = compression_dict
        
        # Ensure __meta__ exists
        if "__meta__" not in self._schema["compression"]:
            self._schema["compression"]["__meta__"] = {
                "schema_version": 1,
                "state_machine": "CompressionState.v1"
            }

    # =========================================================================
    # Phase 4: New Schema API
    # =========================================================================

    @property
    def schema(self):
        """
        Read-only copy of current schema.
        
        Returns:
            dict with keys: "columns", "compression", "subframes"
        """
        return copy.deepcopy(self._schema)
    
    def _restore_schema(self, serialized_schema):
        """
        Internal method to restore full schema from serialized/deserialized data.
        Used by read_tree() and load() for deserialization.
        
        Parameters
        ----------
        serialized_schema : dict
            Deserialized schema dict (already processed by _deserialize_schema)
        """
        # Restore columns
        self._schema["columns"] = serialized_schema.get("columns", {})
        
        # Restore compression
        self._schema["compression"] = serialized_schema.get("compression", {
            "__meta__": {
                "schema_version": 1,
                "state_machine": "CompressionState.v1"
            }
        })
        
        # Ensure __meta__ exists
        if "__meta__" not in self._schema["compression"]:
            self._schema["compression"]["__meta__"] = {
                "schema_version": 1,
                "state_machine": "CompressionState.v1"
            }
        
        # Restore subframes metadata (not actual subframe objects)
        self._schema["subframes"] = serialized_schema.get("subframes", {})

    def update_schema(self, update, validate=True, apply=True, errors="raise"):
        """
        Partial update of schema. Only specified items are changed.
        
        Args:
            update: dict with any of {"columns": {...}, "compression": {...}, "subframes": {...}}
            validate: if True, validate before applying
            apply: if True, apply dtypes to df immediately
            errors: "raise" | "warn" | "ignore" for dtype conversion errors
        """
        if validate:
            self._validate_schema_update(update)
        
        # Update columns section
        if "columns" in update:
            for name, spec in update["columns"].items():
                if name not in self._schema["columns"]:
                    self._schema["columns"][name] = {}
                self._schema["columns"][name].update(spec)
                
                # Apply dtype immediately if requested (for physical columns only)
                if apply and "dtype" in spec:
                    if "expr" not in self._schema["columns"][name]:
                        # Physical column - cast immediately
                        if name in self.df.columns:
                            try:
                                self.df[name] = self.df[name].astype(spec["dtype"])
                            except Exception as e:
                                if errors == "raise":
                                    raise
                                elif errors == "warn":
                                    warnings.warn(f"Failed to cast '{name}' to {spec['dtype']}: {e}")
        
        # Update compression section
        if "compression" in update:
            for name, spec in update["compression"].items():
                if name == "__meta__":
                    self._schema["compression"]["__meta__"].update(spec)
                else:
                    self._schema["compression"][name] = spec
        
        # Update subframes section
        if "subframes" in update:
            for name, spec in update["subframes"].items():
                self._schema["subframes"][name] = spec

    def _validate_schema_update(self, update):
        """
        Validate schema update before applying.
        """
        if "columns" in update:
            for name, spec in update["columns"].items():
                # Check dtype is valid
                if "dtype" in spec:
                    try:
                        np.dtype(spec["dtype"])
                    except TypeError as e:
                        raise ValueError(f"Invalid dtype for column '{name}': {e}")
                
                # Check expr is string or None (None for physical columns)
                if "expr" in spec and spec["expr"] is not None and not isinstance(spec["expr"], str):
                    raise ValueError(f"Expression for '{name}' must be a string or None")
                
                # Check subframe references exist (lightweight check)
                if "expr" in spec and spec["expr"] is not None and "." in spec["expr"]:
                    subframe_refs = re.findall(r'([A-Z][A-Za-z0-9_]*)\.', spec["expr"])
                    for sf_name in subframe_refs:
                        if sf_name not in self._schema["subframes"] and \
                           not self._subframes.has_subframe(sf_name):
                            raise ValueError(
                                f"Column '{name}' references undefined subframe '{sf_name}'"
                            )
        
        if "subframes" in update:
            for name, spec in update["subframes"].items():
                if "index" not in spec:
                    raise ValueError(f"Subframe '{name}' must specify 'index'")

    def apply_aliases(self, aliases_spec):
        """
        Register multiple aliases from a spec dict.
        
        Args:
            aliases_spec: {alias_name: {"expr": str, "dtype": type, "constant": bool}, ...}
        """
        for name, spec in aliases_spec.items():
            expr = spec.get("expr")
            if expr is None:
                raise ValueError(f"Alias '{name}' missing 'expr'")
            dtype = spec.get("dtype")
            constant = spec.get("constant", False)
            self.add_alias(name, expr, dtype=dtype, is_constant=constant)

    def apply_dtypes(self, dtype_spec, errors="raise"):
        """
        Bulk dtype conversion for physical columns.
        
        Args:
            dtype_spec: {col_name: dtype, ...}
            errors: "raise" | "warn" | "ignore"
        """
        for col, dtype in dtype_spec.items():
            if col not in self.df.columns:
                if errors == "raise":
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                elif errors == "warn":
                    warnings.warn(f"Column '{col}' not found, skipping")
                continue
            
            try:
                self.df[col] = self.df[col].astype(dtype)
                # Update schema
                if col not in self._schema["columns"]:
                    self._schema["columns"][col] = {}
                self._schema["columns"][col]["dtype"] = dtype
            except Exception as e:
                if errors == "raise":
                    raise
                elif errors == "warn":
                    warnings.warn(f"Failed to cast '{col}' to {dtype}: {e}")

    def __getattr__(self, item: str):
        """
        Attribute access with DataFrame delegation.
        
        Order of resolution:
        1. DataFrame columns → returns column Series
        2. Defined aliases → materializes and returns column
        3. Registered subframes → returns SubframeProxy
        4. DataFrame methods → delegates to self.df (enables adf.head(), adf.describe(), etc.)
        5. Otherwise → raises AttributeError
        """
        # Avoid infinite recursion during unpickling or when df doesn't exist yet
        if item in ('df', '_schema', '_subframes', 'aliases'):
            raise AttributeError(item)
        
        # 1. Check DataFrame columns
        if item in self.df.columns:
            return self.df[item]
        
        # 2. Check defined aliases
        if item in self.aliases:
            self.materialize_alias(item)
            return self.df[item]
        
        # 3. Check registered subframes
        sf = self._subframes.get(item)
        if sf is not None:
            return sf
        
        # 4. Delegate to DataFrame methods (head, tail, describe, groupby, etc.)
        if hasattr(self.df, item):
            return getattr(self.df, item)
        
        # 5. Not found
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    # =========================================================================
    # SECTION 3: Subframe Registry & Joins
    # =========================================================================
    #
    # Subframes are nested AliasDataFrame instances that can be joined to the
    # parent frame using index columns. Enables hierarchical data access like:
    #   adf.df["track.pt"] or adf.track.df["pt"]
    #
    # Key methods:
    # - register_subframe(): Add a child DataFrame with join keys
    # - auto_alias_subframe(): Create convenience aliases for subframe columns
    # - get_subframe(): Retrieve registered subframe
    #
    # =========================================================================

    def register_subframe(self, name, adf, index_columns, pre_index=False):
        """
        Register a subframe (nested AliasDataFrame) for join operations.
        
        Parameters
        ----------
        name : str
            Name to reference this subframe (e.g., "calibration")
        adf : AliasDataFrame
            The subframe to register
        index_columns : str or list of str
            Column(s) to use for joining. String is auto-converted to single-element list.
        pre_index : bool, default=False
            If True, set index on subframe DataFrame
        
        Phase 4: Also writes to _schema["subframes"] for metadata persistence.
        Phase 4b: Auto-populates subframe schema if empty (fixes v2 export bug).
        """
        # Convert string to list (defensive - prevents iteration over characters)
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        # Auto-populate subframe's _schema["columns"] if empty (v2 fix)
        # This happens when subframes are loaded from ROOT without embedded schema
        if not adf._schema.get("columns") and hasattr(adf, 'df') and adf.df is not None:
            adf._schema["columns"] = {
                col: {"dtype": str(adf.df[col].dtype)}
                for col in adf.df.columns
            }
        
        # Add to runtime registry
        self._subframes.add_subframe(name, adf, index_columns, pre_index=pre_index)
        
        # Also write to schema for persistence
        self._schema["subframes"][name] = {
            "index": index_columns,           # Legacy key (backward compat)
            "index_columns": index_columns,   # New canonical key
        }

    def get_subframe(self, name):
        """
        Get a subframe by name, triggering lazy load if needed.
        
        MODIFIED FOR 7.5a: Triggers lazy loading if subframe is lazy.
        """
        # Check if it's a lazy subframe that needs loading
        if name in self._subframe_readers and not self._subframe_loaded.get(name, False):
            self._load_lazy_subframe(name)
        return self._subframes.get(name)

    def register_subframe_lazy(
        self,
        name: str,
        file: str,
        tree_name: str = None,
        index_columns: List[str] = None,
        columns: List[str] = None,
        alignment: str = 'by_key',
        join_type: str = 'left'
    ) -> None:
        """
        Register a lazy-loaded subframe from a single ROOT file.
        
        Data is NOT loaded immediately. Loading is triggered automatically
        when an alias referencing this subframe is materialized or drawn.
        
        Parameters
        ----------
        name : str
            Subframe name (used in alias expressions as 'Name.column')
        file : str
            File path with optional tree name ('calib.root:tree' or 'calib.root')
        tree_name : str, optional
            Tree name if not specified in file string
        index_columns : List[str]
            Columns for join key (must exist in both main DataFrame and subframe)
        columns : List[str], optional
            Specific columns to load from subframe. None = all columns.
            Index columns are always loaded regardless of this parameter.
        alignment : str, default 'by_key'
            Alignment hint: 'by_key', 'N:1', '1:1'
            Currently informational only (no behavioral change).
        join_type : str, default 'left'
            Join type: 'left', 'inner', 'outer'
            
        Raises
        ------
        ValueError
            If name is already registered (eager or lazy)
        ValueError
            If tree_name not provided and not in file string
        FileNotFoundError
            If file does not exist
        KeyError
            If index_columns don't exist in subframe file
            
        Examples
        --------
        >>> adf.register_subframe_lazy(
        ...     'Calib',
        ...     'calibration.root:tree',
        ...     index_columns=['run', 'sector']
        ... )
        >>> adf.add_alias('corrected', 'signal * Calib.gain')
        >>> adf.draw('corrected')  # Calibration loads here
        
        Notes
        -----
        - After loading, the subframe behaves identically to an eager subframe.
        - Use `ensure_subframe(name)` to explicitly trigger loading.
        """
        from LazyTreeReader import LazyTreeReader
        
        # Validate name not already registered
        if self._subframes.has_subframe(name):
            raise ValueError(f"Subframe '{name}' already registered (eager)")
        if name in self._subframe_readers:
            raise ValueError(f"Subframe '{name}' already registered (lazy)")
        
        # Parse file specification
        if ':' in file:
            file_path, tree = file.rsplit(':', 1)
        else:
            file_path = file
            tree = tree_name
        
        if tree is None:
            raise ValueError(
                f"Tree name required. Use 'file.root:tree' or tree_name parameter."
            )
        
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Subframe file not found: {file_path}")
        
        # Validate index_columns provided
        if not index_columns:
            raise ValueError(f"index_columns required for subframe '{name}'")
        
        # Convert string to list
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        # Validate alignment parameter
        valid_alignments = {'by_key', 'N:1', '1:1'}
        if alignment not in valid_alignments:
            raise ValueError(
                f"Invalid alignment '{alignment}'. Must be one of: {sorted(valid_alignments)}"
            )
        
        # Validate join_type parameter
        valid_join_types = {'left', 'inner', 'outer'}
        if join_type not in valid_join_types:
            raise ValueError(
                f"Invalid join_type '{join_type}'. Must be one of: {sorted(valid_join_types)}"
            )
        
        # Create reader (opens file for metadata only)
        reader = LazyTreeReader(file_path, tree)
        
        # Validate index columns exist in subframe
        missing_idx = set(index_columns) - reader.available_branches
        if missing_idx:
            reader.close()
            raise KeyError(
                f"Subframe '{name}' missing index column(s) in file: {sorted(missing_idx)}. "
                f"Available: {sorted(reader.available_branches)}"
            )
        
        # Validate requested columns exist (if specified)
        if columns:
            # Always include index columns
            columns_to_check = set(columns) | set(index_columns)
            missing_cols = columns_to_check - reader.available_branches
            if missing_cols:
                reader.close()
                raise KeyError(
                    f"Subframe '{name}' missing column(s): {sorted(missing_cols)}. "
                    f"Available: {sorted(reader.available_branches)}"
                )
        
        # Store reader and config
        self._subframe_readers[name] = reader
        self._subframe_loaded[name] = False
        self._subframe_lazy_config[name] = {
            'type': 'file',  # Single-file subframe
            'file': file_path,
            'tree': tree,
            'index_columns': list(index_columns),
            'columns': list(columns) if columns else None,
            'alignment': alignment,
            'join_type': join_type,
        }
        
        # Register in schema (without data) - allows alias validation to work
        if 'subframes' not in self._schema:
            self._schema['subframes'] = {}
        
        self._schema['subframes'][name] = {
            'index': list(index_columns),         # Legacy key (C++ macro compat)
            'index_columns': list(index_columns), # Canonical key
            'join_type': join_type,
            'lazy': True,
            'alignment': alignment,
        }
    
    def register_subframe_chain(
        self,
        name: str,
        files: Union[str, List[str]],
        tree_name: str = None,
        index_columns: List[str] = None,
        columns: List[str] = None,
        alignment: str = 'by_key',
        join_type: str = 'left',
        validate_branches: str = 'first',
        max_open_files: int = 8
    ) -> None:
        """
        Register a lazy-loaded subframe chain from multiple ROOT files.
        
        Data is NOT loaded immediately. Loading is triggered automatically
        when an alias referencing this subframe is materialized or drawn.
        
        Parameters
        ----------
        name : str
            Subframe name (used in alias expressions as 'Name.column')
        files : str or List[str]
            Glob pattern ('calib_*.root:tree') or list of file paths
        tree_name : str, optional
            Tree name if not specified in files pattern
        index_columns : List[str]
            Columns for join key (must exist in ALL subframe files,
            regardless of validation mode)
        columns : List[str], optional
            Specific columns to load. None = all columns.
            Index columns are always loaded regardless of this parameter.
        alignment : str, default 'by_key'
            Alignment hint: 'by_key', 'N:1', '1:1'
            Currently informational only.
        join_type : str, default 'left'
            Join type: 'left', 'inner', 'outer'
        validate_branches : str, default 'first'
            Branch validation mode:
            - 'first': Use first file as reference, warn on differences
            - 'strict': Error if any file differs
            - 'intersection': Only branches in ALL files
            - 'union': All branches, NaN for missing
        max_open_files : int, default 8
            Maximum open file handles (LRU cache size)
            
        Raises
        ------
        ValueError
            If name already registered or invalid parameters
        FileNotFoundError
            If no files match the pattern
        KeyError
            If index_columns don't exist in subframe files
            
        Examples
        --------
        >>> # Calibration chain spanning multiple runs
        >>> adf.register_subframe_chain(
        ...     'Calib',
        ...     'calib_run*.root:tree',
        ...     index_columns=['run_number']
        ... )
        
        >>> # With explicit file list
        >>> adf.register_subframe_chain(
        ...     'Calib',
        ...     ['calib_2024.root:tree', 'calib_2025.root:tree'],
        ...     index_columns=['run_number']
        ... )
        
        Notes
        -----
        - Uses LazyChainReader from Phase 7.4 internally
        - After loading, behaves identically to an eager subframe
        - File handles managed via LRU cache
        """
        from LazyChainReader import LazyChainReader
        
        # Validate name not already registered
        if self._subframes.has_subframe(name):
            raise ValueError(f"Subframe '{name}' already registered (eager)")
        if name in self._subframe_readers:
            raise ValueError(f"Subframe '{name}' already registered (lazy)")
        
        # Validate index_columns provided
        if not index_columns:
            raise ValueError(f"index_columns required for subframe '{name}'")
        
        # Convert string to list
        if isinstance(index_columns, str):
            index_columns = [index_columns]
        
        # Validate alignment parameter
        valid_alignments = {'by_key', 'N:1', '1:1'}
        if alignment not in valid_alignments:
            raise ValueError(
                f"Invalid alignment '{alignment}'. Must be one of: {sorted(valid_alignments)}"
            )
        
        # Validate join_type parameter
        valid_join_types = {'left', 'inner', 'outer'}
        if join_type not in valid_join_types:
            raise ValueError(
                f"Invalid join_type '{join_type}'. Must be one of: {sorted(valid_join_types)}"
            )
        
        # Validate validate_branches parameter
        valid_validations = {'first', 'strict', 'intersection', 'union'}
        if validate_branches not in valid_validations:
            raise ValueError(
                f"Invalid validate_branches '{validate_branches}'. "
                f"Must be one of: {sorted(valid_validations)}"
            )
        
        # Parse file specifications (reuse existing method)
        file_specs = self._parse_chain_files(files, tree_name)
        
        if not file_specs:
            raise FileNotFoundError(f"No files found matching: {files}")
        
        # Create chain reader
        chain_reader = LazyChainReader(
            files=file_specs,
            validation=validate_branches,
            max_open_files=max_open_files,
            add_file_index=False  # Subframes don't need __file_idx__
        )
        
        # Validate index columns exist in subframe
        missing_idx = set(index_columns) - chain_reader.available_branches
        if missing_idx:
            chain_reader.close()
            raise KeyError(
                f"Subframe '{name}' missing index column(s) in files: {sorted(missing_idx)}. "
                f"Available: {sorted(chain_reader.available_branches)}"
            )
        
        # Validate requested columns exist (if specified)
        if columns:
            columns_to_check = set(columns) | set(index_columns)
            missing_cols = columns_to_check - chain_reader.available_branches
            if missing_cols:
                chain_reader.close()
                raise KeyError(
                    f"Subframe '{name}' missing column(s): {sorted(missing_cols)}. "
                    f"Available: {sorted(chain_reader.available_branches)}"
                )
        
        # Store reader and config
        self._subframe_readers[name] = chain_reader
        self._subframe_loaded[name] = False
        self._subframe_lazy_config[name] = {
            'type': 'chain',  # Chain subframe
            'files': file_specs,
            'index_columns': list(index_columns),
            'columns': list(columns) if columns else None,
            'alignment': alignment,
            'join_type': join_type,
            'validate_branches': validate_branches,
        }
        
        # Register in schema (without data)
        if 'subframes' not in self._schema:
            self._schema['subframes'] = {}
        
        self._schema['subframes'][name] = {
            'index': list(index_columns),         # Legacy key (C++ macro compat)
            'index_columns': list(index_columns), # Canonical key
            'join_type': join_type,
            'lazy': True,
            'chain': True,  # Mark as chain subframe
            'alignment': alignment,
            'file_count': len(file_specs),
        }
    
    def ensure_subframe(self, name: str) -> None:
        """
        Ensure subframe data is loaded.
        
        For lazy subframes, triggers loading from file.
        For eager subframes, no-op.
        
        Parameters
        ----------
        name : str
            Subframe name
            
        Raises
        ------
        KeyError
            If subframe not registered
            
        Examples
        --------
        >>> adf.register_subframe_lazy('Calib', 'calib.root:tree', ...)
        >>> adf.ensure_subframe('Calib')  # Force load now
        >>> print('Calib' in adf.loaded_subframes)  # True
        """
        # Check if it's an eager subframe (already loaded)
        if self._subframes.has_subframe(name):
            return  # Already loaded
        
        # Check if it's a lazy subframe
        if name not in self._subframe_readers:
            raise KeyError(f"Subframe '{name}' not registered")
        
        # Check if already loaded
        if self._subframe_loaded.get(name, False):
            return
        
        # Load the subframe
        self._load_lazy_subframe(name)
    
    def _load_lazy_subframe(self, name: str) -> None:
        """
        Internal: Load a lazy subframe from file.
        
        After loading, the subframe is indistinguishable from an eager subframe.
        This is the UNIFICATION PRINCIPLE from architecture review.
        """
        reader = self._subframe_readers[name]
        config = self._subframe_lazy_config[name]
        
        # Determine columns to load
        if config['columns'] is not None:
            # User specified columns - ensure index columns included
            columns_to_load = list(set(config['columns']) | set(config['index_columns']))
        else:
            # Load all columns
            columns_to_load = list(reader.available_branches)
        
        # Load data from file
        df = reader.load_branches(columns_to_load)
        
        # Create AliasDataFrame wrapper for subframe (UNIFICATION)
        # This reuses ALL existing subframe join machinery
        subframe_adf = AliasDataFrame(df)
        
        # Register as eager subframe
        self._subframes.add_subframe(
            name, 
            subframe_adf, 
            config['index_columns']
        )
        
        # Update schema to mark as loaded
        if name in self._schema.get('subframes', {}):
            self._schema['subframes'][name]['lazy'] = False
        
        # Mark as loaded
        self._subframe_loaded[name] = True
        
        # Validate index columns exist in main DataFrame
        # CRITICAL: Check lazy reader FIRST to avoid triggering main load
        if self._lazy_reader is not None:
            # Lazy main: check available branches only (no I/O)
            available = self._lazy_reader.available_branches | self._lazy_reader.loaded_branches
            missing_in_main = set(config['index_columns']) - available
            if missing_in_main:
                warnings.warn(
                    f"Subframe '{name}' index column(s) {sorted(missing_in_main)} "
                    f"not found in main DataFrame available branches."
                )
        elif len(self._df) > 0:
            # Eager main: safe to check columns directly
            missing_in_main = set(config['index_columns']) - set(self._df.columns)
            if missing_in_main:
                warnings.warn(
                    f"Subframe '{name}' index column(s) {sorted(missing_in_main)} "
                    f"not found in main DataFrame columns."
                )
    
    def _get_subframes_for_aliases(self, alias_names: List[str]) -> Set[str]:
        """
        Identify which subframes are referenced by the given aliases.
        
        Recursively resolves alias dependencies to find all subframe references.
        
        Parameters
        ----------
        alias_names : List[str]
            Alias names to analyze
            
        Returns
        -------
        Set[str]
            Names of subframes referenced by these aliases
        """
        subframes_needed = set()
        
        # Get all registered subframe names (eager + lazy)
        all_subframes = set(self._subframes.subframes.keys()) | set(self._subframe_readers.keys())
        
        if not all_subframes:
            return subframes_needed
        
        # Pattern to match subframe references: SubframeName.column
        subframe_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b')
        
        def find_subframes_in_expr(expr: str):
            """Find subframe references in an expression."""
            for match in subframe_pattern.finditer(expr):
                potential_subframe = match.group(1)
                if potential_subframe in all_subframes:
                    subframes_needed.add(potential_subframe)
        
        # Recursively process aliases
        processed = set()
        to_process = list(alias_names) if alias_names else []
        
        while to_process:
            alias_name = to_process.pop()
            if alias_name in processed:
                continue
            processed.add(alias_name)
            
            # Get alias expression
            expr = self.aliases.get(alias_name)
            
            if expr:
                # Find subframe references
                find_subframes_in_expr(expr)
                
                # Find dependent aliases to process
                if alias_name in self.aliases:
                    deps = self._get_alias_dependencies(alias_name, expr)
                    for dep_type, dep_name in deps:
                        if dep_type == 'alias' and dep_name not in processed:
                            to_process.append(dep_name)
        
        return subframes_needed

    @property
    def lazy_subframes(self) -> List[str]:
        """
        Names of registered lazy subframes (loaded or not).
        
        Returns
        -------
        List[str]
            Names of lazy subframes
        """
        return list(self._subframe_readers.keys())
    
    @property
    def loaded_subframes(self) -> List[str]:
        """
        Names of subframes that are currently loaded (eager or lazy-loaded).
        
        Returns
        -------
        List[str]
            Names of loaded subframes
        """
        return list(self._subframes.subframes.keys())

    @property
    def chain_subframes(self) -> List[str]:
        """
        Names of registered subframe chains (multi-file).
        
        Returns
        -------
        List[str]
            Names of chain subframes
        """
        return [
            name for name, config in self._subframe_lazy_config.items()
            if config.get('type') == 'chain'
        ]

    # =========================================================================
    # Fill Configuration Methods
    # =========================================================================
    #
    # Configure how missing keys and invalid values are handled during
    # subframe joins. See set_global_fill() and set_subframe_fill() for details.
    #
    # =========================================================================

    def set_global_fill(
        self,
        fill_missing=None,
        fill_nan=None,
        fill_inf=None,
        fill_invalid=None,
        warn_missing_keys=None,
        warn_threshold=None,
        fill_mode=None,
        # Reserved for Phase 2 - accepted but ignored
        patterns_missing=None,
        patterns_nan=None,
        patterns_inf=None,
        patterns_invalid=None,
    ):
        """
        Set global default fill behavior for all subframes.
        
        Subframe-specific settings (via set_subframe_fill) override these.
        
        Parameters
        ----------
        fill_missing : float, optional
            Fill value for missing keys (row not in subframe).
            If None, missing keys produce NaN.
        
        fill_nan : float, optional
            Fill value for NaN values in subframe data.
            Applied in 'safe' mode only.
        
        fill_inf : float, optional
            Fill value for ±Inf values in subframe data.
            Applied in 'safe' mode only.
        
        fill_invalid : float, optional
            Shortcut: sets both fill_nan and fill_inf.
            Individual fill_nan/fill_inf take precedence if specified.
        
        warn_missing_keys : bool, optional
            Whether to warn about missing keys. Default True.
        
        warn_threshold : float, optional
            Only warn if missing fraction > threshold. Default 0.01 (1%).
        
        fill_mode : str, optional
            Performance mode:
            - 'safe': Separate checks for missing/NaN/Inf (default)
            - 'direct': Fill at join time, no post-processing (fastest)
            - 'fast': Reserved for Phase 2 (raises NotImplementedError)
        
        patterns_missing, patterns_nan, patterns_inf, patterns_invalid : dict, optional
            Reserved for Phase 2. Currently ignored.
        
        Examples
        --------
        >>> # Silence all warnings, fill missing with 0
        >>> adf.set_global_fill(fill_missing=0.0, warn_missing_keys=False)
        
        >>> # Maximum speed for calibration
        >>> adf.set_global_fill(fill_missing=0.0, fill_mode='direct')
        """
        # Validate fill_mode
        if fill_mode is not None:
            if fill_mode == 'fast':
                raise NotImplementedError("fill_mode='fast' will be available in Phase 2")
            if fill_mode not in ('safe', 'direct'):
                raise ValueError(f"fill_mode must be 'safe' or 'direct', got '{fill_mode}'")
            self._global_fill_config['fill_mode'] = fill_mode
        
        # Validate numeric types
        for name, value in [('fill_missing', fill_missing), ('fill_nan', fill_nan),
                            ('fill_inf', fill_inf), ('fill_invalid', fill_invalid)]:
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        
        # Apply values
        if fill_missing is not None:
            self._global_fill_config['fill_missing'] = fill_missing
        if fill_nan is not None:
            self._global_fill_config['fill_nan'] = fill_nan
        if fill_inf is not None:
            self._global_fill_config['fill_inf'] = fill_inf
        if fill_invalid is not None:
            self._global_fill_config['fill_invalid'] = fill_invalid
        if warn_missing_keys is not None:
            self._global_fill_config['warn_missing_keys'] = warn_missing_keys
        if warn_threshold is not None:
            self._global_fill_config['warn_threshold'] = warn_threshold
        
        # Pattern parameters are reserved for Phase 2 - silently ignore

    def set_subframe_fill(
        self,
        subframe_name,
        fill_missing=None,
        fill_nan=None,
        fill_inf=None,
        fill_invalid=None,
        warn_missing_keys=None,
        warn_threshold=None,
        fill_mode=None,
        # Reserved for Phase 2 - accepted but ignored
        patterns_missing=None,
        patterns_nan=None,
        patterns_inf=None,
        patterns_invalid=None,
    ):
        """
        Configure fill behavior for a specific subframe.
        
        Settings here override global defaults from set_global_fill().
        
        Parameters
        ----------
        subframe_name : str
            Name of registered subframe. Must already be registered.
        
        fill_missing : float, optional
            Fill value for missing keys (row not in subframe).
        
        fill_nan : float, optional
            Fill value for NaN values from subframe data.
        
        fill_inf : float, optional
            Fill value for ±Inf values from subframe data.
        
        fill_invalid : float, optional
            Shortcut: sets both fill_nan and fill_inf.
        
        warn_missing_keys : bool, optional
            Whether to warn about missing keys.
        
        warn_threshold : float, optional
            Only warn if missing fraction > threshold.
        
        fill_mode : str, optional
            'safe' or 'direct'. See set_global_fill() for details.
        
        patterns_missing, patterns_nan, patterns_inf, patterns_invalid : dict, optional
            Reserved for Phase 2. Currently ignored.
        
        Raises
        ------
        ValueError
            If subframe_name is not registered or fill_mode is invalid.
        TypeError
            If fill values are not numeric.
        NotImplementedError
            If fill_mode='fast' (reserved for Phase 2).
        
        Examples
        --------
        >>> # Calibration workflow: maximum speed
        >>> adf.set_subframe_fill(
        ...     'DITS0FitSide',
        ...     fill_missing=0.0,
        ...     fill_invalid=0.0,
        ...     warn_missing_keys=False,
        ...     fill_mode='direct',
        ... )
        
        >>> # Debugging: keep NaN to see missing data
        >>> adf.set_subframe_fill(
        ...     'DITS0FitSide',
        ...     fill_missing=None,  # Keep NaN
        ...     warn_missing_keys=True,
        ...     warn_threshold=0.001,  # Warn if >0.1% missing
        ...     fill_mode='safe',
        ... )
        """
        # Validate subframe exists
        if not self._subframes.get_entry(subframe_name):
            available = list(self._subframes.subframes.keys()) if hasattr(self._subframes, 'subframes') else []
            raise ValueError(
                f"Subframe '{subframe_name}' not registered. "
                f"Available subframes: {available}"
            )
        
        # Validate fill_mode
        if fill_mode is not None:
            if fill_mode == 'fast':
                raise NotImplementedError("fill_mode='fast' will be available in Phase 2")
            if fill_mode not in ('safe', 'direct'):
                raise ValueError(f"fill_mode must be 'safe' or 'direct', got '{fill_mode}'")
        
        # Validate numeric types
        for name, value in [('fill_missing', fill_missing), ('fill_nan', fill_nan),
                            ('fill_inf', fill_inf), ('fill_invalid', fill_invalid)]:
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        
        # Initialize config for this subframe if needed
        if subframe_name not in self._subframe_fill_config:
            self._subframe_fill_config[subframe_name] = {}
        
        cfg = self._subframe_fill_config[subframe_name]
        
        # Only set values that are explicitly provided
        if fill_missing is not None:
            cfg['fill_missing'] = fill_missing
        if fill_nan is not None:
            cfg['fill_nan'] = fill_nan
        if fill_inf is not None:
            cfg['fill_inf'] = fill_inf
        if fill_invalid is not None:
            cfg['fill_invalid'] = fill_invalid
        if warn_missing_keys is not None:
            cfg['warn_missing_keys'] = warn_missing_keys
        if warn_threshold is not None:
            cfg['warn_threshold'] = warn_threshold
        if fill_mode is not None:
            cfg['fill_mode'] = fill_mode
        
        # Pattern parameters are reserved for Phase 2 - silently ignore

    def clear_global_fill(self):
        """
        Reset global fill configuration to defaults.
        
        Does not affect subframe-specific configurations.
        """
        self._global_fill_config = {
            'fill_missing': None,
            'fill_nan': None,
            'fill_inf': None,
            'fill_invalid': None,
            'warn_missing_keys': True,
            'warn_threshold': 0.01,
            'fill_mode': 'safe',
        }

    def clear_subframe_fill(self, subframe_name):
        """
        Clear fill configuration for a specific subframe.
        
        After clearing, the subframe will use global defaults.
        
        Parameters
        ----------
        subframe_name : str
            Name of subframe to clear configuration for.
        
        Raises
        ------
        ValueError
            If subframe_name is not registered.
        """
        if not self._subframes.get_entry(subframe_name):
            available = list(self._subframes.subframes.keys()) if hasattr(self._subframes, 'subframes') else []
            raise ValueError(
                f"Subframe '{subframe_name}' not registered. "
                f"Available subframes: {available}"
            )
        
        if subframe_name in self._subframe_fill_config:
            del self._subframe_fill_config[subframe_name]

    def _get_fill_config(self, subframe_name):
        """
        Get resolved fill configuration for a subframe.
        
        Merges global defaults with subframe-specific overrides.
        Handles fill_invalid -> fill_nan/fill_inf expansion.
        
        Parameters
        ----------
        subframe_name : str
            Name of the subframe.
        
        Returns
        -------
        dict
            Resolved configuration with keys:
            - fill_missing: float or None
            - fill_nan: float or None  
            - fill_inf: float or None
            - warn_missing_keys: bool
            - warn_threshold: float
            - fill_mode: str
        """
        # Start with global config
        result = {
            'fill_missing': self._global_fill_config.get('fill_missing'),
            'fill_nan': self._global_fill_config.get('fill_nan'),
            'fill_inf': self._global_fill_config.get('fill_inf'),
            'warn_missing_keys': self._global_fill_config.get('warn_missing_keys', True),
            'warn_threshold': self._global_fill_config.get('warn_threshold', 0.01),
            'fill_mode': self._global_fill_config.get('fill_mode', 'safe'),
        }
        
        # Apply global fill_invalid as fallback for fill_nan/fill_inf
        global_invalid = self._global_fill_config.get('fill_invalid')
        if global_invalid is not None:
            if result['fill_nan'] is None:
                result['fill_nan'] = global_invalid
            if result['fill_inf'] is None:
                result['fill_inf'] = global_invalid
        
        # Override with subframe-specific config
        sf_cfg = self._subframe_fill_config.get(subframe_name, {})
        
        for key in ['fill_missing', 'fill_nan', 'fill_inf', 'warn_missing_keys', 
                    'warn_threshold', 'fill_mode']:
            if key in sf_cfg:
                result[key] = sf_cfg[key]
        
        # Apply subframe fill_invalid (specific overrides general)
        sf_invalid = sf_cfg.get('fill_invalid')
        if sf_invalid is not None:
            # Only apply if specific fill_nan/fill_inf not set at subframe level
            if 'fill_nan' not in sf_cfg:
                result['fill_nan'] = sf_invalid
            if 'fill_inf' not in sf_cfg:
                result['fill_inf'] = sf_invalid
        
        return result

    def _record_missing_stats(self, subframe_name, n_missing, n_total, fill_value):
        """
        Record missing key statistics for aggregated warning.
        
        Called during _prepare_subframe_joins() for each subframe column.
        """
        if subframe_name not in self._missing_key_stats:
            self._missing_key_stats[subframe_name] = {
                'count': 0,
                'total': 0,
                'fill_value': fill_value,
                'columns': 0,
            }
        
        stats = self._missing_key_stats[subframe_name]
        # Track maximum missing count across columns (they should be same for same subframe)
        if n_missing > stats['count']:
            stats['count'] = n_missing
            stats['total'] = n_total
            stats['fill_value'] = fill_value
        stats['columns'] += 1

    def _emit_missing_key_summary(self):
        """
        Emit aggregated warning about missing keys.
        
        Called at end of materialize_aliases() if any subframes had missing keys.
        """
        if not self._missing_key_stats:
            return
        
        # Check which subframes should warn
        warnings_to_emit = []
        
        for sf_name, stats in self._missing_key_stats.items():
            config = self._get_fill_config(sf_name)
            
            if not config['warn_missing_keys']:
                continue
            
            if stats['total'] == 0:
                continue
                
            frac = stats['count'] / stats['total']
            if frac > config['warn_threshold']:
                fill_str = stats['fill_value'] if stats['fill_value'] is not None else 'NaN'
                warnings_to_emit.append(
                    f"  {sf_name}: {stats['count']:,} of {stats['total']:,} keys missing "
                    f"({frac:.2%}), filled with {fill_str}"
                )
        
        if warnings_to_emit:
            msg = "[materialize_aliases] Missing key summary:\n" + "\n".join(warnings_to_emit)
            warnings.warn(msg, UserWarning)
        
        # Clear stats for next materialization
        self._missing_key_stats = {}

    def _apply_fill_config(self, sf_name, values, missing_mask, n_before):
        """
        Apply fill configuration to joined values.
        
        Handles fill_missing, fill_nan, fill_inf based on subframe config.
        
        Parameters
        ----------
        sf_name : str
            Subframe name (for config lookup)
        values : np.ndarray
            Values array to modify
        missing_mask : np.ndarray[bool]
            Mask indicating missing keys (from join)
        n_before : int
            Total row count (for statistics)
            
        Returns
        -------
        np.ndarray
            Modified values array
        """
        fill_config = self._get_fill_config(sf_name)
        fill_mode = fill_config['fill_mode']
        fill_missing = fill_config['fill_missing']
        fill_nan = fill_config['fill_nan']
        fill_inf = fill_config['fill_inf']
        
        n_missing = int(missing_mask.sum())
        
        # Record stats for aggregated warning
        self._record_missing_stats(sf_name, n_missing, n_before, fill_missing)
        
        # Convert to Series for manipulation
        values_series = pd.Series(values)
        
        if fill_mode == 'direct':
            # Direct mode: fill missing keys only
            if fill_missing is not None and n_missing > 0:
                values_series[missing_mask] = fill_missing
        
        elif fill_mode == 'safe':
            # Safe mode: separate handling of missing, NaN, Inf
            
            # 1. Handle missing keys
            if fill_missing is not None and n_missing > 0:
                values_series[missing_mask] = fill_missing
            
            # 2. Handle NaN in original subframe data (distinct from missing keys)
            if fill_nan is not None:
                original_nan_mask = values_series.isna() & ~missing_mask
                if original_nan_mask.any():
                    values_series[original_nan_mask] = fill_nan
            
            # 3. Handle Inf values
            if fill_inf is not None:
                inf_mask = np.isinf(values_series.values)
                if inf_mask.any():
                    values_series[inf_mask] = fill_inf
        
        return values_series.values

    def _run_with_profiling(self, func, profile=False, profile_text=None, profile_binary=None):
        """
        Execute function with optional cProfile profiling.
        
        Parameters
        ----------
        func : callable
            Function to execute (typically a lambda wrapping the main logic)
        profile : bool, default=False
            If True, print profiling summary to stdout
        profile_text : str, optional
            Path to save human-readable text summary (e.g., "run.txt")
        profile_binary : str, optional
            Path to save binary .prof file for snakeviz/pstats (e.g., "run.prof")
            
        Returns
        -------
        any
            Return value from func()
            
        Examples
        --------
        >>> # Print to stdout only
        >>> aDF.materialize_aliases(..., profile=True)
        
        >>> # Save binary for snakeviz
        >>> aDF.materialize_aliases(..., profile_binary="run.prof")
        
        >>> # Save text for review
        >>> aDF.materialize_aliases(..., profile_text="run.txt")
        
        >>> # Save both + print
        >>> aDF.materialize_aliases(..., profile=True, profile_text="run.txt", profile_binary="run.prof")
        """
        if not profile and not profile_text and not profile_binary:
            return func()
        
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func()
        finally:
            profiler.disable()
            
            # Build text summary
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative').print_stats(40)
            s.write("\n" + "="*60 + "\nSorted by total time:\n" + "="*60 + "\n")
            stats.sort_stats('tottime').print_stats(40)
            text_output = s.getvalue()
            
            # Print to stdout if profile=True
            if profile:
                print(text_output)
            
            # Save text file if requested
            if profile_text:
                from pathlib import Path
                Path(profile_text).write_text(text_output)
                print(f"[profiler] Text saved to: {profile_text}")
            
            # Save binary file if requested
            if profile_binary:
                profiler.dump_stats(profile_binary)
                print(f"[profiler] Binary saved to: {profile_binary}")
        
        return result

    def _default_functions(self):
        import math

        # Start with math functions (scalar fallbacks)
        env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

        # CRITICAL: Override with numpy vectorized versions
        # This ensures both arctan2 AND atan2 map to np.arctan2
        env.update(NumpyRootMapper.get_numpy_functions_for_eval())

        env["np"] = np
        for sf_name, sf_entry in self._subframes.items():
            env[sf_name] = sf_entry['frame']

        env["int"] = lambda x: np.asarray(x, dtype=np.int32)
        env["uint"] = lambda x: np.asarray(x, dtype=np.uint32)
        env["float"] = lambda x: np.asarray(x, dtype=np.float32)
        env["round"] = np.round
        env["clip"] = np.clip

        # Phase 13.9: Add registered custom functions
        if hasattr(self, '_registered_functions'):
            env.update(self._registered_functions)

        return env

    def _compute_join_indices(self, sf_name, index_cols):
        """
        Compute join index mapping from main DataFrame to subframe rows.
        
        Uses Numba JIT-compiled lookup (Phase 8b) for single-column integer keys,
        falls back to lightweight merge (keys only) for complex cases.
        
        Parameters
        ----------
        sf_name : str
            Name of the registered subframe
        index_cols : list of str
            Column names to join on
            
        Returns
        -------
        tuple : (indices, missing_mask)
            indices : np.ndarray[int64] of shape (n_main_rows,)
                For each main row, the corresponding subframe row index.
                -1 indicates missing key (no match in subframe).
            missing_mask : np.ndarray[bool] of shape (n_main_rows,)
                True where key was not found in subframe.
        
        Notes
        -----
        - Deduplicates subframe on index_cols only (not full columns)
        - Takes first match for duplicate keys (keep='first')
        - Indices refer to ORIGINAL subframe rows (before deduplication)
        - Phase 8b: Uses Numba for single-column integer keys (>10K rows)
        """
        sub_adf = self.get_subframe(sf_name)
        sub_df = sub_adf.df
        n_main = len(self.df)
        
        # Phase 8b: Try Numba path for single-column integer keys
        if (self._use_numba 
            and numba_compute_join_indices is not None
            and len(index_cols) == 1
            and n_main >= NUMBA_MIN_ROWS):
            
            col = index_cols[0]
            main_keys = self.df[col].to_numpy()
            sub_keys = sub_df[col].to_numpy()
            
            # Check if keys are integer-compatible
            if (np.issubdtype(main_keys.dtype, np.integer) and 
                np.issubdtype(sub_keys.dtype, np.integer)):
                
                # Use Numba index lookup
                indices, missing_mask, used_numba = numba_compute_join_indices(
                    main_keys.astype(np.int64),
                    sub_keys.astype(np.int64)
                )
                
                if used_numba:
                    return indices, missing_mask
        
        # Phase 8c: Try multi-column linearization for composite integer keys
        if (self._use_numba 
            and linearize_multi_column_keys_pair is not None
            and len(index_cols) > 1
            and n_main >= NUMBA_MIN_ROWS):
            
            linear_main, linear_sub, ok = linearize_multi_column_keys_pair(
                self.df, sub_df, index_cols
            )
            
            if ok:
                # Use Phase 8b hash lookup on linearized keys
                indices, missing_mask, used_numba = numba_compute_join_indices(
                    linear_main, linear_sub
                )
                if used_numba:
                    return indices, missing_mask
        
        # Fallback: Pandas merge for multi-column or non-integer keys
        # Build lightweight key table with row indices into ORIGINAL subframe
        # Critical: Add __sub_row__ BEFORE deduplication so indices map to original rows
        sub_keys_df = sub_df[index_cols].copy()
        sub_keys_df['__sub_row__'] = np.arange(len(sub_df), dtype=np.int64)
        
        # Deduplicate on index_cols only, keeping first match
        if sub_keys_df.duplicated(subset=index_cols).any():
            sub_keys_df = sub_keys_df.drop_duplicates(subset=index_cols, keep='first')
        
        # Lightweight merge: main keys -> subframe row indices
        # Left merge preserves main DataFrame row order (Many-to-One join)
        main_keys_df = self.df[index_cols]
        merged = main_keys_df.merge(sub_keys_df, on=index_cols, how='left', sort=False)
        
        # Extract indices and missing mask
        indices = merged['__sub_row__'].fillna(-1).astype(np.int64).to_numpy()
        missing_mask = (indices == -1)
        
        return indices, missing_mask

    def _extract_subframe_values_arrow(self, sf_name, sf_col, indices, missing_mask):
        """
        Extract subframe column values using PyArrow take() - gather operation.
        
        Phase 9b: Uses PyArrow's optimized C++ implementation for gathering
        values from subframe based on precomputed join indices.
        
        Parameters
        ----------
        sf_name : str
            Subframe name
        sf_col : str
            Column name to extract from subframe
        indices : np.ndarray[int64]
            Row indices into subframe (-1 for missing keys)
        missing_mask : np.ndarray[bool]
            Mask indicating missing keys (True where index == -1)
            
        Returns
        -------
        np.ndarray
            Extracted values with NaN for missing keys (before fill config)
            
        Notes
        -----
        This is a GATHER operation: for each row i in main DataFrame,
        we fetch subframe[indices[i]]. Missing keys (indices[i] == -1)
        result in NaN values.
        """
        sub_adf = self.get_subframe(sf_name)
        sub_df = sub_adf.df
        
        # Materialize subframe alias if needed
        if sf_col not in sub_df.columns:
            if sf_col in sub_adf.aliases:
                sub_adf.materialize_alias(sf_col)
                sub_df = sub_adf.df
            else:
                raise KeyError(f"Subframe '{sf_name}' does not contain column or alias '{sf_col}'")
        
        sub_values = sub_df[sf_col].to_numpy()
        
        # Convert subframe column to Arrow array
        sub_arr = pa.array(sub_values)
        
        # Handle missing keys (-1 indices):
        # 1. Replace -1 with 0 so take() doesn't fail
        # 2. Take values
        # 3. Replace values at missing positions with null
        safe_indices = np.where(indices >= 0, indices, 0)
        indices_arr = pa.array(safe_indices)
        
        # Perform the gather operation
        taken = pc.take(sub_arr, indices_arr)
        
        # Apply null mask for missing keys
        if missing_mask.any():
            null_scalar = pa.scalar(None, type=taken.type)
            mask_arr = pa.array(~missing_mask)  # True = keep value, False = null
            taken = pc.if_else(mask_arr, taken, null_scalar)
        
        # Convert to numpy - nulls become NaN for float types
        result = taken.to_numpy(zero_copy_only=False)
        
        # Ensure proper dtype for NaN handling
        if not np.issubdtype(result.dtype, np.floating) and missing_mask.any():
            result = result.astype(np.float64)
            result[missing_mask] = np.nan
        
        return result

    def _extract_subframe_values_cached(self, sf_name, sf_col, indices, missing_mask):
        """
        Extract subframe column values using cached indices.
        
        Uses acceleration in order of preference:
        1. PyArrow take() (Phase 9b) - best for large arrays
        2. Numba JIT scatter (Phase 8a) - good for repeated operations
        3. NumPy advanced indexing - fallback
        
        Parameters
        ----------
        sf_name : str
            Subframe name
        sf_col : str
            Column name to extract from subframe
        indices : np.ndarray[int64]
            Row indices into subframe (-1 for missing)
        missing_mask : np.ndarray[bool]
            Mask indicating missing keys
            
        Returns
        -------
        np.ndarray
            Extracted values with fill config applied
        """
        n = len(indices)
        
        # Phase 9b: Try PyArrow path first (fastest for large arrays)
        if (self._use_arrow and PYARROW_AVAILABLE and n >= NUMBA_MIN_ROWS):
            try:
                values = self._extract_subframe_values_arrow(sf_name, sf_col, indices, missing_mask)
                values = self._apply_fill_config(sf_name, values, missing_mask, n)
                return values
            except Exception as e:
                if not hasattr(self, '_arrow_scatter_warned'):
                    warnings.warn(
                        f"Arrow scatter failed for {sf_name}.{sf_col}, "
                        f"falling back to NumPy/Numba: {e}",
                        RuntimeWarning
                    )
                    self._arrow_scatter_warned = True
        
        # Numba/NumPy fallback path
        sub_adf = self.get_subframe(sf_name)
        sub_df = sub_adf.df
        
        # Materialize subframe alias if needed
        if sf_col not in sub_df.columns:
            if sf_col in sub_adf.aliases:
                sub_adf.materialize_alias(sf_col)
                sub_df = sub_adf.df
            else:
                raise KeyError(f"Subframe '{sf_name}' does not contain column or alias '{sf_col}'")
        
        sub_values = sub_df[sf_col].to_numpy()
        
        # Pre-fill with NaN to safely handle missing keys
        # Must upcast non-float dtypes to allow NaN representation
        if np.issubdtype(sub_values.dtype, np.floating):
            values = np.full(n, np.nan, dtype=sub_values.dtype)
        else:
            values = np.full(n, np.nan, dtype=np.float64)
        
        # Phase 8a: Use Numba scatter if available and worthwhile
        if self._use_numba and n >= NUMBA_MIN_ROWS and numba_scatter is not None:
            # Numba scatter modifies values in-place
            numba_scatter(sub_values, indices, values)
        else:
            # NumPy advanced indexing - fast C-level operation
            valid = indices >= 0
            values[valid] = sub_values[indices[valid]]
        
        # Apply fill configuration (policy stays in Python - GPT's rule)
        values = self._apply_fill_config(sf_name, values, missing_mask, n)
        
        return values

    def _prepare_subframe_joins(self, expr, warn_missing_keys=True, alias_name=None):
        """
        Prepare subframe joins for expression evaluation.
        
        Detects dotted references like `T.mX` and performs left joins to bring
        subframe columns into the main DataFrame. Uses join index caching for
        performance when multiple columns are accessed from the same subframe.
        
        Parameters
        ----------
        expr : str
            Expression containing potential subframe references (e.g., "x - T.mX")
        warn_missing_keys : bool, default=True
            Legacy parameter kept for backward compatibility.
        alias_name : str, optional
            Name of the alias being evaluated (for warning messages)
            
        Returns
        -------
        str
            Modified expression with subframe references replaced by joined column names
        """
        tokens = re.findall(r'(\b\w+)\.(\w+)', expr)
        
        for sf_name, sf_col in tokens:
            entry = self._subframes.get_entry(sf_name)
            if not entry:
                continue
            
            sub_adf = entry['frame']
            index_cols = entry['index']
            if isinstance(index_cols, str):
                index_cols = [index_cols]
            
            suffix = f'__{sf_name}'
            col_renamed = f'{sf_col}{suffix}'
            
            # Skip if column already exists (idempotent behavior)
            if col_renamed in self.df.columns:
                expr = expr.replace(f'{sf_name}.{sf_col}', col_renamed)
                continue
            
            # Check cache for precomputed join indices
            if sf_name in self._join_index_cache:
                cache_entry = self._join_index_cache[sf_name]
                # Validate cache entry (defensive check for future extensibility)
                if (cache_entry['n_rows'] == len(self.df) and 
                    cache_entry['subframe_id'] == id(sub_adf.df)):
                    # CACHE HIT: Use cached indices
                    self._join_cache_hits += 1
                    indices = cache_entry['indices']
                    missing_mask = cache_entry['missing_mask']
                    values = self._extract_subframe_values_cached(
                        sf_name, sf_col, indices, missing_mask
                    )
                    self.df[col_renamed] = values
                    expr = expr.replace(f'{sf_name}.{sf_col}', col_renamed)
                    continue
            
            # CACHE MISS: Compute join indices
            self._join_cache_misses += 1
            indices, missing_mask = self._compute_join_indices(sf_name, index_cols)
            
            # Store in cache
            self._join_index_cache[sf_name] = {
                'indices': indices,
                'missing_mask': missing_mask,
                'n_rows': len(self.df),
                'subframe_id': id(sub_adf.df),
            }
            
            # Extract values using cached indices
            values = self._extract_subframe_values_cached(
                sf_name, sf_col, indices, missing_mask
            )
            
            self.df[col_renamed] = values
            expr = expr.replace(f'{sf_name}.{sf_col}', col_renamed)
        
        return expr

    def _check_for_cycles(self):
        try:
            self._topological_sort()
        except ValueError as e:
            raise ValueError("Cycle detected in alias dependencies") from e

    def validate_no_cycles(self, raise_on_cycle=True, verbose=False):
        """
        Check alias dependency graph for cycles.
        
        Useful for debugging cycle issues. Can be called explicitly to diagnose
        problems before running materialize_aliases().
        
        Args:
            raise_on_cycle: If True, raise ValueError on cycle. If False, return cycles.
            verbose: If True, print detailed cycle information.
            
        Returns:
            list: List of cycles found (each cycle is a list of alias names).
                  Empty list if no cycles.
                  
        Raises:
            ValueError: If raise_on_cycle=True and cycles are found.
            
        Example:
            # Check for cycles
            cycles = adf.validate_no_cycles(raise_on_cycle=False, verbose=True)
            if cycles:
                print(f"Found {len(cycles)} cycles!")
                for cycle in cycles[:5]:
                    print(f"  {' -> '.join(cycle)}")
        """
        # Get subframe names
        subframe_names = set()
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            subframe_names = set(self._subframes.subframes.keys())
        
        # Build dependency graph
        g = nx.DiGraph()
        
        for alias_name, expr in self.aliases.items():
            # Clean expression: remove subframe.column patterns
            expr_cleaned = expr
            for sf_name in subframe_names:
                expr_cleaned = re.sub(rf'\b{sf_name}\.\w+', '', expr_cleaned)
            
            # Find tokens that are aliases
            tokens = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr_cleaned))
            deps = tokens & set(self.aliases.keys())
            
            g.add_node(alias_name)
            for dep in deps:
                g.add_edge(alias_name, dep)
        
        # Find cycles
        cycles = []
        try:
            cycles = list(nx.simple_cycles(g))
        except Exception:
            pass
        
        # Report
        if verbose and cycles:
            print(f"\n⚠️  Found {len(cycles)} cycles in alias dependency graph:")
            
            # Separate self-referential from indirect
            self_refs = [c for c in cycles if len(c) == 1 or (len(c) == 2 and c[0] == c[-1])]
            indirect = [c for c in cycles if c not in self_refs]
            
            if self_refs:
                print(f"\n  Self-referential ({len(self_refs)}):")
                for cycle in self_refs[:10]:
                    name = cycle[0]
                    expr = self.aliases.get(name, 'N/A')
                    expr_display = expr[:50] + '...' if len(expr) > 50 else expr
                    print(f"    {name} -> {name}  (expr: {expr_display})")
                if len(self_refs) > 10:
                    print(f"    ... and {len(self_refs) - 10} more")
            
            if indirect:
                print(f"\n  Indirect cycles ({len(indirect)}):")
                for cycle in indirect[:5]:
                    print(f"    {' -> '.join(cycle)}")
                if len(indirect) > 5:
                    print(f"    ... and {len(indirect) - 5} more")
        
        if cycles and raise_on_cycle:
            raise ValueError(
                f"Found {len(cycles)} cycles in alias dependency graph. "
                f"Use validate_no_cycles(verbose=True) for details."
            )
        
        return cycles

    # =========================================================================
    # SECTION 2: Alias Management
    # =========================================================================
    #
    # Aliases are lazy-evaluated computed columns defined by expressions.
    # They are evaluated on-demand and can depend on other aliases or subframes.
    #
    # Key methods:
    # - add_alias(): Define a new computed column
    # - materialize_alias(): Evaluate and store in DataFrame
    # - get_alias_series/array(): Evaluate without storing (non-materializing)
    # - validate_aliases(): Check all aliases can be evaluated
    #
    # =========================================================================

    def add_alias(self, name, expression, dtype=None, is_constant=False, fill_value=None):
        """
        Define a new alias (lazy computed column).
        
        Args:
            name: Name of the alias.
            expression: Expression string using pandas or NumPy operations.
            dtype: Optional numpy dtype to enforce.
            is_constant: Whether the alias represents a scalar constant.
            fill_value: Optional value to replace inf and NaN in result.
                If set, np.where(np.isfinite(result), result, fill_value)
                is applied after evaluation and before dtype conversion.
            
        Phase 4: Writes to _schema["columns"] as single source of truth.
        
        Raises:
            ValueError: If alias would create a self-referential cycle.
        
        Example:
            >>> adf.add_alias('dsectorM', '18*((y+dy)/x)/pi', dtype=np.float16, fill_value=0)
        """
        # Check for self-reference BEFORE adding to schema
        # This catches cases like: add_alias('x', 'x + 1') when 'x' is already a column
        # or auto_alias creating: add_alias('dEdxTPC', 'T.dEdxTPC') when 'dEdxTPC' exists
        
        # Get subframe names to exclude from self-reference check
        subframe_names = set()
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            subframe_names = set(self._subframes.subframes.keys())
        
        # Clean expression: remove subframe.column patterns
        expr_cleaned = expression
        for sf_name in subframe_names:
            expr_cleaned = re.sub(rf'\b{sf_name}\.\w+', '', expr_cleaned)
        
        # Find remaining tokens
        tokens = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr_cleaned))
        
        # Check if name appears in its own expression (after removing subframe refs)
        if name in tokens:
            # This would create a self-referential cycle
            raise ValueError(
                f"Alias '{name}' would reference itself in expression: {expression}\n"
                f"This typically happens when:\n"
                f"  1. A column '{name}' already exists in the DataFrame\n"
                f"  2. auto_alias_subframe() tries to create alias '{name}' = 'Subframe.{name}'\n"
                f"Solution: Don't create aliases for columns that already exist."
            )
        
        # Build spec for schema
        spec = {"expr": expression}
        if dtype is not None:
            spec["dtype"] = dtype
        if is_constant:
            spec["constant"] = True
        if fill_value is not None:
            spec["fill_value"] = fill_value
        
        # Write to schema
        self._schema["columns"][name] = spec
        
        # Check for cycles (catches indirect cycles like A -> B -> A)
        self._check_for_cycles()

    def _eval_in_namespace(self, expr, context_override=None, warn_missing_keys=True, alias_name=None):
        """
        Evaluate expression in namespace with DataFrame columns, functions, and optional overrides.
        
        Phase 9c: Arrow compute path disabled here - use _materialize_aliases_arrow()
        for zero-copy pipeline instead.
        
        Parameters
        ----------
        expr : str
            Expression to evaluate
        context_override : dict, optional
            Additional variables to inject into namespace. Used by materialize_aliases()
            to pass already-computed alias values so dependent aliases can reference them
            before they're added to the DataFrame.
        warn_missing_keys : bool, default=True
            Whether to warn about missing subframe keys
        alias_name : str, optional
            Name of alias being evaluated (for warning messages)
        """
        expr = self._prepare_subframe_joins(expr, warn_missing_keys=warn_missing_keys, alias_name=alias_name)
        
        # Phase 9c note: Per-expression Arrow compute disabled here.
        # Conversion overhead per expression exceeds benefits.
        # Use _materialize_aliases_arrow() for zero-copy batch processing instead.
        
        # Python eval() path
        local_env = {col: self.df[col] for col in self.df.columns}
        
        # Merge context_override after df columns, before functions
        # This allows batched aliases to see previously computed values
        if context_override:
            local_env.update(context_override)
        
        local_env.update(self._default_functions())

        try:
            return eval(expr, {}, local_env)
        except NameError as e:
            # Function or variable not found
            missing_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
            available_funcs = sorted([k for k in local_env.keys() if callable(local_env.get(k))])[:20]
            raise NameError(
                f"Undefined function or variable '{missing_name}' in expression: {expr}\n"
                f"Available functions include: {', '.join(available_funcs)}\n"
                f"Hint: Common functions are available, including both 'arctan2' and 'atan2'"
            ) from e
        except TypeError as e:
            if "cannot convert the series" in str(e):
                raise TypeError(
                    f"Scalar function used on array data in expression: {expr}\n"
                    f"Error: {e}\n"
                    f"Hint: All math functions should be vectorized (numpy-based). "
                    f"If you see this with standard functions like 'atan2', please report as a bug."
                ) from e
            raise
    
    def _eval_arrow(self, expr, context_override=None, return_arrow=False, arrow_context=None):
        """
        Evaluate expression using PyArrow compute.
        
        Phase 9c/9e: Uses ArrowComputeMapper to compile expressions to PyArrow
        compute function chains, eliminating Python dispatch overhead.
        
        Parameters
        ----------
        expr : str
            Expression to evaluate (already processed by _prepare_subframe_joins)
        context_override : dict, optional
            Additional variables from previously computed aliases (numpy/pandas)
        return_arrow : bool, default=False
            If True, return PyArrow array (for zero-copy pipeline).
            If False, return pandas Series (backward compatible).
        arrow_context : dict, optional
            Pre-converted Arrow arrays (for zero-copy pipeline). If provided,
            skips conversion of DataFrame columns to Arrow.
            
        Returns
        -------
        pa.Array, pd.Series, or None
            - pa.Array if return_arrow=True and Arrow succeeded
            - pd.Series if return_arrow=False and Arrow succeeded  
            - None if should fallback to eval()
            
        Notes
        -----
        Returns None (triggering fallback) for:
        - Expressions with unsupported functions
        - Expressions referencing non-numeric columns
        - Any compilation or execution errors
        """
        # Check if expression is likely supported
        if not ArrowComputeMapper.is_supported(expr):
            return None
        
        # Use pre-built Arrow context if provided (zero-copy pipeline)
        if arrow_context is not None:
            pa_ctx = arrow_context
        else:
            # Build context with numeric columns only
            arrow_ctx = {}
            
            # Add DataFrame columns
            for col in self.df.columns:
                series = self.df[col]
                # Only include numeric columns (Arrow compute is for numeric ops)
                if np.issubdtype(series.dtype, np.number):
                    arrow_ctx[col] = series.values
            
            # Add context overrides (previously computed aliases)
            if context_override:
                for name, value in context_override.items():
                    if hasattr(value, 'values'):
                        # pandas Series
                        if np.issubdtype(value.dtype, np.number):
                            arrow_ctx[name] = value.values
                    elif hasattr(value, '__array__'):
                        # numpy array
                        arr = np.asarray(value)
                        if np.issubdtype(arr.dtype, np.number):
                            arrow_ctx[name] = arr
                    elif isinstance(value, (int, float)):
                        # Scalar
                        arrow_ctx[name] = value
            
            # Convert context to Arrow arrays
            pa_ctx = {}
            for name, value in arrow_ctx.items():
                if isinstance(value, np.ndarray):
                    pa_ctx[name] = pa.array(value)
                elif isinstance(value, (int, float)):
                    pa_ctx[name] = pa.scalar(value)
                else:
                    pa_ctx[name] = value
        
        # Compile and execute
        try:
            compiled = ArrowComputeMapper.compile(expr)
            
            # Execute
            result_arrow = compiled(pa_ctx)
            
            # Return Arrow array for zero-copy pipeline
            if return_arrow:
                return result_arrow
            
            # Convert back to pandas Series (backward compatible)
            result_np = result_arrow.to_numpy(zero_copy_only=False)
            return pd.Series(result_np, index=self.df.index)
            
        except (ValueError, KeyError, TypeError) as e:
            # Expression not fully supported, fall back to eval()
            return None

    def _resolve_dependencies(self):
        """
        Resolve alias dependencies for cycle detection and topological sorting.
        
        Handles:
        - Regular alias references (alias_name)
        - Subframe references (subframe_name.column) - NOT treated as dependencies
        - Self-references are skipped to avoid false cycles
        """
        from collections import defaultdict
        dependencies = defaultdict(set)
        
        # Get all subframe names to exclude them from dependency tracking
        subframe_names = set()
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            subframe_names = set(self._subframes.subframes.keys())
        
        for name, expr in self.aliases.items():
            # Find all word tokens, but handle dotted expressions specially
            # First, remove subframe references like "subframe.column" from consideration
            expr_cleaned = expr
            for sf_name in subframe_names:
                # Remove "subframe.anything" patterns
                expr_cleaned = re.sub(rf'\b{sf_name}\.\w+', '', expr_cleaned)
            
            # Now find remaining tokens
            tokens = re.findall(r'\b\w+\b', expr_cleaned)
            
            for token in tokens:
                # Skip self-references (alias depending on itself is always wrong)
                if token == name:
                    continue
                # Skip subframe names (they're not aliases)
                if token in subframe_names:
                    continue
                # Only add if token is actually an alias
                if token in self.aliases:
                    dependencies[name].add(token)
        
        return dependencies

    def plot_alias_dependencies(self):
        deps = self._resolve_dependencies()
        G = nx.DiGraph()
        for alias, subdeps in deps.items():
            for dep in subdeps:
                G.add_edge(dep, alias)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, arrows=True)
        plt.title("Alias Dependency Graph")
        plt.show()

    def _topological_sort(self):
        from collections import defaultdict, deque
        # Note: Do NOT call _check_for_cycles here - it would cause infinite recursion
        # since _check_for_cycles calls _topological_sort
        dependencies = self._resolve_dependencies()
        reverse_deps = defaultdict(set)
        indegree = defaultdict(int)
        for alias, deps in dependencies.items():
            indegree[alias] = len(deps)
            for dep in deps:
                reverse_deps[dep].add(alias)
        queue = deque([alias for alias in self.aliases if indegree[alias] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in reverse_deps[node]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if len(result) != len(self.aliases):
            raise ValueError("Cycle detected in alias dependencies")
        return result

    def _analyze_expression(self, expr):
        """
        Unified AST analysis for expression (Phase 9e).
        
        Single-pass AST walker that extracts all information needed for
        Arrow pipeline decisions and execution.
        
        Parameters
        ----------
        expr : str
            Expression to analyze
            
        Returns
        -------
        dict with:
            'column_refs': set of column names referenced (excluding functions)
            'subframe_refs': list of (sf_name, sf_col) tuples  
            'is_supported': bool - can ArrowComputeMapper handle it
            'unsupported_reason': str or None
            
        Notes
        -----
        This replaces multiple separate regex/AST parsing passes with
        a single unified analysis. Used by:
        - _can_use_arrow_pipeline()
        - _get_required_columns()
        - Subframe detection
        """
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            return {
                'column_refs': set(),
                'subframe_refs': [],
                'is_supported': False,
                'unsupported_reason': f"Syntax error: {e}"
            }
        
        column_refs = set()
        subframe_refs = []
        unsupported = None
        
        # Get subframe names for disambiguation
        subframe_names = set()
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            subframe_names = set(self._subframes.subframes.keys())
        
        # Known function names to exclude from column refs
        known_funcs = set(ArrowComputeMapper.FUNC_MAP.keys()) if ArrowComputeMapper else set()
        known_funcs.update(['np', 'numpy', 'math', 'abs', 'int', 'float', 'round', 
                           'min', 'max', 'sum', 'len', 'range', 'True', 'False', 'None'])
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                # Skip function names, subframe names, and builtins
                if (name not in known_funcs and 
                    name not in subframe_names and
                    not name.startswith('_')):
                    column_refs.add(name)
                    
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    obj_name = node.value.id
                    attr_name = node.attr
                    
                    # Subframe reference: sf_name.column
                    if obj_name in subframe_names:
                        subframe_refs.append((obj_name, attr_name))
                    # np.pi, math.e, etc. - just skip (not unsupported)
                    elif obj_name not in ('np', 'numpy', 'math'):
                        # Unknown attribute access - mark as unsupported
                        if unsupported is None:
                            unsupported = f"Unsupported attribute: {obj_name}.{attr_name}"
                            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if (ArrowComputeMapper and 
                        func_name not in ArrowComputeMapper.FUNC_MAP and
                        func_name not in ('abs', 'round', 'min', 'max')):
                        if unsupported is None:
                            unsupported = f"Unsupported function: {func_name}"
                elif isinstance(node.func, ast.Attribute):
                    # Handle np.sqrt(), etc.
                    if isinstance(node.func.value, ast.Name):
                        module = node.func.value.id
                        func_name = node.func.attr
                        if module in ('np', 'numpy'):
                            # NumPy function - check if it's in our FUNC_MAP
                            if (ArrowComputeMapper and 
                                func_name not in ArrowComputeMapper.FUNC_MAP):
                                if unsupported is None:
                                    unsupported = f"Unsupported numpy function: np.{func_name}"
                        elif module != 'math':
                            if unsupported is None:
                                unsupported = f"Unsupported method call: {module}.{func_name}"
                                
            elif isinstance(node, ast.Subscript):
                # Array subscripting not supported in Arrow pipeline
                if unsupported is None:
                    unsupported = "Subscript expressions not supported"
        
        return {
            'column_refs': column_refs,
            'subframe_refs': subframe_refs,
            'is_supported': unsupported is None,
            'unsupported_reason': unsupported
        }

    def validate_aliases(self):
        """
        Validate that all aliases can be resolved.
        
        An alias is "broken" if it references variables that don't exist as:
        - DataFrame columns
        - Other defined aliases  
        - Subframe columns (T.column syntax)
        - Known functions/constants (np, pi, etc.)
        
        Returns
        -------
        list
            Names of aliases that cannot be resolved
        """
        broken = []
        
        # Known functions and constants that are always available
        known_names = set(self._default_functions().keys())
        known_names.update(['np', 'pi', 'abs', 'int', 'float', 'round', 'sqrt', 
                           'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'atan2',
                           'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan'])
        
        # All resolvable names: columns + aliases + known functions
        resolvable = set(self.df.columns) | set(self.aliases.keys()) | known_names
        
        for name, expr in self.aliases.items():
            # Extract tokens from expression
            tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
            
            missing = []
            for token in tokens:
                # Skip numeric literals that might be partially matched
                if token.isdigit():
                    continue
                    
                # Check if it's a subframe reference (handled separately)
                if '.' in expr:
                    # Check for T.column pattern
                    subframe_refs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)', expr)
                    for sf_name, sf_col in subframe_refs:
                        if sf_name == token:
                            # This token is a subframe name, check if it exists
                            sf = self.get_subframe(sf_name)
                            if sf is None:
                                missing.append(f"{sf_name} (subframe)")
                            elif sf_col not in sf.df.columns and sf_col not in sf.aliases:
                                missing.append(f"{sf_name}.{sf_col}")
                            continue
                
                # Check if token is resolvable
                if token not in resolvable:
                    # Check if it's part of a subframe reference
                    if not any(token == sf_ref[0] for sf_ref in 
                              re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\.', expr)):
                        missing.append(token)
            
            if missing:
                broken.append(name)
        
        return broken

    # Verbosity flags for describe_aliases (bitmask)
    ALIAS_SHOW_CORE   = 0x01  # name, kind, materialized, dtype, expr (always on)
    ALIAS_SHOW_DEPS   = 0x02  # dependency list
    ALIAS_SHOW_ERRORS = 0x04  # missing symbols, parse errors
    ALIAS_SHOW_STATS  = 0x08  # stats if materialized (mean, std, n_nan)
    ALIAS_SHOW_ALL    = 0x0F  # all flags

    def select_aliases(self, pattern=None, names=None, only_broken=False,
                       only_materialized=False, only_unmaterialized=False,
                       with_dependencies=False):
        """
        Select aliases by pattern and/or names with optional filters.
        
        This is the core selection logic used by describe_aliases, 
        materialize_aliases, etc.
        
        Parameters
        ----------
        pattern : str, optional
            Regex pattern to match alias names
        names : list, optional
            Explicit list of alias names to include
        only_broken : bool, default=False
            If True, include only broken aliases
        only_materialized : bool, default=False
            If True, include only materialized aliases
        only_unmaterialized : bool, default=False
            If True, include only unmaterialized aliases
        with_dependencies : bool, default=False
            If True, expand selection to include all dependencies
            
        Returns
        -------
        list
            List of alias names matching the criteria
            
        Examples
        --------
        >>> adf.select_aliases(pattern=r'is.*')  # Names starting with 'is'
        >>> adf.select_aliases(names=['r', 'phi'])  # Specific names
        >>> adf.select_aliases(only_broken=True)  # All broken aliases
        >>> adf.select_aliases(pattern=r'dy.*', only_unmaterialized=True)
        >>> adf.select_aliases(names=['cosPhi'], with_dependencies=True)  # includes 'phi'
        """
        import re as re_module
        
        # Validate mutually exclusive filters
        exclusive_filters = [only_broken, only_materialized, only_unmaterialized]
        if sum(bool(x) for x in exclusive_filters) > 1:
            raise ValueError(
                "Filters only_broken, only_materialized, only_unmaterialized are mutually exclusive"
            )
        
        # Compile and validate pattern
        regex = None
        if pattern:
            try:
                regex = re_module.compile(pattern)
            except re_module.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        # Get broken aliases for filtering
        broken_aliases = set(self.validate_aliases()) if only_broken else None
        
        # Build result list
        result = []
        
        for name in self.aliases:
            # Filter by explicit names
            if names is not None and name not in names:
                continue
                
            # Filter by pattern
            if regex and not regex.search(name):
                continue
            
            # Filter by broken status
            if only_broken:
                if name not in broken_aliases:
                    continue
            
            # Filter by materialized status
            materialized = name in self.df.columns
            if only_materialized and not materialized:
                continue
            if only_unmaterialized and materialized:
                continue
            
            result.append(name)
        
        # Expand with dependencies if requested
        if with_dependencies and result:
            import networkx as nx
            
            def build_graph():
                g = nx.DiGraph()
                for alias, expr in self.aliases.items():
                    g.add_node(alias)
                    for token in re.findall(r'\b\w+\b', expr):
                        # Exclude self-references to prevent false cycles
                        # (e.g., alias 'val' with expr 'T.val' extracts token 'val')
                        if token in self.aliases and token != alias:
                            g.add_edge(token, alias)
                return g
            
            g = build_graph()
            expanded = set(result)
            for name in result:
                try:
                    expanded |= nx.ancestors(g, name)
                except nx.NetworkXError:
                    pass
            
            # Return in topological order if possible
            try:
                ordered = list(nx.topological_sort(g.subgraph(expanded)))
                result = [n for n in ordered if n in expanded]
            except nx.NetworkXUnfeasible:
                # Find and report cycles with helpful error message
                cycles = list(nx.simple_cycles(g.subgraph(expanded)))
                if cycles:
                    max_cycles = 5
                    shown = cycles[:max_cycles]
                    cycle_info = []
                    for cycle in shown:
                        cycle_str = ' -> '.join(cycle) + ' -> ' + cycle[0]
                        exprs = [f"    {a} = {self.aliases.get(a, '[not found]')}" for a in cycle]
                        cycle_info.append(f"  Cycle: {cycle_str}\n" + '\n'.join(exprs))
                    
                    msg = (
                        f"Dependency cycle detected in aliases "
                        f"({len(cycles)} total, showing first {len(shown)}):\n"
                        + '\n'.join(cycle_info)
                        + "\n\nHint: Self-referential aliases often occur when an alias name "
                        "matches a subframe column name.\n"
                        "To diagnose: adf.validate_no_cycles(raise_on_cycle=False)"
                    )
                    raise ValueError(msg)
                else:
                    # Shouldn't happen, but fallback
                    result = list(expanded)
            except nx.NetworkXError:
                result = list(expanded)
        
        return result

    def describe_aliases(self, verbosity=0x05, pattern=None, names=None, as_dict=False,
                         only_broken=False, only_materialized=False, 
                         only_unmaterialized=False, with_dependencies=False, color=False,
                         expr_width=120):
        """
        Print summary of all aliases with name, type, materialized status, and expression.
        
        Parameters
        ----------
        verbosity : int, default=ALIAS_SHOW_CORE | ALIAS_SHOW_ERRORS (0x05)
            Bitmask controlling output detail:
            - ALIAS_SHOW_CORE (0x01): name, kind, materialized, dtype, expr (always on)
            - ALIAS_SHOW_DEPS (0x02): show dependency list
            - ALIAS_SHOW_ERRORS (0x04): show missing symbols for broken aliases
            - ALIAS_SHOW_STATS (0x08): show stats for materialized aliases
            - ALIAS_SHOW_ALL (0x0F): all flags
        pattern : str, optional
            Regex pattern to filter aliases by name
        names : list, optional
            Explicit list of alias names to include
        as_dict : bool, default=False
            If True, return dict instead of printing
        only_broken : bool, default=False
            If True, show only broken aliases
        only_materialized : bool, default=False
            If True, show only materialized aliases
        only_unmaterialized : bool, default=False
            If True, show only unmaterialized aliases
        with_dependencies : bool, default=False
            If True, expand selection to include all dependencies
        color : bool, default=False
            If True, use ANSI colors in output (green=OK, red=broken)
        expr_width : int or None, default=120
            Maximum width for expression display. If None, no truncation.
            
        Returns
        -------
        dict or None
            If as_dict=True, returns structured dict of alias info
        """
        # Mask unknown verbosity bits
        verbosity &= self.ALIAS_SHOW_ALL
        
        # Use select_aliases for filtering
        selected = self.select_aliases(
            pattern=pattern, names=names,
            only_broken=only_broken, 
            only_materialized=only_materialized,
            only_unmaterialized=only_unmaterialized,
            with_dependencies=with_dependencies
        )
        
        # Gather validation info
        broken_aliases = set(self.validate_aliases())
        
        # Build detailed info for broken aliases
        resolvable = set(self.df.columns) | set(self.aliases.keys()) | set(self._default_functions().keys())
        
        # Compute dependencies once
        deps = self._resolve_dependencies()
        
        # Color codes (used if color=True)
        if color:
            C_RED = '\033[91m'
            C_GREEN = '\033[92m'
            C_YELLOW = '\033[93m'
            C_RESET = '\033[0m'
        else:
            C_RED = C_GREEN = C_YELLOW = C_RESET = ''
        
        # Build result dict
        result = {}
        
        for name in selected:
            expr = self.aliases[name]
            
            # Determine kind
            if name in self.constant_aliases:
                kind = "constant"
            else:
                kind = "alias"
            
            # Check if materialized
            materialized = name in self.df.columns
            
            # Get dtype
            dtype = self.alias_dtypes.get(name)
            dtype_str = dtype.__name__ if dtype and hasattr(dtype, '__name__') else str(dtype) if dtype else None
            
            # Check for broken
            is_broken = name in broken_aliases
            missing = []
            if is_broken:
                tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
                missing = [t for t in tokens if t not in resolvable and not t.isdigit()]
            
            # Get dependencies
            alias_deps = sorted(deps.get(name, []))
            
            # Build entry
            entry = {
                'name': name,
                'kind': kind,
                'materialized': materialized,
                'dtype': dtype_str,
                'expr': expr,
                'broken': is_broken,
                'missing': missing if missing else None,
                'deps': alias_deps if alias_deps else None,
            }
            
            # Add stats if requested and materialized
            if (verbosity & self.ALIAS_SHOW_STATS) and materialized:
                col = self.df[name]
                entry['stats'] = {
                    'mean': float(col.mean()) if col.dtype.kind in 'iuf' else None,
                    'std': float(col.std()) if col.dtype.kind in 'iuf' else None,
                    'n_nan': int(col.isna().sum()),
                    'n_total': len(col),
                }
            
            result[name] = entry
        
        if as_dict:
            return result
        
        # Print output
        print("Aliases:")
        print(f"  {'Name':<28} {'Kind':<10} {'Mat':<5} {'Dtype':<10} Expression")
        print(f"  {'-'*28} {'-'*10} {'-'*5} {'-'*10} {'-'*40}")
        
        for name, info in result.items():
            mat_str = "Yes" if info['materialized'] else "No"
            dtype_str = info['dtype'] if info['dtype'] else "-"
            kind_str = info['kind']
            
            # Mark broken in kind column with optional color
            if info['broken']:
                kind_str = f"{C_RED}BROKEN{C_RESET}" if color else "BROKEN"
            elif info['materialized'] and color:
                mat_str = f"{C_GREEN}Yes{C_RESET}"
            
            # Truncate long expressions based on expr_width
            expr = info['expr']
            if expr_width is None:
                expr_display = expr
            elif len(expr) <= expr_width:
                expr_display = expr
            else:
                expr_display = expr[:expr_width - 3] + "..."
            
            print(f"  {name:<28} {kind_str:<10} {mat_str:<5} {dtype_str:<10} {expr_display}")
            
            # Show errors if requested
            if (verbosity & self.ALIAS_SHOW_ERRORS) and info['missing']:
                missing_str = f"{C_RED}{info['missing']}{C_RESET}" if color else str(info['missing'])
                print(f"  {'':<28} {'^ Missing:':<10} {missing_str}")
            
            # Show deps if requested
            if (verbosity & self.ALIAS_SHOW_DEPS) and info['deps']:
                print(f"  {'':<28} {'Deps:':<10} {info['deps']}")
            
            # Show stats if requested
            if (verbosity & self.ALIAS_SHOW_STATS) and info.get('stats'):
                stats = info['stats']
                if stats['mean'] is not None:
                    print(f"  {'':<28} {'Stats:':<10} mean={stats['mean']:.4g}, std={stats['std']:.4g}, nan={stats['n_nan']}/{stats['n_total']}")
                else:
                    print(f"  {'':<28} {'Stats:':<10} nan={stats['n_nan']}/{stats['n_total']}")
        
        # Summary
        n_broken = sum(1 for info in result.values() if info['broken'])
        n_materialized = sum(1 for info in result.values() if info['materialized'])
        print(f"\nTotal: {len(result)} aliases, {n_materialized} materialized, {n_broken} broken")

    def dependency_tree(self, alias, max_depth=None, show_expr=True, _depth=0, _prefix="", _is_last=True):
        """
        Print hierarchical dependency tree for an alias.
        
        Shows the complete dependency structure with visual tree formatting,
        including DataFrame columns and subframe references.
        
        Parameters
        ----------
        alias : str
            Root alias to show tree for
        max_depth : int, optional
            Maximum depth to traverse (None = unlimited)
        show_expr : bool, default=True
            If True, show expression next to each node
            
        Examples
        --------
        >>> adf.dependency_tree('isOKGBTrackFit0')
        isOKGBTrackFit0 = (row<152) & (abs(dyC0T)<2) & ...
        ├── dyC0T = dy_c - dyC0T_median
        │   ├── dy_c [column]
        │   └── dyC0T_median = DTrack0.dyC0T_median
        │       └── DTrack0.dyC0T_median [subframe]
        └── isNotEdge = abs(y+dy)<(x*(pi/18)-1.5)
            └── dy = T.dy
                └── T.dy [subframe]
        """
        # Handle internal recursion parameters
        if _depth == 0:
            # Root call - print the root node
            if alias in self.aliases:
                expr = self.aliases[alias]
                if show_expr:
                    print(f"{alias} = {expr}")
                else:
                    print(alias)
            elif alias in self.df.columns:
                print(f"{alias} [column]")
                return
            else:
                print(f"{alias} [unknown]")
                return
        
        # Check max depth
        if max_depth is not None and _depth >= max_depth:
            return
        
        # Get dependencies for this alias
        if alias not in self.aliases:
            return
        
        expr = self.aliases[alias]
        
        # Parse expression to find dependencies
        deps = self._get_alias_dependencies(alias, expr)
        
        if not deps:
            return
        
        # Sort dependencies for consistent output
        deps = sorted(deps, key=lambda x: (x[0] != 'alias', x[1]))  # aliases first
        
        for i, (dep_type, dep_name) in enumerate(deps):
            is_last = (i == len(deps) - 1)
            
            # Build the tree connectors
            if _depth == 0:
                connector = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "
            else:
                connector = _prefix + ("└── " if is_last else "├── ")
                child_prefix = _prefix + ("    " if is_last else "│   ")
            
            # Print the node based on type
            if dep_type == 'column':
                print(f"{connector}{dep_name} [column]")
            elif dep_type == 'subframe':
                print(f"{connector}{dep_name} [subframe]")
            elif dep_type == 'alias':
                dep_expr = self.aliases.get(dep_name, "")
                if show_expr and dep_expr:
                    print(f"{connector}{dep_name} = {dep_expr}")
                else:
                    print(f"{connector}{dep_name}")
                # Recurse into alias
                self.dependency_tree(
                    dep_name, 
                    max_depth=max_depth, 
                    show_expr=show_expr,
                    _depth=_depth + 1, 
                    _prefix=child_prefix,
                    _is_last=is_last
                )
    
    def _get_alias_dependencies(self, alias_name, expr):
        """
        Parse expression to extract typed dependencies.
        
        Returns list of (type, name) tuples where type is:
        - 'column': DataFrame column
        - 'alias': Another alias
        - 'subframe': Subframe reference (e.g., T.mX)
        
        Parameters
        ----------
        alias_name : str
            Name of alias being analyzed (to avoid self-reference)
        expr : str
            Expression to parse
            
        Returns
        -------
        list of (str, str)
            List of (dependency_type, dependency_name) tuples
        """
        deps = []
        
        # Get subframe names (eager + lazy)
        subframe_names = set()
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            subframe_names = set(self._subframes.subframes.keys())
        # Also include lazy subframes
        if hasattr(self, '_subframe_readers'):
            subframe_names |= set(self._subframe_readers.keys())
        
        # Known function names to exclude
        known_funcs = set(self._default_functions().keys())
        known_funcs.update(['np', 'numpy', 'math', 'abs', 'int', 'float', 'round', 
                           'min', 'max', 'sum', 'len', 'range', 'True', 'False', 'None',
                           'pi', 'e', 'inf', 'nan'])
        
        # Find subframe references first (T.column pattern)
        subframe_refs = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
        subframe_ref_names = set()
        for sf_name, sf_col in subframe_refs:
            if sf_name in subframe_names:
                deps.append(('subframe', f"{sf_name}.{sf_col}"))
                subframe_ref_names.add(sf_name)
        
        # Find all identifiers
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
        
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            
            # Skip known functions and constants
            if token in known_funcs:
                continue
            
            # Skip subframe names (they're part of subframe refs)
            if token in subframe_names:
                continue
            
            # Skip self-reference
            if token == alias_name:
                continue
            
            # Skip numeric-looking tokens
            if token.isdigit():
                continue
            
            # Classify the dependency
            if token in self.aliases:
                deps.append(('alias', token))
            elif token in self.df.columns:
                deps.append(('column', token))
            # else: unknown identifier (function, constant, etc.) - skip
        
        return deps

    def materialize_alias(self, name, cleanTemporary=False, dtype=None, warn_missing_keys=True,
                          profile=False, profile_text=None, profile_binary=None):
        """
        Evaluate an alias and store its result as a real column.
        
        This is the simple, immediate materialization path. For batch operations,
        use materialize_aliases() which is optimized to avoid DataFrame fragmentation.
        
        Args:
            name: Alias name to materialize.
            cleanTemporary: Whether to clean up intermediate dependencies.
            dtype: Optional override dtype to cast to.
            warn_missing_keys: If True, emit warning when subframe join has missing keys.
                             Missing keys produce NaN (rows are never dropped).
                             This parameter temporarily overrides the fill config setting.
            profile: If True, print profiling summary to stdout.
            profile_text: If provided, save text profile to this file path.
            profile_binary: If provided, save binary .prof file for snakeviz/pstats.

        Raises:
            KeyError: If alias is not defined.
            Exception: If alias evaluation fails.
        """
        def _do_materialize():
            # Reset missing key stats for this single-alias call
            self._missing_key_stats = {}
            
            # Handle legacy warn_missing_keys parameter by temporarily overriding fill config
            original_warn_setting = None
            if not warn_missing_keys:
                original_warn_setting = self._global_fill_config.get('warn_missing_keys', True)
                self._global_fill_config['warn_missing_keys'] = False
            
            try:
                if name not in self.aliases:
                    print(f"[materialize_alias] Warning: alias '{name}' not found.")
                    return
                expr = self.aliases[name]
                
                # Phase 7.5a: Load lazy subframes referenced by this alias
                needed_subframes = self._get_subframes_for_aliases([name])
                for sf_name in needed_subframes:
                    self.ensure_subframe(sf_name)

                # Automatically materialize any referenced aliases or subframe aliases
                # CRITICAL: Match 'word.word' BEFORE 'word' to correctly detect subframe references
                tokens = re.findall(r'\w+\.\w+|\b\w+\b', expr)
                for token in tokens:
                    if '.' in token:
                        sf_name, sf_attr = token.split('.', 1)
                        sf = self.get_subframe(sf_name)
                        if sf:
                            # CRITICAL: Materialize subframe index columns first (if they're aliases)
                            # This fixes the bug where joins fail because index columns aren't materialized
                            entry = self._subframes.get_entry(sf_name)
                            if entry:
                                index_cols = entry['index']
                                if isinstance(index_cols, str):
                                    index_cols = [index_cols]
                                for idx_col in index_cols:
                                    if idx_col in self.aliases and idx_col not in self.df.columns:
                                        self.materialize_alias(idx_col, warn_missing_keys=warn_missing_keys)
                            
                            # Materialize the subframe attribute itself
                            if sf_attr in sf.aliases and sf_attr not in sf.df.columns:
                                sf.materialize_alias(sf_attr)
                    elif token == name:
                        # Skip self-reference to prevent infinite recursion
                        # (alias 'x' referencing 'subframe.x' where 'x' is extracted as a token)
                        continue
                    elif token in self.aliases and token not in self.df.columns:
                        self.materialize_alias(token, warn_missing_keys=warn_missing_keys)

                result = self._eval_in_namespace(expr, warn_missing_keys=warn_missing_keys, alias_name=name)
                
                # Phase 13.9: Apply fill_value for inf/NaN replacement
                alias_spec = self._schema["columns"].get(name, {})
                fill_val = alias_spec.get("fill_value")
                if fill_val is not None:
                    result = np.where(np.isfinite(result), result, fill_val)
                
                result_dtype = dtype or self.alias_dtypes.get(name)
                if result_dtype is not None:
                    try:
                        result = result.astype(result_dtype)
                    except AttributeError:
                        result = result_dtype(result)
                self.df[name] = result
                
                # Emit aggregated warning BEFORE restoring config (so warn_missing_keys=False takes effect)
                self._emit_missing_key_summary()
                
            finally:
                # Restore original warn_missing_keys setting if we changed it
                if original_warn_setting is not None:
                    self._global_fill_config['warn_missing_keys'] = original_warn_setting
        
        return self._run_with_profiling(_do_materialize, profile, profile_text, profile_binary)

    def _materialize_aliases_arrow(self, to_materialize, verbose=False):
        """
        Attempt to materialize aliases using Arrow pipeline.
        
        Phase 9e Step 1: Simple expressions only (no subframe references).
        
        NOTE: Benchmarking showed that PyArrow compute is 8-10x SLOWER than NumPy
        for element-wise math expressions. This method therefore returns None
        to trigger fallback to the standard (faster) path.
        
        Arrow is beneficial for:
        - Scatter operations (pc.take) - handled by Phase 9b in standard path
        - Complex filtering/selection - not yet implemented
        
        Arrow is NOT beneficial for:
        - Element-wise math (sqrt, power, add, etc.) - NumPy is much faster
        
        Parameters
        ----------
        to_materialize : list
            List of alias names to materialize
        verbose : bool, default=False
            If True, print progress information
            
        Returns
        -------
        dict or None
            Always returns None to trigger fallback to standard path,
            which is faster for element-wise expressions.
        """
        # After benchmarking, we found that Arrow compute is significantly slower
        # than NumPy for element-wise math. The standard path (Python eval with
        # NumPy) is the fastest option for expression evaluation.
        #
        # Phase 9b (Arrow scatter via pc.take) is still beneficial and is
        # handled in _extract_subframe_values_arrow via the standard path.
        #
        # This method returns None to ensure we always use the standard path.
        
        if verbose:
            print("[Arrow pipeline] Disabled: NumPy eval is faster than Arrow compute")
        
        return None

    def materialize_aliases(self, pattern=None, names=None, with_dependencies=True,
                            only_unmaterialized=True, cleanTemporary=True, verbose=False,
                            profile=False, profile_text=None, profile_binary=None):
        """
        Materialize aliases matching pattern and/or names using batch optimization.
        
        This method uses batched pd.concat to avoid DataFrame fragmentation, which
        provides ~3x performance improvement over sequential column insertion.
        
        Parameters
        ----------
        pattern : str, optional
            Regex pattern to match alias names (e.g. r'^is' for aliases starting with 'is')
        names : list, optional
            Explicit list of alias names to materialize
        with_dependencies : bool, default=True
            If True, materialize dependencies in correct order
        only_unmaterialized : bool, default=True
            If True, skip aliases already materialized as columns
        cleanTemporary : bool, default=True
            If True, remove intermediate dependencies that weren't targets
        verbose : bool, default=False
            If True, print progress information
        profile : bool, default=False
            If True, print profiling summary to stdout
        profile_text : str, optional
            Path to save human-readable text profile (e.g., "run.txt")
        profile_binary : str, optional
            Path to save binary .prof file for snakeviz/pstats (e.g., "run.prof")
            
        Returns
        -------
        list
            Names of aliases that were materialized
            
        Examples
        --------
        >>> adf.materialize_aliases(pattern=r'is.*')  # All 'is*' aliases
        >>> adf.materialize_aliases(names=['r', 'phi', 'cosPhi'])  # Specific names
        >>> adf.materialize_aliases(pattern=r'dy.*|dz.*')  # dy and dz aliases
        >>> adf.materialize_aliases(names=['x'], profile=True)  # Print to stdout
        >>> adf.materialize_aliases(names=['x'], profile_binary="run.prof")  # For snakeviz
        """
        def _do_materialize():
            # Reset missing key stats for this materialization batch
            self._missing_key_stats = {}
            
            # Reset join cache statistics for this batch (Phase 4)
            self._join_cache_hits = 0
            self._join_cache_misses = 0
            
            # Get primary targets first (without dependencies)
            targets = self.select_aliases(
                pattern=pattern, 
                names=names,
                only_unmaterialized=only_unmaterialized,
                with_dependencies=False
            )
            
            if verbose:
                print(f"[materialize_aliases] Selected {len(targets)} targets: {targets}")
            
            if not targets:
                return []
            
            # Get full list with dependencies in topological order
            if with_dependencies:
                to_materialize = self.select_aliases(
                    names=targets,
                    only_unmaterialized=only_unmaterialized,
                    with_dependencies=True
                )
                if verbose:
                    print(f"[materialize_aliases] With dependencies: {to_materialize}")
            else:
                to_materialize = targets
            
            # =========================================================================
            # PHASE 7.5a: ENSURE LAZY SUBFRAMES ARE LOADED
            # Before materializing, load any lazy subframes referenced by aliases
            # =========================================================================
            
            needed_subframes = self._get_subframes_for_aliases(to_materialize)
            for sf_name in needed_subframes:
                if sf_name in self._subframe_readers and not self._subframe_loaded.get(sf_name, False):
                    if verbose:
                        print(f"[materialize_aliases] Loading lazy subframe: {sf_name}")
                    self.ensure_subframe(sf_name)
            
            # =========================================================================
            # PHASE 9e: TRY ARROW ZERO-COPY PIPELINE FIRST
            # Converts data to Arrow ONCE, executes all ops, converts back ONCE.
            # Falls back to standard path if any expression is unsupported.
            # =========================================================================
            
            n_rows = len(self.df)
            arrow_pipeline_succeeded = False
            results = {}  # Will be populated by either path
            added = []
            
            if (self._use_arrow and ARROW_COMPUTE_AVAILABLE and n_rows >= NUMBA_MIN_ROWS):
                # Try Arrow pipeline - it handles subframe detection internally
                try:
                    arrow_results = self._materialize_aliases_arrow(to_materialize, verbose=verbose)
                    if arrow_results is not None and arrow_results:
                        # Arrow pipeline succeeded
                        for name, arr in arrow_results.items():
                            # Apply dtype if specified
                            result_dtype = self.alias_dtypes.get(name)
                            if result_dtype is not None:
                                arr = arr.astype(result_dtype)
                            results[name] = arr
                            added.append(name)
                        arrow_pipeline_succeeded = True
                        if verbose:
                            print(f"[materialize_aliases] Arrow pipeline: {len(added)} aliases computed")
                except Exception as e:
                    if not hasattr(self, '_arrow_pipeline_warned'):
                        warnings.warn(
                            f"Arrow pipeline failed, using standard path: {e}",
                            RuntimeWarning
                        )
                        self._arrow_pipeline_warned = True
            
            # =========================================================================
            # STANDARD PATH: Python eval() with batch optimization
            # Used when Arrow pipeline is disabled, unsupported, or failed
            # =========================================================================
            
            if not arrow_pipeline_succeeded:
                # results and added already initialized above
                
                for name in to_materialize:
                    # Skip if already a column
                    if name in self.df.columns:
                        continue
                    
                    # Skip if not an alias
                    if name not in self.aliases:
                        continue
                    
                    expr = self.aliases[name]
                    
                    if verbose:
                        print(f"[materialize_aliases] Computing: {name}")
                    
                    # Handle subframe dependencies: index columns and subframe attributes
                    tokens = re.findall(r'(\w+)\.(\w+)', expr)
                    for sf_name, sf_attr in tokens:
                        sf = self.get_subframe(sf_name)
                        if sf:
                            # Materialize index columns if they're aliases
                            entry = self._subframes.get_entry(sf_name)
                            if entry:
                                index_cols = entry['index']
                                if isinstance(index_cols, str):
                                    index_cols = [index_cols]
                                for idx_col in index_cols:
                                    if idx_col in self.aliases and idx_col not in self.df.columns:
                                        if idx_col not in results:  # Not yet computed in batch
                                            if verbose:
                                                print(f"[materialize_aliases]   Materializing index: {idx_col}")
                                            self.materialize_alias(idx_col)
                            
                            # Materialize subframe attribute if it's an alias
                            if sf_attr in sf.aliases and sf_attr not in sf.df.columns:
                                sf.materialize_alias(sf_attr)
                    
                    # Compute with context_override so dependent aliases can see prior results
                    result = self._eval_in_namespace(expr, context_override=results, alias_name=name)
                    
                    # Apply fill_value for inf/NaN replacement (must be before dtype cast)
                    alias_spec = self._schema["columns"].get(name, {})
                    fill_val = alias_spec.get("fill_value")
                    if fill_val is not None:
                        result = np.where(np.isfinite(result), result, fill_val)
                    
                    # Apply dtype if specified
                    result_dtype = self.alias_dtypes.get(name)
                    if result_dtype is not None:
                        try:
                            result = result.astype(result_dtype)
                        except AttributeError:
                            result = result_dtype(result)
                    
                    results[name] = result
                    added.append(name)
            
            # BATCH ADD: Single concat instead of per-alias insert
            if results:
                new_cols_df = pd.DataFrame(results, index=self.df.index)
                self.df = pd.concat([self.df, new_cols_df], axis=1)
                if verbose:
                    print(f"[materialize_aliases] Batch-added {len(results)} columns")
                    print(f"[materialize_aliases] Join cache: {self._join_cache_hits} hits, "
                          f"{self._join_cache_misses} misses")
            
            # BATCH DROP: Single drop instead of per-column removal
            if cleanTemporary and with_dependencies:
                targets_set = set(targets)
                
                # 1. Drop intermediate alias dependencies (existing logic)
                cols_to_drop = [c for c in added if c not in targets_set and c in self.df.columns]
                
                # 2. Drop subframe join columns (NEW: fix for subframe temporaries)
                # These have pattern: {col}__{subframe_name}
                if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
                    subframe_names = set(self._subframes.subframes.keys())
                    for col in list(self.df.columns):
                        # Check if column is a subframe join column
                        if '__' in col and col not in targets_set:
                            # Extract suffix after last '__'
                            parts = col.rsplit('__', 1)
                            if len(parts) == 2 and parts[1] in subframe_names:
                                if col not in cols_to_drop:
                                    cols_to_drop.append(col)
                
                if cols_to_drop:
                    self.df.drop(columns=cols_to_drop, inplace=True)
                    if verbose:
                        print(f"[materialize_aliases] Batch-dropped {len(cols_to_drop)} columns")
            
            # Emit aggregated missing key warnings
            self._emit_missing_key_summary()
            
            return added
        
        result = self._run_with_profiling(_do_materialize, profile, profile_text, profile_binary)
        
        # Clear join cache after batch (Phase 4)
        self._join_index_cache = {}
        
        return result

    def materialize_pattern(self, pattern, cleanTemporary=True, verbose=False, 
                           only_unmaterialized=True):
        """
        Materialize all aliases matching a regex pattern.
        
        DEPRECATED: Use materialize_aliases(pattern=...) instead.
        
        Parameters
        ----------
        pattern : str
            Regex pattern to match alias names
        cleanTemporary : bool, default=True
            If True, remove intermediate dependencies that weren't targets
        verbose : bool, default=False
            If True, print progress information
        only_unmaterialized : bool, default=True
            If True, skip aliases already materialized as columns
            
        Returns
        -------
        list
            Names of aliases that were materialized
        """
        import warnings
        warnings.warn(
            "materialize_pattern() is deprecated. Use materialize_aliases(pattern=...) instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.materialize_aliases(
            pattern=pattern, 
            cleanTemporary=cleanTemporary, 
            verbose=verbose,
            only_unmaterialized=only_unmaterialized
        )

    def get_alias_series(self, name, dtype=None, warn_missing_keys=True):
        """
        Evaluate an alias expression and return the result as a pandas Series,
        without storing the alias itself as a column in self.df.

        IMPORTANT:
        - Alias *dependencies* may still be materialized as columns
          (same behavior as materialize_alias). This is intentional for
          consistency with the existing alias system.

        Parameters
        ----------
        name : str
            Alias name to evaluate. Must be present in self.aliases.
        dtype : optional
            Optional dtype override. If not provided, alias_dtypes[name]
            is used if available.
        warn_missing_keys : bool, default=True
            If True, emit warning when subframe join has missing keys.
            Missing keys produce NaN (rows are never dropped).

        Returns
        -------
        pandas.Series
            Series aligned with self.df.index.

        Raises
        ------
        KeyError
            If the alias is not defined.
        ValueError
            If the evaluated result has incompatible length.
        TypeError
            If dtype conversion fails or evaluation returns unsupported type.

        Examples
        --------
        >>> aDF.add_alias("isOK", "(row < 152) & (abs(dy) < 10)", dtype=bool)
        >>> mask = aDF.get_alias_series("isOK")  # Returns Series, doesn't add column
        >>> df_filtered = aDF.df[mask]
        """
        if name not in self.aliases:
            raise KeyError(f"Alias '{name}' is not defined.")

        # Ensure dependencies are materialized (side effect by design, consistent with materialize_alias)
        expr = self.aliases[name]
        tokens = re.findall(r'\b\w+\b|\w+\.\w+', expr)
        for token in tokens:
            if '.' in token:
                sf_name, sf_attr = token.split('.', 1)
                sf = self.get_subframe(sf_name)
                if sf and sf_attr in sf.aliases and sf_attr not in sf.df.columns:
                    sf.materialize_alias(sf_attr, warn_missing_keys=warn_missing_keys)
            elif token in self.aliases and token not in self.df.columns and token != name:
                self.materialize_alias(token, warn_missing_keys=warn_missing_keys)

        # Evaluate the alias expression
        result = self._eval_in_namespace(expr, warn_missing_keys=warn_missing_keys, alias_name=name)
        n_rows = len(self.df)

        # Normalize result to a Series aligned with self.df.index
        if isinstance(result, pd.Series):
            # Already a Series, use as-is (should be aligned from _eval_in_namespace)
            series = result
        elif isinstance(result, pd.DataFrame):
            # DataFrames are not valid for aliases
            raise TypeError(
                f"Alias '{name}' evaluated to a DataFrame; "
                "aliases must be 1D (Series/array/scalar)."
            )
        elif np.isscalar(result):
            # Broadcast scalar to full length
            series = pd.Series([result] * n_rows, index=self.df.index)
        else:
            # Assume array-like
            try:
                length = len(result)
            except TypeError as exc:
                # Non-scalar, non-sequence → unsupported
                raise TypeError(
                    f"Alias '{name}' evaluated to unsupported type "
                    f"{type(result).__name__}"
                ) from exc

            if length != n_rows:
                raise ValueError(
                    f"Alias '{name}' evaluated to {length} values, "
                    f"but DataFrame has {n_rows} rows."
                )
            series = pd.Series(result, index=self.df.index)

        # Apply dtype if requested or known
        target_dtype = dtype or self.alias_dtypes.get(name)
        if target_dtype is not None:
            try:
                if hasattr(series, "astype"):
                    series = series.astype(target_dtype)
                else:
                    series = target_dtype(series)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"Cannot convert alias '{name}' result to dtype "
                    f"{target_dtype}: {exc}"
                ) from exc

        return series

    def get_alias_array(self, name, dtype=None, warn_missing_keys=True):
        """
        Evaluate an alias and return its values as a NumPy array.

        This is particularly useful for selection aliases, e.g.:

            aDF.add_alias("isOK", "row < 152", dtype=bool)
            mask = aDF.get_alias_array("isOK")
            df_sel = aDF.df[mask]

        Behavior is identical to get_alias_series(), except for the
        return type (numpy array instead of pandas Series).

        Parameters
        ----------
        name : str
            Alias name to evaluate.
        dtype : optional
            Optional dtype override.
        warn_missing_keys : bool, default=True
            If True, emit warning when subframe join has missing keys.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> aDF.add_alias("isOK", "(row < 152) & (abs(dy) < 10)", dtype=bool)
        >>> mask = aDF.get_alias_array("isOK")  # Returns array, doesn't add column
        >>> df_filtered = aDF.df[mask]
        """
        series = self.get_alias_series(name, dtype=dtype, warn_missing_keys=warn_missing_keys)
        # to_numpy is preferred; fallback to np.asarray for safety
        if hasattr(series, "to_numpy"):
            return series.to_numpy()
        return np.asarray(series)

    def materialize_all(self):
        self._check_for_cycles()
        for name in self.aliases:
            self.materialize_alias(name)

    def save(self, path_prefix, dropAliasColumns=True, include_subframes=True):
        """
        Save AliasDataFrame to Parquet format with full schema metadata.
        
        Parameters
        ----------
        path_prefix : str
            Base path for output files (without .parquet extension)
        dropAliasColumns : bool, default=True
            If True, exclude alias columns from saved data (they can be recomputed)
        include_subframes : bool, default=True
            If True, save subframes as separate .parquet files
            
        Notes
        -----
        - Main frame saved as {path_prefix}.parquet
        - Subframes saved as {path_prefix}__subframe__{name}.parquet
        - Full schema metadata stored under SCHEMA_METADATA_KEY
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        if dropAliasColumns:
            cols = [c for c in self.df.columns if c not in self.aliases]
        else:
            cols = list(self.df.columns)
        
        # Capture column dtypes BEFORE any casting (for restoration on load)
        column_dtypes = {col: str(self.df[col].dtype) for col in self.df.columns}
        
        # PyArrow/Parquet does not support numpy.float16 (halffloat)
        # Auto-cast to float32 on export. The column_dtypes metadata ensures
        # correct restoration to float16 on load.
        export_df = self.df[cols].copy()
        for col in export_df.columns:
            if export_df[col].dtype == np.float16:
                export_df[col] = export_df[col].astype(np.float32)
        
        table = pa.Table.from_pandas(export_df)
        
        # Serialize schema with column_dtypes for dtype restoration
        serialized_schema = _serialize_schema(self._schema)
        serialized_schema["column_dtypes"] = column_dtypes
        
        # Store as single JSON blob under dedicated key
        metadata = {
            SCHEMA_METADATA_KEY: json.dumps(serialized_schema)
        }
        
        existing_meta = table.schema.metadata or {}
        combined_meta = existing_meta.copy()
        combined_meta.update({k.encode(): v.encode() for k, v in metadata.items()})
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(table, f"{path_prefix}.parquet", compression="zstd")
        
        # Save subframes recursively
        if include_subframes:
            for sf_name, entry in self._subframes.items():
                sf = entry["frame"]
                sf.save(f"{path_prefix}__subframe__{sf_name}", 
                       dropAliasColumns=dropAliasColumns, 
                       include_subframes=True)

    @staticmethod
    def load(path_prefix, load_subframes=True):
        """
        Load AliasDataFrame from Parquet format with schema restoration.
        
        Parameters
        ----------
        path_prefix : str
            Base path for input files (without .parquet extension)
        load_subframes : bool, default=True
            If True, load subframes from separate .parquet files
            
        Returns
        -------
        AliasDataFrame
            Restored AliasDataFrame with aliases, compression, and subframes
            
        Raises
        ------
        IOError
            If file cannot be read
        ValueError
            If schema metadata is corrupted
        """
        import pyarrow.parquet as pq
        import os
        
        parquet_path = f"{path_prefix}.parquet"
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        adf = AliasDataFrame(df)
        
        meta = table.schema.metadata or {}
        
        # Try new unified schema format first
        if SCHEMA_METADATA_KEY.encode() in meta:
            try:
                serialized = json.loads(meta[SCHEMA_METADATA_KEY.encode()].decode())
                restored_schema = _deserialize_schema(serialized)
                adf._restore_schema(restored_schema)
                
                # Restore column dtypes
                column_dtypes = serialized.get("column_dtypes", {})
                for col, dtype_str in column_dtypes.items():
                    if col in adf.df.columns:
                        try:
                            target_dtype = np.dtype(dtype_str)
                            if adf.df[col].dtype != target_dtype:
                                adf.df[col] = adf.df[col].astype(target_dtype)
                        except (TypeError, ValueError):
                            pass  # Skip if dtype conversion fails
                            
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Corrupted schema metadata in {parquet_path}: {e}\n"
                    f"The file may be damaged or created by an incompatible version."
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to restore schema from {parquet_path}: {e}"
                )
        
        # Fallback: try legacy format for backward compatibility
        elif b"aliases" in meta:
            try:
                aliases_dict = json.loads(meta[b"aliases"].decode())
                adf._restore_aliases_from_dict(aliases_dict)
                
                if b"dtypes" in meta:
                    dtypes_dict = {k: getattr(np, v) for k, v in json.loads(meta[b"dtypes"].decode()).items()}
                    adf._restore_alias_dtypes_from_dict(dtypes_dict)
                
                if b"constants" in meta:
                    constants_list = json.loads(meta[b"constants"].decode())
                    adf._restore_constant_aliases(constants_list)
                
                if b"compression_info" in meta:
                    compression_dict = json.loads(meta[b"compression_info"].decode())
                    adf._restore_compression_info(compression_dict)
                    
            except json.JSONDecodeError as e:
                warnings.warn(
                    f"Failed to parse legacy metadata in {parquet_path}: {e}. "
                    f"Using defaults."
                )
        
        # Load subframes
        if load_subframes:
            # Get subframe names from schema
            subframe_names = list(adf._schema.get("subframes", {}).keys())
            
            # Also check for subframe files that match pattern
            base_dir = os.path.dirname(path_prefix) or "."
            base_name = os.path.basename(path_prefix)
            
            for sf_name in subframe_names:
                sf_path = f"{path_prefix}__subframe__{sf_name}"
                if os.path.exists(f"{sf_path}.parquet"):
                    try:
                        sf = AliasDataFrame.load(sf_path, load_subframes=True)
                        index_columns = adf._schema["subframes"][sf_name].get("index")
                        if index_columns:
                            adf.register_subframe(sf_name, sf, index_columns)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to load subframe '{sf_name}' from {sf_path}: {e}"
                        )
        
        return adf

    def export_tree(self, filename_or_file, treename="tree", dropAliasColumns=True, compression=uproot.ZLIB(level=1), columns=None):
        """
        Export DataFrame to ROOT TTree.
        
        Parameters
        ----------
        filename_or_file : str or uproot file
            Output file path or open uproot file
        treename : str
            Name of output tree
        dropAliasColumns : bool
            If True, don't export columns that are aliases
        compression : uproot compression
            Compression algorithm (default: ZLIB level 1)
        columns : list of str, optional
            If provided, export only these columns (snapshot/cache mode).
            WARNING: Schema, aliases, and subframes are NOT exported.
        """
        import warnings
        
        # Snapshot mode: export only specified columns, no schema/subframes
        if columns is not None:
            missing = set(columns) - set(self.df.columns)
            if missing:
                raise ValueError(f"Requested columns not found: {sorted(missing)}")
            if self._subframes.subframes:
                warnings.warn(
                    "export_tree(columns=...) does not export subframes. "
                    "Only specified columns will be saved.",
                    UserWarning
                )
            dtype_casts = {col: np.float32 for col in columns if self.df[col].dtype == np.float16}
            export_df = self.df[columns].astype(dtype_casts)
            with uproot.recreate(filename_or_file, compression=compression) as f:
                f[treename] = {col: export_df[col].values for col in export_df.columns}
            return
        
        # Full export mode: existing behavior
        is_path = isinstance(filename_or_file, str)

        if is_path:
            with uproot.recreate(filename_or_file,compression=compression) as f:
                self._write_to_uproot(f, treename, dropAliasColumns)
            self._write_metadata_to_root(filename_or_file, treename)
        else:
            self._write_to_uproot(filename_or_file, treename, dropAliasColumns)
        for subframe_name, entry in self._subframes.items():
            entry["frame"]._write_metadata_to_root(filename_or_file, f"{treename}__subframe__{subframe_name}")

    def _write_to_uproot(self, uproot_file, treename, dropAliasColumns):
        export_cols = [col for col in self.df.columns if not dropAliasColumns or col not in self.aliases]
        dtype_casts = {col: np.float32 for col in export_cols if self.df[col].dtype == np.float16}
        export_df = self.df[export_cols].astype(dtype_casts)

        #uproot_file[treename] = export_df
        uproot_file[treename] = {col: export_df[col].values for col in export_df.columns}
        for subframe_name, entry in self._subframes.items():
            entry["frame"].export_tree(uproot_file, f"{treename}__subframe__{subframe_name}", dropAliasColumns)

    def _write_metadata_to_root(self, filename, treename):
        """
        Write schema metadata to ROOT file.
        
        Phase 4b: Uses unified schema serialization format.
        Also sets TTree aliases for ROOT TTree::Draw compatibility.
        """
        f = ROOT.TFile.Open(filename, "UPDATE")
        tree = f.Get(treename)
        
        # Set TTree aliases for ROOT compatibility
        for alias, expr in self.aliases.items():
            try:
                val = float(expr)
                expr_str = f"({val}+0)"
            except Exception:
                expr_str = convert_expr_to_root(expr)
            tree.SetAlias(alias, expr_str)
        
        # Capture all column dtypes for restoration
        column_dtypes = {
            col: str(self.df[col].dtype)
            for col in self.df.columns
        }
        
        # Phase 4b: Serialize full schema
        serialized_schema = _serialize_schema(self._schema)
        serialized_schema["column_dtypes"] = column_dtypes
        
        # Also include legacy fields for backward compatibility with older readers
        # and ROOT macro compatibility
        metadata = {
            # New unified schema format
            SCHEMA_METADATA_KEY: serialized_schema,
            # Legacy fields for backward compatibility
            "aliases": self.aliases,
            "subframe_indices": {k: v["index"] for k, v in self._subframes.items()},
            "dtypes": {k: v.__name__ if hasattr(v, '__name__') else str(v) 
                      for k, v in self.alias_dtypes.items()},
            "constants": list(self.constant_aliases),
            "subframes": list(self._subframes.subframes.keys()),
            "compression_info": self.compression_info,
            "column_dtypes": column_dtypes
        }
        
        jmeta = json.dumps(metadata)
        tree.GetUserInfo().Add(ROOT.TObjString(jmeta))
        tree.Write("", ROOT.TObject.kOverwrite)
        f.Close()

    @staticmethod
    def read_tree(filename, treename="tree", entry_start=None, entry_stop=None, 
                  num_workers=8, load_subframes=True):
        """
        Read AliasDataFrame from ROOT TTree with optimized memory and speed.

        Uses threaded branch-by-branch reading for optimal performance:
        - ~60x faster than previous implementation
        - ~75% less peak memory
        - ~22% smaller final DataFrame (with dtype conversion)

        Parameters
        ----------
        filename : str
            Path to ROOT file
        treename : str, optional
            Name of TTree (default: "tree")
        entry_start : int, optional
            First entry to read (default: None = 0)
        entry_stop : int, optional
            Last entry to read, exclusive (default: None = all entries)
        num_workers : int, optional
            Number of worker threads for parallel branch reading (default: 8).
            Set to 1 for single-threaded reading.
        load_subframes : bool, optional
            If True (default), automatically load and register subframes defined
            in schema. Tries both Python naming ({treename}__subframe__{name})
            and C++ naming ({name}) conventions.

        Returns
        -------
        AliasDataFrame
            Loaded AliasDataFrame with aliases, subframes, and compression info restored

        Notes
        -----
        - Uses branch-by-branch reading for ~75% less peak memory
        - Threading provides ~6x speedup with no memory penalty
        - entry_start/entry_stop apply only to main tree, not subframes
        - Subframes are always fully loaded (they contain small calibration data)
        - Backward compatible with files created by older versions

        Examples
        --------
        >>> # Read full file with default threading (8 workers)
        >>> adf = AliasDataFrame.read_tree("data.root", "tree")

        >>> # Read first 1M entries for testing
        >>> adf = AliasDataFrame.read_tree("data.root", "tree", entry_stop=1_000_000)

        >>> # Single-threaded (for environments with threading issues)
        >>> adf = AliasDataFrame.read_tree("data.root", "tree", num_workers=1)
        
        >>> # Skip subframe loading (faster, for main tree only)
        >>> adf = AliasDataFrame.read_tree("data.root", "tree", load_subframes=False)
        """
        import warnings
        import concurrent.futures

        # =========================================================================
        # Step 1: Read metadata from ROOT file first
        # =========================================================================
        metadata = {
            'aliases': {},
            'alias_dtypes': {},
            'constant_aliases': set(),
            'compression_info': {},
            'subframes': [],
            'subframe_indices': {},
            'column_dtypes': {}  # Phase 2: all column dtypes
        }

        f_root = ROOT.TFile.Open(filename)
        if not f_root or f_root.IsZombie():
            raise IOError(f"Cannot open ROOT file: {filename}")

        try:
            tree = f_root.Get(treename)
            if not tree:
                available_keys = [k.GetName() for k in f_root.GetListOfKeys()]
                raise ValueError(
                    f"Tree '{treename}' not found in {filename}. "
                    f"Available: {available_keys}"
                )

            # Read aliases from TTree alias list
            alias_list = tree.GetListOfAliases()
            if alias_list:
                for alias in alias_list:
                    metadata['aliases'][alias.GetName()] = alias.GetTitle()

            # Read extended metadata from TObjString in UserInfo
            user_info = tree.GetUserInfo()
            new_schema_found = False
            
            for i in range(user_info.GetEntries()):
                obj = user_info.At(i)
                if isinstance(obj, ROOT.TObjString):
                    try:
                        jmeta = json.loads(obj.GetString().Data())
                        
                        # Phase 4b: Check for new unified schema format first
                        if SCHEMA_METADATA_KEY in jmeta:
                            serialized_schema = jmeta[SCHEMA_METADATA_KEY]
                            metadata['restored_schema'] = _deserialize_schema(serialized_schema)
                            metadata['column_dtypes'] = serialized_schema.get("column_dtypes", {})
                            metadata['subframes'] = list(metadata['restored_schema'].get("subframes", {}).keys())
                            metadata['subframe_indices'] = {
                                k: v.get("index") 
                                for k, v in metadata['restored_schema'].get("subframes", {}).items()
                            }
                            new_schema_found = True
                        
                        # Also read legacy fields for backward compatibility / fallback
                        metadata['aliases'].update(jmeta.get("aliases", {}))
                        metadata['alias_dtypes'] = {
                            k: np.dtype(v).type
                            for k, v in jmeta.get("dtypes", {}).items()
                        }
                        metadata['constant_aliases'] = set(jmeta.get("constants", []))
                        metadata['compression_info'] = jmeta.get("compression_info", {})
                        if not new_schema_found:
                            metadata['subframes'] = jmeta.get("subframes", [])
                            metadata['subframe_indices'] = jmeta.get("subframe_indices", {})
                        if 'column_dtypes' not in metadata or not metadata['column_dtypes']:
                            metadata['column_dtypes'] = jmeta.get("column_dtypes", {})
                        break

                    except json.JSONDecodeError as e:
                        warnings.warn(
                            f"Failed to parse metadata JSON in {filename}: {e}. "
                            f"Using defaults."
                        )
                    except Exception as e:
                        warnings.warn(
                            f"Error reading metadata from {filename}: {e}. "
                            f"Using defaults."
                        )
        finally:
            f_root.Close()

        # Repair corrupted subframe_indices from legacy format
        for sf_name, idx_cols in metadata.get('subframe_indices', {}).items():
            metadata['subframe_indices'][sf_name] = _repair_index_columns(idx_cols, sf_name)

        # Ensure __meta__ exists in compression_info
        if "__meta__" not in metadata['compression_info']:
            metadata['compression_info']["__meta__"] = {
                "schema_version": 1,
                "state_machine": "CompressionState.v1"
            }

        # =========================================================================
        # Step 2: Build dtype hints from compression_info
        # =========================================================================
        dtype_hints = {}

        for col_name, info in metadata['compression_info'].items():
            if col_name == "__meta__":
                continue

            compressed_col = info.get('compressed_col')
            compressed_dtype_str = info.get('compressed_dtype')

            if compressed_col and compressed_dtype_str:
                try:
                    dtype_hints[compressed_col] = np.dtype(compressed_dtype_str)
                except TypeError:
                    warnings.warn(
                        f"Unknown dtype '{compressed_dtype_str}' for column '{compressed_col}'. "
                        f"Using default."
                    )

        # =========================================================================
        # Step 2b: Add column_dtypes from metadata (Phase 2)
        # Priority: compression_info > column_dtypes
        # =========================================================================
        if 'column_dtypes' in metadata:
            for col, dtype_str in metadata['column_dtypes'].items():
                if col not in dtype_hints:  # Don't override compression_info
                    try:
                        dtype_hints[col] = np.dtype(dtype_str)
                    except TypeError:
                        warnings.warn(
                            f"Unknown dtype '{dtype_str}' for column '{col}'. "
                            f"Using default."
                        )

        # =========================================================================
        # Step 3: Read branches with uproot (branch-by-branch for memory efficiency)
        # =========================================================================
        with uproot.open(filename) as f:
            tree = f[treename]
            branch_names = list(tree.keys())

            if not branch_names:
                df = pd.DataFrame()

            elif num_workers > 1:
                # Threaded branch-by-branch reading
                def read_branch(branch_name):
                    try:
                        arr = tree[branch_name].array(
                            library="np",
                            entry_start=entry_start,
                            entry_stop=entry_stop
                        )

                        if branch_name in dtype_hints:
                            target_dtype = dtype_hints[branch_name]
                            if arr.dtype != target_dtype:
                                arr = arr.astype(target_dtype)

                        return branch_name, arr

                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to read branch '{branch_name}' from {filename}: {e}"
                        ) from e

                arrays = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(read_branch, name): name
                        for name in branch_names
                    }

                    for future in concurrent.futures.as_completed(futures):
                        branch_name = futures[future]
                        try:
                            name, arr = future.result()
                            arrays[name] = arr
                        except Exception as e:
                            raise RuntimeError(
                                f"Error reading branch '{branch_name}': {e}"
                            ) from e

                df = pd.DataFrame({name: arrays[name] for name in branch_names})

            else:
                # Single-threaded branch-by-branch reading
                arrays = {}
                for branch_name in branch_names:
                    try:
                        arr = tree[branch_name].array(
                            library="np",
                            entry_start=entry_start,
                            entry_stop=entry_stop
                        )

                        if branch_name in dtype_hints:
                            target_dtype = dtype_hints[branch_name]
                            if arr.dtype != target_dtype:
                                arr = arr.astype(target_dtype)

                        arrays[branch_name] = arr

                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to read branch '{branch_name}' from {filename}: {e}"
                        ) from e

                df = pd.DataFrame(arrays)

        # =========================================================================
        # Step 4: Create AliasDataFrame and populate metadata
        # =========================================================================
        adf = AliasDataFrame(df)
        
        # Phase 4b: Use unified schema if available, otherwise legacy restore
        if 'restored_schema' in metadata:
            adf._restore_schema(metadata['restored_schema'])
        else:
            # Legacy restore for backward compatibility
            adf._restore_aliases_from_dict(metadata['aliases'])
            adf._restore_alias_dtypes_from_dict(metadata['alias_dtypes'])
            adf._restore_constant_aliases(list(metadata['constant_aliases']))
            adf._restore_compression_info(metadata['compression_info'])

        # =========================================================================
        # Step 5: Load subframes recursively
        # =========================================================================
        if load_subframes and metadata['subframes']:
            # Warn if entry_range used with subframes
            if entry_start is not None or entry_stop is not None:
                warnings.warn(
                    f"entry_start/entry_stop apply only to main tree '{treename}'. "
                    f"Subframes {metadata['subframes']} will be fully loaded."
                )

            for sf_name in metadata['subframes']:
                try:
                    # Try both naming conventions:
                    # 1. Python convention: {treename}__subframe__{sf_name}
                    # 2. C++/direct convention: {sf_name}
                    sf = None
                    tree_names_to_try = [
                        f"{treename}__subframe__{sf_name}",  # Python export convention
                        sf_name                              # C++/direct tree name
                    ]
                    
                    last_error = None
                    for sf_treename in tree_names_to_try:
                        try:
                            # load_subframes=False prevents recursive subframe loading
                            sf = AliasDataFrame.read_tree(
                                filename,
                                treename=sf_treename,
                                num_workers=num_workers,
                                load_subframes=False
                            )
                            break  # Found it!
                        except (ValueError, KeyError) as e:
                            last_error = e
                            continue  # Try next naming convention
                    
                    if sf is None:
                        raise ValueError(
                            f"Subframe tree not found. Tried: {tree_names_to_try}. "
                            f"Last error: {last_error}"
                        )

                    index_columns = metadata['subframe_indices'].get(sf_name)
                    if index_columns is None:
                        raise ValueError(
                            f"Missing index_columns for subframe '{sf_name}' in metadata. "
                            f"Available indices: {list(metadata['subframe_indices'].keys())}"
                        )

                    adf.register_subframe(sf_name, sf, index_columns=index_columns)

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load subframe '{sf_name}' from {filename}: {e}"
                    ) from e

        return adf
    
    # =========================================================================
    # SECTION 3b: Lazy Branch Loading (Phase 7.1)
    # =========================================================================
    #
    # On-demand branch loading from ROOT files. Load only branches needed
    # for current query, not all 100+ branches.
    #
    # Key methods:
    # - read_tree_lazy(): Create ADF with lazy loading
    # - ensure_branches(): Load specific branches on demand
    # - available_branches: All branches in TTree
    # - loaded_branches: Currently loaded branches
    #
    # =========================================================================
    
    @staticmethod
    def read_tree_lazy(file_path: str,
                       tree_name: str,
                       branches=None,
                       schema=None):
        """
        Create AliasDataFrame with lazy branch loading.
        
        Only specified branches are loaded initially. Additional branches
        can be loaded on demand via ensure_branches() or auto-loaded on
        column access (adf['x']).
        
        Parameters
        ----------
        file_path : str
            Path to ROOT file
        tree_name : str
            Name of TTree
        branches : List[str], optional
            Initial branches to load. If None, loads metadata only.
        schema : dict, optional
            Schema dict to apply
            
        Returns
        -------
        AliasDataFrame
            ADF in lazy mode
            
        Examples
        --------
        >>> # Load specific branches
        >>> adf = AliasDataFrame.read_tree_lazy(
        ...     'data.root', 'tree',
        ...     branches=['x', 'y', 'pt']
        ... )
        
        >>> # Load metadata only, branches later
        >>> adf = AliasDataFrame.read_tree_lazy('data.root', 'tree')
        >>> print(adf.available_branches)  # All branches in TTree
        >>> adf.ensure_branches(['x', 'y'])  # Load on demand
        
        >>> # Auto-load on access
        >>> adf = AliasDataFrame.read_tree_lazy('data.root', 'tree')
        >>> adf['x']  # Auto-loads 'x' branch
        """
        from LazyTreeReader import LazyTreeReader
        
        # Create lazy reader (loads metadata immediately)
        lazy_reader = LazyTreeReader(file_path, tree_name)
        
        # Create initial DataFrame
        if branches:
            # Load requested branches
            df = lazy_reader.ensure_branches(branches, pd.DataFrame())
        else:
            # Empty DataFrame with correct length (metadata only)
            df = pd.DataFrame(index=range(lazy_reader.num_entries))
        
        # Create AliasDataFrame
        adf = AliasDataFrame(df)
        
        # Apply schema if provided
        if schema:
            adf.update_schema(schema)
        
        # Attach lazy reader
        adf._lazy_reader = lazy_reader
        
        # Store chain config (single file for now, Phase 7.4 adds multi-file)
        adf._chain = {
            'files': [{
                'path': file_path, 
                'tree': tree_name, 
                'entries': lazy_reader.num_entries
            }],
            'entry_offsets': [0],
            'total_entries': lazy_reader.num_entries,
            'validation_mode': None
        }
        
        return adf
    
    def _merge_loaded_data(self, existing_df: pd.DataFrame, 
                           new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge newly loaded data into existing DataFrame.
        
        This is the central merge point for all data loading operations.
        Keeping merge logic here (not in readers) enables future Arrow
        backend migration with minimal changes.
        
        Parameters
        ----------
        existing_df : pd.DataFrame
            Current DataFrame (may be empty)
        new_data : pd.DataFrame
            Newly loaded data to merge
            
        Returns
        -------
        pd.DataFrame
            Merged DataFrame
        """
        if new_data is None or len(new_data) == 0:
            return existing_df
        
        if existing_df is None or len(existing_df) == 0:
            return new_data
        
        # Verify row alignment
        if len(existing_df) != len(new_data):
            raise ValueError(
                f"Row count mismatch: existing={len(existing_df)}, "
                f"new={len(new_data)}. Cannot merge misaligned data."
            )
        
        # Add new columns to existing DataFrame
        for col in new_data.columns:
            if col not in existing_df.columns:
                existing_df[col] = new_data[col].values
        
        return existing_df
    
    def ensure_branches(self, names):
        """
        Ensure specified branches are loaded into DataFrame.
        
        For lazy-loaded ADFs, loads branches from file(s) on demand.
        For eager ADFs, verifies branches exist.
        
        Parameters
        ----------
        names : str or List[str]
            Branch name(s) to ensure are loaded
            
        Raises
        ------
        BranchNotFoundError
            If requested branches don't exist
            
        Examples
        --------
        >>> adf.ensure_branches(['eta', 'phi'])
        >>> print('eta' in adf.df.columns)  # True
        """
        if isinstance(names, str):
            names = [names]
        
        if not names:
            return
        
        if self._lazy_reader is None:
            # Eager mode - just verify columns exist
            missing = set(names) - set(self.df.columns)
            if missing:
                raise BranchNotFoundError(missing, set(self.df.columns))
            return
        
        # Lazy mode - load from reader
        names_set = set(names)
        already_loaded = self._lazy_reader.loaded_branches
        to_load = names_set - already_loaded
        
        if not to_load:
            return  # All requested branches already loaded
        
        # Reader returns new data only (doesn't merge)
        new_data = self._lazy_reader.load_branches(list(to_load))
        
        # ADF handles merge (Arrow-compatible pattern)
        self.df = self._merge_loaded_data(self.df, new_data)
    
    @property
    def available_branches(self):
        """
        Get all branches available in TTree (lazy mode only).
        
        Returns
        -------
        Set[str] or None
            Set of branch names, or None if not in lazy mode
        """
        if self._lazy_reader is None:
            return None
        return self._lazy_reader.available_branches
    
    @property
    def loaded_branches(self):
        """
        Get currently loaded branches (lazy mode only).
        
        Returns
        -------
        Set[str] or None
            Set of loaded branch names, or None if not in lazy mode
        """
        if self._lazy_reader is None:
            return None
        return self._lazy_reader.loaded_branches
    
    @property
    def is_lazy(self):
        """
        Check if ADF is in lazy loading mode.
        
        Returns
        -------
        bool
            True if lazy mode, False if eager mode
        """
        return self._lazy_reader is not None
    
    # =========================================================================
    # SECTION 3b2: Chain Mode - Multiple Files (Phase 7.4)
    # =========================================================================
    #
    # Support for reading multiple ROOT files as a single dataset.
    # Uses LazyChainReader with LRU file handle caching.
    #
    # Key methods:
    # - read_chain(): Eager loading of multiple files
    # - read_chain_lazy(): Lazy loading of multiple files
    # - _parse_chain_files(): Parse file specifications
    #
    # =========================================================================
    
    @classmethod
    def read_chain_lazy(cls,
                        files: Union[str, List[str]],
                        tree_name: str = None,
                        branches: List[str] = None,
                        schema: dict = None,
                        validate_branches: str = 'first',
                        add_file_index: bool = False,
                        max_open_files: int = 8) -> 'AliasDataFrame':
        """
        Read multiple ROOT files as a single lazy dataset.
        
        Loads only metadata initially. Data loaded on demand via
        ensure_branches() or automatically during draw().
        
        Parameters
        ----------
        files : str or List[str]
            Glob pattern ('data_*.root:tree') or list of 'path:tree' strings
        tree_name : str, optional
            Tree name if not specified in files pattern
        branches : List[str], optional
            Branches to load initially. None = metadata only.
        schema : dict, optional
            Schema to apply
        validate_branches : str, default 'first'
            Validation mode:
            - 'first': Use first file as reference, warn on differences.
              Missing branches in later files are filled with NaN.
              Extra branches in later files are ignored.
            - 'strict': Error if any file differs from first file
            - 'intersection': Only branches present in ALL files
            - 'union': All branches from any file; missing filled with NaN
        add_file_index : bool, default False
            Add '__file_idx__' column tracking source file
        max_open_files : int, default 8
            Max open file handles (LRU cache size)
            
        Returns
        -------
        AliasDataFrame
            Lazy dataset (loads data on demand)
            
        Examples
        --------
        >>> adf = AliasDataFrame.read_chain_lazy('data_*.root:tree')
        >>> print(adf.available_branches)  # See all branches
        >>> adf.draw('y:x')  # Auto-loads x, y
        
        >>> # With validation mode
        >>> adf = AliasDataFrame.read_chain_lazy(
        ...     'data_*.root:tree',
        ...     validate_branches='strict'
        ... )
        """
        from LazyChainReader import LazyChainReader
        
        # Parse file specifications
        file_specs = cls._parse_chain_files(files, tree_name)
        
        if not file_specs:
            raise ValueError(f"No files found matching: {files}")
        
        # Create chain reader
        chain_reader = LazyChainReader(
            files=file_specs,
            validation=validate_branches,
            max_open_files=max_open_files,
            add_file_index=add_file_index
        )
        
        # Create ADF with empty DataFrame
        adf = cls(pd.DataFrame())
        
        # Apply schema if provided
        if schema:
            adf.update_schema(schema)
        
        adf._lazy_reader = chain_reader
        
        # Store chain config (not serialized with schema)
        adf._chain = {
            'files': file_specs,
            'entry_offsets': chain_reader.entry_offsets,
            'total_entries': chain_reader.entries,
            'validation_mode': validate_branches
        }
        
        # Load initial branches if requested
        if branches:
            adf.ensure_branches(branches)
        
        return adf
    
    @classmethod
    def read_chain(cls,
                   files: Union[str, List[str]],
                   tree_name: str = None,
                   branches: List[str] = None,
                   schema: dict = None,
                   validate_branches: str = 'first',
                   add_file_index: bool = False,
                   max_open_files: int = 8) -> 'AliasDataFrame':
        """
        Read multiple ROOT files as a single dataset (eager loading).
        
        Parameters
        ----------
        files : str or List[str]
            Glob pattern ('data_*.root:tree') or list of 'path:tree' strings
        tree_name : str, optional
            Tree name if not specified in files pattern
        branches : List[str], optional
            Branches to load. None = all branches.
        schema : dict, optional
            Schema to apply
        validate_branches : str, default 'first'
            Validation mode: 'first', 'strict', 'intersection', 'union'
        add_file_index : bool, default False
            Add '__file_idx__' column
        max_open_files : int, default 8
            Max open file handles
            
        Returns
        -------
        AliasDataFrame
            Combined dataset with all data loaded
            
        Examples
        --------
        >>> adf = AliasDataFrame.read_chain('data_*.root:tree')
        >>> adf = AliasDataFrame.read_chain(['f1.root:T', 'f2.root:T'])
        """
        # Create lazy, then load all
        adf = cls.read_chain_lazy(
            files=files,
            tree_name=tree_name,
            branches=branches,
            schema=schema,
            validate_branches=validate_branches,
            add_file_index=add_file_index,
            max_open_files=max_open_files
        )
        
        # Load all requested branches (or all available)
        if branches is None:
            branches = list(adf.available_branches)
        adf.ensure_branches(branches)
        
        return adf
    
    @staticmethod
    def _parse_chain_files(files: Union[str, List[str]], 
                           tree_name: str = None) -> List[dict]:
        """
        Parse file specification into list of {path, tree} dicts.
        
        Supports:
        - Glob patterns: 'data_*.root:tree'
        - Single file: 'data.root:tree'
        - List of files: ['f1.root:tree', 'f2.root:tree']
        - List with separate tree: ['f1.root', 'f2.root'], tree_name='tree'
        """
        file_specs = []
        
        if isinstance(files, str):
            # Single string - could be glob pattern
            if ':' in files:
                pattern, tree = files.rsplit(':', 1)
            else:
                pattern = files
                tree = tree_name
            
            if tree is None:
                raise ValueError("Tree name required. Use 'file.root:tree' or tree_name parameter.")
            
            # Expand glob
            matched = sorted(glob.glob(pattern))
            if not matched:
                # Maybe it's not a glob, just a single file
                if Path(pattern).exists():
                    matched = [pattern]
                else:
                    raise FileNotFoundError(f"No files found: {pattern}")
            
            for path in matched:
                file_specs.append({'path': path, 'tree': tree})
        
        else:
            # List of files
            for f in files:
                if ':' in f:
                    path, tree = f.rsplit(':', 1)
                else:
                    path = f
                    tree = tree_name
                
                if tree is None:
                    raise ValueError(f"Tree name required for {f}")
                
                file_specs.append({'path': path, 'tree': tree})
        
        return file_specs
    
    # Chain properties
    
    @property
    def is_chain(self) -> bool:
        """True if this ADF represents a chain of files."""
        return self._chain is not None and len(self._chain.get('files', [])) > 1
    
    @property
    def file_count(self) -> int:
        """Number of files in chain (1 for single file)."""
        if self._chain is None:
            return 1 if self._lazy_reader is not None else 0
        return len(self._chain.get('files', []))
    
    @property  
    def chain_info(self) -> Optional[dict]:
        """
        Chain information (None for non-chain ADFs).
        
        Returns a shallow copy. Do not mutate the nested structures.
        
        Returns
        -------
        dict or None
            'files': list of file specs
            'entry_offsets': cumulative entry counts
            'total_entries': total entries
            'validation_mode': how branches were validated
        """
        return self._chain.copy() if self._chain else None
    
    # Resource management
    
    def close(self):
        """
        Release all resources (file handles, memory).
        
        After calling close(), the ADF should not be used for lazy operations.
        
        MODIFIED FOR 7.5a: Also closes subframe readers.
        """
        # Close main reader
        if self._lazy_reader is not None:
            self._lazy_reader.close()
            self._lazy_reader = None
        
        # Phase 7.5a: Close subframe readers
        for reader in self._subframe_readers.values():
            reader.close()
        self._subframe_readers.clear()
        self._subframe_loaded.clear()
        self._subframe_lazy_config.clear()
        
        self._chain = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
    
    # Memory estimation
    
    def estimate_memory(self, branches: List[str] = None) -> dict:
        """
        Estimate memory for loading branches.
        
        Parameters
        ----------
        branches : List[str], optional
            Branches to estimate. None = all available/loaded.
            
        Returns
        -------
        dict
            'bytes', 'human', 'branches', 'entries', 'warning' keys
            
        Examples
        --------
        >>> adf = AliasDataFrame.read_chain_lazy('data_*.root:tree')
        >>> est = adf.estimate_memory(['pt', 'eta', 'phi'])
        >>> print(est['human'])  # "2.4 GB"
        >>> if est['warning']:
        ...     print(est['warning'])
        """
        if self._lazy_reader is None:
            # Eager mode - calculate from existing DataFrame
            if branches is None:
                branches = list(self.df.columns)
            
            total = sum(
                self.df[col].nbytes for col in branches if col in self.df.columns
            )
            
            return {
                'bytes': total,
                'human': self._format_bytes(total),
                'branches': len(branches),
                'entries': len(self.df),
                'warning': None
            }
        
        # Lazy mode - delegate to reader
        return self._lazy_reader.estimate_memory(branches)
    
    @staticmethod
    def _format_bytes(n: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    # =========================================================================
    # SECTION 3c: Branch Auto-Detection (Phase 7.2)
    # =========================================================================
    #
    # Automatic detection of required branches from expressions, selections,
    # and alias dependencies. Enables draw() integration to auto-load only
    # needed branches.
    #
    # Key methods:
    # - get_required_branches(): Public API for branch detection
    # - _parse_selection_columns(): AST-based selection parsing
    # - _resolve_to_base_branches(): Alias chain resolution
    #
    # =========================================================================
    
    def _parse_selection_columns(self, selection: str) -> set:
        """
        Extract column/variable names from a selection string.
        
        Uses Python AST to accurately parse expressions and extract
        identifiers while filtering out function calls.
        
        Parameters
        ----------
        selection : str
            Selection expression, e.g., 'isOK && pt > 0.5'
            Supports Python syntax: and, or, not, &, |, ~
            Also supports C-style: &&, ||, !
            
        Returns
        -------
        Set[str]
            Set of variable names found in expression
            
        Examples
        --------
        >>> adf._parse_selection_columns('isOK && pt > 0.5')
        {'isOK', 'pt'}
        
        >>> adf._parse_selection_columns('np.abs(eta) < 2.5')
        {'eta'}  # 'np' and 'abs' filtered as module/function
        
        >>> adf._parse_selection_columns('(x > 0) & (y < 10) | isGood')
        {'x', 'y', 'isGood'}
        """
        if not selection or not isinstance(selection, str):
            return set()
        
        # Normalize syntax: && → and, || → or
        normalized = selection.replace('&&', ' and ').replace('||', ' or ')
        # Handle C-style ! but not !=
        normalized = re.sub(r'!(?!=)', ' not ', normalized)
        
        try:
            tree = ast.parse(normalized, mode='eval')
        except SyntaxError:
            # Fallback to regex for unparseable expressions
            return self._parse_selection_columns_regex(selection)
        
        # Collect all Name nodes (identifiers)
        identifiers = set()
        function_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Call):
                # Track function names to exclude
                if isinstance(node.func, ast.Name):
                    function_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # e.g., np.abs → exclude 'np'
                    if isinstance(node.func.value, ast.Name):
                        function_names.add(node.func.value.id)
        
        # Filter out functions, builtins, and common modules
        excluded = function_names | {
            'np', 'numpy', 'pd', 'pandas', 'math',
            'True', 'False', 'None', 'and', 'or', 'not',
            'abs', 'min', 'max', 'sum', 'len', 'round', 'int', 'float',
            'sqrt', 'exp', 'log', 'log10', 'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan', 'arctan2',
            'sinh', 'cosh', 'tanh', 'floor', 'ceil', 'sign',
            'pi', 'e', 'inf', 'nan'
        }
        
        return identifiers - excluded

    def _parse_selection_columns_regex(self, selection: str) -> set:
        """
        Fallback regex-based extraction for unparseable selections.
        
        Parameters
        ----------
        selection : str
            Selection string
            
        Returns
        -------
        Set[str]
            Variable names found via regex
        """
        # Match identifiers: word characters, not starting with digit
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = set(re.findall(pattern, selection))
        
        # Filter out Python keywords and common names
        excluded = {
            'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None',
            'if', 'else', 'for', 'while', 'np', 'numpy', 'pd', 'pandas', 'math',
            'abs', 'min', 'max', 'sum', 'len', 'round', 'int', 'float',
            'sqrt', 'exp', 'log', 'sin', 'cos', 'tan'
        }
        
        return matches - excluded

    def _resolve_to_base_branches(self, columns: set, _visited: set = None) -> set:
        """
        Resolve column names to base branches, expanding alias dependencies.
        
        Recursively traces through alias definitions to find the underlying
        TTree branches needed.
        
        Parameters
        ----------
        columns : Set[str]
            Column names (may include aliases)
        _visited : Set[str], optional
            Internal tracking for circular dependency detection
            
        Returns
        -------
        Set[str]
            Base branch names (no aliases)
            
        Raises
        ------
        ValueError
            If circular alias dependency detected
            
        Examples
        --------
        >>> adf.add_alias('dEdx', 'signal / trackLength')
        >>> adf.add_alias('normalized', 'dEdx / expected')
        >>> adf._resolve_to_base_branches({'normalized', 'pt'})
        {'signal', 'trackLength', 'expected', 'pt'}
        """
        if _visited is None:
            _visited = set()
        
        base_branches = set()
        aliases = self.aliases  # {name: expr}
        
        for col in columns:
            if col in _visited:
                # Circular dependency detected
                cycle_path = ' → '.join(list(_visited) + [col])
                raise ValueError(
                    f"Circular alias dependency detected: {cycle_path}"
                )
            
            if col in aliases:
                # This is an alias - resolve its dependencies
                _visited.add(col)
                expr = aliases[col]
                
                # Parse the alias expression for dependencies
                alias_deps = self._parse_selection_columns(expr)
                
                # Recursively resolve (shared _visited set per GPT tweak)
                resolved = self._resolve_to_base_branches(alias_deps, _visited)
                base_branches.update(resolved)
                
                # Remove from visited after processing (GPT tweak #2)
                _visited.remove(col)
            else:
                # This is a base column/branch
                base_branches.add(col)
        
        return base_branches

    def get_required_branches(self,
                              expr: str = None,
                              selection: str = None,
                              group_by: str = None,
                              color: str = None,
                              aliases: list = None,
                              validate: bool = False) -> set:
        """
        Get base branches required for expression, selection, and parameters.
        
        Parses all inputs to extract column references, then resolves any
        aliases to their underlying branch dependencies.
        
        Parameters
        ----------
        expr : str, optional
            Plot expression, e.g., 'dEdx:p' or 'pt'
        selection : str, optional
            Selection/cut expression, e.g., 'isOK && pt > 0.5'
        group_by : str, optional
            Group-by column name
        color : str, optional
            Color column name  
        aliases : List[str], optional
            Additional alias names to include
        validate : bool, default False
            If True, filter results to only existing branches/columns
            
        Returns
        -------
        Set[str]
            Base branch names needed
            
        Examples
        --------
        >>> adf.add_alias('dEdx', 'signal / trackLength')
        >>> adf.get_required_branches(
        ...     expr='dEdx:p',
        ...     selection='isOK && pt > 0.5',
        ...     group_by='charge'
        ... )
        {'signal', 'trackLength', 'p', 'isOK', 'pt', 'charge'}
        
        >>> # With validation against available branches
        >>> adf.get_required_branches(expr='x:y', validate=True)
        {'x', 'y'}  # Only if x, y exist
        """
        all_columns = set()
        
        # 1. Parse main expression (e.g., 'dEdx:p' → {'dEdx', 'p'})
        #    Reuse logic from _parse_expr_aliases but get ALL columns (GPT tweak #1)
        if expr:
            parts = expr.replace(' ', '').split(':')
            all_columns.update(parts)
        
        # 2. Parse selection string
        if selection:
            selection_cols = self._parse_selection_columns(selection)
            all_columns.update(selection_cols)
        
        # 3. Add group_by and color
        if group_by:
            all_columns.add(group_by)
        if color and isinstance(color, str):
            all_columns.add(color)
        
        # 4. Add explicit aliases
        if aliases:
            all_columns.update(aliases)
        
        # 5. Resolve aliases to base branches
        base_branches = self._resolve_to_base_branches(all_columns)
        
        # 6. Optionally validate against available branches/columns (GPT tweak #3)
        if validate:
            if self._lazy_reader is not None:
                # Lazy mode: check against TTree branches
                available = self._lazy_reader.available_branches
            else:
                # Eager mode: check against df columns and aliases
                available = set(self.df.columns) | set(self.aliases.keys())
            base_branches = base_branches & available
        
        return base_branches
    
    # =========================================================================
    # SECTION 4: Compression Engine
    # =========================================================================
    #
    # Bidirectional column compression to reduce memory and file size while
    # maintaining data accessibility through lazy decompression aliases.
    #
    # Compression State Machine:
    #
    #   ┌──────────────┐
    #   │  (no state)  │
    #   └──────┬───────┘
    #          │ define_compression_schema()
    #          ▼
    #   ┌──────────────┐
    #   │ SCHEMA_ONLY  │ ◄───────────────────────────┐
    #   └──────┬───────┘                             │
    #          │ compress_columns()                  │ decompress(keep_schema=True)
    #          ▼                                     │
    #   ┌──────────────┐                             │
    #   │  COMPRESSED  │ ────────────────────────────┘
    #   └──────┬───────┘
    #          │ decompress_columns()
    #          ▼
    #   ┌──────────────┐
    #   │ DECOMPRESSED │ ──► compress_columns() ──► COMPRESSED
    #   └──────────────┘
    #
    # Key methods:
    # - compress_columns(): Apply compression transform
    # - decompress_columns(): Restore original values
    # - get_compression_state(): Query column state
    # - define_compression_schema(): Pre-define compression without applying
    #
    # =========================================================================

    def get_compression_state(self, column):
        """
        Get the compression state of a column.

        Parameters
        ----------
        column : str
            Column name to check

        Returns
        -------
        str or None
            CompressionState constant if column is tracked, None otherwise
            Returns SCHEMA_ONLY if compression definition exists but no state

        Examples
        --------
        >>> adf.get_compression_state('dy')
        'compressed'
        """
        if column not in self.compression_info or column == "__meta__":
            return None
        
        state = self.compression_info[column].get('state')
        
        # If compression definition exists but no state, treat as SCHEMA_ONLY
        # This happens when loading a definition-only schema (include_state=False)
        if state is None:
            # Verify this is a real compression definition (has required fields)
            info = self.compression_info[column]
            if info.get('compress_expr') or info.get('decompress_expr'):
                return CompressionState.SCHEMA_ONLY
        
        return state

    def is_compressed(self, column):
        """
        Check if a column is currently in compressed state.

        Parameters
        ----------
        column : str
            Column name to check

        Returns
        -------
        bool
            True if column state is COMPRESSED

        Examples
        --------
        >>> adf.is_compressed('dy')
        True
        """
        return self.get_compression_state(column) == CompressionState.COMPRESSED

    def _schema_from_info(self, column):
        """
        Reconstruct compression spec from stored compression_info.

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        dict
            Compression specification with compress/decompress/dtypes

        Raises
        ------
        ValueError
            If column not in compression_info
        """
        if column not in self.compression_info:
            raise ValueError(f"No compression schema found for column '{column}'")

        info = self.compression_info[column]
        return {
            'compress': info['compress_expr'],
            'decompress': info['decompress_expr'],
            'compressed_dtype': getattr(np, info['compressed_dtype']),
            'decompressed_dtype': getattr(np, info['decompressed_dtype'])
        }

    def _schemas_equal(self, schema1, schema2):
        """
        Compare two compression schemas for equality.

        Checks if compress/decompress expressions and dtypes match.

        Parameters
        ----------
        schema1, schema2 : dict
            Compression specifications to compare

        Returns
        -------
        bool
            True if schemas are equivalent
        """
        keys = ['compress', 'decompress', 'compressed_dtype', 'decompressed_dtype']
        for key in keys:
            if key in ('compressed_dtype', 'decompressed_dtype'):
                # Compare dtype names
                dtype1 = np.dtype(schema1[key]).name
                dtype2 = np.dtype(schema2[key]).name
                if dtype1 != dtype2:
                    return False
            else:
                # Compare expressions (strings)
                if schema1.get(key) != schema2.get(key):
                    return False
        return True

    def compress_columns(self, compression_spec=None, columns=None, suffix='_c', drop_original=True,
                         on_missing='warn',       # NEW
                         return_summary=False,     # NEW
                         measure_precision=False):
        """
        Compress columns using bidirectional transforms with state management.

        Supports five modes:
        1. Define schema-only: columns=[]
        2. Apply existing schema: compression_spec=None, columns=[...]
        3. Compress with inline spec: compression_spec={...}, columns=None
        4. Selective compression: compression_spec={...}, columns=[subset]
        5. Compress all eligible: no parameters (compresses SCHEMA_ONLY/DECOMPRESSED)

        Parameters
        ----------
        compression_spec : dict, optional
            Format: {
                'column_name': {
                    'compress': 'expression',           # e.g., 'round(asinh(dy)*40)'
                    'decompress': 'expression',         # e.g., 'sinh(dy_c/40.)'
                    'compressed_dtype': np.int16,       # Storage dtype
                    'decompressed_dtype': np.float16    # Reconstructed dtype
                }
            }
            If None, reuses existing schemas for specified columns.
        columns : list of str, optional
            Explicit column list. Behavior depends on compression_spec:
            - If [], defines schema-only without data (Pattern 1).
            - If None with spec, processes all columns in spec.
            - If provided with spec, processes only listed columns (Pattern 2).
            - If provided without spec, applies existing schemas.
        suffix : str, optional
            Compressed column name suffix (default: '_c'). Ignored when reusing schema.
        drop_original : bool, optional
            Remove original column after compression (default: True)
        on_missing : {'warn', 'error', 'ignore'}, optional
            How to handle columns that don't exist in DataFrame (default: 'warn'):
            - 'warn': Skip missing columns with warning
            - 'error': Raise KeyError if any column missing
            - 'ignore': Skip missing columns silently
        return_summary : bool, optional
            Return summary dict with compressed/skipped columns (default: False)
        measure_precision : bool, optional
            Compute and store compression precision loss (default: False)

        Returns
        -------
        dict or self
            If return_summary=True: {'compressed': [...], 'skipped': [...]}
            Otherwise: self (for method chaining)

        Raises
        ------
        ValueError
            If invalid state transition, name collision, or missing schema
        KeyError
            If on_missing='error' and columns don't exist in DataFrame

        Examples
        --------
        >>> # Pattern 1: Define schema first, compress subsets later
        >>> adf.define_compression_schema(spec)  # All → SCHEMA_ONLY
        >>> adf.compress_columns(columns=['dy', 'dz'])  # Subset → COMPRESSED
        >>> adf.compress_columns(columns=['tgSlp'])  # Later, compress more

        >>> # Pattern 2: Selective compression (register + compress together)
        >>> adf.compress_columns(spec, columns=['dy', 'dz'])  # Only dy, dz
        >>> adf.compress_columns(spec, columns=['tgSlp'])  # Add tgSlp later

        >>> # Direct compression (all columns in spec)
        >>> adf.compress_columns(spec)  # Compress everything

        >>> # Handle missing columns flexibly
        >>> adf.compress_columns(spec, on_missing='ignore')  # Skip silently
        >>> result = adf.compress_columns(spec, return_summary=True)  # Get summary

        Notes
        -----
        - State transitions: SCHEMA_ONLY → COMPRESSED, DECOMPRESSED → COMPRESSED
        - Cannot re-compress COMPRESSED state without decompressing first
        - Schema reuse ignores new suffix, uses stored compressed_col
        - Pattern 2 allows schema updates for SCHEMA_ONLY/DECOMPRESSED columns
        - Idempotent: re-compressing with same schema is silently skipped
        """
        # Determine mode and target columns
        if compression_spec is None and columns is None:
            # Mode: compress all columns with SCHEMA_ONLY or DECOMPRESSED state
            cols_to_process = [
                col for col in self.compression_info
                if col != "__meta__" and
                   self.compression_info[col].get('state') in (CompressionState.SCHEMA_ONLY, CompressionState.DECOMPRESSED)
            ]
            if not cols_to_process:
                return self  # Nothing to do
            schema_mode = 'reuse'
        elif compression_spec is None and columns is not None:
            # Mode: apply existing schema to specified columns
            cols_to_process = columns
            schema_mode = 'reuse'
        elif compression_spec is not None and columns == []:
            # Mode: schema-only definition
            cols_to_process = list(compression_spec.keys())
            schema_mode = 'define'
        elif compression_spec is not None and columns is None:
            # Mode: compress with inline spec (all columns in spec)
            cols_to_process = list(compression_spec.keys())
            schema_mode = 'inline'
        elif compression_spec is not None and columns is not None and len(columns) > 0:
            # Mode: selective registration + compression from spec
            # Only process columns explicitly listed in 'columns' parameter
            cols_to_process = columns
            schema_mode = 'selective'

            # Validate all requested columns are in spec
            missing_cols = [c for c in columns if c not in compression_spec]
            if missing_cols:
                raise ValueError(
                    f"Columns {missing_cols} not found in compression_spec. "
                    f"Available columns in spec: {list(compression_spec.keys())}"
                )
        else:
            raise ValueError(
                "Invalid parameter combination. Use either:\n"
                "- compress_columns(spec, columns=[]) for schema-only\n"
                "- compress_columns(columns=[...]) to apply existing schema\n"
                "- compress_columns(spec) for direct compression\n"
                "- compress_columns(spec, columns=[...]) for selective compression"
            )

        # === NEW: Filter columns based on on_missing parameter ===
        import warnings

        # Check which columns exist in DataFrame, aliases, OR are already tracked
        existing_in_df = set(self.df.columns)
        existing_in_aliases = set(self.aliases.keys())
        tracked_in_schema = set(self.compression_info.keys()) - {'__meta__'}

        # A column is "available" if:
        # 1. It exists physically in the DataFrame, OR
        # 2. It exists as an alias (compressed columns become aliases), OR
        # 3. It's already tracked in compression_info (for state validation)
        available_cols = [col for col in cols_to_process
                          if col in existing_in_df or
                          col in existing_in_aliases or
                          col in tracked_in_schema]

        # A column is "missing" only if it's nowhere: not in df, not an alias, not tracked
        missing_cols = [col for col in cols_to_process
                        if col not in existing_in_df and
                        col not in existing_in_aliases and
                        col not in tracked_in_schema]

        # Handle missing columns according to on_missing mode
        if missing_cols:
            if on_missing == 'error':
                raise KeyError(
                    f"Missing columns: {missing_cols}\n"
                    f"Available in DataFrame: {list(existing_in_df)[:20]}...\n"
                    f"Available as aliases: {list(existing_in_aliases)[:20]}..."
                )
            elif on_missing == 'warn':
                warnings.warn(
                    f"Skipping missing columns: {missing_cols}\n"
                    f"Hint: Use columns= to restrict, or on_missing='error' for strict mode."
                )
            # else: on_missing == 'ignore', do nothing

        # Update cols_to_process to only include available columns
        cols_to_process = available_cols
        # === END NEW CODE ===

        for orig_col in cols_to_process:
            # Get config (from spec or existing schema)
            if schema_mode == 'reuse':
                if orig_col not in self.compression_info:
                    raise ValueError(
                        f"No compression schema found for column '{orig_col}'. "
                        f"Define schema first with define_compression_schema()."
                    )
                config = self._schema_from_info(orig_col)
                existing_info = self.compression_info[orig_col]
                compressed_col = existing_info['compressed_col']
            elif schema_mode in ('inline', 'define', 'selective'):
                config = compression_spec[orig_col]
                # Validate config
                required_keys = ['compress', 'decompress', 'compressed_dtype', 'decompressed_dtype']
                missing = [k for k in required_keys if k not in config]
                if missing:
                    raise ValueError(
                        f"Compression config for '{orig_col}' missing required keys: {missing}"
                    )
                compressed_col = f"{orig_col}{suffix}"

            # For selective mode, nothing to be done

            # Check current state and validate transitions
            current_state = self.get_compression_state(orig_col)

            if schema_mode == 'define':
                # Schema-only mode: just store metadata
                if current_state is not None:
                    raise ValueError(
                        f"Column '{orig_col}' already has compression schema with state '{current_state}'. "
                        f"Remove existing schema first."
                    )
                # Store schema-only metadata
                self.compression_info[orig_col] = {
                    'compressed_col': compressed_col,
                    'compress_expr': config['compress'],
                    'decompress_expr': config['decompress'],
                    'compressed_dtype': np.dtype(config['compressed_dtype']).name,
                    'decompressed_dtype': np.dtype(config['decompressed_dtype']).name,
                    'state': CompressionState.SCHEMA_ONLY,
                    'original_removed': False
                }
                continue  # Don't compress data, just store schema

            # For actual compression (inline, reuse, or selective mode):
            
            # EARLY CHECK: If data is physically already compressed
            # This handles mixed-state scenarios where embedded schema may be inconsistent
            data_is_physically_compressed = (
                compressed_col in self.df.columns and
                orig_col not in self.df.columns
            )
            if data_is_physically_compressed:
                # Data is already in compressed form
                # If a new schema is being provided, check if it differs
                if schema_mode in ('selective', 'inline') and compression_spec:
                    existing_schema = self._schema_from_info(orig_col)
                    new_schema = compression_spec.get(orig_col, {})
                    if not self._schemas_equal(existing_schema, new_schema):
                        # Different schema - cannot change schema of compressed column
                        raise ValueError(
                            f"Column '{orig_col}' is already compressed with a different schema. "
                            f"Please decompress first before applying new compression schema:\n"
                            f"  adf.decompress_columns(['{orig_col}'], keep_schema=False)\n"
                            f"  adf.compress_columns(new_spec, columns=['{orig_col}'])"
                        )
                # Same schema or no new schema - skip (idempotent)
                # Update state to reflect reality
                if orig_col in self._schema.get('compression', {}):
                    self._schema['compression'][orig_col]['state'] = CompressionState.COMPRESSED
                continue
            
            # Special handling for selective mode with COMPRESSED state
            if schema_mode == 'selective' and current_state == CompressionState.COMPRESSED:
                # Check if schema is the same or different
                existing_schema = self._schema_from_info(orig_col)
                if self._schemas_equal(existing_schema, config):
                    # Same schema, already compressed - skip (idempotent)
                    continue
                else:
                    # Different schema - must decompress first
                    raise ValueError(
                        f"Column '{orig_col}' is already compressed with a different schema. "
                        f"Please decompress first before applying new compression schema:\n"
                        f"  adf.decompress_columns(['{orig_col}'], keep_schema=False)\n"
                        f"  adf.compress_columns(new_spec, columns=['{orig_col}'])"
                    )

            # Standard state validation for non-selective modes
            if current_state == CompressionState.COMPRESSED:
                # Verify data is actually compressed, not just schema state
                # This handles case where schema was loaded but data wasn't compressed
                comp_info = self._schema['compression'].get(orig_col, {})
                compressed_col_name = comp_info.get('compressed_col', f'{orig_col}_c')
                data_is_compressed = (
                    compressed_col_name in self.df.columns and
                    orig_col not in self.df.columns
                )
                
                if data_is_compressed:
                    # Truly compressed - skip (idempotent)
                    continue
                else:
                    # Schema says compressed but data isn't - treat as SCHEMA_ONLY
                    # Update state to reflect reality
                    self._schema['compression'][orig_col]['state'] = CompressionState.SCHEMA_ONLY
                    current_state = CompressionState.SCHEMA_ONLY
                    # Fall through to compress
            
            if current_state == CompressionState.SCHEMA_ONLY:
                # Valid transition: SCHEMA_ONLY → COMPRESSED
                pass
            elif current_state == CompressionState.DECOMPRESSED:
                # Valid transition: DECOMPRESSED → COMPRESSED (recompression)
                pass
            elif current_state is None:
                # Valid transition: None → COMPRESSED (inline compression)
                if schema_mode == 'reuse':
                    raise ValueError(
                        f"Column '{orig_col}' has no compression schema. "
                        f"Cannot reuse non-existent schema."
                    )

            # Collision detection for compressed_col name
            self._validate_compressed_col_name(orig_col, compressed_col)

            # Cache original values if measuring precision
            original_values = None
            if measure_precision and orig_col in self.df.columns:
                original_values = self.df[orig_col].values.copy()

            # Step 1: Create and materialize compressed version
            try:
                # For recompression, remove old compressed column if it exists
                if compressed_col in self.df.columns:
                    self.df.drop(columns=[compressed_col], inplace=True)

                self.add_alias(compressed_col, config['compress'],
                               dtype=config['compressed_dtype'])
                self.materialize_alias(compressed_col)
                # Remove from aliases to avoid false cycle detection
                if compressed_col in self.aliases:
                    del self._schema["columns"][compressed_col]
            except SyntaxError as e:
                raise ValueError(
                    f"Compression failed for '{orig_col}': invalid compress expression.\n"
                    f"Expression: {config['compress']}\n"
                    f"Error: {e}"
                ) from e
            except KeyError as e:
                raise ValueError(
                    f"Compression failed for '{orig_col}': undefined variable in compress expression.\n"
                    f"Expression: {config['compress']}\n"
                    f"Error: {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Compression failed for '{orig_col}' during compress step: {e}"
                ) from e

            # Step 2: Measure precision loss if requested
            precision_info = None
            if measure_precision and original_values is not None:
                precision_info = self._measure_compression_precision(
                    orig_col, original_values, config
                )

            # Step 3: Remove original from storage (if requested and exists)
            if drop_original and orig_col in self.df.columns:
                self.df.drop(columns=[orig_col], inplace=True)

            # Step 4: Remove old decompression alias if it exists (from DECOMPRESSED state)
            if orig_col in self.aliases:
                del self._schema["columns"][orig_col]

            # Step 5: Add decompression alias (original name → decompressed expression)
            try:
                self.add_alias(orig_col, config['decompress'],
                               dtype=config['decompressed_dtype'])
            except SyntaxError as e:
                raise ValueError(
                    f"Compression failed for '{orig_col}': invalid decompress expression.\n"
                    f"Expression: {config['decompress']}\n"
                    f"Error: {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Compression failed for '{orig_col}' during decompress alias creation: {e}"
                ) from e

            # Step 6: Store/update metadata (JSON-safe: dtypes as strings)
            self.compression_info[orig_col] = {
                'compressed_col': compressed_col,
                'compress_expr': config['compress'],
                'decompress_expr': config['decompress'],
                'compressed_dtype': np.dtype(config['compressed_dtype']).name,
                'decompressed_dtype': np.dtype(config['decompressed_dtype']).name,
                'state': CompressionState.COMPRESSED,
                'original_removed': drop_original
            }

            if precision_info is not None:
                self.compression_info[orig_col]['precision'] = precision_info

        # === NEW: Return summary if requested ===
        if return_summary:
            compressed_cols = [col for col in available_cols
                               if schema_mode != 'define' and
                               col in self.compression_info and
                               self.compression_info[col].get('state') == CompressionState.COMPRESSED]
            return {
                'compressed': compressed_cols,
                'skipped': missing_cols
            }
        # === END NEW CODE ===

        return self
    def _validate_compressed_col_name(self, orig_col, compressed_col):
        """
        Validate compressed column name doesn't conflict.

        Three cases:
        1. Matching schema (recompression) - allow
        2. Name used by other column - error
        3. Name exists but not in schema - error
        """
        # Case 1: Check if this is recompression with matching schema
        if orig_col in self.compression_info:
            stored_compressed_col = self.compression_info[orig_col].get('compressed_col')
            if stored_compressed_col == compressed_col:
                # This is recompression - allowed
                return

        # Case 2: Check if another column owns this compressed_col name
        for col, info in self.compression_info.items():
            if col == "__meta__" or col == orig_col:
                continue
            if info.get('compressed_col') == compressed_col:
                raise ValueError(
                    f"Compressed column name '{compressed_col}' is already used by column '{col}'. "
                    f"Choose a different suffix or fix existing schema."
                )

        # Case 3: Check if name exists in df or aliases (not from schema)
        if compressed_col in self.df.columns:
            raise ValueError(
                f"Compressed column name '{compressed_col}' already exists in DataFrame. "
                f"Choose a different suffix or rename the existing column."
            )
        if compressed_col in self.aliases:
            raise ValueError(
                f"Compressed column name '{compressed_col}' conflicts with existing alias. "
                f"Choose a different suffix."
            )

    def _measure_compression_precision(self, orig_col, original_values, config):
        """
        Measure compression precision loss with RMSE and error metrics.

        Returns dict with precision metrics or error info.
        """
        temp_decompressed = f"__temp_decompress_{orig_col}"
        if temp_decompressed in self.df.columns or temp_decompressed in self.aliases:
            raise ValueError(
                f"Internal error: temporary column name '{temp_decompressed}' already exists. "
                f"This should not happen - please report this bug."
            )

        try:
            self.add_alias(temp_decompressed, config['decompress'],
                           dtype=config['decompressed_dtype'])
            self.materialize_alias(temp_decompressed)
            decompressed_values = self.df[temp_decompressed].values

            # Compute precision metrics on finite values only
            orig = original_values.astype(np.float64)
            decomp = decompressed_values.astype(np.float64)
            finite_mask = np.isfinite(orig) & np.isfinite(decomp)

            n_total = len(orig)
            n_finite = int(finite_mask.sum())

            # Always calculate on finite subset (NaN if empty)
            if n_finite > 0:
                diff = orig[finite_mask] - decomp[finite_mask]
                with np.errstate(over='ignore', invalid='ignore'):
                    rmse = float(np.sqrt(np.mean(diff ** 2)))
                    if not np.isfinite(rmse):
                        rmse = float(np.sqrt(np.median(diff ** 2)) * 1.2533)
                max_error = float(np.max(np.abs(diff)))
                mean_error = float(np.mean(diff))
            else:
                rmse = float('nan')
                max_error = float('nan')
                mean_error = float('nan')

            # Always same structure
            precision_info = {
                'n_samples': n_finite,
                'n_total': n_total,
                'fraction_nonfinite': float((n_total - n_finite) / n_total) if n_total > 0 else 0.0,
                'rmse': rmse,
                'max_error': max_error,
                'mean_error': mean_error
            }

            # Clean up temporary column
            self.df.drop(columns=[temp_decompressed], inplace=True)
            if temp_decompressed in self.aliases:
                del self._schema["columns"][temp_decompressed]

            return precision_info
        except Exception as e:
            # Non-fatal: return error info
            return {'error': str(e)}

    def define_compression_schema(self, compression_spec, suffix='_c'):
        """
        Define compression schema without compressing data (forward declaration).

        Creates SCHEMA_ONLY entries that can be applied later when data exists.

        Parameters
        ----------
        compression_spec : dict
            Compression specification (same format as compress_columns)
        suffix : str, optional
            Compressed column name suffix (default: '_c')

        Returns
        -------
        self : AliasDataFrame
            For method chaining

        Examples
        --------
        >>> # Define schema upfront
        >>> spec = {'dy': {...}, 'dz': {...}}
        >>> adf.define_compression_schema(spec)
        >>> # Later, when data exists:
        >>> adf.compress_columns(columns=['dy', 'dz'])
        """
        return self.compress_columns(compression_spec, columns=[], suffix=suffix)

    def decompress_columns(self, columns=None, inplace=False, keep_compressed=True, keep_schema=True):
        """
        Materialize decompressed versions of compressed columns with state management.

        Parameters
        ----------
        columns : list of str, optional
            Columns to decompress. If None, decompress all COMPRESSED columns.
        inplace : bool, optional
            DEPRECATED: Use keep_schema=False instead.
            If True, same as keep_schema=False + keep_compressed=False.
        keep_compressed : bool, optional
            If False, remove compressed columns after decompression (default: True).
        keep_schema : bool, optional
            If True, keep compression schema and transition to DECOMPRESSED state.
            If False, remove all compression metadata (default: True).

        Returns
        -------
        self : AliasDataFrame
            For method chaining

        Raises
        ------
        ValueError
            If column not in COMPRESSED state or data missing

        Examples
        --------
        >>> # Decompress, keep schema for recompression
        >>> adf.decompress_columns(['dy', 'dz'])  # state → DECOMPRESSED

        >>> # Decompress and remove all compression info
        >>> adf.decompress_columns(['dy'], keep_schema=False)  # state → None

        Notes
        -----
        - Always materializes the decompression alias first
        - Removes alias after materialization (col becomes physical column)
        - State transitions: COMPRESSED → DECOMPRESSED or COMPRESSED → None
        - Cannot decompress SCHEMA_ONLY (never compressed) or DECOMPRESSED (already done)
        """
        # Handle legacy inplace parameter
        if inplace:
            keep_schema = False
            keep_compressed = False

        # Determine columns to process
        if columns is None:
            # Only decompress columns in COMPRESSED state
            columns = [
                col for col in self.compression_info
                if col != "__meta__" and
                self.compression_info[col].get('state') == CompressionState.COMPRESSED
            ]

        # Filter __meta__
        columns = [c for c in columns if c != "__meta__"]

        # Phase 13.7: Collect columns to drop, then batch-drop once at end
        # (Avoids O(N_cols × N_rows) pandas reindex per drop)
        cols_to_drop = []

        for col in columns:
            if col not in self.compression_info:
                raise ValueError(
                    f"Column '{col}' has no compression metadata. "
                    f"Available: {[c for c in self.compression_info.keys() if c != '__meta__']}"
                )

            info = self.compression_info[col]
            current_state = info.get('state')

            # Validate state transition
            if current_state == CompressionState.SCHEMA_ONLY:
                # Warn but allow (no-op): never compressed, nothing to decompress
                continue
            elif current_state == CompressionState.DECOMPRESSED:
                # Already decompressed, skip
                continue
            elif current_state != CompressionState.COMPRESSED:
                raise ValueError(
                    f"Column '{col}' is in state '{current_state}', cannot decompress. "
                    f"Only COMPRESSED columns can be decompressed."
                )

            compressed_col = info['compressed_col']

            # Validate compressed column exists
            if compressed_col not in self.df.columns:
                raise ValueError(
                    f"Compressed column '{compressed_col}' for '{col}' is missing. "
                    f"Cannot decompress without source data."
                )

            # Step 1: Materialize decompressed alias
            if col not in self.aliases:
                raise ValueError(
                    f"Internal error: decompression alias for '{col}' is missing. "
                    f"This indicates corrupted compression_info."
                )

            self.materialize_alias(col)

            # Step 2: Enforce decompressed dtype
            target_dtype = np.dtype(info['decompressed_dtype']).type
            self.df[col] = self.df[col].astype(target_dtype)

            # Step 3: Remove decompression alias (col is now physical)
            if col in self.aliases:
                del self._schema["columns"][col]

            # Step 4: Collect compressed column for batch drop
            if not keep_compressed:
                cols_to_drop.append(compressed_col)

            # Step 5: Update state
            if keep_schema:
                # Transition to DECOMPRESSED state
                self.compression_info[col]['state'] = CompressionState.DECOMPRESSED
            else:
                # Remove all compression metadata
                del self.compression_info[col]

        # Batch drop all compressed columns at once (single reindex)
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)

        return self

    def get_compression_info(self, column=None):
        """
        Get compression metadata for columns.

        Parameters
        ----------
        column : str, optional
            Specific column. If None, return all compression info as DataFrame.

        Returns
        -------
        dict or pd.DataFrame
            Compression metadata for specified column or all columns

        Examples
        --------
        >>> adf.get_compression_info('dy')
        {'compressed_col': 'dy_c', 'compress_expr': 'round(asinh(dy)*40)', ...}

        >>> adf.get_compression_info()  # All compressed columns as DataFrame
        """
        if column is None:
            # Filter out __meta__ when returning all info
            info_without_meta = {k: v for k, v in self.compression_info.items() if k != "__meta__"}
            if not info_without_meta:
                return pd.DataFrame()
            return pd.DataFrame.from_dict(info_without_meta, orient='index')
        else:
            return self.compression_info.get(column, {})

    # Verbosity flags for describe_compression (bitmask)
    COMPRESS_SHOW_CORE   = 0x01  # name, state, compressed_col, dtype (always on)
    COMPRESS_SHOW_EXPR   = 0x02  # compress/decompress expressions
    COMPRESS_SHOW_PREC   = 0x04  # precision metrics (RMSE, max, mean)
    COMPRESS_SHOW_STATS  = 0x08  # sample counts, non-finite fraction
    COMPRESS_SHOW_ALL    = 0x0F  # all flags

    # Verbosity flags for describe_data (bitmask)
    DATA_SHOW_CORE   = 0x01  # name, dtype, memory, shape
    DATA_SHOW_STATS  = 0x02  # null count, unique values  
    DATA_SHOW_META   = 0x04  # user metadata (if present)
    DATA_SHOW_SOURCE = 0x08  # physical/alias/subframe_ref
    DATA_SHOW_ALL    = 0x0F  # all flags

    # Verbosity flags for describe_schema (bitmask)
    SCHEMA_SHOW_CORE        = 0x01  # basic overview
    SCHEMA_SHOW_COMPRESSION = 0x02  # compression rules
    SCHEMA_SHOW_METADATA    = 0x04  # user metadata
    SCHEMA_SHOW_SUBFRAMES   = 0x08  # recursive schemas
    SCHEMA_SHOW_ALL         = 0x0F  # all flags

    def select_compression(self, pattern=None, names=None, 
                           only_compressed=False, only_decompressed=False,
                           only_failed=False):
        """
        Select compressed columns by pattern and/or names with optional filters.
        
        Parameters
        ----------
        pattern : str, optional
            Regex pattern to match column names
        names : list, optional
            Explicit list of column names to include
        only_compressed : bool, default=False
            If True, include only columns in 'compressed' state
        only_decompressed : bool, default=False
            If True, include only columns in 'decompressed' state
        only_failed : bool, default=False
            If True, include only columns where user-defined monitor threshold is exceeded.
            Requires 'monitor' to be defined in compression_info for the column.
            
        Returns
        -------
        list
            List of column names matching the criteria
        """
        import re as re_module
        
        # Validate mutually exclusive filters
        exclusive_filters = [only_compressed, only_decompressed]
        if sum(bool(x) for x in exclusive_filters) > 1:
            raise ValueError("Filters only_compressed and only_decompressed are mutually exclusive")
        
        # Compile and validate pattern
        regex = None
        if pattern:
            try:
                regex = re_module.compile(pattern)
            except re_module.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        # Get compression info (excluding __meta__)
        columns_info = {k: v for k, v in self.compression_info.items() if k != "__meta__"}
        
        result = []
        for name, info in columns_info.items():
            # Filter by explicit names
            if names is not None and name not in names:
                continue
            
            # Filter by pattern
            if regex and not regex.search(name):
                continue
            
            # Filter by state
            state = info.get('state', 'unknown')
            if only_compressed and state != 'compressed':
                continue
            if only_decompressed and state != 'decompressed':
                continue
            
            # Filter by failure status (requires user-defined monitor)
            if only_failed:
                monitor = info.get('monitor')
                if monitor is None:
                    # No monitor defined - skip (not considered failed)
                    continue
                if not self._check_monitor_failed(info):
                    continue
            
            result.append(name)
        
        return result

    def _check_monitor_failed(self, info):
        """Check if compression monitor threshold is exceeded.
        
        Returns True if monitor is defined and threshold exceeded.
        Returns False if no monitor or threshold not exceeded.
        """
        monitor = info.get('monitor')
        if monitor is None:
            return False
        
        prec = info.get('precision', {})
        threshold = monitor.get('threshold')
        if threshold is None:
            return False
        
        monitor_type = monitor.get('type')
        
        if monitor_type == 'absolute':
            # Check absolute RMSE
            return prec.get('rmse', 0) > threshold
        
        elif monitor_type == 'relative':
            relative_to = monitor.get('relative_to', 'data_range')
            rmse = prec.get('rmse', 0)
            
            if relative_to == 'data_range':
                data_range = prec.get('data_range')
                if data_range:
                    range_size = abs(data_range[1] - data_range[0])
                    if range_size > 0:
                        return rmse / range_size > threshold
            # Could add other relative_to options here
            return False
        
        elif monitor_type == 'function':
            func = monitor.get('func')
            if func and callable(func):
                try:
                    value = func(prec)
                    return value > threshold
                except Exception:
                    return True  # Error in function = failed
            return False
        
        return False

    def _get_monitor_value(self, info):
        """Compute current monitor value for display."""
        monitor = info.get('monitor')
        if monitor is None:
            return None, None
        
        prec = info.get('precision', {})
        monitor_type = monitor.get('type')
        label = monitor.get('label', monitor_type)
        
        if monitor_type == 'absolute':
            return label or 'RMSE', prec.get('rmse', 0)
        
        elif monitor_type == 'relative':
            relative_to = monitor.get('relative_to', 'data_range')
            rmse = prec.get('rmse', 0)
            
            if relative_to == 'data_range':
                data_range = prec.get('data_range')
                if data_range:
                    range_size = abs(data_range[1] - data_range[0])
                    if range_size > 0:
                        return label or 'RMSE/range', rmse / range_size
            return label or 'relative', None
        
        elif monitor_type == 'function':
            func = monitor.get('func')
            if func and callable(func):
                try:
                    value = func(prec)
                    return label or 'custom', value
                except Exception as e:
                    return label or 'custom', f'error: {e}'
        
        return None, None

    def validate_schema(
        self,
        check_data: bool = True,
        allow_missing_columns: bool = True,
        allow_pending_aliases: bool = True,
        strict: bool = False,
        raise_on_error: bool = False,
        verbose: bool = True
    ):
        """
        Validate schema consistency, optionally checking against DataFrame state.
        
        This method validates the schema (definition/blueprint) and optionally
        checks consistency with the current DataFrame state (runtime).
        
        Two modes:
        - check_data=True (default): Validate schema AND check against DataFrame
        - check_data=False: Validate schema structure only (for templates/blueprints)
        
        Parameters
        ----------
        check_data : bool, default=True
            If True, validate schema against current DataFrame contents.
            If False, validate schema structure only (for definition schemas/templates).
            
        allow_missing_columns : bool, default=True
            If True, columns in schema but not in DataFrame are valid (pending state).
            Schema-as-specification: columns can be defined before they exist in data.
            Only relevant when check_data=True.
            
        allow_pending_aliases : bool, default=True
            If True, aliases with missing dependencies are valid (pending).
            Dependencies may be provided later via subframes or other means.
            Only relevant when check_data=True.
            
        strict : bool, default=False
            Convenience parameter. If True, sets allow_missing_columns=False and
            allow_pending_aliases=False. Use after all subframes are registered
            and before compression to ensure all dependencies are present.
            
        raise_on_error : bool, default=False
            If True, raise ValueError on first error. If False, collect all issues.
            
        verbose : bool, default=True
            If True, print validation results.
            
        Returns
        -------
        dict
            Validation results with keys:
            - 'valid': bool - True if no errors (respecting allow_* flags)
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'pending': list of pending items (missing columns, unresolved aliases)
            - 'info': dict with detailed state info
            
        Examples
        --------
        >>> # Default validation (permissive - pending allowed)
        >>> result = adf.validate_schema()
        
        >>> # Strict validation (no pending allowed)
        >>> result = adf.validate_schema(strict=True)
        
        >>> # Template/blueprint validation (schema only, no data check)
        >>> result = adf.validate_schema(check_data=False)
        """
        # Apply strict mode
        if strict:
            allow_missing_columns = False
            allow_pending_aliases = False
        
        errors = []
        warnings = []
        pending = []
        info = {
            'compression_targets': [],
            'physical_columns': list(self.df.columns),
            'aliases': list(self.aliases.keys()),
            'pending_aliases': [],
            'pending_columns': [],
            'compression_state_mismatches': []
        }
        
        # =====================================================================
        # 1. Schema structure validation (always performed)
        # =====================================================================
        
        # Check for required schema sections
        if 'columns' not in self._schema:
            self._schema['columns'] = {}
        if 'compression' not in self._schema:
            self._schema['compression'] = {}
        
        # Validate compression definitions have required fields
        compression_info = self._schema.get('compression', {})
        for orig_col, comp_info in compression_info.items():
            if orig_col == '__meta__':
                continue
            info['compression_targets'].append(orig_col)
            
            # Check required fields
            required_fields = ['compress_expr', 'decompress_expr']
            for field in required_fields:
                if field not in comp_info:
                    warnings.append(f"Compression '{orig_col}': missing '{field}'")
        
        # =====================================================================
        # 2. Data consistency validation (only if check_data=True)
        # =====================================================================
        
        if check_data:
            # -----------------------------------------------------------------
            # 2a. Check for compression cycle risks (ALWAYS an error)
            # -----------------------------------------------------------------
            for orig_col, comp_info in compression_info.items():
                if orig_col == '__meta__':
                    continue
                
                compressed_col = comp_info.get('compressed_col', f'{orig_col}_c')
                
                # Check if orig_col is registered as alias depending on compressed_col
                if orig_col in self.aliases:
                    expr = self.aliases[orig_col]
                    if compressed_col in expr:
                        # Check actual data state
                        has_compressed = compressed_col in self.df.columns
                        has_original = orig_col in self.df.columns
                        
                        if has_original and not has_compressed:
                            # Original column exists as physical, but alias says it depends on _c
                            # This WILL cause cycle when compressing
                            err = (
                                f"CYCLE RISK: '{orig_col}' is registered as alias "
                                f"depending on '{compressed_col}', but '{orig_col}' "
                                f"is a physical column. Compression will fail.\n"
                                f"  Hint: This usually means a definition schema was loaded "
                                f"with runtime state. Re-export with include_state=False."
                            )
                            errors.append(err)
                            if raise_on_error:
                                raise ValueError(err)
            
            # -----------------------------------------------------------------
            # 2b. Check compression state vs actual data
            # -----------------------------------------------------------------
            for orig_col, comp_info in compression_info.items():
                if orig_col == '__meta__':
                    continue
                    
                compressed_col = comp_info.get('compressed_col', f'{orig_col}_c')
                state = self.get_compression_state(orig_col)
                
                has_compressed = compressed_col in self.df.columns
                has_original = orig_col in self.df.columns
                is_alias = orig_col in self.aliases
                
                # Determine actual state from data
                if has_compressed and not has_original:
                    actual_state = 'compressed'
                elif has_original and not has_compressed:
                    actual_state = 'decompressed' if is_alias else 'uncompressed'
                elif has_original and has_compressed:
                    actual_state = 'both_exist'
                else:
                    actual_state = 'neither_exist'
                
                # Compare with schema state
                if state == CompressionState.COMPRESSED and actual_state != 'compressed':
                    mismatch = f"'{orig_col}': schema says 'compressed' but actual is '{actual_state}'"
                    info['compression_state_mismatches'].append(mismatch)
                    warnings.append(f"STATE MISMATCH: {mismatch}")
                
                if actual_state == 'both_exist':
                    warnings.append(f"UNUSUAL: Both '{orig_col}' and '{compressed_col}' exist in DataFrame")
            
            # -----------------------------------------------------------------
            # 2c. Check for pending aliases (missing dependencies)
            # -----------------------------------------------------------------
            for alias_name, expr in self.aliases.items():
                missing_deps = self._find_missing_dependencies(alias_name)
                
                if missing_deps:
                    info['pending_aliases'].append({
                        'name': alias_name,
                        'missing': missing_deps
                    })
                    
                    pending_msg = f"Pending alias '{alias_name}': missing {missing_deps}"
                    pending.append(pending_msg)
                    
                    if not allow_pending_aliases:
                        errors.append(pending_msg)
                        if raise_on_error:
                            raise ValueError(pending_msg)
            
            # -----------------------------------------------------------------
            # 2d. Check for missing columns (in schema but not in data)
            # -----------------------------------------------------------------
            schema_columns = self._schema.get('columns', {})
            for col_name, col_info in schema_columns.items():
                # Skip aliases (they're computed, not physical)
                if col_info.get('expr'):
                    continue
                
                if col_name not in self.df.columns:
                    info['pending_columns'].append(col_name)
                    pending_msg = f"Pending column '{col_name}': defined in schema but not in DataFrame"
                    pending.append(pending_msg)
                    
                    if not allow_missing_columns:
                        errors.append(pending_msg)
                        if raise_on_error:
                            raise ValueError(pending_msg)
        
        # =====================================================================
        # 3. Build result
        # =====================================================================
        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'pending': pending,
            'info': info
        }
        
        # =====================================================================
        # 4. Print report if verbose
        # =====================================================================
        if verbose:
            print("=" * 60)
            print("SCHEMA VALIDATION REPORT")
            print("=" * 60)
            
            mode = "Full (schema + data)" if check_data else "Schema only (blueprint)"
            print(f"Mode: {mode}")
            print()
            
            if result['valid']:
                print("✓ Schema is valid (no errors)")
            else:
                print(f"✗ Schema has {len(errors)} error(s)")
            
            if errors:
                print("\nERRORS:")
                for err in errors:
                    # Handle multi-line errors
                    lines = err.split('\n')
                    print(f"  • {lines[0]}")
                    for line in lines[1:]:
                        print(f"    {line}")
            
            if warnings:
                print(f"\nWARNINGS ({len(warnings)}):")
                for warn in warnings:
                    print(f"  • {warn}")
            
            if pending and check_data:
                status = "allowed" if (allow_missing_columns and allow_pending_aliases) else "NOT allowed"
                print(f"\nPENDING ({len(pending)}) - {status}:")
                for p in pending[:10]:  # Show first 10
                    print(f"  • {p}")
                if len(pending) > 10:
                    print(f"  ... and {len(pending) - 10} more")
            
            print(f"\nSummary:")
            print(f"  Compression targets: {len(info['compression_targets'])}")
            print(f"  Physical columns: {len(info['physical_columns'])}")
            print(f"  Registered aliases: {len(info['aliases'])}")
            if check_data:
                print(f"  Pending aliases: {len(info['pending_aliases'])}")
                print(f"  Pending columns: {len(info['pending_columns'])}")
            print("=" * 60)
        
        return result

    def _find_missing_dependencies(self, alias_name):
        """
        Find missing dependencies for an alias.
        
        Returns list of column names that are referenced but not available.
        """
        if alias_name not in self.aliases:
            return []
        
        expr = self.aliases[alias_name]
        
        # Get all available columns (physical + other aliases + subframe refs)
        available = set(self.df.columns)
        available.update(self.aliases.keys())
        
        # Add subframe columns (T.column syntax will be handled during eval)
        for sf_name, sf_entry in self._subframes.subframes.items():
            sf_adf = sf_entry.get('frame')
            if sf_adf:
                # Add direct column names (auto_alias_subframe makes them available)
                available.update(sf_adf.df.columns)
                available.update(sf_adf.aliases.keys())
        
        # Simple token extraction (not perfect but catches most cases)
        import re
        # Match identifiers but not inside strings or after dots
        tokens = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr))
        
        # Remove known functions and constants
        builtins = {
            'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'sqrt', 'abs', 'pow',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'arcsin', 'arccos', 'arctan', 'arctan2', 'atan2',
            'floor', 'ceil', 'round', 'int', 'float', 'bool',
            'min', 'max', 'sum', 'mean', 'std', 'var',
            'pi', 'e', 'inf', 'nan', 'True', 'False',
            'where', 'clip', 'sign', 'deg2rad', 'rad2deg',
            'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
            'float16', 'float32', 'float64'
        }
        tokens -= builtins
        
        # Find missing
        missing = []
        for token in tokens:
            if token not in available:
                # Check if it's a subframe reference (T.column)
                # The token would be 'T' and we'd see '.column' after
                if token in self._subframes.subframes:
                    continue  # Subframe name is valid
                missing.append(token)
        
        return missing

    def describe_compression(self, verbosity=0x07, pattern=None, names=None, as_dict=False,
                             only_compressed=False, only_decompressed=False, 
                             only_failed=False, color=False):
        """
        Print human-readable compression summary.
        
        Parameters
        ----------
        verbosity : int, default=COMPRESS_SHOW_CORE | COMPRESS_SHOW_EXPR | COMPRESS_SHOW_PREC (0x07)
            Bitmask controlling output detail:
            - COMPRESS_SHOW_CORE (0x01): name, state, compressed_col, dtype
            - COMPRESS_SHOW_EXPR (0x02): compress/decompress expressions
            - COMPRESS_SHOW_PREC (0x04): precision metrics (RMSE, max, mean)
            - COMPRESS_SHOW_STATS (0x08): sample counts, non-finite fraction
            - COMPRESS_SHOW_ALL (0x0F): all flags
        pattern : str, optional
            Regex pattern to filter columns by name
        names : list, optional
            Explicit list of column names to include
        as_dict : bool, default=False
            If True, return dict instead of printing
        only_compressed : bool, default=False
            If True, show only columns in 'compressed' state
        only_decompressed : bool, default=False
            If True, show only columns in 'decompressed' state
        only_failed : bool, default=False
            If True, show only columns where user-defined monitor threshold is exceeded.
            Requires 'monitor' to be defined in compression_info.
        color : bool, default=False
            If True, use ANSI colors in output
            
        Returns
        -------
        dict or None
            If as_dict=True, returns structured dict of compression info
        """
        # Mask unknown verbosity bits
        verbosity &= self.COMPRESS_SHOW_ALL
        
        # Use select_compression for filtering
        selected = self.select_compression(
            pattern=pattern, names=names,
            only_compressed=only_compressed,
            only_decompressed=only_decompressed,
            only_failed=only_failed
        )
        
        # Get full compression info
        columns_info = {k: v for k, v in self.compression_info.items() if k != "__meta__"}
        
        # Color codes
        if color:
            C_RED = '\033[91m'
            C_GREEN = '\033[92m'
            C_YELLOW = '\033[93m'
            C_CYAN = '\033[96m'
            C_RESET = '\033[0m'
        else:
            C_RED = C_GREEN = C_YELLOW = C_CYAN = C_RESET = ''
        
        # Build result dict
        result = {}
        for name in selected:
            info = columns_info[name]
            # Use get_compression_state() to properly infer state for definition schemas
            state = self.get_compression_state(name)
            if state is None:
                state = 'unknown'
            entry = {
                'name': name,
                'state': state,
                'compressed_col': info.get('compressed_col'),
                'compressed_dtype': info.get('compressed_dtype'),
                'decompressed_dtype': info.get('decompressed_dtype'),
                'compress_expr': info.get('compress_expr'),
                'decompress_expr': info.get('decompress_expr'),
                'original_removed': info.get('original_removed', False),
                'precision': info.get('precision'),
                'monitor': info.get('monitor'),
            }
            # Add monitor status if monitor is defined
            if entry['monitor']:
                label, value = self._get_monitor_value(info)
                entry['monitor_label'] = label
                entry['monitor_value'] = value
                entry['monitor_failed'] = self._check_monitor_failed(info)
            result[name] = entry
        
        if as_dict:
            return result
        
        if not result:
            print("No compressed columns" + (" matching criteria" if pattern or names or only_failed else ""))
            return
        
        # Print header
        print("Compression Info:")
        print(f"  {'Name':<20} {'State':<12} {'Compressed':<15} {'Dtype':<10}")
        print(f"  {'-'*20} {'-'*12} {'-'*15} {'-'*10}")
        
        for name, info in result.items():
            state = info['state']
            state_str = state
            if color:
                if state == 'compressed':
                    state_str = f"{C_GREEN}{state}{C_RESET}"
                elif state == 'decompressed':
                    state_str = f"{C_CYAN}{state}{C_RESET}"
            
            comp_col = info['compressed_col'] or '-'
            comp_dtype = info['compressed_dtype'] or '-'
            
            print(f"  {name:<20} {state_str:<12} {comp_col:<15} {comp_dtype:<10}")
            
            # Show monitor failure if applicable
            if info.get('monitor_failed'):
                monitor = info['monitor']
                label = info.get('monitor_label', 'monitor')
                value = info.get('monitor_value')
                threshold = monitor.get('threshold')
                fail_str = f"{label}={value:.4g} > {threshold}"
                if color:
                    fail_str = f"{C_RED}{fail_str}{C_RESET}"
                print(f"  {'':<20} {'^ Failed:':<12} {fail_str}")
            
            # Show expressions if requested
            if verbosity & self.COMPRESS_SHOW_EXPR:
                if info['compress_expr']:
                    print(f"  {'':<20} {'Compress:':<12} {info['compress_expr']}")
                if info['decompress_expr']:
                    decomp_str = f"{info['decompress_expr']} → {info['decompressed_dtype']}"
                    print(f"  {'':<20} {'Decompress:':<12} {decomp_str}")
            
            # Show precision if requested
            if verbosity & self.COMPRESS_SHOW_PREC:
                prec = info.get('precision')
                if prec:
                    if 'error' in prec:
                        print(f"  {'':<20} {'Precision:':<12} failed ({prec['error']})")
                    else:
                        print(f"  {'':<20} {'Precision:':<12} RMSE={prec.get('rmse', 0):.6f}, "
                              f"Max={prec.get('max_error', 0):.6f}, "
                              f"Mean={prec.get('mean_error', 0):.6f}")
            
            # Show stats if requested
            if verbosity & self.COMPRESS_SHOW_STATS:
                prec = info.get('precision')
                if prec and 'n_samples' in prec:
                    n_samples = prec.get('n_samples', 0)
                    n_total = prec.get('n_total', n_samples)
                    frac_nonfinite = prec.get('fraction_nonfinite', 0.0)
                    print(f"  {'':<20} {'Samples:':<12} {n_samples:,}/{n_total:,}, "
                          f"Non-finite: {frac_nonfinite*100:.2f}%")
        
        # Summary
        n_compressed = sum(1 for info in result.values() if info['state'] == 'compressed')
        n_decompressed = sum(1 for info in result.values() if info['state'] == 'decompressed')
        print(f"\nTotal: {len(result)} columns, {n_compressed} compressed, {n_decompressed} decompressed")

    # =========================================================================
    # SECTION 6: Introspection & Utilities
    # =========================================================================
    #
    # Methods for inspecting data structure, selecting columns, and describing
    # the AliasDataFrame contents.
    #
    # Key methods:
    # - select_data(): Filter columns by pattern, dtype, etc.
    # - describe_data(): Summary of columns with memory usage
    # - describe_structure(): Overall structure summary
    # - convert_dtypes(): Batch dtype conversion
    #
    # =========================================================================

    def select_data(self, pattern=None, names=None, dtype=None,
                    only_physical=False, only_aliases=False, only_compressed=False,
                    min_memory_mb=None, include_subframes=True):
        """
        Select data columns by pattern, dtype, and other filters.
        
        Parameters
        ----------
        pattern : str, optional
            Regex pattern to match column names
        names : list, optional
            Explicit list of column names to include
        dtype : type or list of types, optional
            Filter by dtype (supports single type or list of types)
        only_physical : bool, default=False
            If True, include only physical columns (exclude aliases)
        only_aliases : bool, default=False
            If True, include only alias columns (exclude physical)
        only_compressed : bool, default=False
            If True, include only compressed columns
        min_memory_mb : float, optional
            Minimum memory size in MB (only for physical columns)
        include_subframes : bool, default=True
            If True, include subframe-referencing aliases
            
        Returns
        -------
        list
            List of column names matching the criteria
        """
        import re as re_module
        
        # Validate mutually exclusive filters
        if only_physical and only_aliases:
            raise ValueError("Filters only_physical and only_aliases are mutually exclusive")
        
        # Compile pattern
        regex = None
        if pattern:
            try:
                regex = re_module.compile(pattern)
            except re_module.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        # Normalize dtype to list
        if dtype is not None:
            if not isinstance(dtype, (list, tuple)):
                dtype = [dtype]
            # Convert to numpy dtype objects for comparison
            dtype = [np.dtype(d) for d in dtype]
        
        result = []
        
        # Physical columns
        for col in self.df.columns:
            # Filter by explicit names
            if names is not None and col not in names:
                continue
            
            # Filter by pattern
            if regex and not regex.search(col):
                continue
            
            # Skip if only_aliases
            if only_aliases:
                continue
            
            # Filter by dtype
            if dtype is not None:
                col_dtype = self.df[col].dtype
                if not any(col_dtype == d for d in dtype):
                    continue
            
            # Filter by memory
            if min_memory_mb is not None:
                memory_mb = self.df[col].memory_usage(deep=True) / (1024 * 1024)
                if memory_mb < min_memory_mb:
                    continue
            
            # Filter by compression
            if only_compressed:
                if col not in self.compression_info or self.compression_info[col].get('state') != 'compressed':
                    continue
            
            result.append(col)
        
        # Alias columns
        if not only_physical:
            for alias_name, alias_expr in self.aliases.items():
                # Filter by explicit names
                if names is not None and alias_name not in names:
                    continue
                
                # Filter by pattern
                if regex and not regex.search(alias_name):
                    continue
                
                # Check if subframe reference
                is_subframe_ref = '.' in alias_expr and alias_expr.split('.')[0] in self._subframes.subframes
                
                # Filter subframes if requested
                if not include_subframes and is_subframe_ref:
                    continue
                
                # Filter by dtype (from schema)
                if dtype is not None:
                    col_info = self.schema['columns'].get(alias_name, {})
                    col_dtype = col_info.get('dtype')
                    if col_dtype is not None:
                        try:
                            col_dtype = np.dtype(col_dtype) if isinstance(col_dtype, str) else col_dtype
                            if not any(col_dtype == d for d in dtype):
                                continue
                        except (TypeError, ValueError):
                            continue
                
                result.append(alias_name)
        
        return result

    def describe_data(self, verbosity=0x03, pattern=None, names=None,
                     only_physical=False, only_aliases=False, only_compressed=False,
                     sort_by='name', as_dict=False, color=False):
        """
        Describe data columns with memory usage and metadata.
        
        Parameters
        ----------
        verbosity : int, default=DATA_SHOW_CORE | DATA_SHOW_STATS (0x03)
            Bitmask controlling output detail:
            - DATA_SHOW_CORE (0x01): name, dtype, memory, shape
            - DATA_SHOW_STATS (0x02): null count, unique values
            - DATA_SHOW_META (0x04): user metadata (if present)
            - DATA_SHOW_SOURCE (0x08): physical/alias/subframe_ref
        pattern : str, optional
            Regex pattern to filter columns
        names : list, optional
            Explicit list of column names
        only_physical : bool, default=False
            If True, show only physical columns
        only_aliases : bool, default=False
            If True, show only alias columns
        only_compressed : bool, default=False
            If True, show only compressed columns
        sort_by : str, default='name'
            Sort by: 'name', 'memory', 'dtype'
        as_dict : bool, default=False
            If True, return dict instead of printing
        color : bool, default=False
            If True, use ANSI colors in output
            
        Returns
        -------
        dict or None
            If as_dict=True, returns structured dict of column info
        """
        # Mask unknown verbosity bits
        verbosity &= self.DATA_SHOW_ALL
        
        # Use select_data for filtering
        selected = self.select_data(
            pattern=pattern, names=names,
            only_physical=only_physical,
            only_aliases=only_aliases,
            only_compressed=only_compressed
        )
        
        # Collect info for each column
        result = {}
        for name in selected:
            is_physical = name in self.df.columns
            is_alias = name in self.aliases
            
            entry = {'name': name}
            
            # Determine dtype
            if is_physical:
                entry['dtype'] = str(self.df[name].dtype)
                entry['memory_bytes'] = self.df[name].memory_usage(deep=True)
                entry['shape'] = self.df[name].shape
                entry['source'] = 'physical'
                
                if verbosity & self.DATA_SHOW_STATS:
                    entry['null_count'] = self.df[name].isna().sum()
                    try:
                        entry['unique_count'] = self.df[name].nunique()
                    except (TypeError, ValueError):
                        entry['unique_count'] = None
            elif is_alias:
                col_info = self.schema['columns'].get(name, {})
                col_dtype = col_info.get('dtype')
                # Convert dtype to string name
                if col_dtype is not None:
                    if isinstance(col_dtype, str):
                        entry['dtype'] = col_dtype
                    else:
                        try:
                            entry['dtype'] = np.dtype(col_dtype).name
                        except (TypeError, ValueError):
                            entry['dtype'] = str(col_dtype)
                else:
                    entry['dtype'] = 'unknown'
                entry['memory_bytes'] = 0
                entry['shape'] = '(computed)'
                
                # Determine if subframe reference
                alias_expr = self.aliases[name]
                is_subframe_ref = '.' in alias_expr and alias_expr.split('.')[0] in self._subframes.subframes
                entry['source'] = 'subframe' if is_subframe_ref else 'alias'
                entry['expr'] = alias_expr
            
            # Check metadata
            col_info = self.schema['columns'].get(name, {})
            if 'metadata' in col_info and col_info['metadata']:
                entry['metadata'] = col_info['metadata']
            
            result[name] = entry
        
        if as_dict:
            return result
        
        if not result:
            print("No columns matching criteria")
            return
        
        # Sort results
        if sort_by == 'memory':
            sorted_names = sorted(result.keys(), key=lambda n: result[n].get('memory_bytes', 0), reverse=True)
        elif sort_by == 'dtype':
            sorted_names = sorted(result.keys(), key=lambda n: result[n].get('dtype', ''))
        else:  # name
            sorted_names = sorted(result.keys())
        
        # Print header
        print("Data Columns:")
        if verbosity & self.DATA_SHOW_CORE:
            print(f"  {'Name':<20} {'Dtype':<10} {'Memory':<10} {'Shape':<15}", end='')
            if verbosity & self.DATA_SHOW_SOURCE:
                print(f" {'Source':<10}", end='')
            print()
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15}", end='')
            if verbosity & self.DATA_SHOW_SOURCE:
                print(f" {'-'*10}", end='')
            print()
        
        # Print each column
        total_memory = 0
        n_physical = 0
        n_aliases = 0
        
        for name in sorted_names:
            info = result[name]
            
            # Core info
            if verbosity & self.DATA_SHOW_CORE:
                dtype_str = info.get('dtype', 'unknown')
                mem_bytes = info.get('memory_bytes', 0)
                total_memory += mem_bytes
                
                if mem_bytes > 0:
                    if mem_bytes > 1024*1024*1024:
                        mem_str = f"{mem_bytes/(1024*1024*1024):.2f} GB"
                    elif mem_bytes > 1024*1024:
                        mem_str = f"{mem_bytes/(1024*1024):.1f} MB"
                    elif mem_bytes > 1024:
                        mem_str = f"{mem_bytes/1024:.1f} KB"
                    else:
                        mem_str = f"{mem_bytes} B"
                else:
                    mem_str = "0 bytes"
                
                shape_str = str(info.get('shape', ''))
                source_str = info.get('source', '')
                
                if info['source'] == 'physical':
                    n_physical += 1
                else:
                    n_aliases += 1
                
                print(f"  {name:<20} {dtype_str:<10} {mem_str:<10} {shape_str:<15}", end='')
                if verbosity & self.DATA_SHOW_SOURCE:
                    print(f" {source_str:<10}", end='')
                print()
            
            # Show expression for aliases/subframes
            if (verbosity & self.DATA_SHOW_SOURCE) and 'expr' in info:
                print(f"  {'':<20} {'→'} {info['expr']}")
            
            # Show stats
            if verbosity & self.DATA_SHOW_STATS and info['source'] == 'physical':
                null_count = info.get('null_count', 0)
                unique_count = info.get('unique_count')
                stats_str = f"Nulls: {null_count:,}"
                if unique_count is not None:
                    stats_str += f", Unique: {unique_count:,}"
                print(f"  {'':<20} {stats_str}")
            
            # Show metadata
            if verbosity & self.DATA_SHOW_META and 'metadata' in info:
                meta = info['metadata']
                for key, value in meta.items():
                    print(f"  {'':<20} {key}: {value}")
        
        # Summary
        if total_memory > 1024*1024*1024:
            mem_str = f"{total_memory/(1024*1024*1024):.2f} GB"
        elif total_memory > 1024*1024:
            mem_str = f"{total_memory/(1024*1024):.1f} MB"
        else:
            mem_str = f"{total_memory/1024:.1f} KB"
        
        print(f"\nTotal: {len(result)} columns ({n_physical} physical: {mem_str}, {n_aliases} lazy)")


    def select_schema(self, pattern=None, dtype=None, has_metadata=False, 
                     is_subframe_ref=False, is_compressed=False):
        """
        Select schema entries by filters.
        
        Parameters
        ----------
        pattern : str, optional
            Regex pattern to match column names
        dtype : type or list of types, optional
            Filter by dtype
        has_metadata : bool, default=False
            If True, include only columns with user metadata
        is_subframe_ref : bool, default=False
            If True, include only subframe-referencing aliases
        is_compressed : bool, default=False
            If True, include only columns with compression info
            
        Returns
        -------
        list
            List of column names matching criteria
        """
        import re as re_module
        
        # Compile pattern
        regex = None
        if pattern:
            try:
                regex = re_module.compile(pattern)
            except re_module.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        # Normalize dtype to list
        if dtype is not None:
            if not isinstance(dtype, (list, tuple)):
                dtype = [dtype]
            dtype = [np.dtype(d) for d in dtype]
        
        result = []
        
        for name, info in self.schema['columns'].items():
            # Filter by pattern
            if regex and not regex.search(name):
                continue
            
            # Filter by dtype
            if dtype is not None:
                col_dtype = info.get('dtype')
                if col_dtype is not None:
                    try:
                        col_dtype = np.dtype(col_dtype) if isinstance(col_dtype, str) else col_dtype
                        if not any(col_dtype == d for d in dtype):
                            continue
                    except (TypeError, ValueError):
                        continue
                else:
                    continue
            
            # Filter by metadata
            if has_metadata:
                if 'metadata' not in info or not info['metadata']:
                    continue
            
            # Filter by subframe reference
            if is_subframe_ref:
                expr = info.get('expr', '')
                is_subf_ref = '.' in expr and expr.split('.')[0] in self._subframes.subframes
                if not is_subf_ref:
                    continue
            
            # Filter by compression
            if is_compressed:
                if name not in self.compression_info:
                    continue
            
            result.append(name)
        
        return result

    def describe_schema(self, verbosity=0x0F, sections=None, as_dict=False):
        """
        Describe overall schema structure.
        
        Parameters
        ----------
        verbosity : int, default=SCHEMA_SHOW_ALL (0x0F)
            Bitmask controlling output detail:
            - SCHEMA_SHOW_CORE (0x01): basic overview
            - SCHEMA_SHOW_COMPRESSION (0x02): compression rules
            - SCHEMA_SHOW_METADATA (0x04): user metadata
            - SCHEMA_SHOW_SUBFRAMES (0x08): recursive schemas
        sections : list, optional
            Specific sections to show: ['columns', 'compression', 'subframes']
        as_dict : bool, default=False
            If True, return dict instead of printing
            
        Returns
        -------
        dict or None
            If as_dict=True, returns structured dict of schema info
        """
        # Mask unknown verbosity bits
        verbosity &= self.SCHEMA_SHOW_ALL
        
        # Determine which sections to include
        if sections is None:
            sections = ['columns', 'compression', 'subframes']
        
        result = {
            'version': self.schema.get('__meta__', {}).get('version', 'unknown'),
            'columns': {},
            'compression': {},
            'subframes': {}
        }
        
        # Columns section
        if 'columns' in sections:
            n_physical = 0
            n_aliases = 0
            n_subframe = 0
            n_with_meta = 0
            dtype_counts = {}
            physical_memory = 0
            
            for name, info in self.schema['columns'].items():
                is_alias = 'expr' in info and info['expr'] is not None
                is_subframe = False
                
                if is_alias:
                    expr = info['expr']
                    is_subframe = '.' in expr and expr.split('.')[0] in self._subframes.subframes
                    if is_subframe:
                        n_subframe += 1
                    else:
                        n_aliases += 1
                else:
                    n_physical += 1
                    # Calculate memory for physical columns
                    if name in self.df.columns:
                        physical_memory += self.df[name].memory_usage(deep=True)
                
                # Count dtypes
                dtype = info.get('dtype')
                if dtype is not None:
                    dtype_str = np.dtype(dtype).name if not isinstance(dtype, str) else dtype
                    dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                
                # Count metadata
                if 'metadata' in info and info['metadata']:
                    n_with_meta += 1
            
            result['columns'] = {
                'total': len(self.schema['columns']),
                'physical': n_physical,
                'aliases': n_aliases,
                'subframe_refs': n_subframe,
                'with_metadata': n_with_meta,
                'dtypes': dtype_counts,
                'physical_memory_bytes': physical_memory
            }
        
        # Compression section
        if 'compression' in sections:
            comp_info = {k: v for k, v in self.compression_info.items() if k != '__meta__'}
            n_compressed = sum(1 for v in comp_info.values() if v.get('state') == 'compressed')
            n_decompressed = sum(1 for v in comp_info.values() if v.get('state') == 'decompressed')
            
            result['compression'] = {
                'total': len(comp_info),
                'compressed': n_compressed,
                'decompressed': n_decompressed
            }
        
        # Subframes section
        if 'subframes' in sections:
            subframes_info = {}
            for name, sf_data in self._subframes.subframes.items():
                sf_adf = sf_data['frame']
                subframes_info[name] = {
                    'n_columns': len(sf_adf.df.columns),
                    'index': sf_data['index'],
                    'n_rows': len(sf_adf.df)
                }
            result['subframes'] = subframes_info
        
        if as_dict:
            return result
        
        # Print formatted output
        print("Schema Overview:")
        print("=" * 70)
        print(f"Version: {result['version']}")
        print()
        
        # Columns overview
        if 'columns' in sections and (verbosity & self.SCHEMA_SHOW_CORE):
            col_info = result['columns']
            mem_bytes = col_info['physical_memory_bytes']
            if mem_bytes > 1024*1024*1024:
                mem_str = f"{mem_bytes/(1024*1024*1024):.2f} GB"
            elif mem_bytes > 1024*1024:
                mem_str = f"{mem_bytes/(1024*1024):.1f} MB"
            else:
                mem_str = f"{mem_bytes/1024:.1f} KB"
            
            print(f"Columns:        {col_info['total']} total")
            print(f"  Physical:     {col_info['physical']} ({mem_str} in memory)")
            print(f"  Aliases:      {col_info['aliases']} (computed on demand)")
            print(f"  Subframe refs: {col_info['subframe_refs']} (lazy join)")
            if verbosity & self.SCHEMA_SHOW_METADATA:
                print(f"  With metadata: {col_info['with_metadata']}")
            print()
            
            print("Dtypes:")
            for dtype, count in sorted(col_info['dtypes'].items(), key=lambda x: -x[1]):
                print(f"  {dtype}: {count} columns")
            print()
        
        # Compression overview
        if 'compression' in sections and (verbosity & self.SCHEMA_SHOW_COMPRESSION):
            comp_info = result['compression']
            print(f"Compression:    {comp_info['total']} columns")
            print(f"  Compressed:   {comp_info['compressed']}")
            print(f"  Decompressed: {comp_info['decompressed']}")
            print()
        
        # Subframes overview
        if 'subframes' in sections and (verbosity & self.SCHEMA_SHOW_SUBFRAMES):
            subframes_info = result['subframes']
            print(f"Subframes:      {len(subframes_info)}")
            for name, info in subframes_info.items():
                index_str = ', '.join(f"'{idx}'" for idx in info['index'])
                print(f"  {name}: {info['n_columns']} columns, "
                      f"{info['n_rows']} rows, index=[{index_str}]")
            if subframes_info:
                print()

    # =========================================================================
    # SECTION 5: Schema Persistence (JSON / ROOT / Parquet)
    # =========================================================================
    #
    # Methods for exporting, saving, loading, and applying schemas.
    # Schema can be embedded in ROOT files or saved alongside Parquet files.
    #
    # Key methods:
    # - export_schema(): Get JSON-safe schema dict
    # - save_schema() / load_schema(): JSON file I/O
    # - save_schema_to_root() / load_schema_from_root(): ROOT file embedding
    # - save_schema_to_parquet_metadata(): Parquet sidecar file
    # - apply_schema(): Apply loaded schema to new DataFrame
    #
    # =========================================================================

    def export_schema(self):
        """
        Export schema as JSON-safe dictionary (v2 format).
        
        Returns
        -------
        dict
            JSON-safe schema dictionary
        """
        return self.export_schema_v2(include_precision_stats=True)

    def save_schema(self, path):
        """
        Save schema to JSON file (v2 format).
        
        Parameters
        ----------
        path : str
            Path to save schema JSON file
        """
        self.save_schema_v2(path, include_precision_stats=True)

    @staticmethod
    def load_schema(path):
        """
        Load schema from JSON file (handles v1 and v2 formats).
        
        Parameters
        ----------
        path : str
            Path to schema JSON file
            
        Returns
        -------
        dict
            Schema dictionary
        """
        return AliasDataFrame.load_schema_v2(path)

    # =========================================================================
    # Schema Export v2: Enhanced methods with groups, metadata, smart formatting
    # =========================================================================

    def set_groups(self, groups):
        """
        Define column groups for schema organization.
        
        Groups control the ordering of columns in exported schemas
        and serve as logical documentation (like comments).
        
        Parameters
        ----------
        groups : dict
            {group_name: [column_names], ...}
            
        Returns
        -------
        AliasDataFrame
            self for method chaining
            
        Example
        -------
        >>> adf.set_groups({
        ...     "coordinates": ["x", "y", "z", "r", "phi"],
        ...     "calibtrack": ["dy_TrackFit0", "dz_TrackFit0"],
        ...     "cuts": ["isOK", "isOKGB"]
        ... })
        """
        self._schema['groups'] = groups
        return self

    def get_groups(self):
        """
        Get current column groups.
        
        Returns
        -------
        dict
            {group_name: [column_names], ...} or empty dict
        """
        return self._schema.get('groups', {})

    def set_column_metadata(self, column, **metadata):
        """
        Set metadata for a column. Supports any key-value pairs.
        
        Parameters
        ----------
        column : str
            Column name (physical or alias)
        **metadata : 
            Arbitrary metadata fields (unit, axisLabel, description, etc.)
            
        Returns
        -------
        AliasDataFrame
            self for method chaining
            
        Example
        -------
        >>> adf.set_column_metadata("pt", unit="GeV/c", axisLabel="p_{T} (GeV/c)")
        >>> adf.set_column_metadata("x", 
        ...     unit="cm", 
        ...     axisLabel="x (cm)",
        ...     description="TPC cluster x position",
        ...     range=[-250, 250]
        ... )
        """
        if column not in self._schema['columns']:
            self._schema['columns'][column] = {}
        
        for key, value in metadata.items():
            self._schema['columns'][column][key] = value
        
        return self

    def set_columns_metadata(self, metadata):
        """
        Set metadata for multiple columns at once.
        
        Parameters
        ----------
        metadata : dict
            {column_name: {field: value, ...}, ...}
            Supports any metadata fields per column.
            
        Returns
        -------
        AliasDataFrame
            self for method chaining
            
        Example
        -------
        >>> adf.set_columns_metadata({
        ...     "x": {"unit": "cm", "axisLabel": "x (cm)", "range": [-250, 250]},
        ...     "y": {"unit": "cm", "axisLabel": "y (cm)"},
        ...     "pt": {"unit": "GeV/c", "axisLabel": "p_{T} (GeV/c)"}
        ... })
        """
        for column, col_metadata in metadata.items():
            self.set_column_metadata(column, **col_metadata)
        return self

    def get_column_metadata(self, column):
        """
        Get metadata for a column.
        
        Parameters
        ----------
        column : str
            Column name
            
        Returns
        -------
        dict
            Column metadata (excluding dtype and expr)
        """
        col_info = self._schema.get('columns', {}).get(column, {})
        # Return all fields except dtype and expr
        return {k: v for k, v in col_info.items() if k not in ('dtype', 'expr', 'constant')}

    def export_schema_v2(self, include_precision_stats=False, include_state=True,
                         include_subframes=True, within_group_sort="schema"):
        """
        Export schema as JSON-safe dictionary (v2 format).
        
        Features:
        - Uniform object format for all columns: {"dtype": "..."}
        - No "expr": null for physical columns
        - Groups preserved (simple named lists of column names)
        - Column metadata preserved (unit, axisLabel, etc.)
        - Recursive subframe schemas (nested)
        - Compression always included (definitions required for data interpretation)
        
        Schema v2 canonical order:
        1. __meta__     - version, creation info
        2. columns      - column definitions (physical + aliases)
        3. groups       - logical column groupings (if present)
        4. compression  - storage/compression configuration
        5. subframes    - hierarchical structure (related tables)
        
        Parameters
        ----------
        include_precision_stats : bool, default=False
            Whether to include precision statistics (RMSE, max_error, etc.) in 
            compression section. The compression definitions (expressions, dtypes,
            state) are always included.
        include_state : bool, default=True
            Whether to include runtime state fields (state, original_removed).
            Set to False for "definition-only" schemas suitable for editing/sharing.
            Set to True for "record" schemas that capture current data state.
        include_subframes : bool, default=True
            Whether to include recursive subframe schemas
        within_group_sort : str, default="schema"
            "schema" - preserve order as listed in groups
            "alphabetic" - sort alphabetically within each group
            
        Returns
        -------
        dict
            JSON-safe schema dictionary with canonical key ordering
        """
        from collections import OrderedDict
        result = OrderedDict()
        
        # 1. __meta__ section (always first)
        result['__meta__'] = {
            'schema_version': SCHEMA_VERSION_V2,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'schema_id': self._schema.get('__meta__', {}).get('schema_id')
        }
        
        # 2. Columns section
        columns = OrderedDict()
        schema_columns = self._schema.get('columns', {})
        groups = self._schema.get('groups', {})
        compression_info = self._schema.get('compression', {})
        
        # Build set of compressed column names (e.g., dy_c) to exclude in definition mode
        compressed_col_names = set()
        compression_targets = set()  # Original column names (e.g., dy)
        if not include_state:
            for orig_col, comp_info in compression_info.items():
                if orig_col == '__meta__':
                    continue
                compressed_col_names.add(comp_info.get('compressed_col', f'{orig_col}_c'))
                compression_targets.add(orig_col)
        
        # Physical columns from DataFrame first
        for col in self.df.columns:
            # In definition mode, skip compressed storage columns (e.g., dy_c)
            # They don't exist in fresh data
            if not include_state and col in compressed_col_names:
                continue
            col_info = schema_columns.get(col, {})
            columns[col] = _export_column_spec_v2(col, col_info, self.df)
        
        # Aliases from schema (not in DataFrame)
        for col, col_info in schema_columns.items():
            if col not in columns:
                # In definition mode, compression targets should be exported as physical columns
                # (no expr), because in fresh data they ARE physical columns
                if not include_state and col in compression_targets:
                    # Export as physical column with decompressed dtype
                    decompressed_dtype = compression_info.get(col, {}).get('decompressed_dtype')
                    physical_info = col_info.copy()
                    physical_info.pop('expr', None)  # Remove alias expression
                    if decompressed_dtype:
                        physical_info['dtype'] = decompressed_dtype
                    columns[col] = _export_column_spec_v2(col, physical_info)
                else:
                    columns[col] = _export_column_spec_v2(col, col_info)
        
        # Order by groups
        columns = _order_columns_by_groups(columns, groups, within_group_sort)
        result['columns'] = columns
        
        # 3. Groups section (if present) - simple named lists of column names
        if groups:
            result['groups'] = groups
        
        # 4. Compression section - ALWAYS included when present (required for data interpretation)
        if self._schema.get('compression'):
            comp = OrderedDict()
            for name, info in self._schema['compression'].items():
                if name == '__meta__':
                    # Preserve compression metadata
                    comp['__meta__'] = copy.deepcopy(info)
                    continue
                
                # Build compression entry with essential fields
                entry = {}
                
                # Required fields for data interpretation
                if 'compressed_col' in info:
                    entry['compressed_col'] = info['compressed_col']
                if 'compress_expr' in info:
                    entry['compress_expr'] = info['compress_expr']
                if 'decompress_expr' in info:
                    entry['decompress_expr'] = info['decompress_expr']
                
                # Convert dtypes to strings using helper
                for dtype_field in ['compressed_dtype', 'decompressed_dtype']:
                    if dtype_field in info and info[dtype_field] is not None:
                        entry[dtype_field] = _dtype_to_str(info[dtype_field])
                
                # State and flags (optional - for record schemas, not definition schemas)
                if include_state:
                    if 'state' in info:
                        entry['state'] = info['state']
                    if 'original_removed' in info:
                        entry['original_removed'] = info['original_removed']
                
                # Optional: precision statistics (can be verbose)
                if include_precision_stats and 'precision' in info:
                    entry['precision'] = copy.deepcopy(info['precision'])
                
                # Optional: monitor info (without non-serializable func)
                if 'monitor' in info and info['monitor']:
                    monitor = info['monitor']
                    if 'func' in monitor:
                        entry['monitor'] = {k: v for k, v in monitor.items() if k != 'func'}
                    else:
                        entry['monitor'] = copy.deepcopy(monitor)
                
                comp[name] = entry
            
            if comp:
                result['compression'] = comp
        
        # 5. Subframes section (last - hierarchical structure)
        if include_subframes:
            subframes = OrderedDict()
            
            # Try subframe registry first
            if hasattr(self, '_subframes'):
                for name, entry in self._subframes.items():
                    subframes[name] = _export_subframe_schema_v2(entry, include_precision_stats, include_state)
            
            # Fall back to schema if registry is empty
            if not subframes and self._schema.get('subframes'):
                for name, info in self._schema.get('subframes', {}).items():
                    index_cols = info.get('index', [])
                    if isinstance(index_cols, str):
                        index_cols = [index_cols]
                    subframes[name] = {'index': index_cols}
            
            if subframes:
                result['subframes'] = subframes
        
        # 6. Fit metadata section (Phase 12.4b3)
        if hasattr(self, '_fit_metadata') and self._fit_metadata:
            result['fit_metadata'] = copy.deepcopy(self._fit_metadata)
        
        return result

    def export_definition_schema(self, **kwargs):
        """
        Export blueprint/definition schema without runtime state.
        
        This exports a schema suitable for:
        - Sharing as a template/recipe
        - Applying to fresh data
        - Version control
        - Human editing
        
        Compression targets are exported as physical columns (no expr),
        compressed storage columns are NOT exported, and no state fields
        are included.
        
        This is a convenience wrapper for:
            export_schema_v2(include_state=False, **kwargs)
            
        Parameters
        ----------
        **kwargs
            Additional arguments passed to export_schema_v2
            (include_precision_stats, include_subframes, within_group_sort)
            
        Returns
        -------
        dict
            Definition schema dictionary
            
        See Also
        --------
        export_record_schema : Export with runtime state
        export_schema_v2 : Full export with all options
        """
        return self.export_schema_v2(include_state=False, **kwargs)

    def export_record_schema(self, **kwargs):
        """
        Export schema with current runtime state (snapshot).
        
        This exports a schema that captures:
        - Current compression state (compressed/decompressed)
        - Which columns have been removed (original_removed)
        - Aliases reflecting current decompression expressions
        
        Suitable for:
        - Saving exact current state
        - Reloading to identical DataFrame configuration
        - Debugging/diagnostics
        
        This is a convenience wrapper for:
            export_schema_v2(include_state=True, **kwargs)
            
        Parameters
        ----------
        **kwargs
            Additional arguments passed to export_schema_v2
            (include_precision_stats, include_subframes, within_group_sort)
            
        Returns
        -------
        dict
            Record schema dictionary with runtime state
            
        See Also
        --------
        export_definition_schema : Export without runtime state
        export_schema_v2 : Full export with all options
        """
        return self.export_schema_v2(include_state=True, **kwargs)

    def save_schema_v2(self, path, include_precision_stats=False, include_state=True,
                       include_subframes=True, indent=2, max_line_length=100, 
                       within_group_sort="schema"):
        """
        Save schema to JSON file (v2 format).
        
        Short entries stay on one line, long entries are split.
        Columns ordered by groups, then ungrouped columns.
        Compression definitions are always saved (required for data interpretation).
        
        Parameters
        ----------
        path : str
            Output file path
        include_precision_stats : bool, default=False
            Whether to include precision statistics (RMSE, max_error, etc.) in
            compression section. Compression definitions are always included.
        include_state : bool, default=True
            Whether to include runtime state fields (state, original_removed).
            Set to False for "definition-only" schemas suitable for editing/sharing.
        include_subframes : bool, default=True
            Whether to include recursive subframe schemas
        indent : int, default=2
            Indentation for JSON formatting
        max_line_length : int, default=100
            Maximum line length before splitting
        within_group_sort : str, default="schema"
            "schema" - preserve order as listed in groups
            "alphabetic" - sort alphabetically within each group
        """
        schema = self.export_schema_v2(
            include_precision_stats=include_precision_stats,
            include_state=include_state,
            include_subframes=include_subframes,
            within_group_sort=within_group_sort
        )
        
        # Use smart formatting
        json_str = _format_json_smart(schema, indent=indent, max_line_length=max_line_length)
        
        with open(path, 'w') as f:
            f.write(json_str)

    @staticmethod
    def load_schema_v2(path):
        """
        Load schema from JSON file (handles v1 and v2 formats).
        
        Backward compatible: normalizes old format to new format.
        
        Parameters
        ----------
        path : str
            Path to schema JSON file
            
        Returns
        -------
        dict
            Normalized schema dictionary
        """
        with open(path, 'r') as f:
            schema = json.load(f)
        
        # Detect version
        meta = schema.get('__meta__', {})
        version = meta.get('schema_version', 1)
        # Handle string versions like '2.0'
        if isinstance(version, str):
            try:
                version = float(version)
            except (ValueError, TypeError):
                version = 1
        
        if version >= 2:
            # Already v2 format
            return schema
        
        # Normalize v1 to v2 format
        normalized = {
            '__meta__': {
                'schema_version': 2,
                'schema_id': meta.get('schema_id'),
                'created_at': meta.get('created_at')
            }
        }
        
        # Normalize columns (remove "expr": null)
        if 'columns' in schema:
            normalized['columns'] = {}
            for col, spec in schema['columns'].items():
                if isinstance(spec, str):
                    # Legacy compact format
                    normalized['columns'][col] = {'dtype': spec}
                elif isinstance(spec, dict):
                    # Remove "expr": null
                    clean_spec = {k: v for k, v in spec.items() if not (k == 'expr' and v is None)}
                    normalized['columns'][col] = clean_spec
                else:
                    normalized['columns'][col] = spec
        
        # Copy other sections
        if 'groups' in schema:
            normalized['groups'] = schema['groups']
        
        if 'subframes' in schema:
            normalized['subframes'] = {}
            for name, info in schema['subframes'].items():
                # Ensure index is a list
                index_cols = info.get('index', [])
                if isinstance(index_cols, str):
                    index_cols = [index_cols]
                # Repair corrupted indices (from char iteration bug)
                index_cols = _repair_index_columns(index_cols, name)
                normalized['subframes'][name] = {'index': index_cols}
                # Copy columns if present
                if 'columns' in info:
                    normalized['subframes'][name]['columns'] = info['columns']
        
        if 'compression' in schema:
            normalized['compression'] = schema['compression']
        
        return normalized

    def apply_schema(self, schema, validate=True, warn_missing=True):
        """
        Apply schema to current AliasDataFrame.
        
        Applies dtypes, aliases, compression info, and metadata from schema.
        
        Parameters
        ----------
        schema : dict
            Schema dictionary (from export_schema or load_schema)
        validate : bool, default=True
            If True, validate schema consistency
        warn_missing : bool, default=True
            If True, warn about columns in schema but not in data
        """
        # Check for old-format definition schemas (compression target has expr)
        # This indicates a schema that was exported with runtime state when it
        # should have been exported as a definition schema (include_state=False)
        compression_targets = set(schema.get('compression', {}).keys()) - {'__meta__'}
        columns_info = schema.get('columns', {})
        
        for target in compression_targets:
            if target in columns_info and columns_info[target].get('expr'):
                warnings.warn(
                    f"Column '{target}' is a compression target but has 'expr' in schema. "
                    f"This indicates an old-format definition schema that was exported with "
                    f"runtime state. Re-export with include_state=False (or use "
                    f"export_definition_schema()) to create a proper definition schema. "
                    f"The schema will still load, but may cause issues when compressing.",
                    DeprecationWarning,
                    stacklevel=2
                )
        
        # Apply dtypes to physical columns
        if 'columns' in schema:
            for name, info in schema['columns'].items():
                # Skip aliases (handled separately)
                if 'expr' in info and info['expr'] is not None:
                    continue
                
                # Apply dtype to physical column
                if name in self.df.columns:
                    target_dtype = info.get('dtype')
                    if target_dtype:
                        try:
                            current_dtype = self.df[name].dtype
                            target_dtype_obj = np.dtype(target_dtype)
                            if current_dtype != target_dtype_obj:
                                self.df[name] = self.df[name].astype(target_dtype_obj)
                        except (TypeError, ValueError) as e:
                            if warn_missing:
                                warnings.warn(f"Failed to apply dtype {target_dtype} to column {name}: {e}")
                elif warn_missing:
                    warnings.warn(f"Column '{name}' in schema but not in DataFrame")
        
        # Apply aliases
        if 'columns' in schema:
            for name, info in schema['columns'].items():
                expr = info.get('expr')
                if expr is not None:
                    dtype = info.get('dtype')
                    if dtype:
                        try:
                            dtype = np.dtype(dtype)
                        except (TypeError, ValueError):
                            dtype = None
                    
                    # Add alias (don't overwrite if already materialized as physical column)
                    if name not in self.df.columns:
                        self.add_alias(name, expr, dtype=dtype)
                    elif validate:
                        # Column exists physically but schema says it's an alias
                        # Mark as materialized alias
                        if name not in self.aliases:
                            self.aliases[name] = expr
        
        # Update compression info
        if 'compression' in schema:
            for name, info in schema['compression'].items():
                if name != '__meta__':
                    self.compression_info[name] = info
        
        # Update schema metadata
        self.update_schema(schema, validate=validate)
        
        # Restore fit metadata if present (Phase 12.4b3)
        if 'fit_metadata' in schema:
            if hasattr(self, '_fit_metadata') and self._fit_metadata:
                existing = list(self._fit_metadata.keys())
                warnings.warn(
                    f"Overwriting existing fit metadata: {existing}",
                    UserWarning
                )
            self._fit_metadata = copy.deepcopy(schema['fit_metadata'])

    @classmethod
    def from_schema(cls, schema):
        """
        Create empty AliasDataFrame from schema template.
        
        Parameters
        ----------
        schema : dict
            Schema dictionary
            
        Returns
        -------
        AliasDataFrame
            Empty AliasDataFrame with schema structure
        """
        # Create empty DataFrame with physical columns
        columns_data = {}
        if 'columns' in schema:
            for name, info in schema['columns'].items():
                # Only create physical columns (not aliases)
                if 'expr' not in info or info['expr'] is None:
                    dtype = info.get('dtype', 'float64')
                    try:
                        dtype_obj = np.dtype(dtype)
                        columns_data[name] = pd.Series([], dtype=dtype_obj)
                    except (TypeError, ValueError):
                        columns_data[name] = pd.Series([])
        
        df = pd.DataFrame(columns_data)
        adf = cls(df)
        
        # Apply full schema (aliases, compression, metadata)
        adf.apply_schema(schema, validate=False, warn_missing=False)
        
        return adf

    def convert_dtypes(self, dtype_map):
        """
        Convert dtypes for multiple columns.
        
        Parameters
        ----------
        dtype_map : dict
            Mapping of column_name → target_dtype
        """
        for col, target_dtype in dtype_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(target_dtype)
                    # Update schema
                    if col in self.schema['columns']:
                        self.schema['columns'][col]['dtype'] = np.dtype(target_dtype).name
                except (TypeError, ValueError) as e:
                    warnings.warn(f"Failed to convert {col} to {target_dtype}: {e}")
            else:
                warnings.warn(f"Column '{col}' not found in DataFrame")

    def convert_dtypes_pattern(self, pattern, target_dtype):
        """
        Convert dtypes for columns matching pattern.
        
        Parameters
        ----------
        pattern : str
            Regex pattern to match column names
        target_dtype : type
            Target dtype to convert to
        """
        matching_cols = self.select_data(pattern=pattern, only_physical=True)
        dtype_map = {col: target_dtype for col in matching_cols}
        self.convert_dtypes(dtype_map)
    def describe_structure(self, verbosity=None, return_dict=False):
        """
        Print or return comprehensive structure summary of the AliasDataFrame.
        
        Uses bitmask flags for fine-grained control over output sections.
        
        Parameters
        ----------
        verbosity : int, optional
            Bitmask controlling which sections to display. Use VERBOSITY_* constants.
            Default: VERBOSE_DEFAULT (basic + dtypes + aliases + compression + subframes)
        return_dict : bool, optional
            If True, return structured dict instead of printing (default: False)
        
        Returns
        -------
        dict or None
            If return_dict=True, returns structure dict. Otherwise prints and returns None.
        
        Verbosity Flags
        ---------------
        VERBOSITY_BASIC        (0x01): rows/columns/memory
        VERBOSITY_DTYPES       (0x02): columns grouped by dtype
        VERBOSITY_ALIASES      (0x04): list aliases (short)
        VERBOSITY_ALIASES_FULL (0x08): full alias definitions
        VERBOSITY_COMPRESSION  (0x10): compression summary
        VERBOSITY_COMP_FULL    (0x20): full compression info
        VERBOSITY_SUBFRAMES    (0x40): list subframes
        VERBOSITY_METADATA     (0x80): raw metadata dump
        
        Presets: VERBOSE_MINIMAL, VERBOSE_DEFAULT, VERBOSE_FULL
        
        Examples
        --------
        >>> adf.describe_structure()  # Default output
        >>> adf.describe_structure(VERBOSE_FULL)  # Everything
        >>> adf.describe_structure(VERBOSITY_ALIASES | VERBOSITY_COMPRESSION)  # Just these
        >>> info = adf.describe_structure(return_dict=True)  # Programmatic access
        """
        if verbosity is None:
            verbosity = VERBOSE_DEFAULT
        
        info = {}
        lines = []
        
        # =====================================================================
        # BASIC: rows/columns/memory
        # =====================================================================
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        total_memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        
        info['n_rows'] = n_rows
        info['n_columns'] = n_cols
        info['total_memory_mb'] = total_memory_mb
        
        if verbosity & VERBOSITY_BASIC:
            lines.append("AliasDataFrame Structure")
            lines.append("=" * 50)
            lines.append("")
            lines.append(f"DataFrame: {n_rows:,} rows × {n_cols} columns")
            lines.append(f"Memory: {total_memory_mb:.1f} MB")
            lines.append("")
        
        # =====================================================================
        # DTYPES: columns grouped by dtype
        # =====================================================================
        dtype_groups = {}
        dtype_memory = {}
        for col in self.df.columns:
            dtype_name = str(self.df[col].dtype)
            if dtype_name not in dtype_groups:
                dtype_groups[dtype_name] = []
                dtype_memory[dtype_name] = 0
            dtype_groups[dtype_name].append(col)
            dtype_memory[dtype_name] += self.df[col].memory_usage(deep=True) / 1024 / 1024
        
        info['dtype_groups'] = dtype_groups
        info['dtype_memory_mb'] = dtype_memory
        
        if verbosity & VERBOSITY_DTYPES:
            lines.append("Columns by dtype:")
            for dtype_name in sorted(dtype_groups.keys()):
                cols = dtype_groups[dtype_name]
                mem = dtype_memory[dtype_name]
                lines.append(f"  {dtype_name}: {len(cols)} columns ({mem:.1f} MB)")
            lines.append("")
        
        # =====================================================================
        # ALIASES: list aliases
        # =====================================================================
        n_aliases = len(self.aliases)
        decompression_aliases = []
        regular_aliases = []
        
        for alias, expr in self.aliases.items():
            # Check if this is a decompression alias
            is_decompression = any(
                info_item.get('decompress_expr') == expr 
                for info_item in self.compression_info.values()
                if isinstance(info_item, dict)
            )
            dtype_obj = self.alias_dtypes.get(alias)
            #dtype_str = dtype_obj.__name__ if dtype_obj else 'unspecified'
            dtype_str = dtype_obj.__name__ if hasattr(dtype_obj, '__name__') else str(dtype_obj) if dtype_obj else 'unspecified'
            
            if is_decompression:
                decompression_aliases.append((alias, expr, dtype_str))
            else:
                regular_aliases.append((alias, expr, dtype_str))
        
        info['n_aliases'] = n_aliases
        info['aliases'] = {
            'regular': regular_aliases,
            'decompression': decompression_aliases
        }
        
        if verbosity & VERBOSITY_ALIASES:
            if n_aliases > 0:
                lines.append(f"Aliases: {n_aliases} defined")
                # Show short list
                for alias, expr, dtype_str in regular_aliases[:5]:
                    expr_short = expr[:40] + "..." if len(expr) > 40 else expr
                    lines.append(f"  - {alias}: {expr_short} → {dtype_str}")
                if len(regular_aliases) > 5:
                    lines.append(f"  ... and {len(regular_aliases) - 5} more")
                
                if decompression_aliases:
                    lines.append(f"  Decompression aliases: {len(decompression_aliases)}")
                    for alias, expr, dtype_str in decompression_aliases[:3]:
                        expr_short = expr[:30] + "..." if len(expr) > 30 else expr
                        lines.append(f"    - {alias}: {expr_short} → {dtype_str}")
                lines.append("")
        
        if verbosity & VERBOSITY_ALIASES_FULL:
            if n_aliases > 0:
                lines.append("Full Alias Definitions:")
                for alias, expr, dtype_str in regular_aliases + decompression_aliases:
                    lines.append(f"  {alias} = {expr}  [{dtype_str}]")
                lines.append("")
        
        # =====================================================================
        # COMPRESSION: compression summary
        # =====================================================================
        compressed_columns = []
        for col_name, col_info in self.compression_info.items():
            if col_name == "__meta__":
                continue
            if isinstance(col_info, dict) and 'compressed_col' in col_info:
                compressed_col = col_info.get('compressed_col', f'{col_name}_c')
                compressed_dtype = col_info.get('compressed_dtype', 'unknown')
                decompressed_dtype = col_info.get('decompressed_dtype', 'unknown')
                rmse = col_info.get('precision', {}).get('rmse')
                compressed_columns.append({
                    'name': col_name,
                    'compressed_col': compressed_col,
                    'compressed_dtype': compressed_dtype,
                    'decompressed_dtype': decompressed_dtype,
                    'rmse': rmse,
                    'info': col_info
                })
        
        info['compression'] = compressed_columns
        
        if verbosity & VERBOSITY_COMPRESSION:
            if compressed_columns:
                lines.append(f"Compression: {len(compressed_columns)} columns")
                for comp in compressed_columns[:5]:
                    rmse_str = f", RMSE={comp['rmse']:.4f}" if comp['rmse'] else ""
                    lines.append(
                        f"  - {comp['name']}: COMPRESSED "
                        f"({comp['compressed_col']}, {comp['compressed_dtype']} → "
                        f"{comp['decompressed_dtype']}{rmse_str})"
                    )
                if len(compressed_columns) > 5:
                    lines.append(f"  ... and {len(compressed_columns) - 5} more")
                lines.append("")
        
        if verbosity & VERBOSITY_COMP_FULL:
            if compressed_columns:
                lines.append("Full Compression Details:")
                for comp in compressed_columns:
                    col_info = comp['info']
                    lines.append(f"  {comp['name']}:")
                    lines.append(f"    Compressed as: {comp['compressed_col']} ({comp['compressed_dtype']})")
                    lines.append(f"    Expression: {col_info.get('compress_expr', 'N/A')}")
                    lines.append(f"    Decompression: {col_info.get('decompress_expr', 'N/A')} → {comp['decompressed_dtype']}")
                    if 'precision' in col_info and 'rmse' in col_info['precision']:
                        prec = col_info['precision']
                        lines.append(f"    Precision: RMSE={prec['rmse']:.6f}, Max={prec.get('max_error', 0):.6f}")
                lines.append("")
        
        # =====================================================================
        # SUBFRAMES: list subframes
        # =====================================================================
        subframes_info = []
        for sf_name, entry in self._subframes.items():
            sf = entry['frame']
            index_cols = entry['index']
            subframes_info.append({
                'name': sf_name,
                'rows': len(sf.df),
                'columns': len(sf.df.columns),
                'memory_mb': sf.df.memory_usage(deep=True).sum() / 1e6,
                'index_columns': index_cols
            })
        
        info['subframes'] = subframes_info
        
        if verbosity & VERBOSITY_SUBFRAMES:
            if subframes_info:
                lines.append(f"Subframes: {len(subframes_info)}")
                for sf in subframes_info:
                    index_str = sf['index_columns'] if isinstance(sf['index_columns'], str) else ', '.join(sf['index_columns'])
                    #lines.append(f"  - {sf['name']}: {sf['rows']:,} rows × {sf['columns']} cols, index={index_str}")
                    lines.append(f"  - {sf['name']}: {sf['rows']:,} rows × {sf['columns']} cols, {sf['memory_mb']:.1f} MB, index={index_str}")
                lines.append("")
        
        # =====================================================================
        # METADATA: raw metadata dump
        # =====================================================================
        if verbosity & VERBOSITY_METADATA:
            lines.append("Raw Metadata:")
            lines.append(f"  Aliases: {len(self.aliases)}")
            lines.append(f"  Alias dtypes: {len(self.alias_dtypes)}")
            lines.append(f"  Constant aliases: {len(self.constant_aliases)}")
            lines.append(f"  Compression entries: {len(self.compression_info) - 1}")  # -1 for __meta__
            if "__meta__" in self.compression_info:
                meta = self.compression_info["__meta__"]
                lines.append(f"  Schema version: {meta.get('schema_version', 'unknown')}")
            lines.append("")
        
        # =====================================================================
        # Return or print
        # =====================================================================
        if return_dict:
            return info
        
        if lines:
            print("\n".join(lines))
        return None


    # =========================================================================
    # PHASE B: EXPLICIT SUBFRAME API
    # =========================================================================
    
    def subframe(self, name):
        """
        Access a subframe by name with version compatibility.
        
        Args:
            name: Name of subframe (e.g., 'DITS0FitSide')
        
        Returns:
            AliasDataFrame or dict: The subframe data
        
        Raises:
            KeyError: If subframe doesn't exist
        
        Example:
            sf = adf.subframe('DITS0FitSide')
            print(len(sf.df))
        """
        if not hasattr(self, '_subframes') or not hasattr(self._subframes, 'subframes'):
            raise AttributeError("No subframes loaded. Use read_tree() with subframes.")
        
        if name not in self._subframes.subframes:
            available = list(self._subframes.subframes.keys())
            raise KeyError(
                f"Subframe '{name}' not found. "
                f"Available subframes: {available}"
            )
        
        sf_data = self._subframes.subframes[name]
        
        # Handle version compatibility
        if hasattr(sf_data, 'frame'):
            return sf_data['frame']
        elif hasattr(sf_data, 'df'):
            return sf_data
        elif isinstance(sf_data, dict):
            if 'frame' in sf_data:
                return sf_data['frame']
            else:
                return sf_data
        else:
            return sf_data
    
    def list_subframes(self):
        """
        List all available subframes.
        
        Returns:
            list: Names of loaded subframes
        
        Example:
            print(f"Available: {adf.list_subframes()}")
        """
        if not hasattr(self, '_subframes') or not hasattr(self._subframes, 'subframes'):
            return []
        return list(self._subframes.subframes.keys())
    
    def auto_alias_subframe(self, subframe_name, validate=False, reset_before=False,
                            overwrite=False, verbose=True):
        """
        Explicitly create aliases for all columns in a subframe.
        
        Creates aliases: column_name -> subframe_name.column_name
        Tracks created aliases in self._auto_aliases
        
        Args:
            subframe_name: Name of subframe
            validate: If True, validate against materialized columns (slow!)
            reset_before: If True, remove old auto-aliases for this subframe first
            overwrite: If True, overwrite existing aliases/columns. If False, skip conflicts.
            verbose: If True, print summary of created/skipped aliases
        
        Returns:
            dict: {'created': [...], 'skipped_column': [...], 'skipped_alias': [...], 
                   'validated': [...], 'expressions': {...}}
        
        Example:
            # First time
            result = adf.auto_alias_subframe('DITS0FitSide', validate=True)
            
            # Regenerate later
            result = adf.auto_alias_subframe('DITS0FitSide', reset_before=True)
            
            # Check what was skipped
            print(f"Skipped {len(result['skipped_column'])} existing columns")
        """
        import warnings
        import numpy as np
        
        # Initialize _auto_aliases if not present (backward compatibility)
        if not hasattr(self, '_auto_aliases'):
            self._auto_aliases = {}
        
        # Get subframe
        try:
            sf = self.subframe(subframe_name)
        except KeyError as e:
            raise KeyError(f"Cannot auto-alias: {e}")
        
        # Get subframe DataFrame
        if hasattr(sf, 'df'):
            sf_df = sf.df
        else:
            sf_df = sf
        
        # Get index columns
        if hasattr(self._subframes, 'subframes') and subframe_name in self._subframes.subframes:
            sf_entry = self._subframes.subframes[subframe_name]
            if isinstance(sf_entry, dict) and 'index' in sf_entry:
                index_cols = sf_entry['index']
            else:
                index_cols = self.index_columns.get(subframe_name, [])
        else:
            index_cols = self.index_columns.get(subframe_name, [])
        
        # Ensure index_cols is a list
        if isinstance(index_cols, str):
            index_cols = [index_cols]
        
        # Reset: remove existing auto-aliases for this subframe
        if reset_before:
            old_aliases = [k for k, v in self._auto_aliases.items() 
                          if v == subframe_name]
            if old_aliases:
                if verbose:
                    print(f"  Removing {len(old_aliases)} existing auto-aliases for '{subframe_name}'")
                self.remove_aliases(old_aliases, strict=False)
        
        # Create aliases - track what we create and skip
        aliases_created = {}
        skipped_column = []  # Skipped because column exists in main DataFrame
        skipped_alias = []   # Skipped because alias already exists
        validated = []
        
        for col in sf_df.columns:
            # Skip index columns
            if col in index_cols:
                continue
            
            alias_expr = f"{subframe_name}.{col}"
            
            # BUG FIX: Skip if column already exists in main DataFrame
            # This prevents self-referential cycles like: dEdxTPC -> T.dEdxTPC -> dEdxTPC
            if col in self.df.columns:
                skipped_column.append(col)
                
                if validate:
                    # Validate that existing column matches subframe lookup
                    if verbose:
                        print(f"    Validating '{col}'...", end=' ')
                    temp_name = f'_temp_validate_{col}'
                    self.add_alias(temp_name, alias_expr)
                    self.materialize_alias(temp_name, warn_missing_keys=False)
                    
                    materialized = self.df[col].values
                    alias_result = self.df[temp_name].values
                    
                    match = np.allclose(materialized, alias_result, equal_nan=True, rtol=1e-6)
                    
                    if match:
                        if verbose:
                            print("✓")
                        validated.append(col)
                    else:
                        if verbose:
                            print("✗ MISMATCH")
                        warnings.warn(
                            f"Existing column '{col}' does NOT match subframe lookup "
                            f"'{alias_expr}'. Data may be inconsistent!"
                        )
                    
                    self.df.drop(columns=[temp_name], inplace=True)
                    if temp_name in self._schema.get('columns', {}):
                        del self._schema['columns'][temp_name]
                
                if not overwrite:
                    continue  # Skip - don't create alias for existing column
            
            # Skip if alias already exists (unless overwrite=True)
            if col in self.aliases:
                if not overwrite:
                    skipped_alias.append(col)
                    continue
            
            # Add alias and track it
            self.add_alias(col, alias_expr)
            self._auto_aliases[col] = subframe_name  # Track as auto-created
            aliases_created[col] = alias_expr
        
        # Report
        if verbose:
            print(f"\n  ✓ Created {len(aliases_created)} auto-aliases for '{subframe_name}'")
            if skipped_column:
                preview = skipped_column[:5]
                more = f" ... (+{len(skipped_column)-5} more)" if len(skipped_column) > 5 else ""
                print(f"    ⚠️  Skipped {len(skipped_column)} (column exists in DataFrame): {preview}{more}")
            if skipped_alias:
                preview = skipped_alias[:5]
                more = f" ... (+{len(skipped_alias)-5} more)" if len(skipped_alias) > 5 else ""
                print(f"    Skipped {len(skipped_alias)} (alias already defined): {preview}{more}")
            if validated:
                print(f"    ✓ Validated {len(validated)} existing columns match subframe")
        
        return {
            'created': list(aliases_created.keys()),
            'skipped_column': skipped_column,
            'skipped_alias': skipped_alias,
            'validated': validated,
            'expressions': aliases_created
        }
    
    def auto_alias_all_subframes(self, validate=False, reset_before=False, 
                                  overwrite=False, verbose=True):
        """
        Explicitly create aliases for all loaded subframes.
        
        WARNING: If multiple subframes have same column name, last one wins.
        For production, prefer auto_alias_subframe() for specific subframes.
        
        Args:
            validate: If True, validate aliases (slow!)
            reset_before: If True, remove old auto-aliases first
            overwrite: If True, overwrite existing aliases/columns
            verbose: If True, print summary
        
        Returns:
            dict: {subframe_name: result_dict} where result_dict contains
                  'created', 'skipped_column', 'skipped_alias', etc.
        
        Example:
            all_results = adf.auto_alias_all_subframes(validate=False)
            for sf_name, result in all_results.items():
                print(f"{sf_name}: {len(result['created'])} created, "
                      f"{len(result['skipped_column'])} skipped")
        """
        all_created = {}
        
        subframes = self.list_subframes()
        if verbose:
            print(f"\nAuto-aliasing {len(subframes)} subframes...")
        
        for sf_name in subframes:
            if verbose:
                print(f"\nSubframe '{sf_name}':")
            result = self.auto_alias_subframe(
                sf_name, 
                validate=validate, 
                reset_before=reset_before,
                overwrite=overwrite,
                verbose=verbose
            )
            all_created[sf_name] = result
        
        return all_created
    
    def get_auto_alias_candidates(self):
        """
        Get materialized columns that could be replaced with auto-aliases.
        
        Returns:
            dict: {subframe_name: [columns]}
        
        Example:
            candidates = adf.get_auto_alias_candidates()
            print(f"Can remove {sum(len(v) for v in candidates.values())} columns")
            
            for cols in candidates.values():
                adf.df.drop(columns=cols, inplace=True)
        """
        candidates = {}
        
        for sf_name in self.list_subframes():
            sf = self.subframe(sf_name)
            sf_df = sf.df if hasattr(sf, 'df') else sf
            
            # Get index columns
            if hasattr(self._subframes, 'subframes') and sf_name in self._subframes.subframes:
                sf_entry = self._subframes.subframes[sf_name]
                if isinstance(sf_entry, dict) and 'index' in sf_entry:
                    index_cols = sf_entry['index']
                else:
                    index_cols = self.index_columns.get(sf_name, [])
            else:
                index_cols = self.index_columns.get(sf_name, [])
            
            materialized = [
                col for col in sf_df.columns
                if col not in index_cols and col in self.df.columns
            ]
            
            if materialized:
                candidates[sf_name] = materialized
        
        return candidates
    
    def remove_alias(self, name, *, remove_from_schema=True, strict=True):
        """
        Remove a single alias safely.
        
        Removes from self.aliases, self._auto_aliases, and optionally schema.
        
        Args:
            name: Alias name to remove
            remove_from_schema: If True, also remove from schema
            strict: If True, raise KeyError if alias doesn't exist
        
        Raises:
            KeyError: If alias not found and strict=True
        
        Example:
            adf.remove_alias('my_alias')
            adf.remove_alias('maybe_alias', strict=False)
        """
        # Initialize _auto_aliases if not present (backward compatibility)
        if not hasattr(self, '_auto_aliases'):
            self._auto_aliases = {}
        
        # Check if alias exists
        if name not in self.aliases:
            if strict:
                raise KeyError(f"Alias '{name}' not found in aliases")
            else:
                return
        
        # Remove from schema (which automatically removes from aliases property)
        if name in self._schema["columns"]:
            col_info = self._schema["columns"][name]
            if "expr" in col_info:
                # If only has expr/dtype/auto_alias, remove entire entry
                if set(col_info.keys()) <= {'expr', 'dtype', 'compressed_dtype', 'auto_alias', 'auto_subframe'}:
                    del self._schema["columns"][name]
                else:
                    # More complex entry, just remove expr
                    del col_info["expr"]
                    col_info.pop('auto_alias', None)
                    col_info.pop('auto_subframe', None)
        
        # Remove from auto-aliases tracking
        self._auto_aliases.pop(name, None)
    
    def remove_aliases(self, names, *, remove_from_schema=True, strict=True):
        """
        Remove multiple aliases safely.
        
        Args:
            names: Iterable of alias names
            remove_from_schema: If True, also remove from schema
            strict: If True, raise KeyError on first missing alias
        
        Example:
            adf.remove_aliases(['alias1', 'alias2', 'alias3'])
            adf.remove_aliases(candidate_list, strict=False)
        """
        for name in names:
            self.remove_alias(name, remove_from_schema=remove_from_schema, strict=strict)
    
    def is_auto_alias(self, name):
        """
        Check if an alias was auto-created from a subframe.
        
        Args:
            name: Alias name to check
        
        Returns:
            bool: True if auto-created, False otherwise
        
        Example:
            if adf.is_auto_alias('dyC1_intercept'):
                print("This is an auto-alias")
        """
        if not hasattr(self, '_auto_aliases'):
            return False
        return name in self._auto_aliases
    
    def get_auto_aliases(self, subframe_name=None):
        """
        Get all auto-created aliases (or for specific subframe).
        
        Args:
            subframe_name: If provided, return only aliases from this subframe
        
        Returns:
            dict: {alias_name: subframe_name}
        
        Example:
            all_auto = adf.get_auto_aliases()
            dits_auto = adf.get_auto_aliases('DITS0FitSide')
        """
        if not hasattr(self, '_auto_aliases'):
            return {}
        
        if subframe_name is None:
            return dict(self._auto_aliases)
        return {k: v for k, v in self._auto_aliases.items() if v == subframe_name}
    
    def remove_auto_aliases(self, subframe_name=None):
        """
        Remove auto-created aliases (all or for specific subframe).
        
        Args:
            subframe_name: If provided, remove only aliases from this subframe
        
        Example:
            adf.remove_auto_aliases()  # Remove all auto-aliases
            adf.remove_auto_aliases('DITS0FitSide')  # Remove only DITS0FitSide aliases
        """
        if not hasattr(self, '_auto_aliases'):
            return
        
        if subframe_name is None:
            to_remove = list(self._auto_aliases.keys())
        else:
            to_remove = [k for k, v in self._auto_aliases.items() if v == subframe_name]
        
        if to_remove:
            print(f"  Removing {len(to_remove)} auto-aliases" + 
                  (f" for '{subframe_name}'" if subframe_name else ""))
            self.remove_aliases(to_remove, strict=False)
    
    def list_auto_aliases(self, subframe_name=None):
        """
        List auto-created alias names (all or for specific subframe).
        
        Args:
            subframe_name: If provided, list only aliases from this subframe
        
        Returns:
            list: Alias names
        
        Example:
            print(adf.list_auto_aliases())
            print(adf.list_auto_aliases('DITS0FitSide'))
        """
        if not hasattr(self, '_auto_aliases'):
            return []
        
        if subframe_name is None:
            return list(self._auto_aliases.keys())
        return [k for k, v in self._auto_aliases.items() if v == subframe_name]

    # =========================================================================
    # PHASE B: SCHEMA EMBEDDING IN FILES
    # =========================================================================
    
    def save_schema_to_root(self, root_file, tree_name='tree'):
        """
        Embed schema in ROOT file as TNamed object.
        
        Args:
            root_file: Path to ROOT file or open ROOT.TFile
            tree_name: Name of tree to attach schema to
        
        Example:
            adf.save_schema_to_root('output.root', 'tree')
        """
        import json
        
        if ROOT is None:
            raise ImportError("ROOT is required for save_schema_to_root()")
        
        # Export schema to JSON
        schema_json = json.dumps(self.export_schema(), indent=2)
        
        # Open file
        if isinstance(root_file, str):
            f = ROOT.TFile.Open(root_file, "UPDATE")
            should_close = True
        else:
            f = root_file
            should_close = False
        
        try:
            # Create TNamed with schema
            schema_obj = ROOT.TObjString(schema_json)
            schema_obj.Write("ADF_SCHEMA")
            
            print(f"  ✓ Embedded schema in ROOT file: {f.GetName()}")
            
        finally:
            if should_close:
                f.Close()
    
    def load_schema_from_root(self, root_file):
        """
        Load embedded schema from ROOT file.
        
        Args:
            root_file: Path to ROOT file or open ROOT.TFile
        
        Returns:
            bool: True if schema was found and loaded, False otherwise
        
        Example:
            if adf.load_schema_from_root('input.root'):
                print("Schema loaded from file")
        """
        import json
        
        if ROOT is None:
            raise ImportError("ROOT is required for load_schema_from_root()")
        
        # Open file
        if isinstance(root_file, str):
            f = ROOT.TFile.Open(root_file, "READ")
            should_close = True
        else:
            f = root_file
            should_close = False
        
        try:
            # Try to get schema object
            schema_obj = f.Get("ADF_SCHEMA")
            
            if schema_obj:
                schema_json = schema_obj.GetString().Data()
                schema = json.loads(schema_json)
                self.update_schema(schema)
                
                print(f"  ✓ Loaded embedded schema from ROOT file")
                return True
            else:
                return False
                
        finally:
            if should_close:
                f.Close()
    
    def save_schema_to_parquet_metadata(self, parquet_file):
        """
        Save schema as metadata alongside Parquet file.
        
        Creates: filename.parquet + filename_schema.json
        
        Args:
            parquet_file: Path to Parquet file
        
        Example:
            adf.df.to_parquet('output.parquet')
            adf.save_schema_to_parquet_metadata('output.parquet')
        """
        import json
        from pathlib import Path
        
        # Generate schema JSON filename
        parquet_path = Path(parquet_file)
        schema_path = parquet_path.with_suffix('.schema.json')
        
        # Save schema
        schema = self.export_schema()
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        print(f"  ✓ Saved schema metadata: {schema_path}")
    
    def load_schema_from_parquet_metadata(self, parquet_file):
        """
        Load schema from Parquet metadata file.
        
        Looks for: filename_schema.json
        
        Args:
            parquet_file: Path to Parquet file
        
        Returns:
            bool: True if schema was found and loaded, False otherwise
        
        Example:
            adf = AliasDataFrame(pd.read_parquet('input.parquet'))
            if adf.load_schema_from_parquet_metadata('input.parquet'):
                print("Schema loaded")
        """
        import json
        from pathlib import Path
        
        # Look for schema JSON
        parquet_path = Path(parquet_file)
        schema_path = parquet_path.with_suffix('.schema.json')
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            self.update_schema(schema)
            
            print(f"  ✓ Loaded schema from metadata: {schema_path}")
            return True
        else:
            return False

    # =========================================================================
    # PHASE 6.8: DFDRAW INTEGRATION
    # =========================================================================
    #
    # Seamless plotting with lazy evaluation, axis metadata, and entry selection.
    # dfdraw remains stateless; AliasDataFrame owns all state.
    #
    # Features:
    # - Axis titles stored in schema
    # - Lazy materialization on draw
    # - Entry selection (range + mask)
    # - Memory management (track what we add)
    # - Instance-level defaults with 3-level precedence
    #
    # =========================================================================

    def set_axis_title(self, column: str, title: str) -> None:
        """
        Set display title for a column/alias.
        
        Titles are stored in schema and used by dfdraw for automatic axis labels.
        
        Args:
            column: Column or alias name
            title: Display title (e.g., 'x [cm]', 'dE/dx [MeV/cm]')
        
        Example:
            adf.set_axis_title('x', 'x [cm]')
            adf.set_axis_title('exb_dy', 'ExB Δy [cm]')
        """
        if column not in self._schema.get('columns', {}):
            self._schema.setdefault('columns', {})[column] = {}
        self._schema['columns'][column]['title'] = title

    def get_axis_title(self, column: str):
        """
        Get display title for a column/alias.
        
        Args:
            column: Column or alias name
            
        Returns:
            Title string if set, None otherwise.
        
        Note:
            Used by dfdraw via duck typing for automatic axis labels.
        """
        return self._schema.get('columns', {}).get(column, {}).get('title')

    def _get_materialized_aliases(self):
        """
        Return set of currently materialized alias names.
        
        Returns:
            Set of alias names that exist as columns in the DataFrame.
        
        Note:
            Used internally to track what draw methods add vs. pre-existing.
        """
        aliases = self.aliases  # {name: expr}
        return {name for name in aliases if name in self.df.columns}

    def drop_materialized(self, aliases):
        """
        Drop materialized alias columns from DataFrame.
        
        Args:
            aliases: Alias names to drop (silently ignores non-existent)
        
        Note:
            Only drops columns that are aliases, never physical columns.
        """
        alias_names = set(self.aliases.keys())
        to_drop = [a for a in aliases if a in alias_names and a in self.df.columns]
        if to_drop:
            self.df = self.df.drop(columns=to_drop)

    def _resolve_draw_param(self, param_value, param_name: str):
        """
        Resolve draw parameter with 3-level precedence.
        
        Precedence (highest to lowest):
            1. Per-call parameter (if not None)
            2. Instance property (self.draw_*)
            3. Hard-coded default
        
        Args:
            param_value: Value passed to draw method (or None)
            param_name: One of 'lazy', 'keep_materialized', 'clear_after'
        
        Returns:
            Resolved parameter value
        """
        defaults = {
            'lazy': False,
            'keep_materialized': True,
            'clear_after': True,
        }
        
        if param_value is not None:
            return param_value
        
        instance_attr = f'draw_{param_name}'
        if hasattr(self, instance_attr):
            return getattr(self, instance_attr)
        
        return defaults[param_name]

    def _apply_entry_selection(self, 
                               entry_begin=None,
                               entry_end=None,
                               entry_mask=None):
        """
        Apply entry selection to get a subset of the DataFrame.
        
        Args:
            entry_begin: Start index (positional, inclusive)
            entry_end: End index (positional, exclusive)
            entry_mask: Either boolean array (len=len(df)) or integer indices
        
        Returns:
            Sliced DataFrame
        
        Raises:
            ValueError: If both range (begin/end) AND mask are provided
        
        Semantics:
            - Boolean mask: Uses loc-style selection (must match DataFrame length)
            - Integer array: Uses iloc-style positional selection
            - Range: Uses iloc[begin:end]
        """
        # Disallow mixing
        has_range = entry_begin is not None or entry_end is not None
        has_mask = entry_mask is not None
        
        if has_range and has_mask:
            raise ValueError(
                "Cannot specify both entry_begin/entry_end and entry_mask. "
                "Use one or the other."
            )
        
        if entry_mask is not None:
            # Detect boolean vs integer mask
            mask_array = np.asarray(entry_mask)
            if pd.api.types.is_bool_dtype(mask_array):
                # Boolean mask - use loc-style
                if len(mask_array) != len(self.df):
                    raise ValueError(
                        f"Boolean mask length ({len(mask_array)}) must match "
                        f"DataFrame length ({len(self.df)})"
                    )
                return self.df.loc[mask_array]
            else:
                # Integer indices - use iloc-style
                return self.df.iloc[mask_array]
        
        elif has_range:
            # Range selection
            start = entry_begin if entry_begin is not None else 0
            stop = entry_end  # None means to end
            return self.df.iloc[start:stop]
        
        else:
            # No selection - return full DataFrame
            return self.df

    def _parse_expr_aliases(self, expr: str, group_by=None, color=None):
        """
        Extract alias names from expression and optional parameters.
        
        Args:
            expr: Plot expression like 'y:x' or 'x'
            group_by: Optional group_by column
            color: Optional color column
        
        Returns:
            Set of alias names (not physical columns) needed
        """
        columns_needed = set()
        
        # Parse main expression
        parts = expr.replace(' ', '').split(':')
        columns_needed.update(parts)
        
        # Add group_by and color if present
        if group_by:
            columns_needed.add(group_by)
        if color and isinstance(color, str):
            columns_needed.add(color)
        
        # Filter to only aliases (not physical columns)
        alias_names = set(self.aliases.keys())
        return {c for c in columns_needed if c in alias_names}

    def _eval_alias_on_df(self, alias_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate an alias expression on an arbitrary DataFrame.
        
        Used for slice-first lazy evaluation where we need to compute
        aliases on a subset without modifying self.df.
        
        Args:
            alias_name: Name of the alias to evaluate
            df: DataFrame to evaluate on (may be a subset of self.df)
        
        Returns:
            DataFrame with the alias column added
        
        Note:
            This handles dependency resolution for the target alias.
        """
        if alias_name not in self.aliases:
            raise ValueError(f"'{alias_name}' is not a defined alias")
        
        # Get the expression
        expr = self.aliases[alias_name]
        
        # Build evaluation namespace from the subset df
        namespace = {col: df[col].values for col in df.columns}
        namespace['np'] = np
        namespace['pd'] = pd
        
        # Add numpy functions
        namespace.update(NumpyRootMapper.get_numpy_functions_for_eval())
        
        # Check for alias dependencies and evaluate them first
        alias_deps = self._get_alias_dependencies(alias_name, expr)
        for dep_type, dep_name in alias_deps:
            if dep_type == 'alias' and dep_name not in df.columns:
                df = self._eval_alias_on_df(dep_name, df)
                namespace[dep_name] = df[dep_name].values
        
        # Evaluate the expression
        try:
            result = eval(expr, {"__builtins__": {}}, namespace)
            df = df.copy()
            df[alias_name] = result
            return df
        except Exception as e:
            raise ValueError(f"Failed to evaluate alias '{alias_name}': {e}")

    def _resolve_plot_type(self, expr: str, type_hint: str) -> str:
        """
        Resolve plot method name from expression and type hint.
        
        Args:
            expr: Plot expression
            type_hint: 'auto', 'hist', 'scatter', 'profile', 'hist2d', 'hexbin'
        
        Returns:
            Method name string
        """
        if type_hint != 'auto':
            return type_hint
        
        # Auto-detect from expression
        parts = expr.replace(' ', '').split(':')
        if len(parts) == 1:
            return 'hist'
        else:
            return 'scatter'

    def draw(self,
             expr: str,
             type: str = 'auto',
             *,
             lazy=None,
             keep_materialized=None,
             entry_begin=None,
             entry_end=None,
             entry_mask=None,
             **kwargs):
        """
        Draw a plot with automatic materialization and axis labels.

        All plotting parameters are forwarded to dfdraw. For full docs::

            adf.draw_help()              # list all plot types
            adf.draw_help('profile')     # profile-specific options
            adf.draw_help('hist')        # histogram options

        Parameters
        ----------
        expr : str
            Plot expression (e.g., 'y:x', 'x', 'dEdx:p')
        type : str, default 'auto'
            Plot type: 'auto', 'hist', 'scatter', 'profile', 'hist2d', 'hexbin'.
            'auto' infers from expression (single var → hist, two vars → scatter).
        lazy : bool, optional
            If True, auto-materialize needed aliases. Default from self.draw_lazy.
        keep_materialized : bool, optional
            If False, drop aliases we materialized after draw.
        entry_begin, entry_end : int, optional
            Row range selection.
        entry_mask : array, optional
            Boolean or integer mask for entry selection.
        **kwargs
            Forwarded to dfdraw. Common: selection, group_by, bins, range,
            title, xlabel, ylabel. Profile: return_data, min_entries,
            group_by_bins, group_by_quantiles, sort_groups.

        Returns
        -------
        tuple
            (fig, ax, stats_dict)

        Examples
        --------
        >>> adf.draw('dEdx:p', type='profile', bins=100, group_by='charge')
        >>> adf.draw('dy:row', type='profile', selection='abs(dy)<3',
        ...          group_by='mP3', group_by_bins=5, min_entries=10,
        ...          return_data=True)
        >>> stats = adf.draw('dy:row', type='profile', return_data=True)[2]
        >>> profile_df = stats['profile_data']  # DataFrame for fitting
        """
        # Import dfdraw
        try:
            from dfextensions.dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw package not found. Install it or ensure it's in your path."
            )
        
        # Resolve parameters with 3-level precedence
        effective_lazy = self._resolve_draw_param(lazy, 'lazy')
        effective_keep = self._resolve_draw_param(keep_materialized, 'keep_materialized')
        
        # =================================================================
        # Phase 7.3: Auto-load branches in lazy mode
        # =================================================================
        if self._lazy_reader is not None:
            # Detect required branches from expression and parameters
            required_branches = self.get_required_branches(
                expr=expr,
                selection=kwargs.get('selection'),
                group_by=kwargs.get('group_by'),
                color=kwargs.get('color')
            )
            # Load any branches not already loaded
            branches_to_load = required_branches - self._lazy_reader.loaded_branches
            
            # Phase 6.8a fix: Filter out subframe names (they are not TTree branches)
            all_subframes = set(self._subframes.subframes.keys()) | set(getattr(self, '_subframe_readers', {}).keys())
            branches_to_load = branches_to_load - all_subframes
            
            if branches_to_load:
                self.ensure_branches(list(branches_to_load))
        # =================================================================
        
        # Track what's already materialized
        already_materialized = self._get_materialized_aliases()
        
        # Parse expression to find needed aliases
        needed_aliases = self._parse_expr_aliases(expr, kwargs.get('group_by'), kwargs.get('color'))
        
        # Check if entry selection is requested
        has_entry_selection = (entry_begin is not None or 
                               entry_end is not None or 
                               entry_mask is not None)
        
        if has_entry_selection:
            # SLICE-FIRST: Apply entry selection, then evaluate aliases on subset
            df_subset = self._apply_entry_selection(entry_begin, entry_end, entry_mask)
            
            # Evaluate needed aliases directly on the subset (not on full df)
            if effective_lazy:
                to_evaluate = needed_aliases - already_materialized
                for alias_name in to_evaluate:
                    if alias_name not in df_subset.columns:
                        df_subset = self._eval_alias_on_df(alias_name, df_subset)
            
            # No cleanup needed - we didn't modify self.df
            cleanup_needed = False
        else:
            # NO SELECTION: Use standard lazy materialization on full df
            if effective_lazy:
                to_materialize = needed_aliases - already_materialized
                if to_materialize:
                    self.materialize_aliases(names=list(to_materialize))
            
            df_subset = self.df
            cleanup_needed = not effective_keep
        
        # =================================================================
        # Subframe column resolution for draw
        # Detect 'Subframe.column' patterns in expression and selection,
        # materialize them as temporary columns so dfdraw can access them.
        # Column names use underscores (Sub_col) to avoid pandas eval issues.
        # =================================================================
        subframe_replacements = {}
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            sf_names = set(self._subframes.subframes.keys())
            all_text = expr
            if kwargs.get('selection'):
                all_text += ' ' + kwargs['selection']
            if kwargs.get('group_by'):
                all_text += ' ' + str(kwargs['group_by'])
            
            import re as _re
            for match in _re.finditer(r'\b(\w+)\.(\w+)\b', all_text):
                sf_name, col_name = match.group(1), match.group(2)
                if sf_name in sf_names:
                    dot_ref = f"{sf_name}.{col_name}"
                    flat_ref = f"{sf_name}_{col_name}"
                    if flat_ref not in df_subset.columns and dot_ref not in subframe_replacements:
                        try:
                            sf = self.get_subframe(sf_name)
                            index_cols = self._subframes.get_entry(sf_name)['index']
                            if isinstance(index_cols, str):
                                index_cols = [index_cols]
                            join_idx, missing = self._compute_join_indices(sf_name, index_cols)
                            if col_name in sf.df.columns:
                                if df_subset is self.df:
                                    df_subset = df_subset.copy()
                                df_subset[flat_ref] = sf.df[col_name].values[join_idx]
                                subframe_replacements[dot_ref] = flat_ref
                        except Exception:
                            pass
            
            if subframe_replacements:
                for dot_ref, flat_ref in subframe_replacements.items():
                    expr = expr.replace(dot_ref, flat_ref)
                    if 'selection' in kwargs and kwargs['selection']:
                        kwargs['selection'] = kwargs['selection'].replace(dot_ref, flat_ref)
                    if 'group_by' in kwargs and isinstance(kwargs.get('group_by'), str):
                        kwargs['group_by'] = kwargs['group_by'].replace(dot_ref, flat_ref)
        
        # Create plotter and delegate
        plotter = DFDraw(df_subset)
        
        # Attach self for duck-typed axis title lookup
        plotter._data_source = self
        
        # Determine plot method
        method_name = self._resolve_plot_type(expr, type)
        plot_func = getattr(plotter, method_name)
        
        # Call plot
        result = plot_func(expr, **kwargs)
        
        # Cleanup if requested (only when no entry selection)
        if cleanup_needed:
            we_added = self._get_materialized_aliases() - already_materialized
            if we_added:
                self.drop_materialized(we_added)
        
        return result

    def hist(self, expr: str, **kwargs):
        """Histogram. See ``adf.draw_help('hist')`` for all options."""
        return self.draw(expr, type='hist', **kwargs)

    def scatter(self, expr: str, **kwargs):
        """Scatter plot. See ``adf.draw_help('scatter')`` for all options."""
        return self.draw(expr, type='scatter', **kwargs)

    def profile(self, expr: str, **kwargs):
        """Profile plot (mean of y in bins of x). See ``adf.draw_help('profile')`` for all options."""
        return self.draw(expr, type='profile', **kwargs)

    def hist2d(self, expr: str, **kwargs):
        """2D histogram. See ``adf.draw_help('hist2d')`` for all options."""
        return self.draw(expr, type='hist2d', **kwargs)

    def hexbin(self, expr: str, **kwargs):
        """Hexbin plot. See ``adf.draw_help('hexbin')`` for all options."""
        return self.draw(expr, type='hexbin', **kwargs)

    # =========================================================================
    # Phase 13.9: Registered Functions & Polynomial Support
    # =========================================================================

    def register_function(self, name, func, overwrite=False):
        """
        Register a custom callable for use in alias expressions.

        The function becomes available in the eval namespace,
        callable from alias expressions. Registered functions take
        precedence over column names in alias evaluation.

        Parameters
        ----------
        name : str
            Function name (used in expressions)
        func : callable
            Function that takes numpy arrays, returns numpy array
        overwrite : bool, default False
            If True, allow replacing an existing registered function.

        Raises
        ------
        ValueError
            If name already registered and overwrite=False.

        Example
        -------
        >>> adf.register_function('myFunc', some_numba_function)
        >>> adf.add_alias('result', 'myFunc(col1, col2)')
        """
        if not hasattr(self, '_registered_functions'):
            self._registered_functions = {}

        if not overwrite and name in self._registered_functions:
            raise ValueError(
                f"Function '{name}' already registered. Use overwrite=True to replace."
            )

        self._registered_functions[name] = func

    def register_polynomial_from_subframe(self, func_name, poly_spec,
                                           coefficients_subframe, coeff_select):
        """
        Register a polynomial function that reads coefficients from a subframe.
        Coefficients are accessed via join indices — no column materialization.

        Parameters
        ----------
        func_name : str
            Function name for use in alias expressions (e.g., 'polFit')
        poly_spec : PolynomialSpec
            Polynomial specification (from AliasDataFrame.PolynomialSpec)
        coefficients_subframe : str
            Name of registered subframe containing coefficient columns
        coeff_select : list of str or str
            Coefficient columns, ordered to match poly_spec terms.
            - list: explicit column names (safest)
            - str: regexp pattern to match against subframe columns

        Example
        -------
        >>> from AliasDataFrame.PolynomialSpec import PolynomialSpec
        >>> spec = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1))
        >>> keys = [t[0] for t in spec.basis_expressions()]
        >>> coeff_cols = [f'dyC3_slope_{k}_poly' for k in keys]
        >>> adf.register_polynomial_from_subframe('polFit', spec, 'PolyFit', coeff_cols)
        >>> adf.add_alias('dy_corr', 'polFit(xM, driftM, dsecM, tgSlp)')
        """
        import re

        sf = self.get_subframe(coefficients_subframe)

        # Resolve coeff_select
        if isinstance(coeff_select, str):
            pattern = re.compile(coeff_select)
            coeff_cols = [c for c in sf.df.columns if pattern.match(c)]
            if len(coeff_cols) != poly_spec.n_terms:
                raise ValueError(
                    f"Regexp '{coeff_select}' matched {len(coeff_cols)} columns, "
                    f"but poly_spec has {poly_spec.n_terms} terms"
                )
        else:
            coeff_cols = list(coeff_select)

        # Generate Numba evaluator
        evaluator = poly_spec.numba_evaluator(self, coefficients_subframe, coeff_cols)
        self.register_function(func_name, evaluator)

        # Store in schema for reconstruction
        if not self._schema.get('registered_functions'):
            self._schema['registered_functions'] = {}
        self._schema['registered_functions'][func_name] = {
            **poly_spec.to_schema(),
            'coefficients_subframe': coefficients_subframe,
            'coeff_select': coeff_cols,
        }

    def draw_help(self, plot_type=None):
        """
        Print dfdraw parameter documentation.

        Parameters
        ----------
        plot_type : str, optional
            Specific plot type: 'profile', 'hist', 'scatter', 'hist2d', 'hexbin'.
            If None, list available types.

        Examples
        --------
        >>> adf.draw_help()              # list all plot types
        >>> adf.draw_help('profile')     # profile-specific options
        >>> adf.draw_help('hist')        # histogram options
        """
        try:
            from dfextensions.dfdraw import DFDraw
        except ImportError:
            print("dfdraw package not found. Install it or ensure it's in your path.")
            return

        if plot_type is None:
            print("Available plot types: profile, hist, scatter, hist2d, hexbin")
            print("Usage: adf.draw_help('profile')  # show options for profile plots")
            print("\nDFDraw methods:")
            for method in ['profile', 'hist', 'scatter', 'hist2d', 'hexbin']:
                doc = getattr(DFDraw, method, None)
                if doc and doc.__doc__:
                    first_line = doc.__doc__.strip().split('\n')[0]
                    print(f"  {method:10s} — {first_line}")
        else:
            func = getattr(DFDraw, plot_type, None)
            if func is None:
                print(f"Unknown plot type '{plot_type}'. Available: profile, hist, scatter, hist2d, hexbin")
            else:
                help(func)

    def draw_batch(self,
                   specs,
                   save_dir=None,
                   defaults=None,
                   *,
                   clear_after=None,
                   lazy=None,
                   on_error: str = 'skip',
                   verbose: bool = True,
                   **kwargs):
        """
        Generate multiple plots with optimized materialization.
        
        Optimization: Pre-scans all specs to collect needed aliases,
        materializes ALL at once, then generates plots.
        
        Args:
            specs: Dict of {name: spec} or path to JSON/YAML file
            save_dir: Directory to save plots
            defaults: Default parameters applied to all plots
            clear_after: If True, drop aliases we materialized after batch.
                        Default from self.draw_clear_after
            lazy: If True, auto-materialize. Default from self.draw_lazy
            on_error: 'skip' or 'raise'
            verbose: Print progress
            **kwargs: Additional defaults
        
        Returns:
            Dict with results, _errors, _summary (see dfdraw.draw_batch)
        
        Example:
            specs = {
                'hist_x': {'expr': 'x'},
                'profile_L1': {'expr': 'L1:x', 'type': 'profile'},
                'scatter_L2': {'expr': 'L2:x', 'sample': 10000},
            }
            adf.draw_batch(specs, save_dir='qa/', defaults={'stats': True})
        """
        # Import dfdraw
        try:
            from dfextensions.dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw package not found. Install it or ensure it's in your path."
            )
        
        # Resolve parameters
        effective_lazy = self._resolve_draw_param(lazy, 'lazy')
        effective_clear = self._resolve_draw_param(clear_after, 'clear_after')
        
        # Load specs if path
        if isinstance(specs, str):
            specs = self._load_specs_file_for_draw(specs)
        
        # =================================================================
        # Phase 7.3: Pre-scan and batch-load branches in lazy mode
        # =================================================================
        if self._lazy_reader is not None:
            all_required = set()
            merged_defaults = {**(defaults or {}), **kwargs}
            
            for name, spec in specs.items():
                merged_spec = {**merged_defaults, **spec}
                required = self.get_required_branches(
                    expr=merged_spec.get('expr', name),
                    selection=merged_spec.get('selection'),
                    group_by=merged_spec.get('group_by'),
                    color=merged_spec.get('color')
                )
                all_required.update(required)
            
            # Load all required branches at once
            branches_to_load = all_required - self._lazy_reader.loaded_branches
            
            # Phase 6.8a fix: Filter out subframe names (they are not TTree branches)
            all_subframes = set(self._subframes.subframes.keys()) | set(getattr(self, '_subframe_readers', {}).keys())
            branches_to_load = branches_to_load - all_subframes
            
            if branches_to_load:
                if verbose:
                    print(f"Loading {len(branches_to_load)} branches: {sorted(branches_to_load)}")
                self.ensure_branches(list(branches_to_load))
        # =================================================================
        
        # Track pre-existing materialized aliases
        already_materialized = self._get_materialized_aliases()
        
        # PRE-SCAN: Collect all needed aliases across all specs
        if effective_lazy:
            all_needed = set()
            merged_defaults = {**(defaults or {}), **kwargs}
            
            for name, spec in specs.items():
                merged_spec = {**merged_defaults, **spec}
                expr = merged_spec.get('expr', name)
                group_by = merged_spec.get('group_by')
                color = merged_spec.get('color')
                all_needed.update(self._parse_expr_aliases(expr, group_by, color))
            
            # Materialize ALL at once
            to_materialize = all_needed - already_materialized
            if to_materialize:
                if verbose:
                    print(f"Materializing {len(to_materialize)} aliases: {sorted(to_materialize)}")
                self.materialize_aliases(names=list(to_materialize))
        
        # =================================================================
        # Subframe column resolution for draw_batch
        # Same logic as draw() — detect Subframe.column patterns across
        # all specs, materialize as temporary columns, rewrite expressions.
        # =================================================================
        subframe_replacements = {}
        df_for_plot = self.df
        if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
            sf_names = set(self._subframes.subframes.keys())
            merged_defaults = {**(defaults or {}), **kwargs}
            
            # Collect all text across all specs
            all_text_parts = []
            for name, spec in specs.items():
                merged_spec = {**merged_defaults, **spec}
                all_text_parts.append(merged_spec.get('expr', name))
                if merged_spec.get('selection'):
                    all_text_parts.append(merged_spec['selection'])
                if merged_spec.get('group_by'):
                    all_text_parts.append(str(merged_spec['group_by']))
            all_text = ' '.join(all_text_parts)
            
            import re as _re
            for match in _re.finditer(r'\b(\w+)\.(\w+)\b', all_text):
                sf_name, col_name = match.group(1), match.group(2)
                if sf_name in sf_names:
                    dot_ref = f"{sf_name}.{col_name}"
                    flat_ref = f"{sf_name}_{col_name}"
                    if flat_ref not in df_for_plot.columns and dot_ref not in subframe_replacements:
                        try:
                            sf = self.get_subframe(sf_name)
                            index_cols = self._subframes.get_entry(sf_name)['index']
                            if isinstance(index_cols, str):
                                index_cols = [index_cols]
                            join_idx, missing = self._compute_join_indices(sf_name, index_cols)
                            if col_name in sf.df.columns:
                                if df_for_plot is self.df:
                                    df_for_plot = df_for_plot.copy()
                                df_for_plot[flat_ref] = sf.df[col_name].values[join_idx]
                                subframe_replacements[dot_ref] = flat_ref
                        except Exception:
                            pass
            
            # Rewrite all specs: replace Sub.col → Sub_col
            if subframe_replacements:
                for name, spec in specs.items():
                    for dot_ref, flat_ref in subframe_replacements.items():
                        if 'expr' in spec:
                            spec['expr'] = spec['expr'].replace(dot_ref, flat_ref)
                        if 'selection' in spec and spec['selection']:
                            spec['selection'] = spec['selection'].replace(dot_ref, flat_ref)
                        if 'group_by' in spec and isinstance(spec.get('group_by'), str):
                            spec['group_by'] = spec['group_by'].replace(dot_ref, flat_ref)
        
        # Delegate to dfdraw batch
        plotter = DFDraw(df_for_plot)
        plotter._data_source = self  # For duck-typed axis title lookup
        
        results = plotter.draw_batch(
            specs=specs,
            save_dir=save_dir,
            defaults=defaults,
            on_error=on_error,
            verbose=verbose,
            **kwargs
        )
        
        # Cleanup if requested
        if effective_clear:
            we_added = self._get_materialized_aliases() - already_materialized
            if we_added:
                if verbose:
                    print(f"Clearing {len(we_added)} materialized aliases")
                self.drop_materialized(we_added)
        
        return results

    # =========================================================================
    # Phase 12.4b1: draw_figures() - Composed multi-subplot figures
    # =========================================================================

    def draw_figures(
        self,
        specs: list,
        save_dir: str = None,
        *,
        defaults: dict = None,
        lazy: bool = None,
        clear_after: bool = None,
        max_entries: int = None,
        entry_begin: int = None,
        entry_end: int = None,
        entry_mask: np.ndarray = None,
        on_error: str = 'skip',
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate multiple figures with composed subplots from declarative specs.
        
        Unlike draw_batch() which creates separate figures, this creates
        multi-subplot canvases suitable for QA dashboards.
        
        Args:
            specs: List of figure specifications. Each spec is a dict:
                - name: Figure identifier (optional, auto-generated if missing)
                - plots: List of plot specs (required)
                - suptitle: Figure super-title (optional)
                - ncols: Number of columns in grid (default: 2)
                - figsize: (width, height) tuple (optional, auto-calculated)
                - savefig: Filename to save (optional)
                - sharex/sharey: Axis sharing (optional, default: False)
            save_dir: Directory to save all figures (combined with savefig)
            defaults: Default parameters applied to all plots
            lazy: Auto-materialize aliases. Default: self.draw_lazy
            clear_after: Drop materialized aliases after. Default: self.draw_clear_after
            max_entries: Limit rows for performance
            entry_begin/entry_end: Entry range selection
            entry_mask: Boolean or integer mask for entry selection
            on_error: 'skip' (continue on error) or 'raise' (stop on error)
            verbose: Print progress messages
            **kwargs: Additional defaults for all plots
            
        Returns:
            Dict of {figure_name: {'fig': Figure, 'axes': list, 'stats': list}}
            On error with on_error='skip': includes 'error' key instead
            
        Example:
            specs = [
                {
                    'name': 'residuals',
                    'suptitle': 'TPC Residuals QA',
                    'ncols': 2,
                    'savefig': 'residuals.png',
                    'plots': [
                        {'expr': 'dyC2', 'bins': 100},
                        {'expr': 'dzC2', 'bins': 100},
                        {'expr': 'dyC2:row', 'type': 'profile'},
                        {'expr': 'dzC2:row', 'type': 'profile'},
                    ]
                }
            ]
            results = aDF.draw_figures(specs, save_dir='qa/')
            
        Note:
            Plot specs support short form: 'column' expands to {'expr': 'column'}
        """
        # Import dfdraw
        try:
            from dfextensions.dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw package not found. Install it or ensure it's in your path."
            )
        
        # Validate specs structure
        self._validate_figure_specs(specs)
        
        # Resolve parameters with 3-level precedence
        effective_lazy = self._resolve_draw_param(lazy, 'lazy')
        effective_clear = self._resolve_draw_param(clear_after, 'clear_after')
        
        # Merge defaults
        merged_defaults = {**(defaults or {}), **kwargs}
        
        # Track pre-existing materialized aliases
        already_materialized = self._get_materialized_aliases()
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Pre-scan all specs to collect needed aliases
        # ═══════════════════════════════════════════════════════════════════
        
        all_needed = set()
        for fig_spec in specs:
            plots = fig_spec.get('plots', [])
            for plot_spec in plots:
                # Normalize short form: 'column' -> {'expr': 'column'}
                if isinstance(plot_spec, str):
                    plot_spec = {'expr': plot_spec}
                
                merged_plot = {**merged_defaults, **plot_spec}
                expr = merged_plot.get('expr', '')
                group_by = merged_plot.get('group_by')
                color = merged_plot.get('color')
                
                all_needed.update(self._parse_expr_aliases(expr, group_by, color))
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Batch-load branches in lazy reader mode
        # ═══════════════════════════════════════════════════════════════════
        
        if self._lazy_reader is not None:
            all_required = set()
            for fig_spec in specs:
                for plot_spec in fig_spec.get('plots', []):
                    if isinstance(plot_spec, str):
                        plot_spec = {'expr': plot_spec}
                    merged_plot = {**merged_defaults, **plot_spec}
                    required = self.get_required_branches(
                        expr=merged_plot.get('expr', ''),
                        selection=merged_plot.get('selection'),
                        group_by=merged_plot.get('group_by'),
                        color=merged_plot.get('color')
                    )
                    all_required.update(required)
            
            branches_to_load = all_required - self._lazy_reader.loaded_branches
            
            # Filter out subframe names (Phase 6.8a fix)
            all_subframes = set(self._subframes.subframes.keys()) | set(getattr(self, '_subframe_readers', {}).keys())
            branches_to_load = branches_to_load - all_subframes
            
            if branches_to_load:
                if verbose:
                    print(f"[draw_figures] Loading {len(branches_to_load)} branches")
                self.ensure_branches(list(branches_to_load))
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Materialize all aliases at once (if lazy)
        # ═══════════════════════════════════════════════════════════════════
        
        if effective_lazy:
            to_materialize = all_needed - already_materialized
            if to_materialize:
                if verbose:
                    print(f"[draw_figures] Materializing {len(to_materialize)} aliases")
                self.materialize_aliases(names=list(to_materialize))
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Prepare DataFrame (with entry selection if specified)
        # ═══════════════════════════════════════════════════════════════════
        
        has_entry_selection = (entry_begin is not None or 
                              entry_end is not None or 
                              entry_mask is not None)
        
        if has_entry_selection:
            df_subset = self._apply_entry_selection(entry_begin, entry_end, entry_mask)
        else:
            df_subset = self.df
        
        # Apply max_entries limit
        if max_entries and len(df_subset) > max_entries:
            df_subset = df_subset.iloc[:max_entries]
            if verbose:
                print(f"[draw_figures] Limited to {max_entries} entries")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: Generate figures
        # ═══════════════════════════════════════════════════════════════════
        
        results = {}
        
        for fig_idx, fig_spec in enumerate(specs):
            fig_name = fig_spec.get('name', f'figure_{fig_idx}')
            
            try:
                fig_result = self._draw_single_figure(
                    fig_spec=fig_spec,
                    df=df_subset,
                    defaults=merged_defaults,
                    save_dir=save_dir,
                    on_error=on_error,
                    verbose=verbose,
                )
                results[fig_name] = fig_result
                
                if verbose:
                    n_plots = len(fig_spec.get('plots', []))
                    print(f"[draw_figures] Generated: {fig_name} ({n_plots} plots)")
                    
            except Exception as e:
                if on_error == 'raise':
                    raise
                if verbose:
                    print(f"[draw_figures] Error in '{fig_name}': {e}")
                results[fig_name] = {'fig': None, 'axes': [], 'stats': [], 'error': str(e)}
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: Cleanup (if requested)
        # ═══════════════════════════════════════════════════════════════════
        
        if effective_clear and not has_entry_selection:
            we_added = self._get_materialized_aliases() - already_materialized
            if we_added:
                if verbose:
                    print(f"[draw_figures] Clearing {len(we_added)} materialized aliases")
                self.drop_materialized(we_added)
        
        return results

    def _validate_figure_specs(self, specs):
        """
        Validate figure specifications structure.
        
        Args:
            specs: List of figure specs to validate
            
        Raises:
            ValueError: If specs structure is invalid
        """
        if not isinstance(specs, list):
            raise ValueError(
                f"specs must be a list of figure specifications, got {type(specs).__name__}"
            )
        
        if len(specs) == 0:
            raise ValueError("specs list cannot be empty")
        
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"specs[{i}] must be a dict, got {type(spec).__name__}"
                )
            
            if 'plots' not in spec:
                raise ValueError(
                    f"specs[{i}] missing required 'plots' key"
                )
            
            plots = spec['plots']
            if not isinstance(plots, list):
                raise ValueError(
                    f"specs[{i}]['plots'] must be a list, got {type(plots).__name__}"
                )
            
            if len(plots) == 0:
                raise ValueError(
                    f"specs[{i}]['plots'] cannot be empty"
                )
            
            for j, plot in enumerate(plots):
                if isinstance(plot, str):
                    continue  # Short form is valid
                if not isinstance(plot, dict):
                    raise ValueError(
                        f"specs[{i}]['plots'][{j}] must be str or dict, "
                        f"got {type(plot).__name__}"
                    )
                if 'expr' not in plot:
                    raise ValueError(
                        f"specs[{i}]['plots'][{j}] missing required 'expr' key"
                    )

    def _draw_single_figure(
        self,
        fig_spec: dict,
        df: pd.DataFrame,
        defaults: dict,
        save_dir: str,
        on_error: str,
        verbose: bool,
    ):
        """
        Draw a single figure with multiple subplots.
        
        Args:
            fig_spec: Figure specification dict
            df: DataFrame to plot from
            defaults: Default plot parameters
            save_dir: Directory for saving
            on_error: 'skip' or 'raise'
            verbose: Print progress
            
        Returns:
            Dict with 'fig', 'axes', 'stats' keys
        """
        from dfextensions.dfdraw import DFDraw
        
        plots = fig_spec.get('plots', [])
        ncols = fig_spec.get('ncols', 2)
        nrows = (len(plots) + ncols - 1) // ncols
        
        # Calculate figure size
        figsize = fig_spec.get('figsize')
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        
        sharex = fig_spec.get('sharex', False)
        sharey = fig_spec.get('sharey', False)
        
        # Create figure and axes grid
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=sharex, sharey=sharey,
                                 squeeze=False)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten().tolist()
        
        # Add super-title
        suptitle = fig_spec.get('suptitle')
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
        
        # Create plotter with data source for axis labels
        plotter = DFDraw(df)
        plotter._data_source = self
        
        stats_list = []
        
        for idx, plot_spec in enumerate(plots):
            ax = axes_flat[idx]
            
            # Normalize short form
            if isinstance(plot_spec, str):
                plot_spec = {'expr': plot_spec}
            
            # Merge with defaults (plot-level overrides defaults)
            merged = {**defaults, **plot_spec}
            expr = merged.pop('expr')
            plot_type = merged.pop('type', 'auto')
            title = merged.pop('title', None)
            
            try:
                # Determine plot method
                method_name = self._resolve_plot_type(expr, plot_type)
                plot_func = getattr(plotter, method_name)
                
                # Draw on the specific axis
                _, _, stats = plot_func(expr, ax=ax, **merged)
                stats_list.append(stats)
                
                # Set title if provided
                if title:
                    ax.set_title(title)
                    
            except Exception as e:
                if on_error == 'raise':
                    raise
                # Show error on plot
                ax.text(0.5, 0.5, f'Error:\n{e}', 
                       ha='center', va='center',
                       transform=ax.transAxes, 
                       color='red', fontsize=9,
                       wrap=True)
                ax.set_title(f"[ERROR] {expr}")
                stats_list.append(None)
        
        # Hide unused axes
        for idx in range(len(plots), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        if suptitle:
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        # Save if requested
        savefig = fig_spec.get('savefig')
        if savefig:
            save_path = savefig
            if save_dir:
                save_path = Path(save_dir) / savefig
                save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if verbose:
                print(f"[draw_figures] Saved: {save_path}")
        
        return {
            'fig': fig,
            'axes': axes_flat[:len(plots)],
            'stats': stats_list
        }

    # =========================================================================
    # Phase 12.4b2: Fit Result Registration and QA Visualization
    # =========================================================================

    # Default validation thresholds for fit quality assessment
    _FIT_VALIDATION_DEFAULTS = {
        'pull_mean_threshold': 0.1,
        'pull_std_min': 0.8,
        'pull_std_max': 1.2,
        'outlier_threshold': 5.0,
        'outlier_max_fraction': 0.05,
    }

    def _validate_fit_metadata(self, metadata: dict, validate: str = 'warn') -> list:
        """
        Validate fit metadata schema.
        
        Args:
            metadata: Metadata dict from make_parallel_fit_vX
            validate: 'raise' | 'warn' | 'skip'
            
        Returns:
            List of validation issues (empty if valid)
            
        Raises:
            ValueError: If validate='raise' and issues found
        """
        issues = []
        
        if not isinstance(metadata, dict):
            issues.append(f"metadata must be dict, got {type(metadata).__name__}")
            if validate == 'raise':
                raise ValueError(f"Metadata validation failed: {issues}")
            elif validate == 'warn':
                import warnings
                warnings.warn(f"Metadata validation issues: {issues}")
            return issues
        
        # Check required top-level keys
        required_keys = ['formulas', 'columns', 'parameters']
        for key in required_keys:
            if key not in metadata:
                issues.append(f"Missing required key: '{key}'")
        
        # Check columns structure
        if 'columns' in metadata:
            columns = metadata['columns']
            if not isinstance(columns, dict):
                issues.append("'columns' must be a dict")
            else:
                if 'gb_columns' not in columns:
                    issues.append("Missing 'columns.gb_columns'")
                if 'fit_columns' not in columns:
                    issues.append("Missing 'columns.fit_columns'")
        
        # Check formulas structure
        if 'formulas' in metadata:
            if not isinstance(metadata['formulas'], dict):
                issues.append("'formulas' must be a dict")
        
        # Check parameters structure
        if 'parameters' in metadata:
            if not isinstance(metadata['parameters'], dict):
                issues.append("'parameters' must be a dict")
        
        # Handle validation mode
        if issues:
            msg = f"Metadata validation issues: {issues}"
            if validate == 'raise':
                raise ValueError(msg)
            elif validate == 'warn':
                import warnings
                warnings.warn(msg)
        
        return issues

    def register_fit_result(
        self,
        name: str,
        dfGB: pd.DataFrame,
        metadata: dict = None,
        *,
        index_columns: list = None,
        add_predictions: bool = True,
        add_residuals: bool = True,
        add_pulls: bool = True,
        pull_type: str = None,
        auto_alias_subframe: bool = True,
        prediction_dtype=None,
        residual_dtype=None,
        pull_dtype=None,
        validate: str = 'warn',
    ) -> 'AliasDataFrame':
        """
        Register groupby fit result as subframe with auto-generated aliases.
        
        This is the main integration point for fit results from groupby-regression.
        It performs:
        1. Registers dfGB as a subframe with appropriate index columns
        2. Auto-aliases subframe columns (makes coefficients available)
        3. Creates prediction aliases from metadata['formulas']
        4. Creates residual aliases from metadata['residual_formulas']
        5. Creates pull aliases from metadata['pull_formulas']
        6. Stores metadata for later use (draw_fit_summary, schema export)
        
        Args:
            name: Subframe name (e.g., "DTrackFitAll")
            dfGB: DataFrame with fit coefficients from make_parallel_fit_vX
            metadata: Metadata dict from make_parallel_fit_vX(return_metadata=True)
                      If None, only registers subframe (backward compatibility)
            index_columns: Override gb_columns from metadata. Required if metadata=None.
            add_predictions: Create prediction aliases from metadata['formulas']
            add_residuals: Create residual aliases from metadata['residual_formulas']
            add_pulls: Create pull aliases from metadata['pull_formulas']
            pull_type: Which pull to create: 'rms', 'mad', or 'both'
                       None → use metadata['parameters']['pull_default'] or 'rms'
            auto_alias_subframe: Call auto_alias_subframe() after registration
            prediction_dtype: Override dtype for prediction aliases (default: float32)
            residual_dtype: Override dtype for residual aliases (default: float32)
            pull_dtype: Override dtype for pull aliases (default: float32)
            validate: Metadata validation mode: 'raise' | 'warn' | 'skip'
            
        Returns:
            AliasDataFrame wrapping dfGB (the registered subframe)
            
        Raises:
            ValueError: If metadata is invalid (when validate='raise')
            ValueError: If metadata=None and index_columns not provided
            
        Example:
            # With metadata (recommended)
            _, dfGB, meta = make_parallel_fit_v4(..., return_metadata=True)
            aDF.register_fit_result("DTrackFit", dfGB, meta)
            
            # Access auto-generated aliases
            aDF.draw('dyC2_pred_DTrackFit:sector', type='profile')
            aDF.draw('dyC2_pull_DTrackFit', bins=100)  # Should be ~N(0,1)
            
            # Without metadata (backward compat)
            aDF.register_fit_result("Fit", dfGB, index_columns=['track', 'orbit'])
        """
        import warnings
        
        # Phase 1: Handle backward compatibility (no metadata)
        if metadata is None:
            if index_columns is None:
                raise ValueError(
                    "index_columns required when metadata not provided. "
                    "Use make_parallel_fit_vX(return_metadata=True) for full functionality."
                )
            warnings.warn(
                f"Registering '{name}' without metadata. "
                "Auto-generated prediction/residual/pull aliases will not be available.",
                UserWarning
            )
            aDFGB = AliasDataFrame(dfGB)
            self.register_subframe(name, aDFGB, index_columns=index_columns)
            if auto_alias_subframe:
                self.auto_alias_subframe(name)
            return aDFGB
        
        # Phase 2: Validate metadata schema
        self._validate_fit_metadata(metadata, validate=validate)
        
        # Phase 3: Check for duplicate registration
        if not hasattr(self, '_fit_metadata'):
            self._fit_metadata = {}
        
        if name in self._fit_metadata:
            old_suffix = self._fit_metadata[name].get('parameters', {}).get('suffix', '?')
            warnings.warn(
                f"Overwriting existing fit result '{name}' (old suffix: {old_suffix})",
                UserWarning
            )
        
        # Phase 4: Register subframe
        idx_cols = index_columns or metadata['columns']['gb_columns']
        aDFGB = AliasDataFrame(dfGB)
        self.register_subframe(name, aDFGB, index_columns=idx_cols)
        
        # Phase 5: Auto-alias subframe columns
        if auto_alias_subframe:
            self.auto_alias_subframe(name)
        
        # Phase 6: Store metadata
        self._fit_metadata[name] = metadata
        
        # Phase 7: Resolve pull_type from metadata if not specified
        effective_pull_type = pull_type
        if effective_pull_type is None:
            effective_pull_type = metadata.get('parameters', {}).get('pull_default', 'rms')
        
        # Default dtypes
        pred_dtype = prediction_dtype if prediction_dtype is not None else np.float32
        res_dtype = residual_dtype if residual_dtype is not None else np.float32
        pl_dtype = pull_dtype if pull_dtype is not None else np.float32
        
        # Phase 8a: Add prediction aliases
        if add_predictions and 'formulas' in metadata:
            for alias_name, formula in metadata['formulas'].items():
                self.add_alias(alias_name, formula, dtype=pred_dtype)
        
        # Phase 8b: Add residual aliases
        if add_residuals and 'residual_formulas' in metadata:
            for alias_name, formula in metadata['residual_formulas'].items():
                self.add_alias(alias_name, formula, dtype=res_dtype)
        
        # Phase 8c: Add pull aliases (filtered by pull_type)
        if add_pulls and 'pull_formulas' in metadata:
            for alias_name, formula in metadata['pull_formulas'].items():
                is_mad = '_pull_mad_' in alias_name or alias_name.endswith('_pull_mad')
                
                if effective_pull_type == 'rms' and is_mad:
                    continue
                if effective_pull_type == 'mad' and not is_mad:
                    continue
                # effective_pull_type == 'both' includes all
                
                self.add_alias(alias_name, formula, dtype=pl_dtype)
        
        return aDFGB

    def get_fit_metadata(self, name: str = None) -> dict:
        """
        Get stored fit metadata.
        
        Args:
            name: Specific fit name, or None for all registered fits
            
        Returns:
            If name provided: metadata dict for that fit
            If name is None: dict of {name: metadata} for all fits
            
        Raises:
            KeyError: If name not found in registered fits
            
        Example:
            meta = aDF.get_fit_metadata("DTrackFit")
            print(f"Fit columns: {meta['columns']['fit_columns']}")
        """
        all_meta = getattr(self, '_fit_metadata', {})
        
        if name is None:
            return dict(all_meta)
        
        if name not in all_meta:
            available = list(all_meta.keys())
            raise KeyError(
                f"No fit result registered with name '{name}'. "
                f"Available: {available}"
            )
        
        return all_meta[name]

    def list_fit_results(self) -> list:
        """
        List all registered fit result names.
        
        Returns:
            List of fit names registered via register_fit_result()
            
        Example:
            print(aDF.list_fit_results())  # ['DTrackFit', 'DSectorCorr']
        """
        return list(getattr(self, '_fit_metadata', {}).keys())

    def _apply_pull_transform(self, pull_alias: str, transform: str) -> str:
        """
        Create transformed pull alias if needed (idempotent).
        
        Args:
            pull_alias: Original pull alias name
            transform: None | 'asinh' | 'tanh'
            
        Returns:
            Alias name to use (original or transformed)
        """
        if transform is None:
            return pull_alias
        
        new_alias = f'{pull_alias}_{transform}'
        
        # Idempotent: don't recreate if exists
        if new_alias not in self.aliases:
            if transform == 'asinh':
                self.add_alias(new_alias, f'np.arcsinh({pull_alias})')
            elif transform == 'tanh':
                self.add_alias(new_alias, f'np.tanh({pull_alias})')
            else:
                raise ValueError(
                    f"Unknown pull_transform: '{transform}'. Use None, 'asinh', or 'tanh'"
                )
        
        return new_alias

    def _add_gaussian_overlay(self, ax, mu: float = 0, sigma: float = 1, 
                               label: str = 'N(0,1)', color: str = 'r', 
                               linestyle: str = '--', linewidth: float = 2):
        """
        Add N(0,1) reference curve to pull histogram.
        
        Automatically scales the Gaussian PDF to match histogram counts.
        Handles both Rectangle patches (standard hist) and Polygon patches (DFDraw filled hist).
        
        Args:
            ax: Matplotlib axis
            mu: Mean of Gaussian
            sigma: Std of Gaussian
            label: Legend label
            color: Line color
            linestyle: Line style
            linewidth: Line width
        """
        # Try to get scaling from histogram
        scale_factor = 1.0
        
        patches = ax.patches
        if patches:
            p = patches[0]
            if hasattr(p, 'get_height'):
                # Standard Rectangle patches (bar histogram)
                heights = [patch.get_height() for patch in patches]
                widths = [patch.get_width() for patch in patches]
                total_count = sum(h * w for h, w in zip(heights, widths))
                scale_factor = total_count
            elif hasattr(p, 'get_xy'):
                # Polygon patch (DFDraw filled histogram)
                # Get y-axis limits as proxy for histogram scale
                ylim = ax.get_ylim()
                # Estimate: use max y value * approximate bin width
                # The y-limits typically extend a bit beyond the data
                max_y = ylim[1] * 0.9  # 90% of y-limit as estimate
                xlim = ax.get_xlim()
                x_range = xlim[1] - xlim[0]
                # Assume ~50 bins for typical histogram
                approx_bin_width = x_range / 50
                # Scale so Gaussian peak matches histogram peak
                scale_factor = max_y * approx_bin_width * np.sqrt(2 * np.pi) * sigma
        
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        y = y * scale_factor  # Scale PDF to histogram counts
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    def _compute_fit_validation(self, name: str, fit_columns: list = None,
                                thresholds: dict = None) -> dict:
        """
        Compute validation metrics for fit result.
        
        Args:
            name: Registered fit name
            fit_columns: Subset of fit columns (default: all)
            thresholds: Override validation thresholds (merged with defaults)
            
        Returns:
            Dict with per-column validation results and _overall_pass
        """
        meta = self._fit_metadata[name]
        suffix = meta['parameters']['suffix']
        cols = fit_columns or meta['columns']['fit_columns']
        
        # Merge custom thresholds with defaults
        effective_thresholds = dict(self._FIT_VALIDATION_DEFAULTS)
        if thresholds:
            effective_thresholds.update(thresholds)
        
        validation = {}
        all_pass = True
        
        for col in cols:
            pull_alias = f'{col}_pull{suffix}'
            
            # Skip if pull alias doesn't exist
            if pull_alias not in self.aliases:
                validation[col] = {'error': f'Pull alias {pull_alias} not found'}
                all_pass = False
                continue
            
            try:
                # Materialize if needed
                if pull_alias not in self.df.columns:
                    self.materialize_alias(pull_alias)
                
                pull_values = self.df[pull_alias].dropna().values
                
                if len(pull_values) == 0:
                    validation[col] = {'error': 'No valid pull values'}
                    all_pass = False
                    continue
                
                # Compute metrics
                pull_mean = float(np.mean(pull_values))
                pull_std = float(np.std(pull_values))
                
                # Outlier fraction (|pull| > threshold)
                outlier_count = np.sum(np.abs(pull_values) > effective_thresholds['outlier_threshold'])
                outlier_fraction = outlier_count / len(pull_values)
                
                # Pass/fail checks
                pull_mean_pass = abs(pull_mean) < effective_thresholds['pull_mean_threshold']
                pull_std_pass = (effective_thresholds['pull_std_min'] < pull_std < effective_thresholds['pull_std_max'])
                outlier_pass = outlier_fraction < effective_thresholds['outlier_max_fraction']
                
                col_pass = pull_mean_pass and pull_std_pass and outlier_pass
                
                validation[col] = {
                    'pull_mean': pull_mean,
                    'pull_std': pull_std,
                    'pull_mean_pass': pull_mean_pass,
                    'pull_std_pass': pull_std_pass,
                    'outlier_fraction': float(outlier_fraction),
                    'outlier_pass': outlier_pass,
                    '_pass': col_pass,
                }
                
                if not col_pass:
                    all_pass = False
                    
            except Exception as e:
                validation[col] = {'error': str(e)}
                all_pass = False
        
        validation['_overall_pass'] = all_pass
        return validation

    # =========================================================================
    # Phase 12.4b5: Validation Display Methods
    # =========================================================================
    
    def _add_validation_indicator(self, ax, passed: bool):
        """
        Add PASS/FAIL text indicator to axis.
        
        Fit-specific method for validation visualization.
        
        Args:
            ax: Matplotlib axis
            passed: Whether validation passed
            
        Returns:
            matplotlib.text.Text: The created text artist
        """
        if passed:
            text = "PASS"
            color = 'green'
        else:
            text = "FAIL"
            color = 'red'
        
        text_artist = ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        return text_artist

    def _add_validation_summary(self, fig, validation_results: dict):
        """
        Add validation summary panel to figure.
        
        Fit-specific method for overall validation display.
        
        Args:
            fig: Matplotlib figure
            validation_results: Dict from _compute_fit_validation()
            
        Returns:
            matplotlib.text.Text: The created text artist
        """
        lines = ["Validation Summary", "=" * 20]
        
        overall = validation_results.get('_overall_pass', False)
        lines.append(f"Overall: {'PASS' if overall else 'FAIL'}")
        lines.append("")
        
        for col, metrics in validation_results.items():
            if col.startswith('_'):
                continue
            if not isinstance(metrics, dict):
                continue
                
            col_pass = metrics.get('pass', False)
            status = "PASS" if col_pass else "FAIL"
            
            pull_mean = metrics.get('pull_mean', float('nan'))
            pull_std = metrics.get('pull_std', float('nan'))
            
            lines.append(f"{col}: {status}")
            lines.append(f"  μ={pull_mean:.3f}, σ={pull_std:.3f}")
        
        text = "\n".join(lines)
        
        text_artist = fig.text(
            0.02, 0.02, text, fontsize=8, fontfamily='monospace',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
        
        return text_artist

    def _compute_statistics(self, name: str) -> dict:
        """
        Compute statistics for fit columns.
        
        Args:
            name: Fit result name
            
        Returns:
            dict: Statistics per column {col: {'pull_mean': x, 'pull_std': y, ...}}
        """
        import numpy as np
        
        if not hasattr(self, '_fit_metadata') or name not in self._fit_metadata:
            return {}
        
        metadata = self._fit_metadata[name]
        fit_columns = metadata.get('columns', {}).get('fit_columns', [])
        suffix = metadata.get('parameters', {}).get('suffix', '')
        
        stats = {}
        
        for col in fit_columns:
            pull_alias = f'{col}_pull{suffix}'
            delta_alias = f'{col}_delta{suffix}'
            
            col_stats = {}
            
            # Pull statistics
            if pull_alias in self.aliases or pull_alias in self.df.columns:
                try:
                    self.materialize_aliases(names=[pull_alias])
                    values = self.df[pull_alias].dropna().values
                    col_stats['pull_mean'] = float(np.mean(values))
                    col_stats['pull_std'] = float(np.std(values))
                    col_stats['pull_n'] = len(values)
                except Exception:
                    pass
            
            # Delta statistics
            if delta_alias in self.aliases or delta_alias in self.df.columns:
                try:
                    self.materialize_aliases(names=[delta_alias])
                    values = self.df[delta_alias].dropna().values
                    col_stats['delta_mean'] = float(np.mean(values))
                    col_stats['delta_std'] = float(np.std(values))
                    col_stats['delta_n'] = len(values)
                except Exception:
                    pass
            
            if col_stats:
                stats[col] = col_stats
        
        return stats

    def _build_residuals_figure_spec(
        self, 
        name: str, 
        cols: list, 
        meta: dict, 
        categories: set,
        pull_transform: str = None,
    ) -> dict:
        """Build figure spec for residual and pull distributions (flat list)."""
        suffix = meta['parameters']['suffix']
        
        plots = []
        for col in cols:
            if 'delta_1d' in categories:
                plots.append({
                    'expr': f'{col}_delta{suffix}',
                    'type': 'hist',
                    'bins': 100,
                    'title': f'{col} Δ',
                })
            
            if 'pull_1d' in categories:
                pull_alias = f'{col}_pull{suffix}'
                if pull_transform:
                    pull_alias = self._apply_pull_transform(pull_alias, pull_transform)
                
                title = f'{col} pull'
                if pull_transform:
                    title += f' ({pull_transform})'
                
                plots.append({
                    'expr': pull_alias,
                    'type': 'hist',
                    'bins': 100,
                    'title': title,
                })
        
        # Determine ncols based on categories
        n_plot_cols = sum([
            'delta_1d' in categories,
            'pull_1d' in categories,
        ])
        
        return {
            'name': f'{name}_residuals',
            'suptitle': f'{name}: Residual Distributions',
            'plots': plots,
            'ncols': max(n_plot_cols, 1),
            'figsize': (5 * max(n_plot_cols, 1), 3 * len(cols)),
            'savefig': f'{name}_residuals',
        }

    def _build_error_diagnostics_spec(
        self, 
        name: str, 
        cols: list, 
        meta: dict, 
        categories: set,
    ) -> dict:
        """Build figure spec for delta/pull vs error scatter plots (flat list)."""
        suffix = meta['parameters']['suffix']
        
        plots = []
        for col in cols:
            rms_col = f'{col}_rms{suffix}'
            
            if 'delta_vs_error' in categories:
                plots.append({
                    'expr': f'{col}_delta{suffix}:{rms_col}',
                    'type': 'scatter',
                    'title': f'{col}: Δ vs σ',
                    'alpha': 0.3,
                })
            
            if 'pull_vs_error' in categories:
                plots.append({
                    'expr': f'{col}_pull{suffix}:{rms_col}',
                    'type': 'scatter',
                    'title': f'{col}: pull vs σ',
                    'alpha': 0.3,
                })
        
        n_plot_cols = sum([
            'delta_vs_error' in categories,
            'pull_vs_error' in categories,
        ])
        
        return {
            'name': f'{name}_error_diagnostics',
            'suptitle': f'{name}: Error Diagnostics',
            'plots': plots,
            'ncols': max(n_plot_cols, 1),
            'figsize': (5 * max(n_plot_cols, 1), 4 * len(cols)),
            'savefig': f'{name}_error_diagnostics',
        }

    def _build_quality_figure_spec(self, name: str, cols: list, meta: dict) -> dict:
        """Build figure spec for RMS/MAD distributions (flat list)."""
        suffix = meta['parameters']['suffix']
        
        plots = []
        for col in cols:
            quality_cols = meta['columns'].get('quality', {}).get(col, [])
            
            rms_col = f'{col}_rms{suffix}'
            mad_col = f'{col}_mad{suffix}'
            
            if rms_col in quality_cols or not quality_cols:
                plots.append({
                    'expr': rms_col,
                    'type': 'hist',
                    'bins': 100,
                    'title': f'{col} RMS',
                })
            
            if mad_col in quality_cols or not quality_cols:
                plots.append({
                    'expr': mad_col,
                    'type': 'hist',
                    'bins': 100,
                    'title': f'{col} MAD',
                })
        
        return {
            'name': f'{name}_quality',
            'suptitle': f'{name}: Fit Quality (Error Estimates)',
            'plots': plots,
            'ncols': 2,
            'figsize': (10, 3 * len(cols)),
            'savefig': f'{name}_quality',
        }

    def _build_coefficients_figure_spec(self, name: str, col: str, meta: dict) -> dict:
        """Build figure spec for coefficient distributions for one fit column (flat list)."""
        suffix = meta['parameters']['suffix']
        coefs = meta['columns'].get('coefficients', {}).get(col, [])
        
        if not coefs:
            return None
        
        plots = []
        for coef in coefs:
            plots.append({
                'expr': coef,
                'type': 'hist',
                'bins': 100,
                'title': coef.replace(suffix, ''),
            })
        
        n_coefs = len(coefs)
        n_grid_cols = min(4, n_coefs)
        n_grid_rows = (n_coefs + n_grid_cols - 1) // n_grid_cols
        
        return {
            'name': f'{name}_coefficients_{col}',
            'suptitle': f'{name}: Coefficients for {col}',
            'plots': plots,
            'ncols': n_grid_cols,
            'figsize': (4 * n_grid_cols, 3 * n_grid_rows),
            'savefig': f'{name}_coefficients_{col}',
        }

    def _build_diagnostics_figure_spec(self, name: str, meta: dict) -> dict:
        """Build figure spec for fit diagnostics (nPoints, chi2, etc.) (flat list)."""
        diag_cols = meta['columns'].get('diagnostics', [])
        
        if not diag_cols:
            return None
        
        plots = []
        for diag in diag_cols:
            plots.append({
                'expr': diag,
                'type': 'hist',
                'bins': 100,
                'title': diag,
            })
        
        n_diag = len(diag_cols)
        n_grid_cols = min(4, n_diag)
        n_grid_rows = (n_diag + n_grid_cols - 1) // n_grid_cols
        
        return {
            'name': f'{name}_diagnostics',
            'suptitle': f'{name}: Fit Diagnostics',
            'plots': plots,
            'ncols': n_grid_cols,
            'figsize': (4 * n_grid_cols, 3 * n_grid_rows),
            'savefig': f'{name}_diagnostics',
        }

    def draw_fit_summary(
        self,
        name: str,
        save_dir: str = None,
        *,
        include: list = None,
        exclude: list = None,
        pull_transform: str = None,
        gaussian_overlay: bool = True,
        fit_columns: list = None,
        entry_end: int = None,
        save_format: str = 'png',
        figsize: tuple = None,
        dpi: int = 100,
        on_error: str = 'skip',
        verbose: bool = True,
        validation_thresholds: dict = None,  # Phase 12.4b3
        # Phase 12.4b5: Annotation parameters (all default False for backward compat)
        show_statistics: bool = False,
        show_validation: bool = False,
        show_summary: bool = False,
        **kwargs,
    ) -> dict:
        """
        Generate comprehensive QA plots for a registered fit result.
        
        Uses draw_figures() internally to create multi-panel dashboards
        with standardized layouts for fit quality assessment.
        
        Args:
            name: Registered fit name from register_fit_result()
            save_dir: Directory to save figures (None = don't save)
            include: List of plot categories to include (default: core categories)
                     Categories: 'delta_1d', 'pull_1d', 'delta_vs_error',
                                'pull_vs_error', 'coefficients', 'quality',
                                'diagnostics'
            exclude: List of plot categories to exclude
            pull_transform: Transform for pull display: None, 'asinh', 'tanh'
            gaussian_overlay: Add N(0,1) overlay to pull histograms
            fit_columns: Subset of fit columns to plot (default: all from metadata)
            entry_end: Limit entries for quick testing (default: None = all)
            save_format: 'png' | 'pdf' | 'both'
            figsize: Override figure size (default: auto-calculated)
            dpi: Resolution for saved figures
            on_error: 'skip' or 'raise' on plot errors
            verbose: Print progress
            validation_thresholds: Override validation thresholds. Keys:
                'pull_mean_threshold', 'pull_std_min', 'pull_std_max',
                'outlier_threshold', 'outlier_max_fraction'
            show_statistics: Add μ, σ, n annotations to histograms (Phase 12.4b5)
            show_validation: Add PASS/FAIL indicators to plots (Phase 12.4b5)
            show_summary: Add validation summary panel to figure (Phase 12.4b5)
            **kwargs: Passed to individual draw() calls
            
        Returns:
            Dict of {category_name: {'fig': fig, 'axes': axes, 'stats': stats}}
            Also includes '_validation' key with automated validation metrics
            Also includes '_statistics' key with computed statistics (Phase 12.4b5)
            
        Raises:
            KeyError: If name not found in registered fits
            
        Example:
            # Full QA suite
            aDF.draw_fit_summary("DTrackFit", save_dir="qa/")
            
            # Quick test (first 10k entries, skip coefficients)
            aDF.draw_fit_summary("DTrackFit", entry_end=10000, 
                                exclude=['coefficients', 'diagnostics'])
            
            # Only pull distributions with asinh transform
            aDF.draw_fit_summary("DTrackFit", include=['pull_1d'], 
                                pull_transform='asinh')
            
            # Custom validation thresholds
            aDF.draw_fit_summary("DTrackFit", 
                                validation_thresholds={'pull_mean_threshold': 0.05})
            
            # With statistics and validation display (Phase 12.4b5)
            aDF.draw_fit_summary("DTrackFit", 
                                show_statistics=True, 
                                show_validation=True,
                                show_summary=True)
        """
        import warnings
        
        # Phase 1: Validate fit exists
        if not hasattr(self, '_fit_metadata') or name not in self._fit_metadata:
            available = list(getattr(self, '_fit_metadata', {}).keys())
            raise KeyError(
                f"No fit result registered with name '{name}'. "
                f"Available: {available}"
            )
        
        meta = self._fit_metadata[name]
        suffix = meta['parameters']['suffix']
        
        # Phase 2: Large dataset warning
        if entry_end is None and len(self.df) > 1_000_000:
            warnings.warn(
                f"Large dataset ({len(self.df):,} rows). Consider using entry_end "
                f"for faster iteration. Example: entry_end=100000",
                UserWarning
            )
        
        # Phase 3: Determine which categories to generate
        default_categories = ['delta_1d', 'pull_1d', 'quality']
        all_categories = ['delta_1d', 'pull_1d', 'delta_vs_error', 'pull_vs_error',
                          'quality', 'coefficients', 'diagnostics']
        
        if include is not None:
            categories = set(include)
        else:
            categories = set(default_categories)
        
        if exclude:
            categories -= set(exclude)
        
        # Validate categories
        invalid = categories - set(all_categories)
        if invalid:
            warnings.warn(f"Unknown categories ignored: {invalid}")
            categories -= invalid
        
        # Phase 4: Determine fit columns to process
        cols = fit_columns or meta['columns']['fit_columns']
        
        # Phase 5: Build figure specs for draw_figures()
        figure_specs = []
        
        # Delta & Pull 1D distributions
        if 'delta_1d' in categories or 'pull_1d' in categories:
            residuals_spec = self._build_residuals_figure_spec(
                name, cols, meta, categories, pull_transform
            )
            if figsize:
                residuals_spec['figsize'] = figsize
            figure_specs.append(residuals_spec)
        
        # Delta/Pull vs Error
        if 'delta_vs_error' in categories or 'pull_vs_error' in categories:
            error_spec = self._build_error_diagnostics_spec(name, cols, meta, categories)
            if figsize:
                error_spec['figsize'] = figsize
            figure_specs.append(error_spec)
        
        # Quality (RMS/MAD)
        if 'quality' in categories:
            quality_spec = self._build_quality_figure_spec(name, cols, meta)
            if figsize:
                quality_spec['figsize'] = figsize
            figure_specs.append(quality_spec)
        
        # Coefficients (one figure per fit column)
        if 'coefficients' in categories:
            for col in cols:
                coef_spec = self._build_coefficients_figure_spec(name, col, meta)
                if coef_spec:
                    if figsize:
                        coef_spec['figsize'] = figsize
                    figure_specs.append(coef_spec)
        
        # Diagnostics
        if 'diagnostics' in categories:
            diag_spec = self._build_diagnostics_figure_spec(name, meta)
            if diag_spec:
                if figsize:
                    diag_spec['figsize'] = figsize
                figure_specs.append(diag_spec)
        
        if not figure_specs:
            if verbose:
                print(f"[draw_fit_summary] No figures to generate for categories: {categories}")
            return {'_validation': self._compute_fit_validation(name, cols, validation_thresholds)}
        
        # Phase 6: Handle save format
        if save_dir and save_format in ('png', 'both'):
            for spec in figure_specs:
                if 'savefig' in spec and not spec['savefig'].endswith('.png'):
                    spec['savefig'] = f"{spec['savefig']}.png"
        
        # Phase 7: Call draw_figures()
        # Note: lazy=True ensures aliases are materialized before plotting
        results = self.draw_figures(
            figure_specs,
            save_dir=save_dir,
            entry_end=entry_end,
            on_error=on_error,
            verbose=verbose,
            lazy=True,  # Required for alias resolution
            **kwargs
        )
        
        # Phase 8: Save PDF if requested
        if save_dir and save_format in ('pdf', 'both'):
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for fig_name, fig_data in results.items():
                if fig_data.get('fig') is not None:
                    pdf_name = f"{fig_name}.pdf"
                    fig_data['fig'].savefig(save_path / pdf_name, dpi=dpi, bbox_inches='tight')
                    if verbose:
                        print(f"[draw_fit_summary] Saved PDF: {save_path / pdf_name}")
        
        # Phase 9: Add Gaussian overlay to pull histograms
        if gaussian_overlay and 'pull_1d' in categories:
            residuals_key = f'{name}_residuals'
            if residuals_key in results and results[residuals_key].get('axes'):
                axes = results[residuals_key]['axes']
                only_pulls = 'delta_1d' not in categories
                
                for i, ax in enumerate(axes):
                    if only_pulls or (i % 2 == 1):
                        try:
                            self._add_gaussian_overlay(ax)
                        except Exception:
                            pass
        
        # Phase 10: Compute validation metrics
        validation = self._compute_fit_validation(name, cols, validation_thresholds)
        results['_validation'] = validation
        
        # Phase 11 (12.4b5): Compute and store statistics
        statistics = self._compute_statistics(name)
        results['_statistics'] = statistics
        
        # Phase 12 (12.4b5): Add annotations to residuals figure
        residuals_key = f'{name}_residuals'
        if residuals_key in results and results[residuals_key].get('axes'):
            fig_data = results[residuals_key]
            axes = fig_data['axes']
            fig = fig_data.get('fig')
            
            suffix = meta['parameters']['suffix']
            n_cols = len(cols)
            has_delta = 'delta_1d' in categories
            has_pull = 'pull_1d' in categories
            
            # Iterate through axes
            for i, ax in enumerate(axes):
                try:
                    # Determine which column and type (delta vs pull)
                    if has_delta and has_pull:
                        col_idx = i // 2
                        is_pull = (i % 2 == 1)
                    elif has_pull:
                        col_idx = i
                        is_pull = True
                    else:
                        col_idx = i
                        is_pull = False
                    
                    if col_idx >= n_cols:
                        continue
                    
                    col = cols[col_idx]
                    
                    # Get values for statistics
                    if is_pull:
                        alias = f'{col}_pull{suffix}'
                        expected_mean, expected_std = 0.0, 1.0
                    else:
                        alias = f'{col}_delta{suffix}'
                        expected_mean, expected_std = 0.0, None
                    
                    # Add statistics box (uses DFDraw if available, else inline)
                    if show_statistics and alias in self.df.columns:
                        values = self.df[alias].dropna().values
                        if len(values) > 0:
                            import numpy as np
                            mean = np.mean(values)
                            std = np.std(values)
                            n = len(values)
                            
                            lines = [f"n = {n:,}", f"μ = {mean:.3f}", f"σ = {std:.3f}"]
                            if is_pull:
                                lines.append(f"Δμ = {mean - expected_mean:+.3f}")
                                lines.append(f"Δσ = {std - expected_std:+.3f}")
                            
                            text = "\n".join(lines)
                            ax.text(
                                0.95, 0.95, text, transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='right',
                                fontsize=8, fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                            )
                    
                    # Add validation indicator
                    if show_validation and col in validation:
                        col_passed = validation[col].get('pass', False)
                        self._add_validation_indicator(ax, col_passed)
                        
                except Exception:
                    pass  # Skip annotation errors silently
            
            # Add summary panel to figure
            if show_summary and fig is not None:
                try:
                    self._add_validation_summary(fig, validation)
                    # Adjust layout to make room for summary
                    fig.subplots_adjust(bottom=0.15)
                except Exception:
                    pass
        
        return results

    def _load_specs_file_for_draw(self, path: str):
        """Load specs from JSON or YAML file for draw_batch."""
        from pathlib import Path
        
        path = Path(path)
        with open(path) as f:
            if path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML files: pip install pyyaml")
            else:
                data = json.load(f)
        
        # Handle 'plots' key if present
        if isinstance(data, dict) and 'plots' in data:
            return data['plots']
        return data
