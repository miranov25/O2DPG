import sys
import os; sys.path.insert(1, os.environ.get("O2DPG", "") + "/UTILS/dfextensions")
import pandas as pd
import numpy as np
import json
import uproot
import copy
import warnings
from datetime import datetime, timezone
try:
    import ROOT  # type: ignore
except ImportError as e:
    print(f"[AliasDataFrame] WARNING: ROOT import failed: {e}")
    ROOT = None
import matplotlib.pyplot as plt
import networkx as nx
import re
import ast

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
        "subframes": serialized.get("subframes", {}),
    }
    
    # Ensure compression has __meta__
    if "__meta__" not in result["compression"]:
        result["compression"]["__meta__"] = {
            "schema_version": 1,
            "state_machine": "CompressionState.v1"
        }
    
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

class AliasDataFrame:
    """
    AliasDataFrame allows for defining and evaluating lazy-evaluated column aliases
    on top of a pandas DataFrame, including nested subframes with hierarchical indexing.
    
    Phase 4: Uses unified _schema dict as single source of truth.
    """
    
    def __init__(self, df, schema_id=None):
        """
        Initialize AliasDataFrame with unified schema structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            The underlying pandas DataFrame
        schema_id : str, optional
            User-defined identifier for this schema (e.g., "miranov_lxplus_TPC_calib_v3").
            Useful for parameter scans, test studies, and provenance tracking.
        
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
        if item in self.df.columns:
            return self.df[item]
        if item in self.aliases:
            self.materialize_alias(item)
            return self.df[item]
        sf = self._subframes.get(item)
        if sf is not None:
            return sf
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
        
        Phase 4: Also writes to _schema["subframes"] for metadata persistence.
        """
        # Add to runtime registry
        self._subframes.add_subframe(name, adf, index_columns, pre_index=pre_index)
        
        # Also write to schema for persistence
        self._schema["subframes"][name] = {"index": index_columns}

    def get_subframe(self, name):
        return self._subframes.get(name)

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

        return env

    def _prepare_subframe_joins(self, expr, warn_missing_keys=True, alias_name=None):
        """
        Prepare subframe joins for expression evaluation.
        
        Detects dotted references like `T.mX` and performs left joins to bring
        subframe columns into the main DataFrame.
        
        Parameters
        ----------
        expr : str
            Expression containing potential subframe references (e.g., "x - T.mX")
        warn_missing_keys : bool, default=True
            If True, emit warning when main frame keys are not found in subframe.
            Missing keys produce NaN values (rows are never dropped).
        alias_name : str, optional
            Name of the alias being evaluated (for warning messages)
            
        Returns
        -------
        str
            Modified expression with subframe references replaced by joined column names
            
        Notes
        -----
        - Uses LEFT JOIN to preserve all main frame rows
        - Missing keys in subframe produce NaN (never drops rows)
        - Column naming convention: {column}__{subframe} (e.g., mX__T)
        - TTree::Draw compatible: expressions use dot notation (T.mX)
        """
        import warnings
        
        tokens = re.findall(r'(\b\w+)\.(\w+)', expr)
        for sf_name, sf_col in tokens:
            entry = self._subframes.get_entry(sf_name)
            if not entry:
                continue
            sub_adf = entry['frame']
            sub_df = sub_adf.df
            index_cols = entry['index']
            if isinstance(index_cols, str):
                index_cols = [index_cols]
            merge_cols = index_cols + [sf_col]
            suffix = f'__{sf_name}'
            col_renamed = f'{sf_col}{suffix}'
            
            # Skip if column already exists (idempotent behavior)
            if col_renamed in self.df.columns:
                expr = expr.replace(f'{sf_name}.{sf_col}', col_renamed)
                continue

            try:
                cols_to_merge = sub_df[merge_cols].copy()
            except KeyError:
                if sf_col in sub_adf.aliases:
                    sub_adf.materialize_alias(sf_col)
                    sub_df = sub_adf.df
                    cols_to_merge = sub_df[merge_cols].copy()
                else:
                    raise KeyError(f"Subframe '{sf_name}' does not contain or define alias '{sf_col}'")

            # Handle duplicate keys in subframe by taking first match
            # This prevents the merge from creating more rows than the main frame
            if cols_to_merge.duplicated(subset=index_cols).any():
                cols_to_merge = cols_to_merge.drop_duplicates(subset=index_cols, keep='first')
            # Phase 3A: Use LEFT JOIN to preserve all main frame rows
            # Missing keys in subframe will produce NaN (rows are never dropped)
            n_before = len(self.df)
            
            # Preserve original index for proper alignment
            original_index = self.df.index.copy()
            
            # Add a temporary column to track original row order
            self.df['__row_order__'] = np.arange(len(self.df))
            
            joined = self.df.merge(
                cols_to_merge, 
                on=index_cols, 
                suffixes=('', suffix),
                how='left'  # Critical: preserve all main frame rows
            )
            
            # Sort by original row order to restore alignment
            joined = joined.sort_values('__row_order__').reset_index(drop=True)
            
            # Remove temporary column
            self.df.drop(columns=['__row_order__'], inplace=True)
            
            # Find the actual column name in joined DataFrame
            # If subframe column name collides with main frame column, it gets suffix
            # Priority: check for suffixed version first (collision case), then unsuffixed
            if col_renamed in joined.columns:
                actual_col = col_renamed
            elif sf_col in joined.columns and sf_col not in self.df.columns:
                # Column exists in joined but not in main frame - it's from subframe
                actual_col = sf_col
            elif f'{sf_col}{suffix}' in joined.columns:
                # Column got suffixed due to collision
                actual_col = f'{sf_col}{suffix}'
            else:
                # Fallback: column might have been added without suffix
                actual_col = sf_col if sf_col in joined.columns else None
            
            if actual_col and actual_col in joined.columns:
                # Count missing keys (NaN values introduced by left join)
                n_missing = int(joined[actual_col].isna().sum())
                
                # Emit warning if there are missing keys
                if warn_missing_keys and n_missing > 0:
                    alias_info = f"Alias '{alias_name}': " if alias_name else ""
                    warnings.warn(
                        f"{alias_info}{n_missing:,} of {n_before:,} keys in main frame "
                        f"not found in subframe '{sf_name}'. Filled with NaN.",
                        UserWarning
                    )
                
                # Assign aligned values back to DataFrame with standardized name
                self.df[col_renamed] = joined[actual_col].values
                expr = expr.replace(f'{sf_name}.{sf_col}', col_renamed)
                
        return expr

    def _check_for_cycles(self):
        try:
            self._topological_sort()
        except ValueError as e:
            raise ValueError("Cycle detected in alias dependencies") from e

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

    def add_alias(self, name, expression, dtype=None, is_constant=False):
        """
        Define a new alias (lazy computed column).
        
        Args:
            name: Name of the alias.
            expression: Expression string using pandas or NumPy operations.
            dtype: Optional numpy dtype to enforce.
            is_constant: Whether the alias represents a scalar constant.
            
        Phase 4: Writes to _schema["columns"] as single source of truth.
        """
        # Build spec for schema
        spec = {"expr": expression}
        if dtype is not None:
            spec["dtype"] = dtype
        if is_constant:
            spec["constant"] = True
        
        # Write to schema
        self._schema["columns"][name] = spec
        
        # Check for cycles
        self._check_for_cycles()

    def _eval_in_namespace(self, expr, warn_missing_keys=True, alias_name=None):
        expr = self._prepare_subframe_joins(expr, warn_missing_keys=warn_missing_keys, alias_name=alias_name)
        local_env = {col: self.df[col] for col in self.df.columns}
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
                        if token in self.aliases:
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
            except nx.NetworkXError:
                result = list(expanded)
        
        return result

    def describe_aliases(self, verbosity=0x05, pattern=None, names=None, as_dict=False,
                         only_broken=False, only_materialized=False, 
                         only_unmaterialized=False, with_dependencies=False, color=False):
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
            
            # Truncate long expressions
            expr = info['expr']
            expr_display = expr if len(expr) <= 45 else expr[:42] + "..."
            
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

    def materialize_alias(self, name, cleanTemporary=False, dtype=None, warn_missing_keys=True):
        """
        Evaluate an alias and store its result as a real column.
        
        Args:
            name: Alias name to materialize.
            cleanTemporary: Whether to clean up intermediate dependencies.
            dtype: Optional override dtype to cast to.
            warn_missing_keys: If True, emit warning when subframe join has missing keys.
                             Missing keys produce NaN (rows are never dropped).

        Raises:
            KeyError: If alias is not defined.
            Exception: If alias evaluation fails.
        """
        if name not in self.aliases:
            print(f"[materialize_alias] Warning: alias '{name}' not found.")
            return
        expr = self.aliases[name]

        # Automatically materialize any referenced aliases or subframe aliases
        tokens = re.findall(r'\b\w+\b|\w+\.\w+', expr)
        for token in tokens:
            if '.' in token:
                sf_name, sf_attr = token.split('.', 1)
                sf = self.get_subframe(sf_name)
                if sf and sf_attr in sf.aliases and sf_attr not in sf.df.columns:
                    sf.materialize_alias(sf_attr)
            elif token == name:
                # Skip self-reference to prevent infinite recursion
                # (alias 'x' referencing 'subframe.x' where 'x' is extracted as a token)
                continue
            elif token in self.aliases and token not in self.df.columns:
                self.materialize_alias(token, warn_missing_keys=warn_missing_keys)

        result = self._eval_in_namespace(expr, warn_missing_keys=warn_missing_keys, alias_name=name)
        result_dtype = dtype or self.alias_dtypes.get(name)
        if result_dtype is not None:
            try:
                result = result.astype(result_dtype)
            except AttributeError:
                result = result_dtype(result)
        self.df[name] = result

    def materialize_aliases(self, pattern=None, names=None, with_dependencies=True,
                            only_unmaterialized=True, cleanTemporary=True, verbose=False):
        """
        Materialize aliases matching pattern and/or names.
        
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
            
        Returns
        -------
        list
            Names of aliases that were materialized
            
        Examples
        --------
        >>> adf.materialize_aliases(pattern=r'is.*')  # All 'is*' aliases
        >>> adf.materialize_aliases(names=['r', 'phi', 'cosPhi'])  # Specific names
        >>> adf.materialize_aliases(pattern=r'dy.*|dz.*')  # dy and dz aliases
        """
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
        
        # Materialize in order
        added = []
        for name in to_materialize:
            if name not in self.df.columns:
                if verbose:
                    print(f"[materialize_aliases] Materializing: {name}")
                self.materialize_alias(name, cleanTemporary=False)
                added.append(name)
        
        # Clean temporary dependencies if requested
        if cleanTemporary and with_dependencies:
            targets_set = set(targets)
            for col in added:
                if col not in targets_set and col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
                    if verbose:
                        print(f"[materialize_aliases] Cleaned temporary: {col}")
        
        return added

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

    def export_tree(self, filename_or_file, treename="tree", dropAliasColumns=True,compression=uproot.ZLIB(level=1)):
        """
        uproot.LZMA(level=5)
        :param filename_or_file:
        :param treename:
        :param dropAliasColumns:
        :param compression:
        :return:
        """
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
    def read_tree(filename, treename="tree", entry_start=None, entry_stop=None, num_workers=8):
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
        # Warn if entry_range used with subframes
        if metadata['subframes'] and (entry_start is not None or entry_stop is not None):
            warnings.warn(
                f"entry_start/entry_stop apply only to main tree '{treename}'. "
                f"Subframes {metadata['subframes']} will be fully loaded."
            )

        for sf_name in metadata['subframes']:
            try:
                sf = AliasDataFrame.read_tree(
                    filename,
                    treename=f"{treename}__subframe__{sf_name}",
                    num_workers=num_workers
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

        Examples
        --------
        >>> adf.get_compression_state('dy')
        'compressed'
        """
        if column not in self.compression_info or column == "__meta__":
            return None
        return self.compression_info[column].get('state')

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
                raise ValueError(
                    f"Column '{orig_col}' is already compressed. "
                    f"Use decompress_columns(['{orig_col}']) first to decompress before recompressing."
                )
            elif current_state == CompressionState.SCHEMA_ONLY:
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

            # Step 4: Handle compressed column
            if not keep_compressed:
                self.df.drop(columns=[compressed_col], inplace=True)

            # Step 5: Update state
            if keep_schema:
                # Transition to DECOMPRESSED state
                self.compression_info[col]['state'] = CompressionState.DECOMPRESSED
            else:
                # Remove all compression metadata
                del self.compression_info[col]

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
            entry = {
                'name': name,
                'state': info.get('state', 'unknown'),
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
        Export schema as JSON-safe dictionary.
        
        Converts numpy dtypes to strings and removes non-serializable objects.
        Includes physical column dtypes from DataFrame.
        
        Returns
        -------
        dict
            JSON-safe schema dictionary
        """
        schema_copy = copy.deepcopy(self.schema)
        
        # Add physical column dtypes from DataFrame
        for col in self.df.columns:
            if col not in schema_copy['columns']:
                schema_copy['columns'][col] = {}
            # Store dtype
            schema_copy['columns'][col]['dtype'] = str(self.df[col].dtype)
            # Mark as physical (no expr)
            if 'expr' not in schema_copy['columns'][col]:
                schema_copy['columns'][col]['expr'] = None
        
        # Convert dtypes to strings in columns
        for name, info in schema_copy.get('columns', {}).items():
            if 'dtype' in info and info['dtype'] is not None:
                try:
                    # Convert numpy dtype to string
                    dtype = info['dtype']
                    if not isinstance(dtype, str):
                        info['dtype'] = np.dtype(dtype).name
                except (TypeError, ValueError):
                    info['dtype'] = str(info['dtype'])
        
        # Convert dtypes in compression
        for name, info in schema_copy.get('compression', {}).items():
            if name == '__meta__':
                continue
            for dtype_field in ['compressed_dtype', 'decompressed_dtype']:
                if dtype_field in info and info[dtype_field] is not None:
                    try:
                        dtype = info[dtype_field]
                        if not isinstance(dtype, str):
                            info[dtype_field] = np.dtype(dtype).name
                    except (TypeError, ValueError):
                        info[dtype_field] = str(info[dtype_field])
            
            # Remove monitor functions (not serializable)
            if 'monitor' in info and info['monitor']:
                monitor = info['monitor']
                if 'func' in monitor:
                    # Keep structure but remove function
                    info['monitor'] = {k: v for k, v in monitor.items() if k != 'func'}
        
        return schema_copy

    def save_schema(self, path):
        """
        Save schema to JSON file.
        
        Parameters
        ----------
        path : str
            Path to save schema JSON file
        """
        schema = self.export_schema()
        with open(path, 'w') as f:
            json.dump(schema, f, indent=2)

    @staticmethod
    def load_schema(path):
        """
        Load schema from JSON file.
        
        Parameters
        ----------
        path : str
            Path to schema JSON file
            
        Returns
        -------
        dict
            Schema dictionary
        """
        with open(path, 'r') as f:
            return json.load(f)

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
            dtype_str = dtype_obj.__name__ if dtype_obj else 'unspecified'
            
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
                'index_columns': index_cols
            })
        
        info['subframes'] = subframes_info
        
        if verbosity & VERBOSITY_SUBFRAMES:
            if subframes_info:
                lines.append(f"Subframes: {len(subframes_info)}")
                for sf in subframes_info:
                    index_str = sf['index_columns'] if isinstance(sf['index_columns'], str) else ', '.join(sf['index_columns'])
                    lines.append(f"  - {sf['name']}: {sf['rows']:,} rows × {sf['columns']} cols, index={index_str}")
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
    
    def auto_alias_subframe(self, subframe_name, validate=False, reset_before=False):
        """
        Explicitly create aliases for all columns in a subframe.
        
        Creates aliases: column_name -> subframe_name.column_name
        Tracks created aliases in self._auto_aliases
        
        Args:
            subframe_name: Name of subframe
            validate: If True, validate against materialized columns (slow!)
            reset_before: If True, remove old auto-aliases for this subframe first
        
        Returns:
            dict: {column_name: expression} for created aliases
        
        Example:
            # First time
            aliases = adf.auto_alias_subframe('DITS0FitSide', validate=True)
            
            # Regenerate later
            aliases = adf.auto_alias_subframe('DITS0FitSide', reset_before=True)
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
        
        # Reset: remove existing auto-aliases for this subframe
        if reset_before:
            old_aliases = [k for k, v in self._auto_aliases.items() 
                          if v == subframe_name]
            if old_aliases:
                print(f"  Removing {len(old_aliases)} existing auto-aliases for '{subframe_name}'")
                self.remove_aliases(old_aliases, strict=False)
        
        # Create aliases
        aliases_created = {}
        materialized_found = []
        
        for col in sf_df.columns:
            if col in index_cols:
                continue
            
            alias_expr = f"{subframe_name}.{col}"
            
            # Check if materialized
            if col in self.df.columns:
                materialized_found.append(col)
                
                if validate:
                    print(f"    Validating '{col}'...", end=' ')
                    self.add_alias(f'_temp_validate_{col}', alias_expr)
                    self.materialize_alias(f'_temp_validate_{col}', warn_missing_keys=False)
                    
                    materialized = self.df[col].values
                    alias_result = self.df[f'_temp_validate_{col}'].values
                    
                    match = np.allclose(materialized, alias_result, equal_nan=True, rtol=1e-6)
                    
                    if match:
                        print("✓")
                    else:
                        print("✗ MISMATCH")
                        warnings.warn(
                            f"Alias '{col}' -> '{alias_expr}' does NOT match "
                            f"materialized column '{col}'. Check subframe index!"
                        )
                    
                    self.df.drop(columns=[f'_temp_validate_{col}'], inplace=True)
            
            # Add alias and track it
            self.add_alias(col, alias_expr)
            self._auto_aliases[col] = subframe_name  # Track as auto-created
            aliases_created[col] = alias_expr
        
        # Report
        print(f"\n  ✓ Created {len(aliases_created)} auto-aliases for '{subframe_name}'")
        if materialized_found:
            print(f"    {len(materialized_found)} columns have materialized versions")
            print(f"    Can drop: {materialized_found[:3]}" + 
                  (f" ... (+{len(materialized_found)-3} more)" if len(materialized_found) > 3 else ""))
        
        return aliases_created
    
    def auto_alias_all_subframes(self, validate=False, reset_before=False):
        """
        Explicitly create aliases for all loaded subframes.
        
        WARNING: If multiple subframes have same column name, last one wins.
        For production, prefer auto_alias_subframe() for specific subframes.
        
        Args:
            validate: If True, validate aliases (slow!)
            reset_before: If True, remove old auto-aliases first
        
        Returns:
            dict: {subframe_name: {column: expression}}
        
        Example:
            all_aliases = adf.auto_alias_all_subframes(validate=False)
        """
        all_created = {}
        
        subframes = self.list_subframes()
        print(f"\nAuto-aliasing {len(subframes)} subframes...")
        
        for sf_name in subframes:
            print(f"\nSubframe '{sf_name}':")
            aliases = self.auto_alias_subframe(sf_name, validate=validate, reset_before=reset_before)
            all_created[sf_name] = aliases
        
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

        return [k for k, v in self._auto_aliases.items() if v == subframe_name]
