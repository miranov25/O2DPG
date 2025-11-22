import sys
import os; sys.path.insert(1, os.environ.get("O2DPG", "") + "/UTILS/dfextensions")
import pandas as pd
import numpy as np
import json
import uproot
import copy
import warnings
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
    
    Parameters
    ----------
    schema : dict
        The _schema dict with columns/compression/subframes
        
    Returns
    -------
    dict
        JSON-serializable version of schema
    """
    result = {
        "schema_version": SCHEMA_VERSION,
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
    
    Parameters
    ----------
    serialized : dict
        JSON-parsed schema dict
        
    Returns
    -------
    dict
        Restored _schema dict with proper types
    """
    version = serialized.get("schema_version", 1)
    
    # Future: Add migration logic here
    if version > SCHEMA_VERSION:
        warnings.warn(
            f"Schema version {version} is newer than supported version {SCHEMA_VERSION}. "
            f"Some features may not work correctly."
        )
    
    result = {
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
    
    def __init__(self, df):
        """
        Initialize AliasDataFrame with unified schema structure.
        
        The _schema dict is the single source of truth for:
        - columns: physical column dtypes and aliases (expr + dtype + constant)
        - compression: compression formulas per column
        - subframes: registered subframes with index info
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"AliasDataFrame must be initialized with a pandas.DataFrame. "
                f"Received type: {type(df)}"
            )
        self.df = df
        
        # Unified schema (Phase 4)
        self._schema = {
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
                
                # Check expr is string
                if "expr" in spec and not isinstance(spec["expr"], str):
                    raise ValueError(f"Expression for '{name}' must be a string")
                
                # Check subframe references exist (lightweight check)
                if "expr" in spec and "." in spec["expr"]:
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
        from collections import defaultdict
        dependencies = defaultdict(set)
        for name, expr in self.aliases.items():
            tokens = re.findall(r'\b\w+\b', expr)
            for token in tokens:
                if token in self.aliases:
                    dependencies[name].add(token)
        return dependencies

    def _check_for_cycles(self):
        graph = nx.DiGraph()
        for name, deps in self._resolve_dependencies().items():
            for dep in deps:
                graph.add_edge(dep, name)
        try:
            list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Cycle detected in alias dependencies")

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
        self._check_for_cycles()
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
        broken = []
        for name, expr in self.aliases.items():
            try:
                # Suppress warnings during validation - we're just checking syntax
                self._eval_in_namespace(expr, warn_missing_keys=False, alias_name=name)
            except Exception:
                broken.append(name)
        return broken

    def describe_aliases(self):
        print("Aliases:")
        for name, expr in self.aliases.items():
            print(f"  {name}: {expr}")
        broken = self.validate_aliases()
        if broken:
            print("\nBroken Aliases:")
            for name in broken:
                print(f"  {name}")
        print("\nDependencies:")
        deps = self._resolve_dependencies()
        for k, v in deps.items():
            print(f"  {k}: {sorted(v)}")

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

    def materialize_aliases(self, targets, cleanTemporary=True, verbose=False):
        import networkx as nx
        def build_graph():
            g = nx.DiGraph()
            for alias, expr in self.aliases.items():
                for token in re.findall(r'\b\w+\b', expr):
                    if token in self.aliases:
                        g.add_edge(token, alias)
            return g
        g = build_graph()
        required = set()
        for t in targets:
            if t not in self.aliases:
                if verbose:
                    print(f"[materialize_aliases] Skipping non-alias target: {t}")
                continue
            if t not in g:
                if verbose:
                    print(f"[materialize_aliases] Alias '{t}' not in graph")
                continue
            try:
                required |= nx.ancestors(g, t)
            except nx.NetworkXError:
                continue
            required.add(t)
        ordered = list(nx.topological_sort(g.subgraph(required)))
        added = []
        for name in ordered:
            if name not in self.df.columns:
                self.materialize_alias(name)
                added.append(name)
        if cleanTemporary:
            for col in added:
                if col not in targets and col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
        return added

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
        # ========================================================================
        # Compression Support
        # ========================================================================

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

    def describe_compression(self):
        """
        Print human-readable compression summary.

        Shows compressed columns, expressions, dtypes, state, and precision metrics
        if available.

        Examples
        --------
        >>> adf.describe_compression()
        Compressed Columns:
        -------------------
        dy:
          State: compressed
          Compressed as: dy_c (int16)
          Expression: round(asinh(dy)*40)
          Decompression: sinh(dy_c/40.) → float16
          Precision: RMSE=0.0012, Max=0.0045
        """
        # Filter out __meta__
        columns_info = {k: v for k, v in self.compression_info.items() if k != "__meta__"}

        if not columns_info:
            print("No compressed columns")
            return

        print("Compression Metadata:")
        print("-" * 70)
        for col, info in columns_info.items():
            print(f"\n{col}:")
            print(f"  State: {info.get('state', 'unknown')}")
            print(f"  Compressed as: {info['compressed_col']} ({info['compressed_dtype']})")
            print(f"  Expression: {info['compress_expr']}")
            print(f"  Decompression: {info['decompress_expr']} → {info['decompressed_dtype']}")
            print(f"  Original removed: {info.get('original_removed', False)}")

            if 'precision' in info:
                prec = info['precision']
                if 'error' in prec:
                    print(f"  Precision: measurement failed ({prec['error']})")
                else:
                    print(f"  Precision: RMSE={prec['rmse']:.6f}, "
                          f"Max={prec['max_error']:.6f}, "
                          f"Mean={prec['mean_error']:.6f}")
                    # Add sample count info
                    n_samples = prec.get('n_samples', 0)
                    n_total = prec.get('n_total', n_samples)
                    frac_nonfinite = prec.get('fraction_nonfinite', 0.0)
                    #if frac_nonfinite >= 0:
                    print(f"  Samples: {n_samples:,}/{n_total:,}, "f"Non-finite: {frac_nonfinite*100:.2f}%")

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
