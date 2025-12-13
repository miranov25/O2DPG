"""
DSLCompiler - High-level interface for RDataFrame DSL.

This module provides a simple, user-friendly interface for defining
computed columns using Python DSL syntax and applying them to ROOT
RDataFrame objects.

Usage:
    from RDataFrameDSL import DSLCompiler
    
    # Define schema (column names and C++ types)
    schema = {
        "px": "double",
        "py": "double",
        "pt": "RVec<double>",
    }
    
    # Create compiler and define columns
    dsl = DSLCompiler(schema)
    dsl.define("event_pt", "sqrt(px**2 + py**2)")
    dsl.define("n_tracks", "pt.size()")
    dsl.define("first3", "pt[:3]")
    
    # Aliases can now reference other aliases!
    dsl.define("high_pt", "event_pt > 10.0")  # Uses event_pt alias
    
    # Apply to RDataFrame
    rdf = ROOT.RDataFrame("Events", "data.root")
    rdf = dsl.apply(rdf)
    
    # Export for inspection
    dsl.export_macro("my_functions.C")

Phase 7.9: RDataFrame Validation & C++ Export
Phase 8.1: Alias referencing support (aliases can use other aliases)
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import uuid
import re
import warnings

from .type_inferrer import TypeInferrer
from .ir_builder import IRBuilder
from .ir_types import IRType, IRTypeKind
from .backend_cpp import CppCodeGenerator, FunctionLibrary, GeneratedFunction
from .ir_errors import IRError, IRErrorKind


__all__ = ['DSLCompiler']


def _simple_schema_to_full(simple_schema: Dict[str, str]) -> Dict:
    """
    Convert simple schema to full TypeInferrer format.
    
    Simple: {"px": "double", "pt": "RVec<double>"}
    Full: {"columns": {"px": {"dtype": "double", "rank": 0}, ...}}
    
    Args:
        simple_schema: Dict mapping column names to C++ type strings
        
    Returns:
        Full schema dict for TypeInferrer.from_schema()
    """
    columns = {}
    for name, type_str in simple_schema.items():
        if type_str.startswith("RVec<") and type_str.endswith(">"):
            inner = type_str[5:-1]
            columns[name] = {"dtype": inner, "rank": 1, "cpp_type": type_str}
        elif type_str.startswith("ROOT::RVec<") and type_str.endswith(">"):
            inner = type_str[11:-1]
            columns[name] = {"dtype": inner, "rank": 1, "cpp_type": type_str}
        elif type_str.startswith("std::vector<") and type_str.endswith(">"):
            inner = type_str[12:-1]
            columns[name] = {"dtype": inner, "rank": 1, "cpp_type": type_str}
        else:
            columns[name] = {"dtype": type_str, "rank": 0}
    return {"columns": columns}


def _ir_to_type_string(ir) -> str:
    """
    Convert IR node's type to a simple type string for schema.
    
    Args:
        ir: IR node with dtype and rank attributes
        
    Returns:
        Type string (e.g., "double", "RVec<double>", "RVec<TLorentzVector>")
    """
    # Get base type
    if ir.dtype.cpp_type:
        base_type = ir.dtype.cpp_type
    else:
        base_type = ir.dtype.to_cpp()
    
    # Handle RVec wrapping
    if ir.rank == 1:
        # Check if already wrapped
        if base_type.startswith("RVec<") or base_type.startswith("ROOT::RVec<"):
            return base_type
        else:
            return f"RVec<{base_type}>"
    elif ir.rank == 0:
        return base_type
    else:
        # rank > 1 not fully supported yet
        return base_type


class DSLCompiler:
    """
    High-level DSL compiler for RDataFrame.
    
    This class provides a simple interface for:
    - Defining computed columns using Python DSL syntax
    - Applying all definitions to an RDataFrame
    - Exporting generated C++ for inspection
    
    Aliases can reference other previously defined aliases!
    
    Attributes:
        schema: Column definitions (name -> C++ type)
        library: FunctionLibrary containing generated functions
    
    Example:
        >>> dsl = DSLCompiler({"px": "double", "py": "double"})
        >>> dsl.define("pt", "sqrt(px**2 + py**2)")
        >>> dsl.define("high_pt", "pt > 10.0")  # Can use 'pt' alias!
        >>> rdf = dsl.apply(rdf)
    """
    
    def __init__(self, schema: Dict[str, str], safe_indexing: bool = True):
        """
        Initialize DSL compiler.
        
        Args:
            schema: Dict mapping column names to C++ types
                    e.g. {"px": "double", "pt": "RVec<double>"}
            safe_indexing: Enable bounds checking (default True)
        """
        self.schema = dict(schema)  # Make a copy to allow modifications
        self.safe_indexing = safe_indexing
        
        # Unique ID for this compiler instance (avoids parallel test collisions)
        self._unique_id = uuid.uuid4().hex[:8]
        
        # Convert schema
        full_schema = _simple_schema_to_full(schema)
        self._inferrer = TypeInferrer.from_schema(full_schema)
        
        # Set up generator
        self._generator = CppCodeGenerator(
            type_inferrer=self._inferrer,
            safe_indexing=safe_indexing
        )
        
        # Storage
        self._definitions: List[tuple] = []  # [(name, expr), ...]
        self._functions: Dict[str, GeneratedFunction] = {}
        self.library = FunctionLibrary()
        
        # Phase 12.2: Track if JIT helpers have been declared
        self._helpers_declared = False
    
    def _preprocess_expression(self, expr: str) -> str:
        """
        Convert C++ :: notation to Python dot notation.
        
        Phase 11.1: Handles namespace syntax preprocessing before AST parsing.
        
        Handles:
        - TMath::Pi() -> TMath.Pi()
        - ROOT::Math::VectorUtil::DeltaPhi() -> ROOT.Math.VectorUtil.DeltaPhi()
        
        Skips:
        - String literals to avoid corrupting "Error::Message"
        - Square brackets to preserve slice syntax like pt[::-1]
        
        Args:
            expr: DSL expression potentially with C++ :: notation
            
        Returns:
            Expression with :: replaced by . (except in strings and brackets)
        """
        result = []
        i = 0
        in_string = False
        string_char = None
        bracket_depth = 0  # Track [] nesting for slice syntax
        
        while i < len(expr):
            char = expr[i]
            
            # Track string literals
            if char in ('"', "'") and (i == 0 or expr[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            # Track square brackets (for slice syntax like [::-1])
            if not in_string:
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
            
            # Replace :: with . only outside strings AND outside brackets
            if not in_string and bracket_depth == 0 and expr[i:i+2] == '::':
                result.append('.')
                i += 2
                continue
            
            result.append(char)
            i += 1
        
        return ''.join(result)
    
    def define(self, name: str, expression: str, dtype: str = None) -> 'DSLCompiler':
        """
        Define a new column from a DSL expression.
        
        Expressions can reference:
        - Schema columns (original data)
        - Previously defined aliases
        
        Args:
            name: Output column name
            expression: DSL expression (e.g. "sqrt(px**2 + py**2)")
            dtype: Optional explicit return type. Supported values:
                   - "bool"
                   - "int", "int8", "int16", "int32", "int64"
                   - "uint8", "uint16", "uint32", "uint64"
                   - "float", "float32", "float64", "double"
                   If None, inferred from expression or defaults to double.
        
        Returns:
            self (for chaining)
        
        Raises:
            IRError: If expression is invalid or name conflicts
        
        Example:
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
            >>> dsl.define("high_pt", "pt > 10.0")  # Uses 'pt' alias
            >>> dsl.define("isOK", "(row < 152) & (abs(dy) < 10)", dtype="bool")
            >>> dsl.define("sector", "int(9*phi/pi)", dtype="int8")
        """
        # Check for name collision with original schema only
        # (aliases are allowed to shadow other aliases via redefinition)
        if name in self.schema and name not in [n for n, _ in self._definitions]:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Column name '{name}' conflicts with existing branch",
                suggestions=[f"Use a different name like '{name}_calc'"]
            )
        
        # Check for duplicate definition
        existing_names = [n for n, _ in self._definitions]
        if name in existing_names:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Column '{name}' already defined",
                suggestions=["Each column name must be unique"]
            )
        
        # Phase 11.1: Preprocess C++ :: syntax to Python dot syntax
        preprocessed = self._preprocess_expression(expression)
        
        # Parse and generate
        builder = IRBuilder(self._inferrer)
        ir = builder.build(preprocessed)
        
        # Phase 11.1c: Override return type if explicitly specified
        if dtype is not None:
            ir.dtype = self._parse_dtype(dtype)
        
        # Use unique suffix to avoid collisions in parallel execution
        unique_name = f"{name}_{self._unique_id}"
        func = self._generator.generate(ir, unique_name)
        func.dsl_expression = expression  # Track original DSL
        func.column_name = name  # Track user-friendly name for Define()
        
        # Store
        self._definitions.append((name, expression))
        self._functions[name] = func
        self.library.add(func)
        
        # === NEW: Register alias in schema for future expressions ===
        self._register_alias_type(name, ir, dtype)
        
        return self
    
    def _parse_dtype(self, dtype: str) -> IRType:
        """
        Parse dtype string to IRType.
        
        Phase 11.1c: Supports explicit type specification via dtype parameter.
        
        Args:
            dtype: Type string like "bool", "int8", "float32"
            
        Returns:
            IRType instance
            
        Raises:
            IRError: If dtype is not recognized
        """
        dtype_map = {
            # Boolean
            "bool": IRType(IRTypeKind.Bool),
            
            # Signed integers
            "int": IRType(IRTypeKind.Int32),
            "int8": IRType(IRTypeKind.Int8),
            "int16": IRType(IRTypeKind.Int16),
            "int32": IRType(IRTypeKind.Int32),
            "int64": IRType(IRTypeKind.Int64),
            
            # Unsigned integers
            "uint8": IRType(IRTypeKind.UInt8),
            "uint16": IRType(IRTypeKind.UInt16),
            "uint32": IRType(IRTypeKind.UInt32),
            "uint64": IRType(IRTypeKind.UInt64),
            
            # Floating point
            "float": IRType(IRTypeKind.Float32),
            "float32": IRType(IRTypeKind.Float32),
            "float64": IRType(IRTypeKind.Float64),
            "double": IRType(IRTypeKind.Float64),
        }
        
        if dtype not in dtype_map:
            raise IRError(
                IRErrorKind.TYPE_ERROR,
                f"Unknown dtype '{dtype}'",
                suggestions=[
                    "Supported types:",
                    "  bool",
                    "  int, int8, int16, int32, int64",
                    "  uint8, uint16, uint32, uint64",
                    "  float, float32, float64, double",
                ]
            )
        
        return dtype_map[dtype]
    
    def _dtype_to_string(self, ir_type: IRType) -> str:
        """
        Convert IRType to dtype string for schema.
        
        Phase 11.1c: Maps IRType back to dtype string.
        
        Args:
            ir_type: IRType instance
            
        Returns:
            Dtype string like "bool", "int32", "double"
        """
        type_strings = {
            IRTypeKind.Bool: "bool",
            IRTypeKind.Int8: "int8",
            IRTypeKind.Int16: "int16",
            IRTypeKind.Int32: "int32",
            IRTypeKind.Int64: "int64",
            IRTypeKind.UInt8: "uint8",
            IRTypeKind.UInt16: "uint16",
            IRTypeKind.UInt32: "uint32",
            IRTypeKind.UInt64: "uint64",
            IRTypeKind.Float32: "float32",
            IRTypeKind.Float64: "double",
        }
        return type_strings.get(ir_type.kind, "double")
    
    def _register_alias_type(self, name: str, ir, dtype: str = None) -> None:
        """
        Register a new alias in the schema and rebuild type inferrer.
        
        This allows subsequent define() calls to reference this alias.
        
        Args:
            name: Alias name
            ir: IR node with type information
            dtype: Optional explicit dtype (overrides inference)
        """
        # Phase 11.1c: Use explicit dtype if provided
        if dtype is not None:
            type_str = self._dtype_to_cpp_type(dtype)
        else:
            type_str = _ir_to_type_string(ir)
        
        # Add to simple schema
        self.schema[name] = type_str
        
        # Rebuild TypeInferrer with updated schema
        full_schema = _simple_schema_to_full(self.schema)
        self._inferrer = TypeInferrer.from_schema(full_schema)
        
        # Update generator with new inferrer
        self._generator = CppCodeGenerator(
            type_inferrer=self._inferrer,
            safe_indexing=self.safe_indexing
        )
    
    def _dtype_to_cpp_type(self, dtype: str) -> str:
        """
        Convert dtype string to C++ type string for schema.
        
        Phase 11.1c: Maps dtype to C++ type for schema registration.
        
        Args:
            dtype: Type string like "bool", "int8", "float32"
            
        Returns:
            C++ type string like "bool", "int8_t", "float"
        """
        cpp_types = {
            # Boolean
            "bool": "bool",
            
            # Signed integers
            "int": "int",
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int",
            "int64": "long long",
            
            # Unsigned integers
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "unsigned int",
            "uint64": "unsigned long long",
            
            # Floating point
            "float": "float",
            "float32": "float",
            "float64": "double",
            "double": "double",
        }
        return cpp_types.get(dtype, "double")
    
    def _ensure_helpers_declared(self) -> None:
        """
        Declare custom JIT helpers (once per session).
        
        Phase 12.2: Declares IndicesFromOffsets helper function for 
        Track→Cluster selection patterns.
        
        This is idempotent - safe to call multiple times.
        """
        if self._helpers_declared:
            return
        
        try:
            import ROOT
            from .constants import INDICES_FROM_OFFSETS_CODE
            
            # Declare IndicesFromOffsets helper
            ROOT.gInterpreter.Declare(INDICES_FROM_OFFSETS_CODE)
            self._helpers_declared = True
            
        except ImportError:
            # ROOT not available - will error at compile time if helper is used
            pass
        except Exception:
            # If declaration fails (e.g., already defined), mark as done
            self._helpers_declared = True
    
    def _flatten_rvec_columns(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten RVec columns to 1D arrays for dfdraw compatibility.
        
        Phase 12.2: dfdraw cannot handle arrays of RVec objects (one RVec per row).
        This method detects such columns and concatenates them into flat arrays.
        
        Args:
            result: Dict from rdf.AsNumpy() with column name -> array
            
        Returns:
            Dict with RVec columns flattened to 1D numpy arrays
        """
        import numpy as np
        
        flattened = {}
        for col, data in result.items():
            if data.dtype == object and len(data) > 0:
                first = data[0]
                if hasattr(first, '__len__') and not isinstance(first, str):
                    try:
                        arrays = [np.asarray(x) for x in data if len(x) > 0]
                        flattened[col] = np.concatenate(arrays) if arrays else np.array([])
                    except (ValueError, TypeError):
                        flattened[col] = data
                else:
                    flattened[col] = data
            else:
                flattened[col] = data
        
        return flattened
    
    def _flatten_rvec_with_validation(self, result: Dict[str, Any], 
                                       paired_columns: List[Tuple[str, str]] = None
                                       ) -> Dict[str, Any]:
        """
        Flatten RVec columns with paired column length validation.
        
        Phase 12.3: Enhanced flattening that validates paired RVec columns
        (used in y:x plots) have matching lengths per event.
        
        Args:
            result: Dict from rdf.AsNumpy() with column name -> array
            paired_columns: List of (col1, col2) tuples that must have matching lengths
            
        Returns:
            Dict with RVec columns flattened to 1D numpy arrays
            
        Raises:
            ValueError: If paired columns have mismatched lengths in any event
        """
        import numpy as np
        
        # First, identify which columns are RVec (object dtype with array-like elements)
        rvec_columns = set()
        for col, data in result.items():
            if data.dtype == object and len(data) > 0:
                first = data[0]
                if hasattr(first, '__len__') and not isinstance(first, str):
                    rvec_columns.add(col)
        
        # Validate paired columns have matching lengths per event
        if paired_columns:
            for col1, col2 in paired_columns:
                if col1 in rvec_columns and col2 in rvec_columns:
                    data1, data2 = result[col1], result[col2]
                    for i, (arr1, arr2) in enumerate(zip(data1, data2)):
                        len1, len2 = len(arr1), len(arr2)
                        if len1 != len2:
                            raise ValueError(
                                f"Paired RVec columns '{col1}' and '{col2}' have different "
                                f"lengths in event {i}: {len1} vs {len2}. "
                                f"For 2D plots (y:x), both columns must have the same "
                                f"number of elements per event."
                            )
        
        # Now flatten
        return self._flatten_rvec_columns(result)
    
    def _extract_paired_columns(self, specs: List[dict]) -> List[Tuple[str, str]]:
        """
        Extract column pairs from plot expressions (y:x syntax).
        
        Phase 12.3: Identifies paired columns that need length validation.
        
        Args:
            specs: List of figure specifications
            
        Returns:
            List of (y_col, x_col) tuples
        """
        pairs = []
        for spec in specs:
            for plot in spec.get('plots', []):
                expr = plot.get('expr', '') if isinstance(plot, dict) else plot
                if ':' in expr:
                    parts = expr.replace(' ', '').split(':')
                    if len(parts) == 2:
                        y_col, x_col = parts
                        # Only add if both are simple column names (not expressions)
                        if y_col.isidentifier() and x_col.isidentifier():
                            pairs.append((y_col, x_col))
        return pairs
    
    def _validate_figure_specs(self, specs: List[dict]) -> None:
        """
        Validate figure specifications with helpful error messages.
        
        Phase 12.3: Basic validation of required fields.
        
        Args:
            specs: List of figure specifications
            
        Raises:
            ValueError: If specs are invalid
        """
        if not specs:
            return  # Empty specs is valid (returns empty dict)
        
        for i, spec in enumerate(specs):
            name = spec.get('name', f'figure_{i}')
            
            if 'plots' not in spec:
                raise ValueError(
                    f"Figure '{name}': 'plots' is required. "
                    f"Each figure spec must have a 'plots' list."
                )
            
            if not spec['plots']:
                raise ValueError(
                    f"Figure '{name}': 'plots' cannot be empty. "
                    f"Add at least one plot specification."
                )
            
            for j, plot in enumerate(spec['plots']):
                # Allow short form (string) or full form (dict)
                if isinstance(plot, str):
                    continue  # String is valid (will be converted to {'expr': plot})
                
                if not isinstance(plot, dict):
                    raise ValueError(
                        f"Figure '{name}', plot {j}: must be string or dict, "
                        f"got {type(plot).__name__}"
                    )
                
                if 'expr' not in plot:
                    raise ValueError(
                        f"Figure '{name}', plot {j}: 'expr' is required. "
                        f"Specify what to plot, e.g. {{'expr': 'pt'}} or {{'expr': 'dy:row'}}"
                    )
    
    def draw_figures(self, specs: List[dict], rdf, 
                     save_dir: str = None, defaults: dict = None,
                     max_entries: int = None, show: bool = False
                     ) -> Dict[str, Any]:
        """
        Draw multiple composed figures with automatic column detection.
        
        Phase 12.3: Creates multi-subplot figures from declarative specifications.
        Each figure can contain multiple plots arranged in a grid.
        
        Args:
            specs: List of figure specifications. Each spec is a dict with:
                - name: str (required) - Figure identifier
                - plots: list (required) - List of plot specs or expressions
                - ncols: int (default: 2) - Columns in grid
                - figsize: tuple - (width, height) in inches (auto if None)
                - sharex: bool (default: False) - Share x-axis
                - sharey: bool (default: False) - Share y-axis
                - suptitle: str - Figure title
                - savefig: str - Save path (relative to save_dir)
                - dpi: int (default: 150) - Resolution for saving
                
                Each plot spec can be:
                - str: Simple expression like 'pt' or 'dy:row'
                - dict: Full spec with 'expr' (required) plus optional:
                    - type: 'hist', 'scatter', 'profile', 'hist2d', 'hexbin'
                    - title: Subplot title
                    - selection: Filter expression
                    - bins: Number of bins
                    - Any other dfdraw parameters
                    
            rdf: Applied RDataFrame
            save_dir: Default directory for saving figures
            defaults: Default parameters applied to all plots
            max_entries: Limit entries for large datasets
            show: Call plt.show() after drawing
            
        Returns:
            Dict mapping figure names to {'fig': Figure, 'axes': list, 'stats': list}
            
        Example:
            >>> qa_report = [
            ...     {
            ...         'name': 'overview',
            ...         'suptitle': 'TPC Calibration QA',
            ...         'ncols': 2,
            ...         'savefig': 'qa/overview.png',
            ...         'plots': [
            ...             {'expr': 'chi2_norm', 'selection': 'isGoodTrack'},
            ...             {'expr': 'dy:row', 'type': 'profile'},
            ...             'nClusters',  # Short form
            ...             {'expr': 'pt', 'bins': 100},
            ...         ]
            ...     }
            ... ]
            >>> results = dsl.draw_figures(qa_report, rdf)
        """
        # Lazy imports
        try:
            from dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw is required for visualization.\n"
                "Install with: pip install dfdraw\n"
                "Or use to_dataframe() for manual plotting."
            )
        
        import matplotlib.pyplot as plt
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import warnings
        
        # Validate specs
        self._validate_figure_specs(specs)
        
        if not specs:
            return {}
        
        defaults = defaults or {}
        
        # Collect ALL columns from ALL specs (single AsNumpy call)
        all_columns = set()
        for spec in specs:
            for plot in spec.get('plots', []):
                # Handle short form (string)
                if isinstance(plot, str):
                    plot = {'expr': plot}
                
                all_columns.update(self._collect_draw_dependencies(
                    plot.get('expr', ''),
                    plot.get('selection', defaults.get('selection')),
                    plot.get('group_by', defaults.get('group_by')),
                    plot.get('color', defaults.get('color'))
                ))
        
        if not all_columns:
            warnings.warn("No columns detected in figure specs")
            return {}
        
        # Apply entry limit
        if max_entries is not None:
            rdf = rdf.Range(max_entries)
        
        # Single data extraction (efficient!)
        result = rdf.AsNumpy(list(all_columns))
        
        # Extract paired columns for validation
        paired_columns = self._extract_paired_columns(specs)
        
        # Flatten RVec columns with paired validation
        try:
            result = self._flatten_rvec_with_validation(result, paired_columns)
        except ValueError as e:
            raise ValueError(f"RVec validation error: {e}")
        
        # Check column lengths for DataFrame creation
        lengths = {col: len(arr) for col, arr in result.items()}
        unique_lengths = set(lengths.values())
        
        # Generate figures
        results = {}
        
        for spec in specs:
            name = spec.get('name', f'figure_{len(results)}')
            plots = spec['plots']
            ncols = spec.get('ncols', 2)
            nrows = (len(plots) + ncols - 1) // ncols
            
            # Figure size
            figsize = spec.get('figsize')
            if figsize is None:
                figsize = (5 * ncols, 4 * nrows)
            
            # Create figure
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=figsize,
                sharex=spec.get('sharex', False),
                sharey=spec.get('sharey', False),
                squeeze=False
            )
            axes_flat = axes.flatten()
            
            # Track stats for each subplot
            all_stats = []
            
            # Draw each subplot
            for i, plot_spec in enumerate(plots):
                if i >= len(axes_flat):
                    break
                
                ax = axes_flat[i]
                
                # Handle short form (string)
                if isinstance(plot_spec, str):
                    plot_spec = {'expr': plot_spec}
                
                # Merge defaults
                merged = {**defaults, **plot_spec}
                expr = merged.pop('expr')
                title = merged.pop('title', None)
                
                # Get columns for this plot
                plot_columns = self._collect_draw_dependencies(
                    expr,
                    merged.get('selection'),
                    merged.get('group_by'),
                    merged.get('color')
                )
                
                # Build DataFrame for this plot
                plot_data = {col: result[col] for col in plot_columns if col in result}
                
                # Check if columns have same length
                plot_lengths = [len(arr) for arr in plot_data.values()]
                if len(set(plot_lengths)) > 1:
                    warnings.warn(
                        f"Figure '{name}', plot {i} ('{expr}'): "
                        f"columns have different lengths, skipping"
                    )
                    ax.text(0.5, 0.5, f"Skipped:\n{expr}\n(length mismatch)", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    all_stats.append(None)
                    continue
                
                try:
                    drawer = DFDraw(pd.DataFrame(plot_data))
                    _, _, stats = drawer.draw(expr, ax=ax, **merged)
                    all_stats.append(stats)
                except Exception as e:
                    warnings.warn(f"Figure '{name}', plot {i} ('{expr}'): {e}")
                    ax.text(0.5, 0.5, f"Error:\n{expr}\n{str(e)[:50]}", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    all_stats.append(None)
                    continue
                
                if title:
                    ax.set_title(title)
            
            # Hide unused axes
            for i in range(len(plots), len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            # Suptitle
            suptitle = spec.get('suptitle')
            if suptitle:
                fig.suptitle(suptitle, fontsize=14)
            
            # Tight layout
            if spec.get('tight_layout', True):
                plt.tight_layout()
                if suptitle:
                    plt.subplots_adjust(top=0.93)
            
            # Save
            savefig = spec.get('savefig')
            if savefig:
                if save_dir and not savefig.startswith('/'):
                    savefig = f"{save_dir}/{savefig}"
                Path(savefig).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(savefig, dpi=spec.get('dpi', 150), bbox_inches='tight')
            
            results[name] = {
                'fig': fig,
                'axes': list(axes_flat[:len(plots)]),
                'stats': all_stats
            }
        
        if show:
            plt.show()
        
        return results
    
    def compile_all(self) -> None:
        """
        Compile all defined functions to ROOT.
        
        Raises:
            IRError: If any compilation fails
        """
        # Phase 12.2: Ensure JIT helpers are declared before compilation
        self._ensure_helpers_declared()
        
        for name, expr in self._definitions:
            try:
                self.library.compile(self._functions[name].name)
            except IRError:
                raise
            except Exception as e:
                raise IRError(
                    IRErrorKind.COMPILE_ERROR,
                    f"Failed to compile '{name}' (DSL: {expr})",
                    suggestions=["Check ROOT error log above"]
                ) from e
    
    def apply(self, rdf) -> Any:
        """
        Apply all definitions to an RDataFrame.
        
        This method:
        1. Compiles all functions (if not already compiled)
        2. Calls rdf.Define() for each definition in order
        
        Args:
            rdf: ROOT.RDataFrame instance
        
        Returns:
            Modified RDataFrame with new columns
        
        Example:
            >>> rdf = ROOT.RDataFrame("Events", "data.root")
            >>> rdf = dsl.apply(rdf)
            >>> result = rdf.AsNumpy(["pt", "n_tracks"])
        """
        self.compile_all()
        
        for name, _ in self._definitions:
            func = self._functions[name]
            rdf = rdf.Define(name, func.get_call_expression())
        
        return rdf
    
    def preview(self) -> str:
        """
        Return all generated C++ code without compiling.
        
        Returns:
            String containing all generated C++ code with DSL comments
        
        Example:
            >>> print(dsl.preview())
            // DSL: sqrt(px**2 + py**2)
            double alias_pt(double px, double py) { ... }
        """
        return self.library.preview()
    
    def export_macro(self, filepath: str, include_test: bool = True,
                     tree_name: str = "Events") -> None:
        """
        Export all functions to a .C macro file.
        
        The exported macro includes:
        - All required headers
        - DSL expression as comment before each function
        - Optional test_all() harness for verification
        
        Args:
            filepath: Output file path
            include_test: Include test_all() harness
            tree_name: TTree name for test harness
        
        Example:
            >>> dsl.export_macro("my_dsl.C")
            >>> # Then run: root -l 'my_dsl.C("data.root")'
        """
        self.library.export_to_file(filepath, include_test, tree_name)
    
    def list_definitions(self) -> List[tuple]:
        """
        Return list of (name, expression) pairs in definition order.
        
        Returns:
            List of tuples: [(name, expression), ...]
        """
        return list(self._definitions)
    
    def get_function(self, name: str) -> GeneratedFunction:
        """
        Get generated function by name.
        
        Args:
            name: Definition name (not alias_name)
            
        Returns:
            GeneratedFunction instance
            
        Raises:
            KeyError: If no definition with that name
        """
        if name not in self._functions:
            raise KeyError(f"No definition named '{name}'")
        return self._functions[name]
    
    def __len__(self) -> int:
        """Return number of definitions."""
        return len(self._definitions)
    
    def __repr__(self) -> str:
        return f"DSLCompiler(schema={list(self.schema.keys())}, definitions={len(self._definitions)})"
    
    # =========================================================================
    # Phase 10: UX/API Sugar
    # =========================================================================
    
    @classmethod
    def from_tree(cls, 
                  filename: str, 
                  treename: str,
                  overrides: Optional[Dict[str, str]] = None,
                  safe_indexing: bool = True) -> 'DSLCompiler':
        """
        Create DSLCompiler with schema auto-inferred from ROOT file.
        
        Reads branch types directly from the TTree, eliminating manual
        schema definition. Use `overrides` for branches that need
        manual type specification (e.g., TClonesArray).
        
        Args:
            filename: Path to ROOT file
            treename: Name of TTree
            overrides: Manual type overrides {branch: cpp_type}
                       These REPLACE any auto-detected types.
            safe_indexing: Enable bounds checking (default True)
        
        Returns:
            DSLCompiler with auto-inferred schema
        
        Raises:
            FileNotFoundError: If ROOT file doesn't exist
            KeyError: If tree not found in file
        
        Example:
            >>> dsl = DSLCompiler.from_tree("data.root", "Events")
            >>> print(dsl.schema)  # Auto-detected types
            {'px': 'double', 'py': 'double', 'tracks': 'RVec<TLorentzVector>'}
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
        
        Supported Types:
            - Scalars: double, float, int, unsigned int, bool
            - Vectors: std::vector<T>, RVec<T>
            - Objects: TLorentzVector, TVector3, TParticle
            - ROOT types: Double_t, Float_t, Int_t, UInt_t, Bool_t, etc.
        
        Limitations:
            - TClonesArray requires manual override
            - Nested collections (vector<vector<T>>) not supported
            - Custom classes may need overrides
        
        Note:
            File handle is released after reading metadata.
        """
        import ROOT
        
        # Open file
        f = ROOT.TFile.Open(filename)
        if not f or f.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file: {filename}")
        
        # Get tree
        tree = f.Get(treename)
        if not tree:
            f.Close()
            raise KeyError(f"Tree '{treename}' not found in {filename}")
        
        # Use existing TypeInferrer.from_tree(tree)
        inferrer = TypeInferrer.from_tree(tree)
        
        # Get simple schema
        schema = inferrer.to_simple_schema()
        
        # Close file (metadata extracted)
        f.Close()
        
        # Apply overrides (user overrides win)
        if overrides:
            schema.update(overrides)
        
        # Create instance
        instance = cls(schema, safe_indexing=safe_indexing)
        
        # Store source info for debugging
        instance._source_file = filename
        instance._source_tree = treename
        
        return instance
    
    def show_types(self, include_definitions: bool = True) -> str:
        """
        Display inferred types for all columns and definitions.
        
        Useful for debugging type inference issues and verifying
        that auto-detection worked correctly.
        
        Args:
            include_definitions: Also show types of defined columns
        
        Returns:
            Formatted string showing column types
        
        Example:
            >>> dsl = DSLCompiler.from_tree("data.root", "Events")
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
            >>> print(dsl.show_types())
            Schema columns:
              px          : double
              py          : double
              tracks      : RVec<TLorentzVector>
            
            Defined columns:
              pt          : double         = sqrt(px**2 + py**2)
        """
        lines = ["Schema columns:"]
        
        # Get original schema columns (exclude defined aliases)
        defined_names = {name for name, _ in self._definitions}
        original_schema = {k: v for k, v in self.schema.items() 
                          if k not in defined_names}
        
        for name, dtype in sorted(original_schema.items()):
            lines.append(f"  {name:12}: {dtype}")
        
        # Defined columns
        if include_definitions and self._definitions:
            lines.append("")
            lines.append("Defined columns:")
            for name, expr in self._definitions:
                func = self._functions[name]
                ret_type = func.return_type
                lines.append(f"  {name:12}: {ret_type:14} = {expr}")
        
        return "\n".join(lines)
    
    def validate(self) -> List[str]:
        """
        Validate all definitions without compiling to ROOT.
        
        This is a fast consistency check that verifies all definitions
        have generated C++ code and can be used. Most syntax and type
        errors are raised at define() time, so this method primarily
        confirms that the DSLCompiler is in a valid state.
        
        Does NOT invoke ROOT compilation - use compile_all() for that.
        
        Returns:
            List of error messages (empty if all valid)
        
        Example:
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
            >>> errors = dsl.validate()
            >>> if errors:
            ...     print("Validation failed:", errors)
            >>> else:
            ...     print("All definitions valid")
        """
        errors = []
        
        for name, expr in self._definitions:
            try:
                func = self._functions.get(name)
                if func is None:
                    errors.append(f"{name}: Function not found")
                elif not func.code:
                    errors.append(f"{name}: No code generated")
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
        
        return errors
    
    # =========================================================================
    # Phase 12.1: dfdraw Integration
    # =========================================================================
    
    def _collect_dependencies(self, expr: str) -> Set[str]:
        """
        Extract column dependencies from expression using IR parsing.
        
        Uses existing IRBuilder infrastructure for robust parsing that handles:
        - Namespace functions (TMath.Sin)
        - Nested expressions
        - Aliases referencing other aliases
        
        Args:
            expr: Expression string (e.g., "sqrt(pt**2 + eta**2)")
        
        Returns:
            Set of column/alias names referenced in expression
        """
        from .ir_nodes import VariableNode
        
        dependencies = set()
        
        try:
            # Preprocess and parse using existing infrastructure
            preprocessed = self._preprocess_expression(expr)
            # Create temporary builder (self._builder may not exist)
            builder = IRBuilder(self._inferrer)
            ir = builder.build(preprocessed, "_dep_check")
            
            # Use the built-in walk() method to collect all VariableNodes
            for node in ir.walk():
                if isinstance(node, VariableNode):
                    dependencies.add(node.name)
            
        except Exception:
            # Fallback to tokenization if IR parsing fails
            # (e.g., for selection strings with && operators not in DSL)
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
            keywords = {
                'and', 'or', 'not', 'in', 'True', 'False', 'true', 'false',
                'abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pow',
                'TMath', 'ROOT', 'std', 'int', 'float', 'double', 'bool',
                'RVec', 'Take', 'Range', 'Sum', 'Mean', 'Min', 'Max',
            }
            dependencies = {t for t in tokens if t not in keywords}
        
        return dependencies
    
    def _collect_draw_dependencies(self, expr: str, selection: str = None,
                                    group_by: str = None, color: str = None) -> Set[str]:
        """
        Collect all column dependencies for a draw operation.
        
        Args:
            expr: Plot expression ('pt', 'y:x', 'dy:row')
            selection: Optional filter expression
            group_by: Optional grouping column
            color: Optional color column
        
        Returns:
            Set of column names needed for the draw operation
        """
        columns = set()
        
        # Parse plot expression (handle 'y:x' syntax)
        for part in expr.replace(' ', '').split(':'):
            if part:  # Skip empty parts
                columns.update(self._collect_dependencies(part))
        
        # Parse selection
        if selection:
            columns.update(self._collect_dependencies(selection))
        
        # Direct column references
        if group_by:
            columns.add(group_by)
        if color and isinstance(color, str):
            columns.add(color)
        
        # Check against schema and warn about unknowns
        known_columns = set(self.schema.keys())
        unknown = columns - known_columns
        
        if unknown:
            warnings.warn(
                f"Columns not in schema (will try as raw branches): {unknown}\n"
                f"Known columns: {sorted(known_columns)[:10]}{'...' if len(known_columns) > 10 else ''}"
            )
        
        return columns
    
    def draw(self, expr: str, rdf, columns: List[str] = None,
             max_entries: int = None, **kwargs):
        """
        Draw a plot using dfdraw with automatic column detection.
        
        Args:
            expr: Plot expression ('pt', 'y:x', 'dy:row')
            rdf: Applied RDataFrame (after dsl.apply())
            columns: Optional explicit column list (auto-detected if None)
            max_entries: Optional limit on number of entries (for large datasets)
            **kwargs: Passed to dfdraw.DFDraw.draw()
                - selection: Filter expression (e.g., "isOK && pt > 1.0")
                - type: Plot type ('hist', 'scatter', 'profile', 'hist2d')
                - bins: Number of bins
                - group_by: Grouping column
                - color: Color column
        
        Returns:
            Tuple of (fig, ax, stats) from dfdraw
        
        Raises:
            ImportError: If dfdraw is not installed
        
        Example:
            >>> dsl.draw("pt:eta", rdf, selection="isOK")
            >>> dsl.draw("trackPt", rdf, max_entries=10000)  # RVec column
        """
        # Lazy import with helpful error
        try:
            from dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw is required for visualization.\n"
                "Install with: pip install dfdraw\n"
                "Or use to_dataframe() for manual plotting."
            )
        
        import pandas as pd
        
        # Auto-detect columns if not provided
        if columns is None:
            columns = list(self._collect_draw_dependencies(
                expr,
                kwargs.get('selection'),
                kwargs.get('group_by'),
                kwargs.get('color')
            ))
        
        # Apply entry limit if specified (for large datasets)
        if max_entries is not None:
            rdf = rdf.Range(max_entries)
        
        # Extract data
        result = rdf.AsNumpy(columns)
        
        # Phase 12.2: Flatten RVec columns for dfdraw compatibility
        result = self._flatten_rvec_columns(result)
        
        # Create drawer and draw
        drawer = DFDraw(pd.DataFrame(result))
        return drawer.draw(expr, **kwargs)
    
    def draw_batch(self, specs: Dict[str, dict], rdf,
                   save_dir: str = None, max_entries: int = None,
                   **defaults):
        """
        Draw multiple plots with a single AsNumpy call (efficient).
        
        Args:
            specs: Dict of {name: {expr: str, ...options}}
            rdf: Applied RDataFrame
            save_dir: Optional directory to save plots as PNG
            max_entries: Optional limit on entries
            **defaults: Default options applied to all plots
        
        Returns:
            Dict of {name: {fig, ax, stats}}
        
        Example:
            >>> specs = {
            ...     'pt_dist': {'expr': 'trackPt', 'bins': 50},
            ...     'dy_vs_z': {'expr': 'clusterDy:clusterZ', 'type': 'hist2d'},
            ...     'eta_good': {'expr': 'trackEta', 'selection': 'trackIsOK'},
            ... }
            >>> dsl.draw_batch(specs, rdf, save_dir='qa/')
        """
        # Handle empty specs
        if not specs:
            return {}
        
        # Lazy import
        try:
            from dfdraw import DFDraw
        except ImportError:
            raise ImportError(
                "dfdraw is required for visualization.\n"
                "Install with: pip install dfdraw\n"
                "Or use to_dataframe() for manual plotting."
            )
        
        from pathlib import Path
        import pandas as pd
        import numpy as np
        
        # Collect ALL columns from ALL specs (single AsNumpy call)
        all_columns = set()
        for spec in specs.values():
            all_columns.update(self._collect_draw_dependencies(
                spec.get('expr', ''),
                spec.get('selection', defaults.get('selection')),
                spec.get('group_by', defaults.get('group_by')),
                spec.get('color', defaults.get('color'))
            ))
        
        # Apply entry limit
        if max_entries is not None:
            rdf = rdf.Range(max_entries)
        
        # Single data extraction (efficient!)
        result = rdf.AsNumpy(list(all_columns))
        
        # Phase 12.2: Flatten RVec columns for dfdraw compatibility
        result = self._flatten_rvec_columns(result)
        
        # Check if all columns have the same length after flattening
        lengths = {col: len(arr) for col, arr in result.items()}
        unique_lengths = set(lengths.values())
        
        # Generate all plots
        results = {}
        
        if len(unique_lengths) == 1:
            # All columns same length - can use single DataFrame (efficient)
            drawer = DFDraw(pd.DataFrame(result))
            
            for name, spec in specs.items():
                merged = {**defaults, **spec}
                expr = merged.pop('expr')
                
                fig, ax, stats = drawer.draw(expr, **merged)
                
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(f"{save_dir}/{name}.png", dpi=150, bbox_inches='tight')
                
                results[name] = {'fig': fig, 'ax': ax, 'stats': stats}
        else:
            # Different lengths - draw each spec separately with its own columns
            for name, spec in specs.items():
                merged = {**defaults, **spec}
                expr = merged.pop('expr')
                
                # Get columns needed for this spec
                spec_columns = self._collect_draw_dependencies(
                    expr,
                    merged.get('selection'),
                    merged.get('group_by'),
                    merged.get('color')
                )
                
                # Build DataFrame with only matching-length columns
                spec_data = {col: result[col] for col in spec_columns if col in result}
                
                # Check lengths within this spec
                spec_lengths = [len(arr) for arr in spec_data.values()]
                if len(set(spec_lengths)) > 1:
                    # Columns in this spec have different lengths - skip with warning
                    import warnings
                    warnings.warn(f"Skipping '{name}': columns have different lengths after RVec flattening")
                    continue
                
                drawer = DFDraw(pd.DataFrame(spec_data))
                fig, ax, stats = drawer.draw(expr, **merged)
                
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(f"{save_dir}/{name}.png", dpi=150, bbox_inches='tight')
                
                results[name] = {'fig': fig, 'ax': ax, 'stats': stats}
        
        return results
    
    def to_dataframe(self, rdf, columns: List[str] = None,
                     max_entries: int = None):
        """
        Export RDataFrame result to pandas DataFrame.
        
        Note: RVec columns will be stored as numpy arrays (one array per row).
        
        Args:
            rdf: Applied RDataFrame
            columns: List of columns to export (default: all in schema)
            max_entries: Optional limit on entries
        
        Returns:
            pandas DataFrame
        
        Example:
            >>> df = dsl.to_dataframe(rdf)
            >>> df = dsl.to_dataframe(rdf, columns=['trackPt', 'trackEta'])
            >>> df['trackPt'][0]  # First event's track pT array
        """
        import pandas as pd
        
        if columns is None:
            columns = list(self.schema.keys())
        
        if max_entries is not None:
            rdf = rdf.Range(max_entries)
        
        return pd.DataFrame(rdf.AsNumpy(columns))
