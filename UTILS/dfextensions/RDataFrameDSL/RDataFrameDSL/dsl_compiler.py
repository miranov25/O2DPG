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

from typing import Dict, List, Optional, Any, Set, Tuple, FrozenSet
import uuid
import re
import warnings
import hashlib
import threading
import time
from dataclasses import dataclass, field

from .type_inferrer import TypeInferrer, extract_inner_type, is_collection_type
from .ir_builder import IRBuilder
from .ir_types import IRType, IRTypeKind
from .backend_cpp import CppCodeGenerator, FunctionLibrary, GeneratedFunction
from .ir_errors import IRError, IRErrorKind


# Phase 13.2.DSL: PyArrow detection
try:
    import pyarrow as pa
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False
    pa = None  # For type hints


def _require_pyarrow():
    """Raise ImportError if PyArrow is not available."""
    if not _PYARROW_AVAILABLE:
        raise ImportError(
            "PyArrow required for Arrow export/import. "
            "Install with: pip install pyarrow>=12.0"
        )


def _arrow_type_to_ctype(arrow_type) -> str:
    """
    Map PyArrow type to C++ type string.
    
    Phase 13.2.DSL: Used by from_arrow() to infer schema.
    """
    import pyarrow as pa
    
    if pa.types.is_float64(arrow_type):
        return 'double'
    elif pa.types.is_float32(arrow_type):
        return 'float'
    elif pa.types.is_int32(arrow_type):
        return 'int'
    elif pa.types.is_int64(arrow_type):
        return 'long'
    elif pa.types.is_boolean(arrow_type):
        return 'bool'
    elif pa.types.is_list(arrow_type):
        inner = _arrow_type_to_ctype(arrow_type.value_type)
        return f'RVec<{inner}>'
    else:
        return 'double'  # fallback


__all__ = ['DSLCompiler']


# =============================================================================
# Phase 13.5.B: C++ Function Registration Constants
# =============================================================================

HASH_LENGTH = 16
HASH_SCHEMA_VERSION = 1

# Default headers for C++ function registration
REGISTER_DEFAULT_HEADERS: FrozenSet[str] = frozenset([
    "<cmath>",
    "<ROOT/RVec.hxx>",
])

# Header auto-detection map for register_function_cpp
REGISTER_HEADER_MAP: Dict[str, str] = {
    # Math functions
    'sqrt': '<cmath>', 'sin': '<cmath>', 'cos': '<cmath>',
    'tan': '<cmath>', 'atan2': '<cmath>', 'exp': '<cmath>',
    'log': '<cmath>', 'pow': '<cmath>', 'fabs': '<cmath>',
    'abs': '<cmath>', 'atanh': '<cmath>', 'acos': '<cmath>',
    'asin': '<cmath>', 'ceil': '<cmath>', 'floor': '<cmath>',
    # Algorithm
    'std::max': '<algorithm>', 'std::min': '<algorithm>',
    'std::sort': '<algorithm>', 'std::find': '<algorithm>',
    # RVec
    'RVec': '<ROOT/RVec.hxx>',
    'Sum': '<ROOT/RVec.hxx>', 'Mean': '<ROOT/RVec.hxx>',
    'StdDev': '<ROOT/RVec.hxx>', 'Var': '<ROOT/RVec.hxx>',
    # Physics types (T38: validated)
    'TLorentzVector': '<TLorentzVector.h>',
    'TVector3': '<TVector3.h>',
    'TVector2': '<TVector2.h>',
    # TMath
    'TMath::': '<TMath.h>',
}

# Lambda patterns to reject (FROZEN RULE #1)
LAMBDA_PATTERNS = [
    r'\[\s*[&=]?\s*\]\s*\(',           # [](, [&](, [=](
    r'\[\s*[&=]?\s*\w*\s*\]\s*\(',     # [&x](, [=x](
    r'auto\s+\w+\s*=\s*\[',            # auto f = [
    r'std::function\s*<[^>]+>\s*\w*\s*=\s*\[',  # std::function<...> = [
]


@dataclass
class RegisteredCppFunction:
    """Internal representation of a registered C++ function (Phase 13.5.B)."""
    name: str                           # User-facing name
    cpp_name: str                       # Internal name (dsl_{name}_{hash})
    hash: str                           # Content hash (16 chars)
    params: List[Tuple[str, str]]       # [(param_name, param_type), ...]
    return_type: str                    # Return type
    body: str                           # Function body
    full_cpp: str                       # Complete C++ code
    headers: Set[str]                   # Required headers
    pragmas: Set[str]                   # Raw pragma lines
    declared: bool                      # Successfully declared in ROOT
    timestamp: float = field(default_factory=time.time)


def _simple_schema_to_full(simple_schema: Dict[str, str]) -> Dict:
    """
    Convert simple schema to full TypeInferrer format.
    
    Simple: {"px": "double", "pt": "RVec<double>", "nested": "RVec<RVec<double>>"}
    Full: {"columns": {"px": {"dtype": "double", "rank": 0}, 
                       "pt": {"dtype": "double", "rank": 1},
                       "nested": {"dtype": "double", "rank": 2}, ...}}
    
    Phase 13.3.DSL: Now correctly handles nested RVec types (D1-D4 bug fix).
    Uses extract_inner_type() to properly determine:
    - dtype: innermost element type (e.g., "double" for RVec<RVec<double>>)
    - rank: nesting depth (e.g., 2 for RVec<RVec<double>>)
    
    Args:
        simple_schema: Dict mapping column names to C++ type strings
        
    Returns:
        Full schema dict for TypeInferrer.from_schema()
    """
    columns = {}
    for name, type_str in simple_schema.items():
        if is_collection_type(type_str):
            # Phase 13.3.DSL: Use extract_inner_type for correct nested type handling
            inner_type, depth = extract_inner_type(type_str)
            columns[name] = {
                "dtype": inner_type,
                "rank": depth,
                "cpp_type": type_str,
            }
        else:
            # Scalar type
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
    
    # Phase 13.5.B: Class-level state for C++ function registration (thread-safe)
    _global_cpp_compile_lock: threading.RLock = threading.RLock()
    _global_cpp_declared_names: Set[str] = set()
    
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
        
        # Phase 13.2.DSL: Optional RDataFrame reference for to_arrow()
        self._rdf = None
        
        # Phase 12.2: Track if JIT helpers have been declared
        self._helpers_declared = False
        
        # Phase 13.5.B: Storage for registered C++ functions
        self._registered_cpp_functions: Dict[str, RegisteredCppFunction] = {}
        self._registered_cpp_by_name: Dict[str, List[str]] = {}
    
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
            ValueError: If C-array ND operation attempted without dimensions
        
        Example:
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
            >>> dsl.define("high_pt", "pt > 10.0")  # Uses 'pt' alias
            >>> dsl.define("isOK", "(row < 152) & (abs(dy) < 10)", dtype="bool")
            >>> dsl.define("sector", "int(9*phi/pi)", dtype="int8")
            >>> dsl.define("row0", "mat[0,:]")  # C-array row extraction (D9)
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
        
        # Phase 13.4.D9: Check for C-array expression (AST-authoritative routing)
        if self._is_carray_expression(expression):
            return self._define_carray(name, expression, dtype)
        
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
    
    # =========================================================================
    # Phase 13.4.D9: C-Array Integration
    # =========================================================================
    
    def _is_carray_expression(self, expression: str) -> bool:
        """
        Check if expression contains C-array multi-index operation.
        
        Phase 13.4.D9: AST-authoritative routing via CArrayExpressionAnalyzer.
        
        Returns True only if:
        1. Expression contains multi-index syntax (comma in brackets)
        2. Base variable is a Name node (not attribute/call)
        3. Base variable exists in schema with carray_shape
        
        Args:
            expression: DSL expression string
            
        Returns:
            True if expression should be routed to C-array compiler
        """
        # Quick pre-filter
        if ',' not in expression or '[' not in expression:
            return False
        
        # Build C-array schema and use analyzer
        carray_schema = self._build_carray_schema()
        if not carray_schema:
            return False
        
        try:
            from .dsl_carray import CArrayExpressionAnalyzer
            analyzer = CArrayExpressionAnalyzer(carray_schema)
            return analyzer.is_carray_expression(expression)
        except ImportError:
            return False
    
    def _define_carray(self, name: str, expression: str, dtype: str = None) -> 'DSLCompiler':
        """
        Define a column using C-array expression.
        
        Phase 13.4.D9: Handles C-array ND operations.
        
        Args:
            name: Output column name
            expression: C-array expression (e.g., "mat[0,:]")
            dtype: Optional explicit return type
            
        Returns:
            self (for chaining)
            
        Raises:
            ValueError: If C-array dimensions unknown
        """
        from .dsl_carray import CArrayExpressionAnalyzer, CArrayDSLCompiler
        
        # Build C-array schema
        carray_schema = self._build_carray_schema()
        
        # Extract base variable
        analyzer = CArrayExpressionAnalyzer(carray_schema)
        base_var = analyzer.extract_carray_base(expression)
        
        if base_var is None:
            # Should not happen if _is_carray_expression returned True
            raise ValueError(
                f"Cannot extract C-array base from '{expression}'"
            )
        
        # Check if dimensions are known (P0-3: fail-closed)
        info = self._inferrer._variables.get(base_var)
        
        if info is None or not getattr(info, 'carray_shape', None):
            # Dimensions unknown - cannot compute ND operations
            import warnings
            warnings.warn(
                f"C-array dimensions unknown for '{base_var}'. "
                f"Use from_tree() or provide carray_schema for ND operations.",
                UserWarning
            )
            raise ValueError(
                f"Cannot evaluate '{expression}': C-array dimensions unknown for '{base_var}'. "
                f"Use from_tree() or provide carray_schema for ND operations."
            )
        
        # Compile with CArrayDSLCompiler
        compiler = CArrayDSLCompiler(carray_schema)
        result = compiler.compile(expression)
        
        # Register the definition
        self._definitions.append((name, expression))
        
        # Get generated function and register
        if result and hasattr(result, 'code'):
            # Create a GeneratedFunction-like object
            func = GeneratedFunction(
                name=f"{name}_{self._unique_id}",
                code=result.code,
                return_type=result.return_type if hasattr(result, 'return_type') else 'auto',
                dsl_expression=expression,
            )
            func.column_name = name
            self._functions[name] = func
            self.library.add(func)
            
            # Register alias type
            if dtype:
                self.schema[name] = self._dtype_to_cpp_type(dtype)
            elif hasattr(result, 'return_type'):
                self.schema[name] = result.return_type
            else:
                self.schema[name] = 'auto'
        
        return self
    
    def _build_carray_schema(self) -> Dict:
        """
        Build C-array schema for CArrayDSLCompiler from VariableInfo.
        
        Phase 13.4.D9: Converts VariableInfo with carray_shape to CArrayType.
        
        Returns:
            Dict mapping variable names to CArrayType objects
        """
        try:
            from .schema_parser import CArrayType, Dimension, DimensionKind
        except ImportError:
            return {}
        
        schema = {}
        
        for name, info in self._inferrer._variables.items():
            # Check if info has carray_shape attribute
            carray_shape = getattr(info, 'carray_shape', None)
            if carray_shape:
                # Convert dimensions
                dims = []
                for d in carray_shape:
                    if isinstance(d, int):
                        dims.append(Dimension(DimensionKind.FIXED, d))
                    else:
                        dims.append(Dimension(DimensionKind.VARIABLE, d))
                
                # Use normalized scalar type (P0-4 fix)
                scalar_type = self._get_scalar_cpp_type(info)
                
                schema[name] = CArrayType(
                    base=scalar_type,
                    dims=dims,
                )
        
        return schema
    
    def _get_scalar_cpp_type(self, info) -> str:
        """
        Return canonical scalar C++ type for C-array elements.
        
        Phase 13.4.D9 P0-4: Normalizes container types to scalar.
        
        Args:
            info: VariableInfo object
            
        Returns:
            Scalar C++ type (e.g., 'float', not 'RVec<float>')
        """
        TYPE_MAP = {
            # ROOT types
            "Float_t": "float",
            "Double_t": "double",
            "Int_t": "int",
            "UInt_t": "unsigned int",
            "Long64_t": "long long",
            "ULong64_t": "unsigned long long",
            "Short_t": "short",
            "UShort_t": "unsigned short",
            "Char_t": "char",
            "UChar_t": "unsigned char",
            "Bool_t": "bool",
            
            # RVec types (extract element type)
            "ROOT::VecOps::RVec<float>": "float",
            "ROOT::VecOps::RVec<double>": "double",
            "ROOT::VecOps::RVec<int>": "int",
            "ROOT::VecOps::RVec<Float_t>": "float",
            "ROOT::VecOps::RVec<Double_t>": "double",
            "RVec<float>": "float",
            "RVec<double>": "double",
            "RVec<int>": "int",
            
            # Standard types (pass through)
            "float": "float",
            "double": "double",
            "int": "int",
            "long": "long",
            "short": "short",
            "char": "char",
            "bool": "bool",
        }
        
        cpp_type = getattr(info, 'cpp_type', None) or str(getattr(info, 'dtype', 'float'))
        return TYPE_MAP.get(cpp_type, "float")  # Default to float
    
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
                     max_entries: int = None, show: bool = False,
                     # Phase 12.5.DSL: Statistical annotations
                     show_statistics: bool = False,
                     show_expected: bool = False,
                     expected_mean: float = 0.0,
                     expected_std: float = 1.0,
                     ) -> Dict[str, Any]:
        """
        Draw multiple composed figures with automatic column detection.
        
        Phase 12.3: Creates multi-subplot figures from declarative specifications.
        Each figure can contain multiple plots arranged in a grid.
        
        Phase 12.5.DSL: Added statistical annotation support for QA validation.
        
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
                    - is_pull: bool - Override auto pull detection
                    - Any other dfdraw parameters
                    
            rdf: Applied RDataFrame
            save_dir: Default directory for saving figures
            defaults: Default parameters applied to all plots
            max_entries: Limit entries for large datasets
            show: Call plt.show() after drawing
            show_statistics: Add statistics box (μ, σ, n) to histogram plots
            show_expected: Add Gaussian N(0,1) overlay for pull distributions
            expected_mean: Expected mean for Δμ calculation (default: 0.0)
            expected_std: Expected std for Δσ calculation (default: 1.0)
            
        Returns:
            Dict mapping figure names to {'fig': Figure, 'axes': list, 'stats': list}
            
        Notes:
            Pull detection: Expressions containing 'pull' (case-insensitive) are
            treated as pull distributions. Override with plot_spec['is_pull'].
            
            Statistics and overlays only apply to histogram-like plots.
            
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
            >>> results = dsl.draw_figures(qa_report, rdf, show_statistics=True, show_expected=True)
        """
        # Lazy imports
        try:
            from dfextensions.dfdraw import DFDraw
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
                is_pull_override = merged.pop('is_pull', None)  # Phase 12.5.DSL: Remove before draw()
                
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
                    
                    # === Phase 12.5.DSL: Statistical annotations ===
                    # Parse expression and plot type for annotation decisions
                    plot_type = merged.get('type', 'hist')
                    
                    # Determine if this is a pull distribution
                    if is_pull_override is not None:
                        is_pull = is_pull_override
                    else:
                        is_pull = 'pull' in expr.lower()
                    
                    # Only apply annotations to histogram-like plots
                    is_histogram = plot_type in ('hist', 'histogram', None, 'hist1d')
                    
                    if is_histogram and (show_statistics or show_expected):
                        # Extract the x-expression (before : if 2D)
                        x_expr = expr.split(':')[0].strip() if ':' in expr else expr.strip()
                        
                        # Get values from plot_data
                        values = None
                        if x_expr in plot_data:
                            raw_values = plot_data[x_expr]
                            # Handle NaN values
                            values = raw_values[~np.isnan(raw_values)] if hasattr(raw_values, '__len__') else None
                        
                        # Add statistics box
                        if show_statistics and values is not None and len(values) > 0:
                            try:
                                drawer.add_statistics_box(
                                    ax,
                                    values,
                                    expected_mean=expected_mean if is_pull else None,
                                    expected_std=expected_std if is_pull else None,
                                )
                            except AttributeError:
                                warnings.warn(
                                    "dfdraw.add_statistics_box() not available. "
                                    "Update dfdraw to enable statistics display.",
                                    stacklevel=2
                                )
                            except Exception as e:
                                warnings.warn(f"Could not add statistics box for '{expr}': {e}")
                        
                        # Add Gaussian overlay for pull distributions
                        if show_expected and is_pull:
                            try:
                                drawer.add_reference_overlay(
                                    ax,
                                    func='gaussian',
                                    mu=expected_mean,
                                    sigma=expected_std,
                                    label=f'N({expected_mean},{expected_std})',
                                )
                            except AttributeError:
                                warnings.warn(
                                    "dfdraw.add_reference_overlay() not available. "
                                    "Update dfdraw to enable reference overlays.",
                                    stacklevel=2
                                )
                            except Exception as e:
                                warnings.warn(f"Could not add reference overlay for '{expr}': {e}")
                    # === End Phase 12.5.DSL ===
                    
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
            from dfextensions.dfdraw import DFDraw
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
            from dfextensions.dfdraw import DFDraw
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
    
    # =========================================================================
    # Phase 12.6.DSL: AliasDataFrame Export
    # =========================================================================
    
    def to_aliasdf(
        self,
        include: List[str] = None,
        exclude: List[str] = None,
        dtype_map: Dict[str, str] = None,
    ) -> dict:
        """
        Export DSL definitions to AliasDataFrame schema format.
        
        Exports definitions only (not data). C++ operators are converted
        to Python/pandas equivalents with proper precedence handling.
        
        Phase 12.6.DSL: Enables workflow migration from RDataFrameDSL to
        AliasDataFrame for cases where pandas-based analysis is preferred.
        
        Parameters
        ----------
        include : List[str], optional
            List of definition names to include. None = all.
        exclude : List[str], optional
            List of definition names to exclude.
        dtype_map : Dict[str, str], optional
            Map of definition names to dtypes (e.g., {'pt_gev': 'float32'}).
            
        Returns
        -------
        dict
            Schema compatible with AliasDataFrame.apply_schema():
            {
                'columns': {'name': {'expr': 'expression', 'dtype': 'dtype'}, ...},
                '__meta__': {'source': 'RDataFrameDSL', ...}
            }
            
        Notes
        -----
        Operator conversions (with precedence safety):
            a > 0 && b < 1  →  (a > 0) & (b < 1)
            a > 0 || b < 1  →  (a > 0) | (b < 1)
            !flag           →  ~flag
            a != b          →  a != b (preserved)
            
        Complex C++ expressions (TMath, ROOT functions) may not convert
        correctly. A warning is issued for potentially problematic expressions.
        
        Examples
        --------
        >>> dsl.define("good_track", "pt > 0.5 && nHits > 5")
        >>> schema = dsl.to_aliasdf()
        >>> adf.apply_schema(schema)
        
        >>> # With filtering
        >>> schema = dsl.to_aliasdf(include=['pt_gev'], dtype_map={'pt_gev': 'float32'})
        """
        from datetime import datetime
        
        schema = {
            'columns': {},
            '__meta__': {
                'source': 'RDataFrameDSL',
                'export_version': '1.0',
                'exported_at': datetime.now().isoformat(),
            }
        }
        
        dtype_map = dtype_map or {}
        
        for name, expr in self._definitions:
            # Apply include filter
            if include is not None and name not in include:
                continue
            
            # Apply exclude filter
            if exclude is not None and name in exclude:
                continue
            
            # Convert expression
            try:
                py_expr = self._cpp_to_python_expr(expr)
                
                # Build column spec with expr (AliasDataFrame format)
                col_spec = {'expr': py_expr}
                
                # Add dtype if specified
                if name in dtype_map:
                    col_spec['dtype'] = dtype_map[name]
                
                schema['columns'][name] = col_spec
                    
            except Exception as e:
                warnings.warn(
                    f"Could not convert definition '{name}': {e}. Skipping.",
                    stacklevel=2
                )
        
        return schema
    
    def _cpp_to_python_expr(self, cpp_expr: str) -> str:
        """
        Convert C++ expression syntax to Python/pandas eval syntax.
        
        Handles boolean operators with proper precedence (wraps in parentheses).
        
        Parameters
        ----------
        cpp_expr : str
            C++ expression string
            
        Returns
        -------
        str
            Python/pandas compatible expression
            
        Warns
        -----
        If expression contains potentially unconvertible constructs
        (TMath, ROOT namespace, method calls)
        """
        py_expr = cpp_expr
        
        # Check for problematic constructs first
        problematic = []
        
        if 'TMath::' in cpp_expr or 'TMath.' in cpp_expr:
            problematic.append('TMath functions')
        
        if 'ROOT::' in cpp_expr:
            problematic.append('ROOT namespace')
        
        if re.search(r'\.\w+\s*\(', cpp_expr):
            problematic.append('method calls')
        
        if problematic:
            warnings.warn(
                f"Expression may not convert correctly ({', '.join(problematic)}): "
                f"'{cpp_expr}'",
                stacklevel=3
            )
        
        # Handle boolean operators WITH PRECEDENCE SAFETY
        if '&&' in py_expr or '||' in py_expr:
            py_expr = self._convert_boolean_expr(py_expr)
        
        # Unary NOT (preserve !=)
        # Replace ! that is not followed by =
        py_expr = re.sub(r'!(?!=)', '~', py_expr)
        
        return py_expr
    
    def _convert_boolean_expr(self, expr: str) -> str:
        """
        Convert C++ boolean expression to Python with correct precedence.
        
        C++ precedence: && binds tighter than ||
        Therefore: a && b || c && d  means  (a && b) || (c && d)
        
        Strategy: Split by || first (lower precedence), then && within each part.
        
        Parameters
        ----------
        expr : str
            Expression containing && or ||
            
        Returns
        -------
        str
            Expression with (a > 0) & (b > 0) format
            
        Examples
        --------
        >>> self._convert_boolean_expr('a > 0 && b < 1')
        '(a > 0) & (b < 1)'
        >>> self._convert_boolean_expr('a > 0 || b < 1')
        '(a > 0) | (b < 1)'
        >>> self._convert_boolean_expr('a && b || c && d')
        '(a) & (b) | (c) & (d)'
        """
        result = expr
        
        # Handle || FIRST (lower precedence in C++)
        if '||' in result:
            or_parts = result.split('||')
            converted_or_parts = []
            for part in or_parts:
                part = part.strip()
                # Handle && within each OR-term
                if '&&' in part:
                    and_parts = part.split('&&')
                    wrapped_and = [f'({p.strip()})' for p in and_parts]
                    part = ' & '.join(wrapped_and)
                elif not (part.startswith('(') and part.endswith(')')):
                    part = f'({part})'
                converted_or_parts.append(part)
            result = ' | '.join(converted_or_parts)
        elif '&&' in result:
            # Only && present (no ||)
            parts = result.split('&&')
            wrapped = [f'({p.strip()})' for p in parts]
            result = ' & '.join(wrapped)
        
        return result
    
    def get_definitions(self) -> Dict[str, str]:
        """
        Return all definitions as {name: expression} dict.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping definition names to their expressions
            
        Example
        -------
        >>> dsl.define("pt_gev", "trackPt / 1000")
        >>> dsl.define("good", "pt_gev > 0.5")
        >>> dsl.get_definitions()
        {'pt_gev': 'trackPt / 1000', 'good': 'pt_gev > 0.5'}
        """
        return {name: expr for name, expr in self._definitions}
    
    # =========================================================================
    # Phase 13.2.DSL: ROOT ↔ Arrow Bridge
    # =========================================================================
    
    def to_arrow(
        self,
        rdf=None,
        columns: List[str] = None,
        flatten_rvec: bool = False,
        include_schema: bool = True,
    ):
        """
        Export RDataFrame result as PyArrow Table.
        
        Phase 13.2.DSL: Enables Arrow-based interchange with copy-based
        Phase 1 implementation. Future versions may support zero-copy
        if ROOT adds native Arrow support.
        
        Parameters
        ----------
        rdf : ROOT.RDataFrame, optional
            RDataFrame to export. If None, uses internal _rdf reference.
        columns : List[str], optional
            Columns to export. If None, exports all available columns.
        flatten_rvec : bool, default=False
            If True, flatten RVec columns to 1D (loses event structure).
            If False, preserve as Arrow ListArray (maintains structure).
        include_schema : bool, default=True
            If True, embed DSL schema in Arrow metadata for round-trip.
        
        Returns
        -------
        pa.Table
            PyArrow Table with exported data.
            
        Raises
        ------
        ImportError
            If PyArrow is not installed.
        ValueError
            If no RDataFrame is available.
            
        Notes
        -----
        Phase 1 uses numpy as intermediate layer (copies data).
        RVec → ListArray conversion materializes all events in memory.
        
        Examples
        --------
        >>> table = dsl.to_arrow(columns=['pt', 'eta'])
        >>> pa.parquet.write_table(table, 'output.parquet')
        
        >>> # With schema for round-trip
        >>> table = dsl.to_arrow(include_schema=True)
        >>> new_dsl = DSLCompiler.from_arrow(table)
        """
        _require_pyarrow()
        import pyarrow as pa
        import json
        import numpy as np
        
        rdf = rdf or self._rdf
        if rdf is None:
            raise ValueError(
                "No RDataFrame available. Either pass rdf parameter "
                "or set dsl._rdf after calling apply()."
            )
        
        # Determine columns to export
        # Note: GetColumnNames() returns ROOT strings, convert to Python strings
        if columns is None:
            columns = [str(c) for c in rdf.GetColumnNames()]
        else:
            columns = [str(c) for c in columns]
        
        # Export columns
        arrays = {}
        rvec_columns = []
        
        for col in columns:
            col_type = str(rdf.GetColumnType(col))
            
            if 'RVec' in col_type:
                rvec_columns.append(col)
                if flatten_rvec:
                    arrays[col] = self._flatten_rvec(rdf, col)
                else:
                    arrays[col] = self._rvec_to_listarray(rdf, col)
            else:
                # Standard scalar column
                arrays[col] = rdf.AsNumpy([col])[col]
        
        # Create PyArrow Table
        table = pa.Table.from_pydict(arrays)
        
        # Add schema metadata if requested
        if include_schema:
            schema_dict = self.to_aliasdf()
            metadata = {
                b'dsl_schema': json.dumps(schema_dict).encode('utf-8'),
                b'rvec_columns': json.dumps(rvec_columns).encode('utf-8'),
            }
            # Preserve existing metadata
            existing = table.schema.metadata or {}
            existing.update(metadata)
            table = table.replace_schema_metadata(existing)
        
        return table
    
    def _flatten_rvec(self, rdf, column: str):
        """
        Flatten RVec column to 1D numpy array.
        
        Warning: Loses event structure. Use for aggregate analysis only.
        """
        import numpy as np
        
        nested_data = rdf.AsNumpy([column])[column]
        
        # Concatenate all events
        result = []
        for event_array in nested_data:
            result.extend(event_array)
        
        return np.array(result)
    
    def _rvec_to_listarray(self, rdf, column: str):
        """
        Convert RVec column to PyArrow ListArray.
        
        Preserves jagged event structure as Arrow ListArray.
        
        Note: Materializes all events in memory. For large datasets,
        consider processing in chunks.
        """
        import pyarrow as pa
        import numpy as np
        
        # AsNumpy returns nested object array for RVec
        nested_data = rdf.AsNumpy([column])[column]
        
        # Build ListArray from variable-length arrays
        values = []
        offsets = [0]
        
        for event_array in nested_data:
            values.extend(event_array)
            offsets.append(len(values))
        
        # Warn on large materialization
        if len(nested_data) > 1_000_000:
            warnings.warn(
                f"Large RVec materialization: {len(nested_data)} events. "
                "Memory usage may be high.",
                stacklevel=2
            )
        
        return pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int64()),  # int64 for safety
            pa.array(values)
        )
    
    @classmethod
    def from_arrow(
        cls,
        table,
        apply_schema: bool = True,
    ) -> 'DSLCompiler':
        """
        Create DSLCompiler from PyArrow Table.
        
        Phase 13.2.DSL: Copy-based implementation using numpy intermediate.
        
        Parameters
        ----------
        table : pa.Table
            Input PyArrow Table.
        apply_schema : bool, default=True
            If True and table has DSL schema in metadata, apply definitions.
        
        Returns
        -------
        DSLCompiler
            New DSLCompiler instance with RDataFrame from table data.
            
        Raises
        ------
        ImportError
            If PyArrow or ROOT is not available.
            
        Notes
        -----
        Creates a copy of the data (Phase 1 implementation).
        Schema round-trip uses best-effort expression conversion.
        
        Examples
        --------
        >>> table = pa.parquet.read_table('data.parquet')
        >>> dsl = DSLCompiler.from_arrow(table, apply_schema=True)
        """
        _require_pyarrow()
        import ROOT
        import json
        
        # === NULL CHECK (Phase 13.2.3.DSL) ===
        for field in table.schema:
            column = table[field.name]
            if column.null_count > 0:
                raise TypeError(
                    f"Column '{field.name}' contains {column.null_count} null values. "
                    f"Null values not supported in V1."
                )
        # === END NULL CHECK ===
        
        # Convert Arrow → numpy dict
        numpy_dict = {}
        for col in table.column_names:
            arr = table.column(col).to_numpy()
            numpy_dict[col] = arr
        
        # Create RDataFrame from numpy dict
        rdf = ROOT.RDF.FromNumpy(numpy_dict)
        
        # Infer schema from Arrow types
        schema = {
            col: _arrow_type_to_ctype(table.schema.field(col).type)
            for col in table.column_names
        }
        
        # Create new DSLCompiler instance
        new_dsl = cls(schema)
        new_dsl._rdf = rdf
        
        # Apply schema from metadata if present
        if apply_schema and table.schema.metadata:
            schema_bytes = table.schema.metadata.get(b'dsl_schema')
            if schema_bytes:
                schema_dict = json.loads(schema_bytes.decode('utf-8'))
                new_dsl._apply_aliasdf_schema(schema_dict)
        
        return new_dsl
    
    def _apply_aliasdf_schema(self, schema_dict: dict):
        """
        Apply AliasDataFrame schema to DSL definitions.
        
        Converts Python expressions back to C++ syntax.
        
        Note: Best-effort conversion. Complex expressions may
        require manual review.
        """
        columns = schema_dict.get('columns', {})
        
        if columns:
            warnings.warn(
                "Round-trip expression conversion is best-effort. "
                "Complex boolean logic may require manual review.",
                stacklevel=2
            )
        
        for name, info in columns.items():
            expr = info.get('expr', name)
            # Convert Python operators back to C++
            cpp_expr = self._python_to_cpp_expr(expr)
            try:
                self.define(name, cpp_expr)
            except Exception as e:
                warnings.warn(
                    f"Could not apply definition '{name}': {e}",
                    stacklevel=2
                )
    
    def _python_to_cpp_expr(self, py_expr: str) -> str:
        """
        Convert Python operators to C++ (best-effort).
        
        Warning: Not guaranteed for complex expressions.
        Handles common cases from to_aliasdf() output.
        
        Conversions:
            ' & '  → ' && '
            ' | '  → ' || '
            ~var   → !var
            ~(     → !(
        """
        cpp_expr = py_expr
        
        # Logical operators
        cpp_expr = cpp_expr.replace(' & ', ' && ')
        cpp_expr = cpp_expr.replace(' | ', ' || ')
        
        # Unary NOT: ~var and ~(expr)
        cpp_expr = re.sub(r'~(\w+)', r'!\1', cpp_expr)
        cpp_expr = re.sub(r'~\(', '!(', cpp_expr)
        
        return cpp_expr

    # =========================================================================
    # Phase 13.5.B: C++ Function Registration API (v0.5)
    # =========================================================================
    
    def register_function_cpp(
        self,
        code: str,
        headers: Optional[List[str]] = None,
        pragmas: Optional[List[str]] = None,
        *,
        name: Optional[str] = None,
    ) -> 'DSLCompiler':
        """
        Register a C++ function for use in RDataFrame expressions.
        
        Phase 13.5.B v0.5 Implementation.
        
        Args:
            code: C++ function definition using natural syntax.
                  Example: "double pt(double px, double py) { return sqrt(px*px + py*py); }"
            headers: Optional list of headers. If None, auto-detects.
                     If provided, combines with default headers.
            pragmas: Optional list of RAW pragma lines for I/O operations.
                     Must be complete pragma statements.
                     Example: ["#pragma link C++ class MyStruct+;"]
            name: Override the function name parsed from code.
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If lambda expression detected (FROZEN RULE #1).
            ValueError: If code cannot be parsed.
        
        Example:
            >>> dsl.register_function_cpp('''
            ...     double pt(double px, double py) {
            ...         return sqrt(px*px + py*py);
            ...     }
            ... ''')
            >>> rdf = dsl.apply(rdf)
            >>> rdf = rdf.Define("track_pt", "pt(px, py)")
        
        FROZEN RULE #1: Lambda expressions are PROHIBITED.
        """
        # Check for lambda expressions (FROZEN RULE #1)
        self._check_lambda_cpp(code)
        
        # Parse function signature
        parsed = self._parse_cpp_function(code)
        if not parsed:
            raise ValueError(
                f"Could not parse function signature from:\n{code}\n"
                f"Expected format: return_type name(params) {{ body }}"
            )
        
        func_name, params, return_type, body = parsed
        
        # Override name if provided
        if name is not None:
            func_name = name
        
        # Determine headers
        if headers is not None:
            all_headers = set(REGISTER_DEFAULT_HEADERS) | set(headers)
        else:
            detected = self._detect_cpp_headers(body, return_type, params)
            all_headers = set(REGISTER_DEFAULT_HEADERS) | detected
        
        # Generate hash (v0.5: preserves parameter order)
        func_hash = self._generate_cpp_hash(params, return_type, body, all_headers)
        
        # Create C++ name: dsl_<n>_<hash16>
        cpp_name = f"dsl_{func_name}_{func_hash}"
        
        # Build full C++ code
        full_cpp = self._build_cpp_code(cpp_name, params, return_type, body, all_headers, pragmas)
        
        # Declare with ROOT if available
        declared = self._declare_cpp_function(cpp_name, full_cpp)
        
        # Create registered function
        func = RegisteredCppFunction(
            name=func_name,
            cpp_name=cpp_name,
            hash=func_hash,
            params=params,
            return_type=return_type,
            body=body,
            full_cpp=full_cpp,
            headers=all_headers,
            pragmas=set(pragmas or []),
            declared=declared,
        )
        
        # Register
        self._registered_cpp_functions[cpp_name] = func
        if func_name not in self._registered_cpp_by_name:
            self._registered_cpp_by_name[func_name] = []
        self._registered_cpp_by_name[func_name].append(cpp_name)
        
        return self
    
    def _check_lambda_cpp(self, code: str) -> None:
        """Check for lambda expressions (FROZEN RULE #1)."""
        for pattern in LAMBDA_PATTERNS:
            if re.search(pattern, code):
                raise ValueError(
                    "Lambda expressions are not supported (FROZEN RULE #1).\n"
                    "Use named functions instead:\n"
                    "  ❌ [](double x) { return x * 2; }\n"
                    "  ✅ double my_func(double x) { return x * 2; }"
                )
    
    def _parse_cpp_function(self, code: str) -> Optional[Tuple[str, List[Tuple[str, str]], str, str]]:
        """
        Parse C++ function: returns (name, params, return_type, body) or None.
        
        Handles complex types like const RVec<double>&, namespaces, etc.
        """
        code = code.strip()
        
        # Remove comments
        code_no_comments = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)
        
        # Pattern: return_type name(params) { body }
        pattern = r'([\w:<>&\s]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{(.+)\}'
        match = re.search(pattern, code_no_comments, re.DOTALL)
        
        if not match:
            return None
        
        return_type = ' '.join(match.group(1).split())
        name = match.group(2).strip()
        params_str = match.group(3).strip()
        body = match.group(4).strip()
        
        # Parse parameters (preserve order - v0.5 P0-1 fix)
        params: List[Tuple[str, str]] = []
        if params_str:
            param_pattern = r'([\w:<>&\s]+?)\s+(\w+)\s*(?:,|$)'
            for pmatch in re.finditer(param_pattern, params_str + ','):
                ptype = ' '.join(pmatch.group(1).split())
                pname = pmatch.group(2).strip()
                params.append((pname, ptype))
        
        return name, params, return_type, body
    
    def _generate_cpp_hash(
        self,
        params: List[Tuple[str, str]],
        return_type: str,
        body: str,
        headers: Set[str]
    ) -> str:
        """
        Generate deterministic hash for function.
        
        v0.5 FIXES:
        1. Parameter order PRESERVED (not sorted) — P0-1 fix
        2. Only whitespace normalization — P1-1 fix
        """
        # Normalize body (whitespace only)
        normalized_body = ' '.join(body.split())
        
        # v0.5 FIX: Preserve parameter order!
        # DO NOT sort params — f(int,double) != f(double,int)
        param_types = [self._canonicalize_cpp_type(t) for _, t in params]
        
        ret_type = self._canonicalize_cpp_type(return_type)
        
        # Build hash input (only headers are sorted)
        hash_input = "|".join([
            f"v{HASH_SCHEMA_VERSION}",
            normalized_body,
            ",".join(param_types),  # Order preserved!
            ret_type,
            ",".join(sorted(headers)),
        ])
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:HASH_LENGTH]
    
    def _canonicalize_cpp_type(self, type_str: str) -> str:
        """Normalize C++ type strings for consistent hashing."""
        type_str = ' '.join(type_str.split())
        
        # Expand RVec to full name
        if 'RVec<' in type_str and 'ROOT::VecOps::RVec' not in type_str:
            type_str = re.sub(r'\bRVec<', 'ROOT::VecOps::RVec<', type_str)
        
        # Normalize spacing
        type_str = re.sub(r'<\s+', '<', type_str)
        type_str = re.sub(r'\s+>', '>', type_str)
        type_str = re.sub(r'\s*&', '&', type_str)
        type_str = re.sub(r'\s*\*', '*', type_str)
        
        return type_str
    
    def _detect_cpp_headers(
        self,
        body: str,
        return_type: str,
        params: Optional[List[Tuple[str, str]]] = None
    ) -> Set[str]:
        """Auto-detect required headers from code."""
        headers: Set[str] = set()
        
        # Combine body, return type, and param types
        combined = body + " " + return_type
        if params:
            for _, ptype in params:
                combined += " " + ptype
        
        for keyword, header in REGISTER_HEADER_MAP.items():
            if keyword in combined:
                headers.add(header)
        
        return headers
    
    def _build_cpp_code(
        self,
        cpp_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        body: str,
        headers: Set[str],
        pragmas: Optional[List[str]]
    ) -> str:
        """Build complete C++ code for declaration."""
        lines = []
        
        # Headers
        for h in sorted(headers):
            lines.append(f"#include {h}")
        if lines:
            lines.append("")
        
        # Pragmas (raw lines)
        if pragmas:
            for p in pragmas:
                lines.append(p)
            lines.append("")
        
        # Function
        param_str = ", ".join(f"{ptype} {pname}" for pname, ptype in params)
        lines.append(f"{return_type} {cpp_name}({param_str}) {{")
        lines.append(f"    {body}")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _declare_cpp_function(self, cpp_name: str, cpp_code: str) -> bool:
        """
        Declare C++ code with ROOT (thread-safe).
        
        Uses class-level lock because ROOT's gInterpreter is global.
        v0.5 P0-3 fix: Class-level lock for thread safety.
        """
        # Quick check without lock
        if cpp_name in DSLCompiler._global_cpp_declared_names:
            return True
        
        with DSLCompiler._global_cpp_compile_lock:
            # Double-check after acquiring lock
            if cpp_name in DSLCompiler._global_cpp_declared_names:
                return True
            
            try:
                import ROOT
                result = ROOT.gInterpreter.Declare(cpp_code)
                if result:
                    DSLCompiler._global_cpp_declared_names.add(cpp_name)
                    # Update function record
                    if cpp_name in self._registered_cpp_functions:
                        self._registered_cpp_functions[cpp_name].declared = True
                return result
            except Exception:
                return False
    
    def get_registered_function(self, name: str) -> Optional[RegisteredCppFunction]:
        """Get most recent registered C++ function by name."""
        cpp_names = self._registered_cpp_by_name.get(name, [])
        if not cpp_names:
            return None
        return self._registered_cpp_functions.get(cpp_names[-1])
    
    def get_all_registered_functions(self, name: str) -> List[RegisteredCppFunction]:
        """Get all versions of a registered C++ function by name."""
        cpp_names = self._registered_cpp_by_name.get(name, [])
        return [self._registered_cpp_functions[cn] for cn in cpp_names 
                if cn in self._registered_cpp_functions]
    
    def list_registered_functions(self) -> List[str]:
        """List all registered C++ function names."""
        return list(self._registered_cpp_by_name.keys())
    
    def get_registered_cpp_name(self, name: str) -> Optional[str]:
        """Get internal C++ name for a registered function."""
        func = self.get_registered_function(name)
        return func.cpp_name if func else None
