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

from typing import Dict, List, Optional, Any
import uuid

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
    
    def compile_all(self) -> None:
        """
        Compile all defined functions to ROOT.
        
        Raises:
            IRError: If any compilation fails
        """
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
