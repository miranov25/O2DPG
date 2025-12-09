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
    
    # Apply to RDataFrame
    rdf = ROOT.RDataFrame("Events", "data.root")
    rdf = dsl.apply(rdf)
    
    # Export for inspection
    dsl.export_macro("my_functions.C")

Phase 7.9: RDataFrame Validation & C++ Export
"""

from typing import Dict, List, Optional, Any
import uuid

from .type_inferrer import TypeInferrer
from .ir_builder import IRBuilder
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


class DSLCompiler:
    """
    High-level DSL compiler for RDataFrame.
    
    This class provides a simple interface for:
    - Defining computed columns using Python DSL syntax
    - Applying all definitions to an RDataFrame
    - Exporting generated C++ for inspection
    
    Attributes:
        schema: Column definitions (name -> C++ type)
        library: FunctionLibrary containing generated functions
    
    Example:
        >>> dsl = DSLCompiler({"px": "double", "py": "double"})
        >>> dsl.define("pt", "sqrt(px**2 + py**2)")
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
        self.schema = schema
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
    
    def define(self, name: str, expression: str) -> 'DSLCompiler':
        """
        Define a new column from a DSL expression.
        
        Args:
            name: Output column name
            expression: DSL expression (e.g. "sqrt(px**2 + py**2)")
        
        Returns:
            self (for chaining)
        
        Raises:
            IRError: If expression is invalid or name conflicts
        
        Example:
            >>> dsl.define("pt", "sqrt(px**2 + py**2)")
            >>> dsl.define("n_tracks", "tracks.size()")
        """
        # Check for name collision with schema
        if name in self.schema:
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
        
        # Parse and generate
        builder = IRBuilder(self._inferrer)
        ir = builder.build(expression)
        
        # Use unique suffix to avoid collisions in parallel execution
        unique_name = f"{name}_{self._unique_id}"
        func = self._generator.generate(ir, unique_name)
        func.dsl_expression = expression  # Track original DSL
        func.column_name = name  # Track user-friendly name for Define()
        
        # Store
        self._definitions.append((name, expression))
        self._functions[name] = func
        self.library.add(func)
        
        return self
    
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
