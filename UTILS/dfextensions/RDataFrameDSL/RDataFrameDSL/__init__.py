"""
RDataFrameDSL - Python DSL for ROOT RDataFrame

This package provides a domain-specific language for constructing
ROOT RDataFrame analysis workflows with:

- C++ object navigation via Python syntax (track.getX())
- Automatic type inference from ROOT tree reflection
- N-key composite indices for calibration table joins
- NumPy-style slicing (1D and 2D)
- Two-phase validation (compile before RDF execution)

Basic Usage:
    from RDataFrameDSL import RDFBuilder
    
    builder = RDFBuilder.from_tree("data.root", "tree")
    builder.add_alias("pt", "sqrt(px**2 + py**2)")
    rdf, handle = builder.build()
    rdf.Histo1D("pt").Draw()

Phase 1 Exports (IR Core):
- Types: IRType, IRTypeKind
- Nodes: ConstantNode, VariableNode, BinaryOpNode, etc.
- Errors: IRError, IRErrorKind, ErrorCollector
"""

__version__ = "0.1.0"

# IR Types
from .ir_types import (
    IRTypeKind,
    IRType,
    promote_types,
    comparison_result_type,
    cpp_type_to_ir,
    CPP_TO_IR_TYPE,
    IR_TO_CPP_TYPE,
)

# IR Nodes
from .ir_nodes import (
    # Enums
    UnaryOp,
    BinaryOp,
    # Base
    IRNode,
    # Leaf nodes
    ConstantNode,
    VariableNode,
    # Operator nodes
    UnaryOpNode,
    BinaryOpNode,
    TernaryOpNode,
    # Function/method nodes
    CallNode,
    MethodCallNode,
    PropertyAccessNode,
    # Indexing nodes
    SliceNode,
    SubscriptNode,
    CollectionIndexNode,
    # Helpers
    BroadcastInfo,
    # Factory functions
    make_constant,
    make_variable,
    make_binary_op,
    make_unary_op,
    make_call,
    make_method_call,
    make_subscript,
)

# Class Reflection
from .reflection import (
    ReflectionCache,
    MethodInfo,
    PropertyInfo,
)

# IR Builder
from .ir_builder import (
    IRBuilder,
    BuildContext,
)

# Type Inference
from .type_inferrer import (
    TypeInferrer,
    VariableInfo,
    extract_inner_type,
    is_vector_type,
    is_rvec_type,
)

# IR Errors
from .ir_errors import (
    IRErrorKind,
    SourceLocation,
    IRError,
    ErrorRecoveryMode,
    ErrorCollector,
    # Helper functions
    type_mismatch_error,
    unknown_variable_error,
    method_not_found_error,
    property_not_found_error,
    missing_dictionary_error,
    rank_mismatch_error,
    unsupported_operation_error,
    compile_error,
)

# C++ Code Generation
from .backend_cpp import (
    CppCodeGenerator,
    GeneratedFunction,
    FunctionLibrary,
    FUNCTION_HEADERS,
    CLASS_HEADERS,
    RVEC_HEADER,
    RVEC_METHODS,
)

__all__ = [
    # Version
    '__version__',
    
    # Types
    'IRTypeKind',
    'IRType',
    'promote_types',
    'comparison_result_type',
    'cpp_type_to_ir',
    'CPP_TO_IR_TYPE',
    'IR_TO_CPP_TYPE',
    
    # Type Inference
    'TypeInferrer',
    'VariableInfo',
    'extract_inner_type',
    'is_vector_type',
    'is_rvec_type',
    
    # IR Builder
    'IRBuilder',
    'BuildContext',
    
    # Class Reflection
    'ReflectionCache',
    'MethodInfo',
    'PropertyInfo',
    
    # Nodes
    'UnaryOp',
    'BinaryOp',
    'IRNode',
    'ConstantNode',
    'VariableNode',
    'UnaryOpNode',
    'BinaryOpNode',
    'TernaryOpNode',
    'CallNode',
    'MethodCallNode',
    'PropertyAccessNode',
    'SliceNode',
    'SubscriptNode',
    'CollectionIndexNode',
    'BroadcastInfo',
    'make_constant',
    'make_variable',
    'make_binary_op',
    'make_unary_op',
    'make_call',
    'make_method_call',
    'make_subscript',
    
    # Errors
    'IRErrorKind',
    'SourceLocation',
    'IRError',
    'ErrorRecoveryMode',
    'ErrorCollector',
    'type_mismatch_error',
    'unknown_variable_error',
    'method_not_found_error',
    'property_not_found_error',
    'missing_dictionary_error',
    'rank_mismatch_error',
    'unsupported_operation_error',
    'compile_error',
    
    # C++ Code Generation
    'CppCodeGenerator',
    'GeneratedFunction',
    'FunctionLibrary',
    'FUNCTION_HEADERS',
    'CLASS_HEADERS',
    'RVEC_HEADER',
    'RVEC_METHODS',
]
