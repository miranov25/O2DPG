# Architecture Overview

## Dataflow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                       │
│  dsl = DSLCompiler({"pt": "RVec<double>", "tracks": "RVec<TLorentzVector>"})│
│  dsl.define("high_pt", "pt[pt > 1.0]")                                      │
│  dsl.define("track_pts", "tracks.Pt()")                                     │
│  rdf = dsl.apply(rdf)                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. SCHEMA PARSING (type_inferrer.py)                                        │
│     {"pt": "RVec<double>"} → TypeInferrer with column metadata               │
│     - Determines rank (0=scalar, 1=vector)                                   │
│     - Extracts element types                                                 │
│     - Stores C++ type strings                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. EXPRESSION PARSING (ir_builder.py)                                       │
│     "pt[pt > 1.0]" → Python AST → IR Tree                                   │
│     - Uses Python's ast.parse()                                              │
│     - Walks AST, builds IRNode tree                                          │
│     - Consults TypeInferrer for variable types                               │
│     - Determines operation types (slice, broadcast, etc.)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. IR TREE (ir_nodes.py)                                                    │
│                                                                              │
│     RVecSliceNode(                                                           │
│         target=VariableNode("pt"),                                           │
│         slice_kind=SliceKind.BOOLEAN,                                        │
│         mask=BinaryOpNode(                                                   │
│             op=BinaryOp.GT,                                                  │
│             left=VariableNode("pt"),                                         │
│             right=ConstantNode(1.0)                                          │
│         )                                                                    │
│     )                                                                        │
│                                                                              │
│     Each node has:                                                           │
│     - dtype: IRType (kind, element_type, etc.)                               │
│     - rank: 0 (scalar) or 1 (vector)                                         │
│     - walk() method for traversal                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. CODE GENERATION (backend_cpp.py)                                         │
│     IR Tree → GeneratedFunction                                              │
│                                                                              │
│     CppCodeGenerator._visit_*() methods:                                     │
│     - _visit_constant, _visit_variable                                       │
│     - _visit_binary_op, _visit_unary_op                                      │
│     - _visit_call (math functions)                                           │
│     - _visit_method_call (object methods)                                    │
│     - _visit_rvec_slice (7 slice kinds)                                      │
│     - _visit_method_broadcast (Phase 8)                                      │
│                                                                              │
│     Output:                                                                  │
│     GeneratedFunction(                                                       │
│         name="alias_high_pt_abc123",                                         │
│         code="ROOT::RVec<double> alias_high_pt_abc123(...) {...}",           │
│         inputs=[("pt", "const ROOT::RVec<double>&")],                        │
│         return_type="ROOT::RVec<double>",                                    │
│         headers={"<ROOT/RVec.hxx>"},                                         │
│         dsl_expression="pt[pt > 1.0]"                                        │
│     )                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. COMPILATION (FunctionLibrary in backend_cpp.py)                          │
│     GeneratedFunction → ROOT gInterpreter                                    │
│                                                                              │
│     ROOT.gInterpreter.Declare(func.code)                                     │
│     - JIT compiles C++ to machine code                                       │
│     - Function becomes callable from RDataFrame                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. RDATAFRAME INTEGRATION (dsl_compiler.py)                                 │
│     DSLCompiler.apply(rdf)                                                   │
│                                                                              │
│     for name, expr in definitions:                                           │
│         rdf = rdf.Define(name, func.get_call_expression())                   │
│                                                                              │
│     Example:                                                                 │
│     rdf.Define("high_pt", "alias_high_pt_abc123(pt)")                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. EXECUTION (ROOT RDataFrame)                                              │
│     - Lazy evaluation per event                                              │
│     - Multi-threaded with EnableImplicitMT()                                 │
│     - Event loop calls compiled functions                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### ir_nodes.py (~400 lines)
**Purpose:** Define all IR node types

| Node Type | Purpose | Example DSL |
|-----------|---------|-------------|
| `ConstantNode` | Literal values | `3.14`, `True` |
| `VariableNode` | Column references | `px`, `pt` |
| `BinaryOpNode` | Binary operations | `px + py`, `pt > 1.0` |
| `UnaryOpNode` | Unary operations | `-x`, `not flag` |
| `CallNode` | Function calls | `sqrt(x)`, `abs(y)` |
| `TernaryOpNode` | Conditionals | `x if cond else y` |
| `MethodCallNode` | Object methods | `particle.Pt()` |
| `PropertyAccessNode` | Object properties | `vec.fX` |
| `SubscriptNode` | Indexing | `pt[0]`, `pt[i]` |
| `RVecSliceNode` | Slicing | `pt[:3]`, `pt[mask]` |
| `MethodBroadcastNode` | Element-wise methods | `tracks.Pt()` *(Phase 8)* |

### ir_types.py (~150 lines)
**Purpose:** Type system for IR

- `IRType`: Kind (Scalar, RVec, Object), element type, C++ mapping
- `IRTypeKind`: Enum of type kinds
- `IR_TO_CPP_TYPE`: Maps Python types to C++ types

### ir_builder.py (~600 lines)
**Purpose:** Parse expressions and build IR trees

Key methods:
- `build(expression: str) -> IRNode`: Main entry point
- `_visit_*()`: AST visitor methods for each Python AST node type
- `_classify_slice()`: Determine SliceKind from slice parameters

### type_inferrer.py (~300 lines)
**Purpose:** Infer types from schema and ROOT reflection

- `TypeInferrer.from_schema()`: Build from simple dict
- `TypeInferrer.from_tree()`: Build from TTree *(planned)*
- `get_type()`: Look up column type by name
- `_infer_method_return_type()`: Use TClass reflection

### backend_cpp.py (~1500 lines)
**Purpose:** Generate C++ code from IR

Key classes:
- `CppCodeGenerator`: IR → C++ code string
- `GeneratedFunction`: Holds generated code + metadata
- `FunctionLibrary`: Manages compiled functions

Key methods:
- `generate(ir, name) -> GeneratedFunction`
- `_visit_*()`: Visitor for each IR node type
- `_generate_reflection_access()`: Private member access via TClass

### dsl_compiler.py (~250 lines)
**Purpose:** High-level user API

```python
class DSLCompiler:
    def __init__(self, schema: Dict[str, str])
    def define(self, name: str, expression: str) -> 'DSLCompiler'
    def apply(self, rdf) -> RDataFrame
    def preview(self) -> str
    def export_macro(self, filepath: str)
```

---

## Key Design Decisions

### 1. Safe by Default
- Out-of-bounds indexing returns `NaN`, not crash
- Empty vector slicing returns empty vector
- Size clamping on all slice operations

### 2. Python-like Semantics
- `pt[-1]` = last element (Python style)
- `pt[:3]` = first 3 elements (Python style)
- `pt[::2]` = every other element (Python style)

### 3. Reflection for Private Members
- Uses ROOT TClass API to access protected/private members
- Same behavior as legacy TTree::Draw
- Thread-safe via C++11 magic statics

### 4. UUID Function Names
- Each DSLCompiler generates unique function names
- Prevents gInterpreter collisions in parallel tests
- Column names remain user-friendly

### 5. Lazy Compilation
- Functions compiled only when `apply()` or `compile_all()` called
- Allows validation before ROOT interaction

---

## Test Architecture

```
tests/
├── Unit Tests (no ROOT required)
│   ├── test_ir_core.py          # IR node creation
│   ├── test_ir_builder.py       # Expression parsing
│   ├── test_type_inference.py   # Type system
│   └── test_backend_cpp.py      # Code generation strings
│
├── Mock ROOT Tests
│   ├── test_backend_cpp_rvec.py    # RVec code patterns
│   └── test_backend_cpp_objects.py # Object access patterns
│
└── Integration Tests (ROOT required)
    ├── test_root_integration.py              # ROOT behavior validation
    ├── test_rdataframe_integration_advanced.py # Full pipeline tests
    └── test_backend_cpp_reflection.py        # TClass reflection
```

### Test Categories by Phase

| Phase | Test Files | Focus |
|-------|------------|-------|
| 1-4 | `test_ir_*.py` | IR, types, errors |
| 5 | `test_backend_cpp.py` | Scalar code generation |
| 6a | `test_backend_cpp_objects.py` | Object methods/properties |
| 6b | `test_backend_cpp_rvec.py` | RVec operations |
| 6c | `test_backend_cpp_reflection.py` | Private member access |
| 6.9 | `test_root_integration.py` | ROOT behavior assumptions |
| 7 | `test_backend_cpp_rvec.py` | Slice code generation |
| 7.9 | `test_rdataframe_integration_advanced.py` | Multi-function pipelines |
| 8 | `test_backend_cpp_broadcast.py` *(new)* | Method broadcasting |

---

## Error Handling Philosophy

All errors are `IRError` with:
- `kind`: Error category (TYPE_ERROR, UNSUPPORTED_OP, etc.)
- `message`: Human-readable description
- `suggestions`: List of helpful hints

Example:
```
IRError: Method 'NonExistent' not found on element type 'TLorentzVector'
Suggestions:
  - Did you mean: Pt, Eta, Phi, M, Px, Py, Pz?
  - Note: Broadcasting 'tracks.NonExistent()' on RVec<TLorentzVector>
```

This replaces C++ template error messages that can be hundreds of lines.
