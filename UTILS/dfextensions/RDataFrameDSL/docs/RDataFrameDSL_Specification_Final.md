# RDataFrame DSL – Complete Specification

**Version:** v1.0 Final  
**Date:** 2025-01-XX  
**Status:** Approved for Implementation  
**Reviewers:** GPT (Approved), Gemini (Approved), Owner (Approved)

---

# Part I: Overview

## 1.1 Project Summary

The **RDataFrame DSL** is a Python-based Domain Specific Language for constructing ROOT RDataFrame analysis workflows. It enables:

- **C++ object navigation** via Python syntax (`track.getX()`, `collision[idx].getZ()`)
- **Automatic type inference** from ROOT tree reflection
- **N-key composite indices** for subframe/calibration table joins
- **Full NumPy-style slicing** (1D and 2D)
- **Two-phase validation** (compile before RDF integration)
- **Prebuilt helper function library** for stability and reuse

## 1.2 Architecture

**Standalone `RDFBuilder` class** - no dependency on AliasDataFrame.py initially.

```
┌─────────────────────────────────────────────────────────────┐
│                      USER CODE                              │
│  builder = RDFBuilder.from_tree("data.root", "tree")       │
│  builder.add_alias("pt", "sqrt(px**2 + py**2)")            │
│  rdf, handle = builder.build()                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RDFBuilder                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ TypeInferrer│  │  IRBuilder  │  │ CppBackend  │         │
│  │ (reflection)│  │ (AST → IR)  │  │ (codegen)   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                    ┌───────────┐                            │
│                    │ IR Tree   │                            │
│                    └─────┬─────┘                            │
│                          │                                  │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Python Backend│  │ C++ Backend │  │FunctionLib  │         │
│  │  (testing)   │  │ (helpers)   │  │ (compiled)  │         │
│  └─────────────┘  └──────┬──────┘  └─────────────┘         │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROOT RDataFrame                          │
│  rdf.Define("pt", "helper_pt(px, py)")                     │
│  rdf.Filter("pt > 1.0").Histo1D("pt")                      │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 File Structure

```
RDataFrameDSL/
├── RDataFrameDSL/              # Package
│   ├── __init__.py
│   ├── ir_nodes.py             # IR node classes
│   ├── ir_types.py             # Type system
│   ├── ir_errors.py            # Error handling
│   ├── ir_builder.py           # AST → IR conversion
│   ├── type_inferrer.py        # ROOT reflection for types
│   ├── reflection.py           # Method/property resolution
│   ├── backend_python.py       # Python/numpy evaluation
│   ├── backend_cpp.py          # C++ code generation
│   ├── function_library.py     # Compiled function management
│   └── rdf_builder.py          # Main user-facing class
├── tests/
│   ├── test_ir_core.py         # IR node tests
│   ├── test_type_inference.py  # Reflection tests
│   ├── test_cpp_backend.py     # Code generation tests
│   ├── test_shuffled.py        # Test E: shuffled correctness
│   ├── test_multisubframe.py   # Test F: multiple subframes
│   └── test_missing_keys.py    # Test G: missing key behavior
├── examples/
│   ├── simple_arithmetic.py
│   ├── object_navigation.py
│   └── calibration_workflow.py
├── docs/
│   └── specification.md
├── README.md
├── setup.py
└── pyproject.toml
```

---

# Part II: User Interface

## 2.1 Basic Usage

```python
from RDataFrameDSL import RDFBuilder

# Load tree - types auto-inferred from ROOT reflection
builder = RDFBuilder.from_tree("data.root", "tree")

# Add aliases - types inferred from expressions
builder.add_alias("pt", "sqrt(px**2 + py**2)")
builder.add_alias("eta", "-log(tan(theta/2))")
builder.add_alias("high_pt", "pt > 5.0")  # bool inferred

# Build RDataFrame
rdf, handle = builder.build()

# Use it
rdf.Filter("high_pt").Histo1D("pt").Draw()
```

## 2.2 Object Navigation

```python
from RDataFrameDSL import RDFBuilder

builder = RDFBuilder.from_tree("data.root", "tracks")

# Method calls on C++ objects
builder.add_alias("track_x", "track.getX()")
builder.add_alias("track_pt", "track.getPt()")

# Property access
builder.add_alias("track_px", "track.mPx")

# Cross-collection indexing
builder.add_alias("coll_x", "collision[track.GetCollisionIndex()].getX()")
builder.add_alias("dx", "track_x - coll_x")

rdf, handle = builder.build()
```

## 2.3 Subframes (Calibration Tables)

```python
from RDataFrameDSL import RDFBuilder

# Main data
builder = RDFBuilder.from_tree("residuals.root", "tree")

# Add calibration subframe with N-key composite index
builder.add_subframe("calib", "calibration.root", "calib_tree",
                     index=["row", "sector", "firstTFOrbit"])

# Use subframe columns
builder.add_alias("dy_corrected", "dy - calib.offset")
builder.add_alias("valid", "calib.quality == 1")

rdf, handle = builder.build()
```

## 2.4 Slicing (Full NumPy-style)

```python
# 1D slicing
builder.add_alias("first_three", "tracks[:3]")
builder.add_alias("last", "tracks[-1]")
builder.add_alias("every_other", "tracks[::2]")

# 2D slicing (nested RVec)
builder.add_alias("first_event_tracks", "events[0, :]")
builder.add_alias("first_track_each", "events[:, 0]")

# Fancy indexing
builder.add_alias("selected", "tracks[indices]")
builder.add_alias("passing", "tracks[quality_mask]")
```

## 2.5 RDFBuilder API

```python
class RDFBuilder:
    """Standalone RDataFrame builder with DSL support."""
    
    @classmethod
    def from_tree(cls, filename: str, treename: str,
                  schema: dict = None) -> 'RDFBuilder':
        """
        Create builder from ROOT tree.
        Types are auto-inferred. Schema is optional override.
        """
    
    def add_alias(self, name: str, expr: str) -> 'RDFBuilder':
        """Add alias with automatic type inference."""
    
    def add_subframe(self, name: str,
                     source: Union[str, 'RDFBuilder'],
                     treename: str = None,
                     index: Union[str, List[str]] = None) -> 'RDFBuilder':
        """Add subframe with composite index support."""
    
    def build(self,
              validate: bool = True,
              safe_indexing: bool = True,
              error_mode: str = 'fail_chain') -> Tuple[ROOT.RDataFrame, Any]:
        """Build RDataFrame with all aliases defined."""
    
    def validate(self) -> bool:
        """Validate all expressions without building RDF."""
    
    def show_types(self) -> None:
        """Print inferred types for debugging."""
    
    def save_library(self, path: str) -> None:
        """Save generated C++ to .C or .so file."""
```

## 2.6 Override Behavior

Later actions override earlier definitions:
```python
# Schema provides initial types
builder = RDFBuilder.from_tree("data.root", "tree", schema={
    "columns": {"x": {"dtype": "float32"}}
})

# add_alias() can override schema
builder.add_alias("x", "some_expr")  # Overrides previous definition
```

---

# Part III: IR Specification

## 3.1 Type System

```python
class IRTypeKind(Enum):
    Float32 = "float"
    Float64 = "double"
    Int32 = "int"
    Int64 = "long long"
    UInt32 = "unsigned int"
    UInt64 = "unsigned long"
    Bool = "bool"
    Object = "object"      # C++ class type
    Unknown = "unknown"

@dataclass
class IRType:
    kind: IRTypeKind
    cpp_type: Optional[str] = None  # Full C++ type for Object kind

@dataclass
class IRNode:
    """Base class for all IR nodes."""
    dtype: IRType
    rank: int                       # 0=scalar, 1=RVec, 2=RVec<RVec>
    is_jagged: bool
    source_location: Optional[SourceLocation]
    metadata: Dict[str, Any]
    
    def children(self) -> List['IRNode']: ...
    def walk(self): ...
```

## 3.2 Node Types

### Leaf Nodes
```python
@dataclass
class ConstantNode(IRNode):
    value: Union[int, float, bool]

@dataclass
class VariableNode(IRNode):
    name: str
    namespace: Optional[str] = None     # Subframe name
    is_alias: bool = False
    cpp_type: Optional[str] = None
```

### Operator Nodes
```python
class UnaryOp(Enum):
    NEG = "-"
    NOT = "not"
    BITNOT = "~"

class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOORDIV = "//"
    MOD = "%"
    POW = "**"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    AND = "and"
    OR = "or"
    BITAND = "&"
    BITOR = "|"
    BITXOR = "^"

@dataclass
class UnaryOpNode(IRNode):
    op: UnaryOp
    operand: IRNode

@dataclass
class BinaryOpNode(IRNode):
    op: BinaryOp
    left: IRNode
    right: IRNode
    broadcast_info: Optional[BroadcastInfo] = None

@dataclass
class TernaryOpNode(IRNode):
    condition: IRNode
    if_true: IRNode
    if_false: IRNode
```

### Function and Method Nodes
```python
@dataclass
class CallNode(IRNode):
    """Global function: sqrt(x), TMath::Gaus(x, m, s)"""
    func: str
    args: List[IRNode]
    namespace: Optional[str] = None
    cpp_name: Optional[str] = None
    headers: List[str] = field(default_factory=list)

@dataclass
class MethodCallNode(IRNode):
    """Method on C++ object: track.getX()"""
    object: IRNode
    method_name: str
    args: List[IRNode] = field(default_factory=list)
    class_name: Optional[str] = None
    return_type: Optional[str] = None
    signature: Optional[str] = None

@dataclass
class PropertyAccessNode(IRNode):
    """Property access: track.mPx"""
    object: IRNode
    property_name: str
    class_name: Optional[str] = None
    property_type: Optional[str] = None
```

### Indexing Nodes
```python
@dataclass
class SliceNode(IRNode):
    """Python slice: start:stop:step"""
    start: Optional[IRNode] = None
    stop: Optional[IRNode] = None
    step: Optional[IRNode] = None

@dataclass
class SubscriptNode(IRNode):
    """Indexing/slicing: arr[i], arr[1:10], arr[:, 0]"""
    value: IRNode
    indices: List[Union[IRNode, SliceNode]]
    is_fancy: bool = False
    is_boolean_mask: bool = False

@dataclass
class CollectionIndexNode(IRNode):
    """Cross-collection: collision[track.GetCollisionIndex()]"""
    collection: IRNode
    index_expr: IRNode
    safe_mode: bool = True
    default_value: Optional[Any] = None
```

## 3.3 Type Inference Rules

### Priority Order
1. **Tree branch type** (TLeaf, TBranch, TClass)
2. **Subframe column type** (friend tree reflection)
3. **Expression inference** (operation result types)
4. **User schema** (optional override)

### Type Promotion
| Operation | Result Type |
|-----------|-------------|
| Float + Int | Float |
| Float + Float | Float (wider) |
| Int + Int | Int (wider) |
| Comparison | Bool |
| `obj.method()` | From reflection |

### Rank Inference
| Operation | Result Rank |
|-----------|-------------|
| Scalar op Scalar | 0 |
| Scalar op Rank-k | k (broadcast) |
| Rank-k op Rank-k | k (element-wise) |
| `arr[i]` (scalar index) | rank - 1 |
| `arr[:]` (slice) | rank (preserved) |
| `RVec<T>.method()` → scalar | rank (mapped) |

### Jaggedness Propagation
- Collections of collections: `is_jagged = True`
- Method returning collection: `is_jagged = True` (default)
- Operations on jagged: Result is jagged

---

# Part IV: Type Inference from ROOT

## 4.1 TypeInferrer Class

```python
class TypeInferrer:
    """Infers types from ROOT tree reflection."""
    
    def __init__(self, tree: ROOT.TTree):
        self.tree = tree
        self._branch_types: Dict[str, IRType] = {}
        self._branch_ranks: Dict[str, int] = {}
        self._scan_tree()
    
    def _scan_tree(self):
        """Extract all branch types from tree."""
        for branch in self.tree.GetListOfBranches():
            name = branch.GetName()
            
            # Try object class
            class_name = branch.GetClassName()
            if class_name:
                self._handle_object_branch(name, class_name)
                continue
            
            # Try leaf type
            leaf = branch.GetLeaf(name)
            if leaf:
                self._handle_leaf_branch(name, leaf)
    
    def _handle_object_branch(self, name: str, class_name: str):
        """Handle branch with C++ class."""
        # Check if it's a vector type
        if class_name.startswith("vector<") or "RVec<" in class_name:
            inner_type = self._extract_inner_type(class_name)
            self._branch_types[name] = IRType(IRTypeKind.Object, inner_type)
            self._branch_ranks[name] = 1
        else:
            self._branch_types[name] = IRType(IRTypeKind.Object, class_name)
            self._branch_ranks[name] = 0
    
    def _handle_leaf_branch(self, name: str, leaf: ROOT.TLeaf):
        """Handle primitive leaf branch."""
        type_name = leaf.GetTypeName()
        ir_type = self._cpp_to_ir_type(type_name)
        
        # Check for array
        leaf_count = leaf.GetLeafCount()
        if leaf_count:
            # Variable-length array (jagged)
            self._branch_ranks[name] = 1
            self._branch_jagged[name] = True
        elif leaf.GetLen() > 1:
            # Fixed-length array
            self._branch_ranks[name] = 1
            self._branch_jagged[name] = False
        else:
            self._branch_ranks[name] = 0
        
        self._branch_types[name] = ir_type
    
    def get_variable_type(self, name: str) -> Tuple[IRType, int, bool]:
        """Get (type, rank, is_jagged) for variable."""
        if name not in self._branch_types:
            raise IRError(IRErrorKind.TYPE_ERROR,
                          f"Unknown variable '{name}' - not found in tree")
        return (self._branch_types[name],
                self._branch_ranks.get(name, 0),
                self._branch_jagged.get(name, False))
```

## 4.2 Known Edge Cases (from Reviewers)

### Missing Dictionaries
```python
def _handle_object_branch(self, name: str, class_name: str):
    # Check if dictionary is loaded
    tclass = ROOT.TClass.GetClass(class_name)
    if tclass is None:
        raise IRError(
            IRErrorKind.MISSING_DICT,
            f"Class '{class_name}' not found. Dictionary may not be loaded.",
            suggestions=[
                f"Load dictionary: ROOT.gInterpreter.ProcessLine('.L dict.C+')",
                f"Or provide schema override with cpp_type"
            ]
        )
```

### std::vector vs RVec
```python
def _extract_inner_type(self, class_name: str) -> str:
    """Map std::vector<T> to RVec-compatible type."""
    # Both std::vector<T> and RVec<T> become Rank=1 in IR
    if class_name.startswith("vector<"):
        inner = class_name[7:-1]  # Extract T from vector<T>
        return inner
    if "RVec<" in class_name:
        # Handle ROOT::VecOps::RVec<T>
        start = class_name.find("RVec<") + 5
        end = class_name.rfind(">")
        return class_name[start:end]
    return class_name
```

### Split Branches
```python
# OUT OF SCOPE for v1.0
# If user accesses track.getX() but tree has split branches
# (track.fX, track.fY), raise clear error:
raise IRError(
    IRErrorKind.UNSUPPORTED_OP,
    f"Branch '{name}' appears to be split. "
    f"Split object branches are not supported in v1.0.",
    suggestions=["Access individual leaves directly: track_fX, track_fY"]
)
```

---

# Part V: Reflection Layer

## 5.1 Method Resolution

```python
class ReflectionCache:
    """Caches C++ reflection results."""
    
    def __init__(self):
        self._method_cache: Dict[tuple, MethodInfo] = {}
        self._property_cache: Dict[tuple, PropertyInfo] = {}
    
    def resolve_method(self, class_name: str, method_name: str,
                       arg_types: List[str] = None) -> MethodInfo:
        """Resolve method using ROOT TClass."""
        key = (class_name, method_name, tuple(arg_types or []))
        if key in self._method_cache:
            return self._method_cache[key]
        
        tclass = ROOT.TClass.GetClass(class_name)
        if not tclass:
            raise IRError(IRErrorKind.REFLECTION_ERROR,
                          f"Class '{class_name}' not found")
        
        # Find method
        method = tclass.GetMethod(method_name)
        if not method:
            # Try fuzzy matching for suggestions
            suggestions = self._fuzzy_match_methods(tclass, method_name)
            raise IRError(
                IRErrorKind.REFLECTION_ERROR,
                f"Method '{method_name}' not found in {class_name}",
                suggestions=suggestions
            )
        
        info = MethodInfo(
            class_name=class_name,
            method_name=method_name,
            return_type=method.GetReturnTypeName(),
            signature=str(method.GetSignature()),
            is_const=method.Property() & ROOT.kIsConstMethod
        )
        self._method_cache[key] = info
        return info
    
    def _fuzzy_match_methods(self, tclass, target: str) -> List[str]:
        """Find similar method names for suggestions."""
        suggestions = []
        target_lower = target.lower()
        
        methods = tclass.GetListOfMethods()
        for method in methods:
            name = method.GetName()
            if target_lower in name.lower() or name.lower() in target_lower:
                suggestions.append(f"Did you mean '{name}'?")
        
        return suggestions[:3]  # Limit to 3 suggestions
```

## 5.2 Property Resolution

```python
def resolve_property(self, class_name: str, prop_name: str) -> PropertyInfo:
    """Resolve data member using ROOT TClass."""
    key = (class_name, prop_name)
    if key in self._property_cache:
        return self._property_cache[key]
    
    tclass = ROOT.TClass.GetClass(class_name)
    if not tclass:
        raise IRError(IRErrorKind.REFLECTION_ERROR,
                      f"Class '{class_name}' not found")
    
    member = tclass.GetDataMember(prop_name)
    if not member:
        raise IRError(IRErrorKind.REFLECTION_ERROR,
                      f"Property '{prop_name}' not found in {class_name}")
    
    info = PropertyInfo(
        class_name=class_name,
        property_name=prop_name,
        property_type=member.GetTypeName(),
        offset=member.GetOffset()
    )
    self._property_cache[key] = info
    return info
```

---

# Part VI: Error Handling

## 6.1 Error Types

```python
class IRErrorKind(Enum):
    PARSE_ERROR = "parse"
    TYPE_ERROR = "type"
    RANK_ERROR = "rank"
    REFLECTION_ERROR = "reflect"
    UNSUPPORTED_OP = "unsupported"
    VALIDATION_ERROR = "validation"
    COMPILE_ERROR = "compile"
    CYCLE_ERROR = "cycle"
    MISSING_DICT = "dictionary"

@dataclass
class SourceLocation:
    expr_name: str
    text_span: Tuple[int, int]
    line: Optional[int] = None
    column: Optional[int] = None
    snippet: str = ""

@dataclass
class IRError(Exception):
    kind: IRErrorKind
    message: str
    source_location: Optional[SourceLocation] = None
    underlying: Optional[Exception] = None
    suggestions: List[str] = field(default_factory=list)
```

## 6.2 Error Message Format (from Gemini)

```python
def format_error(self) -> str:
    """Format error with full context."""
    lines = [f"IRError: {self.message}"]
    
    if self.source_location:
        lines.append(f"\n  Expression: \"{self.source_location.snippet}\"")
        lines.append(f"  In alias: '{self.source_location.expr_name}'")
    
    # Include inferred type context if available
    if 'inferred_type' in self.metadata:
        lines.append(f"\n  Context:")
        lines.append(f"    • Variable was inferred as: {self.metadata['inferred_type']}")
        lines.append(f"    • Source: {self.metadata.get('type_source', 'unknown')}")
    
    if self.suggestions:
        lines.append(f"\n  Suggestions:")
        for s in self.suggestions:
            lines.append(f"    • {s}")
    
    return "\n".join(lines)
```

## 6.3 Error Recovery Modes

```python
class ErrorRecoveryMode(Enum):
    FAIL_ALL = "fail_all"           # Stop on first error
    SKIP_CONTINUE = "skip_continue" # Skip failed, continue others
    FAIL_CHAIN = "fail_chain"       # Fail dependency chain, continue independent

class ErrorCollector:
    def __init__(self, mode: ErrorRecoveryMode = ErrorRecoveryMode.FAIL_CHAIN):
        self.mode = mode
        self.errors: List[IRError] = []
        self.failed_aliases: Set[str] = set()
    
    def should_process(self, alias: str, dependencies: Set[str]) -> bool:
        if self.mode == ErrorRecoveryMode.FAIL_ALL:
            return len(self.errors) == 0
        elif self.mode == ErrorRecoveryMode.SKIP_CONTINUE:
            return alias not in self.failed_aliases
        elif self.mode == ErrorRecoveryMode.FAIL_CHAIN:
            return not (dependencies & self.failed_aliases)
        return True
```

---

# Part VII: C++ Backend

## 7.1 Helper Function Generation

**All expressions generate helper functions** (not inline strings).

```python
class CppBackend:
    def __init__(self, reflection: ReflectionCache):
        self.reflection = reflection
        self.generated: Dict[str, str] = {}
        self.headers: Set[str] = set()
    
    def generate_helper(self, ir: IRNode, name: str) -> str:
        """Generate C++ helper function."""
        inputs = self._collect_inputs(ir)
        return_type = self._ir_to_cpp_type(ir.dtype, ir.rank)
        params = self._generate_params(inputs)
        body = self._generate_body(ir)
        
        func = f"""
{return_type} alias_{name}({params})
{{
{body}
}}
"""
        self.generated[name] = func
        return func
```

## 7.2 Cross-Collection Indexing (Safe Mode)

```cpp
// Generated for: collision[track.GetCollisionIndex()].getX()
// With safe_mode=True (DEFAULT)

float alias_dx(
    const ROOT::VecOps::RVec<Collision>& collision,
    const Track& track)
{
    auto idx = track.GetCollisionIndex();
    if (idx >= 0 && static_cast<size_t>(idx) < collision.size()) {
        return collision[idx].getX();
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
}
```

## 7.3 Vectorized Expressions (Rank 1)

```cpp
// Generated for: track[:].getX()

ROOT::VecOps::RVec<float> alias_track_x(
    const ROOT::VecOps::RVec<Track>& track)
{
    auto n = track.size();
    ROOT::VecOps::RVec<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = track[i].getX();
    }
    return out;
}
```

## 7.4 2D Slicing (CRITICAL: Use Loops, Not VecOps)

From Gemini's review: **Must use loops with bounds checking for jagged safety.**

```cpp
// Generated for: events[:, 0]  (first track of each event)
// UNSAFE if any event has 0 tracks - must check!

ROOT::VecOps::RVec<float> alias_first_track(
    const ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>& events)
{
    ROOT::VecOps::RVec<float> out(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        // SAFETY CHECK for jagged data
        if (events[i].size() > 0) {
            out[i] = events[i][0];
        } else {
            out[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return out;
}
```

## 7.5 Function Library

```python
class FunctionLibrary:
    """Manages compiled C++ helper functions."""
    
    def __init__(self):
        self.functions: Dict[str, str] = {}
        self.headers_loaded: Set[str] = set()
        self.compiled: Set[str] = set()
    
    def compile(self, name: str, code: str, headers: List[str]):
        """Compile function via gInterpreter.Declare()."""
        # Load headers once
        for h in headers:
            if h not in self.headers_loaded:
                ROOT.gInterpreter.ProcessLine(f'#include "{h}"')
                self.headers_loaded.add(h)
        
        # Compile
        if not ROOT.gInterpreter.Declare(code):
            raise IRError(IRErrorKind.COMPILE_ERROR,
                          f"Failed to compile helper function '{name}'")
        
        self.functions[name] = code
        self.compiled.add(name)
    
    def save_macro(self, path: str):
        """Save all functions to .C file."""
        with open(path, 'w') as f:
            for h in sorted(self.headers_loaded):
                f.write(f'#include "{h}"\n')
            f.write('\n')
            for code in self.functions.values():
                f.write(code)
                f.write('\n')
```

## 7.6 Function Naming

Use deterministic naming to avoid collisions:
```python
def _generate_function_name(self, alias_name: str, expr: str) -> str:
    """Generate unique function name."""
    # Simple case: just use alias name
    if self._is_unique(alias_name):
        return f"alias_{alias_name}"
    
    # Collision: add hash
    expr_hash = hashlib.md5(expr.encode()).hexdigest()[:8]
    return f"alias_{alias_name}_{expr_hash}"
```

---

# Part VIII: Two-Phase Execution

## 8.1 Execution Flow

```
Phase 1: Validation & Compilation (OUTSIDE RDataFrame)
═══════════════════════════════════════════════════════
    Parse expression → IR
           │
           ▼
    Type inference (from tree reflection)
           │
           ▼
    Generate C++ helper function
           │
           ▼
    gInterpreter.Declare(code)  ◄── Errors caught HERE
           │
═══════════════════════════════════════════════════════

Phase 2: RDataFrame Integration (SAFE)
═══════════════════════════════════════════════════════
           │
           ▼
    rdf.Define("alias", "helper_alias(col1, col2)")
           │
           ▼
    Execute with data
═══════════════════════════════════════════════════════
```

## 8.2 Benefits

1. **Compile errors caught early** - Before event loop, with clear messages
2. **Functions can be saved/reused** - `.C` or `.so` files
3. **Easier debugging** - Named functions in stack traces
4. **Clean organization** - Complex expressions in named functions

---

# Part IX: Slicing Specification

## 9.1 Supported Patterns

| Pattern | Example | C++ Generation |
|---------|---------|----------------|
| Scalar index | `arr[5]` | `arr[5]` |
| Negative index | `arr[-1]` | `arr[arr.size()-1]` with bounds check |
| Basic slice | `arr[1:10]` | Loop with range |
| Step slice | `arr[::2]` | Loop with step |
| Full slice | `arr[:]` | Copy all |
| 2D index | `arr[i, j]` | `arr[i][j]` with bounds check |
| 2D slice | `arr[:, 0]` | Loop with inner bounds check |
| Fancy index | `arr[[1,4,7]]` | Index array lookup |
| Boolean mask | `arr[mask]` | Filtered copy |

## 9.2 Jagged Safety (CRITICAL)

For 2D slicing on jagged data, **always check inner bounds**:

```cpp
// WRONG - will segfault on empty inner vectors:
out[i] = events[i][0];

// CORRECT - safe for jagged:
if (events[i].size() > 0) {
    out[i] = events[i][0];
} else {
    out[i] = NaN;
}
```

---

# Part X: Testing Requirements

## 10.1 Test E: Shuffled Data Correctness (CRITICAL)

Validates that composite keys work when data is not row-aligned.

```python
def test_shuffled_friend_correctness():
    """
    Test that friend tree lookup works with shuffled data.
    
    1. Create main tree with keys in order [0,1,2,3,4]
    2. Create friend tree with SHUFFLED keys [3,1,4,0,2]
    3. Build composite key on both
    4. Add friend, use RDataFrame
    5. Verify correct value lookup (not positional)
    """
    # Implementation details in test file
```

## 10.2 Test F: Multiple Subframes

```python
def test_multiple_subframes():
    """
    Test multiple calibration subframes with different indices.
    
    1. Main tree
    2. Subframe S1 with index [row, sector]
    3. Subframe S2 with index [layer, orbit]
    4. Both correctly joined
    """
```

## 10.3 Test G: Missing Keys

```python
def test_missing_keys_behavior():
    """
    Document ROOT's behavior when friend has missing keys.
    
    1. Main has keys [0,1,2,3,4]
    2. Friend has keys [0,2,4] (missing 1,3)
    3. Verify: no crash, document fill values
    """
```

---

# Part XI: Implementation Roadmap

## Phase 1: IR Core (3-5 days)
- [ ] `ir_types.py` - IRType, IRTypeKind
- [ ] `ir_nodes.py` - All node classes
- [ ] `ir_errors.py` - IRError, SourceLocation, ErrorCollector
- [ ] Unit tests for manual IR construction

## Phase 2: Type Inference (3-5 days)
- [ ] `type_inferrer.py` - TypeInferrer class
- [ ] Tree scanning (TLeaf, TBranch, TClass)
- [ ] Edge case handling (missing dict, std::vector, arrays)
- [ ] Unit tests with mock trees

## Phase 3: AST → IR (3-5 days)
- [ ] `ir_builder.py` - IRBuilder class
- [ ] Python ast parsing
- [ ] Type/rank inference during build
- [ ] Validation pass

## Phase 4: Reflection (2-3 days)
- [ ] `reflection.py` - ReflectionCache
- [ ] Method resolution with fuzzy matching
- [ ] Property resolution
- [ ] Caching

## Phase 5: C++ Backend - Scalar (3-5 days)
- [ ] `backend_cpp.py` - CppBackend class
- [ ] Helper function generation (rank 0)
- [ ] Arithmetic, comparison, logical ops
- [ ] Function calls

## Phase 6: C++ Backend - RVec (5-7 days)
- [ ] Loop generation (rank 1)
- [ ] Method/property dispatch on collections
- [ ] Cross-collection indexing with bounds check

## Phase 7: Slicing (3-5 days)
- [ ] 1D slicing (basic, step, negative)
- [ ] 2D slicing with jagged safety
- [ ] Fancy indexing
- [ ] Boolean masks

## Phase 8: Function Library (2-3 days)
- [ ] `function_library.py` - FunctionLibrary class
- [ ] Compilation via Declare()
- [ ] Save/load macros

## Phase 9: RDF Builder (3-5 days)
- [ ] `rdf_builder.py` - RDFBuilder class
- [ ] Subframe registration
- [ ] Composite key generation
- [ ] Friend tree setup
- [ ] Define() chain

## Phase 10: Integration Tests (3-5 days)
- [ ] Test E (shuffled correctness)
- [ ] Test F (multiple subframes)
- [ ] Test G (missing keys)
- [ ] Full workflow tests

---

# Part XII: Benchmark Test Cases

| # | Expression | Tests |
|---|------------|-------|
| 1 | `sqrt(px**2 + py**2)` | Arithmetic |
| 2 | `(x - y)**2` | Power operator |
| 3 | `track.getX()` | Method call |
| 4 | `collision[track.GetCollisionIndex()].getX()` | Cross-collection |
| 5 | `track[:].getX()` | Vectorized method |
| 6 | `track.clusters()[i].amplitude()` | Nested access |
| 7 | `abs(collision[idx].getX() - track.getX()) < 10` | Mixed + comparison |
| 8 | `collision[idx].vertex().getZ()` | Method chain |

---

# Appendix A: Common Type Mappings

```python
CPP_TO_IR_TYPE = {
    # Floats
    "float": IRTypeKind.Float32,
    "Float_t": IRTypeKind.Float32,
    "double": IRTypeKind.Float64,
    "Double_t": IRTypeKind.Float64,
    
    # Integers
    "int": IRTypeKind.Int32,
    "Int_t": IRTypeKind.Int32,
    "long": IRTypeKind.Int64,
    "Long_t": IRTypeKind.Int64,
    "long long": IRTypeKind.Int64,
    "Long64_t": IRTypeKind.Int64,
    
    # Unsigned
    "unsigned int": IRTypeKind.UInt32,
    "UInt_t": IRTypeKind.UInt32,
    "unsigned long": IRTypeKind.UInt64,
    "ULong_t": IRTypeKind.UInt64,
    "unsigned long long": IRTypeKind.UInt64,
    "ULong64_t": IRTypeKind.UInt64,
    
    # Bool
    "bool": IRTypeKind.Bool,
    "Bool_t": IRTypeKind.Bool,
}
```

---

# Appendix B: Header Registry

```python
HEADER_REGISTRY = {
    # ROOT types
    "TParticle": ["TParticle.h"],
    "TLorentzVector": ["TLorentzVector.h"],
    
    # O2 types
    "o2::tpc::TrackTPC": ["DataFormatsTPC/TrackTPC.h"],
    "o2::track::TrackParCov": ["ReconstructionDataFormats/Track.h"],
    
    # Containers
    "ROOT::VecOps::RVec": ["ROOT/RVec.hxx"],
}
```

---

**End of Specification**
