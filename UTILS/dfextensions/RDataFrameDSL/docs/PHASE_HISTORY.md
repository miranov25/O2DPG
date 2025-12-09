# Phase History

## Overview

The RDataFrameDSL project follows a phased development approach with formal design reviews before each implementation phase.

| Phase | Name | Tests | Status |
|-------|------|-------|--------|
| 1 | IR Core | ~50 | ✅ Complete |
| 2 | Type Inference | ~30 | ✅ Complete |
| 3 | IRBuilder | ~60 | ✅ Complete |
| 4 | Class Reflection | ~40 | ✅ Complete |
| 5 | C++ Code Generation (Scalars) | ~80 | ✅ Complete |
| 6a | Object Methods & Properties | ~40 | ✅ Complete |
| 6b | RVec Operations | ~50 | ✅ Complete |
| 6c | Private Member Reflection | ~30 | ✅ Complete |
| 6.9 | ROOT Integration Validation | ~20 | ✅ Complete |
| 7 | RVec Slicing & Masking | ~36 | ✅ Complete |
| 7.9 | DSLCompiler & RDF Stress Tests | ~17 | ✅ Complete |
| 8 | Method Broadcasting | ~30 | 🔄 In Progress |
| 9 | Documentation & Examples | - | 📋 Planned |

**Current Total: 514 tests passing**

---

## Phase Details

### Phase 1: IR Core
**Goal:** Define the intermediate representation for expressions

**Deliverables:**
- `IRType` and `IRTypeKind` for type system
- Base `IRNode` class with visitor pattern
- `ConstantNode`, `VariableNode`, `BinaryOpNode`, `UnaryOpNode`
- `IRError` with suggestions

**Key Decision:** Use dataclasses for IR nodes (clean, immutable-ish)

---

### Phase 2: Type Inference
**Goal:** Infer expression types from schema

**Deliverables:**
- `TypeInferrer` class
- Schema parsing (simple dict → internal format)
- Type promotion rules (int + double → double)
- RVec element type extraction

**Key Decision:** Support both `RVec<T>` and `std::vector<T>` notation

---

### Phase 3: IRBuilder  
**Goal:** Parse Python expressions to IR

**Deliverables:**
- `IRBuilder.build(expression)` using `ast.parse()`
- Visitor methods for all Python AST node types
- Support for arithmetic, comparisons, function calls
- Conditional expression (`x if cond else y`)

**Key Decision:** Use Python's own parser, not custom grammar

---

### Phase 4: Class Reflection
**Goal:** Support object method/property access with type inference

**Deliverables:**
- `MethodCallNode`, `PropertyAccessNode`
- TClass-based reflection for return types
- Fuzzy matching for error suggestions
- `ReflectionCache` for performance

**Key Decision:** Query TClass at IR-build time, not codegen time

---

### Phase 5: C++ Code Generation (Scalars)
**Goal:** Generate compilable C++ from scalar IR

**Deliverables:**
- `CppCodeGenerator` class
- `GeneratedFunction` dataclass
- `FunctionLibrary` for managing functions
- Header tracking (`<cmath>`, `<TMath.h>`)
- Function naming with collision avoidance

**Key Decision:** Generate standalone functions, not lambdas (better debugging)

---

### Phase 6a: Object Methods & Properties
**Goal:** Generate C++ for object access

**Deliverables:**
- `_visit_method_call()` in backend
- `_visit_property_access()` in backend
- Header tracking for ROOT classes

**Key Decision:** Direct member access for public members

---

### Phase 6b: RVec Operations
**Goal:** Support RVec arithmetic, indexing, methods

**Deliverables:**
- RVec arithmetic (element-wise via ADL)
- Safe indexing with NaN on OOB
- Negative index support (`pt[-1]`)
- RVec methods: `size()`, `empty()`, `at()`

**Key Decision:** Safe indexing ON by default (returns NaN, not crash)

**Bug Found:** ROOT's `gInterpreter.Calc()` returns int for all types - switched to PyROOT calls for validation

---

### Phase 6c: Private Member Reflection
**Goal:** Access protected/private members like TTree::Draw did

**Deliverables:**
- TDataMember access level detection
- Reflection-based access via `GetOffset()`
- `IsBasic()` validation (reject non-POD)
- `IsaPointer()` validation (reject pointers)
- Thread-safe static caching

**Key Decision:** Use TClass reflection (same as TTree::Draw historical behavior)

**Critical Discovery:** Many ALICE O2 classes use private members that physicists expect to access

---

### Phase 6.9: ROOT Integration Validation
**Goal:** Validate ROOT behavior assumptions before proceeding

**Deliverables:**
- Test suite proving ROOT behaviors:
  - `VecOps::Take()` crashes on short vectors (need clamping)
  - `VecOps::Reverse()` not available (use manual loop)
  - Boolean masking works natively
  - Safe index lambda pattern works
- Tests for `EnableImplicitMT()` safety

**Key Decision:** Always clamp slice parameters to vector size

---

### Phase 7: RVec Slicing & Masking
**Goal:** Python-like slicing on RVec

**Deliverables:**
- `SliceKind` enum (7 kinds)
- `RVecSliceNode` IR node
- Slice classification in IRBuilder
- Code generation for all patterns:

| Pattern | SliceKind | Generation |
|---------|-----------|------------|
| `pt[:3]` | FIRST_N | `Take(pt, min(3, size))` |
| `pt[-3:]` | LAST_N | Clamped negative Take |
| `pt[2:]` | FROM_INDEX | Range loop |
| `pt[1:3]` | RANGE | `Range(1, min(3, size))` |
| `pt[::2]` | STEP | Loop with `i += 2` |
| `pt[::-1]` | REVERSE | Manual reverse loop |
| `pt[mask]` | BOOLEAN | Native `pt[mask]` |

**Bug Fixed:** `Take(v, 3)` crashes if `v.size() < 3` - added size clamping everywhere

---

### Phase 7.9: DSLCompiler & RDF Validation
**Goal:** High-level API and multi-function stress tests

**Deliverables:**
- `DSLCompiler` class with simple API
- `export_macro()` with DSL comments
- `preview()` for debugging
- Multi-function pipeline tests (10 functions together)
- Multi-threading tests with `EnableImplicitMT(4)`
- Empty vector edge case tests

**Key Decision:** UUID suffix on function names for parallel test safety

**Bug Fixed:** Parallel pytest workers sharing gInterpreter caused redefinition errors

---

### Phase 8: Method Broadcasting (In Progress)
**Goal:** Element-wise method calls on RVec<Object>

**Target Deliverables:**
- `MethodBroadcastNode`, `PropertyBroadcastNode`
- Detection: RVec<Object>.method() → broadcast loop
- Chaining: `tracks[:3].Pt()` (slice then broadcast)
- Error: `tracks.Pt()[:3]` with helpful suggestion
- Type inference via TClass reflection

**Example:**
```python
dsl.define("track_pts", "tracks.Pt()")  # → RVec<double>
```

**Generated C++:**
```cpp
[&]() -> ROOT::RVec<double> {
    ROOT::RVec<double> result;
    result.reserve(tracks.size());
    for (const auto& elem : tracks) {
        result.push_back(elem.Pt());
    }
    return result;
}()
```

---

### Phase 9: Documentation & Examples (Planned)
**Goal:** Complete documentation and demo examples

**Planned Deliverables:**
- User guide with tutorials
- Developer guide
- Example scripts for ROOT team demo
- Comparison: DSL vs raw RDataFrame
- TClonesArray support (if needed)

---

## Review Process

Each phase follows this workflow:

1. **Design Review Request** - Detailed proposal with questions
2. **Multi-Reviewer Consensus** - GPT, Gemini, Claude review
3. **Coder Instructions** - Detailed implementation guide
4. **Implementation** - Following instructions exactly
5. **Test Validation** - All tests must pass
6. **Owner Approval** - Final sign-off before merge

This ensures:
- Design issues caught early
- Consistent code quality
- No regressions (test count only increases)
- Clear documentation trail
