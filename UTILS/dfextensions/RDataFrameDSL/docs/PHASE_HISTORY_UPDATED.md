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
| 8 | Method Broadcasting | 549 | ✅ Complete |
| 8.1 | Alias Referencing Fix | - | ✅ Complete |
| 9 | RVec Arithmetic Type Propagation | 569 | ✅ Complete |
| 10 | UX/API Sugar | 590 | ✅ Complete |
| 10.5 | RVec Reductions | 661 | ✅ Complete |
| 10.7 | Boolean Mask Filtering | 689 | ✅ Complete |
| 11 | Subframes (Deferred) | - | 📋 Planned |

**Current Total: 689 tests passing, 1 skipped**

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

### Phase 8: Method Broadcasting ✅
**Date:** December 9, 2025  
**Commit:** `465f8c2`  
**Tests:** 549 passed, 3 skipped

**Goal:** Element-wise method calls on RVec<Object>

**Deliverables:**
- `MethodBroadcastNode`, `PropertyBroadcastNode` IR nodes
- Detection: `RVec<Object>.method()` → broadcast loop
- Chaining: `tracks[:3].Pt()` (slice then broadcast)
- Type inference via TClass reflection + fallback maps

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

**Reviewed by:** GPT ✅, Gemini ✅, Claude ✅

---

### Phase 8.1: Alias Referencing Fix ✅
**Date:** December 10, 2025  
**Commit:** `888f158`

**Goal:** Allow aliases to reference other defined aliases

**Problem:** Subsequent definitions couldn't reference earlier aliases:
```python
dsl.define('pt', 'sqrt(px**2 + py**2)')
dsl.define('eta', '-log(tan(atan2(pt, pz)/2))')  # ❌ Failed: "Unknown variable 'pt'"
```

**Fix:** Added `_register_alias_type()` to `dsl_compiler.py`:
- After each `define()`, extract result type from IR
- Add to `self.schema` with proper type string
- Rebuild `TypeInferrer` with updated schema
- Update `CppCodeGenerator` with new inferrer

**Result:** Alias chaining now works correctly.

---

### Phase 9: RVec Arithmetic Type Propagation ✅
**Date:** December 10, 2025  
**Commit:** `aa25c3b`  
**Tests:** 569 passed, 1 skipped (was 549, +20)

**Goal:** Enable arithmetic on RVec results from method broadcasting

**Problem:** Expressions like `sqrt(tracks.Px()**2 + tracks.Py()**2)` failed due to type propagation issues.

**Two fixes implemented:**
1. **Type promotion:** `_get_effective_type_for_promotion()` extracts element type from `RVec<T>`
2. **ADL-compatible code generation:** Unqualified function names (`sqrt()` not `std::sqrt()`) for rank>0

**Key insight:** ROOT's RVec supports element-wise arithmetic via operator overloading. We needed:
1. Proper type promotion for `RVec<scalar>` operations
2. ADL-compatible function names in generated code

**Files modified:**
- `ir_types.py`: Add `_get_effective_type_for_promotion()`
- `backend_cpp.py`: Use unqualified names for rank>0 to enable ADL

**Reviewed by:** GPT ✅, Gemini ✅, Claude ✅

---

### Phase 10: UX/API Sugar ✅
**Date:** December 10, 2025  
**Commit:** `d2e4404`  
**Tests:** 590 passed, 1 skipped (was 569, +21)

**Goal:** Eliminate manual schema definition

**Before (tedious):**
```python
schema = {"px": "double", "py": "double", "tracks": "RVec<TLorentzVector>"}
dsl = DSLCompiler(schema)
```

**After (simple):**
```python
dsl = DSLCompiler.from_tree("data.root", "Events")
dsl.define("pt", "sqrt(px**2 + py**2)")
```

**New methods:**
| Method | Description |
|--------|-------------|
| `from_tree(filename, treename, overrides)` | Auto-infer schema from ROOT file |
| `show_types(include_definitions)` | Display inferred types for debugging |
| `validate()` | Fast consistency check without ROOT compilation |

**Also added:** `TypeInferrer.to_simple_schema()` for schema export

**Reviewed by:** GPT ✅, Gemini ✅, Claude ✅

---

### Phase 10.5: RVec Reductions ✅
**Date:** December 10, 2025  
**Commit:** `bb89961`  
**Tests:** 661 passed, 1 skipped (was 590, +71)

**Goal:** Add reduction functions that operate on RVec and return scalars

**Dual syntax support:**
```python
# Function style
dsl.define("total_pt", "Sum(track_pt)")
dsl.define("avg_pt", "Mean(track_pt)")
dsl.define("n_high", "Sum(track_pt > 1.0)")  # Count pattern

# Method style (equivalent)
dsl.define("total_pt", "track_pt.sum()")
dsl.define("avg_pt", "track_pt.mean()")
```

**Supported functions:**
| Function | Method | C++ | Returns |
|----------|--------|-----|---------|
| `Sum(vec)` | `vec.sum()` | `ROOT::VecOps::Sum` | element type (int for bool) |
| `Mean(vec)` | `vec.mean()` | `ROOT::VecOps::Mean` | double |
| `Max(vec)` | `vec.max()` | `ROOT::VecOps::Max` | element type |
| `Min(vec)` | `vec.min()` | `ROOT::VecOps::Min` | element type |
| `Any(vec)` | `vec.any()` | `ROOT::VecOps::Any` | bool |
| `All(vec)` | `vec.all()` | `ROOT::VecOps::All` | bool |
| `StdDev(vec)` | `vec.std()` | `ROOT::VecOps::StdDev` | double |
| `Var(vec)` | `vec.var()` | `ROOT::VecOps::Var` | double |

**Special case:** `Sum(RVec<bool>)` returns `int` (count of true values)

**Files modified:**
- `constants.py`: Add `REDUCTION_FUNCTIONS`, `RVEC_AGGREGATION_METHODS`
- `ir_builder.py`: Add `is_reduction` flag and method-style handling

**Reviewed by:** GPT ✅, Gemini ✅, Claude ✅

---

### Phase 10.7: Boolean Mask Filtering ✅
**Date:** December 11, 2025  
**Commit:** `11ae1b7`  
**Tests:** 689 passed, 1 skipped (was 661, +28)

**Goal:** Add comprehensive tests for boolean mask filtering

**Key Discovery:** Boolean masking was already implemented via `SliceKind.BOOLEAN` in Phase 7. This phase adds comprehensive tests and documentation.

**Working patterns:**
```python
# Basic filtering
dsl.define("high_pt", "pt[pt > 1.0]")

# Cross-vector filtering
dsl.define("central_pt", "pt[abs(eta) < 1.0]")

# Object vector filtering
dsl.define("good_tracks", "tracks[tracks.Pt() > 1.0]")

# Chaining with reductions
dsl.define("high_pt", "pt[pt > 2.0]")
dsl.define("sum_high", "Sum(high_pt)")
dsl.define("n_high", "high_pt.size()")
```

**Bug Fixed:** Alias registration wasn't calling `TypeInferrer.register_alias()`, preventing chained expressions. Fixed by adding one line in `dsl_compiler.py`:
```python
self._inferrer.register_alias(name, ir.dtype, ir.rank, ir.is_jagged)
```

**Test coverage:**
- Code generation (4 tests)
- Return types (3 tests)
- Compilation (5 tests)
- Execution (5 tests)
- Chaining with reductions (4 tests)
- Object vectors (3 tests)
- Error handling (1 test)

**Reviewed by:** GPT ✅, Gemini ✅, Claude ✅

---

## Complete Feature Summary

### What's Available Now (689 tests)

| Category | Examples | Status |
|----------|----------|--------|
| **Auto-schema** | `DSLCompiler.from_tree("data.root", "Events")` | ✅ |
| **Scalar arithmetic** | `sqrt(px**2 + py**2)` | ✅ |
| **Comparisons** | `pt > 10.0`, `eta < 1.0` | ✅ |
| **Ternary** | `pt if pt > 0 else -pt` | ✅ |
| **RVec indexing** | `tracks[0]`, `tracks[-1]` | ✅ |
| **RVec slicing** | `tracks[:3]`, `tracks[1:5]` | ✅ |
| **RVec methods** | `tracks.size()`, `tracks.empty()` | ✅ |
| **Object methods** | `particle.Pt()`, `track.Px()` | ✅ |
| **Method broadcasting** | `tracks.Px()` → `RVec<double>` | ✅ |
| **RVec arithmetic** | `sqrt(tracks.Px()**2 + tracks.Py()**2)` | ✅ |
| **Reductions** | `Sum(pt)`, `Mean(pt)`, `pt.sum()` | ✅ |
| **Boolean filtering** | `pt[pt > 1.0]`, `tracks[tracks.Pt() > 5.0]` | ✅ |
| **Alias referencing** | `define("high_pt", "pt > 10")` then use `high_pt` | ✅ |
| **Type display** | `dsl.show_types()` | ✅ |

### Not Yet Implemented

| Feature | Example | Priority |
|---------|---------|----------|
| Subframes | `df.tracks.Pt()` | Phase 11 (deferred) |
| Integer array indexing | `pt[indices]` | Low |
| Nested collections | `vector<vector<T>>` | Low |

---

## Git History (Recent Commits)

```
11ae1b7 Phase 10.7: RVec boolean mask filtering tests (689 tests)
bb89961 Phase 10.5: RVec reduction functions (661 tests)
d2e4404 Phase 10: UX/API Sugar - from_tree(), show_types(), validate() (590 tests)
aa25c3b Phase 9: RVec arithmetic type propagation (569 tests)
888f158 Fix: Allow aliases to reference other aliases
ddc4907 Add examples and documentation for Phase 8 demo
465f8c2 Phase 8: Method broadcasting (549 tests)
```

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

---

## Next Steps

1. **Documentation & Examples** - Demo scripts for ROOT team
2. **Phase 11: Subframes** - Deferred until ROOT team feedback

---

*Last updated: December 11, 2025*
