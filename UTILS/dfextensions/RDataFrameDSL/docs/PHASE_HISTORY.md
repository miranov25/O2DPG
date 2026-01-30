# Phase History — RDataFrameDSL

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
| 8 | Method Broadcasting | ~30 | ✅ Complete |
| 9 | Documentation & Examples | - | ✅ Complete |
| 10 | Performance Optimization | ~40 | ✅ Complete |
| 11 | Error Recovery & Diagnostics | ~50 | ✅ Complete |
| 12.1 | dfdraw Integration | +38 | ✅ Complete |
| 12.2 | RVec Selection for draw_figures | +33 | ✅ Complete |
| 12.3 | Composed Canvas | +29 | ✅ Complete |
| 12.5.DSL | Statistical Annotations | +6 | ✅ Complete |
| 12.6.DSL | to_aliasdf() Export | +20 | ✅ Complete |
| 13.2.DSL | ROOT ↔ Arrow Bridge | +31 | ✅ Complete |
| **13.5.B** | **C++ Function Registration** | **+38 (+40 exploration)** | **✅ Complete** |
| **13.5.C** | **DSL Integration (Registered Functions)** | **+18** | **✅ Complete** |
| **13.5.D** | **Numeric Widening (Overload Resolution)** | **+21** | **✅ Complete** |
| **13.6.A** | **RDataFrame Flattening** | **+49 (+11 exploration)** | **✅ Complete** |
| **13.6.C** | **N-D Slicing & Join Strategy** | **+97 (+24 integration)** | **✅ Complete** |
| **13.6.D** | **L1/L2 Resolution & UDF Tests** | **+40 invariance (+12 exploration)** | **✅ Complete** |
| **13.6.F** | **Schema Validation Audit** | **+0 (documentation)** | **✅ Complete** |
| **13.6.G** | **API Improvements & Test Infrastructure** | **+58** | **✅ Complete** |
| **13.7.A** | **Flatten Performance Optimization** | **Exploration complete** | **🟡 In Progress** |
| 13.4 | Integration Testing | - | 🔴 Pending |

**Current Total: ~2127 tests passing**

---

## Recent Phases (Team 2 — RDataFrameDSL)

### Phase 13.7.A: Flatten Performance Optimization (Exploration)
**Status:** 🟡 Exploration Complete - Implementation Pending  
**Date:** Jan 29-30, 2026  
**Goal:** Optimize flatten operations from RDataFrame with 10x+ speedup

#### Key Discovery

**`rdf.Take[]` is 200x faster than `AsNumpy` for RVec columns:**
- AsNumpy: Returns array of RVec objects requiring Python iteration (75k boundary crossings)
- Take: Returns std::vector directly to C++ (single boundary crossing)

#### Exploration Results

**Methods Benchmarked (25k events, 376k 1D elements, 5.5M 2D elements):**

| Method | 1D Time | 2D Time | Speedup | RDF Compatible |
|--------|---------|---------|---------|----------------|
| Baseline (AsNumpy + loop) | 404 ms | 6380 ms | 1.0x | ✅ |
| Take + C++ flatten | **38 ms** | **586 ms** | **10.9x** | ✅ |
| TTree::Draw (reference) | 42 ms | 740 ms | 8.6x | ⚠️ Deprecated |
| uproot (file-based) | 26 ms | 401 ms | 15.9x | ❌ |

**Scaling Analysis (Per-Element Cost):**

| Expression | AsNumpy (µs/elem) | Take+C++ (µs/elem) | Ratio |
|------------|-------------------|-------------------|-------|
| Raw 1D | 0.35 | 0.09 | **3.9x** |
| Raw 2D | 0.31 | 0.10 | **3.2x** |
| With Define() | 0.30-0.37 | 0.08-0.13 | **3-4x** |

**Expression Offset (JIT Cost):**
- Raw column: ~2 ms
- With Define(): ~140 ms (JIT compilation, fixed cost per query)

**Threading Results:**
- MT provides modest 1.35x improvement for Take+C++
- Not essential for optimization

**Failed Methods (Documented):**
- Foreach + C++ callback: Cling JIT segfault (ROOT 6.32 limitation)
- ak.from_rdataframe(): KeyError bug (ROOT/Awkward compatibility)

#### Architecture Decisions

1. **Use Take + C++ Flatten for all flattening** - 10x speedup, RDF-compatible
2. **Accept ~140ms JIT offset for expressions** - Fixed cost, slope dominates for large data
3. **Do not require MT** - Only 1.35x benefit, adds complexity
4. **Document uproot as alternative** - For file-based workflows

#### Deliverables

**Complete:**
- exploration_flatten.py (comprehensive benchmark script)
- exploration_flatten.log (full results)
- results.json (machine-readable)
- profiles/ directory (cProfile outputs per method)

**Pending:**
- [ ] README_exploration_flatten.md
- [ ] Implementation of Take + C++ flatten in flatten.py

**Reviewers:** Claude-Opus-4.5, GPT21, GPT22, Gemini (unanimous approval)

---

### Phase 13.6.G: API Improvements & Test Infrastructure
**Commits:** 78954206, b654031, e0500fc, 7c959af, fde0090, 76ba622, 9dfb69d, 45c9671, ca8ffe8  
**Date:** Jan 26-29, 2026  
**Goal:** API improvements, test infrastructure, and notebook support

#### Deliverables

**1. Generic Index Names**
- Changed `track_idx` → `idx_1`, `cluster_idx` → `idx_2`
- Added `index_names` parameter for custom naming:
  ```python
  flatten_to_dataframe(..., index_names={1: 'track_idx', 2: 'cluster_idx'})
  ```

**2. Configurable Redefinition Policy**
- Added `redefinition` parameter: `error/allow/skip/ifneeded`
- Default `'error'` maintains backwards compatibility
- `'ifneeded'` recommended for Jupyter notebooks (idempotent cells)
- Expression normalization for whitespace-insensitive comparison
- Guard against physical column redefinition

**3. Define Alias Resolution Bug Fix**
- `define('x', 'alias')` now resolves aliases correctly
- Previously failed with 'Unknown variable' error
- Added `_materialize_aliases_for_define()` method

**4. Nested Method Broadcast Support**
- `tracks.clusters().getQ()` → `RVec<RVec<double>>` now works
- Added `is_nested` and `inner_element_type` to MethodBroadcastNode
- Generate nested loops for nested broadcasts in backend_cpp.py

**5. draw() API Improvements**
- `draw()` and `draw_batch()` now use `to_pandas()` internally
- `parent_id_column=None` option for simple operations
- Bool columns accepted as object dtype (ROOT quirk)

**6. Test Infrastructure**
- Parallel execution by default (`PARALLEL_ROOT=1`)
- ~6x speedup for Phase 2 tests
- Added `--help`, `--verbose`, `--quiet`, `--quick` options
- Slowest-first ordering for optimal core utilization
- Total test time: ~3 min (was ~10 min sequential)

**7. Notebook Support**
- 07_dsl_draw.ipynb: TTree::Draw vs DSL comparison
- Pre-flight tests: test_example_07_dsl_draw.py (36 tests)
- TPC layers added to demo mode (r=85-245cm)
- Units converted to cm for physics consistency

#### Test Results

| Test Suite | Count | Status |
|------------|-------|--------|
| Redefinition policy | 18 | ✅ Pass |
| Alias resolution | 4 | ✅ Pass |
| Notebook pre-flight | 36 | ✅ Pass |
| **Total passing** | **2127** | ✅ |

---

### Phase 13.6.F: Schema Validation Audit
**Status:** ✅ Complete (Documentation Only)  
**Date:** Jan 23-24, 2026  
**Document:** SCHEMA_VALIDATION_AUDIT_v1.1.md

#### Purpose

Document current schema validation architecture and propose `alias()` feature for deferred validation.

#### Key Findings

**Current Architecture Verified:**
- Schema propagation: Manual schema OR `from_tree()` auto-inference
- Validation at define-time via `IRBuilder._visit_Name()` (ir_builder.py:678-684)
- Schema update after `define()` via `_register_alias_type()` (dsl_compiler.py:544-545)
- Error handling with suggestions via `_find_similar_names()` (type_inferrer.py:836-855)

**Proposed alias() Feature:**
```python
# TTree::SetAlias style - deferred validation
dsl.alias("pt", "sqrt(px**2 + py**2)")  # Stored, not validated
dsl.alias("high_pt", "pt > 10")         # Can reference other aliases
rdf = dsl.apply(rdf)                    # NOW validated and compiled
```

**Semantics:**
- `define()`: Immediate validation (requires schema)
- `alias()`: Deferred validation (schema-free)
- Enables template libraries without upfront schema

**Implementation Requirements (for future phase):**
- Dependency resolution (topological sort)
- Cycle detection
- Error collection (report all failures together)

**Reviewer Approval:** Claude-Opus-4.5, GPT21, GPT22 (unanimous)  
**Implementation:** Deferred to future phase (alias() is P2 priority)

### Phase 13.6.D: L1/L2 Resolution & UDF Custom Class Tests
**Commits:** 981bc23, 838e825, 8e1135d, 917fbaf, 9986900, 91f4ed6, 7216314 (Jan 20-21, 2026)  
**Tag:** `phase-13.6.D`  
**Goal:** Resolve L1 (mixed-depth join) and L2 (nested RVec reductions) limitations, add UDF custom class member function tests

#### Deliverables

**1. L2 Limitation Resolution (Nested RVec Reductions)**
- **Root Cause:** `_visit_call()` in backend_cpp.py passed nested `RVec<RVec<T>>` (rank > 1) to ROOT functions that don't support them, causing JIT crashes
- **Fix:** Generate explicit C++ nested loops instead of relying on ROOT's vectorized operations
  - Added `_needs_nested_rvec_handling()`: Detect rank > 1 arguments
  - Added `_generate_nested_reduction()`: Sum/Mean/Min/Max with nested loops
  - Added `_generate_nested_elementwise()`: sqrt/abs/sin/cos with nested loops
  - Optimized paths: `_generate_2d_reduction()`, `_generate_3d_reduction()` for common cases
  - Include `<limits>` header for NaN handling in Mean/Min/Max

**2. L2 Invariance Test Suite (40 tests)**
- **25 L2 free function tests** across 1D, 2D, 3D dimensions:
  - Sum linearity: additivity, scalar multiplication, partition invariance
  - Mean definition: `Mean(X) == Sum(X) / Count(X)`, bounds `Min ≤ Mean ≤ Max`
  - sqrt invariances: inverse (`sqrt(x)² ≈ x`), structure preservation, product rule
  - Exact values: Verify against toy_nd deterministic formulas
  - Min/Max ordering: `Min ≤ Mean ≤ Max`, extrema contain all
  - Empty cases: `Sum(empty)==0`, `Mean(empty)==NaN`, `sqrt(empty)==empty`
  - 3D coverage: Separate code path validation

- **6 member function tests** on primitives:
  - Slice-method commutation
  - Structure preservation
  - Reduction monotonicity
  - Nested 2D operations
  - Chained slicing

- **9 UDF custom class tests** (ToyTrack, ToyCluster):
  - Exact value verification (Pythagorean triples: `Pt=5.0`)
  - Slice-then-method correctness
  - Reduction monotonicity on member results
  - Nested member access (`tracks[0].clusters()[0].getQ()`)
  - Sliced nested structures
  - Pythagorean identity validation across all tracks
  - Cluster geometry offsets
  - Total charge aggregation

**3. L1 Mixed-Depth Join Tests (15 E2E tests)**
- **Broadcasting invariances:**
  - 1D→2D broadcast: Track attributes replicated to all clusters
  - 0D→2D broadcast: Event attributes replicated to all clusters
  - 0D→1D broadcast: Event attributes replicated to all tracks
- **Weighted aggregation:** DSL Draw-equivalent functionality
- **Row count consistency:** No rows lost in join operations
- **Join correctness:** Inner join semantics verified

**4. UDF Infrastructure (Custom Classes)**
- **GenerateDictionary():** Proper ROOT dictionary generation for custom classes
  - Replaces fragile `#pragma link` statements
  - Writes class definitions to temp header file for rootcling
  - Generates dictionaries for: ToyCluster, ToyTrack, vector<ToyCluster>, vector<ToyTrack>
- **Type system improvements:**
  - `_is_numeric_type()`: Detect numeric types for correct fallback values
  - Allow MethodCallNode with Unknown type in validation
  - Use `T()` constructor for custom classes vs `quiet_NaN()` for numerics
  - Extract element type when subscripting `RVec<T>` → scalar
- **Pragma management:**
  - `ensure_custom_class_pragmas()`: Centralized pragma registration
  - Correct dependency order: ToyCluster → RVec<ToyCluster> → ToyTrack → RVec<ToyTrack>

**5. ROOT Introspection**
- `root_introspection.py`: Auto-discover methods via ROOT TClass
- Integration with dsl_compiler.py: Extract `_methods`, pass to IRBuilder
- Test coverage: 12 tests for TVector3, TLorentzVector, TNamed, custom classes

#### Test Results

**Total: +40 invariance tests (+12 exploration)**

| Test Suite | Count | Status | Notes |
|------------|-------|--------|-------|
| L2 free functions (1D/2D/3D) | 25 | ✅ Pass | Sum, Mean, sqrt, Min, Max, abs, empty |
| L2 member functions (primitives) | 6 | ✅ Pass | Slice-method patterns |
| L2 UDF (custom classes) | 9 | ✅ Pass | Sequential mode (`-v -n 0`) |
| L1 mixed-depth join E2E | 15 | ✅ Pass | Was 12, added 3 broadcast tests |
| ROOT introspection | 12 | ✅ Pass | Exploration tests |
| Pragma management | 11 | ✅ Pass | Exploration tests |
| **Total passing** | **~1997** | ✅ | Full suite with `-n 12` |

**Performance:**
- UDF tests sequential: 21.74s (9/9 pass)
- UDF tests parallel: ⚠️ Flaky (GenerateDictionary() race condition)
- Full suite parallel: 1957 passed, 6 skipped

#### API

**N-D Slicing with Reductions (L2 resolved):**
```python
# Now works - previously caused ROOT JIT crash
dsl.define("sum_sliced", "Sum(cluster_Q[0:2, :])")
dsl.define("mean_subset", "Mean(cluster_Q[:, 0:3])")
dsl.define("sqrt_sliced", "sqrt(cluster_Q[0:2, 0:3])")

# Generated C++ uses explicit loops, not ROOT reducers:
# for (size_t i = 0; i < 2; ++i) {
#     for (size_t j = 0; j < cluster_Q[i].size(); ++j) {
#         sum += cluster_Q[i][j];
#     }
# }
```

**UDF Member Functions:**
```python
# Custom class member functions on sliced data
dsl.define("first_pt", "tracks[0].Pt()")
dsl.define("sliced_pt", "tracks[:2].Pt()")
dsl.define("sum_pt", "Sum(tracks[:2].Pt())")
dsl.define("nested_Q", "tracks[0].clusters()[0].getQ()")
```

**Mixed-Depth Join (L1 resolved):**
```python
# 2D + 1D + 0D join with broadcast semantics
df = dsl.to_pandas(
    rdf,
    ['cluster_Q', 'track_pt', 'event_weight'],  # 2D, 1D, 0D
    join='inner'
)
# Result: Track attributes replicated to clusters,
#         event attributes replicated to all rows
```

#### Known Limitations

**L3: Unsupported UDF Syntax Patterns (By Design)**

| Unsupported | Supported Alternative | Reason |
|-------------|----------------------|--------|
| `vec.method()[:n]` | `vec[:n].method()` | Performance (slice-first) |
| `vec.size()` | `Sum(vec.Pt() >= 0)` | Disambiguation (container vs broadcast) |

**L4: UDF Tests - Flaky in Parallel Mode (ROOT Constraint)**
- **Cause:** GenerateDictionary() writes shared temp files, races in parallel execution
- **Impact:** UDF tests unreliable with `-n 12` on clean builds
- **Workaround:** Use sequential mode (`-v -n 0`) for UDF tests
- **Behavior:** Cached builds (`.so` files present) sometimes pass, clean builds fail
- **CI/CD:** Always run UDF tests sequentially

#### Implementation Details

**backend_cpp.py Changes (~110 lines):**
```python
def _needs_nested_rvec_handling(self, args):
    """Detect if any argument has rank > 1 (nested RVec)"""
    
def _generate_nested_reduction(self, func_name, arg):
    """Generate nested loops for Sum/Mean/Min/Max on nested RVec"""
    # Dispatches to _generate_2d_reduction, _generate_3d_reduction,
    # or _generate_generic_nested_reduction based on rank
    
def _generate_nested_elementwise(self, func_name, arg):
    """Generate nested loops for sqrt/abs/sin/cos on nested RVec"""
    # Preserves structure: RVec<RVec<T>> → RVec<RVec<T>>
    
def _is_numeric_type(self, dtype):
    """Check if type is numeric (for fallback value selection)"""
```

**ir_builder.py Changes (~15 lines):**
```python
def _visit_subscript(self, node):
    # Extract element type when subscripting RVec<T> to scalar
    # tracks[0] → dtype=ToyTrack (not RVec<ToyTrack>)
```

**toy_nd.py Changes (~100 lines):**
```python
def register_custom_classes():
    """Generate ROOT dictionaries via GenerateDictionary()"""
    # Writes ToyCluster and ToyTrack definitions to temp header
    # Calls ROOT.gInterpreter.GenerateDictionary(...)
    # More robust than #pragma link statements
```

#### Documentation

**CAPABILITY_MATRIX.md Updates:**
- L1 status: ⚠️ Partial → ✅ Resolved
- L2 status: 🧨 Broken → ✅ Resolved
- Custom class member functions: ❌ Not Implemented → ✅ Working (9 tests)
- Added L3 limitation (unsupported syntax patterns - by design)
- Added L4 limitation (parallel mode flakiness - ROOT constraint)

**feature_taxonomy.py (v1.9):**
- Updated test counts for resolved features
- Removed limitation markers from slice_2d, slice_chain, nd_slice_reduction
- Clarified L1 status (mixed-depth joins resolved, same-column constraint documented)

#### Key Design Decisions

**Q1:** How to handle L2 (ROOT doesn't support `RVec<RVec<T>>` in reducers)?  
**A:** Generate explicit nested loops in C++ instead of calling ROOT functions

**Q2:** Add UDF tests now or defer?  
**A:** Add now - member functions are critical for physics analysis (`.Pt()`, `.Eta()`, `.Phi()`)

**Q3:** How to handle parallel test flakiness for UDF tests?  
**A:** Document as L4 limitation, use sequential mode for UDF tests (ROOT constraint, not fixable)

**Q4:** Syntax for member functions on sliced data?  
**A:** Support `vec[:n].method()` (slice-first), not `vec.method()[:n]` (method-first) for performance

**Q5:** Merge L1 and L2 fixes in same phase?  
**A:** Yes - both were blocking issues, better to resolve together than fragment across phases

#### Crisis & Resolution

**Crisis:** L1 test requirements were lost during Phase 13.6.C approval process  
**Root Cause:** Main Reviewer approved "L1 Resolved" without verifying test evidence  
**Resolution:** Added 3 missing E2E tests immediately (broadcast invariances)

**Lessons Learned:**
1. Always demand test code evidence, not just status claims
2. Verify test counts match specifications
3. Cross-check all combinations tested before approving "Resolved"
4. When requirements are lost, fix immediately (don't defer)

#### Files Changed

| File | Lines | Purpose |
|------|-------|---------|
| RDataFrameDSL/backend_cpp.py | ~110 | Nested loop codegen for L2 |
| RDataFrameDSL/ir_builder.py | ~15 | Element type extraction |
| tests/generators/toy_nd.py | ~100 | GenerateDictionary() |
| tests/conftest_nd_additions.py | ~40 | Pragma management |
| tests/test_invariance_nd.py | +31 | L2 invariance tests |
| tests/test_invariance_udf.py | +350 (new) | UDF custom class tests |
| tests/test_invariance_join_e2e.py | +3 | L1 broadcast tests |
| tests/test_root_introspection.py | +140 | ROOT introspection |
| tests/test_pragma_registry.py | 11 | Pragma deduplication |
| docs/CAPABILITY_MATRIX.md | Updated | L1/L2 status, L3/L4 limitations |
| tests/feature_taxonomy.py | v1.9 | Test counts, status updates |

#### Test Results Timeline

| Date | Commit | Event | Tests |
|------|--------|-------|-------|
| Jan 20 | 981bc23 | L2 fix committed | 80 (25 L2) |
| Jan 20 | 838e825 | Member function tests | 86 (+6) |
| Jan 21 | 8e1135d | ROOT introspection | 98 (+12) |
| Jan 21 | 917fbaf | UDF infrastructure | 104 (6/9 UDF) |
| Jan 21 | 9986900 | UDF syntax fixes | 113 (9/9 UDF) ✅ |
| Jan 21 | 7216314 | L1 broadcast tests | 128 (+15 E2E) ✅ |

**Final Status:** 1997 tests passing, L1 ✅ Resolved, L2 ✅ Resolved, L3/L4 documented

#### Reviewers

**Multi-Reviewer Process (MTTU v1.7):**
- GPT7: Identified need for member function tests
- GPT1: Caught fixture stability issues, empty-slice semantic requirements
- GPT10: Emphasized comprehensive P0 coverage, exact value verification
- GPT6: Flagged Capability Matrix inconsistencies
- Claude-Opus-4.5 (Main Reviewer): Consolidated feedback, enforced governance compliance
- Gemini2: Confirmed architectural correctness

**Approval:** Unanimous after requirement gap resolution

#### Next Steps

**Phase 13.6.E (Optional):**
- Pragma management comprehensive testing (if needed)
- Additional UDF patterns (if gaps identified)
- Performance optimization (if user requests)

**Phase 13.7 (Pending):**
- E2E join performance optimization (Awkward Array bottleneck)
- Additional N-D slicing patterns (if needed)

**Phase 13.4 (Deferred):**
- Cross-team integration testing
- PyArrow memory architecture validation

---

### Phase 13.6.C: N-D Slicing & Join Strategy for Mixed-Depth Columns
**Commit:** b192fb7 (Jan 20, 2026)  
**Tag:** `phase-13.6.C`  
**Goal:** Implement N-dimensional slicing (2D-5D) and join strategy for combining columns with different nesting depths

**Deliverables:**
- **N-D Slicing:** Support for `cluster_Q[0:2, 0:3]` syntax up to 5D arrays
- **Join Strategy:** Four join modes (inner/outer/left/right) for mixed-depth columns
- **join_utils.py:** JoinPlan dataclass, join_dataframes(), broadcast_to_depth()
- **Integration:** join parameter added to flatten.py and dsl_compiler.to_pandas()
- **Documentation:** DSL_SPEC_ND_Slicing.md (join strategy specification)
- **Capability Matrix:** CAPABILITY_MATRIX.md updated to 28/28 features working

**API:**
```python
# N-D slicing
dsl.define("sub_cluster", "cluster_Q[0:2, 0:3]")  # 2D slice

# Join strategy for mixed-depth
df = dsl.to_pandas(
    rdf,
    columns=['cluster_Q', 'track_pt'],  # 2D + 1D
    join='inner'  # inner/outer/left/right
)
```

**Join Strategy:**
- **inner:** Intersection of indices (only events with both columns)
- **outer:** Union of indices (all events, NaN padding where missing)
- **left:** Keep all rows from deepest column
- **right:** Keep all rows from shallowest column (broadcast semantics)

**Test Coverage:**

| Test Category | File | Count | Type |
|---------------|------|-------|------|
| Join Utils | test_join_utils.py | 38 | Tier 1 (Python) |
| Flatten Join | test_flatten_join.py | 11 | Tier 1 (Python) |
| End-to-End Join | test_join_e2e.py | 12 | Tier 3 (ROOT) |
| C-Array Integration | test_d9_integration.py | 24 | Tier 3 (ROOT) |
| ND Invariances | test_invariance_nd.py | 14 | Tier 2 (DSL) |
| **Total** | | **97 (+24)** | |

**Known Limitations at Phase End:**
- **L1:** Different-length columns → Defer to separate to_pandas() calls
- **L2:** Reductions on sliced 2D → ROOT JIT crash (documented, tests skipped)

**Tests:** +97 new tests + 24 integration  
**Reviewers:** GPT9, GPT10, GPT6, GPT7, Coder (5/5 unanimous approval)

---

### Phase 13.6.A: RDataFrame Flattening
**Commit:** 70d8ff6 (Jan 13, 2026)  
**Goal:** Implement hierarchical data flattening (RVec → flat arrays) for TTree::Draw-like functionality

**Deliverables:**
- NumPy backend with preallocate strategy (baseline, no dependencies)
- Awkward Array backend for 2-level nesting (RVec<RVec>)
- C++ helper functions (production performance path)
- Support for struct types (RVec<Track> → multiple columns)
- DSL-computed RVec column flattening
- Permanent exploration tests for reproducibility across ROOT versions

**Key Features:**
- Index semantics: `event_id` (replicated), `track_idx` (0-based within event)
- AUTO backend selection heuristic (>1M → C++, nested → Awkward, <100k → NumPy)
- Memory measurement methodology (tracemalloc for Python, RSS for total)

**Performance:**
- 27ms for 500k tracks (target <500ms) ✅
- Roofline performance achieved with preallocate strategy

**Tests:** +49 production + 11 exploration  
**Reviewers:** Claude-Opus-4.5, Claude-Sonnet-4.5, GPT3, GPT4, GPT5, GPT6, Gemini2

---

### Phase 13.5.D: Numeric Widening for Overload Resolution
**Commit:** 8863b52 (Jan 13, 2026)  
**Goal:** Add C++-like implicit numeric conversions to overload resolution

**Deliverables:**
- Numeric widening rules: int→long→float→double
- Cost-based overload selection (exact=0, widening=1, narrowing=∞)
- IRBuilder integration with `_find_best_overload()`

**API:**
```python
# Registered function with double parameter
dsl.register_function_cpp("scale", "double scale(double x) { return x * 2; }")
dsl.define("y", "scale(int_column)")  # int→double widening works
```

**Tests:** +21 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 13.5.C: DSL Integration for Registered Functions
**Commit:** 7eed1e9 (Jan 13, 2026)  
**Goal:** Integrate registered C++ functions with DSL compiler

**Deliverables:**
- `_registered_functions` dictionary in DSLCompiler
- IRBuilder lookup for registered functions
- Type inference from registered function signatures
- Backend code generation for registered function calls

**API:**
```python
dsl.register_function_cpp("myFunc", "double myFunc(double x) { return x * 2; }")
dsl.define("result", "myFunc(column)")  # Uses registered function
```

**Tests:** +18 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 13.5.B: C++ Function Registration API
**Commit:** 06ddc3c (Jan 11, 2026)  
**Goal:** Allow users to register custom C++ functions for use in DSL expressions

**Deliverables:**
- `register_function_cpp(name, code)` API
- Signature parsing from C++ code
- gInterpreter compilation and validation
- Overload support (multiple signatures per name)
- 40 exploration tests for edge cases

**API:**
```python
dsl.register_function_cpp("clamp", """
double clamp(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}
""")
```

**Tests:** +38 production + 40 exploration  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4, GPT5, GPT6, Gemini2

---

### Phase 13.4: C-Array Indexing Support
**Commits:** db5faa6, 926a5b4, 5a73676 (Jan 7-8, 2026)  
**Goal:** Support C-array branches with stride calculation

**Deliverables:**
- Schema parser for C-array notation: `float fHits[3][4]`
- Stride calculation for multi-dimensional arrays
- Automatic conversion to RVec for DSL compatibility
- D9 integration tests

**API:**
```python
schema = {"fHits": "float[3][4]"}  # 3×4 C-array
dsl = DSLCompiler(schema)
dsl.define("hit0", "fHits[0]")  # First row
```

**Tests:** +24 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 13.3: TMatrixD/TVectorD Support & Nested RVec 2D Slicing
**Commits:** 2970aa6, d848b13, c09a2f8, 2277e94 (Jan 6-7, 2026)  
**Goal:** ROOT linear algebra types and nested RVec improvements

**Deliverables:**
- TMatrixD/TVectorD type detection in TypeInferrer
- Linear algebra IR nodes and backend
- Nested RVec type inference bug fix
- 2D slicing for nested RVec

**API:**
```python
# Matrix operations
dsl.define("trace", "matrix.Sum()")  # TVectorD diagonal
dsl.define("elem", "matrix[0][1]")   # Element access
```

**Tests:** +76 tests (54 unit + 22 ROOT integration)  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 13.2.DSL: ROOT ↔ Arrow Bridge
**Commit:** afda6fb (Dec 16, 2025)  
**Goal:** Bidirectional PyArrow interchange for RDataFrame data

**Deliverables:**
- `to_arrow()`: RDataFrame → PyArrow Table
- `from_arrow()`: PyArrow Table → column dict
- Scalar, RVec, and legacy array support
- Import fail-closed semantics (reject unknown types)

**API:**
```python
arrow_table = dsl.to_arrow(rdf, ['px', 'py', 'pt'])
data = dsl.from_arrow(arrow_table)
```

**Tests:** +31 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4, Gemini2

---

### Phase 12.6.DSL: to_aliasdf() Export
**Commit:** 78135bc (Dec 16, 2025)  
**Goal:** Export DSL definitions to AliasDataFrame format

**Deliverables:**
- `to_aliasdf()`: Export definitions for AliasDataFrame
- `get_definitions()`: Retrieve all defined expressions
- Dependency tracking for export ordering

**Tests:** +20 tests  
**Reviewers:** Claude-Opus-4.5, GPT3

---

### Phase 12.5.DSL: Statistical Annotations
**Commit:** 78135bc (Dec 16, 2025)  
**Goal:** Add statistical annotations to draw_figures()

**Deliverables:**
- Mean, std, count annotations on histograms
- Configurable annotation placement
- Integration with dfdraw

**Tests:** +6 tests  
**Reviewers:** Claude-Opus-4.5, GPT3

---

### Phase 12.3: Composed Canvas & Multi-Figure Drawing
**Commit:** 8fb348f (Dec 13, 2025)  
**Goal:** Support multiple figures on composed canvas

**Deliverables:**
- `draw_figures()` with subplot layout
- Figure arrangement (rows × cols)
- Shared axes support

**Tests:** +29 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 12.2: RVec Selection Functions
**Commit:** 1af065f (Dec 13, 2025)  
**Goal:** Selection functions for RVec filtering

**Deliverables:**
- `Take(vec, n)`: First n elements
- `Range(vec, start, end)`: Slice range
- `Where(vec, mask)`: Boolean filtering
- `IndicesFromOffsets()`: Offset conversion

**Tests:** +33 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 12.1: dfdraw Integration
**Commit:** 73d0bf2 (Dec 13, 2025)  
**Goal:** Integrate with dfdraw for visual validation

**Deliverables:**
- `draw()` method for single expressions
- `draw_batch()` for multiple expressions
- Histogram and scatter plot support

**Tests:** +38 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 11.1: Namespace Function Support
**Commits:** 4bd0e55, 7e5dd97, f95cc05 (Dec 12, 2025)  
**Goal:** Support TMath, ROOT.Math namespace functions

**Deliverables:**
- Phase 11.1a: Namespace function calls (`TMath.Sqrt(x)`)
- Phase 11.1b: Scalar-to-vector broadcasting
- Phase 11.1c: Dynamic namespace detection + dtype parameter

**API:**
```python
dsl.define("log_pt", "TMath.Log(pt)")
dsl.define("sin_phi", "ROOT.Math.sin(phi)")
```

**Tests:** +50 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 10: UX/API Sugar & RVec Reductions
**Commits:** d2e4404, bb89961, 11ae1b7 (Dec 10-11, 2025)  
**Goal:** User experience improvements and reduction functions

**Deliverables:**
- Phase 10.0: `from_tree()`, `show_types()`, `validate()`
- Phase 10.5: RVec reductions (Sum, Mean, Min, Max, std)
- Phase 10.7: Boolean mask filtering

**API:**
```python
dsl = DSLCompiler.from_tree(tree)  # Auto-infer schema
dsl.show_types()                    # Display column types
dsl.define("total", "Sum(pt)")      # Reduction
dsl.define("sel", "pt[pt > 1.0]")   # Boolean mask
```

**Tests:** +40 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, GPT4

---

### Phase 9: RVec Arithmetic Type Propagation
**Commit:** aa25c3b (Dec 10, 2025)  
**Goal:** Correct type inference for RVec arithmetic

**Deliverables:**
- Type promotion rules for RVec operations
- int + float → float propagation
- Rank preservation in operations

**Tests:** +20 tests  
**Reviewers:** Claude-Opus-4.5, GPT3

---

### Phase 8: Method Broadcasting
**Commit:** 465f8c2 (Dec 9, 2025)  
**Goal:** Element-wise method calls on RVec<Object>

**Deliverables:**
- MethodBroadcastNode IR node
- Code generation for `vec.Method()` → `RVec<T>`
- Property broadcasting (`vec.fMember`)

**API:**
```python
dsl.define("pts", "tracks.Pt()")      # RVec<double>
dsl.define("etas", "tracks.Eta()")    # RVec<double>
```

**Tests:** +30 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 7.9: DSLCompiler & RDataFrame Validation
**Commit:** 90cc7f3 (Dec 9, 2025)  
**Goal:** DSLCompiler class and end-to-end validation

**Deliverables:**
- DSLCompiler class with `define()`, `apply()`, `export_macro()`
- Multi-function pipeline support
- Unique function names (UUID suffix) for parallel safety
- End-to-end RDataFrame integration tests

**Tests:** +17 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 7: Slicing and Advanced Indexing
**Commit:** 9770608 (Dec 8, 2025)  
**Goal:** Python-like slicing for RVec

**Deliverables:**
- `[:n]` first n, `[-n:]` last n (with size clamping)
- `[n:]` from index, `[a:b]` range
- `[::step]` step slicing, `[::-1]` reverse
- `[mask]` boolean masking

**API:**
```python
dsl.define("first3", "pt[:3]")
dsl.define("last2", "pt[-2:]")
dsl.define("reversed", "pt[::-1]")
```

**Tests:** +36 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 6.9: ROOT Integration Tests
**Commit:** 1c33795 (Dec 8, 2025)  
**Goal:** Comprehensive ROOT integration validation

**Deliverables:**
- Category A: ROOT C++ behavior assumptions (12 tests)
- Category B: DSL code generation correctness (10 tests)
- Category C: RDataFrame integration (5 tests)
- Category D: Future phase techniques (7 tests)
- Bug fixes: `std::std::sqrt` → `std::sqrt`, negative literal folding

**Tests:** +35 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 6c: Private Member Reflection
**Commit:** a2423f3 (Dec 7, 2025)  
**Goal:** Access protected/private data members via TClass reflection

**Deliverables:**
- Access level detection via `TDataMember.Property()`
- Direct access for public members
- Reflection lambda for protected/private (GetOffset pattern)
- `use_reflection` flag (default True)

**Code Generation (protected member):**
```cpp
particle.fPx -> [&]() -> double {
    static TClass* cls = TClass::GetClass("TParticle");
    static TDataMember* dm = cls->GetDataMember("fPx");
    static Long_t offset = dm->GetOffset();
    return *reinterpret_cast<const double*>(
        reinterpret_cast<const char*>(&particle) + offset);
}()
```

**Tests:** +34 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 6b: RVec Operations & Safe Indexing
**Commit:** 5ede4ae (Dec 7, 2025)  
**Goal:** RVec arithmetic and safe element access

**Deliverables:**
- RVec arithmetic: element-wise +, -, *, /, **
- RVec comparisons: <, >, ==, etc.
- Vectorized math via ADL: `sqrt(pt)`, `abs(eta)`
- Simple indexing: `pt[0]`, `pt[i]`
- Negative indexing: `pt[-1]` → `pt[size()-1]`
- Safe bounds checking (default ON, returns NaN)

**Tests:** +39 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phase 6a: Object Methods & Properties
**Commit:** 7079662 (Dec 6, 2025)  
**Goal:** Method calls and property access on C++ objects

**Deliverables:**
- MethodCallNode code generation with `const T&` parameter passing
- PropertyAccessNode for public data members
- CLASS_HEADERS registry for automatic header inclusion

**Validated Classes:**
- TParticle: GetPx(), GetPy(), GetPz(), Pt(), Eta(), Phi()
- TLorentzVector: Px(), Py(), Pz(), E(), Pt(), M()
- TVector3: X(), Y(), Z(), Mag(), Theta(), Phi()
- TString: Length(), IsNull()

**Tests:** +40 tests  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Phases 1-5: Foundation (Initial Commit)
**Commit:** 2edffe2 (Dec 6, 2025)  
**Goal:** Core DSL infrastructure

**Phase 1: IR Core**
- Type system and node hierarchy
- Error model with recovery modes
- ~50 tests

**Phase 2: Type Inference**
- ROOT TTree reflection
- Schema support
- Vector/RVec handling
- ~30 tests

**Phase 3: IR Builder**
- Python AST → IR transformation
- Type promotion and rank broadcasting
- ~60 tests

**Phase 4: Class Reflection**
- TClass-first lookup with schema fallback
- Fuzzy matching for suggestions
- ~40 tests

**Phase 5: C++ Backend**
- Scalar code generation
- gInterpreter compilation
- RDataFrame integration
- ~80 tests

**Key Design Decisions:**
- Two-phase validation: compile helpers before RDataFrame execution
- Standalone helper functions (not lambdas) for debuggability
- TClass primary, schema fallback for reflection
- Alphabetical input ordering for deterministic signatures

**Total Tests:** 313 passing (78 IR + 41 type + 76 builder + 41 reflection + 77 backend)  
**Reviewers:** Claude-Opus-4.5, GPT3, Gemini2

---

### Overview

Phase 13 implements PyArrow-based memory optimization across all teams.

| Phase | Team | Scope | Status |
|-------|------|-------|--------|
| 13.1.GB | Team 3 | GroupBy PyArrow pilot | 🔴 Not started |
| 13.1.DF | Team 3 | dfdraw PyArrow input | 🔴 Not started |
| **13.2.DSL** | **Team 2** | **ROOT ↔ Arrow bridge** | ✅ **Complete** |
| 13.3.ADF | Team 1 | Hybrid ADF implementation | 🟡 Pending pilots |
| 13.4 | All | Integration testing | 🔴 Pending |
| **13.5.B** | **Team 2** | **C++ Function Registration** | ✅ **Complete** |
| **13.5.C** | **Team 2** | **DSL Integration (Registered Functions)** | ✅ **Complete** |
| **13.5.D** | **Team 2** | **Numeric Widening (Overload Resolution)** | ✅ **Complete** |
| **13.6.A** | **Team 2** | **RDataFrame Flattening** | ✅ **Complete** |
| **13.6.C** | **Team 2** | **N-D Slicing & Join Strategy** | ✅ **Complete** |
| **13.6.D** | **Team 2** | **L1/L2 Resolution & UDF Tests** | ✅ **Complete** |
| **13.6.F** | **Team 2** | **Schema Validation Audit** | ✅ **Complete** |
| **13.6.G** | **Team 2** | **API Improvements & Test Infrastructure** | ✅ **Complete** |
| **13.7.A** | **Team 2** | **Flatten Performance Optimization** | 🟡 **In Progress** |

### Key Architecture Decision

**Hybrid Design:** Arrow for storage/transport, NumPy/Pandas for compute

- Phase 9 evidence: PyArrow eval is 8-10× slower than NumPy
- Resolution: Use Arrow only for storage, scatter/gather, and sort
- Compute remains in NumPy/Pandas (proven fast)

---

## Review Process

Each phase follows this workflow:

1. **Design Review Request** — Detailed proposal with questions
2. **Multi-Reviewer Consensus** — GPT, Gemini, Claude review
3. **Coder Instructions** — Detailed implementation guide
4. **Implementation** — Following instructions exactly
5. **Test Validation** — All tests must pass
6. **Owner Approval** — Final sign-off before merge

**Active Reviewers (Phases 13.5+):**
- Claude Opus 4.5 (Main Reviewer, Architecture Lead)
- Claude Sonnet 4.5 (Architecture Support, Technical Reviewer)
- GPT-5.2 Thinking (Detailed Technical Analysis)
- GPT1, GPT3, GPT4, GPT5, GPT6, GPT7, GPT9, GPT10 (Implementation Reviews)
- Gemini2 (RDataFrameDSL Domain Expert)

**Approval Requirement:** Unanimous consent from all reviewers before commit, with Main Reviewer consolidation per MTTU_Reviewer v1.7.

---

## Key Files

| File | Purpose |
|------|---------|
| `dsl_compiler.py` | Main DSLCompiler class with all methods |
| `ir_builder.py` | IR builder with overload resolution |
| `backend_cpp.py` | C++ code generation with nested RVec handling |
| `flatten.py` | Flattening backends (NumPy, Awkward, C++) |
| `join_utils.py` | Join strategy (JoinPlan, join_dataframes, broadcast) |
| `root_introspection.py` | ROOT class method discovery via TClass |
| `tests/generators/toy_nd.py` | Custom class generators (ToyTrack, ToyCluster) |
| `tests/exploration/exploration_flatten.py` | Flatten performance benchmark |
| `tests/test_invariance_nd.py` | N-D slicing + L2 invariance tests |
| `tests/test_invariance_udf.py` | UDF custom class member function tests |
| `tests/test_invariance_join_e2e.py` | Mixed-depth join E2E tests |
| `tests/test_root_introspection.py` | ROOT reflection tests |
| `tests/test_pragma_registry.py` | Pragma deduplication tests |
| `docs/CAPABILITY_MATRIX.md` | Auto-generated feature status matrix |
| `tests/feature_taxonomy.py` | Feature definitions and test tracking |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Original | Phases 1-8 |
| 2.0 | Dec 16, 2025 | Added Phases 12.x and 13.2.DSL |
| 3.0 | Jan 13, 2026 | Added Phases 13.5.B/C/D and 13.6.A |
| 4.0 | Jan 20, 2026 | Added Phase 13.6.C (N-D Slicing & Join Strategy) |
| 4.1 | Jan 21, 2026 | Added Phase 13.6.D (L1/L2 Resolution & UDF Tests) |
| **5.0** | **Jan 30, 2026** | **Added Phases 13.6.F, 13.6.G, 13.7.A exploration** |
