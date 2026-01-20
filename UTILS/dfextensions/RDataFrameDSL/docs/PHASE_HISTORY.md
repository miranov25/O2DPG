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
| 13.4 | Integration Testing | - | 🔴 Pending |

**Current Total: ~1648 tests passing**

---

## Recent Phases (Team 2 — RDataFrameDSL)

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

**API:**
```python
# Flatten RVec columns
df = flatten_to_dataframe(
    data=rdf.AsNumpy(["event_id", "track_pt"]),
    rvec_columns=["track_pt"],
    parent_id_column="event_id",
    backend=FlattenBackend.AUTO  # NumPy, Awkward, or C++
)

# Result:
# event_id: [100, 100, 101, 101, 101]
# track_idx: [0, 1, 0, 1, 2]
# track_pt: [1.2, 3.4, 5.6, 7.8, 9.0]
```

**Tests:** +49 production (27 correctness + 5 benchmarks + 5 integration + 12 extended) + 11 exploration  
**Specification:** PHASE_13_6_A_v02_Proposal.md  
**Reviewers:** Claude-Opus-4.5, Claude-Sonnet-4.5, GPT3, GPT4, GPT5, GPT6, Gemini2

**Next:** Phase 13.6.B (Draw Interface + TTree::Draw-equivalent stress tests)

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

**Implementation:**
```python
# RDataFrameDSL/join_utils.py
@dataclass
class JoinPlan:
    """
    Strategy for joining columns of different depths.
    
    Attributes:
        columns: List of (name, depth) tuples
        target_depth: Maximum depth to align to
        join_mode: 'inner', 'outer', 'left', 'right'
    """
    
def join_dataframes(dfs: List[pd.DataFrame], plan: JoinPlan) -> pd.DataFrame:
    """Join DataFrames with broadcast/alignment strategy."""
    
def broadcast_to_depth(df: pd.DataFrame, target_depth: int) -> pd.DataFrame:
    """Broadcast shallow columns to match deeper nesting."""
```

**Backend Changes:**
```python
# backend_cpp.py: N-D slicing support
def _visit_nd_slice(self, node):
    """Generate C++ for N-D array slicing"""
    # Supports: cluster[i, j], cluster[0:2, :], cluster[:, -1]
    # Up to 5D: array[d1, d2, d3, d4, d5]
    
def _visit_nd_index(self, node):
    """Generate C++ for N-D array indexing"""
    # Supports: cluster[0, 1], hit[i, j, k]
```

**Test Coverage:**

| Test Category | File | Count | Type |
|---------------|------|-------|------|
| Join Utils | test_join_utils.py | 38 | Tier 1 (Python) |
| Flatten Join | test_flatten_join.py | 11 | Tier 1 (Python) |
| End-to-End Join | test_join_e2e.py | 12 | Tier 3 (ROOT) |
| C-Array Integration | test_d9_integration.py | 24 | Tier 3 (ROOT) |
| ND Invariances | test_invariance_nd.py | 14 | Tier 2 (DSL) |
| **Total** | | **97 (+24)** | |

**Resolved Limitations:**
- **L1 (RESOLVED):** Different-length columns in same to_pandas() call
  - Solution: Separate to_pandas() calls per depth, then join
  - Status: 3 previously xfailed tests now pass

**Known Limitations:**
- **L2 (DOCUMENTED):** Reductions on sliced 2D columns cause ROOT JIT crash
  - Examples: `Sum(cluster_Q[:2, :])`, `Mean(cluster_Q[:, 0:3])`
  - Cause: Backend C++ code generation issue, not architectural
  - Workaround: Export to pandas first, then reduce
  - Status: 4 tests skipped with clear documentation

**Performance:**
- Join operations: 49 Tier 1 tests pass in <1s
- E2E tests: 12 tests pass in ~78s (optimization pending Phase 13.7)
- Bottleneck: Awkward Array conversion, not join logic

**Feature Taxonomy Updates (v1.6):**
- Added: `nd_slice_2d`, `nd_slice_3d`, `nd_slice_4d`, `nd_slice_5d`
- Added: `nd_join_strategy` (12 tests)
- Added: `nd_slice_arithmetic`, `nd_slice_order`
- Updated: L1 status (Resolved), L2 added (documented)

**Documentation:**
```markdown
# DSL_SPEC_ND_Slicing.md
- N-D indexing semantics (per-parent indexing)
- Join strategy algorithms (inner/outer/left/right)
- Broadcast replication rules
- Index column generation (event_id, track_idx, cluster_idx, etc.)
```

**Examples:**
```python
# 2D slicing: cluster_Q[event_slice, cluster_slice]
dsl.define("first_two_events", "cluster_Q[0:2, :]")
dsl.define("last_cluster_per_track", "cluster_Q[:, -1]")

# 3D slicing: hit_E[event_slice, cluster_slice, hit_slice]
dsl.define("first_hit_per_cluster", "hit_E[:, :, 0]")

# Mixed-depth join
df = dsl.to_pandas(
    rdf,
    ['cluster_Q', 'track_pt', 'event_weight'],  # 2D, 1D, 0D
    join='inner'
)
# Result has aligned indices with replication:
# event_id | track_idx | cluster_Q | track_pt | event_weight
#    0     |     0     |    123    |   5.2    |     1.0
#    0     |     0     |    456    |   5.2    |     1.0  (replicated)
#    0     |     1     |    789    |   3.1    |     1.0  (replicated)
```

**Key Design Decisions:**
- **Q1:** How many dimensions to support? → **5D** (sufficient for ALICE use cases)
- **Q2:** Join default behavior? → **inner** (safest, no unexpected NaNs)
- **Q3:** Handle L2 (reduction crash)? → **Document + skip tests** (backend issue, not architecture)
- **Q4:** Optimize E2E performance? → **Defer to Phase 13.7** (join logic correct, conversion slow)

**Tests:** +97 new (38 join utils + 11 flatten join + 12 E2E + 14 ND invariance + 22 other) + 24 integration  
**Total:** 1648 passed, 4 skipped (L2 limitation)  
**Specification:** PHASE_13_6_C_Proposal.md (v1.2 approved)  
**Reviewers:** GPT9, GPT10, GPT6, GPT7, Coder (5/5 unanimous approval on infrastructure)

**Git Tag:** `phase-13.6.C` at commit `b192fb7`

**Next:** Phase 13.6.D (Performance optimization for join operations) or Phase 13.7 (Method calls on sliced results)

---

### Phase 13.5.D: Numeric Widening for Overload Resolution
**Commit:** 8863b526 (Jan 13, 2026)  
**Goal:** Add C++-like implicit numeric conversions to overload resolution

**Deliverables:**
- Float32 → Float64 promotion (rank 1)
- Int widening: Int8 → Int16 → Int32 → Int64 (rank 1)
- Cross-type conversion: IntX → Float64 (rank 2)
- Ranked candidate selection (lowest total rank wins)
- Clear error messages for forbidden conversions

**Forbidden Conversions:**
- ❌ Narrowing (Float64 → Float32): precision loss
- ❌ Int → Float32: lossy for values > 16,777,216
- ❌ Signed ↔ Unsigned: ambiguous semantics
- ❌ RVec element widening: exact match only (future phase)

**Implementation:**
```python
# ir_builder.py changes:
CONVERSION_MATRIX: Dict[Tuple[IRTypeKind, IRTypeKind], int]  # 11×11 table
_select_overload()           # Ranked selection algorithm
_compute_conversion_rank()   # Per-candidate scoring
_conversion_rank()           # Per-argument rank lookup
_explain_conversion_failure() # Human-readable errors
```

**Example:**
```python
# Register overloads:
dsl.register_function_cpp('double f(int x) { return x * 2.0; }')
dsl.register_function_cpp('double f(double x) { return x * 3.0; }')

# Use with float32:
dsl.define("result", "f(my_float32)")  
# → Chooses f(double) via Float32→Float64 (rank 1)
# → Better than no match (would fail)
```

**Key Decisions:**
- **Q1:** Widening allowed? → YES (Float32→Float64, Int8→Int64)
- **Q2:** Int→Float32? → NO FORBIDDEN (precision loss for large values)
- **Q3:** Same rank candidates? → ERROR (ambiguity)

**Tests:** +21 new + 2 updated  
**Total:** 1467 passed, 30 skipped  
**Specification:** PHASE_13_5_D_v08_Proposal.md  
**Reviewers:** Claude-Opus-4.5, Gemini2, GPT3, GPT4, GPT6, Claude-Sonnet-4.5

---

### Phase 13.5.C: DSL Integration for Registered Functions
**Commit:** 7eed1e9a (Jan 13, 2026)  
**Goal:** Enable registered C++ functions in `dsl.define()` expressions with overload resolution

**Deliverables:**
- `dsl.define('pt_col', 'pt(px, py)')` now works with registered functions
- Overload resolution by (rank, kind) exact matching (no widening in v0.5)
- Multiple overloads per function name supported
- `define_raw()` escape hatch for complex C++ expressions
- `is_raw` flag on GeneratedFunction for raw expressions
- Zero-parameter function support

**Implementation Changes:**
```python
# ir_builder.py:
_custom_functions: Dict[str, List[Dict]]  # Now List for overloads
register_function()      # Requires param_types for resolution
_select_overload()       # Filters by arity, then (rank, kind)
_signature_matches()     # Exact (rank, kind) matching

# dsl_compiler.py:
_register_function_for_dsl()              # Stores (rank, kind) per param
_cpp_type_to_rank_kind()                  # Type mapping
_register_custom_functions_with_builder() # IRBuilder integration
define_raw()                              # Escape hatch with guardrails
```

**Example:**
```python
# Register scalar and vector overloads:
dsl.register_function_cpp('''
    double pt(double px, double py) {
        return sqrt(px*px + py*py);
    }
''')

dsl.register_function_cpp('''
    RVec<double> pt(const RVec<double>& px, const RVec<double>& py) {
        return sqrt(px*px + py*py);
    }
''')

# Use in DSL:
dsl.define("track_pt", "pt(px, py)")  # Selects correct overload based on arg types
```

**Key Rules (v0.5):**
- Exact (rank, kind) matching: int32 ≠ int64, float32 ≠ float64
- No numeric widening (added in Phase 13.5.D)
- Lambda expressions FORBIDDEN (FROZEN RULE #1)
- Latest registration wins for identical signatures

**Tests:** +18 (OV1-OV10: overloads, AC1-AC3: acceptance, DR1-DR3: define_raw, VAL1-VAL3: validation)  
**Total:** 1439 passed, 37 skipped  
**Specification:** PHASE_13_5_C_v05_Proposal.md  
**Reviewers:** Gemini2, GPT3 (Arch), GPT3 (Team2), GPT6 (5/5 unanimous approval)

---

### Phase 13.5.B: C++ Function Registration API
**Commit:** 06ddc3c5 (Jan 11, 2026)  
**Goal:** Enable registration of user-defined C++ functions for use in DSL expressions

**Deliverables:**
- `register_function_cpp()` — Register C++ function with automatic compilation
- `get_registered_function()` — Query registration details
- `list_registered_functions()` — List all registered functions
- Thread-safe declaration with class-level lock (protects ROOT's global interpreter)
- Lambda rejection enforced (FROZEN RULE #1)
- Hash-based naming: `dsl_<name>_<hash16>` (deterministic, collision-resistant)

**Implementation:**
```python
# dsl_compiler.py:
def register_function_cpp(self, cpp_code: str, headers=None, pragmas=None, name=None):
    """
    Register C++ function for use in DSL expressions.
    
    Example:
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''', headers=["<cmath>"])
    """
    # Parse function signature
    # Generate hash from code (deterministic)
    # Compile via gInterpreter with thread-safe lock
    # Store in registry for later use
```

**Key Features:**
- **Hash naming:** `dsl_pt_a1b2c3d4e5f6` (name + 16-char hash)
- **Thread safety:** Class-level `threading.RLock()` protects ROOT's gInterpreter
- **Header auto-detection:** Common headers (<cmath>, <vector>, etc.) added automatically
- **Registry persistence:** Functions survive across DSL instances (process-global ROOT state)
- **Lambda rejection:** FROZEN RULE #1 enforced at registration time

**Exploration Tests (40 total, T1-T41):**
- T1-T6: ACLiC basics, thread safety, hash determinism
- T7-T14: Macro loading, pragma handling, Cling redeclaration semantics
- T15-T32: Thread safety under ImplicitMT, parser coverage, overload resolution
- T33-T41: Header contracts, I/O snapshots, complex types (TLorentzVector)

**Production Tests:** +38 tests  
**Total:** 1387/1388 passed (1 pre-existing test_draw_integration failure)  
**Specification:** PHASE_13_5_B_v05_Proposal.md  
**Reviewers:** GPT-4, GPT5, GPT6, Gemini2, Claude Opus 4.5, Claude Sonnet 4.5 (7/8 approved, 1 with non-blocking comments)

**Next:** Phase 13.5.C (DSL Integration)

---

### Phase 12.6.DSL: to_aliasdf() Export
**Commit:** 78135bc (Dec 16, 2025)  
**Goal:** Enable workflow migration from RDataFrameDSL to AliasDataFrame

**Deliverables:**
- `to_aliasdf()` method for schema export
- `get_definitions()` helper method
- C++ to Python operator conversion with correct precedence:
  - `&&` → `&` (with parentheses)
  - `||` → `|` (with parentheses)
  - `!` → `~` (preserving `!=`)
- Warnings for unconvertible expressions (TMath, ROOT namespace)
- Include/exclude filters
- dtype_map support

**API:**
```python
schema = dsl.to_aliasdf(
    include=['pt_gev', 'good_track'],
    exclude=['debug_var'],
    dtype_map={'pt_gev': 'float32'}
)

# Returns:
{
    'columns': {
        'pt_gev': {'expr': 'trackPt / 1000', 'dtype': 'float32'},
        'good_track': {'expr': '(trackPt > 0.5) & (nHits > 5)'}
    },
    '__meta__': {
        'source': 'RDataFrameDSL',
        'export_version': '1.0'
    }
}
```

**Key Bug Fixed:** Mixed `&&`/`||` precedence — now splits `||` first, then `&&` (C++ semantics)

**Tests:** +20 tests

---

### Phase 12.5.DSL: Statistical Annotations
**Commit:** 78135bc (Dec 16, 2025)  
**Goal:** Add QA validation annotations to pull distribution plots

**Deliverables:**
- `show_statistics` parameter for μ, σ, n stats box
- `show_expected` parameter for N(0,1) Gaussian overlay
- Auto-detect pull distributions via `'pull' in expr.lower()`
- Per-plot `is_pull` override in plot_spec
- Graceful fallback for older dfdraw versions

**API:**
```python
results = dsl.draw_figures(
    specs, rdf,
    show_statistics=True,   # Add μ, σ, n stats box
    show_expected=True,     # Add N(0,1) Gaussian overlay
)

# Per-plot override
{'expr': 'my_residual', 'is_pull': True}   # Force as pull
```

**Tests:** +6 tests

---

### Phase 12.3: Composed Canvas
**Commit:** Prior to Dec 14, 2025  
**Goal:** Multi-subplot figure generation

**Deliverables:**
- Grid layout specification
- Subplot configuration per plot_spec
- Figure-level styling options

**Tests:** +29 tests

---

### Phase 12.2: RVec Selection for draw_figures
**Commit:** Prior to Dec 14, 2025  
**Goal:** Enable RVec column plotting in draw_figures

**Deliverables:**
- Automatic RVec detection and flattening for histograms
- RVec element selection via index
- Support for jagged array visualization

**Tests:** +33 tests

---

### Phase 12.1: dfdraw Integration
**Commit:** Prior to Dec 14, 2025  
**Goal:** Integrate RDataFrameDSL with dfdraw plotting library

**Deliverables:**
- `draw_figures()` method for batch plotting from DSL definitions
- Integration with DFDraw class
- Support for histogram, scatter, and profile plots

**Tests:** +38 tests

---

### Phase 13.2.DSL: ROOT ↔ Arrow Bridge
**Commit:** afda6fb (Dec 16, 2025)  
**Goal:** Enable zero-copy data transfer between ROOT RDataFrame and PyArrow

**Deliverables:**
- `to_arrow()` method for RDataFrame → PyArrow Table export
- `from_arrow()` classmethod for PyArrow Table → DSLCompiler import
- RVec → ListArray conversion (preserves jagged structure)
- RVec flatten option for aggregate analysis
- DSL schema embedded in Arrow metadata for round-trip
- Arrow → C++ type inference (`_arrow_type_to_ctype`)
- Best-effort Python ↔ C++ expression conversion
- Memory warning for large RVec materialization (>1M events)

**API:**
```python
# Export to Arrow
table = dsl.to_arrow(
    rdf=rdf,
    columns=['pt', 'eta'],
    flatten_rvec=False,      # Keep as ListArray
    include_schema=True       # Embed schema in metadata
)

# Import from Arrow
new_dsl = DSLCompiler.from_arrow(table, apply_schema=True)

# Round-trip with AliasDataFrame
table = dsl.to_arrow(include_schema=True)
adf = AliasDataFrame(table=table, backend='pyarrow')
```

**Implementation Notes:**
- Phase 1: Uses numpy as intermediate layer (copy-based)
- Future: Direct Arrow IPC when ROOT supports it
- Requires: pyarrow>=12.0 (optional dependency)

**Tests:** +31 tests

---

## Phase 13 — Zero-Fragmentation Memory Architecture

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

### Key Architecture Decision

**Hybrid Design:** Arrow for storage/transport, NumPy/Pandas for compute

- Phase 9 evidence: PyArrow eval is 8-10× slower than NumPy
- Resolution: Use Arrow only for storage, scatter/gather, and sort
- Compute remains in NumPy/Pandas (proven fast)

---

## Earlier Phases (Reference)

### Phase 1-7.9: Core DSL Implementation
See original phase history for details on:
- IR Core (Phase 1)
- Type Inference (Phase 2)
- IRBuilder (Phase 3)
- Class Reflection (Phase 4)
- C++ Code Generation (Phase 5)
- Object Methods & Properties (Phase 6a)
- RVec Operations (Phase 6b)
- Private Member Reflection (Phase 6c)
- ROOT Integration Validation (Phase 6.9)
- RVec Slicing & Masking (Phase 7)
- DSLCompiler & RDF Stress Tests (Phase 7.9)

### Phase 8: Method Broadcasting
**Goal:** Element-wise method calls on RVec<Object>

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

## Review Process

Each phase follows this workflow:

1. **Design Review Request** — Detailed proposal with questions
2. **Multi-Reviewer Consensus** — GPT, Gemini, Claude review
3. **Coder Instructions** — Detailed implementation guide
4. **Implementation** — Following instructions exactly
5. **Test Validation** — All tests must pass
6. **Owner Approval** — Final sign-off before merge

**Active Reviewers (Phases 13.5+):**
- Claude Opus 4.5 (Architecture Lead, Proposal Author)
- Claude Sonnet 4.5 (Architecture Support, Technical Reviewer)
- GPT-5.2 Thinking (Detailed Technical Analysis)
- GPT3 (Architecture & Team2)
- GPT4, GPT5, GPT6 (Implementation Reviews)
- Gemini2 (RDataFrameDSL Domain Expert)

**Approval Requirement:** Unanimous consent from all reviewers before commit.

---

## Key Files

| File | Purpose |
|------|---------|
| `dsl_compiler.py` | Main DSLCompiler class with all methods |
| `ir_builder.py` | IR builder with overload resolution |
| `flatten.py` | Flattening backends (NumPy, Awkward, C++) |
| `join_utils.py` | Join strategy (JoinPlan, join_dataframes, broadcast) |
| `tests/test_register_function_cpp.py` | Phase 13.5.B tests |
| `tests/test_phase_13_5_c.py` | Phase 13.5.C tests |
| `tests/test_phase_13_5_d.py` | Phase 13.5.D tests |
| `tests/test_flatten.py` | Phase 13.6.A tests |
| `tests/test_join_utils.py` | Phase 13.6.C join tests (Tier 1) |
| `tests/test_flatten_join.py` | Phase 13.6.C flatten+join tests (Tier 1) |
| `tests/test_join_e2e.py` | Phase 13.6.C end-to-end tests (Tier 3) |
| `tests/test_d9_integration.py` | Phase 13.6.C C-array integration tests |
| `tests/test_invariance_nd.py` | Phase 13.6.C N-D invariance tests |
| `tests/exploration/flatten/` | Permanent exploration tests |
| `tests/test_draw_figures_stats.py` | Phase 12.5.DSL tests |
| `tests/test_to_aliasdf.py` | Phase 12.6.DSL tests |
| `tests/test_arrow_export.py` | Phase 13.2.DSL tests |

---

## Schema Format Reference

```python
# DSLCompiler schema (simple format)
schema = {'x': 'double', 'y': 'float', 'n': 'int'}

# AliasDataFrame schema (columns format)
schema = {
    'columns': {'name': {'expr': '...', 'dtype': '...'}},
    '__meta__': {...}
}
```

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Original | Phases 1-8 |
| 2.0 | Dec 16, 2025 | Added Phases 12.x and 13.2.DSL |
| 3.0 | Jan 13, 2026 | Added Phases 13.5.B/C/D and 13.6.A |
| 4.0 | Jan 20, 2026 | Added Phase 13.6.C (N-D Slicing & Join Strategy) |
