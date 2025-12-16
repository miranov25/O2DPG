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
| 13.4 | Integration Testing | - | 🔴 Pending |

**Current Total: 964 tests passing, 1 skipped**

---

## Recent Phases (Team 2 — RDataFrameDSL)

### Phase 12.1: dfdraw Integration
**Commit:** Prior to Dec 14, 2025  
**Goal:** Integrate RDataFrameDSL with dfdraw plotting library

**Deliverables:**
- `draw_figures()` method for batch plotting from DSL definitions
- Integration with DFDraw class
- Support for histogram, scatter, and profile plots

**Tests:** +38 tests

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

### Phase 12.3: Composed Canvas
**Commit:** Prior to Dec 14, 2025  
**Goal:** Multi-subplot figure generation

**Deliverables:**
- Grid layout specification
- Subplot configuration per plot_spec
- Figure-level styling options

**Tests:** +29 tests

---

### Phase 12.5.DSL: Statistical Annotations
**Commit:** Dec 16, 2025 (78135bc)  
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

### Phase 12.6.DSL: to_aliasdf() Export
**Commit:** Dec 16, 2025 (78135bc)  
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

### Phase 13.2.DSL: ROOT ↔ Arrow Bridge
**Commit:** Dec 16, 2025 (afda6fb)  
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

**Reviewers:**
- Gemini (Architecture)
- GPT-1 (Implementation)
- GPT-2 (Testing/Edge Cases)
- Claude-2 (Integration)

**Approval Requirement:** Unanimous consent from all reviewers before commit.

---

## Key Files

| File | Purpose |
|------|---------|
| `dsl_compiler.py` | Main DSLCompiler class with all methods |
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
