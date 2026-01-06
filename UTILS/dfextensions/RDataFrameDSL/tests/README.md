# RDataFrameDSL Tests

**Version:** 1.0 (V1 Baseline)  
**Date:** 2026-01-06  
**Status:** ✅ Approved

---

## Overview

| Metric | Value |
|--------|-------|
| Test Files | 28 |
| Test Classes | 251 |
| Total Tests | 1,054 |
| Infrastructure Files | 3 |
| Phases Covered | 15+ |

---

## Quick Start

```bash
# Run all tests (requires ROOT)
pytest -v

# Run unit tests only (no ROOT required)
pytest -v -m "not requires_root"

# Run specific phase
pytest -v test_dsl_invariants.py  # Phase 13.2.2

# Run with parallelization
pytest -v -n auto

# Show skip reasons
pytest -rs
```

---

## Test Categories

### Unit Tests (No ROOT Required) — 249 tests

| File | Tests | Purpose |
|------|-------|---------|
| test_ir_core.py | 78 | IR node types, type promotion |
| test_ir_builder.py | 89 | AST → IR transformation |
| test_reflection.py | 41 | Method/property reflection |
| test_type_inference.py | 41 | Type extraction, normalization |

### Code Generation Tests — 210 tests

| File | Tests | Purpose |
|------|-------|---------|
| test_backend_cpp.py | 79 | C++ code generation (core) |
| test_backend_cpp_broadcast.py | 39 | Method broadcasting |
| test_backend_cpp_objects.py | 39 | Object method calls |
| test_backend_cpp_rvec.py | 53 | RVec operations |

### Integration Tests (Requires ROOT) — 131 tests

| File | Tests | Purpose |
|------|-------|---------|
| test_root_integration.py | 45 | ROOT compilation |
| test_root_broadcast_integration.py | 16 | Broadcast execution |
| test_rdataframe_integration_advanced.py | 19 | RDF pipeline |
| test_dsl_invariants.py | 37 | Invariant verification |
| test_arrow_bridge.py | 14 | Arrow export/import |

### Feature Tests — 464 tests

| File | Tests | Phase | Purpose |
|------|-------|-------|---------|
| test_rvec_reductions.py | 71 | 10.5 | Sum, Mean, Min, Max |
| test_rvec_filter.py | 28 | 10.7 | Boolean masking |
| test_namespace_calls.py | 33 | 11.1 | TMath.Sin(), etc. |
| test_namespace_broadcast.py | 41 | 11.1 | Namespace + RVec |
| test_namespace_dynamic.py | 44 | 11.1 | Dynamic namespace |
| test_draw_integration.py | 38 | 12.1 | Draw method |
| test_rvec_selection.py | 33 | 12.2 | Take, Range, Where |
| test_draw_figures.py | 29 | 12.3 | Figure generation |
| test_arrow_export.py | 31 | 13.2 | Arrow round-trip |
| test_generator_sanity.py | 34 | 13.2.1 | Generator correctness |
| test_dsl_invariants.py | 37 | 13.2.2 | DSL invariants |
| test_arrow_bridge.py | 14 | 13.2.3 | Arrow fail-closed |

---

## Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `conftest.py` | Fixtures, markers, helpers |
| `invariant_schema.py` | Schema constants, tolerances |

---

## Pytest Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.future` | Phase 7+ techniques |
| `@pytest.mark.requires_root` | Needs ROOT |
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.invariant` | Invariant correctness |

---

## Key Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `synthetic_scalar_rdf` | function | Simple scalar RDataFrame |
| `synthetic_track_cluster_rdf` | function | Track→Cluster pattern |
| `invariant_tree_path` | session | Test ROOT file |
| `dsl_compiler` | function | DSLCompiler instance |

---

## Coverage Summary

### Well-Covered ✅

- Scalar arithmetic, boolean, comparison
- 1D RVec: indexing, slicing, masking, reductions
- Method broadcasting (Phase 8)
- Namespace functions (Phase 11)
- Arrow export/import (Phase 13.2)
- Invariant testing (Phase 13.2.x)

### Known Gaps (V2 Roadmap) ❌

| Gap | Phase |
|-----|-------|
| Nested RVec (`RVec<RVec<T>>`) | 13.3.DSL |
| TMatrixD/TVectorD | 13.3.DSL |
| Schema auto-detection | 13.4.DSL |
| C-array branches | 13.4.DSL |
| Entry$, Alt$, MinIf$/MaxIf$ | 13.5.DSL |

---

## Regression Gate

All V1 tutorials must pass before any commit:

```bash
# Run as regression gate
cd examples/
python 01_basic_usage.py
python 02_rvec_operations.py
python 03_method_broadcasting.py
python 04_comparison_dsl_vs_raw.py
python 05_export_macro.py
```

---

## Environment Requirements

| Component | Minimum Version |
|-----------|----------------|
| Python | 3.9+ |
| ROOT | 6.26+ |
| pytest | 7.0+ |
| pyarrow | 12.0+ (for Arrow tests) |
| numpy | 1.20+ |

---

## Adding New Tests

1. **Phase Tests:** Name as `test_<feature>.py`
2. **Use Invariants:** Prefer mathematical invariants over hardcoded values
3. **Mark ROOT-Required:** Add `pytest.importorskip("ROOT")` at top
4. **Document Phase:** Add docstring with phase number

Example:
```python
"""
Phase 13.3.DSL: Nested RVec type inference tests.

Tests for RVec<RVec<T>> handling in DSL compiler.
"""

import pytest
ROOT = pytest.importorskip("ROOT")

class TestNestedRVecInference:
    """Verify nested RVec type propagation."""
    
    def test_inner_type_extraction(self):
        """nested[0] returns RVec<T>, not T."""
        # ...
```

---

## Contact

- **Main Architect:** Marian (ALICE/CERN)
- **Project:** RDataFrameDSL for ALICE O2

---

**End of README**
