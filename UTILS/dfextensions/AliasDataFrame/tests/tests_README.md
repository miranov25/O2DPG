# AliasDataFrame Test Suite

## Overview

**Test Files**: 40 (38 Python + 2 C++)
**Supporting Files**: 4 (conftest.py, __init__.py, run_tests.sh, tests_README.md)
**Total Tests**: ~1183 (as of Phase 6.8e/c)
**Last Updated**: 2025-12-11 (Phase 6.8e/c)

## Running Tests

```bash
# Run all Python tests
pytest tests/ -v

# Run with parallel workers
pytest tests/ -n 12

# Run specific test file
pytest tests/test_lazy_loading.py -v

# Run specific test class/method
pytest tests/test_lazy_loading.py::TestLazyTreeReader::test_load_single_branch -v

# Run C++ tests
cd tests && ./run_tests.sh

# Run only dfdraw-related tests (useful when dfdraw is optional)
pytest tests -k "draw" -v

# Run tests excluding dfdraw (when dfdraw not installed)
pytest tests --ignore=tests/test_draw_invariance.py --ignore=tests/test_draw_chain_integration.py --ignore=tests/test_draw_lazy_integration.py -v
```

---

## Test Files by Category

### Core Functionality (Phase 1-4)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_alias_dataframe.py` | ~80 | 1-2 | Core AliasDataFrame functionality |
| `test_alias_subframe.py` | 22 | 3A | Subframe LEFT JOIN, missing keys |
| `test_data_schema.py` | ~15 | 4 | Schema structure validation |
| `test_alias_data_frame_schema.py` | ~50 | 4a | Unified schema as source of truth |
| `test_alias_data_frame_schema_v2.py` | ~30 | 4a | Schema v2 features |
| `test_schema_serialization.py` | ~55 | 4b | JSON-safe schema serialization |
| `test_schema_export_v2.py` | ~20 | 4b | Export with unified schema |
| `test_schema_definition_vs_record.py` | ~10 | 4 | Schema definition validation |
| `test_compression_pytest.py` | 25 | 2 | Compression monitoring |

### RDataFrame Integration (Phase 5)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_AliasDataFrameRDF.py` | ~150 | 5 | RDataFrame code generation |
| `test_rdf_integration.py` | ~30 | 5 | RDF integration with ROOT |
| `test_rdf_integration_final.py` | ~20 | 5 | Final RDF integration tests |
| `test_rdf_real_data.py` | ~15 | 5 | RDF tests with real physics data |
| `test_composite_keys.py` | ~40 | 5 | Multi-key composite index |
| `test_backend_cpp_rvec.py` | ~25 | 5 | RVec indexing operations |
| `test_dependency_tree.py` | ~15 | 5 | Alias dependency resolution |

### C++ Tests (Phase 3)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_composite_index.C` | 10 | 3 | N-key composite index (ROOT C++) |
| `test_AliasDataFrameTree.C` | 8 | 3A | TTree::Draw compatibility (ROOT C++) |

### dfdraw Integration (Phase 6)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_draw_invariance.py` | 17 | 6.8e | Exact verification: y_derived = 2*x |
| `test_draw_chain_integration.py` | 26 | 6.8c | All Phase 7 loading modes × draw() |
| `test_draw_lazy_integration.py` | 22 | 6/7.3 | Draw with lazy loading |
| `test_ttree_draw_subframe.py` | 10 | 3A | TTree::Draw with friend trees |

### Lazy Loading (Phase 7)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_lazy_loading.py` | 35 | 7.1 | LazyTreeReader foundation |
| `test_branch_detection.py` | 51 | 7.2 | Branch auto-detection from expressions |
| `test_chain_loading.py` | 51 | 7.4 | LazyChainReader multi-file |
| `test_lazy_subframes.py` | 52 | 7.5 | Lazy subframe loading |

### Performance Optimization (Phase 8-9)

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_numba_acceleration.py` | ~30 | 8 | Numba JIT-compiled scatter |
| `test_arrow_compute.py` | ~25 | 9 | PyArrow compute integration |
| `test_arrow_expression.py` | ~20 | 9 | Arrow expression evaluation |
| `test_arrow_scatter.py` | ~25 | 9 | Arrow scatter operations |

### Infrastructure & Utilities

| File | Tests | Phase | Description |
|------|-------|-------|-------------|
| `test_proxy_pattern.py` | 35 | - | DataFrame-like access (`adf['x']`) |
| `test_batch_materialization.py` | ~20 | 2 | Batch alias materialization |
| `test_cycle_detection.py` | ~15 | - | Circular alias detection |
| `test_self_referential_cycles.py` | ~10 | - | Self-referential alias handling |
| `test_fill_handling.py` | ~20 | 3A | NaN fill for missing keys |
| `test_join_caching.py` | ~25 | 8 | Join index caching |
| `test_join_index_caching.py` | ~15 | 8 | Index lookup optimization |
| `test_materialize_subframe_index.py` | ~15 | 5 | Subframe index materialization |
| `test_subframe_alias_api.py` | ~20 | 3 | Subframe alias API |
| `test_profiling.py` | ~10 | 3 | cProfile integration |
| `test_clean_temporary.py` | ~10 | - | Temporary file cleanup |

### Configuration Files

| File | Description |
|------|-------------|
| `conftest.py` | Pytest fixtures and configuration |
| `__init__.py` | Package initialization |
| `run_tests.sh` | Shell script for C++ tests |

---

## Test Categories Summary

| Category | Files | Tests | Phases |
|----------|-------|-------|--------|
| Core Functionality | 9 | ~285 | 1-4 |
| RDataFrame Integration | 6 | ~295 | 5 |
| C++ Tests | 2 | 18 | 3 |
| dfdraw Integration | 4 | 75 | 6, 7.3 |
| Lazy Loading | 4 | 189 | 7 |
| Performance Optimization | 4 | ~100 | 8-9 |
| Infrastructure | 11 | ~195 | Various |
| **Total** | **40** | **~1183** | |

---

## Key Test Invariants

Tests use deterministic relationships for exact verification:

```python
# In generate_synthetic_data.py
y_derived = 2 * x                                    # Exact, no noise
gain[sec] = 1.0 + 0.01 * sec                        # SectorCalib subframe
gain[run,sec] = 1.0 + 0.01*sec + 0.001*(run-1000)   # Calibration chain
```

---

## C++ Test Details

### Composite Index Tests (`test_composite_index.C`) - 10 tests

| # | Test | Type | Description |
|---|------|------|-------------|
| 1 | 3-Key Semantic | Correctness | Join values match formula |
| 2 | 4-Key Sparse | Infrastructure | Real orbit values work |
| 3 | Merge Safety | I/O | hadd doesn't break index |
| 4 | Float Detection | Error handling | Non-integer rejected |
| 5 | Uniqueness | Correctness | All combinations correct |
| 6 | Multiple Subframes | Critical | Different indices work |
| 7 | Index Assertion | Structural | GetTreeIndex() exists |
| 8 | Empty Subframe | Edge case | 0-entry subframe |
| 9 | No Matches | Edge case | Non-overlapping keys |
| 10 | Duplicate Keys | Edge case | Consistent behavior |

### AliasDataFrameTree Tests (`test_AliasDataFrameTree.C`) - 8 tests

Validates TTree::Draw compatibility from C++ side with:
- Friend tree attachment
- Dot notation access
- Expression evaluation
- Cut application

---

## Known Bugs

| Bug ID | Test | File | Status | Description |
|--------|------|------|--------|-------------|
| BUG-2025-12-11 | `test_draw_with_subframe` | `test_draw_chain_integration.py` | ✅ Fixed (6.8a) | draw() passes subframe names to ensure_branches(). Fixed by filtering subframe names. |

Tests marked `@pytest.mark.xfail` are known issues with documented fix phases. When the fix is implemented, remove the xfail marker and verify the test passes.

---

## Writing New Tests

### Python Tests

```python
import pytest
from AliasDataFrame import AliasDataFrame

class TestMyFeature:
    def test_basic_functionality(self, tmp_path):
        """Test description."""
        # Setup
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        # Action
        adf.add_alias('y', 'x * 2')
        
        # Assert
        assert 'y' in adf.aliases
        np.testing.assert_array_equal(adf['y'].values, [2, 4, 6])
```

### C++ Tests

```cpp
// Use assertion macros
ASSERT_TRUE(condition, "message");
ASSERT_EQ(actual, expected, "message");
ASSERT_NEAR(actual, expected, tolerance, "message");

// Tests should:
// 1. Create test ROOT file with known data
// 2. Load via LoadADFTree()
// 3. Verify results with assertions
// 4. Clean up temporary files
```

---

## Test Dependencies

### Dependency Levels

| Dependency | Type | Needed For | Notes |
|------------|------|------------|-------|
| `pytest` | required | All Python tests | Core test runner |
| `pytest-xdist` | optional | Parallel execution | `-n auto` flag |
| `numpy`, `pandas` | required | All tests | Core data structures |
| `uproot` | required | ROOT file tests | Python ROOT I/O |
| `ROOT` | optional | C++ tests, RDF tests | Heavy; tests skip if missing |
| `dfdraw` | optional | Draw integration tests | Often missing in CI; tests skip |
| `numba` | optional | Numba acceleration tests | Tests skip if missing |
| `pyarrow` | optional | Arrow compute tests | Tests skip if missing |

### Installation

```bash
# Install required test dependencies
pip install pytest pytest-xdist numpy pandas uproot

# Optional dependencies (tests skip gracefully if missing)
pip install numba pyarrow

# dfdraw: install from local path or package
# ROOT: install via conda or system package manager
```

---

## CI/CD Integration

```bash
# Full test suite (recommended for CI)
pytest tests/ -n auto --tb=short

# Quick smoke test
pytest tests/test_alias_dataframe.py tests/test_lazy_loading.py -v

# With coverage
pytest tests/ --cov=AliasDataFrame --cov-report=html
```
