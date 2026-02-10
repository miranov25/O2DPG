# groupby_regression — Test Infrastructure

**Phase:** 13.8.GB — v4-aligned Sliding Window API + Invariance Tests  
**Subproject:** groupby_regression (ALICE TPC calibration, O2DPG/dfextensions)

---

## Overview

This test suite uses a **two-tier verification standard** to distinguish features that are truly validated from those that merely don't crash.

| Icon | Status | Meaning |
|------|--------|---------|
| ✅ | **Verified** | All tests pass AND ≥1 invariance/integration test exists |
| ☑️ | **Smoke-only** | All tests pass BUT no test catches numerical regressions |
| 🧨 | **Broken** | At least one test fails |
| ⚠️ | **Partial** | Some tests pass, some skipped/xfail |
| 📋 | **Planned** | In taxonomy, no tests tagged |

## Test Layer Classification

Every test is assigned exactly one layer following the decision tree (apply in order, stop at first match):

| # | Question | Layer |
|---|----------|-------|
| 1 | Uses `pytest.raises` as primary assertion? | `validation` |
| 2 | Compares outputs of TWO code paths? (v4 vs v5, Numba vs NumPy, chunks=1 vs N) | `invariance` |
| 3 | Uses `assert_allclose` against KNOWN TRUE VALUE? (MC, analytic, mathematical property) | `integration` |
| 4 | Checks speedup ratios, RSS limits, timing gates? | `performance` |
| 5 | Checks shapes, columns, types, len()>0, or just runs? | `smoke` |
| 6 | None of above? | `smoke` (fail-closed) |

Only `invariance` and `integration` tests grant ✅ Verified status.

### Edge Case Rules

- **E1:** Zero assertions = `smoke` (regardless of computational sophistication)
- **E2:** Shape/length-only assertions = `smoke`
- **E3:** Performance-only tests do NOT grant ✅ Verified
- **E4:** Mathematical property tests with stated tolerance = `integration`
- **E5:** Structural parity (same column names) ≠ `invariance` (requires numerical comparison)

### Fail-Closed Default

If a test has no layer classification, the generator treats it as `smoke` and emits an UNCLASSIFIED warning. The generator never silently defaults to ✅ Verified.

## File Structure

```
tests/
├── conftest.py                          # Marker registration (feature, layer, slow)
├── feature_taxonomy.py                  # 100 features → proof tests + bench_proof
├── test_layer_classification.py         # ~311 tests → layer assignments
├── README.md                            # This file
│
├── test_groupby_regression.py           # Robust fit (make_parallel_fit)
├── test_groupby_regression_kernels.py   # Low-level Numba/NumPy kernels
├── test_groupby_regression_optimized.py # V2/V3/V4 optimized implementations
├── test_phase_12_8_gb.py                # V5 batch API
├── test_phase_12_9_gb.py                # Backend selection
├── test_cross_validation.py             # Cross-engine parity
├── test_fit_metadata.py                 # Metadata schema/formulas (65 tests)
├── test_groupby_regression_sliding_window.py          # Sliding window (v4-aligned)
├── test_groupby_regression_sliding_window_verbose.py  # Verbose duplicate (deduped)
├── test_invariance_sliding_window.py    # SW invariance: recovery, pulls, parity
├── test_invariance_kernels.py           # Kernel invariance: MC truth, Numba/NumPy
├── _invariance_helpers.py               # Shared analytical validation helpers
├── test_pyarrow_backend.py              # PyArrow backend
└── test_tpc_distortion_recovery.py      # TPC pipeline (zero assertions — smoke)

scripts/
├── generate_capability_matrix.py        # Two-tier matrix generator

docs/
├── CAPABILITY_MATRIX.md                 # Auto-generated (do not edit)
└── capability_matrix.json               # Machine-readable matrix
```

## Running Tests

```bash
# Full run: tests + matrix + reviewer.zip
source run_tests.sh

# Quick: tests only, no matrix
source run_tests.sh quick

# Matrix only (from last test results)
source run_tests.sh matrix

# Invariance/integration tests only
source run_tests.sh invariance

# Control parallelism
PYTEST_WORKERS=4 source run_tests.sh
```

Output goes to `test_logs/` with timestamped filenames. `reviewer.zip` is also created in the project root with all artifacts needed for external review.

## Numba Threading — Troubleshooting

Numba uses a threading layer for `@njit(parallel=True)`. Under **pytest-xdist** (parallel
test workers), each worker initialises its own Numba runtime, which can trigger TBB version
conflicts.

### Symptom

```
ValueError: No threading layer could be loaded.
NumbaWarning: The TBB threading layer requires TBB version 2021 update 6 or later
  i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050.
  The TBB threading layer is disabled.
```

This typically happens on Linux where a system TBB (e.g. from `libtbb-dev`) is version
2021.5 (interface 12050) but Numba requires 2021.6+ (interface 12060).

### Fix Options (pick one)

**Option A — Use OpenMP instead of TBB (recommended, zero install)**
```bash
export NUMBA_THREADING_LAYER=omp
source run_tests.sh                     # ← run_tests.sh defaults to 'omp' since Phase 13.8
```

**Option B — Upgrade TBB via pip**
```bash
pip install tbb --upgrade               # installs TBB ≥ 2021.6 into venv
# On Linux you may also need:
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
```

**Option C — Upgrade system TBB (if you have root)**
```bash
# RHEL/Alma/CentOS:
sudo dnf install tbb-devel              # ≥ 2021.6
# Ubuntu/Debian:
sudo apt install libtbb-dev             # ≥ 2021.6
```

### Verification

```bash
python -c "import numba; numba.config.THREADING_LAYER_PRIORITY"
# or
numba -s | grep -A3 Threading
```

Should show at least one of: `TBB Threading Layer Available : True` or
`OpenMP Threading Layer Available : True`.

### Environment Variable Reference

| Variable | Default | Effect |
|----------|---------|--------|
| `NUMBA_THREADING_LAYER` | `omp` (in `run_tests.sh`) | Which threading backend Numba uses |
| `NUMBA_NUM_THREADS` | (all cores) | Limit Numba parallel threads |
| `PYTEST_WORKERS` | `auto` (in `run_tests.sh`) | pytest-xdist worker count |

## Verbose Test Deduplication

`test_groupby_regression_sliding_window_verbose.py` is a copy of the non-verbose suite with added `vprint()` calls. Both files run in pytest, but for the capability matrix:
- Both suites count as **one** for feature coverage
- The verbose copy cannot independently promote a feature's status

## Benchmark Proof

Some features have benchmark-level validation beyond pytest. These appear in the matrix's "Bench" column:
- **✅ GATED** — benchmark asserts with pass/fail threshold (e.g., `validate_correctness()`)
- **📊 MONITOR** — benchmark computes metrics but only prints, no threshold gate

Benchmark proof is informational. Only gated checks contribute to status.

## Key Metrics

| Metric | Value |
|--------|-------|
| Total features | 100 |
| Total test functions | ~311 |
| Effective unique (excl. verbose) | ~283 |
| Invariance + Integration tests | ~50 |
| ✅ Verified features | 21+ |
| ☑️ Smoke-only features | ~78 |
| 📋 Planned features | 1 (SW.weighted / WLS) |

---

*Phase 13.8.GB — v4-aligned sliding window API, invariance tests, capability matrix updates.*
