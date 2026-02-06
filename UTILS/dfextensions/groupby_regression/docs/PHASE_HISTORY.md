# Phase History — GroupBy Regression

## Overview

The GroupBy Regression module provides high-performance grouped linear regression for CERN/ALICE particle physics calibration workflows. Development follows a phased approach with multi-LLM architecture review before each implementation.

**Scale:** 25M fits in production, 100K+ groups, strict memory constraints per core.

---

## Phase Summary

| Phase | Name | Date | Status |
|-------|------|------|--------|
| — | Package Structure | Oct 25, 2025 | ✅ Complete |
| — | v2/v3/v4 Engines | Oct 25, 2025 | ✅ Complete |
| 12.8.GB | Batch Fitting (v5) | Dec 19, 2025 | ✅ Complete |
| 12.9.GB | Numba Parallel Kernel | Dec 20, 2025 | ✅ Complete |
| 12.10.BF | Benchmark Framework | Dec 23-25, 2025 | ✅ Complete |
| 12.11 | n_jobs=1 Fix + Profiling | Dec 26, 2025 | ✅ Complete |
| 13.1.GB | PyArrow Backend | Dec 16, 2025 | ✅ Complete |
| 12.14.GB | Shared Numba Kernel | Dec 31, 2025 | ✅ Complete |
| 12.14a.GB | Test Infrastructure | Dec 31, 2025 | ✅ Complete |
| **12.14b.GB** | **BF Integration** | **Jan 1, 2026** | ✅ Complete |
| **12.14b.GB-add** | **Dual Timing + cProfile** | **Jan 1, 2026** | ✅ Complete |
| **12.14c.GB** | **ADF Visualization** | **Jan 3-4, 2026** | ✅ Complete |
| 12.15.GB | V4 Integration | — | 📋 Planned |

**Current Test Count:** 145 benchmark tests + existing kernel/module suites

---

## Critical Incident: Silent Numba Regression (Nov 2025)

### What Happened

On **November 14, 2025** (commit `db0eb019`), the V4 Numba JIT kernel was accidentally removed during metadata refactoring. This caused a **10× performance regression** that went undetected for **6 weeks**.

### Why It Wasn't Caught

- Unit tests passed (correctness OK)
- No Numba/NumPy parity tests existed
- No performance regression tests in CI
- No memory benchmarks

### Resolution

Phases 12.14.GB through 12.14c.GB address this with:
- Shared kernel module (prevents accidental deletion)
- Numba/NumPy parity tests (catches silent regressions)
- Performance gates (ratio-based, CI-friendly)
- Memory stability tests (streaming validation)
- Benchmark framework with historical tracking and visualization

---

## Recent Phases (Jan 2026)

### Phase 12.14c.GB: AliasDataFrame Visualization (Jan 3-4, 2026)
**Commits:** `5ffec1f8`, `95119b27`, `43ce0895`, `c321a6ec`  
**Goal:** Visual analysis tools for benchmark history using AliasDataFrame

**Deliverables:**

| ID | Component | Description |
|----|-----------|-------------|
| Step 0 | ADF Smoke Test | 16 tests validating AliasDataFrame API |
| D6 | cProfile Fix | Skip profiling for n_jobs > 1 (multiprocessing) |
| D1 | ADF Loader | `load_benchmark_adf()` with TopCPU/TopMemory subframes |
| D2 | Specs YAML | `benchmark_specs.yaml` with plot configurations |
| D3 | CLI Commands | `--history`, `--history-stats`, `--plot DIR` |
| D5 | Documentation | README.md with CV% interpretation guide |

**Key Features:**

```python
# Load benchmark history as AliasDataFrame
from dfextensions.benchmarks.benchmark_adf import (
    load_benchmark_adf,
    compute_benchmark_statistics,
)

adf = load_benchmark_adf("groupby_regression", max_runs=20)
print(adf.subframes['TopCPU'].df.head())  # CPU profile data
print(adf.subframes['TopMemory'].df.head())  # Memory stats

# Noise analysis
stats = compute_benchmark_statistics("groupby_regression", baseline="7d")
print(stats[['benchmark_id', 'mean_time_s', 'cv_pct', 'high_noise']])
```

**CLI Commands:**

```bash
# Benchmark history summary
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history

# Noise statistics (CV%) for alarm tuning
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history-stats

# Generate trend plots
python -m dfextensions.benchmarks.runner --subproject groupby_regression --plot ./plots/
```

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No data found |
| 2 | Partial success (some plots failed) |
| 3 | Dependency missing (matplotlib/PyYAML) |

**Backward Compatibility Fix:**
- `wall_time_s` field made optional with `__post_init__` fallback to `time_s`
- Preserves ability to load pre-12.14b.GB results.json files

**Test Results:** 145 tests passing

**Reviewed by:** Gemini (Architect), GPT-1, GPT-2, Claude-2, Claude-3 (Team 3)

---

### Phase 12.14b.GB-addendum: Benchmark Framework Refinements (Jan 1, 2026)
**Commit:** `c94004cb`  
**Goal:** Fix timing contract and add always-on profiling

**Fixes:**

| Issue | Resolution |
|-------|------------|
| D1: Timing contract | Dual timing: `time_s` (kernel) + `wall_time_s` (wrapper) |
| D2: ID hygiene | `uses_n_jobs` flag; kernel/memory IDs exclude n_jobs=1 |
| D3: Profile storage | Always-on cProfile; `--no-profile` flag for CI |

**Timing Contract:**
```
wall_time_s  ←  Total benchmark time (includes setup, warmup)
time_s       ←  Kernel-only time (excludes overhead)

Ratio: ~80× for kernel benchmarks (wall includes JIT, setup)
```

**Reviewed by:** GPT-1, GPT-3, GPT-4, Gemini, Claude-2, Main Architect

---

### Phase 12.14b.GB: Benchmark Framework Integration (Jan 1, 2026)
**Commits:** `5c84b42e`, `72a0ece2`  
**Goal:** Integrate kernel and memory benchmarks with BF runner

**Features:**
- Multi-source discovery: v5 + kernel + memory benchmarks
- `NumpyEncoder` for `np.bool_` JSON serialization
- `run_benchmarks.sh --review` for artifact generation
- 18/18 benchmarks passing via BF

**Files Added/Modified:**
- `benchmarks/runner.py`: Multi-source discovery
- `benchmarks/schema.py`: NumpyEncoder
- `groupby_regression/benchmarks/bench_*.py`: BF adapter functions
- `groupby_regression/benchmarks/run_benchmarks.sh`: Review script
- `groupby_regression/benchmarks/README.md`: Documentation

**Reviewed by:** GPT-3, GPT-4, Gemini, Claude (Architect)

---

## December 2025 Phases

### Phase 12.14a.GB: Test Infrastructure (Dec 31, 2025)
**Commit:** `29e34be2`  
**Goal:** Comprehensive test suite that would have caught the 1.5-month regression

**Deliverables:**
- 28 tests covering correctness, parity, performance, memory
- Numba/NumPy parity tests (rtol=1e-12, atol=1e-14)
- Large-scale tests (50K groups, 5M points)
- Streaming memory stability tests (50 chunks, RSS growth < 10%)

**Test Categories:**

| Category | Tests | Purpose |
|----------|-------|---------|
| Status decode | 4 | Bitmask encode/decode |
| Single-fit correctness | 5 | Basic, weighted, NaN handling |
| Multi-fit correctness | 1 | Multi-target validation |
| Kernel parity | 1 | Single vs multi-fit equivalence |
| Dispatcher | 3 | Kernel selection logic |
| MC true validation | 4 | Fit vs theory coefficients |
| **Numba/NumPy parity** | **2** | **P0: Catches silent regressions** |
| **Large-scale** | **2** | **50K groups, 5M points** |
| **Streaming memory** | **2** | **RSS stability over chunks** |
| Performance | 2 | Speedup gates |
| Structural JIT | 2 | Numba compilation verification |

**Validation Results:**
```
Numba/NumPy parity: max diff = 3.15e-14 ✓
Large-scale: 99.9%+ within 5σ tolerance ✓
Streaming: RSS growth < 10% over 50 chunks ✓
Performance: 12.4× speedup ✓
```

**Reviewed by:** Gemini, GPT-3, GPT-4, Claude Architects

---

### Phase 12.14.GB: Shared Numba Kernel Module (Dec 31, 2025)
**Commit:** `b7301bae`  
**Goal:** Restore V4 Numba performance with shared kernel architecture

**Problem Addressed:**
- V4 Numba kernel accidentally deleted (commit `db0eb019`, Nov 14)
- 10× performance regression undetected for 6 weeks
- Phase 12.12b.1 discovered regression during benchmarking

**Solution — Dual Kernel Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│  groupby_regression_kernels.py (~900 lines)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  fit_groups_single_numba()     — Single-fit kernel (baseline)   │
│  fit_groups_multifit_numba()   — Multi-fit kernel (XtX sharing) │
│  fit_groups_dispatch()         — Kernel selector                │
│                                                                 │
│  Status bitmask constants (per v2.1 spec)                       │
│  NumPy fallback when Numba unavailable                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Performance (Mac ARM M-series):**

| Metric | Result |
|--------|--------|
| Numba vs NumPy | 8.7-63.7× speedup |
| Multi-fit vs Single×N | 2.3-5.8× speedup |
| Peak throughput | 6.3M groups/sec |

**Files Added:**
- `groupby_regression_kernels.py` (~900 lines)
- `tests/test_groupby_regression_kernels.py` (~800 lines initially)
- `benchmarks/bench_groupby_regression_kernels.py` (~350 lines)

**Reviewed by:** Gemini, GPT-3, GPT-4, Claude Architects

---

### Phase 12.11: Numba Bypass Fix + Profiling (Dec 26, 2025)
**Commits:** `6524fbf0`, `933a054c`  
**Goal:** Fix n_jobs=1 falling back to sequential Python

**Bug Fixed:**
- `_select_parallel_backend()` now uses Numba regardless of n_jobs
- Previously n_jobs=1 fell back to sequential Python (bypassing JIT)

**Profiling Benchmark Added:**
- `bench_groupby_fits.py` with CPU profiling integration
- Supports `--scenarios`, `--engines`, `--n-jobs`
- Outputs `profiles/` with `.prof`, `.txt`, `_cpu.json`

---

### Phase 12.10.BF: Benchmark Framework (Dec 23-25, 2025)
**Commits:** `35281e84`, `0e69568c`  
**Goal:** Foundational benchmark infrastructure

**Deliverables:**
- `schema.py`: JSON schema with env_id (np+nb versions), validation
- `profiler.py`: Cross-platform RSS (Mac/Linux), tracemalloc, memory guards
- `runner.py`: CLI with warmup=2, memory threshold=15%, exit codes 0/1/2
- `scenarios.py`: S1-S5 with ×2 scaling, S5 matches ALICE TPC calibration
- `bench_v5.py`: v5_batch_fit benchmark with JIT warmup tracking

**Test Results:** 33 tests passing

**Reviewed by:** Claude, GPT-1, GPT-2, Gemini, Team 3

---

### Phase 12.9.GB: Numba Parallel Kernel for V5 (Dec 20, 2025)
**Commit:** `be4deda8`  
**Goal:** Add Numba prange parallelization within chunks

**Performance:**
- 23-35× speedup with n_jobs=2-4 (100K rows benchmark)
- Streaming XtX/XtY accumulation (no per-group allocations)
- Cholesky-based solve with condition proxy

**Breaking Change:**
- Per-fit diagnostics replace shared columns
- `diag_n_total_v5` remains shared (1D)
- `diag_n_valid{suffix}`, `diag_n_filtered{suffix}` now per-fit

**API:**
```python
# New parallel_backend parameter
result = make_parallel_fit_v5(
    df, fit_specs,
    parallel_backend='auto'|'numba'|'sequential'
)
```

**Tests:** +20 new tests, full suite 262 passed

**Reviewed by:** Gemini, GPT-1, GPT-2, Claude — all approved

---

### Phase 12.8.GB: Batch Fitting (V5) (Dec 19, 2025)
**Commit:** `4f49119c`  
**Goal:** Multiple fits sharing same groupby in one call

**Features:**
- Single sort (vs N sorts for N separate v4 calls)
- Chunked processing: peak memory O(N/chunks)
- Auto-generated metadata with prediction/residual/pull formulas
- Per-fit: different targets, linear_columns, weights, suffixes

**API:**
```python
result = make_parallel_fit_v5(
    df,
    fit_specs=[
        {'target': 'dX', 'linear_columns': ['a', 'b'], 'suffix': '_dX'},
        {'target': 'dY', 'linear_columns': ['a', 'b'], 'suffix': '_dY'},
    ],
    gb_columns=['sector', 'pad'],
    chunk_size=10000,
)
```

**Tests:** +33 new tests (v4 parity, chunk invariance, validation)

**Reviewed by:** Gemini, GPT-1, GPT-2, Architect — all approved

---

### Phase 13.1.GB: PyArrow Backend (Dec 16, 2025)
**Commit:** `bb29a647`  
**Goal:** Memory-efficient sorting without changing compute logic

**New Parameters:**
```python
make_parallel_fit_v4(
    ...,
    backend='pandas'|'pyarrow'|'auto',  # default: 'auto'
    pyarrow_threshold=1_000_000,         # rows before auto-switch
)
```

**Key Decision:** Hybrid design — Arrow for storage/transport, NumPy for compute
- Phase 9 evidence: PyArrow eval is 8-10× slower than NumPy
- Resolution: Use Arrow only for storage, scatter/gather, and sort

---

## Earlier Phases (Oct 2025)

### Package Restructuring (Oct 25, 2025)
**Commit:** `e43f332e`

Moved files to package structure with `git mv` (preserves history):
```
groupby_regression/
├── __init__.py
├── groupby_regression.py
├── groupby_regression_optimized.py
├── groupby_regression_kernels.py      # Added Dec 31
├── tests/
│   ├── test_groupby_regression.py
│   ├── test_groupby_regression_optimized.py
│   └── test_groupby_regression_kernels.py  # Added Dec 31
├── benchmarks/
│   ├── bench_groupby_regression.py
│   ├── bench_groupby_regression_kernels.py  # Added Dec 31
│   └── run_benchmarks.sh                    # Added Jan 1
└── docs/
    └── PHASE_HISTORY.md
```

### v2/v3/v4 Engine Benchmarks (Oct 25, 2025)
**Commits:** `94386df6`, `ba768f67`

**Engine Comparison:**

| Engine | Method | Speed |
|--------|--------|-------|
| v2 | loky (process-based) | 2.5k-15k groups/s |
| v3 | threads | 2.5k-15k groups/s |
| v4 | Numba JIT | 450k-1.8M groups/s |

**Key Finding:** v4 is 75-264× faster than v2/v3

---

## Specifications

Two specifications govern Phase 12.14:

### PHASE_12_14_V4_NUMBA_RESTORATION_SPEC_v2.1.md
- Problem statement and root cause analysis
- Dual kernel architecture (Option D1)
- Kernel interface specification (frozen for Phase 12.x)
- Performance gates (ratio-based)
- Process gates (inspect_types, deletion-focused diff)

### PHASE_12_14_STATUS_BITMASK_SPEC_v2.1.md
- Status bit definitions (uint8 bitmask)
- Three invalid_handling modes: `assume_clean`, `detect`, `filter`
- Per-target storage for V5 heterogeneous fits
- Output filling rules (truth table)

---

## Key Technical Decisions

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Dual kernels (single + multi) | Multi-fit shares XtX when no filtering | 12.14.GB |
| Filter mode → single-fit only | Different valid rows per target breaks XtX sharing | 12.14.GB |
| Per-target status bitmask | V5 heterogeneous fits need per-fit diagnostics | 12.14.GB |
| Ridge stabilization at κ > 1e10 | Balance between stability and accuracy | 12.14.GB |
| Ratio-based performance gates | Absolute thresholds fail on different CI machines | 12.14a.GB |
| Numba/NumPy parity tests | Would have caught 6-week silent regression | 12.14a.GB |
| Hybrid Arrow/NumPy | Arrow for storage, NumPy for compute (8-10× faster) | 13.1.GB |
| Dual timing (time_s + wall_time_s) | Separate kernel time from wrapper overhead | 12.14b.GB |
| Always-on cProfile | Historical profiles for bottleneck analysis | 12.14b.GB-add |
| Skip cProfile for n_jobs > 1 | cProfile only captures main process | 12.14c.GB D6 |
| wall_time_s optional fallback | Backward compat with pre-12.14b.GB data | 12.14c.GB D1 |

---

## Review Process

Each phase follows this workflow:

1. **Specification** — Main Architect drafts detailed spec
2. **Multi-LLM Review** — GPT, Gemini, Claude architects review
3. **Consensus** — Unanimous approval required (blocking items resolved)
4. **Implementation** — Team 3 Coder implements per spec
5. **Code Review** — Architects verify implementation matches spec
6. **Validation** — All tests pass, performance gates met
7. **Commit** — After all approvals

**Architecture Teams:**

| Team | Role |
|------|------|
| Main Architect (Claude) | Primary design, specifications |
| Second Claude Architect | Independent review |
| GPT Architects (1-4) | Interface design, numerical correctness |
| Gemini Architect | Process rigor, HPC requirements |
| Team 3 Coder | Implementation |

**Approval Requirement:** Unanimous consent from all reviewers before commit.

---

## Key Files

| File | Purpose |
|------|---------|
| `groupby_regression_kernels.py` | Shared Numba kernel module |
| `groupby_regression_optimized.py` | V4/V5 implementations |
| `tests/test_groupby_regression_kernels.py` | 28 kernel tests |
| `benchmarks/runner.py` | BF runner with multi-source discovery |
| `benchmarks/schema.py` | JSON schema + NumpyEncoder |
| `benchmarks/benchmark_adf.py` | AliasDataFrame adapter |
| `benchmarks/visualization_cli.py` | --history, --history-stats, --plot |
| `benchmarks/specs/benchmark_specs.yaml` | Plot specifications |
| `groupby_regression/benchmarks/run_benchmarks.sh` | Review artifact generator |

---

## Planned Phases

### Phase 12.15.GB: V4 Integration
- Update `make_parallel_fit_v4()` to call shared kernel
- Expose `invalid_handling` parameter
- Preserve backward compatibility
- Add integration tests

### Phase 12.16.GB: V5 Integration (Potential)
- Update V5 to use shared kernel
- Handle heterogeneous `linear_columns` in wrapper

---

## Performance Reference

### V4 Numba Kernel (Phase 12.14.GB)

| Scenario | Groups/sec | Numba/NumPy |
|----------|------------|-------------|
| 100 groups, 20 rows | 308,246 | 9.7× |
| 1K groups, 20 rows | 526,582 | 8.7× |
| 10K groups, 20 rows | 6,319,447 | 63.7× |
| 1K groups, 6 targets | 3,584,756 | 36.7× |

### Multi-fit Speedup

| Targets | Multi vs Single×N |
|---------|-------------------|
| 2 | 2.28× |
| 4 | 3.71× |
| 6 | 4.34× |
| 6 (4 feat) | 5.79× |

### Benchmark Framework (Phase 12.14c.GB)

| Metric | Value |
|--------|-------|
| Total benchmarks | 18 |
| Test count | 145 |
| CLI commands | 3 (--history, --history-stats, --plot) |
| Exit codes | 4 (0/1/2/3) |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Dec 16, 2025 | Initial version |
| 2.0 | Dec 31, 2025 | Added Phases 12.14.GB, 12.14a.GB, incident analysis |
| 3.0 | Jan 4, 2026 | Added Phases 12.14b.GB, 12.14b.GB-addendum, 12.14c.GB |
