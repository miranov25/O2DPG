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
| 12.14b.GB | BF Integration | Jan 1, 2026 | ✅ Complete |
| 12.14b.GB-add | Dual Timing + cProfile | Jan 1, 2026 | ✅ Complete |
| 12.14c.GB | ADF Visualization | Jan 3-4, 2026 | ✅ Complete |
| 13.7.GB | Capability Matrix Infrastructure | Feb 7-8, 2026 | ✅ Complete |
| 13.8.GB | SW API Refactor + Invariance Tests | Feb 9-10, 2026 | ✅ Complete |
| 13.9.GB | V3 Incremental Algorithm | Feb 10, 2026 | ✅ Complete |
| 13.8.SW | Parallel Sliding Window + Benchmarks | Feb 12-14, 2026 | ✅ Complete |
| **13.10.GB** | **Non-Linear SW Fit + Evaluator Extension** | **Feb 16-17, 2026** | ✅ Complete |
| **13.10.GB-T** | **K0s THn Tutorial + MC Truth Validation** | **Feb 18, 2026** | ✅ Complete |
| **13.9.GB-Ext** | **agg_columns + WLS Fix + fit_intercept + Lean Output + Parallel Parity** | **Feb 23 – Mar 13, 2026** | ✅ Complete |
| **13.13.GB** | **Expression-Based Linear Columns** | **Mar 23, 2026** | ✅ Complete |
| **13.14.GB** | **Dedicated Sliding Window Aggregation** | **Mar 24, 2026** | ✅ Complete |
| **13.15.GB** | **N-Sigma Cut in SW Aggregation** | **Mar 26, 2026** | ✅ Complete |
| 13.11.GB | Chi2 Audit + Histogram Accumulation | — | 📋 Approved, deferred |
| 13.12.GB | SW Histogram Accumulation Mode | — | 📋 Proposed, deferred |
| 12.15.GB | V4 Integration | — | 📋 Planned |

**Current Test Count:** 500 passed, 4 failed (pre-existing), 133 features (43 verified)
**Capability Matrix:** Phase 13.15.GB — 0 broken, 1 planned

---

## Critical Incident: Silent Numba Regression (Nov 2025)

### What Happened

On **November 14, 2025** (commit `db0eb019`), the V4 Numba JIT kernel was accidentally removed during metadata refactoring. This caused a **10× performance regression** that went undetected for **6 weeks**.

### Resolution

Phases 12.14.GB through 12.14c.GB address this with shared kernel module, Numba/NumPy parity tests, performance gates, memory stability tests, and benchmark framework.

---

## Critical Incident: COG/agg_columns Deferral (Feb 2026)

### What Happened

The original specification (Oct 2025) included aggregation of groupby variables (center-of-gravity) within sliding windows. This was repeatedly deferred across Phases 13.7–13.10 as "V4/V5 extension" without architect approval. Discovered during real-data TPC calibration testing (Feb 19, 2026).

### Root Cause

Same class as Phase 0.1B: scope reduction without architect sign-off (governance failure mode #8). No single reviewer flagged the gap because synthetic tutorial data masked it.

### Resolution

- `agg_columns` parameter added in Phase 13.9.GB Extension
- MTTU_Reviewer v1.11+ Authority Boundary rules cover this case
- Specification compliance checklist added to phase boundary reviews

---

## March 2026 Phases

### Phase 13.10.GB: Non-Linear Sliding Window Fit (Feb 16-17, 2026)
**Commit:** `e1650530` | **Tags:** `PHASE_13_10_GB_BEGIN`, `PHASE_13_10_GB_END`
**Review:** 4/5 approved (Round 2)

Separate `make_nonlinear_sliding_window_fit()` entry point for non-linear models. Named model registry with 6 built-in models (gaussian_plus_line, polynomial2, etc.). Evaluator extended with Option A (interpolate params → evaluate) and Option B (evaluate at corners → interpolate values).

**Deliverables:**
- `groupby_regression_models.py` — 322 lines, model registry + p0 estimators
- `groupby_regression_nonlinear.py` — 652 lines, non-linear SW entry point
- Evaluator: `evaluate_model`, `evaluate_function`, `evaluate_params`
- 39 new tests (7 invariance, 4 integration)
- `scripts/phase_tag.sh` — automated phase tagging

**Key decision:** Separate entry point (Approach B), not extending `make_sliding_window_fit` — non-linear has different parameter space, metadata, and output columns.

**Tests:** 447 passed, 3 failed (pre-existing). 126 features, 43 Verified, 0 Broken.

---

### Phase 13.10.GB-T: K0s THn Tutorial (Feb 18, 2026)
**Review:** 5/5 approved

End-to-end MC truth validation of non-linear SW pipeline. Synthetic K0s invariant mass spectra in 4D phase space (1/pt, tgl, phi, occupancy) with `gaussian_plus_line` fits.

**Issues discovered (tutorial's primary purpose):**
- ISSUE-1: χ²/ndf scales with statistics → needs chi2 audit (Phase 13.11.GB)
- ISSUE-2: SW concatenates rows instead of summing histograms → design limitation (Phase 13.12.GB)
- ISSUE-3: Offset/slope pull σ ≈ 0.66 → parameter correlation, not a bug
- ISSUE-4: SW sigma pull grows with statistics → consequence of ISSUE-2

---

### Phase 13.9.GB Extensions (Feb 23 – Mar 13, 2026)
**Commits:** `ce1361d6` through `9a13130e` | **Review:** 3/3 to 4/4 approved per sub-extension

Five sub-extensions addressing specification gaps and bugs found during real-data testing:

**Extension 1: agg_columns (Feb 23)**
- `agg_columns` parameter for COG/window statistics (mean/std of arbitrary columns)
- All 3 code paths: zerocopy, V3 incremental, V5 numba
- Kernel-weighted aggregation for non-uniform kernels
- 8 new tests. Resolves COG gap from original specification.

**Extension 2: WLS Weight Fix (Mar 11)**
- **P0 bug:** `weights` parameter silently ignored in regression (used only for aggregation)
- Fix: `sqrt(w)` transform in V1 numpy, V2 falls back to V1, V3 weighted XtX/XtY
- R²/RMSE on unweighted residuals (documented convention)
- 7 new tests

**Extension 3: fit_intercept=False Column Fix (Mar 11)**
- **P0 bug:** Output contained intercept columns with 0/NaN when `fit_intercept=False`
- Fix: `_assemble_results` conditional on `fit_intercept`
- 2 new tests

**Extension 4: Lean Output (Mar 12)**
- **Architect decision:** Remove default fit_column stats (mean/std/median/entries/r_squared per target)
- 52 → ~28 columns for 4-target fits
- Opt-in via `agg_columns` to restore. Canonical: `agg_columns = gb_columns + linear_columns`
- 2 new tests, 6 test files updated

**Extension 5: Parallel Parity (Mar 13)**
- Propagate `agg_columns`, `agg_median`, `fit_intercept` to parallel workers
- Option B (minimal parameter propagation, no architecture change)
- RuntimeError if all parallel units fail (was silent empty DataFrame)
- forkserver/spawn context for Numba thread conflicts (256-core fix)
- 4 new tests. 466 passed.

---

### Phase 13.13.GB: Expression-Based Linear Columns (Mar 23, 2026)
**Commits:** `f0cb33f3`, `bee8226b` | **Tags:** `PHASE_13_13_GB_BEGIN`, `PHASE_13_13_GB_END`
**Review:** 4/4 approved

`linear_columns` accepts `(key_name, expression)` tuples alongside strings. Expressions evaluated via `df.eval()` at entry point, kernels unchanged.

**Key design:** Preprocessing layer (`_preprocess_linear_columns`) at API boundary. Kernels receive augmented DataFrame with all columns materialized. Clean naming via explicit key — no `slope_xM**2 * driftM` invalid identifiers.

**Integrated into:** `make_parallel_fit_v4` and `make_sliding_window_fit`. Parallel SW deferred (P2).

**Metadata:** `linear_column_map` + `linear_columns_normalized` stored in output.

**Performance:** `df.eval()` ~13% slower than numpy for 20 expressions — acceptable.

**Tests:** 19 new. 485 passed, 3 failed (pre-existing).

---

### Phase 13.14.GB: Dedicated Sliding Window Aggregation (Mar 24, 2026)
**Commits:** `190dbd63` through `030543737` | **Tags:** `PHASE_13_14_GB_BEGIN`, `PHASE_13_14_GB_END`
**Review:** 4/4 approved (v1.1 corrected proposal)

New `make_sliding_window_aggregate` for pure aggregation (mean/std/count) without regression. Uses per-bin sufficient statistics and scalar window accumulation.

**Motivation:** `make_sliding_window_fit` with `linear_columns=[]` took 440s per sector for 815k bins. Aggregation does not need design matrices or regression.

**Algorithm:**
1. Bin mapping (counting sort) — O(N)
2. Per-bin sufficient stats (np.bincount) — O(N×C)
3. Window accumulation (Numba prange) — O(B×W×C) scalar adds
4. Compute mean/std — O(B×C)

**Performance optimizations (3 rounds):**
- Step 2: `np.bincount` replaces Python loop (2.3s → 0.013s, 177×)
- Step 3: `nb.prange` parallelizes over bins
- `_build_dense_lookup`: vectorized numpy (3.6s → <0.05s)
- **Total: 440s → ~1.5s per time frame (~300× speedup)**

**Key formula correction (all 4 reviewers caught):** Variance uses `kw` not `kw²` for kernel weights. `kw²` is for error propagation (different operation).

**Tests:** 10 new (4 invariance, 1 performance gate). 494 passed.

---

### Phase 13.15.GB: N-Sigma Cut in SW Aggregation (Mar 26, 2026)
**Commit:** `efdfb130` | **Tags:** `PHASE_13_15_GB_BEGIN`, `PHASE_13_15_GB_END`
**Review:** 3/3 approved

Two-pass sigma-clipped aggregation: Pass 1 computes mean/std, sigma cut flags outliers per row, Pass 2 recomputes clean statistics. Reuses existing sufficient stats + Numba kernel.

**Parameter:** `n_sigma_cut` (default `None` — no change). Cost: ~1.7× current.

**Guards:** `safe_std = inf` for std=0 bins (prevents all-outlier marking).

**Design choice:** Cut uses per-bin window mean/std (smoothed by neighbors), not raw per-bin stats. This gives better statistics for sparse bins.

**Tests:** 6 new (5 invariance, 1 smoke). 500 passed, 4 failed (pre-existing).

---

## Key Technical Decisions (New)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Separate non-linear entry point | Different parameter space, metadata, output columns | 13.10.GB |
| Both evaluator options (A+B) | Option A for smooth interpolation, Option B for exact at corners | 13.10.GB |
| `agg_columns` not new parameter — uses existing mechanism | No API proliferation; fit_columns stats opt-in | 13.9.GB-Ext |
| `sqrt(w)` WLS transform (non-mutating) | Keep originals for unweighted diagnostics | 13.9.GB-Ext |
| V2 falls back to V1 for WLS | Same pattern as V5→V3; keep numba kernel simple | 13.9.GB-Ext |
| R²/RMSE on unweighted residuals | Consistent with OLS behavior, backward-compatible | 13.9.GB-Ext |
| Default output: no fit_column stats | Architect direction; 52→28 columns; opt-in via agg_columns | 13.9.GB-Ext |
| Tuple `(key, expr)` for linear_columns | Clean naming; `df.eval()` at entry point, kernels unchanged | 13.13.GB |
| Sufficient statistics for aggregation | O(B×W×C) scalar adds vs O(N×W) per-row; 300× speedup | 13.14.GB |
| Variance formula: `kw` not `kw²` | Weighted variance of data, not error propagation | 13.14.GB |
| Per-column NaN counts | Different NaN patterns across columns; 39 MB acceptable | 13.14.GB |
| Dense lookup array for Numba | O(1) coord→bin; ~16 MB; Numba-friendly (no dict) | 13.14.GB |
| Single-pass sigma clipping | Sufficient for TPC outliers (extreme, sparse); iterative deferred | 13.15.GB |
| Sigma cut uses window mean/std | Better statistics for sparse bins; avoids neighbor contamination | 13.15.GB |

---

## Performance Reference (Updated)

### Sliding Window Aggregation (Phase 13.14.GB, Linux aarch64)

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| 1 sector (815k bins, 2M rows, 4D window) | 440s | ~1.5s | **~300×** |
| 36 sectors sequential | ~4.4 hours | ~54s | **~300×** |
| With n_sigma_cut=3.0 | — | ~2.5s | ~1.7× vs no cut |

### Capability Matrix (Phase 13.15.GB)

| Metric | Value |
|--------|-------|
| Total features | 133 |
| Verified (✅) | 43 (32.3%) |
| Smoke-only (☑️) | 89 (66.9%) |
| Broken (🧨) | 0 (0.0%) |
| Planned (📋) | 1 (0.8%) |
| Total tests passed | 500 |
| Pre-existing failures | 4 |

---

## Planned Phases

| Phase | Content | Status |
|-------|---------|--------|
| 13.11.GB | Chi2 audit + `--exact-model` diagnostic | Approved, deferred |
| 13.12.GB | SW histogram accumulation mode (`fit_mode='histogram'`) | Proposed, deferred |
| 13.13.GB-B | Batched expression evaluation (memory) | Proposed |
| 12.15.GB | V4 shared kernel integration | Planned |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Dec 16, 2025 | Initial version |
| 2.0 | Dec 31, 2025 | Added Phases 12.14.GB, 12.14a.GB, incident analysis |
| 3.0 | Jan 4, 2026 | Added Phases 12.14b.GB, 12.14b.GB-addendum, 12.14c.GB |
| 4.0 | Feb 14, 2026 | Added Phases 13.7.GB–13.8.SW. Updated capability matrix (102 features) |
| **5.0** | **Mar 26, 2026** | **Added Phases 13.10.GB–13.15.GB. COG incident. 133 features, 500 tests. New functions: make_nonlinear_sliding_window_fit, make_sliding_window_aggregate, make_sliding_window_aggregate_parallel. Expression-based linear columns. WLS fix, fit_intercept fix, lean output, sigma cut.** |
