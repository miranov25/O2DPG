# Phase History — GroupBy Regression

**Last Updated:** 2026-04-07
**Current Phase:** 13.16.GB
**Document Version:** 6.0

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
| 13.10.GB | Non-Linear SW Fit + Evaluator Extension | Feb 16-17, 2026 | ✅ Complete |
| 13.10.GB-T | K0s THn Tutorial + MC Truth Validation | Feb 18, 2026 | ✅ Complete |
| 13.9.GB-Ext | agg_columns + WLS Fix + fit_intercept + Lean Output + Parallel Parity | Feb 23 – Mar 13, 2026 | ✅ Complete |
| 13.13.GB | Expression-Based Linear Columns | Mar 23, 2026 | ✅ Complete |
| 13.14.GB | Dedicated Sliding Window Aggregation | Mar 24, 2026 | ✅ Complete |
| 13.15.GB | N-Sigma Cut in SW Aggregation | Mar 26, 2026 | ✅ Complete |
| **13.16.GB** | **Evaluator Lookup + scipy Methods** | **Mar 28-29, 2026** | **✅ Complete** |
| **P0-Fix** | **fit_intercept hardcoded in SW numba (3 locations)** | **Mar 28-29, 2026** | **✅ Fixed** |
| 13.11.GB | Chi2 Audit + Histogram Accumulation | — | 📋 Approved, deferred |
| 13.12.GB | SW Histogram Accumulation Mode | — | 📋 Proposed, deferred |
| — | `boundary='symmetric'` implementation + tests | — | ⚠️ Architect requirement, status disputed (see Incident 4) |
| 12.15.GB | V4 Integration | — | 📋 Planned |

**Current Test Count:** 517 passed, 3 failed (pre-existing), 133 features (43 verified)
**Capability Matrix:** Phase 13.16.GB — 0 broken, 1 planned

---

## Critical Incident 1: Silent Numba Regression (Nov 2025)

### What Happened

On **November 14, 2025** (commit `db0eb019`), the V4 Numba JIT kernel was accidentally removed during metadata refactoring. This caused a **10× performance regression** that went undetected for **6 weeks**.

### Resolution

Phases 12.14.GB through 12.14c.GB address this with shared kernel module, Numba/NumPy parity tests, performance gates, memory stability tests, and benchmark framework.

---

## Critical Incident 2: COG/agg_columns Deferral (Feb 2026)

### What Happened

The original specification (Oct 2025) included aggregation of groupby variables (center-of-gravity) within sliding windows. This was repeatedly deferred across Phases 13.7–13.10 as "V4/V5 extension" without architect approval. Discovered during real-data TPC calibration testing (Feb 19, 2026).

### Root Cause

Same class as Phase 0.1B: scope reduction without architect sign-off (governance failure mode #8). No single reviewer flagged the gap because synthetic tutorial data masked it.

### Resolution

- `agg_columns` parameter added in Phase 13.9.GB Extension
- MTTU_Reviewer v1.11+ Authority Boundary rules cover this case
- Specification compliance checklist added to phase boundary reviews

---

## Critical Incident 3: `fit_intercept` Parameter Not Propagated to SW Numba (Mar 28-29, 2026)

### What Happened

`make_sliding_window_fit` with `fit_intercept=False` silently failed on 100% of bins when using the numba backend. All bins returned `quality_flag='fit_failed'`. No error was raised. The same data worked perfectly with `make_parallel_fit_v4`. **Found by O2DistAI team in production**, not by the test suite.

### Root Cause

Three hardcoded assumptions in `_fit_window_regression_numba`, all ignoring `fit_intercept=False`:

| Location | Bug | Fix |
|----------|-----|-----|
| Line 994 | `fit_intercept=True` hardcoded in kernel call | Pass `fit_intercept` parameter through |
| Line 913 | `n_params = n_pred + 1` hardcoded | `n_params = n_pred + (1 if fit_intercept else 0)` |
| Line 1011 | Result unpacking `out_beta[i, j+1]` assumed intercept at index 0 | `offset = 1 if fit_intercept else 0` |

**Chain of compensating bugs:** With `n_params` wrong, `out_beta` was oversized, so `j+1` indexing didn't crash — it silently read garbage. When `n_params` was fixed, the unpacking crash exposed the third bug. With `fit_intercept=False` and a polynomial basis containing a constant term, the kernel added a second constant column → X'X singular → Cholesky failed → every bin returned `fit_failed`.

### This Is the Third Instance of the Parameter-Not-Propagated Bug Class

| # | Bug | Path | Phase Found |
|---|-----|------|-------------|
| 1 | WLS `weights` silently ignored | V1/V2/V3 regression | 13.9.GB-Ext |
| 2 | `fit_intercept=False` produces intercept columns | `_assemble_results` | 13.9.GB-Ext |
| 3 | `fit_intercept` hardcoded `True` in SW numba (3 locations) | `_fit_window_regression_numba` | This fix |

Pattern: parameters accepted at the API surface but silently ignored in one or more internal code paths. Each path was written independently and not checked against the full parameter contract.

### Why It Survived 4 Review Rounds

All invariance tests used `window=0` (which routes to V5, not V2 numba) or `backend='auto'` (which selected V5). The numba recompute path was never exercised with `fit_intercept=False` and `window > 0`. The tests passed but didn't test the code path where the bug lived.

### Resolution

- 3-part fix across 3 commits: `a9c6d07f`, `0ed8a56c`, `1278986b`
- 10 cross-fitter invariance tests in `test_fit_intercept_all_fitters.py`
- Each test explicitly specifies `backend='numpy'` or `backend='numba'`
- 2 full-chain tests use `window > 0` to exercise the recompute path:
  - `test_sw_window1_numba_matches_manual_windowed_v4`
  - `test_sw_window1_numba_matches_numpy`

### Governance Rule Added (MTTU_Reviewer v1.16, Failure Mode #11)

> Invariance tests must explicitly set path-controlling parameters — no auto/default dispatch. When a parameter is handled by multiple actively used paths, each path needs its own test for that parameter. Reviewers must verify that the test reaches the claimed code path, not just that it passes.

---

## Critical Incident 4: `boundary='symmetric'` — Specified, Approved, Status Disputed (Mar 2026)

### What Happened

The original Phase 13.8.GB proposal specified `boundary='symmetric'` mode. The architect explicitly requested it with reasoning preserved verbatim:

> "Symmetric will truncate... For TPC — drift, radius, and rphi — all should be symmetric. **I do not want to introduce edge bias.**"
>
> — Main Architect (MI)

The proposal definition (architect-approved):

> `boundary='symmetric'`: Truncate window to maximum symmetric extent. If the center can only reach k bins left, also limit to k bins right.

The Phase 13.8.GB commit message (`bd6f042e`, Feb 10, 2026) claims:

> "V3b (boundary + bin weights): boundary='full'|'symmetric'|'periodic', per-dimension"

However, **the architect reports during real-data testing that only `boundary='full'` (asymmetric truncate) actually exists in the code**. There is no `boundary='symmetric'` test in the test suite, no documentation in the Technical Summary, and no usage in production code.

### Status: Disputed — Requires Verification

The discrepancy between commit message claim and architect testimony is unresolved. Possible explanations:
1. Code was implemented but never tested or documented (architect requirement silently dropped from verification)
2. Commit message was aspirational; implementation is incomplete
3. Implementation exists at a different code path the architect did not encounter

**Action required:** Inspect `_get_neighbor_bins` and `_generate_neighbor_offsets` in `groupby_regression_sliding_window.py` to determine actual implementation status. If implemented, add tests and documentation. If not, implement (~20 lines).

### Root Cause

Whether the code exists or not, this is governance failure mode #8 — architect requirement silently dropped from the verification chain. Same pattern as COG (#2 above). Even if the code was written, it was never tested and never documented, making it indistinguishable from missing.

### Why This Matters

This is the **fourth instance** of architect requirements being silently dropped between specification and verified delivery:

| # | Requirement | Failure Mode |
|---|-------------|--------------|
| 1 | COG/agg_columns | Deferred 4 phases without architect approval |
| 2 | WLS weights actually used in regression | Parameter accepted, silently ignored |
| 3 | `fit_intercept=False` works in SW numba | Parameter accepted, silently ignored |
| 4 | `boundary='symmetric'` for TPC | Specified, approved, never tested or documented |

### Resolution

Pending — see "Planned Phases" below.

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

### Phase 13.16.GB: Evaluator Lookup + scipy Interpolation Methods (Mar 28-29, 2026)
**Commit:** `0ed8a56c` (combined with P0 fix) | **Review:** Approved

Extended `GroupByRegressionEvaluator.evaluate()` with new interpolation methods for performance and mixed-grid support.

**New methods:**

| Method | Implementation | Speed | Use case |
|--------|---------------|-------|----------|
| `'multilinear'` | Python corner-weighted (legacy) | Baseline | Default; supports `use_errors=True` |
| `'linear'` | scipy `map_coordinates` order=1 | ~3× faster | Recommended default |
| `'cubic'` | scipy `map_coordinates` order=3 | ~2× faster | C1-continuous output |
| `'nearest'` | searchsorted + snap | Fast | Discrete lookup |
| **`'lookup'`** | **Direct integer array indexing** | **>7× faster** | **Integer grids only** |
| **`dict`** | **Per-dimension method dispatch** | **Variable** | **Mixed integer/continuous grids** |

**Key design — `method='lookup'`:** Skips `_find_cell` (and `searchsorted`) entirely. Positions are used as raw grid indices via numpy fancy indexing (`grid[sector_array, padrow_array]`). Requires integer-like positions on a 0-based contiguous grid. Bounds handling via existing `bounds` parameter (clamp/nan). `ValueError` on non-integer positions.

**Key design — per-dimension method dict:**

```python
ev.evaluate(
    positions={'sector': sectors, 'padrow': padrows, 'zBin': z_bins},
    method={'sector': 'lookup', 'padrow': 'lookup', 'zBin': 'linear'},
)
```

Each dimension uses its specified method independently. Missing keys default to `'linear'`. Lookup dimensions use direct indexing; interpolation dimensions use `map_coordinates`. The mixed case builds a full D-dimensional coordinate array and passes it to `map_coordinates` — integer dimensions act as exact grid positions.

**Motivation:** Detector calibration maps (TPC as example) have mixed grid structure — sector (0–35) and padrow (0–152) are integer categorical, drift/z are continuous. `searchsorted` on integer indices is pure overhead. For 82M track positions × 3D integer grid: ~60s with `nearest` → ~8s with `lookup` (estimated >7×).

**Tests:** 6 new in `test_evaluator_lookup.py`
- `test_lookup_equals_nearest_for_integers` (invariance gate, `assert_array_equal`)
- `test_lookup_out_of_bounds_nan`, `test_lookup_out_of_bounds_clamp`
- `test_lookup_non_integer_raises`
- `test_per_dimension_method_dict` (mixed lookup + linear)
- `test_lookup_performance` (directional, >1.5× faster)

**Tests total:** 517 passed, 3 failed (pre-existing), 0 regressions.

---

### P0 Fix: `fit_intercept` in SW Numba Path (Mar 28-29, 2026)
**Commits:** `a9c6d07f`, `0ed8a56c`, `1278986b`
**Review:** Approved after 2 rounds (first round: tests detected unfixed bug)

See "Critical Incident 3" above for the full root cause analysis.

**Deliverables:**
- 3-line fix in `_fit_window_regression_numba` (parameter, n_params, unpacking offset)
- `test_fit_intercept_all_fitters.py` — 10 cross-fitter invariance tests
- Each test explicitly specifies `backend=` to prevent auto-select masking
- 2 full-chain tests with `window > 0` exercising the complete recompute path
- V2 excluded (legacy API does not support `fit_intercept`)

---

## Key Technical Decisions

### From Phases 13.10.GB – 13.15.GB (preserved from v5.0)

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

### From Phase 13.16.GB and P0-Fix (new in v6.0)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| `'lookup'` method skips `_find_cell` entirely | O(1) array indexing for integer grids; fastest possible path | 13.16.GB |
| Per-dimension method dict (not single mode) | Detector grids have mixed integer/continuous dimensions; single string can't express this | 13.16.GB |
| scipy `map_coordinates` for linear/cubic | C-implemented, ~3× faster than Python multilinear | 13.16.GB |
| Mixed dict builds full coordinate array for `map_coordinates` | `map_coordinates` handles integer coords as exact grid positions; no special case needed | 13.16.GB |
| Cross-fitter invariance tests must specify `backend=` explicitly | `backend='auto'` masked the fit_intercept bug for 4 review rounds | P0-Fix |
| SW invariance tests must use `window > 0` | `window=0` routes to V5, bypassing V2 numba recompute path | P0-Fix |
| V2 excluded from fit_intercept tests | Legacy API does not support `fit_intercept` parameter | P0-Fix |

---

## Performance Reference

### Sliding Window Aggregation (Phase 13.14.GB, Linux aarch64)

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| 1 sector (815k bins, 2M rows, 4D window) | 440s | ~1.5s | **~300×** |
| 36 sectors sequential | ~4.4 hours | ~54s | **~300×** |
| With `n_sigma_cut=3.0` | — | ~2.5s | ~1.7× vs no cut |

### Evaluator Lookup (Phase 13.16.GB)

| Scenario | nearest (searchsorted) | lookup (direct index) | Speedup |
|----------|----------------------|----------------------|---------|
| 100k rows, D=3, integer grid | ~0.07s | ~0.01s | >5× |
| 10M rows, D=3, integer grid | ~7s (est.) | <1s (est.) | >7× |
| 82M rows, D=3, integer grid | ~60s (est.) | ~8s (est.) | >7× |

### Full makeIterationFit0 Pipeline (Mar 31, 2026)

| Configuration | Time | Notes |
|---------------|------|-------|
| `sec` (0–35) included in `gb_columns` (dense 69M-cell grid) | 354s | `np.full` and `_build_dense_lookup` overhead dominated |
| Per-sector loop (1.9M cells × 36 sectors) | 18s | **~20× speedup** |

**Root cause:** Including high-cardinality grouping variables in `gb_columns` creates overly large dense grids. Looping per sector reduces grid size by ~36×. The fix is in user code, not the library, but documents a usage pattern other teams should follow. The `np.full` and `_build_dense_lookup` overhead dropped from 64s to ~1s after the fix.

### Capability Matrix (Phase 13.16.GB)

| Metric | Value |
|--------|-------|
| Total features | 133 |
| Verified (✅) | 43 (32.3%) |
| Smoke-only (☑️) | 89 (66.9%) |
| Broken (🧨) | 0 (0.0%) |
| Planned (📋) | 1 (0.8%) |
| Total tests passed | **517** |
| Pre-existing failures | **3** |

---

## Failure Modes Catalog (Cumulative)

| # | Mode | Origin Incident | Prevention Mechanism |
|---|------|-----------------|----------------------|
| 1 | Silent regression (no test) | Numba kernel removed Nov 2025 | Performance gates, parity tests, benchmark framework |
| 2 | Architect requirement deferred without sign-off | COG/agg_columns | Authority Boundary rules (MTTU v1.11+, failure mode #8) |
| 3 | Parameter accepted at API but ignored in path | WLS weights silently ignored | Cross-path invariance tests |
| 4 | Output columns don't match parameter | `fit_intercept` columns with 0/NaN | Output assertion tests |
| 5 | Chain of compensating bugs | `fit_intercept` n_params + unpacking | Full-chain tests, fix one assumption at a time |
| 6 | Tests bypass buggy code path via auto-dispatch | `backend='auto'`, `window=0` masked numba bug | Explicit `backend=` and `window>0` in invariance tests (failure mode #11, MTTU v1.16) |
| 7 | Architect requirement implemented but never tested or documented | `boundary='symmetric'` (status disputed) | Specification traceability — track every architect requirement to test + doc |

---

## Planned Phases

| Phase | Content | Status |
|-------|---------|--------|
| — | **Verify and complete `boundary='symmetric'`** | **⚠️ Architect requirement, status disputed (see Incident 4) — P0** |
| 13.11.GB | Chi2 audit + `--exact-model` diagnostic | Approved, deferred |
| 13.12.GB | SW histogram accumulation mode (`fit_mode='histogram'`) | Proposed, deferred |
| 13.13.GB-B | Batched expression evaluation (memory) | Proposed |
| — | Evaluator Numba bulk evaluate (only if 82M-row profiling demands it) | Conditional |
| — | `register_evaluator` on AliasDataFrame | ADF team scope |
| — | Summary Coverage Map for TECHNICAL_SUMMARY (per Org-structure v1.24) | Pending |
| 12.15.GB | V4 shared kernel integration | Planned |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Dec 16, 2025 | Initial version |
| 2.0 | Dec 31, 2025 | Added Phases 12.14.GB, 12.14a.GB, incident analysis |
| 3.0 | Jan 4, 2026 | Added Phases 12.14b.GB, 12.14b.GB-addendum, 12.14c.GB |
| 4.0 | Feb 14, 2026 | Added Phases 13.7.GB–13.8.SW. Updated capability matrix (102 features) |
| 5.0 | Mar 26, 2026 | Added Phases 13.10.GB–13.15.GB. COG incident. 133 features, 500 tests. New functions: make_nonlinear_sliding_window_fit, make_sliding_window_aggregate, make_sliding_window_aggregate_parallel. Expression-based linear columns. WLS fix, fit_intercept fix, lean output, sigma cut. |
| **6.0** | **Apr 7, 2026** | **Phase 13.16.GB (evaluator lookup + scipy methods + per-dimension dict). P0 fit_intercept incident in SW numba (3 locations, 10 cross-fitter tests, governance failure mode #11 added). boundary='symmetric' incident documented (status disputed — architect requirement, commit message claims implementation, architect testimony says only 'full' works). Failure Modes Catalog added (7 entries). makeIterationFit0 performance reference (354s → 18s via per-sector loop). 517 tests, 3 pre-existing failures.** |
