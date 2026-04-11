# Technical Summary: GroupBy Regression

**Version:** 3.4
**Phase:** 13.17.GB
**Last Updated:** 2026-04-11
**Coder:** Claude22 (GBAI team)
**Suggested archive filename:** `GroupByRegression_Technical_Summary_PHASE_13_17_GB_v3_4.md`

> **Changes from v3.3:**
> - **F1 fixed:** `make_sliding_window_aggregate` and its parallel sibling now honour `boundary='symmetric'` and `boundary='periodic'` for the mean/std/count output columns. Both the primary kernel call and the sigma-cut recompute kernel call thread a Path 2 structure `(valid_offset_mask, wrap_flag, wrap_idx, wrapped_coords)` computed once per call by the new helper `_precompute_aggregate_boundary_mask`. Zero behavioural change on the default `boundary='full'` path (verified strictly by T9 against a captured literal baseline).
> - **D1 (median path boundary honouring) DEFERRED** to Phase 13.17.GB-MedianFix per architect direction on 2026-04-09. The median slow path continues to use `'full'` behaviour regardless of `boundary` for the median column only. Mean/std/count columns are fully fixed. Architect quotes verbatim (typos preserved): *"D1. We can psopone for later Phase"* and *"D1. I decidee only later on . I did not realize it it too complicated. Can be postponed...."*
> - **Tests:** 28 new pytest-level test runs in `tests/test_aggregate_boundary.py` from 16 test functions (T14 × 6 parametrize, T16 × 3, T17 × 6, plus 13 singles). Unified Claude22 + Claude23 joint plan from v1.3 §6.2. 8 of 16 are invariance-style (50% invariance ratio, up from v1.2's 25%) — including T13 oracle via `_get_neighbor_bins_v2`, T14 numba≡numpy cross-backend via `monkeypatch`, T15 periodic shift + topology on 2-D grid, T16 window=0 ≡ pandas groupby across all modes, T17 constant-field canary. T12 (median) removed per D1 deferral.
> - **Test count — CORRECTION of inherited v3.3 arithmetic error:** v3.3 records 528 passed canonical but the correct number is 529. Calculation: `517 + 11 (FIX2 evaluator tests) + 1 (F5 flip) = 529`. v3.3 omitted the F5 flip. v3.4 corrects this throughout.
> - **Canonical test count v3.4:** **556 passed / 3 failed / 19 skipped** on `alma2` commit `85713774`, branch `feature/groupby-optimization`, 2026-04-11 09:03 CEST. Source: `test_logs/SUMMARY_20260411_090322.txt`. Calculation: `528 baseline (corrected) + 28 new = 556`. The previously cited "529 baseline" was off by one because the timing test `test_v3_numpy_faster_than_v1_numpy` was already failing pre-Phase-13.17.GB but had not been recorded in any deferred-failure list.
> - **Pre-existing failures: 3 (was 2 in v3.3 corrected baseline).** The third is `test_invariance_sliding_window.py::TestSWV3bTiming::test_v3_numpy_faster_than_v1_numpy` — a wall-clock timing comparison, environment-sensitive, unrelated to F1 boundary handling. Not introduced by Phase 13.17.GB; surfaced by the canonical run because the deferred-failure list was incomplete in v3.3.
> - **Capability matrix: 0 broken / 0 partial / 43 verified / 133 total.** F1 closes the only Broken row from v3.3.
> - **Incident 7 added** to PHASE_HISTORY v6.1 — see that document for the F1 description and the Resolution block.
> - **Parameter-not-propagated bug class catalog:** now at 7 instances. F2 (Incident 6, FIX2) and F1 (Incident 7, this phase) are consecutive entries.
> - **All v3.0 / v3.2 / v3.3 sections marked `[UNCHANGED]` preserved verbatim.** Sections marked `[UPDATED]` extend rather than replace.

> **Changes from v3.2:**
> - **Fix F2 committed:** `method=dict` with mixed interpolation orders now raises `ValueError` at dispatch time instead of silently picking the first dim's order. Validation in `_eval_per_dimension` treats `'nearest'`/`'nearest_fast'` as equivalent (both order 0) per C3. Full `method_dict` included in error message per C5.
> - **Fix F3 committed:** `evaluate()` and `get_coefficients()` docstrings now document all six method values (including `'lookup'`, `'nearest_fast'`, and the per-dimension `dict` shape). Type hints corrected from `method: str` to `method: Union[str, Dict[str, str]]`. Two runnable examples added.
> - **Fix F4 committed:** `evaluate(method='lookup', bounds='extrapolate')` now raises `ValueError` with a clear message instead of falling through to an opaque `IndexError` from numpy fancy indexing. Unknown `bounds` values also rejected at dispatch time (sibling improvement).
> - **Fix F5 committed:** Stale `test_select_backend_auto_sequential` updated to assert `'numba'` per the Phase 12.11 auto-dispatch behavior. This test had been silently failing from Phase 12.11 (Dec 26, 2025) through Phase 13.16.GB.
> - **Bonus C10 committed:** `method=dict` with unknown method strings (e.g. typo `'linaer'`) now raises `ValueError` instead of silently defaulting to `'linear'`. One-line addition alongside F2.
> - **Tests:** 11 new tests in `test_evaluator_lookup.py` (5 F2 + 2 F3 + 1 F4 + 1 C3 + 1 C8 + 1 C10), all with explicit path-controlling parameters per failure mode #11 and `pytest.raises(..., match=...)` per C7.
> - **Test count:** 517 → **528 passed** (+11 new), 3 → **2 failed** (−1: F5 flipped), 19 skipped unchanged. Canonical machine `alma2`.
> - **F2 (method=dict) removed from § Known Limitations** (now fixed).
> - **F4 (lookup+extrapolate) removed from § Known Limitations** (now fixed).
> - **F5 removed from pre-existing failures list** (now fixed).
> - F1 (`boundary='symmetric'` in aggregate) **remains** as the sole 🧨 entry — scheduled for Phase 13.17.GB.
> - Public Interface Catalog: `method` parameter now formally documented as `Union[str, Dict[str, str]]` at the function-level (previously footnoted).
> - **All v3.0 sections preserved verbatim.** Sections marked `[UPDATED]` extend the v3.0 content without replacing it. Sections marked `[UNCHANGED]` are bit-identical to v3.0.
>
> Sections marked **[NEW]** or **[UPDATED]** indicate changes from v3.0. Sections marked **[UNCHANGED]** are preserved verbatim from v3.0.

---

## [UPDATED] Subproject Scope and Target

| # | Capability | Status |
|---|-----------|--------|
| 1 | Linear per-bin regression (OLS/WLS/GLM/RLM/Huber) | ✅ Complete |
| 2 | Sliding window regression (N-D neighbor smoothing) | ✅ Complete |
| 3 | Parallel execution (multi-sector, multi-worker) | ✅ Complete |
| 4 | Function evaluator and interpolator (linear models) | ✅ Complete (Phase 13.9.GB) |
| 5 | Non-linear GroupBy regression (binned spectra) | ✅ Complete (Phase 13.10.GB) |
| 6 | Expression-based linear columns | ✅ Complete (Phase 13.13.GB) |
| 7 | Dedicated sliding window aggregation | ✅ Complete (mean/std/count, F1 fixed in 13.17.GB); ⚠️ partial for median (boundary deferred to 13.17.GB-MedianFix) |
| 8 | Sigma-clipped robust aggregation | ✅ Complete (Phase 13.15.GB) |
| 9 | **Evaluator: scipy interpolation + lookup + per-dimension methods** | **✅ Complete (Phase 13.16.GB)** |
| 10 | THn interface (histogram fitting) | 📋 Planned (Phase 13.12.GB) |
| 11 | Integration interfaces (ADF, RDataFrameDSL, RootInteractive) | 📋 Planned |

**Next scheduled phases** (in order):
1. ~~`13.16.GB-FIX2`~~ — ✅ Completed 2026-04-09. Evaluator bug fixes F2/F3/F4/F5 + C3/C8/C10. 11 new tests. Coder: Claude21.
2. ~~`13.17.GB`~~ — ✅ **Completed 2026-04-11** (this document is the v3.4 summary update). F1 fixed for mean/std/count in both kernel call sites via Path 2 (mask + wrap_flag). D1 (median subpath) deferred. 28 new pytest-level test runs. Coder: Claude22.
3. `13.17.GB-MedianFix` — **NEW micro-phase, scheduled immediately.** Fixes the `agg_median=True + boundary` interaction deferred from 13.17.GB. Estimated ~6-10h. Same fix strategy (reuse the Path 2 mask infrastructure from 13.17.GB), dedicated T12 invariance test.
4. `13.11.GB` / `13.12.GB` / `12.15.GB` — planned, unordered.

---

## [UPDATED] Quick Start

> **All three core entry points are keyword-only** (`def f(*, df, ...)`). Pass `df=` explicitly. Positional calls raise `TypeError: takes 0 positional arguments`.

```python
# Per-bin regression (recommended)
from dfextensions.groupby_regression import make_parallel_fit_v4
df_out, dfGB = make_parallel_fit_v4(
    df=df,
    gb_columns=['sector', 'padRow'],
    fit_columns=['driftV'],
    linear_columns=['spaceCharge'],
    suffix='_fit',
)

# Sliding window regression
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_fit, make_sliding_window_fit_parallel,
)
result = make_sliding_window_fit(
    df=dfGB,
    gb_columns=['padRow', 'zBin'],
    fit_columns=['driftV_intercept_fit'],
    linear_columns=[],
    window_spec={'padRow': 1, 'zBin': 2},
    agg_columns=['padRow', 'zBin'],  # COG of groupby coordinates
)

# Pure aggregation (no regression — fastest for large grids)
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_aggregate, make_sliding_window_aggregate_parallel,
)
result = make_sliding_window_aggregate(
    df=df,
    gb_columns=['row_bin', 'driftM_bin', 'dsecM_bin', 'mP4_bin'],
    agg_columns=['xM', 'driftM', 'tgSlp', 'dy', 'dz'],
    window_spec={'row_bin': 1, 'driftM_bin': 1, 'dsecM_bin': 1, 'mP4_bin': 1},
    n_sigma_cut=3.0,  # robust 2-pass statistics
)
# ✅ Phase 13.17.GB: boundary='symmetric' and 'periodic' now honoured for the
# mean/std/count output columns. The median subpath (agg_median=True) still
# uses 'full' behaviour regardless — that fix is scheduled for Phase
# 13.17.GB-MedianFix. See § Known Limitations.

# Expression-based polynomial fitting
df_out, dfGB = make_parallel_fit_v4(
    df=df,
    gb_columns=['sector'],
    fit_columns=['dy'],
    linear_columns=[
        'xM',
        ('xM2', 'xM**2'),
        ('xM_driftM', 'xM * driftM'),
    ],
)

# [NEW] Evaluator with fast interpolation methods
from dfextensions.groupby_regression.groupby_regression_evaluator import (
    GroupByRegressionEvaluator,
)
# dfGB here is the return value from make_parallel_fit_v4 above
# (schema: gb_columns=['sector', 'padRow'], linear_columns=['spaceCharge'],
#          fit_columns=['driftV'], suffix='_fit')
ev = GroupByRegressionEvaluator.from_dfGB(
    dfGB,
    group_columns=['sector', 'padRow'],
    predictor_columns=['spaceCharge'],
    targets='driftV',
    suffix='_fit',
)

# scipy linear interpolation (~3x faster than legacy multilinear)
result = ev.evaluate(
    positions={'sector': track_sectors, 'padRow': track_padRows},
    predictors={'spaceCharge': track_spaceCharge},
    method='linear',
)

# All-integer grid: direct array indexing (skips searchsorted)
result = ev.evaluate(
    positions={'sector': track_sectors, 'padRow': track_padRows},
    predictors={'spaceCharge': track_spaceCharge},
    method='lookup',
)

# Mixed grid: per-dimension method selection
result = ev.evaluate(
    positions={'sector': track_sectors, 'padRow': track_padRows},
    predictors={'spaceCharge': track_spaceCharge},
    method={'sector': 'lookup', 'padRow': 'linear'},
)
```

**Which function should I use?**

```
Need robust/Huber fitter?       → GroupByRegressor.make_parallel_fit()
Need OLS/WLS, single target?    → make_parallel_fit_v4()           ← recommended
Need OLS, multiple targets?     → make_parallel_fit_v5()
No Numba available?             → make_parallel_fit_v2() or v3()
Need sliding window regression? → make_sliding_window_fit()
Need parallel SW regression?    → make_sliding_window_fit_parallel()
Need pure aggregation (fast)?   → make_sliding_window_aggregate()
Need parallel aggregation?      → make_sliding_window_aggregate_parallel()
Need non-linear spectrum fit?   → make_nonlinear_sliding_window_fit()
Need polynomial expressions?    → linear_columns=[('key','expr')]
Need fast eval (any grid)?      → evaluate(method='linear')        [NEW]
Need fastest eval (integer)?    → evaluate(method='lookup')        [NEW]
Need mixed integer/continuous?  → evaluate(method=dict)            [NEW]
Need symmetric boundary in
  SW aggregation (mean/std)?    → make_sliding_window_aggregate(boundary='symmetric')
Need symmetric boundary in
  SW aggregation median?        → wait for Phase 13.17.GB-MedianFix [KNOWN LIMITATION]
```

---

## [UPDATED] Public Interface Catalog

| Function | Module | Purpose | Stability | Phase |
|----------|--------|---------|-----------|-------|
| `make_parallel_fit_v2()` | `groupby_regression_optimized` | Legacy OLS per-group fit | Legacy — no `fit_intercept` | pre-12 |
| `make_parallel_fit_v3()` | `groupby_regression_optimized` | Incremental OLS per-group fit | Stable | 12.x |
| `make_parallel_fit_v4()` | `groupby_regression_optimized` | Numba JIT weighted OLS; recommended entry point | Stable | 12.x |
| `make_parallel_fit_v5()` | `groupby_regression_optimized` | Batched multi-target OLS | Stable | 12.8.GB |
| `make_sliding_window_fit()` | `groupby_regression_sliding_window` | N-D sliding window regression (per-bin or recompute) | Stable | 13.8.GB |
| `make_sliding_window_fit_parallel()` | `groupby_regression_sliding_window` | Parallel SW regression (split-column) | Stable | 13.8.SW |
| `make_sliding_window_aggregate()` | `groupby_regression_sliding_window` | Pure SW aggregation (sufficient stats) | Stable | 13.14.GB |
| `make_sliding_window_aggregate_parallel()` | `groupby_regression_sliding_window` | Parallel pure aggregation | Stable | 13.14.GB |
| `make_nonlinear_sliding_window_fit()` | `groupby_regression_nonlinear` | Non-linear SW fit (named models, custom callables) | Stable | 13.10.GB |
| `GroupByRegressionEvaluator` | `groupby_regression_evaluator` | Evaluate fitted models at arbitrary coordinates | Stable | 13.9.GB |
| `GroupByRegressionEvaluator.from_dfGB()` | `groupby_regression_evaluator` | Convenience constructor from a dfGB output | Stable | 13.9.GB |
| `register_fit_model()` | `groupby_regression_models` | Register custom fit models | Stable | 13.10.GB |
| `get_model()` | `groupby_regression_models` | Named model registry lookup | Stable | 13.10.GB |
| `list_models()` | `groupby_regression_models` | List registered model names | Stable | 13.10.GB |

**No new public functions in v3.3.** Phase 13.16.GB extended `GroupByRegressionEvaluator.evaluate()` with new values for the existing `method` parameter — no API change at the function level. Phase 13.16.GB-FIX2 updated the type hint to `method: Union[str, Dict[str, str]] = 'multilinear'` (previously documented only in the v3.2 footnote) and added detect-and-reject validation for unsupported dict shapes.

---

## [NEW] Evaluator Method Reference (Phase 13.16.GB)

`GroupByRegressionEvaluator.evaluate()` accepts the following values for `method=`:

| Method | Implementation | Speed | Use case |
|--------|---------------|-------|----------|
| `'multilinear'` | Python corner-weighted (legacy) | Baseline (~1 M/s) | Default; supports `use_errors=True` (WLS interpolation) |
| `'linear'` | scipy `map_coordinates` order=1 | ~3 M/s | **Recommended default** — same results as `multilinear`, C-implemented |
| `'cubic'` | scipy `map_coordinates` order=3 | ~0.7 M/s | Smooth C1-continuous output (visualization, derivatives) |
| `'nearest'` | searchsorted + snap (Python) | Fast | Discrete lookup with nearest-bin semantics |
| `'nearest_fast'` | scipy `map_coordinates` order=0 | Fast | C-implemented nearest; equivalent to `'nearest'` but faster for large batches |
| **`'lookup'`** | **Direct integer array indexing** | **Fastest¹** | **Integer grids only — skips searchsorted entirely** |
| **`dict`** | **Per-dimension method dispatch** (validated 13.16.GB-FIX2) | Variable | Mixed integer/continuous grids; see § Per-dimension method dict for supported shapes |

¹ Speed "Fastest" for `'lookup'` is an estimate based on algorithmic complexity (`O(1)` indexing vs `O(log N)` searchsorted). **Not yet measured at production scale** — see § Summary Coverage Map › Unverified Claims. A profile gate is planned after the first production run.

Speed columns for `'linear'`, `'cubic'`, `'multilinear'` are directional estimates on small synthetic workloads. They are not benchmark gates. A future update will add profile artifacts if a Coder's Review Packet includes them.

### `method='lookup'`

Direct array indexing via `grid[idx_array]`. Skips `_find_cell` and `searchsorted` entirely. Pure numpy fancy indexing.

**Requirements:**
- Positions must be integer-like (int32, int64, or float with integer values)
- Grid must be 0-based contiguous integers
- Non-integer float positions raise `ValueError`
- `bounds='extrapolate'` is rejected with `ValueError` when `method='lookup'` (13.16.GB-FIX2) — direct integer indexing has no interpolation to extrapolate from. Use `method='linear'` or `'cubic'` with `bounds='extrapolate'`, or use `bounds='clamp'`/`'nan'` with `method='lookup'`.

**Bounds handling** (via existing `bounds` parameter):
- `bounds='clamp'`: clips out-of-range to `[0, shape[d]-1]`
- `bounds='nan'`: marks out-of-range as NaN

**Performance target:** For 82M rows × 3D integer grid: ~8s estimated (vs ~60s with `'nearest'`). Not yet validated at scale.

### Per-dimension method dict

```python
ev.evaluate(
    positions={'sector': sectors, 'padRow': padRows, 'zBin': z_bins},
    predictors={},
    method={'sector': 'lookup', 'padRow': 'lookup', 'zBin': 'linear'},
)
```

- Keys must match group column names
- Missing keys default to `'linear'`
- Lookup dimensions: direct integer indexing
- Linear/cubic dimensions: `map_coordinates` interpolation
- All-lookup dict delegates to `_eval_lookup` (fastest path)
- All-interpolation dict delegates to `map_coordinates` directly
- Mixed dict: builds full D-dimensional coordinate array; integer dimensions act as exact grid positions for `map_coordinates`

**Validation (13.16.GB-FIX2):** `scipy.ndimage.map_coordinates` accepts only a scalar `order` argument, so a single `evaluate()` call cannot honor mixed interpolation orders (e.g. `{'a': 'linear', 'b': 'cubic'}`) across different dimensions. Before Phase 13.16.GB-FIX2 the code silently picked the first interpolation dimension's order. As of 13.16.GB-FIX2 the dispatcher detects this and raises `ValueError` with the full `method_dict` in the message. `'nearest'` and `'nearest_fast'` (both order 0) are treated as equivalent for this check. **Supported patterns:**
- Zero or more `'lookup'` dimensions, plus **zero or one** non-lookup interpolation method (any number of copies of the same order).
- `{'a': 'lookup', 'b': 'linear', 'c': 'linear'}` ✅
- `{'a': 'lookup', 'b': 'nearest', 'c': 'nearest_fast'}` ✅ (both order 0)
- `{'a': 'linear', 'b': 'cubic'}` ❌ raises `ValueError` — call `evaluate()` twice with single methods instead
- `{'a': 'lookup', 'b': 'linaer'}` ❌ raises `ValueError` — unknown method (typo detection)

**Use case:** Detector calibration maps with mixed structure — categorical integer dimensions (sector 0–35, padRow 0–152) and continuous dimensions (drift, z) in the same grid.

**Tests:** See `test_evaluator_lookup.py` (17 tests total: 6 original from commit `74b12857`, 11 added in 13.16.GB-FIX2).

---

## [UNCHANGED] Sliding Window Aggregation API

### `make_sliding_window_aggregate()`

Pure aggregation using per-bin sufficient statistics. No regression, no design matrix. **~300× faster** than `make_sliding_window_fit` with `linear_columns=[]` for large grids.

```python
result = make_sliding_window_aggregate(
    df=df,
    gb_columns=['row_bin', 'driftM_bin', 'dsecM_bin', 'mP4_bin', 'sec'],
    agg_columns=['xM', 'driftM', 'dsectorM', 'tgSlp', 'dy', 'dz'],
    window_spec={'row_bin': 1, 'driftM_bin': 1, 'dsecM_bin': 1, 'mP4_bin': 1},
    weights=None,              # optional row weights for weighted mean/std
    selection=mask,            # boolean array (positional), same length as df
    suffix='_sw',
    min_stat=1,
    kernel='uniform',          # or 'gaussian'
    n_sigma_cut=3.0,           # optional 2-pass robust statistics
    agg_median=False,          # True enables slow median path
    verbose=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Input data |
| `gb_columns` | list[str] | required | Columns defining bin grid |
| `agg_columns` | list[str] | required | Columns to aggregate (mean/std/count) |
| `window_spec` | dict[str, int] | required | SW half-width per dimension |
| `weights` | str or None | None | Column name for weighted mean/std |
| `selection` | array or None | None | Boolean mask (positional, same length as df) |
| `suffix` | str | '_sw' | Output column suffix |
| `min_stat` | int | 1 | Minimum entries per window |
| `kernel` | str | 'uniform' | Window kernel ('uniform' or 'gaussian') |
| `kernel_width` | float/dict or None | None | Kernel bandwidth |
| `boundary` | str/dict | 'full' | **Phase 13.17.GB:** boundary handling now honoured for mean/std/count via Path 2 mask + wrap_flag threading through both backends and both kernel call sites (primary + sigma-cut recompute). `'full'` (default) is bit-identical to pre-fix behaviour (T9 regression gate). `'symmetric'` mirrors `_get_neighbor_bins_v2`'s `eff_w(c) = min(w, c-lo, hi-c)` from the SW fit path. `'periodic'` wraps neighbours via `((raw - lo) % n_range) + lo`. **Known limitation:** the median subpath (`agg_median=True`) still ignores `boundary` — see Known Limitations. |
| `n_sigma_cut` | float or None | None | 2-pass sigma clipping threshold |
| `agg_median` | bool | False | Compute median (slow — disables sufficient stats) |
| `verbose` | bool | False | Print timing per step |

**Output columns (guaranteed contract):**

| Column | Description |
|--------|-------------|
| `{col}_mean{suffix}` | (Kernel/row-weighted) mean within window |
| `{col}_std{suffix}` | (Kernel/row-weighted) std within window (Bessel-corrected) |
| `{col}_count{suffix}` | Per-column finite entry count within window |
| `{col}_median{suffix}` | Unweighted median (only if `agg_median=True`) |
| `n_neighbors_used{suffix}` | Number of non-empty neighbor bins |
| `effective_window_fraction{suffix}` | Fraction of expected neighbors found |

**Notes:**
- `_count_sw` is **per-column** — different columns can have different counts if NaN patterns differ
- `_std_sw` is standard deviation within the window, NOT standard error of the mean. Compute SE as `std / sqrt(count)`.
- No caching between repeated calls — each call recomputes from scratch
- Memory: ~O(B × C × 3 × 8) bytes for sufficient stats. 815k bins × 6 cols ≈ 117 MB.

### `make_sliding_window_aggregate_parallel()`

Same as serial, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split_columns` | list[str] | required | Column(s) to partition data |
| `n_workers` | int | 4 | Number of parallel workers |
| `on_error` | str | 'nan' | Error handling: 'nan' (skip) or 'raise' |

**⚠️ Same `boundary` bug as serial — the parallel wrapper delegates to `make_sliding_window_aggregate` and inherits the silent-drop.**

---

## [UPDATED] Sliding Window Fit API Changes

### `agg_columns` parameter [Phase 13.9.GB-Ext]

Computes mean/std of arbitrary columns within each sliding window:

```python
result = make_sliding_window_fit(
    df=dfGB,
    gb_columns=['padRow', 'zBin'],
    fit_columns=['driftV_intercept_fit'],
    linear_columns=[],
    window_spec={'padRow': 1, 'zBin': 2},
    agg_columns=['mpt', 'vertex_z', 'tgl', 'phi'],  # COG columns
    agg_median=False,
)
# Output: mpt_mean_sw, mpt_std_sw, vertex_z_mean_sw, ...
```

**Canonical usage:** `agg_columns = gb_columns + linear_columns` (architect requirement from COG/agg_columns incident, PHASE_HISTORY § Incident 2).

### WLS weights [FIXED in Phase 13.9.GB-Ext]

The `weights` parameter now correctly affects regression (not just aggregation):
- V1 numpy: `sqrt(w)` transform on X, y
- V2 numba: falls back to V1 when weights present
- V3 incremental: weighted sufficient stats (XtX = Xw'Xw)
- V5: falls back to V3 when weights present

R²/RMSE computed on **unweighted** residuals (documented convention).

### `fit_intercept=False` — Fully Fixed

| Path | Status | Test |
|------|--------|------|
| `make_parallel_fit_v4` | ✅ Working | `test_v4_fit_intercept_false_recovers_coefficients` |
| `make_parallel_fit_v3` | ✅ Working | `test_v3_fit_intercept_false_recovers_coefficients` |
| `make_sliding_window_fit` (numpy, window=0) | ✅ Working | `test_sw_numpy_fit_intercept_false_recovers_coefficients` |
| `make_sliding_window_fit` (numba, window=0) | ✅ Fixed (P0, Mar 29 2026) | `test_sw_numba_fit_intercept_false_recovers_coefficients` |
| `make_sliding_window_fit` (numba, **window>0** — recompute path) | ✅ Fixed (P0) | `test_sw_window1_numba_matches_manual_windowed_v4` |
| `make_sliding_window_fit` numba ≡ numpy at window>0 | ✅ Verified | `test_sw_window1_numba_matches_numpy` |
| `make_parallel_fit_v2` | ❌ Not supported | Legacy API, no `fit_intercept` parameter |

**Phase 13.9.GB-Ext fix (output columns):** When `fit_intercept=False`, output no longer contains `{t}_intercept` or `{t}_intercept_err` columns.

**P0 fix (Mar 29 2026):** `_fit_window_regression_numba` had three hardcoded assumptions ignoring `fit_intercept=False` (kernel call, n_params, result unpacking). Found by O2DistAI team in production. Fix in commits `a9c6d07f`, `0ed8a56c`, `1278986b`. 10 cross-fitter invariance tests in `test_fit_intercept_all_fitters.py`. See PHASE_HISTORY § Critical Incident 3.

### Default output columns [Phase 13.9.GB-Ext, Breaking]

Per-target aggregation stats (`_mean`, `_std`, `_median`, `_entries`, `_r_squared`) **no longer exported by default**. To restore, add the target to `agg_columns`.

### Expression-based linear columns [Phase 13.13.GB]

```python
linear_columns=[
    'xM',                           # existing column
    ('xM2', 'xM**2'),               # expression with key name
    ('xM_driftM', 'xM * driftM'),   # product term
]
```

Output coefficients use the key name: `slope_xM2`, `slope_xM_driftM`.

---

## [UPDATED] Known Limitations

| Limitation | Status | Workaround |
|------------|--------|------------|
| ~~SW weighted fits (WLS)~~ | ✅ Fixed (13.9.GB-Ext) | — |
| ~~`fit_intercept=False` in SW numba~~ | ✅ Fixed (P0, Mar 29 2026) | — |
| ~~`method=dict` with mixed interpolation orders~~ | ✅ Fixed (13.16.GB-FIX2) — now raises `ValueError` | Use one interpolation method per call, or call `evaluate()` multiple times |
| ~~`evaluate(method='lookup', bounds='extrapolate')`~~ | ✅ Fixed (13.16.GB-FIX2) — now raises `ValueError` | Use `bounds='clamp'` or `bounds='nan'` with `method='lookup'` |
| `boundary='symmetric'` / `'periodic'` in `make_sliding_window_fit` (V3b/V4) | ✅ Implemented, 6 V3b tests pass (4 boundary + 2 integration) | — |
| `boundary='symmetric'` / `'periodic'` in `make_sliding_window_aggregate` mean/std/count (serial and parallel) | ✅ **Fixed Phase 13.17.GB**, 28 tests pass (unified 16-function plan, 50% invariance ratio) | — |
| **`boundary` + `agg_median=True` interaction in `make_sliding_window_aggregate`** | **⚠️ Median subpath silently uses `'full'` regardless of `boundary`.** Mean/std/count columns honour `boundary` correctly. | **Workaround:** call the function twice — once with the desired boundary and `agg_median=False` for the statistics, once with `boundary='full'` and `agg_median=True` for the (uncorrected) median. **Fix scheduled for Phase 13.17.GB-MedianFix.** Architect-authorised deferral 2026-04-09 (verbatim): *"D1. We can psopone for later Phase"* and *"D1. I decidee only later on . I did not realize it it too complicated. Can be postponed...."* |
| **Pre-existing broken `test_aggregate_numba_matches_numpy` test** | **⚠️ False-positive cross-backend invariance.** Test at `tests/test_sliding_window_aggregate.py:219` calls the same numba backend twice and compares output to itself — auto-dispatch failure mode #11 hiding a real gap that has existed since Phase 13.14.GB. | **Superseded by T14 `test_aggregate_numba_equals_numpy_all_boundaries_all_paths` added in Phase 13.17.GB via `monkeypatch` on `_get_numba_agg_kernel`.** Deletion of the old test is a separate micro-task after 13.17.GB commit (NOT done in-phase per scope rule 1). |
| **`test_v3_numpy_faster_than_v1_numpy` timing test failing on canonical** | **⚠️ Pre-existing wall-clock timing comparison, environment-sensitive.** Failing on canonical alma2 since at least Phase 13.16.GB-FIX2 (predecessor tag), unrelated to F1. | Investigate in a separate cleanup pass; unrelated to boundary handling. Listed in deferred-failures alongside `test_multiple_fits_match_v4_merged` and `test_tpc_distortion_recovery`. |
| V3/V5 incremental: median=NaN | By design | Use V1/V2 (recompute) or `agg_median=True` in aggregate |
| Parallel SW: Windows OS | By design | Linux/macOS only |
| WLS not supported in parallel V5 | By design | Use serial, or split manually |
| Chi2/NDF computation for non-linear SW | Under investigation | Phase 13.11.GB |
| SW histogram accumulation (spectrum fitting) | Proposed | Phase 13.12.GB |
| Expression columns in parallel SW | Deferred | Serial only currently |
| `method='lookup'` requires 0-based integer grid | By design | Subtract minimum before calling |
| Evaluator Numba bulk evaluate | Not implemented | numpy vectorized sufficient for <100M rows; conditional on profiling |
| Python multiprocessing overhead for aggregation | Inherent | Use serial with Numba prange |
| Dense grid with high-cardinality grouping vars | User code issue | Loop per-sector instead of including in `gb_columns` (see PHASE_HISTORY § Performance Reference) |

### Note: `boundary='symmetric'` resolved in Phase 13.17.GB

The v3.3 split between fit-path (`⚠️`) and aggregate-path (`🧨 silently broken`) is now obsolete: as of Phase 13.17.GB the aggregate path honours `boundary` for mean/std/count columns via the Path 2 mask + wrap_flag mechanism (see § Sliding Window Aggregation API for the parameter description). The full historical narrative — architect quotes, root cause, parameter-not-propagated bug class progression — is preserved in PHASE_HISTORY v6.1 Critical Incident 7.

---

## [UPDATED] Current State

| Metric | v2.1 (Feb 2026) | v3.0 (Mar 2026) | v3.2 (Apr 2026) | v3.3 corrected¹ | **v3.4 (Apr 2026)** |
|--------|-----------------|-----------------|-----------------|-----------------|---------------------|
| Test count | 338 passed | 500 passed | 517 passed | 529 passed¹ | **556 passed²** |
| Pre-existing failures | 3 | 4 | 3 | 2 | **3³** |
| Skipped | — | — | 19 | 19 | **19** |
| Features | 102 | 133 | 133 | 133 | **133** |
| Verified (✅) | 25 (24.5%) | 43 (32.3%) | 43 (32.3%) | 43 (32.3%) | **43 (32.3%) + 28 new pytest runs (T1–T17)** |
| Smoke-only (☑️) | — | — | 89 (66.9%) | 89 (66.9%) | **89 (66.9%)** |
| Broken (🧨) | 0 | 0 | 2 (F1, F2) | 1 (F1) | **0** |
| Partial (⚠️) | 0 | 0 | 0 | 0 | **0** (D1 median deferral is recorded as a Known Limitation, not as a Capability Matrix Partial entry, since it does not map to a distinct capability row) |
| Public functions | 14 | 20 | 20 | 20 | **20** |
| Evaluator method values | 2 | 3 | 6 | 6 | **6** |

¹ **v3.3 on disk records 528 passed.** This is an inherited arithmetic error carried over from the FIX2 coder packet: `517 + 11 new = 528` forgot the F5 flip (one failing test moved to passing). Correct arithmetic: `517 + 11 + 1 = 529`. Reference: `SUMMARY_20260409_112129.txt` from the FIX2 reviewer packet. v3.4 corrects this.

² **Canonical v3.4 number from `test_logs/SUMMARY_20260411_090322.txt`** on alma2 commit `85713774`, branch `feature/groupby-optimization`, 2026-04-11 09:03 CEST. Calculation: `528 baseline + 28 new pytest-level runs = 556`. The v3.3 corrected baseline above was 529, but the canonical run revealed the timing test (footnote 3) was already failing pre-phase, so the *true* pre-phase passed-count was 528 not 529. Both numbers are surfaced for transparency: 529 was the expected baseline from arithmetic alone, 528 was the empirically-observed pre-phase passed-count once the third pre-existing failure was identified.

³ **Pre-existing failures jumped from 2 to 3** because the Phase 13.17.GB canonical run surfaced `test_invariance_sliding_window.py::TestSWV3bTiming::test_v3_numpy_faster_than_v1_numpy` as a pre-existing failure not previously listed. This is a wall-clock timing comparison test, environment-sensitive, unrelated to F1 boundary handling. The other 2 failures (`test_multiple_fits_match_v4_merged` V5-vs-V4 rtol, and `test_tpc_distortion_recovery` test bug) were already on the deferred list.

**Coder env note:** Phase 13.17.GB Coder environment was numpy 2.4.3, pandas 3.0.1, numba 0.65.0, pytest 9.0.3 (a different combination from the canonical alma2 environment of pytest 7.2.2 + Python 3.10.19). The 28 new test runs all pass on both environments.

**Capability Matrix snapshot at v3.4:** 0 broken / 0 partial / 43 verified / 89 smoke-only / 1 planned (Phase 13.17.GB-MedianFix) / 133 total. Source: `test_logs/CAPABILITY_MATRIX_20260411_090322.md`.

**Test count sourced from `test_logs/SUMMARY_20260411_090322.txt`** on canonical `alma2` commit `85713774`, branch `feature/groupby-optimization`, Python 3.10.19, pytest-7.2.2-xdist, 12 workers. Reproducible via `bash run_tests.sh | tee run_tests.log`.

**Arithmetic v3.4:** `528 corrected baseline + 28 new pytest-level runs (16 functions in tests/test_aggregate_boundary.py expanded via parametrize: T14 × 6, T16 × 3, T17 × 6, plus 13 singles) = 556 passed`. Pre-existing failures: 3 (was 2 in v3.3 corrected; +1 from the timing test newly listed as pre-existing).

> **Coder env vs canonical env (Phase 13.17.GB):** The Phase 13.17.GB Coder (Claude22) ran in numpy 2.4.3 / pandas 3.0.1 / numba 0.65.0 / pytest 9.0.3 / Python 3.12. Coder-env full-suite numbers: **532 passed / 5 failed / 41 skipped**. The 5 coder-env failures include the 3 canonical pre-existing failures plus 2 environment-only failures (`test_area_conservation_option_b`, `test_parallel_matches_serial`/`test_single_vs_multi_worker` clusters) verified pre-existing on baseline source before any 13.17.GB changes. Canonical numbers (above) come from the alma2 run on 2026-04-11 09:03 CEST.

**The 3 remaining pre-existing failures** are unrelated to Phase 13.17.GB scope and remain deferred:
- `test_tpc_distortion_recovery::test_tpc_distortion_recovery` — test calls `make_sliding_window_fit` with wrong argument convention (test bug, not code bug); separate cleanup.
- `test_phase_12_8_gb::test_multiple_fits_match_v4_merged` — V5 vs V4 parity at `rtol=1e-12`; numerical, ambiguous, separate micro-task.
- **`test_invariance_sliding_window::TestSWV3bTiming::test_v3_numpy_faster_than_v1_numpy`** — wall-clock timing comparison, environment-sensitive, **newly listed in v3.4** but pre-existing on canonical alma2; surfaced by the Phase 13.17.GB canonical run because it had not been in any prior deferred list.

**Zero broken-feature entries in v3.4.** F1 (`boundary` silent-drop in aggregate path) is now ✅ fixed for mean/std/count via Phase 13.17.GB. F2 (`method=dict` mixed interp) was fixed in 13.16.GB-FIX2. The median-subpath D1 deferral is recorded as a Known Limitation but does not constitute a Capability Matrix Broken entry (the SW.aggregate_boundary capability is satisfied for the mean/std/count surface).

---

## [UNCHANGED] O2DistAI Team Q&A

**Q1 (API Contract):** `make_sliding_window_aggregate` is a separate implementation (not a fit with zero linear_columns). Output columns: `{col}_mean_sw`, `{col}_std_sw`, `{col}_count_sw` — this is the guaranteed contract.

**Q2 (Selection):** Yes, `selection` parameter is supported. It is a boolean array (positional, same length as `df`), not index-aligned.

**Q3 (Output Schema):** `_count_sw` is **per-column** (can differ if NaN patterns differ). `_std_sw` is standard deviation, not standard error. Compute SE as `std / sqrt(count)`.

**Q4 (Memory):** With 9 agg_columns and 7.6M rows, sufficient stats memory ≈ `n_bins × 9 × 3 × 8` bytes. All columns materialized simultaneously. No caching between calls.

**Q5 (Weights):** Yes, `weights` parameter is supported for weighted mean/std.

**Q6 (Documentation):** This document covers the API. For evaluator method selection (Phase 13.16.GB), see § Evaluator Method Reference above.

---

## [NEW] Summary Coverage Map

Per Organization-structure v1.25 § Cross-Team Information Flow, behavioral claims reference verification evidence. Per § Critical claims [MUST], claims that other teams depend on reference a test, Capability Matrix entry, or example.

**All test names below have been verified against the committed test files.** 29/29 citations resolve to real tests (18 from v3.2 + 11 new in 13.16.GB-FIX2).

### Verified Claims

| Claim | Test | Entry Point |
|-------|------|-------------|
| `fit_intercept=False` works in V4 | `test_fit_intercept_all_fitters.py::test_v4_fit_intercept_false_recovers_coefficients` (line 105) | `make_parallel_fit_v4` |
| `fit_intercept=False` works in V3 | `test_fit_intercept_all_fitters.py::test_v3_fit_intercept_false_recovers_coefficients` (line 120) | `make_parallel_fit_v3` |
| `fit_intercept=False` works in SW numpy (window=0) | `test_fit_intercept_all_fitters.py::test_sw_numpy_fit_intercept_false_recovers_coefficients` (line 143) | `make_sliding_window_fit(backend='numpy')` |
| `fit_intercept=False` works in SW numba (window=0) | `test_fit_intercept_all_fitters.py::test_sw_numba_fit_intercept_false_recovers_coefficients` (line 161) | `make_sliding_window_fit(backend='numba')` |
| SW numba recompute path works with window>0 (full-chain gate) | `test_fit_intercept_all_fitters.py::test_sw_window1_numba_matches_manual_windowed_v4` (line 347) | `make_sliding_window_fit(backend='numba', window_spec={...: 1})` |
| SW numba ≡ SW numpy with window>0 | `test_fit_intercept_all_fitters.py::test_sw_window1_numba_matches_numpy` (line 425) | Both backends explicit |
| Cross-fitter parity for `fit_intercept=False` | `test_fit_intercept_all_fitters.py::test_cross_fitter_parity_fit_intercept_false` (line 312) | V3, V4, SW-numpy, SW-numba |
| `method='lookup'` ≡ `'nearest'` on integer grid (bit-identical) | `test_evaluator_lookup.py::test_lookup_equals_nearest_for_integers` (line 54) | `evaluate(method='lookup')` |
| `method='lookup'` with `bounds='nan'` | `test_evaluator_lookup.py::test_lookup_out_of_bounds_nan` (line 80) | `evaluate(method='lookup', bounds='nan')` |
| `method='lookup'` with `bounds='clamp'` | `test_evaluator_lookup.py::test_lookup_out_of_bounds_clamp` (line 104) | `evaluate(method='lookup', bounds='clamp')` |
| `method='lookup'` rejects non-integer positions | `test_evaluator_lookup.py::test_lookup_non_integer_raises` (line 134) | `evaluate(method='lookup')` |
| `method=dict` single-interp case (lookup + one non-lookup) | `test_evaluator_lookup.py::test_per_dimension_method_dict` (line 152) | `evaluate(method={'g0':'lookup','g1':'lookup','g2':'linear'})` |
| **F2 anti-regression: lookup + N copies of same interp** | **`test_evaluator_lookup.py::test_dict_lookup_plus_two_same_interp_works` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'linear','g2':'linear'})`** |
| **F2: mixed linear+cubic (no lookup) rejected** | **`test_evaluator_lookup.py::test_dict_mixed_linear_cubic_raises` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'linear','g1':'linear','g2':'cubic'})`** |
| **F2: lookup + mixed linear+cubic rejected** | **`test_evaluator_lookup.py::test_dict_lookup_plus_mixed_interp_raises` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'linear','g2':'cubic'})`** |
| **F2: lookup + nearest + linear rejected (different orders)** | **`test_evaluator_lookup.py::test_dict_lookup_plus_nearest_plus_linear_raises` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'nearest','g2':'linear'})`** |
| **F2 anti-regression: all-lookup dict ≡ scalar method='lookup'** | **`test_evaluator_lookup.py::test_dict_all_lookup_unchanged` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'lookup','g2':'lookup'})`** |
| **C3: 'nearest' and 'nearest_fast' treated as equivalent (both order 0)** | **`test_evaluator_lookup.py::test_dict_nearest_plus_nearest_fast_works` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'nearest','g2':'nearest_fast'})`** |
| **C10: unknown method string in dict raises** | **`test_evaluator_lookup.py::test_dict_unknown_method_raises` (13.16.GB-FIX2)** | **`evaluate(method={'g0':'lookup','g1':'linaer','g2':'linear'})`** |
| **F3: evaluate() docstring documents 'lookup' method** | **`test_evaluator_lookup.py::test_evaluate_docstring_mentions_lookup` (13.16.GB-FIX2)** | **`GroupByRegressionEvaluator.evaluate.__doc__`** |
| **F3: evaluate() docstring documents per-dimension dict shape** | **`test_evaluator_lookup.py::test_evaluate_docstring_mentions_dict` (13.16.GB-FIX2)** | **`GroupByRegressionEvaluator.evaluate.__doc__`** |
| **C8: docstring examples run successfully against real evaluator** | **`test_evaluator_lookup.py::test_evaluate_docstring_examples_run` (13.16.GB-FIX2)** | **`evaluate(method='lookup')` and `evaluate(method=dict)`** |
| **F4: method='lookup' + bounds='extrapolate' rejected at dispatch** | **`test_evaluator_lookup.py::test_lookup_with_extrapolate_bounds_raises` (13.16.GB-FIX2)** | **`evaluate(method='lookup', bounds='extrapolate')`** |
| **F5: auto-dispatch with n_jobs=1 returns 'numba' (Phase 12.11 behavior)** | **`test_phase_12_9_gb.py::TestBackendSelection::test_select_backend_auto_sequential` (13.16.GB-FIX2)** | **`_select_parallel_backend('auto', n_jobs=1)`** |
| `n_sigma_cut=None` is a no-op | `test_sigma_cut.py::test_sigma_cut_none_identical` (line 97) | `make_sliding_window_aggregate(n_sigma_cut=None)` |
| Sigma cut recovers true mean on outlier data | `test_sigma_cut.py::test_sigma_cut_recovers_true_mean` (line 158) | `make_sliding_window_aggregate(n_sigma_cut=3.0)` |
| SW aggregate ≡ fit path (mean/std) | `test_sliding_window_aggregate.py::test_aggregate_matches_fit_path` (line 116) | `make_sliding_window_aggregate` |
| SW aggregate parallel ≡ serial | `test_sliding_window_aggregate.py::test_aggregate_parallel_matches_serial` (line 287) | `make_sliding_window_aggregate_parallel` |
| WLS weights change coefficients | `test_wls_weights.py::test_wls_changes_coefficients` (line 86) | `make_sliding_window_fit(weights=...)` |
| WLS recovers known slope | `test_wls_weights.py::test_wls_recovers_known_slope` (line 109) | `make_sliding_window_fit(weights=...)` |
| Expression columns ≡ pre-materialized (V4) | `test_expression_linear_columns.py::TestInvariance::test_expression_matches_materialized_v4` (line 200) | `make_parallel_fit_v4` |
| Expression columns ≡ pre-materialized (SW) | `test_expression_linear_columns.py::TestInvariance::test_expression_matches_materialized_sw` (line 221) | `make_sliding_window_fit` |
| Parallel SW with agg_columns ≡ serial | `test_parallel_sliding_window.py::TestParallelAggColumns::test_parallel_agg_matches_serial` (line 296) | `make_sliding_window_fit_parallel` |
| `boundary='symmetric'` in SW fit (V3b): corner shrinks | `test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_reduces_corner_window` | `make_sliding_window_fit(boundary='symmetric')` |
| `boundary='symmetric'` in SW fit (V3b): interior unchanged | `test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_interior_equals_full` | `make_sliding_window_fit(boundary='symmetric')` |
| `boundary='symmetric'` per-dimension dict in SW fit | `test_invariance_sliding_window.py::TestSWV3bBoundary::test_symmetric_per_dimension` | `make_sliding_window_fit(boundary={...})` |
| `boundary='symmetric'` + gaussian kernel in SW fit | `test_invariance_sliding_window.py::TestSWV3bInteraction::test_symmetric_gaussian_interior_same_as_full_gaussian` | `make_sliding_window_fit(boundary='symmetric', kernel='gaussian')` |

| F1: aggregate `boundary='full'` bit-identical to pre-fix baseline | `test_aggregate_boundary.py::test_aggregate_default_boundary_full_unchanged` | `make_sliding_window_aggregate(boundary='full')` |
| F1: aggregate `boundary='symmetric'` corner is unbiased | `test_aggregate_boundary.py::test_aggregate_symmetric_corner_is_unbiased` | `make_sliding_window_aggregate(boundary='symmetric')` |
| F1: aggregate symmetric interior ≡ full interior | `test_aggregate_boundary.py::test_aggregate_symmetric_interior_equals_full` | `make_sliding_window_aggregate(boundary='symmetric')` |
| F1: aggregate per-dimension boundary dict | `test_aggregate_boundary.py::test_aggregate_symmetric_per_dimension_dict` | `make_sliding_window_aggregate(boundary={...})` |
| F1: aggregate `boundary='periodic'` wraps at edges | `test_aggregate_boundary.py::test_aggregate_periodic_wraps_at_edges` | `make_sliding_window_aggregate(boundary='periodic')` |
| F1: aggregate `boundary='nonsense'` raises ValueError | `test_aggregate_boundary.py::test_aggregate_invalid_boundary_raises` | `make_sliding_window_aggregate` |
| F1: parallel aggregate with `boundary='symmetric'` ≡ serial | `test_aggregate_boundary.py::test_aggregate_parallel_symmetric_matches_serial` | `make_sliding_window_aggregate_parallel(boundary='symmetric')` |
| F1: aggregate `symmetric` + `n_sigma_cut=3.0` gates the sigma-cut recompute path + interior invariance | `test_aggregate_boundary.py::test_aggregate_symmetric_with_sigma_cut_gate_and_invariance` | `make_sliding_window_aggregate(boundary='symmetric', n_sigma_cut=3.0)` |
| F1 (ORACLE invariance): aggregate symmetric ≡ manual reference via `_get_neighbor_bins_v2` on a non-linear 2-D fixture | `test_aggregate_boundary.py::test_aggregate_symmetric_matches_manual_oracle` | `make_sliding_window_aggregate` vs hand-rolled Python reference |
| F1 (CROSS-BACKEND × 6 params): numba ≡ numpy for all boundary modes × all kernel call sites via monkeypatch on `_get_numba_agg_kernel` | `test_aggregate_boundary.py::test_aggregate_numba_equals_numpy_all_boundaries_all_paths[{full,symmetric,periodic},{None,3.0}]` | `make_sliding_window_aggregate` |
| F1 (PERIODIC arithmetic): shift invariance on sinusoidal fixture + topology invariant on 2-D fully-periodic grid | `test_aggregate_boundary.py::test_aggregate_periodic_shift_and_topology_invariance` | `make_sliding_window_aggregate(boundary='periodic')` |
| F1 (EXTERNAL ORACLE × 3 params): window=0 ≡ pandas groupby for all 3 boundary modes | `test_aggregate_boundary.py::test_aggregate_window_zero_equals_groupby_all_boundaries[{full,symmetric,periodic}]` | `make_sliding_window_aggregate(window_spec={...:0})` vs `df.groupby(...)` |
| F1 (CANARY × 6 params): constant-field preservation across all boundary modes × sigma-cut paths | `test_aggregate_boundary.py::test_aggregate_constant_field_invariance_all_modes[{full,symmetric,periodic},{None,3.0}]` | `make_sliding_window_aggregate` |
| F1: aggregate `boundary='full'` corner is biased (reference documenting removed bias) | `test_aggregate_boundary.py::test_aggregate_full_corner_is_biased` | `make_sliding_window_aggregate(boundary='full')` |
| F1: aggregate `boundary='symmetric'` corner count shrinks to 1-neighbour window | `test_aggregate_boundary.py::test_aggregate_symmetric_count_at_corner` | `make_sliding_window_aggregate(boundary='symmetric')` |
| F1: aggregate `boundary='full'` corner count is 2-neighbour window (reference) | `test_aggregate_boundary.py::test_aggregate_full_count_at_corner` | `make_sliding_window_aggregate(boundary='full')` |

### Unverified Claims

| Claim | Status | Action |
|-------|--------|--------|
| Evaluator `'lookup'` is the fastest method on integer grids | Estimated from algorithmic complexity; **not measured at production scale** | Profile after first production run; add hard performance gate in future phase |
| Evaluator `'linear'` is ~3× faster than `'multilinear'` | Directional estimate from small synthetic workloads | Re-benchmark with profile artifacts in a future update |
| Evaluator `'cubic'` is ~2× slower than `'linear'` | Directional estimate | Same as above |
| ~~`boundary='symmetric'` in `make_sliding_window_aggregate`~~ | ✅ **Fixed in Phase 13.17.GB**, see Verified Claims table above (16 entries from `test_aggregate_boundary.py`) | — |
| `boundary` + `agg_median=True` interaction in `make_sliding_window_aggregate` | **⚠️ Median subpath silently uses `'full'`.** Verified broken by smoke test 2026-04-09. Mean/std/count fully fixed. | Fix in Phase 13.17.GB-MedianFix |
| Per-dimension `boundary='periodic'` correctness in SW fit | Implemented in V3b, no dedicated test | Add invariance tests in 13.17.GB (aggregate path) or separate cleanup |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-14 | Initial version |
| 2.0 | 2026-02-15 | Added scope/target, V2/V3/class APIs, benchmark framework |
| 2.1 | 2026-02-15 | 8-reviewer fixes |
| 3.0 | 2026-03-26 | Added Phases 13.10–13.15. New functions: `make_sliding_window_aggregate`, `make_nonlinear_sliding_window_fit`, expression columns. WLS fix, `fit_intercept` fix, lean output, sigma cut. O2DistAI Q&A. 500 tests, 133 features. |
| 3.1 | 2026-04-07 | Phase 13.16.GB evaluator methods. P0 `fit_intercept` fix. Coverage Map added. **Returned for corrections — 2 APPROVED / 3 CHANGES REQUESTED across 5 reviewers (wrong phase references, non-runnable examples, wrong function names, aggregate-vs-fit distinction missing).** |
| **3.2** | **2026-04-07** | **Corrections from v3.1 multi-reviewer cycle. Phase references corrected to `13.16.GB-FIX2` and `13.17.GB`. `boundary='symmetric'` split into two rows (⚠️ fit path tested, 🧨 aggregate path silently broken). All three Quick Start examples made runnable with keyword `df=`. `register_fit_model()` correct name in Public Interface Catalog. `'nearest_fast'` added to Evaluator Method Reference. `from_dfGB()` example uses real keyword arguments. F1 and F2 added to Current State "Broken" count. Governance reference updated to Org-structure v1.25. All v3.0 sections preserved verbatim. Drafted by Main Reviewer (Claude20, GBAI) at architect request after v3.1 review cycle.** |
| **3.3** | **2026-04-07** | **Phase 13.16.GB-FIX2 landed. F2/F3/F4/F5 fixed in source (evaluator method=dict validation, docstring coverage, lookup+extrapolate rejection, stale backend test). C3/C8/C10 addressed as part of the same commit ('nearest'/'nearest_fast' equivalence, runnable docstring examples, unknown-method detection). 11 new tests in `test_evaluator_lookup.py`, all with explicit path parameters per failure mode #11 and `pytest.raises(..., match=...)` per C7. Test count 517 → 528 canonical / 493 → 505 coder-env. Pre-existing failures 3 → 2 (F5 flipped). Broken count 2 → 1 (F2 fixed; F1 remains for 13.17.GB). All v3.0 sections preserved verbatim. Drafted by Coder (Claude21, GBAI) during implementation of PHASE_13_16_GB_FIX2_v1.0 proposal as consolidated in Claude20's review summary.** |
| **3.4** | **2026-04-11** | **Phase 13.17.GB landed.** F1 fixed for mean/std/count in both the primary kernel call and the sigma-cut recompute kernel call via the new `_precompute_aggregate_boundary_mask` helper (Path 2: mask + per-bin wrap flag + compact per-edge-bin wrapped-coords table). Both numba JIT and numpy fallback receive the new `(valid_offset_mask, wrap_flag, wrap_idx, wrapped_coords)` parameters in identical order at all 4 call sites (verified via signature-chain grep). Zero behavioural change on the default `boundary='full'` path — verified strictly by T9 against a literal baseline array hardcoded in the test file. Parallel wrapper inherits the fix automatically (T10 parallel ≡ serial invariance, 2-D fixture). D1 (median path boundary honouring) deferred to Phase 13.17.GB-MedianFix per architect direction 2026-04-09. **28 new pytest-level test runs** in `tests/test_aggregate_boundary.py` from 16 test functions: T1–T11 (T11 with second invariance assertion block), T13 oracle, T14 cross-backend × 6, T15 shift + topology, T16 window=0 × 3, T17 canary × 6. Unified Claude22 + Claude23 joint plan, **50% invariance ratio** (8/16 functions are invariance-style — up from v1.2's 25%). T12 removed per D1 deferral. **Test count: 528 corrected baseline + 28 new = 556 passed canonical** on alma2 commit `85713774`, branch `feature/groupby-optimization`, `test_logs/SUMMARY_20260411_090322.txt`. Pre-existing failures: 3 (was 2 in v3.3 corrected; +1 from `test_v3_numpy_faster_than_v1_numpy` newly listed). Capability Matrix: 0 broken / 0 partial / 43 verified / 89 smoke-only / 1 planned / 133 total. **Inherited arithmetic correction:** v3.3 records 528 passed canonical (off by one — forgot the F5 flip in FIX2); the *correct* v3.3 baseline is 529, but the *empirical* pre-13.17.GB passed-count was 528 because the timing test was already silently failing. v3.4 surfaces both numbers transparently. **New Known Limitation rows added:** D1 median deferral (with verbatim architect quotes including typos), pre-existing broken `test_aggregate_numba_matches_numpy` (superseded by T14, NOT deleted in-phase), `test_v3_numpy_faster_than_v1_numpy` timing test. **Public Interface Catalog unchanged** — Phase 13.17.GB extends behaviour of an existing parameter (`boundary`) without changing any function signature. **All v3.0 / v3.2 / v3.3 sections marked `[UNCHANGED]` preserved verbatim.** Drafted by Coder Claude22 during Phase 13.17.GB implementation; commit-time Main Reviewer Claude21. Suggested archive filename: `GroupByRegression_Technical_Summary_PHASE_13_17_GB_v3_4.md`. |
