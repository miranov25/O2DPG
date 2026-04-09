# Technical Summary: GroupBy Regression

**Version:** 3.3
**Phase:** 13.16.GB-FIX2
**Last Updated:** 2026-04-07
**Coder:** Claude21 (GBAI team)
**Suggested archive filename:** `GroupByRegression_Technical_Summary_PHASE_13_16_GB_FIX2_v3_3.md`

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
| 7 | Dedicated sliding window aggregation | ✅ Complete (Phase 13.14.GB) |
| 8 | Sigma-clipped robust aggregation | ✅ Complete (Phase 13.15.GB) |
| 9 | **Evaluator: scipy interpolation + lookup + per-dimension methods** | **✅ Complete (Phase 13.16.GB)** |
| 10 | THn interface (histogram fitting) | 📋 Planned (Phase 13.12.GB) |
| 11 | Integration interfaces (ADF, RDataFrameDSL, RootInteractive) | 📋 Planned |

**Next scheduled phases** (in order):
1. ~~`13.16.GB-FIX2`~~ — ✅ **Completed 2026-04-07** (this document is the FIX2 summary update). Evaluator bug fixes F2/F3/F4/F5 plus C3/C8/C10 improvements. 11 new tests. Coder: Claude21.
2. `13.17.GB` — `boundary='symmetric'` in `make_sliding_window_aggregate` (silent-drop bug fix). Tier 1 review. ~2.5 days. Proposal drafted, pending review.
3. `13.11.GB` / `13.12.GB` / `12.15.GB` — planned, unordered.

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
# ⚠️ NOTE: boundary='symmetric' is currently silently ignored in this function.
# Fix scheduled in Phase 13.17.GB. See § Known Limitations.

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
  SW aggregation?               → wait for Phase 13.17.GB          [BUG]
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
| `boundary` | str/dict | 'full' | Boundary handling. **⚠️ `'symmetric'` and `'periodic'` silently ignored — see Known Limitations. Fix in Phase 13.17.GB.** |
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
| **`boundary='symmetric'` in `make_sliding_window_fit` (V3b / V4)** | **⚠️ Implemented, 4 V3b tests pass** | **Use the fit path as currently tested. See note below.** |
| **`boundary='symmetric'` in `make_sliding_window_aggregate` (serial and parallel)** | **🧨 Silently broken** | **Use `boundary='full'` explicitly. See note below. Fix in Phase 13.17.GB.** |
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

### Note: `boundary='symmetric'` — Two Distinct Statuses

**Architect requirement (original sliding window proposal, October 2025, quoted in PHASE_HISTORY § Incident 4):**

> "For TPC – drift, radius, and rphi – all should be symmetric. **I do not want to introduce edge bias.**"
>
> — Main Architect (MI)

**Architect clarification (2026-04-07, this phase):**

> "In the TPC use case, we want usually option A 'Truncate to symmetric extent'. This will be our usual default. All option A. We are calibrating diffs."
>
> — Main Architect (MI)

**Definition (Option A, "truncate to symmetric extent"):** For a 1-D grid `[lo..hi]` with window half-width `w` at center bin `c`: `eff_w(c) = min(w, c-lo, hi-c)`, window = `[c - eff_w, c + eff_w]`. The window stays symmetric around the center and shrinks near edges so it never extends past the observed bin range. Interior bins use the full `2w+1` neighborhood; edge bins use less data; no asymmetric bias.

**Status in `make_sliding_window_fit` (V3b path, Phases 13.8.GB–):**
- Implemented in `_get_neighbor_bins_v2` at `groupby_regression_sliding_window.py:412-418` (symmetric) and `:420-426` (periodic)
- 4 invariance tests passing in `test_invariance_sliding_window.py::TestSWV3bBoundary`
  - `test_symmetric_reduces_corner_window`
  - `test_symmetric_interior_equals_full`
  - `test_symmetric_per_dimension`
  - `test_symmetric_gaussian_interior_same_as_full_gaussian`
- **Use with confidence for the fit path.**

**Status in `make_sliding_window_aggregate` (and parallel variant, Phase 13.14.GB):**
- **🧨 Silently broken.** Parameter accepted at line 4224, validated at line 4359 via `_resolve_boundary`, then `boundary_resolved` is **never read again** in the function body (lines 4359–4549). The numba accumulation kernel and the numpy fallback receive only the unfiltered `neighbor_offsets`. Both `'symmetric'` and `'periodic'` are silently dropped — output is identical to `boundary='full'`.
- **Instance #5 of the parameter-not-propagated bug class.** See PHASE_HISTORY § Incident 5 (to be added when 13.17.GB closes).
- **Impact:** Edge bins of every grid computed with `boundary='symmetric'` in this function carry a bias of approximately `0.5 × (local gradient × window size)` in the direction of the grid interior. Interior bins (distance ≥ `window` from every edge) are unaffected. For TPC distortion calibration, this means maps produced with the current version have silently used asymmetric windows at the boundaries of the drift, radius, and rphi dimensions.
- **Verified by smoke test on a linear-rise-in-dsector fixture:** corner bin 0 with `window=1` returned 100 entries / mean ≈ 0.5 instead of the expected 50 entries / mean ≈ 0.0.
- **Fix scheduled in Phase 13.17.GB** (proposal drafted, pending Tier 1 review).
- **Workaround until fixed:** Use `boundary='full'` explicitly and restrict analysis to interior bins (distance ≥ `window` from every edge), or use `make_sliding_window_fit` with `linear_columns=[]` which routes through the correct V3b path (slower but correct).

---

## [UPDATED] Current State

| Metric | v2.1 (Feb 2026) | v3.0 (Mar 2026) | v3.2 (Apr 2026) | v3.3 (Apr 2026) |
|--------|-----------------|-----------------|-----------------|-----------------|
| Test count | 338 passed | 500 passed | 517 passed | **528 passed** |
| Pre-existing failures | 3 | 4 | 3 | **2** |
| Skipped | — | — | 19 | **19** |
| Features | 102 | 133 | 133 | **133** |
| Verified (✅) | 25 (24.5%) | 43 (32.3%) | 43 (32.3%) | **43+ (+11 new tests in test_evaluator_lookup.py)** |
| Smoke-only (☑️) | — | — | 89 (66.9%) | **89 (66.9%)** |
| Broken (🧨) | 0 | 0 | 2 (F1, F2) | **1 (F1 only — F2 fixed)** |
| Public functions | 14 | 20 | 20 | **20** |
| Evaluator method values | 2 | 3 | 6 (5 methods + per-dim dispatch) | **6 (5 methods + per-dim dispatch, now with detect-and-reject validation)** |

**Test count sourced from `run_tests.log` on the canonical machine** (`alma2`, Python 3.10.19, pytest-7.2.2-xdist, 12 workers). Reproducible via `bash run_tests.sh | tee run_tests.log`.

**Arithmetic:** 517 (v3.2) + 11 new tests in `test_evaluator_lookup.py` + 1 flipped (`test_select_backend_auto_sequential` fail→pass) = **528 passed on canonical machine**. Failures: 3 − 1 (F5 flipped) = **2**.

> **Coder env vs canonical env:** The FIX2 Coder (Claude21, implementation environment) runs on pandas 4.x which introduces 3 environmental failures and 21 additional skips unrelated to FIX2 scope. Coder-env numbers: **505 passed / 5 failed / 40 skipped**. Arithmetic: 493 baseline + 11 new + 1 flipped = 505. Failures: 6 baseline − 1 flipped = 5. Canonical-env numbers (above) will be produced when the Coder's commit is tested on `alma2`.

**The 2 remaining pre-existing failures** are unrelated to Phase 13.16.GB-FIX2 scope and remain deferred:
- `test_tpc_distortion_recovery::test_tpc_distortion_recovery` — test calls `make_sliding_window_fit` with wrong argument convention (test bug, not code bug); fix deferred to 13.17.GB or separate cleanup.
- `test_phase_12_8_gb::test_multiple_fits_match_v4_merged` — V5 vs V4 parity at `rtol=1e-12`; numerical, ambiguous, out of scope for FIX2 and 13.17.GB, separate micro-task.

**The 1 remaining broken-feature entry (F1)** is the `boundary='symmetric'` silent-drop in `make_sliding_window_aggregate`, scheduled for fix in Phase 13.17.GB (Tier 1 review, ~2.5 days). F2 (`method=dict` mixed interp) is now ✅ fixed and removed from the Broken count.

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

### Unverified Claims

| Claim | Status | Action |
|-------|--------|--------|
| Evaluator `'lookup'` is the fastest method on integer grids | Estimated from algorithmic complexity; **not measured at production scale** | Profile after first production run; add hard performance gate in future phase |
| Evaluator `'linear'` is ~3× faster than `'multilinear'` | Directional estimate from small synthetic workloads | Re-benchmark with profile artifacts in a future update |
| Evaluator `'cubic'` is ~2× slower than `'linear'` | Directional estimate | Same as above |
| `boundary='symmetric'` in `make_sliding_window_aggregate` | **🧨 Silently broken (verified by smoke test).** | Fix and invariance tests in Phase 13.17.GB |
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
