# Technical Summary: GroupBy Regression — Update v3.0

**Author:** Claude14 (Main Reviewer)
**Date:** 2026-03-26
**Phase:** 13.15.GB
**Version:** 3.0 (update to v2.1)

> **Changes from v2.1:** New functions, updated API contracts, answers to O2DistAI Q&A.
> Sections marked **[NEW]** or **[UPDATED]** indicate changes from v2.1.

---

## [UPDATED] Subproject Scope and Target

| # | Capability | Status |
|---|-----------|--------|
| 1 | Linear per-bin regression (OLS/WLS/GLM/RLM/Huber) | ✅ Complete |
| 2 | Sliding window regression (N-D neighbor smoothing) | ✅ Complete |
| 3 | Parallel execution (multi-sector, multi-worker) | ✅ Complete |
| 4 | Function evaluator and interpolator (linear models) | ✅ Complete (Phase 13.9.GB) |
| 5 | **Non-linear GroupBy regression (binned spectra)** | ✅ Complete (Phase 13.10.GB) |
| 6 | **Expression-based linear columns** | ✅ Complete (Phase 13.13.GB) |
| 7 | **Dedicated sliding window aggregation** | ✅ Complete (Phase 13.14.GB) |
| 8 | **Sigma-clipped robust aggregation** | ✅ Complete (Phase 13.15.GB) |
| 9 | THn interface (histogram fitting) | 📋 Proposed (Phase 13.12.GB) |
| 10 | Integration interfaces (ADF, RDataFrameDSL, RootInteractive) | 📋 Planned |

---

## [UPDATED] Quick Start

```python
# Per-bin regression (recommended)
from dfextensions.groupby_regression import make_parallel_fit_v4
df_out, dfGB = make_parallel_fit_v4(
    df, gb_columns=['sector', 'padRow'], fit_columns=['driftV'],
    linear_columns=['spaceCharge'], suffix='_fit',
)

# Sliding window regression
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_fit, make_sliding_window_fit_parallel,
)
result = make_sliding_window_fit(
    dfGB, gb_columns=['padRow', 'zBin'],
    fit_columns=['driftV_intercept_fit'], linear_columns=[],
    window_spec={'padRow': 1, 'zBin': 2},
    agg_columns=['padRow', 'zBin'],  # COG of groupby coordinates
)

# [NEW] Pure aggregation (no regression — fastest for large grids)
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_aggregate, make_sliding_window_aggregate_parallel,
)
result = make_sliding_window_aggregate(
    df, gb_columns=['row_bin', 'driftM_bin', 'dsecM_bin', 'mP4_bin'],
    agg_columns=['xM', 'driftM', 'tgSlp', 'dy', 'dz'],
    window_spec={'row_bin': 1, 'driftM_bin': 1, 'dsecM_bin': 1, 'mP4_bin': 1},
    n_sigma_cut=3.0,  # robust 2-pass statistics
)

# [NEW] Expression-based polynomial fitting
result = make_parallel_fit_v4(
    df, gb_columns=['sector'], fit_columns=['dy'],
    linear_columns=[
        'xM',
        ('xM2', 'xM**2'),
        ('xM_driftM', 'xM * driftM'),
    ],
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
Need pure aggregation (fast)?   → make_sliding_window_aggregate()     [NEW]
Need parallel aggregation?      → make_sliding_window_aggregate_parallel()  [NEW]
Need non-linear spectrum fit?   → make_nonlinear_sliding_window_fit()  [NEW]
Need polynomial expressions?    → linear_columns=[('key','expr')]      [NEW]
```

---

## [NEW] Public Interface Catalog (Additions)

| Function | Module | Purpose | Stability | Phase |
|----------|--------|---------|-----------|-------|
| `make_sliding_window_aggregate()` | `groupby_regression_sliding_window` | Pure SW aggregation (sufficient stats) | Stable | 13.14.GB |
| `make_sliding_window_aggregate_parallel()` | `groupby_regression_sliding_window` | Parallel pure aggregation | Stable | 13.14.GB |
| `make_nonlinear_sliding_window_fit()` | `groupby_regression_nonlinear` | Non-linear SW fit (named models, custom callables) | Stable | 13.10.GB |
| `GroupByRegressionEvaluator` | `groupby_regression_evaluator` | Evaluate fitted models at arbitrary coordinates | Stable | 13.9.GB |
| `get_model()` | `groupby_regression_models` | Named model registry lookup | Stable | 13.10.GB |
| `register_model()` | `groupby_regression_models` | Register custom fit models | Stable | 13.10.GB |

---

## [NEW] Sliding Window Aggregation API

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
| `boundary` | str/dict | 'full' | Boundary handling |
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

---

## [UPDATED] Sliding Window Fit API Changes

### `agg_columns` parameter [NEW]

Computes mean/std of arbitrary columns within each sliding window:

```python
result = make_sliding_window_fit(
    ...,
    agg_columns=['mpt', 'vertex_z', 'tgl', 'phi'],  # COG columns
    agg_median=False,
)
# Output: mpt_mean_sw, mpt_std_sw, vertex_z_mean_sw, ...
```

**Canonical usage:** `agg_columns = gb_columns + linear_columns`

### WLS weights [FIXED]

The `weights` parameter now correctly affects regression (not just aggregation):
- V1 numpy: `sqrt(w)` transform on X, y
- V2 numba: falls back to V1 when weights present
- V3 incremental: weighted sufficient stats (XtX = Xw'Xw)
- V5: falls back to V3 when weights present

R²/RMSE computed on **unweighted** residuals (documented convention).

### `fit_intercept=False` [FIXED]

When `fit_intercept=False`, output no longer contains `{t}_intercept` or `{t}_intercept_err` columns.

### Default output columns [CHANGED — Breaking]

Per-target aggregation stats (`_mean`, `_std`, `_median`, `_entries`, `_r_squared`) **no longer exported by default**. To restore, add the target to `agg_columns`.

### Expression-based linear columns [NEW]

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
| V3/V5 incremental: median=NaN | By design | Use V1/V2 (recompute) or `agg_median=True` in aggregate |
| Parallel SW: Windows OS | By design | Linux/macOS only |
| WLS not supported in parallel V5 | By design | Use serial, or split manually |
| Chi2/NDF computation for non-linear SW | Under investigation | Phase 13.11.GB |
| SW histogram accumulation (spectrum fitting) | Proposed | Phase 13.12.GB |
| Expression columns in parallel SW | Deferred | Serial only currently |
| Python multiprocessing overhead for aggregation | Inherent | Use serial with Numba prange |

---

## [UPDATED] Current State

| Metric | v2.1 (Feb 2026) | v3.0 (Mar 2026) |
|--------|-----------------|-----------------|
| Test count | 338 passed | **500 passed** |
| Features | 102 | **133** |
| Verified | 25 (24.5%) | **43 (32.3%)** |
| Broken | 0 | 0 |
| Pre-existing failures | 3 | 4 |
| Public functions | 14 | **20** |

---

## [NEW] O2DistAI Team Q&A

**Q1 (API Contract):** `make_sliding_window_aggregate` is a separate implementation (not a fit with zero linear_columns). Output columns: `{col}_mean_sw`, `{col}_std_sw`, `{col}_count_sw` — this is the guaranteed contract.

**Q2 (Selection):** Yes, `selection` parameter is supported. It is a boolean array (positional, same length as `df`), not index-aligned.

**Q3 (Output Schema):** `_count_sw` is **per-column** (can differ if NaN patterns differ). `_std_sw` is standard deviation, not standard error. Compute SE as `std / sqrt(count)`.

**Q4 (Memory):** With 9 agg_columns and 7.6M rows, sufficient stats memory ≈ `n_bins × 9 × 3 × 8` bytes. All columns materialized simultaneously. No caching between calls.

**Q5 (Weights):** Yes, `weights` parameter is supported for weighted mean/std.

**Q6 (Documentation):** This document (v3.0) covers the API.

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-14 | Initial version |
| 2.0 | 2026-02-15 | Added scope/target, V2/V3/class APIs, benchmark framework |
| 2.1 | 2026-02-15 | 8-reviewer fixes |
| **3.0** | **2026-03-26** | **Added Phases 13.10–13.15. New functions: make_sliding_window_aggregate, make_nonlinear_sliding_window_fit, expression columns. WLS fix, fit_intercept fix, lean output, sigma cut. O2DistAI Q&A. 500 tests, 133 features.** |
