# Technical Summary: GroupBy Regression

**Author:** Claude13 (Main Reviewer, GBAI team)
**Date:** 2026-02-15
**Phase:** 13.8.SW (Parallel Sliding Window + Benchmarks)
**Version:** 2.1

> **This document supersedes** `docs/README.md` and `docs/groupby_regression.md` as the
> primary reference for GroupBy Regression until the full documentation overhaul after
> API freeze.

---

## Subproject Scope and Target

The GroupBy Regression subproject provides high-performance grouped regression for CERN/ALICE detector calibration. The intended final scope is:

| # | Capability | Status |
|---|-----------|--------|
| 1 | Linear per-bin regression (OLS/WLS/GLM/RLM/Huber) | ✅ Complete |
| 2 | Sliding window regression (3D neighbor smoothing) | ✅ Complete |
| 3 | Parallel execution (multi-sector, multi-worker) | ✅ Complete |
| 4 | **Function evaluator and interpolator (linear models)** | 📋 Planned (Phase 13.9.GB) |
| 5 | **Non-linear GroupBy regression (binned spectra)** | 📋 Planned |
| 6 | **THn interface** (multidimensional histogram fitting) | 📋 Planned |
| 7 | **Non-linear function evaluator** | 📋 Planned |
| 8 | **Integration interfaces** (ADF, RDataFrameDSL, RootInteractive) | 📋 Planned |

**Freeze strategy:** Once items 1–8 are delivered, the public API will be frozen. Integration contracts with dependent subprojects will be locked. After freeze, new functionality must be additive and backward compatible.

---

## Quick Start

```python
# Per-bin regression (recommended for all new code)
from dfextensions.groupby_regression import make_parallel_fit_v4
df_out, dfGB = make_parallel_fit_v4(
    df, gb_columns=['sector', 'padRow'], fit_columns=['driftV'],
    linear_columns=['spaceCharge'], suffix='_fit',
)

# Sliding window smoothing (recommended: incremental + numba)
from dfextensions.groupby_regression import make_sliding_window_fit
result = make_sliding_window_fit(
    dfGB, gb_columns=['padRow', 'zBin'],
    fit_columns=['driftV_intercept_fit'], linear_columns=[],
    window_spec={'padRow': 1, 'zBin': 2}, algorithm='incremental', backend='numba',
)
```

**Which function should I use?**

```
Need robust/Huber fitter?     → GroupByRegressor.make_parallel_fit()
Need OLS/WLS, single target?  → make_parallel_fit_v4()           ← recommended
Need OLS, multiple targets?   → make_parallel_fit_v5()
No Numba available?           → make_parallel_fit_v2() or v3()
Need sliding window?          → make_sliding_window_fit()
Need parallel over sectors?   → make_sliding_window_fit_parallel()
```

---

## Current State

The module supports two main workflows: per-bin regression (fitting independent models to thousands of detector bins) and sliding window regression (smoothing across neighboring bins in multidimensional grids). Five engine generations are available, from the original statsmodels-based robust engine to the current V5 Numba JIT incremental solver achieving 25–68× speedup over alternatives.

Recent work (Feb 2026) delivered parallel sliding window execution with fork() COW dispatch (3.3× scaling at 16 workers on 112M rows), V4-aligned API conventions, comprehensive invariance testing, and benchmark consolidation with fitted cost models. The module tracks 102 features (25 verified, 76 smoke-tested, 0 broken) through an automated capability matrix.

---

## Public Interface Catalog

### Complete Function Reference

| Function | Module | Purpose | Stability |
|----------|--------|---------|-----------|
| `GroupByRegressor.make_linear_fit()` | `groupby_regression` | OLS per-bin regression | Stable |
| `GroupByRegressor.make_parallel_fit()` | `groupby_regression` | Robust parallel regression (OLS/WLS/GLM/RLM/Huber) | Stable |
| `GroupByRegressor.summarize_diagnostics()` | `groupby_regression` | Diagnostic summary for robust fits | Stable |
| `GroupByRegressor.format_diagnostics_summary()` | `groupby_regression` | Format diagnostics as text | Stable |
| `make_parallel_fit_v2()` | `groupby_regression_optimized` | Process-parallel regression (loky) | Stable |
| `make_parallel_fit_v3()` | `groupby_regression_optimized` | Thread-parallel regression | Stable |
| `make_parallel_fit_v4()` | `groupby_regression_optimized` | Numba JIT regression (recommended for per-bin) | Stable |
| `make_parallel_fit_v5()` | `groupby_regression_optimized` | Batch multi-target regression (shared groupby) | Experimental |
| `GroupByRegressorOptimized` | `groupby_regression_optimized` | Class-based interface wrapping V2–V4 | Stable |
| `make_sliding_window_fit()` | `groupby_regression_sliding_window` | Sliding window regression (V1–V5 backends) | Stable |
| `make_sliding_window_fit_parallel()` | `groupby_regression_sliding_window` | Parallel sliding window over split columns | Stable |
| `make_sliding_window_fit_v5_arrays()` | `groupby_regression_sliding_window` | Low-level numpy array path (advanced) | Internal |
| `fit_groups_single_numba()` | `groupby_regression_kernels` | Single-target Numba kernel | Stable (frozen) |
| `fit_groups_multifit_numba()` | `groupby_regression_kernels` | Multi-target Numba kernel (XtX sharing) | Stable (frozen) |
| `fit_groups_dispatch()` | `groupby_regression_kernels` | Auto-select kernel | Stable (frozen) |

### Public API Freeze Boundary

| Category | Scope |
|----------|-------|
| **Public (will be frozen)** | All functions imported via `__init__.py`, `GroupByRegressor`, `GroupByRegressorOptimized`, `make_parallel_fit_v2/v3/v4`, `make_sliding_window_fit`, `make_sliding_window_fit_parallel` (after export) |
| **Internal (not frozen)** | `make_sliding_window_fit_v5_arrays`, `fit_groups_single_numba` (direct use discouraged), internal kernel helpers, `_build_bin_index_map`, `_counting_sort_indices` |

### Import Paths

```python
# Top-level imports (via __init__.py)
from dfextensions.groupby_regression import (
    GroupByRegressor,                  # Robust engine
    make_parallel_fit_v2,              # V2 loky
    make_parallel_fit_v3,              # V3 threaded
    make_parallel_fit_v4,              # V4 Numba (recommended)
    GroupByRegressorOptimized,         # Class interface
    make_sliding_window_fit,           # SW regression
)

# Full module path required (not yet in __init__.py — will be added)
from dfextensions.groupby_regression.groupby_regression_optimized import (
    make_parallel_fit_v5,              # V5 batch
)
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_fit_parallel,  # Parallel SW
)
from dfextensions.groupby_regression.groupby_regression_kernels import (
    fit_groups_dispatch,               # Kernel dispatch
)
```

> **Note:** `make_sliding_window_fit_parallel` and `make_parallel_fit_v5` will be added to `__init__.py` in the next commit.

---

## Per-Bin Regression API

### Recommended: `make_parallel_fit_v4()`

The fastest general-purpose per-bin regression. Use for all new code unless you need robust fitters (Huber, GLM, RLM).

```python
from dfextensions.groupby_regression import make_parallel_fit_v4

df_out, dfGB = make_parallel_fit_v4(
    df=df,
    gb_columns=['sector', 'padRow'],
    fit_columns=['driftV', 'dEdx'],          # targets
    linear_columns=['spaceCharge', 'temperature'],  # predictors
    weights='weightCol',                      # optional WLS
    median_columns=['r', 'phi'],              # optional group medians
    suffix='_fit',
    selection=df['quality'] > 0,              # optional row filter
    fit_intercept=True,
    min_stat=5,                               # minimum rows per group
    diag=True,                                # enable diagnostics
    cast_dtype='float32',                     # optional output dtype
    return_metadata=False,
    backend='auto',                           # 'auto'|'pandas'|'pyarrow'
    pyarrow_threshold=1_000_000,              # auto-switch to Arrow above this
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Input data |
| `gb_columns` | list[str] | required | Columns defining groups |
| `fit_columns` | list[str] | required | Target (dependent) variables |
| `linear_columns` | list[str] | required | Predictor (independent) variables |
| `weights` | str or None | None | Column name for WLS weights |
| `median_columns` | list[str] or None | None | Columns for group medians |
| `suffix` | str | '_v4' | Suffix appended to output columns |
| `selection` | Series/array or None | None | Boolean mask. Series must be aligned with `df.index`. NumPy array must be length `len(df)` (treated positionally). |
| `fit_intercept` | bool | True | Include intercept term |
| `min_stat` | int | 3 | Minimum rows per group for fitting |
| `diag` | bool | False | Enable diagnostic columns |
| `cast_dtype` | str | 'float64' | Output dtype ('float32', 'float16', etc.) |
| `return_metadata` | bool | False | Return `(df_out, dfGB, metadata)` tuple |
| `backend` | str | 'auto' | Storage backend: 'auto', 'pandas', 'pyarrow' |
| `pyarrow_threshold` | int | 1_000_000 | Row count above which 'auto' selects PyArrow |

**Returns:** `(df_out, dfGB)` — original DataFrame and group-level results. With `return_metadata=True`: `(df_out, dfGB, metadata)`. Note: `df_out` may be the same object as input `df` with added columns — do not assume it is a copy for large-scale workflows.

### Robust Engine: `GroupByRegressor.make_parallel_fit()`

Use when you need robust fitters (Huber, iterative reweighting, outlier rejection) or custom fitter functions. Slower (~26s/1K groups) but supports full statsmodels ecosystem.

```python
from dfextensions.groupby_regression import GroupByRegressor

df_out, dfGB = GroupByRegressor.make_parallel_fit(
    df=df,
    gb_columns=['sector', 'padRow'],
    fit_columns=['driftV'],
    linear_columns=['spaceCharge'],
    fitter='huber',                    # 'ols'|'huber'|'rlm'|callable
    sigma_cut=3.0,                     # outlier rejection threshold
    median_columns=['r'],
    suffix='_rob',
    selection=None,
    n_jobs=4,                          # parallel workers (loky)
)
```

### V2/V3 Engines

Same signature as V4. Use V2 for process-parallel on machines without Numba. Use V3 for thread-parallel (avoids fork overhead).

```python
from dfextensions.groupby_regression import make_parallel_fit_v2, make_parallel_fit_v3
df_out, dfGB = make_parallel_fit_v2(df=df, gb_columns=..., ...)  # loky
df_out, dfGB = make_parallel_fit_v3(df=df, gb_columns=..., ...)  # threads
```

### V5 Batch Engine: `make_parallel_fit_v5()`

Fits multiple targets sharing the same groupby in one call. Single sort + chunked processing.

```python
from dfextensions.groupby_regression.groupby_regression_optimized import make_parallel_fit_v5

result = make_parallel_fit_v5(
    df=df,
    fit_specs=[
        {'target': 'driftV', 'linear_columns': ['spaceCharge', 'temperature'], 'suffix': '_dV'},
        {'target': 'dEdx', 'linear_columns': ['spaceCharge', 'temperature'], 'suffix': '_dE'},
    ],
    gb_columns=['sector', 'padRow'],
    chunk_size=10000,                  # chunked processing for memory
    parallel_backend='auto',           # 'auto'|'numba'|'sequential'
)
```

**V5 constraints:** All targets in `fit_specs` must share the same `gb_columns`. All targets must share the same `selection` (if any). The `linear_columns` may differ per target but `gb_columns` cannot.

**Stability:** Experimental — the `fit_specs` interface may evolve.

### Class-Based Interface: `GroupByRegressorOptimized`

Wraps V2–V4 with object state for repeated fits on the same grouping. **VERIFY-IN-CODE:** the exact method name (`.fit()`, `.fit_v4()`, or `.make_parallel_fit()`) — check source before calling.

```python
from dfextensions.groupby_regression import GroupByRegressorOptimized

gbo = GroupByRegressorOptimized(
    df=df,
    gb_columns=['sector', 'padRow'],
    linear_columns=['spaceCharge'],
)
# Use appropriate method — verify exact name in source
result = gbo.fit(fit_columns=['driftV'], engine='v4')
```

---

## Sliding Window Regression API

### Recommended: `make_sliding_window_fit()`

Fits regression models in a sliding window over a multidimensional bin grid. Each bin's fit uses data from neighboring bins within the window radius.

```python
from dfextensions.groupby_regression import make_sliding_window_fit

result = make_sliding_window_fit(
    df=df,
    gb_columns=['rBin', 'y2xBin', 'z2xBin'],
    fit_columns=['dX', 'dY', 'dZ'],
    linear_columns=['x'],
    window_spec={'rBin': 1, 'y2xBin': 2, 'z2xBin': 1},
    suffix='_sw',
    selection=None,                     # row filter (applied before binning)
    fit_intercept=True,
    min_stat=10,
    algorithm='incremental',            # 'recompute'|'incremental'
    backend='numba',                    # 'numpy'|'numba'|'auto'
    boundary='symmetric',               # 'full'|'symmetric'|'periodic'
    kernel='uniform',                   # 'uniform'|'gaussian'|'epanechnikov'|'linear'|callable
    kernel_width=None,                  # support radius: sigma for Gaussian, compact radius for epanechnikov/linear
    weights=None,                       # WLS weight column — see note below
    return_metadata=False,
    verbose=False,
)
```

> **`weights` parameter note:** Currently only functional with `algorithm='recompute'` (V1/V2 backends). Silently ignored by incremental backends (V3/V5). Full WLS support for incremental backends is planned (see `SW.weighted` in capability matrix).

**Algorithm × Backend matrix:**

| Combination | Code Name | Complexity | Recommended? |
|-------------|-----------|------------|--------------|
| recompute + numpy | V1 | O(N_bins × N_nbr × RPB) | No — baseline |
| recompute + numba | V2 | O(N_bins × N_nbr × RPB) | No — ~15% faster than V1 |
| incremental + numpy | V3 | O(N_rows + N_bins × N_nbr) | Only if no Numba |
| **incremental + numba** | **V5** | **O(N_rows + N_bins × N_nbr)** | **Yes — always** |

**Boundary modes:**
- `'full'`: Only use neighbors that exist in the grid (edge bins have fewer neighbors)
- `'symmetric'`: Reflect at boundaries
- `'periodic'`: Wrap around (requires periodic dimensions, e.g., phi)

**Kernel functions:**
- `'uniform'`: Equal weight to all neighbors (default)
- `'gaussian'`: Gaussian weighting with `kernel_width` as sigma
- `'epanechnikov'`: Parabolic weighting (compact support within `kernel_width`)
- `'linear'`: Linear decay with distance (support within `kernel_width`)
- `callable`: Custom function `f(offset_array, sigma_array) -> float`

`kernel_width` is required when `kernel != 'uniform'`. It specifies the sigma (for Gaussian) or the compact support radius (for epanechnikov, linear).

### Parallel: `make_sliding_window_fit_parallel()`

Splits data by categorical columns (e.g., sector) and dispatches to worker processes.

```python
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_fit_parallel,
)

result = make_sliding_window_fit_parallel(
    df=df,
    gb_columns=['rBin', 'y2xBin', 'z2xBin'],
    fit_columns=['dX', 'dY', 'dZ'],
    linear_columns=['x'],
    split_columns=['sector'],           # parallelize over these
    n_workers=8,                        # number of processes
    window_spec={'rBin': 1, 'y2xBin': 2, 'z2xBin': 1},
    on_error='nan',                     # 'nan'|'raise'|'skip'
    verbose=0,                          # 0=quiet, 1=progress, 2=debug
)
```

**Returns:** Single concatenated DataFrame with results from all split units. Serial and parallel produce identical results (verified: max|diff| = 0.00e+00).

**Platform requirement:** Linux (fork() COW). macOS works with warnings. **Windows OS is not supported** (requires `fork()` for copy-on-write shared memory).

---

## Input/Output Contract

### Input Requirements

All functions expect a pandas DataFrame with:

| Column Type | Required | Dtype | Notes |
|-------------|----------|-------|-------|
| `gb_columns` | Yes | Any hashable (int, str, categorical) | Define the bin grid |
| `fit_columns` | Yes | float64 | Target variables |
| `linear_columns` | Yes | float64 | Predictor variables |
| `weights` | Optional | float64, > 0 | WLS weights (per-bin: all engines; SW: recompute only) |
| `selection` | Optional | bool Series or array | Series: must align with `df.index`. Array: must be length `len(df)`, treated positionally. |
| `split_columns` | For parallel | int or categorical | Sector/stack identifiers |
| `median_columns` | Optional | float64 | Columns for group medians (per-bin only) |

**Error handling:** Functions raise `ValueError` for invalid parameters (unknown columns, incompatible types, invalid algorithm/backend combinations). `KeyError` for missing columns. In parallel mode, worker exceptions are re-raised unless `on_error='nan'`.

### Output Column Schema

**Per-bin regression** output (`dfGB` DataFrame, one row per group):

| Column Pattern | Dtype | Description |
|----------------|-------|-------------|
| `{gb_col}` | same as input | Group key columns (index) |
| `{target}_intercept{suffix}` | float64 | Fitted intercept |
| `{target}_slope_{predictor}{suffix}` | float64 | Fitted slope for each predictor |
| `{target}_intercept_err{suffix}` | float64 | Standard error of intercept |
| `{target}_slope_{predictor}_err{suffix}` | float64 | Standard error of slope |
| `{target}_rms{suffix}` | float64 | Root mean square of residuals |
| `{target}_mad{suffix}` | float64 | Median absolute deviation (robust engine only; V4 may return NaN) |
| `{median_col}{suffix}` | float64 | Group median (if `median_columns` given) |
| `nEffective{suffix}` | int64 | Effective number of rows per group |

**Sliding window regression** output (DataFrame, one row per bin):

| Column Pattern | Dtype | Description |
|----------------|-------|-------------|
| `{gb_col}` | same as input | Bin coordinates |
| `{target}_intercept{suffix}` | float64 | Window-averaged intercept |
| `{target}_slope_{predictor}{suffix}` | float64 | Window-averaged slope |
| `{target}_intercept_err{suffix}` | float64 | Standard error |
| `{target}_slope_{predictor}_err{suffix}` | float64 | Standard error |
| `{target}_rms{suffix}` | float64 | Root mean square of residuals |
| `{target}_std{suffix}` | float64 | Standard deviation of target in window |
| `{target}_r2{suffix}` | float64 | R² goodness of fit |
| `{target}_median{suffix}` | float64 | Median (NaN for V3/V5 incremental — by design, cannot compute from sufficient statistics) |

**NaN semantics:** A row contains NaN values when the bin had fewer than `min_stat` rows, or when the regression failed (singular matrix, all-constant predictors).

### Metadata (when `return_metadata=True`)

```python
df_out, dfGB, metadata = make_parallel_fit_v4(..., return_metadata=True)

metadata = {
    'prediction_formulas': {'driftV_fit': 'driftV_intercept_fit + driftV_slope_spaceCharge_fit * spaceCharge'},
    'pull_formulas': {'driftV_pull_fit': '(driftV - driftV_fit) / driftV_rms_fit'},
    'columns': {
        'coefficients': ['driftV_intercept_fit', 'driftV_slope_spaceCharge_fit'],
        'errors': ['driftV_intercept_err_fit', 'driftV_slope_spaceCharge_err_fit'],
        'medians': ['r_fit'],
    },
}
```

---

## Kernel Dispatch (Advanced)

The shared kernel module provides direct access to Numba JIT kernels for advanced users.

```python
from dfextensions.groupby_regression.groupby_regression_kernels import fit_groups_dispatch

results = fit_groups_dispatch(
    X, Y, group_indices, n_groups,
    weights=None,
    invalid_handling='detect',    # 'assume_clean'|'detect'|'filter'
    fit_intercept=True,
)
```

**`invalid_handling` modes:**
- `'assume_clean'`: Skip NaN/Inf checks (fastest, use when data is pre-cleaned)
- `'detect'`: Check for NaN/Inf, mark in status bitmask (default)
- `'filter'`: Remove NaN/Inf rows per group before fitting (slowest but safest)

**Status bitmask** (uint8 per group):

| Bit | Meaning |
|-----|---------|
| 0 | Valid fit |
| 1 | Insufficient data (< min_stat) |
| 2 | Singular matrix |
| 3 | NaN/Inf detected in input |
| 4 | High condition number (κ > 1e10, ridge applied) |

---

## Benchmark Framework

The benchmark framework provides CLI tools for performance tracking:

```bash
# Run benchmarks via BF runner
python -m dfextensions.benchmarks.runner --subproject groupby_regression

# Benchmark history summary
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history

# Noise statistics (CV%) for alarm tuning
python -m dfextensions.benchmarks.runner --subproject groupby_regression --history-stats

# Generate trend plots
python -m dfextensions.benchmarks.runner --subproject groupby_regression --plot ./plots/
```

**Programmatic API:**

```python
from dfextensions.benchmarks.benchmark_adf import (
    load_benchmark_adf,
    compute_benchmark_statistics,
)

adf = load_benchmark_adf("groupby_regression", max_runs=20)
stats = compute_benchmark_statistics("groupby_regression", baseline="7d")
```

**Sliding window benchmarks** (standalone, not integrated with BF):

```bash
# Serial: all backends × grid × window × RPB
python benchmarks/bench_slidingwindow_parametric.py --quick

# Parallel: scaling curve, sort comparison
python benchmarks/bench_slidingwindow_parallel.py
```

---

## Integration Patterns

### Pattern 1: Direct DataFrame Workflow

The standard pattern for standalone use:

```python
import pandas as pd
from dfextensions.groupby_regression import make_parallel_fit_v4, make_sliding_window_fit

# Step 1: Per-bin regression
df_out, dfGB = make_parallel_fit_v4(
    df=raw_tracks,
    gb_columns=['sector', 'padRow', 'zBin'],
    fit_columns=['driftV'],
    linear_columns=['spaceCharge'],
    suffix='_fit',
)

# Step 2: Sliding window smoothing (on the group-level results)
smoothed = make_sliding_window_fit(
    df=dfGB,
    gb_columns=['sector', 'padRow', 'zBin'],
    fit_columns=['driftV_intercept_fit', 'driftV_slope_spaceCharge_fit'],
    linear_columns=[],    # smoothing only, no additional regression
    window_spec={'padRow': 1, 'zBin': 2},
    suffix='_smooth',
)
```

### Pattern 2: AliasDataFrame Integration (Current)

GroupBy regression results are consumed by AliasDataFrame through subframes and metadata:

```python
from dfextensions.alias_data_frame import AliasDataFrame
from dfextensions.groupby_regression import make_parallel_fit_v4

# Fit (with metadata for formula access)
df_out, dfGB, metadata = make_parallel_fit_v4(
    df=adf.df, gb_columns=['sector', 'padRow'],
    fit_columns=['driftV'], linear_columns=['spaceCharge'],
    suffix='_fit', return_metadata=True,
)

# Wrap group-level results as AliasDataFrame
calibFit_adf = AliasDataFrame(dfGB)

# Register as subframe (correct API — verified against Phase 13.6.B contract tests)
adf.register_subframe('calibFit', calibFit_adf, index_columns=['sector', 'padRow'])

# Access via dot notation after registration
corrected_driftV = adf['calibFit.driftV_intercept_fit']  # left-join lookup

# Prediction formulas available in metadata for RootInteractive visualization
print(metadata['prediction_formulas'])
```

### Pattern 3: Evaluator Integration (Planned — Phase 13.9.GB)

The planned `GroupByRegressionEvaluator` will provide a functional representation. Evaluator introduction **will not change existing fit APIs** — it consumes `dfGB` output.

```python
# PLANNED API — not yet implemented
from dfextensions.groupby_regression import GroupByRegressionEvaluator

# Create evaluator from fit results
evaluator = GroupByRegressionEvaluator.from_fit_result(
    dfGB,
    gb_columns=['padRow', 'zBin'],
    fit_columns=['driftV'],
    linear_columns=['spaceCharge'],
    suffix='_fit',
    interpolation='multilinear',     # 'nearest'|'multilinear'|'cubic'
    missing='nearest_valid',         # 'nan'|'skip'|'nearest_valid'
)

# Evaluate at arbitrary coordinates
predicted = evaluator.evaluate(
    padRow=15.5, zBin=42.3,          # float coordinates (bin centers)
    spaceCharge=0.85,
)

# Serialize for browser-side evaluation (RootInteractive)
json_repr = evaluator.to_json()

# Integration with AliasDataFrame (planned — will follow register_* convention)
# Method name TBD; will be reviewed with ADF/RDataFrameDSL teams before implementation
adf.register_evaluator('driftV_model', evaluator)  # tentative name
```

**Planned evaluator capabilities:**
- Float coordinate system (bin centers, not indices)
- Configurable interpolation (nearest / multilinear / cubic)
- Sparse grid handling (nan / skip / nearest_valid)
- JSON export for browser-side evaluation in RootInteractive
- Interface compatible with AliasDataFrame subframe and RDataFrameDSL table patterns

> **Note:** The evaluator API is a proposal and will be reviewed with ADF and RDataFrameDSL teams before Phase 13.9.GB implementation begins. Feedback welcome.

---

## Memory Requirements

| Engine | Working Memory | Notes |
|--------|----------------|-------|
| V4 per-bin | ~100 bytes/group | In-place accumulation |
| V5 batch | ~100 bytes/group × n_targets | XtX sharing when possible |
| SW V5 incremental | ~20 bytes/row + ~100 bytes/bin | Sufficient statistics |
| Parallel SW | Fork COW — no array duplication | Each worker shares parent data via copy-on-write |
| PyArrow backend | ~16 bytes/row for sort indices | Auto-selected above `pyarrow_threshold` rows |

At TPC scale (112M rows, 54K bins), parallel SW uses ~1.7 GB shared + ~200 MB/worker.

---

## Performance Reference (Estimated)

### Engine Comparison (Per-Bin Regression)

| Engine | Throughput | vs Robust | When to Use |
|--------|------------|-----------|-------------|
| Robust (statsmodels) | ~40 groups/s | 1× | Need Huber/GLM/RLM fitters |
| V2 (loky) | ~3.4K groups/s | 85× | No Numba, need parallelism |
| V3 (threads) | ~3.4K groups/s | 85× | No Numba, avoid fork overhead |
| V4 (Numba) | ~530K groups/s | 17,000× | **Recommended for all OLS/WLS** |
| V5 batch | ~530K groups/s | 17,000× | Multiple targets, single sort |

### Sliding Window (Apple M1 Pro, per-sector)

| Config | V1 | V5 | Speedup |
|--------|------|------|---------|
| 10³ grid, W=1, 10 rpb | 0.114s | 0.005s | 25× |
| 25³ grid, W=1, 10 rpb | 1.867s | 0.050s | 44× |
| 25³ grid, W=2, 10 rpb | 3.515s | 0.089s | 68× |
| 25³ grid, W=1, 50 rpb | 3.377s | 0.086s | 40× |

### TPC Production Estimates (Estimated — Extrapolated from Cost Model)

| Scenario | Serial (36 sectors × 3 targets) | 10-way parallel |
|----------|--------------------------------|-----------------|
| Standard (54K bins, 1K rpb) | ~3.4 min | ~20s |
| High (54K bins, 2K rpb) | ~6.8 min | ~40s |

### Cost Model (V5, fitted on M1 Pro)

```
T_V5tot = 0.060 × N_rows(µs) + 0.028 × N_bins × N_nbr(µs) + 1.45 × N_bins(µs) + 5.5ms
```

Scale to your machine: `your_time ≈ (your_V5tot_25³_W1_r10 / 0.050s) × reference_time`

Full cost models and scaling guidance: `benchmarks/README_sliding_window_benchmark.md`

---

## Known Limitations

### Functional

| Limitation | Status | Workaround |
|------------|--------|------------|
| SW weighted fits (WLS) | Planned | Use per-bin V4 with weights, then smooth results |
| V3/V5 incremental: median=NaN | By design | Use V1/V2 (recompute) if median needed |
| V5 batch: heterogeneous gb_columns | Not supported | All targets must share same `gb_columns` and same `selection` |
| Parallel SW: **Windows OS** not supported | By design | Linux/macOS only (fork required for COW) |
| Numba first-call JIT warmup | ~1s | Subsequent calls are fast |

### Documentation Gaps (being addressed)

| Issue | Priority | Status |
|-------|----------|--------|
| `docs/README.md` 4 months stale | P1 | Superseded by this document |
| `__init__.py` missing new exports | P1 | Next commit |
| `docs/groupby_regression.md` only covers robust API | P1 | Superseded by this document |

### Test Coverage

102 features tracked. 25 verified (24.5%), 76 smoke-only (74.5%), 0 broken. Key gap: most V4 optimized features and PyArrow backend are smoke-only (no analytical invariance tests).

3 pre-existing test failures (not regressions): `test_multiple_fits_match_v4_merged`, `test_select_backend_auto_sequential`, `test_tpc_distortion_recovery`.

---

## Cross-Subproject Dependencies

### What We Depend On

| Dependency | How Used | Min Version |
|------------|----------|-------------|
| NumPy | Core data structures | 1.21+ |
| Pandas | DataFrames, groupby | 1.3+ |
| Numba | JIT compilation for V4/V5/SW kernels | 0.56+ |
| statsmodels | OLS/WLS/GLM/RLM fitters in robust engine | 0.13+ |
| AliasDataFrame (RootInteractive) | Benchmark visualization (`benchmark_adf.py`) | — |
| Benchmark Framework (shared) | `runner.py`, `schema.py` | — |
| Python | Runtime | 3.9+ |

### What Others Need From Us

| Consumer | What They Use | Interface |
|----------|---------------|-----------|
| TPC calibration (O2DPG) | Per-bin regression + sliding window | `make_parallel_fit_v4()`, `make_sliding_window_fit()` |
| dE/dx calibration | Robust regression with outlier rejection | `GroupByRegressor.make_parallel_fit()` |
| RootInteractive notebooks | Per-bin regression for interactive analysis | `make_parallel_fit_v4()` + metadata |
| AliasDataFrame | Fit results as subframes | `register_subframe()` with `AliasDataFrame(dfGB)` |
| RDataFrameDSL | (planned) Evaluator objects | `GroupByRegressionEvaluator` (Phase 13.9.GB) |

### TPC Calibration Data Flow

```
Raw tracks (5M/s, 10–15h)
  → group by (sector × rBin × y2xBin × z2xBin)     [make_parallel_fit_v4]
  → per-bin regression: ~54K bins × 1K rows/bin
  → sliding window smoothing: W=(1,2,1), 3D          [make_sliding_window_fit_parallel]
  → distortion correction maps (dX, dY, dZ)
  → (planned) evaluator for interpolation at arbitrary points
```

---

## Upcoming Changes

### Phase 13.9.GB: GroupBy Regression Evaluator (Next)

The evaluator/interpolator class will become the primary integration point for cross-subproject use. Design status: proposal phase.

Key capabilities: evaluate fitted linear models at arbitrary float coordinates (interpolation between bin centers), serialize/deserialize as JSON for browser-side evaluation in RootInteractive, interface compatible with AliasDataFrame `register_subframe` and `register_evaluator` patterns.

This is the most important upcoming deliverable for dependent teams. The evaluator API will be reviewed with ADF and RDataFrameDSL teams before implementation.

The THn interface (item 6) will provide input THn → fit per-bin → output evaluator, enabling direct evaluator extraction from multidimensional histogram structures.

### Near-Term

| Change | Impact on Others |
|--------|------------------|
| `__init__.py` updated exports | New top-level imports available |
| Phase 12.15.GB: V4 shared kernel integration | None (backward compatible) |
| SW.weighted (WLS sliding window) | New capability, no API break |

### Medium-Term

| Change | Impact on Others |
|--------|------------------|
| Non-linear GroupBy regression | New function, new output columns |
| THn interface | New entry point for histogram-based fitting |
| Non-linear evaluator | Extension of evaluator class |
| ADF/RDataFrameDSL/RootInteractive integration | Standardized contracts |

### No Breaking Changes Planned

All upcoming changes are additive. V4 and sliding window APIs are frozen for current consumers. After scope freeze, all changes must be additive and backward compatible.

---

## Utility Files

| File | Purpose |
|------|---------|
| `synthetic_tpc_distortion.py` | Generate synthetic TPC distortion data for pipeline testing |
| `scripts/generate_capability_matrix.py` | Auto-generate capability matrix from test taxonomy |
| `scripts/profile_sw_variants.py` | Profile sliding window algorithm variants |
| `run_tests.sh` | Unified test runner with reviewer.zip packaging |

---

## Key Files

| File | Purpose |
|------|---------|
| `groupby_regression.py` | Robust engine (OLS/WLS/GLM/RLM/Huber) |
| `groupby_regression_optimized.py` | V2/V3/V4/V5 optimized engines |
| `groupby_regression_sliding_window.py` | Sliding window V1–V5 + parallel |
| `groupby_regression_kernels.py` | Shared Numba kernel module |
| `docs/PHASE_HISTORY.md` | Complete phase history (v4.0) |
| `docs/CAPABILITY_MATRIX.md` | 102-feature capability matrix |
| `benchmarks/README_sliding_window_benchmark.md` | SW performance guide with cost formulas |
| `tests/feature_taxonomy.py` | 102-feature taxonomy |
| `tests/README.md` | Test infrastructure documentation |

---

## Links

- **PHASE_HISTORY:** `docs/PHASE_HISTORY.md` (v4.0, Feb 14, 2026)
- **CAPABILITY_MATRIX:** `docs/CAPABILITY_MATRIX.md` (102 features, Phase 13.8.SW)
- **Benchmark Guide:** `benchmarks/README_sliding_window_benchmark.md`
- **Test Infrastructure:** `tests/README.md`
- **Foundation:** arXiv:2403.19330 (RootInteractive)

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-14 | Initial version |
| 2.0 | 2026-02-15 | Added scope/target, V2/V3/class APIs, benchmark framework, evaluator roadmap, integration patterns |
| 2.1 | 2026-02-15 | **8-reviewer fixes:** P0 ADF API (`register_subframe`), V5 `fit_specs` signature, output column normalization, Windows OS disambiguation, `weights` no-op note, memory table, freeze boundary, error handling, selection contract, decision tree, version compatibility |
