# Capability Matrix — groupby_regression

**Generated:** 2026-02-16 08:00 UTC
**Phase:** 13.9.GB — GroupByRegressionEvaluator + Unified Metadata
**Generator:** `scripts/generate_capability_matrix.py`

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 34 | 30.4% |
| ☑️ Smoke-only | 77 | 68.8% |
| 🧨 Broken | 0 | 0.0% |
| ⚠️ Partial | 0 | 0.0% |
| 📋 Planned | 1 | 0.9% |
| **Total** | **112** | |

## Test Layer Distribution (excluding verbose duplicates)

| Layer | Count |
|-------|------:|
| invariance | 88 |
| integration | 12 |
| performance | 6 |
| smoke | 211 |
| validation | 24 |
| **Total unique** | **341** |

## groupby_regression

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ☑️ | **GROUPBY.cast_dtype** — Output dtype casting | 1 | 0 |  |  |
| ✅ | **GROUPBY.custom_fitter** — Custom fitter support | 1 | 1 |  |  |
| ☑️ | **GROUPBY.diagnostics** — Diagnostic columns (robust) | 1 | 0 |  |  |
| ✅ | **GROUPBY.exact_recovery** — Exact coefficient recovery (robust) | 2 | 1 |  |  |
| ☑️ | **GROUPBY.linear_fitter** — LinearRegression fitter | 1 | 0 |  |  |
| ☑️ | **GROUPBY.min_stat** — Minimum statistics per predictor | 2 | 0 |  |  |
| ☑️ | **GROUPBY.missing_values** — Missing value handling | 1 | 0 |  |  |
| ☑️ | **GROUPBY.ols_basic** — OLS basic fit (make_linear_fit) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.outlier_resilience** — Robust outlier resilience | 1 | 0 |  | STATSMODELS |
| ☑️ | **GROUPBY.prediction** — Prediction output | 1 | 0 |  |  |
| ☑️ | **GROUPBY.robust_basic** — Robust fit (make_parallel_fit) | 2 | 0 | ✓ | STATSMODELS |
| ☑️ | **GROUPBY.sigma_cut** — Sigma cut outlier rejection | 1 | 0 |  |  |
| ☑️ | **XVAL.robust_v2** — Robust vs V2 structural agreement | 1 | 0 |  |  |
| ✅ | **XVAL.robust_v4_parity** — Robust vs V4 numerical parity | 2 | 2 | ✓ |  |

## groupby_regression_evaluator

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ✅ | **EVAL.boundary** — Boundary handling (clamp/nan/extrapolate) | 7 | 7 |  |  |
| ☑️ | **EVAL.construction** — Evaluator construction from dfGB | 7 | 0 |  |  |
| ✅ | **EVAL.export_roundtrip** — JSON export/import roundtrip | 6 | 6 |  |  |
| ✅ | **EVAL.ivar_weighting** — Inverse-variance weighted interpolation | 7 | 7 |  |  |
| ☑️ | **EVAL.metadata_construction** — Evaluator construction from metadata | 7 | 0 |  |  |
| ✅ | **EVAL.multi_target** — Multi-target evaluation | 3 | 3 |  |  |
| ✅ | **EVAL.multilinear** — Multilinear interpolation | 8 | 8 |  |  |
| ✅ | **EVAL.nearest** — Nearest-neighbor evaluation | 7 | 6 |  |  |
| ✅ | **EVAL.sparse_grid** — Sparse grid handling (invalid bins) | 6 | 6 |  |  |
| ✅ | **EVAL.sw_integration** — SW output integration | 2 | 2 |  |  |

## groupby_regression_kernels

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ☑️ | **KERNEL.dispatch** — Kernel dispatch logic | 3 | 0 |  |  |
| ☑️ | **KERNEL.jit_signatures** — JIT compilation signatures | 2 | 0 |  | NUMBA |
| ✅ | **KERNEL.large_scale** — Large-scale correctness (50k+ groups) | 2 | 2 | ✓ | NUMBA |
| ✅ | **KERNEL.mc_validation** — Monte Carlo ground-truth validation | 4 | 4 | ✓ | NUMBA |
| ☑️ | **KERNEL.multi_fit** — Multi-target OLS kernel | 1 | 0 | ✓ | NUMBA |
| ☑️ | **KERNEL.nan_handling** — NaN detection and filtering in kernels | 2 | 0 |  | NUMBA |
| ✅ | **KERNEL.numba_numpy_parity** — Numba vs NumPy backend parity | 2 | 2 | ✓ | NUMBA |
| ☑️ | **KERNEL.numba_speedup** — Numba performance ratios | 2 | 0 |  | NUMBA |
| ✅ | **KERNEL.single_fit** — Single-target OLS kernel | 3 | 1 | ✓ | NUMBA |
| ✅ | **KERNEL.single_multi_parity** — Single-fit vs multi-fit kernel parity | 1 | 1 | ✓ | NUMBA |
| ☑️ | **KERNEL.status_codes** — Status code encoding/decoding | 4 | 0 |  |  |
| ☑️ | **KERNEL.streaming_memory** — Streaming RSS stability | 2 | 0 | ✓ | NUMBA |

## groupby_regression_optimized

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ☑️ | **GROUPBY.batch_strategy** — Batch strategy selection | 2 | 0 |  |  |
| ✅ | **GROUPBY.fast_backend** — Fast backend consistency | 1 | 1 |  | NUMBA |
| ✅ | **GROUPBY.multi_col_parity** — Multi-column groupby V4/V2 parity | 2 | 2 |  | NUMBA |
| ✅ | **GROUPBY.numba_backend** — Numba backend consistency | 1 | 1 |  | NUMBA |
| ☑️ | **GROUPBY.opt_diagnostics** — Diagnostics (optimized) | 1 | 0 |  | NUMBA |
| ☑️ | **GROUPBY.opt_edge_cases** — Edge cases (optimized) | 3 | 0 |  |  |
| ☑️ | **GROUPBY.opt_missing** — Missing values (optimized) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.opt_multi_target** — Multiple targets (optimized) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.opt_outlier** — Outlier resilience (optimized) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.opt_precision** — Statistical precision (optimized) | 2 | 0 |  |  |
| ☑️ | **GROUPBY.opt_prediction** — Prediction accuracy (optimized) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.opt_recovery** — Coefficient recovery (optimized) | 1 | 0 |  |  |
| ☑️ | **GROUPBY.parallel_speedup** — Parallel speedup (optimized) | 1 | 0 |  |  |
| ✅ | **GROUPBY.threading_backend** — Threading backend consistency | 2 | 1 |  |  |
| ☑️ | **GROUPBY.v2_basic** — V2 optimized implementation | 3 | 0 |  |  |
| ☑️ | **GROUPBY.v2_v3_v4_groups** — V2/V3/V4 structural group consistency | 1 | 0 |  |  |
| ☑️ | **GROUPBY.v3_basic** — V3 optimized implementation | 8 | 0 |  |  |
| ✅ | **GROUPBY.v3_v4_parity** — V3/V4 numerical parity | 1 | 1 |  |  |
| ☑️ | **GROUPBY.v4_basic** — V4 optimized implementation | 8 | 0 |  | NUMBA |
| ☑️ | **META.column_selection** — V4 column selection optimization | 7 | 0 |  |  |
| ☑️ | **META.column_validation** — Column name validation | 13 | 0 |  |  |
| ☑️ | **META.consistency** — Metadata consistency | 3 | 0 |  |  |
| ☑️ | **META.edge_cases** — Metadata edge cases | 4 | 0 |  |  |
| ☑️ | **META.formula_eval** — Formula evaluation | 6 | 0 |  |  |
| ☑️ | **META.prediction_formula** — Prediction formula generation | 9 | 0 |  |  |
| ☑️ | **META.pull_formula** — Pull formula generation | 3 | 0 |  |  |
| ☑️ | **META.residual_formula** — Residual formula generation | 3 | 0 |  |  |
| ☑️ | **META.schema** — Metadata schema | 17 | 0 |  |  |
| ☑️ | **PYARROW.backend_selection** — PyArrow backend selection | 5 | 0 |  | PYARROW |
| ☑️ | **PYARROW.edge_cases** — PyArrow edge cases | 5 | 0 |  | PYARROW |
| ☑️ | **PYARROW.memory** — PyArrow memory behavior | 3 | 0 |  | PYARROW |
| ☑️ | **PYARROW.metadata** — PyArrow metadata export | 2 | 0 |  | PYARROW |
| ✅ | **PYARROW.parity** — Pandas/PyArrow numerical parity | 5 | 5 |  | PYARROW |
| ✅ | **PYARROW.sort_stability** — PyArrow sort stability | 4 | 3 |  | PYARROW |
| ☑️ | **STD.column_consistency** — V2/V3/V4 column consistency | 2 | 0 |  |  |
| ☑️ | **STD.merge_fits** — Multiple fits merge with suffix | 1 | 0 |  |  |
| ☑️ | **STD.rms_mad** — RMS/MAD always present | 1 | 0 |  |  |
| ☑️ | **STD.suffix** — Suffix application | 1 | 0 |  |  |
| ☑️ | **V5.backend_selection** — V5 backend auto-selection | 6 | 0 |  | NUMBA |
| ☑️ | **V5.basic** — V5 batch API basic | 6 | 0 |  |  |
| ✅ | **V5.chunk_invariance** — V5 chunk-size independence | 3 | 2 |  |  |
| ☑️ | **V5.diagnostics** — V5 diagnostic output | 3 | 0 |  |  |
| ✅ | **V5.edge_cases** — V5 edge cases | 5 | 1 |  |  |
| ☑️ | **V5.metadata** — V5 metadata generation | 3 | 0 |  |  |
| ✅ | **V5.numba_parity** — V5 Numba matches sequential | 4 | 4 |  | NUMBA |
| ☑️ | **V5.parallel_edge** — V5 parallel edge cases | 4 | 0 |  | NUMBA |
| ☑️ | **V5.per_fit_diag** — V5 per-fit diagnostics | 2 | 0 |  |  |
| ☑️ | **V5.performance** — V5 performance vs V4 | 1 | 0 | ✓ |  |
| ☑️ | **V5.sort_helpers** — V5 internal sort helpers | 3 | 0 |  |  |
| ☑️ | **V5.status_codes** — V5 status codes | 3 | 0 |  |  |
| ☑️ | **V5.threading_env** — V5 threading environment | 1 | 0 |  |  |
| ✅ | **V5.v4_parity** — V5/V4 numerical parity | 3 | 2 |  |  |
| ☑️ | **V5.validation** — V5 input validation | 6 | 0 |  |  |

## groupby_regression_sliding_window

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ☑️ | **SW.aggregation** — Sliding window aggregation | 1 | 0 |  |  |
| ✅ | **SW.backend_auto** — Backend auto-dispatch | 2 | 1 |  |  |
| ☑️ | **SW.basic** — Sliding window basic 3D | 1 | 0 |  |  |
| ☑️ | **SW.bin_helpers** — Internal bin helpers | 2 | 0 |  |  |
| ☑️ | **SW.boundary** — Boundary truncation | 3 | 0 |  |  |
| ☑️ | **SW.empty_window** — Empty window handling | 1 | 0 |  |  |
| ✅ | **SW.invariance** — Sliding window invariance tests | 8 | 8 |  |  |
| ✅ | **SW.linear_fit** — Sliding window linear fit | 1 | 1 |  |  |
| ☑️ | **SW.metadata** — Sliding window metadata | 1 | 0 |  |  |
| ☑️ | **SW.min_stat** — Minimum entries enforcement | 1 | 0 |  |  |
| ✅ | **SW.multi_predictor** — Sliding window multi-predictor | 2 | 1 |  |  |
| ☑️ | **SW.multi_target** — Sliding window multi-target | 1 | 0 |  |  |
| ☑️ | **SW.omitted_dims** — Omitted window dims default to 0 | 1 | 0 |  |  |
| ✅ | **SW.parallel** — Parallel sliding window (split-column) | 8 | 3 | ✓ | NUMBA |
| ☑️ | **SW.return_metadata** — Sliding window return_metadata | 1 | 0 |  |  |
| ☑️ | **SW.selection** — Sliding window selection mask | 1 | 0 |  |  |
| ☑️ | **SW.smoke_gate** — Realistic smoke normalised residuals | 1 | 0 |  |  |
| ☑️ | **SW.suffix** — Sliding window output suffix | 1 | 0 |  |  |
| ✅ | **SW.v4_parity** — SW window-zero parity with V4 | 2 | 2 |  |  |
| ✅ | **SW.v5_dominance** — V5 algorithm dominance across all backends | 2 | 2 | ✓ | NUMBA |
| ☑️ | **SW.validation** — Sliding window input validation | 5 | 0 |  |  |
| 📋 | **SW.weighted** — Sliding window weighted fits (WLS) | 0 | 0 |  | PLANNED |

## synthetic_tpc_distortion

| Status | Feature | Tests | Inv/Int | Bench | Tag |
|--------|---------|------:|--------:|-------|-----|
| ☑️ | **TPC.distortion_recovery** — TPC distortion recovery pipeline | 1 | 0 |  |  |

## Benchmark Proof Catalog

| Feature | Benchmark Check | Gate? |
|---------|-----------------|-------|
| GROUPBY.robust_basic | bench_comparison.py::compute_agreement() [MONITOR] | 📊 MONITOR |
| KERNEL.large_scale | bench_groupby_regression_kernels.py::validate_correctness() [GATED] | ✅ GATED |
| KERNEL.mc_validation | bench_groupby_regression_kernels.py::validate_correctness() [GATED] | ✅ GATED |
| KERNEL.multi_fit | bench_groupby_regression_kernels.py::validate_correctness() [GATED] | ✅ GATED |
| KERNEL.numba_numpy_parity | bench_groupby_regression_kernels.py::Numba/NumPy parity [GATED] | ✅ GATED |
| KERNEL.single_fit | bench_groupby_regression_kernels.py::validate_correctness() [GATED] | ✅ GATED |
| KERNEL.single_multi_parity | bench_groupby_regression_kernels.py::Numba/NumPy parity [GATED] | ✅ GATED |
| KERNEL.streaming_memory | bench_groupby_regression_memory.py::RSS drift < 5% [GATED] | ✅ GATED |
| KERNEL.streaming_memory | bench_groupby_regression_memory.py::RSS CV < 0.1 [GATED] | ✅ GATED |
| SW.parallel | bench_slidingwindow_parallel.py::parallel_scaling (1-36 workers) [GATED] | ✅ GATED |
| SW.parallel | bench_slidingwindow_parallel.py::serial_vs_parallel_validation [GATED] | ✅ GATED |
| SW.parallel | bench_slidingwindow_parallel.py::counting_sort_vs_argsort [MONITOR] | 📊 MONITOR |
| SW.v5_dominance | bench_slidingwindow_parametric.py::V1_vs_V2_vs_V3_vs_V5_validation (30/30 PASS) [GATED] | ✅ GATED |
| SW.v5_dominance | bench_slidingwindow_parametric.py::algorithm_recommendation (V5 wins 10/10) [MONITOR] | 📊 MONITOR |
| SW.v5_dominance | bench_slidingwindow_parametric.py::cost_model_fit (11 models, R2>0.99) [MONITOR] | 📊 MONITOR |
| V5.performance | bench_v5.py::timing [MONITOR] | 📊 MONITOR |
| XVAL.robust_v4_parity | bench_comparison.py::compute_agreement() [MONITOR] | 📊 MONITOR |

---

*Two-tier verification per Phase 13.9.GB v02 proposal.*
*✅ = invariance/integration test exists. ☑️ = smoke tests only — does not catch numerical regressions.*
*Verbose SW duplicates (1 files) deduplicated per §3.6.*