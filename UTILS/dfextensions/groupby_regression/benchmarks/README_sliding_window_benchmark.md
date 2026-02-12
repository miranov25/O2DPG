# Sliding Window Regression — Parametric Benchmark

## Summary

V5 sliding window regression at TPC scale (54K bins, 1000 rows/bin, W=(1,2,1))
takes ~1.9s per sector on numpy arrays, or ~3.4 min serial for the full TPC
(36 sectors x 3 targets), ~20s with 10-way parallelism.

The true smoothing overhead (V5arr / noSW-arr, both numpy I/O) is ~1.05-1.14x
at TPC scale, where Loop1 accumulation dominates.

V5tot (full wrapper) now runs at ~1.1-1.3x V5arr after fixing the bin_map bug
(previously 3-20x due to unconditional _build_bin_index_map call).

## Architecture

```
groupby_regression_optimized.py   -- per-bin regression, DataFrame API (noSW-DF)
groupby_regression_sliding_window.py:
  +-- make_sliding_window_fit()              -- DataFrame wrapper (V5tot)
  |     Now skips _build_bin_index_map on V5 path (fixed)
  +-- make_sliding_window_fit_v5_arrays()    -- numpy API, hot path (V5arr)
  +-- _get_numba_v4_kernels()                -- shared Numba kernels
       +-- accumulate_bin_stats   (Loop 1)
       +-- solve_all_windows      (Loop 2)
```

## What It Measures

| Column   | What                                      | I/O format   |
|----------|-------------------------------------------|--------------|
| noSW-arr | Per-bin regression, Numba kernels, W=0    | numpy->numpy |
| noSW-DF  | Per-bin regression, full DataFrame path   | DF->DF       |
| V5flat   | DataFrame->numpy extraction               | DF->numpy    |
| V5arr    | Sliding window, numpy path                | numpy->dict  |
| V5tot    | Sliding window, full wrapper (fixed)      | DF->DF       |
| V1-SW    | Old recompute algorithm                   | DF->DF       |

Key ratio: SW/noSW = V5arr / noSW-arr (fair, same I/O).

## Cost Models

| Model | Formula | Fits |
|-------|---------|------|
| 1: noSW-arr | a*N_rows + b*N_bins + c | Per-bin solve has per-bin cost |
| 2: noSW-DF | a*N_rows + b*N_bins + c | DataFrame groupby overhead |
| 3: V5flat | a*N_rows + c | ravel_multi_index |
| 4: V5arr | a*N_rows + b*N_bins*N_nbr + d*N_bins + c | 4-param required |
| 5: Loop1 | a*N_rows + c | Accumulation kernel |
| 6: Loop2 | a*N_bins*N_nbr + b*N_bins + c | Solve + Cholesky per-bin |
| 7: NbrTable | a*N_bins*N_nbr + b*N_bins + c | Neighbor construction |
| 8: V1-SW | a*N_bins*N_nbr*RPB + b*N_bins + c | Old recompute |
| 9: V5tot | a*N_rows + b*N_bins*N_nbr + d*N_bins + c | Wrapper (no bin_map) |

## Consistency Checks

1. V5arr = sum(setup + nbr_table + loop1 + loop2 + pack) within [0.9-1.2x]
2. Loop1 per row in [20, 80] ns
3a. (V5arr - NbrTable) < noSW-arr x 3 (compute kernels reasonable)
3b. NbrTable < 0.10 us/(bin x nbr) (scaling bounded)
4. V5tot = flatten + V5arr + assemble within [0.85-1.15x] (no bin_map)
5. DataFrame overhead ~ 40-120 us/bin

## Bug Fix: _build_bin_index_map

Before: Called unconditionally, consumed 85% of V5tot (70 min at TPC Standard).
After: Only called for V3/V1 fallback paths. V5 path computes lightweight
bounds via min()/max() per column.

Impact: V5tot dropped from ~70 min to ~3.4 min at TPC Standard (36x3).

## How to Run

```bash
python bench_parametric.py --quick 2>&1 | tee bench_parametric_v5_fixed.log
python bench_parametric.py 2>&1 | tee bench_parametric_v5_full.log
```
