# Roofline Performance Regression Tests — Algorithm Catalog

**Phase:** 13.22.GB-RooflineTier1
**Proposal:** `PHASE_13_22_GB_RooflineTier1_v1.10_Proposal.md`
**Methodology:** AI_Review_Scientific_Methodology v1.0-FINAL § Hardware-Limit Performance
**Primitives:** phase_12_12b_specification_v2.md Appendix A, lines 933–941

## Overview

Tier 1 CI roofline tests measure `K = T_observed / T_expected` where
`T_expected = Σ(n_ops × t_primitive)` is the sum of hardware-primitive costs
modeling each function's operation sequence. K=1 means at-roofline (ideal
compiled code). K=3 means 3× above ideal.

Run: `NUMBA_THREADING_LAYER=omp pytest tests/test_performance_roofline.py -v -s -m roofline`

## Test Inventory (10 pytest items)

### 5 K-Roofline Tests

| Test | Function | Fixture | Instrument | K (alma2) |
|------|----------|---------|------------|-----------|
| `test_fit_regression_roofline` | `make_sliding_window_fit` | SW2D | perf_counter | ~2.6–12 |
| `test_v4_modeled_roofline` | `make_parallel_fit_v4` | S2 | perf_counter | ~2.3–3.3 |
| `test_assign_bin_ids_modeled_roofline` | `_assign_bin_ids_fast` | SW2D | cProfile | ~2.3 |
| `test_counting_sort_modeled_roofline` | `_counting_sort_indices` | SW2D | cProfile | ~8–31 |
| `test_fit_kernel_modeled_roofline` | `fit_groups_single_numba` | SW2D | perf_counter | ~3.4–14 |

### 1 Dispatch Count Test

| Test | Function | Metric |
|------|----------|--------|
| `test_v4_median_dispatch_count` | `make_parallel_fit_v4` | np.median ncalls ≤ 15000 |

### 4 Meta-Tests

| Test | Purpose |
|------|---------|
| `test_baseline_file_loaded` | Baseline JSON schema valid |
| `test_baseline_self_consistent` | K_threshold = K_floor + 6×1.4826×MAD |
| `test_baseline_machine_matches_runtime` | Runtime machine in baseline |
| `test_baseline_phase_current` | Warns if baseline >90 days old |

## Fixtures

**Fixture A (S2):** 100K rows, 1000 groups, flat 1D `group` column.
Used by V4 tests.

**Fixture B (SW2D):** 100K rows, 32×32=1024 bins, 2D grid `bin_x`/`bin_y`,
`window_spec={"bin_x":2, "bin_y":2}`, `n_linear=2`, `n_features=3`.
Used by SW-fit tests. Effective neighbors per bin ≈ 23.2.

## Primitives (alma2, L3-cache regime at S2 fixture size)

| Primitive | Value | Unit | Source |
|-----------|------:|------|--------|
| M1 stream_read | 41.0 | GB/s | phase_12_12b A:934 |
| M2 gather | 15.5 | GB/s | phase_12_12b A:935 |
| M3w scatter_write | 11.3 | GB/s | phase_12_12b A:937 |
| C1 XᵀWX (V4, 100 rows) | 1216 | ns/call | phase_12_12b A:938 |
| C1 XᵀWX (SW, 2200 rows) | 9937 | ns/call | phase_12_12b A:938 |
| C2 solve | 3103 | ns/call | phase_12_12b A:939 |
| C6 MAD | 23637 | ns/call | phase_12_12b A:940 |

Note: M2=15.5 GB/s reflects L3-cache bandwidth (S2 fixture fits in L3).
Production S5 (~270 MB) would measure ~2.3 GB/s (DRAM). K is valid within
the S2 cache regime; cross-regime comparison requires Phase 13.23.

## T_expected Formulas

### test_fit_regression_roofline (SW fit full pipeline)

```
T_expected =
    4 × n_rows × 8 / (M1 × 1e9)              # input stream
  + 8 × n_rows × 8 / (M1 × 1e9)              # bin-id reads (~8 passes)
  + 5 × n_rows × 8 / (M3w × 1e9)             # bin-id writes (~5 passes)
  + 2 × n_rows × 8 / (M1 × 1e9)              # counting-sort reads
  + 2 × n_rows × 8 / (M3w × 1e9)             # counting-sort writes
  + n_groups × 8 / (M3w × 1e9)               # bincount
  + n_groups × 16 / (M1 × 1e9)               # cumsum
  + n_groups × n_nbr_eff × rpg × 8 / (M2×1e9) # gather
  + n_groups × (C1_sw + C2) × 1e-9            # OLS (window-sized)
  + n_rows × 8 / (M3w × 1e9)                 # output writes
  + n_groups × 14 × 8 / (M3w × 1e9)          # output assembly (M3w)
```

### test_v4_modeled_roofline (V4 pipeline)

```
T_expected =
    4 × n_rows × 8 / (M1 × 1e9)
  + n_groups × (C1_v4 + C2 + C6) × 1e-9
  + n_rows × 8 / (M3w × 1e9)
  + n_rows × 4 × 8 / (M2 × 1e9)              # df_sorted gather
  + n_rows × 4 × 8 / (M3w × 1e9)             # df_sorted write
  + n_groups × 15 × 8 / (M3w × 1e9)          # dfGB write
```

## Threshold Derivation

```
K_floor = max(K_median across machines)
K_threshold[machine] = K_floor + 6 × 1.4826 × MAD(K)
```

Multiplier 6 per architect directive. Conservative — absorbs farm-load variance.

## Failure Interpretation Guide

A failed K-roofline test can mean:

1. **Code regression.** Function got slower — investigate recent commits.
2. **Machine load anomaly.** Re-run on a quiet machine.
3. **Stale baseline.** Regenerate: `python benchmarks/scripts/calibrate_roofline_K.py`
   then `python benchmarks/scripts/update_roofline_baseline.py`.
4. **Operation model stale.** Function's algorithm changed — update §4.5 formulas.
5. **Cross-machine mismatch.** Check `test_baseline_machine_matches_runtime`.

## Pipeline

```
measure_primitives.py  → bench_out/primitives_<machine>.json
calibrate_roofline_K.py → bench_out/roofline_K_<machine>.json
update_roofline_baseline.py → benchmarks/baselines/roofline_baseline.json
pytest -m roofline      → pass/fail against baseline thresholds
```

All files generated, never hand-written. Baseline updates require `[BASELINE-UPDATE]` commit tag.
