# Sliding Window Regression — Benchmark & Performance Guide

## Quick Answer: Which Algorithm Should I Use?

**Use V5 (incremental+numba) for everything.** It is 25–68× faster than all alternatives across every tested configuration. There is no crossover point where V1, V2, or V3 wins.

```python
result = make_sliding_window_fit(
    df, gb_columns=['rBin','y2xBin','z2xBin'],
    fit_columns=['dX','dY','dZ'], linear_columns=['x'],
    window_spec={'rBin': 1, 'y2xBin': 2, 'z2xBin': 1},
    algorithm='incremental', backend='numba',   # ← V5, always
    min_stat=5, suffix='_sw',
)
```

For multi-sector TPC data, add parallelism:

```python
result = make_sliding_window_fit_parallel(
    df, gb_columns=['rBin','y2xBin','z2xBin'],
    fit_columns=['dX','dY','dZ'], linear_columns=['x'],
    split_columns=['sector'],    # parallelize over sectors
    n_workers=8,                 # use 8-16 for best throughput
    window_spec={'rBin': 1, 'y2xBin': 2, 'z2xBin': 1},
    min_stat=5, suffix='_sw',
)
```

## Estimating CPU Time for Your Configuration

Use these formulas to predict execution time (fitted from benchmark data on Apple M1 Pro; scale by your machine's single-core performance):

### V5 Array Path (hot path, numpy I/O)

```
T_V5arr = 0.030 × N_rows  +  0.028 × N_bins × N_nbr  +  1.52 × N_bins  +  2.8 ms
           (µs/row)            (µs/bin/nbr)                (µs/bin)          (fixed)
```

Where: `N_rows = N_bins × RPB`, `N_nbr = (2W₁+1)(2W₂+1)(2W₃+1)`

### V5 Total (full DataFrame wrapper)

```
T_V5tot = 0.060 × N_rows  +  0.028 × N_bins × N_nbr  +  1.45 × N_bins  +  5.5 ms
           (µs/row)            (µs/bin/nbr)                (µs/bin)          (fixed)
```

The ~2× higher per-row cost vs V5arr comes from DataFrame→numpy flattening.

### V1 Recompute (reference, for comparison)

```
T_V1 = 0.096 × N_bins × N_nbr × RPB  +  97.1 × N_bins  -  35 ms
        (µs/bin/nbr/row)                  (µs/bin)          (fixed)
```

### Example Predictions

| Scenario | N_bins | RPB | N_nbr | V5arr | V5tot | V1 |
|----------|--------|-----|-------|-------|-------|----|
| TPC Low | 12,000 | 500 | 27 | 0.21s | 0.38s | ~60s |
| TPC Standard | 54,000 | 1,000 | 45 | 1.79s | 3.41s | ~430s |
| TPC High | 54,000 | 2,000 | 125 | 3.55s | 6.75s | ~2,100s |

Full TPC (36 sectors × 3 targets) with 10-way parallelism: **Standard ≈ 20s, High ≈ 40s**.

### Cost Model Accuracy

The models fit well at scale (R² > 0.99, residuals < 8% for N_bins ≥ 3375) but overpredict at small grids (10³) where fixed overhead dominates. For production TPC configurations, predictions are accurate to ±10%.

## Algorithm Comparison

All four backends produce bit-identical results (max coefficient difference < 2×10⁻¹⁴):

| Algorithm | Code | Complexity | When |
|-----------|------|------------|------|
| V1 (recompute+numpy) | `algorithm='recompute', backend='numpy'` | O(N_bins × N_nbr × RPB) | Baseline, no numba needed |
| V2 (recompute+numba) | `algorithm='recompute', backend='numba'` | O(N_bins × N_nbr × RPB) | ~15% faster than V1 |
| V3 (incremental+numpy) | `algorithm='incremental', backend='numpy'` | O(N_rows + N_bins × N_nbr) | No numba, better scaling |
| **V5 (incremental+numba)** | `algorithm='incremental', backend='numba'` | O(N_rows + N_bins × N_nbr) | **Always fastest** |

Measured speedups on Apple M1 Pro (10 configurations, --quick):

| Config | V1 | V2 | V3 | V5tot | V5 Speedup |
|--------|------|------|------|-------|------------|
| 10³ W=1 r=10 | 0.114s | 0.093s | 0.127s | 0.005s | 25× |
| 25³ W=1 r=10 | 1.867s | 1.609s | 2.179s | 0.050s | 44× |
| 25³ W=2 r=10 | 3.515s | 3.094s | 6.072s | 0.089s | 68× |
| 25³ W=1 r=50 | 3.377s | 2.832s | 2.702s | 0.086s | 40× |

V5 wins every configuration. The speedup *increases* with grid size and window size because V5's O(N_rows) accumulation amortizes better at scale.

## Parallel Scaling (Strategy A)

Parallel performance on Linux (OrbStack aarch64), 112M rows, 36 sectors:

| Workers | Total | Speedup | Bottleneck |
|---------|-------|---------|------------|
| 1 | 16.6s | 1.0× | — |
| 4 | 6.7s | 2.5× | workers (5.6s) |
| 8 | 5.2s | 3.2× | workers (4.0s) |
| 16 | **5.1s** | **3.3×** | workers (3.8s) + extract (0.5s) |
| 36 | 6.2s | 2.7× | fork overhead exceeds compute savings |

Key observations:
- Sweet spot is 8–16 workers for 36 sectors.
- Serial overhead (extract + sort + concat) ≈ 1.1s is irreducible, capping theoretical max at ~15×.
- Counting sort: 44.8× faster than mergesort for grouping 112M rows by sector.
- Serial/parallel parity: **exact** (max|diff| = 0.00e+00 across all worker counts).

## Architecture

```
groupby_regression_sliding_window.py:
  make_sliding_window_fit()              — DataFrame API (V5tot)
  │  └── _flatten_bins_for_v5()          — DF→numpy extraction
  │  └── make_sliding_window_fit_v5_arrays()  — numpy hot path (V5arr)
  │       ├── accumulate_bin_stats()      — Loop 1: O(N_rows) XtX/XtY
  │       └── solve_all_windows()         — Loop 2: O(N_bins × N_nbr) Cholesky
  │  └── _assemble_results_v5()          — dict→DataFrame
  │
  make_sliding_window_fit_parallel()     — multi-sector parallelism
       ├── _counting_sort_indices()       — O(N) sector grouping
       ├── ProcessPoolExecutor (fork COW) — zero-pickle dispatch
       └── _worker_v5_shared()            — per-sector V5 pipeline
```

### V5arr Internal Breakdown

At 25³ grid, W=1, 50 rows/bin (781K rows):

| Component | Time | % of V5arr | Scaling |
|-----------|------|------------|---------|
| NbrTable | 32.0ms | 52% | O(N_bins × N_nbr) |
| Loop1 (accumulate) | 23.2ms | 38% | O(N_rows) |
| Loop2 (solve) | 5.6ms | 9% | O(N_bins × N_nbr) |
| Setup + Pack | 0.2ms | <1% | O(N_bins) |
| **Total** | **61.6ms** | 100% | |

At TPC scale (RPB ≥ 500), Loop1 dominates because it scales with N_rows while NbrTable/Loop2 scale with N_bins.

## Smoothing Overhead

The "fair" comparison — V5arr vs noSW-arr (same Numba kernels, same I/O format, only difference is W > 0):

| RPB | W=1 overhead | W=2 overhead |
|-----|-------------|-------------|
| 10 | 5.5–7.0× | 10.8–12.8× |
| 50 | 2.5–2.7× | — |
| 500+ (TPC) | ~1.1× | ~1.2× |

At TPC scale, sliding window smoothing adds only 10–20% overhead over plain per-bin regression. The neighbor table construction dominates at low RPB but becomes negligible at high RPB.

## Benchmarks

### bench_slidingwindow_parametric.py

Serial benchmark: all backends × grid sizes × window sizes × rows per bin.

```bash
# Quick (10 configs, ~2 min)
python benchmarks/bench_slidingwindow_parametric.py --quick

# Full scan (more configs, ~15 min)
python benchmarks/bench_slidingwindow_parametric.py

# With cProfile profiling
python benchmarks/bench_slidingwindow_parametric.py --quick --cprofile
```

Outputs: algorithm comparison table, 11 linear cost model fits, 6 consistency checks, TPC predictions, V1-vs-V5 validation, component breakdown. With `--cprofile`: per-config `.prof` + `.csv` files in `benchmark_results/profiles/profiles_parametric.zip`.

### bench_slidingwindow_parallel.py

Parallel benchmark: fork overhead, pickle cost, scaling curve, sort comparison.

```bash
# Full benchmark (requires ~16GB RAM for 112M rows)
python benchmarks/bench_slidingwindow_parallel.py

# With cProfile
python benchmarks/bench_slidingwindow_parallel.py --cprofile
```

Outputs: serial per-sector breakdown, fork dispatch overhead, pickle cost measurement, parallel scaling (1–36 workers), counting sort vs argsort, serial-vs-parallel validation.

### Interpreting Results on Your Machine

Run `bench_slidingwindow_parametric.py --quick` and look at the **ALGORITHM RECOMMENDATION** table. If V5tot numbers on your machine are proportionally faster/slower than the M1 Pro numbers above, scale the TPC predictions accordingly:

```
your_TPC_time ≈ (your_V5tot_25³_W1_r10 / 0.050s) × reference_TPC_time
```

## Cost Models Reference

11 linear models are fitted by the parametric benchmark:

| # | Target | Formula | R² | Notes |
|---|--------|---------|-----|-------|
| 1 | noSW-arr | a·N_rows + b·N_bins + c | 0.9999 | Baseline per-bin regression |
| 2 | noSW-DF | a·N_rows + b·N_bins + c | 0.9987 | DataFrame overhead: ~70 µs/bin |
| 3 | V5flat | a·N_rows + c | 0.983 | ravel_multi_index extraction |
| 4 | V5arr | a·N_rows + b·N_bins·N_nbr + d·N_bins + c | 0.995 | **Main prediction model** |
| 5 | Loop1 | a·N_rows + c | 0.999 | ~30 ns/row (accumulation kernel) |
| 6 | Loop2 | a·N_bins·N_nbr + b·N_bins + c | 0.9996 | Cholesky solve per window |
| 7 | NbrTable | a·N_bins·N_nbr + b·N_bins + c | 0.993 | Neighbor index construction |
| 8 | V1-SW | a·N_bins·N_nbr·RPB + b·N_bins + c | 0.995 | Recompute (for comparison) |
| 9 | V5tot | a·N_rows + b·N_bins·N_nbr + d·N_bins + c | 0.992 | Full wrapper with DataFrame I/O |
| 10 | V2-SW | a·N_bins·N_nbr·RPB + b·N_bins + c | 0.990 | Recompute+numba |
| 11 | V3-numpy | a·N_rows + b·N_bins·N_nbr + d·N_bins + c | 0.9997 | Incremental+numpy |

Models 3, 4, 7 show >20% residuals at the smallest grid (10³) where fixed costs dominate. This is a statistical fitting artifact with few data points in `--quick` mode, not a code issue. At production scale (grid ≥ 20³), all models are accurate to ±10%.

## Consistency Checks

The parametric benchmark runs 6 automated consistency checks:

| # | Check | Gate | Status |
|---|-------|------|--------|
| 1 | V5arr = Σ(components) | ratio in [0.9, 1.2] | ✅ PASS |
| 2 | Loop1 per-row cost | 20–80 ns/row | ✅ PASS |
| 3a | V5arr compute ≤ 3× noSW-arr | ratio < 3.0 | ✅ PASS |
| 3b | NbrTable cost bounded | < 0.10 µs/(bin·nbr) | ✅ PASS |
| 4 | V5tot = flatten + V5arr + assemble | ratio in [0.85, 1.15] | ⚠️ FAIL at 10³ |
| 5 | DataFrame overhead | 40–120 µs/bin | ✅ PASS |

Check 4 fails only at 10³ grid (ratio 1.23×) where unmeasured fixed costs dominate. All checks pass at grid ≥ 15³.

## Bug Fixes Applied

### _build_bin_index_map (Phase 13.8)

Before: Called unconditionally on every V5 path call, consuming 85% of V5tot (70 min at TPC Standard). After: Only called for V3/V1 fallback paths. Impact: V5tot dropped from ~70 min to ~3.4 min.

### Parallel pickle overhead (Phase 13.10)

Before: `ProcessPoolExecutor.submit()` pickled the full 4.3GB array (125MB × 36 tasks = 155GB serialization). After: Module-level shared state with fork() COW, only (start, end) integers pickled (~200 bytes/task). Impact: parallel overhead dropped from 125s to <1s.

## Validation

All benchmarks include always-on validation:
- **Parametric**: V1 vs V2 vs V3 vs V5 coefficient comparison (30/30 PASS, max diff < 2×10⁻¹⁴)
- **Parallel**: Serial vs parallel parity at all worker counts (max|diff| = 0.00e+00)
- **Test suite**: 55 invariance tests + 8 parallel tests, all passing

---

## Review Questions for Reviewers

**Q1: Cost model usability.** Are the prediction formulas clear enough for users? Should we add a helper function in the code itself, e.g. `estimate_sliding_window_time(n_bins, rpb, n_nbr, ...)`?

**Q2: Algorithm recommendation.** "Use V5 for everything" — too strong? Edge cases to caveat (no numba, tiny data)?

**Q3: Model fit quality.** Models 3, 4, 7 show >20% residuals at 10³ grid. Sufficient to note as small-grid artifact, or should we widen tolerances / exclude small grids?

**Q4: Parallel scaling plateau.** 3.3× max at 16 workers. Should we add explicit Amdahl's law analysis?

**Q5: Machine portability.** Formulas fitted on M1 Pro. One-line scaling rule adequate, or provide Linux x86 coefficients too?

**Q6: Missing content.** Numba warmup time? Memory requirements? Python 3.9.6 notes? `--cprofile` interpretation guide?

**Q7: Structure and length.** 247 lines — right length, or split into user guide + technical appendix?